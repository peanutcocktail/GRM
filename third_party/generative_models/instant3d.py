from copy import deepcopy
from pytorch_lightning import seed_everything
import random
import string
import math
import os
import sys
from typing import List, Union, Optional

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import autocast
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid

from sgm.util import append_dims, instantiate_from_config



def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    # Hardcoded demo settings; might undergo some changes in the future

    value_dict = {}
    for key in keys:
        if key == "txt":
            assert prompt is not None
            assert negative_prompt is not None

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            value_dict["orig_width"] = init_dict["orig_width"]
            value_dict["orig_height"] = init_dict["orig_height"]

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

    return value_dict



def init_sampling(
    sampler_config,
    steps=100,
    guidance_scale=5.0,
):
    sampler_config = deepcopy(sampler_config)
    sampler_config.params.num_steps = steps
    sampler_config.params.guider_config.params.scale = guidance_scale
    sampler = instantiate_from_config(sampler_config)
    return sampler



def add_noise(x, sigmas, offset_noise_level=0.0):
    noise = torch.randn_like(x)
    if offset_noise_level > 0.0:
        noise = noise + offset_noise_level * append_dims(
            torch.randn(input.shape[0], device=input.device), input.ndim
        )
    noised_x = x + noise * append_dims(sigmas, input.ndim)
    return noised_x



def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def generate_gaussian_image(size=1024, sigma=0.1, mode='2x2'):
    # generate a pytorch image with white background and a gray gaussian blob at the center
    # size: int, size of the image
    # sigma: float, standard deviation of the gaussian blob
    # return: torch.tensor, shape (1, 3, size, size)
    x = torch.arange(size).float()
    y = torch.arange(size).float()
    x, y = torch.meshgrid(x, y)
    x = x[None, None, :, :]
    y = y[None, None, :, :]
    center = size // 2
    sigma = size * sigma
    if sigma > 0.:
        gaussian = torch.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.max()
        gaussian = gaussian.repeat(1, 3, 1, 1)
        gaussian = 1. - gaussian # range in [0, 1]
    else:
        gaussian = torch.ones(1, 3, size, size)
    if mode == '2x2':
        # stack as 2x2
        gaussian = torch.cat([gaussian, gaussian], dim=-1)
        gaussian = torch.cat([gaussian, gaussian], dim=-2)
        # downsample
        gaussian = F.interpolate(gaussian, (size, size), mode='bilinear')
    elif mode == '1x1':
        pass
    else:
        assert False

    return gaussian


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    use_blobs=True,
    gaussian_sigma=0.1,
    force_uc_zero_embeddings: Optional[List] = None,
    batch2model_input: Optional[List] = None,
    return_latents=False,
    # filter=None,
):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []

    # st.text("Sampling")

    # outputs = st.empty()
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                num_samples = [num_samples]
                model.conditioner.cuda()
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                # unload_model(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")
                if use_blobs:
                    gaussian_image = generate_gaussian_image(1024, sigma=gaussian_sigma, mode='2x2') # [1, 3, 1024, 1024] in [0, 1]
                    gaussian_image = gaussian_image.to(randn.device).to(randn.dtype)
                    gaussian_image = gaussian_image * 2. - 1. # [-1, 1]
                    gaussian_latents = model.encode_first_stage(gaussian_image) # [1, 4, 128, 128]
                    _sigma_max = sampler.discretization(sampler.num_steps)[0]
                    assert randn.shape == gaussian_latents.shape
                    randn = randn * _sigma_max + gaussian_latents
                    randn /= torch.sqrt(1.0 + _sigma_max ** 2.0) # in the sampler this factor will be multiplied back at the beginning before feeding to denoiser


                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                model.denoiser.cuda()
                model.model.cuda()
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)

                model.first_stage_model.cuda()
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if return_latents:
                    return samples, samples_z
                return samples


def run_txt2img(
    model,
    sampler_config,
    prompt,
    negative_prompt,
    guidance_scale,
    num_steps=100,
    return_latents=False,
    gaussian_sigma=0.1,
):
    W = H = 1024
    C = 4
    F = 8

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    sampler = init_sampling(sampler_config, steps=num_steps, guidance_scale=guidance_scale)
    num_samples = 1

    out = do_sample(
        model=model,
        sampler=sampler,
        value_dict=value_dict,
        num_samples=num_samples,
        H=H,
        W=W,
        C=C,
        F=F,
        force_uc_zero_embeddings=["txt"],
        gaussian_sigma=gaussian_sigma,
        return_latents=return_latents,
        # filter=filter,
    )
    return out

def build_instant3d_model(config_path, ckpt_path):
    global config
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.cuda()
    return model


def instant3d_pipe(model, prompt, negative_prompt="", guidance_scale=5.0, num_steps=30, gaussian_sigma=0.1):
    out = run_txt2img(
        model=model,
        sampler_config=config.model.params.sampler_config,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        return_latents=False,
        num_steps=num_steps,
        gaussian_sigma=gaussian_sigma,
    )
    if isinstance(out, (tuple, list)):
        samples, samples_z = out
    else:
        samples = out
        samples_z = None

    assert samples.shape[0] == 1 and samples.shape[1] == 3
    sample = samples[0]
    return sample



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='temp')
    parser.add_argument('--num_prompts', type=int, default=0) # 0 means all the prompts
    parser.add_argument('--num_images_per_prompt', type=int, default=2) # 0 means all the prompts
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--gaussian_sigma', type=float, default=0.1) # relative to image size

    args = parser.parse_args()

    save_dir = args.save_dir # real save dir will be a subdir of this
    os.makedirs(save_dir, exist_ok=True)

    import time
    time_start = time.time()


    # build model
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.cuda()


    num_images_per_prompt = args.num_images_per_prompt

    prompts = [line.strip() for line in open(args.prompt_file, 'r')]
    # filter out prompts that are too long
    prompts = [prompt for prompt in prompts if len(prompt) < 70]
    print(f'Total number of prompts: {len(prompts)}')
    if args.num_prompts > 0:
        prompts = prompts[:args.num_prompts]
    print(f'Using {len(prompts)} prompts')
    negative_prompt = ""


    def seed_everything(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    seed_everything(1)


    for prompt_idx, prompt in enumerate(prompts):
        for image_idx in range(num_images_per_prompt):
            out = run_txt2img(
                model=model,
                sampler_config=config.model.params.sampler_config,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=args.guidance_scale,
                return_latents=False,
                num_steps=args.num_steps,
                gaussian_sigma=args.gaussian_sigma,
            )
            if isinstance(out, (tuple, list)):
                samples, samples_z = out
            else:
                samples = out
                samples_z = None

            assert samples.shape[0] == 1 and samples.shape[1] == 3
            sample_np = samples[0].cpu().numpy().transpose(1, 2, 0)
            sample_pil = Image.fromarray((sample_np * 255).astype(np.uint8))
            image = sample_pil
            assert isinstance(image, Image.Image)
            filename = f"PROMPTSTART_{'_'.join(prompt.split())}_{image_idx:03d}.png"
            image.save(os.path.join(save_dir, filename))


    print(f'Total time (minutes): {(time.time() - time_start) / 60:.2f} for {len(prompts)} prompts and {len(prompts) * num_images_per_prompt} images')
