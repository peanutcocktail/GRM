import os
import random
import rembg
import sys
from PIL import Image
import numpy as np
import torch
import imageio
import math
import cv2
import open3d as o3d
from tqdm import tqdm
from utils import saveload_utils
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_folder}/third_party/generative_models')

from third_party.generative_models.instant3d import build_instant3d_model, instant3d_pipe
from third_party.generative_models.scripts.sampling.simple_video_sample import sample as sv3d_pipe

### set gpu
torch.cuda.set_device(0)
device = torch.device(0)


def dump_video(image_sets, path, **kwargs):
    video_out = imageio.get_writer(path, mode='I', fps=30, codec='libx264')
    for image in image_sets:
        video_out.append_data(image)
    video_out.close()

def generate_cameras(r, num_cameras=20, device='cuda:0', pitch=math.pi/8, use_fibonacci=False):
    def normalize_vecs(vectors): return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

    t = torch.linspace(0, 1, num_cameras).reshape(-1, 1)

    pitch = torch.zeros_like(t) + pitch

    directions = 2*math.pi 
    yaw = math.pi
    yaw = directions*t + yaw

    if use_fibonacci:
        cam_pos = fibonacci_sampling_on_sphere(num_cameras)
        cam_pos = torch.from_numpy(cam_pos).float().to(device)
        cam_pos = cam_pos * r
    else:
        z = r*torch.sin(pitch)
        x = r*torch.cos(pitch)*torch.cos(yaw)
        y = r*torch.cos(pitch)*torch.sin(yaw)
        cam_pos = torch.stack([x, y, z], dim=-1).reshape(z.shape[0], -1).to(device)

    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 0, -1], dtype=torch.float,
                                        device=device).reshape(-1).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                                        dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                                        dim=-1))
    rotate = torch.stack(
                    (left_vector, up_vector, forward_vector), dim=-1)

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = rotate

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos
    cam2world = translation_matrix @ rotation_matrix
    return cam2world

def fibonacci_sampling_on_sphere(num_samples=1):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def generate_input_camera(r, poses, device='cuda:0', fov=50):
    def normalize_vecs(vectors): return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    poses = np.deg2rad(poses)
    poses = torch.tensor(poses).float()
    pitch = poses[:, 0]
    yaw = poses[:, 1]

    z = r*torch.sin(pitch)
    x = r*torch.cos(pitch)*torch.cos(yaw)
    y = r*torch.cos(pitch)*torch.sin(yaw)
    cam_pos = torch.stack([x, y, z], dim=-1).reshape(z.shape[0], -1).to(device)

    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 0, -1], dtype=torch.float,
                                        device=device).reshape(-1).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                                        dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                                        dim=-1))
    rotate = torch.stack(
                    (left_vector, up_vector, forward_vector), dim=-1)

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = rotate

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos
    cam2world = translation_matrix @ rotation_matrix

    fx = 0.5/np.tan(np.deg2rad(fov/2))
    fxfycxcy = torch.tensor([fx, fx, 0.5, 0.5], dtype=rotate.dtype, device=device)

    return cam2world, fxfycxcy

def build_grm_model(model_path):
    latest_checkpoint_file, _  = saveload_utils.load_checkpoint(model_path, model=None)
    model_config = latest_checkpoint_file['config'].model_config
    from model import model
    model = model.GRM(model_config).to(device).eval()
    _ = saveload_utils.load_checkpoint(latest_checkpoint_file, model=model)
    return model, model_config

def save_gaussian(latent, gs_path, model, opacity_thr=None):
    xyz = latent['xyz'][0]
    features = latent['feature'][0]
    opacity = latent['opacity'][0]
    scaling = latent['scaling'][0]
    rotation = latent['rotation'][0]

    if opacity_thr is not None:
        index = torch.nonzero(opacity.sigmoid() > opacity_thr)[:, 0]
        xyz = xyz[index]
        features = features[index]
        opacity = opacity[index]
        scaling = scaling[index]
        rotation = rotation[index]

    pc = model.gs_renderer.gaussian_model.set_data(xyz.to(torch.float32), features.to(torch.float32), scaling.to(torch.float32), rotation.to(torch.float32), opacity.to(torch.float32))
    pc.save_ply(gs_path)

def fuse_rgbd_to_mesh(cam_dict, colors, depths, out_mesh_fpath, cam_elev_threshold=0):
    """fuse rgbd into a textured mesh

    Args:
        cam_dict (dict): opencv camera dictionary
        colors (np.ndarray): [N, H, W, 3]; uint8
        depths (np.ndarray): [N, H, W]; float32
        out_mesh_fpath (string): path to obj file
        cam_elev_threshold (float, optional): _description_. Defaults to 0.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=2 * 2.0 / 512.0,
        sdf_trunc=2 * 0.02,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    print("Integrate images into the TSDF volume.")
    for i in tqdm(range(len(cam_dict["frames"]))):
        frame = cam_dict["frames"][i]

        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        cam_elev = np.rad2deg(np.arcsin(cam_pos[2]))
        if cam_elev < cam_elev_threshold:
            # print(f"Camera {i} is below the threshold {cam_elev_threshold}. Skip.")
            continue

        color = o3d.geometry.Image(np.ascontiguousarray(colors[i]))
        depth = o3d.geometry.Image(np.ascontiguousarray(depths[i]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False
        )
        cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        cam_intrinsics.set_intrinsics(
            frame["w"], frame["h"], frame["fx"], frame["fy"], frame["cx"], frame["cy"]
        )
        volume.integrate(
            rgbd,
            cam_intrinsics,
            w2c,
        )

    print("Extract a triangle mesh from the volume and export it.")
    mesh = volume.extract_triangle_mesh()

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 500
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    mesh = mesh.filter_smooth_simple(number_of_iterations=2)
    mesh = mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_mesh_fpath, mesh)


def images2gaussian(images, c2ws, fxfycxcy, model, gs_path, video_path, mesh_path=None, fuse_mesh=False):

    if fuse_mesh:
        fib_camera_path = generate_cameras(r=2.9, num_cameras=200, pitch=np.deg2rad(20), use_fibonacci=True)

    camera_path = generate_cameras(r=2.7, num_cameras=120, pitch=np.deg2rad(20))

    with torch.no_grad():
        with torch.cuda.amp.autocast(
                enabled=True,
                dtype=torch.bfloat16 
        ):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            c2ws = c2ws.to(device, dtype=torch.float32, non_blocking=True)
            fxfycxcy = fxfycxcy.to(device, dtype=torch.float32, non_blocking=True)

            camera_feature =  torch.cat([c2ws.flatten(-2, -1), fxfycxcy], -1)
            gs, _ , _  = model.forward_visual(images, camera=camera_feature, input_fxfycxcy=fxfycxcy, input_c2ws=c2ws)


            filter_mask = torch.nonzero((gs['xyz'].abs() < 1).sum(dim=-1) == 3)
            for key in gs:
                if key == 'depth': continue
                if gs[key] is not None:
                    gs[key] = gs[key][filter_mask[:, 0], filter_mask[:, 1]].unsqueeze(0)

            save_gaussian(gs, gs_path, model, opacity_thr=None)

            gs_rendering = model.gs_renderer.render(latent=gs,
                output_c2ws=camera_path.unsqueeze(0),
                output_fxfycxcy=fxfycxcy[:, 0:1].repeat(1, camera_path.shape[0],1))['image']
            dump_video((gs_rendering[0].permute(0,2,3,1).detach().cpu().numpy()*255).astype(np.uint8), video_path) 

            if fuse_mesh:
                c_nerf_results = model.gs_renderer.render(latent=gs,
                        output_c2ws=fib_camera_path.unsqueeze(0),
                        output_fxfycxcy=fxfycxcy[:, 0:1].repeat(1, fib_camera_path.shape[0],1))

                cnerf_image = c_nerf_results['image'].permute(0, 1, 3, 4, 2)
                cnerf_alpha = c_nerf_results['alpha'].permute(0, 1, 3, 4, 2)
                cnerf_depth = c_nerf_results['depth'].permute(0, 1, 3, 4, 2)


                images = (cnerf_image[0].detach().cpu().numpy()*255).clip(0, 255).astype(np.uint8)

                depths = cnerf_depth[0].detach().cpu().numpy()

                weights_sum = cnerf_alpha[0].detach().cpu().numpy()
                mask = (weights_sum > 1e-2).astype(np.uint8)
                depths = depths * mask - np.ones_like(depths) * (1 - mask)


                cam_dict = {"frames": []}
                hfov = 50

                for j in range(images.shape[0]):
                    frame_dict = {}
                    fx = images[j].shape[1] / 2. / np.tan(np.deg2rad(hfov / 2.0))
                    fy = fx
                    frame_dict["fx"] = fx
                    frame_dict["fy"] = fy
                    frame_dict["cx"] = images[j].shape[1] / 2.
                    frame_dict["cy"] = images[j].shape[0] / 2.
                    frame_dict["h"] = images[j].shape[0]
                    frame_dict["w"] = images[j].shape[1]
                    frame_dict["w2c"] = np.linalg.inv(fib_camera_path[j].detach().cpu().numpy()).tolist()
                    cam_dict["frames"].append(frame_dict)
                fuse_rgbd_to_mesh(cam_dict, images, depths, mesh_path)


def pad_image_to_fit_fov(image, new_fov, old_fov):
    img = Image.fromarray(image)

    scale_factor = math.tan(np.deg2rad(new_fov/2)) / math.tan(np.deg2rad(old_fov/2))

    # Calculate the new size
    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))

    # Calculate padding
    pad_width = (new_size[0]-img.size[0]) // 2
    pad_height = (new_size[1] - img.size[1]) // 2

    # Create padding
    padding = (pad_width, pad_height, pad_width+img.size[0], pad_height+img.size[1])

    # Pad the image
    img_padded = Image.new(img.mode, (new_size[0], new_size[1]), color='white')
    img_padded.paste(img, padding)
    img_padded = np.array(img_padded)
    return img_padded

def instant3d_gs(instant3d_model,
              grm_model,
              grm_model_cfg,
              prompt='a chair',
              guidance_scale=5.0,
              num_steps=30,
              gaussian_sigma=0.1,
              cache_dir='cache',
              fuse_mesh=False):

    image = instant3d_pipe(model=instant3d_model,
                       prompt=prompt,
                       guidance_scale=guidance_scale,
                       num_steps=num_steps,
                       gaussian_sigma=gaussian_sigma)
    torch.cuda.empty_cache()

    # reshape 2H * 2W * C ---> 4* H * W * C
    image = image.permute(1, 2, 0)
    shape = image.shape[0]
    out_s = int(shape//2)
    image = image.reshape(2, out_s, 2, out_s, 3)
    image = image.permute(0, 2, 1, 3, 4)
    image = image.reshape(4, out_s, out_s, 3)
    # normalize
    image = image[None]
    image = (image - 0.5)*2
    # 1, V, C, H, W
    image = image.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[20, 225], [20, 225+90], [20, 225+180], [20, 225+270]], fov=50)
    c2ws = c2ws[None]
    fxfycxcy = (fxfycxcy.unsqueeze(0).unsqueeze(0)).repeat(1, c2ws.shape[1], 1)

    prompt = '_'.join(prompt.split())
    images2gaussian(image, c2ws, fxfycxcy, grm_model, f'./{cache_dir}/{prompt}_gs.ply', f'{cache_dir}/{prompt}.mp4', f'{cache_dir}/{prompt}_mesh.ply', fuse_mesh=fuse_mesh)
    torch.cuda.empty_cache()

def zero123plus_v11(
          zero123_model,
          grm_model,
          grm_model_cfg,
          image_path,
          num_steps=30,
          cache_dir='cache',
          fuse_mesh=False,
          ):
    cond = Image.open(image_path)
    images = zero123_model(cond, num_inference_steps=num_steps).images[0]
    images = np.array(images)

    bg_remover = rembg.new_session()
    shape = images.shape[0]
    out_s = int(shape//3)
    images = images.reshape(3, out_s, 2, out_s, 3)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(6, out_s, out_s, 3)

    input_size = grm_model_cfg.visual.params.input_res
    mv_images = []
    for idx in [0, 2, 4, 5]:
        image = rembg.remove(images[idx], session=bg_remover)
        image = image / 255
        image_fg = image[..., :3]*image[..., 3:] + (1-image[..., 3:])
        image_fg = cv2.resize(image_fg, (input_size, input_size))
        mv_images.append(image_fg) 

    # normalize
    images = np.stack(mv_images, axis=0)[None]
    images = (images - 0.5)*2
    images = torch.tensor(images).to(device)
    # 1, V, C, H, W
    images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[30, 225+30], [30, 225+150], [30, 225+270], [-20, 225+330]], fov=50)
    c2ws = c2ws[None]
    fxfycxcy = (fxfycxcy.unsqueeze(0).unsqueeze(0)).repeat(1, c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]
    images2gaussian(images, c2ws, fxfycxcy, grm_model, f'./{cache_dir}/{name}_gs.ply', f'{cache_dir}/{name}.mp4', f'{cache_dir}/{name}_mesh.ply', fuse_mesh=fuse_mesh)
    torch.cuda.empty_cache()
    

def zero123plus_v12(
          zero123_model,
          grm_model,
          grm_model_cfg,
          image_path,
          num_steps=30,
          cache_dir='cache',
          fuse_mesh=False
          ):
    cond = Image.open(image_path)
    images = zero123_model(cond, num_inference_steps=num_steps).images[0]
    images = np.array(images)

    bg_remover = rembg.new_session()
    shape = images.shape[0]
    out_s = int(shape//3)
    images = images.reshape(3, out_s, 2, out_s, 3)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(6, out_s, out_s, 3)

    input_size = grm_model_cfg.visual.params.input_res
    mv_images = [] 
    for idx in [0, 2, 4, 5]:
        image = rembg.remove(images[idx], session=bg_remover)
        image = image / 255
        image_fg = image[..., :3]*image[..., 3:] + (1-image[..., 3:])
        image_fg = pad_image_to_fit_fov((image_fg*255).astype(np.uint8), 50, 30)
        image_fg = cv2.resize(image_fg, (input_size, input_size))
        image_fg = image_fg / 255
        mv_images.append(image_fg)

    # normalize
    images = np.stack(mv_images, axis=0)[None]
    images = (images - 0.5)*2
    images = torch.tensor(images).to(device)
    # 1, V, C, H, W
    images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[20, 225+30], [20, 225+150], [20, 225+270], [-10, 225+330]], fov=50)
    c2ws = c2ws[None]
    fxfycxcy = (fxfycxcy.unsqueeze(0).unsqueeze(0)).repeat(1, c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]
    images2gaussian(images, c2ws, fxfycxcy, grm_model, f'./{cache_dir}/{name}_gs.ply', f'{cache_dir}/{name}.mp4', f'{cache_dir}/{name}_mesh.ply', fuse_mesh=fuse_mesh)
    torch.cuda.empty_cache()

def sv3d_gs(grm_model,
          grm_model_cfg,
          image_path,
          num_steps=30,
          cache_dir='cache',
          fuse_mesh=False
          ):

    video = sv3d_pipe(model=None,
                input_path=image_path,
                version='sv3d_p',
                elevations_deg=20.0,
                azimuths_deg=[0,10,30,50,90,110,130,150,180,200,220,240,270,280,290,300,310,320,330,340,350],
                output_folder=f'{args.cache_dir}/sv3d')
    torch.cuda.empty_cache()

    input_size = grm_model_cfg.visual.params.input_res
    mv_images = video[[0, 4, 8, 12]]
    
    mv_images = [ cv2.resize(pad_image_to_fit_fov(image, 50, 33.8), (input_size, input_size)) for image in mv_images]


    # normalize
    images = np.stack(mv_images, axis=0)[None]
    images = (images/255 - 0.5)*2
    images = torch.tensor(images).to(device)
    # 1, V, C, H, W
    images = images.permute(0, 1, 4, 2, 3)

    # generate input pose
    c2ws, fxfycxcy = generate_input_camera(2.7, [[20, 225], [20, 225+90], [20, 225+180], [20, 225+270]], fov=50)
    c2ws = c2ws[None]
    fxfycxcy = (fxfycxcy.unsqueeze(0).unsqueeze(0)).repeat(1, c2ws.shape[1], 1)

    name = os.path.splitext(os.path.basename(image_path))[0]
    images2gaussian(images, c2ws, fxfycxcy, grm_model, f'./{cache_dir}/{name}_gs.ply', f'{cache_dir}/{name}.mp4', f'{cache_dir}/{name}_mesh.ply', fuse_mesh=fuse_mesh)
    torch.cuda.empty_cache()

def main(args):

    seed = args.seed 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ### init GRM model
    grm_uniform_path = 'checkpoints/grm_u.pth'
    grm_uniform_model, grm_uniform_config = build_grm_model(grm_uniform_path)

    grm_zero123plus_path = 'checkpoints/grm_zero123plus.pth'
    grm_zero123plus_model, grm_zero123plus_config = build_grm_model(grm_zero123plus_path)

    grm_random_path = 'checkpoints/grm_r.pth'
    grm_random_model, grm_random_config = build_grm_model(grm_random_path)

    os.makedirs(args.cache_dir, exist_ok=True)
    if args.prompt:
        ### initial instant3d model
        instant3d_model = build_instant3d_model(config_path='third_party/generative_models/configs/sd_xl_base.yaml', ckpt_path='checkpoints/instant3d.pth')
        instant3d_gs(instant3d_model,
                  grm_model=grm_uniform_model,
                  grm_model_cfg=grm_uniform_config,
                  prompt=args.prompt,
                  guidance_scale=5.0,
                  num_steps=30,
                  gaussian_sigma=0.1,
                  cache_dir=args.cache_dir,
                  fuse_mesh=args.fuse_mesh)

    else:
        if args.model == 'sv3d':
            sv3d_gs(
                  grm_model=grm_uniform_model,
                  grm_model_cfg=grm_uniform_config,
                  image_path=args.image_path,
                  num_steps=30,
                  cache_dir=args.cache_dir,
                  fuse_mesh=args.fuse_mesh,
                  )
        elif args.model == 'zero123plus-v1.1':
            zero123 = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            zero123.scheduler = EulerAncestralDiscreteScheduler.from_config(
                zero123.scheduler.config, timestep_spacing='trailing'
            )
            zero123.to(device)
            zero123plus_v11(
              zero123,
              grm_model=grm_zero123plus_model,
              grm_model_cfg=grm_zero123plus_config,
              image_path=args.image_path,
              num_steps=30,
              cache_dir=args.cache_dir,
              fuse_mesh=args.fuse_mesh,
              )
        elif args.model == 'zero123plus-v1.2':
            zero123 = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            zero123.scheduler = EulerAncestralDiscreteScheduler.from_config(
                zero123.scheduler.config, timestep_spacing='trailing'
            )
            zero123.to(device)
            zero123plus_v12(
              zero123,
              grm_model=grm_random_model,
              grm_model_cfg=grm_random_config,
              image_path=args.image_path,
              num_steps=30,
              cache_dir=args.cache_dir,
              fuse_mesh=args.fuse_mesh,
              )
        else:
            raise NotImplementedError


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default='examples/image.png',
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
    )
    parser.add_argument(
        "--cache_dir", type=str, default='cache',
        help='The directory to save the output'
    )
    parser.add_argument(
        "--model", type=str, default='zero123plus-v1.1', 
        help='Choose from zero123plus-v1.1/zero123plus-v1.2/sv3d'
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument(
        "--fuse_mesh", type=bool, default=False, help='Whether to get the mesh.'
    )
    args = parser.parse_args()

    main(args)

