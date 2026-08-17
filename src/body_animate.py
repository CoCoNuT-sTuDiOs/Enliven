"""
body_animate.py

Stage 2 of the Enliven pipeline: takes an avatar photo + a driving video,
returns a video of the avatar performing that motion.

This is a direct, faithful port of MimicMotion's own inference.py logic
(preprocess() + run_pipeline() + main()'s loop body), wrapped as a single
reusable function instead of a CLI script tied to a config file.

Requires MimicMotion's repo to be cloned and on the Python path, and its
dependencies installed/patched first — see src/utils/model_setup.py:
    install_mimicmotion_deps()
    patch_mimicmotion(mimicmotion_dir)
"""

import os
import sys
import math
from datetime import datetime

import numpy as np
import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop, to_pil_image
from omegaconf import OmegaConf


def _preprocess(video_path, image_path, aspect_ratio, resolution=576, sample_stride=2):
    from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels)
    h, w = image_pixels.shape[-2:]

    if h > w:
        w_target, h_target = resolution, int(resolution / aspect_ratio // 64) * 64
    else:
        w_target, h_target = int(resolution / aspect_ratio // 64) * 64, resolution

    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target

    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()

    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))

    return (
        torch.from_numpy(pose_pixels.copy()) / 127.5 - 1,
        torch.from_numpy(image_pixels) / 127.5 - 1,
    )


def _run_pipeline(pipeline, image_pixels, pose_pixels, device, num_frames,
                   tile_size, tile_overlap, noise_aug_strength,
                   num_inference_steps, guidance_scale, seed):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    frames = pipeline(
        image_pixels,
        image_pose=pose_pixels,
        num_frames=num_frames,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        height=pose_pixels.shape[-2],
        width=pose_pixels.shape[-1],
        fps=7,
        noise_aug_strength=noise_aug_strength,
        num_inference_steps=num_inference_steps,
        generator=generator,
        min_guidance_scale=guidance_scale,
        max_guidance_scale=guidance_scale,
        decode_chunk_size=8,
        output_type="pt",
        device=device,
    ).frames.cpu()

    video_frames = (frames * 255.0).to(torch.uint8)
    _video_frames = video_frames[0, 1:]
    return _video_frames


def animate_body(
    avatar_path: str,
    driving_video_path: str,
    mimicmotion_dir: str,
    output_dir: str = "outputs",
    num_frames: int = 16,
    resolution: int = 384,
    frames_overlap: int = 4,
    num_inference_steps: int = 25,
    sample_stride: int = 2,
    fps: int = 15,
    seed: int = 42,
    noise_aug_strength: float = 0,
    guidance_scale: float = 2.0,
) -> str:
    if mimicmotion_dir not in sys.path:
        sys.path.insert(0, mimicmotion_dir)

    from mimicmotion.utils.geglu_patch import patch_geglu_inplace
    patch_geglu_inplace()

    from mimicmotion.utils.loader import create_pipeline
    from mimicmotion.utils.utils import save_to_mp4

    constants_path = os.path.join(mimicmotion_dir, "constants.py")
    aspect_ratio_ns = {}
    with open(constants_path) as f:
        exec(f.read(), aspect_ratio_ns)
    aspect_ratio = aspect_ratio_ns["ASPECT_RATIO"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    infer_config = OmegaConf.create({
        "base_model_path": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "ckpt_path": os.path.join(mimicmotion_dir, "models", "MimicMotion_1-1.pth"),
    })

    pipeline = create_pipeline(infer_config, device)

    pose_pixels, image_pixels = _preprocess(
        driving_video_path, avatar_path, aspect_ratio,
        resolution=resolution, sample_stride=sample_stride,
    )

    video_frames = _run_pipeline(
        pipeline, image_pixels, pose_pixels, device,
        num_frames=pose_pixels.size(0),
        tile_size=num_frames,
        tile_overlap=frames_overlap,
        noise_aug_strength=noise_aug_strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.basename(driving_video_path).split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = os.path.join(output_dir, f"{output_name}_{timestamp}.mp4")

    save_to_mp4(video_frames, output_path, fps=fps)

    return output_path
