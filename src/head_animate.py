"""
head_animate.py Animate head using ControlNet + face keypoints

FIX (Aug 2026): Stage 2 was previously pure txt2img it only ever saw the
pose-dot canvas + a text prompt, never the actual avatar photo, so every
frame generated a different random face. Switched to img2img ControlNet so
generation is anchored to source_image (identity) while pose guidance
(motion) comes from ControlNet. `strength` now controls how far the img2img
pass is allowed to drift from the source photo (low = closer to source).
"""
import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from src.utils.device import get_device

class HeadAnimator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.device = get_device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load ControlNet for pose guidance
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            torch_dtype=dtype
        ).to(self.device)

        # Load base Stable Diffusion as img2img so the avatar photo actually
        # anchors identity — this is the fix. Previously this was
        # StableDiffusionControlNetPipeline (txt2img), which ignored the
        # source photo entirely.
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=dtype
        ).to(self.device)

        self.pipe.enable_attention_slicing()

    def animate_frame(
        self,
        source_image: Image.Image,
        keypoints: np.ndarray,
        strength: float = 0.4,
        conditioning_scale: float = 1.0,
    ) -> Image.Image:
        """
        Generate one animated frame that keeps the avatar's identity but
        shifts head pose to match `keypoints`.

        source_image: PIL Image of avatar (used as the img2img base — this
            is what keeps every frame looking like the same person)
        keypoints: (5, 2) array of face keypoints [x, y] for this frame
        strength: img2img denoising strength (0-1). LOW values (0.3-0.5)
            stay close to source_image's identity/appearance. HIGH values
            drift further and start losing who the person is. Do not
            confuse this with conditioning_scale below.
        conditioning_scale: how strongly ControlNet's pose skeleton steers
            the result (0-1ish, can go a bit above 1). This is the pose
            control strength, separate from img2img `strength`.
        Returns: PIL Image, same size as source_image
        """
        pose_img = self._draw_keypoints(source_image.size, keypoints)

        result = self.pipe(
            prompt="a person with natural head movement, same face, photorealistic",
            image=source_image,               # img2img base = the actual avatar photo
            control_image=pose_img,            # ControlNet steers pose only
            strength=strength,
            controlnet_conditioning_scale=conditioning_scale,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]

        return result

    def animate_video(
        self,
        source_image: Image.Image,
        keypoints_seq: np.ndarray,
        output_path: str,
        fps: int = 25,
        strength: float = 0.4,
        conditioning_scale: float = 1.0,
    ) -> str:
        """
        Run animate_frame() over every frame of keypoints_seq and write the
        result to an MP4 at output_path.

        keypoints_seq: (num_frames, 5, 2) array from FaceKeypointExtractor
        Returns: output_path
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        w, h = source_image.size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        try:
            for i, kpts in enumerate(keypoints_seq):
                frame = self.animate_frame(
                    source_image, kpts,
                    strength=strength,
                    conditioning_scale=conditioning_scale,
                )
                frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                if (i + 1) % 5 == 0 or (i + 1) == len(keypoints_seq):
                    print(f"[head_animate] frame {i + 1}/{len(keypoints_seq)}")
        finally:
            writer.release()

        return output_path

    def _draw_keypoints(self, size: tuple, keypoints: np.ndarray) -> Image.Image:
        """
        Draw pose control image from 5 face keypoints
        (nose, left_eye, right_eye, left_ear, right_ear).

        NOTE: lllyasviel/sd-controlnet-openpose was trained on
        full OpenPose skeletons (colored limb lines on a BLACK canvas), not
        isolated dots on white. With only 5 face points we can't draw real
        limbs, but matching the black background + OpenPose-style coloring
        gives the model a signal closer to its training distribution than
        plain dots on white, which it has likely never seen. If pose
        guidance still feels weak/ignored in testing, this is the first
        thing to revisit — may need to switch to a face-specific ControlNet
        (e.g. a landmarks/mediapipe-face conditioned model) instead.
        """
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)  # black canvas

        # Rough OpenPose face-point color convention
        colors = [
            (0, 0, 255),    # nose - red (BGR)
            (0, 255, 0),    # left_eye - green
            (255, 0, 0),    # right_eye - blue
            (0, 255, 255),  # left_ear - yellow
            (255, 0, 255),  # right_ear - magenta
        ]

        for idx, pt in enumerate(keypoints):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < size[0] and 0 <= y < size[1]:
                color = colors[idx % len(colors)]
                cv2.circle(canvas, (x, y), 6, color, -1)

        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
