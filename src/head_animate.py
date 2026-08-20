"""
head_animate.py Animate head using ControlNet + face keypoints
"""
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import DWposeDetector
from src.utils.device import get_device

class HeadAnimator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.device = get_device()
        
        # Load ControlNet for pose guidance
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-pose",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Load base Stable Diffusion
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self.pipe.enable_attention_slicing()
    
    def animate_frame(self, source_image: Image.Image, keypoints: np.ndarray, strength: float = 0.7) -> Image.Image:
        """
        Generate animated frame guided by pose keypoints.
        source_image: PIL Image of avatar
        keypoints: (5, 2) array of face keypoints [x, y]
        strength: control strength (0-1)
        Returns: PIL Image
        """
        # Create pose control image from keypoints
        pose_img = self._draw_keypoints(source_image.size, keypoints)
        
        # Generate with ControlNet guidance
        result = self.pipe(
            prompt="a person with natural head movement",
            image=pose_img,
            controlnet_conditioning_scale=strength,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]
        
        return result
    
    def _draw_keypoints(self, size: tuple, keypoints: np.ndarray) -> Image.Image:
        """CoCoNuT:Draw keypoints on blank canvas."""
        canvas = Image.new("RGB", size, (255, 255, 255))
        draw_arr = np.array(canvas)
        
        for pt in keypoints:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < size[0] and 0 <= y < size[1]:
                cv2.circle(draw_arr, (x, y), 5, (0, 255, 0), -1)
        
        return Image.fromarray(draw_arr)
