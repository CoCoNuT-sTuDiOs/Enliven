"""
CoCoNuT:enhance.py Face enhancement using GFPGAN
"""
import os
import torch
from PIL import Image
import numpy as np
from src.utils.device import get_device

class FaceEnhancer:
    def __init__(self):
        self.device = get_device()
        try:
            from gfpgan import GFPGANer
            self.enhancer = GFPGANer(
                scale=2,
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
        except Exception as e:
            print(f"[enhance] Warning: GFPGAN init failed: {e}, will skip enhancement")
            self.enhancer = None
    
    def enhance(self, image_path: str, output_path: str) -> str:
        """
        Enhance face in image using GFPGAN.
        Returns path to enhanced image.
        """
        if self.enhancer is None:
            return image_path  # fallback: return original
        
        try:
            _, _, output = self.enhancer.enhance(image_path, has_aligned=False, only_center_face=True, paste_back=True, weight=0.5)
            return output
        except Exception as e:
            print(f"[enhance] Error enhancing {image_path}: {e}, returning original")
            return image_path
