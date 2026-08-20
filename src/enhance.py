"""
enhance.py — GFPGAN face enhancement for video frames
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from src.utils.device import get_device

class FaceEnhancer:
    def __init__(self):
        self.device = get_device()
        try:
            from gfpgan import GFPGANer
            # Download model on first use
            self.gfpgan = GFPGANer(
                scale=2,
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            self.available = True
        except Exception as e:
            print(f"[enhance] Warning: GFPGAN init failed: {e}")
            self.gfpgan = None
            self.available = False
    
    def enhance_video(self, video_path: str, output_path: str) -> str:
        """
        Frame-by-frame GFPGAN enhancement of video.
        Reads video, enhances each frame, writes to output MP4.
        
        Args:
            video_path: Input video (from Wav2Lip or LivePortrait)
            output_path: Output enhanced video
        
        Returns:
            output_path if successful, original video_path if enhancement skipped
        """
        if not self.available:
            print("[enhance] GFPGAN unavailable, returning original")
            return video_path
        
        print(f"[enhance] Reading video: {video_path}")
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[enhance] Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"[enhance] Failed to open output video writer")
            cap.release()
            return video_path
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Enhance this frame
                try:
                    _, _, enhanced = self.gfpgan.enhance(frame, has_aligned=False, only_center_face=True, paste_back=True, weight=0.5)
                    # Resize back to original if GFPGAN changed dimensions
                    if enhanced.shape[:2] != frame.shape[:2]:
                        enhanced = cv2.resize(enhanced, (width, height))
                    out.write(enhanced)
                except Exception as e:
                    print(f"[enhance] Warning: Frame {frame_count} failed: {e}, using original")
                    out.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"[enhance] Progress: {frame_count}/{total_frames}")
        
        except Exception as e:
            print(f"[enhance] Error during enhancement: {e}")
        
        finally:
            cap.release()
            out.release()
        
        print(f"[enhance] ✓ Enhanced {frame_count} frames → {output_path}")
        return output_path
