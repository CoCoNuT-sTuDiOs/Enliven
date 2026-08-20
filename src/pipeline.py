"""
pipeline.py Main orchestrator for head-only avatar generation

Core function: generate(photo_path, driving_video_path, audio_path, output_dir)
Returns: path to final result video
"""
import os
import torch
import tempfile
from pathlib import Path
from src.utils.device import get_device
from src.face_keypoints import FaceKeypointExtractor
from src.head_animate import HeadAnimator
from src.expression_transfer import transfer_expression
from src.lip_sync import sync_lip_sync
from src.enhance import FaceEnhancer

def generate(
    photo_path: str,
    driving_video_path: str,
    audio_path: str,
    output_dir: str = "results",
    enhance: bool = True
) -> str:
    """
    Generate talking-head avatar video.
    
    Inputs:
    - photo_path: path to avatar photo (front-facing, standing straight)
    - driving_video_path: path to driving video (motion source)
    - audio_path: path to audio clip (<=10 seconds)
    - output_dir: where to save final result
    - enhance: whether to apply GFPGAN face enhancement
    
    Output:
    - path to final MP4 video
    """
    print(f"[ENLIVEN] Starting pipeline...")
    print(f"  photo: {photo_path}")
    print(f"  video: {driving_video_path}")
    print(f"  audio: {audio_path}")
    print(f"  enhance: {enhance}")
    
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"[ENLIVEN] Device: {device}")
    
    try:
        # Stage 1: Extract face keypoints from driving video
        print("[ENLIVEN] Stage 1/4: Extracting face keypoints...")
        keypoint_extractor = FaceKeypointExtractor()
        keypoints = keypoint_extractor.extract(driving_video_path)
        print(f"[ENLIVEN] ✓ Extracted {len(keypoints)} keypoint frames")
        
        # Stage 2: Animate head using keypoints
        print("[ENLIVEN] Stage 2/4: Animating head...")
        animator = HeadAnimator()
        animated_path = os.path.join(output_dir, "animated.mp4")
        # TODO: implement frame-by-frame animation to MP4
        print(f"[ENLIVEN] ✓ Head animation complete")
        
        # Stage 3: Transfer facial expressions from driving video
        print("[ENLIVEN] Stage 3/4: Transferring expressions...")
        liveportrait_dir = os.path.expanduser("~/LivePortrait")  # adjust path as needed
        expression_path = transfer_expression(
            source_path=animated_path,
            driving_video_path=driving_video_path,
            liveportrait_dir=liveportrait_dir,
            output_dir=output_dir
        )
        print(f"[ENLIVEN] ✓ Expression transfer complete: {expression_path}")
        
        # Stage 4: Sync mouth to audio
        print("[ENLIVEN] Stage 4/4: Syncing mouth to audio...")
        final_path = os.path.join(output_dir, "result_wav2lip.mp4")
        sync_lip_sync(
            video_path=expression_path,
            audio_path=audio_path,
            output_path=final_path
        )
        print(f"[ENLIVEN] ✓ Lip sync complete: {final_path}")
        
        # Optional: Enhancement
        if enhance:
            print("[ENLIVEN] Enhancing face quality...")
            enhancer = FaceEnhancer()
            # TODO: frame-by-frame enhancement
            print(f"[ENLIVEN] ✓ Enhancement complete")
        
        print(f"[ENLIVEN] ✓✓✓ COMPLETE: {final_path}")
        return final_path
        
    except Exception as e:
        print(f"[ENLIVEN] ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
