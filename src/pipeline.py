"""
pipeline.py Main orchestrator for head-only avatar generation

Core function: generate(photo_path, driving_video_path, audio_path, output_dir)
Returns: path to final result video
"""
import os
import cv2
import torch
import tempfile
from pathlib import Path
from src.utils.device import get_device
from src.face_keypoints import FaceKeypointExtractor
from src.head_animate import HeadAnimator
from src.expression_transfer import transfer_expression
from src.lip_sync import sync_lip_sync
from src.enhance import FaceEnhancer


def _get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def _downsample_video(video_path: str, indices, output_path: str, fps: float) -> str:
    """Write out only the frames at `indices` from video_path, in order."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx_set = set(indices)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frames.append(frame)
        i += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames read for downsampling from {video_path}")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return output_path


def generate(
    photo_path: str,
    driving_video_path: str,
    audio_path: str,
    output_dir: str = "results",
    enhance: bool = True,
    test_fps: int = None,
    num_inference_steps: int = 20,
    strength: float = 0.4,
    conditioning_scale: float = 1.0,
) -> str:
    """
    Generate talking-head avatar video.

    Inputs:
    - photo_path: path to avatar photo (front-facing, standing straight)
    - driving_video_path: path to driving video (motion source)
    - audio_path: path to audio clip (<=10 seconds)
    - output_dir: where to save final result
    - enhance: whether to apply GFPGAN face enhancement
    - test_fps: if set, downsample Stage 2 (and the driving video fed into
      Stage 3) to this fps to cut GPU cost for smoke tests. Omit for full
      quality — uses the driving video's native fps.
    - num_inference_steps: diffusion steps per frame in Stage 2. Lower for
      cheap/fast smoke tests, raise for final quality.
    - strength: Stage 2 img2img strength (0-1, low = closer to source photo)
    - conditioning_scale: Stage 2 ControlNet pose conditioning scale

    Output:
    - path to final MP4 video
    """
    print(f"[ENLIVEN] Starting pipeline...")
    print(f"  photo: {photo_path}")
    print(f"  video: {driving_video_path}")
    print(f"  audio: {audio_path}")
    print(f"  enhance: {enhance}")
    print(f"  test_fps: {test_fps}  steps: {num_inference_steps}  strength: {strength}")

    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"[ENLIVEN] Device: {device}")

    try:
        # Stage 1: Extract face keypoints from driving video
        print("[ENLIVEN] Stage 1/4: Extracting face keypoints...")
        keypoint_extractor = FaceKeypointExtractor()
        keypoints = keypoint_extractor.extract(driving_video_path)
        print(f"[ENLIVEN] ✓ Extracted {len(keypoints)} keypoint frames")

        # Optional downsample for cheap smoke tests. Stage 2 (diffusion) is
        # by far the most expensive stage, so cutting frame count here saves
        # the most GPU time. We also re-cut the driving video to the SAME
        # frame indices so Stage 3 (LivePortrait) stays frame-aligned with
        # Stage 2's output — otherwise expression transfer would be fed
        # mismatched frame counts.
        source_fps = _get_video_fps(driving_video_path)
        run_fps = test_fps if test_fps else source_fps
        driving_for_stage3 = driving_video_path

        if test_fps and test_fps < source_fps:
            step = max(1, round(source_fps / test_fps))
            indices = list(range(0, len(keypoints), step))
            keypoints = keypoints[indices]
            driving_for_stage3 = os.path.join(output_dir, "driving_downsampled.mp4")
            _downsample_video(driving_video_path, indices, driving_for_stage3, run_fps)
            print(f"[ENLIVEN] Test mode: downsampled to {run_fps}fps, "
                  f"{len(keypoints)} frames (was {source_fps}fps)")

        # Stage 2: Animate head using keypoints
        print("[ENLIVEN] Stage 2/4: Animating head...")
        from PIL import Image
        animator = HeadAnimator()
        source_image = Image.open(photo_path).convert("RGB")
        animated_path = os.path.join(output_dir, "animated.mp4")
        animator.animate_video(
            source_image=source_image,
            keypoints_seq=keypoints,
            output_path=animated_path,
            fps=run_fps,
            strength=strength,
            conditioning_scale=conditioning_scale,
        )
        print(f"[ENLIVEN] ✓ Head animation complete: {animated_path}")

        # Stage 3: Transfer facial expressions from driving video
        print("[ENLIVEN] Stage 3/4: Transferring expressions...")
        liveportrait_dir = os.path.expanduser("~/LivePortrait")  # adjust path as needed
        expression_path = transfer_expression(
            source_path=animated_path,
            driving_video_path=driving_for_stage3,
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
