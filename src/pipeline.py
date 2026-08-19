"""
pipeline.py

The full Enliven pipeline: avatar photo + driving video + audio -> final video.

Stage order (locked, do not reverse):
  1. MimicMotion  (body_animate)   avatar + driving video -> body-animated video
  2. LivePortrait (expression_transfer) that video's face + driving video's
     expression -> expression-corrected video
  3. Wav2Lip      (lip_sync)  that video + audio -> final lip-synced video

Expression must come from LivePortrait before Wav2Lip's mouth-only pass,
never the reverse Wav2Lip's precise phoneme timing would otherwise get
overwritten by LivePortrait's broader expression pass.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from body_animate import animate_body
from expression_transfer import transfer_expression
from lip_sync import sync_lips


def generate(
    avatar_path: str,
    driving_video_path: str,
    audio_path: str,
    mimicmotion_dir: str,
    liveportrait_dir: str,
    wav2lip_dir: str,
    output_dir: str = "enliven_output",
    num_frames: int = 16,
    resolution: int = 384,
    num_inference_steps: int = 25,
) -> str:
    """
    Runs the full Enliven pipeline end to end.

    Returns:
        Path to the final generated video.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("[1/3] Animating body (MimicMotion)...")
    body_video = animate_body(
        avatar_path=avatar_path,
        driving_video_path=driving_video_path,
        mimicmotion_dir=mimicmotion_dir,
        output_dir=output_dir,
        num_frames=num_frames,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
    )
    print(f"  -> {body_video}")

    print("[2/3] Transferring expression (Enliven LivePortrait)...")
    expression_video = transfer_expression(
        source_path=body_video,
        driving_video_path=driving_video_path,
        liveportrait_dir=liveportrait_dir,
        output_dir=output_dir,
    )
    print(f"  -> {expression_video}")

    print("[3/3] Syncing lips (Eliven Wav2Lip)...")
    final_video = sync_lips(
        video_path=expression_video,
        audio_path=audio_path,
        wav2lip_dir=wav2lip_dir,
    )
    print(f"  -> {final_video}")

    return final_video
