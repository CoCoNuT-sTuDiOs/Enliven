"""
lip_sync.py

"""

import os
import subprocess
import glob


def sync_lips(
    video_path: str,
    audio_path: str,
    wav2lip_dir: str,
    checkpoint_name: str = "wav2lip_gan.pth",
    output_dir: str = "results",
) -> str:
    """
    Runs as a subprocess to sync mouth movement in a video to an
    audio track.

    Args:
        video_path: path to the input video (face must be visible/detectable)
        audio_path: path to the audio file to sync to
        wav2lip_dir: path to the cloned Wav2Lip repo
        checkpoint_name: which checkpoint to use — "wav2lip_gan.pth"
            (better visual quality) or "wav2lip.pth" (more accurate sync,
            per Wav2Lip's own README)
        output_dir: unused directly (Wav2Lip hardcodes its own results
            path), kept for interface consistency with the other stages

    Returns:
        Path to the generated output video.
    """
    video_path = os.path.abspath(video_path)
    audio_path = os.path.abspath(audio_path)
    checkpoint_path = os.path.join("checkpoints", checkpoint_name)

    cmd = [
        "python", "inference.py",
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
    ]

    result = subprocess.run(
        cmd,
        cwd=wav2lip_dir,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"Wav2Lip inference.py failed with exit code {result.returncode}"
        )

    # Wav2Lip's default output path, per their own README
    result_path = os.path.join(wav2lip_dir, "results", "result_voice.mp4")
    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f"Wav2Lip ran successfully but expected output not found at {result_path}"
        )

    return result_path
