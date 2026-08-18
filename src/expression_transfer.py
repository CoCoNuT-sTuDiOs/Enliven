"""
expression_transfer.py

"""

import os
import subprocess
import glob


def transfer_expression(
    source_path: str,
    driving_video_path: str,
    liveportrait_dir: str,
    output_dir: str = "animations",
    flag_use_half_precision: bool = True,
) -> str:
    source_path = os.path.abspath(source_path)
    driving_video_path = os.path.abspath(driving_video_path)

    cmd = [
        "python", "inference.py",
        "--source", source_path,
        "--driving", driving_video_path,
        "--output_dir", output_dir,
    ]
    if not flag_use_half_precision:
        cmd += ["--no_flag_use_half_precision"]  # tyro auto-generates --no_ flags for bool fields

    result = subprocess.run(
        cmd,
        cwd=liveportrait_dir,
        capture_output=True,
        text=True,
    )

    # Always surface stdout/stderr — helpful for debugging even on success,
    # since LivePortrait logs useful info (crop details, output paths) here.
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"LivePortrait inference.py failed with exit code {result.returncode}"
        )

    output_dir_abs = os.path.join(liveportrait_dir, output_dir)
    candidates = [
        f for f in glob.glob(os.path.join(output_dir_abs, "*.mp4"))
        if "_concat" not in f
    ]
    if not candidates:
        raise FileNotFoundError(
            f"LivePortrait ran successfully but no output .mp4 was found in {output_dir_abs}"
        )
    result_path = max(candidates, key=os.path.getmtime)

    return result_path
