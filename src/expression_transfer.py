"""
expression_transfer.py

"""

import os
import sys
import glob


def transfer_expression(
    source_path: str,
    driving_video_path: str,
    liveportrait_dir: str,
    output_dir: str = "animations",
    flag_use_half_precision: bool = True,
    flag_do_crop: bool = True,
    flag_pasteback: bool = True,
) -> str:
    if liveportrait_dir not in sys.path:
        sys.path.insert(0, liveportrait_dir)

    # LivePortrait's own modules assume being run from its own directory
    # (relative imports, relative default paths) — matching how its own
    # inference.py is normally invoked.
    original_cwd = os.getcwd()
    os.chdir(liveportrait_dir)

    try:
        from src.config.argument_config import ArgumentConfig
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline

        def partial_fields(target_class, kwargs):
            return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

        args = ArgumentConfig(
            source=source_path,
            driving=driving_video_path,
            output_dir=output_dir,
            flag_use_half_precision=flag_use_half_precision,
            flag_do_crop=flag_do_crop,
            flag_pasteback=flag_pasteback,
        )

        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        pipeline.execute(args)

        # LivePortrait names its own output based on source/driving
        # basenames (e.g. "s0--d0.mp4" / "s0--d0_concat.mp4"), not a path
        # it directly returns  so we find the most recently created
        # non-concat .mp4 in the output dir instead of guessing the name.
        output_dir_abs = os.path.join(liveportrait_dir, output_dir)
        candidates = [
            f for f in glob.glob(os.path.join(output_dir_abs, "*.mp4"))
            if "_concat" not in f
        ]
        if not candidates:
            raise FileNotFoundError(
                f"LivePortrait ran but no output .mp4 was found in {output_dir_abs}"
            )
        result_path = max(candidates, key=os.path.getmtime)

    finally:
        os.chdir(original_cwd)

    return result_path
