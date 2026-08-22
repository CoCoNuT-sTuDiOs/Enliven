import os
import sys
from time import strftime

from vendor.liveportrait.src.config.argument_config import ArgumentConfig
from vendor.liveportrait.src.config.crop_config import CropConfig
from vendor.liveportrait.src.config.inference_config import InferenceConfig
from vendor.liveportrait.src.live_portrait_pipeline import LivePortraitPipeline
from vendor.liveportrait.src.live_portrait_pipeline_animal import LivePortraitPipelineAnimal


LIVEPORTRAIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "liveportrait")
sys.path.insert(0, LIVEPORTRAIT_DIR)


_pipeline = None


def init_models():
    """Call once at Space startup. Model placement at import time (not inside a
    @spaces.GPU call) is the documented ZeroGPU pattern — actual GPU kernel execution
    only needs to happen inside the decorated function, not model construction."""
    global _pipeline
    print("[JOB_PROCESSOR] Loading LivePortrait pipeline...")
    _pipeline = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())
    print("[JOB_PROCESSOR] Pipeline loaded and ready.")


def run_job(photo_path: str, driving_video_path: str, output_dir: str = "results") -> str:
    """Runs one LivePortrait generation. Returns path to the final .mp4."""
    print(f"[RUN_JOB] photo={photo_path} driving_video={driving_video_path}")
    os.makedirs(output_dir, exist_ok=True)

    args = ArgumentConfig(
        source=photo_path,
        driving=driving_video_path,
        output_dir=output_dir,
    )
    wfp, wfp_concat = _pipeline.execute(args)
    print(f"[RUN_JOB] ✓ done: {wfp}")
    return wfp