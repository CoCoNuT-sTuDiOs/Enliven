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
    """Call once at Space startup. NOTE: this intentionally does NOT build the
    LivePortraitPipeline anymore — the Cropper inside it creates ONNX sessions
    (human_landmark_runner, face_analysis_wrapper) that bind to whatever execution
    provider is available at construction time. On ZeroGPU, no GPU is attached at
    cold start, so building here silently locks those sessions onto CPU forever,
    even once a GPU is attached to a later request. Pipeline construction is
    deferred to the first run_job() call instead, which only happens inside an
    @spaces.GPU-decorated request where a real GPU is actually attached.
    """
    print("[JOB_PROCESSOR] Skipping pipeline load at cold start (deferred to first GPU job).")



def _ensure_pipeline_loaded():
    global _pipeline
    if _pipeline is None:
        print("[JOB_PROCESSOR] Loading LivePortrait pipeline (inside GPU context)...")
        _pipeline = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())
        print("[JOB_PROCESSOR] Pipeline loaded and ready.")


def run_job(photo_path: str, driving_video_path: str, output_dir: str = "results") -> str:
    """Runs one LivePortrait generation. Returns path to the final .mp4."""
    _ensure_pipeline_loaded()
    print(f"[RUN_JOB] photo={photo_path} driving_video={driving_video_path}")
    os.makedirs(output_dir, exist_ok=True)

    args = ArgumentConfig(
        source=photo_path,
        driving=driving_video_path,
        output_dir=output_dir,
        scale=1.7,
    )
    wfp, wfp_concat = _pipeline.execute(args)
    print(f"[RUN_JOB] ✓ done: {wfp}")
    return wfp