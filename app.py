import os
import sys
import types

# --- Shim: gradio_client's schema introspection chokes on a bare boolean
# JSON-schema value. Patch get_type to handle it. (Confirmed needed here too —
# not Checkbox-specific, triggers during Gradio's own API schema introspection.)
from gradio_client import utils as gradio_client_utils

_original_get_type = gradio_client_utils.get_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "boolean" if schema else "None"
    return _original_get_type(schema)


gradio_client_utils.get_type = _safe_get_type

import shutil
import subprocess
import traceback
import requests
import gradio as gr
import spaces

from job_processor import init_models, run_job
MAX_DRIVING_VIDEO_SECONDS = 10



KAGGLE_DATASET = "coconutdummy/enliven-liveportrait-weights-v2"
WEIGHTS_DIR = os.path.join("vendor", "liveportrait", "pretrained_weights")


def download_checkpoints():
    if os.path.isdir(WEIGHTS_DIR) and os.listdir(WEIGHTS_DIR):
        print(f"[STARTUP] ✓ LivePortrait weights already present at {WEIGHTS_DIR}")
        return

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set as HF Space secrets "
            "(from the same Kaggle account/legacy API key used in Part 2 setup)."
        )

    print(f"[STARTUP] Downloading weights from Kaggle dataset: {KAGGLE_DATASET}")
    import kagglehub
    downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"[STARTUP] Downloaded to: {downloaded_path}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for item in os.listdir(downloaded_path):
        src = os.path.join(downloaded_path, item)
        dst = os.path.join(WEIGHTS_DIR, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"[STARTUP] ✓ weights placed at {WEIGHTS_DIR}")


# ---- module import time: models onto GPU (ZeroGPU pattern, see job_processor) ----

download_checkpoints()

print("[STARTUP] Loading models into memory...")
init_models()
print("[STARTUP] Models loaded. Space is ready.")


# ---- helpers ----

def _resolve_input(value: str, dest_path: str) -> str:
    """Accepts either a URL (API call) or a local filepath (Gradio UI upload)
    and returns a local filepath either way."""
    if isinstance(value, str) and value.startswith("http"):
        print(f"[JOB] downloading input: {value} -> {dest_path}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(value, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
        return dest_path
    # already a local path from Gradio's own upload handling
    return value


def _get_video_duration_seconds(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


# ---- ZeroGPU wrapper ----

@spaces.GPU(duration=60)
def _run_job_gpu(photo_path, driving_video_path):
    print(f"[GPU] job started on GPU worker: photo={photo_path} driving_video={driving_video_path}")
    result_path = run_job(photo_path=photo_path, driving_video_path=driving_video_path)
    print(f"[GPU] ✓ job finished on GPU worker, result: {result_path}")
    return result_path


def generate(photo, driving_video):
    print("-" * 60)
    print(f"[JOB] received: photo={photo} driving_video={driving_video}")
    try:
        os.makedirs("job_temp", exist_ok=True)
        photo_path = _resolve_input(photo, "job_temp/photo.jpg")
        driving_video_path = _resolve_input(driving_video, "job_temp/driving.mp4")

        duration = _get_video_duration_seconds(driving_video_path)
        if duration > MAX_DRIVING_VIDEO_SECONDS:
            raise gr.Error(
                f"Driving video is {duration:.1f}s — max allowed is {MAX_DRIVING_VIDEO_SECONDS}s."
            )

        result_path = _run_job_gpu(photo_path, driving_video_path)

        print(f"[JOB] ✓ complete, returning: {result_path}")
        print("-" * 60)
        return result_path

    except Exception as e:
        print(f"[JOB] ✗ FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("-" * 60)
        raise


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="filepath", label="photo"),
        gr.Video(label="driving_video"),
    ],
    outputs=gr.Video(label="result"),
    api_name="generate",
)


if __name__ == "__main__":
    demo.queue()
    demo.launch()