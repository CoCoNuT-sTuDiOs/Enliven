"""
model_setup.py

Applies every known patch from docs/PATCHES.md automatically, and handles
per-session setup (git identity, Hugging Face auth) that doesn't persist
across Kaggle session resets.

Usage (run once per fresh Kaggle session, after cloning the tool repos):

    from src.utils.model_setup import (
        setup_git_identity,
        setup_hf_auth,
        patch_mimicmotion,
        patch_wav2lip,
        reduce_mimicmotion_config,
    )

    setup_git_identity()
    setup_hf_auth()
    patch_mimicmotion("/kaggle/working/MimicMotion")
    patch_wav2lip("/kaggle/working/Wav2Lip")
    reduce_mimicmotion_config("/kaggle/working/MimicMotion/configs/test.yaml",
                               num_frames=16, resolution=384, frames_overlap=4)
"""

import os
import subprocess


def _run(cmd, cwd=None):
    """Run a shell command, raise clearly on failure."""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[model_setup] WARNING: command failed: {cmd}")
        print(result.stderr)
    return result


def setup_git_identity(email=None, name=None):
    """
    Git identity does not persist across fresh Kaggle sessions.
    Call this before any git commit.
    """
    from kaggle_secrets import UserSecretsClient

    if email is None or name is None:
        # Fall back to values stored as Kaggle Secrets if not passed explicitly
        try:
            client = UserSecretsClient()
            email = email or client.get_secret("GIT_EMAIL")
            name = name or client.get_secret("GIT_NAME")
        except Exception:
            raise ValueError(
                "No email/name provided and GIT_EMAIL/GIT_NAME secrets not found. "
                "Pass setup_git_identity(email=..., name=...) explicitly."
            )

    _run(f'git config --global user.email "{email}"')
    _run(f'git config --global user.name "{name}"')
    print(f"[model_setup] Git identity set: {name} <{email}>")


def setup_hf_auth():
    """
    Sets Hugging Face auth token as environment variables.
    Deliberately does NOT use huggingface_hub.login() — that function has
    repeatedly thrown ImportError in this environment (see PATCHES.md).
    """
    from kaggle_secrets import UserSecretsClient

    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("[model_setup] Hugging Face auth token set via environment variables.")


def setup_kaggle_cli_auth():
    """
    Sets up the Kaggle CLI's own credentials (needed for `kaggle datasets ...`
    commands, e.g. attaching/creating Datasets from within a notebook).
    """
    import json
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
    os.makedirs("/root/.config/kaggle", exist_ok=True)
    with open("/root/.config/kaggle/kaggle.json", "w") as f:
        f.write(token)
    os.chmod("/root/.config/kaggle/kaggle.json", 0o600)
    print("[model_setup] Kaggle CLI credentials set.")


def patch_mimicmotion(mimicmotion_dir):
    """
    Fixes the torch.serialization.safe_globals() TypeError caused by an old
    unpacked-args call meeting a newer PyTorch API. See PATCHES.md.
    Safe to call multiple times (idempotent — sed only matches the old pattern).
    """
    loader_path = os.path.join(mimicmotion_dir, "mimicmotion", "utils", "loader.py")
    if not os.path.exists(loader_path):
        print(f"[model_setup] WARNING: {loader_path} not found, skipping patch.")
        return

    old = r"with torch.serialization.safe_globals(\*allowed_modules):"
    new = r"with torch.serialization.safe_globals(allowed_modules):"
    _run(f"sed -i 's/{old}/{new}/' {loader_path}")

    # Verify
    result = _run(f"grep -n 'safe_globals' {loader_path}")
    if "*allowed_modules" in result.stdout:
        print("[model_setup] WARNING: MimicMotion patch may not have applied correctly.")
    else:
        print("[model_setup] MimicMotion safe_globals patch applied.")


def patch_wav2lip(wav2lip_dir):
    """
    Fixes the librosa.filters.mel() TypeError caused by old positional-args
    call meeting a newer librosa API that requires sr= and n_fft= as keywords.
    See PATCHES.md. Safe to call multiple times.
    """
    audio_path = os.path.join(wav2lip_dir, "audio.py")
    if not os.path.exists(audio_path):
        print(f"[model_setup] WARNING: {audio_path} not found, skipping patch.")
        return

    old = "librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,"
    new = "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"
    _run(f"sed -i 's/{old}/{new}/' {audio_path}")

    result = _run(f"grep -n 'librosa.filters.mel' {audio_path}")
    if "sr=hp.sample_rate" in result.stdout:
        print("[model_setup] Wav2Lip librosa patch applied.")
    else:
        print("[model_setup] WARNING: Wav2Lip patch may not have applied correctly.")


def reduce_mimicmotion_config(config_path, num_frames=16, resolution=384, frames_overlap=4):
    """
    Reduces MimicMotion's test.yaml frame count/resolution to fit in a single
    T4's memory. Default config (72 frames, 576 resolution) will OOM.
    """
    if not os.path.exists(config_path):
        print(f"[model_setup] WARNING: {config_path} not found, skipping config reduction.")
        return

    _run(f"sed -i 's/num_frames: 72/num_frames: {num_frames}/' {config_path}")
    _run(f"sed -i 's/resolution: 576/resolution: {resolution}/' {config_path}")
    _run(f"sed -i 's/frames_overlap: 6/frames_overlap: {frames_overlap}/' {config_path}")
    print(f"[model_setup] MimicMotion config reduced: "
          f"num_frames={num_frames}, resolution={resolution}, frames_overlap={frames_overlap}")


def install_mimicmotion_deps():
    """
    Installs MimicMotion's dependencies using modern, mutually-compatible
    versions instead of the repo's old pinned environment.yaml. See PATCHES.md.
    """
    _run("pip install decord==0.6.0 einops omegaconf --quiet")
    _run('pip install "transformers>=4.32.1" --quiet')
    _run("pip install -U huggingface_hub diffusers --quiet")
    _run("pip install onnxruntime-gpu av --quiet")
    print("[model_setup] MimicMotion dependencies installed.")


def install_wav2lip_deps():
    """
    Installs Wav2Lip's dependencies using current versions instead of the
    repo's ancient pinned requirements.txt (torch==1.1.0, numpy==1.17.1).
    Deliberately does not touch torch, which should already be installed.
    """
    _run("pip install librosa opencv-python numba tqdm --quiet")
    print("[model_setup] Wav2Lip dependencies installed.")


def fix_numpy_binary_incompatibility():
    """
    Fixes a recurring "numpy.dtype size changed, may indicate binary
    incompatibility" / "No module named 'numpy.rec'" error hit repeatedly
    across this project. Root cause: installing MimicMotion's, LivePortrait's,
    and Wav2Lip's dependencies (each independently) can leave numpy's
    compiled C extensions in a mixed/inconsistent state, even when
    `pip show numpy` reports a single correct version number.

    MUST be called LAST, after every other repo's install_*_deps() call —
    not before. Any subsequent pip install of another repo's requirements
    can re-trigger the corruption, so calling this too early doesn't help.

    IMPORTANT: after calling this, the Kaggle kernel MUST be restarted
    before numpy (or anything importing it, e.g. torchvision, scipy) is
    used again. Numpy's C extensions cannot be cleanly reloaded within
    the same running Python process — clearing sys.modules is NOT enough.
    This function cannot restart the kernel itself; the caller must do
    Run -> Restart Session manually afterward.
    """
    _run("pip uninstall numpy scipy torchvision -y --quiet")
    _run("pip install --no-cache-dir --force-reinstall numpy==1.26.4 scipy torchvision --quiet")
    print("[model_setup] numpy/scipy/torchvision reinstalled. "
          "RESTART THE KERNEL NOW before running anything that imports numpy.")


def clear_import_caches():
    """
    Clears not just sys.modules but also Python's path-based import
    caches. Needed after a fresh git clone in the same running kernel —
    clearing sys.modules alone has repeatedly been insufficient in this
    project, leaving Python unable to find freshly-cloned packages (e.g.
    "ModuleNotFoundError: No module named 'src.utils'" even when the file
    genuinely exists on disk).
    """
    import sys
    import importlib

    for mod in list(sys.modules):
        if mod.startswith("src") or mod == "pipeline":
            del sys.modules[mod]
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()
    print("[model_setup] Import caches cleared.")

def patch_mimicmotion_video_writer(mimicmotion_dir):
    from pathlib import Path
    utils_path = Path(mimicmotion_dir) / "mimicmotion" / "utils" / "utils.py"
    original = utils_path.read_text()

    old_block = (
        "import logging\n"
        "from pathlib import Path\n"
        "from torchvision.io import write_video\n"
        "logger = logging.getLogger(__name__)\n"
        "def save_to_mp4(frames, save_path, fps=7):\n"
        "    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)\n"
        "    Path(save_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "    write_video(save_path, frames, fps=fps)"
    )

    new_block = (
        "import logging\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import imageio\n"
        "logger = logging.getLogger(__name__)\n"
        "def save_to_mp4(frames, save_path, fps=7):\n"
        "    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)\n"
        "    Path(save_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "    frames = frames.detach().cpu().numpy()\n"
        "    if frames.dtype != np.uint8:\n"
        "        frames = frames.astype(np.uint8)\n"
        "    writer = imageio.get_writer(save_path, fps=fps, codec=\"libx264\", format=\"FFMPEG\")\n"
        "    for frame in frames:\n"
        "        writer.append_data(frame)\n"
        "    writer.close()"
    )

    if old_block not in original:
        raise RuntimeError(
            "patch_mimicmotion_video_writer: expected original save_to_mp4 block "
            "not found — file may have changed upstream, check before patching."
        )

    utils_path.write_text(original.replace(old_block, new_block))
    print("Patched MimicMotion save_to_mp4() to use imageio instead of torchvision.io.write_video")

def patch_mimicmotion_video_writer(mimicmotion_dir):
    from pathlib import Path
    utils_path = Path(mimicmotion_dir) / "mimicmotion" / "utils" / "utils.py"
    original = utils_path.read_text()

    old_block = (
        "import logging\n"
        "from pathlib import Path\n"
        "from torchvision.io import write_video\n"
        "logger = logging.getLogger(__name__)\n"
        "def save_to_mp4(frames, save_path, fps=7):\n"
        "    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)\n"
        "    Path(save_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "    write_video(save_path, frames, fps=fps)"
    )

    new_block = (
        "import logging\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import imageio\n"
        "logger = logging.getLogger(__name__)\n"
        "def save_to_mp4(frames, save_path, fps=7):\n"
        "    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)\n"
        "    Path(save_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "    frames = frames.detach().cpu().numpy()\n"
        "    if frames.dtype != np.uint8:\n"
        "        frames = frames.astype(np.uint8)\n"
        "    writer = imageio.get_writer(save_path, fps=fps, codec=\"libx264\", format=\"FFMPEG\")\n"
        "    for frame in frames:\n"
        "        writer.append_data(frame)\n"
        "    writer.close()"
    )

    if old_block not in original:
        raise RuntimeError(
            "patch_mimicmotion_video_writer: expected original save_to_mp4 block "
            "not found — file may have changed upstream, check before patching."
        )

    utils_path.write_text(original.replace(old_block, new_block))
    print("Patched MimicMotion save_to_mp4() to use imageio instead of torchvision.io.write_video")

def patch_mimicmotion_video_writer(mimicmotion_dir):
    from pathlib import Path
    utils_path = Path(mimicmotion_dir) / "mimicmotion" / "utils" / "utils.py"
    original = utils_path.read_text()

    old_block = (
        "import logging\n"
        "from pathlib import Path\n"
        "from torchvision.io import write_video\n"
        "logger = logging.getLogger(__name__)\n"
        "def save_to_mp4(frames, save_path, fps=7):\n"
        "    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)\n"
        "    Path(save_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "    write_video(save_path, frames, fps=fps)"
    )

    new_block = (
        "import logging\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import imageio\n"
        "logger = logging.getLogger(__name__)\n"
        "def save_to_mp4(frames, save_path, fps=7):\n"
        "    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)\n"
        "    Path(save_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "    frames = frames.detach().cpu().numpy()\n"
        "    if frames.dtype != np.uint8:\n"
        "        frames = frames.astype(np.uint8)\n"
        "    writer = imageio.get_writer(save_path, fps=fps, codec=\"libx264\", format=\"FFMPEG\")\n"
        "    for frame in frames:\n"
        "        writer.append_data(frame)\n"
        "    writer.close()"
    )

    if old_block not in original:
        raise RuntimeError(
            "patch_mimicmotion_video_writer: expected original save_to_mp4 block "
            "not found — file may have changed upstream, check before patching."
        )

    utils_path.write_text(original.replace(old_block, new_block))
    print("Patched MimicMotion save_to_mp4() to use imageio instead of torchvision.io.write_video")


def patch_mimicmotion_dwpose_path(mimicmotion_dir):
    """Fix DWPose's eager-loaded relative ONNX paths to be cwd-independent."""
    import os
    path = os.path.join(mimicmotion_dir, "mimicmotion/dwpose/dwpose_detector.py")
    with open(path, "r") as f:
        content = f.read()
    if "_MIMICMOTION_ROOT" in content:
        return  # already patched
    content = content.replace(
        "import os",
        "import os\n_MIMICMOTION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
        1
    )
    content = content.replace(
        'model_det="models/DWPose/yolox_l.onnx",',
        'model_det=os.path.join(_MIMICMOTION_ROOT, "models/DWPose/yolox_l.onnx"),'
    )
    content = content.replace(
        'model_pose="models/DWPose/dw-ll_ucoco_384.onnx",',
        'model_pose=os.path.join(_MIMICMOTION_ROOT, "models/DWPose/dw-ll_ucoco_384.onnx"),'
    )
    with open(path, "w") as f:
        f.write(content)
