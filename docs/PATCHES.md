# Enliven — Third-Party Repo Patches

These tool repos (MimicMotion, LivePortrait, Wav2Lip) are cloned fresh into `/kaggle/working` each session and are NOT part of the Enliven git repo (excluded via `.gitignore`). This means any patch applied to their code is **lost every time the Kaggle session resets** and the repo is re-cloned.

This file exists so patches can be reapplied in seconds instead of rediscovering each error from scratch. Apply these immediately after cloning each repo, before running anything.

---

## MimicMotion — `mimicmotion/utils/loader.py`

**Problem:** `TypeError: _safe_globals.__init__() takes 2 positional arguments but 5 were given`
Caused by an old `torch.serialization.safe_globals(*allowed_modules)` call (unpacking a list into multiple positional args) meeting a newer PyTorch API that expects a single list argument instead.

**Patch (run once, right after cloning MimicMotion, before first inference run):**
```bash
sed -i "s/with torch.serialization.safe_globals(\*allowed_modules):/with torch.serialization.safe_globals(allowed_modules):/" mimicmotion/utils/loader.py
```

**Verify:**
```bash
grep -n "safe_globals" mimicmotion/utils/loader.py
```
Should show `with torch.serialization.safe_globals(allowed_modules):` — no `*`.

---

## MimicMotion — Dependency install order

Do NOT use the pinned versions from `environment.yaml` as-is. Instead:

```bash
pip install decord==0.6.0 einops omegaconf --quiet
pip install "transformers>=4.32.1" --quiet
pip install -U huggingface_hub diffusers --quiet
pip install onnxruntime-gpu --quiet
```

Do not pin `diffusers==0.27.0` or `huggingface_hub==0.23.4` — these old pins conflict with each other and with newer `transformers`. Let both resolve to current compatible versions instead.

---

## MimicMotion — Gated SVD model authentication

`stabilityai/stable-video-diffusion-img2vid-xt-1-1` is gated. One-time account setup (license acceptance + HF token) persists across sessions, but the auth call must be redone each session:

```python
import os
from kaggle_secrets import UserSecretsClient
hf_token = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
```

**Do NOT use `huggingface_hub.login(token=...)`** — this repeatedly threw `ImportError: cannot import name 'DeviceCodeError'` in this environment, likely due to an internally inconsistent `huggingface_hub` install left over from other version changes. Setting the environment variables directly works and bypasses this broken code path.

**Note (license):** this SVD checkpoint is licensed **non-commercial / research use only**. Fine for current proof-of-capability phase. Must be resolved (commercial license from Stability AI, or a different base model) before any commercial use of Enliven.

---

## MimicMotion — GPU memory (config, not a code bug)

Default `configs/test.yaml` (72 frames, 576 resolution) will OOM on a single T4 (16GB). Reduce before running:

```bash
sed -i 's/num_frames: 72/num_frames: 16/' configs/test.yaml
sed -i 's/resolution: 576/resolution: 384/' configs/test.yaml
sed -i 's/frames_overlap: 6/frames_overlap: 4/' configs/test.yaml
```

Adjust frame count based on target clip length (`frames ÷ fps = seconds`). Product cap is 10 seconds max.

---

## MimicMotion — Missing PyAV

Needed for `torchvision`'s video save step (fails silently late — after the full generation loop already ran, wasting GPU time). Install BEFORE running inference, not after a failure:

```bash
pip install av --quiet
```

---

## LivePortrait — Weight download command

The README's `huggingface-cli` command is outdated — that tool is deprecated. Use `hf` instead:

```bash
pip install -U "huggingface_hub[cli]" --quiet
hf download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" --exclude "README.md" --exclude "docs/*"
```

**Important — `--exclude` syntax changed too.** Each pattern needs its own `--exclude` flag (as shown above). Passing multiple space-separated values after one `--exclude` gets silently misinterpreted as explicit filenames to download, which breaks the exclude logic entirely and can result in only downloading `README.md`.

**Note on symlinks (resolved, no longer an issue):** an earlier version of this workflow used a `--local-dir-use-symlinks False` flag to avoid HF's local caching from creating symlinks instead of real files (which broke zipping the folder for a Kaggle Dataset — the zip only captured tiny pointer files, not actual weight data). That flag no longer exists in the current `hf` CLI — current versions write real files directly by default, no flag needed. **Always verify with `du -sh pretrained_weights/` and check a `.pth` file with `ls -la` (no `->` symlink arrow) before trusting a download completed correctly** — do not just check file count.

---

## LivePortrait — License

Check LivePortrait's own repo license terms before any commercial use — not yet fully audited as of this document's writing. Flag for follow-up.

---

## Wav2Lip — Dependency install

Ignore the repo's own `requirements.txt` (pins `torch==1.1.0`, `numpy==1.17.1` — from 2019, incompatible with current environments). Instead:

```bash
pip install librosa opencv-python numba tqdm --quiet
```
Do not touch the already-installed `torch` — reuse whatever's already present from MimicMotion/LivePortrait setup.

---

## Wav2Lip — Weights (no official HF mirror — Google Drive only in README)

Official README only links Google Drive, which is unreliable for scripted downloads. Use this verified community mirror instead (confirmed matching SHA256 checksum against the original):

```bash
mkdir -p checkpoints
wget -q "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth" -O checkpoints/wav2lip_gan.pth
wget -q "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip.pth" -O checkpoints/wav2lip.pth
```
Verify: both files should be ~416MB each.

**License:** models trained on LRS2 dataset — **commercial use strictly prohibited** per Wav2Lip's own README. Same category as the SVD note above — must be resolved before any commercial use.

---

## Wav2Lip — `audio.py` librosa API mismatch

**Problem:** `TypeError: mel() takes 0 positional arguments but 0 positional arguments (and 3 keyword-only arguments) were given`
Caused by an old `librosa.filters.mel(hp.sample_rate, hp.n_fft, ...)` positional call meeting a newer librosa version that requires `sr=` and `n_fft=` as keyword arguments.

**Patch (run once, right after cloning Wav2Lip, before first inference run):**
```bash
sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,/' audio.py
```

**Verify:**
```bash
sed -n '99,101p' audio.py
```
Should show `sr=hp.sample_rate, n_fft=hp.n_fft` as keywords.

---

## General lessons (see also main HANDOVER.md)

- Model weights persist via Kaggle Datasets, NOT git, NOT notebook-output-attach (that only gives back rendered notebook artifacts, not `/kaggle/working` files).
- Always verify a Dataset upload with actual byte size (via the Kaggle web UI's "Data Explorer" panel, most reliable) — not just file count from the CLI, which has misled us more than once (paginated/zipped listings can look deceptively small).
- Cloned tool repos (MimicMotion, LivePortrait, Wav2Lip) are excluded via `.gitignore` and are NOT tracked by the Enliven repo's own git — do not attempt to `git add`/`commit` inside them, that would try to push into the original authors' upstream repos.
- Kaggle sessions have an idle timeout that wipes `/kaggle/working` — save early, save often, to GitHub (code) and Kaggle Datasets (weights), not just Kaggle "Quick Save" versions alone.
