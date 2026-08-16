# Enliven
### by DuMmY-AI · a CoCoNuT-sTuDiOs product

<div align="center">

<b>TL;DR: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; avatar photo 🙎‍♂️ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; driving video 🎥 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; audio 🎤 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a living, moving, speaking avatar 🎞</b>

</div>

<br>

Enliven brings a single still photo to life — full-body motion, natural facial expression, and audio-matched lip sync — by chaining together four specialist open-source models rather than training one giant model from scratch. Give it a photo of a person standing front-facing, a silent video of someone gesturing/walking/dancing, and an audio clip, and Enliven returns a video of your photo performing that motion while speaking that audio.

Built solo, on free-tier infrastructure, as a proof-of-capability project under the DuMmY-AI brand.

## Status

🚧 **Actively in early development.** Pipeline stages are being built and validated one at a time (see Roadmap below). Not yet packaged for one-command install — check back as this README is updated stage by stage.

## How it works — the pipeline

Rather than one model trying to do everything, Enliven uses four purpose-built stages, each handling exactly one job:

1. **DWPose** — extracts a motion skeleton (body, hands, head) from the driving video
2. **MimicMotion** — animates the avatar photo to follow that skeleton, producing full-body motion with naturally correlated head movement
3. **LivePortrait** — transfers the driving video's facial expression (not audio-predicted) onto the avatar's face, so emotion stays consistent with the body motion
4. **Wav2Lip** — performs the final mouth-only correction, syncing lip movement precisely to the provided audio track

Each stage is independently swappable and independently testable — see `src/` for the per-stage code.

## Why this architecture

Most AI coding/avatar tools either train something from scratch (expensive, slow, out of reach without serious compute budget) or wrap a single black-box model (limited, hard to customize). Enliven takes a third path: assemble proven, purpose-built open-source models — the same way a car is built from an engine, tires, and a chassis sourced from specialists, not forged from raw metal by one team. The engineering work here is in the orchestration: making sure no two stages fight each other (e.g. head pose from one model conflicting with head pose from another), keeping the pipeline device-agnostic (CPU or GPU), and structuring it so it's genuinely runnable by someone else who clones this repo.

## Installation

*(To be filled in as the pipeline is built and tested — will mirror a standard `git clone` → `pip install -r requirements.txt` → download models flow.)*

```bash
git clone https://github.com/CoCoNuT-sTuDiOs/Enliven.git
cd Enliven
pip install -r requirements.txt
```

## Download Models

*(Model download instructions will be added here once weights are finalized — likely via a script similar to SadTalker's `download_models.sh`, pulling from Hugging Face Hub.)*

## Quick Start

*(CLI usage instructions will be added as each pipeline stage is completed and tested.)*

```bash
# planned usage — subject to change as the pipeline is built
python inference.py --avatar <photo.png> \
                     --driving_video <video.mp4> \
                     --audio <audio.wav> \
                     --output <result.mp4>
```

## Device Support

Enliven is built to run on either CPU or GPU, same principle as SadTalker — one codebase, device selected automatically (`src/utils/device.py`). GPU is strongly recommended for the MimicMotion stage specifically, as video diffusion inference on CPU alone can be very slow. Other stages (DWPose, LivePortrait, Wav2Lip) are lighter and more CPU-tolerant.

## Roadmap

- [ ] DWPose extraction validated standalone
- [ ] MimicMotion body animation validated standalone
- [ ] LivePortrait expression transfer validated standalone
- [ ] Wav2Lip lip sync validated standalone
- [ ] Full 4-stage pipeline integrated
- [ ] Public demo (Hugging Face ZeroGPU Space)

## Credits

Enliven does not train any model from scratch. Full credit and thanks to the original research teams and open-source maintainers behind the models this project orchestrates:

- **DWPose** — pose estimation
- **MimicMotion** — pose-guided human image animation
- **LivePortrait** — portrait expression transfer
- **Wav2Lip** — audio-driven lip synchronization

This project exists because of their work being open and freely available — Enliven's contribution is the pipeline and orchestration layer connecting them, not the underlying model research.

## Disclaimer

This is an independent project by CoCoNuT-sTuDiOs / DuMmY-AI. It is not affiliated with or endorsed by the original authors of DWPose, MimicMotion, LivePortrait, or Wav2Lip.

```
1. Please read and comply with the open-source licenses of the underlying models this project uses.
2. This code is intended for legitimate creative, educational, and product-development use.
3. Do not use this project to create content that impersonates real individuals without consent,
   or for any deceptive, harassing, or otherwise harmful purpose.
4. Any legal liabilities arising from misuse of this code are the responsibility of the user, not
   CoCoNuT-sTuDiOs, DuMmY-AI, or the original model authors.
```

---

<div align="center">
Built by <a href="https://github.com/CoCoNuT-sTuDiOs">CoCoNuT-sTuDiOs</a> · part of the <b>DuMmY-AI</b> product line
</div>
