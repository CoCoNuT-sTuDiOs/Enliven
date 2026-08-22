# ![Enliven Logo](Icon.png)

# Enliven: AI Avatar Animation with Real Expressions

**Transfer real human expressions and head movements onto any static avatar photo.**

Enliven is an open-source tool that takes three inputs and creates one powerful output:
- A static avatar photo (headshot)
- A driving video (real person with expressions and movement)  
- Optional: Audio (for future lip-sync features)

**Output:** An animated avatar video where the avatar mimics the real person's expressions, head movements, and blinks — creating natural, expressive talking heads.

---

## Key Features

- **Real Expression Transfer** — Not audio-guessed, but actual expressions from a real person
- **Natural Head Movement** — Full 3D head rotation, nods, and tilts
- **Commercial-Safe** — MIT/Apache-2.0 licensed (no research restrictions)
- **Bundled & Self-Contained** — LivePortrait included, weights downloaded on first run
- **Easy CLI** — One command to generate videos
- **Production-Ready** — Tested on real customer videos

---

## Examples

Below are three character demonstrations showing how Enliven transforms static photos into animated avatars.

### Character 1: Studio Portrait
**Input Photo:**  
[Avatar Photo 1]

**Driving Video (Input):**  
[Driving Video 1 - real person]

**Enliven Output:**  
[Animated Avatar 1 - avatar with real expressions]

---

### Character 2: Professional Headshot
**Input Photo:**  
[Avatar Photo 2]

**Driving Video (Input):**  
[Driving Video 2 - real person]

**Enliven Output:**  
[Animated Avatar 2 - avatar with real expressions]

---

### Character 3: Casual Portrait
**Input Photo:**  
[Avatar Photo 3]

**Driving Video (Input):**  
[Driving Video 3 - real person]

**Enliven Output:**  
[Animated Avatar 3 - avatar with real expressions]

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/CoCoNuT-sTuDiOs/Enliven.git
cd Enliven

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Generate an animated avatar video
python inference.py \
  --photo avatar_photo.jpg \
  --driving_video driving_video.mp4
```

**Output:** `enliven_output/result_final.mp4`

### With Face Enhancement (Optional)

```bash
python inference.py \
  --photo avatar_photo.jpg \
  --driving_video driving_video.mp4 \
  --enhance
```

This applies GFPGAN face enhancement for higher quality (optional, requires additional dependencies).

---

## 📋 Requirements

- Python 3.10+
- 8GB RAM minimum (16GB+ recommended)
- GPU recommended (NVIDIA CUDA). CPU-only will take 13-20+ hours per 10-second video(consider using GPU).

### Input Formats

- **Avatar Photo:** JPG, PNG (any resolution, but 512x512 recommended)
- **Driving Video:** MP4, WebM (any resolution and FPS)

### Output

- **Result Video:** MP4 format, same resolution as driving video, 25 FPS

---

## How It Works

Enliven uses a two-stage pipeline:

**Stage 1: LivePortrait (Expression & Pose Transfer)**
- Detects facial landmarks in the driving video using MediaPipe
- Extracts real expressions and head movements frame-by-frame
- Applies these movements to the avatar photo
- Outputs an animated video with authentic expressions

**Stage 2: GFPGAN (Optional Face Enhancement)**
- Enhances facial details in the final output
- Optional flag (skipped if not available)
- Improves image quality without changing expression

---

## 🛠️ Architecture

```
Enliven/
├── enliven/src/
│   ├── pipeline.py           # Main orchestration
│   ├── expression_transfer.py # LivePortrait integration
│   ├── enhance.py            # GFPGAN face enhancement
│   └── utils/                # Utilities
├── vendor/liveportrait/      # Bundled LivePortrait (MIT license)
├── inference.py              # CLI interface
├── requirements.txt          # Dependencies
└── README.md
```

### Key Components

**Pipeline:** Handles input validation, file copying, and stage orchestration.

**LivePortrait:** Performs the actual expression and pose transfer. Uses MediaPipe Face Mesh for face detection (commercial-safe alternative to InsightFace).

**GFPGAN:** Optional enhancement step. Gracefully skipped if unavailable.

---

## ⚙️ Advanced Usage

### Specify Output Location

```bash
python inference.py \
  --photo avatar.jpg \
  --driving_video video.mp4 \
  --output_dir my_outputs/
```

### With LivePortrait Directory Specification

```bash
python inference.py \
  --photo avatar.jpg \
  --driving_video video.mp4 \
  --liveportrait_dir /path/to/liveportrait/
```

---

## 📝 Tips for Best Results

1. **Avatar Photo:**
   - Clear, well-lit frontal headshot
   - 512x512 pixels or larger
   - Neutral or slight smile (easier to animate)

2. **Driving Video:**
   - Real person with clear facial expressions
   - Good lighting
   - 5-15 seconds duration (tested range)
   - Any FPS (25 FPS output standard)

3. **Processing:**
   - GPU: ~1 minute per 10-second video
   - CPU: ~10-15 minutes per 10-second video
   - Longer videos process proportionally

---

## 🐛 Troubleshooting

**"Face not detected"**
- Ensure the avatar photo is a clear frontal headshot
- Check lighting in both input images

**"LivePortrait not found"**
- Enliven bundles LivePortrait. Ensure full repo is cloned.
- Or specify `--liveportrait_dir` explicitly

**"GFPGAN unavailable"**
- Enhancement is optional. Pipeline continues without it.
- Run without `--enhance` flag if issues occur.

**Slow processing on CPU**
- This is normal. Consider GPU for faster results.
- 13-20+ hours  per 10-second video on CPU is expected(Consider using GPU).

---

## 📜 License

Enliven is released under the **MIT License**. See [LICENSE](LICENSE) for full details.

**Key Points:**
- Fully commercial-safe (no research restrictions)
- Can be used, modified, and redistributed
- Includes bundled LivePortrait (also MIT licensed)
- Uses MediaPipe (Apache-2.0 licensed) for face detection

---

## ⭐ If you find this product useful?
- Please leave a start for CoCoNuT-sTuDiOs as we would very much thoughtful 🙏🏾😁



## 🙏 Credits

**Enliven** is built on:
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) — Expression and pose transfer
- [MediaPipe](https://github.com/google/mediapipe) — Face detection (commercial-safe)
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face enhancement

---

## 🚀 What's Next?

- v1.1: Multi-person support
- v1.2: Audio-synchronized lip movement
- v1.3: Animal/character animations

---

## 💬 Questions or Issues?

Open an issue on GitHub or reach out to the maintainers.

**Built with ❤️ for creators, by CoCoNuT-sTuDiOs.**

---

*Last Updated: August 22, 2026*