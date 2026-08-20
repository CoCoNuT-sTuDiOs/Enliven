"""
inference.py CLI entry point for Enliven

Usage:
    python inference.py --photo photo.jpg --driving_video video.mp4 --audio audio.wav --output result.mp4
"""
import argparse
import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import generate

def main():
    parser = argparse.ArgumentParser(description="Enliven: Avatar Animation Pipeline")
    parser.add_argument("--photo", required=True, help="Path to source photo")
    parser.add_argument("--driving_video", required=True, help="Path to driving video")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output", default="result.mp4", help="Output video path")
    parser.add_argument("--enhance", action="store_true", help="Apply GFPGAN face enhancement")
    parser.add_argument("--output_dir", default="enliven_output", help="Temp output directory")
    parser.add_argument(
        "--test_fps", type=int, default=None,
        help="Downsample Stage 2 animation to this fps (e.g. 8-10) to save GPU quota "
             "during smoke tests. Omit for full quality (uses driving video's native fps)."
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Diffusion steps per frame in Stage 2. Lower (10-12) = faster/cheaper, "
             "worse quality. Use low values for quick smoke tests, raise once pipeline "
             "is confirmed working end to end."
    )
    parser.add_argument(
        "--strength", type=float, default=0.4,
        help="Img2img strength for Stage 2 (0-1). Low = stays closer to source photo "
             "identity. Raise slightly (toward 0.5) only if motion looks too frozen."
    )
    parser.add_argument(
        "--conditioning_scale", type=float, default=1.0,
        help="ControlNet pose conditioning scale for Stage 2."
    )

    args = parser.parse_args()

    try:
        result = generate(
            photo_path=args.photo,
            driving_video_path=args.driving_video,
            audio_path=args.audio,
            output_dir=args.output_dir,
            enhance=args.enhance,
            test_fps=args.test_fps,
            num_inference_steps=args.steps,
            strength=args.strength,
            conditioning_scale=args.conditioning_scale,
        )
        print(f"\n✓ SUCCESS: {result}")
        return 0
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
