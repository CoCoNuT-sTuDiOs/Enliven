"""
inference.py CLI entry point for Enliven

Usage:
    python inference.py --photo photo.jpg --driving_video video.mp4 --audio audio.wav --output result.mp4
    python inference.py --photo photo.jpg --driving_video video.mp4 (no audio)
    python inference.py --photo photo.jpg --driving_video video.mp4 --enhance (with GFPGAN)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import generate

def main():
    parser = argparse.ArgumentParser(
        description="Enliven: Avatar Animation (LivePortrait + Wav2Lip + GFPGAN)"
    )
    parser.add_argument("--photo", required=True, help="Avatar headshot photo")
    parser.add_argument("--driving_video", required=True, help="Motion/expression source video")
    parser.add_argument("--audio", default=None, help="Optional audio clip (<=10 sec)")
    parser.add_argument("--output", default="result.mp4", help="Output video path")
    parser.add_argument("--enhance", action="store_true", help="Apply GFPGAN face enhancement")
    parser.add_argument("--output_dir", default="enliven_output", help="Temp output directory")
    parser.add_argument("--liveportrait_dir", default=None, help="LivePortrait repo path (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    try:
        result = generate(
            photo_path=args.photo,
            driving_video_path=args.driving_video,
            audio_path=args.audio,
            output_dir=args.output_dir,
            enhance=args.enhance,
            liveportrait_dir=args.liveportrait_dir
        )
        print(f"\n✓ COMPLETE: {result}")
        return 0
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
