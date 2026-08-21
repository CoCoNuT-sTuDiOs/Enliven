"""
pipeline.py two-stage avatar animation pipelin

Stages:
1. LivePortrait: photo + driving_video → expression/pose transfer
2. GFPGAN (optional): enhance face quality frame-by-frame
"""
import os
import shutil
from src.expression_transfer import transfer_expression
from src.enhance import FaceEnhancer


def generate(
    photo_path: str,
    driving_video_path: str,
    audio_path: str = None,
    output_dir: str = "enliven_output",
    enhance: bool = False,
    liveportrait_dir: str = None
) -> str:
    """Generate avatar video: photo + driving_video."""
    
    print("[ENLIVEN] Starting pipeline...")
    print(f"  photo: {photo_path}")
    print(f"  video: {driving_video_path}")
    print(f"  enhance: {enhance}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Auto-detect LivePortrait
        if liveportrait_dir is None:
            home = os.path.expanduser("~")
            possible_paths = [
                os.path.join(home, "LivePortrait"),
                "/kaggle/working/LivePortrait",
                "./LivePortrait",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    liveportrait_dir = path
                    break
            if liveportrait_dir is None:
                raise RuntimeError("LivePortrait not found")
        
        print(f"[ENLIVEN] LivePortrait: {liveportrait_dir}")
        
        # Copy inputs to writable location
        print("[ENLIVEN] Copying inputs to writable location...")
        photo_work = os.path.join(output_dir, "source.jpg")
        video_work = os.path.join(output_dir, "driving.mp4")
        shutil.copy(photo_path, photo_work)
        shutil.copy(driving_video_path, video_work)
        print(f"[ENLIVEN] ✓ Inputs copied")
        
        # Stage 1: LivePortrait
        print("[ENLIVEN] Stage 1: LivePortrait...")
        liveportrait_output = transfer_expression(
            source_path=photo_work,
            driving_video_path=video_work,
            liveportrait_dir=liveportrait_dir,
            output_dir=os.path.abspath(output_dir),
            flag_use_half_precision=True
        )
        print(f"[ENLIVEN] ✓ Stage 1 done: {liveportrait_output}")
        
        # Stage 2: Optional GFPGAN enhancement (standalone, doesn't break if unavailable)
        print("[ENLIVEN] Stage 2: GFPGAN enhancement (optional)...")
        final_output = os.path.join(os.path.abspath(output_dir), "result_final.mp4")
        
        if enhance:
            try:
                enhancer = FaceEnhancer()
                if enhancer.available:
                    enhancer.enhance_video(liveportrait_output, final_output)
                    print(f"[ENLIVEN] ✓ Stage 2 done (enhanced): {final_output}")
                else:
                    print("[ENLIVEN] GFPGAN unavailable, using unenhanced output")
                    shutil.copy(liveportrait_output, final_output)
            except Exception as e:
                print(f"[ENLIVEN] GFPGAN failed: {e}, using unenhanced output")
                shutil.copy(liveportrait_output, final_output)
        else:
            shutil.copy(liveportrait_output, final_output)
            print(f"[ENLIVEN] ✓ Stage 2 skipped (enhance=False)")
        
        print(f"[ENLIVEN] ✓✓✓ SUCCESS: {final_output}")
        return final_output
        
    except Exception as e:
        print(f"[ENLIVEN] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
