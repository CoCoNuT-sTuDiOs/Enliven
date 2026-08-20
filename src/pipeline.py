"""
pipeline.py — Three-stage avatar animation pipeline

Stages:
1. LivePortrait: photo + driving_video → expression/pose transfer
2. Wav2Lip (optional): add audio sync to mouth  
3. GFPGAN (optional): enhance face quality frame-by-frame
"""
import os
from src.expression_transfer import transfer_expression
from src.lip_sync import sync_lips
from src.enhance import FaceEnhancer

def generate(
    photo_path: str,
    driving_video_path: str,
    audio_path: str = None,
    output_dir: str = "enliven_output",
    enhance: bool = False,
    liveportrait_dir: str = None
) -> str:
    """
    Generate avatar video: photo + driving_video + optional audio.
    
    Inputs:
    - photo_path: Avatar headshot
    - driving_video_path: Motion/expression source
    - audio_path: Optional audio (<=10 sec)
    - output_dir: Results directory
    - enhance: Apply GFPGAN face enhancement
    - liveportrait_dir: LivePortrait repo path
    
    Returns: Final MP4 path
    """
    print("[ENLIVEN] Starting pipeline...")
    print(f"  photo: {photo_path}")
    print(f"  video: {driving_video_path}")
    print(f"  audio: {audio_path if audio_path else '(none)'}")
    print(f"  enhance: {enhance}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Auto-detect LivePortrait
        if liveportrait_dir is None:
            import os.path
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
                raise RuntimeError("LivePortrait not found. Specify --liveportrait_dir.")
        
        print(f"[ENLIVEN] LivePortrait: {liveportrait_dir}")
        
        # Stage 1: LivePortrait
        print("[ENLIVEN] Stage 1: LivePortrait (expression + pose)...")
        liveportrait_output = transfer_expression(
            source_path=photo_path,
            driving_video_path=driving_video_path,
            liveportrait_dir=liveportrait_dir,
            output_dir=output_dir,
            flag_use_half_precision=True
        )
        print(f"[ENLIVEN] ✓ Stage 1 done: {liveportrait_output}")
        
        current_video = liveportrait_output
        
        # Stage 2: Wav2Lip (if audio)
        if audio_path:
            print("[ENLIVEN] Stage 2: Wav2Lip (mouth sync)...")
            # Auto-detect Wav2Lip
            wav2lip_dir = None
            home = os.path.expanduser("~")
            possible_paths = [
                os.path.join(home, "Wav2Lip"),
                "/kaggle/working/Wav2Lip",
                "./Wav2Lip",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    wav2lip_dir = path
                    break
            if wav2lip_dir is None:
                raise RuntimeError("Wav2Lip not found. Clone it or specify path.")
            
            wav2lip_output = sync_lips(
                video_path=current_video,
                audio_path=audio_path,
                wav2lip_dir=wav2lip_dir
            )
            print(f"[ENLIVEN] ✓ Stage 2 done: {wav2lip_output}")
            current_video = wav2lip_output
        else:
            print("[ENLIVEN] Stage 2: Skipping Wav2Lip (no audio)")
        
        # Stage 3: GFPGAN enhancement
        print("[ENLIVEN] Stage 3: GFPGAN (face enhancement)...")
        if enhance:
            enhancer = FaceEnhancer()
            final_output = os.path.join(output_dir, "result_final.mp4")
            enhancer.enhance_video(current_video, final_output)
            print(f"[ENLIVEN] ✓ Stage 3 done (enhanced): {final_output}")
        else:
            final_output = current_video
            print(f"[ENLIVEN] ✓ Stage 3 skipped (enhance=False)")
        
        print(f"[ENLIVEN] ✓✓✓ SUCCESS: {final_output}")
        return final_output
        
    except Exception as e:
        print(f"[ENLIVEN] ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
