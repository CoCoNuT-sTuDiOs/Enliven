"""

Stages:
1. LivePortrait: photo + driving_video → expression/pose transfer
2. Wav2Lip (optional): add audio sync to mouth  
3. GFPGAN (optional): enhance face quality frame-by-frame
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
    """Generate avatar video: photo + driving_video + optional audio."""
    
    print("[ENLIVEN] Starting pipeline...")
    print(f"  photo: {photo_path}")
    print(f"  video: {driving_video_path}")
    print(f"  audio: {audio_path if audio_path else '(none)'}")
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
                raise RuntimeError("LivePortrait not found. Specify --liveportrait_dir.")
        
        print(f"[ENLIVEN] LivePortrait: {liveportrait_dir}")
        
        # Copy inputs to writable location (Kaggle datasets are read-only)
        print("[ENLIVEN] Copying inputs to writable location...")
        photo_work = os.path.join(output_dir, "source.jpg")
        video_work = os.path.join(output_dir, "driving.mp4")
        shutil.copy(photo_path, photo_work)
        shutil.copy(driving_video_path, video_work)
        if audio_path:
            audio_work = os.path.join(output_dir, "audio.wav")
            # Convert to WAV if needed
            if audio_path.lower().endswith(".mp3"):
                from pydub import AudioSegment
                print("[ENLIVEN] Converting MP3 to WAV...")
                audio = AudioSegment.from_mp3(audio_path)
                audio.export(audio_work, format="wav")
                print("[ENLIVEN] ✓ Converted to WAV")
            else:
                shutil.copy(audio_path, audio_work)
            audio_path = audio_work
        print(f"[ENLIVEN] ✓ Inputs copied to {output_dir}")
        
        # Stage 1: LivePortrait
        print("[ENLIVEN] Stage 1: LivePortrait (expression + pose)...")
        liveportrait_output = transfer_expression(
            source_path=photo_work,
            driving_video_path=video_work,
            liveportrait_dir=liveportrait_dir,
            output_dir=os.path.abspath(output_dir),
            flag_use_half_precision=True
        )
        print(f"[ENLIVEN] ✓ Stage 1 done: {liveportrait_output}")
        
        current_video = liveportrait_output
        
        # Stage 2: Reserved for future audio sync research
        print("[ENLIVEN] Stage 2: Audio sync (reserved for future implementation)")
        
        # Stage 3: GFPGAN enhancement
        print("[ENLIVEN] Stage 3: GFPGAN (face enhancement)...")
        final_output = os.path.join(os.path.abspath(output_dir), "result_final.mp4")
        if enhance:
            enhancer = FaceEnhancer()
            enhancer.enhance_video(current_video, final_output)
            print(f"[ENLIVEN] ✓ Stage 3 done (enhanced): {final_output}")
        else:
            # Copy to final location without enhancement
            import shutil
            shutil.copy(current_video, final_output)
            print(f"[ENLIVEN] ✓ Stage 3 skipped (enhance=False), saved to: {final_output}")
        
        print(f"[ENLIVEN] ✓✓✓ SUCCESS: {final_output}")
        return final_output
        
    except Exception as e:
        print(f"[ENLIVEN] ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
