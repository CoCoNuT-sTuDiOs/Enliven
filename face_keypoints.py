"""
face_keypoints.py Extract face keypoints from driving video using DWPose


We pull nose/eyes/ears from 'bodies' (COCO-style ordering index 0=nose,
1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear), not 'faces', because the
68-point face landmark set does NOT include ear positions only 'bodies'
does. IMPORTANT: this ordering assumption is unverified against this
specific model's actual output — run the sanity check at the bottom of
this file on ONE frame before trusting a full video run, since being wrong
here silently produces garbage keypoints with no crash.

Also fixed: coordinates come back NORMALIZED (0-1 range), not pixel
values. We denormalize by frame width/height here so downstream
head_animate.py's _draw_keypoints (which expects real pixel ints) works
correctly.
"""
import cv2
import numpy as np
from controlnet_dwpose import DWposeDetector
from src.utils.device import get_device


class FaceKeypointExtractor:
    def __init__(
        self,
        model_det_path: str = "yolox_l.onnx",
        model_pose_path: str = "dw-ll_ucoco_384.onnx",
    ):
        """
        model_det_path / model_pose_path: local paths to the ONNX weights.
        These must be downloaded first (they are NOT bundled with the pip
        package) — see download_dwpose_models() below, run once per
        Kaggle session before instantiating this class.
        """
        self.device = get_device()
        onnx_device = "cuda" if self.device == "cuda" else "cpu"
        self.detector = DWposeDetector(
            model_det=model_det_path,
            model_pose=model_pose_path,
            device=onnx_device,
        )

    def extract(self, video_path: str) -> np.ndarray:
        """
        Extract face keypoints from video.
        Returns: (num_frames, 5, 2) array of PIXEL [x, y] coordinates for:
          0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        """
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self.detector(frame_rgb)

            bodies = pose_result.get("bodies")
            if bodies is not None and len(bodies) > 0:
                # first detected person, first 5 body keypoints (normalized)
                kpts_norm = bodies[0][0:5]  # shape (5, 2), values 0-1
                # denormalize to real pixel coordinates
                kpts_px = kpts_norm.copy().astype(np.float32)
                kpts_px[:, 0] *= w
                kpts_px[:, 1] *= h
                keypoints_list.append(kpts_px)
            else:
                # Fallback: use previous frame or zeros
                if keypoints_list:
                    keypoints_list.append(keypoints_list[-1].copy())
                else:
                    keypoints_list.append(np.zeros((5, 2), dtype=np.float32))

        cap.release()
        return np.array(keypoints_list, dtype=np.float32)


def download_dwpose_models(dest_dir: str = ".") -> None:
    
    import os
    import gdown

    pose_path = os.path.join(dest_dir, "dw-ll_ucoco_384.onnx")
    det_path = os.path.join(dest_dir, "yolox_l.onnx")

    if not os.path.exists(pose_path):
        gdown.download(
            "https://drive.google.com/uc?id=12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2",
            pose_path,
            quiet=False,
        )
    if not os.path.exists(det_path):
        gdown.download(
            "https://drive.google.com/uc?id=1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI",
            det_path,
            quiet=False,
        )


def sanity_check_one_frame(video_path: str) -> None:
    """
    RUN THIS FIRST, before extract() on a full video. Grabs frame 0 only,
    prints the raw keypoint values, and confirms the nose/eye/ear ordering
    assumption actually looks right for a real face BEFORE you burn GPU
    time processing 100+ frames on a wrong assumption.

    What to check in the printed output:
      - point 0 (nose) should be roughly center of the face, between eyes
      - points 1/2 (eyes) should be at eye height, left/right of nose
      - points 3/4 (ears) should be near the sides of the head
      If these look scrambled (e.g. point 0 near a shoulder), the ordering
      assumption in extract() is WRONG for this model version and needs
      adjusting.
    """
    extractor = FaceKeypointExtractor()
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read first frame of {video_path}")

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = extractor.detector(frame_rgb)

    print("[sanity_check] raw keys:", list(pose_result.keys()))
    bodies = pose_result.get("bodies")
    if bodies is None or len(bodies) == 0:
        print("[sanity_check] ✗ NO PERSON DETECTED in frame 0 — check input video")
        return

    kpts = bodies[0][0:5]
    labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    print(f"[sanity_check] frame size: {w}x{h}")
    for label, pt in zip(labels, kpts):
        px, py = pt[0] * w, pt[1] * h
        print(f"[sanity_check] {label:10s} normalized={tuple(pt)}  pixel=({px:.0f}, {py:.0f})")
