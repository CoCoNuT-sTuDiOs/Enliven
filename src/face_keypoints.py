"""
face_keypoints.py Extract face keypoints from driving video using DWPose

FIX/PATCHED (Aug 2026, v2): sanity_check_one_frame() revealed the previous ordering
assumption (bodies[0][0:5] = nose/eyes/ears) was wrong on two levels:
  1. `bodies` is a dict {'candidate', 'subset', 'score'}, not a plain array.
  2. `candidate` holds all 18 COCO body keypoints in standard OpenPose order,
     where face points are at the END of the array, not the start:
       0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist, 5=LShoulder,
       6=LElbow, 7=LWrist, 8=RHip, 9=RKnee, 10=RAnkle, 11=LHip, 12=LKnee,
       13=LAnkle, 14=REye, 15=LEye, 16=REar, 17=LEar
     `subset` maps body-part-index -> row index in `candidate` for each
     detected person (-1 = not detected for that person).

We now pull face points via subset[person][BODY_PART_INDEX] -> candidate[idx],
using the confirmed indices 0 (nose), 14 (right eye), 15 (left eye),
16 (right ear), 17 (left ear). Missing points (-1) fall back to the
previous frame's value, or zeros if it's the first frame.

Coordinates in `candidate` are normalized (0-1); we denormalize by frame
width/height here so downstream head_animate.py's _draw_keypoints (which
expects real pixel ints) works correctly.
"""
import cv2
import numpy as np
from controlnet_dwpose import DWposeDetector
from src.utils.device import get_device

# Confirmed COCO-18 body part indices for face points (see module docstring)
NOSE, R_EYE, L_EYE, R_EAR, L_EAR = 0, 14, 15, 16, 17
FACE_PART_INDICES = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]  # order matches labels below
FACE_LABELS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


def _extract_face_points(bodies: dict, person_idx: int = 0) -> np.ndarray:
    """
    Pull the 5 face keypoints (normalized coords) for one detected person
    from the raw DWPose 'bodies' dict, using the subset->candidate mapping.
    Returns (5, 2) array; missing points come back as [-1, -1] so the
    caller can decide how to handle them (fallback, skip, etc.).
    """
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    if subset is None or len(subset) <= person_idx:
        return np.full((5, 2), -1.0, dtype=np.float32)

    person_row = subset[person_idx]
    pts = np.full((5, 2), -1.0, dtype=np.float32)
    for out_i, part_i in enumerate(FACE_PART_INDICES):
        cand_idx = int(person_row[part_i])
        if cand_idx == -1:
            continue
        pts[out_i] = candidate[cand_idx]
    return pts


class FaceKeypointExtractor:
    def __init__(
        self,
        model_det_path: str = "yolox_l.onnx",
        model_pose_path: str = "dw-ll_ucoco_384.onnx",
    ):
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
        Missing detections in a frame fall back to the previous frame's
        values (or zeros if it's the very first frame).
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

            if bodies is not None and bodies.get("subset") is not None and len(bodies["subset"]) > 0:
                kpts_norm = _extract_face_points(bodies, person_idx=0)  # (5, 2)
                kpts_px = kpts_norm.copy()
                missing = kpts_px[:, 0] < 0  # -1 sentinel marks missing points
                kpts_px[:, 0] *= w
                kpts_px[:, 1] *= h

                if missing.any() and keypoints_list:
                    # fill missing points from previous frame
                    kpts_px[missing] = keypoints_list[-1][missing]

                keypoints_list.append(kpts_px)
            else:
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
    Grabs frame 0, extracts face points via the confirmed subset/candidate
    mapping, and prints them for visual sanity checking.

    What to check in the printed output:
      - nose should be roughly centered, between the eyes
      - left_eye/right_eye should be at similar height, flanking the nose
      - left_ear/right_ear should be near the frame edges, roughly eye height
      - any point printed as MISSING means DWPose didn't detect it in this
        frame — check the input video framing if that happens for nose/eyes
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
    if bodies is None or bodies.get("subset") is None or len(bodies["subset"]) == 0:
        print("[sanity_check] ✗ NO PERSON DETECTED in frame 0 — check input video")
        return

    kpts = _extract_face_points(bodies, person_idx=0)
    print(f"[sanity_check] frame size: {w}x{h}")
    for label, pt in zip(FACE_LABELS, kpts):
        if pt[0] < 0:
            print(f"[sanity_check] {label:10s} MISSING (not detected)")
            continue
        px, py = pt[0] * w, pt[1] * h
        print(f"[sanity_check] {label:10s} normalized=({pt[0]:.4f}, {pt[1]:.4f})  pixel=({px:.0f}, {py:.0f})")
