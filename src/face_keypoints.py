"""
face_keypoints.py Extract face/body keypoints from driving video using DWPose

v3 (Aug 2026): Added extract_full_skeleton() alongside the existing 5-point
extract(). Stage 2 testing showed ControlNet pose-following was too weak
with only 5 isolated dots (confirmed on frame 45, a 277px keypoint
displacement produced almost no visible head movement in output) —
fusing/lllyasviel's openpose ControlNet was trained on full 18-point
OpenPose skeletons with limb lines, not sparse dots, so we now preserve
all 18 body points per frame for that purpose.

extract() (5-point, pixel coords) is left as-is/unused by Stage 2 now,
kept in case anything else still depends on it.
"""
import cv2
import numpy as np
from controlnet_dwpose import DWposeDetector
from controlnet_aux.open_pose.body import Keypoint
from src.utils.device import get_device

# Confirmed COCO-18 body part indices for face points (see v2 fix)
NOSE, R_EYE, L_EYE, R_EAR, L_EAR = 0, 14, 15, 16, 17
FACE_PART_INDICES = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
FACE_LABELS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


def _extract_face_points(bodies: dict, person_idx: int = 0) -> np.ndarray:
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


def _extract_full_skeleton(bodies: dict, person_idx: int = 0) -> list:
    """
    Build a length-18 list of Keypoint|None (normalized coords), in the
    exact format controlnet_aux.open_pose.util.draw_bodypose expects.
    """
    frame_kpts = [None] * 18
    subset = bodies.get("subset")
    if subset is None or len(subset) <= person_idx:
        return frame_kpts

    candidate = bodies["candidate"]
    person_row = subset[person_idx]
    score = bodies.get("score")

    for part_i in range(18):
        cand_idx = int(person_row[part_i])
        if cand_idx == -1:
            continue
        x, y = candidate[cand_idx]
        s = float(score[person_idx][part_i]) if score is not None else 1.0
        frame_kpts[part_i] = Keypoint(x=float(x), y=float(y), score=s, id=part_i)
    return frame_kpts


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
        (Legacy, 5-point) Returns (num_frames, 5, 2) PIXEL coords:
        nose, left_eye, right_eye, left_ear, right_ear. Not used by
        Stage 2 anymore — see extract_full_skeleton().
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
                kpts_norm = _extract_face_points(bodies, person_idx=0)
                kpts_px = kpts_norm.copy()
                missing = kpts_px[:, 0] < 0
                kpts_px[:, 0] *= w
                kpts_px[:, 1] *= h
                if missing.any() and keypoints_list:
                    kpts_px[missing] = keypoints_list[-1][missing]
                keypoints_list.append(kpts_px)
            else:
                if keypoints_list:
                    keypoints_list.append(keypoints_list[-1].copy())
                else:
                    keypoints_list.append(np.zeros((5, 2), dtype=np.float32))
        cap.release()
        return np.array(keypoints_list, dtype=np.float32)

    def extract_full_skeleton(self, video_path: str) -> list:
        """
        Returns a list (len = num_frames) of length-18 lists of
        Keypoint|None, normalized 0-1 coords, ready to pass directly
        into controlnet_aux's draw_bodypose(). This is what Stage 2's
        _draw_keypoints() now consumes.
        """
        cap = cv2.VideoCapture(video_path)
        skeletons = []
        prev = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self.detector(frame_rgb)
            bodies = pose_result.get("bodies")
            if bodies is not None and bodies.get("subset") is not None and len(bodies["subset"]) > 0:
                frame_kpts = _extract_full_skeleton(bodies, person_idx=0)
                prev = frame_kpts
            else:
                frame_kpts = prev if prev is not None else [None] * 18
            skeletons.append(frame_kpts)
        cap.release()
        return skeletons


def download_dwpose_models(dest_dir: str = ".") -> None:
    import os
    import gdown
    pose_path = os.path.join(dest_dir, "dw-ll_ucoco_384.onnx")
    det_path = os.path.join(dest_dir, "yolox_l.onnx")
    if not os.path.exists(pose_path):
        gdown.download(
            "https://drive.google.com/uc?id=12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2",
            pose_path, quiet=False,
        )
    if not os.path.exists(det_path):
        gdown.download(
            "https://drive.google.com/uc?id=1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI",
            det_path, quiet=False,
        )


def sanity_check_one_frame(video_path: str) -> None:
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
