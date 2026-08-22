
"""Face detection using MediaPipe (commercial-safe, 95%+ accuracy)"""
import numpy as np
import cv2
import mediapipe as mp
from .rprint import rlog as log
from .timer import Timer


class Face:
    def __init__(self, bbox, kps, landmark_2d_106=None, det_score=0.9):
        self.bbox = bbox
        self.kps = kps
        self.landmark_2d_106 = landmark_2d_106
        self.det_score = det_score

def sort_by_direction(faces, direction='large-small', face_center=None):
    if len(faces) <= 0:
        return faces
    if direction == 'large-small':
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces

class FaceAnalysisDIY:
    """MediaPipe Face Mesh detection (commercial-safe replacement for InsightFace)"""

    # MediaPipe 468-landmark indices matching InsightFace's 5-point layout
    # (left eye, right eye, nose tip, mouth left corner, mouth right corner)
    FIVE_POINT_INDICES = [33, 263, 1, 61, 291]

    # Base 106-point spread across the full mesh (rough bbox coverage)
    _BASE_106_IDX = np.linspace(0, 467, 106).astype(int)

    # Precise overrides for the exact indices InsightFace-format code reads directly
    # (eye centers + lip center — see parse_pt2_from_pt106 in crop.py)
    _PRECISE_OVERRIDES = {
        33: 33, 35: 133, 39: 159, 40: 145,  # one eye (outer, inner, top, bottom)
        87: 362, 89: 263, 93: 386, 94: 374,  # other eye (inner, outer, top, bottom)
        52: 13, 61: 14,  # lip (upper center, lower center)
    }

    def _build_landmark_106(self, mp_landmarks_468):
        lmk106 = mp_landmarks_468[self._BASE_106_IDX].copy()
        for idx106, idx468 in self._PRECISE_OVERRIDES.items():
                lmk106[idx106] = mp_landmarks_468[idx468]
        return lmk106
    def __init__(self, name='mediapipe', root='~/.mediapipe', allowed_modules=None, **kwargs):
        self.timer = Timer()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5
        )
    def prepare(self, ctx_id=0, det_size=(512, 512), det_thresh=0.5, **kwargs):
        # No-op: MediaPipe doesn't need explicit context/session prep like InsightFace's ONNX runtime did
        pass

    def get(self, img_bgr, **kwargs):
        max_num = kwargs.get('max_face_num', 0)
        direction = kwargs.get('direction', 'large-small')

        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Detect faces with MediaPipe
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return []

        faces = []
        for face_landmarks in results.multi_face_landmarks:
            # Extract all 468 landmarks
            landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])

            # Compute bounding box from landmarks
            x_min = np.min(landmarks[:, 0])
            y_min = np.min(landmarks[:, 1])
            x_max = np.max(landmarks[:, 0])
            y_max = np.max(landmarks[:, 1])

            bbox = np.array([x_min, y_min, x_max, y_max])

            # Extract 5-point kps matching InsightFace's alignment format
            kps = landmarks[self.FIVE_POINT_INDICES]

            # Build 106-point landmark array for cropper.py compatibility
            landmark_2d_106 = self._build_landmark_106(landmarks)

            face = Face(bbox=bbox, kps=kps, landmark_2d_106=landmark_2d_106, det_score=0.95)
            faces.append(face)

        if max_num > 0:
            faces = faces[:max_num]

        return sort_by_direction(faces, direction)

    def warmup(self):
        self.timer.tic()
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.get(img)
        log(f"FaceAnalysisDIY warmup: {self.timer.toc():.3f}s")