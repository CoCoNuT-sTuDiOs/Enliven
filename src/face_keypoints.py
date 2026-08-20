"""
face_keypoints.py Extract face keypoints from driving video using DWPose
"""
import cv2
import numpy as np
from dwpose import DWposeDetector
from src.utils.device import get_device

class FaceKeypointExtractor:
    def __init__(self):
        self.device = get_device()
        self.detector = DWposeDetector(
            det_model="yolox_l",
            pose_model="dw-ll_ucoco_384",
            device=self.device
        )
    
    def extract(self, video_path: str) -> np.ndarray:
        """
        Extract face keypoints from video.
        Returns: (num_frames, 5, 2) array of [x, y] coordinates for:
          0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        """
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # DWPose returns full body, extract face only
            dwpose_results = self.detector(frame)
            
            # Extract face keypoints (indices 0-4 in DWPose output)
            if dwpose_results is not None and len(dwpose_results) > 0:
                kpts = dwpose_results[0].keypoints[0:5]  # face only
                keypoints_list.append(kpts)
            else:
                # Fallback: use previous frame or zeros
                if keypoints_list:
                    keypoints_list.append(keypoints_list[-1].copy())
                else:
                    keypoints_list.append(np.zeros((5, 2)))
        
        cap.release()
        return np.array(keypoints_list, dtype=np.float32)
