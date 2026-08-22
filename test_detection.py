import sys
sys.path.insert(0, 'vendor/liveportrait/src')
import cv2
from vendor.liveportrait.src.utils.face_analysis_diy import FaceAnalysisDIY
img = cv2.imread('test.jpeg')
if img is None:
    raise FileNotFoundError("test.jpeg not found run this script from Enliven_lab\\Enliven\\")

analyzer = FaceAnalysisDIY()
faces = analyzer.get(img)

print(f"Faces detected: {len(faces)}")
for i, face in enumerate(faces):
    print(f"Face {i}: bbox={face.bbox}, kps_shape={face.kps.shape}, "
          f"landmark_2d_106_shape={face.landmark_2d_106.shape}, det_score={face.det_score}")