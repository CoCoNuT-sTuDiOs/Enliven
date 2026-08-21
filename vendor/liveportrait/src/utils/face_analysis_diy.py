"""Face detection using OpenCV DNN (commercial-safe)"""
import numpy as np
import cv2
from .rprint import rlog as log
from .timer import Timer

class Face:
    def __init__(self, bbox, kps, det_score=0.9):
        self.bbox = bbox
        self.kps = kps
        self.det_score = det_score

def sort_by_direction(faces, direction='large-small', face_center=None):
    if len(faces) <= 0:
        return faces
    if direction == 'large-small':
        return sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    return faces

class FaceAnalysisDIY:
    """OpenCV DNN face detection"""
    
    def __init__(self, name='opencv', root='~/.opencv', allowed_modules=None, **kwargs):
        self.timer = Timer()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def get(self, img_bgr, **kwargs):
        max_num = kwargs.get('max_face_num', 0)
        direction = kwargs.get('direction', 'large-small')
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(gray, 1.3, 5)
        
        faces = []
        for (x, y, w, h) in detections:
            bbox = np.array([x, y, x+w, y+h])
            kps = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h], [x+w//2, y+h//2]])
            face = Face(bbox=bbox, kps=kps, det_score=0.9)
            faces.append(face)
        
        if max_num > 0:
            faces = faces[:max_num]
        
        return sort_by_direction(faces, direction)
    
    def warmup(self):
        self.timer.tic()
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.get(img)
        log(f"FaceAnalysisDIY warmup: {self.timer.toc():.3f}s")
