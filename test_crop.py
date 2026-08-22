import sys
from vendor.liveportrait.src.config.crop_config import CropConfig
from vendor.liveportrait.src.utils.cropper import Cropper

sys.path.insert(0, 'vendor/liveportrait/src')
import cv2

img = cv2.imread('test.jpeg')
if img is None:
    raise FileNotFoundError("test.jpeg not found run from Enliven_lab\\Enliven\\")

crop_cfg = CropConfig()
cropper = Cropper(crop_cfg=crop_cfg)

result = cropper.crop_source_image(img, crop_cfg)

print(f"Crop result keys: {result.keys() if isinstance(result, dict) else type(result)}")
if isinstance(result, dict) and 'img_crop' in result:
    print(f"img_crop shape: {result['img_crop'].shape}")
    cv2.imwrite('test_cropped_output.jpg', result['img_crop'])
    print("Saved test_cropped_output.jpg — check it visually for a correctly aligned, centered face crop")