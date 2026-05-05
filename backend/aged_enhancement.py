import cv2
import numpy as np

def enhance_fingerprint(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img.astype(np.float32)

    # 1. Illumination correction
    background = cv2.GaussianBlur(img, (31,31), 0)
    corrected = cv2.subtract(img, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    corrected = corrected.astype(np.uint8)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    contrast = clahe.apply(corrected)

    # 3. Unsharp masking
    blur = cv2.GaussianBlur(contrast, (5,5), 0)
    sharpened = cv2.addWeighted(contrast, 1.7, blur, -0.7, 0)

    # 4. Final normalize
    final = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    return final
