import cv2
import numpy as np
from PIL import Image
import io

def upscale_cpu(image_file, scale=4):
    # image_file can be a file-like object or path
    if hasattr(image_file, 'read'):
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")
    else:
        img = Image.open(image_file).convert("RGB")

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    upscaled = cv2.resize(img_cv, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    up_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    return Image.fromarray(up_rgb)
