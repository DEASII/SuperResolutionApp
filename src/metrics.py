import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import time
import io

def calculate_metrics(original_file, upscaled_img):
    # original_file: file-like or path
    if hasattr(original_file, 'read'):
        original_file.seek(0)
        orig = Image.open(original_file).convert("RGB")
    else:
        orig = Image.open(original_file).convert("RGB")

    # upscaled_img is a PIL Image
    up = upscaled_img.convert("RGB")
    img1 = np.array(orig).astype(np.float32)
    img2 = np.array(up).astype(np.float32)

    # If sizes differ, center-crop to the same smallest size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = min(h1, h2)
    w = min(w1, w2)
    img1c = img1[0:h, 0:w]
    img2c = img2[0:h, 0:w]

    psnr_val = psnr(img1c, img2c, data_range=255)
    ssim_val = ssim(img1c.astype(np.uint8), img2c.astype(np.uint8), channel_axis=2)
    return psnr_val, ssim_val

def measure_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start
