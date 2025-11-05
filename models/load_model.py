import torch
from realesrgan import RealESRGAN
from PIL import Image

def load_esrgan_model():
    # ใช้ MPS (GPU ของ Mac) ถ้ามี
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # โหลดโมเดลจากไฟล์ที่เราดาวน์โหลดไว้
    model_path = "models/esrgan/RealESRGAN_x4plus.pth"

    # โหลดโมเดล Real-ESRGAN
    model = RealESRGAN(device, scale=4)
    model.load_weights(model_path, download=False)

    return model, device

def upscale_image(model, image: Image.Image):
    return model.predict(image)
