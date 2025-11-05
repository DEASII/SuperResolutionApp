import torch
import torchvision.transforms as T
from PIL import Image
import io
from models.load_model import load_esrgan_model

def upscale_gpu(image_file):
    # image_file can be a file-like object or path
    model, device = load_esrgan_model()

    if hasattr(image_file, 'read'):
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")
    else:
        img = Image.open(image_file).convert("RGB")

    transform = T.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # output may be on device; move to cpu then to PIL
    output_img = T.ToPILImage()(output.squeeze().cpu())
    return output_img
