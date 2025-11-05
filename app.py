import sys
import torch
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2


def upscale_image(input_path, output_path, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='weights/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    print(f"📂 Loading image: {input_path}")
    img = Image.open(input_path).convert('RGB')

    print("🔄 Upscaling in progress...")
    sr_image = model.predict(img)

    sr_image.save(output_path)
    print(f"✅ Done! Saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python app.py <input_path> <output_path> [scale]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    upscale_image(input_path, output_path, scale)
