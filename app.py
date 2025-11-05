import streamlit as st
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import os
import urllib.request
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ==========================
# โหลดโมเดล ESRGAN (x4)
# ==========================
MODEL_PATH = "RealESRGAN_x4plus.pth"
if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading ESRGAN model from GitHub...")
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    urllib.request.urlretrieve(url, MODEL_PATH)
    st.success("✅ Download complete!")

device = "cpu"

# สร้างโมเดลโครงสร้าง SRVGGNetCompact (ตาม RealESRGAN)
model = SRVGGNetCompact(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_conv=32,
    upscale=4,
    act_type='prelu'
)

# โหลด weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==========================
# Streamlit UI
# ==========================
st.title("🪄 Image Super Resolution (CPU - ESRGAN)")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original", use_container_width=True)

    if st.button("Enhance Image"):
        with st.spinner("Processing..."):
            img_tensor = to_tensor(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
            out_img = to_pil_image(output.squeeze(0).clamp(0, 1))

        st.image(out_img, caption="Enhanced Image (x4)", use_container_width=True)
        out_img.save("output.png")
        st.download_button("📥 Download Result", data=open("output.png", "rb"), file_name="output.png")
