import streamlit as st
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import gdown
from PIL import Image
import os
import cv2
import numpy as np

# ==========================
# โหลดโมเดล ESRGAN (x4)
# ==========================
MODEL_PATH = "RealESRGAN_x4plus.pth"
if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading ESRGAN model...")
    gdown.download(
        "https://drive.google.com/uc?id=1R1b4Scb0v8lzH_QKUXtW-Cm-0MRRf3wU",
        MODEL_PATH,
        quiet=False
    )

device = "cpu"
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# ==========================
# Streamlit UI
# ==========================
st.title("🪄 Image Super Resolution (CPU - Streamlit Cloud Ready)")
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
