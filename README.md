# SuperResolutionApp (Template)
**Image Super-Resolution (CPU vs GPU - MPS)**

This is a ready-to-run template for macOS (M1/M2 or Intel) that demonstrates:
- CPU bicubic upscaling
- GPU upscaling using PyTorch (MPS/Metal backend) with a pretrained ESRGAN model (as a template)
- Streamlit UI to upload an image and compare results
- Metrics: PSNR and SSIM
- Simple performance chart

## How to run (recommended)
1. Create & activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Verify MPS availability:
   ```python
   import torch
   print(torch.backends.mps.is_available())
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Open the URL shown in the terminal (usually http://localhost:8501).

## Notes
- The ESRGAN model is loaded via `torch.hub` as a convenience. Depending on your environment, you may need to adjust model loading (or download weights manually).
- The GPU code uses MPS when available; otherwise it falls back to CPU.
- This template includes a simple generated sample image in `data/input/sample_low.png` to try immediately.
