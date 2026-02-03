import streamlit as st
import cv2
import tempfile
from PIL import Image
from models.dual_stream import ForensicNet
import torch
import torch.nn as nn
import numpy as np
from utils.preprocessing import get_dct_map
# --- PAGE UI ---
st.set_page_config(page_title="ForensicNet", layout="wide")
st.title("🛡️ ForensicNet: Dual-Stream Deepfake Detector")

# --- MODEL LOADING (The Linkage) ---
@st.cache_resource
def load_forensic_model():
    model = ForensicNet()
    try:
        model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    except Exception as e:
        # This will print the EXACT error (e.g., FileNotFoundError or Size Mismatch)
        st.error(f"Error loading weights: {e}")
    return model

model = load_forensic_model()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "png", "mp4"])

# --- PROCESSING ---
if uploaded_file:
    col1, col2 = st.columns(2)
    
    # Handle Video vs Image
    if uploaded_file.name.endswith('.mp4'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        _, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = np.array(Image.open(uploaded_file))

    # Display Original
    with col1:
        st.subheader("🖼️ Spatial Domain")
        st.image(frame, use_container_width=True)

    # Prepare Tensors
    # (B, C, H, W) and normalized to [0, 1]
    input_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    dct_tensor = get_dct_map(input_tensor)

    # Display DCT Map
    with col2:
        st.subheader("📉 Frequency Domain")
        # Visualizing the first (and only) batch item
        dct_viz = dct_tensor[0, 0].numpy()
        st.image(dct_viz, use_container_width=True, clamp=True, caption="Spectral Artifacts")

    # --- INFERENCE ---
    st.divider()
    with torch.no_grad():
        prediction = model(input_tensor, dct_tensor).item()
    
    # Final Result
    if prediction < 0.5:
        st.error(f"🚨 FAKE DETECTED (Score: {prediction:.2f})")
    else:
        st.success(f"✅ AUTHENTIC (Score: {prediction:.2f})")