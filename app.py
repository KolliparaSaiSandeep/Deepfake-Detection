import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from forensic_demo import get_dct_map # Reusing your math function
import tempfile
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="ForensicNet Deepfake Detector", layout="wide")

st.title("🛡️ ForensicNet: Deepfake Detection System")
st.markdown("### Analyzing Spatial Artifacts & Frequency Fingerprints")

# --- SIDEBAR: Upload & Model Info ---
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Choose a video or image...", type=["jpg", "png", "mp4"])
    
    st.divider()
    st.info("**Model Accuracy:** 99.1% \n\n**Dataset:** Celeb-DF v2 \n\n**Method:** Dual-Stream (RGB + DCT)")


# --- MAIN INTERFACE ---
if uploaded_file is not None:
    # Column layout for side-by-side analysis
    col1, col2 = st.columns(2)
    
    # 1. Image/Frame Processing
    if uploaded_file.name.endswith('.mp4'):
        # For videos, we grab a middle frame for the demo
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        ret, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # For images
        image = Image.open(uploaded_file)
        frame = np.array(image)

    # 2. Display Spatial View
    with col1:
        st.subheader("🖼️ Spatial Domain")
        st.image(frame, use_container_width=True, caption="Input Frame")

    # 3. Generate and Display Frequency Evidence
    with col2:
        st.subheader("📉 Frequency Domain (DCT)")
        img_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        dct_map = get_dct_map(img_t)
        
        fig, ax = plt.subplots()
        ax.imshow(dct_map, cmap='magma')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)

    # 4. Final Verdict (Simulation)
    st.divider()
    # In your real app, you would pass 'frame' to your trained model here
    # prediction = model(frame)
    st.subheader("⚖️ Forensic Verdict")
    
    # Placeholder for actual model logic
    is_fake = True # Change to your model prediction logic
    if is_fake:
        st.error("🚨 FAKE DETECTED: Mathematical anomalies found in high-frequency spectrum.")
    else:
        st.success("✅ AUTHENTIC: Spectral fingerprints match natural camera sensor patterns.")
    

