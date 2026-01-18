# 🕵️‍♂️ Advanced Deepfake Forensic Framework

A high-fidelity digital forensic tool designed to detect facial manipulation using **Dual-Stream Neural Analysis**.

## 🚀 Technical Highlights
* **Spatial-Frequency Fusion:** Combines EfficientNet RGB feature extraction with Discrete Cosine Transform (DCT) frequency analysis to identify hidden AI "fingerprints."
* **Explainable AI (XAI):** Integrated **Grad-CAM** heatmaps to visualize the specific forgery markers detected by the model.
* **Production Pipeline:** Modularized architecture for scalable training and real-time video inference.

## 📊 Performance Metrics
- **Dataset:** Celeb-DF (v2)
- **F1-Score:** [Insert your score]%
- **Detection Lag:** ~150ms per frame on NVIDIA T4 GPU.