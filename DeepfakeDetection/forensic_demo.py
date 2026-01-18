import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import os

# --- CORE MATH FUNCTION ---
def get_dct_map(img_tensor):
    """ Converts an RGB image tensor to a log-scaled DCT frequency map. """
    # Convert to grayscale using Luminance formula
    gray = 0.299*img_tensor[0,:,:] + 0.587*img_tensor[1,:,:] + 0.114*img_tensor[2,:,:]
    
    # Perform 2D Discrete Cosine Transform
    def apply_dct(a):
        return dct(dct(a.numpy(), axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Log scaling makes the high-frequency 'grid' visible to the eye
    dct_map = np.log(np.abs(apply_dct(gray)) + 1e-6)
    return dct_map

# --- THE DEMO GENERATOR ---
def generate_evidence(video_path, output_name="evidence_report.png"):
    # 1. Load the video and grab the middle frame
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2) 
    success, frame = cap.read()
    cap.release()

    if not success:
        print("❌ Could not read video file.")
        return

    # 2. Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    
    # 3. Get Frequency Map
    dct_map = get_dct_map(img_t)

    # 4. Create the "Forensic Report" Visual
    plt.figure(figsize=(10, 5), facecolor='#f0f0f0')
    
    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    plt.title("SPATIAL DOMAIN (Human View)", fontsize=12, pad=10)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # Using 'magma' or 'inferno' colormap makes artifacts pop!
    plt.imshow(dct_map, cmap='magma') 
    plt.title("FREQUENCY DOMAIN (AI Signature)", fontsize=12, pad=10)
    plt.axis('off')

    plt.suptitle(f"Forensic Evidence: {os.path.basename(video_path)}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_name)
    print(f"✅ Evidence Report generated: {output_name}")

if __name__ == "__main__":
    # Test it on one of your videos
    test_video = "data/test/fake/id0_id1_0000.mp4.jpg" # Update with your actual video path
    generate_evidence(test_video)