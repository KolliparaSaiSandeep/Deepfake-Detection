import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from utils.preprocessing import get_dct # Using your project's function

def visualize_dct(real_path, fake_path):
    # Load images
    real_img = cv2.imread(real_path)
    fake_img = cv2.imread(fake_path)
    
    # Convert to RGB for plotting
    real_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    fake_rgb = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)

    # Process through your DCT function
    # (Simulating the tensor conversion your model does)
    def process_for_plot(img):
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        dct_map = get_dct(img_t.unsqueeze(0)).squeeze().numpy()
        return dct_map

    real_dct = process_for_plot(real_rgb)
    fake_dct = process_for_plot(fake_rgb)

    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Real Image (Spatial)")
    plt.imshow(real_rgb)
    
    plt.subplot(2, 2, 2)
    plt.title("Real DCT (Frequency)")
    plt.imshow(real_dct, cmap='magma') # Magma helps see the intensity
    
    plt.subplot(2, 2, 3)
    plt.title("Fake Image (Spatial)")
    plt.imshow(fake_rgb)
    
    plt.subplot(2, 2, 4)
    plt.title("Fake DCT (Frequency)")
    plt.imshow(fake_dct, cmap='magma')
    
    plt.tight_layout()
    plt.savefig('dct_comparison.png')
    print("✅ Comparison saved as 'dct_comparison.png'. Open it to see the difference!")

if __name__ == "__main__":
    # Replace these with actual paths from your data folder
    visualize_dct('data/train/real/id0_0000.mp4.jpg', 'data/train/fake/id0_id1_0000.mp4.jpg')
