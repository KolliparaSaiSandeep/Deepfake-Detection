import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.dual_stream import ForensicNet
from utils.preprocessing import get_dct
import numpy as np

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load the full dataset
    full_dataset = datasets.ImageFolder(r'C:\Users\u1196158\OneDrive - IQVIA\Documents\DeepfakeDetection\data\Train', transform=transform)
    
    # 2. Extract indices for each class (Real vs Fake)
    # ImageFolder usually assigns 0 to 'Fake' and 1 to 'Real' based on folder alphabetical order
    indices = np.arange(len(full_dataset))
    labels = np.array(full_dataset.targets)
    
    fake_indices = indices[labels == 0][:500]  # First 500 images from class 0
    real_indices = indices[labels == 1][:500]  # First 500 images from class 1
    
    # Combine them and create the Subset
    subset_indices = np.concatenate([fake_indices, real_indices])
    subset_dataset = Subset(full_dataset, subset_indices)

    # 3. Create Loader for the 1,000 images
    loader = DataLoader(subset_dataset, batch_size=8, shuffle=True) 

    model = ForensicNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    criterion = nn.BCELoss()

    print(f"🚀 Training on a small sample: {len(subset_dataset)} images (500 Fake, 500 Real)")
    
    for epoch in range(10): 
        total_loss = 0
        model.train() # Set model to training mode
        for imgs, labels in loader:
            dct_imgs = get_dct(imgs)
            
            # Forward pass
            outputs = model(imgs, dct_imgs).squeeze()
            
            # Match labels shape to output shape [batch_size]
            loss = criterion(outputs, labels.float())
            
            # Optimization steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/10 | Avg Loss: {total_loss/len(loader):.4f}")

    # 4. Save with the name expected by your app.py
    torch.save(model.state_dict(), 'model_weights.pth')
    print("✅ Training complete. Brain saved as model_weights.pth")

if __name__ == "__main__":
    train()