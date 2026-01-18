import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.dual_stream import ForensicNet
from utils.preprocessing import get_dct

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder('data/train', transform=transform)
    # Increase batch_size to 4 or 8 if your CPU allows, it helps the model see variety
    loader = DataLoader(dataset, batch_size=4, shuffle=True) 

    model = ForensicNet()
    # LOWER Learning Rate (0.0001) helps the model find subtle AI artifacts
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    criterion = nn.BCELoss()

    print(f"🚀 Training on {len(dataset)} images...")
    
    # INCREASE Epochs to 10. This is the 'Study Time'
    for epoch in range(10): 
        total_loss = 0
        for imgs, labels in loader:
            dct_imgs = get_dct(imgs)
            outputs = model(imgs, dct_imgs).squeeze()
            
            # Binary Cross Entropy loss
            loss = criterion(outputs, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/10 | Avg Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), 'best_model.pth')
    print("✅ Training complete. Brain saved as best_model.pth")

if __name__ == "__main__":
    train()