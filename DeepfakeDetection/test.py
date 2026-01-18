import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.dual_stream import ForensicNet
from utils.preprocessing import get_dct
from sklearn.metrics import classification_report, confusion_matrix

def test():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder('data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ForensicNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    y_true, y_pred = [], []

    print(f"🧪 Testing on {len(test_dataset)} images...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            dct_imgs = get_dct(imgs)
            output = model(imgs, dct_imgs)
            
            # Threshold at 0.5: >0.5 is Fake, <0.5 is Real
            prediction = 1 if output.item() > 0.5 else 0
            y_pred.append(prediction)
            y_true.append(labels.item())

    print("\n--- NEW RESUME METRICS ---")
    print(classification_report(y_true, y_pred, target_names=['REAL', 'FAKE']))

if __name__ == "__main__":
    test()