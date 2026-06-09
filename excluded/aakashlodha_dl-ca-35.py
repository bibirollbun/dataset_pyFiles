import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import os

# Dataset path
data_dir = Path("/kaggle/input/musical-instrumemts-sound-classification/Melspectogram_split")
train_dir = data_dir / "train"
val_dir   = data_dir / "val"
test_dir  = data_dir / "test"

print("Train classes:", os.listdir(train_dir))



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])



train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data   = datasets.ImageFolder(root=val_dir, transform=transform)
test_data  = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")
print("Classes:", train_data.classes)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_data.classes)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



num_epochs = 2

for epoch in range(num_epochs): model.train()
train_loss = 0
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    preds = model(xb)
    loss = criterion(preds, yb)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}")


model.eval() 

correct = 0
total = 0

with torch.no_grad(): 
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)     
        _, predicted = torch.max(outputs, 1) 
        total += yb.size(0)               
        correct += (predicted == yb).sum().item()  

accuracy = correct / total
print(f"Validation Accuracy: {accuracy*100:.2f}%")


