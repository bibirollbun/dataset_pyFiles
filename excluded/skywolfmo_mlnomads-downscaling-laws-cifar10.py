import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ===== Config =====
data_root = "/kaggle/input/ml-nomads-downscaling-laws-cifar-10"  # or "eurosat_processed"
train_dir = os.path.join(data_root, "train")
test_dir = os.path.join(data_root, "test")
num_epochs = 5
batch_size = 64
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Transforms =====
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ===== Load Train Data =====
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
class_names = train_dataset.classes
num_classes = len(class_names)

# ===== Simple CNN =====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN(num_classes).to(device)

# ===== Training =====
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

print("ğŸ§  Training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

# ===== Inference on Test Set =====
print("ğŸ”� Predicting on test images...")
model.eval()
predictions = []

# Sort files to match any original order if needed
test_images = sorted(os.listdir(test_dir))

for filename in tqdm(test_images):
    img_path = os.path.join(test_dir, filename)
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred_class = class_names[output.argmax(1).item()]
        predictions.append((filename, pred_class))





# ===== Save Submission CSV =====
submission_path =  "submission.csv"

print("ğŸ’¾ Saving submission.csv...")
submission_df = pd.DataFrame(predictions, columns=["Id", "Label"])
submission_df.to_csv(submission_path, index=False)
print(f"âœ… Submission saved to {submission_path}")





