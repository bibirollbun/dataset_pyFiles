# ====== Imports ======
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ==== Load metadata ====
root_dir = '/kaggle/input/animal-clef-2025'
metadata_dir = '/kaggle/input/animal-clef-2025/metadata.csv'

df = pd.read_csv(metadata_dir)
database_df = df[df["split"] == "database"].dropna(subset=['identity'])
database_df


# ==== Counting the no of images v/s identity in database

identity_counts = df[df['split'] == 'database']['identity'].value_counts().reset_index()
identity_counts.columns = ['identity', 'num_images']
identity_counts


# counting no of classes with only 1 image
identity_counts[identity_counts['num_images'] == 1].value_counts()


# ==== Creating mappings =====
encoder = LabelEncoder()
database_df['label'] = encoder.fit_transform(database_df['identity'])


database_df


# ==== Custom Dataset ====
class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        # print(img_path)
        img_path = os.path.join(root_dir, img_path)
        label = self.df.loc[idx, 'label']
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ==== Transformations ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ==== Dataset and Dataloader ====
# train_df, val_df = tts(database_df, test_size=0.2, random_state=42)

train_dataset = CustomDataset(database_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# val_dataset = CustomDataset(val_df, transform=transform)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# ==== Model ====
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(encoder.classes_))
model = model.to(device)


# === Training setup ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
EPOCHS = 10
# === Training Loop ====
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


# ==== save model and encoder ====
import joblib
torch.save(model.state_dict(), 'reid_classifier.pth')
joblib.dump(encoder, 'label_encoder.pkl')


df = pd.read_csv(metadata_dir)
query_df = df[df['split'] == 'query'].reset_index(drop=True)
query_df


# ==== Load model and encoder ====
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features,  len(joblib.load("label_encoder.pkl").classes_))
model.load_state_dict(torch.load("reid_classifier.pth"))
model = model.to(device)
model.eval()

encoder = joblib.load("label_encoder.pkl")


# === Inference ===
threshold = 0.5
results = []

for i, row in query_df.iterrows():
    img_path = os.path.join(root_dir, row['path'])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        max_prob, pred_class = torch.max(probs, 1)

    if max_prob.item() >= threshold:
        identity = encoder.inverse_transform([pred_class.item()])[0]
    else:
        identity = "new_individual"

    results.append({
        "image_id": row["image_id"],
        "identity": identity
    })

# === Save to CSV ===
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)
print("Saved predictions to sample_submission.csv")




