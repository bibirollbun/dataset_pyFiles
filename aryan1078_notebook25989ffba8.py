import os
print(os.listdir("/kaggle/input/aptos2019-blindness-detection"))



import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_df['image_path'] = train_df['id_code'].apply(lambda x: os.path.join(DATA_DIR, "train_images", f"{x}.png"))
train_df.head()



from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['diagnosis'], random_state=42)
print("Train size:", len(train_df), "Validation size:", len(val_df))



IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class RetinopathyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert("RGB")
        label = int(row['diagnosis'])
        if self.transform:
            img = self.transform(img)
        return img, label



BATCH_SIZE = 16
train_ds = RetinopathyDataset(train_df, transform=train_transform)
val_ds = RetinopathyDataset(val_df, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)



def get_model(name):
    if name == "vgg16":
        model = models.vgg16(pretrained=True)
        for p in model.features.parameters(): p.requires_grad = False
        model.classifier[6] = nn.Linear(4096, 5)
    elif name == "alexnet":
        model = models.alexnet(pretrained=True)
        for p in model.features.parameters(): p.requires_grad = False
        model.classifier[6] = nn.Linear(4096, 5)
    elif name == "resnet50":
        model = models.resnet50(pretrained=True)
        for p in model.parameters(): p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 5)
    else:
        raise ValueError("Unknown model name")
    return model.to(device)



def train_and_eval(model_name, epochs=3):
    model = get_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Training {model_name} (Epoch {epoch+1})"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}")
    
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Validating {model_name}"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds.extend(outputs.argmax(1).cpu().numpy())
            truths.extend(labels.numpy())

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='weighted')
    recall = recall_score(truths, preds, average='weighted')
    precision = precision_score(truths, preds, average='weighted')
    
    print(f"\n{model_name.upper()} Results —")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    
    return acc, f1, recall, precision



results = {}
for name in ["vgg16", "alexnet", "resnet50"]:
    acc, f1, rec, prec = train_and_eval(name, epochs=3)
    results[name] = [acc, f1, rec, prec]



results_df = pd.DataFrame(results, index=["Accuracy","F1 Score","Recall","Precision"]).T
results_df


