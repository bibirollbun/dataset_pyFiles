# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')


train_images_file = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'


df


duplicates_in_csv = df['filename'][df['filename'].duplicated()]
print(f"Number of duplicate filenames in train csv: {len(duplicates_in_csv)}")


df.label.value_counts()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='label', data=df)
plt.title("Train Set Class Distribution")
plt.xticks(rotation=45)
plt.show()


import os
import cv2
import matplotlib.pyplot as plt

def show_samples(df, image_folder, n=5):
    classes = df['label'].unique()
    for breed in classes:
        sample_files = df[df['label'] == breed].sample(n)['filename'].values
        plt.figure(figsize=(15,3))
        for i, file in enumerate(sample_files):
            path = os.path.join(image_folder, file)
            if not os.path.exists(path):
                print(f"âš ï¸� File not found: {path}")
                continue
            img = cv2.imread(path)
            if img is None:
                print(f"âš ï¸� Failed to load image: {path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, n, i+1)
            plt.imshow(img)
            plt.title(breed)
            plt.axis('off')
        plt.suptitle(f"Examples of {breed}")
        plt.show()

show_samples(df, train_images_file)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import timm
from sklearn.metrics import classification_report
import joblib
# -----------------------------
# 1. data prepration
# -----------------------------
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score


class SheepDataset(Dataset):
    def __init__(self, df, img_dir,  base_transform=None, transform_dict=None, label_encoder=None):
        self.data = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.base_transform = base_transform
        self.transform_dict = transform_dict
        self.label_encoder = label_encoder
        self.data['label_encoded'] = self.label_encoder.transform(self.data['label'])
        joblib.dump(self.label_encoder, 'label_encoder.joblib')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = f"{self.img_dir}/{row['filename']}"
        image = Image.open(img_path).convert("RGB")
        label = row['label_encoded']

        transform = self.transform_dict.get(label, self.base_transform)
        if transform:
            image = transform(image)

        return image, label
        
# -----------------------------
# 2. transform setting
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


strong_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

medium_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3)], p=0.6),
    transforms.ToTensor(),
])

light_aug = transforms.Compose([
     transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])



base_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


augmentations = {
    'Barbari': strong_aug,  
    'Harri': strong_aug,    
    'Najdi': strong_aug,   
    'Roman': strong_aug,   
    'Sawakni': medium_aug,  
    'Goat': medium_aug,      
    'Naeimi': medium_aug     
}


from sklearn.model_selection import StratifiedKFold


df = pd.read_csv("/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv")
label_encoder = LabelEncoder()
df["label"] = df["label"].astype(str)
df["label_encoded"] = label_encoder.fit_transform(df["label"])
joblib.dump(label_encoder, "label_encoder.joblib")

num_classes = len(label_encoder.classes_)
img_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 3. Cross-Validation
# -----------------------------
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []


for fold, (train_idx, val_idx) in enumerate(kfold.split(df, df['label_encoded'])):
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = SheepDataset(train_df, img_dir, base_aug, augmentations, label_encoder)
    val_dataset = SheepDataset(val_df, img_dir, base_aug, augmentations, label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"\nğŸ“‚ Fold {fold+1}")
    
    model = timm.create_model(" convformer_s18.sail_in1k_384", pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    
    train_losses = []
    val_losses = []
    

    for epoch in range(10):
        model.train()
        epoch_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

    # -------- evaluate the model --------
    model.eval()
    all_preds_val, all_labels_val = [], []
    epoch_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds_val.extend(preds)
            all_labels_val.extend(labels.cpu().numpy())
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # -------- val matrix --------
    val_acc = accuracy_score(all_labels_val, all_preds_val)
    val_f1 = f1_score(all_labels_val, all_preds_val, average="weighted")
    val_recall = recall_score(all_labels_val, all_preds_val, average="weighted")
    val_precision = precision_score(all_labels_val, all_preds_val, average="weighted")

    # -------- train Accuracy --------
    all_preds_train, all_labels_train = [], []
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds_train.extend(preds)
            all_labels_train.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_labels_train, all_preds_train)

    # -------- save result Fold --------
    fold_results.append({
        "fold": fold + 1,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "val_f1": val_f1,
        "val_recall": val_recall,
        "val_precision": val_precision
    })

    print(f"âœ… Fold {fold+1}")
    print(f"Train Acc: {train_acc:.4f} | Train Loss: {train_losses[-1]:.4f}")
    print(f"Val   Acc: {val_acc:.4f} | Val Loss:   {val_losses[-1]:.4f}")
    print(f"F1: {val_f1:.4f} | Recall: {val_recall:.4f} | Precision: {val_precision:.4f}")

# -----------------------------
# 4. (Train/Val Accuracy for each Fold)
# -----------------------------
df_results = pd.DataFrame(fold_results)

print("\nğŸ“Š Accuracy Matrix (Train vs Validation):")
print(df_results[["fold", "train_accuracy", "val_accuracy"]])

print("\nğŸ“ˆ Average Scores Across Folds:")
print(df_results.mean(numeric_only=True))


# -----------------------------
# 6. Validation
# -----------------------------

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# -----------------------------
# 7. Classification results
# -----------------------------

le = train_dataset.label_encoder
print(classification_report(all_labels, all_preds, target_names=le.classes_))



import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension
    return image

num_classes = 7  

classes = train_dataset.label_encoder.classes_ # ['Naeimi', 'Goat', 'Sawakni', 'Roman', 'Najdi', 'Harri', 'Barbari']

test_folder = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test"
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

results = []

with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(test_folder, img_name)
        img_tensor = load_image(img_path).to(device)

        outputs = model(img_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        pred_label = classes[pred_idx]

        print(f"Image: {img_name} --> Predicted: {pred_label}")

        results.append({"filename": img_name, "predicted_label": pred_label})

df_results = pd.DataFrame(results)
df_results.to_csv("test_predictions_convformer_8.csv", index=False)
print("Predictions saved to test_predictions.csv")





