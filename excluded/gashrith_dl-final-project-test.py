# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViTImageProcessor.from_pretrained("/kaggle/input/vit/pytorch/default/1/Model-1")
model = ViTForImageClassification.from_pretrained("/kaggle/input/vit/pytorch/default/1/Model-1").to(device)
model.eval()


class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, processor, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipping corrupted file: {img_path}")
            return self.__getitem__((idx + 1) % len(self.file_paths))
        
        # Apply augmentations only if specified
        if self.augment:
            image = train_augmentations(image)
        
        # Process using processor
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze()
        
        return {
            'pixel_values': pixel_values,
            'labels': self.labels[idx]
        }


real_dir = "/kaggle/input/deepfake-and-real-images/Dataset/Test/Real"
fake_dir = "/kaggle/input/deepfake-and-real-images/Dataset/Test/Fake"

real_paths = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir)]
fake_paths = [os.path.join(fake_dir, fname) for fname in os.listdir(fake_dir)]

file_paths = real_paths + fake_paths
labels = [0]*len(real_paths) + [1]*len(fake_paths)

dataset = DeepfakeDataset(file_paths, labels, processor, augment=False)


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    eer_index = np.nanargmin(abs_diffs)
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    probs = p.predictions[:, 1]  # Probability of class 1 (fake)
    labels = p.label_ids

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    eer = compute_eer(labels, probs)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'eer': eer
    }

def plot_confusion_matrix(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()



# === Predict on test data ===
all_preds_test = []
all_labels_test = []

for idx in range(len(dataset)):
    batch = dataset[idx]
    img = batch['pixel_values'].unsqueeze(0).to(device)
    label = torch.tensor(batch['labels']).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=img)
        pred = torch.argmax(outputs.logits, dim=1)

    all_preds_test.append(pred.item())
    all_labels_test.append(label.item())

    if idx % 1000 == 0:
        print(f"Test Image: {idx}")

# === Evaluate on test set ===
print("\nTest Set Evaluation:")
print(classification_report(all_labels_test, all_preds_test, target_names=["Real", "Fake"]))


plot_confusion_matrix(all_preds_test, all_labels_test)




