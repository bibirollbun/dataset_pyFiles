!unzip /kaggle/input/grasp-and-lift-eeg-detection/train.zip -d /kaggle/working/train_data


!unzip /kaggle/input/grasp-and-lift-eeg-detection/test.zip -d /kaggle/working/test_data


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


TRAIN_DIR = "/kaggle/working/train_data/train/"

SFREQ = 500      
WINDOW = 250
N_CHANNELS = 32
N_CLASSES = 6

event_names = [
    "HandStart",
    "FirstDigitTouch",
    "BothStartLoadPhase",
    "LiftOff",
    "Replace",
    "BothReleased"
]


ex_data = pd.read_csv("/kaggle/working/train_data/train/subj10_series1_data.csv")
ex_data.head(5)


ex_event = pd.read_csv("/kaggle/working/train_data/train/subj10_series1_events.csv")
ex_event.head(5)


df = pd.read_csv("/kaggle/working/train_data/train/subj10_series1_data.csv")

channels = df.columns[1:33]

plt.figure(figsize=(14, 10))

offset = 2000  
for i, ch in enumerate(channels):
    plt.plot(df[ch].values + i*offset, label=ch)

plt.title("Raw EEG - 32 Channels (offset vertically)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude + offset")
plt.legend(loc="upper right", fontsize=8)
plt.show()


data_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith("_data.csv")])
event_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith("_events.csv")])

selected_subjects = [1, 2, 3, 4, 5.6,7,8]
selected_series = [1,2,3,4,5,6,7]

def match_subject_series(filename, subjects, series):
    parts = filename.replace(".csv","").split("_")
    sid = int(parts[0].replace("subj",""))
    ser = int(parts[1].replace("series",""))
    return sid in subjects and ser in series

data_files_filtered = [f for f in data_files if match_subject_series(f, selected_subjects, selected_series)]
event_files_filtered = [f for f in event_files if match_subject_series(f, selected_subjects, selected_series)]

print("Dipakai data file :", data_files_filtered)
print("Dipakai event file:", event_files_filtered)


def find_event_onsets(event_signal):
    """Ambil ONSET (0→1), bukan seluruh segmen bernilai 1."""
    return np.where((event_signal[1:] == 1) & (event_signal[:-1] == 0))[0] + 1
    
def load_grasp_file(data_path, event_path):
    data = pd.read_csv(os.path.join(TRAIN_DIR, data_path))
    events = pd.read_csv(os.path.join(TRAIN_DIR, event_path))

    data_eeg = data.values[:, 1:33]          
    events_arr = events[event_names].values  

    X_list, y_list = [], []

    # Loop kelas 0..5
    for cls_idx in range(N_CLASSES):
        onsets = find_event_onsets(events_arr[:, cls_idx])

        for start in onsets:
            end = start + WINDOW
            if end <= len(data_eeg):
                seg = data_eeg[start:end].T
                X_list.append(seg)
                y_list.append(cls_idx)

    return np.array(X_list), np.array(y_list)

X_all = []
y_all = []

for dfile, efile in zip(data_files_filtered, event_files_filtered):
    print("Loading:", dfile)
    X_sub, y_sub = load_grasp_file(dfile, efile)
    X_all.append(X_sub)
    y_all.append(y_sub)

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("X dtype:", X.dtype)
print("X shape:", X.shape)
print("TOTAL DATASET:")
print("X:", X.shape)
print("y:", y.shape)
print("Distribusi kelas:", np.bincount(y))



print("\nDistribusi per kelas:")
for i, name in enumerate(event_names):
    print(f"{i} - {name:<20}: {np.sum(y == i)} samples")


X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)

X = X[:, np.newaxis, :, :]

print("Final CNN input:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)


class EEG1DCNN(nn.Module):
    def __init__(self, n_channels=32, n_classes=6):
        super().__init__()

        # Conv temporal (filter 1 x 7)
        self.conv_temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,7), padding=(0,3), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Depthwise conv across channels
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(n_channels,1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(0.25)
        )

        self.conv_separable = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), padding=(0,7), groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(0.25)
        )

        dummy = torch.zeros(1,1,n_channels,250)
        out = self.forward_features(dummy)
        flat = out.numel()

        self.classifier = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward_features(self, x):
        x = self.conv_temporal(x)
        x = self.conv_spatial(x)
        x = self.conv_separable(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEG1DCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss={total_loss:.4f}  Test Acc={acc:.2f}%")


from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

model.eval()
y_true, y_pred = [], []
y_prob = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        preds = probs.argmax(axis=1)

        y_pred.extend(preds)
        y_true.extend(yb.numpy())
        y_prob.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=event_names,
            yticklabels=event_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — 6-Class EEG")
plt.show()

print("\nAccuracy           :", accuracy_score(y_true, y_pred))

y_true_bin = label_binarize(y_true, classes=np.arange(N_CLASSES))

macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro")
weighted_auc = roc_auc_score(y_true_bin, y_prob, average="weighted")

print("\nROC-AUC (macro)    :", macro_auc)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=event_names))




