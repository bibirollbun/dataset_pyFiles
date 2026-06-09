!mkdir /kaggle/working/data


!tar -xzvf '/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013.tar.gz' -C '/kaggle/working/data/'


import pandas as pd
df = pd.read_csv('/kaggle/working/data/fer2013/fer2013.csv')
df.head()


df


# utilities/preprocess_data.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Function to preprocess FER2013 dataset
def preprocess_fer2013(input_csv, output_dir, emotion_labels):
    os.makedirs(output_dir, exist_ok=True)
    for usage in ['train', 'val', 'test']:
        for label in emotion_labels.values():
            os.makedirs(os.path.join(output_dir, usage, label), exist_ok=True)

    df = pd.read_csv(input_csv)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing images'):
        pixels = np.array(row['pixels'].split(), dtype='uint8')
        image = pixels.reshape(48, 48)
        img = Image.fromarray(image)
        label = emotion_labels[row['emotion']]
        usage = row['Usage']
        if usage == 'Training':
            path = os.path.join(output_dir, 'train', label)
        elif usage == 'PublicTest':
            path = os.path.join(output_dir, 'val', label)
        else:
            path = os.path.join(output_dir, 'test', label)
        img.save(os.path.join(path, f'{index}.jpg'))

if __name__ == "__main__":
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    preprocess_fer2013('/kaggle/working/data/fer2013/fer2013.csv',
                       'FER2013_processed',
                       emotion_labels)


import matplotlib.pyplot as plt
import cv2
img = cv2.imread('/kaggle/working/FER2013_processed/train/Angry/1.jpg')
plt.imshow(img)
plt.show()


img.shape


import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from tqdm import tqdm

# Custom model class
class CustomViT(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(CustomViT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# Training function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root="FER2013_processed/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = CustomViT(pretrained=True, num_classes=7).to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, 'cuda')
        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.4f}")


# Save the trained model
torch.save(model.state_dict(), "best_model.pth")
print("Model saved to model.pth")


model1 = CustomViT(num_classes=7).to('cuda')
model1.load_state_dict(torch.load("best_model.pth"))
model1.eval()


model1 = torch.load("best_model.pth", map_location='cuda')


model1.eval()


import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

def evaluate_model(model_path, data_dir, transform, device):
    # model = torch.load(model_path, map_location=device)
    # model.eval()

    model = CustomViT(num_classes=7).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    evaluate_model("best_model.pth", "FER2013_processed/test", transform, 'cuda')


import os
import pandas as pd
import librosa

# Dataset paths
# daic_woz_path = '/kaggle/input/depression-detection/DAIC_woz'
daic_woz_path = '/kaggle/input/daicwoz/daicwoz/daicwoz'
# wav_path = os.path.join(daic_woz_path, ".wav")  # remove dot from '.wav' if it's actually 'wav'

#  Load metadata
train_df = pd.read_csv(os.path.join(daic_woz_path, 'train_split_Depression_AVEC2017.csv'))
dev_df = pd.read_csv(os.path.join(daic_woz_path, 'dev_split_Depression_AVEC2017.csv'))

#  Map PHQ-8 Score to depression class
def map_score(score):
    if score <= 4:
        return 0  # No Depression
    elif score <= 9:
        return 1  # Mild Depression
    else:
        return 2  # Severe Depression

train_df["Depression_Class"] = train_df["PHQ8_Score"].apply(map_score)
dev_df["Depression_Class"] = dev_df["PHQ8_Score"].apply(map_score)

# combine metadata
combined_df = pd.concat([train_df, dev_df], ignore_index=True)

#  Create PHQ-8 score dict
phq_dict = dict(zip(combined_df["Participant_ID"], combined_df["PHQ8_Score"]))
print("PHQ-8 dictionary created for", len(phq_dict), "participants.")

#  match .wav files to PHQ8 participants
# wav_files = [f for f in os.listdir(wav_path) if f.endswith('.wav')]
wav_files = [f for f in os.listdir(daic_woz_path) if f.endswith('.wav')]
matched = []
matched_clnf = []
for wav in wav_files:
    try:
        # pid = int(os.path.splitext(wav)[0])
        pid = int(os.path.splitext(wav)[0][:3])
        # print(pid)
        if pid in phq_dict:
            # print('a')
            matched.append((wav, phq_dict[pid]))
            # print('b')
            # print('a:', (str(pid) + '_CLNF_features.txt', phq_dict[pid]))
            matched_clnf.append((str(pid) + '_CLNF_features.txt', phq_dict[pid]))
    except Exception as e:
        print(e)
        continue

print(f" Matched {len(matched)} WAV files with PHQ-8 scores.")
print(" First 5 matches:", matched[:5])
print(" First 5 matches:", matched_clnf[:5])


import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=40, max_len=400):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # Transpose to shape (time, features)

    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]

    return mfcc


categories = {
    0: "No depression",
    1: "Mild depression",
    2: "Moderate depression",
    3: "Severe depression",
    4: "Extreme depression"  # If needed
}


mfcc_features = []
labels = []
clnf_features = []

print(" Starting MFCC extraction...")

for i, (wav_file, score) in enumerate(matched):
    audio_path = os.path.join(daic_woz_path, wav_file)
    try:
        # print("i is: ", i)
        mfcc = extract_mfcc(audio_path, n_mfcc=40, max_len=400)
        clnf = np.genfromtxt("/kaggle/input/daicwoz/daicwoz/daicwoz/" + matched_clnf[i][0], delimiter=",", skip_header=1)
        mfcc_features.append(mfcc)
        clnf_features.append(clnf)

        # labels.append(int(score >= 10))
        if 0 <= score <= 4:
            x = 0
        elif 5 <= score <= 9:
            x = 1
        elif 10 <= score <= 14:
            x = 2
        elif 15 <= score <= 19:
            x = 3
        elif score >= 20:
            x = 4

        labels.append(x)
    
        # if i % 10 == 0:
        print(f" Processed {i+1}/{len(matched)}: {wav_file}")
    except Exception as e:
        print(f" Error processing {wav_file}: {e}")

print(" Done extracting MFCC features.")


mfcc_features[0].shape


labels


import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. MFCCs (140 samples of shape (400, 40))
num_samples = 140

# 2. Dataset with augmentation
class MFCCDataset(Dataset):
    def __init__(self, mfccs, labels, augment=False):
        self.mfccs = mfcc_features
        self.labels = labels
        self.augment = augment
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)

    def __getitem__(self, idx):
        x = torch.tensor(self.mfccs[idx])  # (400, 40)
        y = torch.tensor(self.labels[idx])

        if self.augment:
            x = self.apply_augmentation(x)

        return x, y

    def __len__(self):
        return len(self.mfccs)

    def apply_augmentation(self, x):
        x = x.clone()
        x = self.time_mask(x)
        x = self.freq_mask(x)
        x = x + torch.randn_like(x) * 0.005
        return x

# 3. Create dataset and dataloader
dataset = MFCCDataset(mfcc_features, labels, augment=True)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# 4. Calculate total shape
all_augmented = []

for batch_x, batch_y in loader:
    print(f"Batch shape: {batch_x.shape}")  # (B, 400, 40)
    all_augmented.append(batch_x)

# Stack all batches
all_augmented_tensor = torch.cat(all_augmented, dim=0)
print("\nTotal augmented data shape:", all_augmented_tensor.shape)


import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np

num_original_samples = 140

# -------------------------
# 1. Dataset Class with Augmentation
# -------------------------
class AugmentedMFCCDataset(Dataset):
    def __init__(self, mfccs, labels, num_augs=5, augment=True):
        """
        mfccs: list of np.arrays of shape (400, 40)
        labels: list of labels
        num_augs: number of augmented samples per original
        """
        self.num_augs = num_augs
        self.augment = augment

        # Define augmentation transforms
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)

        # Expand dataset: repeat each sample 'num_augs' times
        self.expanded_mfccs = []
        self.expanded_labels = []

        for mfcc, label in zip(mfccs, labels):
            for _ in range(num_augs):
                self.expanded_mfccs.append(mfcc)
                self.expanded_labels.append(label)

    def __len__(self):
        return len(self.expanded_mfccs)

    def __getitem__(self, idx):
        x = torch.tensor(self.expanded_mfccs[idx])  # shape: (400, 40)
        y = torch.tensor(self.expanded_labels[idx])  # scalar

        if self.augment:
            x = self.apply_augmentation(x)

        return x, y

    def apply_augmentation(self, x):
        x = x.clone()
        x = self.time_mask(x)
        x = self.freq_mask(x)
        x = x + torch.randn_like(x) * 0.005  # Gaussian noise
        return x

# -------------------------
# 2. Create Dataset and DataLoader
# -------------------------
num_augs_per_sample = 5
dataset = AugmentedMFCCDataset(mfcc_features, labels, num_augs=num_augs_per_sample, augment=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------
# 3. Print Shapes and Final Dataset Size
# -------------------------
print(f"Original dataset size: {len(mfcc_features)}")
print(f"Augmented dataset size: {len(dataset)}\n")

for batch_x, batch_y in loader:
    print("Batch X shape:", batch_x.shape)  # [batch_size, 400, 40]
    print("Batch Y shape:", batch_y.shape)  # [batch_size]
    break  # Just show one batch

# -------------------------
# 4. Combine All Data into Tensors (Optional)
# -------------------------
all_augmented_x = torch.stack([dataset[i][0] for i in range(len(dataset))])  # shape: (700, 400, 40)
all_augmented_y = torch.tensor([dataset[i][1] for i in range(len(dataset))]) # shape: (700,)

print("\nFinal total tensor shape (X):", all_augmented_x.shape)
print("Final total tensor shape (Y):", all_augmented_y.shape)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from tqdm import tqdm

# -------------------------------
# 1. Dataset for MFCCs
# -------------------------------
class MFCCDataset(Dataset):
    def __init__(self, mfcc_data, labels):
        self.mfcc_data = mfcc_data  # shape: (N, 400, 40)
        self.labels = labels        # shape: (N,)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # expects (H, W, C)
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
        ])

    def __len__(self):
        return len(self.mfcc_data)

    def __getitem__(self, idx):
        mfcc = self.mfcc_data[idx]  # (400, 40)
        label = self.labels[idx]    # integer 0–4

        # Convert to 3-channel image (e.g., by repeating or stacking)
        image = torch.tensor(mfcc).unsqueeze(0).repeat(3, 1, 1)  # (3, 400, 40)
        image = self.transform(image)  # (3, 224, 224)
        return image, torch.tensor(label, dtype=torch.long)

# -------------------------------
# 2. Model Definition
# -------------------------------
class CustomViT(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(CustomViT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# -------------------------------
# 3. Training Function
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# -------------------------------
# 4. Main Runner
# -------------------------------
if __name__ == "__main__":
    # Load your MFCC data augmented

    dataset = MFCCDataset(all_augmented_x, all_augmented_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CustomViT(pretrained=True, num_classes=5).to("cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss, acc = train_one_epoch(model, dataloader, optimizer, criterion, "cuda")
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from tqdm import tqdm

# -------------------------------
# 1. Dataset for MFCCs
# -------------------------------
class MFCCDataset(Dataset):
    def __init__(self, mfcc_data, labels):
        self.mfcc_data = mfcc_data  # shape: (N, 400, 40)
        self.labels = labels        # shape: (N,)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # expects (H, W, C)
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
        ])

    def __len__(self):
        return len(self.mfcc_data)

    def __getitem__(self, idx):
        mfcc = self.mfcc_data[idx]           # shape: (400, 40)
        label = self.labels[idx]             # scalar label

        image = torch.tensor(mfcc).unsqueeze(0).repeat(3, 1, 1)  # shape: (3, 400, 40)
        image = self.transform(image)                            # shape: (3, 224, 224)

        return image, torch.tensor(label, dtype=torch.long)

# -------------------------------
# 2. Model Definition
# -------------------------------
class CustomViT(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(CustomViT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# -------------------------------
# 3. Training Function
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# -------------------------------
# 4. Main Execution
# -------------------------------
if __name__ == "__main__":
    # ---- Load your actual MFCC data (shape: [140, 400, 40]) ----

    dataset = MFCCDataset(mfcc_features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CustomViT(pretrained=True, num_classes=5).to("cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss, acc = train_one_epoch(model, dataloader, optimizer, criterion, "cuda")
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")



import matplotlib.pyplot as plt
for i in range(19):
    plt.imshow(all_augmented_x[i])
    plt.show()




