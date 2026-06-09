# References
# 1. https://www.kaggle.com/code/anishasharma12/notebookbc4c6c444a
# 2. https://github.com/mujiyantosvc/Facial-Expression-Recognition-FER-for-Mental-Health-Detection-


import pandas as pd
import os

# Correct folder path where CSVs are located
daic_woz_path = "/kaggle/input/depression-detection/DAIC_woz"

# Now load the CSVs correctly
train_df = pd.read_csv(os.path.join(daic_woz_path, 'train_split_Depression_AVEC2017.csv'))
dev_df = pd.read_csv(os.path.join(daic_woz_path, 'dev_split_Depression_AVEC2017.csv'))

# Display first few rows
train_df.head()


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
# matched_clnf = []
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
            # matched_clnf.append((str(pid) + '_CLNF_features.txt', phq_dict[pid]))
    except Exception as e:
        print(e)
        continue

print(f" Matched {len(matched)} WAV files with PHQ-8 scores.")
print(" First 5 matches:", matched[:5])
# print(" First 5 matches:", matched_clnf[:5])


matched


import librosa
import cv2
import numpy as np

def extract_mfcc(audio_path):
    # Load only mono channel & downsample directly if needed
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Compute mel-spectrogram in float32 (smaller memory footprint)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # Convert to log scale directly (no librosa.display)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)  # shape (128, time)

    # Resize time dimension directly to 224 (downsample)
    mfcc_downsampled = cv2.resize(S_db, (224, 128), interpolation=cv2.INTER_AREA)

    # Resize height to 224×224 (square input)
    mfcc_img = cv2.resize(mfcc_downsampled, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize (min–max) and convert to 3-channel efficiently
    mn, mx = mfcc_img.min(), mfcc_img.max()
    if mx > mn:  # avoid divide by zero
        mfcc_norm = (mfcc_img - mn) / (mx - mn)
    else:
        mfcc_norm = np.zeros_like(mfcc_img, dtype=np.float32)

    # Stack once (avoid Python copies)
    mfcc_3ch = np.repeat(mfcc_norm[..., None], 3, axis=-1)  # (224,224,3)

    return mfcc_3ch.astype(np.float32)



categories = {
    0: "No depression",
    1: "Mild depression",
    2: "Moderate depression",
    3: "Severe depression",
    4: "Extreme depression"  
}


mfcc_features = []
labels = []
# clnf_features = []

print(" Starting MFCC extraction...")

for i, (wav_file, score) in enumerate(matched):
    audio_path = os.path.join(daic_woz_path, wav_file)
    try:
        # print("i is: ", i)
        # mfcc = extract_mfcc(audio_path, n_mfcc=40, max_len=400)
        mfcc = extract_mfcc(audio_path)
        # clnf = np.genfromtxt("/kaggle/input/daicwoz/daicwoz/daicwoz/" + matched_clnf[i][0], delimiter=",", skip_header=1)
        mfcc_features.append(mfcc)
        # clnf_features.append(clnf)

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


import matplotlib.pyplot as plt

plt.imshow(mfcc_features[42], aspect='auto', origin='lower')
plt.title("MFCC (224 × 224 × 3)")
plt.xlabel("Time Frames")           # X-axis → Time
plt.ylabel("MFCC Coefficients")     # Y-axis → Frequency / MFCC bins
plt.colorbar(label="Amplitude (normalized)")
plt.show()


mfcc_f = np.array(mfcc_features)


mfcc_f.shape


import torch
vit_ready_batch = torch.from_numpy(mfcc_f).permute(0, 3, 1, 2)


print(vit_ready_batch.shape) 


torch.save(vit_ready_batch, 'X.pt')


# import pickle
# with open('clnf_features', 'wb') as fp:
#     pickle.dump(clnf_features, fp)


import pickle
with open('labels', 'wb') as fp:
    pickle.dump(labels, fp)


import torch
vit_ready_batch = torch.load('/kaggle/working/X.pt')
# X_clnf = torch.load('/kaggle/working/Y.pt')


# import pickle
# with open('clnf_features', 'rb') as fp:
#     clnf_features = pickle.load(fp)


import numpy as np


# torch.save(X_clnf, 'Y.pt')


import pickle
with open('labels', 'rb') as fp:
    labels = pickle.load(fp)


print(vit_ready_batch.shape)  # torch.Size([140, 3, 224, 224])


import torch
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Dataset
class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images          # Tensor [N, 3, 224, 224]
        self.labels = torch.tensor(labels, dtype=torch.long)  # Tensor [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Custom ViT model
class CustomViT(torch.nn.Module):
    def __init__(self, pretrained, num_classes):
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
def train_one_epoch(model, dataloader, optimizer, criterion, device, class_names=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Metrics Calculations
    epoch_loss = running_loss / len(dataloader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_acc = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, precision, recall, f1, cm

# Main Training Loop
if __name__ == "__main__":
    
    num_classes = 5

    dataset = TensorDataset(vit_ready_batch, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomViT(pretrained=True, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(80):
        loss, acc, precision, recall, f1, cm = train_one_epoch(model, dataloader, optimizer, criterion, device)
        # print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.4f}, Precision=")

        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")
        
        # Print raw confusion matrix
        # print("Confusion Matrix:\n", cm)


plt.figure(figsize=(10, 7))
cn = ["No depression", "Mild depression", "Moderate depression", "Severe depression", "Extreme depression"]
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=cn, yticklabels=cn)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# !rm -rf '/kaggle/working/model1.pth'


# Save the trained model
torch.save(model.state_dict(), "model1.pth")
print("Model saved to model1.pth")


# outputs = model(vit_ready_batch[1:2], X_clnf[1:2])
# # print(outputs)
# predicted_class = torch.argmax(outputs, dim=1).item()
# print("Actual: ", labels[1:2][0])
# print("Prediction is:", categories[predicted_class], '(', predicted_class, ')')


!mkdir /kaggle/working/data


!tar -xzvf '/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013.tar.gz' -C '/kaggle/working/data/'


import pandas as pd
df = pd.read_csv('/kaggle/working/data/fer2013/fer2013.csv')
# df.head()


df


# Add mapping from 7 → 5 classes
fer_to_depression = {
    0: 3,  # Angry → Severe
    1: 2,  # Disgust → Moderate
    2: 2,  # Fear → Moderate
    3: 0,  # Happy → No depression
    4: 4,  # Sad → Extreme
    5: 1,  # Surprise → Mild
    6: 1   # Neutral → Mild
}

depression_labels = {
    0: "No depression",
    1: "Mild depression",
    2: "Moderate depression",
    3: "Severe depression",
    4: "Extreme depression"
}


import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Function to preprocess FER2013 dataset
def preprocess_fer2013(input_csv, output_dir, emotion_labels):
    os.makedirs(output_dir, exist_ok=True)
    for usage in ['train', 'val', 'test']:
        for label in depression_labels.values():
            os.makedirs(os.path.join(output_dir, usage, label), exist_ok=True)

    df = pd.read_csv(input_csv)
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing images'):
        pixels = np.array(row['pixels'].split(), dtype='uint8')
        image = pixels.reshape(48, 48)
        img = Image.fromarray(image)
        
        # Map FER emotion to depression category
        dep_class = fer_to_depression[row['emotion']]
        label = depression_labels[dep_class]
        
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
img = cv2.imread('/kaggle/working/FER2013_processed/train/No depression/10001.jpg')
plt.imshow(img)
plt.show()


# img.shape


import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from tqdm import tqdm

# Custom model class
class CustomViT(torch.nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
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

    model = CustomViT(pretrained=True, num_classes=5).to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(20):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, 'cuda')
        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.4f}")


# Save the trained model
torch.save(model.state_dict(), "model2.pth")
print("Model saved to model2.pth")


import torch
import timm

class CustomViT(torch.nn.Module):
    def __init__(self, pretrained, num_classes):
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mfcc_model = CustomViT(pretrained=False, num_classes=5).to(device)
mfcc_model.load_state_dict(torch.load("/kaggle/working/model1.pth", map_location=device))
mfcc_model.eval()


import pickle
with open('/kaggle/working/labels', 'rb') as fp:
    mfcc_labels = pickle.load(fp)


import torch
mfcc_data = torch.load('/kaggle/working/X.pt')
mfcc_data.shape


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get FER probabilities
fer_model = CustomViT(pretrained=False, num_classes=5).to(device)
fer_model.load_state_dict(torch.load("/kaggle/working/model2.pth", map_location=device))
fer_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="FER2013_processed/train", transform=transform)
fer_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

fer_probs_list, fer_labels_list = [], []

with torch.no_grad():
    for imgs, labels in tqdm(fer_loader, desc="Running FER model"):
        imgs = imgs.to(device)
        logits = fer_model(imgs)
        probs = F.softmax(logits, dim=1)
        fer_probs_list.append(probs.cpu())
        fer_labels_list.append(labels)

fer_probs = torch.cat(fer_probs_list, dim=0)      # [35887, 5]
fer_labels = torch.cat(fer_labels_list, dim=0)    # [35887]
print("FER probabilities shape:", fer_probs.shape)

# Compute per-class averages
fer_class_avg = torch.zeros(5, 5)
for c in range(5):
    mask = (fer_labels == c)
    if mask.sum() > 0:
        fer_class_avg[c] = fer_probs[mask].mean(dim=0)
fer_class_avg = fer_class_avg.to(device)          # [5, 5]
print("FER class average probabilities shape:", fer_class_avg.shape)

# Get MFCC model predictions (batched)
mfcc_model = CustomViT(pretrained=False, num_classes=5).to(device)
mfcc_model.load_state_dict(torch.load("/kaggle/working/model1.pth", map_location=device))
# mfcc_model.eval()

mfcc_dataset = TensorDataset(mfcc_data, torch.tensor(mfcc_labels))
mfcc_loader = DataLoader(mfcc_dataset, batch_size=8, shuffle=False)

mfcc_probs_list, mfcc_labels_list = [], []

with torch.no_grad():
    for batch, labels in tqdm(mfcc_loader, desc="Running MFCC model"):
        batch = batch.to(device)
        logits = mfcc_model(batch)
        probs = F.softmax(logits, dim=1)
        mfcc_probs_list.append(probs.cpu())
        mfcc_labels_list.append(labels)

mfcc_probs = torch.cat(mfcc_probs_list, dim=0)      # [140, 5]
mfcc_labels = torch.cat(mfcc_labels_list, dim=0)    # [140]
print("MFCC probabilities shape:", mfcc_probs.shape)


# Decision-level fusion
alpha = 0.5
mfcc_probs = mfcc_probs.to(device)
mfcc_labels = mfcc_labels.to(device)

fused_probs = torch.zeros_like(mfcc_probs).to(device)

for i, label in enumerate(mfcc_labels):
    fused_probs[i] = alpha * mfcc_probs[i] + (1 - alpha) * fer_class_avg[label]

fused_pred = torch.argmax(fused_probs, dim=1)

# Evaluate
accuracy = (fused_pred.cpu() == mfcc_labels.cpu()).float().mean().item() * 100
print(f"\nDecision-Level Fusion Accuracy (Ensemble averaging): {accuracy:.2f}%")


categories = {
    0: "No depression",
    1: "Mild depression",
    2: "Moderate depression",
    3: "Severe depression",
    4: "Extreme depression"  
}


import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import librosa
import numpy as np
import cv2, os, shutil
from moviepy.editor import VideoFileClip
import timm

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define CustomViT
class CustomViT(torch.nn.Module):
    def __init__(self, pretrained, num_classes):
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

# Load Trained Models
fer_model = CustomViT(pretrained=False, num_classes=5).to(device)
mfcc_model = CustomViT(pretrained=False, num_classes=5).to(device)

fer_model.load_state_dict(torch.load("/kaggle/working/model2.pth", map_location=device))
mfcc_model.load_state_dict(torch.load("/kaggle/working/model1.pth", map_location=device))

fer_model.eval()
mfcc_model.eval()

# Helper Functions
def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extract audio from video using MoviePy (safe version)."""
    clip = VideoFileClip(video_path)
    
    if clip.audio is None:
        print("No audio track found in this video — skipping audio extraction.")
        return None

    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

def extract_frames(video_path, frames_dir="frames", frame_rate=1):
    """Extract 1 frame per second from video using OpenCV."""
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(1, int(fps / frame_rate))
    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(frames_dir, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
    return frames_dir

def extract_mfcc(audio_path):
    """Extract MFCC (224, 224, 3)"""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)  # shape (128, time)
    mfcc_downsampled = cv2.resize(S_db, (224, 128), interpolation=cv2.INTER_AREA)
    mfcc_img = cv2.resize(mfcc_downsampled, (224, 224), interpolation=cv2.INTER_AREA)
    mn, mx = mfcc_img.min(), mfcc_img.max()
    if mx > mn:  # avoid divide by zero
        mfcc_norm = (mfcc_img - mn) / (mx - mn)
    else:
        mfcc_norm = np.zeros_like(mfcc_img, dtype=np.float32)
    mfcc_3ch = np.repeat(mfcc_norm[..., None], 3, axis=-1)  # (224,224,3)

    return mfcc_3ch.astype(np.float32)

def preprocess_audio_to_mfcc_tensor(audio_path):
    """Extract MFCC-based RGB image tensor for ViT."""
    mfcc_img = extract_mfcc(audio_path)  # (224, 224, 3)
    mfcc_img = torch.tensor(mfcc_img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,224,224)
    return mfcc_img

def preprocess_frames_to_tensor_list(frames_dir):
    """Convert extracted frames to tensors for FER model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensors = []
    for f in sorted(os.listdir(frames_dir)):
        if f.endswith(".jpg"):
            img = Image.open(os.path.join(frames_dir, f)).convert("RGB")
            tensors.append(transform(img).unsqueeze(0).to(device))
    return tensors

def get_probs_from_mfcc_model(mfcc_tensor):
    """Run MFCC model inference and return softmax probabilities."""
    with torch.no_grad():
        logits = mfcc_model(mfcc_tensor)
        probs = F.softmax(logits, dim=1)
    return probs


def get_probs_from_frames(frames_dir):
    """Run FER model on all frames and return averaged softmax probabilities."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    probs_list = []
    with torch.no_grad():
        for f in sorted(os.listdir(frames_dir)):
            if f.endswith(".jpg"):
                img_path = os.path.join(frames_dir, f)
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                logits = fer_model(tensor)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs.cpu())
    
    if len(probs_list) == 0:
        print("[WARN] No frames found in directory:", frames_dir)
        return torch.zeros((1, 5), device=device)  # 5 = num_classes
    
    fer_probs = torch.cat(probs_list, dim=0).mean(dim=0, keepdim=True).to(device)
    return fer_probs

def fuse_predictions(mfcc_probs, fer_probs, alpha=0.5):
    """Perform decision-level fusion."""
    fer_avg = fer_probs.mean(dim=0, keepdim=True)  # average over frames
    fused = alpha * mfcc_probs + (1 - alpha) * fer_avg
    pred = torch.argmax(fused, dim=1).item()
    return fused, pred

# Inference Function
def infer_video(video_path, alpha=0.5):
    print(f"\nProcessing video: {video_path}")
    
    audio_path = extract_audio(video_path)
    frames_dir = extract_frames(video_path)
    
    mfcc_probs = None
    if audio_path is not None:
        mfcc_tensor = preprocess_audio_to_mfcc_tensor(audio_path)
        mfcc_probs = get_probs_from_mfcc_model(mfcc_tensor)
    else:
        print("Skipping MFCC model since no audio found.")
    
    fer_probs = get_probs_from_frames(frames_dir)
    
    # If only FER is available, fallback gracefully
    if mfcc_probs is None:
        fused_probs = fer_probs
    else:
        fused_probs = alpha * mfcc_probs + (1 - alpha) * fer_probs
    
    pred_class = torch.argmax(fused_probs, dim=1)
    return pred_class, fused_probs

# Run Inference
video_path = "test.mp4"  # testing file
pred_class, probs = infer_video(video_path, alpha=0.5)
print(f"Predicted class: {pred_class.item()} ({categories[pred_class.item()].strip()})")
print("Class probabilities:", probs.cpu().numpy())

