import os
import cv2
import sys
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
train_csv_path = "/kaggle/input/what-on-the-video/train.csv"
train_dir = "/kaggle/input/what-on-the-video/train/"
test_dir = "/kaggle/input/what-on-the-video/test/"


def extract_resnet_embeddings(video_path, frame_interval=25, device='cuda'):
    try:
        model = models.resnet50(pretrained=True).to(device)
        
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        embeddings = []

        frame_idx = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model(image)
                    embeddings.append(embedding.cpu().view(-1))
            frame_idx += 1

        video.release()
        if not embeddings:
            raise ValueError(f"No frames extracted from {video_path}")
        return torch.stack(embeddings)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


class ResNetRNNClassifier(nn.Module):
    def __init__(self, embedding_dim=2048, hidden_dim=512, num_layers=2, num_classes=9, dropout=0.1):
        super(ResNetRNNClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        return self.classifier(out)


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, device='cpu', cache_dir='embeddings'):
        self.video_paths = video_paths
        self.labels = labels
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.encoded_videos = self.load_embeddings()

    def load_embeddings(self):
        encoded_videos = {}
        for video_path in tqdm(self.video_paths, desc="Loading embeddings"):
            cache_path = os.path.join(self.cache_dir, f"{os.path.basename(video_path)}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    encoded_videos[video_path] = pickle.load(f)
            else:
                embeddings = extract_resnet_embeddings(video_path, frame_interval=25, device=self.device)
                if embeddings is not None:
                    encoded_videos[video_path] = embeddings
                    with open(cache_path, 'wb') as f:
                        pickle.dump(embeddings, f)
        return encoded_videos

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        embeddings = self.encoded_videos.get(video_path)
        if embeddings is None:
            embeddings = torch.zeros((1, 2048))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embeddings, label

def collate_fn(batch):
    embeddings, labels = zip(*batch)
    padded_embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    labels_tensor = torch.stack(labels)
    return padded_embeddings, labels_tensor


def compute_class_wise_accuracy(outputs, labels, threshold=0.5):
    preds = (outputs > threshold).float()
    correct = (preds == labels).float()
    class_acc = correct.sum(dim=0) / labels.size(0)
    return class_acc.mean().item(), class_acc


train_df = pd.read_csv(train_csv_path)
global_labels = ["animal", "car", "cloud", "dance", "fire", "flower", "food", "sunset", "water"]
label_map = {label: i for i, label in enumerate(global_labels)}

train_data = {}
for _, row in train_df.iterrows():
    video_name = os.path.basename(row['path'])
    video_labels = row['labels'].replace(' ', '').split(',')
    label_vec = [0] * len(global_labels)
    for lab in video_labels:
        if lab in label_map:
            label_vec[label_map[lab]] = 1
    train_data[video_name] = label_vec

video_paths = [os.path.join(train_dir, name) for name in train_data.keys()]
labels_tensor = torch.tensor(list(train_data.values()), dtype=torch.float32)

train_idx, val_idx = train_test_split(range(len(video_paths)), test_size=0.2, random_state=42)
train_paths = [video_paths[i] for i in train_idx]
train_labels = labels_tensor[train_idx]
val_paths = [video_paths[i] for i in val_idx]
val_labels = labels_tensor[val_idx]

train_dataset = VideoDataset(train_paths, train_labels, device=device, cache_dir='embeddings_train')
val_dataset = VideoDataset(val_paths, val_labels, device=device, cache_dir='embeddings_val')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


model = ResNetRNNClassifier(num_classes=len(global_labels), num_layers=3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
num_epochs = 100

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    train_loss = 0
    train_correct = torch.zeros(len(global_labels)).to(device)
    train_total = 0

    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
        train_total += labels.size(0)
        class_acc, _ = compute_class_wise_accuracy(torch.sigmoid(outputs), labels)
        train_correct += (torch.sigmoid(outputs) > 0.5).float().eq(labels).sum(dim=0)

    train_loss /= train_total
    train_acc = (train_correct / train_total).mean().item()

    model.eval()
    val_loss = 0
    val_correct = torch.zeros(len(global_labels)).to(device)
    val_total = 0
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            val_total += labels.size(0)
            class_acc, class_accs = compute_class_wise_accuracy(torch.sigmoid(outputs), labels)
            val_correct += (torch.sigmoid(outputs) > 0.5).float().eq(labels).sum(dim=0)

    val_loss /= val_total
    val_acc = (val_correct / val_total).mean().item()
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_loss:.4f}, Train Class-Wise Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Class-Wise Accuracy: {val_acc:.4f}")
    print(f"Val Per-Class Accuracy: {', '.join([f'{global_labels[i]}: {acc:.4f}' for i, acc in enumerate(val_correct / val_total)])}")



test_df = pd.read_csv('/kaggle/input/what-on-the-video/sample_submit.csv')
test_paths = [os.path.join(test_dir, os.path.basename(p)) for p in test_df['file_name']]
test_dataset = VideoDataset(test_paths, torch.zeros(len(test_paths), len(labels)), device=device, cache_dir='embeddings_test')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

model.eval()
all_preds = []
with torch.no_grad():
    for embeddings, _ in test_loader:
        embeddings = embeddings.to(device)
        outputs = torch.sigmoid(model(embeddings))
        all_preds.append(outputs.cpu())
all_preds = torch.cat(all_preds)
pred_labels = (all_preds > 0.5).numpy()

for i in range(len(pred_labels)):
    if np.sum(pred_labels[i]) == 0:
        pred_labels[i][np.argmax(all_preds[i])] = True
pred_labels = [', '.join([global_labels[i] for i, p in enumerate(row) if p]) for row in pred_labels]
submission = pd.DataFrame({
    'index': test_df.index,
    'file_name': test_df['file_name'],
    'label': pred_labels
})
submission.to_csv('submission.csv', index=False)

