import os
import time
import cv2
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

import timm
from PIL import Image

import matplotlib.pyplot as plt
from IPython.display import clear_output
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
device


train_csv_path = "/kaggle/input/automated-video-captioning/train.csv"
test_csv_path = "/kaggle/input/automated-video-captioning/test.csv"
train_dir = "/kaggle/input/automated-video-captioning/train_videos"
test_dir = "/kaggle/input/automated-video-captioning/test_videos"


class VideoDataset(Dataset):
    def __init__(
        self,
        video_paths,
        captions,
        tokenizer,
        feature_extractor_fn,
        feature_dim=768,   # например, ViT-base
        device="cuda",
        cache_dir="cache",
        max_len=50,
        interval=24,
    ):
        self.video_paths = video_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.feature_extractor_fn = feature_extractor_fn
        self.feature_dim = feature_dim
        self.device = device
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.interval = interval

        os.makedirs(cache_dir, exist_ok=True)
        self.encoded_videos = self.load_embeddings()

    def load_embeddings(self):
        encoded_videos = {}
        for video_path in tqdm(self.video_paths, desc="Extracting video features"):
            cache_path = os.path.join(
                self.cache_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.pkl"
            )
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    encoded_videos[video_path] = pickle.load(f)
            else:
                embeddings = self.feature_extractor_fn(
                    video_path, interval=self.interval, device=self.device
                )
                if embeddings is not None:
                    encoded_videos[video_path] = embeddings
                    with open(cache_path, "wb") as f:
                        pickle.dump(embeddings, f)
        return encoded_videos

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        embeddings = self.encoded_videos.get(video_path)

        if embeddings is None:
            embeddings = torch.zeros((1, self.feature_dim))

        caption = self.captions[idx]
        tokens = self.tokenizer(
            caption,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return embeddings, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)


def pad_collate_fn(batch):
    """
    batch: list of (video_emb [T_i, D], text_ids [L], attention_mask [L])
    """
    video_seqs, text_ids_list, attn_masks = zip(*batch)

    # Паддинг видео по T
    max_video_len = max(v.shape[0] for v in video_seqs)
    feat_dim = video_seqs[0].shape[1]
    padded_videos = torch.zeros(len(batch), max_video_len, feat_dim)

    for i, v in enumerate(video_seqs):
        padded_videos[i, :v.shape[0]] = v

    text_ids = torch.stack(text_ids_list)
    attn_masks = torch.stack(attn_masks)

    return padded_videos, text_ids, attn_masks


def extract_video_vit_features(video_path, interval=24, device="cuda"):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = vit_model(img_tensor)  # [1, 768]
                frames.append(features.squeeze(0).cpu())
        idx += 1
    cap.release()
    return torch.stack(frames) if frames else None


# VIT
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.head = torch.nn.Identity()  # cutting the classifier
vit_model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


class VideoCaptioner(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, vocab_size=None, dropout=0.2):
        super().__init__()
        self.video_gru = nn.GRU(768, hidden_dim, num_layers,
                                batch_first=True,
                                dropout=(dropout if num_layers>1 else 0))
        self.text_emb  = nn.Embedding(vocab_size, hidden_dim)
        self.text_gru  = nn.GRU(hidden_dim, hidden_dim, num_layers,
                                batch_first=True,
                                dropout=(dropout if num_layers>1 else 0))
        self.fc_out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, video_emb, text_ids):
        _, vh = self.video_gru(video_emb)  # [num_layers, B, H]
        te, _ = self.text_gru(self.text_emb(text_ids), vh)
        return self.fc_out(te)

    def generate(self, video_emb, tokenizer, max_len=50, device="cuda"):
        _, vh = self.video_gru(video_emb)
        gen = torch.full((video_emb.size(0),1),
                         tokenizer.bos_token_id,
                         dtype=torch.long, device=device)
        for _ in range(max_len):
            output, vh = self.text_gru(self.text_emb(gen[:,-1:]), vh)
            next_token = self.fc_out(output).argmax(-1)
            gen = torch.cat([gen, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        return gen



tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.bos_token = tokenizer.pad_token
tokenizer.bos_token_id = tokenizer.pad_token_id


class Trainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        self.train_losses = []
        self.val_losses = []

    def run_epoch(self, dataloader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = 0

        for video_emb, input_ids, attn_mask in dataloader:
            video_emb = video_emb.to(self.device)         # [B, T, D]
            input_ids = input_ids.to(self.device)         # [B, L]

            target_ids = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                output = self.model(video_emb, input_ids)  # [B, L-1, vocab]
                loss = self.criterion(output.view(-1, output.size(-1)), target_ids.reshape(-1))

                if train:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, epochs=10):
        for epoch in range(1, epochs + 1):
            start = time.time()

            train_loss = self.run_epoch(self.train_loader, train=True)
            val_loss = self.run_epoch(self.val_loader, train=False)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            clear_output(wait=True)
            self.plot_losses()
            print(f"Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time() - start:.2f}s")

    def plot_losses(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


df = pd.read_csv(train_csv_path)
video_paths = [os.path.join(train_dir, fname) for fname in df["file_name"]]
captions = df["caption"].tolist()

train_videos, val_videos, train_captions, val_captions = train_test_split(
    video_paths, captions, test_size=0.2, random_state=42
)


train_dataset = VideoDataset(
    train_videos, train_captions, tokenizer,
    feature_extractor_fn=extract_video_vit_features, # extracting
    feature_dim=768,
    cache_dir="/kaggle/working/cache/1",               # saving
    device=device,
)

val_dataset = VideoDataset(
    val_videos, val_captions, tokenizer,
    feature_extractor_fn=extract_video_vit_features,
    feature_dim=768,
    cache_dir="/kaggle/working/cache/1",
    device=device,
)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)


model = VideoCaptioner(
    hidden_dim=256,
    num_layers=2,
    vocab_size=tokenizer.vocab_size,
    dropout=0.2
).to(device)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=1e-4
)

# Запуск обучения
trainer.fit(epochs=50)



test_paths = list(map(lambda x: os.path.join(test_dir, x), next(os.walk(test_dir))[-1]))


model.eval()
caps = []

for vp in tqdm(test_paths):
    cache_path = os.path.join("cache/2", f"{os.path.basename(vp)}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            emb = pickle.load(f).unsqueeze(0).to(device)  # [1, T, D]
    else:
        emb = extract_video_vit_features(vp, interval=24, device=device)
        if emb is None:
            emb = torch.zeros((1, 10, 768), device=device)  # fallback
        else:
            emb = emb.unsqueeze(0).to(device)

    with torch.no_grad():
        gen_ids = model.generate(emb, tokenizer=tokenizer, max_len=50, device=device)  # [1, <=50]
        caption = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        caps.append(caption)


print(len(test_df.index),
      len(test_df["file_name"]),
      len(caps))


test_df = pd.read_csv(test_csv_path)

ans = pd.DataFrame(
    {
        "index": test_df.index,
        "file_name": test_df["file_name"],
        "caption": caps,
    }
)
ans.to_csv("submission.csv", index=False)
print("submission.csv saved!")




