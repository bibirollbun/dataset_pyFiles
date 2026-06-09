import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import pickle
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
train_csv_path = "/kaggle/input/automated-video-captioning/train.csv"
test_csv_path = "/kaggle/input/automated-video-captioning/test.csv"
train_dir = "/kaggle/input/automated-video-captioning/train_videos"
test_dir = "/kaggle/input/automated-video-captioning/test_videos"


def extract_resnet_embeddings(
    video_path,
    frame_interval=25,
    device="cuda",
):
    try:
        model = models.resnet50(pretrained=True).to(device)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
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


class ResNetGRUCaptioning(nn.Module):
    def __init__(
        self,
        embedding_dim=2048,
        hidden_dim=256,
        num_layers=2,
        vocab_size=50257,
        max_len=50,
        dropout=0.1,
    ):
        super(ResNetGRUCaptioning, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.video_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, video_emb, text_ids):
        batch_size = video_emb.size(0)

        _, video_hidden = self.video_gru(video_emb)

        text_emb = self.text_embedding(text_ids)
        output, _ = self.text_gru(text_emb, video_hidden)
        return self.fc_out(output)

    def generate(
        self,
        video_emb,
        tokenizer,
        max_len=50,
        device="cuda",
    ):
        batch_size = video_emb.size(0)
        _, video_hidden = self.video_gru(video_emb)

        generated = torch.full(
            (batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        for _ in range(max_len):
            text_emb = self.text_embedding(generated)
            output, video_hidden = self.text_gru(text_emb[:, -1:, :], video_hidden)
            logits = self.fc_out(output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == tokenizer.eos_token_id).any():
                break
        return generated


def collate_fn(batch):
    embeddings, text_ids = zip(*batch)
    padded_embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    padded_text_ids = nn.utils.rnn.pad_sequence(
        text_ids, batch_first=True, padding_value=0
    )
    return padded_embeddings, padded_text_ids


class VideoCaptionDataset(Dataset):
    def __init__(
        self,
        video_paths,
        captions,
        tokenizer,
        device="cuda",
        cache_dir="embeddings",
        max_len=50,
    ):
        self.video_paths = video_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.device = device
        self.cache_dir = cache_dir
        self.max_len = max_len
        os.makedirs(cache_dir, exist_ok=True)
        self.encoded_videos = self.load_embeddings()

    def load_embeddings(self):
        encoded_videos = {}
        for video_path in tqdm(self.video_paths, desc="Loading embeddings"):
            cache_path = os.path.join(
                self.cache_dir, f"{os.path.basename(video_path)}.pkl"
            )
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    encoded_videos[video_path] = pickle.load(f)
            else:
                embeddings = extract_resnet_embeddings(
                    video_path, frame_interval=25, device=self.device
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
            embeddings = torch.zeros((1, 2048))
        caption = self.captions[idx]
        tokens = self.tokenizer(
            caption,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_ids = tokens["input_ids"].squeeze(0)
        return embeddings, text_ids


def compute_bleu_score(pred_texts, true_texts):
    scores = []
    for pred, true in zip(pred_texts, true_texts):
        pred_tokens = pred.split()
        true_tokens = true.split()
        score = sentence_bleu(
            [true_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25)
        )
        scores.append(score)
    return np.mean(scores)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token_id = tokenizer.eos_token_id

train_df = pd.read_csv(train_csv_path)
video_paths = [os.path.join(train_dir, name) for name in train_df["file_name"]]
captions = train_df["caption"].tolist()

train_idx, val_idx = train_test_split(
    range(len(video_paths)), test_size=0.2, random_state=42
)
train_paths = [video_paths[i] for i in train_idx]
train_captions = [captions[i] for i in train_idx]
val_paths = [video_paths[i] for i in val_idx]
val_captions = [captions[i] for i in val_idx]

train_dataset = VideoCaptionDataset(
    train_paths, train_captions, tokenizer, device=device, cache_dir="embeddings_train"
)
val_dataset = VideoCaptionDataset(
    val_paths, val_captions, tokenizer, device=device, cache_dir="embeddings_val"
)
train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


# !rm -rf /kaggle/working/embeddings_train
# !rm -rf /kaggle/working/embeddings_val


model = ResNetGRUCaptioning(
    vocab_size=tokenizer.vocab_size, hidden_dim=256, dropout=0.2, num_layers=2
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=2e-4)
num_epochs = 50

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    train_loss = 0
    train_total = 0

    for embeddings, text_ids in train_loader:
        embeddings, text_ids = embeddings.to(device), text_ids.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings, text_ids[:, :-1])
        loss = criterion(
            outputs.view(-1, tokenizer.vocab_size), text_ids[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * embeddings.size(0)
        train_total += embeddings.size(0)

    train_loss /= train_total

    model.eval()
    val_loss = 0
    val_total = 0
    pred_captions = []
    true_captions = val_captions
    with torch.no_grad():
        for embeddings, text_ids in val_loader:
            embeddings, text_ids = embeddings.to(device), text_ids.to(device)
            outputs = model(embeddings, text_ids[:, :-1])
            loss = criterion(
                outputs.view(-1, tokenizer.vocab_size), text_ids[:, 1:].reshape(-1)
            )
            val_loss += loss.item() * embeddings.size(0)
            val_total += embeddings.size(0)

            generated_ids = model.generate(
                embeddings, tokenizer, max_len=50, device=device
            )
            for ids in generated_ids:
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                pred_captions.append(caption)

    val_loss /= val_total
    bleu_score = compute_bleu_score(pred_captions, true_captions)

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, BLEU Score: {bleu_score:.4f}")


test_df = pd.read_csv(test_csv_path)
test_paths = [os.path.join(test_dir, name) for name in test_df["file_name"]]
test_dataset = VideoCaptionDataset(
    test_paths,
    [""] * len(test_paths),
    tokenizer,
    device=device,
    cache_dir="embeddings_test",
)
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
)


model.eval()
pred_captions = []
with torch.no_grad():
    for embeddings, _ in test_loader:
        embeddings = embeddings.to(device)
        generated_ids = model.generate(embeddings, tokenizer, max_len=50, device=device)
        for ids in generated_ids:
            caption = tokenizer.decode(ids, skip_special_tokens=True)
            pred_captions.append(caption)


submission = pd.DataFrame(
    {
        "index": test_df.index,
        "file_name": test_df["file_name"],
        "caption": pred_captions,
    }
)
submission.to_csv("submission.csv", index=False)


!rm -rf /kaggle/working/embeddings_train
!rm -rf /kaggle/working/embeddings_val
!rm -rf /kaggle/working/embeddings_test

