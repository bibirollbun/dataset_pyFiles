import gc
gc.collect()
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
import random
from sklearn.utils import shuffle
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

from torch.optim import AdamW
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import shap

!pip install -q langdetect
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))

!pip install -q pyspellchecker
from spellchecker import SpellChecker
spell_checker = SpellChecker()

import spacy
nlp = spacy.load("en_core_web_sm")
!pip install -q textstat
import textstat
from collections import Counter
from math import log2
!pip install -q langdetect
from langdetect import detect, DetectorFactory
from transformers import AutoModel, AutoConfig
from sklearn.feature_extraction.text import TfidfVectorizer
import random

from sklearn.feature_extraction.text import TfidfVectorizer
import contextlib
import sys
from typing import List, Tuple
import zlib
from scipy.spatial.distance import euclidean
embedder = SentenceTransformer("microsoft/deberta-v3-base") # "BAAI/bge-small-en-v1.5"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

data_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

ATTENTION_EPOCHS = 20
MARGINLOSS_EPOCHS = 15
SIAMESE_EPOCHS = 70


def read_texts_from_dir(dir_path: str) -> pd.DataFrame:
    """
    Reads `file_1.txt` and `file_2.txt` from each `article_XXXX` subfolder
    and returns a DataFrame containing paired text samples.

    Each subfolder is expected to follow the structure:
        article_0001/
            ├─ file_1.txt
            └─ file_2.txt

    Args:
        dir_path (str): Root path to the data directory (train or test),
                        containing subfolders named as `article_XXXX`.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'id': Integer article ID extracted from folder name
            - 'file_1': Content of `file_1.txt`
            - 'file_2': Content of `file_2.txt`

    Notes:
        - Skips folders with missing or malformed files.
        - Uses tqdm for progress tracking.
        - Prints error summary and last error encountered.
        - Set `DEBUG = True` globally to raise exceptions directly.
    """
    data = []
    error_count = 0
    last_error = None

    for folder_name in tqdm(sorted(os.listdir(dir_path)), desc="Reading folders"):
        folder_path = os.path.join(dir_path, folder_name)
        pos_path = os.path.join(folder_path, 'file_1.txt')
        neg_path = os.path.join(folder_path, 'file_2.txt')

        try:
            with open(pos_path, 'r', encoding='utf-8', errors='replace') as f1:
                text1 = f1.read().strip()
            with open(neg_path, 'r', encoding='utf-8', errors='replace') as f2:
                text2 = f2.read().strip()

            index = int(folder_name.split('_')[-1])
            data.append((index, text1, text2))

        except (FileNotFoundError, ValueError, OSError) as e:
            error_count += 1
            last_error = e
            if globals().get('DEBUG', False):
                raise e

    def clrd(msg, kind='info'):
        """Optional color print wrapper (can remove if unused)."""
        return f"[{kind.upper()}] {msg}"

    print(f"Read {clrd(len(data), 'ok')} records with {clrd(error_count, 'error')} errors")
    if error_count > 0:
        print(clrd('Last Error:', 'warn'), last_error)

    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2'])

train = read_texts_from_dir(train_path)
test = read_texts_from_dir(test_path)
train = train.merge(pd.read_csv(data_dir), how='inner', on='id')


train.head(3)


test.head(1)


def analyze_token_lengths(df_list, tokenizer, cols=("file_1", "file_2")):
    """
    Analyze token lengths across one or more DataFrames.

    Args:
        df_list: list of DataFrames (e.g., [train, test])
        tokenizer: HuggingFace tokenizer
        cols: columns in each df that contain text

    Returns:
        dict with stats and recommended max_length
    """
    all_lengths = []

    for df in df_list:
        for col in cols:
            texts = df[col].dropna().astype(str).tolist()
            enc = tokenizer(
                texts,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            lengths = [len(ids) for ids in enc["input_ids"]]
            all_lengths.extend(lengths)

    arr = np.array(all_lengths)

    stats = {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
    }

    stats["recommended_max_length"] = stats["p99"]

    return stats

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

stats = analyze_token_lengths([train, test], tokenizer)
print(stats)


class RealFakeOriginalDataset(Dataset):
    def __init__(self, df):
        """
        Args:
            df: DataFrame with either:
                ['file_1', 'file_2', 'real_text_id']  (original)
            or
                ['text1', 'text2', 'label']  (expanded DPR-style)
        """
        if "file_1" in df.columns and "file_2" in df.columns:
            self.text1 = df["file_1"].tolist()
            self.text2 = df["file_2"].tolist()
            self.labels = [(1 if rid == 1 else 0) for rid in df["real_text_id"]]
        elif "text1" in df.columns and "text2" in df.columns:
            self.text1 = df["text1"].tolist()
            self.text2 = df["text2"].tolist()
            self.labels = df["label"].astype(int).tolist()
        else:
            raise ValueError("DataFrame must have either (file_1, file_2, real_text_id) or (text1, text2, label)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text1[idx], self.text2[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def masked_mean_pool(last_hidden_state, attention_mask):

    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

class AttentionRealFakeClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", dropout=0.35, num_heads=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        self.h = cfg.hidden_size

        # bi-directional cross-attention
        self.cross12 = nn.MultiheadAttention(self.h, num_heads, batch_first=True)
        self.cross21 = nn.MultiheadAttention(self.h, num_heads, batch_first=True)

        # residual + layernorm around each cross-attn
        self.ln12 = nn.LayerNorm(self.h)
        self.ln21 = nn.LayerNorm(self.h)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(self.h, self.h),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        fusion_in = 4 * self.h + 1
        self.gate = nn.Sequential(
            nn.Linear(fusion_in, 2 * self.h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.h, self.h),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.h, 2)
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        
        out1 = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state
        out2 = self.encoder(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state

        c12, _ = self.cross12(query=out1, key=out2, value=out2)
        c21, _ = self.cross21(query=out2, key=out1, value=out1)
        c12 = self.ln12(out1 + self.dropout(c12))
        c21 = self.ln21(out2 + self.dropout(c21))

        p1 = masked_mean_pool(c12, attention_mask1)
        p2 = masked_mean_pool(c21, attention_mask2)

        p1 = self.proj(p1)
        p2 = self.proj(p2)

        diff = torch.abs(p1 - p2)
        prod = p1 * p2
        cos = F.cosine_similarity(p1, p2).unsqueeze(-1)  # [B,1]

        fused = torch.cat([p1, p2, diff, prod, cos], dim=-1)
        fused = self.gate(fused)
        logits = self.classifier(fused)
        return logits

class AttentionTrainer:
    def __init__(self, 
                 model, 
                 tokenizer, 
                 train_loader, 
                 val_loader,
                 epochs=ATTENTION_EPOCHS,
                 lr=2e-5,
                 device=None,
                 patience=3, 
                 seed=42):
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []

        self.best_state = None
        self.best_val_loss = float("inf")
        self.counter = 0
        self.patience = patience

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for text1, text2, labels in self.train_loader:
                inputs1 = self.tokenizer(text1, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                inputs2 = self.tokenizer(text2, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs1.input_ids, inputs1.attention_mask, inputs2.input_ids, inputs2.attention_mask)
                loss = self.loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del inputs1, inputs2, labels, logits
                torch.cuda.empty_cache()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            val_loss, val_acc = self.evaluate()
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break
                    
            torch.cuda.empty_cache()
            gc.collect()

        if self.best_state:
            self.model.load_state_dict(self.best_state)

        self.plot_loss()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for text1, text2, labels in self.val_loader:
                inputs1 = self.tokenizer(text1, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                inputs2 = self.tokenizer(text2, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs1.input_ids, inputs1.attention_mask, inputs2.input_ids, inputs2.attention_mask)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        epochs_ran = len(self.train_losses)
        plt.plot(range(1, epochs_ran + 1), self.train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, epochs_ran + 1), self.val_losses, label="Val Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


train_df, val_df = train_test_split(train, test_size=0.15, stratify=train["real_text_id"], random_state=42)

train_dataset = RealFakeOriginalDataset(train_df)
val_dataset = RealFakeOriginalDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
model = AttentionRealFakeClassifier(model_name="microsoft/deberta-v3-base")

trainer = AttentionTrainer(model, tokenizer, train_loader, val_loader, epochs=ATTENTION_EPOCHS)
trainer.train()


def generate_attention_predictions(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """
    Generate attention-based predictions for either train or test set.

    Args:
        df: DataFrame with at least ['id', 'file_1', 'file_2']
            For training, include 'real_text_id' to compute true labels.
        model: Trained AttentionRealFakeClassifier
        tokenizer: Tokenizer used (e.g., DeBERTa)
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'

    Returns:
        DataFrame with columns: ['id', 'real_score', 'confidence', 'label', 'true_label' (if available)]
    """
    model.eval()
    model.to(device)

    ids, real_scores, confidences, labels, true_labels = [], [], [], [], []

    for i in tqdm(range(0, len(df), batch_size), desc="Running inference"):
        batch = df.iloc[i:i+batch_size]

        text1_batch = list(batch["file_1"])
        text2_batch = list(batch["file_2"])
        ids_batch = list(batch["id"])

        inputs1 = tokenizer(text1_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        inputs2 = tokenizer(text2_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            logits = model(inputs1.input_ids, inputs1.attention_mask, inputs2.input_ids, inputs2.attention_mask)
            probs = torch.softmax(logits, dim=1)

        prob_file1_real = probs[:, 1].cpu().numpy()
        pred_labels = (prob_file1_real >= 0.5).astype(int)
        confidence = np.where(pred_labels == 1, prob_file1_real, 1 - prob_file1_real)

        ids.extend(ids_batch)
        real_scores.extend(prob_file1_real)
        confidences.extend(confidence)
        labels.extend(pred_labels)

        if "real_text_id" in batch.columns:
            batch_true = [1 if lbl == 1 else 2 for lbl in batch["real_text_id"]]
            true_labels.extend(batch_true)

    result = pd.DataFrame({
        "id": ids,
        "confidence": confidences,
        "real_text_id": labels
    })

    if true_labels:
        result["label"] = true_labels

    return result


train_prediction_attention = generate_attention_predictions(train, model, tokenizer)
train_prediction_attention.head(7)


submission_df_attention = generate_attention_predictions(test, model, tokenizer)
submission_df_attention.head(7)


del model, tokenizer
gc.collect()


def build_dpr_style_dataset(
    df: pd.DataFrame,
    embedder=embedder,
    num_hard_neg=2,
    random_state: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """
    Expand dataset with DPR-style positives + hard negatives.
    Reference: Dense Passage Retrieval for Open-Domain Question Answering

    Args:
        df: DataFrame with ['id', 'file_1', 'file_2', 'real_text_id']
        embedder: SentenceTransformer or encoder with .encode
        num_hard_neg: number of hardest negatives per real text
        random_state: reproducibility
        device: cuda or cpu

    Returns:
        Expanded DataFrame with ['id', 'file_1', 'file_2', 'real_text_id']
    """

    random.seed(random_state)
    torch.manual_seed(random_state)

    if embedder is None:
        embedder = SentenceTransformer("microsoft/deberta-v3-base", device=device)

    all_texts = pd.concat([df["file_1"], df["file_2"]], axis=0).unique().tolist()
    text_to_emb = {
        t: embedder.encode(t, convert_to_tensor=True, device=device, show_progress_bar=False)
        for t in all_texts
    }

    expanded_rows = []

    for _, row in df.iterrows():
        real_text = row["file_1"] if row["real_text_id"] == 1 else row["file_2"]
        fake_text = row["file_2"] if row["real_text_id"] == 1 else row["file_1"]

        real_emb = text_to_emb[real_text]

        sims = []
        for t, emb in text_to_emb.items():
            if t == real_text:
                continue
            sim = util.cos_sim(real_emb, emb).item()
            sims.append((t, sim))

        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        hard_negs = [t for t, _ in sims if t != fake_text][:num_hard_neg]

        for neg in hard_negs:
            expanded_rows.append({
                "id": row["id"],
                "file_1": real_text,
                "file_2": neg,
                "real_text_id": 1  # file_1 is real
            })
            expanded_rows.append({
                "id": row["id"],
                "file_1": neg,
                "file_2": real_text,
                "real_text_id": 2  # file_2 is real
            })

        expanded_rows.append(row.to_dict())

    expanded_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    return expanded_df


class DPRTripletDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        """
        Args:
            df: DataFrame with columns ['file_1','file_2','real_text_id']
                or already expanded DPR-style ['text1','text2','label'].
            tokenizer: HuggingFace tokenizer.
        """
        self.tokenizer = tokenizer
        self.samples = []

        for _, row in df.iterrows():
            real = row["file_1"] if row["real_text_id"] == 1 else row["file_2"]
            fake = row["file_2"] if row["real_text_id"] == 1 else row["file_1"]

            self.samples.append((real, real, fake))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor, pos, neg = self.samples[idx]
        return anchor, pos, neg

def masked_mean_pool(last_hidden_state, attention_mask):

    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom
    
class RealFakeEncoder(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        self.h = cfg.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = masked_mean_pool(out, attention_mask)
        return self.proj(pooled)

    
class MarginLossTrainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, lr=2e-5, epochs=10, device=None, margin=0.3, patience=3):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.best_val_loss = float("inf")
        self.best_state = None
        self.patience = patience
        self.counter = 0

    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        return self.model(tokens.input_ids, tokens.attention_mask)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for anchor, pos, neg in self.train_loader:
                anchor_emb = self.encode(list(anchor))
                pos_emb = self.encode(list(pos))
                neg_emb = self.encode(list(neg))

                sim_pos = F.cosine_similarity(anchor_emb, pos_emb)
                sim_neg = F.cosine_similarity(anchor_emb, neg_emb)

                target = torch.ones_like(sim_pos).to(self.device)
                loss = self.loss_fn(sim_pos, sim_neg, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                correct += (sim_pos > sim_neg).sum().item()
                total += sim_pos.size(0)

            avg_train_loss = total_loss / len(self.train_loader)
            train_acc = correct / total
            self.train_losses.append(avg_train_loss)
            self.train_accs.append(train_acc)

            val_loss, val_acc = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)


    def evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for anchor, pos, neg in self.val_loader:
                anchor_emb = self.encode(list(anchor))
                pos_emb = self.encode(list(pos))
                neg_emb = self.encode(list(neg))

                sim_pos = F.cosine_similarity(anchor_emb, pos_emb)
                sim_neg = F.cosine_similarity(anchor_emb, neg_emb)

                target = torch.ones_like(sim_pos).to(self.device)
                loss = self.loss_fn(sim_pos, sim_neg, target)
                total_loss += loss.item()

                correct += (sim_pos > sim_neg).sum().item()
                total += sim_pos.size(0)

        return total_loss / len(self.val_loader), correct / total


train_df, val_df = train_test_split(train, test_size=0.15, stratify=train["real_text_id"], random_state=42)
train_df = build_dpr_style_dataset(train_df, num_hard_neg=3)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

train_dataset = DPRTripletDataset(train_df, tokenizer=tokenizer)
val_dataset = DPRTripletDataset(val_df, tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = RealFakeEncoder()

trainer = MarginLossTrainer(model, tokenizer, train_loader, val_loader, epochs=MARGINLOSS_EPOCHS, margin=0.3)
trainer.train()


from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def _encode_texts_in_batches(texts, model, tokenizer, device, batch_size=16, max_length=512):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        emb = model(toks.input_ids, toks.attention_mask)
        embs.append(emb)
    return torch.cat(embs, dim=0) if len(embs) > 0 else torch.empty(0, model.h, device=device)

@torch.no_grad()
def _build_prototypes_from_train(df_train, model, tokenizer, device, batch_size=16, max_length=512):
    """
    Build 'real' and 'fake' prototypes from a labeled train dataframe with columns:
      ['id','file_1','file_2','real_text_id']
    """
    real_texts, fake_texts = [], []
    for _, r in df_train.iterrows():
        if int(r["real_text_id"]) == 1:
            real_texts.append(r["file_1"])
            fake_texts.append(r["file_2"])
        else:
            real_texts.append(r["file_2"])
            fake_texts.append(r["file_1"])

    real_embs = _encode_texts_in_batches(real_texts, model, tokenizer, device, batch_size, max_length)
    fake_embs = _encode_texts_in_batches(fake_texts, model, tokenizer, device, batch_size, max_length)

    real_proto = F.normalize(real_embs.mean(dim=0, keepdim=True), p=2, dim=1) if real_embs.numel() > 0 else None
    fake_proto = F.normalize(fake_embs.mean(dim=0, keepdim=True), p=2, dim=1) if fake_embs.numel() > 0 else None
    return real_proto, fake_proto

def generate_margin_predictions(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    real_prototype: torch.Tensor = None,
    fake_prototype: torch.Tensor = None,
    max_length: int = 512
) -> pd.DataFrame:
    """
    Inference for the Margin/Triplet encoder using prototype scoring.

    Returns DataFrame with:
      - id
      - real_text_id (prediction: 1 or 2)
      - confidence (probability of the chosen side; in (0.5, 1))
      - label (ground-truth 1 or 2) if available in df
    """
    model.eval()
    model.to(device)

    real_proto, fake_proto = real_prototype, fake_prototype
    if real_proto is None and "real_text_id" in df.columns:
        real_proto, fake_proto = _build_prototypes_from_train(
            df, model, tokenizer, device, batch_size=batch_size, max_length=max_length
        )

    if real_proto is None:
        raise ValueError("real_prototype is required. Pass it explicitly for TEST, or provide a labeled df to build from.")

    real_proto = F.normalize(real_proto.to(device), p=2, dim=1)
    if fake_proto is not None:
        fake_proto = F.normalize(fake_proto.to(device), p=2, dim=1)

    ids_all, preds_all, conf_all, labels_all = [], [], [], []

    for i in tqdm(range(0, len(df), batch_size), desc="Margin inference"):
        batch = df.iloc[i:i+batch_size]
        ids_batch = batch["id"].tolist()

        emb1 = _encode_texts_in_batches(list(batch["file_1"]), model, tokenizer, device, batch_size, max_length)
        emb2 = _encode_texts_in_batches(list(batch["file_2"]), model, tokenizer, device, batch_size, max_length)

        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        s1_real = F.cosine_similarity(emb1, real_proto.expand_as(emb1))
        s2_real = F.cosine_similarity(emb2, real_proto.expand_as(emb2))

        if fake_proto is not None:
            s1_fake = F.cosine_similarity(emb1, fake_proto.expand_as(emb1))
            s2_fake = F.cosine_similarity(emb2, fake_proto.expand_as(emb2))
            score1 = s1_real - s1_fake
            score2 = s2_real - s2_fake
        else:
            score1 = s1_real
            score2 = s2_real

        diff = (score1 - score2)
        pred_real1 = (diff >= 0).long()
        pred_ids = (pred_real1.cpu().numpy() + 1)
        conf = torch.sigmoid(torch.abs(diff)).cpu().numpy()

        ids_all.extend(ids_batch)
        preds_all.extend(pred_ids)
        conf_all.extend(conf.tolist())

        if "real_text_id" in batch.columns:
            labels_all.extend(batch["real_text_id"].astype(int).tolist())

    out = pd.DataFrame({
        "id": ids_all,
        "real_text_id": np.where(preds_all == 1, 1, 2),
        "confidence": conf_all
    })
    if "real_text_id" in df.columns:
        out["label"] = labels_all

    return out

train_prediction_margin = generate_margin_predictions(train, model, tokenizer)
print(train_prediction_margin.head(7))


real_proto, fake_proto = _build_prototypes_from_train(train, model, tokenizer, device)

submission_df_margin = generate_margin_predictions(
    test, model, tokenizer, batch_size=4,
    real_prototype=real_proto, fake_prototype=fake_proto
)
print(submission_df_margin.head(7))


del model, tokenizer
gc.collect()


def create_contrastive_dataset_(
    df: pd.DataFrame,
    embedder=None,
    num_cross_pos: int = 2,
    num_hard_neg: int = 2,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """
    Build a dataset for Siamese+Projection with positives and semantic hard negatives.
    """
    random.seed(seed)

    if embedder is None:
        embedder = SentenceTransformer("microsoft/deberta-v3-base", device=device)

    real_texts, fake_texts = [], []
    for _, row in df.iterrows():
        real = row["file_1"] if row["real_text_id"] == 1 else row["file_2"]
        fake = row["file_2"] if row["real_text_id"] == 1 else row["file_1"]
        real_texts.append(real)
        fake_texts.append(fake)

    all_texts = list(set(real_texts + fake_texts))
    emb_map = {t: embedder.encode(t, convert_to_tensor=True, normalize_embeddings=True) for t in all_texts}

    records = []

    for idx, anchor in enumerate(real_texts):
        records.append({"text1": anchor, "text2": anchor, "label": 1})
        others = real_texts[:idx] + real_texts[idx+1:]
        for pos in random.sample(others, min(num_cross_pos, len(others))):
            records.append({"text1": anchor, "text2": pos, "label": 1})

    for idx, anchor in enumerate(fake_texts):
        records.append({"text1": anchor, "text2": anchor, "label": 1})
        others = fake_texts[:idx] + fake_texts[idx+1:]
        for pos in random.sample(others, min(num_cross_pos, len(others))):
            records.append({"text1": anchor, "text2": pos, "label": 1})

    for real in real_texts:
        sims = [(f, util.cos_sim(emb_map[real], emb_map[f]).item()) for f in fake_texts]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:num_hard_neg]
        for fake, _ in sims:
            records.append({"text1": real, "text2": fake, "label": 0})

    for fake in fake_texts:
        sims = [(r, util.cos_sim(emb_map[fake], emb_map[r]).item()) for r in real_texts]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:num_hard_neg]
        for real, _ in sims:
            records.append({"text1": fake, "text2": real, "label": 0})

    return pd.DataFrame(records)


def create_contrastive_dataset(
    df: pd.DataFrame,
    num_cross_pos: int = 2,
    num_hard_neg: int = 2,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Enhanced contrastive dataset generator with cross-sample positives and hard negatives.

    Args:
        df (pd.DataFrame): Original dataset with 'file_1', 'file_2', 'real_text_id'.
        num_cross_pos (int): Number of cross-example positives per text.
        num_hard_neg (int): Number of cross-example hard negatives per text.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with columns ['text1', 'text2', 'label'].
    """
    random.seed(seed)
    records = []

    # Build real/fake pools
    real_texts = []
    fake_texts = []

    for _, row in df.iterrows():
        real = row["file_1"] if row["real_text_id"] == 1 else row["file_2"]
        fake = row["file_2"] if row["real_text_id"] == 1 else row["file_1"]

        real_texts.append(real)
        fake_texts.append(fake)

        # Basic positive/negative pairs
        records.extend([
            {"text1": real, "text2": real, "label": 1},
            {"text1": fake, "text2": fake, "label": 1},
            {"text1": real, "text2": fake, "label": 0},
            {"text1": fake, "text2": real, "label": 0},
        ])

    # Shuffle for diverse sampling
    real_texts = shuffle(real_texts, random_state=seed)
    fake_texts = shuffle(fake_texts, random_state=seed)

    # Add cross-sample positives
    for idx, anchor in enumerate(real_texts):
        others = real_texts[:idx] + real_texts[idx+1:]
        sampled = random.sample(others, min(num_cross_pos, len(others)))
        for pos in sampled:
            records.append({"text1": anchor, "text2": pos, "label": 1})

    for idx, anchor in enumerate(fake_texts):
        others = fake_texts[:idx] + fake_texts[idx+1:]
        sampled = random.sample(others, min(num_cross_pos, len(others)))
        for pos in sampled:
            records.append({"text1": anchor, "text2": pos, "label": 1})

    # Add cross-sample hard negatives
    for real_text in real_texts:
        sampled_fakes = random.sample(fake_texts, min(num_hard_neg, len(fake_texts)))
        for fake_text in sampled_fakes:
            records.append({"text1": real_text, "text2": fake_text, "label": 0})

    for fake_text in fake_texts:
        sampled_reals = random.sample(real_texts, min(num_hard_neg, len(real_texts)))
        for real_text in sampled_reals:
            records.append({"text1": fake_text, "text2": real_text, "label": 0})

    return pd.DataFrame(records)

class ContrastiveTextDataset(Dataset):
    def __init__(self, df):
        self.text1 = df["text1"].tolist()
        self.text2 = df["text2"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text1[idx], self.text2[idx], torch.tensor(self.labels[idx], dtype=torch.float)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextEncoder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, freeze_encoder=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token
        return cls_embedding

class SiameseNetworkWithProjection(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        projection_dim: int = 128,
        freeze_encoder: bool = True,
        dropout: float = 0.2
    ):
        """
        Siamese Network with BGE encoder + projection head.

        Args:
            model_name (str): Pretrained SentenceTransformer model name.
            projection_dim (int): Output dimension of the projection head.
            freeze_encoder (bool): Whether to freeze the encoder weights.
            dropout (float): Dropout rate for projection head.
        """
        super().__init__()
        self.encoder = SentenceTransformer(model_name)

        encoder_dim = self.encoder.get_sentence_embedding_dimension()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, projection_dim)
        )

    def forward(self, text1, text2):
        """
        Args:
            text1 (List[str]): List of text inputs (anchor or query)
            text2 (List[str]): List of paired text inputs (positive or negative)
        
        Returns:
            Tuple[Tensor, Tensor]: Projected embeddings of both inputs
        """
        emb1 = self.encoder.encode(text1, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        emb2 = self.encoder.encode(text2, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

        proj1 = self.projection_head(emb1)
        proj2 = self.projection_head(emb2)

        return proj1, proj2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance
        distance = torch.norm(output1 - output2, dim=1)
        # Contrastive loss formula
        loss = 0.5 * (label * distance.pow(2) + (1 - label) * torch.clamp(self.margin - distance, min=0.0).pow(2))
        return loss.mean()

class SiameseTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=2e-5,
        margin=1.0,
        epochs=10,
        patience=5,
        device=None,
        save_path="best_siamese_model.pt",
        use_amp=False
    ):
        """
        Trainer for Siamese Network with Contrastive Loss.

        Args:
            model (nn.Module): Siamese model with projection.
            train_loader (DataLoader): Training DataLoader.
            val_loader (DataLoader): Optional validation DataLoader.
            lr (float): Learning rate.
            margin (float): Margin for contrastive loss.
            epochs (int): Number of training epochs.
            patience (int): Early stopping patience.
            device (str): 'cuda' or 'cpu'.
            save_path (str): Where to save the best model.
            use_amp (bool): Enable mixed precision training.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.patience = patience
        self.save_path = save_path
        self.use_amp = use_amp

        self.model.to(self.device)
        self.loss_fn = ContrastiveLoss(margin=margin)
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.train_losses = []
        self.val_losses = []


    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False, disable=True)
            for batch in pbar:
                text1, text2, labels = batch
                labels = labels.to(self.device)

                with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                    emb1, emb2 = self.model(text1, text2)
                    loss = self.loss_fn(emb1, emb2, labels)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)

            val_loss = None
            if self.val_loader:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    # print("Saving best model...")
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print("Early stopping.")
                        break
            else:
                torch.save(self.model.state_dict(), self.save_path)

            if epoch % 5 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch:>2} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch:>2} | Train Loss: {avg_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for text1, text2, labels in self.val_loader:
                labels = labels.to(self.device)
                emb1, emb2 = self.model(text1, text2)
                loss = self.loss_fn(emb1, emb2, labels)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def load_best_model(self):
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
            print("Best model loaded.")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {self.save_path}")

    def plot_losses(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

MODEL_NAME = "BAAI/bge-large-en-v1.5"

model = SiameseNetworkWithProjection(
    model_name=MODEL_NAME,
    projection_dim=128,
    freeze_encoder=True,
    dropout=0.25
)

train_orig, val_orig = train_test_split(train, test_size=0.2, random_state=42)

contrastive_df = create_contrastive_dataset(train_orig)
contrastive_df_train = create_contrastive_dataset(train_orig)
contrastive_df_val = create_contrastive_dataset(val_orig)

train_dataset = ContrastiveTextDataset(contrastive_df_train)
val_dataset = ContrastiveTextDataset(contrastive_df_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

trainer = SiameseTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=SIAMESE_EPOCHS,
    lr=2e-5,
    margin=1.0,
    save_path="siamese_best.pt"
)

trainer.train()


trainer.plot_losses()


def sentence_embedding(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Efficiently computes BGE embeddings for a list of texts using batching.
    """
    return embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def compute_entropy(text: str) -> float:
    if not text: return 0.0
    prob = [freq / len(text) for freq in Counter(text).values()]
    return -sum(p * log2(p) for p in prob)

def ngram_repetition(text, n=3):
    words = word_tokenize(text.lower())
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    count = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(1 for v in count.values() if v > 1)
    return repeated / total if total > 0 else 0

def embedding_coherence(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2: return 1.0
    embeddings = np.vstack([nlp(sent).vector for sent in sentences if nlp(sent).has_vector])
    if embeddings.shape[0] < 2: return 1.0
    sims = cosine_similarity(embeddings)
    tril = sims[np.tril_indices_from(sims, k=-1)]
    return np.mean(tril)

spell_checker = SpellChecker()

def insert_space_in_compounds(text):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # e.g., "highResolution" → "high Resolution"

def count_spelling_errors(words):
    misspelled = spell_checker.unknown(words)
    return len(misspelled)

def extract_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:

    def detect_script_ratios(text: str) -> dict:
        total = len(text)
        if total == 0:
            return {"english_ratio": 0.0, "latin_ratio": 0.0}
        english_count = len(re.findall(r'[a-zA-Z]', text))
        latin_count = len(re.findall(r'[^\x00-\x7F]', text))
        return {
            "english_ratio": english_count / total,
            "latin_ratio": latin_count / total
        }

    def feature_row(text: str) -> dict:
        text = str(text)
        words = word_tokenize(text)
        word_count = len(words)
        num_chars = len(text)
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        punct_count = sum(1 for c in text if c in string.punctuation)
        emdash_count = text.count("—")
        long_words = sum(1 for w in words if len(w) > 6)
        short_words = sum(1 for w in words if len(w) <= 3)
        stopword_count = sum(1 for w in words if w.lower() in stop_words)
        unique_words = len(set(words))
        upper_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0
        ent = compute_entropy(text)
        spelling_errors = count_spelling_errors([w for w in words if w.isalpha()])
        # syllables_per_word = textstat.syllable_count(text) / word_count if word_count else 0
        type_token_ratio_sqrt = unique_words / (word_count ** 0.5) if word_count else 0
        script_ratios = detect_script_ratios(text)

        doc = nlp(text)
        ner_count = len(doc.ents)
        compression_ratio = len(zlib.compress(text.encode())) / len(text.encode()) if text else 1.0
        pos_counts = Counter([token.pos_ for token in doc])
        total_pos = sum(pos_counts.values()) or 1
        noun_ratio = pos_counts.get("NOUN", 0) / total_pos
        verb_ratio = pos_counts.get("VERB", 0) / total_pos
        adj_ratio = pos_counts.get("ADJ", 0) / total_pos
        adv_ratio = pos_counts.get("ADV", 0) / total_pos
        dep_depths = [len(list(token.ancestors)) for token in doc if token.head != token]
        avg_dep_depth = np.mean(dep_depths) if dep_depths else 0.0

        emb = sentence_embedding(text)
        emb_dict = {f"sent_emb_{i}": emb[i] for i in range(len(emb))}
        avg_grad_delta = float(np.mean(np.abs(np.diff(emb)))) if len(emb) > 1 else 0.0


        return {
            'char_count': num_chars,
            'word_count': word_count,
            'sentence_count': num_sentences,
            'avg_word_length': avg_word_len,
            'avg_sentence_length': avg_sentence_len,
            'unique_word_count': unique_words,
            'ttr': unique_words / word_count if word_count else 0,
            'stopword_count': stopword_count,
            'stopword_ratio': stopword_count / word_count if word_count else 0,
            'punctuation_count': punct_count,
            'english_ratio': script_ratios["english_ratio"],
            'latin_ratio': script_ratios["latin_ratio"],
            'digit_count': digit_count,
            'uppercase_ratio': upper_count / num_chars if num_chars else 0,
            'long_word_count': long_words,
            'short_word_count': short_words,
            'type_token_ratio_sqrt': type_token_ratio_sqrt,
            'entropy': ent,
            'emdash_count': emdash_count,
            'ngram_repetition': ngram_repetition(text),
            'ner_count': ner_count,
            'spelling_errors': spelling_errors,
            'embedding_coherence': embedding_coherence(text),
            'compression_ratio': compression_ratio,
            'noun_ratio': noun_ratio,
            'verb_ratio': verb_ratio,
            'adj_ratio': adj_ratio,
            'adv_ratio': adv_ratio,
            'avg_dependency_depth': avg_dep_depth,
            'embedding_gradient_delta': avg_grad_delta,
            **emb_dict
        }

    features = df[text_col].apply(feature_row)
    return pd.concat([df.drop(columns=[text_col]), features.apply(pd.Series)], axis=1)


def fit_tfidf_vectorizer(train_df: pd.DataFrame, text_col: str, max_features: int = 1000, ngram_range=(1, 2)) -> TfidfVectorizer:
    """
    Fits a TfidfVectorizer on training data.

    Args:
        train_df: Training DataFrame.
        text_col: Name of the column containing text.
        max_features: Max number of features.
        ngram_range: N-gram range (default: unigrams and bigrams).

    Returns:
        Trained TfidfVectorizer object.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(train_df[text_col].fillna(""))
    return vectorizer


def transform_tfidf_features(df: pd.DataFrame, vectorizer: TfidfVectorizer, text_col: str = "text") -> pd.DataFrame:
    """
    Transforms a DataFrame using a fitted TfidfVectorizer, safely ignoring unseen words.

    Args:
        df: Input DataFrame.
        vectorizer: Fitted TfidfVectorizer.
        text_col: Name of text column.

    Returns:
        pd.DataFrame of TF-IDF features.
    """
    tfidf_array = vectorizer.transform(df[text_col].fillna("")).toarray()
    tfidf_features = pd.DataFrame(tfidf_array, columns=[f"tfidf_{f}" for f in vectorizer.get_feature_names_out()])
    tfidf_features.reset_index(drop=True, inplace=True)
    return tfidf_features

def generate_all_text_features(df, text_col, tfidf_vectorizer=None, fit_vectorizer=True, max_tfidf_features=500):
    """
    Extracts a full set of text-based features including:
        - Handcrafted statistical/linguistic features
        - TF-IDF features (trained or reused vectorizer)

    Args:
        df (pd.DataFrame): DataFrame with a `text_col` column.
        text_col (str): Name of the column containing text.
        tfidf_vectorizer (TfidfVectorizer): Optional prefit TF-IDF vectorizer.
        fit_vectorizer (bool): If True, fits TF-IDF on this dataset.
        max_tfidf_features (int): Number of TF-IDF features to extract.

    Returns:
        features_df (pd.DataFrame): All features combined.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    text_stat_features = extract_text_features(df.copy(), text_col=text_col)
    if text_col in text_stat_features.columns:
        text_stat_features = text_stat_features.drop(columns=[text_col])

    if tfidf_vectorizer is None and fit_vectorizer:
        tfidf_vectorizer = fit_tfidf_vectorizer(df, text_col, max_features=max_tfidf_features)

    tfidf_features = transform_tfidf_features(df, tfidf_vectorizer, text_col)
    combined_df = pd.concat([text_stat_features, tfidf_features], axis=1)

    return combined_df, tfidf_vectorizer

def drop_high_corr_numeric_features(df: pd.DataFrame, threshold: float = 0.95, return_dropped: bool = False):
    """
    Drop numeric columns that are highly correlated (above threshold), keeping one from each correlated pair.

    Args:
        df (pd.DataFrame): DataFrame with numeric and categorical features.
        threshold (float): Correlation threshold for dropping.
        return_dropped (bool): If True, also return list of dropped columns.

    Returns:
        pd.DataFrame: Reduced DataFrame.
        list (optional): List of dropped feature names.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df_reduced = df.drop(columns=to_drop)

    print(f"Dropped {len(to_drop)} numeric features due to correlation > {threshold}")
    if return_dropped:
        return df_reduced, to_drop
    return df_reduced

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def prepare_categorical_features(X: pd.DataFrame, model_name: str):
    """
    Detect categorical columns, cast to category dtype, and return the correct format
    (column names or indices) for different models.
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Cast to 'category' dtype
    for col in cat_cols:
        X[col] = X[col].astype("category")
    else:
        return X


trainer.load_best_model()
model.eval()

def create_pairwise_classifier_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a DataFrame containing columns: ['id', 'file_1', 'file_2', 'real_text_id']
    into one row per file with a binary 'label' column:
        - label = 1 if the file is the real one
        - label = 0 if the file is fake

    Returns a new DataFrame with columns: ['id', 'text', 'label']
    """
    df_renamed = df.rename(columns={'file_1': 'text_1', 'file_2': 'text_2'})

    df_melted = df_renamed.melt(
        id_vars=['id', 'real_text_id'],
        value_vars=['text_1', 'text_2'],
        var_name='file_source',
        value_name='text'
    )

    df_melted['file_id'] = df_melted['file_source'].str.extract(r'_(\d)').astype(int)
    df_melted['label'] = (df_melted['file_id'] == df_melted['real_text_id']).astype(int)

    return df_melted[['id', 'text', 'label']]

flat_df = create_pairwise_classifier_dataset(train)
texts = flat_df["text"].tolist()
labels = flat_df["label"].tolist()

texts_train, texts_val, y_train, y_val = train_test_split(texts, labels, stratify=labels, test_size=0.1, random_state=42)
y_train = np.array(y_train)
y_val = np.array(y_val)

def clean_text(text):
    """
    Removes control characters and excessive Unicode noise that may break embedding models.
    """
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text
    

def get_projection_embeddings(text_list, prefix: str = "proj_emb_"):
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            # text = clean_text(text)
            emb = model.encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            proj = model.projection_head(emb)
            embeddings.append(proj.cpu().numpy().squeeze())

    embeddings_array = np.array(embeddings)
    feature_names = [f"{prefix}{i}" for i in range(embeddings_array.shape[1])]
    return pd.DataFrame(embeddings_array, columns=feature_names)

def get_projection_embeddings_(text_list, prefix: str = "proj_emb_"):
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            emb = model.encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            proj = model.projection_head(emb)
            embeddings.append(proj.cpu().numpy().squeeze())

    embeddings_array = np.array(embeddings)
    feature_names = [f"{prefix}{i}" for i in range(embeddings_array.shape[1])]
    return pd.DataFrame(embeddings_array, columns=feature_names)

def get_projection_embeddings(text_list, model, prefix: str = "proj_emb_"):
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            emb = model.encoder.encode([text], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            proj = model.projection_head(emb)
            embeddings.append(proj.cpu().numpy().squeeze())

    embeddings_array = np.array(embeddings)
    feature_names = [f"{prefix}{i}" for i in range(embeddings_array.shape[1])]
    return pd.DataFrame(embeddings_array, columns=feature_names)


X_train_embed = get_projection_embeddings(texts_train, model)
X_val_embed = get_projection_embeddings(texts_val, model)

df_train_text = pd.DataFrame({'text': texts_train})
df_val_text = pd.DataFrame({'text': texts_val})

train_features, tfidf_vec = generate_all_text_features(df_train_text, text_col="text")
val_features, _ = generate_all_text_features(df_val_text, text_col="text", tfidf_vectorizer=tfidf_vec, fit_vectorizer=False)

train_features, dropped_cols = drop_high_corr_numeric_features(train_features, threshold=0.85, return_dropped=True)
val_features.drop(columns=dropped_cols, inplace=True, errors="ignore")

print(f"X_train shape (embeddings): {X_train_embed.shape}")
print(f"Handcrafted train features shape: {train_features.shape}")
print(f"X_val shape (embeddings): {X_val_embed.shape}")
print(f"Handcrafted val features shape: {val_features.shape}")

X_train = pd.concat([X_train_embed.reset_index(drop=True), train_features.reset_index(drop=True)], axis=1)
X_val = pd.concat([X_val_embed.reset_index(drop=True), val_features.reset_index(drop=True)], axis=1)

print("\nAfter Concatenation:")
print(f"X_train_final shape: {X_train.shape}")
print(f"X_val_final shape: {X_val.shape}")


n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

best_f1 = -1
best_model = None

oof_preds = []
oof_true = []

y_train = np.array(y_train)
y_val = np.array(y_val)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n========== Fold {fold+1}/{n_splits} ==========")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    clf = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.01,
        loss_function="Logloss",
        eval_metric="F1",
        task_type='GPU',
        verbose=100,
        random_seed=42
    )

    clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)

    y_pred = clf.predict(X_val)
    oof_preds.extend(y_pred)
    oof_true.extend(y_val)

    f1 = f1_score(y_val, y_pred)
    print(f"[Fold {fold+1}] F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = clf
        clf.save_model("best_catboost_model.cbm")
        print(f"[Fold {fold+1}] Best model saved with F1: {f1:.4f}")

print("\n Overall CV Results ")
print(classification_report(oof_true, oof_preds, target_names=["Fake", "Real"]))

best_model = CatBoostClassifier()
best_model.load_model("best_catboost_model.cbm")

y_pred = best_model.predict(X_val)
print("\n Best Model Evaluation ")
print(classification_report(y_val, y_pred, target_names=["Fake", "Real"]))


def shap_summary_for_catboost(model, train_data):
    """
    Generate SHAP summary plot for CatBoost model.
    Uses TreeExplainer for compatibility without GPU dependencies.

    Args:
        model: Trained CatBoost model
        train_data: Training features DataFrame (used during model training)
    """
    plt.figure(figsize=(12, 8))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_data)
    shap.summary_plot(shap_values, train_data, show=False)
    plt.tight_layout()

shap_summary_for_catboost(best_model, X_train)


def flattened_data(df):
    """
    Flatten a DataFrame containing columns: ['id', 'file_1', 'file_2']
    into one row per file with an added 'file_id' column (1 or 2).

    Useful for prediction where real/fake is unknown.

    Returns:
        pd.DataFrame with columns: ['id', 'file_id', 'text']
    """
    df_renamed = df.rename(columns={'file_1': 'text_1', 'file_2': 'text_2'})

    df_melted = df_renamed.melt(
        id_vars=['id'],
        value_vars=['text_1', 'text_2'],
        var_name='file_source',
        value_name='text'
    )

    df_melted['file_id'] = df_melted['file_source'].str.extract(r'_(\d)').astype(int)

    return df_melted[['id', 'file_id', 'text']]

def make_submission(
    df: pd.DataFrame,
    model: nn.Module,
    projection_head: nn.Module,
    classifier,
    handcrafted_feature_fn,
    tfidf_vectorizer=None,
    dropped_cols: List[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 4
) -> pd.DataFrame:
    """
    Generate predictions for either train or test DataFrame.

    Args:
        df: Must contain ['id', 'file_1', 'file_2'].
            For train, also include ['real_text_id'] (ground truth).
    Returns:
        DataFrame with ['id', 'real_text_id', 'confidence'].
        Includes 'label' if ground truth available.
    """
    flat_df = flattened_data(df)
    texts = flat_df["text"].tolist()

    model.eval()
    projection_head.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            emb = model.encode(batch_texts, convert_to_tensor=True,
                               normalize_embeddings=True, show_progress_bar=False)
            proj = projection_head(emb)
            embeddings.extend(proj.cpu().numpy())

    X_embed = np.array(embeddings)
    embedding_dim = X_embed.shape[1]
    embedding_cols = [f"proj_emb_{i}" for i in range(embedding_dim)]

    feats = handcrafted_feature_fn(flat_df.copy(), text_col="text")
    if tfidf_vectorizer:
        tfidf_feats = transform_tfidf_features(flat_df, tfidf_vectorizer, text_col="text")
        feats = pd.concat([feats, tfidf_feats], axis=1)
    if dropped_cols:
        feats.drop(columns=dropped_cols, inplace=True, errors="ignore")

    X_final = np.hstack([X_embed, feats.values])
    X_final = pd.DataFrame(X_final, columns=embedding_cols + list(feats.columns))

    X_final = X_final[classifier.feature_names_].copy()

    probs = classifier.predict_proba(X_final)[:, 1]
    flat_df["real_score"] = probs

    winners = (
    flat_df.loc[flat_df.groupby("id")["real_score"].idxmax(),
                ["id", "file_id", "real_score"]]
    .rename(columns={"real_score": "confidence"})
    .reset_index(drop=True)
    )

    winners["real_text_id"] = winners["file_id"].map({1: 1, 0: 2})
    winners = winners.drop(columns=["file_id"])

    if "real_text_id" in df.columns:
        winners = winners.merge(
            df[["id", "real_text_id"]].rename(columns={"real_text_id": "label"}),
            on="id", how="left"
        )

    return winners

train_prediction_siamese = make_submission(
    df=train,
    model=model.encoder,
    projection_head=model.projection_head,
    classifier=best_model,
    handcrafted_feature_fn=extract_text_features,
    tfidf_vectorizer=tfidf_vec,
    dropped_cols=dropped_cols
)

print(train_prediction_siamese.head(7))


submission_df_siamese = make_submission(
    df=test,
    model=model.encoder,
    projection_head=model.projection_head,
    classifier=best_model,
    handcrafted_feature_fn=extract_text_features,
    tfidf_vectorizer=tfidf_vec,
    dropped_cols=dropped_cols
)
print(submission_df_siamese.head(7))


try:
    del contrastive_df, train_dataset, val_dataset, train_loader, val_loader, X_train_embed, X_val_embed, df_train_text, df_val_text
    del train_features, tfidf_vec, val_features, X_train, X_val, best_model, train, test, trainer
except:
    pass


def train_meta_classifier_and_submit(
    train_predictions_list: List[pd.DataFrame],
    test_predictions_list: List[pd.DataFrame],
    model_names: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train a logistic regression meta-classifier using multiple model outputs (real_text_id only)
    and predict the final real_text_id for test data.

    Parameters:
        train_predictions_list: List of train DataFrames with ['id', 'real_text_id', 'label']
        test_predictions_list:  List of test DataFrames with ['id', 'real_text_id']
        model_names: Optional list of names for each model, e.g. ["Margin", "Siamese", "Attention"]

    Returns:
        final_submission: pd.DataFrame with ['id', 'real_text_id']
        model_importance: pd.DataFrame with importance scores per model
    """

    meta_train = train_predictions_list[0][['id', 'label']].copy()
    for i, df in enumerate(train_predictions_list):
        df_renamed = df[['id', 'real_text_id']].copy()
        df_renamed = df_renamed.rename(columns={'real_text_id': f'pred_{i+1}'})

        df_renamed[f'pred_{i+1}'] = (df_renamed[f'pred_{i+1}'] == 1).astype(int)
        meta_train = meta_train.merge(df_renamed, on='id', how='left')

    X_train = meta_train.drop(columns=['id', 'label'])
    y_train = meta_train['label']

    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(X_train, y_train)

    meta_test = test_predictions_list[0][['id']].copy()
    for i, df in enumerate(test_predictions_list):
        df_renamed = df[['id', 'real_text_id']].copy()
        df_renamed = df_renamed.rename(columns={'real_text_id': f'pred_{i+1}'})
        df_renamed[f'pred_{i+1}'] = (df_renamed[f'pred_{i+1}'] == 1).astype(int)
        meta_test = meta_test.merge(df_renamed, on='id', how='left')

    X_test = meta_test.drop(columns=['id'])
    final_preds = meta_clf.predict(X_test)

    final_submission = pd.DataFrame({
        'id': meta_test['id'],
        'real_text_id': [1 if p == 1 else 2 for p in final_preds]
    })

    coef = meta_clf.coef_[0]
    feat_names = X_train.columns.tolist()
    coef_map = dict(zip(feat_names, coef))

    if model_names is None:
        model_names = [f"model_{i+1}" for i in range(len(train_predictions_list))]

    rows = []
    for i, model_name in enumerate(model_names, start=1):
        p_key = f'pred_{i}'
        p_val = abs(coef_map.get(p_key, 0.0))
        rows.append({
            "model": model_name,
            "importance_pct": p_val
        })

    total = sum(r["importance_pct"] for r in rows) or 1.0
    for r in rows:
        r["importance_pct"] = r["importance_pct"] / total * 100.0

    model_importance = pd.DataFrame(rows).sort_values("importance_pct", ascending=False).reset_index(drop=True)

    return final_submission, model_importance


train_prediction_list = [
    train_prediction_margin,
    train_prediction_siamese,
    train_prediction_attention
]

test_prediction_list = [
    submission_df_margin,
    submission_df_siamese,
    submission_df_attention
]

model_names = ["Margin Loss", "Siamese+CatBoost", "Attention"]

meta_submission, model_importance = train_meta_classifier_and_submit(
    train_prediction_list,
    test_prediction_list,
    model_names=model_names
)

print(meta_submission.head(7))


print(model_importance)


meta_submission.to_csv("submission.csv", index=False)

