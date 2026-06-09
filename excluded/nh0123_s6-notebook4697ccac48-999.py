# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import json
import time
import re
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Config (Kaggle paths + hyperparams)
# ============================================================

DATA_DIR = "/kaggle/input/tensorflow2-question-answering"
TRAIN_PATH = os.path.join(DATA_DIR, "simplified-nq-train.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "simplified-nq-test.jsonl")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

OUTPUT_PATH = "/kaggle/working/submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tune these if you want – they’re chosen to stay under time limits.
if device.type == "cuda":
    MAX_TRAIN_SAMPLES = 60000   # jsonl lines to read
    EPOCHS = 4
    MAX_NEG_PER_Q = 16          # max negatives per question
else:
    MAX_TRAIN_SAMPLES = 15000   # smaller for CPU
    EPOCHS = 3
    MAX_NEG_PER_Q = 10

MAX_Q_LEN = 32
MAX_P_LEN = 128
EMBED_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 64
LR = 1e-3
VAL_FRACTION = 0.2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Utils
# ============================================================

def read_jsonl(path, max_lines=None):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if max_lines is not None and count >= max_lines:
                break


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def tokens_to_ids(tokens, vocab, max_len):
    unk = vocab["<UNK>"]
    pad = vocab["<PAD>"]
    ids = [vocab.get(t, unk) for t in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids += [pad] * (max_len - len(ids))
    return ids


# ============================================================
# Load & split train samples
# ============================================================

start_time = time.time()

print(f"Loading up to {MAX_TRAIN_SAMPLES} training samples from {TRAIN_PATH}")
all_samples = list(read_jsonl(TRAIN_PATH, max_lines=MAX_TRAIN_SAMPLES))
print(f"Loaded {len(all_samples)} training samples from {TRAIN_PATH}")

random.shuffle(all_samples)

num_all = len(all_samples)
num_val = int(num_all * VAL_FRACTION)
num_train = num_all - num_val

train_samples = all_samples[:num_train]
val_samples = all_samples[num_train:]

print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")


# ============================================================
# Build vocabulary
# ============================================================

vocab = {"<PAD>": 0, "<UNK>": 1}
vocab_next_id = 2


def add_tokens_to_vocab(tokens):
    global vocab_next_id
    for t in tokens:
        if t not in vocab:
            vocab[t] = vocab_next_id
            vocab_next_id += 1


print("Building vocabulary...")
for sample in all_samples:
    q_tokens = tokenize(sample["question_text"])
    add_tokens_to_vocab(q_tokens)

    doc_tokens = sample["document_text"].split()
    for cand in sample["long_answer_candidates"]:
        st, end = cand["start_token"], cand["end_token"]
        cand_tokens = doc_tokens[st:end]
        add_tokens_to_vocab([t.lower() for t in cand_tokens])

vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


# ============================================================
# Build training pairs (question–candidate)
# ============================================================

def build_training_pairs(samples, vocab, max_neg_per_q):
    q_ids_list, p_ids_list, labels_list = [], [], []

    for sample in samples:
        doc_tokens = sample["document_text"].split()
        q_tokens = tokenize(sample["question_text"])
        q_ids_single = tokens_to_ids(q_tokens, vocab, MAX_Q_LEN)

        annotations = sample.get("annotations", [])
        gold_indices = set()
        if annotations:
            ann = annotations[0]
            long_answer = ann.get("long_answer", {})
            cand_idx = long_answer.get("candidate_index", -1)
            if cand_idx is not None and cand_idx >= 0:
                gold_indices.add(int(cand_idx))

        cands = sample["long_answer_candidates"]
        if not cands:
            continue

        pos_indices, neg_indices = [], []
        for i, cand in enumerate(cands):
            if i in gold_indices:
                pos_indices.append(i)
            else:
                neg_indices.append(i)

        kept_indices = list(pos_indices)
        if neg_indices:
            random.shuffle(neg_indices)
            kept_indices.extend(neg_indices[:max_neg_per_q])

        if not kept_indices:
            continue

        for i in kept_indices:
            cand = cands[i]
            st, end = cand["start_token"], cand["end_token"]
            cand_tokens = [t.lower() for t in doc_tokens[st:end]]
            p_ids = tokens_to_ids(cand_tokens, vocab, MAX_P_LEN)

            label = 1 if i in gold_indices else 0

            q_ids_list.append(q_ids_single)
            p_ids_list.append(p_ids)
            labels_list.append(label)

    q_ids_tensor = torch.tensor(np.array(q_ids_list), dtype=torch.long)
    p_ids_tensor = torch.tensor(np.array(p_ids_list), dtype=torch.long)
    labels_tensor = torch.tensor(np.array(labels_list), dtype=torch.float32).unsqueeze(1)
    return q_ids_tensor, p_ids_tensor, labels_tensor


print("Building training pairs...")
train_q_ids, train_p_ids, train_labels = build_training_pairs(
    train_samples, vocab, MAX_NEG_PER_Q
)
print(
    f"Prepared {len(train_labels)} training examples "
    f"(question-paragraph candidate pairs)"
)


# ============================================================
# Model
# ============================================================

class BiLSTMQA(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.q_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.p_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, q_ids, p_ids):
        q_emb = self.embedding(q_ids)
        p_emb = self.embedding(p_ids)

        q_out, _ = self.q_lstm(q_emb)
        p_out, _ = self.p_lstm(p_emb)

        q_repr, _ = torch.max(q_out, dim=1)
        p_repr, _ = torch.max(p_out, dim=1)

        h = torch.cat([q_repr, p_repr], dim=1)
        logits = self.fc(h)
        return logits


model = BiLSTMQA(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
pos_weight = torch.tensor([3.0], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)


# ============================================================
# Validation helper (NOTE: no model.eval() here)
# ============================================================

def eval_long_answer_accuracy(model, samples, vocab):
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in samples:
            doc_tokens = sample["document_text"].split()
            q_tokens = tokenize(sample["question_text"])
            q_ids_single = tokens_to_ids(q_tokens, vocab, MAX_Q_LEN)

            candidates = sample["long_answer_candidates"]
            if not candidates:
                continue

            cand_spans = []
            cand_p_ids = []

            for cand in candidates:
                st, end = cand["start_token"], cand["end_token"]
                cand_spans.append((st, end))
                cand_tokens = [t.lower() for t in doc_tokens[st:end]]
                cand_p_ids.append(tokens_to_ids(cand_tokens, vocab, MAX_P_LEN))

            q_batch = torch.tensor(
                np.repeat([q_ids_single], len(cand_spans), axis=0),
                dtype=torch.long,
                device=device,
            )
            p_batch = torch.tensor(np.array(cand_p_ids), dtype=torch.long, device=device)

            logits = model(q_batch, p_batch).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            best_idx = int(probs.argmax())
            pred_cand_idx = best_idx

            annotations = sample.get("annotations", [])
            gold_idx = -1
            if annotations:
                ann = annotations[0]
                la = ann.get("long_answer", {})
                gold_idx = la.get("candidate_index", -1)

            if pred_cand_idx == gold_idx:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


# ============================================================
# Train (model stays in train mode; validation uses no_grad only)
# ============================================================

num_train_pairs = train_labels.size(0)
print(f"Number of training pairs: {num_train_pairs}")

best_val_acc = 0.0
best_state_dict = None
patience = 2
no_improve_epochs = 0

for epoch in range(EPOCHS):
    model.train()

    perm = torch.randperm(num_train_pairs)
    epoch_loss = 0.0

    for start_idx in range(0, num_train_pairs, BATCH_SIZE):
        idx = perm[start_idx : start_idx + BATCH_SIZE]
        q_batch = train_q_ids[idx].to(device)
        p_batch = train_p_ids[idx].to(device)
        y_batch = train_labels[idx].to(device)

        optimizer.zero_grad()
        logits = model(q_batch, p_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_loss += loss.item() * q_batch.size(0)

    epoch_loss /= num_train_pairs
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    val_acc, val_correct, val_total = eval_long_answer_accuracy(model, val_samples, vocab)
    print(
        f"Validation long-answer accuracy (no threshold): {val_acc:.4f} "
        f"({val_correct}/{val_total})"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping: no improvement on validation.")
            break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    model.to(device)
    print(f"Restored best model with val acc = {best_val_acc:.4f}")


# ============================================================
# Inference on test set (now we can safely use eval())
# ============================================================

model.eval()

long_preds = {}
short_preds = {}

print(f"Running inference on test set: {TEST_PATH}")
test_count = 0

with torch.no_grad():
    for sample in read_jsonl(TEST_PATH):
        example_id = str(sample["example_id"])
        base_id = example_id

        doc_tokens = sample["document_text"].split()
        q_tokens = tokenize(sample["question_text"])
        q_ids_single = tokens_to_ids(q_tokens, vocab, MAX_Q_LEN)

        cand_spans = []
        cand_p_ids = []

        for cand in sample["long_answer_candidates"]:
            st, end = cand["start_token"], cand["end_token"]
            cand_spans.append((st, end))
            cand_tokens = [t.lower() for t in doc_tokens[st:end]]
            cand_p_ids.append(tokens_to_ids(cand_tokens, vocab, MAX_P_LEN))

        if len(cand_spans) == 0:
            long_preds[base_id] = ""
            short_preds[base_id] = ""
            test_count += 1
            if test_count % 100 == 0:
                print(f"Processed {test_count} test samples...")
            continue

        q_batch = torch.tensor(
            np.repeat([q_ids_single], len(cand_spans), axis=0),
            dtype=torch.long,
            device=device,
        )
        p_batch = torch.tensor(np.array(cand_p_ids), dtype=torch.long, device=device)

        logits = model(q_batch, p_batch).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        best_idx = int(probs.argmax())
        best_start, best_end = cand_spans[best_idx]

        long_pred_str = f"{best_start}:{best_end}"

        long_text = " ".join(doc_tokens[best_start:best_end]).lower()
        if " yes " in (" " + long_text + " "):
            short_pred_str = "YES"
        elif " no " in (" " + long_text + " "):
            short_pred_str = "NO"
        else:
            short_pred_str = ""

        long_preds[base_id] = long_pred_str
        short_preds[base_id] = short_pred_str

        test_count += 1
        if test_count % 100 == 0:
            print(f"Processed {test_count} test samples...")

print(f"Finished inference on {test_count} test samples.")


# ============================================================
# Build submission
# ============================================================

def base_from_row_id(row_id: str) -> str:
    s = str(row_id)
    if "_" in s:
        return s.rsplit("_", 1)[0]
    return s


def build_submission_from_sample(sample_sub_path, long_preds, short_preds):
    sub = pd.read_csv(sample_sub_path)
    if "PredictionString" not in sub.columns or "example_id" not in sub.columns:
        raise ValueError(
            "sample_submission.csv must have columns 'example_id' and 'PredictionString'."
        )

    sub["PredictionString"] = ""

    is_long = sub["example_id"].astype(str).str.endswith("_long")
    is_short = sub["example_id"].astype(str).str.endswith("_short")

    long_ids = sub.loc[is_long, "example_id"].astype(str)
    sub.loc[is_long, "PredictionString"] = [
        long_preds.get(base_from_row_id(eid), "") for eid in long_ids
    ]

    short_ids = sub.loc[is_short, "example_id"].astype(str)
    sub.loc[is_short, "PredictionString"] = [
        short_preds.get(base_from_row_id(eid), "") for eid in short_ids
    ]

    return sub


print(f"Building submission from {SAMPLE_SUB_PATH}")
submission_df = build_submission_from_sample(SAMPLE_SUB_PATH, long_preds, short_preds)
submission_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved submission to {OUTPUT_PATH}")

elapsed = time.time() - start_time
mins = int(elapsed // 60)
secs = int(elapsed % 60)
print(f"Execution time: {mins} minutes {secs} seconds.")





