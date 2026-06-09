# ====================================================
# Imports & Setup
# ====================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util  # retrieval reranking

# ====================================================
# Config (Offline Paths for Kaggle)
# ====================================================
class CFG:
    roberta_path = "/kaggle/input/roberta-base"  # Local RoBERTa
    sbert_path = "/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2"  # Local SentenceTransformer
    max_len = 256
    batch_size = 32  # Fits well for Kaggle GPU (T4/P100)
    epochs = 3
    lr = 2e-5
    n_splits = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", CFG.device)
if CFG.device == "cuda":
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print("✅ Multi-GPU detected! Using DataParallel.")

# ====================================================
# Load Data
# ====================================================
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample_sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

# Handle missing values
for df in [train, test]:
    df[['QuestionText', 'MC_Answer', 'StudentExplanation']] = df[
        ['QuestionText', 'MC_Answer', 'StudentExplanation']].fillna("")

# Combine labels
train['label'] = train['Category'] + ":" + train['Misconception']
train['label'] = train['label'].str.replace(":NA", ":NA")

# Encode labels
le = LabelEncoder()
train['label_id'] = le.fit_transform(train['label'])
num_classes = len(le.classes_)
print(f"✅ Number of unique labels: {num_classes}")

# ====================================================
# Tokenizer & Preprocessing
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.roberta_path)

def preprocess_text(q, ans, expl):
    q = "" if pd.isna(q) else str(q)
    ans = "" if pd.isna(ans) else str(ans)
    expl = "No explanation provided" if pd.isna(expl) or str(expl).strip() == "" else str(expl)
    return f"Question: {q} Answer: {ans} Explanation: {expl}"

# ====================================================
# Dataset Class
# ====================================================
class MisconceptionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, train=True):
        self.texts = [preprocess_text(q, a, e) for q, a, e in 
                      zip(df['QuestionText'], df['MC_Answer'], df['StudentExplanation'])]
        self.labels = df['label_id'].values if train else None
        self.train = train
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in enc.items()}
        if self.train:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ====================================================
# Model Definition
# ====================================================
class MisconceptionModel(nn.Module):
    def __init__(self, model_path, num_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, 0]  # CLS token
        out = self.dropout(last_hidden)
        return self.fc(out)

# ====================================================
# Training Functions
# ====================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(CFG.device)
        attention_mask = batch['attention_mask'].to(CFG.device)
        labels = batch['labels'].to(CFG.device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_one_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    preds, truths = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            labels = batch['labels'].to(CFG.device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds.append(outputs.softmax(1).cpu().numpy())
            truths.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return total_loss / len(dataloader), preds, truths

def mapk(true, pred, k=3):
    score = 0.0
    for t, p in zip(true, pred):
        if t in p[:k]:
            score += 1.0 / (p[:k].tolist().index(t) + 1)
    return score / len(true)

# ====================================================
# Cross-validation Training
# ====================================================
kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train), num_classes))

for fold, (tr_idx, val_idx) in enumerate(kf.split(train, train['label_id'])):
    print(f"\n========== Fold {fold+1} ==========")
    tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]

    tr_ds = MisconceptionDataset(tr_df, tokenizer, CFG.max_len, train=True)
    val_ds = MisconceptionDataset(val_df, tokenizer, CFG.max_len, train=True)

    tr_dl = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    model = MisconceptionModel(CFG.roberta_path, num_classes).to(CFG.device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, CFG.epochs*len(tr_dl))
    criterion = nn.CrossEntropyLoss()

    best_map = 0
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch+1}")
        train_loss = train_one_epoch(model, tr_dl, optimizer, scheduler, criterion)
        val_loss, val_preds, val_truth = eval_one_epoch(model, val_dl, criterion)
        val_pred_classes = np.argsort(-val_preds, axis=1)
        map3 = mapk(val_truth, val_pred_classes, k=3)
        print(f"Val Loss: {val_loss:.4f} | MAP@3: {map3:.4f}")

        if map3 > best_map:
            best_map = map3
            torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                       f"best_fold_{fold}.pt")
        oof_preds[val_idx] = val_preds

print("\nOOF MAP@3:", mapk(train['label_id'].values, np.argsort(-oof_preds, axis=1)))

# ====================================================
# Retrieval Reranking (Offline SBERT)
# ====================================================
embedder = SentenceTransformer(CFG.sbert_path)
misconception_embeddings = embedder.encode(le.classes_, convert_to_tensor=True, device=CFG.device)

# ====================================================
# Inference + Submission
# ====================================================
test_ds = MisconceptionDataset(test, tokenizer, CFG.max_len, train=False)
test_dl = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)

final_preds = np.zeros((len(test), num_classes))
for fold in range(CFG.n_splits):
    print(f"Loading fold {fold} for inference...")
    model = MisconceptionModel(CFG.roberta_path, num_classes).to(CFG.device)
    model.load_state_dict(torch.load(f"best_fold_{fold}.pt"))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc=f"Inference Fold {fold}"):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            outputs = model(input_ids, attention_mask)
            preds.append(outputs.softmax(1).cpu().numpy())
    final_preds += np.concatenate(preds) / CFG.n_splits

# Retrieval reranking
test_explanations = test['StudentExplanation'].fillna("No explanation").tolist()
explanation_embeddings = embedder.encode(test_explanations, convert_to_tensor=True, device=CFG.device)
cosine_scores = util.cos_sim(explanation_embeddings, misconception_embeddings).cpu().numpy()
combined_scores = final_preds + (0.2 * cosine_scores)
top3 = np.argsort(-combined_scores, axis=1)[:, :3]

# ====================================================
# Submission (Safe Decoding)
# ====================================================
sub = pd.DataFrame()
sub['row_id'] = test['row_id']
predictions = []
for row in top3:
    labels = []
    for idx in row.astype(int):
        if 0 <= idx < len(le.classes_):
            label = le.inverse_transform([idx])[0]
            labels.append(label if pd.notna(label) else "True_Correct:NA")
        else:
            labels.append("True_Correct:NA")
    predictions.append(" ".join(labels))

sub['Category:Misconception'] = predictions

# Match sample submission format
assert list(sub.columns) == list(sample_sub.columns)
assert len(sub) == len(sample_sub)

sub.to_csv("submission.csv", index=False)
print("✅ Submission saved as submission.csv")
print(sub.head())


