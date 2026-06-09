import os
import zipfile

zip_files = [
    '/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip', 
    '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip', 
    '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip', 
    '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
]

extract_base_folder = '/kaggle/working/jigsaw-toxic-comment-classification-challenge'

os.makedirs(extract_base_folder, exist_ok=True)

for zip_file in zip_files:
    extract_folder = os.path.join(extract_base_folder, os.path.splitext(os.path.basename(zip_file))[0])
    
    if not os.path.exists(extract_folder):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"Extracted {zip_file} to {extract_folder}")
    else:
        print(f"Skipping {zip_file}, {extract_folder} already exists.")


import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, GRU, Bidirectional, Dense, Dropout,
    SpatialDropout1D, GlobalMaxPooling1D
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import warnings
warnings.filterwarnings("ignore")


ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


train_path = "/kaggle/working/jigsaw-toxic-comment-classification-challenge/train.csv/train.csv"
test_path = "/kaggle/working/jigsaw-toxic-comment-classification-challenge/test.csv/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


train_df.head()


train_df[label_cols].sum()


train_df.shape


train_df.info()


train_df.isnull().sum()


x = train_df.iloc[:, 2:].sum() # Chỉ lấy các cột label
x


rowsums = train_df.iloc[:, 2:].sum(axis=1) # Lấy các cột label và tính tổng theo từng cột
rowsums


no_label_count = 0

for i, count in rowsums.items():
    if count==0:
        no_label_count += 1
        
print('Tổng số lượng comments: ', len(train_df))
print('Số lượng comment chưa được gán nhãn: ', no_label_count)
print('Số lượng label ', x.sum())


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
ax = sns.barplot(x=x.index, y=x.values, alpha=0.8, palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:red', 'tab:grey'])
plt.title('Phân bố Label của Dataset')
plt.ylabel('Count')
plt.xlabel('Label')

plt.show()


plt.figure(figsize=(6, 4))
ax = sns.countplot(x=rowsums.values, alpha=0.8, palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:red', 'tab:grey'])
plt.title('Phân bố Labels cho mỗi Comment')
plt.ylabel('# of Occurences')
plt.xlabel('# of Labels')

plt.show()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


train_df["comment_text"] = train_df["comment_text"].fillna("").apply(clean_text)
test_df["comment_text"]  = test_df["comment_text"].fillna("").apply(clean_text)


X = train_df["comment_text"].values
y = train_df[label_cols].values.astype("float32")


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=(y.sum(axis=1) > 0)
)


print("Train size:", len(X_train))
print("Val size  :", len(X_val))


pos_freq = y_train.mean(axis=0)
print("Positive frequency per label:")
for name, freq in zip(label_cols, pos_freq):
    print(f"{name}: {freq:.5f}")


label_weights = 1.0 / (pos_freq + 1e-6)
label_weights = label_weights / label_weights.max()

print("\nLabel weights (normalized):")
for name, w in zip(label_cols, label_weights):
    print(f"{name}: {w:.3f}")


label_weights_tf = tf.constant(label_weights, dtype=tf.float32)


# Weighted BCE for multi-label (Keras)

def weighted_bce_multi(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * label_weights_tf)


MAX_NUM_WORDS = 100_000
MAX_LEN = 220


tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq   = tokenizer.texts_to_sequences(X_val)
X_test_seq  = tokenizer.texts_to_sequences(test_df["comment_text"].values)

X_train_seq = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_val_seq   = pad_sequences(X_val_seq, maxlen=MAX_LEN)
X_test_seq  = pad_sequences(X_test_seq, maxlen=MAX_LEN)

word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

print("Vocab size:", len(word_index))
print("Using num_words:", num_words)


FASTTEXT_PATH = "/kaggle/input/fast-text-word-embeddings/wiki-news-300d-1M.vec"

EMBEDDING_DIM = 300
embeddings_index = {}

embedding_matrix = np.random.normal(0, 0.01, (num_words, EMBEDDING_DIM)).astype("float32")

bad_lines = 0
found = 0

with open(FASTTEXT_PATH, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
    # Đọc thử dòng đầu xem có phải header "num_words dim" không
    header = f.readline().rstrip().split(" ")
    if len(header) != 2 or not header[0].isdigit():
        # Không phải header chuẩn → quay lại đầu file, xử lý tất cả như data
        f.seek(0)

    for line in f:
        values = line.rstrip().split(" ")
        # cần ít nhất 1 từ + 300 số
        if len(values) < EMBEDDING_DIM + 1:
            bad_lines += 1
            continue

        word = values[0]
        vector_str = values[1:]

        try:
            coefs = np.asarray(vector_str, dtype="float32")
        except ValueError:
            # Dòng này có kí tự binary / không convert được → bỏ
            bad_lines += 1
            continue

        # Chỉ set embedding nếu từ nằm trong vocab tokenizer
        idx = word_index.get(word)
        if idx is not None and idx < num_words:
            embedding_matrix[idx] = coefs
            found += 1

print("FastText loading done.")
print("  Found vectors for", found, "words in vocab.")
print("  Skipped bad / short lines:", bad_lines)
print("Embedding matrix shape:", embedding_matrix.shape)


def build_bigru_fasttext_model():
    model = Sequential()
    model.add(
        Embedding(
            input_dim=num_words,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_LEN,
            trainable=False  
        )
    )
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(label_cols), activation="sigmoid"))

    model.compile(
        loss=weighted_bce_multi,  # uses label_weights_tf
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"]
    )
    return model


bigru_model = build_bigru_fasttext_model()
bigru_model.summary()


EPOCHS = 6
BATCH_SIZE = 256

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
    )
]

history_bigru = bigru_model.fit(
    X_train_seq,
    y_train,
    validation_data=(X_val_seq, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


y_val_pred_proba = bigru_model.predict(X_val_seq, batch_size=256, verbose=1)
y_val_pred = (y_val_pred_proba > 0.5).astype("int32")

print(classification_report(y_val, y_val_pred, target_names=label_cols, digits=4))
print("Micro F1:", f1_score(y_val, y_val_pred, average="micro"))
print("Macro F1:", f1_score(y_val, y_val_pred, average="macro"))


test_pred_proba_bigru = bigru_model.predict(X_test_seq, batch_size=256, verbose=1)

submission_bigru = pd.DataFrame(test_pred_proba_bigru, columns=label_cols)
submission_bigru.insert(0, "id", test_df["id"].values)
submission_bigru.to_csv("submission_bigru_fasttext.csv", index=False)


# Save BiGRU model 
bigru_model_path = os.path.join(ARTIFACT_DIR, "bigru_fasttext.h5")
bigru_model.save(bigru_model_path)
print("Saved BiGRU model to:", bigru_model_path)

# Save BiGRU tokenizer 
tokenizer_path = os.path.join(ARTIFACT_DIR, "tokenizer_bigru.pkl")
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)
print("Saved BiGRU tokenizer to:", tokenizer_path)


%pip install -q "transformers==4.46.0" "protobuf==3.20.3"


import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = "roberta-base"
tokenizer_roberta = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN_ROBERTA = 220


class JigsawDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=220):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


BATCH_SIZE_ROBERTA = 16

train_dataset_roberta = JigsawDataset(
    X_train, y_train, tokenizer=tokenizer_roberta, max_len=MAX_LEN_ROBERTA
)
val_dataset_roberta = JigsawDataset(
    X_val, y_val, tokenizer=tokenizer_roberta, max_len=MAX_LEN_ROBERTA
)

train_loader = DataLoader(train_dataset_roberta, batch_size=BATCH_SIZE_ROBERTA, shuffle=True)
val_loader   = DataLoader(val_dataset_roberta, batch_size=BATCH_SIZE_ROBERTA, shuffle=False)


model_roberta = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)
model_roberta.to(device)


pos_freq_t = torch.tensor(pos_freq, dtype=torch.float, device=device)
label_weights_t = 1.0 / (pos_freq_t + 1e-6)
label_weights_t = label_weights_t / label_weights_t.max()

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=label_weights_t)

EPOCHS_ROBERTA = 3
LR = 2e-5
optimizer = torch.optim.AdamW(model_roberta.parameters(), lr=LR)

num_training_steps = EPOCHS_ROBERTA * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)


def train_one_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc="Train", leave=False)

    for step, batch in enumerate(progress_bar, start=1):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / step

        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / len(data_loader)



def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(data_loader, desc="Val", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    preds_bin = (all_preds > 0.5).astype("int32")

    micro_f1 = f1_score(all_labels, preds_bin, average="micro")
    macro_f1 = f1_score(all_labels, preds_bin, average="macro")

    return total_loss / len(data_loader), micro_f1, macro_f1


ROBERTA_SAVE_DIR = os.path.join(ARTIFACT_DIR, "roberta_multi")
os.makedirs(ROBERTA_SAVE_DIR, exist_ok=True)

ROBERTA_STATE_PATH = os.path.join(ARTIFACT_DIR, "best_roberta_multi_state_dict.pth")


best_macro_f1 = 0.0

for epoch in range(1, EPOCHS_ROBERTA + 1):
    print(f"\nEpoch {epoch}/{EPOCHS_ROBERTA}")

    train_loss = train_one_epoch(model_roberta, train_loader, optimizer, scheduler, criterion, device)
    val_loss, micro_f1, macro_f1 = eval_model(model_roberta, val_loader, criterion, device)

    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val loss  : {val_loss:.4f}")
    print(f"  Micro F1  : {micro_f1:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1

        # 1) Save full HF model + tokenizer
        model_roberta.save_pretrained(ROBERTA_SAVE_DIR)
        tokenizer_roberta.save_pretrained(ROBERTA_SAVE_DIR)

        # 2) also save state_dict backup
        torch.save(model_roberta.state_dict(), ROBERTA_STATE_PATH)

        print(f"  --> New best model saved (macro F1 = {macro_f1:.4f})")
        print(f"      HF folder : {ROBERTA_SAVE_DIR}")
        print(f"      state_dict: {ROBERTA_STATE_PATH}")


# Load best weights
best_model = AutoModelForSequenceClassification.from_pretrained(
    ROBERTA_SAVE_DIR,
    local_files_only=True
)
best_model.to(device)
best_model.eval()
best_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_SAVE_DIR, local_files_only=True)


test_dataset_roberta = JigsawDataset(
    test_df["comment_text"].values,
    labels=None,
    tokenizer=best_tokenizer,     # use best_tokenizer
    max_len=MAX_LEN_ROBERTA
)
test_loader_roberta = DataLoader(test_dataset_roberta, batch_size=BATCH_SIZE_ROBERTA, shuffle=False)

all_test_preds = []
with torch.no_grad():
    for batch in test_loader_roberta:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_test_preds.append(probs)

test_pred_proba_roberta = np.vstack(all_test_preds)


submission_roberta = pd.DataFrame(test_pred_proba_roberta, columns=label_cols)
submission_roberta.insert(0, "id", test_df["id"].values)
submission_roberta.to_csv("submission_roberta.csv", index=False)

