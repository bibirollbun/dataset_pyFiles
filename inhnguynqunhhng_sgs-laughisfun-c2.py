# import pandas as pd
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split # (Optional, để kiểm thử)

# # --- 1. Tải Dữ liệu ---
# # Đảm bảo bạn thay thế bằng tên thư mục đúng
# data_path = '/kaggle/input/rmit-hackathon-2025/' # <-- THAY THẾ BẰNG TÊN ĐÚNG

# try:
#     train_df = pd.read_csv(data_path + 'train.csv')
#     test_df = pd.read_csv(data_path + 'test.csv')
#     sample_sub = pd.read_csv(data_path + 'sample_submission.csv')

#     print("--- Dữ liệu training (train.csv) ---")
#     print(train_df.head())
#     print(f"\nCác nhãn (label) trong training: {train_df['label'].unique()}")

#     print("\n--- Dữ liệu testing (test.csv) ---")
#     print(test_df.head())

#     print("\n--- Định dạng file nộp bài (sample_submission.csv) ---")
#     print(sample_sub.head())

# except FileNotFoundError:
#     print("LỖI: Không tìm thấy file. Hãy kiểm tra lại đường dẫn 'data_path' và tên file.")


# # --- 2. Chuẩn bị dữ liệu ---
# # X là cột 'text'
# X_train = train_df['text']
# # y là cột 'label' (theo mô tả)
# y_train = train_df['label'] 

# # Dữ liệu text từ file test
# X_test = test_df['text']

# print("Bắt đầu tạo pipeline...")
# # Tạo pipeline như cũ
# model_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=10000)), # Giới hạn số features để chạy nhanh hơn
#     ('log_reg', LogisticRegression(max_iter=1000, random_state=42)) 
# ])

# print("Bắt đầu huấn luyện mô hình...")
# # Huấn luyện mô hình
# model_pipeline.fit(X_train, y_train)

# print("Mô hình huấn luyện xong!")


# # --- 3. Kiểm tra các lớp mà mô hình đã học ---
# # Thư viện scikit-learn thường sắp xếp các lớp theo thứ tự chữ cái
# # Ví dụ: ['benign', 'jailbreak']
# print(f"Thứ tự các lớp (classes) mà mô hình đã học: {model_pipeline.classes_}")

# # --- 4. Dự đoán Xác Suất ---
# # Chúng ta dùng .predict_proba() thay vì .predict()
# # Nó sẽ trả về 2 cột xác suất, tương ứng với 2 lớp ở trên
# # Ví dụ: [ [P(benign), P(jailbreak)], [P(benign), P(jailbreak)], ... ]
# predictions_proba = model_pipeline.predict_proba(X_test)

# print("\nVí dụ về xác suất dự đoán (cho 5 mẫu đầu):")
# print(predictions_proba[:5])

# # Chúng ta cần cột "jailbreak". 
# # Nếu model.classes_ là ['benign', 'jailbreak'], thì 'jailbreak' là cột thứ 2 (index 1)
# # Nếu lỡ nó là ['jailbreak', 'benign'], thì 'jailbreak' là cột thứ 1 (index 0)
# # Dựa theo thứ tự ['benign', 'jailbreak'], chúng ta lấy cột index 1
# try:
#     jailbreak_proba = predictions_proba[:, 1]
#     print("\nĐã trích xuất xác suất 'jailbreak' (cột index 1).")
# except IndexError:
#     print("LỖI: Không thể lấy cột index 1. Hãy kiểm tra lại model.classes_")

# print("Ví dụ 5 xác suất 'jailbreak' đầu tiên:")
# print(jailbreak_proba[:5])


# # --- 5. Tạo file Submission ---

# # QUAN TRỌNG: Tên cột là 'Id' (viết hoa chữ I) theo mô tả
# submission_df = pd.DataFrame({
#     'Id': test_df['Id'],           # Lấy cột 'id' (viết thường) từ test_df
#     'target': jailbreak_proba  # Dùng xác suất 'jailbreak' chúng ta vừa lấy
# })

# # 6. Kiểm tra file
# print("\n--- File nộp bài của tôi (submission.csv) ---")
# print(submission_df.head())

# # 7. Lưu file
# # index=False là bắt buộc
# submission_df.to_csv('submission.csv', index=False)

# print("\nĐã tạo file 'submission.csv'")


# !pip install datasets -q
# !pip install --upgrade transformers huggingface_hub tokenizers -q


# import pandas as pd
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from scipy.special import softmax # Để chuyển logits thành xác suất

# # Tắt cảnh báo của huggingface (cho sạch notebook)
# import warnings
# warnings.filterwarnings("ignore")
# os.environ['WANDB_DISABLED'] = 'true' # Tắt wandb logging

# # --- 1. Tải Dữ liệu ---
# data_path = '/kaggle/input/rmit-hackathon-2025/' # <-- KIỂM TRA LẠI TÊN NÀY
# train_df = pd.read_csv(data_path + 'train.csv')
# test_df = pd.read_csv(data_path + 'test.csv')
# sample_sub = pd.read_csv(data_path + 'sample_submission.csv')

# # --- 2. Chuyển nhãn (label) thành số ---
# # Mô hình cần số (0 và 1), không phải chữ ('benign' và 'jailbreak')
# train_df['label_int'] = train_df['label'].map({'benign': 0, 'jailbreak': 1})

# # --- 3. Chia tập Validation (QUAN TRỌNG) ---
# # Chúng ta chia 4000 mẫu thành 3600 train, 400 validation
# # stratify=train_df['label_int'] đảm bảo tỷ lệ jailbreak/benign ở cả 2 tập là như nhau
# train_data, val_data = train_test_split(
#     train_df,
#     test_size=0.1,  # 10% của 4000 là 400 mẫu
#     random_state=42, # Để kết quả có thể tái lập
#     stratify=train_df['label_int']
# )

# print(f"Số mẫu training: {len(train_data)}")
# print(f"Số mẫu validation: {len(val_data)}")

# # --- 4. Chuyển Pandas DataFrame thành Dataset của Hugging Face ---
# train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
# val_dataset = Dataset.from_pandas(val_data.reset_index(drop=True))
# test_dataset = Dataset.from_pandas(test_df)

# # --- 1. Chọn Model ---
# # 'distilbert-base-uncased' là lựa chọn tuyệt vời: nhanh, nhẹ, hiệu quả
# model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # --- 2. Tạo hàm Tokenize ---
# def tokenize_function(examples):
#     # padding="max_length" và truncation=True đảm bảo mọi prompt đều có độ dài bằng nhau
#     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# print("Đang tokenize dữ liệu... (việc này có thể mất 1-2 phút)")
# # Áp dụng hàm tokenize cho cả 3 tập
# tokenized_train = train_dataset.map(tokenize_function, batched=True)
# tokenized_val = val_dataset.map(tokenize_function, batched=True)
# tokenized_test = test_dataset.map(tokenize_function, batched=True)

# # Đổi tên cột 'label_int' thành 'labels' để Trainer tự động nhận diện
# tokenized_train = tokenized_train.rename_column("label_int", "labels")
# tokenized_val = tokenized_val.rename_column("label_int", "labels")

# print("Tokenize hoàn tất!")

# # Hàm này sẽ được gọi ở cuối mỗi epoch để tính điểm validation
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
    
#     # Chuyển logits (điểm số thô) thành xác suất cho lớp 1 ('jailbreak')
#     # dùng hàm softmax
#     probs = softmax(logits, axis=1)[:, 1]
    
#     # Tính ROC-AUC
#     auc = roc_auc_score(labels, probs)
#     return {"roc_auc": auc}


# # --- 1. Tải Model ---
# # num_labels=2 vì chúng ta có 2 lớp (benign, jailbreak)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # --- 2. Cài đặt các tham số huấn luyện ---
# training_args = TrainingArguments(
#     output_dir="./distilbert_model",       # Thư mục lưu model
#     evaluation_strategy="epoch",           # Báo cáo điểm validation sau mỗi epoch
#     save_strategy="epoch",                 # Lưu model sau mỗi epoch
#     num_train_epochs=3,                    # Train trong 3 epoch là đủ
#     per_device_train_batch_size=32,        # Tăng lên nếu GPU cho phép (ví dụ: 32, 64)
#     per_device_eval_batch_size=32,
#     learning_rate=2e-5,                    # Tốc độ học
#     weight_decay=0.01,
#     logging_steps=50,                      # Log sau mỗi 50 bước
#     load_best_model_at_end=True,           # TỰ ĐỘNG tải model có điểm tốt nhất
#     metric_for_best_model="roc_auc",       # Dùng ROC-AUC để chọn model tốt nhất
#     report_to="none"                       # Không log lên wandb
# )

# # --- 3. Khởi tạo Trainer ---
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_val,
#     compute_metrics=compute_metrics,       # Dùng hàm tính ROC-AUC của chúng ta
#     tokenizer=tokenizer
# )

# print("Bắt đầu huấn luyện...")
# # --- 4. HUẤN LUYỆN ---
# trainer.train()

# print("Huấn luyện hoàn tất!")

# print("Bắt đầu dự đoán trên tập test...")

# # 1. Lấy dự đoán (logits) từ mô hình đã huấn luyện (model tốt nhất)
# raw_predictions = trainer.predict(tokenized_test)
# test_logits = raw_predictions.predictions

# # 2. Chuyển logits thành xác suất của 'jailbreak' (lớp 1)
# test_probs = softmax(test_logits, axis=1)[:, 1]

# print("Dự đoán hoàn tất. 5 xác suất đầu tiên:")
# print(test_probs[:5])

# # 3. Tạo file submission
# submission_df = pd.DataFrame({
#     'Id': test_df['id'],  # Dùng 'id' từ test_df
#     'target': test_probs    # Dùng xác suất vừa dự đoán
# })

# # 4. Lưu file
# submission_df.to_csv('submission.csv', index=False)

# print("\nĐã tạo file 'submission.csv' thành công! Sẵn sàng để nộp bài.")
# print(submission_df.head())


# --- RoBERTa with Advanced Augmentation + Early Stopping ---
!pip install -U transformers huggingface_hub -q

import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# Config and seeding
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/kaggle/input/rmit-hackathon-2025")
MAX_LENGTH = 256

# Load data
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

for df in (train_df, test_df):
    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str)

label2id = {"benign": 0, "jailbreak": 1}
id2label = {v: k for k, v in label2id.items()}
train_df["label_int"] = train_df["label"].map(label2id).astype(int)

# Split train/val
train_data, val_data = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label_int"]
)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# --- Data Augmentation (EDA-lite) ---
def random_deletion(words, drop_prob=0.1):
    if len(words) <= 1:
        return words
    kept = [w for w in words if random.random() > drop_prob]
    return kept if kept else [random.choice(words)]

def random_swap(words, n_swaps=1):
    if len(words) < 2:
        return words
    words = words.copy()
    for _ in range(n_swaps):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words

def random_insertion(words, n_inserts=1):
    if not words:
        return words
    words = words.copy()
    for _ in range(n_inserts):
        w = random.choice(words)
        pos = random.randint(0, len(words))
        words.insert(pos, w)
    return words

def augment_text_once(text):
    words = text.split()
    if not words:
        return text
    op = random.choice(["delete", "swap", "insert"])
    if op == "delete":
        aug = random_deletion(words, drop_prob=0.1)
    elif op == "swap":
        aug = random_swap(words, n_swaps=1)
    else:
        aug = random_insertion(words, n_inserts=1)
    return " ".join(aug)

# Augment training data
aug_texts = [augment_text_once(t) for t in train_data["text"].tolist()]
aug_df = pd.DataFrame({"text": aug_texts, "label_int": train_data["label_int"].values})
train_data = pd.concat([train_data, aug_df], ignore_index=True)
train_data = train_data.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"Augmented train size: {len(train_data)}")

# --- Tokenizer ---
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["text"].tolist()
        self.labels = torch.tensor(df["label_int"].values, dtype=torch.long) if "label_int" in df.columns else None
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

train_dataset = TextDataset(train_data, tokenizer)
val_dataset = TextDataset(val_data, tokenizer)
print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")

# --- Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

if hasattr(model, "gradient_checkpointing_enable"):
    try:
        model.gradient_checkpointing_enable()
    except:
        pass

collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.tensor(logits).softmax(dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:, 1]),
    }

# --- Training Arguments ---
args = TrainingArguments(
    output_dir="./roberta_output",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    report_to="none",
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
)

# --- Train ---
print("Bắt đầu huấn luyện...")
trainer.train()
val_metrics = trainer.evaluate()
print("Val metrics:", val_metrics)

# --- Inference on test ---
model.eval()
test_texts = test_df["text"].tolist()
probs_list = []

with torch.no_grad():
    for i in tqdm(range(0, len(test_texts), 256)):
        batch_texts = test_texts[i:i+256]
        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        outputs = model(**enc)
        p = torch.softmax(outputs.logits, dim=1)[:, 1]
        probs_list.append(p.cpu())

probs = torch.cat(probs_list).numpy()

# --- Submission ---
submission = pd.DataFrame({
    "Id": test_df["Id"].values,
    "TARGET": probs,
})
submission.to_csv("submission.csv", index=False)
print("Đã tạo file submission.csv!")
submission.head()


# # Cài đặt/Update transformers lên bản mới nhất để tránh lỗi tham số
# !pip install --upgrade transformers huggingface_hub -q

# import pandas as pd
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from scipy.special import softmax # Để chuyển logits thành xác suất

# # Tắt cảnh báo của huggingface (cho sạch notebook)
# import warnings
# warnings.filterwarnings("ignore")
# os.environ['WANDB_DISABLED'] = 'true' # Tắt wandb logging

# # --- 1. Tải Dữ liệu ---
# data_path = '/kaggle/input/rmit-hackathon-2025/' # <-- KIỂM TRA LẠI TÊN NÀY
# train_df = pd.read_csv(data_path + 'train.csv')
# test_df = pd.read_csv(data_path + 'test.csv')
# sample_sub = pd.read_csv(data_path + 'sample_submission.csv')

# # --- 2. Chuyển nhãn (label) thành số ---
# train_df['label_int'] = train_df['label'].map({'benign': 0, 'jailbreak': 1})

# # --- 3. Chia tập Validation (QUAN TRỌNG) ---
# train_data, val_data = train_test_split(
#     train_df,
#     test_size=0.1,
#     random_state=42,
#     stratify=train_df['label_int']
# )

# print(f"Số mẫu training: {len(train_data)}")
# print(f"Số mẫu validation: {len(val_data)}")

# train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
# val_dataset = Dataset.from_pandas(val_data.reset_index(drop=True))
# test_dataset = Dataset.from_pandas(test_df)

# model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# print("Đang tokenize dữ liệu... (việc này có thể mất 1-2 phút)")
# tokenized_train = train_dataset.map(tokenize_function, batched=True)
# tokenized_val = val_dataset.map(tokenize_function, batched=True)
# tokenized_test = test_dataset.map(tokenize_function, batched=True)

# # Đổi tên cột 'label_int' thành 'labels' để Trainer tự động nhận diện
# tokenized_train = tokenized_train.rename_column("label_int", "labels")
# tokenized_val = tokenized_val.rename_column("label_int", "labels")

# # Loại bỏ các cột không dùng (đặc biệt là 'label' dạng chuỗi gây lỗi), chỉ giữ input_ids, attention_mask, labels
# def keep_only_model_cols(ds, has_labels):
#     cols = set(ds.column_names)
#     drop = [c for c in ["text", "label", "__index_level_0__"] if c in cols]
#     ds = ds.remove_columns(drop) if drop else ds
#     wanted = ["input_ids", "attention_mask"] + (["labels"] if has_labels else [])
#     ds = ds.with_format(type="torch", columns=[c for c in wanted if c in ds.column_names])
#     return ds

# tokenized_train = keep_only_model_cols(tokenized_train, has_labels=True)
# tokenized_val = keep_only_model_cols(tokenized_val, has_labels=True)
# tokenized_test = keep_only_model_cols(tokenized_test, has_labels=False)

# print("Tokenize hoàn tất!")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     probs = softmax(logits, axis=1)[:, 1]
#     auc = roc_auc_score(labels, probs)
#     return {"roc_auc": auc}

# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # --- Sửa lỗi: dùng eval_strategy thay vì evaluation_strategy, và khớp với save_strategy ---
# training_args = TrainingArguments(
#     output_dir="./distilbert_model",
#     eval_strategy="epoch",                 # ĐÁNH GIÁ mỗi epoch (phiên bản transformers này dùng eval_strategy)
#     save_strategy="epoch",                 # Lưu model sau mỗi epoch
#     num_train_epochs=3,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     logging_steps=50,
#     load_best_model_at_end=True,
#     metric_for_best_model="roc_auc",
#     report_to="none"
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_val,
#     compute_metrics=compute_metrics,
#     tokenizer=tokenizer
# )

# print("Bắt đầu huấn luyện...")
# trainer.train()

# print("Huấn luyện hoàn tất!")

# print("Bắt đầu dự đoán trên tập test...")
# raw_predictions = trainer.predict(tokenized_test)
# test_logits = raw_predictions.predictions
# test_probs = softmax(test_logits, axis=1)[:, 1]

# print("Dự đoán hoàn tất. 5 xác suất đầu tiên:")
# print(test_probs[:5])


# # --- DistilBERT Embedding + XGBoost Pipeline ---
# !pip install -q xgboost transformers
# import torch
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm
# import xgboost as xgb

# # 1. Load DistilBERT (feature extraction only, no fine-tune)
# bert_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(bert_name)
# bert = AutoModel.from_pretrained(bert_name)
# bert.eval()

# # 2. Hàm lấy embedding [CLS] cho 1 list văn bản
# def get_bert_cls_embeddings(texts, batch_size=32, max_length=128):
#     all_embeddings = []
#     for i in tqdm(range(0, len(texts), batch_size)):
#         batch = texts[i:i+batch_size]
#         enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
#         with torch.no_grad():
#             out = bert(**enc)
#             # Lấy embedding của token [CLS] (vị trí 0)
#             cls_emb = out.last_hidden_state[:,0,:].cpu().numpy()
#             all_embeddings.append(cls_emb)
#     return np.vstack(all_embeddings)

# # 3. Lấy embedding cho train/test
# print("Đang tạo embedding DistilBERT cho train...")
# X_train_bert = get_bert_cls_embeddings(train_df['text'].tolist())
# print("Đang tạo embedding DistilBERT cho test...")
# X_test_bert = get_bert_cls_embeddings(test_df['text'].tolist())

# y_train = train_df['label_int'].values

# # 4. Train XGBoost
# print("Huấn luyện XGBoost...")
# model_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.7, random_state=42, use_label_encoder=False, eval_metric='logloss')
# model_xgb.fit(X_train_bert, y_train)

# # 5. Dự đoán xác suất cho test
# probs = model_xgb.predict_proba(X_test_bert)[:,1]

# # 6. Tạo file submission DUY NHẤT
# id_col = 'Id' if 'Id' in test_df.columns else 'id'
# submission_final = pd.DataFrame({
#     'Id': test_df[id_col],
#     'TARGET': probs
# })
# submission_final.to_csv('submission.csv', index=False)
# print("\nĐã tạo file 'submission.csv' thành công! Sẵn sàng để nộp bài.")
# print(submission_final.head())

