import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Đọc file train
df = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")

# Encode nhãn
label2id = {"benign": 0, "jailbreak": 1}
id2label = {0: "benign", 1: "jailbreak"}
df["label"] = df["label"].map(label2id)

# Chia dữ liệu: 80% train, 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Chuyển sang Dataset Hugging Face
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))



from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Giả sử bạn đã upload model lên Kaggle và lưu trong thư mục './my_model'
model_path = "/kaggle/input/m/guychahine/distilbertdistilbert-base-uncased/transformers/default/1"

# Load tokenizer từ local
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model từ local
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


!pip install evaluate


from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np

model_path = "/kaggle/input/distilbertdistilbert-base-uncased/transformers/default/1"  # điều chỉnh theo folder model của bạn
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# --- Load metrics ---
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# --- Hàm tính toán metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=preds, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=preds, references=labels, average="weighted")

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",       # Đánh giá sau mỗi epoch
    save_strategy="epoch",             # Lưu model mỗi epoch
    load_best_model_at_end=True,       # Tự động load model tốt nhất sau khi train
    metric_for_best_model="accuracy",  # Metric dùng để chọn model tốt nhất
    greater_is_better=True,            # accuracy càng cao càng tốt
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",                  # Tắt Weights & Biases
    run_name="distilbert-jailbreak-v2",
    num_train_epochs=6,                # Tăng số epoch để mô hình hội tụ tốt hơn
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,                # LR mặc định tốt cho DistilBERT
    weight_decay=0.01,                 # Regularization nhẹ để tránh overfit
    warmup_ratio=0.1,                  # Warmup 10% đầu epochs
    save_total_limit=2,                # Giữ lại tối đa 2 checkpoint
    logging_first_step=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)



train_result = trainer.train()


trainer.save_model("./best_model")


tokenizer.save_pretrained("./best_model")


trainer.evaluate()


import matplotlib.pyplot as plt

logs = trainer.state.log_history

train_loss = [x["loss"] for x in logs if "loss" in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
eval_acc = [x["eval_accuracy"] for x in logs if "eval_accuracy" in x]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(eval_acc, label="Validation Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.show()


import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model + tokenizer
model_path = "/kaggle/working/best_model"
print(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # inference mode

# Load test data
test_df = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")

# Tokenize test set
test_encodings = tokenizer(
    test_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

# Predict
with torch.no_grad():
    outputs = model(**test_encodings)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1]  # Xác suất class 1 (jailbreak)

# Tạo file submission
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "TARGET": probs.numpy()
})

# Lưu đúng định dạng Kaggle
submission.to_csv("submission.csv", index=False)
print(submission.head())

