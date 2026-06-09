# Part 1: Training Script
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, ClassLabel
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import os

print("="*60)
print("Part 1: Training and Saving the Model")
print("="*60)

# --- 1. 环境与路径设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 预训练模型的路径 (从Kaggle数据集加载)
PRETRAINED_MODEL_PATH = 'bert-base-uncased'
# 训练好的模型的保存路径
FINAL_MODEL_OUTPUT_PATH = './final_model'

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
print(f"Using device: {device}")
print(f"Loading pretrained model from: {PRETRAINED_MODEL_PATH}")
print(f"Final model will be saved to: {FINAL_MODEL_OUTPUT_PATH}")

# --- 2. 加载和预处理数据 ---
print("\nLoading and preparing data...")
dataset = load_dataset(
    'csv',
    data_files={'train': '/kaggle/input/quora-insincere-questions-classification/train.csv'},
    delimiter=','
)

class_label_feature = ClassLabel(num_classes=2, names=['sincere', 'insincere'])
dataset['train'] = dataset['train'].cast_column('target', class_label_feature)

split = dataset['train'].train_test_split(test_size=0.1, seed=42, stratify_by_column='target')
train_ds = split['train']
valid_ds = split['test']
print(f"Training set: {len(train_ds)} samples | Validation set: {len(valid_ds)} samples")

# --- 3. 初始化分词器和模型 ---
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=2)
model.to(device)

# --- 4. 数据编码 ---
max_length = 64
def tokenize_fn(examples):
    return tokenizer(
        examples['question_text'],
        truncation=True,
        max_length=max_length
    )

print("\nEncoding datasets...")
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['question_text'])
valid_ds = valid_ds.map(tokenize_fn, batched=True, remove_columns=['question_text'])
train_ds = train_ds.rename_column("target", "labels")
valid_ds = valid_ds.rename_column("target", "labels")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Encoding complete.")

# --- 5. 定义评估指标和带权重的损失函数 ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {'f1': f1_score(labels, preds)}

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_ds['labels']),
    y=train_ds['labels']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- 6. 配置训练参数 ---
print("\nConfiguring training arguments...")
training_args = TrainingArguments(
    output_dir='./results',  # 检查点保存目录
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    report_to="none",
)

# --- 7. 创建并开始训练 ---
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\n" + "="*60)
print("Starting training...")
print("="*60)
trainer.train()
print("\nTraining complete!")

# --- 8. 评估并保存最终模型 ---
print("\nFinal evaluation on validation set...")
eval_results = trainer.evaluate()
print(f"Validation set F1 Score: {eval_results.get('eval_f1', 'N/A'):.4f}")

print(f"\nSaving the best model to '{FINAL_MODEL_OUTPUT_PATH}'...")
trainer.save_model(FINAL_MODEL_OUTPUT_PATH)
print("Model saved successfully.")
print("\nPart 1 finished.")

# 12. Inference and submission
print("\nStarting inference...")

test_df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

# Create a dataset for inference
test_dataset = load_dataset('csv', data_files={'test': '/kaggle/input/quora-insincere-questions-classification/test.csv'})['test']
encoded_test = test_dataset.map(tokenize_fn, batched=True)

print("Predicting...")
preds_output = trainer.predict(encoded_test)
preds = np.argmax(preds_output.predictions, axis=1)

# Create submission file
submission_df = pd.DataFrame({'qid': test_df['qid'], 'prediction': preds})
submission_df.to_csv('submission.csv', index=False)

print("\nComplete!")
print("Results have been saved to submission.csv")




