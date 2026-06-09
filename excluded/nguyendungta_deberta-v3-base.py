!pip install -q transformers datasets evaluate


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
import torch.nn as nn 

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers import EarlyStoppingCallback


# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sp_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")


def create_input_text(df):
    return (
        "Question: " + df["QuestionText"].str.strip() +
        " [SEP] Chosen Answer: " + df["MC_Answer"].str.strip() +
        " [SEP] Student Explanation: " + df["StudentExplanation"].str.strip()
    )

train_df['input_text'] = create_input_text(train_df)
test_df['input_text'] = create_input_text(test_df)

train_df['Misconception'] = train_df['Misconception'].fillna('None')
train_df['target'] = train_df['Category'] + ':' + train_df['Misconception']
label_encoder = LabelEncoder()
train_df['labels'] = label_encoder.fit_transform(train_df['target'])


class Config:
    MODEL_NAME = "microsoft/deberta-v3-base"  # A strong and efficient baseline model
    MAX_LENGTH = 512       # Max sequence length for the tokenizer
    BATCH_SIZE = 8         # Batch size for training and evaluation
    LEARNING_RATE = 2e-5   # Learning rate for the AdamW optimizer
    EPOCHS = 20      # Number of training epochs
    N_SPLITS = 5           # Number of folds for cross-validation
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {Config.DEVICE}")


def map_at_3(predictions, labels):
    """
    Computes the Mean Average Precision @ 3.
    """
    # Get the top 3 predicted labels (indices) for each example
    top3_preds = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]

    avg_precisions = []
    for i, true_label in enumerate(labels):
        # The true label is a single integer
        pred_labels = top3_preds[i]
        
        score = 0.0
        num_hits = 0.0
        
        for j, p in enumerate(pred_labels):
            if p == true_label:
                num_hits += 1.0
                score += num_hits / (j + 1.0)
        
        avg_precisions.append(score)
        
    return np.mean(avg_precisions)

def compute_metrics(eval_pred):
    """
    Custom metric computation function for the Trainer.
    """
    logits, labels = eval_pred
    return {"map@3": map_at_3(logits, labels)}



from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    # stratify=train_df["labels"], # <--- Báº­t láº¡i stratify
    random_state=42
)

# train_df = train_df.sample(n=100, random_state=42)
# val_df = val_df.sample(n=100, random_state=42)


print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

# --- Convert to Hugging Face Datasets ---
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# --- Tokenization ---
def tokenize_function(examples):
    return tokenizer(
        examples["input_text"],
        max_length=Config.MAX_LENGTH,
        truncation=True,
        # padding="max_length"
    )

columns_to_remove = list(train_dataset.column_names)
columns_to_remove.remove("labels")

train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
test_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

print(f"Columns in tokenized train set: {train_tokenized.column_names}")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# --- Initialize model ---
model = AutoModelForSequenceClassification.from_pretrained(
    Config.MODEL_NAME,
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
).to(Config.DEVICE)

# print(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "query_proj",
        "value_proj",
        "key_proj"  # Báº¡n cÅ©ng cÃ³ thá»ƒ thÃªm lá»›p nÃ y
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS 
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./holdout_validation",
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=Config.BATCH_SIZE,
    per_device_eval_batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    num_train_epochs=Config.EPOCHS,
    warmup_ratio=0.1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="map@3",
    greater_is_better=True,
    save_total_limit=1,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    data_collator=data_collator  
)

# --- Train model ---
trainer.train()

# --- Log & best score ---
# (Giá»¯ nguyÃªn)
log_history = trainer.state.log_history
training_logs = [log for log in log_history if 'loss' in log]
validation_logs = [log for log in log_history if 'eval_loss' in log]

train_log_df = pd.DataFrame(training_logs)
val_log_df = pd.DataFrame(validation_logs)

best_log = val_log_df.sort_values(by='eval_map@3', ascending=False).iloc[0]
print(f"\nâœ… Best Validation MAP@3: {best_log['eval_map@3']:.4f} at Epoch {int(best_log['epoch'])}\n")
print("Huáº¥n luyá»‡n PEFT hoÃ n táº¥t. Adapter Ä‘Ã£ Ä‘Æ°á»£c lÆ°u.")


from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# --- Predict on validation set ---
predictions_output = trainer.predict(val_tokenized)

logits = predictions_output.predictions
labels = predictions_output.label_ids
preds = np.argmax(logits, axis=1)

# --- Confusion matrix ---
cm = confusion_matrix(labels, preds, labels=np.unique(labels))

# --- Plot confusion matrix ---
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# --- Classification report (only existing classes in val set) ---
unique_labels = np.unique(labels)
target_names = label_encoder.inverse_transform(unique_labels)

print(classification_report(labels, preds, labels=unique_labels, target_names=target_names))

unique_labels = np.unique(labels)
target_names = label_encoder.inverse_transform(unique_labels)

# --- Overall metrics (macro and weighted averages) ---
precision_macro = precision_score(labels, preds, average='macro')
recall_macro = recall_score(labels, preds, average='macro')
f1_macro = f1_score(labels, preds, average='macro')

precision_weighted = precision_score(labels, preds, average='weighted')
recall_weighted = recall_score(labels, preds, average='weighted')
f1_weighted = f1_score(labels, preds, average='weighted')

print(f"\nğŸ”� Macro Precision: {precision_macro:.4f}")
print(f"ğŸ”� Macro Recall:    {recall_macro:.4f}")
print(f"ğŸ”� Macro F1-score:  {f1_macro:.4f}")
print(f"ğŸ”� Weighted F1:     {f1_weighted:.4f}")



import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(train_log_df['epoch'], train_log_df['loss'], label='Training Loss')
ax1.plot(val_log_df['epoch'], val_log_df['eval_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(val_log_df['epoch'], val_log_df['eval_map@3'], marker='o', linestyle='-', label='Validation MAP@3')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAP@3 Score')
ax2.legend()
plt.show()


predictions_output = trainer.predict(test_tokenized)
logits = predictions_output.predictions
test_predictions = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

top3_indices = np.argsort(test_predictions, axis=1)[:, -3:][:, ::-1]

# Decode the indices back to their original string labels
predictions_as_labels = label_encoder.inverse_transform(top3_indices.flatten())
predictions_as_labels = predictions_as_labels.reshape(top3_indices.shape)

# Format for submission
submission_strings = [' '.join(pred_row) for pred_row in predictions_as_labels]

submission_df = pd.DataFrame({'Id': test_df.index, 'Category:Misconception': submission_strings})

# Save to csv
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(submission_df.head())




