import os
import tarfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ãƒ¢ãƒ‡ãƒ«ä¿�å­˜é–¢é€£
MODEL_DIR = "/kaggle/working/brainappmodel"
ARCHIVE_PATH = "/kaggle/input/modernbert-brainapp/ModernBERT_finetuned.tar.gz"

def extract_model():
    if os.path.exists(ARCHIVE_PATH):
        print("æ—¢å­˜ãƒ¢ãƒ‡ãƒ«å±•é–‹ä¸­...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=".")
        return True
    return False

def save_model(model, tokenizer):
    print("ãƒ¢ãƒ‡ãƒ«ã�¨ãƒˆãƒ¼ã‚¯ãƒŠã‚¤ã‚¶ãƒ¼ã‚’ä¿�å­˜ãƒ»åœ§ç¸®ä¸­...")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # åœ§ç¸®ãƒ•ã‚¡ã‚¤ãƒ«ã‚’ /kaggle/working/ ã�«ç›´æ�¥ä¿�å­˜
    archive_path = "/kaggle/working/brainappmodel.tar.gz"  # â†� ã�“ã�“ã‚’å¥½ã��ã�ªãƒ•ã‚¡ã‚¤ãƒ«å��ã�«å¤‰æ›´å�¯èƒ½

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(MODEL_DIR, arcname=os.path.basename(MODEL_DIR))

    print(f"åœ§ç¸®å®Œäº†: {archive_path}")



# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
df_train = pd.read_csv('/kaggle/input/c/map-charting-student-math-misunderstandings/train.csv')
df_test = pd.read_csv('/kaggle/input/c/map-charting-student-math-misunderstandings/test.csv')

# æ¬ æ��è£œå®Œã�¨ãƒ©ãƒ™ãƒ«åˆ—ä½œæˆ�
df_train['Misconception'] = df_train['Misconception'].fillna("NA")
df_train['Target'] = df_train['Category'] + ":" + df_train['Misconception']

# ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‰
le = LabelEncoder()
df_train['TargetEncoded'] = le.fit_transform(df_train['Target'])
num_labels = len(le.classes_)

# Train / Valid åˆ†å‰²ï¼ˆå†�ç�¾æ€§ã�®ã�Ÿã‚� random_state å›ºå®šï¼‰

# ã‚¯ãƒ©ã‚¹ã�”ã�¨ã�®ä»¶æ•°ã‚’ã‚«ã‚¦ãƒ³ãƒˆ
label_counts = df_train['TargetEncoded'].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df_train_filtered = df_train[df_train['TargetEncoded'].isin(valid_labels)]

train_df, valid_df = train_test_split(
    df_train_filtered,
    test_size=0.2,
    stratify=df_train_filtered['TargetEncoded'],
    random_state=42
)



if extract_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/huggingfacedebertav3variants/deberta-v3-base")  ## answerdotai/ModernBERT-large
    model = AutoModelForSequenceClassification.from_pretrained(
        "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )



class MyDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(
            texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt'
        )
        self.labels = labels.tolist() if labels is not None else None

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = MyDataset(train_df['StudentExplanation'], train_df['TargetEncoded'])
valid_dataset = MyDataset(valid_df['StudentExplanation'], valid_df['TargetEncoded'])


from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir='/kaggle/working/results',
    run_name='modernbert-experiment1',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",    # è©•ä¾¡ã‚¿ã‚¤ãƒŸãƒ³ã‚°
    save_strategy="epoch",          # ä¿�å­˜ã‚¿ã‚¤ãƒŸãƒ³ã‚° â†� ã�“ã‚Œã‚’å�ˆã‚�ã�›ã‚‹
    logging_strategy="steps",         # â†� ã�“ã‚Œã�Œå¿…è¦�
    logging_steps=100,    
    load_best_model_at_end=True,    # ãƒ™ã‚¹ãƒˆãƒ¢ãƒ‡ãƒ«ã‚’æœ€å¾Œã�«ãƒ­ãƒ¼ãƒ‰
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to='none'
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


if not os.path.exists(MODEL_DIR):
    train_output = trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    # ğŸ”½ å­¦ç¿’ä¸­ã�®ãƒ­ã‚°å±¥æ­´ã‚’ç¢ºèª�
    import pandas as pd
    log_df = pd.DataFrame(trainer.state.log_history)
    print("ğŸ“‰ ãƒ­ã‚°å±¥æ­´ï¼ˆæŠœç²‹ï¼‰:")
    display(log_df[['step', 'loss', 'eval_loss']].dropna())

    # æœ€çµ‚çµ�æ�œã‚‚è¡¨ç¤º
    print("ğŸ“Š å­¦ç¿’å®Œäº†:", train_output.metrics)
else:
    print("âœ… å­¦ç¿’æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«ã‚’åˆ©ç”¨ä¸­")



test_dataset = MyDataset(df_test['StudentExplanation'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in test_dataset.encodings.items()}
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

top3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
top3_labels = le.inverse_transform(top3.flatten()).reshape(top3.shape)
joined_preds = [" ".join(row) for row in top3_labels]

submission = pd.DataFrame({
    'row_id': df_test['row_id'],
    'Category:Misconception': joined_preds
})
submission.to_csv("submission.csv", index=False)
submission.head()


training_log = trainer.state.log_history
train_loss = [log['loss'] for log in training_log if 'loss' in log]
eval_loss = [log['eval_loss'] for log in training_log if 'eval_loss' in log]

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()
plt.show()

