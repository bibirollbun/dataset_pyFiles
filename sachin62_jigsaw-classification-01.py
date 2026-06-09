import pandas as pd
import numpy as np


df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
df_test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')


df.head()


from sklearn.model_selection import train_test_split

df['input_text'] = "Comment: " + df['body'] + " [SEP] Rule: " + df['rule']

X = df['input_text'].tolist()
y = df['rule_violation'].tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained("/kaggle/input/roberta-base/transformers/default/1")
model = RobertaForSequenceClassification.from_pretrained("/kaggle/input/roberta-base/transformers/default/1")


# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=512)



import torch
class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = RedditDataset(train_encodings, y_train)
val_dataset = RedditDataset(val_encodings, y_val)



# from transformers import RobertaForSequenceClassification

# model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)


model


import os
import sys
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("/kaggle/input/roberta-base/transformers/default/1").to("cuda")

# Define training args with proper logging
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    logging_dir='./logs',
    report_to=[],
    disable_tqdm=False
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)



trainer.train()


from sklearn.metrics import roc_auc_score

# Get predictions
preds = trainer.predict(val_dataset)
probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()

# Evaluate
auc = roc_auc_score(y_val, probs)
print(f"Validation AUC: {auc:.4f}")






## on test data 


df_test['input_text'] = "Comment: " + df_test['body'] + " [SEP] Rule: " + df_test['rule']

test_encodings = tokenizer(df_test['input_text'].tolist(), truncation=True, padding=True, max_length=512)

dummy_labels = [0] * len(df_test)
test_dataset = RedditDataset(test_encodings, dummy_labels)

test_outputs = trainer.predict(test_dataset)
probs = torch.nn.functional.softmax(torch.tensor(test_outputs.predictions), dim=1)[:, 1].numpy()

# Step 5: Create submission file
submission_df = pd.DataFrame({
    "row_id": df_test["row_id"],
    "rule_violation": probs
})
submission_df.to_csv("submission.csv", index=False)








