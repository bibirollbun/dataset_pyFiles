import pandas as pd


df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")


df.head()


df.columns


!pip install --upgrade transformers



df.columns


import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- 1. Load CSV ---
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")

# --- 2. Augment dataset ---
texts = []
labels = []

for _, row in df.iterrows():
    # Original comment
    texts.append("comment : "+str(row['body']) + " subredit : "+str(row['subreddit']))
    labels.append(row['rule_violation'])

    # Positive examples (label=1)
    for col in ['positive_example_1', 'positive_example_2']:
        if pd.notna(row[col]):
            texts.append("comment : "+str(row[col]) + " subredit : "+str(row['subreddit']))
            labels.append(1)

    # Negative examples (label=0)
    for col in ['negative_example_1', 'negative_example_2']:
        if pd.notna(row[col]):
            texts.append("comment : "+str(row[col]) + " subredit : "+str(row['subreddit']))
            labels.append(0)

print(f"Total augmented dataset size: {len(texts)}")

# --- 3. Split into train/test ---
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# --- 4. Tokenize ---
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

# --- 5. Create PyTorch Dataset ---
class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CommentDataset(train_encodings, train_labels)
test_dataset = CommentDataset(test_encodings, test_labels)

# --- 6. Load model on GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Changed from distilbert-base-uncased to xlm-roberta-base
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=2
)
model.to(device)

# --- 7. Define metrics ---
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

# --- 8. Training arguments ---
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,                # fewer epochs to avoid overfitting
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    no_cuda=False
)

# --- 9. Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# --- 10. Train with timing ---
import time
start_time = time.time()
print("Training started...")
trainer.train()
end_time = time.time()
print(f"Training finished in {round(end_time - start_time, 2)} seconds.")

# --- 11. Evaluate with timing ---
start_eval = time.time()
print("Evaluation started...")
eval_results = trainer.evaluate()
end_eval = time.time()
print(f"Evaluation finished in {round(end_eval - start_eval, 2)} seconds.")


print("Evaluation metrics:", eval_results)


from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Save model
model.save_pretrained("./our_xlmroberta_model_with_subredit/")

# Save tokenizer (optional but usually needed)
# Changed from distilbert-base-uncased to xlm-roberta-base
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizer.save_pretrained("./our_xlmroberta_model_with_subredit/")


!zip -r my_model.zip our_xlmroberta_model_with_subredit/


# Save model and tokenizer after training
model.save_pretrained("/kaggle/working/my_bert_model")
tokenizer.save_pretrained("/kaggle/working/my_bert_model")



from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer from local directory
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/working/my_bert_model")
tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/my_bert_model")



# test_data = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# texts = test_data['body'].astype(str).tolist()

# # Use the same tokenizer as during training
# test_encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)

# import torch

# class TestDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         return item

#     def __len__(self):
#         return len(self.encodings['input_ids'])

# test_dataset = TestDataset(test_encodings)

# # Make predictions
# predictions = trainer.predict(test_dataset)
# # Get class indices (0 or 1)
# pred_labels = predictions.predictions.argmax(-1)

# submission = pd.DataFrame({
#     'row_id': test_data['row_id'],
#     'rule_violation': pred_labels
# })

# submission.to_csv("submission.csv", index=False)
# print("Submission file created: submission.csv")




