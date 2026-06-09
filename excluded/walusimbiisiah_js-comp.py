import pandas as pd
import numpy as np
import os
import sys
import re
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# === 1. Load and Prepare Data ===
print("Loading competition datasets...")
train_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

# === 2. Improved Text Cleaning Function ===
def clean_text(text):
    """
    Cleans text by removing URLs, HTML tags, newlines, and extra whitespace,
    while preserving important punctuation.
    """
    # Ensure text is a string
    text = str(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Replace newlines and tabs with a single space
    text = re.sub(r'[\n\t]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === 3. Augment Training Data with Test Set Examples ===
print("Augmenting the training data with valuable examples from the test set...")

# Apply cleaning to original datasets
train_data['body'] = train_data['body'].apply(clean_text)
train_data['rule'] = train_data['rule'].apply(clean_text)
test_data['body'] = test_data['body'].apply(clean_text)
test_data['rule'] = test_data['rule'].apply(clean_text)
test_data['positive_example_1'] = test_data['positive_example_1'].apply(clean_text)
test_data['positive_example_2'] = test_data['positive_example_2'].apply(clean_text)
test_data['negative_example_1'] = test_data['negative_example_1'].apply(clean_text)
test_data['negative_example_2'] = test_data['negative_example_2'].apply(clean_text)

# Extract positive examples from the test set. These are rule violations.
test_pos_examples = pd.DataFrame({
    'body': test_data['positive_example_1'].tolist() + test_data['positive_example_2'].tolist(),
    'rule': test_data['rule'].tolist() + test_data['rule'].tolist(),
    'subreddit': test_data['subreddit'].tolist() + test_data['subreddit'].tolist(),
    'rule_violation': 1
})

# Extract negative examples from the test set. These are NOT rule violations.
test_neg_examples = pd.DataFrame({
    'body': test_data['negative_example_1'].tolist() + test_data['negative_example_2'].tolist(),
    'rule': test_data['rule'].tolist() + test_data['rule'].tolist(),
    'subreddit': test_data['subreddit'].tolist() + test_data['subreddit'].tolist(),
    'rule_violation': 0
})

# Concatenate and sample a fraction to avoid memory issues
additional_data = pd.concat([test_pos_examples, test_neg_examples]).sample(frac=0.1, random_state=42).reset_index(drop=True)

# Combine original training data with the new examples
augmented_train_data = pd.concat([train_data, additional_data], ignore_index=True)
print(f"Original training data size: {len(train_data)}")
print(f"Augmented training data size: {len(augmented_train_data)}")

# === 4. Feature Engineering: Create a Single Input String ===
print("Creating a single contextual input string for the model...")
def create_input_text(row):
    """Combines rule, subreddit, and body into a single formatted string."""
    return f"rule: {row['rule']} subreddit: {row['subreddit']} body: {row['body']}"

augmented_train_data['model_input'] = augmented_train_data.apply(create_input_text, axis=1)
test_data['model_input'] = test_data.apply(create_input_text, axis=1)

# Split data for training and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    augmented_train_data['model_input'].tolist(),
    augmented_train_data['rule_violation'].tolist(),
    test_size=0.1,
    random_state=42
)

# === 5. Load Model and Tokenizer ===
print("Loading the trusted RoBERTa model from the original source...")
# ⚠️ Using the new model path you provided
model_path = "/kaggle/input/roberta-base/roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
model = RobertaForSequenceClassification.from_pretrained(model_path, local_files_only=True, num_labels=2)

# Tokenize the datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_data['model_input'].tolist(), truncation=True, padding=True, max_length=512)

# === 6. Create Custom PyTorch Datasets ===
class JigsawDataset(torch.utils.data.Dataset):
    """A custom dataset class for Jigsaw competition data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = JigsawDataset(train_encodings, train_labels)
val_dataset = JigsawDataset(val_encodings, val_labels)

# === 7. Configure and Run Training ===
os.environ["WANDB_DISABLED"] = "true"
model.to("cuda")

training_args = TrainingArguments(
    output_dir='./training_results',
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    logging_dir='./logs',
    report_to=[],
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting the model training process...")
trainer.train()

# === 8. Prediction and Submission ===
print("Generating predictions for the test set...")
test_dummy_labels = [0] * len(test_data)
test_dataset = JigsawDataset(test_encodings, test_dummy_labels)
test_outputs = trainer.predict(test_dataset)

# Convert logits to probabilities
probabilities = torch.nn.functional.softmax(torch.tensor(test_outputs.predictions), dim=1)[:, 1].numpy()

# Create the submission file
submission_df = pd.DataFrame({
    "row_id": test_data["row_id"],
    "rule_violation": probabilities
})

submission_df.to_csv("submission.csv", index=False)
print("Submission file created successfully.")
print(submission_df.head())

