import pandas as pd
import re
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# --- Paths for the competition data ---
train_file_path = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
test_file_path = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
sample_submission_file_path = '/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv'

# --- Load dataframes ---
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
sample_submission_df = pd.read_csv(sample_submission_file_path)

# --- Memory Optimization: Use a smaller subset of data for debugging if needed ---
# Uncomment these lines to test with smaller data if you face Time/Memory Limit Exceeded errors.
# train_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True) # Use 10% of training data
# test_df = test_df.sample(frac=0.1, random_state=42).reset_index(drop=True)   # Use 10% of test data
# Or even smaller for initial tests:
# train_df = train_df.sample(n=1000, random_state=42).reset_index(drop=True)
# test_df = test_df.sample(n=100, random_state=42).reset_index(drop=True)

print("--- Train DataFrame Head ---")
print(train_df.head())
print("\n--- Test DataFrame Head ---")
print(test_df.head())
print("\n--- Sample Submission DataFrame Head ---")
print(sample_submission_df.head())

print("\n--- Initial Data Loading Complete ---")



print("--- Train DataFrame Shape ---")
print(train_df.shape)

print("\n--- Test DataFrame Shape ---")
print(test_df.shape)

print("\n--- Missing values in Train DataFrame ---")
print(train_df.isnull().sum())

print("\n--- Missing values in Test DataFrame ---")
print(test_df.isnull().sum())

# --- Text cleaning function ---
def clean_text(text):
    if pd.isna(text):
        return ""

    text = re.sub(r'\\frac\{([^{}]*?)\}\{([^{}]*?)\}', r'\1/\2', text)
    text = re.sub(r'\\cdot', '*', text)
    text = re.sub(r'\\left\(', '(', text)
    text = re.sub(r'\\right\)', ')', text)
    text = re.sub(r'\\frac', 'fraction', text)
    text = text.replace(r'\(', ' ').replace(r'\)', ' ')
    text = text.replace(r'\[', ' ').replace(r'\]', ' ')
    text = text.replace(r'\\newline', ' ')
    text = text.replace(r'\\', ' ')

    text = re.sub(r'[^a-zA-Z0-9\s/.*+-]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    text = text.lower()

    return text

print("\n--- Applying Text Cleaning ---")
train_df['StudentExplanation_Cleaned'] = train_df['StudentExplanation'].apply(clean_text)
train_df['QuestionText_Cleaned'] = train_df['QuestionText'].apply(clean_text)
train_df['MC_Answer_Cleaned'] = train_df['MC_Answer'].apply(clean_text)

test_df['StudentExplanation_Cleaned'] = test_df['StudentExplanation'].apply(clean_text)
test_df['QuestionText_Cleaned'] = test_df['QuestionText'].apply(clean_text)
test_df['MC_Answer_Cleaned'] = test_df['MC_Answer'].apply(clean_text)

print("Text cleaning complete. Displaying head of cleaned columns:")
print(train_df[['StudentExplanation', 'StudentExplanation_Cleaned',
                'QuestionText', 'QuestionText_Cleaned',
                'MC_Answer', 'MC_Answer_Cleaned']].head())

print("\n--- Data Preprocessing Complete ---")



train_df['target_label'] = train_df['Category'] + ':' + train_df['Misconception'].fillna('NA')

print("\n--- Unique Target Labels (Head) ---")
print(train_df['target_label'].value_counts().head(10))

unique_target_labels = train_df['target_label'].unique().tolist()
print(f"\nTotal unique target labels: {len(unique_target_labels)}")

label_to_id = {label: i for i, label in enumerate(unique_target_labels)}
id_to_label = {i: label for i, label in enumerate(unique_target_labels)}

train_df['target_id'] = train_df['target_label'].map(label_to_id)

print("\n--- Train DataFrame Head with Target ID ---")
print(train_df[['Category', 'Misconception', 'target_label', 'target_id']].head())

print("\n--- Label Encoding Complete ---")



# CRITICAL: THIS IS THE PATH TO THE BERT DATASET.
# Based on your latest screenshot, the correct path is '/kaggle/input/bertbaseuncased/bert-base-uncased/'
BERT_MODEL_PATH = '/kaggle/input/bertbaseuncased/bert-base-uncased/'

# --- Load the BERT tokenizer ---
tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL_PATH, # Pass the local directory path directly
    local_files_only=True,
    trust_remote_code=False,
    # No need for cache_dir here if BERT_MODEL_PATH is the direct source
)

print("Tokenizer loaded successfully!")

print("\nCombining text features for tokenization...")
train_df['combined_text'] = train_df['QuestionText_Cleaned'] + \
                            ' [SEP] ' + train_df['MC_Answer_Cleaned'] + \
                            ' [SEP] ' + train_df['StudentExplanation_Cleaned']

test_df['combined_text'] = test_df['QuestionText_Cleaned'] + \
                           ' [SEP] ' + test_df['MC_Answer_Cleaned'] + \
                           ' [SEP] ' + test_df['StudentExplanation_Cleaned']

print("Text combination complete. Example combined text:")
print(train_df['combined_text'].iloc[0])

print("\nTokenizing text. This might take a moment...")
train_encodings = tokenizer(
    train_df['combined_text'].tolist(),
    truncation=True,
    padding=True,
    max_length=96, # 
    return_tensors='pt'
)

test_encodings = tokenizer(
    test_df['combined_text'].tolist(),
    truncation=True,
    padding=True,
    max_length=96, 
    return_tensors='pt'
)

print("Tokenization complete.")

# --- Define the custom Dataset class ---
class MathMisconceptionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_labels = train_df['target_id'].tolist()
train_dataset = MathMisconceptionDataset(train_encodings, train_labels)

batch_size = 8 # মেমরি অপ্টিমাইজেশনের জন্য 16 থেকে কমানো হয়েছে
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- Load the BERT model for sequence classification ---
model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_PATH, # Pass the local directory path directly
    num_labels=len(unique_target_labels),
    local_files_only=True,
    trust_remote_code=False,
    # No need for cache_dir here if BERT_MODEL_PATH is the direct source
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(f"\nUsing device: {device}")

print("\n--- Tokenization and Model Loading Complete ---")



optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 1 

print(f"\nStarting training for {epochs} epochs...")
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss:.4f}")

print("\nTraining complete!")


del train_encodings
del train_labels
del train_dataset
del train_loader
if torch.cuda.is_available():
    torch.cuda.empty_cache() 
print("Training data and loader removed from memory.")



model.eval()
test_input_ids = test_encodings['input_ids'].to(device)
test_attention_mask = test_encodings['attention_mask'].to(device)

predictions = []

print("\nMaking predictions on test data...")
with torch.no_grad():
    outputs = model(test_input_ids, attention_mask=test_attention_mask)
    logits = outputs.logits

    predicted_ids = torch.argmax(logits, dim=1).tolist()

    for p_id in predicted_ids:
        predictions.append(id_to_label[p_id])

print("Predictions complete. Displaying first 5 predictions:")
print(predictions[:5])

del test_input_ids
del test_attention_mask
del test_encodings 
if torch.cuda.is_available():
    torch.cuda.empty_cache() 
print("Test data encodings removed from memory.")

# --- Create submission file ---
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': predictions
})

print("\n--- Sample Submission DataFrame Head ---")
print(submission_df.head())

submission_file_path = 'submission.csv'
submission_df.to_csv(submission_file_path, index=False)

print(f"\nSubmission file saved to {submission_file_path}")
print("You can now download this file from the Kaggle Notebook output and submit it to the competition.")

print("\n--- Notebook execution complete ---")


