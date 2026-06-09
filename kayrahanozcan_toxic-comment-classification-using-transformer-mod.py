import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re # Regular expressions for text cleaning (use cautiously with BERT)

# Deep Learning Framework - Using PyTorch here
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW

# Hugging Face Transformers
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, hamming_loss, accuracy_score # Accuracy is less informative for multi-label

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # if using CUDA

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration (Adjust these as needed)
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128 # Max sequence length BERT can handle (adjust based on EDA)
BATCH_SIZE = 16 # Adjust based on GPU memory
EPOCHS = 3 # Number of training epochs (BERT fine-tuning usually requires few epochs)
LEARNING_RATE = 2e-5 # Common learning rate for BERT fine-tuning


# Adjust file paths as necessary
try:
    train_df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
    test_df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
    sample_submission_df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset files not found. Please check the input path.")
    # Dummy data for script structure execution if files are missing
    train_df = pd.DataFrame({
        'id': ['1','2','3','4','5'],
        'comment_text': ['This is fine.', 'This is bad and obscene!', 'You are an idiot.', 'Explanation why the edits made under my username Hardcore Metallica Fan were reverted?', 'Go away!'],
        'toxic': [0,1,1,0,0], 'severe_toxic': [0,0,0,0,0], 'obscene': [0,1,0,0,0],
        'threat': [0,0,0,0,0], 'insult': [0,0,1,0,0], 'identity_hate': [0,0,0,0,0]
    })
    test_df = pd.DataFrame({
        'id': ['10','11'],
        'comment_text': ['Testing one two.', 'Another comment here.']
    })
    sample_submission_df = pd.DataFrame({
        'id': ['10','11'], 'toxic': [0.5]*2, 'severe_toxic': [0.5]*2, 'obscene': [0.5]*2,
        'threat': [0.5]*2, 'insult': [0.5]*2, 'identity_hate': [0.5]*2
    })

print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)

# Define target labels
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("\nTrain Data Head:")
print(train_df.head())
print("\nTest Data Head:")
print(test_df.head())


# Calculate label counts and percentages
label_counts = train_df[label_cols].sum()
label_percentages = (label_counts / len(train_df)) * 100

# Create a DataFrame for plotting
label_stats_df = pd.DataFrame({'count': label_counts, 'percentage': label_percentages})
print("Label Distribution:")
print(label_stats_df)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x=label_stats_df.index, y=label_stats_df['count'])
plt.title('Distribution of Toxicity Labels')
plt.ylabel('Number of Comments')
plt.xlabel('Toxicity Type')
plt.xticks(rotation=45)
plt.show()

# Check for comments with no labels (clean comments)
no_label_count = len(train_df[train_df[label_cols].sum(axis=1) == 0])
print(f"\nNumber of comments with NO toxicity labels: {no_label_count} ({no_label_count / len(train_df) * 100:.2f}%)")

# Multi-label Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(train_df[label_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Toxicity Labels')
plt.show()
# Observation: High correlation between 'toxic' and 'obscene'/'insult'. 'Severe toxic' implies 'toxic'.


train_df['comment_length'] = train_df['comment_text'].apply(lambda x: len(x.split()))
test_df['comment_length'] = test_df['comment_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 6))
sns.histplot(train_df['comment_length'], bins=100, kde=True)
plt.title('Distribution of Comment Length (Words)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
# Limit x-axis for better visualization if needed
plt.xlim(0, 500)
plt.show()

print("\nComment Length Statistics (Words):")
print(train_df['comment_length'].describe())
# Observation: Most comments are well below 512 tokens, but some are longer. MAX_LENGTH=128 or 256 might be reasonable start.


def clean_text(text):
    text = str(text)
    # Remove URLs (optional, BERT might handle some context)
    # text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Basic handling of common issues if needed
    # text = text.lower() # BERT uncased models handle this
    return text

# Apply cleaning (demonstrative - may skip depending on BERT variant)
# train_df['comment_text_cleaned'] = train_df['comment_text'].apply(clean_text)
# test_df['comment_text_cleaned'] = test_df['comment_text'].apply(clean_text)
# Use original text for now as BERT benefits from closer-to-raw text
train_df['comment_text_cleaned'] = train_df['comment_text']
test_df['comment_text_cleaned'] = test_df['comment_text']


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Example tokenization
sample_text = "This is a sample comment for tokenization."
tokens = tokenizer.encode_plus(
    sample_text,
    max_length=32,
    padding='max_length', # Pad to max_length
    truncation=True,      # Truncate longer sequences
    return_tensors='pt'   # Return PyTorch tensors
)

print("\nSample Tokenization:")
print(f"Text: {sample_text}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
print(f"Input IDs: {tokens['input_ids']}")
print(f"Attention Mask: {tokens['attention_mask']}") # 1 for real tokens, 0 for padding


class ToxicCommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        target = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,
            return_token_type_ids=False, # Not needed for basic BERT classification
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', # Return PyTorch tensors
        )

        return {
            'comment_text': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.float) # Use float for BCEWithLogitsLoss
        }

# Prepare data for Dataset class
X = train_df['comment_text_cleaned'].values
y = train_df[label_cols].values

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1, # Use 10% for validation
    random_state=SEED,
    # Stratification is complex for multi-label, can skip or use iterative stratification if needed
)

print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")

# Create Datasets
train_dataset = ToxicCommentDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = ToxicCommentDataset(X_val, y_val, tokenizer, MAX_LENGTH)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) # num_workers depends on your system
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Example batch check
data = next(iter(train_dataloader))
print("\nSample batch shapes:")
print("Input IDs:", data['input_ids'].shape)
print("Attention Mask:", data['attention_mask'].shape)
print("Labels:", data['labels'].shape)


# Load BertForSequenceClassification, configuring it for multi-label
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_cols), # Number of output labels = number of toxic categories
    output_attentions=False, # Optional: set to True if you want attention weights
    output_hidden_states=False, # Optional: set to True if you want hidden states
)

# Move the model to the designated device (GPU or CPU)
model.to(device)

print("\nModel loaded successfully.")
# print(model) # Uncomment to see model architecture details


# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

# Total number of training steps
total_steps = len(train_dataloader) * EPOCHS

# Learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0, # Optional: set a number of warmup steps (e.g., 0.1 * total_steps)
    num_training_steps=total_steps
)

# Loss function for multi-label classification
# BCEWithLogitsLoss combines a Sigmoid layer and Binary Cross Entropy loss in one class.
# It's numerically more stable than using a plain Sigmoid followed by BCE Loss.
loss_fn = nn.BCEWithLogitsLoss().to(device)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    for i, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits # Raw model output (before sigmoid)

        # Calculate loss
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients (common practice)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()
        scheduler.step() # Update learning rate

        # Print progress (optional)
        if (i + 1) % 100 == 0:
             print(f'  Batch {i + 1}/{num_batches} | Loss: {loss.item():.4f}')


    avg_train_loss = total_loss / num_batches
    print(f"\n  Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def eval_model(model, data_loader, loss_fn, device):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = len(data_loader)

    with torch.no_grad(): # Disable gradient calculation
        for batch in data_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            # Calculate loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Store predictions (probabilities) and true labels
            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = total_loss / num_batches
    print(f"  Average Validation Loss: {avg_val_loss:.4f}")

    # Concatenate results from all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics (example: ROC AUC per label, then average)
    # Note: Handling potential errors if a label has only one class in the validation batch/set
    roc_auc_scores = {}
    mean_roc_auc = 0
    try:
        # Calculate AUC for each label individually
        for i, label_name in enumerate(label_cols):
             # Check if both classes are present for the current label
             if len(np.unique(all_labels[:, i])) > 1:
                 roc_auc_scores[label_name] = roc_auc_score(all_labels[:, i], all_preds[:, i])
             else:
                 roc_auc_scores[label_name] = np.nan # Or 0.5, or skip
        # Calculate mean AUC, ignoring NaNs
        mean_roc_auc = np.nanmean(list(roc_auc_scores.values()))
        print(f"  Mean ROC AUC: {mean_roc_auc:.4f}")
        print("  Individual ROC AUC Scores:")
        for name, score in roc_auc_scores.items():
             print(f"    {name}: {score:.4f}")
    except Exception as e:
        print(f"  Could not calculate ROC AUC: {e}")


    # Calculate Hamming Loss (fraction of wrongly predicted labels)
    threshold = 0.5
    binary_preds = (all_preds > threshold).astype(int)
    hamming = hamming_loss(all_labels, binary_preds)
    print(f"  Hamming Loss: {hamming:.4f}")

    # (Optional) You can also calculate Micro/Macro F1 scores or Accuracy (less useful)
    # print("\nClassification Report (threshold=0.5):")
    # print(classification_report(all_labels, binary_preds, target_names=label_cols, zero_division=0))


    return avg_val_loss, mean_roc_auc, hamming # Return key metrics


history = {'train_loss': [], 'val_loss': [], 'val_roc_auc': [], 'val_hamming': []}
best_roc_auc = -1
best_model_state = None

print("\nStarting Training...")
for epoch in range(EPOCHS):
    print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')

    train_loss = train_epoch(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        device,
        scheduler
    )
    history['train_loss'].append(train_loss)

    print(f"\n--- Validation Epoch {epoch + 1} ---")
    val_loss, val_roc_auc, val_hamming = eval_model(
        model,
        val_dataloader,
        loss_fn,
        device
    )
    history['val_loss'].append(val_loss)
    history['val_roc_auc'].append(val_roc_auc)
    history['val_hamming'].append(val_hamming)

    # Save the best model based on validation ROC AUC
    if val_roc_auc > best_roc_auc:
        best_roc_auc = val_roc_auc
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model_state.bin')
        print(f"  ** New best model saved with ROC AUC: {best_roc_auc:.4f} **")

print("\nTraining Finished.")
print(f"Best Validation ROC AUC: {best_roc_auc:.4f}")

# Load the best model state for prediction
if best_model_state:
    model.load_state_dict(best_model_state)
    print("Loaded best model state for prediction.")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Train Loss')
plt.plot(range(1, EPOCHS + 1), history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), history['val_roc_auc'], label='Validation Mean ROC AUC')
plt.title('Mean ROC AUC Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean ROC AUC')
plt.legend()

plt.tight_layout()
plt.show()


class TestCommentDataset(Dataset):
    def __init__(self, comments, tokenizer, max_len):
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

test_texts = test_df['comment_text_cleaned'].values
test_dataset = TestCommentDataset(test_texts, tokenizer, MAX_LENGTH)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    print("\nGenerating predictions on test data...")
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.sigmoid(logits) # Convert logits to probabilities (0-1 range)
            predictions.append(probs.cpu().numpy())

    return np.concatenate(predictions, axis=0)

test_predictions = predict(model, test_dataloader, device)
print("Predictions generated successfully.")
print("Shape of predictions:", test_predictions.shape) # Should be (num_test_samples, num_labels)


# Create submission DataFrame
submission_df = pd.DataFrame(test_predictions, columns=label_cols)
submission_df['id'] = test_df['id'] # Add the id column

# Reorder columns to match sample_submission.csv format ('id' first, then label columns)
submission_df = submission_df[['id'] + label_cols]

print("\nSubmission DataFrame Head:")
print(submission_df.head())

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully.")

