# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FAKE TEXT DETECTION MODEL")
print("="*60)

# Configuration - Optimized for Kaggle
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # Reduced for memory efficiency
BATCH_SIZE = 8    # Reduced for memory efficiency
LEARNING_RATE = 3e-5
EPOCHS = 3

print(f"Configuration:")
print(f"Model: {MODEL_NAME}")
print(f"Max Length: {MAX_LENGTH}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("\nLoading training data...")
train_df = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
print(f"Found {len(train_df)} training samples")

# Prepare training data
def load_texts(row):
    article_id = row['id']
    real_text_id = row['real_text_id']
    
    # Format article ID to match directory names
    formatted_article_id = f"article_{int(article_id):04d}"
    
    path1 = f"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/{formatted_article_id}/file_1.txt"
    path2 = f"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/{formatted_article_id}/file_2.txt"
    
    # Check if files exist
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"Warning: Missing files for article {formatted_article_id}")
        return []
    
    try:
        with open(path1, 'r', encoding='utf-8', errors='ignore') as f:
            text1 = f.read()
        with open(path2, 'r', encoding='utf-8', errors='ignore') as f:
            text2 = f.read()
        
        # Real text gets label 1, fake gets label 0
        if real_text_id == 1:
            return [(text1, 1), (text2, 0)]
        else:
            return [(text2, 1), (text1, 0)]
    except Exception as e:
        print(f"Error reading files for article {formatted_article_id}: {e}")
        return []

# Create training dataset
print("Loading text files...")
train_data = []
for idx, row in train_df.iterrows():
    if idx % 20 == 0:
        print(f"Processing sample {idx}/{len(train_df)}")
    train_data.extend(load_texts(row))

if len(train_data) == 0:
    raise ValueError("No training data loaded. Check file paths.")

print(f"Successfully loaded {len(train_data)} text samples")

# Prepare data
texts, labels = zip(*train_data)
label_dist = pd.Series(labels).value_counts().to_dict()
print(f"Label distribution: {label_dist}")

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set: {len(train_texts)} samples")
print(f"Validation set: {len(val_texts)} samples")

# Tokenizer
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("Creating datasets...")
# Create datasets
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Model
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
).to(device)

print("Model loaded successfully!")
print(f"Model parameters: {model.num_parameters():,}")

# Training arguments - NO WANDB, NO LOGGING
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="no",  # Don't save checkpoints
    load_best_model_at_end=False,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_first_step=True,
    report_to=[],  # DISABLE ALL EXTERNAL LOGGING
    seed=42,
    fp16=torch.cuda.is_available(),  # Mixed precision
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

print("="*50)
print("STARTING TRAINING...")
print("="*50)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train model
print("ğŸš€ Starting training process...")
try:
    train_result = trainer.train()
    print("âœ… Training completed successfully!")
    print(f"Training metrics: {train_result.metrics}")
    
    # Evaluate model
    print("\nğŸ“Š Evaluating model...")
    eval_result = trainer.evaluate()
    print("âœ… Evaluation completed!")
    for key, value in eval_result.items():
        print(f"  {key}: {value:.4f}")
    
except Exception as e:
    print(f"â�Œ Training error: {e}")
    import traceback
    traceback.print_exc()

# Test prediction function
def predict_texts(text1, text2, model, tokenizer, device, max_length=256):
    model.eval()
    
    # Tokenize texts
    inputs1 = tokenizer(
        text1, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=max_length
    ).to(device)
    
    inputs2 = tokenizer(
        text2, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=max_length
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        
        import torch.nn.functional as F
        probs1 = F.softmax(outputs1.logits, dim=-1)
        probs2 = F.softmax(outputs2.logits, dim=-1)
        
        # Get probability of being real (label 1)
        prob_real_1 = probs1[0][1].item()
        prob_real_2 = probs2[0][1].item()
        
    # Return prediction: 1 if text1 is real, 2 if text2 is real
    return 1 if prob_real_1 > prob_real_2 else 2

# Prepare test data
print("\n" + "="*50)
print("PROCESSING TEST DATA...")
print("="*50)

test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
test_articles = os.listdir(test_dir)
test_data = []

print(f"Found {len(test_articles)} test articles")

for idx, article_dir in enumerate(test_articles):
    if idx % 50 == 0:
        print(f"Processing test article {idx}/{len(test_articles)}")
    
    try:
        article_id = int(article_dir.replace('article_', ''))
        path1 = os.path.join(test_dir, article_dir, "file_1.txt")
        path2 = os.path.join(test_dir, article_dir, "file_2.txt")
        
        if os.path.exists(path1) and os.path.exists(path2):
            with open(path1, 'r', encoding='utf-8', errors='ignore') as f:
                text1 = f.read()
            with open(path2, 'r', encoding='utf-8', errors='ignore') as f:
                text2 = f.read()
            
            test_data.append((article_id, text1, text2))
    except Exception as e:
        print(f"Error processing {article_dir}: {e}")

print(f"Successfully loaded {len(test_data)} test samples")

# Make predictions
print("\n" + "="*50)
print("MAKING PREDICTIONS...")
print("="*50)

predictions = []
for idx, (article_id, text1, text2) in enumerate(test_data):
    if idx % 50 == 0:
        print(f"Predicting sample {idx}/{len(test_data)}")
    
    try:
        pred = predict_texts(text1, text2, model, tokenizer, device, MAX_LENGTH)
        predictions.append((article_id, pred))
    except Exception as e:
        print(f"Error predicting for article {article_id}: {e}")
        # Default to 1 if error
        predictions.append((article_id, 1))

# Create submission
print("\nCreating submission file...")
submission_df = pd.DataFrame(predictions, columns=['id', 'real_text_id'])
submission_df = submission_df.sort_values('id')
submission_df.to_csv('submission.csv', index=False)

print("âœ… Submission file created!")
print(f"Total predictions: {len(predictions)}")
print("\nFirst 10 predictions:")
print(submission_df.head(10))

# Summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(predictions)}")
if 'eval_f1' in locals():
    print(f"Validation F1: {eval_result.get('eval_f1', 0):.4f}")
    print(f"Validation Accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
print("âœ… Notebook completed successfully!")




