import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the data
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

# Basic info
print("=" * 80)
print("TRAINING DATA")
print("=" * 80)
print(f"Shape: {train.shape}")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nData types:\n{train.dtypes}")
print(f"\nNull values:\n{train.isnull().sum()}")
print(f"\nTarget distribution:\n{train['rule_violation'].value_counts()}")
print(f"\nUnique rules in train: {train['rule'].nunique()}")
print(f"Rules: {train['rule'].unique()}")
print(f"\nUnique subreddits: {train['subreddit'].nunique()}")

print("\n" + "=" * 80)
print("TEST DATA")
print("=" * 80)
print(f"Shape: {test.shape}")
print(f"\nColumns: {test.columns.tolist()}")
print(f"\nUnique rules in test: {test['rule'].nunique()}")
print(f"Rules: {test['rule'].unique()}")
print(f"\nUnique subreddits in test: {test['subreddit'].nunique()}")

print("\n" + "=" * 80)
print("SAMPLE ROWS")
print("=" * 80)
print("\nTrain sample:")
print(train.head(2))
print("\nTest sample:")
print(test.head(2))

# Check for overlap between train and test rules
train_rules = set(train['rule'].unique())
test_rules = set(test['rule'].unique())
print(f"\nRules only in train: {train_rules - test_rules}")
print(f"Rules only in test: {test_rules - train_rules}")
print(f"Overlapping rules: {train_rules & test_rules}")


# Text length analysis
train['body_length'] = train['body'].str.len()
train['body_words'] = train['body'].str.split().str.len()

print("=" * 80)
print("TEXT LENGTH STATISTICS")
print("=" * 80)
print("\nComment body character length:")
print(train['body_length'].describe())
print(f"\nMax length: {train['body_length'].max()}")
print(f"Comments > 500 chars: {(train['body_length'] > 500).sum()}")

print("\nComment body word count:")
print(train['body_words'].describe())

# Analyze examples lengths
for col in ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']:
    train[f'{col}_len'] = train[col].str.len()
    print(f"\n{col} length: {train[f'{col}_len'].mean():.1f} chars (avg)")

# Class distribution by rule
print("\n" + "=" * 80)
print("CLASS DISTRIBUTION BY RULE")
print("=" * 80)
for rule in train['rule'].unique():
    rule_data = train[train['rule'] == rule]
    violations = rule_data['rule_violation'].sum()
    total = len(rule_data)
    print(f"\nRule: {rule[:60]}...")
    print(f"  Violations: {violations}/{total} ({100*violations/total:.1f}%)")

# Check for duplicates
print("\n" + "=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)
print(f"Duplicate comments in train: {train['body'].duplicated().sum()}")

# Sample a few rows to understand the pattern
print("\n" + "=" * 80)
print("DETAILED SAMPLE (Rule 1 - Advertising)")
print("=" * 80)
sample_ad = train[train['rule'].str.contains('Advertising')].iloc[0]
print(f"Comment: {sample_ad['body'][:200]}")
print(f"Violation: {sample_ad['rule_violation']}")
print(f"\nPositive example 1: {sample_ad['positive_example_1'][:150]}")
print(f"Negative example 1: {sample_ad['negative_example_1'][:150]}")

print("\n" + "=" * 80)
print("DETAILED SAMPLE (Rule 2 - Legal Advice)")
print("=" * 80)
sample_legal = train[train['rule'].str.contains('legal')].iloc[0]
print(f"Comment: {sample_legal['body'][:200]}")
print(f"Violation: {sample_legal['rule_violation']}")
print(f"\nPositive example 1: {sample_legal['positive_example_1'][:150]}")
print(f"Negative example 1: {sample_legal['negative_example_1'][:150]}")


# Let's design our strategy
print("=" * 80)
print("COMPETITION STRATEGY")
print("=" * 80)

strategy = """
APPROACH: Few-Shot Learning with Transformers

KEY CHALLENGE: Generalize to unseen rules in the hidden test set

SOLUTION COMPONENTS:

1. INPUT FORMAT (Prompt Engineering):
   - Concatenate: [Rule] + [Positive Examples] + [Negative Examples] + [SEP] + [Comment]
   - This teaches the model to learn from examples (few-shot learning)
   
2. MODEL SELECTION (Ensemble Multiple Architectures):
   - DeBERTa-v3-base: Strong understanding, good for few-shot
   - RoBERTa-base: Robust baseline
   - BERT-base: Additional diversity
   - Each trained with different seeds for variance
   
3. TRAINING STRATEGY:
   - Stratified K-Fold (5-fold) to handle class imbalance by rule
   - Handle duplicates: Keep them in same fold
   - Use focal loss to handle any class imbalance
   - Low learning rate (2e-5) to avoid overfitting small dataset
   
4. AUGMENTATION:
   - Shuffle order of positive/negative examples
   - Create variations of input format
   
5. ENSEMBLE:
   - Weighted average of multiple models
   - Weight optimization on validation set
   
6. TARGET: Public LB > 0.9888
   - With only 2029 samples, need robust CV strategy
   - Focus on generalization, not overfitting to these 2 rules
"""

print(strategy)

print("\n" + "=" * 80)
print("ESTIMATED TIMELINE")
print("=" * 80)
print("""
Step 1: Data preparation & input formatting (15 min)
Step 2: Single model baseline - DeBERTa (1-2 hours training)
Step 3: Add RoBERTa model (1-2 hours training)
Step 4: Ensemble & validation (30 min)
Step 5: Create submission (5 min)

Total GPU time: ~4-5 hours (well within 12-hour limit)
""")

print("=" * 80)
print("NEXT STEP: Create input formatting function")
print("=" * 80)


import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed(42)

# Create input text with few-shot prompt
def create_input_text(row, shuffle_examples=False):
    """
    Format: Rule description + Examples + Comment to classify
    This teaches the model to learn from examples (few-shot learning)
    """
    rule = row['rule']
    
    # Get examples
    pos_examples = [row['positive_example_1'], row['positive_example_2']]
    neg_examples = [row['negative_example_1'], row['negative_example_2']]
    
    # Optional: shuffle to create augmentation variations
    if shuffle_examples:
        random.shuffle(pos_examples)
        random.shuffle(neg_examples)
    
    # Build prompt
    input_text = f"Rule: {rule}\n\n"
    input_text += "Examples that violate the rule:\n"
    input_text += f"- {pos_examples[0]}\n"
    input_text += f"- {pos_examples[1]}\n\n"
    input_text += "Examples that do NOT violate the rule:\n"
    input_text += f"- {neg_examples[0]}\n"
    input_text += f"- {neg_examples[1]}\n\n"
    input_text += f"Comment to classify: {row['body']}"
    
    return input_text

# Test the function
print("=" * 80)
print("EXAMPLE INPUT FORMAT")
print("=" * 80)
sample_row = train.iloc[0]
formatted_input = create_input_text(sample_row)
print(formatted_input)
print(f"\nLength: {len(formatted_input)} characters")
print(f"Target: {sample_row['rule_violation']}")

print("\n" + "=" * 80)
print("CREATING FORMATTED DATASETS")
print("=" * 80)

# Create formatted inputs for train and test
train['input_text'] = train.apply(create_input_text, axis=1)
test['input_text'] = test.apply(create_input_text, axis=1)

print(f"Train samples with formatted text: {len(train)}")
print(f"Test samples with formatted text: {len(test)}")

# Check input length distribution
train['input_length'] = train['input_text'].str.len()
print(f"\nInput text length statistics:")
print(train['input_length'].describe())
print(f"Max input length: {train['input_length'].max()}")

# Estimate tokens (rough: ~4 chars per token)
max_tokens_estimate = train['input_length'].max() / 4
print(f"Estimated max tokens needed: {max_tokens_estimate:.0f}")
print("Recommended max_length for tokenizer: 512 (sufficient)")


# First, let's check what models might already be available
import os

print("=" * 80)
print("CHECKING AVAILABLE MODEL PATHS")
print("=" * 80)

# List all input directories
for dirname, _, filenames in os.walk('/kaggle/input'):
    if 'config.json' in filenames or 'pytorch_model.bin' in filenames:
        print(f"Found model at: {dirname}")

# Common Kaggle model dataset paths
potential_paths = [
    '/kaggle/input/deberta-v3-base',
    '/kaggle/input/deberta-v3-base-hf',
    '/kaggle/input/microsoft-deberta-v3-base',
    '/kaggle/input/roberta-base',
    '/kaggle/input/bert-base-uncased',
]

print("\nChecking potential model paths:")
for path in potential_paths:
    exists = os.path.exists(path)
    print(f"{path}: {'✓ EXISTS' if exists else '✗ Not found'}")


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score
import torch.nn as nn

print("=" * 80)
print("SETTING UP MODEL CONFIGURATION")
print("=" * 80)

# Model configuration - using LOCAL paths
MODEL_NAME = '/kaggle/input/debertav3base'  # Local DeBERTa model
MAX_LENGTH = 512
BATCH_SIZE = 8  # Conservative for P100
LEARNING_RATE = 2e-5
EPOCHS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
N_FOLDS = 5

print(f"Model path: {MODEL_NAME}")
print(f"Max length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Folds: {N_FOLDS}")

# Load tokenizer from local path
print("\nLoading tokenizer from local path...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded successfully!")
print(f"Vocab size: {tokenizer.vocab_size}")

# Create PyTorch Dataset
class RuleViolationDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# Test the dataset
print("\n" + "=" * 80)
print("TESTING DATASET")
print("=" * 80)

sample_texts = train['input_text'].values[:5]
sample_labels = train['rule_violation'].values[:5]

test_dataset = RuleViolationDataset(
    texts=sample_texts,
    labels=sample_labels,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

print(f"Dataset created with {len(test_dataset)} samples")
sample_item = test_dataset[0]
print(f"\nSample item keys: {sample_item.keys()}")
print(f"Input IDs shape: {sample_item['input_ids'].shape}")
print(f"Attention mask shape: {sample_item['attention_mask'].shape}")
print(f"Label: {sample_item['labels']}")

# Check actual token length (without padding)
actual_tokens = (sample_item['attention_mask'] == 1).sum().item()
print(f"Actual tokens used (non-padded): {actual_tokens}")

print("\n" + "=" * 80)
print("SETUP COMPLETE - Ready for training")
print("=" * 80)


from sklearn.model_selection import StratifiedKFold
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
import gc

print("=" * 80)
print("PREPARING CROSS-VALIDATION TRAINING")
print("=" * 80)

# Prepare data
X = train['input_text'].values
y = train['rule_violation'].values

# Create stratified folds
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Storage for OOF predictions and test predictions
oof_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))

print(f"Training {N_FOLDS}-fold cross-validation")
print(f"Total training samples: {len(X)}")
print(f"Total test samples: {len(test)}")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

# Training loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'='*80}")
    
    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    print(f"Train size: {len(X_train_fold)}, Val size: {len(X_val_fold)}")
    print(f"Train class distribution: {np.bincount(y_train_fold)}")
    print(f"Val class distribution: {np.bincount(y_val_fold)}")
    
    # Create datasets
    train_dataset = RuleViolationDataset(
        texts=X_train_fold,
        labels=y_train_fold,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    val_dataset = RuleViolationDataset(
        texts=X_val_fold,
        labels=y_val_fold,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Load model
    print(f"\nLoading model from {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_auc = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Training")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
                # Get probabilities
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"New best AUC: {best_val_auc:.4f}")
    
    # Final validation predictions (OOF)
    model.eval()
    fold_val_preds = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            fold_val_preds.extend(probs.cpu().numpy())
    
    oof_predictions[val_idx] = fold_val_preds
    
    # Test predictions
    test_dataset_fold = RuleViolationDataset(
        texts=test['input_text'].values,
        labels=None,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset_fold,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    fold_test_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test prediction"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            fold_test_preds.extend(probs.cpu().numpy())
    
    test_predictions += np.array(fold_test_preds) / N_FOLDS
    
    print(f"\nFold {fold + 1} Best Val AUC: {best_val_auc:.4f}")
    
    # Clear memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Overall CV score
overall_auc = roc_auc_score(y, oof_predictions)
print(f"\n{'='*80}")
print(f"OVERALL CV AUC: {overall_auc:.4f}")
print(f"{'='*80}")


print("=" * 80)
print("CREATING SUBMISSION FILE")
print("=" * 80)

# Create submission dataframe
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': test_predictions
})

print(f"\nSubmission shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))

print(f"\nPrediction statistics:")
print(submission['rule_violation'].describe())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n✓ Submission file 'submission.csv' created successfully!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Cross-validation AUC: {overall_auc:.4f}")
print(f"Target public LB: > 0.9888")
print(f"Gap to target: {0.9888 - overall_auc:.4f}")
print("\nThis baseline is below target. After submission, we should:")
print("1. Try different prompt formats")
print("2. Add RoBERTa ensemble")
print("3. Adjust training hyperparameters")
print("4. Consider using more training data augmentation")


import pandas as pd
import numpy as np
import random
import torch

# Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed(42)

# Load data
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

print("Data loaded successfully")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# SIMPLIFIED INPUT FORMAT - Remove confusing examples
def create_simple_input(row):
    """
    Much simpler: Just rule + comment
    No examples that might confuse the model
    """
    rule = row['rule']
    comment = row['body']
    
    # Direct, clear format
    input_text = f"Rule: {rule}\n\nComment: {comment}\n\nDoes this comment violate the rule?"
    
    return input_text

print("\n" + "=" * 80)
print("SIMPLIFIED INPUT FORMAT")
print("=" * 80)

# Create old format for comparison
def create_complex_input(row):
    rule = row['rule']
    pos_ex1 = row['positive_example_1']
    pos_ex2 = row['positive_example_2']
    neg_ex1 = row['negative_example_1']
    neg_ex2 = row['negative_example_2']
    comment = row['body']
    
    return f"Rule: {rule}\n\nExamples that violate the rule:\n- {pos_ex1}\n- {pos_ex2}\n\nExamples that do NOT violate the rule:\n- {neg_ex1}\n- {neg_ex2}\n\nComment to classify: {comment}"

# Compare formats
sample_row = train.iloc[0]
complex = create_complex_input(sample_row)
simple = create_simple_input(sample_row)

print("OLD COMPLEX FORMAT:")
print(complex[:300] + "...")
print(f"Length: {len(complex)} chars")

print("\n" + "-" * 80)
print("\nNEW SIMPLE FORMAT:")
print(simple)
print(f"Length: {len(simple)} chars")

print(f"\nReduction: {len(complex) - len(simple)} chars ({100*(len(complex)-len(simple))/len(complex):.1f}% shorter)")

# Create simplified inputs
train['simple_input'] = train.apply(create_simple_input, axis=1)
test['simple_input'] = test.apply(create_simple_input, axis=1)

print(f"\n✓ Created simplified inputs for {len(train)} train and {len(test)} test samples")
print(f"Average train input length: {train['simple_input'].str.len().mean():.0f} chars")
print(f"Max train input length: {train['simple_input'].str.len().max():.0f} chars")


from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import gc

# Configuration
MODEL_NAME = '/kaggle/input/debertav3base'
MAX_LENGTH = 256  # Reduced since inputs are shorter
BATCH_SIZE = 16  # Increased since inputs are shorter
LEARNING_RATE = 2e-5
EPOCHS = 5  # Increased from 4
N_FOLDS = 5

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Max length: {MAX_LENGTH} (reduced from 512)")
print(f"Batch size: {BATCH_SIZE} (increased from 8)")
print(f"Epochs: {EPOCHS} (increased from 4)")
print(f"Learning rate: {LEARNING_RATE}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"\nTokenizer loaded. Vocab size: {tokenizer.vocab_size}")

# Dataset class
class SimpleRuleDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# Prepare data
X = train['simple_input'].values
y = train['rule_violation'].values

# Cross-validation setup
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

print("\n" + "=" * 80)
print("STARTING TRAINING WITH SIMPLIFIED INPUT")
print("=" * 80)

# Training loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'='*80}")
    
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    print(f"Train: {len(X_train_fold)}, Val: {len(X_val_fold)}")
    
    # Create datasets
    train_dataset = SimpleRuleDataset(X_train_fold, y_train_fold, tokenizer, MAX_LENGTH)
    val_dataset = SimpleRuleDataset(X_val_fold, y_val_fold, tokenizer, MAX_LENGTH)
    
    # DataLoaders with no workers to avoid fork warnings
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    best_val_auc = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Val"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss/len(val_loader):.4f}, Val AUC={val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
    
    # Final predictions
    model.eval()
    fold_val_preds = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            fold_val_preds.extend(probs.cpu().numpy())
    
    oof_predictions[val_idx] = fold_val_preds
    
    # Test predictions
    test_dataset = SimpleRuleDataset(test['simple_input'].values, None, tokenizer, MAX_LENGTH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            fold_test_preds.extend(probs.cpu().numpy())
    
    test_predictions += np.array(fold_test_preds) / N_FOLDS
    
    print(f"\nFold {fold + 1} Best Val AUC: {best_val_auc:.4f}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Overall results
overall_auc = roc_auc_score(y, oof_predictions)
print(f"\n{'='*80}")
print(f"OVERALL CV AUC: {overall_auc:.4f}")
print(f"Previous complex format CV: 0.7720")
print(f"Improvement: {overall_auc - 0.7720:+.4f}")
print(f"{'='*80}")

# Create submission
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("\n✓ submission.csv created with simplified input format")


# Quick analysis of current predictions
print("=" * 80)
print("ANALYZING CURRENT PREDICTIONS")
print("=" * 80)

# Check OOF prediction distribution
print(f"\nOOF predictions range: {oof_predictions.min():.4f} to {oof_predictions.max():.4f}")
print(f"OOF predictions mean: {oof_predictions.mean():.4f}")
print(f"Actual labels mean: {y.mean():.4f}")

# Test predictions
print(f"\nTest predictions:")
for i, (row_id, pred) in enumerate(zip(test['row_id'], test_predictions)):
    print(f"  {row_id}: {pred:.4f}")

print(f"\nTest predictions range: {test_predictions.min():.4f} to {test_predictions.max():.4f}")
print(f"Test predictions mean: {test_predictions.mean():.4f}")


import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

print("=" * 80)
print("CRITICAL ANALYSIS - Current Situation")
print("=" * 80)

print("\nOur training rules:")
for i, rule in enumerate(train['rule'].unique(), 1):
    count = (train['rule'] == rule).sum()
    violations = train[train['rule'] == rule]['rule_violation'].sum()
    print(f"\n{i}. {rule}")
    print(f"   Samples: {count}, Violations: {violations} ({100*violations/count:.1f}%)")

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Submission 1 (Complex prompt):  CV=0.7720, Public LB=0.582")
print(f"Submission 2 (Simple prompt):   CV=0.8599, Public LB=0.609")
print(f"Improvement:                    CV=+0.0879, LB=+0.027")
print(f"\nTarget:                         Public LB > 0.9888")
print(f"Current gap to target:          0.3798 (38 percentage points)")

print("\n" + "=" * 80)
print("THE CORE PROBLEM")
print("=" * 80)
print("""
Train-test mismatch:
- Training: Only 2 rules (advertising, legal advice) 
- Test (public): Same 2 rules (performing at 0.609)
- Test (private): Unknown number of UNSEEN rules

The model overfits to the 2 training rules. When it sees new rules,
performance collapses. CV is misleadingly high because validation uses
the same 2 rules as training.

To reach 0.9888, we need the model to learn "how to follow ANY rule"
not just memorize these 2 specific rules.
""")

print("=" * 80)
print("NEXT STEPS - TEAM DECISION REQUIRED")
print("=" * 80)
print("""
Path 1: Quick iterations (Estimated max: 0.75 LB)
- Ensemble with RoBERTa  
- More data augmentation
- Different hyperparameters
Effort: 2-3 more submissions, ~1 day

Path 2: Fundamental redesign (Estimated max: 0.85+ LB)
- Use external toxicity datasets (allowed per rules)
- Pre-train on diverse moderation tasks
- Meta-learning approaches
Effort: Significant, 3-5 days

Path 3: Reality check
- Gap of 0.38 is extremely large
- Leaders may have techniques we don't know
- Consider if 0.9888 is achievable with current resources
""")


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import gc

# Load data
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

# Simple input format (we know this works better)
def create_simple_input(row):
    return f"Rule: {row['rule']}\n\nComment: {row['body']}\n\nDoes this comment violate the rule?"

train['input_text'] = train.apply(create_simple_input, axis=1)
test['input_text'] = test.apply(create_simple_input, axis=1)

# Configuration for RoBERTa
MODEL_NAME = '/kaggle/input/roberta-base'
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
N_FOLDS = 5

print("=" * 80)
print("TRAINING ROBERTA ENSEMBLE")
print("=" * 80)
print("This will be averaged with DeBERTa predictions")

# Dataset class
class SimpleRuleDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

X = train['input_text'].values
y = train['rule_violation'].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
roberta_oof = np.zeros(len(train))
roberta_test = np.zeros(len(test))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'='*80}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_dataset = SimpleRuleDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = SimpleRuleDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    
    best_auc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}: Val AUC={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
    
    # OOF predictions
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            fold_preds.extend(probs.cpu().numpy())
    
    roberta_oof[val_idx] = fold_preds
    
    # Test predictions
    test_dataset = SimpleRuleDataset(test['input_text'].values, None, tokenizer, MAX_LENGTH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    fold_test = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            fold_test.extend(probs.cpu().numpy())
    
    roberta_test += np.array(fold_test) / N_FOLDS
    print(f"Fold {fold+1} Best AUC: {best_auc:.4f}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

roberta_cv = roc_auc_score(y, roberta_oof)
print(f"\n{'='*80}")
print(f"RoBERTa CV AUC: {roberta_cv:.4f}")
print(f"{'='*80}")

# ENSEMBLE: Average DeBERTa and RoBERTa predictions
# Load previous DeBERTa predictions from the earlier run
# For now, just use RoBERTa
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': roberta_test
})

submission.to_csv('submission.csv', index=False)
print("\n✓ RoBERTa submission created")
print(f"Expected improvement: +0.02 to +0.05 on public LB")

