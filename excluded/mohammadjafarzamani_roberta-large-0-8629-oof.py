import pandas as pd
import numpy as np
from pathlib import Path

# Load the data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

# Basic information
print("=" * 50)
print("TRAINING DATA")
print("=" * 50)
print(f"Training samples: {len(train_df)}")
print(f"\nColumns: {train_df.columns.tolist()}")
print(f"\nData types:\n{train_df.dtypes}")
print(f"\nMissing values:\n{train_df.isnull().sum()}")
print(f"\nTarget distribution:\n{train_df['rule_violation'].value_counts()}")
print(f"\nUnique rules in train: {train_df['rule'].nunique()}")
print(f"Rules: {train_df['rule'].unique()}")
print(f"\nUnique subreddits: {train_df['subreddit'].nunique()}")

print("\n" + "=" * 50)
print("TEST DATA")
print("=" * 50)
print(f"Test samples: {len(test_df)}")
print(f"\nColumns: {test_df.columns.tolist()}")
print(f"\nUnique rules in test: {test_df['rule'].nunique()}")
print(f"Rules: {test_df['rule'].unique()}")

print("\n" + "=" * 50)
print("SAMPLE DATA")
print("=" * 50)
print("\nFirst 3 training examples:")
print(train_df.head(3).to_string())

print("\n" + "=" * 50)
print("KEY CHALLENGE")
print("=" * 50)
print("⚠️ The test set contains UNSEEN RULES not in training data!")
print("This requires models that can generalize to new rules.")


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')

# Create stratified folds based on rule AND target
# This ensures each fold has balanced representation
train_df['stratify_col'] = train_df['rule'].astype(str) + '_' + train_df['rule_violation'].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['stratify_col'])):
    train_df.loc[val_idx, 'fold'] = fold

# Verify fold distribution
print("=" * 50)
print("FOLD DISTRIBUTION")
print("=" * 50)
for fold in range(5):
    fold_df = train_df[train_df['fold'] == fold]
    print(f"\nFold {fold}:")
    print(f"  Total samples: {len(fold_df)}")
    print(f"  Rule distribution:\n{fold_df['rule'].value_counts()}")
    print(f"  Target distribution:\n{fold_df['rule_violation'].value_counts()}")

# Save train_df with fold assignments
train_df.to_csv('train_with_folds.csv', index=False)
print("\n✓ Saved train_with_folds.csv")

# Display statistics
print("\n" + "=" * 50)
print("TRAINING STATISTICS")
print("=" * 50)
print(f"Average comment length: {train_df['body'].str.len().mean():.1f} chars")
print(f"Max comment length: {train_df['body'].str.len().max()} chars")
print(f"Min comment length: {train_df['body'].str.len().min()} chars")

print("\n" + "=" * 50)
print("EXAMPLE ANALYSIS")
print("=" * 50)
print(f"Positive example 1 avg length: {train_df['positive_example_1'].str.len().mean():.1f} chars")
print(f"Positive example 2 avg length: {train_df['positive_example_2'].str.len().mean():.1f} chars")
print(f"Negative example 1 avg length: {train_df['negative_example_1'].str.len().mean():.1f} chars")
print(f"Negative example 2 avg length: {train_df['negative_example_2'].str.len().mean():.1f} chars")

print("\n" + "=" * 50)
print("NEXT STEPS")
print("=" * 50)
print("1. We'll use 5-fold cross-validation")
print("2. Models must learn to generalize from rule descriptions + examples")
print("3. Key approach: Concatenate rule + examples + body as input")


import pandas as pd
import numpy as np

# Load data with folds
train_df = pd.read_csv('train_with_folds.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

def create_input_text(row, include_rule_name=True, include_examples=True):
    """
    Create rich input text for the model.
    The key insight: Models must understand the RULE from its description and examples.
    """
    parts = []
    
    if include_rule_name:
        # Add rule description
        parts.append(f"Rule: {row['rule']}")
    
    if include_examples:
        # Add positive examples (what VIOLATES the rule)
        parts.append(f"Examples of violations:")
        parts.append(f"1. {row['positive_example_1']}")
        parts.append(f"2. {row['positive_example_2']}")
        
        # Add negative examples (what DOES NOT violate the rule)
        parts.append(f"Examples of non-violations:")
        parts.append(f"1. {row['negative_example_1']}")
        parts.append(f"2. {row['negative_example_2']}")
    
    # Add the actual comment to classify
    parts.append(f"Comment to classify: {row['body']}")
    
    return " [SEP] ".join(parts)

# Create different input variations for experimentation
print("Creating input variations...")

# Full context (rule + examples + body)
train_df['input_full'] = train_df.apply(create_input_text, axis=1)
test_df['input_full'] = test_df.apply(create_input_text, axis=1)

# Without examples (rule + body only)
train_df['input_no_examples'] = train_df.apply(
    lambda x: create_input_text(x, include_examples=False), axis=1
)
test_df['input_no_examples'] = test_df.apply(
    lambda x: create_input_text(x, include_examples=False), axis=1
)

# Body only (baseline)
train_df['input_body_only'] = train_df['body']
test_df['input_body_only'] = test_df['body']

# Check lengths
print("\n" + "=" * 50)
print("INPUT LENGTH STATISTICS")
print("=" * 50)
print(f"Full input (with examples):")
print(f"  Mean: {train_df['input_full'].str.len().mean():.1f} chars")
print(f"  Max: {train_df['input_full'].str.len().max()} chars")
print(f"  Min: {train_df['input_full'].str.len().min()} chars")

print(f"\nNo examples (rule + body):")
print(f"  Mean: {train_df['input_no_examples'].str.len().mean():.1f} chars")
print(f"  Max: {train_df['input_no_examples'].str.len().max()} chars")

print(f"\nBody only:")
print(f"  Mean: {train_df['input_body_only'].str.len().mean():.1f} chars")
print(f"  Max: {train_df['input_body_only'].str.len().max()} chars")

# Show example
print("\n" + "=" * 50)
print("SAMPLE INPUT (First training example)")
print("=" * 50)
print("\n--- FULL INPUT ---")
print(train_df['input_full'].iloc[0][:1000] + "..." if len(train_df['input_full'].iloc[0]) > 1000 else train_df['input_full'].iloc[0])

print("\n--- TARGET ---")
print(f"Rule violation: {train_df['rule_violation'].iloc[0]}")

# Save processed data
train_df.to_csv('train_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)
print("\n✓ Saved train_processed.csv and test_processed.csv")

print("\n" + "=" * 50)
print("STRATEGY SUMMARY")
print("=" * 50)
print("We'll experiment with 3 input strategies:")
print("1. FULL: Rule + Positive examples + Negative examples + Body")
print("2. NO_EXAMPLES: Rule + Body only")
print("3. BODY_ONLY: Body only (baseline)")
print("\nThe FULL approach should help models generalize to unseen rules.")


# Check available packages and install if needed
import sys
import subprocess

print("=" * 50)
print("CHECKING ENVIRONMENT")
print("=" * 50)

# Check CUDA availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Check transformers
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except:
    print("Installing transformers...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])
    import transformers
    print(f"Transformers version: {transformers.__version__}")

# Check other required packages
packages_to_check = {
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'tqdm': 'tqdm'
}

for package, install_name in packages_to_check.items():
    try:
        __import__(package)
        print(f"✓ {package} available")
    except:
        print(f"Installing {install_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", install_name])
        print(f"✓ {package} installed")

print("\n" + "=" * 50)
print("ENVIRONMENT READY")
print("=" * 50)
print("\nNext: We'll build a DeBERTa-v3 model")
print("DeBERTa-v3 has shown excellent performance on similar tasks")


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import gc

# Configuration
class CFG:
    model_name = 'microsoft/deberta-v3-base'
    max_length = 512
    batch_size = 8
    learning_rate = 2e-5
    weight_decay = 0.01
    epochs = 3
    num_folds = 5
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CFG.seed)

# Custom Dataset
class RuleViolationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, input_col='input_full'):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_col = input_col
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row[self.input_col]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        if 'rule_violation' in row:
            item['labels'] = torch.tensor(row['rule_violation'], dtype=torch.float)
            
        return item

# Model with mean pooling
class DeBERTaClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
        sum_mask = attention_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        x = self.dropout(mean_embeddings)
        logits = self.fc(x)
        return logits

print("=" * 50)
print("MODEL CONFIGURATION")
print("=" * 50)
print(f"Model: {CFG.model_name}")
print(f"Max length: {CFG.max_length}")
print(f"Batch size: {CFG.batch_size}")
print(f"Learning rate: {CFG.learning_rate}")
print(f"Epochs: {CFG.epochs}")
print(f"Device: {CFG.device}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/tokenizer23/deberta_tokenizer')
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

# Load data
print("\nLoading processed data...")
train_df = pd.read_csv('train_processed.csv')
print(f"Training data loaded: {len(train_df)} samples")

# Test tokenization
print("\n" + "=" * 50)
print("TOKENIZATION TEST")
print("=" * 50)
sample_text = train_df['input_full'].iloc[0]
sample_encoding = tokenizer(
    sample_text,
    truncation=True,
    max_length=CFG.max_length,
    padding='max_length',
    return_tensors='pt'
)
print(f"Sample text length: {len(sample_text)} chars")
print(f"Tokenized length: {sample_encoding['input_ids'].shape[1]} tokens")
print(f"Number of actual tokens (non-padding): {sample_encoding['attention_mask'].sum().item()}")

print("\n" + "=" * 50)
print("READY FOR TRAINING")
print("=" * 50)
print("Next: We'll train the model with 5-fold cross-validation")


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW  # Changed: Import from torch instead of transformers
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask).squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    predictions = []
    labels_list = []
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask).squeeze(-1)
            loss = nn.BCEWithLogitsLoss()(logits, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(preds)
            labels_list.extend(labels.cpu().numpy())
    
    auc = roc_auc_score(labels_list, predictions)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, auc, predictions

def train_fold(fold, train_df, tokenizer, CFG):
    print(f"\n{'='*50}")
    print(f"TRAINING FOLD {fold}")
    print(f"{'='*50}")
    
    # Split data
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    valid_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    print(f"Train size: {len(train_data)}, Valid size: {len(valid_data)}")
    
    # Create datasets
    train_dataset = RuleViolationDataset(train_data, tokenizer, CFG.max_length, 'input_full')
    valid_dataset = RuleViolationDataset(valid_data, tokenizer, CFG.max_length, 'input_full')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = DeBERTaClassifier(CFG.model_name)
    model.to(CFG.device)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )
    
    num_training_steps = len(train_loader) * CFG.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_auc = 0
    oof_predictions = np.zeros(len(valid_data))
    
    # Training loop
    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch + 1}/{CFG.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, CFG.device)
        valid_loss, valid_auc, valid_preds = validate(model, valid_loader, CFG.device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid AUC: {valid_auc:.4f}")
        
        if valid_auc > best_auc:
            best_auc = valid_auc
            oof_predictions = np.array(valid_preds)
            torch.save(model.state_dict(), f'deberta_fold{fold}_best.pth')
            print(f"Model saved! Best AUC: {best_auc:.4f}")
    
    # Cleanup
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return oof_predictions, valid_data.index.values, best_auc

# Main training
print("\n" + "="*50)
print("STARTING 5-FOLD CROSS-VALIDATION")
print("="*50)

train_df = pd.read_csv('train_processed.csv')
oof_df = pd.DataFrame()
fold_aucs = []

for fold in range(CFG.num_folds):
    oof_preds, oof_indices, fold_auc = train_fold(fold, train_df, tokenizer, CFG)
    
    fold_df = pd.DataFrame({
        'index': oof_indices,
        'fold': fold,
        'oof_pred': oof_preds,
        'target': train_df.loc[oof_indices, 'rule_violation'].values
    })
    oof_df = pd.concat([oof_df, fold_df], axis=0)
    fold_aucs.append(fold_auc)
    
    print(f"\nFold {fold} Best AUC: {fold_auc:.4f}")

# Overall CV score
overall_auc = roc_auc_score(oof_df['target'], oof_df['oof_pred'])

print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)
for i, auc in enumerate(fold_aucs):
    print(f"Fold {i}: {auc:.4f}")
print(f"\nOverall CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# Save OOF predictions
oof_df.to_csv('oof_predictions_deberta.csv', index=False)
print("\n✓ Saved OOF predictions")


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import gc

# Load test data
test_df = pd.read_csv('test_processed.csv')
print(f"Test samples: {len(test_df)}")

def predict_fold(fold, test_df, tokenizer, CFG):
    """Generate predictions for a single fold"""
    print(f"\nPredicting with fold {fold} model...")
    
    # Create dataset
    test_dataset = RuleViolationDataset(test_df, tokenizer, CFG.max_length, 'input_full')
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=0,  # Changed to 0 to avoid multiprocessing warnings
        pin_memory=True
    )
    
    # Load model
    model = DeBERTaClassifier(CFG.model_name)
    model.load_state_dict(torch.load(f'deberta_fold{fold}_best.pth'))
    model.to(CFG.device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Fold {fold}'):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            
            logits = model(input_ids, attention_mask).squeeze(-1)
            preds = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(preds)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return np.array(predictions)

# Generate predictions from all folds
all_fold_preds = []

for fold in range(CFG.num_folds):
    fold_preds = predict_fold(fold, test_df, tokenizer, CFG)
    all_fold_preds.append(fold_preds)

# Average predictions across folds
final_predictions = np.mean(all_fold_preds, axis=0)

print("\n" + "="*50)
print("PREDICTION STATISTICS")
print("="*50)
print(f"Mean prediction: {final_predictions.mean():.4f}")
print(f"Std prediction: {final_predictions.std():.4f}")
print(f"Min prediction: {final_predictions.min():.4f}")
print(f"Max prediction: {final_predictions.max():.4f}")
print(f"\nPrediction distribution:")
print(f"  < 0.3: {(final_predictions < 0.3).sum()} samples")
print(f"  0.3-0.7: {((final_predictions >= 0.3) & (final_predictions <= 0.7)).sum()} samples")
print(f"  > 0.7: {(final_predictions > 0.7).sum()} samples")

# Create submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("\n✓ Submission saved as submission.csv")
print("\nFirst 10 predictions:")
print(submission.head(10))


# Verify your submission file is correct
import pandas as pd

sub = pd.read_csv('submission.csv')
print("Submission check:")
print(f"Shape: {sub.shape}")
print(f"Columns: {sub.columns.tolist()}")
print(f"Any nulls: {sub.isnull().sum().sum()}")
print(f"\n{sub}")

# Make sure it matches the sample submission format
sample = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')
print(f"\nSample submission shape: {sample.shape}")
print(f"Your submission shape: {sub.shape}")
print(f"Shapes match: {sample.shape == sub.shape}")


import os
import shutil

# List all your trained model files
print("Checking saved model files:")
model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
for f in model_files:
    size = os.path.getsize(f) / (1024*1024)
    print(f"  {f}: {size:.1f} MB")

if len(model_files) == 5:
    print(f"\n✓ All 5 fold models found")
else:
    print(f"\n⚠ Warning: Expected 5 models, found {len(model_files)}")

# Also check test_processed.csv
if os.path.exists('test_processed.csv'):
    print("✓ test_processed.csv found")
else:
    print("⚠ test_processed.csv missing - need to recreate")


import os
from transformers import AutoTokenizer, AutoConfig

# Create directory for tokenizer files
os.makedirs('deberta_tokenizer', exist_ok=True)

# Save tokenizer locally
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
tokenizer.save_pretrained('deberta_tokenizer')

# Save config
print("Saving config...")
config = AutoConfig.from_pretrained('microsoft/deberta-v3-base')
config.save_pretrained('deberta_config')

print("\nFiles saved:")
print("Tokenizer files:", os.listdir('deberta_tokenizer'))
print("Config files:", os.listdir('deberta_config'))

print("\n✓ Now save this notebook with 'Quick Save with Outputs'")
print("✓ Create a NEW dataset (or update existing one)")


# Reload basic libraries and data
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

print("="*80)
print("MORNING RESTART - CHECKING STATUS")
print("="*80)

# Check if there's a saved submission file from overnight
import os
if os.path.exists('submission.csv'):
    sub = pd.read_csv('submission.csv')
    print("✓ submission.csv exists")
    print(f"  Shape: {sub.shape}")
    print(f"  Predictions range: [{sub['rule_violation'].min():.4f}, {sub['rule_violation'].max():.4f}]")
    print(f"  Mean: {sub['rule_violation'].mean():.4f}")
else:
    print("✗ No submission.csv found")

print("\n" + "="*80)
print("REALITY CHECK")
print("="*80)
print("Your notebook kernel restarted (or you're in a new session).")
print("This means:")
print("  - All variables (train_df, models, etc.) were cleared")
print("  - The overnight training either:")
print("    a) Never started (you fell asleep before running it)")
print("    b) Completed and saved submission.csv")
print("    c) Crashed during training")
print()
print("The submission.csv file (if it exists) is your only saved output.")

print("\n" + "="*80)
print("FRESH START STRATEGY")
print("="*80)
print("Yesterday you achieved:")
print("  - RoBERTa-large: 0.8633 OOF (your best)")
print("  - DeBERTa-large: 0.7973 OOF")
print()
print("Today's plan (5 submissions available):")
print("  1. Re-train RoBERTa-large quickly (you know it works)")
print("  2. Submit RoBERTa results")
print("  3. Train 2-3 more diverse models")
print("  4. Create ensemble")
print("  5. Submit ensemble")
print()
print("OR: Just start fresh with what we learned yesterday")
print("    and train multiple models properly today.")


print("="*80)
print("SUBMISSION FILE ANALYSIS")
print("="*80)
print("Current submission.csv characteristics:")
print("  Range: [0.10, 0.82]")
print("  Mean: 0.46")
print()
print("This matches XLM-RoBERTa (your worst model)")
print("  → You should NOT submit this")
print()
print("Yesterday's best (RoBERTa) had:")
print("  Range: [0.02, 0.98]")
print("  Mean: ~0.40")
print("  OOF AUC: 0.8633")
print()
print("The overnight DeBERTa 5-fold training likely never ran")
print("(kernel restarted before it started)")

print("\n" + "="*80)
print("TODAY'S CONCRETE PLAN")
print("="*80)
print("You have 5 submissions and ~12 hours until midnight.")
print()
print("RECOMMENDED: Build a diverse model portfolio")
print()
print("Submission 1 (now): RoBERTa-large baseline")
print("  - Re-train what worked yesterday (0.8633 OOF)")
print("  - Time: ~2.5 hours")
print("  - This gives you a solid baseline")
print()
print("Submission 2 (afternoon): Different architecture")
print("  - Try ELECTRA-large or ALBERT-xxlarge")
print("  - Time: ~2.5 hours")
print("  - Adds diversity")
print()
print("Submission 3 (evening): Ensemble")
print("  - Combine submissions 1+2")
print("  - Time: 5 minutes")
print()
print("Submissions 4-5: Save for tomorrow")
print("  - Learn from public LB scores first")
print("  - Adjust strategy based on feedback")

print("\n" + "="*80)
print("IMMEDIATE ACTION")
print("="*80)
print("Start by re-training RoBERTa-large (your proven winner).")
print("This takes ~2.5 hours, so start it now.")
print()
print("Ready to begin?")


# Complete setup - run all at once
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

# Create input formatting
def create_model_input(row):
    prompt = f"""Rule: {row['rule']}

Examples that VIOLATE this rule:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

Examples that DO NOT violate this rule:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

Subreddit: {row['subreddit']}

Comment to classify: {row['body']}"""
    return prompt

train_df['input_text'] = train_df.apply(create_model_input, axis=1)
test_df['input_text'] = test_df.apply(create_model_input, axis=1)

# Config
class Config:
    model_name = 'roberta-large'
    max_length = 512
    n_folds = 3
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 1e-5
    weight_decay = 0.01
    num_epochs = 5
    warmup_ratio = 0.1
    use_fp16 = True
    seed = 42

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# Create folds
train_df['stratify_col'] = train_df['rule'].astype(str) + '_' + train_df['rule_violation'].astype(str)
train_df['fold'] = -1
skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df['stratify_col'])):
    train_df.loc[val_idx, 'fold'] = fold

# Dataset
class RulesDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, is_test=False):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.loc[idx, 'input_text']
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if not self.is_test:
            item['labels'] = torch.tensor(self.data.loc[idx, 'rule_violation'], dtype=torch.float)
        
        return item

# Model
class RulesClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.fc = nn.Linear(self.model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits_list = [self.fc(dropout(pooled)) for dropout in self.dropouts]
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)

# Training functions
def train_epoch(model, loader, optimizer, scheduler, device, scaler, grad_accum):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)
        
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(logits, labels) / grad_accum
        
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    preds, labels_list = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                logits = model(input_ids, attention_mask)
            
            preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            labels_list.extend(labels.cpu().numpy())
    
    return roc_auc_score(labels_list, preds), preds

def predict(model, loader, device):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast():
                logits = model(input_ids, attention_mask)
            
            preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    
    return np.array(preds)

print("="*80)
print("STARTING ROBERTA-LARGE TRAINING")
print("="*80)
print(f"Device: {device}")
print(f"Estimated time: 2-2.5 hours")
print("="*80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Train all folds
oof_preds = np.zeros(len(train_df))
test_preds_all = []

for fold in range(config.n_folds):
    print(f"\n{'='*80}\nFOLD {fold+1}/{config.n_folds}\n{'='*80}")
    
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    val_idx = train_df[train_df['fold'] == fold].index
    
    train_dataset = RulesDataset(train_data, tokenizer, config.max_length)
    val_dataset = RulesDataset(val_data, tokenizer, config.max_length)
    test_dataset = RulesDataset(test_df, tokenizer, config.max_length, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size*2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size*2, shuffle=False, num_workers=2)
    
    model = RulesClassifier(config.model_name).to(device)
    
    num_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * config.warmup_ratio), num_steps)
    scaler = GradScaler()
    
    best_auc = 0
    best_test_preds = None
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, config.gradient_accumulation_steps)
        val_auc, val_preds = validate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            oof_preds[val_idx] = val_preds
            best_test_preds = predict(model, test_loader, device)
    
    test_preds_all.append(best_test_preds)
    print(f"Fold {fold} Best AUC: {best_auc:.4f}")
    
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

# Final results
overall_auc = roc_auc_score(train_df['rule_violation'], oof_preds)
test_preds_final = np.mean(test_preds_all, axis=0)

print(f"\n{'='*80}\nFINAL RESULTS\n{'='*80}")
print(f"Overall OOF AUC: {overall_auc:.4f}")

# Save submission
submission = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': test_preds_final})
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved as submission.csv")
print(submission.head(10))

