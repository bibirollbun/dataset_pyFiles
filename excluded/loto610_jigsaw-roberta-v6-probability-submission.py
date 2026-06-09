# Import libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 512
BATCH_SIZE = 32
NUM_LABELS = 2
N_FOLDS = 5

# Paths (adjust based on attached datasets/models)
COMPETITION_DATA_PATH = '/kaggle/input/jigsaw-agile-community-rules'
MODEL_PATH = '/kaggle/input/jigsaw-roberta-models-v6'  # Updated to v6
ASSETS_PATH = '/kaggle/input/jigsaw-roberta-models-v6'  # Same as model path
ROBERTA_PATH = '/kaggle/input/jigsaw-roberta-models-v6/roberta-base-local'  # Local roberta-base path

print(f"Device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")
print(f"RoBERTa path: {ROBERTA_PATH}")
print(f"Competition data path: {COMPETITION_DATA_PATH}")


# Model definition (same as training)
class RobertaClassifier(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=2, dropout=0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


# Load data
test_df = pd.read_csv(f'{COMPETITION_DATA_PATH}/test.csv')
sample_submission = pd.read_csv(f'{COMPETITION_DATA_PATH}/sample_submission.csv')

print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")
print(f"Columns: {test_df.columns.tolist()}")


# Initialize tokenizer from local files
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)

# Create dataset and dataloader
test_dataset = TextDataset(
    texts=test_df['body'].tolist(),  # Updated column name for this competition
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Number of batches: {len(test_loader)}")


# Load models and make predictions
all_predictions = np.zeros((len(test_df), NUM_LABELS))

for fold in range(N_FOLDS):
    print(f"Loading fold {fold} model...")
    
    # Initialize model with local roberta-base
    model = RobertaClassifier(
        model_name=ROBERTA_PATH,
        num_labels=NUM_LABELS,
        dropout=0.1
    )
    
    # Load model weights
    model_file = f'{MODEL_PATH}/fold_{fold}_model.pt'
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # Make predictions
        fold_predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.softmax(outputs, dim=-1).cpu().numpy()
                fold_predictions.extend(predictions)
        
        all_predictions += np.array(fold_predictions)
        print(f"Fold {fold} completed")
    else:
        print(f"Warning: Model file not found: {model_file}")

# Average predictions across folds
all_predictions /= N_FOLDS
final_predictions = np.argmax(all_predictions, axis=1)

print(f"Prediction shape: {all_predictions.shape}")
print(f"Final predictions shape: {final_predictions.shape}")
print(f"Class distribution: {np.bincount(final_predictions)}")


# Create submission
submission = sample_submission.copy()
submission['rule_violation'] = final_predictions  # Updated target column name

# Save submission
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
print(f"Submission shape: {submission.shape}")
print("\nFirst 5 rows:")
print(submission.head())
print("\nSubmission target distribution:")
print(submission['rule_violation'].value_counts().sort_index())


# Optional: Save prediction probabilities for analysis
prob_df = test_df.copy()
prob_df['pred_prob_0'] = all_predictions[:, 0]
prob_df['pred_prob_1'] = all_predictions[:, 1]
prob_df['pred_label'] = final_predictions

# Only save a subset to avoid large files
prob_df.head(1000).to_csv('test_predictions_sample.csv', index=False)

print("Prediction probabilities saved: test_predictions_sample.csv")
print(f"Mean prediction probability for class 1: {all_predictions[:, 1].mean():.4f}")
print(f"Std prediction probability for class 1: {all_predictions[:, 1].std():.4f}")


# Verification
print("\n=== Final Verification ===")
print(f"Submission file exists: {os.path.exists('submission.csv')}")
print(f"Submission file size: {os.path.getsize('submission.csv')} bytes")

# Check submission format
final_check = pd.read_csv('submission.csv')
print(f"Final submission shape: {final_check.shape}")
print(f"Columns: {final_check.columns.tolist()}")
print(f"Missing values: {final_check.isnull().sum().sum()}")
print(f"Data types: {final_check.dtypes.tolist()}")

print("\n✅ Inference completed successfully!")

