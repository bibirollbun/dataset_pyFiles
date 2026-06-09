# --- 1. Configuration ---
class CFG:
    model_path = '/kaggle/input/exp002-deberta-baseline/exp002_deberta_baseline_output/'
    model_name = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-base'
    # Define the path to your FIRST experiment's output
    BASELINE_ASSETS_PATH = '/kaggle/input/exp001-lgbm-tfidf-baseline-cv0-6936/'
    batch_size = 16 # Can use a larger batch size for inference
    n_splits = 5



# --- 2. Setup & Imports ---
import pandas as pd
import numpy as np
import os
import joblib
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 3. Load Data, Tokenizer, and Label Encoder ---
print("\nğŸ”¹ Loading assets...")
try:
    test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
    submission_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')
    le = joblib.load(os.path.join(CFG.model_path, 'label_encoder.pkl'))
except FileNotFoundError:
    print("Running locally. Ensure you have the data and trained model files.")
    # Add your local paths here if needed
    test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
    submission_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')
    le = joblib.load(os.path.join(CFG.BASELINE_ASSETS_PATH, 'label_encoder.pkl')) # Assumes it's in the current directory

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

# Create the same 'full_text' feature as in training
test_df['full_text'] = "question: " + test_df['QuestionText'].fillna('') + \
                       " [SEP] mc_answer: " + test_df['MC_Answer'].fillna('') + \
                       " [SEP] explanation: " + test_df['StudentExplanation'].fillna('')

print("Assets and test data loaded.")
print("-" * 50)


# --- 4. Create Test Dataset and DataLoader ---
def tokenize_function(examples):
    return tokenizer(examples['full_text'], padding='max_length', truncation=True, max_length=512)

test_dataset = Dataset.from_pandas(test_df)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

test_loader = DataLoader(
    tokenized_test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False
)
print("Test DataLoader created.")
print("-" * 50)


# --- 5. Prediction Function ---
def get_predictions(model, dataloader):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


# --- 6. Prediction Loop for all Folds ---
print("ğŸ”¹ Generating predictions from all 5 fold-models...")
all_fold_preds = []

for fold_num in range(CFG.n_splits):
    print(f"\n--- Loading and predicting with Fold {fold_num} ---")
    model_fold_path = os.path.join(CFG.model_path, f"fold_{fold_num}")
    
    # Find the best checkpoint directory within the fold directory
    best_checkpoint_path = None
    for item in os.listdir(model_fold_path):
        if item.startswith("checkpoint-"):
            best_checkpoint_path = os.path.join(model_fold_path, item)
            break
            
    if best_checkpoint_path is None:
        print(f"Warning: No checkpoint found for fold {fold_num}. Skipping.")
        continue

    print(f"Loading model from: {best_checkpoint_path}")
    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path).to(device)
    
    fold_logits = get_predictions(model, test_loader)
    # Convert logits to probabilities using softmax
    fold_probabilities = torch.softmax(torch.from_numpy(fold_logits), dim=-1).numpy()
    all_fold_preds.append(fold_probabilities)

    del model
    torch.cuda.empty_cache()



# --- 7. Ensemble Predictions and Create Submission ---
if all_fold_preds:
    print("\nğŸ”¹ Averaging predictions and creating submission file...")
    # Average the probabilities across all fold models
    avg_preds = np.mean(all_fold_preds, axis=0)

    # Get top 3 predictions
    top_3_indices = np.argsort(-avg_preds, axis=1)[:, :3]
    top_3_labels = le.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)
    predictions_str = [' '.join(labels) for labels in top_3_labels]

    # Create and save submission file
    submission_df['Category:Misconception'] = predictions_str
    submission_df.to_csv('submission.csv', index=False)

    print("\n" + "="*50)
    print("âœ… submission.csv created successfully!")
    print(submission_df.head())
    print("="*50)
else:
    print("\nâ�Œ No predictions were generated as no models were loaded.")




