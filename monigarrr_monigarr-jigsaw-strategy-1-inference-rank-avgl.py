# --- Imports and Setup ---
import os
import gc
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from scipy.stats import rankdata

# --- Configuration ---
class CFG:
    # Paths to the input datasets on Kaggle
    COMPETITION_DATA_DIR = "/kaggle/input/jigsaw-agile-community-rules/"
    #MODELS_INPUT_DIR = "/kaggle/input/jigsaw-monigarr-strategy-1-ft-csvfiles/"
    MODELS_INPUT_DIR = "/kaggle/input/monigarr-jigsaw-competition-models-aug-9-2025/final_models/"

    # Use the final, best OOF AUC scores for each model for weighting
    MODELS = {
        "deberta-v3-base": 0.83356,
        "roberta-large": 0.80837,
        "electra-large-discriminator": 0.80272
        #"deberta-v2-xlarge": 0.81955,
        #"deberta-v3-large": 0.82763,
        #"xlm-roberta-large": 0.70696
    }

    MODELS_TO_RUN = ["deberta-v3-base","roberta-large","electra-large-discriminator"]
    
    MAX_LEN = 256
    VALID_BATCH_SIZE = 32


def softmax(x):
    """Computes softmax probabilities from logits."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class JigsawDataset(Dataset):
    """Custom PyTorch Dataset for inference."""
    def __init__(self, tokenized_data):
        self.input_ids = torch.tensor(tokenized_data['input_ids'])
        self.attention_mask = torch.tensor(tokenized_data['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }

# --- Load and Prepare Test Data ---
test_df = pd.read_csv(f"{CFG.COMPETITION_DATA_DIR}test.csv")
test_df['full_text'] = test_df['rule'] + "[SEP]" + test_df['body']

print("Test data prepared successfully.")


# --- Run Inference ---
all_model_preds = {}

# Outer loop for each model architecture
#for model_name in CFG.MODELS.keys():
for model_name in CFG.MODELS_TO_RUN:

    print(f"\n--- Inferencing with {model_name} ---")

    # --- 1. Load the correct tokenizer for THIS model ---
    # We can load it from any fold, e.g., fold0
    tokenizer_path = f"{CFG.MODELS_INPUT_DIR}{model_name}-finetuned/fold0/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # --- 2. Tokenize the test data using THIS tokenizer ---
    test_df['full_text'] = test_df['rule'] + tokenizer.sep_token + test_df['body']
    test_tokenized = tokenizer(
        test_df['full_text'].tolist(),
        max_length=CFG.MAX_LEN,
        padding='max_length',
        truncation=True
    )
    test_dataset = JigsawDataset(test_tokenized)

    
    model_fold_preds = []
    
    # Inner loop for each of the 5 folds
    for fold in range(5):
        print(f"  > Processing Fold {fold}...")
        
        # Define path and load the specific fold's model and tokenizer
        model_path = f"{CFG.MODELS_INPUT_DIR}{model_name}-finetuned/fold{fold}"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create a simple Trainer for fast prediction
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./temp_output",
                per_device_eval_batch_size=CFG.VALID_BATCH_SIZE,
                fp16=True, # Use fp16 for faster inference
                report_to="none",
            ),
        )
        
        # Predict, get probabilities, and append to list
        test_output = trainer.predict(test_dataset)
        test_probs = softmax(test_output.predictions)
        model_fold_preds.append(test_probs[:, 1])
        
        # Clean up memory before loading the next model
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Average the predictions from all 5 folds for this model
    all_model_preds[model_name] = np.mean(model_fold_preds, axis=0)

print("\nInference complete for all models.")


# --- Ensemble Predictions and Create Submission File ---

# Create a DataFrame from the generated predictions
ensemble_df = pd.DataFrame(all_model_preds)
ensemble_df['row_id'] = test_df['row_id']

# --- Rank Average Ensemble ---
# Use the same list of models that you ran inference on
for model_name in CFG.MODELS_TO_RUN:
    ensemble_df[f'rank_{model_name}'] = rankdata(ensemble_df[model_name])

rank_cols = [f'rank_{model_name}' for model_name in CFG.MODELS_TO_RUN]
ensemble_df['rank_avg'] = ensemble_df[rank_cols].mean(axis=1)

# Normalize the final ranks to be between 0 and 1 for submission
ensemble_df['rank_avg_norm'] = (ensemble_df['rank_avg'] - ensemble_df['rank_avg'].min()) / (ensemble_df['rank_avg'].max() - ensemble_df['rank_avg'].min())

# Create the submission DataFrame
submission_df = ensemble_df[['row_id', 'rank_avg_norm']].copy()
submission_df.rename(columns={'rank_avg_norm': 'rule_violation'}, inplace=True)

# --- Final Sanity Check ---
assert not submission_df.isnull().values.any(), "Submission contains NaN values!"
assert submission_df.shape[0] == len(test_df), "Submission has wrong number of rows!"
print("Final predictions validated.")

# --- Save the final submission.csv file ---
submission_df.to_csv('submission.csv', index=False)

print("\nRank Average submission.csv file created successfully!")
submission_df.head()

