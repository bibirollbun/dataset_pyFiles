# --- Imports and Setup ---
import os
import gc
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from scipy.stats import rankdata
import joblib

# --- Configuration ---
class CFG:
    # Paths to the input datasets on Kaggle
    COMPETITION_DATA_DIR = "/kaggle/input/jigsaw-agile-community-rules/"
    #MODELS_INPUT_DIR = "/kaggle/input/jigsaw-monigarr-strategy-1-ft-csvfiles/"
    MODELS_INPUT_DIR = "/kaggle/input/monigarr-jigsaw-competition-models-aug-9-2025/final_models/"

    # Use the final, best OOF AUC scores for each model for weighting
    MODELS = {
        "deberta-v3-base": 0.83356,
        "deberta-v2-xlarge": 0.81955,
        "deberta-v3-large": 0.82763,
        "roberta-large": 0.80837,
        "roberta-base-reddit-pretrained": 0.84000,
        "electra-large-discriminator": 0.80272,
        #"xlm-roberta-large": 0.70696
    }

    #MODELS_TO_RUN = ["deberta-v3-base","roberta-base-reddit-pretrained","deberta-v3-large"] //scored 0.635
    #final score with 3 fold 0.623 MODELS_TO_RUN = ["deberta-v3-base","roberta-large","electra-large-discriminator","roberta-base-reddit-pretrained","deberta-v2-xlarge","deberta-v3-large"] 
    MODELS_TO_RUN = ["deberta-v3-base","roberta-large","electra-large-discriminator","roberta-base-reddit-pretrained","deberta-v2-xlarge","deberta-v3-large"] 

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
for model_name in CFG.MODELS_TO_RUN:
    print(f"\n--- Inferencing with {model_name} ---")

    # --- THE CRITICAL FIX ---
    # Conditionally set the folder suffix based on the model name
    if "pretrained" in model_name:
        folder_name = model_name # Use the name as-is
    else:
        folder_name = f"{model_name}-finetuned" # Add the finetuned suffix
    
    # --- 1. Load the correct tokenizer for THIS model ---
    tokenizer_path = f"{CFG.MODELS_INPUT_DIR}{folder_name}/fold0/"
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
        
        # Use the new folder_name variable to build the correct path
        model_path = f"{CFG.MODELS_INPUT_DIR}{folder_name}/fold{fold}"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create a simple Trainer for fast prediction
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./temp_output",
                per_device_eval_batch_size=CFG.VALID_BATCH_SIZE,
                fp16=True,
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

# Create a DataFrame from the base models' predictions. This is now our feature set.
ensemble_df = pd.DataFrame(all_model_preds)
ensemble_df['row_id'] = test_df['row_id']

# --- Load the 5 trained LightGBM models ---
lgbm_models = []
for fold in range(5):
    model_path = f"/kaggle/input/jigsaw-strategy3-lgbm-models/lgbm_models/lgbm_fold_{fold}.pkl"
    model = joblib.load(model_path)
    lgbm_models.append(model)

print("Successfully loaded 5 LightGBM fold-models.")

# --- Predict with each LightGBM model and average the results ---
test_features = ensemble_df[CFG.MODELS_TO_RUN]
final_predictions = np.zeros(len(test_features))

for model in lgbm_models:
    # We use predict_proba to get the probability of the positive class (1)
    final_predictions += model.predict_proba(test_features)[:, 1] / len(lgbm_models)

# --- Create the final submission file ---
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': final_predictions
})

# --- Final Sanity Check ---
assert not submission_df.isnull().values.any(), "Submission contains NaN values!"
print("Final predictions validated.")

submission_df.to_csv('submission.csv', index=False)
print("\nStacking submission.csv created successfully!")
display(submission_df.head())

