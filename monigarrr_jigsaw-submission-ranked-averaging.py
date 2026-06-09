# --- Imports and Setup ---
import os
import gc
import torch
import pandas as pd
import numpy as np
import sys
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from scipy.stats import rankdata

# Import our custom dataset class and helper functions
SCRIPTS_DIR = "/kaggle/input/jigsaw-scripts-monigarr/"
sys.path.append(SCRIPTS_DIR)

from training_utils import JigsawDataset, softmax

# Instantiate the config
# cfg = CFG()

# --- Configuration ---
class CFG:
    # Paths to the input datasets on Kaggle
    COMPETITION_DATA_DIR = "/kaggle/input/jigsaw-agile-community-rules/"
    MODELS_INPUT_DIR = "/kaggle/input/final-models-monigarr/final_models/"
    
    # Use the final, best OOF AUC scores for each model for weighting
    MODELS = {
        "deberta-v2-xlarge": 0.81955,
        "deberta-v3-large": 0.82763,
        "deberta-v3-base": 0.83356,
        "roberta-large": 0.80837,
        "xlm-roberta-large": 0.70696,
        "electra-large-discriminator": 0.80272,
    }
    
    MAX_LEN = 256
    VALID_BATCH_SIZE = 32


# The path you are trying to load from
debug_path = "/kaggle/input/final-models-monigarr/final_models/deberta-v3-base-finetuned/fold0/"

print(f"Checking contents of: {debug_path}")
try:
    files = os.listdir(debug_path)
    print("Files found:")
    for f in files:
        print(f"- {f}")
except FileNotFoundError:
    print("\n--> ERROR: This directory does not exist! Please check your path.")



# --- Load and Prepare Test Data ---
test_df = pd.read_csv(f"{CFG.COMPETITION_DATA_DIR}test.csv")
tokenizer = AutoTokenizer.from_pretrained(f"{CFG.MODELS_INPUT_DIR}deberta-v3-base-finetuned/fold0/") # Load one tokenizer

# Create the combined text column and pre-tokenize
sep = tokenizer.sep_token
test_df['full_text'] = test_df['rule'] + sep + test_df['body']
test_tokenized = tokenizer(
    test_df['full_text'].tolist(),
    max_length=CFG.MAX_LEN,
    padding='max_length',
    truncation=True
)

# Create the final PyTorch test dataset
test_dataset = JigsawDataset(test_tokenized, [0] * len(test_df))

print("Test data prepared successfully.")


# --- Run Inference ---
all_model_preds = {}

for model_name in CFG.MODELS.keys():
    print(f"\n--- Inferencing with {model_name} ---")
    model_fold_preds = []
    for fold in range(5):
        print(f"  > Processing Fold {fold}...")
        
        # Define path and load the specific fold's model
        model_path = f"{CFG.MODELS_INPUT_DIR}{model_name}/fold{fold}"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create a simple Trainer for fast prediction
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="./temp_output", per_device_eval_batch_size=CFG.VALID_BATCH_SIZE),
        )
        
        # Predict, get probabilities, and append to list
        test_output = trainer.predict(test_dataset)
        test_probs = softmax(test_output.predictions)
        model_fold_preds.append(test_probs[:, 1])
        
        # Clean up memory
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Average the predictions from all 5 folds for this model
    all_model_preds[model_name] = np.mean(model_fold_preds, axis=0)

print("\nInference complete for all models.")


# --- Ensemble Predictions and Create Submission File ---

# Create a DataFrame from the averaged predictions
ensemble_df = pd.DataFrame(all_model_preds)
ensemble_df['row_id'] = test_df['row_id']

# --- Weighted Average Ensemble ---
weights = np.array([score**2 for score in CFG.MODELS.values()])
normalized_weights = weights / np.sum(weights)

ensemble_df['rule_violation'] = np.average(ensemble_df[list(CFG.MODELS.keys())], weights=normalized_weights, axis=1)

submission_df = ensemble_df[['row_id', 'rule_violation']]

# --- Final Sanity Check ---
assert submission_df['rule_violation'].min() >= 0.0
assert submission_df['rule_violation'].max() <= 1.0
assert not submission_df.isnull().values.any()
print("Final predictions validated.")

# --- Save the submission.csv file ---
submission_df.to_csv('submission.csv', index=False)

print("\nsubmission.csv created successfully!")
submission_df.head()

