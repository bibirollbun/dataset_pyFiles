import pandas as pd
import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import joblib
import gc

class CFG:
    # --- Input Paths ---
    COMPETITION_DATA_DIR = "/kaggle/input/jigsaw-agile-community-rules/"
    BASE_MODELS_DIR = "/kaggle/input/monigarr-jigsaw-competition-models-aug-9-2025/final_models/"
    LGBM_MODELS_DIR = "/kaggle/input/monigarr-jigsaw-lgbm-models/"

    MODEL_NAMES = [
        "deberta-v3-base-finetuned",
        "deberta-v3-large-finetuned",
        "roberta-base-reddit-pretrained",
        "deberta-v2-xlarge-finetuned"
    ]
    
     # --- Inference Hyperparameters ---
    MAX_LEN = 256
    VALID_BATCH_SIZE = 32


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class JigsawDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = torch.tensor(tokenized_data['input_ids'])
        self.attention_mask = torch.tensor(tokenized_data['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}

# Load and prepare the competition test data
test_df = pd.read_csv(f"{CFG.COMPETITION_DATA_DIR}test.csv")
test_df['full_text'] = test_df['rule'] + "[SEP]" + test_df['body'] # Placeholder separator
print("Test data loaded successfully.")


all_model_preds = {}

for model_name in CFG.MODEL_NAMES:
    print(f"\n--- Tier 1 Inference: {model_name} ---")
    
    # Conditionally set the folder name
    folder_name = model_name # All your final models use the direct name
    
    # Load the correct tokenizer for this model
    tokenizer_path = f"{CFG.BASE_MODELS_DIR}{folder_name}/fold0/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Tokenize the test data using this model's specific tokenizer
    test_df['full_text'] = test_df['rule'] + tokenizer.sep_token + test_df['body']
    test_tokenized = tokenizer(test_df['full_text'].tolist(), max_length=CFG.MAX_LEN, padding='max_length', truncation=True)
    test_dataset = JigsawDataset(test_tokenized)
    
    model_fold_preds = []
    for fold in range(5):
        print(f"  > Processing Fold {fold}...")
        model_path = f"{CFG.BASE_MODELS_DIR}{folder_name}/fold{fold}"
        model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
        
        trainer = Trainer(model=model, args=TrainingArguments(output_dir="./temp", per_device_eval_batch_size=CFG.VALID_BATCH_SIZE, fp16=True, report_to="none"))
        
        test_output = trainer.predict(test_dataset)
        test_probs = softmax(test_output.predictions)
        model_fold_preds.append(test_probs[:, 1])
        
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    all_model_preds[model_name] = np.mean(model_fold_preds, axis=0)

print("\n--- Tier 1 Inference Complete ---")


# Create a DataFrame from the base models' predictions. This is now our feature set.
test_features_df = pd.DataFrame(all_model_preds)

# Load the 6 trained LightGBM models (0 to 5 = 6)
lgbm_models = []
for fold in range(5):
    model_path = f"{CFG.LGBM_MODELS_DIR}lgbm_fold_{fold}.pkl"
    model = joblib.load(model_path)
    lgbm_models.append(model)
print("Successfully loaded 6 LightGBM fold-models.")

# Predict with each LightGBM model and average the results
final_predictions = np.zeros(len(test_features_df))
for model in lgbm_models:
    final_predictions += model.predict_proba(test_features_df[CFG.MODEL_NAMES])[:, 1] / len(lgbm_models)

# Create the final submission file
submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': final_predictions})

# Final Sanity Check
assert not submission_df.isnull().values.any(), "Submission contains NaN values!"
print("\nFinal predictions validated.")

submission_df.to_csv('submission.csv', index=False)
print("Stacking submission.csv created successfully!")
display(submission_df.head())




