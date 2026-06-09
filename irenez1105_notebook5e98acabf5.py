import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    logging as hf_logging
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils import resample # For balancing subsets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_SIZE = 16

def load_true_test():
    return pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
def predict_proba(model, tokenizer, df):
    """Generates sigmoid probabilities for the dataframe."""
    if model is None or len(df) == 0:
        print("Skipping prediction: Model is None or dataframe is empty.")
        # Return array of 0.5 (neutral prediction) of correct shape if df is empty or model failed
        return np.full(len(df), 0.5)

    print(f"Generating predictions for {len(df)} samples...")
    start_time = time.time()
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # No labels needed for prediction

    # Manual prediction loop (Trainer.predict is sometimes tricky with custom setups)
    model.eval()
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(tokenized_dataset), EVAL_BATCH_SIZE):
            batch_indices = range(i, min(i + EVAL_BATCH_SIZE, len(tokenized_dataset)))
            batch = tokenized_dataset[batch_indices]

            # Manually create input tensors
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            if probs.ndim == 0: # Handle single prediction case
                all_probs.append(float(probs))
            else:
                all_probs.extend(probs.tolist())

    print(f"Finished prediction. Time: {time.time() - start_time:.2f}s")
    return np.array(all_probs)
true_test_df = load_true_test()
model_output_dir = '/kaggle/input/deberta/transformers/default/1/kaggle/working/results_deberta/checkpoint-489'
tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_output_dir).to(device)
proba = predict_proba(model, tokenizer, true_test_df)

submission_df = pd.DataFrame({
    'id': true_test_df['id'],  # Assuming 'id' column is in the test data
    'generated': proba
})
submission_df.to_csv('submission.csv', index=False)
print('all done')

