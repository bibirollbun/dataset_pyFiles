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


# ======================================================================================
# STEP 1: FINAL INFERENCE CONFIGURATION & HELPERS
# ======================================================================================
import torch, gc, os
import pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

class CFG:
    # --- Path to the model tokenizer files ---
    model_name = "/kaggle/input/deberta-v3-large-hf-weights"
    
    # --- Inference Parameters ---
    max_length = 256
    batch_size = 16
    n_folds = 5
    
    # --- Optimization Parameters ---
    temperature = 0.9
    confidence_threshold = 0.10
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Class ---
class MathMisconceptionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompts = df['prompt'].values
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        text = self.prompts[idx]
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len,
                                  padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

# --- Load and prepare the data ---
df_test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
def create_structured_prompt(row):
    return f"Question: {row['QuestionText']}\nStudent chose: {row['MC_Answer']}\nReasoning: {row['StudentExplanation']}"
df_test['prompt'] = df_test.apply(create_structured_prompt, axis=1)

df_train_full = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
df_train_full['Misconception'] = df_train_full['Misconception'].fillna('NA')
df_train_full['target'] = df_train_full['Category'] + ':' + df_train_full['Misconception']
mlb = MultiLabelBinarizer()
mlb.fit([[label] for label in df_train_full['target']])
CLASSES = mlb.classes_
print("Setup Complete. Test data and binarizer are ready.")


# ======================================================================================
# STEP 2: OPTIMIZED INFERENCE, ENSEMBLING, AND POST-PROCESSING (FINAL)
# ======================================================================================

# --- Paths to your 5 trained and optimized model files ---
# VERIFY THAT THESE PATHS AND FILENAMES ARE CORRECT IN YOUR INPUT SECTION
model_paths = [
    "/kaggle/input/train-map-deberta-large-fold-0/deberta_large_fold_0.pth",
    "/kaggle/input/train-map-deberta-large-fold-1/deberta_large_fold_1.pth",
    "/kaggle/input/train-map-deberta-large-fold-2/deberta_large_fold_2.pth",
    "/kaggle/input/map-deberta-large-fold-3/deberta_large_fold_3.pth",
    "/kaggle/input/map-deberta-large-fold-4/deberta_large_fold_4.pth"
]

# --- 1. OPTIMIZED WEIGHTS BASED ON YOUR FOLD SCORES ---
fold_scores = [
    0.9373, # Score from Fold 0
    0.9325, # Score from Fold 1
    0.9346, # Score from Fold 2
    0.9378, # Score from Fold 3
    0.9369  # Score from Fold 4
]
fold_weights = np.array(fold_scores) / np.sum(fold_scores)
print("Optimized Ensemble Weights:", fold_weights.tolist())

# --- 2. INFERENCE LOOP ---
all_fold_preds = []
for fold, path in enumerate(model_paths):
    print(f"\n====== PREDICTING WITH FOLD {fold} ======")
    
    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name, num_labels=len(CLASSES))
    # Load the trained weights
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    test_dataset = MathMisconceptionDataset(df_test, tokenizer, CFG.max_length)
    
    # DataLoader without multiprocessing for stability
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
    
    fold_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting Fold {fold}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # Use Sigmoid for multi-label probabilities
            probabilities = torch.sigmoid(outputs.logits)
            fold_preds.append(probabilities.cpu().numpy())
    
    all_fold_preds.append(np.vstack(fold_preds))
    
    # Clear memory after each fold
    del model, tokenizer, test_dataset, test_loader
    gc.collect()
    torch.cuda.empty_cache()

# --- 3. ENSEMBLE BY WEIGHTED AVERAGING ---
print("\nEnsembling predictions using weighted average...")
final_probs = np.average(all_fold_preds, axis=0, weights=fold_weights)

# --- 4. APPLY TEMPERATURE SCALING ---
if CFG.temperature != 1.0:
    print(f"Applying temperature scaling (T={CFG.temperature})...")
    final_probs = final_probs ** (1 / CFG.temperature)
    np.clip(final_probs, 0, 1, out=final_probs)

# --- 5. POST-PROCESSING WITH CONFIDENCE THRESHOLD ---
print(f"Applying confidence threshold of {CFG.confidence_threshold}...")
top_indices_all = np.argsort(final_probs, axis=1)[:, ::-1]
final_predictions_str = []

for i in range(len(final_probs)):
    preds_for_sample_indices = []
    # Get up to 3 predictions that are above the threshold
    for idx in top_indices_all[i]:
        if final_probs[i, idx] >= CFG.confidence_threshold and len(preds_for_sample_indices) < 3:
            preds_for_sample_indices.append(idx)
            
    # If we have fewer than 3, fill with the next best predictions regardless of threshold
    if len(preds_for_sample_indices) < 3:
        for idx in top_indices_all[i]:
            if idx not in preds_for_sample_indices and len(preds_for_sample_indices) < 3:
                preds_for_sample_indices.append(idx)
    
    # Convert indices to class names
    final_labels = mlb.classes_[preds_for_sample_indices]
    final_predictions_str.append(' '.join(final_labels))

# --- 6. FORMAT SUBMISSION ---
submission_df = pd.DataFrame({'row_id': df_test['row_id'], 'Category:Misconception': final_predictions_str})
submission_df.to_csv('submission.csv', index=False)

print("\nFinal optimized submission file created successfully!")
display(submission_df.head())

