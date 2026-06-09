# --- 1. Configuration ---
class CFG:
    # A great, fast, and high-quality sentence transformer model
    sbert_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    n_splits = 5
    random_state = 42
    debug = False # Set to True to run on a small sample for a quick test


# --- 2. Setup & Imports ---
print("ðŸ”¹ Installing necessary libraries...")
!pip install -q -U sentence-transformers

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
import torch
from tqdm.auto import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 3. Load Data & Preprocessing ---
print("\nðŸ”¹ Loading and preparing data...")
try:
    df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
except FileNotFoundError:
    print("Running locally.")
    df = pd.read_csv('train.csv')

if CFG.debug:
    print("ðŸ”¥ RUNNING IN DEBUG MODE ON A SMALL SUBSET (1000 samples) ðŸ”¥")
    df = df.sample(n=1000, random_state=CFG.random_state).reset_index(drop=True)

# Create target and text features
df['Misconception'] = df['Misconception'].fillna('NA').astype(str)
df['target'] = df['Category'] + ':' + df['Misconception']
le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['target'])

df['QuestionText'] = df['QuestionText'].fillna('')
df['MC_Answer'] = df['MC_Answer'].fillna('')
df['StudentExplanation'] = df['StudentExplanation'].fillna('')

def create_prompt(row):
    is_correct = "Yes" if "True_" in row['Category'] else "No"
    q = f"Question: {row['QuestionText']}"
    a = f"Answer: {row['MC_Answer']}"
    c = f"Correct? {is_correct}"
    e = f"Student Explanation: {row['StudentExplanation']}"
    return f"{q}\n{a}\n{c}\n{e}"

df['full_text'] = df.apply(create_prompt, axis=1)
print("Data loaded and prompts created.")
print("-" * 50)


df.head()


# --- 4. Generate Sentence Embeddings ---
print(f"ðŸ”¹ Loading Sentence-BERT model: {CFG.sbert_model_name}...")
# Load the model onto the GPU
sbert_model = SentenceTransformer(CFG.sbert_model_name, device=device)

print("\nðŸ”¹ Generating embeddings for all text data (this may take a few minutes)...")
# The .encode() method takes a list of strings and returns a numpy array of embeddings
all_embeddings = sbert_model.encode(
    df['full_text'].tolist(), 
    show_progress_bar=True,
    batch_size=64 # Use a good batch size for GPU
)

print(f"Embeddings created. Shape: {all_embeddings.shape}")
print("-" * 50)



# --- 5. Custom MAP@3 Metric Function ---
def map_at_3(y_true, y_pred_proba):
    # (Same map_at_3 function as before)
    top_3_preds = np.argsort(-y_pred_proba, axis=1)[:, :3]
    avg_precisions = []
    for i in range(len(y_true)):
        true_label = y_true[i]
        top_3 = top_3_preds[i]
        if true_label in top_3:
            rank = np.where(top_3 == true_label)[0][0] + 1
            avg_precisions.append(1 / rank)
        else:
            avg_precisions.append(0)
    return np.mean(avg_precisions)


# --- 6. Cross-Validation Training Loop ---
print("ðŸ”¹ Starting 5-fold cross-validation with LightGBM...")
skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_state)
oof_scores = []

# Our features are the embeddings, our target is the encoded label
X = all_embeddings
y = df['target_encoded']

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold} =====")

    # Get train and validation sets for this fold
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Define and train the LightGBM model
    lgbm = lgb.LGBMClassifier(
        objective='multiclass',
        class_weight='balanced',
        n_estimators=500, # A reasonable number of trees
        learning_rate=0.05,
        num_leaves=31,
        random_state=CFG.random_state,
        n_jobs=-1,
        colsample_bytree=0.8, # Add some regularization
    )
    
    print("Training LightGBM model...")
    lgbm.fit(X_train, y_train)
             
    # Make predictions and evaluate
    print("Making predictions...")
    val_pred_probas = lgbm.predict_proba(X_val)
    
    fold_score = map_at_3(y_val.values, val_pred_probas)
    oof_scores.append(fold_score)
    print(f"âœ… Fold {fold} MAP@3 Score: {fold_score:.4f}")

    # Clean up
    del lgbm


# --- 7. Final Results ---
print("\n" + "="*50)
print("Cross-validation complete.")
print(f"Scores for each fold: {[round(s, 4) for s in oof_scores]}")
print(f"ðŸ“ˆ Average CV MAP@3 Score: {np.mean(oof_scores):.4f}")
print(f"Standard Deviation of scores: {np.std(oof_scores):.4f}")
print("="*50)


# --- 7. Final Training and Asset Saving ---
print("\n" + "="*50)
print("âœ… CV finished. Now training final model on 100% of data and saving all assets...")


# --- 7.1: Save the fitted LabelEncoder ---
joblib.dump(le, 'label_encoder.pkl')
print("LabelEncoder saved.")


# --- 7.2: Save the generated training embeddings ---
# This saves you from having to re-calculate them later
np.save('train_embeddings.npy', all_embeddings)
print("Training embeddings saved.")


# --- 7.3: Train the final model on all data ---
X_full_train = all_embeddings
y_full_train = df['target_encoded']

final_lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    class_weight='balanced',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=CFG.random_state,
    n_jobs=-1,
    colsample_bytree=0.8,
)
print("\nTraining final LGBM model on all data...")
final_lgbm.fit(X_full_train, y_full_train)
print("Final model trained.")


# --- 7.4: Save the final trained LGBM model ---
joblib.dump(final_lgbm, 'lgbm_final_model.pkl')
print("Final LGBM model saved.")

print("\nAll assets saved. You can now commit this notebook and use the outputs for inference.")
print("="*50)

