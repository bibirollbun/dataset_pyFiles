# =============================================================
# Notebook: exp007_sbert_lgbm_inference.ipynb
# =============================================================

import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import torch
import os


# --- 1. Configuration & Setup ---
class CFG:
    sbert_model_name = '/kaggle/input/sbert-all-minilm-l6-v2-files/all-MiniLM-L6-v2'
    # Path to the output of your TRAINING notebook
    training_output_path = '/kaggle/input/exp005-sbert-lgbm-training'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Load Pre-trained Artifacts ---
print("ðŸ”¹ Loading pre-trained assets (model, encoder)...")
final_lgbm = joblib.load(os.path.join(CFG.training_output_path, 'lgbm_final_model.pkl'))
le = joblib.load(os.path.join(CFG.training_output_path, 'label_encoder.pkl'))
sbert_model = SentenceTransformer(CFG.sbert_model_name, device=device)
print("Assets loaded.")
print("-" * 50)


# --- 3. Load and Prepare TEST Data ---
print("ðŸ”¹ Preparing test data...")
# We still need train_df to create the 'is_correct' feature for the test set prompt
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')

correct = train_df[train_df.Category.str.contains('True_')].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False).drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
correct['is_correct_pred'] = 1
test_df = test_df.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test_df['is_correct_pred'] = test_df['is_correct_pred'].fillna(0)
test_df['Category'] = test_df['is_correct_pred'].apply(lambda x: 'True_' if x == 1 else 'False_')

def create_prompt(row):
    is_correct = "Yes" if "True_" in row['Category'] else "No"
    q = f"Question: {row['QuestionText']}"
    a = f"Answer: {row['MC_Answer']}"
    c = f"Correct? {is_correct}"
    e = f"Student Explanation: {row['StudentExplanation']}"
    return f"{q}\n{a}\n{c}\n{e}"

test_df['full_text'] = test_df.apply(create_prompt, axis=1)
print("Test prompts created.")
print("-" * 50)



# --- 4. Generate Embeddings for TEST data ---
print("ðŸ”¹ Generating embeddings for test data...")
X_test_embeddings = sbert_model.encode(
    test_df['full_text'].tolist(), 
    show_progress_bar=True,
    batch_size=128
)
print("Test embeddings created.")
print("-" * 50)


# --- 5. Predict and Create Submission ---
print("ðŸ”¹ Predicting on test set...")
test_pred_probas = final_lgbm.predict_proba(X_test_embeddings)

top_3_indices = np.argsort(-test_pred_probas, axis=1)[:, :3]
top_3_labels = le.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)
predictions_str = [' '.join(labels) for labels in top_3_labels]

sample_submission_df['Category:Misconception'] = predictions_str
sample_submission_df.to_csv('submission.csv', index=False)

print("\n" + "="*50)
print("âœ… submission.csv created successfully!")
print(sample_submission_df.head())
print("="*50)

