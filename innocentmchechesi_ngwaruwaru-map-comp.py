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


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import lightgbm as lgb
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration class for parameters
class CFG:
    # Model and tokenizer path (ensure you add the dataset from Kaggle)
    # Example Kaggle dataset: 'microsoft-deberta-v3-base'
    model_path = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-base' 
    
    # Data paths
    train_path = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
    test_path = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
    sample_submission_path = '/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv'
    
    # Model parameters
    seed = 42
    n_splits = 5
    max_length = 256 # Max sequence length for tokenizer
    
    # For GPU acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)


# Load raw data
train_df = pd.read_csv(CFG.train_path)
test_df = pd.read_csv(CFG.test_path)
sample_submission_df = pd.read_csv(CFG.sample_submission_path)

# --- 1. Create the Combined Text Input ---
# The model needs full context. We'll create a 'full_text' column.
# We'll use a [SEP] token to clearly separate the parts for the model.
def create_full_text(df):
    df['full_text'] = (
        df['QuestionText'].str.strip() +
        ' [SEP] ' +
        df['MC_Answer'].str.strip() +
        ' [SEP] ' +
        df['StudentExplanation'].str.strip()
    )
    return df

train_df = create_full_text(train_df)
test_df = create_full_text(test_df)

# --- 2. Create and Aggregate Target Labels ---
# IMPORTANT FIX: Fill NaN values in 'Misconception' with the string 'NA' first.
train_df['Misconception'] = train_df['Misconception'].fillna('NA')

# Now, create the 'Category:Misconception' target format safely.
train_df['target_label'] = train_df['Category'] + ':' + train_df['Misconception']

# Group by the explanation and aggregate the labels into a list.
agg_cols = ['QuestionId', 'StudentExplanation', 'full_text']
train_agg = train_df.groupby(agg_cols).agg({
    'target_label': lambda x: list(x)
}).reset_index()

print(f"Original train rows: {len(train_df)}")
print(f"Aggregated train rows: {len(train_agg)}")
print("\nSample of aggregated data:")
display(train_agg.head())

# --- 3. Prepare Test Data ---
# The test data is already one row per explanation. We just need the 'full_text'.
# We will drop duplicates just in case.
test_processed = test_df.drop_duplicates(subset=['row_id']).reset_index(drop=True)


# Initialize the binarizer
mlb = MultiLabelBinarizer()

# Fit on the aggregated labels and transform to a binary matrix
y = mlb.fit_transform(train_agg['target_label'])

# These are all the unique 'Category:Misconception' pairs
print("Shape of binary target matrix:", y.shape)
print("Number of unique classes:", len(mlb.classes_))
print("Example Classes:", mlb.classes_[:5])


def get_text_features(df):
    """Creates basic statistical features from the full_text."""
    features = pd.DataFrame()
    # Ensure the column exists before processing
    if 'full_text' in df.columns:
        features['text_length'] = df['full_text'].str.len()
        features['word_count'] = df['full_text'].str.split().str.len()
    return features

def get_embeddings(df, model_path, max_length, device):
    """Generates sentence embeddings from the full_text using a transformer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    all_embeddings = []
    text_list = df['full_text'].tolist() if 'full_text' in df.columns else []

    for text in tqdm(text_list, desc="Generating Embeddings"):
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        mean_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        all_embeddings.append(mean_embedding)

    return np.array(all_embeddings)

# --- Generate features for training data ---
print("Generating features for the training data...")
train_text_features = get_text_features(train_agg)
train_embeddings = get_embeddings(train_agg, CFG.model_path, CFG.max_length, CFG.device)
X = np.concatenate([train_embeddings, train_text_features.values], axis=1)

# --- Generate features for test data ---
print("\nGenerating features for the test data...")
test_text_features = get_text_features(test_processed)
test_embeddings = get_embeddings(test_processed, CFG.model_path, CFG.max_length, CFG.device)
X_test = np.concatenate([test_embeddings, test_text_features.values], axis=1)

print(f"\nFinal feature matrix shapes: Train={X.shape}, Test={X_test.shape}")


# We will store models and out-of-fold predictions
models = []
oof_preds = np.zeros(y.shape)

kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{CFG.n_splits} ---")
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    fold_models = []
    for i in range(y.shape[1]):
        lgb_clf = lgb.LGBMClassifier(random_state=CFG.seed)
        lgb_clf.fit(X_train, y_train[:, i])
        val_preds = lgb_clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx, i] = val_preds
        fold_models.append(lgb_clf)

    models.append(fold_models)

# --- Evaluate OOF predictions to find the best threshold ---
best_threshold = 0
best_f1 = 0
for threshold in np.arange(0.1, 0.9, 0.01):
    preds = (oof_preds > threshold).astype(int)
    f1 = f1_score(y, preds, average='micro')
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nOptimal Micro F1 Score on OOF predictions: {best_f1:.4f}")
print(f"Best Threshold found: {best_threshold:.2f}")


# --- Predict on Test Data ---
test_preds_all_folds = np.zeros((len(test_processed), len(mlb.classes_)))

for fold_models in models:
    fold_preds = np.zeros((len(test_processed), len(mlb.classes_)))
    for i, model in enumerate(fold_models):
        fold_preds[:, i] = model.predict_proba(X_test)[:, 1]
    test_preds_all_folds += fold_preds / CFG.n_splits

# --- Format for Submission ---
# Apply the optimal threshold
final_preds_binary = (test_preds_all_folds > best_threshold).astype(int)

# Convert the binary predictions back to label strings
predictions_labels = mlb.inverse_transform(final_preds_binary)

# Handle cases where no label is predicted (model predicts all zeros for a row)
# The competition requires up to three, but allows zero. An empty string is appropriate.
formatted_labels = [' '.join(labels) if labels else '' for labels in predictions_labels]

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'row_id': test_processed['row_id'],
    'Category:Misconception': formatted_labels
})

# Save to submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
display(submission_df.head())

