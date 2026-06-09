import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

# --- Configuration ---
# Set to True to run on a small sample of data for quick debugging
DEBUG = False

# --- Prerequisite: Setup from Step 1 ---
print("Executing Step 1: Data Loading and CV Setup...")

try:
    df_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")
    print("Train, test, and submission files loaded successfully!")
except FileNotFoundError:
    print("Error: Competition files not found. Falling back to dummy data.")
    # Create dummy dataframes that mimic the real data structure
    df_train = pd.DataFrame({
        'body': ['This is a test comment about cats.', 'Another one here about dogs.', 'A third comment about birds.', 'And a fourth about fish.'],
        'rule': ['Rule A', 'Rule B', 'Rule A', 'Rule B'],
        'positive_example_1': ['cats are fluffy', 'dogs are loyal', 'birds can fly', 'fish can swim'],
        'positive_example_2': ['cats are cute', 'dogs are friendly', 'birds have feathers', 'fish live underwater'],
        'negative_example_1': ['dogs are not cats', 'cats are not dogs', 'fish cannot fly', 'birds cannot swim'],
        'negative_example_2': ['cars are fast', 'trains are long', 'boats float', 'planes fly'],
        'rule_violation': [1, 1, 0, 0]
    })
    df_test = pd.DataFrame({
        'row_id': [100, 101],
        'body': ['This is a new comment about cats.', 'A new comment about planes.'],
        'rule': ['Rule A', 'Rule B'],
        'positive_example_1': ['cats are fluffy', 'planes fly high'],
        'positive_example_2': ['cats are cute', 'planes have wings'],
        'negative_example_1': ['dogs are not cats', 'trains do not fly'],
        'negative_example_2': ['cars are fast', 'boats float']
    })

if DEBUG:
    print("DEBUG mode is ON. Using a small sample of the data.")
    df_train = df_train.sample(n=1000, random_state=42).reset_index(drop=True)

# --- Preprocessing and CV Setup ---
# Combine example texts for both train and test sets
for df in [df_train, df_test]:
    df['positive_examples'] = df['positive_example_1'].fillna('') + ' ' + df['positive_example_2'].fillna('')
    df['negative_examples'] = df['negative_example_1'].fillna('') + ' ' + df['negative_example_2'].fillna('')

# Convert rule text to numerical IDs for grouping
df_train['rule_id'], rule_uniques = pd.factorize(df_train['rule'])
num_unique_rules = len(rule_uniques)

N_SPLITS = min(5, num_unique_rules)
if N_SPLITS < 2:
    raise ValueError("Not enough unique rules to perform GroupKFold. Need at least 2.")

# Create the folds
df_train['fold'] = -1
gkf = GroupKFold(n_splits=N_SPLITS)
folds = gkf.split(X=df_train, y=df_train.rule_violation, groups=df_train.rule_id)
for fold_idx, (train_indices, val_indices) in enumerate(folds):
    df_train.loc[val_indices, 'fold'] = fold_idx

print("Cross-validation setup complete.")
print("-" * 50)

# --- Step 2 & 4 & 5: Train, Validate, and Predict ---
print("\nExecuting Full Pipeline: TF-IDF + Logistic Regression...")

text_cols = ['body', 'rule', 'positive_examples', 'negative_examples']
oof_scores = []
test_preds = [] # To store predictions on the test set from each fold model

# Loop through each fold to train, validate, and predict
for fold in range(N_SPLITS):
    print(f"\n===== Fold {fold} =====")

    # 1. Split data
    train_df = df_train[df_train.fold != fold].reset_index(drop=True)
    valid_df = df_train[df_train.fold == fold].reset_index(drop=True)

    print(f"Training on {len(train_df)} samples, validating on {len(valid_df)} samples.")

    # 2. Create and fit TF-IDF Vectorizers
    vectorizers = {}
    for col in text_cols:
        vec = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=5000)
        vectorizers[col] = vec
        vec.fit(train_df[col].fillna(''))

    # 3. Transform data and combine features
    def get_features(df, is_train=True):
        feature_matrices = [vectorizers[col].transform(df[col].fillna('')) for col in text_cols]
        return hstack(feature_matrices)

    X_train = get_features(train_df)
    X_valid = get_features(valid_df, is_train=False)
    y_train = train_df['rule_violation']
    y_valid = valid_df['rule_violation']

    # 4. Train a Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
    model.fit(X_train, y_train)

    # --- Validation ---
    print("Making predictions on validation set...")
    val_preds = model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, val_preds)
    oof_scores.append(auc_score)
    print(f"Fold {fold} ROC AUC Score: {auc_score:.5f}")

    # --- Inference (Step 4) ---
    print("Making predictions on the test set...")
    # Transform the test data using the vectorizers fitted on this fold's training data
    X_test = get_features(df_test, is_train=False)
    fold_test_preds = model.predict_proba(X_test)[:, 1]
    test_preds.append(fold_test_preds)

# --- Final Evaluation & Submission (Step 5) ---
print("-" * 50)
mean_auc = np.mean(oof_scores)
print(f"\nâœ… Mean ROC AUC score across all {N_SPLITS} folds: {mean_auc:.5f}")

# Average the predictions from all fold models
print("\nAveraging test predictions from all folds...")
final_preds = np.mean(test_preds, axis=0)

# Create the submission file
print("Creating submission file...")
submission_df = pd.DataFrame({'row_id': df_test['row_id'], 'rule_violation': final_preds})

# For the dummy data case, ensure the submission matches the sample format
if 'sample_submission' not in locals():
    sample_submission = pd.DataFrame({'row_id': df_test['row_id'], 'rule_violation': 0.5})

# Ensure row_ids match the sample submission
submission_df = sample_submission[['row_id']].merge(submission_df, on='row_id', how='left')
submission_df['rule_violation'] = submission_df['rule_violation'].fillna(0.5) # Fill any potential NaNs

submission_df.to_csv('submission.csv', index=False)
print("\nðŸŽ‰ `submission.csv` created successfully!")
print("Submission file head:")
print(submission_df.head())

