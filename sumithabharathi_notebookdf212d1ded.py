import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import os
import sys

# --- 1. Load Data with Correct Path ---
# FINAL FIX: Using the confirmed path provided by the user.
DATA_DIR = "/kaggle/input/train-and-test/"

df_train = None
df_test = None

try:
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    print(f"✅ Data files loaded successfully from: {DATA_DIR}")

except FileNotFoundError:
    print(f"❌ FATAL ERROR: Data files not found at expected path: {DATA_DIR}")
    print("Please ensure the data folder name is correct.")
    # Stop script to prevent NameError
    sys.exit(1) 

# --- 2. Feature Engineering Setup ---

# Define features and target
X_train_body = df_train['body'].fillna('')
X_test_body = df_test['body'].fillna('')
X_train_rule = df_train[['rule']]
X_test_rule = df_test[['rule']]
y_train = df_train['rule_violation']

# 2a. TF-IDF on 'body' (comment text)
print("⚙️ Fitting TF-IDF on comment body...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_body_tfidf = tfidf.fit_transform(X_train_body)
X_test_body_tfidf = tfidf.transform(X_test_body)

# 2b. One-hot encode 'rule'
print("⚙️ One-Hot Encoding 'rule'...")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_train_rule_ohe = encoder.fit_transform(X_train_rule)
X_test_rule_ohe = encoder.transform(X_test_rule)

# 2c. Combine features
print("⚙️ Combining features...")
X_train_final = hstack([X_train_body_tfidf, X_train_rule_ohe])
X_test_final = hstack([X_test_body_tfidf, X_test_rule_ohe])
print(f"Final training feature shape: {X_train_final.shape}")

# --- 3. Model Training ---

# Use Logistic Regression as a baseline classifier
print("⚙️ Training Logistic Regression model...")
model = LogisticRegression(solver='liblinear', random_state=42, C=1.0)
model.fit(X_train_final, y_train)

# --- 4. Prediction and Submission ---

# Predict the probability of the positive class (rule_violation = 1)
print("⚙️ Generating predictions on test data...")
test_predictions_proba = model.predict_proba(X_test_final)[:, 1]

# Create submission file
submission_df = pd.DataFrame({
    'row_id': df_test['row_id'],
    'rule_violation': test_predictions_proba
})

# Save the submission.csv file, which Kaggle requires for scoring
submission_file_name = "submission.csv"
submission_df.to_csv(submission_file_name, index=False)

print("\n--- Notebook Run Complete ---")
print(f"✅ Output file saved as: {submission_file_name}")

