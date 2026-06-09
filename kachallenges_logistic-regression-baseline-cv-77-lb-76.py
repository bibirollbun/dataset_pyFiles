import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# -------------------------------
# 1. Load the Data
# -------------------------------
train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")  
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")    

X_train = train_df["Question"].values
y_train = train_df["label"].values
X_test  = test_df["Question"].values

# -------------------------------
# 2. Stratified K-Fold on Training Data
# -------------------------------
NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Initialize array for out-of-fold predictions
oof_preds = np.zeros(len(X_train), dtype=int)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {fold+1}/{NUM_FOLDS}")
    
    # Split train/validation for this fold
    X_trn, y_trn = X_train[trn_idx], y_train[trn_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X_trn_tfidf = vectorizer.fit_transform(X_trn)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_trn_tfidf, y_trn)
    
    # Predict on validation fold
    y_val_pred = model.predict(X_val_tfidf)
    oof_preds[val_idx] = y_val_pred
    
    fold_f1 = f1_score(y_val, y_val_pred, average="micro")
    print(f"Fold {fold+1} F1 (micro): {fold_f1:.4f}")

# Overall OOF F1
oof_f1 = f1_score(y_train, oof_preds, average="micro")
print("Overall OOF F1 (micro):", oof_f1)

# -------------------------------
# 3. Train Final Model on Full Training Data
# -------------------------------
final_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = final_vectorizer.fit_transform(X_train)

final_model = LogisticRegression(max_iter=2000, random_state=42)
final_model.fit(X_train_tfidf, y_train)

# -------------------------------
# 4. Make Predictions on the Test Set
# -------------------------------
X_test_tfidf = final_vectorizer.transform(X_test)
test_preds = final_model.predict(X_test_tfidf)

# -------------------------------
# 5. Prepare Submission File
# -------------------------------
submission = pd.DataFrame({"id": test_df["id"], "label": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")


