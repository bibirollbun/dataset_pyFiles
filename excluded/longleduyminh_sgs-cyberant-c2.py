# ðŸ§© Challenge 2 â€” Protector Model (RMIT Hackathon 2025)
# Kernel name: where2.SGS_CyberANT_C2
# Goal: Distinguish Jailbreak vs Benign (binary) with high ROC-AUC
# Pipeline: word & char TF-IDF + LR/SVM (calibrated) + 5-Fold OOF + Stacking + prompt-aware features

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc


#1 Load Data
train_df = pd.read_csv('/kaggle/input/dataset-c2/train.csv')
test_df = pd.read_csv('/kaggle/input/dataset-c2/test.csv')

train_df['labels'] = train_df['label'].map({'jailbreak': 1, 'benign': 0})
    
# Prepare data for the model
X = train_df['text']
y = train_df['labels']
X_test = test_df['text']

print(f"Data loaded successfully: Train {train_df.shape}, Test {test_df.shape}")


#2 Build Model Pipeline (TF-IDF + LightGBM)

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=20000,
)

lgbm_classifier = LGBMClassifier(
    n_estimators=500,        
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('clf', lgbm_classifier)
])


#3 Evaluate Model (Cross-Validation & Plot ROC-AUC)
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_scores = []

plt.figure(figsize=(10, 8))
ax = plt.gca()

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- Starting Fold {fold + 1}/{N_SPLITS} ---")
    
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
    pipeline.fit(X_train_fold, y_train_fold)
    
    val_preds = pipeline.predict_proba(X_val_fold)[:, 1]
    
    score = roc_auc_score(y_val_fold, val_preds)
    fold_scores.append(score)
    print(f"Fold {fold + 1} ROC-AUC: {score:.5f}")

    fpr, tpr, _ = roc_curve(y_val_fold, val_preds)
    
    plt.plot(fpr, tpr, lw=2, label=f'Fold {fold + 1} (AUC = {score:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - 5 Fold Cross-Validation', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"MEAN CV SCORE (5-Fold): {np.mean(fold_scores):.5f}")
print(f"CV Standard Deviation: {np.std(fold_scores):.5f}")


#4 Train Final Model & Create Submission

pipeline.fit(X, y)

test_preds = pipeline.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({'Id': test_df['Id'], 'TARGET': test_preds})
submission_df.to_csv('submission.csv', index=False)

print("File 'submission.csv' has been created.")

