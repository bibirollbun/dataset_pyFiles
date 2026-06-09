import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np


# 1. Load Data
try:
    train_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')
    sample_sub = pd.read_csv('/kaggle/input/rmit-hackathon-2025/sample_submission.csv')

    print(f"Data loaded: Train {train_df.shape}, Test {test_df.shape}")
except FileNotFoundError as e:
    print(f"Error: File not found {e.filename}. Please check dataset path.")


# 2. Preprocessing
train_df['label_numeric'] = train_df['label'].map({'jailbreak': 1, 'benign': 0})
print("\nLabel distribution (0=benign, 1=jailbreak):")
print(train_df['label_numeric'].value_counts(normalize=True))


# 3. Build Model Pipeline
X_train = train_df['text']
y_train = train_df['label_numeric']
X_test = test_df['text']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char',      # GIỮ NGUYÊN: Phân tích ký tự là mấu chốt
        ngram_range=(3, 8),   # THAY ĐỔI: Tăng phạm vi (giống LGBM)
        max_features=75000,   # THAY ĐỔI: Tăng mạnh số lượng features
        stop_words=None       # GIỮ NGUYÊN: Không xóa stop words
    )),
    ('clf', LogisticRegression(
        C=10.0,               # THAY ĐỔI: Tăng C, cho mô hình linh hoạt hơn
        solver='saga',        # THAY ĐỔI: 'saga' tối ưu cho dataset lớn
        penalty='l2',         # Dùng L2 regularization
        random_state=42,
        class_weight='balanced',
        max_iter=2000,        # THAY ĐỔI: Tăng max_iter cho 'saga' hội tụ
        n_jobs=-1             # Dùng tất cả CPU
    ))
])


# 3.5 (TÙY CHỌN) Chạy Cross-Validation để kiểm tra điểm cục bộ
print("\nStarting 5-Fold Cross-Validation (Optional)...")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    print(f"--- Fold {fold + 1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    pipeline.fit(X_train_fold, y_train_fold)
    val_preds = pipeline.predict_proba(X_val_fold)[:, 1]
    score = roc_auc_score(y_val_fold, val_preds)
    
    fold_scores.append(score)
    print(f"Fold {fold + 1} ROC-AUC: {score:.5f}")

print("\n---------------------------------")
print(f"MEAN CV SCORE (5-Fold): {np.mean(fold_scores):.5f}")
print("---------------------------------")


# 4. Train Model
print("\nTraining final model (TF-IDF Char-Ngrams + Logistic Regression)...")
pipeline.fit(X_train, y_train)
print("Training complete!")


# 5. Predict on Test Set
test_preds = pipeline.predict_proba(X_test)[:, 1]

# Tạo file submission
submission_df = pd.DataFrame({'Id': test_df['Id'], 'TARGET': test_preds})
submission_df.to_csv('submission_baseline2.csv', index=False)
print("File 'submission_baseline.csv' has been created.")

