
# Uncomment if CatBoost is not installed
# !pip install -q catboost




import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from catboost import CatBoostClassifier, Pool

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)
RANDOM_STATE = 42




DATA_DIR = "./"
TRAIN_FILE = "train.csv"
TEST_FILE  = "test.csv"
SAMPLE_SUB = "sample_submission.csv"

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)
display(train.head())
display(test.head())
display(sample_sub.head())




TARGET_COL = "y"   # change if your target column has a different name
ID_COL     = "id"  # change if your ID column has a different name

FEATURES = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
X = train[FEATURES].copy()
y = train[TARGET_COL].copy()
X_test = test[FEATURES].copy()

print("Number of features:", len(FEATURES))




cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
print("Categorical columns:", cat_cols)




strat = y if y.nunique() <= 20 else None
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
)
print("Train shape:", X_tr.shape, " Validation shape:", X_va.shape)




train_pool = Pool(X_tr, label=y_tr, cat_features=cat_idx)
valid_pool = Pool(X_va, label=y_va, cat_features=cat_idx)
test_pool  = Pool(X_test, cat_features=cat_idx)




cb = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    learning_rate=0.05,
    depth=6,
    iterations=5000,
    random_seed=RANDOM_STATE,
    verbose=200,
)

cb.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True,
    early_stopping_rounds=200
)




va_pred_proba = cb.predict_proba(valid_pool)[:, 1]
va_pred = (va_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_va, va_pred_proba)
acc = accuracy_score(y_va, va_pred)
f1  = f1_score(y_va, va_pred)
pre = precision_score(y_va, va_pred)
rec = recall_score(y_va, va_pred)

print(f"AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {pre:.4f} | Recall: {rec:.4f}")




importances = cb.get_feature_importance(train_pool, type="PredictionValuesChange")
feat_imp = pd.DataFrame({"feature": X_tr.columns, "importance": importances}).sort_values("importance", ascending=False)
display(feat_imp.head(20))




test_pred = cb.predict_proba(test_pool)[:, 1]

# Follow the sample submission format
id_col_in_sample = sample_sub.columns[0]
target_col_in_sample = sample_sub.columns[-1]

submission = pd.DataFrame({
    id_col_in_sample: test[id_col_in_sample] if id_col_in_sample in test.columns else test[ID_COL],
    target_col_in_sample: test_pred
})
submission_path = "submission_catboost.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")
submission.head()


