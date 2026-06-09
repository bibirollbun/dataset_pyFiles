import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


target = "loan_paid_back"
X = train.drop(columns=[target])
y = train[target]


# Identify categorical columns
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
print("Categorical features:", cat_features)


# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Train CatBoost model
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    eval_metric='AUC',
    random_seed=42,
    cat_features=cat_features,
    early_stopping_rounds=100,
    verbose=200
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)


# Evaluate
val_preds = model.predict_proba(X_val)[:, 1]
roc = roc_auc_score(y_val, val_preds)
print("Validation ROC-AUC:", roc)

# Predict on test data
test_preds = model.predict_proba(test)[:, 1]


# ✅ Create correct submission file
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": (test_preds > 0.5).astype(int)   # Convert probabilities to 0/1
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv (only id and loan_paid_back)")




