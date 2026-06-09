import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

print("âœ… Libraries imported successfully!")


# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


target = "loan_paid_back"
id_col = "id"

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

# Encode categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Scale numeric features
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"âœ… Preprocessing complete! Encoded {len(cat_cols)} categorical and scaled {len(num_cols)} numeric features.")


# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importances = pd.DataFrame()

# Model parameters
xgb_params = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    use_label_encoder=False
)

lgb_params = dict(
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    verbose=-1
)

cat_params = dict(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    eval_metric="AUC",
    random_seed=42,
    verbose=0
)


from lightgbm import early_stopping, log_evaluation

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n========== Fold {fold+1} ==========")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Train models (added class balancing)
    model_xgb = XGBClassifier(**xgb_params, scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]))
    model_lgb = LGBMClassifier(**lgb_params, class_weight="balanced")
    model_cat = CatBoostClassifier(**cat_params, scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]))

    # Fit models with updated LightGBM syntax
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=50,
        verbose=False
    )

    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(100), log_evaluation(0)]
    )

    model_cat.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid)
    )

    # Predictions for validation
    val_pred_xgb = model_xgb.predict_proba(X_valid)[:, 1]
    val_pred_lgb = model_lgb.predict_proba(X_valid)[:, 1]
    val_pred_cat = model_cat.predict_proba(X_valid)[:, 1]

    # Weighted blending
    val_pred = (0.4 * val_pred_lgb) + (0.35 * val_pred_xgb) + (0.25 * val_pred_cat)
    oof_preds[valid_idx] = val_pred

    # Average test predictions across folds
    test_pred = (0.4 * model_lgb.predict_proba(X_test)[:, 1] +
                 0.35 * model_xgb.predict_proba(X_test)[:, 1] +
                 0.25 * model_cat.predict_proba(X_test)[:, 1])
    test_preds += test_pred / kf.n_splits

    # Evaluate
    fold_auc = roc_auc_score(y_valid, val_pred)
    print(f"ğŸ�¯ Fold {fold+1} AUC: {fold_auc:.5f}")

    # Collect feature importances
    fold_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model_lgb.feature_importances_,
        "fold": fold + 1
    })
    feature_importances = pd.concat([feature_importances, fold_importance], axis=0)


cv_score = roc_auc_score(y, oof_preds)
print(f"\nğŸ”¥ Overall CV AUC: {cv_score:.5f}")

# Plot feature importance
feature_importances["importance"] = feature_importances["importance"].astype(float)
feat_mean = (
    feature_importances.groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 6))
sns.barplot(x=feat_mean.values, y=feat_mean.index, palette="viridis")
plt.title("Top 20 Important Features (LightGBM)")
plt.xlabel("Mean Feature Importance")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()


sample_submission[target] = test_preds
sample_submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv created successfully! Ready to submit ğŸš€")


# âœ… Save final predictions for submission
submission = pd.DataFrame({
    "id": test[id_col],
    "loan_paid_back": test_preds  # use your actual target name here
})
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("âœ… Submission file created:", submission.shape)
submission.head()


sample_submission[target] = test_preds
sample_submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv created successfully! Ready to submit ğŸš€")

