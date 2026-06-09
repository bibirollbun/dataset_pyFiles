import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from lightgbm import plot_importance
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import os



parent_dir = "/kaggle/input/halan-hcdr-eda/home-credit-default-risk-working_data"

df_train = pd.read_csv(os.path.join(parent_dir, "application_train.csv"))
df_test = pd.read_csv(os.path.join(parent_dir, "application_test.csv"))

# Clean column names to remove special JSON characters
df_train.columns = df_train.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
df_test.columns = df_test.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)



min_max_to_drop = []

for col in df_train.columns:
    if 'min' in col or 'max' in col:
        min_max_to_drop.append(col)

df_train = df_train.drop(columns=min_max_to_drop)
df_test = df_test.drop(columns=min_max_to_drop)


df_train.shape


# Features & Target
X = df_train.drop(columns=["TARGET", "SK_ID_CURR"], axis=1)
y = df_train["TARGET"]
X_test = df_test.drop("SK_ID_CURR", axis=1)  # drop ID column for prediction

# KFold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Placeholders
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))  # Probabilities, not class labels

# Train KFold
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        nthread=-1,
        n_estimators=5000,
        learning_rate=0.01,
        objective='binary',
        random_state=42,
        verbose=-1,
        ## To address overfit as much as can
        max_depth=11,
        num_leaves=58,
        colsample_bytree=0.613,
        subsample=0.708,
        max_bin=407,
        reg_alpha=3.564,
        reg_lambda=4.930,
        min_child_weight=6,
        min_child_samples=165
    )
    
    model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=500),
        log_evaluation(period=500)
        ],
    eval_metric='auc'
    )
    
    # Predict probabilities
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred_proba
    
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

# Evaluate ROC AUC
auc_score = roc_auc_score(y, oof_preds)
print(f"CV ROC AUC: {auc_score:.5f}")

# Prepare submission (round probs if TARGET is binary class)
submission = pd.DataFrame({
    "SK_ID_CURR": df_test["SK_ID_CURR"].astype(int),
    "TARGET": test_preds
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


plot_importance(model, figsize=(16, 32), max_num_features=100)

