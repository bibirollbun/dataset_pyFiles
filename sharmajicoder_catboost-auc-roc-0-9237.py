import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


df_train.head()


df_train.drop(columns="id", inplace=True)


df_test_full = df_test.copy()


X = df_train.drop(columns="loan_paid_back")
Y = df_train["loan_paid_back"]


categorical_cols = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]

continuous_cols = [
    "annual_income", "debt_to_income_ratio",
    "credit_score", "loan_amount", "interest_rate"
]


plt.figure(figsize = (25, 8), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train[continuous_cols],
    vert = True,
    notch = True,
    # label = continuous_cols
    
)
plt.title("Boxplot of continuous_cols", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("x-axis", fontsize = 20, fontweight = "bold", color = "black")
plt.ylabel("y-axis", fontsize = 20, fontweight = "bold", color = "black")
plt.grid(True, alpha = 0.5, linewidth = 0.5)
plt.legend(loc = "upper right", shadow = True, fontsize = 12)
plt.tight_layout()
plt.show()


for i in range(4):
    for col in continuous_cols:
        # lower, upper  = X[col].quantile(0.2), X[col].quantile(0.8)
        # X[col] = X[col].clip(lower, upper)
        # df_test[col] = df_test[col].clip(lower, upper)
        Q1 = np.percentile(df_train[col], 25)
        Q3 = np.percentile(df_train[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_train.loc[(df_train[col] < lower_bound) | (df_train[col] > upper_bound), col] = df_train[col].median()


plt.figure(figsize = (25, 8), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train[continuous_cols],
    vert = True,
    notch = True,
    # label = continuous_cols
)
plt.title("Boxplot of continuous_cols", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("x-axis", fontsize = 20, fontweight = "bold", color = "black")
plt.ylabel("y-axis", fontsize = 20, fontweight = "bold", color = "black")
plt.grid(True, alpha = 0.5, linewidth = 0.5)
plt.legend(loc = "upper right", shadow = True, fontsize = 12)
plt.tight_layout()
plt.show()


for col in categorical_cols:
    X[col] = X[col].astype(str)
    df_test[col] = df_test[col].astype(str)


df_test = df_test[X.columns]

cat_features = [X.columns.get_loc(col) for col in categorical_cols]

# -------------------------------------------------------------------------------------------
#                                 Model Parameters (GPU)
# -------------------------------------------------------------------------------------------
params = {
    "iterations": 1000,
    "depth": 5,
    "learning_rate": 0.22775461488,
    "l2_leaf_reg": 7.46314929,
    "bagging_temperature": 0.0350283198,
    "border_count": 128,
    "random_strength": 1.59045421e-05,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": False,
    "task_type": "GPU",          # Enable GPU
    "devices": "0,1"               # Use both GPU devices
}

cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

auc_scores = []
all_preds1 = []
oof_preds = np.zeros(len(Y))

# ------------------------------------------------------------------------------------------
#                                    Cross-validation with GPU
# ------------------------------------------------------------------------------------------

for fold, (train_idx, val_idx) in enumerate(cv.split(X, Y), 1):
    print(f"\nðŸš€ Training Fold {fold} on GPU...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    train_pool = Pool(X_train, Y_train, cat_features=cat_features)
    val_pool = Pool(X_val, Y_val, cat_features=cat_features)

    model = CatBoostClassifier(**params, scale_pos_weight = len(Y[Y == 0]) / len(Y[Y == 1]))
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    Y_pred_proba = model.predict_proba(val_pool)[:, 1]
    auc = roc_auc_score(Y_val, Y_pred_proba)
    auc_scores.append(auc)
    oof_preds[val_idx] = Y_pred_proba

    # Predict on test data (GPU)
    test_pool = Pool(df_test, cat_features=cat_features)
    test_proba = model.predict_proba(test_pool)[:, 1]
    all_preds1.append(test_proba)

    print(f"Fold {fold} AUC: {auc:.5f}")

# --------------------------------------------------------------------------------------------
#                                           Results
# --------------------------------------------------------------------------------------------
print(f"\nâœ… Mean AUC-ROC across folds: {np.mean(auc_scores):.5f}")



auc_scores = []
oof_preds = np.zeros(len(X))
all_preds2 = []

df_test = df_test[X.columns]

cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

# Convert categorical features properly
for col in categorical_cols:
    X[col] = X[col].astype("category")
    df_test[col] = df_test[col].astype("category")

# Compute scale_pos_weight manually (for imbalance)
pos_weight = len(Y[Y == 0]) / len(Y[Y == 1])

xgb_params = {
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.12775461488,
    "reg_lambda": 7.46314929,
    "subsample": 0.97,
    "colsample_bytree": 0.8,
    "gamma": 1.59045421e-05,
    "max_bin": 128,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "enable_categorical": True,
    "scale_pos_weight": pos_weight
}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, Y), 1):
    print(f"\nðŸš€ Training Fold {fold} on GPU...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    # Build model
    model = XGBClassifier(**xgb_params)

    # Train
    model.fit(
        X_train,
        Y_train,
        eval_set=[(X_val, Y_val)],
        verbose=False
    )

    # Predict fold probabilities
    Y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(Y_val, Y_pred_proba)
    auc_scores.append(auc)

    # Save OOF predictions
    oof_preds[val_idx] = Y_pred_proba

    # Predict test set
    test_proba = model.predict_proba(df_test)[:, 1]
    all_preds2.append(test_proba)

    print(f"Fold {fold} AUC: {auc:.5f}")

print(f"\nâœ… Mean AUC-ROC across folds: {np.mean(auc_scores):.5f}")


preds = (np.mean(all_preds1, axis = 0) + np.mean(all_preds2, axis = 0)) / 2
submission = pd.DataFrame({
    "id": df_test_full["id"],
    "load_paid_back": preds
})
submission.to_csv("submission.csv", index = False)
submission.to_csv("/kaggle/working/submission.csv", index = False)

