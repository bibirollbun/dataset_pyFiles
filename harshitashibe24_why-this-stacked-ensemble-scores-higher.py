!pip install catboost xgboost lightgbm --quiet


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

TARGET = "diagnosed_diabetes"

X = train.drop(columns=[TARGET])
y = train[TARGET]

test_ids = test["id"]


print(X.columns.tolist())


train["BMI_Age"] = train["bmi"] * train["age"]
test["BMI_Age"]  = test["bmi"] * test["age"]

train["Glucose_BMI"] = train["cholesterol_total"] / (train["bmi"] + 1)
test["Glucose_BMI"]  = test["cholesterol_total"] / (test["bmi"] + 1)

for col in ["cholesterol_total", "triglycerides"]:
    train[f"log_{col}"] = np.log1p(train[col])
    test[f"log_{col}"]  = np.log1p(test[col])




X = train.drop(columns=[TARGET])
y = train[TARGET]


X.columns


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(5,4))
sns.countplot(x=y)
plt.title("Target Distribution (Diagnosed Diabetes)")
plt.xlabel("Diagnosed Diabetes")
plt.ylabel("Count")
plt.show()



# Select numeric columns safely
num_df = X.select_dtypes(include=[np.number])

plt.figure(figsize=(10,6))
sns.heatmap(
    num_df.corr().iloc[:6, :6],  # limit size for readability
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()



print(X.columns)



cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
imp = SimpleImputer(strategy="median")

X[cat_cols] = enc.fit_transform(X[cat_cols])
test[cat_cols] = enc.transform(test[cat_cols])

X[num_cols] = imp.fit_transform(X[num_cols])
test[num_cols] = imp.transform(test[num_cols])


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

pred_lgb = np.zeros(len(test))
pred_cat = np.zeros(len(test))
pred_xgb = np.zeros(len(test))


for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== FOLD {fold+1} / 5 =====")

    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=64,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        class_weight="balanced"
    )
    lgb.fit(X_train, y_train)

    oof_lgb[val_idx] = lgb.predict_proba(X_valid)[:, 1]
    pred_lgb += lgb.predict_proba(test)[:, 1] / kf.n_splits

    # CatBoost
    cat = CatBoostClassifier(
    iterations=1200,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False)
    cat.fit(X_train, y_train)
    oof_cat[val_idx] = cat.predict_proba(X_valid)[:, 1]
    pred_cat += cat.predict_proba(test)[:, 1] / kf.n_splits

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        tree_method="hist"
    )
    xgb.fit(X_train, y_train)

    oof_xgb[val_idx] = xgb.predict_proba(X_valid)[:, 1]
    pred_xgb += xgb.predict_proba(test)[:, 1] / kf.n_splits



oof_blend = 0.4 * oof_lgb + 0.35 * oof_cat + 0.25 * oof_xgb
pred_blend = 0.4 * pred_lgb + 0.35 * pred_cat + 0.25 * pred_xgb


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y, oof_blend)

plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, label="Blended Model ROC")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Blended Ensemble")
plt.legend()
plt.show()



print("\nLightGBM ROC:", roc_auc_score(y, oof_lgb))
print("CatBoost ROC:", roc_auc_score(y, oof_cat))
print("XGBoost ROC:", roc_auc_score(y, oof_xgb))
print("Blended ROC:", roc_auc_score(y, oof_blend))


model_scores = {
    "LightGBM": roc_auc_score(y, oof_lgb),
    "CatBoost": roc_auc_score(y, oof_cat),
    "XGBoost": roc_auc_score(y, oof_xgb),
    "Blended Ensemble": roc_auc_score(y, oof_blend)
}

pd.DataFrame.from_dict(
    model_scores,
    orient="index",
    columns=["ROC-AUC"]
).sort_values("ROC-AUC", ascending=False)



stack_train = np.vstack([oof_lgb, oof_cat, oof_xgb]).T
stack_test  = np.vstack([pred_lgb, pred_cat, pred_xgb]).T

lvl2 = LogisticRegression(max_iter=2000)
lvl2.fit(stack_train, y)

pred_final = lvl2.predict_proba(stack_test)[:, 1]

print("\nFinal Stacked ROC:",
      roc_auc_score(y, lvl2.predict_proba(stack_train)[:, 1]))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Feature Importance (LightGBM)
importance = lgb.feature_importances_
idx = np.argsort(importance)[::-1]

sns.barplot(x=importance[idx][:15], y=X.columns[idx][:15])
plt.title("Top 15 Features — LightGBM Importance")
plt.show()



submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": pred_final
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved successfully!")

