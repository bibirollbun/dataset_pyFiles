import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


X = train.drop(columns=["id", "diagnosed_diabetes"])
y = train["diagnosed_diabetes"]

X_test = test.drop(columns=["id"])


def add_feature_engineering(df):
    df = df.copy()

    # ---- Interaction Features ----
    df["bmi_age"] = df["bmi"] * df["age"]
    df["waist_bmi"] = df["waist_to_hip_ratio"] * df["bmi"]
    df["activity_bmi"] = df["physical_activity_minutes_per_week"] / (df["bmi"] + 1)
    df["screen_sleep"] = df["screen_time_hours_per_day"] / (df["sleep_hours_per_day"] + 0.1)
    df["alcohol_activity"] = df["alcohol_consumption_per_week"] / (df["physical_activity_minutes_per_week"] + 1)

    # ---- Binary Risk Flags ----
    df["is_obese"] = (df["bmi"] >= 30).astype(int)
    df["low_activity"] = (df["physical_activity_minutes_per_week"] < 150).astype(int)
    df["poor_sleep"] = (df["sleep_hours_per_day"] < 6).astype(int)
    df["high_screen"] = (df["screen_time_hours_per_day"] > 6).astype(int)

    # ---- Aggregate Risk Scores ----
    df["lifestyle_risk"] = (
        df["low_activity"] +
        df["poor_sleep"] +
        df["high_screen"] +
        (df["diet_score"] < 5).astype(int)
    )

    df["cardio_metabolic_risk"] = (
        df["is_obese"] +
        df["hypertension_history"] +
        df["cardiovascular_history"] +
        df["family_history_diabetes"]
    )

    return df



X = add_feature_engineering(X)
X_test = add_feature_engineering(X_test)



cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()



num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))



N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))



lgb_model = LGBMClassifier(
    n_estimators=900,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)



xgb_model = XGBClassifier(
    n_estimators=900,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    tree_method="hist"
)



cat_model = CatBoostClassifier(
    iterations=900,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_seed=42
)



for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n Fold {fold + 1}/{N_SPLITS}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ---- LightGBM ----
    lgb_model.fit(X_train, y_train)
    lgb_val = lgb_model.predict_proba(X_val)[:, 1]
    print("LGB AUC:", roc_auc_score(y_val, lgb_val))
    lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

    # ---- XGBoost ----
    xgb_model.fit(X_train, y_train)
    xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    print("XGB AUC:", roc_auc_score(y_val, xgb_val))
    xgb_preds += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

    # ---- CatBoost ----
    cat_model.fit(X_train, y_train)
    cat_val = cat_model.predict_proba(X_val)[:, 1]
    print("CAT AUC:", roc_auc_score(y_val, cat_val))
    cat_preds += cat_model.predict_proba(X_test)[:, 1] / N_SPLITS



final_preds = (
    0.4 * lgb_preds +
    0.35 * xgb_preds +
    0.25 * cat_preds
)



submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": lgb_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully")





