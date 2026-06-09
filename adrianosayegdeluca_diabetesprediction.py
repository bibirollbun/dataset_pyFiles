import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os


train_path = ""
test_path = ""

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        if "train.csv" in filename:
            train_path = path
        elif "test.csv" in filename:
            test_path = path

df = pd.read_csv(train_path, index_col='id')
test_df = pd.read_csv(test_path, index_col='id')


display(df.head())


df.columns


def procesar_datos(df):
    df = df.copy()

    # FISIOLÓGICOS
    df["non_hdl"] = df["cholesterol_total"] - df["hdl_cholesterol"]
    df["friedewald_ldl"] = df["non_hdl"] - (df["triglycerides"] / 5)
    df["ldl_math_error"] = df["ldl_cholesterol"] - df["friedewald_ldl"]
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["bmi_age"] = df["bmi"] * df["age"]
    df["fat_index"] = df["bmi"] * df["waist_to_hip_ratio"]

    # ESTILO DE VIDA
    df["activity_per_bmi"] = df["physical_activity_minutes_per_week"] / (df["bmi"] + 1)
    df["diet_bmi"] = df["diet_score"] * df["bmi"]
    df["alcohol_activity"] = df["alcohol_consumption_per_week"] / (df["physical_activity_minutes_per_week"] + 1)

    # INTERACCIONES
    df["sleep_activity"] = df["sleep_hours_per_day"] * df["physical_activity_minutes_per_week"]
    df["diet_activity"] = df["diet_score"] * df["physical_activity_minutes_per_week"]
    df["screen_sleep"] = df["screen_time_hours_per_day"] * df["sleep_hours_per_day"]
    df["bmi_sleep"] = df["bmi"] * df["sleep_hours_per_day"]
    df["alcohol_bmi"] = df["alcohol_consumption_per_week"] * df["bmi"]

    # TRANSFORMACIONES
    df["log_bmi"] = np.log1p(df["bmi"])
    df["log_alcohol"] = np.log1p(df["alcohol_consumption_per_week"])
    df["sqrt_screen"] = np.sqrt(df["screen_time_hours_per_day"])
    df["log_triglycerides"] = np.log1p(df["triglycerides"])

    # RATIOS CLÍNICOS
    df["chol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1)
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1)
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
    df["tg_chol_ratio"] = df["triglycerides"] / (df["cholesterol_total"] + 1)

    return df



X = procesar_datos(df.drop(columns=["diagnosed_diabetes"]))
y = df["diagnosed_diabetes"]

X_test = procesar_datos(test_df)



cat_cols = [
    "gender", "ethnicity", "education_level",
    "income_level", "smoking_status", "employment_status"
]



from sklearn.preprocessing import LabelEncoder
import pandas as pd

for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([X[col], X_test[col]]).astype(str))
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))



import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cb  = np.zeros(len(X))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cb  = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        tree_method="hist",
        random_state=42 + fold
    )
    model_xgb.fit(X_tr, y_tr)
    oof_xgb[va] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS

    #  LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42 + fold
    )
    model_lgb.fit(X_tr, y_tr)
    oof_lgb[va] = model_lgb.predict_proba(X_va)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_SPLITS

    # CatBoost
    model_cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        verbose=False,
        random_seed=42 + fold
    )
    model_cb.fit(X_tr, y_tr)
    oof_cb[va] = model_cb.predict_proba(X_va)[:, 1]
    test_cb += model_cb.predict_proba(X_test)[:, 1] / N_SPLITS



X_stack = np.column_stack([oof_xgb, oof_lgb, oof_cb])
X_stack_test = np.column_stack([test_xgb, test_lgb, test_cb])

meta = LogisticRegression()
meta.fit(X_stack, y)

final_preds = meta.predict_proba(X_stack_test)[:, 1]



from sklearn.metrics import roc_auc_score

stack_auc = roc_auc_score(y, meta.predict_proba(X_stack)[:, 1])
print(f"Stacking ROC-AUC (CV): {stack_auc:.4f}")
print("XGB AUC:", roc_auc_score(y, oof_xgb))
print("LGB AUC:", roc_auc_score(y, oof_lgb))
print("CB  AUC:", roc_auc_score(y, oof_cb))



submission = pd.DataFrame({
    "id": test_df.index,
    "diagnosed_diabetes": final_preds
})

submission.to_csv("submission.csv", index=False)





