import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train.head()


test.head()


train.info()


test.info()


# Basic cleaning we already know from EDA
train.drop(columns=["id"], inplace=True)
# test.drop(columns=["id"], inplace=True)


train["diagnosed_diabetes"] = train["diagnosed_diabetes"].astype(bool)


# convert categorical columns
cat_cols = [
    "gender", "ethnicity", "education_level", 
    "income_level", "smoking_status", "employment_status"
]


for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


def add_interaction_features(df):
    df["chol_ratio"] = df["cholesterol_total"] / df["hdl_cholesterol"]
    df["bp_ratio"] = df["systolic_bp"] / df["diastolic_bp"]
    df["bmi_age"] = df["bmi"] * df["age"]
    df["activity_screen_ratio"] = df["physical_activity_minutes_per_week"] / (df["screen_time_hours_per_day"] + 1)
    return df


train = add_interaction_features(train)
test = add_interaction_features(test)


train.head()


test.head()


log_cols = ["triglycerides", "cholesterol_total"]

for col in log_cols:
    train[col + "_log"] = np.log1p(train[col])
    test[col + "_log"] = np.log1p(test[col])



num_cols = train.select_dtypes(include=["int64", "float64", "bool"]).columns.drop("diagnosed_diabetes")
cat_cols = train.select_dtypes(include=["category"]).columns



preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])



X = train.drop(columns=["diagnosed_diabetes"])
y = train["diagnosed_diabetes"].astype(int)


X_test = test.copy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)


rf_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", rf_model)
])


rf_pipeline.fit(X_train, y_train)



rf_pred = rf_pipeline.predict_proba(X_val)[:, 1]

rf_auc = roc_auc_score(y_val, rf_pred)
print("Random Forest AUC:", rf_auc)


xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)


xgb_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", xgb_model)
])


xgb_pipeline.fit(X_train, y_train)

xgb_pred = xgb_pipeline.predict_proba(X_val)[:, 1]

xgb_auc = roc_auc_score(y_val, xgb_pred)
print("XGBoost Validation AUC:", xgb_auc)


print("Random Forest AUC:", rf_auc)
print("XGBoost AUC:", xgb_auc)

best_model = "XGBoost" if xgb_auc > rf_auc else "Random Forest"
print("Best Model:", best_model)


if xgb_auc > rf_auc:
    final_model = xgb_pipeline
else:
    final_model = rf_pipeline

final_test_pred = final_model.predict_proba(X_test)[:, 1]


test_id = test["id"].copy()



submission = pd.DataFrame({
    "id": test_id,
    "diagnosed_diabetes": final_test_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()




