import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
pd.set_option('display.max_columns', 100)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
numeric_df = df.select_dtypes(include=["number"])
numeric_df["ldl_hdl_ratio"] = numeric_df["ldl_cholesterol"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["cholesterol_ratio"] = numeric_df["cholesterol_total"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["activity_bmi"] = numeric_df["physical_activity_minutes_per_week"] / (numeric_df["bmi"] + 1e-6)
numeric_df["age_bmi"] = numeric_df["age"] * numeric_df["bmi"]
numeric_df["age_activity"] = numeric_df["age"] * numeric_df["physical_activity_minutes_per_week"]
numeric_df["age_triglycerides"] = numeric_df["age"] * numeric_df["triglycerides"]
numeric_df["high_bmi"] = (numeric_df["bmi"] > 30).astype(int)
numeric_df["high_triglycerides"] = (numeric_df["triglycerides"] > 150).astype(int)
numeric_df.head()


features = [
    "family_history_diabetes",
    "physical_activity_minutes_per_week",
    "activity_bmi",
    "age_bmi",
    "age_triglycerides",
    "age_activity",
    "age",
    "ldl_hdl_ratio",
    "triglycerides",
    "cholesterol_ratio",
    "bmi"
]

X = numeric_df[features]
y = numeric_df["diagnosed_diabetes"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state = 42
    ))
])
pipe.fit(X_train, y_train)


y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]


metrics = {
    "AUC": roc_auc_score(y_test, y_proba),
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred)
}

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": pipe.named_steps["logreg"].coef_[0]
}).sort_values(by="coefficient", ascending=False)

coef_df


lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
y_proba = lgbm.predict_proba(X_test)[:, 1]


metrics = {
    "AUC": roc_auc_score(y_test, y_proba),
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred)
}
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


feat_importance = pd.DataFrame({
    "feature": features,
    "importance": lgbm.feature_importances_
}).sort_values(by="importance", ascending=False)

feat_importance


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-6)
df["cholesterol_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-6)
df["activity_bmi"] = df["physical_activity_minutes_per_week"] / (df["bmi"] + 1e-6)
df["age_bmi"] = df["age"] * df["bmi"]
df["age_activity"] = df["age"] * df["physical_activity_minutes_per_week"]
df["age_triglycerides"] = df["age"] * df["triglycerides"]
df["high_bmi"] = (df["bmi"] > 30).astype(int)
df["high_triglycerides"] = (df["triglycerides"] > 150).astype(int)


df.head()


cat_features = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]


X_cat = df[cat_features]
y = df["diagnosed_diabetes"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cat,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


cat_model_cat_only = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_state=42
)
cat_model_cat_only.fit(
    Xc_train,
    yc_train,
    cat_features=cat_features
)


yc_pred = cat_model_cat_only.predict(Xc_test)
yc_proba = cat_model_cat_only.predict_proba(Xc_test)[:, 1]

metrics_cat_only = {
    "AUC": roc_auc_score(yc_test, yc_proba),
    "Precision": precision_score(yc_test, yc_pred),
    "Recall": recall_score(yc_test, yc_pred),
    "F1": f1_score(yc_test, yc_pred)
}

metrics_cat_only


cat_importance = pd.DataFrame({
    "feature": cat_features,
    "importance": cat_model_cat_only.get_feature_importance()
}).sort_values(by="importance", ascending=False)

cat_importance


X_full = df[cat_features + features]
y = df["diagnosed_diabetes"]
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


cat_model_full = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_state=42
)
cat_feature_indices = [X_full.columns.get_loc(col) for col in cat_features]

cat_model_full.fit(
    Xf_train,
    yf_train,
    cat_features=cat_feature_indices
)





yf_pred = cat_model_full.predict(Xf_test)
yf_proba = cat_model_full.predict_proba(Xf_test)[:, 1]

metrics_cat_full = {
    "AUC": roc_auc_score(yf_test, yf_proba),
    "Precision": precision_score(yf_test, yf_pred),
    "Recall": recall_score(yf_test, yf_pred),
    "F1": f1_score(yf_test, yf_pred)
}

metrics_cat_full


full_importance = pd.DataFrame({
    "feature": X_full.columns,
    "importance": cat_model_full.get_feature_importance()
}).sort_values(by="importance", ascending=False)

full_importance




