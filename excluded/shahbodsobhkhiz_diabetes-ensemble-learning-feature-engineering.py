import numpy as np 
import pandas as pd 
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from category_encoders import CatBoostEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns 
from sklearn.metrics import confusion_matrix
from lightgbm.callback import early_stopping


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.head()


print(train.shape)
print(test.shape)


test.head( )


train.describe()


train = train.sample(frac=1, random_state=42).reset_index(drop=True)


train.info()


train.columns


X = train.drop(['id','diagnosed_diabetes'],axis = 1)
y = train['diagnosed_diabetes']


ordinal_cols = ["education_level", "income_level"]
ordinal_categories = [
    ["No formal", "Highschool", "Graduate", "Postgraduate"], 
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
]

nominal_cols = ["gender", "ethnicity", "smoking_status", "employment_status"]

binary_cols = ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]

numerical_cols = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides'
]


categorical_cols = ["gender", "ethnicity", "education_level", "income_level",
                    "smoking_status", "employment_status",
                    "family_history_diabetes", "hypertension_history", 
                    "cardiovascular_history"]


train['diagnosed_diabetes'].value_counts()


num_df = train[numerical_cols]
num_df_with_target = num_df.copy()
num_df_with_target['diagnosed_diabetes'] = y
corr_matrix = num_df_with_target.corr(method='pearson')
plt.figure(figsize=(10,8))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Matrix: Numerical Features + Target")
plt.show()


train_featured = train.copy()

eps = 1e-6


def apply_feature_engineering(df, eps=1e-6):
    df_fe = df.copy()

    # Blood pressure
    df_fe["pulse_pressure"] = df_fe["systolic_bp"] - df_fe["diastolic_bp"]
    df_fe["mean_arterial_pressure"] = (
        df_fe["diastolic_bp"] + (df_fe["systolic_bp"] - df_fe["diastolic_bp"]) / 3
    )

    # Cholesterol ratios
    df_fe["chol_hdl_ratio"] = df_fe["cholesterol_total"] / (df_fe["hdl_cholesterol"] + eps)
    df_fe["ldl_hdl_ratio"] = df_fe["ldl_cholesterol"] / (df_fe["hdl_cholesterol"] + eps)
    df_fe["trig_hdl_ratio"] = df_fe["triglycerides"] / (df_fe["hdl_cholesterol"] + eps)

    # Obesity + age
    df_fe["bmi_age"] = df_fe["bmi"] * df_fe["age"]
    df_fe["waist_bmi"] = df_fe["waist_to_hip_ratio"] * df_fe["bmi"]

    # Lifestyle
    df_fe["alcohol_age"] = df_fe["alcohol_consumption_per_week"] * df_fe["age"]

    # Drop raw features
    df_fe.drop(columns=[
        "systolic_bp", "diastolic_bp",
        "cholesterol_total", "hdl_cholesterol",
        "ldl_cholesterol", "triglycerides",
        "bmi", "waist_to_hip_ratio",
        "age", "alcohol_consumption_per_week"
    ], inplace=True)

    return df_fe



train_featured = apply_feature_engineering(train)


print("Original shape:", train.shape)
print("Featured shape:", train_featured.shape)

train_featured.head()


ordinal_cols = ["education_level", "income_level"]

ordinal_categories = [
    ["No formal", "Highschool", "Graduate", "Postgraduate"], 
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
]
nominal_cols = [
    "gender",
    "ethnicity",
    "smoking_status",
    "employment_status"
]
binary_cols = [
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history"
]
numerical_cols = [
    # Existing numeric features (kept)
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "heart_rate",

    # Engineered features — blood pressure
    "pulse_pressure",
    "mean_arterial_pressure",

    # Engineered features — cholesterol ratios
    "chol_hdl_ratio",
    "ldl_hdl_ratio",
    "trig_hdl_ratio",

    # Engineered features — obesity & age
    "bmi_age",
    "waist_bmi",

    # Engineered features — lifestyle
    "alcohol_age",
]
categorical_cols = (
    nominal_cols
    + ordinal_cols
    + binary_cols
)


print(sorted(train_featured.columns))



num_df = train_featured[numerical_cols]

# 2) Add target
num_df_with_target = num_df.copy()
num_df_with_target["diagnosed_diabetes"] = y  # y = target series

# 3) Compute correlation matrix
corr_matrix = num_df_with_target.corr(method="pearson")

# 4) Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix: Engineered Numerical Features + Target")
plt.show()


X = train_featured.drop(['id','diagnosed_diabetes'],axis = 1)
y = train_featured['diagnosed_diabetes']


preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
        ("binary", "passthrough", binary_cols),
        ("numeric", StandardScaler(), numerical_cols)
    ]
)


X_processed = preprocessor.fit_transform(X)

print("Final input shape:", X_processed.shape)


# 1. Ordinal feature names (same as original)
ordinal_feature_names = ordinal_cols

# 2. One-hot feature names (expanded)
ohe = preprocessor.named_transformers_["onehot"]
ohe_feature_names = ohe.get_feature_names_out(nominal_cols)

# 3. Binary feature names
binary_feature_names = binary_cols

# 4. Numerical feature names
numerical_feature_names = numerical_cols

# 5. Combine all in final order
final_feature_names = (
    ordinal_feature_names 
    + ohe_feature_names.tolist()
    + binary_feature_names
    + numerical_feature_names
)
df = pd.DataFrame(X_processed, columns=final_feature_names)


df.head()


df.columns


encoded_categorical_cols = (
    ordinal_feature_names
    + ohe_feature_names.tolist()
    + binary_feature_names
)

encoded_numeric_cols = numerical_feature_names



df[encoded_numeric_cols].hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()



X_raw = train_featured.drop(columns=["id", "diagnosed_diabetes"])
y = train_featured["diagnosed_diabetes"]



X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
    X_raw,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


X_train_proc = preprocessor.fit_transform(X_train_raw)
X_valid_proc = preprocessor.transform(X_valid_raw)


cat_model = CatBoostClassifier(
    iterations=20000,
    depth=4,
    learning_rate=0.01,
    loss_function="Logloss",
    eval_metric="AUC",

    task_type="GPU",
    devices="0",
    bootstrap_type="Bernoulli",
    subsample=0.8,
    early_stopping_rounds=200,
    verbose=200
)

cat_model.fit(
    X_train_raw,
    y_train,
    eval_set=(X_valid_raw, y_valid),
    cat_features=categorical_cols
)


xgb_model = XGBClassifier(
    n_estimators=20000,
    learning_rate=0.05,
    max_depth=4,

    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",
    eval_metric="auc",

    tree_method="gpu_hist",
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_valid_proc, y_valid)],
    early_stopping_rounds=200,
    verbose=100
)


# CatBoost
cat_valid_probs = cat_model.predict_proba(X_valid_raw)[:, 1]
cat_valid_preds = (cat_valid_probs >= 0.5).astype(int)

# XGBoost
xgb_valid_probs = xgb_model.predict_proba(X_valid_proc)[:, 1]
xgb_valid_preds = (xgb_valid_probs >= 0.5).astype(int)



def print_confusion(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f"\n{name} Confusion Matrix")
    print(f"TP (True Positive): {tp}")
    print(f"TN (True Negative): {tn}")
    print(f"FP (False Positive): {fp}")
    print(f"FN (False Negative): {fn}")


print_confusion("CatBoost", y_valid, cat_valid_preds)
print_confusion("XGBoost", y_valid, xgb_valid_preds)


ensemble_valid_probs = (
    cat_valid_probs
    + xgb_valid_probs
) / 2

results = []

for t in np.arange(0.30, 0.70, 0.01):
    preds = (ensemble_valid_probs >= t).astype(int)
    
    acc = accuracy_score(y_valid, preds)
    rec = recall_score(y_valid, preds)
    f1  = f1_score(y_valid, preds)
    prec = precision_score(y_valid, preds)
    
    results.append((t, acc, rec, f1, prec))

df_thresh = pd.DataFrame(
    results,
    columns=["threshold", "accuracy", "recall", "f1", "precision"]
)

df_thresh.sort_values("accuracy", ascending=False).head(10)



best_row = df_thresh.sort_values("accuracy", ascending=False).iloc[0]
best_threshold = best_row["threshold"]

best_row


plt.figure(figsize=(8,5))
plt.plot(df_thresh["threshold"], df_thresh["accuracy"], label="Accuracy")
plt.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall")
plt.plot(df_thresh["threshold"], df_thresh["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Threshold tuning on ensemble")
plt.show()


test_featured = apply_feature_engineering(test)


test_featured.head()


X_test_raw = test_featured.drop(columns=["id"])
X_test_proc = preprocessor.transform(X_test_raw)


cat_test_probs = cat_model.predict_proba(X_test_raw)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_test_proc)[:, 1]


len(cat_test_probs) == len(xgb_test_probs) == len(test)


ensemble_test_probs = (
    cat_test_probs
    + xgb_test_probs
) / 2



submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": ensemble_test_probs
})

submission.to_csv("submission.csv", index=False)
submission.head()




