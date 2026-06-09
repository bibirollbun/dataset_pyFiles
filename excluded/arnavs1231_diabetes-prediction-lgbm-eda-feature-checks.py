import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print("Train shape:", train_df.shape)
print("Test shape: ", test_df.shape)
print("Submission shape:", sample_sub.shape)

train_df.head()



train_df.info()



target_col = "diagnosed_diabetes"

train_df[target_col].value_counts(normalize=True)



missing = train_df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)



X = train_df.drop(columns=[target_col])
y = train_df[target_col]

cat_features = X.select_dtypes(include=["object"]).columns.tolist()
num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical features:", cat_features)
print("Numerical features:", num_features)



from sklearn.model_selection import train_test_split

# Drop ID
X = train_df.drop(columns=["diagnosed_diabetes", "id"])
y = train_df["diagnosed_diabetes"]

# Identify categorical features again (after drop)
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Categorical features:", cat_features)



# from catboost import CatBoostClassifier

# model = CatBoostClassifier(
#     iterations=500,
#     learning_rate=0.1,
#     depth=8,
#     loss_function="Logloss",
#     eval_metric="AUC",
#     random_seed=42,
#     verbose=100
# )

# model.fit(
#     X_train, y_train,
#     cat_features=cat_features,
#     eval_set=(X_val, y_val),
#     use_best_model=True
# )



# from sklearn.metrics import roc_auc_score

# val_preds = model.predict_proba(X_val)[:, 1]
# roc_auc = roc_auc_score(y_val, val_preds)

# print("Validation ROC-AUC:", roc_auc)



from sklearn.preprocessing import LabelEncoder

X = train_df.drop(columns=["diagnosed_diabetes", "id"])
y = train_df["diagnosed_diabetes"]

# Identify categorical columns
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Label encode categorical features
for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train-validation split (same as before)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Categorical features:", cat_features)



import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import lightgbm as lgb

lgb_train = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=cat_features,
    free_raw_data=False
)

lgb_val = lgb.Dataset(
    X_val,
    label=y_val,
    categorical_feature=cat_features,
    free_raw_data=False
)



import lightgbm as lgb

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)



from sklearn.metrics import roc_auc_score

val_preds = lgb_model.predict(X_val)
roc_auc = roc_auc_score(y_val, val_preds)

print("LightGBM Validation ROC-AUC:", roc_auc)



import pandas as pd

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": lgb_model.feature_importance()
}).sort_values(by="importance", ascending=False)

feature_importance.head(15)



LOW_IMPORTANCE_FEATURES = [
    "alcohol_consumption_per_week", "waist_to_hip_ratio", "sleep_hours_per_day", "diastolic_bp", "hdl_cholesterol"
]

# Drop features
X_reduced = X.drop(columns=LOW_IMPORTANCE_FEATURES)

X_train_red, X_val_red, y_train_red, y_val_red = train_test_split(
    X_reduced, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

lgb_train_red = lgb.Dataset(
    X_train_red,
    label=y_train_red,
    categorical_feature=[f for f in cat_features if f not in LOW_IMPORTANCE_FEATURES],
    free_raw_data=False
)

lgb_val_red = lgb.Dataset(
    X_val_red,
    label=y_val_red,
    categorical_feature=[f for f in cat_features if f not in LOW_IMPORTANCE_FEATURES],
    free_raw_data=False
)

lgb_model_red = lgb.train(
    params,
    lgb_train_red,
    num_boost_round=1000,
    valid_sets=[lgb_train_red, lgb_val_red],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)



from sklearn.metrics import roc_auc_score

val_preds_red = lgb_model_red.predict(X_val_red)
roc_auc_red = roc_auc_score(y_val_red, val_preds_red)

print("Reduced Feature LightGBM ROC-AUC:", roc_auc_red)



# Load test data
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# Save IDs
test_ids = test_df["id"]

# Drop ID
test_X = test_df.drop(columns=["id"])

# IMPORTANT: force categorical columns to string (FIXES ERROR)
for col in cat_features:
    X[col] = X[col].astype(str)
    test_X[col] = test_X[col].astype(str)

# Encode categorical features SAFELY
from sklearn.preprocessing import LabelEncoder

for col in cat_features:
    le = LabelEncoder()
    le.fit(pd.concat([X[col], test_X[col]], axis=0))
    test_X[col] = le.transform(test_X[col])

# Predict
test_preds = lgb_model.predict(test_X)

# Create submission
submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_preds
})

# Save file
submission.to_csv("submission.csv", index=False)

submission.head()



# Save submission to your Downloads folder
submission_path = r"C:\Users\KIIT\Downloads\submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved at: {submission_path}")


