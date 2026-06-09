# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install lightgbm --quiet



# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model and evaluation
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Warnings off
import warnings
warnings.filterwarnings("ignore")



KAGGLE_PATH = "/kaggle/input/amex-default-prediction"

train_data_path = f"{KAGGLE_PATH}/train_data.csv"
test_data_path = f"{KAGGLE_PATH}/test_data.csv"
labels_path = f"{KAGGLE_PATH}/train_labels.csv"



sample = pd.read_csv(train_data_path, nrows=5000)
numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = ['customer_ID'] + [col for col in numeric_cols if col != 'customer_ID']



# Preview raw dataset
sample_df = pd.read_csv(train_data_path, nrows=100_000)

# View column names and data types
print("Dataset Info:")
print(sample_df.info())

# Show first 5 rows
print("\n Sample Rows:")
display(sample_df.head())

# Check for null values
print("\n Missing Values (Top 20):")
print(sample_df.isnull().sum().sort_values(ascending=False).head(20))

# Quick stats for numerical columns
print("\n Summary Statistics:")
display(sample_df.describe())



agg_funcs = ['mean', 'last']
chunk_size = 500_000  # optionally go smaller if needed
train_storage = {}

reader = pd.read_csv(train_data_path, chunksize=chunk_size, usecols=numeric_cols)

for i, chunk in enumerate(reader):
    print(f"Processing train chunk {i+1}")
    grouped = chunk.groupby("customer_ID").agg(agg_funcs)
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    grouped.reset_index(inplace=True)

    for _, row in grouped.iterrows():
        cust_id = row['customer_ID']
        values = row.drop('customer_ID').values
        train_storage.setdefault(cust_id, []).append(values)



# Get column names once (from any one stored value)
first_customer = next(iter(train_storage.values()))
num_features = len(first_customer[0])
column_names = [f"feat_{i}" for i in range(num_features)]  # generic naming

# Pre-allocate arrays
all_ids = []
all_means = np.empty((len(train_storage), num_features))

for idx, (cust_id, chunks) in enumerate(train_storage.items()):
    all_ids.append(cust_id)
    stacked = np.vstack(chunks)
    all_means[idx, :] = stacked.mean(axis=0)

# Final DataFrame
train_agg = pd.DataFrame(all_means, columns=column_names)
train_agg.insert(0, "customer_ID", all_ids)



labels = pd.read_csv(labels_path)
train = train_agg.merge(labels, on="customer_ID")
train.drop(columns="customer_ID", inplace=True)
train.fillna(train.median(), inplace=True)

X = train.drop(columns="target")
y = train["target"]



test_storage = {}

reader = pd.read_csv(test_data_path, chunksize=chunk_size, usecols=numeric_cols)
for i, chunk in enumerate(reader):
    print(f"Test Chunk {i+1}")
    grouped = chunk.groupby("customer_ID").agg(agg_funcs)
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    grouped.reset_index(inplace=True)

    for _, row in grouped.iterrows():
        cust_id = row['customer_ID']
        values = row.drop('customer_ID').values
        test_storage.setdefault(cust_id, []).append(values)



test_ids = []
test_means = np.empty((len(test_storage), num_features))

for idx, (cust_id, chunks) in enumerate(test_storage.items()):
    test_ids.append(cust_id)
    stacked = np.vstack(chunks)
    test_means[idx, :] = stacked.mean(axis=0)

X_test = pd.DataFrame(test_means, columns=column_names)
X_test.fillna(train.median(), inplace=True)



params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "seed": 42,
    "verbose": -1
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"LightGBM Fold {fold+1}")
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
    dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

    model_lgb = lgb.train(
    params,
    dtrain,
    valid_sets=[dtrain, dval],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)



    oof_preds[val_idx] = model_lgb.predict(X.iloc[val_idx])
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits


print(f"LightGBM CV AUC: {roc_auc_score(y, oof_preds):.5f}")



!pip install xgboost shap --quiet


import xgboost as xgb
xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"XGBoost Fold {fold+1}")
    dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
    dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
    dtest = xgb.DMatrix(X_test)

    model_xgb = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "seed": 42
        },
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    xgb_oof[val_idx] = model_xgb.predict(dval)
    xgb_preds += model_xgb.predict(dtest) / kf.n_splits

print(f"XGBoost CV AUC: {roc_auc_score(y, xgb_oof):.5f}")



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Scale features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

log_oof = np.zeros(len(X))
log_preds = np.zeros(len(X_test))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"LogReg Fold {fold+1}")
    model_log = LogisticRegression(max_iter=1000)
    model_log.fit(X_scaled[train_idx], y.iloc[train_idx])

    log_oof[val_idx] = model_log.predict_proba(X_scaled[val_idx])[:, 1]
    log_preds += model_log.predict_proba(X_test_scaled)[:, 1] / kf.n_splits

log_auc = roc_auc_score(y, log_oof)
print(f"Logistic Regression CV AUC: {log_auc:.5f}")



from sklearn.ensemble import RandomForestClassifier

rf_oof = np.zeros(len(X))
rf_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Random Forest Fold {fold+1}")
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model_rf.fit(X.iloc[train_idx], y.iloc[train_idx])

    rf_oof[val_idx] = model_rf.predict_proba(X.iloc[val_idx])[:, 1]
    rf_preds += model_rf.predict_proba(X_test)[:, 1] / kf.n_splits

rf_auc = roc_auc_score(y, rf_oof)
print(f"Random Forest CV AUC: {rf_auc:.5f}")



!pip install shap --quiet


# SHAP
import shap
import matplotlib.pyplot as plt
import pandas as pd

# LightGBM SHAP Explanation
print("SHAP for LightGBM")
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(X.iloc[:500])  # SHAP is expensive — sample 500
shap.summary_plot(shap_values, X.iloc[:500])

# Logistic Regression Coefficients
print("Logistic Regression Coefficients")
coefs = pd.Series(model_log.coef_[0], index=X.columns)
coefs.sort_values().tail(20).plot(kind='barh', figsize=(8, 6), title='Top 20 Positive Coefficients (LogReg)')
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

# Random Forest Feature Importances
print("Random Forest Feature Importances")
rf_importance = pd.Series(model_rf.feature_importances_, index=X.columns)
rf_importance.sort_values().tail(20).plot(kind='barh', figsize=(8, 6), title='Top 20 Feature Importances (RF)')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()



submission = pd.DataFrame({
    "customer_ID": test_ids,
    "prediction": lgb_preds  
})
submission.to_csv("submission.csv", index=False)
print(" submission.csv saved.")



submission.to_csv("/kaggle/working/submission.csv", index=False)

