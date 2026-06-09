# Load requires libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


# Load datasets

train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

train_data.head()


# # Check if we need to handle missing values
# missing_values_count = train_data.isnull().sum()
# filtered_missing = missing_values_count[missing_values_count > 0]
# filtered_missing.head()

# # Fill missing values
# train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
# train_data[train_data.select_dtypes(include=['object', 'category']).columns] = train_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))

# test_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
# test_data[train_data.select_dtypes(include=['object', 'category']).columns] = train_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))


# Transform numeric columns

scaler = StandardScaler()
num_cols = list(train_data.select_dtypes(exclude=['object']).columns)

train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])


# Object datatype columns encoding

labelEncoder = LabelEncoder()
cat_cols = list(train_data.select_dtypes(include=['object']).columns.difference(['Personality']))

for col_name in cat_cols:
    train_data[col_name]=labelEncoder.fit_transform(train_data[col_name]).astype(int)
    test_data[col_name]=labelEncoder.transform(test_data[col_name]).astype(int)

train_data['Personality_encoded'] = labelEncoder.fit_transform(train_data['Personality'])


# Prepare data

X = train_data.drop(columns=["id", "Personality", "Personality_encoded"])
y = train_data["Personality_encoded"]
X_test = test_data.drop(columns=["id"])

combined = pd.concat([X, X_test], axis=0)

X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)


# Correlation matrix
correlation_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


# XGBoost model configuration

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=10, verbose_eval=False)
    
    oof_preds[val_idx] = model.predict(dval) > 0.5
    test_preds += model.predict(dtest) / skf.n_splits


# Find result
cv_acc = accuracy_score(y, oof_preds)
print(f"Cross-Validation Accuracy: {cv_acc:.4f}")

final_preds = (test_preds > 0.5).astype(int)
submission["Personality"] = labelEncoder.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
submission.head()

