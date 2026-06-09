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


# ============================================================
# Predicting Road Accident Risk - Kaggle Playground S5E10
# Author: Data Enthusiast
# ============================================================

# ========== Setup ==========
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn and LGBM warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error

# ========== Load Data ==========
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ========== Quick EDA ==========
print("\nTrain columns:\n", train.columns)
print("\nTarget variable distribution:")
sns.histplot(train['accident_risk'], kde=True, bins=30)
plt.title("Distribution of Accident Risk")
plt.show()

# Check for missing values
missing = train.isnull().sum()
if missing.sum() > 0:
    print("\nMissing Values:\n", missing[missing > 0])
else:
    print("\nNo missing values found.")

# ========== Preprocessing ==========
cat_cols = train.select_dtypes(include=['object', 'category']).columns
num_cols = train.select_dtypes(exclude=['object', 'category']).drop(['id', 'accident_risk'], axis=1).columns

# Label encode categorical columns
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X = train.drop(columns=['id', 'accident_risk'])
y = train['accident_risk']
X_test = test.drop(columns=['id'])

# ========== Model Training (LightGBM) ==========
params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42,
    'objective': 'regression',
    'n_jobs': -1
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n========== Fold {fold+1} ==========")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[early_stopping(200), log_evaluation(200)]
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOverall CV RMSE: {cv_rmse:.5f}")

# ========== Feature Importance ==========
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importances.head(20))
plt.title('Top 20 Feature Importances (LightGBM)')
plt.show()

# ========== Predictions and Submission ==========
test_preds = np.clip(test_preds, 0, 1)
submission = pd.DataFrame({'id': test['id'], 'accident_risk': test_preds})
submission.to_csv("submission.csv", index=False)

print("\nSubmission file saved as submission.csv")
submission.head()




