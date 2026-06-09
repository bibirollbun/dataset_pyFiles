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


# import libraries
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)


# read the data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
origin = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')


# drop 'id' from train and test
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# concat train and origin
train = pd.concat([train, origin], axis=0, ignore_index=True)
train.info()


# data distribution
features = [col for col in train.columns if train[col].dtype in ['float64', 'int64']]
num_features = len(features)
n_cols = 5
n_rows = (num_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_cols, n_rows, figsize=(20, 6*n_rows))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(train[feature], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}", fontsize=12)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")

for j in range(i + 1, len(features)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# correlation heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(train.corr(), annot=True, fmt='0.2f')
plt.show()


# model params
xgb_reg = XGBRegressor(
        tree_method="gpu_hist",           # Use GPU-accelerated histogram algorithm
        predictor="gpu_predictor",        # Use GPU for predictions
        device="cuda",
        max_depth=16,
        colsample_bytree=0.75,
        subsample=0.9,
        n_estimators=5000,
        learning_rate=0.01,
        gamma=0.01,
        max_delta_step=2,
        # early_stopping_rounds=100,
        eval_metric="rmse",
)


# Prep for model
X = train.drop('BeatsPerMinute', axis=1)
y = train['BeatsPerMinute']
X_test = test.copy()

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_rmse = []

# Kfold setup
n_splits = 20
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"{'=' * 5} Fold {fold + 1} {'=' * 5}")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb_reg.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=0
    )

    # OOF predictions
    val_preds = xgb_reg.predict(X_val)
    oof_preds[val_idx] = val_preds

    # test predictions
    test_preds += xgb_reg.predict(X_test) / n_splits
    
    # Fold RMSE
    rmse = np.sqrt(((y_val - val_preds) ** 2).mean())
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    fold_rmse.append(rmse)


# evaluation
oof_rmse = np.sqrt(((y - oof_preds) ** 2).mean())
print("\nCV mean RMSE:", np.mean(fold_rmse))
print("OOF RMSE:", oof_rmse)


# Save OOF predictions
oof_df = pd.DataFrame({
    "oof_pred": oof_preds,
    "target": y
})
oof_df.to_csv("oof_predictions_xgb.csv", index=False)


# submission
submission['BeatsPerMinute'] = test_preds
submission.to_csv("submission.csv", index=False)
submission.head()




