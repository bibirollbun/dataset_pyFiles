# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
train_df.head(5)


train_df.info()


def data_overview(df):
    summary = pd.DataFrame({
        "DataType": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "%Missing Value": (df.isnull().sum() / len(df)) * 100
    })

    return summary.reset_index().rename(columns={"index": "Features"})

data_overview(train_df)


num_cols = train_df.select_dtypes(include="number").columns
cat_cols = train_df.select_dtypes(exclude="number").columns

train_df[train_df.columns].hist(bins=30, figsize=(16, 12))
plt.tight_layout()
plt.show()


corr_matrix = train_df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()


!pip install pytorch-tabnet


# TabNet regressor setup (definition only)
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
except ImportError as exc:
    raise ImportError(
        "Install pytorch-tabnet with `pip install pytorch-tabnet` before running this cell."
    ) from exc

import torch
from sklearn.model_selection import train_test_split

df = train_df.copy()

target_col = "BeatsPerMinute"
feature_cols = [col for col in df.columns if col not in {target_col, "id"}]

X = df[feature_cols].values
y = df[target_col].values

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tabnet_params = dict(
    n_d=32,
    n_a=32,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    mask_type="entmax",
    scheduler_params={"step_size": 50, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=10,
)

tabnet = TabNetRegressor(**tabnet_params)

print(f"TabNet ready for training with {len(feature_cols)} features. Call `tabnet.fit` when you are ready to train.")



import time
from pytorch_tabnet.callbacks import Callback

class ProgressPrinter(Callback):
    def __init__(self):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.start
        train_loss = logs.get("loss")
        # valid 로그 구조는 버전에 따라 다르니 필요하면 print(logs)로 찍어서 맞춰주세요.
        valid_logs = logs.get("valid") or logs.get("validation") or {}
        if isinstance(valid_logs, dict):
            # eval_name=['valid']이면 {'rmse': 값} 형태로 들어오는 경우가 많습니다.
            valid_rmse = valid_logs.get("rmse")
        else:
            valid_rmse = None

        msg = f"[{elapsed:6.1f}s] epoch {epoch+1}"
        if train_loss is not None:
            msg += f" | train_loss: {train_loss:.5f}"
        if valid_rmse is not None:
            msg += f" | valid_rmse: {valid_rmse:.5f}"
        print(msg)

tabnet.fit(
    X_train=X_train,
    y_train=y_train.reshape(-1, 1),
    eval_set=[(X_valid, y_valid.reshape(-1, 1))],
    eval_name=['valid'],
    eval_metric=['rmse'],
    max_epochs=200,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    callbacks=[ProgressPrinter()],
)



X_test = test_df[feature_cols].values
test_pred = tabnet.predict(X_test).ravel()

submission = pd.DataFrame({"id": test_df["id"], "BeatsPerMinute": test_pred})
submission.to_csv("submission.csv", index=False)

