pip_dir = "/kaggle/input/tabpfn-2-0-1-offline/"
!pip install --no-index --find-links=$pip_dir tabpfn 


import os
import torch
import warnings
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set a seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

TARGET = 'HOMELESS_RATE'

# Determine the device to use (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

OFFLINE_MODEL_PATH = "/kaggle/input/tabpfn-2-0-1-offline/tabpfn-v2-regressor.ckpt"


train_file = "/kaggle/input/california-homelessness-prediction-challenge/train.csv"
test_file = "/kaggle/input/california-homelessness-prediction-challenge/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
submission_df = pd.read_csv("/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv")

features = [col for col in train_df.columns if col != TARGET and col != 'ID']

X = train_df[features].values
y = train_df[TARGET].values
X_test = test_df[features].values


submission_df


# IDをアルファベットと、_ 数字に分けます

train_df_id_split = train_df['ID'].str.split('_', expand=True)
train_df['STATE'] = train_df_id_split[0]
train_df.drop('ID', axis = 1, inplace = True)
train_df


test_df_id_split = test_df['ID'].str.split('_', expand=True)
test_df['STATE'] = test_df_id_split[0]
test_df.drop('ID', axis = 1, inplace =True)
test_df


%%time

test_predictions = []
oof_predictions = np.zeros(len(X))

n_splits = 10

gkf = GroupKFold(n_splits=n_splits)

print(f"Starting {n_splits}-fold cross-validation...")

for fold, (train_index, val_index) in enumerate(gkf.split(X, y, groups=train_df['STATE'])):

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = TabPFNRegressor(device=device, ignore_pretraining_limits=True, model_path=OFFLINE_MODEL_PATH)
    model.fit(X_train,y_train)

    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds
    mse = mean_squared_error(y_val, val_preds)

    print(f"MSE score on validation set for Fold {fold+1}: {mse}")

    test_preds = model.predict(X_test)
    test_predictions.append(test_preds)


# oof_df = pd.DataFrame({'ID': train_df['ID'], TARGET: oof_predictions})
# oof_df.to_csv('oof.csv', index=False)

submission_df[TARGET] = np.mean(test_predictions, axis=0)

submission_df.to_csv('submission.csv', index=False)
submission_df.head()

















