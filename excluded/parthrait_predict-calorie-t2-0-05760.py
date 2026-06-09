import pandas as pd 
import numpy as np 
import os 
import time
import logging 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from category_encoders import TargetEncoder

from tqdm.auto import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

X = train.drop(columns=["id", "Calories"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])


FOLDS = 3
FEATURES = X.columns.tolist()
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof = np.zeros(len(train))
pred = np.zeros(len(test))

# Cross-validation loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    x_test = X_test.copy()
    
    start = time.time()
    
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.01,
        depth=8,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=25,
        verbose=100
    )
    
    model.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        use_best_model=True
    )
    
    # Predictions
    oof[valid_idx] = model.predict(x_valid)
    pred += model.predict(x_test)
    
    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Training time: {time.time() - start:.1f} sec")

# Average predictions
pred /= FOLDS

# Final RMSE on log scale
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE (log): {full_rmse:.4f}")


submission = test[["id"]].copy()
submission["Calories"] = np.expm1(pred)
submission.to_csv("submission_catboost.csv", index=False)
submission.head()

