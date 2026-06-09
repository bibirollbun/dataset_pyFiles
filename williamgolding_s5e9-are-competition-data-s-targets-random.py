# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical functions
from scipy.stats import skew
from scipy.stats import ttest_rel
from scipy.signal import find_peaks

# Machine learning preprocessing and modeling
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from xgboost import XGBRegressor

# Load data
orig = pd.read_csv('../input/playground-series-s5e9/train.csv', index_col='id')
TARGET = orig.columns[-1]
FEATURES = [f for f in orig.columns if f != TARGET]

scores = []
for repeat in range(101):
    train = orig.copy()
    if repeat > 0:
        # shuffle target for random baseline
        t = orig[TARGET].values.copy()  # copy to avoid in-place reuse
        np.random.shuffle(t)
        train[TARGET] = t

    oof_preds = np.zeros(len(train))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        train_X = train.iloc[train_idx][FEATURES]
        train_y = train.iloc[train_idx][TARGET]
        valid_X = train.iloc[val_idx][FEATURES]
        valid_y = train.iloc[val_idx][TARGET]

        # stronger model with stochasticity
        model = XGBRegressor(
            n_estimators=10_000,           # allow many trees
            learning_rate=0.3,           # smaller learning rate
            max_depth=6,
            subsample=0.8,                # add randomness
            colsample_bytree=0.8,         # add randomness
            reg_alpha=2.0,                # L1 regularization
            min_child_weight=10,          # regularization
            tree_method="gpu_hist",       # GPU training
            predictor="gpu_predictor",
            eval_metric="rmse",
            random_state=42,
            verbosity=0
        )

        model.fit(
            train_X, train_y,
            eval_set=[(valid_X, valid_y)],
            early_stopping_rounds=100,    # stop when no improvement
            verbose=0
        )

        oof_preds[val_idx] = model.predict(valid_X)

    # compute OOF CV score
    m = mean_squared_error(train[TARGET], oof_preds, squared=False)  # RMSE
    if repeat == 0:
        print(f"When using original target CV RMSE = {m:.2f}")
    elif repeat == 1:
        print(f"When using random target CV RMSE = {m:.2f}\nAdditional random trials... ", end="")
    else:
        print(f"{repeat-1}, ", end="")
    scores.append(m)

# z-score calculation
s = np.std(scores[1:])
m = np.mean(scores[1:])
z = (scores[0] - m) / s

print(f"\nStandard dev={s}")
print(f"Mean={m}")
print(f"Baseline rmse={scores[0]}")
print(f"\nZ-score = {z:.2f}")




