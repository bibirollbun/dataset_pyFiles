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


import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
from pprint import pprint
from itertools import combinations
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


# ç‰¹å¾�å·¥ç¨‹å‡½æ•°
def feature_engineering(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    
    df = df.copy()

    # # æ€§åˆ«ç¼–ç �
    # le = LabelEncoder()
    # df["Sex"] = le.fit_transform(df["Sex"])  # male=1, female=0
    # æ€§åˆ«ç¼–ç �
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    # df["Age_Group"] = pd.cut(df["Age"], bins=[19, 30, 45, 60, 80], labels=["20-30", "31-45", "46-60", "61-80"])
    # df["Age_Group"] = LabelEncoder().fit_transform(df["Age_Group"].astype(str))

    # # BMI ç‰¹å¾�
    # df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

    # æ•°å€¼ç‰¹å¾�äº¤å�‰é¡¹ï¼ˆ2é˜¶å’Œ3é˜¶ï¼‰+ log1p ç¼©æ”¾
    cross_features = [ "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

    for f1, f2 in combinations(cross_features, 2):
        name = f"{f1}_x_{f2}"
        df[name] = np.log1p(df[f1] * df[f2])

    for f1, f2 in combinations(cross_features, 2):
        name = f"{f1}_y_{f2}"
        df[name] = np.log1p(df[f1] / df[f2])

    # for f1, f2, f3 in combinations(cross_features, 3):
    #     name = f"{f1}_x_{f2}_x_{f3}"
    #     df[name] = np.log1p(df[f1] * df[f2] * df[f3])

    # å¯¹æ•°ç›®æ ‡å€¼
    if is_train and "Calories" in df.columns:
        df["log_Calories"] = np.log1p(df["Calories"])

    return df


# åŠ è½½æ•°æ�®
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

train = feature_engineering(train, is_train=True)
test = feature_engineering(test, is_train=False)


import seaborn as sns
import matplotlib.pyplot as plt

# 1. æ��å�–ç»„å�ˆç‰¹å¾�åˆ—å��
combo_cols = [col for col in train.columns if ("_x_" in col or "_y_" in col)]

# 2. è®¡ç®—ç›¸å…³ç³»æ•°çŸ©é˜µ
combo_corr = train[combo_cols].corr()

# 3. å�¯è§†åŒ–ç›¸å…³æ€§çƒ­åŠ›å›¾
plt.figure(figsize=(12, 10))
sns.heatmap(combo_corr, annot=False, cmap="coolwarm", center=0, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap of Combination Features")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# è®¡ç®—ç»„å�ˆç‰¹å¾�ä¸�ç›®æ ‡ log_Calories çš„ç›¸å…³ç³»æ•°
target_corr = train[combo_cols + ["log_Calories"]].corr()["log_Calories"].drop("log_Calories").sort_values(ascending=False)

# å�¯è§†åŒ–
plt.figure(figsize=(8, 6))
sns.barplot(x=target_corr.values, y=target_corr.index, palette="coolwarm")
plt.title("Correlation of Combination Features with log_Calories")
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Feature")
plt.grid(True, axis="x")
plt.tight_layout()
plt.show()



target = "log_Calories"
drop_cols = ['id', 'Calories', 'log_Calories']  # æ³¨æ„�æ›¿æ�¢ log_Calories

features = [col for col in train.columns if col not in drop_cols]

X = train[features]
y = train[target]
X_test = test[features]


from catboost import CatBoostRegressor, Pool

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 5),
        "random_seed": 42,
        "verbose": 0,
        "loss_function": "RMSE",
        "bootstrap_type": "Bernoulli"
    }

    rmsle_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, valid_idx) in enumerate(tqdm(kf.split(X), desc=f"Trial {trial.number}", leave=False)):
        X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=Pool(X_val, y_val),
                  early_stopping_rounds=50)

        preds = np.maximum(0, model.predict(X_val))
        score = np.sqrt(mean_squared_log_error(y_val, preds))
        rmsle_scores.append(score)

    trial_score = np.mean(rmsle_scores)
    print(f"âœ… Trial {trial.number} RMSLE: {trial_score:.5f}")
    return trial_score


# # XGboost
# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
#         "random_state": 42,
#         "tree_method": "hist"
#     }

#     rmsle_scores = []
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)

#     # æ˜¾ç¤º Trial å†…æŠ˜æ•°è¿›åº¦æ�¡
#     for fold_idx, (train_idx, valid_idx) in enumerate(tqdm(kf.split(X), desc=f"Trial {trial.number}", leave=False)):
#         X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

#         model = XGBRegressor(**params)
#         model.fit(X_train, y_train,
#                   eval_set=[(X_val, y_val)],
#                   early_stopping_rounds=50,
#                   verbose=False)
#         preds = np.maximum(0, model.predict(X_val))
#         score = np.sqrt(mean_squared_log_error(y_val, preds))
#         rmsle_scores.append(score)

#     trial_score = np.mean(rmsle_scores)
#     print(f"âœ… Trial {trial.number} RMSLE: {trial_score:.5f}")
#     return trial_score


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")

N_TRIALS = 30
for trial_num in range(N_TRIALS):
    print(f"\nğŸš€ Starting Trial {trial_num}/{N_TRIALS}")
    study.optimize(objective, n_trials=1, catch=(Exception,))  # æ¯�æ¬¡å�ªè¿�è¡Œ1ä¸ªtrial

# è¾“å‡ºæœ€ä¼˜ç»“æ�œ
print(f"\nâœ… Best RMSLE (5-Fold): {study.best_value:.5f}")
print("âœ… Best Parameters:")
for k, v in study.best_params.items():
    print(f"   {k}: {v}")


best_model = CatBoostRegressor(**study.best_params)
# best_model = XGBRegressor(**study.best_params)
best_model.fit(X, y)


log_preds = best_model.predict(X_test)
calories_preds = np.expm1(log_preds)         # è¿˜å�Ÿ
calories_preds = np.maximum(0, calories_preds)  # æˆªæ–­è´Ÿçƒ­é‡�

submission['Calories'] = calories_preds
submission.to_csv("submission.csv", index=False)
print("ğŸ�¯ submission.csv å·²ä¿�å­˜ï¼ŒåŒ…å�«é¢„æµ‹çš„ Calories åˆ—")

