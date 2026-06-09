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


# # 1. ç‰¹å¾�å·¥ç¨‹å‡½æ•°
# def feature_engineering(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
#     df = df.copy()

#     # æ€§åˆ«ç¼–ç �
#     df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

#     # åŸºç¡€ç‰¹å¾�
#     df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
#     df["Age_Group"] = pd.cut(df["Age"], bins=[19, 30, 45, 60, 80], labels=["20-30", "31-45", "46-60", "61-80"])
#     df["Age_Group"] = LabelEncoder().fit_transform(df["Age_Group"].astype(str))

#     # å¼ºç›¸å…³äº¤äº’ç‰¹å¾�
#     df["Heart_Work"] = df["Heart_Rate"] * df["Duration"]
#     df["BMI_Duration"] = df["BMI"] * df["Duration"]
#     df["BMI_HeartRate"] = df["BMI"] * df["Heart_Rate"]
#     df["Height_Sex"] = df["Height"] * df["Sex"]

#     # æ€§åˆ«äº¤äº’ç‰¹å¾�ï¼ˆæ–°å¢�ï¼‰
#     df["Sex_BMI"] = df["Sex"] * df["BMI"]
#     df["Sex_Duration"] = df["Sex"] * df["Duration"]
#     df["Sex_HeartRate"] = df["Sex"] * df["Heart_Rate"]
#     df["Sex_HeartWork"] = df["Sex"] * df["Heart_Work"]
#     df["Sex_BMI_Duration"] = df["Sex"] * df["BMI_Duration"]

#     # è®­ç»ƒç‰¹æœ‰ç‰¹å¾�ï¼ˆç›®æ ‡/å�•ä½�æ¶ˆè€—ç�‡ï¼‰
#     if is_train and "Calories" in df.columns:
#         df["Cal_per_min"] = df["Calories"] / df["Duration"]

#     return df


# 1. ç‰¹å¾�å·¥ç¨‹å‡½æ•°
def feature_engineering(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    """
    ç‰¹å¾�å·¥ç¨‹ï¼šäºŒé˜¶å’Œä¸‰é˜¶ä¹˜ç§¯é¡¹ä½¿ç”¨ log1p ç¼©æ”¾
    """
    df = df.copy()

    # # æ€§åˆ«ç¼–ç �
    # le = LabelEncoder()
    # df["Sex"] = le.fit_transform(df["Sex"])  # male=1, female=0
    # æ€§åˆ«ç¼–ç �
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age_Group"] = pd.cut(df["Age"], bins=[19, 30, 45, 60, 80], labels=["20-30", "31-45", "46-60", "61-80"])
    df["Age_Group"] = LabelEncoder().fit_transform(df["Age_Group"].astype(str))

    # BMI å�Šå…¶ä¸šåŠ¡è¡�ç”Ÿç‰¹å¾�
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    # df["BMI_Duration"] = df["BMI"] * df["Duration"]
    # df["BMI_HeartRate"] = df["BMI"] * df["Heart_Rate"]
    # df["Sex_BMI"] = df["Sex"] * df["BMI"]
    # df["Sex_BMI_Duration"] = df["Sex"] * df["BMI_Duration"]

    # æ•°å€¼ç‰¹å¾�äº¤å�‰é¡¹ï¼ˆ2é˜¶å’Œ3é˜¶ï¼‰+ log1p ç¼©æ”¾
    cross_features = ["BMI", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

    for f1, f2 in combinations(cross_features, 2):
        name = f"{f1}_x_{f2}"
        df[name] = np.log1p(df[f1] * df[f2])

    for f1, f2 in combinations(cross_features, 2):
        name = f"{f1}_x_{f2}"
        df[name] = np.log1p(df[f1] / df[f2])

    # for f1, f2, f3 in combinations(cross_features, 3):
    #     name = f"{f1}_x_{f2}_x_{f3}"
    #     df[name] = np.log1p(df[f1] * df[f2] * df[f3])

    # å¯¹æ•°ç›®æ ‡å€¼
    if is_train and "Calories" in df.columns:
        df["log_Calories"] = np.log1p(df["Calories"])

    return df



# 2. åŠ è½½æ•°æ�®
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

train = feature_engineering(train, is_train=True)
test = feature_engineering(test, is_train=False)


# 3. æ¨¡å�‹è¾“å…¥
# target = "Calories"
# drop_cols = ['id', 'Calories', 'Cal_per_min']
# features = [col for col in train.columns if col not in drop_cols]

# X = train[features]
# y = train[target]
# X_test = test[features]
# 3. æ¨¡å�‹è¾“å…¥
target = "log_Calories"
drop_cols = ['id', 'Calories', 'log_Calories']  # æ³¨æ„�æ›¿æ�¢ log_Calories

features = [col for col in train.columns if col not in drop_cols]

X = train[features]
y = train[target]
X_test = test[features]


# 4. Optuna è¶…å�‚æ•°ä¼˜åŒ–
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "tree_method": "hist"
    }

    rmsle_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, valid_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50, verbose=False)
        preds = np.maximum(0, model.predict(X_val))
        score = np.sqrt(mean_squared_log_error(y_val, preds))
        rmsle_scores.append(score)

    return np.mean(rmsle_scores)

# å�¯åŠ¨ Optuna ä¼˜åŒ–å™¨ï¼ˆå¸¦è¿›åº¦æ�¡ï¼‰
# è‡ªå®šä¹‰æ‰“å�°æ¯�æ¬¡ trial çš„ç»“æ�œ
def print_trial_result(study, trial):
    print(f"Trial {trial.number}: RMSLE={trial.value:.5f}, Params={trial.params}")

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, callbacks=[print_trial_result])


# è¾“å‡ºç»“æ�œ
print(f"âœ… Best RMSLE (5-Fold): {study.best_value:.5f}")
print("âœ… Best Parameters:")
for k, v in study.best_params.items():
    print(f"   {k}: {v}")

# # å�¯è§†åŒ–è°ƒå�‚è¿‡ç¨‹
# plot_optimization_history(study).show()
# plot_parallel_coordinate(study).show()


# 5. ç”¨æœ€ä¼˜å�‚æ•°å…¨é‡�è®­ç»ƒæ¨¡å�‹
best_model = XGBRegressor(**study.best_params)
best_model.fit(X, y)


# 6. å¯¹ test.csv è¿›è¡Œé¢„æµ‹
log_preds = best_model.predict(X_test)
calories_preds = np.expm1(log_preds)         # è¿˜å�Ÿ
calories_preds = np.maximum(0, calories_preds)  # æˆªæ–­è´Ÿçƒ­é‡�


# 7. ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
submission['Calories'] = calories_preds
submission.to_csv("submission.csv", index=False)
print("ğŸ�¯ submission.csv å·²ä¿�å­˜ï¼ŒåŒ…å�«é¢„æµ‹çš„ Calories åˆ—")

