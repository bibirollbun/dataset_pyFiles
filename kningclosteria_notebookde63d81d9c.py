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
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import itertools
import joblib


def generate_interactions(
    df, 
    num_cols=None, 
    cat_cols=None, 
    mode="all", 
    include_3way=True, 
    poly_degree=True, 
    extra_stats=True, 
    cross_freq=True,
    cat_threshold=20
):
    """
    å…¨é�¢ç‰ˆç‰¹å¾�äº¤äº’ç”Ÿæˆ�å™¨
    å�‚æ•°:
        df: DataFrame
        num_cols: æ•°å€¼å�‹ç‰¹å¾�åˆ—è¡¨
        cat_cols: ç±»åˆ«å�‹ç‰¹å¾�åˆ—è¡¨
        mode: "all" | "num" | "cat" | "mix"
        include_3way: æ˜¯å�¦ç”Ÿæˆ�ä¸‰é˜¶äº¤äº’
        poly_degree: æ˜¯å�¦ç”Ÿæˆ�å¹‚æ¬¡ç‰¹å¾� (å¹³æ–¹ã€�ç«‹æ–¹ã€�logã€�sqrt)
        extra_stats: æ˜¯å�¦ç”Ÿæˆ�é¢�å¤–ç»Ÿè®¡é‡� (sum, skew, quantile)
        cross_freq: æ˜¯å�¦ç”Ÿæˆ�ç±»åˆ«äº¤å�‰é¢‘ç�‡/å”¯ä¸€å€¼æ•°
        cat_threshold: ç±»åˆ«åŸºæ•°é˜ˆå€¼ï¼Œé�¿å…�é«˜åŸºæ•°ç±»åˆ«çˆ†ç‚¸
    è¿”å›�:
        df_new: å¸¦äº¤äº’ç‰¹å¾�çš„æ–° DataFrame
    """
    df_new = df.copy()
    stats = ["mean", "std", "min", "max", "median"]

    # ---------------- æ•°å€¼ç‰¹å¾� ----------------
    if num_cols and mode in ["all", "num"]:
        for c1, c2 in itertools.combinations(num_cols, 2):
            df_new[f"{c1}_plus_{c2}"] = df_new[c1] + df_new[c2]
            df_new[f"{c1}_minus_{c2}"] = df_new[c1] - df_new[c2]
            df_new[f"{c1}_x_{c2}"] = df_new[c1] * df_new[c2]
            df_new[f"{c1}_div_{c2}"] = np.where(df_new[c2]==0, 0, df_new[c1]/df_new[c2])
            df_new[f"{c2}_div_{c1}"] = np.where(df_new[c1]==0, 0, df_new[c2]/df_new[c1])

        # å¹‚æ¬¡ç‰¹å¾�
        if poly_degree:
            for c in num_cols:
                df_new[f"{c}_squared"] = df_new[c] ** 2
                df_new[f"{c}_cubed"] = df_new[c] ** 3
                df_new[f"{c}_log1p"] = np.log1p(df_new[c].clip(lower=0))
                df_new[f"{c}_sqrt"] = np.sqrt(df_new[c].clip(lower=0))

    # ---------------- ç±»åˆ«ç‰¹å¾� ----------------
    if cat_cols and mode in ["all", "cat"]:
        for c1, c2 in itertools.combinations(cat_cols, 2):
            df_new[f"{c1}_{c2}"] = df_new[c1].astype(str) + "_" + df_new[c2].astype(str)

        if cross_freq:
            for c1, c2 in itertools.combinations(cat_cols, 2):
                df_new[f"{c1}_{c2}_freq"] = df_new.groupby([c1, c2])[c1].transform("count")
                df_new[f"{c1}_{c2}_nunique"] = df_new.groupby(c1)[c2].transform("nunique")

    # ---------------- ç±»åˆ« Ã— æ•°å€¼ ----------------
    if num_cols and cat_cols and mode in ["all", "mix"]:
        for c in cat_cols:
            if df_new[c].nunique() <= cat_threshold:
                for n in num_cols:
                    grouped = df_new.groupby(c)[n]
                    for stat in stats:
                        df_new[f"{c}_{n}_{stat}"] = grouped.transform(stat)
                    if extra_stats:
                        df_new[f"{c}_{n}_sum"] = grouped.transform("sum")
                        df_new[f"{c}_{n}_skew"] = grouped.transform("skew")
                        df_new[f"{c}_{n}_q25"] = grouped.transform(lambda x: x.quantile(0.25))
                        df_new[f"{c}_{n}_q75"] = grouped.transform(lambda x: x.quantile(0.75))

    # ---------------- ä¸‰é˜¶äº¤äº’ ----------------
    if include_3way:
        if num_cols:
            for c1, c2, c3 in itertools.combinations(num_cols, 3):
                df_new[f"{c1}_x_{c2}_x_{c3}"] = df_new[c1] * df_new[c2] * df_new[c3]
        if cat_cols:
            for c1, c2, c3 in itertools.combinations(cat_cols, 3):
                df_new[f"{c1}_{c2}_{c3}"] = (
                    df_new[c1].astype(str) + "_" + df_new[c2].astype(str) + "_" + df_new[c3].astype(str)
                )
        # ç±»åˆ« Ã— ç±»åˆ« Ã— æ•°å€¼
        for c1, c2 in itertools.combinations(cat_cols, 2):
            if num_cols:
                for n in num_cols:
                    df_new[f"{c1}_{c2}_{n}_mean"] = df_new.groupby([c1, c2])[n].transform("mean")

    return df_new


# è¯»å�–æ•°æ�®
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
selected_features = joblib.load("/kaggle/input/select/selected_features.pkl")
target = "accident_risk"

full = pd.concat([train.drop(columns=[target,"id"]), test.drop(columns=["id"])], axis=0)


num_cols = full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = full.select_dtypes(include="object").columns.tolist()
# ç”Ÿæˆ�äº¤äº’ç‰¹å¾�
full = generate_interactions(full, num_cols=num_cols, cat_cols=cat_cols, mode="all", 
    include_3way=True, 
    poly_degree=True, 
    extra_stats=True, 
    cross_freq=True)

cat_cols = full.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    full[col] = full[col].astype("category")


X = full.iloc[:len(train)][selected_features]
X_test = full.iloc[len(train):][selected_features]
y = train[target]


kf = KFold(n_splits=5, shuffle=True, random_state=42)


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        
        # ğŸ”‘ æ–°å¢�æ­£åˆ™åŒ–å�‚æ•°
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        
        # æ��å�‡æ•ˆç�‡
        "force_row_wise": True,
        "verbosity": -1
    }

    rmse_list = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[
                      early_stopping(stopping_rounds=20),
                      log_evaluation(period=0)  # è®¾ç½®ä¸º0è¡¨ç¤ºä¸�è¾“å‡ºæ—¥å¿—
                  ])
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_list.append(rmse)

    return np.mean(rmse_list)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)


best_params = study.best_params
print("Best RMSE:", study.best_value)


final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_estimators=1000)
final_model.fit(X, y)


#explainer = shap.TreeExplainer(final_model)
#shap_values = explainer.shap_values(X)  
#shap.summary_plot(shap_values, X)


preds = final_model.predict(X_test)
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": preds
})
submission.to_csv("submission.csv", index=False)


lgb.plot_importance(final_model, max_num_features=20)

