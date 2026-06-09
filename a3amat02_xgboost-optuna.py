import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import optuna

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error

df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")

df.drop("id", axis=1, inplace=True)

def process_date(x):
    date = x.split('-')
    return [int(date[1]), int(date[2])]

df["month"] = df["date"].apply(lambda x: process_date(x)[0])
df["day"] = df["date"].apply(lambda x: process_date(x)[1])

df.drop("date", axis=1, inplace=True)


encoders = []

for i in ["country", "store", "product"]:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].values)
    encoders += [le]


not_na_df = df[df["num_sold"].notna()]

na_df = df[df["num_sold"].isna()]

x = not_na_df.drop("num_sold", axis=1).values
y = not_na_df["num_sold"].values

xgbr = XGBRegressor()

xgbr.fit(x, y)

x_na = na_df.drop("num_sold", axis=1).values

xgbr.predict(x_na)

na_pred = xgbr.predict(x_na)

df.loc[df["num_sold"].isna(), "num_sold"] = na_pred


x = df.drop("num_sold", axis=1).values
y = df["num_sold"].values


import xgboost as xgb

def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        # use exact for small dataset.
        "tree_method": "gpu_hist",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "device": "cuda",
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    # accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    # accuracy = r2_score(preds, valid_y)
    accuracy = mean_absolute_percentage_error(preds, valid_y)
    return accuracy

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, timeout=600)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


params2 = {"booster": "gbtree",
    "lambda": 0.002614889239799295,
    "alpha": 0.09110188563034623,
    "subsample": 0.5590201067263905,
    "colsample_bytree": 0.9025949221420628,
    "max_depth": 9,
    "min_child_weight": 8,
    "eta": 0.35754583238995574,
    "gamma": 2.0847388619015053e-07,
    "grow_policy": "lossguide",
    "device": "cuda"}

model = XGBRegressor(**params2)
model.fit(x, y)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


test_df["month"] = test_df["date"].apply(lambda x: process_date(x)[0])
test_df["day"] = test_df["date"].apply(lambda x: process_date(x)[1])


test_df.drop("date", inplace=True, axis=1)


for i, j in enumerate(["country", "store", "product"]):
    test_df[j] = encoders[i].transform(test_df[j].values)


test_x = test_df.iloc[:, 1:].values

res = model.predict(test_x)


res = [int(round(i, 0)) for i in res]


test_df["num_sold"] = res

sub = test_df[["id", "num_sold"]]

sub.to_csv("submission.csv", index=False)

