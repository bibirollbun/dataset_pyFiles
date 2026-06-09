import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score)


from sklearn.model_selection import KFold

import xgboost as xgb
from xgboost import XGBRegressor
import catboost as catb
from catboost import CatBoostRegressor

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


RANDOM_STATE = 42


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None) 
pd.set_option("display.width", None)  
pd.set_option("display.max_colwidth", None)  


def add_features(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
        
    df["bid_ask_spread_proxy"] = df["ask_qty"] - df["bid_qty"]
    df["total_liquidity"] = df["bid_qty"] + df["ask_qty"]
    df["trade_imbalance"] = df["buy_qty"] - df["sell_qty"]
    df["total_trades"] = df["buy_qty"] + df["sell_qty"]
    
    df["volume_per_trade"] = df["volume"] / (df["buy_qty"] + df["sell_qty"] + 1e-8)
    df["buy_volume_ratio"] = df["buy_qty"] / (df["volume"] + 1e-8)
    df["sell_volume_ratio"] = df["sell_qty"] / (df["volume"] + 1e-8)
    
    df["buying_pressure"] = df["buy_qty"] / (df["buy_qty"] + df["sell_qty"] + 1e-8)
    df["selling_pressure"] = df["sell_qty"] / (df["buy_qty"] + df["sell_qty"] + 1e-8)

    return df


def add_ext_large_lags(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    features = df.columns
    lags = [100, 150, 200]

    for col in features:
        for lag in lags:
            name_col_lag = f"lag_{lag}_{col}"
            
            df[name_col_lag] = df[col].shift(lag).astype(np.float32)
            df[name_col_lag].fillna(df[col].mean(), inplace=True)
        
    return df


def cross_val_kf(model:xgb.XGBRegressor,
                 X_train: np.array,
                 y_train: np.array,
                 X_test: np.array=None,
                 n_splits: int=5,
                 return_oof_test=False):
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    model = model
    scores = defaultdict(list)
    
    if (not X_test is None) and return_oof_test:
        oof_test = np.zeros(len(X_test))
    
    for i, (tr_idx, vl_idx) in enumerate(kf.split(X_train, y_train), 1):
        
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
        if isinstance(model, xgb.XGBRegressor):
            model.fit(X_tr,
                    y_tr,
                    eval_set=[(X_vl, y_vl)],
                    verbose=0)
            
            best_iter = model.best_iteration + 1
            y_pred = model.predict(X_vl, iteration_range=(0, best_iter))
            if return_oof_test:
                oof_test += model.predict(X_test, iteration_range=(0, best_iter)) / n_splits
        elif isinstance(model, catb.CatBoostRegressor):
            model.fit(X_tr,
                      y_tr,
                      eval_set=[(X_vl, y_vl)],
                      use_best_model=True,
                      verbose=0)
        
            y_pred = model.predict(X_vl)
            if return_oof_test:
                oof_test += model.predict(X_test) / n_splits
                
        pear = pearsonr(y_vl, y_pred)[0]
        mse = mean_squared_error(y_vl, y_pred)
        mae = mean_absolute_error(y_vl, y_pred)
        r2 = r2_score(y_vl, y_pred)
        
        scores["pear"].append(pear)
        scores["mse"].append(mse)
        scores["mae"].append(mae)
        scores["r2"].append(r2)
        
        print(f"FOLD {i} | Pearson: {pear:.5f}, Mse: {mse:.5f}, Mae: {mae:.5f}, R2: {r2:.5f}")
        
    average_scores = {name: round(np.mean(score),5) for name, score in scores.items()}
    print(f"\nAverage metrics: {average_scores}\n")
    
    if return_oof_test:
        return oof_test


train_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")


display(train_df.head())
train_df.info()
display(train_df.describe().T[:30])


print(f"Total num of missing: {train_df.isna().sum().sum()}")


base_feature = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
top_100_feature = ["X683", "X140", "X758", "X425", "X752", "X344", "X292", "X646",
                   "X287", "X134", "X385", "X647", "X682", "X279", "X345", "X466",
                   "X381", "X778", "X283", "X739", "X427", "X272", "X684", "X301",
                   "X198", "X465", "X608", "X738", "X384", "X137", "X386", "X581",
                   "X734", "X180", "X589", "X421", "X610", "X780", "X387", "X772",
                   "X654", "X428", "X96", "X779", "X426", "X98", "X591", "X650",
                   "X613", "X566", "X605", "X181", "X750", "X174", "X288", "X607",
                   "X579", "X176", "X508", "X178", "X419", "X219", "X343", "X89",
                   "X678", "X588", "X40", "X293", "X411", "X757", "X337", "X285",
                   "X295", "X341", "X443", "X179", "X575", "X751", "X92", "X562",
                   "X769", "X776", "X501", "X298", "X375", "X95", "X590", "X611",
                   "X94", "X270", "X424", "X86", "X587", "X434", "X638", "X170",
                   "X297", "X136", "X97", "X572"]


top_30_feature = top_100_feature[:30]

FEATURES = [*base_feature, *top_30_feature]

X = train_df[FEATURES].astype("float32")
y_train = train_df["label"].values
test = test_df[FEATURES].astype("float32")

X_train = add_features(X)
X_test = add_features(test)

X_train_ext_lags = add_ext_large_lags(X_train)
X_test_ext_lags = add_ext_large_lags(X_test)

print(f"Original train data shape: {X.shape}")
print(f"Original test data shape: {test.shape}")
print(f"Shape y_train: {y_train.shape}\n")

print(f"Shape X_train: {X_train.shape}")
print(f"Shape X_test: {X_test.shape}\n")

print(f"Shape X_train_ext_lags: {X_train_ext_lags.shape}")
print(f"Shape X_test_ext_lags: {X_test_ext_lags.shape}\n")


# hyperparameters were selected through Optuna
xgb_extl_params = {"objective": "reg:squarederror",
                   "eval_metric": "rmse",
                   "n_estimators": 315,
                   "max_depth": 5,
                   "max_bin": 101,
                   "learning_rate": 0.018525299880551214,
                   "subsample": 0.9341890485283013,
                   "colsample_bytree": 0.7365066282406134,
                   "reg_alpha": 0.0151658416191432,
                   "reg_lambda": 0.3286769939771411,
                   "tree_method": "hist",
                   "early_stopping_rounds": 50,
                   "device": "cuda",
                   "random_state": 42}

catb_params = {"loss_function": "RMSE",
                "eval_metric": "RMSE",
                "iterations": 837,
                "depth": 4,
                "learning_rate": 0.07560459711899144,
                "reg_lambda": 0.023952575229463154,
                "early_stopping_rounds": 50,
                "task_type": "GPU",
                "random_seed": 42}


models = [
    {"name": "xgb_extl", "model": XGBRegressor(**xgb_extl_params), "data": (X_train_ext_lags, X_test_ext_lags)},
    {"name": "catb", "model": CatBoostRegressor(**catb_params), "data": (X_train, X_test)},
]

oof_test = {}

for model in models:
    name = model["name"]
    estimator = model["model"]
    train, test = model["data"]
    
    print(f"Model {name}")
    oof_test[name] = cross_val_kf(estimator,
                                  train.values,
                                  y_train,
                                  test.values,
                                  return_oof_test=True)


submission["prediction"] = oof_test["catb"]
submission.to_csv("catb0.csv", index=False)
submission.head()

