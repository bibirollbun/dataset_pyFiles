import pandas as pd
import numpy as np
import os 
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
import gc
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from tqdm.auto import tqdm
import joblib


class CONFIG:
    target_col = "responder_6"
    feature_cols = ["symbol_id", "time_id"] \
        + [f"feature_{idx:02d}" for idx in range(79)] \
        + [f"responder_{idx}_lag_1" for idx in range(9)]
    categorical_cols = []
    Debug=True

def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:  # 遍历每列的列名
        col_type = df[col].dtype  # 列名的类型
        if col_type != object and str(col_type) != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df.loc[:, col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df.loc[:, col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df.loc[:, col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df.loc[:, col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:
                        df.loc[:, col] = df[col].astype(np.float32)
                    else:
                        df.loc[:, col] = df[col].astype(np.float16)  
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df.loc[:, col] = df[col].astype(np.float32)
                else:
                    df.loc[:, col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


if CONFIG.Debug:
    file_paths = [f"/kaggle/input/janestree-process-data/train_folder/partition_id={i}/train_data_0.parquet" for i in range(3, 9)]
    # dfs = [pl.read_parquet(file).drop("partition_id") for file in file_paths]
    dfs = [pl.read_parquet(file) for file in file_paths]
    train_df = pl.concat(dfs)
    del file_paths, dfs
    gc.collect()
    print("Final shape:", train_df.shape)
else:
    train_df = pl.read_parquet(f"/kaggle/input/janestree-process-data/train_folder")
    print(train_df.shape)
    train_df=train_df.drop("partition_id")
    print("after drop",train_df.shape)


supervised_usable = (train_df.filter(pl.col('responder_6').is_not_null()))
missing_count = (supervised_usable.null_count().transpose(include_header=True,
                   header_name='feature',
                   column_names=['null_count']).sort('null_count', descending=True).with_columns((pl.col('null_count') / len(supervised_usable)).alias('null_ratio'))
)
       
plt.figure(figsize=(6, 20))
plt.title(f'Missing values over the {len(supervised_usable)} samples which have a target')
plt.barh(np.arange(len(missing_count)), missing_count.get_column('null_ratio'), color='coral', label='missing')
plt.barh(np.arange(len(missing_count)), 
            1 - missing_count.get_column('null_ratio'),
            left=missing_count.get_column('null_ratio'),
            color='darkseagreen', label='available')
plt.yticks(np.arange(len(missing_count)), missing_count.get_column('feature'))
plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
plt.xlim(0, 1)
plt.legend()
plt.show()

del missing_count,supervised_usable
gc.collect()


df_lazy = train_df.lazy()

fill_exprs = [pl.col(c).fill_null(0) for c in train_df.columns]
df_lazy = df_lazy.with_columns(*fill_exprs)
train_df = df_lazy.collect()  # 在最后收集计算结果
del df_lazy, fill_exprs
gc.collect()
    
print("Total null values:", train_df.null_count().select(pl.all().sum()).to_series()[0])

train_size = int(len(train_df) * 0.85)
test = train_df[train_size:]
train = train_df[:train_size]
del train_df,train_size
gc.collect()
print(train.shape,test.shape)


train=train.to_pandas()
test=test.to_pandas()
train.dtypes.value_counts()


X_train = train[CONFIG.feature_cols].copy()
y_train = train[CONFIG.target_col].copy()
w_train = train["weight"].copy()
del train
X_valid = test[CONFIG.feature_cols]
y_valid = test[CONFIG.target_col]
w_valid = test["weight"]
del test
gc.collect()

X_train.shape, y_train.shape, w_train.shape, X_valid.shape, y_valid.shape, w_valid.shape


import optuna
from sklearn.metrics import mean_squared_error

# 定义目标函数
def objective(trial):
    # 定义参数搜索空间
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 
        'max_depth': trial.suggest_int('max_depth', 3, 8), 
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),  
        'subsample': trial.suggest_float('subsample', 0.3, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),  
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5, log=True), 
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5, log=True), 
        'random_state': 1212,
        'tree_method': 'gpu_hist', 
        'device': 'cuda', 
        'n_gpus': 2, 
    }

    print(f"Trial {trial.number} - Params: {params}")
    model = XGBRegressor(**params)


    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        early_stopping_rounds=30,
        verbose=30
    )

    y_pred_valid = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred_valid, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')  

study.optimize(objective, n_trials=50)

print("Best Trial:")
print(study.best_trial.params)
best_params = study.best_trial.params


# Custom R2 metric for XGBoost
def r2_xgb(y_true, y_pred, sample_weight=None):
    # 计算加权均值
    if sample_weight is not None:
        y_mean = np.average(y_true, weights=sample_weight)
    else:
        y_mean = np.mean(y_true)
    numerator = np.sum(sample_weight * (y_pred - y_true) ** 2) if sample_weight is not None else np.sum((y_pred - y_true) ** 2)
    # 总平方和 (TSS)
    denominator = np.sum(sample_weight * (y_true - y_mean) ** 2) if sample_weight is not None else np.sum((y_true - y_mean) ** 2)
    # 避免分母为零
    denominator = max(denominator, 1e-38)
    # R² 计算
    r2 = 1 - numerator / denominator
    return r2
    
XGB_Params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'random_state': 1212,
    'tree_method': 'gpu_hist',
    'device' : 'cuda',
    'n_gpus' : 2,
    }
    
# model_xgb = XGBRegressor(**XGB_Params,eval_metric=r2_xgb, disable_default_eval_metric=True)
# model_xgb.fit(X_train, y_train,sample_weight=w_train,
#               eval_set=[(X_valid, y_valid)], 
#               sample_weight_eval_set=[w_valid], verbose=5)
model_xgb = XGBRegressor(**best_params,eval_metric="rmse")
model_xgb.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],early_stopping_rounds=30,
              verbose=15)

y_pred_valid = model_xgb.predict(X_valid)
valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid )
del X_train,y_train, w_train, X_valid, y_valid, w_valid
gc.collect()
valid_score


os.system('mkdir models')
joblib.dump(model_xgb, '/kaggle/working/models/xgb.model')
print("finish training")
# models=joblib.load("/kaggle/working/models/xgb.model")

