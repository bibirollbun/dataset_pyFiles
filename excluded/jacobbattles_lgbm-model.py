# In your Kaggle kernel:
!pip install m5-wrmsse

from m5_wrmsse import wrmsse


import numpy as np
import pandas as pd
import gc, psutil, warnings
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Memory reduction function
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith('int'):
                if c_min >= 0:
                    if c_max < 255:    df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535: df[col] = df[col].astype(np.uint16)
                    else:               df[col] = df[col].astype(np.uint32)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    else:
                        df[col] = df[col].astype(np.int32)
            elif str(col_type).startswith('float'):
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"â�‡ï¸� Mem: {start_mem:.2f}â†’{end_mem:.2f} MB "
              f"({100*(start_mem-end_mem)/start_mem:.1f}% â†“)")
    return df

def print_memory():
    used = psutil.virtual_memory().used / 1024**2
    print(f"ğŸ–¥ï¸� RAM used: {used:.0f} MB")


DATA_PATH = '../input/m5-forecasting-accuracy'
# Load datasets
sales = pd.read_csv(f"{DATA_PATH}/sales_train_evaluation.csv")
calendar = pd.read_csv(f"{DATA_PATH}/calendar.csv")
prices = pd.read_csv(f"{DATA_PATH}/sell_prices.csv")
# Reduce memory usage
sales = reduce_mem_usage(sales)
calendar = reduce_mem_usage(calendar)
prices = reduce_mem_usage(prices)
gc.collect()


# Melt sales
sales_long = sales.melt(
    id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],
    var_name='d', value_name='sales'
)
# Merge calendar and price data
sales_long = sales_long.merge(
    calendar[['d','date', 'wm_yr_wk', 'event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']],
    on='d', how='left'
)
sales_long = sales_long.merge(
    prices,
    on=['store_id','item_id','wm_yr_wk'], how='left'
)
# Reduce memory
sales_long = reduce_mem_usage(sales_long)
gc.collect()


# Convert date and sort
sales_long['date'] = pd.to_datetime(sales_long['date'])
sales_long.sort_values(['id','date'], inplace=True)
# Create lag features
lag_days = [7,28]
for lag in lag_days:
    sales_long[f'sales_lag_{lag}'] = sales_long.groupby('id')['sales'].shift(lag).astype(np.float32)
    gc.collect()
# Rolling statistics
rolling_windows = [7,28]
for window in rolling_windows:
    sales_long[f'rolling_mean_{window}'] = (
        sales_long.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(window).mean())
        .astype(np.float32)
    )
    sales_long[f'rolling_std_{window}'] = (
        sales_long.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(window).std())
        .astype(np.float32)
    )
    gc.collect()
# Date features
sales_long['dayofweek'] = sales_long['date'].dt.dayofweek.astype(np.uint8)
sales_long['month'] = sales_long['date'].dt.month.astype(np.uint8)
sales_long['year'] = sales_long['date'].dt.year.astype(np.int16)
sales_long['is_weekend'] = sales_long['dayofweek'].isin([5,6]).astype(np.uint8)
# Event flag
sales_long['is_event'] = sales_long['event_name_1'].notnull().astype(np.int8)
# Price features
sales_long['price_change_rate'] = sales_long.groupby('id')['sell_price'].pct_change().astype(np.float32)
sales_long['price_event_interaction'] = (sales_long['sell_price'] * sales_long['is_event']).astype(np.float32)
gc.collect()


# Create time index for modeling
sales_long['time_idx'] = (sales_long['date'] - sales_long['date'].min()).dt.days
# Define forecasting horizon and split
t_horizon = 28
max_t = sales_long['time_idx'].max()
train = sales_long[sales_long['time_idx'] <= max_t - t_horizon]
valid = sales_long[sales_long['time_idx'] > max_t - t_horizon]
# Features and target
features = [f'sales_lag_{lag}' for lag in lag_days] + \
           [f'rolling_mean_{w}' for w in rolling_windows] + \
           [f'rolling_std_{w}' for w in rolling_windows] + \
           ['dayofweek','month','year','is_weekend','is_event','price_change_rate','price_event_interaction']
categorical_feats = ['item_id','dept_id','cat_id','store_id','state_id']
target = 'sales'


# current numeric/lag/date features
features = (
    [f"sales_lag_{lag}" for lag in lag_days] +
    [f"rolling_mean_{w}" for w in rolling_windows] +
    [f"rolling_std_{w}" for w in rolling_windows] +
    ["dayofweek","month","year","is_weekend","is_event","price_change_rate","price_event_interaction"]
)

# add your categorical columns
all_features = features + categorical_feats

# cast them to pandas 'category' dtype (recommended)
for col in categorical_feats:
    train[col] = train[col].astype("category")
    valid[col] = valid[col].astype("category")


print(lgb.__version__)


# define X & y for LightGBM / Optuna
train_X = train[all_features].copy()
train_y = train[target].copy()

val_X   = valid[all_features].copy()
val_y   = valid[target].copy()

# if your wrmsse() needs the 'id' for each row, make sure you still have it:
val_ids = valid['id'].values


# â”€â”€â”€ Cell: Optuna tuning over multiple objectives â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# (Uncomment to install Optuna):
# !pip install optuna

import optuna
import lightgbm as lgb

def objective(trial):
    # let Optuna pick the loss/objective
    obj = trial.suggest_categorical(
        "objective",
        ["regression", "tweedie", "poisson", "gamma", "huber", "fair", "quantile", "mape"]
    )

    params = {
        'boosting_type': 'gbdt',
        'objective': obj,
        'metric': 'rmse',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 1023),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'verbose': -1,
    }

    # conditional extra params
    if obj == "tweedie":
        params['tweedie_variance_power'] = trial.suggest_uniform(
            'tweedie_variance_power', 1.1, 1.9
        )
    elif obj == "huber":
        params['alpha'] = trial.suggest_uniform('alpha', 0.3, 3.0)
    elif obj == "fair":
        params['fair_c'] = trial.suggest_uniform('fair_c', 0.5, 2.0)
    elif obj == "quantile":
        params['alpha'] = trial.suggest_uniform('alpha', 0.1, 0.9)
    # poisson, gamma, mape need no extra hyperparameters

    # train
    model = lgb.LGBMRegressor(**params, n_estimators=2000)
    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # predict & reshape for WRMSSE
    preds = model.predict(val_X, num_iteration=model.best_iteration_)
    forecast = preds.reshape(-1, t_horizon)

    return wrmsse(forecast)

# run the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80, timeout=7200)

print("Best Optuna params:", study.best_trial.params)




# delete what you no longer need
del study, objective, train_X, train_y, val_X, val_y, val_ids
gc.collect()


# now build the datasets
lgb_train = lgb.Dataset(train[all_features], train[target],
                        categorical_feature=categorical_feats)
lgb_valid = lgb.Dataset(valid[all_features], valid[target],
                        reference=lgb_train,
                        categorical_feature=categorical_feats)
final_params = {
    'objective': 'quantile',
    'metric': 'rmse',
    'learning_rate': 0.06445960075684132,
    'num_leaves': 935,
    'feature_fraction': 0.4867523559488728,
    'bagging_fraction': 0.7651781066144594,
    'bagging_freq': 3,
    'min_child_samples': 88,
    'lambda_l1': 1.2933416611412045e-08,
    'lambda_l2': 7.153097738521381,
    'alpha': 0.20055054152983542,
    # you can still mute LightGBMâ€™s own logging:
    'verbose': 1
}
model_lgb = lgb.train(
    final_params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[lgb.early_stopping(stopping_rounds=50),
               lgb.log_evaluation(period=100)]
)


# --- Step 6 (batched): Create Submission for LightGBM ---
import numpy as np

# unique series ids
test_ids     = sales_long['id'].unique()
# future dates to forecast
future_dates = pd.date_range(
    sales_long['date'].max() + pd.Timedelta(days=1),
    periods=t_horizon
)

# 1) Precompute static categorical values per series
static_df   = sales_long.groupby('id')[categorical_feats].first()
static_vals = static_df.to_dict(orient='index')

# 2) Initialize a history dict of past sales
history     = sales_long.groupby('id')['sales'].apply(list).to_dict()

# 3) Prepare container for forecasts
preds_dict  = {iid: [] for iid in test_ids}

# 4) For each future day, build one big batch and predict
for date in future_dates:
    rows = []
    for iid in test_ids:
        hist = history[iid]
        row  = {}
        # a) static cats
        row.update(static_vals[iid])
        # b) lag features
        for lag in lag_days:
            row[f"sales_lag_{lag}"] = hist[-lag]
        # c) rolling stats
        for w in rolling_windows:
            window = hist[-w:]
            row[f"rolling_mean_{w}"] = np.mean(window)
            row[f"rolling_std_{w}"]  = np.std(window, ddof=0)
        # d) date features
        row["dayofweek"]             = date.dayofweek
        row["month"]                 = date.month
        row["year"]                  = date.year
        row["is_weekend"]            = int(date.weekday() >= 5)
        # e) event & price (assume 0 for future)
        row["is_event"]               = 0
        row["price_change_rate"]      = 0.0
        row["price_event_interaction"] = 0.0

        rows.append(row)

    # assemble batch
    X_batch = pd.DataFrame(rows)[all_features]
    # cast cats once
    for c in categorical_feats:
        X_batch[c] = X_batch[c].astype("category")

    # vectorized predict
    batch_preds = model_lgb.predict(X_batch)

    # update history & store preds
    for iid, pred in zip(test_ids, batch_preds):
        history[iid].append(pred)
        preds_dict[iid].append(pred)

# 5) Build and save submission
submission_lgb = pd.DataFrame({"id": test_ids})
for i in range(t_horizon):
    submission_lgb[f"F{i+1}"] = submission_lgb["id"].map(lambda iid: preds_dict[iid][i])

submission_lgb.to_csv("submission_lgb_optuna.csv", index=False)
print("LightGBM batched submission saved to submission_lgb.csv")


import pandas as pd

# 1) Load raw wide evaluation history
sales_eval = pd.read_csv(f"{DATA_PATH}/sales_train_evaluation.csv")

# 2) Select only the last 28 known days (d_1914â€¦d_1941)
val_days = [f"d_{i}" for i in range(1914, 1942)]
validation_df = sales_eval[["id"] + val_days].copy()

# 3) Suffix IDs and rename columns to match F1â€¦F28
# 3) Remove any trailing "_evaluation", then suffix "_validation"
validation_df["id"] = (
    validation_df["id"]
      .str.replace(r"_evaluation$", "", regex=True)
      + "_validation"
)
validation_df.columns = ["id"] + [f"F{i+1}" for i in range(28)]

# 4) Append to your evaluation-only submission
final_sub = pd.concat([validation_df, submission_lgb], ignore_index=True)

# 5) Save out the full submission
final_sub.to_csv("optuna_submission.csv", index=False)
print("submission.csv written (validation actuals + evaluation preds).")


