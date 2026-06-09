# importing
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from category_encoders import TargetEncoder

import warnings
warnings.filterwarnings('ignore')



df=pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
te=pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
tte=te.copy()


df


te


numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col].dropna(), bins=200, kde=True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    df[col].value_counts().nlargest(10).plot(kind='bar', color='skyblue')
plt.tight_layout()
plt.show()



plt.figure(figsize=(20,5))
sns.histplot(df['Listening_Time_minutes'], bins=150, kde=True)
plt.show()


plt.figure(figsize=(20,5))
sns.histplot(np.log1p(df['Listening_Time_minutes']), bins=50, kde=True)
plt.show()


df.info()


plt.figure(figsize=(20, 8))
sns.heatmap(df.isnull(), cmap="coolwarm") 
plt.show()


df['Episode_Length_minutes']=df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
df['Guest_Popularity_percentage']=df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())

df["Number_of_Ads"] = df["Number_of_Ads"].fillna(0).astype(int)
mode_value = df["Number_of_Ads"].mode()[0]  
df["Number_of_Ads"] = df["Number_of_Ads"].fillna(mode_value).astype(int)


te.info()


plt.figure(figsize=(20, 8))
sns.heatmap(te.isnull(), cmap="coolwarm") 
plt.show()


te['Episode_Length_minutes']=te['Episode_Length_minutes'].fillna(te['Episode_Length_minutes'].median())
te['Guest_Popularity_percentage']=te['Guest_Popularity_percentage'].fillna(te['Guest_Popularity_percentage'].median())

te["Number_of_Ads"] = te["Number_of_Ads"].fillna(0).astype(int)
mode_value = te["Number_of_Ads"].mode()[0]  
te["Number_of_Ads"] = te["Number_of_Ads"].fillna(mode_value).astype(int)


print(df.info(), te.info())


from itertools import combinations  
from joblib import Parallel, delayed


df["Weekday"] = df["Publication_Day"].map({
    "Sunday": 0, 
    "Monday": 1, 
    "Tuesday": 2, 
    "Wednesday": 3, 
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
})
df["SinWeekday"] = np.sin(2 * np.pi * df["Weekday"] / 7)
df["CosWeekday"] = np.cos(2 * np.pi * df["Weekday"] / 7)
df["Time"] = df["Publication_Time"].map({
    "Morning": 0, 
    "Afternoon": 1, 
    "Evening": 2, 
    "Night": 3, 
})
df["SinTime"] = np.sin(2 * np.pi * df["Time"] / 4)
df["CosTime"] = np.cos(2 * np.pi * df["Time"] / 4)
df["Episode_Title"] = df["Episode_Title"].str.split(" ", expand=True)[1].astype(np.uint16)
df["Number_of_Ads"] = df["Number_of_Ads"].fillna(0).clip(0, 3).astype(np.uint8)
df["Episode_Length_minutes"] = df['Episode_Length_minutes'].fillna(60)
df['SinEpLen'] = np.sin(2 * np.pi * df['Episode_Length_minutes'] / 60)
df['CosEpLen'] = np.cos(2 * np.pi * df['Episode_Length_minutes'] / 60)
del df["Publication_Time"], df["Publication_Day"]
df["ELen_Int"] = np.floor(df["Episode_Length_minutes"])
df["ELen_Dec"] = df["Episode_Length_minutes"] - df["ELen_Int"]
cat_cols = [
    "Podcast_Name", "Episode_Title", "Genre", "Number_of_Ads", 
    "Episode_Sentiment", "ELen_Int"
]
df[cat_cols] = df[cat_cols].astype("string")
for col1, col2 in combinations(cat_cols, 2):
    df[f"{col1}-{col2}"] = df[col1] + "-" + df[col2]


te["Weekday"] = te["Publication_Day"].map({
    "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6
})
te["SinWeekday"] = np.sin(2 * np.pi * te["Weekday"] / 7)
te["CosWeekday"] = np.cos(2 * np.pi * te["Weekday"] / 7)
te["Time"] = te["Publication_Time"].map({
    "Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3
})
te["SinTime"] = np.sin(2 * np.pi * te["Time"] / 4)
te["CosTime"] = np.cos(2 * np.pi * te["Time"] / 4)
te["Episode_Title"] = te["Episode_Title"].str.split(" ", expand=True)[1].astype(np.uint16)
te["Number_of_Ads"] = te["Number_of_Ads"].fillna(0).clip(0, 3).astype(np.uint8)
te["Episode_Length_minutes"] = te['Episode_Length_minutes'].fillna(60)
te['SinEpLen'] = np.sin(2 * np.pi * te['Episode_Length_minutes'] / 60)
te['CosEpLen'] = np.cos(2 * np.pi * te['Episode_Length_minutes'] / 60)
del te["Publication_Time"], te["Publication_Day"]
te["ELen_Int"] = np.floor(te["Episode_Length_minutes"])
te["ELen_Dec"] = te["Episode_Length_minutes"] - te["ELen_Int"]
te[cat_cols] = te[cat_cols].astype("string").fillna("missing")
for col1, col2 in combinations(cat_cols, 2):
    te[f"{col1}-{col2}"] = te[col1] + "-" + te[col2]


print(df.info(), te.info())


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train = df
test = te
string_cols = [col for col in train.columns if train[col].dtype == 'string']
encoders = {}

for col in string_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    encoders[col] = le
for col in string_cols:
    if col in test.columns:
        le = encoders[col]
        test[col] = test[col].astype(str).map(
            lambda x: x if x in le.classes_ else '-1'
        ).map(
            {val: idx for idx, val in enumerate(np.append(le.classes_, '-1'))}
        ).astype(int)


df=train.copy()
te=test.copy()


import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

try:
    import cudf
except ImportError:
    cudf = None

try:
    from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback
    has_optuna_integration = True
except ImportError:
    has_optuna_integration = False
    print("Warning: optuna-integration not found. Running without pruning callbacks.")
    print("Install with: pip install optuna-integration[xgboost,lightgbm]")

def preprocess_data(df, te, target_col='Listening_Time_minutes', use_gpu=True):
    df_local = cudf.from_pandas(df) if use_gpu and cudf else df
    te_local = cudf.from_pandas(te) if use_gpu and cudf else te
    quantiles = df_local[target_col].quantile([0.25, 0.75])
    if cudf and use_gpu:
        quantiles = quantiles.to_pandas().tolist()
        Q1, Q3 = quantiles[0], quantiles[1]
    else:
        Q1, Q3 = quantiles
    IQR = Q3 - Q1
    bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    if cudf and use_gpu:
        mask = df_local[target_col].between(left=bounds[0], right=bounds[1])
    else:
        mask = df_local[target_col].between(*bounds)
    X = df_local.loc[mask].drop(columns=target_col).astype('float32')
    y = df_local.loc[mask, target_col].astype('float32')
    X_test = te_local[X.columns].astype('float32')
    X_median = X.median()
    X_test = X_test.fillna(X_median)
    if use_gpu and cudf:
        X, y, X_test = X.to_pandas(), y.to_pandas(), X_test.to_pandas()
    print(f"Rows after outlier removal: {X.shape[0]} (original: {df.shape[0]})")
    return X, y, X_test

def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'gpu_id': 0 if cudf else -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
        'random_state': 42,
        'n_jobs': -1,
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    dtrain = xgb.DMatrix(X_train, y_train, nthread=-1)
    dval = xgb.DMatrix(X_val, y_val, nthread=-1)
    callbacks = [XGBoostPruningCallback(trial, 'val-rmse')] if has_optuna_integration else None
    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, 'val')], early_stopping_rounds=50,
                      verbose_eval=False, callbacks=callbacks)
    pred_val = model.predict(dval)
    mse = np.mean((y_val - pred_val) ** 2)
    return mse

def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'device': 'gpu' if cudf else 'cpu',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
        'random_state': 42,
        'n_jobs': -1,
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    callbacks = [LightGBMPruningCallback(trial, 'mse')] if has_optuna_integration else None
    model = lgb.train(params, dtrain, num_boost_round=1000,
                      valid_sets=[dval], callbacks=callbacks)
    pred_val = model.predict(X_val)
    mse = np.mean((y_val - pred_val) ** 2)
    return mse

X, y, X_test = preprocess_data(df, te)

study_xgb = optuna.create_study(direction='minimize', 
                                pruner=optuna.pruners.MedianPruner() if has_optuna_integration else None)
study_xgb.optimize(xgb_objective, n_trials=30, timeout=1800)
best_params_xgb = study_xgb.best_params
best_params_xgb.update({
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'gpu_id': 0 if cudf else -1,
    'random_state': 42,
    'n_jobs': -1
})
print("Best XGBoost Params:", best_params_xgb)

study_lgb = optuna.create_study(direction='minimize', 
                                pruner=optuna.pruners.MedianPruner() if has_optuna_integration else None)
study_lgb.optimize(lgb_objective, n_trials=30, timeout=1800)
best_params_lgb = study_lgb.best_params
best_params_lgb.update({
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'device': 'gpu' if cudf else 'cpu',
    'random_state': 42,
    'n_jobs': -1
})
print("Best LightGBM Params:", best_params_lgb)


dtrain_xgb = xgb.DMatrix(X, y, nthread=-1)
dtest_xgb = xgb.DMatrix(X_test, nthread=-1)
model_xgb = xgb.train(best_params_xgb, dtrain_xgb, num_boost_round=500)

dtrain_lgb = lgb.Dataset(X, label=y)
model_lgb = lgb.train(best_params_lgb, dtrain_lgb, num_boost_round=500)

# Predictions
pred_xgb = model_xgb.predict(dtest_xgb)
pred_lgb = model_lgb.predict(X_test)

weights = [0.5, 0.5] 
pred_final = weights[0] * pred_xgb + weights[1] * pred_lgb

pred_final = np.where(np.isfinite(pred_final), pred_final, y.median())
pred_final = np.clip(pred_final, 0, 120)
#pred_final = np.clip(pred_final, 0, None)



submission = pd.DataFrame({'id': te['id'].values, 'Listening_Time_minutes': pred_final})
submission.to_csv('submission.csv', index=False)
print("Submission generated. Shape:", submission.shape)


del dtrain_xgb, dtest_xgb, model_xgb, dtrain_lgb, model_lgb, X, y, X_test
import gc
gc.collect()

