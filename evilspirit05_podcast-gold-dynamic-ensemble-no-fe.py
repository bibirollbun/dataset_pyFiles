import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from pathlib import Path


path = Path('/kaggle/input/predict-podcast-new')
train = pd.read_csv(path/'train.csv')
test = pd.read_csv(path/'test.csv')
sub = pd.read_csv(path/'sample_submission.csv')
train.drop(columns=["id"],axis=1,inplace=True)


print("Check Out Train DaTA Null Values: ",train.isnull().sum())
print("#"*130)
print(f"Train Data Shape: {train.shape}")
print("#"*130)
print(f"Train Data INFO: {train.info()}")
print("#"*130)


print("Check Out Test DaTA Null Values: ",test.isnull().sum())
print("#"*130)
print(f"Test Data Shape: {test.shape}")
print("#"*130)
print(f"Test Data INFO: {test.info()}")
print("#"*130)


def clean_data(df):
    df=df.copy()
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean())
    df["Guest_Popularity_percentage"] = df['Guest_Popularity_percentage'].fillna(df["Guest_Popularity_percentage"].mean())

    return df

train=clean_data(train)
test=clean_data(test)


train.head()


test.head()


cat_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for c in cat_cols:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')


y = train['Listening_Time_minutes'].values
train = train.drop(['Listening_Time_minutes'], axis=1)
test.drop(columns=["id"],axis=1,inplace=True)


kf = KFold(n_splits=10, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))
oof_cb = np.zeros(len(train))
pred_cb = np.zeros(len(test))
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(train)):
    X_tr, X_val = train.iloc[trn_idx], train.iloc[val_idx]
    y_tr, y_val = y[trn_idx], y[val_idx]

    # === LightGBM ===
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 256,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    model_lgb = lgb.train(
        params_lgb,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    oof_lgb[val_idx] = model_lgb.predict(X_val)
    pred_lgb += model_lgb.predict(test) / 10

    # === CatBoost ===
    model_cb = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        random_seed=42,
        verbose=False,
        cat_features=cat_cols
    )

    model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
    oof_cb[val_idx] = model_cb.predict(X_val)
    pred_cb += model_cb.predict(test) / 10

    # === XGBoost ===
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        objective='reg:squarederror',
        random_state=42,
        enable_categorical=True,
        verbosity=0
    )

    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    pred_xgb += model_xgb.predict(test) / 10


oof_df = pd.DataFrame({'lgb': oof_lgb,'cb': oof_cb,'xgb': oof_xgb,'y': y})

def find_best_weights(df):
    best_rmse = 1e9
    best_w = None
    step = 0.01
    for w1 in np.arange(0.0, 1.01, step):
        for w2 in np.arange(0.0, 1.01 - w1, step):
            w3 = 1.0 - w1 - w2
            pred = w1 * df['lgb'] + w2 * df['cb'] + w3 * df['xgb']
            rmse = mean_squared_error(df['y'], pred, squared=False)
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = (w1, w2, w3)
    return best_w

w_lgb, w_cb, w_xgb = find_best_weights(oof_df)
print(f"Best weights → LGB: {w_lgb:.3f} | CB: {w_cb:.3f} | XGB: {w_xgb:.3f}")

final_pred = w_lgb * pred_lgb + w_cb * pred_cb + w_xgb * pred_xgb


sub['Listening_Time_minutes'] = final_pred
sub.to_csv('submission.csv', index=False)
sub.head()
















