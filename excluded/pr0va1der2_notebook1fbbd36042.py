DIR = '/kaggle/input/pump-fun-graduation-february-2025'


!ls {DIR}


import pandas as pd
import os
import catboost


train = pd.read_csv(os.path.join(DIR, 'train.csv'))

train.shape


train.columns


filenames = !ls {DIR}/chunk*.csv
filenames


pip install optuna


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

# === 1. Загрузка и объединение данных ===
def generate_features(filenames):
    all_data = []
    for chunk_filename in filenames:
        all_data.append(pd.read_csv(chunk_filename))
    data = pd.concat(all_data)
    features = data.groupby('base_coin').agg({
        'quote_coin_amount': ['sum', 'mean', 'std', 'min', 'max']
    })
    features.columns = ['_'.join(col) for col in features.columns]
    features = features.reset_index()
    return features

features = generate_features(filenames)


# === 2. Подготовка train/valid ===
Xy = train[['mint', 'has_graduated']].merge(features, left_on='mint', right_on='base_coin', how='left')
Xy = Xy.fillna(0)
feature_names = [col for col in Xy.columns if col not in ['mint', 'has_graduated', 'base_coin']]

X_train, X_valid, y_train, y_valid = train_test_split(
    Xy[feature_names], Xy['has_graduated'], test_size=0.2, random_state=42, stratify=Xy['has_graduated']
)

# === 3. Обучение CatBoost ===
model_catboost = CatBoostClassifier(
    learning_rate=0.04758936724213868,
    depth=7,
    l2_leaf_reg=1.3238501964751692,
    random_strength=2.994700824759074,
    bagging_temperature=0.29416732757599007,
    border_count=236,
    verbose=0
)
model_catboost.fit(X_train, y_train)

# === 4. Обучение LightGBM ===
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.03145858852702148,
    'num_leaves': 82,
    'max_depth': 12,
    'min_child_samples': 63,
    'subsample': 0.8846473499346016,
    'colsample_bytree': 0.8999435498004282,
    'reg_alpha': 0.21677488229728717,
    'reg_lambda': 9.62559335326224,
    'feature_pre_filter': False
}
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid)
model_lgb = lgb.train(
    lgb_params,
    train_set,
    valid_sets=[valid_set],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)


# === 5. Обучение XGBoost ===
xgb_params = {
    'learning_rate': 0.025264491304431442,
    'max_depth': 7,
    'min_child_weight': 0.6396286406065624,
    'subsample': 0.7285846212790773,
    'colsample_bytree': 0.722705040265217,
    'reg_alpha': 2.71726790931753,
    'reg_lambda': 0.01598889492130562,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'verbosity': 0,
    'n_estimators': 1000
}
model_xgb = xgb.XGBClassifier(**xgb_params)
model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=False)

# === 6. Предсказания и ансамбль ===
p_cat = model_catboost.predict_proba(X_valid)[:, 1]
p_lgb = model_lgb.predict(X_valid)
p_xgb = model_xgb.predict_proba(X_valid)[:, 1]

p_ens = (p_cat + p_lgb + p_xgb) / 3

# === 7. Log loss по валидации ===
print("CatBoost Log Loss:", log_loss(y_valid, p_cat))
print("LightGBM Log Loss:", log_loss(y_valid, p_lgb))
print("XGBoost Log Loss:", log_loss(y_valid, p_xgb))
print("Ensemble Log Loss:", log_loss(y_valid, p_ens))


# === 8. Предсказания на тесте и создание сабмишн ===
test = pd.read_csv(os.path.join(DIR, 'test_unlabeled.csv'))
X_test = test[['mint']].merge(features, left_on='mint', right_on='base_coin', how='left')
X_test = X_test.fillna(0)

p_cat_test = model_catboost.predict_proba(X_test[feature_names])[:, 1]
p_lgb_test = model_lgb.predict(X_test[feature_names])
p_xgb_test = model_xgb.predict_proba(X_test[feature_names])[:, 1]

p_ens_test = (p_cat_test + p_lgb_test + p_xgb_test) / 3

submission = X_test[['mint']].copy()
submission['has_graduated'] = p_ens_test
assert submission.shape[0] == test.shape[0]
submission.to_csv('submission.csv', index=False)

print("Submission file has been created successfully!")


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

def generate_features(filenames):
    all_data = []
    for fname in filenames:
        all_data.append(pd.read_csv(fname))
    df = pd.concat(all_data)

    df['block_time'] = pd.to_datetime(df['block_time'])

    # Агрегаты по группировке base_coin
    agg = df.groupby('base_coin').agg({
        'quote_coin_amount': ['sum', 'mean', 'std', 'min', 'max'],
        'base_coin_amount': ['sum', 'mean', 'std', 'min', 'max'],
        'virtual_token_balance_after': ['mean', 'std'],
        'virtual_sol_balance_after': ['mean', 'std'],
        'block_time': ['min', 'max'],
        'tx_idx': 'count'
    })

    agg.columns = ['_'.join(col) for col in agg.columns]
    agg.reset_index(inplace=True)

    # Кол-во покупок и продаж
    buys = df[df['direction'] == 'buy'].groupby('base_coin').size().rename('num_buys')
    sells = df[df['direction'] == 'sell'].groupby('base_coin').size().rename('num_sells')

    features = agg.merge(buys, on='base_coin', how='left').merge(sells, on='base_coin', how='left')
    features['num_buys'] = features['num_buys'].fillna(0)
    features['num_sells'] = features['num_sells'].fillna(0)

    features['total_txs'] = features['tx_idx_count']
    features['buy_ratio'] = features['num_buys'] / (features['total_txs'] + 1e-6)
    features['sell_ratio'] = features['num_sells'] / (features['total_txs'] + 1e-6)

    # Временной диапазон активности
    features['block_time_min'] = pd.to_datetime(features['block_time_min'])
    features['block_time_max'] = pd.to_datetime(features['block_time_max'])
    features['block_time_range'] = (features['block_time_max'] - features['block_time_min']).dt.total_seconds()
    features.drop(columns=['block_time_min', 'block_time_max'], inplace=True)
    return features

features = generate_features(filenames)


# === 2. Подготовка train/valid ===
Xy = train[['mint', 'has_graduated']].merge(features, left_on='mint', right_on='base_coin', how='left')
Xy = Xy.fillna(0)
feature_names = [col for col in Xy.columns if col not in ['mint', 'has_graduated', 'base_coin']]

X_train, X_valid, y_train, y_valid = train_test_split(
    Xy[feature_names], Xy['has_graduated'], test_size=0.2, random_state=42, stratify=Xy['has_graduated']
)

# === 3. Обучение CatBoost ===
model_catboost = CatBoostClassifier(
    learning_rate=0.04758936724213868,
    depth=7,
    l2_leaf_reg=1.3238501964751692,
    random_strength=2.994700824759074,
    bagging_temperature=0.29416732757599007,
    border_count=236,
    verbose=0
)
model_catboost.fit(X_train, y_train)

# === 4. Обучение LightGBM ===
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.03145858852702148,
    'num_leaves': 82,
    'max_depth': 12,
    'min_child_samples': 63,
    'subsample': 0.8846473499346016,
    'colsample_bytree': 0.8999435498004282,
    'reg_alpha': 0.21677488229728717,
    'reg_lambda': 9.62559335326224,
    'feature_pre_filter': False
}
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid)
model_lgb = lgb.train(
    lgb_params,
    train_set,
    valid_sets=[valid_set],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)


# === 5. Обучение XGBoost ===
xgb_params = {
    'learning_rate': 0.025264491304431442,
    'max_depth': 7,
    'min_child_weight': 0.6396286406065624,
    'subsample': 0.7285846212790773,
    'colsample_bytree': 0.722705040265217,
    'reg_alpha': 2.71726790931753,
    'reg_lambda': 0.01598889492130562,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'verbosity': 0,
    'n_estimators': 1000
}
model_xgb = xgb.XGBClassifier(**xgb_params)
model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=False)

# === 6. Предсказания и ансамбль ===
p_cat = model_catboost.predict_proba(X_valid)[:, 1]
p_lgb = model_lgb.predict(X_valid)
p_xgb = model_xgb.predict_proba(X_valid)[:, 1]

p_ens = (p_cat + p_lgb + p_xgb) / 3

# === 7. Log loss по валидации ===
print("CatBoost Log Loss:", log_loss(y_valid, p_cat))
print("LightGBM Log Loss:", log_loss(y_valid, p_lgb))
print("XGBoost Log Loss:", log_loss(y_valid, p_xgb))
print("Ensemble Log Loss:", log_loss(y_valid, p_ens))


# === 8. Предсказания на тесте и создание сабмишн ===
test = pd.read_csv(os.path.join(DIR, 'test_unlabeled.csv'))
X_test = test[['mint']].merge(features, left_on='mint', right_on='base_coin', how='left')
X_test = X_test.fillna(0)

p_cat_test = model_catboost.predict_proba(X_test[feature_names])[:, 1]
p_lgb_test = model_lgb.predict(X_test[feature_names])
p_xgb_test = model_xgb.predict_proba(X_test[feature_names])[:, 1]

p_ens_test = (p_cat_test + p_lgb_test + p_xgb_test) / 3

submission = X_test[['mint']].copy()
submission['has_graduated'] = p_ens_test
assert submission.shape[0] == test.shape[0]
submission.to_csv('submission.csv', index=False)

print("Submission file has been created successfully!")

