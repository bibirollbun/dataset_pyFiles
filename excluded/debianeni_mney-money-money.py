import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import entropy

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


DIR = '/kaggle/input/pump-fun-graduation-february-2025'
train = pd.read_csv(os.path.join(DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DIR, 'test_unlabeled.csv'))
filenames = [os.path.join(DIR, f) for f in os.listdir(DIR) if f.startswith("chunk")]



def generate_features(filenames):
    all_data = []
    for chunk_filename in tqdm(filenames):
        all_data.append(pd.read_csv(chunk_filename))
    data = pd.concat(all_data)
    features = data.groupby('base_coin').agg({
        'quote_coin_amount': 'sum',  # Общий объем торгов в SOL
    }).rename(columns={'quote_coin_amount': 'total_trade_volume'})
    return features

features = generate_features(filenames)




train = train[train["is_valid"] == True]
train = train.merge(features, left_on='mint', right_on='base_coin', how='left')
test = test.merge(features, left_on='mint', right_on='base_coin', how='left')


def extract_features(df):
    df = df.copy()
    df["mint_len"] = df["mint"].str.len()
    df["mint_digit_count"] = df["mint"].str.count(r'\d')
    df["mint_letter_count"] = df["mint"].str.count(r'[A-Za-z]')
    df["mint_entropy"] = df["mint"].apply(lambda x: entropy(list(pd.Series(list(x)).value_counts(normalize=True))))
    df["mint_unique_chars"] = df["mint"].apply(lambda x: len(set(x)))
    df["mint_upper_count"] = df["mint"].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["mint_starts_with_H"] = df["mint"].str.startswith("H").astype(int)

    df["slot_min_unix"] = df["slot_min"] / 2.5 + 1660000000
    df["slot_hour"] = pd.to_datetime(df["slot_min_unix"], unit='s').dt.hour
    df["slot_dayofweek"] = pd.to_datetime(df["slot_min_unix"], unit='s').dt.dayofweek

    return df[[
        "mint_len", "mint_digit_count", "mint_letter_count", "mint_entropy",
        "mint_unique_chars", "mint_upper_count", "mint_starts_with_H",
        "slot_min", "slot_hour", "slot_dayofweek",
        "total_trade_volume"
    ]]

X = extract_features(train)
y = train["has_graduated"]
X_test = extract_features(test)

# Разделение train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




print(" Обучаем CatBoost...")
cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    verbose=100,
    random_state=42
)
cat.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_val_pred = cat.predict_proba(X_val)[:, 1]
cat_test_pred = cat.predict_proba(X_test)[:, 1]
print("CatBoost log loss:", log_loss(y_val, cat_val_pred))


print("\n Обучаем XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)]
    
)
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]
print("XGBoost log loss:", log_loss(y_val, xgb_val_pred))


print("\n Обучаем LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss'
)
lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
lgb_test_pred = lgb_model.predict_proba(X_test)[:, 1]
print("LightGBM log loss:", log_loss(y_val, lgb_val_pred))



val_stack = pd.DataFrame({
    "cat": cat_val_pred,
    "xgb": xgb_val_pred,
    "lgb": lgb_val_pred
})

test_stack = pd.DataFrame({
    "cat": cat_test_pred,
    "xgb": xgb_test_pred,
    "lgb": lgb_test_pred
})

meta_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    eval_metric='logloss',
    random_state=42
)
meta_model.fit(val_stack, y_val, eval_set=[(val_stack, y_val)], verbose=100)


print("Stacked Log loss:", log_loss(y_val, final_val_pred))


final_test_pred = meta_model.predict_proba(test_stack)[:, 1]
submission = test[["mint"]].copy()
submission["has_graduated"] = final_test_pred
submission.to_csv("stacked_submission_renew.csv", index=False)
submission.head()

