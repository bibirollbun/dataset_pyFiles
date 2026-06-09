import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° train/test
train = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/train.csv')
test = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/test_unlabeled.csv')

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ğ¸ Ğ¾ Ñ‚Ğ¾ĞºĞµĞ½Ğ°Ñ…
token_info = pd.read_csv('/kaggle/input/pump-fun-graduation-february-2025/dune_token_info.csv')

# ĞŸĞ¾Ğ´ĞºĞ»Ñ�Ñ‡Ğ°ĞµĞ¼ created_at Ğº train/test
token_info['created_at'] = pd.to_datetime(token_info['created_at'], errors='coerce')
train = train.merge(token_info[['token_mint_address', 'created_at']],
                    left_on='mint', right_on='token_mint_address', how='left')
test = test.merge(token_info[['token_mint_address', 'created_at']],
                   left_on='mint', right_on='token_mint_address', how='left')


from glob import glob

# ĞŸÑƒÑ‚ÑŒ ĞºĞ¾ Ğ²Ñ�ĞµĞ¼ Ñ„Ğ°Ğ¹Ğ»Ğ°Ğ¼ chunk*.csv
chunk_files = glob('/kaggle/input/pump-fun-graduation-february-2025/chunk_*.csv')

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ‘Ğ¼ Ğ¿ÑƒÑ�Ñ‚Ğ¾Ğ¹ Ğ´Ğ°Ñ‚Ğ°Ñ„Ñ€ĞµĞ¹Ğ¼
aggregated_stats = {}

for file in chunk_files:
    print(f"Processing {file}...")
    chunk = pd.read_csv(file, usecols=[
        'base_coin', 'quote_coin_amount', 'base_coin_amount', 'direction'
    ])
    
    # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ¸Ğ¼ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½ÑƒÑ� Ñ„Ğ¸Ñ‡Ñƒ: Ğ±Ñ‹Ğ»Ğ¾ Ğ»Ğ¸ Ñ�Ñ‚Ğ¾ Ğ¿Ğ¾ĞºÑƒĞ¿ĞºĞ¾Ğ¹
    chunk['is_buy'] = (chunk['direction'] == 'buy').astype(int)
    
    # Ğ�Ğ³Ñ€ĞµĞ³Ğ°Ñ†Ğ¸Ñ� Ğ¿Ğ¾ base_coin (mint)
    agg = chunk.groupby('base_coin').agg(
        tx_count=('base_coin_amount', 'count'),
        sol_volume=('quote_coin_amount', 'sum'),
        token_volume=('base_coin_amount', 'sum'),
        avg_sol_per_tx=('quote_coin_amount', 'mean'),
        buy_count=('is_buy', 'sum')
    )
    
    # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ² Ğ¾Ğ±Ñ‰Ğ¸Ğ¹ dict
    for mint, row in agg.iterrows():
        if mint not in aggregated_stats:
            aggregated_stats[mint] = row
        else:
            aggregated_stats[mint] += row

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ¾ Ğ² DataFrame
tx_features = pd.DataFrame.from_dict(aggregated_stats, orient='index').reset_index()
tx_features.rename(columns={'index': 'mint'}, inplace=True)



train = train.merge(tx_features, on='mint', how='left')
test = test.merge(tx_features, on='mint', how='left')


# Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ğ¸Ğ¼ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸ Ğ½ÑƒĞ»Ñ�Ğ¼Ğ¸
for col in ['tx_count', 'sol_volume', 'token_volume', 'avg_sol_per_tx', 'buy_count']:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# Ğ¦ĞµĞ»ĞµĞ²Ğ°Ñ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ°Ñ�
train['has_graduated'] = train['has_graduated'].astype(int)


features = ['tx_count', 'sol_volume', 'token_volume', 'avg_sol_per_tx', 'buy_count']
X = train[features]
y = train['has_graduated']

# Ğ Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ½Ğ° train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

def objective_refined(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 160),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.15, 0.25, log=True),
        'subsample': trial.suggest_float('subsample', 0.9, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.9),
        'gamma': trial.suggest_float('gamma', 0.0, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 7, 11),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    # ĞšÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ�
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    losses = []

    for train_idx, val_idx in kf.split(X, y):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        # ĞœĞ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_cv)
        X_val_scaled = scaler.transform(X_val_cv)

        model = XGBClassifier(**params)
        model.fit(X_train_scaled, y_train_cv)

        preds = model.predict_proba(X_val_scaled)[:, 1]
        loss = log_loss(y_val_cv, preds)
        losses.append(loss)

    return np.mean(losses)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)  # Ğ¼Ğ¾Ğ¶Ğ½Ğ¾ ÑƒĞ²ĞµĞ»Ğ¸Ñ‡Ğ¸Ñ‚ÑŒ Ğ´Ğ¾ 100+ Ğ´Ğ»Ñ� Ğ¼Ğ¾Ñ‰Ğ½Ñ‹Ñ… Ğ¼Ğ°ÑˆĞ¸Ğ½

print("ğŸ”¥ Best trial:")
print(f"  Log Loss: {study.best_value:.4f}")
print("  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")


from sklearn.metrics import log_loss
# ĞœĞ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test[features])

# Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
model = XGBClassifier(
    n_estimators=206,
    max_depth=8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    learning_rate=0.04843705110731193,
    subsample=0.7468505840659705,
    colsample_bytree=0.908674393847986,
    gamma=0.09871987675512672,
    min_child_weight=8
)
model.fit(X_train_scaled, y_train)

# ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹ Ğ½Ğ° Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸
val_probs = model.predict_proba(X_val_scaled)[:, 1]

# Ğ�Ñ†ĞµĞ½ĞºĞ° log loss
val_logloss = log_loss(y_val, val_probs)
print(f"Validation Log Loss: {val_logloss:.4f}")


import matplotlib.pyplot as plt
# Ğ’Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
importances = model.feature_importances_

# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance for XGBoost Model')
plt.gca().invert_yaxis()  # ĞŸĞµÑ€ĞµĞ²Ğ¾Ñ€Ğ°Ñ‡Ğ¸Ğ²Ğ°ĞµĞ¼ Ğ¾Ñ�ÑŒ Ğ´Ğ»Ñ� Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ğ¿Ğ¾Ñ€Ñ�Ğ´ĞºĞ°
plt.show()


# ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹
test['has_graduated'] = model.predict_proba(X_test_scaled)[:, 1]

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ğµ Ñ„Ğ°Ğ¹Ğ»Ğ°
submission = test[['mint', 'has_graduated']]
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv Ğ³Ğ¾Ñ‚Ğ¾Ğ²!")

