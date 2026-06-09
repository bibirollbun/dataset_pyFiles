import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import glob
import gc
from tqdm.auto import tqdm
import os

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler


DATA_PATH = '/kaggle/input/pump-fun-graduation-february-2025'

CHUNK_PATTERN = os.path.join(DATA_PATH, 'chunk*.csv')
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'test_unlabeled.csv')

DUNE_INFO_FILE = os.path.join(DATA_PATH, 'dune_token_info.csv')
ONCHAIN_INFO_FILE = os.path.join(DATA_PATH, 'token_info_onchain_divers.csv')
SUBMISSION_FILE = 'submission.csv'


train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
dune_info_df = pd.read_csv(DUNE_INFO_FILE)
onchain_info_df = pd.read_csv(ONCHAIN_INFO_FILE)

# Combine train and test for easier processing
train_df['is_train'] = 1
test_df['is_train'] = 0
combined_df = pd.concat([train_df, test_df], ignore_index = True)


combined_df


combined_df.info()


# Load and combine chunk files
all_chunk_files = glob.glob(CHUNK_PATTERN)
print(f"Found {len(all_chunk_files)} chunk files.")

chunk_list = []
for f in tqdm(all_chunk_files, desc="Loading chunks"):
    try:
        chunk_list.append(pd.read_csv(f))
    except Exception as e:
        print(f"Error loading {f}: {e}")
if not chunk_list:
    raise ValueError("No chunk files loaded. Check CHUNK_PATTERN and file existence.")

transactions_df = pd.concat(chunk_list, ignore_index=True)


transactions_df.head()


transactions_df.info()


# Convert timestamps/slots for filtering
transactions_df['block_time'] = pd.to_datetime(transactions_df['block_time'], errors='coerce')
# Ensure slot is numeric
transactions_df['slot'] = pd.to_numeric(transactions_df['slot'], errors='coerce')
combined_df['slot_min'] = pd.to_numeric(combined_df['slot_min'], errors='coerce')

# Merge token creation info (slot_min) with transactions
transactions_df = pd.merge(transactions_df, combined_df[['mint', 'slot_min']],
                           left_on = 'base_coin',
                           right_on = 'mint',
                           how = 'left')

BLOCK_LIMIT = 100
# !!! Crucial Filter: Only keep transactions within the first 100 blocks !!!
transactions_df = transactions_df[transactions_df['slot'] <= transactions_df['slot_min'] + BLOCK_LIMIT]
transactions_df.head()


transactions_df.info()


dune_info_df.head()


dune_info_df.info()


onchain_info_df.head()


onchain_info_df.info()


# Rename columns for clarity before merging metadata
dune_info_df = dune_info_df.rename(columns = {'token_mint_address': 'mint'})
# Select relevant columns and handle potential duplicates (keep first)
dune_info_df = dune_info_df[['mint', 'decimals', 'name', 'symbol', 'token_uri', 'created_at', 'init_tx']].drop_duplicates(subset = ['mint'], keep = 'first')
dune_info_df['created_at'] = pd.to_datetime(dune_info_df['created_at'], errors = 'coerce')

onchain_info_df = onchain_info_df.rename(columns={'mint': 'mint'})
# Select relevant columns and handle potential duplicates (keep first)
onchain_info_df = onchain_info_df[['mint', 'creator', 'bundle_size', 'gas_used']].drop_duplicates(subset = ['mint'], keep = 'first')
# Ensure numeric types
onchain_info_df['bundle_size'] = pd.to_numeric(onchain_info_df['bundle_size'], errors='coerce').fillna(0)
onchain_info_df['gas_used'] = pd.to_numeric(onchain_info_df['gas_used'], errors='coerce')


# Merge metadata into the combined train/test dataframe
combined_df = pd.merge(combined_df, dune_info_df, on='mint', how='left')
combined_df = pd.merge(combined_df, onchain_info_df, on='mint', how='left')
combined_df.head()


print("Basic EDA (Conceptual):")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Transactions shape (first 100 blocks): {transactions_df.shape}")
print(f"Combined shape before features: {combined_df.shape}")

# Check missing values in combined metadata
print("\nMissing values in combined metadata:")
combined_df.isnull().sum() / len(combined_df)


# Check target distribution
print("\nTarget Distribution:")
print(combined_df['has_graduated'].value_counts(normalize = True))


transactions_df.head()


# 1) Group transactions by token mint
grouped_tx = transactions_df.groupby('base_coin') # Group by the token's mint address

# Aggregation dictionary
agg_funcs = {
    'tx_idx': ['count'], # Total transactions
    'block_time': ['min', 'max'], # First and last transaction time
    'slot': ['min', 'max', 'nunique'], # First, last, and number of unique blocks with activity
    'signing_wallet': ['nunique'], # Number of unique traders
    'quote_coin_amount': ['sum', 'mean', 'std', 'max'], # SOL volume stats
    'base_coin_amount': ['sum', 'mean', 'std', 'max'], # Token volume stats
    'virtual_sol_balance_after': ['last', 'max', 'min', 'mean', 'std'], # SOL balance proxy
    'virtual_token_balance_after': ['last', 'max', 'min', 'mean', 'std'] # Token balance proxy
}

# Perform aggregation
agg_features = grouped_tx.agg(agg_funcs)
agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values] # Flatten multi-index
agg_features = agg_features.reset_index().rename(columns = {'base_coin': 'mint'})


transactions_df['base_coin'].unique().shape, transactions_df.shape


agg_features.info()


# 2) Buy/Sell specific features
buy_tx = transactions_df[transactions_df['direction'] == 'buy']
sell_tx = transactions_df[transactions_df['direction'] == 'sell']

grouped_buy = buy_tx.groupby('base_coin')
grouped_sell = sell_tx.groupby('base_coin')

buy_agg = grouped_buy.agg({'tx_idx': ['count'],
                           'signing_wallet': ['nunique'],
                           'quote_coin_amount': ['sum', 'mean', 'max'],
                           'base_coin_amount': ['sum', 'mean', 'max']}).reset_index()
buy_agg.columns = ['mint'] + ['buy_' + '_'.join(col).strip() for col in buy_agg.columns[1:]]

sell_agg = grouped_sell.agg({'tx_idx': ['count'],
                             'signing_wallet': ['nunique'],
                             'quote_coin_amount': ['sum', 'mean', 'max'],
                             'base_coin_amount': ['sum', 'mean', 'max'],}).reset_index()
sell_agg.columns = ['mint'] + ['sell_' + '_'.join(col).strip() for col in sell_agg.columns[1:]]

buy_agg.columns, sell_agg.columns


print("Merging aggregated features...")
combined_df = pd.merge(combined_df, agg_features[[c for c in agg_features.columns if c != 'slot_min']], on = 'mint', how = 'left')
# Merge Buy/Sell specific features
combined_df = pd.merge(combined_df, buy_agg, on = 'mint', how = 'left')
combined_df = pd.merge(combined_df, sell_agg, on = 'mint', how = 'left')


combined_df.info()


# 3) Derived Features
print("Calculating derived features...")
# Check if the required columns exist before calculation ---
required_cols_for_duration = ['block_time_max', 'block_time_min', 'slot_max', 'slot_min', 'tx_idx_count', 'slot_nunique']
# Time-based features
combined_df['tx_duration_seconds'] = (combined_df['block_time_max'] - combined_df['block_time_min']).dt.total_seconds()
# Use the slot_min and slot_max derived from the transaction aggregation
combined_df['tx_duration_slots'] = combined_df['slot_max'] - combined_df['slot_min']
combined_df['avg_time_between_tx'] = combined_df['tx_duration_seconds'] / (combined_df['tx_idx_count'] + 1e-6)
combined_df['tx_per_slot'] = combined_df['tx_idx_count'] / (combined_df['slot_nunique'] + 1e-6)


# 4) Ratio features
required_cols_for_ratios = ['buy_tx_idx_count', 'sell_tx_idx_count',
                            'buy_quote_coin_amount_sum', 'sell_quote_coin_amount_sum',
                            'buy_signing_wallet_nunique', 'sell_signing_wallet_nunique', 'signing_wallet_nunique']

combined_df['buy_sell_count_ratio'] = combined_df['buy_tx_idx_count'] / (combined_df['sell_tx_idx_count'] + 1e-6)
combined_df['buy_sell_vol_ratio'] = combined_df['buy_quote_coin_amount_sum'] / (combined_df['sell_quote_coin_amount_sum'] + 1e-6)
combined_df['unique_buyer_ratio'] = combined_df['buy_signing_wallet_nunique'] / (combined_df['signing_wallet_nunique'] + 1e-6)
combined_df['unique_seller_ratio'] = combined_df['sell_signing_wallet_nunique'] / (combined_df['signing_wallet_nunique'] + 1e-6)


# 5) Creator interaction
creator_trades = transactions_df.groupby(['base_coin', 'signing_wallet']).size().reset_index(name = 'trade_count')
creator_trades = pd.merge(creator_trades, onchain_info_df[['mint', 'creator']], left_on = 'base_coin', right_on = 'mint', how = 'inner')
creator_trades = creator_trades[creator_trades['signing_wallet'] == creator_trades['creator']]
creator_trades = creator_trades[['base_coin', 'trade_count']].rename(columns = {'base_coin': 'mint', 'trade_count': 'creator_trade_count'})
creator_trades = creator_trades.drop_duplicates(subset = ['mint'], keep = 'first')


combined_df = pd.merge(combined_df, creator_trades, on = 'mint', how = 'left')
combined_df['creator_traded'] = combined_df['creator_trade_count'].notna().astype(int)
combined_df['creator_trade_count'] = combined_df['creator_trade_count'].fillna(0)


# 6) Final Feature Selection
print("Selecting final features...")
features_to_drop = ['mint', 'has_graduated', 'slot_graduated', 'is_train', 'slot_min',
                    'name', 'symbol', 'token_uri', 'created_at', 'init_tx',
                    'block_time_min', 'block_time_max',
                    'creator',
                    'is_valid', 'Unnamed: 0']

features = [col for col in combined_df.columns if col not in features_to_drop]
categorical_features = ['creator_encoded']
print(f"Using {len(features)} features: {features}")


for f in features:
    if combined_df[f].dtype == 'object':
        print(f"Warning: Feature '{f}' is object type. Ensure proper handling.")
        try:
            combined_df[f] = pd.to_numeric(combined_df[f])
        except:
            print(f"Could not convert {f} to numeric. Consider encoding or dropping.")
            if f in features: features.remove(f)


round(combined_df[features].isna().sum() / combined_df[features].shape[0], 2)


# Separate train and test again
train_processed = combined_df[combined_df['is_train'] == 1].reset_index(drop = True)
train_processed['has_graduated'] = train_processed['has_graduated'].astype(int)
test_processed = combined_df[combined_df['is_train'] == 0].reset_index(drop = True)

X = train_processed[features]
y = train_processed['has_graduated']
X_test = test_processed[features]

del combined_df, transactions_df, agg_features, buy_agg, sell_agg, creator_trades, chunk_list
gc.collect()


def objective(trial):
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'device': 'cpu',
              'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
              'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
              'num_leaves': trial.suggest_int('num_leaves', 20, 1000),
              'max_depth': trial.suggest_int('max_depth', 3, 12),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
              'subsample': trial.suggest_float('subsample', 0.5, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1.0, log=True),
              'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
              'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='binary_logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
                  categorical_feature=[f for f in categorical_features if f in X.columns])
        
        val_preds = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, val_preds)
        scores.append(score)
    
    return np.mean(scores)
'''
# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ study Ğ¸ Ğ·Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ğ¾Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=25, timeout=3600)
print("Best trial:")
trial = study.best_trial
print(f"  LogLoss: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
'''


print("\nTraining LightGBM model")
lgb_preds_val = np.zeros(len(X))
lgb_preds_test = np.zeros(len(X_test))
lgb_models = []

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"Fold LGBM - {fold+1}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    lgb_params = {'objective': 'binary',
                  'metric': 'logloss',
                  'boosting_type': 'gbdt',
                  'device': 'cpu',
                  'n_estimators': 1494, 
                  'learning_rate': 0.004201672054372531, 
                  'num_leaves': 530, 
                  'max_depth': 8, 
                  'colsample_bytree': 0.5924272277627636, 
                  'subsample': 0.9847923138822793, 
                  'reg_alpha': 0.009466630153726846, 
                  'reg_lambda': 0.28542399074977526, 
                  'min_child_samples': 90,
                  'seed': 42 + fold,
                  'n_jobs': -1,
                  'verbose': -1}

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose = 10)],
              categorical_feature = [f for f in categorical_features if f in X.columns])

    lgb_preds_val[val_idx] = model.predict_proba(X_val)[:, 1]
    lgb_preds_test += model.predict_proba(X_test)[:, 1] / 5
    lgb_models.append(model)
    print(f"Fold {fold+1} Val LogLoss: {round(log_loss(y_val, lgb_preds_val[val_idx]), 4)}")

overall_oof_logloss = log_loss(y, lgb_preds_val)
print(f"\nLGBM Overall Val LogLoss: {round(overall_oof_logloss, 4)}")


def objective(trial):
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'booster': 'gbtree', 'tree_method': 'hist',
              'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
              'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
              'max_depth': trial.suggest_int('max_depth', 3, 12),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
              'gamma': trial.suggest_float('gamma', 0.0, 0.5),
              'subsample': trial.suggest_float('subsample', 0.6, 1.0),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1.0, log=True),
              'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
              'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
    
    scores = []
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=10)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, val_preds)
        scores.append(score)
    
    return np.mean(scores)
'''
study = optuna.create_study(direction = 'minimize', sampler = TPESampler(seed = 42), pruner = optuna.pruners.MedianPruner(n_warmup_steps = 5))
study.optimize(objective, n_trials = 10, timeout = 3600)
print("Best trial:")
trial = study.best_trial
print(f"LogLoss: {trial.value:.5f}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
'''


print("\nTraining XGB model")
xgb_preds_val = np.zeros(len(X))
xgb_preds_test = np.zeros(len(X_test))
xgb_models = []

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"Fold XGB - {fold+1}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'booster': 'gbtree', 'tree_method': 'hist',
                  'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
                  'n_estimators': 1955,
                  'learning_rate': 0.035503048581283086,
                  'max_depth': 12,
                  'min_child_weight': 9,
                  'gamma': 0.29894998940554257,
                  'subsample': 0.9687496940092467,
                  'colsample_bytree': 0.6353970008207678,
                  'reg_alpha': 5.805581976088818e-08,
                  'reg_lambda': 2.5529693461039697e-09,
                  'scale_pos_weight': 0.9879954961448965}
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train,
              eval_set = [(X_val, y_val)],
              early_stopping_rounds = 100,
              verbose = 50)

    xgb_preds_val[val_idx] = model.predict_proba(X_val)[:, 1]
    xgb_preds_test += model.predict_proba(X_test)[:, 1] / 5
    xgb_models.append(model)
    print(f"Fold {fold+1} Val LogLoss: {round(log_loss(y_val, xgb_preds_val[val_idx]), 4)}")

overall_oof_logloss = log_loss(y, xgb_preds_val)
print(f"\nXGB Overall Val LogLoss: {round(overall_oof_logloss, 4)}")


def objective(trial):
    params = {'iterations': trial.suggest_int('iterations', 500, 2000),
              'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
              'depth': trial.suggest_int('depth', 3, 12),
              'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
              'border_count': trial.suggest_int('border_count', 32, 255),
              'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
              'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
              'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
              #'max_leaves': trial.suggest_int('max_leaves', 20, 500),
              'loss_function': 'Logloss', 'eval_metric': 'Logloss', 'random_state': 42, 'thread_count': -1, 'verbose': False, 'allow_writing_files': False}

    cat_features = [f for f, dtype in X.dtypes.items() if dtype == 'category']
    scores = []
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set = (X_val, y_val),
                  cat_features = cat_features,
                  early_stopping_rounds = 100,
                  use_best_model = True)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, val_preds)
        scores.append(score)
    
    return np.mean(scores)
'''
study = optuna.create_study(direction = 'minimize', sampler = TPESampler(seed = 42), pruner=optuna.pruners.MedianPruner(n_warmup_steps = 5))
study.optimize(objective, n_trials = 10, timeout = 3600)
print("\nBest trial:")
trial = study.best_trial
print(f"LogLoss: {trial.value:.5f}")
print("Best params:")
for key, value in trial.params.items():
    print(f"{key}: {value}")
'''


print("\nTraining CatBoost model")
cat_preds_val = np.zeros(len(X))
cat_preds_test = np.zeros(len(X_test))
cat_models = []

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"Fold CatBoost - {fold+1}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    cat_params = {'loss_function': 'Logloss', 'eval_metric': 'Logloss', 'random_state': 42, 'thread_count': -1, 'verbose': False, 'allow_writing_files': False,
                  'iterations': 888, 
                  'learning_rate': 0.02113705944064573, 
                  'depth': 6, 
                  'l2_leaf_reg': 0.3632486956676606, 
                  'border_count': 154, 
                  'bagging_temperature': 0.18485445552552704, 
                  'random_strength': 4.964165793318548, 
                  'grow_policy': 'Depthwise', 
                  'min_data_in_leaf': 60}
   
    cat_features = [f for f, dtype in X.dtypes.items() if dtype == 'category']
    model = cb.CatBoostClassifier(**cat_params)
    model.fit(X_train, y_train,
                  eval_set = (X_val, y_val),
                  cat_features = cat_features,
                  early_stopping_rounds = 100,
                  use_best_model = True)

    cat_preds_val[val_idx] = model.predict_proba(X_val)[:, 1]
    cat_preds_test += model.predict_proba(X_test)[:, 1] / 5
    cat_models.append(model)
    print(f"Fold {fold+1} Val LogLoss: {round(log_loss(y_val, cat_preds_val[val_idx]), 4)}")

overall_oof_logloss = log_loss(y, cat_preds_val)
print(f"\nCatBoost Overall Val LogLoss: {round(overall_oof_logloss, 4)}")


#-- lgb_preds_val -- [LB = 0.324]
#-- cgb_preds_val -- [LB = 0.320]
#-- cat_preds_val -- [LB = 0.320]
# -- meta-model   -- [LB = 0.321]
#-- lgb_preds_test -- xgb_preds_test -- cat_preds_test
meta_train = pd.concat([pd.DataFrame(lgb_preds_val, columns = ["LGBM"]), pd.DataFrame(xgb_preds_val, columns = ["XGB"])], axis = 1)
meta_train = pd.concat([meta_train, pd.DataFrame(cat_preds_val, columns = ["CatBoost"])], axis = 1)

meta_test = pd.concat([pd.DataFrame(lgb_preds_test, columns = ["LGBM"]), pd.DataFrame(xgb_preds_test, columns = ["XGB"])], axis = 1)
meta_test = pd.concat([meta_test, pd.DataFrame(cat_preds_test, columns = ["CatBoost"])], axis = 1)

print(f"Meta-Train shape: {meta_train.shape}")
print(f"Meta-Test shape: {meta_test.shape}")
print(f"Target shape: {y.shape}")
print(f"Test data shape: {X_test.shape}")

def objective(trial):
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'booster': 'gbtree', 'tree_method': 'hist',
              'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
              'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
              'max_depth': trial.suggest_int('max_depth', 3, 12),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
              'gamma': trial.suggest_float('gamma', 0.0, 0.5),
              'subsample': trial.suggest_float('subsample', 0.6, 1.0),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1.0, log=True),
              'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
              'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
    
    scores = []
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(meta_train, y)):
        X_train, y_train = meta_train.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = meta_train.iloc[val_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=1000)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, val_preds)
        scores.append(score)
    
    return np.mean(scores)
'''
study = optuna.create_study(direction = 'minimize', sampler = TPESampler(seed = 42), pruner = optuna.pruners.MedianPruner(n_warmup_steps = 5))
study.optimize(objective, n_trials = 25, timeout = 3600)
print("Best trial:")
trial = study.best_trial
print(f"LogLoss: {trial.value:.5f}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
'''

meta_preds_val = np.zeros(len(X))
meta_preds_test = np.zeros(len(X_test))
meta_models = []
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

for fold, (train_idx, val_idx) in enumerate(cv.split(meta_train, y)):
    print(f"Fold Meta-Model - {fold+1}")
    X_train, y_train = meta_train.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = meta_train.iloc[val_idx], y.iloc[val_idx]
    
    xgb_params = {'n_estimators': 1632, 'learning_rate': 0.00333485601627724, 
                  'max_depth': 4, 'min_child_weight': 1, 'gamma': 0.35835286604341793, 
                  'subsample': 0.6072209567705501, 'colsample_bytree': 0.7038278378586684, 
                  'reg_alpha': 4.967709157079716e-06, 'reg_lambda': 9.359255442147762e-06, 
                  'scale_pos_weight': 1.0364426903694701}
    xgb_params_2 = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'booster': 'gbtree', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
    meta_model = xgb.XGBClassifier(**xgb_params, **xgb_params_2)
    meta_model.fit(X_train, y_train)
    meta_preds_val[val_idx] = meta_model.predict_proba(X_val)[:, 1]
    meta_preds_test += meta_model.predict_proba(meta_test)[:, 1] / 5
    meta_models.append(meta_model)
    print(f"Fold {fold+1} Val LogLoss: {round(log_loss(y_val, meta_preds_val[val_idx]), 4)}")

overall_oof_logloss = log_loss(y, meta_preds_val)
print(f"\nMeta-Model Overall Val LogLoss: {round(overall_oof_logloss, 4)}")


#final_test_preds = (lgb_test_preds + cb_test_preds + xgb_test_preds) / 3 # Simple Averaging Ensemble Example
MINT_ID = 'mint'
TARGET = 'has_graduated'
submission_df = pd.DataFrame({MINT_ID: test_processed[MINT_ID],
                              TARGET: cat_preds_test})

submission_df[TARGET] = np.clip(submission_df[TARGET], 0.0001, 0.9999)
submission_df.to_csv(SUBMISSION_FILE, index=False)

print(f"Submission file saved to: {SUBMISSION_FILE}")
print("\nSubmission file head:")
print(submission_df.head())
print("\nScript finished successfully!")

