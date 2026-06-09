import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer

from catboost import CatBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb


from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve,log_loss)

from itertools import combinations
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

print("ok")


def add_simple_features(df):

    # the feature process copy from @molozhenko
    # Basic binary features
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)
    df['balance_zero'] = (df['balance'] == 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['duration_short'] = (df['duration'] < 100).astype(int)
    df['duration_medium'] = ((df['duration'] >= 100) & (df['duration'] <= 300)).astype(int)
    df['duration_long'] = (df['duration'] > 300).astype(int)
    df['duration_very_long'] = (df['duration'] > 600).astype(int)
    df['campaign_single'] = (df['campaign'] == 1).astype(int)
    df['campaign_multiple'] = (df['campaign'] > 2).astype(int)
    df['campaign_high'] = (df['campaign'] > 5).astype(int)
    df['campaign_very_high'] = (df['campaign'] > 10).astype(int)
    
    # Enhanced numerical transformations
    df['log_duration'] = np.log1p(df['duration'])
    df['log_campaign'] = np.log1p(df['campaign'])
    df['log_previous'] = np.log1p(df['previous'])
    df['sqrt_age'] = np.sqrt(df['age'])
    df['sqrt_duration'] = np.sqrt(df['duration'])
    df['log_balance'] = np.sign(df['balance']) * np.log1p(np.abs(df['balance']))
    df['balance_abs'] = np.abs(df['balance'])
    df['balance_abs_log'] = np.log1p(df['balance_abs'])
    
    # Power features
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['duration_squared'] = df['duration'] ** 2
    df['duration_cubed'] = df['duration'] ** 3
    df['campaign_squared'] = df['campaign'] ** 2
    df['balance_squared'] = df['balance'] ** 2
    
    # More detailed binning
    df['age_bin_5'] = pd.cut(df['age'], bins=5, labels=False)
    df['age_bin_10'] = pd.cut(df['age'], bins=10, labels=False)
    df['age_bin_detailed'] = pd.cut(df['age'], bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75, 100], labels=False)
    
    # Quantile binning
    df['duration_qbin_5'] = pd.qcut(df['duration'], q=5, labels=False, duplicates='drop')
    df['duration_qbin_10'] = pd.qcut(df['duration'], q=10, labels=False, duplicates='drop')
    df['duration_qbin_20'] = pd.qcut(df['duration'], q=20, labels=False, duplicates='drop')
    
    df['balance_qbin_5'] = pd.qcut(df['balance'], q=5, labels=False, duplicates='drop')
    df['balance_qbin_10'] = pd.qcut(df['balance'], q=10, labels=False, duplicates='drop')
    df['balance_qbin_20'] = pd.qcut(df['balance'], q=20, labels=False, duplicates='drop')
    
    df['campaign_bin'] = pd.cut(df['campaign'], bins=[-1, 1, 2, 3, 5, 10, 100], labels=False)
    
    # Extended feature interactions
    df['age_duration'] = df['age'] * df['duration']
    df['age_campaign'] = df['age'] * df['campaign']
    df['age_balance'] = df['age'] * df['balance']
    df['age_previous'] = df['age'] * df['previous']
    df['duration_campaign'] = df['duration'] * df['campaign']
    df['duration_balance'] = df['duration'] * df['balance']
    df['duration_previous'] = df['duration'] * df['previous']
    df['campaign_balance'] = df['campaign'] * df['balance']
    df['campaign_previous'] = df['campaign'] * df['previous']
    df['balance_previous'] = df['balance'] * df['previous']
    
    # Three-way interactions
    df['age_duration_campaign'] = df['age'] * df['duration'] * df['campaign']
    df['age_balance_duration'] = df['age'] * df['balance'] * df['duration']
    df['duration_campaign_balance'] = df['duration'] * df['campaign'] * df['balance']
    
    # Enhanced ratios
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_per_age'] = df['duration'] / (df['age'] + 1)
    df['campaign_per_age'] = df['campaign'] / (df['age'] + 1)
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['balance_per_campaign'] = df['balance'] / (df['campaign'] + 1)
    df['previous_per_campaign'] = df['previous'] / (df['campaign'] + 1)
    df['age_per_duration'] = df['age'] / (df['duration'] + 1)
    df['campaign_per_duration'] = df['campaign'] / (df['duration'] + 1)
    
    # Group statistics
    df['balance_rank'] = df['balance'].rank(pct=True)
    df['duration_rank'] = df['duration'].rank(pct=True)
    df['age_rank'] = df['age'].rank(pct=True)
    df['campaign_rank'] = df['campaign'].rank(pct=True)
    
    # Contact-related features
    df['contact_success_rate'] = df['previous'] / (df['campaign'] + df['previous'] + 1)
    df['pdays_binned'] = pd.cut(df['pdays'], bins=[-2, -1, 0, 30, 60, 120, 200, 400, 1000], labels=False)
    df['has_pdays'] = (df['pdays'] != -1).astype(int)
    df['pdays_recent'] = ((df['pdays'] > 0) & (df['pdays'] <= 30)).astype(int)
    df['pdays_old'] = (df['pdays'] > 180).astype(int)
    
    # Seasonal features
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)
    df['quarter'] = ((df['month_num'] - 1) // 3) + 1
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['is_spring'] = df['month'].isin(['mar', 'apr', 'may']).astype(int)
    df['is_summer'] = df['month'].isin(['jun', 'jul', 'aug']).astype(int)
    df['is_autumn'] = df['month'].isin(['sep', 'oct', 'nov']).astype(int)
    df['is_winter'] = df['month'].isin(['dec', 'jan', 'feb']).astype(int)
    df['is_peak_season'] = df['month'].isin(['may', 'jun', 'jul', 'aug', 'nov']).astype(int)
    
    # Day of month features
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['is_month_start'] = (df['day'] <= 5).astype(int)
    df['is_month_end'] = (df['day'] >= 25).astype(int)
    df['is_month_middle'] = ((df['day'] > 10) & (df['day'] <= 20)).astype(int)
    
    return df


print("ok")


# åŠ è½½æ•°æ�®
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv').drop(columns=['id'])
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv').drop(columns=['id'])
original_data = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=";")

# ç›®æ ‡å�˜é‡�å¤„ç�†
original_data['y'] = (original_data['y'] == "yes").astype(int)
TARGET = 'y'

# æ•°æ�®å¢�å¼ºï¼ˆå�¯é€‰ï¼‰
for k in range(1):
    train = pd.concat([train, original_data], ignore_index=True)

#train = add_simple_features(train)
#test = add_simple_features(test)

# ====================================================================
# é«˜çº§ç‰¹å¾�å·¥ç¨‹ï¼ˆæ— éªŒè¯�é›†ç‰ˆæœ¬ï¼‰
# ====================================================================

print("åˆ›å»ºç‰¹å¾�ç»„å�ˆ...")
columns = [c for c in train.columns if c != TARGET]  # æ�’é™¤ç›®æ ‡å�˜é‡�

# åˆ›å»ºç‰¹å¾�ç»„å�ˆï¼ˆ2ä¸ªç‰¹å¾�çš„ç»„å�ˆï¼‰
for r in [2]:
    for cols in tqdm(list(combinations(columns, r))):
        name = '-'.join(cols)
        
        # åˆ›å»ºç»„å�ˆç‰¹å¾�ï¼ˆå­—ç¬¦ä¸²è¿�æ�¥ï¼‰
        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)
        
        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)
        
        # å¯¹ç»„å�ˆç‰¹å¾�è¿›è¡Œå› å­�åŒ–ç¼–ç �ï¼ˆç¡®ä¿�è®­ç»ƒ/æµ‹è¯•é›†ç¼–ç �ä¸€è‡´ï¼‰
        combined = pd.concat([train[name], test[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):]

# é«˜çº§ç›®æ ‡ç¼–ç �å‡½æ•°ï¼ˆæ— éªŒè¯�é›†ç‰ˆæœ¬ï¼‰
def target_encode_no_valid(train, test, col, target=TARGET, smooth=3):
    """æ— éªŒè¯�é›†çš„ç›®æ ‡ç¼–ç �ï¼ˆç›´æ�¥åœ¨è®­ç»ƒé›†ä¸Šè®¡ç®—ï¼‰"""
    col_name = '_'.join(col) if isinstance(col, (list, tuple)) else col
    global_mean = train[target].mean()
    
    # è®¡ç®—å¸¦å¹³æ»‘çš„ç»Ÿè®¡é‡�
    stats = train[[col] + [target]].groupby(col)[target].agg(['mean', 'count'])
    stats['TE_tmp'] = ((stats['mean'] * stats['count']) + (global_mean * smooth)) / (stats['count'] + smooth)
    
    # ç¼–ç �è®­ç»ƒé›†
    train = train.merge(stats[['TE_tmp']], how='left', left_on=col, right_index=True)
    train[f'TE_MEAN_{col_name}'] = train['TE_tmp'].fillna(global_mean).values.astype('float32')
    train = train.drop('TE_tmp', axis=1)
    
    # ç¼–ç �æµ‹è¯•é›†
    test = test.merge(stats[['TE_tmp']], how='left', left_on=col, right_index=True)
    test[f'TE_MEAN_{col_name}'] = test['TE_tmp'].fillna(global_mean).values.astype('float32')
    test = test.drop('TE_tmp', axis=1)
    
    return train, test

# è®¡æ•°ç¼–ç �å‡½æ•°ï¼ˆæ— éªŒè¯�é›†ç‰ˆæœ¬ï¼‰
def count_encode_no_valid(train, test, col):
    """æ— éªŒè¯�é›†çš„è®¡æ•°ç¼–ç �"""
    counts = train[col].value_counts()
    train[f'CE_{col}'] = np.log1p(train[col].map(counts))
    test[f'CE_{col}'] = np.log1p(test[col].map(counts).fillna(0))
    return train, test

# å¯¹åˆ†ç±»åˆ—åº”ç”¨ç¼–ç �
cat_cols = ['job', 'marital', 'education', 'default', 
            'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    train, test = target_encode_no_valid(train, test, col)
    train, test = count_encode_no_valid(train, test, col)

print(train.columns)
train = add_simple_features(train)
test = add_simple_features(test)
print(train.columns)

# ====================================================================
# å‡†å¤‡æœ€ç»ˆç‰¹å¾�å’Œæ ‡ç­¾ï¼ˆæ— éªŒè¯�é›†ï¼‰
# ====================================================================
FEATURES = [c for c in train.columns if c != TARGET]  # æ‰€æœ‰ç‰¹å¾�åˆ—
X = train[FEATURES]  # ç›´æ�¥ä½¿ç”¨å…¨éƒ¨è®­ç»ƒæ•°æ�®
y = train[TARGET]    # ç›®æ ‡å�˜é‡�

# åˆ†ç±»å�˜é‡�ç¼–ç �ï¼ˆä½¿ç”¨ColumnTransformerï¼‰
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = list(set(FEATURES) - set(num_cols))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(), cat_cols)
    ],
    remainder='drop'  # ä¸¢å¼ƒæœªæŒ‡å®šçš„åˆ—
)

# è¾“å‡ºæ ¼å¼�æ£€æŸ¥
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Test features shape: {test[FEATURES].shape}")


%%time
FOLDS = 10
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train)) 
test_preds_proba = np.zeros(len(test)) 
fold_cms = []
val_losses = []  

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nğŸ”� Fold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # X_train_scaled =X_train
    # X_val_scaled = X_val
    # test_scaled=test
    X_train_scaled =preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    test_scaled=preprocessor.transform(test)

    # model = XGBClassifier(
    #     n_estimators=20000,               # æ ‘çš„æ•°é‡� â†’ 8000ï¼ˆä¸�å�Ÿé…�ç½®ä¸€è‡´ï¼‰
    #     max_leaves=127,                  # æœ€å¤§å�¶å­�æ•° â†’ æ–°å¢�ï¼ˆå�Ÿé…�ç½®æ— ï¼‰
    #     min_child_weight=1.5,            # å­�èŠ‚ç‚¹æ‰€éœ€æœ€å°�æ�ƒé‡�å’Œ â†’ 1.5ï¼ˆå�Ÿä¸º5ï¼‰
    #     max_depth=0,                     # æœ€å¤§æ·±åº¦ â†’ 0ï¼ˆä½¿ç”¨max_leavesæ�§åˆ¶å¤�æ�‚åº¦ï¼‰
    #     grow_policy='lossguide',         # æ ‘ç”Ÿé•¿ç­–ç•¥ â†’ æ�Ÿå¤±æŒ‡å¯¼ï¼ˆå�Ÿé…�ç½®æ— ï¼‰
    #     learning_rate=0.008,             # å­¦ä¹ ç�‡ â†’ 0.008ï¼ˆå�Ÿä¸º0.05ï¼Œæ˜¾è‘—é™�ä½�ï¼‰
    #     tree_method='hist',              # æ ‘æ�„å»ºæ–¹æ³• â†’ ç›´æ–¹å›¾ä¼˜åŒ–ï¼ˆå�Ÿé…�ç½®æ— ï¼‰
    #     subsample=0.85,                  # æ ·æœ¬é‡‡æ ·æ¯”ä¾‹ â†’ 0.85ï¼ˆå�Ÿä¸º0.86ï¼‰
    #     colsample_bylevel=0.7,           # æ¯�å±‚ç‰¹å¾�é‡‡æ ·æ¯”ä¾‹ â†’ æ–°å¢�ï¼ˆå�Ÿé…�ç½®æ— ï¼‰
    #     colsample_bytree=0.75,           # æ¯�æ£µæ ‘ç‰¹å¾�é‡‡æ ·æ¯”ä¾‹ â†’ 0.75ï¼ˆå�Ÿä¸º0.86ï¼‰
    #     colsample_bynode=0.85,           # æ¯�ä¸ªèŠ‚ç‚¹ç‰¹å¾�é‡‡æ ·æ¯”ä¾‹ â†’ æ–°å¢�ï¼ˆå�Ÿé…�ç½®æ— ï¼‰
    #     sampling_method='gradient_based', # é‡‡æ ·æ–¹æ³• â†’ æ¢¯åº¦å¯¼å�‘ï¼ˆå�Ÿé…�ç½®æ— ï¼‰
    #     reg_alpha=2.5,                   # L1æ­£åˆ™åŒ– â†’ 2.5ï¼ˆå�Ÿä¸º3ï¼‰
    #     reg_lambda=0.8,                  # L2æ­£åˆ™åŒ– â†’ 0.8ï¼ˆå�Ÿä¸º1.4ï¼‰
    #     early_stopping_rounds=100,
    #     enable_categorical=True,         # å�¯ç”¨ç±»åˆ«ç‰¹å¾�æ”¯æŒ� â†’ Trueï¼ˆä¸€è‡´ï¼‰
    #     max_cat_to_onehot=1,             # ç±»åˆ«ç‰¹å¾�ç‹¬çƒ­ç¼–ç �é˜ˆå€¼ â†’ æ–°å¢�ï¼ˆå�Ÿé…�ç½®æ— ï¼‰
    #     device='cuda',                   # ä½¿ç”¨GPUåŠ é€Ÿ â†’ cudaï¼ˆä¸€è‡´ï¼‰
    #     n_jobs=-1,                       # å¹¶è¡Œçº¿ç¨‹æ•° â†’ -1ï¼ˆä¸€è‡´ï¼‰
    #     random_state=42,                 # éš�æœºç§�å­� â†’ 42ï¼ˆå�Ÿä¸º42+idxï¼Œæ­¤å¤„éœ€å›ºå®šï¼‰
    #     verbosity=0,                     # æ—¥å¿—é�™é»˜ â†’ æ–°å¢�ï¼ˆå�Ÿé…�ç½®æ— ï¼‰
    #     objective='binary:logistic',     # ç›®æ ‡å‡½æ•° â†’ äºŒåˆ†ç±»é€»è¾‘å›�å½’ï¼ˆä¸€è‡´ï¼‰
    #     eval_metric='logloss'                # è¯„ä¼°æŒ‡æ ‡ â†’ AUCï¼ˆå�Ÿä¸ºloglossï¼‰
    # )
    
    # # 5. Fit with early stopping    
    # model.fit(
    #     X_train_scaled, y_train,
    #     eval_set=[(X_val_scaled, y_val)],
    #     verbose=500
    # )
    # 'LGBM': LGBMClassifier(**{'random_state': Config.state,
    #                           'early_stopping_round': Config.early_stop,
    #                           'verbose': -1,
    #                           'n_estimators': 10000,
    #                           'metric': 'AUC',
    #                           'objective': 'binary',
    #                           'max_depth': 16,
    #                           'learning_rate': 0.007366917567300051,
    #                           'min_child_samples': 164,
    #                           'subsample': 0.9022880020285295,
    #                           'colsample_bytree': 0.4213201532077694,
    #                           'num_leaves': 122, 
    #                           'reg_alpha': 1.083996192298843,
    #                           'reg_lambda': 0.0700057221912873
    #                           }),
    # model = lgb.LGBMClassifier(
    #     num_leaves=122,          # æ›¿ä»£ max_depthï¼Œæ�§åˆ¶æ ‘å¤�æ�‚åº¦
    #     max_depth=16,           # ä¸�é™�åˆ¶æ·±åº¦ï¼ˆç”± num_leaves æ�§åˆ¶ï¼‰
    #     subsample=0.9022880020285295,         # ç±»ä¼¼ XGBoost çš„ subsample
    #     colsample_bytree=0.4213201532077694,  # ç±»ä¼¼ XGBoost çš„ colsample_bytree
    #     n_estimators=10000,      # æœ€å¤§è¿­ä»£æ¬¡æ•°
    #     learning_rate=0.007366917567300051,     # å­¦ä¹ ç�‡
    #     min_child_samples=164,    # æ›¿ä»£ min_child_weight
    #     reg_alpha=1.083996192298843,            # L1 æ­£åˆ™åŒ–
    #     reg_lambda=0.0700057221912873,         # L2 æ­£åˆ™åŒ–
    #     objective='binary',     # äºŒåˆ†ç±»ä»»åŠ¡
    #     random_state=42,
    #     n_jobs=-1,
    #     device='gpu',           # ä½¿ç”¨ GPU åŠ é€Ÿ
    #     metric='AUC',
    #     #class_weight = {0:1,1:3}
    # )
    # model = lgb.LGBMClassifier(
    # num_leaves=255,          # å¢�åŠ å�¶å­�èŠ‚ç‚¹æ•°
    # max_depth=-1,           # ä¸�é™�åˆ¶æ·±åº¦ï¼ˆç”± num_leaves æ�§åˆ¶ï¼‰
    # subsample=0.9,          # é€‚å½“é™�ä½�é‡‡æ ·æ¯”ä¾‹
    # colsample_bytree=0.8,   # é€‚å½“å¢�åŠ ç‰¹å¾�é‡‡æ ·æ¯”ä¾‹
    # n_estimators=10000,      # å‡�å°‘è¿­ä»£æ¬¡æ•°ï¼ˆé…�å�ˆ early_stoppingï¼‰
    # learning_rate=0.05,     # å¢�å¤§å­¦ä¹ ç�‡
    # min_child_samples=20,   # æ”¾å®½åˆ†è£‚æ�¡ä»¶
    # reg_alpha=0.1,          # é™�ä½� L1 æ­£åˆ™åŒ–
#     reg_lambda=0.1,         # é™�ä½� L2 æ­£åˆ™åŒ–
#     objective='binary',
#     random_state=42,
#     n_jobs=-1,
#     device='gpu'
# )

    
    model = lgb.LGBMClassifier(
        num_leaves=31,          # æ›¿ä»£ max_depthï¼Œæ�§åˆ¶æ ‘å¤�æ�‚åº¦
        max_depth=-1,           # ä¸�é™�åˆ¶æ·±åº¦ï¼ˆç”± num_leaves æ�§åˆ¶ï¼‰
        subsample=0.86,         # ç±»ä¼¼ XGBoost çš„ subsample
        colsample_bytree=0.86,  # ç±»ä¼¼ XGBoost çš„ colsample_bytree
        n_estimators=20000,      # æœ€å¤§è¿­ä»£æ¬¡æ•°
        learning_rate=0.05,     # å­¦ä¹ ç�‡
        min_child_samples=5,    # æ›¿ä»£ min_child_weight
        reg_alpha=3,            # L1 æ­£åˆ™åŒ–
        reg_lambda=1.4,         # L2 æ­£åˆ™åŒ–
        objective='binary',     # äºŒåˆ†ç±»ä»»åŠ¡
        random_state=42,
        n_jobs=-1,
        device='gpu'           # ä½¿ç”¨ GPU åŠ é€Ÿ
        #class_weight = {0:1,1:3}
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(500), lgb.log_evaluation(period=1000)]
    )
    # å®šä¹‰æ¨¡å�‹
    # model = CatBoostClassifier(
    #     iterations=20000,
    #     learning_rate=0.1,
    #     depth=8,
    #     l2_leaf_reg=1.4,
    #     min_data_in_leaf=21,
    #     random_state=42,
    #     task_type="GPU",
    #     early_stopping_rounds=200,
    #     eval_metric="Logloss",
    #     verbose=1000
    # )
     
    # # è®­ç»ƒæ¨¡å�‹
    # model.fit(
    #     X_train_scaled, y_train,
    #     eval_set=[(X_val_scaled, y_val)],
    #     use_best_model=True
    # )



   #  Predict probabilities for validation
    val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_pred_proba >= 0.5).astype(int)


    # Calculate log loss for the validation set
    val_loss = log_loss(y_val, val_pred_proba)
    val_losses.append(val_loss)
    print(f"Fold {fold} validation log loss: {val_loss:.4f}")


    # Store confusion matrix for this fold
    fold_cms.append(confusion_matrix(y_val, val_pred))
    #  Store OOF predictions
    oof[val_idx] = val_pred_proba
    test_preds_proba += model.predict_proba(test_scaled)[:, 1]

# Calculate average validation loss
avg_val_loss = np.mean(val_losses)
print(f"\n Average validation log loss: {avg_val_loss:.4f}")


#  Compute aggregated confusion matrix 
cm = np.sum(fold_cms, axis=0)
test_preds_proba /= FOLDS

# Compute OOF AUC
oof_auc = roc_auc_score(y, oof)
print(f"OOF AUC: {oof_auc:.4f}")


# Calculate evaluation metrics
roc_auc = roc_auc_score(y, oof) 
fpr, tpr, _ = roc_curve(y, oof)  

# False Positive Rate and True Positive Rate for ROC curve
plt.figure(figsize=(12, 5))

# --- Plot 1: Confusion Matrix ---
plt.subplot(1, 2, 1)  
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# --- Plot 2: ROC Curve ---
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])  # Limit x-axis range (FPR)
plt.ylim([0.0, 1.05])  # Limit y-axis range (TPR)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right") 

plt.tight_layout()
plt.show()


#ä¿�å­˜ç»“æ�œ
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission['y']=test_preds_proba
submission.to_csv('submission_cv_lgbm_advanced_feature_10_8.csv',index=False)
submission.head()


