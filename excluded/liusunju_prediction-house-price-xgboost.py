# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


parameter = {
    'SEED': 42,
    'N_SPLITS': 10,  # å¢�åŠ foldæ•¸é‡�
    'ALPHA_LOWER': 0.05,         # èª¿æ•´åˆ†ä½�æ•¸
    'ALPHA_UPPER': 0.95,         
    'DATA_PATH': '/kaggle/input/prediction-interval-competition-ii-house-price/',
    'OUTPUT_PATH': './',
    
    # XGBoostå�ƒæ•¸å„ªåŒ–
    'N_ESTIMATORS': 2000,        # å¢�åŠ æ¨¹çš„æ•¸é‡�
    'MAX_DEPTH': 8,              # å¢�åŠ æ·±åº¦
    'LEARNING_RATE': 0.02,       # é™�ä½�å­¸ç¿’ç�‡
    'SUBSAMPLE': 0.9,            # æ��é«˜æ�¡æ¨£ç�‡
    'COLSAMPLE_BYTREE': 0.8,     
    'MIN_CHILD_WEIGHT': 1,       # é™�ä½�æœ€å°�è‘‰å­�æ¬Šé‡�
    'REG_ALPHA': 0.1,            # L1æ­£å‰‡åŒ–
    'REG_LAMBDA': 0.1,           # L2æ­£å‰‡åŒ–
    'GAMMA': 0.1,                # æœ€å°�åˆ†å‰²æ��å¤±
    'EARLY_STOPPING_ROUNDS': 100
}

print("âœ… å¢�å¼·ç‰ˆå�ƒæ•¸è¨­å®šå®Œæˆ�")


train_df = pd.read_csv(f"{parameter['DATA_PATH']}dataset.csv")
test_df = pd.read_csv(f"{parameter['DATA_PATH']}test.csv")

print("âœ… è³‡æ–™è®€å�–å®Œæˆ�")
print(f"è¨“ç·´è³‡æ–™: {train_df.shape}")
print(f"æ¸¬è©¦è³‡æ–™: {test_df.shape}")


def advanced_feature_engineer(df):
    df = df.copy()
    
    # è™•ç�†æ—¥æœŸç‰¹å¾µ
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['year'] = df['sale_date'].dt.year
        df['month'] = df['sale_date'].dt.month
        df['dayofweek'] = df['sale_date'].dt.dayofweek
        df['quarter'] = df['sale_date'].dt.quarter
        df['is_weekend'] = df['sale_date'].dt.dayofweek.isin([5, 6]).astype(int)
        df = df.drop('sale_date', axis=1)
    
    # æ•¸å€¼ç‰¹å¾µè™•ç�†
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['id', 'sale_price']:
            # è™•ç�†ç•°å¸¸å€¼ (ä½¿ç”¨IQRæ–¹æ³•)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    # é¡�åˆ¥ç‰¹å¾µç·¨ç¢¼
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # å‰µå»ºè¡�ç”Ÿç‰¹å¾µ
    if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
        df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
    
    if 'total_rooms' in df.columns and 'property_tax' in df.columns:
        df['tax_per_room'] = df['property_tax'] / (df['total_rooms'] + 1)
    
    # å°�æ•¸è½‰æ�›é«˜å��æ–œç‰¹å¾µ
    for col in numeric_cols:
        if col not in ['id', 'sale_price'] and col in df.columns:
            if df[col].skew() > 2:
                df[f'{col}_log'] = np.log1p(df[col] + 1)
    
    return df

# æ‡‰ç”¨å¢�å¼·ç‰ˆç‰¹å¾µå·¥ç¨‹
train_df = advanced_feature_engineer(train_df)
test_df = advanced_feature_engineer(test_df)

print("âœ… å¢�å¼·ç‰ˆç‰¹å¾µå·¥ç¨‹å®Œæˆ�")


features = [col for col in train_df.columns if col not in ['id', 'sale_price']]
X = train_df[features]
y = train_df['sale_price']
X_test = test_df[features]

# ç‰¹å¾µæ¨™æº–åŒ–
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"âœ… ç‰¹å¾µæº–å‚™å®Œæˆ�ï¼Œç‰¹å¾µæ•¸é‡�: {len(features)}")


kf = KFold(n_splits=parameter['N_SPLITS'], shuffle=True, random_state=parameter['SEED'])

# åˆ�å§‹åŒ–é �æ¸¬é™£åˆ—
oof_lower = np.zeros(len(y))
oof_upper = np.zeros(len(y))
test_lower = np.zeros(len(test_df))
test_upper = np.zeros(len(test_df))

print("âœ… é–‹å§‹å¤šé‡�æ¨¡å�‹è¨“ç·´")


for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\nğŸŸ  è¨“ç·´ Fold {fold+1}/{parameter['N_SPLITS']}")
    
    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # ä¸‹ç•Œæ¨¡å�‹å„ªåŒ–
    model_lower = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=parameter['ALPHA_LOWER'],
        tree_method='hist',
        n_estimators=parameter['N_ESTIMATORS'],
        learning_rate=parameter['LEARNING_RATE'],
        max_depth=parameter['MAX_DEPTH'],
        subsample=parameter['SUBSAMPLE'],
        colsample_bytree=parameter['COLSAMPLE_BYTREE'],
        min_child_weight=parameter['MIN_CHILD_WEIGHT'],
        reg_alpha=parameter['REG_ALPHA'],
        reg_lambda=parameter['REG_LAMBDA'],
        gamma=parameter['GAMMA'],
        random_state=parameter['SEED'],
        verbosity=0
    )
    
    # ä¸Šç•Œæ¨¡å�‹å„ªåŒ–
    model_upper = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=parameter['ALPHA_UPPER'],
        tree_method='hist',
        n_estimators=parameter['N_ESTIMATORS'],
        learning_rate=parameter['LEARNING_RATE'],
        max_depth=parameter['MAX_DEPTH'],
        subsample=parameter['SUBSAMPLE'],
        colsample_bytree=parameter['COLSAMPLE_BYTREE'],
        min_child_weight=parameter['MIN_CHILD_WEIGHT'],
        reg_alpha=parameter['REG_ALPHA'],
        reg_lambda=parameter['REG_LAMBDA'],
        gamma=parameter['GAMMA'],
        random_state=parameter['SEED'],
        verbosity=0
    )
    
    # è¨“ç·´æ¨¡å�‹
    model_lower.fit(X_train, y_train)
    model_upper.fit(X_train, y_train)
    
    # é �æ¸¬
    oof_lower[val_idx] = model_lower.predict(X_val)
    oof_upper[val_idx] = model_upper.predict(X_val)
    
    test_lower += model_lower.predict(X_test_scaled) / parameter['N_SPLITS']
    test_upper += model_upper.predict(X_test_scaled) / parameter['N_SPLITS']
    
    print(f"âœ… Fold {fold+1} å®Œæˆ�")


def detailed_evaluation(y_true, lower, upper, alpha=0.1):
    # Winkler Score
    width = np.mean(upper - lower)
    below = y_true < lower
    above = y_true > upper
    penalty = np.mean((2 / alpha) * (lower - y_true) * below)
    penalty += np.mean((2 / alpha) * (y_true - upper) * above)
    winkler = width + penalty
    
    # è¦†è“‹ç�‡
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    # å�€é–“å¯¬åº¦çµ±è¨ˆ
    interval_width = upper - lower
    
    return {
        'winkler_score': winkler,
        'coverage': coverage,
        'mean_width': np.mean(interval_width),
        'median_width': np.median(interval_width),
        'below_rate': np.mean(below),
        'above_rate': np.mean(above)
    }

# è¨ˆç®—è©•ä¼°çµ�æ�œ
results = detailed_evaluation(y, oof_lower, oof_upper)

print(f"\nğŸ“Š è©³ç´°è©•ä¼°çµ�æ�œ:")
print(f"Winkler Score: {results['winkler_score']:.4f}")
print(f"è¦†è“‹ç�‡: {results['coverage']:.4f} (ç›®æ¨™: 0.90)")
print(f"å¹³å�‡å�€é–“å¯¬åº¦: {results['mean_width']:.2f}")
print(f"ä½�æ–¼ä¸‹ç•Œæ¯”ä¾‹: {results['below_rate']:.4f}")
print(f"é«˜æ–¼ä¸Šç•Œæ¯”ä¾‹: {results['above_rate']:.4f}")


# ç¢ºä¿�å�€é–“å�ˆç�†æ€§
mask = test_lower > test_upper
test_lower[mask], test_upper[mask] = test_upper[mask], test_lower[mask]

# èª¿æ•´å�€é–“å¯¬åº¦ (å¦‚æ�œè¦†è“‹ç�‡ä¸�è¶³)
if results['coverage'] < 0.88:
    adjustment = 1.1  # æ“´å¤§å�€é–“10%
    center = (test_lower + test_upper) / 2
    width = test_upper - test_lower
    test_lower = center - width * adjustment / 2
    test_upper = center + width * adjustment / 2
    print(f"ğŸ”§ å·²èª¿æ•´å�€é–“å¯¬åº¦ ({adjustment}x)")


submission = pd.DataFrame({
    'id': test_df['id'],
    'pi_lower': test_lower,
    'pi_upper': test_upper
})

# æœ€çµ‚æª¢æŸ¥
submission['pi_lower'] = np.minimum(submission['pi_lower'], submission['pi_upper'])
submission.to_csv(f"{parameter['OUTPUT_PATH']}submission_improved.csv", index=False)

print("\nâœ… æ”¹é€²ç‰ˆæ��äº¤æª”å·²å»ºç«‹")
print(f"å¹³å�‡å�€é–“å¯¬åº¦: {np.mean(submission['pi_upper'] - submission['pi_lower']):.2f}")
print(submission.head())

