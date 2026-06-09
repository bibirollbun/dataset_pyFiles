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

print("âœ… å¥—ä»¶å·²è¼‰å…¥")


parameter = {
    'SEED': 42,
    'N_SPLITS': 5,
    'ALPHA_LOWER': 0.07,         # ä¸‹ç•Œåˆ†ä½�æ•¸
    'ALPHA_UPPER': 0.93,         # ä¸Šç•Œåˆ†ä½�æ•¸
    'DATA_PATH': '/kaggle/input/prediction-interval-competition-ii-house-price/',  # è·¯å¾‘å·²æ›´æ–°
    'OUTPUT_PATH': './',
    'N_ESTIMATORS': 1000,
    'MAX_DEPTH': 5,
    'LEARNING_RATE': 0.05,
    'SUBSAMPLE': 0.8,
    'COLSAMPLE_BYTREE': 0.7,
    'MIN_CHILD_WEIGHT': 5
}

print("âœ… å�ƒæ•¸è¨­å®šå®Œæˆ� (ä½¿ç”¨ parameter dict)")


train_df = pd.read_csv(f"{parameter['DATA_PATH']}dataset.csv")
test_df = pd.read_csv(f"{parameter['DATA_PATH']}test.csv")

print("âœ… è³‡æ–™è®€å�–å®Œæˆ�")
print("è¨“ç·´è³‡æ–™ç­†æ•¸:", len(train_df))
print("æ¸¬è©¦è³‡æ–™ç­†æ•¸:", len(test_df))


def feature_engineer(df):
    df = df.copy()
    
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['year'] = df['sale_date'].dt.year
        df['month'] = df['sale_date'].dt.month
        df['dayofweek'] = df['sale_date'].dt.dayofweek
        df = df.drop('sale_date', axis=1)
        
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
        
    return df

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

print("âœ… ç‰¹å¾µå·¥ç¨‹å®Œæˆ�")


features = [col for col in train_df.columns if col not in ['id', 'sale_price']]
X = train_df[features]
y = train_df['sale_price']
X_test = test_df[features]

print("âœ… ç‰¹å¾µæº–å‚™å®Œæˆ�")


kf = KFold(n_splits=parameter['N_SPLITS'], shuffle=True, random_state=parameter['SEED'])


oof_lower = np.zeros(len(y))
oof_upper = np.zeros(len(y))
test_lower = np.zeros(len(test_df))
test_upper = np.zeros(len(test_df))

print("âœ… åˆ�å§‹åŒ–å®Œæˆ�ï¼Œé–‹å§‹è¨“ç·´")


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nğŸŸ  Fold {fold+1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # è¨“ç·´ä¸‹ç•Œæ¨¡å�‹ (5% åˆ†ä½�æ•¸)
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
        random_state=parameter['SEED'],
        verbosity=0
    )
    model_lower.fit(X_train, y_train)
    
    # è¨“ç·´ä¸Šç•Œæ¨¡å�‹ (95% åˆ†ä½�æ•¸)
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
        random_state=parameter['SEED'],
        verbosity=0
    )
    model_upper.fit(X_train, y_train)
    
    # é©—è­‰é›†é �æ¸¬
    oof_lower[val_idx] = model_lower.predict(X_val)
    oof_upper[val_idx] = model_upper.predict(X_val)
    
    # æ¸¬è©¦é›†é �æ¸¬
    test_lower += model_lower.predict(X_test) / parameter['N_SPLITS']
    test_upper += model_upper.predict(X_test) / parameter['N_SPLITS']
    
    print(f"âœ… Fold {fold+1} å®Œæˆ�")


def winkler_score(y_true, lower, upper, alpha=0.1):
    score = np.mean(upper - lower)
    below = y_true < lower
    above = y_true > upper
    score += np.mean((2 / alpha) * (lower - y_true) * below)
    score += np.mean((2 / alpha) * (y_true - upper) * above)
    return score

score = winkler_score(y, oof_lower, oof_upper)
print(f"\nğŸ“Š OOF Winkler Score: {score:.2f}")


submission = pd.DataFrame({
    'id': test_df['id'],
    'pi_lower': test_lower,
    'pi_upper': test_upper
})

# ç¢ºä¿�ä¸‹ç•Œ <= ä¸Šç•Œ
submission['pi_lower'] = np.minimum(submission['pi_lower'], submission['pi_upper'])

# å„²å­˜ submission.csv
submission.to_csv(f"{parameter['OUTPUT_PATH']}submission.csv", index=False)

print("\nâœ… æ��äº¤æª”å·²å»ºç«‹")
display(submission.head())

