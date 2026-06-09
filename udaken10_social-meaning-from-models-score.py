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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import time

# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
train_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')

# ä½¿ç”¨ã�™ã‚‹ç‰¹å¾´é‡�
features = ['RACE_BLACK_NH_PCT', 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT', 
            'AGE_25_34_PCT', 'FAMILY_HH_CHILD_LT18_PCT', 
            'VETERAN_POP_PCT', 'FAMILY_HH_TOTAL']

test_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

test = test_df[features]

# ç‰¹å¾´é‡�ã�¨ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã‚’åˆ†é›¢
X = train_df[features]
y = train_df['HOMELESS_RATE']

# è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�«åˆ†å‰²
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ãƒ¢ãƒ‡ãƒ«ã�®å®šç¾©
models = {
    'LightGBM': LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=7,
        random_state=42,
        silent=True
    ),
    'XGBoost': XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    )
}

# çµ�æ�œã‚’æ ¼ç´�ã�™ã‚‹è¾�æ›¸
results = {}

print("=" * 80)
print("ãƒ¢ãƒ‡ãƒ«æ¯”è¼ƒï¼šLightGBM vs CatBoost vs XGBoost")
print("=" * 80)

# å�„ãƒ¢ãƒ‡ãƒ«ã�§å­¦ç¿’ã�¨è©•ä¾¡
for model_name, model in models.items():
    print(f"\nã€�{model_name}ã€‘")
    print("-" * 80)
    
    # å­¦ç¿’æ™‚é–“ã‚’è¨ˆæ¸¬
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # äºˆæ¸¬
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # è©•ä¾¡æŒ‡æ¨™ã�®è¨ˆç®—
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ï¼ˆ5åˆ†å‰²ï¼‰
    cv_scores = cross_val_score(model, X, y, cv=5, 
                                 scoring='neg_mean_squared_error', 
                                 n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_std = np.sqrt(cv_scores.std())
    
    # çµ�æ�œã‚’ä¿�å­˜
    results[model_name] = {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'Test RÂ²': test_r2,
        'CV RMSE': cv_rmse,
        'CV Std': cv_std,
        'Train Time (s)': train_time
    }
    
    # çµ�æ�œã‚’è¡¨ç¤º
    print(f"è¨“ç·´æ™‚é–“:        {train_time:.3f} ç§’")
    print(f"Train RMSE:      {train_rmse:.6f}")
    print(f"Test RMSE:       {test_rmse:.6f}")
    print(f"Test MAE:        {test_mae:.6f}")
    print(f"Test RÂ²:         {test_r2:.6f}")
    print(f"CV RMSE:         {cv_rmse:.6f} (Â±{cv_std:.6f})")

# çµ�æ�œã‚’æ¯”è¼ƒè¡¨ã�¨ã�—ã�¦è¡¨ç¤º
print("\n" + "=" * 80)
print("ã€�æ¯”è¼ƒçµ�æ�œã‚µãƒ�ãƒªãƒ¼ã€‘")
print("=" * 80)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(6)
print(comparison_df)

# æœ€è‰¯ã�®ãƒ¢ãƒ‡ãƒ«ã‚’ç‰¹å®š
best_model_name = comparison_df['Test RMSE'].idxmin()
print(f"\nğŸ�† æœ€è‰¯ãƒ¢ãƒ‡ãƒ«ï¼ˆTest RMSEãƒ™ãƒ¼ã‚¹ï¼‰: {best_model_name}")
print(f"   Test RMSE: {comparison_df.loc[best_model_name, 'Test RMSE']:.6f}")

# ç‰¹å¾´é‡�ã�®é‡�è¦�åº¦ã‚’è¡¨ç¤ºï¼ˆä¸Šä½�10å€‹ï¼‰
print("\n" + "=" * 80)
print("ã€�ç‰¹å¾´é‡�é‡�è¦�åº¦ Top 6ã€‘")
print("=" * 80)

for model_name, model in models.items():
    print(f"\n{model_name}:")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print(feature_importance_df.to_string(index=False))


test_df.head(10)


train_df.columns.to_list()


model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    )
model.fit(X, y)





y_pred = model.predict(test)
y_pred


sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')
sub['HOMELESS_RATE'] = y_pred
sub.to_csv('submission.csv', index = False)
sub




