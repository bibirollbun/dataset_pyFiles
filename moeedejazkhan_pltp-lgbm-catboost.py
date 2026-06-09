# ğŸ“¦ Install necessary libraries (only if missing)
!pip install lightgbm catboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')




train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")




# Checking missing values
train.isnull().sum()




# Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(train['Listening_Time_minutes'], kde=True)
plt.title('Listening Time Distribution')
plt.show()




# Filling missing values
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    median_val = train[col].median()
    train[col].fillna(median_val, inplace=True)
    test[col].fillna(median_val, inplace=True)

# Label Encoding
cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# New Features
train['popularity_ratio'] = (train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage']) / 2
test['popularity_ratio'] = (test['Host_Popularity_percentage'] + test['Guest_Popularity_percentage']) / 2

train['length_per_ad'] = train['Episode_Length_minutes'] / (train['Number_of_Ads'] + 1)
test['length_per_ad'] = test['Episode_Length_minutes'] / (test['Number_of_Ads'] + 1)

# Features
features = [col for col in train.columns if col not in ['id', 'Episode_Title', 'Listening_Time_minutes']]
X = train[features]
y = train['Listening_Time_minutes']
X_test = test[features]




folds = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof = np.zeros(X.shape[0])
cat_oof = np.zeros(X.shape[0])
lgb_preds = np.zeros(X_test.shape[0])
cat_preds = np.zeros(X_test.shape[0])

for fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=8,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)]
    )

    lgb_oof[val_idx] = lgb_model.predict(X_valid)
    lgb_preds += lgb_model.predict(X_test) / folds.n_splits

    # CatBoost
    cat_model = cb.CatBoostRegressor(
        iterations=10000,
        learning_rate=0.01,
        depth=8,
        loss_function='RMSE',
        early_stopping_rounds=100,
        random_state=42,
        verbose=100
    )

    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    cat_oof[val_idx] = cat_model.predict(X_valid)
    cat_preds += cat_model.predict(X_test) / folds.n_splits




final_oof = (lgb_oof + cat_oof) / 2
final_preds = (lgb_preds + cat_preds) / 2

print(f"\nâœ… Final CV RMSE: {np.sqrt(mean_squared_error(y, final_oof)):.5f}")




feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Important Features (LightGBM)')
plt.show()




submission = sample_submission.copy()
submission['Listening_Time_minutes'] = final_preds
submission.to_csv('submission.csv', index=False)

print("ğŸ�‰ Submission file created successfully!")


