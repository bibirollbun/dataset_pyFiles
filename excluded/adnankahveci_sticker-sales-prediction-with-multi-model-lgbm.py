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


# 1. Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc

# Display settings
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')



# 2. Load Data
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')



# 3. Feature Engineering
def create_features(df, is_train=True):
    df = df.copy()
    
    # Date features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Store original categorical values
    categorical_features = {}
    for col in ['country', 'store', 'product']:
        categorical_features[col] = df[col].copy()
    
    # Label encoding
    le = LabelEncoder()
    for col in ['country', 'store', 'product']:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Drop original date column
    df = df.drop('date', axis=1)
    
    return df, categorical_features



# 4. Create global features
def add_global_features(train, test, categorical_features):
    # Combine train and test for consistent encoding
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    
    # Group features
    for col in ['country', 'store', 'product']:
        # Count encoding
        count_enc = all_data.groupby(col).size()
        train[f'{col}_count'] = train[col].map(count_enc)
        test[f'{col}_count'] = test[col].map(count_enc)
        
        # Interaction features
        for col2 in ['country', 'store', 'product']:
            if col != col2:
                train[f'{col}_{col2}_interaction'] = train[col] * train[col2]
                test[f'{col}_{col2}_interaction'] = test[col] * test[col2]
    
    return train, test


# 5. Prepare Data
print("Preparing data...")
train, train_cat_features = create_features(train_df, is_train=True)
test, test_cat_features = create_features(test_df, is_train=False)

# Add global features
train, test = add_global_features(train, test, train_cat_features)

# Separate features and target
feature_cols = [col for col in train.columns if col not in ['id', 'num_sold']]
X = train[feature_cols]
y = train['num_sold']
X_test = test[feature_cols]

print("\nFeature columns:", feature_cols)
print(f"Number of features: {len(feature_cols)}")


# 6. Model Training Function
def train_model_lgb(X, y, X_test, params, n_splits=5, seed=42):
    models = []
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'\nFold {fold + 1}')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
        
        model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            callbacks=callbacks
        )
        
        models.append(model)
        oof_pred[val_idx] = model.predict(X_val)
        test_pred += model.predict(X_test) / n_splits
        
        gc.collect()
    
    return models, oof_pred, test_pred


# 7. Model Configurations
model_configs = [
    {
        'name': 'lgb_base',
        'params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_iterations': 1000
        }
    },
    {
        'name': 'lgb_deep',
        'params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.01,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_depth': 12,
            'verbose': -1,
            'num_iterations': 1500
        }
    }
]


# 8. Train Multiple Models
print("\nTraining multiple models...")
all_models = []
all_oof_preds = []
all_test_preds = []

for config in model_configs:
    print(f"\nTraining {config['name']}...")
    models, oof_pred, test_pred = train_model_lgb(X, y, X_test, config['params'])
    
    all_models.append(models)
    all_oof_preds.append(oof_pred)
    all_test_preds.append(test_pred)
    
    rmse = np.sqrt(np.mean((y - oof_pred) ** 2))
    print(f"{config['name']} RMSE: {rmse:.4f}")



# 9. Create Ensemble Prediction
final_pred = np.mean(all_test_preds, axis=0)


# 10. Create Submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'num_sold': final_pred
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created!")



# 11. Feature Importance Plot
def plot_feature_importance(models, features):
    importance_df = pd.DataFrame()
    
    for model in models[0]:
        model_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance('gain')
        })
        importance_df = pd.concat([importance_df, model_importance])
    
    importance_df = importance_df.groupby('feature')['importance'].mean().reset_index()
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.show()

plot_feature_importance(all_models, X.columns)

