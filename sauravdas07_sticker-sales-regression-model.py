import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split , KFold, RandomizedSearchCV,StratifiedKFold,RepeatedKFold,cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv' ,parse_dates=['date'])
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])
test_df2 = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
train_df.head(2)


print(train_df.info())
print("\n**************************************\n")
print(test_df.info())


train_df.isna().sum()


plt.figure(figsize=(12, 4))
ax = sns.lineplot(
    data=train_df,
    x="date",
    y="num_sold",
    hue="country",
    errorbar=None,
    linewidth=0.4,
    palette = "Dark2"
)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1, 1), ncols=1, fontsize=8)
plt.title("Number Sold | Years | Countries", size=10)
plt.show()


plt.figure(figsize=(12, 4))
ax = sns.lineplot(
    data=train_df,
    x="date",
    y="num_sold",
    hue="store",
    errorbar=None,
    linewidth=0.4,
    palette = "Dark2"
)
ax.set_xlabel("Stor", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1, 1), ncols=1, fontsize=8)
plt.title("Number Sold | years | store", size=10)
plt.show()


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


print("Preparing data...")
train, train_cat_features = create_features(train_df, is_train=True)
test, test_cat_features = create_features(test_df, is_train=False)

# Add global features
train, test = add_global_features(train, test, train_cat_features)

# Separate features and target
feature_cols = [col for col in train.columns if col not in ['num_sold']]
X = train[feature_cols]
y = train['num_sold']
X_test = test[feature_cols]

print("\nFeature columns:", feature_cols)
print(f"Number of features: {len(feature_cols)}")


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


final_pred = np.mean(all_test_preds, axis=0)


submission = pd.DataFrame({
    'id': test_df['id'],
    'num_sold': final_pred
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created!")


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

