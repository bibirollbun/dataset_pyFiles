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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import re
import warnings
warnings.filterwarnings('ignore')


# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


# Set random seeds for reproducibility
np.random.seed(42)

def load_data():
    train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
    
    return train_df, test_df, sample_submission

train_df, test_df, sample_submission = load_data()

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# 1. Target Distribution Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df['Tg'], kde=True, bins=30)
plt.title('Distribution of Target Variable')
plt.xlabel('Target Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(y=train_df['Tg'])
plt.title('Boxplot of Target Variable')
plt.tight_layout()
plt.show()


# 2. SMILES Length Analysis
train_df['smiles_length'] = train_df['SMILES'].apply(len)
test_df['smiles_length'] = test_df['SMILES'].apply(len)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df['smiles_length'], kde=True, bins=30, label='Train')
sns.histplot(test_df['smiles_length'], kde=True, bins=30, label='Test', alpha=0.7)
plt.title('Distribution of SMILES Length')
plt.xlabel('SMILES Length')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=train_df['smiles_length'], y=train_df['Tg'])
plt.title('SMILES Length vs Target')
plt.xlabel('SMILES Length')
plt.ylabel('Target Value')
plt.tight_layout()
plt.show()


# Display basic info
print("\nTrain data columns:")
print(train_df.columns.tolist())
print("\nTest data columns:")
print(test_df.columns.tolist())



# Check for missing values
print("\nMissing values in train data:")
print(train_df.isnull().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum())



# Basic statistics
print("\nTarget variable statistics:")
print(train_df['Tg'].describe())


# Feature engineering
def compute_smiles_features(smiles):
    """Compute features from SMILES string without RDKit"""
    features = {}
    
    # Basic string features
    features['smiles_length'] = len(smiles)
    
    # Count specific characters/patterns
    features['capital_letters'] = sum(1 for c in smiles if c.isupper())
    features['lowercase_letters'] = sum(1 for c in smiles if c.islower())
    features['digits'] = sum(1 for c in smiles if c.isdigit())
    features['parentheses'] = smiles.count('(') + smiles.count(')')
    features['brackets'] = smiles.count('[') + smiles.count(']')
    features['braces'] = smiles.count('{') + smiles.count('}')
    features['equals'] = smiles.count('=')
    features['hashes'] = smiles.count('#')
    features['colons'] = smiles.count(':')
    features['ats'] = smiles.count('@')
    features['slashes'] = smiles.count('/') + smiles.count('\\')
    features['plus_minus'] = smiles.count('+') + smiles.count('-')
    
    # Count specific elements
    features['C_count'] = smiles.count('C') + smiles.count('c')
    features['O_count'] = smiles.count('O') + smiles.count('o')
    features['N_count'] = smiles.count('N') + smiles.count('n')
    features['S_count'] = smiles.count('S') + smiles.count('s')
    features['P_count'] = smiles.count('P') + smiles.count('p')
    features['F_count'] = smiles.count('F') + smiles.count('f')
    features['Cl_count'] = smiles.count('Cl') + smiles.count('cl')
    features['Br_count'] = smiles.count('Br') + smiles.count('br')
    features['I_count'] = smiles.count('I') + smiles.count('i')
    
    # Check for specific patterns
    features['has_ring'] = int(any(char in smiles for char in ['1', '2', '3', '4', '5', '6', '7', '8', '9']))
    features['has_double_bond'] = int('=' in smiles)
    features['has_triple_bond'] = int('#' in smiles)
    features['has_aromatic'] = int(any(c in smiles for c in ['c', 'n', 'o', 's']))
    
    # Element ratios
    features['O_to_C_ratio'] = features['O_count'] / (features['C_count'] + 1e-5)
    features['N_to_C_ratio'] = features['N_count'] / (features['C_count'] + 1e-5)
    features['heteroatom_ratio'] = (features['O_count'] + features['N_count'] + features['S_count'] + features['P_count']) / (features['C_count'] + 1e-5)
    
    return features


def add_smiles_features(df):
    """Add SMILES-based features to dataframe"""
    features_list = []
    
    for smiles in df['SMILES']:
        features = compute_smiles_features(smiles)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list, index=df.index)
    df = pd.concat([df, features_df], axis=1)
    
    return df


# Handle missing values
def handle_missing_values(df):
    """Handle missing values in the dataset"""
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    return df

train_df = handle_missing_values(train_df)
test_df = handle_missing_values(test_df)


print("Adding SMILES features to train data...")
train_df = add_smiles_features(train_df)

print("Adding SMILES features to test data...")
test_df = add_smiles_features(test_df)


# 3. Element Count Visualization
element_cols = ['C_count', 'O_count', 'N_count', 'S_count', 'P_count', 'F_count', 'Cl_count', 'Br_count', 'I_count']

plt.figure(figsize=(15, 10))
for i, col in enumerate(element_cols[:6], 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=train_df[col], y=train_df['Tg'], alpha=0.6)
    plt.title(f'{col} vs Target')
    plt.xlabel(col)
    plt.ylabel('Target Value')
plt.tight_layout()
plt.show()


# Identify common features between train and test
exclude_cols = ['id', 'SMILES', 'Tg']
train_feature_cols = [col for col in train_df.columns if col not in exclude_cols]
test_feature_cols = [col for col in test_df.columns if col not in ['id', 'SMILES']]


common_features = list(set(train_feature_cols) & set(test_feature_cols))
print(f"Number of common features: {len(common_features)}")

X = train_df[common_features]
y = train_df['Tg']
X_test = test_df[common_features]


# 5. Feature Distribution Comparison (Train vs Test)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(common_features[:6], 1):
    plt.subplot(2, 3, i)
    sns.histplot(train_df[feature], label='Train', kde=True, alpha=0.7)
    sns.histplot(test_df[feature], label='Test', kde=True, alpha=0.7)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
plt.tight_layout()
plt.show()


# Model parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

cat_params = {
    'iterations': 5000,
    'learning_rate': 0.01,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_strength': 1,
    'bagging_temperature': 0.8,
    'od_type': 'Iter',
    'od_wait': 100,
    'random_state': 42,
    'verbose': False
}



# Initialize models
lgb_model = lgb.LGBMRegressor(**lgb_params)
xgb_model = xgb.XGBRegressor(**xgb_params)
cat_model = CatBoostRegressor(**cat_params)



# Set up cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store predictions
lgb_oof = np.zeros(len(X))
lgb_test_preds = np.zeros(len(X_test))

xgb_oof = np.zeros(len(X))
xgb_test_preds = np.zeros(len(X_test))

cat_oof = np.zeros(len(X))
cat_test_preds = np.zeros(len(X_test))

lgb_scores, xgb_scores, cat_scores = [], [], []

print("Starting cross-validation training...")


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}/{n_folds}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # LightGBM - use callbacks for early stopping
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    lgb_pred = lgb_model.predict(X_val_scaled)
    lgb_oof[val_idx] = lgb_pred
    lgb_test_preds += lgb_model.predict(X_test_scaled) / n_folds
    lgb_score = np.sqrt(mean_squared_error(y_val, lgb_pred))
    lgb_scores.append(lgb_score)
    
    # XGBoost
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    xgb_pred = xgb_model.predict(X_val_scaled)
    xgb_oof[val_idx] = xgb_pred
    xgb_test_preds += xgb_model.predict(X_test_scaled) / n_folds
    xgb_score = np.sqrt(mean_squared_error(y_val, xgb_pred))
    xgb_scores.append(xgb_score)
    
    # CatBoost
    cat_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    cat_pred = cat_model.predict(X_val_scaled)
    cat_oof[val_idx] = cat_pred
    cat_test_preds += cat_model.predict(X_test_scaled) / n_folds
    cat_score = np.sqrt(mean_squared_error(y_val, cat_pred))
    cat_scores.append(cat_score)
    
    print(f"LGBM Score: {lgb_score:.5f}, XGB Score: {xgb_score:.5f}, CatBoost Score: {cat_score:.5f}")


print("\nCV Results:")
print(f"LGBM: {np.mean(lgb_scores):.5f} ± {np.std(lgb_scores):.5f}")
print(f"XGB: {np.mean(xgb_scores):.5f} ± {np.std(xgb_scores):.5f}")
print(f"CatBoost: {np.mean(cat_scores):.5f} ± {np.std(cat_scores):.5f}")


# 6. Model Performance Comparison
model_names = ['LightGBM', 'XGBoost', 'CatBoost']
model_scores = [np.mean(lgb_scores), np.mean(xgb_scores), np.mean(cat_scores)]
model_stds = [np.std(lgb_scores), np.std(xgb_scores), np.std(cat_scores)]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=model_scores, yerr=model_stds)
plt.title('Model Performance Comparison (RMSE)')
plt.ylabel('RMSE Score')
plt.xlabel('Models')
plt.show()


# 7. OOF Predictions vs Actual Values
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x=y, y=lgb_oof, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('LightGBM: OOF Predictions vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 3, 2)
sns.scatterplot(x=y, y=xgb_oof, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('XGBoost: OOF Predictions vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 3, 3)
sns.scatterplot(x=y, y=cat_oof, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('CatBoost: OOF Predictions vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()


# Ensemble predictions
lgb_weight = 1 / np.mean(lgb_scores)
xgb_weight = 1 / np.mean(xgb_scores)
cat_weight = 1 / np.mean(cat_scores)
total_weight = lgb_weight + xgb_weight + cat_weight

lgb_weight /= total_weight
xgb_weight /= total_weight
cat_weight /= total_weight

ensemble_test_preds = (
    lgb_weight * lgb_test_preds +
    xgb_weight * xgb_test_preds +
    cat_weight * cat_test_preds
)


# Prepare submission
submission = sample_submission.copy()
submission['target'] = ensemble_test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file created!")

print(f"\nSubmission target stats:")
print(f"Min: {submission['target'].min():.4f}")
print(f"Max: {submission['target'].max():.4f}")
print(f"Mean: {submission['target'].mean():.4f}")
print(f"Std: {submission['target'].std():.4f}")



# Plot feature importance - Fixed version
try:
    # Get feature importance from the last trained model
    feature_importance = pd.DataFrame({
        'feature': common_features,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance (LightGBM)')
    plt.tight_layout()
    plt.show()
    
except (ValueError, AttributeError) as e:
    print(f"Could not create feature importance plot: {e}")
    print(f"Number of common features: {len(common_features)}")
    print(f"Number of feature importances: {len(lgb_model.feature_importances_) if hasattr(lgb_model, 'feature_importances_') else 'N/A'}")
    
    # Alternative: Create feature importance using correlation
    correlations = []
    for feature in common_features:
        if pd.api.types.is_numeric_dtype(X[feature]):
            corr = abs(np.corrcoef(X[feature], y)[0, 1])
            correlations.append((feature, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    feature_importance_corr = pd.DataFrame(correlations, columns=['feature', 'correlation']).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='correlation', y='feature', data=feature_importance_corr)
    plt.title('Top 15 Features by Correlation with Target')
    plt.tight_layout()
    plt.show()



# 9. Prediction Distribution Comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.histplot(train_df['Tg'], label='Training Target', kde=True, alpha=0.7)
sns.histplot(submission['Tg'], label='Test Predictions', kde=True, alpha=0.7)
plt.title('Distribution: Training Target vs Test Predictions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(data=[train_df['Tg'], submission['Tg']])
plt.xticks([0, 1], ['Training Target', 'Test Predictions'])
plt.title('Boxplot: Training Target vs Test Predictions')
plt.ylabel('Value')

plt.tight_layout()
plt.show()



ensemble_oof = (lgb_weight * lgb_oof + xgb_weight * xgb_oof + cat_weight * cat_oof)
residuals = y - ensemble_oof

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=ensemble_oof, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


print(f"\nFinal Ensemble Performance:")
print(f"OOF RMSE: {np.sqrt(mean_squared_error(y, ensemble_oof)):.5f}")
print(f"\nSubmission target stats:")
print(f"Min: {submission['target'].min():.4f}")
print(f"Max: {submission['target'].max():.4f}")
print(f"Mean: {submission['target'].mean():.4f}")
print(f"Std: {submission['target'].std():.4f}")

