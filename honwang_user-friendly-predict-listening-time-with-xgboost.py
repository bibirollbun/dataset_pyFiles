# Step 1: Import Libraries

# Basic
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Settings
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Ignore warnings for clean output
import warnings
warnings.filterwarnings('ignore')



# Step 2: Load Dataset

# Load the training and testing data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Quick look at the shape
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Preview the datasets
display(train.head())
display(test.head())

# Quick info about train data
print("\nTrain Data Info:")
train.info()

# Check for missing values
print("\nMissing values in Train:")
print(train.isnull().sum())

print("\nMissing values in Test:")
print(test.isnull().sum())



# Step 3: Exploratory Data Analysis (EDA)

# Target variable
plt.figure(figsize=(8, 6))
sns.histplot(train['Listening_Time_minutes'], bins=50, kde=True)
plt.title('Distribution of Listening Time (minutes)')
plt.xlabel('Listening Time (minutes)')
plt.show()

# Numerical features correlation matrix
numerical_features = [
    'Episode_Length_minutes', 
    'Host_Popularity_percentage', 
    'Guest_Popularity_percentage', 
    'Number_of_Ads',
    'Listening_Time_minutes'
]

plt.figure(figsize=(10, 8))
corr_matrix = train[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numerical Features)')
plt.show()

# Categorical Features vs Target
categorical_features = [
    'Genre', 
    'Publication_Day', 
    'Episode_Sentiment'
]

for cat in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train, x=cat, y='Listening_Time_minutes')
    plt.xticks(rotation=45)
    plt.title(f'Listening Time by {cat}')
    plt.show()

# Missing values heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull(), cbar=False)
plt.title('Missing Values Heatmap (Train Dataset)')
plt.show()



# Step 4: Preprocessing and Feature Engineering

# Copy datasets to avoid messing originals
train_prep = train.copy()
test_prep = test.copy()

# Fill missing 'Episode_Length_minutes' with median per Genre
train_prep['Episode_Length_minutes'] = train_prep.groupby('Genre')['Episode_Length_minutes'].transform(
    lambda x: x.fillna(x.median())
)
test_prep['Episode_Length_minutes'] = test_prep.groupby('Genre')['Episode_Length_minutes'].transform(
    lambda x: x.fillna(x.median())
)

# Fill missing 'Guest_Popularity_percentage' with median per Genre
train_prep['Guest_Popularity_percentage'] = train_prep.groupby('Genre')['Guest_Popularity_percentage'].transform(
    lambda x: x.fillna(x.median())
)
test_prep['Guest_Popularity_percentage'] = test_prep.groupby('Genre')['Guest_Popularity_percentage'].transform(
    lambda x: x.fillna(x.median())
)

# Fill missing 'Number_of_Ads' (only 1 missing) with mode
train_prep['Number_of_Ads'].fillna(train_prep['Number_of_Ads'].mode()[0], inplace=True)

# Extract hour from 'Publication_Time'
def extract_hour(x):
    if ':' in x:
        return int(x.split(':')[0])
    else:
        mapping = {
            'Night': 22,
            'Morning': 8,
            'Afternoon': 15,
            'Evening': 18
        }
        return mapping.get(x, 12)  # Default to 12 (noon) if not matched


# Apply the fixed extraction
train_prep['Publication_Hour'] = train_prep['Publication_Time'].apply(extract_hour)
test_prep['Publication_Hour'] = test_prep['Publication_Time'].apply(extract_hour)

# Drop unnecessary columns
drop_cols = ['id', 'Podcast_Name', 'Episode_Title', 'Publication_Time']
train_prep.drop(columns=drop_cols, inplace=True)
test_prep.drop(columns=drop_cols, inplace=True)

# Encode categorical features using Frequency Encoding
for col in ['Genre', 'Publication_Day', 'Episode_Sentiment']:
    freq_encoding = train_prep[col].value_counts() / len(train_prep)
    train_prep[col] = train_prep[col].map(freq_encoding)
    test_prep[col] = test_prep[col].map(freq_encoding)

# Separate features and target
X = train_prep.drop(columns=['Listening_Time_minutes'])
y = train_prep['Listening_Time_minutes']
X_test = test_prep

print(f"Train Features Shape: {X.shape}")
print(f"Test Features Shape: {X_test.shape}")



# Step 5: Baseline LightGBM Model with K-Fold CV (Final Fix)

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Initialize KFold
folds = KFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store out-of-fold predictions and test predictions
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])

# LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbosity': -1
}

# Start CV
for fold_idx, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f'ğŸ“‚ Training fold {fold_idx+1}...')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=200)
        ]
    )
    
    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

# Overall CV Score
cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f'ğŸ�† Overall CV RMSE: {cv_rmse:.5f}')



# Step 6: Feature Importance Plot

# Extract feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance()
}).sort_values(by='importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - LightGBM Baseline')
plt.show()

# Display the DataFrame for detailed view
importance_df



# Step 7: Advanced Feature Engineering

# Copy to avoid overwriting
X_advanced = X.copy()
X_test_advanced = X_test.copy()

# Feature: Host and Guest Popularity Difference
X_advanced['Host_Guest_Diff'] = X_advanced['Host_Popularity_percentage'] - X_advanced['Guest_Popularity_percentage']
X_test_advanced['Host_Guest_Diff'] = X_test_advanced['Host_Popularity_percentage'] - X_test_advanced['Guest_Popularity_percentage']

# Feature: Host and Guest Popularity Average
X_advanced['Host_Guest_Avg'] = (X_advanced['Host_Popularity_percentage'] + X_advanced['Guest_Popularity_percentage']) / 2
X_test_advanced['Host_Guest_Avg'] = (X_test_advanced['Host_Popularity_percentage'] + X_test_advanced['Guest_Popularity_percentage']) / 2

# Feature: Episode Length per Ad (more ads might impact listening!)
X_advanced['Length_per_Ad'] = X_advanced['Episode_Length_minutes'] / (X_advanced['Number_of_Ads'] + 1)
X_test_advanced['Length_per_Ad'] = X_test_advanced['Episode_Length_minutes'] / (X_test_advanced['Number_of_Ads'] + 1)

# Feature: Publication Hour bins (Morning, Afternoon, Night)
def map_hour_to_period(hour):
    if 5 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 17:
        return 2  # Afternoon
    else:
        return 3  # Evening/Night

X_advanced['Publication_Period'] = X_advanced['Publication_Hour'].apply(map_hour_to_period)
X_test_advanced['Publication_Period'] = X_test_advanced['Publication_Hour'].apply(map_hour_to_period)

print(f"New Train Shape: {X_advanced.shape}")
print(f"New Test Shape: {X_test_advanced.shape}")



# Step 8: Retrain LightGBM on advanced features

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Initialize KFold
folds = KFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store out-of-fold predictions and test predictions
oof_preds_adv = np.zeros(X_advanced.shape[0])
test_preds_adv = np.zeros(X_test_advanced.shape[0])

# LightGBM parameters (same as before for fair comparison)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbosity': -1
}

# Start CV
for fold_idx, (train_idx, val_idx) in enumerate(folds.split(X_advanced, y)):
    print(f'ğŸ“‚ Training fold {fold_idx+1}...')
    
    X_train, X_val = X_advanced.iloc[train_idx], X_advanced.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model_adv = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=200)
        ]
    )
    
    oof_preds_adv[val_idx] = model_adv.predict(X_val, num_iteration=model_adv.best_iteration)
    test_preds_adv += model_adv.predict(X_test_advanced, num_iteration=model_adv.best_iteration) / folds.n_splits

# Overall CV Score
cv_rmse_adv = np.sqrt(mean_squared_error(y, oof_preds_adv))
print(f'ğŸ�† Overall CV RMSE with Advanced Features: {cv_rmse_adv:.5f}')



# Step 9: Hyperparameter Tuning using Optuna

import optuna

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(X_advanced.shape[0])
    
    for trn_idx, val_idx in folds.split(X_advanced, y):
        X_trn, X_val = X_advanced.iloc[trn_idx], X_advanced.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_trn, label=y_trn)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx] = preds
        
    rmse = np.sqrt(mean_squared_error(y, oof))
    return rmse

# Start Optuna Study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f'ğŸ�† Best Trial: {study.best_trial.params}')
print(f'ğŸ�† Best RMSE: {study.best_value:.5f}')



# Step 10: Train XGBoost Regressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np

# Define XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
}

# 5-Fold Cross Validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(X_advanced.shape[0])

for fold, (trn_idx, val_idx) in enumerate(folds.split(X_advanced, y)):
    print(f"ğŸ“‚ Training fold {fold+1}...")
    
    X_trn, X_val = X_advanced.iloc[trn_idx], X_advanced.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_trn, label=y_trn)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    
    xgb_preds = model.predict(dvalid) 
    xgb_oof[val_idx] = xgb_preds  # Save predictions for each fold

# Final Evaluation
xgb_rmse = mean_squared_error(y, xgb_oof, squared=False)
print(f"âœ… XGBoost CV RMSE: {xgb_rmse:.5f}")



# Step 11: Full Feature Engineering + XGBoost Upgrade

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ğŸ›  Step 1:'Publication_Hour'
def extract_hour(x):
    try:
        return int(x.split(':')[0])
    except:
        return -1

for df in [train, test]:
    df['Publication_Hour'] = df['Publication_Time'].apply(extract_hour)

# ğŸ›  Step 2: Feature Engineering
for df in [train, test]:
    df['Host_Guest_Mean'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']) / 2
    df['Ad_per_minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    df['Sentiment_Score'] = df['Episode_Sentiment'].map(sentiment_map)

# ğŸ›  Step 3: Define Features
feature_cols = [
    'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Length_minutes',
    'Genre', 'Publication_Day', 'Number_of_Ads', 'Publication_Hour',
    'Host_Guest_Mean', 'Ad_per_minute', 'Sentiment_Score'
]

# One-hot encode categorical features (Genre, Publication_Day)
X_train = pd.get_dummies(train[feature_cols], columns=['Genre', 'Publication_Day'])
X_test = pd.get_dummies(test[feature_cols], columns=['Genre', 'Publication_Day'])

# Ensure same columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y = train['Listening_Time_minutes']

# ğŸ›  Step 4: XGBoost Parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.02,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'seed': 42,
}

# ğŸ›  Step 5: 5-Fold Cross Validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(X_train.shape[0])

for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y)):
    print(f"ğŸ“‚ Training fold {fold+1}...")
    
    X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_trn, label=y_trn)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=15000,
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=300,
        verbose_eval=200
    )
    
    xgb_preds = model.predict(dvalid)
    xgb_oof[val_idx] = xgb_preds

# ğŸ›  Step 6: Final Evaluation
xgb_rmse = mean_squared_error(y, xgb_oof, squared=False)
print(f"âœ… XGBoost Full Upgrade CV RMSE: {xgb_rmse:.5f}")



import lightgbm as lgb
import catboost as cat

# Initialize OOF arrays
lgb_oof = np.zeros(X_train.shape[0])
cat_oof = np.zeros(X_train.shape[0])

# Common folds (same as before)
folds = KFold(n_splits=5, shuffle=True, random_state=42)

# Train LightGBM
from lightgbm import early_stopping, log_evaluation

for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y)):
    print(f"ğŸ“‚ [LightGBM] Training fold {fold+1}...")
    
    X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_trn, label=y_trn)
    lgb_valid = lgb.Dataset(X_val, label=y_val)
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.02,
        'num_leaves': 256,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0,
        'seed': 42,
    }
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_valid],
        num_boost_round=15000,
        callbacks=[
            early_stopping(stopping_rounds=300),
            log_evaluation(period=200)
        ]
    )
    
    preds = lgb_model.predict(X_val)
    lgb_oof[val_idx] = preds


# Train CatBoost
for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y)):
    print(f"ğŸ“‚ [CatBoost] Training fold {fold+1}...")
    
    X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    cat_model = cat.CatBoostRegressor(
        iterations=15000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3.0,
        eval_metric='RMSE',
        early_stopping_rounds=300,
        random_seed=42,
        verbose=200
    )
    
    cat_model.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True)
    
    preds = cat_model.predict(X_val)
    cat_oof[val_idx] = preds

# Evaluate individually
print(f"âœ… LightGBM CV RMSE: {mean_squared_error(y, lgb_oof, squared=False):.5f}")
print(f"âœ… CatBoost CV RMSE: {mean_squared_error(y, cat_oof, squared=False):.5f}")

# Blending
blended_preds = 0.4 * xgb_oof + 0.3 * lgb_oof + 0.3 * cat_oof
final_rmse = mean_squared_error(y, blended_preds, squared=False)
print(f"ğŸ�† Final Blended CV RMSE: {final_rmse:.5f}")



# âœ… Generate Test Predictions for All Models

# Convert test set to DMatrix for XGBoost
dtest = xgb.DMatrix(X_test)
xgb_preds_test = model.predict(dtest)

# LightGBM prediction (model already trained from previous cell)
lgb_preds_test = lgb_model.predict(X_test)

# CatBoost prediction (model already trained from previous cell)
cat_preds_test = cat_model.predict(X_test)

# ğŸ§ª Blending (same weights as CV evaluation)
final_preds = 0.4 * xgb_preds_test + 0.3 * lgb_preds_test + 0.3 * cat_preds_test

# ğŸ“� Create Submission DataFrame
submission = pd.DataFrame({
    "id": test["id"],
    "Listening_Time_minutes": final_preds
})

# ğŸ’¾ Save to CSV
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")

