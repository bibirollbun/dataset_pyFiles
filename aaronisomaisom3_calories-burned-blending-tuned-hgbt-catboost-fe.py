# !pip install optuna --quiet


# Downcast the float64/int64 
def downcast_numericals(train, test):
    # Downcast numeric columns
    num_cols = test.select_dtypes(np.number).columns.tolist()
    
    for col in num_cols:
        if 'int' in str(train[col].dtype):
            train[col] = pd.to_numeric(train[col], downcast='integer')
            test[col] = pd.to_numeric(test[col], downcast='integer')
        else:
            train[col] = pd.to_numeric(train[col], downcast='float')
            test[col] = pd.to_numeric(test[col], downcast='float')
            
    return train, test

# Encode categoricals
def label_encode_categoricals(train, test):
    cat_cols = train.select_dtypes(exclude=[np.number]).columns.tolist()
        
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
    
    return train, test

# Optuna objective using 5 Folds
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 500, 3000),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 255),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 255),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 10.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 64, 255)
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

        model = HistGradientBoostingRegressor(**params, early_stopping=True, validation_fraction=0.3, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_valid)
        scores.append(rmsle(y_valid, y_pred))

    return np.mean(scores)


# Author: Aaron Isom
# Kaggle Predict Calorie Expenditure
# HistGradientBoostingRegressor and Optuna for hyperparameter tuning using RMSLE

import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# RMSLE function
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)   # Ensure non-negative
    
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Create the RMSLE scorer
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')

# Display first few rows
display(train_df.head(10))
display('Train Shape', train_df.shape)
display('Test Shape', test_df.shape)

display('Missing Train Values:', train_df.isnull().sum())
display('Missing Test Values:', test_df.isnull().sum())

# Describe the data
display(train_df.describe())
display(test_df.describe())

# Display information about dtypes
display('Train Data Info:', train_df.info())
display('Test Data Info:', test_df.info())

# Preprocessing + Feature Engineering
for df in [train_df, test_df]:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df['Duration'] * df['Heart_Rate']
    df['log_Duration'] = np.log1p(df['Duration'])
    df['log_Intensity'] = np.log1p(df['Intensity'])
    df['log_BMI'] = np.log1p(df['BMI'])

# Clip the top 1% of Calories to reduce log-space distortion
cap = train_df['Calories'].quantile(0.99)
train_df['Calories'] = np.minimum(train_df['Calories'], cap)

train_df, test_df = label_encode_categoricals(train_df, test_df)
train_df, test_df = downcast_numericals(train_df, test_df)

# EDA
plt.figure(figsize=(18, 14)) 
sns.heatmap(
    train_df.corr(), 
    annot=True, 
    fmt=".2f", 
    cmap='RdBu_r', 
    annot_kws={'size': 8}, 
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

cor_matrix = train_df.corr()
low_corr_features = cor_matrix['Calories'].abs().sort_values()
print("Low correlation features:\n", low_corr_features[low_corr_features < 0.05])

X,y = train_df.drop('Calories', axis=1),  np.log1p(train_df["Calories"])  # Instead of raw Calories

# Distributions
features = train_df.columns.tolist()
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.histplot(train_df[feature], color='lightgray', edgecolor='black', bins=40, stat='density')
    sns.kdeplot(train_df[feature], color='red', linewidth=2)
    plt.title(f'Distribution of {feature}')
    plt.grid(True)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()


# Run Optuna trials for tuning.
# study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
# study.optimize(objective, n_trials=100, show_progress_bar=True)

# Best trial results
# print("Best Trial RMSLE:", study.best_value)
# print("Best Hyperparameters:", study.best_params)

# Store best parameters for reuse
# best_params = study.best_params
best_params = {'learning_rate': 0.024861765977324476, 'max_iter': 2805, 'max_leaf_nodes': 198, 
               'min_samples_leaf': 33, 'l2_regularization': 0.015288423982051371, 'max_bins': 239}

# K-Fold CV=5 for blended models
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize models
    hgbt = HistGradientBoostingRegressor(**best_params, random_state=42)
    catboost = CatBoostRegressor(depth=7, iterations=2000, cat_features=['Sex'], random_seed=42, verbose=100)

    # Fit both models
    hgbt.fit(X_train, y_train)
    catboost.fit(X_train, y_train)

    # Predict
    preds_hgbt = hgbt.predict(X_val)
    preds_catboost = catboost.predict(X_val)

    # Blend
    blended_preds = 0.4 * preds_catboost + 0.6 * preds_hgbt

    # Score
    score = rmsle(y_val, blended_preds)
    scores.append(score)

    print(f"Fold {fold+1} RMSLE: {score:.5f}")

# Final CV score
print("-" * 40)
print(f"Mean RMSLE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")

# Refit both models on full data for final test prediction
hgbt_final = HistGradientBoostingRegressor(**best_params, random_state=42)
catboost_final = CatBoostRegressor(depth=7, iterations=2000, cat_features=['Sex'], random_seed=42, verbose=100)

hgbt_final.fit(X, y)
catboost_final.fit(X, y)

# Predict on test set (still log1p scale)
final_preds_hgbt = hgbt_final.predict(test_df)
final_preds_catboost = catboost_final.predict(test_df)

# Blend + inverse log1p
final_preds = np.expm1(0.4 * final_preds_catboost + 0.6 * final_preds_hgbt)


# Submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = final_preds
submission.to_csv('submission.csv', index=False)
display(submission)
print('Submission file saved.')

