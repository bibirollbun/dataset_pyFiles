# Install Optuna
# !pip install optuna --quiet


# Author: Aaron Isom
# Kaggle Predict Calorie Expenditure

import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# RMSLE function
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred) # Ensure non-negative
    
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Basal Metabolic Rate function
def add_bmr(df, weight_col='Weight', height_col='Height', age_col='Age', sex_col='Sex'):
    """
    Adds a BMR (Basal Metabolic Rate) column using the Mifflin-St Jeor equation:
    
    For men:    BMR = 10 * weight + 6.25 * height - 5 * age + 5  
    For women:  BMR = 10 * weight + 6.25 * height - 5 * age - 161

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - weight_col (str): Weight in kilograms
    - height_col (str): Height in centimeters
    - age_col (str): Age in years
    - sex_col (str): Should be 'male' or 'female' (case-insensitive)

    Returns:
    - pd.DataFrame with a new column 'BMR'
    """
    # Normalize 
    df[sex_col] = df[sex_col].str.lower()

    # Calculate BMR based on Sex
    df['BMR'] = (
        10 * df[weight_col] +
        6.25 * df[height_col] -
        5 * df[age_col] +
        df[sex_col].map({'male': 5, 'female': -161})
    )

    return df

# Calculate BMI Category
def add_bmi_category(df, bmi_col='BMI'):
    """
    Adds a BMI category column to the DataFrame based on standard WHO ranges.

    Categories:
    - Underweight: BMI < 18.5
    - Healthy weight: 18.5 ≤ BMI < 24.9
    - Overweight: 25 ≤ BMI < 29.9
    - Obesity: BMI ≥ 30
    """
    bins = [0, 18.5, 24.9, 29.9, float('inf')]
    labels = ['Underweight', 'Healthy', 'Overweight', 'Obese']
    df['BMI_Category'] = pd.cut(df[bmi_col], bins=bins, labels=labels)
    df['BMI_Category'] = df['BMI_Category'].astype('category')
    
    return df

def add_met_intensity_level(df, met_col='Estimated_MET'):
    """
    Adds a categorical activity intensity level based on MET:
    - Light: < 3.0
    - Moderate: 3.0–5.9
    - Vigorous: 6.0–8.9
    - Very Vigorous: ≥ 9.0
    """
    bins = [0, 3.0, 6.0, 9.0, float('inf')]
    labels = ['Light', 'Moderate', 'Vigorous', 'Very Vigorous']
    df['MET_Intensity'] = pd.cut(df[met_col], bins=bins, labels=labels)
    df['MET_Intensity'] = df['MET_Intensity'].astype('category')
    
    return df


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


# Create the RMSLE scorer
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Load data
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

X = train_df.drop(['Calories'], axis=1, errors='ignore')
y = np.log1p(train_df['Calories'])  # Instead of raw Calories

# Tuned using Optuna one of my other notbooks
best_params = {'learning_rate': 0.024861765977324476, 'max_iter': 2805, 'max_leaf_nodes': 198, 
               'min_samples_leaf': 33, 'l2_regularization': 0.015288423982051371, 'max_bins': 239}

# StandardScaler for SGDRegressor
sgd_pipe = make_pipeline(StandardScaler(), SGDRegressor(alpha=0.0001, random_state=42, max_iter=1000, tol=1e-3))

# Base models
estimators = [
    ('hgbt', HistGradientBoostingRegressor(**best_params, random_state=42)),
    ('cat', CatBoostRegressor(depth=7, iterations=1000, cat_features=['Sex'], random_seed=42, verbose=0)),
    ('sgd', sgd_pipe)
]

# Ridge as meta-model
meta_model = Ridge(alpha=1.0, random_state=42)

# Stacking Regressor setup
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

# Commented to speed up submission after testing
# Cross-validate stacking model for RMSLE
# cv_scores = cross_val_score(stack, X, y, cv=5, scoring=rmsle_scorer)
# print("Stacking Regressor CV RMSLE (per fold):", -cv_scores)
# print(f"Mean RMSLE: {-cv_scores.mean():.5f} ± {cv_scores.std():.5f}")

# Fit on full data, predict on test
stack.fit(X, y)
stack_preds = np.expm1(stack.predict(test_df))
stack_preds = np.maximum(0, np.round(stack_preds, 2))

# Submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = stack_preds
submission.to_csv('submission.csv', index=False)
display(submission)
print('Submission file saved.')

