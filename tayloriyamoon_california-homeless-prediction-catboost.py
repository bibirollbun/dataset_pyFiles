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
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Feature selection
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# 1. Load Data
print("Loading data...")
train = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/test.csv')
sample_sub = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 2. Initial Data Exploration
print("\nFirst few rows of training data:")
print(train.head())

print("\nColumn names:")
print(train.columns.tolist())

print("\nData types:")
print(train.dtypes.value_counts())

# Identify target variable (should be the column that's in train but not in test)
target_col = [col for col in train.columns if col not in test.columns][0]
print(f"\nTarget variable: {target_col}")

# Check for missing values
print("\nMissing values in train:")
print(train.isnull().sum().sum())
print("\nMissing values in test:")
print(test.isnull().sum().sum())

# 3. Feature Engineering
print("\nPerforming feature engineering...")

# Separate features and target
X_train = train.drop(columns=[target_col])
y_train = train[target_col]
X_test = test.copy()

# Store ID column if exists
id_col = None
for col in ['id', 'ID', 'Id']:
    if col in X_train.columns:
        id_col = col
        train_ids = X_train[id_col]
        test_ids = X_test[id_col]
        X_train = X_train.drop(columns=[id_col])
        X_test = X_test.drop(columns=[id_col])
        break

# Create demographic ratio features based on the ACS tables mentioned
feature_cols = X_train.columns.tolist()

# Age-based ratios
age_cols = [col for col in feature_cols if 'B01001' in col]
if age_cols and 'B01001_001E' in feature_cols:  # Total population
    total_pop = X_train['B01001_001E'].replace(0, 1)  # Avoid division by zero
    total_pop_test = X_test['B01001_001E'].replace(0, 1)
    
    # Youth ratio (under 25)
    youth_cols = [col for col in age_cols if any(x in col for x in ['003E', '004E', '005E', '006E', '007E', '008E', '009E'])]
    if youth_cols:
        X_train['youth_ratio'] = X_train[youth_cols].sum(axis=1) / total_pop
        X_test['youth_ratio'] = X_test[youth_cols].sum(axis=1) / total_pop_test
    
    # Senior ratio (65+)
    senior_cols = [col for col in age_cols if any(x in col for x in ['019E', '020E', '021E', '022E', '023E', '024E', '025E'])]
    if senior_cols:
        X_train['senior_ratio'] = X_train[senior_cols].sum(axis=1) / total_pop
        X_test['senior_ratio'] = X_test[senior_cols].sum(axis=1) / total_pop_test

# Veteran ratios
vet_cols = [col for col in feature_cols if 'B21001' in col]
if 'B21001_002E' in feature_cols and 'B21001_001E' in feature_cols:  # Veterans and total civilian 18+
    X_train['veteran_ratio'] = X_train['B21001_002E'] / X_train['B21001_001E'].replace(0, 1)
    X_test['veteran_ratio'] = X_test['B21001_002E'] / X_test['B21001_001E'].replace(0, 1)

# Disability ratios
disability_cols = [col for col in feature_cols if 'B18101' in col]
if disability_cols and 'B18101_001E' in feature_cols:
    disability_yes_cols = [col for col in disability_cols if any(x in col for x in ['004E', '007E', '010E', '013E', '016E', '019E'])]
    if disability_yes_cols:
        X_train['disability_ratio'] = X_train[disability_yes_cols].sum(axis=1) / X_train['B18101_001E'].replace(0, 1)
        X_test['disability_ratio'] = X_test[disability_yes_cols].sum(axis=1) / X_test['B18101_001E'].replace(0, 1)

# Family household ratios
if 'B11003_001E' in feature_cols and 'B11003_003E' in feature_cols:
    X_train['family_with_children_ratio'] = X_train['B11003_003E'] / X_train['B11003_001E'].replace(0, 1)
    X_test['family_with_children_ratio'] = X_test['B11003_003E'] / X_test['B11003_001E'].replace(0, 1)

# Education ratios (low education)
edu_cols = [col for col in feature_cols if 'B15003' in col]
if edu_cols and 'B15003_001E' in feature_cols:
    low_edu_cols = [col for col in edu_cols if any(x in col for x in ['002E', '003E', '004E', '005E', '006E', '007E', '008E', '009E', '010E', '011E', '012E', '013E', '014E', '015E', '016E'])]
    if low_edu_cols:
        X_train['low_education_ratio'] = X_train[low_edu_cols].sum(axis=1) / X_train['B15003_001E'].replace(0, 1)
        X_test['low_education_ratio'] = X_test[low_edu_cols].sum(axis=1) / X_test['B15003_001E'].replace(0, 1)

# Race/ethnicity diversity index
race_cols = [col for col in feature_cols if 'B03002' in col and col != 'B03002_001E']
if race_cols and 'B03002_001E' in feature_cols:
    # Calculate diversity using Shannon entropy
    race_data = X_train[race_cols].div(X_train['B03002_001E'].replace(0, 1), axis=0)
    race_data_test = X_test[race_cols].div(X_test['B03002_001E'].replace(0, 1), axis=0)
    
    # Replace negative values and infinities with 0
    race_data = race_data.replace([np.inf, -np.inf], 0).fillna(0)
    race_data_test = race_data_test.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate entropy
    entropy = -((race_data * np.log(race_data + 1e-10)).sum(axis=1))
    entropy_test = -((race_data_test * np.log(race_data_test + 1e-10)).sum(axis=1))
    
    X_train['diversity_index'] = entropy
    X_test['diversity_index'] = entropy_test

# Handle any infinities or NaN values created
X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

print(f"Features after engineering: {X_train.shape[1]}")

# 4. Feature Selection
print("\nPerforming feature selection...")

# Remove constant features
constant_features = []
for col in X_train.columns:
    if X_train[col].nunique() == 1:
        constant_features.append(col)

if constant_features:
    print(f"Removing {len(constant_features)} constant features")
    X_train = X_train.drop(columns=constant_features)
    X_test = X_test.drop(columns=constant_features)

# Select top features using mutual information
selector = SelectKBest(score_func=mutual_info_regression, k=min(100, X_train.shape[1]))
selector.fit(X_train, y_train)
feature_scores = pd.DataFrame({
    'feature': X_train.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("\nTop 20 features by mutual information:")
print(feature_scores.head(20))

# Keep top features
top_features = feature_scores.nlargest(min(100, len(feature_scores)), 'score')['feature'].tolist()
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# 5. Model Training
print("\nTraining models...")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Define models
models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.001, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
    'CatBoost': CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, random_state=42, verbose=False)
}

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    cv_scores[name] = {
        'mean': rmse_scores.mean(),
        'std': rmse_scores.std()
    }
    print(f"{name}: RMSE = {rmse_scores.mean():.6f} (+/- {rmse_scores.std():.6f})")

# Select best model
best_model_name = min(cv_scores, key=lambda x: cv_scores[x]['mean'])
print(f"\nBest model: {best_model_name}")

# 6. Final Training and Predictions
print("\nTraining final models for ensemble...")

# Train top 3 models for ensemble
top_models = sorted(cv_scores.items(), key=lambda x: x[1]['mean'])[:3]
predictions = []

for name, _ in top_models:
    print(f"Training {name}...")
    model = models[name]
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    predictions.append(pred)

# Ensemble predictions (weighted average based on CV scores)
weights = []
for name, _ in top_models:
    # Lower RMSE gets higher weight
    weight = 1 / cv_scores[name]['mean']
    weights.append(weight)

weights = np.array(weights) / np.sum(weights)
final_predictions = np.average(predictions, weights=weights, axis=0)

# 7. Create Submission
print("\nCreating submission file...")

# Ensure predictions match submission format
if id_col:
    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: final_predictions
    })
else:
    submission = pd.DataFrame({
        'index': range(len(final_predictions)),
        target_col: final_predictions
    })

# Match sample submission format
submission.columns = sample_sub.columns
submission.to_csv('submission.csv', index=False)

print("\nSubmission file created!")
print(f"Submission shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head(10))

# 8. Feature Importance from best tree-based model
if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']:
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': top_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(feature_importance.head(20))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        feature_importance.head(20).plot(x='feature', y='importance', kind='barh')
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

print("\nProcess completed successfully!")

