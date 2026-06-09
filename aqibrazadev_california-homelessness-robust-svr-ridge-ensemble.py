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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Configuration
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
np.random.seed(42)

print("Libraries Loaded Successfully")


# Load Data
train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

print(f"Train Shape: {train.shape} | Test Shape: {test.shape}")


train.head()


train.info()


# Load Data
train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

print(f"Train Shape: {train.shape} | Test Shape: {test.shape}")

# --- INTERACTIVE EDA: Target Distribution ---
fig = px.histogram(
    train, 
    x='HOMELESS_RATE', 
    nbins=50, 
    title='Distribution of Homeless Rate (Target)',
    color_discrete_sequence=['#636EFA'],
    marginal='box' # Adds a boxplot on top to see outliers
)
fig.update_layout(showlegend=False, xaxis_title="Homeless Rate", yaxis_title="Count")
fig.show()

# --- INTERACTIVE EDA: Correlation Heatmap ---
# We select numeric columns and calculate correlation
corr_matrix = train.select_dtypes(include=[np.number]).corr()

# Filter for top correlations with Target to reduce noise
top_corr_features = corr_matrix['HOMELESS_RATE'].abs().sort_values(ascending=False).head(15).index
filtered_corr = train[top_corr_features].corr()

fig_corr = px.imshow(
    filtered_corr,
    text_auto='.2f',
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title='Top 15 Features Correlated with Homeless Rate'
)
fig_corr.show()


def engineer_features(df):
    df_eng = df.copy()
    
    # 1. Vulnerability Indices (Aggregating related columns)
    df_eng['High_Risk_Age_Sum'] = (
        df_eng['AGE_U18_PCT'] + df_eng['AGE_65_69_PCT'] + 
        df_eng['AGE_70_79_PCT'] + df_eng['AGE_80_PLUS_PCT']
    )
    
    df_eng['Social_Isolation_Index'] = (
        df_eng['INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT'] + 
        df_eng['NONFAMILY_SINGLE_MALE_PCT'] + 
        df_eng['NONFAMILY_SINGLE_FEMALE_PCT']
    )

    # 2. Economic Stressors (Ratios)
    # Adding 1e-6 to avoid division by zero
    df_eng['Dependency_Ratio'] = df_eng['High_Risk_Age_Sum'] / (df_eng['AGE_25_34_PCT'] + df_eng['AGE_35_44_PCT'] + 1e-6)
    
    # 3. Structural Diversity
    # Simpson's Diversity Index calculation for Race columns
    race_cols = [c for c in df.columns if 'RACE' in c]
    df_eng['Diversity_Index'] = 1 - (df_eng[race_cols] / 100).pow(2).sum(axis=1)
    
    # 4. Interaction Features (Combining strong predictors)
    # Validated based on your previous SVR success
    if 'DISABILITY_POP_PCT' in df.columns and 'VETERAN_POP_PCT' in df.columns:
        df_eng['Disability_Veteran_Interaction'] = df_eng['DISABILITY_POP_PCT'] * df_eng['VETERAN_POP_PCT']
    
    if 'TOTAL_HOUSEHOLDS_PCT' in df.columns:
         df_eng['Household_Density'] = 100 / (df_eng['TOTAL_HOUSEHOLDS_PCT'] + 1e-6)

    # 5. Log Transformations for Skewed Features
    # Log transform helps model handle outliers better
    skew_cols = ['TOTAL_HOUSEHOLDS_PCT', 'VETERAN_POP_PCT']
    for col in skew_cols:
        if col in df.columns:
            df_eng[f'Log_{col}'] = np.log1p(df_eng[col])
            
    return df_eng

# Apply Engineering
X = train.drop(['ID', 'HOMELESS_RATE'], axis=1)
y = train['HOMELESS_RATE']
X_test_raw = test.drop('ID', axis=1)

X_eng = engineer_features(X)
X_test_eng = engineer_features(X_test_raw)

print(f"Features after engineering: {X_eng.shape[1]}")


# 1. Remove Low Variance and Highly Correlated Features
# We use a threshold of 0.95 to drop redundant features
corr_matrix = X_eng.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

X_reduced = X_eng.drop(columns=to_drop)
X_test_reduced = X_test_eng.drop(columns=to_drop)

print(f"Dropped {len(to_drop)} redundant features.")

# 2. Robust Scaling
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_reduced), columns=X_reduced.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_reduced), columns=X_reduced.columns)

print("Data Scaled Successfully.")


# Define diverse models
models = {
    'SVR': SVR(C=1.0, epsilon=0.01, kernel='rbf'), # Good for non-linear trends
    'Huber': HuberRegressor(epsilon=1.35), # Robust to outliers
    'Ridge': Ridge(alpha=10), # Regularized linear regression
    'XGB': XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42),
    'LGBM': LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=20, random_state=42)
}

# Training Loop with Cross-Validation
results = {}
preds_test = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Starting Training...\n")
for name, model in models.items():
    # Cross Validation Score
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_root_mean_squared_error')
    avg_rmse = -cv_scores.mean()
    results[name] = avg_rmse
    
    # Train on full data for submission
    model.fit(X_scaled, y)
    preds_test[name] = model.predict(X_test_scaled)
    
    print(f"{name:10} | CV RMSE: {avg_rmse:.6f}")


# Visualize Model Performance
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'RMSE']).sort_values('RMSE')

fig_res = px.bar(
    results_df, 
    x='RMSE', 
    y='Model', 
    orientation='h',
    title='Model Comparison (Lower RMSE is Better)',
    color='RMSE',
    color_continuous_scale='Viridis_r'
)
fig_res.show()


# Calculate weights: Inverse of RMSE (Better score = Higher weight)
weights = {k: 1/v for k, v in results.items()}
total_weight = sum(weights.values())
norm_weights = {k: v/total_weight for k, v in weights.items()}

print("\nEnsemble Weights:")
for m, w in norm_weights.items():
    print(f"{m}: {w:.4f}")

# Compute Final Predictions
final_preds = np.zeros(len(test))
for name, pred in preds_test.items():
    final_preds += pred * norm_weights[name]

# Sanity Check: Ensure no negative predictions (Homeless rate can't be negative)
final_preds = np.maximum(final_preds, 0)



# Create Submission File
submission = pd.DataFrame({
    'ID': test['ID'],
    'HOMELESS_RATE': final_preds
})

submission.to_csv('submission_enhanced.csv', index=False)

# Final Visualization: Predicted Distribution
fig_final = px.histogram(
    final_preds, 
    nbins=50, 
    title='Distribution of Final Predictions',
    labels={'value': 'Predicted Homeless Rate'},
    color_discrete_sequence=['#00CC96']
)
fig_final.show()

print("\n✓ Submission saved as 'submission_enhanced.csv'")




