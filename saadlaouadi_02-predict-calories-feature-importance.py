# install dependencies
!pip install -q watermark


#------------------------------------------------------------------------------
#   Project: Calorie Expenditure Prediction
#
#   Description:    Feature Importance 
# 
#   Author:         Dr. Saad Laouadi
#   
#   Created:        May 14, 2025
#   Last Modified:  May 14, 2025
#   Version:        1.0.0
#------------------------------------------------------------------------------


# *************************************************
#                Environment Setup
# *************************************************

# Standard libraries
import os
import sys
import time
import pathlib
import warnings
from datetime import datetime

# Data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.pyplot import figure

# modeling tools
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression


# Configure warnings
warnings.filterwarnings('ignore')

# Pandas display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', '{:.3f}'.format)


# Matplotlib and seaborn configuration
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
sns.set_context("notebook", font_scale=1.2)


# Display notebook information
%reload_ext watermark
%watermark -iv -v -m -p pandas,numpy,matplotlib,seaborn -ud -a "Dr. Saad Laouadi"

print("\nEnvironment setup completed successfully.")


# CONFIGURATION: File paths
INPUT_DIR = pathlib.Path("/kaggle/input/playground-series-s5e5").resolve()
TRAIN_PATH = INPUT_DIR.joinpath('train.csv')


# *************************************************
#         Preliminary Feature Importance
# *************************************************

print("Preliminary Feature Importance Analysis")
print("=" * 50)

# Load the train data
df = pd.read_csv(TRAIN_PATH)

# Check data info
df.info()


# Let us map the Sex feature
df['Sex'] = df['Sex'].map({'male': 1, 'female':2})


# Set 'Calories' as the target variable
target_col = 'Calories'
X = df.drop([target_col] + ['id'], axis=1)
y = df[target_col]

print(f"Target variable: {target_col}")
print(f"Number of features: {X.shape[1]}")


# 1. Statistical Tests - F-value from ANOVA
print("\n1. Statistical Tests (F-value from ANOVA)")
print("-" * 30)

# Calculate F-statistics
f_values, p_values = f_regression(X, y)

# Create a DataFrame for better visualization
f_test_results = pd.DataFrame({
    'Feature': X.columns,
    'F_value': f_values,
    'p_value': p_values
})

# Sort by F-value in descending order
f_test_results = f_test_results.sort_values('F_value', ascending=False)

print("Top features by F-value (ANOVA):")
display(f_test_results.head(10))


# Visualize the F-values
# =====================

plt.figure(figsize=(12, 6))
sns.barplot(x='F_value',
            y='Feature',
            data=f_test_results.head(10))

plt.title('Feature Importance by F-value')
plt.tight_layout()

plt.show()


# Identify statistically significant features
significant_features = f_test_results[f_test_results['p_value'] < 0.05]
print(
    f"\nStatistically significant features (p < 0.05):"
    f" {len(significant_features)} out of {len(f_test_results)}"
)


# 2. Basic Model Feature Importance
#    Using Random Forest Algorithm:
# =================================:

print("\n2. Basic Model Feature Importance")
print("-" * 30)

# Train a simple random forest model
rf_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X, y)

# Get feature importance
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
rf_importance = rf_importance.sort_values('Importance', ascending=False)

print("Random Forest Feature Importance:")
display(rf_importance.head(10))


# Plot feature importance
# ========================

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance',
            y='Feature',
            data=rf_importance.head(10))

plt.title('Feature Importance by Random Forest')
plt.tight_layout()

plt.show()


# Check predictive performance
# ===========================

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"\nRandom Forest Cross-Validation R² scores: {cv_scores}")
print(f"Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")


# 3. Calculate mutual information
# ===============================
print("\n3. Mutual Information")
print("-" * 30)

mi_scores = mutual_info_regression(X, y, random_state=42)

mi_results = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
})
mi_results = mi_results.sort_values('MI_Score', ascending=False)

print("Mutual Information Scores:")
display(mi_results.head(10))


# Plot mutual information
# =======================
plt.figure(figsize=(12, 6))

sns.barplot(x='MI_Score',
            y='Feature',
            data=mi_results.head(10))

plt.title('Feature Importance by Mutual Information')
plt.tight_layout()

plt.show()


# 4. Compare importance rankings across methods
# =============================================

print("\n4. Comparison of Feature Importance Methods:")
print("-" * 44)

f_test_results.set_index('Feature')['F_value'].rank(ascending=False)


# Create a DataFrame with features and initialize with empty rankings
importance_comparison = pd.DataFrame({'Feature': X.columns})

# First, create DataFrames with the correct feature-to-rank mapping
f_rank_df = pd.DataFrame({'Feature': f_test_results['Feature'],
                          'F_Value_Rank': f_test_results['F_value'].rank(ascending=False)})

rf_rank_df = pd.DataFrame({'Feature': rf_importance['Feature'],
                           'RF_Importance_Rank': rf_importance['Importance'].rank(ascending=False)})

mi_rank_df = pd.DataFrame({'Feature': mi_results['Feature'],
                           'MI_Score_Rank': mi_results['MI_Score'].rank(ascending=False)})

# Merge these into the main DataFrame
importance_comparison = importance_comparison.merge(f_rank_df, on='Feature', how='left')
importance_comparison = importance_comparison.merge(rf_rank_df, on='Feature', how='left')
importance_comparison = importance_comparison.merge(mi_rank_df, on='Feature', how='left')

# Calculate average rank
importance_comparison['Avg_Rank'] = importance_comparison[['F_Value_Rank', 'RF_Importance_Rank', 'MI_Score_Rank']].mean(axis=1)

# Sort by average rank
importance_comparison = importance_comparison.sort_values('Avg_Rank')
importance_comparison


# Top features across all methods (average rank <= 5)
top_5_features = importance_comparison[importance_comparison['Avg_Rank'] <= 5]['Feature'].tolist()
top_5_features


print(f"\nTop 5 features by average rank:")
for i, feature in enumerate(top_5_features, 1):
    avg_rank = importance_comparison.loc[importance_comparison['Feature'] == feature, 'Avg_Rank'].values[0]
    print(f"  {i}. {feature} (Average Rank: {avg_rank:.2f})")


# End of notebook! see you in another notebook where we discuss another topic

