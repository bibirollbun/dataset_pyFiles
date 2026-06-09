import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


print(f"Dataset Shape: {train.shape}")
print(f'Features: {train.columns.tolist()}')


train.info()


train.describe()


print("\nğŸ”� Missing Values:")
missing_values = train.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found!")


print("\nğŸ”¢ Data Types:")
print(train.dtypes)


target_col = 'BeatsPerMinute'
features = [col for col in train.columns if col not in ['id', target_col]]


features


fig = plt.figure(figsize=(20, 25))
plt.subplot(4, 3, 1)
if target_col in train.columns:
    plt.hist(train[target_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequencey')

    # Add statistics
    plt.axvline(train[target_col].mean(), color='red', linestyle='--', label=f'Mean: {train[target_col].mean():.2f}')
    plt.axvline(train[target_col].median(), color='green', linestyle='--', label=f'Median: {train[target_col].median():.2f}')
    plt.legend()

# 2. Feature Distribution
# fig = plt.figure(figsize=(20, 25))
for i, feature in enumerate(features[:11], 2):
    plt.subplot(4, 3, i)
    plt.hist(train[feature], bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Box plots for outlier detection
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Box Plots for Outlier Detection', fontsize=16)

for i, feature in enumerate(features):
    row, col = i // 4, i % 4
    if row < 3:
        axes[row, col].boxplot(train[feature].dropna())
        axes[row, col].set_title(f'{feature}')
        axes[row, col].tick_params(axis='x', rotation=45)
            
# Remove empty subplots
for i in range(len(features), 12):
    row, col = i // 4, i % 4
    if row < 3:
        fig.delaxes(axes[row, col])
plt.tight_layout()
plt.show()


if target_col in train.columns:
    corr_features = features + [target_col]
    corr_matrix = train[corr_features].corr()

    # Correlation with target
    target_corr = corr_matrix[target_col].sort_values(key=abs, ascending=False)
    target_corr_signed = corr_matrix[target_col].loc[target_corr.index]
    print(f"\nğŸ“Š Correlation with {target_col}:")
    for feature in target_corr.index:
        corr_value = target_corr_signed[feature]
        print(f"   {feature}: {corr_value:.4f}")

    # Heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        square=True,
        mask=mask,
        cbar_kws={'shrink': .8}
    )
    plt.title('Correlation Heatmap of Features')
    plt.tight_layout()
    plt.show()

    # Feature correlation with target (bar plot)
    plt.figure(figsize=(10, 6))
    target_corr_for_plot = target_corr_signed[1:]  # Exclude self-correlation
    target_corr_abs_for_plot = target_corr_for_plot.abs().sort_values(ascending=True)
    colors = ['red' if x < 0 else 'blue' for x in target_corr_for_plot[target_corr_abs_for_plot.index]]
    target_corr_abs_for_plot.plot(kind='barh', color=colors)
    plt.title(f'Feature Correlation with {target_col} (Absolute Values)')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.show()


corr_matrix


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 500 

corr_matrix = train[features + [target_col]].corr()
top_features = corr_matrix[target_col].abs().sort_values(ascending=False)[1:5].index
    
fig, axes = plt.subplots(2, 2, figsize=(20, 25))
fig.suptitle(f'Scatter Plots: Top Correlated Features vs {target_col}', fontsize=16)

for i, feature in enumerate(top_features):
    row, col = i // 2, i % 2
    axes[row, col].scatter(train[feature], train[target_col], alpha=0.5, color=f'C{i}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel(target_col)
    axes[row, col].set_title(f'{feature} vs {target_col}')

    # Add trend line
    z = np.polyfit(train[feature], train[target_col], 1)
    p = np.poly1d(z)
    axes[row, col].plot(train[feature], p(train[features]), 'r--', alpha=0.8)
plt.tight_layout()
plt.show()

# Pair plot for top features
top_features_list = list(top_features[:4]) + [target_col]
plt.figure(figsize=(12, 10))
sns.pairplot(train[top_features_list], diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Top Correlated Features', y=1.02)
plt.show()




