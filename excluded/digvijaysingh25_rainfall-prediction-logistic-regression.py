import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', usecols=lambda x: x not in ['id', 'day'])
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', usecols=lambda x: x not in ['id', 'day'])


# Calculate the total missing values in train and test
print(f"Total missing values in the train dataset: {train.isnull().sum().sum()}")
print(f"Total missing values in the test dataset:  {test.isnull().sum().sum()}")

# Replace missing values in the 'winddirection' column of the test dataset with its mean
test.winddirection = test.winddirection.fillna(test.winddirection.mean())


# Return the number of duplicated rows in the train and test DataFrames, respectively
train.duplicated().sum(), test.duplicated().sum()


# Visualize the frequency distribution of the 'rainfall' column in the train dataset
plt.figure(figsize=(5,4))
sns.countplot(data=train, x='rainfall')
plt.title("Distribution of Rainfall (Train)")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Optionally set a global style/theme for Seaborn:
sns.set_style("whitegrid")  # you can try "darkgrid", "white", etc.

def compare_train_test_distribution(feature, train_df, test_df):
    """
    Plots the distribution (hist + kde) and boxplot of a given numeric feature
    from train and test data side by side, with a custom color palette and style.
    """
    # Ensure the feature is in both dataframes
    if feature not in train_df.columns or feature not in test_df.columns:
        print(f"{feature} not found in train or test.")
        return
    
    # Combine for plotting
    combined = pd.concat([
        train_df.assign(Dataset='Train'),
        test_df.assign(Dataset='Test')
    ], axis=0).reset_index(drop=True)
    
    # Create a figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Updated color palette
    palette = ["#2b2d42", "#ef233c"]  # Feel free to replace with colors of your choice
    
    # Distribution plot (Histogram + KDE)
    sns.histplot(
        data=combined,
        x=feature,
        hue='Dataset',
        kde=True,
        bins=30,
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title(f'Distribution of {feature}')
    
    # Boxplot
    sns.boxplot(
        data=combined,
        x=feature,
        y='Dataset',
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title(f'Boxplot of {feature}')
    
    plt.tight_layout()
    plt.show()

for col in train.columns:
    compare_train_test_distribution(col, train, test)



plt.figure(figsize=(10,8))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Correlation Heatmap - Train Set")
plt.show()


sns.pairplot(train, hue='rainfall', diag_kind='kde')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()


plt.figure(figsize=(14, 10))
for i, col in enumerate(train.columns[:-1], 1):
    plt.subplot(3, 4, i)
    sns.kdeplot(train[col][train['rainfall'] == 1], color='red', label='Rainfall: 1')
    sns.kdeplot(train[col][train['rainfall'] == 0], color='blue', label='Rainfall: 0')
    plt.title(f'Distribution of {col} by Rainfall')
    plt.legend()
plt.tight_layout()
plt.show()


# some new features
train['humidity_cloud_interaction'] = train['humidity'] * train['cloud']
train['humidity_sunshine_interaction'] = train['humidity'] * train['sunshine']
train['cloud_sunshine_ratio'] = train['cloud'] / (train['sunshine'] + 1e-5)
train['relative_dryness'] = 100 - train['humidity']
train['sunshine_percentage'] = train['sunshine'] / (train['sunshine'] + train['cloud'] + 1e-5)
train['weather_index'] = (0.4 * train['humidity']) + (0.3 * train['cloud']) - (0.3 * train['sunshine'])

test['humidity_cloud_interaction'] = test['humidity'] * test['cloud']
test['humidity_sunshine_interaction'] = test['humidity'] * test['sunshine']
test['cloud_sunshine_ratio'] = test['cloud'] / (test['sunshine'] + 1e-5)
test['relative_dryness'] = 100 - test['humidity']
test['sunshine_percentage'] = test['sunshine'] / (test['sunshine'] + test['cloud'] + 1e-5)
test['weather_index'] = (0.4 * test['humidity']) + (0.3 * test['cloud']) - (0.3 * test['sunshine'])


# Prepare dataset
X = train.drop(['rainfall'], axis=1)
y = train['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


param_grid = [
    # 1) L1 or L2 penalty with liblinear solver
    {
        'penalty': ['l1', 'l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],  # liblinear supports only l1 or l2
        'class_weight': [None, 'balanced'],
        'max_iter': [50, 100, 300, 500]
    },
    # 2) L2 penalty with lbfgs, sag, saga
    {
        'penalty': ['l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],  # these solvers handle l2
        'class_weight': [None, 'balanced'],
        'max_iter': [10, 20, 30, 40, 50, 100, 300, 500]
    },
    # 3) Elasticnet penalty with saga solver (only combo that supports elasticnet)
    {
        'penalty': ['elasticnet'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [50, 100, 300, 500],
        'l1_ratio': [0, 0.5, 1]  # only relevant for elasticnet
    }
]

log_reg = LogisticRegression()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(log_reg, param_grid, cv=skf, scoring='roc_auc', verbose=3, n_jobs=-1)

grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# Get the best model from the grid search
best_model = grid_search.best_estimator_

# 1. Generate predicted probabilities for the positive class
y_proba_test = best_model.predict_proba(X_test)[:, 1]

print("ROC AUC on the test set:", roc_auc_score(y_test, y_proba_test))


y_pred = best_model.predict_proba(test)[:, 1]
submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission['rainfall'] = y_pred
submission.to_csv('submission.csv', index=False)


submission




