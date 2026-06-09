# import
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.feature_selection import mutual_info_regression

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import optuna
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

import itertools


# define the hp
train_path = '/kaggle/input/playground-series-s5e2/train.csv'
train_extra_path = '/kaggle/input/playground-series-s5e2/training_extra.csv'
test_path = '/kaggle/input/playground-series-s5e2/test.csv'

rd_state = 1234


# load the data
train_data = pd.read_csv(train_path)
train_extra_data = pd.read_csv(train_extra_path)
test_data = pd.read_csv(test_path)


train_data.head()


# where is NaN
train_nan_counts = train_data.isnull().sum()
train_extra_nan_counts = train_extra_data.isnull().sum()
print(train_nan_counts, '\n', train_extra_nan_counts)


# fill the NaN
def fill_with_mode(data, group_col, target_col):
    mode_fill = data.groupby(group_col)[target_col].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
    data.fillna({target_col: mode_fill}, inplace=True)


fill_with_mode(train_data, 'Compartments', 'Size')
fill_with_mode(train_data, 'Compartments', 'Brand')

fill_with_mode(train_data, 'Brand', 'Material')
fill_with_mode(train_data, 'Size', 'Weight Capacity (kg)')
fill_with_mode(train_data, 'Brand', 'Color')
fill_with_mode(train_data, 'Brand', 'Style')
fill_with_mode(train_data, 'Size', 'Laptop Compartment')
fill_with_mode(train_data, 'Brand', 'Waterproof')

train_nan_counts = train_data.isnull().sum()
print(train_nan_counts)


train_data['Laptop Compartment'] = train_data['Laptop Compartment'].replace({'Yes': True, 'No': False})
train_data['Waterproof'] = train_data['Waterproof'].replace({'Yes': True, 'No': False})


# Encoding
def encode_features(df, onehot_cols=None, ordinal_cols=None, ordinal_order=None):

    df = df.copy()  

    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)  
    
    if ordinal_cols and ordinal_order:
        for col in ordinal_cols:
            if col in ordinal_order:
                encoder = OrdinalEncoder(categories=[ordinal_order[col]])
                df[col] = encoder.fit_transform(df[[col]])
    
    return df


onehot_features = ['Brand', 'Material', 'Style', 'Color']

ordinal_features = ['Size']
ordinal_mapping = {'Size': ['Small', 'Medium', 'Large']} 

train_data_encoded = encode_features(train_data, onehot_cols=onehot_features, ordinal_cols=ordinal_features, ordinal_order=ordinal_mapping)

train_data_encoded.head()


# visualization
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

sns.boxplot(x='Size', y='Price', data=train_data, ax=axes[0, 0])
axes[0, 0].set_title('Price Distribution by Size')

sns.boxplot(x='Brand', y='Price', data=train_data, ax=axes[0, 1])
axes[0, 1].set_title('Price Distribution by Brand')

sns.boxplot(x='Material', y='Price', data=train_data, ax=axes[0, 2])
axes[0, 2].set_title('Price Distribution by Material')

sns.boxplot(x='Laptop Compartment', y='Price', data=train_data, ax=axes[1, 0])
axes[1, 0].set_title('Price Distribution by Laptop Compartment')

sns.boxplot(x='Waterproof', y='Price', data=train_data, ax=axes[1, 1])
axes[1, 1].set_title('Price Distribution by Waterproof')

sns.boxplot(x='Style', y='Price', data=train_data, ax=axes[1, 2])
axes[1, 2].set_title('Price Distribution by Style')

sns.boxplot(x='Color', y='Price', data=train_data, ax=axes[2, 0])
axes[2, 0].set_title('Price Distribution by Color')

sns.boxplot(x='Compartments', y='Price', data=train_data, ax=axes[2, 1])
axes[2, 1].set_title('Price Distribution by Compartments')

# sns.boxplot(x='Style', y='Price', data=train_data, ax=axes[2, 2])
# axes[2, 2].set_title('Price Distribution by Style')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 10))

sns.scatterplot(x='Compartments', y='Price', data=train_data, ax=axes[0])
axes[0].set_title('Price vs Compartments')

sns.scatterplot(x='Weight Capacity (kg)', y='Price', data=train_data, ax=axes[1])
axes[1].set_title('Price vs Weight Capacity (kg)')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(train_data_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()


size = train_data_encoded['Size']
weight_capacity = train_data_encoded['Weight Capacity (kg)']
price = train_data_encoded['Price']

fig, ax = plt.subplots(figsize=(10, 8))

hb = ax.hexbin(size, weight_capacity, C=price, gridsize=30, cmap='viridis')

cb = fig.colorbar(hb, ax=ax)
cb.set_label('Price')

ax.set_xlabel('Size')
ax.set_ylabel('Weight Capacity (kg)')

plt.show()


categorical_features = ['Brand','Material','Size','Compartments','Laptop Compartment','Waterproof','Style','Color']        

combinations = itertools.combinations(categorical_features, 2)

plt.figure(figsize=(50, 50))

for i, (feat1, feat2) in enumerate(combinations, 1):
    plt.subplot(len(categorical_features) * (len(categorical_features) - 1) // 2, 1, i)  
    
    pivot_df = train_data.pivot_table(index=feat1, columns=feat2, values='Price', aggfunc='mean')
    
    sns.heatmap(pivot_df, annot=True, cmap='viridis', cbar_kws={'label': 'Price'})
    
    plt.title(f'Price by {feat1} and {feat2}')
    plt.xlabel(feat2)
    plt.ylabel(feat1)

plt.tight_layout()
plt.show()


# define X and y
X = train_data_encoded.drop(['id', 'Price'], axis=1)
y = train_data_encoded['Price']


# MI score
mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)

print(mi_scores)


# PCA
std = StandardScaler()
X_std = std.fit_transform(X)

pca = PCA(n_components=0.95)

reduced_data = pca.fit_transform(X_std)

print(pca.explained_variance_ratio_)
print(pca.n_components_)


# XGBoost
def objective(trial):
    param = {
        'objective': 'reg:squarederror',  
        'eval_metric': 'rmse',        
        'max_depth': trial.suggest_int('max_depth', 3, 10), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), 
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), 
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), 
        'gamma': trial.suggest_float('gamma', 0, 5), 
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 
        'random_state': rd_state,
    }
    model = xgb.XGBRegressor(**param)  
    score = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    return score.mean()


study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

study.optimize(objective, n_trials=100, n_jobs=-1)

print("XGB Best hyperparameters: ", study.best_params)
print("XGB Best accuracy: ", study.best_value)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),  
        'depth': trial.suggest_int('depth', 4, 10),  
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),  
        'model_size_reg': trial.suggest_float('model_size_reg', 0.5, 10.0), 
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'RMSE',  
        'verbose': False,
        'random_state': rd_state,
    }
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, n_jobs=-1)  

print("Catboost Best hyperparameters: ", study.best_params)
print("Catboost Best accuracy: ", study.best_value)


def objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 1000),  
        'max_depth': trial.suggest_int('max_depth', 3, 10), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),  
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),  
        'max_bins': trial.suggest_int('max_bins', 2, 255), 
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
        'verbose': 0,
        'random_state': rd_state,
    }
    model = HistGradientBoostingRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error') 
    return scores.mean()

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, n_jobs=-1)  

print("HGB Best hyperparameters: ", study.best_params)
print("HGB Best accuracy: ", study.best_value)

