from IPython.display import Image
Image("/kaggle/input/rmsle-png/RMSLE.png")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
fig, axes = plt.subplots(1, 1, figsize = (10, 5))
# Hist
sns.histplot(df_train['Calories'], bins = 30, kde = True, ax = axes, color = 'blue')
axes.set_title('Hist Calories (Target)');


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
fig, axes = plt.subplots(1, 1, figsize = (10, 5))
# Hist
sns.histplot(np.log1p(df_train['Calories']), bins = 30, kde = True, ax = axes, color = 'blue')
axes.set_title('Hist Calories (Log-Target)');


# Base
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from typing import Optional, Tuple, Union

#Statistics
from scipy.stats import skew
from scipy import stats
from scipy.stats import norm, cramervonmises, anderson, kstest, norm, cramervonmises, randint
from statsmodels.stats.diagnostic import lilliefors, normal_ad, het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera, durbin_watson
! pip install arch
from arch.unitroot import VarianceRatio
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
import featuretools as ft
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
#import cudf

#Feature engineering
from sklearn.feature_selection import mutual_info_regression

#Transformers and Pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

#Models ML (Linear and Tree)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from xgboost import XGBRegressor
from xgboost import plot_importance
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#Model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

#Stacking
from sklearn.ensemble import StackingRegressor


import sys
from importlib.metadata import version
packages = [
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('plotly', 'plotly'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('scikit-learn', 'scikit-learn'),
    ('optuna', 'optuna'),
    ('xgboost', 'xgboost'),
    ('catboost', 'catboost'),
    ('dateutil', 'python-dateutil'),
    ('shap', 'shap'),
    ('scipy', 'scipy'),
    ('IPython', 'IPython'),
    ('pickle', 'pickle')]

print(f"Python version: {sys.version.split()[0]}\n")

for import_name, package_name in packages:
    try:
        if package_name == 'pickle':
            print(f"{import_name} version: Part of Python {sys.version.split()[0]}")
        else:
            ver = version(package_name)
            print(f"{import_name} version: {ver}")
    except Exception as e:
        print(f"{import_name} version: Not found ({str(e)})")

print("\nStandard libraries (os, warnings, time): Part of Python", sys.version.split()[0])


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_train.head()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df_test.head()


df_orig = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")
df_orig = df_orig.rename(columns = {'User_ID': 'id', 'Gender': 'Sex'})
df_orig.head()


df_train.info(), df_test.info(), df_orig.info()


#df_train = pd.concat([df_train, df_orig], axis = 0, ignore_index = True)
#df_train.head()


df_train.describe(exclude = np.number).T


round(df_train.describe(exclude = 'object').T, 2).style.background_gradient(axis = 1, low = 0.3, high = 1.0)


plt.figure(figsize=(15, 5))
numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.drop(['id', 'Calories'])
df_numeric = df_train[['id'] + list(numeric_cols)]
colors = plt.cm.rainbow(np.linspace(0, 1, len(numeric_cols)))

for idx, col in enumerate(numeric_cols):
    plt.scatter(df_train['id'][(df_train['id'] < 0.75 * 1e7)], 
                df_train[col][(df_train['id']  < 0.75 * 1e7)], 
                color=colors[idx], 
                label=col,
                alpha=0.8, s = 0.5)

plt.xlabel('ID', fontsize=12)
plt.ylabel('Numerical values', fontsize=12)
plt.title('Feature dependencies on ID for train.csv', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))
numeric_cols = df_orig.select_dtypes(include=['int64', 'float64']).columns.drop(['id', 'Calories'])
df_numeric = df_orig[['id'] + list(numeric_cols)]
colors = plt.cm.rainbow(np.linspace(0, 1, len(numeric_cols)))

for idx, col in enumerate(numeric_cols):
    plt.scatter(df_orig['id'], 
                df_orig[col], 
                color=colors[idx], 
                label=col,
                alpha=0.8, s = 0.5)

plt.xlabel('ID', fontsize=12)
plt.ylabel('Numerical values', fontsize=12)
plt.title('Feature dependencies on ID for calories.csv', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 1, figsize = (15, 10))
# Hist
sns.histplot(df_train['Calories'], bins = 30, kde = True, ax = axes[0], color = 'blue')
axes[0].set_title('Hist Calories (Target)')
# BoxPlot
sns.boxplot(x = df_train['Calories'], ax = axes[1], color = 'blue')
axes[1].set_title('Boxplot Calories (Target)')


def nan_values(df):
    for i in df.columns:
        if df[i].isna().sum() > 0:
            print(f"For column - {i}, we have {df[i].isna().sum()} nan values")
        else:
            print(f"Our column {i} have zero nan values")
            print(f"Ideal!")


nan_values(df_train)
nan_values(df_test)


# Checking the normality of the target variable distribution
shapiro_test = stats.shapiro(df_train['Calories'])
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue:.3f}")
# If the p-value is < 0.05, the distribution is NOT normal.
# # We got a near-zero value, which tells us that the remnants of our target are distributed normally, therefore the final remnants of the model will also be distributed abnormally
# Let's try to prolog the target variable to improve the normal distribution


# Checking the normality of the log(target) variable distribution
shapiro_test = stats.shapiro(np.log(df_train['Calories']))
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue:.3f}")
# Also has a zero value, which means we are working with what we have


def z_metrics(df):
    results = []
    for feature in df.columns:
        mean = df[feature].mean()
        std = df[feature].std()
        df[f"{feature}_normal"] = (df[feature] - mean) / std
        # Calculating anomalies
        anomalies = df[np.abs(df[f"{feature}_normal"]) > 3]
        n_anomalies = len(anomalies)
        percentage = n_anomalies / len(df) * 100
        results.append({'Feature': feature,
                        'Number of anomalies': n_anomalies,
                        'Percentage of anomalies': round(percentage, 2)})
    report_df = pd.DataFrame(results, columns=['Feature', 'Number of anomalies', 'Percentage of anomalies'])
    return report_df

z_report = z_metrics(df_train.drop(columns=['id', 'Sex']))
z_report
# The test results show that the largest percentage of abnormal values will be found in the Body_Temp predictor, let's plot it.


plt.figure(figsize=(13, 7))
plt.scatter(df_train['id'], 
           df_train['Body_Temp'],
           color = 'red',
           label="Body_Temp",
           s = 0.5,
           alpha=0.6)
plt.title("Body_Temp graphs for train.csv")
plt.xlabel("id")
plt.ylabel("Body_Temp");


# Categorical Variable Analysis (Sex)
gender_dist = df_train['Sex'].value_counts(normalize=True)
fig = px.pie(gender_dist, 
       names=gender_dist.index, 
       title='Gender Distribution',
       color_discrete_sequence=px.colors.qualitative.Pastel)
init_notebook_mode(connected=True)
py.iplot(fig)


# 4. Numerical Features Distribution vs Sex
num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data = df_train, x = col, kde = True, hue = 'Sex', color='skyblue')
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()


# Correlation Analysis
corr_matrix = df_train[num_cols + ['Calories']].corr()
fig = px.imshow(round(corr_matrix, 2),
                text_auto=True,
                color_continuous_scale='rainbow',
                title='Feature Correlation Matrix')
fig.update_layout(width=800, height=800)
init_notebook_mode(connected=True)
py.iplot(fig)


# Target vs Features Relationships (Sampled 1%)
fig = px.scatter_matrix(df_train,
                        dimensions=['Age', 'Duration', 'Heart_Rate', 'Calories'],
                        color='Sex',
                        title='Pairwise Relationships',
                        opacity=0.5)
fig.update_layout(width = 1300, height = 800)
fig.update_traces(diagonal_visible = False)
init_notebook_mode(connected=True)
py.iplot(fig)


# 3D Visualization
fig = px.scatter_3d(df_train.sample(frac=0.2),
                    x='Age',
                    y='Duration',
                    z='Heart_Rate',
                    color='Calories',
                    title='3D Relationship: Age vs Duration vs Heart Rate',
                    color_continuous_scale='Viridis')
fig.update_layout(width = 500, height = 500)
init_notebook_mode(connected=True)
py.iplot(fig)


# Outlier Detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train[num_cols], orient='h', palette='Set2')
plt.title('Numerical Features Boxplot Comparison')
plt.xlabel('Values')
plt.show()


# PairPlot
plt.figure(figsize=(12, 6))
sns.pairplot(df_train.drop(columns = ['id', 'Sex']), corner = True, diag_kind = "kde", hue = None)
plt.xlabel('Values')
plt.show();


#1. Initializing EntitySet
es = ft.EntitySet(id='workout_data')

#2. Adding an entity
try:
    es = es.add_dataframe(
        dataframe_name='workouts',
        dataframe=df_train.drop(columns=['Calories']), 
        index='id'
    )
    print("âœ… The 'workouts' entity has been added.")
except Exception as e:
    print(f"â�Œ Error: {e}")
    raise

#3. Feature generation
try:
    trans_primitives = ['multiply_numeric', 'divide_numeric', 'subtract_numeric']
    #groupby_trans_primitives = ['sum', 'mean', 'max']

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='workouts',
        trans_primitives = trans_primitives,
        #groupby_trans_primitives = groupby_trans_primitives,
        max_depth=2,
        verbose=True
    )
    print("âœ… The signs are generated.")
except Exception as e:
    print(f"â�Œ Error in DFS: {e}")
    raise

# 3. Automatic selection of features by patterns
enhanced_df = feature_matrix
# ---------------------------------------------------------------------- I can't create aggregated attributes in any way


# Thank you for genious feature engineering: https://www.kaggle.com/code/onurkoc83/catboost-xgboost-with-new-features
df_train['Sex'] = df_train.Sex.map({'male':1,'female':0})
df_test['Sex'] =  df_test.Sex.map({'male':1,'female':0})

df_train.drop(columns = ['id'], inplace=True)
df_test.drop(columns = ['id'], inplace=True)

cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

df_train = df_train.drop_duplicates(subset = df_train.columns, keep = 'first').reset_index(drop = True)
df_train = df_train.groupby(by = cols)['Calories'].min().reset_index()


# Create a space of Sex intersections with non-linearly dependent features - 'Duration', 'Heart_Rate', 'Body_Temp'
df_train['Sex_Reversed'] = 1 - df_train['Sex']
df_test['Sex_Reversed']  =  1 - df_test['Sex']

def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

list_1 = ['Duration', 'Heart_Rate', 'Body_Temp']
list_2 = ['Sex', 'Sex_Reversed']

df_train = add_feature_cross_terms(df_train, list_1, list_2)
df_test = add_feature_cross_terms(df_test, list_1, list_2)
df_train.drop(columns=['Sex_Reversed'], inplace=True)
df_test.drop(columns=['Sex_Reversed'], inplace=True)


# Let's create aggregated features (the base for boosting models)
# In our case, i can be equal to only 1
def add_categorical_aggregations(df):
    categorical_cols = ['Sex']
    numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']

    for i in range(1, len(categorical_cols) + 1):
        if i == 1:
            for cat_col in categorical_cols:
                aggs = df.groupby(cat_col).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
                aggs.columns = [f"{cat_col}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                df = df.merge(aggs, on=cat_col, how='left')
        elif i == 2:
            for j in range(len(categorical_cols)):
                for k in range(j + 1, len(categorical_cols)):
                    cat_col1 = categorical_cols[j]
                    cat_col2 = categorical_cols[k]
                    aggs = df.groupby([cat_col1, cat_col2]).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
                    aggs.columns = [f"{cat_col1}_{cat_col2}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                    df = df.merge(aggs, on = [cat_col1, cat_col2], how = 'left')
        elif i == 3:
            aggs = df.groupby(categorical_cols).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
            aggs.columns = [f"all_cat_{num_col}_{agg}" for num_col, agg in aggs.columns]
            df = df.merge(aggs, on = categorical_cols, how = 'left')
    return df

df_train = add_categorical_aggregations(df_train)
df_test = add_categorical_aggregations(df_test)


df_train.info()


df_train.drop(columns=['Sex'],inplace=True)
df_test.drop(columns=['Sex'],inplace=True)


# Let's check if the column order is the same
columns_match = df_train.columns.equals(df_test.columns.append(pd.Index(['Calories'])))
print(f"Is the column order the same: {columns_match}")

# If the column order is not the same, let's fix
if not columns_match:
    # Let's drop the Calories column
    df_train_without_calories = df_train.drop(columns=['Calories'])
    # Let's take the order of the df_test columns and apply it to df_train
    common_columns = [col for col in df_test.columns if col in df_train_without_calories.columns]
    # Let's recreate the df_train and df_test dataframes in the new order
    df_train_without_calories = df_train_without_calories[common_columns]
    df_test = df_test[common_columns]
    # Let's add calories again
    df_train = pd.concat([df_train_without_calories, df_train['Calories']], axis = 1)
    print("The column order has been corrected")

# Let's check again
df_train_without_calories = df_train.drop(columns=['Calories'])
columns_match_after_drop = df_train_without_calories.columns.equals(df_test.columns)
print(f"Is the column order the same after calories fall: {columns_match_after_drop}")


df_train.info()


df_test.info()


# We separate the data for modeling and scale it
y_full = df_train['Calories'].reset_index(drop = True)
X_full = df_train.drop(columns = ['Calories']).reset_index(drop = True)
X_test = df_test

# We check the importance of features on a model with standard hyperparameter values
default_param = {'objective': 'reg:squaredlogerror',
                 'eval_metric': 'rmsle',
                 'tree_method': 'gpu_hist',
                 'device': 'cuda',
                 'seed': 42}

trained_model_XGB = XGBRegressor(**default_param).fit(X_full, y_full)

# 1. Correct definition of RMSLE (getting rid of negative prediction values)
def root_mean_squared_log_error(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)  # Ğ“Ğ°Ñ€Ğ°Ğ½Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ½ĞµĞ¾Ñ‚Ñ€Ğ¸Ñ†Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ�
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# 2. Calculating permutation importance based on the direction of the metric
results = permutation_importance(trained_model_XGB,
                                 X_full,
                                 y_full,
                                 scoring = make_scorer(root_mean_squared_log_error, greater_is_better = False),
                                 n_repeats = 10,
                                 random_state = 42)
# 3. Handling negative importance values
importance = pd.DataFrame({'feature': X_full.columns,
                           'importance': results.importances_mean})
# Absolute values for visualization
importance['abs_importance'] = np.abs(importance['importance'])
# Sorting by absolute importance
importance = importance.sort_values('abs_importance', ascending = True)

# 4. Visualization of the top N features
top_n = 25
plt.figure(figsize=(12, 8))
bars = plt.barh(importance['feature'][::-1].head(top_n),
                importance['abs_importance'][::-1].head(top_n),
                color = 'cyan')

# 5. Add annotations with real values
for bar in bars:
    width = bar.get_width()
    plt.text(width * 1.02, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va = 'center')

plt.xlabel('Absolute Importance')
plt.title('Permutation Importance XGB (Top 25 Features)')
plt.gca().invert_yaxis()
plt.grid(axis = 'x', alpha = 0.3)
plt.tight_layout()
plt.show()


feature_names = X_full.columns
importances = trained_model_XGB.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y = sorted_features[-25:], width=sorted_importances[-25:], color='cyan',edgecolor='black')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("XGB Feature Importance", fontsize=14, pad = 20)
plt.grid(axis='x', linestyle='--', alpha=0.7)
for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)
plt.tight_layout()
plt.show()


predictors = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


vif_data = pd.DataFrame()
vif_data["feature"] = X_full[predictors].columns
vif_data["VIF"] = [variance_inflation_factor(X_full[predictors].values, i) for i in range(X_full[predictors].shape[1])]
print("\nMulticollinearity check (For Tree and Boosting Models VIF > 1 000 000 is a problem (I found this statement in one of the articles on multicollinearity.) AND for Linear Models VIF > 10 (5) have a strong influnce on weight coefficients:")
print(vif_data.sort_values("VIF", ascending=False))


# Best RMSLE = 0.6298 calories - Optuna
# The heteroscedasticity in the data is too high, so we use logorithmization (you can also use the Box-Cox transform)
def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int("n_estimators", 500, 5000, step = 100),
        'max_depth': trial.suggest_int("max_depth", 6, 15, step = 2),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.9, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        'subsample': trial.suggest_float("subsample", 0.5, 0.95),
        'gamma': trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 0.95),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        'objective': 'reg:squaredlogerror',
        'eval_metric': 'rmsle',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'seed': 42}

    model = XGBRegressor(**xgb_params)
    
    # Cross-validation configuration
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full)):
        print(f"\n Fold XGB (Optim) {fold + 1}")
        # Data separation
        X_train = X_full[predictors].iloc[idx_train].copy()
        X_valid = X_full[predictors].iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())

        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds = 500,
                  verbose = 100)

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid])))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold XGB (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_full), np.expm1(y_pred_val)))
    print(f"\nğŸ�¯ Overall CV XGB (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


#sampler = TPESampler(seed = 42)
#study_1 = optuna.create_study(direction = "minimize", sampler=sampler)
#study_1.optimize(objective, n_trials = 25)


# Best RMSLE on CV-Kfold = 0.05982 calories
# Final TEST prediction RMSLE = 0.05694 calories (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int("n_estimators", 2000, 5000, step = 100),
        'max_depth': trial.suggest_int("max_depth", 7, 15, step = 1),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.5, log = True),
        'subsample': trial.suggest_float("subsample", 0.5, 0.99),
        'gamma': trial.suggest_float("gamma", 1e-3, 1, log  =True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 0.95),
        'eval_metric': 'rmse',
        'enable_categorical': True,
        'random_state': 42,
        'early_stopping_rounds': 100,
        'tree_method': 'gpu_hist'}

    model = XGBRegressor(**xgb_params)
    
    # Cross-validation configuration
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full)):
        print(f"\n Fold XGB (Optim) {fold + 1}")
        # Data separation
        X_train = X_full.iloc[idx_train].copy()
        X_valid = X_full.iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())

        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  verbose = 500)

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid])))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold XGB (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val)))
    print(f"\nğŸ�¯ Overall CV XGB (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


#sampler = TPESampler(seed = 42)
#study_2 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_2.optimize(objective, n_trials = 25)


xgb_params_1 = {'max_depth': 9,
                'colsample_bytree': 0.70,
                'subsample': 0.90,
                'n_estimators': 3000,
                'learning_rate': 0.010,
                'gamma': 0.010,
                'max_delta_step': 2}

xgb_params_2 = {'eval_metric': 'rmse',
                'enable_categorical': True,
                'random_state': 42,
                #'early_stopping_rounds': 500,
                'tree_method': 'gpu_hist'}

model_1 = XGBRegressor(**xgb_params_1, **xgb_params_2)
model_1.fit(X_full, np.log1p(y_full))
y_xgb_train_pred = model_1.predict(X_full)
rmsle_xgb_train_pred = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_xgb_train_pred)))
print(f"Root mean squared log error on the full dataset = {round(rmsle_xgb_train_pred, 5)} calories")

xgb_params_2 = {'eval_metric': 'rmse',
                'enable_categorical': True,
                'random_state': 42,
                'early_stopping_rounds': 500,
                'tree_method': 'gpu_hist'}
model_1 = XGBRegressor(**xgb_params_1, **xgb_params_2)


# Setting the initial parameters for validation
cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
rmsle_scores_valid_xgb = []
rmsle_scores_train_xgb = []
y_pred_val_xgb = np.zeros(len(X_full))
y_pred_train_xgb = np.zeros(len(X_full))
y_pred_test_xgb = np.zeros(len(X_test))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full)):
    print(f"\n Fold XGB (Final) {fold + 1}")
    # Separating the training and radiation data from the source dataset
    X_train = X_full.iloc[idx_train].copy()
    X_valid = X_full.iloc[idx_valid].copy()
    X_test = X_test.copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    model_1.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              verbose = 100)
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_xgb[idx_valid] = model_1.predict(X_valid)
    y_pred_train_xgb[idx_train] = model_1.predict(X_train)
    y_pred_test_xgb += model_1.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_xgb[idx_valid])))
    fold_rmsle_train = np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_xgb[idx_train])))
    rmsle_scores_valid_xgb.append(fold_rmsle_valid)
    rmsle_scores_train_xgb.append(fold_rmsle_train)
    print(f"Fold XGB (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid:.5f}")
    print(f"Fold XGB (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train:.5f}")

# It is much more reliable to calculate the error on the already calculated data, rather than averaging it over 5 fouls.
overall_rmsle_valid_xgb = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val_xgb)))
overall_rmsle_train_xgb = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_train_xgb)))
print(f"\nğŸ�¯ Overall CV XGB (Final) RMSLE on valid data: {overall_rmsle_valid_xgb:.5f}")
print(f"\nğŸ�¯ Overall CV XGB (Final) RMSLE on train data: {overall_rmsle_train_xgb:.5f}")
# Since the predicted data on the test set was summed up every fold (5), therefore we divide our sum by the number of folds
y_pred_test_xgb /= 5


# Create figure
fig = go.Figure()

# Add fold validation scores
fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_train_xgb,
                         mode = 'lines+markers', name = 'Train RMSLE_xgb per Fold',
                         line=dict(color = 'blue', dash = 'dash'), marker = dict(size = 8)))

fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_valid_xgb,
                         mode = 'lines+markers', name = 'Valid RMSLE_xgb per Fold',
                         line = dict(color = 'red', dash = 'dash'), marker = dict(size = 8)))

# Add overall horizontal lines
fig.add_shape(type="line", 
              x0 = 0.5, y0 = overall_rmsle_train_xgb, 
              x1 = 5.5, y1 = overall_rmsle_train_xgb,
              line=dict(color="blue", width = 1), name='Overall Train RMSLE_xgb')

fig.add_shape(type="line",
              x0=0.5, y0=overall_rmsle_valid_xgb,
              x1=5.5, y1=overall_rmsle_valid_xgb,
              line=dict(color="red", width = 1), name='Overall Valid RMSLE_xgb')

# Add annotations for overall scores
fig.add_annotation(x=5.3, y=0.050,
                   text=f'Train: {overall_rmsle_train_xgb:.5f}',
                   showarrow=False, font=dict(color='blue'))
    
fig.add_annotation(x = 5.3, y = 0.061,
                   text=f'Valid: {overall_rmsle_valid_xgb:.5f}',
                   showarrow=False, font=dict(color='red'))

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Cross-Validation RMSLE_xgb Scores</b>',
        font=dict(size=24, family='Arial'),
        x = 0.5
    ),
    xaxis=dict(
        title='Fold Number',
        tickmode = 'array',
        tickvals=list(range(1, 6)),
        gridcolor='lightgrey',
        title_font=dict(size=16)
    ),
    yaxis=dict(
        title='RMSLE',
        gridcolor='lightgrey',
        title_font=dict(size=16),
        range=[0.045, 0.062]
    ),
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    ),
    margin=dict(l=80, r=80, t=100, b=80),
    height=600,
    width=1300
)

init_notebook_mode(connected=True)
py.iplot(fig)


results_xgb = model_1.evals_result()
plt.figure(figsize=(10,5))
plt.plot(results_xgb["validation_0"]["rmse"], label="Training loss")
#plt.plot(results_xgb["validation_1"]["rmse"], label="Validation loss")
plt.axvline(1500, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.title('Graphics of Loss function (RMSLE) for XGBoostModels')
plt.legend();


# Best RMSLE = 0.5998 calories
# Final TEST prediction RMSLE = 0.05793 calories (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
def objective(trial):
    lgbm_params = {'n_estimators': trial.suggest_int("n_estimators", 2000, 4000, step = 100),
                   'num_leaves': trial.suggest_int("num_leaves", 32, 128, step = 2),
                   'max_depth': trial.suggest_int("max_depth", 8, 15),
                   'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                   'subsample': trial.suggest_float("subsample", 0.6, 0.99),
                   'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.99),
                   'device': 'gpu',
                   'random_state': 42,
                   'verbose': -1}

    model = LGBMRegressor(**lgbm_params)
    
    # Cross-validation configuration
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full)):
        print(f"\n Fold LGBM (Optim) {fold + 1}")
        # Data separation
        X_train = X_full.iloc[idx_train].copy()
        X_valid = X_full.iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())

        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(500, verbose = 1000)])

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid])))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold LGBM (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val)))
    print(f"\nğŸ�¯ Overall CV LGBM (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


# Best RMSLE = 0.5998 calories
# Final TEST prediction RMSLE = 0.05793 calories (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
import lightgbm as lgb
def objective(trial):
    lgbm_params = {'n_estimators': trial.suggest_int("n_estimators", 2000, 4000, step = 100),
                   'num_leaves': trial.suggest_int("num_leaves", 32, 128, step = 2),
                   'max_depth': trial.suggest_int("max_depth", 8, 15),
                   'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                   'subsample': trial.suggest_float("subsample", 0.6, 0.99),
                   'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.99),
                   'device': 'gpu',
                   'random_state': 42,
                   'verbose': -1}
    
    # Cross-validation configuration
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full)):
        print(f"\n Fold LGBM (Optim) {fold + 1}")
        # Data separation
        X_train = X_full.iloc[idx_train].copy()
        X_valid = X_full.iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_valid, label=y_valid)
        
        model = lgb.train(lgbm_params,
                          dtrain,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(500), lgb.log_evaluation(500)])

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid])))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold LGBM (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val)))
    print(f"\nğŸ�¯ Overall CV LGBM (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


#sampler = TPESampler(seed = 42)
#study_3 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_3.optimize(objective, n_trials = 25)


lgbm_params = {'n_estimators': 4000, 
               'num_leaves': 94, 
               'max_depth': 10, 
               'learning_rate': 0.01, 
               'subsample': 0.87, 
               'colsample_bytree': 0.60,
               'device': 'gpu',
               'random_state': 42,
               'verbose': -1}
#model_2 = LGBMRegressor(**lgbm_params)
#model_2.fit(X_full, np.log1p(y_full))
#y_lgbm_train_pred = model_2.predict(X_full)
#rmsle_lgbm_train_pred = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_lgbm_train_pred)))
#print(f"Root mean squared log error on the full dataset = {round(rmsle_lgbm_train_pred, 5)} calories")


# Setting the initial parameters for validation
cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
rmsle_scores_valid_lgbm = []
rmsle_scores_train_lgbm = []
y_pred_val_lgbm = np.zeros(len(X_full))
y_pred_train_lgbm = np.zeros(len(X_full))
y_pred_test_lgbm = np.zeros(len(X_test))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full)):
    print(f"\n Fold LGBM (Final) {fold + 1}")
    # Separating the training and radiation data from the source dataset
    X_train = X_full.iloc[idx_train].copy()
    X_valid = X_full.iloc[idx_valid].copy()
    X_test  = X_test.copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    #model_2.fit(X_train, y_train,
                #eval_set=[(X_valid, y_valid)],
                #eval_metric='rmse',
                #callbacks=[lgb.early_stopping(500, verbose = 100)])
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_valid, label=y_valid)
    model_2 = lgb.train(lgbm_params, 
                        dtrain,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(500), lgb.log_evaluation(500)])
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_lgbm[idx_valid] = model_2.predict(X_valid)
    y_pred_train_lgbm[idx_train] = model_2.predict(X_train)
    y_pred_test_lgbm += model_2.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid_lgbm = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_lgbm[idx_valid])))
    fold_rmsle_train_lgbm = np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_lgbm[idx_train])))
    rmsle_scores_valid_lgbm.append(fold_rmsle_valid_lgbm)
    rmsle_scores_train_lgbm.append(fold_rmsle_train_lgbm)
    print(f"Fold LGBM (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid_lgbm:.5f}")
    print(f"Fold LGBM (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train_lgbm:.5f}")
    
# It is much more reliable to calculate the error on the already calculated data, rather than averaging it over 5 fouls.
overall_rmsle_valid_lgbm = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val_lgbm)))
overall_rmsle_train_lgbm = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_train_lgbm)))
print(f"\nğŸ�¯ Overall CV LGBM (Final) RMSLE on valid data: {overall_rmsle_valid_lgbm:.5f}")
print(f"\nğŸ�¯ Overall CV LGBM (Final) RMSLE on train data: {overall_rmsle_train_lgbm:.5f}")
# Since the predicted data on the test set was summed up every fold (5), therefore we divide our sum by the number of folds
y_pred_test_lgbm /= 5


# Create figure
fig = go.Figure()

# Add fold validation scores
fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_train_lgbm,
                         mode = 'lines+markers', name = 'Train RMSLE_lgbm per Fold',
                         line=dict(color = 'blue', dash = 'dash'), marker = dict(size = 8)))

fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_valid_lgbm,
                         mode = 'lines+markers', name = 'Valid RMSLE_lgbm per Fold',
                         line = dict(color = 'red', dash = 'dash'), marker = dict(size = 8)))

# Add overall horizontal lines
fig.add_shape(type="line", 
              x0 = 0.5, y0 = overall_rmsle_train_lgbm, 
              x1 = 5.5, y1 = overall_rmsle_train_lgbm,
              line=dict(color="blue", width = 1), name='Overall Train RMSLE_lgbm')

fig.add_shape(type="line",
              x0=0.5, y0=overall_rmsle_valid_lgbm,
              x1=5.5, y1=overall_rmsle_valid_lgbm,
              line=dict(color="red", width = 1), name='Overall Valid RMSLE_lgbm')

# Add annotations for overall scores
fig.add_annotation(x=5.3, y=0.050,
                   text=f'Train: {overall_rmsle_train_lgbm:.5f}',
                   showarrow=False, font=dict(color='blue'))
    
fig.add_annotation(x = 5.3, y = 0.061,
                   text=f'Valid: {overall_rmsle_valid_lgbm:.5f}',
                   showarrow=False, font=dict(color='red'))

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Cross-Validation RMSLE_LightGBM Scores</b>',
        font=dict(size=24, family='Arial'),
        x = 0.5
    ),
    xaxis=dict(
        title='Fold Number',
        tickmode = 'array',
        tickvals=list(range(1, 6)),
        gridcolor='lightgrey',
        title_font=dict(size=16)
    ),
    yaxis=dict(
        title='RMSLE',
        gridcolor='lightgrey',
        title_font=dict(size=16),
        range=[0.045, 0.062]
    ),
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    ),
    margin=dict(l=80, r=80, t=100, b=80),
    height=600,
    width=1300
)

init_notebook_mode(connected=True)
py.iplot(fig)


'''
results_lgbm = model_2.evals_result_
plt.figure(figsize=(10,5))
plt.plot(results_lgbm['training']['l2'], label="Training loss")
plt.plot(results_lgbm['valid_1']['l2'], label="Validation loss")
plt.axvline(200, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.title('Graphics of Loss function (RMSLE) for LightGBMModels')
plt.legend();
'''


# Feature engineering (+ some specific features with age and duration)
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Thank you for code: https://www.kaggle.com/code/onurkoc83/catboost-xgboost-with-new-features
df_train['Sex'] = df_train.Sex.map({'male':1,'female':0})
df_test['Sex'] =  df_test.Sex.map({'male':1,'female':0})
df_train.drop(columns = ['id'], inplace=True)
df_test.drop(columns = ['id'], inplace=True)
cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
df_train = df_train.drop_duplicates(subset = df_train.columns, keep = 'first').reset_index(drop = True)
df_train = df_train.groupby(by = cols)['Calories'].min().reset_index()

print(df_train.shape)

# Find the unique Duration values
unique_durations_train = df_train['Duration'].unique()
unique_durations_test  =  df_test['Duration'].unique()

# Creating new features for each Duration value for df_train
for duration in unique_durations_train:
    # Creating new column names
    heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
    body_temp_col  = f'Body_Temp_Duration_{int(duration)}'
    # If the Duration value is equal to a certain value, get the Heart_Rate and Body_Temp values, if not, make it equal to 0.
    df_train[heart_rate_col] = np.where(df_train['Duration'] == duration, df_train['Heart_Rate'], 0)
    df_train[body_temp_col] =  np.where(df_train['Duration'] == duration,  df_train['Body_Temp'], 0)

# Creating new features for each Duration value for df_test
for duration in unique_durations_test:
    # Creating new column names
    heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
    body_temp_col  = f'Body_Temp_Duration_{int(duration)}'
    # If the Duration value is equal to a certain value, get the Heart_Rate and Body_Temp values, if not, do 0
    df_test[heart_rate_col] = np.where(df_test['Duration'] == duration, df_test['Heart_Rate'], 0)
    df_test[body_temp_col] =  np.where(df_test['Duration'] == duration,  df_test['Body_Temp'], 0)

print(df_train.shape)

# We'll do the same for Age
unique_ages_df_train = df_train['Age'].unique()
unique_ages_df_test = df_test['Age'].unique()

for age in unique_ages_df_train:
    heart_rate_col = f'Heart_Rate_Age_{int(age)}'
    body_temp_col = f'Body_Temp_Age_{int(age)}'
    df_train[heart_rate_col] = np.where(df_train['Age'] == age, df_train['Heart_Rate'], 0)
    df_train[body_temp_col] = np.where(df_train['Age'] == age, df_train['Body_Temp'], 0)

for age in unique_ages_df_test:
    heart_rate_col = f'Heart_Rate_Age_{int(age)}'
    body_temp_col = f'Body_Temp_Age_{int(age)}'
    df_test[heart_rate_col] = np.where(df_test['Age'] == age, df_test['Heart_Rate'], 0)
    df_test[body_temp_col] = np.where(df_test['Age'] == age, df_test['Body_Temp'], 0)

print(df_train.shape)

def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new
list_1 = ['Duration', 'Heart_Rate', 'Body_Temp']
list_2 = ['Sex']
df_train = add_feature_cross_terms(df_train, list_1, list_2)
df_test = add_feature_cross_terms(df_test, list_1, list_2)

print(df_train.shape)

df_train['Sex_Reversed'] = 1 - df_train['Sex']
df_test['Sex_Reversed']  =  1 - df_test['Sex']
def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new
list_1 = ['Duration', 'Heart_Rate', 'Body_Temp']
list_2 = ['Sex', 'Sex_Reversed']
df_train = add_feature_cross_terms(df_train, list_1, list_2)
df_test = add_feature_cross_terms(df_test, list_1, list_2)
df_train.drop(columns=['Sex_Reversed'], inplace=True)
df_test.drop(columns=['Sex_Reversed'], inplace=True)

print(df_train.shape)

def add_categorical_aggregations(df):
    categorical_cols = ['Sex']
    numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']

    for i in range(1, len(categorical_cols) + 1):
        if i == 1:
            for cat_col in categorical_cols:
                aggs = df.groupby(cat_col).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
                aggs.columns = [f"{cat_col}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                df = df.merge(aggs, on=cat_col, how='left')
        elif i == 2:
            for j in range(len(categorical_cols)):
                for k in range(j + 1, len(categorical_cols)):
                    cat_col1 = categorical_cols[j]
                    cat_col2 = categorical_cols[k]
                    aggs = df.groupby([cat_col1, cat_col2]).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
                    aggs.columns = [f"{cat_col1}_{cat_col2}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                    df = df.merge(aggs, on = [cat_col1, cat_col2], how = 'left')
        elif i == 3:
            aggs = df.groupby(categorical_cols).agg({num_col: ['min', 'max'] for num_col in numerical_cols})
            aggs.columns = [f"all_cat_{num_col}_{agg}" for num_col, agg in aggs.columns]
            df = df.merge(aggs, on = categorical_cols, how = 'left')
    return df

df_train = add_categorical_aggregations(df_train)
df_test = add_categorical_aggregations(df_test)

print(df_train.shape)

cat_features = ['Sex']
for col in cat_features:
    df_train[col] = df_train[col].astype('int32').astype('category')
    df_test[col]  = df_test[col].astype('int32').astype('category')

# Let's check if the column order is the same
columns_match = df_train.columns.equals(df_test.columns.append(pd.Index(['Calories'])))
print(f"Is the column order the same: {columns_match}")
# If the column order is not the same, let's fix
if not columns_match:
    # Let's drop the Calories column
    df_train_without_calories = df_train.drop(columns=['Calories'])
    # Let's take the order of the df_test columns and apply it to df_train
    common_columns = [col for col in df_test.columns if col in df_train_without_calories.columns]
    # Let's recreate the df_train and df_test dataframes in the new order
    df_train_without_calories = df_train_without_calories[common_columns]
    df_test = df_test[common_columns]
    # Let's add calories again
    df_train = pd.concat([df_train_without_calories, df_train['Calories']], axis = 1)
    print("The column order has been corrected")
# Let's check again
df_train_without_calories = df_train.drop(columns=['Calories'])
columns_match_after_drop = df_train_without_calories.columns.equals(df_test.columns)
print(f"Is the column order the same after calories fall: {columns_match_after_drop}")

y_full = df_train['Calories'].reset_index(drop = True)
X_full = df_train.drop(columns = ['Calories']).reset_index(drop = True)
X_test = df_test
print(X_full.shape)


# Best RMSLE = 0.05966 calories
# Final TEST prediction RMSLE = 0.05694 (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
def objective(trial):
    class RMSLE(object):
        def is_max_optimal(self):
            return False
        def evaluate(self, approxes, target, weight):
            assert len(approxes) == 1
            approx = approxes[0]
            # Converting predictions and goals from logarithmic form
            approx_exp = np.expm1(approx)
            target_exp = np.expm1(target)
            # Calculating the RMSLE
            error = np.sqrt(np.mean(np.square(np.log1p(approx_exp) - np.log1p(target_exp))))
            return error, 1  # We return the error and weight
        def get_final_error(self, error, weight):
            return error
    
    cat_params = {'iterations': trial.suggest_int("n_estimators", 1500, 5000, step = 100),
        'depth': trial.suggest_int("depth", 8, 15),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.9, log = True),
        'l2_leaf_reg': trial.suggest_int("l2_leaf_reg", 1, 5, step = 1),
        'loss_function': 'RMSE',
        'random_seed': 42,     
        'eval_metric': RMSLE(),
        'early_stopping_rounds': 500,
        'cat_features': cat_features,
        'verbose': 100,
        'task_type': 'GPU',
        'random_seed': 42}

    model = CatBoostRegressor(**cat_params)
    
    # Cross-validation configuration
    cv = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    n_bins = 10
    discretizer = KBinsDiscretizer(n_bins = n_bins, encode='ordinal', strategy='quantile')
    duration_binned = discretizer.fit_transform(df_train[['Duration']]).astype(int).flatten()

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full, duration_binned)):
        print(f"\n Fold CatBoost (Optim) {fold + 1}")
        # Data separation
        X_train = X_full.iloc[idx_train].copy()
        X_valid = X_full.iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())

        model.fit(X_train, y_train,
                  eval_set=(X_valid, y_valid),
                  use_best_model=True,
                  verbose=1000)

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = root_mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid]))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold CatBoost (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = root_mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val))
    print(f"\nğŸ�¯ Overall CV CatBoost (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


#sampler = TPESampler(seed = 42)
#study_4 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_4.optimize(objective, n_trials = 25)


class RMSLE(object):
        def is_max_optimal(self):
            return False
        def evaluate(self, approxes, target, weight):
            assert len(approxes) == 1
            approx = approxes[0]
            # Converting predictions and goals from logarithmic form
            approx_exp = np.expm1(approx)
            target_exp = np.expm1(target)
            # Calculating the RMSLE
            error = np.sqrt(np.mean(np.square(np.log1p(approx_exp) - np.log1p(target_exp))))
            return error, 1  # We return the error and weight
        def get_final_error(self, error, weight):
            return error
            
cat_params_1 = {'iterations': 3000,
                'learning_rate': 0.02,
                'depth': 12,
                'l2_leaf_reg': 3}

cat_params_2 = {'eval_metric': 'RMSE',
                'early_stopping_rounds': 500,
                'cat_features': cat_features,
                'verbose': False,
                'task_type': 'GPU',
                'random_seed': 42}

model_3 = CatBoostRegressor(**cat_params_1, **cat_params_2)
model_3.fit(X_full, np.log1p(y_full))
y_cat_train_pred = model_3.predict(X_full)
rmsle_cat_train_pred = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_cat_train_pred)))
print(f"Root mean squared log error on the full dataset = {round(rmsle_cat_train_pred, 5)} calories")


# Setting the initial parameters for validation
cv = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

rmsle_scores_valid_cat = []
rmsle_scores_train_cat = []
y_pred_val_cat = np.zeros(len(X_full))
y_pred_train_cat = np.zeros(len(X_full))
y_pred_test_cat = np.zeros(len(df_test))

n_bins = 10
discretizer = KBinsDiscretizer(n_bins = n_bins, encode='ordinal', strategy='quantile')
duration_binned = discretizer.fit_transform(df_train[['Duration']]).astype(int).flatten()

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full, duration_binned)):
    print(f"\n Fold CatBoost (Final) {fold + 1}")
    # Separating the training and radiation data from the source dataset
    X_train = X_full.iloc[idx_train].copy()
    X_valid = X_full.iloc[idx_valid].copy()
    X_test  = df_test.copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    model_3.fit(X_train, y_train,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
                verbose=1000)
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_cat[idx_valid] = model_3.predict(X_valid)
    y_pred_train_cat[idx_train] = model_3.predict(X_train)
    y_pred_test_cat += model_3.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid_cat = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_cat[idx_valid])))
    fold_rmsle_train_cat = np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_cat[idx_train])))
    rmsle_scores_valid_cat.append(fold_rmsle_valid_cat)
    rmsle_scores_train_cat.append(fold_rmsle_train_cat)
    print(f"Fold CatBoost (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid_cat:.5f}")
    print(f"Fold CatBoost (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train_cat:.5f}")

# It is much more reliable to calculate the error on the already calculated data, rather than averaging it over 5 fouls.
overall_rmsle_valid_cat = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val_cat)))
overall_rmsle_train_cat = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_train_cat)))
print(f"\nğŸ�¯ Overall CV CatBoost (Final) RMSLE on valid data: {overall_rmsle_valid_cat:.5f}")
print(f"\nğŸ�¯ Overall CV CatBoost (Final) RMSLE on train data: {overall_rmsle_train_cat:.5f}")
# Since the predicted data on the test set was summed up every fold (5), therefore we divide our sum by the number of folds
y_pred_test_cat /= 5


# Create figure
fig = go.Figure()

# Add fold validation scores
fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_train_cat,
                         mode = 'lines+markers', name = 'Train RMSLE_cat per Fold',
                         line=dict(color = 'blue', dash = 'dash'), marker = dict(size = 8)))

fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_valid_cat,
                         mode = 'lines+markers', name = 'Valid RMSLE_cat per Fold',
                         line = dict(color = 'red', dash = 'dash'), marker = dict(size = 8)))

# Add overall horizontal lines
fig.add_shape(type="line", 
              x0 = 0.5, y0 = overall_rmsle_train_cat,
              x1 = 5.5, y1 = overall_rmsle_train_cat,
              line=dict(color="blue", width = 1), name='Overall Train RMSLE_cat')

fig.add_shape(type="line",
              x0=0.5, y0=overall_rmsle_valid_cat,
              x1=5.5, y1=overall_rmsle_valid_cat,
              line=dict(color="red", width = 1), name='Overall Valid RMSLE_cat')

# Add annotations for overall scores
fig.add_annotation(x=5.3, y=0.050,
                   text=f'Train: {overall_rmsle_train_cat:.5f}',
                   showarrow=False, font=dict(color='blue'))
    
fig.add_annotation(x = 5.3, y = 0.061,
                   text=f'Valid: {overall_rmsle_valid_cat:.5f}',
                   showarrow=False, font=dict(color='red'))

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Cross-Validation RMSLE_CatBoost Scores</b>',
        font=dict(size=24, family='Arial'),
        x = 0.5
    ),
    xaxis=dict(
        title='Fold Number',
        tickmode = 'array',
        tickvals=list(range(1, 6)),
        gridcolor='lightgrey',
        title_font=dict(size=16)
    ),
    yaxis=dict(
        title='RMSLE',
        gridcolor='lightgrey',
        title_font=dict(size=16),
        range=[0.045, 0.062]
    ),
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    ),
    margin=dict(l=80, r=80, t=100, b=80),
    height=600,
    width=1300
)

init_notebook_mode(connected=True)
py.iplot(fig)


#=============================================
#                 STEP 1
#=============================================
import tensorflow as tf
import keras
! pip install scikeras
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from keras import initializers
from keras import regularizers
#=============================================
#                 END. STEP 1
#=============================================

#=============================================
#                 STEP 2
#=============================================
def build_model_optuna(hyperparams, input_shape):
    model = Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    n_layers = hyperparams.suggest_int("n_layers", 3, 8, 1)
    for i in range(n_layers):
        
        n_units = hyperparams.suggest_int(f"units_{i}", 16, 512, step = 2)
        
        activation = hyperparams.suggest_categorical(f"activation_{i}", ["relu", "tanh"])

        # Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ´Ğ»Ñ� Ğ½ĞµÑ‡ĞµÑ‚Ğ½Ñ‹Ñ… Ñ�Ğ»Ğ¾ĞµĞ² (Ğ½Ğ°Ñ‡Ğ¸Ğ½Ğ°Ñ� Ñ� 0-Ğ³Ğ¾ Ğ¸Ğ½Ğ´ĞµĞºÑ�Ğ°)
        if i % 2 == 1:  # Ğ�ĞµÑ‡ĞµÑ‚Ğ½Ñ‹Ğµ Ñ�Ğ»Ğ¾Ğ¸ (1Ğ¹, 3Ğ¹, 5Ğ¹...)
            l1_reg = hyperparams.suggest_float(f"l1_reg_{i}", 1e-5, 1e-1, log=True)
            l2_reg = hyperparams.suggest_float(f"l2_reg_{i}", 1e-5, 1e-1, log=True)
            kernel_reg = regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        else:
            kernel_reg = None  # Ğ§ĞµÑ‚Ğ½Ñ‹Ğµ Ñ�Ğ»Ğ¾Ğ¸ Ğ±ĞµĞ· Ñ€ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ¸Ğ·Ğ°Ñ†Ğ¸Ğ¸

        model.add(layers.Dense(units=n_units,
                               activation=activation,
                               kernel_regularizer=kernel_reg))
    
        if i == 2 and n_layers == 5:
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))
        if i == 3 and n_layers == 6:
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))
        if (i == 2 or i == 4) and (n_layers == 7):
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))
        if (i == 3 or i == 5) and (n_layers == 8):
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(1, activation='linear'))
    
    optim = hyperparams.suggest_categorical("optimizer", ["adam", "rmsprop"])
    learning_rate = hyperparams.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    
    if optim == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)

    model.compile(optimizer=optimizer, loss='mse')
    return model
#=============================================
#                 END. STEP 2
#=============================================

#=============================================
#                 STEP 3
#=============================================
def objective(trial):
    X_full['Sex'] = X_full['Sex'].astype('int32')
    model = build_model_optuna(trial, X_full.shape[1])
    keras_reg = KerasRegressor(model = model, epochs = 50, verbose = 0)
    
    # Cross-validation configuration
    cv = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(df_train))

    n_bins = 10
    discretizer = KBinsDiscretizer(n_bins = n_bins, encode='ordinal', strategy='quantile')
    duration_binned = discretizer.fit_transform(df_train[['Duration']]).astype(int).flatten()

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full, duration_binned)):
        print(f"\n Fold NN (Optim) {fold + 1}")
        # Data separation
        X_train = X_full.iloc[idx_train].copy()
        X_valid = X_full.iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())

        keras_reg.fit(X_train, y_train, epochs = 50, validation_data = [X_valid, y_valid], batch_size = 4096, verbose = 0)

        y_pred_val[idx_valid] = keras_reg.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid])))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold NN (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(df_train['Calories'])), np.expm1(y_pred_val)))
    print(f"\nğŸ�¯ Overall CV NN (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid
#=============================================
#                 END. STEP 3
#=============================================

#=============================================
#                 STEP 4
#=============================================
#sampler = TPESampler(seed = 42)
#study_0 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_0.optimize(objective, n_trials = 25)
#=============================================
#                 END. STEP 4
#=============================================
    
#=============================================
#                 STEP 5
#=============================================
def Evaluate_Optuna_Model(hyperparams, input_shape):
    model = Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    n_layers = hyperparams.get("n_layers", 4)
    for i in range(n_layers):
        # ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ¸Ğ· Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€Ñ�
        n_units = hyperparams.get(f"units_{i}", 64)
        activation = hyperparams.get(f"activation_{i}", "relu")
        l1_reg = hyperparams.get(f"l1_reg_{i}", 0.0)
        l2_reg = hyperparams.get(f"l2_reg_{i}", 0.0)
        model.add(layers.Dense(
            units=n_units,
            activation=activation,
            kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
        if f"dropout_rate_{i}" in hyperparams:
            dropout_rate = hyperparams[f"dropout_rate_{i}"]
            model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(1, activation='linear'))
    optim = hyperparams.get("optimizer", "adam")
    learning_rate = hyperparams.get("learning_rate", 1e-3)
    optimizer = tf.keras.optimizers.get(optim)
    optimizer.learning_rate = learning_rate
    model.compile(optimizer=optimizer, loss="mse", metrics = [keras.metrics.MeanSquaredError()])
    return model
#=============================================
#                 END. STEP 5
#=============================================

#=============================================
#                 STEP 6
#=============================================
nn_params = {'n_layers': 4, 'units_0': 190, 'activation_0': 'relu', 
             'units_1': 410, 'activation_1': 'tanh', 'l1_reg_1': 1.224665117421575e-05, 'l2_reg_1': 0.0005516789129968931, 
             'units_2': 178, 'activation_2': 'relu', 
             'units_3': 504, 'activation_3': 'tanh', 'l1_reg_3': 0.008365047330010568, 'l2_reg_3': 0.00011972670385924422, 
             'optimizer': 'adam', 
             'learning_rate': 0.0015503996764528489}
model = Evaluate_Optuna_Model(nn_params, X_full.shape[1])
model.summary()
print(f"Model has {len(model.layers)} layers")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({type(layer)})")

    if isinstance(layer, layers.Dense):
        print(f"  - Units: {layer.units}")
        print(f"  - Activation: {layer.activation}")

        if layer.kernel_regularizer:
            print(f"  - Kernel Regularizer: {layer.kernel_regularizer.__class__.__name__}")
            print(f"    - l1: {layer.kernel_regularizer.l1 if hasattr(layer.kernel_regularizer, 'l1') else None}")
            print(f"    - l2: {layer.kernel_regularizer.l2 if hasattr(layer.kernel_regularizer, 'l2') else None}")

        if layer.bias_regularizer:
            print(f"  - Bias Regularizer: {layer.bias_regularizer.__class__.__name__}")

    elif isinstance(layer, layers.Dropout):
        print(f"  - Rate: {layer.rate}")

if hasattr(model, 'optimizer') and model.optimizer is not None:
    lr = model.optimizer.learning_rate.numpy()
    print(f"\nOptimizer: {model.optimizer.__class__.__name__}")
    print(f"Learning rate: {lr:.2e}")
else:
    print("\nModel optimizer is not compiled yet!")
#=============================================
#                 END. STEP 6
#=============================================

#=============================================
#                 STEP 7
#=============================================
X_full['Sex'] = X_full['Sex'].astype('int32')  
X_test['Sex'] = X_test['Sex'].astype('int32')  
# Cross-validation configuration
cv = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
rmsle_scores_valid = []
y_pred_val_NN = np.zeros(len(df_train))
y_pred_test_NN = np.zeros(len(df_test))

n_bins = 10
discretizer = KBinsDiscretizer(n_bins = n_bins, encode='ordinal', strategy='quantile')
duration_binned = discretizer.fit_transform(df_train[['Duration']]).astype(int).flatten()

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full, duration_binned)):
    print(f"\n Fold NN (Optim) {fold + 1}")
    # Data separation
    X_train = X_full.iloc[idx_train].copy()
    X_valid = X_full.iloc[idx_valid].copy()
    X_test = X_test.copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    model.fit(X_train, y_train, epochs = 50, validation_data = [X_valid, y_valid], batch_size = 4096, verbose = 0)

    y_pred_val_NN[idx_valid] = model.predict(X_valid).flatten()
    y_pred_test_NN += model.predict(X_test).flatten()

    fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_NN[idx_valid])))
    rmsle_scores_valid.append(fold_rmsle_valid)
    print(f"Fold NN (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(df_train['Calories'])), np.expm1(y_pred_val_NN)))
print(f"\nğŸ�¯ Overall CV NN (Optim) RMSLE: {overall_rmsle_valid:.5f}")
y_pred_test_NN /= 5


'''
estimators = [('XGBoost', model_1),
              ('LightGBM', model_2),
              ('CatBoost', model_3)]
model_4 = StackingRegressor(estimators = estimators, final_estimator = Lasso(alpha = 0.001))
'''


'''
# Setting the initial parameters for validation
cv = KFold(n_splits = 5, random_state = 41, shuffle = True)
rmsle_scores_valid_stacking = []
rmsle_scores_train_stacking = []
y_pred_val_stacking = np.zeros(len(X_full))
y_pred_train_stacking = np.zeros(len(X_full))
y_pred_test_stacking = np.zeros(len(X_test))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
    print(f"\n Fold Stacking (Final) {fold + 1}")
    # Separating the training and radiation data from the source dataset
    X_train = X_full[predictors].iloc[idx_train].copy()
    X_valid = X_full[predictors].iloc[idx_valid].copy()
    X_test  = X_test[predictors].copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    model_4.fit(X_train, y_train)
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_stacking[idx_valid] = model_4.predict(X_valid)
    y_pred_train_stacking[idx_train] = model_4.predict(X_train)
    y_pred_test_stacking += model_4.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid_stacking = root_mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_stacking[idx_valid]))
    fold_rmsle_train_stacking = root_mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_stacking[idx_train]))
    rmsle_scores_valid_stacking.append(fold_rmsle_valid_stacking)
    rmsle_scores_train_stacking.append(fold_rmsle_train_stacking)
    print(f"Fold Stacking (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid_stacking:.5f}")
    print(f"Fold Stacking (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train_stacking:.5f}")
'''


# y_pred_test_stacking /= 5


# The Heteroscedasticity Test
X_train_const = add_constant(X_full)
residuals_train = y_full-y_pred_train_cat
if X_train_const.shape[1] < 2:
    raise ValueError("After removing the features, there are less than 2 variables left. The test is not possible.")
bp_test = het_breuschpagan(residuals_train, X_train_const)
print(f"\nBreusch-Pagan Test: p-value = {bp_test[1]:.10f}")
if bp_test[1] < 0.05:
    print("Heteroskedasticity is present (p < 0.05)")
else:
    print("No heteroskedasticity was detected")


# The Darbin-Watson autocorrelation test
dw_test = durbin_watson(residuals_train)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}")
if dw_test < 1.5 or dw_test > 2.5:
    print("Autocorrelation of residues detected")
else:
    print("No residue autocorrelation was detected")


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub.head()


sub['Calories'] = np.clip(np.expm1((y_pred_test_xgb+y_pred_test_cat)/(2)), 1, 314)#np.clip(np.expm1(y_pred_test_NN), 1, 314)
sub.head()


sub.info()


sub.to_csv('submission_5_Keras.csv', index = False)


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
fig = plt.figure(figsize = (20, 10))

X_mpl = X_test
X_mpl['Calories_xgb'] = y_pred_test_xgb
X_mpl['Calories_lgbm'] = y_pred_test_lgbm
X_mpl['Calories_cat'] = y_pred_test_cat
grouped = X_mpl.groupby(['Duration']).max().reset_index()

plt.plot(grouped['Duration'], grouped['Calories_xgb'], color = 'green', linewidth = 2, label = 'XGB-prediction')
plt.plot(grouped['Duration'], pd.concat([grouped[(grouped['Duration'] <= 4)]['Calories_lgbm']*1, grouped[(grouped['Duration'] > 4)]['Calories_lgbm']], axis = 0), color = 'red', linewidth = 2, label = 'LGBM-prediction')
plt.plot(grouped['Duration'], grouped['Calories_cat'], color = 'blue', linewidth = 2, label = 'CatBoost-prediction')
plt.title('Graphs of the predicted variable for 3 algorithms')
plt.legend()
plt.xlabel('Duration')
plt.ylabel('Calories');


fig, axes = plt.subplots(2, 1, figsize = (15, 10))
# Hist
sns.histplot(sub['Calories'], bins = 30, kde = True, ax = axes[0], color = 'blue')
axes[0].set_title('Hist Calories - Test Pred')
# BoxPlot
sns.boxplot(x = sub['Calories'], ax = axes[1], color = 'blue')
axes[1].set_title('Boxplot Calories - Test Pred');

