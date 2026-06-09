# Base
import os
import glob
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from itertools import combinations
from IPython.display import Image

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py
from plotly.offline import init_notebook_mode

import warnings
warnings.filterwarnings("ignore")

#Statistics
from scipy.stats import skew
from scipy import stats
from scipy.stats import randint
from scipy.stats import norm, cramervonmises, anderson, kstest, norm, cramervonmises
from statsmodels.stats.diagnostic import lilliefors, normal_ad, het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
!pip install arch
from arch.unitroot import VarianceRatio
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import TransformedTargetRegressor
from cuml.preprocessing import TargetEncoder
# I'm facing a problem that I can't import the cuML library into my local jupiter-notebook (or PyCharm) on my computer. 
# You can use TargetEncoder through the sklearn functionality, but it only works with 1 column, which puts a heavy load on calculations. 
# I have good video card + processor characteristics, as a result of which calculations would be performed much faster on my local device than on the Kaggle cloud platform. 
# There is a description of this library on the official Anaconda website and it supports it only on the Linux operating system. 
# I have a Windows operating system. But the developer's website says that it can also be installed on windows if you create an environment with Python=3.6, 3.7. 
# As a result, I was unable to install this library on my local computer. If anyone has encountered this problem, please contact me or write in the comments, thank you.
import cudf

#Feature engineering
from sklearn.feature_selection import mutual_info_regression

#Transformers and Pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn import set_config

#Models ML (Linear and Tree)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from xgboost import XGBRegressor
from xgboost import plot_importance
import lightgbm as lgb
from lightgbm import LGBMRegressor

#Model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
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
    ('dateutil', 'python-dateutil'),
    ('shap', 'shap'),
    ('scipy', 'scipy'),
    ('IPython', 'IPython'),
    ('pip', 'pip')]

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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv').drop(columns = ['id'])
df_train.head()


df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv').drop(columns = ['id'])
df_test.head()


df_train.info()


df_test.info()


nan_columns = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']

for i in nan_columns:
    mode_value_train = df_train[i].mode()[0]
    mode_value_test = df_test[i].mode()[0]
    df_train[i] = df_train[i].fillna(mode_value_train)
    df_test[i] = df_test[i].fillna(mode_value_test)

#df_train = df_train.drop(columns = ['Podcast_Name'])
df_train.info(), df_test.info()


df_train.Episode_Title.unique() # replace Episode N - N (type - int32)


df_train.Genre.unique()


df_train.Publication_Day.unique()


df_train.Publication_Time.unique()


df_train.Episode_Sentiment.unique()


# Target - Listening_Time_minutes
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize = (20, 30))

sns.histplot(data = df_train, x = 'Listening_Time_minutes', bins=30, kde=True, hue = 'Publication_Day', ax=ax1, palette='Set1', linewidth = 1)
ax1.set_title('Histogram of the target variable distribution - (agg - Publication_Day)')
ax1.set_ylabel("Number of values", fontsize = 12)
ax1.set_xlabel("Podcast duration (minutes) - (agg - Publication_Day)", fontsize = 12)
ax1.grid()
ax1.tick_params(axis = 'x', labelrotation = 0, labelsize = 14)
ax1.tick_params(axis = 'y', labelrotation = 0, labelsize = 14)

sns.histplot(data = df_train, x = 'Listening_Time_minutes', bins=30, kde=True, hue = 'Publication_Time', ax=ax2, palette='Set1', linewidth = 1)
ax2.set_title('Histogram of the target variable distribution - (agg - Publication_Time)')
ax2.set_ylabel("Number of values", fontsize = 12)
ax2.set_xlabel("Podcast duration (minutes)", fontsize = 12)
ax2.grid()
ax2.tick_params(axis = 'x', labelrotation = 0, labelsize = 14)
ax2.tick_params(axis = 'y', labelrotation = 0, labelsize = 14)

sns.histplot(data = df_train, x = 'Listening_Time_minutes', bins=30, kde=True, hue = 'Episode_Sentiment', ax=ax3, palette='Set1', linewidth = 1)
ax3.set_title('Histogram of the target variable distribution - (agg - Episode_Sentiment)')
ax3.set_ylabel("Number of values", fontsize = 12)
ax3.set_xlabel("Podcast duration (minutes)", fontsize = 12)
ax3.grid()
ax3.tick_params(axis = 'x', labelrotation = 0, labelsize = 14)
ax3.tick_params(axis = 'y', labelrotation = 0, labelsize = 14)

sns.boxplot(x = df_train['Listening_Time_minutes'], ax = ax4, color = 'red')
ax4.set_title('Boxplot of target variable allocation')
ax4.set_xlabel("Podcast duration (minutes)", fontsize = 12)
ax4.grid()
ax4.tick_params(axis = 'x', labelrotation = 0, labelsize = 14)
ax4.tick_params(axis = 'y', labelrotation = 0, labelsize = 14)


# Boxplot for numeric features
num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train[num_cols], orient='h')
plt.title('Distribution of numerical features')
plt.show()


# Interactive histogram
fig = px.histogram(df_train, x='Listening_Time_minutes', nbins=30,  title='Allocation of listening time')
init_notebook_mode(connected=True)
py.iplot(fig)


# Distribution by genre
plt.figure(figsize=(12, 6))
sns.countplot(data=df_train, y='Genre', order=df_train['Genre'].value_counts().index)
plt.title('Number of episodes by genre')
plt.show()

# Publication time by day
plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='Publication_Day', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Distribution of publication days')
plt.show()


# Interactive pie chart for publication time
fig = px.pie(df_train, names = 'Publication_Time', title = 'Share of publication time')
init_notebook_mode(connected=True)
py.iplot(fig)


# The influence of genre on listening time
plt.figure(figsize=(14, 8))
sns.boxplot(data = df_train, x = 'Listening_Time_minutes', y = 'Genre')
plt.title('Listening time by genre')
plt.show()

# The correlation matrix
corr = df_train[num_cols + ['Listening_Time_minutes']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin = -1, vmax = 1)
plt.title('Correlation of numerical features')
plt.show()


# Interactive scatter plot
fig = px.scatter(df_train, x = 'Episode_Length_minutes', y = 'Listening_Time_minutes', color = 'Genre', title = 'The relationship between episode length and listening time')
init_notebook_mode(connected=True)
py.iplot(fig)


# Combined schedule of the day and time of publication
plt.figure(figsize=(14, 8))
sns.countplot(data=df_train, x='Publication_Day', hue='Publication_Time',order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Distribution of publications by day and time')
plt.show()


# 3D graph for Host/Guest Popularity and Listening_Time communication
fig = px.scatter_3d(df_train, x = 'Host_Popularity_percentage', y = 'Guest_Popularity_percentage', z = 'Listening_Time_minutes', color = 'Genre')
init_notebook_mode(connected=True)
py.iplot(fig)


# The influence of sentiment on listening time
plt.figure(figsize=(10, 6))
sns.barplot(data = df_train, x ='Episode_Sentiment', y ='Listening_Time_minutes', order = ['Negative', 'Neutral', 'Positive'])
plt.title('Average listening time by sentiment')
plt.show()


# Interactive violin plot
fig = px.violin(df_train, x = 'Episode_Sentiment', y = 'Listening_Time_minutes', box = True, title = 'Distribution of listening time by sentiment')
init_notebook_mode(connected=True)
py.iplot(fig)


# The relationship between the amount of advertising and listening time
plt.figure(figsize=(10, 6))
sns.lineplot(data = df_train, x = 'Number_of_Ads', y = 'Listening_Time_minutes', errorbar=None)
plt.title('The effect of advertising on listening time')
plt.show()


# Listening time trend by episode numbers
fig = px.line(df_train, x='Episode_Title', y = 'Listening_Time_minutes', title = 'Dynamics of listening time by episode')
init_notebook_mode(connected=True)
py.iplot(fig)


df_train.info()


df_orig = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
df_orig.head()


df_orig.info()


df_orig = df_orig[(df_orig['Listening_Time_minutes'].isna() == False)].reset_index(drop = True)


df_orig


df_orig.info()


nan_columns = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

for i in nan_columns:
    mode_value = df_orig[i].mode()[0]
    df_orig[i] = df_orig[i].fillna(mode_value)

#df_orig = df_orig.drop(columns = ['Podcast_Name'])
df_orig.info()


df_train = pd.concat([df_train, df_orig], axis = 0).reset_index(drop = True)
df_train.info()


# Delete the word "Episode" and the space, then convert to integers
df_train['Episode_Num'] = (df_train['Episode_Title'].str.replace('Episode ', '', regex = False).astype('category'))
df_test['Episode_Num'] = (df_test['Episode_Title'].str.replace('Episode ', '', regex = False).astype('category'))
# Checking the result
print(df_train.Episode_Num.unique()), print(df_test.Episode_Num.unique())


df_train = df_train.drop(columns = ['Episode_Title'])
df_test = df_test.drop(columns = ['Episode_Title'])


df_train.head()


df_train.info()


df_train.describe().T


df_train.describe(exclude = np.number).T


df_train = df_train[df_train['Number_of_Ads'] < 10].reset_index(drop = True)
df_train.shape


cat_cols = ['Episode_Num', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment','Podcast_Name','Genre']
def update_cols(df):
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df
df_train = update_cols(df_train)
df_test = update_cols(df_test)


df_train.info()


# TRAIN DATA
# 1. Ad Density
df_train['ads_per_minute'] = df_train['Number_of_Ads'] / (df_train['Episode_Length_minutes'] + 1e-3)

# 2. Is Weekend
df_train['is_weekend'] = df_train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

# 3. Time of Day Features
df_train['is_morning'] = (df_train['Publication_Time'] == 'Morning').astype(int)
df_train['is_night'] = (df_train['Publication_Time'] == 'Night').astype(int)

# 4. Episode Length Buckets
df_train['length_bucket'] = pd.cut(df_train['Episode_Length_minutes'], bins=[0, 30, 60, 90, 200], labels=['short', 'medium', 'long', 'very_long'])

# 5. Sentiment Ordinal Mapping
sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df_train['sentiment_score'] = df_train['Episode_Sentiment'].map(sentiment_map)

# 6. Host-Guest Popularity Ratio
df_train['popularity_ratio'] = df_train['Guest_Popularity_percentage'] / (df_train['Host_Popularity_percentage'] + 1e-3)

# 7. Genre + Sentiment Interaction
df_train['genre_sentiment'] = df_train['Genre'].astype(str) + "_" + df_train['Episode_Sentiment'].astype(str)
df_train['genre_sentiment'] = df_train['genre_sentiment'].astype('category')
# Handle Missing Values
# Fill numeric columns using Genre-wise mean
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    df_train[col] = df_train.groupby('Genre')[col].transform(lambda x: x.fillna(x.mean()))


df_train.info()


# TEST DATA
# 1. Ad Density
df_test['ads_per_minute'] = df_test['Number_of_Ads'] / (df_test['Episode_Length_minutes'] + 1e-3)

# 2. Is Weekend
df_test['is_weekend'] = df_test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

# 3. Time of Day Features
df_test['is_morning'] = (df_test['Publication_Time'] == 'Morning').astype(int)
df_test['is_night'] = (df_test['Publication_Time'] == 'Night').astype(int)

# 4. Episode Length Buckets
df_test['length_bucket'] = pd.cut(df_test['Episode_Length_minutes'], bins=[0, 30, 60, 90, 200], labels=['short', 'medium', 'long', 'very_long'])

# 5. Sentiment Ordinal Mapping
sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df_test['sentiment_score'] = df_test['Episode_Sentiment'].map(sentiment_map)

# 6. Host-Guest Popularity Ratio
df_test['popularity_ratio'] = df_test['Guest_Popularity_percentage'] / (df_test['Host_Popularity_percentage'] + 1e-3)

# 7. Genre + Sentiment Interaction
df_test['genre_sentiment'] = df_test['Genre'].astype(str) + "_" + df_test['Episode_Sentiment'].astype(str)
df_test['genre_sentiment'] = df_test['genre_sentiment'].astype('category')

# Handle Missing Values
# Fill numeric columns using Genre-wise mean
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    df_test[col] = df_test.groupby('Genre')[col].transform(lambda x: x.fillna(x.mean()))


df_test.info()


encode_col = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
pair_size = [2, 3, 4, 5]
for ps in pair_size:
    for col in tqdm(list(combinations(encode_col, ps))):
        new_col_name = '_'.join(col)
        
        df_train[new_col_name] = df_train[list(col)].astype(str).agg('_'.join, axis=1)
        df_train[new_col_name] = df_train[new_col_name].astype('category')
        
        df_test[new_col_name] = df_test[list(col)].astype(str).agg('_'.join, axis=1)
        df_test[new_col_name] = df_test[new_col_name].astype('category')


Y = df_train['Listening_Time_minutes']
X = df_train.drop(columns = ['Listening_Time_minutes'])


#scaler = StandardScaler().set_output(transform = "pandas")
#X_scaled = scaler.fit_transform(X)
#X_test_scaled = scaler.transform(df_test)


def objective(trial):
    
    xgb_params = {
        'n_estimators': trial.suggest_int("n_estimators", 500, 3000, step = 100),
        'max_depth': trial.suggest_int("max_depth", 6, 18, step = 2),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.9, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        'subsample': trial.suggest_float("subsample", 0.5, 0.95),
        'gamma': trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 0.95),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'gpu',
        'seed': 42
    }

    model = XGBRegressor(**xgb_params, enable_categorical = True)

    # Cross-validation configuration
    cv = KFold(n_splits = 7, random_state = 42, shuffle = True)
    rmse_scores = []
    y_pred_val = np.zeros(len(X))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X, Y)):
        print(f"\n Fold XGB (Optim) {fold + 1}")
        X_train = cudf.from_pandas(X.iloc[idx_train].copy())
        X_valid = cudf.from_pandas(X.iloc[idx_valid].copy())
        X_test  = cudf.from_pandas(df_test[X.columns].copy())
        y_train = cudf.Series(Y.iloc[idx_train].copy())
        y_valid = Y.iloc[idx_valid].copy()

        categorical_columns = X_train.select_dtypes(include=['category']).columns
        print("Target encoding (XGBoost Optim): ", end = "")
        for c in tqdm(categorical_columns, desc = "Encoding columns"):
            encoder = TargetEncoder(n_folds = 5,
                                    smooth = 0,
                                    split_method = 'random',
                                    stat = 'mean')
            X_train[c] = encoder.fit_transform(X_train[[c]], y_train)
            X_valid[c] = encoder.transform(X_valid[[c]])
            X_test[c]  = encoder.transform(X_test[[c]])

        X_train = X_train.to_pandas()
        X_valid = X_valid.to_pandas()
        X_test  = X_test.to_pandas()
        y_train = y_train.to_pandas()

        model.fit(X_train, y_train,
                  eval_set = [(X_valid, y_valid)],
                  early_stopping_rounds = 500,
                  verbose = 100)

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmse = mean_squared_error(y_valid, y_pred_val[idx_valid]) ** 0.5
        rmse_scores.append(fold_rmse)
        print(f"Fold (XGBoost Optim) {fold + 1} RMSE (XGBoost Optim): {fold_rmse:.5f}")

    overall_rmse = mean_squared_error(Y, y_pred_val) ** 0.5
    print(f"\n Overall CV (XGBoost Optim) RMSE: {overall_rmse:.5f}")

    return overall_rmse


#sampler = TPESampler(seed=42)
#study_1 = optuna.create_study(direction="minimize", sampler=sampler)
#study_1.optimize(objective, n_trials = 10)


import pickle
# A function for saving a study
def save_study(study, filename):
    with open(filename, "wb") as f:
        pickle.dump(study, f)

# Function for uploading a study
def load_study(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# save_study(study_1, "study_1_XGB_New.pkl")


#study_1 = load_study('study_1_XGB_New.pkl')


Xgb_params_1 = {'n_estimators': 2200, 'max_depth': 10, 'learning_rate': 0.03438801091888567, 
                'reg_alpha': 0.0005414413211338525, 'subsample': 0.5831845049864872, 
                'gamma': 0.08105016126411585, 'colsample_bytree': 0.8038363351847244, 
                'min_child_weight': 10, 'reg_lambda': 0.029794544625913636}

Xgb_params_2 = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                'tree_method': 'gpu_hist', 'device': 'cuda', 'seed': 42, 'enable_categorical': True}

model_1 = XGBRegressor(**Xgb_params_1, **Xgb_params_2)
model_1


cv = KFold(n_splits = 7, random_state = 42, shuffle = True)
rmse_scores = []
y_pred_val = np.zeros(len(X))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X, Y)):
    print(f"\n Fold XGB (Check) {fold + 1}")
    X_train = cudf.from_pandas(X.iloc[idx_train].copy())
    X_valid = cudf.from_pandas(X.iloc[idx_valid].copy())
    y_train = cudf.Series(Y.iloc[idx_train].copy())
    y_valid = Y.iloc[idx_valid].copy()

    categorical_columns = X_train.select_dtypes(include=['category']).columns
    print("Target encoding (XGBoost Check): ", end = "")
    for c in tqdm(categorical_columns, desc = "Encoding columns"):
        encoder = TargetEncoder(n_folds = 5, smooth = 0,split_method = 'random', stat = 'mean')
        X_train[c] = encoder.fit_transform(X_train[[c]], y_train)
        X_valid[c] = encoder.transform(X_valid[[c]])

    X_train = X_train.to_pandas()
    X_valid = X_valid.to_pandas()
    y_train = y_train.to_pandas()

    model_1.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              early_stopping_rounds = 500,
              verbose = 100)

    y_pred_val[idx_valid] = model_1.predict(X_valid)

    fold_rmse = mean_squared_error(y_valid, y_pred_val[idx_valid]) ** 0.5
    rmse_scores.append(fold_rmse)
    print(f"Fold (XGBoost Check) {fold + 1} RMSE (XGBoost Check): {fold_rmse:.5f}")

overall_rmse = mean_squared_error(Y, y_pred_val) ** 0.5
print(f"\n Overall CV (XGBoost Check) RMSE: {overall_rmse:.5f}")


feature_names = X.columns
importances = model_1.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y=sorted_features[-25:], width=sorted_importances[-25:],color='#3498db',edgecolor='black')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("XGB Feature Importance - Target Encoding)", fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)

plt.tight_layout()
plt.show()


def objective(trial):
    lgbm_params = {
        'n_estimators': trial.suggest_int("n_estimators", 1000, 3000, step=500),
        'num_leaves': trial.suggest_int("num_leaves", 31, 127, step=32),
        'max_depth': trial.suggest_int("max_depth", 3, 8),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        'subsample': trial.suggest_float("subsample", 0.6, 0.95),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.95),
        'min_child_weight': trial.suggest_float("min_child_weight", 1e-4, 1e-1, log=True),
        'device': 'cpu', #I don't have enough GPU memory to calculate such a large number of features, so we will use the CPU, even though it will be a long time.
        'random_state': 42,
        'verbose': -1
    }

    model = LGBMRegressor(**lgbm_params)
    cv = KFold(n_splits = 7, random_state=42, shuffle=True) 
    rmse_scores = []
    y_pred_val = np.zeros(len(X))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X, Y)):
        print(f"\nFold LGBM (Optim) {fold + 1}")
        X_train = cudf.from_pandas(X.iloc[idx_train].copy())
        X_valid = cudf.from_pandas(X.iloc[idx_valid].copy())
        y_train = cudf.Series(Y.iloc[idx_train].copy())
        y_valid = Y.iloc[idx_valid].copy()

        categorical_columns = X_train.select_dtypes(include=['category']).columns
        print("Target encoding (LGBM Optim): ", end = "")
        for c in tqdm(categorical_columns, desc = "Encoding columns"):
            encoder = TargetEncoder(n_folds = 5, 
                                    smooth = 0, 
                                    split_method = 'random', 
                                    stat = 'mean')
            X_train[c] = encoder.fit_transform(X_train[[c]], y_train)
            X_valid[c] = encoder.transform(X_valid[[c]])

        X_train = X_train.to_pandas()
        X_valid = X_valid.to_pandas()
        y_train = y_train.to_pandas()
        
        model.fit(X_train, y_train, 
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(500, verbose = 100)])

        y_pred_val[idx_valid] = model.predict(X_valid)
        fold_rmse = mean_squared_error(y_valid, y_pred_val[idx_valid]) ** 0.5
        rmse_scores.append(fold_rmse)
        print(f"Fold (LGBM Optim) {fold + 1} RMSE (LGBM Optim): {fold_rmse:.5f}")

    overall_rmse = mean_squared_error(Y, y_pred_val) ** 0.5
    print(f"\n Overall CV (LGBM Optim) RMSE: {overall_rmse:.5f}")

    return overall_rmse


sampler = TPESampler(seed=42)
study_2 = optuna.create_study(direction="minimize", sampler=sampler)
study_2.optimize(objective, n_trials = 5)


save_study(study_2, "study_2_LGBM_New.pkl")


# study_2 = load_study('study_2_LGBM_New.pkl')


LGBM_params_1 = study_2.best_params

LGBM_params_2 = {'random_state': 42, 'device': 'cpu', 'verbose': -1}

model_2 = LGBMRegressor(**LGBM_params_1, **LGBM_params_2)
model_2


cv = KFold(n_splits = 7, random_state = 42, shuffle = True)
rmse_scores = []
y_pred_val = np.zeros(len(X))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X, Y)):
    print(f"\n Fold LGBM (Check) {fold + 1}")
    X_train = cudf.from_pandas(X.iloc[idx_train].copy())
    X_valid = cudf.from_pandas(X.iloc[idx_valid].copy())
    y_train = cudf.Series(Y.iloc[idx_train].copy())
    y_valid = Y.iloc[idx_valid].copy()

    categorical_columns = X_train.select_dtypes(include=['category']).columns
    print("Target encoding (LGBM Check): ", end = "")
    for c in tqdm(categorical_columns, desc = "Encoding columns"):
        encoder = TargetEncoder(n_folds = 5, smooth = 0, split_method = 'random', stat = 'mean')
        X_train[c] = encoder.fit_transform(X_train[[c]], y_train)
        X_valid[c] = encoder.transform(X_valid[[c]])

    X_train = X_train.to_pandas()
    X_valid = X_valid.to_pandas()
    y_train = y_train.to_pandas()

    model_2.fit(X_train, y_train, 
                eval_set=[(X_valid, y_valid)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(500, verbose = 100)])

    y_pred_val[idx_valid] = model_2.predict(X_valid)

    fold_rmse = mean_squared_error(y_valid, y_pred_val[idx_valid]) ** 0.5
    rmse_scores.append(fold_rmse)
    print(f"Fold (LGBM Check) {fold + 1} RMSE (LGBM Check): {fold_rmse:.5f}")

overall_rmse = mean_squared_error(Y, y_pred_val) ** 0.5
print(f"\n Overall CV (LGBM Check) RMSE: {overall_rmse:.5f}")


feature_names = X.columns
importances = model_2.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y = sorted_features[-25:], width=sorted_importances[-25:],color='#3498db',edgecolor='black')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("LGBM Feature Importance", fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)

plt.tight_layout()
plt.show()


Image("/kaggle/input/5-04-kaggle/5-04-Kaggle_1.png")


# Defining a meta-model
estimators = [('XGB', model_1),
              ('LGBM', model_2)]
SR = StackingRegressor(estimators = estimators, final_estimator = Lasso(alpha = 0.01))

# Uploading the prediction file
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

n_splits = 7

# We use cross-validation to predict test data.
# Defining the initial parameters
# Validation method
cv = KFold(n_splits = 7, random_state = 42, shuffle = True)
# An empty array of errors on each fold
rmse_scores = []
# A null array for filling in (by indexes) the prediction data on the validation sample
y_pred_val_SR = np.zeros(len(X))
# A null array for filling in (by indexes) the prediction data on the train sample
y_pred_train_SR = np.zeros(len(X))
# Zero array to fill ((sum of values from each fold)/(number of folds)) of prediction data in the test sample
y_pred_test = np.zeros(len(sub))

# Starting the iterative validation process
for fold, (idx_train, idx_valid) in enumerate(cv.split(X, Y)):
    # Output the fold number
    print(f"\n Fold {fold + 1}")
    # To speed up calculations, we use the cuda library instead of pandas.
    X_train = cudf.from_pandas(X.iloc[idx_train].copy())
    X_valid = cudf.from_pandas(X.iloc[idx_valid].copy())
    X_test  = cudf.from_pandas(df_test[X.columns].copy())
    y_train = cudf.Series(Y.iloc[idx_train].copy())
    y_valid = Y.iloc[idx_valid].copy()

    # Defining categorical features in the dataset
    categorical_columns = X_train.select_dtypes(include=['category']).columns
    print("Target encoding (Stacking Check): ", end = "")
    # We visualize the toolbar encoding our features
    for c in tqdm(categorical_columns, desc = "Encoding columns"):
        # Learning from the training sample data via the Target Encoder
        encoder = TargetEncoder(n_folds = 5, smooth = 0, split_method = 'random', stat = 'mean')
        X_train[c] = encoder.fit_transform(X_train[[c]], y_train)
        X_valid[c] = encoder.transform(X_valid[[c]])
        X_test[c] = encoder.transform(X_test[[c]])

    # We return to pandas to work with the model (XGB and LGBM do not support working with cudf)
    X_train = X_train.to_pandas()
    X_valid = X_valid.to_pandas()
    X_test = X_test.to_pandas()
    y_train = y_train.to_pandas()

    # Training the meta-model
    SR.fit(X_train, y_train)
    # Let's summarize the predictions on the test sample from each fold
    y_pred_test += SR.predict(X_test)
    # Filling in the train prediction array by idx_train
    y_pred_train_SR[idx_train] = SR.predict(X_train)
    # Filling in the validation prediction array by idx_valid
    y_pred_val_SR[idx_valid] = SR.predict(X_valid)

    # We get an error on every fold
    fold_rmse = mean_squared_error(y_valid, y_pred_val_SR[idx_valid]) ** 0.5
    # We add errors from each fold to the rmse_scores array to improve the final error.
    rmse_scores.append(fold_rmse)
    print(f"Fold Stacking {fold + 1} RMSE Stacking: {fold_rmse:.5f}")

# We get the final error from our validation algorithm
overall_rmse_valid = mean_squared_error(Y, y_pred_val_SR) ** 0.5
# We get the final error from our train algorithm
overall_rmse_train = mean_squared_error(Y, y_pred_train_SR) ** 0.5
# and divide the sum of predictions from the test sample by the number of folds.
y_pred_test /= n_splits
print(f"\n Overall CV Stacking RMSE TRAIN: {overall_rmse_train:.5f}")
print(f"\n Overall CV Stacking RMSE VALID: {overall_rmse_valid:.5f}")


submission_df = pd.DataFrame({'id': sub['id'],
                              'Listening_Time_minutes': y_pred_test})

submission_df.to_csv('submission.csv', index = False)


'''
#Calculate the sum of squares of residual
ss_res=np.sum((Y-y_pred_train_SR)**2) 

#Calculate the total sum of squares
ss_tot=np.sum((Y-np.mean(Y))**2)

#Calculate the R-Squared using the formula we discussed earlier 
r_squared= 1-(ss_res/ss_tot)

#Printing the R-squared score 
print("The R_Squared score for the model_1 is", round(r_squared, 2))

#Plotting the data 
plt.scatter(X_scaled['Episode_Length_minutes'], Y, s = 0.1, label='Data', color = 'red')
plt.scatter(X_scaled['Episode_Length_minutes'], y_pred_train_SR, color='green', s = 0.1, label='Goodness-of-fit Model')
plt.xlim([-2, 2])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Non-Linear Model_1 Fitting")
plt.legend()
plt.show()
'''


'''
# Residuals on training data
residuals_train = Y - y_pred_train_SR

# Assumption of normality of residuals
sigma = np.std(residuals_train)
log_likelihood = np.sum(norm.logpdf(residuals_train, loc=0, scale=sigma))

# We get a text dump of all the trees
trees_dump = model_1.get_booster().get_dump()

# We count the total number of leaves in all trees
num_leaves = sum(tree_str.count("leaf=") for tree_str in trees_dump)

# An alternative way is through the built-in XGBoost methods
#num_trees = model_1.get_booster().num_boosted_rounds()
#num_leaves_alt = model_1.get_booster().get_score(importance_type="weight").get("total_leaf", 0)

k = num_leaves  # We use the counted number of leaves
n = len(Y)

# Calculation of criteria
BIC = -2 * log_likelihood + k * np.log(n)
print(f"BIC: {BIC:.2f}")
'''


'''
AIC = -2 * log_likelihood + 2 * k
print(f"AIC: {AIC:.2f}")
'''


'''
# Ğ�Ğ¾Ñ€Ğ¼Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ¾Ñ�Ñ‚Ğ°Ñ‚ĞºĞ¾Ğ²
residuals_normalized = (residuals_train - np.mean(residuals_train)) / np.std(residuals_train)

# Ğ¢ĞµÑ�Ñ‚ CvM
result = cramervonmises(residuals_normalized, 'norm')
print(f"CvM statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")
'''


def normality_tests(data, p_level=0.95, a_level=0.05):
    results = []
    n = len(data)
    alpha = 1 - p_level  # The level of significance
    # 1. Shapiro-Wilk Test
    try:
        statistic, p_value = stats.shapiro(data)
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(['Shapiro-Wilk', p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(['Shapiro-Wilk', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Shapiro-Wilk test:', e)
        
    # 2. Epps-Pulley Test
    try:
        def epps_pulley_test(x, alpha=0.05):
            n = len(x)
            x = np.sort(x)
            z = (x - np.mean(x)) / np.std(x, ddof=1)
            # Calculating statistics
            k = np.arange(1, n+1)
            term = (z @ (2*k - n - 1)) / (n * np.sqrt(n))
            T = (np.sqrt(n)/3) * (1 + 2 * term**2)
            # Critical values (tabular)
            cv_table = {0.10: 0.347,
                        0.05: 0.363,
                        0.01: 0.461}
            critical_value = cv_table[alpha]
            return T, critical_value
        if n < 8:
            raise ValueError("Minimum 8 observations required")
        statistic, critical_value = epps_pulley_test(data, alpha=a_level)
        reject = statistic > critical_value
        conclusion = "gaussian" if not reject else "non-gaussian"
        results.append(['Epps-Pulley', p_level, a_level, None, None, statistic, critical_value, not reject, conclusion])
    except Exception as e:
        results.append(['Epps-Pulley', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Epps-Pulley test:', e)

    # 3. D'Agostino's K-squared test
    try:
        statistic, p_value = stats.normaltest(data)
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(["D'Agostino", p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(["D'Agostino", p_level, a_level, None, None, None, None, None, "Error in print"])
        print('DAgostinos K-squared test:', e)

    # 4. Anderson-Darling test
    try:
        result = anderson(data)
        statistic = result.statistic
        critical_values = result.critical_values
        significance_level = result.significance_level
        idx = min(range(len(significance_level)), key=lambda i: abs(significance_level[i]/100 - alpha))
        critical_value = critical_values[idx]
        reject = statistic > critical_value
        conclusion = "gaussian" if not reject else "non-gaussian"
        results.append(['Anderson-Darling', p_level, a_level, None, None, statistic, critical_value, not reject, conclusion])
    except Exception as e:
        results.append(['Anderson-Darling', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Anderson-Darling test:', e)

    # 5. Kolmogorov-Smirnov test
    try:
        if n < 50:
            raise ValueError("Minimum 50 observations required")
        # Normalizing the data
        data_norm = (data - data.mean()) / data.std()
        statistic, p_value = kstest(data_norm, 'norm')
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(['Kolmogorov-Smirnov', p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(['Kolmogorov-Smirnov', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Kolmogorov-Smirnov test:', e)

    # 6. Lilliefors test
    try:
        statistic, p_value = lilliefors(data, pvalmethod='approx')
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(['Lilliefors', p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(['Lilliefors', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Lilliefors test:', e)

    # 7. Cramer-von Mises test
    try:
        if n < 20:
            raise ValueError("Minimum 20 observations required")
        # Normalizing the data
        data_norm = (data - data.mean()) / data.std()
        result = cramervonmises(data_norm, norm.cdf)
        statistic = result.statistic
        p_value = result.pvalue
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(['Cramer-von Mises', p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(['Cramer-von Mises', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Cramer-von Mises test:', e)

    # 8. Chi-squared test
    try:
        if n < 100:
            raise ValueError("Minimum 100 observations required")
        # Data binning
        _, bins = pd.qcut(data, 10, retbins=True)
        observed = pd.cut(data, bins=bins).value_counts().sort_index()
        expected = len(data)/10
        statistic, p_value = stats.chisquare(observed, f_exp=expected)
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(['Chi-squared', p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(['Chi-squared', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Chi-squared test:', e)

    # 9. Jarque-Bera test
    try:
        if n < 2000:
            raise ValueError("Minimum 2000 observations recommended")
        statistic, p_value, _, _ = jarque_bera(data)
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(['Jarque-Bera', p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(['Jarque-Bera', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Jarque-Bera test:', e)

    # 10. Skewtest
    try:
        statistic, p_value = stats.skewtest(data)
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(['Skewtest', p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(['Skewtest', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Skewtest test:', e)

    # 11. Kurtosistest
    try:
        statistic, p_value = stats.kurtosistest(data)
        conclusion = "gaussian" if p_value >= a_level else "non-gaussian"
        results.append(['Kurtosistest', p_level, a_level, p_value, p_value >= a_level, statistic, None, None, conclusion])
    except Exception as e:
        results.append(['Kurtosistest', p_level, a_level, None, None, None, None, None, "Error in print"])
        print('Kurtosistest test:', e)

    df = pd.DataFrame(
        results,
        columns=['Test', 'Confidence Level', 'Alpha','P-Value', 'P-Value >= Alpha', 'Statistic','Critical Value', 'Stat < Critical', 'Conclusion'])
    return df


# normality_tests(residuals_train)


#vif_data = pd.DataFrame()
#vif_data["feature"] = X_train.columns
#vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
#print("\nMulticollinearity check (VIF > 10 is a problem)::")
#print(vif_data.sort_values("VIF", ascending=False))


'''
# The Heteroscedasticity Test
X_train_const = add_constant(X_train)
if X_train_const.shape[1] < 2:
    raise ValueError("After removing the features, there are less than 2 variables left. The test is not possible.")
bp_test = het_breuschpagan(residuals_train, X_train_const)
print(f"\nBreusch-Pagan Test: p-value = {bp_test[1]:.10f}")
if bp_test[1] < 0.05:
    print("Heteroskedasticity is present (p < 0.05)")
else:
    print("No heteroskedasticity was detected")
'''


'''
# The Darbin-Watson autocorrelation test
dw_test = durbin_watson(residuals_train)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}")
if dw_test < 1.5 or dw_test > 2.5:
    print("Autocorrelation of residues detected")
else:
    print("No residue autocorrelation was detected")
'''

