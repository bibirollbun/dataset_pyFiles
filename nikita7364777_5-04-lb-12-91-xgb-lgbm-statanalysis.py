# Base
import os
import glob
import numpy as np
import pandas as pd

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
from lightgbm import LGBMRegressor

#Model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import shap
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
df_train['Episode_Title'] = (df_train['Episode_Title'].str.replace('Episode ', '', regex = False).astype('int32'))
df_test['Episode_Title'] = (df_test['Episode_Title'].str.replace('Episode ', '', regex = False).astype('int32'))
# Checking the result
print(df_train.Episode_Title.unique()), print(df_test.Episode_Title.unique())


df_train.head()


df_train.info()


df_train.describe().T


df_train.describe(exclude = np.number).T


df_train = df_train[df_train['Number_of_Ads'] < 10].reset_index(drop = True)
df_train.shape


cat_col = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
num_col = ['Episode_Title', 'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']


corr = df_train[num_col].corr()
fig, ax = plt.subplots(figsize = (25, 15))
mask = np.triu(np.ones_like(corr, dtype = bool), k = 1)
sns.heatmap(corr, cmap = 'seismic', mask = mask, vmin = -1, vmax = 1, fmt = ".2f", annot=True)
plt.tight_layout()


label_encoder = {column: LabelEncoder() for column in cat_col}
for i in cat_col:
    df_train[i] = label_encoder[i].fit_transform(df_train[i])
    df_test[i] = label_encoder[i].transform(df_test[i])
    df_train[i] = df_train[i].astype('category')
    df_test[i] = df_test[i].astype('category')


df_train.head()


df_train.info()


cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
global_mean = df_train['Listening_Time_minutes'].mean()
target_encodings = {}

for col in cat_cols:
    # Initializing a column with the correct type
    df_train[f'{col}_target_enc'] = np.nan
    df_train[f'{col}_target_enc'] = df_train[f'{col}_target_enc'].astype('float32')
    
    # We save the original type and convert it to a string for mapping
    original_dtype = df_train[col].dtype
    df_train[col] = df_train[col].astype(str)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df_train):
        train_part = df_train.iloc[train_idx]
        val_part = df_train.iloc[val_idx]
        
        # Calculating the average values
        enc_map = train_part.groupby(col)['Listening_Time_minutes'].mean().to_dict()
        target_encodings[col] = enc_map
        
        # Mapping and filling in gaps
        val_encoded = val_part[col].map(enc_map).fillna(global_mean)
        df_train.loc[val_idx, f'{col}_target_enc'] = val_encoded.astype('float32')
    
    # Returning the original data type
    df_train[col] = df_train[col].astype(original_dtype)
    
    # Fill in the remaining gaps
    df_train[f'{col}_target_enc'] = df_train[f'{col}_target_enc'].fillna(global_mean)


# For Podcast_Name
df_train['podcast_episode_count'] = df_train.groupby('Podcast_Name')['Episode_Title'].transform('count')
df_train['podcast_avg_ads'] = df_train.groupby('Podcast_Name')['Number_of_Ads'].transform('mean')
# For Genre
df_train['genre_avg_length'] = df_train.groupby('Genre')['Episode_Length_minutes'].transform('mean')


# Combination of publication time and day of the week
df_train['Day_Time_Interaction'] = (df_train['Publication_Day'].astype(str) + '_' + df_train['Publication_Time'].astype(str)).astype('category')
# The interaction of genre and sentiment
df_train['Genre_Sentiment_Interaction'] = (df_train['Genre'].astype(str) + '_' + df_train['Episode_Sentiment'].astype(str)).astype('category')


# Binning the duration of episodes
bins = [0, 30, 60, 90, 120, 150]
labels = ['very_short', 'short', 'medium', 'long', 'very_long']
df_train['Episode_Length_bin'] = pd.cut(df_train['Episode_Length_minutes'], bins = bins, labels = labels).astype('category')
# Popularity logarithm
for col in ['Guest_Popularity_percentage']:
    df_train[f'log_{col}'] = np.log1p(df_train[col])


# Converting categories to codes
for col in cat_col + ['Day_Time_Interaction', 'Genre_Sentiment_Interaction']:
    # Explicitly reducing the column to the categorical type
    df_train[col] = df_train[col].astype('category')
    df_train[col] = df_train[col].cat.codes

# Deleting the original categorical columns (optional)
df_train.drop(cat_cols, axis = 1, inplace = True)


df_train.info()


df_train = df_train.drop(columns = ['podcast_episode_count', 'podcast_avg_ads', 'genre_avg_length'])


# Using the saved encodings
for col in cat_cols:
    # Convert to a string for compatibility with dictionary keys
    original_dtype = df_test[col].dtype
    df_test[col] = df_test[col].astype(str)
    # Mapping values
    df_test[f'{col}_target_enc'] = (df_test[col].map(target_encodings.get(col, {})).fillna(global_mean))
    # Returning the original type
    df_test[col] = df_test[col].astype(original_dtype)
    # Type Conversion
    df_test[f'{col}_target_enc'] = df_test[f'{col}_target_enc'].astype('float32')


# For Podcast_Name
df_test['podcast_episode_count'] = df_test.groupby('Podcast_Name')['Episode_Title'].transform('count')
df_test['podcast_avg_ads'] = df_test.groupby('Podcast_Name')['Number_of_Ads'].transform('mean')
# For Genre
df_test['genre_avg_length'] = df_test.groupby('Genre')['Episode_Length_minutes'].transform('mean')


# Combination of publication time and day of the week
df_test['Day_Time_Interaction'] = (df_test['Publication_Day'].astype(str) + '_' + df_test['Publication_Time'].astype(str)).astype('category')
# The interaction of genre and sentiment
df_test['Genre_Sentiment_Interaction'] = (df_test['Genre'].astype(str) + '_' + df_test['Episode_Sentiment'].astype(str)).astype('category')


# Binning the duration of episodes
bins = [0, 30, 60, 90, 120, 150]
labels = ['very_short', 'short', 'medium', 'long', 'very_long']
df_test['Episode_Length_bin'] = pd.cut(df_test['Episode_Length_minutes'], bins = bins, labels = labels).astype('category')
# Popularity logarithm
for col in ['Guest_Popularity_percentage']:
    df_test[f'log_{col}'] = np.log1p(df_test[col])


# Converting categories to codes
for col in cat_col + ['Day_Time_Interaction', 'Genre_Sentiment_Interaction']:
    # Explicitly reducing the column to the categorical type
    df_test[col] = df_test[col].astype('category')
    df_test[col] = df_test[col].cat.codes

# Deleting the original categorical columns (optional)
df_test.drop(cat_cols, axis = 1, inplace = True)


df_test.info()


df_test = df_test.drop(columns = ['podcast_episode_count', 'podcast_avg_ads', 'genre_avg_length'])


Y = df_train['Listening_Time_minutes']
X = df_train.drop(columns = ['Listening_Time_minutes'])


scaler = StandardScaler().set_output(transform = "pandas")

Episode_Length_bin_train = df_train['Episode_Length_bin']
X_scaled = scaler.fit_transform(X.drop(columns = ['Episode_Length_bin']))
X_scaled = pd.concat([X_scaled, Episode_Length_bin_train.reset_index(drop=True)], axis=1)

Episode_Length_bin_test = df_test['Episode_Length_bin']
X_test_scaled = scaler.transform(df_test.drop(columns = ['Episode_Length_bin']))
X_test_scaled = pd.concat([X_test_scaled, Episode_Length_bin_test.reset_index(drop=True)], axis=1)


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
        'device': 'cuda',
        'seed': 42
    }

    model = XGBRegressor(**xgb_params, enable_categorical = True)

    # Cross-validation configuration
    cv = ShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 42)
    rmse_scores = []

    for train_idx, val_idx in cv.split(X_scaled, Y):
        # Data separation
        X_fold_train, X_fold_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_fold_train, y_fold_val = Y.iloc[train_idx], Y.iloc[val_idx]
        # Learning and prediction
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
        y_pred = model.predict(X_fold_val)
        # RMSE Calculation
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        rmse_scores.append(fold_rmse)

    # We return the average RMSE for all folds
    return np.mean(rmse_scores)


#sampler = TPESampler(seed=42)
#study_1 = optuna.create_study(direction="minimize", sampler=sampler)
#study_1.optimize(objective, n_trials = 25)


import pickle
# A function for saving a study
def save_study(study, filename):
    with open(filename, "wb") as f:
        pickle.dump(study, f)

# Function for uploading a study
def load_study(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


#save_study(study_1, "study_1_XGB_Optuna_LongLoading.pkl")


#study_1 = load_study('study_1_XGB_Optuna_LongLoading.pkl')


Xgb_params_1 = {'n_estimators': 2700, 'max_depth': 12,
                'learning_rate': 0.014424752059091096, 'reg_alpha': 0.027431215270868243,
                'subsample': 0.8425579407020668, 'gamma': 0.00039063308776607114,
                'colsample_bytree': 0.48653880802887245, 'min_child_weight': 1, 'reg_lambda': 0.0018237654881012358} # model on aprior - Y


Xgb_params_2 = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                       'tree_method': 'hist', 'device': 'cuda', 'seed': 42, 'enable_categorical': True}

model_1 = XGBRegressor(**Xgb_params_1, **Xgb_params_2)
model_1.fit(X_scaled, Y)


y_pred_train_XGB = model_1.predict(X_scaled)


y_pred_test_XGB = model_1.predict(X_test_scaled)


print(f"RMSE on train data - {round(np.sqrt(mean_squared_error(Y, y_pred_train_XGB)), 2)} minutes")


feature_names = X_scaled.columns
importances = model_1.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y=sorted_features[-25:], width=sorted_importances[-25:],color='#3498db',edgecolor='black')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("XGB Feature Importance (ShuffleSplit (5, 0.3, 42))", fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)

plt.tight_layout()
plt.show()


def objective(trial):
    
    lgbm_params = {'n_estimators': trial.suggest_int("n_estimators", 3000, 7000, step = 100),
                   'num_leaves': trial.suggest_int("num_leaves", 15, 255, step = 10),
                   'max_depth': trial.suggest_int("max_depth", 3, 12, step = 2),
                   'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                   'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
                   'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
                   'subsample': trial.suggest_float("subsample", 0.6, 0.95),
                   'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.95),
                   'min_child_weight': trial.suggest_float("min_child_weight", 1e-6, 1e-1, log=True),
                   'bagging_freq': trial.suggest_int("bagging_freq", 1, 5),
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'device': 'cpu',
        'verbose': -1}

    model = LGBMRegressor(**lgbm_params)
    # Cross-validation configuration
    cv = ShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 42)
    rmse_scores = []

    for train_idx, val_idx in cv.split(X_scaled, Y):
        # Data separation
        X_fold_train, X_fold_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_fold_train, y_fold_val = Y.iloc[train_idx], Y.iloc[val_idx]
        # Learning and prediction
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)])
        y_pred = model.predict(X_fold_val)
        # RMSE Calculation
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        rmse_scores.append(fold_rmse)

    # We return the average RMSE for all folds
    return np.mean(rmse_scores)


#sampler = TPESampler(seed=42)
#study_2 = optuna.create_study(direction="minimize", sampler=sampler)
#study_2.optimize(objective, n_trials = 25)


LGBM_params_1 = {'n_estimators': 3400, 'num_leaves': 185, 
                 'max_depth': 9, 'learning_rate': 0.024566974547738343, 
                 'reg_alpha': 0.0071587286315002, 'reg_lambda': 0.00029442723591496795, 
                 'subsample': 0.7829564902836978, 'colsample_bytree': 0.7496393564254924, 
                 'min_child_weight': 1.3399717231179736e-06, 'bagging_freq': 1}  # model on aprior - Y

LGBM_params_2 = {'objective': 'regression', 'metric': 'rmse',
                 'random_state': 42, 'device': 'cpu', 'verbose': -1}

model_2 = LGBMRegressor(**LGBM_params_1, **LGBM_params_2)
model_2.fit(X_scaled, Y)


y_pred_train_LGBM = model_2.predict(X_scaled)


y_pred_test_LGBM = model_2.predict(X_test_scaled)


print(f"RMSE on train data - {round(np.sqrt(mean_squared_error(Y, y_pred_train_LGBM)), 2)} minutes")


feature_names = X_scaled.columns
importances = model_2.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y=sorted_features[-25:], width=sorted_importances[-25:],color='#3498db',edgecolor='black')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("LGBM Feature Importance (ShuffleSplit (5, 0.3, 42))", fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)

plt.tight_layout()
plt.show()


estimators = [('XGB', model_1),
              ('LGBM', model_2)]
SR = StackingRegressor(estimators = estimators, final_estimator = LinearRegression())
SR = SR.fit(X_scaled, Y)

y_pred_train_SR = SR.predict(X_scaled)
y_pred_test_SR = SR.predict(X_test_scaled)

print(f"RMSE on train data - {round(np.sqrt(mean_squared_error(Y, y_pred_train_SR)), 2)} minutes")


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


AIC = -2 * log_likelihood + 2 * k
print(f"AIC: {AIC:.2f}")


# Ğ�Ğ¾Ñ€Ğ¼Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ¾Ñ�Ñ‚Ğ°Ñ‚ĞºĞ¾Ğ²
residuals_normalized = (residuals_train - np.mean(residuals_train)) / np.std(residuals_train)

# Ğ¢ĞµÑ�Ñ‚ CvM
result = cramervonmises(residuals_normalized, 'norm')
print(f"CvM statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")


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


normality_tests(residuals_train)


vif_data = pd.DataFrame()
vif_data["feature"] = X_scaled.drop(columns = ['Episode_Length_bin']).columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled.drop(columns = ['Episode_Length_bin']).values, i) for i in range(X_scaled.drop(columns = ['Episode_Length_bin']).shape[1])]
print("\nMulticollinearity check (VIF > 10 is a problem)::")
print(vif_data.sort_values("VIF", ascending=False))


# The Heteroscedasticity Test
X_train_const = add_constant(X_scaled.drop(columns = ['Episode_Length_bin']))
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


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub.head()


submission_df = pd.DataFrame({'id': sub['id'],
                              'Listening_Time_minutes': y_pred_test_SR})

submission_df.to_csv('submission.csv', index = False)

