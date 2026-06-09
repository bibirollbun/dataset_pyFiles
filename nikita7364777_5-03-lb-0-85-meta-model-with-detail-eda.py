# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#Statistics
from scipy.stats import skew
from scipy.stats import randint
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
#---
import category_encoders as ce
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

#Model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

#Stacking
from sklearn.ensemble import StackingClassifier


path = '/kaggle/input/playground-series-s5e3/'


df_train = pd.read_csv(path + 'train.csv').drop(columns = ['id'])
df_train.head()


df_origin = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')
df_origin.head()


# Check the name - winddirection (we must fix)
# humidity and cloud in df_train has type float64 (we must fix)
# Check the value of rainfall (yes, no) {we must fix, yes - 1, no - 0}
df_origin.info()


df_origin.columns = df_origin.columns.str.replace(' ', '')
df_origin['rainfall'] = df_origin['rainfall'].map({'no': 0, 'yes': 1})
df_origin['humidity']=df_origin['humidity'].astype('float64')
df_origin['cloud']=df_origin['cloud'].astype('float64')
df_train = pd.concat([df_train, df_origin], axis = 0, ignore_index = True)
pipe_data = df_train.copy()


df_train = df_train.drop_duplicates()


df_train.head()


df_train.describe().T


df_train.info()


df_train['rainfall'].value_counts()


correlation_matrix = df_train.corr(numeric_only = True)
mask = np.triu(np.ones_like(correlation_matrix, dtype = bool), k=1)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='seismic', fmt=".2f", linewidths=.1, mask = mask)
plt.title('Correlation matrix numerical atributes')
plt.show()


plt.figure(figsize=(15, 20))
numeric_columns = df_train.select_dtypes(include=np.number).columns

for i, column in enumerate(numeric_columns, 1):
    plt.subplot(5, 3, i)
    sns.histplot(
        data=df_train, 
        x=column, 
        kde=True, hue='rainfall'
    )
    plt.title(f'Distribution of {column}')
    sns.color_palette("Spectral", as_cmap=True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 10))
sns.boxplot(data=df_train.drop(columns=['day', 'pressure', 'winddirection', 'rainfall', 'sunshine']), orient='h')
plt.title('Distribution (Boxplot)')
plt.show()


plt.figure(figsize=(5, 5))
sns.boxplot(data=df_train[['winddirection']], orient='h')
plt.title('Distribution winddirection (Boxplot)')
plt.show()


plt.figure(figsize=(5, 5))
sns.boxplot(data=df_train[['pressure']], orient='h')
plt.title('Distribution pressure (Boxplot)')
plt.show()


plt.figure(figsize=(5, 5))
sns.boxplot(data=df_train[['sunshine']], orient='h')
plt.title('Distribution sunshine (Boxplot)')
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(data=df_train[['maxtemp', 'mintemp', 'dewpoint']])
plt.title('Comparison of temperature distributions')
plt.show()


plt.figure(figsize=(12, 6))
df_train['rainfall'].value_counts().plot(kind='bar')
plt.title('Precipitation distribution (0 - no, 1 - yes)')
plt.xticks(rotation=0)
plt.show()


plt.figure(figsize=(10, 6))
sns.ecdfplot(data=df_train, x='pressure', complementary=True)
plt.title('CDF pressure')
plt.show()


g = sns.pairplot(df_train[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
                           'dewpoint', 'humidity', 'cloud', 'sunshine', 
                           'winddirection', 'windspeed', 'rainfall']])

#g.fig.savefig(
#    "pairplot.png",      
#    dpi=300,                
#    bbox_inches="tight",  
#    facecolor="white"    
#)

plt.show()


print(df_train['rainfall'].value_counts(normalize=True))


mutual_df = df_train.drop(columns = ['rainfall']).fillna(0)
y = df_train['rainfall']
mutual_info = mutual_info_regression(mutual_df, y, random_state= 42)
mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df.columns
pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["Mutual_Info_Regression"] ).style.background_gradient("rainbow")


# Drop low-influence features based on Mutual Information scores
df_train = df_train.drop(['windspeed'], axis=1)
# Apply IQR-based outlier clipping for winddirection
Q1 = df_train['winddirection'].quantile(0.25)
Q3 = df_train['winddirection'].quantile(0.75)
IQR = Q3 - Q1
df_train['winddirection'] = np.clip(df_train['winddirection'], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
# Drop one of the highly correlated features
df_train = df_train.drop(['dewpoint'], axis=1)


df_train.head()


# 1. Temperature Range
df_train['temp_range'] = df_train['maxtemp'] - df_train['mintemp']

# 2. Daily Temperature Variation
df_train['temp_variation'] = df_train[['temparature', 'maxtemp', 'mintemp']].std(axis=1)

# 3. Pressure Change Rate
df_train['pressure_change'] = df_train['pressure'].diff().fillna(0)

# 4. Humidity-Pressure Interaction
df_train['humidity_pressure'] = df_train['humidity'] * df_train['pressure']

# 5. Wind Direction Categories
df_train['wind_quadrant'] = pd.cut(df_train['winddirection'],
                                   bins=[0, 90, 180, 270, 360],
                                   labels=['NE', 'SE', 'SW', 'NW'])

# 6. Sunshine Duration Categories
df_train['sunshine_category'] = pd.cut(df_train['sunshine'],
                                       bins=[0, 4, 8, 12],
                                       labels=['low', 'medium', 'high'])

# 7. Cloud-Humidity Interaction
df_train['cloud_humidity'] = df_train['cloud'] * df_train['humidity']

# 8. Rolling Average of Temperature
df_train['temp_rolling_avg'] = df_train['temparature'].rolling(window=3).mean().fillna(df_train['temparature'])

# 9. Seasonal Indicators (Create binary flags for seasons)
# Summer: June 1 - August 31 (Day 152-244)
df_train['is_summer'] = np.where(df_train['day'].isin(range(152, 245)), 1, 0)
# Winter: December 1 - February 28/29 (Day 335-365 and 1-59)
df_train['is_winter'] = np.where((df_train['day'].isin(range(335, 366))) | (df_train['day'].isin(range(1, 60))), 1, 0)
# Spring: March 1 - May 31 (Day 60-151)
df_train['is_spring'] = np.where(df_train['day'].isin(range(60, 152)), 1, 0)
# Autumn: September 1 - November 30 (Day 245-334)
df_train['is_autumn'] = np.where(df_train['day'].isin(range(245, 335)), 1, 0)


df_train.head()


df_train.info()


# Fill missing values in 'wind_quadrant' with the mode
wind_quadrant_mode = df_train['wind_quadrant'].mode()[0]
df_train['wind_quadrant'] = df_train['wind_quadrant'].fillna(wind_quadrant_mode)

# Fill missing values in 'sunshine_category' with the mode
sunshine_category_mode = df_train['sunshine_category'].mode()[0]
df_train['sunshine_category'] = df_train['sunshine_category'].fillna(sunshine_category_mode)


df_train.info()


encoder = OneHotEncoder(sparse_output=False)  # Drop first category to avoid multicollinearity
# Apply OneHotEncoding to 'wind_quadrant'
wind_quadrant_encoded = encoder.fit_transform(df_train[['wind_quadrant']])
wind_quadrant_encoded_df = pd.DataFrame(wind_quadrant_encoded,
                                        columns=encoder.get_feature_names_out(['wind_quadrant']))

# Apply OneHotEncoding to 'sunshine_category'
sunshine_category_encoded = encoder.fit_transform(df_train[['sunshine_category']])
sunshine_category_encoded_df = pd.DataFrame(sunshine_category_encoded,
                                            columns=encoder.get_feature_names_out(['sunshine_category']))

# Concatenate the encoded features with the original dataframe
df_train = pd.concat([df_train.drop(['wind_quadrant', 'sunshine_category'], axis=1),
                      wind_quadrant_encoded_df,
                      sunshine_category_encoded_df],axis=1)


df_train.columns


new_columns = ['temp_range', 'temp_variation', 'pressure_change', 'humidity_pressure', 
                'cloud_humidity', 'temp_rolling_avg', 
               'is_summer', 'is_winter', 'is_spring', 'is_autumn', 'wind_quadrant_NE', 'wind_quadrant_NW',
               'wind_quadrant_SE', 'wind_quadrant_SW', 'sunshine_category_high',
               'sunshine_category_low', 'sunshine_category_medium', 'rainfall']


mutual_df = df_train[new_columns].drop(columns = ["rainfall"])
y = df_train['rainfall']
mutual_info = mutual_info_regression(mutual_df, y, random_state= 42)
mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df.columns
pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["Mutual_Info_Regression_New_Columns"] ).style.background_gradient("rainbow")


correlation_matrix = df_train[new_columns].corr(numeric_only = True)
mask = np.triu(np.ones_like(correlation_matrix, dtype = bool), k=1)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='seismic', fmt=".2f", linewidths=.1, mask = mask)
plt.title('Corr matrix with new numerical atributes')
plt.show()


g = sns.pairplot(df_train[new_columns])
#g.fig.savefig(
#    "pairplot_2.png",      
#    dpi=300,                
#    bbox_inches="tight",  
#    facecolor="white")
plt.show()


# Drop low-impact and redundant features
df_train = df_train.drop([
    'wind_quadrant_NW', 
    'wind_quadrant_NE', 
    'pressure_change', 
    'is_summer', 
    'is_autumn', 
    'temp_variation', 
    'wind_quadrant_SE', 
    'sunshine_category_medium',
    'day',
    'is_spring', 
    'is_winter', 
    'wind_quadrant_SW', 
    'sunshine_category_high', 
    'sunshine_category_low',
    'maxtemp',
    'mintemp'], axis=1)


# Checking the result
df_train.head()


df_train.info()


numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                        ('Scaller', StandardScaler())])


y = df_train['rainfall']
df_train = df_train.drop(columns = ["rainfall"])


skew_features = df_train.select_dtypes(exclude=['object']).skew().sort_values(ascending=False)
skew_features = pd.DataFrame({'Skew' : skew_features})
skew_features.style.background_gradient('seismic')


# >1
skewed_features = ['winddirection', 'sunshine']
skewness_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                       ('PowerTransformer', PowerTransformer(method='yeo-johnson', standardize=True))])


preprocessor = ColumnTransformer(remainder=numerical_transformer, transformers=[('skewness_transformer', skewness_transformer, skewed_features)])


pd.DataFrame(preprocessor.fit_transform(df_train))


set_config(display="diagram")
preprocessor


# The maximum accuracy that I was able to obtain with the meta-ensemble was 0.76. 
# Using all the features below. I decided to try to remove some of the signs that 
# I calculated based on empirical formulas and got the accuracy.
class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, add_attributes=True):
        self.add_attributes = add_attributes

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.add_attributes:
            #Copy from Feature Engineering
            X_copy = X.copy()
            # Drop low-influence features based on Mutual Information scores
            X_copy = X_copy.drop(['windspeed'], axis=1)
            # Apply IQR-based outlier clipping for winddirection
            Q1 = X_copy['winddirection'].quantile(0.25)
            Q3 = X_copy['winddirection'].quantile(0.75)
            IQR = Q3 - Q1
            X_copy['winddirection'] = np.clip(X_copy['winddirection'], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            # Drop one of the highly correlated features (sunshine has higher MI score)
            X_copy = X_copy.drop(['dewpoint'], axis=1)
            # 1. Temperature Range
            X_copy['temp_range'] = X_copy['maxtemp'] - X_copy['mintemp']
            # 2. Daily Temperature Variation
            X_copy['temp_variation'] = X_copy[['temparature', 'maxtemp', 'mintemp']].std(axis=1)
            # 3. Pressure Change Rate
            X_copy['pressure_change'] = X_copy['pressure'].diff().fillna(0)
            # 4. Humidity-Pressure Interaction
            X_copy['humidity_pressure'] = X_copy['humidity'] * X_copy['pressure']
            # 5. Wind Direction Categories
            X_copy['wind_quadrant'] = pd.cut(X_copy['winddirection'],
                                            bins=[0, 90, 180, 270, 360],
                                            labels=['NE', 'SE', 'SW', 'NW'])
            # 6. Sunshine Duration Categories
            X_copy['sunshine_category'] = pd.cut(X_copy['sunshine'],
                                                 bins=[0, 4, 8, 12],
                                                 labels=['low', 'medium', 'high'])
            # 7. Cloud-Humidity Interaction
            X_copy['cloud_humidity'] = X_copy['cloud'] * X_copy['humidity']
            # 8. Rolling Average of Temperature
            X_copy['temp_rolling_avg'] = X_copy['temparature'].rolling(window=3).mean().fillna(X_copy['temparature'])
            # 9. Seasonal Indicators (Create binary flags for seasons)
            # Summer: June 1 - August 31 (Day 152-244)
            X_copy['is_summer'] = np.where(X_copy['day'].isin(range(152, 245)), 1, 0)
            # Winter: December 1 - February 28/29 (Day 335-365 and 1-59)
            X_copy['is_winter'] = np.where((X_copy['day'].isin(range(335, 366))) | (X_copy['day'].isin(range(1, 60))), 1, 0)
            # Spring: March 1 - May 31 (Day 60-151)
            X_copy['is_spring'] = np.where(X_copy['day'].isin(range(60, 152)), 1, 0)
            # Autumn: September 1 - November 30 (Day 245-334)
            X_copy['is_autumn'] = np.where(X_copy['day'].isin(range(245, 335)), 1, 0)
            # Fill missing values in 'wind_quadrant' with the mode
            wind_quadrant_mode = X_copy['wind_quadrant'].mode()[0]
            X_copy['wind_quadrant'] = X_copy['wind_quadrant'].fillna(wind_quadrant_mode)
            # Fill missing values in 'sunshine_category' with the mode
            sunshine_category_mode = X_copy['sunshine_category'].mode()[0]
            X_copy['sunshine_category'] = X_copy['sunshine_category'].fillna(sunshine_category_mode)
            encoder = OneHotEncoder(sparse_output=False)
            # Apply OneHotEncoding to 'wind_quadrant'
            wind_quadrant_encoded = encoder.fit_transform(X_copy[['wind_quadrant']])
            wind_quadrant_encoded_df = pd.DataFrame(wind_quadrant_encoded,columns=encoder.get_feature_names_out(['wind_quadrant']))
            # Apply OneHotEncoding to 'sunshine_category'
            sunshine_category_encoded = encoder.fit_transform(X_copy[['sunshine_category']])
            sunshine_category_encoded_df = pd.DataFrame(sunshine_category_encoded, columns=encoder.get_feature_names_out(['sunshine_category']))
            # Concatenate the encoded features with the original dataframe
            X_copy = pd.concat([X_copy.drop(['wind_quadrant', 'sunshine_category'], axis=1),wind_quadrant_encoded_df, sunshine_category_encoded_df],axis=1)
            # Drop low-impact and redundant features
            X_copy = X_copy.drop(['wind_quadrant_NW', 
                                  'wind_quadrant_NE', 
                                  'pressure_change', 
                                  'is_summer', 
                                  'is_autumn', 
                                  'temp_variation', 
                                  'wind_quadrant_SE', 
                                  'sunshine_category_medium'], axis=1)
            return X_copy
        else:
            return X_copy


Creator = FeatureCreator(add_attributes = True)
y = pipe_data.rainfall
pipe_data = pipe_data.drop("rainfall", axis=1)


df_train.info()


def objective(trial):
    xgb_params = dict(
        n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50),
        max_depth = trial.suggest_int("max_depth", 7, 15, step=2),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        reg_alpha = trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        subsample = trial.suggest_float("subsample", 0.5, 0.9),
        gamma = trial.suggest_float("gamma", 1e-3, 1e-1, log=True),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.22, 0.9),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 3),
        reg_lambda = trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        tree_method='hist',
        enable_categorical=False
    )

    XGB_classifier = XGBClassifier(**xgb_params, error_score='raise')
    XGB_pipeline = make_pipeline(preprocessor, XGB_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(XGB_pipeline, df_train, y, scoring ='roc_auc',  cv = ss, error_score='raise')
    score = score.mean()
    return score


'''
sampler = TPESampler(seed=42)
study_1 = optuna.create_study(direction="maximize", sampler=sampler)
study_1.optimize(objective, n_trials = 25)
'''


params_1 = {'n_estimators': 800, 
            'max_depth': 9, 
            'learning_rate': 0.002474170476131244, 
            'reg_alpha': 9.510095724508383e-05, 
            'subsample': 0.6011536883207419, 
            'gamma': 0.010258136378769388, 
            'colsample_bytree': 0.3241789068452381, 
            'min_child_weight': 2, 
            'reg_lambda': 0.0030096337619537937, 
            'objective': 
            'binary:logistic', 
            'eval_metric': 'auc',
            'use_label_encoder': False, 
            'tree_method': 'hist', 
            'enable_categorical': False}
model_1 = XGBClassifier(**params_1)
pipe_XGB_1 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_1', model_1)])


#plot_param_importances(study_1)


# Train the pipeline
pipe_XGB_1.fit(df_train, y)

# Get feature names after preprocessing
feature_names = pipe_XGB_1.named_steps['preprocessor'].get_feature_names_out()

plt.figure(figsize=(12, 8))
ax = plot_importance(
    pipe_XGB_1.named_steps['model_1'],
    importance_type='weight',
    max_num_features=25,
    title='Feature Importance for XGBoost Model',
    grid=False,
    height=0.8)

ax.yaxis.set_ticklabels([feature_names[int(tick.get_text().replace('f', ''))] 
                        for tick in ax.get_yticklabels()])

# Styling
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('F-score Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)
plt.gcf().set_facecolor('#f5f5f5')

plt.tight_layout()
plt.show()


def objective(trial):
    catboost_params = dict(
        iterations=trial.suggest_int("iterations", 100, 1000, step = 100),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        depth=trial.suggest_int("depth", 7, 15, step = 2),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bootstrap_type='Bernoulli',
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        thread_count = -1,
        early_stopping_rounds=100,
        verbose=False,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42
    )

    CatBoost_classifier = CatBoostClassifier(**catboost_params)
    CatBoost_pipeline = make_pipeline(preprocessor, CatBoost_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(CatBoost_pipeline, df_train, y, scoring ='roc_auc',  cv = ss)
    score = score.mean()
    return score


'''
sampler = TPESampler(seed=42)
study_2 = optuna.create_study(direction="maximize", sampler=sampler)
study_2.optimize(objective, n_trials = 25)
'''


params_2 = {'iterations': 500, 'learning_rate': 0.00473777238603483, 'depth': 7, 'l2_leaf_reg': 0.7504651444859775, 'subsample': 0.7339530232151068,
            'verbose': False, 'loss_function': 
            'Logloss', 'eval_metric': 'AUC', 'random_seed': 42}
model_2 = CatBoostClassifier(**params_2)
pipe_CBC_2 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_2', model_2)])


#plot_param_importances(study_2)


def objective(trial):
    gbm_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step = 50),
        max_depth=trial.suggest_int("max_depth", 6, 16, step = 2),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        subsample=trial.suggest_float("subsample", 0.40, 0.90),
        min_samples_split=trial.suggest_float("min_samples_split", 0.3, 0.9),
    )

    GBM_classifier = GradientBoostingClassifier(**gbm_params)
    GBM_pipeline = make_pipeline(preprocessor, GBM_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(GBM_pipeline, df_train, y, scoring ='roc_auc',  cv = ss)
    score = score.mean()
    return score


'''
sampler = TPESampler(seed=42)
study_3 = optuna.create_study(direction="maximize", sampler=sampler)
study_3.optimize(objective, n_trials = 25)
'''


params_3 = {'n_estimators': 900, 'max_depth': 10, 'learning_rate': 0.007088834460121464, 'subsample': 0.8946233875078891, 'min_samples_split': 0.6337584186236781}
model_3 = GradientBoostingClassifier(**params_3)
pipe_GBM_3 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_3', model_3)])


#plot_param_importances(study_3)


def objective(trial):
    lgbm_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=10),
        max_depth=trial.suggest_int("max_depth", 6, 16, step=2),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        subsample=trial.suggest_float("subsample", 0.4, 0.9),
        max_bin=trial.suggest_int("max_bin", 100, 300, step=10),
        feature_fraction=trial.suggest_float("feature_fraction", 0.1, 0.5),
        num_leaves=trial.suggest_int("num_leaves", 20, 100, step=10),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 50, step=10),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1e-1, log = True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1e-1, log = True),
        objective='binary',
        verbosity = -1)

    LGBM_classifier = LGBMClassifier(**lgbm_params)
    LGBM_pipeline = make_pipeline(preprocessor, LGBM_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(LGBM_pipeline, df_train, y, scoring ='roc_auc',  cv = ss)
    score = score.mean()
    return score


'''
sampler = TPESampler(seed=42)
study_4 = optuna.create_study(direction="maximize", sampler=sampler)
study_4.optimize(objective, n_trials = 25)
'''


params_4 = {'n_estimators': 550, 'max_depth': 8, 'learning_rate': 0.006650703798778187, 
            'subsample': 0.6464532492548499, 'max_bin': 170, 'feature_fraction': 0.32759213385998276, 
            'num_leaves': 70, 'min_child_samples': 30, 'reg_alpha': 0.002479913114800354, 'reg_lambda': 0.0024819794938205353, 
            'objective':'binary', 'verbosity': -1}
model_4 = LGBMClassifier(**params_4)
pipe_LGBM_4 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_4', model_4)])


#plot_param_importances(study_4)


def objective(trial):
    laso_params = dict(
        penalty='l1',
        solver='liblinear',
        C=trial.suggest_float("C", 1e-4, 1, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 1000))

    Lasso_classifier = LogisticRegression(**laso_params)
    Lasso_pipeline = make_pipeline(preprocessor, Lasso_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(Lasso_pipeline, df_train, y, scoring ='roc_auc',  cv = ss)
    score = score.mean()
    return score


'''
sampler = TPESampler(seed=42)
study_5 = optuna.create_study(direction="maximize", sampler=sampler)
study_5.optimize(objective, n_trials = 25)
'''


params_5 = {'C': 0.11363180493902578, 'max_iter': 531, 
            'penalty':'l1', 'solver': 'liblinear'}
model_5 = LogisticRegression(**params_5)
pipe_Lasso_5 = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model_5', model_5)])


#plot_param_importances(study_5)


def objective(trial):
    ridge_params = dict(
        penalty='l2',
        solver='liblinear',
        C=trial.suggest_float("C", 1e-4, 1.0, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 1000))

    Ridge_classifier = LogisticRegression(**ridge_params)
    Ridge_pipeline = make_pipeline(preprocessor, Ridge_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(Ridge_pipeline, df_train, y, scoring ='roc_auc',  cv = ss)
    score = score.mean()
    return score


'''
sampler = TPESampler(seed=42)
study_6 = optuna.create_study(direction="maximize", sampler=sampler)
study_6.optimize(objective, n_trials = 25)
'''


params_6 = {'C': 0.015496476581234079, 'max_iter': 111, 
            'penalty':'l2', 'solver': 'liblinear'}
model_6 = LogisticRegression(**params_6)
pipe_Ridge_6 = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model_6', model_6)])


#plot_param_importances(study_6)


def objective(trial):
    elastic_params = dict(
        penalty='elasticnet',
        solver='saga',
        C=trial.suggest_float("C", 1e-4, 1.0, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        max_iter=trial.suggest_int("max_iter", 100, 1000))

    ElasticNet_classifier = LogisticRegression(**elastic_params)
    ElasticNet_pipeline = make_pipeline(preprocessor, ElasticNet_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(ElasticNet_pipeline, df_train, y, scoring ='roc_auc',  cv = ss)
    score = score.mean()
    return score


'''
sampler = TPESampler(seed=42)
study_7 = optuna.create_study(direction="maximize", sampler=sampler)
study_7.optimize(objective, n_trials = 25)
'''


params_7 = {'C': 0.018091728041976275, 'l1_ratio': 0.15962761036738962, 'max_iter': 337,
            'penalty':'elasticnet', 'solver': 'saga'}
model_7 = LogisticRegression(**params_7)
pipe_ElasticNet_7 = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model_7', model_7)])


#plot_param_importances(study_7)


estimators = [
    ("Elasticnet", pipe_ElasticNet_7),
    ("Lasso", pipe_Lasso_5),
    ("Ridge", pipe_Ridge_6),
    ("pipe_xgb", pipe_XGB_1),
    ("pipe_gbm", pipe_GBM_3),
    ("pipe_lgbm", pipe_LGBM_4),
    ("pipe_catboost", pipe_CBC_2)]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator = LogisticRegression(C = 0.01))


final_stack = Pipeline(steps=[('stacking_classifier', stacking_classifier)])


'''
grid_params = {'stacking_classifier__final_estimator__C': [0.0001, 0.001, 0.01, 1, 10]}
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
stack_search = GridSearchCV(final_stack, param_grid = grid_params, scoring= 'roc_auc', cv = ss, n_jobs = -1)
stack_search.fit(df_train, y)
'''


#stack_search.best_params_


df_test = pd.read_csv(path + 'test.csv')
pipe_test = df_test.copy()


pipe_test = Creator.fit_transform(pipe_test)


pipe_test.head()


pipe_test = pipe_test.drop(columns = ['id', 'day', 'maxtemp', 'mintemp', 'is_winter', 'is_spring', 'wind_quadrant_SW', 'sunshine_category_high', 'sunshine_category_low'])


pipe_test.head()


df_train.head()


df_train.shape[1] == pipe_test.shape[1]


final_stack.fit(df_train, y)
pred_final_0 = final_stack.predict_proba(df_train)[:, 1]
score = roc_auc_score(y, pred_final_0)
print(f"Score on Train Data (full dataset) is - {score}")

#IsReg = IsotonicRegression(out_of_bounds="clip")
#IsReg.fit(pred_final_0, y)
pred_final_1 = final_stack.predict_proba(pipe_test)[:, 1]
#pred_final_2 = IsReg.transform(pred_final_1)


sub = pd.read_csv(path + 'sample_submission.csv')
output = pd.DataFrame({'id': sub.id, 'rainfall': pred_final_1})
output.to_csv('submission.csv', index = False)


# Ğ“Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ğ° Ñ� Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸ĞµĞ¼ seaborn (Ğ±Ğ¾Ğ»ĞµĞµ Ñ�Ñ‚Ğ¸Ğ»ÑŒĞ½Ğ°Ñ�)
plt.figure(figsize=(10, 6))
sns.histplot(output['rainfall'], bins=30, kde=True, color='green')
plt.title('Ğ“Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ğ° Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� rainfall Ñ� KDE')
plt.xlabel('Ğ—Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� rainfall')
plt.ylabel('Ğ§Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ°')
plt.grid(True)
plt.show()

