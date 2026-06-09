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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
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
sns.boxplot(data=df_train.drop(columns=['day', 'pressure', 'winddirection', 'rainfall']), orient='h')
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


# 1. dew_humidity - Product of dew point and humidity.
df_train['dew_humidity'] = df_train['dewpoint']*df_train['humidity']

# 2. temp_gap - Difference between daily maximum and minimum temperature.
df_train['temp_gap'] = df_train['maxtemp'] - df_train['mintemp']

# 3. wind_speeddirection - Interaction between wind speed and direction.
df_train['wind_speeddirection'] = df_train['windspeed']*df_train['winddirection']

# 4. cloud_windspeed - Combined effect of cloud cover and wind speed.
df_train['cloud_windspeed'] = df_train['cloud']*df_train['windspeed']

# 5. cloud_to_humidity - Ratio of cloud cover to humidity.
df_train['cloud_to_humidity'] = df_train['cloud']/df_train['humidity']

# 6. temp_to_humidity - Ratio of cloud cover to humidity (potential naming inconsistency).
df_train['temp_to_humidity'] = df_train['cloud']/df_train['humidity']

# 7. temp_to_sunshine - Sunshine duration normalized by temperature.
df_train['temp_to_sunshine'] = df_train['sunshine']/df_train['temparature']

# 8. month - Calendar month derived from day-of-year.
df_train['month'] = pd.cut(df_train['day'], bins=12, labels=range(1, 13)).astype('int')

# 9. temp_previous_day - Lagged temperature from preceding day.
df_train['temp_previous_day'] = df_train['temparature'].shift(1).fillna(df_train['temparature'].mode()[0])

# 10. humidity_previous_day - Lagged humidity from preceding day.
df_train['humidity_previous_day'] = df_train['humidity'].shift(1).fillna(df_train['humidity'].mode()[0])

# 11. pressure_previous_day - Lagged atmospheric pressure from preceding day.
df_train['pressure_previous_day'] = df_train['pressure'].shift(1).fillna(df_train['pressure'].mode()[0])

df_train = df_train.drop(columns=['maxtemp', 'mintemp'])


df_train.head()


df_train.info()


df_train.columns


new_columns = ['dew_humidity','temp_gap', 'wind_speeddirection', 'cloud_windspeed',
               'cloud_to_humidity', 'temp_to_humidity', 'temp_to_sunshine', 'month',
               'temp_previous_day', 'humidity_previous_day', 'pressure_previous_day', 'rainfall']


mutual_df = df_train[new_columns].drop(columns = ["rainfall"]).fillna(0)
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
g.fig.savefig(
    "pairplot_2.png",      
    dpi=300,                
    bbox_inches="tight",  
    facecolor="white")
plt.show()


numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                        ('Scaller', StandardScaler())])


y = df_train['rainfall']
df_train = df_train.drop(columns = ["rainfall"])


skew_features = df_train.select_dtypes(exclude=['object']).skew().sort_values(ascending=False)
skew_features = pd.DataFrame({'Skew' : skew_features})
skew_features.style.background_gradient('seismic')


# >1
skewed_features = ['wind_speeddirection', 'temp_to_sunshine']
skewness_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                       ('PowerTransformer', PowerTransformer(method='yeo-johnson', standardize=True))])


preprocessor = ColumnTransformer(remainder=numerical_transformer, transformers=[('skewness_transformer', skewness_transformer, skewed_features)])


pd.DataFrame(preprocessor.fit_transform(df_train))


set_config(display="diagram")
preprocessor


class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, add_attributes=True):
        self.add_attributes = add_attributes

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.add_attributes:
            #Copy from Feature Engineering
            X_copy = X.copy()
            # 1. dew_humidity - Product of dew point and humidity.
            X_copy['dew_humidity'] = X_copy['dewpoint']*X_copy['humidity']

            # 2. temp_gap - Difference between daily maximum and minimum temperature.
            X_copy['temp_gap'] = X_copy['maxtemp'] - X_copy['mintemp']

            # 3. wind_speeddirection - Interaction between wind speed and direction.
            X_copy['wind_speeddirection'] = X_copy['windspeed']*X_copy['winddirection']

            # 4. cloud_windspeed - Combined effect of cloud cover and wind speed.
            X_copy['cloud_windspeed'] = X_copy['cloud']*X_copy['windspeed']

            # 5. cloud_to_humidity - Ratio of cloud cover to humidity.
            X_copy['cloud_to_humidity'] = X_copy['cloud']/X_copy['humidity']

            # 6. temp_to_humidity - Ratio of cloud cover to humidity.
            X_copy['temp_to_humidity'] = X_copy['cloud']/X_copy['humidity']

            # 7. temp_to_sunshine - Sunshine duration normalized by temperature.
            X_copy['temp_to_sunshine'] = X_copy['sunshine']/X_copy['temparature']

            # 8. month - Calendar month derived from day-of-year.
            X_copy['month'] = pd.cut(X_copy['day'], bins=12, labels=range(1, 13)).astype('int')

            # 9. temp_previous_day - Lagged temperature from preceding day.
            X_copy['temp_previous_day'] = X_copy['temparature'].shift(1).fillna(X_copy['temparature'].mode()[0])

            # 10. humidity_previous_day - Lagged humidity from preceding day.
            X_copy['humidity_previous_day'] = X_copy['humidity'].shift(1).fillna(X_copy['humidity'].mode()[0])

            # 11. pressure_previous_day - Lagged atmospheric pressure from preceding day.
            X_copy['pressure_previous_day'] = X_copy['pressure'].shift(1).fillna(X_copy['pressure'].mode()[0])

            X_copy = X_copy.drop(columns=['maxtemp', 'mintemp'])
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


params_1 = {'n_estimators': 350, 
            'max_depth': 11, 
            'learning_rate': 0.0017849133139355075, 
            'reg_alpha': 0.00011599322714137964, 
            'subsample': 0.833977695405836,
            'gamma': 0.04954472490002859, 
            'colsample_bytree': 0.2896328731616218, 
            'min_child_weight': 2, 
            'reg_lambda': 0.0345746946973553,
            'objective': 'binary:logistic', 
            'eval_metric': 'auc',
            'use_label_encoder': False, 
            'tree_method': 'hist', 
            'enable_categorical': False}
model_1 = XGBClassifier(**params_1)
pipe_XGB_1 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_1', model_1)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_XGB_1, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


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


params_2 = {'iterations': 600, 
            'learning_rate': 0.01529349562212324, 
            'depth': 9, 
            'l2_leaf_reg': 12.265895449515982, 
            'subsample': 0.8723817957242327,
            'verbose': False, 
            'loss_function': 'Logloss', 
            'eval_metric': 'AUC', 
            'random_seed': 42}
model_2 = CatBoostClassifier(**params_2)
pipe_CBC_2 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_2', model_2)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_CBC_2, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


pipe_CBC_2.fit(df_train, y)

trained_model = pipe_CBC_2.named_steps['model_2']

preprocessor = pipe_CBC_2.named_steps['preprocessor']
feature_names = preprocessor.get_feature_names_out()

importances = trained_model.feature_importances_

sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y=sorted_features[-25:], width=sorted_importances[-25:],color='#3498db',edgecolor='black')

plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("CatBoost Feature Importance (Trained in Pipeline)", fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)

plt.tight_layout()
plt.show()


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


params_3 = {'n_estimators': 700, 
            'max_depth': 8, 
            'learning_rate': 0.010968217207529524, 
            'subsample': 0.6733551396716398, 
            'min_samples_split': 0.4109126733153162}
model_3 = GradientBoostingClassifier(**params_3)
pipe_GBM_3 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_3', model_3)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_GBM_3, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


pipe_GBM_3.fit(df_train, y)

trained_model = pipe_GBM_3.named_steps['model_3']

preprocessor = pipe_GBM_3.named_steps['preprocessor']
feature_names = preprocessor.get_feature_names_out()

importances = trained_model.feature_importances_

sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y=sorted_features[-25:], width=sorted_importances[-25:],color='#3498db',edgecolor='black')

plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("GBM Feature Importance (Trained in Pipeline)", fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)

plt.tight_layout()
plt.show()


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


params_4 = {'n_estimators': 330, 
            'max_depth': 10, 
            'learning_rate': 0.0015406257999739722, 
            'subsample': 0.7338962247920462, 
            'max_bin': 270, 
            'feature_fraction': 0.2426815878832284, 
            'num_leaves': 40, 
            'min_child_samples': 20, 
            'reg_alpha': 0.0900213377357564, 
            'reg_lambda': 0.014181565882607015,
            'objective':'binary', 
            'verbosity': -1}
model_4 = LGBMClassifier(**params_4)
pipe_LGBM_4 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_4', model_4)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_LGBM_4, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


pipe_LGBM_4.fit(df_train, y)

trained_model = pipe_LGBM_4.named_steps['model_4']

preprocessor = pipe_LGBM_4.named_steps['preprocessor']
feature_names = preprocessor.get_feature_names_out()

importances = trained_model.feature_importances_

sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y=sorted_features[-25:], width=sorted_importances[-25:],color='#3498db',edgecolor='black')

plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("GBM Feature Importance (Trained in Pipeline)", fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)

plt.tight_layout()
plt.show()


#plot_param_importances(study_4)


def objective(trial):
    rf_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=10),
        max_depth=trial.suggest_int("max_depth", 6, 16, step=2),
        max_features=trial.suggest_float("max_features", 0.1, 0.9),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20, step=2),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10, step=1),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=42,
        n_jobs=-1
    )

    RF_classifier = RandomForestClassifier(**rf_params)
    RF_pipeline = make_pipeline(preprocessor, RF_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(RF_pipeline, df_train, y, scoring='roc_auc', cv=ss)
    return score.mean()


'''
sampler = TPESampler(seed=42)
study_4_1 = optuna.create_study(direction="maximize", sampler=sampler)
study_4_1.optimize(objective, n_trials = 25)
'''


params_4_1 = {'n_estimators': 760, 
              'max_depth': 10, 
              'max_features': 0.24664428465126065, 
              'min_samples_split': 18, 
              'min_samples_leaf': 5, 
              'bootstrap': True,
              'criterion': 'entropy', 
              'class_weight': None}
model_4_1 = RandomForestClassifier()
pipe_RF_4_1 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_4_1', model_4_1)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_RF_4_1, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


def objective(trial):
    hgb_params = dict(
        max_iter=trial.suggest_int("max_iter", 100, 1000, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 100, step=10),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True),
        max_bins=trial.suggest_int("max_bins", 100, 255, step=10),
        early_stopping=trial.suggest_categorical("early_stopping", [False]),
        random_state=42,
        verbose=0
    )

    HGB_classifier = HistGradientBoostingClassifier(**hgb_params)
    HGB_pipeline = make_pipeline(preprocessor, HGB_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(HGB_pipeline, df_train, y, scoring='roc_auc', cv=ss)
    return score.mean()


'''
sampler = TPESampler(seed=42)
study_4_2 = optuna.create_study(direction="maximize", sampler=sampler)
study_4_2.optimize(objective, n_trials = 25)
'''


params_4_2 = {'max_iter': 350, 
              'max_depth': 3, 
              'learning_rate': 0.008449949805975113, 
              'min_samples_leaf': 10, 
              'l2_regularization': 0.0005284639638893639, 
              'max_bins': 150, 
              'early_stopping': False}
model_4_2 = HistGradientBoostingClassifier()
pipe_HGBM_4_2 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model_4_2', model_4_2)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_HGBM_4_2, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


def objective(trial):
    et_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
        max_depth=trial.suggest_int("max_depth", 6, 16, step=2),
        max_features=trial.suggest_float("max_features", 0.1, 0.9),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20, step=2),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10, step=1),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        ccp_alpha=trial.suggest_float("ccp_alpha", 1e-5, 0.1, log=True),
        random_state=42,
        n_jobs=-1
    )

    ET_classifier = ExtraTreesClassifier(**et_params)
    ET_pipeline = make_pipeline(preprocessor, ET_classifier)
    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(ET_pipeline, df_train, y, scoring='roc_auc', cv=ss)
    return score.mean()


'''
sampler = TPESampler(seed=42)
study_4_3 = optuna.create_study(direction="maximize", sampler=sampler)
study_4_3.optimize(objective, n_trials = 25)
'''


params_4_3 = {'n_estimators': 600, 
              'max_depth': 8, 
              'max_features': 0.8756677022116469, 
              'min_samples_split': 16, 
              'min_samples_leaf': 10, 
              'bootstrap': True, 
              'criterion': 'gini', 
              'class_weight': None, 
              'ccp_alpha': 0.0002001342062287998}
model_4_3 = ExtraTreesClassifier()
pipe_ETC_4_3 = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model_4_3', model_4_3)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_ETC_4_3, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


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


params_5 = {'C': 0.05443738050058162, 
            'max_iter': 627, 
            'penalty':'l1', 
            'solver': 'liblinear'}
model_5 = LogisticRegression(**params_5)
pipe_Lasso_5 = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model_5', model_5)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_Lasso_5, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


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


params_6 = {'C': 0.019583837264683768, 
            'max_iter': 695, 
            'penalty':'l2', 
            'solver': 'liblinear'}
model_6 = LogisticRegression(**params_6)
pipe_Ridge_6 = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model_6', model_6)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_Ridge_6, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


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


params_7 = {'C': 0.0270651855955485, 
            'l1_ratio': 0.40095596968204394, 
            'max_iter': 518,
            'penalty':'elasticnet', 
            'solver': 'saga'}
model_7 = LogisticRegression(**params_7)
pipe_ElasticNet_7 = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model_7', model_7)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_ElasticNet_7, df_train, y, 
                        scoring='roc_auc', cv=cv, n_jobs=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), scores, marker='o', linestyle='--', color='b')
plt.axhline(y=scores.mean(), color='r', linestyle='-', 
            label=f'Mean: {scores.mean():.4f}')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.title('Cross-validation fold estimates', fontsize=14)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


#plot_param_importances(study_7)


estimators = [
    ("Elasticnet Classifier", pipe_ElasticNet_7),
    ("Lasso Classifier", pipe_Lasso_5),
    ("Ridge Classifier", pipe_Ridge_6),
    ("XGBoostClassifier", pipe_XGB_1),
    ("GBMClassifier", pipe_GBM_3),
    ("LGBMClassifier", pipe_LGBM_4),
    ("CatBoostClassifier", pipe_CBC_2),
    ('RandomForestClassifier', pipe_RF_4_1),
    ('HistGradientBoostingClassifier', pipe_HGBM_4_2),
    ('ExtraTreesClassifier', pipe_ETC_4_3),]


plt.figure(figsize=(12, 7))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

colors = plt.cm.tab10(range(len(estimators)))

for (name, model), color in zip(estimators, colors):
    scores = cross_val_score(model, df_train, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    plt.plot(range(1, 6), scores, marker='o', linestyle='--', color=color, label=name, markersize=8, linewidth=2)

plt.title('Comparison of models based on cross-validation folds', fontsize=14)
plt.xlabel('Fold number', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.xticks(range(1, 6))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()


from sklearn.ensemble import VotingClassifier
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


pipe_test = pipe_test.drop(columns = ['id'])


df_train.head()


df_train.shape[1] == pipe_test.shape[1]


final_stack.fit(df_train, y)
pred_final_0 = final_stack.predict_proba(df_train)[:, 1]
score = roc_auc_score(y, pred_final_0)
print(f"Score on Train Data (full dataset) is - {score}")

# Calibration
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


output['rainfall']

