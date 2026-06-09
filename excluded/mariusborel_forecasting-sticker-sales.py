import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# change default colormap
plt.rcParams['image.cmap'] = 'Set3'

# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error, mean_absolute_percentage_error

# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from sklearn.feature_selection import SequentialFeatureSelector as sk_sfs
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor, 
                              StackingRegressor, BaggingRegressor,VotingRegressor)
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, plot_importance, cv

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras import layers

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, minmax_scale,
                                   OneHotEncoder, FunctionTransformer)

import yellowbrick
from yellowbrick.classifier import ClassificationReport, DiscriminationThreshold, confusion_matrix
from yellowbrick.regressor import PredictionError
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from yellowbrick.regressor import ResidualsPlot

import optuna
from optuna.samplers import TPESampler
import plotly.express as px

# Set the color scheme 
my_scheem = 'copper_r'
sns.set_palette(my_scheem)

pd.set_option('display.max_columns', 100)
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')
print(f'yellowbrick version: {yellowbrick.__version__}')


train_00 = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_00 = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_00.head(5)


train_00.info()


train_00.shape


train_00.isna().mean()*100


# drop the rows that are missing values

train_01 = train_00.dropna()


def date_processor(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['weekend'] = df['day_of_week'].apply(lambda x: 'weekend' if x>=5 else 'weekday')
    return df


train_02 = date_processor(train_01)
test_02 = date_processor(test_00)
train_02.sample(5)


plt.plot(train_01['date'], train_01['num_sold'], marker='o', alpha=0.5)


mean_day_sold = train_01.groupby('date').max('num_sold')[['num_sold']]
mean_day_sold


train_01.info()


sns.countplot(train_01, x='product', hue='store')


sns.countplot(train_01, x='country', hue='store')


sns.kdeplot(train_01, x='num_sold', hue='store', fill=True, alpha=.8)


sns.histplot(train_01, x='num_sold', hue='store', fill=True, alpha=.8, bins=30)


cat_feats = train_01.select_dtypes(exclude='number')


test_00.head(5)


test_00.shape


train_00.describe(exclude='number')


train_00['product'].value_counts()


cat_feats = ['country', 'store', 'product']

for cat_feat in cat_feats:
    feat_count = train_00[cat_feat].value_counts().to_frame()
    display(feat_count)


train_00['num_sold'].plot.hist(bins=20)


sns.boxplot(train_01, x='num_sold', y='weekend')


sns.boxplot(train_01, x='num_sold', y='product')


sns.boxplot(train_01, x='num_sold', y='store')


sns.boxplot(train_01, x='num_sold', y='country')


sns.boxenplot(train_01, x='num_sold', y='country')


X = train_01.copy()
y = X.pop('num_sold')

X


n_splits=5

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

kfold


for f, (tr_ind, ts_ind) in enumerate(kfold.split(X,y)):
    X_tr, X_ts = X.iloc[tr_ind], X.iloc[ts_ind]
    y_tr, y_ts = y.iloc[tr_ind], y.iloc[ts_ind]
    print(f'fold_{f}')
    display(X_tr)


plt.figure(figsize=(8,8))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=2, height=3)

# Map the plot to the facet grid
# g.map(sns.lineplot, 'date', 'value')
g.map(sns.boxplot, 'num_sold', 'country')

# Add titles and adjust layout
g.set_titles("{col_name} category")
g.set_axis_labels("Date", "Value")
plt.tight_layout()
plt.show()



plt.figure(figsize=(12,6))
# Create the facet grid
g = sns.FacetGrid(train_01, col="product", col_wrap=3, height=3)

# Map the plot to the facet grid
# g.map(sns.lineplot, 'date', 'value')
g.map(sns.boxplot, 'num_sold', 'country')

# Add titles and adjust layout
g.set_titles("{col_name} category")
g.set_axis_labels("num_sold")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,6))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)

# Map the plot to the facet grid
# g.map(sns.lineplot, 'date', 'value')
g.map(sns.boxplot, 'num_sold', 'product')

# Add titles and adjust layout
g.set_titles("{col_name} category")
g.set_axis_labels("num_sold", "Value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,6))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)

# Map the plot to the facet grid
# g.map(sns.lineplot, 'date', 'value')
g.map(sns.kdeplot, 'num_sold')

# Add titles and adjust layout
g.set_titles("{col_name} category")
g.set_axis_labels("num_sold", "Value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,6))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)

# Map the plot to the facet grid
# g.map(sns.lineplot, 'date', 'value')
g.map(sns.boxenplot, 'num_sold', 'weekend')

# Add titles and adjust layout
g.set_titles("{col_name} category")
g.set_axis_labels("num_sold", "Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,8))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)

# Map the plot to the facet grid
# g.map(sns.lineplot, 'date', 'value')
g.map(sns.boxenplot, 'num_sold', 'country')

# Add titles and adjust layout
g.set_titles("{col_name} category")
g.set_axis_labels("num_sold", "country")
plt.tight_layout()
plt.show()


cat_feats = test_02.select_dtypes(exclude='number').columns.tolist()
cat_feats.remove('date')


import category_encoders as ce

# feat_to_scale = X_ts.select_dtypes(include='number').columns.tolist()
features_trans = make_column_transformer(
    ('drop', ['date', 'month']),
    (ce.CatBoostEncoder(), cat_feats),
    remainder='passthrough', 
    sparse_threshold=0)


model = make_pipeline(features_trans, XGBRegressor())
model


import optuna
from xgboost import XGBRegressor
def objective(trial):
    
    xgb_params = {
        'device': 'gpu',
        'n_estimators': trial.suggest_int('n_estimators', 200,1000) ,
        'learning_rate': trial.suggest_float('learning_rate',0.0001, 0.01), 
        'max_depth': trial.suggest_int('max_depth', 3,20), 
        'min_child_weight': trial.suggest_int('min_child_weight',5, 100),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0), 
        'gamma': trial.suggest_float('gamma', 0.001,1.0), 
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0),
        'enable_categorical':True,
        'random_state':42
        
    }
    xgb = XGBRegressor(**xgb_params)
    model = make_pipeline(features_trans, xgb)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(
        X_train, y_train,
        # eval_set=[(X_val, y_val)],
        # verbose=False
    )

    y_pred = model.predict(X_val)


    mape = mean_absolute_percentage_error(y_val, y_pred)

    return mape

def run_optimizer(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
    else:
        best_params = {'n_estimators': 1371, 
               'learning_rate': 0.007399265018907012, 
               'max_depth': 18, 
               'min_child_weight': 6, 
               'subsample': 0.6203042077280833, 
               'colsample_bytree': 0.9351411779367126, 
               'gamma': 0.6683204324718459, 
               'reg_alpha': 0.8896595359422812
              }
    return best_params


best_params = run_optimizer(n_trials=50)


print(best_params)


model = make_pipeline(features_trans, XGBRegressor(**best_params))

model.fit(X, y)


preds = model.predict(test_02)

preds


pd.Series(preds).plot.kde(figsize=(6, 3))


pd.Series(preds).plot.hist(figsize=(6, 3), bins=50)


submission_00 = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission_00['num_sold'] = preds


submission_00.to_csv('submission.csv', index=False)

