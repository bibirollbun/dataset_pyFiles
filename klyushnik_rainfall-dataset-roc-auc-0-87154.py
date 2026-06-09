# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy import stats
from scipy.optimize import minimize
from scipy.stats import mstats

import catboost
from catboost import CatBoostClassifier
from catboost.utils import get_fnr_curve, get_fpr_curve, get_roc_curve

import lightgbm as lgb
import xgboost as xgb

from mlxtend.classifier import StackingCVClassifier

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import (SelectKBest, RFECV, chi2,
                                       VarianceThreshold, SequentialFeatureSelector)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             log_loss, roc_curve, roc_auc_score)
from sklearn.model_selection import (KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import (LabelEncoder, QuantileTransformer, StandardScaler,
                                   PowerTransformer, MaxAbsScaler, MinMaxScaler,
                                   RobustScaler, PolynomialFeatures, OrdinalEncoder,
                                   OneHotEncoder, FunctionTransformer, KBinsDiscretizer)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from category_encoders import TargetEncoder, MEstimateEncoder
# from cuml.preprocessing import TargetEncoder

from imblearn.over_sampling import (SMOTE, ADASYN,
                                    BorderlineSMOTE, RandomOverSampler,
                                    KMeansSMOTE)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline, Pipeline

import optuna

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras

mpl.rcParams.update(mpl.rcParamsDefault)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

%matplotlib inline


def plot_numerical_features(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.histplot(df[feature], bins=30, kde=True, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=14)
        axes[i].set_ylabel('Frequency', fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)  

        mean_value = df[feature].mean()
        axes[i].axvline(mean_value, color='red', linestyle='--', label='Mean')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_numerical_boxplots(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.boxplot(x=df[feature], ax=axes[i], color='lightgreen')
        axes[i].set_title(f'Boxplot of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)  

        median_value = df[feature].median()
        axes[i].axvline(median_value, color='orange', linestyle='--', label='Median')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_qq_plot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        stats.probplot(df[feature], dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ Plot of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel('Theoretical Quantiles', fontsize=14)
        axes[i].set_ylabel('Sample Quantiles', fontsize=14)
        axes[i].grid(True, linestyle='--', alpha= 0.7)  

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, method='spearman'):
    num_df = df.select_dtypes(include=[np.number])
    
    corr = num_df.corr(method=method)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, linewidths=.5)
    plt.title(f'Correlation Matrix ({method.capitalize()} Correlation)', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def plot_pairplot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    sns.pairplot(df[num_features], diag_kind='kde', plot_kws={'alpha': 0.6, 'edgecolor': 'k'}, height=2.5)
    plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=18, fontweight='bold')
    plt.show()

def plot_categorical_features(df, ncols=2, top_n=None):
    cat_features = df.select_dtypes(include=[object]).columns
    nrows = (len(cat_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(cat_features):
        if top_n is not None:
            top_categories = df[feature].value_counts().nlargest(top_n).index
            sns.countplot(data=df[df[feature].isin(top_categories)], y=feature, ax=axes[i], palette='viridis', order=top_categories)
        else:
            sns.countplot(data=df, y=feature, ax=axes[i], palette='viridis')
        
        axes[i].set_title(f'Count of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel('Count', fontsize=14)
        axes[i].set_ylabel(feature, fontsize=14)
        axes[i].tick_params(axis='y', rotation=0)
        axes[i].grid(True, linestyle='--', alpha=0.7)  
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def PolynomialFeatures_labeled(input_df,power):
   
    poly = preprocessing.PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s+%d" % (variable,power)
                if final_label == "":         #If the final label isn't yet specified
                    final_label = intermediary_label
                else:
                    final_label = final_label + "x" + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    return output_df

def variance_threshold(df,th):
    var_thres=VarianceThreshold(threshold=th)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:,new_cols]
   
def optimize_memory_usage(df, print_size=True):
    """
    Optimizes memory usage in a DataFrame by downcasting numeric columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to optimize.
        print_size (bool): If True, prints memory usage before and after optimization.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    # Types for optimization.
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Memory usage size before optimize (Mb).
    before_size = df.memory_usage().sum() / 1024**2
    
    for column in df.columns:
        column_type = df[column].dtype
        
        if column_type in numerics:
            try:
                if str(column_type).startswith('int'):
                    df[column] = pd.to_numeric(df[column], downcast='integer')
                else:
                    df[column] = pd.to_numeric(df[column], downcast='float')
                logger.info(f"Optimized column {column}: {column_type} -> {df[column].dtype}")
            except Exception as e:
                logger.error(f"Failed to optimize column {column}: {e}")
    
    # Memory usage size after optimize (Mb).
    after_size = df.memory_usage().sum() / 1024**2
    
    if print_size:
        print(
            'Memory usage size: before {:5.4f} Mb - after {:5.4f} Mb ({:.1f}%).'.format(
                before_size, after_size, 100 * (before_size - after_size) / before_size
            )
        )
    
    return df


def generate_features(df):
    ## lag feature
    for lag in [1, 3, 7]:
        df[f'Pressure_lag{lag}'] = df['pressure'].shift(lag)
        df[f'Humidity_lag{lag}'] = df['humidity'].shift(lag)

    for c in ['maxtemp', 'temparature', 'mintemp','dewpoint', 'cloud', 'sunshine', 'winddirection','windspeed']:
        for gap in [1, 3, 7]:
            df[c+f"_shift{gap}"]=df[c].shift(gap)
            df[c+f"_diff{gap}"]=df[c].diff(gap)

    # # day features
    # df['month']=df['day']//31
    # df['sin_day']=np.sin(2*np.pi*df['day']/365)
    # df['cos_day']=np.cos(2*np.pi*df['day']/365)
    
    ## amount of change
    df['Pressure_change_1d'] = df['pressure'] - df['pressure'].shift(1)
    df['Humidity_change_1d'] = df['humidity'] - df['humidity'].shift(1)
    df['Maxtemp_change_1d'] = df['maxtemp'] - df['maxtemp_shift1']
    df['Temparature_change_1d'] = df['temparature'] - df['temparature_shift1']
    df['Mintemp_change_1d'] = df['mintemp'] - df['mintemp_shift1']
    df['Dewpoint_change_1d'] = df['dewpoint'] - df['dewpoint_shift1']
    df['Cloud_change_1d'] = df['cloud'] - df['cloud_shift1']
    df['Sunshine_change_1d'] = df['sunshine'] - df['sunshine_shift1']
    df['Winddirection_change_1d'] = df['winddirection'] - df['winddirection_shift1']
    df['Windspeed_change_1d'] = df['windspeed'] - df['windspeed_shift1']
    
    ## temperature related
    df['Temp_range'] = df['maxtemp'] - df['mintemp']
    df["avg_temp"] = (df["maxtemp"] + df["mintemp"]) / 2
    df['Dewpoint_diff'] = df['temparature'] - df['dewpoint']

    ## sunshine, cloud amount
    df['Sunshine_per_hour'] = df['sunshine'] / 24
    df['Cloud_per_hour'] = df['cloud'] / 24
    df['Cloud_Humidity_ratio'] = df['cloud'] / (df['humidity'] + 1e-5)
    df['Cloud_Sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1e-5)

    ## wind related
    df['Wind_x'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    df['Wind_y'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))

    ## others
    df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
    df['humidity_sunshine_interaction'] = df['humidity'] * df['sunshine']
    df['Pressure_Humidity_Interaction'] = df['pressure'] * df['humidity']
    df["cloud_wind_interaction"] = df["cloud"] * df["windspeed"]
    df['relative_dryness'] = 100 - df['humidity']
    df['sunshine_percentage'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['cloud_percentage'] = df['cloud'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['weather_index'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    df['Temp_Ratio'] = df['temparature'] / df['maxtemp'].max()

    # wet-bulb temperature
    def calc_wet_bulb(T, RH):
        return T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
               np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
               0.00391838 * RH**(3/2) * np.arctan(0.023101 * RH) - 4.686035

    df['wet_bulb_temp'] = calc_wet_bulb(df['temparature'], df['humidity'])

    # saturated vapor pressure
    def calc_saturation_vapor_pressure(temp):
        return 6.11 * np.exp((17.27 * temp) / (temp + 237.3))

    df['e_s_temp'] = calc_saturation_vapor_pressure(df['temparature'])
    df['e_s_dewpoint'] = calc_saturation_vapor_pressure(df['dewpoint'])

    # vapor pressure deficit
    df['vapor_pressure_deficit'] = df['e_s_temp'] - df['e_s_dewpoint']
    
    df.fillna(method='bfill', inplace=True)
    
    return df


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

train.shape, test.shape, original.shape


test = test.drop(['id'], axis =1)
train = train.drop(['id'], axis =1)

train.shape, test.shape


original.columns = original.columns.str.replace(' ', '')
original = original[original.columns].copy()
original['rainfall'] = original['rainfall'].map({'no': 0, 'yes': 1})
original['humidity']=original['humidity'].astype(float)
original['cloud']=original['cloud'].astype(float)
train_features=list(train)
original=original[train_features]
train = pd.concat([train, original], axis=0, ignore_index=True)

train.shape, test.shape


train.head()


train.info()


duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

train = train.drop_duplicates()


for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


train.describe().T


train = optimize_memory_usage(train)
test = optimize_memory_usage(test)


print(train['day'].min(),',', train['day'].max())


plot_numerical_features(train)


plot_numerical_boxplots(train)


plot_qq_plot(train)


plot_correlation_matrix(train)


test['winddirection']=test['winddirection'].fillna(value=test['winddirection'].mean())
train['winddirection']=train['winddirection'].fillna(value=train['winddirection'].mean())
train['windspeed']=train['windspeed'].fillna(value=train['windspeed'].mean()) 


plot_pairplot(train)


plt.figure(figsize=(10, 6))
sns.countplot(x='rainfall', data=train, palette='muted')
plt.title('Distribution of Rainfall', fontsize=18)
plt.xlabel('Rainfall', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y')
plt.show()


X = sm.add_constant(train.select_dtypes(include=[np.number]).iloc [:, 1:])

VIFs = pd.DataFrame()
VIFs['Variable'] = X.columns
VIFs['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(VIFs)


# train = generate_features(train)
# test = generate_features(test)

# train.shape, test.shape


test = test.drop(['maxtemp', 'mintemp'], axis =1)
train = train.drop(['maxtemp', 'mintemp'], axis =1)

train.shape, test.shape


def categorize_variable(df, column, labels):
    bins = [-float('inf')] + [df[column].quantile(0.25), df[column].quantile(0.75)] + [float('inf')]
    df[f'{column}_group'] = pd.cut(df[column], bins=bins, labels=labels)

categorize_variable(train, 'pressure', ["low", "normal", 'high'])
categorize_variable(train, 'temparature', ["low", "normal", 'high'])
categorize_variable(train, 'dewpoint', ["low", "normal", 'high'])
categorize_variable(train, 'humidity', ["low", "normal", 'high'])
categorize_variable(train, 'cloud', ["low", "normal", 'high'])

categorize_variable(test, 'pressure', ["low", "normal", 'high'])
categorize_variable(test, 'temparature', ["low", "normal", 'high'])
categorize_variable(test, 'dewpoint', ["low", "normal", 'high'])
categorize_variable(test, 'humidity', ["low", "normal", 'high'])
categorize_variable(test, 'cloud', ["low", "normal", 'high'])

categorize_variable(train, 'sunshine', ['high', "normal","low"])
categorize_variable(test, 'sunshine',['high', "normal","low"])


categorize_variable(train, 'winddirection', ["low", "normal", 'high'])
categorize_variable(train, 'windspeed', ["low", "normal", 'high'])

categorize_variable(test, 'winddirection', ["low", "normal", 'high'])
categorize_variable(test, 'windspeed', ["low", "normal", 'high'])

train_shape = train.shape
test_shape = test.shape

train.shape, test.shape


col = ['pressure_group','temparature_group', 'dewpoint_group', 'humidity_group', 'cloud_group',
       'sunshine_group', 'winddirection_group', 'windspeed_group']
col_num = ['pressure', 'temparature', 'dewpoint', 'humidity', 'cloud',
       'sunshine', 'winddirection', 'windspeed']

TE = MEstimateEncoder(cols=col, m=5.0)


train[col] = TE.fit_transform(train[col], train['rainfall'])
test[col] = TE.transform(test[col])

train.shape, test.shape


X = train.drop(columns=['rainfall'])
y = train['rainfall']
print('before threshold:',X.shape, y.shape)

X = variance_threshold(X,0.01)
list_name = (X.columns)
test = test[list_name]

print('after threshold:',X.shape, y.shape)


scaler = StandardScaler()

X[X.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(X[X.select_dtypes(include=[np.number]).columns])
test[X.select_dtypes(include=[np.number]).columns] = scaler.transform(test[X.select_dtypes(include=[np.number]).columns])

oversample = make_pipeline(
    ADASYN(sampling_strategy='all', random_state=42, n_neighbors = 7),
    KMeansSMOTE(sampling_strategy='all', random_state=42, k_neighbors = 7),
    RandomOverSampler(random_state=42)               
)

X, y = oversample.fit_resample(X, y)

X.shape, y.shape, test.shape


def objective(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 300, 1500, step=10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 14, step=1),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10, step=1),
        'eval_metric': 'Logloss',  
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 2, 10, step=1),
        'bagging_temperature': trial.suggest_int('bagging_temperature', 1, 10, step=1),
        'random_seed': 42, 
        'use_best_model': True
    }

    model = CatBoostClassifier(**param, early_stopping_rounds=300, verbose=0)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = []

    for kfold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_test, y_test = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        preds = model.predict_proba(X_test)[:, 1] 
        score = roc_auc_score(y_test, preds)  
        scores.append(score)

    return np.mean(scores)  

study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=50)

cat_param = study.best_params

print("Best parameters found: ", cat_param)


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=10),  
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 14, step=1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),  
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),  
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 2, 10, step=1),
        'random_state': 42,  
        'use_label_encoder': False
    }

    model = xgb.XGBClassifier(**param, early_stopping_rounds=300)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = []

    for kfold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_test, y_test = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict_proba(X_test)[:, 1] 
        score = roc_auc_score(y_test, preds)  
        scores.append(score)

    return np.mean(scores)  


study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=50) 

xgb_param = study.best_params

print("Best parameters found: ", xgb_param)


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=10),  
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 14, step=1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),  
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),  
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 2, 10, step=1),
        'random_state': 42,  
        'metric': 'auc' 
        
    }

    model = lgb.LGBMClassifier(**param, early_stopping_rounds=300, verbose=-1)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = []

    for kfold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_test, y_test = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict_proba(X_test)[:, 1] 
        score = roc_auc_score(y_test, preds)  
        scores.append(score)

    return np.mean(scores)  

study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=50)  

lgb_param = study.best_params

print("Best parameters found: ", lgb_param)


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")
    
if tpu:
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu,) 
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy() 
    print('Running on single GPU ', gpus[0].name)
else:
    strategy = tf.distribute.get_strategy() 
    print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)


input_shape = [X.shape[1]]


model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(input_shape)), 
    layers.BatchNormalization(),
    layers.Dense(256, kernel_initializer='he_uniform', activation='relu'), 
    layers.Dropout(0.2),
    layers.Dense(128, kernel_initializer='he_uniform', activation='relu'), 
    layers.Dropout(0.2),
    layers.Dense(32, kernel_initializer='he_uniform', activation='relu'), 
    layers.Dropout(0.2),
    layers.Dense(16, kernel_initializer='he_uniform', activation='relu'),  
    layers.Dense(1, activation='sigmoid')  
])
model.summary()


cat_param_0 = {'iterations': 1370, 'learning_rate': 0.04563823977777935, 'max_depth': 11, 
               'l2_leaf_reg': 1, 'rsm': 0.8, 'scale_pos_weight': 4, 'bagging_temperature': 6}

xgb_param_0 = {'n_estimators': 760, 'learning_rate': 0.07802492725288919, 'max_depth': 14, 
               'reg_alpha': 0.054284094896553725, 'reg_lambda': 8.690431594512857, 'subsample': 0.8,
               'scale_pos_weight': 3}

lgb_param_0 = {'n_estimators': 740, 'learning_rate': 0.05239976744883696, 'max_depth': 12, 
               'reg_alpha': 2.251458225967053, 'reg_lambda': 0.4615033264861923, 'subsample': 1.0, 
               'scale_pos_weight': 7}


folds = 5
shuffle=True

predictions_cat, oof_cat = np.zeros(len(test)), np.zeros(len(X))
predictions_xgb, oof_xgb = np.zeros(len(test)), np.zeros(len(X))
predictions_lgbm, oof_lgbm = np.zeros(len(test)), np.zeros(len(X))
predictions_cat_0, oof_cat_0 = np.zeros(len(test)), np.zeros(len(X))
predictions_xgb_0, oof_xgb_0 = np.zeros(len(test)), np.zeros(len(X))
predictions_lgbm_0, oof_lgbm_0 = np.zeros(len(test)), np.zeros(len(X))
predictions_clf_2, oof_clf_2 = np.zeros(len(test)), np.zeros(len(X))


kf = StratifiedKFold(n_splits=folds, shuffle=shuffle)

for kfold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model_cat = CatBoostClassifier(**cat_param, verbose=0)
    model_cat.fit(X_train, y_train,
                  eval_set=(X_val, y_val), plot=False)
    
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1] / folds
    predictions_cat += model_cat.predict_proba(test)[:, 1] / folds  
    
    roc_auc = roc_auc_score(y, oof_cat)
    
    print(f'Fold {kfold + 1}/{folds}, ROC AUC CatBoostClassifier: {roc_auc:.4f}')

    model_xgb = xgb.XGBClassifier(**xgb_param)
    model_xgb.fit(X_train, y_train)
    
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1] / folds
    predictions_xgb += model_xgb.predict_proba(test)[:, 1] / folds  
    
    roc_auc = roc_auc_score(y, oof_xgb)
    
    print(f'Fold {kfold + 1}/{folds}, ROC AUC XGBClassifier: {roc_auc:.4f}')

    model_lgbm = lgb.LGBMClassifier(**lgb_param, verbose = -1)
    model_lgbm.fit(X_train, y_train)
    
    oof_lgbm[val_idx] = model_lgbm.predict_proba(X_val)[:, 1] / folds
    predictions_lgbm += model_lgbm.predict_proba(test)[:, 1] / folds  
    
    roc_auc = roc_auc_score(y, oof_lgbm)
    
    print(f'Fold {kfold + 1}/{folds}, ROC AUC LGBMClassifier: {roc_auc:.4f}')

    model_cat_0 = CatBoostClassifier(**cat_param_0, verbose=0)
    model_cat_0.fit(X_train, y_train,
                  eval_set=(X_val, y_val), plot=False)
    
    oof_cat_0[val_idx] = model_cat_0.predict_proba(X_val)[:, 1] / folds
    predictions_cat_0 += model_cat_0.predict_proba(test)[:, 1] / folds  
    
    roc_auc = roc_auc_score(y, oof_cat_0)
    
    print(f'Fold {kfold + 1}/{folds}, ROC AUC CatBoostClassifier_0: {roc_auc:.4f}')

    model_xgb_0 = xgb.XGBClassifier(**xgb_param_0)
    model_xgb_0.fit(X_train, y_train)
    
    oof_xgb_0[val_idx] = model_xgb_0.predict_proba(X_val)[:, 1] / folds
    predictions_xgb_0 += model_xgb_0.predict_proba(test)[:, 1] / folds  
    
    roc_auc = roc_auc_score(y, oof_xgb_0)
    
    print(f'Fold {kfold + 1}/{folds}, ROC AUC XGBClassifier_0: {roc_auc:.4f}')

    model_lgbm_0 = lgb.LGBMClassifier(**lgb_param_0, verbose = -1)
    model_lgbm_0.fit(X_train, y_train)
    
    oof_lgbm_0[val_idx] = model_lgbm_0.predict_proba(X_val)[:, 1] / folds
    predictions_lgbm_0 += model_lgbm_0.predict_proba(test)[:, 1] / folds  
    
    roc_auc = roc_auc_score(y, oof_lgbm_0)
    
    print(f'Fold {kfold + 1}/{folds}, ROC AUC LGBMClassifier_0: {roc_auc:.4f}')

    model.compile(loss  = 'binary_crossentropy',
                  metrics  = ['binary_accuracy'],
                  optimizer = keras.optimizers.Adam(learning_rate=1e-4))
    early_stopping = keras.callbacks.EarlyStopping(patience=5,
                                                   min_delta=0.001,
                                                   restore_best_weights=True,) 
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor = 0.1, patience = 2, mode = 'min', verbose = 1,)
    
    model.fit(X_train, y_train,
             batch_size=150, epochs=30, 
              validation_data=(X_val, y_val),
              callbacks=[reduce_lr,early_stopping],)
    oof_clf_2[val_idx] = model.predict(X_val).flatten()/ folds
    predictions_clf_2 += model.predict(test).flatten() / folds 

    roc_auc = roc_auc_score(y, oof_clf_2)

    print(f'Fold {kfold + 1}/{folds}, ROC AUC Neural Network: {roc_auc:.4f}')    


def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve {model_name}', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


plot_roc_curve(y, oof_cat, 'CatBoostClassifier')


plot_roc_curve(y, oof_xgb, 'XGBClassifier')


plot_roc_curve(y, oof_lgbm, 'LGBMClassifier')


blend_df = pd.DataFrame({'1': oof_cat,
                         '2': oof_xgb,
                         '3': oof_lgbm,
                         '4': oof_cat_0,
                         '5': oof_xgb_0,
                         '6': oof_lgbm_0,
                         '7': oof_clf_2,
                         })

blend_test_df = pd.DataFrame({  '1': predictions_cat,  
                                '2': predictions_xgb, 
                                '3': predictions_lgbm, 
                                '4': predictions_cat_0,  
                                '5': predictions_xgb_0, 
                                '6': predictions_lgbm_0, 
                                '7': predictions_clf_2,
                        })

def calculate_roc_auc(weights, blend_df, y_):
    weighted_predictions = np.dot(blend_df, weights)
    return -roc_auc_score(y_, weighted_predictions)  
    
def constraint(weights):
    return np.sum(weights) - 1 

initial_weights = np.array([0.2] * blend_df.shape[1])  

constraints = {'type': 'eq', 'fun': constraint}
bounds = [(0, 1) for _ in range(blend_df.shape[1])]  

result = minimize(calculate_roc_auc, initial_weights, args=(blend_df, y), 
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
optimal_roc_auc = -result.fun  

print(f"Optimal weights: {optimal_weights}")
print(f"Best ROC AUC: {optimal_roc_auc:.4f}")


sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sample['rainfall'] = np.dot(blend_test_df, optimal_weights)
sample.to_csv('submission.csv', index=False)


sample.head()

