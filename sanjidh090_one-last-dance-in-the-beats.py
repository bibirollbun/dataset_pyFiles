!pip install --upgrade scikit-learn scikit-learn==1.7.1 xgboost==3.0.1 lightgbm==4.6.0 catboost==1.2.8 numpy==1.26.4 scipy==1.14.1


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, label_binarize, OrdinalEncoder, QuantileTransformer, TargetEncoder, RobustScaler
from category_encoders import CatBoostEncoder, MEstimateEncoder

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LinearRegression, BayesianRidge, Ridge, ElasticNet, Lasso

from sklearn import set_config
import os

import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, root_mean_squared_error, mean_squared_error, precision_recall_curve, make_scorer, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, matthews_corrcoef
from scipy.stats import norm, skew

from colorama import Fore, Style, init
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from pprint import pprint
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, KFold, RepeatedKFold, cross_val_score, StratifiedGroupKFold
from xgboost import DMatrix, XGBClassifier, XGBRegressor
from lightgbm import log_evaluation, early_stopping, LGBMClassifier, LGBMRegressor, Dataset
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from tqdm.notebook import tqdm
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import HyperbandPruner
from functools import partial
from IPython.display import display_html, clear_output
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
import gc
import re
from typing import Literal, NamedTuple
from itertools import combinations

import keras
from keras.models import Sequential
from keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional

import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
from itertools import combinations
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")


class Config:
    target = 'BeatsPerMinute'
    train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
    train_org = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = 42
    n_splits = 10
    early_stop = 100
    metric = 'rmse'
    task_type = "regression"
    task_is_regression = task_type == 'regression'
    
    if task_is_regression:
        n_classes = None
    else:
        n_classes = train[target].nunique()
        labels = list(train[target].unique())
    
    outliers = False
    log_trf = False
    feature_eng = True
    missing = False
    training = False


class EDA(Config):
    
    def __init__(self):
        super().__init__()

        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object']).columns.tolist()
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object']).columns.tolist()
        self.data_info()
        self.heatmap()
        self.dist_plots()
        self.cat_feature_plots()
        if self.task_is_regression:
            self.target_plot()
        else:
            self.target_pie()
                
    def data_info(self):
        
        for data, label in zip([self.train, self.test], ['Train', 'Test']):
            table_style = [{'selector': 'th:not(.index_name)',
                            'props': [('background-color', '#3cb371'),
                                      ('color', '#FFFFFF'),
                                      ('font-weight', 'bold'),
                                      ('border', '1px solid #DCDCDC'),
                                      ('text-align', 'center')]
                            }, 
                            {'selector': 'tbody td',
                             'props': [('border', '1px solid #DCDCDC'),
                                       ('font-weight', 'normal')]
                            }]
            print(Style.BRIGHT+Fore.GREEN+f'\n{label} head\n')
            display(data.head().style.set_table_styles(table_style))
                           
            print(Style.BRIGHT+Fore.GREEN+f'\n{label} info\n'+Style.RESET_ALL)               
            display(data.info())
                           
            print(Style.BRIGHT+Fore.GREEN+f'\n{label} describe\n')
            display(data.describe().drop(index='count', columns=self.target, errors = 'ignore').T
                    .style.set_table_styles(table_style).format('{:.3f}'))
            
            print(Style.BRIGHT+Fore.GREEN+f'\n{label} missing values\n'+Style.RESET_ALL)               
            display(data.isna().sum())
        return self
    
    def heatmap(self):
        print(Style.BRIGHT+Fore.GREEN+f'\nCorrelation Heatmap\n')
        plt.figure(figsize=(10,10))
        corr = self.train.select_dtypes(exclude='object').corr(method='pearson')
        sns.heatmap(corr, fmt = '0.4f', cmap = 'Greens', annot=True, cbar=False)
        plt.show()
        
    def dist_plots(self):
        print(Style.BRIGHT+Fore.GREEN+f"\nDistribution analysis\n")
        df = pd.concat([self.train[self.num_features].assign(Source = 'Train'), 
                        self.test[self.num_features].assign(Source = 'Test'),], 
                        axis=0, ignore_index = True)

        fig, axes = plt.subplots(len(self.num_features), 2 ,figsize = (18, len(self.num_features) * 6), 
                                 gridspec_kw = {'hspace': 0.3, 
                                                'wspace': 0.2, 
                                                'width_ratios': [0.70, 0.30]
                                               }
                                )
        for i,col in enumerate(self.num_features):
            ax = axes[i,0]
            sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', 
                        palette = ['#3cb371', 'r'], ax = ax, linewidth = 2
                       )
            ax.set(xlabel = '', ylabel = '')
            ax.set_title(f"\n{col}")
            ax.grid()

            ax = axes[i,1]
            sns.boxplot(data = df, y = col, x=df.Source, width = 0.5,
                        linewidth = 1, fliersize= 1,
                        ax = ax, palette=['#3cb371', 'r']
                       )
            ax.set_title(f"\n{col}")
            ax.set(xlabel = '', ylabel = '')
            ax.tick_params(axis='both', which='major')
            ax.set_xticklabels(['Train', 'Test'])

        plt.tight_layout()
        plt.show()
               
    def cat_feature_plots(self):
        fig, axes = plt.subplots(max(len(self.cat_features), 1), 2 ,figsize = (18, len(self.cat_features) * 6), 
                                 gridspec_kw = {'hspace': 0.5, 
                                                'wspace': 0.2,
                                               }
                                )
        if len(self.cat_features) == 1:
            axes = np.array([axes])
            
        for i, col in enumerate(self.cat_features):
            ax = axes[i,0]
            sns.barplot(data=self.train[col].value_counts().nlargest(10).reset_index(), x=col, y='count', ax=ax, color='#3cb371')
            ax.set(xlabel = '', ylabel = '')
            ax.set_title(f"\n{col} Train")
            
            ax = axes[i,1]
            sns.barplot(data=self.train[col].value_counts().nlargest(10).reset_index(), x=col, y='count', ax=ax, color='r')
            ax.set(xlabel = '', ylabel = '')
            ax.set_title(f"\n{col} Test")

        plt.tight_layout()
        plt.show()

    def target_pie(self):
        print(Style.BRIGHT+Fore.GREEN+f"\nTarget feature distribution\n")
        targets = self.train[self.target]
        plt.figure(figsize=(6, 6))
        plt.pie(targets.value_counts(), labels=targets.value_counts().index, autopct='%1.2f%%', colors=sns.color_palette('viridis', len(targets.value_counts())))
        plt.show()

    def target_plot(self):
        print(Style.BRIGHT+Fore.GREEN+f"\nTarget feature distribution\n")
        
        fig, axes = plt.subplots(1, 2 ,figsize = (14, 6), 
                                 gridspec_kw = {'hspace': 0.3, 
                                                'wspace': 0.2, 
                                                'width_ratios': [0.70, 0.30]
                                               }
                                )
        ax = axes[0]
        sns.kdeplot(data = self.train[self.target], 
                    color = '#3cb371', ax = ax, linewidth = 2
                   )
        ax.set(xlabel = '', ylabel = '')
        ax.set_title(f"\n{self.target}")
        ax.grid()

        ax = axes[1]
        sns.boxplot(data = self.train, y = self.target, width = 0.5,
                    linewidth = 1, fliersize= 1,
                    ax = ax, color = '#3cb371'
                   )
        ax.set_title(f"\n{self.target}")
        ax.set(xlabel = '', ylabel = '')
        ax.tick_params(axis='both', which='major')

        plt.tight_layout()
        plt.show() 


eda = EDA()


class Transform(Config):
    
    def __init__(self):
        super().__init__()
            
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
        
        if self.missing:
            self.missing_values()
        
        if self.feature_eng:
            self.train = self.new_features(self.train)
            self.test = self.new_features(self.test)

        if self.outliers:    
            self.remove_outliers()
            
        if self.log_trf:
            self.log_transformation()

        self.encode()
        
    def __call__(self):

        self.y = self.train[self.target]        
        self.X = self.train.drop(self.target, axis=1)
        self.X_enc = self.train_enc.drop(self.target, axis=1)
        
        self.X = self.reduce_mem(self.X)
        self.test = self.reduce_mem(self.test)
        self.X_enc = self.reduce_mem(self.X_enc)
        self.test_enc = self.reduce_mem(self.test_enc)
        return self.X, self.X_enc, self.y, self.test, self.test_enc, self.cat_features, self.num_features, self.cat_features_card
    
    def encode(self):
        self.train_enc = self.train.copy()
        self.test_enc = self.test.copy()
        
        self.cat_features_card = []
        for f in self.cat_features:
            self.cat_features_card.append(self.train[f].nunique())
            
        data = pd.concat([self.train_enc, self.test_enc])
        self.cat_features_card = []
        for f in self.cat_features:
            self.cat_features_card.append(data[f].nunique())
            
        oe = OrdinalEncoder()
        data[self.cat_features] = oe.fit_transform(data[self.cat_features]).astype('int')

        scaler = StandardScaler()
        data[self.num_features] = scaler.fit_transform(data[self.num_features])
        
        self.train_enc = data[~data[self.target].isna()]
        self.test_enc = data[data[self.target].isna()].drop(self.target, axis=1)
            
    def new_features(self, data):
        for c1, c2 in list(combinations(self.num_features,2)):
            data[f"{c1}_{c2}"] = data[c1] * data[c2]
            data[f'{c1}_div_{c2}'] = data[c1] / (data[c2] + 1e-6)

        for c in self.num_features:
            data[f"{c}_quartile"] = pd.cut(data[c], bins=4, labels=False, include_lowest=True)
            data[f"{c}_decile"] = pd.cut(data[c], bins=10, labels=False, include_lowest=True)

        return data
        
    def log_transformation(self):
        self.train[self.target] = np.log1p(self.train[self.target]) 
        
        return self
        
    def remove_outliers(self):
        Q1 = self.train[self.targets].quantile(0.25)
        Q3 = self.train[self.targets].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5*IQR
        upper_limit = Q3 + 1.5*IQR
        self.train = self.train[(self.train[self.targets] >= lower_limit) & (self.train[self.targets] <= upper_limit)]
        self.train.reset_index(drop=True, inplace=True)
    
    def missing_values(self):
        self.train[self.cat_features] = self.train[self.cat_features].fillna('NaN')
        self.test[self.cat_features] = self.test[self.cat_features].fillna('NaN')
        return self

    def reduce_mem(self, df):

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', "uint16", "uint32", "uint64"]
        
        for col in df.columns:
            col_type = df[col].dtypes
            
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()

                if "int" in str(col_type):
                    if c_min >= np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min >= np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min >= np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    if c_min >= np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)  

        return df


t = Transform()
X, X_enc, y, test, test_enc, cat_features, num_features, cat_cardinalities = t()


def build_model(cat_features, num_features):
    
    x_input_cats = layers.Input(shape=(len(cat_features),))
    embs = []
    for j in range(len(cat_features)):
        e = layers.Embedding(cat_cardinalities[j], int(np.ceil(np.sqrt(cat_cardinalities[j]))))
        x = e(x_input_cats[:,j])
        x = layers.Flatten()(x)
        embs.append(x)
        
    x_input_nums = layers.Input(shape=(len(num_features),))
    
    x = layers.Concatenate(axis=-1)(embs+[x_input_nums]) 
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1)(x)

    model = keras.Model(inputs=[x_input_cats,x_input_nums], outputs=x)
    return model


from sklearn.base import BaseEstimator, RegressorMixin
import contextlib, io
import ydf; ydf.verbose(2)
from ydf import GradientBoostedTreesLearner

def YDFRegressor(learner_class):

    class YDFXRegressor(BaseEstimator, RegressorMixin):

        def __init__(self, params={}):
            self.params = params

        def fit(self, X, y):
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            target = y.name
            params = self.params.copy()
            params['label'] = target
            params['task'] = ydf.Task.REGRESSION
            X = pd.concat([X, y], axis=1)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.model = learner_class(**params).train(X)
            return self

        def predict(self, X):
            assert isinstance(X, pd.DataFrame)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                return self.model.predict(X)

    return YDFXRegressor


models = {
    'XGB': XGBRegressor(**{'tree_method': 'hist',
                           'n_estimators': 5000,
                           'objective': 'reg:squarederror',
                           'random_state': Config.state,
                           'enable_categorical': True,
                           'verbosity': 0,
                           'early_stopping_rounds': Config.early_stop,
                           'eval_metric': 'rmse',
                           'booster': 'gbtree',
                           'n_jobs': -1,
                           'learning_rate': 0.01,
                           'max_depth': 6, 
                           'min_child_weight': 13, 
                           'subsample': 0.7962753063760202, 
                           'reg_alpha': 0.2850105437401931, 
                           'reg_lambda': 0.9299227390692836,
                           'colsample_bytree': 0.8931705932966567,
                           "device": "cuda",
                           }),
    'XGB2': XGBRegressor(**{'n_estimators': 620,         
                            'max_leaves': 211,            
                            'min_child_weight': 1.5,     
                            'max_depth': 6,               
                            'grow_policy': 'lossguide',   
                            'learning_rate': 0.0021858703356597603,      
                            'tree_method': 'hist',        
                            'subsample': 0.85,            
                            'colsample_bylevel': 0.6787051322531533,     
                            'colsample_bytree': 0.6843905004927857,       
                            'colsample_bynode': 0.442116057736592,     
                            'sampling_method': 'uniform',  
                            'reg_alpha': 2.5,             
                            'reg_lambda': 0.8,            
                            'enable_categorical': True,    
                            'max_cat_to_onehot': 1,       
                            'device': 'cuda',            
                            'n_jobs': -1,                 
                            'random_state': 0,     
                            'verbosity': 0,               
                            }),
    'XGB3': XGBRegressor(**{'tree_method': 'hist',
                            'n_estimators': 5000,
                            'objective': 'reg:squarederror',
                            'random_state': Config.state,
                            'enable_categorical': True,
                            'verbosity': 0,
                            'early_stopping_rounds': Config.early_stop,
                            'eval_metric': 'rmse',
                            'booster': 'gbtree',
                            'n_jobs': -1,
                            'learning_rate': 0.001,
                            'max_depth': 6,
                            'min_child_weight': 10, 
                            'subsample': 0.7471419432086788,
                            'reg_alpha': 0.45207095319304624,
                            'reg_lambda': 0.582640629871527, 
                            'colsample_bytree': 0.9009247278149717
                           }),
    'LGBM': LGBMRegressor(**{'random_state': Config.state,
                             'early_stopping_round': Config.early_stop,
                             'categorical_feature': cat_features,
                             'verbose': -1,
                             'boosting_type': 'gbdt',
                             'n_estimators': 5000,
                             'eval_metric': 'rmse',
                             'objective': 'regression_l2',
                             'learning_rate': 0.01,
                             'max_depth': 5,
                             'num_leaves': 658,
                             'min_child_samples': 26,
                             'min_child_weight': 13,
                             'colsample_bytree': 0.48706267983851514, 
                             'reg_alpha': 0.18196424243062836,
                             'reg_lambda': 0.7798810879456173,
                             }),
    'LGBM2': LGBMRegressor(**{'learning_rate': 0.001502328415098844,
                              'num_leaves': 79, 
                              'max_depth': 14,
                              'feature_fraction': 0.8933016300882094,
                              'bagging_fraction': 0.9754103048412501,
                              'bagging_freq': 7, 
                              'min_child_samples': 40,
                              'lambda_l1': 7.10897934678165e-07,
                              'lambda_l2': 7.81564014894075e-08,
                              'random_state' : 0,
                              'n_jobs' : -1,
                              'verbosity': -1,
                              'n_estimators': 643
                              }),
    'LGBM3': LGBMRegressor(**{'random_state': Config.state,
                              'early_stopping_round': Config.early_stop,
                              'categorical_feature': cat_features,
                              'verbose': -1,
                              'boosting_type': 'gbdt',
                              'n_estimators': 5000,
                              'eval_metric': 'rmse',
                              'objective': 'regression_l2',
                              'learning_rate': 0.01,
                              'max_depth': 11,
                              'num_leaves': 29,
                              'min_child_samples': 9, 
                              'min_child_weight': 10,
                              'colsample_bytree': 0.47846854863406507,
                              'reg_alpha': 0.1409347783927628,
                              'reg_lambda': 0.6614029522925577
                             }),
    'HGB': HistGradientBoostingRegressor(**{'max_iter': 5000,
                                            'random_state': Config.state,
                                            'early_stopping': True,
                                            'categorical_features': "from_dtype",
                                            'learning_rate': 0.01,
                                            'loss': 'squared_error',
                                            'max_depth': 14,
                                            'l2_regularization': 0.004249214140047362,
                                            'min_samples_leaf': 33,
                                            'max_leaf_nodes': 12
                                            }),
    'HGB2': HistGradientBoostingRegressor(**{'max_iter': 5000,
                                             'random_state': Config.state,
                                             'early_stopping': True,
                                             'categorical_features': "from_dtype",
                                             'learning_rate': 0.01,
                                             'loss': 'squared_error',
                                             'max_depth': 13,
                                             'l2_regularization': 0.00036180595088386176,
                                             'min_samples_leaf': 13,
                                             'max_leaf_nodes': 20
                                             }),
    'YDF': YDFRegressor(GradientBoostedTreesLearner)({'num_trees': 1000,
                                                      'max_depth': -1,
                                                      'random_seed': Config.state,
                                                      'growing_strategy': 'BEST_FIRST_GLOBAL'
                                                      }),
    'YDF2': YDFRegressor(GradientBoostedTreesLearner)({'num_trees': 1000,
                                                       'max_depth': 5,
                                                       'random_seed': Config.state,
                                                       'growing_strategy': 'BEST_FIRST_GLOBAL'
                                                       }),
    'NN': _,
    'Ridge': Ridge(alpha=1.0, tol=1e-2, max_iter=1000000, random_state=Config.state),
    'Elastic Net': ElasticNet(alpha=1.0, random_state=Config.state),
}


class Trainer(Config):
    
    def __init__(self, X, X_enc, y, test, test_enc, models, training=True):
        self.X = X
        self.X_enc = X_enc
        self.test = test
        self.test_enc = test_enc
        self.y = y
        self.models = models
        self.training = training
        self.scores = pd.DataFrame(columns=['Score'])
        self.OOF_preds = pd.DataFrame(dtype=float)
        self.TEST_preds = pd.DataFrame(dtype=float)
        self.folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.state)

    def ScoreMetric(self, y_true, y_pred):
        if self.metric == 'roc_auc':
            return roc_auc_score(y_true, y_pred, multi_class="ovr") if self.n_classes > 2 else roc_auc_score(y_true, y_pred)
        elif self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted') if self.n_classes > 2 else f1_score(y_true, y_pred)
        elif self.metric == 'precision':
            return precision_score(y_true, y_pred, average='weighted') if self.n_classes > 2 else precision_score(y_true, y_pred)
        elif self.metric == 'recall':
            return recall_score(y_true, y_pred, average='weighted') if self.n_classes > 2 else recall_score(y_true, y_pred)
        elif self.metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.metric == 'r2':
            return r2_score(y_true, y_pred)
        elif self.metric == 'rmse':
            return root_mean_squared_error(y_true, y_pred)
        elif self.metric == 'rmsle':
            return root_mean_squared_error(y_true, y_pred)
        elif self.metric == 'mse':
            return mean_squared_error(y_true, y_pred, squared=True)
            
    def train(self, model, X, y, test, model_name):
        oof_pred = np.zeros(X.shape[0], dtype=float)
        test_pred = np.zeros(test.shape[0], dtype=float)

        print('='*20)
        print(model_name)
                
        for n_fold, (train_id, valid_id) in enumerate(self.folds.split(X, y)):
            X_train = X.loc[train_id].copy()
            y_train = y.iloc[train_id]
            X_val = X.loc[valid_id].copy()
            y_val = y.iloc[valid_id]
            X_test = test.copy()
            
            print(f'Fold {n_fold+1}')
            
            if 'NN' in model_name:
                self.num_features = [col for col in X.columns if col not in cat_features]
                X_train_cats = X_train[cat_features]
                X_train_nums = X_train[num_features]

                X_val_cats = X_val[cat_features]
                X_val_nums = X_val[num_features]

                X_test_cats = X_test[cat_features]
                X_test_nums = X_test[num_features]
                
                model = build_model(cat_features, num_features)                        
                keras.utils.set_random_seed(self.state)
                optimizer = keras.optimizers.AdamW(learning_rate=1e-2, weight_decay=1e-3)
                model.compile(optimizer=optimizer, loss='mean_squared_error')
                
                model.fit([X_train_cats,X_train_nums], y_train, 
                          validation_data=([X_val_cats, X_val_nums], y_val),
                          epochs=20,
                          batch_size=1000,
                          callbacks=[keras.callbacks.ReduceLROnPlateau(patience=1),
                                     keras.callbacks.EarlyStopping(patience=3)
                                    ])
                y_pred_val = model.predict([X_val_cats, X_val_nums]).squeeze()                      
                test_pred += model.predict([X_test_cats, X_test_nums]).squeeze() / self.n_splits             
                                      
            else:
                if any(model in model_name for model in ['TabM', 'CAT', 'XGB']):
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                elif "LGBM" in model_name:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              categorical_feature=cat_features,
                              feature_name='auto',
                              callbacks=[log_evaluation(0),early_stopping(self.early_stop, verbose=False)]
                             )

                else:                           
                    model.fit(X_train, y_train)

                if self.task_type == "regression" :
                    y_pred_val = model.predict(X_val)            
                    test_pred += model.predict(X_test) / self.n_splits
                elif self.task_type == "classification" :
                    y_pred_val = model.predict_proba(X_val)[:, 1]            
                    test_pred += model.predict_proba(X_test)[:, 1] / self.n_splits
                elif self.task_type == "multiclassification" :
                    y_pred_val = model.predict_proba(X_val)            
                    test_pred += model.predict_proba(X_test) / self.n_splits
                
            oof_pred[valid_id] = y_pred_val
            score = self.ScoreMetric(y_val, y_pred_val)
            print(score)
            self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = score                                      

        self.scores.loc[f'{model_name}', 'Score'] = self.scores.loc[f'{model_name}'][1:].mean()

        return oof_pred, test_pred

    def run(self):
        for model_name, model in tqdm(self.models.items()):

            if self.training:                
                if any(model in model_name for model in ['LGBM', 'CAT', 'XGB', 'HGB', 'YDF']):
                    X = self.X.copy()
                    test = self.test.copy()
                else:
                    X = self.X_enc.copy()
                    test = self.test_enc.copy()

                oof_pred, test_pred = self.train(model, X, y, test, model_name)
                pd.DataFrame(oof_pred, columns=[f'{model_name}']).to_csv(f'{model_name}_oof.csv', index=False)
                pd.DataFrame(test_pred, columns=[f'{model_name}']).to_csv(f'{model_name}_test.csv', index=False)
            
            else:
                oof_pred = pd.read_csv(f'/kaggle/input/models-org-2/{model_name}_oof.csv')
                test_pred = pd.read_csv(f'/kaggle/input/models-org-2/{model_name}_test.csv')
                for n_fold, (train_id, valid_id) in enumerate(self.folds.split(oof_pred, self.y)):
                    y_pred_val, y_val = oof_pred.loc[valid_id], self.y.loc[valid_id]
                    self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = self.ScoreMetric(y_val, y_pred_val)
                self.scores.loc[f'{model_name}', 'Score'] = self.scores.loc[f'{model_name}'][1:].mean()

            self.OOF_preds[f'{model_name}'] = oof_pred
            self.TEST_preds[f'{model_name}'] = test_pred
            
        if len(self.models)>1:
            if self.task_is_regression:
                meta_model = BayesianRidge(tol=1e-2, max_iter=1000000)
            else:
                meta_model = LogisticRegression(C = 0.1, random_state = self.state, max_iter = 1000)
            self.OOF_preds["Ensemble"], self.TEST_preds["Ensemble"] = self.train(meta_model, self.OOF_preds, y, self.TEST_preds, 'Ensemble')
            self.scores = self.scores.sort_values('Score')
            self.score_bar()
            self.plot_result(self.OOF_preds["Ensemble"])
            return self.TEST_preds["Ensemble"]
        else:
            print(Style.BRIGHT+Fore.GREEN+f'{model_name} score {self.scores.loc[f"{model_name}", "Score"]:.6f}\n')
            self.plot_result(self.OOF_preds[f'{model_name}'])
            return self.TEST_preds[f'{model_name}']

    def score_bar(self):
        plt.figure(figsize=(18, 6))      
        colors = ['#3cb371' if i != 'Ensemble' else 'r' for i in self.scores.Score.index]
        hbars = plt.barh(self.scores.index, self.scores.Score, color=colors, height=0.8)
        plt.bar_label(hbars, fmt='%.6f')
        plt.xlim(26.44, 26.48)
        plt.ylabel('Models')
        plt.xlabel('Score')
        plt.show()
        
    def plot_result(self, oof):
        if self.task_is_regression:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].scatter(y, oof, alpha=0.5, s=15, edgecolors='#3cb371')
            axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            axes[0].set_xlabel('Actual')
            axes[0].set_ylabel('Predicted')
            axes[0].set_title('Actual vs. Predicted')
    
            axes[1].scatter(oof, y - oof, alpha=0.5, s=15, edgecolors='#3cb371')
            axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Residual Plot')
    
            plt.tight_layout()
            plt.show()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
            for col in self.OOF_preds:
                RocCurveDisplay.from_predictions(self.y, self.OOF_preds[col], name=f"{col}", ax=axes[0])            
            axes[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC')
            axes[0].legend(loc="lower right")
            
            ConfusionMatrixDisplay.from_predictions(y, (oof>=0.5).astype(int), display_labels=self.labels, colorbar=False, ax=axes[1], cmap = 'Greens')
            axes[1].set_title('Confusion Matrix')
            
            plt.tight_layout()
            plt.show()


models


trainer = Trainer(X, X_enc, y, test, test_enc, models, training=True)
TEST_preds = trainer.run()


!pip install cir_model
from cir_model import CenteredIsotonicRegression


CIR = CenteredIsotonicRegression().fit(trainer.OOF_preds["Ensemble"], y)
oof_preds = CIR.transform(trainer.OOF_preds["Ensemble"])
print(f'Score after IsotonicRegression: {root_mean_squared_error(y, oof_preds)}')
TEST_preds = CIR.transform(TEST_preds)


import os
import pandas as pd

# ---- inputs you gave ----
paths = [
    "/kaggle/input/ps5e9-ensemble-isotonic-regression/submission_***1.csv",
    "/kaggle/input/a-few-blends-to-cave/submission_file.csv",
    "/kaggle/input/beats-per-minute-ensemble-s5e9/submission.csv",
    "/kaggle/input/playgrounds5e9-top-solution/weighted_blends_copied.csv",
]

# ---- helper: load a submission and return (key_col, preds_col_renamed) ----
def load_sub(path):
    df = pd.read_csv(path)
    # try to find an ID-like column
    id_candidates = ["Id", "id", "ID", "track_id", "row_id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)

    # choose a prediction column (if there are 2+ non-id columns, take the last)
    non_id_cols = [c for c in df.columns if c != id_col]
    if not non_id_cols:
        raise ValueError(f"No prediction column found in {path}")
    pred_col = non_id_cols[-1]

    # a friendly column name for this model
    model_name = os.path.splitext(os.path.basename(path))[0]

    if id_col is None:
        # no explicit key; use row order as key
        out = df[[pred_col]].copy()
        out.insert(0, "ROW_INDEX", range(len(out)))
        out = out.rename(columns={pred_col: model_name})
        key = "ROW_INDEX"
    else:
        out = df[[id_col, pred_col]].copy()
        out = out.rename(columns={id_col: "Id", pred_col: model_name})
        key = "Id"

    return key, out

# ---- load all submissions and outer-merge on the discovered key ----
loaded = [load_sub(p) for p in paths]
# decide the merge key: prefer "Id" if any file has it; otherwise "ROW_INDEX"
merge_key = "Id" if any(k == "Id" for k, _ in loaded) else "ROW_INDEX"

blender = None
for key, df in loaded:
    if key != merge_key:
        # convert key to merge_key if needed (only case: fill ROW_INDEX from range)
        if merge_key == "ROW_INDEX" and "ROW_INDEX" not in df.columns:
            # build ROW_INDEX based on current row order
            df = df.reset_index(drop=True)
            df.insert(0, "ROW_INDEX", range(len(df)))
        elif merge_key == "Id" and "Id" not in df.columns:
            # cannot promote ROW_INDEX to Id meaningfully; align by position instead
            df = df.reset_index(drop=True)
            df.insert(0, "ROW_INDEX", range(len(df)))
        # else: already aligned

    if blender is None:
        blender = df.copy()
    else:
        blender = pd.merge(blender, df, how="outer", on=merge_key)

# ---- write blender.csv with all model columns ----
# reorder: key first, then prediction columns
pred_cols = [c for c in blender.columns if c != merge_key]
blender = blender[[merge_key] + pred_cols]
blender.to_csv("blender.csv", index=False)

# ---- create the averaged submission ----
# default target for this competition; change if needed
TARGET = "BeatsPerMinute"

submission = pd.DataFrame()
submission[merge_key] = blender[merge_key]
submission[TARGET] = blender[pred_cols].mean(axis=1)

# if Kaggle expects "Id" specifically, rename when we used ROW_INDEX
if merge_key == "ROW_INDEX":
    submission = submission.rename(columns={"ROW_INDEX": "id"})

submission.to_csv("submission_blend.csv", index=False)

print("Wrote blender.csv and submission_blend.csv")






submission = Config.submission
blender = pd.read_csv('blender.csv')
blender[f'{trainer.scores.loc["Ensemble", "Score"]:.6f}'] = TEST_preds
submission[Config.target] = blender.mean(axis=1)
submission.to_csv("submission_blend.csv", index=False)

display(submission.head())
plt.figure(figsize=(14, 8))
sns.distplot(submission[Config.target], color='#3cb371', bins=100, hist_kws={'alpha': 0.7})
plt.show()

