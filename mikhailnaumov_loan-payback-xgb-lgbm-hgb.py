!pip install --upgrade scikit-learn scikit-learn==1.7.2 xgboost==3.1.1 lightgbm==4.6.0 catboost==1.2.8 numpy==1.26.4 scipy==1.14.1


import numpy as np
# %load_ext cudf.pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, label_binarize, OrdinalEncoder, QuantileTransformer, TargetEncoder, RobustScaler
from category_encoders import CatBoostEncoder, MEstimateEncoder

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LinearRegression, BayesianRidge, Ridge, ElasticNet, Lasso

from sklearn import set_config
import os

import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, root_mean_squared_error, mean_squared_error, precision_recall_curve, make_scorer, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, matthews_corrcoef, r2_score
from scipy.stats import norm, skew

from colorama import Fore, Style, init
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from pprint import pprint
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, KFold, RepeatedKFold, cross_val_score, StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from xgboost import DMatrix, XGBClassifier, XGBRegressor
import xgboost as xgb
from lightgbm import log_evaluation, early_stopping, LGBMClassifier, LGBMRegressor, Dataset
import lightgbm
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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance

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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
    target = 'loan_paid_back'
    train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
    orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = 42
    n_splits = 10
    early_stop = 200
    metric = 'roc_auc'
    task_type = "binary"
    task_is_regression = task_type == 'regression'
    if task_is_regression:
        n_classes = None
    else:
        n_classes = train[target].nunique()
        labels = list(train[target].unique())

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=state)

    outliers = False
    log_trf = False
    missing = False


class EDA(Config):
    
    def __init__(self):
        super().__init__()

        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
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
        plt.figure(figsize=(6, 6))
        corr = self.train[self.num_features+[self.target]].corr(method='pearson')
        sns.heatmap(corr, fmt = '0.4f', cmap = 'Greens', square=True, annot=True, linewidths=1, cbar=False)
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


class Preprocessing(Config):
    
    def __init__(self, n_splits=5, random_state=42, smoothing=20):
        super().__init__()
        self.global_stats = {}
        self.encodings = {}
        self.freq_encodings = {}
        self.count_encodings = {}
        self.n_splits = n_splits
        self.random_state = random_state
        self.smoothing = smoothing

    def fit_transform(self):
        self.prepare_data()
        if self.missing:
            self.missing_values()

        combine = pd.concat([self.X, self.test])
        combine = self.feature_engineering(combine)
        self.X = combine.iloc[:len(self.X)].copy()
        self.test = combine.iloc[len(self.X):].copy()

        self.num_features = self.test.select_dtypes(exclude=['object', 'bool', 'category']).columns.tolist()
        self.cat_features = self.test.select_dtypes(include=['object', 'bool','category']).columns.tolist()

        if self.outliers:
            self.remove_outliers()
        if self.log_trf:
            self.log_transformation()

        return self.X, self.y, self.test, self.cat_features, self.num_features

    def prepare_data(self):
        self.train_raw = self.train.copy()
        self.y = self.train[self.target]
        self.X = self.train.drop(self.target, axis=1)

        self.num_features = self.X.select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.X.select_dtypes(include=['object', 'bool']).columns.tolist()

    def feature_engineering(self, data):
        df = data.copy()
        highcard = ['annual_income', 'loan_amount']
        lowcard = [col for col in self.num_features if col not in highcard]
        
        global_stats = {'mean': self.orig[self.target].mean(), 'count': 0}
        for c in self.num_features + self.cat_features:
            for a in ['mean', 'count']:
                col = f'{c}_org_{a}'
                tmp = (self.orig.groupby(c)[self.target]
                       .agg(a)
                       .rename(col)
                       .reset_index())
                df = df.merge(tmp, on=c, how='left')
                df[col] = df[col].fillna(global_stats[a])

        for c in self.num_features:
            df[f"Log_{c}"] = np.log1p(df[c])
            df[f"{c}_sq"] = df[c]**2

        self.numtocat_features = []
        for c in lowcard:
            df[f"{c}_cat"], _ = df[c].factorize()
            df[f"{c}_cat"] = df[f"{c}_cat"].astype('category')
            self.numtocat_features.append(f"{c}_cat")

        for c in highcard:
            df[f'{c}_round'] = df[c].round(0)
            df[f"{c}_round"], _ = pd.factorize(df[f"{c}_round"])
            df[f"{c}_round"] = df[f"{c}_round"].astype('category')
            self.numtocat_features.append(f"{c}_round")
            df[f'{c}_thousands'] = df[c].round(-3)
            df[f"{c}_thousands"], _ = pd.factorize(df[f"{c}_thousands"])
            df[f"{c}_thousands"] = df[f"{c}_thousands"].astype('category')
            self.numtocat_features.append(f"{c}_thousands")

        df['grade_number'] = df['grade_subgrade'].str[1].astype(int)
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        df['grade_rank'] = df['grade_subgrade'].str[0].map(grade_map)

        all_cats = self.numtocat_features+self.cat_features
        for c in all_cats:
            freqs = df[c].value_counts(normalize=True)
            df[f"{c}_fe"] = df[c].map(freqs)

        df[self.cat_features] = df[self.cat_features].astype('category')
        return df

    def log_transformation(self):
        self.y = np.log1p(self.y)

    def remove_outliers(self):
        Q1 = self.y.quantile(0.25)
        Q3 = self.y.quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        mask = (self.y >= lower_limit) & (self.y <= upper_limit)
        self.X = self.X[mask]
        self.y = self.y[mask]
        self.X.reset_index(drop=True, inplace=True)

    def missing_values(self):
        self.X[self.cat_features] = self.X[self.cat_features].fillna('NaN')
        self.test[self.cat_features] = self.test[self.cat_features].fillna('NaN')


X, y, test, cat_features, num_features = Preprocessing().fit_transform()


models = {
    'XGB_TE16': XGBClassifier(**{'tree_method': 'hist',
                                 'n_estimators': 10000,
                                 'objective': 'binary:logistic',
                                 'random_state': Config.state,
                                 'enable_categorical': True,
                                 'verbosity': 0,
                                 'eval_metric': 'auc',
                                 'booster': 'gbtree',
                                 'n_jobs': -1,
                                 'learning_rate': 0.01,
                                 "device": "cuda",
                                 'lambda': 0.31304742836116506,
                                 'alpha': 4.446958778347978, 
                                 'colsample_bytree': 0.10024697719858375,
                                 'subsample': 0.6689970856296092, 
                                 'max_depth': 8,
                                 'min_child_weight': 2,
                                 "min_samples_split": 5,
                                 'max_bin': 512
                               }),
    'XGB6': XGBClassifier(**{'tree_method': 'hist',
                             'n_estimators': 10000,
                             'objective': 'binary:logistic',
                             'random_state': Config.state,
                             'enable_categorical': True,
                             'verbosity': 0,
                             'eval_metric': 'auc',
                             'booster': 'gbtree',
                             'n_jobs': -1,
                             'learning_rate': 0.01,
                             "device": "cuda",
                             'lambda': 1.7433102083275247,
                             'alpha': 1.648659627164799,
                             'colsample_bytree': 0.10650947532921384,
                             'subsample': 0.9225209864222714,
                             'max_depth': 4, 
                             'min_child_weight': 3,
                             'max_bin': 512,
                           }),
    'XGB6_cl': XGBClassifier(**{'tree_method': 'hist',
                                'n_estimators': 10000,
                                'objective': 'binary:logistic',
                                'random_state': Config.state,
                                'enable_categorical': True,
                                'verbosity': 0,
                                'eval_metric': 'logloss',
                                'booster': 'gbtree',
                                'n_jobs': -1,
                                'learning_rate': 0.01,
                                "device": "cuda",
                                'lambda': 1.7433102083275247,
                                'alpha': 1.648659627164799,
                                'colsample_bytree': 0.10650947532921384,
                                'subsample': 0.9225209864222714,
                                'max_depth': 4, 
                                'min_child_weight': 3,
                                'max_bin': 512,
                               }),
    'LGBM2': LGBMClassifier(**{'random_state': Config.state,
                               'early_stopping_round': Config.early_stop,
                               'verbose': -1,
                               'n_estimators': 10000,
                               'metric': 'auc',
                               'objective': 'binary',
                               'max_bin': 500,
                               'max_depth': 5, 
                               'learning_rate': 0.03064219130978399, 
                               'min_child_samples': 190,
                               'subsample': 0.4736164696953288,
                               'colsample_bytree': 0.21677995910966458,
                               'num_leaves': 318,
                               'reg_alpha': 4.818090956944737,
                               'reg_lambda': 0.016624049322167257,
                              }),
    'LGBM3': LGBMClassifier(**{'random_state': Config.state,
                               'early_stopping_round': Config.early_stop,
                               'verbose': -1,
                               'n_estimators': 10000,
                               'metric': 'AUC',
                               'objective': 'binary',
                               'learning_rate': 0.01,
                               'max_depth': 5,
                               'min_child_samples': 162, 
                               'subsample': 0.44186829450007403,
                               'colsample_bytree': 0.23231047438980407,
                               'num_leaves': 332, 
                               'reg_alpha': 0.049317572788057186,
                               'reg_lambda': 7.073507415197327,
                               'max_bin': 500,
                              }),
    'HGB3': HistGradientBoostingClassifier(**{'max_iter': 10000,
                                              'random_state': Config.state,
                                              'early_stopping': True,
                                              'categorical_features': "from_dtype",
                                              'learning_rate': 0.01,
                                              'loss': 'log_loss',
                                              'scoring': 'loss',
                                              'l2_regularization': 0.011425355549456015,
                                              'max_depth': 4,
                                              'max_leaf_nodes': 85,
                                              'min_samples_leaf': 50
                                            }),
    'predict':_,
}


class FeatureEncoder:
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.ohe_cols = None

    def fit(self, X):
        self.ohe.fit(X[self.cat_features])
        self.ohe_cols = self.ohe.get_feature_names_out(self.cat_features)
        self.scaler.fit(X[self.num_features])
        
    def transform_fold(self, X_train, X_val, X_test):
        def transform(X):
            X[self.num_features] = self.scaler.transform(X[self.num_features])

            X_ohe = self.ohe.transform(X[self.cat_features])
            X_ohe_df = pd.DataFrame(X_ohe, columns=self.ohe_cols, index=X.index)   
            X = pd.concat([X.drop(columns=self.cat_features).reset_index(drop=True),
                                        X_ohe_df.reset_index(drop=True)], axis=1)
            return X
        return transform(X_train), transform(X_val), transform(X_test)


class Trainer(Config):
    
    def __init__(self, X, y, test, models, num_features, cat_features, training=True):
        self.X = X
        self.test = test
        self.y = y
        self.models = models
        self.training = training
        self.scores = pd.DataFrame(columns=['Score'], dtype=float)
        self.OOF_preds = pd.DataFrame(dtype=float)
        self.TEST_preds = pd.DataFrame(dtype=float)
        self.num_features = num_features
        self.cat_features = cat_features

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
        params=model.get_params()
        for n_fold, (train_id, valid_id) in enumerate(self.folds.split(X, y)):
            features = X.columns.to_list()

            X_train = X[features].loc[train_id].copy()
            y_train = y[train_id]
            X_val = X[features].iloc[valid_id].copy()
            y_val = y[valid_id]
            X_test = test[features].copy()             

            if model_name != 'Ensemble':
                TE = TargetEncoder(random_state=42, shuffle=True, cv=5, smooth=15)
                X_train[self.cat_features] = te.fit_transform(X_train[self.cat_features], y_train).astype('float32')
                X_val[self.cat_features] = te.transform(X_val[self.cat_features]).astype('float32')
                X_test[self.cat_features] = te.transform(X_test[self.cat_features]).astype('float32')
            
            print(f'Fold {n_fold+1}')
            
            if "LGBM" in model_name:
                X_train = lightgbm.Dataset(X_train, label=y_train, categorical_feature=self.cat_features)
                val_dataset = lightgbm.Dataset(X_val, label=y_val, categorical_feature=self.cat_features)
                model = lightgbm.train(
                    params=params,
                    train_set=X_train,
                    valid_sets=[val_dataset]
                )

            elif any(model in model_name for model in ["NN", "TabM"]):
                model.num_features = X_train.select_dtypes(exclude=['category']).columns.tolist()
                model.cat_features = X_train.select_dtypes(include=['category']).columns.tolist()
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
            elif "XGB" in model_name:
                X_train = DMatrix(X_train, label=y_train, enable_categorical=True)
                X_val   = DMatrix(X_val, label=y_val, enable_categorical=True)
                X_test  = DMatrix(X_test, enable_categorical=True)
                model = xgb.train(
                    params=params,
                    dtrain=X_train,
                    evals=[(X_val, "valid")],
                    num_boost_round=100_000,
                    early_stopping_rounds=200,
                    verbose_eval=False
                )

            elif "CAT" in model_name:
                X_train = Pool(X_train, label=y_train, cat_features=self.cat_features)
                X_val = Pool(X_val, label=y_val, cat_features=self.cat_features)
                X_test = Pool(test, cat_features=self.cat_features)
                model.fit(X_train, eval_set=X_val, verbose=False)
                
            elif any(model in model_name for model in ["HGB", "YDF"]):
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                
            elif "Ensemble" in model_name:
                model = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("ridge", model)
                ])
                model.fit(X_train, y_train)
                
            else:
                encoder = FeatureEncoder(num_features=self.num_features, cat_features=self.cat_features)
                encoder.fit(X)
                X_train, X_val, X_test = encoder.transform_fold(X_train, X_val, X_test)
          
                model.fit(X_train, y_train)

            if self.task_type == "regression" :
                y_pred_val = model.predict(X_val)           
                test_pred += model.predict(X_test) / self.n_splits
            elif self.task_type == "binary" :
                y_pred_val = model.predict_proba(X_val)[:, 1]            
                test_pred += model.predict_proba(X_test)[:, 1] / self.n_splits
            elif self.task_type == "multiclass" :
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
                X = self.X.copy()
                test = self.test.copy()

                oof_pred, test_pred = self.train(model, X, self.y, test, model_name)
                pd.DataFrame(oof_pred, columns=[f'{model_name}']).to_csv(f'{model_name}_oof.csv', index=False)
                pd.DataFrame(test_pred, columns=[f'{model_name}']).to_csv(f'{model_name}_test.csv', index=False)
            
            else:
                oof_pred = pd.read_csv(f'/kaggle/input/loan-models/{model_name}_oof.csv')
                test_pred = pd.read_csv(f'/kaggle/input/loan-models/{model_name}_test.csv')
                if 'predict' in model_name:
                    oof_pred = oof_pred['loan_paid_back']
                    test_pred = test_pred['loan_paid_back']
                for n_fold, (train_id, valid_id) in enumerate(self.folds.split(oof_pred, self.y)):
                    y_pred_val, y_val = oof_pred.loc[valid_id], self.y.loc[valid_id]
                    self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = self.ScoreMetric(y_val, y_pred_val)
                self.scores.loc[f'{model_name}', 'Score'] = self.scores.loc[f'{model_name}'][1:].mean()

            self.OOF_preds[f'{model_name}'] = oof_pred
            self.TEST_preds[f'{model_name}'] = test_pred
            
        if len(self.models)>1:
            if self.task_is_regression:
                meta_model = LinearRegression()
            else:
                meta_model = LogisticRegression(C = 0.1, random_state = self.state, max_iter = 1000)
            
            self.OOF_preds["Ensemble"], self.TEST_preds["Ensemble"] = self.train(meta_model, self.OOF_preds, y, self.TEST_preds, 'Ensemble')            
            self.scores = self.scores.sort_values('Score')
            self.score_bar()
            self.plot_result(self.OOF_preds["Ensemble"])
            return self.TEST_preds["Ensemble"]
        else:
            print(Style.BRIGHT+Fore.GREEN+f'{model_name} score {self.scores.loc[f"{model_name}", "Score"]:.7f}\n')
            self.plot_result(self.OOF_preds[f'{model_name}'])
            return self.TEST_preds[f'{model_name}']
            
    def score_bar(self):
        plt.figure(figsize=(18, 7))      
        colors = ['#3cb371' if i != 'Ensemble' else 'r' for i in self.scores.Score.index]
        hbars = plt.barh(self.scores.index, self.scores.Score, color=colors, height=0.8)
        plt.bar_label(hbars, fmt='%.6f')
        plt.ylabel('Models')
        plt.xlabel('Score')
        plt.show()
        
    def plot_result(self, oof):
        if self.task_is_regression:
            cmap = LinearSegmentedColormap.from_list("red2green", ["#3cb371", "r"], N=10)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            errors = np.abs(y - oof)
            axes[0].scatter(y, oof, c=errors, cmap=cmap, alpha=0.5, s=5)
            axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            axes[0].set_xlabel('Actual')
            axes[0].set_ylabel('Predicted')
            axes[0].set_title('Actual vs. Predicted')
            
            residuals = y - oof
            axes[1].scatter(oof, residuals, c=errors, cmap=cmap, alpha=0.5, s=5)
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


trainer = Trainer(X, y, test, models, num_features, cat_features, training=False)
TEST_preds = trainer.run()


submission = Config.submission
submission[Config.target] = TEST_preds
submission.to_csv("submission.csv", index=False)

display(submission.head())
plt.figure(figsize=(14, 8))
sns.distplot(submission[Config.target], bins=100, hist_kws={'alpha': 1, 'color': '#3cb371'}, kde_kws={'color': 'red', 'linewidth': 2})
plt.show()

