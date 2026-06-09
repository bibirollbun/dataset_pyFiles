!pip install --upgrade scikit-learn scikit-learn==1.7.1 xgboost==3.0.1 lightgbm==4.6.0 catboost==1.2.8 numpy==1.26.4 scipy==1.14.1


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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, KFold, RepeatedKFold, cross_val_score, StratifiedGroupKFold, GroupKFold
from sklearn.isotonic import IsotonicRegression
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
from matplotlib.colors import LinearSegmentedColormap

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
    target = 'accident_risk'
    train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
    orig = pd.concat(
        [pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv") 
         for k in (2, 10, 100)],
        axis=0,
        ignore_index=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = 42
    n_splits = 10
    early_stop = 100
    metric = 'rmse'
    task_type = "regression"
    task_is_regression = task_type == 'regression'
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=state)

    if task_is_regression:
        n_classes = None
    else:
        n_classes = train[target].nunique()
        labels = list(train[target].unique())
    
    original_data = False
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
    
    def __init__(self):
        super().__init__()

    def fit_transform(self):
        self.prepare_data()
        if self.missing: self.missing_values()
            
        self.X = self.feature_engineering(self.X)
        self.test = self.feature_engineering(self.test)
        self.num_features = self.test.select_dtypes(exclude=['object', 'bool', 'string', 'category']).columns.tolist()
        self.cat_features = self.test.select_dtypes(include=['object', 'bool', 'string', 'category']).columns.tolist()

        if self.outliers: self.remove_outliers()
        if self.log_trf: self.log_transformation()
            
        self.X['Log_target'] = np.log1p(pd.read_csv('/kaggle/input/risk-models/preds_oof.csv'))
        self.test['Log_target'] = np.log1p(pd.read_csv('/kaggle/input/risk-models/preds_test.csv'))    

        return self.X, self.y, self.test, self.cat_features, self.num_features
        
    def prepare_data(self):
        if self.original_data:
            self.train = pd.concat([self.train, self.train_org], ignore_index=True).drop_duplicates()
            self.train.reset_index(drop=True, inplace=True)
            
        self.train_raw = self.train.copy()
        self.y = self.train[self.target]        
        self.X = self.train.drop(self.target, axis=1)
        
        self.num_features = self.X.select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.X.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    def feature_engineering(self, data):
        mean = self.orig.accident_risk.mean()
        median = self.orig.accident_risk.median()
        for c in self.num_features+self.cat_features:
            tmp = (self.orig.groupby(c).accident_risk           
                .agg(['mean', 'median'])
                .rename(columns=lambda a: f'{c}_org_{a}')
                .reset_index())
            data = data.merge(tmp, on=c, how='left')

        data['curvature_org_mean'] = data['curvature_org_mean'].fillna(mean)
        data['curvature_org_median'] = data['curvature_org_median'].fillna(median)
        
        for c in self.num_features:
            data[f"{c}_quartile"] = pd.cut(data[c], bins=4, labels=False, include_lowest=True).astype('category')
        
        for c in ['curvature', 'speed_limit']:
            data[f"{c}_sq"] = data[c]**2

        data["is_high_speed_night"] = ((data["speed_limit"] > 60) & (data["lighting"] == "night")).astype(int)

        def f(X):
            return \
            0.3 * X["curvature"] + \
            0.2 * (X["lighting"] == "night").astype(int) + \
            0.1 * (X["weather"] != "clear").astype(int) + \
            0.2 * (X["speed_limit"] >= 60).astype(int) + \
            0.1 * (X["num_reported_accidents"] > 2).astype(int)
        
        def clip(f):
            def clip_f(X):
                sigma = 0.05
                mu = f(X)
                a, b = -mu/sigma, (1-mu)/sigma
                Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
                phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
                return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
            return clip_f

        data["meta"] = clip(f)(data).values
            
        data[self.cat_features] = data[self.cat_features].astype('category')
        return data
    
    def log_transformation(self):
        self.y = np.log1p(self.y)         
        return self
        
    def remove_outliers(self):
        Q1 = self.y.quantile(0.25)
        Q3 = self.y.quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5*IQR
        upper_limit = Q3 + 1.5*IQR
        self.X = self.X[(self.y >= lower_limit) & (self.y <= upper_limit)]
        self.X.reset_index(drop=True, inplace=True)
    
    def missing_values(self):
        self.X[self.cat_features] = self.X[self.cat_features].fillna('NaN')
        self.test[self.cat_features] = self.test[self.cat_features].fillna('NaN')
        return self


X, y, test, cat_features, num_features = Preprocessing().fit_transform()


from sklearn.base import BaseEstimator
import contextlib, io
import ydf

class YDFModel(BaseEstimator, Config):
    def __init__(self, params=None):
        self.params = {} if params is None else params.copy()
        self.learner_class = (ydf.GradientBoostedTreesLearner if self.task_is_regression 
                              else ydf.RandomForestLearner)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        target = y.name
        params = self.params.copy()
        params['label'] = target
        params['task'] = ydf.Task.REGRESSION if self.task_is_regression else ydf.Task.CLASSIFICATION

        df = pd.concat([X, y], axis=1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model = self.learner_class(**params).train(df)

        if not self.task_is_regression:
            self.classes_ = list(y.unique())
            self.n_classes_ = len(self.classes_)

        return self

    def predict(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw_pred = self.model.predict(X)

        if self.task_is_regression:
            return raw_pred
        else:
            proba = np.asarray(raw_pred)
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            idx = proba.argmax(axis=1)
            return np.array(self.classes_)[idx]

    def predict_proba(self, X: pd.DataFrame):
        if self.task_is_regression:
            raise AttributeError("predict_proba is not available for regression task")
        assert isinstance(X, pd.DataFrame)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw_pred = self.model.predict(X)

        proba = np.asarray(raw_pred)
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        return proba


models = {
    'YDF': YDFModel({'num_trees': 1000,
                     'max_depth': 9,
                     'random_seed': Config.state,
                     'growing_strategy': 'BEST_FIRST_GLOBAL'
                    }),
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

        for n_fold, (train_id, valid_id) in enumerate(self.folds.split(X, y)):
            features = X.columns.to_list()

            X_train = X[features].loc[train_id].copy()
            y_train = y[train_id]
            X_val = X[features].iloc[valid_id].copy()
            y_val = y[valid_id]
            X_test = test[features].copy()             

            print(f'Fold {n_fold+1}')
            
            if "LGBM" in model_name:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    categorical_feature=self.cat_features,
                    feature_name="auto",
                    callbacks=[log_evaluation(0), early_stopping(self.early_stop, verbose=False)]
                )
            elif any(model in model_name for model in ['CAT', 'XGB', "NN", "TabM"]):
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            elif "HGB" in model_name:
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            elif "YDF" in model_name:
                model.fit(X_train, y_train)

            elif "Ensemble" in model_name:
                model.fit(X_train, y_train)
                
            else:
                encoder = FeatureEncoder(self.num_features, self.cat_features)
                encoder.fit(X)
                X_train, X_val, X_test = encoder.transform_fold(X_train, X_val, X_test)
            
                model.fit(X_train, y_train)

            if self.task_type == "regression" :
                y_pred_val = np.clip(model.predict(X_val), 0, 1)           
                test_pred += np.clip(model.predict(X_test), 0, 1) / self.n_splits
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
                oof_pred = pd.read_csv(f'/kaggle/input/risk-models/{model_name}_oof.csv')
                test_pred = pd.read_csv(f'/kaggle/input/risk-models/{model_name}_test.csv')
                oof_pred = np.clip(oof_pred, 0, 1)
                test_pred = np.clip(test_pred, 0, 1)
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
        plt.bar_label(hbars, fmt='%.7f')
        plt.xlim(0.04, 0.065)
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


trainer = Trainer(X, y, test, models, num_features, cat_features, training=True)
TEST_preds = trainer.run()


submission = Config.submission
submission[Config.target] = TEST_preds
submission.to_csv("submission.csv", index=False)

display(submission.head())
plt.figure(figsize=(14, 8))
sns.distplot(submission[Config.target], color='#3cb371', bins=100, hist_kws={'alpha': 0.7})
plt.show()

