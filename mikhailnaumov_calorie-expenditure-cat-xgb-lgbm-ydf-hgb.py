!pip install --upgrade scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, label_binarize, OrdinalEncoder, QuantileTransformer, TargetEncoder
from category_encoders import CatBoostEncoder, MEstimateEncoder

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LinearRegression, BayesianRidge, Ridge

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

import warnings
warnings.filterwarnings("ignore")


class Config:
    
    state = 42
    n_splits = 50
    early_stop = 100
        
    target = 'Calories'
    train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
    train_org = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv', index_col='User_ID')
    
    original_data = 'Y'
    outliers = 'N'
    log_trf = 'Y'
    feature_eng = 'Y'
    missing = 'N'


class EDA(Config):
    
    def __init__(self):
        super().__init__()

        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object']).columns.tolist()
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object']).columns.tolist()
        self.data_info()
        self.heatmap()
        self.dist_plots()
        self.cat_feature_plots()
        self.target_plot()
                
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
        plt.figure(figsize=(7,7))
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
        if Config.original_data == 'Y':
            self.train_org.rename(columns={'Gender': 'Sex'}, inplace=True)
            self.train = pd.concat([self.train, self.train_org], ignore_index=True).drop_duplicates()
            self.train.reset_index(drop=True, inplace=True)
            
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
        
        if self.missing == 'Y':
            self.missing_values()

        self.train_raw = self.train.copy()
        
        if self.feature_eng == 'Y':
            self.train = self.new_features(self.train)
            self.test = self.new_features(self.test)
            
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool', 'string']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
            
        if self.outliers == 'Y':    
            self.remove_outliers()
            
        if self.log_trf == 'Y':
            self.log_transformation()
            
        self.train_enc = self.train.copy()
        self.test_enc = self.test.copy()
        self.encode()
        
        if self.outliers == 'Y' or self.log_trf == 'Y':
            self.distribution()
        
    def __call__(self):
        self.train[self.cat_features] = self.train[self.cat_features].astype('category')
        self.test[self.cat_features] = self.test[self.cat_features].astype('category')
        self.y = self.train[self.target]
        self.X = self.train.drop(self.target, axis=1)
        self.X_enc = self.train_enc.drop(self.target, axis=1)
    
        return self.X, self.X_enc, self.y, self.test, self.test_enc, self.cat_features, self.num_features
    
    def encode(self):
        self.train_enc[self.num_features] = self.train_enc[self.num_features].fillna(self.train_enc[self.num_features].median())
        self.test_enc[self.num_features] = self.test_enc[self.num_features].fillna(self.test_enc[self.num_features].median())
        self.train_enc[self.cat_features] = self.train_enc[self.cat_features].fillna('NaN')
        self.test_enc[self.cat_features] = self.test_enc[self.cat_features].fillna('NaN')
        
        self.cat_features_card = []
        for f in self.cat_features:
            self.cat_features_card.append(self.train[f].nunique())
            
        data = pd.concat([self.train_enc, self.test_enc])
        oe = OrdinalEncoder()
        data[self.cat_features] = oe.fit_transform(data[self.cat_features]).astype('int')
        
        scaler = StandardScaler()
        data[self.num_features + [self.target]] = scaler.fit_transform(data[self.num_features + [self.target]])
        
        self.train_enc = data[~data[self.target].isna()]
        self.test_enc = data[data[self.target].isna()].drop(self.target, axis=1)
            
    def new_features(self, data):
        data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
        
        for c1, c2 in list(combinations(self.num_features,2)):
            data[f"{c1}_{c2}"] = data[c1]*data[c2]        
        return data

    def log_transformation(self):
        self.train[self.target] = np.log1p(self.train[self.target]) 
        
        return self
    
    def distribution(self):
        print(Style.BRIGHT+Fore.GREEN+f'\nHistograms of distribution\n')
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        ax_r, ax_n = axes

        ax_r.set_title(f'{self.target} ($\mu=$ {self.train_raw[self.target].mean():.2f} and $\sigma=$ {self.train[self.target].std():.2f} )')
        ax_r.hist(self.train_raw[self.target], bins=30, color='#3cb371')
        ax_r.axvline(self.train_raw[self.target].mean(), color='r', label='Mean')
        ax_r.axvline(self.train_raw[self.target].median(), color='y', linestyle='--', label='Median')
        ax_r.legend()

        ax_n.set_title(f'{self.target} Normalized ($\mu=$ {self.train_enc[self.target].mean():.2f} and $\sigma=$ {self.train_enc[self.target].std():.2f} )')
        ax_n.hist(self.train_enc[self.target], bins=30, color='#3cb371')
        ax_n.axvline(self.train_enc[self.target].mean(), color='r', label='Mean')
        ax_n.axvline(self.train_enc[self.target].median(), color='y', linestyle='--', label='Median')
        ax_n.legend()
        
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
X, X_enc, y, test, test_enc, cat_features, num_features = t()


def build_model(cat_features, num_features):
    
    x_input_cats = layers.Input(shape=(len(cat_features),))
    embs = []
    for j in range(len(cat_features)):
        e = layers.Embedding(t.cat_features_card[j], int(np.ceil(np.sqrt(t.cat_features_card[j]))))
        x = e(x_input_cats[:,j])
        x = layers.Flatten()(x)
        embs.append(x)
        
    x_input_nums = layers.Input(shape=(len(num_features),))
    
    x = layers.Concatenate(axis=-1)(embs+[x_input_nums]) 
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
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
    'XGB': [XGBRegressor(**{'tree_method': 'hist',
                            'n_estimators': 3000,
                            'objective': 'reg:squarederror',
                            'random_state': Config.state,
                            'enable_categorical': True,
                            'verbosity': 0,
                            'early_stopping_rounds': Config.early_stop,
                            'eval_metric': 'rmse',
                            'booster': 'gbtree',
                            "device": "cuda",
                            'n_jobs': -1,
                            'max_depth': 8,
                            'min_child_weight': 10, 
                            'subsample': 0.8260966788901262,
                            'reg_alpha': 0.27469472188551974, 
                            'reg_lambda': 0.5613776857654753, 
                            'colsample_bytree': 0.7965527339281658,
                            'learning_rate': 0.01,
                           }),
            False],
    'XGB2': [XGBRegressor(**{'max_depth': 10,
                             'colsample_bytree': 0.7,
                             'subsample': 0.9, 
                             'n_estimators': 2000,
                             'learning_rate': 0.02,
                             'gamma': 0.01,
                             'max_delta_step': 2,
                             'early_stopping_rounds': Config.early_stop,
                             'eval_metric': 'rmse',
                             'enable_categorical': True,
                             'random_state': Config.state,
                             'device': "cuda",
                            }), 
            False],
    'LGBM': [LGBMRegressor(**{'random_state': Config.state,
                              'early_stopping_round': Config.early_stop,
                              'categorical_feature': cat_features,
                              'verbose': -1,
                              'boosting_type': 'gbdt',
                              'n_estimators': 5000,
                              'eval_metric': 'rmse',
                              'objective': 'regression_l2',
                              "device": "gpu",
                              'learning_rate': 0.01,
                              'max_depth': 10,
                              'num_leaves': 928, 
                              'min_child_samples': 8,
                              'min_child_weight': 18, 
                              'colsample_bytree': 0.4009405711855729,
                              'reg_alpha': 0.22713546532680443,
                              'reg_lambda': 0.6266447966186705,
                              }),
             False],
    'CAT': [CatBoostRegressor(**{'verbose': 0,
                                 'random_state': Config.state,
                                 'cat_features': cat_features,
                                 'early_stopping_rounds': Config.early_stop,
                                 'eval_metric': "RMSE",
                                 'n_estimators' : 5000,
                                 'objective': 'RMSE', 
                                 'learning_rate': 0.01,
                                 "task_type": "GPU",
                              }),
            False],
    'HGB': [HistGradientBoostingRegressor(**{'max_iter': 5000,
                                             'random_state': Config.state,
                                             'early_stopping': True,
                                             'categorical_features': "from_dtype",
                                             'learning_rate': 0.01,
                                             'max_depth': 10,
                                             'loss': 'squared_error',
                                             'l2_regularization': 0.007634789701004957,
                                             'min_samples_leaf': 28,
                                             'max_leaf_nodes': 40
                                             }),
             False],
    'YDF': [YDFRegressor(GradientBoostedTreesLearner)({'num_trees': 1000,
                                                       'max_depth': 5,
                                                       'random_seed': Config.state,
                                                       'growing_strategy': 'BEST_FIRST_GLOBAL'
                                                       }),
            False],
}


class Model(Config):
    
    def __init__(self, X, X_enc, y, test, test_enc, models):
        self.y = y
        self.models = models
        self.scores = pd.DataFrame(columns=['Score'])
        self.OOF_preds = pd.DataFrame()
        self.TEST_preds = pd.DataFrame()
        
    def train(self):
        
        self.folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.state)
 
        for model_name, [model, training] in tqdm(self.models.items()):
            oof_pred = np.zeros(X.shape[0])
            test_pred = np.zeros(test.shape[0])
            if training:
                print('='*20)
                print(model_name)
                if any(model in model_name for model in ['LGBM', 'CAT', 'XGB', 'HGB', 'YDF']):
                    self.X = X.copy()
                    self.test = test.copy()
 
                else:
                    self.X = X_enc.copy()
                    self.test = test_enc.copy()
                    
                if 'NN' in model_name:
                    for n_fold, (train_id, valid_id) in enumerate(self.folds.split(self.X)):
                        X_train = self.X.loc[train_id].copy()
                        y_train = self.y.iloc[train_id]
                        X_val = self.X.loc[valid_id].copy()
                        y_val = self.y.iloc[valid_id]
                        
                        X_train_cats = X_train[cat_features]
                        X_train_nums = X_train[num_features]

                        X_val_cats = X_val[cat_features]
                        X_val_nums = X_val[num_features]

                        X_test_cats = self.test[cat_features]
                        X_test_nums = self.test[num_features]
                        print(f'Fold {n_fold+1}')
                        
                        model = build_model(cat_features, num_features)                        
                        keras.utils.set_random_seed(self.state)
                        optimizer = keras.optimizers.Adam(learning_rate=1e-2, weight_decay=1e-3)
                        model.compile(optimizer=optimizer, loss='mean_squared_error')
                        model.fit([X_train_cats,X_train_nums], y_train, 
                                  validation_data=([X_val_cats, X_val_nums], y_val),
                                  epochs=20,
                                  batch_size=1000,
                                  callbacks=[keras.callbacks.ReduceLROnPlateau(patience=1),
                                             keras.callbacks.EarlyStopping(patience=3)
                                            ])
                        
                        y_pred_val = model.predict([X_val_cats, X_val_nums])
                        oof_pred[valid_id] = y_pred_val.flatten()
                        test_pred += model.predict([X_test_cats, X_test_nums]).flatten() / self.n_splits
                        
                        score = root_mean_squared_error(y_val, y_pred_val)
                        print(score)
                        self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = score                 
                                              
                else:                          
                    for n_fold, (train_id, valid_id) in enumerate(self.folds.split(self.X, self.y)):
                        X_train = self.X.loc[train_id]
                        y_train = self.y.iloc[train_id]
                        X_val = self.X.loc[valid_id]
                        y_val = self.y.iloc[valid_id]  
                        X_test = self.test.copy()
                        
                        print(f'Fold {n_fold+1}')

                        if "XGB" in model_name:
                            model.fit(X_train, y_train, 
                                      eval_set = [(X_val, y_val)], 
                                      verbose = False
                                     )

                        elif "CAT" in model_name:
                            model.fit(X_train, y_train, 
                                      eval_set = [(X_val, y_val)],
                                      verbose=False
                                      ) 

                        elif "LGBM" in model_name:
                            model.fit(X_train, y_train, 
                                       eval_set = [(X_val, y_val)], 
                                       callbacks = [log_evaluation(0),
                                                    early_stopping(self.early_stop, verbose = False)
                                                   ])  

                        else:                           
                            model.fit(X_train, y_train)

                        y_pred_val = model.predict(X_val)
                        oof_pred[valid_id] = y_pred_val
                        test_pred += model.predict(X_test) / self.n_splits
                        
                        score = root_mean_squared_error(y_val, y_pred_val)
                        print(score)
                        self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = score

                self.OOF_preds[f'{model_name}'] = oof_pred
                self.TEST_preds[f'{model_name}'] = test_pred
                self.OOF_preds[f'{model_name}'].to_csv(f'{model_name}_oof.csv', index=False)
                self.TEST_preds[f'{model_name}'].to_csv(f'{model_name}_test.csv', index=False)
            
            else:
                self.OOF_preds[f'{model_name}'] = pd.read_csv(f'/kaggle/input/calories-models/{model_name}_oof.csv')
                self.TEST_preds[f'{model_name}'] = pd.read_csv(f'/kaggle/input/calories-models/{model_name}_test.csv')
                
                for n_fold, (train_id, valid_id) in enumerate(self.folds.split(self.OOF_preds[f'{model_name}'], self.y)):
                    y_pred_val, y_val = self.OOF_preds[f'{model_name}'].iloc[valid_id], self.y.iloc[valid_id]
                    self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = root_mean_squared_error(y_val, y_pred_val)
                    
            self.scores.loc[f'{model_name}', 'Score'] = self.scores.loc[f'{model_name}'][1:].mean()
        self.scores.loc['Ensemble', 'Score'], self.OOF_preds["Ensemble"], self.TEST_preds["Ensemble"] = self.ensemble(self.OOF_preds, self.y, self.TEST_preds)
        self.scores = self.scores.sort_values('Score')

        self.result()

        return self.TEST_preds
    
    def ensemble(self, X, y, test):
        scores = []
        oof_pred = np.zeros(X.shape[0])
        test_pred = np.zeros(test.shape[0])
        model = Ridge(tol=1e-2, max_iter=1000000)
        kf = KFold(n_splits=self.n_splits, random_state=self.state, shuffle=True)

        for fold_idx, (train_idx, val_idx) in enumerate(self.folds.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)
            oof_pred[val_idx] = y_pred_val
            test_pred += model.predict(test) / self.n_splits

            score = root_mean_squared_error(y_val, y_pred_val)

            scores.append(score)
                   
        return np.mean(scores), oof_pred, test_pred
    
    def result(self):
               
        plt.figure(figsize=(14, 8))
        colors = ['#3cb371' if i != 'Ensemble' else 'r' for i in self.scores.Score.index]
        hbars = plt.barh(self.scores.index, self.scores.Score, color=colors, height=0.8)
        plt.bar_label(hbars, fmt='%.5f')
        plt.ylabel('Models')
        plt.xlabel('Score')              
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].scatter(y, self.OOF_preds['Ensemble'], alpha=0.5, s=15, edgecolors='#3cb371')
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Actual vs. Predicted')

        axes[1].scatter(self.OOF_preds['Ensemble'], y - self.OOF_preds['Ensemble'], alpha=0.5, s=15, edgecolors='#3cb371')
        axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')

        plt.tight_layout()
        plt.show()


model = Model(X, X_enc, y, test, test_enc, models)
TEST_preds = model.train()


submission = Config.submission
blender = pd.read_csv('/kaggle/input/calories-models/blender.csv')
blender[f'{model.scores.loc["Ensemble", "Score"]:.6f}'] = np.clip(np.expm1(TEST_preds['Ensemble'].values), 1, 314)
submission[Config.target] = np.clip(blender.mean(axis=1), 1, 314)
submission.to_csv("submission.csv", index=False)

display(submission.head())
plt.figure(figsize=(14, 6))
submission[Config.target].hist(color='#3cb371', bins=50)
plt.show()

