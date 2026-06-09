import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, label_binarize, OrdinalEncoder
from category_encoders import CatBoostEncoder, MEstimateEncoder

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LinearRegression, BayesianRidge, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn import set_config
import os

import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error, precision_recall_curve, make_scorer, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, matthews_corrcoef
import scipy.stats as stats

from colorama import Fore, Style, init
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from pprint import pprint
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, KFold, RepeatedKFold, cross_val_score, StratifiedGroupKFold
from sklearn.ensemble import StackingClassifier
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
from cuml.preprocessing import TargetEncoder
import gc
import re

import keras
from keras.models import Sequential
from keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

from cuml.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")


class Config:
    
    state = 42
    n_splits = 5
    early_stop = 200
        
    target = 'Calories'
    train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
    train_org = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv', index_col='User_ID')
    train_org.rename(columns={'Gender':'Sex'}, inplace=True)
    original_data = 'N'
    outliers = 'N'
    scaler_trf = 'Y'
    trans_target = 'Y'
    feature_eng = 'Y'
    missing = 'Y'


class EDA(Config):
    
    def __init__(self):
        super().__init__()

        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object']).columns.tolist()
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object']).columns.tolist()
        self.data_info()
        self.heatmap()
        self.dist_plots()
        #self.cat_feature_plots()
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

        #plt.tight_layout()
        plt.show()
               
    def cat_feature_plots(self):
        fig, axes = plt.subplots(len(self.cat_features), 2 ,figsize = (18, len(self.cat_features) * 6), 
                                 gridspec_kw = {'hspace': 0.5, 
                                                'wspace': 0.2,
                                               }
                                )

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
        print(Style.BRIGHT+Fore.GREEN+f"\nTarget label distribution\n")
        
        fig, ax = plt.subplots(1, 1 ,figsize = (14, 6), 
                                 gridspec_kw = {'hspace': 0.3, 
                                                'wspace': 0.2, 
                                                'width_ratios': [1.0]
                                               }
                                )
        sns.barplot(data=self.train[self.target].value_counts().nlargest(10).reset_index(), x=self.target, y='count', ax=ax, color='#3cb371')
        #sns.kdeplot(data = self.train[self.target], 
        #            color = '#3cb371', ax = ax, linewidth = 2
        #           )
        ax.set(xlabel = '', ylabel = '')
        ax.set_title(f"\n{self.target}")
        ax.grid()

        #plt.tight_layout()
        plt.show()    


eda = EDA()


class Transform(Config):
    
    def __init__(self):
        super().__init__()
        if Config.original_data == 'Y':
            self.train = pd.concat([self.train, self.train_org], ignore_index=True).drop_duplicates()
            self.train.reset_index(drop=True, inplace=True)
        else:
            cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
            self.train = self.train.drop_duplicates(subset=self.train.columns, keep='first').reset_index(drop=True)
            self.train = self.train.groupby(by=cols)['Calories'].min().reset_index()

        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
        
        self.train_raw = self.train.copy()
        
        if self.missing == 'Y':
            self.missing_values()
        
        if self.feature_eng == 'Y':
            self.train = self.new_features(self.train)
            self.test = self.new_features(self.test)
            self.train_raw = self.new_features(self.train_raw)
        
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
            
        if self.outliers == 'Y':    
            self.remove_outliers()
        
        if self.scaler_trf == 'Y':
            self.scaler()
            
        self.train_enc = self.train.copy()
        self.test_enc = self.test.copy()
        self.encode()
        
        if self.outliers == 'Y' or self.scaler_trf =='Y':
            self.distribution()

        if self.trans_target == 'Y':
            self.log_transformation()

    def __call__(self):
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
        print(f'num_features:{self.num_features}')
        print(f'cat_features:{self.cat_features}')
        
        self.cat_features_card = []
        for f in self.cat_features:
            self.cat_features_card.append(self.train[f].nunique())

        print('target isna: ', self.train[self.target].isna().sum())
        self.train.dropna(subset=[Config.target],inplace=True)
        self.y = self.train[self.target]
        self.train = self.train.drop(self.target, axis=1)
        self.train_enc = self.train_enc.drop(self.target, axis=1)
        
        return self.train, self.train_enc, self.y, self.test, self.test_enc, self.cat_features
    
    def encode(self):
        data = pd.concat([self.test, self.train])
        oe = OrdinalEncoder()
        data[self.cat_features] = oe.fit_transform(data[self.cat_features]).astype('int')
        
        scaler = StandardScaler()
        data[self.num_features + [self.target]] = scaler.fit_transform(data[self.num_features + [self.target]])
        
        self.train_enc = data[~data[self.target].isna()]
        self.test_enc = data[data[self.target].isna()].drop(self.target, axis=1)
            
    def new_features(self, data):
        data['bmi'] = data['Height'] / (data['Weight'] ** 2)
        data['bti'] = data['Height'] / (data['Weight'] ** 0.5)
        for i in range(len(self.num_features)):
            for j in range(i+1, len(self.num_features)):
                data[f'{self.num_features[i]}_{self.num_features[j]}'] = data[self.num_features[i]] / (data[self.num_features[j]] + 1e-6)
                data[f'{self.num_features[i]}_{self.num_features[j]}_diff'] = data[self.num_features[i]] - data[self.num_features[j]]
                data[f'{self.num_features[i]}_{self.num_features[j]}_sum'] = data[self.num_features[i]] + data[self.num_features[j]]
                data[f'{self.num_features[i]}_{self.num_features[j]}_mul'] = data[self.num_features[i]] * (data[self.num_features[j]] + 1e-6)
        
        return data

    def log_transformation(self):
        self.train[self.target] = np.log1p(self.train[self.target]) 
        
        return self

    def sqrt_transformation(self):
        self.train[self.target] = np.power(self.train[self.target], 0.5)
        return self
    
    def distribution(self):
        print(Style.BRIGHT+Fore.GREEN+f'\nHistograms of distribution\n')
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        ax_r, ax_n = axes

        ax_r.set_title(f'{self.target} ($\mu=$ {self.train_raw[self.target].mean():.2f} and $\sigma=$ {self.train_raw[self.target].std():.2f} )')
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
        
    def scaler(self):
        scaler = StandardScaler()
        self.train[self.num_features] = scaler.fit_transform(self.train[self.num_features])
        self.test[self.num_features] = scaler.transform(self.test[self.num_features])
        return self
    
    def missing_values(self):
        for col in self.test.columns:
            if col in self.num_features:
                self.train[col] = self.train[col].fillna(self.train[col].median())
                self.test[col] = self.test[col].fillna(self.test[col].median())
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
X, X_enc, y, test, test_enc, cat_features = t()


models = {
    'CAT': [CatBoostRegressor(**{'verbose': 0,
                                 'random_state': Config.state,
                                 'loss_function': 'RMSE',
                                 'eval_metric': 'RMSE',
                                 'cat_features': cat_features,
                                 'task_type':'CPU'
                                }),
            True],
    'CAT2': [CatBoostRegressor(**{'verbose': 0,
                                 'random_state': Config.state,
                                 'loss_function': 'RMSE',
                                 'eval_metric': 'RMSE',
                                 'cat_features': cat_features,
                                 'task_type':'CPU',
                                 'iterations': 2000,
                                 'learning_rate': 0.013827929228103705,
                                 'depth': 11,
                                 'l2_leaf_reg': 0.28124494560152086,
                                 'border_count': 223, 
                                 'bagging_temperature': 0.6817302202614061, 
                                 'random_strength': 0.11363827435224598, 
                                 'grow_policy': 'Depthwise', 
                                 'min_data_in_leaf': 17
                                }),
            True],
    'XGB': [XGBRegressor(**{'random_state': Config.state,
                            'n_estimators': 2000,
                            'max_depth': 10,
                            'learning_rate': 0.02,
                            'subsample': 0.9,
                            'colsample_bytree': 0.7,
                            'gamma': 0.01,
                            'random_state': 42,
                            'max_delta_step': 2, 
                            'early_stopping_rounds': 100,
                            'eval_metric': 'rmse',
                            'n_jobs': -1,
                            }),
            True],
}


class Model(Config):
    
    def __init__(self, X, X_enc, y, test, test_enc, models, trf_target):
        self.X = X
        self.X_enc = X_enc
        self.y = y
        self.test = test
        self.test_enc = test_enc
        self.models = models
        self.trf_target = trf_target
        self.scores = pd.DataFrame(columns=['Score'])
        self.OOF_preds = pd.DataFrame()
        self.TEST_preds = pd.DataFrame()

    def train(self):
        
        self.folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.state)
 
        for model_name, [model, training] in tqdm(self.models.items()):
            
            if training:
                print('='*20)
                print(model_name)
                if any(model in model_name for model in ['LGBM', 'LGBM2', 'CAT', 'CAT2', 'XGB', 'RF', 'KNN']):
                    self.X = X.fillna(0.0)
                    self.test = test.fillna(0.0)
                    full_df = pd.concat([self.X, self.test], ignore_index=True)
                    for col in cat_features:
                        le = LabelEncoder()
                        print(full_df[col].values)
                        le.fit_transform([str(x) for x in full_df[col].values])
                        self.X[col] = le.transform(self.X[col])
                        test[col] = le.transform(test[col])
                        test[col] = test[col].astype(int)
                else:
                    self.X = X_enc
                    self.test = test_enc
                
                for n_fold, (train_id, valid_id) in enumerate(self.folds.split(self.X, self.y)):
                    X_train, y_train = self.X.iloc[train_id].copy(), self.y.iloc[train_id]
                    X_val, y_val = self.X.iloc[valid_id].copy(), self.y.iloc[valid_id]
                    self.test = test.copy()
                    
                    # TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
                    # for col in self.X.columns:
                    #     X_train[f"TE_{col}"] = TE.fit_transform(X_train[col], y_train)
                    #     X_val[f"TE_{col}"] = TE.transform(X_val[col])
                    #     self.test[f"TE_{col}"] = TE.transform(self.test[col])

                    # X_train[cat_features] = X_train[cat_features].fillna('Missing').astype('category')
                    # X_val[cat_features] = X_val[cat_features].fillna('Missing').astype('category')
                    # self.test[cat_features] = self.test[cat_features].fillna('Missing').astype('category')
                    
                    oof_preds = pd.DataFrame(columns=[model_name], index=X_val.index)
                    test_preds = pd.DataFrame(columns=[model_name], index=test.index)
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
                    elif "KNN" in model_name:
                        model.fit(X_train.values, y_train.values)
                    else:
                        model.fit(X_train, y_train)

                    if "KNN" in model_name:
                        y_pred_val = model.predict(X_val.values)
                        test_pred = model.predict(self.test.values)
                    else:
                        y_pred_val = model.predict(X_val)
                        test_pred = model.predict(self.test)

                    score = mean_squared_error(y_val, y_pred_val)
                    print(score)
                    self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = score

                    oof_preds[model_name] = y_pred_val
                    test_preds[model_name] = test_pred
                    self.OOF_preds = pd.concat([self.OOF_preds, oof_preds], axis = 0, ignore_index = False)
                    self.TEST_preds = pd.concat([self.TEST_preds, test_preds], axis = 0, ignore_index = False)

                self.OOF_preds = self.OOF_preds.groupby(level=0).mean()
                self.TEST_preds = self.TEST_preds.groupby(level=0).mean()

                self.OOF_preds[f'{model_name}'].to_csv(f'{model_name}_oof.csv', index=False)
                self.TEST_preds[f'{model_name}'].to_csv(f'{model_name}_test.csv', index=False)
            
            else:
                self.OOF_preds[f'{model_name}'] = pd.read_csv(f'/kaggle/input/playground-series-s5e5-oof/v1_results/{model_name}_oof.csv')
                self.TEST_preds[f'{model_name}'] = pd.read_csv(f'/kaggle/input/playground-series-s5e5-oof/v1_results/{model_name}_test.csv')
                
                for n_fold, (train_id, valid_id) in enumerate(self.folds.split(self.OOF_preds[f'{model_name}'], self.y)):
                    y_pred_val, y_val = self.OOF_preds[f'{model_name}'].iloc[valid_id], self.y.iloc[valid_id]
                    self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = mean_squared_error(y_val, y_pred_val)
                    
            self.scores.loc[f'{model_name}', 'Score'] = self.scores.loc[f'{model_name}'][1:].mean()
        self.scores.loc['Ensemble'], self.OOF_preds["Ensemble"], self.TEST_preds["Ensemble"] = self.ensemble(self.OOF_preds, self.y, self.TEST_preds)
        self.scores = self.scores.sort_values('Score')

        return self.TEST_preds
    
    def ensemble(self, X, y, test):
        scores = []
        oof_pred = np.zeros(X.shape[0])
        test_pred = np.zeros(test.shape[0])
        ridge_params = {"alpha": 1.01, "random_state": Config.state}
        model = Ridge(**ridge_params)
        for fold_idx, (train_idx, val_idx) in enumerate(self.folds.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            print('y_train:', y_train)
            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(test)
            
            score = mean_squared_error(y_val, y_pred_val)
            scores.append(score)
            oof_pred[val_idx] = y_pred_val
            test_pred += y_pred_test / self.n_splits

        if self.trf_target == 'Y':
            oof_pred = np.expm1(oof_pred)
            test_pred = np.expm1(test_pred)
        return np.mean(scores), oof_pred, test_pred


model = Model(X, X_enc, y, test, test_enc, models, trf_target='Y')
TEST_preds = model.train()


submission = Config.submission
submission[Config.target] = y_preds = np.clip(TEST_preds['Ensemble'].values, 1, 314)
submission.to_csv("submission.csv", index=False)

display(submission.head())
plt.figure(figsize=(14, 6))
submission[Config.target].hist(color='#3cb371', bins=50)
plt.show()

