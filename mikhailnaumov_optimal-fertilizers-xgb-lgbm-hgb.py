!pip install --upgrade scikit-learn scikit-learn==1.6.1 xgboost==3.0.1 lightgbm==4.6.0 numpy==1.26.4 scipy==1.14.1


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
    n_splits = 10
    early_stop = 50
        
    target = 'Fertilizer Name'
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
    train_org = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
    
    original_data = 'Y'
    outliers = 'N'
    log_trf = 'N'
    feature_eng = 'Y'
    missing = 'N'

    labels = list(train[target].unique())
    nclass = len(labels)


class EDA(Config):
    
    def __init__(self):
        super().__init__()

        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object']).columns.tolist()
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object']).columns.tolist()
        self.data_info()
        self.heatmap()
        self.dist_plots()
        self.cat_feature_plots()
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

    def target_pie(self):
        print(Style.BRIGHT+Fore.GREEN+f"\nTarget feature distribution\n")
        targets = self.train[self.target]
        plt.figure(figsize=(6, 6))
        plt.pie(targets.value_counts(), labels=targets.value_counts().index, autopct='%1.2f%%', colors=sns.color_palette('viridis', len(targets.value_counts())))
        plt.show()


eda = EDA()


class Transform(Config):
    
    def __init__(self):
        super().__init__()
        if Config.original_data == 'Y':
            original_copy = self.train_org.copy()
            for k in range(4):
                self.train_org = pd.concat([self.train_org,original_copy],axis=0)
            
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
        
        if self.missing == 'Y':
            self.missing_values()

        self.train_raw = self.train.copy()
        
        if self.feature_eng == 'Y':
            self.train = self.new_features(self.train)
            self.test = self.new_features(self.test)
            self.train_org = self.new_features(self.train_org)
            
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
        self.train_org[self.cat_features] = self.train_org[self.cat_features].astype('category')

        self.y = self.train[self.target]
        self.y = pd.DataFrame(label_binarize(self.y, classes=self.labels), columns=self.labels)
        self.y['Target'] = np.argmax(self.y[self.labels].values, axis=1)
        
        self.X = self.train.drop(self.target, axis=1)
        self.X_enc = self.train_enc.drop(self.target, axis=1)
        self.X = self.reduce_mem(self.X)
        self.test = self.reduce_mem(self.test)
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
        data[self.num_features] = scaler.fit_transform(data[self.num_features])
        
        self.train_enc = data[~data[self.target].isna()]
        self.test_enc = data[data[self.target].isna()].drop(self.target, axis=1)
            
    def new_features(self, data):
        for col in self.num_features:
            data[f'{col}_Binned'] = data[col].astype(str).astype('object')        
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
    x = layers.Dense(Config.nclass, activation='softmax')(x)
    
    model = keras.Model(inputs=[x_input_cats,x_input_nums], outputs=x)
    return model


from sklearn.base import BaseEstimator, ClassifierMixin
import contextlib, io
import ydf; ydf.verbose(2)
from ydf import RandomForestLearner

def YDFClassification(learner_class):

    class YDFXClassification(BaseEstimator, ClassifierMixin):

        def __init__(self, params={}):
            self.params = params

        def fit(self, X, y):
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            target = y.name
            params = self.params.copy()
            params['label'] = target
            params['task'] = ydf.Task.CLASSIFICATION
            X = pd.concat([X, y], axis=1)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.model = learner_class(**params).train(X)
            return self

        def predict_proba(self, X):
            assert isinstance(X, pd.DataFrame)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                return self.model.predict(X)
            
    return YDFXClassification


models = {
    'XGB': [XGBClassifier(**{'tree_method': 'hist',
                             'n_estimators': 5000,
                             'objective': 'multi:softprob',
                             'random_state': Config.state,
                             'enable_categorical': True,
                             'verbosity': 0,
                             'early_stopping_rounds': Config.early_stop,
                             'eval_metric': 'mlogloss',
                             'booster': 'gbtree',
                             "device": "cuda",
                             'n_jobs': -1,
                             'learning_rate': 0.1,
                             'num_class': Config.nclass,
                             'lambda': 0.05656209749983576,
                             'alpha': 5.620898657099113,
                             'colsample_bytree': 0.2587327850345624, 
                             'subsample': 0.8276149323901826,
                             'max_depth': 20,
                             'min_child_weight': 10,
                             'max_bin': 128
                           }),
            False],
    'XGB3': [XGBClassifier(**{'tree_method': 'gpu_hist',
                              'n_estimators': 5000,
                              'objective': 'multi:softprob',
                              'random_state': Config.state,
                              'enable_categorical': True,
                              'verbosity': 0,
                              'early_stopping_rounds': Config.early_stop,
                              'eval_metric': 'mlogloss',
                              'booster': 'gbtree',
                              "device": "cuda",
                              'n_jobs': -1,
                              'learning_rate': 0.1,
                              'num_class': Config.nclass,
                              'lambda': 9.502925972205883,
                              'alpha': 3.3445262697665257,
                              'gamma': 0.1002450169200304, 
                              'colsample_bytree': 0.30982086792950825,
                              'subsample': 0.45020106355947187,
                              'max_depth': 16,
                              'min_child_weight': 7
                               }),
        False],
    'LGBM': [LGBMClassifier(**{'random_state': Config.state,
                               'early_stopping_round': Config.early_stop,
                               'categorical_feature': cat_features,
                               'verbose': -1,
                               'n_estimators': 5000,
                               'eval_metric': 'multi_logloss',
                               'objective': 'multiclass',
                               'learning_rate': 0.1,
                               'num_class': Config.nclass,
                               'max_depth': 6,
                               'min_child_weight': 3.5475157004364806,
                               'min_child_samples': 13,
                               'subsample': 0.599755580554485,
                               'colsample_bytree': 0.32616721667826837,
                               'num_leaves': 353,
                               'reg_lambda': 59.63158675639122
                              }),
             False],
    'LGBM-goss': [LGBMClassifier(**{'random_state': Config.state,
                                    'early_stopping_round': 50,
                                    'categorical_feature': cat_features,
                                    'verbose': -1,
                                    'n_estimators': 10000,
                                    'eval_metric': 'multi_logloss',
                                    'objective': 'multiclass',
                                    'learning_rate': 0.1,
                                    'num_class': Config.nclass,
                                    'max_depth': 17,
                                    'min_child_weight': 1.651257115493134,
                                    'min_child_samples': 5,
                                    'subsample': 0.8111731065960308, 
                                    'colsample_bytree': 0.33818230959584533, 
                                    'num_leaves': 251, 
                                    'reg_lambda': 84.49548220501224,
                                    'boosting_type': 'goss'                                
                                    }),
         False],
    'HGB': [HistGradientBoostingClassifier(**{'max_iter': 5000,
                                              'random_state': Config.state,
                                              'early_stopping': True,
                                              'categorical_features': "from_dtype",
                                              'learning_rate': 0.1,
                                              'loss': 'log_loss',
                                              'l2_regularization': 2.786601939402124e-08,
                                              'max_depth': 5,
                                              'max_leaf_nodes': 37, 
                                              'min_samples_leaf': 75
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
        self.OOF_Ensemble = pd.DataFrame(columns=self.labels)
        self.TEST_Ensemble = pd.DataFrame(columns=self.labels)

        self.X_original = t.train_org.drop(columns=["Fertilizer Name"])
        y_original = t.train_org["Fertilizer Name"]
        self.y_original = pd.DataFrame(label_binarize(y_original, classes=self.labels), columns=self.labels)
        self.y_original['Target'] = np.argmax(self.y_original[self.labels].values, axis=1)
        
    def mapk(self, actual, predicted, k=3):
        def apk(a, p, k):
            p = p[:k]
            score = 0.0
            hits = 0
            seen = set()
            for i, pred in enumerate(p):
                if pred in a and pred not in seen:
                    hits += 1
                    score += hits / (i + 1.0)
                    seen.add(pred)
            return score / min(len(a), k)
        return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
    
    def train(self):
        
        self.folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.state)
 
        for model_name, [model, training] in tqdm(self.models.items()):
            oof_pred = np.zeros((X.shape[0], self.nclass))
            test_pred = np.zeros((test.shape[0], self.nclass))

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
                        optimizer = keras.optimizers.AdamW(learning_rate=1e-2, weight_decay=1e-3)
                        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
                        model.fit([X_train_cats,X_train_nums], y_train, 
                                  validation_data=([X_val_cats, X_val_nums], y_val),
                                  epochs=20,
                                  batch_size=1000,
                                  callbacks=[keras.callbacks.ReduceLROnPlateau(patience=1),
                                             keras.callbacks.EarlyStopping(patience=3)
                                            ])

                        y_pred_val = model.predict([X_val_cats, X_val_nums])                        
                        oof_pred[valid_id] = y_pred_val
                        test_pred += model.predict([X_test_cats, X_test_nums]) / self.n_splits

                        y_pred_val = np.argsort(y_pred_val, axis=1)[:, -3:][:, ::-1]
                        y_val = [[label] for label in y_val]
                        score = self.mapk(y_val, y_pred_val)
                        print(score)
                        self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = score                 
                                              
                else:                          
                    for n_fold, (train_id, valid_id) in enumerate(self.folds.split(self.X, self.y.Target)):
                        X_train = self.X.iloc[train_id]
                        y_train = self.y.Target.iloc[train_id]
                        X_val = self.X.iloc[valid_id]
                        y_val = self.y.Target.iloc[valid_id]  
                        X_test = self.test.copy()

                        X_train = pd.concat([X_train, self.X_original], axis=0, ignore_index=True)
                        y_train = pd.concat([y_train, self.y_original.Target], axis=0, ignore_index=True)
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

                        y_pred_val = model.predict_proba(X_val)                        
                        oof_pred[valid_id] = y_pred_val
                        test_pred += model.predict_proba(X_test) / self.n_splits

                        y_pred_val = np.argsort(y_pred_val, axis=1)[:, -3:][:, ::-1]
                        y_val = [[label] for label in y_val]
                        score = self.mapk(y_val, y_pred_val)
                        print(score)
                        self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = score

                oof_pred = pd.DataFrame(oof_pred, columns=self.labels)
                test_pred = pd.DataFrame(test_pred, columns=self.labels)
                oof_pred.to_csv(f'{model_name}_oof.csv', index=False)
                test_pred.to_csv(f'{model_name}_test.csv', index=False)
            
            else:

                oof_pred = pd.read_csv(f'/kaggle/input/fertilizers-models/{model_name}_oof.csv')
                test_pred = pd.read_csv(f'/kaggle/input/fertilizers-models/{model_name}_test.csv')
                for n_fold, (train_id, valid_id) in enumerate(self.folds.split(oof_pred, self.y.Target)):
                    y_pred_val, y_val = oof_pred.iloc[valid_id], self.y.Target.iloc[valid_id]
                    y_pred_val = y_pred_val.apply(lambda row: np.argsort(row.values)[-3:][::-1], axis=1)
                    y_val = [[label] for label in y_val]
                    self.scores.loc[f'{model_name}', f'Fold {n_fold+1}'] = self.mapk(y_val, y_pred_val)

            self.scores.loc[f'{model_name}', 'Score'] = self.scores.loc[f'{model_name}'][1:].mean()
            
            if len(self.models)>1:
                self.OOF_preds = pd.concat([self.OOF_preds, oof_pred.add_prefix(f'{model_name}_')], axis=1).reset_index(drop=True)
                self.TEST_preds = pd.concat([self.TEST_preds, test_pred.add_prefix(f'{model_name}_')], axis=1).reset_index(drop=True)

            else:
                self.OOF_Ensemble = oof_pred
                print(Style.BRIGHT+Fore.GREEN+f'{model_name} score {self.scores.loc[f"{model_name}", "Score"]:.5f}\n')
                self.result()
                return test_pred
                
        self.scores.loc['Ensemble', 'Score'], self.OOF_Ensemble, self.TEST_Ensemble = self.ensemble(self.OOF_preds, self.y.Target, self.TEST_preds)
        self.scores = self.scores.sort_values('Score')

        self.result()

        return self.TEST_Ensemble
    
    def ensemble(self, X, y, test):
        scores = []
        oof_pred = np.zeros((X.shape[0],self.nclass))
        test_pred = np.zeros((test.shape[0],self.nclass))
        model = XGBClassifier(objective='multi:softprob',
                              num_class=len(np.unique(y)),
                              learning_rate=0.03,
                              max_depth=8,
                              n_estimators=10000,
                              subsample=0.8,
                              colsample_bytree=0.45,
                              random_state=42,
                              eval_metric='mlogloss',
                              device='cuda',
                              tree_method='hist',
                              early_stopping_rounds=100
                              )
        for fold_idx, (train_idx, val_idx) in enumerate(self.folds.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred_val = model.predict_proba(X_val)            
            oof_pred[val_idx] = y_pred_val
            test_pred += model.predict_proba(test) / self.n_splits

            y_pred_val = np.argsort(y_pred_val, axis=1)[:, -3:][:, ::-1]
            y_val = [[label] for label in y_val]
            score = self.mapk(y_val, y_pred_val)

            scores.append(score)
                   
        return np.mean(scores), pd.DataFrame(oof_pred, columns=self.labels), pd.DataFrame(test_pred, columns=self.labels)
    
    def result(self):
               
        if len(self.models)>1:       
            plt.figure(figsize=(14, 8))
            colors = ['#3cb371' if i != 'Ensemble' else 'r' for i in self.scores.Score.index]
            hbars = plt.barh(self.scores.index, self.scores.Score, color=colors, height=0.5)
            plt.bar_label(hbars, fmt='%.5f')
            plt.xlim(0.1,0.4)
            plt.ylabel('Models')
            plt.xlabel('Score')              
            plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for i, col in enumerate(self.labels):
            RocCurveDisplay.from_predictions(self.y[col].sort_index(), self.OOF_Ensemble[col], name=f"{col}", ax=axes[0])
            
        axes[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC')
        axes[0].legend(loc="lower right")
        
        ConfusionMatrixDisplay.from_predictions(self.y.Target.sort_index(), np.argmax(self.OOF_Ensemble, axis=1), display_labels=self.labels, xticks_rotation='vertical', colorbar=False, ax=axes[1], cmap = 'Greens')
        axes[1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.show()


model = Model(X, X_enc, y, test, test_enc, models)
TEST_preds = model.train()


submission = Config.submission
blender = pd.read_csv('/kaggle/input/fertilizers-models/blender.csv')
blender[f'{model.scores.loc["Ensemble", "Score"]:.6f}'] = TEST_preds.apply(lambda row: ' '.join(row.nlargest(3).index), axis=1)
submission[Config.target] = blender.mode(axis=1).iloc[:, 0]
submission.to_csv("submission.csv", index=False)

display(submission.head())
counts = submission[Config.target].value_counts().sample(100)
plt.figure(figsize=(20, 10))
sns.barplot(x=counts.index, y=counts.values, color='#3cb371', width=0.9)
plt.xticks(rotation=90)
plt.show()

