# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from scipy.stats import skew, kurtosis, mode

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

import optuna

import category_encoders as ce

import os
import sys
import contextlib

# TensorFlow imports for Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from warnings import simplefilter

simplefilter("ignore")

rc_params = {'legend.fontsize': 6,
             'axes.labelsize': 8,
             'axes.titlesize':8,
             'xtick.labelsize':6,
             'ytick.labelsize':6,
             'figure.figsize': [8, 6]
            }
plt.rcParams.update(rc_params)
print('All Done!')


# Check if GPU is available using torch (if installed)
try:
    import torch
    gpu_available = torch.cuda.is_available()
except ImportError:
    gpu_available = False

print(f"GPU available: {gpu_available}")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col=0)
original_df = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv').drop('User_ID', axis=1)
original_df = original_df.rename(columns={'Gender': 'Sex'})
sub_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv', index_col=0)
print('train data shape: ', train_df.shape)
display(train_df.head(4))
print('\noriginal data shape: ', original_df.shape)
display(original_df.head(4))
print('\ntest data shape: ', test_df.shape)
display(test_df.head(4))
print('\nsample submission data shape: ', sub_df.shape)
display(sub_df.head(4))


train_info = pd.concat([train_df.dtypes, train_df.count(), train_df.nunique(), train_df.isnull().sum()], axis=1, keys=['dype', 'count', 'nunique', 'missing'])
original_info = pd.concat([original_df.dtypes, original_df.count(), original_df.nunique(), original_df.isnull().sum()], axis=1, keys=['dype', 'count', 'nunique', 'missing'])
test_info = pd.concat([test_df.dtypes, test_df.count(), test_df.nunique(), test_df.isnull().sum()], axis=1, keys=['dype', 'count', 'nunique', 'missing'])
pd.concat([train_info, original_info, test_info], axis=1, keys=['train data', 'original data',  'test data'])


# Declare features and target
FEATURES = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
NUMERICAL_FEATURES = [col for col in FEATURES if train_df[col].dtype == float or train_df[col].dtype==int]
CATEGORICAL_FEATURES = [col for col in FEATURES if train_df[col].dtype == object]
TARGET = 'Calories'
print(f'Number of features: {len(FEATURES)}\nNumber of numerical featues: {len(NUMERICAL_FEATURES)}\nNumber of categorical features: {len(CATEGORICAL_FEATURES)}\nThe Target has a type of: {train_df[TARGET].dtype}')


# Correlation matrix
corr = train_df[NUMERICAL_FEATURES + [TARGET]].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr[mask] = np.nan
(corr
 .style
 .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
 .highlight_null(color='#f1f1f1')  # Color NaNs grey
 .format(precision=2))


palette = sns.color_palette('deep', n_colors=train_df['Sex'].nunique())
for column in NUMERICAL_FEATURES + [TARGET]:
    fig = plt.figure(figsize=(12, 2))
    # Set up gridspec with 3 columns
    gs = gridspec.GridSpec(1, 3,
                           width_ratios = [2, 2, 1]
                          )
    # Create a figure
    ax0 = fig.add_subplot(gs[0])  # Histogram
    ax1 = fig.add_subplot(gs[1])  # Scatter plot
    ax2 = fig.add_subplot(gs[2])  # Box plot
    
    sns.histplot(train_df.sample(frac=0.01)[column], stat='density', kde=True, ax=ax0)
    ax0.set_title(f'Histogram of {column}')

    sns.scatterplot(train_df.sample(frac=0.01),
                y=TARGET,
                x=column,
                ax=ax1,
                s=1,
    )
    for i, (cat, group) in enumerate(train_df.sample(frac=0.01).groupby('Sex')):
        sns.regplot(
            data=group,
            x=column,
            y=TARGET,
            scatter=False,
            line_kws={'color': palette[i], 'label': f'{cat}', 'lw': 1},
            ax=ax1
        )
        ax1.plot([], [], color=palette[i], label=f'Sex=={cat}')
    ax1.legend()
    ax1.set_title(f'Scatter plot of Sex per {column}')
    
    sns.boxplot(train_df[column], ax=ax2)
    ax2.set_title(f'BoxPlot of {column}')


# Evaluation metric: RMSE in log-space
def rmse_in_log(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Optional feature engineering: Generate polynomial interaction features
def feature_engineering(X, degree=2):
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_new = np.hstack([X.values, X_poly])
    new_cols = list(X.columns) + [f"poly_{i}" for i in range(X_poly.shape[1])]
    return pd.DataFrame(X_new, columns=new_cols, index=X.index)

# Build a simple neural network using Keras
def build_nn_model(input_dim, units=64, dropout=0.2, num_layers=2, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=input_dim))
    if dropout > 0:
        model.add(Dropout(dropout))
    for _ in range(num_layers - 1):
        model.add(Dense(units, activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Preprocessor class to handle sampling, target extraction and mapping
class DataPreprocessor:
    def __init__(self, train_df, test_df, target, mapping_cols=None, sample_frac=None, apply_poly=False, poly_degree=2, random_state=42):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.target = target
        self.mapping_cols = mapping_cols or {}
        self.sample_frac = sample_frac
        self.apply_poly = apply_poly
        self.poly_degree = poly_degree
        self.random_state = random_state
        
    def preprocess(self):
        if self.sample_frac is not None:
            df_sample = self.train_df.sample(frac=self.sample_frac, random_state=self.random_state)
        else:
            df_sample = self.train_df

        X = df_sample.copy()
        y = np.log1p(X.pop(self.target))
        
        for col, mapping in self.mapping_cols.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
            if col in self.test_df.columns:
                self.test_df[col] = self.test_df[col].map(mapping)
        if self.apply_poly:
            X = feature_engineering(X, degree=self.poly_degree)
            poly = PolynomialFeatures(degree=self.poly_degree, interaction_only=True, include_bias=False)
            X_test_poly = poly.fit_transform(self.test_df)
            self.test_df = pd.DataFrame(np.hstack([self.test_df.values, X_test_poly]), 
                                        columns=list(self.test_df.columns) + [f"poly_{i}" for i in range(X_test_poly.shape[1])],
                                        index=self.test_df.index)
        X_test = self.test_df.copy()
        return X, y, X_test

# Model optimizer uses Optuna to perform hyperparameter tuning via cross-validation
class ModelOptimizer:
    def __init__(self, X, y, n_splits=3, n_trials=30, random_state=42):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state
    
    def optimize_lgbm(self):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': self.random_state,
                'verbose': -1,
                'device': 'gpu'
            }
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scores = []
            for train_idx, valid_idx in cv.split(self.X):
                X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[valid_idx]
                model = LGBMRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(rmse_in_log(y_val, preds))
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def optimize_xgb(self):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': self.random_state,
                'tree_method': 'gpu_hist'
            }
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scores = []
            for train_idx, valid_idx in cv.split(self.X):
                X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[valid_idx]
                model = XGBRegressor(verbosity=0, **params)
                model.fit(X_tr, y_tr, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                scores.append(rmse_in_log(y_val, preds))
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def optimize_rf(self):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.random_state,
                'n_jobs': -1,
            }
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scores = []
            for train_idx, valid_idx in cv.split(self.X):
                X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[valid_idx]
                model = RandomForestRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(rmse_in_log(y_val, preds))
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params
    
    def optimize_cat(self):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
                'depth': trial.suggest_int('depth', 3, 16),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': self.random_state,
                'task_type': 'GPU'
            }
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scores = []
            for train_idx, valid_idx in cv.split(self.X):
                X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[valid_idx]
                model = CatBoostRegressor(verbose=False, **params)
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(rmse_in_log(y_val, preds))
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def optimize_et(self):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.random_state,
                'n_jobs': -1,
            }
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scores = []
            for train_idx, valid_idx in cv.split(self.X):
                X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[valid_idx]
                model = ExtraTreesRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(rmse_in_log(y_val, preds))
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def optimize_nn(self):
        def objective(trial):
            units = trial.suggest_int("units", 16, 128)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            epochs = trial.suggest_int("epochs", 20, 50)
            batch_size = trial.suggest_int("batch_size", 16, 64)
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scores = []
            for train_idx, valid_idx in cv.split(self.X):
                X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[valid_idx]
                model = build_nn_model(input_dim=X_tr.shape[1],
                                       units=units, dropout=dropout,
                                       num_layers=num_layers, learning_rate=learning_rate)
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, 
                          batch_size=batch_size, verbose=0, callbacks=[early_stop])
                preds = model.predict(X_val).flatten()
                scores.append(rmse_in_log(y_val, preds))
            return np.mean(scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def optimize_lr(self):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        for train_idx, valid_idx in cv.split(self.X):
            X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            model = LinearRegression()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(rmse_in_log(y_val, preds))
        return {}  # No hyperparameters for LinearRegression

# EnsembleTrainer class trains each model using CV, collects OOF and test predictions, and provides ensemble methods.
class EnsembleTrainer:
    def __init__(self, X, y, X_test, cv_folds=5, random_state=42):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.cv_folds = cv_folds
        self.random_state = random_state
        # Keys: 'lgbm','xgb','rf','cat','et','nn','lr'
        self.best_params = {}
        self.oof_preds = {}
        self.test_preds = {}
    
    def set_best_params(self, params_dict):
        self.best_params = params_dict
        
    def fit(self):
        n_samples = len(self.X)
        n_test = len(self.X_test)
        model_keys = ['lgbm', 'xgb', 'rf', 'cat', 'et', 'nn', 'lr']
        for key in model_keys:
            self.oof_preds[key] = np.zeros(n_samples)
            self.test_preds[key] = np.zeros(n_test)
            
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.X), 1):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            
            # LGBM
            model_lgbm = LGBMRegressor(**self.best_params.get('lgbm', {}))
            model_lgbm.fit(X_train, y_train)
            preds_lgbm = model_lgbm.predict(X_val)
            self.oof_preds['lgbm'][valid_idx] = preds_lgbm
            self.test_preds['lgbm'] += model_lgbm.predict(self.X_test) / self.cv_folds
            
            # XGBoost
            model_xgb = XGBRegressor(verbosity=0, **self.best_params.get('xgb', {}))
            model_xgb.fit(X_train, y_train)
            preds_xgb = model_xgb.predict(X_val)
            self.oof_preds['xgb'][valid_idx] = preds_xgb
            self.test_preds['xgb'] += model_xgb.predict(self.X_test) / self.cv_folds
            
            # RandomForest
            model_rf = RandomForestRegressor(**self.best_params.get('rf', {}))
            model_rf.fit(X_train, y_train)
            preds_rf = model_rf.predict(X_val)
            self.oof_preds['rf'][valid_idx] = preds_rf
            self.test_preds['rf'] += model_rf.predict(self.X_test) / self.cv_folds
            
            # CatBoost
            model_cat = CatBoostRegressor(**self.best_params.get('cat', {}))
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model_cat.fit(X_train, y_train)
            preds_cat = model_cat.predict(X_val)
            self.oof_preds['cat'][valid_idx] = preds_cat
            self.test_preds['cat'] += model_cat.predict(self.X_test) / self.cv_folds
            
            # ExtraTrees
            model_et = ExtraTreesRegressor(**self.best_params.get('et', {}))
            model_et.fit(X_train, y_train)
            preds_et = model_et.predict(X_val)
            self.oof_preds['et'][valid_idx] = preds_et
            self.test_preds['et'] += model_et.predict(self.X_test) / self.cv_folds
            
            # Neural Network
            nn_params = self.best_params.get('nn', {'units': 64, 'dropout': 0.2, 'num_layers': 2, 'learning_rate': 0.001, 'epochs': 50, 'batch_size': 32})
            model_nn = build_nn_model(input_dim=X_train.shape[1],
                                       units=nn_params['units'],
                                       dropout=nn_params['dropout'],
                                       num_layers=nn_params['num_layers'],
                                       learning_rate=nn_params['learning_rate'])
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model_nn.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=nn_params['epochs'],
                         batch_size=nn_params['batch_size'],
                         verbose=0, callbacks=[early_stop])
            preds_nn = model_nn.predict(X_val).flatten()
            self.oof_preds['nn'][valid_idx] = preds_nn
            self.test_preds['nn'] += model_nn.predict(self.X_test).flatten() / self.cv_folds
            
            # Linear Regression
            model_lr = LinearRegression()
            model_lr.fit(X_train, y_train)
            preds_lr = model_lr.predict(X_val)
            self.oof_preds['lr'][valid_idx] = preds_lr
            self.test_preds['lr'] += model_lr.predict(self.X_test) / self.cv_folds
            
            print(f"Fold {fold} RMSLEs:")
            for key in model_keys:
                error = rmse_in_log(y_val, self.oof_preds[key][valid_idx])
                print(f"  {key.upper()}: {error:.4f}")
    
    # Equal weight ensemble: returns tuple (OOF error, test predictions)
    def equal_weight_ensemble(self):
        keys = list(self.oof_preds.keys())
        oof_eq = sum(self.oof_preds[k] for k in keys) / len(keys)
        test_eq = sum(self.test_preds[k] for k in keys) / len(keys)
        error = rmse_in_log(self.y, oof_eq)
        return error, test_eq
    
    # Stacking ensemble using a meta-model: returns tuple (OOF error, test predictions)
    def stacking_ensemble(self, meta_model=None):
        keys = ['lgbm', 'xgb', 'rf', 'cat', 'et', 'nn', 'lr']
        meta_features = np.vstack([self.oof_preds[k] for k in keys]).T
        if meta_model is None:
            meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_features, self.y)
        oof_meta = meta_model.predict(meta_features)
        error = rmse_in_log(self.y, oof_meta)
        test_meta_features = np.vstack([self.test_preds[k] for k in keys]).T
        test_meta = meta_model.predict(test_meta_features)
        return error, test_meta
    
    # Hill climbing ensemble to optimize weights: returns tuple (OOF error, test predictions)
    def hill_climbing_ensemble(self, step=0.01, max_iter=1000):
        keys = ['lgbm', 'xgb', 'rf', 'cat', 'et', 'nn', 'lr']
        meta_features = np.vstack([self.oof_preds[k] for k in keys]).T
        best_weights, best_score = self._hill_climbing_weights(meta_features, self.y, step, max_iter)
        test_meta_features = np.vstack([self.test_preds[k] for k in keys]).T
        test_hc = np.dot(test_meta_features, best_weights)
        return best_score, test_hc
    
    # Performance weighted ensemble: returns tuple (OOF error, test predictions)
    def performance_weighted_ensemble(self, epsilon=1e-6):
        models = ['lgbm', 'xgb', 'rf', 'cat', 'et', 'nn', 'lr']
        inv_errors = {}
        for m in models:
            error = rmse_in_log(self.y, self.oof_preds[m])
            inv_errors[m] = 1 / (error + epsilon)
        total_inv = sum(inv_errors.values())
        weights = {m: inv_errors[m] / total_inv for m in models}
        oof_weighted = sum(self.oof_preds[m] * weights[m] for m in models)
        error = rmse_in_log(self.y, oof_weighted)
        test_weighted = sum(self.test_preds[m] * weights[m] for m in models)
        return error, test_weighted
    
    @staticmethod
    def _hill_climbing_weights(base_preds, y_true, step, max_iter):
        n_models = base_preds.shape[1]
        weights = np.ones(n_models) / n_models
        best_weights = weights.copy()
        best_score = np.sqrt(np.mean((y_true - np.dot(base_preds, weights)) ** 2))
        improvement = True
        iterations = 0
        while improvement and iterations < max_iter:
            improvement = False
            iterations += 1
            for i in range(n_models):
                for delta in [-step, step]:
                    new_weights = weights.copy()
                    new_weights[i] += delta
                    new_weights = new_weights / np.sum(new_weights)
                    score = np.sqrt(np.mean((y_true - np.dot(base_preds, new_weights)) ** 2))
                    if score < best_score:
                        best_score = score
                        weights = new_weights
                        best_weights = new_weights.copy()
                        improvement = True
        return best_weights, best_score

    # Select the best ensemble (with the lowest OOF error) for submission.
    # Returns final test predictions (after applying expm1 if calibration is enabled).
    def select_best_for_submission(self, calibrate=True):
        eq_error, eq_test = self.equal_weight_ensemble()
        stack_error, stack_test = self.stacking_ensemble()
        hill_error, hill_test = self.hill_climbing_ensemble()
        perf_error, perf_test = self.performance_weighted_ensemble()
        
        errors = {
            'equal': eq_error,
            'stacking': stack_error,
            'hill': hill_error,
            'performance': perf_error
        }
        best_method = min(errors, key=errors.get)
        print("Ensemble method OOF errors:", errors)
        print(f"Selected method for submission: {best_method}")
        
        if best_method == 'equal':
            best_test = eq_test
        elif best_method == 'stacking':
            best_test = stack_test
        elif best_method == 'hill':
            best_test = hill_test
        else:
            best_test = perf_test
        
        if calibrate:
            best_test = np.expm1(best_test)
        return best_test

if __name__ == "__main__":
    mapping = {'Sex': {'male': 0, 'female': 1}}
    
    preprocessor = DataPreprocessor(train_df, test_df, TARGET, mapping_cols=mapping, 
                                     sample_frac=0.1, apply_poly=True, poly_degree=2)
    X, y, X_test = preprocessor.preprocess()
    
    n_trials = 5
    n_splits = 5
    optimizer = ModelOptimizer(X, y, n_splits=5, n_trials=20, random_state=42)
    best_lgbm = optimizer.optimize_lgbm()
    best_xgb = optimizer.optimize_xgb()
    best_rf = optimizer.optimize_rf()
    best_cat = optimizer.optimize_cat()
    best_et = optimizer.optimize_et()
    best_nn = optimizer.optimize_nn()
    best_lr = optimizer.optimize_lr()  # {} for LinearRegression
    print("Best parameters for LGBM:", best_lgbm)
    print("Best parameters for XGBoost:", best_xgb)
    print("Best parameters for RandomForest:", best_rf)
    print("Best parameters for CatBoost:", best_cat)
    print("Best parameters for ExtraTrees:", best_et)
    print("Best parameters for NN:", best_nn)
    print("Best parameters for LR:", best_lr)
    
    best_params = {
        'lgbm': best_lgbm,
        'xgb': best_xgb,
        'rf': best_rf,
        'cat': best_cat,
        'et': best_et,
        'nn': best_nn,
        'lr': best_lr,
    }
    
    trainer = EnsembleTrainer(X, y, X_test, cv_folds=5, random_state=42)
    trainer.set_best_params(best_params)
    trainer.fit()
    
    eq_error, _ = trainer.equal_weight_ensemble()
    print("Equal Weight Ensemble OOF RMSLE:", eq_error)
    
    final_submission_predictions = trainer.select_best_for_submission(calibrate=True)


sub_df[TARGET] = final_submission_predictions
sub_df.to_csv('submission.csv', index=False)
print(sub_df.head(4))

