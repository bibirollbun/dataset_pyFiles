# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')


test_idx = test['id']
test.drop('id',axis=1,inplace=True)


train.head()


train.info()


train.isnull().sum()


target = 'loan_status'


all_features = [i for i in train.columns if i not in ['id',target]]


num_columns = train[all_features].select_dtypes(include=['int64','float64']).columns.tolist()
cat_columns = train[all_features].select_dtypes(include=['object']).columns.tolist()


le = LabelEncoder()




def label_e(df):
    for i in cat_columns:
        df[i] = le.fit_transform(df[i])
    return df


train = label_e(train)
test = label_e(test)


train.info()


train.head()


X= train[all_features].copy()
y= train[target].copy()


# def evaluate_model_cv(model, X, y, cv_splits=5):
#     """Cross-validation ile model değerlendirme"""
#     kf = KFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    
#     accuracies = []
#     auc_scores = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#         X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
#         y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
#         model_clone = clone(model)
#         model_clone.fit(X_train_fold, y_train_fold)
        
#         y_pred = model_clone.predict(X_val_fold)
#         y_pred_proba = model_clone.predict_proba(X_val_fold)[:, 1]
        
#         accuracy = accuracy_score(y_val_fold, y_pred)
#         auc = roc_auc_score(y_val_fold, y_pred_proba)
        
#         accuracies.append(accuracy)
#         auc_scores.append(auc)
    
#     return np.mean(accuracies), np.mean(auc_scores)



# def evaluate_model_cv(model,X,y,cv_split=5):
#     kf = KFold(n_splits=cv_splits,shuffle=True,random_state=SEED)

#     accuracies = []
#     auc_scores = []

#     for fold,(train_index,val_index) in enumerate(kf.split(X)):
#         X_train, X_val = X.iloc[train_index],X.iloc[val_index]
#         y_train, y_val = y.iloc[train_index],y.iloc[val_index]

#         model_clone = clone(model)
#         model_clone.fit(X_train, y_train)

#         y_pred = model_clone.predict(X_val)
#         y_pred_proba = model_clone.predict_proba(X_val)[:,1]

#         accuary = accuary_score(y_val,y_pred)
#         auc = roc_auc_score(y_val_fold, y_pred_proba)

#         accuuracies.append(accuracy)
#         auc_scores.append(auc)

#     return np.mean(accuracies), np.mean(auc_scores)


# def objective(trial):
#     """Optuna objective function"""
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'random_state': SEED,
#         'eval_metric': 'logloss',
#         'use_label_encoder': False
#     }
    
#     model = XGBClassifier(**params)
    
#     # 3-fold CV for speed during optimization
#     kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
#     auc_scores = []
    
#     for train_idx, val_idx in kf.split(X):
#         X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
#         y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
#         model.fit(X_train_fold, y_train_fold, verbose=False)
#         y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
#         auc = roc_auc_score(y_val_fold, y_pred_proba)
#         auc_scores.append(auc)
    
#     return np.mean(auc_scores)


# def objective(trial):
#     """Optuna objective function"""
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'random_state': SEED,
#         'eval_metric': 'logloss',
#         'use_label_encoder': False
#     }
#     model = XGBClassifier(**params)
#     kf = KFold(n_splits = 3, shuffle=True, random_state=SEED)
#     auc_scores = []

#     for train_idx, val_idx in kf.split(X):
#         X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
#         y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
#         model.fit(X_train_fold, y_train_fold, verbose=False)
#         y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
#         auc = roc_auc_score(y_val_fold, y_pred_proba)
#         auc_scores.append(auc)
    
#     return np.mean(auc_scores)


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# best_params = study.best_params
# best_score = study.best_value


# for param, value in best_params.items():
#     print(f"   {param}: {value}")



# final_model = XGBClassifier(**best_params)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=SEED, stratify=y
# )

# # Model eğitimi
# final_model.fit(X_train, y_train)



# Temel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier

# XGBoost
from xgboost import XGBClassifier

# LightGBM
import lightgbm as lgb

# CatBoost
from catboost import CatBoostClassifier

# Optuna
import optuna
import optuna.visualization.matplotlib

# Uyarıları bastırmak için
import warnings
warnings.filterwarnings("ignore")



SEED = 42
N_SPLITS = 10
N_TRIALS = 100



def cross_validation(model, X, y, cv_folds=5, scoring='roc_auc'):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    scores = {
        'auc':[],
        'accuracy':[],
        'logloss':[]
    }
    oof_predictions = np.zeros(X.shape[0])

    for fold,(train_idx, val_idx) in enumerate(skf.split(X,y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clone_model = clone(model)
        clone_model.fit(X_train,y_train)

        #predictiion

        y_pred_proba = clone_model.predict_proba(X_val)[:,1]
        y_pred = clone_model.predict(X_val)

        oof_predictions[val_idx] = y_pred_proba

        #calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)
        ll = log_loss(y_val, y_pred_proba)

        scores['auc'].append(auc)
        scores['accuracy'].append(acc)
        scores['logloss'].append(ll)
    #cakculate mean and std

    results = {}
    for metric, values in scores.items():
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)
    results['oof_predictions'] = oof_predictions
    return results


def create_advanced_objective(X, y):
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 50.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 50.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 10.0),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'random_state': SEED,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_jobs': -1
        }
        scale_pos_weight = trial.suggest_float('scale_pos_weight', 0.1, 10.0)
        params['scale_pos_weight'] = scale_pos_weight
        
        model = XGBClassifier(**params)
        
        # Fast 5-fold 
        cv_results = cross_validation(model, X, y, cv_folds=5, scoring='roc_auc')
        
        return cv_results['auc_mean']
    return objective


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=10)
)
objective_func = create_advanced_objective(X, y)

study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=True)




best_params = study.best_params
best_score = study.best_value


final_model = XGBClassifier(**best_params)


final_model.fit(X, y)


test_preds = final_model.predict_proba(test[all_features])[:, 1]

submission = pd.DataFrame({
    "id": test_idx,
    "loan_status": test_preds
})

submission.to_csv("submission.csv", index=False)



submission




