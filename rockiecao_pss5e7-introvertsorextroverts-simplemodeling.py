import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import optuna
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings('ignore')


class Config:
    TRAIN_FILE = "/kaggle/input/playground-series-s5e7/train.csv"
    TEST_FILE = "/kaggle/input/playground-series-s5e7/test.csv"
    SEED = 37
    N_FOLDS = 60
    N_TRIALS = 5
    TARGET = "Personality"


train = pd.read_csv(Config.TRAIN_FILE)
test = pd.read_csv(Config.TEST_FILE)
train_y = train[Config.TARGET]
n_train, n_test = train.shape[0], test.shape[0]

df = pd.concat([train, test])
df.head(10)


train.isnull().sum()


test.isnull().sum()


df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].median(), inplace=True)
df['Social_event_attendance'].fillna(df['Social_event_attendance'].mean(), inplace=True)
df['Going_outside'].fillna(df['Going_outside'].mean(), inplace=True)
df['Friends_circle_size'].fillna(df['Friends_circle_size'].median(), inplace=True)
df['Post_frequency'].fillna(df['Post_frequency'].median(), inplace=True)


df['Stage_fear']=df['Stage_fear'].map({'Yes':1, 'No':0})
df['Stage_fear'].value_counts()


df['Drained_after_socializing']=df['Drained_after_socializing'].map({'Yes':1, 'No':0})
df['Drained_after_socializing'].value_counts()


train = df[:len(train)].copy()
train[Config.TARGET] = train[Config.TARGET].map({'Introvert':0, 'Extrovert':1})
test = df[len(train):].copy()
test.drop(columns=[Config.TARGET,'id'], inplace=True)


X = train.drop(columns=[Config.TARGET,'id'])
y = train[Config.TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=Config.SEED, stratify=y)
kfold = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

print(X_train.columns)
print(test.columns)


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-2, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1.0),
        'random_state': Config.SEED,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
    }
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=Config.N_TRIALS)  
best_xgb_params = study.best_params

# best trial
print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Best Params:")
for key, value in trial.params.items():
    print(f"{key}: {value}")


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
        'depth': trial.suggest_int('depth', 3, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'logging_level': 'Silent',
        'random_seed': Config.SEED,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0, early_stopping_rounds=50)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=Config.N_TRIALS)
best_catb_params = study.best_params
print("Best parameters:", study.best_params)
print("Best accuracy:", study.best_value)


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 150),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 7),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 7),
        'verbose': -1,
        'random_state': Config.SEED
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    preds = (preds >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=Config.N_TRIALS)
best_lgbm_params = study.best_params
print("Best parameters:", study.best_params)
print("Best accuracy:", study.best_value)


def get_trained_models(X_train, y_train):
    # LGBM
    lgbm_model = lgb.LGBMClassifier(**best_lgbm_params, verbose=-1)
    lgbm_model.fit(X_train, y_train)
    
    # XGBoost
    xgb_model = XGBClassifier(**best_xgb_params, random_state=Config.SEED, use_label_encoder=False, verbosity=0, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    # CatBoost
    catb_model = CatBoostClassifier(**best_catb_params)
    catb_model.fit(X_train, y_train, verbose=0, early_stopping_rounds=50)
    return xgb_model, catb_model, lgbm_model


cv = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
y_pred = np.zeros((test.shape[0], Config.N_FOLDS))

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
    y_train_f, y_val_f = y.iloc[train_idx],y.iloc[val_idx]
    xgb, catb, lgbm = get_trained_models(X_train, y_train)
    voting_classifier = VotingClassifier(estimators=[
        ('xgb', xgb),
        ('cat', catb),
        ('lgb', lgbm)
    ], voting='soft', verbose=False)
    voting_classifier.fit(X_train_f, y_train_f)
    y_pred[:, fold] = voting_classifier.predict_proba(test)[:, 1]
    print(f"Fold {fold} done!")

y_pred = np.mean(y_pred, axis=1)


sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

ensemble_preds = (y_pred >= 0.5).astype(int)
sub[Config.TARGET] = ensemble_preds
sub[Config.TARGET] = sub[Config.TARGET].map({0:'Introvert', 1:'Extrovert'})
sub.to_csv("submission.csv", index=False)


sub.head()




