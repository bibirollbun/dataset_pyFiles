target = 'rainfall'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# change default colormap
plt.rcParams['image.cmap'] = 'Dark2'

# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import roc_auc_score, make_scorer, roc_curve, confusion_matrix
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from sklearn.feature_selection import SequentialFeatureSelector as sk_sfs
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier, 
                              StackingClassifier, BaggingClassifier,VotingClassifier)
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, cv

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras import layers

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_contour
from optuna.visualization import plot_slice
import plotly.express as px

pd.set_option('display.max_columns', 100)
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')


train_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
train_raw.tail(3)


test_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
test_raw.head(3)


orig_raw = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')
orig_raw.head(3)


# Remove empty spaces from the features names in original dataset
orig_raw.columns = orig_raw.columns.str.replace(' ', '')

# Reorder the features in original dataset to match that of competition
orig_raw = orig_raw[train_raw.columns].copy()

# Binarize the target in the original dataset
orig_raw[target] = orig_raw[target].map({'no': 0, 'yes': 1})

# fill the missing values in test and original datasets
orig_raw = orig_raw.fillna(method='bfill')
test_raw = test_raw.fillna(method='bfill')


train_raw.loc[train_raw['maxtemp'] < train_raw['mintemp']][['day', 'mintemp','maxtemp', 'temparature']]


# Check for duplicates in the datasets
for df_name, df in [('train', train_raw), ('original', orig_raw)]:
    num_of_duplicates = df.duplicated().sum()
    if num_of_duplicates != 0:
        print(f'The {df_name} dataset has {num_of_duplicates} duplicates. They need to be dropped.')
    else:
        print(f'The {df_name} dataset has no duplicates')


train_comb = pd.concat([train_raw, orig_raw], ignore_index=True)
train_comb.tail()


tra = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

tra.day.plot(figsize=(12,3), marker='o');



orig_raw.day.plot(figsize=(12,3), marker='o');


test_raw.day.plot(figsize=(12,3), marker='o');


# Engeneer new features and separate data from the target in each set
def df_processing(df):
    # df['dew_humidity'] = df['dewpoint']*df['humidity']
    # df['temp_gap'] = df['maxtemp'] - df['mintemp']
    # df['temp_gap_ratio'] = df['temp_gap']/df['temparature']
    # df['wind_speeddirection'] = df['windspeed']*df['winddirection']
    # df['cloud_windspeed'] = df['cloud']*df['windspeed']
    # df['cloud_to_humidity'] = df['cloud']/df['humidity']
    # df['temp_to_humidity'] = df['cloud']/df['humidity']
    # df['temp_to_sunshine'] = df['sunshine']/df['temparature']
    # df['month'] = pd.cut(df['day'], bins=12, labels=range(1, 13)).astype('int')
    df['temp_previous_day'] = df['temparature'].shift(1).fillna(0)
    df['temp_next_day'] = df['temparature'].shift(-1).fillna(0)
    df['humidity_previous_day'] = df['humidity'].shift(1).fillna(0)
    df['pressure_previous_day'] = df['pressure'].shift(1).fillna(0)
    df['day_bins'] = pd.cut(df['day'], bins=12, labels=range(1, 13))
    # df = df.drop(columns=['maxtemp', 'mintemp'])
    X = df.copy()
    try:
        y = X.pop(target)
        return X, y
    except:
        pass
        return X
        # pass


tr = train_raw.copy()
X_tr, y_tr = df_processing(tr)
X_tr.info()

og = orig_raw.copy()
X_og, y_og = df_processing(og)
# X_og.info()

ts = test_raw.copy()
X_ts = df_processing(ts)
# X_ts.info()

tr_c = train_comb.copy()
X_tr_c, y_tr_c = df_processing(tr_c)
# X_tr_c.info()


column_trans = make_column_transformer(
    (OneHotEncoder(), X_tr.select_dtypes('object').columns.tolist()),
    # (QuantileTransformer(), X_tr.select_dtypes('number').columns), 
    remainder='passthrough', 
    sparse_threshold=0)


pd.DataFrame(column_trans.fit_transform(X_tr), columns=X_tr.columns).head()


def objective(trial):
    cat_param_grid = {
        "iterations": trial.suggest_int("iterations", 100, 1000, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
        # "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1),
        "random_strength": trial.suggest_float("random_strength", 0.1, 0.7),
        "depth": trial.suggest_int("depth", 1, 10),
        # "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "boosting_type": "Plain",
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "used_ram_limit": "5gb",
    }

    if cat_param_grid["bootstrap_type"] == "Bayesian":
        cat_param_grid["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)
    elif cat_param_grid["bootstrap_type"] == "Bernoulli":
        cat_param_grid["subsample"] = trial.suggest_float("subsample", 0.5, 1)

    # Train the model
    model = make_pipeline(column_trans, 
                          PCA(n_components=10), 
                          CatBoostClassifier(**cat_param_grid, verbose=0))
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=8)
    model.fit(X_train, y_train)

    # Evaluate the model
    preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, preds)
    return auc_score

def Run_Pass_cat_study(n_trials=1):
    if n_trials>1:
        # Create and run the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=36000, show_progress_bar=True)
        best_study_params = study.best_params
        # Print the best trial
        print('Number of finished trials: {}'.format(len(study.trials)))
        trial = study.best_trial
        print('Best trial auc_score: {:.6f}'.format(trial.value))

    else:
        print('No need to run optuna, we will use the parameters obtained earlier')
        best_study_params = {'iterations': 380,
                                'learning_rate': 0.02994469538378683,
                                'objective': 'Logloss',
                                'colsample_bylevel': 0.8704558522886159,
                                'random_strength': 0.6467195903372872,
                                'depth': 8,
                                'bootstrap_type': 'Bernoulli',
                                'subsample': 0.8301544244176395}
    
    print('best params: {}'.format(best_study_params))
    return best_study_params


cat_best_params = Run_Pass_cat_study(n_trials=200)


n_max = X_tr.shape[1] +1


for n in range(2, n_max):
    cat_clf_pipe = make_pipeline(column_trans, 
                                 PCA(n_components=n),
                                 CatBoostClassifier(**cat_best_params, verbose=0)
                                ).fit(X_tr, y_tr)
    
    # Evaluate the model on the original data
    og_pred = cat_clf_pipe.predict_proba(X_og)[:, 1]
    score = roc_auc_score(y_og, og_pred)
    print('auc_score_on_orig with {} components: {:.5f}'.format(n, score))


cat_clf_pipe = make_pipeline(column_trans, 
                             PCA(n_components=12),
                             CatBoostClassifier(**cat_best_params, verbose=0)
                            ).fit(X_tr, y_tr)

# Evaluate the model on the original data
og_pred = cat_clf_pipe.predict_proba(X_og)[:, 1]
score = roc_auc_score(y_og, og_pred)
print('auc_score_on_orig: {:.5f}'.format(score))


def oof_cat_scorer(X, y, n_splits, seed):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_ind, va_ind) in enumerate(kfold.split(X, y), start=1):
        X_train, X_val = X.iloc[tr_ind], X.iloc[va_ind]
        y_train, y_val = y.iloc[tr_ind], y.iloc[va_ind]

        cat_clf_pipe.fit(X_train, y_train)
        y_hat_val = cat_clf_pipe.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_hat_val)

        print('Fold_{} auc_score: {:.5f}'.format(f, score))


oof_cat_scorer(X_tr, y_tr, 5, 35)


oof_cat_scorer(X_tr_c, y_tr_c, 5, 35)


cat_clf_pipe = make_pipeline(column_trans,
                             PCA(n_components=7),
                             CatBoostClassifier(**cat_best_params, verbose=0)
                            ).fit(X_tr_c, y_tr_c)

y_hat_ts = cat_clf_pipe.predict_proba(X_ts)[:, 1]

test_pred_df = pd.Series(y_hat_ts)


plt.subplot(121)
test_pred_df.plot.hist(bins=25, color='grey', figsize=(10, 3), title='Hist of pred_proba in test set')
plt.xlabel('Predicted Proba')
plt.subplot(122)
(test_pred_df > 0.5).value_counts().plot.pie(labels=['rain', 'no rain'], autopct='%1.1f%%', 
                                             explode=[0.05, 0.05], colors=['lightblue', 'grey'], radius=1.3)
plt.ylabel('');


sub_raw = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub_df = sub_raw.copy()
sub_df['rainfall'] = y_hat_ts

display(sub_df.head(10))

sub_df.to_csv('submission.csv', index=False)
print('The file is ready for submission')


from sklearn.model_selection import TimeSeriesSplit

X, y = X_tr, y_tr

tscv = TimeSeriesSplit(n_splits=8)
score = 0
for f, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(eval_metric='AUC')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0, early_stopping_rounds=100, cat_features=['day_bins'])

    val_pred = model.predict_proba(X_val)[:, 1]
    score += roc_auc_score(y_val, val_pred)
    print('Fold_{} roc_auc: {:.8f}'.format(f, roc_auc_score(y_val, val_pred)))


X, y = X_tr, y_tr

cv = StratifiedKFold(8, shuffle=True, random_state=1)
cv_splits = cv.split(X, y)
scores = []
test_preds = []
X_test_pool = Pool(X_ts, cat_features=['day_bins'])
for f, (train_idx, val_idx) in enumerate(cv_splits, start=1):
    model = CatBoostClassifier(eval_metric='AUC')
    X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    X_train_pool = Pool(X_train_fold, y_train_fold, cat_features=['day_bins'])
    X_valid_pool = Pool(X_val_fold, y_val_fold, cat_features=['day_bins'])
    model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=0, early_stopping_rounds=100)
    val_pred = model.predict_proba(X_valid_pool)[:, 1]
    score = roc_auc_score(y_val_fold, val_pred)
    scores.append(score)
    test_pred = model.predict_proba((X_test_pool))[:, 1]
    test_preds.append(test_pred)
    print('Fold_{} roc_auc_score: {:.8f}'.format(f, score))
print(f'\naverage auc_score: {np.mean(scores):.8f} Â± {np.std(scores):.8f}\n ')

test_preds_series = np.mean(pd.Series(test_preds))

sub_df = sub_raw.copy()
sub_df['rainfall'] = test_preds_series

display(sub_df.head(10))

sub_df.to_csv('submission.csv', index=False)
print('The file is ready for submission')

