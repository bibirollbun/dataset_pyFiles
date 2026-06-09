# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -qq shap optuna xgboost catboost


!pip install -U scikit-learn==1.4


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import r2_score, mean_squared_error, roc_auc_score, roc_curve
from sklearn.compose import make_column_transformer
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold, TimeSeriesSplit as TSS)
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, plot_importance, cv

from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
import optuna

from yellowbrick.regressor import ResidualsPlot, PredictionError
from sklearn.model_selection import KFold

import catboost as ctb
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor # Added for Meta-Learner

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import gc

import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")



train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
sub_df = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

target = 'Listening_Time_minutes'
target_col = 'Listening_Time_minutes'
train_df.head(3)


sub_df.head(3)


for df_name, df in [('train', train_df), ('test', test_df), ('external', sub_df)]:
    nunb_of_duplicates = df.duplicated().sum()
    if nunb_of_duplicates != 0:
        print(f'{df_name} dataset has {nunb_of_duplicates} duplicates.')
    else:
        print(f'{df_name} dataset has no duplicates')


# Remove duplicates and display the change in dataset size
sub_raw_shape = sub_df.shape
sub_df = sub_df.drop_duplicates()
print(f'The length of the original dataset has changed from {sub_raw_shape} to {len(sub_df)}')


from matplotlib.gridspec import GridSpec

plt.figure(figsize=(15, 10))
# Create a GridSpec layout with 2 rows and 3 columns
gs = GridSpec(3, 3)

# First subplot
ax1 = plt.subplot(gs[0, 0])  # Top-left
sns.countplot(data=train_df, x='Publication_Time', ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

# Second subplot
ax2 = plt.subplot(gs[0, 1])  # Top-middle
sns.countplot(data=train_df, x='Publication_Day', ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.set_ylabel('')

# Third subplot
ax3 = plt.subplot(gs[0, 2])  # Top-right
sns.countplot(data=train_df, x='Episode_Sentiment', ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
ax3.set_ylabel('')

# Fourth subplot
ax4 = plt.subplot(gs[1, 0])  # Bottom-left
sns.countplot(data=train_df, x='Genre', ax=ax4)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)

# Fifth subplot (spanning bottom-middle and bottom-right)
ax5 = plt.subplot(gs[1, 1:])  # Bottom spanning last two columns
sns.countplot(data=train_df, x='Podcast_Name', ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)
ax5.set_ylabel('')

# sixth subplot (spanning bottom)
ax6 = plt.subplot(gs[2, :])  # Bottom spanning
sns.countplot(data=train_df, x='Episode_Title', ax=ax6)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=90)
ax6.set_ylabel('')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Function to streamline the creation of subplots
def create_subplot(grid, row, col_span, data, column, rotation=90, ylabel=''):
    ax = plt.subplot(grid[row, col_span])
    sns.countplot(data=data, x=column, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    if ylabel == '':
        ax.set_ylabel('')
    return ax

# Setting up the figure and layout
plt.figure(figsize=(15, 10))
gs = GridSpec(3, 3)

# Create subplots
create_subplot(gs, 0, 0, train_df, 'Publication_Time')  
create_subplot(gs, 0, 1, train_df, 'Publication_Day')  
create_subplot(gs, 0, 2, train_df, 'Episode_Sentiment')  
create_subplot(gs, 1, 0, train_df, 'Genre')  
create_subplot(gs, 1, slice(1, 3), train_df, 'Podcast_Name') 
create_subplot(gs, 2, slice(0, 3), train_df, 'Episode_Title')  

# Adjust layout and display
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Function to streamline subplot creation
def create_boxplot(grid, row, col_span, data, x, y, rotation=90, xtick_color='black', ylabel=''):
    ax = plt.subplot(grid[row, col_span])
    sns.boxplot(data=data, x=x, y=y, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, color=xtick_color)
    if ylabel == '':
        ax.set_ylabel('')
    return ax

# Setting up the figure and layout
plt.figure(figsize=(15, 9))
gs = GridSpec(3, 3)

# Create subplots
create_boxplot(gs, 0, 0, train_df, 'Publication_Time', target, rotation=90, xtick_color='red')  # Top-left
create_boxplot(gs, 0, 1, train_df, 'Publication_Day', target, rotation=90, xtick_color='darkgreen')  # Top-middle
create_boxplot(gs, 0, 2, train_df, 'Episode_Sentiment', target, rotation=90, xtick_color='blue')  # Top-right
create_boxplot(gs, 1, 0, train_df, 'Genre', target, rotation=90, xtick_color='maroon')  # Bottom-left
create_boxplot(gs, 1, slice(1, 3), train_df, 'Podcast_Name', target, rotation=90)  # Bottom-middle spanning
create_boxplot(gs, 2, slice(0, 3), train_df, 'Episode_Title', target, rotation=90)  # Bottom spanning entire row

# Adjust layout and display
plt.tight_layout()
plt.show()


import pandas as pd

non_numeric_summary = (
    train_df.select_dtypes(exclude='number')  
          .nunique()                        
          .sort_values(ascending=True)      
          .to_frame(name='Unique Values')   
          .style.background_gradient(cmap='coolwarm', axis=0) )


non_numeric_summary



# Basic preprocessing
train_df['Episode_Length_minutes'] = pd.to_numeric(train_df['Episode_Length_minutes'], errors='coerce')
train_df['Guest_Popularity_percentage'] = pd.to_numeric(train_df['Guest_Popularity_percentage'], errors='coerce')
train_df['Completion_Ratio'] = train_df['Listening_Time_minutes'] / train_df['Episode_Length_minutes']



plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='Genre', order=train_df['Genre'].value_counts().index, palette='Set2')
plt.title('Number of Episodes per Genre')
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Number of Episodes')
plt.tight_layout()
plt.show()


null_count = pd.DataFrame({'train': train_df.isna().sum(), 
                           'test': test_df.isna().sum(), 
                           'external': sub_df.isna().sum()}
                         )
null_count.style.format("{:.0f}").background_gradient(cmap='Reds')


print(f'There are {sub_df[target].isna().sum()} missing targets in the external data. We decided to drop them.')

sub_df = sub_df.dropna(subset=target)



# test_df


INPUT_DIR = '/kaggle/input/playground-series-s5e4'
GLUON_PREDS_DIR = '/kaggle/input/12-38095-predict-podcast-listening-time' 


train = pd.read_csv(f'{INPUT_DIR}/train.csv')
test = pd.read_csv(f'{INPUT_DIR}/test.csv')
sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
# gluon_preds_df = pd.read_csv(f'{GLUON_PREDS_DIR}/submission.csv') 


train['episode_num'] = [int(x.split()[-1]) for x in train['Episode_Title'].values]
test['episode_num'] = [int(x.split()[-1]) for x in test['Episode_Title'].values]


N_FOLDS = 3
RANDOM_STATE = 42
train_cols = ['Podcast_Name', 'Episode_Title', 'Episode_Length_minutes', 'Genre',
       'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time',
       'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'episode_num']

cat_cols = train_df.select_dtypes(object).columns.values 



# Prepare target variable
y_train = train[target_col]


from sklearn.metrics import mean_squared_error


def objective_ctb(trial: optuna.Trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 7500),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.3, log=True),
        'depth': trial.suggest_int('depth', 2, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),

        'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
        'task_type': 'GPU' # Ensure GPU is available and drivers are installed
    }


    all_losses = []
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE) # Added shuffle and random_state
    for i, (train_idx, val_idx) in enumerate(cv.split(train)):
        train_data = train.iloc[train_idx]
        val_data = train.iloc[val_idx]
        train_data[cat_cols] = train_data[cat_cols].fillna('NaN')
        val_data[cat_cols] = val_data[cat_cols].fillna('NaN')
        train_data['Podcast_Name_text'] = train_data['Podcast_Name'].copy()
        val_data['Podcast_Name_text'] = val_data['Podcast_Name'].copy()
        X_train = train_data[train_cols + ['Podcast_Name_text']]
        y_train_fold = train_data[target_col] # Renamed to avoid conflict
        X_val = val_data[train_cols+ ['Podcast_Name_text']]
        y_val_fold = val_data[target_col] # Renamed to avoid conflict
        train_pool = ctb.Pool(X_train, y_train_fold, cat_features=cat_cols, text_features=['Podcast_Name_text'])
        val_pool = ctb.Pool(X_val, y_val_fold, cat_features=cat_cols, text_features=['Podcast_Name_text'])
        bst = ctb.train(train_pool, params=params, verbose=0, eval_set=val_pool, early_stopping_rounds=100) # Increased early stopping
        preds = bst.predict(X_val)
        loss = mean_squared_error(y_val_fold, preds)
        if loss > 14.5: 
            return loss
        all_losses.append(loss)
        print(f"Fold #{i} RMSE: {loss}") 
    return np.mean(all_losses)


study_ctb = optuna.create_study(direction='minimize')
study_ctb.optimize(objective_ctb, n_trials=5)
params_ctb = study_ctb.best_params


params_ctb.update({'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
        'task_type': 'GPU', 
        'random_seed': RANDOM_STATE, 
        'early_stopping_rounds': 100, 
        'verbose': 100 
       })


def objective_xgb(trial: optuna.Trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist', # Ensure GPU is available
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000), # Increased range slightly
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True), # Narrowed range
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),

        'gamma': trial.suggest_float('gamma', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'seed': RANDOM_STATE # Added
    }
    num_boost_round = trial.suggest_int('num_boost_round', 500, 5000) # Use num_boost_round instead of n_estimators directly with xgb.train


    all_losses = []
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE) # Added shuffle and random_state
    for i, (train_idx, val_idx) in enumerate(cv.split(train)):
        train_data = train.iloc[train_idx]
        val_data = train.iloc[val_idx]
        # Handle categoricals - convert to 'category' dtype
        train_data[cat_cols] = train_data[cat_cols].fillna('NaN').astype('category')
        val_data[cat_cols] = val_data[cat_cols].fillna('NaN').astype('category')

        X_train = train_data[train_cols]
        y_train_fold = train_data[target_col] # Renamed
        X_val = val_data[train_cols]
        y_val_fold = val_data[target_col] # Renamed

        train_pool = xgb.DMatrix(X_train, y_train_fold, enable_categorical=True, )
        val_pool = xgb.DMatrix(X_val, y_val_fold, enable_categorical=True)
        evallist  = [(val_pool,'eval'),]
        bst = xgb.train(params, train_pool, num_boost_round, evals=evallist, early_stopping_rounds=100, verbose_eval=False) # Increased early stopping
        preds = bst.predict(val_pool)
        loss = mean_squared_error(y_val_fold, preds)
        if loss > 14.5: 
            return loss
        all_losses.append(loss)
        print(f"Fold #{i} RMSE: {loss}")
    return np.mean(all_losses)


study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=10)



num_boost_round_xgb = 4075


params_xgb = study_xgb.best_params
params_xgb.update({ 'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'tree_method': 'gpu_hist', # Ensure GPU is available
                    'seed': RANDOM_STATE # Added
                  })


if 'n_estimators' in params_xgb:
    del params_xgb['n_estimators']


def objective_lgb(trial: optuna.Trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse', # Added metric explicitly
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000), # Increased range slightly
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True), # Narrowed range
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        # 'num_leaves': trial.suggest_int('num_leaves', 20, 3000, step=20), # Example, might add this
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), # Renamed from min_child_weight
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': RANDOM_STATE, # Added
        'n_jobs': -1, # Added
        'verbose': -1
    }
    num_boost_round = trial.suggest_int('num_boost_round', 500, 5000) # Add this if using lgb.train

    all_losses = []
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE) # Added shuffle and random_state
    for i, (train_idx, val_idx) in enumerate(cv.split(train)):
        train_data = train.iloc[train_idx]
        val_data = train.iloc[val_idx]
        # Handle categoricals - convert to 'category' dtype
        train_data[cat_cols] = train_data[cat_cols].fillna('NaN').astype('category')
        val_data[cat_cols] = val_data[cat_cols].fillna('NaN').astype('category')

        X_train = train_data[train_cols]
        y_train_fold = train_data[target_col] # Renamed
        X_val = val_data[train_cols]
        y_val_fold = val_data[target_col] # Renamed

        # Using sklearn API style for simplicity in final training
        lgbm = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(100, verbose=False)] # Increased early stopping
        lgbm.fit(X_train, y_train_fold,
                 eval_set=[(X_val, y_val_fold)],
                 eval_metric='rmse',
                 callbacks=callbacks)

        preds = lgbm.predict(X_val)
        loss = mean_squared_error(y_val_fold, preds)
        if loss > 14.5: 
            return loss
        all_losses.append(loss)
        print(f"Fold #{i} RMSE: {loss}") 
    return np.mean(all_losses)


study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=10) 
params_lgb = study_lgb.best_params


params_lgb.update({ 'objective': 'regression',
                    'metric': 'rmse', 
                    'random_state': RANDOM_STATE, 
                    'n_jobs': -1, 
                    'verbose': -1
                  })


if 'min_child_weight' in params_lgb:
    params_lgb['min_child_samples'] = params_lgb.pop('min_child_weight')


print("CatBoost Params")
print(params_ctb)
print('-' * 10)
print("XGBoost Params")
print(params_xgb)
# Add the num_boost_round back for xgb.train
print(f"XGBoost num_boost_round: {num_boost_round_xgb}")
print('-' * 10)
print("LightGBM Params")
print(params_lgb)


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


oof_preds_ctb = np.zeros(len(train))
oof_preds_xgb = np.zeros(len(train))
oof_preds_lgbm = np.zeros(len(train))


test_preds_ctb_folds = []
test_preds_xgb_folds = []
test_preds_lgbm_folds = []

oof_scores_ctb = []
oof_scores_xgb = []
oof_scores_lgbm = []


print("Starting Level-1 Training (CTB, XGB, LGBM) with Cross-Validation for Stacking...")


ctb_train_cols = train_cols + ['Podcast_Name_text']
xgb_lgb_train_cols = train_cols


test_ctb = test.copy()
test_ctb[cat_cols] = test_ctb[cat_cols].astype(object).fillna('NaN')
test_ctb['Podcast_Name_text'] = test_ctb['Podcast_Name'].copy()


test_xgb_lgb = test.copy()
test_xgb_lgb[cat_cols] = test_xgb_lgb[cat_cols].fillna('NaN').astype('category')


for fold, (train_idx, val_idx) in enumerate(kf.split(train, y_train)):
    print(f"\n----- Fold {fold + 1} / {N_FOLDS} -----")

    # --- Prepare fold data ---
    train_fold_df = train.iloc[train_idx].copy() 
    val_fold_df = train.iloc[val_idx].copy()
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]

    # --- CatBoost ---
    print("Training CatBoost...")
    train_fold_df[cat_cols] = train_fold_df[cat_cols].astype(object).fillna('NaN')
    val_fold_df[cat_cols] = val_fold_df[cat_cols].astype(object).fillna('NaN')
    train_fold_df['Podcast_Name_text'] = train_fold_df['Podcast_Name']
    val_fold_df['Podcast_Name_text'] = val_fold_df['Podcast_Name']

    X_train_ctb = train_fold_df[ctb_train_cols]
    X_val_ctb = val_fold_df[ctb_train_cols]

    ctb_model = ctb.CatBoostRegressor(**params_ctb)
    ctb_model.fit(X_train_ctb, y_train_fold,
                  eval_set=[(X_val_ctb, y_val_fold)],
                  cat_features=list(cat_cols), 
                  text_features=['Podcast_Name_text'], 
                  # Use fit verbose, not param verbose
                  verbose=params_ctb.get('verbose', 100))

    oof_preds_ctb[val_idx] = ctb_model.predict(X_val_ctb)
    test_preds_ctb_folds.append(ctb_model.predict(test_ctb[ctb_train_cols]))
    fold_score_ctb = mean_squared_error(y_val_fold, oof_preds_ctb[val_idx])
    oof_scores_ctb.append(fold_score_ctb)
    print(f"Fold {fold + 1} CTB RMSE: {fold_score_ctb:.5f}")
    del ctb_model, X_train_ctb, X_val_ctb; gc.collect()


    # --- XGBoost ---
    print("Training XGBoost...")
   
    train_fold_df[cat_cols] = train_fold_df[cat_cols].fillna('NaN').astype('category')
    val_fold_df[cat_cols] = val_fold_df[cat_cols].fillna('NaN').astype('category')

    X_train_xgb = train_fold_df[xgb_lgb_train_cols]
    X_val_xgb = val_fold_df[xgb_lgb_train_cols]

    train_pool_xgb = xgb.DMatrix(X_train_xgb, y_train_fold, enable_categorical=True)
    val_pool_xgb = xgb.DMatrix(X_val_xgb, y_val_fold, enable_categorical=True)
    test_pool_xgb = xgb.DMatrix(test_xgb_lgb[xgb_lgb_train_cols], enable_categorical=True) 

    evallist_xgb  = [(val_pool_xgb,'eval'),]
    xgb_model = xgb.train(params_xgb, train_pool_xgb, num_boost_round_xgb,
                          evals=evallist_xgb,
                          early_stopping_rounds=params_ctb.get('early_stopping_rounds', 100),
                          verbose_eval=False)

    oof_preds_xgb[val_idx] = xgb_model.predict(val_pool_xgb)
    test_preds_xgb_folds.append(xgb_model.predict(test_pool_xgb))
    fold_score_xgb = mean_squared_error(y_val_fold, oof_preds_xgb[val_idx])
    oof_scores_xgb.append(fold_score_xgb)
    print(f"Fold {fold + 1} XGB RMSE: {fold_score_xgb:.5f}")
    del xgb_model, X_train_xgb, X_val_xgb, train_pool_xgb, val_pool_xgb, test_pool_xgb; gc.collect()


    # --- LightGBM ---
    print("Training LightGBM...")
    
    X_train_lgbm = train_fold_df[xgb_lgb_train_cols]
    X_val_lgbm = val_fold_df[xgb_lgb_train_cols]

    # Ensure categorical features are treated correctly by LGBM sklearn API
    X_train_lgbm[cat_cols] = X_train_lgbm[cat_cols].astype('category')
    X_val_lgbm[cat_cols] = X_val_lgbm[cat_cols].astype('category')
  

    lgbm_model = lgb.LGBMRegressor(**params_lgb)
    callbacks_lgbm = [lgb.early_stopping(params_ctb.get('early_stopping_rounds', 100), verbose=False)] 
    lgbm_model.fit(X_train_lgbm, y_train_fold,
                   eval_set=[(X_val_lgbm, y_val_fold)],
                   eval_metric='rmse',
                   callbacks=callbacks_lgbm)

    oof_preds_lgbm[val_idx] = lgbm_model.predict(X_val_lgbm)
    test_preds_lgbm_folds.append(lgbm_model.predict(test_xgb_lgb[xgb_lgb_train_cols])) 
    fold_score_lgbm = mean_squared_error(y_val_fold, oof_preds_lgbm[val_idx])
    oof_scores_lgbm.append(fold_score_lgbm)
    print(f"Fold {fold + 1} LGBM RMSE: {fold_score_lgbm:.5f}")
    del lgbm_model, X_train_lgbm, X_val_lgbm; gc.collect()


print("\n----- Base Model OOF Scores -----")
print(f"CTB  | Mean RMSE: {np.mean(oof_scores_ctb):.5f} | Std Dev: {np.std(oof_scores_ctb):.5f}")
print(f"XGB  | Mean RMSE: {np.mean(oof_scores_xgb):.5f} | Std Dev: {np.std(oof_scores_xgb):.5f}")
print(f"LGBM | Mean RMSE: {np.mean(oof_scores_lgbm):.5f} | Std Dev: {np.std(oof_scores_lgbm):.5f}")



# --- Overall OOF Score for Base Models ---
overall_rmse_ctb = mean_squared_error(y_train, oof_preds_ctb)
overall_rmse_xgb = mean_squared_error(y_train, oof_preds_xgb)
overall_rmse_lgbm = mean_squared_error(y_train, oof_preds_lgbm)
print(f"\nOverall OOF CTB RMSE:  {overall_rmse_ctb:.5f}")
print(f"Overall OOF XGB RMSE:  {overall_rmse_xgb:.5f}")
print(f"Overall OOF LGBM RMSE: {overall_rmse_lgbm:.5f}")


# --- Prepare Meta-Learner Features ---
print("\nPreparing Meta-Learner Features...")
X_meta_train = pd.DataFrame({
    'oof_ctb': oof_preds_ctb,
    'oof_xgb': oof_preds_xgb,
    'oof_lgbm': oof_preds_lgbm
})


# Average the test predictions across folds for each base model
test_preds_ctb = np.mean(test_preds_ctb_folds, axis=0)
test_preds_xgb = np.mean(test_preds_xgb_folds, axis=0)
test_preds_lgbm = np.mean(test_preds_lgbm_folds, axis=0)



# Combine averaged test predictions for the test set
X_meta_test = pd.DataFrame({
    'oof_ctb': test_preds_ctb,
    'oof_xgb': test_preds_xgb,
    'oof_lgbm': test_preds_lgbm
})

print("Meta-features prepared.")
print("Shape X_meta_train:", X_meta_train.shape)
print("Shape X_meta_test:", X_meta_test.shape)
print("Columns X_meta_train:", X_meta_train.columns)
print("Columns X_meta_test:", X_meta_test.columns)


# --- Train Random Forest Meta-Learner ---
print("\n----- Training Level-2 Meta-Learner (Random Forest) -----")

# Define parameters for Random Forest Meta-Learner
rf_meta_params = {
    'n_estimators': 200,      
    'max_depth': 7,           
    'min_samples_leaf': 10,   
    'min_samples_split': 15,   
    'random_state': RANDOM_STATE,
    'n_jobs': -1,             
    'max_features': 0.8        
}

meta_model = RandomForestRegressor(**rf_meta_params)
meta_model.fit(X_meta_train, y_train)

print("Meta-learner training complete.")


# --- Evaluate Meta-Model (using OOF predictions as a proxy) ---
meta_oof_preds = meta_model.predict(X_meta_train)
meta_oof_score = mean_squared_error(y_train, meta_oof_preds)
print(f"Meta-Learner (RF) OOF RMSE (proxy): {meta_oof_score:.5f}")


# --- Generate Final Predictions with Stacking ---
print("\n----- Generating Final Test Predictions with Stacking -----")
stacking_preds = meta_model.predict(X_meta_test)

print("Stacking predictions generated.")


# Load gluon predictions
gluon_preds = sub_df['Listening_Time_minutes'].values
print("Gluon predictions loaded. Shape:", gluon_preds.shape)
print("Stacking predictions shape:", stacking_preds.shape)


print("\nBlending Stacking predictions with Gluon predictions (40/60)...")
# Ensure shapes match before blending
if len(stacking_preds) == len(gluon_preds):
    final_predictions = stacking_preds * 0.4 + gluon_preds * 0.6
    print("Blending complete.")
    sub['Listening_Time_minutes'] = final_predictions
else:
    print(f"Error: Prediction shapes do not match! Stacking: {len(stacking_preds)}, Gluon: {len(gluon_preds)}")
    # Fallback or raise error - using stacking preds only as a fallback here
    sub['Listening_Time_minutes'] = stacking_preds


print("\nCreating submission file...")
sub.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")

