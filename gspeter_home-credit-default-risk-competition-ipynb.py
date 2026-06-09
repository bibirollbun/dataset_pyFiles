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


import optuna
from sklearn.metrics import average_precision_score,roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np # Make sure numpy is imported
import polars 
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
import seaborn as sns 


def make_nan_free(path : str):
    
    df = polars.read_csv(path)
    # nan values
    # df = df.select('AMT_INCOME_TOTAL','AMT_CREDIT', 'AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED' ,'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3','TARGET')   
    # df = df.drop('DAYS_BRITH')
    
    null_df = df.null_count()
    null_col = [] 
    for col_name, values in null_df.row(0,named = True).items():
        if values > 0 : 
            null_col.append(col_name)
    
    numeric_null_col = df.select(null_col).select(polars.col(polars.Float64)).columns ; cate_null_col = len(null_col) - len(numeric_null_col) 
    print(f' initial total numeric null col : {len(numeric_null_col)}') ; print(f' initial total cat null col : {cate_null_col}')
    
    
    df = df.with_columns([
        polars.col(columns).median()
        for columns in numeric_null_col
            ]
        )  
    df = df.with_columns(
        [
            polars.col(columns).fill_null('Unknown')
            for columns in list(set(null_col) - set(numeric_null_col))
        ]
    )
    print(f'total null values in full dataset :  {sum(list(df.null_count().row(0)))}')


    return df 



df = make_nan_free('/kaggle/input/home-credit-default-risk/application_train.csv')



import polars 
def make_efficient(df): 
    numeric_dataframe = df.select(polars.col(polars.Int64, polars.Float64))
    df = df.with_columns([
        (polars.col(columns) - polars.mean(columns)) / (polars.std(columns) + 1e-3 ) .alias(columns) 
        for columns in numeric_dataframe.columns if columns != 'TARGET'
    ])
    return df 
    
df = make_efficient(df)



x = df.drop('TARGET')
y = df.select('TARGET')

sns.histplot(y)
plt.show() 

x_train, x_val , y_train, y_val = train_test_split(x,y,test_size = 0.2, random_state  = 42, stratify = y )
#class_1 = y.filter(polars.col('TARGET') == 1).select(polars.sum('TARGET')).item() # that is not too much efficient 
# i have better one 

class_1 = y.filter(polars.col('TARGET') == 1).height
class_0 = y.height - class_1 

print(class_1 / y.height) ; print(class_0 / y.height) # that is exactly ( class_1:- 0.08 : class_0 :- 0.91 )

def corr(x,y,df):
    x_mean = df.select(polars.mean(x))
    x_std = df.select(polars.std(x)).item()
    
    y_mean = df.select(polars.mean(y))
    y_std = df.select(polars.std(y)).item() 
    df = df.with_columns(polars.lit(x_mean.item()).alias('x_mean'))
    
    df = df.with_columns(polars.lit(y_mean.item()).alias('y_mean'))
    sums_values = sum((df[x] - df['x_mean']) * (df[y] - df['y_mean']))
    
    return (sums_values / (df.height - 1)) / ((x_std * y_std) + 1e-6 )

def create_columns(df,name = 'df'):
    num_col_s = df.select(polars.col(polars.Float64,polars.Int64)).columns
    corr_score = np.array([corr('TARGET',_columns,df) for _columns in num_col_s if _columns != 'TARGET'])     
    
    for col_index in np.argsort(corr_score)[::-1][:3]: 
        try : 
            df = df.with_columns(
                (polars.col(num_col_s[col_index]) ** 2).alias(f'{num_col_s[col_index]}_squre') 
            )
            print(f'"{num_col_s[col_index]}_squre"  is created in "{name}" ')
        except :
            raise RuntimeError(f'error happend is due to {df[num_col_s[col_index]]} ')
    
    return df 

x_train = polars.concat([x_train,y_train], how = 'horizontal')
x_train = create_columns(x_train,name = 'x_train') 
x_train = x_train.drop('TARGET')
x_val = polars.concat([x_val, y_val], how = 'horizontal')
x_val = create_columns(x_val,name = 'x_test')
x_val = x_val.drop('TARGET')


cate_features = x_train.select(polars.col(polars.String)).columns
x_train = x_train.with_columns(
    polars.col(cate_features).cast(polars.Categorical)
)

x_val = x_val.with_columns(
    polars.col(cate_features).cast(polars.Categorical)
)

x_train = x_train.to_pandas()
x_val = x_val.to_pandas() 
y_train = y_train.to_pandas()
y_val = y_val.to_pandas()


pip install optuna-integration[lightgbm]


# sfk = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# N_ESTIMATORS = 2000
# EARLY_STOPPING_ROUNDS = 50 # Use a fixed value
# roc_importance =  0.4 
# aucpr_importance = 0.6 

# def objective(trial):
    
#     try:
#         neg_count  = (y['TARGET'].to_pandas() == 0).sum()
#         pos_count = (y['TARGET'].to_pandas() == 1).sum()
#         initial_scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
#     except NameError:
#         print("Warning: Global 'y' DataFrame or TARGET_COLUMN_NAME not found for scale_pos_weight calculation. Using default 1.0")
#         initial_scale_pos = 1.0


#     params = {
#         'n_estimators': N_ESTIMATORS,
#         'boosting_type': 'gbdt',
#         'objective': 'binary',
#         'metric': 'average_precision',
#         'scale_pos_weight': trial.suggest_float('scale_pos_weight', initial_scale_pos / 4,
#                                                 initial_scale_pos * 4, log=True), # Better range
#         'max_depth': trial.suggest_int('max_depth', 30, 100),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 500),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 150),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # Correct name
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'n_jobs': -1,
#         'seed': 42,
#         'verbose': -1, # Suppress verbose output during tuning
#     }
    
#     call_back = optuna.integration.LightGBMPruningCallback(trial, 'average_precision')
#     aucpr_score_list = [] # Store scores for EACH fold validation set
#     auc_score_list  = [] 
#     for fold, (train_idx, test_idx) in enumerate(sfk.split(x_train, y_train['TARGET'])):
        
#         # --- Data Splitting (Assuming Pandas) ---
#         X_train_fold = x_train.iloc[train_idx,:]
#         X_test_fold = x_train.iloc[test_idx,:]
#         y_train_fold_ravel = y_train.iloc[train_idx]['TARGET'].to_numpy()
#         y_test_fold_ravel = y_train.iloc[test_idx]['TARGET'].to_numpy()
    
    
#         if len(np.unique(y_test_fold_ravel)) < 2:
#             print(f"WARN: Fold {fold+1} validation set has only one class. Skipping fold for trial {trial.number}.")
#             # Optionally, report a dummy low score or handle differently if needed
#             # trial.report(0.0, fold) # Report a bad score?
#             continue # Skip to the next fold
#         # --------------------------------
    
#         # --- Sanity check training data (already had this) ---
#         if len(np.unique(y_train_fold_ravel)) < 2:
#              print(f"WARN: Fold {fold+1} training set has only one class. Skipping fold for trial {trial.number}.")
#              continue # Skip to the next fold
    
    
#         model = lgb.LGBMClassifier(**params)
#         model.fit(
#             X_train_fold,
#             y_train_fold_ravel,
#             eval_set=[(X_test_fold, y_test_fold_ravel)],
#             eval_metric='average_precision', # Correctly specified
#             callbacks=[
#                 lgb.early_stopping(stopping_rounds= 500, verbose=False), # Use fixed value
#                 # lgb.log_evaluation(period=0), # Turn off log_evaluation during tuning
#                 # call_back
#             ]
#             # Pass categorical_feature=['col1', 'col2'] here if needed
#             )
#         pt = model.predict_proba(X_test_fold)[:, 1]
#         aucpr_score = average_precision_score(y_test_fold_ravel, pt)
#         roc_score = roc_auc_score(y_test_fold_ravel,pt)
#         aucpr_score_list.append(aucpr_score)
#         auc_score_list.append(roc_score)

#         custom_loss = aucpr_importance * aucpr_score + roc_importance * roc_score 
#         trial.report(custom_loss,step = fold)
#         if trial.should_prune(): 
#             raise optuna.TrialPruned() 
            
    
#     average_aucpr_score = np.mean(aucpr_score_list)
#     average_auc_score = np.mean(auc_score_list)
#     return aucpr_importance * average_aucpr_score + roc_importance * average_auc_score 

# # --- Study Creation (Fix seed, adjust pruner) ---
# create_study = optuna.create_study(
#     direction ='maximize', 
#     sampler=optuna.samplers.TPESampler(
#         multivariate=True, # Optional ( but better for dependent hyperparameters )
#         seed=42 
#     ),
#     pruner=optuna.pruners.MedianPruner(
#         n_warmup_steps = 2 , 
#         interval_steps = 1
#     )
# )

# create_study.optimize(objective, n_trials= 10) # IF YOU INCREASE n_trials best hyperparameters you found ( my machince is not do that )








# best_params = create_study.best_params # if you use optuna turning uncommit that 
best_params = {
                'scale_pos_weight': 11.028053440991876, 'max_depth': 53,
                'num_leaves': 50, 'learning_rate': 0.005893060761114622, 
                'min_child_samples': 55, 'reg_alpha': 0.8287522363768158,
                'reg_lambda': 0.35500125258511606,
                'subsample': 0.9436063712881633, 'colsample_bytree': 0.7361074625809747
               }

best_params['n_estimators'] = 2000
best_params['boosting_type'] = 'gbdt'
best_params['objective'] = 'binary'
best_params['metric'] = 'average_precision'
best_params['n_jobs'] = -1
best_params['seed'] = 42
best_params['verbose'] = -1
# best_params['num_leaves'] = 500
# best_params['max_depth'] = -1
# best_params['learning_rate'] = 0.02
# best_params['scale_pos_weight'] = 11.5

print(best_params)
model = lgb.LGBMClassifier(**best_params)


loss_decrease = {} 
model.fit(
    x_train, 
    y_train['TARGET'], 
    eval_metric = 'average_precision',
    categorical_feature = 'auto',
    eval_set = [(x_val,y_val['TARGET'])], 
    callbacks = [
        lgb.early_stopping(stopping_rounds = 150), 
        lgb.log_evaluation(period = 10), 
        lgb.record_evaluation(loss_decrease)
    ]
) 




y_pred_proba = model.predict_proba(x_val)[:,1]
print(f'roc auc : = {roc_auc_score(y_val, y_pred_proba)}')
print(average_precision_score(y_val,y_pred_proba))
y_pred = model.predict(x_val)
print(confusion_matrix(y_val,y_pred))





