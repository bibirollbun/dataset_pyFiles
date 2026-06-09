import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
sns.set_theme()
import torch

SEED = 42
DEVICEx = 'gpu' if torch.cuda.is_available() else 'cpu'
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

import optuna


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
original_df = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')

use_ids = test_df.id ##use in submission
train_df.drop(columns=['id'],inplace=True)
test_df.drop(columns=['id'],inplace=True)


print(
    f'Shape of the train dataset: {train_df.shape}'
    '\n'
    f'Shape of the test dataset: {test_df.shape}'
    '\n'
    f'Shape of the original dataset: {original_df.shape}'
    '\n'
    f'Test dataset as the proportion of the train dataset: {100*len(test_df)/len(train_df):.2f}%'
)

print('=> Train Data')
display(train_df.head())
print('=> Original Data')
display(original_df.head())


nulls_train = train_df.isnull().sum().sum()
nulls_test = test_df.isnull().sum().sum()
nulls_original = original_df.isnull().sum().sum()

if all([data == 0 for data in [nulls_train,nulls_test,nulls_original]]): #first time using all(), this function needs a list of bools !!
    print(f'We have no nulls anywhere.')
else:
    print(f'We have nulls.')


def enum(data:list):
    return enumerate(data)

def get_cols(data):
    # Fixed: .columns is an attribute
    cats = data.select_dtypes(exclude='number').columns.tolist()
    nums = data.select_dtypes(include='number').columns.tolist()
    return cats, nums

def process(data, org_data, org_cols_te, cats, MAP=None, test=False, target='diagnosed_diabetes'):
    if test and MAP is None:
        raise ValueError("Must provide MAP for test set")
        
    df = data.copy()
    te_map = {} if not test else MAP
    for col in org_cols_te:
        vc = org_data.groupby(col)[target].value_counts(normalize=True)
    
        prob_df = vc.unstack(fill_value=0)
    
        if 1 in prob_df.columns:
            target_map = prob_df[1] 
        else:
            target_map = prob_df.get(1, pd.Series(0, index=prob_df.index))

        df[f'org_{col}_TE'] = df[col].map(target_map)


    eps = 1e-4

    df['activitbysleep'] = df['physical_activity_minutes_per_week'] / (eps + df['sleep_hours_per_day'] * 7 * 60)
    df['isAdult'] = (df['age'] > 18).astype(int)
    
    # Log10 for AtheroPlasma (standard medical index)
    df['AtheroPlasma'] = np.log10(df['triglycerides'] / (eps + df['hdl_cholesterol']))
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['PulsePressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['bmi_to_waist'] = df['bmi'] / (eps + df['waist_to_hip_ratio'])
    df['waist_to_age'] = df['waist_to_hip_ratio'] / (eps + df['age'])
    df['bmi_to_age'] = df['bmi'] / (eps + df['age'])

    # 3. Cross Binning
    df['incomexeducation'] = df['income_level'].astype(str) + "_" + df['education_level'].astype(str)
    df['ethnicityxgender'] = df['ethnicity'].astype(str) + "_" + df['gender'].astype(str)
    df['smokingxgender'] = df['smoking_status'].astype(str) + "_" + df['gender'].astype(str)
    added_cols = ['incomexeducation', 'ethnicityxgender', 'smokingxgender']

    if not test:
        for col in cats:
            # A. Get probability of target distribution via value_counts
            vc = df.groupby(col)[target].value_counts(normalize=True)
            
            # B. Unstack to get valid map for Class 1
            prob_df = vc.unstack(fill_value=0)
            
            # C. Grab Class 1 map
            # This represents: P(Diabetes=1 | Category)
            if 1 in prob_df.columns:
                class_1_map = prob_df[1]
            else:
                class_1_map = pd.Series(0, index=prob_df.index)

            df[f'{col}_TE'] = df[col].map(class_1_map)
            te_map[col] = class_1_map
    else:
        for col in cats:
            df[f'{col}_TE'] = df[col].map(te_map.get(col)).fillna(0)

    for col in cats + added_cols:
        df[col] = df[col].astype('category')

    if test:
        return df
    return df, te_map


n_splits = 5
splitter = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=SEED)
target = 'diagnosed_diabetes'


cats,nums = get_cols(test_df)


active_df = train_df
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',         
    'tree_method': 'hist',        
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': SEED,
    'device':DEVICEx
}

params = {
    'objective': 'binary',            # 'binary:logistic' in xgb
    'metric': 'auc',                  # 'eval_metric' in xgb
    'boosting_type': 'gbdt',          # default (similar to 'hist' logic)
    'learning_rate': 0.05,
    
    # Tree Control
    'max_depth': 6,
    # LGBM is leaf-wise. 2^6 = 64, so <64 leaves constrains it similarly to depth 6
    'num_leaves': 63,                 
    
    # Stochastic sampling
    'subsample': 0.8,                 # Also called 'bagging_fraction'
    'subsample_freq': 1,              # Required to enable bagging (resample every k iterations)
    'colsample_bytree': 0.8,          # Also called 'feature_fraction'
    
    # System
    'random_state': SEED,             # 'seed' in xgb
    'device': 'gpu',                  # LGBM uses 'gpu' (or 'cuda' in specific builds), not 'cuda:0'
    'verbosity': -1                   # Suppress warnings
}
scores = []
idx = 0
for train_idx,val_idx in splitter.split(np.zeros(len(active_df)),active_df[target]):
    print(f'Fold: {idx+1}')
    TRAIN,VAL = active_df.iloc[train_idx],active_df.iloc[val_idx]
    TRAIN_PROCESSED,MAP = process(TRAIN,original_df,cats,cats)
    VAL_PROCESSED = process(VAL,original_df,cats,cats,MAP=MAP,test=True)

    X,y = TRAIN_PROCESSED.drop(columns=[target]),TRAIN_PROCESSED[target]
    XVal,yval = VAL_PROCESSED.drop(columns=[target]),VAL_PROCESSED[target]


    dtrain = lgb.Dataset(X, label=y, categorical_feature=cats)
    dval = lgb.Dataset(XVal, label=yval, categorical_feature=cats, reference=dtrain)
        
        # Train
    model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'eval'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(100)
            ]
        )
        
    val_preds = model.predict(XVal, num_iteration=model.best_iteration)
    fold_score = roc_auc_score(yval, val_preds)
    scores.append(fold_score)
    models.append(model)

    # dtrain = xgb.DMatrix(data=X,label=y,enable_categorical=True)
    # dval = xgb.DMatrix(data=XVal,label=yval,enable_categorical=True)

    # model = xgb.XGBRegressor(random_state=SEED)

    # watchlist = [(dtrain, 'train'), (dval, 'eval')]
    
    # model = xgb.train(
    #     params=params,
    #     dtrain=dtrain,
    #     num_boost_round=1000,
    #     evals=watchlist,
    #     early_stopping_rounds=100,
    #     verbose_eval=250
    # )

    val_preds = model.predict(dval)
    fold_score = roc_auc_score(yval, val_preds)
    scores.append(fold_score)
    idx+=1
print(scores)
print(f'Baseline Score: {np.mean(scores):.4f}±{np.std(scores):.5f}')


def train_xgb_model(params, DATA, original_df, splitter, cats, target, early_rounds=False,rounds=5000):
    active_df = DATA
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(splitter.split(active_df, active_df[target])):
        TRAIN = active_df.iloc[train_idx]
        VAL = active_df.iloc[val_idx]
        
        TRAIN_PROCESSED, MAP = process(TRAIN, original_df, cats, cats)
        VAL_PROCESSED = process(VAL, original_df, cats, cats, MAP=MAP, test=True)
        
        X, y = TRAIN_PROCESSED.drop(columns=[target]), TRAIN_PROCESSED[target]
        XVal, yval = VAL_PROCESSED.drop(columns=[target]), VAL_PROCESSED[target]
        
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y), 
            y=y
        )
        # Create a dictionary to map class -> weight
        class_weight_dict = dict(zip(np.unique(y), weights))
        # Map weights to each sample in the training set
        sample_weights = y.map(class_weight_dict)
        
        # Pass 'weight' to DMatrix
        dtrain = xgb.DMatrix(data=X, label=y, weight=sample_weights, enable_categorical=True)
        dval = xgb.DMatrix(data=XVal, label=yval, enable_categorical=True)
        
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=rounds,
            evals=watchlist,
            early_stopping_rounds=100,
            verbose_eval=early_rounds
        )
        
        val_preds = model.predict(dval)
        fold_score = roc_auc_score(yval, val_preds)
        scores.append(fold_score)
        
    mean_score = np.mean(scores)
    return mean_score

def train_lgbm_model(params, DATA, original_df, splitter, cats, target, early_rounds=False,rounds=5000):
    active_df = DATA
    scores = []
    verbose_eval = early_rounds if isinstance(early_rounds, int) else 0
    
    for fold, (train_idx, val_idx) in enumerate(splitter.split(active_df, active_df[target])):
        TRAIN = active_df.iloc[train_idx]
        VAL = active_df.iloc[val_idx]
        
        TRAIN_PROCESSED, MAP = process(TRAIN, original_df, cats, cats)
        VAL_PROCESSED = process(VAL, original_df, cats, cats, MAP=MAP, test=True)
        
        X, y = TRAIN_PROCESSED.drop(columns=[target]), TRAIN_PROCESSED[target]
        XVal, yval = VAL_PROCESSED.drop(columns=[target]), VAL_PROCESSED[target]
        
        # --- NEW: Compute Class Weights ---
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y), 
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), weights))
        sample_weights = y.map(class_weight_dict)
        
        # Pass 'weight' to Dataset
        dtrain = lgb.Dataset(X, label=y, categorical_feature=cats, weight=sample_weights)
        dval = lgb.Dataset(XVal, label=yval, categorical_feature=cats, reference=dtrain)
        
        model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=rounds,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'eval'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(verbose_eval)
            ]
        )
        
        val_preds = model.predict(XVal, num_iteration=model.best_iteration)
        fold_score = roc_auc_score(yval, val_preds)
        scores.append(fold_score)
        
    mean_score = np.mean(scores)
    return mean_score


active,hold = train_test_split(train_df,test_size=0.1,stratify=train_df[target],random_state=SEED)


import warnings 
warnings.filterwarnings('ignore')

def op_xgb(trial):
    params = {
        'device':DEVICEx,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5)
    }
    loss = train_xgb_model(params=params,DATA=active,original_df=original_df,splitter=splitter,cats=cats,target=target,early_rounds=1000)
    return loss
    
def op_lgb(trial):
    params = params = {
        'device': 'gpu', # Use 'gpu' for LGBM if available
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),

        'subsample': trial.suggest_float('subsample', 0.4, 0.9),
        'subsample_freq': 1,
    
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
    
        'max_bin': trial.suggest_int('max_bin', 63, 128),
    
        'num_leaves': trial.suggest_int('num_leaves', 20, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 1)
    }
    
    score= train_lgbm_model(params, active, original_df, splitter, cats, target)
    return score
    
#optimize xgb and store params
# study = optuna.create_study(study_name='optim01',direction='maximize')
# study.optimize(op_xgb,n_trials=50,show_progress_bar=True,n_jobs=1)
# best_params_xgb = study.best_trial.params
# best_params_xgb['device'] = DEVICEx
# best_params_xgb['objective'] = 'binary:logistic'
# best_params_xgb['tree_method'] = 'hist'
# best_params_xgb['eval_metric'] = 'auc'
# best_params_xgb['random_state'] = SEED
# best_score_xgb = study.best_trial.value
import gc 
gc.collect()
torch.cuda.empty_cache()

#optimize lgb and store params
# study_lgb = optuna.create_study(study_name='optim01',direction='maximize')
# study_lgb.optimize(op_lgb,n_trials=50,show_progress_bar=True,n_jobs=1)
best_params_lgbm = {'learning_rate': 0.05193315768151256, 'subsample': 0.7855921031344533, 'colsample_bytree': 0.5746121430300201, 'max_bin': 128, 'num_leaves': 46, 'max_depth': 10, 'min_child_weight': 44, 'lambda_l1': 0.03205117568406326, 'lambda_l2': 0.024325807905848932, 'min_split_gain': 0.6940379420777143}
best_params_lgbm['device'] = DEVICEx
best_params_lgbm['objective'] = 'binary'
best_params_lgbm['boosting_type'] = 'gbdt'
best_params_lgbm['metric'] = 'auc'
best_params_lgbm['random_state'] = SEED
best_params_lgbm['verbosity'] = -1
best_params_lgbm['subsample_freq'] = 1
# best_score_lgbm = study_lgb.best_trial.value


best_params_lgbm


from sklearn.utils.class_weight import compute_class_weight

# --- 1. SHARED PRE-PROCESSING (VALIDATION SPLIT) ---
# Process data only once
active_processed, MAP = process(active, original_df, cats, cats)
hold_processed = process(hold, original_df, cats, cats, MAP=MAP, test=True)

# Prepare X and y
X, y = active_processed.drop(columns=[target]), active_processed[target]
XVal, yval = hold_processed.drop(columns=[target]), hold_processed[target]

# Compute weights once
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), weights))
sample_weights = y.map(class_weight_dict)

# --- 2. XGBOOST: FIND BEST ROUNDS ---
dtrain_xgb = xgb.DMatrix(data=X, label=y, weight=sample_weights, enable_categorical=True)
dval_xgb = xgb.DMatrix(data=XVal, label=yval, enable_categorical=True)

watchlist = [(dtrain_xgb, 'train'), (dval_xgb, 'eval')]

xgb_model_cv = xgb.train(
    params=best_params_xgb, # Ensure this is your XGB params dict
    dtrain=dtrain_xgb,
    num_boost_round=10000,
    evals=watchlist,
    early_stopping_rounds=100,
    verbose_eval=100
)
xgb_best_rounds = xgb_model_cv.best_iteration
print(f"XGB Best Rounds: {xgb_best_rounds}")

# --- 3. LIGHTGBM: FIND BEST ROUNDS ---
dtrain_lgb = lgb.Dataset(X, label=y, categorical_feature=cats, weight=sample_weights)
dval_lgb = lgb.Dataset(XVal, label=yval, categorical_feature=cats, reference=dtrain_lgb)

lgb_model_cv = lgb.train(
    params=best_params_lgbm, # Ensure this is your LGB params dict
    train_set=dtrain_lgb,
    num_boost_round=10000,
    valid_sets=[dtrain_lgb, dval_lgb],
    valid_names=['train', 'eval'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(100)
    ]
)
lgb_best_rounds = lgb_model_cv.best_iteration
print(f"LGB Best Rounds: {lgb_best_rounds}")

# --- 4. SHARED PRE-PROCESSING (FULL TRAIN) ---
# Process full training data once
train_processed, MAP = process(train_df, original_df, cats, cats)
X_full, y_full = train_processed.drop(columns=[target]), train_processed[target]

# Compute weights for full data once
weights_full = compute_class_weight(class_weight='balanced', classes=np.unique(y_full), y=y_full)
class_weight_dict_full = dict(zip(np.unique(y_full), weights_full))
sample_weights_full = y_full.map(class_weight_dict_full)

# --- 5. FINAL RETRAINING ---

# XGBoost Final
dtrain_full_xgb = xgb.DMatrix(data=X_full, label=y_full, weight=sample_weights_full, enable_categorical=True)
final_xgb_model = xgb.train(
    params=best_params_xgb,
    dtrain=dtrain_full_xgb,
    num_boost_round=xgb_best_rounds
)

# LightGBM Final
dtrain_full_lgb = lgb.Dataset(X_full, label=y_full, categorical_feature=cats, weight=sample_weights_full)
final_lgb_model = lgb.train(
    params=best_params_lgbm,
    train_set=dtrain_full_lgb,
    num_boost_round=lgb_best_rounds
)

print('BOTH MODELS TRAINED ON FULL DATA')


# active_processed,MAP = process(active,original_df,cats,cats)
# hold_processed = process(hold,original_df,cats,cats,MAP=MAP,test=True)

# X,y = active_processed.drop(columns=[target]),active_processed[target]
# XVal,yval = hold_processed.drop(columns=[target]),hold_processed[target]
# dtrain = xgb.DMatrix(data=X,label=y,enable_categorical=True)
# dval = xgb.DMatrix(data=XVal,label=yval,enable_categorical=True)
# watchlist = [(dtrain, 'train'), (dval, 'eval')]
# model = xgb.train(
#     params=best_params_xgb,
#     dtrain=dtrain,
#     num_boost_round=10000,
#     evals=watchlist,
#     early_stopping_rounds=100,
#     verbose_eval=100
# )
# n_rounds = model.best_iteration

# train_processed,MAP = process(train_df,original_df,cats,cats)
# X,y = train_processed.drop(columns=[target]),train_processed[target]
# dtrain = xgb.DMatrix(data=X,label=y,enable_categorical=True)
# model = xgb.train(
#     params=best_params_xgb,
#     dtrain=dtrain,
#     num_boost_round=n_rounds,
#     # evals=watchlist,
#     # early_stopping_rounds=100,
#     # verbose_eval=early_rounds
# )
# print(f'MODEL TRAINED, MAP STORED')


test_processed = process(test_df,original_df,cats,cats,MAP=MAP,test=True)
X = test_processed
dtest = xgb.DMatrix(data=X,enable_categorical=True)
dtestl = lgb.Dataset(data=X, categorical_feature=cats)
# ids = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')['id']
preds1 = final_xgb_model.predict(dtest)
preds2 = final_lgb_model.predict(X)

# sub = pd.DataFrame({
#     'id':use_ids,
#     f'{target}':preds
# })

# sub.to_csv('sub6.csv',index=False)

preds1,preds2


sub = pd.DataFrame({
    'id':use_ids,
    f'{target}':preds1*0.6+preds2*0.4
})

sub.to_csv('sub6.csv',index=False)





