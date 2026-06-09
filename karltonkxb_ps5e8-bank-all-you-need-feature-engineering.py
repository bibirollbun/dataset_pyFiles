!pip install optuna-integration[xgboost] -q
# !pip install xgboost==3.0 -q


%reset -f
%load_ext autoreload
%autoreload 2


%load_ext cudf.pandas

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import optuna, shap, os, json
from itertools import combinations


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, PrecisionRecallDisplay

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.neural_network import MLPClassifier 

import cuml
from cuml.preprocessing import TargetEncoder
from cuml import ElasticNet, KernelRidge, LabelEncoder, Lasso, LogisticRegression, RandomForestClassifier, Ridge, SVC


import warnings

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# original data
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')


train.shape, test.shape, original.shape, submission.shape


# encode original data target column
original['y'] = original['y'].map({'yes': 1.0, 'no': 0.0})


# Reduce memory usage

def reduce_mem_usage(df, verbose=True):
    mem_before = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {mem_before:.2f} MB")
    
    # Ignore warnings related to pandas downcasting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if str(col_type)[:3] == 'int':
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

                        
            elif str(col_type)[:5] == 'float':
                c_min = df[col].min()
                c_max = df[col].max()
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)

    
    mem_after = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after optimization is: {mem_after:.2f} MB")
        print(f"Decreased by {(100 * (mem_before - mem_after) / mem_before):.1f}%")
        
    return df



# separate columns
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

def preprocess_numerical_cols(df, numerical_cols):

    # save original col names
    original_col_names = df.columns.tolist()
    
    for col in numerical_cols:
        lower_bound = df[col].quantile(0.005)
        upper_bound = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    df['duration_log'] = np.log1p(df['duration'])
    df['campaign_log'] = np.log1p(df['campaign'])
    df['pdays_log'] = np.log1p(df['pdays'] + 2)
    df['previous_log'] = np.log1p(df['previous'] + 1)

    new_numerical_cols = ['duration_log', 'campaign_log', 'pdays_log', 'previous_log']
    
    # calculate correlation between new generated cols and original cols. if correlation >= 90 drop new generated col
    drop_cols = set()
    for new_cols in new_numerical_cols:
        for num_cols in numerical_cols:
            correlation = df[num_cols].corr(df[new_cols])
            if correlation >= 0.90 or correlation <= -0.90:
                print(f"There is high correlation - {correlation:.2f} between {num_cols} and {new_cols}. {new_cols} has been dropped to avoid multicollinarity!")
                drop_cols.add(new_cols)
                
    drop_cols = ['id'] + list(drop_cols)
    
    return df[[col for col in df.columns if col not in drop_cols]]


def preprocess_categorical_cols(df, cat_cols, encoder_type='onehot'):
    # Defensive check to ensure the input DataFrame is not None
    if df is None:
        raise ValueError("Input DataFrame for categorical preprocessing is None. "
                         "Please check the return value of the preceding function.")

    # Select the encoder based on the encoder_type argument
    if encoder_type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif encoder_type == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        raise ValueError("Invalid encoder_type. Choose 'onehot' or 'ordinal'.")

    # Fit and transform the categorical columns
    encoded_cols = encoder.fit_transform(df[cat_cols])
    
    # Get the feature names for the new columns
    feature_names = encoder.get_feature_names_out(cat_cols)
    
    # Create a DataFrame for the encoded columns
    encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df.index)
    
    # Select the remaining columns
    remainder_cols = df.drop(columns=cat_cols)

    # Concatenate the encoded columns with the remaining columns
    processed_df = pd.concat([encoded_df, remainder_cols], axis=1)

    return processed_df

numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
def new_features(df, num_cols=numerical_cols):
    df = df.copy()
    df['duration_balance'] = df['duration'] * df['balance']
    df['duration_age'] = df['duration'] * df['age']
    df['duration_age_balance'] = df['duration'] * df['age'] * df['balance']
    df['duration_day'] = df['duration'] * df['day']
    df['duration_age_day'] = df['duration'] * df['age'] * df['day']

    for col in num_cols:
        if col != 'duration':
            df[f"{col}_duration_weight"] = df['duration'] / (df[col] + 0.1)

    df['balance_log'] = np.log1p(df['balance']).clip(lower=0)
    df['job_edu'] = df['job'].astype(str) + "_" + df['education'].astype(str)
    df['contacted_before'] = (df['pdays'] != -1).astype(int)
    df['age_squared'] = df['age'] ** 2

    df['duration_sin'] = np.sin(2*np.pi * df['duration'] / 400)
    df['duration_cos'] = np.cos(2*np.pi * df['duration'] / 400)

    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_map).astype('int')

    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    df['duration_bin'] = pd.cut(
        df['duration'],
        bins=[0, 60, 300, 600, 900, float('inf')],
        labels=['short', 'medium', 'long', 'very_long', 'extreme'],
        right=False
    )
    
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 45, 60, 100],
        labels=['young', 'mid', 'senior', 'elder']
    )

    df.drop('month_num',axis=1,inplace=True)    
    return df

def create_categorical_interaction_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    # Create a list of all unique pairs of categorical columns
    pair_combinations = list(combinations(cat_cols, 2))
    
    # Iterate through each pair to create a new feature
    for col1, col2 in pair_combinations:
        new_col_name = f'{col1}_{col2}_interaction'
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
    
    return df


train = pd.concat([original, train.iloc[:400000, :], original, train.iloc[400000:, :]])
train.reset_index(drop=True)

train = new_features(train)
test = new_features(test)

cat_cols = train.select_dtypes(exclude='number').columns.to_list()

train = create_categorical_interaction_features(train, cat_cols)
test = create_categorical_interaction_features(test, cat_cols)

cat_cols = train.select_dtypes(exclude='number').columns.to_list()

for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


train['data_type'] = 'train'
test['data_type'] = 'test'
data = pd.concat([train, test])

# data_processed = preprocess_numerical_cols(data, numerical_cols)
data_processed = preprocess_categorical_cols(data, cat_cols, encoder_type='ordinal')


train_processed = data_processed[data_processed['data_type'] == 'train'].drop(columns=['data_type',])
test_processed = data_processed[data_processed['data_type'] == 'test'].drop(columns=['data_type', 'y'])

train.shape, test.shape


# target encoding
cat_cols = train.select_dtypes(exclude='number').columns.to_list()
num_cols = train.select_dtypes(include='number').columns.to_list()


cat_cols1 = []
num_cols1 = []
for col in train.columns[:-1]:
    t = "CAT"
    if train[col].dtype in ["object", "category"]:
        cat_cols1.append(col)
    else:
        num_cols1.append(col)
        t = "NUM"
    n = train[col].nunique()
    na = train[col].isna().sum()
    print(f"[{t}] {col} has {n} unique and {na} NA")

print(f"Cat cols: {cat_cols1}")
print(f"Num cols: {num_cols1}")


train = reduce_mem_usage(train)
print()
test = reduce_mem_usage(test)


# features and target
X = train.drop(columns=['data_type', 'id', 'y'])
y = train.y

test= test.drop(columns=['data_type', 'id'])


X.shape, test.shape


# XGBoost pruner is helpful to drop not promising trails
from optuna_integration.xgboost import XGBoostPruningCallback
from optuna_integration.lightgbm import LightGBMPruningCallback
from optuna.integration.catboost import CatBoostPruningCallback


SEED = 42


def objective(trial):
    # Define hyperparameters to be tuned
    params = {
        # 'objective': 'binary:logistic',
        # 'eval_metric': 'logloss',
        'tree_method': 'hist',
        'predictor': 'gpu_predictor',
        'device':'cuda',
        'n_jobs':-1,
        'random_state': SEED,

        # Tunable hyperparameters
        'n_estimators': 10_000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.4, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-2, 1e2, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 3),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1e-2, 1e2, log=True)
    }

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_loglosses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params, use_label_encoder=False)

        pruning_callback = XGBoostPruningCallback(trial, "validation_0-logloss")

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False,
            callbacks=[pruning_callback]
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        fold_logloss = log_loss(y_val, val_preds)
        cv_loglosses.append(fold_logloss)

    return np.mean(cv_loglosses)


optimize_xgb = False

if optimize_xgb:
    # Create Optuna study with pruning
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=200)


def objective(trial):

    try:
        if cat.get_gpu_device_count() > 0:
            task_type = 'GPU'
            devices = '0:1' 
        else:
            task_type = 'CPU'
            devices = None
    except:
        task_type = 'CPU'
        devices = None
    
    print(f"Using task_type: {task_type}")

    params = {
        # 'objective': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': "GPU",
        'devices':1,
        'early_stopping_rounds': 150,
        'verbose': False,
        'random_state': SEED,

        'n_estimators': 10_000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 7),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'loss_function': 'Logloss'
    }

    # Use Stratified K-Fold for cross-validation to ensure balanced folds
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    cv_loglosses = []
    
    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # use a pruning callback to stop unpromising trials early
        pruning_callback = CatBoostPruningCallback(trial, "Logloss")

        model = cat.CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[pruning_callback],
            early_stopping_rounds=150,
            verbose = 1000
        )
        

        val_preds = model.predict_proba(X_val)[:, 1]
        cv_loglosses.append(log_loss(y_val, val_preds))
    
    return np.mean(cv_loglosses)

optimize_cat = False
if optimize_cat:
    print("Starting Optuna tuning for CatBoost...")
    study_cat = optuna.create_study(
        direction='minimize', study_name='optimize catboost',
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    
    study_cat.optimize(objective, n_trials=50)



if optimize_xgb:
    print("Best Parameters:")
    best_xgb_params = study.best_params
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Log Loss: {study.best_value:.4f}")
else:
    best_xgb_params = {
          'learning_rate': 0.2724334508970556,
          'max_depth': 7,
          'subsample': 0.9059271845181917,
          'colsample_bytree': 0.6597068172714076,
          'min_child_weight': 0.026504291784205564,
          'gamma': 0.817082052740708,
          'reg_alpha': 1.125641877048605e-06,
          'reg_lambda': 9.404736184120482,
          'num_parallel_tree': 3,
          'scale_pos_weight':  0.9386836476714302,
          "enable_categorical" : True,
    }

if optimize_cat:
    print("Optimization finished!")
    print("Best parameters found:")
    best_cat_params = study_cat.best_params
    for key, value in study_cat.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest log loss: {study_cat.best_value:.4f}")
else:
    # best_cat_params = {
    # 'n_estimators': 10000,
    # 'eval_metric': 'Logloss',
    # 'task_type': 'GPU',
    # 'early_stopping_rounds': 150,
    # 'cat_features':cat_cols,
    # 'verbose': 1000,
    # 'random_state': SEED
    # }
    
    best_cat_params = {
    'bootstrap_type': "MVS",
    'boosting_type': "Plain",
    'loss_function': "Logloss",
    'random_state': 42,
    'iterations': 5000,
    'learning_rate': 0.0001,
    'depth': 8,
    'subsample': 0.9,
    'random_strength': 2.0,
    'grow_policy': 'SymmetricTree',
    'task_type': "GPU",
    'devices':'0',
    }


lgb_params = {
    'learning_rate': 0.06,
    'max_depth': 7,  
    'num_leaves': 100,  
    'max_bin': 255,  
    'subsample': 0.82,  
    'colsample_bytree': 0.65,  
    'subsample_freq': 1,
    'reg_alpha': 0.81,  
    'reg_lambda': 2.1,
    'min_child_samples': 25,  
    'min_split_gain': 0.005,  
    'extra_trees': True,
    'bagging_seed': SEED,
    'feature_fraction_seed': SEED,
    'n_estimators':20000,
}


# early_stop = xgb.callback.EarlyStopping(rounds=300, metric_name='logloss', data_name='validation_0')

parameters_xgboost = {
    'n_estimators': 10000,
    'max_leaves': 127,
    'min_child_weight': 1.5,
    'max_depth': 0,
    'grow_policy': 'lossguide',
    'learning_rate': 0.005,
    'tree_method': 'hist',
    'subsample': 0.85,
    'colsample_bylevel': 0.7,
    'colsample_bytree': 0.75,
    'colsample_bynode': 0.85,
    'sampling_method': 'gradient_based',
    'reg_alpha': 2.5,
    'reg_lambda': 0.8,
    'enable_categorical': True,
    'max_cat_to_onehot': 1,
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    # 'callbacks':[early_stop]
}


model_lgb = lgb.LGBMClassifier(
    max_depth=6,
    n_estimators=20000,
    learning_rate=0.03,
    reg_alpha=1.8,
    reg_lambda=3.5,
    colsample_bytree=0.5,
    subsample=0.8,
    max_bin=4523,
    # categorical_feature=cat_cols,

    objective= 'binary',
    metric= 'binary_logloss',
    boosting_type= 'gbdt',
    n_jobs= -1,
    random_state= SEED,
    verbose= -1,
    #device= 'gpu',
    #gpu_platform_id= 0,
    #gpu_device_id= 0,
)

model_lgb2 = lgb.LGBMClassifier(
    max_depth=0,
    n_estimators=20000,
    learning_rate=0.01,
    reg_alpha=2.8,
    reg_lambda=3.5,
    colsample_bytree=0.5,
    subsample=0.8,
    max_bin=4523,
    # categorical_feature=cat_cols,

    objective= 'binary',
    metric= 'binary_logloss',
    boosting_type= 'gbdt',
    n_jobs= -1,
    random_state= SEED,
    verbose= -1,
    #device= 'gpu',
    #gpu_platform_id= 0,
    #gpu_device_id= 0,
)

# XGBoost model using the best parameters from tuning
model_xgb = xgb.XGBClassifier(
    **parameters_xgboost
)

# catboost
model_cat = cat.CatBoostClassifier(**best_cat_params)


print(f"Target encoding {len(cat_cols1)} features ... ", end="")
for i, c in enumerate(cat_cols1):
    if i%10==0: print(f"{i}, ", end="")
    TE = TargetEncoder(n_folds=10, smooth=1.5, split_method="random", stat="mean")
    X[c] = TE.fit_transform(X[c], y).astype("float32")
    test[c]  = TE.transform(test[c]).astype("float32")

# X[cat_cols1] = X[cat_cols1].astype("category")
# X_val[cat_cols1] = X_val[cat_cols1].astype("category")
# test[cat_cols1] = test[cat_cols1].astype("category")


original_encoded = pd.concat([X, y], axis=1).iloc[:original.shape[0], :]
X_orig = original_encoded.drop(columns="y")
y_orig = original_encoded["y"]


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# Out-of-fold (OOF) predictions for blending on the training data
oof_lgb = np.zeros(X.shape[0])
oof_xgb = np.zeros(X.shape[0])
oof_lgb2 = np.zeros(X.shape[0])  
oof_cat = np.zeros(X.shape[0])  

# Predictions on the test set
lgb_preds = np.zeros(test.shape[0])
xgb_preds = np.zeros(test.shape[0])
lgb_preds2 = np.zeros(test.shape[0]) 
cat_preds = np.zeros(test.shape[0]) 



for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining fold {fold + 1}/{n_splits}")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    print("  - Training LightGBM... 1 ")
    model_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(period=500)]
    )
    
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    lgb_preds += model_lgb.predict_proba(test)[:, 1] / n_splits
    print(f"  - LightGBM Fold {fold + 1} validation log loss: {log_loss(y_val, oof_lgb[val_idx]):.4f}")


    print("  - Training LightGBM... 2 ")
    model_lgb2.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(period=500)]
    )
    
    oof_lgb2[val_idx] = model_lgb2.predict_proba(X_val)[:, 1]
    lgb_preds2 += model_lgb2.predict_proba(test)[:, 1] / n_splits
    print(f"  - LightGBM Fold {fold + 1} validation log loss: {log_loss(y_val, oof_lgb2[val_idx]):.4f}")

    
    print("  - Training XGBoost...")
    model_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose= 500,
        early_stopping_rounds=100,
    )

    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    xgb_preds += model_xgb.predict_proba(test)[:, 1] / n_splits
    print(f"  - XGBoost Fold {fold + 1} validation log loss: {log_loss(y_val, oof_xgb[val_idx]):.4f}")



oof_pred_probs ={
    "LGBM":oof_lgb,
    "XGB":oof_xgb,
    "LGBM2":oof_lgb2,
}

test_pred_probs = {
    "LGBM":lgb_preds,
    "XGB":xgb_preds,
    "LGBM2":lgb_preds2,
}


oof_pred_df = pd.DataFrame(oof_pred_probs)
test_pred_df = pd.DataFrame(test_pred_probs)


meta_model = RandomForestClassifier(
    n_estimators=2000,       
    max_depth=7,          
    min_samples_split=5,     
    min_samples_leaf=3,      
    max_features='sqrt',     
    bootstrap=True,          
    random_state=42,
)

meta_model.fit(oof_pred_df, y)

meta_oof = meta_model.predict_proba(oof_pred_df).iloc[:, 1]
meta_test_preds = meta_model.predict_proba(test_pred_df).iloc[:, 1]
print("Stacking RF Log Loss:", log_loss(y, meta_oof))


oof_pred_probs["meta_model"] = meta_oof
test_pred_probs["meta_model"] = meta_test_preds

def objective(trial):
    weights = np.array([trial.suggest_float(f'w{m}', -1, 1) for m in oof_pred_probs.keys()])
    weights /= np.sum(weights)

    preds = np.zeros(len(y))
    for m, weight in zip(oof_pred_probs.keys(), weights):
        preds += oof_pred_probs[m] * weight

    threshold = trial.suggest_float('threshold', 0, 1)

    return roc_auc_score(y, (preds > threshold).astype(int))

sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=5)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100, n_jobs=-1)


best_params = study.best_params

# scores['WeightedAverage'] = [study.best_value] * 5

best_weights = np.array([study.best_params[f'w{m}'] for m in oof_pred_probs.keys()])
best_weights /= np.sum(best_weights)

best_weights = {
    model: weight for model, weight in sorted(
        zip(oof_pred_probs.keys(), best_weights),
        key=lambda x: x[1],
        reverse=True
    )
}
print(json.dumps(best_weights, indent=2))


best_threshold = study.best_params['threshold']
print(f'Best threshold: {best_threshold:.3f}')


weighted_test_preds = np.zeros(len(test_pred_probs["XGB"]))
weighted_train_preds = np.zeros(len(oof_pred_probs["XGB"]))

for m, weight in best_weights.items():
    weighted_test_preds += test_pred_probs[m] * weight
    weighted_train_preds += oof_pred_probs[m] * weight

print(f"Weighted oof train Logloss: {log_loss(y, weighted_train_preds):.5f}")


submission['y'] = weighted_test_preds
submission.to_csv('submission.csv', index=False)
submission.head(15)


submission['y'] = xgb_preds
submission.to_csv('submission_xgb_alone.csv', index=False)
submission['y'] = lgb_preds
submission.to_csv('submission_lgb_alone.csv', index=False)
submission['y'] = lgb_preds2
submission.to_csv('submission_lgb2_alone.csv', index=False)


