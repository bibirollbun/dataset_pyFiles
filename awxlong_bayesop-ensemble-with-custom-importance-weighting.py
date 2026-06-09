!pip install -q lifelines
!pip install -q optuna


import numpy as np, pandas as pd

import matplotlib.pyplot as plt
import optuna
from lifelines.utils import concordance_index

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


train['race_group'].value_counts()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter
from scipy.stats import rankdata
from sklearn.preprocessing import quantile_transform

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y

def transform_rank_log(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times
    
    From https://www.kaggle.com/code/cdeotte/nn-mlp-baseline-cv-670-lb-676"""
    transformed = time.values.copy()
    mx = transformed[event == 1].max() # last patient who dies
    mn = transformed[event == 0].min() # first patient who survives
    transformed[event == 0] = time[event == 0] + mx - mn
    transformed = rankdata(transformed)
    transformed[event == 0] += len(transformed) * 2
    transformed = transformed / transformed.max()
    transformed = np.log(transformed)
    return - transformed

def transform_quantile(time, event):
    """Transform the target by stretching the range of eventful efs_times and compressing the range of event_free efs_times
    
    From https://www.kaggle.com/code/ambrosm/esp-eda-which-makes-sense"""
    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(- time[event == 1].values.reshape(-1, 1)).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_dead.min() - 0.3
    return transformed


race_group=sorted(train['race_group'].unique())
for race in race_group:
    # KP Meier
    train.loc[train['race_group']==race,"y"] = transform_survival_probability(train[train['race_group']==race], time_col='efs_time', event_col='efs')
    gap = 0.7*(train.loc[(train['race_group']==race)&(train['efs']==0)]['y'].max()-train.loc[(train['race_group']==race)&(train['efs']==1)]['y'].min())/2
    train.loc[(train['race_group']==race)&(train['efs']==0),'y']-=gap
    
    # Quantile KP Meier
    train.loc[train['race_group']==race,"quantile_kp_meier"] = transform_quantile(time = train[train['race_group']==race].efs_time, event=train[train['race_group']==race].efs)
    
    # Rank Loss KP Meier
    train.loc[train['race_group']==race,"rank_kp_meier"] = transform_rank_log(time = train[train['race_group']==race].efs_time, event=train[train['race_group']==race].efs)
    
plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()

plt.hist(train.loc[train.efs==1,"quantile_kp_meier"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"quantile_kp_meier"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Quantile Target y")
plt.ylabel("Density")
plt.title("Quantile KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()

plt.hist(train.loc[train.efs==1,"rank_kp_meier"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"rank_kp_meier"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Rank Log Target y")
plt.ylabel("Density")
plt.title("Rank Log KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


from metric import score


MIN_YEAR = train['year_hct'].min() # 2008
nunique2=[col for col in train.columns if train[col].nunique()==2 and col!='efs'] 
#nunique<50
nunique50=[col for col in train.columns if train[col].nunique()<50 and col not in ['efs','weight', 'year_hct']]+['age_group','dri_score_NA'] + ['year_hct_relative']

def FE(df):
    print("< deal with outlier >")
    df['nan_value_each_row'] = df.isnull().sum(axis=1)
    #year_hct=2020 only 4 rows.
    print("<convert to year_hct relative")
    df['year_hct_relative'] = df['year_hct'] - MIN_YEAR
    df.drop(columns=['year_hct'], inplace=True)
    # df['year_hct']=df['year_hct'].replace(2020,2019)
    df['age_group']=df['age_at_hct']//10
    #karnofsky_score 40 only 10 rows.
    df['karnofsky_score']=df['karnofsky_score'].replace(40,50)
    #hla_high_res_8=2 only 2 rows.
    df['hla_high_res_8']=df['hla_high_res_8'].replace(2,3)
    #hla_high_res_6=0 only 1 row.
    df['hla_high_res_6']=df['hla_high_res_6'].replace(0,2)
    #hla_high_res_10=3 only 1 row.
    df['hla_high_res_10']=df['hla_high_res_10'].replace(3,4)
    #hla_low_res_8=2 only 1 row.
    df['hla_low_res_8']=df['hla_low_res_8'].replace(2,3)
    df['dri_score']=df['dri_score'].replace('Missing disease status','N/A - disease not classifiable')
    df['dri_score_NA']=df['dri_score'].apply(lambda x:int('N/A' in str(x)))
    for col in ['diabetes','pulm_moderate','cardiac']:
        df.loc[df[col].isna(),col]='Not done'

    print("< cross feature >")
    df['donor_age-age_at_hct']=df['donor_age']-df['age_at_hct']
    df['comorbidity_score+karnofsky_score']=df['comorbidity_score']+df['karnofsky_score']
    df['comorbidity_score-karnofsky_score']=df['comorbidity_score']-df['karnofsky_score']
    df['comorbidity_score*karnofsky_score']=df['comorbidity_score']*df['karnofsky_score']
    df['comorbidity_score/karnofsky_score']=df['comorbidity_score']/df['karnofsky_score']
    
    print("< fillna >")
    df[nunique50]=df[nunique50].astype(str).fillna('NaN')
    
    print("< combine category feature >")
    for i in range(len(nunique2)):
        for j in range(i+1,len(nunique2)):
            df[nunique2[i]+nunique2[j]]=df[nunique2[i]].astype(str)+df[nunique2[j]].astype(str)
    
    # print("< drop useless columns >")
    # df.drop(['ID'],axis=1,inplace=True,errors='ignore')
    return df

train = FE(train)
test = FE(test)


RMV = ["ID","efs","efs_time","y", 'quantile_kp_meier', 'rank_kp_meier']
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)
print("Combined data shape:", combined.shape )
# # Store the original race group names and their corresponding codes
# race_group_categories = combined['race_group'].astype('category').cat.categories

# # Label encode race_group and capture the mapping
# combined['race_group'], _ = combined['race_group'].factorize()
# race_code_to_name = {code: name for code, name in enumerate(race_group_categories)}

# # Print the mapping for verification
# print("Race group encoding mapping:")
# for code, name in race_code_to_name.items():
#     print(f"{code}: {name}")
# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


race_code_to_name = {
    0: 'American Indian or Alaska Native',
1: 'Asian',
2: 'Black or African-American',
3: 'More than one race',
4: 'Native Hawaiian or other Pacific Islander',
5: 'White'

}





# Define target distribution and calculate importance weights
target_props = {
    'White': 1,  # Adjusted to realistic proportion from paper
    'Black or African-American': 0.09,
    'Asian': 0.04,
    'Native Hawaiian or other Pacific Islander': 0.02,
    'American Indian or Alaska Native': 0.03,
    'More than one race': 0.02
}

# Calculate importance weights for each sample
train_props = 1/len(target_props)  # Balanced training assumption
# Calculate importance weights using the correct mapping
importance_weights = (
    train['race_group']
    .map(lambda x:  train_props/ target_props[race_code_to_name[x]])
    .astype(float)
    .values
)

# Clip extreme weights for stability
importance_weights = np.clip(importance_weights, 0.1, 5)

importance_weights


RMV_FEATURES =  ['psych_disturb', 'graft_type', 'prod_type', 'in_vivo_tcd', 'dri_score_NA', 'rituximab']

FEATURES = [i for i in FEATURES if i not in RMV_FEATURES]
CATS = [i for i in CATS if i not in RMV_FEATURES]
len(FEATURES), len(CATS)



# Define the objective function for Optuna
def objective_xgb(trial, target_col = 'y'):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 128, 2048)
    max_depth = trial.suggest_int('max_depth', 2, 8)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    # gamma = trial.suggest_float('gamma', 0, 0.5)
    min_child_weight = trial.suggest_int('min_child_weight', 2, 8)
    reg_lambda = trial.suggest_float('reg_lambda', 0.1, 1)
    reg_alpha = trial.suggest_float('reg_alpha', 0.1, 1)
    
    # Initialize the model with the suggested hyperparameters
    model = XGBRegressor(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        # gamma=gamma,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        enable_categorical=True,
        tree_method='hist', # change to 'gpu_hist' in kaggle
        device='cuda',
        objective='reg:squarederror',
        eval_metric='rmse'
    )
    
    # Setup KFold CV; use KFold (not StratifiedKFold) for regression tasks.
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for train_index, valid_index in kf.split(train):
        # Extract training and validation splits from your train DataFrame.
        # Assumes train has columns: FEATURES + 'ID', 'efs', 'efs_time', 'race_group', and target 'quantile_kp_meier'
        X_train = train.loc[train_index, FEATURES].copy()
        y_train = train.loc[train_index, target_col]
        X_valid = train.loc[valid_index, FEATURES].copy()
        y_valid = train.loc[valid_index, target_col]
        
        # Get the importance weights for the training fold
        fold_weights = importance_weights[train_index]
        
        # Fit the model
        model.fit(
            X_train, y_train,
            sample_weight=fold_weights,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            # early_stopping_rounds=1024
        )
        
        # Predict on the validation fold
        preds = model.predict(X_valid)
        
        # Create the solution DataFrame from validation fold ground truth.
        # Must include 'ID', 'efs', 'efs_time', and 'race_group'
        solution = train.loc[valid_index, ["ID", "efs", "efs_time", "race_group"]].copy()
        
        # Create a submission DataFrame with predictions.
        submission = solution[["ID"]].copy()
        submission["prediction"] = preds
        
        # Compute the custom metric for this fold
        fold_metric = score(solution, submission, "ID")
        fold_scores.append(fold_metric)
    
    # Return the mean custom metric over the folds
    return np.mean(fold_scores)



# study_xgb = optuna.create_study(direction="maximize", study_name="XGB_Optim_race_kp_meier")
# study_xgb.optimize(lambda trial: objective_xgb(trial, target_col='y'), n_trials=64)
# print("XGBoost best hyperparameters:", study_xgb.best_trial.params)
# print("XGBoost best custom metric score:", study_xgb.best_trial.value)


# study_xgb = optuna.create_study(direction="maximize", study_name="XGB_Optim_quantile_kp_meier")
# study_xgb.optimize(lambda trial: objective_xgb(trial, target_col='quantile_kp_meier'), n_trials=64)
# print("XGBoost best hyperparameters:", study_xgb.best_trial.params)
# print("XGBoost best custom metric score:", study_xgb.best_trial.value)


# study_xgb = optuna.create_study(direction="maximize", study_name="XGB_Optim_ranklog_kp_meier")
# study_xgb.optimize(lambda trial: objective_xgb(trial, target_col='rank_kp_meier'), n_trials=64)
# print("XGBoost best hyperparameters:", study_xgb.best_trial.params)
# print("XGBoost best custom metric score:", study_xgb.best_trial.value)


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


def objective_cat(trial, target_col='y'):
    # Suggest hyperparameters for CatBoost
    bagging_temperature = trial.suggest_float('bagging_temperature', 0.1, 1.0)
    iterations = trial.suggest_int('iterations', 512, 1024)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 4, 10)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 0.1, 8)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 16, 64)
    random_strength = trial.suggest_float('random_strength', 0.1, 1.0)
    
    
    model = CatBoostRegressor(
        random_state=42,
        eval_metric='MAE',
        task_type='GPU',
        bagging_temperature=bagging_temperature,
        iterations=iterations,
        learning_rate=learning_rate,
        max_depth=max_depth,
        l2_leaf_reg=l2_leaf_reg,
        min_data_in_leaf=min_data_in_leaf,
        random_strength=random_strength,
        verbose=0
    )
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_index, valid_index in kf.split(train):
        X_train = train.loc[train_index, FEATURES].copy()
        y_train = train.loc[train_index, target_col]
        X_valid = train.loc[valid_index, FEATURES].copy()
        y_valid = train.loc[valid_index, target_col]
        
        fold_weights = importance_weights[train_index]
        
        model.fit(
            X_train, y_train,
            sample_weight=fold_weights,
            eval_set=(X_valid, y_valid),
            cat_features=CATS,
            early_stopping_rounds=100,
            verbose=False
        )
        
        preds = model.predict(X_valid)
        
        solution = train.loc[valid_index, ["ID", "efs", "efs_time", "race_group"]].copy()
        submission = solution[["ID"]].copy()
        submission["prediction"] = preds
        
        fold_metric = score(solution, submission, "ID")
        fold_scores.append(fold_metric)
        
    return np.mean(fold_scores)


# study_cat = optuna.create_study(direction="maximize", study_name="CatBoost_Optim_race_specific_kp_meier")
# study_cat.optimize(lambda trial: objective_cat(trial, target_col='y'), n_trials=64)
# print("CatBoost best hyperparameters:", study_cat.best_trial.params)
# print("CatBoost best custom metric score:", study_cat.best_trial.value)


study_cat = optuna.create_study(direction="maximize", study_name="CatBoost_Optim_quantile_kp_meier")
study_cat.optimize(lambda trial: objective_cat(trial, target_col='quantile_kp_meier'), n_trials=64)
print("CatBoost best hyperparameters:", study_cat.best_trial.params)
print("CatBoost best custom metric score:", study_cat.best_trial.value)


study_cat = optuna.create_study(direction="maximize", study_name="CatBoost_Optim_rank_kp_meier")
study_cat.optimize(lambda trial: objective_cat(trial, target_col='rank_kp_meier'), n_trials=64)
print("CatBoost best hyperparameters:", study_cat.best_trial.params)
print("CatBoost best custom metric score:", study_cat.best_trial.value)


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


def objective_lgb(trial, target_col='y'):
    # Suggest hyperparameters for LightGBM
    max_depth = trial.suggest_int('max_depth', 6, 12)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int('n_estimators', 512, 2048)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 0.8)
    colsample_bynode = trial.suggest_float('colsample_bynode', 0.4, 0.8)
    reg_alpha = trial.suggest_float('reg_alpha', 0.1, 1.0)
    reg_lambda = trial.suggest_float('reg_lambda', 1.0, 10.0)
    num_leaves = trial.suggest_int('num_leaves', 32, 128)
    
    model = LGBMRegressor(
        boosting_type="gbdt",
        metric='mae',
        random_state=42,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        colsample_bytree=colsample_bytree,
        colsample_bynode=colsample_bynode,
        verbose=-1,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        extra_trees=True,
        num_leaves=num_leaves,
        max_bin=255,
        importance_type='gain',
        device='gpu',
        gpu_use_dp=True
    )
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_index, valid_index in kf.split(train):
        X_train = train.loc[train_index, FEATURES].copy()
        y_train = train.loc[train_index, target_col]
        X_valid = train.loc[valid_index, FEATURES].copy()
        y_valid = train.loc[valid_index, target_col]
        
        fold_weights = importance_weights[train_index]
        
        model.fit(
            X_train, y_train,
            sample_weight=fold_weights,
            eval_set=[(X_valid, y_valid)],
            # early_stopping_rounds=100,
            # callbacks=[callback.early_stopping(100)]
            categorical_feature=CATS,
            # verbose=False
        )
        
        preds = model.predict(X_valid)
        
        solution = train.loc[valid_index, ["ID", "efs", "efs_time", "race_group"]].copy()
        submission = solution[["ID"]].copy()
        submission["prediction"] = preds
        
        fold_metric = score(solution, submission, "ID")
        fold_scores.append(fold_metric)
        
    return np.mean(fold_scores)


study_lgb = optuna.create_study(direction="maximize", study_name="LGBM_Optim_race_specific")
study_lgb.optimize(lambda trial: objective_lgb(trial, target_col='y'), n_trials=64)
print("LightGBM best hyperparameters:", study_lgb.best_trial.params)
print("LightGBM best custom metric score:", study_lgb.best_trial.value)


study_lgb = optuna.create_study(direction="maximize", study_name="LGBM_Optim_quantile")
study_lgb.optimize(lambda trial: objective_lgb(trial, target_col='quantile_kp_meier'), n_trials=64)
print("LightGBM best hyperparameters:", study_lgb.best_trial.params)
print("LightGBM best custom metric score:", study_lgb.best_trial.value)


study_lgb = optuna.create_study(direction="maximize", study_name="LGBM_Optim_rank")
study_lgb.optimize(lambda trial: objective_lgb(trial, target_col='rank_kp_meier'), n_trials=64)
print("LightGBM best hyperparameters:", study_lgb.best_trial.params)
print("LightGBM best custom metric score:", study_lgb.best_trial.value)


from concurrent.futures import ProcessPoolExecutor, as_completed

# Assume these objective functions are defined as in previous examples:
# - objective_xgb(trial, target_col)
# - objective_cat_cox(trial, target_col)
# - objective_lgb(trial, target_col)

# Wrap each study in its own function.
def run_xgb_study(target_col):
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"XGB_Optim_{target_col}"
    )
    study.optimize(lambda trial: objective_xgb(trial, target_col=target_col), n_trials=50)
    return {
        'model': 'XGBoost',
        'target_col': target_col,
        'best_params': study.best_trial.params,
        'best_score': study.best_trial.value
    }

def run_cat_study(target_col):
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"CatBoost_Cox_Optim_{target_col}"
    )
    study.optimize(lambda trial: objective_cat(trial, target_col=target_col), n_trials=64)
    return {
        'model': 'CatBoost_Cox',
        'target_col': target_col,
        'best_params': study.best_trial.params,
        'best_score': study.best_trial.value
    }

def run_lgb_study(target_col):
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"LGBM_Optim_{target_col}"
    )
    study.optimize(lambda trial: objective_lgb(trial, target_col=target_col), n_trials=64)
    return {
        'model': 'LightGBM',
        'target_col': target_col,
        'best_params': study.best_trial.params,
        'best_score': study.best_trial.value
    }




# # Define the list of target columns you want to optimize for.
# target_cols = ['y', 'quantile_kp_meier', 'rank_kp_meier']

# # Build a list of jobs for each combination of model and target column.
# jobs = []
# with ProcessPoolExecutor(max_workers=2) as executor:
#     for target in target_cols:
#         # jobs.append(executor.submit(run_xgb_study, target))
#         jobs.append(executor.submit(run_cat_study, target))
#         jobs.append(executor.submit(run_lgb_study, target))
    
#     # Collect the results as they complete.
#     results = []
#     for future in as_completed(jobs):
#         results.append(future.result())

# # Print results from all studies.
# for res in results:
#     print(f"Model: {res['model']}, Target: {res['target_col']}")
#     print("Best Params:", res['best_params'])
#     print("Best Score:", res['best_score'])
#     print("="*50)


# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1



# Define the objective function for Optuna
def objective_xgb_cox(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 128, 2048)
    max_depth = trial.suggest_int('max_depth', 2, 8)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    # gamma = trial.suggest_float('gamma', 0, 0.5)
    min_child_weight = trial.suggest_int('min_child_weight', 2, 8)
    reg_lambda = trial.suggest_float('reg_lambda', 0.1, 1)
    reg_alpha = trial.suggest_float('reg_alpha', 0.1, 1)
    
    # Initialize the model with the suggested hyperparameters
    model = XGBRegressor(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        # gamma=gamma,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        enable_categorical=True,
        tree_method='hist', # change to 'gpu_hist' in kaggle
        device='cuda',
        objective='survival:cox',
        eval_metric='cox-nloglik'
    )
    
    # Setup KFold CV; use KFold (not StratifiedKFold) for regression tasks.
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for train_index, valid_index in kf.split(train):
        # Extract training and validation splits from your train DataFrame.
        # Assumes train has columns: FEATURES + 'ID', 'efs', 'efs_time', 'race_group', and target 'quantile_kp_meier'
        X_train = train.loc[train_index, FEATURES].copy()
        y_train = train.loc[train_index, 'efs_time2']
        X_valid = train.loc[valid_index, FEATURES].copy()
        y_valid = train.loc[valid_index, 'efs_time2']
        
        # Get the importance weights for the training fold
        fold_weights = importance_weights[train_index]
        
        # Fit the model
        model.fit(
            X_train, y_train,
            sample_weight=fold_weights,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            # early_stopping_rounds=1024
        )
        
        # Predict on the validation fold
        preds = model.predict(X_valid)
        
        # Create the solution DataFrame from validation fold ground truth.
        # Must include 'ID', 'efs', 'efs_time', and 'race_group'
        solution = train.loc[valid_index, ["ID", "efs", "efs_time", "race_group"]].copy()
        
        # Create a submission DataFrame with predictions.
        submission = solution[["ID"]].copy()
        submission["prediction"] = preds
        
        # Compute the custom metric for this fold
        fold_metric = score(solution, submission, "ID")
        fold_scores.append(fold_metric)
    
    # Return the mean custom metric over the folds
    return np.mean(fold_scores)


study_xgb = optuna.create_study(direction="maximize", study_name="XGB_Optim_cox")
study_xgb.optimize(objective_xgb_cox, n_trials=64)
print("XGBoost best hyperparameters:", study_xgb.best_trial.params)
print("XGBoost best custom metric score:", study_xgb.best_trial.value)


def objective_cat_cox(trial):
    # Suggest hyperparameters for CatBoost
    # grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Lossguide', 'Depthwise'])
    min_child_samples = trial.suggest_int('min_child_samples', 1, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    num_trees = trial.suggest_int('num_trees', 4096, 7200)
    reg_lambda = trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
    num_leaves = trial.suggest_int('num_leaves', 16, 64)
    depth = trial.suggest_int('depth', 4, 10)
    
    # Initialize CatBoost model with Cox survival loss
    model = CatBoostRegressor(
        grow_policy='Lossguide',
        min_child_samples=min_child_samples,
        loss_function='Cox',
        learning_rate=learning_rate,
        random_state=42,
        task_type='CPU',
        num_trees=num_trees,
        reg_lambda=reg_lambda,
        num_leaves=num_leaves,
        depth=depth,
        verbose=0
    )
    
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_index, valid_index in kf.split(train):
        X_train = train.loc[train_index, FEATURES].copy()
        y_train = train.loc[train_index, 'efs_time2']
        X_valid = train.loc[valid_index, FEATURES].copy()
        y_valid = train.loc[valid_index, 'efs_time2']
        
        fold_weights = importance_weights[train_index]
        
        model.fit(
            X_train, y_train,
            sample_weight=fold_weights,
            eval_set=(X_valid, y_valid),
            cat_features=CATS,
            # early_stopping_rounds=100,
            verbose=False
        )
        
        preds = model.predict(X_valid)
        
        solution = train.loc[valid_index, ["ID", "efs", "efs_time", "race_group"]].copy()
        submission = solution[["ID"]].copy()
        submission["prediction"] = preds
        
        fold_metric = score(solution, submission, "ID")
        fold_scores.append(fold_metric)
        
    return np.mean(fold_scores)


study_cat_cox = optuna.create_study(direction="maximize", study_name="CatBoost_Optim_cox")
study_cat_cox.optimize(objective_cat_cox, n_trials=64)

# Print best hyperparameters and score
print("Best hyperparameters:", study_cat_cox.best_trial.params)
print("Best custom metric score:", study_cat_cox.best_trial.value)


# Wrap each study in its own function.
def run_xgb_cox_study():
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"XGB_Optim_cox"
    )
    study.optimize(objective_xgb_cox, n_trials=64)
    return {
        'model': 'XGBoost_cox',
        'best_params': study.best_trial.params,
        'best_score': study.best_trial.value
    }

def run_cat_cox_study():
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"CatBoost_Cox_Optim_cox"
    )
    study.optimize(objective_cat_cox, n_trials=64)
    return {
        'model': 'CatBoost_Cox',
        'best_params': study.best_trial.params,
        'best_score': study.best_trial.value
    }


# jobs = []
# with ProcessPoolExecutor(max_workers=2) as executor:
#     jobs.append(executor.submit(run_xgb_cox_study))
#     jobs.append(executor.submit(run_cat_cox_study))
    
#     results = []
#     for future in as_completed(jobs):
#         results.append(future.result())

# for res in results:
#     print(f"Model: {res['model']}")
#     print("Best Params:", res['best_params'])
#     print("Best Score:", res['best_score'])
#     print("="*50)


lgb_params={"boosting_type": "gbdt","metric": 'mae',
            'random_state': 42,  "max_depth": 9,"learning_rate": 0.1,
            "n_estimators": 768,"colsample_bytree": 0.6,"colsample_bynode": 0.6,
            "verbose": -1,"reg_alpha": 0.2,
            "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
            'importance_type': 'gain',#better than 'split'
            'device':'gpu','gpu_use_dp':True
           }

cat_params={'random_state':42,'eval_metric' : 'MAE',
            'bagging_temperature': 0.50,'iterations': 650,
            'learning_rate': 0.1,'max_depth': 8,
            'l2_leaf_reg': 1.25,'min_data_in_leaf': 24,
            'random_strength' : 0.25, 'verbose': 0,
            'task_type':'GPU',
            }
xgb_params={'random_state': 42, 'n_estimators': 256, 
            'learning_rate': 0.1, 'max_depth': 6,
            'reg_alpha': 0.08, 'reg_lambda': 0.8, 
            'subsample': 0.95, 'colsample_bytree': 0.6, 
            'min_child_weight': 3,'early_stopping_rounds':1024,
             'enable_categorical':True,'tree_method':'gpu_hist'
            }

ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'reg_lambda': 8.0,
        'depth': 8
    }

lgb_params = {
    'objective': 'regression',
    'min_child_samples': 32,
    'num_iterations': 6000,
    'learning_rate': 0.03,
    'extra_trees': True,
    'reg_lambda': 8.0,
    'reg_alpha': 0.1,
    'num_leaves': 64,
    'metric': 'rmse',
    'max_depth': 8,
    'device': 'cpu',
    'max_bin': 128,
    'verbose': -1,
    'seed': 42
}

cox1_params = {
    'grow_policy': 'Depthwise',
    'min_child_samples': 8,
    'loss_function': 'Cox',
    'learning_rate': 0.03,
    'random_state': 42,
    'task_type': 'CPU',
    'num_trees': 6000,
    'reg_lambda': 8.0,
    'depth': 8
}

cox2_params = {
    'grow_policy': 'Lossguide',
    'min_child_samples': 2,
    'loss_function': 'Cox',
    'learning_rate': 0.03,
    'random_state': 42,
    'task_type': 'CPU',
    'num_trees': 6000,
    'reg_lambda': 8.0,
    'num_leaves': 32,
    'depth': 8
}

