import os
import re
import numpy as np
import polars as pl
import pandas as pd
from copy import deepcopy
from scipy.optimize import minimize
from scipy.stats import mode
from colorama import Fore, Style
from tqdm import tqdm
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.base import clone
import matplotlib.pyplot as plt

SEED = 2025
n_splits = 7


# Processing parquet files is not really matter cause we don't use them anyway..
def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"Stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    
    return df

train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")
time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

# merge HBN instruments & actigraphy file
train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


def feature_engineering(df):
    # convert season columns to integer
    season_cols = [col for col in df.columns if 'Season' in col]
    cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 
          'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']
    
    def update(df):
        for c in cat_c: 
            df[c] = df[c].fillna('Missing')
            df[c] = df[c].astype('category')
        return df
            
    df = update(df)
    
    def create_mapping(column, dataset):
        unique_values = dataset[column].unique()
        return {value: idx for idx, value in enumerate(unique_values)}
    
    for col in cat_c:
        mapping_df = create_mapping(col, df)
        
        df[col] = train[col].replace(mapping_df).astype(int)
        
    # assign age group for each participants for normalization
    def assign_age_group(age):
        thresholds = [5, 6, 7, 8, 10, 12, 14, 18, 22]
        for i, j in enumerate(thresholds):
            if age <= j:
                return i
        return np.nan
    
    # age groups
    df["age_group"] = df['Basic_Demos-Age'].apply(assign_age_group)

    df['BMI'] = df[['Physical-BMI', 'BIA-BIA_BMI']].max(axis=1)
    df['FGC_GS'] =  df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].max(axis=1)
    df['FGC_SR'] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].max(axis=1)
    
    features_to_normalize = [
        'BMI',
        'FGC_GS',
        'FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL',
        'FGC_SR',
        'BIA-BIA_BMR', 'BIA-BIA_DEE', 
        'BIA-BIA_FFM'
    ]
    # Normalized Value= Standard Deviation / Original Value−Mean
    group_stats = df.groupby('age_group')[features_to_normalize].agg(['mean', 'std']).reset_index()

    group_stats.columns = ['_'.join(col).strip('_') for col in group_stats.columns.values]
    
    df = df.merge(group_stats, on='age_group', how='left')
    
    for feature in features_to_normalize:
        df[f"{feature.split('_')[-1]}_norm"] = (df[feature] - df[f"{feature}_mean"]) / df[f"{feature}_std"]

    # Intracellular Water / Extracellular Water rate
    df["ICW_ECW"] = df["BIA-BIA_ECW"] / df["BIA-BIA_ICW"]
    # FGC_Zones_mean = sum of FGC_Zone
    zones = ['FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
         'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
         'FGC-FGC_TL_Zone']
    df['FGC_Zones_mean'] = df[zones].sum(axis=1)
    # Internet_Hours_Age
    df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
    # columns to drops
    drop_feats = ['FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
                  'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone',
                  'Physical-BMI', 'BIA-BIA_BMI', 'FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL', 'FGC-FGC_SRL', 'FGC-FGC_SRR',
                 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_Frame_num', "BIA-BIA_FFM",'age_group']
    df = df.drop(drop_feats, axis=1) 

    return df



train = feature_engineering(train)
test = feature_engineering(test)
# all the important 
featuresCols = [
                'Basic_Demos-Age', 'Basic_Demos-Sex',
                'CGAS-CGAS_Score',
                'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC','BIA-BIA_ECW',
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat',
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                'BIA-BIA_TBW',  'PAQ_A-PAQ_A_Total',
                'PAQ_C-PAQ_C_Total', 'SDS-SDS_Total_Raw','SDS-SDS_Total_T',
                'FGC_Zones_mean',
                'GS_norm',"CU_norm","PU_norm","TL_norm","SR_norm","BMR_norm","DEE_norm","FFM_norm",
                "BMI_norm",
                'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 
                'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season','Internet_Hours_Age'
]
featuresCols += time_series_cols

trainFeatures = featuresCols + ['sii']
train = train[trainFeatures]
train = train.dropna(subset='sii')

test = test[featuresCols]
# drops rows with null sii
train = train[train['sii'].notnull()]


# Show null percentage of features
data_trainning = train[train['sii'].notnull()]
data_trainning_null = data_trainning.isnull().sum() * 100/len(data_trainning)
data_trainning_sorted = data_trainning_null.sort_values()

plt.figure(figsize=(24,5))
sns.barplot(x=data_trainning_sorted.index, y=data_trainning_sorted.values, color='skyblue')
plt.title("Percentage of missing values in each column")
plt.xlabel("Columns")
plt.xticks(rotation=90)
plt.ylabel("Percentage of missing values")
plt.show()


# drops columns with greater than 60% NaN values
columns_to_drop = data_trainning_sorted[(data_trainning_sorted > 60)].index
train = train.drop(columns=columns_to_drop)
test  = test.drop(columns=columns_to_drop)


feature_col = train.drop(['sii'], axis=1).columns
all_importances = pd.DataFrame({'feature': feature_col})
# scoring function
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))
# round thresholds before scoring
def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

# train function
def TrainML(model_class, test_data):
    X = train.drop(['sii'], axis=1)
    y = train['sii']

    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    train_S = []
    test_S = []
    
    oof_non_rounded = np.zeros(len(y), dtype=float) 
    oof_rounded = np.zeros(len(y), dtype=int) 
    test_preds = np.zeros((len(test_data), n_splits))

    feature_names = X.columns

    lgb_importances_list = []
    xgb_importances_list = []
    cat_importances_list = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded

        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        test_S.append(val_kappa)
        
        test_preds[:, fold] = model.predict(test_data)
        
        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")
        clear_output(wait=True)

        named_estimators = model.named_estimators_
        
        # LightGBM
        if 'lightgbm' in named_estimators and hasattr(named_estimators['lightgbm'], 'feature_importances_'):
            lgb_importances_list.append(named_estimators['lightgbm'].feature_importances_)

        # XGBoost
        if 'xgboost' in named_estimators and hasattr(named_estimators['xgboost'], 'feature_importances_'):
            xgb_importances_list.append(named_estimators['xgboost'].feature_importances_)

        # CatBoost
        if 'catboost' in named_estimators and hasattr(named_estimators['catboost'], 'get_feature_importance'):
            cat_importances_list.append(named_estimators['catboost'].get_feature_importance())


    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")

    # optimize sii thresholds
    KappaOPtimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded), 
                              method='Nelder-Mead')
    assert KappaOPtimizer.success, "Optimization did not converge."
    
    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)
    print(f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}")

    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_Rounder(tpm, KappaOPtimizer.x)
    
    submission = pd.DataFrame({
        'id': sample['id'],
        'sii': tpTuned
    })

    def mean_importances(importances_list):
        if len(importances_list) > 0:
            return np.mean(importances_list, axis=0)
        else:
            return None

    lgb_mean = mean_importances(lgb_importances_list)
    xgb_mean = mean_importances(xgb_importances_list)
    cat_mean = mean_importances(cat_importances_list)

    # plot features importance
    def normalize_importances(importance_array):
        if importance_array is not None:
            return importance_array / importance_array.sum()
        else:
            return None

    lgb_mean_normalized = normalize_importances(lgb_mean)
    xgb_mean_normalized = normalize_importances(xgb_mean)
    cat_mean_normalized = normalize_importances(cat_mean)
    
    if lgb_mean_normalized is not None:
        lgb_df = pd.DataFrame({'feature': feature_names, 'importance': lgb_mean_normalized}).sort_values('importance', ascending=False)
        plt.figure(figsize=(10,20))
        plt.barh(lgb_df['feature'], lgb_df['importance'])
        plt.gca().invert_yaxis()
        plt.title("LightGBM Feature Importance")
        plt.show()
        all_importances['LightGBM'] = lgb_mean_normalized

    if xgb_mean_normalized is not None:
        xgb_df = pd.DataFrame({'feature': feature_names, 'importance': xgb_mean_normalized}).sort_values('importance', ascending=False)
        plt.figure(figsize=(10,20))
        plt.barh(xgb_df['feature'], xgb_df['importance'])
        plt.gca().invert_yaxis()
        plt.title("XGBoost Feature Importance")
        plt.show()
        all_importances['XGBoost'] = xgb_mean_normalized

    if cat_mean_normalized is not None:
        cat_df = pd.DataFrame({'feature': feature_names, 'importance': cat_mean_normalized}).sort_values('importance', ascending=False)
        plt.figure(figsize=(10,20))
        plt.barh(cat_df['feature'], cat_df['importance'])
        plt.gca().invert_yaxis()
        plt.title("CatBoost Feature Importance")
        plt.show()
        all_importances['CatBoost'] = cat_mean_normalized

    return submission


# define custom scorer for quadratic weighted kappa
def quadratic_weighted_kappa_scorer(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

kappa_scorer = make_scorer(quadratic_weighted_kappa_scorer, greater_is_better=True)

# parameter grid for LightGBM
lightgbm_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [6, 8, 10, 12,14],
    'num_leaves': [31, 63, 127, 255],
    'min_data_in_leaf': [20, 40, 60, 80, 100, 120, 140, 160],
    'feature_fraction': [0.2, 0.4, 0.8, 1.0],
    'bagging_fraction': [0.2, 0.4, 0.8, 1.0],
    'bagging_freq': [1, 3, 5, 7, 9, 11],
    'lambda_l1': [0, 0.1, 1, 10],
    'lambda_l2': [0, 0.1, 1, 10],
    'n_estimators': [100, 200, 300, 400, 500]
}

# parameter grid for XGBoost
xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9, 11],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1, 5, 10],
    'reg_lambda': [0, 0.1, 1, 5, 10],
    'gamma': [0, 0.1, 0.5, 1, 2, 5, 10]
}



def optimize_xgboost(X, y, param_grid, cv, scoring, n_iter=20, random_state=SEED):
    xgb = XGBRegressor(random_state=random_state, verbosity=0, tree_method='auto')
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )
    random_search.fit(X, y)
    print(f"Best XGBoost Params: {random_search.best_params_}")
    print(f"Best XGBoost Score: {random_search.best_score_}")
    return random_search.best_estimator_

def optimize_lightgbm(X, y, param_grid, cv, scoring, n_iter=20, random_state=SEED):
    lgbm = LGBMRegressor(random_state=random_state, verbose=-1)
    random_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=4,
        random_state=random_state,
        n_jobs=-1,
        refit = True
    )
    random_search.fit(X, y)
    print(f"Best LightGBM Params: {random_search.best_params_}")
    print(f"Best LightGBM Score: {random_search.best_score_}")
    return random_search.best_estimator_

X = train.drop(['sii'], axis=1)
y = train['sii']

# cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

best_lgbm = optimize_lightgbm(X, y, lightgbm_param_grid, cv, kappa_scorer)
best_xgb = optimize_xgboost(X, y, xgb_param_grid, cv, kappa_scorer)


# optimized parameters
Light_Params = {
'num_leaves': 31, 'n_estimators': 100, 'min_data_in_leaf': 40, 'max_depth': 14, 'learning_rate': 0.05, 'lambda_l2': 1, 'lambda_l1': 0, 'feature_fraction': 0.8, 'bagging_freq': 7, 'bagging_fraction': 1.0,
'random_state' :SEED,
               }
XGB_Params = {
'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 200, 'max_depth': 11, 'learning_rate': 0.4, 'gamma': 0.5, 'colsample_bytree': 1.0,    'random_state': SEED,
    'tree_method': 'auto',
}
Light = LGBMRegressor(**Light_Params, verbose=-1)
XGB_Model = XGBRegressor(**XGB_Params)
voting_model = VotingRegressor(estimators=[
   ('lightgbm', Light),
   ('xgboost', XGB_Model)
])

submission = TrainML(voting_model, test)
submission.to_csv('submission.csv', index=False)




