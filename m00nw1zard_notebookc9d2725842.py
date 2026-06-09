import torch

import numpy as np
import pandas as pd
import os
import re
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim

from colorama import Fore, Style
from IPython.display import clear_output
import warnings
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None



def process_parquet_file_activity_day_night(filename, dirname, daytime_start='06:00:00', daytime_end='20:00:00',nighttime_start='20:00:00', nighttime_end='06:00:00'):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    daytime_start_dt  = pd.to_datetime(daytime_start, format='%H:%M:%S').time()
    daytime_start_sec = (daytime_start_dt.hour * 3600 + daytime_start_dt.minute * 60 + daytime_start_dt.second) * 10 ** 9
    daytime_end_dt  = pd.to_datetime(daytime_end, format='%H:%M:%S').time()
    daytime_end_sec = (daytime_end_dt.hour * 3600 + daytime_end_dt.minute * 60 + daytime_end_dt.second) * 10 ** 9
    
    nighttime_start_dt  = pd.to_datetime(nighttime_start, format='%H:%M:%S').time()
    nighttime_start_sec = (nighttime_start_dt.hour * 3600 + nighttime_start_dt.minute * 60 + nighttime_start_dt.second) * 10 ** 9
    nighttime_end_dt  = pd.to_datetime(nighttime_end, format='%H:%M:%S').time()
    nighttime_end_sec = (nighttime_end_dt.hour * 3600 + nighttime_end_dt.minute * 60 + nighttime_end_dt.second) * 10 ** 9
    
    df['is_daytime'] = (df['time_of_day'] >= daytime_start_sec) & (df['time_of_day'] < daytime_end_sec)
    df['is_nighttime'] = (df['time_of_day'] >= nighttime_start_sec) | (df['time_of_day'] < nighttime_end_sec)
    
    df_worn = df[df['non-wear_flag'] == 0.0]
    target_columns = ["enmo", "light", "battery_voltage"]
    time_categories = ["is_daytime", "is_nighttime"]
    def extract_stats(data):
        return [
            data.mean(), 
            data.std(), 
            data.max(), 
            data.min(), 
            data.diff().mean(), 
            data.diff().std()
        ]
    time_features = {}
    for col in target_columns:
        for time_cat in time_categories:
            if time_cat == 'is_daytime':
                time_label = 'day'
            elif time_cat == 'is_nighttime':
                time_label = 'night'
            else:
                time_label = time_cat
            
            filtered_data = df.loc[df[time_cat], col]
            
            stats = extract_stats(filtered_data)
            
            stat_names = ['mean', 'std', 'max', 'min', 'diff_mean', 'diff_std']
            for stat, stat_name in zip(stats, stat_names):
                feature_key = f"{col}_{time_label}_{stat_name}"
                time_features[feature_key] = stat
    
    time_features['id'] = filename.split('=')[1]
    time_series_df = pd.DataFrame([time_features])
    return time_series_df
def load_time_series_activity_day_night(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_parquet_file_activity_day_night(fname, dirname), ids), total=len(ids)))
    
    return pd.concat(results, ignore_index=True)

def feature_engineering(df):
    # season_cols = [col for col in df.columns if 'Season' in col]
    # df = df.drop(season_cols, axis=1) 
    
    # From here on own features
    def assign_group(age):
        thresholds = [5, 6, 7, 8, 10, 12, 14, 17, 22]
        for i, j in enumerate(thresholds):
            if age <= j:
                return i
        return np.nan
    
    # Age groups
    df["group"] = df['Basic_Demos-Age'].apply(assign_group)
    
    # BMI 
    BMI_map = {0: 16.3,1: 15.9,2: 16.1,3: 16.8,4: 17.3,5: 19.2,6: 20.2,7: 22.3, 8: 23.6}
    df['BMI_mean_norm'] = df[['Physical-BMI', 'BIA-BIA_BMI']].mean(axis=1) / df["group"].map(BMI_map)
    # df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
    # FGC zone aggregate
    zones = ['FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
             'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
             'FGC-FGC_TL_Zone']
    
    df['FGC_Zones_mean'] = df[zones].mean(axis=1)
    df['FGC_Zones_min'] = df[zones].min(axis=1)
    df['FGC_Zones_max'] = df[zones].max(axis=1)
    
    # Grip
    GSD_max_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 16.2, 5: 19.9, 6: 26.1, 7: 31.3, 8: 35.4}
    GSD_min_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 14.4, 5: 17.8, 6: 23.4, 7: 27.8, 8: 31.1}
    
    df['GS_max'] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].max(axis=1) / df["group"].map(GSD_max_map)
    df['GS_min'] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].min(axis=1) / df["group"].map(GSD_min_map)
    
    # Curl-ups, push-ups, trunk-lifts... normalized based on age-group
    cu_map = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 10.0, 5: 14.0, 6: 20.0, 7: 20.0, 8: 20.0}
    pu_map = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 7.0, 6: 8.0, 7: 10.0, 8: 14.0}
    tl_map = {0: 8.0, 1: 8.0, 2: 8.0, 3: 9.0, 4: 9.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0}
    
    df["CU_norm"] = df['FGC-FGC_CU'] / df['group'].map(cu_map)
    df["PU_norm"] = df['FGC-FGC_PU'] / df['group'].map(pu_map)
    df["TL_norm"] = df['FGC-FGC_TL'] / df['group'].map(tl_map)
    
    # Reach 
    df["SR_min"] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].min(axis=1)
    df["SR_max"] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].max(axis=1)

    # BIA Features
    # Energy Expenditure
    bmr_map = {0: 934.0, 1: 941.0, 2: 999.0, 3: 1048.0, 4: 1283.0, 5: 1255.0, 6: 1481.0, 7: 1519.0, 8: 1650.0}
    dee_map = {0: 1471.0, 1: 1508.0, 2: 1640.0, 3: 1735.0, 4: 2132.0, 5: 2121.0, 6: 2528.0, 7: 2566.0, 8: 2793.0}
    df["BMR_norm"] = df["BIA-BIA_BMR"] / df["group"].map(bmr_map)
    df["DEE_norm"] = df["BIA-BIA_DEE"] / df["group"].map(dee_map)
    df["DEE_BMR"] = df["BIA-BIA_DEE"] - df["BIA-BIA_BMR"]

    # FMM
    ffm_map = {0: 42.0, 1: 43.0, 2: 49.0, 3: 54.0, 4: 60.0, 5: 76.0, 6: 94.0, 7: 104.0, 8: 111.0}
    df["FFM_norm"] = df["BIA-BIA_FFM"] / df["group"].map(ffm_map)

    # ECW ICW
    df["ICW_ECW"] = df["BIA-BIA_ECW"] / df["BIA-BIA_ICW"]
    
    drop_feats = ['FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
                  'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone',
                  'Physical-BMI', 'BIA-BIA_BMI', 'FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL', 'FGC-FGC_SRL', 'FGC-FGC_SRR',
                 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_Frame_num', "BIA-BIA_FFM"]
    df = df.drop(drop_feats, axis=1) 
    return df



train_ts = load_time_series_activity_day_night("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series_activity_day_night("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")



train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
season_cols = [col for col in train.columns if 'Season' in col]
train = train.drop(season_cols, axis=1) 
season_cols = [col for col in test.columns if 'Season' in col]
test = test.drop(season_cols, axis=1) 
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')
train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')
PCIAT_drops = ['PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 
               'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13',
               'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20']
train = train.drop(PCIAT_drops, axis=1) 


imputer = KNNImputer(n_neighbors=10)
numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns
# infer NaN values of numerical cols by using KNNImputer
imputed_data = imputer.fit_transform(train[numeric_cols])
train_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)
train_imputed['sii'] = train_imputed['sii'].round().astype(int)
# keep the non-numerical columns
for col in train.columns:
    if col not in numeric_cols:
        train_imputed[col] = train[col]


train = train_imputed
train = train.dropna(thresh=20, axis=0)

pciat_feature_cols = [col for col in train.columns 
                      if col not in ['id','sii','PCIAT-PCIAT_Total']]
X_pciat = train[pciat_feature_cols].copy()
y_pciat = train['PCIAT-PCIAT_Total'].copy()


Light_Params = {
    'learning_rate': 0.020781216498091417,
    'max_depth': 12, 
    'num_leaves': 124,
    'min_data_in_leaf': 44, 
    'feature_fraction': 0.7738396785928227, 
    'bagging_fraction': 0.9094982620399213, 
    'bagging_freq': 4,
    'lambda_l1': 8.867251253806305,
    'lambda_l2': 8.039317955277623, 
    'n_estimators': 457
}

pciat_model = LGBMRegressor(**Light_Params, seed=2025,verbose = -1)
pciat_model.fit(X_pciat, y_pciat)

# Predict PCIAT for the train set (for optional inspection):
train['PCIAT_pred'] = pciat_model.predict(X_pciat)

# Now predict PCIAT for the test set
# (the same columns used in X_pciat must exist in test)
test['PCIAT-PCIAT_Total'] = pciat_model.predict(test[pciat_feature_cols])
train = feature_engineering(train)
test = feature_engineering(test)

train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)


# Plot distribution of total scores which determine the sii
# Note the excess zeros -> consider other objective functions
sns.set_theme(style="whitegrid")
plt.hist(train['PCIAT-PCIAT_Total'], bins=50, color="darkorange")
plt.title('Score Distribution')
plt.show()


df_train = train_ts.drop('id', axis=1)
df_test = test_ts.drop('id', axis=1)
time_series_cols = df_train.columns.tolist() 
train = train.drop('id', axis=1)
test  = test.drop('id', axis=1)   


featuresCols = [
    'Basic_Demos-Age', 'Basic_Demos-Sex',
                'CGAS-CGAS_Score',
                # 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC','BIA-BIA_ECW',
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat',
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                'BIA-BIA_TBW',  'PAQ_A-PAQ_A_Total',
                'PAQ_C-PAQ_C_Total', 'SDS-SDS_Total_Raw','SDS-SDS_Total_T',
                'PreInt_EduHx-computerinternet_hoursday', 'FGC_Zones_mean','FGC_Zones_min','FGC_Zones_max',
                'GS_max','GS_min',"CU_norm","PU_norm","TL_norm","SR_min","SR_max","BMR_norm","DEE_norm","DEE_BMR","ICW_ECW","FFM_norm",
                'BMI_mean_norm','PCIAT-PCIAT_Total'
]

featuresCols += time_series_cols

trainFeatures = featuresCols + ['sii']

train = train[trainFeatures]
train = train.dropna(subset='sii')
test = test[featuresCols]


feature_col = train.drop(['sii'], axis=1).columns
all_importances = pd.DataFrame({'feature': feature_col})

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)
    
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


SEED = 2025
n_splits = 5
# Model parameters for LightGBM
Light_Params = {
    'learning_rate': 0.09166688365135439,
    'max_depth': 12,
    'num_leaves': 432,
    'min_data_in_leaf': 17, 
    'feature_fraction': 0.9718570020641087, 
    'bagging_fraction': 0.8811731013548587,
    'bagging_freq': 3, 
    'lambda_l1': 0.4980199306309456,
    'lambda_l2': 0.6452996657412411, 
    'n_estimators': 424,
    'random_state':SEED,
    'device': 'cpu'
}


# XGBoost parameters
XGB_Params = {
    'learning_rate': 0.15782155024687242,
    'max_depth': 4, 
    'n_estimators': 365, 
    'subsample': 0.9226564408362041,
    'colsample_bytree': 0.7198200130798219, 
    'reg_alpha': 0.3728342426755272, 
    'reg_lambda': 3.662453322506941,
    'random_state': SEED,
    'tree_method': 'auto'

}


CatBoost_Params = {
    'learning_rate': 0.05,
    'depth': 6,
    'iterations': 200,
    'random_seed': SEED,
    'verbose': 0,
    'l2_leaf_reg': 10,  # Increase this value
    'task_type': 'CPU'

}

# Create model instances
Light = LGBMRegressor(**Light_Params,  verbose=-1)
XGB_Model = XGBRegressor(**XGB_Params)
CatBoost_Model = CatBoostRegressor(**CatBoost_Params)

voting_model = VotingRegressor(estimators=[
    ('lightgbm', Light),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model)
])
submission = TrainML(voting_model, test)
submission.to_csv('submission.csv', index=False)





