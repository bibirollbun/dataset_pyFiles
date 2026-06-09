import numpy as np
import polars as pl
import pandas as pd
from sklearn.base import clone
from copy import deepcopy
import optuna
from scipy.optimize import minimize
import os
from scipy.stats import mode

import re
from colorama import Fore, Style

from tqdm import tqdm
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import *
from sklearn.metrics import *

SEED = 42
n_splits = 5


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

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

featuresCols = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
                'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
                'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
                'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
                'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
                'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
                'PAQ_C-PAQ_C_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',
                'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
                'PreInt_EduHx-computerinternet_hoursday', 'sii']

featuresCols += time_series_cols

train = train[featuresCols]
train = train.dropna(subset='sii')

cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season',
            'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

def update(df):
    for c in cat_c:
        df[c] = df[c].fillna('Missing')
        df[c] = df[c].astype('category')
    return df

train = update(train)
test = update(test)

def create_mapping(column, dataset):
    unique_values = dataset[column].unique()
    return {value: idx for idx, value in enumerate(unique_values)}

for col in cat_c:
    mapping_train = create_mapping(col, train)

    train[col] = train[col].replace(mapping_train).astype(int)
    test[col] = test[col].replace(mapping_train).astype(int)

print(f'Train Shape : {train.shape} || Test Shape : {test.shape}')

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.base import clone
from scipy.optimize import minimize
from tqdm import tqdm
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

# Evaluation functions
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

# Model configurations
n_splits = 5
seed_list = [42, 0, 2024, 7269173, 1234]

# LightGBM parameters
lgb_params = {
    'learning_rate': 0.03755757104848504,
    'max_depth': 12,
    'num_leaves': 18,
    'min_data_in_leaf': 3,
    'feature_fraction': 0.723690362968002,
    'bagging_fraction': 0.688232590484764,
    'bagging_freq': 5,
    'lambda_l1': 0.18512987285245963,
    'lambda_l2': 0.18435628737334625,
    'verbose': -1,
    'n_estimators': 200,
    'objective': 'regression'
}

# XGBoost parameters
xgb_params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'min_child_weight': 3,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': 0
}

# Features and target
X = train.drop(['sii'], axis=1)
y = train['sii']
test_data = test

# Define model training function
def TrainML(model_class, test_data, seed_list):
    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_S = []
    test_S = []

    oof_non_rounded = np.zeros(len(y), dtype=float)
    oof_rounded = np.zeros(len(y), dtype=int)
    test_preds = np.zeros((len(test_data), n_splits))

    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y.astype(int)), desc="Training Folds", total=n_splits)):

        random_seed = np.random.choice(seed_list)

        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(model_class)
        if hasattr(model, 'random_state'):
            model.set_params(random_state=random_seed)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        test_preds[:, fold] = model.predict(test_data)

        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = np.round(y_val_pred).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded
        train_kappa = quadratic_weighted_kappa(y_train, np.round(y_train_pred).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)
        train_S.append(train_kappa)
        test_S.append(val_kappa)

        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")

    print(f"Train : {np.mean(train_S):.4f}")
    print(f"Validation : {np.mean(test_S):.4f}")

    KappaOPtimizer = minimize(evaluate_predictions,
                                 x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded),
                                 method='Nelder-Mead')

    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)
    print(f"----> || Optimized QWK: {tKappa:.3f}")
    print(f"Optimal thresholds: {KappaOPtimizer.x}")

    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_Rounder(tpm, KappaOPtimizer.x)

    return tpTuned, oof_non_rounded, KappaOPtimizer.x

# Initialize models
lgb_model = lgb.LGBMRegressor(**lgb_params)
#xgb_model = xgb.XGBRegressor(**xgb_params)

print("Training LightGBM model...")
lgb_preds, lgb_oof, lgb_thresholds = TrainML(lgb_model, test_data, seed_list)

#print("\nTraining XGBoost model...")
#xgb_preds, xgb_oof, xgb_thresholds = TrainML(xgb_model, test_data, seed_list)



#print("\nTraining XGBoost model...")
#xgb_preds, xgb_oof, xgb_thresholds = TrainML(xgb_model, test_data, seed_list)

# # Create ensemble by averaging predictions
# print("\nCreating ensemble...")
# # Simple average of OOF predictions
# ensemble_oof = (lgb_oof + xgb_oof) / 2

# # Optimize thresholds for ensemble
# ensemble_optimizer = minimize(evaluate_predictions,
#                                  x0=[0.5, 1.5, 2.5], args=(y, ensemble_oof),
#                                  method='Nelder-Mead')

# ensemble_tuned_oof = threshold_Rounder(ensemble_oof, ensemble_optimizer.x)
# ensemble_kappa = quadratic_weighted_kappa(y, ensemble_tuned_oof)
# print(f"Ensemble OOF QWK: {ensemble_kappa:.4f}")
# print(f"Ensemble optimal thresholds: {ensemble_optimizer.x}")

# # Create ensemble test predictions
# ensemble_test = (lgb_preds + xgb_preds) / 2
# final_preds = threshold_Rounder(ensemble_test, ensemble_optimizer.x)



xgb_submission = pd.DataFrame({'id': sample['id'], 'sii': lgb_preds})
xgb_submission.to_csv('submission.csv', index=False)
# Create submission file
submission = pd.DataFrame({
    'id': sample['id'],
    'sii': lgb_preds
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")


submission

