from tqdm.notebook import tqdm
from pathlib import Path
import joblib 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init
from scipy.optimize import minimize
from itertools import chain

import re

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.cluster import KMeans
from datetime import datetime, date


import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import optuna
import torch
from sklearn.metrics import mean_squared_error

init(autoreset=True)

import warnings
warnings.filterwarnings("ignore")


# GPU setup
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
SNS_CMAP = 'Pastel1'
sns.set_palette(SNS_CMAP)

colors = sns.palettes.color_palette(SNS_CMAP)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)


SEED = 42
FOLDS = 5
ALPHA = 0.1


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
   """Competitor's exact Winkler Score implementation."""
   y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
   width = upper - lower
   penalty_lower = 2 / alpha * (lower - y_true)
   penalty_upper = 2 / alpha * (y_true - upper)
   score = width + np.where(y_true < lower, penalty_lower, 0) + np.where(y_true > upper, penalty_upper, 0)
   if return_coverage:
       coverage = np.mean((y_true >= lower) & (y_true <= upper))
       return np.mean(score), coverage
   return np.mean(score)

train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")
train.index = train['id']
test.index = test['id']

submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")


target = 'sale_price'
print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


# --- 2. EDA & Preprocessing ---
print("\nPreprocessing...")

 



def create_zoning_flags(zoning_code):
    zoning_code = zoning_code.upper()

    return {
        "is_single_family": int(bool(re.search(r'\bRS|SF|R-|SR\b', zoning_code))),
        "is_multifamily": int(bool(re.search(r'\bMF|RM|LR|MR|RA\d|R-3|R-4|UR\b', zoning_code))),
        "is_commercial_allowed": int(bool(re.search(r'\bC|CB|NC|PR-C|COMMERCIAL\b', zoning_code))),
        "is_mixed_use": int(bool(re.search(r'\bMU|MUR|NC\d|LR\d RC|RC\b', zoning_code))),
        "is_planned_dev": int(bool(re.search(r'\bPUD|DPA|PR-C\b', zoning_code))),
        "is_overlay_zone": int(bool(re.search(r'\(M|RC|SO|SHO|SSHO|SL|PSO|OP\b', zoning_code))),
        "is_high_density": int(bool(re.search(r'RA\d|R-3|R-4|LR1|LR2|LR3|MR\b', zoning_code))),
        "has_lot_size_restriction": int(bool(re.search(r'\d{3,5}', zoning_code))),  # e.g. 7200
        "is_industrial": int(bool(re.search(r'\bIG|MIT|L-1|L-2|L-3|IND\b', zoning_code))),
        "is_agricultural": int(bool(re.search(r'\bRA|A10\b', zoning_code))),
        "is_near_urban_core": int(bool(re.search(r'\bUR|UC|NC|TC\b', zoning_code))),
        "is_townhouse_allowed": int(bool(re.search(r'RC|LR\d|MR\b', zoning_code))),
    }



def retrieve_neighbours(model, X, y, k=5, exclude_0=False):
   X, y = np.array(X), np.array(y)
   distances, indices = model.kneighbors(X, n_neighbors=k + 1 if exclude_0 else k)
   preds, dists = [], []
   for d, idxs in zip(distances, indices):
       if exclude_0:
           d, idxs = d[1:], idxs[1:]
       preds.append(np.mean(y[idxs]))
       dists.append(np.mean(d))
   return np.array(preds), np.array(dists)

def preprocess_knn_features(X_tr, X_va, y_tr, knn_features, knn_params):
   scaler = StandardScaler()
   X_tr_knn, X_va_knn = scaler.fit_transform(X_tr[knn_features]), scaler.transform(X_va[knn_features])
   knn = KNeighborsRegressor(**knn_params).fit(X_tr_knn, y_tr)
   k = knn_params["n_neighbors"]
   price_tr, d_tr = retrieve_neighbours(knn, X_tr_knn, y_tr, k=k, exclude_0=True)
   price_va, d_va = retrieve_neighbours(knn, X_va_knn, y_tr, k=k, exclude_0=False)
   X_tr, X_va = X_tr.copy(), X_va.copy()
   X_tr["k_dist"], X_va["k_dist"] = d_tr, d_va
   X_tr["price_knn"], X_va["price_knn"] = price_tr, price_va
   return X_tr, X_va



def preprocess(df):
    df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame
 

    # --- Date Features ---
    df_copy['sale_date'] = pd.to_datetime(df_copy['sale_date'])
    df_copy['sale_year'] = df_copy['sale_date'].dt.year
    df_copy['sale_month'] = df_copy['sale_date'].dt.month

    # --- Simple Feature Engineering ---
    # Age of the property at the time of sale
    df_copy['age_at_sale'] = df_copy['sale_year'] - df_copy['year_built']
    df_copy['age_at_sale'] = df_copy['age_at_sale'].clip(lower=0) 
    df_copy = df_copy.drop('sale_date', axis=1)

    # Age since renovation at the time of sale
    df_copy['reno_age_at_sale'] = np.where( df_copy['year_reno'] > 0, df_copy['sale_year'] - df_copy['year_reno'], df_copy['age_at_sale'] )
    df_copy['reno_age_at_sale'] = df_copy['reno_age_at_sale'].clip(lower=0)
    df_copy['was_renovated'] = (df_copy['year_reno'] > 0).astype(int) # Binary flag: 1 if renovated, 0 otherwise

    for a, b in [("imp_val", "land_val"), ("sqft", "sqft_lot"), ("imp_val", "sqft"), ("area", "sqft")]:
        df_copy[f"{a}_x_{b}"] = df_copy[a] * df_copy[b]
        
    # Ratio-based features
    df_copy["imp_land_ratio"] = df_copy["imp_val"] / (df["land_val"] + 1)
    df_copy["sqft_density"]   = df_copy["sqft"]    / (df["sqft_lot"] + 1)
    df_copy["basement_share"] = df_copy["sqft_fbsmt"] / (df["sqft"] + 1)

    #Log-transform heavy-tailed variables
    for col in ["land_val", "imp_val", "sqft", "sqft_lot"]:
        df_copy[f"log_{col}"] = np.log1p(df_copy[col])
    
    df_copy = df_copy.drop( ["land_val", "imp_val", "sqft", "sqft_lot"], axis=1)

    """
    for feature in ["sale_month"]:
        min_f = df_copy[feature].min() 
        max_f = df_copy[feature].max()
        
        rel_diff = (df_copy[feature] - min_f) / (max_f - min_f)
        df_copy[f'sin_{feature}'] = np.sin(2 * np.pi * rel_diff)
        df_copy[f'cos_{feature}'] = np.cos(2 * np.pi * rel_diff)
    
    
    """
    
    #zoning_flags = df_copy["zoning"].apply(create_zoning_flags).apply(pd.Series)
    #df_copy = pd.concat([df_copy, zoning_flags], axis=1)
    #df_copy = df_copy.drop('zoning', axis=1)
 
    
    return df_copy


train = preprocess(train)
test = preprocess(test)

# Target variable
TARGET = 'sale_price'

# Log transform the target (common for prices)
#train[TARGET] = np.log1p(train[TARGET])



# TWO-STAGE UNCERTAINTY MODEL

class TwoStageUncertaintyModel:
   def __init__(self, model0, model1, n_splits=5, method="squared_error", seed=None, lower_bound=1000, alpha=0.1, gamma0=1.65, gamma1=1.75, features1=None):
       self.model0, self.model1 = model0, model1
       self.n_splits, self.method, self.seed = n_splits, method, seed
       self.gamma0, self.gamma1 = gamma0, gamma1
       self.lower_bound, self.alpha, self.features1 = lower_bound, alpha, features1
       self.fitted_ = False

   def _prepare_features_for_model1(self, X, y_pred):
       X_tmp = X[self.features1].copy() if self.features1 != "same" else X.copy()
       X_tmp["y_pred"] = y_pred
       return X_tmp

   def _get_target(self, y, oof_preds):
       return (y - oof_preds) ** 2 + 1e-6 if self.method == "squared_error" else np.abs(y - oof_preds)

   def fit(self, X, y):
       y = np.asarray(y)
       oof_preds = np.zeros_like(y, dtype=float)
       kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
       for train_idx, val_idx in kf.split(X):
           X_tr, X_val, y_tr = X.iloc[train_idx], X.iloc[val_idx], y[train_idx]
           self.model0.fit(X_tr, y_tr)
           oof_preds[val_idx] = self.model0.predict(X_val)
       target = self._get_target(y, oof_preds)
       X_resid_feat = self._prepare_features_for_model1(X, oof_preds) if self.features1 else oof_preds.reshape(-1, 1)
       self.model1.fit(X_resid_feat, target)
       self.model0.fit(X, y)
       self.fitted_ = True
       return self

   def predict_components(self, X):
       if not self.fitted_: raise ValueError("Call fit() before predict()")
       y_hat = self.model0.predict(X)
       X_resid_feat = self._prepare_features_for_model1(X, y_hat) if self.features1 else y_hat.reshape(-1, 1)
       err_hat = self.model1.predict(X_resid_feat)
       err_hat = np.maximum(err_hat, self.lower_bound)
       return y_hat, err_hat

   def build_interval(self, y_hat, err_hat):
       err_hat_sqrt = np.sqrt(err_hat) if self.method == "squared_error" else err_hat
       lower = y_hat - self.gamma0 * err_hat_sqrt
       upper = y_hat + self.gamma1 * err_hat_sqrt
       return lower, upper

   def predict(self, X):
       y_hat, err_hat = self.predict_components(X)
       lower, upper = self.build_interval(y_hat, err_hat)
       return y_hat, lower, upper



"""

train['sale_warning_clean'] = train['sale_warning'].str.strip()
test['sale_warning_clean'] = test['sale_warning'].str.strip()

# Combine both train and test to extract all unique codes
combined = pd.concat([train['sale_warning_clean'], test['sale_warning_clean']])
all_codes = combined.dropna().apply(lambda x: x.split()).tolist()
unique_codes = sorted(set(chain.from_iterable(all_codes)))

print("Total unique warning codes (train + test):", unique_codes)

# Function to apply encoding
def encode_warning_codes(df, codes):
    for code in codes:
        df[f'sale_warning_{code}'] = df['sale_warning_clean'].apply(
            lambda x: int(str(code) in x.split()) if isinstance(x, str) else 0
        )
    return df

# Apply to both datasets
train = encode_warning_codes(train, unique_codes)
test = encode_warning_codes(test, unique_codes)



train.drop(['sale_warning_clean', 'sale_warning'], axis =1, inplace = True ) 
test.drop( ['sale_warning_clean', 'sale_warning'], axis =1, inplace = True )
"""


cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c != 'sale_price']
num_cols = list(set(test.columns) - set(cat_cols))
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train[cat_cols] = encoder.fit_transform(train[cat_cols]).astype(int)
test[cat_cols] = encoder.transform(test[cat_cols]).astype(int)
print(f"Data ready: Train {train.shape}, Test {test.shape}")



print("Categorical cols are: ", cat_cols)
print("Numerical cols are: ", num_cols)


"""
[I 2025-07-26 16:35:34,612] Trial 4 finished with value: 308095.1292025 and parameters: {'xgb0_n_estimators': 1481, 'xgb0_max_depth': 9, 'xgb0_learning_rate': 0.027189476681481566, 'xgb0_subsample': 0.9021582289462893, 'xgb0_colsample_bytree': 0.7293053869397392, 'xgb0_reg_alpha': 6.956929850273742e-05, 'xgb0_reg_lambda': 0.18658749885082565, 'xgb0_min_child_weight': 10, 'xgb1_n_estimators': 1235, 'xgb1_max_depth': 4, 'xgb1_learning_rate': 0.05473378006078029, 'xgb1_subsample': 0.7288927975921082, 'xgb1_colsample_bytree': 0.8989015371064655, 'xgb1_reg_alpha': 7.436519130026102, 'xgb1_reg_lambda': 0.00013677050554575782, 'xgb1_min_child_weight': 6}. Best is trial 4 with value: 308095.1292025.
[I 2025-07-26 16:53:52,018] Trial 5 finished with value: 306799.7639740039 and parameters: {'xgb0_n_estimators': 1454, 'xgb0_max_depth': 9, 'xgb0_learning_rate': 0.04000669933177982, 'xgb0_subsample': 0.8858106626160609, 'xgb0_colsample_bytree': 0.7424416526462457, 'xgb0_reg_alpha': 0.0005182476494290203, 'xgb0_reg_lambda': 0.15558534352585884, 'xgb0_min_child_weight': 9, 'xgb1_n_estimators': 1208, 'xgb1_max_depth': 4, 'xgb1_learning_rate': 0.058840309501534724, 'xgb1_subsample': 0.7852544799380983, 'xgb1_colsample_bytree': 0.9020074695569282, 'xgb1_reg_alpha': 6.813563548328901, 'xgb1_reg_lambda': 2.8535004907448244e-05, 'xgb1_min_child_weight': 8}. Best is trial 5 with value: 306799.7639740039.
[I 2025-07-26 17:14:46,951] Trial 6 finished with value: 310416.9217906836 and parameters: {'xgb0_n_estimators': 1619, 'xgb0_max_depth': 9, 'xgb0_learning_rate': 0.042216022954189594, 'xgb0_subsample': 0.8997563917428105, 'xgb0_colsample_bytree': 0.7013273254491192, 'xgb0_reg_alpha': 0.0004857316860401368, 'xgb0_reg_lambda': 0.1520734541689653, 'xgb0_min_child_weight': 6, 'xgb1_n_estimators': 1210, 'xgb1_max_depth': 3, 'xgb1_learning_rate': 0.03847692991413517, 'xgb1_subsample': 0.7579699472072916, 'xgb1_colsample_bytree': 0.9012150926895628, 'xgb1_reg_alpha': 8.03198854084714, 'xgb1_reg_lambda': 6.369376533563973e-05, 'xgb1_min_child_weight': 7}. Best is trial 5 with value: 306799.7639740039.
[I 2025-07-26 17:36:05,531] Trial 7 finished with value: 304728.7181674609 and parameters: {'xgb0_n_estimators': 1702, 'xgb0_max_depth': 9, 'xgb0_learning_rate': 0.04091088106237259, 'xgb0_subsample': 0.9318245255622781, 'xgb0_colsample_bytree': 0.6599994593637779, 'xgb0_reg_alpha': 1.3253033965889253e-05, 'xgb0_reg_lambda': 0.1546133450962964, 'xgb0_min_child_weight': 8, 'xgb1_n_estimators': 1014, 'xgb1_max_depth': 5, 'xgb1_learning_rate': 0.058672479063535246, 'xgb1_subsample': 0.7980521296599157, 'xgb1_colsample_bytree': 0.8735853366289013, 'xgb1_reg_alpha': 9.264297531109069, 'xgb1_reg_lambda': 0.00017280174497074594, 'xgb1_min_child_weight': 8}. Best is trial 7 with value: 304728.7181674609.
[I 2025-07-26 17:58:43,182] Trial 8 finished with value: 307089.3526224805 and parameters: {'xgb0_n_estimators': 1759, 'xgb0_max_depth': 9, 'xgb0_learning_rate': 0.02332647907903021, 'xgb0_subsample': 0.9384536168072171, 'xgb0_colsample_bytree': 0.6941927714294489, 'xgb0_reg_alpha': 0.00028548906853235686, 'xgb0_reg_lambda': 0.1357733543371042, 'xgb0_min_child_weight': 8, 'xgb1_n_estimators': 1228, 'xgb1_max_depth': 6, 'xgb1_learning_rate': 0.03240552383452514, 'xgb1_subsample': 0.7519776144483987, 'xgb1_colsample_bytree': 0.9279049277959731, 'xgb1_reg_alpha': 6.595924684742771, 'xgb1_reg_lambda': 0.0001460814282075008, 'xgb1_min_child_weight': 9}. Best is trial 7 with value: 304728.7181674609.
[I 2025-07-26 18:09:47,741] Trial 9 finished with value: 322407.2449414258 and parameters: {'xgb0_n_estimators': 1521, 'xgb0_max_depth': 6, 'xgb0_learning_rate': 0.022766914354926535, 'xgb0_subsample': 0.8734089598785163, 'xgb0_colsample_bytree': 0.7069081105891335, 'xgb0_reg_alpha': 1.6146131094076974e-05, 'xgb0_reg_lambda': 0.1298168335206227, 'xgb0_min_child_weight': 10, 'xgb1_n_estimators': 1100, 'xgb1_max_depth': 4, 'xgb1_learning_rate': 0.03901961802099979, 'xgb1_subsample': 0.7229481925960197, 'xgb1_colsample_bytree': 0.8898963074184659, 'xgb1_reg_alpha': 7.183431039837434, 'xgb1_reg_lambda': 1.1970754680416245e-05, 'xgb1_min_child_weight': 6}. Best is trial 7 with value: 304728.7181674609.
[I 2025-07-26 18:36:12,273] Trial 10 finished with value: 307357.62202234374 and parameters: {'xgb0_n_estimators': 1566, 'xgb0_max_depth': 10, 'xgb0_learning_rate': 0.049837809975523055, 'xgb0_subsample': 0.9208360990144833, 'xgb0_colsample_bytree': 0.650560375701227, 'xgb0_reg_alpha': 3.421235333309955e-05, 'xgb0_reg_lambda': 0.11601874973460247, 'xgb0_min_child_weight': 7, 'xgb1_n_estimators': 1294, 'xgb1_max_depth': 5, 'xgb1_learning_rate': 0.04756422118008695, 'xgb1_subsample': 0.6564028931976067, 'xgb1_colsample_bytree': 0.8504106859901824, 'xgb1_reg_alpha': 5.499344745874657, 'xgb1_reg_lambda': 0.0004979247382686304, 'xgb1_min_child_weight': 8}. Best is trial 7 with value: 304728.7181674609.



features = cat_cols + num_cols + ["price_knn", "k_dist"]
knn_features = ["latitude", "longitude", "sale_year"]
y = train["sale_price"]
knn_params = {'n_neighbors': 10}



def objective(trial):
    # === model0: point predictor (xgb_params) ===
    xgb0_params = {
        'n_estimators': trial.suggest_int('xgb0_n_estimators', 1400, 1800),  # around 1597
        'max_depth': trial.suggest_int('xgb0_max_depth', 6, 10),             # around 8
        'learning_rate': trial.suggest_float('xgb0_learning_rate', 0.02, 0.05),  # around 0.032
        'subsample': trial.suggest_float('xgb0_subsample', 0.85, 0.95),          # around 0.90
        'colsample_bytree': trial.suggest_float('xgb0_colsample_bytree', 0.65, 0.75),  # around 0.68
        'reg_alpha': trial.suggest_float('xgb0_reg_alpha', 1e-5, 1e-3, log=True),     # around 0.00035
        'reg_lambda': trial.suggest_float('xgb0_reg_lambda', 0.1, 0.2),               # around 0.14
        'min_child_weight': trial.suggest_int('xgb0_min_child_weight', 6, 10),        # around 8
        'random_state': SEED,
        'tree_method': 'gpu_hist',
        'gpu_id': 1,
        'eval_metric': 'rmse'
    }

    # === model1: uncertainty predictor (xgb_params1) ===
    xgb1_params = {
        'n_estimators': trial.suggest_int('xgb1_n_estimators', 1000, 1300),  # around 1190
        'max_depth': trial.suggest_int('xgb1_max_depth', 3, 6),              # around 4
        'learning_rate': trial.suggest_float('xgb1_learning_rate', 0.03, 0.06),  # around 0.041
        'subsample': trial.suggest_float('xgb1_subsample', 0.65, 0.80),         # around 0.72
        'colsample_bytree': trial.suggest_float('xgb1_colsample_bytree', 0.85, 0.95),  # around 0.91
        'reg_alpha': trial.suggest_float('xgb1_reg_alpha', 5, 10),                 # around 7.36
        'reg_lambda': trial.suggest_float('xgb1_reg_lambda', 1e-5, 1e-3, log=True),# around 0.00057
        'min_child_weight': trial.suggest_int('xgb1_min_child_weight', 6, 10),    # around 8
        'random_state': SEED,
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': "reg:gamma"
    }

    # === Define two-stage uncertainty model ===
    model0 = XGBRegressor(**xgb0_params)
    model1 = XGBRegressor(**xgb1_params)

    uncert_model = TwoStageUncertaintyModel(
        model0=model0,
        model1=model1,
        seed=SEED,
        method="squared_error",
        n_splits=10,
        features1="same",
        gamma0=1.65,
        gamma1=1.75
    )

    # === Cross-validation ===
    scores = []
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (train_idx, val_idx) in enumerate(cv.split(train, y), 1):
        X_tr, X_vl = train.iloc[train_idx], train.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        X_tr, X_vl = preprocess_knn_features(X_tr, X_vl, y_tr, knn_features, knn_params)

        uncert_model.fit(X_tr[features], y_tr)
        y_hat_vl, err_hat_vl = uncert_model.predict_components(X_vl[features])
        pi_lower, pi_upper = uncert_model.build_interval(y_hat_vl, err_hat_vl)

        y_min, y_max = y_tr.min(), y_tr.max()
        pi_lower = np.clip(pi_lower, y_min, y_max)
        pi_upper = np.clip(pi_upper, y_min, y_max)

        score = winkler_score(y_vl.values, pi_lower, pi_upper, alpha=ALPHA)
        print(f"Wrinkler Score for fold {i} is {score.mean()}")
        scores.append(score)

    return np.mean(scores)


study = optuna.create_study(direction="minimize", study_name="two_stage_uncertainty_tuned")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best trial:")
print(study.best_trial)

"""



# MODEL CONFIGURATION

knn_params = {'n_neighbors': 10}
xgb_params = {'n_estimators': 1702, 'max_depth': 9, 'learning_rate': 0.04091088106237259, 'subsample': 0.9318245255622781, 
           'colsample_bytree': 0.6599994593637779, 'reg_alpha': 1.3253033965889253e-05, 'reg_lambda': 0.1546133450962964, 
           'min_child_weight': 8, 'random_state': 42, 'tree_method': 'gpu_hist','gpu_id': 1, 'eval_metric': 'rmse',
         }
xgb_params1 = {'n_estimators': 1014, 'max_depth': 5, 'learning_rate': 0.058672479063535246, 'subsample': 0.7980521296599157, 
 'colsample_bytree': 0.8735853366289013, 'reg_alpha': 9.264297531109069, 'reg_lambda':  0.00017280174497074594, 
 'min_child_weight': 8, 'random_state': SEED,'tree_method': 'hist', 'device': device, 'objective': "reg:gamma"
        }
model0 = XGBRegressor(**xgb_params)
model1 = XGBRegressor(**xgb_params1)
uncert_model = TwoStageUncertaintyModel(model0=model0, model1=model1, seed=SEED, method="squared_error", n_splits=10, features1="same", gamma0=1.65, gamma1=1.75)
features = cat_cols + num_cols + ["price_knn", "k_dist"]
knn_features = ["latitude", "longitude", "sale_year"]
y = train["sale_price"]


# CROSS-VALIDATION & OOF PREDICTION COLLECTION

print(f"\n Running Cross-Validation and Collecting OOF Predictions...")
oof_y_true, oof_y_hat, oof_err_hat, oof_indices = [], [], [], []
scores, coverages = [], []
cv = KFold(shuffle=True, random_state=SEED, n_splits=FOLDS)
for i, (train_idx, val_idx) in enumerate(cv.split(train, y), 1):
    print(f"\n=== Fold: {i} ===")
    X_tr, X_vl, y_tr, y_vl = train.iloc[train_idx], train.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
    X_tr, X_vl = preprocess_knn_features(X_tr, X_vl, y_tr, knn_features, knn_params)
    model = uncert_model.fit(X_tr[features], y_tr)
    y_hat_vl, err_hat_vl = model.predict_components(X_vl[features])
    pi_lower, pi_upper = model.build_interval(y_hat_vl, err_hat_vl)
    y_min, y_max = y_tr.min(), y_tr.max()
    pi_lower, pi_upper = np.clip(pi_lower, y_min, y_max), np.clip(pi_upper, y_min, y_max)
    score, coverage = winkler_score(y_vl.values, pi_lower, pi_upper, alpha=ALPHA, return_coverage=True)
    scores.append(score)
    coverages.append(coverage)
    oof_y_true.extend(y_vl.values), oof_y_hat.extend(y_hat_vl), oof_err_hat.extend(err_hat_vl), oof_indices.extend(val_idx)
    print(f"Winkler (fixed gamma): {score:,.0f} | Coverage: {coverage:.4f}")

oof_df = pd.DataFrame({'y_true': oof_y_true, 'y_hat': oof_y_hat, 'err_hat': oof_err_hat}, index=oof_indices).sort_index()
print(f"\n Initial CV Winkler (fixed gamma): {np.mean(scores):,.0f} ± {np.std(scores):,.0f}")


print("\n Optimizing Gamma Scaling Factors...")
def winkler_objective(gammas, y_true, y_hat, err_hat):
    gamma0, gamma1 = gammas
    err_hat_sqrt = np.sqrt(err_hat)
    lower = y_hat - gamma0 * err_hat_sqrt
    upper = y_hat + gamma1 * err_hat_sqrt
    return winkler_score(y_true, lower, upper)

initial_gammas = [uncert_model.gamma0, uncert_model.gamma1]
bounds = [(0.5, 4.0), (0.5, 4.0)]
opt_result = minimize(
    winkler_objective,
    initial_gammas,
    args=(oof_df['y_true'], oof_df['y_hat'], oof_df['err_hat']),
    method='Nelder-Mead',
    bounds=bounds
)

optimized_gamma0, optimized_gamma1 = opt_result.x
uncert_model.gamma0, uncert_model.gamma1 = optimized_gamma0, optimized_gamma1
optimized_score = opt_result.fun

print(f" Gamma optimization complete.")
print(f"   Initial Gammas: {initial_gammas[0]:.3f}, {initial_gammas[1]:.3f} -> Score: {np.mean(scores):,.0f}")
print(f"   {Fore.GREEN}Optimal Gammas: {optimized_gamma0:.3f}, {optimized_gamma1:.3f} -> Score: {optimized_score:,.0f}{Style.RESET_ALL}")
print(f"   Improvement: {np.mean(scores) - optimized_score:+,.0f} points")


# FINAL SUBMISSION

print(f"\n{Fore.GREEN} PROCEEDING WITH OPTIMIZED SUBMISSION{Style.RESET_ALL}")
print(" Training final model on full dataset with optimized gammas...")
X_train, X_test = preprocess_knn_features(train, test, y, knn_features, knn_params)
final_model = uncert_model.fit(X_train[features], y)

print(" Generating victory predictions...")
_, pi_lower, pi_upper = final_model.predict(X_test[features])
y_min, y_max = y.min(), y.max()
pi_lower, pi_upper = np.clip(pi_lower, y_min, y_max), np.clip(pi_upper, y_min, y_max)
submission_df = pd.DataFrame({'id': test.index, 'pi_lower': pi_lower, 'pi_upper': pi_upper})
submission_df.to_csv("submission_optimized_gamma.csv", index=False)

print(f"\n OPTIMIZED SUBMISSION CREATED!")

