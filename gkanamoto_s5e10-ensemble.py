# Import libraries
import lightgbm as lgb
import xgboost as xgb
import catboost as catb
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


# CONFIG
SEED = 42
VAL_SIZE = 0.2
FOLDS = 5


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')


train_df.head()


test_df.head()


print("Train:", train_df.shape)
print(" Test:", test_df.shape)


# check the missing values and data types
print("-"*20)
print(train_df.isnull().sum())
print("-"*20)
print(train_df.dtypes)


# check the missing values and data types
print("-"*20)
print(test_df.isnull().sum())
print("-"*20)
print(test_df.dtypes)


TARGET = 'accident_risk'
BASE = [col for col in train_df.columns if col not in ['id', TARGET]]
CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

print(f'{len(BASE)} Base Features:{BASE}')


ORIG = []

for col in BASE:
    tmp = orig.groupby(col)[TARGET].mean()
    new_col_name = f"orig_{col}"
    tmp.name = new_col_name
    train_df = train_df.merge(tmp, on=col, how='left')
    test_df = test_df.merge(tmp, on=col, how='left')
    ORIG.append(new_col_name)

print(len(ORIG), 'Orig Features Created!!')


META = []

for df in [train_df, test_df, orig]:
    base_risk = (
        0.3 * df["curvature"] + 
        0.2 * (df["lighting"] == "night").astype(int) + 
        0.1 * (df["weather"] != "clear").astype(int) + 
        0.2 * (df["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )
    df['Meta'] = base_risk

META.append('Meta')


train_df['orig_curvature'] = train_df['orig_curvature'].fillna(orig[TARGET].mean())
test_df['orig_curvature'] = test_df['orig_curvature'].fillna(orig[TARGET].mean())


FEATURES = BASE + ORIG + META
print(len(FEATURES), 'Features.')


# Label encoding
cols = ['road_type', 'lighting', 'weather', 'road_signs_present', 
        'public_road', 'time_of_day', 'holiday', 'school_season']

le = LabelEncoder()

for i in cols:
    train_df[i] = le.fit_transform(train_df[i])
    test_df[i] = le.transform(test_df[i])


X = train_df[FEATURES]
y = train_df[TARGET]

# Dropping id from test dataset
test_id = test_df.id
test_df = test_df[FEATURES]


# Custom settings - LGBM
def customLGBM(**params):
    default_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'boosting': 'gbdt',
        'max_depth': 6,
        'random_state': 0,
        'n_jobs': -1,
        'verbose': -1
    }
    default_params.update(params)
    return lgb.LGBMRegressor(**default_params)

# Custom settings - XGB
def customXGB(**params):
    default_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'random_state': 0,
        'eval_metric': 'rmse',
        'tree_method': 'hist'
    }
    default_params.update(params)
    return xgb.XGBRegressor(**default_params)

# Custom settings - CatBoost
def customCat(**params):
    default_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'random_state': 0,
        'verbose': 0,
        'task_type': 'CPU'
    }
    default_params.update(params)
    return catb.CatBoostRegressor(**default_params)


# define the parameters sets
param_sets = [
    # ===== LGBM =====
    {"type": "lgbm", "iter": 1000, "lr": 0.05, "boosting": 'gbdt', "depth": -1, "seeds": SEED},
    {"type": "lgbm", "iter": 2000, "lr": 0.01, "boosting": 'gbdt', "depth": 7, "seeds": SEED+1},
    {"type": "lgbm", "iter": 2000, "lr": 0.03, "boosting": 'gbdt', "depth": 12, "seeds": SEED+2},
    {"type": "lgbm", "iter": 1500, "lr": 0.1, "boosting": 'gbdt', "depth": 6, "seeds": SEED+3},
    {"type": "lgbm", "iter": 2500, "lr": 0.02, "boosting": 'gbdt', "depth": 16, "seeds": SEED+4},
    {"type": "lgbm", "iter": 3000, "lr": 0.01, "boosting": 'gbdt', "depth": 24, "seeds": SEED+5},
    {"type": "lgbm", "iter": 2000, "lr": 0.03, "boosting": 'goss', "depth": 8, "seeds": SEED+6},
    {"type": "lgbm", "iter": 3000, "lr": 0.02, "boosting": 'goss', "depth": 20, "seeds": SEED+7},
    {"type": "lgbm", "iter": 3000, "lr": 0.03, "boosting": 'goss', "depth": 32, "seeds": SEED+8},
    {"type": "lgbm", "iter": 2500, "lr": 0.05, "boosting": 'goss', "depth": 64, "seeds": SEED+9},
    {"type": "lgbm", "iter": 2000, "lr": 0.05, "boosting": 'goss', "depth": -1, "seeds": SEED+10},
    {"type": "lgbm", "iter": 2500, "lr": 0.02, "boosting": 'dart', "depth": 12, "seeds": SEED+11},
    {"type": "lgbm", "iter": 3000, "lr": 0.01, "boosting": 'dart', "depth": 18, "seeds": SEED+12},
    {"type": "lgbm", "iter": 1500, "lr": 0.07, "boosting": 'dart', "depth": 10, "seeds": SEED+13},

    # ===== XGB =====
    {"type": "xgb", "iter": 1500, "lr": 0.05, "depth": 8, "seeds": SEED, "use_gpu": True},
    {"type": "xgb", "iter": 2000, "lr": 0.03, "depth": 10, "seeds": SEED+1, "use_gpu": True},
    {"type": "xgb", "iter": 2500, "lr": 0.03, "depth": 12, "colsample_bytree": 0.8, "seeds": SEED+2, "use_gpu": True},
    {"type": "xgb", "iter": 3000, "lr": 0.02, "depth": 6, "subsample": 0.7, "seeds": SEED+3, "use_gpu": True},
    {"type": "xgb", "iter": 3000, "lr": 0.02, "depth": 12, "colsample_bytree": 0.9, "subsample": 0.9, "seeds": SEED+4, "use_gpu": True},
    {"type": "xgb", "iter": 4000, "lr": 0.01, "depth": 5, "colsample_bytree": 0.7, "seeds": SEED+5, "use_gpu": True},
    {"type": "xgb", "iter": 2000, "lr": 0.05, "depth": 7, "subsample": 0.8, "seeds": SEED+6, "use_gpu": True},
    {"type": "xgb", "iter": 2500, "lr": 0.03, "depth": 9, "colsample_bytree": 0.6, "seeds": SEED+7, "use_gpu": True},
    {"type": "xgb", "iter": 3000, "lr": 0.02, "depth": 11, "colsample_bytree": 1.0, "subsample": 0.8, "seeds": SEED+8, "use_gpu": True},
    {"type": "xgb", "iter": 3500, "lr": 0.015, "depth": 10, "seeds": SEED+9, "use_gpu": True},

    # ===== CatBoost =====
    {"type": "cat", "iterations": 1200, "learning_rate": 0.05, "depth": 6, "random_state": SEED, "task_type": "GPU"},
    {"type": "cat", "iterations": 1500, "learning_rate": 0.03, "depth": 8, "random_state": SEED+1, "task_type": "GPU"},
    {"type": "cat", "iterations": 2000, "learning_rate": 0.02, "depth": 4, "random_state": SEED+2, "task_type": "GPU"},
    {"type": "cat", "iterations": 2500, "learning_rate": 0.05, "depth": 10, "random_state": SEED+3, "task_type": "GPU"},
    {"type": "cat", "iterations": 3000, "learning_rate": 0.01, "depth": 7, "random_state": SEED+4, "task_type": "GPU"},
    {"type": "cat", "iterations": 2000, "learning_rate": 0.04, "depth": 5, "random_state": SEED+5, "task_type": "GPU"},
    {"type": "cat", "iterations": 2500, "learning_rate": 0.02, "depth": 9, "random_state": SEED+6, "task_type": "GPU"},
    {"type": "cat", "iterations": 1500, "learning_rate": 0.07, "depth": 6, "random_state": SEED+7, "task_type": "GPU"},
]


# =====================
# CV & OOF Loop
# =====================
def run_cv_models(X, y, param_sets, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    oof_preds = []
    model_names = []

    for i, ps in enumerate(param_sets):
        model_type = ps["type"]
        params = {k: v for k, v in ps.items() if k != "type"}

        if model_type == "lgbm":
            model = customLGBM(**params)
        elif model_type == "xgb":
            model = customXGB(**params)
        elif model_type == "cat":
            model = customCat(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        oof = np.zeros(len(y))
        
        # CV Loop
        for tr_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict(X_val)
        
        rmse = mean_squared_error(y, oof, squared=False)
        name = f"{ps['type']}_{i}"
        print(f"{name} OOF RMSE: {rmse:.5f}")
        
        oof_preds.append(oof)
        model_names.append(name)
    
    return np.array(oof_preds).T, model_names


oof_preds, model_names = run_cv_models(X, y, param_sets, n_splits=FOLDS)


# ============================
# Hill Climb Ensemble
# ============================
def hillclimb(oof_preds, y, model_names):
    selected = []
    best_score = np.inf
    remaining = list(range(oof_preds.shape[1]))

    while remaining:
        improved = False
        best_candidate = None
        best_candidate_score = best_score

        for i in remaining:
            candidate_idx = selected + [i]
            blend = np.mean(oof_preds[:, candidate_idx], axis=1)
            score = mean_squared_error(y, blend, squared=False)

            if score <= best_candidate_score:
                best_candidate_score = score
                best_candidate = i
                improved = True

        if improved:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
            print(f"Add {model_names[best_candidate]} -> RMSE {best_score:.5f}")
        else:
            break
    
    return selected


selected_idx = hillclimb(oof_preds, y, model_names)


def fit_full_and_predict(X, y, X_test, param_sets, selected_idx):
    test_preds = []
    
    for i in selected_idx:
        ps = param_sets[i]
        model_type = ps["type"]
        params = {k: v for k, v in ps.items() if k != "type"}
        
        if model_type == "lgbm":
            model = customLGBM(**params)
        elif model_type == "xgb":
            model = customXGB(**params)
        elif model_type == "cat":
            model = customCat(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y)
        test_preds.append(model.predict(X_test))
    
    final_pred = np.mean(test_preds, axis=0)
    return final_pred


final_pred = fit_full_and_predict(X, y, test_df, param_sets, selected_idx)


submission = pd.DataFrame({"id": test_id, "accident_risk": final_pred})
submission.to_csv("submission.csv", index=False)
print("Complete!!!!")


submission

