# for gpu only
!pip uninstall xgboost -y


!pip install xgboost>=2.0.0


!pip uninstall lightgbm -y


!pip install lightgbm>=4.0.0


import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                            classification_report, roc_curve,
                            roc_auc_score, precision_score,
                            recall_score, f1_score)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

import joblib
import optuna

import xgboost as xgb
import lightgbm as lgb
import catboost as cb  # Added CatBoost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings("ignore")

rand_seed = 7


base_dir = '/kaggle/input/playground-series-s5e12'

df_train = pd.read_csv(os.path.join(base_dir, "train.csv"))
df_test = pd.read_csv(os.path.join(base_dir, "test.csv"))
df_submission = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))


print(f"Training data size: {df_train.shape}")
print(f"Testg data size: {df_test.shape}")


df_train.head()


df_train.info()


df_train.isna().sum()


df_test.isna().sum()


df_train.head()


df_train.describe()


string_data = df_train.select_dtypes(include=['object'])
str_cols = string_data.columns.tolist()
num_cols = df_train.select_dtypes(include=['number']).columns.tolist()
print("Numerical Columns:", len(num_cols))
for i in num_cols:
    print(i, end=', ')
print("\nString Columns: ", len(str_cols))
for i in str_cols:
    print(i, end = ", ")


df_train[str_cols].head()


df_train[num_cols].head()


class OneHotFeatureEncoder:
    def __init__(self):
        self.dir_path = None
        self.ohe = None
        self.columns = []
    
    def fit(self, data, columns):
        self.columns = columns
        self.ohe = OneHotEncoder(
            handle_unknown='ignore', # CRITICAL: Ignore categories not seen during fit
            sparse_output=False
        )
        self.ohe.fit(data[self.columns])

    def transform(self, data):
        res = self.ohe.transform(data[self.columns])
        return pd.DataFrame(
            data = res,
            columns = self.ohe.get_feature_names_out() 
        )

    def save(self, data_path):
        joblib.dump(self.columns, os.path.join(dir_path, 'ohe_columns.joblib'))
        joblib.dump(self.ohe, os.path.join(dir_path, 'ohe_encoder.joblib'))

    def load(self, data_path):
        self.ohe = joblib.load(os.path.join(dir_path, 'ohe_encoder.joblib'))
        self.columns = joblib.load(os.path.join(dir_path, 'ohe_columns.joblib'))




class OrdinalFeatureEncoder:
    def __init__(self):
        self.dir_path = None
        self.ode = None
        self.columns = []
    def fit(self, data, columns):
        self.columns = columns
        self.ode = OrdinalEncoder( )
        self.ode.fit(data[self.columns])

    def transform(self, data):
        res = self.ode.transform(data[self.columns])
        return pd.DataFrame(
            data = res,
            columns = self.ode.get_feature_names_out() 
        )

    def save(self, dir_path):
        joblib.dump(self.columns, os.path.join(dir_path, 'ode_columns.joblib'))
        joblib.dump(self.ode, os.path.join(dir_path, 'ode_encoder.joblib'))

    def load(self, dir_path):
        self.ode = joblib.load(os.path.join(dir_path, 'ode_encoder.joblib'))
        self.columns = joblib.load(os.path.join(dir_path, 'ode_columns.joblib'))


class CustomScaler:
    def __init__(self):
        self.scaler = None

    def fit(self, data):
        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def save(self, dir_path):
        joblib.dump(self.scaler, os.path.join(dir_path, 'scaler.joblib'))

    def load(self, dir_path):
        self.scaler = joblib.load(os.path.join(dir_path, 'scaler.joblib'))


dir_path = '/kaggle/working'


def create_encoders(data, dir_path, ordinal_columns=[], onehot_columns=[]):
    eod = OrdinalFeatureEncoder()
    eod.fit(df_train, ordinal_columns)
    eod.save(dir_path)
    
    enc = OneHotFeatureEncoder()
    enc.fit(df_train, onehot_columns)
    enc.save(dir_path)

def create_features_scaler(data, dir_path):
    custom_scaler = CustomScaler()
    custom_scaler.fit(X)
    custom_scaler.save(dir_path)

def transform_features(data:pd.DataFrame, ordinal_encoder, onehot_encoder):
    res_ordinal_encoder = ordinal_encoder.transform(data)
    res_onehot_encoder = onehot_encoder.transform(data)

    all_columns = ordinal_encoder.columns + onehot_encoder.columns
    print(all_columns)
    
    return pd.concat([data, res_ordinal_encoder, res_onehot_encoder], axis=1).drop(columns=all_columns)



create_encoders(df_train, 
                dir_path, 
                ordinal_columns=['education_level','income_level'], 
                onehot_columns=['gender', 'ethnicity', 'smoking_status', 'employment_status'])


ordinal_encoder = OrdinalFeatureEncoder()
ordinal_encoder.load(dir_path)

onehot_encoder = OneHotFeatureEncoder()
onehot_encoder.load(dir_path)


data = transform_features(df_train, ordinal_encoder, onehot_encoder)


data.head()


X = data.drop(['id','diagnosed_diabetes'],axis = 1) 
y = data['diagnosed_diabetes']


create_features_scaler(X, dir_path)


custom_scaler = CustomScaler()
custom_scaler.load(dir_path)

X_scaled = custom_scaler.transform(X)


#1/0


# X = np.random.rand(10000, 50)
# y = np.random.randint(0, 2, 10000)

# model = xgb.XGBClassifier(
#     tree_method="hist",
#     device="cuda",
#     n_estimators=100
# )

# model.fit(X, y)
# print("âœ… GPU training successful")


print(xgb.__version__)
print(xgb.get_config())


# -------------------------------------------------
# Unified objective function
# -------------------------------------------------
def objective(trial, X, y, model_type="xgb", use_gpu=False):

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        # =======================
        # XGBOOST
        # =======================
        if model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "device": "cuda" if use_gpu else "cpu",
                "random_state": 42,
                "verbosity": 0
            }

            model = XGBClassifier(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False
            )

        # =======================
        # LIGHTGBM
        # =======================
        elif model_type == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "objective": "binary",
                "metric": "auc",
                "scale_pos_weight": scale_pos_weight,
                "device": "gpu" if use_gpu else "cpu",
                "verbosity": -1,
                "random_state": 42
            }

            model = LGBMClassifier(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc"
            )

        # =======================
        # CATBOOST
        # =======================
        elif model_type == "cat":
            params = {
                "iterations": trial.suggest_int("iterations", 300, 2000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "scale_pos_weight": scale_pos_weight,
                "task_type": "GPU" if use_gpu else "CPU",
                "devices": "0",
                "bootstrap_type": "Bayesian",
                "random_seed": 42,
                "verbose": False
            }

            model = CatBoostClassifier(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=50,
                use_best_model=True
            )

        else:
            raise ValueError("Invalid model_type")

        preds = model.predict_proba(X_valid)[:, 1]
        scores.append(roc_auc_score(y_valid, preds))

    return np.mean(scores)


# -------------------------------------------------
# Tuner API
# -------------------------------------------------
def tune_model(X, y, model_type="xgb", use_gpu=False, n_trials=50):

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_type}_optuna",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        lambda trial: objective(trial, X, y, model_type, use_gpu),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\nðŸ”¥ Best ROC-AUC:", study.best_value)
    print("ðŸ”¥ Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study



# XGBoost GPU
study_xgb = tune_model(X_scaled, y, model_type="xgb", use_gpu=True, n_trials=1)


# LightGBM CPU
study_lgbm = tune_model(X_scaled, y, model_type="lgbm", use_gpu=False, n_trials=1)


# CatBoost GPU
study_cat = tune_model(X_scaled, y, model_type="cat", use_gpu=True, n_trials=1)


final_xgb_model = XGBClassifier(
    **study_xgb.best_params,
    tree_method="hist",
    device="cuda",
    eval_metric="auc"
)

final_xgb_model.fit(X_scaled, y)


final_catboost_model = CatBoostClassifier(
    **study_cat.best_params, 
    verbose=False
)
final_catboost_model.fit(X_scaled, y)


final_lgbm_model = LGBMClassifier(
    **study_lgbm.best_params
)

final_lgbm_model.fit(X_scaled,y)


def predict(model, data):
    ordinal_encoder = OrdinalFeatureEncoder()
    ordinal_encoder.load(dir_path)
    
    onehot_encoder = OneHotFeatureEncoder()
    onehot_encoder.load(dir_path)

    data = transform_features(data, ordinal_encoder, onehot_encoder)

    custom_scaler = CustomScaler()
    custom_scaler.load(dir_path)
    
    X_scaled = custom_scaler.transform(data)

    #res = model.predict(X_scaled)
    res = model.predict_proba(X_scaled)[:,1]

    return res


res_xgb = predict(final_xgb_model, df_train.drop(columns=['id', 'diagnosed_diabetes']))
res_catboost = predict(final_catboost_model, df_train.drop(columns=['id', 'diagnosed_diabetes']))
res_lgbm = predict(final_lgbm_model, df_train.drop(columns=['id', 'diagnosed_diabetes']))

w_lgb = 0.30
w_xgb = 0.30
w_cat = 0.40

final_res = (w_lgb * res_lgbm) + (w_xgb * res_xgb) + (w_cat * res_catboost)

ensemble_score = roc_auc_score(y, final_res)
print(f"Weighted Ensemble AUC: {ensemble_score:.5f}")



res_xgb = predict(final_xgb_model, df_test.drop(columns=['id']))
res_catboost = predict(final_catboost_model, df_test.drop(columns=['id']))
res_lgbm = predict(final_lgbm_model, df_test.drop(columns=['id']))

final_res = (w_lgb * res_lgbm) + (w_xgb * res_xgb) + (w_cat * res_catboost)

df_sub = pd.DataFrame(data={
    'id': df_test['id'],
    'diagnosed_diabetes': final_res
})
df_sub.head()
df_sub.to_csv(os.path.join("submission_ens.csv"), index=False)







