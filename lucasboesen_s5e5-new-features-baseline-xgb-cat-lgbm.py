from typing import NoReturn, Dict
from joblib import Parallel, delayed

import numpy as np
import polars as pl
import pandas as pd
import xgboost as xgb
import catboost as cat
import lightgbm as lgb
from torch import cuda

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_log_error  # Remember to take the sqrt of this
import optuna

import seaborn as sns
import matplotlib.pyplot as plt


CFG = {
    'use_gpu': False if not cuda.is_available() else True
}


class DataHandler:
    def __init__(self, base_path: str = "/kaggle/input/playground-series-s5e5/", create_all_combinations: bool = False):
        self.base_path = base_path
        self.create_all_combinations = create_all_combinations

        # Setting the features to use
        self.unused_feats = ['Id', "Calories"]
        self.target = 'Calories'

        # Reads all data when the class is initialized
        self._read_data()

        # Fixes data
        self.train = self._create_features(self.train)
        self.train = self._encode_categories(self.train)
        used_feats = self.train.columns
        used_feats = list(set(used_feats) - set(self.unused_feats))
        self.X_train = self.train[used_feats].to_pandas()
        self.y_train = self.train[self.target].to_pandas()
        self.test = self._create_features(self.test)
        self.test = self._encode_categories(self.test)
        self.X_test = self.test[used_feats].to_pandas()
    
    def _read_data(self) -> NoReturn:
        self.test = pl.read_csv(self.base_path + "/test.csv")
        self.train = pl.read_csv(self.base_path + "/train.csv")
        self.sample_submission = pl.read_csv(self.base_path + "/sample_submission.csv")

    def _create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Creating Body Mass Index feature
        df = df.with_columns(
            (pl.col("Weight") / (pl.col("Height")/100)**2)
            .alias("BMI")
        )

        # Thanks to: https://www.kaggle.com/competitions/playground-series-s5e5/discussion/575843
        # Calculates the avg. heat rate times the duration this was the heart rate
        df = df.with_columns([
            (pl.col("Heart_Rate") * pl.col("Duration")).alias("HR_Duration"),
            (pl.col("Body_Temp") * pl.col("Duration")).alias("Temp_Duration")
        ])

        # Creates all combinations in mult and mult with log
        # if self.create_all_combinations:
        #     cols = df.columns
        #     cols = set(cols) - set(["Sex", "Calories"])
        #     for col1 in cols:
        #         for col2 in cols:
        #             df = df.with_columns(
        #                 [
        #                     (pl.col(col1) * pl.col(col2)).alias(f"{col1}_mul_{col2}"),
        #                     (np.log1p(pl.col(col1)) * np.log1p(pl.col(col2))).alias(f"{col1}_log_mul_{col2}")
        #                 ]
        #             )

        # Calculating BMR - basal metabolic rate (from https://www.medicalnewstoday.com/articles/319731#what-is-a-calorie)
        df = df.with_columns(
            pl.when(
                pl.col("Sex") == "male"
            ).then(
                pl.col("Weight") * 9.65 + (pl.col("Height") / 100) * 573 - pl.col("Age") * 5.08 + 260
            ).otherwise(
                pl.col("Weight") * 7.38 + (pl.col("Height") / 100) * 607 - pl.col("Age") * 2.31 + 43
            )
            .alias("BMR")
        )

        # Calories pr. min
        df = df.with_columns(
            pl.when(
                pl.col("Sex") == "male"
            ).then(
                pl.col("Age") * 0.2017 + pl.col("Weight") * 0.09036 + pl.col("Heart_Rate") * 0.6309 - 55.0969
            ).otherwise(
                pl.col("Age") * 0.074 + pl.col("Weight") * 0.05741 + pl.col("Heart_Rate") * 0.4472 - 20.4022
            )
            .alias("expected_cal_pr_min")
        )

        # Calories burned
        df = df.with_columns(
            (pl.col("expected_cal_pr_min") * pl.col("Duration")).alias("expected_cal_burned")
        )
        
        return df

    def _encode_categories(self, df: pl.DataFrame) -> pl.DataFrame:
        # We only have Sex as our categorical
        df = df.to_dummies(columns="Sex")

        return df
        
                


DR = DataHandler()


cols = ['Calories', 'Sex_female', 'Sex_male', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'HR_Duration', 'Temp_Duration', 'BMR', 'expected_cal_pr_min', 'expected_cal_burned']


train_corr = DR.train.to_pandas()[cols].corr()
plt.figure(figsize=(16, 12)) 
sns.heatmap(train_corr, annot=True, cmap="coolwarm")
plt.tight_layout()
plt.show()


class XGBoostModel:
    def __init__(self, params: Dict = None, log_target: bool = True):
        self.params = params
        self.log_target = log_target

        # Initializing model
        self.init_model()
        
    def init_model(self):
        if self.params:
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBRegressor()

    def fit(self, X, y):
        if self.log_target:
            y = np.log1p(y)
            
        self.model.fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)

        if self.log_target:
            predictions = np.expm1(predictions)
        
        predictions = self.fix_predictions(predictions)
        
        return predictions

    def fix_predictions(self, predictions):
        predictions = np.maximum(predictions, 1)

        return predictions

class CatBoostModel:
    def __init__(self, params: Dict = None, log_target: bool = True):
        self.params = params
        self.log_target = log_target

        # Initializing model
        self.init_model()
        
    def init_model(self):
        if self.params:
            self.model = cat.CatBoostRegressor(**self.params)
        else:
            self.model = cat.CatBoostRegressor()

    def fit(self, X, y):
        if self.log_target:
            y = np.log1p(y)
            
        self.model.fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)

        if self.log_target:
            predictions = np.expm1(predictions)
        
        predictions = self.fix_predictions(predictions)
        
        return predictions

    def fix_predictions(self, predictions):
        predictions = np.maximum(predictions, 1)

        return predictions

class LightGBMModel:
    def __init__(self, params: Dict = None, log_target: bool = True):
        self.params = params
        self.log_target = log_target

        # Initializing model
        self.init_model()
        
    def init_model(self):
        if self.params:
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.model = lgb.LGBMRegressor()

    def fit(self, X, y):
        if self.log_target:
            y = np.log1p(y)
            
        self.model.fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)

        if self.log_target:
            predictions = np.expm1(predictions)
        
        predictions = self.fix_predictions(predictions)
        
        return predictions

    def fix_predictions(self, predictions):
        predictions = np.maximum(predictions, 1)
        return predictions


def run_cv(n_splits: int = 5, params: Dict = None, n_jobs: int = -1, model_name: str = "XGB"):
    # Convert to NumPy for speed
    X = DR.X_train.to_numpy()
    y = DR.y_train.to_numpy()
    indices = np.arange(X.shape[0])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def train_and_evaluate(train_idx, test_idx, fold_id, model_name):
        if model_name == "XGB":
            model = XGBoostModel(params=params)
        elif model_name == "CAT":
            model = CatBoostModel(params=params)
        elif model_name == "LGBM":
            model = LightGBMModel(params=params)
            
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        rmsle = np.sqrt(mean_squared_log_error(y[test_idx], preds))
        print(f"Fold {fold_id}: RMSLE = {rmsle:.4f}")
        preds_df = pd.DataFrame({'preds': preds,
                                 'id': test_idx})
        return rmsle, preds_df

    # Parallel processing across folds
    res = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(train_idx, test_idx, i, model_name)
        for i, (train_idx, test_idx) in enumerate(kf.split(indices))
    )

    rmsle_vals, preds_dfs = zip(*res)

    return np.mean(rmsle_vals), preds_dfs



params_xgb = {'random_state': 42, 'device': "cuda" if CFG['use_gpu'] else None, 'n_estimators': 1204, 'learning_rate': 0.010963358609400503, 'max_depth': 12, 'min_child_weight': 6, 'gamma': 0.11562716976469375, 'subsample': 0.7246113223440827, 'colsample_bytree': 0.734092929579249, 'reg_alpha': 0.22003477113027226, 'reg_lambda': 1.0117897222583774, 'eval_metric': 'rmse'}
RMSLE, preds_df_xgb = run_cv(params=params_xgb)
print(f"We get an average RMSLE of {RMSLE}")


params_cat = {'iterations': 1504, 'depth': 8, 'learning_rate': 0.07923161748568001, 'l2_leaf_reg': 8.834684049259137, 'border_count': 241, 'bagging_temperature': 0.5296294033480199, 'random_strength': 0.5765944876293042, 'od_type': 'Iter', 'od_wait': 40, 'eval_metric': 'RMSE', 'verbose': 0, 'task_type': 'CPU', 'random_seed': 42}
RMSLE, preds_df_cat = run_cv(params=params_cat, model_name="CAT")
print(f"We get an average RMSLE of {RMSLE}")


params_lgbm = {'n_estimators': 4923, 'learning_rate': 0.02192410412966341, 'max_depth': 5, 'num_leaves': 368, 'min_child_samples': 25, 'subsample': 0.6285647030061281, 'colsample_bytree': 0.4323227040774758, 'reg_alpha': 0.00010369737855852863, 'reg_lambda': 0.008974160518156832, 'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'verbose': -1, 'device': 'cpu', 'random_state': 42}
RMSLE, preds_df_lgbm = run_cv(params=params_lgbm, model_name="LGBM")
print(f"We get an average RMSLE of {RMSLE}")


from itertools import product

def blend_predictions(y_true, *model_predictions, step=0.01):
    # Concatenate and sort by 'id'
    blended_dfs = []

    for model_tuple in model_predictions:
        model_df = pd.concat(model_tuple).sort_values(by='id').reset_index(drop=True)
        blended_dfs.append(model_df)

    n_models = len(blended_dfs)
    weights = np.arange(0, 1 + step, step)

    best_score = float('inf')
    best_weights = None

    # Generate weight combinations that sum to 1 (within rounding tolerance)
    def weight_combinations(n, step):
        ranges = [weights] * n
        for combo in product(*ranges):
            if abs(sum(combo) - 1.0) < step / 2:
                yield combo

    for combo in weight_combinations(n_models, step):
        blended_preds = sum(w * df['preds'] for w, df in zip(combo, blended_dfs))
        score = np.sqrt(mean_squared_log_error(y_true, blended_preds))

        if score < best_score:
            best_score = score
            best_weights = combo

    return best_weights, best_score


best_weights, best_score = blend_predictions(
    DR.y_train,
    preds_df_xgb,
    preds_df_cat,
    preds_df_lgbm
)

print(f"Best RMSLE: {best_score}")
print(f"Best weights: {best_weights}")


best_params_xgb = params_xgb
best_params_cat = params_cat
best_params_lgbm = params_lgbm


xgb_w, cat_w, lgbm_w = best_weights


final_cat_m = CatBoostModel(best_params_cat)
final_cat_m.fit(DR.X_train, DR.y_train)
final_predictions_cat = final_cat_m.predict(DR.X_test)

final_xgb_m = XGBoostModel(best_params_xgb)
final_xgb_m.fit(DR.X_train, DR.y_train)
final_predictions_xgb = final_xgb_m.predict(DR.X_test)

final_lgbm_m = LightGBMModel(best_params_lgbm)
final_lgbm_m.fit(DR.X_train, DR.y_train)
final_predictions_lgbm = final_xgb_m.predict(DR.X_test)


final_predictions = final_predictions_cat * cat_w + final_predictions_xgb * xgb_w + final_predictions_lgbm * lgbm_w


final_predictions = np.clip(final_predictions, 1, 325)

submission = DR.sample_submission
submission = submission.with_columns(
    pl.Series("Calories", final_predictions)
    .alias("Calories")
)

submission.write_csv("submission.csv")


importance_dict = final_xgb_m.model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df = importance_df.head(n=20)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()


def get_optuna_params(trial, model):
    if model == "XGB": 
         params = {
            'n_estimators': trial.suggest_int('n_estimators', 250, 5000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'eval_metric': 'rmse',
            'device': "cuda" if CFG['use_gpu'] else None,
             'random_state': 42,
            }
    elif model == "CAT":
        params = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'depth': trial.suggest_int('depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
            'od_type': 'Iter',
            'od_wait': trial.suggest_int('od_wait', 20, 50),
            'eval_metric': 'RMSE',
            'verbose': 0,
            'task_type': "GPU" if CFG['use_gpu'] else 'CPU',
            'random_seed': 42,
            }
    elif model == "LGBM":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 250, 5000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 16, 512),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'objective': 'regression',
            'metric': 'rmse',  # gets passed to eval_metric in fit()
            'verbosity': -1,
            'verbose': -1,
            'device': 'gpu' if CFG['use_gpu'] else 'cpu',
            'random_state': 42,
        }

    return params


def objective(trial, model: str="XGB"):
    # Define the hyperparameters to tune
    hyperparams = get_optuna_params(trial, model)
    print(hyperparams)

    rmsle, _ = run_cv(n_splits=5, params=hyperparams, model_name=model)

    return rmsle

MODEL = "LGBM"
# Uncomment the following two lines to run hyperparameter optimization
# study = optuna.create_study(direction='minimize')
# study.optimize(lambda trial: objective(trial, MODEL), n_trials=50)

# # Initializing with the params we used last time
# study.enqueue_trial(params_lgbm)

# print("Best hyperparams:", study.best_params)


# LGBM
# {'n_estimators': 1588, 'learning_rate': 0.01927495481257143, 'max_depth': 9, 'num_leaves': 336, 'min_child_samples': 21, 'subsample': 0.869332764995689, 'colsample_bytree': 0.5714615170499757, 'reg_alpha': 0.7893051593875492, 'reg_lambda': 6.038059701186995e-08}
# CAT
# {'iterations': 1504, 'depth': 8, 'learning_rate': 0.07923161748568001, 'l2_leaf_reg': 8.834684049259137, 'border_count': 241, 'bagging_temperature': 0.5296294033480199, 'random_strength': 0.5765944876293042, 'od_type': 'Iter', 'od_wait': 40, 'eval_metric': 'RMSE', 'verbose': 0, 'task_type': 'CPU', 'random_seed': 42}

