# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.special import logit

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import shutil
import optuna
import glob
import json

warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s5e10/train.csv'
    test_path = '/kaggle/input/playground-series-s5e10/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
    
    target = 'accident_risk'
    n_folds = 10
    seed = 42
    
    cv = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    metric = accuracy_score
    
    n_optuna_trials = 500


train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test


train.head()


object_cols = []
num_cols = []
bool_cols = []
for col in X.columns:
    if X[col].dtype == 'O':
        object_cols.append(col)
        print('\n',col)
        print(X[col].unique())
    elif X[col].dtype == 'bool':
        bool_cols.append(col)
    else:
        num_cols.append(col)


num_cols


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Keep this for the set_style line

sns.set_style("whitegrid")

plot_cols = X[num_cols].columns 
num_features = len(plot_cols)

n_cols = int(np.ceil(np.sqrt(num_features)))
n_rows = int(np.ceil(num_features / n_cols))

# Create the figure and subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(plot_cols):
    # The data X[col] is correct because 'col' is guaranteed to be in X
    axes[i].hist(X[col], bins=20, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# Remove unused axes
for j in range(num_features, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


sns.set_style("white")
plt.figure(figsize=(8, 8))

corr_train = X[num_cols].corr()
mask_train = np.triu(np.ones_like(corr_train, dtype=bool), k=1)

sns.heatmap(
    data=corr_train,
    annot=True,
    fmt='.4f',
    mask=mask_train,
    square=True,
    cmap='coolwarm',
    annot_kws={'size': 8},
    cbar=False
)

plt.tight_layout()
plt.show()


import os
import math
import time
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib
import optuna
import lightgbm as lgb  # for callbacks
import logging
import warnings
from tqdm.auto import tqdm

from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, ParameterSampler, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore", message=".*gpu_hist.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*predictor.*is not used.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*", category=UserWarning)
logging.getLogger("xgboost").setLevel(logging.ERROR)


class ModelTrainer:
    """
    Trainer with automatic object->category conversion and fit_params for cat features
    Supports xgb, lgb, cat. Ensures models can see categorical columns.
    """
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        mode: str = "regression",
        n_splits: int = 5,
        seed: int = 42,
        n_optuna_trials: int = 100,
        use_gpu_if_available: bool = True,
        early_stopping_rounds: int = 50,
        n_jobs: int = -1,
    ):
        assert mode in ("regression", "classification")
        self.X = X.copy()
        self.y = y.copy()
        self.X_test = X_test.copy() if X_test is not None else None
        self.mode = mode
        self.n_splits = n_splits
        self.seed = seed
        self.n_optuna_trials = n_optuna_trials
        self.use_gpu_if_available = use_gpu_if_available
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs

        # scorer
        if self.mode == "regression":
            self.scorer = make_scorer(lambda y_true, y_pred: -math.sqrt(mean_squared_error(y_true, y_pred)))
        else:
            from sklearn.metrics import accuracy_score
            self.scorer = make_scorer(accuracy_score)

        # GPU heuristic
        self.gpu_available = self._check_gpu_available() if self.use_gpu_if_available else False

        # storage
        self.best_models: Dict[str, Any] = {}
        self.search_results: Dict[str, Any] = {}

        # convert object dtypes -> categorical and prepare lists for fit params
        self.cat_cols: List[str] = []
        self.cat_idx: List[int] = []
        self._convert_object_to_category()

    def _check_gpu_available(self) -> bool:
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "":
            return True
        try:
            with os.popen("nvidia-smi -L") as p:
                out = p.read()
            if "GPU" in out:
                return True
        except Exception:
            pass
        return False

    def _convert_object_to_category(self):
        # convert object columns in X (and X_test when available) to pandas categorical
        obj_cols = self.X.select_dtypes(include=["object"]).columns.tolist()
        if len(obj_cols) == 0:
            # also consider string dtype if present (pandas >= 1.0 has string dtype)
            obj_cols = self.X.select_dtypes(include=["string"]).columns.tolist()
        for c in obj_cols:
            # convert preserving missing values
            self.X[c] = self.X[c].astype("category")
            if self.X_test is not None and c in self.X_test.columns:
                self.X_test[c] = self.X_test[c].astype("category")
        # store categorical columns and indices (useful for CatBoost)
        self.cat_cols = self.X.select_dtypes(include=["category"]).columns.tolist()
        self.cat_idx = [self.X.columns.get_loc(c) for c in self.cat_cols]

    def _get_cv_splitter(self):
        return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed) if self.mode=="classification" else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

    def _base_params(self, model_key: str) -> Dict[str, Any]:
        gpu = self.gpu_available
        if model_key == "xgb":
            if self.mode == "regression":
                params = {"objective": "reg:squarederror", "random_state": self.seed, "verbosity": 0}
            else:
                params = {"objective": "binary:logistic", "random_state": self.seed, "verbosity": 0}
            # enable categorical support in XGBoost (newer xgboost versions)
            params.update({"enable_categorical": True})
            if gpu:
                params.update({"device": "cuda", "tree_method": "hist"})
            return params

        if model_key == "lgb":
            if self.mode == "regression":
                params = {"objective": "regression", "random_state": self.seed}
            else:
                params = {"objective": "binary", "random_state": self.seed}
            if gpu:
                params.update({"device_type": "gpu", "device": "gpu"})
            return params

        if model_key == "cat":
            if self.mode == "regression":
                params = {"loss_function": "RMSE", "random_seed": self.seed}
            else:
                params = {"loss_function": "Logloss", "random_seed": self.seed}
            if gpu:
                params.update({"task_type": "GPU", "devices": "0"})
            return params

        raise ValueError("Unknown model key")

    def _init_model(self, model_key: str, **override):
        base = self._base_params(model_key)
        base.update(override or {})
        if model_key == "xgb":
            return XGBRegressor(**base) if self.mode == "regression" else XGBClassifier(**base)
        if model_key == "lgb":
            return LGBMRegressor(**base, n_jobs=self.n_jobs) if self.mode == "regression" else LGBMClassifier(**base, n_jobs=self.n_jobs)
        if model_key == "cat":
            return CatBoostRegressor(**base) if self.mode == "regression" else CatBoostClassifier(**base)
        raise ValueError("Unknown model key")

    def _build_fit_params(self, model_key: str) -> Dict[str, Any]:
        """
        Build dict of fit/time kwargs so that fit/cv functions know which columns are categorical.
        - CatBoost: cat_features (list of names or indices)
        - LightGBM: categorical_feature (list of names)
        - XGBoost: enabled via enable_categorical in constructor (no fit param)
        """
        fit_params: Dict[str, Any] = {}
        if not self.cat_cols:
            return fit_params
        if model_key == "cat":
            # CatBoost accepts cat_features as list of names or indices
            fit_params["cat_features"] = self.cat_cols  # names are fine
        elif model_key == "lgb":
            # LightGBM sklearn API accepts categorical_feature in fit()
            fit_params["categorical_feature"] = self.cat_cols
        # xgb handled via enable_categorical in constructor
        return fit_params

    def _default_param_distributions(self, model_key: str):
        if model_key == "xgb":
            return {
                "n_estimators": [100, 200, 400, 800, 1200],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "max_depth": [3, 5, 7, 9, 12],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.4, 0.6, 0.7, 0.8, 1.0],
                "reg_alpha": [0, 1e-3, 1e-2, 0.1, 1.0],
                "reg_lambda": [0, 1e-3, 1e-2, 0.1, 1.0],
            }
        if model_key == "lgb":
            return {
                "n_estimators": [100, 200, 400, 800, 1200],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "num_leaves": [16, 31, 60, 100, 200],
                "max_depth": [-1, 5, 7, 10, 15],
                "feature_fraction": [0.4, 0.6, 0.7, 0.9, 1.0],
                "bagging_fraction": [0.4, 0.6, 0.8, 1.0],
                "lambda_l1": [0, 1e-3, 1e-2, 0.1],
                "lambda_l2": [0, 1e-3, 1e-2, 0.1],
            }
        if model_key == "cat":
            return {
                "iterations": [200, 400, 800, 1200],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "depth": [3, 5, 7, 9, 12],
                "l2_leaf_reg": [1, 3, 5, 7, 10],
                "random_strength": [0.0, 1.0, 5.0, 10.0],
            }
        return {}

    def fit_random_search(
        self,
        model_key: str,
        param_distributions: Optional[Dict[str, list]] = None,
        n_iter: int = 40,
        cv: Optional[Any] = None,
        scoring=None,
        verbose: int = 1,
        n_jobs: Optional[int] = None,
        refit: bool = True,
        use_tqdm: bool = False,
    ):
        if model_key not in ("xgb", "lgb", "cat"):
            raise ValueError("model_key must be one of 'xgb','lgb','cat'")

        param_distributions = param_distributions or self._default_param_distributions(model_key)
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        cv_splitter = cv if cv is not None else self._get_cv_splitter()
        fit_params = self._build_fit_params(model_key)

        # Choose estimator base
        base_estimator = self._init_model(model_key)

        if use_tqdm:
            sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=self.seed))
            best_score = None
            best_est = None
            results = []
            pbar = tqdm(sampler, desc=f"RandomSearch({model_key})", leave=True)
            for params in pbar:
                try:
                    est = self._init_model(model_key, **params)
                    scores = cross_val_score(est, self.X, self.y, cv=cv_splitter, scoring=scoring or self.scorer, n_jobs=n_jobs, fit_params=fit_params)
                    mean_score = np.mean(scores)
                    results.append((params, mean_score))
                    if best_est is None or mean_score > (best_score if best_score is not None else -np.inf):
                        best_score = mean_score
                        best_est = est
                    pbar.set_postfix({"score": float(mean_score)})
                except Exception as e:
                    pbar.write(f"param set failed: {e}")
                    continue
            if best_est is not None:
                # fit best estimator on full data with fit params
                best_est.fit(self.X, self.y, **fit_params)
                self.best_models[model_key] = best_est
                self.search_results[("random", model_key)] = results
            return self.search_results.get(("random", model_key), None)
        else:
            rand = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scoring or self.scorer,
                cv=cv_splitter,
                random_state=self.seed,
                n_jobs=n_jobs,
                refit=refit,
            )
            # pass fit_params so underlying fit() receives categorical info
            rand.fit(self.X, self.y, **fit_params)
            self.search_results[("random", model_key)] = rand
            self.best_models[model_key] = rand.best_estimator_
            return rand

    def fit_optuna(self, model_key: str, trials: Optional[int] = None, timeout: Optional[int] = None):
        if model_key not in ("xgb", "lgb", "cat"):
            raise ValueError("model_key must be one of 'xgb','lgb','cat'")

        trials = trials or self.n_optuna_trials
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner()

        def objective(trial: optuna.Trial):
            if model_key == "xgb":
                params = {
                    "n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 800, 1200]),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                    "random_state": self.seed,
                }
                params.update(self._base_params("xgb"))
                model_cls = XGBRegressor if self.mode == "regression" else XGBClassifier
                params.setdefault("verbosity", 0)
            elif model_key == "lgb":
                params = {
                    "n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 800, 1200]),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                    "max_depth": trial.suggest_int("max_depth", -1, 16),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
                    "random_state": self.seed,
                }
                params.update(self._base_params("lgb"))
                model_cls = LGBMRegressor if self.mode == "regression" else LGBMClassifier
            else:
                params = {
                    "iterations": trial.suggest_categorical("iterations", [300, 600, 1000]),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "depth": trial.suggest_int("depth", 3, 12),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    "random_seed": self.seed,
                }
                params.update(self._base_params("cat"))
                model_cls = CatBoostRegressor if self.mode == "regression" else CatBoostClassifier

            cv_splitter = self._get_cv_splitter()
            fold_scores = []

            for fold_idx, (trn_idx, val_idx) in enumerate(tqdm(list(cv_splitter.split(self.X, self.y)),
                                                               desc=f"trial-{trial.number}-folds", leave=False, total=self.n_splits)):
                X_tr, X_val = self.X.iloc[trn_idx], self.X.iloc[val_idx]
                y_tr, y_val = self.y.iloc[trn_idx], self.y.iloc[val_idx]
                fit_params = self._build_fit_params(model_key)

                if model_key == "xgb":
                    model = model_cls(**params)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], **fit_params)
                    preds = model.predict(X_val)

                elif model_key == "lgb":
                    model = model_cls(**params, n_jobs=self.n_jobs)
                    callbacks = [lgb.log_evaluation(0)]
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks, **fit_params)
                    preds = model.predict(X_val)

                else:
                    model = model_cls(**params)
                    # pass cat features to Pool (CatBoost)
                    eval_pool = Pool(X_val, y_val, cat_features=self.cat_cols) if self.cat_cols else Pool(X_val, y_val)
                    model.fit(X_tr, y_tr, eval_set=eval_pool, cat_features=self.cat_cols if self.cat_cols else None)
                    preds = model.predict(X_val)

                if self.mode == "regression":
                    score = math.sqrt(mean_squared_error(y_val, preds))
                else:
                    from sklearn.metrics import accuracy_score
                    score = 1.0 - accuracy_score(y_val, (preds > 0.5).astype(int) if preds.ndim == 1 else preds)

                fold_scores.append(score)
                trial.report(np.mean(fold_scores), fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(fold_scores))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed),
                                   pruner=optuna.pruners.MedianPruner())
        print(f"[Optuna] Starting study for {model_key} with trials={trials} gpu={self.gpu_available}")
        study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=True)

        best_params = study.best_params
        print(f"[Optuna] Best params: {best_params}")

        final_params = dict(best_params)
        final_params.update(self._base_params(model_key))

        fit_params_final = self._build_fit_params(model_key)

        if model_key == "xgb":
            final_model = XGBRegressor(**final_params) if self.mode == "regression" else XGBClassifier(**final_params)
            final_model.fit(self.X, self.y, **fit_params_final)
        elif model_key == "lgb":
            final_model = LGBMRegressor(**final_params, n_jobs=self.n_jobs) if self.mode == "regression" else LGBMClassifier(**final_params, n_jobs=self.n_jobs)
            final_model.fit(self.X, self.y, **fit_params_final)
        else:
            final_model = CatBoostRegressor(**final_params) if self.mode == "regression" else CatBoostClassifier(**final_params)
            # CatBoost: pass cat_features when fitting
            final_model.fit(self.X, self.y, cat_features=self.cat_cols if self.cat_cols else None)

        self.search_results[("optuna", model_key)] = study
        self.best_models[model_key] = final_model
        return study

    def cross_validate_model(self, model, return_preds: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        cv_splitter = self._get_cv_splitter()
        oof_preds = np.zeros(len(self.y))
        scores = []

        # attempt to infer model key for fit_params
        if isinstance(model, (XGBRegressor, XGBClassifier)):
            mk = "xgb"
        elif isinstance(model, (LGBMRegressor, LGBMClassifier)):
            mk = "lgb"
        else:
            mk = "cat"
        fit_params = self._build_fit_params(mk)

        for fold_idx, (trn_idx, val_idx) in enumerate(tqdm(list(cv_splitter.split(self.X, self.y)),
                                                           desc="CV folds", total=self.n_splits)):
            X_tr, X_val = self.X.iloc[trn_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y.iloc[trn_idx], self.y.iloc[val_idx]

            if isinstance(model, (XGBRegressor, XGBClassifier)):
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], **fit_params)
                preds = model.predict(X_val)

            elif isinstance(model, (LGBMRegressor, LGBMClassifier)):
                callbacks = [lgb.log_evaluation(0)]
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks, **fit_params)
                preds = model.predict(X_val)

            else:  # CatBoost
                eval_pool = Pool(X_val, y_val, cat_features=self.cat_cols) if self.cat_cols else Pool(X_val, y_val)
                model.fit(X_tr, y_tr, eval_set=eval_pool, cat_features=self.cat_cols if self.cat_cols else None)
                preds = model.predict(X_val)

            oof_preds[val_idx] = preds
            if self.mode == "regression":
                score = math.sqrt(mean_squared_error(y_val, preds))
            else:
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, (preds > 0.5).astype(int) if preds.ndim == 1 else preds)
            scores.append(score)

        mean_score = float(np.mean(scores))
        if return_preds:
            return mean_score, oof_preds
        return mean_score, None

    def predict_test(self, model_key: str):
        if model_key not in self.best_models:
            raise ValueError(f"No trained model for key {model_key}")
        if self.X_test is None:
            raise ValueError("No X_test provided")
        model = self.best_models[model_key]
        # ensure X_test has category dtype already set in __init__; just pass through
        return model.predict(self.X_test)

    def save_model(self, model_key: str, path: str):
        if model_key not in self.best_models:
            raise ValueError("No model to save for this key")
        model = self.best_models[model_key]
        joblib.dump(model, path)
        print(f"[Save] Model {model_key} saved to {path}")

    def load_model(self, path: str):
        return joblib.load(path)

    def feature_importances(self, model_key: str, top_n: int = 30):
        if model_key not in self.best_models:
            raise ValueError("No trained model found for that key")
        model = self.best_models[model_key]
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            fi = model.get_feature_importance()
        else:
            raise ValueError("Model does not expose feature importances")
        cols = self.X.columns
        df = pd.DataFrame({"feature": cols, "importance": fi}).sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n)

    def summary(self):
        for k, v in self.best_models.items():
            print(f"Model: {k} -> {type(v)}")



trainer = ModelTrainer(X, y, X_test=X_test, mode='regression', n_splits=CFG.n_folds, seed=CFG.seed)
rand_res = trainer.fit_random_search('xgb', n_iter=30)
study = trainer.fit_optuna('lgb', trials=100)
print(trainer.cross_validate_model(trainer.best_models['lgb']))
preds = trainer.predict_test('lgb')
trainer.save_model('lgb', '/kaggle/working/lgb_best.pkl')


model


import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "/kaggle/working/lgb_best.pkl"
OUT_SUB = "submission_lgb_best.csv"
FALLBACK_STRATEGY = "first"  # "first" (mặc định) | "mode_if_in_train" | None

model = joblib.load(MODEL_PATH)

def _get_booster(m):
    return getattr(m, "booster_", None) or getattr(m, "_Booster", None) or getattr(m, "Booster", None)

booster = _get_booster(model)
feat_names = None
try:
    feat_names = booster.feature_name() if booster is not None else None
except Exception:
    feat_names = None

pandas_cats = getattr(booster, "pandas_categorical", None)
print("Booster.feature_name() length:", len(feat_names) if feat_names is not None else None)
print("type(pandas_cats):", type(pandas_cats), "len(pandas_cats):", (len(pandas_cats) if isinstance(pandas_cats, (list,tuple,dict)) else None))

# determine candidate categorical columns in X_test (object or category dtypes)
candidate_cat_cols = list(X_test.select_dtypes(include=["object", "category"]).columns)
print("Candidate categorical columns in X_test:", candidate_cat_cols)

# Try to build a mapping col_name -> list_of_categories (exactly as train) in cat_map
cat_map = {}

if pandas_cats is None:
    print("No pandas_categorical metadata found in booster. Will cast object -> category as fallback.")
else:
    # Case A: dict mapping col -> categories
    if isinstance(pandas_cats, dict):
        for k, v in pandas_cats.items():
            cat_map[k] = list(v) if v is not None else None

    # Case B: list/tuple
    elif isinstance(pandas_cats, (list, tuple)):
        # If same length as feat_names -> map by position
        if feat_names is not None and len(pandas_cats) == len(feat_names):
            for fname, cats in zip(feat_names, pandas_cats):
                cat_map[fname] = list(cats) if cats is not None else None
        else:
            # If length matches number of candidate_cat_cols -> map by that order
            if len(pandas_cats) == len(candidate_cat_cols):
                for fname, cats in zip(candidate_cat_cols, pandas_cats):
                    cat_map[fname] = list(cats) if cats is not None else None
            else:
                # Try to detect items that look like (name, cats)
                found = False
                for item in pandas_cats:
                    if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                        name, cats = item
                        cat_map[name] = list(cats) if cats is not None else None
                        found = True
                if not found:
                    # As a last resort: zip into feat_names for first min length
                    if feat_names is not None:
                        for fname, cats in zip(feat_names, pandas_cats):
                            cat_map[fname] = list(cats) if cats is not None else None
                    else:
                        # give up but keep cat_map empty
                        pass
    else:
        print("Unhandled pandas_categorical structure:", type(pandas_cats))

# Display summary of cat_map (only keys present in X_test)
print("Built cat_map summary (only showing cols present in X_test):")
for k, v in cat_map.items():
    if k in X_test.columns:
        print(f"  {k}: {'None' if v is None else f'len={len(v)} sample={v[:5]}'}")

# Now apply stored categories to X_test where possible
X = X_test.copy()
for col, stored_cats in cat_map.items():
    if col not in X.columns:
        continue
    if stored_cats is None:
        # booster thinks this feature is categorical but no stored categories? skip
        print(f"Skipping {col}: stored_cats is None")
        continue
    # convert to category then set exact categories as stored
    X[col] = X[col].astype("category")
    # set categories to exactly stored list
    X[col] = X[col].cat.set_categories(stored_cats)
    # Replace unseen values (which became NaN) with fallback that exists in stored_cats
    if X[col].isna().any():
        # choose fallback
        fallback = None
        if FALLBACK_STRATEGY == "first":
            fallback = stored_cats[0] if len(stored_cats) > 0 else None
        elif FALLBACK_STRATEGY == "mode_if_in_train":
            try:
                mode_val = X[col].mode(dropna=True)
                if len(mode_val) > 0 and mode_val.iloc[0] in stored_cats:
                    fallback = mode_val.iloc[0]
                else:
                    fallback = stored_cats[0] if len(stored_cats) > 0 else None
            except Exception:
                fallback = stored_cats[0] if len(stored_cats) > 0 else None
        else:
            fallback = None

        if fallback is not None:
            # ensure fallback is among categories (it is, since from stored_cats)
            X[col] = X[col].fillna(fallback)
        else:
            # leave NaN (LightGBM allows NaN but we prefer explicit fill)
            pass

# For any remaining object dtypes that booster had no metadata for, cast to category (best-effort)
for c in X.select_dtypes(include=["object"]).columns:
    X[c] = X[c].astype("category")

# Final debug: list categorical columns and their categories (first few)
for c in X.columns:
    if pd.api.types.is_categorical_dtype(X[c].dtype):
        cats = list(X[c].cat.categories)
        print(f"Final COL {c}: categories count={len(cats)} sample={cats[:8]}{'...' if len(cats)>8 else ''}")

# Try predict
try:
    preds = model.predict(X)
except ValueError as e:
    print("LightGBM predict error (still):", e)
    # extra debug: lengths
    try:
        print("len(feat_names):", len(feat_names) if feat_names is not None else None)
        print("len(pandas_cats):", len(pandas_cats) if isinstance(pandas_cats, (list,tuple,dict)) else None)
        # dump a short sample of pandas_cats for debug
        if isinstance(pandas_cats, (list,tuple)):
            for i, item in enumerate(pandas_cats[:30]):
                print(f" pandas_cats[{i}] type={type(item)} repr={str(item)[:200]}")
        elif isinstance(pandas_cats, dict):
            for k in list(pandas_cats.keys())[:30]:
                print(f" pandas_cats[{k}] len={len(pandas_cats[k]) if pandas_cats[k] is not None else None}")
    except Exception as dbg:
        print("Debug failure:", dbg)
    raise

# postprocess preds same as before
if isinstance(preds, np.ndarray) and preds.ndim > 1 and preds.shape[1] == 1:
    preds = preds.ravel()
elif isinstance(preds, np.ndarray) and preds.ndim > 1 and preds.shape[1] > 1:
    preds = preds[:, 0]
preds = np.asarray(preds).ravel()

print("preds shape:", preds.shape, "n_test rows:", len(X))

# save submission
sub = pd.read_csv(CFG.sample_sub_path, index_col='id')
if set(X.index) == set(sub.index):
    sub = sub.reindex(X.index)
    sub[sub.columns[0]] = preds
else:
    sub = sub.reindex(X.index)
    sub[sub.columns[0]] = preds
sub.to_csv(OUT_SUB)
print(f"Saved submission to {OUT_SUB}")





