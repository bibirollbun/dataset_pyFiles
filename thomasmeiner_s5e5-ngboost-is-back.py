# ------------------------------------------------------------
# 0‒---------  Imports & helpers
# ------------------------------------------------------------
!pip install ngboost -q
!pip uninstall xgboost --y
!pip install -q xgboost>=1.7.6
import warnings, optuna, numpy as np, pandas as pd, re
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from xgboost   import XGBRegressor        # pip install xgboost
from lightgbm  import LGBMRegressor       # pip install lightgbm
from ngboost   import NGBoost             # pip install ngboost
from ngboost.distns import Normal
from ngboost.scores import MLE


# ------------------------------------------------------------
# 1‒---------  Data
# ------------------------------------------------------------
PATH = "/kaggle/input/playground-series-s5e5"
train = pd.read_csv(f"{PATH}/train.csv")
test  = pd.read_csv(f"{PATH}/test.csv")

target = "Calories"

#train = train.sample(100000, random_state=4000)

X = train.drop(columns=[target]).copy()
train[target] = np.log1p(train[target])

y = train[target].values
test_id = test["id"]

# map Sex → 0/1 (and keep as numeric column)
X["Sex"]    = X["Sex"].map({"male": 0, "female": 1}).astype("int8")
test["Sex"] = test["Sex"].map({"male": 0, "female": 1}).astype("int8")

NUM_COLS = X.columns            # after the mapping every column is numeric


# ------------------------------------------------------------
# 2‒---------  Reusable scorer
# ------------------------------------------------------------
def rmsle(y_true, y_pred):
    """Root mean-squared logarithmic error with non-negative preds."""
    y_pred = np.maximum(0, np.expm1(y_pred))
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), y_pred))


def cv_rmsle(model, X, y, cv):
    """3-fold CV and graceful failure (returns np.nan if fit crashes)."""
    scores = []
    for tr, va in cv.split(X):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_va, y_va = X.iloc[va], y[va]

        try:
            model.fit(X_tr, y_tr)            # <-- wrapped as requested
        except Exception as e:               # any learner can blow up
            return np.nan

        scores.append(rmsle(y_va, model.predict(X_va)))

    return float(np.mean(scores))


CV = KFold(n_splits=5, shuffle=True, random_state=42)


# ------------------------------------------------------------
# 3‒---------  Optuna objective  (NGBoost only)
# ------------------------------------------------------------
def objective(trial: optuna.trial.Trial) -> float:

    # ------------------------------------------------ 3.1 pick the base learner
    base_name = trial.suggest_categorical(
        "base", ["dtree", "linear", "xgb", "lgbm", "rf", "hgb"]
    )

    if base_name == "dtree":                       # classic CART stump
        from sklearn.tree import DecisionTreeRegressor
        Base = DecisionTreeRegressor(
            max_depth          = trial.suggest_int("dt_depth", 2, 6),
            min_samples_split  = trial.suggest_int("dt_split", 2, 20),
            random_state       = 42,
        )

    elif base_name == "linear":                    # ridge as in the paper
        from sklearn.linear_model import Ridge
        Base = Ridge(alpha=trial.suggest_float("ridge_alpha", 0.1, 10.0, log=True))

    elif base_name == "xgb":                       # **new**
        Base = XGBRegressor(
            n_estimators      = trial.suggest_int("xgb_n", 50, 300),
            max_depth         = trial.suggest_int("xgb_depth", 3, 7),
            learning_rate     = trial.suggest_float("xgb_lr", 0.05, 0.3, log=True),
            subsample         = trial.suggest_float("xgb_sub", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("xgb_col", 0.4, 1.0),
            objective         = "reg:squarederror",
            tree_method       = "hist",
            random_state      = 42,
            n_jobs            = -1,
        )

    elif base_name == "lgbm":                      # **new**
        Base = LGBMRegressor(
            n_estimators      = trial.suggest_int("lgbm_n", 50, 300),
            max_depth         = trial.suggest_int("lgbm_depth", -1, 8),
            learning_rate     = trial.suggest_float("lgbm_lr", 0.05, 0.3, log=True),
            num_leaves        = trial.suggest_int("lgbm_leaves", 8, 64),
            subsample         = trial.suggest_float("lgbm_sub", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("lgbm_col", 0.4, 1.0),
            reg_lambda        = trial.suggest_float("lgbm_l2", 0.0, 3.0),
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )

    elif base_name == "rf":                        # random forest stump
        Base = RandomForestRegressor(
            n_estimators      = trial.suggest_int("rf_n", 30, 100),
            max_depth         = trial.suggest_int("rf_depth", 3, 8),
            max_features      = "sqrt",
            n_jobs            = -1,
            random_state      = 42,
        )

    else:  # "hgb"                                 # hist-GBR stump
        Base = HistGradientBoostingRegressor(
            max_depth         = trial.suggest_int("hgb_depth", 3, 7),
            learning_rate     = 1.0,      # single-tree learner ⇒ lr=1
            max_iter          = 1,        # *one* tree per NGB iteration
            random_state      = 42,
        )

    # ------------------------------------------------ 3.2 NGB hyper-params
    mdl = NGBoost(
        Dist            = Normal,
        Score           = MLE,
        Base            = Base,
        n_estimators    = trial.suggest_int("ngb_n", 300, 1200),
        learning_rate   = trial.suggest_float("ngb_lr", 0.01, 0.3, log=True),
        minibatch_frac  = trial.suggest_float("ngb_mb", 0.5, 1.0),
        natural_gradient= True,
        random_state    = 42,
        verbose         = False,
    )

    # ------------------------------------------------ 3.3 Pipeline + CV
    pipe  = make_pipeline(StandardScaler(), mdl)
    score = cv_rmsle(pipe, X, y, CV)

    if np.isnan(score):                       # defensive pruning
        raise optuna.exceptions.TrialPruned()

    return score


# ------------------------------------------------------------
# 4‒---------  Run the study (unchanged except fewer trials)
# ------------------------------------------------------------
SEED  = 42
study = optuna.create_study(
    direction   = "minimize",
    sampler     = optuna.samplers.TPESampler(seed=SEED),
    study_name  = "ngboost_calories_rmsle",
)

study.optimize(
    objective,
    n_trials           = 300,      # explore a bit more now that search space grew
    timeout            = 3600 * 10,
    show_progress_bar  = True,
    catch              = (Exception,),
)

print("Best RMSLE :", study.best_value)
print("Best params:", study.best_params)


# ------------------------------------------------------------
# 5‒---------  Train the best pipeline on full data & predict
# ------------------------------------------------------------
def build_best_ngb(params):
    """Re-instantiates the winning NGBoost model from Optuna params."""
    base_name = params["base"]

    # ---------- 5.1 rebuild the base learner -------------------------------
    if base_name == "dtree":
        from sklearn.tree import DecisionTreeRegressor
        Base = DecisionTreeRegressor(
            max_depth         = params["dt_depth"],
            min_samples_split = params["dt_split"],
            random_state      = 42,
        )
    elif base_name == "linear":
        from sklearn.linear_model import Ridge
        Base = Ridge(alpha=params["ridge_alpha"])

    elif base_name == "xgb":
        Base = XGBRegressor(
            n_estimators      = params["xgb_n"],
            max_depth         = params["xgb_depth"],
            learning_rate     = params["xgb_lr"],
            subsample         = params["xgb_sub"],
            colsample_bytree  = params["xgb_col"],
            objective         = "reg:squarederror",
            tree_method       = "hist",
            random_state      = 42,
            n_jobs            = -1,
        )

    elif base_name == "lgbm":
        Base = LGBMRegressor(
            n_estimators      = params["lgbm_n"],
            max_depth         = params["lgbm_depth"],
            learning_rate     = params["lgbm_lr"],
            num_leaves        = params["lgbm_leaves"],
            subsample         = params["lgbm_sub"],
            colsample_bytree  = params["lgbm_col"],
            reg_lambda        = params["lgbm_l2"],
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )

    elif base_name == "rf":
        Base = RandomForestRegressor(
            n_estimators      = params["rf_n"],
            max_depth         = params["rf_depth"],
            max_features      = "sqrt",
            n_jobs            = -1,
            random_state      = 42,
        )

    else:  # "hgb"
        Base = HistGradientBoostingRegressor(
            max_depth         = params["hgb_depth"],
            learning_rate     = 1.0,
            max_iter          = 1,
            random_state      = 42,
        )

    # ---------- 5.2 rebuild NGBoost itself ---------------------------------
    mdl = NGBoost(
        Dist             = Normal,
        Score            = MLE,
        Base             = Base,
        n_estimators     = params["ngb_n"],
        learning_rate    = params["ngb_lr"],
        minibatch_frac   = params["ngb_mb"],
        natural_gradient = True,
        random_state     = 42,
        verbose          = False,
    )

    return make_pipeline(StandardScaler(), mdl)


best_pipe = build_best_ngb(study.best_params)

best_pipe.fit(X, y)                       # full-data training
pred = np.maximum(0, np.expm1(best_pipe.predict(test[NUM_COLS])))

# ------------------------------------------------------------
# 6‒---------  Submission (unchanged)
# ------------------------------------------------------------
pd.DataFrame({"id": test_id, "Calories": pred}).to_csv("submission.csv", index=False)




