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


import gc, warnings, sys
from typing import Sequence, Dict, Tuple
 
import numpy as np
import pandas as pd
 
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.metrics         import make_scorer
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import StratifiedShuffleSplit

import optuna
 
warnings.filterwarnings("ignore")


df_tr = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_te = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
OUTPUT_CSV = "/kaggle/working/results_multi_stacking3.csv"


RND            = 42
N_SPLITS       = 5
N_TRIALS_BLEND = 50         # weight optimiser


def prep(
    df_tr: pd.DataFrame, df_te: pd.DataFrame, tgt="Personality", idx="id"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Preprocess the training and test datasets.
    """  
    # Define feature groups
    num_base = df_tr.select_dtypes(include=['float64']).columns.tolist()
    cat_base = df_tr.select_dtypes(include=['object']).columns.tolist()

    # Remove target and id from feature lists if present
    if tgt in cat_base:
        cat_base.remove(tgt)
    if idx in num_base:
        num_base.remove(idx)
    if idx in cat_base:
        cat_base.remove(idx)

    # Combine feature groups
    num_features = num_base
    cat_features = cat_base

    # Downcast numerical columns
    for c in num_features:
        if c in df_tr.columns:
            df_tr[c] = pd.to_numeric(df_tr[c], downcast="float")
        if c in df_te.columns:
            df_te[c] = pd.to_numeric(df_te[c], downcast="float")

    # Convert categorical columns
    for c in cat_features:
        if c in df_tr.columns:
            df_tr[c] = df_tr[c].fillna('missing')
            df_tr[c] = df_tr[c].astype("category")
        if c in df_te.columns:
            df_te[c] = df_te[c].fillna('missing')
            df_te[c] = df_te[c].astype("category")

    # Drop the index column if it exists
    if idx in df_tr.columns:
        df_tr = df_tr.drop(columns=[idx])
    if idx in df_te.columns:
        df_te = df_te.drop(columns=[idx])

    # FIX: Use single imputer fitted on training data for both datasets
    imputer = IterativeImputer(max_iter=1000, random_state=42)

    # Fit imputer on training data only
    imputer.fit(df_tr[num_features])

    # Transform both train and test using the same fitted imputer
    df_tr[num_features] = imputer.transform(df_tr[num_features])

    # Only transform test data if it has the same numerical features
    test_num_features = [c for c in num_features if c in df_te.columns]
    if test_num_features:
        df_te[test_num_features] = imputer.transform(df_te[test_num_features])

    # Encode the target variable
    le_tgt = LabelEncoder()
    ytr = pd.Series(le_tgt.fit_transform(df_tr[tgt]), name=tgt)
    df_tr = df_tr.drop(columns=[tgt])

    return df_tr, df_te, ytr, le_tgt


def build_stack_with_params(params: dict, seed: int) -> Pipeline:
    """
    Build a stacking classifier with fixed parameters instead of Optuna trial.
    
    Parameters:
    - params: Dictionary containing hyperparameters
    - seed: Random seed for reproducibility
    
    Returns:
    - Pipeline: Stacking classifier pipeline
    """
    cat_base = ["Stage_fear", "Drained_after_socializing"]
    cat_columns = cat_base 

    # Build XGBoost parameters
    xgb_params = {
        "tree_method": "gpu_hist",
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "enable_categorical": True,
        "random_state": seed,
        "n_estimators": params["xgb_n"],
        "learning_rate": params["xgb_lr"],
        "max_depth": params["xgb_d"],
        "subsample": params["xgb_sub"],
        "colsample_bytree": params["xgb_col"],
        "reg_alpha": params["xgb_alpha"],
        "reg_lambda": params["xgb_lambda"],
        "gamma": params["xgb_gamma"],
        "min_child_weight": params["xgb_min_child"],
        "grow_policy": params["xgb_grow"],
        "max_bin": 256,
        "verbosity": 0
    }
    
    # Add max_leaves if grow_policy is lossguide
    if params["xgb_grow"] == "lossguide":
        xgb_params["max_leaves"] = params["xgb_leaves"]
        
    xgb_clf = xgb.XGBClassifier(**xgb_params)
 
    # Build LightGBM parameters
    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        device_type="gpu",
        verbose=-1,
        random_state=seed,
        categorical_feature=cat_columns,
        n_estimators=params["lgb_n"],
        learning_rate=params["lgb_lr"],
        max_depth=params["lgb_d"],
        subsample=params["lgb_sub"],
        colsample_bytree=params["lgb_col"],
        num_leaves=params["lgb_leaves"],
        min_child_samples=params["lgb_min_child"],
        min_child_weight=params["lgb_min_weight"],
        reg_alpha=params["lgb_alpha"],
        reg_lambda=params["lgb_lambda"],
        cat_smooth=params["lgb_cat_smooth"],
        cat_l2=params["lgb_cat_l2"],
        max_bin=255,
        min_data_in_bin=params["lgb_min_data_bin"],
        boost_from_average=True,
        force_row_wise=True,
        path_smooth=params["lgb_path_smooth"],
    )

    # Build CatBoost parameters
    cat_params = {
        "task_type": "GPU",
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_state": seed,
        "cat_features": cat_columns,
        "iterations": params["cat_n"],
        "learning_rate": params["cat_lr"],
        "depth": params["cat_d"],
        "l2_leaf_reg": params["cat_l2"],
        "random_strength": params["cat_rs"],
        "leaf_estimation_iterations": params["cat_leaf_iters"],
        "grow_policy": params["cat_grow"],
        "min_data_in_leaf": params["cat_min_data"],
        "bootstrap_type": params["cat_bootstrap"],
        "border_count": params["cat_border_count"],
        "verbose": False
    }
    
    # Add bootstrap-type-specific parameters
    if params["cat_bootstrap"] == "Bayesian":
        cat_params["bagging_temperature"] = params["cat_temp"]
    elif params["cat_bootstrap"] in ["Bernoulli", "MVS"]:
        cat_params["subsample"] = params["cat_subsample"]
    
    cat_clf = cb.CatBoostClassifier(**cat_params)
    
    # Meta-learner for binary classification
    meta = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=1.0
    )
 
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
 
    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf),
                    ("lgb", lgb_clf),
                    ("cat", cat_clf)],
        final_estimator=meta,
        stack_method="predict",
        cv=skf_inner,
    )
 
    return Pipeline([("stk", stk)])


def build_stack_c_with_params(params: dict, seed: int) -> Pipeline:
    """
    Build Stack C (XGBoost + CatBoost) with fixed parameters instead of Optuna trial.
    
    Parameters:
    - params: Dictionary containing hyperparameters
    - seed: Random seed for reproducibility
    
    Returns:
    - Pipeline: Stacking classifier pipeline
    """
    cat_base = ["Stage_fear", "Drained_after_socializing"]
    cat_columns = cat_base 

    # Build XGBoost parameters
    xgb_params = {
        "tree_method": "gpu_hist",
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "enable_categorical": True,
        "random_state": seed,
        "n_estimators": params["xgb_n"],
        "learning_rate": params["xgb_lr"],
        "max_depth": params["xgb_d"],
        "subsample": params["xgb_sub"],
        "colsample_bytree": params["xgb_col"],
        "reg_alpha": params["xgb_alpha"],
        "reg_lambda": params["xgb_lambda"],
        "gamma": params["xgb_gamma"],
        "min_child_weight": params["xgb_min_child"],
        "grow_policy": params["xgb_grow"],
        "max_bin": 255,
        "verbosity": 0
    }
    
    # Add max_leaves if grow_policy is lossguide
    if params["xgb_grow"] == "lossguide":
        xgb_params["max_leaves"] = params["xgb_leaves"]
        
    xgb_clf = xgb.XGBClassifier(**xgb_params)

    # Build CatBoost parameters
    cat_params = {
        "task_type": "CPU",
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_state": seed,
        "cat_features": cat_columns,
        "iterations": params["cat_n"],
        "learning_rate": params["cat_lr"],
        "depth": params["cat_d"],
        "l2_leaf_reg": params["cat_l2"],
        "random_strength": params["cat_rs"],
        "leaf_estimation_iterations": params["cat_leaf_iters"],
        "grow_policy": params["cat_grow"],
        "min_data_in_leaf": params["cat_min_data"],
        "bootstrap_type": params["cat_bootstrap"],
        "border_count": params["cat_border_count"],
        "verbose": False
    }
    
    # Add bootstrap-type-specific parameters
    if params["cat_bootstrap"] == "Bayesian":
        cat_params["bagging_temperature"] = params["cat_temp"]
    elif params["cat_bootstrap"] in ["Bernoulli", "MVS"]:
        cat_params["subsample"] = params["cat_subsample"]
    
    cat_clf = cb.CatBoostClassifier(**cat_params)
    
    # Meta-learner
    meta = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=1.0
    )
 
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
 
    stk = StackingClassifier(
        estimators=[("xgb", xgb_clf),
                    ("cat", cat_clf)],  # Only XGBoost + CatBoost
        final_estimator=meta,
        stack_method="predict",
        cv=skf_inner,
    )
 
    return Pipeline([("stk", stk)])


def oof_probs(model_builder, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    """
    Generate out-of-fold probabilities for blending.

    Parameters:
    - model_builder: Function to build the model.
    - X: Training features as a pandas DataFrame.
    - y: Target variable as a pandas Series.
    - X_test: Test features as a pandas DataFrame.

    Returns:
    - oof: Out-of-fold probabilities.
    - preds_test: Test predictions.
    """
    # Determine the number of classes from the target variable
    oof = np.zeros(len(y), dtype=np.float32)
    preds_test = np.zeros(len(X_test), dtype=np.float32)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

    for tr_idx, val_idx in skf.split(X, y):
        mdl = model_builder()
        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        mdl.fit(X_train, y_train)
        oof[val_idx] = mdl.predict(X_val)
        preds_test += mdl.predict(X_test) / N_SPLITS
        gc.collect()
    return oof, preds_test


X_full, X_test, y_full, le = prep(df_tr, df_te)


prep_parameters_a = {
    'xgb_grow': 'lossguide', 
    'xgb_n': 493, 
    'xgb_lr': 0.023868262254178974, 
    'xgb_d': 7, 
    'xgb_sub': 0.8753072008027474, 
    'xgb_col': 0.781495914291181, 
    'xgb_alpha': 0.04878515811662384, 
    'xgb_lambda': 2.2383318217344836, 
    'xgb_gamma': 0.008080175150365632, 
    'xgb_min_child': 9, 
    'xgb_leaves': 63, 
    'lgb_n': 603, 
    'lgb_lr': 0.08081869122041364, 
    'lgb_d': 9, 
    'lgb_sub': 0.9020699753674115, 
    'lgb_col': 0.6231154461331532, 
    'lgb_leaves': 52, 
    'lgb_min_child': 43, 
    'lgb_min_weight': 0.0012604052359173753, 
    'lgb_alpha': 0.0324554485035341, 
    'lgb_lambda': 0.6559322100104664, 
    'lgb_cat_smooth': 14, 
    'lgb_cat_l2': 3.3433774954219633, 
    'lgb_min_data_bin': 17, 
    'lgb_path_smooth': 0.07067298907315209, 
    'cat_bootstrap': 'Bayesian', 
    'cat_n': 324, 
    'cat_lr': 0.026044813438110267, 
    'cat_d': 4, 
    'cat_l2': 8.802463666209638, 
    'cat_rs': 4.707878201632036, 
    'cat_leaf_iters': 9, 
    'cat_grow': 'Lossguide', 
    'cat_min_data': 4, 
    'cat_border_count': 174, 
    'cat_temp': 0.590885582751283
}


prep_parameters_b = {
    'xgb_grow': 'lossguide', 
    'xgb_n': 832, 
    'xgb_lr': 0.0233479494631388, 
    'xgb_d': 4, 
    'xgb_sub': 0.7820963791108293, 
    'xgb_col': 0.8264989606417548, 
    'xgb_alpha': 0.3189942267344725, 
    'xgb_lambda': 3.6264988370832234, 
    'xgb_gamma': 4.450566082167217, 
    'xgb_min_child': 7, 
    'xgb_leaves': 124, 
    'lgb_n': 498, 
    'lgb_lr': 0.1330989456416932, 
    'lgb_d': 1, 
    'lgb_sub': 0.6747355686220816, 
    'lgb_col': 0.930144787279552, 
    'lgb_leaves': 57, 
    'lgb_min_child': 38, 
    'lgb_min_weight': 0.21114008315137103, 
    'lgb_alpha': 1.4496988019147825, 
    'lgb_lambda': 7.092432075748109, 
    'lgb_cat_smooth': 27, 
    'lgb_cat_l2': 6.130651492914251, 
    'lgb_min_data_bin': 19, 
    'lgb_path_smooth': 0.059154498371806696, 
    'cat_bootstrap': 'Bernoulli', 
    'cat_n': 799, 
    'cat_lr': 0.09552454970211413, 
    'cat_d': 7, 
    'cat_l2': 3.8802465047753403, 
    'cat_rs': 7.649032883725466, 
    'cat_leaf_iters': 4, 
    'cat_grow': 'SymmetricTree', 
    'cat_min_data': 2, 
    'cat_border_count': 161, 
    'cat_subsample': 0.6806723539774329
}

# Optimized parameters for Stack C (XGBoost + CatBoost combination)
prep_parameters_c = {
    'xgb_grow': 'lossguide', 
    'xgb_n': 300, 
    'xgb_lr': 0.034871141409842626, 
    'xgb_d': 8, 
    'xgb_sub': 0.9501009276231958, 
    'xgb_col': 0.7808014305373993, 
    'xgb_alpha': 0.11615571784741217, 
    'xgb_lambda': 3.486752038540946, 
    'xgb_gamma': 2.538190490278329, 
    'xgb_min_child': 8, 
    'xgb_leaves': 52, 
    'cat_bootstrap': 'Bayesian', 
    'cat_n': 205, 
    'cat_lr': 0.06332905131826944, 
    'cat_d': 5, 
    'cat_l2': 4.919985708819264, 
    'cat_rs': 3.666048922259925, 
    'cat_leaf_iters': 5, 
    'cat_grow': 'Lossguide', 
    'cat_min_data': 9, 
    'cat_border_count': 172, 
    'cat_temp': 0.7161621156130958
}


def builder_A():
    return build_stack_with_params(prep_parameters_a, seed=RND)

def builder_B():
    return build_stack_with_params(prep_parameters_b, seed=2024)

def builder_C():
    return build_stack_c_with_params(prep_parameters_c, seed=1337)


print("Generating OOF on tuning sample â€¦")
oof_A, _ = oof_probs(builder_A, X_full, y_full, X_test[:1])  # dummy test
oof_B, _ = oof_probs(builder_B, X_full, y_full, X_test[:1])
oof_C, _ = oof_probs(builder_C, X_full, y_full, X_test[:1])


print("Refitting Stack A on full data â€¦")
mdl_A = builder_A(); 
mdl_A.fit(X_full, y_full)


mdl_B = builder_B(); 
mdl_B.fit(X_full, y_full)


mdl_C = builder_C(); 
mdl_C.fit(X_full, y_full)


def blend_obj(trial):
    wA = trial.suggest_float("wA", 0.1, 0.8)
    wB = trial.suggest_float("wB", 0.1, 0.8)
    wC = trial.suggest_float("wC", 0.1, 0.8)
    
    # Normalize weights to sum to 1.0
    total = wA + wB + wC
    wA /= total
    wB /= total
    wC /= total
    
    blended_continuous = wA * oof_A + wB * oof_B + wC * oof_C
    blended_predictions = (blended_continuous >= 0.5).astype(int)
    return accuracy_score(y_full, blended_predictions)

study_blend = optuna.create_study(direction="maximize")
study_blend.optimize(blend_obj, n_trials=N_TRIALS_BLEND)

# Get normalized weights
wA = study_blend.best_params["wA"]
wB = study_blend.best_params["wB"]
wC = study_blend.best_params["wC"]

total = wA + wB + wC
wA /= total
wB /= total
wC /= total
print(f"Blend weights: wA={wA:.3f}, wB={wB:.3f}, wC={wC:.3f}")


# Get continuous predictions
proba_test_continuous = (
    wA * mdl_A.predict(X_test) + 
    wB * mdl_B.predict(X_test) + 
    wC * mdl_C.predict(X_test)
)
# Convert to discrete predictions using threshold
proba_test_discrete = (proba_test_continuous >= 0.5).astype(int)
# Apply inverse transform to discrete predictions
personality = le.inverse_transform(proba_test_discrete)


Sub = pd.DataFrame(
    {
        "id": submission.id, 
        "Personality": personality
    }
)


Sub


Sub.to_csv(OUTPUT_CSV, index=False)

