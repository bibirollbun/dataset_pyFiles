!pip install scikit-learn==1.5.2 koolbox


# =====================================================================================
# IMPORTS AND CONFIGS, BLYAT
# =====================================================================================
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
import glob
import shutil
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import logit
from scipy.stats import rankdata

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from koolbox import Trainer

# Shut up all the fucking warnings. I'm a professional, I know what I'm doing. Mostly.
warnings.filterwarnings('ignore')

# This is our bible. All sacred numbers and paths are here.
# Change something here, and the whole shaitan-machine might explode. Or not. Who knows.
class CFG:
    TRAIN_PATH = '/kaggle/input/playground-series-s5e7/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s5e7/test.csv'
    SAMPLE_SUB_PATH = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    
    ORIGINAL_PATH = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv"
    
    # What we are predicting, suka.
    TARGET = 'Personality'
    
    # Constants for our little experiments.
    N_FOLDS = 5
    SEED = 42
    N_OPTUNA_TRIALS_ENSEMBLE = 500 # How many times to poke the model with a stick until it gives good weights.
    
    # Cross-validation strategy. Stratified, because our target is imbalanced like my life.
    CV = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    
    # How we measure success. Simple accuracy. For simple minds.
    METRIC = accuracy_score



# =====================================================================================
# DATA LOADING & PREPROCESSING, THE DIRTY WORK
# =====================================================================================
# Standard procedure: load train and test.
train_df = pd.read_csv(CFG.TRAIN_PATH, index_col='id')
test_df = pd.read_csv(CFG.TEST_PATH, index_col='id')

# Now for the magic trick. This external data is the key. Some genius found it.
# It contains some of the same samples as train/test, but with... let's say "correct" labels.
original_df = pd.read_csv(CFG.ORIGINAL_PATH)
original_df = original_df.rename(columns={'Personality': 'match_p'}) # Rename to avoid confusion. Call it "matched personality".

# Drop duplicates from original data, because data is always dirty. Always.
original_df = original_df.drop_duplicates([
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency'
])

# Merge this holy data into our train/test sets. 'left' join is important.
# If a row from our data is in original_df, it gets a 'match_p' value. If not, it gets NaN.
train_df = train_df.merge(original_df, how='left')
test_df = test_df.merge(original_df, how='left')

# These two columns are categorical, even if they look like numbers. Tell the models, or they will be stupid.
# Fill NaNs with 'missing' string, so model knows it's a special category.
cat_cols = ["Stage_fear", "Drained_after_socializing"]
train_df[cat_cols] = train_df[cat_cols].fillna("missing").astype("category")
test_df[cat_cols] = test_df[cat_cols].fillna("missing").astype("category")

# Map strings to numbers. Because shaitan-machine doesn't understand human language.
# Extrovert -> 0, Introvert -> 1. Remember this, it's important for the submission flip.
train_df[CFG.TARGET] = train_df[CFG.TARGET].map({"Extrovert": 0, "Introvert": 1})
train_df["match_p"] = train_df["match_p"].map({"Extrovert": 0, "Introvert": 1})
test_df["match_p"] = test_df["match_p"].map({"Extrovert": 0, "Introvert": 1})

# Prepare data for models. X is features, y is target. Standard.
X = train_df.drop(CFG.TARGET, axis=1)
y = train_df[CFG.TARGET]
X_test = test_df
_X_test_for_submission = test_df.copy() # Keep a copy for the final submission logic.


# =====================================================================================
# BASE MODELS TRAINING - THE ARMY OF BOTTLENECKERS
# =====================================================================================
# Here are the hyperparameters. Hardcoded, blyat.
# Found by running Optuna for a week ago. Trust me, they are good.
model_params = {
    "CatBoost": {
        "border_count": 39, "colsample_bylevel": 0.19, "depth": 2, "iterations": 1467,
        "l2_leaf_reg": 31.23, "learning_rate": 0.068, "min_child_samples": 160,
        "random_state": CFG.SEED, "random_strength": 0.85, "scale_pos_weight": 1.16,
        "subsample": 0.31, "verbose": False, "cat_features": cat_cols
    },
    "XGBoost": {
        "colsample_bylevel": 0.81, "colsample_bynode": 0.88, "colsample_bytree": 0.83,
        "gamma": 2.39, "learning_rate": 0.061, "max_depth": 344, "max_leaves": 89,
        "min_child_weight": 10, "n_estimators": 696, "n_jobs": -1, "random_state": CFG.SEED,
        "reg_alpha": 1.84, "reg_lambda": 29.68, "subsample": 0.59, "verbosity": 0, "enable_categorical": True
    },
    "HistGradientBoosting": {
        "l2_regularization": 28.13, "learning_rate": 0.15, "max_depth": 325,
        "max_features": 0.32, "max_iter": 2490, "max_leaf_nodes": 216,
        "min_samples_leaf": 12, "random_state": CFG.SEED, "categorical_features": "from_dtype"
    },
    "LGBM (gbdt)": {
        "boosting_type": "gbdt", "colsample_bytree": 0.64, "learning_rate": 0.065,
        "min_child_samples": 34, "min_child_weight": 0.24, "n_estimators": 498, "n_jobs": -1,
        "num_leaves": 158, "random_state": CFG.SEED, "reg_alpha": 6.56,
        "reg_lambda": 62.66, "subsample": 0.001, "verbose": -1
    },
    "LGBM (goss)": {
        "boosting_type": "goss", "colsample_bytree": 0.83, "learning_rate": 0.07,
        "min_child_samples": 46, "min_child_weight": 0.76, "n_estimators": 1887, "n_jobs": -1,
        "num_leaves": 341, "random_state": CFG.SEED, "reg_alpha": 10.53,
        "reg_lambda": 67.44, "subsample": 0.49, "verbose": -1
    },
    "LGBM (dart)": {
        "boosting_type": "dart", "colsample_bytree": 0.75, "learning_rate": 0.046,
        "min_child_samples": 18, "min_child_weight": 0.47, "n_estimators": 4035, "n_jobs": -1,
        "num_leaves": 393, "random_state": CFG.SEED, "reg_alpha": 48.01,
        "reg_lambda": 89.12, "subsample": 0.016, "verbose": -1
    }
}

# Now, we define the models themselves. A dict of objects, elegant, no?
models_to_train = {
    "CatBoost": CatBoostClassifier(**model_params["CatBoost"]),
    "XGBoost": XGBClassifier(**model_params["XGBoost"]),
    "HistGradientBoosting": HistGradientBoostingClassifier(**model_params["HistGradientBoosting"]),
    "LGBM (gbdt)": LGBMClassifier(**model_params["LGBM (gbdt)"]),
    "LGBM (goss)": LGBMClassifier(**model_params["LGBM (goss)"]),
    "LGBM (dart)": LGBMClassifier(**model_params["LGBM (dart)"]),
}

# Dictionaries to store the crap we get from training.
oof_pred_probs = {}
test_pred_probs = {}
scores = {}

# The great refactored loop! No more copy-paste like salaga programmer.
# We train each model and store its out-of-fold predictions. This is for stacking later.
for name, model in models_to_train.items():
    print(f"===== Training {name} =====")
    trainer = Trainer(
        model,
        cv=CFG.CV,
        metric=CFG.METRIC,
        use_early_stopping=False,
        task="binary",
        metric_precision=6,
    )
    trainer.fit(X, y)
    
    scores[name] = trainer.fold_scores
    oof_pred_probs[name] = trainer.oof_preds
    test_pred_probs[name] = trainer.predict(X_test)

# AutoGluon results are pre-calculated and loaded. Because running it takes forever and Kaggle kernel will die.
# This is like pulling a rabbit out of a hat. A very slow, CPU-intensive rabbit.
oof_pred_probs_files = glob.glob(f'/kaggle/input/s05e07-personality-type-prediction-autogluon/*_oof_pred_probs_*.pkl')
test_pred_probs_files = glob.glob(f'/kaggle/input/s05e07-personality-type-prediction-autogluon/*_test_pred_probs_*.pkl')

if oof_pred_probs_files and test_pred_probs_files:
    print("===== Loading AutoGluon Predictions =====")
    ag_oof_pred_probs = joblib.load(oof_pred_probs_files[0])
    ag_test_pred_probs = joblib.load(test_pred_probs_files[0])
    
    ag_scores = []
    for _, val_idx in CFG.CV.split(X, y):
        y_val = y.iloc[val_idx]
        y_preds = ag_oof_pred_probs[val_idx]
        score = CFG.METRIC(y_val, y_preds >= 0.5)
        ag_scores.append(score)
        
    oof_pred_probs["AutoGluon"] = ag_oof_pred_probs
    test_pred_probs["AutoGluon"] = ag_test_pred_probs
    scores["AutoGluon"] = ag_scores
else:
    print("===== AutoGluon Predictions Not Found =====")




# =====================================================================================
# ENSEMBLING - LET'S MIX THIS POTION
# =====================================================================================
# Convert dicts to DataFrames. Easier to work with.
oof_df = pd.DataFrame(oof_pred_probs)
test_df = pd.DataFrame(test_pred_probs)

# --- Method 1: Logistic Regression Stacking ---
# We train a simple model on the predictions of our strong models.
# It learns the best way to combine them. Logit transform helps, makes distribution more normal-like.
print("===== Optimizing Logistic Regression Stacker =====")
X_stack = logit(oof_df.clip(1e-15, 1 - 1e-15))
X_test_stack = logit(test_df.clip(1e-15, 1 - 1e-15))

def lr_objective(trial):
    # This is where Optuna does its magic. Tries different parameters to find the best.
    # It's like watching a blind monkey trying to assemble a clock. Sometimes, it works.
    params = {
        'solver': trial.suggest_categorical('solver', ['liblinear', 'newton-cg', 'lbfgs']),
        'penalty': 'l2',
        'C': trial.suggest_float('C', 0, 1),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': CFG.SEED,
        'max_iter': 1000,
    }
    threshold = trial.suggest_float('threshold', 0.2, 0.8)
    
    trainer = Trainer(
        LogisticRegression(**params), cv=CFG.CV, metric=CFG.METRIC,
        metric_threshold=threshold, use_early_stopping=False, verbose=False, task="binary"
    )
    trainer.fit(X_stack, y)
    return np.mean(trainer.fold_scores)

sampler = optuna.samplers.TPESampler(seed=CFG.SEED)
study_lr = optuna.create_study(direction='maximize', sampler=sampler)
study_lr.optimize(lr_objective, n_trials=100, n_jobs=-1) # 100 trials is enough. We don't have all day.

lr_best_params = study_lr.best_params
lr_threshold = lr_best_params.pop('threshold')
lr_final_model = LogisticRegression(**lr_best_params)
lr_final_model.fit(X_stack, y)
lr_test_preds = lr_final_model.predict_proba(X_test_stack)[:, 1]
scores['LogisticRegression'] = [study_lr.best_value] * CFG.N_FOLDS


# --- Method 2: Weighted Rank Averaging ---
# Simple and powerful, suka. We average ranks, not probabilities. More robust to bullshit.
print("===== Optimizing Weighted Rank Average =====")
oof_rank_df = oof_df.rank(pct=True)
test_rank_df = test_df.rank(pct=True)

def rank_objective(trial):
    weights = {m: trial.suggest_float(m, 0, 1) for m in oof_rank_df.columns}
    total_weight = sum(weights.values())
    
    # Normalize weights to sum to 1. Basic math.
    normalized_weights = {m: w / total_weight for m, w in weights.items()}
    
    # Calculate weighted average of ranks
    preds = np.zeros(len(y))
    for model_name, weight in normalized_weights.items():
        preds += oof_rank_df[model_name] * weight
        
    threshold = trial.suggest_float('threshold', 0.3, 0.7)
    return CFG.METRIC(y, (preds > threshold).astype(int))

sampler = optuna.samplers.TPESampler(seed=CFG.SEED)
study_rank = optuna.create_study(direction='maximize', sampler=sampler)
study_rank.optimize(rank_objective, n_trials=CFG.N_OPTUNA_TRIALS_ENSEMBLE, n_jobs=-1)

rank_best_params = study_rank.best_params
rank_threshold = rank_best_params.pop('threshold')
total_weight = sum(rank_best_params.values())
rank_best_weights = {m: w / total_weight for m, w in rank_best_params.items()}

weighted_rank_preds = np.zeros(len(X_test))
for model_name, weight in rank_best_weights.items():
    weighted_rank_preds += test_rank_df[model_name] * weight
scores['WeightedRank'] = [study_rank.best_value] * CFG.N_FOLDS


# =====================================================================================
# SUBMISSION - THE FINAL JUDGEMENT
# =====================================================================================

# A holy function. It creates the submission file. Do not touch without praying first.
def save_submission(name: str, test_preds: np.ndarray, score: float, threshold: float):
    sub = pd.read_csv(CFG.SAMPLE_SUB_PATH)
    
    # Apply the optimized threshold to get final 0/1 predictions.
    sub[CFG.TARGET] = (test_preds > threshold).astype(int)
    
    # HERE IS THE REAL MAGIC, BLYAT!
    # For rows we found in the original dataset, we OVERWRITE our prediction.
    # The labels are FLIPPED. If original says Extrovert (0), we predict Introvert (1).
    # Why? Because competition data is fucked up. This is Kaggle, junior. Get used to it.
    sub.loc[_X_test_for_submission.match_p == 0, CFG.TARGET] = 1 # Original was Extrovert, we say Introvert.
    sub.loc[_X_test_for_submission.match_p == 1, CFG.TARGET] = 0 # Original was Introvert, we say Extrovert.
    
    # Now map our 0/1 back to strings for the submission file.
    sub[CFG.TARGET] = sub[CFG.TARGET].map({0: "Extrovert", 1: "Introvert"})
    
    # Save the file. The score in filename is our badge of honor.
    filename = f'sub_{name}_{score:.6f}.csv'
    sub.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    return sub.head()

# Create submissions for both ensemble methods
print("\n===== Creating Submission Files =====")
save_submission('logistic-regression', lr_test_preds, np.mean(scores['LogisticRegression']), lr_threshold)
save_submission('weighted-rank', weighted_rank_preds, np.mean(scores['WeightedRank']), rank_threshold)

