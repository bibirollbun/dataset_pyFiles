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


# ðŸ“¦ Imports
import pandas as pd
import numpy as np
import random
import joblib
import json
import optuna
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from scipy.special import logit
from koolbox import Trainer
from autogluon.tabular import TabularPredictor
import warnings

warnings.filterwarnings("ignore")

# ðŸ§ª Configuration
class CFG:
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    target = 'Personality'
    n_folds = 5
    seed = 42
    n_optuna_trials = 500
    cv = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    metric = accuracy_score

# ðŸ“… Load data
train = pd.read_csv(CFG.train_path, index_col='id')
test = pd.read_csv(CFG.test_path, index_col='id')
sample_submission = pd.read_csv(CFG.sample_sub_path)

train["Stage_fear"] = train["Stage_fear"].map({"No": 0, "Yes": 1})
train["Drained_after_socializing"] = train["Drained_after_socializing"].map({"No": 0, "Yes": 1})
test["Stage_fear"] = test["Stage_fear"].map({"No": 0, "Yes": 1})
test["Drained_after_socializing"] = test["Drained_after_socializing"].map({"No": 0, "Yes": 1})
train[CFG.target] = train[CFG.target].map({"Extrovert": 0, "Introvert": 1})

# ðŸ”§ Handle missing values before casting to int
cat_cols = ["Stage_fear", "Drained_after_socializing"]
train[cat_cols] = train[cat_cols].fillna(-1).astype(int)
test[cat_cols] = test[cat_cols].fillna(-1).astype(int)

X = train.drop(columns=[CFG.target])
y = train[CFG.target]
X_test = test

# ðŸ“Š Mutual Info Heatmap
plt.figure(figsize=(8, 6))
mutual_info = mutual_info_regression(X.fillna(0), y, random_state=CFG.seed)
mutual_info_series = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)

sns.barplot(x=mutual_info_series.values, y=mutual_info_series.index, palette="RdYlGn")
plt.title("Mutual Information Scores")
plt.tight_layout()
plt.show()

# ðŸ§  Base models
scores, oof_pred_probs, test_pred_probs = {}, {}, {}

# âœ… Define all tuned models
model_configs = {
    "CatBoost": CatBoostClassifier(border_count=39, colsample_bylevel=0.1946, depth=2, iterations=1467,
                                     l2_leaf_reg=31.2, learning_rate=0.0685, min_child_samples=160,
                                     random_state=CFG.seed, random_strength=0.85, scale_pos_weight=1.17,
                                     subsample=0.3192, verbose=False, cat_features=["Stage_fear", "Drained_after_socializing"]),
    "XGBoost": XGBClassifier(colsample_bylevel=0.8168, colsample_bynode=0.885, colsample_bytree=0.838,
                              gamma=2.4, learning_rate=0.0617, max_depth=344, max_leaves=89,
                              min_child_weight=10, n_estimators=696, n_jobs=-1, random_state=CFG.seed,
                              reg_alpha=1.85, reg_lambda=29.68, subsample=0.5903, verbosity=0,
                              enable_categorical=True),
    "HistGradientBoosting": HistGradientBoostingClassifier(l2_regularization=28.14, learning_rate=0.154,
                                                            max_depth=325, max_features=0.324, max_iter=2490,
                                                            max_leaf_nodes=216, min_samples_leaf=12,
                                                            random_state=CFG.seed),
    "LightGBM (gbdt)": LGBMClassifier(boosting_type='gbdt', colsample_bytree=0.6467, learning_rate=0.065,
                                      min_child_samples=34, min_child_weight=0.244, n_estimators=498,
                                      num_leaves=158, random_state=CFG.seed, reg_alpha=6.57,
                                      reg_lambda=62.66, subsample=0.0011, verbose=-1, n_jobs=-1),
    "LightGBM (goss)": LGBMClassifier(boosting_type='goss', colsample_bytree=0.8385, learning_rate=0.070,
                                      min_child_samples=46, min_child_weight=0.763, n_estimators=1887,
                                      num_leaves=341, random_state=CFG.seed, reg_alpha=10.53,
                                      reg_lambda=67.45, subsample=0.4925, verbose=-1, n_jobs=-1),
    "LightGBM (dart)": LGBMClassifier(boosting_type='dart', colsample_bytree=0.7593, learning_rate=0.0461,
                                      min_child_samples=18, min_child_weight=0.474, n_estimators=4035,
                                      num_leaves=393, random_state=CFG.seed, reg_alpha=48.02,
                                      reg_lambda=89.13, subsample=0.0163, verbose=-1, n_jobs=-1)
}

for name, model in model_configs.items():
    trainer = Trainer(model, cv=CFG.cv, metric=CFG.metric, use_early_stopping=False,
                      task="binary", metric_precision=6)
    trainer.fit(X, y)
    scores[name] = trainer.fold_scores
    oof_pred_probs[name] = trainer.oof_preds
    test_pred_probs[name] = trainer.predict(X_test)

# ðŸ§  AutoGluon
ag_train = train.copy()
ag_train[CFG.target] = ag_train[CFG.target].map({0: "Extrovert", 1: "Introvert"})

predictor = TabularPredictor(label=CFG.target).fit(ag_train)
ag_oof = predictor.predict_proba(ag_train)['Introvert'].values
ag_test = predictor.predict_proba(test)['Introvert'].values
oof_pred_probs["AutoGluon"] = ag_oof
test_pred_probs["AutoGluon"] = ag_test

# ðŸ§  Meta Model (Logistic Regression)
X_meta = logit(pd.DataFrame(oof_pred_probs).clip(1e-15, 1-1e-15))
X_test_meta = logit(pd.DataFrame(test_pred_probs).clip(1e-15, 1-1e-15))

def objective(trial):
    solver_penalty = trial.suggest_categorical("solver_penalty", [
        ("liblinear", "l1"), ("liblinear", "l2"),
        ("lbfgs", "l2"), ("lbfgs", None),
        ("newton-cg", "l2"), ("newton-cg", None),
        ("newton-cholesky", "l2"), ("newton-cholesky", None)
    ])
    solver, penalty = solver_penalty
    params = {
        'random_state': CFG.seed,
        'max_iter': 1000,
        'C': trial.suggest_float('C', 0.0001, 1.0),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'solver': solver,
        'penalty': penalty
    }
    threshold = trial.suggest_float('threshold', 0, 1)
    trainer = Trainer(LogisticRegression(**params), cv=CFG.cv, metric=CFG.metric,
                      metric_threshold=threshold, metric_precision=6, task="binary")
    trainer.fit(X_meta, y)
    return np.mean(trainer.fold_scores)

sampler = optuna.samplers.TPESampler(seed=CFG.seed)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=CFG.n_optuna_trials, n_jobs=-1)

best_params = study.best_params
threshold = best_params.pop('threshold')
solver, penalty = best_params.pop('solver_penalty')
lr_params = {**best_params, 'solver': solver, 'penalty': penalty, 'random_state': CFG.seed, 'max_iter': 1000}

lr_model = LogisticRegression(**lr_params)
lr_model.fit(X_meta, y)
logreg_test_preds = lr_model.predict_proba(X_test_meta)[:, 1]

scores["LogisticRegression"] = [study.best_value] * CFG.n_folds
oof_pred_probs["LogisticRegression"] = lr_model.predict_proba(X_meta)[:, 1]
test_pred_probs["LogisticRegression"] = logreg_test_preds

# ðŸŒŸ Weighted Ensembling
def weight_objective(trial):
    weights = np.array([trial.suggest_float(m, -1, 1) for m in oof_pred_probs.keys()])
    weights /= np.sum(weights)
    blended = sum(w * oof_pred_probs[m] for w, m in zip(weights, oof_pred_probs))
    threshold = trial.suggest_float('threshold', 0, 1)
    return accuracy_score(y, (blended > threshold).astype(int))

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=CFG.seed))
study.optimize(weight_objective, n_trials=CFG.n_optuna_trials, n_jobs=-1)

best_weights = {m: study.best_params[m] for m in oof_pred_probs}
threshold = study.best_params['threshold']

final_preds = np.zeros(len(X_test))
for m, w in best_weights.items():
    final_preds += test_pred_probs[m] * w

# ðŸ“† Submission
sub = sample_submission.copy()
sub[CFG.target] = (final_preds > threshold).astype(int)
sub[CFG.target] = sub[CFG.target].map({0: "Extrovert", 1: "Introvert"})
sub.to_csv("submission.csv", index=False)
print("\nâœ… Final weighted submission saved as 'submission.csv'")
print(sub.head())

# ðŸ“Š Final Score Visualization
score_df = pd.DataFrame(scores)
plt.figure(figsize=(10, 5))
sns.boxplot(data=score_df, orient="h", palette="coolwarm")
plt.title("Fold Accuracy Comparison")
plt.tight_layout()
plt.show()

shutil.rmtree("catboost_info", ignore_errors=True)





