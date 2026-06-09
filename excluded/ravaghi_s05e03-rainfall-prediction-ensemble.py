from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
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
import os
import gc

warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/playground-series-s5e3/train.csv"
    test_path = "/kaggle/input/playground-series-s5e3/test.csv"
    sample_sub_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"
    
    original_path = "/kaggle/input/hongkongrainfall/hongkong.csv"
    original_path_2 = "/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv"
    
    target = "rainfall"
    n_folds = 10
    seed = 2100
    
    run_optuna = True
    optuna_trials = 250


train = pd.read_csv(CFG.train_path, index_col="id")
test = pd.read_csv(CFG.test_path, index_col="id")


test.winddirection = test.winddirection.fillna(test.winddirection.median())


original = pd.read_csv(CFG.original_path, encoding="gbk")
original["date"] = pd.to_datetime(original[["year", "month", "day"]])
original = original.drop(["year", "month", "day", "low visibility hour", "radiation", "evaporation"], axis=1)
original["day"] = original.date.dt.dayofyear
original = original.drop("date", axis=1)
original.rainfall = original.rainfall.apply(lambda x: 1 if str(x).replace('.', '', 1).isdigit() else x)
original.rainfall = original.rainfall.replace({'微量': 1, '-': 0}).astype(int)
original.sunshine = original.sunshine.replace('-', 0).astype(float)
original.windspeed = original.windspeed.fillna(original.windspeed.mean())
for col in original.columns:
    original[col] = original[col].astype(train[col].dtype)


original_2 = pd.read_csv(CFG.original_path_2)
original_2.columns = original_2.columns.str.replace(" ", "")
original_2[CFG.target] = original_2[CFG.target].map({"yes": 1, "no": 0})
original_2.winddirection = original_2.winddirection.fillna(original_2.winddirection.mean())
original_2.windspeed = original_2.windspeed.fillna(original_2.windspeed.mean())
original_2.day = original_2.index + 1
for col in original_2.columns:
    original_2[col] = original_2[col].astype(train[col].dtype)


original_combined = pd.concat([original, original_2], axis=0).reset_index(drop=True)
original_combined = original_combined.drop_duplicates().reset_index(drop=True)


X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test.copy()

X_original = original_combined.drop(CFG.target, axis=1)
y_original = original_combined[CFG.target]


os.makedirs("oof_files", exist_ok=True)


class Trainer:
    def __init__(self, model, config=CFG, is_ensemble_model=False):
        self.model = model
        self.config = config
        self.is_ensemble_model = is_ensemble_model

    def fit_predict(self, X, y, X_test, X_original=None, y_original=None):
        print(f"Training {self.model.__class__.__name__}\n")
        
        scores = []        
        coeffs = np.zeros((1, X.shape[1]))
        oof_pred_probs = np.zeros(X.shape[0])
        test_pred_probs = np.zeros(X_test.shape[0])
        
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if X_original is not None and y_original is not None:                            
                X_train = pd.concat([X_train, X_original], ignore_index=True)
                y_train = pd.concat([y_train, y_original], ignore_index=True)
            
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            if self.is_ensemble_model:
                coeffs += model.coef_ / self.config.n_folds
                if isinstance(self.model, LogisticRegression):
                    n_iters = model.n_iter_[0]
            
            y_pred_probs = model.predict(X_val) if isinstance(self.model, Ridge) else model.predict_proba(X_val)[: ,1]
            oof_pred_probs[val_idx] = y_pred_probs 
            
            temp_test_pred_probs = model.predict(X_test) if isinstance(self.model, Ridge) else model.predict_proba(X_test)[:, 1]
            test_pred_probs += temp_test_pred_probs / self.config.n_folds
            
            score = roc_auc_score(y_val, y_pred_probs)
            scores.append(score)
            
            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()
            
            if self.is_ensemble_model and isinstance(self.model, LogisticRegression):
                print(f"--- Fold {fold_idx + 1} - ROC AUC: {score:.6f} ({n_iters} iterations)")
            else:
                print(f"--- Fold {fold_idx + 1} - ROC AUC: {score:.6f}")
            
        overall_score = roc_auc_score(y, oof_pred_probs)
            
        print(f"\n------ Overall: {overall_score:.6f} | Average: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
        
        if self.is_ensemble_model:
            return oof_pred_probs, test_pred_probs, overall_score, scores, coeffs
        else:
            self._save_oof_files(oof_pred_probs, test_pred_probs, overall_score)
            return oof_pred_probs, test_pred_probs, overall_score, scores
        
    def tune(self, X, y):             
        oof_pred_probs = np.zeros(X.shape[0])
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            y_pred_probs = model.predict(X_val) if isinstance(self.model, Ridge) else model.predict_proba(X_val)[: ,1]
            oof_pred_probs[val_idx] = y_pred_probs 
            
            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()
            
        return roc_auc_score(y, oof_pred_probs)
        
    def _save_oof_files(self, oof_pred_probs, test_pred_probs, cv_score):
        dirname = self.model.__class__.__name__.lower().replace("classifier", "")
        if (isinstance(self.model, LGBMClassifier)):
            if self.model.boosting_type == "goss":
                dirname += "-goss"
        os.makedirs(f"oof_files/{dirname}", exist_ok=True)
        joblib.dump(oof_pred_probs, f"oof_files/{dirname}/{dirname}_oof_pred_probs_{cv_score:.6f}.pkl")
        joblib.dump(test_pred_probs, f"oof_files/{dirname}/{dirname}_test_pred_probs_{cv_score:.6f}.pkl")


def save_submission(name, test_pred_probs, score):
    sub = pd.read_csv(CFG.sample_sub_path)
    sub[CFG.target] = test_pred_probs
    sub.to_csv(f"sub_{name}_{score:.6f}.csv", index=False)
    return sub.head()


scores = {}
overall_scores = {}
oof_pred_probs = {}
test_pred_probs = {}


xgb_params = {
    'colsample_bylevel': 0.7921066237164537,
    'colsample_bynode': 0.6431557579286489,
    'colsample_bytree': 0.33314916328121835,
    'gamma': 2.6533897486162306,
    'learning_rate': 0.0995872230739346,
    'max_depth': 488,
    'max_leaves': 313,
    'min_child_weight': 9,
    'n_estimators': 4644,
    'n_jobs': -1,
    'random_state': 2100,
    'reg_alpha': 0.07653965420877373,
    'reg_lambda': 56.09661479066265,
    'subsample': 0.987487242879055,
    'verbosity': 0
}

lgbm_params = {
    'boosting_type': 'gbdt',
    'colsample_bytree': 0.4631207130753891,
    'learning_rate': 0.088361660753031,
    'min_child_samples': 403,
    'min_child_weight': 0.4745749540750245,
    'n_estimators': 1332,
    'n_jobs': -1,
    'num_leaves': 266,
    'random_state': 2100,
    'reg_alpha': 23.886417233917868,
    'reg_lambda': 4.283869171990928,
    'subsample': 0.4581272309859017,
    'verbose': -1
}

lgbm_goss_params = {
    'boosting_type': 'goss',
    'colsample_bytree': 0.5342219347521369,
    'learning_rate': 0.027933718824492148,
    'min_child_samples': 119,
    'min_child_weight': 0.32511481168533785,
    'n_estimators': 178,
    'n_jobs': -1,
    'num_leaves': 446,
    'random_state': 2100,
    'reg_alpha': 8.432039948050928,
    'reg_lambda': 17.819535549058962,
    'subsample': 0.7901926393348829,
    'verbose': -1
}

cb_params = {
    'border_count': 134,
    'colsample_bylevel': 0.9849872675758802,
    'depth': 6,
    'iterations': 539,
    'l2_leaf_reg': 61.50587067708284,
    'learning_rate': 0.004062605169435353,
    'min_child_samples': 235,
    'random_state': 2100,
    'random_strength': 0.008446274380078389,
    'subsample': 0.5566962326912488,
    'verbose': False
}

adb_params = {
    'learning_rate': 0.0980729594457042,
    'n_estimators': 156,
    'random_state': 2100
}

rf_params = {
    'min_samples_leaf': 10,
    'min_samples_split': 7,
    'n_estimators': 1147,
    'n_jobs': -1,
    'random_state': 2100
}

et_params = {
    'class_weight': None,
    'criterion': 'log_loss',
    'min_samples_leaf': 3,
    'min_samples_split': 60,
    'n_estimators': 248,
    'n_jobs': -1,
    'random_state': 2100
}

histgb_params = {
    'l2_regularization': 8.627208188446245,
    'learning_rate': 0.02563674957851913,
    'max_depth': 370,
    'max_iter': 116,
    'max_leaf_nodes': 45,
    'min_samples_leaf': 457,
    'random_state': 2100
}

gb_params = {
    'learning_rate': 0.027839422097051633,
    'max_depth': 3,
    'max_features': 0.6246118798224226,
    'max_leaf_nodes': 8,
    'min_samples_leaf': 0.016482964092890184,
    'min_samples_split': 0.08735068463936646,
    'min_weight_fraction_leaf': 0.04933563916241057,
    'n_estimators': 240,
    'random_state': 2100,
    'subsample': 0.9417926258447458
}

lr_params = {
    'random_state': 2100,
    'max_iter': 500,
    'C': 57.03909649998898,
    'tol': 0.0030827492543970035,
    'fit_intercept': True,
    'class_weight': None,
    'solver': 'liblinear',
    'penalty': 'l1'
}

ridge_params = {
    'random_state': 2100,
    'alpha': 81.85404532128871,
    'tol': 0.005228545098266321,
    'positive': False,
    'fit_intercept': False
}


xgb_model = XGBClassifier(**xgb_params)
xgb_trainer = Trainer(xgb_model)
oof_pred_probs["XGBoost"], test_pred_probs["XGBoost"], overall_scores["XGBoost"], scores["XGBoost"] = xgb_trainer.fit_predict(X, y, X_test, X_original, y_original)


lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_trainer = Trainer(lgbm_model)
oof_pred_probs["LightGBM"], test_pred_probs["LightGBM"], overall_scores["LightGBM"], scores["LightGBM"] = lgbm_trainer.fit_predict(X, y, X_test, X_original, y_original)


lgbm_goss_model = LGBMClassifier(**lgbm_goss_params)
lgbm_goss_trainer = Trainer(lgbm_goss_model)
oof_pred_probs["LightGBM (goss)"], test_pred_probs["LightGBM (goss)"], overall_scores["LightGBM (goss)"], scores["LightGBM (goss)"] = lgbm_goss_trainer.fit_predict(X, y, X_test, X_original, y_original)


cb_model = CatBoostClassifier(**cb_params)
cb_trainer = Trainer(cb_model)
oof_pred_probs["CatBoost"], test_pred_probs["CatBoost"], overall_scores["CatBoost"], scores["CatBoost"] = cb_trainer.fit_predict(X, y, X_test, X_original, y_original)


adb_model = AdaBoostClassifier(**adb_params)
adb_trainer = Trainer(adb_model)
oof_pred_probs["AdaBoost"], test_pred_probs["AdaBoost"], overall_scores["AdaBoost"], scores["AdaBoost"] = adb_trainer.fit_predict(X, y, X_test, X_original, y_original)


rf_model = RandomForestClassifier(**rf_params)
rf_trainer = Trainer(rf_model)
oof_pred_probs["RandomForest"], test_pred_probs["RandomForest"], overall_scores["RandomForest"], scores["RandomForest"] = rf_trainer.fit_predict(X, y, X_test, X_original, y_original)


et_model = ExtraTreesClassifier(**et_params)
et_trainer = Trainer(et_model)
oof_pred_probs["ExtraTrees"], test_pred_probs["ExtraTrees"], overall_scores["ExtraTrees"], scores["ExtraTrees"] = et_trainer.fit_predict(X, y, X_test, X_original, y_original)


histgb_model = HistGradientBoostingClassifier(**histgb_params)
histgb_trainer = Trainer(histgb_model)
oof_pred_probs["HistGradientBoosting"], test_pred_probs["HistGradientBoosting"], overall_scores["HistGradientBoosting"], scores["HistGradientBoosting"] = histgb_trainer.fit_predict(X, y, X_test, X_original, y_original)


gb_model = GradientBoostingClassifier(**gb_params)
gb_trainer = Trainer(gb_model)
oof_pred_probs["GradientBoosting"], test_pred_probs["GradientBoosting"], overall_scores["GradientBoosting"], scores["GradientBoosting"] = gb_trainer.fit_predict(X, y, X_test, X_original, y_original)


lr_model = LogisticRegression(**lr_params)
lr_trainer = Trainer(lr_model)
oof_pred_probs["LogisticRegression"], test_pred_probs["LogisticRegression"], overall_scores["LogisticRegression"], scores["LogisticRegression"] = lr_trainer.fit_predict(X, y, X_test, X_original, y_original)


ridge_model = Ridge(**ridge_params)
ridge_trainer = Trainer(ridge_model)
oof_pred_probs["Ridge"], test_pred_probs["Ridge"], overall_scores["Ridge"], scores["Ridge"] = ridge_trainer.fit_predict(X, y, X_test, X_original, y_original)


tabpfn_model = TabPFNClassifier(random_state=CFG.seed, n_jobs=-1)
tabpfn_trainer = Trainer(tabpfn_model)
oof_pred_probs["TabPFN"], test_pred_probs["TabPFN"], overall_scores["TabPFN"], scores["TabPFN"] = tabpfn_trainer.fit_predict(X, y, X_test, X_original, y_original)


def plot_weights(weights, title):
    sorted_indices = np.argsort(weights[0])[::-1]
    sorted_coeffs = np.array(weights[0])[sorted_indices]
    sorted_model_names = np.array(list(oof_pred_probs.keys()))[sorted_indices]

    plt.figure(figsize=(10, weights.shape[1] * 0.3))
    ax = sns.barplot(x=sorted_coeffs, y=sorted_model_names, palette="RdYlGn_r")

    for i, (value, name) in enumerate(zip(sorted_coeffs, sorted_model_names)):
        if value >= 0:
            ax.text(value, i, f"{value:.3f}", va="center", ha="left", color="black")
        else:
            ax.text(value, i, f"{value:.3f}", va="center", ha="right", color="black")

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1 * abs(xlim[0]), xlim[1] + 0.1 * abs(xlim[1]))

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


X = logit(pd.DataFrame(oof_pred_probs).clip(1e-15, 1-1e-15))
X_test = logit(pd.DataFrame(test_pred_probs).clip(1e-15, 1-1e-15))


def objective(trial):
    solver_penalty_options = [
        ("liblinear", "l1"),
        ("liblinear", "l2"),
        ("lbfgs", "l2"),
        ("lbfgs", None),
        ("newton-cg", "l2"),
        ("newton-cg", None),
        ("newton-cholesky", "l2"),
        ("newton-cholesky", None)
    ]
    solver, penalty = trial.suggest_categorical("solver_penalty", solver_penalty_options)
    
    params = {
        "random_state": CFG.seed,
        "max_iter": 500,
        "C": trial.suggest_float("C", 0, 1),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "solver": solver,
        "penalty": penalty
    }
    
    model = LogisticRegression(**params)
    trainer = Trainer(model, is_ensemble_model=True)
    return trainer.tune(X, y)

if CFG.run_optuna:
    sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True, n_startup_trials=CFG.optuna_trials // 10)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=CFG.optuna_trials, n_jobs=-1)
    best_params = study.best_params
    
    solver, penalty = best_params["solver_penalty"]
    lr_params = {
        "random_state": CFG.seed,
        "max_iter": 500,
        "C": best_params["C"],
        "tol": best_params["tol"],
        "fit_intercept": best_params["fit_intercept"],
        "class_weight": best_params["class_weight"],
        "solver": solver,
        "penalty": penalty
    }
else:
    lr_params = {
      "random_state": 42
    }


print(json.dumps(lr_params, indent=2))


lr_model = LogisticRegression(**lr_params)
lr_trainer = Trainer(lr_model, is_ensemble_model=True)
lr_oof_pred_probs, lr_test_pred_probs, overall_scores["Ensemble LR"], scores["Ensemble LR"], lr_coeffs = lr_trainer.fit_predict(X, y, X_test)


save_submission("ensemble-lr", lr_test_pred_probs, np.mean(scores["Ensemble LR"]))


plot_weights(lr_coeffs, "LR Coefficients")


X = pd.DataFrame(oof_pred_probs)
X_test = pd.DataFrame(test_pred_probs)


def objective(trial):    
    params = {
        "random_state": CFG.seed,
        "alpha": trial.suggest_float("alpha", 0, 10),
        "tol": trial.suggest_float("tol", 1e-7, 1e-2),
        "positive": trial.suggest_categorical("positive", [True, False]),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])
    }
    
    model = Ridge(**params)
    trainer = Trainer(model, is_ensemble_model=True)
    return trainer.tune(X, y)

if CFG.run_optuna:
    sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True, n_startup_trials=CFG.optuna_trials // 10)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=CFG.optuna_trials, n_jobs=-1)
    best_params = study.best_params
    
    ridge_params = {
        "random_state": CFG.seed,
        "alpha": best_params["alpha"],
        "tol": best_params["tol"],
        "positive": best_params["positive"],
        "fit_intercept": best_params["fit_intercept"]
    }
else:
    ridge_params = {
      "random_state": 42
    }


print(json.dumps(ridge_params, indent=2))


ridge_model = Ridge(**ridge_params)
ridge_trainer = Trainer(ridge_model, is_ensemble_model=True)
lr_oof_pred_probs, lr_test_pred_probs, overall_scores["Ensemble Ridge"], scores["Ensemble Ridge"], ridge_coeffs = ridge_trainer.fit_predict(X, y, X_test)


save_submission("ensemble-ridge", lr_test_pred_probs, np.mean(scores["Ensemble Ridge"]))


plot_weights(ridge_coeffs, "Ridge Coefficients")


scores_df = pd.DataFrame(scores)
overall_scores_series = pd.Series({k: v for k, v in overall_scores.items()})
order = overall_scores_series.sort_values(ascending=False).index.tolist()

min_score = overall_scores_series.min()
max_score = overall_scores_series.max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, len(scores_df) * 0.3))

sns.boxplot(data=scores_df, order=order, ax=axs[0], orient="h", palette="RdYlGn_r")
axs[0].set_title("Fold ROC AUC")
axs[0].set_xlabel("")
axs[0].set_ylabel("")

barplot = sns.barplot(x=overall_scores_series, y=overall_scores_series.index, ax=axs[1], palette="RdYlGn_r", order=order)
axs[1].set_title("Overall ROC AUC")
axs[1].set_xlabel("")
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel("")

for i, score in enumerate(overall_scores_series[order]):
    barplot.text(score, i, f"{score:.6f}", va="center")

plt.tight_layout()
plt.show()


shutil.rmtree("catboost_info", ignore_errors=True)

