"""
capstone_agent.py
How to use:
1. Put train.csv and test.csv in the working directory (Kaggle notebook).
   - If these files are missing, the script generates a synthetic dataset automatically.
2. Add your OpenAI API key below (or export OPENAI_API_KEY as env var).
3. Run the script. It will:
   - load or generate datasets
   - preprocess and engineer features
   - run Optuna to tune XGBoost
   - train XGBoost and a small Keras MLP
   - blend predictions
   - compute CV metrics and produce SHAP explanations
   - save "submission.csv" to current dir

Dependencies:
pandas, numpy, scikit-learn, xgboost, optuna, shap, tensorflow (or keras), openai
The script will attempt to pip-install missing packages automatically (Kaggle usually provides most).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Auto-install required packages (only if missing)
# ----------------------------
required = [
    "pandas", "numpy", "scikit-learn", "xgboost", "optuna", "shap", "tensorflow", "openai"
]
import importlib
to_install = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception:
        to_install.append(pkg)

if to_install:
    print("Installing missing packages:", to_install)
    # Use pip programmatically
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])

# Now import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import optuna
import shap
import json
import time
import joblib

# OpenAI import optional (will only be used if key provided)
try:
    import openai
except Exception:
    openai = None

# ----------------------------
# Configuration - add your OpenAI key manually here if you want reasoning/reports
# ----------------------------
# Option A: set as environment variable before running: export OPENAI_API_KEY="sk-..."
# Option B: paste your key below (replace None with "sk-..."), but it's safer to set env var.
OPENAI_API_KEY = None  # <-- Replace with your key string if you want the OpenAI features.
# Example (not recommended in shared repos): OPENAI_API_KEY = "sk-REPLACE_THIS_WITH_YOURS"

if OPENAI_API_KEY:
    if openai is None:
        print("OpenAI library not available; skipping OpenAI features.")
    else:
        openai.api_key = OPENAI_API_KEY
else:
    # fallback to environment variable if provided by runner
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    if OPENAI_API_KEY and openai is not None:
        openai.api_key = OPENAI_API_KEY

# ----------------------------
# Utility functions
# ----------------------------
def seed_everything(seed=42):
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

seed_everything(42)

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

# ----------------------------
# Data loading / generation
# ----------------------------
def load_or_generate_data():
    """
    If train.csv and test.csv exist in working dir, load them.
    Else generate a synthetic classification dataset (binary) and save as train.csv/test.csv
    """
    train_path = "train.csv"
    test_path = "test.csv"

    train_df = safe_read_csv(train_path)
    test_df = safe_read_csv(test_path)

    if train_df is not None and test_df is not None:
        print("Found train.csv and test.csv - using them.")
        return train_df, test_df

    # Generate synthetic data
    print("train.csv/test.csv not found. Generating synthetic dataset...")
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, n_features=25, n_informative=10, n_redundant=5,
        n_classes=2, flip_y=0.02, class_sep=1.0, random_state=42
    )
    feature_names = [f"f_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Add a few categorical columns
    df["cat_1"] = np.random.choice(["A", "B", "C"], size=len(df))
    df["cat_2"] = np.random.choice(["X", "Y"], size=len(df))

    # Create train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)
    train_df.to_csv("train.csv", index=False)
    # For test.csv, remove target column to mimic Kaggle test
    test_df_no_target = test_df.drop(columns=["target"]).copy()
    test_df_no_target.to_csv("test.csv", index=False)
    print("Synthetic train.csv and test.csv saved.")
    return train_df.reset_index(drop=True), test_df_no_target.reset_index(drop=True)

# ----------------------------
# Preprocessing pipeline
# ----------------------------
def build_preprocessing_pipeline(df, id_column=None, target_column="target"):
    """
    Detect numeric and categorical columns and return a ColumnTransformer pipeline.
    """
    if target_column in df.columns:
        features = df.drop(columns=[target_column])
    else:
        features = df.copy()

    if id_column and id_column in features.columns:
        features = features.drop(columns=[id_column])

    numeric_cols = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transform, numeric_cols),
            ("cat", categorical_transform, categorical_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }

    return preprocessor, meta

# ----------------------------
# Model: XGBoost + MLP + blending
# ----------------------------
def objective_xgb(trial, X, y, cv_splits=3, random_state=42):
    params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": 500,
        "random_state": random_state
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=25, verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)

def tune_xgb(X, y, n_trials=20):
    print("Starting Optuna tuning for XGBoost...")
    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective_xgb(trial, X, y)
    study.optimize(func, n_trials=n_trials, show_progress_bar=True)
    print("Best trial:", study.best_trial.params)
    return study.best_trial.params

def train_xgb_final(X, y, best_params):
    # set sensible training parameters
    params = best_params.copy()
    params.update({
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 1000,
        "use_label_encoder": False
    })
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, early_stopping_rounds=30, eval_set=[(X, y)], verbose=False)
    return model

def train_mlp(X_train, y_train, X_val=None, y_val=None):
    print("Training a small MLP as second model...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                        solver="adam", learning_rate_init=1e-3, max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp

# ----------------------------
# SHAP explanation
# ----------------------------
def explain_with_shap(model, X_sample, feature_names, n_top=10):
    print("Computing SHAP explanations (may take a moment)...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    # top features by mean abs shap
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-n_top:][::-1]
    important = [(feature_names[i], mean_abs_shap[i]) for i in top_idx]
    return important, shap_values

# ----------------------------
# OpenAI reasoning / report (optional)
# ----------------------------
def openai_report(summary_dict, openai_api_key=None):
    """
    Use OpenAI to generate a polished report summarizing the run.
    If no key provided, return a plain-text summary.
    """
    text_summary = json.dumps(summary_dict, indent=2)
    if openai_api_key is None or openai is None:
        print("OpenAI key or library not available - returning text summary.")
        return "SUMMARY (no OpenAI):\n" + text_summary

    try:
        openai.api_key = openai_api_key
        prompt = (
            "You are an expert ML engineer and data scientist. "
            "Summarize the following training run results into a clear report with: "
            "1) short executive summary, 2) data issues found, 3) model performance & suggestions, "
            "4) next steps for improvement.\n\n"
            f"Run data:\n{text_summary}\n\nReport:"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # if unavailable change to "gpt-4o" or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.0
        )
        content = response["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print("OpenAI call failed:", e)
        return "SUMMARY (text only due to OpenAI failure):\n" + text_summary

# ----------------------------
# Full agent
# ----------------------------
class CapstoneAgent:
    def __init__(self, train_df, test_df, id_column=None, target_column="target"):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.id_column = id_column
        self.target_column = target_column
        self.preprocessor = None
        self.meta = None
        self.models = {}
        self.feature_names = None

    def analyze_data(self):
        print("=== Data analysis ===")
        df = self.train_df
        print("Rows:", len(df))
        print("Columns:", df.shape[1])
        print("Sample target distribution:")
        if self.target_column in df:
            print(df[self.target_column].value_counts(normalize=True))
        # simple missingness
        miss = df.isna().mean()
        missing_cols = miss[miss > 0]
        if not missing_cols.empty:
            print("Columns with missingness (>0):")
            print(missing_cols.sort_values(ascending=False).head(10))
        else:
            print("No missing values detected in training set.")

    def prepare(self):
        print("=== Preparing preprocessing pipeline ===")
        self.preprocessor, self.meta = build_preprocessing_pipeline(self.train_df, id_column=self.id_column, target_column=self.target_column)
        # Fit preprocessor
        X = self.train_df.drop(columns=[self.target_column])
        if self.id_column and self.id_column in X.columns:
            X = X.drop(columns=[self.id_column])
        self.preprocessor.fit(X)
        # get feature names after transformation (sklearn 1.0+)
        try:
            self.feature_names = self.preprocessor.get_feature_names_out()
        except Exception:
            # fallback name generation
            num = len(self.meta["numeric_cols"])
            cat_ohe_cols = []
            for cat in self.meta["categorical_cols"]:
                cat_ohe_cols.append(cat)
            self.feature_names = np.array(self.meta["numeric_cols"] + cat_ohe_cols)
        print("Prepared. Feature count after preprocessing:", len(self.feature_names))

    def vectorize(self, df, is_train=True):
        X = df.copy()
        if is_train and self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
        if self.id_column and self.id_column in X.columns:
            X = X.drop(columns=[self.id_column])
        Xt = self.preprocessor.transform(X)
        return Xt

    def tune_and_train(self, n_trials=20):
        print("=== Tuning and training ===")
        X = self.vectorize(self.train_df, is_train=True)
        y = self.train_df[self.target_column].values
        best_params = tune_xgb(X, y, n_trials=n_trials)
        model_xgb = train_xgb_final(X, y, best_params)
        self.models['xgb'] = model_xgb

        # Small MLP trained on same vectors
        mlp = train_mlp(X, y)
        self.models['mlp'] = mlp

        # Save models
        joblib.dump(self.models['xgb'], "model_xgb.joblib")
        joblib.dump(self.models['mlp'], "model_mlp.joblib")
        print("Models saved: model_xgb.joblib, model_mlp.joblib")

    def cross_val_blend(self, folds=5):
        print("=== Cross-validated blend evaluation ===")
        X = self.vectorize(self.train_df, is_train=True)
        y = self.train_df[self.target_column].values
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y))
        fold_scores = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            # Retrain per fold for reliable OOF
            xgb_clf = xgb.XGBClassifier(**self.models['xgb'].get_params())
            xgb_clf.fit(X_tr, y_tr, early_stopping_rounds=30, eval_set=[(X_val, y_val)], verbose=False)
            mlp_clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, random_state=42)
            mlp_clf.fit(X_tr, y_tr)
            pred_x = xgb_clf.predict_proba(X_val)[:,1]
            pred_m = mlp_clf.predict_proba(X_val)[:,1]
            blended = 0.75*pred_x + 0.25*pred_m
            oof_preds[val_idx] = blended
            score = roc_auc_score(y_val, blended)
            fold_scores.append(score)
            print(f"Fold {fold} AUC: {score:.4f}")

        overall_auc = roc_auc_score(y, oof_preds)
        print(f"OOF blended AUC: {overall_auc:.4f} (mean folds: {np.mean(fold_scores):.4f})")
        return overall_auc, fold_scores

    def fit_on_full(self):
        print("=== Fitting final models on full training data ===")
        X = self.vectorize(self.train_df, is_train=True)
        y = self.train_df[self.target_column].values
        # XGBoost final
        xgb_final = xgb.XGBClassifier(**self.models['xgb'].get_params())
        xgb_final.fit(X, y, early_stopping_rounds=30, eval_set=[(X,y)], verbose=False)
        self.models['xgb_final'] = xgb_final
        # MLP final
        mlp_final = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42)
        mlp_final.fit(X, y)
        self.models['mlp_final'] = mlp_final
        joblib.dump(self.models['xgb_final'], "model_xgb_final.joblib")
        joblib.dump(self.models['mlp_final'], "model_mlp_final.joblib")
        print("Final models saved.")

    def predict_test_and_save(self, submission_filename="submission.csv"):
        print("=== Predicting test set and saving submission ===")
        test = self.test_df.copy()
        ids = None
        if self.id_column and self.id_column in test.columns:
            ids = test[self.id_column].values
        X_test = self.vectorize(test, is_train=False)
        pred_x = self.models['xgb_final'].predict_proba(X_test)[:,1]
        pred_m = self.models['mlp_final'].predict_proba(X_test)[:,1]
        blended = 0.75*pred_x + 0.25*pred_m
        # Create submission DataFrame
        if ids is not None:
            submission = pd.DataFrame({self.id_column: ids, "target": blended})
        else:
            submission = pd.DataFrame({"id": np.arange(len(blended)), "target": blended})
        submission.to_csv(submission_filename, index=False)
        print(f"Submission saved to {submission_filename}")
        return submission

    def explain_models(self, n_samples=200):
        print("=== Running SHAP explanation ===")
        X_full = self.vectorize(self.train_df, is_train=True)
        # sample small portion for SHAP
        if X_full.shape[0] > n_samples:
            idx = np.random.choice(X_full.shape[0], n_samples, replace=False)
            Xs = X_full[idx]
        else:
            Xs = X_full
        model = self.models['xgb_final']
        important, shap_values = explain_with_shap(model, Xs, list(self.feature_names))
        print("Top important features (by mean |SHAP|):")
        for name, score in important:
            print(f"  {name}: {score:.5f}")
        return important

    def run_openai_report(self, run_summary):
        print("=== Generating OpenAI report (optional) ===")
        if OPENAI_API_KEY is None:
            print("No OpenAI key provided; skipping.")
            return None
        report = openai_report(run_summary, openai_api_key=OPENAI_API_KEY)
        # Save report to file
        with open("run_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("OpenAI report saved to run_report.txt")
        return report

# ----------------------------
# Main orchestration
# ----------------------------
def main():
    print("=== Starting Capstone AI Agent ===")
    train_df, test_df = load_or_generate_data()
    agent = CapstoneAgent(train_df, test_df, id_column=None, target_column="target")

    start_time = time.time()
    agent.analyze_data()
    agent.prepare()

    # Quick vectorization test
    X_vec = agent.vectorize(train_df, is_train=True)
    print("Vectorized training matrix shape:", X_vec.shape)

    # Tune & train (keep n_trials small for Kaggle runtime; set higher for final runs)
    agent.tune_and_train(n_trials=20)

    # Cross-validated blending evaluation
    auc, fold_scores = agent.cross_val_blend(folds=5)

    # Fit final models on full training data
    agent.fit_on_full()

    # Create submission
    submission = agent.predict_test_and_save(submission_filename="submission.csv")

    # SHAP explanation
    important_feats = agent.explain_models(n_samples=300)

    # Compose run summary
    run_summary = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "feature_count_post_preprocessing": len(agent.feature_names) if agent.feature_names is not None else None,
        "oof_auc_blended": float(auc),
        "fold_scores": [float(x) for x in fold_scores],
        "top_features": [name for name, _ in important_feats]
    }

    # Optional OpenAI report
    report = None
    if OPENAI_API_KEY:
        report = agent.run_openai_report(run_summary)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Done. Elapsed time: {elapsed/60:.2f} minutes")
    print("Artifacts produced: submission.csv, model_xgb.joblib, model_xgb_final.joblib, model_mlp.joblib, model_mlp_final.joblib, run_report.txt (if OpenAI used)")

    # Print short summary for immediate feedback
    print("\n=== Run summary ===")
    print(json.dumps(run_summary, indent=2))
    if report:
        print("\nOpenAI Report Preview:\n")
        print(report[:1000])  # print first 1000 chars

if __name__ == "__main__":
    main()





