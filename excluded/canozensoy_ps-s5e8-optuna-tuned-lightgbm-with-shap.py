# Import Libraries
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings("ignore")


# Paths
path = "/kaggle/input/playground-series-s5e8/"
train = pd.read_csv(path + "train.csv", index_col="id")
test  = pd.read_csv(path + "test.csv", index_col="id")
sample_submission = pd.read_csv(path + "sample_submission.csv", index_col="id")


# Target
X = train.drop(columns="y")
y = train.y.copy()
X_test = test.copy()


# === Feature Engineering ===
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])


# Log1p transform for skewed numerical features
log_cols = ["duration", "campaign", "pdays", "previous", "balance"]
for col in log_cols:
    X[col] = np.log1p(X[col])
    X_test[col] = np.log1p(X_test[col])


# === Optuna Tuning ===
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 16, 96),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "n_estimators": 2000,
        "objective": "binary",
        "verbosity": -1
    }

    oof_preds = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc"
        )
        oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]

    return roc_auc_score(y, oof_preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, show_progress_bar=True)
best_params = study.best_trial.params
best_params.update({"objective": "binary", "metric": "auc"})


# === Final Model Training ===
# Prepare arrays for out-of-fold and test predictions
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))

# Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold}...")

    # Split data for this fold
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Define and configure the model
    model = lgb.LGBMClassifier(
        **best_params,
        n_estimators=2000,
        random_state=42
    )
    model.set_params(verbosity=-1)  # Suppress training output

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc"
    )

    # Store out-of-fold predictions and accumulate test predictions
    oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
    preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

    # SHAP analysis for fold 0 only (to avoid slowing down all folds)
    if fold == 0:
        print("Generating SHAP values for fold 0...")
        explainer = shap.TreeExplainer(model)  # Safe for LightGBM
        shap_values = explainer.shap_values(X_valid)

        # Display SHAP summary plot (bar style) for top 15 features
        shap.summary_plot(shap_values, X_valid, plot_type="bar", max_display=15)


# === Final Evaluation and Submission ===
cv_score = roc_auc_score(y, oof)
print(f"Final CV ROC AUC: {cv_score:.6f}")

sample_submission["y"] = preds
sample_submission.to_csv("submission_lgbm_optuna.csv")

