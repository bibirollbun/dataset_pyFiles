!pip install -qU scikit-learn


import numpy as np
import pandas as pd

import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import joblib
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
og = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")

train = pd.concat([train.drop(columns=["id"]), og], join="inner", ignore_index=True)
train = train.drop_duplicates().reset_index(drop=True)
print("Train data shape", train.shape)

target = "diagnosed_diabetes"
cat_cols = train.select_dtypes(include="object").columns.tolist()


sample = og[train.columns]

X_sample = sample.drop(columns=[target])
y_sample = sample[target]

X = train.drop(columns=[target])
y = train[target]
X_test = test.drop(columns=["id"])

for col in cat_cols:
    X_sample[col] = X_sample[col].astype("category")
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03),
        "max_iter": trial.suggest_int("max_iter", 600, 1200),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 32, 96),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 0.3),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
        X_train, y_train = X_sample.iloc[train_idx], y_sample.iloc[train_idx]
        X_val, y_val = X_sample.iloc[val_idx], y_sample.iloc[val_idx]
        
        model = HistGradientBoostingClassifier(
            **params,
            early_stopping=True,
            scoring="roc_auc",
            n_iter_no_change=15,
            random_state=42,
            verbose=1
        )

        model.fit(
            X_train, y_train,
            X_val=X_val,
            y_val=y_val,
        )

        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)

        trial.report(score, fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores)


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

joblib.dump(study, "hgb_optuna.pkl")

best_params = study.best_params
print("**Best Params:**\n", best_params)

best_params.update({
    "max_iter": int(best_params["max_iter"] * 2),
    "learning_rate": best_params["learning_rate"] * 0.95
})


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = []
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"===== Fold {fold} =====")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = HistGradientBoostingClassifier(
         **best_params,
        early_stopping=True,
        scoring="roc_auc",
        n_iter_no_change=25,
        random_state=42,
        verbose=1
    )

    model.fit(
        X_train, y_train,
        X_val=X_val,
        y_val=y_val,
    )

    preds = model.predict_proba(X_val)[:, 1]
    test_preds.append(model.predict_proba(X_test)[:, 1])

    oof[val_idx] = preds
    score = roc_auc_score(y_val, preds)
    scores.append(score)
    joblib.dump(model, f"hgb_fold_{fold}.pkl")



# Final Scoring
print("OOF:", roc_auc_score(y, oof))
print("Mean:", np.mean(scores))
print("Std:", np.std(scores))


test_preds = np.column_stack(test_preds)
weights = np.array(scores) / np.sum(scores)
final_pred = (test_preds * weights).sum(axis=1)

np.save("hgb_oof.npy", oof)
np.save("hgb_pred.npy", final_pred)

submission = pd.DataFrame({"id": test["id"], target: final_pred})
submission.to_csv("submission.csv", index=False)

print("Done.")




