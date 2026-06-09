!pip install -qU scikit-learn


import numpy as np
import pandas as pd

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation

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


sample_frac = 0.1
sample = train.groupby(target, group_keys=False).apply(
    lambda x: x.sample(int(len(x) * sample_frac), random_state=42)
).sample(frac=1, random_state=42).reset_index(drop=True)

X_sample = sample.drop(columns=[target])
y_sample = sample[target]

X = train.drop(columns=[target])
y = train[target]
X_test = test.drop(columns=["id"])

for col in cat_cols:
    X_sample[col] = X_sample[col].astype("category")
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

pos_weight = y.value_counts()[1] / y.value_counts()[0]


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 16, 48),
        "max_depth": trial.suggest_int("max_depth", -1, 6),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 500, 1500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.85),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.85),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
        "max_bin": trial.suggest_int("max_bin", 128, 255),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.5),
        "n_estimators": trial.suggest_int("n_estimators", 2000, 5000),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
        X_train, y_train = X_sample.iloc[train_idx], y_sample.iloc[train_idx]
        X_val, y_val = X_sample.iloc[val_idx], y_sample.iloc[val_idx]
        
        model = LGBMClassifier(
            **params,
            boosting_type="gbdt",
            device="gpu",
            objective="binary",
            metric="auc",
            scale_pos_weight=pos_weight,
            random_state=42,
            verbose=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(100),
                log_evaluation(100),
            ],
            categorical_feature=cat_cols,
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

joblib.dump(study, "lightgbm_optuna.pkl")

best_params = study.best_params
print(best_params)

best_params.update({
    "n_estimators": int(best_params["n_estimators"] * 2.5),
    "learning_rate": best_params["learning_rate"] * 0.7
})


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = []
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        **best_params,
        boosting_type="gbdt",
        device="gpu",
        objective="binary",
        metric="auc",
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(100),
            log_evaluation(100),
        ],
        categorical_feature=cat_cols,
    )

    preds = model.predict_proba(X_val)[:, 1]
    test_preds.append(model.predict_proba(X_test)[:, 1])

    oof[val_idx] = preds
    score = roc_auc_score(y_val, preds)
    scores.append(score)
    joblib.dump(model, f"lgbm_fold_{fold}.pkl")


print("OOF:", roc_auc_score(y, oof))
print("Mean:", np.mean(scores))
print("Std:", np.std(scores))


test_preds = np.column_stack(test_preds)
weights = np.array(scores) / np.sum(scores)
final_pred = (test_preds * weights).sum(axis=1)

np.save("lgbm_oof.npy", oof)
np.save("lgbm_pred.npy", final_pred)

submission = pd.DataFrame({"id": test["id"], target: final_pred})
submission.to_csv("submission.csv", index=False)

print("Done.")




