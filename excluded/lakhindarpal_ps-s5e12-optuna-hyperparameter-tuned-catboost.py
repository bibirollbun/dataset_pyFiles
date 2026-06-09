!pip install -qU scikit-learn optuna catboost


import numpy as np
import pandas as pd

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

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
        "iterations": trial.suggest_int("iterations", 3000, 6000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "depth": trial.suggest_int("depth", 5, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5, 40, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 1.2),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 128),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "feature_border_type": trial.suggest_categorical(
            "feature_border_type",
            ["GreedyLogSum", "Median", "Uniform"]
        ),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", pos_weight*0.5, pos_weight*1.5),
        "random_strength": trial.suggest_float("random_strength", 0.5, 3.0),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
        "one_hot_max_size": 32,
        "max_ctr_complexity": 4,
        "random_state": 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
        X_train, y_train = X_sample.iloc[train_idx], y_sample.iloc[train_idx]
        X_val, y_val = X_sample.iloc[val_idx], y_sample.iloc[val_idx]
        
        model = CatBoostClassifier(
            **params,
            task_type="GPU",
            loss_function="Logloss",
            eval_metric="AUC",
            bootstrap_type="Bayesian",
            leaf_estimation_method="Newton",
            cat_features=cat_cols,
            early_stopping_rounds=200,
            verbose=False
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)

        trial.report(score, fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores)


# study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
# study.optimize(objective, n_trials=50, show_progress_bar=True)

# best_params = study.best_params
# print(best_params)

# best_params.update({"random_state": 42})

# joblib.dump(study, "catboost_optuna.pkl")

# [Not tuning again, just reusing params]
best_params = {
    "iterations": 4332,
    "learning_rate": 0.012464639977101061,
    "depth": 5,
    "l2_leaf_reg": 8.315216687232203,
    "bagging_temperature": 0.4507484772210721,
    "grow_policy": "Lossguide",
    "min_data_in_leaf": 56,
    "border_count": 224,
    "feature_border_type": "GreedyLogSum",
    "scale_pos_weight": 1.8667107264620153,
    "random_strength": 2.8288687124001317,
    "leaf_estimation_iterations": 10,
    "random_state": 42,
}
best_params.update({
    "random_state": 42,
    "iterations": int(best_params["iterations"] * 1.5),
    "learning_rate": best_params["learning_rate"] * 0.7
})


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = []
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        **best_params,
        task_type="GPU",
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bayesian",
        leaf_estimation_method="Newton",
        cat_features=cat_cols,
        early_stopping_rounds=200,
        verbose=200
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    preds = model.predict_proba(X_val)[:, 1]
    test_preds.append(model.predict_proba(X_test)[:, 1])

    oof[val_idx] = preds
    score = roc_auc_score(y_val, preds)
    scores.append(score)
    joblib.dump(model, f"cat_fold_{fold}.pkl")


print("OOF:", roc_auc_score(y, oof))
print("Mean:", np.mean(scores), "Std:", np.std(scores))


test_preds = np.column_stack(test_preds)
weights = np.array(scores) / np.sum(scores)
final_pred = (test_preds * weights).sum(axis=1)

np.save("cat_oof.npy", oof)
np.save("cat_pred.npy", final_pred)

submission = pd.DataFrame({"id": test["id"], target: final_pred})
submission.to_csv("submission.csv", index=False)

print("Done.")

