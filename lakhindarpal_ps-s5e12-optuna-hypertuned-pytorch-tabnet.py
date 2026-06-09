!pip install -qU torch pytorch-tabnet


import pandas as pd
import numpy as np
import torch
import optuna

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
og = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")

# train = og[list(set(train.columns) & set(og.columns))]
train = pd.concat([train.drop(columns=["id"]), og], join="inner", ignore_index=True)
train = train.drop_duplicates().reset_index(drop=True)
print("Train data shape", train.shape)


target = "diagnosed_diabetes"
features = train.drop(columns=[target]).columns.tolist()
cat_cols = train.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in features if c not in cat_cols]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le

scaler = RobustScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


X = train[features].values
y = train[target].values
X_test = test[features].values

cat_idxs = [features.index(c) for c in cat_cols]
cat_dims = [len(encoders[c].classes_) for c in cat_cols]

weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
class_weights = dict(zip(np.unique(y), weights_array))
print("Class Weights", class_weights)

# Sample for Optuna
sample_frac = 0.10
sample = train.groupby(target, group_keys=False).apply(lambda x: x.sample(int(len(x)*sample_frac), random_state=42)).sample(frac=1, random_state=42)

X_sample = sample[features].values
y_sample = sample[target].values


def objective(trial):
    params = {
        "n_d": trial.suggest_categorical("n_d", [8, 16, 24, 32]),
        "n_steps": trial.suggest_int("n_steps", 3, 6),
        "gamma": trial.suggest_float("gamma", 1.0, 1.9),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-3, log=True),
        "momentum": trial.suggest_float("momentum", 0.01, 0.4),
        "n_independent": trial.suggest_int("n_independent", 1, 4),
        "n_shared": trial.suggest_int("n_shared", 1, 4),
    }

    params["n_a"] = params["n_d"]
    lr = trial.suggest_float("lr", 1e-4, 2e-2, log=True)
    
    scheduler_params = {
        "step_size": trial.suggest_int("scheduler_step_size", 5, 20),
        "gamma": trial.suggest_float("scheduler_gamma", 0.85, 0.98),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for tr, va in skf.split(X_sample, y_sample):
        clf = TabNetClassifier(
            **params,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=scheduler_params,
            mask_type="entmax",
            clip_value=2.0,
            seed=42,
            verbose=0
        )

        clf.fit(
            X_sample[tr], y_sample[tr],
            eval_set=[(X_sample[va], y_sample[va])],
            eval_metric=["auc"],
            max_epochs=150,
            patience=25,
            batch_size=2048,
            virtual_batch_size=128,
            weights=class_weights
        )

        scores.append(roc_auc_score(y_sample[va], clf.predict_proba(X_sample[va])[:,1]))

        trial.report(scores[-1], len(scores))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores)


# study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
# study.optimize(objective, n_trials=40, show_progress_bar=True)

# best_params = study.best_params
# print("**Best Params**\n", best_params)

## reusing optuna param from previous run 
best_params = {
    "n_d": 16,
    "n_steps": 6,
    "gamma": 1.2862031274746775,
    "lambda_sparse": 1.6599837974449204e-05,
    "momentum": 0.09889471339135725,
    "n_independent": 2,
    "n_shared": 4,
    "lr": 0.009562399642465934,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.9163971493350835,
}

best_params["n_a"] = best_params["n_d"]
best_lr = best_params.pop("lr")
scheduler_params = {
    "step_size": best_params.pop("scheduler_step_size"), 
    "gamma": best_params.pop("scheduler_gamma")
}


oof = np.zeros_like(y, dtype=float)
test_preds = []
scores = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i,(tr,va) in enumerate(kf.split(X,y)):
    print(f"===== Fold {i+1} =====")

    clf = TabNetClassifier(
        **best_params,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": best_lr},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=scheduler_params,
        mask_type="entmax",
        clip_value=2.0,
        seed=42
    )

    clf.fit(
        X[tr], y[tr],
        eval_set=[(X[va],y[va])],
        eval_metric=["auc"],
        max_epochs=200,
        patience=40,
        batch_size=2048,
        virtual_batch_size=128,
        weights=class_weights
    )

    oof[va] = clf.predict_proba(X[va])[:,1]
    scores.append(roc_auc_score(y[va], oof[va]))
    test_preds.append(clf.predict_proba(X_test)[:,1])
    
    clf.save_model(f'tabnet_fold_{i}.zip')


print("OOF:", roc_auc_score(y,oof))
print("Mean:", np.mean(scores))
print("Std:", np.std(scores))


test_preds = np.column_stack(test_preds)
weights = np.array(scores) / np.sum(scores)
final_pred = (test_preds * weights).sum(axis=1)

np.save("tabnet_oof.npy", oof)
np.save("tabnet_pred.npy", final_pred)

submission = pd.DataFrame({"id": test["id"], target: final_pred})
submission.to_csv("submission.csv", index=False)

print("Done.")


pd.read_csv("/kaggle/working/submission.csv")

