%reset -f

import warnings

warnings.simplefilter("ignore")

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)


data_dir = "equity-post-HCT-survival-predictions"
train = pd.read_csv(f"../input/{data_dir}/train.csv").set_index("ID").sort_index()
train.index = train.index.astype("int32")
test = pd.read_csv(f"../input/{data_dir}/test.csv").set_index("ID").sort_index()
test.index = test.index.astype("int32")


def calc_score(y_pred, detail=False, debug=False, indent=0):
    df = train.iloc[y_pred.index].assign(prediction=y_pred)
    ci_race = []

    for race, i in df.groupby(["race_group"]).groups.items():
        df_race = df.loc[i]
        ci = concordance_index(df_race["efs_time"], -df_race["prediction"], df_race["efs"])
        ci_race.append((ci, race))

    ci_race.sort(reverse=True)
    ci_list = [ci for ci, _ in ci_race]
    mean = np.mean(ci_list)
    std = np.std(ci_list)
    score = float(mean - std)

    if debug:
        print(f"{' '*indent}{score:.4f}: mean={mean:.4f} std={std:.4f}")

        for ci, race in ci_race:
            print(f"{' '*(indent+2)}{ci:.4f} {race}")

    if detail:
        return score, mean, std, ci_race

    return score


def optimize_ci_race(ci_race, fold_n, n_trials=1):
    ci_min = ci_race[-1][0]

    def objective(trial):
        ci_list = []

        for i, (ci, race) in enumerate(ci_race):
            high = ci

            if i > 0 and ci > ci_min:
                high = min(ci, ci_list[-1])

            ci_ = trial.suggest_float(race, ci_min, high)
            ci_list.append(ci_)

        mean = np.mean(ci_list)
        std = np.std(ci_list)
        trial.set_user_attr("mean", mean)
        trial.set_user_attr("std", std)
        return mean - std

    study = optuna.create_study(direction="maximize")

    try:
        with open(f"hct-noise-2-{fold_n}.json") as f:
            study.enqueue_trial((json.load(f)))

    except FileNotFoundError:
        study.enqueue_trial({r: c for c, r in ci_race})

    study.optimize(objective, n_trials=n_trials)
    return study


def preprocess_X(debug=False):
    X = pd.concat([train.drop(columns=["efs", "efs_time"]), test])
    Xi = X.select_dtypes("int").astype("int32")
    Xf = X.select_dtypes("float").astype("float32")
    Xo = X.select_dtypes("object").astype("category")
    cat_features = Xo.columns.to_list()

    for col in Xo:
        Xo[col], _ = Xo[col].factorize(use_na_sentinel=False)
        Xo[col] = Xo[col].astype("int32").astype("category")

    X = pd.concat([Xi, Xf, Xo], axis=1)

    if debug:
        X.info()
        for col in X.select_dtypes("category"):
            print(f"{X[col].cat.categories} {col}")

    X_train = X[: len(train)]
    assert X_train.shape == (28800, 57)
    X_test = X[len(train) :]
    return X_train, X_test, cat_features


def preprocess_y_kms_race(i_fold):
    train_fold = train.iloc[i_fold]
    Y = pd.DataFrame()
    f = KaplanMeierFitter(label="y")

    for race in train_fold["race_group"].unique():
        Y_race = train_fold[train_fold["race_group"] == race]
        f.fit(Y_race["efs_time"], Y_race["efs"])
        Y_race = Y_race.join(f.survival_function_, on="efs_time")
        gap = 0.35 * (Y_race.loc[train_fold["efs"] == 0, "y"].max() - Y_race.loc[train_fold["efs"] == 1, "y"].min())
        Y_race.loc[train_fold["efs"] == 0, "y"] -= gap
        Y = pd.concat([Y, Y_race])

    Y = Y.sort_index()
    return Y["y"]


np.random.seed(42)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
race_efs = pd.Series(str(t) for t in zip(train['race_group'], train['efs']))
m = xgb.XGBRegressor(enable_categorical=True, verbosity=0)
X, _, _ = preprocess_X()
n_trials = 1000


for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train.index, race_efs)):
    print(f"fold {fold_n}")
    y_fold = preprocess_y_kms_race(i_fold)
    m.fit(X.iloc[i_fold], y_fold, verbose=False)
    y_pred = m.predict(X.iloc[i_oof])
    y_pred = pd.Series(y_pred, name="y_pred", index=i_oof)
    score, mean, stddev, ci_race = calc_score(y_pred, detail=True, debug=True, indent=2)
    study = optimize_ci_race(ci_race, fold_n, n_trials=n_trials)
    
    with open(f"hct-noise-2-{fold_n}.json", "w") as f:
        json.dump(study.best_params, f)

    a = study.best_trial.user_attrs
    print(f"\n  {study.best_value:.4f}: mean={a['mean']:.4f} std={a['std']:.4f}")

    for ci, race in [(c, r) for r, c in study.best_params.items()]:
        print(f"    {ci:.4f} {race}")

    print()




