import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools


import warnings

warnings.filterwarnings(
    "ignore",
    message=".*is_categorical_dtype is deprecated.*",
    category=DeprecationWarning,
    module="pandas"
)

warnings.filterwarnings(
    "ignore",
    message=".*DataFrame is highly fragmented.*",
    category=pd.errors.PerformanceWarning
)


def preprocess_dataframe(
    df: pd.DataFrame,
    categorical_encoding: str = 'ordinal',
    handle_null_numerical: str = 'median',
    handle_null_categorical: str = 'new_category'
) -> pd.DataFrame:
    processed_df = df.copy()

    numerical_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()

    if handle_null_numerical == 'mean':
        for col in numerical_cols:
            processed_df[col].fillna(processed_df[col].mean(), inplace=True)
    elif handle_null_numerical == 'median':
        for col in numerical_cols:
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
    elif handle_null_numerical == '-1':
        for col in numerical_cols:
            processed_df[col].fillna(-1, inplace=True)
    elif handle_null_numerical == 'drop':
        processed_df.dropna(subset=numerical_cols, inplace=True)
    else:
        raise ValueError("Invalid option for handle_null_numerical. Choose from 'mean', 'median', '-1', 'drop'.")

    if handle_null_categorical == 'new_category':
        for col in categorical_cols:
            if pd.api.types.is_categorical_dtype(processed_df[col]):
                if 'Missing' not in processed_df[col].cat.categories:
                    processed_df[col] = processed_df[col].cat.add_categories('Missing')
            processed_df[col].fillna('Missing', inplace=True)
    elif handle_null_categorical == 'drop':
        processed_df.dropna(subset=categorical_cols, inplace=True)
    else:
        raise ValueError("Invalid option for handle_null_categorical. Choose from 'new_category', 'drop'.")

    if numerical_cols:
        scaler = MinMaxScaler()
        processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])

    if categorical_encoding == 'ordinal':
        for col in categorical_cols:
            processed_df[col] = pd.factorize(processed_df[col])[0]
    elif categorical_encoding == 'one-hot':
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    else:
        raise ValueError("Invalid option for categorical_encoding. Choose from 'ordinal', 'one-hot'.")

    return processed_df

def numerical_to_categorical(
    df: pd.DataFrame,
    bins: int | dict,
    labels: list = None,
    modify_original: bool = False
) -> pd.DataFrame:
    binned_df = df.copy()
    numerical_cols = binned_df.select_dtypes(include=np.number).columns.tolist()

    for col in numerical_cols:
        num_bins = bins if isinstance(bins, int) else bins.get(col)

        if num_bins:
            col_bins = pd.cut(binned_df[col], bins=num_bins, include_lowest=True, retbins=True)
            
            bin_edges = col_bins[1]
            bin_labels = labels

            if not labels:
                bin_labels = []
                for i in range(len(bin_edges) - 1):
                    lower = round(bin_edges[i], 2)
                    upper = round(bin_edges[i + 1], 2)
                    bin_labels.append((lower+upper)/2)

            target_col = col if modify_original else f"{col}_binned_{num_bins}"
            binned_df[target_col] = pd.cut(
                binned_df[col],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True
            )

    return binned_df


def add_cat_combinations(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    max_level: int | None = None,
    sep: str = "_",
    drop_original: bool = False,
    inplace: bool = True,
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    if cat_cols is None:
        cat_cols = [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == object]
    if len(cat_cols) < 2:
        raise ValueError("Need at least two categorical columns to combine")
    max_level = max_level or len(cat_cols)
    max_level = min(max_level, len(cat_cols))
    for k in range(2, max_level + 1):
        for cols in itertools.combinations(cat_cols, k):
            new_name = sep.join(cols)
            df[new_name] = (
                df[list(cols)]
                .astype(str)
                .apply(sep.join, axis=1)
                .replace({sep.join(["nan"] * k): np.nan})
            )
    if drop_original:
        df.drop(columns=cat_cols, inplace=True)
    return df

def add_num_transforms(
    df: pd.DataFrame,
    num_cols: list[str] | None = None,
    *,
    log_offset: float = 1e-9,
    inplace: bool = True
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    if num_cols is None:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        x = df[col].astype(float)
        df[f"{col}_log"] = np.log(x + log_offset)
        df[f"{col}_sin"] = np.sin(x)
        df[f"{col}_cos"] = np.cos(x)
    return df



train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv").set_index("id")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv").set_index("id")


X, y = train_df.drop("Personality", axis=1), train_df["Personality"].astype('category').cat.codes


X = add_cat_combinations(X)
test_df = add_cat_combinations(test_df)
X = add_num_transforms(X)
test_df = add_num_transforms(test_df)


X = preprocess_dataframe(X)
test_df = preprocess_dataframe(test_df)


for i in range(3, 5):
    X = numerical_to_categorical(X, i)
    test_df = numerical_to_categorical(test_df, i)


import numpy as np, optuna, warnings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
warnings.filterwarnings("ignore")


def train_and_create_voting_ensemble_optuna(
    X, y, test_size=0.30, random_state=42, n_trials=32
):

    warnings.filterwarnings("ignore")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    def build_clf(name, p):
        if name == "LogReg":
            return LogisticRegression(
                C=p["C"], solver="liblinear", class_weight="balanced",
                max_iter=1000, random_state=random_state
            )
        if name == "SGD":
            return SGDClassifier(
                loss="log_loss", alpha=p["alpha"], class_weight="balanced",
                max_iter=1000, random_state=random_state
            )
        if name == "RF":
            return RandomForestClassifier(
                n_estimators=p["n_est"], max_depth=p["depth"],
                class_weight="balanced", random_state=random_state
            )
        if name == "ExtraTrees":
            return ExtraTreesClassifier(
                n_estimators=p["n_est"], max_depth=p["depth"],
                class_weight="balanced", random_state=random_state
            )
        if name == "GBoost":
            return GradientBoostingClassifier(
                n_estimators=p["n_est"], learning_rate=p["lr"],
                max_depth=p["depth"], random_state=random_state
            )
        if name == "AdaBoost":
            return AdaBoostClassifier(
                n_estimators=p["n_est"], learning_rate=p["lr"],
                random_state=random_state
            )
        if name == "SVC":
            return SVC(
                C=p["C"], gamma=p["gamma"], probability=True,
                class_weight="balanced", random_state=random_state
            )
        if name == "KNN":
            return KNeighborsClassifier(
                n_neighbors=p["k_nn"], weights="distance"
            )
        if name == "XGB":
            return XGBClassifier(
                n_estimators=p["n_est"], max_depth=p["depth"],
                learning_rate=p["lr"], subsample=p["sub"],
                colsample_bytree=p["col"], scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", use_label_encoder=False,
                random_state=random_state
            )
        if name == "LightGBM":
            return lgb.LGBMClassifier(
                n_estimators=p["n_est"], learning_rate=p["lr"],
                num_leaves=p["leaves"], scale_pos_weight=scale_pos_weight,
                random_state=random_state
            )

    def suggest_params(trial, name):
        if name == "LogReg":
            return dict(C=trial.suggest_float("C", 1e-2, 100, log=True))
        if name == "SGD":
            return dict(alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True))
        if name in ["RF", "ExtraTrees"]:
            return dict(
                n_est=trial.suggest_int("n_est", 200, 3000),
                depth=trial.suggest_int("depth", 3, 20)
            )
        if name == "GBoost":
            return dict(
                n_est=trial.suggest_int("n_est", 100, 1000),
                lr=trial.suggest_float("lr", 0.01, 0.3, log=True),
                depth=trial.suggest_int("depth", 1, 5)
            )
        if name == "AdaBoost":
            return dict(
                n_est=trial.suggest_int("n_est", 50, 800),
                lr=trial.suggest_float("lr", 0.01, 1.0, log=True)
            )
        if name == "SVC":
            return dict(
                C=trial.suggest_float("C", 1e-2, 100, log=True),
                gamma=trial.suggest_float("gamma", 1e-3, 1.0, log=True)
            )
        if name == "KNN":
            return dict(k_nn=trial.suggest_int("k_nn", 3, 25, step=2))
        if name == "XGB":
            return dict(
                n_est=trial.suggest_int("n_est", 200, 2000),
                depth=trial.suggest_int("depth", 3, 12),
                lr=trial.suggest_float("lr", 0.01, 0.3, log=True),
                sub=trial.suggest_float("sub", 0.5, 1.0),
                col=trial.suggest_float("col", 0.5, 1.0)
            )
        if name == "LightGBM":
            return dict(
                n_est=trial.suggest_int("n_est", 200, 1500),
                lr=trial.suggest_float("lr", 0.01, 0.3, log=True),
                leaves=trial.suggest_int("leaves", 20, 300)
            )

    def objective(trial, name):
        k = trial.suggest_int("k", 10, X_tr.shape[1])
        p = suggest_params(trial, name)
        clf = build_clf(name, p)
        pipe = Pipeline([
            ("sel", SelectKBest(score_func=f_classif, k=k)),
            ("clf", clf)
        ])
        pipe.fit(X_tr, y_tr)
        return accuracy_score(y_te, pipe.predict(X_te))

    model_names = [
        "RF", "ExtraTrees", "GBoost", "AdaBoost", "XGB", "LightGBM",
        "LogReg", "SVC", "SGD", "KNN"
    ]

    individual, tuned_pipes = {}, []
    for name in model_names:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t, n=name: objective(t, n),
            n_trials=n_trials, n_jobs=-1
        )
        params = study.best_params
        k_best = params.pop("k")
        clf = build_clf(name, params)
        best_pipe = Pipeline([
            ("sel", SelectKBest(score_func=f_classif, k=k_best)),
            ("clf", clf)
        ])
        best_pipe.fit(X_tr, y_tr)
        pred = best_pipe.predict(X_te)
        proba = best_pipe.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, pred)
        individual[name] = {
            "best_params": {**params, "k": k_best},
            "accuracy": acc,
            "roc_auc": roc_auc_score(y_te, proba),
            "report": classification_report(y_te, pred)
        }
        tuned_pipes.append((name, best_pipe))

    accs = np.array([m["accuracy"] for m in individual.values()])
    median = float(np.median(accs))
    chosen = [
        (n, p) for n, p in tuned_pipes
        if individual[n]["accuracy"] >= median
    ]
    weights = [individual[n]["accuracy"] for n, _ in chosen]
    ensemble = VotingClassifier(
        estimators=chosen, voting="soft", weights=weights, n_jobs=-1
    )
    ensemble.fit(X_tr, y_tr)
    ens_pred = ensemble.predict(X_te)
    ens_proba = ensemble.predict_proba(X_te)[:, 1]
    ensemble_report = {
        "accuracy": accuracy_score(y_te, ens_pred),
        "roc_auc": roc_auc_score(y_te, ens_proba),
        "confusion_matrix": confusion_matrix(y_te, ens_pred),
        "classification_report": classification_report(y_te, ens_pred)
    }
    return ensemble, ensemble_report, individual



result = train_and_create_voting_ensemble_optuna(X, y)


pipeline, r1, r2 = result


predictions = pipeline.predict(test_df)


test_df["Personality"] = list(predictions)


result = test_df["Personality"]


mapping = {0: 'Extrovert', 1: 'Introvert'}


result = result.replace(mapping)


result.to_csv("submission.csv")

