VERSION = "023"
TIME_LIMIT_HOURS = 6  # stop training after this much time

# List of models to run in THIS version
# MODELS_TO_RUN = ["logreg","decision_tree","random_forest","xgboost_gpu","lightgbm_gpu","catboost_gpu","cuml_logreg","cuml_rf","cuml_svm","tabpfn_v2_5"]
MODELS_TO_RUN = [
    # "logreg",
    # "decision_tree",
    # "random_forest",
    # "xgboost_gpu",
    "lightgbm_gpu",
    # "catboost_gpu",
    # "cuml_logreg",
    # "cuml_rf",
    # "cuml_svm",
    #"tabpfn_v2_5",   # requires model file
]

N_SPLITS = 5
TARGET = "loan_paid_back"

OUTPUT_DIR = "model_outputs_v" + VERSION

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running version:", VERSION)
print("Models:", MODELS_TO_RUN)
print("Time limit (hrs):", TIME_LIMIT_HOURS)


import time, datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

# RAPIDS
# import cudf
# from cuml.linear_model import LogisticRegression as CuMLLogReg
# from cuml.ensemble import RandomForestClassifier as CuMLRF
# from cuml.svm import SVC as CuMLSVM

# GBDT
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import cupy as cp


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

features = [c for c in train.columns if c not in ["id", TARGET]]

cat_cols = train[features].select_dtypes("object").columns.tolist()
num_cols = [c for c in features if c not in cat_cols]

print("Features:", len(features))
print("Categorical:", cat_cols)
print("Numerical:", num_cols)

START_TIME = time.time()
TIME_LIMIT = TIME_LIMIT_HOURS * 3600

def time_up():
    return (time.time() - START_TIME) >= TIME_LIMIT


import cupy as cp
import numpy as np
import pandas as pd

def to_numpy_proba(proba):
    """Convert model.predict_proba output into a clean NumPy array."""
    # CuDF / RAPIDS DataFrame
    if hasattr(proba, "to_pandas"):
        return proba.to_pandas().values
    
    # Pandas Series/DataFrame
    if isinstance(proba, (pd.Series, pd.DataFrame)):
        return proba.values
    
    # CuPy array
    if isinstance(proba, cp.ndarray):
        return proba.get()
    
    # Already numpy or list-like
    return np.asarray(proba)

def save_results(name, oof_pred, test_pred):
    oof_path  = f"{OUTPUT_DIR}/oof_{name}_v{VERSION}.csv"
    sub_path  = f"{OUTPUT_DIR}/sub_{name}_v{VERSION}.csv"

    if name != "catboost_gpu":
        pd.DataFrame({"id": train.id, TARGET: oof_pred}).to_csv(oof_path, index=False)
        print("Saved:", oof_path)
    else:
        print("Skipped saving OOF for CatBoost.")

    # Always save submission
    pd.DataFrame({"id": test.id, TARGET: test_pred}).to_csv(sub_path, index=False)
    print("Saved:", sub_path)


def encode_for_model(X_train, X_val, X_test, model_name):

    # models that REQUIRE encoding
    encode_needed = [
        "logreg",
        "decision_tree",
        "random_forest",
        "cuml_logreg",
        "cuml_rf",
        "cuml_svm",
        "xgboost_gpu",
        "lightgbm_gpu"
    ]

    # CatBoost must receive original raw categorical strings.
    if model_name == "catboost_gpu":
        return X_train.copy(), X_val.copy(), X_test.copy()

    # For all others: ordinal encode
    if model_name in encode_needed:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr = X_train.copy()
        Xva = X_val.copy()
        Xte = X_test.copy()

        Xtr[cat_cols] = enc.fit_transform(X_train[cat_cols])
        Xva[cat_cols] = enc.transform(X_val[cat_cols])
        Xte[cat_cols] = enc.transform(X_test[cat_cols])

        return Xtr, Xva, Xte

    return X_train, X_val, X_test

def enc1(df):
    """Ordinal encoding"""
    df = df.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = enc.fit_transform(df[cat_cols])
    return df

def enc2(df):
    """No encoding (CatBoost style raw categories)"""
    return df.copy()

def enc3(df):
    """Label-like ordinal with stable mapping"""
    df = df.copy()
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(df[c].unique())}
        df[c] = df[c].map(mapping).fillna(-1).astype(int)
    return df

def enc5(df):
    """Hash encoding"""
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype(str).apply(lambda x: hash(x) % 5000)
    return df

def scale1(df):
    return df.copy()

def scale2(df):
    df2 = df.copy()
    scaler = StandardScaler()
    df2[num_cols] = scaler.fit_transform(df[num_cols])
    return df2

def scale3(df):
    df2 = df.copy()
    scaler = MinMaxScaler()
    df2[num_cols] = scaler.fit_transform(df[num_cols])
    return df2

def scale4(df):
    df2 = df.copy()
    scaler = RobustScaler()
    df2[num_cols] = scaler.fit_transform(df[num_cols])
    return df2

def out1(df):
    return df.copy()

def out2(df):
    df2 = df.copy()
    for c in num_cols:
        q1, q3 = df2[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        df2[c] = df2[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df2

def out3(df):
    df2 = df.copy()
    for c in num_cols:
        upper = df2[c].quantile(0.99)
        df2[c] = df2[c].clip(upper=upper)
    return df2

def out4(df):
    df2 = df.copy()
    for c in num_cols:
        lower = df2[c].quantile(0.01)
        df2[c] = df2[c].clip(lower=lower)
    return df2

def preprocess_combo(df, enc_fn, scale_fn, out_fn):
    df2 = df.copy()
    df2 = out_fn(df2)
    df2 = enc_fn(df2)
    df2 = scale_fn(df2)
    return df2



HP_GRID = {
    "logreg": [{"C": c} for c in [0.5, 1, 2]],
    "decision_tree": [{"max_depth": d} for d in [8, 12, 16]],
    "random_forest": [{"n_estimators": n} for n in [200, 300]],
    "xgboost_gpu": [{"max_depth": d, "learning_rate": lr} for d in [6, 7] for lr in [0.02, 0.03]],
    "lightgbm_gpu": [{"num_leaves": nl} for nl in [32, 64, 128]],
    "catboost_gpu": [{"depth": d} for d in [6, 8]],
    "cuml_logreg": [{"max_iter": mi} for mi in [200, 300]],
    "cuml_rf": [{"n_estimators": n} for n in [300, 500]],
    "cuml_svm": [{"C": c} for c in [0.5, 1, 2]],
}


def get_model(name, params=None):
    if params is None:
        params = {}

    if name == "logreg":
        return LogisticRegression(max_iter=1000, **params)

    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params)

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=-1, **params)

    if name == "xgboost_gpu":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            enable_categorical=True,
            device="cuda",
            n_estimators=3000,
            subsample=0.8,
            colsample_bytree=0.8,
            **params
        )

    if name == "lightgbm_gpu":
        return LGBMClassifier(
            device="gpu",
            boosting_type="gbdt",
            objective="binary",
            metric="auc",
            n_estimators=3000,
            learning_rate=0.02,
            **params
        )

    if name == "catboost_gpu":
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=3000,
            learning_rate=0.02,
            task_type="GPU",
            verbose=False,
            **params
        )

    if name == "cuml_logreg":
        return CuMLLogReg(**params)

    if name == "cuml_rf":
        return CuMLRF(**params)

    if name == "cuml_svm":
        return CuMLSVM(probability=True, **params)

    raise ValueError("Unknown model:", name)


ENCODERS = [enc1, enc2, enc3, enc5]
SCALERS  = [scale1, scale2, scale3, scale4]
OUTLIERS = [out1, out2, out3, out4]

COMBOS = [(e, s, o) for e in ENCODERS for s in SCALERS for o in OUTLIERS]

results = []

def save_results_df():
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/results_v{VERSION}.csv", index=False)
    return df

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for enc_fn, scale_fn, out_fn in COMBOS:
    for model_name in MODELS_TO_RUN:
        for hp in HP_GRID[model_name]:

            iter_start = time.time()

            print("\n=================================================")
            print(f"Model: {model_name} | HP: {hp}")
            print(f"ENC: {enc_fn.__name__} | SCALE: {scale_fn.__name__} | OUT: {out_fn.__name__}")
            print("=================================================")

            # SKIP invalid combos (enc2 raw categories + non-CatBoost)
            if enc_fn.__name__ == "enc2" and model_name != "catboost_gpu":
                print("Skipping: enc2 only valid for CatBoost")
                continue

            if time_up():
                print("TIME LIMIT — STOPPING")
                break

            # Preprocess once
            train_prep = preprocess_combo(train[features], enc_fn, scale_fn, out_fn)
            test_prep  = preprocess_combo(test[features],  enc_fn, scale_fn, out_fn)

            model = get_model(model_name, hp)
            oof = np.zeros(len(train))
            test_pred = np.zeros(len(test))

            for fold, (tr_idx, va_idx) in enumerate(skf.split(train_prep, train[TARGET]), 1):

                if time_up():
                    print("⏳ TIME inside fold — STOPPING")
                    break

                Xtr = train_prep.iloc[tr_idx]
                Xva = train_prep.iloc[va_idx]
                ytr = train[TARGET].iloc[tr_idx]
                yva = train[TARGET].iloc[va_idx]

                if model_name == "catboost_gpu":
                    # raw categories
                    Xtr_cb = Xtr.reset_index(drop=True)
                    Xva_cb = Xva.reset_index(drop=True)
                    Xte_cb = test_prep.reset_index(drop=True)
                    ytr_cb = ytr.reset_index(drop=True)
                    yva_cb = yva.reset_index(drop=True)

                    cat_idx = [list(Xtr_cb.columns).index(c) for c in cat_cols]

                    model.fit(
                        Xtr_cb, ytr_cb,
                        cat_features=cat_idx,
                        eval_set=(Xva_cb, yva_cb),
                        verbose=False
                    )

                    preds = model.predict_proba(Xva_cb)[:, 1]
                    preds_test = model.predict_proba(Xte_cb)[:, 1]

                else:
                    # Fit
                    model.fit(Xtr, ytr)
                    preds = to_numpy_proba(model.predict_proba(Xva))[:, 1]
                    preds_test = to_numpy_proba(model.predict_proba(test_prep))[:, 1]

                oof[va_idx] = preds
                test_pred += preds_test / N_SPLITS

                print(f"Fold {fold}: AUC = {roc_auc_score(yva, preds):.5f}")

            # GLOBAL AUC
            auc = roc_auc_score(train[TARGET], oof)
            print(f"FINAL AUC = {auc:.5f}")

            # Unique filename
            combo_name = f"{model_name}_{enc_fn.__name__}_{scale_fn.__name__}_{out_fn.__name__}"

            # Save OOF except for CatBoost
            if model_name != "catboost_gpu":
                pd.DataFrame({"id": train.id, TARGET: oof}).to_csv(
                    f"{OUTPUT_DIR}/oof_{combo_name}_v{VERSION}.csv", index=False
                )

            # ALWAYS save submission
            pd.DataFrame({"id": test.id, TARGET: test_pred}).to_csv(
                f"{OUTPUT_DIR}/sub_{combo_name}_v{VERSION}.csv", index=False
            )

            # Save pipeline used
            pd.to_pickle(
                (enc_fn, scale_fn, out_fn, hp),
                f"{OUTPUT_DIR}/pipeline_{combo_name}_v{VERSION}.pkl"
            )

            iter_time = time.time() - iter_start

            # Log results
            results.append({
                "model": model_name,
                "encoder": enc_fn.__name__,
                "scaler": scale_fn.__name__,
                "outlier": out_fn.__name__,
                "hp": hp,
                "auc": auc,
                "iteration_time": iter_time,
                "runtime_total": time.time() - START_TIME
            })

            # Save after every iteration
            df_now = save_results_df()
            print(df_now.tail(3))

            if time_up():
                print("STOP due to time limit")
                break

    if time_up():
        break


results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/results_v{VERSION}.csv", index=False)

results_df.sort_values("auc", ascending=False).head(20)


plt.figure(figsize=(10,6))
plt.barh(results_df.model, results_df.auc)
plt.gca().invert_yaxis()
plt.title(f"Model Performance (Version {VERSION})")
plt.xlabel("OOF AUC")
plt.tight_layout()
plt.show()




