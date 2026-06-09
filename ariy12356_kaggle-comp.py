!rm -rf /kaggle/working/clean
!python -m pip -q install -U virtualenv

!virtualenv -p python3 /kaggle/working/clean

# обновим базу в venv
!/kaggle/working/clean/bin/python -m pip -q install -U pip wheel setuptools

# поставим wrapt (иногда нужен разным пакетам; и уберём шум, если где-то подхватится)
!/kaggle/working/clean/bin/python -m pip -q install -U wrapt

# поставить fedot (без -q чтобы видеть ошибки, если будут)
!/kaggle/working/clean/bin/python -m pip install -U fedot


!/kaggle/working/clean/bin/python -c "import sys, importlib.util; print('PY:', sys.executable); spec=importlib.util.find_spec('fedot'); print('fedot spec:', None if spec is None else spec.origin)"


%%writefile /kaggle/working/run_fedot.py
from pathlib import Path
from datetime import datetime
import json, sys, platform, subprocess, shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fedot import FedotBuilder


# =========================
# 0) Helpers: make predictions robust to FEDOT vs sklearn-style APIs
# =========================
from fedot.core.pipelines.pipeline import Pipeline as FedotPipeline

def safe_predict(model, X):
    # Если это FEDOT Pipeline — НИКАКОГО sklearn-fallback
    if isinstance(model, FedotPipeline):
        return model.predict(features=X)

    # иначе пробуем FEDOT-стиль, потом sklearn-стиль
    try:
        return model.predict(features=X)
    except TypeError:
        return model.predict(X if hasattr(X, "iloc") else np.asarray(X))



def safe_predict_proba(model, X):
    """Return probabilities with compatibility for FEDOT and sklearn-like pipelines.
    Returns None if not available.
    """
    # FEDOT-style: predict_proba(features=..., probs_for_classes=True)
    try:
        return model.predict_proba(features=X, probs_for_classes=True)
    except TypeError:
        pass
    except AttributeError:
        pass

    # sklearn-style: predict_proba(X)
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except TypeError:
            return None
    return None


def proba_class1(proba):
    """Extract probability of class=1 from various shapes returned by different APIs."""
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.reshape(-1)


# =========================
# 1) Load data (adjust paths/target if needed)
# =========================
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"
TARGET_COL = "diagnosed_diabetes"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

y_train = train[TARGET_COL]
X_train = train.drop(columns=[TARGET_COL])
X_test  = test.copy()

# If competition has an ID column, keep it for submission
ID_COL_CANDIDATES = ["id", "ID", "Id"]
ID_COL = next((c for c in ID_COL_CANDIDATES if c in X_test.columns), None)

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =========================
# 2) Run dir (artifacts)
# =========================
RUN_DIR = Path("/kaggle/working") / "fedot_runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for d in ["meta", "history", "cache", "preds", "best_pipeline"]:
    (RUN_DIR / d).mkdir(parents=True, exist_ok=True)

# anchors
np.save(RUN_DIR / "meta" / "train_index.npy", np.asarray(getattr(X_train, "index", pd.RangeIndex(len(X_train)))))
np.save(RUN_DIR / "meta" / "test_index.npy",  np.asarray(getattr(X_test,  "index", pd.RangeIndex(len(X_test)))))

folds = []
for tr_idx, va_idx in cv.split(X_train, y_train):
    folds.append({"train_idx": tr_idx.tolist(), "valid_idx": va_idx.tolist()})
(RUN_DIR / "meta" / "cv_folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

# =========================
# 3) FEDOT builder with allowed ops
# =========================
ALLOWED_OPS = [
    # models (tabular)
    "bernb", "multinb",
    "catboost", "lgbm", "xgboost",
    "rf", "dt", "knn", "logit", "lda", "qda",
    # preprocessing
    "simple_imputation", "one_hot_encoding", "label_encoding",
    "scaling", "normalization",
    "pca", "kernel_pca", "fast_ica", "poly_features",
    # selection/robust/outliers
    "rfe_lin_class", "rfe_non_lin_class",
    "isolation_forest_class",
    "class_decompose", "resample",
]

fedot = (
    FedotBuilder(problem="classification")
    .setup_composition(timeout=660, seed=42)
    .setup_parallelization(n_jobs=2)
    .setup_output(logging_level=20)
    .setup_pipeline_structure(available_operations=ALLOWED_OPS)
    .setup_pipeline_evaluation(metric="roc_auc")
    .build()
)

best_pipe = None

try:
    # =========================
    # 4) Fit
    # =========================
    best_pipe = fedot.fit(features=X_train, target=y_train)

    # =========================
    # 5) Save pipeline + history
    # =========================
    best_json = RUN_DIR / "best_pipeline" / "best_pipeline.json"
    if getattr(fedot, "current_pipeline", None) is not None:
        fedot.current_pipeline.save(path=str(best_json))

    if getattr(fedot, "history", None) is not None:
        fedot.history.save(str(RUN_DIR / "history" / "history.json"))

        # =========================
    # 6) Predictions (robust)
    # =========================
    preds_dir = RUN_DIR / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    proba = safe_predict_proba(best_pipe, X_test)
    if proba is not None:
        p1 = proba_class1(proba)
        print("Using predict_proba().")
    else:
        print("No predict_proba available; using predict() output as score for class 1.")
        pred_raw = safe_predict(best_pipe, X_test)
        p1 = np.asarray(pred_raw).reshape(-1)

    # Save scores
    pd.DataFrame({"p1": p1}).to_csv(preds_dir / "test_p1.csv", index=False)

    # Kaggle submission
    sub = pd.DataFrame()
    if ID_COL is not None:
        sub[ID_COL] = X_test[ID_COL].values
    sub[TARGET_COL] = p1
    sub.to_csv(preds_dir / "submission.csv", index=False)
    print("Saved:", preds_dir / "submission.csv")


finally:
    # =========================
    # 7) Meta always saved
    # =========================
    (RUN_DIR / "meta" / "python_executable.txt").write_text(sys.executable, encoding="utf-8")
    (RUN_DIR / "meta" / "python.txt").write_text(sys.version, encoding="utf-8")
    (RUN_DIR / "meta" / "platform.txt").write_text(platform.platform(), encoding="utf-8")
    (RUN_DIR / "meta" / "pip_freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        encoding="utf-8"
    )

    # =========================
    # 8) Zip even if something failed
    # =========================
    shutil.rmtree(RUN_DIR / "cache", ignore_errors=True)
    zip_path = shutil.make_archive(str(RUN_DIR), "zip", root_dir=str(RUN_DIR))
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    print("Artifacts saved to:", zip_path)


!/kaggle/working/clean/bin/python /kaggle/working/run_fedot.py

