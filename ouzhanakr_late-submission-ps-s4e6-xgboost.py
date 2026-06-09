# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, time

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer

import optuna
from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import optuna
from xgboost import XGBClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')


train.isnull().sum()


train.info()


train.head()


RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 25            
USE_GPU = False           


CATEGORICAL_COLS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]



all_features = [i for i in train.columns if i not in ['id','Target']]


X = train[all_features].copy()
X_test = test[all_features].copy()


present_cats = [c for c in CATEGORICAL_COLS if c in all_features]
for c in present_cats:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")


le = LabelEncoder()
y = le.fit_transform(train['Target'])
n_classes = len(le.classes_)
print(list(le.classes_))


def build_pipeline_trial(trial, n_classes):
    clf = XGBClassifier(
        n_estimators      = trial.suggest_int("n_estimators", 600, 1600),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth         = trial.suggest_int("max_depth", 5, 10),
        min_child_weight  = trial.suggest_float("min_child_weight", 1.0, 10.0),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.0, 20.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 5.0),
        gamma             = trial.suggest_float("gamma", 0.0, 5.0),
        max_bin           = trial.suggest_categorical("max_bin", [256, 512, 1024, 2048]),
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        enable_categorical=True,                     
        tree_method="gpu_hist" if USE_GPU else "hist",
        random_state=RANDOM_STATE,
        n_jobs=1                                       
    )

    pipe = Pipeline(steps=[('xgb',clf)])
    return pipe


def build_pipeline_from_params(params, n_classes):
    clf = XGBClassifier(
        n_estimators     = params["n_estimators"],
        learning_rate    = params["learning_rate"],
        max_depth        = params["max_depth"],
        min_child_weight = params["min_child_weight"],
        subsample        = params["subsample"],
        colsample_bytree = params["colsample_bytree"],
        reg_lambda       = params["reg_lambda"],
        reg_alpha        = params["reg_alpha"],
        gamma            = params["gamma"],
        max_bin          = params["max_bin"],
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        enable_categorical=True,
        tree_method="gpu_hist" if USE_GPU else "hist",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    pipe = Pipeline(steps=[('xgb',clf)])
    return pipe


def cross_validation_simple_cls(
    model,
    X_df,
    y_array,
    X_test_df=None,
    n_splits=5,
    shuffle=True,
    random_state=42,
    single_fold=False,
    n_repeats=1,
    predict_test=False,
    label_encoder=None
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    n_classes = len(label_encoder.classes_) if label_encoder is not None else len(np.unique(y_array))

    oof_proba = np.zeros((len(X_df), n_classes),dtype=float)
    fold_scores = []

    t0 = time.time()

    for fold, (i_tr,i_val) in enumerate(skf.split(X_df,y_array), 1):
        X_tr, y_tr = X_df.iloc[i_tr], y_array[i_tr]
        X_val, y_val = X_df.iloc[i_val], y_array[i_val]
        val_proba = np.zeros((len(i_val),n_classes), dtype=float)

        for rep in range(n_repeats):
            m = clone(model)
            m.fit(X_tr,y_tr)
            val_proba += m.predict_proba(X_val)

        val_proba /= n_repeats
        y_pred = np.argmax(val_proba, axis=1)


        f1 = f1_score(y_val, y_pred, average="macro")
        fold_scores.append(f1)
        oof_proba[i_val] = val_proba
        print(f"# Fold {fold}: F1_macro={f1:.5f}")

        if single_fold:
            break

    f1_mean = float(np.mean(fold_scores))
    mins = int(round((time.time() - t0) / 60))
    print(f"# Overall: F1_macro={f1_mean:.5f} (single_fold={single_fold}) {mins} min")

    result = {
        "f1_mean": f1_mean,
        "fold_scores": fold_scores,
        "oof_proba": oof_proba,
        "oof_pred": np.argmax(oof_proba, axis=1),
    }

    if predict_test and X_test_df is not None:
        m = clone(model)
        m.fit(X_df, y_array)
        test_proba = m.predict_proba(X_test_df)
        test_pred_idx = np.argmax(test_proba, axis=1)
        test_pred_lbl = le.inverse_transform(test_pred_idx)
        result["test_pred"] = test_pred_lbl
        result["test_proba"] = test_proba

    return result


def objective(trial):
    pipe = build_pipeline_trial(trial, n_classes=n_classes)
    res = cross_validation_simple_cls(
        model=pipe,
        X_df=X,
        y_array=y,
        X_test_df=None,
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
        single_fold=False,
        n_repeats=1,
        predict_test=False,
        label_encoder=le
    )
    return res["f1_mean"]



sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_nativecat_optuna")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Best macro-F1:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")



ID_COL    = "id"
TARGET_COL = "Target"



best_pipeline = build_pipeline_from_params(study.best_params, n_classes=n_classes)

final_res = cross_validation_simple_cls(
    model=best_pipeline,
    X_df=X,
    y_array=y,
    X_test_df=X_test,
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
    single_fold=False,
    n_repeats=1,
    predict_test=True,
    label_encoder=le
)

submission = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET_COL: final_res["test_pred"]
})
submission.to_csv("submission.csv", index=False)

submission.head()



submission




