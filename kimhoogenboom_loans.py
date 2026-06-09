import os, gc
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation

from catboost import CatBoostClassifier, Pool


SEED = 26
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 100
MAX_BOOST_ROUNDS = 4000
LEARNING_RATE = 0.02
LOG_EVERY = 100

USE_CATBOOST = False
TRAIN_SPECIALISTS = True
MIN_SPECIALIST_ROWS = 10000 


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


TARGET = "loan_paid_back"
ID_COL = "id"


def map_grade_subgrade_to_order(s):
    # Expect patterns like 'A1'...'G5' or similar; map to a single ordered integer.
    # A=0,...,G=6 and subgrade 1..5
    if pd.isna(s) or not isinstance(s, str) or len(s) < 2:
        return np.nan
    letter = s[0].upper()
    sub = s[1:]
    letter_map = {ch:i for i, ch in enumerate(list("ABCDEFG"))}
    if letter not in letter_map:
        return np.nan
    try:
        sub = int(sub)
    except:
        return np.nan
    # Compress into 0..34 (7*5=35 combos)
    return letter_map[letter] * 5 + (sub - 1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ordered grade
    out["grade_order"] = out["grade_subgrade"].map(map_grade_subgrade_to_order)

    # Ratios / interactions (guard divide by zero)
    eps = 1e-6
    out["loan_to_income_ratio"] = out["loan_amount"] / (out["annual_income"] + eps)
    out["interest_x_dti"] = out["interest_rate"] * out["debt_to_income_ratio"]
    out["income_per_credit"] = out["annual_income"] / (out["credit_score"] + eps)
    out["rate_to_income"] = out["interest_rate"] / (out["annual_income"] + eps)
    out["loan_per_dti"] = out["loan_amount"] / (out["debt_to_income_ratio"] + eps)

    return out


train = add_features(train)
test  = add_features(test)


num_cols = [
    "annual_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "interest_rate",
    "grade_order", "loan_to_income_ratio", "interest_x_dti",
    "income_per_credit", "rate_to_income", "loan_per_dti"
]

cat_cols = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]


all_features = num_cols + cat_cols

X = train[all_features].copy()
y = train[TARGET].astype(int).copy()
X_test = test[all_features].copy()


numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0
)


def build_lgbm():
    return LGBMClassifier(
        n_estimators=MAX_BOOST_ROUNDS,
        learning_rate=LEARNING_RATE,
        objective="binary",
        boosting_type="gbdt",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        max_depth=-1,
        num_leaves=63,
        random_state=SEED,
        n_jobs=-1
    )


def build_catboost():
    # CatBoost can handle categoricals natively, but here we’re passing transformed arrays.
    # Using plain numeric pipeline; if you want native categorical handling, split the pipeline.
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=MAX_BOOST_ROUNDS,
        learning_rate=LEARNING_RATE,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=SEED,
        verbose=False,
        od_type="Iter",
        od_wait=EARLY_STOPPING_ROUNDS,
        task_type="CPU"
    )


def fit_global_model(X, y, X_test):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_pred = np.zeros(len(X))
    test_pred_folds = []

    fold_auc = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]

        if USE_CATBOOST:
            model = build_catboost()
            # Build per-fold preprocessing to avoid leakage (fit only on train)
            pre = preprocessor.fit(X_tr)
            X_tr_p = pre.transform(X_tr)
            X_va_p = pre.transform(X_va)
            X_te_p = pre.transform(X_test)

            # CatBoost expects Pool for proper eval metric
            train_pool = Pool(X_tr_p, label=y_tr)
            valid_pool = Pool(X_va_p, label=y_va)

            model.fit(train_pool, eval_set=valid_pool, verbose=False)
            va_pred = model.predict_proba(X_va_p)[:, 1]
            te_pred = model.predict_proba(X_te_p)[:, 1]

        else:
            model = build_lgbm()
            # Wrap in a pipeline so preprocessing is fold-fitted
            pipe = Pipeline([
                ("prep", preprocessor),
                ("clf", model)
            ])

            pipe.fit(
                X_tr, y_tr,
                clf__eval_set=[(preprocessor.fit_transform(X_va), y_va)],
                clf__callbacks=[early_stopping(EARLY_STOPPING_ROUNDS), log_evaluation(LOG_EVERY)],
            )
            va_pred = pipe.predict_proba(X_va)[:, 1]
            te_pred = pipe.predict_proba(X_test)[:, 1]

            model = pipe  # keep the pipeline as model for storage

        auc = roc_auc_score(y_va, va_pred)
        fold_auc.append(auc)
        oof_pred[val_idx] = va_pred
        test_pred_folds.append(te_pred)

        print(f"[Global] Fold {fold}/{N_SPLITS} AUC: {auc:.5f}")

        gc.collect()

    oof_auc = roc_auc_score(y, oof_pred)
    test_pred = np.mean(test_pred_folds, axis=0)

    print(f"[Global] OOF AUC: {oof_auc:.5f} | Folds: {', '.join(f'{a:.5f}' for a in fold_auc)}")
    return oof_pred, test_pred


def fit_specialists(X, y, X_test, group_col="loan_purpose"):
    specialists = {}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Default predictions (zeros); we’ll fill where specialist applies
    oof_spec = np.zeros(len(X))
    test_spec = np.zeros(len(X_test))
    mask_oof = np.zeros(len(X), dtype=bool)
    mask_test = np.zeros(len(X_test), dtype=bool)

    groups = X[group_col].values
    groups_test = X_test[group_col].values
    unique_groups = pd.Series(groups).unique()

    for g in unique_groups:
        idx_tr = np.where(groups == g)[0]
        if len(idx_tr) < MIN_SPECIALIST_ROWS:
            continue

        print(f"[Specialist] Training group '{g}' with {len(idx_tr)} rows")
        Xg = X.iloc[idx_tr]
        yg = y.iloc[idx_tr]

        oof_pred_g = np.zeros(len(Xg))
        test_pred_g_folds = []

        for fold, (trn_idx, val_idx) in enumerate(skf.split(Xg, yg), 1):
            X_tr, X_va = Xg.iloc[trn_idx], Xg.iloc[val_idx]
            y_tr, y_va = yg.iloc[trn_idx], yg.iloc[val_idx]

            if USE_CATBOOST:
                model = build_catboost()
                pre = preprocessor.fit(X_tr)
                X_tr_p = pre.transform(X_tr)
                X_va_p = pre.transform(X_va)
                X_te_p = pre.transform(X_test[X_test[group_col] == g])

                train_pool = Pool(X_tr_p, label=y_tr)
                valid_pool = Pool(X_va_p, label=y_va)

                model.fit(train_pool, eval_set=valid_pool, verbose=False)
                va_pred = model.predict_proba(X_va_p)[:, 1]
                te_pred = model.predict_proba(X_te_p)[:, 1]
            else:
                model = build_lgbm()
                pipe = Pipeline([
                    ("prep", preprocessor),
                    ("clf", model)
                ])
                pipe.fit(
                    X_tr, y_tr,
                    clf__eval_set=[(preprocessor.fit_transform(X_va), y_va)],
                    clf__callbacks=[early_stopping(EARLY_STOPPING_ROUNDS), log_evaluation(LOG_EVERY)],
                )
                va_pred = pipe.predict_proba(X_va)[:, 1]
                te_pred = pipe.predict_proba(X_test[X_test[group_col] == g])[:, 1]

            oof_pred_g[val_idx] = va_pred
            test_pred_g_folds.append(te_pred)

        # assign OOF preds back
        oof_spec[idx_tr] = oof_pred_g
        mask_oof[idx_tr] = True

        # test indices for this group
        idx_te = np.where(groups_test == g)[0]
        if len(idx_te) > 0:
            test_spec[idx_te] = np.mean(test_pred_g_folds, axis=0)
            mask_test[idx_te] = True

        gc.collect()

    return oof_spec, test_spec, mask_oof, mask_test


print("Training global model...")
oof_global, test_global = fit_global_model(X, y, X_test)

if TRAIN_SPECIALISTS:
    print("Training specialist models (per loan_purpose)...")
    oof_spec, test_spec, mask_oof, mask_test = fit_specialists(X, y, X_test, group_col="loan_purpose")
else:
    # No specialists, set masks False
    oof_spec = np.zeros(len(X))
    test_spec = np.zeros(len(X_test))
    mask_oof = np.zeros(len(X), dtype=bool)
    mask_test = np.zeros(len(X_test), dtype=bool)


oof_final = oof_global.copy()
oof_final[mask_oof] = oof_spec[mask_oof]

test_final = test_global.copy()
test_final[mask_test] = test_spec[mask_test]

oof_auc_final = roc_auc_score(y, oof_final)
print(f"[Final] OOF AUC after specialist overlay: {oof_auc_final:.5f}")


sub = sample_sub.copy()
sub["loan_paid_back"] = test_final
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

# Optional: quick model calibration check (distribution)
print("Prediction summary (test):", pd.Series(test_final).describe())

