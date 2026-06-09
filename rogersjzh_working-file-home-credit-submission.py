import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import gc

import re
import numpy as np
import pandas as pd
import polars as pl

from glob import glob
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


class DataPrep:

    # Special columns:
    # case_id - This is the unique identifier for each credit case. You'll need this ID to join relevant tables to the base table.
    # date_decision - This refers to the date when a decision was made regarding the approval of the loan.
    # WEEK_NUM - This is the week number used for aggregation. In the test sample, WEEK_NUM continues sequentially from the last training value of WEEK_NUM.
    # MONTH - This column represents the month and is intended for aggregation purposes.
    # target - This is the target value, determined after a certain period based on whether or not the client defaulted on the specific credit case (loan).
    # num_group1 - This is an indexing column used for the historical records of case_id in both depth=1 and depth=2 tables.
    # num_group2 - This is the second indexing column for depth=2 tables' historical records of case_id. The order of num_group1 and num_group2 is important and will be clarified in feature definitions.
    # All other raw columns in the tables serve as predictors. Their definitions can be found in the file feature_definitions.csv. For depth=0 tables, predictors can be directly used as features. However, for tables with depth>0, you may need to employ aggregation functions that will condense the historical records associated with each case_id into a single feature. In case num_group1 or num_group2 stands for person index (this is clear with predictor definitions) the zero index has special meaning. When num_groupN=0 it is the applicant (the person who applied for a loan).
    
    # Various predictors were transformed, therefore we have the following notation for similar groups of transformations
    # P - Transform DPD (Days past due)
    # M - Masking categories
    # A - Transform amount
    # D - Transform date
    # T - Unspecified Transform
    # L - Unspecified Transform
    
    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    # Handle dates
    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
        df = df.drop("date_decision", "MONTH")

        return df

    # Filter columns
    # If the column name is not in the reserved list and the null value ratio of the column is greater than 0.95, then delete the column.
    # If the column name is not in the reserved list, the column data type is String, and the number of unique values is 1 or greater than 200, then delete the column.
    @staticmethod
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()

                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df


class Aggregator:
    # Generate maximum aggregate expression for numeric columns
    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    # Generate maximum aggregate expression for date type columns
    @staticmethod
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    # Generate a maximum aggregate expression for a column of type string
    @staticmethod
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    # Generate maximum aggregate expressions for columns of other types
    @staticmethod
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    # Generate the maximum aggregate expression for a specific column "num_group"
    @staticmethod
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    # Get all types of aggregate expressions
    @staticmethod
    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)
        return exprs


# Read a single file and preprocess it
def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(DataPrep.set_table_dtypes)
    # If the depth parameter is 1 or 2, the data is aggregated by "case_id"
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


# Read multiple files and preprocess them
def read_files(regex_path, depth=None):
    chunks = []
    for path in glob(str(regex_path)):
        chunks.append(pl.read_parquet(path).pipe(DataPrep.set_table_dtypes))
    df = pl.concat(chunks, how="vertical_relaxed")
    # If the depth parameter is 1 or 2, the data is aggregated by "case_id"
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


# Feature engineering function for adding new features and merging dataframes
def data_preprocessing(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base
        .with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(DataPrep.handle_dates)
    return df_base


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


ROOT = Path("/kaggle/input/home-credit-credit-risk-model-stability")

TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"


# df = pl.read_parquet(TRAIN_DIR / "train_credit_bureau_b_2.parquet")
# df.head(10)


# Read all training data sets into a variable
training_data_store = {
    "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
        read_files(TRAIN_DIR / "train_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
        read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_other_1.parquet", 1),
        read_file(TRAIN_DIR / "train_person_1.parquet", 1),
        read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
        read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
        # read_file(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet",1),
    ],
    "depth_2": [
        read_file(TRAIN_DIR / "train_applprev_2.parquet", 2),
        read_file(TRAIN_DIR / "train_person_2.parquet", 2),
        # read_file(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),
        read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
    ]
}


# Load and Pre-process all training datasets
df_train = data_preprocessing(**training_data_store)
print("train data shape:\t", df_train.shape)


# Read all test data sets into a variable
test_data_store = {
    "df_base": read_file(TEST_DIR / "test_base.parquet"),
    "depth_0": [
        read_file(TEST_DIR / "test_static_cb_0.parquet"),
        read_files(TEST_DIR / "test_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
        read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
        read_file(TEST_DIR / "test_other_1.parquet", 1),
        read_file(TEST_DIR / "test_person_1.parquet", 1),
        read_file(TEST_DIR / "test_deposit_1.parquet", 1),
        read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
        # read_file(TEST_DIR / "test_credit_bureau_a_1_*.parquet",1),
    ],
    "depth_2": [
        read_file(TEST_DIR / "test_applprev_2.parquet", 2),
        read_file(TEST_DIR / "test_person_2.parquet", 2),
        # read_file(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
        read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
    ]
}


# Load and Pre-process all test datasets
df_test = data_preprocessing(**test_data_store)
print("test data shape:\t", df_test.shape)


# Remove useless columns
df_train = df_train.pipe(DataPrep.filter_cols)
df_test = df_test.select([col for col in df_train.columns if col != "target"])

print("train data shape:\t", df_train.shape)
print("test data shape:\t", df_test.shape)


# Convert to Pandas
df_train, cat_cols = to_pandas(df_train)
df_test, cat_cols = to_pandas(df_test, cat_cols)


df_train.head(20)


df_feature_definition = pd.read_csv(ROOT / "feature_definitions.csv")
df_feature_definition.head(20)


# Training Data Shape & Column Suffix Counts

def cols_by_suffix(df, sfx):
    return [c for c in df.columns if c.endswith(sfx)]
    
print("Training Data Shape (rows, cols):", df_train.shape)
for s in ["P","A","M","D","T","L"]:
    print(f"#{s}-suffix columns:", len(cols_by_suffix(df_train, s)))


# Target Distribution
if "target" in df_train.columns:
    n = len(df_train)
    n_pos = df_train["target"].sum()
    pos_rate = 100.0 * df_train["target"].mean()
    print(f"\nThere are {n:,} total number of samples.\nThere are {int(n_pos):,} positive number of samples.\nThe positive percentage is {pos_rate:.2f}%")


# Top Features with Missing Values
missing = (df_train.isna().mean().sort_values(ascending=False) * 100).round(2)
print("\nTop-15 missingness (%):")
display(missing.head(15).to_frame("missing_%"))


# Quick numeric summaries on 3 least-missing numeric columns
num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    miss_num = df_train[num_cols].isna().mean()
    chosen = miss_num.sort_values().head(3).index.tolist()
    desc = pd.concat({
        c: pd.Series({
            "mean": df_train[c].mean(),
            "std": df_train[c].std(),
            "p01": df_train[c].quantile(0.01),
            "p50": df_train[c].quantile(0.50),
            "p99": df_train[c].quantile(0.99),
            "missing_%": df_train[c].isna().mean() * 100,
        }) for c in chosen
    }, axis=1).T
    print("\nNumeric summaries (sampled columns):")
    display(desc)


 # Draw Histograms For Each Chosen Feature
plot_sample = df_train[chosen].sample(n=min(len(df_train), 200_000), random_state=42)
for c in chosen:
    plt.figure()
    plot_sample[c].dropna().hist(bins=50)
    plt.title(f"Histogram: {c}")
    plt.xlabel(c); plt.ylabel("count")
    plt.show()


# # ================================================================
# # MODELLING: LR, RF, LightGBM with robust preprocessing
# # Metrics: AUC, Gini, Gini Stability (OOF); Pick best and submit
# # ================================================================
# import os, gc, numpy as np, pandas as pd
# from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_categorical_dtype

# from sklearn import set_config
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
# from sklearn.metrics import roc_auc_score
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MaxAbsScaler, FunctionTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.base import clone
# from scipy import sparse
# from sklearn.preprocessing import StandardScaler

# from lightgbm import LGBMClassifier
# import lightgbm as lgb

# # ---------------- thread caps to reduce RAM spikes ----------------
# os.environ["OMP_NUM_THREADS"] = "2"
# os.environ["OPENBLAS_NUM_THREADS"] = "2"
# os.environ["MKL_NUM_THREADS"] = "2"
# os.environ["NUMEXPR_NUM_THREADS"] = "2"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

# # Ensure sklearn transformers return NumPy arrays by default (not pandas)
# set_config(transform_output="default")

# # --------------------- competition stability metric ---------------------
# def gini_stability(base, score_col="score", w_fallingrate=88.0, w_resstd=-0.5):
#     gini_in_time = (
#         base.loc[:, ["WEEK_NUM", "target", score_col]]
#             .sort_values("WEEK_NUM")
#             .groupby("WEEK_NUM")[["target", score_col]]
#             .apply(lambda x: 2*roc_auc_score(x["target"], x[score_col]) - 1)
#             .tolist()
#     )
#     x = np.arange(len(gini_in_time), dtype=float)
#     y = np.array(gini_in_time, dtype=float)
#     a, b = np.polyfit(x, y, 1)                    # trend
#     res_std = np.std(y - (a*x + b))               # volatility
#     return y.mean() + w_fallingrate*min(0.0, a) + w_resstd*res_std

# # --------------------- Optional quick subsample for speed ---------------------
# FAST_SAMPLE = True
# SAMPLE_ROWS = 450_000

# df = df_train.replace([np.inf, -np.inf], np.nan).copy()
# if FAST_SAMPLE and len(df) > SAMPLE_ROWS:
#     idx = np.arange(len(df))
#     _, idx_small = train_test_split(idx, train_size=SAMPLE_ROWS,
#                                     stratify=df["target"], random_state=42)
#     df = df.iloc[idx_small].copy()

# TARGET_COL = "target"
# WEEK_COL   = "WEEK_NUM"
# drop_cols  = {TARGET_COL, WEEK_COL, "case_id"}

# # --------------------- Robust feature split & dtype hygiene ---------------------
# X = df.drop(columns=list(drop_cols)).copy()
# y = df[TARGET_COL].astype(int)
# week_num = df[WEEK_COL].values

# # initial numeric guess by dtype (exclude bools)
# num_cols = [c for c in X.columns if is_numeric_dtype(X[c]) and not is_bool_dtype(X[c])]
# cat_cols_all = [c for c in X.columns if c not in num_cols]

# # demote any "numeric" columns that actually contain non-numeric tokens
# bad_num = []
# for c in list(num_cols):
#     coerced = pd.to_numeric(X[c], errors="coerce")
#     frac_bad = ((~X[c].isna()) & (coerced.isna())).mean()
#     if frac_bad > 0.0:
#         bad_num.append(c)
# for c in bad_num:
#     num_cols.remove(c)
#     if c not in cat_cols_all:
#         cat_cols_all.append(c)

# # set categoricals to category dtype; downcast numerics for memory
# for c in cat_cols_all:
#     if not is_categorical_dtype(X[c]):
#         X[c] = X[c].astype("category")
# for c in num_cols:
#     if X[c].dtype == "float64":
#         X[c] = X[c].astype("float32")
#     elif X[c].dtype == "int64":
#         X[c] = X[c].astype("int32")

# # low-card subset for LR one-hot (keeps LR sparse matrix small)
# low_card_cats = [c for c in cat_cols_all if X[c].nunique(dropna=True) <= 30]

# # helpers: keep matrices memory-friendly
# to_csr32 = FunctionTransformer(
#     lambda A: sparse.csr_matrix(A, dtype=np.float32) if not sparse.issparse(A) else A.astype(np.float32),
#     accept_sparse=True
# )
# to_numpy32 = FunctionTransformer(
#     lambda A: (A.to_numpy(dtype=np.float32) if isinstance(A, pd.DataFrame) else A.astype(np.float32, copy=False)),
#     accept_sparse=False
# )

# # helper: convert any DataFrame/ndarray to a numpy object array of strings
# cat_to_object = FunctionTransformer(
#     lambda A: (
#         A.astype(str).to_numpy(object) if isinstance(A, pd.DataFrame)
#         else np.asarray(A).astype(str)
#     ),
#     accept_sparse=False, feature_names_out="one-to-one"
# )


# # =============================== 1) LOGISTIC REGRESSION ===============================
# to_csr_strict = FunctionTransformer(
#     lambda A: (A.tocsr().astype(np.float32) if sparse.issparse(A)
#                else sparse.csr_matrix(A, dtype=np.float32)),
#     accept_sparse=True
# )

# pre_lr = ColumnTransformer(
#     transformers=[
#         # Numeric: impute -> force CSR -> Standardize (sparse-safe with_mean=False)
#         ("num", make_pipeline(SimpleImputer(strategy="median"),
#                               to_csr_strict,
#                               StandardScaler(with_mean=False)),
#          num_cols),

#         # Categorical (low-card only for LR): OHE sparse
#         ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse=True), low_card_cats),
#     ],
#     remainder="drop",
#     sparse_threshold=1.0   # keep the combined matrix sparse
# )

# lr_pipe = Pipeline([
#     ("prep", pre_lr),
#     ("clf", LogisticRegression(
#         solver="saga", C=1.0, tol=1e-3, max_iter=150,
#         n_jobs=-1, random_state=42
#     )),
# ])

# # =============================== 2) RANDOM FOREST ===============================
# pre_tree = ColumnTransformer(
#     transformers=[
#         ("num", SimpleImputer(strategy="median"), num_cols),
#         ("cat", make_pipeline(
#             cat_to_object,  # <--- NEW: force string/object dtype so imputer won't try to float-cast
#             SimpleImputer(strategy="most_frequent"),
#             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
#         ), cat_cols_all),
#     ],
#     remainder="drop",
#     n_jobs=1
# )

# rf_pipe = Pipeline([
#     ("prep", pre_tree),
#     ("to32", to_numpy32),
#     ("clf", RandomForestClassifier(
#         n_estimators=160, max_depth=12, max_features="sqrt",
#         max_samples=0.6, bootstrap=True,
#         n_jobs=2, random_state=42, class_weight="balanced_subsample"
#     )),
# ])

# # =============================== 3) LIGHTGBM ===============================
# try:
#     lgb_est = LGBMClassifier(
#         n_estimators=600, learning_rate=0.06, num_leaves=64,
#         max_depth=-1, subsample=0.8, colsample_bytree=0.8,
#         reg_lambda=1.0, n_jobs=2, objective="binary", metric="auc",
#         device_type="gpu"   # falls back to CPU if no GPU
#     )
# except TypeError:
#     lgb_est = LGBMClassifier(
#         n_estimators=600, learning_rate=0.06, num_leaves=64,
#         max_depth=-1, subsample=0.8, colsample_bytree=0.8,
#         reg_lambda=1.0, n_jobs=2, objective="binary", metric="auc",
#         device="gpu"
#     )

# lgb_pipe = Pipeline([
#     ("prep", pre_tree),
#     ("to32", to_numpy32),
#     ("clf", lgb_est),
# ])


# # =============================== CV runner (OOF + GS) ===============================
# def cv_oof_report(name, pipeline, X, y, week_num, n_splits=3, random_state=42):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     oof = np.zeros(len(y), dtype=np.float32)

#     for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
#         X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

#         model = clone(pipeline)
#         fit_params = {}
#         if name.lower().startswith("lgb"):
#             fit_params = {
#                 "clf__eval_set": [(X_va, y_va)],
#                 # Use callbacks for early stopping & silence logs
#                 "clf__callbacks": [lgb.early_stopping(50, verbose=False),
#                                    lgb.log_evaluation(period=-1)]
#             }

#         model.fit(X_tr, y_tr, **fit_params)
#         oof[va_idx] = model.predict_proba(X_va)[:, 1].astype(np.float32)

#         print(f"[{name}] Fold {fold}/{n_splits} AUC={roc_auc_score(y_va, oof[va_idx]):.5f}")

#         del X_tr, X_va, y_tr, y_va, model
#         gc.collect()

#     auc  = roc_auc_score(y, oof)
#     gini = 2*auc - 1
#     base = pd.DataFrame({WEEK_COL: week_num, "target": y.values, "score": oof})
#     gs   = gini_stability(base)

#     print(f"\n[{name}] OOF AUC={auc:.5f} | Gini={gini:.5f} | GiniStability={gs:.5f}\n")
#     return {"model": name, "oof_auc": auc, "oof_gini": gini, "gini_stability": gs, "oof": oof, "pipe": pipeline}


# # =============================== Run CV for all models ===============================
# out_lr  = cv_oof_report("LogReg_saga",  lr_pipe, X, y, week_num)


# out_rf  = cv_oof_report("RandomForest", rf_pipe, X, y, week_num)


# out_lgb = cv_oof_report("LGB_fast",     lgb_pipe, X, y, week_num)


# summary = pd.DataFrame([out_lr, out_rf, out_lgb]).drop(columns=["oof", "pipe"]).sort_values("oof_auc", ascending=False)
# print("=== CV Summary ===")
# print(summary.to_string(index=False))


# # =============================== Pick best, refit, predict test, submit ===============================
# SELECT_BY = "gini_stability"   # or "oof_auc"
# best = max([out_lr, out_rf, out_lgb], key=lambda d: d[SELECT_BY])
# print(f"[BEST MODEL] {best['model']} by {SELECT_BY}: {best[SELECT_BY]:.5f}")

# best_pipe = clone(best["pipe"])


# # Refit: LGB uses a tiny validation split for early stopping; others fit on full data
# if best["model"].lower().startswith("lgb"):
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
#     tr_idx, va_idx = next(sss.split(X, y))
#     X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
#     X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

#     best_pipe.fit(
#         X_tr, y_tr,
#         clf__eval_set=[(X_va, y_va)],
#         clf__callbacks=[lgb.early_stopping(100, verbose=False),
#                         lgb.log_evaluation(period=-1)]
#     )

#     del X_tr, X_va, y_tr, y_va
#     gc.collect()
# else:
#     best_pipe.fit(X, y)


# # --------------------- Build X_test aligned to training columns ---------------------
# TEST_DROP = ["case_id", "WEEK_NUM"]
# case_ids = df_test["case_id"].values
# X_test = df_test.drop(columns=[c for c in TEST_DROP if c in df_test.columns]).copy()

# # add any train-only columns missing in test as NaN
# missing_in_test = [c for c in X.columns if c not in X_test.columns]
# for c in missing_in_test:
#     X_test[c] = np.nan

# # drop extras and match column order exactly
# X_test = X_test[X.columns]

# # enforce dtypes consistent with training split
# for c in X_test.columns:
#     if c in cat_cols_all:
#         if not is_categorical_dtype(X_test[c]):
#             X_test[c] = X_test[c].astype("category")
#     else:
#         X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype("float32")


# # --------------------- Predict and write submission ---------------------
# test_scores = best_pipe.predict_proba(X_test)[:, 1]
# pred_series = pd.Series(test_scores, index=case_ids, name="score")

# try:
#     df_subm = pd.read_csv(ROOT / "sample_submission.csv")
# except Exception:
#     df_subm = pd.read_csv("sample_submission.csv")

# df_subm = df_subm.set_index("case_id")
# df_subm["score"] = pred_series.reindex(df_subm.index).fillna(0.0)
# df_subm.to_csv("submission.csv")

# print("[SUBMISSION] Saved submission.csv")
# print(df_subm.head())


# ---------------- Custom Voting Model ----------------
# It averages predictions from multiple trained estimators (e.g., LGBM models).
class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        # No fitting needed here; models are already trained.
        return self

    def predict(self, X):
        # Average the predicted values from all models.
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        # Average the predicted probabilities from all models.
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)


# ---------------- Data Preparation ----------------
# Define training features (X), target labels (y), and group info (weeks)
X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
y = df_train["target"]
weeks = df_train["WEEK_NUM"]

# ---------------- Cross-validation setup ----------------
# StratifiedGroupKFold ensures balanced label distribution and that each week stays within a single fold (to prevent leakage)
cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

# ---------------- LightGBM model parameters ----------------
# LGBM model parameters after the cross-validation process
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 10,
    "learning_rate": 0.05,
    "max_bin": 255,
    "n_estimators": 1200,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees": True,
    "num_leaves": 64,
    "device": "gpu", 
}

# ---------------- Model training loop ----------------
fitted_models = []
cv_scores = []

# Loop through each fold for training and validation
for idx_train, idx_valid in cv.split(X, y, groups=weeks):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

    print("Validation week range: ", (weeks.iloc[idx_valid].min(), weeks.iloc[idx_valid].max()))

    # Initialize and train the LightGBM model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)]  # Early stop if no improvement
    )

    # Save the trained model
    fitted_models.append(model)

    # Evaluate AUC on the validation fold
    y_pred_valid = model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_pred_valid)
    cv_scores.append(auc_score)

# ---------------- Ensemble & Results ----------------
# Combine all fold models into a simple average voting model
model = VotingModel(fitted_models)

# Print all CV fold scores and their mean
print("CV AUC scores: ", cv_scores)
print("Average CV AUC score: ", sum(cv_scores) / len(cv_scores))


X_test = df_test.drop(columns=["WEEK_NUM"])
X_test = X_test.set_index("case_id")

lgb_pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)


# Prepare for submission
df_subm = pd.read_csv(ROOT / "sample_submission.csv")
df_subm = df_subm.set_index("case_id")

df_subm["score"] = lgb_pred

print("Check null: ", df_subm["score"].isnull().any())


df_subm.head()


df_subm.to_csv("submission.csv")




