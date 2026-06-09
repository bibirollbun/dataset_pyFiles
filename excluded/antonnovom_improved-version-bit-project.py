import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef

from catboost import CatBoostClassifier, Pool

DATA_PATH = "/kaggle/input/playground-series-s4e8/"
N_SPLITS = 3
USE_GPU = True

MODEL_SPECS = [
    {"seed": 1,  "params": dict(depth=8, l2_leaf_reg=6.0, random_strength=1.0, bagging_temperature=0.8)},
    {"seed": 7,  "params": dict(depth=9, l2_leaf_reg=10.0, random_strength=2.0, bagging_temperature=0.6)},
    {"seed": 42, "params": dict(depth=7, l2_leaf_reg=4.0, random_strength=1.5, bagging_temperature=1.0)},
]


JITTER_STRENGTH = 0.01


N_BINS = 48

RARE_MIN_COUNT = 50

MAX_HASHED_CROSSES = 8


train = pd.read_csv(DATA_PATH + "train.csv")
test  = pd.read_csv(DATA_PATH + "test.csv")

y = (train["class"] == "p").astype(np.int8)
train_ids = train["id"].values
test_ids  = test["id"].values

X = train.drop(columns=["class", "id"])
X_test = test.drop(columns=["id"])

def replace_question_marks(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].replace("?", np.nan)
    return df

def add_missing_flags_and_fill(df, cat_cols):
    df = df.copy()
    # флаги пропусков
    for c in cat_cols:
        df[c + "__isna"] = df[c].isna().astype(np.int8)
        df[c] = df[c].fillna("missing").astype("string")
    # общий счётчик пропусков (по исходным cat)
    df["missing_count"] = df[[c + "__isna" for c in cat_cols]].sum(axis=1).astype(np.int16)
    return df

def group_rare_categories(train_df, test_df, cat_cols, min_count=50):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in cat_cols:
        vc = train_df[c].value_counts(dropna=False)
        rare_vals = set(vc[vc < min_count].index.tolist())
        # rare в train
        train_df[c] = train_df[c].where(~train_df[c].isin(rare_vals), "rare")
        # unseen/rare в test
        seen_vals = set(vc.index.tolist())
        test_df[c] = test_df[c].where(test_df[c].isin(seen_vals), "unseen")
        test_df[c] = test_df[c].where(~test_df[c].isin(rare_vals), "rare")
    return train_df, test_df

def add_freq_enc(train_df, test_df, cat_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    n = len(train_df)
    for c in cat_cols:
        vc = train_df[c].value_counts(dropna=False)
        train_df[c + "__freq"] = train_df[c].map(vc).fillna(0).astype(np.float32)
        test_df[c + "__freq"]  = test_df[c].map(vc).fillna(0).astype(np.float32)

        train_df[c + "__freq_norm"] = (train_df[c + "__freq"] / n).astype(np.float32)
        test_df[c + "__freq_norm"]  = (test_df[c + "__freq"] / n).astype(np.float32)

        train_df[c + "__logcnt"] = np.log1p(train_df[c + "__freq"]).astype(np.float32)
        test_df[c + "__logcnt"]  = np.log1p(test_df[c + "__freq"]).astype(np.float32)
    return train_df, test_df

def add_numeric_bins(train_df, test_df, num_cols, n_bins=48):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in num_cols:
        # квантили по train
        qs = np.quantile(train_df[c].values, np.linspace(0, 1, n_bins + 1))
        qs = np.unique(qs)
        # если мало уникальных — пропускаем биннинг
        if len(qs) <= 3:
            continue
        # digitize: 0..(len(qs)-2)
        cut_points = qs[1:-1]
        train_df[c + "__bin"] = np.digitize(train_df[c].values, cut_points, right=True).astype(np.int16)
        test_df[c + "__bin"]  = np.digitize(test_df[c].values,  cut_points, right=True).astype(np.int16)
    return train_df, test_df

def add_numeric_transforms(train_df, test_df, num_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in num_cols:
        train_df[c + "__log1p"] = np.log1p(np.maximum(train_df[c].values, 0)).astype(np.float32)
        test_df[c + "__log1p"]  = np.log1p(np.maximum(test_df[c].values,  0)).astype(np.float32)

    def has(col): return col in train_df.columns
    if has("stem-height") and has("stem-width"):
        train_df["stem_hw"] = (train_df["stem-height"] * train_df["stem-width"]).astype(np.float32)
        test_df["stem_hw"]  = (test_df["stem-height"]  * test_df["stem-width"]).astype(np.float32)

        train_df["stem_h_div_w"] = (train_df["stem-height"] / (train_df["stem-width"] + 1e-6)).astype(np.float32)
        test_df["stem_h_div_w"]  = (test_df["stem-height"]  / (test_df["stem-width"]  + 1e-6)).astype(np.float32)

    if has("cap-diameter") and has("stem-height"):
        train_df["cap_div_stem_h"] = (train_df["cap-diameter"] / (train_df["stem-height"] + 1e-6)).astype(np.float32)
        test_df["cap_div_stem_h"]  = (test_df["cap-diameter"]  / (test_df["stem-height"]  + 1e-6)).astype(np.float32)

    return train_df, test_df

def add_hashed_crosses(train_df, test_df, cat_cols, max_pairs=8):
    """
    Делает “комбо”-фичи, но без строк:
    хэш от пары значений -> int64, и мы скажем CatBoost, что это categorical.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    nun = train_df[cat_cols].nunique()
    chosen = nun.sort_values().index.tolist()[:10]  # топ-10 по низкой кардинальности
    pairs = []
    for i in range(len(chosen)):
        for j in range(i+1, len(chosen)):
            pairs.append((chosen[i], chosen[j]))
    pairs = pairs[:max_pairs]

    for a, b in pairs:
        name = f"{a}__X__{b}"
        train_df[name] = pd.util.hash_pandas_object(train_df[[a, b]], index=False).astype(np.int64)
        test_df[name]  = pd.util.hash_pandas_object(test_df[[a, b]],  index=False).astype(np.int64)

    return train_df, test_df

# --- определяем исходные типы
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 1) ? -> NaN
X = replace_question_marks(X, cat_cols)
X_test = replace_question_marks(X_test, cat_cols)

# 2) numeric: median impute (сразу, чтобы дальше bins/трансформы работали)
for c in num_cols:
    med = X[c].median()
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

# 3) missing flags + fill cats
X = add_missing_flags_and_fill(X, cat_cols)
X_test = add_missing_flags_and_fill(X_test, cat_cols)

# 4) rare grouping
X, X_test = group_rare_categories(X, X_test, cat_cols, min_count=RARE_MIN_COUNT)

# 5) freq encodings
X, X_test = add_freq_enc(X, X_test, cat_cols)

# 6) numeric bins + transforms + interactions
X, X_test = add_numeric_bins(X, X_test, num_cols, n_bins=N_BINS)
X, X_test = add_numeric_transforms(X, X_test, num_cols)

# 7) hashed crosses (пара штук)
X, X_test = add_hashed_crosses(X, X_test, cat_cols, max_pairs=MAX_HASHED_CROSSES)

# --- список “категориальных” для CatBoost:
# исходные cat (string) + бины (int16) + hashed crosses (int64) считаем категориальными
bin_cols = [c for c in X.columns if c.endswith("__bin")]
cross_cols = [c for c in X.columns if "__X__" in c]
cat_like_cols = cat_cols + bin_cols + cross_cols

# индексы cat-фич
cat_idx = [X.columns.get_loc(c) for c in cat_like_cols if c in X.columns]

# jitter только на “реальных” num_cols (не трогаем freq/log/bin/cross)
NUMERIC_FOR_JITTER = [c for c in num_cols if c in X.columns]


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof = np.zeros(len(X), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)

def jitter_inplace(df, cols, strength, seed):
    rng = np.random.default_rng(seed)
    for c in cols:
        std = df[c].std()
        if std and np.isfinite(std) and std > 0:
            df[c] = df[c].values + rng.normal(0, strength * std, size=len(df))
    return df

for spec_i, spec in enumerate(MODEL_SPECS, start=1):
    seed = spec["seed"]
    params = spec["params"]

    oof_m = np.zeros(len(X), dtype=np.float32)
    test_m = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_va = y.iloc[va_idx].copy()

        # jitter только train
        X_tr = jitter_inplace(X_tr, NUMERIC_FOR_JITTER, JITTER_STRENGTH, seed + fold)

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        val_pool   = Pool(X_va, y_va, cat_features=cat_idx)

        model = CatBoostClassifier(
            iterations=4000,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=seed,
            od_type="Iter",
            od_wait=120,
            allow_writing_files=False,
            max_ctr_complexity=2,  # пары категорий внутри CatBoost (вместо тонны явных кроссов)
            task_type=("GPU" if USE_GPU else "CPU"),
            **params
        )

        model.fit(train_pool, eval_set=val_pool, verbose=200)

        oof_m[va_idx] = model.predict_proba(X_va)[:, 1].astype(np.float32)
        test_m += model.predict_proba(X_test)[:, 1].astype(np.float32) / N_SPLITS

        # уборка65
        del X_tr, X_va, y_tr, y_va, train_pool, val_pool, model
        gc.collect()

    # усредняем по моделям
    oof += oof_m / len(MODEL_SPECS)
    test_pred += test_m / len(MODEL_SPECS)

ths = np.linspace(0.01, 0.99, 981)
mccs = [matthews_corrcoef(y, (oof >= t).astype(np.int8)) for t in ths]
best_t = float(ths[int(np.argmax(mccs))])
best_mcc = float(np.max(mccs))

print(f"OOF MCC = {best_mcc:.6f} at threshold t = {best_t:.4f}")


sub = pd.DataFrame({
    "id": test_ids,
    "class": np.where(test_pred >= best_t, "p", "e")
})
sub.to_csv("submission_ensemble_oof.csv", index=False)
print(sub.head())


