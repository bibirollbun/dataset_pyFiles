import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import warnings, os, gc

warnings.filterwarnings("ignore")
np.random.seed(42)


TRAIN_PATH = "/kaggle/input/playground-series-s5e8/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e8/test.csv"
SUB_PATH   = "/kaggle/input/playground-series-s5e8/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH, index_col="id")
test  = pd.read_csv(TEST_PATH,  index_col="id")

TARGET = "y"
assert TARGET in train.columns, f"Target '{TARGET}' not found in training data!"

#Quick shape + trget distribution checks
print(f"train shape: {train.shape} | test shape: {test.shape}")
print("Target distribution:\n", train[TARGET].value_counts(normalize=True).rename("ratio"))


X = train.drop(columns=[TARGET])
y = train[TARGET].astype(int)

#Categorical detection by dtype
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"#numeric: {len(num_cols)} | #categorical: {len(cat_cols)}")
if len(cat_cols) > 0:
#(Optional) ensure categorical dtype is 'category' for speed/memory
    for c in cat_cols:
        if X[c].dtype.name != "category":
            X[c] = X[c].astype("category")
            test[c] = test[c].astype("category")


const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if const_cols:
    print(f"Dropping constant columns: {const_cols}")
    X = X.drop(columns=const_cols)
    test = test.drop(columns=const_cols)
    #keep cat_cols/num_cols in sync
    cat_cols = [c for c in cat_cols if c not in const_cols]
    num_cols = [c for c in num_cols if c not in const_cols]

#Add _isna flags for numeric columns with missing values in train or test
added_flags = []
for c in num_cols:
    if X[c].isna().any() or test[c].isna().any():
        flag = f"{c}_isna"
        X[flag] = X[c].isna().astype(np.int8)
        test[flag] = test[c].isna().astype(np.int8)
        added_flags.append(flag)

if added_flags:
    print(f"Added missing flags: {len(added_flags)} columns")


def make_pool(df, y=None):
    return Pool(
        data=df,
        label=y,
        cat_features=cat_cols if len(cat_cols) else None
    )


N_SPLITS = 5
SEEDS = [0, 1, 2]
AUC_EVAL_PERIOD = 1
BASE_PARAMS = {
"task_type": "GPU",
"devices": "0:1",              #to use both T4 GPUs
"loss_function": "Logloss",
"eval_metric": "Logloss",
"custom_metric": ["AUC"],      #just for logging (computed on CPU) it will return warning below, no problem
"use_best_model": False,       #IMPORTANT!!!: do not auto-shrink to Logloss-best
"iterations": 8000,            #cap; we will scan AUC over all built trees
"learning_rate": 0.055,        #a bit higher to need fewer trees
"depth": 7,
"border_count": 128,
"bootstrap_type": "Bernoulli",
"subsample": 0.66,
"sampling_frequency": "PerTree",
"random_strength": 0.8,
"l2_leaf_reg": 8.0,
"auto_class_weights": "Balanced",
"allow_writing_files": False,
"verbose": 100,
"thread_count": -1
}


oof_rank_sum = np.zeros(len(X), dtype=float)
test_rank_sum = np.zeros(len(test), dtype=float)
seed_oof_scores = []

for seed in SEEDS:
    print("\n" + "="*60)
    print(f"Starting SEED {seed}")
    print("="*60)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(X), dtype=float)
    test_probs = np.zeros(len(test), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n---- Seed {seed} | Fold {fold}/{N_SPLITS} ----")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = CatBoostClassifier(**{**BASE_PARAMS, "random_seed": seed})
        model.fit(
            make_pool(X_tr, y_tr),
            eval_set=make_pool(X_va, y_va)
        )

        #pick the exact iteration that maximizes AUC on the validation set
        total_trees = model.tree_count_
        mets = model.eval_metrics(
            make_pool(X_va, y_va),
            metrics=["AUC"],
            ntree_start=0,
            ntree_end=total_trees,   # evaluate over all built trees
            eval_period=AUC_EVAL_PERIOD
        )
        auc_vals = mets["AUC"]                      #length==total_trees/AUC_EVAL_PERIOD
        best_off = int(np.argmax(auc_vals))         #0-based within auc_vals
        
        #map back to absolute iteration index
        best_auc_it = (best_off + 1) * AUC_EVAL_PERIOD
        print(f"AUC-best iteration: {best_auc_it} / {total_trees} | Val AUC: {auc_vals[best_off]:.6f}")

        #keep trees up to and including AUC-best
        model.shrink(ntree_end=best_auc_it, ntree_start=0)

        #predict with the AUC-aligned model
        oof_probs[va_idx] = model.predict_proba(make_pool(X_va))[:, 1]
        test_probs += model.predict_proba(make_pool(test))[:, 1] / N_SPLITS
        del X_tr, X_va, y_tr, y_va, model, mets, auc_vals
        gc.collect()

    #seed-level OOF AUC
    seed_auc = roc_auc_score(y, oof_probs)
    seed_oof_scores.append((seed, seed_auc))
    print(f"\nSeed {seed} OOF AUC (probs): {seed_auc:.6f}")

    #rank-average for AUC robustness
    oof_rank = pd.Series(oof_probs).rank(pct=True).values
    test_rank = pd.Series(test_probs).rank(pct=True).values
    oof_rank_sum += oof_rank
    test_rank_sum += test_rank

#finalize rank-averaged predictions
oof_final = oof_rank_sum / len(SEEDS)
test_final = test_rank_sum / len(SEEDS)
final_oof_auc = roc_auc_score(y, oof_final)

print("\n" + "="*60)
print("SEED OOF AUC (raw probs) per seed:", seed_oof_scores)
print(f"Final OOF AUC (rank-avg across seeds): {final_oof_auc:.6f}")
print("="*60)


#submission
sub = pd.read_csv(SUB_PATH)
sub["y"] = test_final.astype(float)
sub.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")
print(sub.head(5))




