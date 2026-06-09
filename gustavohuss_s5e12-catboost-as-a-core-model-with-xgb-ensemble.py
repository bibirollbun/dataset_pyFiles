import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool
import time
import itertools


# Paths
TRAIN_PATH = '/kaggle/input/playground-series-s5e12/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e12/test.csv'
ORIG_PATH = '/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv'

# Constants
TARGET = 'diagnosed_diabetes'
ID_COL = 'id'
RANDOM_STATE = 42
SEEDS = [42,72,144]
N_SPLITS = 10


# Strategy toggles 
CFG = {
    
    "use_orig_stats": True,  

    "use_rank_features": False,
    "use_log_features": False,
    "use_interactions": True ,
   
  
    "use_sample_weight": False,    
    "tail_weight": 1.0,   


    "print_feature_manifest": True,
}

# Hardware check
task_device = 'GPU' if torch.cuda.is_available() else 'CPU'
print(f" Environment Check: Device set to: {task_device}")


def engineer_features(df, numeric_cols):
    df = df.copy()

    # 1) Rank features (very safe & strong)
    if CFG["use_rank_features"]:
        for c in numeric_cols:
            df[f"{c}_rank"] = df[c].rank(pct=True)

    # 2) Log features (only if values are non-negative / skewed)
    if CFG["use_log_features"]:
        for c in numeric_cols:
            # safe log1p (clip at 0)
            df[f"{c}_log1p"] = np.log1p(df[c].clip(lower=0))

    # 3) Small interaction set (keep it tiny)
    if CFG["use_interactions"]:
        # choose only a few strong candidates once you see feature importance
        pairs = [("age", "bmi"), ("glucose", "bmi")]
        for a,b in pairs:
            if a in df.columns and b in df.columns:
                df[f"{a}_x_{b}"] = df[a] * df[b]
                df[f"{a}_div_{b}"] = df[a] / (df[b] + 1e-6)

    return df

def add_orig_stats(orig_df, df_to_update, numeric_cols):
    """Map diabetes risk statistics from the original dataset."""
    for col in numeric_cols:
        if col in [ID_COL, TARGET] or col not in orig_df.columns: continue
        
        mean_map = orig_df.groupby(col)[TARGET].mean()
        count_map = orig_df.groupby(col).size()
        global_mean = orig_df[TARGET].mean()
        
        df_to_update[f'orig_mean_{col}'] = df_to_update[col].map(mean_map).fillna(global_mean)
        df_to_update[f'orig_count_{col}'] = df_to_update[col].map(count_map).fillna(0).astype(int)
    return df_to_update

def get_feature_lists(train_df):
    """
    Returns (feature_cols, cat_feature_cols, numeric_feature_cols)
    after FE is applied.
    """
    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET]]
    cat_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, cat_cols, num_cols


# =========================
# 5. Load Data & Preprocess (Yours)
# =========================

print("ğŸ“‚ Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
orig  = pd.read_csv(ORIG_PATH)

# Identify base numeric columns from TRAIN (excluding id/target)
base_numeric_cols = (
    train.select_dtypes(include=[np.number])
         .columns
         .tolist()
)
base_numeric_cols = [c for c in base_numeric_cols if c not in [ID_COL, TARGET]]

# Optional: map orig stats (strategy toggle)
if CFG["use_orig_stats"]:
    print("ğŸ§© Applying orig_stats mapping...")
    train = add_orig_stats(orig, train, base_numeric_cols)
    test  = add_orig_stats(orig, test,  base_numeric_cols)

# Recompute numeric cols after orig_stats adds new numeric columns
numeric_cols_after_orig = (
    train.select_dtypes(include=[np.number])
         .columns
         .tolist()
)
numeric_cols_after_orig = [c for c in numeric_cols_after_orig if c not in [ID_COL, TARGET]]

print("ğŸ›  Applying feature engineering...")
train = engineer_features(train, numeric_cols_after_orig)
test  = engineer_features(test,  numeric_cols_after_orig)

# Final feature lists
FEATURES, CAT_FEATURES, NUM_FEATURES = get_feature_lists(train)

if CFG["print_feature_manifest"]:
    print("\nğŸ“Œ Feature Manifest")
    print(f"  Train shape: {train.shape} | Test shape: {test.shape}")
    print(f"  #FEATURES: {len(FEATURES)} | #NUM: {len(NUM_FEATURES)} | #CAT: {len(CAT_FEATURES)}")
    print(f"  Example features: {FEATURES[:15]}")
    if len(CAT_FEATURES) > 0:
        print(f"  Cat features: {CAT_FEATURES[:15]}")

print("âœ… Data & features ready.")



# --- Sample weights (simple, controllable) ---
def compute_weights(y):
    """
    Base weighting: inverse class frequency (good default).
    You can replace this with your own strategy later.
    """
    y = np.asarray(y)
    if not CFG["use_sample_weight"]:
        return np.ones_like(y, dtype=float)

    # inverse frequency weighting
    classes = np.unique(y)
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    freq = counts / counts.sum()
    inv = 1.0 / np.clip(freq, 1e-12, None)

    weight_map = {c: inv[i] for i, c in enumerate(classes)}
    w = np.array([weight_map[v] for v in y], dtype=float)

    
    if CFG.get("tail_weight", 1.0) != 1.0:
        
        pass

    return w


# --- Model Hyperparameters (baseline for CFG) ---
params = {
    "learning_rate": 0.06,
    "depth": 5,                    
    "l2_leaf_reg": 10,
    
    "random_strength": 1,
    "iterations": 4000,
    "early_stopping_rounds": 100,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "task_type": task_device,
    "logging_level": "Silent",
    "metric_period": 200,
    "allow_writing_files": False
}


# --- CV setup ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_matrix  = np.zeros((len(SEEDS), len(train)))
test_matrix = np.zeros((len(SEEDS), len(test)))

fold_scores_by_seed = []

print("Training started...")
for s, seed in enumerate(SEEDS):
    print(f"\nğŸŒ± Seed {seed}")
    seed_test_preds = np.zeros(len(test))
    fold_scores = []

    y_all = train[TARGET].values
    weights_all = compute_weights(y_all)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train[FEATURES], y_all), 1):
        X_tr  = train.iloc[trn_idx][FEATURES]
        y_tr  = y_all[trn_idx]
        w_tr  = weights_all[trn_idx]

        X_val = train.iloc[val_idx][FEATURES]
        y_val = y_all[val_idx]

        # Pools
        train_pool = Pool(X_tr, y_tr, cat_features=CAT_FEATURES, weight=w_tr)
        val_pool   = Pool(X_val, y_val, cat_features=CAT_FEATURES)

        # Train
        model = CatBoostClassifier(**params, random_seed=seed)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            verbose=False
        )

        # Predict OOF
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_matrix[s, val_idx] = val_pred

        # Predict test (fold average)
        seed_test_preds += model.predict_proba(test[FEATURES])[:, 1] / N_SPLITS

        # Fold score
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"  Fold {fold:02d} AUC: {fold_auc:.5f}")

    test_matrix[s] = seed_test_preds

    seed_oof_auc = roc_auc_score(train[TARGET], oof_matrix[s])
    print(f"  ğŸ�† Seed {seed} OOF AUC: {seed_oof_auc:.5f}")
    print(f"  Fold AUC meanÂ±std: {np.mean(fold_scores):.5f} Â± {np.std(fold_scores):.5f}")

    fold_scores_by_seed.append(fold_scores)

# Final averages across seeds
final_oof  = oof_matrix.mean(axis=0)
final_test = test_matrix.mean(axis=0)

final_auc = roc_auc_score(train[TARGET], final_oof)
print(f"\nğŸ�† FINAL SEED-AVERAGED OOF AUC: {final_auc:.6f}")


def eval_config(cfg_params, splits, seeds=SEEDS):
    y_all = train[TARGET].values
    fold_aucs_all = []
    seed_aucs = []
    
    for seed in seeds:
        oof = np.zeros(len(train))
        fold_aucs = []
        
        for fold, (trn_idx, val_idx) in enumerate(splits, 1):
            X_tr = train.iloc[trn_idx][FEATURES]
            y_tr = y_all[trn_idx]
            X_va = train.iloc[val_idx][FEATURES]
            y_va = y_all[val_idx]

            train_pool = Pool(X_tr, y_tr, cat_features=CAT_FEATURES)
            val_pool   = Pool(X_va, y_va, cat_features=CAT_FEATURES)

            params = BASE_PARAMS.copy()
            params.update(cfg_params)

            model = CatBoostClassifier(**params, random_seed=seed)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            val_pred = model.predict_proba(X_va)[:, 1]
            oof[val_idx] = val_pred

            fold_auc = roc_auc_score(y_va, val_pred)
            fold_aucs.append(fold_auc)

        seed_auc = roc_auc_score(y_all, oof)
        seed_aucs.append(seed_auc)
        fold_aucs_all.append((np.mean(fold_aucs), np.std(fold_aucs)))

    return {
        "seed_mean_auc": float(np.mean(seed_aucs)),
        "seed_std_auc": float(np.std(seed_aucs)) if len(seed_aucs) > 1 else 0.0,
        "fold_mean_auc": float(np.mean([m for m, s in fold_aucs_all])),
        "fold_std_auc": float(np.mean([s for m, s in fold_aucs_all])),
    }


splits = list(skf.split(train[FEATURES], train[TARGET]))

# Base params (keep your working defaults here)
BASE_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    task_type="GPU",
    verbose=False,
    iterations=5000,             
    od_type="Iter",
    od_wait=200,                
)

# Grid to test
GRID = {
    "depth": [4,5,6],
    "learning_rate": [0.03, 0.06],
    "l2_leaf_reg": [5,10,20],
    "random_strength": [0.2,0.5,1],
}

results = []
keys = list(GRID.keys())
for vals in itertools.product(*[GRID[k] for k in keys]):
    cfg = dict(zip(keys, vals))
    t0 = time.time()
    out = eval_config(cfg, splits, seeds=SEEDS)
    dt = time.time() - t0
    
    row = {**cfg, **out, "minutes": dt/60}
    results.append(row)
    print(f"âœ… {cfg} | OOF={out['seed_mean_auc']:.5f} | fold_std={out['fold_std_auc']:.5f} | {dt/60:.1f} min")

df = pd.DataFrame(results).sort_values(["seed_mean_auc", "fold_std_auc"], ascending=[False, True])
display(df.head(15))

df.to_csv("/kaggle/working/catboost_hparam_sweep.csv", index=False)
print("ğŸ’¾ Saved: /kaggle/working/catboost_hparam_sweep.csv")


params = {
    "learning_rate": 0.06,
    "depth": 5,                    
    "l2_leaf_reg": 10,
    "random_strength": 1,
    "iterations": 4000,
    "early_stopping_rounds": 100,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "task_type": task_device,
    "logging_level": "Silent",
    "allow_writing_files": False
}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_matrix  = np.zeros((len(SEEDS), len(train)))
test_matrix = np.zeros((len(SEEDS), len(test)))

fold_scores_by_seed = []

print("Training started...")
for s, seed in enumerate(SEEDS):
    print(f"\nğŸŒ± Seed {seed}")
    seed_test_preds = np.zeros(len(test))
    fold_scores = []

    y_all = train[TARGET].values
    weights_all = compute_weights(y_all)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train[FEATURES], y_all), 1):
        X_tr  = train.iloc[trn_idx][FEATURES]
        y_tr  = y_all[trn_idx]
        w_tr  = weights_all[trn_idx]

        X_val = train.iloc[val_idx][FEATURES]
        y_val = y_all[val_idx]

        # Pools
        train_pool = Pool(X_tr, y_tr, cat_features=CAT_FEATURES, weight=w_tr)
        val_pool   = Pool(X_val, y_val, cat_features=CAT_FEATURES)
        # Train
        model = CatBoostClassifier(**params, random_seed=seed)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            verbose=False
        )

        # Predict OOF
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_matrix[s, val_idx] = val_pred

        # Predict test (fold average)
        seed_test_preds += model.predict_proba(test[FEATURES])[:, 1] / N_SPLITS

        # Fold score
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"  Fold {fold:02d} AUC: {fold_auc:.5f}")

    test_matrix[s] = seed_test_preds

    seed_oof_auc = roc_auc_score(train[TARGET], oof_matrix[s])
    print(f"  ğŸ�† Seed {seed} OOF AUC: {seed_oof_auc:.5f}")
    print(f"  Fold AUC meanÂ±std: {np.mean(fold_scores):.5f} Â± {np.std(fold_scores):.5f}")

    fold_scores_by_seed.append(fold_scores)

# Final averages across seeds
final_oof  = oof_matrix.mean(axis=0)
final_test = test_matrix.mean(axis=0)
final_auc = roc_auc_score(train[TARGET], final_oof)
print(f"\nğŸ�† FINAL SEED-AVERAGED OOF AUC: {final_auc:.6f}")


# =========================
# 9. Feature Importance 
# =========================

print("\nğŸ“Š Feature Importance ")

# --- Safety checks ---
if "FEATURES" not in globals():
    raise NameError("â�Œ FEATURES not found. Run preprocessing first.")
if "model" not in globals():
    raise NameError("â�Œ model not found. Run training first.")

if "fold_importances" in globals() and len(fold_importances) > 0:
    
    imp_df = (pd.concat(fold_importances, ignore_index=True)
                .groupby("feature", as_index=False)["importance"]
                .mean()
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))
    print("âœ… Using fold-averaged feature importance across CV.")
else:
    # last trained model (still useful)
    importances = model.get_feature_importance(type="PredictionValuesChange")
    imp_df = (pd.DataFrame({"feature": FEATURES, "importance": importances})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    
# --- Helper: group features for narrative + pruning ---
def group_name(feat: str) -> str:
    if feat.startswith("orig_mean_") or feat.startswith("orig_count_"):
        return "orig_stats"
    if feat.endswith("_rank"):
        return "rank_features"
    if feat.endswith("_log1p"):
        return "log_features"
    if "_x_" in feat or "_div_" in feat:
        return "interactions"
    return "base_features"


imp_df["group"] = imp_df["feature"].apply(group_name)

print(f"\nâœ… #features: {len(imp_df)}")
print("\nğŸ”� Top 15 features:")
display(imp_df.head(15))

# Save
imp_df.to_csv("/kaggle/working/feature_importance.csv", index=False)
print("\nğŸ’¾ Saved: /kaggle/working/feature_importance.csv")

# --- 2) Plot Top-N ---
TOP_N = 30
top_df = imp_df.head(TOP_N).iloc[::-1]

plt.figure(figsize=(10, 10))
plt.barh(top_df["feature"], top_df["importance"])
plt.title(f"Top {TOP_N} Feature Importances")
plt.xlabel("Importance (PredictionValuesChange)")
plt.tight_layout()
plt.show()
# --- 3) Grouped importance ---
group_df = (imp_df.groupby("group", as_index=False)["importance"]
              .sum()
              .sort_values("importance", ascending=False))

print("\nğŸ“¦ Total Importance by Feature Group:")
display(group_df)

plt.figure(figsize=(8, 4))
plt.bar(group_df["group"], group_df["importance"])
plt.title("Total Importance by Feature Group")
plt.ylabel("Total Importance")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# --- 4) Dead features (pruning candidates) ---
ZERO_THRESH = 1e-9
dead = imp_df[imp_df["importance"] <= ZERO_THRESH]
print(f"\n Near-zero importance features: {len(dead)}")
if len(dead) > 0:
    display(dead.head(30))


from xgboost import XGBClassifier


if "CAT_FEATURES" in globals() and len(CAT_FEATURES) > 0:
    for c in CAT_FEATURES:
        if c in train.columns:
            train[c] = train[c].astype("category")
        if "test" in globals() and c in test.columns:
            test[c] = test[c].astype("category")

XGB_PARAMS = dict(
    n_estimators=8000,           
    learning_rate=0.03,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_lambda=2.0,
    reg_alpha=0.0,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",           
    enable_categorical=True,    
    n_jobs=-1,
)
XGB_PARAMS["device"] = "cuda"
def train_xgb_oof(train_df, test_df, features, target, splits, params, seed=42, verbose=200):
    oof = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr = train_df.iloc[tr_idx][features]
        y_tr = train_df.iloc[tr_idx][target]

        X_va = train_df.iloc[va_idx][features]
        y_va = train_df.iloc[va_idx][target]

        model = XGBClassifier(**params, random_state=seed + fold)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=verbose,
            early_stopping_rounds=200
        )

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(test_df[features])[:, 1] / len(splits)

        fold_auc = roc_auc_score(y_va, oof[va_idx])
        print(f"Fold {fold} AUC: {fold_auc:.5f} | best_iteration: {model.get_booster().best_iteration}")

    full_auc = roc_auc_score(train_df[target], oof)
    print(f"\nOOF AUC: {full_auc:.5f}")
    return oof, test_pred

# ------------------------------------------------------------------
# 4) Run
# ------------------------------------------------------------------
splits = list(skf.split(train[FEATURES], train[TARGET]))
xgb_oof, xgb_test = train_xgb_oof(train, test, FEATURES, TARGET, splits, XGB_PARAMS)
 


# =========================
# Cell 1 â€” Blend (CatBoost + XGBoost) using the notebook's real variables
# =========================
from sklearn.metrics import roc_auc_score
import numpy as np

# Labels (from your notebook config)
y_true = train[TARGET].values

# CatBoost preds from your notebook
cb_oof  = np.asarray(final_oof).ravel()
cb_test = np.asarray(final_test).ravel()

# XGBoost preds from your notebook
xgb_oof_  = np.asarray(xgb_oof).ravel()
xgb_test_ = np.asarray(xgb_test).ravel()

# --- sanity checks ---
assert cb_oof.shape == (len(train),), f"final_oof shape {cb_oof.shape} != ({len(train)},)"
assert xgb_oof_.shape == (len(train),), f"xgb_oof shape {xgb_oof_.shape} != ({len(train)},)"
assert cb_test.shape == (len(test),), f"final_test shape {cb_test.shape} != ({len(test)},)"
assert xgb_test_.shape == (len(test),), f"xgb_test shape {xgb_test_.shape} != ({len(test)},)"

# Optional: single model AUCs (nice for notebook)
auc_cb  = roc_auc_score(y_true, cb_oof)
auc_xgb = roc_auc_score(y_true, xgb_oof_)
print(f"CatBoost OOF AUC: {auc_cb:.5f}")
print(f"XGBoost  OOF AUC: {auc_xgb:.5f}")

def best_blend_weight(oof_a, oof_b, y, step=0.005):
    grid = np.arange(0.0, 1.0 + step, step)
    best_auc, best_w = -1.0, 0.5
    for w in grid:
        oof_blend = w * oof_a + (1.0 - w) * oof_b
        auc = roc_auc_score(y, oof_blend)
        if auc > best_auc:
            best_auc, best_w = auc, w
    return best_w, best_auc

w_cb, auc_ens = best_blend_weight(cb_oof, xgb_oof_, y_true, step=0.005)
print(f"Best blend weight (CatBoost): {w_cb:.3f} | Ensemble OOF AUC: {auc_ens:.5f}")

oof_ens  = w_cb * cb_oof  + (1.0 - w_cb) * xgb_oof_
test_ens = w_cb * cb_test + (1.0 - w_cb) * xgb_test_

# Safety: keep probabilities in [0, 1]
oof_ens  = np.clip(oof_ens, 0.0, 1.0)
test_ens = np.clip(test_ens, 0.0, 1.0)



# =========================
# Submission (no sample_submission.csv)
# =========================
import pandas as pd
import numpy as np

# ---- REQUIRED ----
# test      : test DataFrame
# test_ens : blended predictions (1D, len(test))

# Identify ID column safely
if "id" in test.columns:
    ID_COL = "id"
else:
    raise ValueError("â�Œ No 'id' column found in test dataframe")

# Target column name required by the competition
TARGET_COL = "diagnosed_diabetes"   # <-- change ONLY if competition specifies a different name

# Final submission dataframe
sub = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET_COL: np.clip(test_ens, 0.0, 1.0)
})

out_path = "/kaggle/working/submission_ensemble.csv"
sub.to_csv(out_path, index=False)

print("âœ… Saved:", out_path)
sub.head()



