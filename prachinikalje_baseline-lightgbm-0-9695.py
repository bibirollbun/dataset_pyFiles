# ====================================================
# ðŸ“˜ Bank Term Deposit Subscription Prediction
# Kaggle Playground Series 2025
# ====================================================
import os, gc, sys, math, json, logging, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectFromModel

import lightgbm as lgb

# -------------------- CONFIG --------------------
CFG = {
    "SEED": 42,
    "N_SPLITS": 5,
    "TARGET": "y",
    "ID": "id",
    "PATH_TRAIN": "/kaggle/input/playground-series-s5e8/train.csv",
    "PATH_TEST": "/kaggle/input/playground-series-s5e8/test.csv",
    "PATH_SUB": "/kaggle/input/playground-series-s5e8/sample_submission.csv",
    "MODEL_DIR": "./models",
    "USE_CLASS_WEIGHT": True,        # helps if imbalanced
    "FS_CORR_THRESH": 0.98,          # drop highly correlated numeric features
    "FS_VARIANCE_THRESH": 0.0,       # drop constant / quasi-constant after ohe
    "LGB_PARAMS": {
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    },
    "EARLY_STOPPING_ROUNDS": 200,
    "VERBOSE_EVAL": 200
}
os.makedirs(CFG["MODEL_DIR"], exist_ok=True)

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)
def set_seed(seed=42):
    np.random.seed(seed)

set_seed(CFG["SEED"])

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int"):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = pd.to_numeric(df[col], downcast="float")
    end_mem = df.memory_usage().sum() / 1024**2
    logging.info(f"Memory reduced: {start_mem:.2f} â†’ {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% saved)")
    return df

def auc(y_true, y_pred, name=""):
    score = roc_auc_score(y_true, y_pred)
    logging.info(f"{name} ROC-AUC: {score:.5f}")
    return score



train = pd.read_csv(CFG["PATH_TRAIN"])
test  = pd.read_csv(CFG["PATH_TEST"])
sub   = pd.read_csv(CFG["PATH_SUB"])

logging.info(f"Train shape: {train.shape} | Test shape: {test.shape}")
train = reduce_memory(train)
test  = reduce_memory(test)

display(train.head(3))



train_cols = set(train.columns) - {CFG["TARGET"]}
test_cols  = set(test.columns)
assert train_cols == test_cols, "Train/Test feature mismatch!"
logging.info(f"Common features: {len(train_cols)}")



TARGET = CFG["TARGET"]; ID = CFG["ID"]
feat_cols = [c for c in train.columns if c not in [TARGET, ID]]

num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feat_cols if c not in num_cols]

logging.info(f"Numerical: {len(num_cols)} | Categorical: {len(cat_cols)}")



fig, ax = plt.subplots(figsize=(4,3))
sns.countplot(x=TARGET, data=train, ax=ax)
ax.set_title("Target distribution")
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x()+0.25, p.get_height()*1.01))
plt.show()

train[TARGET].value_counts(normalize=True).rename("ratio")



for col in num_cols[:8]:  # cap for readability; expand as needed
    fig, ax = plt.subplots(figsize=(5,3))
    sns.histplot(data=train, x=col, hue=TARGET, bins=40, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(f"{col} distribution by target")
    plt.tight_layout()
    plt.show()



for col in cat_cols[:8]:  # cap for speed
    fig, ax = plt.subplots(figsize=(6,3))
    order = train[col].value_counts().index[:15]  # top-15 categories
    sns.countplot(data=train, x=col, order=order, hue=TARGET, ax=ax)
    ax.set_title(f"{col} vs target (top-15)")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()



if len(num_cols) > 1:
    corr = train[num_cols].corr()
    fig, ax = plt.subplots(figsize=(min(12, 0.5*len(num_cols)+4), 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlations")
    plt.show()



def drop_high_corr_numeric(df: pd.DataFrame, numeric_cols, thr=0.98):
    """Return a list of numeric columns to keep after dropping highly correlated ones."""
    keep = set(numeric_cols)
    if len(numeric_cols) < 2:
        return list(keep)
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > thr)]
    keep = [c for c in numeric_cols if c not in set(to_drop)]
    logging.info(f"Dropping highly correlated numeric columns: {len(to_drop)}")
    return keep

num_cols_uncorr = drop_high_corr_numeric(train, num_cols, thr=CFG["FS_CORR_THRESH"])



numeric_features = num_cols_uncorr
categorical_features = cat_cols

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=True, with_std=True))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features)
    ],
    sparse_threshold=0.3  # likely to produce sparse matrix
)



def run_cv_with_feature_selection(
    X_df: pd.DataFrame,
    y: pd.Series,
    X_test_df: pd.DataFrame,
    preprocessor,
    cfg,
    toarray_on_fs=True
):
    skf = StratifiedKFold(n_splits=cfg["N_SPLITS"], shuffle=True, random_state=cfg["SEED"])
    oof = np.zeros(len(X_df))
    test_pred = np.zeros(len(X_test_df))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_df, y), 1):
        logging.info(f"===== Fold {fold}/{cfg['N_SPLITS']} =====")
        X_tr_df, X_va_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # 1) Fit preprocessor on training fold only
        X_tr = preprocessor.fit_transform(X_tr_df)
        X_va = preprocessor.transform(X_va_df)
        X_te = preprocessor.transform(X_test_df)

        # 2) Variance threshold on encoded space
        vt = VarianceThreshold(threshold=cfg["FS_VARIANCE_THRESH"])
        X_tr = vt.fit_transform(X_tr)
        X_va = vt.transform(X_va)
        X_te = vt.transform(X_te)

        # Convert to dense for MI + LGBM-based FS if requested
        if toarray_on_fs and hasattr(X_tr, "toarray"):
            X_tr_dense = X_tr.toarray()
            X_va_dense = X_va.toarray()
            X_te_dense = X_te.toarray()
        else:
            # skip MI if still sparse
            X_tr_dense = X_tr if isinstance(X_tr, np.ndarray) else X_tr
            X_va_dense = X_va if isinstance(X_va, np.ndarray) else X_va
            X_te_dense = X_te if isinstance(X_te, np.ndarray) else X_te

        # 3) Mutual Information (optional if dense)
        if isinstance(X_tr_dense, np.ndarray):
            mi = mutual_info_classif(X_tr_dense, y_tr, random_state=cfg["SEED"])
            mi_mask = mi > 0  # keep features that carry any MI
        else:
            mi_mask = np.ones(X_tr.shape[1], dtype=bool)  # skip MI when sparse

        # 4) Model-based FS with a small LGBM to get importances
        fs_lgb = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg["SEED"],
            n_jobs=-1
        )

        # If sparse, pass dense copy; LightGBM can accept CSR, but importances expect fitted Booster
        X_tr_fsfit = X_tr_dense if isinstance(X_tr_dense, np.ndarray) else X_tr.toarray()
        fs_lgb.fit(X_tr_fsfit, y_tr)
        imp = fs_lgb.feature_importances_
        imp_mask = imp > 0  # keep useful (non-zero) features

        # 5) Combine FS masks (union: conservative)
        keep_mask = (imp_mask) & (mi_mask)  # stricter; swap to | for more features
        # If mask is too aggressive, fallback to importance-only
        if keep_mask.sum() < max(50, int(0.01 * keep_mask.size)):
            logging.info("FS mask too strict; using importance-only mask.")
            keep_mask = imp_mask

        logging.info(f"Selected features: {keep_mask.sum()} / {keep_mask.size}")

        # Reduce matrices
        def apply_mask(M, mask):
            if hasattr(M, "toarray"):  # sparse
                M = M.toarray()
            return M[:, mask]

        X_tr_sel = apply_mask(X_tr, keep_mask)
        X_va_sel = apply_mask(X_va, keep_mask)
        X_te_sel = apply_mask(X_te, keep_mask)

        # 6) Train final LGBM on selected features
        lgbm = lgb.LGBMClassifier(**cfg["LGB_PARAMS"])
        fit_params = {
            "eval_set": [(X_va_sel, y_va)],
            "eval_metric": "auc",
            "callbacks": [lgb.early_stopping(cfg["EARLY_STOPPING_ROUNDS"]), lgb.log_evaluation(cfg["VERBOSE_EVAL"])]
        }
        if cfg["USE_CLASS_WEIGHT"]:
            # approximate weighting: inverse frequency
            w_pos = (len(y_tr) - y_tr.sum()) / y_tr.sum()
            lgbm.set_params(class_weight={0:1.0, 1:float(w_pos)})

        lgbm.fit(X_tr_sel, y_tr, **fit_params)

        # 7) Evaluate + OOF
        va_pred = lgbm.predict_proba(X_va_sel)[:, 1]
        oof[va_idx] = va_pred
        fold_auc = auc(y_va, va_pred, name=f"Fold {fold}")
        fold_scores.append(fold_auc)

        # 8) Predict test (fold-avg)
        test_pred += lgbm.predict_proba(X_te_sel)[:, 1] / cfg["N_SPLITS"]

        # Save model + FS mask metadata
        joblib.dump(
            {"model": lgbm, "vt": vt, "fs_mask": keep_mask, "preprocessor": preprocessor},
            os.path.join(cfg["MODEL_DIR"], f"lgbm_fold{fold}.pkl")
        )

        # cleanup
        del X_tr, X_va, X_te, X_tr_sel, X_va_sel, X_te_sel, X_tr_dense, X_va_dense, X_te_dense
        gc.collect()

    logging.info(f"CV Mean AUC: {np.mean(fold_scores):.5f} | Std: {np.std(fold_scores):.5f}")
    return oof, test_pred, fold_scores

X = train.drop(columns=[ID, TARGET])
y = train[TARGET]
X_test = test.drop(columns=[ID])

oof_pred, test_pred, fold_scores = run_cv_with_feature_selection(
    X_df=X, y=y, X_test_df=X_test, preprocessor=preprocessor, cfg=CFG, toarray_on_fs=True
)

final_oof_auc = auc(y, oof_pred, name="OOF blended")



# Refit preprocessor on full data
P = preprocessor.fit_transform(X)
VT = VarianceThreshold(threshold=CFG["FS_VARIANCE_THRESH"])
P = VT.fit_transform(P)

# Build a proxy FS mask using a quick LGB on full data
P_dense = P.toarray() if hasattr(P, "toarray") else P
proxy_lgb = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.05, random_state=CFG["SEED"])
proxy_lgb.fit(P_dense, y)
proxy_imp = proxy_lgb.feature_importances_
proxy_mask = proxy_imp > 0
P_sel = P_dense[:, proxy_mask]

final_lgb = lgb.LGBMClassifier(**CFG["LGB_PARAMS"])
final_lgb.fit(P_sel, y)

# Importance plot (top 30)
imp_vals = final_lgb.feature_importances_
# Fetch feature names after OHE:
ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
num_names = preprocessor.transformers_[0][2]  # numeric_features list
cat_names = list(ohe.get_feature_names_out(input_features=categorical_features))
all_feats = list(num_names) + cat_names

# Align with VT & proxy mask
vt_mask = VT.get_support()            # after variance threshold
feat_after_vt = np.array(all_feats)[vt_mask]
feat_after_proxy = feat_after_vt[proxy_mask]

imp_df = pd.DataFrame({
    "feature": feat_after_proxy,
    "importance": imp_vals
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8,10))
sns.barplot(data=imp_df.head(30), x="importance", y="feature")
plt.title("Final Model â€“ Top 30 Feature Importances")
plt.tight_layout()
plt.show()



try:
    import shap
    shap.initjs()
    # Use a 2k sample for speed
    sample_idx = np.random.RandomState(CFG["SEED"]).choice(len(P_sel), size=min(2000, len(P_sel)), replace=False)
    explainer = shap.TreeExplainer(final_lgb)
    shap_values = explainer.shap_values(P_sel[sample_idx])
    shap.summary_plot(shap_values, pd.DataFrame(P_sel[sample_idx], columns=feat_after_proxy), plot_type="bar", show=True)
except Exception as e:
    logging.warning(f"SHAP summary skipped: {e}")



sub_out = sub.copy()
sub_out[CFG["TARGET"]] = test_pred
sub_out.to_csv("submission.csv", index=False)
logging.info("âœ… submission.csv created.")
display(sub_out.head(5))





