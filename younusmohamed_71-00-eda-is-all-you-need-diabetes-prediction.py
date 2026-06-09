import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import gc, itertools, json, math, os, sys, time, warnings 
import pandas as pd
import seaborn as sns
import shap

from pathlib import Path
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pd.set_option("display.max_columns", 200)
warnings.filterwarnings("ignore")
sns.set()


LGB_AVAILABLE = True
SHAP_AVAILABLE = True


CONFIG = {
    "comp_dir": Path("/kaggle/input/playground-series-s5e12"),
    "orig_csv": Path("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"),
    "target_col": "diagnosed_diabetes",
    "id_col": "id",
    "random_state": 42,
    "n_jobs": -1,
    "plots_dir": Path("./eda_outputs/plots"),
    "tables_dir": Path("./eda_outputs/tables"),
    "save_fig_dpi": 120,
    "perm_importance_n_repeats": 5,
    "kfold_splits": 5,
}

os.makedirs(CONFIG["plots_dir"], exist_ok=True)
os.makedirs(CONFIG["tables_dir"], exist_ok=True)

def tprint(*a):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a)


def safe_read_csv(path: Path, name: str):
    tprint(f"Loading {name}")
    if not path or not Path(path).exists():
        tprint(f"  -> {name} not found (skipping).")
        return None
    df = pd.read_csv(path)
    tprint(f"  -> shape={df.shape}")
    return df

train_path = CONFIG["comp_dir"] / "train.csv"
test_path  = CONFIG["comp_dir"] / "test.csv"
sample_sub_path = CONFIG["comp_dir"] / "sample_submission.csv"

train = safe_read_csv(train_path, "train")
test  = safe_read_csv(test_path,  "test")
sample_sub = safe_read_csv(sample_sub_path, "sample_submission")

orig = safe_read_csv(CONFIG["orig_csv"], "original") if CONFIG["orig_csv"] else None


from IPython.display import display

def split_features(df, target=None, id_col=None):
    cols = list(df.columns)
    if target in cols: cols.remove(target)
    if id_col in cols: cols.remove(id_col)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols

def save_table(df, fname):
    out = CONFIG["tables_dir"] / fname
    if not out.name.endswith(".csv"):
        out = out.with_suffix(".csv")
    df.to_csv(out, index=False)
    tprint(f"Saved table -> {out}")
    # Display in full (may be long)
    display(df)

def save_fig(fname):
    out = CONFIG["plots_dir"] / fname
    if not out.name.endswith(".png"):
        out = out.with_suffix(".png")
    plt.tight_layout()
    plt.savefig(out, dpi=CONFIG["save_fig_dpi"], bbox_inches="tight")
    tprint(f"Saved fig -> {out}")
    plt.show()
    plt.close()

def basic_summary(df, name, target=None, id_col=None):
    tprint(f"=== Basic Summary: {name} ===")
    print("shape:", df.shape)
    display(df.head(3))
    miss = df.isna().sum().sort_values(ascending=False).to_frame("missing")
    dup = int(df.duplicated().sum())
    print("\nMissing values (top 20):")
    display(miss.head(20))
    print(f"Duplicates: {dup}")
    num_cols, cat_cols = split_features(df, target, id_col)
    print(f"Numeric columns ({len(num_cols)}):", num_cols[:20], ("..." if len(num_cols)>20 else ""))
    print(f"Categorical columns ({len(cat_cols)}):", cat_cols[:20], ("..." if len(cat_cols)>20 else ""))
    return num_cols, cat_cols


def distrib_numeric(df, name, num_cols, bins=50):
    if not num_cols: return
    for c in num_cols:
        col = df[c].dropna()
        if col.empty: continue
        plt.figure(figsize=(6,3.2))
        plt.hist(col, bins=bins)
        plt.title(f"{name} — {c}")
        plt.xlabel(c); plt.ylabel("count")
        save_fig(f"{name}__num__{c}.png")

def distrib_categorical(df, name, cat_cols, topk=None):
    if not cat_cols: return
    for c in cat_cols:
        vc = df[c].value_counts() if topk is None else df[c].value_counts().head(topk)
        plt.figure(figsize=(6,3.2))
        vc.iloc[::-1].plot(kind="barh")
        plt.title(f"{name} — {c}")
        save_fig(f"{name}__cat__{c}.png")

def correlation_matrices(df, name, num_cols, cmap="coolwarm"):
    if len(num_cols) < 2: 
        return
    corr_p = df[num_cols].corr(numeric_only=True, method="pearson")
    corr_s = df[num_cols].corr(numeric_only=True, method="spearman")

    # Save as full tables and display
    save_table(corr_p.reset_index().rename(columns={"index":"feature"}), f"{name}__corr_pearson.csv")
    save_table(corr_s.reset_index().rename(columns={"index":"feature"}), f"{name}__corr_spearman.csv")

    # Heatmaps
    plt.figure(figsize=(7.0, 6.0))
    sns.heatmap(corr_p, cmap=cmap, cbar=True)
    plt.title(f"{name} — correlation (pearson)")
    save_fig(f"{name}__corr_pearson.png")

    plt.figure(figsize=(7.0, 6.0))
    sns.heatmap(corr_s, cmap=cmap, cbar=True)
    plt.title(f"{name} — correlation (spearman)")
    save_fig(f"{name}__corr_spearman.png")

def cramers_v_matrix(df, name, cat_cols):
    if len(cat_cols) < 2:
        return
    def cramers_v(x, y):
        tbl = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(tbl, correction=False)[0]
        n = tbl.values.sum()
        phi2 = chi2 / n
        r,k = tbl.shape
        phi2corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
        rcorr = r - (r-1)**2/(n-1)
        kcorr = k - (k-1)**2/(n-1)
        return np.sqrt(phi2corr / max(1e-9, min((kcorr-1), (rcorr-1))))
    cols = cat_cols  # no cap
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i,c1 in enumerate(cols):
        for j in range(i+1, len(cols)):
            c2 = cols[j]
            try:
                v = cramers_v(df[c1].astype("category"), df[c2].astype("category"))
            except Exception:
                v = np.nan
            mat.loc[c1, c2] = mat.loc[c2, c1] = v
    save_table(mat.reset_index().rename(columns={"index":"feature"}), f"{name}__cramersV.csv")

    plt.figure(figsize=(7.0, 6.0))
    sns.heatmap(mat.astype(float), vmin=0, vmax=1, cmap="viridis")
    plt.title(f"{name} — Cramér's V")
    save_fig(f"{name}__cramersV.png")

def target_analysis(df, name, target, num_cols, cat_cols):
    if target not in df.columns: 
        tprint(f"[{name}] target '{target}' not present (skipping target analysis).")
        return
    # balance
    vc = df[target].value_counts(normalize=True).rename("fraction")
    vc_raw = df[target].value_counts().rename("count")
    out = pd.concat([vc_raw, vc], axis=1).reset_index().rename(columns={"index": target})
    save_table(out, f"{name}__target_balance.csv")

    # numeric separability (KDE full data)
    for c in num_cols:
        plt.figure(figsize=(6,3.2))
        try:
            sns.kdeplot(data=df, x=c, hue=target, common_norm=False)
        except Exception:
            # fallback: hist overlay
            a = df[df[target]==0][c].dropna()
            b = df[df[target]==1][c].dropna()
            plt.hist(a, bins=50, alpha=0.6, label="0")
            plt.hist(b, bins=50, alpha=0.6, label="1")
            plt.legend(title=target)
        plt.title(f"{name} - {c} by {target}")
        save_fig(f"{name}__num_by_target__{c}.png")

    # categorical rate tables
    for c in cat_cols:
        tmp = df.groupby(c, observed=True)[target].agg(["mean","count"]).sort_values("mean")
        tmp["rate"] = tmp["mean"]
        tmp = tmp.reset_index().rename(columns={"mean":"posit_rate"})
        save_table(tmp, f"{name}__cat_by_target__{c}.csv")

def outlier_report(df, name, num_cols):
    rows = []
    for c in num_cols:
        s = df[c].dropna()
        if len(s)==0: continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3-q1
        low, high = q1-1.5*iqr, q3+1.5*iqr
        z = (s - s.mean())/s.std(ddof=0)
        rows.append({
            "feature": c,
            "iqr_low": low, "iqr_high": high,
            "iqr_outlier_frac": ((s<low)|(s>high)).mean(),
            "z>3_frac": (np.abs(z)>3).mean(),
            "min": s.min(), "max": s.max(), "mean": s.mean(), "std": s.std(ddof=0)
        })
    rep = pd.DataFrame(rows).sort_values("iqr_outlier_frac", ascending=False)
    save_table(rep, f"{name}__outliers.csv")

def vif_report(df, name, num_cols, sample_n=150000):
    if len(num_cols) < 2: return
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    X = df[num_cols].dropna()
    if len(X) > sample_n:
        X = X.sample(sample_n, random_state=CONFIG["random_state"])
    try:
        X1 = add_constant(X, has_constant='add')
        vifs = []
        for i, col in enumerate(X1.columns):
            if col == "const": continue
            vifs.append({"feature": col, "VIF": variance_inflation_factor(X1.values, i)})
        vifs = pd.DataFrame(vifs).sort_values("VIF", ascending=False)
        save_table(vifs, f"{name}__vif.csv")
    except Exception as e:
        tprint(f"VIF failed: {e}")


def psi(expected, actual, bins=20):
    # Population Stability Index
    e_perc, bins_edges = np.histogram(expected, bins=bins)
    a_perc, _ = np.histogram(actual, bins=bins_edges)
    e_perc = e_perc / max(1, e_perc.sum())
    a_perc = a_perc / max(1, a_perc.sum())
    vals = []
    for e,a in zip(e_perc, a_perc):
        if e==0 or a==0:
            e += 1e-6; a += 1e-6
        vals.append((a - e) * np.log(a / e))
    return float(np.sum(vals))

def compare_datasets(dfA, nameA, dfB, nameB, num_cols, cat_cols):
    # Numeric deltas & tests
    rows = []
    for c in num_cols:
        a = dfA[c].dropna()
        b = dfB[c].dropna()
        if len(a)==0 or len(b)==0: continue
        ks = ks_2samp(a, b).statistic
        rows.append({
            "feature": c,
            f"{nameA}_mean": a.mean(), f"{nameB}_mean": b.mean(),
            "mean_delta": a.mean()-b.mean(),
            f"{nameA}_std": a.std(ddof=0), f"{nameB}_std": b.std(ddof=0),
            "ks_stat": ks,
            "psi": psi(a.values, b.values, bins=20)
        })
    num_cmp = pd.DataFrame(rows).sort_values(["psi","ks_stat","mean_delta"], ascending=False)
    save_table(num_cmp, f"compare__{nameA}_vs_{nameB}__numeric.csv")

    # Categorical distribution distance
    rows = []
    for c in cat_cols:
        a_v = dfA[c].astype("category").value_counts(normalize=True)
        b_v = dfB[c].astype("category").value_counts(normalize=True)
        overlap = set(a_v.index).intersection(set(b_v.index))
        jacc = len(overlap) / max(1, len(set(a_v.index).union(set(b_v.index))))
        aligned = pd.concat([a_v, b_v], axis=1).fillna(0)
        l1 = np.abs(aligned.iloc[:,0]-aligned.iloc[:,1]).sum()
        rows.append({"feature": c, "support_jaccard": jacc, "L1_dist": l1})
    cat_cmp = pd.DataFrame(rows).sort_values(["L1_dist","support_jaccard"], ascending=[False, True])
    save_table(cat_cmp, f"compare__{nameA}_vs_{nameB}__categorical.csv")



def prepare_xy(df, target, id_col):
    y = df[target].astype(int) if target in df.columns else None
    X = df.drop(columns=[c for c in [target, id_col] if c in df.columns]).copy()

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        n_jobs=CONFIG["n_jobs"]
    )
    return X, y, num_cols, cat_cols, pre

def fi_mutual_info(X, y, num_cols, cat_cols):
    df = pd.concat([X, y.rename("y")], axis=1)
    rows = []
    for c in num_cols:
        v = mutual_info_classif(df[[c]].fillna(df[c].median()), df["y"], discrete_features=False, random_state=CONFIG["random_state"])
        rows.append({"feature": c, "mi": float(v[0])})
    for c in cat_cols:
        x = df[[c]].copy()
        x[c] = x[c].astype("category")
        v = mutual_info_classif(pd.get_dummies(x, dummy_na=True), df["y"], discrete_features=True, random_state=CONFIG["random_state"])
        rows.append({"feature": c, "mi": float(np.mean(v))})
    mi_df = pd.DataFrame(rows).sort_values("mi", ascending=False)
    return mi_df

def fi_logreg(X, y, pre):
    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=200, n_jobs=CONFIG["n_jobs"], solver="saga", penalty="l2"))
    ])
    clf.fit(X, y)
    feat_names = clf.named_steps["pre"].get_feature_names_out()
    coefs = np.abs(clf.named_steps["lr"].coef_).ravel()
    return pd.DataFrame({"feature_transformed": feat_names, "abs_coef": coefs}).sort_values("abs_coef", ascending=False)

def fi_rf(X, y, pre):
    clf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"]))
    ])
    clf.fit(X, y)
    feat_names = clf.named_steps["pre"].get_feature_names_out()
    imp = clf.named_steps["rf"].feature_importances_
    return pd.DataFrame({"feature_transformed": feat_names, "rf_importance": imp}).sort_values("rf_importance", ascending=False)

def fi_lgb(X, y, pre):
    if not LGB_AVAILABLE:
        return None
    clf = Pipeline([
        ("pre", pre),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=1200, learning_rate=0.03,
            num_leaves=64, subsample=0.8, colsample_bytree=0.8,
            random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"]
        ))
    ])
    clf.fit(X, y)
    feat_names = clf.named_steps["pre"].get_feature_names_out()
    imp = clf.named_steps["lgb"].feature_importances_
    return pd.DataFrame({"feature_transformed": feat_names, "lgb_importance": imp}).sort_values("lgb_importance", ascending=False)

def fi_permutation_auc(X, y, pre, id_col_name="id"):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=CONFIG["random_state"], stratify=y)
    clf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=400, random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"]))
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_valid)[:,1]
    base_auc = roc_auc_score(y_valid, pred)
    pi = permutation_importance(
        clf, X_valid, y_valid, n_repeats=CONFIG["perm_importance_n_repeats"],
        random_state=CONFIG["random_state"], scoring="roc_auc", n_jobs=CONFIG["n_jobs"]
    )
    feat_names = clf.named_steps["pre"].get_feature_names_out()
    df = pd.DataFrame({
        "feature_transformed": feat_names,
        "perm_importance_mean": pi.importances_mean,
        "perm_importance_std": pi.importances_std
    }).sort_values("perm_importance_mean", ascending=False)
    df.attrs["base_auc"] = base_auc
    return df

def fi_shap_lgb(X, y, pre):
    if not (LGB_AVAILABLE and SHAP_AVAILABLE): 
        return None
    # Fit on a subset for speed (this is not a plot; adjust if desired)
    X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.6, random_state=CONFIG["random_state"], stratify=y)
    model = Pipeline([
        ("pre", pre),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8, random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"]
        ))
    ])
    model.fit(X_sample, y_sample)
    Xp = model.named_steps["pre"].transform(X_sample)
    feat_names = model.named_steps["pre"].get_feature_names_out()
    explainer = shap.TreeExplainer(model.named_steps["lgb"])
    shap_values = explainer.shap_values(Xp)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    imp = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({"feature_transformed": feat_names, "shap_importance": imp}).sort_values("shap_importance", ascending=False)


def run_single_dataset_eda(df, name, target=None, id_col=None):
    num_cols, cat_cols = basic_summary(df, name, target, id_col)

    # dtypes table
    dtypes_tbl = pd.DataFrame({"feature": df.columns, "dtype": df.dtypes.astype(str)})
    save_table(dtypes_tbl, f"{name}__dtypes.csv")

    # Distributions (all data)
    distrib_numeric(df, name, num_cols)
    distrib_categorical(df, name, cat_cols)

    # Correlations
    correlation_matrices(df, name, num_cols)
    cramers_v_matrix(df, name, cat_cols)

    # Target analysis
    target_analysis(df, name, target, num_cols, cat_cols)

    # Outliers & VIF
    outlier_report(df, name, num_cols)
    vif_report(df, name, num_cols)
    return num_cols, cat_cols

def run_compare(A, nameA, B, nameB, target=None, id_col=None):
    colsA = set(A.columns); colsB = set(B.columns)
    common = list((colsA & colsB) - {target})
    num_cols = [c for c in common if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])]
    cat_cols = [c for c in common if c not in num_cols]
    compare_datasets(A[common], nameA, B[common], nameB, num_cols, cat_cols)

def run_importances(train, target, id_col):
    X, y, num_cols, cat_cols, pre = prepare_xy(train, target, id_col)

    fi_mi = fi_mutual_info(X.copy(), y, num_cols, cat_cols)
    save_table(fi_mi, "fi__mutual_info.csv")

    fi_lr = fi_logreg(X.copy(), y, pre)
    save_table(fi_lr, "fi__logreg_abscoef.csv")

    fi_rf_df = fi_rf(X.copy(), y, pre)
    save_table(fi_rf_df, "fi__randomforest.csv")

    if LGB_AVAILABLE:
        fi_lgb_df = fi_lgb(X.copy(), y, pre)
        if fi_lgb_df is not None:
            save_table(fi_lgb_df, "fi__lightgbm.csv")

    # fi_perm = fi_permutation_auc(X.copy(), y, pre)
    # save_table(fi_perm, "fi__permutation_auc.csv")
    # tprint(f"Permutation base AUC: {fi_perm.attrs.get('base_auc')}")

    if SHAP_AVAILABLE:
        fi_shap_df = fi_shap_lgb(X.copy(), y, pre)
        if fi_shap_df is not None:
            save_table(fi_shap_df, "fi__shap_lgb.csv")


def leakage_checks(train, test, target, id_col):
    rep = {}
    if id_col in train.columns and target in train.columns:
        try:
            corr = np.corrcoef(train[id_col].astype(float), train[target].astype(float))[0,1]
        except Exception:
            corr = np.nan
        rep["id_vs_target_corr"] = corr
    if id_col in train.columns and id_col in test.columns:
        overlap = len(set(train[id_col]) & set(test[id_col]))
        rep["train_test_id_overlap"] = int(overlap)
    rep["train_duplicated_rows"] = int(train.duplicated().sum())
    rep["test_duplicated_rows"]  = int(test.duplicated().sum())
    save_table(pd.DataFrame([rep]), "leakage_checks.csv")


tr_num, tr_cat = run_single_dataset_eda(
    train, "train",
    target=CONFIG["target_col"],
    id_col=CONFIG["id_col"]
)


te_num, te_cat = run_single_dataset_eda(
    test, "test",
    target=None,  # no target in test
    id_col=CONFIG["id_col"]
)


run_compare(train, "train", test, "test", target=CONFIG["target_col"], id_col=CONFIG["id_col"])


leakage_checks(train, test, CONFIG["target_col"], CONFIG["id_col"])


run_importances(train, CONFIG["target_col"], CONFIG["id_col"])


tgt = CONFIG["target_col"] if CONFIG["target_col"] in orig.columns else None
idc = CONFIG["id_col"] if CONFIG["id_col"] in orig.columns else None
run_single_dataset_eda(orig, "original", target=tgt, id_col=idc)


tgt = CONFIG["target_col"] if CONFIG["target_col"] in orig.columns else None
idc = CONFIG["id_col"] if CONFIG["id_col"] in orig.columns else None
run_compare(orig, "original", train, "train", target=tgt, id_col=idc)


run_compare(orig, "original", test,  "test",  target=tgt, id_col=idc)


run_importances(orig, CONFIG["target_col"], CONFIG["id_col"])


tprint("EDA complete. Outputs saved under:", CONFIG["plots_dir"], "and", CONFIG["tables_dir"])


assert train is not None and test is not None, "Train/Test not loaded."
assert CONFIG["target_col"] in train.columns, f"Target '{CONFIG['target_col']}' not in train."


# Split features
features = [c for c in train.columns if c not in [CONFIG["target_col"], CONFIG["id_col"]]]
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in features if c not in num_cols]

X = train[features]
y = train[CONFIG["target_col"]].astype(int)
X_test = test[features]

tprint(f"Training LGBM with {len(num_cols)} numeric and {len(cat_cols)} categorical features.")


# Preprocess (impute numeric median, impute+OHE categorical)
pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse=True))
        ]), cat_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0  # keep as sparse if possible to save memory
)


# Train/valid split
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=CONFIG["random_state"], stratify=y
)

# Fit preprocessor separately so we can pass transformed arrays to LGBM (for early stopping)
tprint("Fitting preprocessor...")
Xp_tr = pre.fit_transform(X_tr)
Xp_va = pre.transform(X_va)
Xp_te = pre.transform(X_test)

tprint(f"Shapes -> train: {Xp_tr.shape}, valid: {Xp_va.shape}, test: {Xp_te.shape}")


# LGBM model
lgbm = lgb.LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=CONFIG["random_state"],
    n_jobs=CONFIG["n_jobs"]
)

tprint("Fitting LightGBM...")
lgbm.fit(
    Xp_tr, y_tr,
    eval_set=[(Xp_va, y_va)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(200)]
)


# Validation AUC
va_pred = lgbm.predict_proba(Xp_va, raw_score=False)[:, 1]
va_auc = roc_auc_score(y_va, va_pred)
tprint(f"Validation AUC: {va_auc:.6f}")
print("Best iteration:", lgbm.best_iteration_)


# Predict test set
test_pred = lgbm.predict_proba(Xp_te, raw_score=False)[:, 1]


# Build submission
sub_col = CONFIG["target_col"]
submission = pd.DataFrame({
    CONFIG["id_col"]: test[CONFIG["id_col"]].values,
    sub_col: test_pred
})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
tprint(f"Saved submission -> {out_path} (rows={len(submission)})")




