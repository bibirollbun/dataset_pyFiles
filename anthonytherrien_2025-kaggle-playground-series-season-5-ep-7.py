# Import standard libraries
import os
import gc
import time
import math
from pathlib import Path
from typing import List, Tuple, Dict

# Import third‑party libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Gradient boosting libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Optional optimization
try:
    import optuna
except Exception:
    optuna = None

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for interpretation
try:
    import shap
except Exception:
    shap = None

# IPython display helpers
from IPython.display import display, HTML


# Define global configuration dictionary
CONFIG = {
    "COMPETITION_NAME": "playground-series-s5e7",
    "RANDOM_STATE": 42,
    "N_FOLDS": 10,
    "N_TRIALS": 30,
    "PCA_COMPONENTS": 8,
    "N_CLUSTERS": 5,
    "THRESH_SEARCH": True,
    "USE_OPTUNA": False,
    "VERBOSE": 0
}


# Define a reproducible random seed function
def set_seed(seed: int):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Call the seed setter
set_seed(CONFIG["RANDOM_STATE"])


# Define a simple timer context manager
class Timer:
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.name}: {time.time() - self.start:.2f}s")


# Define a memory reduction helper for numeric columns
def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif str(col_type).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


# Define a function to safely install packages in Kaggle
def ensure_packages(packages: List[str]):
    missing = []
    for p in packages:
        try:
            __import__(p.split("==")[0])
        except ImportError:
            missing.append(p)
    if missing:
        print("Installing:", missing)
        os.system("pip install " + " ".join(missing) + " -q")


# Define paths for train/test
INPUT_DIR = Path("/kaggle/input/playground-series-s5e7")
TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH  = INPUT_DIR / "test.csv"
SAMPLE_PATH = INPUT_DIR / "sample_submission.csv"


# Load data
with Timer("Load data"):
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_PATH)


# Show head of data
display(train.head())
display(test.head())


# Reduce memory usage
train = reduce_memory_usage(train)
test  = reduce_memory_usage(test)


# Identify target and features
TARGET_COL = "Personality"
ID_COL = "id"
features = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]


# Map binary categorical strings to 0/1
BIN_MAP = {"Yes": 1, "No": 0}
cat_cols = train[features].select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    train[col] = train[col].replace(BIN_MAP)
    test[col]  = test[col].replace(BIN_MAP)


# Map target to numeric
label_mapping = {"Introvert": 0, "Extrovert": 1}
train["target"] = train[TARGET_COL].map(label_mapping)


# Check basic info
print(train.shape, test.shape)
print(train.dtypes.value_counts())


# Check missing values
missing_train = train[features].isnull().mean().sort_values(ascending=False)
missing_test  = test[features].isnull().mean().sort_values(ascending=False)
display(pd.DataFrame({"train_missing": missing_train, "test_missing": missing_test}).head(10))


# Check class balance
train.target.value_counts(normalize=True).plot(kind="bar")
plt.title("Target distribution")
plt.show()


# Plot feature distributions by target
def plot_distributions(df: pd.DataFrame, cols: List[str], target: str, n: int = 10):
    sel = cols[:n]
    rows = math.ceil(len(sel) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(sel):
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.kdeplot(data=df, x=col, hue=target, ax=axes[i], common_norm=False)
        else:
            sns.countplot(data=df, x=col, hue=target, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()


# Call distribution plotter
plot_distributions(train, features, TARGET_COL, n=min(12, len(features)))


# Build correlation heatmap (Spearman) using numeric columns only
num_cols = train[features + ["target"]].select_dtypes(include=[np.number]).columns.tolist()
corr = train[num_cols].corr(method="spearman")
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Spearman Correlation Heatmap")
plt.show()


# Define PCA projection helper with imputation
def pca_projection(df: pd.DataFrame, cols: List[str], target: str, n_comp: int = 2):
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df[cols])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    n_valid = min(n_comp, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if n_valid < 2:
        return
    pca = PCA(n_components=n_valid, random_state=CONFIG["RANDOM_STATE"])
    comps = pca.fit_transform(X_scaled)
    tmp = pd.DataFrame({"PC1": comps[:, 0], "PC2": comps[:, 1], target: df[target].values})
    plt.figure(figsize=(7,6))
    sns.scatterplot(data=tmp, x="PC1", y="PC2", hue=target, alpha=0.6, s=30)
    plt.title("PCA 2D Projection")
    plt.show()


# Call PCA projection
pca_projection(train, [c for c in num_cols if c != "target"], "target")


# Define a feature engineering function
def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    imputer = SimpleImputer(strategy="median")
    tr[cols] = imputer.fit_transform(tr[cols])
    te[cols] = imputer.transform(te[cols])
    skewness = tr[cols].skew().abs().sort_values(ascending=False)
    skew_cols = skewness[skewness > 1].index.tolist()
    for c in skew_cols:
        tr[f"log_{c}"] = np.log1p(tr[c].clip(lower=0))
        te[f"log_{c}"] = np.log1p(te[c].clip(lower=0))
    scaler = StandardScaler()
    tr_scaled = scaler.fit_transform(tr[cols])
    te_scaled = scaler.transform(te[cols])
    n_pca = min(CONFIG["PCA_COMPONENTS"], tr_scaled.shape[1], tr_scaled.shape[0] - 1)
    if n_pca > 0:
        pca = PCA(n_components=n_pca, random_state=CONFIG["RANDOM_STATE"])
        tr_pca = pca.fit_transform(tr_scaled)
        te_pca = pca.transform(te_scaled)
        for i in range(n_pca):
            tr[f"pca_{i}"] = tr_pca[:, i]
            te[f"pca_{i}"] = te_pca[:, i]
    n_clusters = min(CONFIG["N_CLUSTERS"], max(2, tr_scaled.shape[0] // 5))
    km = KMeans(n_clusters=n_clusters, random_state=CONFIG["RANDOM_STATE"], n_init="auto")
    tr["cluster"] = km.fit_predict(tr_scaled)
    te["cluster"] = km.predict(te_scaled)
    return tr, te


# Run feature engineering
with Timer("Feature engineering"):
    train_fe, test_fe = engineer_features(train, test, features)


# Update feature list
all_features = [c for c in train_fe.columns if c not in [TARGET_COL, "target", ID_COL]]
print(len(all_features), "features after engineering")


# Define cross-validation splitter
skf = StratifiedKFold(n_splits=CONFIG["N_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_STATE"])


# Define a baseline logistic regression pipeline
def run_logreg_cv(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        pipe = Pipeline([("scaler", scaler), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        oof[va_idx] = pipe.predict_proba(X_va)[:, 1]
        preds += pipe.predict_proba(X_test)[:, 1] / CONFIG["N_FOLDS"]
        score = accuracy_score(y_va, (oof[va_idx] > 0.5).astype(int))
        fold_scores.append(score)
        print(f"LogReg Fold {fold}: {score:.5f}")
    print("LogReg OOF Accuracy:", accuracy_score(y, (oof > 0.5).astype(int)))
    return oof, preds, np.mean(fold_scores)


# Run logistic regression CV
with Timer("LogReg baseline"):
    oof_lr, pred_lr, score_lr = run_logreg_cv(train_fe[all_features], train_fe["target"], test_fe[all_features])


# Define generic CV runner for tree models
def run_tree_model_cv(model_name: str, params: Dict, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        if model_name == "lgb":
            train_ds = lgb.Dataset(X_tr, label=y_tr)
            valid_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)
            model = lgb.train(
                params,
                train_ds,
                valid_sets=[valid_ds],
            )
            oof[va_idx] = model.predict(X_va)
            preds += model.predict(X_test) / CONFIG["N_FOLDS"]
        elif model_name == "xgb":
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            preds += model.predict_proba(X_test)[:, 1] / CONFIG["N_FOLDS"]
        elif model_name == "cat":
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            preds += model.predict_proba(X_test)[:, 1] / CONFIG["N_FOLDS"]
        else:
            raise ValueError("Unknown model")
        score = accuracy_score(y_va, (oof[va_idx] > 0.5).astype(int))
        fold_scores.append(score)
        print(f"{model_name} Fold {fold}: {score:.5f}")
    print(f"{model_name} OOF Accuracy:", accuracy_score(y, (oof > 0.5).astype(int)))
    return oof, preds, np.mean(fold_scores)


# LightGBM params
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": CONFIG["RANDOM_STATE"],
    "verbose": -1
}


# XGBoost params
xgb_params = {
    "tree_method": "hist",
    "eval_metric": "logloss",
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": CONFIG["RANDOM_STATE"],
    "n_estimators": 510
}


# CatBoost params
cat_params = {
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "learning_rate": 0.04,
    "depth": 6,
    "random_seed": CONFIG["RANDOM_STATE"],
    "iterations": 1020,
    "l2_leaf_reg": 3,
    "verbose": False
}


# Run LightGBM CV
with Timer("LightGBM CV"):
    oof_lgb, pred_lgb, score_lgb = run_tree_model_cv("lgb", lgb_params, train_fe[all_features], train_fe["target"], test_fe[all_features])


# Run XGBoost CV
with Timer("XGBoost CV"):
    oof_xgb, pred_xgb, score_xgb = run_tree_model_cv("xgb", xgb_params, train_fe[all_features], train_fe["target"], test_fe[all_features])


# Run CatBoost CV
with Timer("CatBoost CV"):
    oof_cat, pred_cat, score_cat = run_tree_model_cv("cat", cat_params, train_fe[all_features], train_fe["target"], test_fe[all_features])


# Collect OOF and test predictions
oof_df = pd.DataFrame({
    "lr": oof_lr,
    "lgb": oof_lgb,
    "xgb": oof_xgb,
    "cat": oof_cat
})
test_df_pred = pd.DataFrame({
    "lr": pred_lr,
    "lgb": pred_lgb,
    "xgb": pred_xgb,
    "cat": pred_cat
})


# Define a simple meta-model stacking with logistic regression
def stack_predictions(oof_mat: pd.DataFrame, y: pd.Series, test_mat: pd.DataFrame):
    oof_stack = np.zeros(len(y))
    test_stack = np.zeros(len(test_mat))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_mat, y)):
        X_tr, X_va = oof_mat.iloc[tr_idx], oof_mat.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_tr, y_tr)
        oof_stack[va_idx] = clf.predict_proba(X_va)[:, 1]
        test_stack += clf.predict_proba(test_mat)[:, 1] / CONFIG["N_FOLDS"]
        score = accuracy_score(y_va, (oof_stack[va_idx] > 0.5).astype(int))
        print(f"Stack Fold {fold}: {score:.5f}")
    print("Stack OOF Accuracy:", accuracy_score(y, (oof_stack > 0.5).astype(int)))
    return oof_stack, test_stack


# Run stacking
with Timer("Stacking"):
    oof_stack, pred_stack = stack_predictions(oof_df, train_fe["target"], test_df_pred)


# Optional threshold search to maximize accuracy on OOF
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    thresholds = np.linspace(0.3, 0.7, 401)
    best_t = 0.5
    best_acc = 0
    for t in thresholds:
        acc = accuracy_score(y_true, (y_prob > t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


# Run threshold search
if CONFIG["THRESH_SEARCH"]:
    best_t, best_acc = find_best_threshold(train_fe["target"].values, oof_stack)
    print("Best threshold:", best_t, "OOF acc:", best_acc)
else:
    best_t = 0.5
    best_acc = accuracy_score(train_fe["target"], (oof_stack > best_t).astype(int))


# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(train_fe["target"], (oof_stack > best_t).astype(int))
disp = ConfusionMatrixDisplay(cm, display_labels=["Introvert", "Extrovert"])
disp.plot(values_format="d")
plt.title("Stacked Model Confusion Matrix (OOF)")
plt.show()


# Train final LightGBM on full data for importance/SHAP
train_ds_full = lgb.Dataset(train_fe[all_features], label=train_fe["target"])
final_lgb = lgb.train(lgb_params, train_ds_full)


# LightGBM feature importance
lgb.plot_importance(final_lgb, max_num_features=15)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()


# SHAP values for LightGBM if available
if shap is not None:
    try:
        shap_values = shap.TreeExplainer(final_lgb).shap_values(train_fe[all_features])
        shap.summary_plot(shap_values, train_fe[all_features], show=False)
        plt.title("SHAP Summary (LightGBM)")
        plt.show()
    except Exception as e:
        print("SHAP failed:", e)
else:
    print("SHAP not installed, skipping interpretation.")


# Create submission dataframe
sub = sample_sub.copy()
sub[TARGET_COL] = np.where(pred_stack > best_t, "Extrovert", "Introvert")
sub.to_csv("/kaggle/working/submission.csv", index=False)
display(sub.head())


# Save score summary
score_summary = pd.Series({
    "LogReg": score_lr,
    "LGB": score_lgb,
    "XGB": score_xgb,
    "CatBoost": score_cat,
    "Stacked": best_acc,
    "Best_Threshold": best_t
})
score_summary.to_csv("/kaggle/working/score_summary.csv")
display(score_summary)


# Garbage collection
gc.collect()

