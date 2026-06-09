import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings
warnings.filterwarnings("ignore")


import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import shap
# For better plots
plt.style.use("seaborn-v0_8")


# Main competition data
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


# External Kaggle diabetes dataset (same-ish feature universe)
external = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")

# BRFSS datasets (for pattern mining only – NEVER merged into Kaggle data)
brfhi1 = pd.read_csv(
    "/kaggle/input/d/alexteboul/diabetes-health-indicators-dataset/"
    "diabetes_012_health_indicators_BRFSS2015.csv"
)
brfhi2 = pd.read_csv(
    "/kaggle/input/d/alexteboul/diabetes-health-indicators-dataset/"
    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)
brfhi3 = pd.read_csv(
    "/kaggle/input/d/alexteboul/diabetes-health-indicators-dataset/"
    "diabetes_binary_health_indicators_BRFSS2015.csv"
)


print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("External   :", external.shape)
print("BRFFS1     :", brfhi1.shape)
print("BRFFS2     :", brfhi2.shape)
print("BRFSS3     :", brfhi3.shape)
print("\nSample submission unique values:", sub["diagnosed_diabetes"].unique())


train.info()
test.info()
external.info()
brfhi1.info()
brfhi2.info()
brfhi3.info()


predict = "diagnosed_diabetes"

# Categorical columns only from Kaggle train
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()

# Numeric columns (common to train & test, not touching external yet)
num_cols_base = [
    col for col in train.columns
    if col not in ["id", predict]
    and col in test.columns
    and is_numeric_dtype(train[col])
    and is_numeric_dtype(test[col])
]

print("\nCategorical columns:", cat_cols)
print("Numeric base columns:", num_cols_base)

# Align external to Kaggle train columns
external = external.reindex(columns=train.columns, fill_value=np.nan)
print("\n'external' aligned to train columns. diagnosed_diabetes in external?:",
      predict in external.columns)


# Use train medians to fill numeric NaNs
train_medians = train.median(numeric_only=True)

for df in [train, test, external]:
    for col in df.columns:
        if col in ["id", predict]:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(train_medians[col])

# Label encode categorical columns jointly across (train, test, external)
encoders = {}
for col in cat_cols:
    all_values = (
        pd.concat([train[col], test[col], external[col]])
        .astype(str)
        .unique()
    )

    le = LabelEncoder()
    le.fit(all_values)

    train[col]    = le.transform(train[col].astype(str))
    test[col]     = le.transform(test[col].astype(str))
    external[col] = le.transform(external[col].astype(str))

    encoders[col] = le

print("\nCategorical encoding done for train/test/external.")
print("External diagnosed_diabetes values:", external[predict].unique())
print("BRFSS datasets remain RAW and untouched.")


print("\n=== Target distribution: train vs external ===")

train_summary = (
    train[predict]
    .value_counts()
    .rename_axis(predict)
    .reset_index(name="count")
)
train_summary["source"] = "train"

external_summary = (
    external[predict]
    .value_counts()
    .rename_axis(predict)
    .reset_index(name="count")
)
external_summary["source"] = "external"

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(
    data=train_summary,
    x="source",
    y="count",
    hue=predict,
    palette="viridis",
    ax=axs[0]
)
axs[0].set_title("Train Dataset")
axs[0].set_xlabel("")
axs[0].set_ylabel("Count")

sns.barplot(
    data=external_summary,
    x="source",
    y="count",
    hue=predict,
    palette="viridis",
    ax=axs[1]
)
axs[1].set_title("External Dataset")
axs[1].set_xlabel("")
axs[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

print("\n=== Adversarial validation: train vs test (logistic regression) ===")

# --- 1. Build adversarial dataset ---
train_av = train.copy()
test_av  = test.copy()

train_av["is_train"] = 1
test_av["is_train"]  = 0

full_av = pd.concat([train_av, test_av], axis=0, ignore_index=True)

X_av = full_av.drop(columns=["id", predict, "is_train"], errors="ignore")
y_av = full_av["is_train"]

# --- 2. Preprocessing ---
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_base),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# --- 3. Adversarial model ---
av_pipeline = Pipeline(
    steps=[
        ("prep", preprocess),
        ("lr", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        )),
    ]
)

# --- 4. Cross-validated adversarial AUC ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_av = np.zeros(len(full_av))

for tr_idx, val_idx in skf.split(X_av, y_av):
    X_tr, X_val = X_av.iloc[tr_idx], X_av.iloc[val_idx]
    y_tr, y_val = y_av.iloc[tr_idx], y_av.iloc[val_idx]

    av_pipeline.fit(X_tr, y_tr)
    oof_av[val_idx] = av_pipeline.predict_proba(X_val)[:, 1]

av_auc = roc_auc_score(y_av, oof_av)
print(f"Adversarial AUC (logistic regression): {av_auc:.4f}")



# Fit on full data for diagnostics
av_pipeline.fit(X_av, y_av)

feature_names = av_pipeline.named_steps["prep"].get_feature_names_out()

coef = pd.Series(
    av_pipeline.named_steps["lr"].coef_[0],
    index=feature_names,
).sort_values(key=np.abs, ascending=False)

print("\nTop adversarial features:")
print(coef.head(20))



print("\n=== Jensen–Shannon divergence heatmap (base numeric) ===")

def jsd(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return jensenshannon(p, q)

results = []
for col in num_cols_base:
    t = train[col].values
    e = test[col].values
    o = external[col].values

    bins = 40
    hist_train, _ = np.histogram(t, bins=bins, density=True)
    hist_test,  _ = np.histogram(e, bins=bins, density=True)
    hist_ext,   _ = np.histogram(o, bins=bins, density=True)

    results.append({
        "Feature": col,
        "Train vs Test": jsd(hist_train, hist_test),
        "Train vs External": jsd(hist_train, hist_ext),
        "Test vs External": jsd(hist_test, hist_ext),
    })

df_jsd = pd.DataFrame(results).set_index("Feature")

plt.figure(figsize=(10, max(4, 0.35 * len(num_cols_base))))
sns.heatmap(df_jsd, annot=True, cmap="coolwarm_r", fmt=".3f")
plt.title("JS Divergence Heatmap (train/test/external)")
plt.tight_layout()
plt.show()

print("\n=== KDE distributions (base numeric) ===")

train_plot = train[num_cols_base].replace([np.inf, -np.inf], np.nan)
test_plot  = test[num_cols_base].replace([np.inf, -np.inf], np.nan)
ext_plot   = external[num_cols_base].replace([np.inf, -np.inf], np.nan)

n = len(num_cols_base)
ncols = 4
nrows = int(np.ceil(n / ncols))

plt.figure(figsize=(20, 3 * nrows))
for i, col in enumerate(num_cols_base, 1):
    plt.subplot(nrows, ncols, i)
    sns.kdeplot(train_plot[col].dropna().to_numpy(), label="train", fill=True, alpha=0.4)
    sns.kdeplot(test_plot[col].dropna().to_numpy(),  label="test",  fill=True, alpha=0.4)
    sns.kdeplot(ext_plot[col].dropna().to_numpy(),   label="external", fill=True, alpha=0.4)
    plt.title(col, fontsize=9)
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.legend(loc="upper right")
plt.show()


print("\n=== Baseline XGB for SHAP (raw train vs external) ===")

y_raw = train[predict].astype(int)
X_raw = train.drop(columns=["id", predict], errors="ignore")

X_ext_raw = external.reindex(columns=X_raw.columns)

baseline_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
baseline_model.fit(X_raw, y_raw)

print("Baseline XGB trained. Running SHAP on a sample (raw features)...")

# If runtime is an issue, reduce sample size
train_sample_raw = X_raw.sample(min(3000, len(X_raw)), random_state=42)
ext_sample_raw   = X_ext_raw.sample(min(3000, len(X_ext_raw)), random_state=42)

explainer_raw = shap.Explainer(baseline_model)
shap_train_raw = explainer_raw(train_sample_raw)
shap_ext_raw   = explainer_raw(ext_sample_raw)

shap.summary_plot(shap_train_raw.values, train_sample_raw, plot_type="bar")
shap.summary_plot(shap_ext_raw.values,   ext_sample_raw,   plot_type="bar")


print("\n=== BRFSS baseline model (pattern mining only) ===")

brf = brfhi3.copy()
brf_target_col = "Diabetes_binary"

y_brf = brf[brf_target_col].astype(int)
X_brf = brf.drop(columns=[brf_target_col])

brf_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    random_state=4242,
    n_jobs=-1,
)
brf_model.fit(X_brf, y_brf)

print("BRFSS XGB trained. SHAP for BRFSS to confirm risk hierarchy...")

brf_sample = X_brf.sample(min(3000, len(X_brf)), random_state=42)
explainer_brf = shap.Explainer(brf_model)
shap_brf = explainer_brf(brf_sample)
shap.summary_plot(shap_brf.values, brf_sample, plot_type="bar")


print("\n=== Building SHAP-guided, adversarial-safe FE datasets ===")

import numpy as np
import pandas as pd

EPS = 1e-6

# ------------------------------------------------------------------
# 1. Feature groups (SHAP-informed)
# ------------------------------------------------------------------

core_numeric = [
    "age",
    "bmi",
    "waist_to_hip_ratio",
    "triglycerides",
    "hdl_cholesterol",
    "cholesterol_total",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
]

binary_clinical = [
    "family_history_diabetes",
    "hypertension_history",
]

lifestyle_numeric = [
    "physical_activity_minutes_per_week",
    "diet_score",
    "screen_time_hours_per_day",
]

categorical = [
    "gender",
    "ethnicity",
    "income_level",
    "smoking_status",
]

drop_features = [
    "cardiovascular_history",
    "education_level",
    "sleep_hours_per_day",
    "employment_status",
    "alcohol_consumption_per_week",  # weak SHAP + drift-prone
]

# ------------------------------------------------------------------
# 2. Global clipping (TRAIN + TEST)
# ------------------------------------------------------------------

def compute_global_clip_bounds(df, cols, q_low=0.005, q_high=0.995):
    return {
        c: (df[c].quantile(q_low), df[c].quantile(q_high))
        for c in cols if c in df.columns
    }

full_for_bounds = pd.concat(
    [train.drop(columns=[predict], errors="ignore"), test],
    axis=0,
    ignore_index=True,
)

clip_cols = core_numeric + lifestyle_numeric
CLIP_BOUNDS = compute_global_clip_bounds(full_for_bounds, clip_cols)

# ------------------------------------------------------------------
# 3. SHAP-optimal FE function
# ------------------------------------------------------------------

def build_optimal_fe(df, clip_bounds):
    df = df.copy()
    df = df.drop(columns=[c for c in drop_features if c in df.columns], errors="ignore")

    g = df.get

    # -------------------------------
    # Global clipping
    # -------------------------------
    for c, (lo, hi) in clip_bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lo, hi)

    # -------------------------------
    # Canonical monotonic transforms
    # (ONLY where SHAP supports it)
    # -------------------------------
    if "physical_activity_minutes_per_week" in df.columns:
        df["pa_log"] = np.log1p(g("physical_activity_minutes_per_week").clip(lower=0))

    if "triglycerides" in df.columns:
        df["tg_log"] = np.log1p(g("triglycerides").clip(lower=0))

    # -------------------------------
    # Age structure (strong SHAP)
    # -------------------------------
    if "age" in df.columns:
        df["age2"] = g("age") ** 2
        df["age_decade"] = (g("age") // 10).astype(int)

    # -------------------------------
    # Lipid axis (SHAP-consistent)
    # ONE relative + ONE absolute
    # -------------------------------
    if {"cholesterol_total", "hdl_cholesterol"}.issubset(df.columns):
        df["non_hdl_chol"] = g("cholesterol_total") - g("hdl_cholesterol")

    if {"triglycerides", "hdl_cholesterol"}.issubset(df.columns):
        df["tg_hdl_ratio"] = g("triglycerides") / (g("hdl_cholesterol") + EPS)

    # -------------------------------
    # Blood pressure
    # -------------------------------
    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        df["pulse_pressure"] = g("systolic_bp") - g("diastolic_bp")

    # -------------------------------
    # Anthropometry × age (SHAP-backed)
    # -------------------------------
    if {"bmi", "age"}.issubset(df.columns):
        df["bmi_age"] = g("bmi") * g("age")

    if {"waist_to_hip_ratio", "age"}.issubset(df.columns):
        df["whr_age"] = g("waist_to_hip_ratio") * g("age")

    # -------------------------------
    # Physical activity interactions
    # (restricted to strongest)
    # -------------------------------
    if "pa_log" in df.columns:
        if "bmi" in df.columns:
            df["bmi_pa"] = g("bmi") * df["pa_log"]
        if "age" in df.columns:
            df["age_pa"] = g("age") * df["pa_log"]

    # -------------------------------
    # Family history interactions
    # (SHAP top feature)
    # -------------------------------
    if "family_history_diabetes" in df.columns:
        fhx = g("family_history_diabetes")
        if "bmi" in df.columns:
            df["fhx_bmi"] = fhx * g("bmi")
        if "tg_log" in df.columns:
            df["fhx_tg"] = fhx * df["tg_log"]

    # -------------------------------
    # Metabolic syndrome (minimal)
    # -------------------------------
    score = 0
    if "bmi" in df.columns:
        score += (g("bmi") >= 30).astype(int)
    if "triglycerides" in df.columns:
        score += (g("triglycerides") >= 150).astype(int)
    if "hdl_cholesterol" in df.columns:
        score += (g("hdl_cholesterol") < 40).astype(int)
    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        score += ((g("systolic_bp") >= 130) | (g("diastolic_bp") >= 85)).astype(int)

    df["metabolic_syndrome_score"] = score

    # -------------------------------
    # Lifestyle proxy (simple, stable)
    # -------------------------------
    lifestyle_terms = []
    if "diet_score" in df.columns:
        lifestyle_terms.append(g("diet_score"))
    if "screen_time_hours_per_day" in df.columns:
        lifestyle_terms.append(-g("screen_time_hours_per_day"))
    if "pa_log" in df.columns:
        lifestyle_terms.append(df["pa_log"])
    if "heart_rate" in df.columns:
        lifestyle_terms.append(-g("heart_rate"))

    if lifestyle_terms:
        df["gen_health_proxy"] = np.mean(
            np.vstack([t.to_numpy() for t in lifestyle_terms]),
            axis=0,
        )

    # -------------------------------
    # Cleanup
    # -------------------------------
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

# ------------------------------------------------------------------
# 4. Build datasets
# ------------------------------------------------------------------

train_fe = build_optimal_fe(train, CLIP_BOUNDS)
test_fe  = build_optimal_fe(test,  CLIP_BOUNDS)
external_fe = build_optimal_fe(external, CLIP_BOUNDS)

print("FE shapes (train / test / external):",
      train_fe.shape, test_fe.shape, external_fe.shape)



print("\n=== Adversarial validation on FE ===")

train_av_fe = train_fe.drop(columns=[predict], errors="ignore").copy()
test_av_fe  = test_fe.copy()

train_av_fe["is_test"] = 0
test_av_fe["is_test"]  = 1

adv_fe = pd.concat([train_av_fe, test_av_fe], axis=0)
y_adv_fe = adv_fe["is_test"]
X_adv_fe = adv_fe.drop(columns=["is_test", "id"], errors="ignore")

adv_model_fe = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2025
)

skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_adv_fe = np.zeros(len(X_adv_fe))

for tr_idx, va_idx in skf_adv.split(X_adv_fe, y_adv_fe):
    adv_model_fe.fit(X_adv_fe.iloc[tr_idx], y_adv_fe.iloc[tr_idx])
    oof_adv_fe[va_idx] = adv_model_fe.predict_proba(X_adv_fe.iloc[va_idx])[:, 1]

adv_auc_fe = roc_auc_score(y_adv_fe, oof_adv_fe)
print(f"Adversarial AUC on FE: {adv_auc_fe:.4f}")


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

adv_lr = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
        )),
    ]
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X_adv_fe))

for tr, va in skf.split(X_adv_fe, y_adv_fe):
    adv_lr.fit(X_adv_fe.iloc[tr], y_adv_fe.iloc[tr])
    oof[va] = adv_lr.predict_proba(X_adv_fe.iloc[va])[:, 1]

print("Adversarial AUC on FE (LR):", roc_auc_score(y_adv_fe, oof))



print("\n=== Multi-model CV training on FE (fixed) ===")

import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# ------------------------------------------------
# Data
# ------------------------------------------------
X = train_fe.drop(columns=["id", predict], errors="ignore")
y = train[predict].astype(int)

X_test = (
    test_fe
    .drop(columns=["id"], errors="ignore")
    .reindex(columns=X.columns)
)

num_cols = X.columns[X.dtypes != "object"]

# ------------------------------------------------
# Model factory (IMPORTANT)
# ------------------------------------------------
def get_models():
    return {
        "xgb1": XGBClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
        ),

        "xgb2": XGBClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            eval_metric="logloss",
            random_state=52,
        ),

        "rf": RandomForestClassifier(
            n_estimators=600,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=62,
        ),

        "ext": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=92,
        ),

        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=1000,
                random_state=82,
            ))
        ]),
    }

models = get_models()
model_names = list(models.keys())

# ------------------------------------------------
# CV containers
# ------------------------------------------------
oof_preds = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(X_test), len(models)))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------------------
# CV loop
# ------------------------------------------------
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # fold-wise imputation (NO leakage)
    medians = X_tr[num_cols].median()
    X_tr[num_cols] = X_tr[num_cols].fillna(medians)
    X_va[num_cols] = X_va[num_cols].fillna(medians)
    X_test_fold = X_test.copy()
    X_test_fold[num_cols] = X_test_fold[num_cols].fillna(medians)

    for mi, name in enumerate(model_names):
        model = copy.deepcopy(models[name])

        if name.startswith("xgb"):
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=50,
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr)

        oof_preds[va_idx, mi] = model.predict_proba(X_va)[:, 1]
        test_preds[:, mi] += model.predict_proba(X_test_fold)[:, 1] / skf.n_splits

        auc = roc_auc_score(y_va, oof_preds[va_idx, mi])
        print(f"{name:6s} | Fold AUC: {auc:.5f}")

# ------------------------------------------------
# Final OOF AUC per model
# ------------------------------------------------
print("\n=== OOF AUC Summary ===")
for i, name in enumerate(model_names):
    auc = roc_auc_score(y, oof_preds[:, i])
    print(f"{name:6s} | OOF AUC: {auc:.5f}")



print("\n=== Calibrating base model predictions ===")

calibrated_oof = np.zeros_like(oof_preds)
calibrated_test = np.zeros_like(test_preds)

for mi in range(oof_preds.shape[1]):
    lr = LogisticRegression()
    lr.fit(oof_preds[:, mi].reshape(-1,1), y)
    
    calibrated_oof[:, mi]  = lr.predict_proba(oof_preds[:, mi].reshape(-1,1))[:, 1]
    calibrated_test[:, mi] = lr.predict_proba(test_preds[:, mi].reshape(-1,1))[:, 1]


print("\n=== Selecting elite 3-model stack ===")

# Compute volatility (std dev of OOF predictions)
volatility = oof_preds.std(axis=0)

# Single-model calibrated AUCs
single_auc = np.array([
    roc_auc_score(y, calibrated_oof[:, i])
    for i in range(len(models))
])

# Ranking table
rank_df = pd.DataFrame({
    "model": model_names,
    "oof_auc": single_auc,
    "volatility": volatility
})

# Risk-adjusted score — higher is better
rank_df["final_score"] = rank_df["oof_auc"] - 0.05 * rank_df["volatility"]
rank_df = rank_df.sort_values("final_score", ascending=False).reset_index(drop=True)

print(rank_df[["model", "oof_auc", "volatility", "final_score"]])

# Pick top 3 models
elite_models = rank_df["model"].head(3).tolist()
elite_indices = [model_names.index(m) for m in elite_models]

print("\nElite 3 models selected:", elite_models)

# Extract elite model predictions
elite_oof  = calibrated_oof[:, elite_indices]
elite_test = calibrated_test[:, elite_indices]

# Train meta-model ONLY on elite predictions
meta = LogisticRegression(max_iter=2000)
meta.fit(elite_oof, y)

stack_oof  = meta.predict_proba(elite_oof)[:, 1]
stack_test = meta.predict_proba(elite_test)[:, 1]

oof_auc_stack = roc_auc_score(y, stack_oof)
pr_auc_stack  = average_precision_score(y, stack_oof)

print("\n=== FINAL STACKED CV (Elite 3) ===")
print(f"OOF ROC-AUC: {oof_auc_stack:.5f}")
print(f"OOF PR-AUC : {pr_auc_stack:.5f}")



print("\n=== Model correlation heatmap (OOF) ===")

oof_df = pd.DataFrame(oof_preds, columns=model_names)
oof_df["stack"] = stack_oof

plt.figure(figsize=(10, 6))
sns.heatmap(oof_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Model Correlation Heatmap (OOF Predictions)")
plt.tight_layout()
plt.show()


print("\n=== Interpretability XGB on FE ===")

X_int = train_fe.drop(columns=["id", predict], errors="ignore")
y_int = train[predict].astype(int)

xgb_interp = XGBClassifier(
    n_estimators=700,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    random_state=2025,
)
xgb_interp.fit(X_int, y_int)

fi = pd.DataFrame({
    "feature": X_int.columns,
    "importance": xgb_interp.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(
    data=fi.head(40),
    x="importance",
    y="feature",
    orient="h"
)
plt.title("XGBoost Feature Importance (Top 40, FE)")
plt.tight_layout()
plt.show()

print("Running SHAP on interpretability model (FE)...")
shap.initjs()

sample_int = X_int.sample(min(3000, len(X_int)), random_state=42)
explainer_int = shap.TreeExplainer(xgb_interp)
shap_values_int = explainer_int(sample_int)

shap.summary_plot(shap_values_int, sample_int, plot_type="bar")
shap.summary_plot(shap_values_int, sample_int)


print("\n=== Leakage check (single-feature AUC) ===")

for col in X.columns:
    try:
        auc_col = roc_auc_score(y, X[col])
        if auc_col > 0.80:
            print(f"⚠ Potential leakage-like feature: {col}, single-feature AUC={auc_col:.3f}")
    except Exception:
        # Some engineered features might be constant or weird; ignore them
        continue



print("\n=== FE pruning suggestions (based on FI) ===")

SHAP_THRESHOLD = 1e-4
low_importance_features = fi[fi["importance"] < SHAP_THRESHOLD]["feature"].tolist()

base_safe = [
    "age", "bmi", "waist_to_hip_ratio", "triglycerides",
    "hdl_cholesterol", "ldl_cholesterol", "cholesterol_total",
    "systolic_bp", "diastolic_bp", "heart_rate",
    "physical_activity_minutes_per_week",
    "family_history_diabetes"
]

low_importance_features = [f for f in low_importance_features if f not in base_safe]

print("Low-importance engineered features (candidates to drop in a second pass):")
for f in low_importance_features:
    print(" -", f)


print("\n=== Creating submission variants ===")

# A: Elite stack (primary)
sub_A = pd.DataFrame({
    "id": test["id"],
    predict: stack_test.clip(0.001, 0.999),
})
sub_A.to_csv("submission.csv", index=False)

# B: Best single calibrated model (by OOF AUC)
best_single_idx = np.argmax(single_auc)
best_single_name = model_names[best_single_idx]
best_single_test = calibrated_test[:, best_single_idx]

sub_B = pd.DataFrame({
    "id": test["id"],
    predict: best_single_test.clip(0.001, 0.999),
})
sub_B.to_csv("submission_B.csv", index=False)

# C: Mean of all calibrated models
sub_C = pd.DataFrame({
    "id": test["id"],
    predict: calibrated_test.mean(axis=1).clip(0.001, 0.999),
})
sub_C.to_csv("submission_C.csv", index=False)

# D: Tempered version of elite stack (slightly less confident)
def temper(p, t=0.92):
    return np.power(p, t)

sub_D = pd.DataFrame({
    "id": test["id"],
    predict: temper(stack_test.clip(0.001, 0.999), t=0.92),
})
sub_D.to_csv("submission_D.csv", index=False)

print("\nSaved:")
print(" - submission.csv   (Primary candidate)")
print(" - submission_B_best_single.csv    (Stability backup)")
print(" - submission_C_calibrated_mean.csv")
print(" - submission_D_stack_tempered.csv")

# Quick sanity check on prediction distributions
plt.figure(figsize=(8, 6))
sns.kdeplot(sub_A[predict], label="Stack Elite3", fill=True, alpha=0.4)
sns.kdeplot(sub_B[predict], label="Best Single", fill=True, alpha=0.4)
sns.kdeplot(sub_C[predict], label="Calibrated Mean", fill=True, alpha=0.4)
sns.kdeplot(sub_D[predict], label="Stack Tempered", fill=True, alpha=0.4)
plt.title("Submission Prediction Distributions")
plt.legend()
plt.tight_layout()
plt.show()


print("\n=== Final checklist ===")
print("=== Note to self: ===")
print("""
1. Upload A/B/C/D to Kaggle.
2. Watch how public LB reacts:
   - If A > B, C, D → keep elite stack as main strategy.
   - If C or B beats A by a clear margin → your stack/FE may be overfitting; prefer that variant.
   - If D behaves more stably than A across time → consider D for final day.
3. If public LB is unstable or lower than expected:
   - Drop the low-importance engineered features printed above.
   - Retrain only on the pruned FE matrix.
   - Rebuild the elite 3-model stack.
4. Final 24–48 hours:
   - Pick 2–3 strongest, most stable submissions (from this notebook + pruned second pass).
   - Use the most stable high-scoring one as your final submission.
""")

