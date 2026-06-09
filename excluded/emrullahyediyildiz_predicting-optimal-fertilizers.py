# =======================
# 0) Load Kaggle Data
# =======================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

print("âœ… Data loaded from Kaggle paths:")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_submission.shape}")



train.head()


test.head()


sample_submission.head()


train.shape, test.shape


train.info()


train.describe()


cat_cols = ["Fertilizer Name", "Soil Type", "Crop Type"]

for col in cat_cols:
    plt.figure(figsize=(10,6))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index, palette="Set2")
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()



train.columns



# --- Identify target ---
target_col = list(set(train.columns) - set(test.columns))[0]
print("Target column:", target_col)

# --- 1. Target distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x=target_col, data=train, order=train[target_col].value_counts().index, palette="viridis")
plt.title(f"Target Distribution: {target_col}")
plt.show()

# --- 2. Categorical features ---
for col in ["Soil Type", "Crop Type", "Fertilizer Name"]:
    plt.figure(figsize=(10,5))
    sns.countplot(x=col, data=train, order=train[col].value_counts().index, palette="Set2")
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()

# --- 3. Selected numeric features ---
for col in ["Temparature", "Nitrogen"]:
    if col in train.columns:
        col_data = train[col].replace([np.inf, -np.inf], np.nan).dropna()
        plt.figure(figsize=(8,4))
        sns.histplot(col_data, kde=True, bins=30, color="skyblue")
        plt.title(f"{col} Distribution (cleaned)")
        plt.show()

# --- 4. Correlation heatmap ---
num_cols = train.select_dtypes(include=["int64","float64"]).columns
num_cols = [c for c in num_cols if c not in ["id", target_col]]

plt.figure(figsize=(10,6))
corr = train[num_cols].corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# --- 5. Example boxplot (Nitrogen vs Fertilizer Name) ---
plt.figure(figsize=(10,5))
sns.boxplot(data=train, x="Fertilizer Name", y="Nitrogen", palette="Set1")
plt.title("Nitrogen vs. Fertilizer Name")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Note: Similar boxplots for Phosphorous and Potassium show comparable distributions
# and are omitted for brevity.



# ============================
# v2 â€” minimal, GPU-safe LightGBM K-Fold (clean)
# ============================
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# toggles
QUICK_RUN = False          # â†� use full 600k for proper signal
USE_RATIO = True

# CV + params (keep them exactly like this)
FOLDS = 5
LR    = 0.02
N_EST = 5000
STOP  = 300

# detect target
target_col = list(set(train.columns) - set(test.columns))[0]
print("Target:", target_col)



# ---------- Feature Engineering ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Nutrients"] = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
    df["Avg_NPK"] = df["Total_Nutrients"] / 3.0
    df["log_TN"] = np.log1p(df["Total_Nutrients"])
    df["NPK_Ratio"] = df["Nitrogen"] / (df["Phosphorous"] + df["Potassium"] + 1.0)
    df["Nutrient_Imbalance"] = df[["Nitrogen","Phosphorous","Potassium"]].std(axis=1)
    df["N_share"] = df["Nitrogen"] / (df["Total_Nutrients"] + 1e-6)
    df["Soil_Moisture_Index"] = df["Moisture"] / (df["Humidity"] + 1.0)
    df["Temp_Nutrient_Interaction"] = df["Temparature"] * df["Total_Nutrients"]
    df["Hum_x_Mois"] = df["Humidity"] * df["Moisture"]
    soil_codes, _ = pd.factorize(df["Soil Type"])
    crop_codes, _ = pd.factorize(df["Crop Type"])
    df["SoilCropCombo"] = soil_codes * 10 + crop_codes
    return df

train_v2 = add_features(train)
test_v2  = add_features(test)


# ---------- Optional: ratio-to-label features ----------
if USE_RATIO:
    def _parse_label(lab: str):
        lab = str(lab).strip()
        if lab == "Urea": return np.array([46.,0.,0.])
        if lab == "DAP":  return np.array([18.,46.,0.])
        parts = lab.split("-")
        try:
            if len(parts)==3: return np.array(list(map(float, parts)))
            if len(parts)==2: return np.array([float(parts[0]), float(parts[1]), 0.])
        except: pass
        return np.array([0.,0.,0.])

    labels = sorted(train[target_col].astype(str).unique())
    vecs   = {lab: _parse_label(lab) for lab in labels}
    unit   = {lab: vecs[lab] / (np.linalg.norm(vecs[lab]) + 1e-9) for lab in labels}

    def add_ratio_feats(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        N = df["Nitrogen"].astype(float).values
        P = df["Phosphorous"].astype(float).values
        K = df["Potassium"].astype(float).values
        v = np.stack([N,P,K], axis=1)
        v_unit = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        sims = np.stack([(v_unit * unit[lab]).sum(axis=1) for lab in labels], axis=1)
        top = sims.argmax(axis=1)
        df["NearestLabel_byCosine"] = np.array(labels, dtype=object)[top]
        df["MaxCosine_toLabel"] = sims[np.arange(len(df)), top]
        return df

    train_v2 = add_ratio_feats(train_v2)
    test_v2  = add_ratio_feats(test_v2)

# cast categoricals
for c in ["Soil Type","Crop Type","NearestLabel_byCosine"]:
    if c in train_v2.columns:
        train_v2[c] = train_v2[c].astype("category")
        test_v2[c]  = test_v2[c].astype("category")
train_v2[target_col] = train_v2[target_col].astype("category")

# features / target
feat_cols = [c for c in train_v2.columns if c not in ["id", target_col]]
X, y = train_v2[feat_cols], train_v2[target_col]

# quick downsample to iterate faster
if QUICK_RUN:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=min(120_000, len(X)), replace=False)
    X, y = X.iloc[idx], y.iloc[idx]



def make_lgbm_classifier():
    try:
        return lgb.LGBMClassifier(device="gpu", objective="multiclass",
                                  n_estimators=N_EST, learning_rate=LR,
                                  num_leaves=96, max_depth=-1,
                                  subsample=0.9, colsample_bytree=0.9,
                                  min_child_samples=40, reg_lambda=1.0,
                                  random_state=42, n_jobs=-1)
    except TypeError:
        return lgb.LGBMClassifier(objective="multiclass",
                                  n_estimators=N_EST, learning_rate=LR,
                                  num_leaves=96, max_depth=-1,
                                  subsample=0.9, colsample_bytree=0.9,
                                  min_child_samples=40, reg_lambda=1.0,
                                  random_state=42, n_jobs=-1)

# ---- K-Fold Loop (clean) ----
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_proba   = np.zeros((len(X), y.nunique()))
test_proba  = np.zeros((len(test_v2), y.nunique()))
class_order = None
scores      = []

# fÃ¼r mittlere Feature-Importance Ã¼ber Folds
importances = np.zeros(len(feat_cols), dtype=float)

for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold}/{FOLDS} =====")
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    model = make_lgbm_classifier()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        categorical_feature=[c for c in ["Soil Type","Crop Type","NearestLabel_byCosine"] if c in X_tr.columns],
        callbacks=[lgb.early_stopping(STOP), lgb.log_evaluation(200)]
    )

    # --- Feature-Importance (gain) sammeln
    importances += model.booster_.feature_importance(importance_type="gain")

    # --- Vorhersagen
    proba_va = model.predict_proba(X_va)
    proba_te = model.predict_proba(test_v2[feat_cols])

    if class_order is None:
        class_order = list(model.classes_)

    # Klassenreihenfolge angleichen
    fold_cols = list(model.classes_)
    remap = [fold_cols.index(c) for c in class_order]
    proba_va = proba_va[:, remap]
    proba_te = proba_te[:, remap]

    # speichern
    oof_proba[va, :] = proba_va
    test_proba += proba_te / FOLDS

    # Fold-Score
    y_va_pred = np.array(class_order)[proba_va.argmax(axis=1)]
    f1 = f1_score(y_va, y_va_pred, average="macro")
    scores.append(f1)
    print(f"[Fold {fold}] Macro F1 = {f1:.4f}")


# OOF macro-F1
oof_pred = np.array(class_order)[oof_proba.argmax(axis=1)]
print(f"\nOOF Macro F1: {f1_score(y, oof_pred, average='macro'):.4f} | Folds: {[f'{s:.4f}' for s in scores]}")



# Use the nearest-label feature as a rule-based prediction on the last fold's val set
rb_pred = X_va["NearestLabel_byCosine"].astype(str).values
print("Rule-based Macro F1 (last fold):", f1_score(y_va.astype(str), rb_pred, average="macro"))



# --- NACH der Schleife: Importance mitteln/plotten
fi = pd.Series(importances / FOLDS, index=feat_cols).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=fi.values, y=fi.index)
plt.title("Feature Importance (gain, mean over folds)")
plt.xlabel("Gain importance")
plt.ylabel("Feature")
plt.show()


# === Results only (no retrain) ===
from sklearn.metrics import f1_score
import numpy as np

# OOF metric
oof_pred = np.array(class_order)[oof_proba.argmax(axis=1)]
print("OOF Macro F1:", f1_score(y, oof_pred, average="macro"))

# CV-ensemble submission
sub_cv = sample_submission.copy()
sub_cv[target_col] = np.array(class_order)[test_proba.argmax(axis=1)]
sub_cv.to_csv("submission_v2_cv.csv", index=False)
print("ğŸ“‚ saved: submission_v2_cv.csv")





