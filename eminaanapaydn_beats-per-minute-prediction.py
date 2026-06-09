import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Filepaths (Kaggle)
TRAIN_PATH = '/kaggle/input/playground-series-s5e9/train.csv'
TEST_PATH  = '/kaggle/input/playground-series-s5e9/test.csv'
SAMPLE_PATH= '/kaggle/input/playground-series-s5e9/sample_submission.csv'

df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)
df_sample= pd.read_csv(SAMPLE_PATH)

# Test id’lerini sakla (submission için)
test_ids = df_test['id'].copy()

print(df_train.shape, df_test.shape)
df_train.head(3)


# İnfinity değerleri NaN yap (bazı görsellerde uyarıları keser)
df_train_eda = df_train.replace([np.inf, -np.inf], np.nan).copy()
df_test_eda  = df_test.replace([np.inf, -np.inf], np.nan).copy()

# Hızlı yardımcılar
TARGET = "BeatsPerMinute"
FEATURES = [c for c in df_train.columns if c not in ["id", TARGET]]
print("Train shape:", df_train.shape, "| Test shape:", df_test.shape)
print("Num features:", len(FEATURES))


# === EDA-1: KALİTE KONTROLLERİ ===
print("Missing (train):\n", df_train_eda[FEATURES + [TARGET]].isna().sum().sort_values(ascending=False)[:10], "\n")
print("Missing (test):\n", df_test_eda[FEATURES].isna().sum().sort_values(ascending=False)[:10], "\n")

print("Duplicate rows in train:", int(df_train_eda.duplicated().sum()))
# Sabit sütun var mı?
nunique = df_train_eda[FEATURES + [TARGET]].nunique()
constant_cols = list(nunique[nunique==1].index)
print("Constant columns:", constant_cols if constant_cols else "None")


# === EDA-2: TARGET DAĞILIMI ===
fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.histplot(df_train_eda[TARGET], bins=60, kde=True, ax=ax[0])
ax[0].set_title("BeatsPerMinute Distribution")

sns.boxplot(x=df_train_eda[TARGET], ax=ax[1])
ax[1].set_title("BeatsPerMinute Boxplot")
plt.show()

print(df_train_eda[TARGET].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]))


# === EDA-3: SAYISAL KOLONLARIN DAĞILIMLARI ===
num_cols = FEATURES  # hepsi numerik
n = min(12, len(num_cols))
cols = 3
rows = int(np.ceil(n/cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3.5))
axes = axes.flatten()

for i, col in enumerate(num_cols[:n]):
    sns.histplot(df_train_eda[col], bins=60, kde=True, ax=axes[i])
    axes[i].set_title(f"{col} dist.")
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.show()


# === EDA-4: TRAIN vs TEST KARŞILAŞTIRMA ===
sample_tr = df_train_eda.sample(n=50000, random_state=42) if len(df_train_eda)>50000 else df_train_eda
sample_te = df_test_eda.sample(n=50000, random_state=42)  if len(df_test_eda)>50000  else df_test_eda

fig, axes = plt.subplots(3, 3, figsize=(15,12))
axes = axes.flatten()
check_cols = ["Energy","RhythmScore","AudioLoudness","MoodScore","InstrumentalScore","LivePerformanceLikelihood","AcousticQuality","TrackDurationMs","VocalContent"]

for i, col in enumerate(check_cols):
    sns.kdeplot(sample_tr[col], ax=axes[i], label="train")
    sns.kdeplot(sample_te[col], ax=axes[i], label="test")
    axes[i].set_title(f"Train vs Test: {col}")
    axes[i].legend()
plt.tight_layout()
plt.show()


# === EDA-5: KORELASYON (TARGET) ===
corr_series = df_train_eda[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
plt.figure(figsize=(6,6))
sns.barplot(y=corr_series.index, x=corr_series.values)
plt.title("Correlation with BPM")
plt.xlabel("Pearson r")
plt.ylabel("")
plt.show()

display(corr_series.to_frame("corr_with_BPM").round(4))


# === EDA-6: KORELASYON HEATMAP (SAMPLE) ===
corr_sample = df_train_eda.sample(30000, random_state=42) if len(df_train_eda)>30000 else df_train_eda
corr_mat = corr_sample[FEATURES + [TARGET]].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_mat, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation Heatmap (sampled)")
plt.show()


# === EDA-7: TARGET vs FEATURE SCATTER (SAMPLE) ===
sc_sample = df_train_eda.sample(40000, random_state=123) if len(df_train_eda)>40000 else df_train_eda
top_pos = corr_series.head(4).index.tolist()
top_neg = corr_series.tail(4).index.tolist()
plot_cols = top_pos + top_neg

cols = 3
rows = int(np.ceil(len(plot_cols)/cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()

for i, col in enumerate(plot_cols):
    sns.scatterplot(data=sc_sample, x=col, y=TARGET, alpha=0.25, s=10, ax=axes[i])
    axes[i].set_title(f"{col} vs BPM")
for j in range(i+1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.show()


# === EDA-8: BİNLENMİŞ FEATURE → BPM BOXPLOT ===
def qbin(series, q=4):
    try:
        return pd.qcut(series, q=q, labels=[f"Q{i}" for i in range(1,q+1)])
    except Exception:
        # eşit olmayan dağılımda sorun olursa fallback
        return pd.cut(series, bins=q, labels=[f"B{i}" for i in range(1,q+1)])

box_cols = ["LivePerformanceLikelihood","Energy","RhythmScore","AudioLoudness"]
fig, axes = plt.subplots(2, 2, figsize=(12,8))
axes = axes.flatten()
for i, col in enumerate(box_cols):
    sns.boxplot(x=qbin(df_train_eda[col], q=4), y=df_train_eda[TARGET], ax=axes[i])
    axes[i].set_title(f"BPM by {col} quartiles")
plt.tight_layout()
plt.show()


# === EDA-9: LGBM FEATURE IMPORTANCE (HIZLI) ===
from lightgbm import LGBMRegressor

X_imp = df_train_eda[FEATURES].copy()
y_imp = df_train_eda[TARGET].astype(float).values

lgb_imp = LGBMRegressor(
    n_estimators=400, learning_rate=0.05,
    num_leaves=64, subsample=0.8, colsample_bytree=0.9,
    random_state=42, n_jobs=-1
)
lgb_imp.fit(X_imp, y_imp)

imp = pd.Series(lgb_imp.feature_importances_, index=FEATURES).sort_values(ascending=False)
plt.figure(figsize=(6,6))
sns.barplot(y=imp.head(20).index, x=imp.head(20).values)
plt.title("Top-20 Feature Importances (LGBM)")
plt.xlabel("gain/importance")
plt.ylabel("")
plt.show()

display(imp.to_frame("importance").head(30))


# === EDA-10: TRAIN vs TEST UYUM METRİĞİ (KABA) ===
# PSI benzeri kaba bir ölçü: histogram farklarının toplamı (sadece fikir verir)
def psi_like(train_s, test_s, bins=20):
    t_hist, edges = np.histogram(train_s, bins=bins, density=True)
    q_hist, _     = np.histogram(test_s,  bins=edges, density=True)
    t_hist = np.where(t_hist==0, 1e-9, t_hist)
    q_hist = np.where(q_hist==0, 1e-9, q_hist)
    return np.sum((q_hist - t_hist) * np.log(q_hist / t_hist))

psi_scores = {col: psi_like(df_train_eda[col].values, df_test_eda[col].values, bins=20) for col in FEATURES}
psi_ser = pd.Series(psi_scores).sort_values(ascending=False)
display(psi_ser.head(10))
plt.figure(figsize=(6,5))
sns.barplot(y=psi_ser.head(10).index, x=psi_ser.head(10).values)
plt.title("Top Train-Test Shift (psi-like)")
plt.xlabel("shift score (higher = more shift)")
plt.ylabel("")
plt.show()


# === 3) FEATURE ENGINEERING ===
def make_features(df):
    df = df.copy()

    # 1) Basic conversions
    df["DurationMin"] = df["TrackDurationMs"] / 60000.0
    df["LoudnessPos"] = -df["AudioLoudness"]  # dB negatif → pozitif ölçek

    # 2) Interactions
    df["Energy_x_Mood"] = df["Energy"] * df["MoodScore"]
    df["Energy_x_Live"] = df["Energy"] * df["LivePerformanceLikelihood"]
    df["Mood_x_Live"] = df["MoodScore"] * df["LivePerformanceLikelihood"]
    df["Acoustic_x_Instrumental"] = df["AcousticQuality"] * df["InstrumentalScore"]
    df["Vocal_over_Instrumental"] = df["VocalContent"] / (df["InstrumentalScore"] + 1e-6)

    # 3) Non-linear
    df["Energy_sq"] = df["Energy"] ** 2
    df["Mood_sq"] = df["MoodScore"] ** 2
    df["Rhythm_sq"] = df["RhythmScore"] ** 2
    df["sqrt_LoudnessPos"] = np.sqrt(df["LoudnessPos"].clip(lower=0))

    # 4) Robust clip
    q1, q99 = df["DurationMin"].quantile([0.01, 0.99])
    df["DurationMin"] = df["DurationMin"].clip(lower=q1, upper=q99)

    return df

feature_cols_base = [
    "RhythmScore","AudioLoudness","VocalContent","AcousticQuality",
    "InstrumentalScore","LivePerformanceLikelihood","MoodScore",
    "TrackDurationMs","Energy"
]
target_col = "BeatsPerMinute"

X_full = make_features(df_train[feature_cols_base])
y_full = df_train[target_col].astype(float).values
X_test_full = make_features(df_test[feature_cols_base])

bpm_min, bpm_max = y_full.min(), y_full.max()
print("Train shape:", X_full.shape, "| Test shape:", X_test_full.shape, "| BPM range:", (bpm_min, bpm_max))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def kfold_oof(model_ctor, X, y, X_test, n_splits=5, seed=42, scale=False, verbose=True):
    """
    model_ctor : lambda/func → fresh model
    scale      : True ise StandardScaler + model pipeline
    return     : oof_preds, test_preds_mean, cv_rmse
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)
    test_fold_preds = []

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        base_model = model_ctor()
        model = Pipeline([("scaler", StandardScaler()), ("model", base_model)]) if scale else base_model
        model.fit(X_tr, y_tr)

        oof[va] = model.predict(X_va)
        test_fold_preds.append(model.predict(X_test))
        if verbose:
            print(f"  Fold {fold} RMSE: {rmse(y_va, oof[va]):.5f}")

    oof_rmse = rmse(y, oof)
    test_mean = np.mean(np.vstack(test_fold_preds), axis=0)
    if verbose:
        print(f"OOF RMSE: {oof_rmse:.5f}")
    return oof, test_mean, oof_rmse


from sklearn.linear_model import Lasso, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

elastic_ctor = lambda: ElasticNet(alpha=0.001, l1_ratio=0.5)

lgbm_ctor = lambda: LGBMRegressor(
    n_estimators=800, learning_rate=0.04,
    num_leaves=64, subsample=0.85, colsample_bytree=0.90,
    random_state=42, n_jobs=-1
)

xgb_ctor = lambda: XGBRegressor(
    n_estimators=900, learning_rate=0.04, max_depth=8,
    subsample=0.85, colsample_bytree=0.90,
    random_state=42, n_jobs=-1, verbosity=0
)


print(">> ElasticNet (scaled)")
oof_lin, test_lin, cv_lin = kfold_oof(elastic_ctor, X_full, y_full, X_test_full, n_splits=5, seed=42, scale=True)

print("\n>> LightGBM")
oof_lgb, test_lgb, cv_lgb = kfold_oof(lgbm_ctor, X_full, y_full, X_test_full, n_splits=5, seed=42, scale=False)

print("\n>> XGBoost")
oof_xgb, test_xgb, cv_xgb = kfold_oof(xgb_ctor, X_full, y_full, X_test_full, n_splits=5, seed=42, scale=False)

print("\nOOF RMSE | Lasso: %.5f | LGBM: %.5f | XGB: %.5f" % (cv_lin, cv_lgb, cv_xgb))


best = (1e18, None)
# Linear ağırlığını genelde yüksek tutmak iyi (bu task'ta lineer güçlü)
for w_lin in np.arange(0.35, 0.80, 0.05):
    for w_lgb in np.arange(0.10, 0.60, 0.05):
        w_xgb = 1.0 - w_lin - w_lgb
        if w_xgb < 0 or w_xgb > 0.50:  # XGB'nin payını sınırlı tut
            continue
        oof_blend = w_lin*oof_lin + w_lgb*oof_lgb + w_xgb*oof_xgb
        score = rmse(y_full, oof_blend)
        if score < best[0]:
            best = (score, (w_lin, w_lgb, w_xgb))

best_rmse, (w_lin, w_lgb, w_xgb) = best
print("Best OOF blend RMSE: %.5f | weights → lin:%.2f  lgb:%.2f  xgb:%.2f" % (best_rmse, w_lin, w_lgb, w_xgb))


# OOF'tan gelen test mean tahminlerini ağırlıklı ortala
blend = w_lin*test_lin + w_lgb*test_lgb + w_xgb*test_xgb
blend = np.clip(blend, bpm_min, bpm_max)

# Quantile mapping (distribution shaping)
def quantile_map(y_ref, preds, noise=1e-6):
    preds = preds.astype(float) + np.random.normal(0, noise, size=len(preds))  # tie kır
    pct = pd.Series(preds).rank(pct=True).values
    return np.quantile(y_ref, pct)

# Adaylar
pred_safe   = blend.copy()  # QM yok → en stabil
pred_qm     = np.clip(quantile_map(y_full, blend), bpm_min, bpm_max)  # full QM → risky
alpha       = 0.4  # hybrid karışım oranı (0.3–0.5 iyi aralık)
pred_hybrid = np.clip((1 - alpha)*blend + alpha*pred_qm, bpm_min, bpm_max)


sub_safe = pd.DataFrame({"id": test_ids, "BeatsPerMinute": pred_safe})
sub_safe.to_csv("submission.csv", index=False)

sub_hybrid = pd.DataFrame({"id": test_ids, "BeatsPerMinute": pred_hybrid})
sub_hybrid.to_csv("submission_hybrid_qm.csv", index=False)

sub_risky = pd.DataFrame({"id": test_ids, "BeatsPerMinute": pred_qm})
sub_risky.to_csv("submission_risky_fullqm.csv", index=False)

print("Saved: submission_safe_noqm.csv | submission_hybrid_qm.csv | submission_risky_fullqm.csv")
display(sub_safe.head(3))

