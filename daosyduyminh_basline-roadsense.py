# ==============================================================================
# RoadSense (etiq-roadsense) – Phiên bản nâng cấp v2.0
# Cải tiến:
# - Sử dụng Stratified K-Fold Cross-Validation để tăng độ ổn định.
# - Feature Engineering nâng cao: làm sạch biến phân loại, thêm đặc trưng thời gian
#   (cuối tuần, buổi trong ngày, đặc trưng tuần hoàn sin/cos).
# - Tối ưu hóa ngưỡng trên toàn bộ dự đoán OOF (Out-of-Fold).
# - Ensemble (trung bình) các dự đoán từ tất cả các fold.
# ==============================================================================

import os, warnings, re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")

COMP_DIR = "/kaggle/input/etiq-roadsense"

# ------------------------------- Utils -------------------------------
def read_csv_safe(path):
    na_vals = ["", " ", "  ", "NA", "N/A", "NULL", "null", "-", "na"]
    return pd.read_csv(path, na_values=na_vals, keep_default_na=True)

def normalize_accident_key(s):
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.replace(r"^0+(\d+)$", r"\1", regex=True)
    return s

def coerce_accident_id(df, col="AccidentId"):
    if col in df.columns:
        df[col] = normalize_accident_key(df[col])
    return df

def reduce_many_to_one(df, key="AccidentId", max_cat=20, prefix="pl"):
    df = df.copy()
    df = coerce_accident_id(df, key)
    df = df.loc[:, df.notna().any(axis=0)]
    if df.empty: return pd.DataFrame({key: []})
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != key]
    out = df.groupby(key).size().to_frame(f"{prefix}_n_rows")
    if num_cols:
        num_cols = [c for c in num_cols if c != key]
        if num_cols:
            g = df.groupby(key)[num_cols].agg(["mean","min","max"])
            g.columns = [f"{prefix}_{a}_{b}" for a,b in g.columns]
            out = out.join(g, how="left")
    for c in cat_cols:
        mode_series = df[[key, c]].dropna().groupby(key)[c].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0] if len(s) else np.nan)
        out[f"{prefix}_mode_{c}"] = mode_series
        if df[c].nunique(dropna=True) <= max_cat:
            vc = df.groupby([key, c]).size().unstack(fill_value=0)
            vc.columns = [f"{prefix}_cnt_{c}={str(col)}" for col in vc.columns]
            vc_ratio = vc.div(out[f"{prefix}_n_rows"], axis=0)
            vc_ratio.columns = [col.replace("cnt_", "ratio_") for col in vc.columns]
            out = out.join(vc).join(vc_ratio)
    return out.reset_index()

def aggregate_entity(df, key="AccidentId", id_col=None, max_cat=12, prefix="ent"):
    df = df.copy()
    df = coerce_accident_id(df, key)
    df = df.loc[:, df.notna().any(axis=0)]
    if df.empty: return pd.DataFrame({key: []})
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != key]
    out = df.groupby(key).size().to_frame(f"{prefix}_n_rows")
    if id_col and id_col in df.columns:
        out[f"{prefix}_nunique_{id_col}"] = df.groupby(key)[id_col].nunique()
    if num_cols:
        num_cols = [c for c in num_cols if c != key]
        if num_cols:
            g = df.groupby(key)[num_cols].agg(["mean","min","max"])
            g.columns = [f"{prefix}_{a}_{b}" for a,b in g.columns]
            out = out.join(g, how="left")
    for c in cat_cols:
        if df[c].nunique(dropna=True) <= max_cat:
            vc = df.groupby([key, c]).size().unstack(fill_value=0)
            vc.columns = [f"{prefix}_cnt_{c}={str(col)}" for col in vc.columns]
            vc_ratio = vc.div(out[f"{prefix}_n_rows"], axis=0)
            vc_ratio.columns = [col.replace("cnt_", "ratio_") for col in vc.columns]
            out = out.join(vc).join(vc_ratio)
    return out.reset_index()

def clean_categorical_features(df):
    """Chuẩn hóa các giá trị trong các cột phân loại."""
    df = df.copy()
    if "Gender" in df.columns:
        gender_map = {"Male": "M", "Homme": "M", "H": "M", "Female": "F", "Femme": "F", "0": "M", "1": "F"}
        df["Gender"] = df["Gender"].astype(str).map(gender_map).fillna(df["Gender"])
    if "SafetyDeviceUsed" in df.columns:
        safety_map = {"Yes": "True", "Y": "True", "1": "True", "No": "False", "Non": "False", "0": "False"}
        df["SafetyDeviceUsed"] = df["SafetyDeviceUsed"].astype(str).map(safety_map).fillna(df["SafetyDeviceUsed"])
    return df

def enrich_time_advanced(df):
    """Tạo thêm các đặc trưng thời gian nâng cao."""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["year"]  = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"]   = df["Date"].dt.day
        df["dow"]   = df["Date"].dt.dayofweek
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
    else:
        for c in ["year","month","day","dow", "is_weekend"]:
            df[c] = np.nan

    if "Hour" in df.columns:
        h = pd.to_datetime(df["Hour"], errors="coerce", format="%H:%M:%S")
        df["hour"] = np.where(h.notna(), h.dt.hour, pd.to_numeric(df["Hour"], errors="coerce"))
        conditions = [
            (df['hour'] >= 5) & (df['hour'] < 12),
            (df['hour'] >= 12) & (df['hour'] < 17),
            (df['hour'] >= 17) & (df['hour'] < 21),
            (df['hour'] >= 21) | (df['hour'] < 5)
        ]
        choices = ['morning', 'afternoon', 'evening', 'night']
        df['time_of_day'] = np.select(conditions, choices, default='unknown')
    else:
        df["hour"] = np.nan
        df["time_of_day"] = "unknown"

    for col, max_val in [('month', 12), ('day', 31), ('dow', 7), ('hour', 24)]:
        if col in df.columns and df[col].notna().any():
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def normalize_gravity(s):
    s = s.astype(str).str.strip().str.lower()
    s = s.replace({"lethal": "lethal", "nonlethal": "nonlethal", "non lethal": "nonlethal", "non_lethal": "nonlethal", '"lethal"': "lethal", '"nonlethal"': "nonlethal"})
    return s

def align_train_test_columns(tr, te, exclude=("AccidentId", "Gravity", "_Gravity_norm_")):
    tr_cols = set(tr.columns) - set(exclude)
    te_cols = set(te.columns) - set(exclude)
    for col in sorted(tr_cols - te_cols): te[col] = np.nan
    for col in sorted(te_cols - tr_cols): tr[col] = np.nan
    te = te.reindex(columns=list(tr.columns), fill_value=np.nan)
    return tr, te

def safe_merge(left, right, on="AccidentId"):
    left  = coerce_accident_id(left, on).copy()
    right = coerce_accident_id(right, on).copy()
    dup = [c for c in right.columns if c in left.columns and c != on]
    if dup: right = right.rename(columns={c: f"{c}__r" for c in dup})
    return left.merge(right, on=on, how="left")

def build_panel(acc, pl_agg, us_agg, ve_agg):
    out = safe_merge(acc, pl_agg, on="AccidentId")
    out = safe_merge(out, us_agg, on="AccidentId")
    out = safe_merge(out, ve_agg, on="AccidentId")
    return out

# ------------------------------- Load & Preprocess -------------------------------
acc_tr  = read_csv_safe(f"{COMP_DIR}/accidents_train.csv")
acc_te  = read_csv_safe(f"{COMP_DIR}/accidents_test.csv")
pl_tr   = read_csv_safe(f"{COMP_DIR}/places_train.csv")
pl_te   = read_csv_safe(f"{COMP_DIR}/places_test.csv")
us_tr   = read_csv_safe(f"{COMP_DIR}/users_train.csv")
us_te   = read_csv_safe(f"{COMP_DIR}/users_test.csv")
ve_tr   = read_csv_safe(f"{COMP_DIR}/vehicles_train.csv")
ve_te   = read_csv_safe(f"{COMP_DIR}/vehicles_test.csv")

us_tr = clean_categorical_features(us_tr)
us_te = clean_categorical_features(us_te)

for _df in [acc_tr, acc_te, pl_tr, pl_te, us_tr, us_te, ve_tr, ve_te]:
    coerce_accident_id(_df, "AccidentId")

acc_tr = enrich_time_advanced(acc_tr)
acc_te = enrich_time_advanced(acc_te)

pl_tr_agg = reduce_many_to_one(pl_tr, key="AccidentId", max_cat=20, prefix="pl")
pl_te_agg = reduce_many_to_one(pl_te, key="AccidentId", max_cat=20, prefix="pl")
users_tr_agg = aggregate_entity(us_tr, key="AccidentId", id_col="VehicleId", prefix="user")
users_te_agg = aggregate_entity(us_te, key="AccidentId", id_col="VehicleId", prefix="user")
vehicles_tr_agg = aggregate_entity(ve_tr, key="AccidentId", id_col="VehicleId", prefix="veh")
vehicles_te_agg = aggregate_entity(ve_te, key="AccidentId", id_col="VehicleId", prefix="veh")

tr = build_panel(acc_tr, pl_tr_agg, users_tr_agg, vehicles_tr_agg)
te = build_panel(acc_te, pl_te_agg, users_te_agg, vehicles_te_agg)

if tr["AccidentId"].duplicated().any(): tr = tr.drop_duplicates("AccidentId", keep="first")
if te["AccidentId"].duplicated().any(): te = te.drop_duplicates("AccidentId", keep="first")

# ------------------------------- Target -------------------------------
tr["_Gravity_norm_"] = normalize_gravity(tr["Gravity"])
target_map = {"nonlethal":0, "lethal":1}
y = tr["_Gravity_norm_"].map(target_map)
bad = y.isna()
if bad.any():
    print(f"[WARN] Found {bad.sum()} rows with invalid Gravity -> dropping.")
    tr = tr.loc[~bad].copy()
    y = y.loc[~bad].copy()
y = y.astype(int)
tr = tr.drop(columns=["Gravity", "_Gravity_norm_"])

# ------------------------------- Feature Preparation -------------------------------
for col in ["Date", "Hour"]:
    if col in tr.columns: tr = tr.drop(columns=[col])
    if col in te.columns: te = te.drop(columns=[col])

tr, te = align_train_test_columns(tr, te, exclude=("AccidentId",))

for df in (tr, te):
    for c in df.columns:
        if c == "AccidentId": continue
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().fillna("__MISSING__")

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in tr.columns if c not in num_cols and c != "AccidentId"]

for c in num_cols:
    med = tr[c].median()
    tr[c] = tr[c].fillna(med)
    te[c] = te[c].fillna(med)

if cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr[cat_cols] = oe.fit_transform(tr[cat_cols].astype(str))
    te[cat_cols] = oe.transform(te[cat_cols].astype(str))

feature_cols = [c for c in tr.columns if c != "AccidentId"]
X = tr[feature_cols].copy()
X_test = te[feature_cols].copy()

# ------------------------------- LightGBM with Stratified K-Fold CV -------------------------------
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importances = pd.DataFrame(index=feature_cols)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"===== Fold {fold+1} =====")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_

# ------------------------------- Threshold Tuning on OOF -------------------------------
def best_threshold(probs, y_true):
    best_f1, best_t = -1, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true, (probs >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

t_best, f1_best = best_threshold(oof_preds, y)
print(f"\n[OOF CV] Best threshold = {t_best:.3f} | Macro-F1 = {f1_best:.4f}")
print("Classification Report on OOF predictions:")
print(classification_report(y, (oof_preds >= t_best).astype(int), digits=4))

# ------------------------------- Predict & Submit -------------------------------
test_pred_final = (test_preds >= t_best).astype(int)
inv_target_map = {0: "NonLethal", 1: "Lethal"}
sub = pd.DataFrame({
    "AccidentId": te["AccidentId"],
    "Gravity": pd.Series(test_pred_final, index=te.index).map(inv_target_map)
})
print("\nSubmission file head:")
print(sub.head())
out_path = "/kaggle/working/submission_upgraded.csv"
sub.to_csv(out_path, index=False)
print("Saved:", out_path)

# ------------------------------- Feature Importance -------------------------------
feature_importances['mean'] = feature_importances.mean(axis=1)
feature_importances.sort_values('mean', ascending=False, inplace=True)
print("\nTop-50 features by mean importance across folds:\n")
print(feature_importances.head(50)['mean'].to_string())

