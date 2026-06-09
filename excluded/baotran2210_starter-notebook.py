# =============================================================
# Baseline v4 — AI-Based Modeling for Energy-Efficient Buildings (Kaggle)
# Fixes:
# 1) **IndexingError (unalignable boolean Series)** → Chuẩn hoá timezone và **align** y với X bằng reindex.
# 2) Panel lúc trước chỉ phủ 1 tháng → **gộp theo sensor (object_id)** qua nhiều tháng rồi mới concat.
# 3) Lọc cảm biến theo **coverage trong TRAIN (Jan–May)** để tránh cột chỉ có ở Jul.
# 4) Giữ đúng **submission**: ID, TARGET_VARIABLE trên lưới 10 phút (Jun–Jul), tuân thủ **lead 3h**.
# =============================================================

import os, sys, glob, warnings, re
warnings.filterwarnings("ignore")
from typing import List, Dict, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb

# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "INPUT_DIR": "/kaggle/input/ai-based-modeling-for-energy-efficient-buildings",
    "BUILDING_ID": "B205",

    # Month folders present (Jan..Jul 2025)
    "MONTH_FOLDERS": [
        "RBHU-2025-01",
        "RBHU-2025-02",
        "RBHU-2025-03",
        "RBHU-2025-04",
        "RBHU-2025-05",
        "RBHU-2025-06",
        "RBHU-2025-07",
    ],

    # Train on Jan..May only (labels exist there per competition)
    "TRAIN_MONTH_INDEXES": [0,1,2,3,4],
    # Submission months (fixed): Jun & Jul 2025
    "SUBMIT_MONTH_INDEXES": [5,6],

    # Target sensor (organizer-defined)
    "TARGET_OID": "B205WC000.AM02",

    # Resample frequency MUST be 10 minutes to match the submission grid
    "RESAMPLE_FREQ": "10min",

    # Feature selection / filtering
    "MAX_SENSOR_FEATURES": 140,
    "MIN_TRAIN_COVERAGE": 0.25,   # >=25% số mốc trong Jan–May
    "ROLL_WINDOWS": [3, 6, 12, 36, 72, 144],  # 30m,1h,2h,6h,12h,24h (10min steps)
    "LAG_STEPS":    [1, 2, 3, 6, 12, 18, 36],

    # Causality lead: 3h → 18 steps
    "LEAD_STEPS": 18,

    # LightGBM params
    "LGB_PARAMS": {
        "objective": "regression",
        "metric": ["l1", "l2"],
        "learning_rate": 0.05,
        "num_leaves": 96,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "n_estimators": 6000,
        "verbosity": -1,
        "random_state": 42,
    },

    # Clip predictions để tránh outlier quá đà
    "CLIP_PREDICTIONS": True,
}

# -----------------------------
# IO helpers
# -----------------------------

def _join_input(*parts: str) -> str:
    return os.path.join(CONFIG["INPUT_DIR"], *parts)


def read_metadata() -> pd.DataFrame:
    p1 = _join_input("metadata.parquet"); p2 = _join_input("metadata.xlsx")
    if os.path.exists(p1):
        meta = pd.read_parquet(p1)
    elif os.path.exists(p2):
        meta = pd.read_excel(p2)
    else:
        print("[WARN] metadata not found — proceeding with filenames only.")
        return pd.DataFrame()
    for c in ["object_id","description","dimension_text","bde_channel_typ","device_class","device","channel","file","data_group_id","class_id"]:
        if c in meta.columns:
            meta[c] = meta[c].astype(str)
    return meta


def list_parquet_files_for_building(month_folder: str, building_id: str) -> List[str]:
    abs_month = _join_input(month_folder)
    patterns = [
        os.path.join(abs_month, "RBHU", building_id, "**", "*.parquet"),
        os.path.join(abs_month, building_id, "**", "*.parquet"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(list(set(files)))


# -----------------------------
# Time utilities
# -----------------------------

def parse_year_month_from_folder(name: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"(20\d{2})[-_/](0[1-9]|1[0-2])", name)
    if not m:
        m = re.search(r"(20\d{2}).*?(0[1-9]|1[0-2])", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def build_months_grid(idxs: List[int], freq: str) -> pd.DatetimeIndex:
    # Tạo lưới thời gian cho các tháng (UTC-naive)
    stamps: List[pd.DatetimeIndex] = []
    for i in idxs:
        if i < 0 or i >= len(CONFIG["MONTH_FOLDERS"]):
            continue
        y, m = parse_year_month_from_folder(CONFIG["MONTH_FOLDERS"][i])
        if y is None:
            continue
        start = pd.Timestamp(year=y, month=m, day=1, hour=0, minute=0, second=0)
        # lấy ngày cuối tháng bằng trick: thêm 32 ngày rồi về đầu tháng, lùi 1 phút
        if m == 12:
            y2, m2 = y+1, 1
        else:
            y2, m2 = y, m+1
        end = pd.Timestamp(year=y2, month=m2, day=1, hour=0, minute=0, second=0) - pd.Timedelta(minutes=10)
        stamps.append(pd.date_range(start, end, freq=freq))
    if not stamps:
        return pd.DatetimeIndex([])
    return stamps[0].union_many(stamps[1:])


# -----------------------------
# Loading & preprocessing series
# -----------------------------

def to_utc_naive(ts: pd.Series) -> pd.Series:
    # Đưa về UTC-aware rồi bỏ timezone để tránh mismatch
    ts = pd.to_datetime(ts, errors="coerce", utc=True)
    return ts.dt.tz_convert(None)


def load_and_resample_sensor(parquet_path: str, freq: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Tìm cột thời gian
    ts_col = None
    for c in df.columns:
        lc = c.lower()
        if "time" in lc or "timestamp" in lc:
            ts_col = c; break
    if ts_col is None:
        ts_col = df.columns[0]
    val_cols = [c for c in df.columns if c != ts_col]
    if not val_cols:
        raise ValueError(f"No value column in {parquet_path}")
    val_col = val_cols[0]

    df = df[[ts_col, val_col]].copy()
    df[ts_col] = to_utc_naive(df[ts_col])
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    df = df.resample(freq).ffill()
    df.columns = ["value"]
    return df


def group_files_by_oid(all_files: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for fp in all_files:
        oid = os.path.splitext(os.path.basename(fp))[0]
        groups.setdefault(oid, []).append(fp)
    return groups


def concat_oid_series(paths: List[str], freq: str) -> pd.DataFrame:
    series_list = []
    for fp in paths:
        try:
            s = load_and_resample_sensor(fp, freq)
            series_list.append(s)
        except Exception:
            continue
    if not series_list:
        return pd.DataFrame()
    s_all = pd.concat(series_list, axis=0).sort_index()
    s_all = s_all[~s_all.index.duplicated(keep="last")]
    return s_all


def build_panel_excluding_oids(all_files: List[str], exclude_oids: List[str], meta: pd.DataFrame,
                                train_grid: pd.DatetimeIndex) -> Tuple[pd.DataFrame, Dict[str,str]]:
    groups = group_files_by_oid(all_files)

    series_list = []
    col_map: Dict[str,str] = {}

    # Duyệt theo object_id (không lặp theo từng file/tháng)
    for oid, paths in groups.items():
        if any(ex in oid for ex in exclude_oids):
            continue
        s = concat_oid_series(paths, CONFIG["RESAMPLE_FREQ"])  # DataFrame col="value"
        if s.empty:
            continue
        # Coverage trong train months (bao nhiêu mốc có mặt trong Jan–May)
        pts_in_train = len(s.index.intersection(train_grid))
        cov_train = pts_in_train / max(1, len(train_grid))
        if cov_train < CONFIG["MIN_TRAIN_COVERAGE"]:
            continue

        col_name = oid
        # Map từ col → object_id (ở đây chính là oid)
        s = s.rename(columns={"value": col_name})
        series_list.append(s)
        col_map[col_name] = oid

        if len(col_map) >= CONFIG["MAX_SENSOR_FEATURES"]:
            break

    if not series_list:
        raise RuntimeError("No usable non-target sensors after TRAIN coverage filter. Try lowering MIN_TRAIN_COVERAGE.")

    df = pd.concat(series_list, axis=1).sort_index().ffill()
    # bỏ cột toàn NA nếu có
    drop_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=drop_cols, errors='ignore')
    for c in drop_cols:
        col_map.pop(c, None)
    return df, col_map


def load_target_series(oid: str) -> pd.Series:
    # Quét toàn bộ tháng và gộp theo oid
    all_files = []
    for mf in CONFIG["MONTH_FOLDERS"]:
        abs_month = _join_input(mf)
        all_files += glob.glob(os.path.join(abs_month, "**", f"{oid}.parquet"), recursive=True)
    if not all_files:
        # fallback: basename contains oid
        for mf in CONFIG["MONTH_FOLDERS"]:
            abs_month = _join_input(mf)
            all_files += [fp for fp in glob.glob(os.path.join(abs_month, "**", "*.parquet"), recursive=True)
                          if oid in os.path.basename(fp)]
    if not all_files:
        raise FileNotFoundError(f"Target files not found for OID={oid}.")

    s = concat_oid_series(sorted(list(set(all_files))), CONFIG["RESAMPLE_FREQ"])  # DataFrame col="value"
    if s.empty:
        raise RuntimeError("Could not load any target series after resampling.")
    y = s.iloc[:, 0].copy()
    y.name = "y"
    return y


# -----------------------------
# Feature engineering
# -----------------------------

def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["minute"] = index.minute
    out["hour"] = index.hour
    out["day"] = index.day
    out["dow"] = index.dayofweek
    out["week"] = index.isocalendar().week.astype(int)
    out["month"] = index.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def add_lag_roll_feats(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in base_cols:
        X[f"{c}_diff1"] = X[c].diff(1)
        for lag in CONFIG["LAG_STEPS"]:
            X[f"{c}_lag{lag}"] = X[c].shift(lag)
        for w in CONFIG["ROLL_WINDOWS"]:
            X[f"{c}_rollmean{w}"] = X[c].rolling(w).mean()
            X[f"{c}_rollstd{w}"]  = X[c].rolling(w).std()
    return X


def build_id_grid() -> pd.DatetimeIndex:
    start = pd.Timestamp('2025-06-01 00:00:00')
    end   = pd.Timestamp('2025-07-31 23:50:00')
    return pd.date_range(start, end, freq=CONFIG["RESAMPLE_FREQ"])  # 8784 stamps


# -----------------------------
# Train / Predict
# -----------------------------

def fit_lgbm(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: Optional[pd.DataFrame], y_va: Optional[pd.Series]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**CONFIG["LGB_PARAMS"])
    if X_va is None or len(X_va) == 0:
        model.fit(X_tr, y_tr)
    else:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=["l1", "l2"],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(50)],
        )
    return model


def make_submission(model, features: List[str], X_all_shifted: pd.DataFrame, train_median: pd.Series,
                    y_clip: Optional[Tuple[float,float]] = None, out_path: str = "submission.csv") -> pd.DataFrame:
    grid = build_id_grid()
    Xg = X_all_shifted.reindex(grid)
    Xg = Xg.ffill().fillna(train_median)
    preds = model.predict(Xg[features])
    if y_clip is not None:
        lo, hi = y_clip
        preds = np.clip(preds, lo, hi)
    sub = pd.DataFrame({
        "ID": [ts.strftime('%Y-%m-%d_%H:%M:%S') for ts in grid],
        "TARGET_VARIABLE": preds,
    })
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} with {len(sub)} rows (expected 8784).")
    return sub


# -----------------------------
# Main
# -----------------------------

def main():
    meta = read_metadata()

    # 0) Lưới thời gian cho TRAIN (Jan–May) để filter coverage
    train_grid = build_months_grid(CONFIG["TRAIN_MONTH_INDEXES"], CONFIG["RESAMPLE_FREQ"])  # UTC-naive

    # 1) Tập file theo toà nhà
    month_files: List[List[str]] = []
    for mf in CONFIG["MONTH_FOLDERS"]:
        files = list_parquet_files_for_building(mf, CONFIG["BUILDING_ID"])
        month_files.append(files)
        print(f"Found {len(files)} parquet files in {mf} for {CONFIG['BUILDING_ID']}")
    all_files = sorted(list(set([f for sub in month_files for f in sub])))

    # 2) Load **đúng** target series (Jan–May có label)
    target_oid = CONFIG["TARGET_OID"]
    y_all = load_target_series(target_oid)  # pd.Series index UTC-naive
    print(f"Loaded target series '{target_oid}' with {len(y_all)} points spanning [{y_all.index.min()} → {y_all.index.max()}]")

    # 3) Build panel từ **non-target** sensors, gộp theo oid và lọc theo coverage trong TRAIN
    panel, col_map = build_panel_excluding_oids(all_files, exclude_oids=[target_oid], meta=meta, train_grid=train_grid)
    print(f"Panel (non-target) shape: {panel.shape} | sensors used: {len(col_map)}")

    # 4) Time features
    tf = time_features(panel.index)
    X_base = pd.concat([panel, tf], axis=1)

    # 5) Kỹ thuật đặc trưng (không dùng target) + enforce lead 3h bằng shift
    base_sensor_cols = list(panel.columns)
    X_eng = add_lag_roll_feats(X_base, base_sensor_cols)
    feature_cols = [c for c in X_eng.columns]
    X_all_shifted = X_eng[feature_cols].shift(CONFIG["LEAD_STEPS"]).copy()

    # 6) Build TRAIN frame trên **giao thời gian** giữa X và y (fix unalign)
    #    Dùng index train từ X, rồi **reindex** y cho khớp để tránh IndexingError
    train_mask = X_all_shifted.index.isin(train_grid)
    train_times = X_all_shifted.index[train_mask]

    X_tr_all = X_all_shifted.loc[train_times]
    y_tr_all = y_all.reindex(train_times)  # align theo timestamp của X

    train_df = pd.concat([y_tr_all.rename("y"), X_tr_all], axis=1).dropna()
    if len(train_df) == 0:
        raise RuntimeError("Training frame empty after dropna. Hãy hạ MIN_TRAIN_COVERAGE hoặc tăng MAX_SENSOR_FEATURES.")

    # 7) Time split 80/20 trong TRAIN
    train_df = train_df.sort_index()
    cut = int(len(train_df) * 0.8)
    tr = train_df.iloc[:cut]
    va = train_df.iloc[cut:]

    X_tr, y_tr = tr.drop(columns=["y"]), tr["y"]
    X_va, y_va = va.drop(columns=["y"]), va["y"]

    print(f"Train: {X_tr.shape} | Valid: {X_va.shape}")

    # 8) Fit model
    model = fit_lgbm(X_tr, y_tr, X_va, y_va)

    # 9) Evaluate
    if len(X_va) > 0:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        va_pred = model.predict(X_va)
        mae = mean_absolute_error(y_va, va_pred)
        rmse = mean_squared_error(y_va, va_pred, squared=False)
        print(f"VALID — MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        y_lo, y_hi = np.percentile(y_tr.values, 1), np.percentile(y_tr.values, 99)
    else:
        print("[INFO] No validation set; trained on all training data.")
        y_lo, y_hi = np.percentile(y_tr.values, 1), np.percentile(y_tr.values, 99)

    # 10) Medians để fill NA khi suy luận
    train_median = X_tr.median(numeric_only=True)

    # 11) Submission (Jun–Jul @10min)
    clip = (y_lo, y_hi) if CONFIG["CLIP_PREDICTIONS"] else None
    sub = make_submission(model, feature_cols, X_all_shifted, train_median, y_clip=clip, out_path="submission.csv")

    # 12) Feature importance (tuỳ chọn)
    try:
        import matplotlib.pyplot as plt
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)[:30]
        plt.figure(figsize=(8, 10))
        imp.iloc[::-1].plot(kind='barh')
        plt.title('Top 30 Feature Importances (LightGBM)')
        plt.tight_layout(); plt.savefig('feature_importance.png', dpi=160)
        print("Saved feature_importance.png")
    except Exception as e:
        print(f"[WARN] Could not save feature importance: {e}")

    return sub


if __name__ == "__main__":
    _ = main()


