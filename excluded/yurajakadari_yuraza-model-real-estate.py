"""
China Real Estate Demand Prediction — Final Refined Baseline
- T+12 forecasting using past-only features (sector-safe asof for test)
- Stronger features: extensive lags/rolls + MoM/YoY rates across series
- Forward-expanding time CV (warm-up) + OOF inverse-RMSE model blending
- Seasonal adjustment, sector-volatility clamps, and strict zero-guard
- Writes submission.csv and sample_submission.csv
"""

import os
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------
# Optional libs (auto-detect)
# ---------------------------
HAS_LGB, HAS_CAT, HAS_XGB = True, True, True
try:
    import lightgbm as lgb
except Exception:
    HAS_LGB = False
try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    HAS_CAT = False
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------
# Paths & globals
# ---------------------------
DATA_DIR = Path("/kaggle/input/china-real-estate-demand-prediction/train")
KAGGLE_TEST = Path("/kaggle/input/china-real-estate-demand-prediction/test.csv")
ALT_TEST = Path("/mnt/data/test.csv")  # fallback (e.g., local)

SEED = 42
np.random.seed(SEED)

# ---------------------------
# Utilities
# ---------------------------
def parse_month_any(s: str) -> pd.Timestamp:
    s = str(s).replace("_", " ")
    for fmt in ("%Y-%b", "%Y %b", "%Y-%m"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")


def add_time_cols(df, month_col="month_dt"):
    df["year"] = df[month_col].dt.year
    df["month_num"] = df[month_col].dt.month
    df["quarter"] = df[month_col].dt.quarter
    df["sin_m"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month_num"] / 12)
    return df


def make_lags_safe(df, group_cols, sort_col, lag_cols, lags=(1, 2, 3, 6, 12)):
    df = df.sort_values(group_cols + [sort_col]).copy()
    for c in lag_cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df.groupby(group_cols, sort=False)[c].shift(L)
        for w in (3, 6, 12):
            df[f"{c}_rollmean{w}"] = (
                df.groupby(group_cols, sort=False)[c]
                .shift(1)
                .transform(lambda s: s.rolling(w, min_periods=max(1, w // 2)).mean())
            )
    return df


def add_rate_features(df, cols, eps=1e-9):
    """Add MoM (%Δ vs lag1) and YoY (%Δ vs lag12) for provided columns."""
    for c in cols:
        l1, l12 = f"{c}_lag1", f"{c}_lag12"
        if l1 in df.columns:
            df[f"{c}_mom"] = np.where(
                np.abs(df[l1]) > eps, (df[c] - df[l1]) / (np.abs(df[l1]) + eps), 0.0
            )
        else:
            df[f"{c}_mom"] = 0.0
        if l12 in df.columns:
            df[f"{c}_yoy"] = np.where(
                np.abs(df[l12]) > eps, (df[c] - df[l12]) / (np.abs(df[l12]) + eps), 0.0
            )
        else:
            df[f"{c}_yoy"] = 0.0
    return df


def sectorwise_fill(df, group_col="sector", exclude=("target", "y_ahead")):
    out = df.copy()
    num_cols = [c for c in out.columns if out[c].dtype != "O" and c not in exclude]
    out = out.sort_values([group_col, "month_dt"])
    out[num_cols] = (
        out.groupby(group_col, group_keys=False)[num_cols]
        .apply(lambda g: g.ffill().bfill())
    )
    out[num_cols] = out[num_cols].fillna(0)
    return out


def asof_by_group(left_df, right_df, by_col, left_on, right_on, direction="backward"):
    """Sector-wise merge_asof with per-group sorting to avoid 'left keys must be sorted'."""
    out = []
    for key, left_g in left_df.groupby(by_col, sort=False):
        right_g = right_df[right_df[by_col] == key]
        if right_g.empty:
            out.append(left_g.copy())
            continue
        lg = left_g.sort_values(left_on).copy()
        rg = right_g.sort_values(right_on).copy()
        merged = pd.merge_asof(
            lg, rg,
            left_on=left_on, right_on=right_on,
            direction=direction, allow_exact_matches=True
        )
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(
        np.sqrt(
            np.mean(
                (np.log1p(np.clip(y_pred, 0, None)) - np.log1p(np.clip(y_true, 0, None)))
                ** 2
            )
        )
    )


def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def custom_competition_score(y_true, y_pred):
    """Two-stage metric emulation (for CV logging; not exact but faithful)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.zeros_like(y_true, dtype=float)
    nz = y_true != 0
    ape[nz] = np.abs(y_pred[nz] - y_true[nz]) / np.abs(y_true[nz])
    # zeros: any nonzero pred → inf penalty
    ape[~nz] = (np.abs(y_pred[~nz]) > 0).astype(float) * np.inf
    if np.mean(ape > 1.0) > 0.30:
        return 0.0
    ok = ape <= 1.0
    if ok.sum() == 0:
        return 0.0
    scaled_mape = ape[ok].mean() / ok.mean()
    return float(1.0 - scaled_mape)


def coerce_numeric(df):
    for c in df.columns:
        if df[c].dtype == "O":
            ser = pd.to_numeric(df[c], errors="ignore")
            if ser.dtype != "O":
                df[c] = ser
    return df


# ---------------------------
# Load data
# ---------------------------
def load_all():
    base = pd.read_csv(DATA_DIR / "new_house_transactions.csv")
    base["month_dt"] = base["month"].map(parse_month_any)

    pre_owned = pd.read_csv(DATA_DIR / "pre_owned_house_transactions.csv")
    pre_owned["month_dt"] = pre_owned["month"].map(parse_month_any)

    land = pd.read_csv(DATA_DIR / "land_transactions.csv")
    land["month_dt"] = land["month"].map(parse_month_any)

    nh_near = pd.read_csv(DATA_DIR / "new_house_transactions_nearby_sectors.csv")
    nh_near["month_dt"] = nh_near["month"].map(parse_month_any)

    po_near = pd.read_csv(DATA_DIR / "pre_owned_house_transactions_nearby_sectors.csv")
    po_near["month_dt"] = po_near["month"].map(parse_month_any)

    land_near = pd.read_csv(DATA_DIR / "land_transactions_nearby_sectors.csv")
    land_near["month_dt"] = land_near["month"].map(parse_month_any)

    poi = pd.read_csv(DATA_DIR / "sector_POI.csv")

    search = pd.read_csv(DATA_DIR / "city_search_index.csv")
    search["month_dt"] = search["month"].map(parse_month_any)

    # Aggregate search (PC/mobile flags + total + optional top-5 keywords)
    agg_search = (
        search.assign(
            src_is_pc=(search["source"].astype(str).str.contains("PC")).astype(int),
            src_is_mobile=(~search["source"].astype(str).str.contains("PC")).astype(int),
        )
        .groupby("month_dt", as_index=False)
        .agg(
            search_total=("search_volume", "sum"),
            search_pc=("src_is_pc", "sum"),
            search_mobile=("src_is_mobile", "sum"),
        )
    )

    # Optional: top keywords pivot
    if {"keyword", "search_volume"}.issubset(search.columns):
        topk = (
            search.groupby("keyword")["search_volume"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        if len(topk) > 0:
            pivot_topk = (
                search.assign(is_top=search["keyword"].isin(topk))
                .loc[lambda d: d["is_top"]]
                .pivot_table(
                    index="month_dt",
                    columns="keyword",
                    values="search_volume",
                    aggfunc="sum",
                )
                .fillna(0.0)
                .add_prefix("kw_")
                .reset_index()
            )
            search_feat = agg_search.merge(pivot_topk, on="month_dt", how="left")
        else:
            search_feat = agg_search.copy()
    else:
        search_feat = agg_search.copy()

    # City macro (annual)
    city = pd.read_csv(DATA_DIR / "city_indexes.csv")
    if "city_indicator_data_year" in city.columns:
        city["year"] = city["city_indicator_data_year"]
    else:
        year_cols = [c for c in city.columns if "year" in c.lower()]
        if not year_cols:
            raise ValueError("Cannot find year column in city_indexes.csv")
        city["year"] = pd.to_numeric(city[year_cols[0]], errors="coerce")
    macro_keep = [c for c in city.columns if c != "city_indicator_data_year"]
    macro_cols = [c for c in macro_keep if city[c].dtype != "O"]
    city_macro = city[["year"] + macro_cols].copy()

    return (
        base,
        pre_owned,
        land,
        nh_near,
        po_near,
        land_near,
        poi,
        search_feat,
        city_macro,
    )


# ---------------------------
# Build features (predict T+12)
# ---------------------------
def build_features():
    (
        base,
        pre_owned,
        land,
        nh_near,
        po_near,
        land_near,
        poi,
        search_feat,
        city_macro,
    ) = load_all()

    # Harmonize sector col to string ("sector n")
    for df in (base, pre_owned, land, nh_near, po_near, land_near):
        if "sector" in df.columns:
            df["sector"] = df["sector"].astype(str)

    # Merge wide panel at time t
    panel = base.merge(pre_owned.drop(columns=["month"]), on=["month_dt", "sector"], how="left")
    panel = panel.merge(land.drop(columns=["month"]), on=["month_dt", "sector"], how="left")
    panel = panel.merge(nh_near.drop(columns=["month"]), on=["month_dt", "sector"], how="left")
    panel = panel.merge(po_near.drop(columns=["month"]), on=["month_dt", "sector"], how="left")
    panel = panel.merge(land_near.drop(columns=["month"]), on=["month_dt", "sector"], how="left")
    panel = panel.merge(search_feat, on="month_dt", how="left")
    panel = panel.merge(poi, on="sector", how="left")
    panel = add_time_cols(panel, "month_dt")
    panel = coerce_numeric(panel)

    # Macro join — avoid duplicate 'year'
    city_macro_ = city_macro.loc[:, ~city_macro.columns.duplicated()].rename(columns={"year": "macro_year"})
    panel = panel.rename(columns={"year": "panel_year"})
    panel = (
        panel.merge(city_macro_, left_on="panel_year", right_on="macro_year", how="left")
             .drop(columns=["macro_year"])
             .rename(columns={"panel_year": "year"})
    )

    # Target and candidate series for lags/rates
    target_col = "amount_new_house_transactions"
    panel["target"] = panel[target_col].astype(float)

    base_cols = [c for c in [
        "target",
        "num_new_house_transactions",
        "area_new_house_transactions",
        "price_new_house_transactions",
        "num_new_house_available_for_sale",
        "area_new_house_available_for_sale",
        "period_new_house_sell_through",
        "num_pre_owned_house_transactions",
        "area_pre_owned_house_transactions",
        "price_pre_owned_house_transactions",
        "num_land_transactions",
        "construction_area",
        "planned_building_area",
        "transaction_amount",
    ] if c in panel.columns]

    # Lags & rolls computed at time t
    panel = make_lags_safe(panel, ["sector"], "month_dt", base_cols, lags=(1, 2, 3, 6, 12))
    panel = add_rate_features(panel, base_cols)

    # Label setup: predict y at t+12 from features at t
    panel = panel.sort_values(["sector", "month_dt"]).reset_index(drop=True)
    panel["y_ahead"] = panel.groupby("sector")["target"].shift(-12)
    panel["label_month_dt"] = panel["month_dt"] + pd.offsets.DateOffset(months=12)

    # Require at least 12 months history and non-null label
    panel = panel.groupby("sector").apply(lambda g: g.iloc[12:]).reset_index(drop=True)
    panel = panel.dropna(subset=["y_ahead"]).reset_index(drop=True)

    # Seasonal medians for later tweak
    seasonal = (
        panel.groupby(["sector", "month_num"])["target"]
        .median()
        .rename("sector_month_med")
        .reset_index()
    )
    sector_med = (
        panel.groupby("sector")["target"]
        .median()
        .rename("sector_med_train")
        .reset_index()
    )
    seas = seasonal.merge(sector_med, on="sector", how="left")
    seas["season_factor"] = np.clip(seas["sector_month_med"] / (seas["sector_med_train"] + 1e-9), 0.5, 1.5)

    # Sector volatility stats for clamps
    vol = panel.groupby("sector")["target"].agg(["mean", "std", "median"]).reset_index()
    vol = vol.rename(columns={"mean": "sec_mean", "std": "sec_std", "median": "sec_median"})
    vol["sec_std"] = vol["sec_std"].fillna(0.0)

    # Fill numeric gaps
    panel = sectorwise_fill(panel, "sector", exclude=("target", "y_ahead"))

    return panel, seas, vol


def build_time_folds(dates: pd.Series, n_splits: int = 4, warm_up_months: int = 24) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Forward-expanding folds over label_month_dt. Each val block is contiguous in time."""
    uniq = np.sort(dates.unique())
    if len(uniq) <= warm_up_months + n_splits:
        # Degenerate: tiny timeline → simple KFold-like chronological splits
        cutpoints = np.array_split(uniq[warm_up_months:], n_splits)
    else:
        cutpoints = np.array_split(uniq[warm_up_months:], n_splits)

    folds = []
    for i, val_months in enumerate(cutpoints):
        if len(val_months) == 0:
            continue
        val_mask = dates.isin(val_months)
        # Train up to the first val month (exclusive)
        train_mask = dates < val_months.min()
        tr_idx = np.where(train_mask)[0]
        va_idx = np.where(val_mask)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        folds.append((tr_idx, va_idx))
    return folds


# ---------------------------
# Modeling
# ---------------------------
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "id", "month", "month_dt", "label_month_dt", "y_ahead", "target",
        "sector_month_med", "sector_med_train", "season_factor"  # safety
    }
    feats = [c for c in df.columns if (df[c].dtype != "O") and (c not in exclude)]
    return feats


def fit_and_oof_blend(panel: pd.DataFrame, feats: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, object]]:
    X = panel[feats].values
    y = panel["y_ahead"].values
    dates = panel["label_month_dt"]
    folds = build_time_folds(dates, n_splits=4, warm_up_months=24)

    oof = {}
    models = {}
    oof_preds = {k: np.zeros(len(panel)) for k in ["lgb", "cat", "xgb", "hgb"]}

    # Prepare algorithms
    algos = []
    if HAS_LGB:
        algos.append(("lgb", lgb.LGBMRegressor(
            n_estimators=1600, learning_rate=0.035, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2,
            random_state=SEED
        )))
    if HAS_CAT:
        algos.append(("cat", CatBoostRegressor(
            depth=8, learning_rate=0.03, iterations=1600, l2_leaf_reg=4.0,
            loss_function="RMSE", random_seed=SEED, verbose=False
        )))
    if HAS_XGB:
        algos.append(("xgb", XGBRegressor(
            n_estimators=1600, learning_rate=0.035, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2,
            tree_method="hist", random_state=SEED
        )))
    # Always include a robust fallback
    algos.append(("hgb", HistGradientBoostingRegressor(
        max_depth=None, max_iter=800, learning_rate=0.05,
        l2_regularization=0.0, random_state=SEED
    )))

    cat_idx = []  # all features numeric already
    # CV
    for name, est in algos:
        fold_models = []
        for (tr_idx, va_idx) in folds:
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xva, yva = X[va_idx], y[va_idx]

            if name == "cat":
                est_fold = CatBoostRegressor(**est.get_params())
                est_fold.fit(Xtr, ytr, verbose=False)
            else:
                est_fold = est.__class__(**est.get_params())
                est_fold.fit(Xtr, ytr)

            p = est_fold.predict(Xva)
            oof_preds[name][va_idx] = p
            fold_models.append(est_fold)

        models[name] = fold_models

    # OOF scores and inverse-RMSE weights
    weights = {}
    for name in oof_preds:
        if oof_preds[name].sum() == 0 and name != "hgb":
            continue
        rmse = rmse_np(y, oof_preds[name])
        weights[name] = 1.0 / (rmse + 1e-9)

    # normalize and keep only algos that participated
    total_w = sum(weights.values())
    if total_w == 0:
        weights = {"hgb": 1.0}
    else:
        for k in list(weights.keys()):
            weights[k] /= total_w

    # CV diagnostics
    blend_oof = np.zeros_like(y, dtype=float)
    for k, w in weights.items():
        blend_oof += w * oof_preds[k]
    print(f"[CV] RMSE blend: {rmse_np(y, blend_oof):.4f}  |  RMSLE: {rmsle(y, blend_oof):.4f}  |  custom: {custom_competition_score(y, blend_oof):.5f}")
    print("[Blend weights]", weights)

    return oof_preds, weights, models


def refit_full_models(panel: pd.DataFrame, feats: List[str], weights: Dict[str, float]) -> Dict[str, object]:
    X = panel[feats].values
    y = panel["y_ahead"].values

    fitted = {}
    if "lgb" in weights and HAS_LGB:
        est = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.2,
            random_state=SEED
        )
        est.fit(X, y)
        fitted["lgb"] = est
    if "cat" in weights and HAS_CAT:
        est = CatBoostRegressor(
            depth=8, learning_rate=0.03, iterations=2000, l2_leaf_reg=4.0,
            loss_function="RMSE", random_seed=SEED, verbose=False
        )
        est.fit(X, y, verbose=False)
        fitted["cat"] = est
    if "xgb" in weights and HAS_XGB:
        est = XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=8,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.2,
            tree_method="hist", random_state=SEED
        )
        est.fit(X, y)
        fitted["xgb"] = est

    if len(fitted) == 0 or "hgb" in weights:
        est = HistGradientBoostingRegressor(
            max_depth=None, max_iter=1200, learning_rate=0.05,
            l2_regularization=0.0, random_state=SEED
        )
        est.fit(X, y)
        fitted["hgb"] = est

    return fitted


# ---------------------------
# Test mapping & postprocess
# ---------------------------
def load_test() -> pd.DataFrame:
    if KAGGLE_TEST.exists():
        t = pd.read_csv(KAGGLE_TEST)
    elif ALT_TEST.exists():
        t = pd.read_csv(ALT_TEST)
    else:
        raise FileNotFoundError("test.csv not found at Kaggle path or ALT_TEST.")
    # Parse to month/sector while preserving row order
    if "id" in t.columns and "month" not in t.columns:
        # id format: "YYYY Mon_sector n"
        parts = t["id"].astype(str).str.split("_", n=1, expand=True)
        t["month"] = parts[0]
        t["sector"] = parts[1]
    if "sector" not in t.columns:
        raise ValueError("test.csv must contain 'sector' (or be derivable from id).")
    t["sector"] = t["sector"].astype(str)
    t["test_month_dt"] = t["month"].map(parse_month_any)
    return t


def build_test_features(panel: pd.DataFrame, feats: List[str], test_df: pd.DataFrame) -> pd.DataFrame:
    # Right table: one row per (sector, label_month_dt), carrying features at t
    feat_tab = panel[["sector", "label_month_dt"] + feats].copy()
    feat_tab = feat_tab.sort_values(["sector", "label_month_dt"]).reset_index(drop=True)

    # Left table: test rows keyed by (sector, test_month_dt)
    left = test_df[["sector", "test_month_dt"]].copy()

    # sector-wise asof on time (<=) using label_month_dt to align t → t+12
    merged = asof_by_group(
        left_df=left.assign(dummy=0),
        right_df=feat_tab.assign(dummy=0),
        by_col="sector",
        left_on="test_month_dt",
        right_on="label_month_dt",
        direction="backward"
    )
    # Keep in original order
    merged = merged.sort_index()

    # Attach back the test id/order
    out = test_df.copy()
    out = pd.concat([out.reset_index(drop=True), merged[feats].reset_index(drop=True)], axis=1)
    return out


def seasonal_tweak(preds: pd.Series, test_df: pd.DataFrame, seas: pd.DataFrame) -> np.ndarray:
    # Map month_num/sector → season_factor
    map_df = test_df[["sector", "test_month_dt"]].copy()
    map_df["month_num"] = map_df["test_month_dt"].dt.month
    m = map_df.merge(seas[["sector", "month_num", "season_factor"]], on=["sector", "month_num"], how="left")
    sf = m["season_factor"].fillna(1.0).clip(0.8, 1.2).values
    # Blend toward seasonal expectation mildly (30%)
    return preds * (0.7 + 0.3 * sf)


def zero_guard(preds: np.ndarray, test_feats: pd.DataFrame) -> np.ndarray:
    """Force zero if key drivers are structurally zero at latest known state."""
    z = preds.copy()
    drivers = [c for c in test_feats.columns if any(
        key in c for key in [
            "num_new_house_transactions_lag1",
            "area_new_house_transactions_lag1",
            "price_new_house_transactions_lag1",
            "num_pre_owned_house_transactions_lag1",
            "area_pre_owned_house_transactions_lag1",
            "price_pre_owned_house_transactions_lag1"
        ]
    )]
    if len(drivers) == 0:
        return np.clip(z, 0, None)
    mask_zero = (test_feats[drivers].fillna(0).abs().sum(axis=1) == 0).values
    z[mask_zero] = 0.0
    return np.clip(z, 0, None)


def sector_clamp(preds: np.ndarray, test_df: pd.DataFrame, vol: pd.DataFrame) -> np.ndarray:
    z = preds.copy()
    m = test_df[["sector"]].merge(vol, on="sector", how="left")
    upper = (m["sec_mean"].fillna(0) + 3.0 * m["sec_std"].fillna(0)).values
    lower = np.maximum(0.0, (m["sec_mean"].fillna(0) - 2.0 * m["sec_std"].fillna(0)).values)
    # If stats missing, fall back to non-negative
    upper = np.where(np.isnan(upper) | (upper <= 0), np.inf, upper)
    lower = np.where(np.isnan(lower) | (lower < 0), 0.0, lower)
    z = np.minimum(np.maximum(z, lower), upper)
    return z


# ---------------------------
# Main
# ---------------------------
def main():
    print("Loading & building features ...")
    panel, seas, vol = build_features()
    # Label encode sector (numeric feature)
    le = LabelEncoder()
    panel["sector_le"] = le.fit_transform(panel["sector"].astype(str))

    feats = get_feature_columns(panel)
    print(f"Feature count: {len(feats)}")

    print("Cross-validating & computing blend weights ...")
    _, weights, _ = fit_and_oof_blend(panel, feats)

    print("Refitting on full training data ...")
    fitted = refit_full_models(panel, feats, weights)

    # Prepare test
    print("Loading test & aligning T→T+12 features ...")
    test_df = load_test()
    # Add sector_le to test via mapping
    sec_map = pd.DataFrame({"sector": le.classes_, "sector_le": np.arange(len(le.classes_))})
    # Build test features via sector-wise asof on label_month_dt
    test_feats = build_test_features(panel, feats, test_df)
    # If sector_le was among feats, patch it from mapping in case merge missed
    if "sector_le" in feats and "sector_le" not in test_feats.columns:
        test_feats = test_feats.merge(sec_map, on="sector", how="left")

    # Predict by blended models
    preds = np.zeros(len(test_feats), dtype=float)
    Xtest = test_feats[feats].values
    for name, w in weights.items():
        if name not in fitted:
            continue
        est = fitted[name]
        p = est.predict(Xtest)
        preds += w * p

    # Post-processing: seasonal tweak → zero guard → sector clamp
    preds = seasonal_tweak(pd.Series(preds), test_feats, seas)
    preds = zero_guard(preds, test_feats)
    preds = sector_clamp(preds, test_feats, vol)

    # Build submission preserving row order
    if "id" in test_df.columns:
        sub = pd.DataFrame({
            "id": test_df["id"].astype(str),
            "new_house_transaction_amount": preds.astype(float)
        })
    else:
        # Construct id as "YYYY Mon_sector n"
        id_series = test_df["test_month_dt"].dt.strftime("%Y %b") + "_" + test_df["sector"].astype(str)
        sub = pd.DataFrame({
            "id": id_series,
            "new_house_transaction_amount": preds.astype(float)
        })

    # Non-negative + float
    sub["new_house_transaction_amount"] = np.clip(sub["new_house_transaction_amount"], 0, None)

    # Write files
    out_path = Path(".")
    sub_path = out_path / "submission.csv"
    sample_path = out_path / "sample_submission.csv"
    sub.to_csv(sub_path, index=False)
    sub.to_csv(sample_path, index=False)

    print(f"Wrote {sub_path.resolve()}  (rows={len(sub)})")
    print(f"Wrote {sample_path.resolve()}  (rows={len(sub)})")


if __name__ == "__main__":
    main()


