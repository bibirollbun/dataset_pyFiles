from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
DATA_GLOB = "input_*.csv"              # point this at your FULL per-frame tracking (all players)
OUT_DIR = Path("out")
W = 5                                  # trailing window size (frames)
EPS = 1e-6                             # numeric stability for divisions
MIN_VALID_FRAMES = 3                   # require at least this many valid frames in window

# Reasonable physical caps (yards / yards-per-sec)
CLIP_BOUNDS = {
    "x": (-5.0, 125.0),
    "y": (-10.0, 65.0),
    "dist": (0.0, 120.0),
    "speed": (0.0, 12.0),              # ~ max sprint ~ 10-11 yds/s; allow a bit higher
    "count": (0.0, 11.0),
    "angle": (-np.pi, np.pi),
}

# ---------- UTIL ----------
def _finite(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df

def _clip_cols(df: pd.DataFrame, cols_bounds: dict[str, tuple[float, float]]) -> None:
    for col, (lo, hi) in cols_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

def _load_inputs() -> pd.DataFrame:
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No {DATA_GLOB} found in {Path('.').resolve()}")

    dfs = [pd.read_csv(f) for f in files]
    raw = pd.concat(dfs, ignore_index=True)

    required = {
        "game_id","play_id","frame_id","nfl_id","x","y","player_position","play_direction"
    }
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns in inputs: {sorted(missing)}")

    # ---- Handle speed or acceleration ----
    if "speed" not in raw.columns:
        if "a" in raw.columns:
            print("âš™ï¸�  Using 'a' column as proxy for 'speed'")
            raw = raw.rename(columns={"a": "speed"})
        else:
            print("âš™ï¸�  Computing speed from x/y frame deltas (no 'speed' or 'a' found)")
            raw = raw.sort_values(["game_id","play_id","nfl_id","frame_id"])
            raw["dx"] = raw.groupby(["game_id","play_id","nfl_id"])["x"].diff()
            raw["dy"] = raw.groupby(["game_id","play_id","nfl_id"])["y"].diff()
            # assume 10 frames per second (Big Data Bowl standard)
            raw["speed"] = np.hypot(raw["dx"], raw["dy"]) * 10.0
            raw.drop(columns=["dx","dy"], inplace=True)

    # ---------- Type cleaning ----------
    raw = raw.copy()
    for c in ["game_id","play_id","frame_id","nfl_id"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").astype("Int64")
    for c in ["x","y","speed"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if "play_direction" in raw.columns:
        raw["play_direction"] = raw["play_direction"].astype(str).str.lower()

    # ---------- Clip to sane physical bounds ----------
    _clip_cols(raw, {"x": CLIP_BOUNDS["x"], "y": CLIP_BOUNDS["y"], "speed": CLIP_BOUNDS["speed"]})
    raw = _finite(raw)
    return raw

def _load_singleframe_labels(path: str | Path = "out") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the single-frame train/candidate CSVs written earlier and returns:
      - train_sf: labels for targeted WRs at the decision frame
      - cand_sf:  candidate keys (includes decision_frame if available)
    Normalizes: id â†’ wr_id, label â†’ complete
    """
    p = Path(path)
    tr_p = p / "catch_model_train.csv"
    ca_p = p / "catch_candidates.csv"

    if not tr_p.exists():
        raise FileNotFoundError(f"Missing {tr_p}")
    if not ca_p.exists():
        raise FileNotFoundError(f"Missing {ca_p}")

    train = pd.read_csv(tr_p)
    cands = pd.read_csv(ca_p)

    # normalize id column to wr_id
    if "wr_id" not in train.columns:
        if "nfl_id" in train.columns:
            train = train.rename(columns={"nfl_id": "wr_id"})
        else:
            raise KeyError("train missing 'wr_id' or 'nfl_id'")

    if "wr_id" not in cands.columns and "nfl_id" in cands.columns:
        cands = cands.rename(columns={"nfl_id": "wr_id"})

    # keys
    need = ["game_id","play_id","wr_id"]
    for col in need:
        if col not in train.columns:
            raise KeyError(f"train missing required column: {col}")

    # find completion/label column
    label_col = None
    for cand in ["complete", "is_complete", "caught"]:
        if cand in train.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError("train missing completion label column (complete/is_complete/caught)")

    train_sf = train[need + [label_col]].rename(columns={label_col: "complete"}).copy()
    train_sf = train_sf.drop_duplicates(subset=need, keep="last")

    keep_c = [c for c in cands.columns if c in (need + ["decision_frame"])]
    if not keep_c:
        keep_c = need
    cand_sf = cands[keep_c].drop_duplicates(subset=need, keep="last").copy()

    # type normalization
    for c in need:
        train_sf[c] = pd.to_numeric(train_sf[c], errors="coerce").astype("Int64")
        cand_sf[c]  = pd.to_numeric(cand_sf[c],  errors="coerce").astype("Int64")
    train_sf["complete"] = pd.to_numeric(train_sf["complete"], errors="coerce").fillna(0).astype(int)

    # decision_frame may be missing for some plays; that's okay
    if "decision_frame" in cand_sf.columns:
        cand_sf["decision_frame"] = pd.to_numeric(cand_sf["decision_frame"], errors="coerce").astype("Int64")

    return train_sf, cand_sf

def _normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize field so that offense always goes left->right (increasing x)."""
    d = df.copy()
    # detect direction at play level
    play_dir = d.groupby(["game_id","play_id"], as_index=False)["play_direction"]\
                .agg(lambda s: s.iloc[0] if len(s) else "right")
    play_dir_map = {(r.game_id, r.play_id): ("left" in str(r.play_direction)) for _, r in play_dir.iterrows()}
    flip_mask = d[["game_id","play_id"]]\
        .apply(lambda r: play_dir_map.get((r.game_id, r.play_id), False), axis=1).values

    # flip if going right->left so that increasing x is always offense direction
    d.loc[flip_mask, "x"] = 120.0 - d.loc[flip_mask, "x"]
    # y stays the same in the usual normalization
    d["play_dir_flipped"] = flip_mask.astype(int)
    return d

def _tag_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Map positions to qb/wr/db role buckets."""
    pos = df["player_position"].astype(str).str.upper()
    is_qb = pos.eq("QB")
    # WR/TE/RB eligible receivers; keep it broad for non-target candidates
    is_wr = pos.isin(["WR","TE","RB","FB","HB"])
    # Defensive backfield
    is_db = pos.isin(["CB","DB","FS","SS","S","NB"])  # nickel as NB
    d = df.copy()
    d["is_qb"] = is_qb.astype(int)
    d["is_wr"] = is_wr.astype(int)
    d["is_db"] = is_db.astype(int)
    return d

def _pivot_roles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    qbs = d[d["is_qb"]==1][["game_id","play_id","frame_id","nfl_id","x","y","speed"]].rename(
        columns={"nfl_id":"qb_id","x":"qb_x","y":"qb_y","speed":"qb_speed"}
    )
    wrs = d[d["is_wr"]==1][["game_id","play_id","frame_id","nfl_id","x","y","speed"]].rename(
        columns={"nfl_id":"wr_id","x":"wr_x","y":"wr_y","speed":"wr_speed"}
    )
    dbs = d[d["is_db"]==1][["game_id","play_id","frame_id","nfl_id","x","y","speed"]].rename(
        columns={"nfl_id":"db_id","x":"db_x","y":"db_y","speed":"db_speed"}
    )
    # attach QB to WR rows for geometry
    core = wrs.merge(qbs, on=["game_id","play_id","frame_id"], how="left")
    return core, dbs, qbs

def _nearest_db(core: pd.DataFrame, dbs: pd.DataFrame) -> pd.DataFrame:
    """Select nearest DB to each WR per frame; if none, drop that frame for that WR."""
    j = core.merge(dbs, on=["game_id","play_id","frame_id"], how="left", suffixes=("","_db"))
    # distance WR-DB per row
    j["d_wr_db"] = np.hypot(j["wr_x"] - j["db_x"], j["wr_y"] - j["db_y"])
    # Keep only rows that actually have a DB
    j = j[~j["db_id"].isna()].copy()

    # Pick min d_wr_db per (game, play, frame, wr_id)
    idx = j.groupby(["game_id","play_id","frame_id","wr_id"])["d_wr_db"].idxmin()
    pick = j.loc[idx, ["game_id","play_id","frame_id","wr_id","db_id","d_wr_db","db_speed"]]

    # Join back to core
    out = core.merge(pick, on=["game_id","play_id","frame_id","wr_id"], how="left")
    return out

def _compute_frame_geometry(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # QBâ†’WR vector & distance
    d["qb_wr_dx"] = d["wr_x"] - d["qb_x"]
    d["qb_wr_dy"] = d["wr_y"] - d["qb_y"]
    d["qb_wr_dist"] = np.hypot(d["qb_wr_dx"], d["qb_wr_dy"]).clip(*CLIP_BOUNDS["dist"])

    # Angle (bearing) QBâ†’WR
    d["theta"] = np.arctan2(d["qb_wr_dy"], d["qb_wr_dx"])
    d["theta_cos"] = np.cos(d["theta"])
    d["theta_sin"] = np.sin(d["theta"])

    # Simple pressure/lane stand-ins (set to 0 if absent)
    if "pressure_cnt" not in d.columns:
        d["pressure_cnt"] = 0.0
    if "lane_clear_min_dist" not in d.columns:
        d["lane_clear_min_dist"] = 0.0
    d["pressure_cnt_cap3"] = np.clip(d["pressure_cnt"], *CLIP_BOUNDS["count"])

    # Separation ratios (guard denom)
    d["sep_over_throw"]   = d["d_wr_db"] / (d["qb_wr_dist"] + EPS)
    d["pressure_per_sep"] = d["pressure_cnt"] / (d["d_wr_db"] + EPS)
    d["wr_speed_sep"]     = d["wr_speed"] * d["d_wr_db"]
    d["inv_throw"]        = 1.0 / (d["qb_wr_dist"] + EPS)

    # Interactions & squares
    d["throw_dist_sq"]    = d["qb_wr_dist"]**2
    d["sep_sq"]           = d["d_wr_db"]**2
    d["press_x_throw"]    = d["pressure_cnt"] * d["qb_wr_dist"]
    d["press_x_lane"]     = d["pressure_cnt"] * d["lane_clear_min_dist"]

    # Yardline bins if present
    if "absolute_yardline_number" in d.columns:
        d["yardline_20s"] = (d["absolute_yardline_number"] // 20).clip(0, 5)
        d["red_zone"] = (d["absolute_yardline_number"] <= 20).astype(int)
    else:
        d["yardline_20s"] = 3
        d["red_zone"] = 0

    # Alias common name expected downstream
    d["nearest_db_dist"] = d["d_wr_db"]

    # Clip
    _clip_cols(d, {
        "qb_wr_dist": CLIP_BOUNDS["dist"],
        "d_wr_db": CLIP_BOUNDS["dist"],
        "nearest_db_dist": CLIP_BOUNDS["dist"],
        "wr_speed": CLIP_BOUNDS["speed"],
        "db_speed": CLIP_BOUNDS["speed"],
        "qb_speed": CLIP_BOUNDS["speed"],
        "lane_clear_min_dist": CLIP_BOUNDS["dist"],
        "pressure_cnt": CLIP_BOUNDS["count"],
        "pressure_cnt_cap3": CLIP_BOUNDS["count"],
        "theta": CLIP_BOUNDS["angle"],
    })
    return _finite(d)

def _window_agg(
    frames: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    label_col: str | None = "complete",
    w: int = W,
) -> pd.DataFrame:
    """
    Aggregate a trailing window of w frames *ending at the group's last frame*
    for each (game_id, play_id, wr_id). Returns ONE row per (game_id, play_id, wr_id).

    - Computes mean over the last w frames for numeric columns â†’ *_mean_W{w}
    - Computes (last - first) deltas for a guarded subset â†’ *_delta_W{w}
    - Records frames_used_W{w}
    - Optionally joins labels (normalized to keys + `label_col`)
    """
    # --- normalize WR id column in frames ---
    key = ["game_id", "play_id", "wr_id"]
    if "wr_id" not in frames.columns:
        if "nfl_id" in frames.columns:
            frames = frames.rename(columns={"nfl_id": "wr_id"})
        else:
            raise KeyError("Frames must contain 'wr_id' (or 'nfl_id' to be renamed).")

    if "frame_id" not in frames.columns:
        raise KeyError("Frames must contain 'frame_id' for temporal aggregation.")

    # Sort to ensure 'last' really is the last frame
    frames = frames.sort_values(key + ["frame_id"]).copy()

    # Numeric columns eligible for window means
    exclude = set(key + ["frame_id", "complete"])
    num_cols = [c for c in frames.select_dtypes(include=[np.number]).columns if c not in exclude]

    # Columns eligible for last-first deltas (only if they exist & numeric)
    delta_candidates = [
        "qb_wr_dist", "wr_speed", "db_speed", "nearest_db_dist",
        "lane_clear_min_dist", "pressure_cnt", "qb_x", "qb_y", "wr_x", "wr_y",
        "qb_wr_dx", "qb_wr_dy", "theta",
    ]
    delta_cols = [c for c in delta_candidates if c in num_cols]

    # Grouped aggregation helper
    records = []
    for (gid, pid, wid), g in frames.groupby(key, sort=False):
        g = g.sort_values("frame_id")
        tail = g.tail(max(1, w))  # if <w frames exist, use what's available
        frames_used = len(tail)

        rec = {"game_id": gid, "play_id": pid, "wr_id": wid, f"frames_used_W{w}": frames_used}

        # Means over the trailing window
        means = tail[num_cols].mean(numeric_only=True)
        for c, v in means.items():
            rec[f"{c}_mean_W{w}"] = v

        # Deltas: last - first (guard NaNs/Infs)
        first = tail.iloc[0]
        last  = tail.iloc[-1]
        for c in delta_cols:
            v_first = first[c]
            v_last  = last[c]
            if pd.isna(v_first) or pd.isna(v_last) or np.isinf(v_first) or np.isinf(v_last):
                rec[f"{c}_delta_W{w}"] = np.nan
            else:
                rec[f"{c}_delta_W{w}"] = float(v_last - v_first)

        records.append(rec)

    agg = pd.DataFrame.from_records(records)

    # Enforce minimum frames used
    agg = agg[agg[f"frames_used_W{w}"] >= MIN_VALID_FRAMES].copy()

    # Clean up any infs that may have slipped through
    agg.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Join labels if provided ---
    if labels is not None and label_col is not None:
        # Normalize WR id column in labels
        if "wr_id" not in labels.columns:
            wr_id_candidates = ["wr_id", "nfl_id", "target_nfl_id", "receiver_id"]
            wr_col = next((c for c in wr_id_candidates if c in labels.columns), None)
            if wr_col is None:
                raise KeyError(f"labels missing WR id; looked for {wr_id_candidates}, got: {list(labels.columns)}")
            labels = labels.rename(columns={wr_col: "wr_id"})

        # Normalize completion column name to label_col
        if label_col not in labels.columns:
            y_candidates = ["complete", "is_complete", "caught"]
            y_col = next((c for c in y_candidates if c in labels.columns), None)
            if y_col is None:
                raise KeyError(f"labels missing completion column; looked for {y_candidates}, got: {list(labels.columns)}")
            labels = labels.rename(columns={y_col: label_col})

        need = set(["game_id","play_id","wr_id"])
        miss = need - set(labels.columns)
        if miss:
            raise KeyError(f"labels missing keys for join: {sorted(miss)}")

        labels_small = labels[list(need) + [label_col]].drop_duplicates(subset=list(need))
        agg = agg.merge(labels_small, on=["game_id","play_id","wr_id"], how="left")

    return agg

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load & prep frames
    print("Loading inputsâ€¦")
    raw = _load_inputs()
    raw = _normalize_direction(raw)
    raw = _tag_roles(raw)

    # 2) Load decision/labels (from single-frame build)
    print("Loading single-frame decision/labelsâ€¦")
    train_sf, cand_sf = _load_singleframe_labels()

    # 3) Build per-frame geometry with nearest DB
    print("Computing per-frame geometryâ€¦")
    core, dbs, _ = _pivot_roles(raw)
    core = _nearest_db(core, dbs)
    core = _compute_frame_geometry(core)

    # --- NEW: restrict to decision frame and keep ALL receivers up to that moment ---
    if "decision_frame" in cand_sf.columns:
        print("Restricting frames to decision_frame (<=) where available, keeping ALL eligible receiversâ€¦")
        df_map = cand_sf[["game_id","play_id","decision_frame"]].dropna().drop_duplicates()
        df_map["decision_frame"] = pd.to_numeric(df_map["decision_frame"], errors="coerce").astype("Int64")

        # Join to each WR frame and drop frames after the decision
        core = core.merge(df_map, on=["game_id","play_id"], how="left")
        before = len(core)
        core = core[(core["decision_frame"].isna()) | (core["frame_id"] <= core["decision_frame"])].copy()
        after = len(core)
        print(f"  Frame rows: {before:,} -> {after:,} after decision_frame clipping")
        core.drop(columns=["decision_frame"], inplace=True, errors="ignore")
    else:
        print("decision_frame not found in candidates; proceeding without time clipping (may include post-throw frames).")

    # 4) Aggregate last W frames ending at last available frame (train/cands)
    print(f"Aggregating trailing window W={W} with rich statsâ€¦")
    train_w = _window_agg(core, labels=train_sf, label_col="complete", w=W)   # labeled: targeted WRs only
    cand_w  = _window_agg(core, labels=None,      label_col=None,     w=W)   # unlabeled: ALL eligible WRs

    # 5) Final numeric cleanup
    for df in (train_w, cand_w):
        num = df.select_dtypes(include=[np.number]).columns
        df[num] = df[num].replace([np.inf,-np.inf], np.nan)

    # 6) Write outputs
    tr_out = OUT_DIR / f"catch_model_train_W{W}_rich.csv"
    ca_out = OUT_DIR / f"catch_candidates_W{W}_rich.csv"
    train_w.to_csv(tr_out, index=False)
    cand_w.to_csv(ca_out, index=False)

    print("Wrote:")
    print(f"  {tr_out}")
    print(f"  {ca_out}")
    print(f"Train rows: {len(train_w):,} | Cand rows: {len(cand_w):,}")

    # 7) Quick audit: unique WRs per play in CANDIDATES (should be >=2 for most)
    print("\nVerifying WR options per play in candidatesâ€¦")
    wr_counts = (
        cand_w.groupby(["game_id","play_id"])["wr_id"]
              .nunique(dropna=True)
              .rename("unique_wr_per_play")
              .reset_index()
    )
    dist = wr_counts["unique_wr_per_play"].value_counts().sort_index()
    for k, v in dist.items():
        pct = 100.0 * v / len(wr_counts)
        print(f"  {k}: {v} plays ({pct:.1f}%)")

    singletons = wr_counts[wr_counts["unique_wr_per_play"] == 1]
    wr_counts.to_csv(OUT_DIR / "verify_wr_counts_per_play.csv", index=False)
    singletons.to_csv(OUT_DIR / "verify_single_wr_plays.csv", index=False)
    print("\nSaved audits:")
    print(f"  - {OUT_DIR / 'verify_wr_counts_per_play.csv'}")
    print(f"  - {OUT_DIR / 'verify_single_wr_plays.csv'}")

    if not singletons.empty:
        print("\nSample singletons:")
        print(singletons.head(10))

if __name__ == "__main__":
    main()



# 12_8_ML_W5_rich_train.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib

OUT = Path("out")
TRAIN_CSV = OUT / "catch_model_train_W5_rich.csv"
CANDS_CSV = OUT / "catch_candidates_W5_rich.csv"
SCORED_CSV = OUT / "catch_candidates_scored_W5_rich.csv"
METRICS_JSON = OUT / "metrics_W5_rich.json"
MODEL_FILE = OUT / "catch_model_W5_rich_xgb.joblib"

RANDOM_SEED = 42
MAX_NAN_COL_RATE = 0.30  # drop any feature with >30% NaN in TRAIN
KEYS = ["game_id", "play_id", "wr_id"]

def load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")
    return pd.read_csv(p)

def pick_features(df: pd.DataFrame) -> list[str]:
    """Use only numeric *_mean_W5 / *_delta_W5 features to avoid raw-column drift."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num
             if (c.endswith("_mean_W5") or c.endswith("_delta_W5"))
             and c not in KEYS
             and c != "complete"]
    # Some datasets include duplicates of position coordsâ€”dedupe just in case
    feats = sorted(list(dict.fromkeys(feats)))
    return feats

def replace_infs_nan(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    num = d.select_dtypes(include=[np.number]).columns
    d[num] = d[num].replace([np.inf, -np.inf], np.nan)
    return d

def drop_bad_columns(train: pd.DataFrame, feats: list[str]) -> list[str]:
    keep = []
    for c in feats:
        na_rate = train[c].isna().mean() if c in train.columns else 1.0
        if (c in train.columns) and (na_rate <= MAX_NAN_COL_RATE) and (train[c].nunique(dropna=True) > 1):
            keep.append(c)
    return keep

def median_impute(train: pd.DataFrame, other: pd.DataFrame, feats: list[str]):
    """Impute NaN with train medians. Returns (train_filled, other_filled, medians)."""
    med = {c: train[c].median() for c in feats}
    tr = train.copy()
    ot = other.copy()
    for c in feats:
        tr[c] = tr[c].fillna(med[c])
        ot[c] = ot[c].fillna(med[c]) if c in ot.columns else med[c]
    return tr, ot, med

def group_train_test_split(df: pd.DataFrame, y: np.ndarray, group_col="game_id", test_size=0.2, seed=RANDOM_SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[group_col].astype("int64").values
    idx_tr, idx_te = next(gss.split(df, y, groups))
    return idx_tr, idx_te

def eval_probs(p, y):
    return {
        "AUC": float(roc_auc_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p)),
        "BaseRate": float(np.mean(y)),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Load
    print("Loading W5 training/candidate dataâ€¦")
    train_raw = load_csv(TRAIN_CSV)
    cand_raw = load_csv(CANDS_CSV)

    # 2) Normalize ids/labels
    # Keep only rows with explicit labels in train (0/1). Some rows may be NaN for non-targets.
    if "complete" not in train_raw.columns:
        raise KeyError("TRAIN file missing 'complete' column.")
    train_raw["complete"] = pd.to_numeric(train_raw["complete"], errors="coerce")
    train_raw = train_raw[train_raw["complete"].isin([0.0, 1.0])].copy()
    train_raw["complete"] = train_raw["complete"].astype(int)

    # 3) Basic cleaning
    train_raw = replace_infs_nan(train_raw)
    cand_raw = replace_infs_nan(cand_raw)

    # 4) Feature selection (robust)
    print("Selecting features dynamicallyâ€¦")
    candidate_feats = pick_features(train_raw)
    print(f"{len(candidate_feats)} numeric candidates (first 20): {candidate_feats[:20]}")

    # 5) Drop columns that are too NaN or constant
    feats = drop_bad_columns(train_raw, candidate_feats)
    print(f"Using {len(feats)} features after NaN/constant filtering.")

    if len(feats) < 5:
        raise RuntimeError("Too few usable features after filtering. Check your W5 build.")

    # 6) Impute with train medians; align columns for candidates
    X_train = train_raw[feats].copy()
    X_cands = cand_raw.reindex(columns=feats).copy()
    X_train, X_cands, medians = median_impute(X_train, X_cands, feats)

    # Safety: fill any remaining NaN (shouldnâ€™t happen)
    X_train = X_train.fillna(0.0)
    X_cands = X_cands.fillna(0.0)

    y = train_raw["complete"].values.astype(int)

    # 7) Group-wise split by game_id
    print("Splitting by game_id for unbiased testâ€¦")
    tr_idx, te_idx = group_train_test_split(train_raw, y, group_col="game_id", test_size=0.2)
    Xtr, ytr = X_train.iloc[tr_idx].values, y[tr_idx]
    Xte, yte = X_train.iloc[te_idx].values, y[te_idx]

    # 8) Train XGB + Platt calibration (stable, no early stopping arg)
    print("Training + calibrating XGBoostâ€¦")
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=0,
        random_state=RANDOM_SEED,
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    calib = CalibratedClassifierCV(xgb, cv=5, method="sigmoid")
    calib.fit(Xtr, ytr)

    # 9) Metrics
    p_tr = calib.predict_proba(Xtr)[:, 1]
    p_te = calib.predict_proba(Xte)[:, 1]
    metrics = {
        "train": eval_probs(p_tr, ytr),
        "test": eval_probs(p_te, yte),
        "n_train": int(ytr.size),
        "n_test": int(yte.size),
        "features": feats,
    }
    print(json.dumps(metrics, indent=2))

    # 10) Score candidates
    print("Scoring candidatesâ€¦")
    cand_scored = cand_raw.copy()
    cand_scored["catch_prob"] = calib.predict_proba(X_cands.values)[:, 1]

    # Optional: within-play rank (top option) using catch_prob; safe if keys present
    if all(k in cand_scored.columns for k in ["game_id", "play_id"]):
        cand_scored["option_rank_in_play"] = (
            cand_scored.groupby(["game_id", "play_id"])["catch_prob"]
            .rank(method="first", ascending=False).astype(int)
        )
        cand_scored["is_top_option"] = (cand_scored["option_rank_in_play"] == 1).astype(int)

    # 11) Save
    cand_scored.to_csv(SCORED_CSV, index=False)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({"model": calib, "features": feats, "medians": medians}, MODEL_FILE)

    print(f"\nWrote {SCORED_CSV} and {METRICS_JSON}\nSaved model â†’ {MODEL_FILE}")
    cols_to_show = [c for c in ["game_id", "play_id", "wr_id", "catch_prob", "option_rank_in_play", "is_top_option"] if c in cand_scored.columns]
    print(cand_scored.head(10)[cols_to_show])

if __name__ == "__main__":
    main()



# WR_id_to_name.py â€” build nflId â†’ player_name mapping and mrge into scored CSV
import pandas as pd
from pathlib import Path

# Folder containing all weekly tracking files
DATA = Path(r"C:\Users\Germa\PycharmProjects\PythonProject")
OUT = Path("out")

MAPPING = OUT / "player_mapping.csv"
SCORED = OUT / "catch_candidates_scored_W5_rich.csv"
NAMED = OUT / "catch_candidates_scored_named.csv"

def main():
    # Collect all weeks
    weeks = []
    for w in range(1, 19):  # weeks 1â€“18
        fname = DATA / f"input_2023_w{w:02d}.csv"
        print(f"Loading {fname}...")
        df = pd.read_csv(fname, usecols=["nfl_id","player_name","player_position"])
        weeks.append(df)

    # Build master mapping
    mapping = pd.concat(weeks).drop_duplicates("nfl_id")
    mapping["nfl_id"] = mapping["nfl_id"].astype(int)

    # Save mapping for reuse
    mapping.to_csv(MAPPING, index=False)
    print(f"âœ… Wrote master mapping to {MAPPING}")

    # Merge with scored CSV
    scored = pd.read_csv(SCORED)
    scored["wr_id"] = scored["wr_id"].astype(int)

    scored = scored.merge(mapping, left_on="wr_id", right_on="nfl_id", how="left")
    scored.rename(columns={
        "player_name":"wr_name",
        "player_position":"wr_position"
    }, inplace=True)
    scored.drop(columns=["nfl_id"], inplace=True)

    # Save scoutâ€‘friendly file
    scored.to_csv(NAMED, index=False)
    print(f"âœ… Wrote scoutâ€‘ready file to {NAMED}")
    print(scored[["game_id","play_id","wr_id","wr_name","catch_prob"]].head().to_string(index=False))

if __name__ == "__main__":
    main()


scored = pd.read_csv("out/catch_candidates_scored_named.csv")

# Keep only the key columns
scored_small = scored[["game_id","play_id","wr_name","wr_position","catch_prob","wr_id"]]
scored_top = scored_small.sort_values(["game_id","play_id","catch_prob"], ascending=[True,True,False])
scored_top = scored_top.groupby(["game_id","play_id"]).head(1).reset_index(drop=True)
scored_sample = scored_small.head(500)
scored_high = scored_small[scored_small["catch_prob"] >= 0.7]
print(scored_high)
print(scored_sample)

scored_small.to_csv("out/catch_candidates_scored_scout_named_clean.csv", index=False)



import pandas as pd
import numpy as np

CATCH_PATH = "out/catch_candidates_scored_scout_named_clean.csv"
SUPP_PATH  = "supplementary_data.csv"
OUT_PATH   = "out/xwp_skeleton.csv"

# --- Load ---
catch = pd.read_csv(CATCH_PATH)
supp  = pd.read_csv(SUPP_PATH)

# --- Normalize keys: coerce to int (safer than str), drop NA keys ---
def to_int(s):
    # handle stray spaces or floats like 101.0
    return pd.to_numeric(pd.Series(s).astype(str).str.strip(), errors="coerce").astype("Int64")

catch["game_id_norm"] = to_int(catch["game_id"])
catch["play_id_norm"] = to_int(catch["play_id"])
supp["game_id_norm"]  = to_int(supp["game_id"])
supp["play_id_norm"]  = to_int(supp["play_id"])

# Drop rows without valid keys
catch = catch.dropna(subset=["game_id_norm","play_id_norm"])
supp  = supp.dropna(subset=["game_id_norm","play_id_norm"])

# --- Quick overlap diagnostics ---
catch_keys = set(zip(catch["game_id_norm"].astype(int), catch["play_id_norm"].astype(int)))
supp_keys  = set(zip(supp["game_id_norm"].astype(int),  supp["play_id_norm"].astype(int)))
intersect  = catch_keys & supp_keys

print(f"Catch rows:         {len(catch):,} (unique keys: {len(catch_keys):,})")
print(f"Supplemental rows:  {len(supp):,}  (unique keys: {len(supp_keys):,})")
print(f"Key intersection:   {len(intersect):,}")

if len(intersect) == 0:
    # Show a few example keys from each to eyeball the pattern
    print("Example catch keys:", list(catch_keys)[:10])
    print("Example supp  keys:", list(supp_keys)[:10])

# --- Compute team-specific WP from supplemental ---
need_cols = [
    "game_id","play_id","home_team_abbr","visitor_team_abbr",
    "possession_team","defensive_team","down","yards_to_go",
    "yardline_side","yardline_number",
    "pre_snap_home_team_win_probability","pre_snap_visitor_team_win_probability",
    "home_team_win_probability_added","visitor_team_win_probility_added",
    "pass_result","yards_gained"
]
# keep only existing
need_cols = [c for c in need_cols if c in supp.columns]
supp_small = supp[need_cols + ["game_id_norm","play_id_norm"]].copy()

is_home = (supp_small["possession_team"] == supp_small.get("home_team_abbr", ""))
supp_small["wp_base"] = np.where(
    is_home,
    supp_small["pre_snap_home_team_win_probability"],
    supp_small["pre_snap_visitor_team_win_probability"]
)
supp_small["wp_actual"] = supp_small["wp_base"] + np.where(
    is_home,
    supp_small["home_team_win_probability_added"],
    supp_small["visitor_team_win_probility_added"]
)

# --- Receiver-side columns from catch ---
recv_cols = [c for c in ["wr_id","wr_name","wr_position","catch_prob","air_yards",
                         "nearest_defender_dist","defenders_within_3","defenders_within_5",
                         "receiver_speed","sideline_dist","route","alignment"]
             if c in catch.columns]
catch_small = catch[["game_id","play_id","game_id_norm","play_id_norm"] + recv_cols].copy()

# --- Robust one-to-many merge on normalized keys ---
df = catch_small.merge(
    supp_small[
        ["game_id_norm","play_id_norm","possession_team","defensive_team",
         "home_team_abbr","visitor_team_abbr","down","yards_to_go",
         "yardline_side","yardline_number","yards_gained","pass_result",
         "wp_base","wp_actual"]
        ],
    on=["game_id_norm","play_id_norm"],
    how="left",
    validate="m:1"  # many catch rows -> one supp row
)

# Report missing matches (should be near zero once keys align)
miss = df["wp_base"].isna().sum()
print(f"Rows after merge: {len(df):,} | Missing wp_base/wp_actual: {miss:,}")

if miss:
    # Show a few problematic keys to debug
    print(df.loc[df["wp_base"].isna(), ["game_id","play_id"]].drop_duplicates().head(15))

# --- Final skeleton ---
final_cols = (
    ["game_id","play_id","wr_id","wr_name","wr_position","catch_prob"] +
    [c for c in ["air_yards","nearest_defender_dist","defenders_within_3","defenders_within_5",
                 "receiver_speed","sideline_dist","route","alignment"] if c in df.columns] +
    ["possession_team","defensive_team","home_team_abbr","visitor_team_abbr",
     "down","yards_to_go","yardline_side","yardline_number",
     "yards_gained","pass_result","wp_base","wp_actual"]
)
final = df[final_cols].sort_values(["game_id","play_id","catch_prob"], ascending=[True, True, False])
final.to_csv(OUT_PATH, index=False)
print(f"âœ… Wrote {OUT_PATH} with {len(final):,} rows.")



# THIS FILE IS USED FOR CALCULATING THE EXPECTED YARDS FOR EVERY POSSIBLE RECEIVER ON THE FIELD TO THEN FEED TO GET A WIN PROBAILITY. THIS FILE IS NEEDED FOR SUBMIISION

import pandas as pd
import numpy as np

SKELETON = "out/xwp_skeleton.csv"
CANDS    = "out/catch_candidates_scored_named.csv"

skel = pd.read_csv(SKELETON)
cand = pd.read_csv(CANDS)

# normalize keys
for df in (skel, cand):
    for c in ["game_id","play_id","wr_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# merge rich features into skeleton
use_cols = [
    "game_id","play_id","wr_id",
    # air / geometry
    "qb_wr_dist_mean_W5","theta_cos_mean_W5","throw_dist_sq_mean_W5",
    # yac drivers
    "wr_speed_mean_W5","nearest_db_dist_mean_W5","db_speed_mean_W5",
    "lane_clear_min_dist_mean_W5","pressure_cnt_cap3_mean_W5",
    "wr_speed_sep_mean_W5","sep_over_throw_mean_W5","pressure_per_sep_mean_W5",
    # (this file also has catch_prob; skeleton already has, but keep for safety)
    "catch_prob"
]
cand_small = cand[use_cols].copy()

df = skel.merge(cand_small, on=["game_id","play_id","wr_id"], how="left", validate="1:1")

# yardline_100 (distance to opponent EZ) from side/number if not present
def yardline_100_from_side_number(row):
    side, num, off = row["yardline_side"], row["yardline_number"], row["possession_team"]
    if pd.isna(side) or pd.isna(num) or pd.isna(off): return np.nan
    num = int(num)
    return num if str(side) != str(off) else 100 - num

if "yardline_100" not in df.columns:
    df["yardline_100"] = df.apply(yardline_100_from_side_number, axis=1)

# expected AIR yards from QBâ†’WR geometry
def expected_air_yards(r, fallback=8.0):
    d   = r.get("qb_wr_dist_mean_W5")
    cos = r.get("theta_cos_mean_W5")
    if pd.notna(d) and pd.notna(cos): return max(0.0, float(d) * abs(float(cos)))
    if pd.notna(d):                   return max(0.0, float(d))
    t2 = r.get("throw_dist_sq_mean_W5")
    if pd.notna(t2):                  return max(0.0, np.sqrt(float(t2)))
    return float(fallback)

# expected YAC (conservative, monotonic)
def expected_yac(r):
    yac = 1.5
    nd = r.get("nearest_db_dist_mean_W5");     ws = r.get("wr_speed_mean_W5")
    lmin = r.get("lane_clear_min_dist_mean_W5"); pres = r.get("pressure_cnt_cap3_mean_W5")
    wss = r.get("wr_speed_sep_mean_W5");       lpress = r.get("pressure_per_sep_mean_W5")
    sot = r.get("sep_over_throw_mean_W5")
    if pd.notna(nd):   yac += 0.75 * min(float(nd), 6)
    if pd.notna(ws):   yac += 0.45 * max(float(ws) - 7.5, 0)
    if pd.notna(lmin): yac += 0.40 * min(max(float(lmin),0), 5)
    if pd.notna(wss):  yac += 0.20 * max(float(wss), 0)
    if pd.notna(pres): yac -= 0.40 * float(pres)
    if pd.notna(lpress): yac -= 0.15 * max(float(lpress), 0)
    if pd.notna(sot):  yac += 0.10 * float(sot)
    return max(0.0, float(yac))

df["exp_air_yards"] = df.apply(expected_air_yards, axis=1)
df["exp_yac"]       = df.apply(expected_yac,       axis=1)
df["exp_gain"]      = df["exp_air_yards"] + df["exp_yac"]

# catch-state from expected total gain
def build_catch_state(r, kickoff_yl=75):
    yl0  = float(r["yardline_100"])
    ytg0 = int(r["yards_to_go"])
    d0   = int(r["down"])
    g    = int(max(0, np.floor(r["exp_gain"])))
    new_yl = max(1, yl0 - g)
    if (yl0 - g) <= 0:  # TD â†’ kickoff-ish reset
        return pd.Series({"down_c": 1, "ydstogo_c": 10, "yl_c": int(kickoff_yl)})
    if g >= ytg0:       # first down
        return pd.Series({"down_c": 1, "ydstogo_c": int(min(10, new_yl)), "yl_c": int(new_yl)})
    return pd.Series({"down_c": min(4, d0+1), "ydstogo_c": int(max(1, ytg0 - g)), "yl_c": int(new_yl)})

catch_states = df.apply(build_catch_state, axis=1)
df = pd.concat([df, catch_states], axis=1)

# handoff for R
df["row_id"] = np.arange(len(df))
df[["row_id","down_c","ydstogo_c","yl_c"]].to_csv("out/wp_catch_states_for_R.csv", index=False)

# keep full working table
df.to_csv("out/xwp_working_with_states.csv", index=False)
print("âœ… wrote out/wp_catch_states_for_R.csv and out/xwp_working_with_states.csv")



import pandas as pd
import numpy as np

WORK = "out/xwp_working_with_states.csv"      # has exp_yac/exp_gain + catch-state
SUPP = "supplementary_data.csv"               # one row per (game_id, play_id)
OUT_R = "out/wp_catch_states_for_R.csv"

# --- load ---
work = pd.read_csv(WORK)
supp = pd.read_csv(SUPP, usecols=[
    "game_id","play_id","quarter","game_clock",
    "pre_snap_home_score","pre_snap_visitor_score"
])

# --- normalize keys & many-to-one merge (keeps ALL receiver rows) ---
for df in (work, supp):
    for c in ["game_id","play_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

pre_n = len(work)
work = work.merge(supp, on=["game_id","play_id"], how="left", validate="m:1")
assert len(work) == pre_n, "Row count changed â€” merge should be many-to-one, investigate keys."

# --- derive time remaining (regulation) ---
def mmss_to_seconds(s):
    if pd.isna(s): return np.nan
    try:
        m, sec = str(s).split(":")
        return int(m)*60 + int(sec)
    except:
        return np.nan

qsec = work["game_clock"].apply(mmss_to_seconds)
work["game_seconds_remaining"] = (4 - work["quarter"].clip(1,4)) * 900 + qsec

# --- score diff from posteam perspective BEFORE play ---
is_home = (work["possession_team"] == work["home_team_abbr"])
home_diff = work["pre_snap_home_score"] - work["pre_snap_visitor_score"]
away_diff = work["pre_snap_visitor_score"] - work["pre_snap_home_score"]
work["score_diff_base"] = np.where(is_home, home_diff, away_diff)

# --- TD in counterfactual? use your exp_gain and starting yardline_100 ---
yl0 = work["yardline_100"].astype(float)
g   = np.floor(work["exp_gain"].clip(lower=0)).astype(int)
td_c = (yl0 - g) <= 0
work["score_differential"] = work["score_diff_base"] + np.where(td_c, 7, 0)

# --- build nflfastR input (keep row_id for perfect merge back) ---
if "row_id" not in work.columns:
    work["row_id"] = np.arange(len(work))

r_out = work.rename(columns={
    "down_c": "down",
    "ydstogo_c": "ydstogo",
    "yl_c": "yardline_100",
    "possession_team": "posteam",
    "defensive_team": "defteam",
    "home_team_abbr": "home_team",
    "visitor_team_abbr": "away_team",
})[[
    "row_id",
    "down","ydstogo","yardline_100",
    "posteam","defteam","home_team","away_team",
    "score_differential","game_seconds_remaining"
]]

# quick sanity
print("Rows to send to R:", len(r_out), "| missing any required?",
      r_out.isna()[["down","ydstogo","yardline_100","posteam","defteam","home_team","away_team","score_differential","game_seconds_remaining"]].sum().to_dict())

r_out.to_csv(OUT_R, index=False)
work.to_csv(WORK, index=False)  # keep augmented context alongside your exp_yac, etc.
print(f"âœ… Rewrote {OUT_R}. Now rerun your Python wrapper to compute wp_catch.")



import pandas as pd, numpy as np

WORK = "out/xwp_working_with_states.csv"      # per-receiver rows (source of truth)
SUPP = "supplementary_data.csv"               # optional fallback if anything missing
OUT_R = "out/wp_catch_states_for_R.csv"

# ---------------- Helpers ----------------
def mmss_to_seconds(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    if not s: return np.nan
    parts = s.split(":")
    if len(parts) < 2: return np.nan
    m = int(parts[0])
    sec = float(parts[1])
    return int(round(m*60 + sec))

def pick(df, *names):
    """Return the first existing column name from names, else None."""
    for n in names:
        if n in df.columns:
            return n
    return None

# ---------------- Load ----------------
work = pd.read_csv(WORK)

# Optional: load SUPP only if needed later
try:
    supp = pd.read_csv(SUPP)
except Exception:
    supp = pd.DataFrame()

# Keys numeric
for df in (work, supp):
    if not df.empty:
        for c in ("game_id","play_id"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# ---------------- Time remaining ----------------
# Prefer game_seconds_remaining already in work; compute only if missing
if "game_seconds_remaining" in work.columns and work["game_seconds_remaining"].notna().any():
    gsr = pd.to_numeric(work["game_seconds_remaining"], errors="coerce")
else:
    q_col = pick(work, "quarter")
    gc_col = pick(work, "game_clock")
    if q_col is None or gc_col is None:
        # try from supplementary
        if q_col is None and "quarter" in supp.columns:
            work = work.merge(supp[["game_id","play_id","quarter"]].drop_duplicates(), on=["game_id","play_id"], how="left")
            q_col = "quarter"
        if gc_col is None and "game_clock" in supp.columns:
            work = work.merge(supp[["game_id","play_id","game_clock"]].drop_duplicates(), on=["game_id","play_id"], how="left")
            gc_col = "game_clock"
    if q_col is None or gc_col is None:
        raise KeyError("quarter/game_clock not found in either working or supplementary data.")
    q = pd.to_numeric(work[q_col], errors="coerce").clip(lower=1, upper=4).fillna(1).astype(int)
    qsec = work[gc_col].apply(mmss_to_seconds)
    gsr = (4 - q) * 900 + qsec

work["game_seconds_remaining"] = gsr
q = pd.to_numeric(pick(work, "quarter") and work["quarter"], errors="coerce").clip(lower=1, upper=4).fillna(1).astype(int)
half_end = np.where(q <= 2, 1800, 3600)
work["half_seconds_remaining"] = half_end - (3600 - work["game_seconds_remaining"])

# ---------------- Pre-snap score diff (already present, but recompute if absent) ----------------
if "score_diff_base" in work.columns:
    score_diff_base = pd.to_numeric(work["score_diff_base"], errors="coerce")
else:
    is_home = (work["possession_team"] == work["home_team_abbr"])
    home_diff = work["pre_snap_home_score"] - work["pre_snap_visitor_score"]
    away_diff = work["pre_snap_visitor_score"] - work["pre_snap_home_score"]
    score_diff_base = np.where(is_home, home_diff, away_diff)

# ---------------- Source pre-snap state ----------------
down0 = pd.to_numeric(work[pick(work, "down_c", "down")], errors="coerce").astype("Int64")
ydtg0 = pd.to_numeric(work[pick(work, "ydstogo_c", "yards_to_go")], errors="coerce").astype("Int64")
yl0   = pd.to_numeric(work[pick(work, "yl_c", "yardline_100")], errors="coerce")

# Expected gain (non-negative integer)
g = np.floor(pd.to_numeric(work["exp_gain"], errors="coerce").clip(lower=0)).fillna(0).astype(int)

# ---------------- Post-catch yardline ----------------
yl1 = np.maximum(yl0 - g, 0)
td  = (yl1 <= 0)

# Score differential after catch: +7 for TD (simple tonight-friendly assumption)
score_differential = score_diff_base + np.where(td, 7, 0)

# ---------------- Down & distance after catch ----------------
down1 = down0.copy()
ydtg1 = ydtg0.copy()

# First down achieved (but not TD)
first_down = (g >= ydtg0) & (~td)

# If first down â†’ 1st & goal-to-go inside 10, else 1st & 10
goal_to_go = np.minimum(10, np.maximum(yl1, 1))  # yards to goal line (cap at 10; min 1)
down1 = np.where(first_down, 1, down1)
ydtg1 = np.where(first_down, goal_to_go, ydtg1)

# If no first down and no TD â†’ next down & reduce distance (min 1)
no_fd = (~first_down) & (~td)
down1 = np.where(no_fd, np.minimum(down0 + 1, 4), down1)
ydtg1 = np.where(no_fd, np.maximum(1, ydtg0 - g), ydtg1)

# TD rows: keep pre-snap down & distance (we already bumped score)
# (Full kickoff state modeling can be added later if desired.)

# ---------------- Safe defaults required by nflfastR ----------------
receive_2h_ko = 0
spread_line   = 0

# Ensure row_id exists
if "row_id" not in work.columns:
    work["row_id"] = np.arange(len(work))

# ---------------- Build R states CSV (counterfactual, post-catch) ----------------
states = pd.DataFrame({
    "row_id": work["row_id"],
    "down":   pd.to_numeric(down1, errors="coerce").astype("int64"),
    "ydstogo": pd.to_numeric(ydtg1, errors="coerce").astype("int64"),
    "yardline_100": pd.to_numeric(yl1, errors="coerce"),
    "posteam": work["possession_team"],
    "defteam": work["defensive_team"],
    "home_team": work["home_team_abbr"],
    "away_team": work["visitor_team_abbr"],
    "score_differential": pd.to_numeric(score_differential, errors="coerce").astype("int64"),
    "game_seconds_remaining": pd.to_numeric(work["game_seconds_remaining"], errors="coerce"),
    "half_seconds_remaining": pd.to_numeric(work["half_seconds_remaining"], errors="coerce"),
    "spread_line": spread_line,
    "receive_2h_ko": receive_2h_ko,
    "game_id": work["game_id"],
    "play_id": work["play_id"],
})

states.to_csv(OUT_R, index=False)
print("âœ… Rewrote", OUT_R)



# 12_13_wp_python_wrapper.py
import os, subprocess, sys
import pandas as pd
import numpy as np  # used by rebuild

# ---------------- Paths ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "out")

WORK_PATH    = os.path.join(OUT, "xwp_working_with_states.csv")   # per-receiver rows
STATES_PATH  = os.path.join(OUT, "wp_catch_states_for_R.csv")     # post-catch counterfactual states
RSCRIPT_OUT  = os.path.join(OUT, "wp_catch_results.csv")
RSCRIPT_FILE = os.path.join(OUT, "wp_catch_calc.R")
FINAL_OUT    = os.path.join(OUT, "xwp_matrix.csv")

os.makedirs(OUT, exist_ok=True)

# R-friendly (forward slashes)
STATES_PATH_FWD = STATES_PATH.replace("\\", "/")
RSCRIPT_OUT_FWD = RSCRIPT_OUT.replace("\\", "/")

# ---------------- Helpers ----------------
def _mmss_to_seconds(s):
    if pd.isna(s): return None
    s = str(s).strip()
    if not s or ":" not in s: return None
    m, sec = s.split(":")[0], s.split(":")[1]
    try:
        return int(m)*60 + int(float(sec))
    except:
        return None

def rebuild_states_from_work(work_df: pd.DataFrame) -> pd.DataFrame:
    # ensure keys exist
    if "row_id" not in work_df.columns:
        work_df = work_df.reset_index().rename(columns={"index":"row_id"})

    down0 = pd.to_numeric(work_df.get("down_c", work_df.get("down")), errors="coerce")
    ydtg0 = pd.to_numeric(work_df.get("ydstogo_c", work_df.get("yards_to_go")), errors="coerce")
    yl0   = pd.to_numeric(work_df.get("yl_c", work_df.get("yardline_100")), errors="coerce")
    g     = pd.to_numeric(work_df.get("exp_gain"), errors="coerce").clip(lower=0).fillna(0).astype(int)

    yl1 = (yl0 - g).clip(lower=0)
    td  = (yl1 <= 0)

    # score base
    if "score_diff_base" in work_df.columns:
        sbase = pd.to_numeric(work_df["score_diff_base"], errors="coerce")
    else:
        is_home = (work_df["possession_team"] == work_df["home_team_abbr"])
        home_diff = work_df["pre_snap_home_score"] - work_df["pre_snap_visitor_score"]
        away_diff = work_df["pre_snap_visitor_score"] - work_df["pre_snap_home_score"]
        sbase = np.where(is_home, home_diff, away_diff)

    score_diff = pd.to_numeric(sbase, errors="coerce").fillna(0) + np.where(td, 7, 0)

    # down & distance after catch
    down1 = down0.copy()
    ydtg1 = ydtg0.copy()
    first_down = (g >= ydtg0) & (~td)
    goal_to_go = np.minimum(10, np.maximum(yl1, 1))
    down1 = np.where(first_down, 1, down1)
    ydtg1 = np.where(first_down, goal_to_go, ydtg1)
    no_fd = (~first_down) & (~td)
    down1 = np.where(no_fd, np.minimum((down0.fillna(1)).astype(int) + 1, 4), down1)
    ydtg1 = np.where(no_fd, np.maximum(1, (ydtg0.fillna(10)).astype(int) - g), ydtg1)

    # game seconds remaining
    if "game_seconds_remaining" in work_df.columns:
        gsr = pd.to_numeric(work_df["game_seconds_remaining"], errors="coerce")
    else:
        q = pd.to_numeric(work_df["quarter"], errors="coerce").clip(lower=1, upper=4).fillna(1).astype(int)
        qsec = work_df["game_clock"].apply(_mmss_to_seconds)
        gsr = (4 - q) * 900 + pd.Series(qsec)

    states_new = pd.DataFrame({
        "row_id": work_df["row_id"],
        # wrap np.where outputs in Series before fillna / astype
        "down": pd.Series(down1, index=work_df.index).astype(float).fillna(1).astype(int),
        "ydstogo": pd.Series(ydtg1, index=work_df.index).astype(float).fillna(10).astype(int),
        "yardline_100": pd.to_numeric(yl1, errors="coerce"),
        "posteam": work_df["possession_team"],
        "defteam": work_df["defensive_team"],
        "home_team": work_df["home_team_abbr"],
        "away_team": work_df["visitor_team_abbr"],
        "score_differential": pd.to_numeric(score_diff, errors="coerce").fillna(0).astype(int),
        "game_seconds_remaining": pd.to_numeric(gsr, errors="coerce"),
        "game_id": work_df["game_id"],
        "play_id": work_df["play_id"],
    })

    return states_new

# ---------------- Basic file checks ----------------
if not os.path.exists(WORK_PATH):
    print(f"Missing required file: {WORK_PATH}")
    sys.exit(1)

# ---------------- Force-sync & debug BEFORE calling R ----------------
print("ğŸ”� ROOT      :", ROOT)
print("ğŸ”� OUT       :", OUT)
print("ğŸ”� WORK_PATH :", WORK_PATH)
print("ğŸ”� STATES_PATH:", STATES_PATH)

# Always (re)build states from the CURRENT work so R sees the right columns
work_df = pd.read_csv(WORK_PATH)
states_df = rebuild_states_from_work(work_df)
states_df.to_csv(STATES_PATH, index=False)
print("ğŸ”� Rewrote states â†’", STATES_PATH)

# Show what Python is about to pass to R
_py_hdr = pd.read_csv(STATES_PATH, nrows=1)
print("ğŸ�� Python sees columns:", list(_py_hdr.columns))
print("ğŸ�� File size (bytes):", os.path.getsize(STATES_PATH))

# ---------------- Write the R script ----------------
r_code = r'''
options(timeout = 600)

# --- Library path: put userlib FIRST so our pinned packages win ---
ver_major <- R.version$major
ver_minor_major <- strsplit(R.version$minor, "\\.")[[1]][1]
ver_short <- paste(ver_major, ver_minor_major, sep=".")
userlib <- Sys.getenv("R_LIBS_USER", unset = file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", ver_short))
dir.create(userlib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(userlib, .libPaths()))

ipkg <- function(pkgs){
  installed <- rownames(installed.packages(lib.loc = .libPaths()))
  new <- pkgs[!(pkgs %in% installed)]
  if (length(new)) install.packages(new, repos=c("https://nflverse.r-universe.dev","https://cloud.r-project.org"), lib=userlib, quiet=TRUE)
  for (p in pkgs) suppressPackageStartupMessages(library(p, character.only=TRUE, quietly=TRUE, warn.conflicts=FALSE))
}

# Ensure 'remotes' exists for version pinning
ipkg(c("remotes"))

# --- PIN xgboost to 1.7.7.1 BEFORE loading nflfastR/fastrmodels ---
need_xgb <- TRUE
if ("xgboost" %in% rownames(installed.packages(lib.loc = .libPaths()))) {
  ver <- as.character(utils::packageVersion("xgboost"))
  need_xgb <- (utils::compareVersion(ver, "1.7.7.1") != 0)
}
if (need_xgb) {
  message("Installing xgboost 1.7.7.1 for model compatibility â€¦")
  remotes::install_version("xgboost", version = "1.7.7.1",
                           repos = c("https://cloud.r-project.org"),
                           upgrade = "never", quiet = TRUE, lib = userlib)
}
suppressPackageStartupMessages(library(xgboost, quietly=TRUE, warn.conflicts=FALSE, lib.loc=userlib))

# Now load core deps (after xgboost is in place)
ipkg(c("readr","dplyr","fastrmodels","nflfastR"))

cat("R version:", R.version.string, "\n")
cat("xgboost version:", as.character(utils::packageVersion("xgboost")), "\n")
cat("fastrmodels version:", as.character(utils::packageVersion("fastrmodels")), "\n")
cat("nflfastR version:", as.character(utils::packageVersion("nflfastR")), "\n")
cat(".libPaths():", paste(.libPaths(), collapse=" | "), "\n")

suppressWarnings(suppressMessages({
  states_path <- Sys.getenv("STATES_PATH")
  out_path    <- Sys.getenv("RSCRIPT_OUT")
  if (!nzchar(states_path)) stop("STATES_PATH env var is empty")
  if (!nzchar(out_path))    stop("RSCRIPT_OUT env var is empty")

  states <- readr::read_csv(states_path, show_col_types = FALSE)

  # Safe normalizer: dots OR whitespace â†’ underscore
  norm <- function(x){ tolower(trimws(gsub("[.[:space:]]+","_", x))) }
  names(states) <- norm(names(states))

  cat("ğŸ§° R states_path:", states_path, "\n")
  cat("ğŸ§° R columns (normalized):", paste(names(states), collapse=", "), "\n")
  cat("ğŸ§° R first row preview:\n"); print(utils::head(states, 1))

  required <- c("row_id","down","ydstogo","yardline_100","posteam","defteam","home_team","away_team","score_differential","game_seconds_remaining")
  miss <- setdiff(required, names(states))
  if (length(miss)) stop("Missing required columns in states: ", paste(miss, collapse=", "))

  inp <- states |>
    dplyr::transmute(
      row_id                 = .data$row_id,
      down                   = as.integer(.data$down),
      ydstogo                = as.integer(.data$ydstogo),
      yardline_100           = as.numeric(.data$yardline_100),
      posteam                = .data$posteam,
      defteam                = .data$defteam,
      home_team              = .data$home_team,
      away_team              = .data$away_team,
      score_differential     = as.integer(.data$score_differential),
      game_seconds_remaining = as.numeric(.data$game_seconds_remaining)
    )

    # --- Defaults required by nflfastR ---
  if (!"spread_line" %in% names(inp))                  inp$spread_line <- 0
  if (!"posteam_timeouts_remaining" %in% names(inp))   inp$posteam_timeouts_remaining <- 3
  if (!"defteam_timeouts_remaining"   %in% names(inp)) inp$defteam_timeouts_remaining <- 3
  if (!"receive_2h_ko" %in% names(inp))                inp$receive_2h_ko <- 0
  if (!"half_seconds_remaining" %in% names(inp)) {
    gsr <- as.numeric(inp$game_seconds_remaining)
    inp$half_seconds_remaining <- ifelse(is.na(gsr), NA_real_,
                                         ifelse(gsr > 1800, gsr - 1800, gsr))
  }


  # With xgboost 1.7.7.1, the fastrmodels WP binary will load
  res <- nflfastR::calculate_win_probability(inp) |>
    dplyr::transmute(row_id, wp_catch = wp)

  # carry keys back for easier merging/debug
  if ("game_id" %in% names(states)) {
    res <- dplyr::left_join(res, states |> dplyr::select(row_id, game_id), by="row_id")
  }
  if ("play_id" %in% names(states)) {
    res <- dplyr::left_join(res, states |> dplyr::select(row_id, play_id), by="row_id")
  }

  readr::write_csv(res, out_path)
}))

'''

with open(RSCRIPT_FILE, "w", encoding="utf-8") as f:
    f.write(r_code)

# show the R call line for sanity
with open(RSCRIPT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if "calculate_win_probability" in line:
            print("[R call]", line.strip())

# ---------------- Run R ----------------
def run_rscript(script_path):
    env = dict(os.environ)
    env["STATES_PATH"] = STATES_PATH_FWD
    env["RSCRIPT_OUT"] = RSCRIPT_OUT_FWD
    cmd = ["Rscript", "--vanilla", script_path]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        if completed.stdout.strip(): print("[R stdout]", completed.stdout.strip())
        if completed.stderr.strip(): print("[R stderr]", completed.stderr.strip())
    except FileNotFoundError:
        print("ERROR: 'Rscript' not found. Install R and ensure Rscript is on PATH."); sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("ERROR running Rscript:", e)
        print("STDOUT:\n", e.stdout); print("STDERR:\n", e.stderr); sys.exit(1)

print("â–¶ Running R to compute wp_catch â€¦")
print("R file:", RSCRIPT_FILE)
run_rscript(RSCRIPT_FILE)
if not os.path.exists(RSCRIPT_OUT):
    print("ERROR: R script completed but wp_catch_results.csv was not created."); sys.exit(1)
print("âœ… R finished and wrote", RSCRIPT_OUT)

# ---------------- Merge + metrics ----------------
work = pd.read_csv(WORK_PATH)
if "row_id" not in work.columns:
    work = work.reset_index().rename(columns={"index": "row_id"})

wp_c = pd.read_csv(RSCRIPT_OUT)

# If R didn't carry keys back, hydrate them from WORK now
if "game_id" not in wp_c.columns or "play_id" not in wp_c.columns:
    wp_c = wp_c.merge(
        work[["row_id", "game_id", "play_id"]].drop_duplicates("row_id"),
        on="row_id", how="left"
    )

# Prefer row_id merge; fall back to (game_id, play_id) if too many NAs
df = work.merge(
    wp_c[["row_id","wp_catch","game_id","play_id"]].drop_duplicates("row_id"),
    on="row_id", how="left"
)
# --- Ensure game_id / play_id exist in df using the STATES file as ground truth ---
try:
    states_map = pd.read_csv(STATES_PATH, usecols=["row_id", "game_id", "play_id"])
except Exception:
    states_map = None

def _coalesce_into(df, base, src):
    if base not in df.columns and src in df.columns:
        df[base] = df[src]
    elif base in df.columns and src in df.columns:
        # preserve nullable ints if present
        if str(df[base].dtype) == "Int64" or str(df[src].dtype) == "Int64":
            df[base] = df[base].astype("Int64")
            df[src]  = df[src].astype("Int64")
        df[base] = df[base].where(df[base].notna(), df[src])
    return df

if states_map is not None:
    df = df.merge(states_map, on="row_id", how="left", suffixes=("", "_from_states"))
    df = _coalesce_into(df, "game_id", "game_id_from_states")
    df = _coalesce_into(df, "play_id", "play_id_from_states")
    # clean up helpers if they exist
    for c in ["game_id_from_states", "play_id_from_states"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)


if df["wp_catch"].isna().mean() > 0.01 and {"game_id","play_id"}.issubset(wp_c.columns) and {"game_id","play_id"}.issubset(work.columns):
    print("â„¹ï¸� Falling back to merge on (game_id, play_id)â€¦")
    df = work.merge(
        wp_c[["game_id","play_id","wp_catch"]].drop_duplicates(["game_id","play_id"]),
        on=["game_id","play_id"], how="left"
    )

# catch prob column detection
catch_prob_col = next((c for c in ["catch_prob_y","catch_prob_x","catch_prob"] if c in df.columns), None)
if catch_prob_col is None:
    print("ERROR: No catch probability column found (tried catch_prob_y/x)."); sys.exit(1)

# Required baselines
for c in ["wp_base","wp_actual"]:
    if c not in df.columns:
        print(f"ERROR: Required column '{c}' missing in {WORK_PATH}."); sys.exit(1)

# Metrics
df["decision_value"]   = df["wp_catch"] - df["wp_base"]
df["expected_gain_wp"] = df[catch_prob_col].clip(0,1).fillna(0) * df["decision_value"]
df["opportunity_gap"]  = df["wp_catch"] - df["wp_actual"]
df["score"]            = df["expected_gain_wp"]

preferred_cols = [
    "game_id","play_id","wr_id","wr_name","wr_position",
    catch_prob_col, "exp_air_yards","exp_yac","exp_gain",
    "down","yards_to_go","yardline_100",
    "wp_base","wp_catch","wp_actual",
    "decision_value","expected_gain_wp","opportunity_gap","score"
]
final_cols = [c for c in preferred_cols if c in df.columns]
missing = [c for c in preferred_cols if c not in df.columns]
if missing: print("âš ï¸� Missing in export (skipped):", missing)

# Sort by whatever keys exist (robust)
sort_keys = [c for c in ["game_id","play_id","score"] if c in final_cols]
if not sort_keys:
    sort_keys = [c for c in ["row_id","score"] if c in df.columns]

df[final_cols].sort_values(sort_keys, ascending=[True]*(len(sort_keys)-1)+[False]).to_csv(FINAL_OUT, index=False)
print("âœ… Wrote", FINAL_OUT)


