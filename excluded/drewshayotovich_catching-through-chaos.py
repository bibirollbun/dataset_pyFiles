from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# Kaggle paths
# -----------------------
KAGGLE_INPUT = Path("/kaggle/input/nfl-big-data-bowl-2026-analytics")
RAW = KAGGLE_INPUT / "114239_nfl_competition_files_published_analytics_final" / "train"
SUPP = KAGGLE_INPUT / "114239_nfl_competition_files_published_analytics_final" / "supplementary_data.csv"

# writeable outputs
ROOT_OUT = Path("/kaggle/working")
OUTDIR = ROOT_OUT / "data" / "interim"
FIGDIR = ROOT_OUT / "figures" / "gallery"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

print("[INFO] RAW:", RAW)
print("[INFO] SUPP:", SUPP)
print("[INFO] OUTDIR:", OUTDIR)
print("[INFO] FIGDIR:", FIGDIR)

# -----------------------
# 0) Discover available weeks
# -----------------------
week_files = sorted(RAW.glob("input_2023_w*.csv"))
weeks = [int(p.stem.split("w")[-1]) for p in week_files]

print(f"[INFO] Found {len(week_files)} input week files.")
print(f"[INFO] Found weeks: {weeks}")

# Optional sanity check: ensure matching outputs exist
missing_out = []
for wk in weeks:
    out_path = RAW / f"output_2023_w{wk:02d}.csv"
    if not out_path.exists():
        missing_out.append(wk)
if missing_out:
    print(f"[WARN] Missing output files for weeks: {missing_out}")

# -----------------------
# 1) Load Supplementary (once)
# -----------------------
supp = pd.read_csv(SUPP, low_memory=False)

supp_keep = [
    "game_id","play_id","season","week","game_date","quarter","game_clock",
    "down","yards_to_go","pass_result","pass_length","route_of_targeted_receiver",
    "team_coverage_man_zone","team_coverage_type","possession_team","defensive_team",
    "offense_formation","receiver_alignment","expected_points","expected_points_added"
]
supp_keep = [c for c in supp_keep if c in supp.columns]
supp_small = supp[supp_keep].drop_duplicates(["game_id","play_id"]).copy()

print(f"[INFO] Supplementary rows: {len(supp_small)} | cols: {supp_small.columns.tolist()}")



# -----------------------
# 2) Process all weeks
# -----------------------
long_parts = []
meta_parts = []
role_parts = []

for wk in tqdm(weeks, desc="Weeks"):
    inp_path = RAW / f"input_2023_w{wk:02d}.csv"
    out_path = RAW / f"output_2023_w{wk:02d}.csv"
    if not inp_path.exists() or not out_path.exists():
        print(f"[WARN] Missing week {wk:02d} input/output; skipping.")
        continue

    df_in = pd.read_csv(inp_path, low_memory=False)
    df_out = pd.read_csv(out_path, low_memory=False)

    # Normalize role/side strings
    for c in ("player_role","player_side"):
        if c in df_in.columns:
            df_in[c] = df_in[c].astype(str).str.strip()

    # Target plays (one targeted WR per play)
    mask_trg = (df_in["player_role"] == "Targeted Receiver")
    trg = (
        df_in.loc[mask_trg, ["game_id","play_id","nfl_id","num_frames_output","ball_land_x","ball_land_y"]]
          .drop_duplicates(["game_id","play_id"])
          .rename(columns={"nfl_id":"target_nfl_id"})
          .copy()
    )
    keep_keys = set(zip(trg["game_id"], trg["play_id"]))

    # Roles table per player-play (for later joins / DPR)
    role_cols = ["game_id","play_id","nfl_id","player_side","player_role","player_position","player_name"]
    role_cols = [c for c in role_cols if c in df_in.columns]
    roles = df_in[role_cols].drop_duplicates(["game_id","play_id","nfl_id"]).copy()
    role_parts.append(roles)

    # --- NEW: QB location at time of throw (approx = last input frame) ---
    if "player_role" in df_in.columns:
        qb_in = df_in[df_in["player_role"] == "Passer"].copy()
        if not qb_in.empty:
            qb_release = (
                qb_in.sort_values("frame_id")
                     .groupby(["game_id","play_id"], as_index=False)
                     .tail(1)[["game_id","play_id","x","y"]]
                     .rename(columns={"x": "qb_x", "y": "qb_y"})
            )
        else:
            qb_release = pd.DataFrame(columns=["game_id","play_id","qb_x","qb_y"])
    else:
        qb_release = pd.DataFrame(columns=["game_id","play_id","qb_x","qb_y"])


    # Filter output to only those plays
    out_keep = df_out[df_out[["game_id","play_id"]].apply(tuple, axis=1).isin(keep_keys)].copy()
    if out_keep.empty:
        print(f"[WARN] Week {wk:02d}: no overlapping plays in output; skipping.")
        continue

    # Join roles
    out_ann = out_keep.merge(roles, on=["game_id","play_id","nfl_id"], how="left", validate="many_to_one")

    # --- NEW: attach qb_x, qb_y (per play) ---
    if not qb_release.empty:
        out_ann = out_ann.merge(
            qb_release,
            on=["game_id","play_id"],
            how="left",
            validate="many_to_one",
        )


    # Prepare per-play lookups
    t_map = {(int(r.game_id), int(r.play_id)): int(r.target_nfl_id) for _, r in trg.iterrows()}
    t2_map = {(int(r.game_id), int(r.play_id)): int(r.num_frames_output) for _, r in trg.iterrows()}
    blx_map = {(int(r.game_id), int(r.play_id)): float(r.ball_land_x) for _, r in trg.iterrows()}
    bly_map = {(int(r.game_id), int(r.play_id)): float(r.ball_land_y) for _, r in trg.iterrows()}

    # Flags/constants
    keys = out_ann[["game_id","play_id"]].apply(tuple, axis=1)
    out_ann["is_target"] = out_ann.apply(lambda r: int(r.nfl_id) == t_map[(int(r.game_id), int(r.play_id))], axis=1)
    out_ann["t0"] = 1
    out_ann["t2"] = keys.map(t2_map)
    out_ann["ball_land_x"] = keys.map(blx_map)
    out_ann["ball_land_y"] = keys.map(bly_map)

    # Supplementary join
    out_ann = out_ann.merge(supp_small, on=["game_id","play_id"], how="left", validate="many_to_one")

    # Downcast for size
    for col in ("x","y","s","a","dir","ball_land_x","ball_land_y","pass_length","expected_points","expected_points_added"):
        if col in out_ann.columns:
            out_ann[col] = pd.to_numeric(out_ann[col], errors="coerce").astype("float32")

    long_parts.append(out_ann)

    meta = (
        out_ann[out_ann["is_target"]]
        .drop_duplicates(["game_id","play_id","nfl_id"])
        .rename(columns={"nfl_id":"target_nfl_id"})
        .copy()
    )
    meta_keep = [
        "game_id","play_id","target_nfl_id","t0","t2","ball_land_x","ball_land_y",
        "season","week","pass_result","pass_length","route_of_targeted_receiver",
        "team_coverage_man_zone","team_coverage_type","possession_team","defensive_team",
        "offense_formation","receiver_alignment","expected_points","expected_points_added",
        "player_name","player_position",
        # NEW:
        "qb_x","qb_y",
    ]
    meta = meta[[c for c in meta_keep if c in meta.columns]]
    meta_parts.append(meta)


# -----------------------
# 3) Concatenate + sort + write
# -----------------------
if not long_parts:
    raise RuntimeError("No weeks processed. Check file locations.")
targets_long = pd.concat(long_parts, ignore_index=True)
targets_meta = pd.concat(meta_parts, ignore_index=True).drop_duplicates(["game_id","play_id","target_nfl_id"])

sort_cols = [c for c in ["game_id","play_id","frame_id","nfl_id"] if c in targets_long.columns]
targets_long = targets_long.sort_values(sort_cols).reset_index(drop=True)
targets_meta = targets_meta.sort_values(["game_id","play_id"]).reset_index(drop=True)

out_meta = OUTDIR / "targets_meta.parquet"
out_long = OUTDIR / "targets_long.parquet"
targets_meta.to_parquet(out_meta, index=False)
targets_long.to_parquet(out_long, index=False)

print("\n[SUMMARY] Build complete")
print(" meta rows:", len(targets_meta), "| plays:", targets_meta[['game_id','play_id']].drop_duplicates().shape[0])
print(" long rows:", len(targets_long))
print(" defenders per play (median):",
      int(targets_long[targets_long['player_side'].eq('Defense')].groupby(['game_id','play_id'])['nfl_id'].nunique().median()))
print(" frame span check (target rows only):")
chk = targets_long[targets_long["is_target"]].groupby(["game_id","play_id"])["frame_id"].agg(["min","max"]).reset_index()
mrg = chk.merge(targets_meta[["game_id","play_id","t2"]], on=["game_id","play_id"], how="left")
print("  min==1:", bool((mrg["min"]==1).all()), "| max==t2:", bool((mrg["max"]==mrg["t2"]).all()))
print(f" [WRITE] {out_meta}")
print(f" [WRITE] {out_long}")


# =========================
# 1) DATA LOAD & PREVIEW
# =========================

from pathlib import Path
import pandas as pd

# Load interim datasets
meta_path = Path("/kaggle/working/data/interim/targets_meta.parquet")
long_path = Path("/kaggle/working/data/interim/targets_long.parquet")

meta = pd.read_parquet(meta_path)
long = pd.read_parquet(long_path)

print(f"[INFO] meta rows: {len(meta):,}")
print(f"[INFO] long rows: {len(long):,}")

# Quick column peek
print("\n[INFO] META columns:")
print(list(meta.columns)[:12], "...")

print("\n[INFO] LONG columns:")
print(list(long.columns)[:12], "...")

# Sanity: sample a few plays
print("\n[SAMPLE META ROWS]")
display(meta.sample(3, random_state=1))

# Verify a few unique route names
print("\n[UNIQUE ROUTE COUNT]")
print(meta['route_of_targeted_receiver'].nunique())
print(sorted(meta['route_of_targeted_receiver'].dropna().unique())[:10])



# =========================
# 2) ROUTE FAMILIES + PLAY DIRECTION BACKFILL
# =========================

from pathlib import Path
import pandas as pd
import numpy as np
import glob

# ---- 2A. Normalize route names (use in-memory `meta` from Section 1) ----
meta_path_base = Path("/kaggle/working/data/interim/targets_meta.parquet")
long_path      = Path("/kaggle/working/data/interim/targets_long.parquet")
meta_path_fam  = Path("/kaggle/working/data/interim/targets_meta_families.parquet")

assert 'meta' in globals(), "[ERROR] `meta` not found (run Section 1 first)"
assert 'long' in globals(), "[ERROR] `long` not found (run Section 1 first)"

meta["route_clean"] = (
    meta["route_of_targeted_receiver"]
    .astype(str).str.strip().str.upper()
)

print(f"[INFO] Unique route names (pre-mapping): {meta['route_clean'].nunique()}")
print(sorted(meta["route_clean"].unique()))

# ---- 2B. Define route → family mapping ----
route_family_map = {
    # VERTICAL family
    "GO": "VERTICAL", "POST": "VERTICAL", "WHEEL": "VERTICAL", "CORNER": "VERTICAL",
    # BREAKER family
    "IN": "BREAKER", "OUT": "BREAKER", "HITCH": "BREAKER", "SLANT": "BREAKER",
    # CROSSER family
    "CROSS": "CROSSER", "ANGLE": "CROSSER",
    # SHORT family
    "SCREEN": "SHORT", "FLAT": "SHORT",
    # If the route is missing/odd, keep UNKNOWN for now (fine for EDA)
    "NONE": "UNKNOWN",
}

meta["route_family"] = meta["route_clean"].map(route_family_map).fillna("UNKNOWN")

# ---- 2C. Quick distribution summary ----
family_counts = meta["route_family"].value_counts()
print("\n[INFO] Route family counts:")
print(family_counts)

print("\n[INFO] Route → family mapping sample:")
print(
    meta[["route_clean", "route_family"]]
    .drop_duplicates()
    .sort_values("route_clean")
    .reset_index(drop=True)
)

# ---- 2D. Save enriched META with families ----
meta.to_parquet(meta_path_fam, index=False)
print(f"\n[WRITE] Saved enriched file: {meta_path_fam}")

# =========================
# 2E) Backfill play_direction into META & LONG from raw inputs
# =========================

raw_glob = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w*.csv"
files = sorted(glob.glob(raw_glob))
assert len(files) > 0, f"[ERROR] No raw input files found at pattern: {raw_glob}"
print(f"[INFO] Found {len(files)} raw input files")

# Load only required columns to keep it light
usecols = ["game_id", "play_id", "play_direction"]
dfs = []
for f in files:
    df = pd.read_csv(f, usecols=usecols)
    dfs.append(df)
raw_in = pd.concat(dfs, ignore_index=True)

# Defensive checks
cols = set(raw_in.columns)
assert "play_direction" in cols and "game_id" in cols and "play_id" in cols, \
    f"[ERROR] Missing expected columns in raw inputs. Got: {sorted(cols)}"

# Clean / normalize
raw_in["play_direction"] = (
    raw_in["play_direction"]
    .astype(str).str.strip().str.lower()
    .map({"left": "left", "right": "right"})
)

# Ensure each (game_id, play_id) has a single consistent value
map_df = (
    raw_in.dropna(subset=["play_direction"])
          .drop_duplicates(subset=["game_id", "play_id", "play_direction"])
)

dupes = (map_df.groupby(["game_id", "play_id"])["play_direction"]
               .nunique().reset_index(name="n"))
if (dupes["n"] > 1).any():
    bad = dupes[dupes["n"] > 1]
    raise ValueError(f"[ERROR] Inconsistent play_direction for some plays:\n{bad.head()}")

# Collapse to unique mapping
play_dir_map = map_df.drop_duplicates(subset=["game_id", "play_id"])[
    ["game_id", "play_id", "play_direction"]
]
print(f"[INFO] Unique (game_id, play_id) with play_direction: {len(play_dir_map):,}")

# --- Merge mapping into META_FAMILIES & LONG (read from disk, write back) ---
META = pd.read_parquet(meta_path_fam)
LONG = pd.read_parquet(long_path)

META = META.drop(columns=["play_direction"], errors="ignore").merge(
    play_dir_map, on=["game_id", "play_id"], how="left"
)
LONG = LONG.drop(columns=["play_direction"], errors="ignore").merge(
    play_dir_map, on=["game_id", "play_id"], how="left"
)

# Ensure the column exists even if mapping were empty (avoids KeyError later)
if "play_direction" not in META.columns:
    META["play_direction"] = pd.Series(index=META.index, dtype="object")
if "play_direction" not in LONG.columns:
    LONG["play_direction"] = pd.Series(index=LONG.index, dtype="object")

# Report coverage
missing_meta = META["play_direction"].isna().sum()
missing_long = LONG["play_direction"].isna().sum()
print(f"[COVERAGE] META rows missing play_direction: {missing_meta:,} / {len(META):,}")
print(f"[COVERAGE] LONG rows missing play_direction: {missing_long:,} / {len(LONG):,}")
if missing_meta > 0 or missing_long > 0:
    print("[WARN] Some plays are missing play_direction after merge (likely absent in raw inputs).")

# Persist enriched files (overwrite) + keep in-memory refs in sync
META.to_parquet(meta_path_fam, index=False)
LONG.to_parquet(long_path, index=False)
meta, long = META, LONG
print(f"[WRITE] Updated: {meta_path_fam}")
print(f"[WRITE] Updated: {long_path}")



# =========================
# 3) CHAOS EVALUATION (WR + DEFENDER-INFORMED)
# =========================

import numpy as np
import pandas as pd

# ---- 3A. Quick sanity checks on structures ----
print("\n[INFO] META columns:")
print(list(meta.columns))

print("\n[INFO] LONG columns:")
print(list(long.columns))

print("\n[INFO] player_side values (for identifying defenders):")
print(long["player_side"].value_counts().head())

print("\n[INFO] is_target values (for targeted WR check):")
print(long["is_target"].value_counts().head())

# Adjust this if your defense tag is named differently in player_side
DEFENSE_TAG = "defense"  # change to "away"/"home"/etc. if needed
OFF_POSITIONS = ["QB", "RB", "HB", "FB", "WR", "TE", "C", "G", "T", "OG", "OT", "OC"]

# Close-contact chaos hyperparameters
CONTACT_RADIUS = 1.5       # yards (or tracking units ~yards)
MIN_CONTACT_FRAMES = 3     # minimum frames in close contact to flag chaos

# ---- 3B. Helper: WR-only kinematic chaos in t0→t2 window ----

def compute_wr_kinematic_chaos(window: pd.DataFrame) -> bool:
    """
    window: LONG rows for the targeted WR, restricted to t0→t2 and sorted by frame_id.
    Uses only x, y, frame_id to derive heading/speed/accel.
    """
    if len(window) < 3:
        return False

    dx = window["x"].diff()
    dy = window["y"].diff()

    # Heading of movement (degrees)
    heading = np.degrees(np.arctan2(dy, dx))
    d_heading = heading.diff().abs()

    # Speed and acceleration (per frame)
    speed = np.sqrt(dx**2 + dy**2)
    accel = speed.diff()

    # Simple, tunable thresholds
    HEAD_THR_DEG = 25.0   # big turn
    ACC_THR      = 1.0    # burst / decel

    cond_heading = d_heading.max(skipna=True) > HEAD_THR_DEG
    cond_accel   = accel.abs().max(skipna=True) > ACC_THR

    return bool(cond_heading or cond_accel)


# ---- 3C. Helper: defender-induced chaos via nearest-defender closing ----

def compute_defender_chaos(window_wr: pd.DataFrame,
                           df_def: pd.DataFrame,
                           debug: bool = False,
                           label: str = "") -> bool:
    """
    window_wr: targeted WR rows in t0→t2 (sorted by frame_id).
    df_def: all defender rows for that play (any frames).
    debug: if True, print intermediate info for inspection.
    label: optional identifier for the play (e.g., "game-play").
    """
    if debug:
        print("\n[DEBUG] compute_defender_chaos called", f"({label})" if label else "")
        print(f"  window_wr.shape = {window_wr.shape}")
        print(f"  df_def.shape    = {df_def.shape}")

    if window_wr.empty:
        if debug:
            print("  -> window_wr is empty, returning False")
        return False

    if df_def.empty:
        if debug:
            print("  -> df_def is empty (no defenders?), returning False")
        return False

    records = []
    wr_frames = window_wr["frame_id"].unique()
    def_frames = df_def["frame_id"].unique()

    if debug:
        print(f"  unique WR frames in window: {wr_frames[:10]}{'...' if len(wr_frames) > 10 else ''}")
        print(f"  unique DEF frames total:    {def_frames[:10]}{'...' if len(def_frames) > 10 else ''}")
        common_frames = np.intersect1d(wr_frames, def_frames)
        print(f"  common frames WR ∩ DEF:     {common_frames[:10]}{'...' if len(common_frames) > 10 else ''}")
        print(f"  count common frames:        {len(common_frames)}")

    for _, wr_row in window_wr.iterrows():
        f = wr_row["frame_id"]
        df_def_f = df_def[df_def["frame_id"] == f]
        if df_def_f.empty:
            continue

        dx = df_def_f["x"].to_numpy() - wr_row["x"]
        dy = df_def_f["y"].to_numpy() - wr_row["y"]
        dist = np.sqrt(dx**2 + dy**2).min()

        records.append({"frame_id": f, "dist": dist})

    if debug:
        print(f"  records collected (frame/dist): {len(records)}")
        if records:
            print("  first few records:", records[:5])

    if len(records) < 3:
        if debug:
            print("  -> fewer than 3 records, returning False")
        return False

    dist_df = pd.DataFrame(records).sort_values("frame_id")
    dist_df["closing_speed"] = -dist_df["dist"].diff()
    dist_df["closing_accel"] = dist_df["closing_speed"].diff()

    max_close_speed = dist_df["closing_speed"].max(skipna=True)
    max_close_accel = dist_df["closing_accel"].max(skipna=True)

    if debug:
        print("\n  dist_df head:")
        print(dist_df.head())
        print(f"  max_close_speed = {max_close_speed:.4f}")
        print(f"  max_close_accel = {max_close_accel:.4f}")

    # Simple thresholds (we may tune after seeing debug output)
    #CLOSE_SPEED_THR = 0.8
    #CLOSE_ACCEL_THR = 0.6

    CLOSE_SPEED_THR = 0.7
    CLOSE_ACCEL_THR = 0.5

    cond_speed = max_close_speed > CLOSE_SPEED_THR
    cond_accel = max_close_accel > CLOSE_ACCEL_THR

    if debug:
        print(f"  cond_speed (>{CLOSE_SPEED_THR}) = {cond_speed}")
        print(f"  cond_accel (>{CLOSE_ACCEL_THR}) = {cond_accel}")
        print(f"  -> defender chaos = {cond_speed or cond_accel}")

    return bool(cond_speed or cond_accel)

# ---- 3D. Combined chaos detector (WR path + defender pressure) ----

def detect_chaos_play(df_play: pd.DataFrame, debug: bool = False, label: str = "") -> dict:
    """
    df_play: all LONG rows for a single (game_id, play_id).

    Returns:
      - chaos_flag      (bool): WR OR defender OR close-contact chaos
      - wr_chaos        (bool)
      - def_chaos       (bool)
    """
    df_wr = df_play[df_play["is_target"] == 1]
    if df_wr.empty:
        if debug:
            print(f"[DEBUG] No targeted WR rows for {label}, returning no chaos")
        return {
            "chaos_flag": False,
            "wr_chaos": False,
            "def_chaos": False,
        }

    t0 = df_wr["t0"].iloc[0]
    t2 = df_wr["t2"].iloc[0]

    window_wr = (
        df_wr[(df_wr["frame_id"] >= t0) & (df_wr["frame_id"] <= t2)]
        .sort_values("frame_id")
        .copy()
    )

    # All defenders for this play = non-offensive positions
    df_def = df_play[~df_play["player_position"].isin(OFF_POSITIONS)].copy()

    if debug:
        print(f"[DEBUG] Play {label}: window_wr={window_wr.shape}, df_def={df_def.shape}")

    wr_chaos       = compute_wr_kinematic_chaos(window_wr)
    def_chaos      = compute_defender_chaos(window_wr, df_def, debug=debug, label=label)

    chaos_flag = wr_chaos or def_chaos

    return {
        "chaos_flag": chaos_flag,
        "wr_chaos": wr_chaos,
        "def_chaos": def_chaos,
    }



# ---- 3E. Iterate over plays, tag chaos, and summarize by route family ----

results = []
n_plays = len(meta)
print(f"\n[INFO] Evaluating chaos on {n_plays:,} plays (targeted WR only)...")

for idx, row in meta.iterrows():
    gid = row["game_id"]
    pid = row["play_id"]
    fam = row["route_family"]

    df_play = long[(long["game_id"] == gid) & (long["play_id"] == pid)]

    chaos_info = detect_chaos_play(df_play)

    results.append({
        "game_id": gid,
        "play_id": pid,
        "route_family": fam,
        "chaos_flag": chaos_info["chaos_flag"],
        "wr_chaos": chaos_info["wr_chaos"],
        "def_chaos": chaos_info["def_chaos"],
    })


    # Light progress indicator every 1000 plays
    if (idx + 1) % 1000 == 0:
        print(f"[PROGRESS] Processed {idx + 1:,} / {n_plays:,} plays")

chaos_df = pd.DataFrame(results)
print("\n[INFO] Sample chaos_df rows:")
print(chaos_df.head())


# ---- 3F. Chaos rates by route family ----

print("\n[SUMMARY] % of plays with ANY chaos (WR OR defender) by route family:")
chaos_rate_any = (
    chaos_df.groupby("route_family")["chaos_flag"]
            .mean()
            .sort_values(ascending=False)
)
print(chaos_rate_any)

print("\n[SUMMARY] % of plays with WR-path-only chaos by route family:")
chaos_rate_wr = (
    chaos_df.groupby("route_family")["wr_chaos"]
            .mean()
            .sort_values(ascending=False)
)
print(chaos_rate_wr)

print("\n[SUMMARY] % of plays with DEFENDER-only chaos by route family:")
chaos_rate_def = (
    chaos_df.groupby("route_family")["def_chaos"]
            .mean()
            .sort_values(ascending=False)
)
print(chaos_rate_def)


chaos_df.to_parquet("/kaggle/working/data/interim/chaos_metrics.parquet", index=False)
chaos = pd.read_parquet("/kaggle/working/data/interim/chaos_metrics.parquet")


# --- Clean out any existing pass_result columns before merging ---
for col in chaos_df.columns:
    if col.startswith("pass_result"):
        chaos_df = chaos_df.drop(columns=[col])

# Also drop is_complete if you want a fully clean rebuild
if "is_complete" in chaos_df.columns:
    chaos_df = chaos_df.drop(columns=["is_complete"])

# --- Now safely merge pass_result from META ---
chaos_df = chaos_df.merge(
    meta[["game_id", "play_id", "pass_result"]],
    on=["game_id", "play_id"],
    how="left",
    validate="one_to_one"
)

# --- Recompute is_complete ---
chaos_df["is_complete"] = chaos_df["pass_result"].eq("C")



def classify_chaos(row):
    if row["wr_chaos"] and row["def_chaos"]:
        return "both"
    elif row["wr_chaos"]:
        return "wr_only"
    elif row["def_chaos"]:
        return "def_only"
    else:
        return "none"

if "chaos_type" not in chaos_df.columns:
    chaos_df["chaos_type"] = chaos_df.apply(classify_chaos, axis=1)



# Remove BOTH-chaos cases
chaos_df_simple = chaos_df[chaos_df["chaos_type"] != "both"].copy()

# Compute completion % + counts
family_cmp_counts = (
    chaos_df_simple
    .groupby(["route_family", "chaos_type"])["is_complete"]
    .agg(cmp_pct="mean", n="count")
    .unstack("chaos_type")
    .sort_index()
)

# Desired column order
ordered_cols = [
    ("cmp_pct", "none"),
    ("n",       "none"),
    ("cmp_pct", "wr_only"),
    ("n",       "wr_only"),
    ("cmp_pct", "def_only"),
    ("n",       "def_only"),
]

# Filter to only columns that exist (in case of missing categories in some datasets)
ordered_cols = [col for col in ordered_cols if col in family_cmp_counts.columns]

# Reorder table
family_cmp_counts = family_cmp_counts.loc[:, ordered_cols]

print("\n[SUMMARY] Completion % & counts (ordered cleanly):")
print(family_cmp_counts)



# --- STEP 2: SAFE MERGE FOR WR identity + completion results ---

wr_cols = ["game_id", "play_id", "player_name", "pass_result"]

# 1. Drop old versions if already merged (avoid _x / _y explosion)
for col in ["player_name", "pass_result"]:
    if col in chaos_df.columns:
        chaos_df = chaos_df.drop(columns=[col])

# 2. Perform clean merge (no duplicates)
chaos_df = chaos_df.merge(
    meta[wr_cols],
    on=["game_id", "play_id"],
    how="left"
)

# 3. Add completion boolean
if "is_complete" in chaos_df.columns:
    chaos_df = chaos_df.drop(columns=["is_complete"])

chaos_df["is_complete"] = chaos_df["pass_result"].eq("C")



# --- 3A. Helper to summarize a single WR's chaos profile ---

def summarize_wr(df):
    out = {}

    # Volume
    out["plays"] = len(df)

    # Chaos exposure rates
    out["pct_wr_only"]  = (df["chaos_type"] == "wr_only").mean()
    out["pct_def_only"] = (df["chaos_type"] == "def_only").mean()

    # Chaos counts
    out["count_none"]     = (df["chaos_type"] == "none").sum()
    out["count_wr_only"]  = (df["chaos_type"] == "wr_only").sum()
    out["count_def_only"] = (df["chaos_type"] == "def_only").sum()
    out["count_both"]     = (df["chaos_type"] == "both").sum()

    # Baseline: no-chaos completion
    df_none = df[df["chaos_type"] == "none"]
    out["cmp_none"] = df_none["is_complete"].mean() if len(df_none) else np.nan

    # WR-only chaos completion
    df_wr = df[df["chaos_type"] == "wr_only"]
    out["cmp_wr"] = df_wr["is_complete"].mean() if len(df_wr) else np.nan

    # DEF-only chaos completion
    df_def = df[df["chaos_type"] == "def_only"]
    out["cmp_def"] = df_def["is_complete"].mean() if len(df_def) else np.nan

    # Deltas vs clean baseline
    out["delta_wr"]  = out["cmp_wr"]  - out["cmp_none"] if pd.notna(out["cmp_wr"])  else np.nan
    out["delta_def"] = out["cmp_def"] - out["cmp_none"] if pd.notna(out["cmp_def"]) else np.nan

    return pd.Series(out)


# --- 3B. Apply per WR and sort by volume ---

wr_profiles = (
    chaos_df
    .groupby("player_name")
    .apply(summarize_wr, include_groups=False)
    .sort_values("plays", ascending=False)
)

# --- 3C. Reorder columns ---

desired_order = [
    "plays",
    "pct_wr_only",
    "count_wr_only",
    "cmp_wr",
    "pct_def_only",
    "count_def_only",
    "cmp_def",
    "cmp_none",
    "delta_wr",
    "delta_def",
    "count_none",
]

existing = [col for col in desired_order if col in wr_profiles.columns]
wr_profiles = wr_profiles[existing]

# --- 3D. FORMAT PERCENTAGE COLUMNS (NEW) ---

pct_cols = [
    "pct_wr_only",
    "pct_def_only",
    "cmp_wr",
    "cmp_def",
    "cmp_none",
    "delta_wr",
    "delta_def",
]

def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"    # convert 0.217 → 21.7%

for col in pct_cols:
    if col in wr_profiles.columns:
        wr_profiles[col] = wr_profiles[col].apply(fmt_pct)

# --- 3E. CAST COUNT COLUMNS TO INTEGERS (NEW) ---

count_cols = ["plays", "count_wr_only", "count_def_only", "count_none", "count_both"]

for col in count_cols:
    if col in wr_profiles.columns:
        wr_profiles[col] = wr_profiles[col].astype("Int64")  # nullable integer



wr_profiles.head(15)


# -----------------------------------------
# STEP B — WR RANKINGS BASED ON CHAOS METRICS
# -----------------------------------------

# Thresholds
MIN_WR_ONLY = 7
MIN_DEF_ONLY = 5
MIN_NONE = 50

# Helper: safely convert percentages back to numeric for sorting
def pct_to_float(s):
    if isinstance(s, str) and s.endswith("%"):
        try:
            return float(s.replace("%","")) / 100.0
        except ValueError:
            return np.nan
    return np.nan

# Start from wr_profiles, make a working copy
tmp = wr_profiles.copy()

# Numeric versions for sorting
tmp["cmp_none_num"]  = tmp["cmp_none"].apply(pct_to_float)
tmp["cmp_wr_num"]    = tmp["cmp_wr"].apply(pct_to_float)
tmp["cmp_def_num"]   = tmp["cmp_def"].apply(pct_to_float)
tmp["delta_wr_num"]  = tmp["delta_wr"].apply(pct_to_float)
tmp["delta_def_num"] = tmp["delta_def"].apply(pct_to_float)

# -----------------------------------------
# Add POSITION info (to filter out RBs)
# -----------------------------------------
# Build a mapping from player_name -> position using meta
pos_map = (
    meta[["player_name", "player_position"]]
    .drop_duplicates(subset="player_name")
    .set_index("player_name")["player_position"]
)

# tmp index is player_name
tmp["position"] = tmp.index.map(pos_map)

# -----------------------------------------
# 1) TOP WRs — WR-ONLY CHAOS PERFORMANCE
# -----------------------------------------
rank_wr_only = (
    tmp[tmp["count_wr_only"] >= MIN_WR_ONLY]
    .sort_values("delta_wr_num", ascending=False)
)

print("\n===============================")
print(" TOP WRs — WR-ONLY CHAOS (min 7 plays) ")
print("===============================")
print(
    rank_wr_only[
        ["plays", "count_wr_only", "pct_wr_only", "cmp_wr", "cmp_none", "delta_wr"]
    ].head(15)
)

# 1b) WORST WRs — WR-ONLY CHAOS PERFORMANCE
rank_wr_only_worst = (
    tmp[tmp["count_wr_only"] >= MIN_WR_ONLY]
    .sort_values("delta_wr_num", ascending=True)
)

print("\n===============================")
print(" WORST WRs — WR-ONLY CHAOS (min 7 plays) ")
print("===============================")
print(
    rank_wr_only_worst[
        ["plays", "count_wr_only", "pct_wr_only", "cmp_wr", "cmp_none", "delta_wr"]
    ].head(15)
)

# -----------------------------------------
# 2) TOP WRs — DEFENDER-ONLY CHAOS PERFORMANCE
# -----------------------------------------
rank_def_only = (
    tmp[tmp["count_def_only"] >= MIN_DEF_ONLY]
    .sort_values("delta_def_num", ascending=False)
)

print("\n===============================")
print(" TOP WRs — DEF-ONLY CHAOS (min 5 plays) ")
print("===============================")
print(
    rank_def_only[
        ["plays", "count_def_only", "pct_def_only", "cmp_def", "cmp_none", "delta_def"]
    ].head(15)
)

# 2b) WORST WRs — DEFENDER-ONLY CHAOS PERFORMANCE
rank_def_only_worst = (
    tmp[tmp["count_def_only"] >= MIN_DEF_ONLY]
    .sort_values("delta_def_num", ascending=True)
)

print("\n===============================")
print(" WORST WRs — DEF-ONLY CHAOS (min 5 plays) ")
print("===============================")
print(
    rank_def_only_worst[
        ["plays", "count_def_only", "pct_def_only", "cmp_def", "cmp_none", "delta_def"]
    ].head(15)
)

# -----------------------------------------
# 3) MOST CHAOS-EXPOSED WRs (volume only)
# -----------------------------------------
tmp["total_chaos"] = tmp["count_wr_only"] + tmp["count_def_only"]

rank_exposed = tmp.sort_values("total_chaos", ascending=False)

print("\n===============================")
print(" MOST CHAOS-EXPOSED WRs ")
print("===============================")
print(
    rank_exposed[
        ["plays", "count_wr_only", "count_def_only", "total_chaos", "pct_wr_only", "pct_def_only"]
    ].head(15)
)

# -----------------------------------------
# 4) TOP WRs IN CLEAN ENVIRONMENTS (baseline CMP_none)
#     Filter out RBs (RB catches are mostly uncontested)
# -----------------------------------------
rank_clean = (
    tmp[
        (tmp["count_none"] >= MIN_NONE) &
        (tmp["position"] != "RB")
    ]
    .sort_values("cmp_none_num", ascending=False)
)

print("\n===============================")
print(" TOP WRs — BEST CLEAN BASELINE (min 50 clean plays, non-RBs) ")
print("===============================")
print(
    rank_clean[
        ["plays", "count_none", "cmp_none", "position"]
    ].head(15)
)

# -----------------------------------------
# 5) CHAOS-INDEPENDENT WRs (good under both WR & DEF chaos)
#     Use sum of deltas as a simple index
# -----------------------------------------
tmp["chaos_independent"] = tmp["delta_wr_num"] + tmp["delta_def_num"]

rank_independent = (
    tmp[
        (tmp["count_wr_only"] >= MIN_WR_ONLY) &
        (tmp["count_def_only"] >= MIN_DEF_ONLY)
    ]
    .sort_values("chaos_independent", ascending=False)
)

print("\n===============================")
print(" CHAOS-INDEPENDENT WRs (min 7 WR chaos & 5 DEF chaos) ")
print("===============================")
print(
    rank_independent[
        ["plays", "count_wr_only", "count_def_only", "delta_wr", "delta_def"]
    ].head(15)
)

# 5b) CHAOS-SENSITIVE WRs (struggle under both types of chaos)
rank_sensitive = (
    tmp[
        (tmp["count_wr_only"] >= MIN_WR_ONLY) &
        (tmp["count_def_only"] >= MIN_DEF_ONLY)
    ]
    .sort_values("chaos_independent", ascending=True)
)

print("\n===============================")
print(" CHAOS-SENSITIVE WRs (min 7 WR chaos & 5 DEF chaos) ")
print("===============================")
print(
    rank_sensitive[
        ["plays", "count_wr_only", "count_def_only", "delta_wr", "delta_def"]
    ].head(15)
)



import pandas as pd

# --- 1A. Ensure chaos_plays exists and is clean (idempotent) ---

required_cols = {"route_of_targeted_receiver", "route_family", "player_name", "pass_result", "is_complete"}
missing_cols = [c for c in required_cols if c not in chaos_df.columns]

if missing_cols:
    chaos_plays = chaos_df.merge(
        meta[["game_id", "play_id"] + missing_cols],
        on=["game_id", "play_id"],
        how="left",
        validate="many_to_one"
    )
else:
    chaos_plays = chaos_df.copy()

print(f"[INFO] chaos_plays columns: {sorted(chaos_plays.columns.tolist())}")


# --- 1B. Lookup basic play info (matchup, WR, route, quarter, clock) ---

def lookup_play_info(game_id, play_id):
    row = meta[(meta["game_id"] == game_id) & (meta["play_id"] == play_id)]
    if row.empty:
        print("No matching play found in meta.")
        return None

    row = row.iloc[0]

    # quarter / clock from LONG
    df_long_play = long[(long["game_id"] == game_id) & (long["play_id"] == play_id)]
    if df_long_play.empty:
        qtr = None
        clock = None
    else:
        snap_row = df_long_play.sort_values("frame_id").iloc[0]
        qtr = snap_row.get("quarter", None)
        clock = snap_row.get("game_clock", None)

    info = {
        "season": row["season"],
        "week": row["week"],
        "offense": row["possession_team"],
        "defense": row["defensive_team"],
        "matchup": f"{row['possession_team']} vs {row['defensive_team']}",
        "player_name": row["player_name"],
        "route_family": row["route_family"],
        "route": row["route_of_targeted_receiver"],
        "quarter": qtr,
        "game_clock": clock,
    }
    return info


# --- 1C. Find first chaos frame (using your existing detect_chaos_play) ---

def get_chaos_timing(game_id, play_id):
    """
    Returns (chaos_frame_index, total_frames) for a given play,
    where chaos_frame_index is 1-based. If no chaos, returns (None, total_frames).
    """
    df_play = (
        long[(long["game_id"] == game_id) & (long["play_id"] == play_id)]
        .sort_values("frame_id")
    )
    if df_play.empty:
        print("[WARN] No tracking rows for this play in LONG.")
        return None, 0

    frames = df_play["frame_id"].unique()
    total_frames = len(frames)

    chaos_frame_idx = None
    for i, fid in enumerate(frames, start=1):
        df_sub = df_play[df_play["frame_id"] <= fid]
        info = detect_chaos_play(df_sub)
        if info.get("chaos_flag", False):
            chaos_frame_idx = i
            break

    return chaos_frame_idx, total_frames


# --- 1D. Pretty printer combining both ---

def print_play_with_chaos_timing(game_id, play_id, label=None):
    info = lookup_play_info(game_id, play_id)
    if info is None:
        return

    chaos_idx, total = get_chaos_timing(game_id, play_id)

    if label:
        print(f"\n=== {label} ===")
    else:
        print("\n=== Play Info ===")

    print(f"Week {info['week']} • {info['season']} — {info['matchup']}")
    print(f"Quarter: {info['quarter']}   Clock: {info['game_clock']}")
    print(f"Target WR: {info['player_name']}")
    print(f"Route: {info['route_family']} ({info['route']})")
    print(f"Game: {game_id}")
    print(f"Play: {play_id}")

    if chaos_idx is None:
        print("Chaos frame: None (no chaos detected for this play)")
    else:
        print(f"Chaos frame: {chaos_idx}/{total} "
              f"({chaos_idx/total:.1%} of the way through)")



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# ------------------------
# Field drawing (updated colors)
# ------------------------
def draw_field(ax):
    """Basic NFL field for Big Data Bowl coordinates."""
    ax.set_facecolor("#154734")  # dark green
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_xticks([])
    ax.set_yticks([])

    # Yard lines every 10 yards (low alpha)
    for x in range(10, 111, 10):
        ax.axvline(x, color="white", lw=1.0, alpha=0.1)

    # Sidelines/boundaries
    ax.axvline(0, color="white", lw=2)
    ax.axvline(120, color="white", lw=2)
    ax.axhline(0, color="white", lw=2)
    ax.axhline(53.3, color="white", lw=2)


def get_play_tracking(game_id, play_id):
    df_play = ( long[(long["game_id"] == game_id) & (long["play_id"] == play_id)]
        .sort_values("frame_id") .copy() )
    
    if df_play.empty: 
        print(f"[ERROR] No tracking rows for game_id={game_id}, play_id={play_id}") 
    else: print(f"[INFO] Frames: {df_play['frame_id'].nunique()} Rows: {len(df_play):,}") 
        
    return df_play

def build_label_block_from_chaos(game_id, play_id, chaos_df):
    """
    Build the label_block dict for make_chaos_animation.

    Stats logic:
      - Baseline: completion rate for this route_family with NO chaos (all WRs).
      - Chaos:    completion rate for this route_family with THIS chaos type,
                  but only for THIS WR (player_name).

    Expects chaos_df to have at least:
        game_id, play_id,
        wr_chaos, def_chaos,
        pass_result OR is_complete,
        route_family, player_name
    """
    # Locate this play in chaos_df
    df = chaos_df.copy()
    row = df[(df["game_id"] == game_id) & (df["play_id"] == play_id)]
    if row.empty:
        print(f"[WARN] build_label_block_from_chaos: play {game_id}-{play_id} not found in chaos_df.")
        return None

    row = row.iloc[0]

    # Ensure we have an is_complete flag (1 = completion, 0 = otherwise)
    if "is_complete" not in df.columns:
        df["is_complete"] = (df["pass_result"] == "C")

    fam = row.get("route_family", None)
    wr_name = row.get("player_name", None)

    # Determine chaos type for THIS play
    wr_c = bool(row.get("wr_chaos", False))
    def_c = bool(row.get("def_chaos", False))

    if wr_c and not def_c:
        chaos_label_name = "WR-Path Chaos"
        chaos_type = "wr"
    elif def_c and not wr_c:
        chaos_label_name = "Defender Chaos"
        chaos_type = "def"
    elif wr_c and def_c:
        chaos_label_name = "WR + Defender Chaos"
        chaos_type = "both"
    else:
        chaos_label_name = "Chaos"
        chaos_type = "none"

    # -----------------
    # Baseline group: same route family, *no chaos* (all WRs)
    # -----------------
    baseline_mask = (
        (df["route_family"] == fam) &
        (df["wr_chaos"] == False) &
        (df["def_chaos"] == False)
    )

    # -----------------
    # Chaos group: same route family, this chaos type, THIS WR (if name available)
    # -----------------
    if chaos_type == "wr":
        chaos_mask = (
            #(df["route_family"] == fam) &
            (df["wr_chaos"] == True) &
            (df["def_chaos"] == False)
        )
    elif chaos_type == "def":
        chaos_mask = (
            (df["route_family"] == fam) &
            (df["wr_chaos"] == False) &
            (df["def_chaos"] == True)
        )
    elif chaos_type == "both":
        chaos_mask = (
            (df["route_family"] == fam) &
            (df["wr_chaos"] == True) &
            (df["def_chaos"] == True)
        )
    else:
        # no chaos on this play → compare to all chaotic plays in this family
        chaos_mask = (
            (df["route_family"] == fam) &
            ((df["wr_chaos"] == True) | (df["def_chaos"] == True))
        )

    # If we know the WR, restrict chaos_mask to this player so chaos_pct is WR-specific
    if wr_name is not None and "player_name" in df.columns:
        chaos_mask = chaos_mask & (df["player_name"] == wr_name)

    baseline_group = df[baseline_mask]
    chaos_group = df[chaos_mask]

    baseline_pct = np.nan
    chaos_pct = np.nan

    if not baseline_group.empty:
        baseline_pct = baseline_group["is_complete"].mean()
    if not chaos_group.empty:
        chaos_pct = chaos_group["is_complete"].mean()

    return {
        "baseline_pct": baseline_pct,
        "chaos_pct": chaos_pct,
        "chaos_label_name": chaos_label_name,
    }


def make_chaos_animation(
    game_id,
    play_id,
    save_path="chaos_play.mp4",
    fps=10,
    chaos_df=None,
    label_block=None,
):
    """
    Animate a single Big Data Bowl chaos play.

    Inputs:
        game_id, play_id : keys for the play
        save_path        : MP4 output path (uses FFMpegWriter)
        fps              : frames per second for video
        chaos_df         : full chaos dataframe (one row per play) for label stats
        label_block      : optional dict overriding stats in bottom-left block.
                           If None, will use build_label_block_from_chaos(game_id, play_id, chaos_df)
    """
    # 1) Grab tracking for this play
    df_play = get_play_tracking(game_id, play_id)
    if df_play.empty:
        print("[WARN] No tracking for this play, aborting.")
        return

    df_play = df_play.sort_values("frame_id").copy()
    frames = df_play["frame_id"].unique().tolist()

    # 2) Precompute chaos state per frame
    chaos_states = []
    for fid in frames:
        df_sub = df_play[df_play["frame_id"] <= fid]
        info = detect_chaos_play(df_sub) or {}
        chaos_states.append({
            "chaos_flag": bool(info.get("chaos_flag", False)),
            "wr_chaos": bool(info.get("wr_chaos", False)),
            "def_chaos": bool(info.get("def_chaos", False)),
        })

    chaos_start_idx = None
    for i, cs in enumerate(chaos_states):
        if cs["chaos_flag"]:
            chaos_start_idx = i
            break

    frame_groups = {fid: g for fid, g in df_play.groupby("frame_id")}

    # --- Use df_play (long) for QB + ball info ---
    cols = df_play.columns

    t0 = int(df_play["t0"].iloc[0]) if "t0" in cols else 1
    t2 = int(df_play["t2"].iloc[0]) if "t2" in cols else frames[-1]

    if "ball_land_x" in cols and "ball_land_y" in cols:
        ball_land_x = float(df_play["ball_land_x"].iloc[0])
        ball_land_y = float(df_play["ball_land_y"].iloc[0])
    else:
        ball_land_x = ball_land_y = None

    if "qb_x" in cols and "qb_y" in cols:
        qb_x_val = df_play["qb_x"].iloc[0]
        qb_y_val = df_play["qb_y"].iloc[0]
        qb_x = float(qb_x_val) if qb_x_val is not None and not np.isnan(qb_x_val) else None
        qb_y = float(qb_y_val) if qb_y_val is not None and not np.isnan(qb_y_val) else None
    else:
        qb_x = qb_y = None

    # 3) Play metadata (for title + basic labels)
    info = lookup_play_info(game_id, play_id) or {}
    title_base = f"Week {info.get('week', '?')} • {info.get('season', '?')} — {info.get('matchup', '')}"
    wr_name_meta = info.get("player_name", "")
    route_family_meta = info.get("route_family", "")
    route_meta = info.get("route", "")
    qtr = info.get("quarter", "")
    clock = info.get("game_clock", "")

    # 3b) Decide label_block to use
    if label_block is not None:
        label_block_effective = label_block
    elif chaos_df is not None:
        label_block_effective = build_label_block_from_chaos(game_id, play_id, chaos_df)
    else:
        label_block_effective = None

    # 4) Figure + artists
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax)

    scat_players = ax.scatter([], [], s=60, edgecolors="black", alpha=0.9)
    scat_ball = ax.scatter([], [], s=90, color="saddlebrown", edgecolors="white", linewidths=1.2)
    scat_wr = ax.scatter([], [], s=80, color="cyan", edgecolors="black", linewidths=1.2)

    halo = Circle((0, 0), radius=1.0, edgecolor="#F5A623",
                  facecolor="none", linewidth=2.25, alpha=0.65, visible=False)
    ax.add_patch(halo)

    def_line = Line2D([], [], color="gold", linewidth=2.0, alpha=0.9)
    def_line.set_visible(False)
    ax.add_line(def_line)

    ball_trail = Line2D([], [], color="white", linewidth=1.5, alpha=0.9)
    ax.add_line(ball_trail)

    # WR trail (last N frames)
    wr_trail = Line2D([], [], color="cyan", linewidth=2.0, alpha=0.6)
    ax.add_line(wr_trail)

    # DB trails (faint gray)
    db_trail = Line2D([], [], color="lightgray", linewidth=1.0, alpha=0.2)
    ax.add_line(db_trail)


    subtitle_text = ax.text(
        0.01, 0.97, "",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        ha="left",
        va="top",
    )

    bottom_text = ax.text(
        0.01, 0.06, "",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.4, edgecolor="none"),
    )

    fig.suptitle(title_base, fontsize=14, color="white", y=0.95)

    def _fmt_pct(p):
        if p is None or np.isnan(p):
            return "NA"
        val = float(p)
        if val <= 1.0:
            val *= 100.0
        return f"{val:.0f}%"

    def init():
        scat_players.set_offsets(np.empty((0, 2)))
        scat_ball.set_offsets(np.empty((0, 2)))
        scat_wr.set_offsets(np.empty((0, 2)))
        halo.set_visible(False)
        def_line.set_visible(False)
        ball_trail.set_data([], [])
        wr_trail.set_data([], [])
        db_trail.set_data([], [])

        subtitle_text.set_text(f"Q{qtr}  {clock}")

        # ---- Bottom-left label block ----
        if label_block_effective is not None:
            baseline = label_block_effective.get("baseline_pct", None)
            chaos_val = label_block_effective.get("chaos_pct", None)
            chaos_label_name = label_block_effective.get("chaos_label_name", "Chaos")

            name = wr_name_meta
            fam = route_family_meta
            route = route_meta

            if fam and route:
                line1 = f"{name} • {fam} ({route})"
            elif fam:
                line1 = f"{name} • {fam}"
            else:
                line1 = f"{name}"

            if baseline is not None and chaos_val is not None:
                delta = None
                try:
                    b_raw = float(baseline)
                    c_raw = float(chaos_val)
                    if b_raw <= 1.0 and c_raw <= 1.0:
                        delta = (c_raw - b_raw) * 100.0
                    else:
                        delta = c_raw - b_raw
                except Exception:
                    delta = None

                line2 = f"Baseline: {_fmt_pct(baseline)}   {chaos_label_name}: {_fmt_pct(chaos_val)}"

                if delta is not None:
                    sign = "+" if delta >= 0 else "–"
                    line3 = f"({sign}{abs(delta):.0f}%)"
                    bottom_text.set_text(line1 + "\n" + line2 + "\n" + line3)
                else:
                    bottom_text.set_text(line1 + "\n" + line2)
            else:
                bottom_text.set_text(line1)
        else:
            bottom_text.set_text(f"Target WR: {wr_name_meta} — {route_family_meta} ({route_meta})")

        #return scat_players, scat_ball, scat_wr, halo, def_line, ball_trail, subtitle_text, bottom_text
        return (
            scat_players, scat_ball, scat_wr,
            halo, def_line, ball_trail,
            wr_trail, db_trail,
            subtitle_text, bottom_text
        )


    def update(frame_idx):
        fid = frames[frame_idx]
        df_f = frame_groups[fid]
        cs = chaos_states[frame_idx]

        others = df_f

        if "is_target" in df_f.columns:
            wr_f = df_f[df_f["is_target"] == True]
        else:
            wr_f = df_f.iloc[0:0]

        if not others.empty:
            scat_players.set_offsets(others[["x", "y"]].to_numpy())
            if "player_side" in others.columns:
                sides = others["player_side"].astype(str).str.lower()
                facecolors = np.where(sides == "offense", "white", "lightgray")
                scat_players.set_facecolor(facecolors)
            else:
                scat_players.set_facecolor("white")
        else:
            scat_players.set_offsets(np.empty((0, 2)))
            scat_players.set_facecolor([])

        if not wr_f.empty:
            wr_xy = wr_f[["x", "y"]].to_numpy()
            scat_wr.set_offsets(wr_xy)
            wr_x, wr_y = wr_xy[0]
            halo.center = (wr_x, wr_y)
        else:
            scat_wr.set_offsets(np.empty((0, 2)))

        ball_valid = (
            qb_x is not None and qb_y is not None and
            ball_land_x is not None and ball_land_y is not None and
            t2 != t0
        )

        if ball_valid:
            alpha = (fid - t0) / (t2 - t0)
            alpha = float(np.clip(alpha, 0.0, 1.0))

            bx = qb_x + alpha * (ball_land_x - qb_x)
            by = qb_y + alpha * (ball_land_y - qb_y)

            scat_ball.set_offsets(np.array([[bx, by]]))
            ball_trail.set_data([qb_x, bx], [qb_y, by])
        else:
            if ball_land_x is not None:
                scat_ball.set_offsets(np.array([[ball_land_x, ball_land_y]]))
            else:
                scat_ball.set_offsets(np.empty((0, 2)))
            ball_trail.set_data([], [])

        scat_players.set_sizes([60] * len(others))
        if ball_valid or ball_land_x is not None:
            scat_ball.set_sizes([90])
        else:
            scat_ball.set_sizes([0])

        # Base WR size
        wr_sizes = [80] * len(wr_f)

        # WR acceleration pulse (only if we have 'a' for the WR)
        if not wr_f.empty and "a" in df_f.columns:
            try:
                wr_a = float(wr_f["a"].iloc[0])
                a_mag = abs(wr_a)
                # Threshold + scaling: tweak 2.5 and 40 as you like
                if a_mag > 2.5:
                    boost = min(a_mag * 10.0, 40.0)  # cap boost
                    wr_sizes = [80 + boost] * len(wr_f)
            except Exception:
                pass

        scat_wr.set_sizes(wr_sizes)


        # -------------------------
        # WR TRAIL (last N frames)
        # -------------------------
        trail_len = 6  # frames to keep in trail (tune as desired)

        if "is_target" in df_play.columns:
            wr_hist = df_play[df_play["is_target"] == True]
        else:
            wr_hist = df_play.iloc[0:0]

        if not wr_hist.empty:
            wr_hist = wr_hist[wr_hist["frame_id"] <= fid].sort_values("frame_id")
            if len(wr_hist) > trail_len:
                wr_hist = wr_hist.iloc[-trail_len:]
            wr_trail.set_data(wr_hist["x"].values, wr_hist["y"].values)
        else:
            wr_trail.set_data([], [])

        # -------------------------
        # DB TRAILS (all defenders, faint)
        # -------------------------
        #if "player_side" in df_play.columns:
        #    db_hist = df_play[df_play["player_side"].str.lower() == "defense"]
        #else:
        #    db_hist = df_play.iloc[0:0]

        #if not db_hist.empty:
        #    db_hist = db_hist[db_hist["frame_id"] <= fid].sort_values(["nfl_id", "frame_id"])
        #    # Join all DB paths into one polyline – visually reads as "defensive flow"
        #    db_trail.set_data(db_hist["x"].values, db_hist["y"].values)
        #else:
        #    db_trail.set_data([], [])


        # -------------------------
        # DB TRAILS (all defenders, faint, no cross-connections)
        # -------------------------
        if "player_side" in df_play.columns:
            db_hist = df_play[df_play["player_side"].str.lower() == "defense"]
        else:
            db_hist = df_play.iloc[0:0]

        if not db_hist.empty:
            db_hist = db_hist[db_hist["frame_id"] <= fid].sort_values(["nfl_id", "frame_id"])

            xs = []
            ys = []
            for _, g in db_hist.groupby("nfl_id"):
                xs.extend(g["x"].values.tolist())
                ys.extend(g["y"].values.tolist())
                # NaN separator so matplotlib breaks the line between DBs
                xs.append(np.nan)
                ys.append(np.nan)

            db_trail.set_data(xs, ys)
        else:
            db_trail.set_data([], [])



        halo.set_visible(False)
        def_line.set_visible(False)

        if cs["chaos_flag"] and not wr_f.empty:
            if cs["wr_chaos"] and cs["def_chaos"]:
                halo.set_edgecolor("#F5A623")
                halo.set_linewidth(2.25)
                halo.set_radius(2.5)
            elif cs["wr_chaos"]:
                halo.set_edgecolor("cyan")
                halo.set_linewidth(2.25)
                halo.set_radius(2.2)
            elif cs["def_chaos"]:
                halo.set_edgecolor("#F5A623")
                halo.set_linewidth(2.25)
                halo.set_radius(2.0)

            halo.set_visible(True)

            if cs["def_chaos"]:
                if not wr_f.empty and "player_side" in df_f.columns:
                    wr_side = wr_f["player_side"].iloc[0]
                    def_candidates = others[others["player_side"] != wr_side] if wr_side in ["offense", "defense"] else others
                else:
                    def_candidates = others

                if not def_candidates.empty:
                    wr_x, wr_y = wr_f[["x", "y"]].iloc[0]
                    dx = def_candidates["x"].values - wr_x
                    dy = def_candidates["y"].values - wr_y
                    d2 = dx * dx + dy * dy
                    j = int(np.argmin(d2))
                    db_x, db_y = def_candidates[["x", "y"]].iloc[j]
                    def_line.set_data([db_x, wr_x], [db_y, wr_y])
                    def_line.set_visible(True)

        if chaos_start_idx is not None:
            if frame_idx < chaos_start_idx:
                chaos_str = "Chaos: pending"
            elif frame_idx == chaos_start_idx:
                chaos_str = "Chaos: START"
            else:
                chaos_str = "Chaos: active"
        else:
            chaos_str = "Chaos: none"

        subtitle_text.set_text(
            f"Q{qtr}  {clock}    Frame {frame_idx+1}/{len(frames)}    {chaos_str}"
        )

        return scat_players, scat_ball, scat_wr, halo, def_line, ball_trail, subtitle_text, bottom_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        blit=True,
        interval=1000 / fps,
    )

    if save_path is not None:
        writer = FFMpegWriter(fps=fps)
        ani.save(save_path, writer=writer)
        plt.close(fig)
        print(f"[INFO] Saved animation to {save_path}")

    return ani



# Identify plays of interest for animation
g1 = 2023111209
p1 = 2679

g2 = 2023111209
p2 = 1830

g3 = 2023112607
p3 = 2596

g4 = 2023091400
p4 = 4107


from IPython.display import HTML

label = build_label_block_from_chaos(gid, pid, chaos_df)

ani = make_chaos_animation(g1, p1, save_path=None, fps=10, label_block=label)
HTML(ani.to_jshtml())


from IPython.display import HTML

label = build_label_block_from_chaos(gid, pid, chaos_df)

ani = make_chaos_animation(g2, p2, save_path=None, fps=10, label_block=label)
HTML(ani.to_jshtml())


from IPython.display import HTML

label = build_label_block_from_chaos(gid, pid, chaos_df)

ani = make_chaos_animation(g3, p3, save_path=None, fps=10, label_block=label)
HTML(ani.to_jshtml())


from IPython.display import HTML

label = build_label_block_from_chaos(gid, pid, chaos_df)

ani = make_chaos_animation(g4, p4, save_path=None, fps=10, label_block=label)
HTML(ani.to_jshtml())




