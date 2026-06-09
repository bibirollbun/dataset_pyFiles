from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path("/kaggle/working")  # writable and preserved
TEMP_DIR = Path("/kaggle/temp")     # temporary session-only

DATASET_ROOT = Path("/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final")
DATA_DIR = DATASET_ROOT / "train"  # weekly input/output CSVs
SUPP_PATH = DATASET_ROOT / "supplementary_data.csv" # labels & context

# Output directories
ANALYS_DIR = BASE_DIR / "outputs" / "analysis"; ANALYS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = BASE_DIR / "outputs" / "figs"; FIG_DIR.mkdir(parents=True, exist_ok=True)

# Field geometry (yards)
FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3

# Frame cadence (seconds per frame in tracking)
FRAME_DT = 0.1

# Baseline kinematics / disruption params
PARAMS = dict(
    s_cap=9.5,              # max speed (yd/s)
    a_max=3.5,              # max accel (yd/s^2)
    turn_rate_deg=240,      # max turn rate (deg/s) for TTB / orientation
    hands_up_align_deg=40,  #  WR catch posture: alignment with ball bearing, degrees
    # Defender disruption radius & window
    disrupt_R=1.5,         # arm's-length reach (yd)  <-- main knob for disruption
    last_N_frames=12         # disruption window size (frames) before ball arrival, ~1.2s
)

# Colors
COLOR_OFFENSE = "#2a9d8f"
COLOR_DEFENSE = "#457b9d"
COLOR_BALL    = "#8b0000"
COLOR_TR      = "#000000"  # targeted receiver
COLOR_DEF_HERO = "#ff7f0e" # highlight top defender


def normalize_orientation(df, play_direction):
    """
    Normalize so all offenses move left->right (mirror left-moving plays).
    We only flip x/y; orientation & dir stay in original coordinate frame,
    but relative geometry is mirrored.
    """
    out = df.copy()
    if str(play_direction).lower().startswith("left"):
        out["x"] = FIELD_X_MAX - out["x"]
        out["y"] = FIELD_Y_MAX - out["y"]
    return out

def bearing_deg(x0, y0, x1, y1):
    """Compute bearing (degrees) from (x0,y0) -> (x1,y1)."""
    return (math.degrees(math.atan2(y1 - y0, x1 - x0)) + 360) % 360

def ang_diff_deg(a, b):
    """Smallest absolute angular difference in degrees."""
    d = (a - b + 180) % 360 - 180
    return abs(d)


import re

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _parse_week_info(name: str):
    """
    Extract season and week from file name pattern like 'output_2023_w01'.
    Returns (season:int, week:int). If not found, returns (None, None).
    """
    m = re.search(r'_(\d{4})_w(\d{1,2})', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def _add_week_meta(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Add source_name, season, and week columns based on the CSV stem.
    This is what makes 'season' and 'week' available later.
    """
    season, week = _parse_week_info(name)
    df = df.copy()
    df["source_name"] = name
    df["season"] = season
    df["week"] = week
    return df

def load_play_frames(input_data: dict, output_data: dict):
    """Merge weekly input/output dicts into two frames."""
    # INPUTS
    inp_frames = []
    for name, df in sorted(input_data.items()):
        df1 = _strip_cols(df)
        df1 = _add_week_meta(df1, name)
        inp_frames.append(df1)
    if not inp_frames:
        raise ValueError("No input_* DataFrames found in input_data.")
    inp_all = pd.concat(inp_frames, ignore_index=True)

    # OUTPUTS
    out_frames = []
    for name, df in sorted(output_data.items()):
        df1 = _strip_cols(df)
        df1 = _add_week_meta(df1, name)
        out_frames.append(df1)
    if not out_frames:
        raise ValueError("No output_* DataFrames found in output_data.")
    out_all = pd.concat(out_frames, ignore_index=True)

    # Minimal validation
    required_out = {"game_id","play_id","frame_id","nfl_id","x","y"}
    miss_out = required_out - set(out_all.columns)
    if miss_out:
        raise ValueError(f"Output missing required columns: {sorted(miss_out)}")

    required_in = {"game_id","play_id","frame_id","nfl_id",
                   "player_side","player_role","play_direction",
                   "ball_land_x","ball_land_y","num_frames_output"}
    miss_in = required_in - set(inp_all.columns)
    if miss_in:
        raise ValueError(f"Input missing required columns: {sorted(miss_in)}")

    return inp_all, out_all

# Build weekly dicts, then load
input_data, output_data = {}, {}
for p in sorted(DATA_DIR.rglob("*.csv")):
    stem = p.stem.lower()
    if stem.startswith("input_"):
        input_data[stem]  = pd.read_csv(p)
    elif stem.startswith("output_"):
        output_data[stem] = pd.read_csv(p)

print("Weeks loaded -> input:", len(input_data), "output:", len(output_data))
inp, out = load_play_frames(input_data, output_data)
print("Shapes -> inp:", inp.shape, "out:", out.shape)
print("Columns on out:", [c for c in out.columns if c in ["season","week"]])



def compute_kinematics_from_output(out_frames: pd.DataFrame) -> pd.DataFrame:
    """
    Derive speed (s), accel (a), and motion direction (dir) from consecutive OUTPUT x,y.
    Orientation 'o' ≈ dir.
    """
    df = out_frames.sort_values(["game_id","play_id","nfl_id","frame_id"]).copy()
    nxt = df.groupby(["game_id","play_id","nfl_id"], sort=False)[["x","y"]].shift(-1)
    prv = df.groupby(["game_id","play_id","nfl_id"], sort=False)[["x","y"]].shift(+1)
    df["x_next"], df["y_next"] = nxt["x"], nxt["y"]
    df["x_prev"], df["y_prev"] = prv["x"], prv["y"]

    dx_f = (df["x_next"] - df["x"]) / FRAME_DT
    dy_f = (df["y_next"] - df["y"]) / FRAME_DT
    dx_b = (df["x"] - df["x_prev"]) / FRAME_DT
    dy_b = (df["y"] - df["y_prev"]) / FRAME_DT

    dx = dx_f.where(dx_f.notna() & dx_b.notna(), dx_f.fillna(dx_b))
    dy = dy_f.where(dy_f.notna() & dy_b.notna(), dy_f.fillna(dy_b))

    s = np.hypot(dx, dy).fillna(0.0)
    dir_deg = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    s_next = s.groupby([df["game_id"], df["play_id"], df["nfl_id"]]).shift(-1)
    a = ((s_next - s) / FRAME_DT).fillna(0.0)

    df["s"] = s
    df["a"] = a
    df["dir"] = dir_deg.fillna(0.0)
    df["o"] = df["dir"]
    return df.drop(columns=["x_next","y_next","x_prev","y_prev"])


def merge_roles_and_playconst_from_input(out_kin: pd.DataFrame, inp_frames: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-player roles (player_side, player_role, play_direction,
    player_name, player_position) and per-play constants (ball_land_x/y, num_frames_output)
    from INPUT into enriched OUTPUT.
    """
    role_cols = ["player_side","player_role","play_direction","player_name","player_position"]
    keep_player = ["game_id","play_id","nfl_id"] + [c for c in role_cols if c in inp_frames.columns]
    roles = (
        inp_frames.sort_values(["game_id","play_id","nfl_id","frame_id"])
        .groupby(["game_id","play_id","nfl_id"], as_index=False)
        .last()[keep_player]
    )

    play_cols = ["game_id","play_id","ball_land_x","ball_land_y","num_frames_output","season","week"]
    # season/week come from _add_week_meta via input files
    base_cols_present = [c for c in play_cols if c in inp_frames.columns]
    play_ctx = (
        inp_frames[base_cols_present]
        .drop_duplicates(subset=["game_id","play_id"], keep="last")
    )

    out1 = out_kin.merge(roles, on=["game_id","play_id","nfl_id"], how="left")
    out2 = out1.merge(play_ctx, on=["game_id","play_id"], how="left")

    need = ["player_side","player_role","play_direction","ball_land_x","ball_land_y","num_frames_output"]
    missing = [c for c in need if c not in out2.columns]
    if missing:
        raise ValueError(f"Missing after merge: {missing}")
    return out2

out = compute_kinematics_from_output(out)
out = merge_roles_and_playconst_from_input(out, inp)

print("Enriched 'out' has:", {c: (c in out.columns) for c in
      ["s","a","dir","o","player_side","player_role","play_direction","ball_land_x","ball_land_y","num_frames_output","player_name","player_position"]})
out_tracks = out.copy()


def load_supp(supp_path=SUPP_PATH):
    s = pd.read_csv(supp_path, low_memory=False)
    s.columns = [c.strip() for c in s.columns]
    need = {"game_id","play_id"}
    miss = need - set(s.columns)
    if miss: raise ValueError(f"Supplementary missing keys: {sorted(miss)}")
    return s

supp = load_supp()
print("supp rows:", len(supp), "cols:", len(supp.columns))

# Completion label helper
def get_labels_from_supp(supp_df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"Complete":1,"COMPLETE":1,"C":1,
               "Incomplete":0,"INCOMPLETE":0,"I":0,
               "Interception":0,"INTERCEPTION":0,"IN":0,"INT":0}
    lab = (supp_df[["game_id","play_id","pass_result"]]
           .dropna(subset=["pass_result"]).copy())
    lab["label_completion"] = lab["pass_result"].astype(str).map(mapping)
    lab = lab.dropna(subset=["label_completion"]).astype({"label_completion": int})
    lab = lab.drop_duplicates(subset=["game_id","play_id"], keep="last")
    return lab[["game_id","play_id","label_completion"]]



def get_throw_and_arrival_info(out_frames: pd.DataFrame):
    """
    Per-play summary from enriched OUTPUT:
      t0, t_end, play_direction, ball_land_x/y, num_frames_output
      and, if available, season/week.
    """
    def _first_nonnull(s):
        return s.dropna().iloc[0] if s.notna().any() else np.nan

    agg_dict = {
        "t0": ("frame_id","min"),
        "t_end": ("frame_id","max"),
        "play_direction": ("play_direction", _first_nonnull),
        "ball_land_x": ("ball_land_x", _first_nonnull),
        "ball_land_y": ("ball_land_y", _first_nonnull),
        "num_frames_output": ("num_frames_output", _first_nonnull),
    }

    # Only add season/week if they exist on out_frames
    if "season" in out_frames.columns:
        agg_dict["season"] = ("season","first")
    if "week" in out_frames.columns:
        agg_dict["week"] = ("week","first")

    plays = (
        out_frames.groupby(["game_id","play_id"])
        .agg(**agg_dict)
        .reset_index()
    )
    return plays

plays_tbl = get_throw_and_arrival_info(out)
print("plays_tbl shape:", plays_tbl.shape)
print("plays_tbl columns:", plays_tbl.columns.tolist())


def is_hands_up(traj_wr: pd.DataFrame, ball_xy, params=PARAMS):
    """
    Mark frames where WR is in a "hands-up" catch posture:
      - Facing within hands_up_align_deg of the ball bearing.

    Returns a boolean Series aligned with traj_wr.index.
    """
    if len(traj_wr) == 0:
        return pd.Series([], dtype=bool, index=traj_wr.index)

    bx, by = ball_xy
    aligns = []
    for _, r in traj_wr.iterrows():
        aligns.append(
            ang_diff_deg(
                float(r["o"]),
                bearing_deg(float(r["x"]), float(r["y"]), bx, by)
            ) <= params["hands_up_align_deg"]
        )

    aligns = np.array(aligns, dtype=bool)
    return pd.Series(aligns, index=traj_wr.index)



def min_dist_over_horizon(def_row, wr_row, horizon_s=0.5, dt=0.05):
    """
    Simulate defender and WR forward in time and return the minimum distance
    between them over the horizon.
    Simple kinematic extrapolation using current speed & orientation.
    """
    xD, yD = float(def_row["x"]), float(def_row["y"])
    sD, oD = float(def_row["s"]), float(def_row["o"])
    xW, yW = float(wr_row["x"]), float(wr_row["y"])
    sW, oW = float(wr_row["s"]), float(wr_row["o"])

    def step(x, y, s, o, dt):
        rad = math.radians(o)
        return x + s*dt*math.cos(rad), y + s*dt*math.sin(rad)

    steps = int(horizon_s / dt)
    mind = math.hypot(xD - xW, yD - yW)

    for _ in range(steps):
        xD, yD = step(xD, yD, sD, oD, dt)
        xW, yW = step(xW, yW, sW, oW, dt)
        d = math.hypot(xD - xW, yD - yW)
        if d < mind:
            mind = d
    return mind



def compute_defender_disruption_for_play(out_frames_play: pd.DataFrame,
                                         plays_tbl_row: pd.Series,
                                         params=PARAMS):
    """
    For one play:
      - Focus on last N frames before arrival (t_end, last_N_frames).
      - Identify WR hands-up frames.
      - For each hands-up frame:
          * evaluate EVERY defender's horizon-min distance to the WR
          * if min_d <= disrupt_R, credit that defender for that frame
      => Multiple defenders may be credited on the same frame.
      
    Returns:
      DataFrame with columns:
        game_id, play_id, frame_id,
        wr_nfl_id,
        def_nfl_id,
        min_dist,
        credited (0/1)
    """

    # Identify window frames
    t_end = int(plays_tbl_row["t_end"])
    t0   = int(plays_tbl_row["t0"])
    N    = int(params["last_N_frames"])
    disrupt_R = float(params["disrupt_R"])

    last_frames = list(range(max(t0, t_end - N + 1), t_end + 1))

    sub = out_frames_play[out_frames_play["frame_id"].isin(last_frames)].copy()
    sub = normalize_orientation(sub, plays_tbl_row["play_direction"])

    ball_xy = (
        float(plays_tbl_row["ball_land_x"]),
        float(plays_tbl_row["ball_land_y"])
    )

    # WR trajectory selection
    wr_mask = sub["player_role"].astype(str).str.lower().eq("targeted receiver")
    if wr_mask.sum() == 0:
        wr_mask = sub["player_side"].eq("Offense")

    wr_traj = sub[wr_mask].sort_values(["frame_id", "nfl_id"])
    if wr_traj.empty:
        return pd.DataFrame(columns=[
            "game_id","play_id","frame_id","wr_nfl_id","def_nfl_id",
            "min_dist","credited"
        ])

    wr_id = int(wr_traj["nfl_id"].mode().iloc[0])
    wr_traj = wr_traj[wr_traj["nfl_id"] == wr_id].sort_values("frame_id")
    wr_frames = wr_traj["frame_id"].values

    # Hands-up frames
    hands = is_hands_up(wr_traj, ball_xy, params)
    hands = hands.reindex(wr_traj.index).fillna(False).values

    # Evaluate all defenders
    records = []

    for frame, is_hu in zip(wr_frames, hands):
        if not is_hu:
            continue

        wr_f = wr_traj[wr_traj["frame_id"] == frame].iloc[0]

        defs_f = sub[(sub["player_side"] == "Defense") &
                     (sub["frame_id"] == frame)]
        if defs_f.empty:
            continue

        # Evaluate each defender independently
        for _, drow in defs_f.iterrows():
            md = min_dist_over_horizon(drow, wr_f,
                                       horizon_s=0.5,
                                       dt=0.05)

            credited = int(md <= disrupt_R)

            records.append(dict(
                game_id = int(plays_tbl_row["game_id"]),
                play_id = int(plays_tbl_row["play_id"]),
                frame_id = int(frame),
                wr_nfl_id = wr_id,
                def_nfl_id = int(drow["nfl_id"]),
                min_dist = float(md),
                credited = credited
            ))

    if not records:
        return pd.DataFrame(columns=[
            "game_id","play_id","frame_id","wr_nfl_id","def_nfl_id",
            "min_dist","credited"
        ])

    return pd.DataFrame.from_records(records)



def build_defender_disruption_events(out_frames: pd.DataFrame,
                                     plays_tbl: pd.DataFrame,
                                     params=PARAMS):
    """
    Loop over all plays and accumulate defender disruption-window events.
    """
    events = []

    # Pre-split out_frames by game_id/play_id for efficiency
    out_grouped = out_frames.groupby(["game_id","play_id"], sort=False)

    for _, pr in plays_tbl.iterrows():
        gid, pid = int(pr["game_id"]), int(pr["play_id"])
        try:
            out_play = out_grouped.get_group((gid, pid))
        except KeyError:
            continue
        df_ev = compute_defender_disruption_for_play(out_play, pr, params=params)
        if not df_ev.empty:
            events.append(df_ev)

    if not events:
        return pd.DataFrame(columns=[
            "game_id","play_id","frame_id","wr_nfl_id","def_nfl_id",
            "min_dist","credited"
        ])

    ddw_events = pd.concat(events, ignore_index=True)
    return ddw_events

ddw_events = build_defender_disruption_events(out, plays_tbl, params=PARAMS)
print("ddw_events shape:", ddw_events.shape)
ddw_events.head()



def build_defender_opportunities(inp_frames: pd.DataFrame, supp_df: pd.DataFrame):
    """
    One row per (game_id, play_id, def_nfl_id) where the defender was in
    'Defensive Coverage' on a pass play (ball actually goes in the air).
    """
    # Restrict to defensive coverage roles
    cov = inp_frames[
        (inp_frames["player_side"]=="Defense") &
        (inp_frames["player_role"].astype(str).str.lower()=="defensive coverage")
    ].copy()

    # One row per defender-play
    cov_play = (
        cov.groupby(["game_id","play_id","nfl_id"], as_index=False)
           .agg(player_name=("player_name","last"),
                player_position=("player_position","last"))
    )

    # Restrict to plays with a pass result (ball thrown)
    pass_mask = supp_df["pass_result"].isin(
        ["C","I","IN","INT","Complete","Incomplete","Interception"]
    )
    pass_plays = (
        supp_df.loc[pass_mask, ["game_id","play_id","defensive_team"]]
        .drop_duplicates()
    )

    opp = cov_play.merge(pass_plays, on=["game_id","play_id"], how="inner")
    opp = opp.rename(columns={"nfl_id":"def_nfl_id", "defensive_team":"def_team"})
    return opp

opp_df = build_defender_opportunities(inp, supp)
print("opp_df shape:", opp_df.shape)
opp_df.head()



def build_defender_metrics(ddw_events: pd.DataFrame,
                           opp_df: pd.DataFrame):
    """
    Aggregate disruption events and opportunities to defender-level metrics.
    Metrics:
      - OPP: pass coverage opportunities (plays)
      - DWP: Disruption Window Plays (plays where defender credited >=1 frame)
      - DWF: Disruption Window Frames (credited frames)
      - DWP_rate = DWP / OPP
      - DWF_per_play = DWF / OPP
      - avg_min_dist: average min distance on frames where the defender
                      was the primary candidate (for context, not ranking)
    """
    if ddw_events.empty:
        raise ValueError("No disruption window events computed.")

    per_play = (
        ddw_events
        .groupby(["def_nfl_id", "play_id"])
        .agg(
            credited_frames=("credited", "sum"),
            min_dist=("min_dist", "mean")
        )
        .reset_index()
    )
    
    per_play["credited_play"] = per_play["credited_frames"] > 0

    agg_events = (
        per_play.groupby("def_nfl_id")
        .agg(
            DWF=("credited_frames", "sum"),              # total credited frames
            DWP=("credited_play", "sum"),                # plays with ≥1 credited frame
            avg_min_dist=("min_dist", "mean")            # context metric
        )
        .reset_index()
    )

    # Opportunities per defender
    agg_opp = (
        opp_df.groupby("def_nfl_id")
        .agg(
            OPP=("play_id", lambda x: x.nunique()), # OPP = number of pass coverage plays where the defender was tagged "Defensive Coverage" for that play
            player_name=("player_name","last"),
            player_position=("player_position","last"),
            def_team=("def_team","last")
        )
        .reset_index()
    )

    df = agg_opp.merge(agg_events, on="def_nfl_id", how="left")

    # Fill “no events” defenders with zeros for counts; keep avg_min_dist NaN
    df[["DWF","DWP"]] = df[["DWF","DWP"]].fillna(0)

    # Rates: defenders with 0 OPP will get NaN rates
    df["DWP_rate"] = df["DWP"] / df["OPP"].replace(0, np.nan)
    df["DWF_per_play"] = df["DWF"] / df["OPP"].replace(0, np.nan)

    return df


def position_group(pos_raw: str):
    """
    Map raw player_position into larger buckets for leaderboards.
    Focus on LB / CB / S; everything else -> 'Other'.
    """
    if not isinstance(pos_raw, str):
        return "Other"
    pos = pos_raw.upper()
    if pos in {"LB","ILB","MLB","OLB"}:
        return "LB"
    if pos in {"CB","DB","NCB","SCB"}:
        return "CB"
    if pos in {"S","FS","SS","SS/FS"}:
        return "S"
    return "Other"


def_metrics = build_defender_metrics(ddw_events, opp_df)

# Replace infs with NaN, then clean for display
def_metrics = def_metrics.replace([np.inf, -np.inf], np.nan)

# Keep avg_min_dist as-is (NaN = defender never got close enough to be a candidate)

# For downstream use, treat NaN rates as 0.0
def_metrics["DWP_rate"] = def_metrics["DWP_rate"].fillna(0.0)
def_metrics["DWF_per_play"] = def_metrics["DWF_per_play"].fillna(0.0)

# Add position group
def_metrics["pos_group"] = def_metrics["player_position"].apply(position_group)

print("def_metrics shape:", def_metrics.shape)
display(def_metrics.head())



def top_defenders_by_group(def_metrics: pd.DataFrame,
                           group: str,
                           sort_col: str = "DWP_rate",
                           min_opp: int = 20,
                           min_dwp: int = 2,
                           top_n: int = 10):
    """
    Build a leaderboard for a given position group.
    Only includes defenders with:
      - OPP >= min_opp   (enough opportunities)
      - DWP >= min_dwp   (at least some disruptive plays)
    """
    df = def_metrics[def_metrics["pos_group"] == group].copy()

    # Sample-size and disruption filters
    df = df[(df["OPP"] >= min_opp) & (df["DWP"] >= min_dwp)]

    if df.empty:
        return df

    df = df.sort_values([sort_col, "DWP", "OPP"], ascending=[False, False, False])

    cols = [
        "def_nfl_id","player_name","def_team","player_position",
        "OPP","DWP","DWF","DWP_rate","DWF_per_play","avg_min_dist"
    ]
    return df[cols].head(top_n)


top_lb = top_defenders_by_group(def_metrics, "LB", min_opp=15, min_dwp=6, top_n=10)
top_cb = top_defenders_by_group(def_metrics, "CB", min_opp=20, min_dwp=6, top_n=10)
top_s  = top_defenders_by_group(def_metrics, "S",  min_opp=15, min_dwp=6, top_n=10)

print("Top LBs:")
display(top_lb)
print("Top CBs:")
display(top_cb)
print("Top Safeties:")
display(top_s)



# Per-play disruption scores + highlight plays

def build_defender_play_scores(ddw_events: pd.DataFrame, params=PARAMS):
    """
    Per-play disruption scoring for defenders.
    Returns a table with:
      - def_nfl_id
      - game_id
      - play_id
      - frames: credited frames
      - best_min_dist: smallest horizon distance in that play
      - score: combined quality metric
    """
    disrupt_R = float(params["disrupt_R"])

    play_scores = (
        ddw_events.groupby(["def_nfl_id","game_id","play_id"])
        .agg(
            frames=("credited","sum"),
            best_min_dist=("min_dist","min")
        )
        .reset_index()
    )

    # Combined quality metric: more frames + closer distance
    play_scores["score"] = (
        play_scores["frames"] +
        (disrupt_R - play_scores["best_min_dist"]).clip(lower=0.0)
    )
    return play_scores


play_scores = build_defender_play_scores(ddw_events, PARAMS)


def top_plays_for_defender(def_nfl_id: int,
                           play_scores: pd.DataFrame,
                           n: int = 5,
                           require_disruption: bool = True):
    """
    Return top-n plays for a given defender, sorted by disruption score.
    If require_disruption=True, only include plays with frames > 0.
    """
    df = play_scores[play_scores["def_nfl_id"] == def_nfl_id].copy()
    if df.empty:
        return df

    if require_disruption:
        df = df[df["frames"] > 0]

    if df.empty:
        return df  # defender has no disruptive plays

    df = df.sort_values(["score", "frames"], ascending=[False, False])
    return df[["game_id","play_id","frames","best_min_dist","score"]].head(n)


def print_highlight_tables(leaderboard: pd.DataFrame, label: str, n: int = 3):
    print(f"\n=== Highlight Plays for {label} ===")
    for _, row in leaderboard.head(n).iterrows():
        def_id = int(row["def_nfl_id"])
        name = row["player_name"]
        team = row["def_team"]
        print(f"\nDefender: {name} ({team})  |  NFL ID {def_id}")
        top_plays = top_plays_for_defender(def_id, play_scores, n=3)
        display(top_plays)

print_highlight_tables(top_cb, "Top CBs")
print_highlight_tables(top_lb, "Top LBs")
print_highlight_tables(top_s,  "Top Safeties")



# Distribution of Disruption Scores (Log-Scaled Y-Axis)

import seaborn as sns

# Extract scores, drop NaNs just in case
scores = play_scores["score"].dropna()

plt.figure(figsize=(10, 6))

sns.histplot(
    scores,
    kde=True,
    bins=40,
    alpha=0.6,
    edgecolor="black"
)

# Set y-axis to log scale (counts)
plt.yscale("log")

# Avoid log(0) issues on the lower bound
plt.ylim(bottom=1)

plt.title("Distribution of Defender Disruption Scores (Log Count)", fontsize=18, weight='bold')
plt.xlabel("Disruption Score", fontsize=14)
plt.ylabel("Count (log scale)", fontsize=14)

plt.grid(alpha=0.25, which="both", axis="y")
sns.despine()

plt.show()



from matplotlib.patches import Patch, Rectangle
import warnings

warnings.filterwarnings("ignore", message=".*use_inf_as_na.*", category=FutureWarning)

# Build defender_ratings from def_metrics (position-relative)
base = def_metrics[def_metrics["OPP"] > 0].copy()

# Core per-opportunity disruption metric
base["mean_score"] = base["DWF_per_play"]

# Empirical-Bayes-style shrinkage toward league-wide mean
k = 50
global_mean = base["mean_score"].mean()

reliability = base["OPP"] / (base["OPP"] + k)
base["adj_score"] = (
    reliability * base["mean_score"]
    + (1 - reliability) * global_mean
)

tier_palette = [
    {"name": "Needs Support",      "color": "#E74C3C", "short": "Bottom 30%"},
    {"name": "Baseline",           "color": "#F39C12", "short": "30-60%"},
    {"name": "Above Average",      "color": "#27AE60", "short": "60-90%"},
    {"name": "Outstanding",        "color": "#16A085", "short": "Top 10%"},
]

def _scale_within_pos_group(grp: pd.DataFrame) -> pd.DataFrame:
    """Scale adj_score to 0–100 and assign tiers within each pos_group."""
    grp = grp.copy()

    min_s = grp["adj_score"].min()
    max_s = grp["adj_score"].max()
    if max_s == min_s:
        grp["disruption_rating"] = 50.0
    else:
        grp["disruption_rating"] = 100 * (grp["adj_score"] - min_s) / (max_s - min_s)

    grp["percentile"] = grp["disruption_rating"].rank(pct=True) * 100

    r = grp["disruption_rating"].dropna()
    if len(r) >= 4:
        p30, p60, p90 = np.percentile(r, [30, 60, 90])
    else:
        p30, p60, p90 = 25.0, 50.0, 75.0

    grp["p30"] = p30
    grp["p60"] = p60
    grp["p90"] = p90

    tier_idx = []
    for val in grp["disruption_rating"]:
        if val < p30:
            tier_idx.append(0)
        elif val < p60:
            tier_idx.append(1)
        elif val < p90:
            tier_idx.append(2)
        else:
            tier_idx.append(3)

    grp["tier_idx"]   = tier_idx
    grp["tier_name"]  = [tier_palette[i]["name"]  for i in tier_idx]
    grp["tier_color"] = [tier_palette[i]["color"] for i in tier_idx]

    return grp

# Build defender_ratings per pos_group
pieces = []
for pg, grp in base.groupby("pos_group"):
    if pd.isna(pg):
        continue
    g = _scale_within_pos_group(grp)
    g["pos_group"] = pg
    pieces.append(g)

defender_ratings = pd.concat(pieces, ignore_index=True)

# Distribution plot

desired_order = ["CB", "LB", "S", "Other"]
available = defender_ratings["pos_group"].dropna().unique().tolist()
pos_groups = [pg for pg in desired_order if pg in available]

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(
    len(pos_groups), 1,
    figsize=(18, 3.5 * len(pos_groups)),
    sharex=False
)

plt.rcParams.update({
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

if len(pos_groups) == 1:
    axes = [axes]

fig.suptitle('Defender Disruption Rating Distribution by Position',
             fontsize=16, fontweight='bold', y=0.995)

for ax, pg in zip(axes, pos_groups):
    grp = defender_ratings[defender_ratings["pos_group"] == pg]
    r = grp["disruption_rating"].dropna()

    if r.empty:
        ax.text(0.5, 0.5, f"No data for {pg}", ha="center", va="center")
        ax.set_yticks([])
        continue

    # Position-specific thresholds
    p30 = grp["p30"].iloc[0]
    p60 = grp["p60"].iloc[0]
    p90 = grp["p90"].iloc[0]

    bands = [
        (0,   p30, tier_palette[0]["color"]),
        (p30, p60, tier_palette[1]["color"]),
        (p60, p90, tier_palette[2]["color"]),
        (p90, 100, tier_palette[3]["color"]),
    ]

    # Background colored bands
    for low, high, color in bands:
        ax.axvspan(low, high, color=color, alpha=0.15, zorder=0)

    # Histogram
    n, bins, patches = ax.hist(
        r,
        bins=25,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.8,
        color='#3498DB',
        zorder=2
    )

    # Color bars by tier
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        for low, high, color in bands:
            if low <= bin_center < high:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                break

    from scipy import stats
    kde = stats.gaussian_kde(r)
    x_range = np.linspace(r.min(), r.max(), 200)
    kde_values = kde(x_range)
    kde_scaled = kde_values * len(r) * (bins[1] - bins[0])

    ax2 = ax.twinx()
    ax2.plot(x_range, kde_scaled, color='#2C3E50', linewidth=2.5,
             alpha=0.8, label='Density', zorder=5)
    ax2.fill_between(x_range, kde_scaled, alpha=0.1,
                     color='#2C3E50', zorder=1)
    ax2.set_ylabel('')
    ax2.set_yticks([])

    cut_scores = [p30, p60, p90]

    for t_val in cut_scores:
        ax.axvline(t_val, color='black', linestyle='--', linewidth=1.8,
                   alpha=0.9, zorder=4)

        ax2.text(
            t_val, 0.97, f"{t_val:.0f}",
            transform=ax2.get_xaxis_transform(),
            ha='center', va='top', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white',
                      edgecolor='black',
                      alpha=0.9),
            zorder=10
        )

    # Statistics
    stats_text = f"n={len(r)} | μ={r.mean():.1f} | σ={r.std():.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=16, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    # Log-scale Y axis
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.set_ylabel("Player Count (log scale)", fontsize=16, fontweight='bold')

    # X-axis & limits
    ax.set_xlabel("Disruption Rating", fontsize=16, fontweight='bold')
    ax.set_xlim(-5, 105)

    # Position label badge
    ax.text(0.98, 0.98, f"{pg}", transform=ax.transAxes,
            fontsize=18, va='top', ha='right', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                      edgecolor='black', alpha=0.7))

    ax.grid(axis="y", alpha=0.3, which="both", linestyle=':')
    ax.grid(axis="x", alpha=0.2, linestyle=':')

    # Legend with fixed tier ranges
    handles = [
        Patch(
            facecolor=t["color"],
            alpha=0.7,
            label=f"{t['name']} ({t['short']})"
        )
        for t in tier_palette
    ]

    legend = ax.legend(
        handles=handles,
        title="Performance Tiers",
        loc="upper right",
        bbox_to_anchor=(0.94, 1),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=14
    )
    legend.get_title().set_fontsize(16)
    legend.get_title().set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, top=0.97)

# Save figure
fig_path = FIG_DIR / "disruption_rating_tiers_by_position.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight")

plt.show()
print("Saved figure to:", fig_path)



# Interactive Defender Disruption Explorer (Position + Team + Player)

import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.patches import Patch

# Widgets: position, team, player

pos_options = ["All positions"] + sorted(
    defender_ratings["pos_group"].dropna().unique().tolist()
)

pos_dd = widgets.Dropdown(
    options=pos_options,
    value="All positions",
    description="Position:",
    layout=widgets.Layout(width="35%"),
)

team_dd = widgets.Dropdown(
    options=["All teams"],
    value="All teams",
    description="Team:",
    layout=widgets.Layout(width="30%"),
)

player_dd = widgets.Dropdown(
    options=[],
    description="Player:",
    layout=widgets.Layout(width="60%"),
)

explorer_out = widgets.Output()


def get_filtered_df():
    """Apply current position + team filters to defender_ratings."""
    df = defender_ratings.copy()
    pos = pos_dd.value
    team = team_dd.value

    if pos != "All positions":
        df = df[df["pos_group"] == pos]
    if team != "All teams":
        df = df[df["def_team"] == team]

    return df


def refresh_teams(*args):
    """Refresh team dropdown based on current position filter."""
    df = defender_ratings.copy()
    pos = pos_dd.value
    if pos != "All positions":
        df = df[df["pos_group"] == pos]

    teams = ["All teams"] + sorted(df["def_team"].dropna().unique().tolist())
    team_dd.options = teams
    if team_dd.value not in teams:
        team_dd.value = "All teams"


def refresh_players(*args):
    """Refresh player dropdown based on current position + team filters."""
    df = get_filtered_df()
    df = df.sort_values(["def_team", "player_name"])

    options = [
        (f"{row.player_name} ({row.def_team})", int(row.def_nfl_id))
        for _, row in df.iterrows()
    ]

    if not options:
        options = [("No defenders found", -1)]

    player_dd.options = options
    player_dd.value = options[0][1]


def show_player(change=None):
    """Display metrics + tier-colored visuals for selected defender."""
    with explorer_out:
        clear_output()

        def_id = player_dd.value
        if def_id == -1:
            print("No defender available for this selection.")
            return

        row = defender_ratings.loc[
            defender_ratings["def_nfl_id"] == def_id
        ].iloc[0]

        # Summary table
        summary = pd.DataFrame({
            "Metric": [
                "Team",
                "Player",
                "Position group",
                "Tier",
                "Coverage opportunities (OPP)",
                "Disruptive plays (DWP)",
                "Disruptive frames (DWF)",
                "DWP rate (plays with disruption)",
                "DWF per play",
                "Adj. disruption score",
                "Disruption rating (0–100, within position)",
                "Percentile within position",
            ],
            "Value": [
                row["def_team"],
                row["player_name"],
                row["pos_group"],
                row["tier_name"],
                int(row["OPP"]),
                int(row["DWP"]),
                int(row["DWF"]),
                f"{row['DWP_rate']:.3f}",
                f"{row['DWF_per_play']:.3f}",
                f"{row['adj_score']:.3f}",
                f"{row['disruption_rating']:.1f}",
                f"{row['percentile']:.1f}th",
            ],
        })

        display(summary.style.hide(axis="index"))

        # Tier-colored mini bar for this player
        fig, ax = plt.subplots(figsize=(6, 1.4))

        # If a specific position is selected, use its thresholds for bands
        df_pos = defender_ratings[
            defender_ratings["pos_group"] == row["pos_group"]
        ]
        p30 = df_pos["p30"].iloc[0]
        p60 = df_pos["p60"].iloc[0]
        p90 = df_pos["p90"].iloc[0]

        bands = [
            (0,   p30, tier_palette[0]["color"]),
            (p30, p60, tier_palette[1]["color"]),
            (p60, p90, tier_palette[2]["color"]),
            (p90, 100, tier_palette[3]["color"]),
        ]

        for low, high, color in bands:
            ax.axvspan(low, high, color=color, alpha=0.2)

        ax.barh([""], [row["disruption_rating"]],
                color=row["tier_color"],
                edgecolor="black")

        ax.set_xlim(0, 100)
        ax.set_xlabel("Disruption rating (0–100, within position)")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.3)
        sns.despine(left=True, bottom=False)
        plt.tight_layout()
        plt.show()

        # Distribution for current filter + highlight player
        df = get_filtered_df()
        if df.empty:
            print("No defenders in this filter to show distribution.")
            return

        fig, ax = plt.subplots(figsize=(8, 2.5))

        # If filter is on a single position, use that group's thresholds for background
        if pos_dd.value != "All positions":
            df_pos = defender_ratings[
                defender_ratings["pos_group"] == pos_dd.value
            ]
            p30 = df_pos["p30"].iloc[0]
            p60 = df_pos["p60"].iloc[0]
            p90 = df_pos["p90"].iloc[0]
            bands = [
                (0,   p30, tier_palette[0]["color"]),
                (p30, p60, tier_palette[1]["color"]),
                (p60, p90, tier_palette[2]["color"]),
                (p90, 100, tier_palette[3]["color"]),
            ]
            for low, high, color in bands:
                ax.axvspan(low, high, color=color, alpha=0.12)

        xs = df["disruption_rating"].values
        ys = np.zeros_like(xs)
        colors = df["tier_color"].values

        ax.scatter(xs, ys, alpha=0.7, s=30, c=colors, edgecolor="none")

        # Highlight selected defender
        ax.scatter(
            [row["disruption_rating"]], [0],
            s=100, edgecolor="black", linewidth=1.5,
            c=[row["tier_color"]],
            zorder=3,
        )
        ax.text(
            row["disruption_rating"], 0.02,
            row["player_name"],
            ha="center", va="bottom", fontsize=9,
        )

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        filt_label = []
        if pos_dd.value != "All positions":
            filt_label.append(pos_dd.value)
        if team_dd.value != "All teams":
            filt_label.append(team_dd.value)
        title_suffix = " | ".join(filt_label) if filt_label else "League"
        ax.set_xlabel("Disruption rating (0–100, within position)")
        ax.set_title(f"Disruption Rating Distribution ({title_suffix})")
        sns.despine(left=True, bottom=False)
        plt.tight_layout()
        plt.show()


def _on_pos_change(change):
    refresh_teams()
    refresh_players()

pos_dd.observe(_on_pos_change, names="value")
team_dd.observe(lambda ch: refresh_players(), names="value")
player_dd.observe(show_player, names="value")

# Initialize widgets
refresh_teams()
refresh_players()
show_player()

ui = widgets.VBox([
    widgets.HBox([pos_dd, team_dd]),
    player_dd,
    explorer_out,
])

display(ui)



# Best plays by position group (play-level highlights)

def build_play_scores_with_pos(play_scores: pd.DataFrame,
                               def_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Attach position-group info to play_scores.
    Returns play_scores with:
      - player_name
      - player_position
      - def_team
      - pos_group (LB / CB / S / Other)
    """
    pos_cols = [
        "def_nfl_id",
        "player_name",
        "player_position",
        "def_team",
        "pos_group",
    ]
    return play_scores.merge(def_metrics[pos_cols], on="def_nfl_id", how="left")


play_scores_pos = build_play_scores_with_pos(play_scores, def_metrics)


def top_plays_by_position(play_scores_pos: pd.DataFrame,
                          group: str,
                          n: int = 10,
                          require_disruption: bool = True) -> pd.DataFrame:
    """
    Return top-n plays for a given position group (LB / CB / S / Other),
    sorted by disruption score.

    Each row is:
      - pos_group
      - game_id, play_id
      - def_nfl_id, player_name, def_team, player_position
      - frames (credited frames)
      - best_min_dist
      - score
    """
    df = play_scores_pos[play_scores_pos["pos_group"] == group].copy()
    if df.empty:
        return df

    if require_disruption:
        df = df[df["frames"] > 0]

    if df.empty:
        return df

    df = df.sort_values(["score", "frames"], ascending=[False, False])

    cols = [
        "pos_group",
        "game_id",
        "play_id",
        "def_nfl_id",
        "player_name",
        "def_team",
        "player_position",
        "frames",
        "best_min_dist",
        "score",
    ]
    return df[cols].head(n)


def print_position_top_plays(play_scores_pos: pd.DataFrame,
                             groups=("CB", "LB", "S"),
                             n: int = 5):
    """
    Pretty-print top plays by position group.
    """
    for g in groups:
        print(f"\n=== Top Plays for Position Group: {g} ===")
        top_df = top_plays_by_position(play_scores_pos, g, n=n)
        if top_df.empty:
            print("  (No qualifying plays)")
        else:
            display(top_df)


# Example: top 5 plays for CB / LB / S
print_position_top_plays(play_scores_pos, groups=("CB", "LB", "S"), n=5)



# Play-level disruption flag (any defender)
play_disrupt = (
    ddw_events.groupby(["game_id","play_id"])
    .agg(DWF_total=("credited","sum"))
    .reset_index()
)
play_disrupt["has_disruption"] = (play_disrupt["DWF_total"] > 0).astype(int)

# Labels: 1 = complete, 0 = incomplete / INT
labels = get_labels_from_supp(supp)  # already defined earlier
play_eval = labels.merge(play_disrupt, on=["game_id","play_id"], how="left")
play_eval["has_disruption"] = play_eval["has_disruption"].fillna(0).astype(int)

comp_by_disrupt = (
    play_eval.groupby("has_disruption")["label_completion"]
    .mean()
    .rename(index={0: "no_disruption", 1: "disruption"})
)

print("Completion rate by disruption presence:")
display(comp_by_disrupt)


def visualize_disruption_for_defender(def_nfl_id: int,
                                      ddw_events: pd.DataFrame,
                                      def_metrics: pd.DataFrame,
                                      out_frames: pd.DataFrame,
                                      plays_tbl: pd.DataFrame,
                                      n_plays: int = 3):
    """
    For a selected defender, show a few plays where he's credited
    in the disruption window: simple scatter of T0 + ball landing.
    """
    # Which plays did he get credited on?
    plays_def = (
        ddw_events[ddw_events["def_nfl_id"] == def_nfl_id]
        .query("credited == 1")[["game_id","play_id"]]
        .drop_duplicates()
    )
    if plays_def.empty:
        print("No credited disruption plays for this defender.")
        return

    # Limit to first n_plays
    plays_def = plays_def.head(n_plays)

    # Pull defender meta
    row_meta = def_metrics[def_metrics["def_nfl_id"] == def_nfl_id].iloc[0]
    print(f"Defender {row_meta['player_name']} ({row_meta['player_position']}, {row_meta['def_team']})")
    print(
        f"OPP={row_meta['OPP']}, DWP={row_meta['DWP']}, "
        f"DWF={row_meta['DWF']}, DWP_rate={row_meta['DWP_rate']:.3f}"
    )

    for _, r in plays_def.iterrows():
        gid, pid = int(r["game_id"]), int(r["play_id"])

        pr = plays_tbl[
            (plays_tbl["game_id"].eq(gid)) &
            (plays_tbl["play_id"].eq(pid))
        ].iloc[0]

        # Use out_frames (the tracking DataFrame), not the widget
        out_play = out_frames[
            (out_frames["game_id"].eq(gid)) &
            (out_frames["play_id"].eq(pid))
        ]

        t_end = int(pr["t_end"])
        N = int(PARAMS["last_N_frames"])
        last_frames = list(range(max(int(pr["t0"]), t_end - N + 1), t_end + 1))
        sub = out_play[out_play["frame_id"].isin(last_frames)].copy()
        sub = normalize_orientation(sub, pr["play_direction"])

        ball_xy = (float(pr["ball_land_x"]), float(pr["ball_land_y"]))

        # WR at each frame
        wr_mask = sub["player_role"].astype(str).str.lower().eq("targeted receiver")
        if wr_mask.sum() == 0:
            wr_mask = sub["player_side"].eq("Offense")

        wr_traj = sub[wr_mask].sort_values(["frame_id","nfl_id"])
        if wr_traj.empty:
            continue
        wr_id = wr_traj["nfl_id"].mode().iloc[0]
        wr_traj = wr_traj[wr_traj["nfl_id"].eq(wr_id)].sort_values("frame_id")

        # Defender trajectory
        def_traj = sub[sub["nfl_id"].eq(def_nfl_id)].sort_values("frame_id")

        plt.figure(figsize=(8, 4.5))
        # All offense/defense at final frame
        df_t_end = sub[sub["frame_id"] == t_end]
        off = df_t_end[df_t_end["player_side"] == "Offense"]
        deff = df_t_end[df_t_end["player_side"] == "Defense"]

        plt.scatter(off["x"], off["y"], s=40, label="Offense", c=COLOR_OFFENSE, alpha=0.8)
        plt.scatter(deff["x"], deff["y"], s=40, label="Defense", c=COLOR_DEFENSE, alpha=0.6)

        # WR path
        plt.plot(wr_traj["x"], wr_traj["y"], lw=2, c=COLOR_TR, label="Targeted WR path")

        # Defender path
        if not def_traj.empty:
            plt.plot(def_traj["x"], def_traj["y"], lw=2, c=COLOR_DEF_HERO, label="Defender path")

        plt.scatter([ball_xy[0]], [ball_xy[1]], s=80, marker="x",
                    linewidths=2.0, c=COLOR_BALL, label="Ball landing")

        plt.xlim(FIELD_X_MIN, FIELD_X_MAX)
        plt.ylim(FIELD_Y_MIN, FIELD_Y_MAX)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(
            f"Game {gid} Play {pid} - Disruption Window "
            f"(last {PARAMS['last_N_frames']} frames)"
        )
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.show()


# Example: visualize for the #1 CB if available
if not top_cb.empty:
    example_def_id = int(top_cb.iloc[0]["def_nfl_id"])
    visualize_disruption_for_defender(
        example_def_id,
        ddw_events=ddw_events,
        def_metrics=def_metrics,
        out_frames=out_tracks,
        plays_tbl=plays_tbl,
        n_plays=3,
    )



def_metrics.to_csv(ANALYS_DIR / "defender_disruption_metrics150.csv", index=False)
top_lb.to_csv(ANALYS_DIR / "leaderboard_LB150.csv", index=False)
top_cb.to_csv(ANALYS_DIR / "leaderboard_CB150.csv", index=False)
top_s.to_csv(ANALYS_DIR / "leaderboard_S150.csv", index=False)

print("Saved defender metrics and leaderboards under:", ANALYS_DIR)


