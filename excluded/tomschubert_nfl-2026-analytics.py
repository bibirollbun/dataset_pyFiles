import os
import sys
ARE_WE_IN_KAGGLE = len([e for e in os.environ if e.startswith("KAGGLE_")]) > 1
if ARE_WE_IN_KAGGLE:
    sys.path.append("/kaggle/input/nfl-utils-package")

    print("Running in Kaggle environment")
    import plotly.io as pio
    pio.renderers.default = "iframe"   # or "iframe_connected"

import numpy as np
import pandas as pd
import torch
import math
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
from functools import partial
from utils.feature_engineering.preprocessing_utils import *
import gc
from utils.feature_engineering.Config import CONFIG
import plotly.graph_objects as go
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import orjson


config = CONFIG()

RDBU = [
    [0.00, "rgb(178,24,43)"],    # red (deep)
    [0.18, "rgb(214,96,77)"],    # red-orange
    [0.32, "rgb(244,165,130)"],  # orange/salmon
    [0.50, "rgb(247,247,247)"],  # white
    [0.68, "rgb(146,197,222)"],  # light blue
    [0.82, "rgb(67,147,195)"],   # medium blue
    [1.00, "rgb(33,102,172)"],   # blue (deep)
]

RDBU_MPL = LinearSegmentedColormap.from_list(
    "RDBU_richer",
    [
        (0.00, (178/255,  24/255,  43/255)),  # deep red
        (0.18, (214/255,  96/255,  77/255)),  # red-orange
        (0.32, (244/255, 165/255, 130/255)),  # light orange/salmon
        (0.50, (247/255, 247/255, 247/255)),  # white
        (0.68, (146/255, 197/255, 222/255)),  # light blue
        (0.82, ( 67/255, 147/255, 195/255)),  # medium blue
        (1.00, ( 33/255, 102/255, 172/255)),  # deep blue
    ],
)


df = pd.read_parquet(config.FEATUREDIR / f"all_features_fraction_{config.DEBUG_FRACTION}.parquet")
df_targets = pd.read_parquet(config.FEATUREDIR/ f"targets_fraction_{config.DEBUG_FRACTION}.parquet")
n_before = len(df_targets)
# enhance data with import info for visu

try: 
    import nflreadpy as nfl
    pbp = nfl.load_pbp([2022, 2023, 2024]).to_pandas()
    pbp.to_parquet(config.FEATUREDIR / f"pbp_extended.parquet", index=False)
    print(list(pbp.columns))
except:
    print("nflreadpy not available, loading from feature dir")
    pbp = pd.read_parquet(config.FEATUREDIR / f"pbp_extended.parquet")


# Make sure merge keys match dtype
pbp[["old_game_id", "play_id"]] = pbp[["old_game_id", "play_id"]].astype("int64")
df_targets[["game_id", "play_id"]] = df_targets[["game_id", "play_id"]].astype("int64")

cols_to_add = [
    "old_game_id", "play_id",
    "home_team", "away_team",
    "posteam", "defteam", "posteam_type",
    "qtr", "down", "ydstogo",
    "yardline_100",  # yards to TD / end zone
    "total_home_score", "total_away_score",
    "posteam_score", "defteam_score",
    "time", "quarter_seconds_remaining",
]

# If pbp has duplicates per key, keep the last by order_sequence if present
if "order_sequence" in pbp.columns:
    pbp_key = (pbp[cols_to_add + ["order_sequence"]]
            .sort_values(["old_game_id", "play_id", "order_sequence"])
            .drop_duplicates(["old_game_id", "play_id"], keep="last")
            .drop(columns=["order_sequence"]))
else:
    pbp_key = (pbp[cols_to_add]
            .drop_duplicates(["old_game_id", "play_id"], keep="last"))

df_targets = df_targets.merge(
    pbp_key,
    left_on=["game_id", "play_id"],
    right_on=["old_game_id", "play_id"],
    how="left",
    validate="m:1",
)

n_after = len(df_targets)
assert n_after == n_before, f"Row count changed! before={n_before}, after={n_after}"


# scaler = joblib.load(config.FEATUREDIR / f"scaler_fraction_{config.DEBUG_FRACTION}.joblib")


import numpy as np
import pandas as pd

DT = 0.1

def compute_player_params(df: pd.DataFrame,
                          q: float = 0.995,
                          min_a_fwd: float = 2.0,
                          min_a_back: float = 2.5,
                          min_a_lat: float = 2.5,
                          min_vmax: float = 6.0,
                          dt: float = DT,
                          min_speed_for_accel: float = 1.0) -> pd.DataFrame:
    """
    Per-player params computed ONLY from x/y (and dir for yaw rate):
      v_max:  q-quantile of speed_xy
      a_fwd:  q-quantile of a_signed where a_signed > 0
      a_back: q-quantile of (-a_signed) where a_signed < 0
      a_lat:  q-quantile of |speed_xy * yaw_rate|

    Units: x,y in yards, dt in seconds => v in yd/s, a in yd/s^2, yaw_rate in rad/s.
    """

    d = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()

    # dir is already radians
    d["theta"] = d["dir"].to_numpy(float)

    def _per_track(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()

        # --- unwrap heading + yaw rate ---
        th = np.unwrap(g["theta"].to_numpy(float))
        yaw_rate = np.concatenate([[0.0], np.diff(th) / dt])  # rad/s
        g["theta_u"] = th
        g["yaw_rate"] = yaw_rate

        # --- kinematics from x/y ---
        x = g["x"].to_numpy(float)
        y = g["y"].to_numpy(float)

        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)

        speed_xy = np.sqrt(vx * vx + vy * vy)

        # signed tangential accel (projection of a onto v)
        denom = np.clip(speed_xy, 1e-6, None)
        a_signed = (ax * vx + ay * vy) / denom

        g["vx"] = vx
        g["vy"] = vy
        g["ax"] = ax
        g["ay"] = ay
        g["speed_xy"] = speed_xy
        g["a_signed"] = a_signed
        g["a_mag_xy"] = np.sqrt(ax * ax + ay * ay)

        # lateral accel magnitude approx
        g["a_lat_obs"] = np.abs(speed_xy * yaw_rate)

        return g

    d = d.groupby(["game_id", "play_id", "nfl_id"], group_keys=False).apply(_per_track)

    # optional: avoid low-speed blow-ups in accel quantiles
    fast = d["speed_xy"] >= float(min_speed_for_accel)

    g_player = d.groupby("nfl_id", sort=False)

    v_max = g_player["speed_xy"].quantile(q).rename("v_max")

    a_fwd = (
        d.loc[fast & (d["a_signed"] > 0)]
         .groupby("nfl_id")["a_signed"]
         .quantile(q)
         .rename("a_fwd")
    )

    a_back = (
        (-d.loc[fast & (d["a_signed"] < 0), "a_signed"])
        .groupby(d.loc[fast & (d["a_signed"] < 0), "nfl_id"])
        .quantile(q)
        .rename("a_back")
    )

    a_lat = (
        d.loc[fast]
         .groupby("nfl_id")["a_lat_obs"]
         .quantile(q)
         .rename("a_lat")
    )

    params = pd.concat([v_max, a_fwd, a_back, a_lat], axis=1)

    # floors + fill
    params["v_max"]  = params["v_max"].fillna(min_vmax).clip(lower=min_vmax)
    params["a_fwd"]  = params["a_fwd"].fillna(min_a_fwd).clip(lower=min_a_fwd)
    params["a_back"] = params["a_back"].fillna(min_a_back).clip(lower=min_a_back)
    params["a_lat"]  = params["a_lat"].fillna(min_a_lat).clip(lower=min_a_lat)

    return params



import numpy as np
import pandas as pd

DT_BALL = 0.1  # seconds per frame for num_frames_output
FIELD_X_MIN = 0.0
FIELD_X_MAX = 120.0
FIELD_Y_MIN = 0.0
FIELD_Y_MAX = 160/3.0

def build_throw_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (game_id, play_id):
      dist_yards: distance from passer at last frame to ball_land (euclidean)
      t_air:      num_frames_output * 0.1
    """
    required = ["game_id","play_id","frame_id","is_passer","x","y",
                "ball_land_x","ball_land_y","num_frames_output"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # last frame per play
    last_frame = df.groupby(["game_id","play_id"])["frame_id"].max().rename("last_frame")
    d = df.merge(last_frame, on=["game_id","play_id"], how="inner")

    # passer at last frame
    d = d[(d["frame_id"] == d["last_frame"]) & (d["is_passer"] == True)].copy()

    # if multiple passers at last frame, keep one (you can refine if needed)
    d = d.sort_values(["game_id","play_id","nfl_id"]).drop_duplicates(["game_id","play_id"], keep="first")

    # compute distance and air time
    dx = d["ball_land_x"].to_numpy(float) - d["x"].to_numpy(float)
    dy = d["ball_land_y"].to_numpy(float) - d["y"].to_numpy(float)
    dist = np.sqrt(dx*dx + dy*dy)

    t_air = d["num_frames_output"].to_numpy(float) * DT_BALL

    out = pd.DataFrame({
        "game_id": d["game_id"].to_numpy(),
        "play_id": d["play_id"].to_numpy(),
        "dist_yards": dist,
        "t_air": t_air,
    })

    out = out[np.isfinite(out["dist_yards"]) & np.isfinite(out["t_air"]) & (out["t_air"] > 0)]
    return out



def fit_ball_time_quantile_model(df: pd.DataFrame,
                                 bin_width=1.0,
                                 q_low=0.05,
                                 q_high=0.995,
                                 min_samples_per_bin=50):
    """
    Learns t_low(dist), t_high(dist) from data and returns a function:
        (t_low, t_high) = f(dist_yards_array)
    using distance bins + linear interpolation.
    """
    samples = build_throw_samples(df)

    if len(samples) == 0:
        raise ValueError("No valid throw samples found.")

    # bin distances
    dmin = float(np.floor(samples["dist_yards"].min() / bin_width) * bin_width)
    dmax = float(np.ceil(samples["dist_yards"].max() / bin_width) * bin_width)
    edges = np.arange(dmin, dmax + bin_width + 1e-9, bin_width)

    samples["bin"] = pd.cut(samples["dist_yards"], bins=edges, include_lowest=True, right=False)
    grp = samples.groupby("bin")["t_air"]

    stats = pd.DataFrame({
        "count": grp.size(),
        "t_low": grp.quantile(q_low),
        "t_high": grp.quantile(q_high),
    }).dropna()

    stats = stats[stats["count"] >= min_samples_per_bin].copy()
    if len(stats) < 2:
        # fallback: global quantiles (still returns a valid callable)
        g_low = float(samples["t_air"].quantile(q_low))
        g_high = float(samples["t_air"].quantile(q_high))
        def f(dist_yards, q_low=q_low, q_high=q_high):
            dist_yards = np.asarray(dist_yards, dtype=float)
            return np.full_like(dist_yards, g_low, dtype=float), np.full_like(dist_yards, g_high, dtype=float)
        return f, stats, samples

    # bin centers from interval
    centers = np.array([iv.left + 0.5 * bin_width for iv in stats.index.categories if iv in stats.index], dtype=float)
    # The above line is a bit fiddly; simplest robust approach:
    centers = np.array([iv.left + 0.5 * bin_width for iv in stats.index], dtype=float)

    t_low_vals = stats["t_low"].to_numpy(float)
    t_high_vals = stats["t_high"].to_numpy(float)

    # ensure sorted by center
    order = np.argsort(centers)
    centers = centers[order]
    t_low_vals = t_low_vals[order]
    t_high_vals = t_high_vals[order]

    def ball_time_quantiles_from_distance(dist_yards: np.ndarray,
                                          q_low=q_low, q_high=q_high):
        dist_yards = np.asarray(dist_yards, dtype=float)
        # linear interpolation; clamp outside range
        t_low = np.interp(dist_yards, centers, t_low_vals, left=t_low_vals[0], right=t_low_vals[-1])
        t_high = np.interp(dist_yards, centers, t_high_vals, left=t_high_vals[0], right=t_high_vals[-1])
        return t_low, t_high

    return ball_time_quantiles_from_distance, stats, samples



ball_q_fn, dist_stats, throw_samples = fit_ball_time_quantile_model(
    df,
    bin_width=1.0,
    q_low=0.1,
    q_high=0.9,
    min_samples_per_bin=25
)

distances = np.arange(0,50,1)

times = []
lows = []
highs = []

for dist in distances:
    low, high = ball_q_fn(dist)
    mean = (low + high) / 2.0
    times.append(mean)
    lows.append(low)
    highs.append(high)

plt.plot(distances, times)
plt.fill_between(distances, lows, highs, color="lightgray", alpha=0.5)
plt.xlabel("Distance (yards)")
plt.ylabel("Mean ball travel time (seconds)")
plt.title("Estimated ball travel time vs distance")





import numpy as np

def _sec_to_mmss(sec):
    try:
        sec = int(sec)
    except Exception:
        return None
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

def reachable_joint_tau_sparse_global(
    p0, v0, a_fwd, a_back, a_lat, v_max,
    *,
    # global grid definition (always provided)
    X_MIN: float,
    X_MAX: float,
    Y_MIN: float,
    Y_MAX: float,
    step: float,

    taus,  # e.g. np.arange(dt_tau, T_max + 1e-9, dt_tau)

    # ball info (always provided)
    qb_pos,
    ball_quantile_fn,
    q_low: float,
    q_high: float,

    # gating + joint time behavior
    require_within_interval: bool = False,  # False: player arrives by t_high
    ball_time_for_joint: str = "low",       # "low" or "high"

    # performance
    pad_cells: int = 2,
):
    """
    Sparse global reachability on a global lattice, evaluated only in a local index window.

    Returns:
      idx_phys   : int64 (N,) global linear indices for physics-reachable cells
      tau_player : float32 (N,) min player arrival time for idx_phys

      idx_gate   : int64 (M,) global linear indices for ball-gated cells
      tau_joint  : float32 (M,) joint time = max(min player time, ball time)

      nx, ny     : global grid shape, with linear_idx = j*nx + i
    """

    p0 = np.asarray(p0, dtype=float).reshape(2,)
    v0 = np.asarray(v0, dtype=float).reshape(2,)
    qb_pos = np.asarray(qb_pos, dtype=float).reshape(2,)

    nx = int(np.floor((X_MAX - X_MIN) / step + 1e-9)) + 1
    ny = int(np.floor((Y_MAX - Y_MIN) / step + 1e-9)) + 1

    # --- conservative spatial bound to avoid full-field evaluation ---
    T_max = float(np.max(taus)) if len(taus) else 0.0
    a_max = float(max(a_fwd, a_back, a_lat))
    R = v_max * T_max + 0.5 * a_max * (T_max ** 2) + pad_cells * step

    i0 = int(np.rint((p0[0] - X_MIN) / step))
    j0 = int(np.rint((p0[1] - Y_MIN) / step))
    di = int(np.ceil(R / step))
    dj = int(np.ceil(R / step))

    i_lo = max(0, i0 - di); i_hi = min(nx - 1, i0 + di)
    j_lo = max(0, j0 - dj); j_hi = min(ny - 1, j0 + dj)

    ii = np.arange(i_lo, i_hi + 1, dtype=np.int32)
    jj = np.arange(j_lo, j_hi + 1, dtype=np.int32)
    I, J = np.meshgrid(ii, jj, indexing="xy")  # (H,W)

    X = X_MIN + I.astype(float) * step
    Y = Y_MIN + J.astype(float) * step
    grid = np.stack([X, Y], axis=-1)  # (H,W,2)

    # --- local frame ---
    vnorm = np.linalg.norm(v0)
    t_hat = np.array([1.0, 0.0]) if vnorm < 1e-6 else (v0 / vnorm)
    n_hat = np.array([-t_hat[1], t_hat[0]])

    best_tau = np.full(X.shape, np.inf, dtype=float)

    for tau in taus:
        tau = float(tau)
        a_req = 2.0 * (grid - p0 - v0 * tau) / (tau * tau)

        a_par = a_req[..., 0] * t_hat[0] + a_req[..., 1] * t_hat[1]
        a_per = a_req[..., 0] * n_hat[0] + a_req[..., 1] * n_hat[1]

        a_par_lim = np.where(a_par >= 0.0, a_fwd, a_back)
        slack = (a_par / a_par_lim) ** 2 + (a_per / a_lat) ** 2

        v1 = v0 + a_req * tau
        speed_ok = np.linalg.norm(v1, axis=-1) <= v_max

        ok = (slack <= 1.0) & speed_ok
        best_tau = np.where(ok & (tau < best_tau), tau, best_tau)

    # --- physics reachable sparse output ---
    reachable_phys = np.isfinite(best_tau)
    phys_i = I[reachable_phys].astype(np.int32)
    phys_j = J[reachable_phys].astype(np.int32)
    idx_phys = phys_j.astype(np.int64) * nx + phys_i.astype(np.int64)
    tau_player = best_tau[reachable_phys].astype(np.float32)

    # --- ball quantiles + gate ---
    dist = np.linalg.norm(grid - qb_pos, axis=-1)
    t_low, t_high = ball_quantile_fn(dist, q_low=q_low, q_high=q_high)

    if require_within_interval:
        gate_mask = reachable_phys & (best_tau <= t_high)
    else:
        gate_mask = reachable_phys & (best_tau <= t_high)

    if ball_time_for_joint == "low":
        t_ball = t_low
    elif ball_time_for_joint == "high":
        t_ball = t_high
    else:
        raise ValueError("ball_time_for_joint must be 'low' or 'high'")

    joint_tau_grid = np.maximum(best_tau, t_ball)

    gate_i = I[gate_mask].astype(np.int32)
    gate_j = J[gate_mask].astype(np.int32)
    idx_gate = gate_j.astype(np.int64) * nx + gate_i.astype(np.int64)
    tau_joint = joint_tau_grid[gate_mask].astype(np.float32)

    return idx_phys, tau_player, idx_gate, tau_joint, nx, ny



### helper functions

def v0_from_s_dir(row) -> np.ndarray:
    """
    Assumes:
      - row["dir"] is angle in radians
      - dir=0 points along +x
      - angle increases CCW (standard math convention)
    """
    theta = float(row["dir"])
    s = float(row["s"])
    return s * np.array([np.cos(theta), np.sin(theta)], dtype=float)

def get_qb_pos_from_df(df: pd.DataFrame,
                       game_id,
                       play_id,
                       frame_id,
                       qb_nfl_id=None) -> np.ndarray:
    """
    Returns qb_pos = np.array([x,y]) for the given game/play/frame.

    Priority:
      1) qb_nfl_id (if provided)
      2) is_passer == True (your dataset)
    If multiple passers exist, chooses the one closest to (ball_land_x, ball_land_y) if present.
    """

    d = df[(df["game_id"] == game_id) &
           (df["play_id"] == play_id) &
           (df["frame_id"] == frame_id)]

    if len(d) == 0:
        raise ValueError(f"No rows for game/play/frame={game_id}/{play_id}/{frame_id}")

    # 1) explicit nfl_id
    if qb_nfl_id is not None:
        qb = d[d["nfl_id"] == qb_nfl_id]
        if len(qb) != 1:
            raise ValueError(f"QB row not uniquely found for qb_nfl_id={qb_nfl_id} "
                             f"(found {len(qb)} rows) at game/play/frame={game_id}/{play_id}/{frame_id}")
        return qb[["x", "y"]].iloc[0].to_numpy(dtype=float)

    # 2) is_passer
    if "is_passer" not in d.columns:
        raise ValueError("df has no 'is_passer' column")

    qb = d[d["is_passer"] == True]
    if len(qb) == 1:
        return qb[["x", "y"]].iloc[0].to_numpy(dtype=float)

    if len(qb) == 0:
        raise ValueError(f"No passer found (is_passer=True) at game/play/frame={game_id}/{play_id}/{frame_id}")

    return qb[["x", "y"]].iloc[0].to_numpy(dtype=float)


def v0_from_s_dir(row) -> np.ndarray:
    """
    Assumes:
      - row["dir"] is angle in radians
      - dir=0 points along +x
      - angle increases CCW (standard math convention)
    """
    theta = float(row["dir"])
    s = float(row["s"])
    return s * np.array([np.cos(theta), np.sin(theta)], dtype=float)





def _compute_reachability_for_row_sparse(
    df_play: pd.DataFrame,
    player_params: pd.DataFrame,
    row: pd.Series,
    qb_pos: np.ndarray,
    ball_quantile_fn,
    *,
    # global grid definition
    step: float = 0.5,

    T_max: float = 3.0,
    dt_tau: float = 0.05,

    q_low: float = 0.05,
    q_high: float = 0.995,

    require_within_interval: bool = False,
    ball_time_for_joint: str = "low",   # "low" or "high"
):
    p0 = np.array([float(row["x"]), float(row["y"])], dtype=float)
    v0 = v0_from_s_dir(row)

    pid = int(row["nfl_id"])
    par = player_params.loc[pid]
    v_max  = float(par["v_max"])
    a_fwd  = float(par["a_fwd"])
    a_back = float(par["a_back"])
    a_lat  = float(par["a_lat"])

    taus = np.arange(dt_tau, T_max + 1e-9, dt_tau)

    idx_phys, tau_player, idx_gate, tau_joint, nx, ny = reachable_joint_tau_sparse_global(
        p0, v0,
        a_fwd=a_fwd, a_back=a_back, a_lat=a_lat, v_max=v_max,
        X_MIN=FIELD_X_MIN, X_MAX=FIELD_X_MAX,
        Y_MIN=FIELD_Y_MIN, Y_MAX=FIELD_Y_MAX,
        step=step,
        taus=taus,
        qb_pos=qb_pos,
        ball_quantile_fn=ball_quantile_fn,
        q_low=q_low, q_high=q_high,
        require_within_interval=require_within_interval,
        ball_time_for_joint=ball_time_for_joint,
    )

    # minimal merge payload (both physics + gated)
    # physics: merge earliest player arrival
    # gated:   merge earliest joint time (player + ball) and also gives you binary gate via idx_gate
    return idx_phys, tau_player, idx_gate, tau_joint, nx, ny



player_params = pd.read_parquet(config.FEATUREDIR / f"player_params_new.parquet")


import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

_G = {}

def _init_pool(df_play, player_params, ball_quantile_fn,
               R, step, T_max, dt_tau, q_low, q_high, require_within_interval):
    _G["df_play"] = df_play
    _G["player_params"] = player_params
    _G["ball_quantile_fn"] = ball_quantile_fn
    _G["step"] = step; _G["T_max"] = T_max; _G["dt_tau"] = dt_tau
    _G["q_low"] = q_low; _G["q_high"] = q_high
    _G["require_within_interval"] = require_within_interval

def _reach_task_small(task):
    frame_id, pid, row_dict, qb_pos_list = task
    row = pd.Series(row_dict)

    idx_phys, tau_player, idx_gate, tau_joint, nx, ny = _compute_reachability_for_row_sparse(
        _G["df_play"], _G["player_params"], row, np.asarray(qb_pos_list), _G["ball_quantile_fn"],
        step=_G["step"], T_max=_G["T_max"], dt_tau=_G["dt_tau"],
        q_low=_G["q_low"], q_high=_G["q_high"],
        require_within_interval=_G["require_within_interval"]
    )
    return frame_id, pid, idx_phys, tau_player, idx_gate, tau_joint, nx, ny


import numpy as np

def build_nfl_field_layer(
    x_min=0.0, x_max=120.0,
    y_min=0.0, y_max=53.3,
    endzone_depth=10.0,
    field_fill="rgba(144,238,144,0.5)",
    endzone_fill="rgba(0,0,0,0.10)",
    line_color="white",
    yard_every=5,
    major_every=10,
    add_numbers=True,
    add_hashmarks=True,
    hash_every=1,
    hash_half_len=0.45,      # make them visible
    number_y_offset=4.0,
):
    shapes = []
    annotations = []

    # Field (goes first!)
    shapes.append(dict(
        type="rect",
        x0=x_min, x1=x_max, y0=y_min, y1=y_max,
        line=dict(color="black", width=2),
        fillcolor=field_fill,
        layer="below",
    ))

    # Endzones (still below)
    shapes.append(dict(
        type="rect",
        x0=x_min, x1=x_min + endzone_depth, y0=y_min, y1=y_max,
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor=endzone_fill,
        layer="below",
    ))
    shapes.append(dict(
        type="rect",
        x0=x_max - endzone_depth, x1=x_max, y0=y_min, y1=y_max,
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor=endzone_fill,
        layer="below",
    ))

    # Yard lines only in the playing field
    x_play0 = x_min + endzone_depth
    x_play1 = x_max - endzone_depth

    # Goal lines a bit thicker
    for gx in (x_play0, x_play1):
        shapes.append(dict(
            type="line", x0=gx, x1=gx, y0=y_min, y1=y_max,
            line=dict(color=line_color, width=3.0),
            layer="below",
        ))

    for x in np.arange(x_play0, x_play1 + 1e-9, yard_every):
        is_major = (int(round(x - x_play0)) % major_every == 0)
        shapes.append(dict(
            type="line",
            x0=float(x), x1=float(x),
            y0=y_min, y1=y_max,
            line=dict(color=line_color, width=(2.2 if is_major else 1.1)),
            layer="below",
        ))

    # Hash marks (NFL)
    if add_hashmarks:
        y_hash1 = y_min + 23.583
        y_hash2 = y_max - 23.583
        for x in np.arange(x_play0 + 1.0, x_play1, hash_every):
            for yh in (y_hash1, y_hash2):
                shapes.append(dict(
                    type="line",
                    x0=float(x), x1=float(x),
                    y0=float(yh - hash_half_len), y1=float(yh + hash_half_len),
                    line=dict(color=line_color, width=2),
                    layer="below",
                ))

    # Numbers
    if add_numbers:
        for x in np.arange(x_play0 + 10.0, x_play1, 10.0):
            num = int(min(x - x_play0, x_play1 - x))  # 10..50..10
            if num <= 0:
                continue
            # bottom
            annotations.append(dict(
                x=float(x), y=float(y_min + number_y_offset),
                text=str(num),
                showarrow=False,
                font=dict(size=18, color=line_color),
                xanchor="center", yanchor="bottom",
            ))
            # top (rotated 180°)
            annotations.append(dict(
                x=float(x), y=float(y_max - number_y_offset),
                text=str(num),
                showarrow=False,
                textangle=180,
                font=dict(size=18, color=line_color),
                xanchor="center", yanchor="top",
            ))

    return shapes, annotations



import io, base64

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
def contour_png_dataurl(z, x_vec, y_vec, *,
                        vmin, vmax, levels=20,
                        dpi=140, alpha=0.8):
    # z: (ny, nx)
    fig = Figure(figsize=(7.5, 3.3), dpi=dpi)
    FigureCanvas(fig)  # Agg canvas (schnell, keine Notebook-Render-Overhead)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    cs = ax.contourf(
        x_vec, y_vec, z,
        levels=np.linspace(vmin, vmax, levels),
        cmap=RDBU_MPL,
        vmin=vmin, vmax=vmax,
        antialiased=True
    )

    # <- das ist der wichtige Fix:
    try:
        cs.set_alpha(alpha)  # funktioniert breit über Versionen
    except Exception:
        for c in getattr(cs, "collections", []):
            c.set_alpha(alpha)

    ax.set_xlim(x_vec[0], x_vec[-1])
    ax.set_ylim(y_vec[0], y_vec[-1])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


import plotly.graph_objects as go
from plotly.offline import iplot
from tqdm import tqdm

FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3

def make_play_trajectories_with_reachability_iplot(
    df_play: pd.DataFrame,
    df_targets: pd.DataFrame,
    player_params: pd.DataFrame,
    ball_quantile_fn,
    qb_nfl_id=None,
    R: float = 20.0,
    step: float = 1.0,
    T_max: float = 2.8,
    dt_tau: float = 0.05,
    q_low: float = 0.05,
    q_high: float = 0.995,
    require_within_interval: bool = True,
    future_opacity: float = 0.18,
    fill_alpha_offense: float = 0.18,
    fill_alpha_defense: float = 0.8,
    defense_on_top: bool = True,
    max_workers: int | None = None,
    use_multiprocessing = False,
    timeToCatchThreshold: float = 0.0,
):
    if df_play.empty:
        raise ValueError("df_play is empty")

    df_play = df_play.sort_values(["frame_id", "nfl_id"]).copy()
    if df_play[["game_id", "play_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("df_play must contain exactly one (game_id, play_id).")

    if df_targets.empty:
        raise ValueError("df_targets is empty")

    if df_targets[["game_id", "play_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("df_targets must contain exactly one (game_id, play_id).")

    gid_p = int(df_play["game_id"].iloc[0])
    pid_p = int(df_play["play_id"].iloc[0])

    gid_t = int(df_targets["game_id"].iloc[0])
    pid_t = int(df_targets["play_id"].iloc[0])

    if (gid_t, pid_t) != (gid_p, pid_p):
        raise ValueError(f"df_targets play mismatch: df_play={(gid_p,pid_p)} vs df_targets={(gid_t,pid_t)}")


    t0 = df_targets.iloc[0]

    home = str(t0.get("home_team", "HOME"))
    away = str(t0.get("away_team", "AWAY"))

    home_score = t0.get("total_home_score", None)
    away_score = t0.get("total_away_score", None)
    home_score = int(home_score) if pd.notna(home_score) else 0
    away_score = int(away_score) if pd.notna(away_score) else 0
    homeOffense = bool(t0.get("defteam") != home) if "defteam" in t0 else None
    down = t0.get("down", None)
    down = int(down) if pd.notna(down) else None

    qtr_val = t0.get("qtr", None)
    qtr_val = int(qtr_val) if pd.notna(qtr_val) else None

    # game clock: prefer "time" (usually like '12:34'), else derive from quarter_seconds_remaining
    clock_str = t0.get("time", None)
    if pd.isna(clock_str) or clock_str is None or str(clock_str).strip() == "":
        qsec = t0.get("quarter_seconds_remaining", None)
        clock_str = _sec_to_mmss(qsec) if pd.notna(qsec) else None
    clock_str = str(clock_str) if clock_str is not None else "??:??"

    ydstogo = t0.get("ydstogo", None)
    ydstogo = int(ydstogo) if pd.notna(ydstogo) else None

    yardline_100 = t0.get("yardline_100", None)
    yardline_100 = int(yardline_100) if pd.notna(yardline_100) else None

    goal_to_go = bool(t0.get("goal_to_go", False))
    complete = bool(t0.get("complete_pass", False))

    def _distance_line():
        parts = []
        if ydstogo is not None:
            parts.append(f"To 1st: {ydstogo} yd")
        if yardline_100 is not None and (goal_to_go or yardline_100 <= 15):
            parts.append(f"To TD: {yardline_100} yd")
        return " • ".join(parts) if parts else "Distance: n/a"

    def _title_text(fr_int: int, offA: float, risk: float, pot_gain: float, homeOffense: bool, down: int):
        # offense = blue, defense = red
        if homeOffense is None:
            home_color = "black"
            away_color = "black"
        else:
            home_color = "royalblue" if homeOffense else "crimson"
            away_color = "crimson" if homeOffense else "royalblue"

        line1 = (
            f"<span style='color:{home_color}'><b>{home}</b> {home_score}</span>"
            f" <b>vs</b> "
            f"<span style='color:{away_color}'>{away_score} <b>{away}</b></span>"
        )
        line2 = (
            f"Q{qtr_val if qtr_val is not None else '?'}"
            f" • {clock_str}"
            f" • {down}{'st' if down==1 else 'nd' if down==2 else 'rd' if down==3 else 'th'} down"
            f" • {_distance_line()}"
            f" • Complete: {'✅' if complete else '❌'}"
        )
        line3 = (
            f"Risk: {risk:.3f}"
            f" • Potential gain: {pot_gain:.1f} yd"
            f" • Free offense area: {offA:.1f} yd²"
        )
        return f"{line1}<br>{line2}<br>{line3}"




    frame_ids = np.sort(df_play["frame_id"].unique())
    initial_frame = int(frame_ids[0])

    # ---- meta + player list ----
    meta_cols = ["nfl_id", "player_name", "is_offense", "is_defense", "is_passer", "is_target_receiver"]
    for c in meta_cols:
        if c not in df_play.columns:
            df_play[c] = False if c.startswith("is_") else ""

    players_meta = df_play[meta_cols].drop_duplicates("nfl_id").set_index("nfl_id")
    player_ids = players_meta.index.to_list()

    # ---- player params for hover ----
    param_cols = ["v_max", "a_fwd", "a_back", "a_lat"]
    pp = player_params
    if "nfl_id" in pp.columns and pp.index.name != "nfl_id":
        pp = pp.set_index("nfl_id")

    player_param_vals = {}
    for pid in player_ids:
        if pid in pp.index:
            r = pp.loc[pid]
            player_param_vals[pid] = [float(r.get(c, np.nan)) for c in param_cols]
        else:
            player_param_vals[pid] = [np.nan] * 4

    def _player_color(is_offense: bool, is_defense: bool, is_passer: bool, is_target: bool) -> str:
        if is_target: return "black"
        if is_passer: return "orange"
        if is_offense and not is_defense: return "royalblue"
        if is_defense and not is_offense: return "crimson"
        return "gray"

    # ---- trajectories ----
    traj = {}
    for pid in player_ids:
        d = df_play[df_play["nfl_id"] == pid].sort_values("frame_id")
        col = _player_color(bool(players_meta.loc[pid,"is_offense"]),
                            bool(players_meta.loc[pid,"is_defense"]),
                            bool(players_meta.loc[pid,"is_passer"]),
                            bool(players_meta.loc[pid, "is_target_receiver"]))
        traj[pid] = dict(
            frame_id=d["frame_id"].to_numpy(),
            x=d["x"].to_numpy(),
            y=d["y"].to_numpy(),
            s=d["s"].to_numpy(),
            a=d["a"].to_numpy(),
            name=str(d["player_name"].iloc[0]) if "player_name" in d else str(pid),
            color=col,
            is_passer=bool(players_meta.loc[pid,"is_passer"]),
            is_target_receiver=bool(players_meta.loc[pid, "is_target_receiver"])
        )

    def _split(pid, frame_id):
        t = traj[pid]
        frames = t["frame_id"]
        k = np.searchsorted(frames, frame_id, side="right")

        xp, yp = t["x"][:k].tolist(), t["y"][:k].tolist()
        sp, ap = t["s"][:k].tolist(), t["a"][:k].tolist()

        xf, yf = t["x"][k:].tolist(), t["y"][k:].tolist()
        sf, af = t["s"][k:].tolist(), t["a"][k:].tolist()

        vmax, afwd, aback, alat = player_param_vals[pid]
        cp = [[s, a, vmax, afwd, aback, alat] for s, a in zip(sp, ap)]
        cf = [[s, a, vmax, afwd, aback, alat] for s, a in zip(sf, af)]
        name = t["name"]
        if t["is_target_receiver"]:
            name += " (target)"
        tp = [name] * len(xp)
        tf = [name] * len(xf)
        return (xp, yp, cp, tp), (xf, yf, cf, tf)

    # ---- helpers ----
    def _get_qb_pos(frame_id):
        return np.array(
            get_qb_pos_from_df(
                df_play,
                game_id=df_play["game_id"].iloc[0],
                play_id=df_play["play_id"].iloc[0],
                frame_id=frame_id,
                qb_nfl_id=qb_nfl_id,
            ),
            dtype=float
        )

    def _get_row(pid, frame_id):
        r = df_play[(df_play["nfl_id"] == pid) & (df_play["frame_id"] == frame_id)]
        return None if r.empty else r.iloc[0]

    # ---- LOS / ball landing ----
    los_x = float(df_play["absolute_yardline_number"].iloc[0]) if "absolute_yardline_number" in df_play.columns else None
    ball_land_pt = None
    if "ball_land_x" in df_play.columns and "ball_land_y" in df_play.columns:
        bx = df_play["ball_land_x"].dropna()
        by = df_play["ball_land_y"].dropna()
        if len(bx) and len(by):
            ball_land_pt = (float(bx.iloc[0]), float(by.iloc[0]))

    # ============================================================
    # PRECOMPUTE reachability in parallel
    # ============================================================
    tasks = []
    for fr in frame_ids:
        qb_pos = _get_qb_pos(int(fr))
        for pid in player_ids:
            if traj[pid]["is_passer"]:
                continue
            row = _get_row(pid, int(fr))
            if row is None:
                continue
            tasks.append((int(fr), int(pid), row.to_dict(), qb_pos.tolist()))

    reach_cache = {}

    if not use_multiprocessing:
        # single process (often faster on Kaggle than spawning lots of workers)
        for fr, pid, row_dict, qb_pos_list in tqdm(tasks, desc="reachability (single)"):
            row = pd.Series(row_dict)
            idx_phys, tau_player, idx_gate, tau_joint, nx, ny = _compute_reachability_for_row_sparse(
                df_play, player_params, row, np.asarray(qb_pos_list), ball_quantile_fn,
                step=step, T_max=T_max, dt_tau=dt_tau, q_low=q_low, q_high=q_high,
                require_within_interval=require_within_interval
            )
            reach_cache[(fr, pid)] = (idx_phys, tau_player, idx_gate, tau_joint, nx, ny)


    else:
        workers = max_workers or None
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_pool,
            initargs=(df_play, player_params, ball_quantile_fn, R, step, T_max, dt_tau, q_low, q_high, require_within_interval)
        ) as ex:
            # lower overhead than submit+as_completed
            for fr, pid, idx_phys, tau_player, idx_gate, tau_joint, nx, ny in tqdm(
                ex.map(_reach_task_small, tasks, chunksize=50),
                total=len(tasks),
                desc="reachability (mp)"
            ):
                reach_cache[(fr, pid)] = (idx_phys, tau_player, idx_gate, tau_joint, nx, ny)



    def _compute_global_bounds_sparse(
        df_play: pd.DataFrame,
        reach_cache: dict,
        *,
        X_MIN: float,
        X_MAX: float,
        Y_MIN: float,
        Y_MAX: float,
        step: float,
        pad: float = 2.0,
    ):
        # trajectories bounds
        xmin = float(df_play["x"].min())
        xmax = float(df_play["x"].max())
        ymin = float(df_play["y"].min())
        ymax = float(df_play["y"].max())

        # global lattice size (trust the passed global definition)
        nx = int(np.floor((X_MAX - X_MIN) / step + 1e-9)) + 1
        ny = int(np.floor((Y_MAX - Y_MIN) / step + 1e-9)) + 1
        n = nx * ny

        # reachability bounds (decode idx_gate -> x,y)
        for (_fr, _pid), (idx_phys, tau_player, idx_gate, tau_joint, _nx, _ny) in reach_cache.items():
            if idx_gate is None or len(idx_gate) == 0:
                continue

            idx = np.asarray(idx_gate, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < n)]
            if idx.size == 0:
                continue

            i = idx % nx
            j = idx // nx

            xs = X_MIN + i.astype(np.float64) * step
            ys = Y_MIN + j.astype(np.float64) * step

            xmin = min(xmin, float(xs.min()))
            xmax = max(xmax, float(xs.max()))
            ymin = min(ymin, float(ys.min()))
            ymax = max(ymax, float(ys.max()))

        return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)


    x0, x1, y0, y1 = _compute_global_bounds_sparse(
        df_play,
        reach_cache,
        X_MIN=FIELD_X_MIN,
        X_MAX=FIELD_X_MAX,
        Y_MIN=FIELD_Y_MIN,
        Y_MAX=FIELD_Y_MAX,
        step=step,
        pad=2.0,
    )


    # ============================================================
    # Build figure
    # ============================================================
    fig = go.Figure()
    dyn_indices = []

    hover_tmpl = (
        "<b>%{text}</b><br>"
        "x=%{x:.2f}<br>"
        "y=%{y:.2f}<br>"
        "s=%{customdata[0]:.2f}<br>"
        "a=%{customdata[1]:.2f}<br>"
        "<br><b>player_params</b><br>"
        "v_max=%{customdata[2]:.2f}<br>"
        "a_fwd=%{customdata[3]:.2f}<br>"
        "a_back=%{customdata[4]:.2f}<br>"
        "a_lat=%{customdata[5]:.2f}"
        "<extra></extra>"
    )

    # trajectories (past+future)
    for pid in player_ids:
        (xp,yp,cp,tp), (xf,yf,cf,tf) = _split(pid, initial_frame)
        col = traj[pid]["color"]

        fig.add_trace(go.Scatter(
            x=xp, y=yp, mode="lines+markers",
            line=dict(width=2, color=col), marker=dict(size=6, color=col),
            name=traj[pid]["name"], showlegend=True, legendgroup=str(pid),
            customdata=cp, text=tp, hovertemplate=hover_tmpl,
        ))
        dyn_indices.append(len(fig.data)-1)

        fig.add_trace(go.Scatter(
            x=xf, y=yf, mode="lines+markers",
            line=dict(width=2, color=col), marker=dict(size=5, color=col),
            opacity=future_opacity,
            showlegend=False, legendgroup=str(pid),
            customdata=cf, text=tf, hovertemplate=hover_tmpl,
        ))
        dyn_indices.append(len(fig.data)-1)

    if ball_land_pt is not None:
        fig.add_trace(go.Scatter(
            x=[ball_land_pt[0]], y=[ball_land_pt[1]],
            mode="markers", marker=dict(size=12, symbol="x", color="magenta"),
            name="Ball landing", showlegend=True,
            hovertemplate="Ball landing<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        ))

    # trajectory during ball flight
    req = {"nfl_id", "frame_id", "x", "y"}
    missing = req - set(df_targets.columns)
    if missing:
        raise ValueError(f"df_targets missing required trajectory columns: {missing}")

    df_tt = df_targets.dropna(subset=["nfl_id", "frame_id", "x", "y"]).copy()
    df_tt["nfl_id"] = df_tt["nfl_id"].astype(int)
    df_tt["frame_id"] = df_tt["frame_id"].astype(int)


    df_tt = df_tt.sort_values(["nfl_id", "frame_id"])

    gray = "rgba(120,120,120,0.85)"
    for pid2, g in df_tt.groupby("nfl_id", sort=False):
        xs = g["x"].to_list()
        ys = g["y"].to_list()

        nm = players_meta.loc[pid2, "player_name"] if pid2 in players_meta.index else str(pid2)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            line=dict(width=2, color=gray, dash="dot"),
            marker=dict(size=5, color=gray),
            name=f"{nm} (targets)",
            showlegend=False,
            hoverinfo="skip",
        ))

    # reachability traces (fill + line) – ordering so defense is on top
    items_off, items_def, items_oth = [], [], []
    reach_order = []      # list of pids in the order we add reach traces
    reach_idx = {}        # pid -> (fill_idx, line_idx)

    (_, _), (_idx_phys, _tau_player, _idx_gate, _tau_joint, nx, ny) = next(iter(reach_cache.items()))
    x_vec = np.arange(nx, dtype=float) * step
    y_vec = np.arange(ny, dtype=float) * step

    
    ####
    def build_offense_advantage_map(frame_id: int, *, max_margin: float = 2.0):
        offenseGrid = []
        defenseGrid = []

        # NEW: keep each offense player's tau grid
        offense_tau_by_pid = {}

        # build per-team tau grids (inf = unreachable)
        for pid in player_ids:
            row = _get_row(pid, frame_id)
            if traj[pid]["is_passer"] or row is None or (frame_id, pid) not in reach_cache:
                continue

            idx_phys, tau_player, idx_gate, tau_joint, nx, ny = reach_cache[(frame_id, pid)]

            ztau_flat = np.full(nx * ny, np.inf, dtype=np.float32)
            ztau_flat[idx_gate] = tau_joint
            ztau = ztau_flat.reshape(ny, nx)

            if bool(players_meta.loc[pid, "is_defense"]):
                defenseGrid.append(ztau)
            elif bool(players_meta.loc[pid, "is_offense"]):
                offenseGrid.append(ztau)
                offense_tau_by_pid[pid] = ztau  # NEW

        if len(offenseGrid) == 0 or len(defenseGrid) == 0:
            score = np.full((ny, nx), np.nan, dtype=np.float32)
            return score, 0.0, 0.0, {}  # NEW
        
        area_interest = 2 # yards around ball landing point
        if ball_land_pt is not None:
            x_min_interest = max(FIELD_X_MIN, ball_land_pt[0] - area_interest // 2)
            x_max_interest = min(FIELD_X_MAX, ball_land_pt[0] + area_interest // 2)
            y_min_interest = max(FIELD_Y_MIN, ball_land_pt[1] - area_interest // 2)
            y_max_interest = min(FIELD_Y_MAX, ball_land_pt[1] + area_interest // 2)

            i_min = int(np.rint((x_min_interest - FIELD_X_MIN) / step))
            i_max = int(np.rint((x_max_interest - FIELD_X_MIN) / step))
            j_min = int(np.rint((y_min_interest - FIELD_Y_MIN) / step))
            j_max = int(np.rint((y_max_interest - FIELD_Y_MIN) / step))


            # Also adjust ball landing indices
            i0 = int(np.rint((ball_land_pt[0] - x_min_interest) / step))
            j0 = int(np.rint((ball_land_pt[1] - y_min_interest) / step))
        else:
            print("No ball landing point provided; using full field for advantage map.")



        offense_min = np.min(np.stack(offenseGrid), axis=0)
        defense_min = np.min(np.stack(defenseGrid), axis=0)

        off_reach = np.isfinite(offense_min)
        def_reach = np.isfinite(defense_min)

        both = off_reach & def_reach
        def_only = def_reach & ~off_reach
        off_only = off_reach & ~def_reach

        # raw advantage (positive good for offense)
        adv_raw = defense_min - offense_min

        score = np.full_like(adv_raw, np.nan, dtype=np.float32)
        
        score[def_only] = -max_margin
        score[off_only] = +max_margin

        adv_use = adv_raw - timeToCatchThreshold
        # ball land area
        score[both] = np.clip(adv_use[both], -max_margin, +max_margin).astype(np.float32)
        time_score_offense = np.sum(np.nan_to_num(score[j_min:j_max, i_min:i_max], nan=0.0))

        offensArea = float(np.sum(score > 0.0) * (step ** 2))
        defenseArea = float(np.sum(score <= 0.0) * (step ** 2))

        freedom_by_pid = {}
        for pid, tau_off in offense_tau_by_pid.items():
            off_r = np.isfinite(tau_off)
            if not np.any(off_r):
                freedom_by_pid[pid] = float("nan")
                continue

            both_p = off_r & def_reach
            off_only_p = off_r & ~def_reach

            adv_use_p = (defense_min - tau_off) - timeToCatchThreshold  # per-player adv_use

            # per-cell freedom: offense-only => 1, both => clip(adv_use/max_margin, 0..1)
            freedom_cell = np.zeros_like(tau_off, dtype=np.float32)
            freedom_cell[off_only_p] = 1.0
            freedom_cell[both_p] = np.clip(adv_use_p[both_p] / max_margin, 0.0, 1.0).astype(np.float32)

            # aggregate to a single percentage-like value per player
            freedom_by_pid[pid] = float(np.mean(freedom_cell[off_r]))

            potentialForwardGainOffense = ball_land_pt[0] - row["absolute_yardline_number"] if row["play_direction"]== "right" else row["absolute_yardline_number"] - ball_land_pt[0]

        return score, offensArea, defenseArea, freedom_by_pid, time_score_offense, potentialForwardGainOffense


    #####
    metrics = {"frame_id": [], "offensArea": [], "defenseArea": [], "freedom_by_pid": [], "riskyBall": None, "potentialForwardGainOffense": None}

    metrics_by_frame = {}  # fr -> (offA, defA)
    adv_by_frame = {}  # optional if you also need the grid later
    img_by_frame = {}
    for fr in frame_ids.astype(int):
        adv, offensArea, defenseArea, freedom_by_pid, ball_offense_hit, potentialForwardGainOffense = build_offense_advantage_map(fr, max_margin=T_max/4)
        metrics["frame_id"].append(fr)
        metrics["offensArea"].append(offensArea)
        metrics["defenseArea"].append(defenseArea)
        metrics["freedom_by_pid"].append(sum(list(freedom_by_pid.values())) / len(freedom_by_pid) if freedom_by_pid else float(0.0))
        metrics_by_frame[fr] = (offensArea, defenseArea)
        adv_by_frame[fr] = adv  # optional
        img_by_frame[fr] = contour_png_dataurl(adv_by_frame[fr], x_vec, y_vec, vmin=-T_max/4, vmax=T_max/4, levels=100)
    metrics["riskyBall"] = [float(ball_offense_hit)]
    metrics["potentialForwardGainOffense"] = [float(potentialForwardGainOffense)]
    metrics["game_id"] = [gid_p]
    metrics["play_id"] = [pid_p]
    metrics["complete_pass"] = [bool(df_targets['complete_pass'].iloc[0])]
    risk_val = float(metrics["riskyBall"][0])
    pot_gain_val = float(metrics["potentialForwardGainOffense"][0])


    adv0 = adv_by_frame[initial_frame]  # (ny, nx)

    z = adv0.astype(float)
    z = np.where(np.isfinite(z), z, None).tolist()  # None => gaps (like NaN)
    x = x_vec.astype(float).tolist()
    y = y_vec.astype(float).tolist()
    dummHEatMap = go.Heatmap(
        x=x, y=y, z=z,
        colorscale=RDBU,
        zmin=-T_max/4, zmax=+T_max/4,
        zsmooth="best",          # smooth interpolation
        opacity=0.0,
        showscale=True,
        colorbar=dict(
            x=1.0,                # <= IMPORTANT: keep it inside
            xanchor="left",
            thickness=18,
            len=0.9,
            title="Advantage Offense<br><span style='font-size:10px'>in seconds</span>",
            tickmode="array",
            tickvals=[-T_max/4, +T_max/4],
            ticktext=["Disadvantage (-0.7s)", "Advantage (+0.7s)"],
            ),
        hoverinfo="skip",
        zmid=0.0, 
        )
    fig.add_trace(dummHEatMap)

    fig.update_layout(
        images=[dict(
            source=img_by_frame[initial_frame],
            xref="x", yref="y",
            x=x_vec[0], y=y_vec[-1],
            sizex=(x_vec[-1] - x_vec[0]),
            sizey=(y_vec[-1] - y_vec[0]),
            sizing="stretch",
            layer="below",
            opacity=1.0
        )],
        margin=dict(l=20, r=260, t=90, b=120), 
        legend=dict(x=0.01, xanchor="left", y=0.99, yanchor="top")
    )


    # shapes + layout
    field_shapes, field_ann = build_nfl_field_layer(
    x_min=FIELD_X_MIN, x_max=FIELD_X_MAX,
    y_min=FIELD_Y_MIN, y_max=FIELD_Y_MAX,
    )
    los_shape = None
    if los_x is not None:
        los_shape = dict(
            type="line", x0=los_x, x1=los_x, y0=FIELD_Y_MIN, y1=FIELD_Y_MAX,
            line=dict(color="green", width=2, dash="dash"),
            layer="above",
        )

    all_shapes = field_shapes + ([los_shape] if los_shape else [])


    # where is the data located inside [x0, x1]?
    data_min = float(df_play["x"].min())
    data_max = float(df_play["x"].max())

    gap_left  = data_min - x0
    gap_right = x1 - data_max

    legend_on_right = gap_right >= gap_left

    existing_ann = list(fig.layout.annotations) if fig.layout.annotations else []

    PAD = 2.0
    x0 = max(x0, FIELD_X_MIN - PAD)
    x1 = min(x1, FIELD_X_MAX + PAD)
    y0 = max(y0, FIELD_Y_MIN - PAD)
    y1 = min(y1, FIELD_Y_MAX + PAD)

    offA, defA = metrics_by_frame[initial_frame]
    fig.update_layout(
        title=dict(
            text=_title_text(int(initial_frame), offA, risk_val, pot_gain_val, homeOffense, down),
            x=0.5
        ),

        xaxis=dict(range=[x0, x1], title="x", showgrid=False, zeroline=False),
        yaxis=dict(range=[y0, y1], title="y", showgrid=False, zeroline=False,
                scaleanchor="x", scaleratio=1),

        shapes=all_shapes,
        annotations=existing_ann + field_ann,

        autosize=True,
        height=900,
        margin=dict(l=20, r=120, t=90, b=20),

        legend=dict(
            orientation="v",
            y=1.0, yanchor="top",
            x=(1.0 if legend_on_right else 0.0),
            xanchor=("right" if legend_on_right else "left"),
            bgcolor="rgba(255,255,255,0.7)",
        ),

        plot_bgcolor="rgba(230,230,230,1)",
        paper_bgcolor="rgba(230,230,230,1)",
    )


    # ============================================================
    # Frames (use cached reachability, only update fill+line x/y/z)
    # ============================================================
    frames = []
    for fr in tqdm(frame_ids, desc="building frames"):
        updates = []

        # trajectories
        for pid in player_ids:
            (xp,yp,cp,tp), (xf,yf,cf,tf) = _split(pid, int(fr))
            updates.append(go.Scatter(x=xp, y=yp, customdata=cp, text=tp))
            updates.append(go.Scatter(x=xf, y=yf, customdata=cf, text=tf))

        offA, defA = metrics_by_frame[int(fr)]
        frames.append(go.Frame(
        name=str(int(fr)),            # <<< IMPORTANT
        data=updates,
        traces=dyn_indices,
        layout=go.Layout(
            images=[dict(
                source=img_by_frame[fr],
                xref="x", yref="y",
                x=x_vec[0], y=y_vec[-1],
                sizex=(x_vec[-1] - x_vec[0]),
                sizey=(y_vec[-1] - y_vec[0]),
                sizing="stretch",
                layer="below",
                opacity=1.0
            )],
            title=dict(
                text=_title_text(int(fr), offA, risk_val, pot_gain_val, homeOffense, down),
                x=0.5
            )
            )
        ))
    fig.frames = frames

    frame_names = [str(int(fr)) for fr in frame_ids]

    steps = [
        dict(
            method="animate",
            args=[
                [nm],  # <- single target frame
                dict(mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0))
            ],
            label=nm,
        )
        for nm in frame_names
    ]

    play_ms = 200  # speed: milliseconds per frame
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="frame_id="),
            pad=dict(t=30),
            steps=steps,
        )],

        # --- THIS is what px adds for Play/Pause ---
        updatemenus=[dict(
            type="buttons",
            direction="left",
            showactive=True,          # shows pressed state
            active=-1,
            x=0.0,
            y=0.05,                   # <-- move up (tweak 0.08..0.18)
            xanchor="left",
            yanchor="bottom",
            pad=dict(r=10, t=10),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[frame_names, dict(
                        mode="immediate",
                        fromcurrent=True,
                        frame=dict(duration=play_ms, redraw=True),
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    )],
                )
            ],
        )],
    )

    return fig, metrics



def transformAllPlaysFromOriginCenterLOSAndPlayDirectionRightToGlobalNEW(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inverse of transformAllPlaysToOriginCenterLOSAndPlayDirectionRight.
    Since we use precomputed features from the prediciton competion we need to invert
    those transformations to get back to the original global frame.

    Input frame (current):
      - Origin at LOS center: x=0 on LOS, y=0 at field midline (26.65).
      - All plays move toward +X (left-moving plays were rotated 180°).
      - Angles ('dir','o') are in math convention, radians (0 along +X, CCW).

    Output frame (global/raw tracking-like):
      - x,y back in original global coordinates.
      - If 'dir'/'o' present, converted back to original tracking convention:
            degrees with 0° along +Y, clockwise.
      - If 'ball_land_x'/'ball_land_y' present, also inverted.

    Assumes 'absolute_yardline_number' is the LOS x-location in the raw frame.
    """
    FIELD_WIDTH = 53.3
    MID_Y = FIELD_WIDTH / 2.0  # 26.65

    out = df.copy()

    # mask for plays that were point-reflected in the forward transform
    left_mask = out['play_direction'].astype(str).str.lower().eq('left')

    # 1) Undo the 180° rotation (point reflection) for left-moving plays
    out.loc[left_mask, ['x', 'y']] *= -1.0
    if 'ball_land_x' in out.columns:
        out.loc[left_mask, 'ball_land_x'] *= -1.0
    if 'ball_land_y' in out.columns:
        out.loc[left_mask, 'ball_land_y'] *= -1.0

    # Angles: subtract π for left plays (inverse of "+ π" from the forward pass)
    for col in ('dir', 'o'):
        if col in out.columns:
            out.loc[left_mask, col] = (out.loc[left_mask, col] - np.pi) % (2 * np.pi)

    # 2) Undo the translation: add LOS x back and re-center y to field coordinates
    out['x'] = out['x'] + out['absolute_yardline_number']
    out['y'] = out['y'] + MID_Y

    if 'ball_land_x' in out.columns:
        out['ball_land_x'] = out['ball_land_x'] + out['absolute_yardline_number']
    if 'ball_land_y' in out.columns:
        out['ball_land_y'] = out['ball_land_y'] + MID_Y

    return out


import random

df_visual = df.copy()
df_visual = transformAllPlaysFromOriginCenterLOSAndPlayDirectionRightToGlobalNEW(df_visual)
df_targets_visual = df_targets.copy()
df_targets_visual = transformAllPlaysFromOriginCenterLOSAndPlayDirectionRightToGlobalNEW(df_targets_visual)


gb_train = df_visual.groupby(["game_id", "play_id"], sort=False)
gb_target = df_targets_visual.groupby(["game_id", "play_id"], sort=False)

print(df_visual.columns)

print(len(gb_train))


#pick a random play
group_names = list(gb_train.groups.keys())
random_group_name = random.choice(group_names)

# random_group_name = (2018091612, 760)
# random_group_name = (2018090880, 3233)

print(random_group_name)
# random_group_name = (2023112603, 1209)
# Removed plays: {(2023091100, 3167), (2023122100, 1450), (2023091711, 4627)}
random_play_train = gb_train.get_group(random_group_name)
random_play_target = gb_target.get_group(random_group_name)


ui, metrics = make_play_trajectories_with_reachability_iplot(
    df_play=random_play_train,                 # one play only
    df_targets=random_play_target,
    player_params=player_params,     # indexed by nfl_id; contains v_max,a_fwd,a_back,a_lat
    ball_quantile_fn=ball_q_fn,    # your ball time quantile model
    qb_nfl_id=None,                  # or set QB id if you want
    R=30.0,
    step=0.4,                        # increase (e.g. 1.0) if it’s heavy
    use_multiprocessing=(not ARE_WE_IN_KAGGLE),
)


# ui.layout = widgets.Layout(width="100%")
if ARE_WE_IN_KAGGLE:
    ui.show()  
else:
    out_path = Path("/home/tom/projects/NFL_2026/AnalysisResults/figures") / f"{random_play_train['game_id'].iloc[0]}_{random_play_train['play_id'].iloc[0]}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ui.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)

    display(ui)



fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 6 * len(metrics)))

# If only one metric, axes won't be a list
if len(metrics) == 1:
    axes = [axes]

for i, (metric_name, values) in enumerate(metrics.items()):
    axes[i].plot(values, marker='o', linewidth=2, markersize=6)
    axes[i].set_title(f'{metric_name} over time', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Frame')
    axes[i].set_ylabel(metric_name)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(0, len(values)-1)

plt.tight_layout()
plt.show()

