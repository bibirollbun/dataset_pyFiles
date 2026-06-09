import os

os.makedirs("/kaggle/working/code1_main", exist_ok=True)
os.makedirs("/kaggle/working/code2_3d_pe", exist_ok=True)
os.makedirs("/kaggle/working/code3_atd", exist_ok=True)

print(os.listdir("/kaggle/working"))



import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as widgets
import itertools
from IPython.display import display, clear_output, HTML, Javascript

# --- Colab widget support ---
# try:
#     from google.colab import output as colab_output
#     colab_output.enable_custom_widget_manager()
# except Exception:
#     pass

pio.renderers.default = "iframe"
print("Renderer set to:", pio.renderers.default)

# ---------- CONFIG: CHANGE PATHS IF NEEDED ----------
INPUT_FILE  = "/kaggle/input/nfl-analytics/combined_input_2023.csv"
OUTPUT_FILE = "/kaggle/input/nfl-analytics/combined_output_2023.csv"
SUPP_FILE   = "/kaggle/input/nfl-analytics/supplementary_data.csv"
# ---------- CSS STYLES FOR KPI DASHBOARD & PLAYER RANKS ----------
# REDESIGNED UI: Dark "Cyber" Theme with Neon Accents
KPI_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');

    :root {
        --bg-dark: #0b0f19;
        --panel-bg: #151b2b;
        --card-bg: #1e2538;
        --text-main: #f1f5f9;
        --text-sub: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-gold: #f59e0b;
        --neon-border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* GENERAL PANEL STYLES */
    .kpi-panel, .rank-panel, .explain-panel {
        background-color: var(--panel-bg);
        color: var(--text-main);
        padding: 20px;
        border-radius: 16px;
        font-family: 'Roboto', sans-serif;
        height: 600px;
        overflow-y: auto;
        border: var(--neon-border);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }

    /* HEADERS */
    .panel-header {
        font-size: 1.1em;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-main);
        margin-bottom: 4px;
        border-bottom: 2px solid #334155;
        padding-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .panel-sub {
        font-size: 0.8em;
        color: var(--text-sub);
        margin-bottom: 15px;
        font-weight: 400;
    }

    /* KPI CARDS */
    .kpi-card {
        background: linear-gradient(145deg, #1e2538, #171d2d);
        border: 1px solid rgba(255,255,255,0.05);
        border-left: 4px solid var(--accent-blue);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .kpi-title {
        font-size: 0.75em;
        font-weight: 700;
        color: var(--text-sub);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.6em;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    }
    .kpi-desc {
        font-size: 0.8em;
        color: #64748b;
        margin-top: 4px;
        line-height: 1.3;
    }
    .kpi-simple {
        font-size: 0.75em;
        color: var(--accent-blue);
        margin-top: 6px;
        font-style: italic;
        padding-top: 6px;
        border-top: 1px dashed rgba(255, 255, 255, 0.1);
    }

    /* KPI ACCENTS */
    .card-red { border-left-color: var(--accent-red); }
    .card-green { border-left-color: var(--accent-green); }
    .card-gold { border-left-color: var(--accent-gold); }

    /* PLAYER RANKINGS */
    .mom-card {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #fff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);
        position: relative;
        overflow: hidden;
    }
    .mom-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
        transform: rotate(30deg);
    }
    .mom-badge {
        background: rgba(0,0,0,0.3);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7em;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 10px;
    }
    .mom-name {
        font-size: 1.6em;
        font-weight: 900;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 5px;
    }
    .mom-stat {
        font-size: 1.1em;
        font-weight: 700;
        opacity: 0.95;
    }

    .rank-section-header {
        font-size: 0.85em;
        color: var(--text-sub);
        text-transform: uppercase;
        font-weight: 700;
        margin: 20px 0 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        border-bottom: 1px solid #334155;
        padding-bottom: 5px;
    }

    .player-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255,255,255,0.03);
        padding: 10px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        border-left: 3px solid transparent;
        transition: background 0.2s;
    }
    .player-row:hover { background: rgba(255,255,255,0.06); }
    .row-best { border-left-color: var(--accent-green); }
    .row-worst { border-left-color: var(--accent-red); }

    .p-rank { font-size: 0.8em; color: #64748b; width: 20px; }
    .p-name { font-weight: 600; color: #e2e8f0; font-size: 0.9em; flex-grow: 1; }
    .p-score { font-family: 'Roboto Mono', monospace; font-weight: 700; color: #cbd5e1; }

    /* EXPLANATION / COACHES NOTE */
    .explain-container {
        background: #ffffff;
        color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        font-family: 'Roboto', sans-serif;
    }
    .explain-title {
        font-size: 1.2em;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 15px;
        border-bottom: 3px solid #f59e0b;
        display: inline-block;
        padding-bottom: 2px;
    }
    .section-box {
        background: #f8fafc;
        border-left: 4px solid #cbd5e1;
        padding: 12px 16px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
    }
    .section-header {
        font-weight: 700;
        color: #334155;
        margin-bottom: 8px;
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .bullet-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .bullet-item {
        position: relative;
        padding-left: 18px;
        margin-bottom: 8px;
        font-size: 0.9em;
        line-height: 1.5;
        color: #475569;
    }
    .bullet-item::before {
        content: 'â€¢';
        position: absolute;
        left: 0;
        color: #f59e0b;
        font-weight: bold;
    }
    .sub-bullet-list {
        list-style: none;
        padding-left: 10px;
        margin-top: 4px;
        border-left: 2px solid #e2e8f0;
        margin-left: 5px;
    }
    .sub-bullet-item {
        font-size: 0.85em;
        margin-bottom: 4px;
        padding-left: 10px;
        color: #64748b;
    }
</style>
"""

# ---------- GLOBALS ----------
input_df  = pd.DataFrame()
output_df = pd.DataFrame()
supp_df   = pd.DataFrame()
_df       = pd.DataFrame()    # for week/game/play selectors


# ---------- LOAD DATA ----------
def load_all():
    global input_df, output_df, supp_df, _df

    # ---- Input (pre-throw) ----
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        input_df = pd.DataFrame()
    else:
        inp = pd.read_csv(INPUT_FILE, low_memory=False)

        col_map = {
            "game_id": "gameId",
            "play_id": "playId",
            "frame_id": "frameId",
            "player_name": "displayName",
            "player_side": "teamName",
            "nfl_id": "nflId",
            "week": "week",
        }
        inp.rename(columns=col_map, inplace=True)

        for c in ["x", "y", "s", "dir", "frameId", "ball_land_x", "ball_land_y", "week"]:
            if c in inp.columns:
                inp[c] = pd.to_numeric(inp[c], errors="coerce")

        if {"x", "y", "frameId"}.issubset(inp.columns):
            inp = inp.dropna(subset=["x", "y", "frameId"])

        input_df = inp
        _df = inp.copy()
        print("Input rows:", input_df.shape[0])

    # ---- Output (post-throw) ----
    if not os.path.exists(OUTPUT_FILE):
        print(f"Output file not found: {OUTPUT_FILE}")
        output_df = pd.DataFrame()
    else:
        out = pd.read_csv(OUTPUT_FILE, low_memory=False)

        col_map_out = {
            "game_id": "gameId",
            "play_id": "playId",
            "frame_id": "frameId",
            "nfl_id": "nflId",
            "week": "week",
        }
        out.rename(columns=col_map_out, inplace=True)

        for c in ["x", "y", "frameId", "week"]:
            if c in out.columns:
                # Use a temporary variable 'cast' to avoid multiple evaluation of out[c]
                out[c] = pd.to_numeric(out[c], errors="coerce")

        if {"x", "y", "frameId"}.issubset(out.columns):
            out = out.dropna(subset=["x", "y", "frameId"])

        output_df = out
        print("Output rows:", output_df.shape[0])

    # ---- Supplementary ----
    if os.path.exists(SUPP_FILE):
        s = pd.read_csv(SUPP_FILE, low_memory=False)
        s.rename(columns={"game_id": "gameId", "play_id": "PlayId", "play_id": "playId"}, inplace=True)
        if "PlayId" in s.columns and "playId" in s.columns:
            s.drop(columns=["PlayId"], inplace=True)
        supp_df = s
        print("Supplementary rows:", supp_df.shape[0])
    else:
        print(f"Supplementary not found: {SUPP_FILE}")
        supp_df = pd.DataFrame()


load_all()


# ---------- GOALPOSTS (UPDATED) ----------
def add_goalposts(fig, x_pos, y_center=26.65, z_crossbar=3.33, z_top=13.33, width=6.17, color="#facc15"):
    """
    Adds a 'Slingshot' style NFL goalpost to the 3D Plotly figure.
    """
    hw = width / 2.0  # Half width

    # 1. Base Support
    fig.add_trace(go.Scatter3d(
        x=[x_pos, x_pos], y=[y_center, y_center], z=[0, z_crossbar],
        mode="lines", line=dict(color=color, width=12),
        hoverinfo="text", text="Goal Post Base", showlegend=False
    ))
    # 2. Crossbar
    fig.add_trace(go.Scatter3d(
        x=[x_pos, x_pos], y=[y_center - hw, y_center + hw], z=[z_crossbar, z_crossbar],
        mode="lines", line=dict(color=color, width=12), hoverinfo="none", showlegend=False
    ))
    # 3. Left Upright
    fig.add_trace(go.Scatter3d(
        x=[x_pos, x_pos], y=[y_center - hw, y_center - hw], z=[z_crossbar, z_top],
        mode="lines", line=dict(color=color, width=10), hoverinfo="none", showlegend=False
    ))
    # 4. Right Upright
    fig.add_trace(go.Scatter3d(
        x=[x_pos, x_pos], y=[y_center + hw, y_center + hw], z=[z_crossbar, z_top],
        mode="lines", line=dict(color=color, width=10), hoverinfo="none", showlegend=False
    ))

# ---------- NEW: QB STICK FIGURE VISUALIZER ----------
def add_qb_stick_figure(fig, x, y, land_x, land_y, color="#fbbf24"):
    """
    Adds a static 3D stick figure representing the QB throwing the ball.
    """
    if x is None or y is None or land_x is None or land_y is None:
        return

    # Calculate direction vector to face the throw
    dx = land_x - x
    dy = land_y - y
    norm = np.hypot(dx, dy)
    if norm > 0:
        ux, uy = dx/norm, dy/norm
    else:
        ux, uy = 1, 0 # Default facing X positive

    # Perpendicular vector for width (shoulders/hips)
    px, py = -uy, ux

    # Scale factors for body parts
    shoulder_w = 0.3
    arm_len = 0.4
    leg_spread = 0.3

    # --- COORDINATES ---
    # Center Base: (x, y, 0)
    # Hips: (x, y, 1.0)
    # Shoulders: (x, y, 1.7)
    # Head: (x, y, 1.9)

    # 1. Body Line (Hips to Shoulders)
    body_x = [x, x]
    body_y = [y, y]
    body_z = [1.0, 1.7]

    # 2. Left Leg (Back/Balance)
    ll_x = [x, x - px * leg_spread * 0.5 - ux * 0.2]
    ll_y = [y, y - py * leg_spread * 0.5 - uy * 0.2]
    ll_z = [1.0, 0.0]

    # 3. Right Leg (Forward/Step)
    rl_x = [x, x + px * leg_spread * 0.5 + ux * 0.3]
    rl_y = [y, y + py * leg_spread * 0.5 + uy * 0.3]
    rl_z = [1.0, 0.0]

    # 4. Right Arm (Throwing - Up and Forward)
    ra_x = [x + px * shoulder_w, x + px * shoulder_w + ux * arm_len]
    ra_y = [y + py * shoulder_w, y + py * shoulder_w + uy * arm_len]
    ra_z = [1.7, 2.1]

    # 5. Left Arm (Balance)
    la_x = [x - px * shoulder_w, x - px * shoulder_w - ux * 0.2]
    la_y = [y - py * shoulder_w, y - py * shoulder_w - uy * 0.2]
    la_z = [1.7, 1.4]

    # Combine all lines into one trace for efficiency (using None to break lines)
    # Order: Body, L-Leg, R-Leg, R-Arm, L-Arm
    all_x = body_x + [None] + ll_x + [None] + rl_x + [None] + ra_x + [None] + la_x
    all_y = body_y + [None] + ll_y + [None] + rl_y + [None] + ra_y + [None] + la_y
    all_z = body_z + [None] + ll_z + [None] + rl_z + [None] + ra_z + [None] + la_z

    fig.add_trace(go.Scatter3d(
        x=all_x, y=all_y, z=all_z,
        mode="lines", line=dict(color=color, width=5),
        name="QB Throwing", showlegend=False
    ))

    # Head (Sphere/Marker)
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[1.9],
        mode="markers", marker=dict(size=6, color=color, line=dict(color='black', width=1)),
        name="QB Head", showlegend=False
    ))


# ---------- 3D PATH EFFICIENCY HELPERS ----------

def cubic_bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)
    t = t.reshape(-1, 1)
    B = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    return B


def compute_3d_pe(p_track, elev_z=2.0, bezier_points=200):
    if p_track is None or p_track.empty:
        return None, None, None, None

    # Use first and last actual player positions as endpoints
    p0_xy = np.array([float(p_track["x"].iloc[0]), float(p_track["y"].iloc[0])])
    p3_xy = np.array([float(p_track["x"].iloc[-1]), float(p_track["y"].iloc[-1])])

    if np.allclose(p0_xy, p3_xy):
        return 0.0, 0.0, 0.0, np.array([[p0_xy[0], p0_xy[1], 0.0]])

    vec = p3_xy - p0_xy
    perp = np.array([-vec[1], vec[0]])
    if np.linalg.norm(perp) > 0:
        perp = perp / np.linalg.norm(perp)
    else:
        perp = np.array([0.0, 0.0])

    p1_xy = p0_xy + 0.25 * vec + 0.1 * np.linalg.norm(vec) * perp
    p2_xy = p0_xy + 0.75 * vec + 0.1 * np.linalg.norm(vec) * perp

    p0 = np.array([p0_xy[0], p0_xy[1], 0.0])
    p1 = np.array([p1_xy[0], p1_xy[1], elev_z])
    p2 = np.array([p2_xy[0], p2_xy[1], elev_z])
    p3 = np.array([p3_xy[0], p3_xy[1], 0.0])

    ideal_pts = cubic_bezier(p0, p1, p2, p3, n=bezier_points)

    direct_3d_distance = float(np.linalg.norm(p3 - p0))

    coords = np.vstack(
        [
            p_track["x"].astype(float).values,
            p_track["y"].astype(float).values,
            np.zeros(len(p_track)),
        ]
    ).T

    if coords.shape[0] < 2:
        actual_route_length = 0.0
    else:
        deltas = coords[1:] - coords[:-1]
        dists = np.sqrt((deltas**2).sum(axis=1))
        actual_route_length = float(dists.sum())

    if actual_route_length <= 0:
        pe = 0.0
    else:
        pe = direct_3d_distance / actual_route_length
        pe = float(np.clip(pe, 0.0, 2.0))

    return direct_3d_distance, actual_route_length, pe, ideal_pts


# ---------- GAME-LEVEL BEST DEFENDER LOGIC (NEW) ----------
def get_game_best_defender(game_id, full_data):
    d = full_data[full_data["gameId"] == game_id].copy()
    if d.empty: return None

    if "player_role" not in d.columns: return None

    wr_df = d[d["player_role"] == "Targeted Receiver"][["playId", "nflId"]].drop_duplicates()

    defender_scores = {}
    defender_names = {}

    for _, row in wr_df.iterrows():
        play_id = row["playId"]
        wr_id = row["nflId"]

        play_data = d[d["playId"] == play_id]
        wr_track = play_data[play_data["nflId"] == wr_id][["frameId", "x", "y"]]
        if wr_track.empty: continue

        def_ids = play_data[play_data["teamName"] == "Defense"]["nflId"].unique()

        for did in def_ids:
            def_track = play_data[play_data["nflId"] == did][["frameId", "x", "y", "displayName"]]
            if def_track.empty: continue

            if did not in defender_names:
                defender_names[did] = def_track["displayName"].iloc[0]

            merged = wr_track.merge(def_track, on="frameId", suffixes=("_wr", "_def"))
            if merged.empty: continue

            dist = np.sqrt((merged["x_wr"] - merged["x_def"])**2 + (merged["y_wr"] - merged["y_def"])**2)
            avg_dist = dist.mean()

            if did not in defender_scores: defender_scores[did] = []
            defender_scores[did].append(avg_dist)

    if not defender_scores: return None

    final_scores = {did: np.mean(vals) for did, vals in defender_scores.items()}
    best_did = min(final_scores, key=final_scores.get)

    return {
        "nflId": best_did,
        "name": defender_names.get(best_did, str(best_did)),
        "avg_dist": final_scores[best_did]
    }

# ---------- GAME-LEVEL WORST DEFENDER LOGIC (NEW) ----------
def get_game_worst_defender(game_id, full_data):
    d = full_data[full_data["gameId"] == game_id].copy()
    if d.empty: return None

    if "player_role" not in d.columns: return None

    wr_df = d[d["player_role"] == "Targeted Receiver"][["playId", "nflId"]].drop_duplicates()

    defender_scores = {}
    defender_names = {}

    for _, row in wr_df.iterrows():
        play_id = row["playId"]
        wr_id = row["nflId"]

        play_data = d[d["playId"] == play_id]
        wr_track = play_data[play_data["nflId"] == wr_id][["frameId", "x", "y"]]
        if wr_track.empty: continue

        def_ids = play_data[play_data["teamName"] == "Defense"]["nflId"].unique()

        for did in def_ids:
            def_track = play_data[play_data["nflId"] == did][["frameId", "x", "y", "displayName"]]
            if def_track.empty: continue

            if did not in defender_names:
                defender_names[did] = def_track["displayName"].iloc[0]

            merged = wr_track.merge(def_track, on="frameId", suffixes=("_wr", "_def"))
            if merged.empty: continue

            dist = np.sqrt((merged["x_wr"] - merged["x_def"])**2 + (merged["y_wr"] - merged["y_def"])**2)
            avg_dist = dist.mean()

            if did not in defender_scores: defender_scores[did] = []
            defender_scores[did].append(avg_dist)

    if not defender_scores: return None

    final_scores = {did: np.mean(vals) for did, vals in defender_scores.items()}
    worst_did = max(final_scores, key=final_scores.get)

    return {
        "nflId": worst_did,
        "name": defender_names.get(worst_did, str(worst_did)),
        "avg_dist": final_scores[worst_did]
    }

# ---------- GAME-LEVEL SEPARATION SCORE (GSS) LOGIC ----------
def compute_game_gss(game_id, full_data):
    if full_data.empty: return None

    d = full_data[full_data["gameId"] == game_id].copy()
    if d.empty: return None

    if "player_role" not in d.columns: return None

    wr_game = d[d["player_role"] == "Targeted Receiver"][["playId", "nflId"]].drop_duplicates()

    team_separation_data = {}
    team_player_scores = {}

    for _, row in wr_game.iterrows():
        ply_id = row["playId"]
        wr_id = row["nflId"]

        play_data = d[d["playId"] == ply_id]
        if play_data.empty: continue

        if "possessionTeam" in play_data.columns:
            poss_team = play_data["possessionTeam"].iloc[0]
        elif "teamName" in play_data.columns:
            wr_info = play_data[play_data["nflId"] == wr_id]
            if not wr_info.empty:
                poss_team = wr_info["teamName"].iloc[0]
            else:
                poss_team = "Unknown"
        else:
            poss_team = "Unknown"

        if poss_team not in team_separation_data:
            team_separation_data[poss_team] = []
            team_player_scores[poss_team] = {}

        wr_track = play_data[play_data["nflId"] == wr_id][["frameId", "x", "y", "displayName"]]
        if wr_track.empty: continue

        wr_name = wr_track["displayName"].iloc[0] if "displayName" in wr_track.columns else str(wr_id)

        fr_last = wr_track["frameId"].max()
        wr_last = wr_track[wr_track["frameId"] == fr_last]
        if wr_last.empty: continue

        wx, wy = float(wr_last["x"].iloc[0]), float(wr_last["y"].iloc[0])

        def_last = play_data[
            (play_data["teamName"] == "Defense") &
            (play_data["frameId"] == fr_last)
        ][["x", "y"]]

        if def_last.empty: continue

        dists = np.hypot(def_last["x"] - wx, def_last["y"] - wy)

        if len(dists) > 0:
            sep_val = float(dists.min())
            team_separation_data[poss_team].append(sep_val)

            if wr_name not in team_player_scores[poss_team]:
                team_player_scores[poss_team][wr_name] = []
            team_player_scores[poss_team][wr_name].append(sep_val)

    if not team_separation_data:
        return None

    results = {}

    for team, scores in team_separation_data.items():
        avg_sep = round(float(np.mean(scores)), 2)

        best_wr_name = "N/A"
        best_wr_val = 0.0

        p_scores = team_player_scores.get(team, {})
        if p_scores:
            p_avgs = {p: np.mean(v) for p, v in p_scores.items()}
            best_wr_name = max(p_avgs, key=p_avgs.get)
            best_wr_val = round(p_avgs[best_wr_name], 2)

        if avg_sep < 1.5: txt = "Tight"
        elif avg_sep < 3.0: txt = "Competitive"
        elif avg_sep < 5.0: txt = "Open"
        else: txt = "Explosive"

        results[team] = {
            "score": avg_sep,
            "text": txt,
            "top_wr": best_wr_name,
            "top_val": best_wr_val
        }

    return results


# ---------- KPI COMPUTATION HELPERS ----------

def _select_targeted_receiver(full_df, frames):
    if full_df.empty:
        return None, None

    if "player_role" in full_df.columns:
        wr = full_df[full_df["player_role"] == "Targeted Receiver"]
        if not wr.empty:
            wr_id = wr["nflId"].iloc[0]
            wr_name = (
                wr["displayName"].dropna().iloc[0]
                if "displayName" in wr.columns and wr["displayName"].notna().any()
                else str(wr_id)
            )
            return wr_id, wr_name

    if "player_side" in full_df.columns:
        off = full_df[full_df["player_side"] == "Offense"]
    else:
        off = full_df[full_df.get("teamName", "") == "Offense"]

    if off.empty:
        return None, None

    last_frame = max(frames)
    off_last = off[off["frameId"] == last_frame]
    if off_last.empty:
        off_last = off

    off_last = off_last.sort_values("x", ascending=False)
    wr_id = off_last["nflId"].iloc[0]
    wr_name = (
        off_last["displayName"].dropna().iloc[0]
        if "displayName" in off_last.columns and off_last["displayName"].notna().any()
        else str(wr_id)
    )
    return wr_id, wr_name

def nearest_defender_at_frame(x0, y0, frame_id, all_play_data):
    def_df = all_play_data[
        (all_play_data["teamName"] == "Defense") &
        (all_play_data["frameId"] == frame_id)
    ][["displayName","x","y"]]

    if def_df.empty:
        return "No Defender", np.inf

    def_df["dist"] = np.sqrt((def_df["x"] - x0)**2 + (def_df["y"] - y0)**2)

    if def_df["dist"].empty:
        return "No Defender", np.inf

    row = def_df.loc[def_df["dist"].idxmin()]
    return row["displayName"], float(row["dist"])

# --- ABSR HELPER FUNCTIONS ---

def get_ball_parabola(start_x, start_y, land_x, land_y, n_frames):
    if n_frames <= 0:
        return np.array([]), np.array([]), np.array([])
    t = np.linspace(0.0, 1.0, n_frames)
    ball_x = start_x + (land_x - start_x) * t
    ball_y = start_y + (land_y - start_y) * t
    ball_z = 12.0 * 4.0 * t * (1.0 - t)
    return ball_x, ball_y, ball_z

def compute_absr_for_play_single(play_df, start_x, start_y, land_x, land_y, frames, defender_prefix='def'):
    if play_df is None or play_df.empty or None in (start_x, start_y, land_x, land_y, frames):
        return None

    ball_x, ball_y, ball_z = get_ball_parabola(float(start_x), float(start_y), float(land_x), float(land_y), len(frames))

    if not (len(ball_x) == len(ball_y) == len(ball_z) == len(frames)):
        return None

    df = play_df.copy()

    if 'frameId' not in df.columns or 'x' not in df.columns or 'y' not in df.columns or 'teamName' not in df.columns:
        return None
    df['team_lc'] = df['teamName'].astype(str).str.lower().fillna('')

    frame_defs = {}
    for fr, grp in df.groupby('frameId'):
        defs = grp[grp['team_lc'].str.startswith(defender_prefix.lower())]

        if defs.empty:
            defs = grp[grp['team_lc'].str.contains(r'\bd', na=False)]

        if not defs.empty:
            frame_defs[int(fr)] = defs[['x', 'y']].astype(float).values
        else:
            frame_defs[int(fr)] = np.empty((0, 2))

    min_dists = []
    defender_z = 0.9

    for i, fr in enumerate(frames):
        try:
            fr_int = int(fr)
        except Exception:
            fr_int = fr

        defs_xy = frame_defs.get(fr_int)

        if defs_xy is None or defs_xy.size == 0:
            continue

        bx, by, bz = float(ball_x[i]), float(ball_y[i]), float(ball_z[i])

        dx = defs_xy[:, 0] - bx
        dy = defs_xy[:, 1] - by
        dz = defender_z - bz

        dists = np.sqrt(dx * dx + dy * dy + dz * dz)

        if dists.size > 0:
            min_dists.append(float(np.min(dists)))

    if not min_dists:
        return None

    return float(np.mean(min_dists))

def calculate_game_absr(game_id, full_game_df):
    game_data = full_game_df[full_game_df["gameId"] == game_id].copy()
    if game_data.empty:
        return None

    absr_scores = []

    plays_with_target = game_data[game_data["player_role"] == "Targeted Receiver"][['playId']].drop_duplicates()

    if plays_with_target.empty:
        return None

    for play_id in plays_with_target['playId'].tolist():
        play_df = game_data[game_data['playId'] == play_id].copy()

        frames = sorted(play_df["frameId"].unique())
        if len(frames) < 2: continue

        qb_track = play_df[play_df["player_role"] == "Passer"].sort_values("frameId")
        if not qb_track.empty:
            qb_last = qb_track.iloc[-1]
            start_x = float(qb_last["x"])
            start_y = float(qb_last["y"])
        else:
            fr0 = play_df[play_df["frameId"] == frames[0]]
            off0 = fr0[fr0["teamName"] == "Offense"]
            start_x = off0["x"].mean() if not off0.empty else fr0["x"].mean()
            start_y = off0["y"].mean() if not off0.empty else fr0["y"].mean()

        land_x = play_df["ball_land_x"].dropna().iloc[0] if play_df["ball_land_x"].notna().any() else None
        land_y = play_df["ball_land_y"].dropna().iloc[0] if play_df["ball_land_y"].notna().any() else None

        if land_x is None or land_y is None or start_x is None or start_y is None:
            continue

        absr_play_val = compute_absr_for_play_single(
            play_df,
            start_x=start_x, start_y=start_y,
            land_x=land_x, land_y=land_y,
            frames=frames
        )
        if absr_play_val is not None:
            absr_scores.append(absr_play_val)

    if not absr_scores:
        return None

    game_absr = np.mean(absr_scores)
    return float(game_absr)


def absr_broadcast_message(absr_value):
    if absr_value is None:
        return {
            "display": "N/A",
            "interpretation": "ABSR not available",
            "broadcast": "ABSR not computed for this game. Data missing."
        }

    disp = f"{absr_value:.2f} yds"

    if absr_value >= 3.0:
        interp = "Safe and precise (High Safety Margin)"
        broadcast = f"This QB has an ABSR of {absr_value:.2f} yards â€” exhibiting safe and precise passing habits."
    elif absr_value >= 1.5:
        interp = "Moderate safety â€” requires caution"
        broadcast = f"This QB has an ABSR of {absr_value:.2f} yards â€” moderate safety; keep an eye on tight windows."
    else:
        interp = "Risky and interception-prone (Low Safety Margin)"
        broadcast = f"This QB has an ABSR of only {absr_value:.2f} yards â€” heâ€™s throwing into danger all night."

    return {"display": disp, "interpretation": interp, "broadcast": broadcast}


# --- NEW BFSG CALCULATION ---
def compute_bfsg_metrics(play_df, wr_track):
    if wr_track.shape[0] < 2:
        return None

    release_row = wr_track.iloc[0]
    release_frame = int(release_row["frameId"])
    rel_x, rel_y = float(release_row["x"]), float(release_row["y"])

    arrival_row = wr_track.iloc[-1]
    arrival_frame = int(arrival_row["frameId"])
    arr_x, arr_y = float(arrival_row["x"]), float(arrival_row["y"])

    if np.isnan(rel_x) or np.isnan(rel_y) or np.isnan(arr_x) or np.isnan(arr_y):
        return None

    rel_def_name, rel_sep = nearest_defender_at_frame(rel_x, rel_y, release_frame, play_df)
    arr_def_name, arr_sep = nearest_defender_at_frame(arr_x, arr_y, arrival_frame, play_df)

    if rel_sep == np.inf or arr_sep == np.inf:
        return None

    bfsg = arr_sep - rel_sep

    if bfsg >= 2.0:
        rating = "Elite Gain"
        commentary = "The receiver created massive separation during the ball's flight, indicative of a perfect route adjustment or defensive lapse."
    elif bfsg >= 1.0:
        rating = "Significant Gain"
        commentary = "A notable gain in separation, making the catch much easier for the receiver."
    elif bfsg >= 0.5:
        rating = "Mild Gain"
        commentary = "The receiver maintained or slightly improved separation, a solid outcome."
    elif bfsg >= 0.0:
        rating = "Maintained"
        commentary = "No significant change in separation. The catch window remained constant."
    else:
        rating = "Lost Ground"
        commentary = "The defender closed in faster than the receiver could pull away. Very tight coverage at the catch point."

    return {
        "bfsg": round(bfsg, 2),
        "rating": rating,
        "rel_sep": round(rel_sep, 2),
        "arr_sep": round(arr_sep, 2),
        "commentary": commentary
    }


def compute_dcsi_metrics(wr_track, defenders):
    if wr_track.empty or defenders.empty:
        return None

    def get_separation(x0, y0, frame_id, def_df):
        ddf = def_df[def_df["frameId"] == frame_id]

        if ddf.empty:
            return None, np.inf

        dist = np.sqrt((ddf["x"] - x0)**2 + (ddf["y"] - y0)**2)
        if dist.empty:
            return None, np.inf
        min_idx = dist.idxmin()
        return ddf.loc[min_idx, "displayName"], dist.min()

    if "dir" not in wr_track.columns or wr_track["dir"].isna().all():
        return None

    dir_change = wr_track["dir"].diff().abs()
    if dir_change.isna().all():
        return None

    bp_idx = dir_change.idxmax()
    bp_row = wr_track.loc[bp_idx]
    bp_frame = bp_row["frameId"]
    bp_x, bp_y = bp_row["x"], bp_row["y"]

    wr_last = wr_track.iloc[-1]
    af_frame = wr_last["frameId"]
    af_x, af_y = wr_last["x"], wr_last["y"]

    time_diff = (af_frame - bp_frame) / 10.0
    if time_diff <= 0:
        return None

    bp_def_name, bp_sep = get_separation(bp_x, bp_y, bp_frame, defenders)
    af_def_name, af_sep = get_separation(af_x, af_y, af_frame, defenders)

    if bp_sep == np.inf or af_sep == np.inf:
        return None

    dcsi = (bp_sep - af_sep) / time_diff

    if dcsi > 1.0:
        rating = "Elite Closing Speed"
    elif dcsi > 0.5:
        rating = "Strong Closing"
    elif dcsi > 0.0:
        rating = "Mild Closing"
    else:
        rating = "Losing Ground"

    return {
        "dcsi": round(dcsi, 2),
        "rating": rating,
        "bp_sep": round(bp_sep, 2),
        "af_sep": round(af_sep, 2),
        "time_diff": round(time_diff, 2),
        "bp_def": bp_def_name,
        "af_def": af_def_name
    }


def compute_kpis(full_df, players_all, frames, ball_x, ball_y, ball_z):
    kpis = {
        "best_defender": {"name": "N/A", "dist": 0.0, "nflId": None},
        "worst_defender": {"name": "N/A", "dist": 0.0, "nflId": None},
        "sqi": 0.0,
        "cds": 0.0,
        "cwd": 0.0,
        "pe": 0.0,
        "pe_direct": 0.0,
        "pe_actual": 0.0,
        "target_wr_name": None,
        "ideal_pts": None,
        "dcsi_data": None,
        "bfsg_data": None,
        "absr_data": None,
    }

    if full_df.empty or players_all.empty or len(frames) == 0:
        return kpis

    wr_id, wr_name = _select_targeted_receiver(full_df, frames)
    if wr_id is None:
        return kpis

    kpis["target_wr_name"] = wr_name

    wr_track = full_df[full_df["nflId"] == wr_id].copy()
    wr_track = wr_track.sort_values("frameId")
    if wr_track[["x", "y"]].isna().all(axis=None):
        return kpis

    direct_d, actual_len, pe_val, ideal_pts = compute_3d_pe(wr_track[["x", "y"]])
    if pe_val is not None:
        kpis["pe"] = round(pe_val, 3)
        kpis["pe_direct"] = round(direct_d, 3) if direct_d is not None else 0.0
        kpis["pe_actual"] = round(actual_len, 3) if actual_len is not None else 0.0
        kpis["ideal_pts"] = ideal_pts

    if "player_side" in full_df.columns:
        defenders = full_df[full_df["player_side"] == "Defense"].copy()
    else:
        defenders = full_df[full_df.get("teamName", "") == "Defense"].copy()

    if defenders.empty:
        return kpis

    defenders = defenders.sort_values("frameId")

    dcsi_res = compute_dcsi_metrics(wr_track, defenders)
    if dcsi_res:
        kpis["dcsi_data"] = dcsi_res

    bfsg_res = compute_bfsg_metrics(full_df, wr_track)
    if bfsg_res:
        kpis["bfsg_data"] = bfsg_res

    best_def_id = None
    best_def_name = None
    best_avg_dist = None

    if "player_role" in defenders.columns:
        coverage_first = defenders[defenders["player_role"] == "Defensive Coverage"]
    else:
        coverage_first = pd.DataFrame()

    use_df = coverage_first if not coverage_first.empty else defenders

    for did, dtrack in use_df.groupby("nflId"):
        dtrack = dtrack.sort_values("frameId")
        merged = wr_track[["frameId", "x", "y"]].merge(
            dtrack[["frameId", "x", "y"]],
            on="frameId",
            suffixes=("_wr", "_def"),
        )
        if merged.empty:
            continue
        d = np.hypot(merged["x_wr"] - merged["x_def"], merged["y_wr"] - merged["y_def"])
        avg_d = float(d.mean())
        if (best_avg_dist is None) or (avg_d < best_avg_dist):
            best_avg_dist = avg_d
            best_def_id = did
            if "displayName" in dtrack.columns and dtrack["displayName"].notna().any():
                best_def_name = dtrack["displayName"].dropna().iloc[0]
            else:
                best_def_name = str(did)

    if best_def_id is not None and best_avg_dist is not None:
        kpis["best_defender"] = {
            "name": best_def_name,
            "dist": round(best_avg_dist, 2),
            "nflId": best_def_id,
        }

    sep_dists = []
    for fr in wr_track["frameId"].unique():
        wr_fr = wr_track[wr_track["frameId"] == fr]
        def_fr = defenders[defenders["frameId"] == fr]
        if wr_fr.empty or def_fr.empty:
            continue
        wx, wy = float(wr_fr["x"].iloc[0]), float(wr_fr["y"].iloc[0])
        dx = def_fr["x"].to_numpy()
        dy = def_fr["y"].to_numpy()
        d = np.hypot(dx - wx, dy - wy)
        d_min = float(d.min())
        sep_dists.append((fr, d_min))

    if sep_dists:
        sep_dists = sorted(sep_dists, key=lambda t: t[0])
        vals = [v for _, v in sep_dists]
        k = max(3, len(vals) // 3)
        sqi_val = float(np.mean(vals[-k:]))
        kpis["sqi"] = round(sqi_val, 2)

    sep_frames = [int(x[0]) for x in sep_dists]
    sep_values = [float(x[1]) for x in sep_dists]

    final_fr = int(wr_track["frameId"].max())
    wr_final = wr_track[wr_track["frameId"] == final_fr]
    def_final = defenders[defenders["frameId"] == final_fr]
    if not wr_final.empty and not def_final.empty:
        wx, wy = float(wr_final["x"].iloc[0]), float(wr_final["y"].iloc[0])
        dx = def_final["x"].to_numpy()
        dy = def_final["y"].to_numpy()
        d_all = np.hypot(dx - wx, dy - wy)
        if len(d_all) > 0:
            d_min = float(d_all.min())
            congestion = int((d_all < 5.0).sum())
            tight_component = max(0.0, 5.0 - d_min)
            cds_raw = tight_component + 1.5 * congestion
            cds_raw = min(cds_raw, 10.0)
            kpis["cds"] = round(cds_raw, 2)

    wr_pos_by_frame = {
        int(fr): (float(row["x"]), float(row["y"]))
        for fr, row in (
            wr_track[["frameId", "x", "y"]]
            .dropna()
            .groupby("frameId")
            .first()
            .iterrows()
        )
    }

    catch_radius = 1.5
    near_flags = []

    ball_x = np.asarray(ball_x)
    ball_y = np.asarray(ball_y)

    if len(ball_x) > 0:
        ball_depth = ball_x - ball_x[0]
        kpis["ball_depth"] = ball_depth.tolist()
    else:
        kpis["ball_depth"] = []

    kpis["ball_height"] = list(ball_z)

    dist_in_window = []

    frame_to_index = {int(f): i for i, f in enumerate(frames)}

    for idx, fr_int in enumerate(sep_frames):
        if fr_int not in wr_pos_by_frame or fr_int not in frame_to_index:
            near_flags.append(False)
            continue
        i = frame_to_index[fr_int]
        wx, wy = wr_pos_by_frame[fr_int]
        bx, by = float(ball_x[i]), float(ball_y[i])
        dist_ball = float(np.hypot(bx - wx, by - wy))
        is_near = dist_ball <= catch_radius
        near_flags.append(is_near)
        if is_near:
            dist_in_window.append(sep_values[idx])

    kpis["near_flags"] = near_flags

    if wr_pos_by_frame and len(ball_x) > 0:
        last_wr_frame = max(wr_pos_by_frame.keys())
        wx, wy = wr_pos_by_frame[last_wr_frame]

        if last_wr_frame in frame_to_index:
            i_final = frame_to_index[last_wr_frame]
        else:
            i_final = len(ball_x) - 1

        bx, by = float(ball_x[i_final]), float(ball_y[i_final])
        dist_final = float(np.hypot(bx - wx, by - wy))

        kpis["cwd"] = round(dist_final, 2)

        wr_name_df = full_df[full_df["nflId"] == wr_id]["displayName"]
        wr_name = wr_name_df.iloc[0] if not wr_name_df.empty else "Unknown"
        kpis["cwd_wr"] = wr_name

    else:
        kpis["cwd"] = 0.0
        kpis["cwd_wr"] = "Unknown"

    return kpis

# ---------- NEW: ABBREVIATE NAME HELPER ----------
def shorten_name(name):
    if not isinstance(name, str):
        return ""
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return name

# ---------- MAIN 3D BROADCAST ANIMATION ----------
def create_3d_broadcast_animation(game_id, play_id):
    """
    Returns:
      fig  : Plotly 3D Figure
      kpis : dict with KPI values (see compute_kpis)
      player_ranks: list of dicts with player performance stats
    """
    global input_df, output_df, supp_df

    empty_kpi = {
        "best_defender": {"name": "N/A", "dist": 0.0, "nflId": None},
        "worst_defender": {"name": "N/A", "dist": 0.0, "nflId": None},
        "sqi": 0.0,
        "cds": 0.0,
        "cwd": 0.0,
        "pe": 0.0,
        "pe_direct": 0.0,
        "pe_actual": 0.0,
        "target_wr_name": None,
        "ideal_pts": None,
        "gss_data": None,
        "dcsi_data": None,
        "bfsg_data": None,
        "absr_data": None,
    }

    if input_df.empty and output_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data loaded", showarrow=False)
        return fig, empty_kpi, []

    inp_play = input_df[(input_df["gameId"] == game_id) & (input_df["playId"] == play_id)].copy()
    out_play = output_df[(output_df["gameId"] == game_id) & (output_df["playId"] == play_id)].copy()

    if inp_play.empty and out_play.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for this Game/Play", showarrow=False)
        return fig, empty_kpi, []

    # ---- Ball landing ----
    land_x = None
    land_y = None

    if "ball_land_x" in inp_play.columns and inp_play["ball_land_x"].notna().any():
        land_x = float(inp_play["ball_land_x"].dropna().iloc[0])
    if "ball_land_y" in inp_play.columns and inp_play["ball_land_y"].notna().any():
        land_y = float(inp_play["ball_land_y"].dropna().iloc[0])

    if (land_x is None or land_y is None) and not supp_df.empty:
        sp = supp_df[(supp_df["gameId"] == game_id) & (supp_df["playId"] == play_id)]
        if not sp.empty:
            if "ball_land_x" in sp.columns and sp["ball_land_x"].notna().any():
                land_x = float(sp["ball_land_x"].dropna().iloc[0])
            if "ball_land_y" in sp.columns and sp["ball_land_y"].notna().any():
                land_y = float(sp["ball_land_y"].dropna().iloc[0])

    if land_x is None:
        land_x = float(inp_play["x"].mean() if not inp_play.empty else out_play["x"].mean())
    if land_y is None:
        land_y = float(inp_play["y"].mean() if not inp_play.empty else out_play["y"].mean())

    # ---- Build per-player full trajectory ----
    players_in = set(inp_play["nflId"].unique()) if not inp_play.empty else set()
    players_out = set(out_play["nflId"].unique()) if not out_play.empty else set()
    all_players = sorted(players_in | players_out)

    info = {}
    if not inp_play.empty:
        tmp = inp_play[["nflId", "displayName", "teamName"]].drop_duplicates()
        info = {
            pid: {"name": row["displayName"], "team": row["teamName"]}
            for pid, row in tmp.set_index("nflId").iterrows()
        }

    full_tracks = []
    for pid in all_players:
        pin = inp_play[inp_play["nflId"] == pid].copy()
        pout = out_play[out_play["nflId"] == pid].copy()

        for df_ in (pin, pout):
            for col in ["x", "y", "frameId", "s"]:
                if col in df_.columns:
                    df_[col] = pd.to_numeric(df_[col], errors="coerce")

        if not pin.empty:
            pin = pin.sort_values("frameId")
        if not pout.empty:
            pout = pout.sort_values("frameId")

        if (not pin.empty) and (not pout.empty):
            max_in = pin["frameId"].max()
            pout["frameId"] = pout["frameId"] + max_in

        if not pout.empty:
            if "displayName" not in pout.columns or pout["displayName"].isna().all():
                name = info.get(pid, {}).get("name", None)
                pout["displayName"] = name
            if "teamName" not in pout.columns or pout["teamName"].isna().all():
                team = info.get(pid, {}).get("team", None)
                pout["teamName"] = team

        p_full = pd.concat([pin, pout], ignore_index=True)
        if not p_full.empty:
            full_tracks.append(p_full)

    if not full_tracks:
        fig = go.Figure()
        fig.add_annotation(text="No combined trajectories for this play", showarrow=False)
        return fig, empty_kpi, []

    full_df = pd.concat(full_tracks, ignore_index=True)
    players_all = full_df[["frameId", "nflId", "displayName", "teamName", "x", "y", "s", "dir", "player_role"]].dropna(subset=["x", "y", "frameId"])
    frames = sorted(players_all["frameId"].unique())
    if not frames:
        fig = go.Figure()
        fig.add_annotation(text="No frames after merge", showarrow=False)
        return fig, empty_kpi, []

    # --- FIX: Ensure all players exist in all frames (Forward/Backward Fill) ---
    unique_players = players_all[['nflId', 'displayName', 'teamName']].drop_duplicates(subset=['nflId'])
    scaffold = pd.DataFrame(
        list(itertools.product(unique_players['nflId'], frames)),
        columns=['nflId', 'frameId']
    )
    scaffold = scaffold.merge(unique_players, on='nflId', how='left')
    players_anim = scaffold.merge(
        players_all[['nflId', 'frameId', 'x', 'y']],
        on=['nflId', 'frameId'],
        how='left'
    )
    players_anim['x'] = players_anim.groupby('nflId')['x'].ffill()
    players_anim['y'] = players_anim.groupby('nflId')['y'].ffill()
    players_anim['x'] = players_anim.groupby('nflId')['x'].bfill()
    players_anim['y'] = players_anim.groupby('nflId')['y'].bfill()
    players_anim = players_anim.dropna(subset=['x', 'y'])

    # ---- Ball start & QB ID Capture ----
    start_x = None
    start_y = None
    qb_id = None

    if not inp_play.empty and "player_role" in inp_play.columns:
        qb_track = inp_play[inp_play["player_role"] == "Passer"].sort_values("frameId")
        if not qb_track.empty:
            qb_last = qb_track.iloc[-1]
            qb_id = qb_last["nflId"]
            start_x = float(qb_last["x"])
            start_y = float(qb_last["y"])

    if start_x is None or start_y is None:
        first_frame = frames[0]
        fr0 = players_all[players_all["frameId"] == first_frame]
        off0 = fr0[fr0["teamName"] == "Offense"]
        start_x = off0["x"].mean() if not off0.empty else fr0["x"].mean()
        start_y = off0["y"].mean() if not off0.empty else fr0["y"].mean()

    # ---- FORCE QB POSITION FIX ----
    # Overwrite the QB's position in the animation DataFrame to remain static at the release point
    if qb_id is not None and start_x is not None and start_y is not None:
         players_anim.loc[players_anim["nflId"] == qb_id, "x"] = start_x
         players_anim.loc[players_anim["nflId"] == qb_id, "y"] = start_y

    # ---- CALCULATE PLAYER RANKINGS ----
    player_stats = []
    for pid in full_df["nflId"].unique():
        p_data = full_df[full_df["nflId"] == pid].sort_values("frameId")
        if p_data.empty or "x" not in p_data.columns:
            continue
        p_name = p_data["displayName"].iloc[0] if "displayName" in p_data.columns and p_data["displayName"].notna().any() else str(pid)
        max_s = 0.0
        if "s" in p_data.columns:
            max_s = p_data["s"].max()
        coords = p_data[["x", "y"]].values
        if len(coords) > 1:
            deltas = coords[1:] - coords[:-1]
            dist = np.sum(np.sqrt((deltas**2).sum(axis=1)))
        else:
            dist = 0.0
        if np.isnan(max_s): max_s = 0.0
        if np.isnan(dist): dist = 0.0
        score = (max_s * 1.5) + (dist / 10.0)
        player_stats.append({
            "name": p_name, "score": score, "max_speed": max_s, "dist": dist
        })
    player_ranks = sorted(player_stats, key=lambda x: x["score"], reverse=True)

    t = np.linspace(0, 1, len(frames))
    ball_x = start_x + (land_x - start_x) * t
    ball_y = start_y + (land_y - start_y) * t
    ball_z = 12 * 4 * t * (1 - t)

    # ---- Compute KPIs ----
    kpis = compute_kpis(full_df, players_all, frames, ball_x, ball_y, ball_z)
    game_absr_val = calculate_game_absr(game_id, input_df)
    kpis["absr_data"] = absr_broadcast_message(game_absr_val)

    game_best_def = get_game_best_defender(game_id, input_df)
    if game_best_def:
        kpis["best_defender"] = {
            "name": game_best_def["name"],
            "dist": round(game_best_def["avg_dist"], 2),
            "nflId": game_best_def["nflId"]
        }
        best_def_id = game_best_def["nflId"]
    else:
        best_def_id = kpis["best_defender"]["nflId"]

    game_worst_def = get_game_worst_defender(game_id, input_df)
    if game_worst_def:
        kpis["worst_defender"] = {
            "name": game_worst_def["name"],
            "dist": round(game_worst_def["avg_dist"], 2),
            "nflId": game_worst_def["nflId"]
        }

    gss_data = compute_game_gss(game_id, input_df)
    if gss_data:
        kpis["gss_data"] = gss_data

    target_name = kpis.get("target_wr_name", None)

    # ---- DYNAMIC FIELD BOUNDS ----
    p_min_x = players_all["x"].min()
    p_max_x = players_all["x"].max()
    p_min_y = players_all["y"].min()
    p_max_y = players_all["y"].max()

    x_buffer = 10.0
    y_buffer = 8.0

    min_x = p_min_x - x_buffer
    max_x = p_max_x + x_buffer
    min_y = p_min_y - y_buffer
    max_y = p_max_y + y_buffer

    if min_x < 30: min_x = max(-2, min_x)
    if max_x > 90: max_x = min(122, max_x)

    if max_x - min_x < 40.0:
        center_x = (min_x + max_x) / 2.0
        min_x = center_x - 20.0
        max_x = center_x + 20.0
        min_x = max(-5.0, min_x)
        max_x = min(125.0, max_x)

    min_y = max(-5.0, min_y)
    max_y = min(58.3, max_y)

    fig = go.Figure()

    # ---------- FIELD ----------
    fig.add_trace(go.Mesh3d(
        x=[min_x, min_x, max_x, max_x], y=[min_y, max_y, max_y, min_y], z=[0, 0, 0, 0],
        color="#0f4c3a", opacity=1.0, showlegend=False, lighting=dict(ambient=0.8, diffuse=0.9)
    ))

    start_yard = int(np.floor(min_x / 10.0) * 10)
    end_yard = int(np.ceil(max_x / 10.0) * 10)

    for x_line in range(start_yard, end_yard + 1, 10):
        fig.add_trace(go.Scatter3d(
            x=[x_line, x_line], y=[min_y, max_y], z=[0.05, 0.05],
            mode="lines", line=dict(color="rgba(255,255,255,0.7)", width=2), showlegend=False,
        ))

    fig.add_trace(go.Scatter3d(
        x=[min_x, max_x, max_x, min_x, min_x], y=[min_y, min_y, max_y, max_y, min_y], z=[0.05, 0.05, 0.05, 0.05, 0.05],
        mode="lines", line=dict(color="white", width=4), showlegend=False
    ))

    if min_x <= 0: add_goalposts(fig, 0)
    if max_x >= 120: add_goalposts(fig, 120)

    # ---------- LEGEND ----------
    target_legend_name = f"Target WR Route ({target_name})" if target_name else "Target WR Route"
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color="#facc15", line=dict(color="black", width=1)), name="Offense"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color="#3b82f6", line=dict(color="black", width=1)), name="Defense"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=12, color="rgba(239,68,68,0.3)", line=dict(color="#ef4444", width=4)), name="Best Defender"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=8, color="#f97316", line=dict(color="black", width=1)), name="Ball"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line=dict(color="#fbbf24", width=5), name="QB Release"))
    if target_name:
         fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line=dict(color="#fbbf24", width=4), name=target_legend_name))

    # ---------- STATIC PATHS ----------
    for name in players_all["displayName"].dropna().unique():
        p = players_all[players_all["displayName"] == name].sort_values("frameId")
        side = p["teamName"].iloc[0] if p["teamName"].notna().any() else None
        color = "#facc15" if side == "Offense" else "#3b82f6"
        fig.add_trace(go.Scatter3d(
            x=p["x"], y=p["y"], z=[0.1] * len(p),
            mode="lines", line=dict(color=color, width=5, dash="dot"), showlegend=False, opacity=1.0
        ))

    if target_name is not None and target_name in players_all["displayName"].values:
        wr_path = players_all[players_all["displayName"] == target_name].sort_values("frameId")
        fig.add_trace(go.Scatter3d(
            x=wr_path["x"], y=wr_path["y"], z=[0.2] * len(wr_path),
            mode="lines", line=dict(color="#fbbf24", width=5), name=f"Target: {target_name}", showlegend=False,
        ))

    fig.add_trace(go.Scatter3d(
        x=ball_x, y=ball_y, z=ball_z,
        mode="lines", line=dict(color="#f97316", width=6), showlegend=False, opacity=1.0
    ))

    add_qb_stick_figure(fig, start_x, start_y, land_x, land_y)

    # ---------- ANIMATION FRAMES ----------
    dynamic_trace_start_idx = len(fig.data)
    frames_list = []
    short_names_map = {name: shorten_name(name) for name in players_all["displayName"].dropna().unique()}

    for i, fr in enumerate(frames):
        fr_df = players_anim[players_anim["frameId"] == fr].copy()

        colors = ["#facc15" if t_side == "Offense" else "#3b82f6" for t_side in fr_df["teamName"]]
        fr_df["shortName"] = fr_df["displayName"].map(short_names_map)

        # New Stagger logic (keep consistent with stems)
        stagger_z = [2.5 + (k % 5) * 1.5 for k in range(len(fr_df))]

        # 1. STEMS (Vertical Lines connecting Marker to Text)
        stem_x, stem_y, stem_z = [], [], []

        for k, (idx, row) in enumerate(fr_df.iterrows()):
             # Line from (x, y, 0.5) to (x, y, stagger_z[k])
             # Using None to break lines within one trace
             stem_x.extend([row['x'], row['x'], None])
             stem_y.extend([row['y'], row['y'], None])
             stem_z.extend([0.5, stagger_z[k], None])

        trace_stems = go.Scatter3d(
            x=stem_x, y=stem_y, z=stem_z,
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.3)", width=1), # Thin, faint stem
            hoverinfo="none",
            showlegend=False
        )

        # 2. Markers (Slightly smaller to not clutter)
        trace_markers = go.Scatter3d(
            x=fr_df["x"], y=fr_df["y"], z=[0.5] * len(fr_df),
            mode="markers", marker=dict(size=7, color=colors, line=dict(color="black", width=1)),
            hovertext=fr_df["displayName"], hoverinfo="text", showlegend=False,
        )

        # 3. Labels (Clean text)
        trace_labels = go.Scatter3d(
            x=fr_df["x"], y=fr_df["y"], z=stagger_z,
            mode="text", text=fr_df["shortName"].apply(lambda x: f"<b>{x}</b>"),
            textposition="top center",
            textfont=dict(size=11, color="white", family="Consolas, 'Courier New', monospace"),
            hoverinfo="none", showlegend=False,
        )

        # 4. Ball
        trace_ball = go.Scatter3d(
            x=[float(ball_x[i])], y=[float(ball_y[i])], z=[float(ball_z[i])],
            mode="markers", marker=dict(symbol="circle", size=10, color="#f97316", line=dict(color="black", width=1)),
            showlegend=False,
        )

        data_list = [trace_stems, trace_markers, trace_labels, trace_ball]

        # 4 dynamic traces + optional ring
        current_indices = [
            dynamic_trace_start_idx,
            dynamic_trace_start_idx+1,
            dynamic_trace_start_idx+2,
            dynamic_trace_start_idx+3
        ]

        # 5. Best Defender Ring
        if best_def_id is not None:
            bd_frame = fr_df[fr_df["nflId"] == best_def_id]
            if not bd_frame.empty:
                ring_trace = go.Scatter3d(
                    x=[float(bd_frame["x"].iloc[0])], y=[float(bd_frame["y"].iloc[0])], z=[0.5],
                    mode="markers", marker=dict(size=20, color="rgba(239,68,68,0.25)", line=dict(color="#ef4444", width=4)),
                    showlegend=False,
                )
                data_list.append(ring_trace)
                current_indices.append(dynamic_trace_start_idx+4)

        frames_list.append(go.Frame(data=data_list, name=f"f{fr}", traces=current_indices))

    # --- STATIC INIT TRACES (Pre-animation state) ---
    init_df = players_anim[players_anim["frameId"] == frames[0]].copy()
    init_colors = ["#facc15" if t_side == "Offense" else "#3b82f6" for t_side in init_df["teamName"]]
    init_df["shortName"] = init_df["displayName"].map(short_names_map)
    init_stagger_z = [2.5 + (k % 5) * 1.5 for k in range(len(init_df))]

    # Init Stems
    init_stem_x, init_stem_y, init_stem_z = [], [], []
    for k, (idx, row) in enumerate(init_df.iterrows()):
         init_stem_x.extend([row['x'], row['x'], None])
         init_stem_y.extend([row['y'], row['y'], None])
         init_stem_z.extend([0.5, init_stagger_z[k], None])

    # Index = dynamic_trace_start_idx (Stems)
    fig.add_trace(go.Scatter3d(
        x=init_stem_x, y=init_stem_y, z=init_stem_z,
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.3)", width=1),
        hoverinfo="none", showlegend=False
    ))

    # Index = dynamic_trace_start_idx + 1 (Markers)
    fig.add_trace(go.Scatter3d(
        x=init_df["x"], y=init_df["y"], z=[0.5] * len(init_df),
        mode="markers", marker=dict(size=7, color=init_colors, line=dict(color="black", width=1)),
        hovertext=init_df["displayName"], hoverinfo="text", showlegend=False,
    ))

    # Index = dynamic_trace_start_idx + 2 (Labels)
    fig.add_trace(go.Scatter3d(
        x=init_df["x"], y=init_df["y"], z=init_stagger_z,
        mode="text", text=init_df["shortName"].apply(lambda x: f"<b>{x}</b>"),
        textposition="top center",
        textfont=dict(size=11, color="white", family="Consolas, 'Courier New', monospace"),
        showlegend=False,
    ))

    # Index = dynamic_trace_start_idx + 3 (Ball)
    fig.add_trace(go.Scatter3d(
        x=[ball_x[0]], y=[ball_y[0]], z=[ball_z[0]],
        mode="markers", marker=dict(symbol="circle", size=10, color="#f97316", line=dict(color="black", width=1)), showlegend=False,
    ))

    # Index = dynamic_trace_start_idx + 4 (Optional Ring)
    if best_def_id is not None:
        bd0 = init_df[init_df["nflId"] == best_def_id]
        if not bd0.empty:
            fig.add_trace(go.Scatter3d(
                x=[float(bd0["x"].iloc[0])], y=[float(bd0["y"].iloc[0])], z=[0.5],
                mode="markers", marker=dict(size=20, color="rgba(239,68,68,0.25)", line=dict(color="#ef4444", width=4)), showlegend=False,
            ))
        else:
             fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", showlegend=False))
    else:
          fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", showlegend=False))

    fig.frames = frames_list

    # --- CAMERA CONFIGURATION ---
    fixed_camera_view = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.1),
        eye=dict(x=-1.5, y=-1.5, z=1.2),
    )

    team_a_name = "Team A"
    team_b_name = "Team B"

    if not supp_df.empty:
        game_supp = supp_df[supp_df["gameId"] == game_id]
        if not game_supp.empty:
            if "home_team_abbr" in game_supp.columns:
                team_a_name = str(game_supp["home_team_abbr"].iloc[0])
            if "visitor_team_abbr" in game_supp.columns:
                team_b_name = str(game_supp["visitor_team_abbr"].iloc[0])

    if team_a_name == "Team A" and not inp_play.empty:
        if "possessionTeam" in inp_play.columns and not inp_play["possessionTeam"].isna().all():
            team_a_name = inp_play["possessionTeam"].dropna().iloc[0]
        elif "homeTeamAbbr" in inp_play.columns and not inp_play["homeTeamAbbr"].isna().all():
              team_a_name = inp_play["homeTeamAbbr"].dropna().iloc[0]

    if team_b_name == "Team B" and not inp_play.empty:
        if "defensiveTeam" in inp_play.columns and not inp_play["defensiveTeam"].isna().all():
            team_b_name = inp_play["defensiveTeam"].dropna().iloc[0]
        elif "visitorTeamAbbr" in inp_play.columns and not inp_play["visitorTeamAbbr"].isna().all():
              team_b_name = inp_play["visitorTeamAbbr"].dropna().iloc[0]

    fig.update_layout(
        autosize=True,
        title=dict(
            text=f"<b>{team_a_name} vs {team_b_name}</b>",
            font=dict(size=24, color="#f8fafc"),
            x=0.5, xanchor='center', y=0.95, yanchor='top'
        ),
        scene_camera=fixed_camera_view,
        scene=dict(
            xaxis=dict(range=[min_x, max_x], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[min_y, max_y], showgrid=False, zeroline=False, visible=False),
            zaxis=dict(range=[0, 15], showgrid=False, showticklabels=False, visible=False),
            aspectmode="manual",
            aspectratio=dict(x=3.0, y=1.5, z=0.3),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(family="Roboto, sans-serif", size=14, color="#e5e7eb"),
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(size=11, color="#f1f5f9"),
            itemsizing='constant'
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02, y=0.98, xanchor='left', yanchor='top',
                bgcolor="rgba(30, 41, 59, 0.9)",
                bordercolor="#334155",
                font=dict(color="#ffffff", size=11),
                buttons=[
                    dict(
                        label="â–¶ PLAY",
                        method="animate",
                        args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)],
                    )
                ],
            )
        ],
    )

    return fig, kpis, player_ranks


# ============================================================
# UI WIDGETS (Week -> Game -> Play) + KPI CARDS + EXPLANATION
# ============================================================

widget_style = """
<style>
    .widget-dropdown > select {
        background-color: #1e2538 !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-family: 'Roboto', sans-serif !important;
        font-size: 1em !important;
        outline: none !important;
        box-shadow: none !important;
        max-width: 100% !important;
        min-width: 150px !important;
    }
    .widget-dropdown > select:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    .widget-dropdown .widget-label {
        color: #94a3b8 !important;
        font-family: 'Roboto', sans-serif !important;
        font-size: 0.9em !important;
        font-weight: 500 !important;
        margin-right: 15px !important;
    }
    .gen-button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1em !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
    }
    .gen-button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 6px 12px -2px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    .gen-button:active {
        transform: translateY(0px) !important;
        box-shadow: none !important;
    }
</style>
"""

w_week = widgets.Dropdown(
    description="Week:",
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px', height='40px' ,margin='0 10px 10px 0')
)
w_game = widgets.Dropdown(
    description="Game ID:",
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px', height='40px' , margin='0 10px 10px 0')
)
w_play = widgets.Dropdown(
    description="Play ID:",
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px', height='40px' , margin='0 0 10px 0')
)

w_week.add_class("widget-dropdown")
w_game.add_class("widget-dropdown")
w_play.add_class("widget-dropdown")

btn_gen = widgets.Button(
    description="GENERATE VISUALIZATION",
    button_style="",
    layout=widgets.Layout(width='100%', height='45px', margin='15px 0 0 0')
)
btn_gen.add_class("gen-button")

out_viz = widgets.Output(layout=widgets.Layout(width='100%', height='650px'))
out_explain = widgets.Output(layout=widgets.Layout(width='100%'))

left_col = widgets.VBox([out_viz, out_explain], layout=widgets.Layout(width='55%'))

out_kpi = widgets.Output(layout=widgets.Layout(width='22.5%'))
out_players = widgets.Output(layout=widgets.Layout(width='22.5%'))


def update_selectors():
    if _df.empty:
        return
    weeks = sorted(_df["week"].dropna().unique().astype(int))
    w_week.options = weeks
    if weeks:
        w_week.value = weeks[0]


def update_games(*_):
    if _df.empty:
        return
    wk = w_week.value
    gms = sorted(_df[_df["week"] == wk]["gameId"].dropna().unique())
    w_game.options = gms
    if gms:
        w_game.value = gms[0]


def update_plays(*_):
    if _df.empty or w_game.value is None:
        return
    mask = (_df["week"] == w_week.value) & (_df["gameId"] == w_game.value)
    pls = sorted(_df[mask]["playId"].dropna().unique())
    w_play.options = pls
    if pls:
        w_play.value = pls[0]


w_week.observe(update_games, names="value")
w_game.observe(update_plays, names="value")


def build_explanation_html(game_id, play_id, kpis):
    bd = kpis["best_defender"]
    wd = kpis.get("worst_defender", {"name": "N/A", "dist": 0.0})
    sqi = kpis["sqi"]
    cds = kpis["cds"]
    cwd = kpis["cwd"]
    pe = kpis.get("pe", 0.0)
    pe_direct = kpis.get("pe_direct", 0.0)
    pe_actual = kpis.get("pe_actual", 0.0)
    target_name = kpis.get("target_wr_name", "targeted receiver")
    dcsi_data = kpis.get("dcsi_data", None)
    bfsg_data = kpis.get("bfsg_data", None)
    absr_data = kpis.get("absr_data", None)

    gss_data = kpis.get("gss_data", {})
    gss_html_block = ""
    if gss_data:
        for team, data in gss_data.items():
             if team == "Offense" or team == "Unknown":
                 header_text = f"Game-Level Separation Score: {data['score']} yds"
             else:
                 header_text = f"Game-Level Separation Score ({team}): {data['score']} yds"

             gss_html_block += f'''
             <li class="bullet-item">
                 <strong style="color:#334155">{header_text}</strong>
                 <ul class="sub-bullet-list">
                     <li class="sub-bullet-item"><i>Definition:</i> Average separation distance at catch point across all targeted routes.</li>
                     <li class="sub-bullet-item"><i>Observation:</i> {data["text"]} â€” Overall receiver group effectiveness.</li>
                     <li class="sub-bullet-item"><i>Top Performer:</i> {data["top_wr"]} ({data["top_val"]} yds).</li>
                 </ul>
             </li>'''
    else:
        gss_html_block = '<li class="bullet-item">No Separation Score Data Available</li>'

    absr_html_block = ""
    if absr_data and absr_data['display'] != 'N/A':
        absr_html_block = f"""
        <li class="bullet-item">
            <strong style="color:#334155">ABSR â€” Airborne Ball Safety Rating: {absr_data['display']}</strong>
            <ul class="sub-bullet-list">
                <li class="sub-bullet-item"><i>Definition:</i> Average minimum 3D distance (in yards) between the ball's flight path and the nearest defender, aggregated across all game passes.</li>
                <li class="sub-bullet-item"><i>Observation:</i> {absr_data['interpretation']}. This indicates the QB's overall risk profile when throwing.</li>
                <li class="sub-bullet-item"><i>Broadcast Example:</i> "{absr_data['broadcast']}"</li>
            </ul>
        </li>
        """
    else:
        absr_html_block = '<li class="bullet-item">ABSR Data Unavailable (Insufficient game tracking data)</li>'

    dcsi_html_block = ""
    if dcsi_data:
        dcsi_html_block = f"""
        <li class="bullet-item">
            <strong style="color:#334155">DCSI â€” Defender Closing Speed Index: {dcsi_data['dcsi']} yds/sec</strong>
            <ul class="sub-bullet-list">
                <li class="sub-bullet-item"><i>Definition:</i> How fast the nearest DB closes in toward the catch point.</li>
                <li class="sub-bullet-item"><i>Observation:</i> Closed from {dcsi_data['bp_sep']} yds to {dcsi_data['af_sep']} yds during the throw.</li>
                <li class="sub-bullet-item"><i>Insight:</i> {dcsi_data['rating']} â€” highlights defensive greatness in ball-in-air phase.</li>
            </ul>
        </li>
        """
    else:
        dcsi_html_block = '<li class="bullet-item">DCSI Data Unavailable (Requires direction tracking)</li>'

    bfsg_html_block = ""
    if bfsg_data:
        bfsg_html_block = f"""
        <li class="bullet-item">
            <strong style="color:#334155">BFSG â€” Ball Flight Separation Gain: {bfsg_data['bfsg']} yds</strong>
            <ul class="sub-bullet-list">
                <li class="sub-bullet-item"><i>Definition:</i> Separation difference between the release frame and the arrival frame.</li>
                <li class="sub-bullet-item"><i>Observation:</i> Separation changed from {bfsg_data['rel_sep']} yds (Release) to {bfsg_data['arr_sep']} yds (Arrival).</li>
                <li class="sub-bullet-item"><i>Insight:</i> {bfsg_data['rating']} â€” {bfsg_data['commentary']}</li>
            </ul>
        </li>
        """
    else:
        bfsg_html_block = '<li class="bullet-item">BFSG Data Unavailable (Insufficient tracking data)</li>'

    if bd["dist"] < 1:
        cov_text = "indicates very tight coverage, requiring perfect ball placement."
        cov_insight = "Defender is effectively mirroring the route; likely Man Coverage."
    elif bd["dist"] < 2:
        cov_text = "shows tight but manageable coverage."
        cov_insight = "Defender is in phase but a perfect throw beats the coverage."
    else:
        cov_text = "suggests looser coverage where the receiver has room to work."
        cov_insight = "Defender is playing off or passing off in Zone; deep shot prevention."

    if wd["dist"] > 10:
        wd_text = "gave significant cushion, likely protecting against the deep ball."
        wd_insight = "Defender bailed early or was in deep zone responsibility."
    elif wd["dist"] > 5:
        wd_text = "maintained a standard zone spacing."
        wd_insight = "Playing safe leverage to keep the play in front."
    else:
        wd_text = "was the furthest away but still relatively near the action."
        wd_insight = "Tight zone or compressed formation limited separation."

    if sqi < 1:
        sqi_text = "Receiver rarely gains clean separation."
        sqi_insight = "High Risk Throw: Requires a back-shoulder fade or aggressive contest."
    elif sqi < 2:
        sqi_text = "Separation is average; windows are open but not huge."
        sqi_insight = "Medium Risk: QB must throw away from the defender's leverage."
    else:
        sqi_text = "Receiver is often clearly open with comfortable space."
        sqi_insight = "Green Light: This should be a primary read for the QB."

    if cds < 3:
        cds_level = "Easy"
        cds_insight = "Clean passing lanes; QB has high visibility of the target."
    elif cds < 6:
        cds_level = "Moderate"
        cds_insight = "Standard traffic; requires normal layering of the ball."
    else:
        cds_level = "Difficult"
        cds_insight = "Congested Box: QB likely needs to slide in the pocket to find a lane."

    if cwd < 0.3:
        cwd_text = "QB has almost no time to hit the receiver once open."
        cwd_insight = "Anticipation Required: Ball must be out before the receiver breaks."
    elif cwd < 0.8:
        cwd_text = "QB has a reasonable window to throw into."
        cwd_insight = "Standard Timing: QB can verify openness before releasing."
    else:
        cwd_text = "QB enjoys a long, forgiving window to target the receiver."
        cwd_insight = "Extended Play: Receiver successfully found a soft spot in coverage."

    suggestion_blocks = []

    if bd["dist"] < 1.5 and cds >= 6:
        points = [
            "<b>Implement Rub Routes:</b> Use crossing patterns and 'mesh' concepts to create natural picks against man coverage.",
            "<b>Bunch Formations:</b> Compress the formation to force defense into zone checks and confuse assignments at the snap.",
            "<b>Pre-Snap Motion:</b> Use jet motion to force the defense to reveal coverage type early and loosen the box.",
            "<b>Attack the Seams:</b> Look for soft spots between zone defenders in the middle of the field rather than outside."
        ]
        suggestion_blocks.append({"title": "Offensive Scheme Adjustment (Tournament Strategy)", "points": points})

    if sqi < 1.5:
        points = [
            "<b>Release Packages:</b> Drill varying release techniques (foot-fire, speed release) to avoid getting jammed at the line.",
            "<b>Hand Fighting:</b> Emphasize clearing the defender's hands at the break point of the route to create late separation.",
            "<b>Stemming:</b> Coach receivers to stem routes directly at defenders to manipulate their hips before breaking.",
            "<b>Stacking:</b> Teach receivers to get back on top of the defender vertically to own the leverage."
        ]
        suggestion_blocks.append({"title": "Training Focus (Skill Development)", "points": points})

    if cwd < 0.4:
        points = [
            "<b>Anticipation Throws:</b> QB must throw to a spot before the receiver looks back; trust the timing.",
            "<b>Read Key Defenders:</b> Identify the 'conflict defender' (e.g., linebacker in RPO) rather than staring down the receiver.",
            "<b>Quick Game:</b> Incorporate 3-step drop concepts to neutralize a heavy pass rush.",
            "<b>Pocket Movement:</b> Slide slightly within the pocket to find clear passing lanes in congested areas."
        ]
        suggestion_blocks.append({"title": "QB Decision Making (Game Management)", "points": points})

    if not suggestion_blocks:
        points = [
            "<b>Maintain Tempo:</b> Use no-huddle concepts to limit defensive substitutions and simplify coverages.",
            "<b>Isolate Matchups:</b> Identify the weakest defensive back and target them repeatedly in 1-on-1 situations.",
            "<b>Play Action:</b> Use run fakes to freeze linebackers and open intermediate windows behind them.",
            "<b>Protect the Ball:</b> Prioritize possession; don't force throws into double coverage."
        ]
        suggestion_blocks.append({"title": "General Tournament Strategy", "points": points})

    sugg_html = ""
    for block in suggestion_blocks:
        sugg_html += f"<div style='margin-bottom:15px; border-bottom:1px solid #e2e8f0; padding-bottom:10px;'><b style='color:#334155; text-transform:uppercase; font-size:0.9em;'>{block['title']}</b><ul class='sub-bullet-list'>"
        for p in block['points']:
            sugg_html += f"<li class='sub-bullet-item'>{p}</li>"
        sugg_html += "</ul></div>"


    html = f"""
    <div class="explain-container">

      <div class="explain-title">
        COACH'S GAME NOTE <span style="font-weight:400; color:#64748b; font-size:0.8em; margin-left:10px;">Play {play_id} Breakdown</span>
      </div>

      <div class="section-box">
        <div class="section-header">ğŸ”� How to Read this Play</div>
          <p style="font-size:0.9em; margin:0; color:#475569;">
            <b style="color:#f59e0b">OFFENSE (Yellow)</b> vs <b style="color:#3b82f6">DEFENSE (Blue)</b>.<br>
            The <b style="color:#ef4444">RED HALO</b> marks the Best Defender.
            Solid Line = Actual Route.
          </p>
      </div>

      <div class="section-box">
        <div class="section-header">ğŸ“Š KPI METRICS ANALYSIS</div>
        <ul class="bullet-list">
            <li class="bullet-item">
                <strong style="color:#334155">Best Defender ({bd['name']}): {bd['dist']} yds</strong>
                <ul class="sub-bullet-list">
                    <li class="sub-bullet-item"><i>Definition:</i> Closest defender throughout route.</li>
                    <li class="sub-bullet-item"><i>Observation:</i> {cov_text}</li>
                    <li class="sub-bullet-item"><i>Tactical Note:</i> {cov_insight}</li>
                </ul>
            </li>

            <li class="bullet-item">
                <strong style="color:#334155">Worst Defender ({wd['name']}): {wd['dist']} yds</strong>
                <ul class="sub-bullet-list">
                    <li class="sub-bullet-item"><i>Definition:</i> Defender allowing the most separation avg.</li>
                    <li class="sub-bullet-item"><i>Observation:</i> {wd_text}</li>
                    <li class="sub-bullet-item"><i>Tactical Note:</i> {wd_insight}</li>
                </ul>
            </li>

            {gss_html_block}

            {absr_html_block}

            {bfsg_html_block}

            {dcsi_html_block}

            <li class="bullet-item">
                <strong style="color:#334155">Separation Quality Index: {sqi} yds</strong>
                <ul class="sub-bullet-list">
                    <li class="sub-bullet-item"><i>Definition:</i> Avg open space from nearest defender.</li>
                    <li class="sub-bullet-item"><i>Observation:</i> {sqi_text}</li>
                    <li class="sub-bullet-item"><i>Confidence:</i> {sqi_insight}</li>
                </ul>
            </li>

            <li class="bullet-item">
                <strong style="color:#334155">Coverage Difficulty Score: {cds}/10 ({cds_level})</strong>
                <ul class="sub-bullet-list">
                    <li class="sub-bullet-item"><i>Definition:</i> Composite rating (0-10) based on defender proximity and crowd density.</li>
                    <li class="sub-bullet-item"><i>Impact:</i> {cds_insight}</li>
                </ul>
            </li>

            <li class="bullet-item">
                <strong style="color:#334155">Catch Window Distance: {cwd} yds</strong>
                <ul class="sub-bullet-list">
                    <li class="sub-bullet-item"><i>Definition:</i> Separation between ball and receiver at arrival.</li>
                    <li class="sub-bullet-item"><i>Observation:</i> {cwd_text}</li>
                    <li class="sub-bullet-item"><i>Tactical Note:</i> Anticipation Required: Ball must be out before the receiver breaks.</li>
                </ul>
            </li>
        </ul>
      </div>

      <div class="section-box" style="border-left-color:#f59e0b; background:#fffbeb;">
        <div class="section-header" style="color:#b45309;">ğŸ“‹ STRATEGIC RECOMMENDATIONS</div>
        {sugg_html}
      </div>
    </div>
    """
    return html


def build_player_rank_html(player_ranks):
    if not player_ranks:
        return "<div>No player tracking data available.</div>"

    motm = player_ranks[0]
    top_5 = player_ranks[:5]
    worst_5 = player_ranks[-5:] if len(player_ranks) >= 5 else []
    worst_5 = sorted(worst_5, key=lambda x: x['score'])

    html = f"""{KPI_STYLE}
    <div class="rank-panel">
        <div class="panel-header">
            <span>âš¡ PLAYER IMPACT</span>
        </div>
        <div class="panel-sub">Speed & Distance Perf. Index</div>

        <div class="mom-card">
            <span class="mom-badge">ğŸ�† MAN OF THE MATCH</span>
            <div class="mom-name">{motm['name']}</div>
            <div class="mom-stat">Perf Score: {motm['score']:.1f}</div>
            <div style="font-size:0.75em; margin-top:8px; opacity:0.8;">
                Max Speed: {motm['max_speed']:.1f} mph â€¢ Dist: {motm['dist']:.1f} yds
            </div>
        </div>

        <div class="rank-section-header">
            <span>ğŸ”¥ Top 5 Performers</span>
            <span style="font-size:0.8em; opacity:0.7">SCORE</span>
        </div>
        """

    for i, p in enumerate(top_5):
        html += f"""
        <div class="player-row row-best">
            <span class="p-rank">#{i+1}</span>
            <span class="p-name">{p['name']}</span>
            <span class="p-score">{p['score']:.1f}</span>
        </div>
        """

    html += """
        <div class="rank-section-header" style="margin-top:25px;">
            <span>â�„ï¸� Bottom 5 Performers</span>
            <span style="font-size:0.8em; opacity:0.7">SCORE</span>
        </div>
    """

    for i, p in enumerate(worst_5):
        html += f"""
        <div class="player-row row-worst">
            <span class="p-rank"></span>
            <span class="p-name">{p['name']}</span>
            <span class="p-score">{p['score']:.1f}</span>
        </div>
        """

    html += "</div>"
    return html


def on_click_gen(_):

    out_viz.clear_output()
    out_kpi.clear_output()
    out_players.clear_output()
    out_explain.clear_output()

    gid = w_game.value
    pid = w_play.value
    if gid is None or pid is None:
        return

    fig, kpis, player_ranks = create_3d_broadcast_animation(gid, pid)
    fig.write_html(
    "/kaggle/working/code1_main/main_code.html"
     )

    with out_viz:
        fig.show()

    with out_kpi:
        bd = kpis["best_defender"]
        wd = kpis.get("worst_defender", {"name": "N/A", "dist": 0.0})
        wr_name = kpis.get("target_wr_name", "Unknown")
        absr_data = kpis.get("absr_data", {"display": "N/A", "interpretation": "N/A", "broadcast": "N/A"})

        absr_val = absr_data["display"].split()[0]
        absr_class = ""
        try:
            absr_num = float(absr_val)
            if absr_num >= 3.0: absr_class = "card-green"
            elif absr_num >= 1.5: absr_class = "card-gold"
            else: absr_class = "card-red"
        except ValueError:
            absr_class = ""

        absr_card_html = f"""
        <div class="kpi-card {absr_class}">
            <div class="kpi-title">Airborne Ball Safety Rating (ABSR)</div>
            <div class="kpi-value">{absr_data['display']}</div>
            <div class="kpi-desc">Avg min distance from ball to defender (Game)</div>
            <div class="kpi-simple">{absr_data['interpretation']}</div>
        </div>
        """

        gss_data = kpis.get("gss_data", {})

        gss_cards_html = ""
        if gss_data:
            for team_name, data in gss_data.items():
                score = data.get("score", 0.0)
                text = data.get("text", "N/A")
                top_wr = data.get("top_wr", "N/A")
                top_val = data.get("top_val", 0.0)

                gss_class = ""
                if score < 1.5: gss_class = "card-red"
                elif score < 3.0: gss_class = "card-gold"
                else: gss_class = "card-green"

                gss_cards_html += f"""
                <div class="kpi-card {gss_class}">
                    <div class="kpi-title">Game Level Separation Score </div>
                    <div class="kpi-value">{score} <span style="font-size:0.5em">yds</span></div>
                    <div class="kpi-desc">Avg separation (Game) â€¢ {text}</div>
                    <div class="kpi-simple">Top: {top_wr} ({top_val} yds)</div>
                </div>
                """
        else:
             gss_cards_html = """
             <div class="kpi-card">
                 <div class="kpi-title">SEPARATION SCORE</div>
                 <div class="kpi-desc">No Data Available</div>
             </div>
             """

        def wr_div(name):
             return f"<div style='font-size:0.9em; font-weight:700; color:#e2e8f0; margin-bottom:4px;'>{name}</div>"

        html_kpi = f"""
        {KPI_STYLE}
        <div class="kpi-panel">
            <div class="panel-header">
                <span>ğŸ“¡ BROADCAST FEED <span style="font-size:0.6em; color:#94a3b8; margin-left:8px; border:1px solid #334155; padding:2px 6px; border-radius:4px;">GAME LEVEL</span></span>
            </div>
            <div class="panel-sub">Play Analysis â€¢ Game {gid}</div>

            {absr_card_html}

            <div class="kpi-card card-red">
                <div class="kpi-title">Best Defender</div>
                <div class="kpi-value">{bd['name']}</div>
                <div class="kpi-desc">Avg Proximity: {bd['dist']} yds</div>
                <div class="kpi-simple">Identify who stuck closest to the WR</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-title">Worst Defender</div>
                <div class="kpi-value">{wd['name']}</div>
                <div class="kpi-desc">Avg Proximity: {wd['dist']} yds</div>
                <div class="kpi-simple">Most average separation allowed</div>
            </div>

            {gss_cards_html}
        </div>
        """
        display(HTML(html_kpi))

    with out_players:
        bd = kpis["best_defender"]
        wd = kpis.get("worst_defender", {"name": "N/A", "dist": 0.0})
        wr_name = kpis.get("target_wr_name", "Unknown")
        dcsi_data = kpis.get("dcsi_data", None)
        bfsg_data = kpis.get("bfsg_data", None)

        def wr_div(name):
             return f"<div style='font-size:0.9em; font-weight:700; color:#e2e8f0; margin-bottom:4px;'>{name}</div>"

        dcsi_card = ""
        if dcsi_data:
            dcsi_val = dcsi_data["dcsi"]
            rating = dcsi_data["rating"]
            if dcsi_val > 1.0:
                d_class = "card-red"
            elif dcsi_val > 0.0:
                d_class = "card-gold"
            else:
                d_class = "card-green"

            dcsi_card = f"""
            <div class="kpi-card {d_class}">
                <div class="kpi-title">Defender Closing Speed Index</div>
                <div class="kpi-value">{dcsi_val} <span style="font-size:0.5em">yds/s</span></div>
                <div class="kpi-desc">{rating}</div>
                <div class="kpi-simple">Speed of nearest DB closing in</div>
            </div>
            """
        else:
            dcsi_card = f"""
            <div class="kpi-card">
                <div class="kpi-title">Defender Closing Speed Index</div>
                <div class="kpi-value">N/A</div>
                <div class="kpi-desc">Insufficient tracking data</div>
            </div>
            """

        bfsg_card = ""
        if bfsg_data:
            bfsg_val = bfsg_data["bfsg"]
            rating = bfsg_data["rating"]

            if bfsg_val >= 1.0:
                b_class = "card-green"
            elif bfsg_val >= 0.0:
                b_class = "card-gold"
            else:
                b_class = "card-red"

            bfsg_card = f"""
            <div class="kpi-card {b_class}">
                <div class="kpi-title">Ball Flight Separation Gain (BFSG)</div>
                <div class="kpi-value">{bfsg_val} <span style="font-size:0.5em">yds</span></div>
                <div class="kpi-desc">{rating}</div>
                <div class="kpi-simple">Separation change during ball flight</div>
            </div>
            """
        else:
             bfsg_card = f"""
            <div class="kpi-card">
                <div class="kpi-title">Ball Flight Separation Gain (BFSG)</div>
                <div class="kpi-value">N/A</div>
                <div class="kpi-desc">Insufficient tracking data</div>
            </div>
            """

        html_kpi_right = f"""
        {KPI_STYLE}
        <div class="kpi-panel">
            <div class="panel-header">
                <span>ğŸ�¯ TARGET METRICS <span style="font-size:0.6em; color:#94a3b8; margin-left:8px; border:1px solid #334155; padding:2px 6px; border-radius:4px;">PLAY LEVEL</span></span>
            </div>
            <div class="panel-sub">Receiver Performance Analysis</div>

            <div class="kpi-card card-gold">
                <div class="kpi-title">Separation Quality Index</div>
                <div class="kpi-value">{kpis['sqi']} <span style="font-size:0.5em">yds</span></div>
                <div class="kpi-desc">Avg separation near catch</div>
                <div class="kpi-simple">How "Open" was the receiver?</div>
            </div>

            <div class="kpi-card card-green">
                <div class="kpi-title">Coverage Difficulty Score</div>
                <div class="kpi-value">{kpis['cds']} <span style="font-size:0.5em">/ 10</span></div>
                <div class="kpi-desc">Coverage Difficulty Score</div>
                <div class="kpi-simple">Higher = More defenders nearby</div>
            </div>

            {bfsg_card}

            <div class="kpi-card">
                <div class="kpi-title">Catch Window Distance</div>
                <div class="kpi-value">{kpis['cwd']} <span style="font-size:0.5em">yds</span></div>
                <div class="kpi-desc">Proximity at catch point</div>
                <div class="kpi-simple">Distance between WR and Ball</div>
            </div>

            {dcsi_card}
        </div>
        """
        display(HTML(html_kpi_right))

    with out_explain:
        explanation_html = build_explanation_html(gid, pid, kpis)
        display(HTML(explanation_html))

        display(Javascript("setTimeout(function() { window.scrollTo({ top: 0, behavior: 'smooth' }); }, 50);"))


btn_gen.on_click(on_click_gen)

update_selectors()
update_games()
update_plays()

header_html = """
<div style="background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
    <h2 style="color: #f8fafc; margin: 0; font-family: 'Roboto', sans-serif; font-weight: 900; letter-spacing: 1px;">
        ğŸ�ˆ NFL NEXT GEN STATS <span style="color:#3b82f6">PRO</span>
    </h2>
    <p style="color: #94a3b8; margin: 5px 0 0 0; font-family: 'Roboto', sans-serif;">
        3D Broadcast Reconstruction â€¢ Micro-KPIs â€¢ Performance Index
    </p>
</div>
"""

controls_container = widgets.VBox([
    widgets.HTML(widget_style),
    widgets.HBox([w_week, w_game, w_play], layout=widgets.Layout(
        justify_content='flex-start',
        align_items='center',
        width='100%',
        flex_flow='row wrap'
    )),
    btn_gen
], layout=widgets.Layout(
    background_color='#151b2b',
    padding='20px',
    border_radius='16px',
    border='1px solid rgba(255, 255, 255, 0.1)',
    margin='0 0 25px 0'
))

ui = widgets.VBox([
    widgets.HTML(header_html),
    controls_container,
    widgets.HBox([left_col, out_players, out_kpi])
])

display(ui)


# Combined script: 3D NFL Path Efficiency visualization + KPI panel + Coach's Game Note


import os
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Javascript

# # --- Colab widget support ---
# try:
#     from google.colab import output as colab_output
#     colab_output.enable_custom_widget_manager()
# except Exception:
#     pass

pio.renderers.default = "iframe"
print("Renderer:", pio.renderers.default)

# ---------- CONFIG: CHANGE PATHS IF NEEDED ----------
INPUT_FILE  = "/kaggle/input/nfl-analytics/combined_input_2023.csv"
OUTPUT_FILE = "/kaggle/input/nfl-analytics/combined_output_2023.csv"
SUPP_FILE   = "/kaggle/input/nfl-analytics/supplementary_data.csv"

# ---------- CSS STYLES (CYBER THEME) ----------
KPI_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');

    :root {
        --bg-dark: #0b0f19;
        --panel-bg: #151b2b;
        --card-bg: #1e2538;
        --text-main: #f1f5f9;
        --text-sub: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-gold: #f59e0b;
        --neon-border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* GENERAL PANEL STYLES */
    .kpi-panel, .rank-panel, .explain-panel {
        background-color: var(--panel-bg);
        color: var(--text-main);
        padding: 20px;
        border-radius: 16px;
        font-family: 'Roboto', sans-serif;
        height: 720px;
        overflow-y: auto;
        border: var(--neon-border);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }

    /* HEADERS */
    .panel-header {
        font-size: 1.1em;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-main);
        margin-bottom: 4px;
        border-bottom: 2px solid #334155;
        padding-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }
    .panel-sub {
        font-size: 0.8em;
        color: var(--text-sub);
        margin-bottom: 15px;
        font-weight: 400;
    }

    /* KPI CARDS */
    .kpi-card {
        background: linear-gradient(145deg, #1e2538, #171d2d);
        border: 1px solid rgba(255,255,255,0.05);
        border-left: 4px solid var(--accent-blue);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .kpi-title {
        font-size: 0.75em;
        font-weight: 700;
        color: var(--text-sub);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.25em; /* Adjusted for tighter layout */
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-desc {
        font-size: 0.8em;
        color: #64748b;
        margin-top: 4px;
        line-height: 1.3;
    }
    .kpi-simple {
        font-size: 0.75em;
        color: var(--accent-blue);
        margin-top: 6px;
        font-style: italic;
        padding-top: 6px;
        border-top: 1px dashed rgba(255, 255, 255, 0.1);
    }

    /* KPI ACCENTS */
    .card-red { border-left-color: var(--accent-red); }
    .card-green { border-left-color: var(--accent-green); }
    .card-gold { border-left-color: var(--accent-gold); }

    /* PLAYER RANKINGS */
    .mom-card {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #fff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);
        position: relative;
        overflow: hidden;
    }
    .mom-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
        transform: rotate(30deg);
    }
    .mom-badge {
        background: rgba(0,0,0,0.3);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7em;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 10px;
    }
    .mom-name {
        font-size: 1.4em;
        font-weight: 900;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 5px;
    }
    .mom-stat {
        font-size: 1.1em;
        font-weight: 700;
        opacity: 0.95;
    }

    .rank-section-header {
        font-size: 0.85em;
        color: var(--text-sub);
        text-transform: uppercase;
        font-weight: 700;
        margin: 20px 0 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        border-bottom: 1px solid #334155;
        padding-bottom: 5px;
    }

    .player-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255,255,255,0.03);
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        border-left: 3px solid transparent;
        transition: background 0.2s;
    }
    .player-row:hover { background: rgba(255,255,255,0.06); }
    .row-best { border-left-color: var(--accent-green); }
    .row-worst { border-left-color: var(--accent-red); }

    .p-rank { font-size: 0.8em; color: #64748b; width: 25px; }
    .p-name { font-weight: 600; color: #e2e8f0; font-size: 0.85em; flex-grow: 1; }
    .p-score { font-family: 'Roboto Mono', monospace; font-weight: 700; color: #cbd5e1; font-size: 0.9em; width: 50px; text-align: right;}
    .p-detail { font-size: 0.75em; color: #64748b; width: 60px; text-align: right; margin-right: 10px;}

    /* LISTS INSIDE CARDS */
    .tiny-list { list-style: none; padding: 0; margin: 0; }
    .tiny-list li {
        font-size: 0.85em; color: #cbd5e1;
        padding: 4px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        display: flex; justify-content: space-between;
    }
    .tiny-list li:last-child { border-bottom: none; }
    .tiny-val { font-weight: 700; color: var(--accent-blue); }

    /* COLUMNS */
    .kpi-col-header {
        font-size: 0.8em; font-weight: 800; color: #94a3b8; text-transform: uppercase;
        border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 10px;
    }

    /* --- COACH'S GAME NOTE (WHITE BOX) --- */
    .explain-container {
        background: #ffffff;
        color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        font-family: 'Roboto', sans-serif;
    }
    .explain-title {
        font-size: 1.2em;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 15px;
        border-bottom: 3px solid #f59e0b;
        display: inline-block;
        padding-bottom: 2px;
    }
    .section-box {
        background: #f8fafc;
        border-left: 4px solid #cbd5e1;
        padding: 12px 16px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
    }
    .section-header {
        font-weight: 700;
        color: #334155;
        margin-bottom: 8px;
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .bullet-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .bullet-item {
        position: relative;
        padding-left: 18px;
        margin-bottom: 8px;
        font-size: 0.9em;
        line-height: 1.5;
        color: #475569;
    }
    .bullet-item::before {
        content: 'â€¢';
        position: absolute;
        left: 0;
        color: #f59e0b;
        font-weight: bold;
    }
    .sub-bullet-list {
        list-style: none;
        padding-left: 10px;
        margin-top: 4px;
        border-left: 2px solid #e2e8f0;
        margin-left: 5px;
    }
    .sub-bullet-item {
        font-size: 0.85em;
        margin-bottom: 4px;
        padding-left: 10px;
        color: #64748b;
    }
</style>
"""

# ---------- GLOBALS ----------
input_df  = pd.DataFrame()
output_df = pd.DataFrame()
supp_df   = pd.DataFrame()
_df       = pd.DataFrame()

# ---------- LOAD DATA ----------
def load_all():
    global input_df, output_df, supp_df, _df

    # ---- Input (pre-throw) ----
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        input_df = pd.DataFrame()
    else:
        inp = pd.read_csv(INPUT_FILE, low_memory=False)

        col_map = {
            "game_id": "gameId",
            "play_id": "playId",
            "frame_id": "frameId",
            "player_name": "displayName",
            "player_side": "teamName",
            "nfl_id": "nflId",
            "week": "week",
        }
        inp.rename(columns=col_map, inplace=True)

        for c in ["x", "y", "s", "dir", "frameId", "ball_land_x", "ball_land_y", "week"]:
            if c in inp.columns:
                inp[c] = pd.to_numeric(inp[c], errors="coerce")

        if {"x", "y", "frameId"}.issubset(inp.columns):
            inp = inp.dropna(subset=["x", "y", "frameId"])

        input_df = inp
        _df = inp.copy()
        print("Input rows:", input_df.shape[0])

    # ---- Output (post-throw) ----
    if not os.path.exists(OUTPUT_FILE):
        print(f"Output file not found: {OUTPUT_FILE}")
        output_df = pd.DataFrame()
    else:
        out = pd.read_csv(OUTPUT_FILE, low_memory=False)

        col_map_out = {
            "game_id": "gameId",
            "play_id": "playId",
            "frame_id": "frameId",
            "nfl_id": "nflId",
            "week": "week",
        }
        out.rename(columns=col_map_out, inplace=True)

        for c in ["x", "y", "frameId", "week"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        if {"x", "y", "frameId"}.issubset(out.columns):
            out = out.dropna(subset=["x", "y", "frameId"])

        output_df = out
        print("Output rows:", output_df.shape[0])

    # ---- Supplementary ----
    if os.path.exists(SUPP_FILE):
        s = pd.read_csv(SUPP_FILE, low_memory=False)
        s.rename(columns={"game_id": "gameId", "play_id": "PlayId", "play_id": "playId"}, inplace=True)
        if "PlayId" in s.columns and "playId" in s.columns:
            s.drop(columns=["PlayId"], inplace=True)
        supp_df = s
        print("Supplementary rows:", supp_df.shape[0])
    else:
        print(f"Supplementary not found: {SUPP_FILE}")
        supp_df = pd.DataFrame()

load_all()

# ---------- VISUAL HELPERS (CYBER STYLE) ----------
def add_goalposts(fig, x_pos, y_center=26.65, z_crossbar=3.33, z_top=13.33, width=6.17, color="#facc15"):
    """Adds a 'Slingshot' style NFL goalpost to the 3D Plotly figure."""
    hw = width / 2.0
    # 1. Base Support
    fig.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_center, y_center], z=[0, z_crossbar], mode="lines", line=dict(color=color, width=12), hoverinfo="text", text="Goal Post Base", showlegend=False))
    # 2. Crossbar
    fig.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_center - hw, y_center + hw], z=[z_crossbar, z_crossbar], mode="lines", line=dict(color=color, width=12), hoverinfo="none", showlegend=False))
    # 3. Left Upright
    fig.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_center - hw, y_center - hw], z=[z_crossbar, z_top], mode="lines", line=dict(color=color, width=10), hoverinfo="none", showlegend=False))
    # 4. Right Upright
    fig.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_center + hw, y_center + hw], z=[z_crossbar, z_top], mode="lines", line=dict(color=color, width=10), hoverinfo="none", showlegend=False))

def add_qb_stick_figure(fig, x, y, land_x, land_y, color="#fbbf24"):
    """Adds a static 3D stick figure representing the QB throwing the ball."""
    if x is None or y is None or land_x is None or land_y is None: return
    dx, dy = land_x - x, land_y - y
    norm = np.hypot(dx, dy)
    ux, uy = (dx/norm, dy/norm) if norm > 0 else (1, 0)
    px, py = -uy, ux
    shoulder_w, arm_len, leg_spread = 0.3, 0.4, 0.3

    body_x, body_y, body_z = [x, x], [y, y], [1.0, 1.7]
    ll_x, ll_y, ll_z = [x, x - px*leg_spread*0.5 - ux*0.2], [y, y - py*leg_spread*0.5 - uy*0.2], [1.0, 0.0]
    rl_x, rl_y, rl_z = [x, x + px*leg_spread*0.5 + ux*0.3], [y, y + py*leg_spread*0.5 + uy*0.3], [1.0, 0.0]
    ra_x, ra_y, ra_z = [x + px*shoulder_w, x + px*shoulder_w + ux*arm_len], [y + py*shoulder_w, y + py*shoulder_w + uy*arm_len], [1.7, 2.1]
    la_x, la_y, la_z = [x - px*shoulder_w, x - px*shoulder_w - ux*0.2], [y - py*shoulder_w, y - py*shoulder_w - uy*0.2], [1.7, 1.4]

    all_x = body_x + [None] + ll_x + [None] + rl_x + [None] + ra_x + [None] + la_x
    all_y = body_y + [None] + ll_y + [None] + rl_y + [None] + ra_y + [None] + la_y
    all_z = body_z + [None] + ll_z + [None] + rl_z + [None] + ra_z + [None] + la_z

    fig.add_trace(go.Scatter3d(x=all_x, y=all_y, z=all_z, mode="lines", line=dict(color=color, width=5), name="QB Throwing", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[1.9], mode="markers", marker=dict(size=6, color=color, line=dict(color='black', width=1)), name="QB Head", showlegend=False))

def shorten_name(name):
    if not isinstance(name, str): return ""
    parts = name.strip().split()
    if len(parts) >= 2: return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return name

# ---------- PE MATH HELPERS ----------
def cubic_bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n).reshape(-1, 1)
    B = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    return B

def compute_3d_pe(p_track_xy, elev_z=2.0, bezier_points=200):
    if p_track_xy is None or p_track_xy.empty: return None, None, None, None
    p0_xy = np.array([float(p_track_xy["x"].iloc[0]), float(p_track_xy["y"].iloc[0])])
    p3_xy = np.array([float(p_track_xy["x"].iloc[-1]), float(p_track_xy["y"].iloc[-1])])

    if np.allclose(p0_xy, p3_xy): return 0.0, 0.0, 0.0, np.array([[p0_xy[0], p0_xy[1], 0.0]])

    vec = p3_xy - p0_xy
    perp = np.array([-vec[1], vec[0]])
    norm_val = np.linalg.norm(perp)
    perp = perp / norm_val if norm_val > 0 else np.array([0.0, 0.0])

    p1_xy = p0_xy + 0.25 * vec + 0.1 * np.linalg.norm(vec) * perp
    p2_xy = p0_xy + 0.75 * vec + 0.1 * np.linalg.norm(vec) * perp

    p0, p1, p2, p3 = np.array([*p0_xy, 0.0]), np.array([*p1_xy, elev_z]), np.array([*p2_xy, elev_z]), np.array([*p3_xy, 0.0])
    ideal_pts = cubic_bezier(p0, p1, p2, p3, n=bezier_points)

    direct_3d_distance = float(np.linalg.norm(p3 - p0))
    coords = np.vstack([p_track_xy["x"].astype(float).values, p_track_xy["y"].astype(float).values, np.zeros(len(p_track_xy))]).T

    if coords.shape[0] < 2: actual_route_length = 0.0
    else:
        deltas = coords[1:] - coords[:-1]
        actual_route_length = float(np.sqrt((deltas**2).sum(axis=1)).sum())

    pe = direct_3d_distance / actual_route_length if actual_route_length > 0 else 0.0
    return direct_3d_distance, actual_route_length, float(np.clip(pe, 0.0, 2.0)), ideal_pts

def compute_pe_all_players(full_df):
    pe_dict = {}
    if full_df.empty: return pe_dict

    full_df = full_df.dropna(subset=["x", "y", "frameId"])
    for pid, df_p in full_df.groupby("nflId"):
        df_p = df_p.sort_values("frameId")
        direct_d, actual_len, pe_val, ideal_pts = compute_3d_pe(df_p[["x", "y"]])

        if pe_val is None: continue

        name = df_p["displayName"].iloc[0] if "displayName" in df_p.columns else str(pid)
        team = df_p["teamName"].iloc[0] if "teamName" in df_p.columns else "Unknown"

        # Calculate total distance covered (raw)
        coords = df_p[["x", "y"]].astype(float).values
        if coords.shape[0] >= 2:
            deltas = coords[1:] - coords[:-1]
            total_dist = float(np.sqrt((deltas**2).sum(axis=1)).sum())
        else:
            total_dist = 0.0

        pe_dict[pid] = {
            "nflId": pid, "name": name, "team": team,
            "direct": round(direct_d, 3), "route_len": round(actual_len, 3),
            "pe": round(pe_val, 3), "ideal_pts": ideal_pts,
            "total_dist": total_dist
        }
    return pe_dict

# ---------- MAIN ANIMATION ----------
def create_3d_pe_animation(game_id, play_id, marker_size_players=6, label_text_size=10, show_labels=True,
                           show_landing_goal=True, frame_duration=80, camera_preset="Default",
                           boxed_highlights=False):
    global input_df, output_df, supp_df

    empty_pe = {}
    inp_play = input_df[(input_df["gameId"] == int(game_id)) & (input_df["playId"] == int(play_id))].copy()
    out_play = output_df[(output_df["gameId"] == int(game_id)) & (output_df["playId"] == int(play_id))].copy()

    if inp_play.empty and out_play.empty:
        fig = go.Figure(); fig.add_annotation(text="No data for this Game/Play", showarrow=False); return fig, empty_pe, {}

    # Ball Landing
    land_x, land_y = None, None
    if "ball_land_x" in inp_play and inp_play["ball_land_x"].notna().any(): land_x = float(inp_play["ball_land_x"].iloc[0])
    if "ball_land_y" in inp_play and inp_play["ball_land_y"].notna().any(): land_y = float(inp_play["ball_land_y"].iloc[0])
    if (land_x is None) and not supp_df.empty:
        sp = supp_df[(supp_df["gameId"]==int(game_id)) & (supp_df["playId"]==int(play_id))]
        if not sp.empty:
            if "ball_land_x" in sp: land_x = float(sp["ball_land_x"].iloc[0])
            if "ball_land_y" in sp: land_y = float(sp["ball_land_y"].iloc[0])

    if land_x is None:
        if not inp_play.empty:
            land_x = float(inp_play["x"].mean())
        else:
            land_x = 0.0
    if land_y is None:
        if not inp_play.empty:
            land_y = float(inp_play["y"].mean())
        else:
            land_y = 26.65

    # Trajectories
    full_tracks = []
    players_in = set(inp_play["nflId"].unique()) if not inp_play.empty else set()
    players_out = set(out_play["nflId"].unique()) if not out_play.empty else set()

    info = {}
    if not inp_play.empty:
        tmp = inp_play[["nflId", "displayName", "teamName"]].drop_duplicates()
        info = {pid: {"name": row["displayName"], "team": row["teamName"]} for pid, row in tmp.set_index("nflId").iterrows() }

    for pid in (players_in | players_out):
        pin = inp_play[inp_play["nflId"] == pid].sort_values("frameId")
        pout = out_play[out_play["nflId"] == pid].sort_values("frameId")
        if not pin.empty and not pout.empty:
            pout["frameId"] += pin["frameId"].max()

        if not pout.empty:
            if "displayName" not in pout or pout["displayName"].isna().all(): pout["displayName"] = info.get(pid, {}).get("name")
            if "teamName" not in pout or pout["teamName"].isna().all(): pout["teamName"] = info.get(pid, {}).get("team")

        p_full = pd.concat([pin, pout], ignore_index=True)
        if not p_full.empty: full_tracks.append(p_full)

    full_df = pd.concat(full_tracks, ignore_index=True) if full_tracks else pd.DataFrame()
    if full_df.empty: return go.Figure(), empty_pe, {}

    players_all = full_df[["frameId", "nflId", "displayName", "teamName", "x", "y"]].dropna()
    frames = sorted(players_all["frameId"].unique())

    # Fix: Scaffolding for animation
    unique_players = players_all[['nflId', 'displayName', 'teamName']].drop_duplicates(subset=['nflId'])
    scaffold = pd.DataFrame(list(itertools.product(unique_players['nflId'], frames)), columns=['nflId', 'frameId'])
    scaffold = scaffold.merge(unique_players, on='nflId', how='left')
    players_anim = scaffold.merge(players_all[['nflId', 'frameId', 'x', 'y']], on=['nflId', 'frameId'], how='left')
    players_anim['x'] = players_anim.groupby('nflId')['x'].ffill().bfill()
    players_anim['y'] = players_anim.groupby('nflId')['y'].ffill().bfill()

    # PE Calculation
    pe_dict = compute_pe_all_players(full_df)

    # Ball Trajectory
    start_x, start_y = None, None
    qb_id = None # Added tracking for QB ID
    if not inp_play.empty and "player_role" in inp_play:
        qb = inp_play[inp_play["player_role"] == "Passer"]
        if not qb.empty:
            start_x, start_y = float(qb.iloc[-1]["x"]), float(qb.iloc[-1]["y"])
            qb_id = qb.iloc[0]["nflId"]

    if start_x is None:
        fr0 = players_all[players_all["frameId"] == frames[0]]
        off = fr0[fr0["teamName"] == "Offense"]
        start_x = off["x"].mean() if not off.empty else fr0["x"].mean()
        start_y = off["y"].mean() if not off.empty else fr0["y"].mean()

    # FIX: Force QB to remain static at launch point
    if qb_id is not None and start_x is not None and start_y is not None:
        players_anim.loc[players_anim["nflId"] == qb_id, "x"] = start_x
        players_anim.loc[players_anim["nflId"] == qb_id, "y"] = start_y

    t = np.linspace(0, 1, len(frames))
    ball_x = start_x + (land_x - start_x) * t
    ball_y = start_y + (land_y - start_y) * t
    ball_z = 12 * 4 * t * (1 - t)

    # Metrics (Proximity etc)
    metrics = {}
    frame_groups = {}
    for fr, fr_df in players_all.groupby("frameId"):
        offs = fr_df[fr_df["teamName"].str.lower() == "offense"]
        defs = fr_df[fr_df["teamName"].str.lower() == "defense"]
        frame_groups[fr] = {
            "offs": offs[["nflId", "x", "y"]].to_numpy() if not offs.empty else np.empty((0,3)),
            "defs": defs[["nflId", "x", "y"]].to_numpy() if not defs.empty else np.empty((0,3))
        }
    for pid, info_p in pe_dict.items():
        metrics[pid] = {"avg_proximity": None, "final_sep": None, "cwd": None}
        dists = []
        rows = full_df[full_df["nflId"] == pid]
        if rows.empty: continue
        for _, row in rows.iterrows():
            fr = row["frameId"]; x, y = row["x"], row["y"]
            fg = frame_groups.get(fr)
            if fg is None: continue
            offs = fg["offs"]
            if offs.size == 0: continue
            dx = offs[:,1].astype(float) - float(x); dy = offs[:,2].astype(float) - float(y)
            ds = np.sqrt(dx*dx + dy*dy); dmin = float(np.min(ds)) if ds.size>0 else np.nan
            dists.append(dmin)
        metrics[pid]["avg_proximity"] = float(np.nanmean(dists)) if dists else None

        last_frame_row = rows.sort_values("frameId").iloc[-1]
        lx, ly = last_frame_row["x"], last_frame_row["y"]
        try:
            metrics[pid]["cwd"] = float(np.sqrt((lx - land_x)**2 + (ly - land_y)**2))
        except Exception:
            metrics[pid]["cwd"] = None

    # --- VISUALIZATION (CYBER STYLE) ---
    min_x, max_x = players_all["x"].min() - 10, players_all["x"].max() + 10
    min_y, max_y = players_all["y"].min() - 8, players_all["y"].max() + 8
    if min_x < 30: min_x = max(-2, min_x)
    if max_x > 90: max_x = min(122, max_x)
    if (max_x - min_x) < 40:
        cx = (min_x + max_x)/2
        min_x, max_x = cx - 20, cx + 20
    min_y, max_y = max(-5, min_y), min(58.3, max_y)

    fig = go.Figure()

    # Field Mesh
    fig.add_trace(go.Mesh3d(x=[min_x, min_x, max_x, max_x], y=[min_y, max_y, max_y, min_y], z=[0,0,0,0], color="#0f4c3a", opacity=1.0, showlegend=False, lighting=dict(ambient=0.8, diffuse=0.9)))

    # Lines
    sx, ex = int(np.floor(min_x/10)*10), int(np.ceil(max_x/10)*10)
    for xl in range(sx, ex+1, 10):
        fig.add_trace(go.Scatter3d(x=[xl, xl], y=[min_y, max_y], z=[0.05, 0.05], mode="lines", line=dict(color="rgba(255,255,255,0.7)", width=2), showlegend=False))

    # Goalposts
    if min_x <= 0: add_goalposts(fig, 0)
    if max_x >= 120: add_goalposts(fig, 120)

    # Static Traces (Actual Routes + Ideal PE Routes)
    for pid, p_info in pe_dict.items():
        p_df = full_df[full_df["nflId"] == pid].sort_values("frameId")
        color = "#facc15" if p_info["team"] == "Offense" else "#3b82f6"
        # Make actual route bolder (width=6, opacity=0.6)
        fig.add_trace(go.Scatter3d(x=p_df["x"], y=p_df["y"], z=[0.1]*len(p_df), mode="lines", line=dict(color=color, width=6, dash="dot"), opacity=0.6, showlegend=False))
        ideal = p_info["ideal_pts"]
        if ideal is not None:
             fig.add_trace(go.Scatter3d(x=ideal[:,0], y=ideal[:,1], z=ideal[:,2], mode="lines", line=dict(color="#ec4899", width=3), opacity=0.3, showlegend=False))

    # QB Stick Figure
    add_qb_stick_figure(fig, start_x, start_y, land_x, land_y)

    # Static Ball Route (Bolder)
    fig.add_trace(go.Scatter3d(x=ball_x, y=ball_y, z=ball_z, mode="lines", line=dict(color="#f97316", width=6), opacity=0.8, showlegend=False))

    # Legend Placeholders (Styled as Requested)
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color="#facc15"), name="Offense"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color="#3b82f6"), name="Defense"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line=dict(color="#ec4899", width=4), name="Ideal PE Route"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line=dict(color="#f97316", width=4), name="Ball Route"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=8, color="#f97316"), name="Ball"))

    # --- ANIMATION FRAMES (Cyber Style Stems) ---
    frames_list = []
    short_names = {name: shorten_name(name) for name in players_all["displayName"].unique()}

    init_df = players_anim[players_anim["frameId"] == frames[0]].copy()
    colors = ["#facc15" if t == "Offense" else "#3b82f6" for t in init_df["teamName"]]
    init_df["short"] = init_df["displayName"].map(short_names)
    stagger_z = [2.5 + (k % 5) * 1.5 for k in range(len(init_df))]

    # Init dynamic traces
    stem_x, stem_y, stem_z = [], [], []
    for k, (_, row) in enumerate(init_df.iterrows()):
        stem_x.extend([row['x'], row['x'], None])
        stem_y.extend([row['y'], row['y'], None])
        stem_z.extend([0.5, stagger_z[k], None])

    trace_stems = go.Scatter3d(x=stem_x, y=stem_y, z=stem_z, mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1), showlegend=False)
    trace_markers = go.Scatter3d(x=init_df["x"], y=init_df["y"], z=[0.5]*len(init_df), mode="markers", marker=dict(size=marker_size_players, color=colors, line=dict(color="black", width=1)), showlegend=False)

    # Conditional Labels
    lbl_mode = "text" if show_labels else "none"
    trace_labels = go.Scatter3d(x=init_df["x"], y=init_df["y"], z=stagger_z, mode=lbl_mode, text=init_df["short"].apply(lambda x: f"<b>{x}</b>"), textposition="top center", textfont=dict(size=11, color="white", family="Consolas"), showlegend=False)
    trace_ball = go.Scatter3d(x=[ball_x[0]], y=[ball_y[0]], z=[ball_z[0]], mode="markers", marker=dict(size=8, color="#f97316", line=dict(color="black", width=1)), showlegend=False)

    fig.add_trace(trace_stems)
    fig.add_trace(trace_markers)
    fig.add_trace(trace_labels)
    fig.add_trace(trace_ball)

    n_traces = len(fig.data)
    dynamic_indices = [n_traces-4, n_traces-3, n_traces-2, n_traces-1]

    for i, fr in enumerate(frames):
        fr_df = players_anim[players_anim["frameId"] == fr].copy()
        curr_colors = ["#facc15" if t == "Offense" else "#3b82f6" for t in fr_df["teamName"]]
        fr_df["short"] = fr_df["displayName"].map(short_names)

        sx, sy, sz = [], [], []
        for k, (_, row) in enumerate(fr_df.iterrows()):
            sx.extend([row['x'], row['x'], None])
            sy.extend([row['y'], row['y'], None])
            sz.extend([0.5, stagger_z[k], None])

        t_stems = go.Scatter3d(x=sx, y=sy, z=sz, mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1))
        t_markers = go.Scatter3d(x=fr_df["x"], y=fr_df["y"], z=[0.5]*len(fr_df), mode="markers", marker=dict(size=marker_size_players, color=curr_colors, line=dict(color="black", width=1)))
        t_labels = go.Scatter3d(x=fr_df["x"], y=fr_df["y"], z=stagger_z, mode=lbl_mode, text=fr_df["short"].apply(lambda x: f"<b>{x}</b>"), textposition="top center", textfont=dict(size=11, color="white", family="Consolas"))
        t_ball = go.Scatter3d(x=[ball_x[i]], y=[ball_y[i]], z=[ball_z[i]], mode="markers", marker=dict(size=8, color="#f97316", line=dict(color="black", width=1)))

        frames_list.append(go.Frame(data=[t_stems, t_markers, t_labels, t_ball], traces=dynamic_indices, name=f"f{fr}"))

    fig.frames = frames_list

    # Camera Presets
    if camera_preset == "Default": eye = dict(x=1.6, y=1.1, z=0.7)
    elif camera_preset == "Overhead": eye = dict(x=0.01, y=0.01, z=3.0)
    elif camera_preset == "Sideline": eye = dict(x=2.8, y=0.3, z=0.5)
    else: eye = dict(x=1.6, y=1.1, z=0.7)

    fig.update_layout(
        autosize=True,
        title=dict(text=f"<b>Game {game_id} â€¢ Play {play_id}</b>", font=dict(size=24, color="#f8fafc"), x=0.5, y=0.95),
        scene_camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=-0.1), eye=eye),
        scene=dict(
            xaxis=dict(range=[min_x, max_x], visible=False),
            yaxis=dict(range=[min_y, max_y], visible=False),
            zaxis=dict(range=[0, 15], visible=False),
            aspectmode="manual", aspectratio=dict(x=3.0, y=1.5, z=0.3)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        legend=dict(
            yanchor="top", y=0.88,
            xanchor="right", x=0.95,
            bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="#334155", borderwidth=1,
            font=dict(size=11, color="#f1f5f9"),
            title=dict(text="Legend", font=dict(size=12, color="white"))
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.98,
            bgcolor="rgba(30, 41, 59, 0.9)", bordercolor="#334155", font=dict(color="#ffffff"),
            buttons=[dict(label="â–¶ PLAY", method="animate", args=[None, dict(frame=dict(duration=frame_duration, redraw=True), fromcurrent=True)])]
        )]
    )

    return fig, pe_dict, metrics

# ---------- UI & KPI CONSTRUCTION ----------
# Styles for widgets
widget_style = """
<style>
    .widget-dropdown > select {
        background-color: #1e2538 !important; color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 8px !important;
        padding: 8px 12px !important; font-family: 'Roboto', sans-serif !important; font-size: 1em !important;
    }
    .widget-dropdown > select:focus { border-color: #3b82f6 !important; }
    .widget-chk { color: #f1f5f9 !important; margin-right: 15px; }
    .gen-button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 8px !important;
        font-family: 'Roboto', sans-serif !important; font-weight: 700 !important; font-size: 1em !important;
        text-transform: uppercase !important; cursor: pointer !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .gen-button:hover { background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; transform: translateY(-1px); }
</style>
"""

# Widgets
w_week = widgets.Dropdown(description="Week:", layout=widgets.Layout(width='180px', height='40px'))
w_game = widgets.Dropdown(description="Game ID:", layout=widgets.Layout(width='260px', height='40px'))
w_play = widgets.Dropdown(description="Play ID:", layout=widgets.Layout(width='180px', height='40px'))
w_mode = widgets.Dropdown(description="KPI Mode:", options=["Play","Game"], value="Play", layout=widgets.Layout(width='160px', height='40px'))
camera_presets = widgets.Dropdown(options=["Default","Overhead","Sideline"], value="Default", description="Camera", layout=widgets.Layout(width='180px', height='40px'))

chk_show_labels = widgets.Checkbox(value=True, description="Labels", layout=widgets.Layout(width='auto'))
chk_boxed_highlights = widgets.Checkbox(value=False, description="Boxed", layout=widgets.Layout(width='auto'))
chk_landing_goal = widgets.Checkbox(value=True, description="Goal", layout=widgets.Layout(width='auto'))

slider_marker = widgets.IntSlider(value=6, min=4, max=14, step=1, description='Size', continuous_update=False, layout=widgets.Layout(width='200px'))
slider_speed = widgets.IntSlider(value=80, min=20, max=250, step=10, description='Speed (ms)', continuous_update=False, layout=widgets.Layout(width='200px'))

w_week.add_class("widget-dropdown")
w_game.add_class("widget-dropdown")
w_play.add_class("widget-dropdown")
w_mode.add_class("widget-dropdown")
camera_presets.add_class("widget-dropdown")

btn_gen = widgets.Button(description="GENERATE VISUALIZATION", layout=widgets.Layout(width='100%', height='45px', margin='15px 0 0 0'))
btn_gen.add_class("gen-button")

# Layout Containers
out_viz = widgets.Output(layout=widgets.Layout(width='65%', height='720px'))
out_kpi = widgets.Output(layout=widgets.Layout(width='35%', height='720px'))
out_explain = widgets.Output(layout=widgets.Layout(width='100%', height='320px'))  # white box placed beneath visualization

def update_selectors():
    if _df.empty: return
    weeks = sorted(_df["week"].dropna().unique().astype(int))
    w_week.options = weeks
    if weeks: w_week.value = weeks[0]

def update_games(*_):
    if _df.empty: return
    wk = w_week.value
    gms = sorted(_df[_df["week"] == wk]["gameId"].dropna().unique())
    w_game.options = gms
    if gms: w_game.value = gms[0]

def update_plays(*_):
    if _df.empty or w_game.value is None: return
    mask = (_df["week"] == w_week.value) & (_df["gameId"] == w_game.value)
    pls = sorted(_df[mask]["playId"].dropna().unique())
    w_play.options = pls
    if pls: w_play.value = pls[0]

w_week.observe(update_games, names="value")
w_game.observe(update_plays, names="value")

def compute_pe_for_game(game_id):
    if input_df.empty and output_df.empty: return {}
    inp_game = input_df[input_df["gameId"] == int(game_id)].copy() if not input_df.empty else pd.DataFrame()
    out_game = output_df[output_df["gameId"] == int(game_id)].copy() if not output_df.empty else pd.DataFrame()
    all_tracks = []
    for pid in set(list(inp_game.get("nflId", [])) + list(out_game.get("nflId", []))):
        pin = inp_game[inp_game["nflId"] == pid].copy(); pout = out_game[out_game["nflId"] == pid].copy()
        for df_ in (pin, pout):
            for col in ["x","y","frameId","s"]:
                if col in df_.columns: df_[col] = pd.to_numeric(df_[col], errors="coerce")
        p_full = pd.concat([pin, pout], ignore_index=True)
        if not p_full.empty: all_tracks.append(p_full)
    if not all_tracks: return {}
    full_game_df = pd.concat(all_tracks, ignore_index=True)
    return compute_pe_all_players(full_game_df)

def get_stats_from_pe_dict(pe_dict):
    if not pe_dict:
        return None, None, 0.0

    rows = sorted(pe_dict.values(), key=lambda x: x["pe"], reverse=True)
    avg_pe = np.mean([r["pe"] for r in rows])

    # Top Offense
    off_players = [r for r in rows if str(r.get("team","")).lower().startswith("off")]
    top_off = off_players[0] if off_players else None

    # Lowest Overall
    lowest = rows[-1] if rows else None

    return top_off, lowest, avg_pe

def build_kpi_html(pe_play, pe_game, game_id, play_id):
    # Calculate Play Stats
    p_top, p_low, p_avg = get_stats_from_pe_dict(pe_play)

    # Calculate Game Stats
    g_top, g_low, g_avg = get_stats_from_pe_dict(pe_game)

    def create_card_html(title, player_name, score, sub_text, color_class):
        return f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{player_name}</div>
            <div class="kpi-desc">PE Score: {score}</div>
            <div class="kpi-simple">{sub_text}</div>
        </div>
        """

    # List aggregations (Using Play Data for lists as usual)
    rows_play = sorted(pe_play.values(), key=lambda x: x["pe"], reverse=True) if pe_play else []
    defenders = [r for r in rows_play if str(r.get("team","")).lower().startswith("def")]
    if not defenders: defenders = [r for r in rows_play if "def" in str(r.get("team","")).lower()]
    top_def = sorted(defenders, key=lambda x: x["pe"], reverse=True)[:5]
    active_def = sorted(defenders, key=lambda x: x.get("total_dist",0.0), reverse=True)[:5]

    def generate_list_html(items, metric_key, unit=""):
        if not items:
            return "<div style='color:#64748b; font-size:0.9em;'>No data</div>"
        lis = ""
        for p in items:
            val = p.get(metric_key, 0)
            if isinstance(val, float):
                fmt = f"{val:.3f}" if val < 2 else f"{val:.1f}"
            else:
                fmt = str(val)
            lis += f"<li><span>{p['name']}</span><span class='tiny-val'>{fmt}{unit}</span></li>"
        return f"<ul class='tiny-list'>{lis}</ul>"

    # HTML Construction
    # Left Column (Game)
    g_top_html = create_card_html("Top Offense (Game)", g_top['name'] if g_top else "N/A", f"{g_top['pe']:.3f}" if g_top else "N/A", "Best in Game", "card-green")
    g_low_html = create_card_html("Lowest PE (Game)", g_low['name'] if g_low else "N/A", f"{g_low['pe']:.3f}" if g_low else "N/A", "Worst in Game", "card-red")
    g_avg_html = create_card_html("Avg PE (Game)", f"{g_avg:.3f}", "Game Average", "Across all plays", "card-gold")

    # Right Column (Play)
    p_top_html = create_card_html("Top Offense (Play)", p_top['name'] if p_top else "N/A", f"{p_top['pe']:.3f}" if p_top else "N/A", "Best this Play", "card-green")
    p_low_html = create_card_html("Lowest PE (Play)", p_low['name'] if p_low else "N/A", f"{p_low['pe']:.3f}" if p_low else "N/A", "Worst this Play", "card-red")
    p_avg_html = create_card_html("Avg PE (Play)", f"{p_avg:.3f}", "Play Average", "Current play only", "card-gold")

    # Table Rows
    table_rows = ""
    for i, p in enumerate(rows_play[:8]): # Show top 8 for space
        border_class = "row-best" if p["pe"] > 0.9 else ("row-worst" if p["pe"] < 0.6 else "")
        table_rows += f"""
        <div class="player-row {border_class}">
            <span class="p-rank">#{i+1}</span>
            <span class="p-name">{p['name']}</span>
            <span class="p-score">{p['pe']:.3f}</span>
        </div>
        """

    html = f"""
    {KPI_STYLE}
    <div class="kpi-panel">
        <div class="panel-header">
            <span>ğŸš€ DUAL-STAT FEED</span>
        </div>
        <div class="panel-sub">Game {game_id} â€¢ Play {play_id}</div>

        <div style="display:flex; gap:15px; margin-bottom:15px;">
            <!-- Left Column: Game Level -->
            <div style="flex:1;">
                <div class="kpi-col-header">GAME LEVEL</div>
                {g_top_html}
                {g_low_html}
                {g_avg_html}
            </div>

            <!-- Right Column: Play Level -->
            <div style="flex:1;">
                <div class="kpi-col-header">PLAY LEVEL</div>
                {p_top_html}
                {p_low_html}
                {p_avg_html}
            </div>
        </div>

        <div style="display:flex; gap:10px; margin-bottom:15px;">
            <div class="kpi-card" style="flex:1">
                <div class="kpi-title">Top Defenders (Play)</div>
                {generate_list_html(top_def, 'pe')}
            </div>
            <div class="kpi-card" style="flex:1">
                <div class="kpi-title">Most Active (Play)</div>
                {generate_list_html(active_def, 'total_dist', ' yds')}
            </div>
        </div>

        <div class="rank-section-header">
            <span>ğŸ”¥ Play Leaderboard</span>
            <span style="font-size:0.8em; opacity:0.7">PE SCORE</span>
        </div>

        {table_rows}
    </div>
    """
    return html

def build_explanation_html_from_pe(pe_play, pe_game, game_id, play_id):
    """
    Build the coach's white note style HTML using pe_play and pe_game dictionaries.
    Safe formatting: compute display strings first, then insert into the f-string.
    """
    # Helper: pick top / lowest / avg
    def pick_top_low_avg(pe_dict):
        if not pe_dict:
            return None, None, 0.0
        rows = sorted(pe_dict.values(), key=lambda x: x.get("pe", 0.0), reverse=True)
        top = rows[0] if rows else None
        low = rows[-1] if rows else None
        avg = float(np.mean([r.get("pe", 0.0) for r in rows])) if rows else 0.0
        return top, low, avg

    p_top, p_low, p_avg = pick_top_low_avg(pe_play or {})
    g_top, g_low, g_avg = pick_top_low_avg(pe_game or {})

    # Safe formatting for display
    p_top_name = p_top["name"] if p_top else "N/A"
    p_top_pe   = f"{p_top['pe']:.3f}" if (p_top and isinstance(p_top.get("pe"), (float, int))) else "N/A"

    p_low_name = p_low["name"] if p_low else "N/A"
    p_low_pe   = f"{p_low['pe']:.3f}" if (p_low and isinstance(p_low.get("pe"), (float, int))) else "N/A"

    g_top_name = g_top["name"] if g_top else "N/A"
    g_top_pe   = f"{g_top['pe']:.3f}" if (g_top and isinstance(g_top.get("pe"), (float, int))) else "N/A"

    g_low_name = g_low["name"] if g_low else "N/A"
    g_low_pe   = f"{g_low['pe']:.3f}" if (g_low and isinstance(g_low.get("pe"), (float, int))) else "N/A"

    p_avg_str = f"{p_avg:.3f}" if isinstance(p_avg, (float, int)) else "N/A"
    g_avg_str = f"{g_avg:.3f}" if isinstance(g_avg, (float, int)) else "N/A"

    # Lists: top defenders and most active (from play-level pe_play)
    rows_play = sorted((pe_play or {}).values(), key=lambda x: x.get("pe", 0.0), reverse=True)
    defenders = [r for r in rows_play if str(r.get("team", "")).lower().startswith("def")]
    # fallback if team label not standard
    if not defenders:
        defenders = [r for r in rows_play if "def" in str(r.get("team", "")).lower()]

    top_def = sorted(defenders, key=lambda x: x.get("pe", 0.0), reverse=True)[:5]
    most_active = sorted((r for r in rows_play), key=lambda x: x.get("total_dist", 0.0), reverse=True)[:5]

    def list_html(items, metric_key, unit=""):
        if not items:
            return "<div style='color:#64748b; font-size:0.9em;'>No data</div>"
        lis = ""
        for it in items:
            name = it.get("name","N/A")
            val = it.get(metric_key, None)
            if val is None:
                val_str = "N/A"
            else:
                if metric_key == "pe":
                    val_str = f"{val:.3f}" if isinstance(val, (float,int)) else str(val)
                else:
                    # distance or other numeric
                    val_str = f"{val:.1f}{unit}" if isinstance(val, (float,int)) else str(val)
            lis += f"<li class='sub-bullet-item' style='display:flex; justify-content:space-between;'><span>{name}</span><span style='font-weight:700'>{val_str}</span></li>"
        return f"<ul class='sub-bullet-list' style='margin:0; padding-left:10px;'>{lis}</ul>"

    # Build the white coach-style box content (keeps the original visual structure)
    html = f"""
    <div style="background:#ffffff; color:#0f172a; padding:18px; border-radius:12px; border:1px solid #e6eef6; font-family:Roboto, sans-serif; max-width:900px;">

        <!-- HEADER -->
        <div style="font-weight:800; font-size:1.15em; color:#0f172a; margin-bottom:6px; border-bottom:3px solid #f59e0b; display:inline-block; padding-bottom:4px;">
            COACH'S GAME NOTE
            <span style="font-weight:400; color:#64748b; font-size:0.85em; margin-left:10px;">Play {play_id} Breakdown</span>
        </div>

        <!-- HOW TO READ SECTION -->
        <div style="background:#f8fafc; border-left:4px solid #cbd5e1; padding:12px 14px; border-radius:6px; margin-top:12px;">
            <div style="font-weight:700; color:#334155; margin-bottom:6px;">ğŸ”� How to Read this Play</div>
            <div style="color:#475569; font-size:0.92em;">
                <b style="color:#f59e0b">OFFENSE</b> (Yellow) vs
                <b style="color:#3b82f6">DEFENSE</b> (Blue).
                The <b style="color:#ef4444">RED HALO</b> marks the Best Defender.
                Solid Line = Actual Route.
            </div>
        </div>

        <!-- KPI SECTION -->
        <div style="background:#f8fafc; border-left:4px solid #cbd5e1; padding:12px 14px; border-radius:6px; margin-top:12px;">
            <div style="font-weight:700; color:#334155; margin-bottom:8px;">ğŸ“Š KPI METRICS ANALYSIS</div>

            <ul style="margin:0; padding-left:0; list-style:none; color:#475569; font-size:0.92em;">

                <!-- AVG PE (GAME) -->
                <li style="margin-bottom:12px;">
                    <strong style="color:#334155">Avg PE (Game): {g_avg_str}</strong>
                    <ul style="margin:6px 0 0 10px; padding-left:0; list-style:none;">
                        <li style="color:#64748b; font-size:0.9em;">Definition: The mean path efficiency of all tracked players across the entire game.</li>
                        <li style="color:#64748b; font-size:0.9em;">Observation: Reflects overall movement discipline and route sharpness over all plays.</li>
                        <li style="color:#64748b; font-size:0.9em;">Insight: Higher Avg PE suggests consistency and strong offensive structure.</li>
                    </ul>
                </li>

                <!-- AVG PE (PLAY) -->
                <li style="margin-bottom:12px;">
                    <strong style="color:#334155">Avg PE (Play): {p_avg_str}</strong>
                    <ul style="margin:6px 0 0 10px; padding-left:0; list-style:none;">
                        <li style="color:#64748b; font-size:0.9em;">Definition: Average path efficiency of all players on this play.</li>
                        <li style="color:#64748b; font-size:0.9em;">Observation: Measures how clean and direct the movement patterns were.</li>
                        <li style="color:#64748b; font-size:0.9em;">Insight: Higher values indicate strong timing, spacing, and reduced wasted motion.</li>
                    </ul>
                </li>

                <!-- TOP OFFENSE (PLAY) -->
                <li style="margin-bottom:12px;">
                    <strong style="color:#334155">Top Offense (Play): {p_top_name} â€” PE {p_top_pe}</strong>
                    <ul style="margin:6px 0 0 10px; padding-left:0; list-style:none;">
                        <li style="color:#64748b; font-size:0.9em;">Definition: Offensive player with the highest path efficiency on this play.</li>
                        <li style="color:#64748b; font-size:0.9em;">Observation: Demonstrated the most direct and efficient movement relative to the play design.</li>
                        <li style="color:#64748b; font-size:0.9em;">Insight: Indicates strong route execution and play understanding.</li>
                    </ul>
                </li>

                <!-- LOWEST PE (PLAY) -->
                <li style="margin-bottom:12px;">
                    <strong style="color:#334155">Lowest PE (Play): {p_low_name} â€” PE {p_low_pe}</strong>
                    <ul style="margin:6px 0 0 10px; padding-left:0; list-style:none;">
                        <li style="color:#64748b; font-size:0.9em;">Definition: Player with the least efficient movement on this play.</li>
                        <li style="color:#64748b; font-size:0.9em;">Observation: Movement included rounding, drift, or delayed redirection.</li>
                        <li style="color:#64748b; font-size:0.9em;">Insight: Coaching opportunityâ€”may indicate hesitation or misread routes.</li>
                    </ul>
                </li>

                <!-- TOP OFFENSE (GAME) -->
                <li style="margin-bottom:12px;">
                    <strong style="color:#334155">Top Offense (Game): {g_top_name} â€” PE {g_top_pe}</strong>
                    <ul style="margin:6px 0 0 10px; padding-left:0; list-style:none;">
                        <li style="color:#64748b; font-size:0.9em;">Definition: Player showing the highest cumulative PE over the entire game.</li>
                        <li style="color:#64748b; font-size:0.9em;">Observation: Consistently displayed sharp, efficient movements across all plays.</li>
                        <li style="color:#64748b; font-size:0.9em;">Insight: Indicates elite route discipline and sustained performance.</li>
                    </ul>
                </li>

                <!-- LOWEST PE (GAME) -->
                <li style="margin-bottom:12px;">
                    <strong style="color:#334155">Lowest PE (Game): {g_low_name} â€” PE {g_low_pe}</strong>
                    <ul style="margin:6px 0 0 10px; padding-left:0; list-style:none;">
                        <li style="color:#64748b; font-size:0.9em;">Definition: Player with the lowest cumulative PE across the full game.</li>
                        <li style="color:#64748b; font-size:0.9em;">Observation: Showed inefficient or indirect movement patterns over multiple plays.</li>
                        <li style="color:#64748b; font-size:0.9em;">Insight: May require coaching on alignment, play recognition or pursuit angles.</li>
                    </ul>
                </li>

            </ul>
        </div>

       </div>
     """

    return html

# --- UI BUILD ---
controls_container = widgets.VBox([
    widgets.HTML(widget_style),
    widgets.HBox([w_week, w_game, w_play, w_mode], layout=widgets.Layout(gap='10px', margin='0 0 10px 0', flex_flow='row wrap')),
    widgets.HBox([camera_presets, slider_marker, slider_speed], layout=widgets.Layout(gap='10px', margin='0 0 10px 0', flex_flow='row wrap')),
    widgets.HBox([chk_show_labels, chk_boxed_highlights, chk_landing_goal], layout=widgets.Layout(gap='15px')),
    btn_gen
], layout=widgets.Layout(background_color='#151b2b', padding='20px', border_radius='16px', border='1px solid rgba(255, 255, 255, 0.1)', margin='0 0 25px 0'))

left_col = widgets.VBox([out_viz, out_explain], layout=widgets.Layout(width='65%'))
ui = widgets.VBox([
    widgets.HTML("""
    <div style="background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
        <h2 style="color: #f8fafc; margin: 0; font-family: 'Roboto', sans-serif; font-weight: 900; letter-spacing: 1px;">
            ğŸ�ˆ NFL PATH EFFICIENCY <span style="color:#ec4899">3D</span>
        </h2>
        <p style="color: #94a3b8; margin: 5px 0 0 0; font-family: 'Roboto', sans-serif;">
            Broadcast Reconstruction â€¢ Bezier Curve Analysis â€¢ Player Tracking
        </p>
    </div>
    """),
    controls_container,
    widgets.HBox([left_col, out_kpi], layout=widgets.Layout(gap='20px'))
])

def on_click_gen(_):
    out_viz.clear_output(); out_kpi.clear_output(); out_explain.clear_output()
    gid, pid, mode = w_game.value, w_play.value, w_mode.value
    if gid is None or pid is None: return

    show_lbl = chk_show_labels.value
    m_size = slider_marker.value
    speed = slider_speed.value
    cam = camera_presets.value

    # Build fig & play-level PE dict
    fig, pe_play, metrics = create_3d_pe_animation(gid, pid,
                                          marker_size_players=m_size,
                                          show_labels=show_lbl,
                                          frame_duration=speed,
                                          camera_preset=cam)

    # Always compute Game PE now for the left column
    pe_game = compute_pe_for_game(gid)

    fig.write_html(
    "/kaggle/working/code2_3d_pe/3d_path_efficiency.html"
      )


    with out_viz:
        fig.show()

    with out_kpi:
        display(HTML(build_kpi_html(pe_play, pe_game, gid, pid)))

    # Build coach's white note from pe dicts and display it
    with out_explain:
        coach_html = build_explanation_html_from_pe(pe_play, pe_game, gid, pid)
        display(HTML(coach_html))
        # auto-scroll so coach note is visible in notebook
        display(Javascript("setTimeout(function() { window.scrollTo({ top: 0, behavior: 'smooth' }); }, 50);"))

btn_gen.on_click(on_click_gen)

# Initialize selectors
update_selectors()
update_games()
update_plays()

display(ui)



# Updated script â€” Play & Game ATD + Top/Bottom 5 performers; removed DCSI, BFSG, GSS, SQI, CDS, CWD, PE
import os
import math
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# --- Colab widget support (optional) ---
# try:
#     from google.colab import output as colab_output
#     colab_output.enable_custom_widget_manager()
# except Exception:
#     pass

pio.renderers.default = "iframe"
print("Renderer:", pio.renderers.default)

# ---------- CONFIG: CHANGE PATHS IF NEEDED ----------
INPUT_FILE  = "/kaggle/input/nfl-analytics/combined_input_2023.csv"
OUTPUT_FILE = "/kaggle/input/nfl-analytics/combined_output_2023.csv"
SUPP_FILE   = "/kaggle/input/nfl-analytics/supplementary_data.csv"

# ---------- CSS STYLES (Dark "Cyber" Theme) ----------
KPI_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');

    :root {
        --bg-dark: #0b0f19;
        --panel-bg: #151b2b;
        --card-bg: #1e2538;
        --text-main: #f1f5f9;
        --text-sub: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-gold: #f59e0b;
        --neon-border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* GENERAL PANEL STYLES */
    .kpi-panel, .rank-panel, .explain-panel {
        background-color: var(--panel-bg);
        color: var(--text-main);
        padding: 20px;
        border-radius: 16px;
        font-family: 'Roboto', sans-serif;
        height: 600px;
        overflow-y: auto;
        border: var(--neon-border);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }

    /* HEADERS */
    .panel-header {
        font-size: 1.1em;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-main);
        margin-bottom: 4px;
        border-bottom: 2px solid #334155;
        padding-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .panel-sub {
        font-size: 0.8em;
        color: var(--text-sub);
        margin-bottom: 15px;
        font-weight: 400;
    }

    /* KPI CARDS */
    .kpi-card {
        background: linear-gradient(145deg, #1e2538, #171d2d);
        border: 1px solid rgba(255,255,255,0.05);
        border-left: 4px solid var(--accent-blue);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .kpi-title {
        font-size: 0.75em;
        font-weight: 700;
        color: var(--text-sub);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.6em;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    }
    .kpi-desc {
        font-size: 0.8em;
        color: #64748b;
        margin-top: 4px;
        line-height: 1.3;
    }
    .kpi-simple {
        font-size: 0.75em;
        color: var(--accent-blue);
        margin-top: 6px;
        font-style: italic;
        padding-top: 6px;
        border-top: 1px dashed rgba(255, 255, 255, 0.1);
    }

    /* KPI ACCENTS */
    .card-red { border-left-color: var(--accent-red); }
    .card-green { border-left-color: var(--accent-green); }
    .card-gold { border-left-color: var(--accent-gold); }

    /* PLAYER RANKINGS */
    .mom-card {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #fff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);
        position: relative;
        overflow: hidden;
    }
    .mom-badge {
        background: rgba(0,0,0,0.3);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7em;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 10px;
    }
    .mom-name {
        font-size: 1.6em;
        font-weight: 900;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 5px;
    }
    .mom-stat {
        font-size: 1.1em;
        font-weight: 700;
        opacity: 0.95;
    }

    .rank-section-header {
        font-size: 0.85em;
        color: var(--text-sub);
        text-transform: uppercase;
        font-weight: 700;
        margin: 20px 0 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        border-bottom: 1px solid #334155;
        padding-bottom: 5px;
    }

    .player-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255,255,255,0.03);
        padding: 10px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        border-left: 3px solid transparent;
        transition: background 0.2s;
    }
    .player-row:hover { background: rgba(255,255,255,0.06); }
    .row-best { border-left-color: var(--accent-green); }
    .row-worst { border-left-color: var(--accent-red); }

    .p-rank { font-size: 0.8em; color: #64748b; width: 20px; }
    .p-name { font-weight: 600; color: #e2e8f0; font-size: 0.9em; flex-grow: 1; }
    .p-score { font-family: 'Roboto Mono', monospace; font-weight: 700; color: #cbd5e1; }

    /* EXPLANATION / COACHES NOTE */
    .explain-container {
        background: #ffffff;
        color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        font-family: 'Roboto', sans-serif;
    }
    .explain-title {
        font-size: 1.2em;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 15px;
        border-bottom: 3px solid #f59e0b;
        display: inline-block;
        padding-bottom: 2px;
    }
    .section-box {
        background: #f8fafc;
        border-left: 4px solid #cbd5e1;
        padding: 12px 16px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
    }
    .section-header {
        font-weight: 700;
        color: #334155;
        margin-bottom: 8px;
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .bullet-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .bullet-item {
        position: relative;
        padding-left: 18px;
        margin-bottom: 8px;
        font-size: 0.9em;
        line-height: 1.5;
        color: #475569;
    }
    .bullet-item::before {
        content: 'â€¢';
        position: absolute;
        left: 0;
        color: #f59e0b;
        font-weight: bold;
    }
    .sub-bullet-list {
        list-style: none;
        padding-left: 10px;
        margin-top: 4px;
        border-left: 2px solid #e2e8f0;
        margin-left: 5px;
    }
    .sub-bullet-item {
        font-size: 0.85em;
        margin-bottom: 4px;
        padding-left: 10px;
        color: #64748b;
    }
</style>
"""

# ---------- GLOBALS ----------
input_df  = pd.DataFrame()
output_df = pd.DataFrame()
supp_df   = pd.DataFrame()
_df       = pd.DataFrame()

# ---------- LOAD DATA ----------
def load_all():
    global input_df, output_df, supp_df, _df

    # Input file
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        input_df = pd.DataFrame()
    else:
        inp = pd.read_csv(INPUT_FILE, low_memory=False)
        col_map = {"game_id": "gameId", "play_id": "playId", "frame_id": "frameId", "player_name": "displayName",
                   "player_side": "teamName", "nfl_id": "nflId", "week": "week"}
        inp.rename(columns=col_map, inplace=True)
        for c in ["x", "y", "s", "dir", "frameId", "ball_land_x", "ball_land_y", "week"]:
            if c in inp.columns:
                inp[c] = pd.to_numeric(inp[c], errors="coerce")
        if {"x", "y", "frameId"}.issubset(inp.columns):
            inp = inp.dropna(subset=["x", "y", "frameId"])
        input_df = inp
        _df = inp.copy()
        print("Input rows:", input_df.shape[0])

    # Output file
    if not os.path.exists(OUTPUT_FILE):
        print(f"Output file not found: {OUTPUT_FILE}")
        output_df = pd.DataFrame()
    else:
        out = pd.read_csv(OUTPUT_FILE, low_memory=False)
        col_map_out = {"game_id": "gameId", "play_id": "playId", "frame_id": "frameId", "nfl_id": "nflId", "week": "week"}
        out.rename(columns=col_map_out, inplace=True)
        for c in ["x", "y", "frameId", "week"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        if {"x", "y", "frameId"}.issubset(out.columns):
            out = out.dropna(subset=["x", "y", "frameId"])
        output_df = out
        print("Output rows:", output_df.shape[0])

    # Supplementary
    if os.path.exists(SUPP_FILE):
        s = pd.read_csv(SUPP_FILE, low_memory=False)
        if "game_id" in s.columns:
            s = s.rename(columns={"game_id": "gameId"})
        if "play_id" in s.columns:
            s = s.rename(columns={"play_id": "playId"})
        supp_df = s
        print("Supplementary rows:", supp_df.shape[0])
    else:
        print(f"Supplementary not found: {SUPP_FILE}")
        supp_df = pd.DataFrame()

load_all()

# ---------- GOALPOSTS ----------
def add_goalposts(fig, x_pos, y_center=26.65, z_crossbar=3.33, z_top=13.33, width=6.17, color="#facc15"):
    hw = width / 2.0
    fig.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_center, y_center], z=[0, z_crossbar], mode="lines", line=dict(color=color, width=12), hoverinfo="text", text="Goal Post Base", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_center - hw, y_center + hw], z=[z_crossbar, z_crossbar], mode="lines", line=dict(color=color, width=12), hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_center - hw, y_center - hw], z=[z_crossbar, z_top], mode="lines", line=dict(color=color, width=10), hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x_pos, x_pos], y=[y_center + hw, y_center + hw], z=[z_crossbar, z_top], mode="lines", line=dict(color=color, width=10), hoverinfo="none", showlegend=False))

# ---------- QB STICK FIGURE ----------
def add_qb_stick_figure(fig, x, y, land_x, land_y, color="#fbbf24"):
    if x is None or y is None or land_x is None or land_y is None: return
    dx = land_x - x; dy = land_y - y
    norm = np.hypot(dx, dy)
    ux, uy = (dx/norm, dy/norm) if norm > 0 else (1, 0)
    px, py = -uy, ux
    shoulder_w = 0.3; arm_len = 0.4; leg_spread = 0.3
    body_x = [x, x]; body_y = [y, y]; body_z = [1.0, 1.7]
    ll_x = [x, x - px * leg_spread * 0.5 - ux * 0.2]; ll_y = [y, y - py * leg_spread * 0.5 - uy * 0.2]; ll_z = [1.0, 0.0]
    rl_x = [x, x + px * leg_spread * 0.5 + ux * 0.3]; rl_y = [y, y + py * leg_spread * 0.5 + uy * 0.3]; rl_z = [1.0, 0.0]
    ra_x = [x + px * shoulder_w, x + px * shoulder_w + ux * arm_len]; ra_y = [y + py * shoulder_w, y + py * shoulder_w + uy * arm_len]; ra_z = [1.7, 2.1]
    la_x = [x - px * shoulder_w, x - px * shoulder_w - ux * 0.2]; la_y = [y - py * shoulder_w, y - py * shoulder_w - uy * 0.2]; la_z = [1.7, 1.4]
    all_x = body_x + [None] + ll_x + [None] + rl_x + [None] + ra_x + [None] + la_x
    all_y = body_y + [None] + ll_y + [None] + rl_y + [None] + ra_y + [None] + la_y
    all_z = body_z + [None] + ll_z + [None] + rl_z + [None] + ra_z + [None] + la_z
    fig.add_trace(go.Scatter3d(x=all_x, y=all_y, z=all_z, mode="lines", line=dict(color=color, width=5), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[1.9], mode="markers", marker=dict(size=6, color=color, line=dict(color='black', width=1)), showlegend=False))

# ---------- PATH / BALL HELPERS ----------
def get_ball_parabola(start_x, start_y, land_x, land_y, n_frames):
    if n_frames <= 0: return np.array([]), np.array([]), np.array([])
    t = np.linspace(0.0, 1.0, n_frames)
    ball_x = start_x + (land_x - start_x) * t
    ball_y = start_y + (land_y - start_y) * t
    ball_z = 12.0 * 4.0 * t * (1.0 - t)
    return ball_x, ball_y, ball_z

# ---------- ATD CALCULATION ----------
def compute_atd_frames(players_anim_df, frames, ball_x, ball_y, ball_z, target_nflId, adv_margin=0.5, catch_radius=1.5, defender_vertical_z=0.9):
    N = len(frames)
    if N == 0: return [], 0.0, {}
    frame_groups = {fr: players_anim_df[players_anim_df["frameId"] == fr] for fr in frames}
    atd_flags = []
    for i, fr in enumerate(frames):
        fr_df = frame_groups.get(fr)
        if fr_df is None or fr_df.empty: atd_flags.append(False); continue
        bx, by, bz = float(ball_x[i]), float(ball_y[i]), float(ball_z[i])
        wr_row = fr_df[fr_df["nflId"] == target_nflId]
        if wr_row.empty: atd_flags.append(False); continue
        wr_x, wr_y = float(wr_row["x"].iloc[0]), float(wr_row["y"].iloc[0])
        wr_dist_3d = math.hypot(math.hypot(wr_x - bx, wr_y - by), 0.0 - bz)
        defs_df = fr_df[fr_df["nflId"] != target_nflId]
        if "teamName" in fr_df.columns:
            tgt_team = fr_df.loc[fr_df["nflId"] == target_nflId, "teamName"].iloc[0]
            defs_df = fr_df[(fr_df["teamName"] == "Defense") | (fr_df["teamName"] != tgt_team)]
        if defs_df.empty: atd_flags.append(True); continue
        min_def_dist = np.inf
        for _, r in defs_df.iterrows():
            d = math.hypot(math.hypot(float(r["x"]) - bx, float(r["y"]) - by), defender_vertical_z - bz)
            if d < min_def_dist: min_def_dist = d
        dominant = (wr_dist_3d <= (min_def_dist - adv_margin)) or (wr_dist_3d <= catch_radius and min_def_dist > wr_dist_3d)
        atd_flags.append(dominant)
    atd_pct = (sum(atd_flags) / float(N)) * 100.0 if N > 0 else 0.0
    return atd_flags, atd_pct, {}

# ---------- Fast helpers to compute play & game ATD ----------
def compute_play_level_atd(game_id, play_id):
    """
    Build minimal play-level scaffolding and compute ATD% using compute_atd_frames.
    Returns: atd_pct (float) or None if cannot compute
    """
    inp_play = input_df[(input_df["gameId"] == game_id) & (input_df["playId"] == play_id)].copy()
    out_play = output_df[(output_df["gameId"] == game_id) & (output_df["playId"] == play_id)].copy()

    if (inp_play.empty) and (out_play.empty):
        return None

    # Build tracks for this play
    full_tracks = []
    all_pids = set(inp_play.get("nflId", []).unique()) | set(out_play.get("nflId", []).unique())
    info = {}
    if not inp_play.empty:
        tmp = inp_play[["nflId", "displayName", "teamName"]].drop_duplicates()
        info = {pid: {"displayName": row["displayName"], "teamName": row["teamName"]} for pid, row in tmp.set_index("nflId").iterrows()}

    for pid in all_pids:
        p1 = inp_play[inp_play["nflId"] == pid]
        p2 = out_play[out_play["nflId"] == pid]
        if not p1.empty and not p2.empty:
            p2 = p2.copy()
            p2["frameId"] += p1["frameId"].max()
        pf = pd.concat([p1, p2], ignore_index=True)
        if not pf.empty:
            if "displayName" not in pf.columns or pf["displayName"].isna().all():
                pf["displayName"] = info.get(pid, {}).get("displayName", str(pid))
            if "teamName" not in pf.columns or pf["teamName"].isna().all():
                pf["teamName"] = info.get(pid, {}).get("teamName", "Unknown")
            full_tracks.append(pf)

    if not full_tracks:
        return None

    full_df = pd.concat(full_tracks, ignore_index=True)
    players_all = full_df.dropna(subset=["x", "y", "frameId"]).sort_values("frameId")
    if players_all.empty:
        return None

    frames = sorted(players_all["frameId"].unique())

    # scaffold -> players_anim
    scaffold = pd.DataFrame(list(itertools.product(players_all['nflId'].unique(), frames)), columns=['nflId', 'frameId'])
    info_df = players_all[['nflId', 'displayName', 'teamName']].drop_duplicates('nflId')
    scaffold = scaffold.merge(info_df, on='nflId', how='left')
    players_anim = scaffold.merge(players_all[['nflId', 'frameId', 'x', 'y']], on=['nflId', 'frameId'], how='left')
    players_anim['x'] = players_anim.groupby('nflId')['x'].ffill().bfill()
    players_anim['y'] = players_anim.groupby('nflId')['y'].ffill().bfill()
    players_anim = players_anim.dropna(subset=['x', 'y'])

    # ball parabola: need start (QB) and landing
    start_x, start_y = None, None
    if "player_role" in inp_play.columns:
        qb = inp_play[inp_play["player_role"] == "Passer"]
        if not qb.empty:
            start_x, start_y = float(qb.iloc[-1]["x"]), float(qb.iloc[-1]["y"])
    if start_x is None:
        fr0 = players_all[players_all["frameId"] == frames[0]]
        off = fr0[fr0["teamName"] == "Offense"]
        start_x = float(off["x"].mean()) if not off.empty else float(fr0["x"].mean())
        start_y = float(off["y"].mean()) if not off.empty else float(fr0["y"].mean())

    land_x = None; land_y = None
    if "ball_land_x" in inp_play.columns and inp_play["ball_land_x"].notna().any(): land_x = float(inp_play["ball_land_x"].iloc[0])
    if "ball_land_y" in inp_play.columns and inp_play["ball_land_y"].notna().any(): land_y = float(inp_play["ball_land_y"].iloc[0])
    if land_x is None or land_y is None:
        land_x = float(players_all["x"].mean())
        land_y = float(players_all["y"].mean())

    ball_x, ball_y, ball_z = get_ball_parabola(start_x, start_y, land_x, land_y, len(frames))

    # target (try targeted receiver)
    target_nflId = None
    if "player_role" in full_df.columns:
        tgt = full_df[full_df["player_role"] == "Targeted Receiver"]
        if not tgt.empty:
            try:
                target_nflId = int(tgt["nflId"].iloc[0])
            except:
                target_nflId = tgt["nflId"].iloc[0]
    if target_nflId is None:
        try:
            last_fr = frames[-1]
            last_off = players_all[(players_all["frameId"] == last_fr) & (players_all["teamName"] == "Offense")]
            if not last_off.empty:
                target_nflId = int(last_off.sort_values("x", ascending=False).iloc[0]["nflId"])
            else:
                target_nflId = int(players_all.iloc[0]["nflId"])
        except Exception:
            target_nflId = int(players_all.iloc[0]["nflId"])

    atd_flags, atd_pct, _ = compute_atd_frames(players_anim, frames, ball_x, ball_y, ball_z, target_nflId)
    return round(atd_pct, 2) if atd_pct is not None else None

def compute_game_level_atd(game_id):
    """
    Compute average ATD across all plays in a given game.
    Returns dict: {"avg_atd": float, "num_plays": int, "play_atd_list": [(playId, atd_pct), ...]}
    """
    if input_df.empty:
        return {"avg_atd": 0.0, "num_plays": 0, "play_atd_list": []}

    game_plays = sorted(input_df[input_df["gameId"] == game_id]["playId"].dropna().unique())
    play_results = []
    for pid in game_plays:
        atd = compute_play_level_atd(game_id, pid)
        if atd is not None:
            play_results.append((pid, atd))

    if not play_results:
        return {"avg_atd": 0.0, "num_plays": 0, "play_atd_list": []}

    avg = round(sum(v for _, v in play_results) / len(play_results), 2)
    return {"avg_atd": avg, "num_plays": len(play_results), "play_atd_list": play_results}

# ---------- Defender helpers ----------
def _select_targeted_receiver(full_df, frames):
    if full_df.empty: return None, None
    if "player_role" in full_df.columns:
        wr = full_df[full_df["player_role"] == "Targeted Receiver"]
        if not wr.empty:
            wr_id = wr["nflId"].iloc[0]
            wr_name = wr["displayName"].dropna().iloc[0] if "displayName" in wr.columns and wr["displayName"].notna().any() else str(wr_id)
            return wr_id, wr_name

    last_frame = max(frames)
    off_last = full_df[(full_df.get("teamName", "") == "Offense") & (full_df["frameId"] == last_frame)]
    if off_last.empty: off_last = full_df[full_df.get("teamName", "") == "Offense"]
    if off_last.empty: off_last = full_df

    if not off_last.empty:
        off_last = off_last.sort_values("x", ascending=False)
        wr_id = off_last["nflId"].iloc[0]
        wr_name = off_last["displayName"].dropna().iloc[0] if "displayName" in off_last.columns and off_last["displayName"].notna().any() else str(wr_id)
        return wr_id, wr_name
    return None, None

def nearest_defender_at_frame(x0, y0, frame_id, all_play_data):
    def_df = all_play_data[(all_play_data["teamName"] == "Defense") & (all_play_data["frameId"] == frame_id)][["displayName","x","y"]]
    if def_df.empty: return "No Defender", np.inf
    def_df = def_df.copy()
    def_df["dist"] = np.sqrt((def_df["x"] - x0)**2 + (def_df["y"] - y0)**2)
    if def_df["dist"].empty: return "No Defender", np.inf
    row = def_df.loc[def_df["dist"].idxmin()]
    return row["displayName"], float(row["dist"])

# ---------- ABSR helpers (kept but not shown by default) ----------
def compute_absr_for_play_single(play_df, start_x, start_y, land_x, land_y, frames, defender_prefix='def'):
    if play_df is None or play_df.empty or None in (start_x, start_y, land_x, land_y, frames): return None
    ball_x, ball_y, ball_z = get_ball_parabola(float(start_x), float(start_y), float(land_x), float(land_y), len(frames))
    if not (len(ball_x) == len(ball_y) == len(ball_z) == len(frames)): return None
    df = play_df.copy()
    if 'frameId' not in df.columns or 'x' not in df.columns or 'y' not in df.columns or 'teamName' not in df.columns: return None
    df['team_lc'] = df['teamName'].astype(str).str.lower().fillna('')
    frame_defs = {}
    for fr, grp in df.groupby('frameId'):
        defs = grp[grp['team_lc'].str.startswith(defender_prefix.lower()) | grp['team_lc'].str.contains(r'\bd', na=False)]
        frame_defs[int(fr)] = defs[['x', 'y']].astype(float).values if not defs.empty else np.empty((0, 2))
    min_dists = []; defender_z = 0.9
    for i, fr in enumerate(frames):
        fr_int = int(fr) if not isinstance(fr, int) else fr
        defs_xy = frame_defs.get(fr_int)
        if defs_xy is None or defs_xy.size == 0: continue
        bx, by, bz = float(ball_x[i]), float(ball_y[i]), float(ball_z[i])
        dx = defs_xy[:, 0] - bx; dy = defs_xy[:, 1] - by; dz = defender_z - bz
        dists = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dists.size > 0: min_dists.append(float(np.min(dists)))
    return float(np.mean(min_dists)) if min_dists else None

def calculate_game_absr(game_id, full_game_df):
    game_data = full_game_df[full_game_df["gameId"] == game_id].copy()
    if game_data.empty: return None
    absr_scores = []
    plays_with_target = game_data[game_data["player_role"].fillna('').str.contains("Targeted", na=False)][['playId']].drop_duplicates()
    if plays_with_target.empty: return None
    for play_id in plays_with_target['playId'].tolist():
        play_df = game_data[game_data['playId'] == play_id].copy()
        frames = sorted(play_df["frameId"].unique())
        if len(frames) < 2: continue
        qb_track = play_df[play_df["player_role"] == "Passer"].sort_values("frameId")
        start_x, start_y = (float(qb_track.iloc[-1]["x"]), float(qb_track.iloc[-1]["y"])) if not qb_track.empty else (play_df[play_df["frameId"] == frames[0]]["x"].mean(), play_df[play_df["frameId"] == frames[0]]["y"].mean())
        land_x = play_df["ball_land_x"].dropna().iloc[0] if play_df["ball_land_x"].notna().any() else None
        land_y = play_df["ball_land_y"].dropna().iloc[0] if play_df["ball_land_y"].notna().any() else None
        if None in (land_x, land_y, start_x, start_y): continue
        absr_play_val = compute_absr_for_play_single(play_df, start_x, start_y, land_x, land_y, frames)
        if absr_play_val is not None: absr_scores.append(absr_play_val)
    return float(np.mean(absr_scores)) if absr_scores else None

def absr_broadcast_message(absr_value):
    if absr_value is None or np.isnan(absr_value): return {"display": "N/A", "interpretation": "ABSR not available", "broadcast": "ABSR not computed for this game. Data missing."}
    disp = f"{absr_value:.2f} yds"
    if absr_value >= 3.0: interp = "Safe and precise (High Safety Margin)"; broadcast = f"This QB has an ABSR of {absr_value:.2f} yards â€” exhibiting safe and precise passing habits."
    elif absr_value >= 1.5: interp = "Moderate safety â€” requires caution"; broadcast = f"This QB has an ABSR of {absr_value:.2f} yards â€” moderate safety; keep an eye on tight windows."
    else: interp = "Risky and interception-prone (Low Safety Margin)"; broadcast = f"This QB has an ABSR of only {absr_value:.2f} yards â€” heâ€™s throwing into danger all night."
    return {"display": disp, "interpretation": interp, "broadcast": broadcast}

# ---------- Full KPI Computation (trimmed) ----------
def compute_kpis(full_df, players_all, frames, ball_x, ball_y, ball_z):
    """
    Only compute play-level defender proximity and minimal play info.
    Removed: DCSI, BFSG, SQI, CDS, CWD, PE, GSS.
    """
    kpis = {
        "best_defender": {"name": "N/A", "dist": 0.0, "nflId": None},
        "worst_defender": {"name": "N/A", "dist": 0.0, "nflId": None},
        "atd": {"pct": 0.0},
        "target_wr_name": None,
        "absr_data": None
    }

    if full_df.empty or players_all.empty or len(frames) == 0:
        return kpis

    wr_id, wr_name = _select_targeted_receiver(full_df, frames)
    if wr_id is None:
        return kpis
    kpis["target_wr_name"] = wr_name

    wr_track = full_df[full_df["nflId"] == wr_id].copy().sort_values("frameId")
    if wr_track[["x", "y"]].isna().all(axis=None):
        return kpis

    # Defenders proximity: best and worst (avg proximity across frames where both present)
    defenders = full_df[full_df.get("teamName", "") == "Defense"].copy().sort_values("frameId")
    if defenders.empty:
        return kpis

    # Best defender: smallest average distance to WR
    best_avg_dist = None; best_def_id = None; best_def_name = None
    for did, dtrack in defenders.groupby("nflId"):
        merged = wr_track[["frameId", "x", "y"]].merge(dtrack[["frameId", "x", "y"]], on="frameId", suffixes=("_wr", "_def"))
        if merged.empty: continue
        d = np.hypot(merged["x_wr"] - merged["x_def"], merged["y_wr"] - merged["y_def"])
        avg_d = float(d.mean())
        if (best_avg_dist is None) or (avg_d < best_avg_dist):
            best_avg_dist = avg_d
            best_def_id = did
            best_def_name = dtrack["displayName"].dropna().iloc[0] if "displayName" in dtrack.columns and dtrack["displayName"].notna().any() else str(did)

    # Worst defender (largest average distance)
    worst_avg_dist = None; worst_def_id = None; worst_def_name = None
    for did, dtrack in defenders.groupby("nflId"):
        merged = wr_track[["frameId", "x", "y"]].merge(dtrack[["frameId", "x", "y"]], on="frameId", suffixes=("_wr", "_def"))
        if merged.empty: continue
        d = np.hypot(merged["x_wr"] - merged["x_def"], merged["y_wr"] - merged["y_def"])
        avg_d = float(d.mean())
        if (worst_avg_dist is None) or (avg_d > worst_avg_dist):
            worst_avg_dist = avg_d
            worst_def_id = did
            worst_def_name = dtrack["displayName"].dropna().iloc[0] if "displayName" in dtrack.columns and dtrack["displayName"].notna().any() else str(did)

    if best_def_id is not None:
        kpis["best_defender"] = {"name": best_def_name, "dist": round(best_avg_dist, 2), "nflId": best_def_id}
    if worst_def_id is not None:
        kpis["worst_defender"] = {"name": worst_def_name, "dist": round(worst_avg_dist, 2), "nflId": worst_def_id}

    return kpis

def create_3d_broadcast_animation(game_id, play_id, atd_adv_margin=0.5, atd_catch_radius=1.5):
    global input_df, output_df, supp_df

    empty_kpi = {"best_defender": {"name": "N/A", "dist": 0.0, "nflId": None},
                 "worst_defender": {"name": "N/A", "dist": 0.0, "nflId": None},
                 "atd": {"pct": 0.0}, "target_wr_name": None, "absr_data": None}

    if input_df.empty and output_df.empty:
        return go.Figure(), empty_kpi, []

    inp_play = input_df[(input_df["gameId"] == game_id) & (input_df["playId"] == play_id)].copy()
    out_play = output_df[(output_df["gameId"] == game_id) & (output_df["playId"] == play_id)].copy()
    if inp_play.empty and out_play.empty: return go.Figure(), empty_kpi, []

    # Ball landing fallback
    land_x, land_y = None, None
    if "ball_land_x" in inp_play.columns: land_x = inp_play["ball_land_x"].max()
    if "ball_land_y" in inp_play.columns: land_y = inp_play["ball_land_y"].max()
    if pd.isna(land_x): land_x = inp_play["x"].mean()
    if pd.isna(land_y): land_y = inp_play["y"].mean()

    # Build full_tracks (input + output concatenation)
    full_tracks = []
    all_pids = set(inp_play.get("nflId", []).unique()) | set(out_play.get("nflId", []).unique())
    info = {}
    if not inp_play.empty:
        info = {pid: inp_play[inp_play["nflId"]==pid].iloc[0][["displayName","teamName"]].to_dict() for pid in inp_play["nflId"].unique()}

    for pid in all_pids:
        p1 = inp_play[inp_play["nflId"] == pid]
        p2 = out_play[out_play["nflId"] == pid]
        if not p1.empty and not p2.empty:
            p2 = p2.copy()
            p2["frameId"] += p1["frameId"].max()
        pf = pd.concat([p1, p2], ignore_index=True)
        if not pf.empty:
            if "displayName" not in pf.columns or pf["displayName"].isna().all():
                pf["displayName"] = info.get(pid, {}).get("displayName", str(pid))
            if "teamName" not in pf.columns or pf["teamName"].isna().all():
                pf["teamName"] = info.get(pid, {}).get("teamName", "Unknown")
            full_tracks.append(pf)

    if not full_tracks: return go.Figure(), empty_kpi, []

    full_df = pd.concat(full_tracks, ignore_index=True)
    players_all = full_df.dropna(subset=["x", "y", "frameId"]).sort_values("frameId")
    frames = sorted(players_all["frameId"].unique())

    # scaffold players_anim
    scaffold = pd.DataFrame(list(itertools.product(players_all['nflId'].unique(), frames)), columns=['nflId', 'frameId'])
    info_df = players_all[['nflId', 'displayName', 'teamName']].drop_duplicates('nflId')
    scaffold = scaffold.merge(info_df, on='nflId', how='left')
    players_anim = scaffold.merge(players_all[['nflId', 'frameId', 'x', 'y']], on=['nflId', 'frameId'], how='left')
    players_anim['x'] = players_anim.groupby('nflId')['x'].ffill().bfill()
    players_anim['y'] = players_anim.groupby('nflId')['y'].ffill().bfill()
    players_anim = players_anim.dropna(subset=['x', 'y'])

    # Ball path start
    start_x, start_y = None, None
    qb_id = None
    if "player_role" in inp_play.columns:
        qb = inp_play[inp_play["player_role"] == "Passer"]
        if not qb.empty:
            start_x, start_y = qb.iloc[-1][["x","y"]]
            qb_id = qb.iloc[-1]["nflId"]
    if start_x is None:
        start_x, start_y = players_anim[players_anim["frameId"]==frames[0]][["x","y"]].mean()

    # force QB position fix
    if qb_id is not None and start_x is not None and start_y is not None:
        players_anim.loc[players_anim["nflId"] == qb_id, "x"] = start_x
        players_anim.loc[players_anim["nflId"] == qb_id, "y"] = start_y

    t = np.linspace(0, 1, len(frames))
    ball_x = start_x + (land_x - start_x) * t
    ball_y = start_y + (land_y - start_y) * t
    ball_z = 12 * 4 * t * (1 - t)

    # identify target for ATD
    target_nflId = None
    if "player_role" in full_df.columns:
        tgt = full_df[full_df["player_role"]=="Targeted Receiver"]
        if not tgt.empty: target_nflId = tgt["nflId"].iloc[0]
    if target_nflId is None:
        target_nflId = players_all["nflId"].iloc[0]

    # KPI computations (trimmed)
    kpis = compute_kpis(full_df, players_all, frames, ball_x, ball_y, ball_z)

    # Game-level metrics (ABSR kept optional)
    game_absr_val = calculate_game_absr(game_id, input_df)
    kpis["absr_data"] = absr_broadcast_message(game_absr_val)

    # ATD calculation
    atd_flags, atd_pct, _ = compute_atd_frames(players_anim, frames, ball_x, ball_y, ball_z, target_nflId, adv_margin=atd_adv_margin, catch_radius=atd_catch_radius)
    kpis["atd"] = {"pct": round(atd_pct, 1), "flags": atd_flags, "target_nflId": target_nflId}
    kpis["target_wr_name"] = players_all[players_all["nflId"]==target_nflId]["displayName"].iloc[0] if target_nflId else "Unknown"

    # Play-level player ranks (computed as before)
    player_stats = []
    for pid, grp in full_df.groupby("nflId"):
        coords = grp[["x","y"]].values
        if len(coords) > 1:
            d = np.sum(np.sqrt(np.sum((coords[1:] - coords[:-1])**2, axis=1)))
        else: d = 0.0
        s = float(grp["s"].max()) if "s" in grp.columns else 0.0
        player_stats.append({"name": grp["displayName"].iloc[0], "score": s*1.5 + d/10.0, "max_speed": s, "dist": d})
    play_level_ranks = sorted(player_stats, key=lambda x: x["score"], reverse=True)

    # ---------------------------
    # NEW: Prefer GAME-level ranks if available
    # ---------------------------
    try:
        game_level_ranks = compute_player_ranks_for_game(game_id)
        if game_level_ranks:
            player_ranks = game_level_ranks
        else:
            player_ranks = play_level_ranks
    except Exception:
        player_ranks = play_level_ranks
    # ---------------------------

    best_def_id = kpis["best_defender"]["nflId"]

    # Dynamic field bounds
    p_min_x = players_all["x"].min(); p_max_x = players_all["x"].max()
    p_min_y = players_all["y"].min(); p_max_y = players_all["y"].max()
    x_buffer = 10.0; y_buffer = 8.0
    min_x = p_min_x - x_buffer; max_x = p_max_x + x_buffer
    min_y = p_min_y - y_buffer; max_y = p_max_y + y_buffer
    if min_x < 30: min_x = max(-2, min_x)
    if max_x > 90: max_x = min(122, max_x)
    if max_x - min_x < 40.0:
        center_x = (min_x + max_x) / 2.0
        min_x = center_x - 20.0; max_x = center_x + 20.0
        min_x = max(-5.0, min_x); max_x = min(125.0, max_x)
    min_y = max(-5.0, min_y); max_y = min(58.3, max_y)

    fig = go.Figure()

    # Field mesh & lines
    fig.add_trace(go.Mesh3d(x=[min_x, min_x, max_x, max_x], y=[min_y, max_y, max_y, min_y], z=[0, 0, 0, 0], color="#0f4c3a", opacity=1.0, showlegend=False, lighting=dict(ambient=0.8, diffuse=0.9)))
    start_yard = int(np.floor(min_x / 10.0) * 10)
    end_yard = int(np.ceil(max_x / 10.0) * 10)
    for x_line in range(start_yard, end_yard + 1, 10):
        fig.add_trace(go.Scatter3d(x=[x_line, x_line], y=[min_y, max_y], z=[0.05, 0.05], mode="lines", line=dict(color="rgba(255,255,255,0.7)", width=2), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min_x, max_x, max_x, min_x, min_x], y=[min_y, min_y, max_y, max_y, min_y], z=[0.05]*5, mode="lines", line=dict(color="white", width=4), showlegend=False))

    if min_x <= 0: add_goalposts(fig, 0)
    if max_x >= 120: add_goalposts(fig, 120)
    add_qb_stick_figure(fig, start_x, start_y, land_x, land_y)

    # legend markers
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color="#facc15", line=dict(color="black", width=1)), name="Offense"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color="#3b82f6", line=dict(color="black", width=1)), name="Defense"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=12, color="#10b981", line=dict(color="black", width=1)), name="ATD Advantage"))

    # static trails
    for name in players_all["displayName"].dropna().unique():
        p = players_all[players_all["displayName"] == name].sort_values("frameId")
        side = p["teamName"].iloc[0] if p["teamName"].notna().any() else None
        color = "#facc15" if side == "Offense" else "#3b82f6"
        fig.add_trace(go.Scatter3d(x=p["x"], y=p["y"], z=[0.1] * len(p), mode="lines", line=dict(color=color, width=5, dash="dot"), showlegend=False, opacity=1.0))

    # ball path
    fig.add_trace(go.Scatter3d(x=ball_x, y=ball_y, z=ball_z, mode="lines", line=dict(color="#f97316", width=6), showlegend=False, opacity=1.0))

    # frames
    frames_list = []
    short_names_map = {name: (name.split()[0] + ("" if len(name.split()) == 1 else " " + name.split()[-1])) for name in players_all["displayName"].dropna().unique()}

    init_df = players_anim[players_anim["frameId"] == frames[0]].copy()
    atd_flags_local = atd_flags  # from compute_atd_frames above
    init_colors = []
    for _, row in init_df.iterrows():
        if row["nflId"] == target_nflId:
            dom = atd_flags_local[0] if len(atd_flags_local) > 0 else False
            init_colors.append("#10b981" if dom else "#fbbf24")
        else:
            init_colors.append("#facc15" if row["teamName"] == "Offense" else "#3b82f6")

    init_stagger_z = [2.5 + (k % 5) * 1.5 for k in range(len(init_df))]
    init_stem_x, init_stem_y, init_stem_z = [], [], []
    for k, (_, row) in enumerate(init_df.iterrows()):
        init_stem_x.extend([row['x'], row['x'], None])
        init_stem_y.extend([row['y'], row['y'], None])
        init_stem_z.extend([0.5, init_stagger_z[k], None])

    fig.add_trace(go.Scatter3d(x=init_stem_x, y=init_stem_y, z=init_stem_z, mode="lines", line=dict(color="rgba(255, 255, 255, 0.3)", width=1), hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter3d(x=init_df["x"], y=init_df["y"], z=[0.5] * len(init_df), mode="markers", marker=dict(size=7, color=init_colors, line=dict(color="black", width=1)), hovertext=init_df["displayName"], hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter3d(x=init_df["x"], y=init_df["y"], z=init_stagger_z, mode="text", text=[f"<b>{short_names_map.get(n,n)}</b>" for n in init_df["displayName"]], textposition="top center", textfont=dict(size=11, color="white", family="Consolas, 'Courier New', monospace"), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[ball_x[0]], y=[ball_y[0]], z=[ball_z[0]], mode="markers", marker=dict(symbol="circle", size=10, color="#f97316", line=dict(color="black", width=1)), showlegend=False))

    # adjust dynamic trace start index for 4 dynamic traces (stems, markers, labels, ball)
    dynamic_trace_start_idx = len(fig.data) - 4

    for i, fr in enumerate(frames):
        fr_df = players_anim[players_anim["frameId"] == fr].copy()
        if fr_df.empty: continue

        f_colors = []
        for _, row in fr_df.iterrows():
            if row["nflId"] == target_nflId:
                dom = atd_flags_local[i] if i < len(atd_flags_local) else False
                f_colors.append("#10b981" if dom else "#fbbf24")
            else:
                f_colors.append("#facc15" if row["teamName"] == "Offense" else "#3b82f6")

        stagger_z = [2.5 + (k % 5) * 1.5 for k in range(len(fr_df))]

        stem_x, stem_y, stem_z = [], [], []
        for k, (_, row) in enumerate(fr_df.iterrows()):
            stem_x.extend([row['x'], row['x'], None])
            stem_y.extend([row['y'], row['y'], None])
            stem_z.extend([0.5, stagger_z[k], None])

        trace_stems = go.Scatter3d(x=stem_x, y=stem_y, z=stem_z, mode="lines", line=dict(color="rgba(255, 255, 255, 0.3)", width=1), hoverinfo="none", showlegend=False)
        trace_markers = go.Scatter3d(x=fr_df["x"], y=fr_df["y"], z=[0.5] * len(fr_df), mode="markers", marker=dict(size=7, color=f_colors, line=dict(color="black", width=1)), hovertext=fr_df["displayName"], hoverinfo="text", showlegend=False)
        trace_labels = go.Scatter3d(x=fr_df["x"], y=fr_df["y"], z=stagger_z, mode="text", text=[f"<b>{short_names_map.get(n,n)}</b>" for n in fr_df["displayName"]], textposition="top center", textfont=dict(size=11, color="white", family="Consolas, 'Courier New', monospace"), showlegend=False)
        trace_ball = go.Scatter3d(x=[float(ball_x[i])], y=[float(ball_y[i])], z=[float(ball_z[i])], mode="markers", marker=dict(symbol="circle", size=10, color="#f97316", line=dict(color="black", width=1)), showlegend=False)

        current_indices = [dynamic_trace_start_idx, dynamic_trace_start_idx+1, dynamic_trace_start_idx+2, dynamic_trace_start_idx+3]
        frames_list.append(go.Frame(data=[trace_stems, trace_markers, trace_labels, trace_ball], name=f"f{fr}", traces=current_indices))

    fig.frames = frames_list

    fixed_camera_view = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=-0.1), eye=dict(x=-1.5, y=-1.5, z=1.2))

    fig.update_layout(
        autosize=True,
        title=dict(text=f"<b>Air Time Dominance</b>", font=dict(size=24, color="#f8fafc"), x=0.5, xanchor='center', y=0.95, yanchor='top'),
        scene_camera=fixed_camera_view,
        scene=dict(
            xaxis=dict(range=[min_x, max_x], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[min_y, max_y], showgrid=False, zeroline=False, visible=False),
            zaxis=dict(range=[0, 15], showgrid=False, showticklabels=False, visible=False),
            aspectmode="manual",
            aspectratio=dict(x=3.0, y=1.5, z=0.3),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
        font=dict(family="Roboto, sans-serif", size=14, color="#e5e7eb"),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="#334155", borderwidth=1, font=dict(size=11, color="#f1f5f9"), itemsizing='constant'),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.98, xanchor='left', yanchor='top', bgcolor="rgba(30, 41, 59, 0.9)", bordercolor="#334155",
                          font=dict(color="#ffffff", size=11),
                          buttons=[dict(label="â–¶ PLAY", method="animate", args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)])])],
    )
    return fig, kpis, player_ranks


# ---------- HTML Builders (trimmed to removed metrics) ----------
def build_explanation_html(game_id, play_id, kpis, game_atd=None):
    """
    Minimal explanation: show only Play ATD and Game-Level ATD.
    """
    atd_pct = kpis.get("atd", {}).get("pct", "N/A")
    try:
        atd_val = float(atd_pct)
        if atd_val >= 60:
            atd_interp = "Receiver dominated the air â€” very strong positional control."
        elif atd_val >= 30:
            atd_interp = "Moderate air control â€” receiver had opportunities."
        else:
            atd_interp = "Low air dominance â€” defender contested or disrupted the catch area."
    except:
        atd_interp = "ATD unavailable"

    game_atd_block = ""
    if game_atd is not None:
        try:
            g_val = float(game_atd)
            gclass = "Strong" if g_val >= 60 else ("Moderate" if g_val >= 30 else "Low")
            game_atd_block = f"""
            <li class="bullet-item">
                <strong style="color:#334155">Game-Level ATD: {g_val}%</strong>
                <ul class="sub-bullet-list">
                    <li class="sub-bullet-item"><i>Definition:</i> Average ATD across all plays in this game.</li>
                    <li class="sub-bullet-item"><i>Insight:</i> {gclass} overall air dominance for the offense.</li>
                </ul>
            </li>
            """
        except:
            game_atd_block = '<li class="bullet-item">Game-Level ATD Data Unavailable.</li>'

    html = f"""
    <div class="explain-container">
      <div class="explain-title">COACH'S GAME NOTE <span style="font-weight:400; color:#64748b; font-size:0.8em; margin-left:10px;">Play {play_id} Breakdown</span></div>

      <div class="section-box">
        <div class="section-header">ğŸ”� How to Read this Play</div>
        <p style="font-size:0.9em; margin:0; color:#475569;">
            The <b style="color:#10b981">GREEN</b> marker shows Air Time Dominance (ATD) for the targeted receiver.
        </p>
      </div>

      <div class="section-box">
        <div class="section-header">ğŸ“Š KPI METRICS ANALYSIS</div>
        <ul class="bullet-list">
            <li class="bullet-item">
                <strong style="color:#334155">ATD â€” Air Time Dominance (Play): {atd_pct}%</strong>
                <ul class="sub-bullet-list">
                    <li class="sub-bullet-item"><i>Definition:</i> % of ball-flight frames where the targeted receiver had positional advantage.</li>
                    <li class="sub-bullet-item"><i>Interpretation:</i> {atd_interp}</li>
                </ul>
            </li>

            {game_atd_block}
        </ul>
      </div>
    </div>
    """
    return html


def build_player_rank_html(player_ranks):
    if not player_ranks:
        return "<div>No player tracking data available.</div>"

    motm = player_ranks[0]
    top_5 = player_ranks[:5]
    worst_5 = player_ranks[-5:] if len(player_ranks) >= 5 else []
    worst_5 = sorted(worst_5, key=lambda x: x['score'])

    html = f"""{KPI_STYLE}
    <div class="rank-panel">
        <div class="panel-header"><span>âš¡ PLAYER IMPACT</span></div>
        <div class="panel-sub">Speed & Distance Perf. Index</div>

        <div class="mom-card"><span class="mom-badge">ğŸ�† MAN OF THE MATCH</span><div class="mom-name">{motm['name']}</div><div class="mom-stat">Perf Score: {motm['score']:.1f}</div>
        <div style="font-size:0.75em; margin-top:8px; opacity:0.8;">Max Speed: {motm['max_speed']:.1f} â€¢ Dist: {motm['dist']:.1f}</div></div>

        <div class="rank-section-header"><span>ğŸ”¥ Top 5 Performers</span><span style="font-size:0.8em; opacity:0.7">SCORE</span></div>
        """

    for i, p in enumerate(top_5):
        html += f"""
        <div class="player-row row-best">
            <span class="p-rank">#{i+1}</span>
            <span class="p-name">{p['name']}</span>
            <span class="p-score">{p['score']:.1f}</span>
        </div>
        """

    html += """
        <div class="rank-section-header" style="margin-top:25px;">
            <span>â�„ï¸� Bottom 5 Performers</span>
            <span style="font-size:0.8em; opacity:0.7">SCORE</span>
        </div>
    """

    for i, p in enumerate(worst_5):
        html += f"""
        <div class="player-row row-worst">
            <span class="p-rank">#{i+1}</span>
            <span class="p-name">{p['name']}</span>
            <span class="p-score">{p['score']:.1f}</span>
        </div>
        """

    html += "</div>"
    return html

# ---------- WIDGETS & UI ----------
widget_style = """
<style>
    .widget-dropdown > select { background-color: #1e2538 !important; color: #f1f5f9 !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important; padding: 8px 12px !important; font-family: 'Roboto', sans-serif !important; font-size: 1em !important; outline: none !important; box-shadow: none !important; max-width: 100% !important; min-width: 150px !important; }
    .widget-dropdown > select:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important; }
    .gen-button { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; color: #ffffff !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important; font-family: 'Roboto', sans-serif !important; font-weight: 700 !important; font-size: 1em !important; text-transform: uppercase !important; letter-spacing: 1px !important; transition: all 0.2s ease-in-out !important; cursor: pointer !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important; }
    .gen-button:hover { background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; box-shadow: 0 6px 12px -2px rgba(59, 130, 246, 0.5) !important; transform: translateY(-1px) !important; }
</style>
"""

w_week = widgets.Dropdown(description="Week:", style={'description_width': 'initial'}, layout=widgets.Layout(width='200px', height='40px' ,margin='0 10px 10px 0'))
w_game = widgets.Dropdown(description="Game ID:", style={'description_width': 'initial'}, layout=widgets.Layout(width='300px', height='40px' , margin='0 10px 10px 0'))
w_play = widgets.Dropdown(description="Play ID:", style={'description_width': 'initial'}, layout=widgets.Layout(width='200px', height='40px' , margin='0 0 10px 0'))
w_week.add_class("widget-dropdown"); w_game.add_class("widget-dropdown"); w_play.add_class("widget-dropdown")
btn_gen = widgets.Button(description="GENERATE VISUALIZATION", button_style="", layout=widgets.Layout(width='100%', height='45px', margin='15px 0 0 0'))
btn_gen.add_class("gen-button")

out_viz = widgets.Output(layout=widgets.Layout(width='100%', height='650px'))
out_explain = widgets.Output(layout=widgets.Layout(width='100%'))
out_kpi = widgets.Output(layout=widgets.Layout(width='22.5%'))
out_players = widgets.Output(layout=widgets.Layout(width='22.5%'))


# ---------- on_click_gen (updated to show Game ATD + Play ATD) ----------
def on_click_gen(_):
    out_viz.clear_output(); out_kpi.clear_output(); out_players.clear_output(); out_explain.clear_output()
    gid = w_game.value; pid = w_play.value
    if gid is None or pid is None: return

    try:
        fig, kpis, player_ranks = create_3d_broadcast_animation(gid, pid)
        fig.write_html(
        "/kaggle/working/code3_atd/atd.html"
            )

        with out_viz: fig.show()
    except Exception as e:
        with out_viz: print(f"Error generating animation or calculating KPIs: {e}")
        kpis = {}
        player_ranks = []
        return

    # compute Game ATD
    game_atd_data = compute_game_level_atd(gid)
    game_atd_val = game_atd_data.get("avg_atd", 0.0)
    game_atd_count = game_atd_data.get("num_plays", 0)

    # KPI Card (show Game ATD + Play ATD)
        # KPI Card (show Game ATD + Play ATD) - minimal (only ATD & Game ATD)
    with out_kpi:
        atd_obj = kpis.get("atd", {"pct": 0.0})
        pct = atd_obj.get("pct", 0.0)
        atd_class = "card-green" if pct >= 60 else "card-gold" if pct >= 30 else "card-red"
        atd_card_html = f"""<div class="kpi-card {atd_class}"><div class="kpi-title">Air Time Dominance (Play)</div><div class="kpi-value">{pct}%</div><div class="kpi-desc">Receiver Airspace Control</div></div>"""

        g_atd_class = "card-green" if game_atd_val >= 60 else "card-gold" if game_atd_val >= 30 else "card-red"
        game_atd_card_html = f"""<div class="kpi-card {g_atd_class}"><div class="kpi-title">Game ATD (Avg)</div><div class="kpi-value">{game_atd_val}%</div><div class="kpi-desc">Avg ATD across {game_atd_count} plays</div></div>"""

        html_kpi = f"""
        {KPI_STYLE}
        <div class="kpi-panel">
            <div class="panel-header"><span>ğŸ“¡ BROADCAST FEED</span></div>
            <div class="panel-sub">Play Analysis â€¢ Game {gid}</div>
            <div style="display:flex; flex-direction:column; gap:8px;">
                {game_atd_card_html}
                {atd_card_html}
            </div>
        </div>
        """
        display(HTML(html_kpi))


    # Player ranks (top and bottom 5)
    with out_players:
        ranks = compute_player_ranks_for_game(gid)
        if not ranks: ranks = player_ranks
        html_kpi_right = build_player_rank_html(ranks)
        display(HTML(html_kpi_right))

    # Explanation (pass game_atd)
    with out_explain:
        explanation_html = build_explanation_html(gid, pid, kpis, game_atd=game_atd_val)
        display(HTML(explanation_html))

btn_gen.on_click(on_click_gen)

# ---------- helper for computing game-level ranks (kept) ----------
_game_ranks_cache = {}
def compute_player_ranks_for_game(game_id, use_cache=True):
    global _game_ranks_cache
    if use_cache and game_id in _game_ranks_cache: return _game_ranks_cache[game_id]
    inp_game = input_df[input_df["gameId"] == game_id]
    if inp_game.empty: return []
    stats = []
    for pid, grp in inp_game.groupby("nflId"):
        max_s = grp["s"].max() if "s" in grp.columns else 0
        coords = grp[["x", "y"]].values
        dist = np.sum(np.sqrt(np.sum((coords[1:] - coords[:-1])**2, axis=1))) if len(coords) > 1 else 0
        score = (max_s * 1.5) + (dist / 10.0)
        name = grp["displayName"].iloc[0] if "displayName" in grp.columns else str(pid)
        stats.append({"name": name, "score": score, "max_speed": max_s, "dist": dist})
    ranks = sorted(stats, key=lambda x: x["score"], reverse=True)
    if use_cache: _game_ranks_cache[game_id] = ranks
    return ranks

# ---------- UI wiring ----------
def update_selectors():
    if _df.empty: return
    try:
        weeks = sorted(_df["week"].dropna().unique().astype(int))
    except:
        weeks = sorted(_df["week"].dropna().unique())
    w_week.options = weeks
    if weeks: w_week.value = weeks[0]

def update_games(*_):
    if _df.empty: return
    gms = sorted(_df[_df["week"] == w_week.value]["gameId"].dropna().unique())
    w_game.options = gms
    if gms: w_game.value = gms[0]

def update_plays(*_):
    if _df.empty or w_game.value is None: return
    pls = sorted(_df[(_df["week"] == w_week.value) & (_df["gameId"] == w_game.value)]["playId"].dropna().unique())
    w_play.options = pls
    if pls: w_play.value = pls[0]

w_week.observe(update_games, names="value")
w_game.observe(update_plays, names="value")
update_selectors(); update_games(); update_plays()

header_html = """
<div style="background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
    <h2 style="color: #f8fafc; margin: 0; font-family: 'Roboto', sans-serif; font-weight: 900; letter-spacing: 1px;">
        ğŸ�ˆ NFL NEXT GEN STATS <span style="color:#3b82f6">PRO</span>
    </h2>
    <p style="color: #94a3b8; margin: 5px 0 0 0; font-family: 'Roboto', sans-serif;">
        3D Broadcast Reconstruction â€¢ Air Time Dominance â€¢ Player Impact
    </p>
</div>
"""

ui = widgets.VBox([
    widgets.HTML(widget_style + header_html),
    widgets.VBox([widgets.HBox([w_week, w_game, w_play]), btn_gen], layout=widgets.Layout(padding='20px', background_color='#151b2b', border_radius='16px', border='1px solid rgba(255,255,255,0.1)', margin='0 0 25px 0')),
    widgets.HBox([widgets.VBox([out_viz, out_explain], layout=widgets.Layout(width='55%')), out_kpi, out_players])
])
display(ui)



import os

print(os.listdir("/kaggle/input"))



print(os.listdir("/kaggle/input/nfl-big-data-bowl-2026-analytics"))
1


print(
    os.listdir(
        "/kaggle/input/nfl-big-data-bowl-2026-analytics/"
        "114239_nfl_competition_files_published_analytics_final"
    )
)



import pandas as pd

# Create a dummy submission file (Analytics track workaround)
submission = pd.DataFrame({
    "note": ["Analytics notebook output"],
    "author": ["Sowmya"],
    "project": ["NFL Big Data Bowl 2026 â€“ Analytics"]
})

submission.to_csv("/kaggle/working/submission.csv", index=False)

print("âœ… submission.csv created")



import os
print(os.listdir("/kaggle/working"))



submission = pd.DataFrame({
    "section": ["Visualization", "Metrics", "UI"],
    "content": [
        "3D Broadcast Reconstruction",
        "ATD, 3D Path Efficiency, GSS",
        "Interactive Dashboard"
    ]
})

submission.to_csv("/kaggle/working/submission.csv", index=False)





