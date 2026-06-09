import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Compute ROC curves
fpr_base, tpr_base, _ = roc_curve(y_te, p_base)
fpr_cli,  tpr_cli,  _ = roc_curve(y_te, p_both)

auc_base = roc_auc_score(y_te, p_base)
auc_cli  = roc_auc_score(y_te, p_both)

# Plot
plt.figure(figsize=(5.5, 4.5))

# Baseline — Blue
plt.plot(
    fpr_base, tpr_base,
    linewidth=2.5,
    color="#1f78b4",
    label=f"Baseline (pass length only) — AUC = {auc_base:.3f}"
)

# CLI Augmented — Red
plt.plot(
    fpr_cli, tpr_cli,
    linewidth=2.5,
    color="#b23a2f",
    label=f"Baseline + CLI — AUC = {auc_cli:.3f}"
)

# Diagonal (chance)
plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    linewidth=1.2,
    color="#9e9e9e",
    alpha=0.9
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC: Does CLI Add Predictive Signal?")
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(
    "/kaggle/working/roc_cli_vs_baseline.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()



# --------------------------------------------
# Consistent labels + colors (reuse everywhere)
# --------------------------------------------
TYPE_LABEL = {
    "completion": "Completion",
    "def_collapse_INC": "Incomplete — Late breakup",
    "off_control_INC": "Incomplete — Offense-owned miss",
    "never_viable_INC": "Incomplete — Never viable",
}

TYPE_COLOR = {
    "Completion": "#1b9e77",                 # green
    "Incomplete — Late breakup": "#d95f02",  # orange
    "Incomplete — Offense-owned miss": "#1f78b4",  # 
    "Incomplete — Never viable": "#b23a2f",  
}
plot_types = [
    "completion",
    "def_collapse_INC",
    "off_control_INC",
    "never_viable_INC",
]

plt.figure(figsize=(9.2, 5.4))

for t_key in plot_types:
    sub = agg_grid[agg_grid["type"] == t_key]
    if sub.empty:
        continue

    label = t_key
    color = TYPE_COLOR[TYPE_LABEL[t_key]]

    tau = sub["tau"].values
    m   = sub["mean"].values
    q25 = sub["q25"].values
    q75 = sub["q75"].values

    plt.plot(tau, m, lw=2.2, color=color, label=label)
    plt.fill_between(tau, q25, q75, color=color, alpha=0.18)

# reference line
plt.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

plt.xlabel("Normalized time in air (τ)\n0 = throw, 1 = arrival")
plt.ylabel("Catch Leverage Index  (CLI = P_off − P_def)")

plt.title(
    "Outcome-level CLI(t) trajectories\n"
    "Mean ± interquartile range (25–75%)",
    fontsize=12,
    weight="bold"
)

plt.legend(frameon=True, framealpha=0.9, fontsize=9)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("/kaggle/working/cli_agg.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
import os, glob
print("Saved files:", glob.glob("/kaggle/working/*.png")[:20])




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


# ----------------------------
# Field drawing helpers
# ----------------------------
def draw_field(ax):
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")

    # Field background
    ax.add_patch(Rectangle((0, 0), 120, 53.3, facecolor="#2e7d32", edgecolor="none"))
    # Endzones
    ax.add_patch(Rectangle((0, 0), 10, 53.3, facecolor="#1e4f8a", alpha=0.35, edgecolor="none"))
    ax.add_patch(Rectangle((110, 0), 10, 53.3, facecolor="#1e4f8a", alpha=0.35, edgecolor="none"))

    # Yard lines
    for x in range(10, 111, 5):
        lw = 1.6 if x % 10 == 0 else 0.8
        ax.plot([x, x], [0, 53.3], color="white", lw=lw, alpha=0.85)

    # Hash marks (light)
    for x in range(11, 110):
        ax.plot([x, x], [23.3, 23.8], color="white", lw=0.6, alpha=0.35)
        ax.plot([x, x], [29.5, 30.0], color="white", lw=0.6, alpha=0.35)


def draw_yard_numbers(ax, x_min=0, x_max=120, y_top=53.3, y_bot=0.0, step=10):
    # Faint broadcast-like numbers top+bottom: 10 20 30 40 50 40 30 ...
    for x in range(int(np.ceil(x_min / step) * step), int(x_max) + 1, step):
        if x in (0, 120):
            continue
        label = str(x if x <= 50 else 120 - x)
        ax.text(x, y_top - 2.0, label, ha="center", va="top",
                fontsize=9, color="white", alpha=0.55)
        ax.text(x, y_bot + 2.0, label, ha="center", va="bottom",
                fontsize=9, color="white", alpha=0.55)


def maybe_flip_play(df_play, play_direction_col="play_direction"):
    """
    Normalize view: offense always goes left->right.
    Returns (df_flipped, did_flip_bool).
    """
    did_flip = False
    if play_direction_col in df_play.columns:
        pdn = df_play[play_direction_col].dropna().astype(str)
        if len(pdn) and pdn.iloc[0].lower().startswith("left"):
            df_play = df_play.copy()
            df_play["x"] = 120 - df_play["x"]
            did_flip = True
    return df_play, did_flip


# ----------------------------
# Main GIF maker
# ----------------------------
def make_play_cli_gif(
    tracking_bia_kin,
    curves_all,
    game_id,
    play_id,
    out_gif,
    title="",
    max_frames=80,
    fps=12,
    speed_mult=0.70,          # 0.7x slower (user request)
    show_peak_badge=True
):
    # --- tracking frames
    dfp = tracking_bia_kin[
        (tracking_bia_kin.game_id == game_id) &
        (tracking_bia_kin.play_id == play_id)
    ].copy()

    if dfp.empty:
        raise ValueError(f"No tracking for game_id={game_id}, play_id={play_id}")

    # If your df has a 'kind' column, keep ball-in-air frames
    if "kind" in dfp.columns:
        kind = dfp["kind"].astype(str).str.lower()
        bia = dfp[kind.str.contains("ball", na=False)].copy()
        if not bia.empty:
            dfp = bia

    # Flip play if needed (for consistent viewing)
    dfp, did_flip = maybe_flip_play(dfp)

    # Landing point (from input merge)
    xL = None
    yL = None
    if "ball_land_x" in dfp.columns and dfp["ball_land_x"].notna().any():
        xL = float(dfp["ball_land_x"].dropna().iloc[0])
    if "ball_land_y" in dfp.columns and dfp["ball_land_y"].notna().any():
        yL = float(dfp["ball_land_y"].dropna().iloc[0])

    # If we flipped player x, also flip landing x
    if did_flip and xL is not None:
        xL = 120 - xL

    # Frame list (sample uniformly if too many)
    frames = np.sort(dfp["frame_id"].unique())
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = frames[idx]

    # --- CLI curve
    curve = curves_all[
        (curves_all.game_id == game_id) &
        (curves_all.play_id == play_id)
    ].copy()

    if curve.empty:
        raise ValueError(f"No CLI curve for game_id={game_id}, play_id={play_id}")

    curve = curve.sort_values("t_rel").reset_index(drop=True)

    # Peak time (for vertical line + badge)
    i_peak = int(curve["CLI"].values.argmax())
    t_peak = float(curve.loc[i_peak, "t_rel"])

    # map animation frame k to curve index
    def curve_idx_for_k(k):
        return int(np.clip(round(k * (len(curve) - 1) / max(1, (len(frames) - 1))), 0, len(curve) - 1))

    # ----------------------------
    # Styling (single source of truth)
    # ----------------------------
    COL_DEF = "#4E79A7"  # defense
    COL_TR  = "#E15759"  # targeted receiver
    COL_LAND = "black"   # landing marker

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Defenders',
               markerfacecolor=COL_DEF, markeredgecolor="white", markersize=9),
        Line2D([0], [0], marker='o', color='w', label='Targeted Receiver',
               markerfacecolor=COL_TR, markeredgecolor="white", markersize=9),
        Line2D([0], [0], marker='x', color=COL_LAND, label='Ball landing',
               markersize=10, linewidth=0, markeredgewidth=2.5),
    ]

    # --- build figure (top: field, bottom: CLI)
    fig = plt.figure(figsize=(7.6, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.3], hspace=0.15)

    ax_field = fig.add_subplot(gs[0])
    ax_cli   = fig.add_subplot(gs[1])

    draw_field(ax_field)
    draw_yard_numbers(ax_field, x_min=0, x_max=120, y_top=53.3, y_bot=0.0, step=10)

    ax_field.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        framealpha=0.85,
        fontsize=8,
        fancybox=True
    )

    # Title + frame label
    title_text = ax_field.text(
        0.5, 1.01, title,
        transform=ax_field.transAxes,
        ha="center", va="bottom",
        fontsize=12.5, weight="bold"
    )
    frame_text = ax_field.text(
        0.02, 0.98, "",
        transform=ax_field.transAxes,
        ha="left", va="top",
        fontsize=10.5,
        bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", boxstyle="round,pad=0.3")
    )

    # Field scatters (created once; updated each frame)
    scat_off = ax_field.scatter([], [], s=80, c=COL_TR, edgecolors="white", linewidths=0.8, zorder=3)
    scat_def = ax_field.scatter([], [], s=80, c=COL_DEF, edgecolors="white", linewidths=0.8, zorder=3)
    scat_tr  = ax_field.scatter([], [], s=140, c=COL_TR, edgecolors="white", linewidths=1.2, zorder=5)

    if xL is not None and yL is not None:
        ax_field.scatter([xL], [yL], s=170, marker="x", c=COL_LAND, linewidths=3, zorder=6)

    # CLI axis
    ax_cli.set_xlim(float(curve["t_rel"].min()), float(curve["t_rel"].max()))
    ax_cli.set_ylim(-1.05, 1.05)
    ax_cli.axhline(0, lw=1.0, alpha=0.4)
    ax_cli.set_xlabel("Time since throw (t_rel)")
    ax_cli.set_ylabel("CLI(t)")
    ax_cli.plot(curve["t_rel"], curve["CLI"], lw=2.2)

    dot_cli, = ax_cli.plot([curve["t_rel"].iloc[0]], [curve["CLI"].iloc[0]],
                           marker="o", markersize=7,color=COL_LAND)

    # vertical peak line
    ax_cli.axvline(
        t_peak,
        linestyle="--",
        linewidth=1.2,
        color="gray",
        alpha=0.45,
        zorder=0
        )

    # text centered on the vertical line, 30% from the top
    ax_cli.text(
        t_peak,
        0.70,  # <-- 70% up the y-axis (i.e., slightly below top)
        "Leverage peaks here",
        transform=ax_cli.get_xaxis_transform(),  # x=data, y=axes fraction
        ha="center",
        va="center",
        fontsize=8.6,
        color="black",
        bbox=dict(
        facecolor="white",
        alpha=0.85,
        edgecolor="none",
        boxstyle="round,pad=0.25"
        )
      )

    def get_roles(df_frame):
        side = df_frame["player_side"].astype(str).str.lower() if "player_side" in df_frame.columns else pd.Series([""] * len(df_frame))
        is_off = side.str.contains("off", na=False)
        is_def = side.str.contains("def", na=False)

        role = df_frame["player_role"].astype(str).str.lower() if "player_role" in df_frame.columns else pd.Series([""] * len(df_frame))
        is_tr = role.str.contains("target", na=False)

        return is_off, is_def, is_tr

    def update(k):
        fr = frames[k]
        df_f = dfp[dfp["frame_id"] == fr].copy()

        is_off, is_def, is_tr = get_roles(df_f)

        off_xy = df_f.loc[is_off, ["x", "y"]].to_numpy()
        def_xy = df_f.loc[is_def, ["x", "y"]].to_numpy()
        tr_xy  = df_f.loc[is_tr,  ["x", "y"]].to_numpy()

        scat_off.set_offsets(off_xy if len(off_xy) else np.zeros((0, 2)))
        scat_def.set_offsets(def_xy if len(def_xy) else np.zeros((0, 2)))
        scat_tr.set_offsets(tr_xy  if len(tr_xy)  else np.zeros((0, 2)))

        # CLI dot
        ci = curve_idx_for_k(k)
        dot_cli.set_data([curve["t_rel"].iloc[ci]], [curve["CLI"].iloc[ci]])

        frame_text.set_text(f"Frame {int(fr)} | CLI={curve['CLI'].iloc[ci]:+.2f}")

        return scat_off, scat_def, scat_tr, dot_cli, frame_text, title_text

    # 0.7x slower: increase frame interval
    interval_ms = (1000.0 / fps) / max(1e-6, float(speed_mult))

    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)

    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    ani.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return out_gif



import numpy as np
import pandas as pd

# --------------------------------------------
# Select 4 representative plays (examples)
# 1 Completion + 3 distinct CLI failure mechanisms
# --------------------------------------------
def select_4_plays(summ_all):
    sel = {}

    # ------------------------------------------------
    # Completion example: offense clearly owns leverage
    # (high peak_CLI, large separation)
    # ------------------------------------------------
    sel["C_completion"] = (
        summ_all[summ_all.pass_result == "C"]
        .sort_values(["peak_CLI", "CDI"], ascending=False)
        .iloc[212][["game_id", "play_id", "pass_result"]]
    )

    I = summ_all[summ_all.pass_result == "I"].copy()

    # ------------------------------------------------
    # Incompletion example 1:
    # Defensive collapse late in flight (high VCR)
    # → def_collapse_INC
    # ------------------------------------------------
    if "VCR" in I.columns:
        sel["I_def_collapse"] = (
            I.sort_values("VCR", ascending=False)
            .iloc[10][["game_id", "play_id", "pass_result"]]
        )
    else:
        sel["I_def_collapse"] = (
            I.sort_values("CDI", ascending=False)
            .iloc[10][["game_id", "play_id", "pass_result"]]
        )

    # ------------------------------------------------
    # Incompletion example 2:
    # Offense controlled leverage but still incomplete
    # → off_control_INC
    # ------------------------------------------------
    sel["I_off_control"] = (
        I.sort_values("final_CLI", ascending=False)
        .iloc[200][["game_id", "play_id", "pass_result"]]
    )

    # ------------------------------------------------
    # Incompletion example 3:
    # Offense never had a viable window
    # → never_viable_INC
    # ------------------------------------------------
    sel["I_never_viable"] = (
        I.sort_values("final_CLI", ascending=True)
        .iloc[10][["game_id", "play_id", "pass_result"]]
    )

    return (
        pd.DataFrame(sel)
        .T
        .reset_index()
        .rename(columns={"index": "label"})
    )

# Run selection
sel = select_4_plays(summ_all)
sel





from IPython.display import HTML, display
import numpy as np

def play_title(summ_all, game_id, play_id, label):
    row = summ_all[
        (summ_all.game_id == game_id) &
        (summ_all.play_id == play_id)
    ]

    if row.empty:
        return label

    r = row.iloc[0]

    return (
        f"{label.replace('_',' ').title()}  |  "
        f"Peak CLI={r['peak_CLI']:.2f}, "
        f"Final CLI={r['final_CLI']:.2f}, "
        f"CDI={r['CDI']:.2f}"
    )


gif_paths = []

for _, row in sel.iterrows():
    label = row["label"]
    g = row["game_id"]
    p = row["play_id"]

    out = f"/kaggle/working/cli_play_{label}.gif"
    title = play_title(summ_all, g, p, label)

    print(f"Making GIF for {label} (game {g}, play {p})")

    make_play_cli_gif(
        tracking_bia_kin=tracking_bia_kin,
        curves_all=curves_all,
        game_id=g,
        play_id=p,
        out_gif=out,
        title=title,
        max_frames=85,   # sweet spot: readable + small file
        fps = int(12 * 0.7)  # to slow it a bit

    )

    gif_paths.append(out)


import base64
from IPython.display import HTML, display

def gif_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


cells = []

for path in gif_paths:
    b64 = gif_to_base64(path)
    cells.append(
        f"""
        <td style="padding:12px; text-align:center;">
            <img src="data:image/gif;base64,{b64}"
                 style="width:520px;
                        border:1px solid #ccc;
                        border-radius:10px;">
        </td>
        """
    )

html = f"""
<table style="margin:auto;">
  <tr>
    {cells[0]} {cells[1]}
  </tr>
  <tr>
    {cells[2]} {cells[3]}
  </tr>
</table>
"""

display(HTML(html))



# --- Install once (safe to re-run) ---
!pip -q install imageio imageio-ffmpeg pillow

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import os

# gif_paths should be length 4
assert len(gif_paths) == 4, f"Expected 4 GIFs, got {len(gif_paths)}"

# -----------------------------
# Controls (match your HTML look)
# -----------------------------
PANEL_W = 520           # match your <img style="width:520px">
BORDER = 6              # px border around each panel
GAP    = 12             # padding between panels
BG     = (255, 255, 255)  # white background like notebook
FPS    = 10             # tracking rate-ish; adjust 8–12 if needed

out_path = "/kaggle/working/cli_film_study_2x2.mp4"

# -----------------------------
# Helpers
# -----------------------------
def to_rgb(frame):
    """Ensure frame is RGB uint8."""
    if frame.ndim == 2:  # grayscale
        frame = np.stack([frame]*3, axis=-1)
    if frame.shape[-1] == 4:  # RGBA -> RGB on white
        rgba = frame.astype(np.float32)
        alpha = rgba[..., 3:4] / 255.0
        rgb = rgba[..., :3] * alpha + (255.0 * (1 - alpha))
        frame = rgb.astype(np.uint8)
    return frame.astype(np.uint8)

def resize_keep_aspect(frame, target_w):
    """Resize to target width, keep aspect ratio."""
    img = Image.fromarray(frame)
    w, h = img.size
    target_h = int(round(h * (target_w / w)))
    img = img.resize((target_w, target_h), Image.BICUBIC)
    return np.array(img)

def add_border(frame, border=BORDER, color=(204,204,204)):
    """Add a thin gray border like your HTML."""
    h, w, _ = frame.shape
    out = np.full((h + 2*border, w + 2*border, 3), color, dtype=np.uint8)
    out[border:border+h, border:border+w] = frame
    return out

# -----------------------------
# Load GIFs (frames)
# -----------------------------
gifs = [imageio.mimread(p) for p in gif_paths]
min_frames = min(len(g) for g in gifs)
gifs = [g[:min_frames] for g in gifs]

# Convert + resize + border each frame
proc = []
for gi in gifs:
    frames = []
    for fr in gi:
        fr = to_rgb(np.array(fr))
        fr = resize_keep_aspect(fr, PANEL_W)
        fr = add_border(fr, BORDER, color=(204,204,204))
        frames.append(fr)
    proc.append(frames)

# Make all panels same height (pad with white if needed)
panel_h = max(fr.shape[0] for frames in proc for fr in frames)
panel_w = proc[0][0].shape[1]  # after border, width consistent

def pad_to_h(frame, H, bg=BG):
    h, w, _ = frame.shape
    if h == H:
        return frame
    out = np.full((H, w, 3), bg, dtype=np.uint8)
    top = (H - h) // 2
    out[top:top+h] = frame
    return out

proc = [[pad_to_h(fr, panel_h) for fr in frames] for frames in proc]

# -----------------------------
# Build 2×2 grid frames
# -----------------------------
grid_frames = []
H = 2*panel_h + GAP
W = 2*panel_w + GAP

for t in range(min_frames):
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    # positions
    # top-left
    canvas[0:panel_h, 0:panel_w] = proc[0][t]
    # top-right
    canvas[0:panel_h, panel_w+GAP:panel_w+GAP+panel_w] = proc[1][t]
    # bottom-left
    canvas[panel_h+GAP:panel_h+GAP+panel_h, 0:panel_w] = proc[2][t]
    # bottom-right
    canvas[panel_h+GAP:panel_h+GAP+panel_h, panel_w+GAP:panel_w+GAP+panel_w] = proc[3][t]

    grid_frames.append(canvas)

print("Grid video frames:", len(grid_frames), "Frame size:", grid_frames[0].shape)

# -----------------------------
# Write MP4 (YouTube-ready)
# -----------------------------
imageio.mimsave(
    out_path,
    grid_frames,
    fps=FPS,
    codec="libx264",
    quality=8
)

print("Saved:", out_path, "bytes:", os.path.getsize(out_path))



import numpy as np
import pandas as pd
from IPython.display import HTML, display

# ---------- choose groups ----------
groups = [
    "completion",
    "def_collapse_INC",
    "off_control_INC",
    "never_viable_INC",
]

#  FINAL, CONSISTENT DISPLAY NAMES
group_name = {
    "completion":        "Completion (C)",
    "def_collapse_INC":  "Incomplete — Late breakup",
    "off_control_INC":   "Incomplete — Offense-owned miss",
    "never_viable_INC":  "Incomplete — Never viable",
}

metrics = ["peak_CLI", "final_CLI", "CDI", "VCT_sec", "VCR"]

# Metric names also standardized
metric_defs = {
    "peak_CLI":  ("Peak CLI",  r"max$_t$ CLI(t)",                 "Highest offensive leverage achieved"),
    "final_CLI": ("Final CLI", r"CLI(t$_{final}$)",               "Leverage at arrival (who controls catch point)"),
    "CDI":       ("Collapse Depth (CDI)", r"peak − final",        "Total leverage lost after peak"),
    "VCT_sec":   ("Viability Collapse Time (VCT)", r"first t≥t$_{peak}$: drop>0.10", "How quickly the window closes"),
    "VCR":       ("Viability Collapse Rate (VCR)", r"CDI / (t$_{final}$−t$_{peak}$)", "Speed of leverage loss"),
}

df = summ_all.copy()
df = df[df["type"].isin(groups)].copy()

for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")

def pack_stats(x):
    x = x.dropna()
    if len(x) == 0:
        return dict(n=0, med=np.nan, q25=np.nan, q75=np.nan)
    return dict(
        n=len(x),
        med=float(x.median()),
        q25=float(x.quantile(0.25)),
        q75=float(x.quantile(0.75)),
    )

def fmt_cell(s):
    if s["n"] == 0 or np.isnan(s["med"]):
        return "—"
    return (
        f"{s['med']:.3f} "
        f"[{s['q25']:.3f}, {s['q75']:.3f}]"
        f"<br><span style='color:#666;'>n={s['n']}</span>"
    )

rows = []
for met in metrics:
    mname, mdef, mmeaning = metric_defs[met]
    row = {
        "Metric": f"<b>{mname}</b>",
        "Definition": f"<span style='white-space:nowrap;'>{mdef}</span>",
        "What it measures": f"<span style='color:#333;'>{mmeaning}</span>",
    }
    for g in groups:
        s = pack_stats(df.loc[df["type"] == g, met])
        row[group_name[g]] = fmt_cell(s)
    rows.append(row)

table_df = pd.DataFrame(rows)

# ---- Clean, judge-friendly render (Markdown-safe HTML) ----
styles = """
<style>
.cli-table {border-collapse: collapse; width: 100%; font-size: 12.5px;}
.cli-table th, .cli-table td {border: 1px solid #e6e6e6; padding: 8px; vertical-align: top;}
.cli-table th {background: #fafafa;}
.cli-table td {line-height: 1.25;}
</style>
"""

html = styles + table_df.to_html(index=False, escape=False, classes="cli-table")
display(HTML(html))



import dataframe_image as dfi

# Save to Kaggle working directory
out_path = "/kaggle/working/cli_metrics_summary.png"

dfi.export(
    table_df,
    out_path,
    table_conversion="matplotlib"  # safest in Kaggle
)

print("Saved:", out_path)



import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# (A) Colors keyed ONLY by the raw group keys (what you want)
# ---------------------------------------------------------
COL = {
    # outcome subtype keys
    "completion": "#1b9e77",
    "def_collapse_INC": "#d95f02",
    "off_control_INC": "#1f78b4",
    "never_viable_INC": "#b23a2f",

    # man/zone cleaned keys
    "ZONE_COVERAGE": "#1f78b4",
    "MAN_COVERAGE":  "#b23a2f",
}



# ---------------------------------------------------------
# (B) Utility: normalize man_zone into MAN_COVERAGE / ZONE_COVERAGE
# ---------------------------------------------------------
def normalize_man_zone(series):
    mz = series.astype(str).str.upper().str.strip()
    # common variants
    mz = mz.replace({
        "MAN": "MAN_COVERAGE",
        "ZONE": "ZONE_COVERAGE",
        "MAN_COVERAGE": "MAN_COVERAGE",
        "ZONE_COVERAGE": "ZONE_COVERAGE",
    })
    # anything else -> NaN (so it drops cleanly)
    mz = mz.where(mz.isin(["MAN_COVERAGE", "ZONE_COVERAGE"]))
    return mz

# ---------------------------------------------------------
# (C) Plot KM curves on a given axis (labels stay as keys)
# Requires: km_curve, km_median, bootstrap_km_band already defined above
# ---------------------------------------------------------
def plot_km_on_ax(
    ax,
    df,
    group_col,
    groups,
    min_n=200,
    title="",
    DELTA=0.10,
    N_BOOT=0,
    BAND_Q=(0.10, 0.90),
    show_median_lines=True,
):
    use = df.dropna(subset=[group_col]).copy()

    # If groups not given, auto-order by frequency
    if groups is None:
        groups = list(use[group_col].value_counts().index)

    grid = np.linspace(0, 1, 101)

    for g in groups:
        dfg = use[use[group_col] == g]
        n = len(dfg)
        if n < min_n:
            continue

        km = km_curve(dfg["t"].values, dfg["event"].values)
        med = km_median(km)

        ax.step(
            km["t"], km["S"], where="post",
            lw=2.4, alpha=0.95,
            color=COL.get(g, None),
            label=f"{g} (n={n}, med={med:.2f})"
        )

        # Bootstrap band (optional)
        if N_BOOT and n >= 50:
            band = bootstrap_km_band(dfg, n_boot=N_BOOT, q=BAND_Q, grid=grid)
            if band is not None:
                ggrid, lo, hi = band
                ax.fill_between(
                    ggrid, lo, hi, step="post",
                    alpha=0.10, linewidth=0,
                    color=COL.get(g, None)
                )

        # Median vertical line
        if show_median_lines and np.isfinite(med):
            ax.axvline(med, ls="--", lw=1.2, alpha=0.55, color=COL.get(g, None))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Normalized time in air (τ)")
    ax.set_ylabel(f"Survival: P(window alive)  (alive if CLI ≥ peak−{DELTA:.2f})")
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)

# ---------------------------------------------------------
# (D) Side-by-side figure: (1) outcome subtype, (2) man vs zone
# ---------------------------------------------------------
def plot_layer2_side_by_side(
    plays,
    DELTA=0.10,
    N_BOOT=0,          # set 200–400 for subtle bands
    BAND_Q=(0.10, 0.90),
    min_n_type=50,     # IMPORTANT: set <=61 so never_viable_INC shows
    min_n_mz=500,
):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)

    # ---- Left: by outcome subtype (FORCE 4 lines) ----
    type_groups = ["completion", "def_collapse_INC", "off_control_INC", "never_viable_INC"]
    plot_km_on_ax(
        axes[0],
        plays,
        group_col="type",
        groups=type_groups,
        min_n=min_n_type,
        title="Catch window survival by outcome subtype",
        DELTA=DELTA,
        N_BOOT=N_BOOT,
        BAND_Q=BAND_Q,
        show_median_lines=True,
    )

    # ---- Right: by Man/Zone ----
    tmp = plays.copy()
    tmp["man_zone_clean"] = normalize_man_zone(tmp["man_zone"]) if "man_zone" in tmp.columns else np.nan

    plot_km_on_ax(
        axes[1],
        tmp,
        group_col="man_zone_clean",
        groups=["ZONE_COVERAGE", "MAN_COVERAGE"],  # show Zone then Man
        min_n=min_n_mz,
        title="Catch window survival by team coverage (Man vs Zone)",
        DELTA=DELTA,
        N_BOOT=N_BOOT,
        BAND_Q=BAND_Q,
        show_median_lines=True,
    )

    fig.suptitle(
        "Layer 2 (Meso): When does the catch window die at the population level?",
        fontsize=13, weight="bold", y=1.02
    )

    plt.tight_layout()
    # Save BEFORE show
    plt.savefig(
        "/kaggle/working/fig_layer2_KM.png",
        dpi=200,
        bbox_inches="tight")
    plt.show()

plot_layer2_side_by_side(
    plays,
    DELTA=0.10,
    N_BOOT=0,          # set to 250 if you want subtle bands
    BAND_Q=(0.10, 0.90),
    min_n_type=50,     # shows never_viable_INC (n≈61)
    min_n_mz=500
)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#1f78b4"
RED  = "#b23a2f"

# blue -> light neutral -> red (nice around vcenter=0)
CVI_RISK_CMAP = LinearSegmentedColormap.from_list(
    "cvi_risk",
    [BLUE, "#f2f2f2", RED],
    N=256
)

# ============================================================
# 0) Parameters (keep these consistent across the notebook)
# ============================================================
TAIL_THR = 0.90

MIN_N_MAN  = 60
MIN_N_ZONE = 150

LABEL_Q_TAIL = 0.90   # label top 10% tail_rate
LABEL_Q_SURV = 0.10   # label bottom 10% survival

# ============================================================
# 1) Build a clean per-play table `pp2` (expects pp already exists)
#    Required cols in pp:
#      scheme, defensive_team, peak_CLI, t, expected_points_added
# ============================================================
pp2 = pp.dropna(subset=["scheme", "defensive_team", "peak_CLI", "t"]).copy()
pp2["scheme"] = pp2["scheme"].astype(str).str.strip().str.title()  # "Man", "Zone", etc.

print("pp2 usable:", pp2.shape)

# ============================================================
# 2) Aggregate team metrics (scheme-split)
# ============================================================
base = (
    pp2.groupby(["scheme", "defensive_team"])
       .size()
       .reset_index(name="n")
)

metrics = (
    pp2.groupby(["scheme", "defensive_team"])
       .agg(
           tail_rate=("peak_CLI", lambda x: float((x >= TAIL_THR).mean())),
           median_tau_alive=("t", lambda x: float(np.median(x))),
           peak_med=("peak_CLI", lambda x: float(np.median(x))),
           peak_p90=("peak_CLI", lambda x: float(np.quantile(x, 0.90))),
           mean_tail_epa=("expected_points_added",
                          lambda s: float(s[pp2.loc[s.index, "peak_CLI"] >= TAIL_THR].mean())
                          if (pp2.loc[s.index, "peak_CLI"] >= TAIL_THR).any() else np.nan)
       )
       .reset_index()
)

team_scheme = base.merge(metrics, on=["scheme", "defensive_team"], how="left")

# filter small samples differently for Man/Zone
team_scheme = team_scheme[
    ((team_scheme["scheme"] == "Man")  & (team_scheme["n"] >= MIN_N_MAN)) |
    ((team_scheme["scheme"] == "Zone") & (team_scheme["n"] >= MIN_N_ZONE))
].copy()

print("team_scheme after MIN_N filters:", team_scheme.shape)
display(team_scheme.head())

# ============================================================
# 3) Build league-wide team_profile (combine Man+Zone rows per team)
#    Weighted by n for stability.
# ============================================================
def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    den = np.sum(w)
    return float(np.sum(x * w) / den) if den > 0 else np.nan

team_profile = (
    team_scheme.dropna(subset=["tail_rate", "median_tau_alive"])
    .groupby("defensive_team")
    .apply(lambda g: pd.Series({
        "n": int(g["n"].sum()),
        "tail_rate": weighted_mean(g["tail_rate"].values, g["n"].values),
        "median_tau_alive": weighted_mean(g["median_tau_alive"].values, g["n"].values),
        "mean_tail_epa": weighted_mean(
            np.nan_to_num(g["mean_tail_epa"].values, nan=0.0),
            g["n"].values
        )
    }))
    .reset_index()
)

# ============================================================
# 4) Plot helpers
# ============================================================
def safe_sizes(values, min_s=260, max_s=2600):
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=0.0)
    mx = v.max()
    if mx <= 1e-9:
        return np.full(len(v), min_s, dtype=float)
    return min_s + (max_s - min_s) * (v / mx)

def style_axes(ax, tail_thr):
    ax.set_xlabel(f"Catastrophic leverage rate  P(peak_CLI ≥ {tail_thr:.2f})")
    ax.set_ylabel("Median catch-window survival time (τ)")
    ax.grid(alpha=0.25)

def label_extremes(ax, df, xcol, ycol, labelcol,
                   qx=0.90, qy=0.10, dx=0.002, fontsize=9,
                   alpha=0.95):
    x_thr = df[xcol].quantile(qx)
    y_thr = df[ycol].quantile(qy)
    for _, r in df.iterrows():
        if (r[xcol] >= x_thr) or (r[ycol] <= y_thr):
            ax.text(
                r[xcol] + dx, r[ycol], str(r[labelcol]),
                fontsize=fontsize, weight="bold", color="black", alpha=alpha
            )

# ============================================================
# 5) Plot A: League-wide defensive risk map (one bubble chart)
# ============================================================
fig, ax = plt.subplots(figsize=(10.8, 6.3))

sizes = safe_sizes(team_profile["mean_tail_epa"], min_s=260, max_s=2600)

# Risk score (color): higher = worse (more catastrophic peaks + earlier collapse)
risk_score = (
    (team_profile["tail_rate"] - team_profile["tail_rate"].median()) /
    (team_profile["tail_rate"].std(ddof=0) + 1e-9)
    -
    (team_profile["median_tau_alive"] - team_profile["median_tau_alive"].median()) /
    (team_profile["median_tau_alive"].std(ddof=0) + 1e-9)
)

# Center color scale at 0 so blue=lower risk, red=higher risk consistently
norm = TwoSlopeNorm(vmin=float(np.nanmin(risk_score)),
                    vcenter=0.0,
                    vmax=float(np.nanmax(risk_score)))

sc = ax.scatter(
    team_profile["tail_rate"],
    team_profile["median_tau_alive"],
    s=sizes,
    c=risk_score,
    cmap=CVI_RISK_CMAP,
    norm=norm,
    alpha=0.85,
    edgecolor="k",
    linewidth=0.5
)

# League medians (quadrants)
x_med = float(team_profile["tail_rate"].median())
y_med = float(team_profile["median_tau_alive"].median())
ax.axvline(x_med, linestyle="--", alpha=0.35)
ax.axhline(y_med, linestyle="--", alpha=0.35)

ax.set_title(
    "When coverage breaks: how often, how fast, and how costly?\n"
    "(bubble size = EPA on catastrophic plays; color = composite risk)",
    fontsize=12.5, weight="bold"
)

# Label only structurally interesting teams (top tail risk OR bottom survival)
label_extremes(
    ax, team_profile,
    xcol="tail_rate", ycol="median_tau_alive", labelcol="defensive_team",
    qx=0, qy=0, dx=0.002, fontsize=9
)

style_axes(ax, TAIL_THR)


# Colorbar 
cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(
    "Composite defensive risk score\n"
    "(higher = more frequent breakdowns & slower recovery)",
    fontsize=10
)

plt.tight_layout()
plt.savefig(
    "/kaggle/working/bubble1.png",
    dpi=250,
    bbox_inches="tight"
)
plt.show()

# ============================================================
# 6) Plot B: Man vs Zone split (two panels, shared axes)
# ============================================================

SCHEME_COLOR = {
    "Man":  RED,
    "Zone": BLUE
}

def plot_scheme_panel(ax, df_panel, title, color):
    if df_panel.empty:
        ax.set_title(title + " (no data after filtering)")
        ax.axis("off")
        return

    sizes = safe_sizes(df_panel["mean_tail_epa"], min_s=240, max_s=2200)

    ax.scatter(
        df_panel["tail_rate"],
        df_panel["median_tau_alive"],
        s=sizes,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
        color=color   # <-- force scheme color
    )

    x_med = float(df_panel["tail_rate"].median())
    y_med = float(df_panel["median_tau_alive"].median())
    ax.axvline(x_med, linestyle="--", alpha=0.35, color="#9e9e9e")
    ax.axhline(y_med, linestyle="--", alpha=0.35, color="#9e9e9e")

    label_extremes(
        ax, df_panel,
        xcol="tail_rate", ycol="median_tau_alive", labelcol="defensive_team",
        qx=LABEL_Q_TAIL, qy=LABEL_Q_SURV, dx=0.002, fontsize=9
    )

    ax.set_title(title, fontsize=12, weight="bold")
    style_axes(ax, TAIL_THR)


fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.1), sharex=True, sharey=True)

plot_scheme_panel(
    axes[0],
    team_scheme[team_scheme["scheme"] == "Man"].copy(),
    "Defensive risk profiles (Man-heavy plays)",
    color=SCHEME_COLOR["Man"]
)

plot_scheme_panel(
    axes[1],
    team_scheme[team_scheme["scheme"] == "Zone"].copy(),
    "Defensive risk profiles (Zone-heavy plays)",
    color=SCHEME_COLOR["Zone"]
)

fig.suptitle(
    "Scheme split: is defensive risk driven by scheme or execution?\n"
    "(bubble = EPA on catastrophic plays)",
    y=1.02, fontsize=13, weight="bold"
)

plt.tight_layout()
# Save BEFORE show
plt.savefig(
    "/kaggle/working/bubble.png",
    dpi=250,
    bbox_inches="tight"
)
plt.show()



FILM_PLAYS = [
    {
        "label": "FAST collapse (I) — Pickett, g=2023110200 p=932",
        "game_id": 2023110200,
        "play_id": 932,
        "pass_result": "I",
        "team_coverage_type": "COVER_1_MAN",
        "pass_length": 13,
        "peak_CLI": 0.998104,
        "VCT_sec": 0.3,
        "t_peak": 0.2,
        "t_final": 1.0,
        "tau_alive": 0.30,
        "play_description": "(2:07) (Shotgun) K.Pickett pass incomplete ...",
    },
    {
        "label": "SLOW collapse (C) — Walker, g=2023101504 p=1585",
        "game_id": 2023101504,
        "play_id": 1585,
        "pass_result": "C",
        "team_coverage_type": "COVER_3_ZONE",
        "pass_length": 23,
        "peak_CLI": 1.000000,
        "VCT_sec": 1.9,
        "t_peak": 0.2,
        "t_final": 2.0,
        "tau_alive": 0.95,
        "play_description": "(7:45) (Shotgun) P.Walker pass deep left ...",
    },
]



from IPython.display import HTML, display

# --- your two chosen plays ---
film_sel = [
    ("FAST_collapse_INCOMPLETE", 2023110200, 932),
    ("SLOW_collapse_CCOMPLETE", 2023101504, 1585),
]

def play_title(summ_all, df_supp, game_id, play_id, label):
    # 1) Try summ_all
    row = summ_all[(summ_all.game_id == game_id) & (summ_all.play_id == play_id)]
    cov = None

    if not row.empty and "team_coverage_type" in row.columns:
        cov = row.iloc[0].get("team_coverage_type", None)

    # 2) Fallback to df_supp only if needed
    if (cov is None or (isinstance(cov, float) and np.isnan(cov))) and df_supp is not None:
        r2 = df_supp[(df_supp.game_id == game_id) & (df_supp.play_id == play_id)]
        if (not r2.empty) and ("team_coverage_type" in r2.columns):
            cov = r2.iloc[0].get("team_coverage_type", None)

    cov = cov if cov is not None else "—"

    # Metrics (from summ_all)
    if row.empty:
        return f"{label} | {cov}"

    r = row.iloc[0]
    return (
        f"{label} | {cov}\n"
        f"peak={r['peak_CLI']:.2f}, final={r['final_CLI']:.2f}, "
        f"CDI={r['CDI']:.2f}, VCT={r['VCT_sec']:.2f}s, VCR={r['VCR']:.2f}"
    )


gif_paths = []

for label, g, p in film_sel:
    out = f"/kaggle/working/film_{label}_{g}_{p}.gif"

    # USE THIS (NOT film_play_title)
    title = play_title(summ_all,df_supp, g, p, label)

    print("Making:", out)
    make_play_cli_gif(
        tracking_bia_kin=tracking_bia_kin,
        curves_all=curves_all,
        game_id=g,
        play_id=p,
        out_gif=out,
        title=title,
        max_frames=95,
        fps=12,
        speed_mult=0.50
    )
    gif_paths.append(out)

# --- display 1×2 grid ---
cells = []
for path in gif_paths:
    b64 = gif_to_base64(path)
    cells.append(
        f"""
        <td style="padding:14px; text-align:center; vertical-align:top;">
            <img src="data:image/gif;base64,{b64}"
                 style="width:520px; border:1px solid #ccc; border-radius:10px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.15);">
        </td>
        """
    )

html = f"""
<table style="margin:auto;">
  <tr>
    {cells[0]}
    {cells[1]}
  </tr>
</table>
"""

display(HTML(html))



import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image

# -----------------------------
# 1) Pick TWO plays (1×2)
# -----------------------------
film_sel = [
    ("FAST_collapse_INCOMPLETE", 2023110200, 932),
    ("SLOW_collapse_COMPLETE",   2023101504, 1585),
]

def play_title(summ_all, df_supp, game_id, play_id, label):
    row = summ_all[(summ_all.game_id == game_id) & (summ_all.play_id == play_id)]
    cov = None

    if not row.empty and "team_coverage_type" in row.columns:
        cov = row.iloc[0].get("team_coverage_type", None)

    if (cov is None or (isinstance(cov, float) and np.isnan(cov))) and df_supp is not None:
        r2 = df_supp[(df_supp.game_id == game_id) & (df_supp.play_id == play_id)]
        if (not r2.empty) and ("team_coverage_type" in r2.columns):
            cov = r2.iloc[0].get("team_coverage_type", None)

    cov = cov if cov is not None else "—"
    if row.empty:
        return f"{label} | {cov}"

    r = row.iloc[0]
    return (
        f"{label} | {cov}\n"
        f"peak={r['peak_CLI']:.2f}, final={r['final_CLI']:.2f}, "
        f"CDI={r['CDI']:.2f}, VCT={r['VCT_sec']:.2f}s, VCR={r['VCR']:.2f}"
    )

# -----------------------------
# 2) Generate 2 GIFs → gif_paths
# -----------------------------
gif_paths = []
for label, g, p in film_sel:
    out = f"/kaggle/working/film_{label}_{g}_{p}.gif"
    title = play_title(summ_all, df_supp, g, p, label)

    print("Making:", out)
    make_play_cli_gif(
        tracking_bia_kin=tracking_bia_kin,
        curves_all=curves_all,
        game_id=g,
        play_id=p,
        out_gif=out,
        title=title,
        max_frames=95,
        fps=12,
        speed_mult=0.50
    )
    gif_paths.append(out)

assert len(gif_paths) == 2, f"Expected 2 GIFs, got {len(gif_paths)}"
print("GIFs ready:", gif_paths)

# -----------------------------
# 3) Convert 2 GIFs → 1×2 MP4 (YouTube-ready)
# -----------------------------
PANEL_W = 520
BORDER  = 6
GAP     = 12
BG      = (255, 255, 255)
FPS     = 10

out_path = "/kaggle/working/cli_film_study_1x2.mp4"

def to_rgb(frame):
    if frame.ndim == 2:
        frame = np.stack([frame]*3, axis=-1)
    if frame.shape[-1] == 4:
        rgba = frame.astype(np.float32)
        alpha = rgba[..., 3:4] / 255.0
        rgb = rgba[..., :3] * alpha + (255.0 * (1 - alpha))
        frame = rgb.astype(np.uint8)
    return frame.astype(np.uint8)

def resize_keep_aspect(frame, target_w):
    img = Image.fromarray(frame)
    w, h = img.size
    target_h = int(round(h * (target_w / w)))
    img = img.resize((target_w, target_h), Image.BICUBIC)
    return np.array(img)

def add_border(frame, border=BORDER, color=(204,204,204)):
    h, w, _ = frame.shape
    out = np.full((h + 2*border, w + 2*border, 3), color, dtype=np.uint8)
    out[border:border+h, border:border+w] = frame
    return out

# load + align by shortest GIF
gifs = [imageio.mimread(p) for p in gif_paths]
min_frames = min(len(g) for g in gifs)
gifs = [g[:min_frames] for g in gifs]

# preprocess frames (resize + border)
proc = []
for gi in gifs:
    frames = []
    for fr in gi:
        fr = to_rgb(np.array(fr))
        fr = resize_keep_aspect(fr, PANEL_W)
        fr = add_border(fr, BORDER, color=(204,204,204))
        frames.append(fr)
    proc.append(frames)

# pad to equal height
panel_h = max(fr.shape[0] for frames in proc for fr in frames)
panel_w = proc[0][0].shape[1]

def pad_to_h(frame, H, bg=BG):
    h, w, _ = frame.shape
    if h == H:
        return frame
    out = np.full((H, w, 3), bg, dtype=np.uint8)
    top = (H - h) // 2
    out[top:top+h] = frame
    return out

proc = [[pad_to_h(fr, panel_h) for fr in frames] for frames in proc]

# build 1×2 grid
grid_frames = []
H = panel_h
W = 2*panel_w + GAP

for t in range(min_frames):
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    canvas[:, 0:panel_w] = proc[0][t]
    canvas[:, panel_w+GAP:panel_w+GAP+panel_w] = proc[1][t]
    grid_frames.append(canvas)

print("Grid frames:", len(grid_frames), "Frame size:", grid_frames[0].shape)

# write MP4
imageio.mimsave(out_path, grid_frames, fps=FPS, codec="libx264", quality=8)

print("Saved:", out_path, "bytes:", os.path.getsize(out_path))



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob, os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_PATH = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final"
TRAIN_PATH = f"{BASE_PATH}/train"
SUPP_PATH = f"{BASE_PATH}/supplementary_data.csv"

input_files = sorted(glob.glob(f"{TRAIN_PATH}/input_2023_w*.csv"))
output_files = sorted(glob.glob(f"{TRAIN_PATH}/output_2023_w*.csv"))

df_supp = pd.read_csv(SUPP_PATH, low_memory=False)

len(input_files), len(output_files), df_supp.shape



import os
import numpy as np
import pandas as pd

def load_weeks(files, kind, dtype=None):
    """
    Load weekly CSVs and attach a `week` and `kind` column.

    kind: "input" or "output"
    """
    dfs = []
    for f in files:
        wk = int(os.path.basename(f).split("_w")[-1].split(".csv")[0])
        df = pd.read_csv(f, dtype=dtype, low_memory=False)
        df["week"] = wk
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["kind"] = kind
    return out

df_input_all  = load_weeks(input_files,  "input")
df_output_all = load_weeks(output_files, "output")

print("df_input_all:", df_input_all.shape)
print("df_output_all:", df_output_all.shape)




print("INPUT:", df_input_all.shape)
print(df_input_all.columns.tolist())
print()

print("OUTPUT:", df_output_all.shape)
print(df_output_all.columns.tolist())
print()

print("SUPP:", df_supp.shape)
print(df_supp.columns.tolist())



# --- 1) Play-level table: landing point + ball flight + context ---

# From input: aggregate to play-level
play_agg = (
    df_input_all
    .groupby(["game_id", "play_id"], as_index=False)
    .agg(
        week=("week", "first"),
        play_direction=("play_direction", "first"),
        absolute_yardline_number=("absolute_yardline_number", "first"),
        ball_land_x=("ball_land_x", "first"),
        ball_land_y=("ball_land_y", "first"),
        num_frames_output=("num_frames_output", "first"),
    )
)

# From supplementary: keep only columns we care about
supp_keep = [
    "game_id", "play_id",
    "pass_result", "pass_length",
    "route_of_targeted_receiver",
    "team_coverage_man_zone", "team_coverage_type",
    "down", "yards_to_go",
    "expected_points", "expected_points_added",
]
supp_small = df_supp[supp_keep].drop_duplicates()

# Merge to form play-level table
plays = play_agg.merge(supp_small, on=["game_id", "play_id"], how="left")

print("plays:", plays.shape)
plays.head()



#  2) Player state at throw: last input frame per (game, play, player) ---

input_core = df_input_all[
    [
        "game_id", "play_id", "nfl_id", "frame_id",
        "player_position", "player_side", "player_role",
        "x", "y", "s", "a", "dir", "o",
    ]
].copy()

# Sort so that tail(1) gives last frame per (game_id, play_id, nfl_id)
throw_state = (
    input_core
    .sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
    .tail(1)
    .rename(
        columns={
            "x": "x_throw",
            "y": "y_throw",
            "s": "s_throw",
            "a": "a_throw",
            "dir": "dir_throw",
            "o": "o_throw",
        }
    )
)

print("throw_state:", throw_state.shape)
throw_state.head()



# --- 3) Ball-in-air tracking table (output + throw_state + play context) ---

# Core columns from output (ball-in-air)
output_core = df_output_all[
    ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "week"]
].copy()

# Merge in player throw frame state
tracking_bia = output_core.merge(
    throw_state,
    on=["game_id", "play_id", "nfl_id"],
    how="left",
)

# Merge in play-level info (landing, pass result, coverage, etc.)
tracking_bia = tracking_bia.merge(
    plays,
    on=["game_id", "play_id"],
    how="left",
)

print("tracking_bia:", tracking_bia.shape)
tracking_bia.head()



import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FPS = 10  # tracking frequency (Hz)

def compute_smoothed_kinematics(g, fps=FPS):
    """
    Clean velocity + acceleration for ball-in-air frames of ONE player
    (single group of game_id, play_id, nfl_id), sorted by frame_id.
    """
    g = g.sort_values("frame_id").copy()

    # Raw finite-diff velocity from x,y
    g["dx"] = g["x"].diff().fillna(0.0)
    g["dy"] = g["y"].diff().fillna(0.0)

    g["vx_raw"] = g["dx"] * fps
    g["vy_raw"] = g["dy"] * fps
    g["speed_raw"] = np.sqrt(g["vx_raw"]**2 + g["vy_raw"]**2)

    # ---- Smooth speed with Savitzky–Golay (if enough frames) ----
    if len(g) >= 5:
        g["speed_smooth"] = savgol_filter(
            g["speed_raw"].values, window_length=5, polyorder=2
        )
    else:
        g["speed_smooth"] = g["speed_raw"]

    # Acceleration from smoothed speed
    g["accel_smooth"] = g["speed_smooth"].diff().fillna(0.0) * fps

    # Clip to physically reasonable range
    g["accel_smooth"] = g["accel_smooth"].clip(-8.0, 8.0)

    # Heading from velocity (still using raw vx/vy is fine)
    heading_rad = np.arctan2(g["vy_raw"], g["vx_raw"])
    g["heading_bia"] = np.degrees(heading_rad).fillna(0.0)

    return g


def build_post_throw_kinematics(tracking_bia, fps=FPS):
    """
    Enrich tracking_bia (ball-in-air tracking) with:
      - vx, vy           : velocity components (yd/s)
      - speed_bia        : smoothed speed magnitude (yd/s)
      - accel_bia        : smoothed acceleration magnitude (yd/s^2)
      - heading_bia      : movement direction (deg) from velocity
      - angle_to_land    : angle from player to landing point (deg)
      - dist_to_land     : distance to landing point (yd)
      - dist_to_TR       : distance to targeted receiver (yd)
      - t_since_throw    : seconds since throw (0 at frame_id=1)
      - time_remaining   : seconds until catch/incomplete
    Keeps throw-frame orientation/dir as `o_throw`, `dir_throw`.
    """
    df = tracking_bia.copy()

    # --- Clean up column names from the merges ---
    if "frame_id_x" in df.columns:
        df = df.rename(columns={"frame_id_x": "frame_id"})
    if "week_x" in df.columns:
        df = df.rename(columns={"week_x": "week"})
    for col in ["frame_id_y", "week_y"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # --- Basic time axes during ball-in-air ---
    df["t_since_throw"] = (df["frame_id"] - 1) / fps
    df["time_remaining"] = (df["num_frames_output"] - df["frame_id"]) / fps

    # --- Smoothed velocity & acceleration from output x,y ---
    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

    df = (
        df.groupby(["game_id", "play_id", "nfl_id"], group_keys=False)
          .apply(lambda g: compute_smoothed_kinematics(g, fps=fps))
    )

    # Use smoothed series as our main kinematic features
    df["vx"] = df["vx_raw"]
    df["vy"] = df["vy_raw"]
    df["speed_bia"] = df["speed_smooth"]
    df["accel_bia"] = df["accel_smooth"]

    # (You can drop raw columns if you want to keep table tidy)
    df = df.drop(columns=["vx_raw", "vy_raw", "speed_raw", "speed_smooth", "accel_smooth"], errors="ignore")

    # --- Geometry to ball landing point ---
    dx_land = df["ball_land_x"] - df["x"]
    dy_land = df["ball_land_y"] - df["y"]
    df["dist_to_land"] = np.sqrt(dx_land**2 + dy_land**2)
    angle_land = np.arctan2(dy_land, dx_land)
    df["angle_to_land"] = np.degrees(angle_land)

    # --- Distance to Targeted Receiver (TR) at same frame ---
    tr_pos = (
        df[df["player_role"] == "Targeted Receiver"]
        [["game_id", "play_id", "frame_id", "x", "y"]]
        .rename(columns={"x": "tr_x", "y": "tr_y"})
    )

    df = df.merge(
        tr_pos,
        on=["game_id", "play_id", "frame_id"],
        how="left"
    )

    df["tr_x"] = df["tr_x"].fillna(df["ball_land_x"])
    df["tr_y"] = df["tr_y"].fillna(df["ball_land_y"])

    df["dist_to_TR"] = np.sqrt(
        (df["x"] - df["tr_x"])**2 + (df["y"] - df["tr_y"])**2
    )

    return df


# build enriched ball-in-air kinematics table
tracking_bia_kin = build_post_throw_kinematics(tracking_bia)
print(tracking_bia_kin.shape)
tracking_bia_kin.head()


tracking_bia_kin[["speed_bia", "accel_bia"]].describe()



def effective_speed_toward_land(speed_bia, heading_bia_deg, angle_to_land_deg):
    """
    Effective speed toward the ball-landing point:
        v_eff = s * cos(heading - angle_to_land)
    Negative components are clamped to zero.
    """
    heading = np.deg2rad(heading_bia_deg)
    theta   = np.deg2rad(angle_to_land_deg)
    cos_diff = np.cos(heading - theta)
    v_eff = speed_bia * cos_diff
    return np.maximum(v_eff, 0.0)



def time_to_cover_distance(d, v0, vmax, amax):
    """
    Time to cover distance d using a simple bang-bang profile:
      - accelerate from v0 up to vmax with acceleration amax
      - then cruise at vmax if needed

    All inputs can be scalars or 1D arrays of same shape.
    """
    d    = np.asarray(d, dtype=float)
    v0   = np.maximum(np.asarray(v0, dtype=float), 0.0)
    vmax = np.maximum(np.asarray(vmax, dtype=float), v0 + 1e-6)
    amax = np.maximum(np.asarray(amax, dtype=float), 1e-6)

    # time and distance to reach vmax from v0
    t1 = (vmax - v0) / amax
    d1 = v0 * t1 + 0.5 * amax * (t1 ** 2)

    t_reach = np.empty_like(d)
    mask = d <= d1  # reach before hitting vmax

    # case 1: never hit vmax, solve 0.5 a t^2 + v0 t - d = 0
    disc = v0**2 + 2 * amax * d
    t_reach[mask] = (-v0[mask] + np.sqrt(disc[mask])) / amax[mask]

    # case 2: accelerate to vmax, then cruise
    d2 = d - d1
    t2 = d2 / vmax
    t_reach[~mask] = t1[~mask] + t2[~mask]

    return t_reach



# ---- Global hyperparameters for CLI + NIT ---- Chosen from interative experimenting

FPS = 10  # frames per second 

ALPHA       = 2.5   # late-arrival penalty sharpness
BETA_ANGLE  = 0.35  # curvature penalty for angle misalignment
LAMBDA_THETA = 0.6  # how much NIT modifies defender weights

R_DEF_MAX   = 12.0  # defenders farther than this from landing point get ~0 weight

W_TR        = 1.5   # boost for targeted receiver
W_OFF_OTHER = 0.7   # downweight for other offensive players






def wrap_angle_deg(delta):
    """
    Wrap an angle in degrees into [-180, 180].
    """
    return (delta + 180.0) % 360.0 - 180.0



from tqdm import tqdm

def build_nit_training_data_bia(
    tracking_bia_kin,
    weeks=None,
    max_plays_per_week=None,
    max_frames_back=8,
    label_radius=1.5
):
    """
    Build NIT training data from ball-in-air tracking (tracking_bia_kin).

    For each play:
      - Label y_threat = 1 for the defender closest to landing point
        in the *final* ball-in-air frame, if within label_radius (yd).
      - All defenders in that play get features on some of the *last*
        frames (up to max_frames_back).
      - Those defender-frames are used to train a classifier θ_i(t).

    Returns:
      X_nit : np.ndarray of shape (N, n_features)
      y_nit : np.ndarray of shape (N,)
      nit_feat_cols : list of feature names
    """
    df = tracking_bia_kin.copy()

    if weeks is None:
        weeks = sorted(df["week"].unique())
    else:
        weeks = list(weeks)

    feat_rows = []

    for w in weeks:
        df_w = df[df["week"] == w]
        plays = df_w[["game_id", "play_id"]].drop_duplicates()

        if max_plays_per_week is not None:
            plays = plays.head(max_plays_per_week)

        for _, pr in tqdm(plays.iterrows(), total=len(plays), desc=f"NIT wk {w}"):
            g, p = pr.game_id, pr.play_id
            df_play = df_w[(df_w.game_id == g) & (df_w.play_id == p)]
            if df_play.empty:
                continue

            # defenders only
            df_def = df_play[df_play["player_role"] == "Defensive Coverage"].copy()
            if df_def.empty:
                continue

            # --- Label: who is nearest to landing at final frame? ---
            max_frame = df_def["frame_id"].max()
            df_last = df_def[df_def["frame_id"] == max_frame].copy()
            if df_last.empty:
                continue

            # distance to landing (we already have dist_to_land in tracking_bia_kin)
            df_last = df_last.assign(
                dist_last=df_last["dist_to_land"].values
            )

            # defenders within label_radius of landing point
            close = df_last[df_last["dist_last"] <= label_radius]
            if close.empty:
                # no one truly "at" the ball; label all 0
                df_def["y_threat"] = 0.0
            else:
                # closest defender at catch
                nearest_id = close.sort_values("dist_last").iloc[0]["nfl_id"]
                df_def["y_threat"] = (df_def["nfl_id"] == nearest_id).astype(float)

            # --- Features: only last max_frames_back frames ---
            frame_cut = max_frame - max_frames_back
            df_def = df_def[df_def["frame_id"] >= frame_cut].copy()
            if df_def.empty:
                continue

            # cos(theta) between movement heading and angle_to_land
            angle_diff = wrap_angle_deg(df_def["heading_bia"] - df_def["angle_to_land"])
            df_def["cos_theta"] = np.cos(np.deg2rad(angle_diff))

            feat_cols = [
                "dist_to_land",
                "dist_to_TR",
                "speed_bia",
                "accel_bia",
                "cos_theta",
                "time_remaining",
                "t_since_throw",
            ]

            # handle any NaNs with per-play median fill on these features
            df_def[feat_cols] = df_def[feat_cols].fillna(df_def[feat_cols].median())

            # keep rows
            feat_rows.append(df_def[["game_id", "play_id", "frame_id", "nfl_id"] + feat_cols + ["y_threat"]])

    if not feat_rows:
        return None, None, None

    data = pd.concat(feat_rows, ignore_index=True)

    nit_feat_cols = [
        "dist_to_land",
        "dist_to_TR",
        "speed_bia",
        "accel_bia",
        "cos_theta",
        "time_remaining",
        "t_since_throw",
    ]

    X = data[nit_feat_cols].values.astype("float32")
    y = data["y_threat"].values.astype("int32")

    return X, y, nit_feat_cols



from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---- Build NIT training data ----
X_nit, y_nit, nit_feat_cols = build_nit_training_data_bia(
    tracking_bia_kin,
    weeks=range(1, 13),         # weeks 1–12 train 
    max_plays_per_week=None,    # or some cap if too slow
    max_frames_back=8,
    label_radius=1.5,
)

print("NIT data shape:", X_nit.shape, "pos_rate:", y_nit.mean())

# ---- Simple negative downsampling to control imbalance ----
pos_idx = np.where(y_nit == 1)[0]
neg_idx = np.where(y_nit == 0)[0]

if len(pos_idx) > 0 and len(neg_idx) > 0:
    # keep all positives, and at most 5x negatives
    rng = np.random.default_rng(42)
    neg_keep = rng.choice(neg_idx, size=min(len(neg_idx), 5 * len(pos_idx)), replace=False)
    keep_idx = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep_idx)

    X_sub = X_nit[keep_idx]
    y_sub = y_nit[keep_idx]
else:
    X_sub, y_sub = X_nit, y_nit

print("After subsampling:", X_sub.shape, "pos_rate:", y_sub.mean())

# ---- Train/val split ----
X_train, X_val, y_train, y_val = train_test_split(
    X_sub, y_sub,
    test_size=0.2,
    random_state=42,
    stratify=y_sub if y_sub.sum() > 0 else None,
)

# ---- Pipeline: scaler + MLP ----
nit_model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=40,
        random_state=42,
    )
)

nit_model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

y_val_pred = nit_model.predict_proba(X_val)[:, 1]
print("NIT AUC (val):", roc_auc_score(y_val, y_val_pred))



# Works so retraining on all weeks 


# Build training data for ALL weeks 
X_all, y_all, nit_feat_cols = build_nit_training_data_bia(
    tracking_bia_kin,
    weeks=range(1, 19),       # set to full season weeks you have
    max_plays_per_week=None,
    max_frames_back=8,
    label_radius=1.5,
)

# Same imbalance handling 
pos_idx = np.where(y_all == 1)[0]
neg_idx = np.where(y_all == 0)[0]

if len(pos_idx) > 0 and len(neg_idx) > 0:
    rng = np.random.default_rng(42)
    neg_keep = rng.choice(neg_idx, size=min(len(neg_idx), 5 * len(pos_idx)), replace=False)
    keep_idx = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep_idx)
    X_fit, y_fit = X_all[keep_idx], y_all[keep_idx]
else:
    X_fit, y_fit = X_all, y_all

print("Refitting nit_model on ALL weeks:", X_fit.shape, "pos_rate:", y_fit.mean())

# Refit the SAME object name (overwrites the fitted weights inside)
nit_model.fit(X_fit, y_fit)

print("nit_model refit complete (all weeks).")



def nit_predict_threat_bia(df_def_frame):
    """
    Use trained nit_model to produce θ_j(t) for each defender in df_def_frame.
    df_def_frame is a *single-frame* subset of tracking_bia_kin for defenders.
    Returns np.array of shape (n_defenders,).
    """
    if df_def_frame.empty:
        return np.array([])

    # cos_theta again
    angle_diff = wrap_angle_deg(df_def_frame["heading_bia"] - df_def_frame["angle_to_land"])
    cos_theta  = np.cos(np.deg2rad(angle_diff))

    feat_mat = np.stack([
        df_def_frame["dist_to_land"].values.astype("float32"),
        df_def_frame["dist_to_TR"].values.astype("float32"),
        df_def_frame["speed_bia"].values.astype("float32"),
        df_def_frame["accel_bia"].values.astype("float32"),
        cos_theta.astype("float32"),
        df_def_frame["time_remaining"].values.astype("float32"),
        df_def_frame["t_since_throw"].values.astype("float32"),
    ], axis=1)

    proba = nit_model.predict_proba(feat_mat)[:, 1]  # threat probability
    return proba




def compute_cli_for_play_bia(df_play, fps=FPS,
                                 alpha=ALPHA,
                                 beta_angle=BETA_ANGLE,
                                 lambda_theta=LAMBDA_THETA,
                                 r_def_max=R_DEF_MAX):
    """
    Compute CLI(t) for a single play using **ball-in-air tracking** + NIT.

    df_play: subset of tracking_bia_kin for one (game_id, play_id)
             must include:
               - frame_id, player_role, player_position, player_side, nfl_id
               - speed_bia, heading_bia, dist_to_land, angle_to_land,
                 time_remaining, t_since_throw, dist_to_TR
    Uses:
      - empirical vmax, amax via get_vmax, get_amax
      - reaction_delay()
      - time_to_cover_distance()
      - nit_predict_threat_bia()

    Returns DataFrame with:
      frame_id, t_rel, P_off, P_def, CLI
    """
    if df_play.empty:
        return pd.DataFrame(columns=["frame_id", "t_rel", "P_off", "P_def", "CLI"])

    df_play = df_play.sort_values("frame_id").copy()
    frames  = df_play["frame_id"].unique()

    # relative time axis (0 at first ball-in-air frame)
    t0 = frames[0]
    t_rel_map = {f: (f - t0) / fps for f in frames}

    off_roles = ["Targeted Receiver", "Other Route Runner", "Passer"]
    rows = []

    for f in frames:
        df_f = df_play[df_play["frame_id"] == f].copy()
        if df_f.empty:
            continue

        # base geometric & timing quantities
        delta_t = df_f["time_remaining"].values      # remaining ball time
        dist    = df_f["dist_to_land"].values        # distance to landing

        # effective speed toward landing
        v0 = effective_speed_toward_land(
            df_f["speed_bia"].values,
            df_f["heading_bia"].values,
            df_f["angle_to_land"].values,
        )

        # player-specific vmax and amax
        vmax_arr = []
        amax_arr = []
        for _, r in df_f.iterrows():
            vmax_arr.append(get_vmax(r["nfl_id"], r["player_position"]))
            amax_arr.append(get_amax(r["nfl_id"], r["player_position"]))
        vmax_arr = np.asarray(vmax_arr, dtype=float)
        amax_arr = np.asarray(amax_arr, dtype=float)

        # reaction delays (role-based)
        delta_react = df_f.apply(reaction_delay, axis=1).values

        # movement time under accel→vmax→cruise
        t_move = time_to_cover_distance(dist, v0, vmax_arr, amax_arr)

        # total time-to-arrive
        tau = delta_react + t_move

        # ---- physics-based weight ----
        w_phys = np.exp(-alpha * np.maximum(tau - delta_t, 0.0))

        # ---- angle penalty (quadratic in normalized angle) ----
        angle_diff = wrap_angle_deg(df_f["heading_bia"] - df_f["angle_to_land"])
        angle_norm = np.abs(angle_diff) / 45.0     # 45° → 1, 90° → 2, etc.
        w_angle   = np.exp(-beta_angle * (angle_norm ** 2))

        w_base = w_phys * w_angle  # base weight before NIT + side weighting

        # Start with base weights
        df_f["w_base"] = w_base
        df_f["tau"]    = tau

        # ---- Distance gating for defenders far from landing ----
        far_def_mask = (
            (df_f["player_side"] == "Defense") &
            (df_f["dist_to_land"] > r_def_max)
        )
        df_f.loc[far_def_mask, "w_base"] = 0.0

        # ---- NIT threat gating for defenders ----
        is_def_cov = df_f["player_role"].eq("Defensive Coverage")
        df_def = df_f[is_def_cov].copy()

        theta = None
        if not df_def.empty:
            theta = nit_predict_threat_bia(df_def)
            theta = np.clip(theta, 0.0, 1.0)
            df_f.loc[is_def_cov, "theta"] = theta
        else:
            df_f["theta"] = np.nan

        # defenders not in coverage (e.g., pass rush) → low/default threat
        df_f["theta"] = df_f["theta"].fillna(0.1)

        # defender gating: w = w_base * [(1-λ) + λ * θ]
        gate_def = (1.0 - lambda_theta) + lambda_theta * df_f["theta"].values
        w_def_all = df_f["w_base"].values * gate_def

        # Initialize final weights as base
        w_final = df_f["w_base"].values.copy()

        # For defenders (Coverage or other defensive players):
        def_side_mask = df_f["player_side"].eq("Defense")
        w_final[def_side_mask.values] = w_def_all[def_side_mask.values]

        df_f["w_final"] = w_final

        # ---- Offensive role weighting (TR vs others) ----
        for idx, r in df_f.iterrows():
            if r["player_side"] != "Offense":
                continue
            if r["player_role"] == "Targeted Receiver":
                df_f.at[idx, "w_final"] *= W_TR
            elif r["player_role"] in off_roles:
                df_f.at[idx, "w_final"] *= W_OFF_OTHER
            else:
                # offensive players not involved in route (rare here) → slight downweight
                df_f.at[idx, "w_final"] *= 0.5

        # ---- Aggregate offense vs defense ----
        off_mask = df_f["player_side"].eq("Offense")
        def_mask = df_f["player_side"].eq("Defense")

        w_off = df_f.loc[off_mask, "w_final"].sum()
        w_def = df_f.loc[def_mask, "w_final"].sum()
        w_tot = w_off + w_def + 1e-9

        P_off = w_off / w_tot
        P_def = w_def / w_tot

        rows.append({
            "frame_id": f,
            "t_rel": t_rel_map[f],
            "P_off": P_off,
            "P_def": P_def,
            "CLI": P_off - P_def,
        })

    return pd.DataFrame(rows)



def build_empirical_vmax(tracking_bia_kin):
    """
    Compute empirical vmax per (nfl_id, position) from real ball-in-air speeds.
    Uses the 75th percentile of speed_bia for robustness.
    """
    g = tracking_bia_kin.groupby(["nfl_id", "player_position"])["speed_bia"]
    
    vmax_emp = g.quantile(0.75).reset_index()
    vmax_emp = vmax_emp.rename(columns={"speed_bia": "vmax_empirical"})
    
    # Clip to reasonable biomechanical limits
    vmax_emp["vmax_empirical"] = vmax_emp["vmax_empirical"].clip(lower=4.0, upper=11.0)
    
    return vmax_emp



vmax_table = build_empirical_vmax(tracking_bia_kin)
print(vmax_table.head())



def build_player_movement_capacity(tracking_bia_kin, fps=FPS,
                                   min_frames=30):
    """
    Build a per-player movement capacity model from ball-in-air tracking:
      - vmax_empirical: 75th percentile of speed_bia  (yd/s)
      - amax_empirical: 75th percentile of positive accel_bia (yd/s^2)

    Aggregated over all plays / games for each (nfl_id, player_position).
    """
    df = tracking_bia_kin.copy()

    # --- Filter out obviously garbage rows (if any) ---
    df = df[np.isfinite(df["speed_bia"]) & np.isfinite(df["accel_bia"])]

    # --- v_max: robust top speed per player ---
    speed_group = df.groupby(["nfl_id", "player_position"])
    vmax_emp = speed_group["speed_bia"].quantile(0.75).reset_index()
    vmax_emp = vmax_emp.rename(columns={"speed_bia": "vmax_empirical"})

    # Clip vmax to reasonable human bounds (≈ walk/jog → elite sprint)
    vmax_emp["vmax_empirical"] = vmax_emp["vmax_empirical"].clip(lower=4.0, upper=11.0)

    # --- a_max: robust top *positive* acceleration per player ---
    acc_pos = df[df["accel_bia"] > 0].copy()
    acc_group = acc_pos.groupby(["nfl_id", "player_position"])
    amax_emp = acc_group["accel_bia"].quantile(0.75).reset_index()
    amax_emp = amax_emp.rename(columns={"accel_bia": "amax_empirical"})

    # Clip amax to reasonable limits (elite bursts ~ 4–5 yd/s^2)
    amax_emp["amax_empirical"] = amax_emp["amax_empirical"].clip(lower=0.5, upper=5.0)

    # --- Merge into one capacity table ---
    cap = vmax_emp.merge(
        amax_emp,
        on=["nfl_id", "player_position"],
        how="outer"
    )

    # Drop players with too few frames if you want more robustness
    counts = df.groupby(["nfl_id", "player_position"]).size().reset_index(name="n_frames")
    cap = cap.merge(counts, on=["nfl_id", "player_position"], how="left")
    cap = cap[cap["n_frames"] >= min_frames].copy()

    return cap

player_capacity = build_player_movement_capacity(tracking_bia_kin)
player_capacity.head()



# Player-level lookups
vmax_by_player = {
    (int(row.nfl_id), str(row.player_position)): row.vmax_empirical
    for _, row in player_capacity.iterrows()
}

amax_by_player = {
    (int(row.nfl_id), str(row.player_position)): row.amax_empirical
    for _, row in player_capacity.iterrows()
}

# Position-level fallbacks (empirical medians)
vmax_by_pos = (
    player_capacity.groupby("player_position")["vmax_empirical"]
    .median()
    .to_dict()
)

amax_by_pos = (
    player_capacity.groupby("player_position")["amax_empirical"]
    .median()
    .to_dict()
)

def get_vmax(nfl_id, position):
    """
    Empirical vmax (yd/s) for this player.
    Fallbacks:
      (nfl_id, pos) → position median → global default
    """
    pos = str(position)
    key = (int(nfl_id), pos)
    if key in vmax_by_player:
        return vmax_by_player[key]
    if pos in vmax_by_pos:
        return vmax_by_pos[pos]
    return 8.0  # neutral fallback

def get_amax(nfl_id, position):
    """
    Empirical amax (yd/s^2) for this player.
    Fallbacks:
      (nfl_id, pos) → position median → global default
    """
    pos = str(position)
    key = (int(nfl_id), pos)
    if key in amax_by_player:
        return amax_by_player[key]
    if pos in amax_by_pos:
        return amax_by_pos[pos]
    return 3.0  # neutral fallback



def reaction_delay(row):
    """
    Reaction latency (sec) after ball release.
    Based on role:
      - Targeted WR anticipates ball flight → smallest delay
      - Other route runners → medium
      - Defensive coverage → slightly later
      - Others → default mid-range
    """
    role = str(row.get("player_role", "")).lower()

    if "targeted receiver" in role:
        return 0.15   # anticipatory
    if "other route runner" in role:
        return 0.18
    if "passer" in role:
        return 0.25
    if "defensive coverage" in role:
        return 0.25

    return 0.18


# pick a random play
pr = plays.iloc[0]  # plays = your per-play table with ball_land_x/y, etc.
g, p = pr.game_id, pr.play_id

df_play = tracking_bia_kin.query("game_id == @g and play_id == @p")

cli_curve = compute_cli_for_play_bia(df_play, fps=FPS)
cli_curve.head()



cli_curve.tail()


import matplotlib.pyplot as plt

plt.plot(cli_curve["t_rel"], cli_curve["CLI"])
plt.xlabel("Time Since Throw (s)")
plt.ylabel("CLI")
plt.title(f"CLI Curve for Play {g}-{p}")
plt.grid(True)
plt.show()


all_summaries = []
all_curves = []

weeks = sorted(tracking_bia_kin["week"].unique())

for w in weeks:
    df_w = tracking_bia_kin[tracking_bia_kin["week"] == w]
    plays = df_w[["game_id", "play_id"]].drop_duplicates()

    for _, pr in plays.iterrows():
        g, p = pr.game_id, pr.play_id

        df_play = df_w[(df_w.game_id == g) & (df_w.play_id == p)]
        if df_play.empty:
            continue

        # ---- compute CLI(t) for this play ----
        cli_df = compute_cli_for_play_bia(df_play)
        if cli_df.empty:
            continue

        # store curves
        cli_df["game_id"] = g
        cli_df["play_id"] = p
        cli_df["week"]    = w
        all_curves.append(cli_df)

        # ---- summary metrics ----
        peak_CLI = cli_df["CLI"].max()
        CDI      = peak_CLI - cli_df["CLI"].iloc[-1]

        # find VCT (first time CLI drops > 0.1 from peak)
        peak_idx = cli_df["CLI"].idxmax()
        peak_val = cli_df.loc[peak_idx, "CLI"]

        drop = peak_val - cli_df["CLI"]
        VCT_frame = None
        for i, d in drop.items():
            if d > 0.10:
                VCT_frame = cli_df.loc[i, "frame_id"]
                break

        pass_result = df_play["pass_result"].iloc[0]

        all_summaries.append({
            "game_id": g,
            "play_id": p,
            "week": w,
            "peak_CLI": peak_CLI,
            "CDI": CDI,
            "VCT_frame": VCT_frame,
            "pass_result": pass_result
        })

summ_all = pd.DataFrame(all_summaries)
curves_all = pd.concat(all_curves, ignore_index=True)

print("summ_all:", summ_all.shape)
print("curves_all:", curves_all.shape)



# --- Recompute initial_CLI, final_CLI, and a consistent CDI from curves_all ---

curves_sorted = curves_all.sort_values(["game_id", "play_id", "frame_id"])

agg = (
    curves_sorted
    .groupby(["game_id", "play_id"])["CLI"]
    .agg(
        initial_CLI     = "first",
        final_CLI       = "last",
        peak_CLI_curves = "max",
    )
    .reset_index()
)

# harmonize CDI as (peak - final)
agg["CDI_new"] = agg["peak_CLI_curves"] - agg["final_CLI"]

# merge back into summ_all
summ_all = (
    summ_all
    .drop(columns=[c for c in ["initial_CLI", "final_CLI", "CDI"]
                   if c in summ_all.columns], errors="ignore")
    .merge(
        agg[["game_id", "play_id", "initial_CLI", "final_CLI", "CDI_new"]],
        on=["game_id", "play_id"],
        how="left"
    )
)

summ_all = summ_all.rename(columns={"CDI_new": "CDI"})
print(summ_all.columns)



import numpy as np

summ_all = summ_all.copy()

FPS      = 10
thresh   = 0.10   # "never really viable" threshold for peak_CLI
neg_thr  = -0.05  # "definitely defense control" threshold for final_CLI

summ_all["type"] = np.nan

# 1) Completions
summ_all.loc[
    (summ_all.pass_result == "C"),
    "type"
] = "completion"

# 2) Defensive-collapse incompletions (I)
summ_all.loc[
    (summ_all.pass_result == "I") &
    (summ_all["peak_CLI"] > thresh) &
    (summ_all["final_CLI"] < neg_thr) &
    (summ_all["VCT_frame"].notna()),
    "type"
] = "def_collapse_INC"

# 3) Defensive-collapse interceptions (IN)
summ_all.loc[
    (summ_all.pass_result == "IN") &
    (summ_all["peak_CLI"] > thresh) &
    (summ_all["final_CLI"] < neg_thr) &
    (summ_all["VCT_frame"].notna()),
    "type"
] = "def_collapse_INT"

# 4) Never-viable incompletions: offense never really had a window
summ_all.loc[
    (summ_all.pass_result == "I") &
    (summ_all["peak_CLI"] < thresh),
    "type"
] = "never_viable_INC"

# 5) Offense-controlled incompletions:
#    remaining incompletions that aren't collapse or never-viable
summ_all.loc[
    (summ_all.pass_result == "I") &
    (summ_all["type"].isna()),
    "type"
] = "off_control_INC"

print("Type counts:\n", summ_all["type"].value_counts(dropna=False))





import matplotlib.pyplot as plt
import pandas as pd

def pick_example_play(df, play_type, sort_col, ascending):
    """
    Pick one 'representative' play of a given type,
    using sort_col to choose an extreme.
    """
    cand = df[df["type"] == play_type].dropna(subset=[sort_col])
    if cand.empty:
        print(f"[WARN] No plays of type {play_type}")
        return None
    row = cand.sort_values(sort_col, ascending=ascending).iloc[0]
    return (play_type, row.game_id, row.play_id)

example_plays = []

# 1) Completion: most positive final_CLI
ep = pick_example_play(summ_all, "completion", "final_CLI", ascending=False)
if ep: example_plays.append(ep)

# 2) Def-collapsed incompletion: most negative final_CLI
ep = pick_example_play(summ_all, "def_collapse_INC", "final_CLI", ascending=True)
if ep: example_plays.append(ep)

# 3) Def-collapsed interception: most negative final_CLI
ep = pick_example_play(summ_all, "def_collapse_INT", "final_CLI", ascending=True)
if ep: example_plays.append(ep)

# 4) Offense-controlled incompletion: largest CDI
ep = pick_example_play(summ_all, "off_control_INC", "CDI", ascending=False)
if ep: example_plays.append(ep)

# 5) Never-viable incompletion: smallest peak_CLI
ep = pick_example_play(summ_all, "never_viable_INC", "peak_CLI", ascending=True)
if ep: example_plays.append(ep)

print("\nExample plays selected:")
for t, g, p in example_plays:
    print(f"  {t}: game_id={g}, play_id={p}")

# ---- Plot CLI(t) for those 5 plays ----
plt.figure(figsize=(9, 5))

for t, g, p in example_plays:
    curve = (
        curves_all[(curves_all.game_id == g) & (curves_all.play_id == p)]
        .sort_values("t_rel")
    )
    if curve.empty:
        continue
    plt.plot(curve["t_rel"], curve["CLI"], label=t)

plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Time since throw (s)")
plt.ylabel("CLI = P_off - P_def")
plt.title("CLI(t) for one **representative** play from each outcome type")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------
# Build normalized CLI(tau) grid for each play
# tau in [0,1], 0 = throw, 1 = arrival
# --------------------------------------------
N_GRID = 30
tau_grid = np.linspace(0.0, 1.0, N_GRID)

# Fast index for type, pass_result
summ_idx = (
    summ_all
    .set_index(["game_id", "play_id"])
    [["type", "pass_result"]]
)

norm_rows = []

for (g, p), grp in curves_all.groupby(["game_id", "play_id"]):
    grp = grp.sort_values("t_rel")
    t = grp["t_rel"].values
    y = grp["CLI"].values
    if len(t) < 2:
        continue

    # normalize time to [0,1]
    t0, t1 = t[0], t[-1]
    span = max(t1 - t0, 1e-6)
    tau = (t - t0) / span

    # interpolate CLI onto common tau_grid
    y_interp = np.interp(tau_grid, tau, y)

    # get type
    if (g, p) not in summ_idx.index:
        continue
    play_type = summ_idx.loc[(g, p), "type"]
    if pd.isna(play_type):
        continue

    norm_rows.append(
        pd.DataFrame({
            "game_id": g,
            "play_id": p,
            "type": play_type,
            "tau": tau_grid,
            "CLI": y_interp,
        })
    )

if not norm_rows:
    raise ValueError("No normalized curves were constructed.")

curves_norm = pd.concat(norm_rows, ignore_index=True)
print("curves_norm:", curves_norm.shape)



# Aggregate mean and 25–75% band per type, per tau
agg_grid = (
    curves_norm
    .groupby(["type", "tau"])["CLI"]
    .agg(
        mean = "mean",
        q25  = lambda v: v.quantile(0.25),
        q75  = lambda v: v.quantile(0.75),
    )
    .reset_index()
)

plot_types2 = [
    "completion",
    "def_collapse_INC",
    "def_collapse_INT",
    "off_control_INC",
    "never_viable_INC",
]

plt.figure(figsize=(9, 5))

for t_name in plot_types2:
    sub = agg_grid[agg_grid["type"] == t_name]
    if sub.empty:
        continue

    tau = sub["tau"].values
    m   = sub["mean"].values
    q25 = sub["q25"].values
    q75 = sub["q75"].values

    plt.plot(tau, m, label=t_name)
    plt.fill_between(tau, q25, q75, alpha=0.15)

plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Normalized time in air (0 = throw, 1 = arrival)")
plt.ylabel("CLI = P_off - P_def")
plt.title("Average CLI(t) shape by outcome type\n(mean ± 25–75% band)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()




# --------------------------------------------
# Add final_CLI, VCR, VCT_sec using curves_all
# --- Recompute final_CLI, CDI, VCT_sec, VCR, t_peak, t_final from curves_all ---

summary_extra = []
TAU_DROP = 0.10   # same threshold as in VCT definition

for (g, p), grp in curves_all.groupby(["game_id", "play_id"]):
    grp = grp.sort_values("t_rel")
    if grp.empty:
        continue

    # peak
    peak_idx  = grp["CLI"].idxmax()
    peak_CLI  = grp.loc[peak_idx, "CLI"]
    t_peak    = grp.loc[peak_idx, "t_rel"]

    # final
    last      = grp.iloc[-1]
    final_CLI = last["CLI"]
    t_final   = last["t_rel"]

    # CDI = peak - final
    CDI = peak_CLI - final_CLI

    # VCT_sec
    VCT_sec = np.nan
    drop = peak_CLI - grp["CLI"]
    after_peak = grp["t_rel"] >= t_peak
    mask_collapse = (drop > TAU_DROP) & after_peak
    if mask_collapse.any():
        idx_vct = grp.index[mask_collapse][0]
        VCT_sec = grp.loc[idx_vct, "t_rel"] - t_peak


    # VCR
    dt  = max(t_final - t_peak, 1e-6)
    VCR = (peak_CLI - final_CLI) / dt

    summary_extra.append({
        "game_id": g,
        "play_id": p,
        "final_CLI": final_CLI,
        "CDI": CDI,
        "VCT_sec": VCT_sec,
        "VCR": VCR,
        "t_peak": t_peak,
        "t_final": t_final,
    })

summary_extra = pd.DataFrame(summary_extra)

# ---- clean merge: drop old copies then add new ones ----
cols_to_drop = ["final_CLI", "CDI", "VCT_sec", "VCR", "t_peak", "t_final"]
summ_all = summ_all.drop(
    columns=[c for c in cols_to_drop if c in summ_all.columns],
    errors="ignore"
)

summ_all = summ_all.merge(
    summary_extra,
    on=["game_id", "play_id"],
    how="left"
)

summ_all.head()


metrics = ["peak_CLI", "CDI", "final_CLI", "VCT_sec", "VCR"]

(
    summ_all
    .groupby("pass_result")[metrics]
    .describe()
    .round(3)
)



label_map = {"C": "Completion", "I": "Incomplete", "IN": "Interception"}

for col in metrics:
    plt.figure(figsize=(7, 4))
    for pr in ["C", "I", "IN"]:
        vals = summ_all.loc[summ_all["pass_result"] == pr, col].dropna()
        if vals.empty:
            continue
        plt.hist(
            vals,
            bins=40,
            density=True,
            alpha=0.35,
            label=label_map.get(pr, pr)
        )
    plt.title(f"Distribution of {col} by pass_result")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



completions = summ_all[summ_all["pass_result"] == "C"]

num_total = len(completions)
num_negative = (completions["final_CLI"] < 0).sum()
pct_negative = num_negative / num_total * 100

print(f"Total completions: {num_total}")
print(f"Completions with negative final_CLI: {num_negative} ({pct_negative:.2f}%)")



# After you've built summ_all from all_summaries
# Make sure supplementary has 'game_id', 'play_id', 'pass_length'

summ_all = summ_all.merge(
    df_supp[["game_id", "play_id", "pass_length"]],
    on=["game_id", "play_id"],
    how="left"
)



# -------------------------------------------
# Sanity 2: does CLI carry signal beyond a simple baseline?
# (same as yours, but returns probs for ROC + single consistent split)
# -------------------------------------------
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

baseline_feat = "pass_length"   # change if needed

cols_needed = ["pass_result", baseline_feat, "peak_CLI", "CDI", "final_CLI"]
df_model = summ_all.dropna(subset=cols_needed).copy()

df_model["y_complete"] = (df_model["pass_result"] == "C").astype(int)

X_base = df_model[[baseline_feat]].values
X_cli  = df_model[["peak_CLI", "CDI", "final_CLI"]].values
X_both = df_model[[baseline_feat, "peak_CLI", "CDI", "final_CLI"]].values
y      = df_model["y_complete"].values

# --- one split so everything lines up perfectly
idx = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=42, stratify=y)

Xb_tr, Xb_te = X_base[idx_tr], X_base[idx_te]
Xc_tr, Xc_te = X_cli[idx_tr],  X_cli[idx_te]
Xa_tr, Xa_te = X_both[idx_tr], X_both[idx_te]
y_tr, y_te   = y[idx_tr],      y[idx_te]

def fit_auc_and_probs(X_tr, X_te, y_tr, y_te):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, p), p

auc_base, p_base = fit_auc_and_probs(Xb_tr, Xb_te, y_tr, y_te)
auc_cli,  _      = fit_auc_and_probs(Xc_tr, Xc_te, y_tr, y_te)   # optional
auc_both, p_both = fit_auc_and_probs(Xa_tr, Xa_te, y_tr, y_te)

print(f"Baseline (only {baseline_feat}) AUC: {auc_base:.3f}")
print(f"CLI-only AUC:                        {auc_cli:.3f}")
print(f"Baseline + CLI AUC:                  {auc_both:.3f}")



import numpy as np
import pandas as pd

# -----------------------------------
# Build normalized CLI grid per play
# tau in [0,1], 0 = throw, 1 = arrival
# -----------------------------------
N_GRID = 30
tau_grid = np.linspace(0.0, 1.0, N_GRID)

# Fast lookup for type / pass_result
summ_idx = (
    summ_all
    .set_index(["game_id", "play_id"])
    [["type", "pass_result"]]
)

rows = []

for (g, p), grp in curves_all.groupby(["game_id", "play_id"]):
    grp = grp.sort_values("t_rel")
    if grp.empty:
        continue

    t = grp["t_rel"].values
    cli = grp["CLI"].values

    # Ignore degenerate case (1 frame)
    if len(t) < 2 or (t[-1] - t[0]) < 1e-6:
        continue

    # Normalize time to [0,1]
    t_norm = (t - t[0]) / (t[-1] - t[0])

    # Interpolate CLI onto uniform grid
    cli_grid = np.interp(tau_grid, t_norm, cli)

    # Look up type / outcome
    if (g, p) not in summ_idx.index:
        continue
    t_type       = summ_idx.loc[(g, p), "type"]
    pass_result  = summ_idx.loc[(g, p), "pass_result"]

    for tau, v in zip(tau_grid, cli_grid):
        rows.append({
            "game_id": g,
            "play_id": p,
            "tau": tau,          # normalized time in air
            "CLI": v,
            "type": t_type,
            "pass_result": pass_result
        })

cli_grid_df = pd.DataFrame(rows)
print(cli_grid_df.head())



import matplotlib.pyplot as plt

sub_types = ["completion", "def_collapse_INC", "never_viable_INC", "off_control_INC"]

plt.figure(figsize=(8, 5))

for t_type in sub_types:
    df_t = cli_grid_df[cli_grid_df["type"] == t_type]
    if df_t.empty:
        continue

    grouped = df_t.groupby("tau")["CLI"]
    mean = grouped.mean()
    q25  = grouped.quantile(0.25)
    q75  = grouped.quantile(0.75)

    tau_vals = mean.index.values
    plt.plot(tau_vals, mean.values, label=t_type)
    plt.fill_between(tau_vals, q25.values, q75.values, alpha=0.15)

plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Normalized time in air (0 = throw, 1 = arrival)")
plt.ylabel("CLI = P_off - P_def")
plt.title("CLI(t): completions vs different types of incompletions\n(mean ± 25–75% band)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# -----------------------------------
# Compute per-play collapse rate d(CLI)/d tau on the grid
# -----------------------------------
rows_deriv = []

for (g, p), grp in cli_grid_df.groupby(["game_id", "play_id"]):
    grp = grp.sort_values("tau")
    cli_vals = grp["CLI"].values
    tau_vals = grp["tau"].values

    if len(cli_vals) < 2:
        continue

    # finite difference
    d_cli = np.diff(cli_vals)
    d_tau = np.diff(tau_vals)
    deriv = d_cli / d_tau
    deriv = np.concatenate([[deriv[0]], deriv])  # copy first slope forward

    for tau, dval, t_type, pr in zip(
        tau_vals, deriv, grp["type"].values, grp["pass_result"].values
    ):
        rows_deriv.append({
            "game_id": g,
            "play_id": p,
            "tau": tau,
            "CollapseRate": dval,
            "type": t_type,
            "pass_result": pr
        })

cli_deriv_df = pd.DataFrame(rows_deriv)
print(cli_deriv_df.head())



plt.figure(figsize=(8, 5))

label_map = {"C": "Completion", "I": "Incompletion", "IN": "Interception"}
order = ["C", "I", "IN"]

for pr in order:
    df_pr = cli_deriv_df[cli_deriv_df["pass_result"] == pr]
    if df_pr.empty:
        continue

    grouped = df_pr.groupby("tau")["CollapseRate"]
    mean = grouped.mean()
    q25  = grouped.quantile(0.25)
    q75  = grouped.quantile(0.75)

    tau_vals = mean.index.values
    plt.plot(tau_vals, mean.values, label=label_map.get(pr, pr))
    plt.fill_between(tau_vals, q25.values, q75.values, alpha=0.15)

plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Normalized time in air (0 = throw, 1 = arrival)")
plt.ylabel("CollapseRate(t) = d CLI / d t (normalized)")
plt.title("How fast leverage changes during ball flight\n(mean ± 25–75% band)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



rows_peak = []

for (g, p), grp in cli_grid_df.groupby(["game_id", "play_id"]):
    grp = grp.sort_values("tau")
    cli_vals = grp["CLI"].values
    tau_vals = grp["tau"].values
    if len(cli_vals) == 0:
        continue

    # index of peak CLI
    k_peak = np.argmax(cli_vals)
    tau_peak = tau_vals[k_peak]

    tau_align = tau_vals - tau_peak  # 0 = moment of max CLI

    t_type      = grp["type"].iloc[0]
    pass_result = grp["pass_result"].iloc[0]

    for ta, v in zip(tau_align, cli_vals):
        rows_peak.append({
            "game_id": g,
            "play_id": p,
            "tau_align": ta,
            "CLI": v,
            "type": t_type,
            "pass_result": pass_result
        })

cli_peak_df = pd.DataFrame(rows_peak)

# For plotting, restrict to a reasonable window around the peak
mask_window = (cli_peak_df["tau_align"] >= -0.6) & (cli_peak_df["tau_align"] <= 0.4)
cli_peak_df = cli_peak_df[mask_window]



plt.figure(figsize=(8, 5))

for pr, label in [("C", "Completion"), ("I", "Incompletion"), ("IN", "Interception")]:
    df_pr = cli_peak_df[cli_peak_df["pass_result"] == pr]
    if df_pr.empty:
        continue

    grouped = df_pr.groupby("tau_align")["CLI"]
    mean = grouped.mean()
    q25  = grouped.quantile(0.25)
    q75  = grouped.quantile(0.75)

    x = mean.index.values
    plt.plot(x, mean.values, label=label)
    plt.fill_between(x, q25.values, q75.values, alpha=0.15)

plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
plt.axhline(0.0, color="gray", linestyle=":", linewidth=1)
plt.xlabel("Time relative to peak CLI (0 = peak)")
plt.ylabel("CLI = P_off - P_def")
plt.title("What happens around the peak leverage moment?")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Parameters you control
# =========================
DELTA = 0.10           # window is "alive" if CLI >= peak - DELTA
AFTER_PEAK_ONLY = True # collapse only counts after the peak moment
N_BOOT = 250           # 0 to disable bootstrap bands; 200-500 is fine
BAND_Q = (0.10, 0.90)  # 10-90% band
TEAM_MIN_N = 250       # min plays for team curves (adjust)

# =========================
# 1) Build tau and per-play peak
# =========================
dfc = curves_all[["game_id","play_id","t_rel","CLI"]].dropna().copy()
dfc = dfc.sort_values(["game_id","play_id","t_rel"])

tmin = dfc.groupby(["game_id","play_id"])["t_rel"].transform("min")
tmax = dfc.groupby(["game_id","play_id"])["t_rel"].transform("max")
span = (tmax - tmin).clip(lower=1e-6)
dfc["tau"] = (dfc["t_rel"] - tmin) / span

# peak CLI + tau of peak (first occurrence)
g = dfc.groupby(["game_id","play_id"], sort=False)
dfc["peak_cli"] = g["CLI"].transform("max")
peak_idx = g["CLI"].idxmax()
peak_tau_map = dfc.loc[peak_idx, ["game_id","play_id","tau"]].rename(columns={"tau":"peak_tau"})
dfc = dfc.merge(peak_tau_map, on=["game_id","play_id"], how="left")

# =========================
# 2) Merge labels you want to compare by
# =========================
lbl_cols = ["game_id","play_id","type","pass_result"]
labels = summ_all[lbl_cols].drop_duplicates().copy()

# add coverage/team fields from df_supp (these exist in your df_supp)
extra_cols = ["game_id","play_id","defensive_team","team_coverage_man_zone","team_coverage_type"]
extra = df_supp[extra_cols].drop_duplicates().copy()

labels = labels.merge(extra, on=["game_id","play_id"], how="left")
dfc = dfc.merge(labels, on=["game_id","play_id"], how="left")

# (Optional) Make MAN/ZONE cleaner labels
if "team_coverage_man_zone" in dfc.columns:
    dfc["man_zone"] = dfc["team_coverage_man_zone"].astype(str).str.upper().str.strip()
else:
    dfc["man_zone"] = np.nan

# =========================
# 3) Define event time per play (KM)
#    Event = first tau when CLI < (peak - DELTA)
#    Censor = never fails by tau=1
# =========================
thr = dfc["peak_cli"] - DELTA

dfc_evt = dfc[dfc["tau"] >= dfc["peak_tau"]].copy() if AFTER_PEAK_ONLY else dfc.copy()
dfc_evt["failed"] = (dfc_evt["CLI"] < thr.loc[dfc_evt.index]).astype(int)

first_fail = (
    dfc_evt[dfc_evt["failed"] == 1]
    .groupby(["game_id","play_id"])["tau"]
    .min()
    .rename("t_event")
    .reset_index()
)

# one row per play for KM
plays = dfc[[
    "game_id","play_id","type","pass_result","defensive_team",
    "team_coverage_type","man_zone"
]].drop_duplicates().copy()

plays = plays.merge(first_fail, on=["game_id","play_id"], how="left")
plays["event"] = (~plays["t_event"].isna()).astype(int)
plays["t"] = plays["t_event"].fillna(1.0)      # censored at end of flight
plays = plays.dropna(subset=["type"])          # require a grouping label

# =========================
# 4) Kaplan–Meier estimator + median survival
# =========================
def km_curve(times, events):
    """
    times: event/censor times in [0,1]
    events: 1 if event occurred, 0 if censored
    returns df with step function points (t, S)
    """
    df = pd.DataFrame({"t": np.asarray(times), "e": np.asarray(events)}).sort_values("t")
    uniq_t = np.sort(df.loc[df["e"] == 1, "t"].unique())

    S = 1.0
    out_t, out_S = [0.0], [1.0]

    for t in uniq_t:
        at_risk = (df["t"] >= t).sum()
        d = ((df["t"] == t) & (df["e"] == 1)).sum()
        if at_risk > 0:
            S *= (1.0 - d / at_risk)
        out_t.append(float(t))
        out_S.append(float(S))

    if out_t[-1] < 1.0:
        out_t.append(1.0)
        out_S.append(out_S[-1])

    return pd.DataFrame({"t": out_t, "S": out_S})

def km_median(km_df):
    hit = km_df[km_df["S"] <= 0.5]
    return np.nan if hit.empty else float(hit["t"].iloc[0])

def bootstrap_km_band(df_group, n_boot=200, q=(0.1,0.9), grid=np.linspace(0,1,101)):
    if n_boot <= 0 or len(df_group) < 50:
        return None

    T = df_group["t"].values
    E = df_group["event"].values
    n = len(df_group)

    S_mat = np.zeros((n_boot, len(grid)), dtype=float)
    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        km = km_curve(T[idx], E[idx])

        # Evaluate step function on grid
        s = []
        for tg in grid:
            s.append(km.loc[km["t"] <= tg, "S"].iloc[-1])
        S_mat[b,:] = np.array(s)

    lo = np.quantile(S_mat, q[0], axis=0)
    hi = np.quantile(S_mat, q[1], axis=0)
    return grid, lo, hi

# =========================
# 5) Plot helper
# =========================
def plot_km_compare(df, group_col, groups=None, title=None, min_n=200):
    use = df.dropna(subset=[group_col]).copy()

    # keep only sufficiently large groups
    counts = use[group_col].value_counts()
    keep = counts[counts >= min_n].index
    use = use[use[group_col].isin(keep)]

    if groups is None:
        groups = list(use[group_col].value_counts().index)

    plt.figure(figsize=(10,6))

    for gname in groups:
        dfg = use[use[group_col] == gname]
        if len(dfg) < min_n:
            continue

        km = km_curve(dfg["t"].values, dfg["event"].values)
        med = km_median(km)

        plt.step(km["t"], km["S"], where="post",
                 label=f"{gname} (n={len(dfg)}, med={med:.2f})")

        band = bootstrap_km_band(dfg, n_boot=N_BOOT, q=BAND_Q)
        if band is not None:
            grid, lo, hi = band
            plt.fill_between(grid, lo, hi, step="post", alpha=0.12)

    plt.xlabel("Normalized time in air (τ)")
    plt.ylabel(f"Survival = P(window alive)  (alive if CLI ≥ peak−{DELTA:.2f})")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.title(title or f"Catch window survival (Kaplan–Meier) by {group_col}")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()

# =========================
# 6) A) by your existing outcome bucket (type)
# =========================
plot_km_compare(
    plays,
    group_col="type",
    groups=["completion","def_collapse_INC","off_control_INC","never_viable_INC"],
    title="Catch window survival (Kaplan–Meier) by outcome type",
    min_n=200
)

# =========================
# 7) B) by MAN vs ZONE (team-level coverage)
# =========================
plot_km_compare(
    plays,
    group_col="man_zone",
    groups=None,
    title="Catch window survival (Kaplan–Meier) by team coverage: MAN vs ZONE",
    min_n=500
)

# =========================
# 8) C) by coverage family (team_coverage_type)
# =========================
plot_km_compare(
    plays,
    group_col="team_coverage_type",
    groups=None,   # auto-select frequent ones
    title="Catch window survival (Kaplan–Meier) by team coverage type",
    min_n=500
)

# =========================
# 9) D) by defensive team (optional, heavier)
# =========================
plot_km_compare(
    plays,
    group_col="defensive_team",
    groups=None,
    title="Catch window survival (Kaplan–Meier) by defensive team",
    min_n=TEAM_MIN_N
)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Build per-play tau-grid curve (if you don't already have cli_grid_df)
N_GRID = 31
tau_grid = np.linspace(0, 1, N_GRID)

dfc = curves_all[["game_id","play_id","t_rel","CLI"]].dropna().copy()
dfc = dfc.sort_values(["game_id","play_id","t_rel"])

rows = []
for (g,p), grp in dfc.groupby(["game_id","play_id"]):
    t = grp["t_rel"].values
    y = grp["CLI"].values
    if len(t) < 2 or (t[-1]-t[0]) < 1e-6:
        continue
    tau = (t - t[0])/(t[-1]-t[0])
    y_grid = np.interp(tau_grid, tau, y)
    rows.append(pd.DataFrame({"game_id":g,"play_id":p,"tau":tau_grid,"CLI":y_grid}))

cli_grid = pd.concat(rows, ignore_index=True)

# merge coverage + keep a clean outcome subset (optional)
cov = df_supp[["game_id","play_id","team_coverage_man_zone","team_coverage_type","pass_result"]].drop_duplicates()
cli_grid = cli_grid.merge(cov, on=["game_id","play_id"], how="left")

# choose one split: MAN vs ZONE
cli_grid = cli_grid.dropna(subset=["team_coverage_man_zone"])
cli_grid["cov"] = cli_grid["team_coverage_man_zone"].replace({
    "MAN_COVERAGE":"MAN",
    "ZONE_COVERAGE":"ZONE"
})

def bootstrap_band(df_sub, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    plays = df_sub[["game_id","play_id"]].drop_duplicates().values
    if len(plays) < 50:
        return None

    # pivot to [n_plays x n_tau]
    mat = (
        df_sub.pivot_table(index=["game_id","play_id"], columns="tau", values="CLI")
              .reindex(columns=tau_grid)
              .values
    )
    mat = np.nan_to_num(mat, nan=np.nanmean(mat))

    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, mat.shape[0], size=mat.shape[0])
        boot_means.append(np.nanmean(mat[idx], axis=0))
    boot_means = np.vstack(boot_means)

    m = np.nanmean(mat, axis=0)
    lo = np.quantile(boot_means, 0.10, axis=0)
    hi = np.quantile(boot_means, 0.90, axis=0)
    return m, lo, hi

plt.figure(figsize=(9,5))

for cov_name in ["MAN","ZONE"]:
    sub = cli_grid[(cli_grid["cov"] == cov_name) & (cli_grid["pass_result"].isin(["C","I","IN"]))]
    out = bootstrap_band(sub, n_boot=250)
    if out is None:
        continue
    m, lo, hi = out
    plt.plot(tau_grid, m, label=cov_name)
    plt.fill_between(tau_grid, lo, hi, alpha=0.15)

plt.axhline(0, linestyle="--", alpha=0.4)
plt.xlabel("tau")
plt.ylabel("Mean CLI(tau)")
plt.title("CLI shape by coverage (bootstrap 10–90% band)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- map targeted receiver nfl_id per play (from throw_state; safest) ---
tr_map = (
    throw_state[throw_state["player_role"] == "Targeted Receiver"]
    [["game_id","play_id","nfl_id","player_position"]]
    .drop_duplicates()
    .rename(columns={"nfl_id":"tr_nfl_id", "player_position":"tr_pos"})
)

# --- primary defender: closest Defensive Coverage to landing at final frame (using tracking_bia_kin) ---
df_def = tracking_bia_kin[tracking_bia_kin["player_role"] == "Defensive Coverage"].copy()

# last frame per play
last_frame = df_def.groupby(["game_id","play_id"])["frame_id"].transform("max")
df_last = df_def[df_def["frame_id"] == last_frame].copy()

# pick closest defender
idx = df_last.groupby(["game_id","play_id"])["dist_to_land"].idxmin()
prim = (
    df_last.loc[idx, ["game_id","play_id","nfl_id","player_position","dist_to_land"]]
    .rename(columns={"nfl_id":"def_nfl_id","player_position":"def_pos","dist_to_land":"def_dist_at_arrival"})
)

# --- merge with your per-play summary metrics + team/coverage context ---
play_ctx = df_supp[[
    "game_id","play_id","defensive_team","possession_team",
    "team_coverage_man_zone","team_coverage_type","pass_result","pass_length"
]].drop_duplicates()

df_match = (
    summ_all.merge(prim, on=["game_id","play_id"], how="left")
            .merge(tr_map, on=["game_id","play_id"], how="left")
            .merge(play_ctx, on=["game_id","play_id"], how="left")
)

# keep plays where we found a primary defender + TR
df_match = df_match.dropna(subset=["def_nfl_id","tr_nfl_id","final_CLI","peak_CLI"])

# defender leaderboard
MIN_TARGETS = 40
def_board = (
    df_match.groupby(["def_nfl_id","def_pos"])
            .agg(
                n_targets=("play_id","count"),
                mean_final_CLI=("final_CLI","mean"),
                mean_late_drop=("CDI","mean"),
                mean_peak=("peak_CLI","mean"),
                share_def_collapse=("type", lambda s: np.mean(s=="def_collapse_INC")),
            )
            .reset_index()
)
def_board = def_board[def_board["n_targets"] >= MIN_TARGETS].copy()

# "Closer score": big late drop + low final CLI
def_board["closer_score"] = def_board["mean_late_drop"] - def_board["mean_final_CLI"]

top = def_board.sort_values("closer_score", ascending=False).head(15)

plt.figure(figsize=(9,5))
plt.barh(top["def_nfl_id"].astype(int).astype(str), top["closer_score"])
plt.xlabel("Closer score = mean(CDI) − mean(final_CLI)")
plt.ylabel("Primary defender nfl_id")
plt.title(f"Top 'window closers' (min {MIN_TARGETS} targets)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()



print("df_supp columns:")
print(sorted(df_supp.columns.tolist()))
print("summ_all columns:")
print(sorted(summ_all.columns.tolist()))



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Merge defensive team onto per-play summaries
team_play = summ_all.merge(
    df_supp[["game_id","play_id","defensive_team"]],
    on=["game_id","play_id"],
    how="left"
).dropna(subset=["defensive_team"])

# Keep only what we need
team_play = team_play[[
    "defensive_team",
    "peak_CLI",
    "final_CLI"
]].dropna()



# Filter to teams with enough data
MIN_PLAYS = 200
counts = team_play["defensive_team"].value_counts()
keep_teams = counts[counts >= MIN_PLAYS].index
tp = team_play[team_play["defensive_team"].isin(keep_teams)]

# Sort teams by median final CLI (strong → weak)
order = (
    tp.groupby("defensive_team")["final_CLI"]
      .median()
      .sort_values()
      .index
)

# League medians
league_peak = tp["peak_CLI"].median()
league_final = tp["final_CLI"].median()

plt.figure(figsize=(12,6))

# Peak CLI violins
parts = plt.violinplot(
    [tp[tp["defensive_team"] == t]["peak_CLI"] for t in order],
    positions=np.arange(len(order)) - 0.15,
    widths=0.25,
    showextrema=False
)

for pc in parts["bodies"]:
    pc.set_alpha(0.4)

# Final CLI violins
parts2 = plt.violinplot(
    [tp[tp["defensive_team"] == t]["final_CLI"] for t in order],
    positions=np.arange(len(order)) + 0.15,
    widths=0.25,
    showextrema=False
)

for pc in parts2["bodies"]:
    pc.set_alpha(0.7)

# League medians
plt.axhline(league_peak, linestyle="--", alpha=0.4, label="League median (peak)")
plt.axhline(league_final, linestyle=":", alpha=0.6, label="League median (final)")

plt.xticks(np.arange(len(order)), order, rotation=90)
plt.ylabel("CLI")
plt.title("Defensive team profiles as distributions (peak vs final catch viability)")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()



def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

# Pick a few illustrative teams (top, middle, bottom)
example_teams = list(order[:2]) + list(order[len(order)//2:len(order)//2+2]) + list(order[-2:])

plt.figure(figsize=(8,6))

for t in example_teams:
    x, y = ecdf(tp[tp["defensive_team"] == t]["peak_CLI"])
    plt.plot(x, y, label=t)

plt.axvline(league_peak, linestyle="--", alpha=0.4, label="League median peak CLI")
plt.xlabel("Peak CLI allowed")
plt.ylabel("ECDF")
plt.title("Tail risk: how often do defenses allow extreme leverage?")
plt.legend(fontsize=9)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()



# --- Build per-play peak CLI with defensive team ---
team_peaks = (
    summ_all[["game_id","play_id","peak_CLI"]]
    .merge(
        df_supp[["game_id","play_id","defensive_team"]],
        on=["game_id","play_id"],
        how="left"
    )
    .dropna(subset=["defensive_team","peak_CLI"])
)

# minimum sample filter
MIN_PLAYS = 200
team_counts = team_peaks["defensive_team"].value_counts()
valid_teams = team_counts[team_counts >= MIN_PLAYS].index
team_peaks = team_peaks[team_peaks["defensive_team"].isin(valid_teams)]



# example: manually select illustrative teams
teams_to_plot = ["HOU", "LAC", "MIA", "PHI"]
league_vals = team_peaks["peak_CLI"].values



plt.figure(figsize=(9,6))

# league ECDF
x = np.sort(league_vals)
y = np.arange(1, len(x)+1) / len(x)
plt.plot(x, y, color="black", lw=2, linestyle="--", label="League")

# team ECDFs
for team in teams_to_plot:
    vals = team_peaks.loc[
        team_peaks["defensive_team"] == team, "peak_CLI"
    ].values
    x = np.sort(vals)
    y = np.arange(1, len(x)+1) / len(x)
    plt.plot(x, y, lw=2, label=team)

# catastrophic threshold
THRESH = 0.9
plt.axvline(THRESH, color="gray", linestyle=":", alpha=0.7)
plt.text(THRESH+0.01, 0.05, "Catastrophic\nleverage", fontsize=9)

plt.xlabel("Peak CLI allowed")
plt.ylabel("ECDF (fraction of plays)")
plt.title("Tail risk: how often defenses allow extreme leverage")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()



tail_stats = (
    team_peaks.assign(catastrophic=lambda d: d["peak_CLI"] >= 0.9)
    .groupby("defensive_team")
    .agg(
        n=("peak_CLI","count"),
        tail_rate=("catastrophic","mean")
    )
    .sort_values("tail_rate", ascending=False)
)

tail_stats.head(8)



# --- Team risk profile table ---
team_profile = (
    team_peaks
    .groupby("defensive_team")
    .agg(
        n=("peak_CLI","count"),
        peak_med=("peak_CLI","median"),
        peak_p90=("peak_CLI", lambda x: np.quantile(x, 0.90)),
        tail_rate=("peak_CLI", lambda x: (x >= 0.9).mean()),
    )
    .reset_index()
)

# bring in KM median survival
km_medians = (
    plays.groupby("defensive_team")
    .apply(lambda d: km_median(km_curve(d["t"], d["event"])))
    .rename("median_tau_alive")
    .reset_index()
)

team_profile = team_profile.merge(km_medians, on="defensive_team", how="left")

team_profile = team_profile[team_profile["n"] >= 200]
team_profile = team_profile.sort_values("tail_rate", ascending=False)

team_profile.head(10)



top = team_profile.sort_values("tail_rate").head(6)
bot = team_profile.sort_values("tail_rate").tail(6)
rank_df = pd.concat([top, bot])

plt.figure(figsize=(7,4))
plt.hlines(
    y=rank_df["defensive_team"],
    xmin=0,
    xmax=rank_df["tail_rate"]
)
plt.scatter(rank_df["tail_rate"], rank_df["defensive_team"])
plt.xlabel("Catastrophic leverage rate (peak CLI ≥ 0.9)")
plt.title("Which defenses allow rare but extreme breakdowns?")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- settings ---
TAIL_THR = 0.90   # peak CLI threshold defining a "catastrophic leverage" play

# Defensive team per play
team_map = df_supp[["game_id","play_id","defensive_team"]].drop_duplicates()

# Peak CLI per play (from summ_all)
peak_map = summ_all[["game_id","play_id","peak_CLI"]].drop_duplicates()

# EPA per play (use expected_points_added, exists in df_supp)
epa_map = df_supp[["game_id","play_id","expected_points_added"]].drop_duplicates()

# Build a play-level table
play_tail = (
    peak_map
    .merge(team_map, on=["game_id","play_id"], how="left")
    .merge(epa_map, on=["game_id","play_id"], how="left")
)

# Keep tail events (catastrophic leverage)
tail_events = play_tail[play_tail["peak_CLI"] >= TAIL_THR].copy()

# Mean EPA on tail events per team
team_tail_epa = (
    tail_events
    .groupby("defensive_team")["expected_points_added"]
    .mean()
    .rename("mean_tail_epa")
    .reset_index()
)

# Merge into team_profile (won't break if a team has no tail events)
team_profile = team_profile.merge(team_tail_epa, on="defensive_team", how="left")
team_profile["mean_tail_epa"] = team_profile["mean_tail_epa"].fillna(0.0)

print("Added mean_tail_epa. Columns now:", team_profile.columns.tolist())



plt.figure(figsize=(10,6))

# --- Bubble size = damage when things go wrong ---
# scale for visual balance
sizes = 300 + 3000 * (
    team_profile["mean_tail_epa"] 
    / team_profile["mean_tail_epa"].max()
)

# --- Color by risk regime (high tail & slow close = red) ---
risk_score = (
    (team_profile["tail_rate"] - team_profile["tail_rate"].median()) /
    team_profile["tail_rate"].std()
    -
    (team_profile["median_tau_alive"] - team_profile["median_tau_alive"].median()) /
    team_profile["median_tau_alive"].std()
)

plt.scatter(
    team_profile["tail_rate"],
    team_profile["median_tau_alive"],
    s=sizes,
    c=risk_score,
    cmap="coolwarm",
    alpha=0.85,
    edgecolor="k",
    linewidth=0.5
)

# --- League medians (quadrants) ---
x_med = team_profile["tail_rate"].median()
y_med = team_profile["median_tau_alive"].median()

plt.axvline(x_med, linestyle="--", alpha=0.4)
plt.axhline(y_med, linestyle="--", alpha=0.4)

# --- Label only structurally interesting teams ---
for _, r in team_profile.iterrows():
    if (
        r["tail_rate"] > team_profile["tail_rate"].quantile(0.85) or
        r["median_tau_alive"] < team_profile["median_tau_alive"].quantile(0.15)
    ):
        plt.text(
            r["tail_rate"] + 0.002,
            r["median_tau_alive"],
            r["defensive_team"],
            fontsize=9,
            weight="bold"
        )
# change tittle When coverage breaks: how often, how fast, and how costly?
plt.xlabel("Catastrophic leverage rate (peak CLI ≥ 0.9)")
plt.ylabel("Median catch-window survival time (τ)")
plt.title(
    "Defensive risk profiles:\n"
    "How often coverage breaks vs how fast defenses recover\n"
    "(Bubble size = EPA allowed when breakdown occurs)"
)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10,6))

# bubble size from mean_tail_epa (safe even if all zeros)
mx = team_profile["mean_tail_epa"].max()
if mx <= 1e-9:
    sizes = np.full(len(team_profile), 300.0)
else:
    sizes = 250 + 2500 * (team_profile["mean_tail_epa"] / mx)

# risk score: high tail_rate + low median_tau_alive = risky
risk_score = (
    (team_profile["tail_rate"] - team_profile["tail_rate"].median()) / (team_profile["tail_rate"].std() + 1e-9)
    -
    (team_profile["median_tau_alive"] - team_profile["median_tau_alive"].median()) / (team_profile["median_tau_alive"].std() + 1e-9)
)

plt.scatter(
    team_profile["tail_rate"],
    team_profile["median_tau_alive"],
    s=sizes,
    c=risk_score,
    cmap="coolwarm",
    alpha=0.85,
    edgecolor="k",
    linewidth=0.5
)

x_med = team_profile["tail_rate"].median()
y_med = team_profile["median_tau_alive"].median()
plt.axvline(x_med, linestyle="--", alpha=0.4)
plt.axhline(y_med, linestyle="--", alpha=0.4)

# label only meaningful outliers
for _, r in team_profile.iterrows():
    if (
        r["tail_rate"] > team_profile["tail_rate"].quantile(0.85) or
        r["median_tau_alive"] < team_profile["median_tau_alive"].quantile(0.15)
    ):
        plt.text(r["tail_rate"] + 0.002, r["median_tau_alive"], r["defensive_team"], fontsize=9, weight="bold")
#change title : When coverage breaks: how often, how fast, and how costly?
plt.xlabel("Catastrophic leverage rate (peak CLI ≥ 0.9)")
plt.ylabel("Median catch-window survival time (τ)")
plt.title("Defensive risk profiles: breakdown frequency vs recovery speed\n(bubble size = mean EPA on catastrophic plays)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()




import numpy as np
import pandas as pd

# --- 1) Start from summ_all (play-level CLI metrics) ---
pp = summ_all[[
    "game_id","play_id",
    "peak_CLI","final_CLI",  # and cli_drop if you want
    "pass_result","type"
]].drop_duplicates().copy()

print("pp from summ_all:", pp.shape)

# --- 2) Bring scheme + team + EPA from df_supp ---
supp_cols = [
    "game_id","play_id",
    "defensive_team",
    "team_coverage_man_zone",   # MAN_COVERAGE vs ZONE_COVERAGE
    "team_coverage_type",       # COVER_1_MAN, COVER_3_ZONE, etc.
    "expected_points_added"
]
pp = pp.merge(df_supp[supp_cols].drop_duplicates(), on=["game_id","play_id"], how="left")

print("pp after merging df_supp:", pp.shape)
print("Missing scheme:", pp["team_coverage_man_zone"].isna().mean(), "missing defensive_team:", pp["defensive_team"].isna().mean())

# --- 3) Define a clean scheme label ---
pp["scheme"] = pp["team_coverage_man_zone"].map({
    "MAN_COVERAGE": "Man",
    "ZONE_COVERAGE": "Zone"
})

print("scheme counts:\n", pp["scheme"].value_counts(dropna=False))
print("defensive_team non-null:", pp["defensive_team"].notna().mean())



# plays should have: game_id, play_id, t (tau event/censor time), event
print("plays:", plays.shape)
print("plays cols:", plays.columns.tolist())

pp = pp.merge(plays[["game_id","play_id","t","event"]].drop_duplicates(),
              on=["game_id","play_id"], how="left")

print("pp after merging survival t:", pp.shape)
print("Missing t:", pp["t"].isna().mean())



TAIL_THR = 0.90

# Safety: drop rows missing essentials
pp2 = pp.dropna(subset=["scheme","defensive_team","peak_CLI","t"]).copy()
print("pp2 usable:", pp2.shape)

# Base counts -> guarantees `n`
base = (
    pp2.groupby(["scheme","defensive_team"])
       .size()
       .reset_index(name="n")
)

metrics = (
    pp2.groupby(["scheme","defensive_team"])
       .agg(
           tail_rate=("peak_CLI", lambda x: (x >= TAIL_THR).mean()),
           median_tau_alive=("t", "median"),
           peak_med=("peak_CLI", "median"),
           peak_p90=("peak_CLI", lambda x: np.quantile(x, 0.90)),
           mean_tail_epa=("expected_points_added",
                          lambda s: s[pp2.loc[s.index, "peak_CLI"] >= TAIL_THR].mean())
       )
       .reset_index()
)

team_scheme = base.merge(metrics, on=["scheme","defensive_team"], how="left")

print("team_scheme:", team_scheme.shape)
print(team_scheme.head())

MIN_N_MAN  = 60
MIN_N_ZONE = 150


team_scheme = team_scheme[
    ((team_scheme["scheme"] == "Man")  & (team_scheme["n"] >= MIN_N_MAN)) |
    ((team_scheme["scheme"] == "Zone") & (team_scheme["n"] >= MIN_N_ZONE))
].copy()
print("team_scheme after MIN_N_TEAM:", team_scheme.shape)


# 3) Plot helper (same plot, faceted)
def plot_scheme_panel(ax, df_panel, title):
    if df_panel.empty:
        ax.set_title(title + " (no data after filtering)")
        ax.axis("off")
        return

    # bubble size: mean EPA on tail plays (safe even if zeros)
    mx = df_panel["mean_tail_epa"].max()
    if mx <= 1e-9:
        sizes = np.full(len(df_panel), 300.0)
    else:
        sizes = 300 + 2000 * (df_panel["mean_tail_epa"] / mx)

    sc = ax.scatter(
        df_panel["tail_rate"],
        df_panel["median_tau_alive"],
        s=sizes,
        alpha=0.85
    )

    x_med = df_panel["tail_rate"].median()
    y_med = df_panel["median_tau_alive"].median()
    ax.axvline(x_med, linestyle="--", alpha=0.35)
    ax.axhline(y_med, linestyle="--", alpha=0.35)

    # label only extremes (top tail risk OR bottom survival)
    for _, r in df_panel.iterrows():
        if (
            r["tail_rate"] >= df_panel["tail_rate"].quantile(0.90) or
            r["median_tau_alive"] <= df_panel["median_tau_alive"].quantile(0.10)
        ):
            ax.text(r["tail_rate"], r["median_tau_alive"], r["defensive_team"], fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(f"Catastrophic leverage rate  P(peak CLI ≥ {TAIL_THR:.2f})")
    ax.set_ylabel("Median window survival time (τ)")
    ax.grid(alpha=0.25)

# 4) Make the 2-panel figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

plot_scheme_panel(
    axes[0],
    team_scheme[team_scheme["scheme"] == "Man"].copy(),
    "Defensive risk profiles (Man-heavy plays)"
)
plot_scheme_panel(
    axes[1],
    team_scheme[team_scheme["scheme"] == "Zone"].copy(),
    "Defensive risk profiles (Zone-heavy plays)"
)


fig.suptitle("Scheme split: is risk driven by scheme or execution?  (bubble = EPA when breakdown occurs)", y=1.02)
plt.tight_layout()
plt.show()



team_scheme.groupby("scheme")[["tail_rate","median_tau_alive"]].agg(["mean","std"])



pivot = team_scheme.pivot(
    index="defensive_team",
    columns="scheme",
    values=["tail_rate","median_tau_alive"]
)

pivot["tail_rate_diff"] = pivot["tail_rate"]["Man"] - pivot["tail_rate"]["Zone"]
pivot["tau_diff"] = pivot["median_tau_alive"]["Man"] - pivot["median_tau_alive"]["Zone"]

pivot.dropna().sort_values("tail_rate_diff", ascending=False).head()



import numpy as np
import pandas as pd

# --- 1) Build per-play summary for film selection ---
film_tbl = (
    summ_all.merge(
        df_supp[[
            "game_id", "play_id",
            "team_coverage_type",
            "down", "yards_to_go",
            "play_description"
        ]],
        on=["game_id", "play_id"],
        how="left"
    )
    .copy()
)

# --- 2) Window survival normalized (from your curve-derived metrics) ---
# If VCT_sec is "time of first collapse" (absolute t_rel), then time-to-collapse after peak is:
# tau_alive = (VCT_sec - t_peak) / (t_final - t_peak)
# This is usually the cleanest survival fraction.
den = (film_tbl["t_final"] - film_tbl["t_peak"]).replace(0, np.nan)
film_tbl["tau_alive"] = (film_tbl["VCT_sec"] - film_tbl["t_peak"]) / den

# Fallback: if you actually stored VCT_sec as "time after peak" already, use this instead:
# film_tbl["tau_alive"] = film_tbl["VCT_sec"] / (film_tbl["t_final"] - film_tbl["t_peak"])

# --- 3) Keep only reasonable downfield plays for film study ---
film_tbl = film_tbl[
    (film_tbl["pass_length"].between(8, 25)) &
    (film_tbl["peak_CLI"] >= 0.85) &
    (film_tbl["tau_alive"].between(0.30, 0.95)) &
    (film_tbl["team_coverage_type"].notna())
].copy()

# --- 4) Bin by peak CLI + pass length to enforce similarity ---
film_tbl["peak_bin"] = pd.cut(
    film_tbl["peak_CLI"],
    bins=[0.85, 0.90, 0.95, 1.01],
    include_lowest=True
)
film_tbl["len_bin"] = pd.cut(
    film_tbl["pass_length"],
    bins=[8, 12, 16, 20, 25],
    include_lowest=True
)

# --- 5) Find bins that contain BOTH fast and slow collapses ---
# (big spread in tau_alive within same peak+length bucket)
candidates = (
    film_tbl
    .groupby(["peak_bin", "len_bin"], dropna=True)
    .filter(lambda x: (x["tau_alive"].quantile(0.90) - x["tau_alive"].quantile(0.10)) > 0.30)
)

print("film_tbl:", film_tbl.shape)
print("candidates:", candidates.shape)

# Show a few "fast collapse" examples (small tau_alive)
display(
    candidates[[
        "game_id","play_id",
        "peak_CLI","final_CLI","CDI",
        "tau_alive",
        "team_coverage_type","pass_length",
        "down","yards_to_go",
        "play_description"
    ]]
    .sort_values("tau_alive")
    .head(12)
)



# --- 2) Bin by peak CLI and pass length to enforce similarity ---

film_tbl["peak_bin"] = pd.cut(
    film_tbl["peak_CLI"],
    bins=[0.85, 0.90, 0.95, 1.01]
)

film_tbl["len_bin"] = pd.cut(
    film_tbl["pass_length"],
    bins=[8, 12, 16, 20, 25]
)

# Look for bins with both fast and slow collapses
candidates = (
    film_tbl
    .groupby(["peak_bin","len_bin"])
    .filter(lambda x: x["tau_alive"].quantile(0.9) - x["tau_alive"].quantile(0.1) > 0.30)
)

print("Candidate plays:", candidates.shape)
candidates[[
    "game_id","play_id","peak_CLI","tau_alive",
    "team_coverage_type","pass_length","play_description"
]].sort_values("tau_alive").head(10)



# --- 3) Select your two plays 

# Fast collapse
play_A = candidates.sort_values("tau_alive").iloc[0]

# Slow collapse
play_B = candidates.sort_values("tau_alive").iloc[-1]

play_A, play_B



import matplotlib.pyplot as plt
import numpy as np

def plot_single_play_cli(game_id, play_id, title, annotations=None):
    dfp = curves_all[
        (curves_all["game_id"] == game_id) &
        (curves_all["play_id"] == play_id)
    ].sort_values("t_rel").copy()

    # normalize time in air
    tmin, tmax = dfp["t_rel"].min(), dfp["t_rel"].max()
    dfp["tau"] = (dfp["t_rel"] - tmin) / (tmax - tmin)

    # peak CLI
    peak_idx = dfp["CLI"].idxmax()
    peak_tau = dfp.loc[peak_idx, "tau"]
    peak_cli = dfp.loc[peak_idx, "CLI"]

    plt.figure(figsize=(7, 4))

    plt.plot(dfp["tau"], dfp["CLI"], lw=2)
    plt.scatter(peak_tau, peak_cli, color="red", zorder=3)

    plt.axhline(peak_cli - 0.10, linestyle="--", alpha=0.4)

    #  ADD FILM ANNOTATIONS (exactly what you want)
    if annotations is not None:
        for ann in annotations:
            plt.annotate(**ann)

    plt.xlabel("Normalized time in air (τ)")
    plt.ylabel("Catch Leverage Index (CLI)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()



annotations_A = [
    dict(
        text="Peak leverage\n(DB loses step)",
        xy=(0.2, 1.0),
        xytext=(0.35, 0.95),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9
    ),
    dict(
        text="Window collapses\nbefore recovery",
        xy=(0.6, 0.78),
        xytext=(0.75, 0.85),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9
    )
]



annotations_B = [
    dict(
        text="Sustained leverage\n(ball in air)",
        xy=(0.5, 0.98),
        xytext=(0.3, 1.05),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9
    ),
    dict(
        text="Late collapse\n(help arrives)",
        xy=(0.9, 0.75),
        xytext=(0.7, 0.5),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9
    )
]



plot_single_play_cli(
    play_A.game_id,
    play_A.play_id,
    "Play A — Fast collapse (window closes quickly)",
    annotations=annotations_A
)

plot_single_play_cli(
    play_B.game_id,
    play_B.play_id,
    "Play B — Slow collapse (window survives)",
    annotations=annotations_B
)





