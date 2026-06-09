pip install fastdtw


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def save_and_show(fig_name):
    plt.savefig(
        f"{FIG_DIR}/{fig_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# Load weekly tracking data
weeks = []
for i in range(1, 19):
    path = f"/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w{i:02d}.csv"
    weeks.append(pd.read_csv(path))

tracking = pd.concat(weeks, ignore_index=True)

# Player lookup (build ONCE)
player_lookup = (
    tracking[["nfl_id", "player_name"]]
    .drop_duplicates()
)

# Load supplementary data
supp = pd.read_csv(
    "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv",
    low_memory=False
)

# Column reduction
tracking = tracking[
    ["game_id", "play_id", "frame_id", "nfl_id",
     "player_role", "play_direction", "x", "y"]
]

supp = supp[
    ["game_id", "play_id", "route_of_targeted_receiver",
     "team_coverage_man_zone", "pass_result",
     "play_nullified_by_penalty", "expected_points_added"]
]

# Valid play filter
valid_plays = supp[
    (supp["route_of_targeted_receiver"].notna()) &
    (supp["play_nullified_by_penalty"] == "N") &
    (supp["pass_result"].isin(["C", "I", "IN"]))
]

# Merge & keep targeted WRs
df = tracking.merge(valid_plays, on=["game_id", "play_id"], how="inner")
df = df[df["player_role"] == "Targeted Receiver"]



route_records = []

for (gid, pid, nid), g in df.groupby(["game_id", "play_id", "nfl_id"]):
    g = g.sort_values("frame_id")

    route_records.append({
        "game_id": gid,
        "play_id": pid,
        "nfl_id": nid,
        "route_type": g["route_of_targeted_receiver"].iloc[0],
        "play_direction": g["play_direction"].iloc[0],
        "path_raw": np.array(g[["x", "y"]])
    })

wr_paths = pd.DataFrame(route_records)



FIELD_CENTER_Y = 26.65

def canonicalize_path(path, play_direction):
    x, y = path[:, 0], path[:, 1]

    # Depth normalization
    if str(play_direction).lower() == "left":
        depth = 120 - x
    else:
        depth = x

    # Width centered at field midline
    width = y - FIELD_CENTER_Y

    pts = np.column_stack([depth, width])

    # Anchor at (0,0)
    pts = pts - pts[0]

    # Mirror left-breaking routes
    if pts[-1, 1] < 0:
        pts[:, 1] *= -1

    return pts

wr_paths["path_canon"] = wr_paths.apply(
    lambda r: canonicalize_path(r["path_raw"], r["play_direction"]),
    axis=1
)



sample = wr_paths.sample(40, random_state=0)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for p in sample["path_raw"]:
    plt.plot(p[:,0], p[:,1], alpha=0.4)
plt.title("Raw Routes")
plt.xlabel("X"); plt.ylabel("Y")

plt.subplot(1,2,2)
for p in sample["path_canon"]:
    plt.plot(p[:,1], p[:,0], alpha=0.4)
plt.title("Canonicalized Routes")
plt.xlabel("Width"); plt.ylabel("Depth")

plt.tight_layout()
save_and_show("raw_vs_canonicalized_routes")
plt.show()



from scipy.interpolate import interp1d

def normalize_path(path, N=50):
    diffs = np.diff(path, axis=0)
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs, axis=1))])

    if dist[-1] == 0:
        return np.zeros((N, 2))

    dist /= dist[-1]
    fx = interp1d(dist, path[:,0])
    fy = interp1d(dist, path[:,1])
    grid = np.linspace(0, 1, N)

    return np.column_stack([fx(grid), fy(grid)])

wr_paths["path_norm"] = wr_paths["path_canon"].apply(normalize_path)



templates = {}

for rt, g in wr_paths.groupby("route_type"):
    stack = np.stack(g["path_norm"])
    mean_template = np.mean(stack, axis=0)

    # Distance to mean
    dists = np.linalg.norm(stack - mean_template, axis=(1,2))
    keep = dists <= np.percentile(dists, 90)

    templates[rt] = np.mean(stack[keep], axis=0)



# Plot empirical templates for ALL routes
for rt in sorted(templates.keys()):
    subset = wr_paths[wr_paths["route_type"] == rt]

    # Skip routes with very small sample sizes
    if len(subset) < 20:
        continue

    plt.figure(figsize=(6, 8))

    # Plot a random sample of routes for visual clarity
    for p in subset["path_norm"].sample(
        min(100, len(subset)),
        random_state=0
    ):
        plt.plot(p[:, 1], p[:, 0], color="gray", alpha=0.15)

    # Plot the empirical template
    t = templates[rt]
    plt.plot(t[:, 1], t[:, 0], color="black", linewidth=3)

    plt.title(f"{rt} — Route Template")
    plt.xlabel("Route Width (yards)")
    plt.ylabel("Route Depth (yards)")

    plt.xlim(-12, 12)
    plt.ylim(0, 35)
    plt.gca().set_aspect("equal")
    plt.grid(False)
    
    save_and_show(f"template_{rt.lower()}")
    plt.show()




from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dtw_score(path, template):
    d, _ = fastdtw(path, template, dist=euclidean)
    return d

wr_paths["execution_raw"] = wr_paths.apply(
    lambda r: dtw_score(r["path_norm"], templates[r["route_type"]]),
    axis=1
)



# Best vs Worst execution for ALL routes
for rt in sorted(templates.keys()):
    subset = wr_paths[wr_paths["route_type"] == rt]

    if subset.empty:
        continue

    # Require enough samples to be meaningful
    if len(subset) < 10:
        continue

    best = subset.nsmallest(5, "execution_raw")
    worst = subset.nlargest(5, "execution_raw")

    plt.figure(figsize=(6, 8))

    # Worst execution (red)
    for _, r in worst.iterrows():
        p = r["path_norm"]
        plt.plot(p[:,1], p[:,0], color="red", alpha=0.4)

    # Best execution (blue)
    for _, r in best.iterrows():
        p = r["path_norm"]
        plt.plot(p[:,1], p[:,0], color="blue", alpha=0.6)

    # Template
    t = templates[rt]
    plt.plot(t[:,1], t[:,0], color="black", linewidth=3)

    plt.title(f"{rt} — Best (Blue) vs Worst (Red) Execution")
    plt.xlabel("Route Width (yards)")
    plt.ylabel("Route Depth (yards)")
    plt.xlim(-12, 12)
    plt.ylim(0, 35)
    plt.gca().set_aspect("equal")
    plt.grid(False)
    save_and_show(f"best_worst_{rt.lower()}")
    plt.show()



wr_scored = wr_paths.merge(
    valid_plays[["game_id","play_id","pass_result","expected_points_added", "team_coverage_man_zone"]],
    on=["game_id","play_id"],
    how="left"
).merge(
    player_lookup,
    on="nfl_id",
    how="left"
)

wr_scored["completed"] = (wr_scored["pass_result"] == "C").astype(int)



# Dataframe used for route-level execution distribution visuals
df_plot = (
    wr_scored
    .dropna(subset=["execution_raw", "route_type"])
    .copy()
)

route_order = (
    df_plot
    .groupby("route_type")["execution_raw"]
    .median()
    .sort_values()
    .index.tolist()
)

n_routes = len(route_order)
cols = 4
rows = (n_routes + cols - 1) // cols

plt.figure(figsize=(14, rows * 2.6))

for i, rt in enumerate(route_order, 1):
    plt.subplot(rows, cols, i)

    subset = df_plot[df_plot["route_type"] == rt]["execution_raw"]

    plt.hist(subset, bins=30, alpha=0.75)
    plt.title(rt)
    plt.xlabel("DTW Score")
    plt.ylabel("Count")
    plt.grid(False)

plt.suptitle(
    "Route Execution Score Distributions by Route Type\n(Lower = Cleaner Execution)",
    fontsize=14,
    y=1.02
)

plt.tight_layout()
save_and_show("execution_dist")
plt.show()



df_roll = wr_scored.sort_values("execution_raw")
WINDOW = 150

df_roll["roll_comp"] = df_roll["completed"].rolling(WINDOW, center=True).mean()
df_roll["roll_epa"] = df_roll["expected_points_added"].rolling(WINDOW, center=True).mean()

plt.figure(figsize=(7,4))
plt.plot(df_roll["execution_raw"], df_roll["roll_comp"])
plt.xlabel("DTW Score"); plt.ylabel("Completion Rate")
plt.title("Execution Quality vs Completion Rate")
save_and_show("dtw_vs_completion_rate")
plt.show()

plt.figure(figsize=(7,4))
plt.plot(df_roll["execution_raw"], df_roll["roll_epa"])
plt.xlabel("DTW Score"); plt.ylabel("Mean EPA")
plt.title("Execution Quality vs EPA")
save_and_show("dtw_vs_epa")
plt.show()



def plot_man_zone(route_type, df):
    subset = df[
        (df["route_type"] == route_type) &
        (df["team_coverage_man_zone"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"]))
    ]

    if subset.empty:
        return

    plt.figure(figsize=(5,4))

    for cov, g in subset.groupby("team_coverage_man_zone"):
        plt.hist(
            g["execution_raw"],
            bins=30,
            alpha=0.6,
            label=f"{cov.replace('_', ' ')} (n={len(g)})"
        )

    plt.xlabel("DTW Route Execution Score\n(Lower = Cleaner Route)")
    plt.ylabel("Count")
    plt.title(f"{route_type} — Execution by Coverage")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    save_and_show(f"man_zone_{rt.lower()}")
    plt.show()



# Plot man vs zone execution for all routes
for rt in sorted(wr_scored["route_type"].dropna().unique()):
    plot_man_zone(rt, wr_scored)



coverage_summary = (
    wr_scored[
        wr_scored["team_coverage_man_zone"].isin(["MAN_COVERAGE","ZONE_COVERAGE"])
    ]
    .groupby(["route_type","team_coverage_man_zone"])
    .execution_raw.mean()
    .unstack()
    .dropna()
)

coverage_summary["man_minus_zone"] = (
    coverage_summary["MAN_COVERAGE"] -
    coverage_summary["ZONE_COVERAGE"]
)

coverage_summary = coverage_summary.sort_values("man_minus_zone")



plt.figure(figsize=(7,5))
plt.barh(
    coverage_summary.index,
    coverage_summary["man_minus_zone"]
)

plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Mean DTW Difference (Man − Zone)")
plt.title("Coverage Impact on Route Execution\n(Positive = Cleaner Route in Zone)")
plt.grid(False)
plt.tight_layout()
save_and_show(f"coverage_imapct")
plt.show()



wr_scored["execution_z"] = (
    wr_scored.groupby("route_type")["execution_raw"]
    .transform(lambda x: (x - x.mean()) / x.std())
)

player_summary = (
    wr_scored.groupby(["nfl_id","player_name","route_type"])
    .agg(
        mean_z=("execution_z","mean"),
        n=("execution_z","count")
    )
    .reset_index()
)

player_summary = player_summary[player_summary["n"] >= 10]



MIN_ROUTES_PER_PLAYER = 10

# Player-route summaries
player_route = (
    wr_scored
    .dropna(subset=["execution_raw"])
    .groupby(["route_type", "nfl_id", "player_name"])
    .agg(
        mean_exec=("execution_raw", "mean"),
        n_routes=("execution_raw", "count")
    )
    .reset_index()
)

# Enforce route-specific minimum sample size
player_route = player_route[player_route["n_routes"] >= MIN_ROUTES_PER_PLAYER].copy()

# Empirical Bayes shrinkage
def empirical_bayes(group, route_type):
    """
    Shrinks player means toward the route-level mean
    based on sample size (Empirical Bayes).
    """
    mu = group["mean_exec"].mean()
    tau2 = group["mean_exec"].var(ddof=1)

    sigma2 = (
        wr_scored.loc[wr_scored["route_type"] == route_type, "execution_raw"]
        .var(ddof=1)
    )

    # Edge-case protection
    if tau2 == 0 or np.isnan(tau2) or np.isnan(sigma2):
        group["eb_exec"] = mu
        return group

    w = tau2 / (tau2 + sigma2 / group["n_routes"])
    group["eb_exec"] = w * group["mean_exec"] + (1 - w) * mu
    return group

# Apply EB shrinkage
out = []
for rt in player_route["route_type"].unique():
    g = player_route[player_route["route_type"] == rt].copy()

    # Only keep the columns needed for EB math
    g = g[["nfl_id", "player_name", "mean_exec", "n_routes"]]

    g = empirical_bayes(g, rt)
    g["route_type"] = rt  # explicitly reattach route_type

    out.append(g)

player_route_eb = pd.concat(out, ignore_index=True)



ROUTES_TO_SHOW = (
    wr_paths
    .groupby("route_type")
    .size()
    .loc[lambda x: x >= 200]
    .index
    .tolist()
)


def select_best_worst(df, route_type):
    sub = df[df["route_type"] == route_type]

    best = sub.sort_values("eb_exec").iloc[0]
    worst = sub.sort_values("eb_exec").iloc[-1]

    return best, worst



def plot_player_vs_template(nfl_id, route_type, label, color):
    subset = wr_scored[
        (wr_scored["nfl_id"] == nfl_id) &
        (wr_scored["route_type"] == route_type)
    ]

    subset = subset.sample(min(len(subset), 20), random_state=0)

    plt.figure(figsize=(6,8))

    for _, r in subset.iterrows():
        plt.plot(
            r["path_norm"][:,1],
            r["path_norm"][:,0],
            color=color,
            alpha=0.4
        )

    t = templates[route_type]
    plt.plot(t[:,1], t[:,0], color="black", linewidth=3)

    plt.title(label)
    plt.xlabel("Route Width (yards)")
    plt.ylabel("Route Depth (yards)")
    plt.xlim(-12, 12)
    plt.ylim(0, 35)
    plt.gca().set_aspect("equal")
    plt.grid(False)
    save_and_show(f"best_execution_{rt.lower()}")
    plt.show()


for rt in ROUTES_TO_SHOW:
    best, worst = select_best_worst(player_route_eb, rt)

    print(f"\n{rt} ROUTE")
    print(f"Best (EB):  {best['player_name']}  | n={best['n_routes']}")
    print(f"Worst (EB): {worst['player_name']} | n={worst['n_routes']}")

    plot_player_vs_template(
        best["nfl_id"],
        rt,
        f"{rt} — Best Execution (Bias-Corrected)",
        color="blue"
    )

    plot_player_vs_template(
        worst["nfl_id"],
        rt,
        f"{rt} — Worst Execution (Bias-Corrected)",
        color="red"
    )



# Get all route types with sufficient data
route_types = (
    wr_scored["route_type"]
    .dropna()
    .unique()
)

MIN_ROUTES_PER_PLAYER = 10  # same threshold you used

for rt in sorted(route_types):

    sub = wr_scored[wr_scored["route_type"] == rt]

    # Player-level stats
    stats = (
        sub.groupby(["nfl_id", "player_name"])
        .execution_z
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Enforce minimum sample size
    stats = stats[stats["count"] >= MIN_ROUTES_PER_PLAYER]

    if stats.empty:
        continue

    plt.figure(figsize=(7, 6))
    plt.scatter(
        stats["mean"],
        stats["std"],
        alpha=0.6
    )

    # Reference line for league-average execution
    plt.axvline(
        0,
        linestyle="--",
        color="gray",
        linewidth=1
    )

    plt.xlabel("Mean Execution (z-score)")
    plt.ylabel("Execution Variability (Std Dev)")
    plt.title(f"{rt} Route — Player Profile")

    # Select players to label:
    label_df = pd.concat([
        stats.nsmallest(2, "mean"),   # best execution
        stats.nlargest(2, "mean"),    # worst execution
        stats.nsmallest(2, "std"),    # most consistent
        stats.nlargest(2, "std")      # most volatile
    ]).drop_duplicates()

    # Add labels
    for _, r in label_df.iterrows():
        plt.text(
            r["mean"],
            r["std"],
            r["player_name"],
            fontsize=8,
            alpha=0.85
        )

    plt.grid(False)
    plt.tight_layout()
    save_and_show(f"reliability_{rt.lower()}")
    plt.show()



TOP_N = 5
MIN_ROUTES_PER_PLAYER = 10

def build_route_table(df, route_type, top_n=5):
    g = df[
        (df["route_type"] == route_type) &
        (df["n_routes"] >= MIN_ROUTES_PER_PLAYER)
    ].copy()

    if len(g) < top_n * 2:
        return None

    best = g.nsmallest(top_n, "eb_exec").assign(Rank="Best")
    worst = g.nlargest(top_n, "eb_exec").assign(Rank="Worst")

    out = pd.concat([best, worst])

    return (
        out[["Rank", "player_name", "n_routes", "eb_exec"]]
        .rename(columns={
            "player_name": "Player",
            "n_routes": "Routes Run",
            "eb_exec": "EB Execution Score"
        })
        .sort_values(["Rank", "EB Execution Score"])
        .reset_index(drop=True)
    )

def style_best_worst(df):
    def highlight(row):
        if row["Rank"] == "Best":
            return [
                "background-color: #e6f2ff; color: black; font-weight: bold"
                for _ in row
            ]
        else:
            return [
                "background-color: #fdecea; color: black; font-weight: bold"
                for _ in row
            ]

    return (
        df.style
        .apply(highlight, axis=1)
        .format({
            "EB adj Execution Score": "{:.2f}"
        })
        .hide(axis="index")
    )

# Render tables for all routes
routes = sorted(player_route_eb["route_type"].unique())

for rt in routes:
    tbl = build_route_table(player_route_eb, rt, TOP_N)

    if tbl is None:
        continue

    print(f"\n{rt} Route Execution")
    display(style_best_worst(tbl))



def save_route_table_image(df, route_type, save_dir="tables"):
    """
    Render a Best/Worst execution table as a matplotlib image and save it.
    """
    # Filter
    g = df[
        (df["route_type"] == route_type) &
        (df["n_routes"] >= MIN_ROUTES_PER_PLAYER)
    ].copy()

    if len(g) < TOP_N * 2:
        return

    best = g.nsmallest(TOP_N, "eb_exec").assign(Rank="Best")
    worst = g.nlargest(TOP_N, "eb_exec").assign(Rank="Worst")
    out = pd.concat([best, worst])

    table_df = (
        out[["Rank", "player_name", "n_routes", "eb_exec"]]
        .rename(columns={
            "player_name": "Player",
            "n_routes": "Routes Run",
            "eb_exec": "EB Execution Score"
        })
        .sort_values(["Rank", "EB Execution Score"])
        .reset_index(drop=True)
    )

    # create figure
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(table_df) + 1))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        else:
            rank = table_df.iloc[row - 1]["Rank"]
            if rank == "Best":
                cell.set_facecolor("#e6f2ff")
            else:
                cell.set_facecolor("#fdecea")

    plt.title(f"{route_type} Route — Bias-Corrected Execution", fontsize=12, pad=12)
    save_and_show(f"{rt} Route — Bias-Corrected Execution")



for rt in sorted(player_route_eb["route_type"].unique()):
    save_route_table_image(player_route_eb, rt)

