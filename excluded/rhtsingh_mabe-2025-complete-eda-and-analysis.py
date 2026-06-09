from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# consistent color palette
PALETTE = {
    "deep_blue": "#1E3A8A",
    "slate_gray": "#374151",
    "teal": "#0D9488",
    "muted": "#94A3B8",
    "bg": "#FFFFFF"
}

# seaborn/matplotlib defaults
sns.set_theme(
    context="talk",
    style="whitegrid",
    rc={
        "figure.dpi": 120,
        "axes.facecolor": PALETTE["bg"],
        "figure.facecolor": PALETTE["bg"],
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "grid.color": "#E6EEF2",
        "grid.linewidth": 0.8
    }
)

# dataset root
DATA = Path("/kaggle/input/MABe-mouse-behavior-detection")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

# read train.csv
train_csv_path = DATA / "train.csv"
if not train_csv_path.exists():
    raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
train_csv = pd.read_csv(train_csv_path)
train_csv["video_id"] = train_csv["video_id"].astype(str)

# folder structure
folders = {
    "train_tracking": DATA / "train_tracking",
    "train_annotation": DATA / "train_annotation",
    "test_tracking": DATA / "test_tracking"
}

# collect summary
summary_data = []
for folder_name, folder_path in folders.items():
    if not folder_path.exists():
        summary_data.append((folder_name, 0, 0))
        continue
    total_files = 0
    lab_dirs = [p for p in folder_path.iterdir() if p.is_dir()]
    for lab_dir in lab_dirs:
        files_in_lab = len(list(lab_dir.glob("*.parquet")))
        total_files += files_in_lab
    summary_data.append((folder_name, total_files, len(lab_dirs)))

summary_df = pd.DataFrame(summary_data, columns=["Directory", "Total Files", "Lab Count"]).sort_values("Total Files", ascending=True)

# plot
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=summary_df, x="Total Files", y="Directory", palette=[PALETTE["deep_blue"]], ax=ax)

for p in ax.patches:
    ax.annotate(f"{int(p.get_width()):,}", 
                (p.get_width(), p.get_y() + p.get_height() / 2),
                xytext=(6, 0), textcoords="offset points", 
                va="center", color=PALETTE["slate_gray"], fontsize=11)

ax.set_title("Top-level file counts by directory", color=PALETTE["slate_gray"], fontsize=14)
ax.set_xlabel("Total parquet files")
ax.set_ylabel("")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()


lab_inventory = []
for lab_id in sorted(train_csv["lab_id"].unique()):
    tracking_dir = folders["train_tracking"] / lab_id
    annotation_dir = folders["train_annotation"] / lab_id
    tracking_count = len(list(tracking_dir.glob("*.parquet"))) if tracking_dir.exists() else 0
    annotation_count = len(list(annotation_dir.glob("*.parquet"))) if annotation_dir.exists() else 0
    difference = tracking_count - annotation_count
    lab_inventory.append((lab_id, tracking_count, annotation_count, difference))

inventory_df = pd.DataFrame(
    lab_inventory, 
    columns=["Lab ID", "Tracking Files", "Annotation Files", "Difference (T-A)"]
).sort_values(["Annotation Files", "Tracking Files"], ascending=[False, False]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=inventory_df, 
    x="Lab ID", 
    y="Tracking Files", 
    color=PALETTE["deep_blue"], 
    ax=ax, 
    label="Tracking Files"
)
sns.barplot(
    data=inventory_df, 
    x="Lab ID", 
    y="Annotation Files", 
    color=PALETTE["teal"], 
    ax=ax, 
    label="Annotation Files"
)
ax.set_title("Per-lab tracking vs annotation files", color=PALETTE["slate_gray"], fontsize=15)
ax.set_xlabel("Lab ID")
ax.set_ylabel("Number of files")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



viz_df = inventory_df.copy()
viz_df["Lab Short"] = viz_df["Lab ID"].str.replace("_", " ").str.slice(0, 30)
top_labs = viz_df.nlargest(25, "Tracking Files").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
y = np.arange(len(top_labs))
bar_h = 0.35

ax.barh(y - bar_h/2, top_labs["Tracking Files"], height=bar_h, label="Tracking Files", color=PALETTE["deep_blue"], alpha=0.85)
ax.barh(y + bar_h/2, top_labs["Annotation Files"], height=bar_h, label="Annotation Files", color=PALETTE["teal"], alpha=0.95)

for i, (t, a) in enumerate(zip(top_labs["Tracking Files"], top_labs["Annotation Files"])):
    ax.text(t + max(top_labs["Tracking Files"]) * 0.003, i - bar_h/2, f"{int(t):,}", va="center", ha="left", color=PALETTE["slate_gray"], fontsize=10)
    ax.text(a + max(top_labs["Tracking Files"]) * 0.003, i + bar_h/2, f"{int(a):,}", va="center", ha="left", color=PALETTE["slate_gray"], fontsize=10)

ax.set_yticks(y)
ax.set_yticklabels(top_labs["Lab Short"], color=PALETTE["slate_gray"])
ax.invert_yaxis()
ax.set_xlabel("Number of Parquet Files", color=PALETTE["slate_gray"])
ax.set_title("File Availability by Lab — Tracking vs Annotation", color=PALETTE["slate_gray"], fontsize=16, pad=12)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.show()


from IPython.display import HTML, display

anno_root = folders["train_annotation"]
valid_annos = set()
if anno_root.exists():
    for lab_dir in sorted(anno_root.iterdir()):
        if not lab_dir.is_dir():
            continue
        for f in lab_dir.glob("*.parquet"):
            valid_annos.add((lab_dir.name, f.stem))

labs_no_annotations = inventory_df.loc[inventory_df["Annotation Files"] == 0, "Lab ID"].tolist()
usable_mask = train_csv.apply(lambda r: (r["lab_id"], r["video_id"]) in valid_annos, axis=1)
usable_train_rows = train_csv[usable_mask]

total_rows = len(train_csv)
usable_rows = len(usable_train_rows)
usable_pct = (usable_rows / total_rows * 100) if total_rows > 0 else 0.0

summary_html = f"""
<div style="border-left:4px solid {PALETTE['teal']}; padding:12px; font-family:sans-serif; background:#ffffff;">
  <h3 style="margin:0; color:{PALETTE['deep_blue']};">Missing Annotation Analysis</h3>
  <div style="color:{PALETTE['slate_gray']}; margin-top:6px;">
    <div><strong>Labs with zero annotation files</strong>: {', '.join(labs_no_annotations) if labs_no_annotations else 'None'}</div>
    <div style="margin-top:6px;"><strong>Total rows in train.csv</strong>: {total_rows:,}</div>
    <div><strong>Usable rows with annotation parquet</strong>: {usable_rows:,}</div>
    <div><strong>Usable percentage</strong>: {usable_pct:.1f}%</div>
  </div>
</div>
"""
display(HTML(summary_html))



top_labs = inventory_df.head(15)
bottom_labs = inventory_df.tail(15)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120, sharey=True)

sns.barplot(data=top_labs, x="Tracking Files", y="Lab ID", color=PALETTE["deep_blue"], ax=axes[0], label="Tracking")
sns.barplot(data=top_labs, x="Annotation Files", y="Lab ID", color=PALETTE["teal"], ax=axes[0], label="Annotation")
axes[0].set_title("Top 15 Labs by File Count", color=PALETTE["slate_gray"], fontsize=14)
axes[0].set_xlabel("Number of Files")
axes[0].set_ylabel("")
axes[0].legend(frameon=False)

sns.barplot(data=bottom_labs, x="Tracking Files", y="Lab ID", color=PALETTE["deep_blue"], ax=axes[1], label="Tracking")
sns.barplot(data=bottom_labs, x="Annotation Files", y="Lab ID", color=PALETTE["teal"], ax=axes[1], label="Annotation")
axes[1].set_title("Bottom 15 Labs by File Count", color=PALETTE["slate_gray"], fontsize=14)
axes[1].set_xlabel("Number of Files")
axes[1].set_ylabel("")
axes[1].legend(frameon=False)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()


all_annotations = []
annotation_stats = []

for _, row in usable_train_rows.iterrows():
    lab_id = row["lab_id"]
    video_id = row["video_id"]
    anno_path = DATA / "train_annotation" / lab_id / f"{video_id}.parquet"
    try:
        df_anno = pd.read_parquet(anno_path)
        df_anno["lab_id"] = lab_id
        df_anno["video_id"] = video_id
        df_anno["duration"] = df_anno["stop_frame"] - df_anno["start_frame"] + 1
        df_anno["is_self_directed"] = df_anno["agent_id"] == df_anno["target_id"]
        all_annotations.append(df_anno)
        annotation_stats.append((lab_id, video_id, len(df_anno)))
    except Exception:
        continue

combined_annotations = pd.concat(all_annotations, ignore_index=True)
stats_df = pd.DataFrame(annotation_stats, columns=["lab_id", "video_id", "annotation_count"])

annotation_overview = {
    "Behavioral Instances": len(combined_annotations),
    "Unique Actions": combined_annotations["action"].nunique(),
    "Videos with Annotations": len(stats_df),
    "Labs with Annotations": combined_annotations["lab_id"].nunique()
}

fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
sizes = list(annotation_overview.values())
labels = [f"{k}\n{v:,}" for k, v in annotation_overview.items()]
colors = [PALETTE["deep_blue"], PALETTE["teal"], "#60A5FA", "#F59E0B"]

wedges, _ = ax.pie(
    sizes,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.45),
    colors=colors
)
ax.set(aspect="equal")
ax.set_title("Behavioral Annotation Overview", color=PALETTE["slate_gray"], fontsize=15, pad=15)
ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.tight_layout()
plt.show()


action_counts = combined_annotations["action"].value_counts()
action_stats = (
    combined_annotations.groupby("action")["duration"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(2)
)
action_stats["percentage"] = (action_stats["count"] / len(combined_annotations) * 100).round(2)
action_stats = action_stats.sort_values("count", ascending=False).reset_index()

top20 = action_stats.head(20)

fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
sns.barplot(
    data=top20,
    x="count",
    y="action",
    palette=sns.color_palette([PALETTE["deep_blue"], PALETTE["teal"]]),
    ax=ax
)
for i, (c, pct) in enumerate(zip(top20["count"], top20["percentage"])):
    ax.text(c + max(top20["count"]) * 0.003, i, f"{c:,} ({pct:.1f}%)",
            va="center", ha="left", fontsize=10, color=PALETTE["slate_gray"])

ax.set_title("Top 20 Most Frequent Actions", color=PALETTE["slate_gray"], fontsize=15, pad=12)
ax.set_xlabel("Number of Instances")
ax.set_ylabel("")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

if "combined_annotations" not in globals():
    if "usable_train_rows" in globals():
        all_annotations = []
        for _, row in usable_train_rows.iterrows():
            lab_id = row["lab_id"]
            video_id = row["video_id"]
            anno_path = DATA / "train_annotation" / lab_id / f"{video_id}.parquet"
            try:
                df_anno = pd.read_parquet(anno_path)
                df_anno["lab_id"] = lab_id
                df_anno["video_id"] = video_id
                df_anno["duration"] = df_anno["stop_frame"] - df_anno["start_frame"] + 1
                df_anno["is_self_directed"] = df_anno["agent_id"] == df_anno["target_id"]
                all_annotations.append(df_anno)
            except Exception:
                continue
        if len(all_annotations) == 0:
            raise NameError("No annotations found when attempting to rebuild combined_annotations.")
        combined_annotations = pd.concat(all_annotations, ignore_index=True)
    else:
        raise NameError("combined_annotations not found in the session. Re-run the annotation-loading cell first.")

sd_counts = combined_annotations["is_self_directed"].value_counts()
social = int(sd_counts.get(False, 0))
self_dir = int(sd_counts.get(True, 0))
total = social + self_dir
social_pct = social / total * 100 if total > 0 else 0.0
self_pct = self_dir / total * 100 if total > 0 else 0.0

summary_html = f"""
<div style="border-left:4px solid {PALETTE['teal']}; padding:12px; font-family:sans-serif; background:#ffffff; max-width:680px;">
  <h3 style="margin:0; color:{PALETTE['deep_blue']};">Self-directed vs Social Behaviors</h3>
  <div style="color:{PALETTE['slate_gray']}; margin-top:8px;">
    <div><strong>Social behaviors (agent != target)</strong>: {social:,} ({social_pct:.1f}%)</div>
    <div style="margin-top:6px;"><strong>Self-directed behaviors (agent == target)</strong>: {self_dir:,} ({self_pct:.1f}%)</div>
    <div style="margin-top:8px; color:{PALETTE['muted']}; font-size:90%;">Total behavioral instances: {total:,}</div>
  </div>
</div>
"""
display(HTML(summary_html))

fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
sizes = [social, self_dir]
labels = [f"Social\n{social:,}\n{social_pct:.1f}%", f"Self-directed\n{self_dir:,}\n{self_pct:.1f}%"]
colors = [PALETTE["deep_blue"], PALETTE["teal"]]

wedges, texts = ax.pie(sizes, startangle=90, counterclock=False, wedgeprops=dict(width=0.45), colors=colors)
ax.set(aspect="equal")
ax.set_title("Social vs Self-directed Behavioral Instances", color=PALETTE["slate_gray"], fontsize=14, pad=12)
ax.legend(wedges, labels, title="", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.tight_layout()
plt.show()


# Compute duration statistics per action
duration_stats = (
    combined_annotations.groupby("action")["duration"]
    .agg(["mean", "median", "std"])
    .round(1)
    .sort_values("mean", ascending=False)
)

# Prepare data for top 15 actions
top_actions = duration_stats.head(15).index.tolist()
plot_df = combined_annotations[combined_annotations["action"].isin(top_actions)].copy()
plot_df = plot_df[plot_df["duration"] > 0]

# Plot distributions
plt.figure(figsize=(12, 8), dpi=120)
ax = sns.boxplot(
    data=plot_df,
    x="duration",
    y="action",
    order=top_actions,
    showfliers=False,
    width=0.6,
    palette=[PALETTE["teal"] if i % 2 == 0 else PALETTE["deep_blue"] for i in range(len(top_actions))]
)
ax.set_xscale("log")
ax.set_xlabel("Duration (frames) — log scale", color=PALETTE["slate_gray"])
ax.set_ylabel("")
ax.set_title("Action Duration Distributions — Top 15 by Mean Duration", color=PALETTE["slate_gray"], fontsize=15, pad=12)

# annotate medians
medians = plot_df.groupby("action")["duration"].median().reindex(top_actions)
x_off = plot_df["duration"].median() * 0.03
for i, (act, med) in enumerate(medians.items()):
    ax.text(med + x_off, i, f"{int(med):,}", va="center", ha="left", color=PALETTE["slate_gray"], fontsize=9)

plt.tight_layout()
plt.show()


# Build lab-action availability matrix
lab_action_matrix = combined_annotations.groupby(["lab_id", "action"]).size().unstack(fill_value=0)
lab_action_presence = (lab_action_matrix > 0).astype(int)

top_actions = combined_annotations["action"].value_counts().head(30).index.tolist()
subset = lab_action_presence[top_actions]

plt.figure(figsize=(16, 10), dpi=120)
ax = sns.heatmap(
    subset,
    cmap=sns.color_palette([PALETTE["muted"], PALETTE["teal"]]),
    linewidths=0.3,
    linecolor="#E5E7EB",
    cbar_kws={"label": "Action Presence"}
)

ax.set_title("Cross-Lab Action Availability Heatmap (Top 30 Actions)", color=PALETTE["slate_gray"], fontsize=15, pad=16)
ax.set_xlabel("Actions", color=PALETTE["slate_gray"])
ax.set_ylabel("Labs", color=PALETTE["slate_gray"])
ax.tick_params(axis="x", rotation=65)
plt.tight_layout()
plt.show()


class_imbalance = action_counts / action_counts.sum()
rare_actions = action_counts[action_counts <= 10]
common_actions = action_counts[action_counts >= 1000]

rare_count = len(rare_actions)
common_count = len(common_actions)
most_common = action_counts.iloc[0]
least_common = action_counts.iloc[-1]
imbalance_ratio = most_common / least_common if least_common > 0 else np.nan

fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

# Histogram of action frequencies (log scale)
sns.histplot(action_counts, bins=40, log_scale=(True, False), color=PALETTE["deep_blue"], ax=ax[0])
ax[0].set_title("Distribution of Action Frequencies", color=PALETTE["slate_gray"], fontsize=14, pad=12)
ax[0].set_xlabel("Instances per Action (log scale)")
ax[0].set_ylabel("Number of Actions")
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].grid(axis="y", linestyle="--", alpha=0.35)

# Bar plot for rare vs common classes
ax[1].bar(["Rare (≤10)", "Common (≥1000)"], [rare_count, common_count], color=[PALETTE["teal"], PALETTE["deep_blue"]])
for i, v in enumerate([rare_count, common_count]):
    ax[1].text(i, v + 1, f"{v:,}", ha="center", va="bottom", color=PALETTE["slate_gray"], fontsize=11)

ax[1].set_title("Counts of Rare vs Common Actions", color=PALETTE["slate_gray"], fontsize=14, pad=12)
ax[1].set_ylabel("Number of Actions")
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].grid(axis="y", linestyle="--", alpha=0.35)

plt.suptitle(
    f"Class Imbalance Analysis\nMost common: {most_common:,} • Least common: {least_common:,} • Ratio: {imbalance_ratio:.1f}x",
    color=PALETTE["slate_gray"], fontsize=13, y=1.02
)

plt.tight_layout()
plt.show()


if "usable_train_rows" not in globals():
    anno_root = folders["train_annotation"]
    valid_annos = set()
    if anno_root.exists():
        for lab_dir in sorted(anno_root.iterdir()):
            if not lab_dir.is_dir():
                continue
            for f in lab_dir.glob("*.parquet"):
                valid_annos.add((lab_dir.name, f.stem))
    usable_mask = train_csv.apply(lambda r: (r["lab_id"], r["video_id"]) in valid_annos, axis=1)
    usable_train_rows = train_csv[usable_mask]

sample_labs = list(pd.unique(usable_train_rows["lab_id"]))[:8]
sample_videos = []
for lab in sample_labs:
    lab_videos = usable_train_rows[usable_train_rows["lab_id"] == lab].head(3)
    sample_videos.extend(lab_videos[["lab_id", "video_id"]].values.tolist())

tracking_samples = []
bodypart_inventory = {}
coordinate_stats = []

for lab_id, video_id in sample_videos:
    track_path = DATA / "train_tracking" / lab_id / f"{video_id}.parquet"
    if not track_path.exists():
        continue
    try:
        df_track = pd.read_parquet(track_path)
        df_track = df_track.head(1000).copy()
        df_track["lab_id"] = lab_id
        df_track["video_id"] = video_id
        tracking_samples.append(df_track)
        unique_bodyparts = df_track["bodypart"].unique()
        bodypart_inventory[lab_id] = sorted(unique_bodyparts.tolist())
        coord_stats = df_track[["x", "y"]].describe()
        coordinate_stats.append({
            "lab_id": lab_id,
            "video_id": video_id,
            "x_min": coord_stats.loc["min", "x"],
            "x_max": coord_stats.loc["max", "x"],
            "y_min": coord_stats.loc["min", "y"],
            "y_max": coord_stats.loc["max", "y"],
            "x_mean": coord_stats.loc["mean", "x"],
            "y_mean": coord_stats.loc["mean", "y"],
            "sampled_frames": len(df_track),
            "unique_bodyparts": len(unique_bodyparts)
        })
    except Exception:
        continue

if len(tracking_samples) == 0:
    display(HTML(f"<div style='color:{PALETTE['slate_gray']}'>No tracking samples found for the selected labs/videos.</div>"))
else:
    combined_tracking = pd.concat(tracking_samples, ignore_index=True)
    coord_df = pd.DataFrame(coordinate_stats).sort_values(["lab_id", "video_id"]).reset_index(drop=True)

    coord_df_display = coord_df.copy()
    coord_df_display["x_range"] = (coord_df_display["x_max"] - coord_df_display["x_min"]).round(1)
    coord_df_display["y_range"] = (coord_df_display["y_max"] - coord_df_display["y_min"]).round(1)
    coord_df_display = coord_df_display[[
        "lab_id", "video_id", "sampled_frames", "unique_bodyparts",
        "x_min", "x_max", "x_range", "x_mean",
        "y_min", "y_max", "y_range", "y_mean"
    ]]

    display(HTML(f"""
    <div style="border-left:4px solid {PALETTE['teal']}; padding:10px; margin-bottom:6px;">
      <h3 style="margin:0; color:{PALETTE['deep_blue']}; font-family:sans-serif;">Pose Data Sample — Coordinate Summary</h3>
      <div style="color:{PALETTE['slate_gray']}; margin-top:6px;">
        <strong>Total sampled frames</strong>: {len(combined_tracking):,} &nbsp;•&nbsp;
        <strong>Labs sampled</strong>: {combined_tracking['lab_id'].nunique()} &nbsp;•&nbsp;
        <strong>Videos sampled</strong>: {combined_tracking['video_id'].nunique()} &nbsp;•&nbsp;
        <strong>Unique mice</strong>: {combined_tracking['mouse_id'].nunique()}
      </div>
    </div>
    """))

    display(
        coord_df_display.style.format({
            "sampled_frames":"{:,}",
            "unique_bodyparts":"{:,}",
            "x_min":"{:.1f}", "x_max":"{:.1f}", "x_range":"{:.1f}", "x_mean":"{:.1f}",
            "y_min":"{:.1f}", "y_max":"{:.1f}", "y_range":"{:.1f}", "y_mean":"{:.1f}"
        }).set_table_styles([{"selector":"th","props":[("background-color",PALETTE["muted"]),("color","white")]}])
    )


all_bodyparts = set()
for bodyparts in bodypart_inventory.values():
    all_bodyparts.update(bodyparts)

bodypart_matrix = pd.DataFrame(index=sorted(bodypart_inventory.keys()), columns=sorted(all_bodyparts))
for lab, bodyparts in bodypart_inventory.items():
    for bp in all_bodyparts:
        bodypart_matrix.loc[lab, bp] = 1 if bp in bodyparts else 0
bodypart_matrix = bodypart_matrix.astype(int)

total_bodyparts = len(all_bodyparts)
per_lab_counts = bodypart_matrix.sum(axis=1)
min_bp, max_bp = int(per_lab_counts.min()), int(per_lab_counts.max())

plt.figure(figsize=(14, max(4, 0.25 * bodypart_matrix.shape[0])), dpi=120)
cmap = sns.color_palette([PALETTE["muted"], PALETTE["teal"]])
ax = sns.heatmap(
    bodypart_matrix,
    cmap=cmap,
    cbar=False,
    linewidths=0.25,
    linecolor="#E6EEF2"
)
ax.set_title(
    f"Bodypart Tracking Overview — {total_bodyparts} unique bodyparts "
    f"(per lab range: {min_bp}–{max_bp})",
    color=PALETTE["slate_gray"], fontsize=14, pad=12
)
ax.set_xlabel("Bodyparts", color=PALETTE["slate_gray"])
ax.set_ylabel("Labs", color=PALETTE["slate_gray"])
ax.tick_params(axis="x", rotation=65)
plt.tight_layout()
plt.show()



common_bodyparts = bodypart_matrix.sum(axis=0).sort_values(ascending=False)
top15_bodyparts = common_bodyparts.head(15).reset_index()
top15_bodyparts.columns = ["Bodypart", "Lab Count"]

plt.figure(figsize=(12, 6), dpi=120)
ax = sns.barplot(
    data=top15_bodyparts,
    x="Lab Count", y="Bodypart",
    palette=[PALETTE["teal"]] * len(top15_bodyparts)
)

for i, v in enumerate(top15_bodyparts["Lab Count"]):
    ax.text(v + 0.3, i, f"{v}", va="center", ha="left", color=PALETTE["slate_gray"], fontsize=10)

ax.set_title("Most Commonly Tracked Bodyparts (Top 15)", color=PALETTE["slate_gray"], fontsize=14, pad=12)
ax.set_xlabel("Number of Labs Tracking")
ax.set_ylabel("")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 7), dpi=120)

for i, row in coord_df.iterrows():
    plt.plot([row["x_min"], row["x_max"]], [i, i], color=PALETTE["deep_blue"], linewidth=2)
    plt.scatter(row["x_min"], i, color=PALETTE["teal"], s=50, zorder=3)
    plt.scatter(row["x_max"], i, color=PALETTE["teal"], s=50, zorder=3)

plt.yticks(range(len(coord_df)), coord_df["lab_id"])
plt.xlabel("X-coordinate Range (pixels)", color=PALETTE["slate_gray"])
plt.title("Coordinate System Analysis — X-axis Ranges by Lab", color=PALETTE["slate_gray"], fontsize=14, pad=12)
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=120, constrained_layout=True)

# 1) Arena centers by lab (mean x, mean y)
ax = axes[0]
palette_lab = sns.color_palette("tab20", n_colors=len(coord_df))
sc = ax.scatter(coord_df["x_mean"], coord_df["y_mean"], c=range(len(coord_df)), cmap="tab20", s=90, alpha=0.85)
for i, row in coord_df.reset_index().iterrows():
    ax.annotate(row["lab_id"][:12], (row["x_mean"], row["y_mean"]), xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=PALETTE["slate_gray"])
ax.set_xlabel("Mean X Coordinate", color=PALETTE["slate_gray"])
ax.set_ylabel("Mean Y Coordinate", color=PALETTE["slate_gray"])
ax.set_title("Arena Centers by Lab", color=PALETTE["slate_gray"], fontsize=13, pad=8)

# 2) Arena dimensions by lab (range x, range y)
ax = axes[1]
x_range = coord_df["x_max"] - coord_df["x_min"]
y_range = coord_df["y_max"] - coord_df["y_min"]
ax.scatter(x_range, y_range, c=range(len(coord_df)), cmap="tab20", s=90, alpha=0.85)
for i, row in coord_df.reset_index().iterrows():
    xr = row["x_max"] - row["x_min"]
    yr = row["y_max"] - row["y_min"]
    ax.annotate(row["lab_id"][:10], (xr, yr), xytext=(6, 4), textcoords="offset points", fontsize=8, color=PALETTE["slate_gray"])
ax.set_xlabel("X Range (px)", color=PALETTE["slate_gray"])
ax.set_ylabel("Y Range (px)", color=PALETTE["slate_gray"])
ax.set_title("Arena Dimensions by Lab", color=PALETTE["slate_gray"], fontsize=13, pad=8)

# 3) Coordinate distribution sample (pose points colored by lab)
ax = axes[2]
sample_n = min(12000, len(combined_tracking))
sample_df = combined_tracking.sample(sample_n, random_state=42)
labs = sample_df["lab_id"].unique()
pal = sns.color_palette("tab20", n_colors=len(labs))
lab_to_color = {lab: pal[i % len(pal)] for i, lab in enumerate(labs)}
ax.scatter(sample_df["x"], sample_df["y"], s=6,
           c=[lab_to_color[l] for l in sample_df["lab_id"].values], alpha=0.6)
ax.invert_yaxis()
ax.set_xlabel("x (pixels)", color=PALETTE["slate_gray"])
ax.set_ylabel("y (pixels)", color=PALETTE["slate_gray"])
ax.set_title("Coordinate Distribution Sample (colored by lab)", color=PALETTE["slate_gray"], fontsize=13, pad=8)
# compact legend: show only top 8 labs to avoid clutter
top_labs_for_legend = list(pd.Series(sample_df["lab_id"]).value_counts().head(8).index)
handles = [plt.Line2D([], [], marker="o", linestyle="", color=lab_to_color[l], markersize=6) for l in top_labs_for_legend]
ax.legend(handles, top_labs_for_legend, title="Top labs", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

plt.suptitle("Coordinate System Summary — centers, dimensions, and sampled point cloud", color=PALETTE["slate_gray"], fontsize=14, y=1.02)
plt.show()


if "sample_videos" not in globals():
    sample_labs = list(pd.unique(usable_train_rows["lab_id"]))[:8]
    sample_videos = []
    for lab in sample_labs:
        lab_videos = usable_train_rows[usable_train_rows["lab_id"] == lab].head(3)
        sample_videos.extend(lab_videos[["lab_id", "video_id"]].values.tolist())

temporal_analysis = []
for lab_id, video_id in sample_videos[:10]:
    track_path = DATA / "train_tracking" / lab_id / f"{video_id}.parquet"
    if not track_path.exists():
        continue
    try:
        df = pd.read_parquet(track_path)
        df_pivot = df.pivot_table(index="video_frame", columns=["mouse_id", "bodypart"], values=["x", "y"])
        if len(df_pivot) > 100:
            sample_coords = df_pivot.iloc[:100, :4].values.astype(float)
            velocities = np.diff(sample_coords, axis=0)
            vel_magnitude = np.sqrt(np.nansum(velocities**2, axis=1))
            temporal_analysis.append({
                "lab_id": lab_id,
                "video_id": video_id,
                "mean_velocity": float(np.nanmean(vel_magnitude)),
                "max_velocity": float(np.nanmax(vel_magnitude)),
                "velocity_std": float(np.nanstd(vel_magnitude))
            })
    except Exception:
        continue

temporal_df = pd.DataFrame(temporal_analysis)
if temporal_df.empty:
    raise ValueError("No temporal samples found for the chosen videos. Re-run sampling or choose different videos.")

temporal_df = temporal_df.round({
    "mean_velocity": 2,
    "max_velocity": 2,
    "velocity_std": 2
}).sort_values(["mean_velocity", "max_velocity"], ascending=[False, False]).reset_index(drop=True)

display(
    temporal_df.style.background_gradient(
        subset=["mean_velocity", "max_velocity", "velocity_std"], cmap="YlGnBu"
    ).set_caption("Temporal Consistency Analysis — Velocity Statistics")
)


if "missing_summary" not in globals():
    # ensure usable_train_rows exists
    if "usable_train_rows" not in globals():
        anno_root = folders["train_annotation"]
        valid_annos = set()
        if anno_root.exists():
            for lab_dir in sorted(anno_root.iterdir()):
                if not lab_dir.is_dir():
                    continue
                for f in lab_dir.glob("*.parquet"):
                    valid_annos.add((lab_dir.name, f.stem))
        usable_mask = train_csv.apply(lambda r: (r["lab_id"], r["video_id"]) in valid_annos, axis=1)
        usable_train_rows = train_csv[usable_mask]

    # ensure combined_tracking exists (sample if necessary)
    if "combined_tracking" not in globals():
        sample_labs = list(pd.unique(usable_train_rows["lab_id"]))[:8]
        sample_videos = []
        for lab in sample_labs:
            lab_videos = usable_train_rows[usable_train_rows["lab_id"] == lab].head(3)
            sample_videos.extend(lab_videos[["lab_id", "video_id"]].values.tolist())

        tracking_samples = []
        for lab_id, video_id in sample_videos:
            track_path = DATA / "train_tracking" / lab_id / f"{video_id}.parquet"
            if not track_path.exists():
                continue
            try:
                df_track = pd.read_parquet(track_path)
                df_track = df_track.head(1000).copy()
                df_track["lab_id"] = lab_id
                df_track["video_id"] = video_id
                tracking_samples.append(df_track)
            except Exception:
                continue

        if len(tracking_samples) == 0:
            raise RuntimeError("No tracking samples found to compute missing_summary. Ensure tracking files are available.")
        combined_tracking = pd.concat(tracking_samples, ignore_index=True)

    # groupby + apply with include_groups=False to silence deprecation warning
    missing_analysis = combined_tracking.groupby(['lab_id', 'mouse_id', 'bodypart']).apply(
        lambda x: pd.Series({
            'total_frames': len(x),
            'missing_x': int(x['x'].isnull().sum()),
            'missing_y': int(x['y'].isnull().sum()),
            'missing_either': int((x['x'].isnull() | x['y'].isnull()).sum())
        }),
        include_groups=False
    ).reset_index()

    # aggregate to lab-level
    missing_summary = missing_analysis.groupby('lab_id').agg({
        'missing_either': ['sum', 'mean'],
        'total_frames': 'sum'
    }).round(3)
    missing_summary.columns = ['total_missing', 'avg_missing_rate', 'total_frames']
    missing_summary['missing_percentage'] = (missing_summary['total_missing'] / missing_summary['total_frames'] * 100).round(2)
    missing_summary = missing_summary.reset_index()

# compute KPIs using coord_df and missing_summary (assumes coord_df exists)
coord_range_variation = coord_df["x_max"].max() - coord_df["x_min"].min()
arena_cv = ((coord_df["x_max"] - coord_df["x_min"]).std() /
            (coord_df["x_max"] - coord_df["x_min"]).mean()) if (coord_df["x_max"] - coord_df["x_min"]).mean() != 0 else np.nan
missing_min, missing_max = missing_summary["missing_percentage"].min(), missing_summary["missing_percentage"].max()

from IPython.display import HTML, display
display(HTML(f"""
<div style="border-left:4px solid {PALETTE['teal']}; padding:12px; background:#ffffff; font-family:sans-serif; max-width:720px;">
  <h3 style="margin:0; color:{PALETTE['deep_blue']};">Summary Statistics</h3>
  <ul style="margin-top:8px; color:{PALETTE['slate_gray']}; line-height:1.6;">
    <li><strong>Coordinate range variation</strong>: {coord_range_variation:.0f} px (X-axis span across labs)</li>
    <li><strong>Arena size consistency</strong>: CV = {arena_cv:.2f}</li>
    <li><strong>Missing data rate</strong>: {missing_min:.1f}% – {missing_max:.1f}% across labs</li>
  </ul>
</div>
"""))




