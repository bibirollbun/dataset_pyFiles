from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


# Adjust these constants when running on Kaggle. The defaults mirror the path
# requested by the user so copy/paste works out of the box.
TRACKING_PARQUET = Path("/kaggle/input/MABe-mouse-behavior-detection/train_tracking/AdaptableSnail/1212811043.parquet")
ANNOTATION_PARQUET = Path("/kaggle/input/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/1212811043.parquet")
OUTPUT_PNG = Path("behavior_labels_phase_plot.png")
MIN_LABEL_WIDTH = 30
LABEL_STYLE = "lane"  


@dataclass
class TimelineData:
    tracking_frames: pd.Series
    annotation_df: pd.DataFrame
    video_id: str
    lab_id: str


def load_parquet_pair(tracking_path: Path, annotation_path: Path) -> TimelineData:
    tracking_path = tracking_path.resolve()
    annotation_path = annotation_path.resolve()
    if not tracking_path.exists():
        raise FileNotFoundError(f"Tracking parquet not found: {tracking_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation parquet not found: {annotation_path}")

    tracking_frames = pd.read_parquet(tracking_path, columns=["video_frame"])
    annotation_df = pd.read_parquet(annotation_path)
    if annotation_df.empty:
        raise ValueError(f"No annotation rows found in {annotation_path}")

    return TimelineData(
        tracking_frames=tracking_frames["video_frame"],
        annotation_df=annotation_df,
        video_id=tracking_path.stem,
        lab_id=tracking_path.parent.name,
    )

def build_color_map(actions: Iterable[str]) -> Dict[str, tuple]:
    actions_sorted = sorted(set(actions))
    cmap = plt.get_cmap("tab20", len(actions_sorted))
    return {action: cmap(idx) for idx, action in enumerate(actions_sorted)}


def format_mouse_label(agent_id: int, target_id: int) -> str:
    agent_label = f"m{int(agent_id)}"
    target_label = f"m{int(target_id)}"
    if int(agent_id) == int(target_id):
        return agent_label
    return f"{agent_label}->{target_label}"


def draw_interaction_lane(
    lane_ax,
    annotation_df: pd.DataFrame,
    color_map: Dict[str, tuple],
    min_label_width: int,
    pair_labels: List[str],
) -> None:
    if not pair_labels:
        lane_ax.set_visible(False)
        return

    pair_positions = {label: idx for idx, label in enumerate(pair_labels)}
    lane_ax.set_ylim(-0.5, len(pair_labels) - 0.5)
    lane_ax.set_yticks(list(pair_positions.values()))
    lane_ax.set_yticklabels(pair_labels)
    lane_ax.set_ylabel("Mouse Pair")
    lane_ax.grid(axis="x", linestyle="--", alpha=0.3)

    bar_height = 0.6
    for _, row in annotation_df.iterrows():
        start = int(row["start_frame"])
        stop = int(row["stop_frame"])
        duration = stop - start
        if duration <= 0:
            continue
        color = color_map.get(row["action"], "#999999")
        label = format_mouse_label(row["agent_id"], row["target_id"])
        lane_ax.broken_barh(
            [(start, duration)],
            (pair_positions[label] - bar_height / 2, bar_height),
            facecolors=color,
            edgecolors="none",
            alpha=0.8,
        )
        if duration >= max(min_label_width * 2, 80):
            lane_ax.text(
                start + duration / 2,
                pair_positions[label],
                row["action"],
                ha="center",
                va="center",
                fontsize=7,
                color="#111",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.2"),
            )


def annotate_behavior_bars(
    ax,
    row,
    action_positions: Dict[str, int],
    color_map: Dict[str, tuple],
    bar_height: float,
    min_label_width: int,
    label_style: str,
) -> None:
    start = int(row["start_frame"])
    stop = int(row["stop_frame"])
    duration = stop - start
    if duration <= 0:
        return

    center = action_positions[row["action"]]
    ax.broken_barh(
        [(start, duration)],
        (center - bar_height / 2, bar_height),
        facecolors=[color_map[row["action"]]],
        edgecolors="none",
    )

    if duration < min_label_width or label_style in {"lane", "none"}:
        return

    label = format_mouse_label(row["agent_id"], row["target_id"])
    text_x = start + duration / 2
    if label_style == "overlay":
        ax.text(
            text_x,
            center,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2"),
        )
    elif label_style == "top":
        ax.annotate(
            label,
            xy=(text_x, center + bar_height / 2),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222",
            annotation_clip=False,
        )


def render_timeline(data: TimelineData, output_path: Path, min_label_width: int, label_style: str) -> Path:
    total_frames = int(data.tracking_frames.max())
    actions = sorted(data.annotation_df["action"].unique())
    color_map = build_color_map(actions)
    action_positions = {action: idx for idx, action in enumerate(actions)}
    pair_labels = sorted(
        {
            format_mouse_label(int(row["agent_id"]), int(row["target_id"]))
            for _, row in data.annotation_df.iterrows()
        }
    )

    action_panel_height = max(2.8, 0.8 * max(1, len(actions)))
    lane_height = 0.0
    lane_ax = None

    if label_style == "lane":
        lane_height = max(1.2, 0.35 * max(1, len(pair_labels)))
        fig = plt.figure(figsize=(14, action_panel_height + lane_height))
        gs = fig.add_gridspec(2, 1, height_ratios=[action_panel_height, lane_height], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        lane_ax = fig.add_subplot(gs[1], sharex=ax)
        ax.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=(14, action_panel_height))

    bar_height = 0.6
    for _, row in data.annotation_df.iterrows():
        annotate_behavior_bars(
            ax=ax,
            row=row,
            action_positions=action_positions,
            color_map=color_map,
            bar_height=bar_height,
            min_label_width=min_label_width,
            label_style=label_style,
        )

    ax.set_xlim(0, total_frames)
    ax.set_ylim(-1, len(actions))
    ax.set_yticks(list(action_positions.values()))
    ax.set_yticklabels(actions)
    ax.set_ylabel("Behavior Label")
    ax.set_title(f"Behavior Labels for {data.lab_id}/{data.video_id}")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    legend_handles = [Patch(facecolor=color_map[action], label=action) for action in actions]
    ax.legend(handles=legend_handles, title="Behavior", loc="upper right")

    if label_style == "lane" and lane_ax is not None:
        draw_interaction_lane(
            lane_ax=lane_ax,
            annotation_df=data.annotation_df,
            color_map=color_map,
            min_label_width=min_label_width,
            pair_labels=pair_labels,
        )
        lane_ax.set_xlabel("Frame Number")
        fig.subplots_adjust(hspace=0.05)
    else:
        ax.set_xlabel("Frame Number")
        fig.tight_layout()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# === Phase 5: Driver ==========================================================

def main() -> None:
    data = load_parquet_pair(TRACKING_PARQUET, ANNOTATION_PARQUET)
    output_path = render_timeline(
        data=data,
        output_path=OUTPUT_PNG,
        min_label_width=max(1, MIN_LABEL_WIDTH),
        label_style=LABEL_STYLE,
    )
    print(f"Saved phased timeline to {output_path}")


if __name__ == "__main__":
    main()



def main() -> None:
    data = load_parquet_pair(TRACKING_PARQUET, ANNOTATION_PARQUET)
    output_path = render_timeline(
        data=data,
        output_path=OUTPUT_PNG,
        min_label_width=max(1, MIN_LABEL_WIDTH),
        label_style=LABEL_STYLE,
    )
    print(f"Saved phased timeline to {output_path}")


if __name__ == "__main__":
    main()


