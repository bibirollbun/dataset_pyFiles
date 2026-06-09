from pathlib import Path


data_dir = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")


!ls -l {data_dir}


NUMERIC_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
    *[f"thm_{i}" for i in range(1, 6)],
    *[f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)],
]


import polars as pl

train_df = pl.read_csv(data_dir / "train.csv").with_columns(
    pl.col(c).replace(-1, None) for c in NUMERIC_COLUMNS if c.startswith("tof_")
)
train_df.head()


print(train_df.columns)


train_df["subject"].n_unique()


train_df["sequence_id"].n_unique()


train_df["gesture"].n_unique()


train_df.group_by("subject").agg(
    pl.col("sequence_id").n_unique().alias("unique_sequences")
).sort("unique_sequences", descending=True).to_pandas().plot(
    kind="bar",
    x="subject",
    y="unique_sequences",
    title="Number of Unique Sequences per Subject",
    xlabel="Subject",
    ylabel="Unique Sequences",
    figsize=(16, 3),
)


train_df.group_by("sequence_id").agg(
    pl.col("sequence_counter").n_unique().alias("unique_counters")
).sort("unique_counters", descending=True).to_pandas().plot(
    kind="hist", bins=100,
    title="Distribution of Unique Sequence Counters",
    xlabel="Unique Sequence Counters",
    ylabel="Frequency",
)


train_df.group_by("phase").agg(
    pl.col("sequence_counter").n_unique().alias("unique_counters")
).sort("unique_counters", descending=True).to_pandas().plot(
    kind="bar",
    x="phase",
    y="unique_counters",
    title="Number of Unique Sequence Counters per Phase",
    xlabel="Phase",
    ylabel="Unique Sequence Counters",
    figsize=(8, 4),
)


GESTURES = sorted(train_df["gesture"].unique().to_list())
GESTURES


PHASES = sorted(train_df["phase"].unique().to_list())
PHASES


BEHAVIORS = sorted(train_df["behavior"].unique().to_list())
BEHAVIORS


ORIENTATIONS = sorted(train_df["orientation"].unique().to_list())
ORIENTATIONS


train_df.group_by("sequence_id").agg(pl.col("sequence_type").first()).group_by(
    "sequence_type"
).agg(pl.len().alias("count")).sort("count", descending=True)


cols = train_df.columns
df = train_df.group_by("sequence_id").agg(
    *[pl.all().n_unique()] + [pl.col("sequence_counter").len().alias("seq_len")]
)
display(df)

per_sequence_cols = []
for col  in cols:
    if col not in ["sequence_id", "sequence_counter"]:
        if (df[col] == 1).all():
            per_sequence_cols.append(col)

print("Columns with only one unique value per sequence:")
for col in per_sequence_cols:
    print(f"- {col}")


def sanitize_col_name(s):
    return s.lower().replace(" ", "_")


sequence_meta_df = (
    train_df.group_by("sequence_id", maintain_order=True)
    .agg(
        *[
            pl.col(c).first()
            for c in [
                "sequence_type",
                "subject",
                "orientation",
                "gesture",
            ]
        ]
        + [
            pl.col("phase").n_unique().alias("diversity_phase"),
            pl.col("behavior").n_unique().alias("diversity_behavior"),
        ]
        + [
            pl.col("sequence_counter").len().alias("seq_len"),
            *[
                (pl.col("phase") == p)
                .sum()
                .alias(f"count_phase_{sanitize_col_name(p)}")
                for p in PHASES
            ],
            *[
                (pl.col("behavior") == b)
                .sum()
                .alias(f"count_behavior_{sanitize_col_name(b)}")
                for b in BEHAVIORS
            ],
        ]
    )
    .with_columns(
        [
            *[
                (
                    pl.col(f"count_phase_{sanitize_col_name(p)}") / pl.col("seq_len")
                ).alias(f"ratio_phase_{sanitize_col_name(p)}")
                for p in PHASES
            ],
            *[
                (
                    pl.col(f"count_behavior_{sanitize_col_name(b)}") / pl.col("seq_len")
                ).alias(f"ratio_behavior_{sanitize_col_name(b)}")
                for b in BEHAVIORS
            ],
        ]
    )
)
print(sequence_meta_df.height)
sequence_meta_df.head()


sequence_meta_df.columns


sequence_meta_df.filter(pl.col("ratio_behavior_moves_hand_to_target_location") == 0).height


sequence_meta_df.filter(pl.col("ratio_behavior_relaxes_and_moves_hand_to_target_location") == 0).height


sequence_meta_df.filter(
    (pl.col("ratio_behavior_moves_hand_to_target_location") == 0)
    & (pl.col("ratio_behavior_relaxes_and_moves_hand_to_target_location") == 0)
).height


target_sequence_meta_df = sequence_meta_df.filter(pl.col("sequence_type") == "Target")
non_target_sequence_meta_df = sequence_meta_df.filter(
    pl.col("sequence_type") == "Non-Target"
)


import matplotlib.pyplot as plt

_, ax = plt.subplots()
target_sequence_meta_df["gesture"].value_counts().sort("count", descending=True).to_pandas().plot(
    kind="barh",
    x="gesture",
    y="count",
    title="Gesture Counts (Target)",
    xlabel="Gesture",
    ylabel="Count",
    figsize=(8, 4),
    ax=ax,
    color="blue",
    alpha=0.7,
    label="Target",
)
_, ax = plt.subplots()
non_target_sequence_meta_df["gesture"].value_counts().sort("count", descending=True).to_pandas().plot(
    kind="barh",
    x="gesture",
    y="count",
    title="Gesture Counts (Non-Target)",
    xlabel="Gesture",
    ylabel="Count",
    figsize=(8, 4),
    ax=ax,
    color="orange",
    alpha=0.7,
    label="Non-Target",
)


sequence_meta_df["diversity_behavior"].value_counts().sort("count", descending=True)


sequence_meta_df.filter(pl.col("diversity_behavior") == 2)


sequence_meta_df["diversity_phase"].value_counts().sort("count", descending=True)


sequence_meta_df.filter(pl.col("diversity_phase") == 1)


(sequence_meta_df["count_phase_gesture"] / sequence_meta_df["seq_len"]).to_pandas().plot(
    kind="hist", bins=100,
    title="Distribution of Gesture Phase Count per Sequence Length",
    xlabel="Gesture Phase Ratio",
    ylabel="Frequency",
)


sequence_meta_df.columns


import matplotlib.pyplot as plt

for b in BEHAVIORS:
    _, ax = plt.subplots()
    for seq_type, df in zip(
        ["Target", "Non-Target"], [target_sequence_meta_df, non_target_sequence_meta_df]
    ):
        (df[f"count_behavior_{sanitize_col_name(b)}"] / df["seq_len"]).to_pandas().plot(
            kind="hist",
            bins=100,
            title=f"Distribution of `{b}`",
            xlabel="Behavior Ratio",
            ylabel="Frequency",
            alpha=0.5,
            ax=ax,
            xlim=(0, 1),
            label=seq_type,
            color="blue" if seq_type == "Target" else "orange",
            legend=True,
        )


import matplotlib.pyplot as plt

b1, b2 = "Moves Hand to Target Location", "Relaxes and Moves Hand to Target Location"
_, ax = plt.subplots()
for seq_type, df in zip(
    ["Target", "Non-Target"], [target_sequence_meta_df, non_target_sequence_meta_df]
):
    (
        (
            df[f"count_behavior_{sanitize_col_name(b1)}"]
            + df[f"count_behavior_{sanitize_col_name(b2)}"]
        )
        / df["seq_len"]
    ).to_pandas().plot(
        kind="hist",
        bins=100,
        title=f"Distribution of `{b1}` and `{b2}` Combined",
        xlabel="Behavior Ratio",
        ylabel="Frequency",
        alpha=0.5,
        ax=ax,
        xlim=(0, 1),
        label=seq_type,
        color="blue" if seq_type == "Target" else "orange",
        legend=True,
    )


train_df["behavior"].unique().to_list()


import numpy as np
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R, Slerp


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def visualize_tof(tof1, tof2, tof3, tof4, tof5, x_gesture, behavior_dict):
    for name, x in zip(
        ["tof1", "tof2", "tof3", "tof4", "tof5"], [tof1, tof2, tof3, tof4, tof5]
    ):
        x = x.copy()
        x = x.reshape(-1, 64)
        fig, ax = plt.subplots(figsize=(8, 3))
        x_max = np.nanmax(x)
        x_min = np.nanmin(x)

        colors = plt.cm.magma(np.linspace(0, 1, x.shape[1]))
        for i in range(x.shape[1]):
            ax.plot(
                x[:, i],
                color=colors[i],
                alpha=0.3,
                linewidth=0.5,
            )

        ax.plot(
            x_gesture * (x_max - x_min) + x_min,
            color="red",
            alpha=0.7,
            linewidth=2,
            label="Gesture Phase",
        )

        colors = ["blue", "orange", "green", "purple", "brown"]
        for i, b in enumerate(BEHAVIORS):
            ax.plot(
                behavior_dict[b] * (x_max - x_min) + x_min,
                label=f"Behavior: {b}",
                linewidth=1,
                alpha=0.7,
                color=colors[i % len(colors)],
                linestyle="--",
            )
        ax.set(
            title=f"{name}",
            xlabel="Sequence Counter",
            ylabel="Value",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()


def visualize_sensor(acc, acc_world, rot, thm, x_gesture, behavior_dict):
    for name, x, channel_names in zip(
        ["acc_sensor", "acc_world", "rot", "trm"],
        [acc, acc_world, rot, thm],
        [
            ["acc_x", "acc_y", "acc_z"],
            ["acc_x", "acc_y", "acc_z"],
            ["rot_x", "rot_y", "rot_z", "rot_w"],
            [f"thm_{i}" for i in range(1, 6)],
        ],
    ):
        x = x.copy()
        fig, ax = plt.subplots(figsize=(8, 3))
        x_max = np.max(x)
        x_min = np.min(x)

        colors = ["blue", "orange", "green", "purple", "brown"]
        for i in range(x.shape[1]):
            ax.plot(
                x[:, i],
                color=colors[i % len(colors)],
                alpha=0.7,
                linewidth=0.5,
                label=channel_names[i],
            )

        ax.plot(
            x_gesture * (x_max - x_min) + x_min,
            color="red",
            alpha=0.7,
            linewidth=2,
            label="Gesture Phase",
        )
        for i, b in enumerate(BEHAVIORS):
            ax.plot(
                behavior_dict[b] * (x_max - x_min) + x_min,
                label=f"Behavior: {b}",
                linewidth=1,
                alpha=0.7,
                color=colors[i % len(colors)],
                linestyle="--",
            )
        ax.set(
            title=f"{name}",
            xlabel="Sequence Counter",
            ylabel="Value",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()


def plot_in_3d(x, title="3D Acceleration Trajectory", unit="m/sÂ²"):
    n_samples = x.shape[0]

    # ç·šåˆ†ç”Ÿæˆ�ï¼ˆ2ç‚¹ã�”ã�¨ã�®ã�¤ã�ªã��ï¼‰
    points = x.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # ã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—ç”¨ã�®æ­£è¦�åŒ–
    norm = Normalize(vmin=0, vmax=n_samples - 1)
    colors = cm.viridis(norm(np.arange(n_samples - 1)))

    # 3Dç·šåˆ†ã�«è‰²ã‚’ã�¤ã�‘ã‚‹
    lc = Line3DCollection(segments, colors=colors, linewidth=1)

    # æ��ç”»
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(lc)
    # è»¸è¨­å®š
    ax.set_xlim(np.nanmin(x[:, 0]), np.nanmax(x[:, 0]))
    ax.set_ylim(np.nanmin(x[:, 1]), np.nanmax(x[:, 1]))
    ax.set_zlim(np.nanmin(x[:, 2]), np.nanmax(x[:, 2]))
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit}")
    ax.set_aspect("equal")
    ax.set_title(title)

    # ã‚«ãƒ©ãƒ¼ãƒ�ãƒ¼è¿½åŠ 
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])  # ãƒ€ãƒŸãƒ¼ãƒ‡ãƒ¼ã‚¿ï¼ˆmatplotlibä»•æ§˜ï¼‰
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Time step index")

    plt.show()


def plot_transparent_sphere(
    ax, center=(0, 0, 0), radius=1.0, color="cyan", alpha=0.3, resolution=100
):
    """
    å�Šé€�æ˜�ã�ªç�ƒä½“ã‚’ 3D ãƒ—ãƒ­ãƒƒãƒˆã�«æ��ç”»ã�™ã‚‹

    Parameters:
        ax        : matplotlib ã�® 3D Axes ã‚ªãƒ–ã‚¸ã‚§ã‚¯ãƒˆ
        center    : ç�ƒä½“ã�®ä¸­å¿ƒåº§æ¨™ (x, y, z)
        radius    : ç�ƒä½“ã�®å�Šå¾„
        color     : ç�ƒä½“ã�®è‰²
        alpha     : é€�æ˜�åº¦ï¼ˆ0.0ã€œ1.0ï¼‰
        resolution: ç·¯åº¦ãƒ»çµŒåº¦ã�®åˆ†å‰²æ•°ï¼ˆç´°ã�‹ã�•ï¼‰
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none")


def plot_in_3d_with_vector(x, title="3D Acceleration Trajectory", step=10):
    n_samples = x.shape[0]

    # ç·šåˆ†ç”Ÿæˆ�ï¼ˆ2ç‚¹ã�”ã�¨ã�®ã�¤ã�ªã��ï¼‰
    points = x.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # ã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—ç”¨ã�®æ­£è¦�åŒ–
    norm = Normalize(vmin=0, vmax=n_samples - 1)
    colors = cm.viridis(norm(np.arange(n_samples - 1)))

    # 3Dç·šåˆ†ã�«è‰²ã‚’ã�¤ã�‘ã‚‹
    lc = Line3DCollection(segments, colors=colors, linewidth=1)

    # æ��ç”»
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(lc)

    for i in range(0, n_samples, step):
        # origin of arrow (e.g., can use acc[i] instead of (0,0,0) if preferred)
        origin = np.array([0, 0, 0])
        direction = x[i]
        ax.quiver(*origin, *direction, normalize=False, color="red", alpha=0.6)

    ax.scatter([0], [0], [0], color="red", s=50, label="Origin")
    plot_transparent_sphere(ax, center=(0, 0, 0), radius=1.0, color="cyan", alpha=0.1)

    # è»¸è¨­å®š
    ax.set(
        xlim=(-1, 1),
        ylim=(-1, 1),
        zlim=(-1, 1),
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        aspect="equal",
        title=title,
    )

    # ã‚«ãƒ©ãƒ¼ãƒ�ãƒ¼è¿½åŠ 
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])  # ãƒ€ãƒŸãƒ¼ãƒ‡ãƒ¼ã‚¿ï¼ˆmatplotlibä»•æ§˜ï¼‰
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Time step index")

    plt.show()


def compute_acc_world_with_slerp(acc_sensor, rot, delay_step=0, fill=np.nan):
    """
    Convert sensor acceleration to world coordinates, correcting for constant delay.

    Parameters:
        acc_sensor: (N, 3) array of acceleration in sensor coordinates
        rot: (M, 4) array of quaternions (x, y, z, w)
        delay_step: constant offset between acc_sensor and rotation (in steps)

    Returns:
        acc_world: (N, 3) array of acceleration in world coordinates
    """
    N = len(acc_sensor)
    M = len(rot)
    rot_times = np.arange(M)
    acc_times = np.arange(N) + delay_step

    # ãƒ�ã‚¹ã‚¯ã�—ã�¦ acc_times ã�Œ rot_times ã�®ç¯„å›²å†…ã�®ã‚‚ã�®ã� ã�‘æ®‹ã�™
    valid_idx = (acc_times >= rot_times[0]) & (acc_times <= rot_times[-1])
    acc_times_valid = acc_times[valid_idx]
    acc_sensor_valid = acc_sensor[valid_idx]

    # è£œé–“ã�¨å›�è»¢é�©ç”¨
    rot_objs = R.from_quat(rot)  # shape: (M,)
    slerp = Slerp(rot_times, rot_objs)
    interp_rots = slerp(acc_times_valid)  # shape: (valid_N,)
    acc_world_valid = interp_rots.apply(acc_sensor_valid)

    # å‡ºåŠ›é…�åˆ—ã‚’æ§‹ç¯‰ï¼ˆNaNã�§åˆ�æœŸåŒ–ã�—ã€�validã�ªä½�ç½®ã�«çµ�æ�œã‚’åŸ‹ã‚�ã‚‹ï¼‰
    acc_world = np.full_like(acc_sensor, fill)
    acc_world[valid_idx] = acc_world_valid

    return acc_world


def visualize_sequence(train_df, sequence_id, delay_step=0):
    single_sequence = train_df.filter(pl.col("sequence_id") == sequence_id)

    display(
        single_sequence.select(
            ["sequence_id", "subject", "orientation", "gesture", "sequence_type"]
        )
        .head(1)
        .to_pandas()
        .T
    )

    tof1 = single_sequence.select(f"tof_1_v{i}" for i in range(64)).to_numpy()
    tof1 = tof1.reshape(-1, 8, 8)
    tof2 = single_sequence.select(f"tof_2_v{i}" for i in range(64)).to_numpy()
    tof2 = tof2.reshape(-1, 8, 8)
    tof3 = single_sequence.select(f"tof_3_v{i}" for i in range(64)).to_numpy()
    tof3 = tof3.reshape(-1, 8, 8)
    tof4 = single_sequence.select(f"tof_4_v{i}" for i in range(64)).to_numpy()
    tof4 = tof4.reshape(-1, 8, 8)
    tof5 = single_sequence.select(f"tof_5_v{i}" for i in range(64)).to_numpy()
    tof5 = tof5.reshape(-1, 8, 8)
    acc = single_sequence.select("acc_x", "acc_y", "acc_z").to_numpy()
    rot = single_sequence.select("rot_x", "rot_y", "rot_z", "rot_w").to_numpy()
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    thm = single_sequence.select(f"thm_{i}" for i in range(1, 6)).to_numpy()
    print(f"rot_norm: mean={np.nanmean(rot_norm):.3f}, std={np.nanstd(rot_norm):.5f}")

    try:
        # convert sensor coord to world coord
        acc_world = compute_acc_world_with_slerp(acc, rot, delay_step=delay_step)
    except Exception as e:
        print(f"Error converting sensor coordinates to world coordinates: {e}")
        acc_world = np.zeros_like(acc)

    try:
        # convert sensor coord to world coord
        r = R.from_quat(rot)
        sensor_z = np.zeros_like(np.ones_like(acc))
        sensor_z[:, 2] = 1.0  # Assuming Z is the vertical axis
        world_z = compute_acc_world_with_slerp(sensor_z, rot)
    except Exception as e:
        print(f"Error converting sensor Z coordinates to world coordinates: {e}")
        world_z = np.zeros_like(np.ones_like(acc))
        world_z[:, 2] = 1.0  # Default to Z-axis

    x_gesture = (single_sequence["phase"] == "Gesture").to_numpy().astype(np.float32)
    gestures_start_index = np.where(x_gesture == 1)[0][0]
    hand_at_target_index = np.where(
        single_sequence["behavior"] == "Hand at target location"
    )[0][0]

    print(f"Hand at target index: {hand_at_target_index}")
    print(f"Gesture start index: {gestures_start_index}")
    behavior_dict = {}

    for b in BEHAVIORS:
        behavior_dict[b] = (
            (single_sequence["behavior"] == b).to_numpy().astype(np.float32)
        )

    print(f"{tof1.shape=}")
    print(f"{tof2.shape=}")
    print(f"{tof3.shape=}")
    print(f"{tof4.shape=}")
    print(f"{tof5.shape=}")
    print(f"{acc.shape=}")
    print(f"{rot.shape=}")
    print(f"{thm.shape=}")

    gravity = np.array([0, 0, 9.81])

    visualize_sensor(acc, acc_world, rot, thm, x_gesture, behavior_dict)
    plot_in_3d(
        acc,
        title="Acceleration Trajectory (Sensor Coordinates)",
    )
    plot_in_3d(
        acc_world - gravity,
        title="Acceleration Trajectory (World Coordinates, Gravity Subtracted)",
    )
    plot_in_3d(
        (acc_world - gravity)[hand_at_target_index:],
        title="Acceleration Trajectory After Moved to the Target Location (World Coordinates, Gravity Subtracted)",
    )
    plot_in_3d_with_vector(world_z, title="Z Trajectory (World Coordinates)")
    plot_in_3d_with_vector(
        world_z[hand_at_target_index:],
        title="Z Trajectory After Moved to the Target Location (World Coordinates)",
    )
    visualize_tof(tof1, tof2, tof3, tof4, tof5, x_gesture, behavior_dict)


for i, g in enumerate(BEHAVIORS):
    print(f"{i}: {g}")


gesture_to_type = {
    d["gesture"]: d["sequence_type"]
    for d in train_df.group_by("gesture")
    .agg(pl.col("sequence_type").eq("Target").first())
    .sort("sequence_type", "gesture")
    .to_dicts()
}

for i, (k, v) in enumerate(gesture_to_type.items()):
    tag = "ğŸš¨" if v else "âœ…"
    print(f"{tag} {i}: {k}: {v}")


from IPython.display import Markdown

for i, (gesture, is_target) in enumerate(gesture_to_type.items()):
    sequence_id = sequence_meta_df.filter(pl.col("gesture") == gesture).sample(
        1, seed=2, with_replacement=False
    )["sequence_id"][0]
    tag = "ğŸš¨" if is_target else "âœ…"
    display(Markdown(f"### {tag} {i}: {gesture=}, {sequence_id=}"))
    visualize_sequence(train_df, sequence_id, delay_step=0)


import polars as pl

train_demo_df = pl.read_csv(data_dir / "train_demographics.csv")
train_demo_df

