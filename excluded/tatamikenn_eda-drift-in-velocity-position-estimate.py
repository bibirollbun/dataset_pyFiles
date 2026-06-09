%load_ext autoreload
%autoreload 2


import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def moving_average(data, window_size=5, pad_mode="reflect"):
    """
    Moving average with customizable padding (e.g., 'edge', 'reflect', 'symmetric').

    Parameters:
        data : ndarray (1D or 2D)
        window_size : int
        pad_mode : str, padding mode for np.pad (e.g., 'edge', 'reflect', 'symmetric')

    Returns:
        Smoothed data, same shape as input.
    """
    data = np.asarray(data)
    pad = window_size // 2

    if data.ndim == 1:
        padded = np.pad(data, pad_width=pad, mode=pad_mode)  # type: ignore
        kernel = np.ones(window_size) / window_size
        return np.convolve(padded, kernel, mode="valid")

    elif data.ndim == 2:
        padded = np.pad(data, ((pad, pad), (0, 0)), mode=pad_mode)  # type: ignore
        kernel = np.ones(window_size) / window_size
        return np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="valid"), axis=0, arr=padded
        )
    else:
        raise ValueError("data must be 1D or 2D")


def plot_in_3d(x, title="3D Acceleration Trajectory", unit="m/sÂ²"):
    n_samples = x.shape[0]

    # ç·šåˆ†ç”Ÿæˆ�ï¼ˆ2ç‚¹ã�”ã�¨ã�®ã�¤ã�ªã��ï¼‰
    points = x.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # ã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—ç”¨ã�®æ­£è¦�åŒ–
    norm = Normalize(vmin=0, vmax=n_samples - 1)
    colors = cm.viridis(norm(np.arange(n_samples - 1)))  # type: ignore

    # 3Dç·šåˆ†ã�«è‰²ã‚’ã�¤ã�‘ã‚‹
    lc = Line3DCollection(segments, colors=colors, linewidth=1)

    # æ��ç”»
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(lc)  # type: ignore
    # è»¸è¨­å®š
    ax.set_xlim(np.nanmin(x[:, 0]), np.nanmax(x[:, 0]))
    ax.set_ylim(np.nanmin(x[:, 1]), np.nanmax(x[:, 1]))
    ax.set_zlim(np.nanmin(x[:, 2]), np.nanmax(x[:, 2]))  # type: ignore
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit}")  # type: ignore
    ax.set_aspect("equal")
    ax.set_title(title)

    # ã‚«ãƒ©ãƒ¼ãƒ�ãƒ¼è¿½åŠ 
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)  # type: ignore
    sm.set_array([])  # ãƒ€ãƒŸãƒ¼ãƒ‡ãƒ¼ã‚¿ï¼ˆmatplotlibä»•æ§˜ï¼‰
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Time step index")

    plt.show()


def correct_quaternion_sign_consistency(q):
    """
    q: (N, 4) ndarray of quaternions [x, y, z, w]
    """
    q_corrected = np.copy(q)
    for i in range(1, len(q)):
        if np.dot(q_corrected[i], q_corrected[i - 1]) < 0:
            q_corrected[i] *= -1
    return q_corrected


def plot_feature(feature, names, hand_at_target_index, gestures_start_index):
    _, ax = plt.subplots(figsize=(5, 3))
    for i in range(feature.shape[1]):
        ax.plot(feature[:, i], label=names[i])
    ax.axvspan(
        0,
        hand_at_target_index,
        color="red",
        alpha=0.1,
        label="Transition Phase",
    )
    ax.axvspan(
        hand_at_target_index,
        gestures_start_index,
        color="green",
        alpha=0.1,
        label="Initial Phase",
    )
    ax.axvspan(
        gestures_start_index,
        len(feature),
        color="blue",
        alpha=0.1,
        label="Gesture Phase",
    )
    ax.set(
        title="Feature Visualization",
        xlabel="Time step",
        ylabel="Feature Value",
    )
    ax.grid()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def visualize_sequence(
    train_df, sequence_id, dt=0.1, window_size=21, zero_reset="none"
):
    single_sequence = train_df.filter(pl.col("sequence_id") == sequence_id)

    print(
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
    quat = single_sequence.select("rot_x", "rot_y", "rot_z", "rot_w").to_numpy()

    quat = correct_quaternion_sign_consistency(quat)

    try:
        # convert sensor coord to world coord
        rot = R.from_quat(quat)
        acc_world = rot.apply(acc)
    except Exception as e:
        print(f"Error converting sensor coordinates to world coordinates: {e}")
        acc_world = np.zeros_like(acc)

    x_gesture = (single_sequence["phase"] == "Gesture").to_numpy().astype(np.float32)
    gestures_start_index = np.where(x_gesture == 1)[0][0]
    hand_at_target_index = np.where(
        single_sequence["behavior"] == "Hand at target location"
    )[0][0]

    print(f"Hand at target index: {hand_at_target_index}")
    print(f"Gesture start index: {gestures_start_index}")

    gravity = np.array([0, 0, 9.81])
    acc_linear = acc_world - gravity

    acc_abs = np.linalg.norm(acc_linear, axis=1, keepdims=True)

    if zero_reset == "target_location":
        # Reset to zero at the target location
        acc_linear[: hand_at_target_index + 1] = 0
        acc_abs[: hand_at_target_index + 1] = 0
    elif zero_reset == "gesture_start":
        # Reset to zero at the gesture start
        acc_linear[: gestures_start_index + 1] = 0
        acc_abs[: gestures_start_index + 1] = 0
    elif zero_reset == "none":
        # Do not reset to zero
        pass
    else:
        raise ValueError(
            "zero_reset must be 'target_location', 'gesture_start', or 'none'"
        )

    velocity_linear = np.cumsum(acc_linear * dt, axis=0)
    velocity_ma = moving_average(velocity_linear, window_size=window_size)
    velocity_abs = np.linalg.norm(velocity_linear, axis=1, keepdims=True)

    position_linear = np.cumsum(velocity_linear * dt, axis=0)
    position_ma = moving_average(position_linear, window_size=window_size)
    position_abs = np.linalg.norm(position_linear, axis=1, keepdims=True)

    plot_feature(
        np.concatenate([acc_linear, acc_abs], axis=1),
        ("$a_x$", "$a_y$", "$a_z$", "abs $a$"),
        hand_at_target_index,
        gestures_start_index,
    )
    # plot velocity
    plot_feature(
        np.concatenate([velocity_linear, velocity_abs, velocity_ma], axis=1),
        ("$v_x$", "$v_y$", "$v_z$", "abs $v$", "MA $v_x$", "MA $v_y$", "MA $v_z$"),
        hand_at_target_index,
        gestures_start_index,
    )
    plot_feature(
        np.concatenate([position_linear, position_abs, position_ma], axis=1),
        ("$p_x$", "$p_y$", "$p_z$", "abs $p$", "MA $p_x$", "MA $p_y$", "MA $p_z$"),
        hand_at_target_index,
        gestures_start_index,
    )

    plot_in_3d(acc_linear, title="acc_linear", unit="m/sÂ²")
    plot_in_3d(velocity_linear, title="velocity_linear", unit="m/s")
    plot_in_3d(position_linear, title="position_linear", unit="m")



from pathlib import Path
import polars as pl


data_dir = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")


NUMERIC_COLUMNS = [
    *[f"thm_{i}" for i in range(1, 6)],
    *[f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)],
]
train_df = pl.read_csv(data_dir / "train.csv").with_columns(
    pl.col(c).replace(-1, None) for c in NUMERIC_COLUMNS if c.startswith("tof_")
)
train_df.head()


def sanitize_col_name(s):
    return s.lower().replace(" ", "_")


sequence_meta_df = train_df.group_by("sequence_id", maintain_order=True).agg(
    *[
        pl.col(c).first()
        for c in [
            "sequence_type",
            "subject",
            "orientation",
            "gesture",
        ]
    ]
)
sequence_meta_df


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

for gesture in gesture_to_type.keys():
    sequence_id = sequence_meta_df.filter(pl.col("gesture") == gesture).sample(
        1, seed=2, with_replacement=False
    )["sequence_id"][0]
    display(Markdown(f"### ğŸ“Š {gesture=}, {sequence_id=}"))
    visualize_sequence(
        train_df, sequence_id, window_size=7, zero_reset="none"
    )
    break


from IPython.display import Markdown

for gesture in gesture_to_type.keys():
    sequence_id = sequence_meta_df.filter(pl.col("gesture") == gesture).sample(
        1, seed=2, with_replacement=False
    )["sequence_id"][0]
    display(Markdown(f"### ğŸ“Š {gesture=}, {sequence_id=}"))
    visualize_sequence(
        train_df, sequence_id, window_size=7, zero_reset="target_location"
    )
    break


from IPython.display import Markdown

for gesture in gesture_to_type.keys():
    sequence_id = sequence_meta_df.filter(pl.col("gesture") == gesture).sample(
        1, seed=2, with_replacement=False
    )["sequence_id"][0]
    display(Markdown(f"### ğŸ“Š {gesture=}, {sequence_id=}"))
    visualize_sequence(
        train_df, sequence_id, window_size=7, zero_reset="gesture_start"
    )
    break

