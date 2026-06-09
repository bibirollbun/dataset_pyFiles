import os
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict
from scipy.fft import rfft, rfftfreq
from scipy.spatial.transform import Rotation as R

# Set to true if you want to export plots to PDF files.
EXPORT_PLOTS_TO_PDF = False

DATA_DIR = Path("../kaggle/input/cmi-detect-behavior-with-sensor-data").resolve()
PLOT_DIR = Path("../plots").resolve()

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)
    DATA_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data").resolve()
    PLOT_DIR = Path("/kaggle/working/plots").resolve()

PLOT_DIR.mkdir(exist_ok=True)


train_data = pd.read_csv(DATA_DIR.joinpath("train.csv"))
train_demographics_data = pd.read_csv(DATA_DIR.joinpath("train_demographics.csv"))
test_data = pd.read_csv(DATA_DIR.joinpath("test.csv"))
test_demographics_data = pd.read_csv(DATA_DIR.joinpath("test_demographics.csv"))


train_data.head()


def get_filtered_columns(df: pd.DataFrame, redundant_prefixes: set[str]) -> list[str]:
    """Filters out duplicate column groups based on redundant prefixes.

    Args:
        df (pd.DataFrame): Training data frame.
        redundant_prefixes (set[str]): A set of column name prefixes that are considered redundant if they appear more
            than once. Only the first column with each redundant prefix will be kept.

    Returns:
        list[str]: A filtered list of column names, preserving order, with redundant prefix groups reduced to a single
            representative column.
    """
    seen_prefixes: set[str] = set()
    filtered_columns: list[str] = []
    for col in df.columns:
        prefix = col.split("_")[0]
        if prefix in redundant_prefixes:
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
        filtered_columns.append(col)
    return filtered_columns


redundant_prefixes = {"acc", "rot", "thm", "tof"}
filtered_columns = get_filtered_columns(train_data, redundant_prefixes)
train_data[filtered_columns].head()


train_data.tail()


def get_seq_no(df: pd.DataFrame, index: int) -> int:
    """Returns the sequence number from an index of the training data frame.

    Args:
        df (pd.DataFrame): Training data frame.
        index (int): Data frame row index.

    Returns:
        int: Sequence number.
    """
    return int(str(df.iloc[index]["sequence_id"]).removeprefix("SEQ_"))


if train_data.equals(train_data.sort_values(["sequence_id", "sequence_counter"])):
    print(f"train_data is sorted from sequence {get_seq_no(train_data, 0)} to {get_seq_no(train_data, -1)}")

unique_ids = train_data["sequence_id"].drop_duplicates().str.extract(r"SEQ_(\d+)")[0].astype(int).tolist()

if unique_ids != list(range(get_seq_no(train_data, 0), get_seq_no(train_data, -1) + len(unique_ids))):
    print("train_data's sequences are not sequential.")


plt.figure(figsize=(10, 5))

bin_length = 1_000
bins = list(range(0, 65_000, bin_length))
counts, _bin_edges, _patches = plt.hist(unique_ids, bins=bins, edgecolor="black")
counts = pd.Series(counts)

print(
    f"Each bin of length {bin_length} contains an average of {int(counts.mean())} "
    f"({int(counts.mean() / bin_length * 100)}%) sequences with a standard deviation of {int(counts.std())} and "
    f"coefficient of variation of {int(counts.std() / counts.mean() * 100)}%."
)

plt.title("Distribution of Sequence Numbers")
plt.xlabel("Sequence Number")
plt.ylabel("Count")
plt.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()


sequence_lens = train_data.groupby("sequence_id")["sequence_counter"].max()

print(f"Total number of sequences: {len(sequence_lens)}")
print(f"Shortest sequence: {min(sequence_lens)}")
print(f"Longest sequence: {max(sequence_lens)}")

plt.figure(figsize=(10, 5))

bins = list(range(0, 701, 10))
counts, bin_edges, _patches = plt.hist(sequence_lens, bins=bins, edgecolor="black")
max_count = np.max(counts)
max_bin_index = np.argmax(counts)
max_bin_range = (int(bin_edges[max_bin_index]), int(bin_edges[max_bin_index + 1]))

print(
    f"Most common sequence length range {max_bin_range} occurred {int(max_count)} times or "
    f"{int(max_count / sum(counts) * 100)}% of the time."
)

plt.xlabel("Length of sequence")
plt.ylabel("Frequency")
plt.title("Histogram of sequence lengths")
plt.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()


columns_only_in_test = [col for col in test_data.columns if col not in train_data.columns]
columns_only_in_train = [col for col in train_data.columns if col not in test_data.columns]
if not columns_only_in_test:
    print("All columns in the test data exist in the training data.")
print(f"The following columns are ONLY in the training data and not in the test data: {columns_only_in_train}")


train_data["phase"].unique()


PHASE_ORDER = {"Transition": 0, "Gesture": 1}


def does_phase_dec(seq: pd.DataFrame) -> bool:
    """Returns whether the phases for a given sequence decrease (i.e `Gesture` -> `Transition`).

    Args:
        seq (pd.DataFrame): Training data sequence.

    Returns:
        bool: Whether the phases for a given sequence decrease.
    """
    mapped: pd.Series = seq["phase"].map(PHASE_ORDER)
    return not mapped.is_monotonic_increasing


invalid_sequences = train_data.groupby("sequence_id").filter(does_phase_dec)

if invalid_sequences.empty:
    print("All sequences follow the correct Transition → Gesture order.")

missing_phase_info = train_data.groupby("sequence_id")["phase"].apply(lambda phases: set(PHASE_ORDER) - set(phases))

sequences_missing_phases = missing_phase_info[missing_phase_info.apply(bool)]

for seq_id, missing in sequences_missing_phases.items():
    print(f"Sequence {seq_id} is missing: {', '.join(missing)}")


train_data["behavior"].unique()


def behavior_phase_seq(seq: pd.DataFrame) -> Any:
    """Returns all ordered pairs of behavior and phase for a given sequence from the training data.

    Args:
        seq (pd.DataFrame): A sequence from the training data.

    Returns:
        Any: Ordered pairs of behavior and phase for a given sequence.
    """
    unique_pairs = seq[["behavior", "phase"]].drop_duplicates()
    return tuple(map(tuple, unique_pairs.to_numpy()))


behavior_phase_orders = (
    train_data[["sequence_id", "behavior", "phase"]]
    .groupby("sequence_id")
    .apply(behavior_phase_seq, include_groups=False)
)

order_counts = Counter(behavior_phase_orders)

for i, (order, count) in enumerate(order_counts.items(), 1):
    print(f"{i}. {order} → {count} sequences")


train_data[filtered_columns].isnull().sum()


def get_col_groups_missing_data(df: pd.DataFrame) -> set[str]:
    """Returns all column groups that contain missing data (e.g. thermopiles or time-of-flight sensor columns).

    Args:
        df (pd.DataFrame): Training data frame.

    Returns:
        set[str]: Collection of column groups with missing data.
    """
    s = df.isnull().sum()
    nullish_indices = s[s > 0].index.tolist()
    index_prefixes: set[str] = set()

    for i in nullish_indices:
        index_prefixes.add(str(i).split("_")[0])

    return index_prefixes


col_groups_missing_data = get_col_groups_missing_data(train_data)
print(f"The following are all the column groups with rows missing data: {col_groups_missing_data}")


border = "-" * 25
for prefix in col_groups_missing_data:
    print(f"{border}\n{prefix}:\n{border}")
    print(train_data[[col for col in train_data.columns if prefix in col]].isnull().sum())


uniform_flag = True
for prefix in [f"tof_{i}" for i in range(1, 6)]:
    s = train_data[[col for col in train_data.columns if prefix in col]].isnull().sum()
    if s.nunique() != 1:
        uniform_flag = False

if uniform_flag:
    print("The number of rows with missing data for all pixels of a given time-of-flight sensor are uniform.")

for prefix in [f"tof_{i}_v0" for i in range(1, 6)]:
    print(train_data[[col for col in train_data.columns if prefix in col]].isnull().sum())


missing_summary = train_data.isna().groupby(train_data["sequence_id"]).any().any(axis=1)

sequences_with_missing_df = missing_summary[missing_summary].reset_index()
total_num_seqs = len(train_data.groupby(train_data["sequence_id"]))
print(
    f"{len(sequences_with_missing_df)} sequences (or {len(sequences_with_missing_df) / total_num_seqs * 100:.2f}% of "
    f"{total_num_seqs} sequences) have missing data."
)


sensors_by_group = train_data.filter(regex=r"^(rot_w|thm_\d+|tof_\d+_v0)$").columns
missing_mask = train_data[sensors_by_group].isna()

num_seqs_missing_data_per_col = missing_mask.groupby(train_data["sequence_id"]).any().sum()

print("Number of sequences missing data for each column group:")
print(num_seqs_missing_data_per_col)


grouped = missing_mask.groupby(train_data["sequence_id"])

num_seqs_partially_missing_per_col = (grouped.any() & ~grouped.all()).sum()

print("Number of sequences partially missing data per column:")
print(num_seqs_partially_missing_per_col)


# Map all sequences (x-axis) to whether each sensor (y-axis) is missing any data for that sequence.
sequence_missing_map = (missing_mask.groupby(train_data["sequence_id"]).any()).T

# Turn the sequence IDs into integers.
sequence_missing_map.columns = sequence_missing_map.columns.str.extract(r"SEQ_(\d+)")[0].astype(int)

plt.figure(figsize=(12, 6))

# Give each sensor an index and plot the value of that index for each sequence ID for which it has missing data.
for i, sensor in enumerate(sequence_missing_map.index):
    missing_seq_ids = sequence_missing_map.columns[sequence_missing_map.loc[sensor]]
    plt.scatter(missing_seq_ids, [i] * len(missing_seq_ids), s=1, label=sensor)

plt.yticks(ticks=range(len(sequence_missing_map.index)), labels=sequence_missing_map.index.astype(str).to_list())
plt.xlabel("Sequence ID")
plt.ylabel("Sensor")
plt.title("Sensors with Missing Data by Sequence")
plt.grid(True, axis="x", linestyle=":", alpha=0.3)
plt.tight_layout()
plt.show()


train_demographics_data.head()


train_demographics_data.isnull().sum()


def get_unq_subject_ids(series: pd.Series) -> set[int]:
    """Returns the collection of unique subject IDs from the relevant column.

    Args:
        series (pd.Series): The subject IDs column of a data frame.

    Returns:
        set[int]: The collection of unique subject IDs.
    """
    return set(series.str.extract(r"SUBJ_(\d+)", expand=False).dropna().astype(int))


ids_train = get_unq_subject_ids(train_data["subject"])
ids_demo = get_unq_subject_ids(train_demographics_data["subject"])

if ids_train == ids_demo:
    print("Unique subjects in the training table have a bijective map to the demographics table!")


def cv_percent(series: pd.Series) -> float:
    """Returns the coefficient of variation as a percentage for a given feature.

    Args:
        series (pd.Series): A feature from a data frame.

    Returns:
        float: The coefficient of variation.
    """
    return abs(series.std() / series.mean() * 100)


stats_df = train_demographics_data.select_dtypes(include="number").agg(["mean", "std", "min", "max", cv_percent])
stats_df = stats_df.round(2)

print(stats_df)


subject_sequence_counts = (
    train_data[["subject", "sequence_id"]].drop_duplicates().groupby("subject").size().reset_index(name="num_sequences")
)

stats_df = subject_sequence_counts["num_sequences"].agg(["mean", "std", "min", "max", cv_percent])
stats_df = stats_df.round(2)

print(stats_df)


gesture_orientation = train_data[["orientation", "gesture"]].drop_duplicates()
gesture_orientation["gesture_code"] = gesture_orientation["gesture"].astype("category").cat.codes
gesture_orientation["orientation_code"] = gesture_orientation["orientation"].astype("category").cat.codes

plt.figure(figsize=(12, 6))

plt.scatter(gesture_orientation["gesture_code"], gesture_orientation["orientation_code"], alpha=0.7, edgecolors="k")

gesture_categories = gesture_orientation["gesture"].astype("category").cat.categories
orientation_categories = gesture_orientation["orientation"].astype("category").cat.categories
plt.xticks(
    ticks=range(len(gesture_categories)),
    labels=gesture_categories.astype(str).to_list(),
    rotation=45,
    ha="right",
    fontsize=7,
)
plt.yticks(ticks=range(len(orientation_categories)), labels=orientation_categories.astype(str).to_list())

plt.xlabel("Gesture")
plt.ylabel("Orientation")
plt.title("Orientations per Gesture")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


train_data["time_norm"] = train_data.groupby("sequence_id")["sequence_counter"].transform(lambda x: x / x.max())
train_data[["row_id", "time_norm"]].head()


class Phase(Enum):
    """The phase of the sequence."""

    TRANSITION = ("Transition",)
    GESTURE = ("Gesture",)
    FULL = ("Transition", "Gesture")


class PlotTimeseriesProps(BaseModel):
    """Properties for plotting timeseries data for a group of sensors in a given sequence phase."""

    train_data: pd.DataFrame
    train_demographics_data: pd.DataFrame
    sequence_id: str
    cols: list[str]
    phase: Phase

    model_config = ConfigDict(arbitrary_types_allowed=True)


def plot_seq_timeseries(props: PlotTimeseriesProps, show_labels: bool = True, show_stats: bool = True) -> Figure:
    """Plots timeseries data for a group of sensors in a given sequence phase.

    Args:
        props (PlotTimeseriesProps): Properties for plotting the timeseries data.
        show_labels (bool, optional): Whether to show labels on the plot. Defaults to True.
        show_stats (bool, optional): Whether to print statistics for the data. Defaults to True.

    Returns:
        Figure: The figure object.
    """
    data = props.train_data[
        (props.train_data["sequence_id"] == props.sequence_id) & (props.train_data["phase"].isin(props.phase.value))
    ]
    subject = str(data.iloc[0]["subject"])
    subj_data = props.train_demographics_data[props.train_demographics_data["subject"] == subject]
    handedness = subj_data.iloc[0]["handedness"]

    title = "\n".join(
        [
            f"Gesture: {data.iloc[0]['gesture']}",
            f"Orientation: {data.iloc[0]['orientation']}",
            f"Phase: {props.phase}",
            f"Handedness: {'Right' if handedness else 'Left'}",
            f"Sequence ID: {props.sequence_id}",
        ]
    )

    if show_stats:
        stats_df = (
            data[props.cols]
            .select_dtypes(include="number")
            .agg(["min", "max", "mean", "std", cv_percent, "skew", "kurtosis"])
        )
        stats_df = stats_df.round(2)
        print(f"Statistics:\n{stats_df}")

    fig, ax = plt.subplots(figsize=(10, 5))

    for col in props.cols:
        if show_labels:
            sns.lineplot(x=data["time_norm"], y=data[col], label=col)
        else:
            sns.lineplot(x=data["time_norm"], y=data[col])

    if show_labels:
        ax.legend()

    if props.phase == Phase.FULL:
        gesture_start_idx = data[data["phase"] == "Gesture"].index[0]
        gesture_start_time = data["time_norm"].loc[gesture_start_idx]
        ax.axvline(x=gesture_start_time, color="gray", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("")
    ax.grid()
    return fig


gesture = "Above ear - pull hair"
orientation = "Seated Straight"
sequence_id = (
    train_data[(train_data["gesture"] == gesture) & (train_data["orientation"] == orientation)]["sequence_id"]
    .drop_duplicates()
    .iloc[30]
)
sequence_id


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)
plt.show()


video_len_s = 5
num_gest_in_video = 6

gesture_data = train_data[(train_data["sequence_id"] == sequence_id) & (train_data["phase"] == "Gesture")]
seq_len = len(train_data[train_data["sequence_id"] == sequence_id])
gesture_len = len(gesture_data)
video_gest_freq = num_gest_in_video / video_len_s
sample_rate = gesture_len / video_len_s

print(
    "\n".join(
        [
            f"Length of video (seconds): {video_len_s}",
            f"Gestures in video: {num_gest_in_video}",
            f"Video gesture frequency (gestures/second): {video_gest_freq}",
            f"Approximate sample rate (samples/second): {sample_rate}",
            f"Total sequence time (seconds): {seq_len / sample_rate:.2f}",
        ]
    )
)


acc_data = cast(pd.DataFrame, train_data[train_data["sequence_id"] == sequence_id]).copy(deep=True)
acc_data["acc_mag"] = np.sqrt(acc_data["acc_x"] ** 2 + acc_data["acc_y"] ** 2 + acc_data["acc_z"] ** 2)

props = PlotTimeseriesProps(
    train_data=acc_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_mag"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)

acc_grav = 9.81
acc_data["acc_mag_no_grav"] = acc_data["acc_mag"] - acc_grav

props = PlotTimeseriesProps(
    train_data=acc_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_mag_no_grav"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)
plt.show()


props = PlotTimeseriesProps(
    train_data=acc_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)
plt.show()


seq_data = cast(pd.DataFrame, train_data[train_data["sequence_id"] == sequence_id]).copy(deep=True)
qw, qx, qy, qz = [seq_data[col] for col in ["rot_w", "rot_x", "rot_y", "rot_z"]]
acc_x, acc_y, acc_z = [seq_data[col] for col in ["acc_x", "acc_y", "acc_z"]]
quaternions = np.stack([qx, qy, qz, qw], axis=1)
acc_body_fixed = np.stack([acc_x, acc_y, acc_z], axis=1)
r = R.from_quat(quaternions)
acc_space_fixed = r.apply(acc_body_fixed)
gravity = np.array([0, 0, 9.81])
acc_no_gravity = acc_space_fixed - gravity
seq_data[["acc_x_nograv", "acc_y_nograv", "acc_z_nograv"]] = acc_no_gravity

props = PlotTimeseriesProps(
    train_data=seq_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_x_nograv", "acc_y_nograv", "acc_z_nograv"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)
plt.show()


gesture = "Above ear - pull hair"
orientation = "Seated Straight"

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)

sequence_id = (
    train_data[(train_data["gesture"] == gesture) & (train_data["orientation"] == orientation)]["sequence_id"]
    .drop_duplicates()
    .iloc[4]
)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)
plt.show()


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)

sequence_id = (
    train_data[(train_data["gesture"] == gesture) & (train_data["orientation"] == orientation)]["sequence_id"]
    .drop_duplicates()
    .iloc[4]
)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_001225",
    cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)
plt.show()


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)

gesture = "Above ear - pull hair"
orientation = "Seated Straight"
sequence_id = (
    train_data[(train_data["gesture"] == gesture) & (train_data["orientation"] == orientation)]["sequence_id"]
    .drop_duplicates()
    .iloc[0]
)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)
plt.show()


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)

gesture = "Above ear - pull hair"
orientation = "Lie on Side - Non Dominant"
sequence_id = (
    train_data[(train_data["gesture"] == gesture) & (train_data["orientation"] == orientation)]["sequence_id"]
    .drop_duplicates()
    .iloc[0]
)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)
plt.show()


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)
plt.show()


train_data["orientation"].drop_duplicates()


gesture = "Above ear - pull hair"
orientation = "Seated Lean Non Dom - FACE DOWN"
sequence_id = (
    train_data[(train_data["gesture"] == gesture) & (train_data["orientation"] == orientation)]["sequence_id"]
    .drop_duplicates()
    .iloc[0]
)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)
plt.show()


gesture = "Above ear - pull hair"
orientation = "Lie on Back"
sequence_id = (
    train_data[(train_data["gesture"] == gesture) & (train_data["orientation"] == orientation)]["sequence_id"]
    .drop_duplicates()
    .iloc[0]
)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id=sequence_id,
    cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_stats=False)
plt.show()


def plot_fft(props: PlotTimeseriesProps, show_labels: bool = True) -> Figure:
    """Plots the 1-D discrete Fourier Transform for a given sequence's collection of features.

    Args:
        props (PlotTimeseriesProps): Properties for plotting the timeseries data.
        show_labels (bool, optional): Whether to show labels on the plot. Defaults to True.

    Returns:
        Figure: The figure object.
    """
    data = props.train_data[
        (props.train_data["sequence_id"] == props.sequence_id) & (props.train_data["phase"].isin(props.phase.value))
    ]
    subject = str(data.iloc[0]["subject"])
    subj_data = props.train_demographics_data[props.train_demographics_data["subject"] == subject]
    handedness = subj_data.iloc[0]["handedness"]

    title = "\n".join(
        [
            f"Gesture: {data.iloc[0]['gesture']}",
            f"Orientation: {data.iloc[0]['orientation']}",
            f"Phase: {props.phase}",
            f"Handedness: {'Right' if handedness else 'Left'}",
            f"Sequence ID: {props.sequence_id}",
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    peak_ticks: list[float] = []

    for col in props.cols:
        signal = data[col].astype(float).to_numpy()

        # Remove the DC component.
        signal = signal - np.mean(signal)
        fft_vals = np.abs(np.array(rfft(signal)))
        fft_vals[0] = 0

        fft_freqs = rfftfreq(len(signal), d=1)
        peak_freq = fft_freqs[np.argmax(fft_vals)]
        peak_ticks.append(peak_freq)

        plt.axvline(x=peak_freq, linestyle="--", color="gray", alpha=0.6)
        sns.lineplot(x=fft_freqs, y=fft_vals, label=f"fft_{col}" if show_labels else None, ax=ax)

    all_ticks = sorted({*ax.get_xticks(), *peak_ticks})
    tick_labels = [f"{tick:.2f}" for tick in all_ticks]
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_xlim(0.0, 0.5)

    if show_labels:
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Frequency (Cycles per Sequence Count)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    plt.tight_layout()
    return fig


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.GESTURE,
)
plot_seq_timeseries(props, show_stats=False)
plot_fft(props)
plt.show()


sample_freq = 6.2
x_peak = 0.03
z_peak = 0.23
gest_len = len(train_data[(train_data["sequence_id"] == "SEQ_011548") & (train_data["phase"] == "Gesture")])

print(
    "\n".join(
        [
            f"acc_x peak frequency (Hz): {x_peak * sample_freq:.2f}",
            f"acc_z peak frequency (Hz): {z_peak * sample_freq:.2f}",
            f"acc_x cycles: {x_peak * gest_len:.2f}",
            f"acc_z cycles: {z_peak * gest_len}",
        ]
    )
)


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_001225",
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.GESTURE,
)
plot_seq_timeseries(props, show_stats=False)
plot_fft(props)
plt.show()


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    phase=Phase.GESTURE,
)
plot_seq_timeseries(props, show_stats=False)
plot_fft(props)
plt.show()


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_000092",
    cols=["acc_x", "acc_y", "acc_z"],
    phase=Phase.GESTURE,
)
plot_seq_timeseries(props, show_stats=False)
plot_fft(props)
plt.show()


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=[f"thm_{i}" for i in range(1, 6)],
    phase=Phase.FULL,
)
plot_seq_timeseries(props)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=[f"thm_{i}" for i in range(1, 6)],
    phase=Phase.GESTURE,
)
plot_fft(props)
plt.show()


props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=[f"tof_1_v{i}" for i in range(64)],
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_labels=False, show_stats=False)

props = PlotTimeseriesProps(
    train_data=train_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=[f"tof_1_v{i}" for i in range(64)],
    phase=Phase.GESTURE,
)
plot_fft(props, show_labels=False)
plt.show()


tof_cols = [f"tof_1_v{i}" for i in range(64)]
tof_data = train_data[train_data["sequence_id"] == "SEQ_011548"].copy(deep=True)
tof_data.replace(-1, np.nan, inplace=True)

# Filter out sensors that never picked up any data.
tof_cols = [col for col in tof_cols if not tof_data[col].isna().all()]

props = PlotTimeseriesProps(
    train_data=tof_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=tof_cols,
    phase=Phase.FULL,
)
plot_seq_timeseries(props, show_labels=False, show_stats=False)

props = PlotTimeseriesProps(
    train_data=tof_data,
    train_demographics_data=train_demographics_data,
    sequence_id="SEQ_011548",
    cols=tof_cols,
    phase=Phase.GESTURE,
)
plot_fft(props, show_labels=False)
plt.show()


if EXPORT_PLOTS_TO_PDF:
    cols_groups = [
        ["acc_x", "acc_y", "acc_z"],
        ["rot_w", "rot_x", "rot_y", "rot_z"],
        [f"thm_{i}" for i in range(1, 6)],
        *[[f"tof_{j}_v{i}" for i in range(64)] for j in range(1, 6)],
    ]

    orientations = [
        "Seated Straight",
        "Seated Lean Non Dom - FACE DOWN",
    ]

    for gesture in train_data["gesture"].drop_duplicates():
        sequence_id = None

        for orientation in orientations:
            filtered = train_data[(train_data["gesture"] == str(gesture)) & (train_data["orientation"] == orientation)][
                "sequence_id"
            ].drop_duplicates()

            if not filtered.empty:
                sequence_id = filtered.iloc[0]
                break

        if sequence_id is None:
            raise RuntimeError(
                f"Could not find a sequence with gesture {gesture} and any of the following orientations {orientations}."
            )

        gesture_str = str(gesture).replace("- ", "").replace(" ", "_").replace("/", "_").lower()
        pdf_path = PLOT_DIR.joinpath(f"{gesture_str}.pdf")

        with PdfPages(pdf_path) as pdf:
            for cols in cols_groups:
                is_tof = any("tof_" in col for col in cols)

                props = PlotTimeseriesProps(
                    train_data=train_data,
                    train_demographics_data=train_demographics_data,
                    sequence_id=sequence_id,
                    cols=cols,
                    phase=Phase.FULL,
                )
                ts_plot = plot_seq_timeseries(props, show_stats=False, show_labels=not is_tof)
                pdf.savefig(ts_plot)
                plt.close(ts_plot)

                props = PlotTimeseriesProps(
                    train_data=train_data,
                    train_demographics_data=train_demographics_data,
                    sequence_id=sequence_id,
                    cols=cols,
                    phase=Phase.GESTURE,
                )
                fft_plot = plot_fft(props, show_labels=not is_tof)
                pdf.savefig(fft_plot)
                plt.close(fft_plot)




