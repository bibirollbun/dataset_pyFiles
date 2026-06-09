from typing import Dict
from pathlib import Path
from types import SimpleNamespace
from multiprocessing import Pool
from functools import partial

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.core.display import Video
from tqdm import tqdm

plt.style.use("ggplot")

cfg = SimpleNamespace()
cfg.INPUT = Path("/kaggle/input/asl-signs")
cfg.OUTPUT = Path("/kaggle/working/animation")
cfg.DEBUG = True

%load_ext autoreload
%autoreload 2


train = pl.read_csv(cfg.INPUT / "train.csv")


train


print(f"# Unique participants: {len(train['participant_id'].unique())}")
print(f"# Unique sequence: {len(train['sequence_id'].unique()):,}")
print(f"# Unique signs: {len(train['sign'].unique())}")


sequence_per_participant = (
    train.group_by("participant_id")
    .agg(
        sequence_count=pl.col("sequence_id").unique().count()
    )
)

sequence_per_participant


sequence_per_participant = (
    train.group_by("participant_id")
    .agg(
        sequence_count=pl.col("sequence_id").unique().count()
    )
    .with_columns(pl.col("participant_id").cast(str))
)

print(sequence_per_participant)

_, ax, = plt.subplots()
ax.barh(
    y=sequence_per_participant['participant_id'],
    width=sequence_per_participant['sequence_count']
)
ax.set(
    xlabel="#sequences",
    ylabel="participant_id",
    title="#sequence per participant"
)
plt.show()


sequence_per_sign = (
    train.group_by("sign").agg(
        sequence_count=pl.col("sequence_id").count()
    ).sort("sequence_count")
)

print(sequence_per_sign)

_, ax = plt.subplots()
ax.hist(sequence_per_sign["sequence_count"], bins=20)
ax.set(xlabel="#sequences/sign", ylabel="count", title="#sequences per sign")
plt.show()


sequence_per_sign = (
    train.group_by("sign").agg(
        pl.col("participant_id").count()
    ).sort("participant_id").select(sign=pl.col("sign") ,participant_count=pl.col("participant_id"))
)

print(sequence_per_sign)

_, ax = plt.subplots()
ax.hist(sequence_per_sign["participant_count"], bins=20)
ax.set(xlabel="#participants/sign", ylabel="count", title="#participants per sign")
plt.show()


edges = {
    "left_hand": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (0, 17),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ],
    "right_hand": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (0, 17),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ],
    "pose": [
        (8, 6),
        (6, 5),
        (6, 4),
        (4, 0),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (10, 9),
        (11, 12),
        (11, 13),
        (11, 23),
        (13, 15),
        (15, 21),
        (15, 17),
        (15, 19),
        (17, 19),
        (12, 14),
        (12, 24),
        (14, 16),
        (16, 22),
        (16, 20),
        (16, 18),
        (18, 20),
        (23, 24),
        # discard landmarks of lower body
        # (24, 26),
        # (26, 28),
        # (28, 30),
        # (28, 32),
        # (30, 32),
        # (23, 25),
        # (25, 27),
        # (27, 29),
        # (27, 31),
        # (29, 31),
    ],
}


lm_data = {k: v for k, v in zip(train.columns, train.row(10))}

df_landmark = pl.read_parquet(cfg.INPUT / lm_data["path"])
lm_first_frame = df_landmark.partition_by("frame")[0]
lms = lm_first_frame.partition_by("type")


lm_first_frame.partition_by("type")


for lm in lms:
    lm = lm.filter(
        (pl.col("type") != "pose") | (pl.col("landmark_index") < 25)
    )    
    print(lm.row(0)[2])


_, axes = plt.subplots(2,2,figsize=(8,8))
axes = axes.ravel()

for lm, ax in zip(lms, axes):
    lm = lm.filter(
        (pl.col("type") != "pose") | (pl.col("landmark_index") < 25)
    )
    lm_type = lm.row(0)[2]
    ax.scatter(lm["x"], lm["y"])

    if lm_type != "face":
        for row in lm.iter_rows():
            dt = {k: v for k, v in zip(lm.columns, row)}
            x, y, idx = dt["x"], dt["y"], dt["landmark_index"]

            if (x is not None) & (y is not None):
                ax.text(x,y,idx)

        for edge in edges[lm_type]:
            i, j = edge
            x1, x2, y1, y2 = lm["x"][i], lm["x"][j], lm["y"][i], lm["y"][j]
            if not ((x1 is None) | (x2 is None) | (y1 is None) | (y2 is None)):
                ax.plot((x1, x2), (y1, y2), color="gray")

    ax.set(title=f"{lm_type}")
    ax.invert_yaxis()


_, axes = plt.subplots(1,3,figsize=(12,4))

for ax, col in zip(axes, ["x", "y", "z"]):
    ax.hist(df_landmark[col], bins=20, alpha=0.5, label=col)
    ax.set(xlabel=col)

plt.title("Distribution of axes of normalized co-ordinate")
plt.show()


def make_animation(row, columns, fps: int = 5):
    data = {k: v for k, v in zip(columns, row)}
    sign, participant_id, sequence_id = (
        data["sign"],
        data["participant_id"],
        data["sequence_id"],
    )

    df_landmark = pl.read_parquet(cfg.INPUT / data["path"])
    use_cols = ["x", "y", "z"]
    df_landmark = df_landmark.sort(["frame", "type", "landmark_index"]).with_columns(
        [pl.col(col).interpolate().over(["type", "landmark_index"]) for col in use_cols]
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()

    lms_all = df_landmark.partition_by("frame")

    def draw_frame(frame):
        lms = lms_all[frame].partition_by("type")

        for lm, ax in zip(lms, axes):
            lm = lm.filter((pl.col("type") != "pose") | (pl.col("landmark_index") < 25))
            ax.cla()
            lm_type = lm.row(0)[2]
            frame = lm.row(0)[0]

            ax.scatter(lm["x"], lm["y"])
            if lm_type != "face":
                for row in lm.iter_rows():
                    dt = {k: v for k, v in zip(lm.columns, row)}
                    if (dt["x"] is not None) & (dt["y"] is not None):
                        ax.text(dt["x"], dt["y"], dt["landmark_index"])
            if lm_type in ["left_hand", "right_hand", "pose"]:
                for edge in edges[lm_type]:
                    i, j = edge
                    x1, x2, y1, y2 = lm["x"][i], lm["x"][j], lm["y"][i], lm["y"][j]
                    if not ((x1 is None) | (x2 is None) | (y1 is None) | (y2 is None)):
                        ax.plot((x1, x2), (y1, y2), color="gray")
            ax.set(title=f"{lm_type}")
            ax.invert_yaxis()
        plt.suptitle(f'sign: "{sign}" [frame={frame}]')

    ani = animation.FuncAnimation(
        fig, draw_frame, frames=range(len(lms_all)), interval=1000 / fps
    )

    if not (cfg.OUTPUT / sign).exists():
        (cfg.OUTPUT / sign).mkdir()
    ani.save(
        cfg.OUTPUT / sign / f"{participant_id}_{sequence_id}.mp4",
        writer="ffmpeg",
        fps=fps,
        codec="h264",
    )
    plt.close(fig)


train_unique_signs = train.filter(
    (pl.arange(0, pl.len())).shuffle(seed=42).over("sign") < 20
)

train_unique_signs


train_unique_signs.group_by("sign").agg(pl.len()).head()


if not cfg.OUTPUT.exists():
    cfg.OUTPUT.mkdir()

if cfg.DEBUG:
    df = train_unique_signs.head(4)
else:
    df = train_unique_signs

for row in tqdm(df.iter_rows(), total=len(df)):
    make_animation(row, df.columns)


!du -sh animation


!ls animation/table


!cp animation/have/49445_1025455481.mp4 sample001.mp4
!cp animation/jacket/55372_1008254374.mp4 sample002.mp4
!cp animation/night/61333_1014909824.mp4 sample003.mp4
!cp animation/table/55372_1044122718.mp4 sample004.mp4


Video("sample001.mp4", embed=True)


Video("sample003.mp4", embed=True)

