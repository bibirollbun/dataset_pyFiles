# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from pathlib import Path
import polars as pl
import warnings
warnings.simplefilter('ignore')


ROOT = Path("/kaggle")
DATA_DIR = ROOT / "input/MABe-mouse-behavior-detection"


mice_recording_setups_df = pl.read_csv(DATA_DIR / "train.csv")

u_lab_id = mice_recording_setups_df["lab_id"].unique().to_list()
u_body_parts_traced = mice_recording_setups_df["body_parts_tracked"].unique().to_list()
u_behaviors_labeled = mice_recording_setups_df["behaviors_labeled"].unique().to_list()

print(f"{len(u_lab_id)=}")
print(f"{len(u_body_parts_traced)=}")
print(f"{len(u_behaviors_labeled)=}")


mice_recording_setups_df.group_by("lab_id").agg(
    pl.col("body_parts_tracked").n_unique(),
    pl.col("behaviors_labeled").n_unique(),
).select("body_parts_tracked", "behaviors_labeled").max()


import seaborn as sns

pivot = mice_recording_setups_df.pivot(
    index="lab_id",
    on="mouse1_strain",
    values="mouse1_id",
    aggregate_function=pl.len(),
)

sns.heatmap(
    pivot.to_pandas().set_index("lab_id"),
    cmap="viridis",
    annot=True,
    # fmt="d",
)


import matplotlib.pyplot as plt
import seaborn as sns

features = ["strain", "color", "sex", "age"]
for feature in features:
    fig, axes = plt.subplots(1, 4, figsize=(24, 4), sharey=True, sharex=True)
    fig.suptitle(feature)
    for i in range(4):
        ax = axes[i]
        sns.histplot(
            data=mice_recording_setups_df,
            x=f"mouse{i+1}_{feature}",
            # hue="lab_id",
            # multiple="stack",
            ax=ax,
        )
        # using a FixedLocator to set the tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()


fig, axes = plt.subplots(1, 4, figsize=(24, 4))
for i in range(4):
    ax = axes[i]
    sns.histplot(
        data=mice_recording_setups_df,
        x=f"mouse{i+1}_strain",
        y=f"mouse{i+1}_color",
        cmap="viridis",
        ax=ax,
        cbar=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()


import seaborn as sns

pivot = mice_recording_setups_df.pivot(
    index="arena_shape",
    on="arena_type",
    values="lab_id",
    aggregate_function=pl.len(),
)

g = sns.heatmap(
    pivot.to_pandas().set_index("arena_shape"),
    cmap="viridis",
    annot=True,
    # fmt="d",
)
g.set_xlabel("arena_type");


bpbl_pair_counts = (
    mice_recording_setups_df
    .filter(
        (pl.col("arena_shape") == "square") & (pl.col("arena_type") == "neutral")
        | (pl.col("arena_shape") == "rectangular") & (pl.col("arena_type") == "resident-intruder")
    )
    .group_by("body_parts_tracked", "behaviors_labeled")
    .len("count")
    .sort(["count", "body_parts_tracked", "behaviors_labeled"], descending=True)
)

bpbl_pair_counts.head(10)


filtered = mice_recording_setups_df.filter(
    (
        (pl.col("arena_shape") == "square") & (pl.col("arena_type") == "neutral")
        | (pl.col("arena_shape") == "rectangular") & (pl.col("arena_type") == "resident-intruder")
    )
    & (pl.col("behaviors_labeled").is_null())
)

print(f"{filtered['lab_id'].unique().to_list()=}")
print(f"{filtered['frames_per_second'].unique().to_list()=}")
print(f"{filtered['video_duration_sec'].unique().to_list()=}")
print(f"{filtered['pix_per_cm_approx'].unique().to_list()=}")
print(f"{filtered['video_width_pix'].unique().to_list()=}")
print(f"{filtered['video_height_pix'].unique().to_list()=}")
print(f"{filtered['arena_width_cm'].unique().to_list()=}")
print(f"{filtered['arena_height_cm'].unique().to_list()=}")
print(f"{filtered['tracking_method'].unique().to_list()=}")


mice_recording_setups_df.filter(
    pl.col("lab_id").is_in(["MABe22_keypoints", "MABe22_movies"])
).height


import matplotlib.pyplot as plt
import seaborn as sns

numerical_features = ["frames_per_second", "video_duration_sec", "pix_per_cm_approx", "video_width_pix", "video_height_pix", "arena_width_cm", "arena_height_cm"]
fig, axes = plt.subplots(1, len(numerical_features), figsize=(24, 4), sharey=True)
for i, feature in enumerate(numerical_features):
    ax = axes[i]
    sns.histplot(
        data=mice_recording_setups_df,
        x=feature,
        # hue="lab_id",
        # multiple="stack",
        ax=ax,
    )
    # using a FixedLocator to set the tick labels
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(feature)
    ax.set_yscale("log")
plt.show()


import seaborn as sns
import numpy as np

corr = mice_recording_setups_df.select(numerical_features).corr().to_pandas()
corr.index = corr.columns
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    cmap="viridis",
    annot=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=True,
    mask=mask,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75},
)



features = ["frames_per_second", "video_duration_sec"]
counts = mice_recording_setups_df.select(
    pl.col("lab_id"),
    *[pl.col(col) for col in features],
).unique().sort("lab_id").group_by("lab_id").len("count")

totals = mice_recording_setups_df.group_by("lab_id").len("total")

counts = (
    counts
    .join(totals, on="lab_id", how="left")
    .with_columns(
        (pl.col("count") / pl.col("total")).alias("ratio"),
    )
    .sort("ratio", descending=True)
)
display(counts.head(10))
display(counts.tail(10))


features = ["pix_per_cm_approx", "video_width_pix", "video_height_pix"]
counts = mice_recording_setups_df.select(
    pl.col("lab_id"),
    *[pl.col(col) for col in features],
).unique().sort("lab_id").group_by("lab_id").len("count")

totals = mice_recording_setups_df.group_by("lab_id").len("total")

counts = (
    counts
    .join(totals, on="lab_id", how="left")
    .with_columns(
        (pl.col("count") / pl.col("total")).alias("ratio"),
    )
    .sort("ratio", descending=True)
)
display(counts.head(10))
display(counts.tail(10))


features = ["arena_width_cm", "arena_height_cm", "arena_shape", "arena_type"]
counts = mice_recording_setups_df.select(
    pl.col("lab_id"),
    *[pl.col(col) for col in features],
).unique().sort("lab_id").group_by("lab_id").len("count")

totals = mice_recording_setups_df.group_by("lab_id").len("total")

counts = (
    counts
    .join(totals, on="lab_id", how="left")
    .with_columns(
        (pl.col("count") / pl.col("total")).alias("ratio"),
    )
    .sort("ratio", descending=True)
)
display(counts.head(10))
display(counts.tail(10))

