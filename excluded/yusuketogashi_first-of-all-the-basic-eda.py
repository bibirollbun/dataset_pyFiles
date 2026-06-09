#Train Data
#/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv
#/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv

#Test Data
#kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv
#kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)



DATA_DIR = "/kaggle/input/cmi-detect-behavior-with-sensor-data"

train       = pd.read_csv(f"{DATA_DIR}/train.csv")
test        = pd.read_csv(f"{DATA_DIR}/test.csv")
train_demo  = pd.read_csv(f"{DATA_DIR}/train_demographics.csv")
test_demo   = pd.read_csv(f"{DATA_DIR}/test_demographics.csv")

print(f"train shape: {train.shape}")
print(f"test  shape: {test.shape}")



train.info(memory_usage="deep")
train.isna().mean().sort_values(ascending=False).head(20)  # 欠損率 TOP20



assert train["row_id"].is_unique
n_seq_train = train["sequence_id"].nunique()
n_seq_test  = test["sequence_id"].nunique()
print("unique sequences – train:", n_seq_train, " test:", n_seq_test)



seq_level = train.groupby("sequence_id").agg(
    gesture        = ("gesture", "first"),
    sequence_type  = ("sequence_type", "first"),
    rows_in_seq    = ("row_id", "count"),
    subject        = ("subject", "first"),
)
display(seq_level.head())

fig, ax = plt.subplots(figsize=(10,4))
seq_level["gesture"].value_counts().plot.bar(ax=ax)
ax.set_title("Gesture count (train)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()



plt.figure(figsize=(6,4))
sns.histplot(seq_level["rows_in_seq"], bins=50, log_scale=True)
plt.title("Distribution of rows per sequence")
plt.xlabel("#rows in one sequence")
plt.show()



sensor_cols = [c for c in train.columns if c.startswith(("acc_", "rot_", "thm_", "tof_"))]
missing = train[sensor_cols].isna()

# 欠損率
miss_rate = missing.mean().sort_values(ascending=False)
print(miss_rate.head())

# センサ種別ごと概要
for prefix in ["acc_", "rot_", "thm_", "tof_"]:
    cols = [c for c in sensor_cols if c.startswith(prefix)]
    print(prefix, "mean missing =", missing[cols].mean().mean().round(3))



imu_cols = [c for c in sensor_cols if c.startswith(("acc_", "rot_"))]
imu_stats = train[imu_cols].describe().T[["mean", "std", "min", "max"]]
display(imu_stats.head(10))



thm_cols = [f"thm_{i}" for i in range(1,6)]
train[thm_cols].hist(bins=50, figsize=(12,4), layout=(1,5))
plt.suptitle("Thermopile temperature distribution (°C)")



# --- 1. Required libraries & inline magic ---
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 2. Data loading & creating seq_level ---
DATA_DIR = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
train    = pd.read_csv(f"{DATA_DIR}/train.csv")

# count the number of rows for each sequence_id
seq_level = train.groupby("sequence_id")["row_id"].count().rename("rows_in_seq")

# --- 3. Automatically select sequences that can be visualized ---
# Here we look for items with more than 100 lines
candidate = seq_level[seq_level >= 100]
if len(candidate)==0:
    raise ValueError("No sequences found with more than 100 rows")
EXAMPLE_SEQ = candidate.index[0]
print(f"Selected sequence {EXAMPLE_SEQ} has number of lines {candidate.iloc[0]} ")

# --- 4. Frame extraction & index clip ---
seq_df    = train[train["sequence_id"] == EXAMPLE_SEQ].reset_index(drop=True)
desired_idx = 100
frame_idx  = min(desired_idx, len(seq_df)-1)  # Prevent out-of-bounds
print(f"Frame index to actually plot:{frame_idx}")

row = seq_df.iloc[frame_idx]

# --- 5. Draw a heat map of 5 ToF ---
for sensor_id in range(1, 6):
    cols   = [f"tof_{sensor_id}_v{i}" for i in range(64)]
    pixels = row[cols].values.astype(float)
    pixels[pixels == -1] = np.nan
    grid   = pixels.reshape(8, 8)

    plt.figure(figsize=(3, 3))
    sns.heatmap(
        grid,
        vmin=0, vmax=254,
        cmap="viridis",
        cbar=False,
        square=True,
        linewidth=0.1,
        linecolor="gray",
    )
    plt.title(f"Seq {EXAMPLE_SEQ}  Frame {frame_idx}  ToF {sensor_id}", fontsize=10)
    plt.axis("off")
    plt.show()



# 1) Convert basic information for each sequence into a DataFrame
seq_info = (
    train
    .groupby("sequence_id")
    .agg(
        # Get "gesture", "sequence_type" and "subject" from the first line for each sequence_id
        gesture        = ("gesture",       "first"),
        sequence_type  = ("sequence_type", "first"),
        rows_in_seq    = ("row_id",        "count"),
        subject        = ("subject",       "first"),
    )
    .reset_index() # index=sequence_id → columnization
)

# 2) Load demographics (subject attributes) data
train_demo = pd.read_csv(f"{DATA_DIR}/train_demographics.csv")

# 3) Combine seq_info and train_demo by subject
seq_demo = seq_info.merge(train_demo, on="subject", how="left")

# 4) Count the number of sequences with adult_child (child/adult) and sex (gender)
pivot = seq_demo.groupby(["adult_child", "sex"])["sequence_id"] \
                .count() \
                .unstack(fill_value=0)

print(pivot)




# If Thermopile is all NaN, it is determined as IMU-only
is_imu_only = train.groupby("sequence_id")["thm_1"].apply(lambda x: x.isna().all())
print(is_imu_only.value_counts())


