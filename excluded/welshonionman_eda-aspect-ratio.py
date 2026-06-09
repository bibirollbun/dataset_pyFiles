import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


COMP_NAME = "byu-locating-bacterial-flagellar-motors-2025"
COMP_DIR = f"/kaggle/input/{COMP_NAME}"
LABEL = f"{COMP_DIR}/train_labels.csv"

label = pd.read_csv(LABEL)
label



label_ = (
    label.groupby("tomo_id")
    .agg(
        {
            "Array shape (axis 0)": "first",
            "Array shape (axis 1)": "first",
            "Array shape (axis 2)": "first",
            "Voxel spacing": "first",
            "Number of motors": "max",
        }
    )
    .rename(
        columns={
            "Array shape (axis 0)": "z",
            "Array shape (axis 1)": "y",
            "Array shape (axis 2)": "x",
        }
    )
    .reset_index()
)
label_["y/z"] = np.round(label_["z"] / label_["y"], 2)
label_["x/z"] = np.round(label_["z"] / label_["x"], 2)
label_["x/y"] = np.round(label_["y"] / label_["x"], 2)
label_


label_[["x/y", "x/z", "y/z"]].describe()


plt.figure(figsize=(12, 5))
col = "x/y"
plt.hist(
    label_[label_["Number of motors"] == 0][col],
    range=(label_[col].min(), label_[col].max()),
    bins=np.arange(label_[col].min(), label_[col].max(), 0.01),
    alpha=0.5,
    label="Number of motors = 0",
)


plt.hist(
    label_[label_["Number of motors"] >= 1][col],
    range=(label_[col].min(), label_[col].max()),
    bins=np.arange(label_[col].min(), label_[col].max(), 0.01),
    alpha=0.5,
    label="Number of motors >= 1",
)
plt.title("distribution of x/y ratio")

plt.xlabel("x/y ratio")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()

# Tomographs with width/height ratio less than 0.9 contain no motors



plt.figure(figsize=(12, 5))
col = "x/z"
plt.hist(
    label_[label_["Number of motors"] == 0][col],
    range=(label_[col].min(), label_[col].max()),
    bins=np.arange(label_[col].min(), label_[col].max(), 0.01),
    alpha=0.5,
    label="Number of motors = 0",
)


plt.hist(
    label_[label_["Number of motors"] >= 1][col],
    range=(label_[col].min(), label_[col].max()),
    bins=np.arange(label_[col].min(), label_[col].max(), 0.01),
    alpha=0.5,
    label="Number of motors >= 1",
)
plt.title("distribution of x/z ratio")

plt.xlabel("x/z ratio")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()

# Tomographs with width/depth ratio greater than 0.6 contain no motors


plt.figure(figsize=(12, 5))
col = "y/z"

plt.hist(
    label_[label_["Number of motors"] == 0][col],
    range=(label_[col].min(), label_[col].max()),
    bins=np.arange(label_[col].min(), label_[col].max(), 0.01),
    alpha=0.5,
    label="Number of motors = 0",
)


plt.hist(
    label_[label_["Number of motors"] >= 1][col],
    range=(label_[col].min(), label_[col].max()),
    bins=np.arange(label_[col].min(), label_[col].max(), 0.01),
    alpha=0.5,
    label="Number of motors >= 1",
)
plt.title("distribution of y/z ratio")

plt.xlabel("y/z ratio")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()



plt.figure(figsize=(12, 5))
col = "Voxel spacing"

plt.hist(
    label_[label_["Number of motors"] == 0][col],
    range=(label_[col].min(), label_[col].max()),
    bins=np.arange(label_[col].min(), label_[col].max(), 0.1),
    alpha=0.5,
    label="Number of motors = 0",
)


plt.hist(
    label_[label_["Number of motors"] >= 1][col],
    range=(label_[col].min(), label_[col].max()),
    bins=np.arange(label_[col].min(), label_[col].max(), 0.1),
    alpha=0.5,
    label="Number of motors >= 1",
)
plt.title("distribution of Voxel spacing")

plt.xlabel("Voxel spacing")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()


