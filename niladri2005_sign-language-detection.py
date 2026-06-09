!pip install --upgrade mediapipe tensorflow tensorflow-transform tensorflow-serving-api


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

plt.style.use("seaborn-v0_8-colorblind")


!pip install "black[jupyter]"


!ls ../input/asl-signs/ -GFlash --color


BASE_DIR = '../input/asl-signs/'
train = pd.read_csv(f'{BASE_DIR}/train.csv')


# Train.csv has the path to each parquet file, the particpant id, sequence_id and sign.
train.head()


train.head()


fig, ax = plt.subplots(figsize=(8, 8))
train["sign"].value_counts().head(50).sort_values(ascending=True).plot(
    kind="barh", ax=ax, title="Top 50 Signs in Training Dataset"
)
ax.set_xlabel("Number of Training Examples")
plt.show()


fig, ax = plt.subplots(figsize=(8, 8))
train["sign"].value_counts().tail(50).sort_values(ascending=True).plot(
    kind="barh", ax=ax, title="Bottom 50 Signs in Training Dataset"
)
ax.set_xlabel("Number of Training Examples")
plt.show()


example_fn = train.query('sign == "listen"')["path"].values[0]

example_landmark = pd.read_parquet(f"{BASE_DIR}/{example_fn}")
example_landmark.head()


unique_frames = example_landmark["frame"].nunique()
unique_types = example_landmark["type"].nunique()
types_in_video = example_landmark["type"].unique()
print(
    f"The file has {unique_frames} unique frames and {unique_types} unique types: {types_in_video}"
)


listen_files = train.query('sign == "listen"')["path"].values
for i, f in enumerate(listen_files):
    example_landmark = pd.read_parquet(f"{BASE_DIR}/{f}")
    unique_frames = example_landmark["frame"].nunique()
    unique_types = example_landmark["type"].nunique()
    types_in_video = example_landmark["type"].unique()
    print(
        f"The file has {unique_frames} unique frames and {unique_types} unique types: {types_in_video}"
    )
    if i == 20:
        break


N_PARQUETS_TO_READ = 100_000  # So we don't have to load all 95k

combined_meta = {}
for i, d in tqdm(train.iterrows(), total=len(train)):
    file_path = d["path"]
    example_landmark = pd.read_parquet(f"{BASE_DIR}/{file_path}")
    # Get the number of landmarks with x,y,z data per type
    meta = (
        example_landmark.dropna(subset=["x", "y", "z"])["type"].value_counts().to_dict()
    )
    meta["frames"] = example_landmark["frame"].nunique()
    xyz_meta = (
        example_landmark.agg(
            {
                "x": ["min", "max", "mean"],
                "y": ["min", "max", "mean"],
                "z": ["min", "max", "mean"],
            }
        )
        .unstack()
        .to_dict()
    )

    for key in xyz_meta.keys():
        new_key = key[0] + "_" + key[1]
        meta[new_key] = xyz_meta[key]
    combined_meta[file_path] = meta
    if i >= N_PARQUETS_TO_READ:
        break


train_with_meta = train.merge(
    pd.DataFrame(combined_meta).T.reset_index().rename(columns={"index": "path"}),
    how="left",
)
train_with_meta.to_parquet("train_with_meta.parquet")


train_with_meta[["face", "pose", "left_hand", "right_hand"]].sum().sort_values().plot(
    kind="barh", title="Sum of Rows by Landmark Type"
)
plt.show()


# checking to see if the number of landmarks for this type is zero
(
    train_with_meta.query("index < 1000").fillna(0)[
        ["face", "pose", "left_hand", "right_hand"]
    ]
    > 0
).mean().plot(kind="barh", title="Rate of Frame/Keypoints with Data")


example_fn = train_with_meta.dropna().query('sign == "shhh"')["path"].values[0]
example_landmark = pd.read_parquet(f"{BASE_DIR}/{example_fn}")


example_landmark.query("frame == 25")["type"].value_counts()  # Middle of the video


example_landmark["no_xyz"] = example_landmark["x"].isna()


example_landmark.groupby("frame")["no_xyz"].sum().plot(
    title="missing xyz per frame", kind="bar"
)


import plotly.express as px

example_frame = example_landmark.query("frame == 17")
px.scatter_3d(example_frame, x="x", y="y", z="z", color="type")


example_landmark["y_"] = example_landmark["y"] * -1
example_frame = example_landmark.query("frame == 17 and type== 'face'")
px.scatter(example_frame, x="x", y="y_", color="type")





import mediapipe as mp

mp_hands = mp.solutions.hands


example_landmark["y_"] = example_landmark["y"] * -1

fig, ax = plt.subplots(figsize=(5, 5))

for hand in ["left_hand", "right_hand"]:
    example_hand = example_landmark.query("frame == 17 and type == @hand")

    ax.scatter(example_hand["x"], example_hand["y_"])

    for connection in mp_hands.HAND_CONNECTIONS:
        point_a = connection[0]
        point_b = connection[1]
        x1, y1 = example_hand.query("landmark_index == @point_a")[["x", "y_"]].values[0]
        x2, y2 = example_hand.query("landmark_index == @point_b")[["x", "y_"]].values[0]
        plt.plot([x1, x2], [y1, y2], color="purple")
ax.set_title("Shhh - Hands Data")
plt.show()























































