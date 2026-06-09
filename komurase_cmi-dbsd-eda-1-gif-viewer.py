import os
from pathlib import Path
from typing import List

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import base64
from IPython import display as dd


# input
input_path = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data/")
train_df = pd.read_csv(input_path / "train.csv")


# output directory
save_dir = Path("./mp4")
save_fig = Path("./images")
# sample = train_df.loc[train_df["sequence_id"] == "SEQ_056304"]
# sample


test_df = pd.read_csv(input_path / "test.csv")
test_df


train_df.sample(10)


train_df[["subject","gesture"]].value_counts().to_frame().reset_index().sort_values(by="subject")





os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_fig, exist_ok=True)


train_df["gesture"].value_counts()


gestures = train_df["gesture"].value_counts().index.values
gestures


train_df.sample(10)


train_df[["sequence_id","gesture"]].value_counts().to_frame().reset_index().sort_values(by="sequence_id")


def quaternion_to_euler(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame内のクォータニオン（rot_w, rot_x, rot_y, rot_z）を
    オイラー角（yaw, pitch, roll）に変換して新しいカラムに追加
    """
    output = df.copy()
    # scipyでは (x, y, z, w) の順番が必要なので並べ替える
    quats = df[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    r = R.from_quat(quats)

    # ZYX順 → yaw(ヨー), pitch(ピッチ), roll(ロール)
    eulers = r.as_euler("zyx", degrees=True)

    output.loc[:, "yaw"] = eulers[:, 0]
    output.loc[:, "pitch"] = eulers[:, 1]
    output.loc[:, "roll"] = eulers[:, 2]
    return output


def plot_euler_angles(df: pd.DataFrame, sequence_id: str) -> pd.DataFrame:
    seq = df[df["sequence_id"] == sequence_id]

    plt.figure(figsize=(12, 4))
    for angle in ["yaw", "pitch", "roll"]:
        plt.plot(seq["sequence_counter"], seq[angle], label=angle)

    plt.title(f"Euler Angles from Quaternion (sequence_id = {sequence_id})")
    plt.xlabel("Time step")
    plt.ylabel("Angle (degrees)")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=15)
    plt.grid(True)
    plt.show()


def make_plot(sample: pd.DataFrame, gesture: str, idx: int):
    # Figure全体を作成
    fig = plt.figure(figsize=(15, 15))  # 横長のFigureを指定

    # GridSpecで行の高さ比を調整（上:中:下 = 1:1:2）
    gs = gridspec.GridSpec(nrows=3, ncols=5, height_ratios=[1, 1, 2])

    # 時系列プロット（上の1行5列全体を1つのAxesに）
    ax_ts = fig.add_subplot(gs[0, :])  # 1行目すべてを使う
    ax_ts.set_title(f"Time Series Plot | Gesture is {gesture}")

    for angle in ["yaw", "pitch", "roll"]:
        ax_ts.plot(sample["sequence_counter"], sample[angle], label=angle)
    ax_ts.axvspan(
        sample.loc[idx]["sequence_counter"],
        sample.loc[idx + 1]["sequence_counter"],
        color="red",
        alpha=0.2,
        # edgecolor="none",
        label="time",
    )
    ax_ts.grid(True)
    ax_ts.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=15)

    #####################################
    # 時系列プロット（上の2行5列全体を1つのAxesに）
    ax_ts_2 = fig.add_subplot(gs[1, :])  # 1行目すべてを使う
    # ax_ts_2.set_title(f"Time Series gesture is {gesture}")

    for angle in ["acc_x", "acc_y", "acc_z"]:
        ax_ts_2.plot(sample["sequence_counter"], sample[angle], label=angle)
    ax_ts_2.axvspan(
        sample.loc[idx]["sequence_counter"],
        sample.loc[idx + 1]["sequence_counter"],
        color="red",
        alpha=0.2,
        # edgecolor="none",
        # label="time",
    )
    ax_ts_2.grid(True)
    ax_ts_2.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=15)

    #####################################
    # ヒートマップ（2行目：5個並べる）
    for i in range(5):
        tof_features = [c for c in tof_columns if f"tof_{i+1}" in c]
        ax_hm = fig.add_subplot(gs[2, i])
        im = ax_hm.imshow(
            sample[tof_features].values[idx].reshape(8, 8), cmap="Blues", aspect="equal"
        )  # 正方形
        ax_hm.set_title(f"time of fright {idx}")
        ax_hm.axis("off")  # 軸を消すと見やすくなる

    return fig


def make_image_from_sequence(input_df: pd.DataFrame, sequence_id: str, save_fig=False) -> List[str]:
    sample = input_df[input_df["sequence_id"] == sequence_id].reset_index(drop=True)
    sample = quaternion_to_euler(sample)
    gesture = sample["gesture"].unique()
    gesture = ",".join(gesture)

    image_paths = []

    for idx in range(len(sample)):
        if idx == len(sample) - 1:
            continue
        fig = make_plot(sample, gesture, idx)
        path = save_fig / f"{sequence_id}_{idx:03}.png"

        if save_fig:
            fig.savefig(path, bbox_inches="tight")
            image_paths.append(path)
        plt.close(fig)
    return image_paths


target = ["gesture"]
train_only_columns = ["sequence_type", "orientation"]
tof_columns = [c for c in train_df.columns if "tof" in c]


gesture_seq_id = {}

for gesture in gestures:
    gesture_seq_id[gesture] = train_df[train_df["gesture"]==gesture]["sequence_id"].sample(1, random_state=0).values


gesture_seq_id


train_df["sequence_id"].value_counts()


train_df[train_df["sequence_id"]=="SEQ_039523"]["gesture"].value_counts()


#!rm /kaggle/working/images/*.png


# for g, ID in gesture_seq_id.items():
#     make_image_from_sequence(input_df=train_df, sequence_id=ID[0])


g='Text on phone'
time=0
ID = gesture_seq_id[g]
sample = train_df[train_df["sequence_id"] == ID[0]].reset_index(drop=True)
sample = quaternion_to_euler(sample)
fig = make_plot(sample=sample, gesture=g, idx=time)


for time in range(60, 80, 1):
    fig = make_plot(sample=sample, gesture=g, idx=time)


gesture_seq_id


g='Wave hello'
time=0
ID = train_df[train_df["gesture"]==gesture]["sequence_id"].sample(1, random_state=0).values[0]
sample = train_df[train_df["sequence_id"] == ID].reset_index(drop=True)
sample = quaternion_to_euler(sample)

fig = make_plot(sample=sample, gesture=g, idx=time)


g='Wave hello'
time=0
ID = train_df[train_df["gesture"]==gesture]["sequence_id"].sample(1, random_state=20).values[0]
sample = train_df[train_df["sequence_id"] == ID].reset_index(drop=True)
sample = quaternion_to_euler(sample)

fig = make_plot(sample=sample, gesture=g, idx=time)


g='Wave hello'
ID = train_df[train_df["gesture"]==gesture]["sequence_id"].sample(1, random_state=20).values[0]
sample = train_df[train_df["sequence_id"] == ID].reset_index(drop=True)
sample = quaternion_to_euler(sample)

sample.shape


for time in range(60,100,5):
    fig = make_plot(sample=sample, gesture=g, idx=time)


agg_df = train_df.groupby(["gesture", "sequence_counter"], as_index=False)[["acc_x", "acc_y", "acc_z"]].agg("mean")
agg_df


g='Eyebrow - pull hair'
time=0
ID = train_df[train_df["gesture"]==gesture]["sequence_id"].sample(1, random_state=0).values[0]
sample = train_df[train_df["sequence_id"] == ID].reset_index(drop=True)
sample = quaternion_to_euler(sample)

fig = make_plot(sample=sample, gesture=g, idx=time)


g='Eyebrow - pull hair'
time=0
ID = train_df[train_df["gesture"]==gesture]["sequence_id"].sample(1, random_state=100).values[0]
sample = train_df[train_df["sequence_id"] == ID].reset_index(drop=True)
sample = quaternion_to_euler(sample)

fig = make_plot(sample=sample, gesture=g, idx=time)


g='Eyebrow - pull hair'
time=0
ID = train_df[train_df["gesture"]==gesture]["sequence_id"].sample(1, random_state=101).values[0]
sample = train_df[train_df["sequence_id"] == ID].reset_index(drop=True)
sample = quaternion_to_euler(sample)

fig = make_plot(sample=sample, gesture=g, idx=time)


g='Eyebrow - pull hair'
time=0
ID = train_df[train_df["gesture"]==gesture]["sequence_id"].sample(1, random_state=101).values[0]
sample = train_df[train_df["sequence_id"] == ID].reset_index(drop=True)
sample = quaternion_to_euler(sample)

os.makedirs(f"/kaggle/working/images/{ID}/", exist_ok=True)

image_paths = []
for time in tqdm(range(0,70)):
    fig = make_plot(sample=sample, gesture=g, idx=time)
    path=f"/kaggle/working/images/{ID}/{g}_{time}.png"
    fig.savefig(path, bbox_inches="tight")
    image_paths.append(path)
    plt.close(fig)


frames = [Image.open(p) for p in image_paths]
frames[0].save(
    "output.gif", save_all=True, append_images=frames[1:], duration=500, loop=0  # ms
)


!ls


with open("output.gif", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
    
display(dd.HTML(f'<img src="data:image/gif;base64,{b64}" />'))

