import os

import numpy as np
import pandas as pd
import imageio.v2 as imageio

from path import Path
from PIL import Image
from IPython.display import Video


rt = Path("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025")
labels = pd.read_csv(rt/"train_labels.csv")


def generate_video(tomo_id):
    img_pth = rt/"train"/tomo_id
    filenames = os.listdir(img_pth)
    tomo_df = labels.loc[labels["tomo_id"]==tomo_id]

    radius = 1000 // tomo_df["Voxel spacing"].values[0]
    point_size = 10

    images = []
    for fn in sorted(filenames):
        img_pil = Image.open(img_pth/fn)
        img_array = np.array(img_pil)
        images.append(img_array)

    images = np.array(images)
    mx = images.max()

    targets = np.zeros(images.shape, dtype=np.float16)
    points = np.zeros(images.shape, dtype=np.uint8)
    X, Y, Z = np.indices(images.shape)

    for i, row in tomo_df.iterrows():
        x, y, z = row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]
        distance = np.sqrt((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2)
        targets[(distance < radius)] += mx
        points[(distance < point_size)] = mx

    del distance, X, Y, Z

    frames = []
    for fi in range(images.shape[0]):
        img_array = images[fi, :, :]
        img_target = targets[fi, :, :]
        img_point = points[fi, :, :]
        img_frame = np.stack([img_target, img_array, img_point], axis=-1)
        frames.append(img_frame)

    imageio.mimsave(f"{tomo_id}.mp4", frames, fps=30, format="mp4")

    return Video(f"/kaggle/working/{tomo_id}.mp4", embed=True, width=500)


generate_video("tomo_7ca7c0")

