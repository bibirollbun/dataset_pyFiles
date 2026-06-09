%load_ext autoreload
%autoreload 2


import polars as pl

label_df = pl.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")
label_df = label_df.with_columns(
    volume=pl.col("Array shape (axis 0)") * pl.col("Array shape (axis 1)") * pl.col("Array shape (axis 2)")
)
tomo_ids = label_df["tomo_id"].unique(maintain_order=True).to_numpy()


label_df


from pathlib import Path


raw_tomo_path = Path("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train")


from pathlib import Path
import numpy as np
import imageio.v3 as imageio


def load_tomo(tomo_id, tomo_dir, stride=1):
    tomo_path = Path(tomo_dir) / tomo_id
    # Load the tomogram
    voxel = []
    for path in sorted(tomo_path.glob("slice_*.jpg")):
        slice_id = int(path.stem.split("_")[-1])
        if slice_id % stride != 0:
            continue
        try:
            tomogram = np.array(imageio.imread(path))
            voxel.append(tomogram)
        except FileNotFoundError:
            continue
    voxel = np.stack(voxel, axis=0)
    return voxel


from tqdm import tqdm


def check_outplier(raw_tomo_path, tomo_ids, stride=50):
    outlier_lefts = []
    outlier_rights = []
    outlier_scores = []
    for tomo_id in tqdm(tomo_ids):
        voxel = load_tomo(
            tomo_id,
            raw_tomo_path,
            stride=stride,
        )  # Load the first tomogram
        count, left = np.histogram(voxel.flatten(), bins=255, range=(0, 255))
        outlier_lefts.append(count[:50].sum() / count.sum())
        outlier_rights.append(count[-50:].sum() / count.sum())
        outlier_scores.append((count[:50].sum() + count[-50].sum()) / count.sum())
    
    outlier_lefts = np.array(outlier_lefts)
    outlier_rights = np.array(outlier_rights)
    outlier_scores = np.array(outlier_scores)
    return outlier_scores


import pickle


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


outlier_path = Path("outlier_scores.pkl")
outlier_scores = check_outplier(raw_tomo_path, tomo_ids, stride=50)
save_pickle(outlier_scores, outlier_path)


import matplotlib.pyplot as plt

plt.hist(outlier_scores, bins=100)
plt.show()


THRESHOLD = 0.5


anomaly_idxs = np.where(outlier_scores > THRESHOLD)[0]
anomaly_tomo_ids = tomo_ids[anomaly_idxs]
print(anomaly_tomo_ids)


save_pickle(anomaly_tomo_ids, "anomaly_tomo_ids.pkl")


volume_df = label_df.filter(pl.col("tomo_id").is_in(anomaly_tomo_ids)).group_by("tomo_id").agg(pl.all().first())
total_volume = volume_df["volume"].sum()
print(f"{total_volume=}")


volume_df.describe()


def percentile_norm(image, lb=1, ub=99):
    """
    Normalize the image using percentile normalization.
    """
    pl, pu = np.percentile(image, (lb, ub))
    image = (image - pl) / (pu - pl)
    image = np.clip(image, 0, 1)
    return image


from pathlib import Path
import numpy as np
import imageio.v3 as imageio


def load_slice(tomo_id, tomo_dir, slice_id):
    slice_path = Path(tomo_dir) / tomo_id / f"slice_{slice_id:04d}.jpg"
    image = np.array(imageio.imread(slice_path))
    return image


def correct_tomo(x):
    x = (x.astype(np.uint8) + 127).astype(np.float32)
    return x


for tomo_id in anomaly_tomo_ids:
    tomo_path = raw_tomo_path / tomo_id
    depth = len(list(tomo_path.glob("*.jpg")))
    
    slice_id = depth // 2
    image = load_slice(tomo_id, raw_tomo_path, slice_id)
    image_org = image.copy()
    image = correct_tomo(image)

    image_norm = (percentile_norm(image, 1, 99) * 255).astype(np.uint8)

    print(f"{tomo_id=}")

    _, ax = plt.subplots()
    ax.hist(image_org.flatten(), bins=255, range=(0, 255))
    plt.show()

    _, ax = plt.subplots()
    ax.hist(image.flatten(), bins=255, range=(0, 255))
    plt.show()

    _, ax = plt.subplots()
    ax.hist(image_norm.flatten(), bins=255)
    plt.show()

    _, ax = plt.subplots()
    ax.imshow(image_org, cmap="gray")
    plt.show()

    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    plt.show()

    _, ax = plt.subplots()
    ax.imshow(image_norm, cmap="gray")
    plt.show()


def calc_hist(x_hist, lb=1, ub=99, bins=512):
    hist, bins = np.histogram(x_hist, bins=bins)
    cdf = np.cumsum(hist) / hist.sum()
    l_idx = np.searchsorted(cdf, lb / 100)
    u_idx = np.searchsorted(cdf, ub / 100)
    lower, upper = bins[l_idx], bins[u_idx]
    return lower, upper


def process_volume(x, lower, upper):
    x = correct_tomo(x)
    
    x = np.clip(x, lower, upper)
    x = (x - lower) / (upper - lower)

    # Quantize
    x = np.round(x.clip(0, 1) * 255).astype(np.uint8)

    return x


out_dir = Path("volumes")
out_dir.mkdir(exist_ok=True, parents=True)


import os

DEBUG = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == 'Interactive'
DEBUG


from tqdm.auto import tqdm
from PIL import Image


target_tomo_ids = anomaly_tomo_ids
if DEBUG:
    target_tomo_ids = ["tomo_3b8291", "tomo_b18127"]
    target_tomo_ids = ["tomo_b18127"]
for tomo_id in tqdm(target_tomo_ids):
    voxel = load_tomo(
        tomo_id,
        raw_tomo_path,
        stride=1,
    )
    lower, upper = calc_hist(voxel.flatten(), lb=1, ub=99, bins=512)
    voxel_norm = process_volume(voxel, lower, upper)

    voxel_path = out_dir / tomo_id
    voxel_path.mkdir(exist_ok=True, parents=True)
    for slice_id in tqdm(range(voxel.shape[0])):
        slice_path = voxel_path / f"slice_{slice_id:04d}.jpg"
        slice_path.parent.mkdir(exist_ok=True, parents=True)
        img = Image.fromarray(voxel_norm[slice_id])
        img.save(slice_path, format="JPEG", quality=50)


from matplotlib.patches import Rectangle


for tomo_id in anomaly_tomo_ids:
    df = label_df.filter(pl.col("tomo_id") == tomo_id)
    for row in df.iter_rows(named=True):
        has_motor = row["Number of motors"] > 0
        z = (
            row["Motor axis 0"] if has_motor else row["Array shape (axis 0)"] // 2
        )
        y = row["Motor axis 1"] if has_motor else 0
        x = row["Motor axis 2"] if has_motor else 0
        voxel_spacing = row["Voxel spacing"]

        try:
            image = load_slice(
                tomo_id,
                out_dir,
                int(z),
            )
            ax = plt.gca()
            ax.hist(image.flatten(), bins=255)
            ax.set(title=f"{tomo_id=}, {z=}, {y=}, {x=}")
            plt.show()
    
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image, cmap="gray")
            if has_motor:
                s = 1000 / voxel_spacing
                ax.add_patch(
                    Rectangle(
                        (x - s // 2, y - s // 2),
                        s,
                        s,
                        linewidth=0.5,
                        edgecolor="r",
                        facecolor="none",
                    )
                )
                ax.set(title=f"{tomo_id=}, {z=}, {y=}, {x=}")
            plt.show()
        except Exception as e:
            print(f"âš ï¸� Failed to process {tomo_id}: {e}")


!du -sh

