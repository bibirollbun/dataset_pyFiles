import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_working(directory_path: str = "/kaggle/working/"):
    """
    Clean kaggle output directory.
    """
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            if item == "submission.csv":
                continue
            item_path = os.path.join(directory_path, item)
            os.remove(item_path) if os.path.isfile(item_path) else shutil.rmtree(item_path)
        print(f"All items in '{directory_path}' have been removed.")
    else:
        print(f"'{directory_path}' does not exist.")
        
clean_working()


if not os.path.exists('/tmp/'):
    os.mkdir('/tmp/')


DATA_DIR= "/kaggle/input/cryoet-flagellar-motors-dataset/"
SEED= 0

df= pd.read_csv(os.path.join(DATA_DIR, "labels_new.csv"))
df= df[~df["tomo_id"].str.startswith("tomo_")]
df["coordinates"] = df["coordinates"].apply(lambda x: eval(x))
df= df.reset_index(drop=True)
print(df.shape)
df.head()


print("Unique Tomograms: {}".format(df["tomo_id"].nunique()))


idx= 123
row= df.iloc[idx].to_dict()
row


fpath= os.path.join(DATA_DIR, "volumes_704", row["tomo_id"] + ".npy")
arr= np.load(fpath)
print(arr.shape)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title(row["tomo_id"])
ax.imshow(arr[int(row["z"]), ...], cmap="gray")
ax.scatter(row["coordinates"][0][2]*704, row["coordinates"][0][1]*704, c="red", s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

plt.tight_layout()
plt.show()


# Sample rows
tmp= df.sample(frac=1, random_state=SEED)

# Create figure
fig, axes = plt.subplots(2, 8, figsize=(24, 6))
axes= axes.flatten()

for idx in range(len(axes)):
    row= tmp.iloc[idx].to_dict()

    # Load tomo
    fpath= os.path.join(DATA_DIR, "volumes_704", row["tomo_id"] + ".npy")
    arr= np.load(fpath)

    # Visualize
    frame= int(row["coordinates"][0][0]*128)
    axes[idx].imshow(arr[frame, ...], cmap="gray")
    axes[idx].scatter(row["coordinates"][0][2]*704, row["coordinates"][0][1]*704, c="red", s=50)
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])
    axes[idx].set_frame_on(False)
    
plt.tight_layout()
plt.show()
plt.show()


try:
    from cryoet_data_portal import Client
except:
    !pip install zarr cryoet_data_portal -q


import scipy


def calc_irregular_score(x, bins=256, lb=50, ub=200):
    hist, _ = np.histogram(x, bins=bins)
    return (hist[:lb].sum() + hist[ub:].sum()) / x.size


def process_volume(x, target_shape=(128, 512, 512), lb=1, ub=99, bins=512, resample_z=False):
    # Approximate percentile normalization
    hist, bins = np.histogram(x, bins=bins)
    cdf = np.cumsum(hist) / hist.sum()
    l_idx = np.searchsorted(cdf, lb/100)
    u_idx = np.searchsorted(cdf, ub/100)
    lower, upper = bins[l_idx], bins[u_idx]
    x = np.clip(x, lower, upper)
    x = (x - lower) / (upper - lower)

    if resample_z:
        # Resample
        indices = np.linspace(0, x.shape[0]-1, target_shape[0]).astype(int)
        x= x[indices]    

    # Resize
    zoom_factor = tuple(ts / xs for ts, xs in zip(target_shape, x.shape))
    x = scipy.ndimage.zoom(x, zoom_factor, order=1)

    # Quantize
    x = np.round(x.clip(0, 1) * 255).astype(np.uint8)

    return x


from typing import Tuple
from pathlib import Path
import shutil
import traceback

import zarr
from cryoet_data_portal import Client, Dataset, Run
import numpy as np


class CziiCollector():
    def __init__(
        self,
        tmp_dir: str = "/tmp/", 
        out_dir: str = "/kaggle/working/volumes/", 
        img_size: Tuple[int] = (128, 704, 704),
    ):
        super().__init__()

        self.client = Client()
        self.tmp_dir = Path(tmp_dir)
        self.img_size = img_size

        # Tmp dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self):
        shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def process_tomogram(self, x):
        print("original_shape: {}".format(x.shape))        
        x = process_volume(x, target_shape=self.img_size)
        print("final_shape: {}".format(x.shape))
        return x
        
    def run(self, tomo_ids, irregular_threshold=0.6):
        client = self.client

        for tomo_id in tomo_ids:            
            run = Run.find(client, query_filters=[Run.name == tomo_id])
            if len(run) == 0:
                print("MISSING: ", tomo_id)
                continue
            else:
                run = run[0]

            zarr_path = Path(self.tmp_dir) / f"{run.name}.zarr"
            if not zarr_path.exists():
                # download
                tomo = run.tomograms[0]
                tomo.download_omezarr(dest_path=self.tmp_dir)    

            try:
                # Load tomo
                tomo_pixels= tomo.size_z * tomo.size_x * tomo.size_y
                print("tomo_pixels: {:_}".format(tomo_pixels))

                # Take smaller version of XL tomograms
                if tomo_pixels < 2_000_000_000:
                    x = zarr.open(zarr_path / "0", mode='r')
                else:
                    x = zarr.open(zarr_path / "1", mode='r')

                # irregular check
                x = x[:]
                irregular_score = calc_irregular_score(x)
                if irregular_score > irregular_threshold:
                    x = x.astype("uint8").astype("float32")

                # Preprocess
                x = self.process_tomogram(x)

                # Save
                out_path = f"{run.name}.npy"
                np.save(out_path, x)
                print(f"Success: {tomo_id}")
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                print(f"Failed: {tomo_id}")
            finally:
                print("-"*25)

            self.cleanup()
        return


# Sample a few
IDXS= df["tomo_id"].unique()
IDXS= IDXS[:2]

# Run collection
p= CziiCollector()
p.run(tomo_ids=IDXS)

