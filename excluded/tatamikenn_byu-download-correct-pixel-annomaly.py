import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
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


import os


IS_INTERACTIVE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'


IS_INTERACTIVE


%%writefile irregular_labels.csv

tomo_id,z
aba2014-02-21-14,18.0
mba2011-02-16-103,62.0
mba2011-02-16-106,66.0
mba2011-02-16-108,42.0
mba2011-02-16-111,56.0
mba2011-02-16-115,70.0
mba2011-02-16-116,68.0
mba2011-02-16-11,66.0
mba2011-02-16-122,42.0
mba2011-02-16-123,62.0
mba2011-02-16-129,64.0
mba2011-02-16-12,60.0
mba2011-02-16-133,59.0
mba2011-02-16-139,60.0
mba2011-02-16-141,66.0
mba2011-02-16-143,57.0
mba2011-02-16-145,48.0
mba2011-02-16-145,51.0
mba2011-02-16-147,51.0
mba2011-02-16-150,62.0
mba2011-02-16-153,55.0
mba2011-02-16-153,63.0
mba2011-02-16-155,33.0
mba2011-02-16-157,60.0
mba2011-02-16-15,56.0
mba2011-02-16-15,50.0
mba2011-02-16-160,60.0
mba2011-02-16-160,51.0
mba2011-02-16-162,62.0
mba2011-02-16-170,62.0
mba2011-02-16-173,52.0
mba2011-02-16-176,59.0
mba2011-02-16-17,68.0
mba2011-02-16-19,70.0
mba2011-02-16-1,66.0
mba2011-02-16-1,46.0
mba2011-02-16-20,65.0
mba2011-02-16-23,59.0
mba2011-02-16-26,59.0
mba2011-02-16-27,63.0
mba2011-02-16-28,59.0
mba2011-02-16-28,68.0
mba2011-02-16-29,65.0
mba2011-02-16-30,66.0
mba2011-02-16-32,75.0
mba2011-02-16-33,56.0
mba2011-02-16-34,54.0
mba2011-02-16-35,57.0
mba2011-02-16-37,70.0
mba2011-02-16-3,63.0
mba2011-02-16-40,59.0
mba2011-02-16-40,55.0
mba2011-02-16-42,44.0
mba2011-02-16-42,26.0
mba2011-02-16-46,64.0
mba2011-02-16-48,63.0
mba2011-02-16-52,59.0
mba2011-02-16-53,54.0
mba2011-02-16-55,46.0
mba2011-02-16-60,54.0
mba2011-02-16-64,56.0
mba2011-02-16-65,66.0
mba2011-02-16-67,60.0
mba2011-02-16-68,60.0
mba2011-02-16-71,59.0
mba2011-02-16-75,47.0
mba2011-02-16-79,43.0
mba2011-02-16-79,44.0
mba2011-02-16-88,69.0
mba2011-02-16-90,60.0
mba2011-02-16-95,63.0


label_df = pl.read_csv("/kaggle/input/cryoet-flagellar-motors-dataset/labels.csv")


irregular_df = pl.read_csv("irregular_labels.csv")
print(irregular_df.shape)
irregular_df.head()


print("Unique Tomograms: {}".format(irregular_df["tomo_id"].n_unique()))


try:
    from cryoet_data_portal import Client
except:
    ! pip install 'zarr<3.0' cryoet_data_portal -q


tomo_ids = irregular_df["tomo_id"].unique(maintain_order=True).to_list()
print("N_TOMOS: {:_}".format(len(tomo_ids)))
tomo_ids


import scipy


def calc_irregular_score(x, bins=256, lb=50, ub=200):
    hist, _ = np.histogram(x, bins=bins)
    return (hist[:lb].sum() + hist[ub:].sum()) / x.size


def process_volume(x, target_shape=(128, 512, 512), lb=1, ub=99, bins=512, resample_z=True):
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
import os
import traceback
import glob
import shutil
import scipy

import zarr
from cryoet_data_portal import Client, Dataset, Run
import numpy as np
from tqdm import tqdm


class CziiCollector():
    def __init__(
        self,
        tmp_dir: str = "/tmp/", 
        out_dir: str = "/kaggle/working/volumes/", 
        img_size: Tuple[int] = (128, 512, 512),
    ):
        super().__init__()

        self.client = Client()
        self.tomo_ids = tomo_ids
        self.tmp_dir = Path(tmp_dir)
        self.out_dir = Path(out_dir)
        self.img_size = img_size

        # Tmp dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

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
                x = zarr.open(zarr_path / "0", mode='r')

                # irregular check
                x = x[:]
                irregular_score = calc_irregular_score(x)
                if irregular_score > irregular_threshold:
                    x = x.astype("uint8").astype("float32")

                # Preprocess
                x = self.process_tomogram(x)

                # Save
                out_path = Path(self.out_dir) / f"{run.name}.npy"
                np.save(out_path, x)
                print(f"Success: {tomo_id}")
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                print(f"Failed: {tomo_id}")

            if not IS_INTERACTIVE:
                self.cleanup()
        return


%%time
if IS_INTERACTIVE:
    run_tomo_ids = ["mba2011-02-16-27", "mba2011-02-16-123", "mba2011-02-16-45"]
    run_tomo_ids = ["mba2011-02-16-27", "mba2011-02-16-45"]
else:
    run_tomo_ids = tomo_ids


p = CziiCollector()
p.run(run_tomo_ids)


for i, tomo_id in enumerate(run_tomo_ids):
    if i > 3:
        break
    df = label_df.filter(pl.col("tomo_id") == tomo_id)

    path = Path("volumes") / f"{tomo_id}.npy"
    x = np.load(path)
    for row in df.iter_rows(named=True):
        z = int(row["z"])
        plt.imshow(x[z], cmap="gray")
        plt.show()


!du -sh volumes

