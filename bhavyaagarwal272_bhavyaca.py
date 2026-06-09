# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import csv
import pathlib, html
from IPython.display import HTML
import utils
import network
import transforms as T


# Root Paths
BASE_DIR = '/kaggle/input/waveform-inversion'
TRAIN_DIR = os.path.join(BASE_DIR, 'train_samples')
TEST_DIR = os.path.join(BASE_DIR, 'test')


def collect_input_files(
    data_dir: str = TRAIN_DIR
) -> list:
    """
    Recursively search for .npy files in data_dir that contain 'seis' or 'data' in their filename.

    Parameters:
    ----------
    data_dir : str, default = TRAIN_DIR
        The data_dir that need searching.

    Returns:
    -------
    list
        A list contains the paths of all input files in our data.
    """
    return [f for f in Path(data_dir).rglob("*.npy") if ("seis" in f.stem) or ("data" in f.stem)]


def map_input_to_output(
    input_files: list
) -> list:
    """
    Map each input file to its corresponding output file by replacing keywords.

    Parameters:
    ----------
    input_files : list
        The list that contains the paths of all input files in our data.

    Returns:
    -------
    list
        A list contains the paths of all output files in our data.
    """
    return [Path(str(f).replace("seis", "vel").replace("data", "model")) for f in input_files]


random_model = np.load('/kaggle/input/waveform-inversion/train_samples/FlatFault_A/seis4_1_0.npy')
random_model.shape


print("batch_size : ", random_model.shape[0])
print("num_sources, : ", random_model.shape[1])
print("time_steps : ", random_model.shape[2])
print('num_receivers: ',random_model.shape[3])


random_velocity = np.load('/kaggle/input/waveform-inversion/train_samples/FlatFault_A/vel4_1_0.npy')
random_velocity.shape


print("batch_size : ", random_velocity.shape[0])
print("num_sources, : ", random_velocity.shape[1])
print("height : ", random_velocity.shape[2])
print('width: ',random_velocity.shape[3])


ROOT = next(pathlib.Path("/kaggle/input").rglob("train_samples"), None)
assert ROOT and ROOT.is_dir(), "❌  /train_samples not found"

# ---------- helpers ----------
def human_bytes(n):
    units = ['B','KB','MB','GB','TB']; i = 0
    while n >= 1024 and i < len(units)-1: n /= 1024; i += 1
    return f"{n:.1f}{units[i]}"

def npy_info(p):
    arr = np.load(p, mmap_mode='r'); shp, dt = arr.shape, arr.dtype; arr._mmap.close()
    return shp, dt

def folder_desc(name):
    if name.startswith("FlatVel"):   base = "Vel—flat, gently layered"; 
    elif name.startswith("CurveVel"): base = "Vel—curved/folded layers";
    elif name.startswith("FlatFault"): base = "Fault—flat layers with breaks";
    elif name.startswith("CurveFault"):base = "Fault—curved layers with breaks";
    elif name.startswith("Style"):   base = "Style—random texture pattern";
    else:                             base = "Unknown pattern"
    level = "A simpler" if name.endswith("_A") else "B more complex" if name.endswith("_B") else ""
    return f"{base} ({level})".strip()

KIND_INFO = {
    "Data":  "4‑D seismic waveforms (sources × time × receivers)",
    "Model": "2‑D velocity ground truth",
    "Seis":  "Seismic recordings (prefix layout)",
    "Vel":   "Velocity maps (prefix layout)",
    "Files": "Misc. .npy files"
}

# ---------- gather ----------
tree = {}
for fld in sorted(ROOT.iterdir()):
    if not fld.is_dir(): continue
    kinds={}
    def add(k, paths):
        items=[(p.name, f"shape={shp}, dtype={dt}, size={human_bytes(p.stat().st_size)}")
               for p in paths if (shp:=npy_info(p)[0]) or True for dt in [npy_info(p)[1]]]
        if items: kinds[k]={"count":len(items),"files":items}
    add("Data",  (fld/"data").glob("*.npy")  if (fld/"data").is_dir()  else [])
    add("Model", (fld/"model").glob("*.npy") if (fld/"model").is_dir() else [])
    add("Seis", fld.glob("seis*.npy"));  add("Vel", fld.glob("vel*.npy"))
    if not kinds: add("Files", fld.glob("*.npy"))
    tree[fld.name]={"desc":folder_desc(fld.name),"kinds":kinds}

# ---------- html ----------
html_parts=["""
<style>
:root{--accent:#136efd;--bg:#fafafa;--border:#ddd;--font:system-ui,sans-serif}
#exp{font-family:var(--font);background:var(--bg);padding:1rem;border:1px solid var(--border);
     border-radius:8px;max-width:1050px;margin:auto}
#exp h2{margin:0 0 1rem;text-align:center;font-size:1.35rem}
.folder{border:1px solid var(--border);border-radius:6px;margin:.7rem 0;background:#fff}
.folder summary{cursor:pointer;display:flex;gap:.6rem;align-items:center;padding:.5rem .75rem}
.fname{font-weight:600}
.fdesc{font-size:.8rem;color:#555}
.kind-line{margin-left:1.3rem;margin-top:.45rem}
.kind-tag{background:var(--accent);color:#fff;border-radius:4px;padding:.18rem .55rem;font-size:.75rem}
.kind-info{font-size:.78rem;color:#222;margin-left:.45rem}
.count{color:#555;font-size:.78rem;margin-left:.25rem}
.file-list{margin:.25rem 0 .8rem 2.5rem;font-size:.85rem;color:#333}
.file-list li{margin:.04rem 0;list-style-type:disc}
</style>
<div id="exp">
  <h2>Training .npy Explorer — Inline Explanations</h2>
"""]

for name,info in tree.items():
    kinds=info["kinds"]; desc=html.escape(info["desc"])
    html_parts.append("<details class='folder'>")
    html_parts.append(f"<summary><span class='fname'>{html.escape(name)}</span>"
                      f"<span class='fdesc'>— {desc}</span></summary>")
    for kind,meta in kinds.items():
        html_parts.append(f"<div class='kind-line'><span class='kind-tag'>{kind}</span>"
                          f"<span class='count'>({meta['count']})</span>"
                          f"<span class='kind-info'>{html.escape(KIND_INFO.get(kind,''))}</span></div>")
        html_parts.append("<ul class='file-list'>")
        for fn,tip in meta["files"]:
            html_parts.append(f"<li title='{html.escape(tip)}'>{html.escape(fn)}</li>")
        html_parts.append("</ul>")
    html_parts.append("</details>")

html_parts.append("</div>")
HTML("".join(html_parts))


batch_size, num_sources, time_steps, num_receivers = random_model.shape
print(f"Data form: {random_model.shape}")


selected_batch = 0

fig, axes = plt.subplots(nrows=num_sources, figsize=(20, 25), sharex=True, sharey=True)
for source in range(num_sources):
    data = random_model[selected_batch, source, :, :].T
    im = axes[source].imshow(
        data,
        cmap='seismic',
        aspect='auto',
        extent=[0, time_steps, num_receivers, 0],
        vmin=-np.abs(data).max(),
        vmax=np.abs(data).max()
    )
    axes[source].set_title(f'Source {source}', pad=15)
    axes[source].set_ylabel('Receiver number')
    axes[source].set_xlabel('Time step' if source == num_sources-1 else '')
plt.tight_layout()
plt.show();


selected_batch = 0
selected_source = 2

data = random_model[selected_batch, selected_source, :, :]
X, Y = np.meshgrid(np.arange(num_receivers), np.arange(time_steps))
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, data,cmap='seismic', rstride=1, cstride=1)
ax.set_title(f'3D visualization of seismic data (batch={selected_batch}, source={selected_source})')
ax.set_xlabel('Receiver number')
ax.set_ylabel('Time step')
ax.set_zlabel('The amplitude')
fig.colorbar(surf, ax=ax, shrink=0.5, label='The amplitude')
plt.show();


class SeismicDataset(Dataset):
    """
    Dataset handling seismic files with multiple examples per file.
    """
    def __init__(self, in_files: list, out_files: list, examples_per_file: int = 500):
        assert len(in_files) == len(out_files)
        self.in_files = in_files
        self.out_files = out_files
        self.examples_per_file = examples_per_file

    def __len__(self):
        return len(self.in_files) * self.examples_per_file

    def __getitem__(self, idx: int):
        file_index = idx // self.examples_per_file
        sample_index = idx % self.examples_per_file

        # Memory map the file to reduce memory usage
        x_data = np.load(self.in_files[file_index], mmap_mode="r")
        y_data = np.load(self.out_files[file_index], mmap_mode="r")
        try:
            return x_data[sample_index].copy(), y_data[sample_index].copy()
        finally:
            del x_data, y_data


inputs_all = collect_input_files(TRAIN_DIR)
outputs_all = map_input_to_output(inputs_all)

# Check all output files exist
assert all(f.exists() for f in outputs_all)


train_inputs = [inputs_all[i] for i in range(0, len(inputs_all), 2)]
valid_inputs = [f for f in inputs_all if f not in train_inputs]
train_outputs = map_input_to_output(train_inputs)
valid_outputs = map_input_to_output(valid_inputs)


train_dataset = SeismicDataset(train_inputs, train_outputs, examples_per_file=500)
valid_dataset = SeismicDataset(valid_inputs, valid_outputs, examples_per_file=500)


train_loader = DataLoader(
    train_dataset, # Dataset from which to load the data.
    batch_size=64, # How many samples per batch to load.
    shuffle=True, # Set to True to have the data reshuffled at every epoch.
    pin_memory=True, # Copy Tensors into device/CUDA pinned memory before returning them. 
    drop_last=True, # Set to True to drop the last incomplete batch.
    num_workers=4, # How many subprocesses to use for data loading.
    persistent_workers=True # Will not shutdown the worker processes after a dataset has been consumed once.
)

valid_loader = DataLoader(
    valid_dataset, # Dataset from which to load the data.
    batch_size=64, # How many samples per batch to load.
    shuffle=True, # Set to True to have the data reshuffled at every epoch
    pin_memory=True, # Copy Tensors into device/CUDA pinned memory before returning them.
    drop_last=True, # Set to True to drop the last incomplete batch.
    num_workers=4, # How many subprocesses to use for data loading.
    persistent_workers=True # Will not shutdown the worker processes after a dataset has been consumed once.
)


train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

