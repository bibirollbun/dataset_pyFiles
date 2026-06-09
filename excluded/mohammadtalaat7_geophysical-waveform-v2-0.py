import os
import glob
import gc
import csv
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Callable, List

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

import time
import datetime
import random

import pathlib, html
from IPython.display import HTML

import openfwi_utils as utils
import openfwi_network as network
import openfwi_transforms as T
from openfwi_scheduler import WarmupMultiStepLR


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
    return [
        f 
        for f in Path(data_dir).rglob("*.npy")
        if ("seis" in f.stem) 
        or ("data" in f.stem)
    ]


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
    return [
        Path(str(f).replace("seis", "vel").replace("data", "model"))
        for f in input_files
    ]


random_model = np.load('/kaggle/input/waveform-inversion/train_samples/FlatFault_A/seis4_1_0.npy')
random_model.shape


print("batch_size : ", random_model.shape[0])
print("num_sources : ", random_model.shape[1])
print("time_steps : ", random_model.shape[2])
print('num_receivers: ',random_model.shape[3])


random_velocity = np.load('/kaggle/input/waveform-inversion/train_samples/FlatFault_A/vel4_1_0.npy')
random_velocity.shape


print("batch_size : ", random_velocity.shape[0])
print("num_sources : ", random_velocity.shape[1])
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


del random_model, random_velocity
gc.collect()


args = {
    # model related
    "model": 'InversionNet', #  'generator name'
    "model_d": None, # 'discriminator name'
    "up_mode": None, # 'upsampling layer mode such as "nearest", "bicubic", etc.'
    "sample_spatial": 1.0, # 'spatial sampling ratio'
    "sample_temporal": 1, # 'temporal sampling ratio'

    # Loss related
    "lambda_g1v": 100.0,
    "lambda_g2v": 0.0,
    "lambda_adv": 1.0,
    "lambda_gp": 10.0,

    # Training ralted
    "k": 1, # 'k in log transformation'
    "weight_decay": 1e-4, # weight decay coefficient in AdamW Optimizer
    "batch_size": 64, # Batch Size in DataLoader objects
    "n_critic": 5, # 'generator & discriminator update ratio'
    "lr_g": 0.0001, # 'initial learning rate of generator'
    "lr_d": 0.0001, # 'initial learning rate of discriminator'
    "lr_milestones": [], # 'decrease lr on milestones'
    "momentum": 0.9, # momentum
    "lr_gamma": 0.1, # 'decrease lr by a factor of lr-gamma'
    "lr_warmup_epochs": 0, # 'number of warmup epochs'
    "epoch_block": 40, # 'epochs in a saved block'
    "num_block": 5, # 'number of saved block'
    "workers": 4, # How many subprocesses to use in loading data
    "print_freq": 20, # 'print frequency'
    "start_epoch": 0, # 'start epoch'

    "pretrained": True,
    "pretrain_path": '/kaggle/input/waveform-inversion-models/pretrained_models/VelocityGAN/flatvel_b_l2_480.pth',

    "resume": None,

    "output_path": '/kaggle/working/',

    "seed": 42,

    "run_train": True,
}


def seed_torch(
    seed_value: int
) -> None:
    """
    Controlling a unified seed value for Python, NumPy, and PyTorch (CPU, GPU).

    Parameters:
    ----------
    seed_value : int
        The unified random seed value.
    """
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_torch(args["seed"])


# Data config from OpenFWI Repository
# https://github.com/lanl/OpenFWI/blob/main/dataset_config.json
data_config = {
    "flatvel-a": {
        "data_min": -26.95,
        "data_max": 52.77,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvevel-a": {
        "data_min": -27.11,
        "data_max": 55.10,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatvel-b": {
        "data_min": -27.17,
        "data_max": 56.05,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvevel-b": {
        "data_min": -29.04,
        "data_max": 57.03,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
	"flatfault-a": {
        "data_min": -26.10,
        "data_max": 50.86,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvefault-a": {
        "data_min": -26.48,
        "data_max": 52.32,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatfault-b": {
        "data_min": -24.86,
        "data_max": 50.28,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvefault-b": {
        "data_min": -24.93,
        "data_max": 50.98,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "style-a": {
        "data_min": -24.96,
        "data_max": 48.93,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "style-b": {
        "data_min": -23.76,
        "data_max": 46.01,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatvel-tutorial": {
        "data_min": -26.95,
        "data_max": 52.77,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 120,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    }
}


class SeismicDataset(Dataset):
    """
    PyTorch Dataset for handling seismic data with multiple examples per file.

    This dataset supports loading seismic data from `.npy` files, optionally preloading
    into memory, and applying transformations to both input data and labels.

    Attributes
    ----------
    in_files : list of str
        List of paths to input seismic data files (`.npy` format).
    out_files : list of str or None
        List of paths to corresponding label files (`.npy` format).
        If `None`, labels are omitted.
    preload : bool, optional (default=True)
        If `True`, loads all data into memory during initialization for faster access.
    sample_ratio : int, optional (default=1)
        Downsampling ratio applied along the seismic time axis
        (e.g., `2` for half resolution).
    examples_per_file : int, optional (default=500)
        Number of samples (examples) contained in each `.npy` file.
    transform_data : torchvision.transforms.Compose, optional (default=None)
        Transformations applied to the input seismic data.
    transform_label : torchvision.transforms.Compose, optional (default=None)
        Transformations applied to the labels.
    """
    def __init__(
        self,
        in_files: list,
        out_files: list,
        preload: bool = True,
        sample_ratio: int = 1,
        examples_per_file: int = 500,
        transform_data = None,
        transform_label = None
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        in_files : list of str
            Paths to input seismic data files (`.npy` format).
        out_files : list of str or None
            Paths to label files (`.npy` format).
            If `None`, labels are omitted.
        preload : bool, optional (default=True)
            Preload all data into memory for faster access during training.
        sample_ratio : int, optional (default=1)
            Downsampling ratio for the seismic time axis.
        examples_per_file : int, optional (default=500)
            Number of examples per `.npy` file.
        transform_data : torchvision.transforms.Compose, optional (default=None)
            Transforms for input data.
        transform_label : torchvision.transforms.Compose, optional (default=None)
            Transforms for labels.
        """
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.examples_per_file = examples_per_file
        self.transform_data = transform_data
        self.transform_label = transform_label
        if out_files is not None:
            self.batches = [
                str(in_files[i])
                +'&'
                +str(out_files[i]) for i in range(len(in_files))
            ]
        else:
            self.batches = [str(in_files[i]) for i in range(len(in_files))]
        if preload: 
            self.data_list, self.label_list = [], []
            for batch in self.batches: 
                data, label = self.load_every(batch)
                self.data_list.append(data)
                if label is not None:
                    self.label_list.append(label)

    def load_every(
        self,
        batch: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load a batch of data and labels from file paths.

        Parameters
        ----------
        batch : str
            String formatted as `"data_path&label_path"` or `"data_path"` (if no labels).

        Returns
        -------
        data : np.ndarray
            Loaded seismic data.
        label : np.ndarray or None
            Loaded labels (if available), otherwise `None`.
        """
        batch = batch.split('&')
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')
        if len(batch) > 1:
            label_path = batch[1]
            label = np.load(label_path)
            label = label.astype('float32')
        else:
            label = None
        
        return data, label

    def __len__(self) -> int:
        """Total number of samples in the dataset."""
        return len(self.batches) * self.examples_per_file

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, np.ndarray]]:
        """
        Retrieve a single sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample to fetch.

        Returns
        -------
        data : torch.Tensor
            Transformed seismic data sample.
        label : torch.Tensor or np.ndarray
            Transformed label (if available), else an empty array.
        """
        batch_idx, sample_idx = (
            idx // self.examples_per_file,
            idx % self.examples_per_file
        )
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx] if len(
                self.label_list
            ) != 0 else None
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        return data, label if label is not None else np.array([])


inputs_all = collect_input_files(TRAIN_DIR)
outputs_all = map_input_to_output(inputs_all)

# Check all output files exist
assert all(f.exists() for f in outputs_all)


# Split dataset into training and validation based on sampling frequency
train_inputs = [inputs_all[i] for i in range(0, len(inputs_all), 2)]
valid_inputs = [f for f in inputs_all if f not in train_inputs]
train_outputs = map_input_to_output(train_inputs)
valid_outputs = map_input_to_output(valid_inputs)


transform_data = Compose([
    T.LogTransform(k=1),
    T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
])

transform_label = Compose([
    T.MinMaxNormalize(2000, 6000)
])


dataset = SeismicDataset(
    train_inputs[:1],
    train_outputs[:1],
    transform_data = transform_data,
    transform_label = transform_label,
    examples_per_file = 500
)
data, label = dataset[0]
print(data.shape)
print(label is None)
print(label.shape)

del dataset, data, label
gc.collect()


ctx = data_config['flatfault-b']

log_data_min = T.log_transform(ctx['data_min'], k=args["k"])
log_data_max = T.log_transform(ctx['data_max'], k=args["k"])

transform_data = Compose([
    T.LogTransform(k=args["k"]),
    T.MinMaxNormalize(log_data_min, log_data_max)
])

transform_label = Compose([
    T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
])

if not args["run_train"]:
    train_inputs = train_inputs[:10]
    train_outputs = train_outputs[:10]

    valid_inputs = valid_inputs[:10]
    valid_outputs = valid_outputs[:10]
    
dataset_train = SeismicDataset(
        train_inputs,
        train_outputs,
        preload=True,
        sample_ratio=args["sample_temporal"],
        examples_per_file=ctx['file_size'],
        transform_data=transform_data,
        transform_label=transform_label
    )

dataset_valid = SeismicDataset(
    valid_inputs,
    valid_outputs,
    preload=True,
    sample_ratio=args["sample_temporal"],
    examples_per_file=ctx['file_size'],
    transform_data=transform_data,
    transform_label=transform_label
)

train_sampler = RandomSampler(dataset_train)
valid_sampler = RandomSampler(dataset_valid)


dataloader_train = DataLoader(
    dataset_train, # Dataset from which to load the data.
    batch_size=args["batch_size"], # How many samples per batch to load.
    sampler=train_sampler, # Defines the strategy to draw samples from the dataset.
    num_workers=args["workers"], # How many subprocesses to use for data loading.
    pin_memory=True, # Copy Tensors into device/CUDA pinned memory before returning them.
    drop_last=True, # Set to True to drop the last incomplete batch.
    collate_fn=default_collate, # Merges a list of samples to form a mini-batch of Tensor(s).
    persistent_workers = True
)

dataloader_valid = DataLoader(
    dataset_valid, # Dataset from which to load the data.
    batch_size=args["batch_size"], # How many samples per batch to load.
    sampler=valid_sampler, # Defines the strategy to draw samples from the dataset.
    num_workers=args["workers"], # How many subprocesses to use for data loading.
    pin_memory=True, # Copy Tensors into device/CUDA pinned memory before returning them.
    collate_fn=default_collate, # Merges a list of samples to form a mini-batch of Tensor(s).
    persistent_workers = True
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = network.model_dict[args["model"]](
    upsample_mode=args["up_mode"],
    sample_spatial=args["sample_spatial"],
    sample_temporal=args["sample_temporal"]
).to(device)

if args["model_d"] is not None:
    model_d = network.model_dict[args["model_d"]]().to(device)
else:
    model_d = None


# Scale lr according to effective batch size
lr_g = args["lr_g"]
optimizer_g = torch.optim.AdamW(
    model.parameters(),
    lr=lr_g,
    betas=(0, 0.9),
    weight_decay=args["weight_decay"]
)

# Conditionally create discriminator optimizer
optimizer_d = None
if model_d is not None:
    lr_d = args["lr_d"]
    optimizer_d = torch.optim.AdamW(
        model_d.parameters(),
        lr=lr_d,
        betas=(0, 0.9),
        weight_decay=args["weight_decay"]
    )

# Convert scheduler to be per iteration instead of per epoch
warmup_iters = args["lr_warmup_epochs"] * len(dataloader_train)
lr_milestones = [len(dataloader_train) * m for m in args["lr_milestones"]]

# Create schedulers only for existing optimizers
optimizers = [optimizer_g]
if model_d is not None:
    optimizers.append(optimizer_d)

lr_schedulers = [
    WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=args["lr_gamma"],
        warmup_iters=warmup_iters,
        warmup_factor=1e-5
    ) for optimizer in optimizers
]

model_without_ddp = model
model_d_without_ddp = model_d if model_d is not None else None

if args["resume"]:
    checkpoint = torch.load(args["resume"], map_location='cpu')
    model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
    
    # Only load discriminator components if they exist
    if model_d is not None:
        model_d_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model_d']))
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
    
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    args.start_epoch = checkpoint['epoch'] + 1
    step = checkpoint['step']
    
    for i in range(len(lr_schedulers)):
        lr_schedulers[i].load_state_dict(checkpoint['lr_schedulers'][i])
    for lr_scheduler in lr_schedulers:
        lr_scheduler.milestones = lr_milestones
    
if args["pretrained"] and args["run_train"]:
    checkpoint = torch.load(args["pretrain_path"])
    model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
    
    # Only load discriminator if it exists
    if model_d is not None:
        model_d_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model_d']))


l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

def criterion_g(
    pred: torch.Tensor,
    gt: torch.Tensor,
    model_d: Optional[torch.nn.Module] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generator loss function combining L1, L2, and adversarial losses.

    Computes a weighted sum of L1 (MAE) and L2 (MSE) losses between predictions and ground truth.
    If a discriminator model (`model_d`) is provided, includes an adversarial loss term to improve
    generative performance.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted output tensor from the generator model.
    gt : torch.Tensor
        Ground truth tensor with the same shape as `pred`.
    model_d : torch.nn.Module, optional (default=None)
        Discriminator model used for adversarial loss computation.
        If `None`, adversarial loss is omitted.

    Returns
    -------
    loss : torch.Tensor
        Total generator loss (weighted sum of all components).
    loss_g1v : torch.Tensor
        L1 loss term (MAE) between `pred` and `gt`.
    loss_g2v : torch.Tensor
        L2 loss term (MSE) between `pred` and `gt`.

    Notes
    -----
    - Loss weights (`lambda_g1v`, `lambda_g2v`, `lambda_adv`) are read from a global `args` dictionary.
    - Adversarial loss is computed as `-torch.mean(model_d(pred))` to encourage the generator to fool the discriminator.
    """
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    loss = args["lambda_g1v"] * loss_g1v + args["lambda_g2v"] * loss_g2v
    if model_d is not None:
        loss_adv = -torch.mean(model_d(pred))
        loss += args["lambda_adv"] * loss_adv
    return loss, loss_g1v, loss_g2v

if model_d is not None:
    criterion_d = utils.Wasserstein_GP(device, args["lambda_gp"])
else:
    criterion_d = None


def train_one_epoch(
    model: torch.nn.Module,
    criterion_g: Callable,
    optimizer_g: torch.optim.Optimizer,
    lr_schedulers: List[torch.optim.lr_scheduler._LRScheduler],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int,
    model_d: Optional[torch.nn.Module] = None,
    criterion_d: Optional[Callable] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    n_critic:int = 5
) -> None:
    """
    Train the model for one epoch with optional adversarial training.

    Supports two modes of operation:
    1. Standard training (when model_d is None): Only updates the main model
    2. GAN training (when model_d provided): Alternates between discriminator
       and generator updates according to n_critic schedule

    Parameters
    ----------
    model : torch.nn.Module
        Primary model (generator in GAN mode) to be trained.
    model_d : torch.nn.Module, optional
        Discriminator model for adversarial training. If None, runs in standard mode.
    criterion_g : Callable
        Loss function for the main model. Signature:
        - GAN mode: f(pred, gt, model_d) -> (total_loss, l1_loss, l2_loss)
        - Standard mode: f(pred, gt) -> (total_loss, l1_loss, l2_loss)
    criterion_d : Callable, optional
        Discriminator loss function. Required if model_d is provided.
        Signature: f(gt, pred, model_d) -> (total_loss, diff_loss, gp_loss)
    optimizer_g : torch.optim.Optimizer
        Optimizer for the main model.
    optimizer_d : torch.optim.Optimizer, optional
        Optimizer for discriminator. Required if model_d is provided.
    lr_schedulers : List[torch.optim.lr_scheduler._LRScheduler]
        Learning rate schedulers to update after each batch.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing (data, label) batches.
    device : torch.device
        Device to run training on (e.g., 'cuda' or 'cpu').
    epoch : int
        Current epoch number (for logging purposes).
    print_freq : int
        Frequency in batches to print training metrics.
    n_critic : int, default=5
        In GAN mode: number of discriminator updates per generator update.

    Notes
    -----
    - Uses a global `step` counter which should be initialized outside this function.
    - For GAN training (model_d provided), all GAN-related parameters must be specified.
    - Learning rate schedulers are stepped after every batch.
    - Metrics tracked:
        * lr_g: Learning rate of main model
        * lr_d: Learning rate of discriminator (GAN mode only)
        * samples/s: Processing speed
        * loss_g1v: L1 loss component
        * loss_g2v: L2 loss component
        * loss_diff: Discriminator real/fake loss (GAN mode only)
        * loss_gp: Gradient penalty loss (GAN mode only)

    Examples
    --------
    >>> # Standard training
    >>> train_one_epoch(
    ...     model=generator,
    ...     criterion_g=simple_loss,
    ...     optimizer_g=opt_g,
    ...     lr_schedulers=[scheduler_g],
    ...     dataloader=train_loader,
    ...     device='cuda',
    ...     epoch=0,
    ...     print_freq=100
    ... )

    >>> # GAN training
    >>> train_one_epoch(
    ...     model=generator,
    ...     model_d=discriminator,
    ...     criterion_g=gan_g_loss,
    ...     criterion_d=gan_d_loss,
    ...     optimizer_g=opt_g,
    ...     optimizer_d=opt_d,
    ...     lr_schedulers=[scheduler_g, scheduler_d],
    ...     dataloader=train_loader,
    ...     device='cuda',
    ...     epoch=0,
    ...     print_freq=100,
    ...     n_critic=5
    ... )
    """
    global step
    model.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr_g',
        utils.SmoothedValue(window_size=1, fmt='{value}')
    )
    if model_d is not None:
        model_d.train()
        
        # Validate GAN mode parameters
        assert criterion_d is not None, "criterion_d required for GAN training"
        assert optimizer_d is not None, "optimizer_d required for GAN training"
        
        metric_logger.add_meter(
            'lr_d',
            utils.SmoothedValue(window_size=1, fmt='{value}')
        )
    metric_logger.add_meter(
        'samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}')
    )
    header = 'Epoch: [{}]'.format(epoch)
    
    itr = 0 # step in this epoch
    max_itr = len(dataloader)


    for data, label in tqdm(
        metric_logger.log_every(dataloader, print_freq, header),
        desc = f"Train Epoch {epoch}",
        leave = False
    ):
        start_time = time.time()
        data, label = data.to(device), label.to(device)

        if model_d is not None:
            # Update discribminator first
            optimizer_d.zero_grad()
            with torch.no_grad():
                pred = model(data)
            loss_d, loss_diff, loss_gp = criterion_d(label, pred, model_d)
            loss_d.backward()
            optimizer_d.step()
            metric_logger.update(loss_diff=loss_diff, loss_gp=loss_gp)

            # Update generator occasionally 
            if ((itr + 1) % n_critic == 0) or (itr == max_itr - 1):
                optimizer_g.zero_grad()
                pred = model(data)
                loss_g, loss_g1v, loss_g2v = criterion_g(pred, label, model_d)
                loss_g.backward()
                optimizer_g.step()
                metric_logger.update(loss_g1v=loss_g1v, loss_g2v=loss_g2v)

            batch_size = data.shape[0]
            metric_logger.update(
                lr_g=optimizer_g.param_groups[0]['lr'],
                lr_d=optimizer_d.param_groups[0]['lr']
            )
            metric_logger.meters['samples/s'].update(
                batch_size / (time.time() - start_time)
            )

        else:
            optimizer_g.zero_grad()
            pred = model(data)
            loss_g, loss_g1v, loss_g2v = criterion_g(pred, label)
            loss_g.backward()
            optimizer_g.step()

            metric_logger.update(loss_g1v=loss_g1v, loss_g2v=loss_g2v)

            batch_size = data.shape[0]
            metric_logger.update(
                lr_g=optimizer_g.param_groups[0]['lr'],
            )
            metric_logger.meters['samples/s'].update(
                batch_size / (time.time() - start_time)
            )
        step += 1
        itr += 1
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()


def evaluate(
    model: torch.nn.Module,
    criterion: Callable,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Evaluate model performance on validation/test data.

    Computes evaluation metrics, denormalizes outputs, and optionally visualizes
    results. Supports distributed evaluation with metric synchronization.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be evaluated.
    criterion : Callable
        Loss function that returns (total_loss, l1_loss, l2_loss).
        Signature: f(pred, label) -> (loss, loss_g1v, loss_g2v).
    dataloader : torch.utils.data.DataLoader
        DataLoader providing (data, label) batches.
    device : torch.device
        Device to run evaluation on (e.g., 'cuda' or 'cpu').
    epoch : int
        Current epoch number (used for visualization and logging).

    Returns
    -------
    avg_loss : float
        Average total loss across all batches.
    l1loss_eval : float
        L1 loss computed on denormalized full dataset.

    Notes
    -----
    - Performs min-max denormalization using global `ctx['label_min/max']`.
    - Visualizes first sample every 4 epochs (matplotlib required).
    - Handles distributed evaluation with `synchronize_between_processes()`.
    - All metrics are computed on denormalized data for final reporting.

    Examples
    --------
    >>> model = MyModel()
    >>> criterion = MyLoss()
    >>> val_loader = DataLoader(val_dataset, batch_size=32)
    >>> avg_loss, l1_loss = evaluate(model, criterion, val_loader, 'cuda', epoch=10)
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data, label in tqdm(
            metric_logger.log_every(dataloader, 20, header),
            desc="Validating",
            leave=False
        ):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pred = model(data)
            loss, loss_g1v, loss_g2v = criterion(pred, label)
            metric_logger.update(
                loss=loss.item(),
                loss_g1v=loss_g1v.item(),
                loss_g2v=loss_g2v.item()
            )

            all_outputs.append(pred.cpu())
            all_labels.append(label.cpu())


    all_output = torch.concat(all_outputs, axis=0)
    all_label = torch.concat(all_labels, axis=0)
    all_output = T.minmax_denormalize(
        all_output,
        ctx['label_min'],
        ctx['label_max']
    )
    all_label = T.minmax_denormalize(
        all_label,
        ctx['label_min'],
        ctx['label_max']
    )
    l1loss_eval = l1loss(all_output, all_label)
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        ' * Loss {loss.global_avg:.8f}, L1_loss {l1loss_eval:.8f} \n'
        .format(
            loss=metric_logger.loss,
            l1loss_eval=l1loss_eval
        )
    )
    
    if epoch % 4 == 0:
        y = all_label[0, 0].detach().cpu()
        y_pred = all_output[0, 0].detach().cpu()
        
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
        fig.suptitle(f'Epoch {epoch} | Valid: {l1loss_eval:.5f}')
        ax[0].imshow(y)
        ax[1].imshow(y_pred)
        plt.show()

    return metric_logger.loss.global_avg, l1loss_eval


step = 0

if args["run_train"]:

    print('Start training')
    start_time = time.time()
    args["epochs"] = args["epoch_block"] * args["num_block"]
    
    best_loss = 5000
    with tqdm(
        range(args["start_epoch"], args["epochs"]),
        desc="Training Progress",
        unit="epoch"
    ) as epoch_pbar:
        
        for epoch in epoch_pbar:
            train_one_epoch(
                model,
                criterion_g,
                optimizer_g,
                lr_schedulers,
                dataloader_train,
                device,
                epoch,
                args["print_freq"],
                model_d,
                criterion_d if model_d else None,
                optimizer_d if model_d else None,
                args["n_critic"] if model_d else 1
            )
        
            loss_global_avg, l1loss_eval = evaluate(
                model,
                criterion_g,
                dataloader_valid,
                device,
                epoch
            )
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'lr_schedulers': [scheduler.state_dict() for scheduler in lr_schedulers],
                'epoch': epoch,
                'step': step,
                'args': args
            }
        
            # Only include GAN components if they exist
            if model_d is not None:
                checkpoint.update({
                    'model_d': model_d_without_ddp.state_dict(),
                    'optimizer_d': optimizer_d.state_dict()
                })
    
            if l1loss_eval < best_loss:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args["output_path"], 'best_model.pth'))
                best_loss = l1loss_eval
        
            utils.save_on_master(
                checkpoint,
                os.path.join(args["output_path"], 'checkpoint.pth'))
        
            # Save checkpoint every epoch block
            if args["output_path"] and (epoch + 1) % args["epoch_block"] == 0:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args["output_path"], 'model_{}.pth'.format(epoch + 1))
                )
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


%%time
test_files = list(Path('/kaggle/input/waveform-inversion/test').glob('*.npy'))
len(test_files)


x_cols = [f'x_{i}' for i in range(1, 70, 2)]
fieldnames = ['oid_ypos'] + x_cols


class TestDataset(Dataset):
    """
    PyTorch Dataset for loading test/validation seismic data files.

    This dataset loads individual seismic data files during inference/prediction,
    optionally applying transformations, and returns both the data and original
    filename (without extension) for tracking purposes.

    Attributes
    ----------
    test_files : list
        Stores the list of input file paths.
    transform_data : callable or None
        Stores the data transformation function.

    Examples
    --------
    >>> from pathlib import Path
    >>> test_files = list(Path('data/test').glob('*.npy'))
    >>> dataset = TestDataset(test_files)
    >>> len(dataset)  # Number of test files
    50
    >>> sample, fname = dataset[0]  # First sample and its filename
    """
    def __init__(
        self,
        test_files: List[Union[str, Path]],
        transform_data: Optional[Callable] = None
    ) -> None:
        """
        Initialize the TestDataset with file paths and optional transforms.

        Parameters
        ----------
        test_files : list of PathLike
            List of paths to seismic data files (typically .npy format).
        transform_data : callable, optional
            Transformations to apply to each loaded sample. If None, no transforms are applied.
            Expected signature: transform(data: np.ndarray) -> transformed_data.
        """
        self.test_files = test_files
        self.transform_data = transform_data


    def __len__(self) -> int:
        """
        Return the number of test files in the dataset.

        Returns
        -------
        int
            Number of samples/files in the dataset.
        """
        return len(self.test_files)


    def __getitem__(
        self,
        i: int
    ) -> Tuple[np.ndarray, str]:
        """
        Load and return the i-th sample from the dataset.

        Parameters
        ----------
        i : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            Contains:
            - data : np.ndarray
                Loaded seismic data array
            - str
                Base filename (without extension) of the loaded file

        Notes
        -----
        - Automatically handles pathlib.Path or string file paths
        - Applies transforms if transform_data was specified
        - Uses numpy.load() for loading .npy files
        """
        test_file = self.test_files[i]
        data = np.load(test_file)
        if self.transform_data:
            data = self.transform_data(data)

        return data, test_file.stem


ctx_test = data_config['flatfault-b']


log_data_min = T.log_transform(ctx_test['data_min'], k=args["k"])
log_data_max = T.log_transform(ctx_test['data_max'], k=args["k"])
transform_data = Compose([
    T.LogTransform(k=args["k"]),
    T.MinMaxNormalize(log_data_min, log_data_max)
])


ds = TestDataset(test_files, transform_data)
dl = DataLoader(
    ds,
    batch_size=8,
    num_workers=4,
    pin_memory=True
)


checkpoint = torch.load('/kaggle/input/openfwi-gans/pytorch/default/2/best_model(1).pth')

model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))

# Train
model.eval()
with open('submission.csv', 'wt', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for inputs, oids_test in tqdm(dl, desc='test'):
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = model(inputs)

        y_preds = outputs[:, 0].cpu().numpy()
        y_preds = T.minmax_denormalize(
            y_preds,
            ctx_test['label_min'],
            ctx_test['label_max']
        )
        
        for y_pred, oid_test in zip(y_preds, oids_test):
            for y_pos in range(70):
                row = dict(
                    zip(
                        x_cols,
                        [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]
                    )
                )
                row['oid_ypos'] = f"{oid_test}_y_{y_pos}"
            
                writer.writerow(row)



#checkpoint = torch.load('/kaggle/input/openfwi-gans/pytorch/default/1/best_model.pth')
#model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))

# Ensure the model is in evaluation mode
#model.eval()

# Export the model to ONNX
# Get a sample input batch from the DataLoader
#sample_inputs, _ = next(iter(dl))
#sample_inputs = sample_inputs.to(device)

# Export the model
#torch.onnx.export(
#    model,  # Model to export
#    sample_inputs,  # Example input
#    "model.onnx",  # Output file name
#    export_params=True,  # Include model parameters
#    opset_version=12,  # ONNX opset version
#    do_constant_folding=True,  # Optimize constants
#    input_names=['input'],  # Input tensor name
#    output_names=['output'],  # Output tensor name
#    dynamic_axes={  # Allow dynamic batch size
#        'input': {0: 'batch_size'},
#        'output': {0: 'batch_size'}
#    }
#)




