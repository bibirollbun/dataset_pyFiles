# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    break
    for filename in filenames:
        continue
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
localizer_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")

print(train_df.head())
print(localizer_df.head())



import pydicom
import os
import numpy as np

# Example: load one scan series
series_uid = train_df["SeriesInstanceUID"].iloc[0]
series_path = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_uid}/"

# Read all slices
slices = []
for fname in sorted(os.listdir(series_path)):
    dcm = pydicom.dcmread(os.path.join(series_path, fname))
    img = dcm.pixel_array
    slices.append(img)

# Convert to 3D volume (H, W, D)
volume = np.stack(slices, axis=-1)
print(volume.shape)



import nibabel as nib

seg_path = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381.nii"
seg = nib.load(seg_path)
seg_data = seg.get_fdata()

print(seg_data.shape)  # 3D mask volume



from sklearn.model_selection import train_test_split

train_ids, val_ids = train_test_split(train_df["SeriesInstanceUID"], test_size=0.2, random_state=42)


# complete_rewrite_aneurysm_dataset_fixed.py
import os
import random
import ast
import re
from glob import glob

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from scipy.ndimage import zoom
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------
def parse_coords_field(coords):
    """
    Robustly parse the 'coordinates' field from train_localizers.csv.
    Accepts:
      - dict-like string: "{'x': 258.36, 'y': 261.36}"
      - list-like string: "[258.36, 261.36]"
      - tuple-like string: "(258.36, 261.36)"
      - actual dict/list/tuple/np array
      - fallback: extract first two floats from string
    Returns (x, y) floats or None.
    """
    if coords is None:
        return None
    if isinstance(coords, (list, tuple, np.ndarray)):
        try:
            return float(coords[0]), float(coords[1])
        except Exception:
            return None
    if isinstance(coords, dict):
        try:
            x = coords.get("x") or coords.get("X")
            y = coords.get("y") or coords.get("Y")
            return float(x), float(y)
        except Exception:
            return None
    if isinstance(coords, str):
        s = coords.strip()
        try:
            val = ast.literal_eval(s)
            return parse_coords_field(val)
        except Exception:
            nums = re.findall(r"[-+]?\d*\.?\d+|\d+", s)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
            return None
    return None


def load_dicom_series(series_path, sort_by_instance=True):
    """
    Load DICOM files from a series folder.
    Returns:
      - volume: numpy array (D, H, W) dtype float32
      - sop_uids: list of SOPInstanceUID aligned with slices
      - meta_list: list of pydicom Dataset objects (ordered)
    """
    files = sorted(glob(os.path.join(series_path, "*.dcm")))
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOMs found in {series_path}")

    slices = []
    instance_nums = []
    sop_uids = []
    meta_list = []

    for f in files:
        d = pydicom.dcmread(f, force=True)
        img = d.pixel_array.astype(np.float32)
        intercept = float(getattr(d, "RescaleIntercept", 0.0))
        slope = float(getattr(d, "RescaleSlope", 1.0))
        img = img * slope + intercept
        slices.append(img)
        instance_nums.append(getattr(d, "InstanceNumber", None))
        sop = getattr(d, "SOPInstanceUID", None) or os.path.basename(f).replace(".dcm", "")
        sop_uids.append(sop)
        meta_list.append(d)

    if sort_by_instance and any(x is not None for x in instance_nums):
        paired = sorted(zip(instance_nums, slices, sop_uids, meta_list),
                        key=lambda x: (x[0] if x[0] is not None else 0))
        slices = [p[1] for p in paired]
        sop_uids = [p[2] for p in paired]
        meta_list = [p[3] for p in paired]

    volume = np.stack(slices, axis=0)  # (D, H, W)
    return volume, sop_uids, meta_list


def load_nifti_mask(seg_dir, series_uid, target_shape=None):
    """
    Robustly find and load segmentation for a series UID.
    Accepts files like:
      {series_uid}.nii(.gz)  OR  {series_uid}_cowseg.nii  OR any file starting with series_uid
    Returns:
      mask numpy array shaped (D, H, W) dtype int16 OR None
    """
    if seg_dir is None:
        return None

    # first try exact names
    candidates = []
    exact1 = os.path.join(seg_dir, f"{series_uid}.nii")
    exact2 = os.path.join(seg_dir, f"{series_uid}.nii.gz")
    if os.path.exists(exact1):
        candidates.append(exact1)
    if os.path.exists(exact2):
        candidates.append(exact2)

    # try any file starting with series_uid
    if len(candidates) == 0:
        patt = os.path.join(seg_dir, f"{series_uid}*.nii*")
        candidates = glob(patt)

    # fallback: any file that contains series_uid in its basename
    if len(candidates) == 0:
        all_files = glob(os.path.join(seg_dir, "*.nii*"))
        candidates = [p for p in all_files if series_uid in os.path.basename(p)]

    if len(candidates) == 0:
        return None

    # prefer exact if present, else pick first sorted candidate
    chosen = None
    for c in candidates:
        base = os.path.basename(c)
        if base == f"{series_uid}.nii" or base == f"{series_uid}.nii.gz":
            chosen = c
            break
    if chosen is None:
        candidates = sorted(candidates)
        chosen = candidates[0]

    try:
        nii = nib.load(chosen)
        mask = nii.get_fdata().astype(np.int16)
    except Exception as e:
        print(f"[load_nifti_mask] failed to load {chosen}: {e}")
        return None

    if mask.ndim != 3:
        return None

    # If target_shape provided (expected D,H,W from dicom stack) try to align
    if target_shape is not None:
        td, th, tw = target_shape  # expected D,H,W
        # common case: mask is H,W,D -> transpose
        if mask.shape == (th, tw, td):
            mask = np.transpose(mask, (2, 0, 1))
        elif mask.shape == (td, th, tw):
            pass  # already (D,H,W)
        else:
            # heuristics: check if any axis equals td (depth)
            if mask.shape[0] == td:
                pass
            elif mask.shape[2] == td:
                mask = np.transpose(mask, (2, 0, 1))
            elif mask.shape[1] == td:
                mask = np.transpose(mask, (1, 2, 0))
            else:
                # final fallback: resample by zoom to target_shape
                try:
                    factors = (td / mask.shape[0], th / mask.shape[1], tw / mask.shape[2])
                    mask = zoom(mask, factors, order=0)
                except Exception:
                    pass
    else:
        # if no target provided, try to guess and put depth first if last axis is smallest
        if mask.shape[2] < mask.shape[0] and mask.shape[2] < mask.shape[1]:
            mask = np.transpose(mask, (2, 0, 1))

    return mask


def find_sop_index(sop_csv, sop_list):
    """
    Robustly match a SOPInstanceUID from CSV to the sop_list from DICOM loading.
    Returns index (int) or None.
    Strategies tried in order:
      - exact match
      - suffix/prefix match
      - last-N chars match
      - substring containment
    """
    if sop_csv is None:
        return None
    sop_csv_s = str(sop_csv).strip()

    # exact
    for i, s in enumerate(sop_list):
        if s is None:
            continue
        if sop_csv_s == str(s).strip():
            return i

    # suffix/prefix
    for i, s in enumerate(sop_list):
        if s is None:
            continue
        s_s = str(s).strip()
        if sop_csv_s.endswith(s_s) or s_s.endswith(sop_csv_s):
            return i

    # last-N chars
    N = 32
    csv_tail = sop_csv_s[-N:]
    for i, s in enumerate(sop_list):
        if s is None:
            continue
        if csv_tail == str(s)[-N:]:
            return i

    # substring
    for i, s in enumerate(sop_list):
        if s is None:
            continue
        if str(s) in sop_csv_s or sop_csv_s in str(s):
            return i

    return None


def normalize_image_slice(slice_2d, clip_pct=(0.5, 99.5)):
    low, high = np.percentile(slice_2d, clip_pct)
    img = np.clip(slice_2d, low, high)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32)


# -------------------------
# Dataset
# -------------------------
class AneurysmDataset(Dataset):
    LOCATION_COLS = [
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation"
    ]

    def __init__(self, base_dir, df, localizer_df=None, seg_dir=None,
                 max_slices=32, slice_strategy="center", target_hw=(512,512),
                 transforms=None, debug=False):
        """
        debug: when True, prints helpful diagnostics for SOP/mask matching
        """
        self.base_dir = base_dir
        self.series_dir = os.path.join(base_dir, "series")
        self.seg_dir = seg_dir
        self.df = df.reset_index(drop=True)
        self.localizer_df = localizer_df
        self.max_slices = max_slices
        self.slice_strategy = slice_strategy
        self.target_hw = target_hw
        self.transforms = transforms
        self.debug = debug

    def __len__(self):
        return len(self.df)

    def _get_row(self, idx):
        return self.df.iloc[idx]

    def _get_localizers_for_series(self, series_uid):
        if self.localizer_df is None:
            return []
        rows = self.localizer_df[self.localizer_df["SeriesInstanceUID"] == series_uid]
        out = []
        for _, r in rows.iterrows():
            coords_raw = r.get("coordinates", None)
            coords_xy = parse_coords_field(coords_raw)
            out.append({
                "SOPInstanceUID": r.get("SOPInstanceUID", None),
                "coordinates_orig": coords_xy,
                "location": r.get("location", None)
            })
        return out

    def __getitem__(self, idx):
        row = self._get_row(idx)
        series_uid = row["SeriesInstanceUID"]
        series_path = os.path.join(self.series_dir, series_uid)

        # load volume & sop list
        volume, sop_list, meta_list = load_dicom_series(series_path)
        volume = volume.squeeze()
        orig_D, orig_H, orig_W = volume.shape

        # load mask (if any)
        mask = None
        if self.seg_dir is not None:
            mask = load_nifti_mask(self.seg_dir, series_uid, target_shape=(orig_D, orig_H, orig_W))
            if self.debug:
                print(f"[DEBUG] series {series_uid} mask found: {mask is not None}")

        # depth handling
        if self.max_slices is not None and self.slice_strategy != "full":
            D, H, W = volume.shape
            if D == self.max_slices:
                vol_crop = volume
                mask_crop = mask
                sop_crop = sop_list
            elif D > self.max_slices:
                if self.slice_strategy == "center":
                    start = max(0, (D - self.max_slices)//2)
                elif self.slice_strategy == "random":
                    start = random.randint(0, D - self.max_slices)
                else:
                    start = 0
                vol_crop = volume[start: start + self.max_slices]
                mask_crop = mask[start: start + self.max_slices] if mask is not None else None
                sop_crop = sop_list[start: start + self.max_slices]
            else:
                factor = self.max_slices / float(D)
                vol_crop = zoom(volume, (factor, 1, 1), order=1)
                mask_crop = zoom(mask, (factor, 1, 1), order=0) if mask is not None else None
                sop_crop = [None] * vol_crop.shape[0]
            volume = vol_crop
            mask = mask_crop
            sop_list = sop_crop

        # record pre-resize dims
        D_before, H_before, W_before = volume.shape

        # resize in-plane to target_hw
        if self.target_hw is not None:
            target_h, target_w = self.target_hw
            D_now, H_now, W_now = volume.shape
            vol_resized = np.zeros((D_now, target_h, target_w), dtype=volume.dtype)
            for i in range(D_now):
                vol_resized[i] = resize(volume[i], (target_h, target_w), order=1, preserve_range=True, anti_aliasing=True)
            volume = vol_resized
            if mask is not None:
                mask_resized = np.zeros((volume.shape[0], target_h, target_w), dtype=mask.dtype)
                for i in range(mask.shape[0]):
                    mask_resized[i] = resize(mask[i], (target_h, target_w), order=0, preserve_range=True, anti_aliasing=False)
                mask = mask_resized

        # after-resize dims
        D_after, H_after, W_after = volume.shape

        # scaling factors for coords
        scale_x = (W_after / float(W_before)) if W_before and W_after else 1.0
        scale_y = (H_after / float(H_before)) if H_before and H_after else 1.0

        # localizers mapping and scaling
        localizers = self._get_localizers_for_series(series_uid)
        for loc in localizers:
            sop_csv = loc.get("SOPInstanceUID", None)
            idx_found = find_sop_index(sop_csv, sop_list)
            loc["slice_index"] = int(idx_found) if idx_found is not None else None
            orig = loc.get("coordinates_orig", None)
            if orig is not None:
                x0, y0 = orig
                x_scaled = x0 * scale_x
                y_scaled = y0 * scale_y
                loc["coordinates_scaled"] = (x_scaled, y_scaled)
                loc["coordinates_norm"] = (x_scaled / float(W_after), y_scaled / float(H_after))
            else:
                loc["coordinates_scaled"] = None
                loc["coordinates_norm"] = None
            if self.debug:
                print(f"[DEBUG] loc sop_csv={str(sop_csv)[:60]} found_idx={loc['slice_index']} scaled={loc['coordinates_scaled']}")

        # normalize and to tensor
        volume_norm = np.stack([normalize_image_slice(s) for s in volume], axis=0)  # (D,H,W)
        volume_norm = np.expand_dims(volume_norm, axis=0).astype(np.float32)        # (1,D,H,W)
        volume_t = torch.from_numpy(volume_norm)

        mask_t = torch.from_numpy(mask.astype(np.int16)) if mask is not None else None

        # labels & meta
        label = int(row.get("Aneurysm Present", 0))
        locations = {col: int(row.get(col, 0)) for col in self.LOCATION_COLS}
        metadata = {
            "PatientAge": row.get("PatientAge", None),
            "PatientSex": row.get("PatientSex", None),
            "Modality": row.get("Modality", None)
        }

        item = {
            "series_uid": series_uid,
            "volume": volume_t,
            "label": label,
            "locations": locations,
            "localizers": localizers,
            "mask": mask_t,  # (D,H,W) tensor or None
            "meta": metadata
        }

        if self.transforms is not None:
            item = self.transforms(item)
        return item


# -------------------------
# Collate and plotting
# -------------------------
def aneurysm_collate(batch):
    volumes = torch.stack([b["volume"] for b in batch], dim=0)  # (B, C, D, H, W)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    series_uids = [b["series_uid"] for b in batch]
    metas = [b.get("meta", {}) for b in batch]
    locations = [b["locations"] for b in batch]
    localizers = [b["localizers"] for b in batch]

    masks_list = [b["mask"] for b in batch]
    if all(m is not None for m in masks_list):
        masks_stacked = torch.stack([m for m in masks_list], dim=0)  # (B, D, H, W)
    else:
        masks_stacked = masks_list

    return {
        "volume": volumes,
        "label": labels,
        "series_uid": series_uids,
        "meta": metas,
        "locations": locations,
        "localizers": localizers,
        "mask": masks_stacked
    }


def plot_two_random_slices_with_annotations(item, figsize=(10, 6), seed=None):
    if seed is not None:
        random.seed(seed)

    vol = item["volume"].numpy()[0]  # (D,H,W)
    D, H, W = vol.shape
    if D == 0:
        raise ValueError("Empty volume")

    if D == 1:
        idxs = [0, 0]
    else:
        idxs = random.sample(range(D), k=2 if D >= 2 else 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Series: {item['series_uid']}  |  Aneurysm: {item['label']}", fontsize=13)

    coords_by_slice = {}
    for loc in item.get("localizers", []):
        slice_idx = loc.get("slice_index", None)
        coords_scaled = loc.get("coordinates_scaled", None)
        if slice_idx is not None and coords_scaled is not None:
            coords_by_slice.setdefault(int(slice_idx), []).append((coords_scaled, loc.get("location", None)))
        elif slice_idx is None and coords_scaled is not None:
            mid = D // 2
            coords_by_slice.setdefault(mid, []).append((coords_scaled, loc.get("location", None)))

    for row_i, slice_idx in enumerate(idxs):
        img = vol[slice_idx]
        ax_raw = axes[row_i, 0]
        ax_overlay = axes[row_i, 1]

        ax_raw.imshow(img, cmap="gray")
        ax_raw.set_title(f"Slice {slice_idx} (raw)")
        ax_raw.axis("off")

        ax_overlay.imshow(img, cmap="gray")
        mask = item.get("mask", None)
        if mask is not None and isinstance(mask, torch.Tensor):
            m = mask.numpy()
            if m.ndim == 3 and m.shape[0] == D:
                mask_slice = m[slice_idx]
                ax_overlay.imshow(np.ma.masked_where(mask_slice == 0, mask_slice), alpha=0.45, cmap="jet")

        coords_list = coords_by_slice.get(slice_idx, [])
        for (coords_scaled, location_name) in coords_list:
            x, y = coords_scaled
            ax_overlay.scatter([x], [y], s=80, marker='x', color='yellow')
            if location_name is not None:
                ax_overlay.text(x + 5, y + 5, location_name, color='yellow', fontsize=8, backgroundcolor='black')

        ax_overlay.set_title(f"Slice {slice_idx} (overlay)")
        ax_overlay.axis("off")

    locs = item.get("locations", {})
    pos = [k for k, v in locs.items() if v == 1]
    pos_txt = ", ".join(pos) if len(pos) > 0 else "No positive location flags"
    meta = item.get("meta", {})
    info_txt = f"Age: {meta.get('PatientAge', 'N/A')}   Sex: {meta.get('PatientSex','N/A')}   Modality: {meta.get('Modality','N/A')}"
    plt.figtext(0.5, 0.02, f"{info_txt}  |  Positive location flags: {pos_txt}", wrap=True, ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# -------------------------
# Example usage snippet (adjust base_dir as appropriate)
# -------------------------
if __name__ == "__main__":
    base_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    df = pd.read_csv(os.path.join(base_dir, "train.csv"))
    local_df = pd.read_csv(os.path.join(base_dir, "train_localizers.csv"))
    seg_dir = os.path.join(base_dir, "segmentations")

    ds = AneurysmDataset(base_dir=base_dir, df=df, localizer_df=local_df, seg_dir=seg_dir,
                         max_slices=32, slice_strategy="center", target_hw=(512, 512), debug=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=aneurysm_collate, num_workers=2, pin_memory=True)

    # load a sample and plot
    sample_item = ds[2]
    print("Series:", sample_item["series_uid"])
    print("Volume shape (C,D,H,W):", sample_item["volume"].shape)
    print("Label:", sample_item["label"])
    print("Localizers (first):", sample_item["localizers"][:2])
    plot_two_random_slices_with_annotations(sample_item, seed=42)




sample_item


df.iloc[0]['SeriesInstanceUID']


df[df['SeriesInstanceUID'] == '1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647']


df


len(df)


np.unique(df['Modality'],return_counts=True)


df['Modality'].value_counts()


df['SeriesInstanceUID'].nunique()


df['Modality'].unique()


local_df['SeriesInstanceUID'].nunique()


local_df


import torch

# Load the pretrained 3D ResNet model
model = torch.hub.load("Warvito/MedicalNet-models", 'medicalnet_resnet10_23datasets')


!wget https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/main/resnet_10_23dataset.pth


import torch
import torch.nn as nn
from torchvision import models

class AneurysmNet(nn.Module):
    def __init__(self, n_modalities=4, metadata_dim=2, pretrained=True):
        super().__init__()

        # Load 3D ResNet backbone from MedicalNet
        self.backbone = torch.hub.load(
            "Warvito/MedicalNet-models", 
            'medicalnet_resnet10_23datasets', 
        )

        if pretrained:
            pretrained_weights = torch.load(
                "/kaggle/working/resnet_10_23dataset.pth", 
                map_location="cpu"
            )
            self.backbone.load_state_dict(pretrained_weights, strict=False)

        # Remove classifier head (fc) -> keep feature extractor
        self.backbone.fc = nn.Identity()

        # Modality embedding
        self.modality_emb = nn.Embedding(n_modalities, 32)

        # Metadata branch
        self.meta_fc = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x, modality_idx=None, metadata=None):
        B = x.shape[0]

        # MedicalNet expects 3D input: [B, C, D, H, W]
        feat = self.backbone(x)  # [B, 512]
        # Global average pooling over D,H,W
        feat = torch.mean(feat, dim=[2,3,4])  # now feat: [B, C]

        if modality_idx is not None:
            m_emb = self.modality_emb(modality_idx)  # [B, 32]
        else:
            m_emb = torch.zeros(B, 32, device=x.device)

        if metadata is not None:
            meta_feat = self.meta_fc(metadata)  # [B, 32]
        else:
            meta_feat = torch.zeros(B, 32, device=x.device)

        combined = torch.cat([feat, m_emb, meta_feat], dim=1)  # [B, 576]
        out = self.classifier(combined)
        return torch.sigmoid(out)



local_df


df


import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------------
# Map modalities to indices
# -------------------------------
modality_map = {"CTA":0, "MRA":1, "MRI T2":2, "MRI T1post":3}

base_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection"
df = pd.read_csv(os.path.join(base_dir, "train.csv"))
local_df = pd.read_csv(os.path.join(base_dir, "train_localizers.csv"))

# -------------------------------
# Helper: prepare batch
# -------------------------------
def prepare_batch(batch):
    volumes = batch['volume'].float()                    # [B,1,D,H,W]
    labels = batch['label'].float().unsqueeze(1)         # [B,1]

    # modality encoding
    modality_idx = torch.tensor(
        [modality_map[m['Modality']] for m in batch['meta']],
        device=volumes.device
    )

    # metadata: Age normalized + Sex encoded
    # metadata: Age normalized + Sex encoded
    ages = torch.tensor(
        [m.get('PatientAge', 0) / 100. for m in batch['meta']],
        device=volumes.device,
        dtype=torch.float32   # <--- enforce float32
    ).unsqueeze(1)
    
    sexes = torch.tensor(
        [0 if m.get('PatientSex','Male')=='Male' else 1 for m in batch['meta']],
        device=volumes.device,
        dtype=torch.float32   # <--- enforce float32
    ).unsqueeze(1)
    
    metadata = torch.cat([ages, sexes], dim=1)

    return volumes, modality_idx, metadata, labels

# -------------------------------
# Dataset split
# -------------------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['Aneurysm Present']
)
train_loc_df = local_df[local_df['SeriesInstanceUID'].isin(train_df['SeriesInstanceUID'])]
val_loc_df   = local_df[local_df['SeriesInstanceUID'].isin(val_df['SeriesInstanceUID'])]

seg_dir = os.path.join(base_dir, "segmentations")

train_dataset = AneurysmDataset(base_dir=base_dir, localizer_df=train_loc_df, df=train_df, seg_dir=seg_dir)
val_dataset   = AneurysmDataset(base_dir=base_dir, localizer_df=val_loc_df, df=val_df, seg_dir=seg_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=aneurysm_collate)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=aneurysm_collate)


# -------------------------------
# Model, optimizer, loss
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AneurysmNet(n_modalities=4, metadata_dim=2, pretrained=True).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# -------------------------------
# Training loop
# -------------------------------
n_epochs = 3
train_losses, val_losses, val_accs = [], [], []

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")

    for batch in pbar:
        volumes, modality_idx, metadata, labels = prepare_batch(batch)
        volumes, modality_idx, metadata, labels = (
            volumes.to(device), modality_idx.to(device), metadata.to(device), labels.to(device)
        )

        optimizer.zero_grad()
        outputs = model(volumes, modality_idx, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * volumes.size(0)
        pbar.set_postfix({"loss": loss.item()})

    train_loss /= len(train_loader.dataset)

    # -------------------------------
    # Validation
    # -------------------------------
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            volumes, modality_idx, metadata, labels = prepare_batch(batch)
            volumes, modality_idx, metadata, labels = (
                volumes.to(device), modality_idx.to(device), metadata.to(device), labels.to(device)
            )

            outputs = model(volumes, modality_idx, metadata)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * volumes.size(0)

            preds = (outputs > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # -------------------------------
    # Plot predictions vs labels
    # -------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(all_labels[:100], "g.-", label="Ground Truth")
    plt.plot(all_preds[:100], "r.-", label="Predictions")
    plt.title(f"Predicted vs Ground Truth (Epoch {epoch+1})")
    plt.xlabel("Sample idx")
    plt.ylabel("Label")
    plt.legend()
    plt.show()

    # -------------------------------
    # Visualize some volumes + predictions
    # -------------------------------
    example_vol = volumes[0,0].cpu().numpy()   # pick first sample, channel 0
    mid_slice = example_vol[example_vol.shape[0]//2]  # middle slice along depth
    plt.figure(figsize=(6,6))
    plt.imshow(mid_slice, cmap="gray")
    plt.title(f"Example slice | True: {labels[0].item()} | Pred: {preds[0].item()}")
    plt.axis("off")
    plt.show()

# -------------------------------
# Plot training curves
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.legend()
plt.title("Training Curves")
plt.show()








