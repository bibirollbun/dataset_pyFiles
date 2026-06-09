!pip install -q git+https://github.com/qubvel/segmentation_models.pytorch


## IMPORTS ##
import os
import glob
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
from tqdm import tqdm
import gc
from torch.utils.data import Dataset, DataLoader, ConcatDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENCODER_NAME = 'resnet18'

## MRI SIZE ##
PATCH_H = 512
PATCH_W = 512

BASE_PATH_IMG = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
SAGITTAL_MODEL_PATH = '/kaggle/input/lumbar-spine-keypoint-detection-models/Sagittal_T2_sagittal_level_segmentation_2_v2'
AXIAL_MODEL_PATH = '/kaggle/input/lumbar-spine-keypoint-detection-models/Axial_T2_axial_side_segmentation_1'


class myUNet(nn.Module):
    def __init__(self, classes):
        super(myUNet, self).__init__()
        self.classes = classes
        self.UNet = smp.Unet(
            encoder_name=ENCODER_NAME,
            classes=classes,
            in_channels=1
        ).to(DEVICE)

    def forward(self, X):
        H, W = X.shape[-2:]
        x = self.UNet(X.view(-1, 1, H, W)).view(-1, H*W)
        # MinMaxScaling along the class plane to generate a heatmap
        min_values = x.min(-1)[0].view(-1, 1)
        max_values = x.max(-1)[0].view(-1, 1)
        d = (max_values - min_values)
        d[d == 0] = 1
        x = (x - min_values) / d
        return x.view(-1, self.classes, H, W)


def pixel_to_patient_3d(dcm, x, y):
    """
    Converts 2D pixel coordinates (x, y) to 3D Patient Coordinates (x, y, z).
    x: Column index
    y: Row index
    """
    ipp = np.asarray(dcm.ImagePositionPatient, dtype=np.float64)
    iop = np.asarray(dcm.ImageOrientationPatient, dtype=np.float64)
    row_cosines = iop[:3]
    col_cosines = iop[3:]
    row_spacing, col_spacing = map(float, dcm.PixelSpacing)

    return (
        ipp
        + x * col_spacing * col_cosines
        + y * row_spacing * row_cosines
    )


def collect_dicom_files(series_folder):
    patterns = ["*.dcm", "*.DCM", "*.dicom"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(series_folder, p)))
    # Sort by instance number to ensure correct volume order
    files.sort(key=lambda f: pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
    return files


def load_models():
    """Loads the models. Requires the class definition above to be in scope."""
    print("Loading Sagittal Model...")
    sag_model = torch.load(SAGITTAL_MODEL_PATH, map_location=DEVICE, weights_only=False)
    sag_model.to(DEVICE)
    sag_model.eval()

    print("Loading Axial Model...")
    ax_model = torch.load(AXIAL_MODEL_PATH, map_location=DEVICE, weights_only=False)
    ax_model.to(DEVICE)
    ax_model.eval()
    
    return sag_model, ax_model


torch_resize = torchvision.transforms.Resize((PATCH_H, PATCH_W), antialias=True)

def preprocess_image(pixel_array):
    image = pixel_array.astype(np.float32)
    H, W = image.shape
    
    h_start, w_start = 0, 0
    crop_size = 0
    
    if H > W:
        crop_size = W
        h_start = (H - crop_size) // 2
        image = image[h_start : h_start + crop_size, :]
    elif H < W:
        crop_size = H
        w_start = (W - crop_size) // 2
        image = image[:, w_start : w_start + crop_size]
    else:
        crop_size = H
        
    img_max = np.max(image)
    if img_max > 0:
        image = image / img_max
        
    img_tensor = torch.tensor(image).unsqueeze(0) 

    ### IMPORTANT ### 
    # use the torchvision Resize for resizing, torch.interpolate gave bad results
    # Resize to PATCH_H, PATCH_W using torchvision (Training standard)
    img_tensor = torch_resize(img_tensor)    
    img_tensor = img_tensor.unsqueeze(0).float().to(DEVICE)
    
    return img_tensor, (h_start, w_start, crop_size), (H, W)


def extract_coordinates(heatmaps, crop_info, original_shape):
    """
    Converts model output heatmaps back to original image coordinates.
    """
    h_start, w_start, crop_size = crop_info
    # orig_H, orig_W = original_shape # Not strictly needed for coord calc, but good for validation
    
    bs, n_classes, h_map, w_map = heatmaps.shape
    coords = []
    
    for c in range(n_classes):
        hm = heatmaps[0, c, :, :].detach().cpu().numpy()
        
        # Argmax gives (row, col) i.e. (y, x)
        y_idx, x_idx = np.unravel_index(np.argmax(hm), hm.shape)
        
        # Map resize -> Crop
        # Note: h_map and w_map should be 512
        x_crop = x_idx * (crop_size / w_map)
        y_crop = y_idx * (crop_size / h_map)
        
        # Map Crop -> Original
        x_orig = x_crop + w_start
        y_orig = y_crop + h_start
        
        coords.append((x_orig, y_orig))
        
    return coords


## LOAD THE MODELS
sagittal_model, axial_model = load_models()


## Logic to perform inference on data. 
import pandas as pd
train_coords = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv")
train_desc = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv")
sample_data = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")
train_coords.drop(columns=['instance_number'], inplace = True)
train_desc = train_desc[train_desc['series_description'] != 'Sagittal T1']


df = sample_data.merge(train_desc, on=['study_id'], how='inner')


df


levels = [
    "spinal_canal_stenosis_l1_l2",
    "spinal_canal_stenosis_l2_l3",
    "spinal_canal_stenosis_l3_l4",
    "spinal_canal_stenosis_l4_l5",
    "spinal_canal_stenosis_l5_s1",
]

def spinal_canal_class_frequency(df):
    freq = {}
    for lvl in levels:
        freq[lvl] = df[lvl].value_counts()
    class_freq = pd.DataFrame(freq).fillna(0).astype(int)
    class_freq.columns = class_freq.columns.str.replace(
        "spinal_canal_stenosis_", "", regex=False
    )
    return class_freq

# Usage
class_freq = spinal_canal_class_frequency(sample_data)
print(class_freq)


def load_data_into_memory(
    df,
    base_img_path,
    sag_model,
    device="cuda",
    final_size=384,        # Final output size (Reference uses 384)
    sag_offsets=(-1, 0, 1),
    num_axial_slices=3
):
    processed = []
    levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    label_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    sag_model.eval().to(device)
    
    study_ids = df["study_id"].unique()

    for study_id in tqdm(study_ids, desc="Loading & Preprocessing"):
        try:
            # ------------------ 1. Setup ------------------ #
            row = df[df.study_id == study_id].iloc[0]
            labels = torch.tensor(
                [label_map[row[f"spinal_canal_stenosis_{lvl.lower().replace('/', '_')}"]] for lvl in levels],
                dtype=torch.long
            )

            # --------- CHANGED PART: series lookup from df --------- #
            try:
                sag_sid = df.loc[
                    (df.study_id == study_id) &
                    (df.series_description == "Sagittal T2/STIR"),
                    "series_id"
                ].iloc[0]

                ax_sid = df.loc[
                    (df.study_id == study_id) &
                    (df.series_description == "Axial T2"),
                    "series_id"
                ].iloc[0]
            except IndexError:
                continue
            # ------------------------------------------------------- #

            sag_path = os.path.join(base_img_path, str(study_id), str(sag_sid))
            ax_path  = os.path.join(base_img_path, str(study_id), str(ax_sid))

            sag_files = collect_dicom_files(sag_path)
            ax_files  = collect_dicom_files(ax_path)

            if not sag_files or not ax_files:
                continue

            # ------------------ 2. Sagittal Processing ------------------ #
            mid_idx = len(sag_files) // 2
            sag_stack = []

            mid_slice_kp = None
            mid_slice_dcm = None

            for off in sag_offsets:
                idx = max(0, min(mid_idx + off, len(sag_files) - 1))
                dcm = pydicom.dcmread(sag_files[idx])
                img = dcm.pixel_array

                t_tensor, meta, orig_shape = preprocess_image(img)

                with torch.no_grad():
                    heatmaps = sag_model(t_tensor.to(device))
                    coords = extract_coordinates(heatmaps, meta, orig_shape)

                kp_dict = {lvl: (float(x), float(y)) for lvl, (x, y) in zip(levels, coords)}

                if off == 0:
                    mid_slice_kp = kp_dict
                    mid_slice_dcm = dcm

                img_norm = img.astype(float)
                img_norm -= img_norm.min()
                if img_norm.max() != 0:
                    img_norm /= img_norm.max()
                img_norm = (img_norm * 255).astype(np.uint8)

                h, w = img_norm.shape
                slice_crops = []

                for lvl in levels:
                    cx, cy = kp_dict[lvl]
                    pad_h = int(0.09 * h)
                    pad_w = int(0.09 * w)

                    ymin, ymax = max(0, int(cy - pad_h)), min(h, int(cy + pad_h))
                    xmin, xmax = max(0, int(cx - pad_w)), min(w, int(cx + pad_w))

                    crop = img_norm[ymin:ymax, xmin:xmax]

                    if crop.size == 0:
                        crop = np.zeros((final_size, final_size), dtype=np.uint8)
                    else:
                        crop = cv2.resize(crop, (final_size, final_size), interpolation=cv2.INTER_LINEAR)

                    slice_crops.append(torch.from_numpy(crop).unsqueeze(0))

                sag_stack.append(torch.stack(slice_crops))

            sagittal_tensor = torch.stack(sag_stack)  # [3, 5, 1, 384, 384]

            # ------------------ 3. Axial Processing ------------------ #
            ax_meta = []
            for f in ax_files:
                d_ax = pydicom.dcmread(f, stop_before_pixels=True)
                ax_meta.append({
                    "path": f,
                    "Z": float(d_ax.ImagePositionPatient[2])
                })

            axial_stack = []

            for lvl in levels:
                cx, cy = mid_slice_kp[lvl]
                z_target = pixel_to_patient_3d(mid_slice_dcm, cx, cy)[2]

                ax_meta.sort(key=lambda a: abs(a["Z"] - z_target))
                selected_axials = ax_meta[:num_axial_slices]

                lvl_axials = []
                for meta in selected_axials:
                    d_ax = pydicom.dcmread(meta["path"])
                    img = d_ax.pixel_array

                    img_norm = img.astype(float)
                    img_norm -= img_norm.min()
                    if img_norm.max() != 0:
                        img_norm /= img_norm.max()
                    img_norm = (img_norm * 255).astype(np.uint8)

                    h_ax, w_ax = img_norm.shape
                    crop_size = 160
                    cy_ax, cx_ax = h_ax // 2, w_ax // 2

                    ymin, ymax = max(0, cy_ax - crop_size//2), min(h_ax, cy_ax + crop_size//2)
                    xmin, xmax = max(0, cx_ax - crop_size//2), min(w_ax, cx_ax + crop_size//2)

                    crop = img_norm[ymin:ymax, xmin:xmax]

                    if crop.size == 0:
                        crop = np.zeros((final_size, final_size), dtype=np.uint8)
                    else:
                        crop = cv2.resize(crop, (final_size, final_size), interpolation=cv2.INTER_LINEAR)

                    lvl_axials.append(torch.from_numpy(crop).unsqueeze(0))

                axial_stack.append(torch.stack(lvl_axials))

            axial_tensor = torch.stack(axial_stack)  # [5, 3, 1, 384, 384]

            # ------------------ 4. Store ------------------ #
            processed.append({
                "sagittal": sagittal_tensor.to(torch.uint8),
                "axial": axial_tensor.to(torch.uint8),
                "label": labels
            })

            if len(processed) % 50 == 0:
                gc.collect()

        except Exception:
            continue

    return processed


class SpinalStenosisDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Inputs are uint8 [Batch, Level, Channel, H, W]
        sag = sample["sagittal"] 
        ax = sample["axial"]    
        labels = sample["labels"] if "labels" in sample else sample["label"]

        # --- Apply Augmentations ---
        if self.transform:
            # 1. Sagittal Processing
            # Reshape to [N, H, W] for looping
            B, L, C, H, W = sag.shape
            sag_flat = sag.reshape(-1, H, W).numpy() 
            
            sag_aug_list = []
            for i in range(sag_flat.shape[0]):
                # Add channel dimension for Albumentations: [H, W] -> [H, W, 1]
                img = sag_flat[i][:, :, None] 
                
                # Apply Transform -> Returns Tensor [C, H, W]
                res = self.transform(image=img)["image"] 
                sag_aug_list.append(res)
            
            # Stack: List of Tensors -> Tensor
            sag = torch.stack(sag_aug_list).reshape(B, L, 1, 384, 384)

            # 2. Axial Processing
            L, B, C, H, W = ax.shape
            ax_flat = ax.reshape(-1, H, W).numpy()
            
            ax_aug_list = []
            for i in range(ax_flat.shape[0]):
                img = ax_flat[i][:, :, None]
                
                # Apply Transform -> Returns Tensor [C, H, W]
                res = self.transform(image=img)["image"]
                ax_aug_list.append(res)
            
            # Stack: List of Tensors -> Tensor
            ax = torch.stack(ax_aug_list).reshape(L, B, 1, 384, 384)
            
        else:
            # Fallback (Manual conversion if no transform provided)
            # This path is unlikely used if you always pass transforms_val
            sag = sag.float() / 255.0
            ax = ax.float() / 255.0

        return {
            "sagittal": sag,
            "axial": ax,
            "labels": labels
        }


import torchvision.models as models
import timm

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        """
        query, key_value: [B, 5, F]
        """
        out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + out)

class SpinalStenosisNet(nn.Module):
    def __init__(self, feature_dim=512, num_classes=3, 
                 sag_pretrained_path=None, ax_pretrained_path=None):
        super().__init__()
        
        # --- 1. Sagittal Backbone ---
        print("Creating Sagittal Backbone...")
        self.sag_backbone = timm.create_model(
            'efficientnetv2_rw_t.ra2_in1k', pretrained=True, in_chans=1, num_classes=0
        )
        if sag_pretrained_path:
            self._load_weights(self.sag_backbone, sag_pretrained_path)
            
        for param in self.sag_backbone.parameters(): param.requires_grad = False
            
        # --- 2. Axial Backbone ---
        print("Creating Axial Backbone...")
        self.ax_backbone = timm.create_model(
            'efficientnetv2_rw_t.ra2_in1k', pretrained=True, in_chans=1, num_classes=0
        )
        if ax_pretrained_path:
            self._load_weights(self.ax_backbone, ax_pretrained_path)
        
        for param in self.ax_backbone.parameters(): param.requires_grad = False

        # Projections
        bb_dim = self.sag_backbone.num_features # 1024
        self.sag_proj = nn.Linear(bb_dim, feature_dim)
        self.ax_proj = nn.Linear(bb_dim, feature_dim)

        # RNN & Attention
        self.level_gru = nn.GRU(feature_dim, feature_dim, batch_first=True, bidirectional=True)
        self.ax_to_sag = CrossAttentionBlock(feature_dim * 2)
        self.sag_to_ax = CrossAttentionBlock(feature_dim * 2)

        # Heads
        self.level_heads = nn.ModuleDict({
            lvl: nn.Sequential(
                nn.Linear(feature_dim * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            ) for lvl in ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        })

    def _load_weights(self, model, path):
        try:
            state_dict = torch.load(path, map_location='cpu')
            # Strip classifier keys
            state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
            model.load_state_dict(state_dict, strict=False)
            print(f"âœ… Loaded weights from {path}")
        except Exception as e:
            print(f"âš ï¸� Failed load {path}: {e}")

    def forward(self, batch):
        sag = batch["sagittal"] # [B, 3, 5, 1, 384, 384]
        ax = batch["axial"]     # [B, 5, 3, 1, 384, 384]
        B = sag.shape[0]

        # --- Sagittal Features ---
        sag_feats = []
        for s in range(3):
            # Flatten: [B*5, 1, 384, 384]
            x = sag[:, s].reshape(B * 5, 1, 384, 384)
            f = self.sag_backbone(x)
            f = self.sag_proj(f).reshape(B, 5, -1)
            sag_feats.append(f)
        # Average over 3 slices
        sag_feats = torch.stack(sag_feats).mean(dim=0) # [B, 5, F]
        sag_ctx, _ = self.level_gru(sag_feats)         # [B, 5, 2F]

        # --- Axial Features ---
        ax_feats = []
        for l in range(5):
            # Flatten: [B*3, 1, 384, 384]
            x = ax[:, l].reshape(B * 3, 1, 384, 384)
            f = self.ax_backbone(x)
            f = self.ax_proj(f).reshape(B, 3, -1).mean(dim=1) # Average over 3 slices
            ax_feats.append(f)
        
        ax_feats = torch.stack(ax_feats, dim=1) # [B, 5, F]
        ax_ctx, _ = self.level_gru(ax_feats)    # [B, 5, 2F]

        # --- Cross Attention ---
        sag_att = self.ax_to_sag(sag_ctx, ax_ctx)
        ax_att = self.sag_to_ax(ax_ctx, sag_ctx)

        # --- Prediction ---
        outputs = {}
        for i, lvl in enumerate(["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]):
            fused = torch.cat([sag_att[:, i], ax_att[:, i]], dim=1)
            outputs[lvl] = self.level_heads[lvl](fused)

        return outputs


import torch.optim as optim
import copy

# Configuration
IMG_PATH = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define Transforms with ToTensorV2
transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.9),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.6),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=1.0),
    A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=0.9),    
    
    # Normalization (Converts uint8 [0,255] to float [-1,1])
    A.Normalize(mean=0.5, std=0.5),
    
    # Conversion to Tensor (HWC -> CHW)
    ToTensorV2()
])

transforms_val = A.Compose([
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])


from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df, test_size = 0.2, random_state = 11)

print("Preprocessing Train Data...")
train_data_list = load_data_into_memory(df_train, IMG_PATH, sagittal_model, DEVICE)

print("Preprocessing Val Data...")
val_data_list = load_data_into_memory(df_val, IMG_PATH, sagittal_model, DEVICE)


# Initialize Datasets
train_ds1 = SpinalStenosisDataset(train_data_list, transform = transforms_train)
train_ds2 = SpinalStenosisDataset(train_data_list, transform = transforms_val)

train_ds = ConcatDataset([train_ds1, train_ds2])
val_ds = SpinalStenosisDataset(val_data_list, transform = transforms_val)

# Initialize Dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)


import matplotlib.pyplot as plt
import numpy as np
import torch

# 1. Get one batch
# The loader returns a dictionary
batch = next(iter(train_loader))

# 2. Select the first patient in the batch
patient_idx = 0

# Shapes Reminder:
# Sagittal: [Batch, Stack(3), Level(5), Ch(1), H, W]
# Axial:    [Batch, Level(5), Stack(3), Ch(1), H, W]
# Labels:   [Batch, Level(5)]

sag_imgs = batch['sagittal'][patient_idx] # [3, 5, 1, 384, 384]
ax_imgs  = batch['axial'][patient_idx]    # [5, 3, 1, 384, 384]
labels   = batch['labels'][patient_idx]   # [5]

# 3. Setup Visualization
levels_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
severity_map = {0: "Normal/Mild", 1: "Moderate", 2: "Severe"}

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# Function to un-normalize [-1, 1] -> [0, 1]
def unnorm(img_tensor):
    img = img_tensor.cpu().numpy()
    img = img * 0.5 + 0.5
    return np.clip(img, 0, 1)

# 4. Loop through the 5 Levels
for i in range(5):
    lbl_text = severity_map[labels[i].item()]
    
    # --- Row 1: Sagittal (Middle Slice) ---
    # Shape: [Stack, Level, Ch, H, W] -> Get Stack=1 (Middle)
    sag_img = sag_imgs[1, i, 0] 
    axes[0, i].imshow(unnorm(sag_img), cmap='gray')
    axes[0, i].set_title(f"Sagittal {levels_names[i]}\n{lbl_text}", fontsize=10)
    axes[0, i].axis('off')
    
    # --- Row 2: Axial (Middle Slice) ---
    # Shape: [Level, Stack, Ch, H, W] -> Get Stack=1 (Middle)
    ax_img = ax_imgs[i, 1, 0]
    axes[1, i].imshow(unnorm(ax_img), cmap='gray')
    axes[1, i].set_title(f"Axial {levels_names[i]}", fontsize=10)
    axes[1, i].axis('off')

plt.suptitle(f"Training Sample (Patient {patient_idx}) - Middle Slices Only", fontsize=16)
plt.show()





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from transformers import get_cosine_schedule_with_warmup
import os

# -------------------------------------------------------------------
# CONFIGURATION (Matching Reference)
# -------------------------------------------------------------------
EPOCHS = 40
LR = 2e-4
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

# Class Weights [1.0, 2.0, 4.0] for Normal, Moderate, Severe
CE_CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32).to(DEVICE)

# -------------------------------------------------------------------
# MODEL & OPTIMIZER SETUP
# -------------------------------------------------------------------
# Initialize model (Ensure you provide paths if you have them, otherwise None)
# Note: You should ideally have the 'Sagittal_PreTrain_EffNetV2.pth' from previous steps
model = SpinalStenosisNet(
    feature_dim=512,
    num_classes=3,
    sag_pretrained_path="/kaggle/input/lumbar-spine-keypoint-detection-models/Sagittal_PreTrain_EffNetV2.pth", 
    ax_pretrained_path="/kaggle/input/lumbar-spine-keypoint-detection-models/Axial_PreTrain_EffNetV2.pth" 
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=CE_CLASS_WEIGHTS)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scaler = torch.cuda.amp.GradScaler()

# Scheduler: Cosine with Warmup (Calculated per batch)
num_training_steps = EPOCHS * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps) # 10% Warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def move_batch_to_device(batch, device):
    return {
        "sagittal": batch["sagittal"].to(device),
        "axial": batch["axial"].to(device),
        "labels": batch["labels"].to(device)
    }

def ce_predict(logits):
    return torch.argmax(logits, dim=1)

def validate(model, loader, device):
    model.eval()

    val_stats = {
        lvl: {"correct": 0, "total": 0, "y_true": [], "y_pred": []}
        for lvl in LEVELS
    }

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            labels = batch["labels"]

            for i, lvl in enumerate(LEVELS):
                preds = ce_predict(logits[lvl])
                gt = labels[:, i]

                val_stats[lvl]["correct"] += (preds == gt).sum().item()
                val_stats[lvl]["total"] += gt.size(0)
                val_stats[lvl]["y_true"].append(gt.cpu().numpy())
                val_stats[lvl]["y_pred"].append(preds.cpu().numpy())

    overall_f1 = []

    for lvl in LEVELS:
        y_true = np.concatenate(val_stats[lvl]["y_true"])
        y_pred = np.concatenate(val_stats[lvl]["y_pred"])
        val_stats[lvl]["f1_macro"] = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        overall_f1.append(val_stats[lvl]["f1_macro"])

    overall_acc = sum(val_stats[lvl]["correct"] for lvl in LEVELS) / \
                  sum(val_stats[lvl]["total"] for lvl in LEVELS)

    return overall_acc, float(np.mean(overall_f1)), val_stats


# -------------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------------
train_acc_steps = []
val_metrics_history = [] # Stores (step, acc, f1)
global_step = 0

best_val_f1 = 0.0
best_model_path = "SpinalNet_Best_F1.pth"

print(f"Starting Training: {EPOCHS} Epochs, LR={LR}, Warmup={num_warmup_steps} steps")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in pbar:
        batch = move_batch_to_device(batch, DEVICE)
        labels = batch["labels"]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            logits = model(batch)
            # Sum loss over all 5 levels
            loss = sum(
                criterion(logits[lvl], labels[:, i])
                for i, lvl in enumerate(LEVELS)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # Update LR every batch

        epoch_loss += loss.item()

        # Calculate Batch Accuracy for monitoring
        correct, total = 0, 0
        with torch.no_grad():
            for i, lvl in enumerate(LEVELS):
                preds = ce_predict(logits[lvl])
                correct += (preds == labels[:, i]).sum().item()
                total += labels.size(0)

        acc = correct / total
        train_acc_steps.append(acc)
        global_step += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{acc:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    # ----------------- END OF EPOCH ----------------- #
    avg_train_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f}")

    # Validation
    val_acc, val_f1, lvl_stats = validate(model, val_loader, DEVICE)
    val_metrics_history.append((global_step, val_acc, val_f1))

    print(f"Validation Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f}")
    
    # Save Best Model
    if val_f1 > best_val_f1:
        print(f"ğŸ”¥ New Best F1! ({best_val_f1:.4f} -> {val_f1:.4f}). Saving model...")
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
    
    # Print per-level stats
    print("-" * 40)
    for lvl in LEVELS:
        print(f"{lvl}: Acc={lvl_stats[lvl]['correct']/lvl_stats[lvl]['total']:.3f}, "
              f"F1={lvl_stats[lvl]['f1_macro']:.3f}")
    print("-" * 40)

# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------
steps, v_accs, v_f1s = zip(*val_metrics_history)
train_steps = range(len(train_acc_steps))

plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_steps, train_acc_steps, label="Train Acc", alpha=0.3, color='blue')
plt.plot(steps, v_accs, "o-", label="Val Acc", color='orange', linewidth=2)
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

# Plot F1
plt.subplot(1, 2, 2)
plt.plot(steps, v_f1s, "o-", label="Val F1", color='green', linewidth=2)
plt.xlabel("Steps")
plt.ylabel("Macro F1 Score")
plt.title("Validation F1 Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Training Complete. Best F1: {best_val_f1:.4f}")


torch.cuda.empty_cache()


import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from tqdm import tqdm

LABEL_MAP = {'Normal/Mild' : 0, 'Moderate' : 1, 'Severe' : 2}
TRAIN_IMG = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
COLUMNS = ['study_id', 'spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1']


df = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")[COLUMNS].replace(LABEL_MAP)
coords_df = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv")
train_desc = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv")


coords_with_desc = coords_df.merge(train_desc, on=['study_id', 'series_id'], how='left')
coords = coords_with_desc[coords_with_desc['series_description'] == 'Axial T2']
final = coords.merge(df, on=['study_id'], how='inner')


def load_dicom(path):
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
        
        # 1. Shift to 0
        data = data - np.min(data)
        
        # 2. Scale to 0-1 using the Max value
        if np.max(data) != 0:
            data = data / np.max(data)
        
        # 3. Convert to uint8 (0-255)
        # Albumentations expects this format for the best compatibility
        data = (data * 255).astype(np.uint8)
        return data
    except Exception as e:
        return np.zeros((256, 256)).astype(np.uint8)


def load_axial_data(df, base_img_path):
    processed_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Axial Samples"):
        try:
            if row["series_description"] != "Axial T2":
                continue
            image_path = os.path.join(
                base_img_path,
                str(row["study_id"]),
                str(row["series_id"]),
                f"{row['instance_number']}.dcm"
            )

            if not os.path.exists(image_path):
                continue
                
            image = load_dicom(image_path)

            # This below logic for crop has been taken directly from MSCAN paper
            x, y = row["x"], row["y"]
            h, w = image.shape

            pad_h = 0.09 * h
            pad_w = 0.09 * w
            
            ymin = int(y - pad_h)
            ymax = int(y + pad_h)
            xmin = int(x - pad_w)
            xmax = int(x + pad_w)

            # Clamp boundaries to be within the image size
            ymin = max(0, ymin)
            ymax = min(h, ymax)
            xmin = max(0, xmin)
            xmax = min(w, xmax)

            crop = image[ymin:ymax, xmin:xmax]

            # If coordinates were way off and crop is empty, skip
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            # Constructs column name like "spinal_canal_stenosis_l4_l5"
            col = "spinal_canal_stenosis_" + row["level"].replace("/", "_").lower()
            label = int(row[col])

            # Store
            processed_samples.append({
                "image": crop,  
                "label": label,
            })

        except Exception as e:
            print(f"[Axial Loader Error] {e}") 
            continue
            
    return processed_samples


def load_sagittal_data(df, base_img_path):
    processed_samples = []

    # Iterate through the dataframe
    # We use tqdm to show a progress bar because loading thousands of DICOMs takes time
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Sagittal Samples"):
        try:
            # 1. Filter: Ensure we are only looking at Sagittal T2/STIR images
            if row["series_description"] != "Sagittal T2/STIR":
                continue

            # 2. Path Construction
            image_path = os.path.join(
                base_img_path,
                str(row["study_id"]),
                str(row["series_id"]),
                f"{row['instance_number']}.dcm"
            )

            # 3. Load Image
            if not os.path.exists(image_path):
                # Only strictly necessary if your dataset might have missing files
                continue
                
            image = load_dicom(image_path)

            # 4. Crop Logic (MATCHING REFERENCE CODE EXACTLY)
            # Reference: int(y - 0.09 * h) to int(y + 0.09 * h)
            x, y = row["x"], row["y"]
            h, w = image.shape

            # Calculate boundaries
            pad_h = 0.09 * h
            pad_w = 0.09 * w
            
            ymin = int(y - pad_h)
            ymax = int(y + pad_h)
            xmin = int(x - pad_w)
            xmax = int(x + pad_w)

            # Clamp boundaries to be within the image size
            ymin = max(0, ymin)
            ymax = min(h, ymax)
            xmin = max(0, xmin)
            xmax = min(w, xmax)

            # Perform the crop
            crop = image[ymin:ymax, xmin:xmax]

            # 5. Sanity Check
            # If coordinates were way off and crop is empty, skip
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            # 6. Label Extraction
            # Constructs column name like "spinal_canal_stenosis_l4_l5"
            col = "spinal_canal_stenosis_" + row["level"].replace("/", "_").lower()
            label = int(row[col])

            # 7. Store
            processed_samples.append({
                "image": crop,  # We store the crop (numpy array)
                "label": label,
                "study_id": row["study_id"], # Optional: keep metadata for debugging
                "level": row["level"]
            })

        except Exception as e:
            # If a specific image fails, print error but don't stop the whole loop
            # print(f"[Sagittal Loader Error] {e}") 
            continue
            
    return processed_samples


import albumentations as A
import torchvision.transforms as transforms

transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=1.0),

    A.CoarseDropout(
        max_holes=4,  # int | None
        max_height=8,  # ScalarType | None
        max_width=8,  # ScalarType | None
        min_holes=2,  # int | None
        min_height=None,  # ScalarType | None
        min_width=None,  # ScalarType | None
        fill_value=0,  # Union[float, Sequence[float]]
        mask_fill_value=None,  # Union[float, Sequence[float], NoneType]
        num_holes_range=(1, 1),  # tuple[int, int]
        hole_height_range=(8, 8),  # tuple[ScalarType, ScalarType]
        hole_width_range=(8, 8),  # tuple[ScalarType, ScalarType]
        always_apply=None,  # bool | None
        p=0.7,  # float
    ),
    A.Defocus(
    radius=(3, 5),  # ScaleIntType
    alias_blur=(0.1, 0.5),  # ScaleFloatType
    always_apply=None,  # bool | None
    p=0.4,  # float
    ),


    A.Perspective(
    scale=(0.05, 0.1),  # ScaleFloatType
    keep_size=True,  # bool
    pad_mode=0,  # int
    pad_val=0,  # ColorType
    mask_pad_val=0,  # ColorType
    fit_output=False,  # bool
    interpolation=1,  # <class 'int'>
    always_apply=None,  # bool | None
    p=0.2,  # float
    ),
    A.PixelDropout(
    dropout_prob=0.01,  # float
    per_channel=False,  # bool
    drop_value=0,  # ScaleFloatType | None
    mask_drop_value=None,  # ScaleFloatType | None
    always_apply=None,  # bool | None
    p=0.5,  # float
    ),

    A.RandomToneCurve(
    scale=0.1,  # float
    per_channel=False,  # bool
    always_apply=None,  # bool | None
    p=0.4,  # float
    ),
    A.RandomGamma(
    gamma_limit=(80, 120),  # ScaleIntType
    always_apply=None,  # bool | None
    p=0.5,  # float
    ),
    A.ShiftScaleRotate(
    shift_limit=(-0.0625, 0.0625),  # ScaleFloatType
    scale_limit=(-0.1, 0.1),  # ScaleFloatType
    rotate_limit=(-15, 15),  # ScaleFloatType
    interpolation=1,  # <class 'int'>
    border_mode=4,  # int
    value=0,  # ColorType
    mask_value=0,  # ColorType
    shift_limit_x=None,  # ScaleFloatType | None
    shift_limit_y=None,  # ScaleFloatType | None
    rotate_method="largest_box",  # Literal['largest_box', 'ellipse']
    always_apply=None,  # bool | None
    p=1.0,  # float
    ),


    #grey scale 3 channel
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
    A.Normalize(mean=0.5, std=0.5),
    A.Resize(384, 384),

    
])

transforms_val = A.Compose([
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
    A.Normalize(mean=0.5, std=0.5),
    A.Resize(384, 384)
])


class PretrainingDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"]          # numpy array, uint8, H x W
        label = item["label"]
        
        # FIX: Keep as uint8. 
        # Albumentations expects uint8 inputs for proper noise/blur application.
        # A.Normalize will handle the float conversion later.
        
        # Albumentations expects H x W or H x W x C
        if img.ndim == 2:
            img = img[..., None]     # H x W x 1

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]  # numpy array, float32, H x W x C (normalized)

        # Convert to torch tensor, Channel-First (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, torch.tensor(label, dtype=torch.long)


from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

samples = load_axial_data(
    df=final,
    base_img_path=TRAIN_IMG
)


print(f"Total samples loaded: {len(samples)}")

labels = [s["label"] for s in samples]

train_samples, val_samples = train_test_split(
    samples,
    test_size=0.2,    # 10% for validation (matches reference ~0.1 split)
    stratify=labels,
    random_state=42
)

print(f"Training samples (Before concat): {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")

# (Uses transforms_train: Noise, Dropout, Distortions)
train_ds_aug = PretrainingDataset(
    data_list=train_samples,
    transform=transforms_train 
)
# (Uses transforms_val: Only Resize & Normalize)
# This prevents the model from forgetting what a real spine looks like.
train_ds_clean = PretrainingDataset(
    data_list=train_samples,
    transform=transforms_val 
)
# This effectively doubles your epoch size (50% clean, 50% augmented)
train_ds_final = ConcatDataset([train_ds_aug, train_ds_clean])

val_ds = PretrainingDataset(
    data_list=val_samples,
    transform=transforms_val
)


batch_size = 32
train_dl = DataLoader(
    train_ds_final,
    batch_size=batch_size,
    shuffle=True,       # CRITICAL: Mixes clean and dirty images in every batch
    num_workers=4,      # Adjust based on CPU cores
    pin_memory=True,    # Faster transfer to GPU
    drop_last=True      # Drops incomplete batch at the end
)

val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,      # No need to shuffle validation
    num_workers=4,
    pin_memory=True
)

print(f"Final Training DataLoader length: {len(train_dl)} batches")
print(f"Final Validation DataLoader length: {len(val_dl)} batches")


import matplotlib.pyplot as plt
import numpy as np

# Reverse label map for display
INV_SEVERITY_MAP = {
    0: "Normal/Mild",
    1: "Moderate",
    2: "Severe"
}

# 1. Get one batch
images, labels = next(iter(train_dl))

# 2. Move to CPU
images = images.cpu()
labels = labels.cpu()

# 3. Setup Plot for 16 images
# Ensure we don't crash if the batch size is smaller than 16
N = min(16, images.size(0)) 

# Create a 4x4 grid (or smaller if N < 16)
rows = 4
cols = 4
plt.figure(figsize=(16, 16))

for i in range(N):
    plt.subplot(rows, cols, i + 1)
    
    # Extract image: [1, H, W] -> [H, W]
    img = images[i, 0].numpy()
    
    # --- UN-NORMALIZE ---
    # Reverses A.Normalize(mean=0.5, std=0.5) => [-1, 1] to [0, 1]
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    
    plt.imshow(img, cmap="gray")
    plt.title(f"{INV_SEVERITY_MAP[int(labels[i])]}")
    plt.axis("off")

plt.suptitle(f"Sagittal ROI Batch (Showing {N} images)\nLook for the mix of 'Clean' vs 'Augmented'", fontsize=16)
plt.tight_layout()
plt.show()


import torch
import torch.nn as nn
import timm
from torch.optim import AdamW
from tqdm import tqdm
import sys

# --------------------------------------------------
# Model
# --------------------------------------------------
model_name = 'efficientnetv2_rw_t.ra2_in1k'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Creating model: {model_name}")
model = timm.create_model(
    model_name,
    pretrained=True,
    in_chans=1,
    num_classes=3
).to(device)

# --------------------------------------------------
# Loss (Weighted for Class Imbalance)
# --------------------------------------------------
# Reference weights: 1.0 (Normal), 2.0 (Moderate), 4.0 (Severe)
weights = torch.tensor([1.0, 2.0, 4.0], device=device)
criterion = nn.CrossEntropyLoss(weight=weights)

# --------------------------------------------------
# Optimizer
# --------------------------------------------------
# Reference LR: 0.00005
optimizer = AdamW(model.parameters(), lr=0.00005)

# --------------------------------------------------
# Scheduler
# --------------------------------------------------
# Reference: ReduceLROnPlateau, mode='max' (Maximize Accuracy), factor=0.1, patience=2
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # Changed to MAX because we are tracking Accuracy
    factor=0.1,
    patience=2,
    verbose=True
)

# --------------------------------------------------
# Training config
# --------------------------------------------------
epochs = 20  # Reference used 100
best_acc = 0.0 # Reference tracks Best Accuracy, not Loss
early_stopping_patience = 5
early_stopping_counter = early_stopping_patience
save_path = "Axial_PreTrain_EffNetV2.pth"

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
print(f"Starting training on {device}...")

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 30)

    for phase in ["train", "val"]:
        is_train = phase == "train"
        model.train() if is_train else model.eval()

        dataloader = train_dl if is_train else val_dl
        dataset_size = len(dataloader.dataset)

        running_loss = 0.0
        running_corrects = 0
        
        # Determine dataset length for the progress bar
        # (len(dataloader) gives number of batches)
        pbar = tqdm(dataloader, desc=phase.upper(), leave=True)

        with torch.set_grad_enabled(is_train):
            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if is_train:
                    loss.backward()
                    optimizer.step()

                # Statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar with current batch loss
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Epoch Statistics
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # --------------------------------------------------
        # Scheduler & Checkpointing
        # --------------------------------------------------
        
        # Step Scheduler on Validation Accuracy (Maximize)
        if phase == "val":
            scheduler.step(epoch_acc)

            # Save if Accuracy Improves
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                early_stopping_counter = early_stopping_patience
                torch.save(model.state_dict(), save_path)
                print(f"âœ… Best model saved! (Acc: {best_acc:.4f})")
            else:
                early_stopping_counter -= 1
                print(f"â�³ Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
    # Early Stopping Trigger (checked after val phase)
    if early_stopping_counter == 0:
        print("\nğŸ›‘ Early stopping triggered. Training finished.")
        break


torch.cuda.empty_cache()


from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

# 1. Setup
model.eval()
y_pred = []
y_true = []

# 2. Inference Loop
print("Running inference on validation set...")
with torch.no_grad():
    for inputs, labels in tqdm(val_dl, desc="Validating"):
        inputs = inputs.to(device)
        
        # Forward pass
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        
        # Store results
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# 3. Calculate Macro F1
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"\nğŸ”¥ Validation Macro F1 Score: {macro_f1:.4f}")

# 4. Generate & Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
target_names = ['Normal/Mild', 'Moderate', 'Severe']

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', ax=ax, values_format='d')

plt.title(f'Confusion Matrix (Macro F1: {macro_f1:.4f})', fontsize=14)
plt.show()

# 5. Print Per-Class Accuracy (Optional but helpful)
# Normalize CM by row (true label) to see recall per class
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Recall (Accuracy):")
for i, name in enumerate(target_names):
    print(f"{name}: {cm_normalized[i, i]*100:.2f}%")


!pip install -q git+https://github.com/qubvel/segmentation_models.pytorch



import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import segmentation_models_pytorch as smp
from tqdm import tqdm
import gc
import timm
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss, accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import warnings
import math
warnings.filterwarnings('ignore')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENCODER_NAME = 'resnet18'

# Image dimensions for keypoint detection
PATCH_H = 512
PATCH_W = 512

# ROI dimensions (matching SOTA)
ROI_SIZE = 384

# Kaggle paths
BASE_PATH_IMG = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
SAGITTAL_KP_MODEL_PATH = '/kaggle/input/lumbar-spine-keypoint-detection-models/Sagittal_T2_sagittal_level_segmentation_2_v2'
AXIAL_KP_MODEL_PATH = '/kaggle/input/lumbar-spine-keypoint-detection-models/Axial_T2_axial_side_segmentation_1'
SAGITTAL_ENCODER_PATH = "/kaggle/input/lumbar-spine-keypoint-detection-models/Sagittal_PreTrain_EffNetV2.pth"
AXIAL_ENCODER_PATH = "/kaggle/input/lumbar-spine-keypoint-detection-models/Axial_PreTrain_EffNetV2.pth"

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
LABEL_MAP = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

# Training config (matching SOTA)
EPOCHS = 6
BATCH_SIZE = 24  # Small batch like SOTA
GRAD_ACC = 2
TGT_BATCH_SIZE = 32
LR = 2e-4 * TGT_BATCH_SIZE / 32
WD = 1e-2
USE_AMP = True
SEED = 42

print(f"Using device: {DEVICE}")


transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.9),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.6),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=1.0),
    A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=0.9),
    # CLAHE BEFORE normalization (matching pre-training)
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    # CLAHE BEFORE normalization (matching pre-training)
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
    A.Normalize(mean=0.5, std=0.5)
])


class myUNet(nn.Module):
    """UNet model for keypoint detection using heatmaps"""
    def __init__(self, classes):
        super(myUNet, self).__init__()
        self.classes = classes
        self.UNet = smp.Unet(
            encoder_name=ENCODER_NAME,
            classes=classes,
            in_channels=1
        ).to(DEVICE)

    def forward(self, X):
        H, W = X.shape[-2:]
        x = self.UNet(X.view(-1, 1, H, W)).view(-1, H*W)
        min_values = x.min(-1)[0].view(-1, 1)
        max_values = x.max(-1)[0].view(-1, 1)
        d = (max_values - min_values)
        d[d == 0] = 1
        x = (x - min_values) / d
        return x.view(-1, self.classes, H, W)


def pixel_to_patient_3d(dcm, x, y):
    """Converts 2D pixel coordinates to 3D Patient Coordinates"""
    ipp = np.asarray(dcm.ImagePositionPatient, dtype=np.float64)
    iop = np.asarray(dcm.ImageOrientationPatient, dtype=np.float64)
    row_cosines = iop[:3]
    col_cosines = iop[3:]
    row_spacing, col_spacing = map(float, dcm.PixelSpacing)
    return (
        ipp
        + x * col_spacing * col_cosines
        + y * row_spacing * row_cosines
    )

def collect_dicom_files(series_folder):
    """Collects and sorts DICOM files by instance number"""
    patterns = ["*.dcm", "*.DCM", "*.dicom"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(series_folder, p)))
    files.sort(key=lambda f: pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
    return files

torch_resize = torchvision.transforms.Resize((PATCH_H, PATCH_W), antialias=True)

def preprocess_for_keypoint_detection(pixel_array):
    """Preprocesses image for keypoint detection"""
    image = pixel_array.astype(np.float32)
    H, W = image.shape
    
    h_start, w_start = 0, 0
    crop_size = 0
    
    if H > W:
        crop_size = W
        h_start = (H - crop_size) // 2
        image = image[h_start : h_start + crop_size, :]
    elif H < W:
        crop_size = H
        w_start = (W - crop_size) // 2
        image = image[:, w_start : w_start + crop_size]
    else:
        crop_size = H
    
    img_max = np.max(image)
    if img_max > 0:
        image = image / img_max
        
    img_tensor = torch.tensor(image).unsqueeze(0)
    img_tensor = torch_resize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).float().to(DEVICE)
    
    return img_tensor, (h_start, w_start, crop_size), (H, W)

def extract_keypoint_coordinates(heatmaps, crop_info, original_shape):
    """Converts model output heatmaps back to original coordinates"""
    h_start, w_start, crop_size = crop_info
    bs, n_classes, h_map, w_map = heatmaps.shape
    coords = []
    
    for c in range(n_classes):
        hm = heatmaps[0, c, :, :].detach().cpu().numpy()
        y_idx, x_idx = np.unravel_index(np.argmax(hm), hm.shape)
        
        x_crop = x_idx * (crop_size / w_map)
        y_crop = y_idx * (crop_size / h_map)
        
        x_orig = x_crop + w_start
        y_orig = y_crop + h_start
        
        coords.append((x_orig, y_orig))
        
    return coords

def extract_roi_from_image(img_array, cx, cy, pad_factor=0.09, target_size=384):
    """Extracts ROI centered at (cx, cy) - MATCHES PRETRAINING"""
    h, w = img_array.shape
    
    pad_h = int(pad_factor * h)
    pad_w = int(pad_factor * w)
    
    ymin = max(0, int(cy - pad_h))
    ymax = min(h, int(cy + pad_h))
    xmin = max(0, int(cx - pad_w))
    xmax = min(w, int(cx + pad_w))
    
    crop = img_array[ymin:ymax, xmin:xmax]
    
    if crop.size == 0:
        crop = np.zeros((target_size, target_size), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return crop

def normalize_dicom_to_uint8(pixel_array):
    """Normalizes DICOM pixel array to uint8 [0, 255]"""
    img = pixel_array.astype(float)
    img -= img.min()
    if img.max() != 0:
        img /= img.max()
    img = (img * 255).astype(np.uint8)
    return img


print("Loading Sagittal Keypoint Detection Model...")
sagittal_kp_model = torch.load(SAGITTAL_KP_MODEL_PATH, map_location=DEVICE, weights_only=False)
sagittal_kp_model.to(DEVICE)
sagittal_kp_model.eval()

print("Loading Axial Keypoint Detection Model...")
axial_kp_model = torch.load(AXIAL_KP_MODEL_PATH, map_location=DEVICE, weights_only=False)
axial_kp_model.to(DEVICE)
axial_kp_model.eval()

print("âœ… Keypoint detection models loaded")


print("Loading Sagittal Feature Encoder...")
sagittal_encoder = timm.create_model(
    'efficientnetv2_rw_t.ra2_in1k',
    pretrained=False,
    in_chans=1,
    num_classes=0
)
sag_state_dict = torch.load(SAGITTAL_ENCODER_PATH, map_location=DEVICE)
sag_state_dict = {k: v for k, v in sag_state_dict.items() if 'classifier' not in k}
sagittal_encoder.load_state_dict(sag_state_dict, strict=False)
sagittal_encoder.to(DEVICE)
sagittal_encoder.eval()
for param in sagittal_encoder.parameters():
    param.requires_grad = False

print("Loading Axial Feature Encoder...")
axial_encoder = timm.create_model(
    'efficientnetv2_rw_t.ra2_in1k',
    pretrained=False,
    in_chans=1,
    num_classes=0
)
ax_state_dict = torch.load(AXIAL_ENCODER_PATH, map_location=DEVICE)
ax_state_dict = {k: v for k, v in ax_state_dict.items() if 'classifier' not in k}
axial_encoder.load_state_dict(ax_state_dict, strict=False)
axial_encoder.to(DEVICE)
axial_encoder.eval()
for param in axial_encoder.parameters():
    param.requires_grad = False

FEATURE_DIM = sagittal_encoder.num_features  # 1024
print(f"âœ… Feature encoders loaded (dim: {FEATURE_DIM})")


def load_roi_data_into_memory(train_df, train_desc, base_img_path,
                                sag_kp_model, ax_kp_model, device):
    """
    Extracts ROIs and stores them as uint8 arrays in memory.
    Does NOT encode - encoding happens on-the-fly with augmentations!
    """
    processed = []
    study_ids = train_df["study_id"].unique()
    
    for study_id in tqdm(study_ids, desc="Extracting ROIs"):
        try:
            # Get series IDs
            sag_row = train_desc[
                (train_desc.study_id == study_id) &
                (train_desc.series_description == "Sagittal T2/STIR")
            ]
            ax_row = train_desc[
                (train_desc.study_id == study_id) &
                (train_desc.series_description == "Axial T2")
            ]
            
            if sag_row.empty or ax_row.empty:
                continue
            
            sag_sid = sag_row.iloc[0].series_id
            ax_sid = ax_row.iloc[0].series_id
            
            sag_path = os.path.join(base_img_path, str(study_id), str(sag_sid))
            ax_path = os.path.join(base_img_path, str(study_id), str(ax_sid))
            
            sag_files = collect_dicom_files(sag_path)
            ax_files = collect_dicom_files(ax_path)
            
            if not sag_files or not ax_files:
                continue
            
            # --- SAGITTAL PROCESSING ---
            mid_idx = len(sag_files) // 2
            sag_dcm = pydicom.dcmread(sag_files[mid_idx])
            sag_img = sag_dcm.pixel_array
            
            sag_tensor, sag_meta, sag_orig_shape = preprocess_for_keypoint_detection(sag_img)
            with torch.no_grad():
                sag_heatmaps = sag_kp_model(sag_tensor)
                sag_keypoints = extract_keypoint_coordinates(sag_heatmaps, sag_meta, sag_orig_shape)
            
            sag_img_uint8 = normalize_dicom_to_uint8(sag_img)
            
            # Extract 5 sagittal ROIs (keep as uint8)
            sag_rois = []
            for cx, cy in sag_keypoints:
                roi = extract_roi_from_image(sag_img_uint8, cx, cy)
                sag_rois.append(roi)
            
            # --- AXIAL PROCESSING ---
            ax_meta = []
            for f in ax_files:
                d_ax = pydicom.dcmread(f, stop_before_pixels=True)
                ax_meta.append({
                    "path": f,
                    "Z": float(d_ax.ImagePositionPatient[2])
                })
            
            # For each level, get 3 closest axial slices
            axial_rois = []
            
            for level_idx, (cx, cy) in enumerate(sag_keypoints):
                z_target = pixel_to_patient_3d(sag_dcm, cx, cy)[2]
                ax_meta_sorted = sorted(ax_meta, key=lambda a: abs(a["Z"] - z_target))
                selected_axials = ax_meta_sorted[:3]
                
                level_axial_rois = []
                for ax_info in selected_axials:
                    ax_dcm = pydicom.dcmread(ax_info["path"])
                    ax_img = ax_dcm.pixel_array
                    
                    ax_tensor, ax_meta_crop, ax_orig_shape = preprocess_for_keypoint_detection(ax_img)
                    with torch.no_grad():
                        ax_heatmaps = ax_kp_model(ax_tensor)
                        ax_keypoints = extract_keypoint_coordinates(ax_heatmaps, ax_meta_crop, ax_orig_shape)
                    
                    if len(ax_keypoints) > 0:
                        ax_cx, ax_cy = ax_keypoints[0]
                    else:
                        h_ax, w_ax = ax_img.shape
                        ax_cx, ax_cy = w_ax // 2, h_ax // 2
                    
                    ax_img_uint8 = normalize_dicom_to_uint8(ax_img)
                    ax_roi = extract_roi_from_image(ax_img_uint8, ax_cx, ax_cy)
                    level_axial_rois.append(ax_roi)
                
                axial_rois.append(level_axial_rois)
            
            # Get labels
            study_row = train_df[train_df.study_id == study_id].iloc[0]
            labels = np.array([
                LABEL_MAP[study_row[f"spinal_canal_stenosis_{lvl.lower().replace('/', '_')}"]]
                for lvl in LEVELS
            ], dtype=np.int64)
            
            # Store ROIs as uint8 (NOT encoded yet!)
            processed.append({
                'sag_rois': sag_rois,  # List of 5 uint8 [384, 384] arrays
                'ax_rois': axial_rois,  # List of 5 lists, each with 3 uint8 [384, 384] arrays
                'labels': labels
            })
            
            if len(processed) % 50 == 0:
                gc.collect()
        
        except Exception as e:
            continue
    
    return processed


class SpineDataset(Dataset):
    """
    Dataset that stores ROIs as uint8 and returns them.
    Encoding happens in batch in custom collate_fn for GPU efficiency!
    """
    def __init__(self, data_list, transform):
        self.data = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Apply augmentations to all ROIs
        sag_rois_transformed = []
        for roi in sample['sag_rois']:
            transformed = self.transform(image=roi)
            img = transformed['image']  # [H, W] float32
            sag_rois_transformed.append(torch.from_numpy(img).float())
        
        ax_rois_transformed = []
        for level_rois in sample['ax_rois']:
            level_transformed = []
            for roi in level_rois:
                transformed = self.transform(image=roi)
                img = transformed['image']
                level_transformed.append(torch.from_numpy(img).float())
            ax_rois_transformed.append(level_transformed)
        
        return {
            'sag_rois': sag_rois_transformed,  # List of 5 tensors [H, W]
            'ax_rois': ax_rois_transformed,     # List of 5 lists, each with 3 tensors [H, W]
            'labels': torch.from_numpy(sample['labels'])
        }


def collate_and_encode(batch, sag_encoder, ax_encoder, device):
    """
    Custom collate function that:
    1. Batches the augmented ROIs
    2. Encodes them in batches on GPU (much faster!)
    3. Returns batched features
    """
    batch_size = len(batch)
    
    # --- Batch encode sagittal ROIs ---
    # Collect all sagittal ROIs from batch
    all_sag_rois = []
    for sample in batch:
        all_sag_rois.extend(sample['sag_rois'])  # 5 per sample
    
    # Stack and batch encode (5*batch_size ROIs at once!)
    sag_batch = torch.stack(all_sag_rois).unsqueeze(1).to(device)  # [B*5, 1, H, W]
    
    with torch.no_grad():
        sag_features_flat = sag_encoder(sag_batch)  # [B*5, feat_dim]
    
    # Reshape back to [B, 5, feat_dim]
    sag_features = sag_features_flat.reshape(batch_size, 5, -1)
    
    # --- Batch encode axial ROIs ---
    all_ax_rois = []
    for sample in batch:
        for level_rois in sample['ax_rois']:
            all_ax_rois.extend(level_rois)  # 3 per level, 5 levels per sample = 15 per sample
    
    # Stack and batch encode (15*batch_size ROIs at once!)
    ax_batch = torch.stack(all_ax_rois).unsqueeze(1).to(device)  # [B*15, 1, H, W]
    
    with torch.no_grad():
        ax_features_flat = ax_encoder(ax_batch)  # [B*15, feat_dim]
    
    # Reshape back to [B, 5, 3, feat_dim]
    ax_features = ax_features_flat.reshape(batch_size, 5, 3, -1)
    
    # --- Stack labels ---
    labels = torch.stack([sample['labels'] for sample in batch])  # [B, 5]
    
    return {
        'sag_features': sag_features,  # Already on GPU!
        'ax_features': ax_features,     # Already on GPU!
        'labels': labels.to(device)
    }


from itertools import repeat

class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1])
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


class SOTASpineClassifier(nn.Module):
    """
    Exact match to SOTA architecture:
    - 2-layer bidirectional GRU for sagittal
    - 2-layer bidirectional LSTM for axial (per level)
    - 2-layer bidirectional GRU for aggregated axial
    - 2-head cross-attention
    - Takes LAST timestep only
    - Spatial dropout
    - Single FC layer â†’ 15 outputs
    """
    def __init__(self, feature_dim=1024, hidden_dim=512, num_classes=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.seq_len = 5  # 5 levels
        
        # Project encoder features to hidden_dim
        self.sag_proj = nn.Linear(feature_dim, hidden_dim)
        self.ax_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Sagittal Bi-GRU (2 layers, matching SOTA)
        self.sag_gru = nn.GRU(
            hidden_dim, hidden_dim // 2, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True
        )
        
        # Axial LSTM per level (2 layers, matching SOTA)
        self.axial_lstms = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True)
            for _ in range(5)
        ])
        
        # Axial Bi-GRU on aggregated features (2 layers, matching SOTA)
        self.ax_gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Cross-attention (2 heads, matching SOTA)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            batch_first=True
        )
        
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            batch_first=True
        )
        
        # Spatial dropout
        self.spatial_dropout = SpatialDropout(0.1)
        
        # Single FC layer for all 15 outputs (5 levels Ã— 3 classes)
        self.fc = nn.Linear(hidden_dim * 2, 15)
    
    def forward(self, sag_features, ax_features):
        """
        Args:
            sag_features: [B, 5, feature_dim]
            ax_features: [B, 5, 3, feature_dim]
        
        Returns:
            logits: [B, 15] (5 levels Ã— 3 classes, flattened)
        """
        B = sag_features.shape[0]
        
        # --- Sagittal Path ---
        sag_proj = self.sag_proj(sag_features)  # [B, 5, hidden_dim]
        sag_out, _ = self.sag_gru(sag_proj)  # [B, 5, hidden_dim]
        
        # --- Axial Path ---
        # For each level, process 3 slices with LSTM
        axial_level_features = []
        for level_idx in range(5):
            ax_level = ax_features[:, level_idx, :, :]  # [B, 3, feature_dim]
            ax_level_proj = self.ax_proj(ax_level)  # [B, 3, hidden_dim]
            
            # LSTM processes 3 slices, take final hidden state
            lstm_out, _ = self.axial_lstms[level_idx](ax_level_proj)
            # Take last timestep
            h_last = lstm_out[:, -1, :]  # [B, hidden_dim]
            axial_level_features.append(h_last)
        
        # Stack all 5 level features
        ax_aggregated = torch.stack(axial_level_features, dim=1)  # [B, 5, hidden_dim]
        
        # Axial Bi-GRU on aggregated features
        ax_out, _ = self.ax_gru(ax_aggregated)  # [B, 5, hidden_dim]
        
        # --- Cross-Attention ---
        # Axial â†’ Sagittal
        out1, _ = self.cross_attn(ax_out, sag_out, sag_out)  # [B, 5, hidden_dim]
        
        # Sagittal â†’ Axial
        out2, _ = self.cross_attn_2(sag_out, ax_out, ax_out)  # [B, 5, hidden_dim]
        
        # --- Take LAST timestep only (CRITICAL SOTA difference) ---
        out1 = out1[:, -1, :]  # [B, hidden_dim]
        out2 = out2[:, -1, :]  # [B, hidden_dim]
        
        # Spatial dropout
        out1 = self.spatial_dropout(out1)
        out2 = self.spatial_dropout(out2)
        
        # Concatenate
        out = torch.cat([out1, out2], dim=1)  # [B, 2*hidden_dim]
        
        # Single FC layer â†’ 15 outputs
        logits = self.fc(out)  # [B, 15]
        
        return logits


train_df = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")
train_desc = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv")
train_desc = train_desc[train_desc['series_description'] != 'Sagittal T1']

print(f"Total studies: {train_df['study_id'].nunique()}")


print("Extracting ROIs (this will take ~15-20 minutes)...")

processed_data = load_roi_data_into_memory(
    train_df=train_df,
    train_desc=train_desc,
    base_img_path=BASE_PATH_IMG,
    sag_kp_model=sagittal_kp_model,
    ax_kp_model=axial_kp_model,
    device=DEVICE
)

print(f"âœ… Extracted ROIs from {len(processed_data)} studies")


train_data, val_data = train_test_split(
    processed_data,
    test_size=0.2,
    random_state=SEED
)

print(f"Train: {len(train_data)} studies")
print(f"Val: {len(val_data)} studies")


train_ds_aug = SpineDataset(train_data, transforms_train)
train_ds_clean = SpineDataset(train_data, transforms_val)

# ConcatDataset: Mix augmented + clean (matching SOTA)
train_ds = ConcatDataset([train_ds_aug, train_ds_clean])

val_ds = SpineDataset(val_data, transforms_val)

# Create collate function with encoders
from functools import partial
train_collate_fn = partial(collate_and_encode, 
                            sag_encoder=sagittal_encoder, 
                            ax_encoder=axial_encoder, 
                            device=DEVICE)

val_collate_fn = partial(collate_and_encode,
                          sag_encoder=sagittal_encoder,
                          ax_encoder=axial_encoder,
                          device=DEVICE)

train_loader = DataLoader(
    train_ds,
    batch_size=24,
    shuffle=True,
    collate_fn=train_collate_fn,  # Batch encoding happens here!
    num_workers=0,  # Must be 0 for GPU encoding
    pin_memory=False,  # Already on GPU
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=val_collate_fn,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")


model = SOTASpineClassifier(
    feature_dim=FEATURE_DIM,
    hidden_dim=512,
    num_classes=3
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Loss with class weights
CE_WEIGHTS = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=CE_WEIGHTS.to(DEVICE))
criterion_val = nn.CrossEntropyLoss(weight=CE_WEIGHTS)  # CPU version for validation

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WD
)

# Scheduler (matching SOTA)
from transformers import get_cosine_schedule_with_warmup
warmup_steps = int(EPOCHS / 10 * len(train_loader) / GRAD_ACC)
num_total_steps = EPOCHS * len(train_loader) // GRAD_ACC
num_cycles = 0.475  # SOTA uses 0.475

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_total_steps,
    num_cycles=num_cycles
)

# Mixed precision
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

print("âœ… Training setup complete")


best_acc = 0.0
best_wll = 1.2
best_model_path = "SOTA_Exact_Best.pth"

train_losses = []
val_losses = []
val_accuracies = []
val_wlls = []

print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"{'='*60}")
    
    # --- TRAIN ---
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    with tqdm(train_loader, desc="Training") as pbar:
        for idx, batch in enumerate(pbar):
            # Data is already on GPU from collate_fn!
            sag_feat = batch['sag_features']
            ax_feat = batch['ax_features']
            labels = batch['labels']
            
            with autocast:
                logits = model(sag_feat, ax_feat)  # [B, 15]
                
                # Compute loss for each level
                loss = 0
                for col in range(5):  # 5 levels
                    pred = logits[:, col*3:col*3+3]  # Extract 3 classes
                    gt = labels[:, col]
                    loss += criterion(pred, gt) / 5
                
                if GRAD_ACC > 1:
                    loss = loss / GRAD_ACC
            
            if not math.isfinite(loss):
                print(f"Loss is {loss}, stopping training")
                break
            
            total_loss += loss.item() * GRAD_ACC
            
            scaler.scale(loss).backward()
            
            if (idx + 1) % GRAD_ACC == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            pbar.set_postfix({
                'loss': f'{loss.item()*GRAD_ACC:.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.3e}'
            })
    
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # --- VALIDATION ---
    model.eval()
    total_loss = 0
    y_preds = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Data is already on GPU from collate_fn!
            sag_feat = batch['sag_features']
            ax_feat = batch['ax_features']
            labels = batch['labels']
            
            with autocast:
                logits = model(sag_feat, ax_feat)
                
                # Compute loss
                loss = 0
                for col in range(5):
                    pred = logits[:, col*3:col*3+3]
                    gt = labels[:, col]
                    loss += criterion(pred, gt) / 5
                    
                    # Collect predictions for this level
                    y_preds.append(pred.float().cpu())
                    labels_list.append(gt.cpu())
                
                total_loss += loss.item()
    
    val_loss = total_loss / len(val_loader)
    val_losses.append(val_loss)
    
    # Compute WLL and accuracy
    y_preds = torch.cat(y_preds, dim=0)  # [N*5, 3]
    labels_cat = torch.cat(labels_list)  # [N*5]
    
    val_wll = criterion_val(y_preds, labels_cat)
    y_pred_classes = y_preds.argmax(1)
    acc = (y_pred_classes == labels_cat).sum().item() / len(labels_cat)
    
    val_accuracies.append(acc)
    val_wlls.append(val_wll.item())
    
    print(f"\nTrain Loss: {train_loss:.6f}")
    print(f"Val Loss: {val_loss:.6f} | Val WLL: {val_wll:.6f} | Val Acc: {acc:.6f}")
    
    # Save best model
    if acc > best_acc:
        print(f"ğŸ”¥ Best accuracy updated: {best_acc:.6f} â†’ {acc:.6f}")
        best_acc = acc
        torch.save(model.state_dict(), best_model_path)
    
    if val_wll < best_wll:
        print(f"ğŸ”¥ Best WLL updated: {best_wll:.6f} â†’ {val_wll:.6f}")
        best_wll = val_wll.item()

print(f"\nâœ… Training complete!")
print(f"Best Accuracy: {best_acc:.6f}")
print(f"Best WLL: {best_wll:.6f}")


print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

model.load_state_dict(torch.load(best_model_path))
model.eval()

y_preds = []
labels_list = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Final Eval"):
        sag_feat = batch['sag_features'].to(DEVICE)
        ax_feat = batch['ax_features'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        with autocast:
            logits = model(sag_feat, ax_feat)
            
            for col in range(5):
                pred = logits[:, col*3:col*3+3]
                gt = labels[:, col]
                
                y_preds.append(pred.float().cpu())
                labels_list.append(gt.cpu())

y_preds = torch.cat(y_preds, dim=0)
labels_cat = torch.cat(labels_list)

val_wll = criterion_val(y_preds, labels_cat)
y_pred_classes = y_preds.argmax(1)
acc = (y_pred_classes == labels_cat).sum().item() / len(labels_cat)

print(f"\nFinal Results:")
print(f"  Accuracy: {acc:.6f} ({acc*100:.2f}%)")
print(f"  WLL: {val_wll:.6f}")

# SOTA Comparison
print(f"\nSOTA Comparison:")
print(f"  SOTA Accuracy: 0.938 (93.8%)")
print(f"  Our Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"  Difference:    {(acc - 0.938)*100:+.2f}%")
print(f"\n  SOTA WLL: 0.318")
print(f"  Our WLL:  {val_wll:.4f}")
print(f"  Difference: {(0.318 - val_wll.item()):+.4f}")

if acc > 0.938:
    print("\nğŸ”¥ğŸ”¥ğŸ”¥ CONGRATULATIONS! YOU BEAT SOTA ACCURACY! ğŸ”¥ğŸ”¥ğŸ”¥")
if val_wll < 0.318:
    print("ğŸ”¥ğŸ”¥ğŸ”¥ CONGRATULATIONS! YOU BEAT SOTA WLL! ğŸ”¥ğŸ”¥ğŸ”¥")




