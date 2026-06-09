import os
import time
import random
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm # For Kaggle notebooks

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast

from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


CONFIG = {
    "seed": 42,
    "competition_data_path": "/kaggle/input/el-hackathon-2025",
    "h5_file_name": "elucidata_ai_challenge_data.h5",
    "output_dir": "/kaggle/working/",

    # --- Model Configuration ---
    "model_architecture_name": "resnet18_custom", # Specific to this setup
    "num_classes": 35,
    "custom_resnet18_checkpoint_path": "/kaggle/input/ckpt-file/tenpercent_resnet18.ckpt", # From original notebook

    # --- Training Hyperparameters ---
    "batch_size": 32,
    "num_workers": min(os.cpu_count(), 4), # Kaggle typically has 2 or 4 effective cores for DL
    "learning_rate": 1e-4,  # Lowered from 0.003, common for fine-tuning
    "weight_decay": 1e-4,   # Lowered from 1e-1, more standard value
    "max_epochs": 15,       # Increased from 5, allow more fine-tuning
    "early_stopping_patience": 5, # Number of epochs to wait for improvement
    "min_delta_spearman": 0.001, # Min change in val_spearman for improvement

    # Scheduler: ReduceLROnPlateau settings
    "scheduler_mode": "max",        # Monitor val_spearman for maximization
    "scheduler_factor": 0.2,        # Factor to reduce LR by
    "scheduler_patience": 3,        # Patience for scheduler
    "scheduler_min_lr": 1e-7,

    # --- Data Augmentation & Preprocessing ---
    "image_resize_to": (160, 160), # Adjusted slightly, divisible by common factors
    "patch_size": 54,
    "apply_imagenet_normalization": False, # Set to True if ResNet18 custom ckpt was based on ImageNet norm

    # --- Mixed Precision & W&B ---
    "use_mixed_precision": True, # Enable for speed and memory benefits on compatible GPUs
    "use_wandb": False,
    "wandb_project_name": "elucidata-hackathon-resnet18",
    "wandb_run_name": "resnet18_custom_improved_run",

    # --- Data Splits ---
    "train_slides": ["S_1", "S_2", "S_3", "S_4", "S_5"],
    "val_slides": ["S_6"],
    "test_slide_id": "S_7",

    # --- Submission ---
    # Path to sample_submission.csv (if available and used for IDs)
    # If None, sequential IDs will be generated.
    "sample_submission_path": "/kaggle/input/el-hackathon-2025/sample_submission.csv", # Assumes it's with main comp data
    "submission_id_prefix": "spot_",
}

# Derived paths
CONFIG["h5_full_path"] = os.path.join(CONFIG["competition_data_path"], CONFIG["h5_file_name"])
CONFIG["best_model_save_path"] = os.path.join(CONFIG["output_dir"], f"best_model_{CONFIG['model_architecture_name']}.pt")
os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("--- Configuration Loaded ---")
for k, v in CONFIG.items(): print(f"{k}: {v}")
print("-----------------------------")



def set_seed(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU
    # imgaug.seed(seed_value) # If using imgaug
    # More deterministic, but can be slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(CONFIG["seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda': print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")


def calculate_spearman_for_sample(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]): return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0

def calculate_mean_spearman(targets_batch: np.ndarray, preds_batch: np.ndarray) -> float:
    correlations = [calculate_spearman_for_sample(targets_batch[i], preds_batch[i]) for i in range(len(targets_batch))]
    return np.mean(correlations) if correlations else 0.0 # Handle empty list


class HistopathologyDataset(Dataset):
    def __init__(self, h5_file_path: str, slide_ids_list: list,
                 patch_sz: int, num_classes_expected: int,
                 transform_pipeline: T.Compose = None, mode: str = "train"):
        self.h5_path = h5_file_path
        self.slide_ids = slide_ids_list
        self.patch_size = patch_sz
        self.num_classes = num_classes_expected
        self.transform = transform_pipeline
        self.mode = mode.lower()
        self.data_materials = []

        self._prepare_data()

    def _prepare_data(self):
        print(f"Loading {self.mode} data for slides: {self.slide_ids}...")
        with h5py.File(self.h5_path, "r") as hf:
            img_group_key = "images/Train" if self.mode != "test" else "images/Test"
            spot_group_key = "spots/Train" if self.mode != "test" else "spots/Test"
            
            h5_images = hf[img_group_key]
            h5_spots = hf[spot_group_key]

            for slide_id in tqdm(self.slide_ids, desc=f"Processing {self.mode} slides"):
                if slide_id not in h5_images or slide_id not in h5_spots:
                    print(f"Warning: Slide {slide_id} not found for mode '{self.mode}'. Skipping.")
                    continue

                image_np_arr = np.array(h5_images[slide_id])
                spots_structured_array = np.array(h5_spots[slide_id])
                
                img_h, img_w, _ = image_np_arr.shape
                half_patch = self.patch_size // 2

                for spot_record in spots_structured_array:
                    center_x, center_y = int(spot_record["x"]), int(spot_record["y"])
                    y_s, y_e = max(0, center_y-half_patch), min(img_h, center_y+half_patch+(self.patch_size%2))
                    x_s, x_e = max(0, center_x-half_patch), min(img_w, center_x+half_patch+(self.patch_size%2))
                    extracted_patch_np = image_np_arr[y_s:y_e, x_s:x_e, :]
                    patch_to_store = np.zeros((self.patch_size,self.patch_size,image_np_arr.shape[2]),dtype=image_np_arr.dtype)
                    if extracted_patch_np.shape[0]!=self.patch_size or extracted_patch_np.shape[1]!=self.patch_size:
                        py_s,px_s=max(0,half_patch-(center_y-y_s)),max(0,half_patch-(center_x-x_s))
                        patch_to_store[py_s:py_s+extracted_patch_np.shape[0],px_s:px_s+extracted_patch_np.shape[1],:]=extracted_patch_np
                    else: patch_to_store = extracted_patch_np
                    
                    if self.mode in ["train", "val"]:
                        try:
                            cell_abundances = np.array([spot_record[f"C{i+1}"] for i in range(self.num_classes)], dtype=np.float32)
                            self.data_materials.append((patch_to_store, cell_abundances))
                        except ValueError:
                             print(f"Warning: C1-C{self.num_classes} fields not in spot record for slide {slide_id} (mode: {self.mode}). Spot: x={center_x},y={center_y}. Skipping.")
                             continue
                    elif self.mode == "test":
                        # For test, store 'Test_Set' (public/private flag) or another identifier if needed.
                        # If generating sequential IDs, only patch is strictly needed for model input.
                        self.data_materials.append((patch_to_store, spot_record['Test_Set'])) # Example
        print(f"Initialized {len(self.data_materials)} patches for {self.mode} set.")

    def __len__(self) -> int: return len(self.data_materials)
    def __getitem__(self, idx: int):
        patch_np_array, target_info = self.data_materials[idx]
        patch_pil_image = T.ToPILImage()(patch_np_array)
        image_tensor = self.transform(patch_pil_image) if self.transform else T.ToTensor()(patch_pil_image)
        if self.mode in ["train", "val"]: return image_tensor, torch.tensor(target_info, dtype=torch.float32)
        elif self.mode == "test": return image_tensor, target_info # Return patch and its auxiliary info
        else: raise ValueError(f"Invalid dataset mode: {self.mode}")

def get_data_transforms(cfg_image_resize_to: tuple, cfg_apply_imagenet_norm: bool):
    train_augs = [
        T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(-45, 45)),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)], p=0.8),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ]
    common_final = [T.Resize(cfg_image_resize_to), T.ToTensor()]
    if cfg_apply_imagenet_norm:
        print("Applying ImageNet normalization.")
        common_final.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    train_tfm = T.Compose(train_augs + common_final)
    val_test_common = [T.Resize(cfg_image_resize_to), T.ToTensor()]
    if cfg_apply_imagenet_norm: val_test_common.append(T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]))
    val_test_tfm = T.Compose(val_test_common)
    return train_tfm, val_test_tfm

# --- Initialize Transforms, Datasets, DataLoaders ---
train_data_transforms, val_data_transforms = get_data_transforms(
    CONFIG["image_resize_to"], CONFIG["apply_imagenet_normalization"]
)
train_dataset = HistopathologyDataset(
    CONFIG["h5_full_path"], CONFIG["train_slides"], CONFIG["patch_size"],
    CONFIG["num_classes"], train_data_transforms, "train"
)
val_dataset = HistopathologyDataset(
    CONFIG["h5_full_path"], CONFIG["val_slides"], CONFIG["patch_size"],
    CONFIG["num_classes"], val_data_transforms, "val"
)
train_loader = DataLoader(
    train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False,
    num_workers=CONFIG["num_workers"], pin_memory=True
)
print(f"DataLoaders ready: Train {len(train_loader)} batches, Val {len(val_loader)} batches.")


def _clean_state_dict_keys(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        nk = k.replace('model.', '').replace('resnet.', '').replace('module.', '')
        cleaned[nk] = v
    return cleaned

def create_resnet18_with_custom_checkpoint(num_classes_out: int, ckpt_path: str, dev: torch.device):
    print(f"Creating ResNet18. Attempting to load custom checkpoint: {ckpt_path}")
    model = models.resnet18(weights=None) # No default ImageNet weights

    if not os.path.exists(ckpt_path):
        print(f"WARNING: Custom checkpoint NOT FOUND at {ckpt_path}. Model will have random weights for backbone.")
    else:
        print(f"Loading weights from: {ckpt_path}")
        try:
            ckpt_data = torch.load(ckpt_path, map_location=dev)
            sd_to_load = ckpt_data.get('state_dict', ckpt_data) # Handles common checkpoint structures
            cleaned_sd = _clean_state_dict_keys(sd_to_load)
            
            # Load backbone weights, allowing classifier mismatch initially
            missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
            if missing and not all(k.startswith('fc.') for k in missing): print(f"Missing backbone keys: {[k for k in missing if not k.startswith('fc.')]}")
            if unexpected and not all(k.startswith('fc.') for k in unexpected): print(f"Unexpected backbone keys: {[k for k in unexpected if not k.startswith('fc.')]}")
            print("Custom weights loaded (non-strict).")
        except Exception as e:
            print(f"Error loading custom checkpoint: {e}. Model may have random backbone weights.")

    model.fc = nn.Linear(model.fc.in_features, num_classes_out)
    print(f"ResNet18 classifier adapted for {num_classes_out} classes.")
    return model.to(dev)

# --- Initialize Model ---
model = create_resnet18_with_custom_checkpoint(
    num_classes_out=CONFIG["num_classes"],
    ckpt_path=CONFIG["custom_resnet18_checkpoint_path"],
    dev=DEVICE
)


class DifferentiableSpearmanLoss(nn.Module):
    def __init__(self, regularization_strength: float = 1.0):
        super().__init__(); self.reg_strength = regularization_strength
    def _soft_rank(self, x_in: torch.Tensor) -> torch.Tensor:
        x = x_in.unsqueeze(-1); diffs = x - x.transpose(-1, -2)
        pairwise_comp = torch.sigmoid(-self.reg_strength * diffs)
        return pairwise_comp.sum(dim=-1)
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ypf, ytf = y_pred.float(), y_true.float()
        pr, tr = self._soft_rank(ypf), self._soft_rank(ytf)
        pr_c, tr_c = pr - pr.mean(dim=1, keepdim=True), tr - tr.mean(dim=1, keepdim=True)
        pr_n, tr_n = torch.linalg.norm(pr_c,dim=1,keepdim=True)+1e-8, torch.linalg.norm(tr_c,dim=1,keepdim=True)+1e-8
        spear_val = torch.sum((pr_c/pr_n)*(tr_c/tr_n), dim=1)
        return 1.0 - spear_val.mean()

class CombinedLoss(nn.Module):
    def __init__(self, l1_w: float = 1.0, spear_w: float = 0.5, spear_reg: float = 1.0):
        super().__init__(); self.l1 = nn.L1Loss(); self.spear = DifferentiableSpearmanLoss(spear_reg)
        self.l1_weight, self.spear_weight = l1_w, spear_w
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss1 = self.l1(y_pred, y_true); losss = self.spear(y_pred, y_true)
        if torch.isnan(losss): return self.l1_weight * loss1
        return (self.l1_weight * loss1) + (self.spear_weight * losss)

# --- Initialize Loss, Optimizer, Scheduler ---
criterion = CombinedLoss(l1_w=1.0, spear_w=0.6, spear_reg=1.0).to(DEVICE) # Example weights
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode=CONFIG["scheduler_mode"], factor=CONFIG["scheduler_factor"],
    patience=CONFIG["scheduler_patience"], verbose=True, min_lr=CONFIG["scheduler_min_lr"]
)
scaler = GradScaler() if CONFIG["use_mixed_precision"] and DEVICE.type == "cuda" else None
if scaler: print("Using Mixed Precision Training.")


def train_one_epoch(m: nn.Module, dl: DataLoader, crit: nn.Module, opt: torch.optim.Optimizer,
                    dev: torch.device, sc: GradScaler, ep: int, max_ep: int):
    m.train(); e_loss = 0.0; all_pr_np, all_lb_np = [], []
    pb = tqdm(dl, desc=f"Epoch {ep+1}/{max_ep} [TRAIN]", leave=False)
    for ib, lb in pb:
        ib, lb = ib.to(dev), lb.to(dev); opt.zero_grad()
        if sc:
            with autocast(): ob = m(ib); lv = crit(ob, lb)
            sc.scale(lv).backward(); sc.step(opt); sc.update()
        else: ob = m(ib); lv = crit(ob, lb); lv.backward(); opt.step()
        e_loss += lv.item(); all_pr_np.append(ob.detach().cpu().numpy()); all_lb_np.append(lb.detach().cpu().numpy())
        pb.set_postfix(loss=f"{lv.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
    avg_loss = e_loss/len(dl); preds=np.concatenate(all_pr_np); labels=np.concatenate(all_lb_np)
    spear_score = calculate_mean_spearman(labels, preds)
    return avg_loss, spear_score

@torch.no_grad()
def validate_one_epoch(m: nn.Module, dl: DataLoader, crit: nn.Module, dev: torch.device, ep: int, max_ep: int):
    m.eval(); e_loss = 0.0; all_pr_np, all_lb_np = [], []
    pb = tqdm(dl, desc=f"Epoch {ep+1}/{max_ep} [VALID]", leave=False)
    for ib, lb in pb:
        ib, lb = ib.to(dev), lb.to(dev); ob = m(ib); lv = crit(ob, lb)
        e_loss += lv.item(); all_pr_np.append(ob.cpu().numpy()); all_lb_np.append(lb.cpu().numpy())
        pb.set_postfix(loss=f"{lv.item():.4f}")
    avg_loss = e_loss/len(dl); preds=np.concatenate(all_pr_np); labels=np.concatenate(all_lb_np)
    spear_score = calculate_mean_spearman(labels, preds)
    return avg_loss, spear_score

# --- Main Training Loop ---
best_val_spearman = -float('inf'); epochs_no_improve = 0; history = []
print(f"\n--- Starting Training: {CONFIG['model_architecture_name']} for {CONFIG['max_epochs']} Epochs ---")
t_start = time.time()

for epoch in range(CONFIG["max_epochs"]):
    ep_t_start = time.time()
    train_loss, train_spearman = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler, epoch, CONFIG["max_epochs"])
    val_loss, val_spearman = validate_one_epoch(model, val_loader, criterion, DEVICE, epoch, CONFIG["max_epochs"])
    ep_dur = time.time() - ep_t_start; cur_lr = optimizer.param_groups[0]['lr']
    print(f"E{epoch+1:02d} {ep_dur:5.1f}s | LR {cur_lr:.1e} | Tr L {train_loss:.4f} Sp {train_spearman:.4f} | Vl L {val_loss:.4f} Sp {val_spearman:.4f}")
    history.append({'ep':epoch+1,'lr':cur_lr,'tL':train_loss,'tS':train_spearman,'vL':val_loss,'vS':val_spearman})
    
    scheduler.step(val_spearman)
    if val_spearman > best_val_spearman + CONFIG["min_delta_spearman"]:
        print(f"  Val Sp Imp: {best_val_spearman:.4f} -> {val_spearman:.4f}. Save: {CONFIG['best_model_save_path']}")
        best_val_spearman = val_spearman; epochs_no_improve = 0
        torch.save({'ep':epoch,'sd':model.state_dict(),'opt_sd':optimizer.state_dict(),'sch_sd':scheduler.state_dict(),
                    'best_vS':best_val_spearman,'cfg':CONFIG}, CONFIG['best_model_save_path'])
    else:
        epochs_no_improve +=1; print(f"  No Sp Imp for {epochs_no_improve} ep. Best: {best_val_spearman:.4f}")
    if epochs_no_improve >= CONFIG["early_stopping_patience"]: print(f"Early stop @ ep {epoch+1}. Best: {best_val_spearman:.4f}"); break
    if cur_lr <= CONFIG["scheduler_min_lr"]*(1+1e-4) and epoch > CONFIG["scheduler_patience"]: print(f"LR min @ ep {epoch+1}. Stop."); break
            
total_t_dur = time.time() - t_start
print(f"--- Train End. Time: {total_t_dur//60:.0f}m {total_t_dur%60:.0f}s. Best Val Sp: {best_val_spearman:.4f} ---")
if os.path.exists(CONFIG['best_model_save_path']): print(f"Best model @ {CONFIG['best_model_save_path']}")
else: print("Warning: Best model not saved.")


# --- Load Best Model for Inference ---
if os.path.exists(CONFIG["best_model_save_path"]):
    print(f"\nLoading best model for inference: {CONFIG['best_model_save_path']}")
    # Re-create model instance to load state_dict into clean architecture
    inference_model = create_resnet18_with_custom_checkpoint(
        num_classes_out=CONFIG["num_classes"],
        ckpt_path=CONFIG["custom_resnet18_checkpoint_path"], # Load base custom weights
        dev=DEVICE
    )
    checkpoint = torch.load(CONFIG["best_model_save_path"], map_location=DEVICE)
    inference_model.load_state_dict(checkpoint['sd']) # 'sd' was key for model_state_dict
    print(f"Best model (Ep {checkpoint['ep']}, Val Sp {checkpoint['best_vS']:.4f}) loaded for inference.")
else:
    print("WARNING: No best model checkpoint found. Using model from last training epoch if available, or re-initialize.")
    inference_model = model # Use the model instance as it was at the end of training (might not be the best)
inference_model.eval()





# --- TTA Transforms for Inference ---
def get_tta_inference_transforms(cfg_img_resize: tuple, cfg_apply_norm: bool):
    base_tfm = [T.Resize(cfg_img_resize), T.ToTensor()]
    if cfg_apply_norm: base_tfm.append(T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]))
    tta_vars = [[], [T.RandomHorizontalFlip(p=1.0)], [T.RandomVerticalFlip(p=1.0)],
                [T.RandomRotation(degrees=(90,90))],[T.RandomRotation(degrees=(180,180))],[T.RandomRotation(degrees=(270,270))]]
    return [T.Compose(var_list + base_tfm) for var_list in tta_vars]

tta_pipelines = get_tta_inference_transforms(CONFIG["image_resize_to"], CONFIG["apply_imagenet_normalization"])
print(f"Using {len(tta_pipelines)} TTA variations for inference.")



# --- Prediction on Test Set ---
@torch.no_grad()
def predict_on_test_slide_with_tta(m: nn.Module, h5_path: str, slide_id_test: str, patch_s: int,
                                   tta_list: list, dev: torch.device, use_amp: bool): # use_amp comes from CONFIG
    m.eval(); all_preds_avg = []
    print(f"\n--- TTA Predictions for Test Slide: {slide_id_test} ---")
    # ... (h5py file opening and other initializations) ...
    with h5py.File(h5_path, "r") as hf:
        if not ("images/Test" in hf and slide_id_test in hf["images/Test"] and \
                "spots/Test" in hf and slide_id_test in hf["spots/Test"]):
            raise FileNotFoundError(f"Test data for slide '{slide_id_test}' not in HDF5: {h5_path}")
        img_np, spots_data = np.array(hf["images/Test"][slide_id_test]), np.array(hf["spots/Test"][slide_id_test])
    h, w, _ = img_np.shape; half_p = patch_s // 2

    for spot_rec in tqdm(spots_data, desc=f"Predicting S_7 spots"): # Corrected tqdm description
        cx,cy=int(spot_rec["x"]),int(spot_rec["y"])
        ys,ye=max(0,cy-half_p),min(h,cy+half_p+(patch_s%2)); xs,xe=max(0,cx-half_p),min(w,cx+half_p+(patch_s%2))
        patch_extr = img_np[ys:ye,xs:xe,:]
        patch_pil_in = np.zeros((patch_s,patch_s,img_np.shape[2]),dtype=img_np.dtype)
        if patch_extr.shape[0]!=patch_s or patch_extr.shape[1]!=patch_s:
            pys,pxs=max(0,half_p-(cy-ys)),max(0,half_p-(cx-xs))
            patch_pil_in[pys:pys+patch_extr.shape[0],pxs:pxs+patch_extr.shape[1],:]=patch_extr
        else: patch_pil_in = patch_extr
        pil_patch = T.ToPILImage()(patch_pil_in)
        
        tta_spot_p = []
        for tta_p in tta_list: # tta_p is a T.Compose object
            inp_t = tta_p(pil_patch).unsqueeze(0).to(dev)
            
            
            if use_amp and dev.type == "cuda":
                with autocast():  # `with` statement starts a new block, indented under `if`
                    out_l = m(inp_t) # Indented under `with`
            else:
                out_l = m(inp_t)
            
            tta_spot_p.append(out_l.cpu().numpy())
        all_preds_avg.append(np.mean(np.array(tta_spot_p), axis=0).squeeze())
        
    preds_arr = np.array(all_preds_avg)
    print(f"Generated {len(preds_arr)} preds for {slide_id_test}. Shape: {preds_arr.shape if len(preds_arr)>0 else 'N/A'}")
    return preds_arr

test_predictions = predict_on_test_slide_with_tta(
    inference_model, CONFIG["h5_full_path"], CONFIG["test_slide_id"], CONFIG["patch_size"],
    tta_pipelines, DEVICE, CONFIG["use_mixed_precision"]
)




# --- Submission File Generation ---
def generate_submission_file(preds_arr: np.ndarray, cfg_sample_sub_path: str,
                             out_csv_path: str, num_cls: int, id_pref: str):
    print(f"\n--- Generating Submission File ---")
    if preds_arr is None or len(preds_arr)==0: print("ERROR: No preds for submission."); return
    num_p = len(preds_arr); ids_sub = []
    if cfg_sample_sub_path and os.path.exists(cfg_sample_sub_path):
        print(f"Using sample sub for IDs: {cfg_sample_sub_path}")
        try:
            sdf=pd.read_csv(cfg_sample_sub_path)
            if 'ID' not in sdf.columns: raise ValueError("'ID' col missing.")
            ids_sub=sdf['ID'].tolist()
            if len(ids_sub)!=num_p: print(f"WARN: Preds({num_p}) vs SampleIDs({len(ids_sub)}). Fallback to seq IDs."); ids_sub=[f"{id_pref}{i}" for i in range(num_p)]
        except Exception as e: print(f"ERR reading sample_sub: {e}. Fallback to seq IDs."); ids_sub=[f"{id_pref}{i}" for i in range(num_p)]
    else:
        if cfg_sample_sub_path: print(f"WARN: Sample sub NOT FOUND @ '{cfg_sample_sub_path}'.")
        print("Generating sequential IDs."); ids_sub=[f"{id_pref}{i}" for i in range(num_p)]
    print(f"Generated/loaded {len(ids_sub)} IDs.")
    pred_cols=[f"C{i+1}" for i in range(num_cls)]; preds_df=pd.DataFrame(preds_arr,columns=pred_cols)
    sub_df=pd.DataFrame({'ID':ids_sub}); sub_df=pd.concat([sub_df,preds_df],axis=1)
    exp_cols=['ID']+pred_cols
    if list(sub_df.columns)!=exp_cols: sub_df.columns=exp_cols; print("Renamed cols.")
    try:
        sub_df.to_csv(out_csv_path,index=False); print(f"Submission saved: {out_csv_path}")
        if len(sub_df)>0: print(f"Head:\n{sub_df.head()}")
    except Exception as e: print(f"ERR saving CSV: {e}")

submission_path = os.path.join(CONFIG["output_dir"], "submission.csv")
generate_submission_file(
    test_predictions, CONFIG.get("sample_submission_path"), submission_path,
    CONFIG["num_classes"], CONFIG["submission_id_prefix"]
)
print(f"\n--- ResNet18 Pipeline Finished ---")

