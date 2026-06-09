# ================== ISIC-2024 Kaggle single-cell (robust torch.load retry) ==================
# Paste entire cell into a Kaggle notebook cell and run (do NOT reinstall torch).

import warnings, os, sys, glob, time, math, random
from pathlib import Path
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic.*")
print("Starting robust ISIC pipeline (torch.load retry)...")

import torch
print("torch.__version__:", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        pass

# ---------- AMP compatibility helper ----------
from contextlib import nullcontext
def make_amp_helpers():
    if not torch.cuda.is_available():
        return None, nullcontext
    try:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and hasattr(torch.amp, "autocast"):
            try:
                scaler_try = torch.amp.GradScaler("cuda")
                def autocast_factory():
                    return torch.amp.autocast(device_type="cuda")
                return scaler_try, autocast_factory
            except Exception:
                pass
    except Exception:
        pass
    try:
        scaler = torch.cuda.amp.GradScaler()
        autocast_factory = torch.cuda.amp.autocast
        return scaler, autocast_factory
    except Exception:
        return None, nullcontext

scaler_global, autocast = make_amp_helpers()
print("AMP helpers:", "scaler:", type(scaler_global).__name__ if scaler_global is not None else None, "autocast:", autocast)

# ---------- Config ----------
MAX_SAMPLES = 4000
BATCH_SIZE = 16
IMAGE_SIZE = 384
N_SPLITS_DESIRED = 5
FOLD_TO_RUN = 0
USE_AMP = True
USE_DATAPARALLEL = True
NUM_EPOCHS_DEFAULT = 3
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
SAVE_EVERY_N_EPOCHS = 1
SEED = 42
NUM_WORKERS = 2

def seed_everything(seed=SEED):
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_everything()

# ---------- detect input folder ----------
input_candidates = [p for p in glob.glob('/kaggle/input/*') if os.path.isdir(p)]
if not input_candidates:
    raise RuntimeError("No /kaggle/input dataset found.")
INPUT_ROOT = None
for p in input_candidates:
    if 'isic' in os.path.basename(p).lower():
        INPUT_ROOT = p; break
if INPUT_ROOT is None:
    INPUT_ROOT = input_candidates[0]
print("Using input folder:", INPUT_ROOT)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------- find files ----------
train_csv_candidates = glob.glob(os.path.join(INPUT_ROOT, '*train*metadata*.csv')) + glob.glob(os.path.join(INPUT_ROOT, '*train*.csv')) + glob.glob(os.path.join(INPUT_ROOT, '**/*train*metadata*.csv'), recursive=True)
train_csv = train_csv_candidates[0] if train_csv_candidates else None
train_h5_candidates = glob.glob(os.path.join(INPUT_ROOT, '*train*image*.hdf5')) + glob.glob(os.path.join(INPUT_ROOT, '**/*train*image*.hdf5'), recursive=True)
train_h5 = train_h5_candidates[0] if train_h5_candidates else None
image_files = glob.glob(os.path.join(INPUT_ROOT, '**/*.jpg'), recursive=True) + glob.glob(os.path.join(INPUT_ROOT, '**/*.png'), recursive=True)
has_image_files = len(image_files) > 0
sample_submission_candidates = glob.glob(os.path.join(INPUT_ROOT, '*sample*submission*.csv')) + glob.glob(os.path.join(INPUT_ROOT, '**/*sample*submission*.csv'), recursive=True)
sample_submission = sample_submission_candidates[0] if sample_submission_candidates else None

print("Found:", "train_csv=", train_csv, "train_h5=", train_h5, "plain_images=", len(image_files), "sample_submission=", sample_submission)

# ---------- read metadata ----------
import pandas as pd, numpy as np
if train_csv is None:
    raise RuntimeError("train CSV not found in input folder.")
df = pd.read_csv(train_csv, low_memory=False)
print("metadata shape:", df.shape)
if 'isic_id' not in df.columns and 'image_name' in df.columns:
    df = df.rename(columns={'image_name':'isic_id'})
if 'isic_id' not in df.columns:
    df = df.rename(columns={df.columns[0]:'isic_id'})
if 'target' not in df.columns:
    poss = [c for c in df.columns if c.lower() in ('label','malignant','diagnosis','benign','target')]
    if poss:
        df = df.rename(columns={poss[0]:'target'})
    else:
        df['target'] = 0.0
df['isic_id'] = df['isic_id'].astype(str)
df['target'] = df['target'].astype(float)

# ---------- HDF5 accessor ----------
import h5py, cv2
class H5ImageAccessor:
    def __init__(self, path):
        self.path = path
        self.h5 = h5py.File(path, 'r')
        self.image_key = None; self.id_key = None
        for k in self.h5.keys():
            try:
                obj = self.h5[k]
                if hasattr(obj, 'shape') and len(getattr(obj, 'shape', [])) >= 3:
                    self.image_key = k; break
            except Exception:
                pass
        if self.image_key is None:
            for k in self.h5.keys():
                self.image_key = k; break
        for k in self.h5.keys():
            if 'id' in k.lower() or 'isic' in k.lower() or 'name' in k.lower():
                self.id_key = k; break
        self.id_map = {}
        if self.id_key:
            try:
                raw = list(self.h5[self.id_key])
                conv = []
                for t in raw:
                    if isinstance(t, (bytes, bytearray)): conv.append(t.decode())
                    else: conv.append(str(t))
                for i,v in enumerate(conv): self.id_map[str(v)] = i
            except Exception as e:
                print("Could not build id_map:", e)
        self.is_group = False
        try:
            if hasattr(self.h5[self.image_key], 'keys'):
                self.is_group = True
                self.group_keys = sorted(list(self.h5[self.image_key].keys()))
                self.group_map = {k:i for i,k in enumerate(self.group_keys)}
        except Exception:
            self.is_group = False
        print("H5 chosen image_key:", self.image_key, "id_key:", self.id_key, "id_map_len:", len(self.id_map), "is_group:", self.is_group)
    def get_by_index(self, idx):
        if self.is_group:
            key = self.group_keys[idx]
            obj = self.h5[self.image_key][key]
            arr = obj[()]
        else:
            ds = self.h5[self.image_key]; arr = ds[idx]
        if isinstance(arr, np.ndarray) and arr.dtype != np.dtype('O'):
            if arr.ndim == 3: return arr
            if arr.ndim == 3 and arr.shape[0] in (1,3): return np.transpose(arr,(1,2,0))
            return arr
        try:
            if isinstance(arr, (bytes, bytearray)) or hasattr(arr, 'tobytes'):
                b = arr.tobytes() if not isinstance(arr, (bytes,bytearray)) else arr
                img = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None: raise RuntimeError("cv2.imdecode returned None")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        try:
            arr2 = np.asarray(arr)
            if arr2.ndim == 3: return arr2
        except Exception:
            pass
        raise RuntimeError(f"Could not decode image at index {idx} from {self.image_key}")
    def get_by_id(self, isic_id):
        if isic_id in self.id_map:
            return self.get_by_index(self.id_map[isic_id])
        if self.is_group:
            matches = [k for k in self.group_keys if isic_id in k]
            if matches:
                key = matches[0]
                obj = self.h5[self.image_key][key]
                arr = obj[()]
                if isinstance(arr, np.ndarray) and arr.ndim == 3: return arr
                try:
                    if isinstance(arr, (bytes, bytearray)) or hasattr(arr, 'tobytes'):
                        b = arr.tobytes() if not isinstance(arr,(bytes,bytearray)) else arr
                        img = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None: raise RuntimeError("cv2.imdecode returned None")
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass
        raise KeyError(f"isic_id {isic_id} not found in HDF5 id_map or group keys")
    def close(self):
        try: self.h5.close()
        except: pass

h5_accessor = H5ImageAccessor(train_h5) if train_h5 else None

# ---------- plain image map ----------
img_map = {}
if has_image_files:
    for p in image_files:
        img_map[Path(p).stem] = p
    df['image_path'] = df['isic_id'].map(img_map)
print("Plain image map size:", len(img_map))

# ---------- stratified subsample ----------
from sklearn.model_selection import StratifiedShuffleSplit
if len(df) > MAX_SAMPLES:
    print("Subsampling to", MAX_SAMPLES)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_SAMPLES, random_state=SEED)
    try:
        train_idx, _ = next(sss.split(df, df['target']))
        df = df.iloc[train_idx].reset_index(drop=True)
    except Exception as e:
        print("Stratified failed, fallback random:", e)
        df = df.sample(n=MAX_SAMPLES, random_state=SEED).reset_index(drop=True)
print("Working dataset length:", len(df))

# ---------- set N_SPLITS safely ----------
from collections import Counter
counts = Counter(df['target'].astype(int).tolist())
min_count = min(counts.values()) if counts else 0
N_SPLITS = max(2, min(N_SPLITS_DESIRED, min_count))
if min_count < N_SPLITS_DESIRED:
    print(f"Adjusted N_SPLITS to {N_SPLITS} because smallest class has {min_count} items")

# ---------- transforms (use Affine instead of ShiftScaleRotate) ----------
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
def get_transforms(image_size=IMAGE_SIZE, train=True):
    if train:
        return A.Compose([
            A.SmallestMaxSize(max_size=image_size),
            A.RandomCrop(width=image_size, height=image_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.Affine(translate_percent=0.0625, scale=(0.9,1.1), rotate=(-15,15), p=0.5),
            A.ColorJitter(p=0.4), A.Normalize(), ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(width=image_size, height=image_size),
            A.Normalize(), ToTensorV2(),
        ])

class ISICKaggleDataset(Dataset):
    def __init__(self, df, transforms=None, h5_accessor=None, img_map=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.h5_accessor = h5_accessor
        self.img_map = img_map or {}
    def __len__(self): return len(self.df)
    def _read_img(self, row):
        path = row.get('image_path', None)
        if isinstance(path, str) and os.path.exists(path):
            img = cv2.imread(path)
            if img is None: raise RuntimeError("cv2 failed to read " + path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        iid = str(row['isic_id'])
        if self.h5_accessor is not None:
            try: return self.h5_accessor.get_by_id(iid)
            except Exception:
                try: return self.h5_accessor.get_by_index(int(row.name))
                except Exception: pass
        if iid in self.img_map:
            p = self.img_map[iid]; img = cv2.imread(p)
            if img is None: raise RuntimeError("cv2 failed to read " + p)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for k,p in self.img_map.items():
            if iid in k:
                img = cv2.imread(p)
                if img is None: continue
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise RuntimeError(f"Image for isic_id {iid} not found.")
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._read_img(row)
        if self.transforms: img = self.transforms(image=img)['image']
        else: img = ToTensorV2()(image=img)['image']
        label = torch.tensor(row['target'], dtype=torch.float32)
        return img, label

# ---------- model & training utils ----------
import torch.nn as nn, torch.optim as optim, timm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device); 
if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True

class TimmModel(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s_in21k', pretrained=True, num_classes=1, dropout=0.2):
        super().__init__()
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
    def forward(self, x): return self.net(x).view(-1)

def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    running_loss = 0.0; preds=[]; targets=[]
    loop = tqdm(enumerate(loader), total=len(loader), desc="Train batches", leave=False)
    for step, (imgs, labels) in loop:
        imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                outputs = model(imgs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            outputs = model(imgs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        running_loss += float(loss.item())
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds.extend(probs.tolist()); targets.extend(labels.detach().cpu().numpy().tolist())
        loop.set_postfix(loss=float(running_loss/(step+1)))
    avg_loss = running_loss / max(1, len(loader))
    auc = roc_auc_score(targets, preds) if len(set(targets))>1 else 0.5
    y_pred = (np.array(preds) >= 0.5).astype(int); y_true = np.array(targets).astype(int)
    return avg_loss, auc, dict(acc=accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0, precision=precision_score(y_true, y_pred, zero_division=0), recall=recall_score(y_true, y_pred, zero_division=0), f1=f1_score(y_true, y_pred, zero_division=0))

def valid_one_epoch(model, loader, criterion):
    model.eval(); running_loss = 0.0; preds=[]; targets=[]
    loop = tqdm(enumerate(loader), total=len(loader), desc="Valid batches", leave=False)
    with torch.no_grad():
        for step, (imgs, labels) in loop:
            imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            outputs = model(imgs); loss = criterion(outputs, labels)
            running_loss += float(loss.item())
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(probs.tolist()); targets.extend(labels.cpu().numpy().tolist())
            loop.set_postfix(loss=float(running_loss/(step+1)))
    avg_loss = running_loss / max(1, len(loader))
    auc = roc_auc_score(targets, preds) if len(set(targets))>1 else 0.5
    y_pred = (np.array(preds) >= 0.5).astype(int); y_true = np.array(targets).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return avg_loss, auc, np.array(preds), dict(acc=accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0, precision=precision_score(y_true, y_pred, zero_division=0), recall=recall_score(y_true, y_pred, zero_division=0), f1=f1_score(y_true, y_pred, zero_division=0), cm=cm)

# ---------- helper: robust torch.load with retry ----------
def robust_torch_load(path, map_location=None):
    """
    Try torch.load normally, if it fails with UnpicklingError or related,
    retry with weights_only=False. Returns loaded object (dict).
    """
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"[robust_torch_load] primary torch.load failed for {path}: {repr(e)}")
        # Try retry with weights_only=False (less-restrictive)
        try:
            print(f"[robust_torch_load] retrying with weights_only=False for {path}")
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception as e2:
            print(f"[robust_torch_load] retry with weights_only=False also failed: {repr(e2)}")
            # As a last resort, attempt loading only allowed keys by using dill/pickle fallback not provided here.
            raise

# ---------- training orchestration ----------
from sklearn.model_selection import StratifiedKFold
def run_training_fold(df, model_name='tf_efficientnetv2_s_in21k', image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, lr=2e-4, epochs=NUM_EPOCHS_DEFAULT, fold_idx=FOLD_TO_RUN, n_splits=N_SPLITS, out_dir=CHECKPOINT_DIR, use_amp=USE_AMP, use_dataparallel=USE_DATAPARALLEL):
    os.makedirs(out_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, valid_idx = list(skf.split(df, df['target']))[fold_idx]
    train_df = df.iloc[train_idx].reset_index(drop=True); valid_df = df.iloc[valid_idx].reset_index(drop=True)
    train_ds = ISICKaggleDataset(train_df, transforms=get_transforms(image_size=image_size, train=True), h5_accessor=h5_accessor, img_map=img_map)
    valid_ds = ISICKaggleDataset(valid_df, transforms=get_transforms(image_size=image_size, train=False), h5_accessor=h5_accessor, img_map=img_map)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = TimmModel(model_name, pretrained=True).to(device)
    if use_dataparallel and torch.cuda.device_count() > 1:
        print("DataParallel across", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scaler = scaler_global if (use_amp and torch.cuda.is_available()) else None

    best_auc = 0.0; best_path=None; history=[]
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs} GPU mem allocated MB: {torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0:.1f}")
        t0=time.time()
        train_loss, train_auc, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler=scaler)
        valid_loss, valid_auc, val_preds, val_metrics = valid_one_epoch(model, valid_loader, criterion)
        t1=time.time()
        print(f"Epoch {epoch} summary: train_loss {train_loss:.4f} train_auc {train_auc:.4f} | valid_loss {valid_loss:.4f} valid_auc {valid_auc:.4f}")
        print("Valid metrics:", val_metrics)
        history.append(dict(model=model_name,fold=fold_idx,epoch=epoch,train_loss=train_loss,train_auc=train_auc,valid_loss=valid_loss,valid_auc=valid_auc,epoch_time=t1-t0))

        periodic_path = os.path.join(out_dir, f"{model_name.replace('/','_')}_fold{fold_idx}_epoch{epoch}.pth")
        torch.save({'epoch':epoch,'model_state': (model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()), 'optim_state': optimizer.state_dict(), 'valid_auc': valid_auc}, periodic_path)
        if SAVE_EVERY_N_EPOCHS and epoch % SAVE_EVERY_N_EPOCHS == 0:
            print("Saved periodic checkpoint:", periodic_path)
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_path = os.path.join(out_dir, f"{model_name.replace('/','_')}_fold{fold_idx}_best.pth")
            torch.save({'epoch':epoch,'model_state': (model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()), 'optim_state': optimizer.state_dict(), 'valid_auc': valid_auc}, best_path)
            print("Saved best checkpoint:", best_path)
        if torch.cuda.is_available():
            print(f"GPU mem allocated MB: {torch.cuda.memory_allocated()/1024**2:.1f} reserved MB: {torch.cuda.memory_reserved()/1024**2:.1f}")
    return best_auc, best_path, history

# ---------- run a small model suite ----------
model_suite = [
    ("tf_efficientnetv2_s_in21k", IMAGE_SIZE, 2),
    ("swin_base_patch4_window12_384", IMAGE_SIZE, 2),
]
all_hist=[]; results=[]
for model_name,img_size,epochs in model_suite:
    print("\n=== Training", model_name, "===")
    best_auc, best_ckpt, hist = run_training_fold(df, model_name=model_name, image_size=img_size, batch_size=BATCH_SIZE, lr=2e-4, epochs=epochs, fold_idx=FOLD_TO_RUN, n_splits=N_SPLITS, out_dir=CHECKPOINT_DIR, use_amp=USE_AMP, use_dataparallel=USE_DATAPARALLEL)
    results.append((model_name,best_auc,best_ckpt)); all_hist.extend(hist)
    print("Done", model_name, "best_auc", best_auc, "ckpt", best_ckpt)

if all_hist:
    pd.DataFrame(all_hist).to_csv('/kaggle/working/training_history_robust.csv', index=False)
    print("Saved history to /kaggle/working/training_history_robust.csv")

# ---------- ensemble inference with robust loading + safe AUC ----------
from glob import glob
ckpts = glob(os.path.join(CHECKPOINT_DIR, '*_fold0_best.pth'))
print("Found best ckpts:", ckpts)

def safe_roc_auc(y_true, y_score):
    labels = np.unique(y_true)
    if len(labels) < 2:
        print(f"Warning: only one class present: {labels}. ROC AUC undefined -> NaN")
        return np.nan
    return float(roc_auc_score(y_true, y_score))

def predict_checkpoint(ckpt, model_name, image_size=IMAGE_SIZE, df_infer=None, batch_size=32):
    # build model and load weights safely
    m = TimmModel(model_name, pretrained=False).to(device)
    # robustly load checkpoint dict
    state = robust_torch_load(ckpt, map_location=device)
    # Some checkpoint files might have a nested structure; try to extract model_state
    if isinstance(state, dict) and 'model_state' in state:
        model_state = state['model_state']
    elif isinstance(state, dict) and any(k.endswith('model_state') for k in state.keys()):
        # try best-effort pick
        model_state = next((v for k,v in state.items() if 'model' in k and isinstance(v, dict)), None) or state
    else:
        model_state = state
    # If model_state looks like a full state_dict -> load
    try:
        if isinstance(model_state, dict):
            m.load_state_dict(model_state)
        else:
            # last resort: attempt to load top-level as state dict
            m.load_state_dict(state)
    except Exception as e:
        print(f"Warning: load_state_dict failed: {e}. Attempting non-strict load (if keys differ).")
        try:
            m.load_state_dict(model_state, strict=False)
        except Exception as e2:
            print("Failed non-strict load as well:", e2)
            raise

    m.eval()
    ds = ISICKaggleDataset(df_infer, transforms=get_transforms(image_size=image_size, train=False), h5_accessor=h5_accessor, img_map=img_map)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    preds=[]
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f'Predict {model_name}', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            out = m(imgs)
            preds.extend(torch.sigmoid(out).cpu().numpy().tolist())
    return np.array(preds)

if ckpts:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    _, valid_idx = list(skf.split(df, df['target']))[0]
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    ensemble_preds = np.zeros(len(valid_df))
    for ck in ckpts:
        model_key = os.path.basename(ck).split('_fold')[0]
        print("Predicting with", model_key, "ckpt", ck)
        p = predict_checkpoint(ck, model_key, image_size=IMAGE_SIZE, df_infer=valid_df, batch_size=BATCH_SIZE*2)
        if len(p) != len(valid_df):
            raise RuntimeError(f"Prediction length {len(p)} != valid length {len(valid_df)} for {ck}")
        ensemble_preds += p
    ensemble_preds /= len(ckpts)

    y_true = valid_df['target'].astype(int).values
    auc = safe_roc_auc(y_true, ensemble_preds)
    if np.isnan(auc):
        print("Ensemble AUC: NaN (single-class).")
    else:
        print("Ensemble AUC:", auc)

    y_pred = (ensemble_preds >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print("Ensemble metrics:", dict(acc=acc, prec=prec, rec=rec, f1=f1))
    print("Confusion matrix:\n", cm)
    pd.DataFrame([dict(ensemble_auc=(None if np.isnan(auc) else float(auc)), ensemble_acc=float(acc), ensemble_precision=float(prec), ensemble_recall=float(rec), ensemble_f1=float(f1))]).to_csv('/kaggle/working/ensemble_metrics_robust.csv', index=False)
    print("Saved ensemble metrics to /kaggle/working/ensemble_metrics_robust.csv")
else:
    print("No best ckpts found; skipping ensemble.")

# cleanup
if h5_accessor is not None:
    h5_accessor.close()

print("Done. Checkpoints in:", CHECKPOINT_DIR)
# ========================================================================


