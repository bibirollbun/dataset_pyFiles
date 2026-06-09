import os, gc, math, json, random
import numpy as np
import pandas as pd
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import timm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
seed_everything(42)

# Paths
DATA_DIR = "/kaggle/input/grand-xray-slam-division-a"
TRAIN_CSV = f"{DATA_DIR}/train1.csv"
IMG_TRAIN_DIR = f"{DATA_DIR}/train1"
IMG_TEST_DIR  = f"{DATA_DIR}/test1"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission_1.csv"

# === Add Data: dataset từ Notebook A (chứa artifacts/) ===
# Ví dụ: /kaggle/input/<your-view-mtl-output-dataset>/artifacts/...
VIEW_DATASET_DIR = "/kaggle/input/grand-a"  # <-- sửa tên dataset bạn vừa tạo
ART_DIR = Path(f"{VIEW_DATASET_DIR}/artifacts")

# Device & Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()
print("CUDA:", torch.cuda.is_available(), "| GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])

MAIN_BACKBONE = "convnext_tiny"
IMG_SIZE_MAIN = 320
BATCH_MAIN    = 24
EPOCHS_MAIN   = 10
LR_MAIN       = 2e-4
WD_MAIN       = 1e-4
ACCUM_TARGET  = 64

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum",
    "Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]


df = pd.read_csv(TRAIN_CSV)
IMAGE_COL_TRAIN = "Image_name" if "Image_name" in df.columns else "Image_Name"
df["image_path"] = df[IMAGE_COL_TRAIN].apply(lambda x: os.path.join(IMG_TRAIN_DIR, x))

# Folds theo bệnh nhân (nếu cần)
if "fold" not in df.columns or (df["fold"]<0).all():
    df["fold"] = -1
    gkf = GroupKFold(n_splits=5)
    for f,(tr,va) in enumerate(gkf.split(df, groups=df["Patient_ID"])):
        df.loc[va,"fold"] = f

# Base meta từ Age/Sex
def build_base_meta_train(_df):
    meta = pd.DataFrame(index=_df.index)
    age = _df["Age"].astype(float)
    mu = age.mean(skipna=True); sd = age.std(skipna=True) or 1.0
    meta["Age_z"] = ((age.fillna(mu) - mu) / sd).astype(np.float32)

    sex = _df["Sex"].astype(str).str.upper()
    meta["Sex_M"] = (sex=="MALE").astype(np.float32)
    meta["Sex_F"] = (sex=="FEMALE").astype(np.float32)
    meta["Sex_U"] = (~sex.isin(["MALE","FEMALE"])).astype(np.float32)
    return meta

df_meta_true = build_base_meta_train(df)
print("df_meta_true:", df_meta_true.shape)


# Load view meta từ dataset A
meta_train_view = np.load(ART_DIR/"meta_train_view.npy")
meta_test_view  = np.load(ART_DIR/"meta_test_view.npy")
with open(ART_DIR/"meta_cols_view.json") as f: meta_cols_view = json.load(f)
print("view meta shapes:", meta_train_view.shape, meta_test_view.shape)

# Base train meta
base_train_np = df_meta_true.values.astype(np.float32)
base_cols = list(df_meta_true.columns)

# Base test meta (Age/Sex không có -> zero/neutral)
sub = pd.read_csv(SAMPLE_SUB)
base_test = pd.DataFrame(0, index=np.arange(len(sub)), columns=base_cols, dtype=np.float32)
if "Age_z" in base_test.columns: base_test["Age_z"] = 0.0
base_test_np = base_test.values.astype(np.float32)

# Final META
meta_train_np = np.concatenate([base_train_np, meta_train_view], axis=1)
test_meta_np  = np.concatenate([base_test_np,  meta_test_view],  axis=1)
meta_cols = base_cols + meta_cols_view
META_DIM = meta_train_np.shape[1]
print("Final META:", meta_train_np.shape, test_meta_np.shape, "| META_DIM:", META_DIM)


assert all(c in df.columns for c in LABELS), "Thiếu một số cột 14 labels trong train1.csv"

def tfms_main(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE_MAIN, IMG_SIZE_MAIN)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE_MAIN, IMG_SIZE_MAIN)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

class MainTrainDS(Dataset):
    def __init__(self, df_idx, meta_np, tfm):
        self.df = df.loc[df_idx].reset_index(drop=True)
        self.meta = meta_np[df_idx].astype(np.float32)
        self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.loc[i]
        img = Image.open(row["image_path"]).convert("L").convert("RGB")
        img = self.tfm(img)
        y = torch.tensor(row[LABELS].values.astype(np.float32), dtype=torch.float32)
        m = torch.tensor(self.meta[i], dtype=torch.float32)
        return img, m, y

class MainTestDS(Dataset):
    def __init__(self, paths, meta_np, tfm):
        self.paths = list(paths); self.meta = meta_np.astype(np.float32); self.tfm=tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L").convert("RGB")
        img = self.tfm(img)
        m = torch.tensor(self.meta[i], dtype=torch.float32)
        return img, m


class ConvNextWithMeta(nn.Module):
    def __init__(self, name=MAIN_BACKBONE, meta_dim=8, n_out=14):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
        feat = self.backbone.num_features
        self.meta_bn = nn.BatchNorm1d(meta_dim)
        self.head = nn.Sequential(
            nn.Linear(feat+meta_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, n_out),
        )
    def forward(self, x, meta):
        f = self.backbone(x)
        meta = self.meta_bn(meta)
        z = torch.cat([f, meta], dim=1)
        return self.head(z)


# import time 
# # Helpers để format thời gian & lấy LR
# def fmt_hms(sec: float):
#     m = int(sec // 60); s = int(sec % 60); h = m // 60; m = m % 60
#     return f"{h:d}h{m:02d}m{s:02d}s" if h>0 else f"{m:02d}m{s:02d}s"

# def get_lr(optimizer):
#     for pg in optimizer.param_groups:
#         return pg.get("lr", None)

# # ====== REPLACE HÀM NÀY (bản không dùng tqdm) ======
# def train_one_fold_main(fold, epochs=EPOCHS_MAIN, log_every=50):
#     tr_idx = df.index[df.fold!=fold].values
#     va_idx = df.index[df.fold==fold].values

#     dl_tr = DataLoader(
#         MainTrainDS(tr_idx, meta_train_np, tfms_main(True)),
#         batch_size=BATCH_MAIN, shuffle=True, num_workers=4,
#         pin_memory=True, drop_last=True
#     )
#     dl_va = DataLoader(
#         MainTrainDS(va_idx, meta_train_np, tfms_main(False)),
#         batch_size=BATCH_MAIN*2, shuffle=False, num_workers=4,
#         pin_memory=True
#     )

#     model = ConvNextWithMeta(name=MAIN_BACKBONE, meta_dim=META_DIM, n_out=len(LABELS))
#     if torch.cuda.device_count()>1: model = nn.DataParallel(model)
#     model = model.to(DEVICE).to(memory_format=torch.channels_last)

#     y_tr = df.loc[tr_idx, LABELS].values.astype(np.float32)
#     pos  = y_tr.sum(axis=0) + 1e-3
#     neg  = (y_tr.shape[0] - y_tr.sum(axis=0)) + 1e-3
#     pos_w = torch.tensor(neg/pos, dtype=torch.float32).to(DEVICE)
#     crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
#     opt   = optim.AdamW(model.parameters(), lr=LR_MAIN, weight_decay=WD_MAIN)
#     sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=LR_MAIN*0.1)
#     scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
#     grad_accum = max(1, math.ceil(ACCUM_TARGET / BATCH_MAIN))

#     n_train_steps = len(dl_tr)
#     n_val_steps   = len(dl_va)
#     print(f"[MAIN] fold{fold} | train_steps={n_train_steps} | val_steps={n_val_steps} "
#           f"| bs={BATCH_MAIN} | accum={grad_accum} | lr0={LR_MAIN:g}")

#     best_auc = -1.0
#     if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

#     for ep in range(1, epochs+1):
#         # ---------- TRAIN ----------
#         model.train(); tr_loss = 0.0; seen_imgs = 0
#         opt.zero_grad(set_to_none=True)

#         ep_start = time.time()
#         # mốc để tính ips/eta ổn định sau vài bước
#         tick0 = None
#         for it, (x, m, y) in enumerate(dl_tr, 1):
#             if tick0 is None: tick0 = time.time()
#             x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
#             m = m.to(DEVICE, non_blocking=True)
#             y = y.to(DEVICE, non_blocking=True)

#             with autocast("cuda", enabled=torch.cuda.is_available()):
#                 logits = model(x, m)
#                 loss   = crit(logits, y) / grad_accum

#             scaler.scale(loss).backward()
#             if it % grad_accum == 0:
#                 scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

#             tr_loss   += (loss.item() * grad_accum) * x.size(0)
#             seen_imgs += x.size(0)

#             # LOG THỦ CÔNG mỗi log_every step
#             if (it % log_every == 0) or (it == n_train_steps):
#                 elapsed = max(1e-6, time.time() - tick0)
#                 ips     = seen_imgs / elapsed
#                 remain_imgs = n_train_steps * BATCH_MAIN - seen_imgs
#                 eta_s   = remain_imgs / max(1e-6, ips)
#                 avg_loss = tr_loss / max(1, seen_imgs)
#                 print(f"[fold{fold}] train ep{ep}/{epochs} | step {it:5d}/{n_train_steps:5d} "
#                       f"| loss {avg_loss:.4f} | lr {get_lr(opt):.2e} | ips {ips:.1f} img/s | eta {fmt_hms(eta_s)}")

#         # flush nếu còn dư tích lũy
#         if (it % grad_accum) != 0:
#             scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

#         tr_time = time.time() - ep_start
#         tr_loss = tr_loss / max(1, seen_imgs)
#         sch.step()

#         # ---------- VALID ----------
#         model.eval(); preds = []; targs = []
#         val_start = time.time()

#         val_seen = 0
#         tickv0 = None
#         with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
#             for j, (x, m, y) in enumerate(dl_va, 1):
#                 if tickv0 is None: tickv0 = time.time()
#                 x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
#                 m = m.to(DEVICE, non_blocking=True)
#                 logits = model(x, m).float().cpu().numpy()
#                 preds.append(1/(1+np.exp(-logits)))
#                 targs.append(y.numpy())

#                 # LOG thủ công cho valid cũng theo nhịp
#                 val_seen += x.size(0)
#                 if (j % log_every == 0) or (j == n_val_steps):
#                     elapsed = max(1e-6, time.time() - tickv0)
#                     ips     = val_seen / elapsed
#                     remain_imgs = n_val_steps * (BATCH_MAIN*2) - val_seen
#                     eta_s   = remain_imgs / max(1e-6, ips)
#                     print(f"[fold{fold}] valid ep{ep}/{epochs} | step {j:5d}/{n_val_steps:5d} "
#                           f"| ips {ips:.1f} img/s | eta {fmt_hms(eta_s)}")

#         va_time = time.time() - val_start
#         P = np.vstack(preds); T = np.vstack(targs)

#         aucs = []
#         for k in range(len(LABELS)):
#             try:   aucs.append(roc_auc_score(T[:,k], P[:,k]))
#             except: pass
#         mean_auc = float(np.mean(aucs)) if len(aucs)>0 else 0.0

#         peak_mem = (torch.cuda.max_memory_allocated()/1024**3) if torch.cuda.is_available() else 0.0
#         print(f"[MAIN fold{fold}] ep{ep}/{epochs} | train_loss {tr_loss:.4f} | val mAUC {mean_auc:.4f} "
#               f"| train {fmt_hms(tr_time)} | valid {fmt_hms(va_time)} | peak {peak_mem:.2f} GB")

#         if mean_auc > best_auc:
#             best_auc = mean_auc
#             sp = f"./main_fold{fold}.pth"
#             torch.save(model.module.state_dict() if hasattr(model,"module") else model.state_dict(), sp)
#             print("  -> saved", sp)

#         torch.cuda.empty_cache()
#         if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

#     return best_auc


# # =========== REPLACE: vòng chạy CV có log tổng thể ===========
# cv_scores = []
# t0_all = time.perf_counter()
# for fold in range(5):
#     print(f"\n===== START MAIN FOLD {fold} =====")
#     t_fold0 = time.perf_counter()
#     s = train_one_fold_main(fold); cv_scores.append(s)
#     fold_time = time.perf_counter() - t_fold0
#     done = fold + 1
#     remain = 5 - done
#     avg_per_fold = (time.perf_counter() - t0_all) / done
#     eta_total = remain * avg_per_fold
#     print(f"[DONE fold{fold}] AUC={s:.4f} | time {fmt_hms(fold_time)} | ETA total ~{fmt_hms(eta_total)}")

# print("\nMAIN fold AUCs:", [f"{x:.4f}" for x in cv_scores], "mean:", float(np.mean(cv_scores)))
# print("TOTAL time:", fmt_hms(time.perf_counter() - t0_all))


# ================== UNIFIED PREDICT CELL (no global collisions) ==================
import os, gc, glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

def run_inference():
    # ----------- 0) Các phụ thuộc/cấu hình lấy từ global nếu có -----------
    DEVICE      = globals().get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_MAIN  = globals().get('BATCH_MAIN', 16)
    MAIN_BACKBONE = globals().get('MAIN_BACKBONE', 'convnext_base')  # sửa đúng nếu bạn train khác

    # Class model phải có sẵn (định nghĩa như lúc train)
    if 'ConvNextWithMeta' not in globals():
        raise NameError("Chưa có class ConvNextWithMeta (hãy dán lại định nghĩa đúng như lúc train).")

    # ----------- 1) Sample submission + LABELS -----------
    sample_sub = '/kaggle/input/grand-xray-slam-division-a/sample_submission1.csv'
    if not os.path.exists(sample_sub):
        sample_sub = '/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv'
    sub = pd.read_csv(sample_sub)
    LABELS = [c for c in sub.columns if c != 'Image_name']

    # ----------- 2) Resolve test image paths -----------
    IMG_TEST_DIR = globals().get('IMG_TEST_DIR', None)
    if IMG_TEST_DIR is None:
        cands = [
            '/kaggle/input/grand-xray-slam-division-a/test_images',
            '/kaggle/input/grand-xray-slam-division-a/test',
            '/kaggle/input/grand-xray-slam-division-a/images/test',
            '/kaggle/input/grand-xray-slam-division-a'
        ]
        IMG_TEST_DIR = next((p for p in cands if os.path.isdir(p)), cands[-1])

    def resolve_path(root, name):
        p = os.path.join(root, name)
        if os.path.exists(p): return p
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            q = os.path.join(root, name + ext)
            if os.path.exists(q): return q
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if not hits:
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                hits = glob.glob(os.path.join(root, "**", name + ext), recursive=True)
                if hits: break
        return hits[0] if hits else os.path.join(root, name)

    test_paths = [resolve_path(IMG_TEST_DIR, n) for n in sub['Image_name']]
    miss = sum(not os.path.exists(p) for p in test_paths)
    if miss:
        print(f"[WARN] {miss} ảnh không tìm thấy. Kiểm tra tên file/đuôi ảnh.")

    # ----------- 3) Transforms -----------
    try:
        tfms = globals()['tfms_main'](False)
    except Exception:
        import torchvision.transforms as T
        IMG_SIZE = 384  # sửa đúng size đã train
        def tfms_main(is_train=False):
            return T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        tfms = tfms_main(False)

    # ----------- 4) Chọn weights -----------
    fold_weights = [f'./main_fold{f}.pth' for f in range(5)]
    have_5fold = all(os.path.exists(p) for p in fold_weights)
    single_main_weight = globals().get('SINGLE_MAIN_WEIGHT',
        '/kaggle/input/model1/pytorch/default/1/main_fold0.pth')  # chỉnh nếu khác
    weight_paths = fold_weights if have_5fold else [single_main_weight]
    print("[INFO] Weights:", weight_paths)

    # ----------- 5) Suy META_DIM từ ckpt + tạo meta_test_np -----------
    def _load_state_dict_any(path):
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                return obj['state_dict']
            return obj
        raise RuntimeError("Không nhận dạng được định dạng checkpoint.")

    def infer_meta_dim_from_ckpt(path):
        sd = _load_state_dict_any(path)
        keys = list(sd.keys())
        cands = [k for k in keys if k.endswith('meta_bn.weight') or k.endswith('meta_bn.bias')]
        if not cands:
            cands = [k for k in keys if 'meta' in k and (k.endswith('.weight') or k.endswith('.bias'))]
        if not cands:
            print("[INFO] Không thấy nhánh meta trong ckpt -> META_DIM=0")
            return 0
        t = sd[cands[0]]
        try: return int(t.shape[0])
        except: return int(t.numel())

    required_meta_dim = infer_meta_dim_from_ckpt(weight_paths[0])
    print(f"[INFO] Checkpoint yêu cầu META_DIM = {required_meta_dim}")

    # Nếu có sẵn meta_test_np global thì dùng; nếu không, tạo zeros đúng kích thước
    meta_test_np = globals().get('meta_test_np', None)
    if meta_test_np is None:
        n_test = len(sub)
        meta_test_np = np.zeros((n_test, max(required_meta_dim, 0)), dtype=np.float32)
        if required_meta_dim > 0:
            print("[WARN] Ckpt có nhánh meta nhưng bạn chưa cung cấp meta -> dùng zeros.")

    def adjust_meta(meta_np, k):
        c = meta_np.shape[1]
        if c == k: return meta_np
        if k == 0:
            print(f"[WARN] Model không dùng meta (k=0). Bỏ {c} cột meta.")
            return np.zeros((meta_np.shape[0], 0), dtype=meta_np.dtype)
        if c < k:
            pad = np.zeros((meta_np.shape[0], k - c), dtype=meta_np.dtype)
            print(f"[WARN] Padded meta {c} → {k}.")
            return np.concatenate([meta_np, pad], axis=1)
        print(f"[WARN] Sliced meta {c} → {k}. Đảm bảo thứ tự cột trùng lúc train.")
        return meta_np[:, :k]

    meta_test_np = adjust_meta(meta_test_np, required_meta_dim)
    META_DIM = required_meta_dim
    print("[INFO] meta_test_np shape:", meta_test_np.shape)

    # ----------- 6) Dataset/DataLoader (scope local) -----------
    class _MainTestDS(Dataset):
        def __init__(self, paths, meta_np, tfm):
            self.paths = list(paths)
            self.meta  = meta_np.astype(np.float32)
            self.tfm   = tfm
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("L").convert("RGB")
            img = self.tfm(img)
            m = torch.tensor(self.meta[i], dtype=torch.float32)
            return img, m

    dl_test = DataLoader(
        _MainTestDS(test_paths, meta_test_np, tfms),
        batch_size=BATCH_MAIN*2, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"[INFO] Test steps: {len(dl_test)} | batch: {BATCH_MAIN*2}")

    # ----------- 7) Load state_dict linh hoạt -----------
    def load_state_dict_flexible(model, sd):
        model_keys = set(model.state_dict().keys())
        sd_keys = set(sd.keys())
        if all(k.startswith('module.') for k in sd_keys):
            sd = {k.replace('module.','',1): v for k,v in sd.items()}
            sd_keys = set(sd.keys())
        try:
            model.load_state_dict(sd, strict=True)
        except Exception:
            missing = [k for k in model_keys if k not in sd_keys][:10]
            unexpected = [k for k in sd_keys if k not in model_keys][:10]
            print("[WARN] strict=False do lệch key.")
            if missing: print("  missing:", missing, "...")
            if unexpected: print("  unexpected:", unexpected, "...")
            model.load_state_dict(sd, strict=False)
        return model

    # ----------- 8) Hàm build model & predict -----------
       # ----------- 8) Hàm build model & predict -----------
    def build_model(weight_path):
        try:
            m = ConvNextWithMeta(name=MAIN_BACKBONE, meta_dim=META_DIM, n_out=len(LABELS))
        except Exception:
            m = ConvNextWithMeta(meta_dim=META_DIM, n_out=len(LABELS))
        sd = _load_state_dict_any(weight_path)
        m = load_state_dict_flexible(m, sd)
        m.eval()
        if torch.cuda.device_count() > 1:
            m = nn.DataParallel(m)
        return m.to(DEVICE)

    preds = np.zeros((len(sub), len(LABELS)), dtype=np.float32)
    with torch.no_grad():
        # vòng ngoài: theo dõi từng weight
        for wi, wpath in enumerate(weight_paths, 1):
            print(f"\n[RUN] Loading weight {wi}/{len(weight_paths)}: {wpath}")
            model = build_model(wpath)

            chunk = []
            seen = 0
            t0 = time.perf_counter()

            # vòng trong: theo dõi từng batch
            pbar = tqdm(
                DataLoader(
                    _MainTestDS(test_paths, meta_test_np, tfms),
                    batch_size=BATCH_MAIN*2, shuffle=False,
                    num_workers=4, pin_memory=True
                ),
                desc=f"[infer {wi}/{len(weight_paths)}]",
                leave=False
            )
            for x, mmeta in pbar:
                x = x.to(DEVICE, non_blocking=True)
                mmeta = mmeta.to(DEVICE, non_blocking=True)
                with autocast("cuda", enabled=(str(DEVICE).startswith("cuda") and torch.cuda.is_available())):
                    logits = model(x, mmeta)
                    probs  = torch.sigmoid(logits).detach().cpu().numpy()
                chunk.append(probs)

                seen += x.size(0)
                dt = max(1e-6, time.perf_counter() - t0)
                pbar.set_postfix(imgs=seen, ips=f"{seen/dt:.1f}")

            fold_pred = np.vstack(chunk)
            if len(weight_paths) == 5:
                preds += fold_pred / 5.0
            else:
                preds = fold_pred

            del model; gc.collect()
            if DEVICE != "cpu":
                torch.cuda.empty_cache()

    # ----------- 9) Save submission -----------
    sub_out = pd.DataFrame({'Image_name': sub['Image_name']})
    for i, lab in enumerate(LABELS):
        sub_out[lab] = preds[:, i].clip(0,1)
    sub_out.to_csv("submission.csv", index=False)
    print("\n[SAVED] submission.csv | shape:", sub_out.shape)
    return sub_out

# Chạy:
_ = run_inference()

