# ---------------------------
#  Import libraries
# ---------------------------
import os
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.transforms.functional import normalize
import timm
import pydicom

# ---------------------------
#  Hard-coded dataset references
# ---------------------------
SERIES_ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/"
TRAIN_CSV   = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"

# ---------------------------
#  Config (assume CUDA + AMP)
# ---------------------------
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

BACKBONE = "efficientvit_b1"
OUT_INDICES = (0, 1, 2, 3)
FPN_OUT = 256
NUM_CLASSES = 2
BATCH_SIZE = 2
EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50
LOG_AVG = 128
AMP = True
CPU_COUNT = max(1, os.cpu_count() or 1)

# ---------------------------
#  Slice selection policy
# ---------------------------
SLICES_PER_SERIES = 3

# ---------------------------
#  EfficientViT backbone + FPN
# ---------------------------
class EfficientViTBackboneWithFPN(nn.Module):
    def __init__(self, model_name: str, out_indices: Tuple[int, ...], out_channels: int):
        super().__init__()
        self.body = timm.create_model(
            model_name,
            features_only=True,
            out_indices=out_indices,
            pretrained=True
        )
        in_channels_list = self.body.feature_info.channels()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict:
        feats = self.body(x)
        feat_dict = OrderedDict({str(i): f for i, f in enumerate(feats)})
        return self.fpn(feat_dict)

# ---------------------------
#  Build Mask R-CNN with EfficientViT+FPN
# ---------------------------
def build_model() -> MaskRCNN:
    backbone = EfficientViTBackboneWithFPN(BACKBONE, OUT_INDICES, FPN_OUT)
    model = MaskRCNN(backbone, num_classes=NUM_CLASSES)
    return model

# ---------------------------
#  DICOM sorting helpers
# ---------------------------
def _safe_get_instance_number(d) -> float:
    value = getattr(d, "InstanceNumber", None)
    return float(value) if value is not None else 0.0

def _safe_get_z(d) -> Optional[float]:
    ipp = getattr(d, "ImagePositionPatient", None)
    if ipp is None:
        return None
    try:
        return float(ipp[2])
    except Exception:
        return None

def _sort_dcm_paths_by_z(dcm_paths: List[str]) -> List[str]:
    try:
        mets = []
        for p in dcm_paths:
            d = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            z = _safe_get_z(d)
            key = z if z is not None else _safe_get_instance_number(d)
            mets.append((key, p))
        mets.sort(key=lambda t: t[0])
        return [p for _, p in mets]
    except Exception:
        return sorted(dcm_paths)

# ---------------------------
#  Per-series indexing helper (returns multiple well-placed slices)
# ---------------------------
def _pick_well_placed_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n < 3 or k == 1:
        return [n // 2]
    if k == 3:
        return [n // 4, n // 2, (3 * n) // 4]
    if k == 5:
        return [n // 6, n // 3, n // 2, (2 * n) // 3, (5 * n) // 6]
    step = max(1, n // (k + 1))
    centers = [step * (i + 1) for i in range(k)]
    centers = [min(n - 1, max(0, c)) for c in centers]
    return sorted(list(dict.fromkeys(centers)))

def _index_one_series(args: Tuple[str, str, int, int]) -> Optional[List[Tuple[str, int]]]:
    series_root, sid, flag, k = args
    series_dir = os.path.join(series_root, sid)
    if not os.path.isdir(series_dir):
        return None

    dcm_paths = []
    for r, _, files in os.walk(series_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dcm_paths.append(os.path.join(r, f))
    if not dcm_paths:
        return None

    try:
        sorted_paths = _sort_dcm_paths_by_z(dcm_paths)
    except Exception:
        sorted_paths = sorted(dcm_paths)

    n = len(sorted_paths)
    idxs = _pick_well_placed_indices(n, k)
    picks = [sorted_paths[i] for i in idxs]
    return [(p, int(flag)) for p in picks]

# ---------------------------
#  RSNA CTA slice dataset with parallel indexing
# ---------------------------
class RSNACTAMaskDataset(Dataset):
    def __init__(self, series_root: str, csv_path: str, slices_per_series: int = SLICES_PER_SERIES):
        self.root = series_root
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Modality"].astype(str).str.upper() == "CTA"]
        self.series_ids = self.df["SeriesInstanceUID"].astype(str).tolist()
        self.ap_flags = self.df["Aneurysm Present"].astype(float).fillna(0.0).astype(int).tolist()
        self.series_to_flag = dict(zip(self.series_ids, self.ap_flags))
        self.k = max(1, int(slices_per_series))

        print(f"[Index] Starting parallel indexing with {CPU_COUNT} processes over {len(self.series_ids)} series...")
        self.samples = self._parallel_index_slices(self.series_ids, self.series_to_flag, self.k)
        print(f"[Index] Completed. Indexed {len(self.samples)} slice paths from {len(self.series_ids)} series.")

    def _parallel_index_slices(self, series_ids: List[str], flag_map: Dict[str, int], k: int) -> List[Tuple[str, int]]:
        tasks = [(self.root, sid, flag_map.get(sid, 0), k) for sid in series_ids]
        out: List[Tuple[str, int]] = []
        with ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
            futures = [ex.submit(_index_one_series, t) for t in tasks]
            done = 0
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    out.extend(res)
                done += 1
                if done % 256 == 0:
                    print(f"[Index] processed {done}/{len(futures)} series...")
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _read_dicom_image(self, path: str) -> torch.Tensor:
        dcm = pydicom.dcmread(path, force=True)
        arr = dcm.pixel_array.astype("float32")
        m = float(arr.mean())
        s = float(arr.std()) + 1e-6
        arr = (arr - m) / s
        t = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[0] != 3:
            t = t.permute(2, 0, 1)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
        return normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx: int):
        dcm_path, ap = self.samples[idx]
        image = self._read_dicom_image(dcm_path)
        H = int(image.shape[1])
        W = int(image.shape[2])

        if ap > 0:
            cx = W // 2
            cy = H // 2
            rw = max(16, int(W * 0.6))
            rh = max(16, int(H * 0.6))
            x1 = max(0, cx - rw // 2)
            y1 = max(0, cy - rh // 2)
            x2 = min(W - 1, x1 + rw)
            y2 = min(H - 1, y1 + rh)
            boxes = torch.tensor([[float(x1), float(y1), float(x2), float(y2)]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)
            masks = torch.zeros((1, H, W), dtype=torch.uint8)
            masks[0, y1:y2, x1:x2] = 1
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, H, W), dtype=torch.uint8)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.numel() else torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        return image, target

# ---------------------------
#  Data utilities
# ---------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------
#  One training epoch
# ---------------------------
def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(loader):
        images = [img.to(DEVICE, non_blocking=True) for img in images]
        targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=AMP):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += losses.item()
        if (i + 1) % PRINT_FREQ == 0:
            print(f"[Train] epoch={epoch} iter={i+1}/{len(loader)} loss={losses.item():.4f}")
        if (i + 1) % LOG_AVG == 0:
            avg_loss = running_loss / LOG_AVG
            print(f"[Train] epoch={epoch} iter={i+1} avg_loss(last {LOG_AVG})={avg_loss:.4f}")
            running_loss = 0.0

# ---------------------------
#  Evaluation using train set (compute loss)
# ---------------------------
@torch.no_grad()
def evaluate_loss(model, loader, epoch):
    model.train()
    total = 0.0
    count = 0
    for i, (images, targets) in enumerate(loader):
        images = [img.to(DEVICE, non_blocking=True) for img in images]
        targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
        with torch.amp.autocast(device_type="cuda", enabled=AMP):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
        total += float(loss.item())
        count += 1
        if (i + 1) % PRINT_FREQ == 0:
            print(f"[Eval] epoch={epoch} iter={i+1}/{len(loader)} loss={loss.item():.4f}")
    return total / max(1, count)

# ---------------------------
#  Main training loop
# ---------------------------
def main():
    print(f"[Setup] Using {CPU_COUNT} CPU workers for dataset indexing and DataLoader.")
    dataset = RSNACTAMaskDataset(series_root=SERIES_ROOT, csv_path=TRAIN_CSV, slices_per_series=SLICES_PER_SERIES)
    print(f"[Data] Loaded dataset with {len(dataset)} slice samples")

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=CPU_COUNT,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=False
    )

    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=CPU_COUNT,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=False
    )

    print("[Model] Building EfficientViT+FPN Mask R-CNN...")
    model = build_model().to(DEVICE)
    print("[Model] Built. Starting training...")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        val_loss = evaluate_loss(model, val_loader, epoch)
        print(f"[Eval] epoch={epoch} avg_val_loss={val_loss:.4f}")
        scheduler.step()
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "maskrcnn_efficientvit_best.pth")
            print(f"[Checkpoint] Saved new best model at epoch={epoch} val_loss={val_loss:.4f}")

# ---------------------------
#  Entrypoint
# ---------------------------
if __name__ == "__main__":
    main()

