import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import pandas as pd
import csv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

COMP_PATH = '/kaggle/input/waveform-inversion'
train_dir = os.path.join(COMP_PATH, "train_samples")
test_dir = os.path.join(COMP_PATH, "test")
BATCH_SIZE = 16
N_FOLDS = 5
NUM_EPOCHS = 30
PATIENCE = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def find_label_file(casedir):
    # velocity.npy, vel.npy, model.npy, parent含む
    for vname in ["velocity.npy", "vel.npy", "model.npy"]:
        for search_dir in [casedir, os.path.join(casedir, "data")]:
            vpath = os.path.join(search_dir, vname)
            if os.path.isfile(vpath):
                return vpath
    for root, dirs, files in os.walk(casedir):
        for fname in files:
            if any(x in fname for x in ["velocity", "vel", "model"]) and fname.endswith(".npy"):
                return os.path.join(root, fname)
    return None

def get_train_triples(train_dir, verbose=True):
    triples = []
    max_chan = 0
    for case in sorted(os.listdir(train_dir)):
        case_dir = os.path.join(train_dir, case)
        label_file = find_label_file(case_dir)
        if not label_file:
            if verbose:
                print(f"警告: ラベルファイル見つからずスキップ: {case_dir}")
            continue
        for root, dirs, files in os.walk(case_dir):
            for fname in sorted(files):
                if fname.startswith("data") and fname.endswith(".npy"):
                    data_path = os.path.join(root, fname)
                    arr = np.load(data_path, mmap_mode='r')
                    shape = arr.shape
                    # shape: (N, C, 1000, 70) or (C, 1000, 70)
                    if len(shape) == 4:
                        n_sample, n_chan, tlen, wlen = shape
                    elif len(shape) == 3:
                        n_sample, n_chan, tlen, wlen = 1, *shape
                    else:
                        raise ValueError(f"data shapeが異常: {data_path} {shape}")
                    max_chan = max(max_chan, n_chan)
                    for i in range(n_sample):
                        triples.append({
                            "data_path": data_path,
                            "label_path": label_file,
                            "case": case,
                            "file": fname,
                            "idx": i,
                            "n_chan": n_chan,
                            "shape": (n_chan, tlen, wlen)
                        })
    if verbose:
        print(f"Train triples: {len(triples)}, Max channel: {max_chan}")
    return triples, max_chan

def get_test_triples(test_dir, verbose=True):
    triples = []
    max_chan = 0
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(test_dir, fname)
        arr = np.load(fpath, mmap_mode='r')
        shape = arr.shape
        # shape: (C, 1000, 70)
        if len(shape) == 3:
            n_chan, tlen, wlen = shape
        else:
            raise ValueError(f"test shape異常: {fpath} {shape}")
        max_chan = max(max_chan, n_chan)
        triples.append({
            "data_path": fpath,
            "label_path": None,
            "case": None,
            "file": fname,
            "idx": 0,
            "n_chan": n_chan,
            "shape": (n_chan, tlen, wlen)
        })
    if verbose:
        print(f"Test triples: {len(triples)}, Max channel: {max_chan}")
    return triples, max_chan



class WaveformDataset(Dataset):
    def __init__(self, triples, normalize=True, desired_channels=5):
        self.triples = triples
        self.normalize = normalize
        self.desired_channels = desired_channels
        arr = np.load(self.triples[0]["data_path"], mmap_mode='r')
        if len(arr.shape) == 4:
            ex = arr[0]
        else:
            ex = arr
        self.global_mean = ex.mean()
        self.global_std = ex.std() + 1e-8

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        arr = np.load(triple["data_path"], mmap_mode='r')
        if len(arr.shape) == 4:
            waves = arr[triple["idx"]]
        else:
            waves = arr
        n_chan = waves.shape[0]
        if n_chan < self.desired_channels:
            pad = ((0, self.desired_channels - n_chan), (0,0), (0,0))
            waves = np.pad(waves, pad)
        elif n_chan > self.desired_channels:
            waves = waves[:self.desired_channels]
        if self.normalize:
            waves = (waves - self.global_mean) / self.global_std
        waves = np.ascontiguousarray(waves, dtype=np.float32)
        # ラベル
        if triple["label_path"] is None:
            return torch.from_numpy(waves), torch.zeros(1, 70, 70)
        label_arr = np.load(triple["label_path"], mmap_mode='r')
        if label_arr.ndim == 4:
            velocity = label_arr[triple["idx"]]
        elif label_arr.ndim == 3:
            velocity = label_arr
        elif label_arr.ndim == 2:
            velocity = np.expand_dims(label_arr, axis=0)
        else:
            raise RuntimeError(f"label shape異常: {triple['label_path']} shape={label_arr.shape}")
        velocity = np.ascontiguousarray(velocity, dtype=np.float32)
        assert velocity.shape[-2:] == (70, 70), f"velocity shape不正: {velocity.shape}"
        return torch.from_numpy(waves), torch.from_numpy(velocity)



train_triples, max_train_chan = get_train_triples(train_dir)
test_triples, max_test_chan = get_test_triples(test_dir)
DESIRED_CHANNELS = min(max_train_chan, max_test_chan, 8)
print(f"Train max channels: {max_train_chan} | Test max channels: {max_test_chan} | Using: {DESIRED_CHANNELS}")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(train_triples))
fold_loaders = []
for train_idx, val_idx in folds:
    train_list = [train_triples[i] for i in train_idx]
    val_list = [train_triples[i] for i in val_idx]
    train_dataset = WaveformDataset(train_list, normalize=True, desired_channels=DESIRED_CHANNELS)
    val_dataset = WaveformDataset(val_list, normalize=True, desired_channels=DESIRED_CHANNELS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    fold_loaders.append((train_loader, val_loader))



class WaveformInversionUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up4_conv = nn.Sequential(nn.Conv2d(256+256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up3_conv = nn.Sequential(nn.Conv2d(128+128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.final_conv = nn.Conv2d(128, 1, 1)
    def forward(self, x):
        d1 = self.conv1(x)
        p1 = F.max_pool2d(d1, (2,2))
        d2 = self.conv2(p1)
        p2 = F.max_pool2d(d2, (2,1))
        d3 = self.conv3(p2)
        if d3.shape[2] % 2 == 1:
            d3 = F.pad(d3, (0,0,0,1))
        skip3 = d3
        p3 = F.max_pool2d(d3, (2,1))
        d4 = self.conv4(p3)
        if d4.shape[2] % 2 == 1:
            d4 = F.pad(d4, (0,0,0,1))
        skip4 = d4
        p4 = F.max_pool2d(d4, (2,1))
        d5 = self.conv5(p4)
        up4 = F.interpolate(d5, size=(skip4.shape[2], skip4.shape[3]), mode='bilinear', align_corners=False)
        u4 = self.up4_conv(torch.cat([up4, skip4], 1))
        up3 = F.interpolate(u4, size=(skip3.shape[2], skip3.shape[3]), mode='bilinear', align_corners=False)
        u3 = self.up3_conv(torch.cat([up3, skip3], 1))
        out = self.final_conv(u3)
        out = F.interpolate(out, size=(70, 70), mode='bilinear', align_corners=False)
        out = 1500.0 + F.relu(out - 1500.0)
        return out



def total_variation_loss(img):
    diff_i = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    diff_j = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return diff_i + diff_j

criterion_l1 = nn.L1Loss()
criterion_l2 = nn.MSELoss()

fold_models = []
for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
    print(f"\n===== Training Fold {fold_idx} =====")
    model = WaveformInversionUNet(in_channels=DESIRED_CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        train_loss = 0.0
        for batch_wave, batch_vel in tqdm(train_loader, desc=f"Fold{fold_idx+1} Epoch{epoch} [Train]", leave=False):
            batch_wave = batch_wave.to(DEVICE, dtype=torch.float, non_blocking=True)
            batch_vel = batch_vel.to(DEVICE, dtype=torch.float, non_blocking=True)
            optimizer.zero_grad()
            pred_vel = model(batch_wave)
            loss_val = criterion_l1(pred_vel, batch_vel) + criterion_l2(pred_vel, batch_vel)
            loss_tv = total_variation_loss(pred_vel)
            loss = loss_val + 1e-4 * loss_tv
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_wave, batch_vel in val_loader:
                batch_wave = batch_wave.to(DEVICE, dtype=torch.float, non_blocking=True)
                batch_vel = batch_vel.to(DEVICE, dtype=torch.float, non_blocking=True)
                pred_vel = model(batch_wave)
                loss_val = criterion_l1(pred_vel, batch_vel) + criterion_l2(pred_vel, batch_vel)
                loss_tv = total_variation_loss(pred_vel)
                loss = loss_val + 1e-4 * loss_tv
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    fold_models.append(model.cpu())



torch.backends.cudnn.benchmark = True  # 推論高速化（重要）
test_loader = DataLoader(
    WaveformDataset(test_triples, desired_channels=DESIRED_CHANNELS),
    batch_size=8, shuffle=False, num_workers=4, pin_memory=True
)

submit_path = '/kaggle/working/submission.csv'
rows_buffer = []
buffer_size = 50000  # 適度なバッファリングサイズ

with open(submit_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Id", "Predicted"])

for fold_idx, model in enumerate(fold_models):
    model.to(DEVICE).eval()

# 推論はバッチ単位でfoldごとにensembleを即時計算しRAM節約
with torch.inference_mode():
    for batch_idx, (batch_wave, _) in enumerate(tqdm(test_loader, desc="Fast Inference")):
        batch_wave = batch_wave.to(DEVICE, dtype=torch.float, non_blocking=True)
        
        preds = []
        for model in fold_models:
            pred_vel = model(batch_wave).cpu().numpy()  # (B, 1, 70, 70)
            preds.append(pred_vel)
        ensemble_pred = np.mean(preds, axis=0)[:, 0]  # (B, 70, 70)

        # CSV用行作成
        for i, pred_ens in enumerate(ensemble_pred):
            triple_idx = batch_idx * test_loader.batch_size + i
            triple = test_triples[triple_idx]
            base = os.path.basename(triple["data_path"]).replace('.npy', '')
            case = triple.get("case") or "test"
            sample_idx = triple["idx"]
            
            for x in range(70):
                for y in range(70):
                    rows_buffer.append([
                        f"{case}_{base}_{sample_idx}_{x}_{y}",
                        pred_ens[x, y]
                    ])

            # バッファが一定数を超えたら書き込み
            if len(rows_buffer) >= buffer_size:
                with open(submit_path, 'a', newline='') as f:
                    csv.writer(f).writerows(rows_buffer)
                rows_buffer = []

# 残りデータをすべて書き込み
if rows_buffer:
    with open(submit_path, 'a', newline='') as f:
        csv.writer(f).writerows(rows_buffer)


