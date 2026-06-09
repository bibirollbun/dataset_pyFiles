# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import gc


# 定数
DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
MODEL_PATH = "/kaggle/input/2d_cnn0521/pytorch/default/1/2dcnn_fold5_final_20250521_064150.pth"
OUTPUT_DIR = "/kaggle/working"


# ✅ これが抜けているとエラーになる！
df = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))

param_dict = {}
for label in ["Array shape (axis 0)", "Array shape (axis 1)", "Array shape (axis 2)"]:
    df[f"{label}_dist"] = df[label] * df["Voxel spacing"]

max_dist = max(df["Array shape (axis 1)_dist"].max(), df["Array shape (axis 2)_dist"].max())
voxel_spacing_mean = df["Voxel spacing"].mean()
param_dict["max_dist"] = max_dist
param_dict["voxel_spacing_mean"] = voxel_spacing_mean

final_img_size = int(np.ceil(max_dist / voxel_spacing_mean))
print(f"✅ final_img_size = {final_img_size}")





# test_dir = os.path.join(DATA_DIR, "test")
# save_base = os.path.join(OUTPUT_DIR, "processed")
# os.makedirs(save_base, exist_ok=True)

# test_tomos = os.listdir(test_dir)
# for tomo_id in tqdm(test_tomos):
#     tomo_dir = os.path.join(test_dir, tomo_id)
#     save_dir = os.path.join(save_base, tomo_id)
#     os.makedirs(save_dir, exist_ok=True)

#     voxel_spacing = voxel_spacing_mean  # testには個別のspacing情報がない前提
#     target_size = int(np.ceil(max_dist / voxel_spacing))

#     image_names = sorted([f for f in os.listdir(tomo_dir) if f.endswith(".jpg")])
#     for img_name in image_names:
#         img_path = os.path.join(tomo_dir, img_name)
#         img = Image.open(img_path).convert("L")
#         img_np = np.array(img).astype(np.float32)

#         # 標準化 → 平均0埋め → リサイズ
#         mean, std = img_np.mean(), img_np.std()
#         img_std = img_np - mean if std == 0 else (img_np - mean) / std
#         h, w = img_std.shape

#         if h > target_size or w > target_size:
#             img_std = cv2.resize(img_std, (target_size, target_size), interpolation=cv2.INTER_AREA)
#             h, w = img_std.shape

#         padded = np.zeros((target_size, target_size), dtype=np.float32)
#         padded[:h, :w] = img_std
#         resized = cv2.resize(padded, (final_img_size, final_img_size), interpolation=cv2.INTER_LINEAR)

#         # # 正規化＆保存
#         # out_img = np.clip((resized - resized.min()) / (resized.ptp() + 1e-8) * 255, 0, 255).astype(np.uint8)
#         # Image.fromarray(out_img).save(os.path.join(save_dir, img_name))


class MultiTaskCNN(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
            nn.Conv2d(256,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.2), nn.MaxPool2d(2),
        )
        dummy = torch.zeros(1, 1, img_size, img_size)
        with torch.no_grad():
            f = self.features(dummy)
        self.cls_head = nn.Sequential(nn.Flatten(), nn.Linear(f.numel(), 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 2))
        h_dim, v_dim = f.shape[1]*f.shape[2], f.shape[1]*f.shape[3]
        self.reg_head = nn.Sequential(nn.Linear(h_dim + v_dim, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 2), nn.Sigmoid())

    def forward(self, x):
        f = self.features(x)
        logits = self.cls_head(f)
        h_pool = f.max(dim=3)[0]
        v_pool = f.max(dim=2)[0]
        h_flat, v_flat = h_pool.view(h_pool.size(0), -1), v_pool.view(v_pool.size(0), -1)
        coords = self.reg_head(torch.cat([h_flat, v_flat], dim=1))
        return logits, coords


final_img_size = 224  # 学習済みモデルに合わせる

model = MultiTaskCNN(img_size=final_img_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

print("✅ モデルのロード完了")


# --- 変換 ---
transform = transforms.Compose([
    transforms.Resize((final_img_size, final_img_size), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- データセット定義 ---
class TestDataset(Dataset):
    def __init__(self, base_dir, transform, final_img_size, voxel_spacing_mean, max_dist):
        self.samples = []
        self.final_img_size = final_img_size
        self.voxel_spacing_mean = voxel_spacing_mean
        self.max_dist = max_dist
        self.transform = transform

        for tomo_id in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, tomo_id)
            for fname in sorted(os.listdir(dir_path)):
                if fname.endswith(".jpg"):
                    z = int(fname.replace("slice_", "").replace(".jpg", ""))
                    self.samples.append((tomo_id, z, os.path.join(dir_path, fname)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tomo_id, z, path = self.samples[idx]
        try:
            img = Image.open(path).convert("L").resize((self.final_img_size, self.final_img_size))
            img_np = np.array(img)
            if self.transform:
                img = self.transform(img)
            return img, tomo_id, z

        except Exception as e:
            print(f"[ERROR] Failed to load image: {path} ({e})")
            dummy = torch.zeros(1, self.final_img_size, self.final_img_size)
            return dummy, tomo_id, z

# --- データローダ作成 ---
dataset = TestDataset(
    base_dir=os.path.join(DATA_DIR, "test"),
    transform=transform,
    final_img_size=final_img_size,
    voxel_spacing_mean=voxel_spacing_mean,
    max_dist=max_dist
)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

# --- 推論 ---
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with torch.no_grad():
    for imgs, tomo_ids, zs in tqdm(loader):
        imgs = imgs.to(device)
        logits, coords = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        coords = coords.cpu().numpy()
        for tomo_id, z, p, (x, y) in zip(tomo_ids, zs, probs, coords):
            results.append({
                "tomo_id": tomo_id,
                "slice_index": int(z),
                "x": float(x),
                "y": float(y),
                "prob": float(p)
            })

# --- 結果保存 ---
df_sub = pd.DataFrame(results)

df_best = df_sub.loc[df_sub.groupby("tomo_id")["prob"].idxmax()].copy()

df_best = df_best.rename(columns={
    "slice_index": "Motor axis 0",
    "y": "Motor axis 1",
    "x": "Motor axis 2"
})[["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]]

df_best.to_csv("/kaggle/working/submission.csv", index=False)

print("✅ formatted submission.csv saved!")


import os
print("submission.csv exists:", os.path.exists("/kaggle/working/submission.csv"))




