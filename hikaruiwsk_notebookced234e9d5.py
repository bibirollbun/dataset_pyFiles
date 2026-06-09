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

# å®šæ•°
DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
MODEL_PATH = "/kaggle/input/cnn0527/pytorch/default/1/cnn_2d_fold5.pth"
OUTPUT_DIR = "/kaggle/working"


class CNN_2D(nn.Module):
    def __init__(self, img_size=1249, conv_drop=0.2, dense_drop=0.4):
        super().__init__()
        self.img_size = img_size  # æ‰‹å‹•æŒ‡å®šã�•ã‚Œã�Ÿã‚µã‚¤ã‚ºã‚’ä½¿ã�†

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(conv_drop), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(conv_drop), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(conv_drop), nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(conv_drop), nn.MaxPool2d(2),
            nn.Conv2d(256,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(conv_drop), nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            f_dummy = self.features(dummy)
            B, C, H, W = f_dummy.shape
            feat_dim = C * H * W
            h_dim = C * H
            v_dim = C * W

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(dense_drop),
            nn.Linear(512, 1), nn.Sigmoid()
        )

        self.reg_head = nn.Sequential(
            nn.Linear(h_dim + v_dim, 256), nn.ReLU(), nn.Dropout(dense_drop),
            nn.Linear(256, 2), nn.Sigmoid()
        )

    def forward(self, x):
        f = self.features(x)
        class_prob = self.cls_head(f)

        h_pool = f.max(dim=3)[0]
        v_pool = f.max(dim=2)[0]
        h_flat = h_pool.view(h_pool.size(0), -1)
        v_flat = v_pool.view(v_pool.size(0), -1)
        dir_feat = torch.cat([h_flat, v_flat], dim=1)

        coords = self.reg_head(dir_feat)
        return class_prob, coords


# # ãƒ¢ãƒ‡ãƒ«ã�®æº–å‚™
# model = CNN_2D().to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()

# img_size = model.img_size
# transform = transforms.Compose([
#     transforms.Resize((img_size, img_size), antialias=True),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# results = []
# test_dir = os.path.join(DATA_DIR, "test")
# print(f"ğŸ”� test_dir: {test_dir}")

# for tomo_id in tqdm(os.listdir(test_dir), desc="ğŸ“‚ Processing tomo_ids"):
#     tomo_path = os.path.join(test_dir, tomo_id)
#     if not os.path.isdir(tomo_path):
#         print(f"â�­ï¸� Skipping non-directory: {tomo_path}")
#         continue

#     print(f"ğŸ“� Now processing: {tomo_id} ({tomo_path})")

#     for fname in sorted(os.listdir(tomo_path)):
#         if not fname.endswith(".jpg"):
#             continue

#         z = int(fname.replace("slice_", "").replace(".jpg", ""))
#         img_path = os.path.join(tomo_path, fname)

#         try:
#             print(f"ğŸ–¼ï¸� Loading image: {img_path}")
#             img = Image.open(img_path).convert("L").filter(ImageFilter.SHARPEN)
#             edge = cv2.Canny(np.array(img), 50, 150)
#             img_tensor = transform(Image.fromarray(edge)).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 class_prob, coords = model(img_tensor)
#                 pred_cls = (class_prob.squeeze().item() > 0.5)

#                 if pred_cls:
#                     x_norm, y_norm = coords.squeeze().tolist()
#                     axis_0 = z
#                     axis_1 = int(round(y_norm * img_size))
#                     axis_2 = int(round(x_norm * img_size))
#                     print(f"âœ… Motor detected: prob=1 â†’ (x={axis_2}, y={axis_1}, z={axis_0})")
#                 else:
#                     axis_0, axis_1, axis_2 = -1, -1, -1
#                     print(f"â�Œ No motor detected: prob=0 at z={z}")

#                 results.append([tomo_id, axis_0, axis_1, axis_2])

#         except Exception as e:
#             print(f"[ERROR] Failed to process {img_path}: {e}")
#             continue


# âœ… é«˜é€ŸåŒ–ï¼šOpenCVãƒ™ãƒ¼ã‚¹ + ãƒ­ã‚°ä»˜ã��ã�®æ�¨è«–å‡¦ç�†
import cv2
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ãƒ¢ãƒ‡ãƒ«ã�®æº–å‚™
model = CNN_2D(img_size=1249).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

img_size = model.img_size
results = []
test_dir = os.path.join(DATA_DIR, "test")
print(f"ğŸ”� test_dir: {test_dir}")

for tomo_id in tqdm(os.listdir(test_dir), desc="ğŸ“‚ Processing tomo_ids"):
    tomo_path = os.path.join(test_dir, tomo_id)
    if not os.path.isdir(tomo_path):
        print(f"â�­ï¸� Skipping non-directory: {tomo_path}")
        continue

    print(f"ğŸ“� Now processing: {tomo_id} ({tomo_path})")

    for fname in sorted(os.listdir(tomo_path)):
        if not fname.endswith(".jpg"):
            continue

        z = int(fname.replace("slice_", "").replace(".jpg", ""))
        img_path = os.path.join(tomo_path, fname)

        try:
            print(f"ğŸ–¼ï¸� Loading image: {img_path}")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            edge = cv2.Canny(img, 50, 150)
            edge_resized = cv2.resize(edge, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            img_np = edge_resized.astype(np.float32) / 255.0
            img_np = (img_np - 0.5) / 0.5
            img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                class_prob, coords = model(img_tensor)
                pred_cls = (class_prob.squeeze().item() > 0.5)

                if pred_cls:
                    x_norm, y_norm = coords.squeeze().tolist()
                    axis_0 = z
                    axis_1 = int(round(y_norm * img_size))
                    axis_2 = int(round(x_norm * img_size))
                    print(f"âœ… Motor detected: prob=1 â†’ (x={axis_2}, y={axis_1}, z={axis_0})")
                else:
                    axis_0, axis_1, axis_2 = -1, -1, -1
                    print(f"â�Œ No motor detected: prob=0 at z={z}")

                results.append([tomo_id, axis_0, axis_1, axis_2])

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")
            continue



submission = pd.DataFrame(results, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… submission.csv saved!")


import os
print("submission.csv exists:", os.path.exists("/kaggle/working/submission.csv"))

