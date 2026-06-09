import pandas as pd

root = "/kaggle/input/histopathologic-cancer-detection"
df = pd.read_csv(f"{root}/train_labels.csv")
print(df.head())
print("rows:", len(df))
print(df["label"].value_counts())



#随机病理图像示例
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# 图像文件夹路径（按你 notebook 实际路径）
img_dir = "/kaggle/input/histopathologic-cancer-detection/train"

# 随机选一张图
img_name = random.choice(os.listdir(img_dir))
img_path = os.path.join(img_dir, img_name)

# 读取并显示
img = Image.open(img_path)

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis("off")
plt.title("Random histopathology image sample")

# 保存图片（关键）
plt.savefig("figure_4_1_random_image.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 随机病理图像示例（真实分辨率展示，避免模糊）
# =========================
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

img_dir = "/kaggle/input/histopathologic-cancer-detection/train"

valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)]
img_name = random.choice(img_files)
img_path = os.path.join(img_dir, img_name)

img = Image.open(img_path).convert("RGB")

# ---- 关键点：figsize 按“原始分辨率比例”来 ----
# 96px ÷ 96 DPI ≈ 1 inch → 接近原始显示
fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=96)

ax.imshow(img, interpolation="nearest")
ax.set_axis_off()
ax.set_title("Random histopathology image sample", fontsize=8)

# 保存时再用高 DPI（用于论文）
fig.savefig(
    "figure_4_1_random_image.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02
)

plt.show()

print(f"Image: {img_name}")



#标签分布柱状图
import pandas as pd
import matplotlib.pyplot as plt

# 读取标签
csv_path = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
df = pd.read_csv(csv_path)

# 统计并画图
label_counts = df["label"].value_counts()

plt.figure(figsize=(4, 4))
label_counts.plot(kind="bar")
plt.xticks(rotation=0)
plt.xlabel("Label")
plt.ylabel("Number of samples")
plt.title("Label distribution")

# 保存图片
plt.savefig("figure_4_1_label_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 0) 导入
# =========================
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# 为了复现性（可选）
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# =========================
# 1) 路径检查
# =========================
root = "/kaggle/input/histopathologic-cancer-detection"
print("root exists:", os.path.exists(root))
print("root content:", os.listdir(root))

train_csv = os.path.join(root, "train_labels.csv")
train_dir = os.path.join(root, "train")

print("train_csv exists:", os.path.exists(train_csv))
print("train_dir exists:", os.path.exists(train_dir))

# =========================
# 2) 读取标签 + 抽样（跑得快）
# =========================
df = pd.read_csv(train_csv)  # columns: id,label
print("total rows:", len(df))
print("label counts:\n", df["label"].value_counts())

# 抽 20000 张，先跑通（你以后可以改大）
df_small = df.sample(20000, random_state=42).reset_index(drop=True)
print("df_small rows:", len(df_small))

# =========================
# 3) Dataset
# =========================
class PatchDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir

        # 这个数据本来就是 96x96，小尺寸训练更适合CPU
        self.tf = T.Compose([
            T.Resize((96, 96)),
            T.ToTensor(),
            # 不用预训练时，norm 用 0.5/0.5 也行；用 ImageNet norm 也可
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "id"]
        label = float(self.df.loc[idx, "label"])  # 0/1
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")

        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

train_ds = PatchDataset(df_small, train_dir)
train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# =========================
# 4) 模型（关键：weights=None，避免下载失败）
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = models.resnet18(weights=None)  # <- 不会去联网下载
model.fc = nn.Linear(model.fc.in_features, 1)  # 输出一个logit
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# =========================
# 5) 训练（先跑 300 step 看效果）
# =========================
model.train()
for step, (x, y) in enumerate(train_loader):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    logits = model(x).squeeze(1)   # [B]
    loss = criterion(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            acc = ((prob >= 0.5).float() == y).float().mean().item()
        print(f"step={step:04d} loss={loss.item():.4f} acc={acc:.4f}")

    if step == 300:
        break

print("Training loop finished.")



# =========================================================
# Attention-MIL on Histopathologic Cancer Detection (Pseudo-bags)
# No internet needed, CPU OK.
# =========================================================
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# -------------------------
# 0) Utils
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# 1) Paths + Load CSV
# -------------------------
root = "/kaggle/input/histopathologic-cancer-detection"
train_csv = os.path.join(root, "train_labels.csv")
train_dir = os.path.join(root, "train")

df = pd.read_csv(train_csv)   # columns: id,label
print("total rows:", len(df))
print("label counts:\n", df["label"].value_counts())

# 为了让CPU也能较快：抽一个子集
# 你之后可以把 sample_n 改大，比如 80000 / 150000
sample_n = 40000
df = df.sample(sample_n, random_state=42).reset_index(drop=True)
print("working subset rows:", len(df))

df0 = df[df["label"] == 0].reset_index(drop=True)
df1 = df[df["label"] == 1].reset_index(drop=True)
print("subset label0:", len(df0), "label1:", len(df1))


# -------------------------
# 2) Pseudo-Bag Dataset
#    一个bag由K个patch组成，bag label 取 0/1
# -------------------------
class PseudoBagDataset(Dataset):
    """
    每次getitem随机生成一个bag：
      - 先随机选择 bag label ∈ {0,1}
      - 再从对应类别中随机采样 K 个 patch
    输出：
      X: [K, 3, H, W]
      y: scalar (float32)
    """
    def __init__(self, df0, df1, img_dir, bag_size=32, bags_per_epoch=2000, img_size=96):
        self.df0 = df0
        self.df1 = df1
        self.img_dir = img_dir
        self.bag_size = bag_size
        self.bags_per_epoch = bags_per_epoch

        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ])

    def __len__(self):
        return self.bags_per_epoch

    def _sample_ids(self, label):
        src = self.df1 if label == 1 else self.df0
        idx = np.random.randint(0, len(src), size=self.bag_size)
        return src.loc[idx, "id"].tolist()

    def __getitem__(self, idx):
        label = np.random.randint(0, 2)  # 0 or 1
        ids = self._sample_ids(label)

        imgs = []
        for _id in ids:
            path = os.path.join(self.img_dir, f"{_id}.tif")
            img = Image.open(path).convert("RGB")
            imgs.append(self.tf(img))

        X = torch.stack(imgs, dim=0)  # [K, 3, H, W]
        y = torch.tensor(float(label), dtype=torch.float32)
        return X, y


# -------------------------
# 3) CNN Feature Extractor
# -------------------------
class ResNet18_Feature(nn.Module):
    """
    ResNet18 去掉最后分类层，输出 feature:
      input:  [K,3,H,W]
      output: [K, D]
    """
    def __init__(self, out_dim=512):
        super().__init__()
        base = models.resnet18(weights=None)  # 不联网
        # 去掉fc
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # -> [B, 512, 1, 1]
        self.out_dim = out_dim

    def forward(self, x):
        f = self.backbone(x)          # [B, 512, 1, 1]
        f = f.flatten(1)              # [B, 512]
        return f


# -------------------------
# 4) Attention-MIL (Ilse-style)
# -------------------------
class AttentionMIL(nn.Module):
    """
    输入 instance features: [K, D]
    输出 bag logit: scalar
    返回 attention weights: [K]
    """
    def __init__(self, in_dim=512, attn_dim=128, dropout=0.25):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(attn_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, X):
        # X: [K, D]
        H = self.embed(X)                      # [K, A]
        A = torch.tanh(self.attn_V(H))         # [K, A]
        A = self.attn_w(A).squeeze(-1)         # [K]
        a = torch.softmax(A, dim=0)            # [K]

        M = torch.sum(a.unsqueeze(-1) * H, dim=0)  # [A]
        logit = self.classifier(M).squeeze(-1)     # scalar
        return logit, a


# -------------------------
# 5) Full MIL Model = CNN + AttentionMIL
# -------------------------
class MILModel(nn.Module):
    def __init__(self, attn_dim=128):
        super().__init__()
        self.feat = ResNet18_Feature(out_dim=512)
        self.mil  = AttentionMIL(in_dim=512, attn_dim=attn_dim)

    def forward(self, bag_imgs):
        """
        bag_imgs: [K,3,H,W]
        """
        feats = self.feat(bag_imgs)           # [K,512]
        logit, attn = self.mil(feats)         # scalar, [K]
        return logit, attn


# -------------------------
# 6) Train
# -------------------------
bag_size = 32
bags_per_epoch = 1500     # CPU建议 1000~2000
img_size = 96
batch_size = 1            # MIL常用 batch=1，最简单

train_ds = PseudoBagDataset(df0, df1, train_dir,
                            bag_size=bag_size,
                            bags_per_epoch=bags_per_epoch,
                            img_size=img_size)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

model = MILModel(attn_dim=128).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 2  # 先跑通；你之后可以设成 10/20

print("\nStart Training Attention-MIL ...")
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for step, (bag_imgs, y) in enumerate(train_loader):
        # bag_imgs: [1,K,3,H,W] -> [K,3,H,W]
        bag_imgs = bag_imgs.squeeze(0).to(device)
        y = y.to(device)

        logit, attn = model(bag_imgs)          # scalar, [K]
        loss = criterion(logit.unsqueeze(0), y)  # make shapes [1]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prob = torch.sigmoid(logit).item()
            pred = 1.0 if prob >= 0.5 else 0.0
            acc = 1.0 if pred == float(y.item()) else 0.0

        running_loss += loss.item()
        running_acc += acc

        if step % 50 == 0:
            print(f"epoch={epoch} step={step:04d} loss={loss.item():.4f} prob={prob:.3f} y={float(y.item())} "
                  f"attn_max={attn.max().item():.3f} attn_min={attn.min().item():.3f}")

        # 先跑少一点，确认没问题
        if step == 300:
            break

    print(f"Epoch {epoch} done. avg_loss={running_loss/(step+1):.4f} avg_acc={running_acc/(step+1):.4f}")

print("Training finished.")

# -------------------------
# 7) 导出一个 bag 的 attention（用于你后续可视化/汇报）
# -------------------------
model.eval()
with torch.no_grad():
    bag_imgs, y = train_ds[0]
    bag_imgs = bag_imgs.to(device)
    logit, attn = model(bag_imgs)
    prob = torch.sigmoid(logit).item()
    attn_cpu = attn.cpu()

print("\nExample bag attention exported:")
print("bag label:", float(y.item()), "prob:", prob)
print("top-5 attention weights:", torch.topk(attn_cpu, k=5).values.numpy())



import matplotlib.pyplot as plt

# 重新取一个 bag
bag_imgs, y = train_ds[0]
bag_imgs = bag_imgs.to(device)

model.eval()
with torch.no_grad():
    logit, attn = model(bag_imgs)

attn = attn.cpu().numpy()        # [K]
bag_imgs = bag_imgs.cpu()        # [K,3,H,W]

# 找 attention 最大和最小的 patch
top_idx = attn.argmax()
low_idx = attn.argmin()

def show_patch(img_tensor, title):
    img = img_tensor.permute(1,2,0).numpy()
    img = (img * 0.5 + 0.5).clip(0,1)  # 反归一化
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
show_patch(bag_imgs[top_idx], f"Top attention\nweight={attn[top_idx]:.3f}")

plt.subplot(1,2,2)
show_patch(bag_imgs[low_idx], f"Low attention\nweight={attn[low_idx]:.3f}")

plt.suptitle(f"Bag label={int(y.item())}  pred_prob={torch.sigmoid(logit).item():.3f}")
plt.show()


