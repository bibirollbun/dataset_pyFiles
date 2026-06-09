import pandas as pd
import numpy as np
from pathlib import Path

TRANSACTIONS = "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"

OUT_DIR = Path("/kaggle/working/hm_subset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

dtype_tx = {
    "customer_id": "string",
    "article_id": "int32",
}
tx = pd.read_csv(
    TRANSACTIONS,
    usecols=["customer_id", "article_id", "t_dat"],
    dtype=dtype_tx,
    parse_dates=["t_dat"]
).rename(columns={"customer_id": "user", "article_id": "item", "t_dat": "ts"})

print(tx.shape, tx.ts.min(), tx.ts.max())


MONTHS = 9 

max_date = tx["ts"].max()
cutoff = max_date - pd.DateOffset(months=MONTHS)
tx_recent = tx[tx["ts"] >= cutoff].copy()

print(f"Max date: {max_date.date()} | Cutoff: {cutoff.date()} | Kept: {len(tx_recent):,} rows")


def core_filter(df, u_col="user", i_col="item", min_u=5, min_i=5, max_loops=5):
    prev_len = -1
    cur = df
    for _ in range(max_loops):
        if len(cur) == prev_len: break
        prev_len = len(cur)

        u_cnt = cur[u_col].value_counts()
        cur = cur[cur[u_col].isin(u_cnt[u_cnt >= min_u].index)]

        i_cnt = cur[i_col].value_counts()
        cur = cur[cur[i_col].isin(i_cnt[i_cnt >= min_i].index)]
    return cur

tx_filt = core_filter(tx_recent, min_u=5, min_i=5)
print(f"After 5-core: {len(tx_filt):,} rows | users: {tx_filt.user.nunique():,} | items: {tx_filt.item.nunique():,}")



# 1) Tạo danh sách ID duy nhất (sau khi đã 5-core)
u_list = tx_filt["user"].drop_duplicates().reset_index(drop=True)
i_list = tx_filt["item"].drop_duplicates().reset_index(drop=True)

# 2) Ánh xạ về chỉ số liên tục 0-based
user2idx = {u: i for i, u in enumerate(u_list)}
item2idx = {it: i for i, it in enumerate(i_list)}

# 3) Gắn cột chỉ số mới
tx_filt["u"] = tx_filt["user"].map(user2idx).astype("int32")
tx_filt["v"] = tx_filt["item"].map(item2idx).astype("int32")

# 4) Sắp theo (u, ts) để chuẩn bị cho split theo thời gian ở bước sau
tx_filt = tx_filt.sort_values(["u", "ts"]).reset_index(drop=True)


n_rows = len(tx_filt)
n_users = tx_filt["u"].nunique()
n_items = tx_filt["v"].nunique()
rows_per_user = n_rows / n_users

u_deg = tx_filt.groupby("u").size()
i_deg = tx_filt.groupby("v").size()

summary = {
    "rows_after_5core": n_rows,
    "users": n_users,
    "items": n_items,
    "avg_interactions_per_user": round(rows_per_user, 2),
    "min_deg_user": int(u_deg.min()),
    "p5_deg_user": int(np.percentile(u_deg, 5)),
    "p50_deg_user": int(np.percentile(u_deg, 50)),
    "p95_deg_user": int(np.percentile(u_deg, 95)),
    "min_deg_item": int(i_deg.min()),
    "p5_deg_item": int(np.percentile(i_deg, 5)),
    "p50_deg_item": int(np.percentile(i_deg, 50)),
    "p95_deg_item": int(np.percentile(i_deg, 95)),
    "time_span": (tx_filt["ts"].min().date().isoformat(), tx_filt["ts"].max().date().isoformat())
}
summary



OUT_DIR = Path("/kaggle/working/hm_subset_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lưu file interactions 
tx_filt[["u","v","ts"]].to_parquet(OUT_DIR / "hm_9m_5core.parquet", index=False)

# Lưu mapping 
pd.DataFrame({"raw_user_id": u_list, "user_idx": range(len(u_list))}).to_csv(OUT_DIR / "user_id_map_9m.csv", index=False)
pd.DataFrame({"raw_item_id": i_list, "item_idx": range(len(i_list))}).to_csv(OUT_DIR / "item_id_map_9m.csv", index=False)

print("Saved:", OUT_DIR)



SUBSET_DIR = Path("/kaggle/working/hm_subset_final") 
PARQUET = SUBSET_DIR / "hm_9m_5core.parquet"          
OUT_DIR = Path("/kaggle/working/hm_split_9m")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET)
df = df.sort_values(["u", "ts"]).reset_index(drop=True)


# SPLIT: val = last-1, test = last, còn lại train 
train_pairs, val_pairs, test_pairs = {}, {}, {}
skipped_users = 0

for u, g in df.groupby("u", sort=False):
    items = g["v"].tolist()  # giữ thứ tự thời gian và repeat
    if len(items) < 3:
        skipped_users += 1
        continue

    test_item = items[-1]
    val_item  = items[-2]
    train_items = items[:-2]

    # Khi ghi train.txt, mỗi item xuất hiện MỘT lần (đúng format LightGCN/NGCF)
    train_unique = list(dict.fromkeys(train_items))  # giữ thứ tự, remove duplicates

    if len(train_unique) == 0:
        skipped_users += 1
        continue

    train_pairs[u] = train_unique
    val_pairs[u]   = [val_item]
    test_pairs[u]  = [test_item]

def write_user_item_list(path, user_dict):
    with open(path, "w") as f:
        for u, items in user_dict.items():
            f.write(str(u))
            for it in items:
                f.write(" " + str(it))
            f.write("\n")

write_user_item_list(OUT_DIR/"train.txt", train_pairs)
write_user_item_list(OUT_DIR/"val.txt",   val_pairs)
write_user_item_list(OUT_DIR/"test.txt",  test_pairs)

stats = {
    "users_total_in_subset": int(df["u"].nunique()),
    "users_used_for_split":  int(len(train_pairs)),
    "users_skipped_lt3":     int(skipped_users),
    "train_lines":           sum(1 for _ in open(OUT_DIR/"train.txt","r")),
    "val_lines":             sum(1 for _ in open(OUT_DIR/"val.txt","r")),
    "test_lines":            sum(1 for _ in open(OUT_DIR/"test.txt","r")),
    "policy": "per-user temporal split (val = last-1, test = last), keep repeats; train.txt lists unique items"
}

import json, os
with open(OUT_DIR/"split_manifest.json", "w") as f:
    json.dump(stats, f, indent=2)

for p in ["train.txt","val.txt","test.txt","split_manifest.json"]:
    sz = os.path.getsize(OUT_DIR/p)/1024**2
    print(f"{p}: {sz:.2f} MB")

repeat_val = sum(1 for u in val_pairs if val_pairs[u][0] in set(train_pairs[u]))
repeat_test = sum(1 for u in test_pairs if test_pairs[u][0] in set(train_pairs[u]))
print({"val_item_in_train_cnt": repeat_val, "test_item_in_train_cnt": repeat_test})


import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import math

# ========= 0. PATH SETUP =========
INPUT_DIR = Path("/kaggle/input/hm-subset-final")     # Thư mục input (read-only)
PARQUET_PATH = INPUT_DIR / "hm_9m_5core.parquet"      # File parquet (u, v, ts)

WORK_SPLIT_DIR = Path("/kaggle/working/hm_splits")    # Nơi sẽ chứa train/val/test + weights
WORK_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

print("INPUT_DIR:", INPUT_DIR)
print("WORK_SPLIT_DIR:", WORK_SPLIT_DIR)

# ========= 1. COPY train/val/test.txt SANG /kaggle/working =========
for name in ["train.txt", "val.txt", "test.txt"]:
    src = INPUT_DIR / name
    dst = WORK_SPLIT_DIR / name
    if src.exists():
        shutil.copy(src, dst)
        print(f"✅ Copied {src} -> {dst}")
    else:
        print(f"⚠ File không tồn tại: {src}")

# ========= 2. LOAD parquet gốc (u, v, ts) =========
df = pd.read_parquet(PARQUET_PATH)
print("\n=== PARQUET HEAD ===")
print(df.head())
print("Min ts:", df["ts"].min(), "| Max ts:", df["ts"].max())

# ========= 3. LOAD train.txt hiện tại (ở WORK_SPLIT_DIR) =========
train_path = WORK_SPLIT_DIR / "train.txt"
train_pairs = []

with open(train_path) as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        u = int(parts[0])
        items = list(map(int, parts[1:]))
        for v in items:
            train_pairs.append((u, v))

edges_train = pd.DataFrame(train_pairs, columns=["u", "v"])
edges_train = edges_train.drop_duplicates()

print("\nSố cặp (u,v) trong train.txt:", len(edges_train))

# ========= 4. LỌC parquet chỉ giữ (u,v) có trong train.txt =========
df_merge = df.merge(edges_train, on=["u", "v"], how="inner")
print("Số dòng khớp (u,v) giữa parquet & train:", len(df_merge))

# ========= 5. TÍNH TIME-DECAY (half-life = 45 ngày) =========
max_ts = df_merge["ts"].max()
print("\nMax ts dùng làm mốc thời gian:", max_ts)

HALF_LIFE_DAYS = 45.0
lambda_ = math.log(2.0) / HALF_LIFE_DAYS  # ln 2 / half-life

df_merge["age_days"] = (max_ts - df_merge["ts"]).dt.days.astype("float32")
df_merge["weight_raw"] = np.exp(-lambda_ * df_merge["age_days"])

print("\n=== THỐNG KÊ WEIGHT_RAW ===")
print("Min age_days:", df_merge["age_days"].min(), "Max:", df_merge["age_days"].max())
print("Mean weight_raw:", df_merge["weight_raw"].mean())
print("Min weight_raw:", df_merge["weight_raw"].min(), "Max:", df_merge["weight_raw"].max())

# ========= 6. GỘP THEO (u,v): cộng weight cho các lần mua lặp lại =========
edges = (
    df_merge
    .groupby(["u", "v"], as_index=False)["weight_raw"]
    .sum()
    .rename(columns={"weight_raw": "weight"})
)

print("\nEdges before norm (5 dòng đầu):")
print(edges.head())

# ========= 7. CHUẨN HOÁ THEO USER (tổng weight mỗi user = 1) =========
edges["sum_u"] = edges.groupby("u")["weight"].transform("sum")
edges["weight_norm"] = edges["weight"] / edges["sum_u"]

edges_final = edges[["u", "v", "weight_norm"]].rename(columns={"weight_norm": "weight"})

# ========= 8. LƯU RA CSV Ở /kaggle/working/hm_splits =========
out_path = WORK_SPLIT_DIR / "train_time_weights.csv"
edges_final.to_csv(out_path, index=False)
print("\n✅ Saved time-decay weights to:", out_path)

summary = {
    "num_train_pairs": int(len(edges_train)),
    "num_weighted_edges": int(len(edges_final)),
    "half_life_days": HALF_LIFE_DAYS,
}
print("\n=== SUMMARY ===")
print(summary)



import pandas as pd
import numpy as np
import math
from pathlib import Path

VIB_ROOT = Path("/kaggle/input/vibrent-clothes-rental-dataset")
OUT_DIR  = Path("/kaggle/working/vibrent_cf_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

vib_raw = pd.read_csv(
    VIB_ROOT / "user_activity_triplets.csv",
    sep=";" 
)

vib_raw = vib_raw[["customer.id", "outfit.id", "rentalPeriod.start"]].copy()
vib_raw = vib_raw.rename(columns={
    "customer.id": "user_raw",
    "outfit.id": "item_raw",
    "rentalPeriod.start": "ts",
})
vib_raw["ts"] = pd.to_datetime(vib_raw["ts"])

print("==== [RAW VIBRENT] ====")
print(f"Rows raw:   {len(vib_raw):,}")
print(f"Users raw:  {vib_raw['user_raw'].nunique():,}")
print(f"Items raw:  {vib_raw['item_raw'].nunique():,}")
print(f"Earliest ts: {vib_raw['ts'].min()}")
print(f"Latest ts:   {vib_raw['ts'].max()}")
print(f"Span days:   {(vib_raw['ts'].max() - vib_raw['ts'].min()).days}")
display(vib_raw.head())


min_interactions = 3

user_counts = vib_raw.groupby("user_raw")["item_raw"].count()
good_users = user_counts[user_counts >= min_interactions].index

vib = vib_raw[vib_raw["user_raw"].isin(good_users)].copy()

print("\n==== [AFTER FILTER USER >= 3] ====")
print(f"Rows filtered: {len(vib):,}")
print(f"Users filtered: {vib['user_raw'].nunique():,}")
print(f"Items filtered: {vib['item_raw'].nunique():,}")


vib = vib.sort_values(["user_raw", "ts"]).reset_index(drop=True)

u_list = vib["user_raw"].drop_duplicates().reset_index(drop=True)
i_list = vib["item_raw"].drop_duplicates().reset_index(drop=True)

user2idx = {u: i for i, u in enumerate(u_list)}
item2idx = {it: i for i, it in enumerate(i_list)}

vib["u"] = vib["user_raw"].map(user2idx).astype("int32")
vib["v"] = vib["item_raw"].map(item2idx).astype("int32")

print("\n==== [MAPPING] ====")
print(f"num_users (CF index): {vib['u'].nunique():,}")
print(f"num_items (CF index): {vib['v'].nunique():,}")

# Lưu mapping (chỉ cho user đã lọc)
pd.DataFrame({"raw_user_id": u_list, "user_idx": range(len(u_list))}).to_csv(
    OUT_DIR / "user_id_map_vibrent.csv", index=False
)
pd.DataFrame({"raw_item_id": i_list, "item_idx": range(len(i_list))}).to_csv(
    OUT_DIR / "item_id_map_vibrent.csv", index=False
)


# =========================
# SPLIT LOO: train / val / test
# =========================
df = vib[["u", "v", "ts"]].copy()
df = df.sort_values(["u", "ts"]).reset_index(drop=True)

train_lines, val_lines, test_lines = [], [], []
total_users_cf = df["u"].nunique()
used_users = 0
skipped_users_loo = 0

for u, g in df.groupby("u"):
    items = g["v"].tolist()
    if len(items) < 3:
        # theoretically không nên xảy ra vì đã lọc user >=3 interactions,
        # nhưng mình vẫn check cho chắc
        skipped_users_loo += 1
        continue

    train_items = sorted(set(items[:-2]))
    val_item    = items[-2]
    test_item   = items[-1]

    if len(train_items) == 0:
        skipped_users_loo += 1
        continue

    used_users += 1
    train_lines.append(" ".join([str(u)] + [str(it) for it in train_items]))
    val_lines.append(f"{u} {val_item}")
    test_lines.append(f"{u} {test_item}")

split_dir = OUT_DIR
with open(split_dir / "train.txt", "w") as f:
    f.write("\n".join(train_lines))
with open(split_dir / "val.txt", "w") as f:
    f.write("\n".join(val_lines))
with open(split_dir / "test.txt", "w") as f:
    f.write("\n".join(test_lines))

print("\n==== [SPLIT LOO] ====")
print(f"Tổng user CF (sau filter): {total_users_cf:,}")
print(f"User thực sự có trong train/val/test: {used_users:,}")
print(f"User bị skip ở bước LOO (nếu có):    {skipped_users_loo:,}")
print("train.txt lines:", len(train_lines))
print("val.txt lines:  ", len(val_lines))
print("test.txt lines: ", len(test_lines))

# =========================
# LƯU INTERACTIONS (u, v, ts)
# =========================
df_inter = df.sort_values(["u", "ts"]).reset_index(drop=True)
parquet_path = OUT_DIR / "vibrent_interactions.parquet"
df_inter.to_parquet(parquet_path, index=False)

print("\n==== [INTERACTIONS PARQUET] ====")
print("Saved to:", parquet_path)
print("Rows:", len(df_inter))
print("Users in df_inter:", df_inter['u'].nunique())
print("Items in df_inter:", df_inter['v'].nunique())


# =========================
# TIME-DECAY WEIGHTS (HALF-LIFE = 365 DAYS)
# =========================
print("\n==== [TIME-DECAY WEIGHTING] ====")

df_all = df_inter.copy().sort_values(["u", "ts"]).reset_index(drop=True)

# đọc train.txt để lấy các cặp (u,v) trong train
train_pairs = []
with open(split_dir / "train.txt") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        u = int(parts[0])
        items = list(map(int, parts[1:]))
        for v_ in items:
            train_pairs.append((u, v_))

edges_train = pd.DataFrame(train_pairs, columns=["u", "v"]).drop_duplicates()

print("Users in train.txt:", len({u for u, _ in train_pairs}))
print("Unique (u,v) in train:", len(edges_train))

# merge để chỉ giữ interactions thuộc train
df_merge = df_all.merge(edges_train, on=["u", "v"], how="inner")
print("Interactions thuộc train:", len(df_merge))
print("Users trong df_merge:", df_merge['u'].nunique())

# half-life
HALF_LIFE_DAYS = 365.0
lambda_ = math.log(2.0) / HALF_LIFE_DAYS

max_ts = df_merge["ts"].max()
df_merge["age_days"] = (max_ts - df_merge["ts"]).dt.days.astype("float32")
df_merge["weight_raw"] = np.exp(-lambda_ * df_merge["age_days"])

print("\n[Stats] age_days:")
print(df_merge["age_days"].describe())
print("\n[Stats] weight_raw:")
print(df_merge["weight_raw"].describe())

# gộp (u,v), chuẩn hóa theo user
edges = (
    df_merge
    .groupby(["u", "v"], as_index=False)["weight_raw"]
    .sum()
    .rename(columns={"weight_raw": "weight"})
)

edges["sum_u"] = edges.groupby("u")["weight"].transform("sum")
edges["weight_norm"] = edges["weight"] / edges["sum_u"]
edges_final = edges[["u", "v", "weight_norm"]].rename(columns={"weight_norm": "weight"})

print("\n[Stats] weight_norm (sau chuẩn hóa theo user):")
print(edges_final["weight"].describe())
print("Users trong weights:", edges_final["u"].nunique())

weights_path = OUT_DIR / "train_time_weights_vibrent.csv"
edges_final.to_csv(weights_path, index=False)

print("\n==== DONE VIBRENT PIPELINE (CLEAN) ====")
print("Output folder:", OUT_DIR)
print("  - train.txt")
print("  - val.txt")
print("  - test.txt")
print("  - user_id_map_vibrent.csv")
print("  - item_id_map_vibrent.csv")
print("  - vibrent_interactions.parquet")
print("  - train_time_weights_vibrent.csv")


import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# Kiểm tra device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Competition H&M (ảnh)
IMAGES_ROOT = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations/images")

# Dataset custom của bạn
SUBSET_ROOT = Path("/kaggle/input/hm-subset-final")

# File mapping item
META_PATH = SUBSET_ROOT / "item_id_map_9m.csv"

# Nơi lưu output
OUT_FEAT_NPY  = Path("/kaggle/working/image_features_resnet50.npy")
OUT_FEAT_META = Path("/kaggle/working/image_features_meta.csv")

print("Images root:", IMAGES_ROOT)
print("Meta path  :", META_PATH)
meta = pd.read_csv(META_PATH)
print(meta.head())
print(meta.columns)



# Chuẩn hóa tên cột: raw_item_id -> article_id
meta = pd.read_csv(META_PATH)

# Đổi tên cột
meta = meta.rename(columns={"raw_item_id": "article_id"})

# Ép kiểu cho chắc
meta["article_id"] = meta["article_id"].astype(int)
meta["item_idx"]   = meta["item_idx"].astype(int)

print(meta.head())
print(meta.dtypes)



def article_id_to_path(article_id: int) -> Path:
    """
    Map article_id → đường dẫn ảnh trong folder images/
    Ví dụ: 705854001 -> '0705854001' -> folder '070' -> '0705854001.jpg'
    """
    aid_str = f"{int(article_id):010d}"   # zero-pad 10 digits (10 chữ số)
    folder = aid_str[:3]                  # 3 chữ số đầu làm tên folder
    filename = aid_str + ".jpg"
    return IMAGES_ROOT / folder / filename

# Tạo cột path
meta["image_path"] = meta["article_id"].apply(article_id_to_path)

# Kiểm tra file có tồn tại không
meta["has_image"] = meta["image_path"].apply(lambda p: p.exists())

print("Tổng item trong meta:", len(meta))
print("Số item có ảnh:", meta["has_image"].sum())

# Lọc ra chỉ giữ item thực sự có ảnh
meta = meta[meta["has_image"]].reset_index(drop=True)

print("Sau khi lọc, còn:", len(meta))
meta.head()


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Transform theo chuẩn ImageNet cho ResNet50
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # ResNet50 input 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # Mean ImageNet
        std=[0.229, 0.224, 0.225],          # Std ImageNet
    ),
])


class HMImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path   = row["image_path"]
        item_idx   = int(row["item_idx"])
        article_id = int(row["article_id"])

        # Đọc ảnh
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # đảm bảo 3 kênh

        # Áp dụng transform
        if self.transform is not None:
            img = self.transform(img)

        # Trả về: tensor ảnh + 2 id để lưu sau này
        return img, item_idx, article_id

dataset = HMImageDataset(meta, transform=img_transform)
print("Số ảnh trong dataset:", len(dataset))


BATCH_SIZE = 128       # nếu bị thiếu VRAM thì giảm 32
NUM_WORKERS = 2       # 2–4 tùy Kaggle

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,      # không cần shuffle, chỉ extract
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print("Số batch:", len(dataloader))

# Thử lấy 1 batch để kiểm tra shape
imgs, item_idx_batch, article_id_batch = next(iter(dataloader))
print("Batch image shape:", imgs.shape)          # [B, 3, 224, 224]
print("Batch item_idx shape:", item_idx_batch.shape)
print("Batch article_id shape:", article_id_batch.shape)


import torch
from torch import nn
from torchvision import models

# Đảm bảo có device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Thử API mới của torchvision trước
try:
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V1
    backbone = resnet50(weights=weights)
    print("Loaded ResNet50 with new weights API")
except Exception as e:
    print("New torchvision API failed, fallback to old pretrained=True. Error:", e)
    backbone = models.resnet50(pretrained=True)
    print("Loaded ResNet50 with pretrained=True")

# Bỏ lớp fully-connected cuối cùng, chỉ lấy feature 2048-d
backbone.fc = nn.Identity()

backbone.to(device)
backbone.eval()

# Tắt gradient cho nhanh & đỡ tốn VRAM
for p in backbone.parameters():
    p.requires_grad = False

print(backbone.fc)   # để thấy nó là Identity()



# Lấy 1 batch từ dataloader
imgs, item_idx_batch, article_id_batch = next(iter(dataloader))
imgs = imgs.to(device)

with torch.no_grad():
    feats = backbone(imgs)   # dự kiến: [B, 2048]

print("Input batch shape :", imgs.shape)
print("Feature batch shape:", feats.shape)



import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Đảm bảo lại đường dẫn output (nếu trước đó bạn đã khai báo thì có thể bỏ phần này)
OUT_FEAT_NPY  = Path("/kaggle/working/image_features_resnet50.npy")
OUT_FEAT_META = Path("/kaggle/working/image_features_meta.csv")

all_features   = []
all_item_idx   = []
all_article_id = []

backbone.eval()
with torch.no_grad():
    for imgs, item_idx_batch, article_id_batch in tqdm(dataloader, desc="Extracting features"):
        # Đưa ảnh lên GPU
        imgs = imgs.to(device)

        # Forward qua ResNet50 -> [B, 2048]
        feats = backbone(imgs)

        # Đưa về CPU & numpy
        feats = feats.cpu().numpy()
        item_idx_batch   = item_idx_batch.numpy()
        article_id_batch = article_id_batch.numpy()

        # Lưu vào list
        all_features.append(feats)
        all_item_idx.append(item_idx_batch)
        all_article_id.append(article_id_batch)

# Ghép tất cả batch lại
all_features   = np.concatenate(all_features, axis=0)        # [N, 2048]
all_item_idx   = np.concatenate(all_item_idx, axis=0)        # [N]
all_article_id = np.concatenate(all_article_id, axis=0)      # [N]

print("Final feature shape   :", all_features.shape)
print("Số item_idx   thu được:", len(all_item_idx))
print("Số article_id thu được:", len(all_article_id))



# 6.1. Lưu features dạng .npy
np.save(OUT_FEAT_NPY, all_features)
print(f"Saved features to: {OUT_FEAT_NPY}")

# 6.2. Lưu meta tương ứng mỗi dòng trong .npy
feat_meta = pd.DataFrame({
    "row_idx": np.arange(len(all_item_idx)),   # index dòng trong .npy
    "item_idx": all_item_idx,
    "article_id": all_article_id,
})

feat_meta.to_csv(OUT_FEAT_META, index=False)
print(f"Saved feature meta to: {OUT_FEAT_META}")

feat_meta.head()



# Load lại thử
feats = np.load(OUT_FEAT_NPY)
meta_loaded = pd.read_csv(OUT_FEAT_META)

print("feats.shape      :", feats.shape)
print("meta_loaded.shape:", meta_loaded.shape)

print(meta_loaded.head())


import shutil
from pathlib import Path

# Thư mục nguồn trên Kaggle (chứa các file picture.***)
SRC_DIR = Path("/kaggle/input/vibrent-clothes-rental-dataset/embeddings/EfficientNet_V2_L_final")

# Tên file zip sẽ tạo trong /kaggle/working
ZIP_BASE = Path("/kaggle/working/vibrent_image_embeddings_efficientnet_v2_l")

print("Source dir exists:", SRC_DIR.exists())
print("Zipping from:", SRC_DIR)

# Tạo file .zip
shutil.make_archive(str(ZIP_BASE), 'zip', SRC_DIR)

print("Saved zip to:", ZIP_BASE.with_suffix(".zip"))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "/kaggle/input/hm-subset-final/hm_9m_5core.parquet"
df = pd.read_parquet(path)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.head()


# Tự detect cột user/item/time theo những tên phổ biến trong pipeline của bạn
cols = set(df.columns)

user_col = "user" if "user" in cols else ("customer_id" if "customer_id" in cols else ("u" if "u" in cols else None))
item_col = "item" if "item" in cols else ("article_id" if "article_id" in cols else ("v" if "v" in cols else None))
time_col = "t_dat" if "t_dat" in cols else ("ts" if "ts" in cols else ("date" if "date" in cols else None))

assert user_col and item_col and time_col, f"Cannot detect columns. Found: {df.columns.tolist()}"
print("Detected:", user_col, item_col, time_col)

df = df[[user_col, item_col, time_col]].copy()
df.columns = ["user", "item", "t_dat"]
df["t_dat"] = pd.to_datetime(df["t_dat"])
df = df.sort_values(["user", "item", "t_dat"]).reset_index(drop=True)

df.head()



# diff ngày giữa 2 lần mua liên tiếp của cùng (user,item)
df["prev_t"] = df.groupby(["user", "item"])["t_dat"].shift(1)
df["gap_days"] = (df["t_dat"] - df["prev_t"]).dt.days

# Chỉ lấy những record thật sự là repeat (gap_days not null and >=0)
rep = df[df["gap_days"].notna() & (df["gap_days"] >= 0)].copy()

print("Total interactions:", len(df))
print("Repeat interactions (non-first per user-item):", len(rep))
print("Repeat ratio:", len(rep)/len(df))

# Thống kê percentiles
gaps = rep["gap_days"].values
pct = [10, 25, 50, 75, 90, 95, 99]
stats = {f"p{p}": np.percentile(gaps, p) for p in pct}
stats["mean"] = float(np.mean(gaps))
stats["median(p50)"] = float(np.median(gaps))
stats



plt.figure(figsize=(8,4))
plt.hist(rep["gap_days"], bins=100)
plt.title("Repeat purchase gap days (raw histogram)")
plt.xlabel("gap_days")
plt.ylabel("count")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(rep["gap_days"], bins=200, log=True)
plt.title("Repeat purchase gap days (log y-scale)")
plt.xlabel("gap_days")
plt.ylabel("count (log)")
plt.show()



def weight(age_days, half_life):
    lam = np.log(2) / half_life
    return np.exp(-lam * age_days)

cands = [15, 30, 45, 60, 90]
ages_to_check = [30, 45, 60, 90, 120, 180]

table = []
for hl in cands:
    row = {"half_life": hl}
    for a in ages_to_check:
        row[f"w@{a}d"] = weight(a, hl)
    table.append(row)

pd.DataFrame(table)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

VIB_ROOT = Path("/kaggle/input/vibrent-clothes-rental-dataset")

# 1) Load raw (giống bạn)
vib_raw = pd.read_csv(VIB_ROOT / "user_activity_triplets.csv", sep=";")
vib_raw = vib_raw[["customer.id", "outfit.id", "rentalPeriod.start"]].copy()
vib_raw = vib_raw.rename(columns={
    "customer.id": "user_raw",
    "outfit.id": "item_raw",
    "rentalPeriod.start": "ts",
})
vib_raw["ts"] = pd.to_datetime(vib_raw["ts"], errors="coerce")
vib_raw = vib_raw.dropna(subset=["user_raw","item_raw","ts"])

print("==== [RAW VIBRENT] ====")
print(f"Rows raw:   {len(vib_raw):,}")
print(f"Users raw:  {vib_raw['user_raw'].nunique():,}")
print(f"Items raw:  {vib_raw['item_raw'].nunique():,}")
print(f"Earliest ts: {vib_raw['ts'].min()}")
print(f"Latest ts:   {vib_raw['ts'].max()}")
print(f"Span days:   {(vib_raw['ts'].max() - vib_raw['ts'].min()).days}")

# 2) Filter user >= 3 interactions (giống bạn)
min_interactions = 3
user_counts = vib_raw.groupby("user_raw")["item_raw"].count()
good_users = user_counts[user_counts >= min_interactions].index
vib = vib_raw[vib_raw["user_raw"].isin(good_users)].copy()

print("\n==== [AFTER FILTER USER >= 3] ====")
print(f"Rows filtered: {len(vib):,}")
print(f"Users filtered: {vib['user_raw'].nunique():,}")
print(f"Items filtered: {vib['item_raw'].nunique():,}")

# 3) Sort & compute return gaps (ngày giữa 2 lần thuê liên tiếp của cùng user)
vib = vib.sort_values(["user_raw","ts"]).reset_index(drop=True)
vib["prev_ts"] = vib.groupby("user_raw")["ts"].shift(1)
vib["gap_days"] = (vib["ts"] - vib["prev_ts"]).dt.days
gaps = vib["gap_days"].dropna().astype(int)

print("\n==== [RETURN GAP STATS] ====")
print(f"#gaps (user consecutive rentals): {len(gaps):,}")

def pct_summary(x):
    return {
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
    }

stats = pct_summary(gaps.values)
display(pd.DataFrame([stats]))

# 4) Plot: histogram + CDF (rất hữu ích để “justify” half-life)
plt.figure()
plt.hist(gaps.clip(0, 365), bins=60)
plt.title("Vibrent: gap days between consecutive rentals (clipped to 0..365)")
plt.xlabel("gap_days")
plt.ylabel("count")
plt.show()

# log-scale histogram (nếu tail dài)
plt.figure()
plt.hist(gaps[gaps>0], bins=60, log=True)
plt.title("Vibrent: gap_days histogram (log count, gap>0)")
plt.xlabel("gap_days")
plt.ylabel("count (log)")
plt.show()

# CDF
x_sorted = np.sort(gaps.values)
cdf = np.arange(1, len(x_sorted)+1)/len(x_sorted)
plt.figure()
plt.plot(x_sorted, cdf)
plt.title("Vibrent: CDF of gap_days")
plt.xlabel("gap_days")
plt.ylabel("CDF")
plt.xlim(0, np.percentile(x_sorted, 99))  # zoom tới p99 cho dễ nhìn
plt.show()

# 5) Table weights for candidate half-life values
def weight_at(days, half_life):
    return 2 ** (-days/half_life)  # same as exp(-ln2*days/hl)

candidate_hl = [30, 60, 90, 180, 365, 540, 730]
probe_days = [30, 90, 180, 365, 540, 730]

rows = []
for hl in candidate_hl:
    row = {"half_life": hl}
    for d in probe_days:
        row[f"w@{d}d"] = weight_at(d, hl)
    rows.append(row)

df_w = pd.DataFrame(rows)
display(df_w)


