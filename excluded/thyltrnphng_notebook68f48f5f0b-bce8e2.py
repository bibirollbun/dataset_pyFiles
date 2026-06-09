#Import thư viện cơ bản
import os, random, warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
#Thư viện xử lý dữ liệu & hình ảnh
import h5py
from PIL import Image
import matplotlib.pyplot as plt
#Thư viện Machine Learning (sklearn)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
#Thư viện PyTorch (Deep Learning)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
#Thư viện torchvision (Computer Vision)
import torchvision.transforms as T
import torchvision.models as models
from tqdm.auto import tqdm
#seed để tái lập kết quả
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
#Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
torch.backends.cudnn.benchmark = True



#Đặt đường dẫn cơ sở (BASE)
BASE = "/kaggle/input/isic-2024-challenge"

#Đường dẫn cụ thể cho từng file
TRAIN_META = f"{BASE}/train-metadata.csv"
TEST_META  = f"{BASE}/test-metadata.csv"
TRAIN_H5   = f"{BASE}/train-image.hdf5"
TEST_H5    = f"{BASE}/test-image.hdf5"

#Kiểm tra sự tồn tại của file
print("TRAIN_META exists:", os.path.exists(TRAIN_META))
print("TEST_META  exists:", os.path.exists(TEST_META))
print("TRAIN_H5   exists:", os.path.exists(TRAIN_H5))
print("TEST_H5    exists:", os.path.exists(TEST_H5))



# Kích thước ảnh theo ImageNet
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

#Kích thước tập dữ liệu huấn luyện
TARGET_TOTAL = 20_000

#Tham số huấn luyện mô hình
BATCH_SIZE = 32    
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30          
NUM_WORKERS = 2



#Load metadata từ CSV
train_all = pd.read_csv(TRAIN_META)
test_meta = pd.read_csv(TEST_META)

print("Full train:", train_all.shape)
print("Full target dist:\n", train_all["target"].value_counts())

#Lấy toàn bộ mẫu dương tính (class 1)
df_pos = train_all[train_all["target"] == 1].copy()
n_pos = len(df_pos)

#Tính số lượng mẫu âm tính cần thêm
n_neg_need = TARGET_TOTAL - n_pos
if n_neg_need <= 0:
    raise ValueError(f"TARGET_TOTAL ({TARGET_TOTAL}) phải > số positive ({n_pos}).")

#Lấy ngẫu nhiên mẫu âm tính (class 0)
df_neg = train_all[train_all["target"] == 0].sample(n=n_neg_need, random_state=SEED)

#Ghép lại thành subset và shuffle
df_sub = pd.concat([df_pos, df_neg], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)

#Kiểm tra subset
print("Subset shape:", df_sub.shape)
print("Subset dist:\n", df_sub["target"].value_counts())
print("Subset pos rate:", df_sub["target"].mean())



import matplotlib.pyplot as plt

# Phân bố trước khi subset
target_dist_full = train_all["target"].value_counts()

# Phân bố sau khi subset
target_dist_sub = df_sub["target"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Biểu đồ trước subset
axes[0].bar(target_dist_full.index.astype(str), target_dist_full.values, color=["skyblue", "salmon"])
axes[0].set_title("Phân bố target trước khi subset")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Số lượng mẫu")
for i, v in enumerate(target_dist_full.values):
    axes[0].text(i, v + 100, str(v), ha="center")

# Biểu đồ sau subset
axes[1].bar(target_dist_sub.index.astype(str), target_dist_sub.values, color=["skyblue", "salmon"])
axes[1].set_title("Phân bố target sau khi subset")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Số lượng mẫu")
for i, v in enumerate(target_dist_sub.values):
    axes[1].text(i, v + 100, str(v), ha="center")

plt.tight_layout()
plt.show()



#Chia train 80% và temp 20%
train_df, temp_df = train_test_split(
    df_sub,
    test_size=0.20,
    random_state=SEED,
    stratify=df_sub["target"]
)

# Chia temp 20% thành val 10% và test_internal 10%
val_df, test_internal_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["target"]
)

#Kiểm tra kết quả chia
print("Final split sizes:")
print("Train:", len(train_df), "| dist:\n", train_df["target"].value_counts())
print("Val  :", len(val_df),   "| dist:\n", val_df["target"].value_counts())
print("Test :", len(test_internal_df), "| dist:\n", test_internal_df["target"].value_counts())



#Khai báo danh sách đặc trưng
TAB_CAT = ["sex", "anatom_site_general", "tbp_tile_type", "tbp_lv_location"]
TAB_NUM = [
    "age_approx",
    "clin_size_long_diam_mm",
    "tbp_lv_areaMM2",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_nevi_confidence",
]

#Chọn cột cần thiết cho train/test
use_cols_train = ["isic_id", "target"] + TAB_CAT + TAB_NUM
use_cols_test  = ["isic_id"] + TAB_CAT + TAB_NUM

train_df = train_df[use_cols_train].copy()
val_df   = val_df[use_cols_train].copy()
test_internal_df = test_internal_df[use_cols_train].copy()

comp_test_df = test_meta[use_cols_test].copy()

# Xử lý giá trị thiếu cho categorical
for c in TAB_CAT:
    train_df[c] = train_df[c].fillna("unknown").astype(str)
    val_df[c]   = val_df[c].fillna("unknown").astype(str)
    test_internal_df[c] = test_internal_df[c].fillna("unknown").astype(str)
    comp_test_df[c] = comp_test_df[c].fillna("unknown").astype(str)

# Xử lý giá trị thiếu cho numerical
for c in TAB_NUM:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    med = train_df[c].median()
    train_df[c] = train_df[c].fillna(med)

    val_df[c] = pd.to_numeric(val_df[c], errors="coerce").fillna(med)
    test_internal_df[c] = pd.to_numeric(test_internal_df[c], errors="coerce").fillna(med)
    comp_test_df[c] = pd.to_numeric(comp_test_df[c], errors="coerce").fillna(med)

#In thông báo hoàn tất
print("Tabular cleaned")



#Encode biến phân loại (LabelEncoder)
encoders = {}
for c in TAB_CAT:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[c], pd.Series(["unknown"])], axis=0))

    train_df[c] = le.transform(train_df[c])

    val_vals = val_df[c].where(val_df[c].isin(le.classes_), "unknown")
    val_df[c] = le.transform(val_vals)

    test_vals = test_internal_df[c].where(test_internal_df[c].isin(le.classes_), "unknown")
    test_internal_df[c] = le.transform(test_vals)

    comp_vals = comp_test_df[c].where(comp_test_df[c].isin(le.classes_), "unknown")
    comp_test_df[c] = le.transform(comp_vals)

    encoders[c] = le

#Chuẩn hóa dữ liệu số (StandardScaler)
scaler = StandardScaler()
X_train_tab = scaler.fit_transform(train_df[TAB_CAT + TAB_NUM].values.astype(np.float32))
X_val_tab   = scaler.transform(val_df[TAB_CAT + TAB_NUM].values.astype(np.float32))
X_test_int_tab = scaler.transform(test_internal_df[TAB_CAT + TAB_NUM].values.astype(np.float32))
X_comp_test_tab = scaler.transform(comp_test_df[TAB_CAT + TAB_NUM].values.astype(np.float32))

#Tách nhãn (target)
y_train = train_df["target"].values.astype(np.float32)
y_val   = val_df["target"].values.astype(np.float32)
y_test_int = test_internal_df["target"].values.astype(np.float32)

#Kiểm tra kết quả
print("Tab shapes:", X_train_tab.shape, X_val_tab.shape, X_test_int_tab.shape, X_comp_test_tab.shape)
print("Pos rates:", y_train.mean(), y_val.mean(), y_test_int.mean())



#Biến đổi ảnh cho tập train (có augmentation)
train_transforms = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

#Biến đổi ảnh cho tập validation/test (không augmentation)
eval_transforms = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])



#Mở file HDF5 chứa ảnh train
train_h5 = h5py.File(TRAIN_H5, "r")

#Lấy một sample ID từ train_df
sid = train_df["isic_id"].iloc[0]

#Đọc ảnh từ HDF5 và chuyển sang RGB
img = Image.open(io.BytesIO(train_h5[sid][()])).convert("RGB")

#Hiển thị ảnh bằng matplotlib
plt.figure(figsize=(4,4))
plt.imshow(img)
plt.axis("off")
plt.title(f"HDF5 OK: {sid} | y={int(train_df['target'].iloc[0])}")
plt.show()



#Định nghĩa lớp Dataset tùy chỉnh
class ISICDataset(Dataset):
    def __init__(self, df_meta, tab_array, y_array=None, h5_path=None, transforms=None):
        self.df = df_meta.reset_index(drop=True)
        self.tab = tab_array
        self.y = y_array
        self.h5_path = h5_path
        self.transforms = transforms
        self._h5 = None

#Hàm mở file HDF5
    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

#Hàm trả về độ dài dataset
    def __len__(self):
        return len(self.df)

#Hàm lấy một item (ảnh + tabular + nhãn)
    def __getitem__(self, idx):
        sid = self.df.loc[idx, "isic_id"]
        bs = self._get_h5()[sid][()]
        img = Image.open(io.BytesIO(bs)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        tab = torch.tensor(self.tab[idx], dtype=torch.float32)

        if self.y is None:
            return img, tab, sid

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, tab, y

#Tạo dataset cho từng tập
train_ds = ISICDataset(train_df, X_train_tab, y_train, TRAIN_H5, transforms=train_transforms)
val_ds   = ISICDataset(val_df,   X_val_tab,   y_val,   TRAIN_H5, transforms=eval_transforms)
test_int_ds = ISICDataset(test_internal_df, X_test_int_tab, y_test_int, TRAIN_H5, transforms=eval_transforms)

comp_test_ds = ISICDataset(comp_test_df, X_comp_test_tab, None, TEST_H5, transforms=eval_transforms)

# Xử lý imbalance bằng WeightedRandomSampler
y_int = y_train.astype(int)
class_count = np.bincount(y_int)  # [neg, pos]
class_weight = 1.0 / class_count
sample_weights = class_weight[y_int]

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True
)

#Tạo DataLoader cho từng tập
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_int_loader = DataLoader(test_int_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
comp_test_loader = DataLoader(comp_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

# Kiểm tra batch mẫu
img_b, tab_b, y_b = next(iter(train_loader))
print("Train batch:", img_b.shape, tab_b.shape, y_b.shape, "| batch pos rate:", y_b.mean().item())



# Tính pos_weight cho hàm mất mát
n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
print("n_pos:", n_pos, "n_neg:", n_neg, "pos_weight:", float(pos_weight.item()))

#Khởi tạo hàm mất mát BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#Định nghĩa mô hình EfficientNetB0 Fusion
class EfficientNetB0Fusion(nn.Module):
    def __init__(self, tab_dim, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

#Nhánh tabular (MLP)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )

#Head kết hợp ảnh + tabular
        self.head = nn.Sequential(
            nn.Linear(in_feats + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

#Forward pass
    def forward(self, img, tab):
        img_feat = self.backbone(img)
        tab_feat = self.tab_mlp(tab)
        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(x).squeeze(1)  # logits

#Khởi tạo mô hình + optimizer + scaler
model = EfficientNetB0Fusion(tab_dim=X_train_tab.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler()

print("Model ready")



#Hàm huấn luyện một epoch
def train_one_epoch():
    model.train()
    total_loss = 0.0
    for img, tab, y in tqdm(train_loader, leave=False):
        img = img.to(device, non_blocking=True)
        tab = tab.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(img, tab)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * img.size(0)

    return total_loss / len(train_loader.dataset)





#Hàm đánh giá (validation/test)
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    probs_all, y_all = [], []

    for img, tab, y in tqdm(loader, leave=False):
        img = img.to(device, non_blocking=True)
        tab = tab.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        with autocast():
            logits = model(img, tab)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

        total_loss += loss.item() * img.size(0)
        probs_all.append(probs.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)

    roc = roc_auc_score(y_all, probs_all)
    pr  = average_precision_score(y_all, probs_all)
    return total_loss / len(loader.dataset), roc, pr




#Vòng lặp huấn luyện nhiều epoch + lưu mô hình tốt nhất
best_roc = -1
best_path = "/kaggle/working/best_effb0_fusion.pt"

for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch()
    va_loss, va_roc, va_pr = evaluate(val_loader)

    print(f"Epoch {epoch:02d} | train_loss={tr_loss:.5f} | val_loss={va_loss:.5f} | ROC-AUC={va_roc:.5f} | PR-AUC={va_pr:.5f}")

    if va_roc > best_roc:
        best_roc = va_roc
        torch.save(model.state_dict(), best_path)
        print("Saved best:", best_path)




#Load lại mô hình tốt nhất
model.load_state_dict(torch.load(best_path, map_location=device))

#Đánh giá trên tập test_internal (10%)
test_loss, test_roc, test_pr = evaluate(test_int_loader)

#n kết quả
print("Internal TEST results:")
print("test_loss:", test_loss)
print("ROC-AUC  :", test_roc)
print("PR-AUC   :", test_pr)



#Đặt mô hình ở chế độ đánh giá
model.eval()

#Khởi tạo danh sách lưu kết quả
all_ids, all_probs = [], []

#Hàm dự đoán competition test
@torch.no_grad()
def predict_comp_test():
    for img, tab, sid in tqdm(comp_test_loader, leave=False):
        img = img.to(device, non_blocking=True)
        tab = tab.to(device, non_blocking=True)

        with autocast():
            logits = model(img, tab)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.extend(probs.tolist())
        all_ids.extend(list(sid))

#Gọi hàm dự đoán
predict_comp_test()

#Tạo DataFrame submission
submission = pd.DataFrame({"isic_id": all_ids, "target": all_probs})
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

#In thông báo & xem trước
print(".Saved:", out_path)
submission.head()






import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import numpy as np
import pandas as pd
import tensorflow as tf
import h5py

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Giảm lỗi VRAM / CUDA lặt vặt
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set memory growth:", e)

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))


DATA_DIR = "/kaggle/input/isic-2024-challenge"  # đổi nếu chạy local
META_PATH = f"{DATA_DIR}/train-metadata.csv"
HDF5_PATH = f"{DATA_DIR}/train-image.hdf5"

df = pd.read_csv(META_PATH, low_memory=False)

assert "isic_id" in df.columns and "target" in df.columns, "Thiếu cột isic_id hoặc target"

print("Total rows:", len(df))
print("Label counts:\n", df["target"].value_counts())


NEG_SAMPLE = 19650

pos_df = df[df["target"] == 1].copy()
neg_df = df[df["target"] == 0].sample(n=NEG_SAMPLE, random_state=SEED)

sample_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)

print("After sampling:\n", sample_df["target"].value_counts())


from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    sample_df,
    test_size=0.3,              # 70% train, 30% còn lại cho val+test
    random_state=SEED,
    stratify=sample_df["target"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,              # 15% val, 15% test
    random_state=SEED,
    stratify=temp_df["target"]
)

print("Train split:\n", train_df["target"].value_counts())
print("Val split:\n", val_df["target"].value_counts())
print("Test split:\n", test_df["target"].value_counts())


h5f = h5py.File(HDF5_PATH, "r")
print("HDF5 keys sample:", list(h5f.keys())[:5])

test_id = train_df.iloc[0]["isic_id"]
raw = h5f[test_id][()]

print("Test id:", test_id)
print("raw type:", type(raw))
print("raw dtype:", getattr(raw, "dtype", None))
print("raw shape:", getattr(raw, "shape", None))


# Loader chuẩn ISIC 2024
import matplotlib.pyplot as plt

def load_img_uint8(isic_id: str) -> np.ndarray:
    """
    Load ảnh từ HDF5 theo isic_id.
    Hỗ trợ:
      - ảnh array (H,W,3)/(H,W,4)
      - JPEG/PNG bytes (np.void/bytes/1D ndarray)
    Return: uint8 RGB (H,W,3)
    """
    data = h5f[isic_id][()]

    # Case 1: đã là ảnh array
    if isinstance(data, np.ndarray) and data.ndim == 3:
        if data.shape[-1] == 3:
            return data.astype(np.uint8)
        if data.shape[-1] == 4:
            return data[..., :3].astype(np.uint8)

    # Case 2: bytes -> decode
    if isinstance(data, np.void):
        img_bytes = data.tobytes()
    elif isinstance(data, (bytes, bytearray)):
        img_bytes = bytes(data)
    elif isinstance(data, np.ndarray):
        img_bytes = data.tobytes()
    else:
        img_bytes = bytes(data)

    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    return img.numpy().astype(np.uint8)

# Test loader 1 ảnh
img = load_img_uint8(test_id)
print("Loaded image:", img.shape, img.dtype)

plt.figure(figsize=(3,3))
plt.imshow(img)
plt.axis("off")
plt.show()


# Preprocess cho ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 224

def preprocess_resnet(img_uint8: np.ndarray) -> np.ndarray:
    img = tf.convert_to_tensor(img_uint8, dtype=tf.uint8)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img.numpy()


# Build ResNet50 (trích feature, không train)
from tensorflow.keras.applications import ResNet50

feat_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
feat_model.trainable = False

print("Feature output shape:", feat_model.output_shape)  # (None, 2048)


# Hàm extract_features theo batch
BATCH_SIZE = 64

def extract_features(df_in: pd.DataFrame, batch_size=64) -> np.ndarray:
    ids = df_in["isic_id"].astype(str).tolist()

    feats_list = []
    batch_imgs = []

    for i, isic_id in enumerate(ids):
        img = load_img_uint8(isic_id)
        img = preprocess_resnet(img)
        batch_imgs.append(img)

        # chạy khi đủ batch hoặc tới cuối
        if len(batch_imgs) >= batch_size or i == len(ids) - 1:
            x = np.stack(batch_imgs, axis=0)        # (B,224,224,3)
            f = feat_model.predict(x, verbose=0)    # (B,2048)
            feats_list.append(f)
            batch_imgs.clear()

        if (i + 1) % 2000 == 0:
            print(f"Extracted {i+1}/{len(ids)}")

    return np.concatenate(feats_list, axis=0)


# Test nhanh 128 ảnh
X_tmp = extract_features(train_df.iloc[:128], batch_size=64)
print("X_tmp shape:", X_tmp.shape)  # (128, 2048)


# Extract full features train/val
print("Extract train features...")
X_train = extract_features(train_df, batch_size=BATCH_SIZE)
y_train = train_df["target"].values.astype(int)

print("Extract val features...")
X_val = extract_features(val_df, batch_size=BATCH_SIZE)
y_val = val_df["target"].values.astype(int)

print("Extract test features...")   # --- THÊM ---
X_test = extract_features(test_df, batch_size=BATCH_SIZE)
y_test = test_df["target"].values.astype(int)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)


# 7. Scale features
# =========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)


# 8. SMOTE + undersampling (THÊM)
# =========================
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

over = SMOTE(random_state=SEED)
under = RandomUnderSampler(random_state=SEED)

pipeline = Pipeline([
    ('smote', over),
    ('under', under)
])

X_train_bal, y_train_bal = pipeline.fit_resample(X_train_s, y_train)

print("Before balancing:", np.bincount(y_train))
print("After balancing :", np.bincount(y_train_bal))


# Train SVM
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Train SVM
svm = SVC(
    kernel="linear",            # nếu chậm -> đổi "linear"
    C=2.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

print("Training SVM...")
svm.fit(X_train_bal, y_train_bal)



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

val_prob = svm.predict_proba(X_val_s)[:, 1]
test_prob = svm.predict_proba(X_test_s)[:, 1]

print("\n===== VAL SVM =====")
print("Accuracy :", accuracy_score(y_val, (val_prob>=0.3)))
print("Recall   :", recall_score(y_val, (val_prob>=0.3)))
print("F1-score :", f1_score(y_val, (val_prob>=0.3)))
print("AUC      :", roc_auc_score(y_val, val_prob))

print("\n===== TEST SVM =====")
print("Accuracy :", accuracy_score(y_test, (test_prob>=0.3)))
print("Recall   :", recall_score(y_test, (test_prob>=0.3)))
print("F1-score :", f1_score(y_test, (test_prob>=0.3)))
print("AUC      :", roc_auc_score(y_test, test_prob))



# 9. ROC Curve cho VAL

from sklearn.metrics import roc_curve, roc_auc_score

fpr_val, tpr_val, _ = roc_curve(y_val, val_prob)
auc_val = roc_auc_score(y_val, val_prob)

plt.figure(figsize=(7,6))
plt.plot(fpr_val, tpr_val, label=f"Val ROC (AUC = {auc_val:.4f})")
plt.plot([0,1],[0,1],"--",label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validation Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# 10. ROC Curve cho TEST  

fpr_test, tpr_test, _ = roc_curve(y_test, test_prob)
auc_test = roc_auc_score(y_test, test_prob)

plt.figure(figsize=(7,6))
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {auc_test:.4f})", color="red")
plt.plot([0,1],[0,1],"--",label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("Test AUC:", auc_test)



# So sánh metrics khi đổi threshold (0.3 / 0.4 / 0.5)
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

thresholds = [0.3, 0.4, 0.5]
# --- VAL ---
val_results = []
for th in thresholds:
    val_pred_th = (val_prob >= th).astype(int)
    val_results.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_val, val_pred_th),
        "Precision": precision_score(y_val, val_pred_th, zero_division=0),
        "Recall": recall_score(y_val, val_pred_th, zero_division=0),
        "F1-score": f1_score(y_val, val_pred_th, zero_division=0)
    })
val_results_df = pd.DataFrame(val_results)
print("\n===== VAL Threshold Comparison =====")
print(val_results_df)

# --- TEST ---
test_results = []
for th in thresholds:
    test_pred_th = (test_prob >= th).astype(int)
    test_results.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_test, test_pred_th),
        "Precision": precision_score(y_test, test_pred_th, zero_division=0),
        "Recall": recall_score(y_test, test_pred_th, zero_division=0),
        "F1-score": f1_score(y_test, test_pred_th, zero_division=0)
    })
test_results_df = pd.DataFrame(test_results)
print("\n===== TEST Threshold Comparison =====")
print(test_results_df)



from sklearn.feature_selection import SelectKBest, f_classif

K = 50  # thử 10, 20, 50, 100 (10 thường hơi ít)
selector = SelectKBest(score_func=f_classif, k=K)

X_train_k = selector.fit_transform(X_train_s, y_train)
X_val_k   = selector.transform(X_val_s)
X_test_k = selector.transform(X_test_s)

top_idx = selector.get_support(indices=True)
print("Selected K =", K)
print("Top feature indices:", top_idx[:20], "...")
print("Shapes:", X_train_k.shape, X_val_k.shape, X_test_k.shape)


for th in [0.05, 0.1, 0.2, 0.3]:
    pred = (val_prob >= th).astype(int)
    print(f"\nThreshold = {th}")
    print("Predicted positives:", np.sum(pred == 1))
    print("Recall:", recall_score(y_val, pred, zero_division=0))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print("Scaled shapes:", X_train_s.shape, X_val_s.shape, X_test_s.shape)


from sklearn.linear_model import LogisticRegression

print("Training Logistic Regression...")

lr = LogisticRegression(
    max_iter=3000,          # đủ lớn cho hội tụ
    class_weight="balanced",# xử lý mất cân bằng
    solver="liblinear"      # ổn định cho binary classification
)

lr.fit(X_train_bal, y_train_bal)


val_prob_lr = lr.predict_proba(X_val_s)[:, 1]
print("val_prob range:", val_prob_lr.min(), "→", val_prob_lr.max())



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

th = 0.3
val_pred_lr = (val_prob_lr >= th).astype(int)

print("\n===== VAL Logistic Regression =====")
print("Threshold:", th)
print("Accuracy :", accuracy_score(y_val, val_pred_lr))
print("Precision:", precision_score(y_val, val_pred_lr, zero_division=0))
print("Recall   :", recall_score(y_val, val_pred_lr, zero_division=0))
print("F1-score :", f1_score(y_val, val_pred_lr, zero_division=0))
print("AUC      :", roc_auc_score(y_val, val_prob_lr))

# --- TEST 
test_prob_lr = lr.predict_proba(X_test_s)[:, 1]
test_pred_lr = (test_prob_lr >= th).astype(int)

print("\n===== TEST Logistic Regression =====")
print("Accuracy :", accuracy_score(y_test, test_pred_lr))
print("Precision:", precision_score(y_test, test_pred_lr, zero_division=0))
print("Recall   :", recall_score(y_test, test_pred_lr, zero_division=0))
print("F1-score :", f1_score(y_test, test_pred_lr, zero_division=0))
print("AUC      :", roc_auc_score(y_test, test_prob_lr))


import pandas as pd
import numpy as np

thresholds = np.round(np.linspace(0.01, 0.5, 50), 3)
rows = []

for th in thresholds:
    pred = (val_prob_lr >= th).astype(int)
    rows.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_val, pred),
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
        "Pred_Pos": int((pred == 1).sum())
    })

df_lr = pd.DataFrame(rows)
df_lr.sort_values("Recall", ascending=False).head(10)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# ROC Curve cho Logistic Regression
fpr_val, tpr_val, _ = roc_curve(y_val, val_prob_lr)
auc_val = roc_auc_score(y_val, val_prob_lr)

fpr_test, tpr_test, _ = roc_curve(y_test, test_prob_lr)
auc_test = roc_auc_score(y_test, test_prob_lr)

plt.figure(figsize=(7,6))
plt.plot(fpr_val, tpr_val, label=f"Val ROC (AUC={auc_val:.4f})")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC={auc_test:.4f})", color="red")
plt.plot([0,1],[0,1],"--",label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (ResNet50 features)")
plt.legend()
plt.grid(True)
plt.show()


# Train Random Forest
from sklearn.ensemble import RandomForestClassifier

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,          # số cây
    max_depth=None,            # để cây tự học
    min_samples_leaf=5,        # tránh overfit
    class_weight="balanced",   # CỰC KỲ quan trọng
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_bal, y_train_bal)


val_prob_rf = rf.predict_proba(X_val)[:, 1]
test_prob_rf = rf.predict_proba(X_test)[:, 1]

print("val_prob range:", val_prob.min(), "→", val_prob.max())


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

thresholds = [0.01, 0.03, 0.05, 0.1]
rows = []

for th in thresholds:
    val_pred_rf = (val_prob_rf >= th).astype(int)
    rows.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_val, val_pred_rf),
        "Precision": precision_score(y_val, val_pred_rf, zero_division=0),
        "Recall": recall_score(y_val, val_pred_rf, zero_division=0),
        "F1": f1_score(y_val, val_pred_rf, zero_division=0),
        "Pred_Pos": int((val_pred_rf == 1).sum())
    })

df_rf = pd.DataFrame(rows)
print("\n===== VAl Random Forest =====")
print(df_rf)

# --- TEST ---
test_prob_rf = rf.predict_proba(X_test)[:, 1]
test_pred_rf = (test_prob_rf >= 0.1).astype(int)   # threshold ví dụ

print("\n===== TEST Random Forest =====")
print("Accuracy :", accuracy_score(y_test, test_pred_rf))
print("Precision:", precision_score(y_test, test_pred_rf, zero_division=0))
print("Recall   :", recall_score(y_test, test_pred_rf, zero_division=0))
print("F1-score :", f1_score(y_test, test_pred_rf, zero_division=0))
print("AUC      :", roc_auc_score(y_test, test_prob_rf))



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- VAL Random Forest ---
th = 0.3   # em có thể đổi 0.1–0.5 để thử
val_pred_rf = (val_prob_rf >= th).astype(int)

print("\n===== VAL Random Forest =====")
print("Threshold:", th)
print("Accuracy :", accuracy_score(y_val, val_pred_rf))
print("Precision:", precision_score(y_val, val_pred_rf, zero_division=0))
print("Recall   :", recall_score(y_val, val_pred_rf, zero_division=0))
print("F1-score :", f1_score(y_val, val_pred_rf, zero_division=0))
print("AUC      :", roc_auc_score(y_val, val_prob_rf))

# --- TEST Random Forest ---
test_pred_rf = (test_prob_rf >= th).astype(int)

print("\n===== TEST Random Forest =====")
print("Threshold:", th)
print("Accuracy :", accuracy_score(y_test, test_pred_rf))
print("Precision:", precision_score(y_test, test_pred_rf, zero_division=0))
print("Recall   :", recall_score(y_test, test_pred_rf, zero_division=0))
print("F1-score :", f1_score(y_test, test_pred_rf, zero_division=0))
print("AUC      :", roc_auc_score(y_test, test_prob_rf))



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# --- ROC Curve cho VAL ---
fpr_val, tpr_val, _ = roc_curve(y_val, val_prob_rf)
auc_val = roc_auc_score(y_val, val_prob_rf)

# --- ROC Curve cho TEST ---
fpr_test, tpr_test, _ = roc_curve(y_test, test_prob_rf)
auc_test = roc_auc_score(y_test, test_prob_rf)

plt.figure(figsize=(7,6))
plt.plot(fpr_val, tpr_val, label=f"Val ROC (AUC={auc_val:.4f})")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC={auc_test:.4f})", color="red")
plt.plot([0,1],[0,1],"--",label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



# =========================
# 1. Setup & Config
# =========================

# =========================
# 2. Load metadata & oversample
# =========================
META_PATH = "/kaggle/input/isic-2024-challenge/train-metadata.csv"
df = pd.read_csv(META_PATH, low_memory=False)[["isic_id","target"]].copy()

# Oversample class 1 trước khi split
pos_df = df[df["target"] == 1].copy()
neg_df = df[df["target"] == 0].sample(n=19650, random_state=SEED)

# Nhân class 1 lên 3 lần
pos_df = pd.concat([pos_df]*3, ignore_index=True)

# Gộp lại và shuffle
sample_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)
print("Label counts sau oversample:", sample_df["target"].value_counts())

# Split train/val/test
train_df, temp_df = train_test_split(sample_df, test_size=0.3, random_state=SEED, stratify=sample_df["target"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["target"])

# =========================
# 3. Dataset class
# =========================
IMG_SIZE = 224
DATA_DIR = "/kaggle/input/isic-2024-challenge/train-image"

class ISICDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        isic_id = self.df.loc[idx, "isic_id"]
        label = self.df.loc[idx, "target"]

        # Load ảnh từ file .jpg
        img_path = f"{DATA_DIR}/{isic_id}.jpg"
        img = io.read_image(img_path).float() / 255.0  # CHW format

        if self.transform:
            img = self.transform(img)

        return img, label

# Transform
transform_train = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip()
])
transform_eval = T.Resize((IMG_SIZE, IMG_SIZE))

train_ds = ISICDataset(train_df, transform=transform_train)
val_ds   = ISICDataset(val_df, transform=transform_eval)
test_ds  = ISICDataset(test_df, transform=transform_eval)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

# =========================
# 4. CNN Model
# =========================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # output logits
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = CNNModel().to(device)

# =========================
# 5. Loss xử lý imbalance
# =========================
neg = int((train_df["target"] == 0).sum())
pos = int((train_df["target"] == 1).sum())
pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

print("Train neg/pos:", neg, pos, "| pos_weight:", pos_weight.item())

# =========================


# =========================
# 7. Training
# =========================
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    val_prob, val_true = eval_epoch(val_loader)
    val_pred = (val_prob >= 0.5).astype(int)
    val_acc = accuracy_score(val_true, val_pred)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

# =========================
# 8. Evaluation function
# =========================
def evaluate_model(prob, true, threshold=0.5, split="VAL"):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    auc = roc_auc_score(true, prob)

    print(f"\n===== {split} CNN @ Threshold={threshold} =====")
    print("Accuracy :", accuracy_score(true, pred))
    print("Precision:", precision_score(true, pred, zero_division=0))
    print("Recall   :", recall_score(true, pred, zero_division=0))
    print("F1-score :", f1_score(true, pred, zero_division=0))
    print("AUC      :", auc)
    print("Confusion Matrix:\n", confusion_matrix(true, pred))
    print(classification_report(true, pred, digits=4, zero_division=0))

# =========================
# 9. Final evaluation
# =========================
val_prob, val_true = eval_epoch(val_loader)
test_prob, test_true = eval_epoch(test_loader)

evaluate_model(val_prob, val_true, threshold=0.5, split="VAL")
evaluate_model(test_prob, test_true, threshold=0.5, split="TEST")



# 1. Setup & Config
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)



# 2. Load metadata & oversample
# =========================
META_PATH = "/kaggle/input/isic-2024-challenge/train-metadata.csv"
df = pd.read_csv(META_PATH, low_memory=False)[["isic_id","target"]].copy()

# Oversample class 1 trước khi split
pos_df = df[df["target"] == 1].copy()
neg_df = df[df["target"] == 0].sample(n=19650, random_state=SEED)

# Nhân class 1 lên 3 lần
pos_df = pd.concat([pos_df]*3, ignore_index=True)

# Gộp lại và shuffle
sample_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)
print("Label counts sau oversample:", sample_df["target"].value_counts())

# Split train/val/test
train_df, temp_df = train_test_split(sample_df, test_size=0.3, random_state=SEED, stratify=sample_df["target"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["target"])


# 3. Dataset class
# =========================
IMG_SIZE = 224
DATA_DIR = "/kaggle/input/isic-2024-challenge/train-image"

class ISICDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        isic_id = self.df.loc[idx, "isic_id"]
        label = self.df.loc[idx, "target"]

        # Load ảnh từ file .jpg
        img_path = f"{DATA_DIR}/{isic_id}.jpg"
        img = io.read_image(img_path).float() / 255.0  # CHW format

        if self.transform:
            img = self.transform(img)

        return img, label

# Transform
transform_train = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip()
])
transform_eval = T.Resize((IMG_SIZE, IMG_SIZE))

train_ds = ISICDataset(train_df, transform=transform_train)
val_ds   = ISICDataset(val_df, transform=transform_eval)
test_ds  = ISICDataset(test_df, transform=transform_eval)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)


# 4. CNN Model
# =========================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # output logits
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = CNNModel().to(device)


# 5. Loss xử lý imbalance
# =========================
neg = int((train_df["target"] == 0).sum())
pos = int((train_df["target"] == 1).sum())
pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
print("Train neg/pos:", neg, pos, "| pos_weight:", pos_weight.item())



# 6. Train & Eval loop
# =========================
def train_epoch(loader):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(loader):
    model.eval()
    all_prob, all_true = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            logits = model(images)
            prob = torch.sigmoid(logits)
            all_prob.append(prob.cpu().numpy())
            all_true.append(labels.cpu().numpy())
    return np.vstack(all_prob).ravel(), np.vstack(all_true).ravel()


# 7. Training
# =========================
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    val_prob, val_true = eval_epoch(val_loader)
    val_pred = (val_prob >= 0.5).astype(int)
    val_acc = accuracy_score(val_true, val_pred)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")


# 7. Evaluate CNN
# =========================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import numpy as np

# THÊM: lấy y_true trực tiếp từ dataset để khớp với predict
val_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
test_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

# predict probability
val_prob = cnn.predict(val_ds).ravel()
test_prob = cnn.predict(test_ds).ravel()

# predict label với threshold 0.3
val_pred = (val_prob >= 0.3).astype(int)
test_pred = (test_prob >= 0.3).astype(int)

print("\n===== VAL CNN =====")
print("Accuracy :", accuracy_score(val_true, val_pred))
print("Precision:", precision_score(val_true, val_pred, zero_division=0))
print("Recall   :", recall_score(val_true, val_pred, zero_division=0))
print("F1-score :", f1_score(val_true, val_pred, zero_division=0))
print("AUC      :", roc_auc_score(val_true, val_prob))

# THÊM: in confusion matrix + classification report
print("Confusion Matrix:\n", confusion_matrix(val_true, val_pred))
print(classification_report(val_true, val_pred, digits=4, zero_division=0))

print("\n===== TEST CNN =====")
print("Accuracy :", accuracy_score(test_true, test_pred))
print("Precision:", precision_score(test_true, test_pred, zero_division=0))
print("Recall   :", recall_score(test_true, test_pred, zero_division=0))
print("F1-score :", f1_score(test_true, test_pred, zero_division=0))
print("AUC      :", roc_auc_score(test_true, test_prob))

# THÊM: in confusion matrix + classification report
print("Confusion Matrix:\n", confusion_matrix(test_true, test_pred))
print(classification_report(test_true, test_pred, digits=4, zero_division=0))



from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n=== EVAL @ Threshold = {threshold:.2f} ===")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion (tn, fp, fn, tp): ({tn}, {fp}, {fn}, {tp})")

    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# Gọi hàm cho val và test
evaluate_model(val_df["target"].values, val_prob, threshold=0.3)
evaluate_model(test_df["target"].values, test_prob, threshold=0.3)

# Thử thêm threshold khác
evaluate_model(val_df["target"].values, val_prob, threshold=0.1)
evaluate_model(test_df["target"].values, test_prob, threshold=0.1)



print("Train label:", np.bincount([y.numpy() for _, y in train_ds.unbatch()]))



plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss over epochs")
plt.show()



plt.hist(val_prob, bins=50, color="skyblue")
plt.title("Distribution of predicted probabilities (val)")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.show()



# 8. ROC Curve CNN
# =========================
fpr_val, tpr_val, _ = roc_curve(val_true, val_prob)
fpr_test, tpr_test, _ = roc_curve(test_true, test_prob)

plt.figure(figsize=(7,6))
plt.plot(fpr_val, tpr_val, label=f"Val ROC (AUC={roc_auc_score(val_true,val_prob):.4f})")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC={roc_auc_score(test_true,test_prob):.4f})", color="red")
plt.plot([0,1],[0,1],"--",label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CNN (tf.data.Dataset)")
plt.legend()
plt.grid(True)
plt.show()

