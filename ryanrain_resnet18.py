# ==============================================
# Mayo-Clinic-STRIP-AI | ResNet-18 embedding → SVM baseline
# ==============================================
!pip install -q timm tqdm

from pathlib import Path
from collections import defaultdict
import random, warnings, gc
import numpy as np, pandas as pd
import cv2, tifffile, torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A; import albumentations.pytorch
import timm
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ------------------ CONFIG ------------------
class CFG:
    TILE_SIZE  = 448        # 小一点，加速 & 省显存
    BATCH      = 16
    NUM_WK     = 0         # worker 数
    SEED       = 42
    DEBUG      = True
    DEBUG_FRAC = 0.5        # ← 0.5 × 数据
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

DATA = Path("/kaggle/input/mayo-clinic-strip-ai")
train_csv = pd.read_csv(DATA/"train.csv")
test_csv  = pd.read_csv(DATA/"test.csv")

# ---------- DEBUG 抽样 ----------
if CFG.DEBUG:
    pats = train_csv.patient_id.unique()
    sel  = np.random.RandomState(CFG.SEED).choice(
        pats, size=int(len(pats)*CFG.DEBUG_FRAC), replace=False)
    train_csv = train_csv[train_csv.patient_id.isin(sel)]
    print(f"[DEBUG] {len(train_csv)} tiles  ({train_csv.patient_id.nunique()} patients)")

label_map = {l:i for i,l in enumerate(sorted(train_csv.label.unique()))}
train_csv["label_id"] = train_csv.label.map(label_map)

# simple 80 / 20 patient split
val_pat = (train_csv.groupby("patient_id").first()
           .sample(frac=0.2, random_state=CFG.SEED).index)
train_df = train_csv[~train_csv.patient_id.isin(val_pat)].reset_index(drop=True)
val_df   = train_csv[ train_csv.patient_id.isin(val_pat)].reset_index(drop=True)

# ---------- Dataset ----------
tfm = A.Compose([
    A.Resize(CFG.TILE_SIZE, CFG.TILE_SIZE),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    A.pytorch.ToTensorV2(),
])

class TileDS(Dataset):
    def __init__(self, df, split):
        self.df, self.split = df, split
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = "train" if "label_id" in row else "test"
        img = tifffile.imread(DATA/folder/f"{row.image_id}.tif")
        if img.ndim==2: img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = tfm(image=img)["image"]
        y   = row.label_id if "label_id" in row else -1
        return img.float(), y, row.patient_id

def loader(df, split):
    return DataLoader(TileDS(df, split),
                      batch_size=CFG.BATCH, shuffle=False,
                      num_workers=CFG.NUM_WK)

# ---------- ResNet-18 embedder ----------
fe = timm.create_model("resnet18", pretrained=True,
                       num_classes=0, global_pool="avg").to(CFG.DEVICE).eval()

@torch.no_grad()
def get_embed(dl):
    feat, y, pid = [], [], []
    for x, lbl, p in tqdm(dl, leave=False):
        f = fe(x.to(CFG.DEVICE,non_blocking=True)).cpu().numpy()
        feat.append(f); y.extend(lbl.numpy()); pid.extend(p)
    return np.vstack(feat), np.array(y), np.array(pid)

def pool_mean(feats, labels, pids):
    bag = defaultdict(list)
    for f,l,p in zip(feats,labels,pids): bag[p].append((f,l))
    X, y, ids = [], [], []
    for p,lst in bag.items():
        vec,_y = zip(*lst)
        X.append(np.mean(vec,0)); y.append(_y[0]); ids.append(p)
    return np.vstack(X), np.array(y), np.array(ids)

# ---------- extract embeddings ----------
print("⏳ embeddings …")
X_tr,y_tr,_ = pool_mean(*get_embed(loader(train_df,"train")))
X_va,y_va,_ = pool_mean(*get_embed(loader(val_df,"train")))

# ---------- PCA 128 → 线性 SVM ----------
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("pca",    PCA(n_components=128, whiten=True, random_state=CFG.SEED)),
    ("svm",    SVC(kernel="linear", probability=True,
                   class_weight="balanced", C=1.0, random_state=CFG.SEED)),
])

pipe.fit(X_tr, y_tr)
pred_va = pipe.predict_proba(X_va)[:,1]
print(f"Val AUC = {roc_auc_score(y_va, pred_va):.3f}")

# ---------- test ----------
print("⏳ test embedding …")
X_test, _, pid_test = pool_mean(*get_embed(loader(test_csv,"test")))
prob_laa = pipe.predict_proba(X_test)[:,1]

bag = defaultdict(list)
for p,pr in zip(pid_test, prob_laa): bag[p].append(pr)

sub = pd.DataFrame({
    "patient_id": list(bag.keys()),
    "LAA": [np.mean(v) for v in bag.values()]
})
sub["CE"] = 1 - sub["LAA"]
sub[["CE","LAA"]] = sub[["CE","LAA"]].clip(1e-15,1-1e-15)
sub = sub[["patient_id","CE","LAA"]]
sub.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!", sub.head())

