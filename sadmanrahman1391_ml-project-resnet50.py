# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip -q install -U albumentations==1.4.7 timm==0.9.16 segmentation-models-pytorch==0.3.3 torchmetrics==1.4.0 opencv-python-headless==4.10.0.84 tifffile==2024.8.30



DATA = Path("/kaggle/input/blood-vessel-segmentation")  # change if local
TRAIN = DATA/"train"
TEST  = DATA/"test"
RLE_CSV = DATA/"train_rles.csv"  # optional for sparse labels
OUTDIR = Path("./outputs"); OUTDIR.mkdir(exist_ok=True)

CFG = dict(
    image_size=512,            # resize short or pad — we’ll letterbox
    batch_size=4,
    epochs=20,
    lr=1e-3,
    weight_decay=1e-5,
    encoder="resnet34",        # try "resnet50" if VRAM allows
    encoder_weights="imagenet",
    num_workers=2,
    pos_threshold=0.5,         # any-vessel threshold on mean prob
    mask_threshold=0.35,       # binarization threshold
    tta=True,                  # horizontal/vertical flips at test
    seed=42,
    in_channels=1,             # TIFFs are grayscale
)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])



# --- RLE encoding/decoding for Kaggle ---
def rle_encode(mask):
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def rle_decode(rle, shape):
    s = np.asarray([int(x) for x in rle.split()], dtype=int)
    starts, lengths = s[0::2]-1, s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends): img[lo:hi] = 1
    return img.reshape(shape, order="F")

# --- map id -> rle if CSV exists (for sparse sets) ---
rle_map = {}
if RLE_CSV.exists():
    df_rle = pd.read_csv(RLE_CSV)
    rle_map = dict(zip(df_rle.id.values, df_rle.rle.fillna("")))



def read_tiff(path):
    img = cv2.imread(str(path), -1)      # preserves 16-bit
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:                    # safety: keep one channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def find_pairs(train_root):
    items = []
    for ds in sorted(os.listdir(train_root)):
        dpath = train_root/ds
        if not (dpath/"images").exists(): continue
        for ipath in sorted(glob(str(dpath/"images"/*.tif))):
            fname = Path(ipath).name
            slice_id = f"{ds}_{Path(fname).stem}"
            # Preferred: label .tif if present
            lpath = dpath/"labels"/fname
            if lpath.exists():
                items.append((ipath, str(lpath), slice_id))
            else:
                # fallback to RLE
                items.append((ipath, None, slice_id))
    return items

pairs = find_pairs(TRAIN)
print("Total train slices:", len(pairs))

class KidneyDataset(Dataset):
    def __init__(self, items, augment=True):
        self.items = items
        size = CFG["image_size"]
        if augment:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Affine(scale=(0.9,1.1), rotate=(-10,10), shear=(-8,8), p=0.5),
                A.ElasticTransform(p=0.2, alpha=50, sigma=7, alpha_affine=10),
                A.RandomBrightnessContrast(p=0.35),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.GaussianBlur(blur_limit=(3,5), p=0.2),
                A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.2),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ])
    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        ipath, lpath, slice_id = self.items[i]
        img = read_tiff(ipath)
        if lpath is not None:
            mask = read_tiff(lpath)
            mask = (mask>0).astype(np.uint8)
        else:
            rle = rle_map.get(slice_id, "")
            if rle=="":
                mask = np.zeros_like(img, dtype=np.uint8)
            else:
                mask = rle_decode(rle, img.shape)
        # albumentations expects HWC
        aug = self.tf(image=img[...,None], mask=mask)
        x = aug["image"].float().permute(2,0,1)  # (1,H,W)
        y = aug["mask"].float().unsqueeze(0)     # (1,H,W)
        return x, y, slice_id



# simple random split (you can stratify by dataset if desired)
random.shuffle(pairs)
split = int(0.9*len(pairs))
train_ds = KidneyDataset(pairs[:split], augment=True)
valid_ds = KidneyDataset(pairs[split:], augment=False)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

# U-Net with ResNet encoder
model = smp.Unet(
    encoder_name=CFG["encoder"],
    encoder_weights=CFG["encoder_weights"],
    in_channels=CFG["in_channels"],
    classes=1
)

# add an "any-vessel" classifier head from bottleneck features
class AnyVesselHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, 1)
    def forward(self, feats):
        # feats: encoder last feature map
        x = self.pool(feats).flatten(1)
        return self.fc(x)

# wrap model to expose encoder features
class UNetWithAux(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        # infer channels of last encoder stage
        in_ch = self.base.encoder.out_channels[-1]
        self.aux = AnyVesselHead(in_ch)
    def forward(self, x):
        features = self.base.encoder(x)
        decoder_output = self.base.decoder(*features)
        masks = self.base.segmentation_head(decoder_output)
        logits_any = self.aux(features[-1])
        return masks, logits_any

net = UNetWithAux(model).cuda()

bce = nn.BCEWithLogitsLoss()
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2*(pred*target).sum(dim=(2,3)) + eps
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return 1 - (num/den).mean()

def loss_fn(mask_logits, mask_true, any_logits):
    loss_mask = 0.5*bce(mask_logits, mask_true) + 0.5*dice_loss(mask_logits, mask_true)
    any_true = (mask_true.sum(dim=(2,3))>0).float()
    loss_any = bce(any_logits, any_true)
    return loss_mask + 0.2*loss_any, loss_mask.detach(), loss_any.detach()

optimizer = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scaler = GradScaler()



def evaluate():
    net.eval()
    dices = []
    with torch.no_grad():
        for x, y, _ in valid_loader:
            x, y = x.cuda(), y.cuda()
            m, _ = net(x)
            p = (torch.sigmoid(m) > CFG["mask_threshold"]).float()
            # torchmetrics dice expects probs/labels; compute per-batch
            for i in range(p.size(0)):
                dices.append(tm_dice(p[i,0], y[i,0]).item())
    return float(np.mean(dices))

best_dice = 0.0
for epoch in range(1, CFG["epochs"]+1):
    net.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    loss_sum = 0
    for x, y, _ in pbar:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            mask_logits, any_logits = net(x)
            loss, lm, la = loss_fn(mask_logits, y, any_logits)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
        pbar.set_postfix(loss=f"{loss_sum/ (pbar.n or 1):.4f}")
    val_dice = evaluate()
    print(f"val dice: {val_dice:.4f}")
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(net.state_dict(), OUTDIR/"best.pt")
        print("Saved best.")



# collect test slice paths
def list_test_slices(test_root):
    items=[]
    for ds in sorted(os.listdir(test_root)):
        dpath = test_root/ds
        for ipath in sorted(glob(str(dpath/"images"/*.tif))):
            slice_id = f"{ds}_{Path(ipath).stem}"
            items.append((ipath, slice_id))
    return items

test_items = list_test_slices(TEST)

# simple TTA: flips
def predict_mask(img_t):
    img_t = img_t.cuda()
    with torch.no_grad(), autocast():
        m, anylog = net(img_t)
        prob = torch.sigmoid(m)
        anyprob = torch.sigmoid(anylog).squeeze(1)
        if CFG["tta"]:
            # hflip
            m2, a2 = net(torch.flip(img_t, dims=[-1]))
            m2 = torch.flip(m2, dims=[-1])
            # vflip
            m3, a3 = net(torch.flip(img_t, dims=[-2]))
            m3 = torch.flip(m3, dims=[-2])
            prob = (prob + torch.sigmoid(m2) + torch.sigmoid(m3))/3.0
            anyprob = (anyprob + torch.sigmoid(a2).squeeze(1) + torch.sigmoid(a3).squeeze(1))/3.0
    return prob, anyprob

# load best weights
net.load_state_dict(torch.load(OUTDIR/"best.pt", map_location="cuda"))
net.eval()

ids, rles, any_flags, any_probs = [], [], [], []
size = CFG["image_size"]

for batch_start in range(0, len(test_items), CFG["batch_size"]):
    batch = test_items[batch_start: batch_start+CFG["batch_size"]]
    imgs, metas = [], []
    for ipath, sid in batch:
        raw = read_tiff(ipath)
        H, W = raw.shape
        tf = A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
        out = tf(image=raw[...,None])
        x = out["image"].float().permute(2,0,1)    # (1,h,w)
        imgs.append(x)
        metas.append((sid, (H,W), out))            # store to unpad/resize back
    imgs = torch.stack(imgs, dim=0)

    prob, anyprob = predict_mask(imgs)             # (B,1,h,w), (B,)
    prob = prob[:,0].cpu().numpy()
    anyprob = anyprob.cpu().numpy()

    for i,(sid,(H,W),out) in enumerate(metas):
        # invert pad/resize to original shape
        # since we used letterbox via LongestMaxSize + PadIfNeeded,
        # simply resize back and crop center if needed
        p = (prob[i]*255).astype(np.uint8)
        p = cv2.resize(p, (out["image"].shape[1], out["image"].shape[0]), interpolation=cv2.INTER_LINEAR)
        # remove pad to (H,W)
        p = p[:out["image"].shape[1], :out["image"].shape[2]] if False else p  # no-op (kept square)
        p = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
        m = (p/255.0 > CFG["mask_threshold"]).astype(np.uint8)

        ids.append(sid)
        rles.append(rle_encode(m))
        any_probs.append(float(anyprob[i]))
        any_flags.append(int(anyprob[i] >= CFG["pos_threshold"]))

sub = pd.DataFrame({"id": ids, "rle": rles})
sub.to_csv("submission.csv", index=False)

aux = pd.DataFrame({"id": ids, "has_vessel": any_flags, "prob_any_vessel": any_probs})
aux.to_csv("slice_posneg.csv", index=False)

print("Wrote:", os.path.abspath("submission.csv"))
print("Wrote:", os.path.abspath("slice_posneg.csv"))








