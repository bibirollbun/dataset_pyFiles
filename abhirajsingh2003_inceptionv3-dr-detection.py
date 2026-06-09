# ==================== CELL 1: IMPORTS & SETUP ====================
import os, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# misc
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# circular crop helper
def circular_crop(img):
    """Crop to the largest bright contour (retina). Input: BGR image from cv2."""
    if img is None:
        return None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = img[y:y+h, x:x+w]
    return cropped


# ==================== CELL 2: PATHS & DATAFRAME ====================
DATA_ROOT = Path('/kaggle/input/aptos2019-blindness-detection')
assert DATA_ROOT.exists(), f"Dataset not found at {DATA_ROOT}"

TRAIN_CSV = DATA_ROOT / 'train.csv'
TRAIN_DIR = DATA_ROOT / 'train_images'

train_df = pd.read_csv(TRAIN_CSV)
train_df['path'] = train_df['id_code'].map(lambda x: str(TRAIN_DIR / f"{x}.png"))
train_df['diagnosis'] = train_df['diagnosis'].astype(int)
print("Total samples:", len(train_df))
train_df.head()


# ==================== CELL 3: QUICK EDA ====================
print("Class counts:\n", train_df['diagnosis'].value_counts())
plt.figure(figsize=(6,4))
train_df['diagnosis'].value_counts().sort_index().plot(kind='bar')
plt.title('Class distribution')
plt.show()


# ==================== CELL 4: TRANSFORMS (PRE-RESIZE) ====================
import albumentations as A
from albumentations.pytorch import ToTensorV2

# InceptionV3 expects 299×299 input images
IMG_SIZE = 299
BATCH_SIZE = 8   # reduce to 4 if GPU runs out of memory

# --- TRAINING AUGMENTATIONS ---
# Using RandomCrop (stable across albumentations versions)
train_transforms = A.Compose([
    A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5)
    ], p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- VALIDATION AUGMENTATIONS ---
valid_transforms = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

print("✅ Transforms ready.")
print(f"IMG_SIZE = {IMG_SIZE},  BATCH_SIZE = {BATCH_SIZE}")



# ==================== CELL 5: ROBUST DATASET (PRE-RESIZE) ====================
class APTOSDataset(Dataset):
    def __init__(self, df, transforms=None, img_size=IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.df)

    def _safe_resize_and_to_tensor(self, img):
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_resized = img_resized.astype(np.float32) / 255.0
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        std  = np.array((0.229, 0.224, 0.225), dtype=np.float32)
        img_resized = (img_resized - mean) / std
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        return img_tensor

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        path = row['path']
        label = int(row['diagnosis'])

        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = circular_crop(img)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pre-resize to fixed size BEFORE passing to Albumentations (helps stability)
        try:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        except Exception:
            return self._safe_resize_and_to_tensor(img), label

        if self.transforms:
            try:
                sample = self.transforms(image=img)
                img_out = sample['image']
                if isinstance(img_out, np.ndarray):
                    img_tensor = torch.from_numpy(img_out.astype(np.float32)).permute(2,0,1)
                elif torch.is_tensor(img_out):
                    img_tensor = img_out
                else:
                    img_tensor = self._safe_resize_and_to_tensor(img)
                if img_tensor.dim() != 3 or img_tensor.shape[1] != self.img_size or img_tensor.shape[2] != self.img_size:
                    img_tensor = self._safe_resize_and_to_tensor(img)
            except Exception:
                img_tensor = self._safe_resize_and_to_tensor(img)
        else:
            img_tensor = self._safe_resize_and_to_tensor(img)

        img_tensor = img_tensor.float()
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0).repeat(3,1,1)
        if img_tensor.shape[0] != 3:
            if img_tensor.shape[-1] == 3 and img_tensor.dim() == 3:
                img_tensor = img_tensor.permute(2,0,1)
            else:
                img_tensor = self._safe_resize_and_to_tensor(img)

        return img_tensor, label

print("Dataset class ready.")


# ==================== CELL 6: SPLIT & DATALOADERS (single split) ====================
SEED = 42
train_df_, test_df = train_test_split(train_df, test_size=0.10, stratify=train_df['diagnosis'], random_state=SEED)
train_df, val_df = train_test_split(train_df_, test_size=0.10, stratify=train_df_['diagnosis'], random_state=SEED)

print(f"Sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

train_ds = APTOSDataset(train_df, transforms=train_transforms, img_size=IMG_SIZE)
val_ds   = APTOSDataset(val_df,   transforms=valid_transforms, img_size=IMG_SIZE)
test_ds  = APTOSDataset(test_df,  transforms=valid_transforms, img_size=IMG_SIZE)

num_workers = 2
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

imgs, labels = next(iter(train_loader))
print("Sample batch shapes:", imgs.shape, labels.shape)


# ==================== CELL 7: FOCAL LOSS & CLASS WEIGHTS ====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', logits=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** self.gamma) * ce_loss
            if self.alpha is not None:
                at = self.alpha.gather(0, targets)
                loss = at * loss
        else:
            pt = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-12)
            if self.alpha is not None:
                at = self.alpha.gather(0, targets)
                loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# Compute class weights (inverse frequency) for optional use in loss
counts = train_df['diagnosis'].value_counts().sort_index().values
class_weights = torch.tensor(1.0 / (counts + 1e-8), dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(counts)  # normalize scale
print("Class counts:", counts)
print("Class weights (normalized):", class_weights.numpy())


# ==================== CELL 8: MODEL (InceptionV3) - robust for different torchvision versions ====================
import torchvision.models as models
import torch.nn as nn
import torchvision
import warnings

def get_inception_v3(num_classes=5, pretrained=True, aux_logits=False, freeze_backbone=False):
    """
    Robust loader for torchvision's inception_v3 across versions.
    - If torchvision expects aux_logits=True for the pretrained weights, we still load the weights
      but then wrap the model so forward() returns a single logits tensor.
    - Returns a nn.Module that outputs (B, num_classes) logits only.
    """
    # Helper wrapper to make forward always return single logits tensor
    class InceptionWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            # torchvision Inception may return (logits, aux_logits) depending on aux_logits flag
            if isinstance(out, tuple) or isinstance(out, list):
                return out[0]
            return out

    model = None
    # Newer torchvision uses 'weights=' API
    try:
        # Try to use the weights enum if available (torchvision >= 0.13+)
        if pretrained:
            # find available attribute names for weights enum
            # This will fail if torchvision older; wrapped in try/except
            try:
                Weights = models.Inception_V3_Weights  # type: ignore
                weights = Weights.IMAGENET1K_V1
                # Some torchvision versions will force aux_logits=True for pretrained weights.
                # We pass aux_logits=True when required and later wrap to return a single output.
                model = models.inception_v3(weights=weights, aux_logits=aux_logits)
            except Exception:
                # fallback: older signature may be models.inception_v3(pretrained=True, aux_logits=...)
                model = models.inception_v3(pretrained=True, aux_logits=aux_logits)
        else:
            # no pretrained weights requested
            model = models.inception_v3(pretrained=False, aux_logits=aux_logits)
    except ValueError as e:
        # Common case: pretrained weights require aux_logits=True in this torchvision version.
        # Re-call with aux_logits=True, then wrap.
        warnings.warn(f"InceptionV3 loader raised ValueError: {e}. Retrying with aux_logits=True and wrapping the output.")
        try:
            # try with weights=... if available
            try:
                Weights = models.Inception_V3_Weights  # type: ignore
                weights = Weights.IMAGENET1K_V1 if pretrained else None
                model = models.inception_v3(weights=weights, aux_logits=True)
            except Exception:
                model = models.inception_v3(pretrained=pretrained, aux_logits=True)
        except Exception as e2:
            # As a last fallback, create without pretrained weights
            warnings.warn(f"Failed to load pretrained inception_v3 due to: {e2}. Creating model without pretrained weights.")
            model = models.inception_v3(pretrained=False, aux_logits=False)

    # Replace final fc to match num_classes
    # InceptionV3 has attribute `fc`
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Optionally freeze backbone parameters (keep final fc trainable)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    # Wrap to ensure forward returns single logits tensor (works for both aux_logits True/False)
    model = InceptionWrapper(model)
    return model

# Create model (set freeze_backbone=True to train only fc first)
model = get_inception_v3(num_classes=5, pretrained=True, aux_logits=False, freeze_backbone=False)
model = model.to(DEVICE)
print("InceptionV3 model loaded. Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

best_ckpt = "inception_v3_best.pth"
last_ckpt = "inception_v3_last.pth"



# ==================== CELL 9: METRICS & EVAL HELPERS ====================
scaler = torch.amp.GradScaler(device="cuda") if torch.cuda.is_available() else torch.amp.GradScaler()

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    if criterion is None:
        criterion = FocalLoss(gamma=2.0, logits=True)
    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=device_type):
            logits = model(imgs)
            loss = criterion(logits, targets)
        preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        running_loss += loss.item() * imgs.size(0)
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    q = qwk(all_targets, all_preds)
    return avg_loss, acc, q, np.array(all_targets), np.array(all_preds)


# ==================== CELL 10: TRAIN LOOP (MIXED PRECISION + BEST CHECKPOINT) ====================
EPOCHS = 12
# use focal loss, optionally pass class_weights as alpha
alpha = class_weights.to(DEVICE)  # try using this or set to None
criterion = FocalLoss(gamma=2.0, alpha=alpha, logits=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_qwk':[]}
best_qwk = -1.0

# Optional: MixUp helper (commented out by default)
# def mixup_data(x, y, alpha=0.4):
#     if alpha <= 0:
#         return x, y, None, 1.0
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"
    for imgs, targets in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type):
            logits = model(imgs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.softmax(dim=1).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{correct/total:.4f}'})

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    val_loss, val_acc, val_qwk, _, _ = evaluate(model, val_loader, criterion=criterion)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_qwk'].append(val_qwk)

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_qwk={val_qwk:.4f}")

    if val_qwk > best_qwk:
        best_qwk = val_qwk
        torch.save(model.state_dict(), best_ckpt)
        print(f"Saved best model with val_qwk={best_qwk:.4f}")

    scheduler.step()

torch.save(model.state_dict(), last_ckpt)
print("Training finished.")


# ==================== CELL 11: PLOT TRAINING METRICS ====================
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.plot(history['val_qwk'], label='val_qwk')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Training Accuracy / Val Accuracy / Val QWK')
plt.show()



# ==================== CELL 12: TEST EVALUATION (with optional TTA) ====================
from sklearn.metrics import classification_report, confusion_matrix

# load best
ckpt_path = best_ckpt
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Simple inference function
@torch.no_grad()
def inference(model, loader):
    all_preds = []
    all_targets = []
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())
    return np.array(all_targets), np.array(all_preds)

y_true, y_pred = inference(model, test_loader)
test_acc = accuracy_score(y_true, y_pred)
test_qwk = qwk(y_true, y_pred)
test_f1_macro = f1_score(y_true, y_pred, average='macro')
print(f"✅ Loaded checkpoint: {ckpt_path}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test QWK: {test_qwk:.4f}")
print(f"Test F1 (macro): {test_f1_macro:.4f}\n")
print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]), ha='center', va='center')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix'); plt.colorbar(); plt.show()


# ==================== CELL 13: GRAD-CAM VISUALIZATION ====================
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

# --- Grad-CAM class ---
class GradCAM:
    def __init__(self, model, target_layer=None, use_cuda=True, smooth=True, sigma=3):
        self.model = model
        self.device = next(model.parameters()).device if use_cuda else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.smooth = smooth
        self.sigma = sigma

        # find last Conv2d layer if not specified
        if target_layer is None:
            for m in reversed(list(self.model.modules())):
                if isinstance(m, torch.nn.Conv2d):
                    target_layer = m
                    break
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        try:
            self.target_layer.register_full_backward_hook(backward_hook)
        except Exception:
            self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        assert input_tensor.dim() == 4, "input_tensor must be 4D (B,C,H,W)"
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()
        logits = self.model(input_tensor)
        B = logits.shape[0]

        if target_class is None:
            target_class = logits.softmax(dim=1).argmax(dim=1).cpu().tolist()
        elif isinstance(target_class, int):
            target_class = [target_class] * B
        else:
            target_class = [int(x) for x in target_class]

        cams = []
        for i in range(B):
            score = logits[i, target_class[i]]
            score.backward(retain_graph=True)
            grads = self.gradients[i]
            acts = self.activations[i]
            weights = grads.mean(dim=(1, 2))
            cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)
            cam = F.relu(cam)
            cam_np = cam.cpu().numpy()
            if cam_np.max() > 0:
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            if self.smooth:
                cam_np = gaussian_filter(cam_np, sigma=self.sigma)
            cams.append(cam_np)
            self.model.zero_grad()

        cams = np.stack(cams, axis=0)
        _, _, H, W = input_tensor.shape
        cams_t = torch.from_numpy(cams).unsqueeze(1).to(self.device)
        cams_resized = F.interpolate(cams_t, size=(H, W), mode="bilinear", align_corners=False)
        cams_resized = cams_resized.squeeze(1).cpu().numpy()
        out = []
        for c in cams_resized:
            if c.max() > 0:
                c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            out.append(c)
        return np.stack(out, axis=0)

# --- helper to unnormalize ---
def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.clip(img * std + mean, 0, 1)
    return (img * 255).astype(np.uint8)

# --- Generate and save overlays ---
os.makedirs("gradcam_inception_outputs", exist_ok=True)
cam_tool = GradCAM(model, smooth=True, sigma=3)

# choose indices to visualize
indices = [0, 10, 25, 50]
print("Producing Grad-CAM overlays for indices:", indices)

for idx in indices:
    img_t, label = test_ds[idx]
    inp = img_t.unsqueeze(0).to(DEVICE)
    cam_map = cam_tool.generate_cam(inp)[0]
    cam_uint8 = np.uint8(255 * cam_map)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    orig = unnormalize(img_t)
    orig_f = orig.astype(np.float32) / 255.0
    overlay = 0.4 * heatmap + 0.6 * orig_f
    overlay = np.clip(overlay, 0, 1)

    base = f"idx{idx}_lbl{label}"
    cv2.imwrite(f"gradcam_inception_outputs/{base}_orig.png", cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"gradcam_inception_outputs/{base}_heatmap.png", cv2.cvtColor((heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"gradcam_inception_outputs/{base}_overlay.png", cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(orig); plt.title(f"Orig idx={idx} label={label}"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(heatmap); plt.title("Heatmap"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
    plt.show()

print("✅ Saved Grad-CAM images to ./gradcam_inception_outputs/")



# ==================== CELL 14: SAVE TEST PREDICTIONS ====================
out_df = test_df.copy().reset_index(drop=True)
out_df["true"] = y_true
out_df["pred"] = y_pred
out_df.to_csv("inception_v3_test_predictions.csv", index=False)
print("✅ Saved test predictions → inception_v3_test_predictions.csv")

# quick peek
print(out_df.head())


