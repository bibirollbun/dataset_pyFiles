import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm
import pydicom
from PIL import Image
!pip install torchcam
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask


# Paths
DATA_DIR    = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'
CSV_PATH    = os.path.join(DATA_DIR, 'stage_2_train.csv')
DCM_DIR     = os.path.join(DATA_DIR, 'stage_2_train')
PNG_ROOT    = '/kaggle/working/png_data'
OUTPUT_DIR  = '/kaggle/working/output_task4'
ROC_DIR     = os.path.join(OUTPUT_DIR, 'roc_curves')
CAM_DIR     = os.path.join(OUTPUT_DIR, 'gradcam')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')

# Create directories
for d in [PNG_ROOT, OUTPUT_DIR, ROC_DIR, CAM_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)



# Hyperparameters
BATCH_SIZE    = 8
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-6
PATIENCE      = 10
NUM_FOLDS     = 5
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES   = ['epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular']


def generate_gradcams(
    model, df, probs, targets, classes,
    png_root, transform, out_dir, fold,
    n_tp=5, n_fp=5, n_fn=5, device=DEVICE
):

    os.makedirs(out_dir, exist_ok=True)
    cam_extractor = SmoothGradCAMpp(model)
    preds = (probs >= 0.5).astype(int)

    for i, cls in enumerate(classes):
        # 选索引
        tp_idxs = np.where((targets[:,i]==1)&(preds[:,i]==1))[0][:n_tp]
        fp_idxs = np.where((targets[:,i]==0)&(preds[:,i]==1))[0][:n_fp]
        fn_idxs = np.where((targets[:,i]==1)&(preds[:,i]==0))[0][:n_fn]

        for kind, idxs in [('TP',tp_idxs), ('FP',fp_idxs), ('FN',fn_idxs)]:
            for j, idx in enumerate(idxs):
                img_id = df.loc[idx, 'image']
                pil = Image.open(os.path.join(png_root, img_id+'.png')).convert('RGB')
                inp = transform(pil).unsqueeze(0).to(device)
                out = model(inp)
                activation_map = cam_extractor(class_idx=i, scores=out)[0].cpu().numpy()
                heatmap = overlay_mask(pil, activation_map, alpha=0.5)

                fname = f'gradcam_{cls}_{kind}_fold{fold}_{j}.png'
                heatmap.save(os.path.join(out_dir, fname))
    print(f"[Fold {fold}] Saved Grad-CAM to {out_dir}")


BALANCE_N   = 500   # 每類正/負例上限

# 3. 讀 CSV，拆出 image & subtype
df = pd.read_csv(CSV_PATH)
df[['image','subtype']] = df['ID'].str.rsplit('_', n=1, expand=True)

# 4. Pivot 成 multi-hot 格式
df_ml = df.pivot_table(
    index='image',
    columns='subtype',
    values='Label',
    aggfunc='max',
    fill_value=0
).reset_index()
df_ml['any'] = df_ml[CLASS_NAMES].max(axis=1)

# 5. Balance 子集：對每個 class 抽正/負例
frames = []
for cls in CLASS_NAMES:
    pos = df_ml[df_ml[cls]==1]
    neg = df_ml[df_ml[cls]==0]
    # 如果正例超過上限，就隨機抽 BALANCE_N；否則全要
    pos = pos.sample(BALANCE_N, random_state=42) if len(pos)>BALANCE_N else pos
    # 同理抽負例
    neg = neg.sample(BALANCE_N, random_state=42) if len(neg)>BALANCE_N else neg
    frames.append(pos)
    frames.append(neg)

# 合併並去重，確保同一張圖只出現一次
balanced_df = pd.concat(frames).drop_duplicates(subset='image').reset_index(drop=True)
print(f"Balanced subset size: {len(balanced_df)} images")




# 6. 存成 CSV
balanced_csv = os.path.join(METRICS_DIR, 'subset_multilabel_balanced.csv')
balanced_df.to_csv(balanced_csv, index=False)

# 7. 只針對子集進行 DICOM → PNG 轉檔
print("Converting balanced subset DICOM to PNG...")
for img_id in tqdm(balanced_df['image'].unique(), desc="DICOM→PNG"):
    dcm_path = os.path.join(DCM_DIR, img_id + '.dcm')
    png_path = os.path.join(PNG_ROOT, img_id + '.png')
    if not os.path.exists(dcm_path):
        print(f"Missing file: {dcm_path}")
        continue
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(png_path)

print("Done! Balanced subset PNGs are in:", PNG_ROOT)
#///


class MultiHemoDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root = root_dir
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec = self.df.iloc[idx]
        # 1) 先读影像
        path = os.path.join(self.root, rec['image'] + '.png')
        img = Image.open(path).convert('RGB')
        if self.tf:
            img = self.tf(img)
        # 2) 明确把 label 列转成 float32 数组
        #    这样就不会是 object dtype 了
        label_arr = rec[CLASS_NAMES].astype(np.float32).to_numpy()
        labels = torch.from_numpy(label_arr)          # dtype=torch.float32
        return img, labels


# 3. Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# 4. Cross-validation setup
kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
X = balanced_df['image']
y = balanced_df['any']

all_metrics   = []
all_roc_data  = {cls: [] for cls in CLASS_NAMES}
fold          = 0

for train_idx, val_idx in kf.split(X, y):
    fold += 1
    print(f"\n=== Fold {fold}/{NUM_FOLDS} ===")
    train_df = balanced_df.iloc[train_idx].reset_index(drop=True)
    val_df   = balanced_df.iloc[val_idx].reset_index(drop=True)

    # Save splits
    train_df.to_csv(os.path.join(METRICS_DIR, f'train_fold_{fold}.csv'), index=False)
    val_df.to_csv(  os.path.join(METRICS_DIR, f'val_fold_{fold}.csv'),   index=False)

    train_loader = DataLoader(
        MultiHemoDataset(train_df, PNG_ROOT, train_transform),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(
        MultiHemoDataset(val_df,   PNG_ROOT, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model / Loss / Optimizer
    model     = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc  = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model     = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    best_val_loss    = np.inf
    # Prepare lists for logging
    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    # --- Epoch Loop ---
    for epoch in range(1, NUM_EPOCHS+1):
        # — Training —
        model.train()
        running_loss = running_corr = running_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item() * imgs.size(0)
            preds         = (torch.sigmoid(outputs) >= 0.5).long()
            running_corr  += (preds == labels).all(dim=1).sum().item()
            running_total += imgs.size(0)

        epoch_train_loss = running_loss / running_total
        epoch_train_acc  = running_corr  / running_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # — Validation —
        model.eval()
        val_running_loss = val_running_corr = val_running_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

                val_running_loss  += loss.item() * imgs.size(0)
                preds              = (torch.sigmoid(outputs) >= 0.5).long()
                val_running_corr  += (preds == labels).all(dim=1).sum().item()
                val_running_total += imgs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / val_running_total
        epoch_val_acc  = val_running_corr  / val_running_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # Print epoch results
        print(f"Epoch {epoch}/{NUM_EPOCHS} — "
              f"Train loss: {epoch_train_loss:.4f}, acc: {epoch_train_acc:.4f} | "
              f" Val loss: {epoch_val_loss:.4f}, acc: {epoch_val_acc:.4f}")

            # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improve    = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_fold{fold}.pth'))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        
            # --- Plot & save Loss/Accuracy curves ---
    epochs = range(1, len(train_losses)+1)
    # Loss
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Fold {fold} Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'loss_fold_{fold}.png'))
    plt.close()
    # Accuracy
    plt.figure()
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs,   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title(f'Fold {fold} Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'acc_fold_{fold}.png'))
    plt.close()


    # Compute metrics
    metrics = {}
    for i, cls in enumerate(CLASS_NAMES):
        preds = (val_probs[:, i] >= 0.5).astype(int)
        p = precision_score(val_targets[:, i], preds, zero_division=0)
        r = recall_score(   val_targets[:, i], preds, zero_division=0)
        f = f1_score(       val_targets[:, i], preds, zero_division=0)
        metrics[cls] = {'precision': p, 'recall': r, 'f1': f}
        # ROC per fold
        fpr, tpr, _ = roc_curve(val_targets[:, i], val_probs[:, i])
        roc_auc = auc(fpr, tpr)
        all_roc_data[cls].append((fpr, tpr, roc_auc))
        # Save ROC per fold
        plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}"); plt.plot([0,1],[0,1],'--')
        plt.title(f"ROC {cls} Fold{fold}"); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
        plt.savefig(os.path.join(ROC_DIR, f'roc_{cls}_fold{fold}.png')); plt.close()

    # Micro/Macro
    preds_all = (val_probs>=0.5).astype(int)
    metrics['micro'] = {} ; metrics['macro'] = {}
    for avg in ['micro','macro']:
        p,r,f,_ = precision_score(val_targets, preds_all, average=avg, zero_division=0), \
                  recall_score(  val_targets, preds_all, average=avg, zero_division=0), \
                  f1_score(      val_targets, preds_all, average=avg, zero_division=0), None
        metrics[avg] = {'precision':p, 'recall':r, 'f1':f}

    # Save metrics per fold
    df_metrics = []
    for key,val in metrics.items():
        df_metrics.append({'fold':fold, 'class':key, **val})
    pd.DataFrame(df_metrics).to_csv(
        os.path.join(METRICS_DIR, f'metrics_fold_{fold}.csv'), index=False)
    all_metrics.append(metrics)

    # Standardized Grad-CAM analysis: TP/FP/FN samples
    generate_gradcams(
        model=model,
        df=val_df,
        probs=val_probs,
        targets=val_targets,
        classes=CLASS_NAMES,
        png_root=PNG_ROOT,
        transform=val_transform,
        out_dir=os.path.join(CAM_DIR, f'fold_{fold}'),
        fold=fold,
        n_tp=5, n_fp=5, n_fn=5
    )



# 5. Average metrics across folds
avg_metrics = {}
for key in all_metrics[0].keys():
    vals = [fold_m[key] for fold_m in all_metrics]
    avg_metrics[key] = {m: np.mean([v[m] for v in vals]) for m in ['precision','recall','f1']}
pd.DataFrame([{'class':k, **v} for k,v in avg_metrics.items()])\
    .to_csv(os.path.join(METRICS_DIR, 'metrics_average.csv'), index=False)


# 6. Aggregate ROC curves per class
def plot_avg_roc(cls):
    mean_fpr = np.linspace(0,1,100)
    tprs = []
    aucs = []
    for fpr,tpr,roc_auc in all_roc_data[cls]:
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    plt.plot(mean_fpr, mean_tpr, label=f"{cls} (AUC={mean_auc:.2f})")

plt.figure()
for cls in CLASS_NAMES:
    plot_avg_roc(cls)
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Average ROC')
plt.legend(); plt.savefig(os.path.join(ROC_DIR, 'roc_average.png'))
plt.close()

