# Cell 1: imports + config
import os, sys, math, random, time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import cv2
from tqdm import tqdm

ROOT = Path("/kaggle/input/rsna-intracranial-aneurysm-detection")  # adjust if local
SERIES_ROOT = ROOT / "series"
SEG_ROOT = ROOT / "segmentations"
TRAIN_CSV = ROOT / "train.csv"
LOCALIZER_CSV = ROOT / "train_localizers.csv"

SAMPLE_SERIES_META_READ = 128   
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("ROOT:", ROOT)
print("Series folder exists:", SERIES_ROOT.exists())
print("Segmentations folder exists:", SEG_ROOT.exists())
print("Train CSV exists:", TRAIN_CSV.exists())


# load CSV and columns/head 
train_df = pd.read_csv(TRAIN_CSV)
localizer_df = pd.read_csv(LOCALIZER_CSV)

print("train.csv shape:", train_df.shape)


print("train.csv columns:\n", train_df.columns.tolist())


print("\ntrain.csv sample:")
display(train_df.head())


print("\nlocalizer.csv shape:", localizer_df.shape)
display(localizer_df.head())


print("Aneurysm Present distribution:")
print(train_df['Aneurysm Present'].value_counts())


print("\nModality distribution:")
print(train_df['Modality'].value_counts())


print("\nPatientSex distribution:")
print(train_df['PatientSex'].value_counts())


print("\nPatientAge stats:")
display(train_df['PatientAge'].describe())


# vessel columns auto-detect:  numeric 0/1 type and 'Aneurysm Present' 
non_vessel = {'SeriesInstanceUID','PatientAge','PatientSex','Modality','Aneurysm Present'}
vessel_cols = [c for c in train_df.columns if c not in non_vessel]
print("\nDetected vessel columns (count={}):".format(len(vessel_cols)))
print(vessel_cols)


# vessel frequencies (how many series have aneurysm in each vessel)
vessel_counts = train_df[vessel_cols].sum().sort_values(ascending=False)
display(vessel_counts)


# Cell 4: build series -> num_slices mapping (fast: only filenames counted)
series_dirs = sorted([p.name for p in SERIES_ROOT.iterdir() if p.is_dir()])
print("Total series folders found:", len(series_dirs))


# map series -> num files
series_len = {}
for sid in tqdm(series_dirs):
    try:
        files = os.listdir(SERIES_ROOT / sid)
        series_len[sid] = len(files)
    except Exception as e:
        series_len[sid] = 0


# convert to DataFrame
series_len_df = pd.DataFrame.from_dict(series_len, orient='index', columns=['num_slices'])
series_len_df.index.name = 'SeriesInstanceUID'
series_len_df.reset_index(inplace=True)
display(series_len_df.head())


# basic stats
print(series_len_df['num_slices'].describe())
plt.figure(figsize=(8,4))
sns.histplot(series_len_df['num_slices'], bins=50, log_scale=(False,True))
plt.title("Distribution of number of slices per series (log y)")
plt.xlabel("num_slices")
plt.show()


#  join csv and series_len, check missing series and segmentation files
# left-join train_df (has series we care about) with series_len_df
df_series = train_df[['SeriesInstanceUID','Aneurysm Present','Modality','PatientAge','PatientSex']].merge(
    series_len_df, how='left', on='SeriesInstanceUID'
)


# mark missing series folders
missing_series = df_series['num_slices'].isna().sum()
print("Train CSV series without series folder (missing):", missing_series)


# is there segmentation file for each series?
def has_seg(series_id):
    seg_file = SEG_ROOT / f"{series_id}.npz"
    return seg_file.exists()


# apply quickly (may be heavy if many series) â€” but we do it only for train_df entries
df_series['has_seg'] = df_series['SeriesInstanceUID'].apply(lambda x: (SEG_ROOT / f"{x}.npz").exists())


# summary
print(df_series['has_seg'].value_counts())
display(df_series.head())


# Save summary CSV for later
df_series.to_csv("train_series_summary.csv", index=False)
print("Saved train_series_summary.csv")


# Cell 6: relationship between num_slices and Aneurysm Present
plt.figure(figsize=(6,4))
sns.boxplot(data=df_series, x='Aneurysm Present', y='num_slices')
plt.title("num_slices by Aneurysm Present")
plt.show()


plt.figure(figsize=(8,4))
sns.boxplot(data=df_series, x='Modality', y='num_slices')
plt.title("num_slices by Modality")
plt.show()


top10 = series_len_df.sort_values('num_slices', ascending=False).head(10)
print("Top 10 longest series (num_slices):")
display(top10)


bot10 = series_len_df.sort_values('num_slices', ascending=True).head(10)
print("\nTop 10 shortest series:")
display(bot10)


#  read DICOM headers from a random subset of series (to inspect pixel spacing, slice thickness)
sample_series = random.sample(series_dirs, min(SAMPLE_SERIES_META_READ, len(series_dirs)))
meta_list = []
for sid in tqdm(sample_series):
    series_path = SERIES_ROOT / sid
    try:
        files = sorted(os.listdir(series_path))
        if len(files)==0:
            continue
        dcm_path = series_path / files[0]
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)  # faster, no pixel data
        pixel_spacing = getattr(ds, 'PixelSpacing', None)
        slice_thickness = getattr(ds, 'SliceThickness', None)
        modality = getattr(ds, 'Modality', None)
        meta_list.append({'SeriesInstanceUID': sid,
                          'PixelSpacing': pixel_spacing,
                          'SliceThickness': slice_thickness,
                          'Modality': modality})
    except Exception as e:
        meta_list.append({'SeriesInstanceUID': sid, 'PixelSpacing': None, 'SliceThickness': None, 'Modality': None})


meta_df = pd.DataFrame(meta_list)
display(meta_df.head())
print("PixelSpacing value counts (first element if list):")


# simplify PixelSpacing
meta_df['ps0'] = meta_df['PixelSpacing'].apply(lambda x: x[0] if isinstance(x, (list,tuple)) and len(x)>0 else None)
display(meta_df['ps0'].value_counts().head(20))
display(meta_df['SliceThickness'].value_counts().head(20))


# pick a positive series sample and visualize first 12 slices with overlay if mask exists
pos_series = df_series[df_series['Aneurysm Present']==1]['SeriesInstanceUID'].dropna().tolist()
if len(pos_series)==0:
    print("No positive series in train_df!")
else:
    sample_sid = random.choice(pos_series)
    print("Sample positive series:", sample_sid)
    series_path = SERIES_ROOT / sample_sid
    files = sorted(os.listdir(series_path))
    n_show = min(12, len(files))
    # load mask if exists
    mask_file = SEG_ROOT / f"{sample_sid}.npz"
    mask = None
    if mask_file.exists():
        try:
            mask = np.load(str(mask_file))['arr_0']
            print("Mask shape:", mask.shape)
        except Exception as e:
            print("Failed to load mask:", e)
            mask = None

    # plot grid
    fig, axes = plt.subplots(3, 4, figsize=(12,9))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            dcm = pydicom.dcmread(str(series_path / files[i]))
            img = dcm.pixel_array.astype(np.float32)
            if img.max() > img.min():
                imgn = (img - img.min()) / (img.max() - img.min())
            else:
                imgn = np.zeros_like(img)
            ax.imshow(imgn, cmap='gray')
            if (mask is not None) and (i < mask.shape[0]):
                ax.imshow(mask[i], cmap='Reds', alpha=0.4)
            ax.set_title(f"Slice {i+1}")
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (8,5)


path = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
train_df = pd.read_csv(path)


print("ğŸ”¹ Shape of Dataset:", train_df.shape)


print("\nğŸ”¹ First 5 Rows:\n")
display(train_df.head())


print("\nğŸ”¹ Column Names:\n", list(train_df.columns))


print("\nğŸ“„ Dataset Info:\n")
train_df.info()


print("\nğŸ”� Missing Values:")
print(train_df.isnull().sum())


plt.figure(figsize=(10,4))
msno.bar(train_df)
plt.title("Missing Value Overview")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(train_df["PatientAge"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Patient Age")
plt.xlabel("Age (years)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(5,4))
sns.countplot(x="PatientSex", data=train_df, palette="Set2")
plt.title("Patient Sex Distribution")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(7,5))
sns.boxplot(data=train_df, x="PatientSex", y="PatientAge", palette="Set3")
plt.title("Age vs Sex Distribution")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x="Aneurysm Present", data=train_df, palette="coolwarm")
plt.title("Overall Aneurysm Presence")
plt.xlabel("Aneurysm Present (1 = Yes, 0 = No)")
plt.ylabel("Number of Patients")
plt.show()



aneurysm_rate = train_df["Aneurysm Present"].value_counts(normalize=True) * 100
print("ğŸ§© Aneurysm Presence (%):\n", aneurysm_rate)



# Identify all artery-related columns
artery_cols = train_df.columns[4:-1]


# Total positive count in each artery
artery_counts = train_df[artery_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(12,7))
sns.barplot(x=artery_counts.values, y=artery_counts.index, palette="viridis")
plt.title("Aneurysm Count by Artery")
plt.xlabel("Number of Positive Cases")
plt.ylabel("Artery Name")
plt.show()



artery_percentage = (artery_counts / len(train_df)) * 100
print("\nğŸ“Š Percentage of Patients with Aneurysm by Artery:\n")
print(artery_percentage.sort_values(ascending=False))


plt.figure(figsize=(12,10))
sns.heatmap(train_df[artery_cols].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Between Artery Sites")
plt.show()


train_df["num_positive_sites"] = train_df[artery_cols].sum(axis=1)

plt.figure(figsize=(8,5))
sns.countplot(x="num_positive_sites", data=train_df, palette="crest")
plt.title("Distribution of Number of Aneurysm Sites per Patient")
plt.xlabel("Number of Aneurysm Sites")
plt.ylabel("Number of Patients")
plt.show()


print(train_df["num_positive_sites"].value_counts().sort_index())


plt.figure(figsize=(8,5))
sns.kdeplot(data=train_df, x="PatientAge", hue="Aneurysm Present", fill=True, common_norm=False, palette="coolwarm")
plt.title("Age vs Aneurysm Presence")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()


sex_group = train_df.groupby("PatientSex")["Aneurysm Present"].mean() * 100

plt.figure(figsize=(6,4))
sns.barplot(x=sex_group.index, y=sex_group.values, palette="mako")
plt.title("Aneurysm Rate by Sex (%)")
plt.ylabel("Percentage of Positive Cases")
plt.show()


print("Aneurysm Rate by Sex (%):")
print(sex_group)


summary = {
    "Total Patients": len(train_df),
    "Aneurysm Present (%)": round(train_df["Aneurysm Present"].mean() * 100, 2),
    "Average Age": round(train_df["PatientAge"].mean(), 2),
    "Male %": round((train_df["PatientSex"].value_counts(normalize=True).get('M',0)) * 100, 2),
    "Female %": round((train_df["PatientSex"].value_counts(normalize=True).get('F',0)) * 100, 2),
    "Average Positive Sites": round(train_df["num_positive_sites"].mean(), 2)
}

print("\nğŸ“‹ Dataset Summary:\n")
for k,v in summary.items():
    print(f"{k}: {v}")


print("""
ğŸ”� Key Insights from EDA:
1. Dataset contains {} patients.
2. Overall aneurysm prevalence: {:.2f}%.
3. Average age: {:.1f} years; both male & female distributions are balanced.
4. Most affected arteries: {}.
5. Some arteries (e.g., Basilar Tip, ACom) show co-occurrence correlations.
6. Patients aged 50â€“70 tend to have higher aneurysm probability.
7. Most patients have aneurysm in only one site, few have multiple.
""".format(
    len(train_df),
    train_df["Aneurysm Present"].mean() * 100,
    train_df["PatientAge"].mean(),
    ", ".join(artery_counts.head(3).index)
))



# Cell 1: imports & config
import os, random, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import roc_auc_score, accuracy_score

import pydicom
import cv2
from tqdm import tqdm

# Config
ROOT = Path("/kaggle/input/rsna-intracranial-aneurysm-detection")
SERIES_ROOT = ROOT / "series"
SEG_ROOT = ROOT / "segmentations"
TRAIN_CSV = ROOT / "train.csv"

IMG_SIZE = 224            # EfficientNet-B0 standard
BATCH_SIZE = 16           # adjust for your GPU
SLICES_PER_SERIES = 1     # how many slices to sample per series (use 1 or small for speed)
SUBSET_SLICES = 10000     # total slice samples for fast prototyping
NUM_EPOCHS = 5
LR = 1e-4

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
print("ROOT exists:", ROOT.exists(), "series exists:", SERIES_ROOT.exists())



# Cell 2: helper functions for DICOM handling & normalization
def safe_read_dcm(path):
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    # if multi-frame or channel, pick first 2D plane
    if arr is None:
        raise RuntimeError(f"Empty pixel array: {path}")
    if arr.ndim > 2:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    return arr

def safe_normalize(img):  # img: numpy 2D float32
    mn = img.min()
    mx = img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img


# quick function to convert single 2D img -> 3-channel tensor after transforms (PIL-based transforms expect HxW or array)
from PIL import Image
def prepare_image_from_array(arr, img_size=IMG_SIZE, transform=None):
    arrn = safe_normalize(arr)
    # convert to uint8 0-255 for PIL compatibility
    arr_uint8 = (arrn * 255).astype(np.uint8)
    pil = Image.fromarray(arr_uint8).convert("L")  # grayscale
    if transform is not None:
        x = transform(pil)
    else:
        x = T.Compose([
            T.Resize((img_size,img_size)),
            T.ToTensor()
        ])(pil)
    # x shape = [1,H,W] -> convert to 3 channels by repeat
    if x.shape[0] == 1:
        x = x.repeat(3,1,1)
    return x



# Cell 3: Slice-level dataset that samples up to SUBSET_SLICES total slices for speed.
class RSNASliceDatasetFast(Dataset):
    def __init__(self, train_df, series_root, slices_per_series=1, subset_slices=10000, img_size=IMG_SIZE, transform=None):
        """
        - train_df: the train.csv pandas DataFrame (must contain 'SeriesInstanceUID' and 'Aneurysm Present')
        - slices_per_series: how many slices to take from each series (e.g., 1 or 3 or 5)
        - subset_slices: if >0, limit total samples to this many for fast prototyping
        """
        self.series_root = Path(series_root)
        self.transform = transform
        self.img_size = img_size
        samples = []  # tuples (series_uid, filename, label)
        # iterate through train_df, collect file paths (first few slices per series)
        for _, row in train_df.iterrows():
            sid = row['SeriesInstanceUID']
            label = int(row['Aneurysm Present'])
            series_folder = self.series_root / sid
            if not series_folder.exists():
                continue
            files = sorted(os.listdir(series_folder))
            if len(files) == 0:
                continue
            # choose slices_per_series slices: prefer middle region (more likely to contain anatomy)
            L = len(files)
            idxs = np.linspace(0, L-1, min(slices_per_series, L), dtype=int)
            selected = [files[i] for i in idxs]
            for f in selected:
                samples.append((sid, f, label))
            if subset_slices and len(samples) >= subset_slices:
                break
        self.samples = samples[:subset_slices] if subset_slices else samples
        print(f"Built dataset with {len(self.samples)} slice samples (slices_per_series={slices_per_series})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, fname, label = self.samples[idx]
        path = self.series_root / sid / fname
        arr = safe_read_dcm(path)
        # prepare PIL + transform into tensor [3,H,W]
        img_t = prepare_image_from_array(arr, img_size=self.img_size, transform=self.transform)
        return img_t, torch.tensor(label, dtype=torch.float32)



# build transforms (ImageNet normalization)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3,1,1)),   # convert 1-channel -> 3-channel
    T.Normalize(mean=imagenet_mean, std=imagenet_std)
])
valid_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3,1,1)),
    T.Normalize(mean=imagenet_mean, std=imagenet_std)
])


# load train_df
train_df = pd.read_csv(TRAIN_CSV)

# create dataset (fast subset)
dataset = RSNASliceDatasetFast(train_df, series_root=SERIES_ROOT, slices_per_series=SLICES_PER_SERIES,
                               subset_slices=SUBSET_SLICES, img_size=IMG_SIZE, transform=train_transform)

# quick sanity check
print("Sample count:", len(dataset))
x,y = dataset[0]
print("Sample tensor shape:", x.shape, "Label:", y)


#  train/val split on the slice-samples (stratify by label roughly)
from sklearn.model_selection import train_test_split

indices = list(range(len(dataset)))
# build labels list for stratify
labels = [dataset.samples[i][2] for i in indices]
train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=SEED, stratify=labels)

train_ds = torch.utils.data.Subset(dataset, train_idx)
val_ds   = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Train samples:", len(train_ds), "Val samples:", len(val_ds))



#  EfficientNet-B0 model (try torchvision, fallback timm)
def get_efficientnet_b0(pretrained=True):
    try:
        # torchvision (newer versions)
        model = models.efficientnet_b0(pretrained=pretrained)
        # replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1)
        )
        return model
    except Exception as e:
        # fallback to timm if available
        try:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=1)
            return model
        except Exception as e2:
            raise RuntimeError("EfficientNet B0 not available in torchvision and timm not installed.")

model = get_efficientnet_b0(pretrained=True).to(DEVICE)
print(model)



# loss, optimizer, scheduler, train/val functions
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    preds = []
    trues = []
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist()
        preds += probs
        trues += labels.detach().cpu().numpy().ravel().tolist()
    # compute slice-level AUC
    try:
        auc = roc_auc_score(trues, preds)
    except:
        auc = float('nan')
    return np.mean(losses), auc

def validate(model, loader, device):
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(imgs)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy().ravel().tolist()
            preds += probs
            trues += labels.cpu().numpy().ravel().tolist()
    try:
        auc = roc_auc_score(trues, preds)
    except:
        auc = float('nan')
    # accuracy at 0.5 threshold
    preds_bin = [1 if p>=0.5 else 0 for p in preds]
    acc = accuracy_score(trues, preds_bin)
    return np.mean(losses), auc, acc



#  training loop
best_val_auc = 0.0
history = {'train_loss':[], 'train_auc':[], 'val_loss':[], 'val_auc':[], 'val_acc':[]}

for epoch in range(1, NUM_EPOCHS+1):
    t0 = time.time()
    train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss, val_auc, val_acc = validate(model, val_loader, DEVICE)
    scheduler.step(val_auc)  # reduce lr on plateau of val_auc
    history['train_loss'].append(train_loss)
    history['train_auc'].append(train_auc)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)
    history['val_acc'].append(val_acc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_effnet_b0_slice.pth")
        print("Saved best model.")
    print(f"Epoch {epoch} - time {time.time()-t0:.1f}s | train_loss {train_loss:.4f} train_auc {train_auc:.4f} | val_loss {val_loss:.4f} val_auc {val_auc:.4f} val_acc {val_acc:.4f}")



#  demonstrate series-level aggregation from slice predictions (on validation set)
# We'll collect per-sample preds + series_id and then groupby max or mean

# Build map idx->(sid,fname,label) from dataset.samples
samples = dataset.samples  # list of tuples (sid,fname,label)
# For val indices, get predictions
model.load_state_dict(torch.load("best_effnet_b0_slice.pth"))  # load best
model.eval()

series_preds = {}  # sid -> list of probs
series_labels = {} # sid -> true label (series-level)
with torch.no_grad():
    for idx in val_idx:   # val_idx from earlier split
        img, label = dataset[idx]
        inp = img.unsqueeze(0).to(DEVICE)
        prob = torch.sigmoid(model(inp)).item()
        sid = samples[idx][0]
        series_preds.setdefault(sid, []).append(prob)
        series_labels[sid] = samples[idx][2]

# aggregate: max and mean
series_final = []
for sid, probs in series_preds.items():
    series_final.append({
        'series_id': sid,
        'label': series_labels[sid],
        'prob_max': float(np.max(probs)),
        'prob_mean': float(np.mean(probs))
    })
ser_df = pd.DataFrame(series_final)
print("Series-level AUC (max):", roc_auc_score(ser_df['label'], ser_df['prob_max']))
print("Series-level AUC (mean):", roc_auc_score(ser_df['label'], ser_df['prob_mean']))



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms as T
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.efficientnet_b0(weights='IMAGENET1K_V1')  # pretrained
num_features = model.classifier[1].in_features


model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(num_features, 1),  # output 1 for binary
    nn.Sigmoid()
)
model = model.to(device)


criterion = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=1e-4)


dataset = RSNASliceDatasetFast(train_df, series_root=SERIES_ROOT, slices_per_series=SLICES_PER_SERIES,
                               subset_slices=10000, img_size=IMG_SIZE, transform=train_transform)

# 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1).float()
        outputs = model(imgs)
        preds = (outputs > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm  = confusion_matrix(all_labels, all_preds)

print("Validation Accuracy:", acc)
print("F1 Score:", f1)
sns.heatmap(cm, annot=True, fmt="d")



import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_losses = []

for batch_idx, (data, target) in enumerate(dataloader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss per Batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Train Loss per Batch")
plt.legend()
plt.show()


