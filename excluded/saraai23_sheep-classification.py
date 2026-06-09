import os, random, copy
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import imagehash


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_WORKERS = 2
NUM_CLASSES = 7
EPOCHS = 20
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.makedirs("models", exist_ok=True)


INPUT_DIR = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
df = pd.read_csv(LABELS_PATH)


le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])


def is_blurry(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

clean_filenames = []
for fn in df['filename']:
    path = os.path.join(TRAIN_DIR, fn)
    img = cv2.imread(path)
    if img is not None and not is_blurry(img):
        clean_filenames.append(fn)
df = df[df['filename'].isin(clean_filenames)].reset_index(drop=True)


hashes = {}
duplicates = []
for fn in tqdm(df['filename']):
    path = os.path.join(TRAIN_DIR, fn)
    img = Image.open(path)
    h = str(imagehash.phash(img))
    if h in hashes:
        duplicates.append(fn)
    else:
        hashes[h] = fn
df = df[~df['filename'].isin(duplicates)].reset_index(drop=True)


counts = Counter(df['label_enc'])
max_count = max(counts.values())
balanced_extra = []
for lbl, cnt in counts.items():
    if cnt < max_count:
        times = (max_count - cnt) // cnt + 1
        grp = df[df['label_enc'] == lbl]
        new_rows = [grp.copy() for _ in range(times)]
        balanced_extra.append(pd.concat(new_rows, ignore_index=True))
if balanced_extra:
    df = pd.concat([df] + balanced_extra, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def crop_sheep(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image


class SheepDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        fn = self.df.loc[idx, 'filename']
        img = np.array(Image.open(os.path.join(self.img_dir, fn)).convert("RGB"))
        img = crop_sheep(img)
        if self.transform:
            img = self.transform(image=img)['image']
        label = self.df.loc[idx, 'label_enc']
        return img, label


train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=25, p=0.6),
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])


class_weights = compute_class_weight('balanced', classes=np.unique(df['label_enc']), y=df['label_enc'])
weights = torch.tensor(class_weights, dtype=torch.float)


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion):
    model.eval()
    running_loss, all_preds, all_targets = 0, [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='macro')
    return running_loss / len(dataloader), f1


from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
import os, copy
import torch.nn as nn

def create_model(model_name, num_classes, freeze=True):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    return model

model_names = ["deit_base_patch16_224", "swin_base_patch4_window7_224", "convnext_base"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

for model_name in model_names:
    print(f"\n========== Training: {model_name} ==========\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_enc']), 1):
        print(f"Training Fold {fold}")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = SheepDataset(train_df, TRAIN_DIR, train_transforms)
        val_ds = SheepDataset(val_df, TRAIN_DIR, val_transforms)

        class_weight_map = {i: w for i, w in enumerate(class_weights)}
        sample_weights = train_df['label_enc'].map(class_weight_map).values
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = create_model(model_name, NUM_CLASSES, freeze=True).to(device)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_f1 = 0
        patience = 0

        for epoch in range(1, 8):  # Freeze phase
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_f1 = validate(model, val_loader, criterion)
            scheduler.step()
            print(f"[{model_name}] Fold {fold} | Freeze Epoch {epoch} | Train Acc: {train_acc:.3f} | Val F1: {val_f1:.3f}")
            if val_f1 > best_f1:
                best_state = copy.deepcopy(model.state_dict())
                best_f1 = val_f1
                patience = 0
            else:
                patience += 1
            if patience >= 4:
                break

        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 7)
        patience = 0

        for epoch in range(8, EPOCHS + 1):  # Unfreeze phase
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_f1 = validate(model, val_loader, criterion)
            scheduler.step()
            print(f"[{model_name}] Fold {fold} | Unfreeze Epoch {epoch} | Train Acc: {train_acc:.3f} | Val F1: {val_f1:.3f}")
            if val_f1 > best_f1:
                best_state = copy.deepcopy(model.state_dict())
                best_f1 = val_f1
                patience = 0
            else:
                patience += 1
            if patience >= 4:
                break

        save_path = f"models/{model_name}"
        os.makedirs(save_path, exist_ok=True)
        torch.save(best_state, f"{save_path}/best_model_fold{fold}_f1_{best_f1:.4f}.pt")


import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

tta_transforms = [
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomBrightnessContrast(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
]

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = TEST_DIR
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.loc[idx, 'filename']
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = crop_sheep(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img

test_df = pd.DataFrame({'filename': sorted(os.listdir(TEST_DIR))})

model_names = ["deit_base_patch16_224", "swin_base_patch4_window7_224", "convnext_base"]
all_preds = []

for model_name in model_names:
    for fold in range(1, 6):
        model_path = glob.glob(f"models/{model_name}/best_model_fold{fold}_f1_*.pt")[0]
        print(f"Loading model: {model_path}")
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        tta_preds = []

        for tta in tta_transforms:
            dataset = TestDataset(test_df, transform=tta)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            preds = []
            with torch.no_grad():
                for images in loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    preds.append(probs.cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            tta_preds.append(preds)

        fold_avg = np.mean(np.array(tta_preds), axis=0)
        all_preds.append(fold_avg)

final_avg = np.mean(np.array(all_preds), axis=0)
final_labels = final_avg.argmax(axis=1)

test_df['label'] = le.inverse_transform(final_labels)
test_df[['filename', 'label']].to_csv("submission_all.csv", index=False)
print("âœ… Predictions complete. Results saved to submission.csv")


import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

tta_transforms = [
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomBrightnessContrast(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
]

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = TEST_DIR
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.loc[idx, 'filename']
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = crop_sheep(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img

test_df = pd.DataFrame({'filename': sorted(os.listdir(TEST_DIR))})
all_preds = []

model_name = "swin_base_patch4_window7_224"

for fold in range(1, 6):
    model_path = glob.glob(f"models/{model_name}/best_model_fold{fold}_f1_*.pt")[0]
    print(f"Loading model: {model_path}")
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tta_preds = []

    for tta in tta_transforms:
        dataset = TestDataset(test_df, transform=tta)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        preds = []
        with torch.no_grad():
            for images in loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tta_preds.append(preds)

    fold_avg = np.mean(np.array(tta_preds), axis=0)
    all_preds.append(fold_avg)

final_avg = np.mean(np.array(all_preds), axis=0)
final_labels = final_avg.argmax(axis=1)

test_df['label'] = le.inverse_transform(final_labels)
test_df[['filename', 'label']].to_csv("submission_swin_only.csv", index=False)
print("âœ… Done. Predictions saved to submission_swin_only.csv")


import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CONFIDENCE_THRESHOLD = 0.90
pseudo_images = []
pseudo_labels = []

tta_transforms = [
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomBrightnessContrast(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
]

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = TEST_DIR
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.loc[idx, 'filename']
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = crop_sheep(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img

test_df = pd.DataFrame({'filename': sorted(os.listdir(TEST_DIR))})

model_name = "swin_base_patch4_window7_224"
all_fold_preds = []

for fold in range(1, 6):
    model_path = glob.glob(f"models/{model_name}/best_model_fold{fold}_f1_*.pt")[0]
    print(f"Loading fold {fold} from {model_path}")

    swin_model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    swin_model.load_state_dict(torch.load(model_path, map_location=device))
    swin_model.to(device)
    swin_model.eval()

    tta_preds = []

    for tta in tta_transforms:
        dataset = TestDataset(test_df, transform=tta)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        preds = []
        with torch.no_grad():
            for images in loader:
                images = images.to(device)
                outputs = swin_model(images)
                probs = F.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tta_preds.append(preds)

    fold_avg = np.mean(np.array(tta_preds), axis=0)
    all_fold_preds.append(fold_avg)

avg_preds = np.mean(np.array(all_fold_preds), axis=0)

for i, probs in enumerate(avg_preds):
    label = probs.argmax()
    confidence = probs[label]
    if confidence >= CONFIDENCE_THRESHOLD:
        pseudo_images.append(test_df.loc[i, "filename"])
        pseudo_labels.append(label)

pseudo_df = pd.DataFrame({
    'filename': pseudo_images,
    'label_enc': pseudo_labels,
    'label': le.inverse_transform(pseudo_labels)
})

print(f"Total confident pseudo-labels: {len(pseudo_df)}")


combined_df = pd.concat([df[['filename', 'label_enc', 'label']], pseudo_df], ignore_index=True)


import glob
import torch.nn.functional as F

tta_transforms = [
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomBrightnessContrast(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
]

class SheepDataset(Dataset):
    def __init__(self, df, train_dir, test_dir=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.train_dir = train_dir
        self.test_dir = test_dir if test_dir is not None else train_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, 'filename']
        
        img_path = os.path.join(
            self.test_dir if fn in os.listdir(self.test_dir) else self.train_dir,
            fn
        )

        img = np.array(Image.open(img_path).convert("RGB"))
        img = crop_sheep(img)
        if self.transform:
            img = self.transform(image=img)['image']
        label = self.df.loc[idx, 'label_enc']
        return img, label

test_df = pd.DataFrame({'filename': sorted(os.listdir(TEST_DIR))})
all_preds = []

for fold in range(1, 6):
    model_path = glob.glob(f"models/{model_name}/best_model_fold{fold}_f1_*.pt")[0]
    print(f"Loading fold {fold} from {model_path}")
    swin_model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    swin_model.load_state_dict(torch.load(model_path, map_location=device))
    swin_model.to(device)
    swin_model.eval()


    tta_preds = []

    for tta in tta_transforms:
        dataset = TestDataset(test_df, transform=tta)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        preds = []
        with torch.no_grad():
            for images in loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tta_preds.append(preds)

    fold_avg = np.mean(np.array(tta_preds), axis=0)
    all_preds.append(fold_avg)

final_avg = np.mean(np.array(all_preds), axis=0)
final_labels = final_avg.argmax(axis=1)
test_df['label'] = le.inverse_transform(final_labels)


max_probs = final_avg.max(axis=1)
confidence_threshold = 0.95
pseudo_indices = np.where(max_probs >= confidence_threshold)[0]

pseudo_filenames = test_df.iloc[pseudo_indices]['filename'].values
pseudo_labels = final_labels[pseudo_indices]

pseudo_df = pd.DataFrame({
    'filename': pseudo_filenames,
    'label_enc': pseudo_labels,
    'label': le.inverse_transform(pseudo_labels)
})


os.makedirs("pseudo_models", exist_ok=True)
            
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

for fold, (train_idx, val_idx) in enumerate(skf.split(augmented_df, augmented_df['label_enc']), 1):
    print(f"Training Fold {fold}")
    train_df, val_df = augmented_df.iloc[train_idx], augmented_df.iloc[val_idx]
    
    train_ds = SheepDataset(train_df, TRAIN_DIR, test_dir=TEST_DIR, transform=train_transforms)
    val_ds = SheepDataset(val_df, TRAIN_DIR, test_dir=TEST_DIR, transform=val_transforms)

    class_weight_map = {i: w for i, w in enumerate(class_weights)}
    sample_weights = train_df['label_enc'].map(class_weight_map).values
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = create_model(NUM_CLASSES, freeze=True).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0
    patience = 0

    for epoch in range(1, 8):  # Freeze phase
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1 = validate(model, val_loader, criterion)
        scheduler.step()
        print(f"[Fold {fold}] Freeze Epoch {epoch} | Train Acc: {train_acc:.3f} | Val F1: {val_f1:.3f}")
        if val_f1 > best_f1:
            best_state = copy.deepcopy(model.state_dict())
            best_f1 = val_f1
            patience = 0
        else:
            patience += 1
        if patience >= 4:
            break

    # Unfreeze phase
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 7)
    patience = 0

    for epoch in range(8, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1 = validate(model, val_loader, criterion)
        scheduler.step()
        print(f"[Fold {fold}] Unfreeze Epoch {epoch} | Train Acc: {train_acc:.3f} | Val F1: {val_f1:.3f}")
        if val_f1 > best_f1:
            best_state = copy.deepcopy(model.state_dict())
            best_f1 = val_f1
            patience = 0
        else:
            patience += 1
        if patience >= 4:
            break



    torch.save(best_state, f"pseudo_models/pseudo_best_model_fold{fold}_f1_{best_f1:.4f}.pt")


augmented_df = pd.concat([df[['filename', 'label_enc', 'label']], pseudo_df], ignore_index=True)


def create_model(num_classes, freeze=True):
    
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=NUM_CLASSES)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    return model


import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

tta_transforms = [
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomBrightnessContrast(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]),
    A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
]

class SheepDataset(Dataset):
    def __init__(self, df, train_dir, test_dir=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.train_dir = train_dir
        self.test_dir = test_dir if test_dir is not None else train_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, 'filename']
        img_path = os.path.join(
            self.test_dir if fn in os.listdir(self.test_dir) else self.train_dir,
            fn
        )
        img = np.array(Image.open(img_path).convert("RGB"))
        img = crop_sheep(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img

test_df = pd.DataFrame({'filename': sorted(os.listdir(TEST_DIR))})
all_preds = []

for fold in range(1, 6):
    model_path = glob.glob(f"pseudo_models/pseudo_best_model_fold{fold}_f1_*.pt")[0]
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tta_preds = []

    for tta in tta_transforms:
        dataset = SheepDataset(test_df, TRAIN_DIR, test_dir=TEST_DIR, transform=tta)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        preds = []
        with torch.no_grad():
            for images in loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tta_preds.append(preds)

    fold_avg = np.mean(np.array(tta_preds), axis=0)
    all_preds.append(fold_avg)

final_avg = np.mean(np.array(all_preds), axis=0)
final_labels = final_avg.argmax(axis=1)
test_df['label'] = le.inverse_transform(final_labels)
test_df[['filename', 'label']].to_csv("submission_last.csv", index=False)
print("submission.csv saved using pseudo-trained model")

