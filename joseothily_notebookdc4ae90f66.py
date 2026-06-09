import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from transformers import AutoFeatureExtractor, AutoModel

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# from huggingface_hub import snapshot_download
# from insightface.app import FaceAnalysis
# import numpy as np
# import cv2

# snapshot_download(
#     "fal/AuraFace-v1",
#     local_dir="models/auraface",
# )
# face_app = FaceAnalysis(
#     name="auraface",
#     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
#     root=".",
# )

# input_image = cv2.imread("test.png")

# cv2_image = np.array(input_image.convert("RGB"))

# cv2_image = cv2_image[:, :, ::-1]
# faces = face_app.get(cv2_image)
# embedding = faces[0].normed_embedding



DATA_DIR = "/kaggle/input/anip-reconnaissance-faciale-estimation-ages-ocr/dataset_tache_1/dataset_tache_1"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "test")

assert os.path.isdir(TRAIN_IMG_DIR), f"Train images not found: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Test images not found: {TEST_IMG_DIR}"


import os
import random
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from albumentations import Compose, OneOf, Rotate, Affine, Perspective, RandomBrightnessContrast, HueSaturationValue, GaussianBlur, MotionBlur

# — Configuration — #

IMG_DIR = "/kaggle/input/anip-reconnaissance-faciale-estimation-ages-ocr/dataset_tache_1/dataset_tache_1/train"
AUG_DIR = "/kaggle/working/aug_images"

AUGMENT_RATE = 0.45
MIN_VARIANTS = 1
MAX_VARIANTS = 4
MASK_PROB = 0.5
MIN_TRANSFORMS = 1
MAX_TRANSFORMS = 3
MASK_OPTIONS = [
    "remove_top_third", "remove_bottom_third",
    "remove_left_half", "remove_right_half",
    "remove_eyes", "remove_chin", "remove_forehead"
]
MASK_MODES = ["black", "blur", "mix"]

TARGET_SIZE = (256, 256)  # width, height

# --- segmentation model setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)
model.eval()

LABEL_MAP = {
    "skin": [1],
    "nose": [2],
    "eye_g": [3],
    "left_eye": [4],
    "right_eye": [5],
    "l_brow": [6],
    "r_brow": [7],
    "mouth": [10],
    "u_lip": [11],
    "l_lip": [12],
    "hair": [13],
    "hat": [14],
    "ear_l": [8],
    "ear_r": [9],
    "neck_l": [16],
    "neck": [17],
    "cloth": [18],
}

def get_label_map(image: np.ndarray) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    up = torch.nn.functional.interpolate(
        logits, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    label_map = torch.argmax(up[0], dim=0).cpu().numpy()
    return label_map

def make_face_mask(label_map: np.ndarray) -> np.ndarray:
    face_labels = []
    for region, labels in LABEL_MAP.items():
        if region in ["skin", "nose", "left_eye", "right_eye", "mouth", "u_lip", "l_lip", "l_brow", "r_brow"]:
            face_labels.extend(labels)
    mask = np.isin(label_map, face_labels).astype(np.uint8)
    return mask

def get_region_mask_by_option(label_map: np.ndarray, face_mask: np.ndarray, option: str) -> np.ndarray:
    h, w = label_map.shape
    mask_vis = face_mask.copy()
    # (your existing logic for removing top, bottom, etc.)
    ys, xs = np.where(face_mask == 1)
    if option == "remove_top_third":
        if len(ys) > 0:
            y0, y1 = ys.min(), ys.max()
            tier = (y1 - y0 + 1) / 4.0
            boundary1 = int(y0 + tier)
            mask_vis[y0:boundary1, :] = 0
    elif option == "remove_bottom_third":
        if len(ys) > 0:
            y0, y1 = ys.min(), ys.max()
            tier = (y1 - y0 + 1) / 4.0
            boundary2 = int(y0 + 2 * tier)
            mask_vis[boundary2 : y1 + 1, :] = 0
    elif option == "remove_left_half":
        mid = w // 3
        mask_vis[:, :mid] = 0
    elif option == "remove_right_half":
        mid = w // 3
        mask_vis[:, mid:] = 0
    elif option == "remove_eyes":
        eye_labels = LABEL_MAP["left_eye"] + LABEL_MAP["right_eye"]
        eye_mask = np.isin(label_map, eye_labels)
        mask_vis[eye_mask] = 0
    elif option == "remove_chin":
        if len(ys) > 0:
            y0, y1 = ys.min(), ys.max()
            tier = (y1 - y0 + 1) / 3.0
            boundary2 = int(y0 + 2 * tier)
            mask_vis[boundary2 : y1 + 1, :] = 0
    elif option == "remove_forehead":
        if len(ys) > 0:
            y0, y1 = ys.min(), ys.max()
            tier = (y1 - y0 + 1) / 3.0
            boundary1 = int(y0 + tier)
            mask_vis[y0:boundary1, :] = 0
    return mask_vis

def apply_mask_to_image(img: np.ndarray, mask_vis: np.ndarray, mode="black", blur_kernel=(15, 15)) -> np.ndarray:
    masked = img.copy()
    mask_hide = (mask_vis == 0)
    if mode == "black":
        masked[mask_hide] = 0
    elif mode == "blur":
        blurred = cv2.GaussianBlur(img, blur_kernel, 0)
        masked[mask_hide] = blurred[mask_hide]
    elif mode == "mix":
        alpha = 0.5
        blurred = cv2.GaussianBlur(img, blur_kernel, 0)
        masked = np.where(
            mask_hide[:, :, None],
            (alpha * blurred + (1 - alpha) * img).astype(np.uint8),
            img,
        )
    return masked

base_augment = Compose([
    OneOf([
        Rotate(limit=15, border_mode=cv2.BORDER_REFLECT),
        Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.9, 1.1)),
        Perspective(scale=(0.05, 0.15))
    ], p=0.6),
    OneOf([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15),
    ], p=0.5),
    OneOf([GaussianBlur(blur_limit=3), MotionBlur(blur_limit=5)], p=0.3),
])

def resize_or_pad_to_target(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize (or optionally pad/crop) to the target size."""
    h, w = img.shape[:2]
    tw, th = target_size  # target width, height

    # If already target size, nothing to do
    if (w, h) == (tw, th):
        return img

    # Simple resize (may distort aspect ratio)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return img_resized

    # If you prefer preserving aspect ratio with padding, you can implement that instead:
    """
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # pad to target
    pad_w = tw - new_w
    pad_h = th - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded
    """

def augment_image_with_occlusion(img: np.ndarray) -> np.ndarray:
    # First, optionally resize input to target size to standardize pipeline
    # img = resize_or_pad_to_target(img, TARGET_SIZE)

    n_trans = random.randint(MIN_TRANSFORMS, MAX_TRANSFORMS)
    aug = img.copy()
    for _ in range(n_trans):
        aug = base_augment(image=aug)["image"]
    if random.random() < MASK_PROB:
        label_map = get_label_map(aug)
        face_mask = make_face_mask(label_map)
        option = random.choice(MASK_OPTIONS)
        mask_vis = get_region_mask_by_option(label_map, face_mask, option)
        mode = random.choice(MASK_MODES)
        aug = apply_mask_to_image(aug, mask_vis, mode=mode)

    # Final resizing to ensure exactly target size
    # aug = resize_or_pad_to_target(aug, TARGET_SIZE)
    return aug

def augment_row(row,
                img_dir=IMG_DIR,
                aug_dir=AUG_DIR,
                augment_rate=AUGMENT_RATE,
                min_variants=MIN_VARIANTS,
                max_variants=MAX_VARIANTS):
    augmented_rows = []
    augmented_rows.append({
        "img1": row["img1"],
        "img2": row["img2"],
        "label": row["label"]
    })

    if random.random() < augment_rate:
        n_var = random.randint(min_variants, max_variants)
        for v in range(n_var):
            do1 = (random.random() < 0.5)
            do2 = (random.random() < 0.5)
            if not (do1 or do2):
                do1 = True

            new1 = row["img1"]
            new2 = row["img2"]

            if do1:
                path1 = os.path.join(img_dir, row["img1"])
                img1 = cv2.imread(path1)
                if img1 is not None:
                    aug1 = augment_image_with_occlusion(img1)
                    base, ext = os.path.splitext(row["img1"])
                    new_fn1 = f"{base}_aug{v}{ext}"
                    os.makedirs(aug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(aug_dir, new_fn1), aug1)
                    new1 = new_fn1

            if do2:
                path2 = os.path.join(img_dir, row["img2"])
                img2 = cv2.imread(path2)
                if img2 is not None:
                    aug2 = augment_image_with_occlusion(img2)
                    base, ext = os.path.splitext(row["img2"])
                    new_fn2 = f"{base}_aug{v}{ext}"
                    os.makedirs(aug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(aug_dir, new_fn2), aug2)
                    new2 = new_fn2

            augmented_rows.append({
                "img1": new1,
                "img2": new2,
                "label": row["label"]
            })

    return augmented_rows


def get_id_from_filename(fn):
    # Ex : "001_1.jpg" ou "001_2.png" -> renvoie "001"
    return os.path.splitext(fn)[0].rsplit("_", 1)[0]

# Lister tous les fichiers dans le dossier d’entraînement
train_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if os.path.isfile(os.path.join(TRAIN_IMG_DIR, f))])

# Regrouper par ID
ids = {}
for fn in train_files:
    id_ = get_id_from_filename(fn)
    ids.setdefault(id_, []).append(fn)

# Garde seulement les IDs avec exactement ou au moins 2 fichiers
ids = {k: v for k, v in ids.items() if len(v) >= 2}
print("Nombre d’IDs valides :", len(ids))

# Créer la liste des paires (img1, img2), sans label pour l’instant
pairs = []
for k, imgs in sorted(ids.items()):
    imgs = sorted(imgs)
    # On suppose que pour chaque ID il y a exactement 2 images (ex : "001_1", "001_2")
    pairs.append((imgs[0], imgs[1]))

# Maintenant appliquer le schéma : 200 premières paires → positives, 200 suivantes → négatives, etc.
pairs_labeled = []
cycle = 400
half = 200

for idx, (fn1, fn2) in enumerate(pairs):
    mod = idx % cycle
    if mod < half:
        label = 1
    else:
        label = 0
    pairs_labeled.append((fn1, fn2, label))

pairs_df = pd.DataFrame(pairs_labeled, columns=["img1", "img2", "label"])
print("Total paires avec labels:", len(pairs_df))
print(pairs_df["label"].value_counts())




# train_df est ton DataFrame avec colonnes "img1", "img2", "label"

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(pairs_df, test_size=0.15, stratify=pairs_df["label"], random_state=42)

list_of_dfs = []
for _, row in train_df.iterrows():
    list_of_dfs.append(pd.DataFrame(augment_row(row)))

train_df = pd.concat(list_of_dfs, ignore_index=True)
train_df = train_df.reset_index(drop=True)




# 4. Split train / val etc.




train_df


train_df["label"].value_counts()



import matplotlib.pyplot as plt
from PIL import Image

import os
import matplotlib.pyplot as plt
from PIL import Image

def show_pairs_from_df(df_pairs, img_dir, aug_dir=None, n_pairs=10):
    """
    Affiche les n_pairs premières entrées de df_pairs,
    chaque paire (img1, img2) côte à côte.
    Si une image n’est pas trouvée dans img_dir et qu’aug_dir est donné,
    essaie de la charger depuis aug_dir.
    """
    n = min(n_pairs, len(df_pairs))
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    for i in range(n):
        row = df_pairs.iloc[i]
        fn1 = row["img1"]
        fn2 = row["img2"]
        lbl = row["label"]

        # fonction utilitaire pour charger image depuis img_dir ou aug_dir
        def load_img(fn):
            path1 = os.path.join(img_dir, fn)
            if os.path.exists(path1):
                return Image.open(path1).convert("RGB")
            if aug_dir is not None:
                path2 = os.path.join(aug_dir, fn)
                if os.path.exists(path2):
                    return Image.open(path2).convert("RGB")
            # Si pas trouvé, lever erreur ou retourner une image vide
            raise FileNotFoundError(f"Image '{fn}' non trouvée ni dans '{img_dir}' ni dans '{aug_dir}'")

        img1 = load_img(fn1)
        img2 = load_img(fn2)

        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        ax1.imshow(img1)
        ax1.axis("off")
        ax1.set_title(fn1)
        ax2.imshow(img2)
        ax2.axis("off")
        ax2.set_title(f"{fn2}\nLabel = {lbl}")

    plt.tight_layout()
    plt.show()


# Exemple d’utilisation avec train_df :
show_pairs_from_df(train_df, TRAIN_IMG_DIR, aug_dir="/kaggle/working/aug_images", n_pairs=20)






MODEL_NAME = "google/vit-base-patch16-224-in21k"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
vit = AutoModel.from_pretrained(MODEL_NAME).to(device)

# On peut geler le modèle ou certaines couches
vit.eval()
for p in vit.parameters():
    p.requires_grad = False

print("Loaded ViT backbone:", MODEL_NAME)


class FacePairDataset(Dataset):
    def __init__(self, df, img_dir, feature_extractor, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_image(self, fn):
        path1 = os.path.join(self.img_dir, fn)
        if os.path.exists(path1):
            img = Image.open(path1).convert("RGB")
            return img
        else :
            path2 = os.path.join("/kaggle/working/aug_images", fn)
            img = Image.open(path2).convert("RGB")
            return img
     

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = self.load_image(row["img1"])
        img2 = self.load_image(row["img2"])
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # feature_extractor expects a PIL image (or list), returns dict with 'pixel_values'
        enc1 = self.feature_extractor(images=img1, return_tensors="pt")
        enc2 = self.feature_extractor(images=img2, return_tensors="pt")
        pv1 = enc1["pixel_values"].squeeze(0)
        pv2 = enc2["pixel_values"].squeeze(0)
        label = torch.tensor(row["label"], dtype=torch.float)
        return {"pv1": pv1, "pv2": pv2, "label": label}


size = (224, 224)
transform = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(p=0.5),
    # éventuellement ColorJitter, etc.
])

train_dataset = FacePairDataset(train_df, TRAIN_IMG_DIR, feature_extractor, transform=transform)
val_dataset   = FacePairDataset(val_df, TRAIN_IMG_DIR, feature_extractor, transform=T.Resize(size))

train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False,  pin_memory=True)


@torch.no_grad()
def embed_from_pv(pv):
    out = vit(pixel_values=pv.to(device))
    # out.last_hidden_state shape (B, seq_len, hidden_size)
    # Le token [CLS] est à l'index 0
    cls = out.last_hidden_state[:, 0, :]
    return cls.cpu()



with torch.no_grad():
    dummy = torch.zeros((1, ) + train_dataset[0]["pv1"].shape).to(device)
    emb_dim = embed_from_pv(dummy).shape[-1]
print("Embedding dimension:", emb_dim)

class ComparisonHead(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
            
        )
    def forward(self, e1, e2):
        x = torch.cat([e1, e2], dim=1)
        return self.net(x).squeeze(1)

head = ComparisonHead(emb_dim).to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


best_auc = 0.0
best_path = "best_head_vit.pt"

for epoch in range(10):
    head.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
        pv1 = batch["pv1"]
        pv2 = batch["pv2"]
        lbl = batch["label"].to(device)
        e1 = embed_from_pv(pv1)
        e2 = embed_from_pv(pv2)
        optimizer.zero_grad()
        logits = head(e1.to(device), e2.to(device))

        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * e1.size(0)
    scheduler.step()
    avg_loss = total_loss / len(train_loader.dataset)
    print("Train loss:", avg_loss)

    # validation
    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            e1 = embed_from_pv(batch["pv1"])
            e2 = embed_from_pv(batch["pv2"])
            logits = head(e1.to(device), e2.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            preds.extend(probs)
            trues.extend(batch["label"].numpy().tolist())
    auc = roc_auc_score(trues, preds)
    pred_bin = [1 if p >= 0.5 else 0 for p in preds]
    acc = accuracy_score(trues, pred_bin)
    print(f"Val AUC: {auc:.4f}, Acc: {acc:.4f}")
    if auc > best_auc:
        best_auc = auc
        torch.save(head.state_dict(), best_path)
        print("=> Best head saved")

print("Best validation AUC:", best_auc)
head.load_state_dict(torch.load(best_path, map_location=device))
head.eval()


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import ViTImageProcessor, ViTModel

# ========== Configurations ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEAD_CHECKPOINT = "/kaggle/working/best_head_vit.pt"
IMAGE_FOLDER = "/kaggle/input/anip-reconnaissance-faciale-estimation-ages-ocr/dataset_tache_1/dataset_tache_1/test"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True

# ========== Modèles ==========

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()
vit_model.to(device)

# Définis la tête; ici un exemple, remplace-la par ta vraie implémentation


# ========== Dataset pour le chargement des images ==========

class ImageDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        # inputs sont des tensors de forme (1, …), on retire la dimension batch
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return path, inputs

def collect_image_paths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            _, ext = os.path.splitext(fname.lower())
            if ext in IMG_EXTS:
                paths.append(os.path.join(root, fname))
    return paths

# ========== Extraction des embeddings en lot ==========

def extract_all_embeddings(image_paths):
    dataset = ImageDataset(image_paths, processor)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY,
                        collate_fn=lambda batch: batch)
    emb_cache = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract embeddings"):
            paths = [item[0] for item in batch]
            # on doit regrouper les inputs pour les passer au modèle
            batched = {}
            for k in batch[0][1].keys():
                batched[k] = torch.stack([item[1][k] for item in batch], dim=0).to(device)
            outputs = vit_model(**batched)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # token [CLS]
            for i, p in enumerate(paths):
                emb_cache[p] = cls_emb[i].detach().cpu()
    return emb_cache

# ========== Comparaison via la tête ==========

def find_best_matches_from_embeddings(emb_cache):
    image_paths = list(emb_cache.keys())
    n = len(image_paths)
    emb_list = [emb_cache[p].to(device) for p in image_paths]
    emb_tensor = torch.stack(emb_list, dim=0)  # shape (n, dim)

    best_matches = {}
    with torch.no_grad():
        for i in tqdm(range(n), desc="Finding best match pairs"):
            e1 = emb_tensor[i : i + 1, :]  # (1, dim)
            e1_expand = e1.repeat(n, 1)
            e2_all = emb_tensor
            logits = head(e1_expand, e2_all) 
            torch.sigmoid(logits).cpu().numpy().tolist()
            # print("logits.shape:", logits.shape)# (n, 1)
            probs = torch.sigmoid(logits)  # (n,)
            # print(probs)
            probs[i] = -1.0  # éviter le self

            max_idx = torch.argmax(probs).item()
            best_prob = probs[max_idx].item()
            best_path = image_paths[max_idx]
            best_matches[image_paths[i]] = (best_path, best_prob)
    return best_matches

# ========== Script principal ==========
dim_embedding = vit_model.config.hidden_size
head.load_state_dict(torch.load(HEAD_CHECKPOINT, map_location=device))
head.eval()

image_paths = collect_image_paths(IMAGE_FOLDER)
if len(image_paths) < 2:
    print("Pas assez d’images pour comparer.")

# 1. extraire tous les embeddings
emb_cache = extract_all_embeddings(image_paths)

# 2. trouver les meilleures correspondances
best_matches = find_best_matches_from_embeddings(emb_cache)








# best_matches est : dict image → (best_match, prob)
# Création d’un DataFrame
data = []
for img, (match, prob) in best_matches.items():
    data.append({
        "image": img,
        "best_match": match,
        "probability": prob
    })

df = pd.DataFrame(data)
# Optionnel : trier par probabilité décroissante

print(df.head())  # affiche les 5 premières lignes



import matplotlib.pyplot as plt
from PIL import Image

def show_pair_from_df(row):
    img_path = row["image"]
    match_path = row["best_match"]
    prob = row["probability"]

    img1 = Image.open(img_path).convert("RGB")
    img2 = Image.open(match_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img1)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title(f"Match (prob = {prob:.3f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# Exemple : afficher les top 5 paires les plus probables
for idx, row in df.head(5).iterrows():
    show_pair_from_df(row)



from numpy.linalg import norm
def cos_sim(a, b):
    return float(np.dot(a, b) / (norm(a)*norm(b) + 1e-8))

def head_score(a, b):
    ta = torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(device)
    tb = torch.tensor(b, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = head(ta, tb).item()
    return 1 / (1 + np.exp(-logit))

matches = []
for tfn, temb in test_emb.items():
    sims = [(fn, cos_sim(temb, train_emb[fn])) for fn in train_emb.keys()]
    best_fn, best_sim = max(sims, key=lambda x: x[1])
    prob = head_score(temb, train_emb[best_fn])
    matches.append({
        "test_image": tfn,
        "train_image": best_fn,
        "train_id": get_id_from_filename(best_fn),
        "cosine": best_sim,
        "head_prob": prob
    })
df_match = pd.DataFrame(matches)
df_match.to_csv("submission_vit_test2train.csv", index=False)
print("Submission file generated: submission_vit_test2train.csv")




