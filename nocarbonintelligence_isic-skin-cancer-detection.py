import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os

# DÃ©finition de la racine du jeu de donnÃ©es
ROOT = "/kaggle/input/isic-2024-challenge"

print("--- Chargement des MÃ©tadonnÃ©es ---")
# Chargement des mÃ©tadonnÃ©es
train_df = pd.read_csv(os.path.join(ROOT, 'train-metadata.csv'))
test_df = pd.read_csv(os.path.join(ROOT, 'test-metadata.csv'))

# Affichage des dimensions
print(f"Train Metadata Shape: {train_df.shape}")
print(f"Test Metadata Shape: {test_df.shape}")

# Affichage des 5 premiÃ¨res lignes du DataFrame d'entraÃ®nement
print("\n--- AperÃ§u des MÃ©tadonnÃ©es d'EntraÃ®nement ---")
print(train_df.head())

# Affichage des informations sur les colonnes (type de donnÃ©es, valeurs manquantes)
print("\n--- Informations sur les Colonnes et les Types de DonnÃ©es ---")
train_df.info(verbose=False, memory_usage="deep")
print(f"Nombre de colonnes uniques dans les mÃ©tadonnÃ©es: {len(train_df.columns)}")


# --- 1. Analyse de la Variable Cible (Target) ---
print("\n--- 1. Analyse de la Variable Cible (Target) ---")
target_counts = train_df['target'].value_counts(normalize=True).mul(100).rename({0: 'Benign', 1: 'Malignant'})
print(target_counts)
plt.figure(figsize=(6, 4))
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title('Distribution de la Variable Cible (Target)')
plt.ylabel('Pourcentage')
plt.show()
# 


# --- 2. Analyse des DonnÃ©es Contextuelles (Patient) ---
print("\n--- 2. Analyse des DonnÃ©es Contextuelles (Patient) ---")
print(f"Nombre total de patients uniques dans le train: {train_df['patient_id'].nunique()}")
# Identifier le nombre moyen de lÃ©sions par patient
lesions_per_patient = train_df['patient_id'].value_counts()
print(f"LÃ©sions Max par patient: {lesions_per_patient.max()}")
print(f"LÃ©sions Min par patient: {lesions_per_patient.min()}")
print(f"LÃ©sions Moyennes par patient: {lesions_per_patient.mean():.2f}")


# --- 3. Gestion des Valeurs Manquantes ---
print("\n--- 3. Gestion des Valeurs Manquantes ---")
# Afficher le pourcentage de valeurs manquantes par colonne dans les mÃ©tadonnÃ©es
missing_vals = train_df.isnull().sum()
missing_perc = (missing_vals[missing_vals > 0] / len(train_df)) * 100
print(f"Nombre de colonnes avec des valeurs manquantes: {len(missing_perc)}")
print(missing_perc.sort_values(ascending=False).head(10))


# --- 4. VÃ©rification du Fichier HDF5 (Images/CaractÃ©ristiques) ---
print("\n--- 4. VÃ©rification des Fichiers HDF5 (CaractÃ©ristiques) ---")
# VÃ©rifier si les fichiers HDF5 existent
train_hdf5_path = os.path.join(ROOT, 'train-image.hdf5')
test_hdf5_path = os.path.join(ROOT, 'test-image.hdf5')

if os.path.exists(train_hdf5_path):
    # Ouvrir le fichier pour vÃ©rifier les clÃ©s
    with h5py.File(train_hdf5_path, 'r') as f:
        print(f"Le fichier HDF5 d'entraÃ®nement (train-image.hdf5) existe.")
        print(f"Exemple de clÃ© ISIC dans HDF5: {list(f.keys())[0]}")
        # Note: L'extraction rÃ©elle des images se fait Ã  l'Ã©tape de modÃ©lisation/chargement de donnÃ©es pour l'efficacitÃ©.

if os.path.exists(test_hdf5_path):
    print(f"Le fichier HDF5 de test (test-image.hdf5) existe (contient les 3 exemples initiaux).")

# --- 5. AperÃ§u des CaractÃ©ristiques Manuelles (LV features) ---
# Afficher les 5 colonnes de features 'tbp_lv_' les plus complÃ¨tes
lv_features = [col for col in train_df.columns if col.startswith('tbp_lv_')]
print(f"\nNombre de features 'tbp_lv_': {len(lv_features)}")
print("AperÃ§u des 5 premiÃ¨res colonnes de features 'tbp_lv_' :")
print(train_df[lv_features].head())

# Exemple d'analyse des mÃ©tadonnÃ©es
plt.figure(figsize=(10, 5))
sns.histplot(train_df['age_approx'].dropna(), bins=30, kde=True)
plt.title('Distribution d\'Ã¢ge approximatif')
plt.show()


import pandas as pd
import numpy as np
import os
import random
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# ============================================================
# 1. DÃ‰TERMINISME GLOBAL
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 2. CHARGEMENT DES DONNÃ‰ES
# ============================================================
ROOT = "/kaggle/input/isic-2024-challenge"
train_df = pd.read_csv(os.path.join(ROOT, "train-metadata.csv"), low_memory=False)

IMAGE_DIR = os.path.join(ROOT, 'train-image', 'image')
train_df['image_path'] = train_df['isic_id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))

print("--- PrÃ©-traitement DÃ©terministe & Sans Fuite Patient ---")

# ============================================================
# 3. SPLIT SANS FUITE (par patient_id)
# ============================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(gss.split(train_df, groups=train_df["patient_id"]))

train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
val_df_split   = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Taille Train : {len(train_df_split)}")
print(f"Taille Validation : {len(val_df_split)}")

# ============================================================
# 4. TBP FEATURES
# ============================================================
TBP_LV_NUMERIC = [
    "tbp_lv_A","tbp_lv_Aex","tbp_lv_B","tbp_lv_Bext","tbp_lv_C","tbp_lv_Cext",
    "tbp_lv_H","tbp_lv_Hext","tbp_lv_L","tbp_lv_Lext","tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio","tbp_lv_color_std_mean","tbp_lv_deltaA",
    "tbp_lv_deltaB","tbp_lv_deltaL","tbp_lv_deltaLBnorm","tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM","tbp_lv_nevi_confidence","tbp_lv_norm_border",
    "tbp_lv_norm_color","tbp_lv_perimeterMM","tbp_lv_radial_color_std_max",
    "tbp_lv_stdL","tbp_lv_stdLExt","tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle","tbp_lv_x","tbp_lv_y","tbp_lv_z"
]
TBP_LV_CATEGORICAL = ["tbp_lv_location","tbp_lv_location_simple"]

TBP_LV_NUMERIC = [c for c in TBP_LV_NUMERIC if c in train_df_split.columns]
TBP_LV_CATEGORICAL = [c for c in TBP_LV_CATEGORICAL if c in train_df_split.columns]

NUMERIC_IMPUTE_COLS = [
    "age_approx", "clin_size_long_diam_mm", "mel_mitotic_index", "mel_thick_mm"
] + TBP_LV_NUMERIC

CATEGORICAL_COLS = [
    "sex", "anatom_site_general", "image_type", "tbp_tile_type"
] + TBP_LV_CATEGORICAL

# ============================================================
# 5. INSPECTION ET NETTOYAGE NUMÃ‰RIQUE
# ============================================================
print("\nğŸ”� Inspection colonnes numÃ©riques avant imputation...")
NUMERIC_IMPUTE_COLS_EXIST = []
for col in NUMERIC_IMPUTE_COLS:
    if col not in train_df_split.columns:
        continue
    # Conversion forcÃ©e
    train_df_split[col] = pd.to_numeric(train_df_split[col], errors="coerce")
    val_df_split[col]   = pd.to_numeric(val_df_split[col], errors="coerce")
    
    # VÃ©rification si colonne est vraiment numÃ©rique
    if np.issubdtype(train_df_split[col].dtype, np.number):
        NUMERIC_IMPUTE_COLS_EXIST.append(col)

# Supprimer les doublons et colonnes entiÃ¨rement NaN
NUMERIC_IMPUTE_COLS_EXIST = [c for c in dict.fromkeys(NUMERIC_IMPUTE_COLS_EXIST) 
                             if train_df_split[c].notna().sum() > 0]

print(f"Colonnes retenues pour imputation : {len(NUMERIC_IMPUTE_COLS_EXIST)}")
print(NUMERIC_IMPUTE_COLS_EXIST)

# ============================================================
# 6. IMPUTATION NUMÃ‰RIQUE
# ============================================================
num_imputer = SimpleImputer(strategy="median")

train_df_split[NUMERIC_IMPUTE_COLS_EXIST] = pd.DataFrame(
    num_imputer.fit_transform(train_df_split[NUMERIC_IMPUTE_COLS_EXIST]),
    columns=NUMERIC_IMPUTE_COLS_EXIST,
    index=train_df_split.index
)

val_df_split[NUMERIC_IMPUTE_COLS_EXIST] = pd.DataFrame(
    num_imputer.transform(val_df_split[NUMERIC_IMPUTE_COLS_EXIST]),
    columns=NUMERIC_IMPUTE_COLS_EXIST,
    index=val_df_split.index
)

print("âœ… Imputation numÃ©rique OK")

# ============================================================
# 7. ENCODAGE CATÃ‰GORIEL
# ============================================================
for col in CATEGORICAL_COLS:
    train_df_split[col] = train_df_split[col].astype(str).fillna("missing")
    val_df_split[col]   = val_df_split[col].astype(str).fillna("missing")

ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_df_split[CATEGORICAL_COLS] = ord_encoder.fit_transform(train_df_split[CATEGORICAL_COLS])
val_df_split[CATEGORICAL_COLS]   = ord_encoder.transform(val_df_split[CATEGORICAL_COLS])

# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================
train_df_split["has_lesion_id"] = train_df_split["lesion_id"].notna().astype(int)
val_df_split["has_lesion_id"]   = val_df_split["lesion_id"].notna().astype(int)

train_df_split.drop(columns=["lesion_id"], errors="ignore", inplace=True)
val_df_split.drop(columns=["lesion_id"], errors="ignore", inplace=True)

# ============================================================
# 9. FEATURES FINALES
# ============================================================
FINAL_TABULAR_FEATURES = NUMERIC_IMPUTE_COLS_EXIST + CATEGORICAL_COLS + ["has_lesion_id"]
FINAL_TABULAR_FEATURES = list(dict.fromkeys(FINAL_TABULAR_FEATURES))

print(f"\nNombre total de features tabulaires : {len(FINAL_TABULAR_FEATURES)}")
print(train_df_split[FINAL_TABULAR_FEATURES].dtypes.head())



# ============================================================
# CELLULE UNIQUE : Extraction du SKIN TONE + IntÃ©gration
# ============================================================

import numpy as np
import cv2
from tqdm.auto import tqdm

print("=== Extraction automatique du SKIN TONE (LAB-space) ===")

# ------------------------------------------------------------
# 1. Fonction dâ€™extraction du skin tone moyen (sur la peau SEULE)
# ------------------------------------------------------------
def extract_skin_tone(image_path):
    """
    Retourne la composante 'L' moyenne du ton de peau dans lâ€™espace LAB.
    On masque la lÃ©sion en utilisant un seuillage basÃ© sur la saturation.
    Valeurs possibles : 0â€“100 normalisÃ© â†’ 0.0â€“1.0
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return np.nan
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convertir en LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)

        # Masque peau approximatif :
        # peau = faible saturation â†’ |Aâˆ’128| + |Bâˆ’128| < seuil
        sat = np.abs(A - 128) + np.abs(B - 128)
        skin_mask = sat < 25

        # Ã‰viter les zones trop sombres/bruitÃ©es
        skin_mask &= (L > 20)

        if skin_mask.sum() < 50:  # aucune peau suffisante dÃ©tectÃ©e
            return np.nan

        skin_L_mean = L[skin_mask].mean() / 255.0  # normalisation
        return skin_L_mean

    except Exception:
        return np.nan


# ------------------------------------------------------------
# 2. Extraction pour TRAIN et VALID
# ------------------------------------------------------------
def compute_skin_tone_series(df):
    tones = []
    for p in tqdm(df["image_path"], desc="Skin Tone Extraction", miniters=500):
        tones.append(extract_skin_tone(p))
    return np.array(tones, dtype=np.float32)


train_skin = compute_skin_tone_series(train_df_split)
val_skin   = compute_skin_tone_series(val_df_split)

train_df_split["skin_tone"] = train_skin
val_df_split["skin_tone"]   = val_skin

# Imputer skin tone manquant (rare)
median_skin = np.nanmedian(train_skin)
train_df_split["skin_tone"].fillna(median_skin, inplace=True)
val_df_split["skin_tone"].fillna(median_skin, inplace=True)

# ------------------------------------------------------------
# 3. Ajout automatique Ã  FINAL_TABULAR_FEATURES
# ------------------------------------------------------------
if "skin_tone" not in FINAL_TABULAR_FEATURES:
    FINAL_TABULAR_FEATURES.append("skin_tone")

print(f"Skin tone ajoutÃ©. Total features = {len(FINAL_TABULAR_FEATURES)}")
print("AperÃ§u:", train_df_split["skin_tone"].head())



# ============================================================
# CELLULE: EntraÃ®nement PyTorch multi-modal (robuste & optimisÃ©)
# ============================================================

import os, random, math, copy, time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torchvision import transforms
import timm

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

# -----------------------
# 0) DÃ‰TERMINISME GLOBAL
# -----------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Pour CUDA determinisme CuBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

def worker_init_fn(worker_id):
    seed = SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)

# -----------------------
# 1) PrÃ©paration des donnÃ©es tabulaires
# -----------------------
assert 'train_df_split' in globals(), "Run preprocessing cell first (train_df_split)"
assert 'FINAL_TABULAR_FEATURES' in globals(), "Run preprocessing cell first (FINAL_TABULAR_FEATURES)"

train_df_local = train_df_split.copy()  # PREPROCESS COMPLET + skin_tone

# Imputation par mÃ©diane et conversion float
for c in FINAL_TABULAR_FEATURES:
    if c in train_df_local.columns:
        median_val = train_df_local[c].median()
        train_df_local[c] = pd.to_numeric(train_df_local[c], errors='coerce').fillna(median_val).astype(np.float32)

# -----------------------
# 2) Dataset class
# -----------------------
class ISICDataset(Dataset):
    def __init__(self, df, feature_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.tabular = torch.tensor(self.df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(self.df['target'].values, dtype=torch.float32).unsqueeze(1)
        self.image_paths = self.df['image_path'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224,224), color=(128,128,128))
        if self.transform:
            img = self.transform(img)
        tab = self.tabular[idx]
        lbl = self.labels[idx]
        return img, tab, lbl

# -----------------------
# 3) Transforms
# -----------------------
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# -----------------------
# 4) Folds par patient (anti-leak)
# -----------------------
n_splits = 3
patient_df = train_df_local[['patient_id', 'target']].drop_duplicates(subset=['patient_id'])
patient_labels = patient_df.groupby('patient_id')['target'].max().astype(int).values
patient_ids = patient_df['patient_id'].values
skf_patients = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# -----------------------
# 5) Model + Loss
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        loss = self.alpha * (1-p_t)**self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class MultiModalModel(nn.Module):
    def __init__(self, num_tabular_features, model_name='resnet18', pretrained=True):
        super().__init__()
        # Tentative de chargement des poids prÃ©-entraÃ®nÃ©s
        try:
            self.cnn = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as e:
            print(f"[WARN] Impossible de charger les poids prÃ©-entraÃ®nÃ©s ({e}), fallback sur model non prÃ©-entraÃ®nÃ©.")
            self.cnn = timm.create_model(model_name, pretrained=False, num_classes=0)
        cnn_feat = self.cnn.num_features
        tab_hidden = 64
        self.mlp = nn.Sequential(
            nn.Linear(num_tabular_features, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        fusion = cnn_feat + tab_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x_img, x_tab):
        img_feat = self.cnn(x_img)
        tab_feat = self.mlp(x_tab)
        comb = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(comb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# 6) Training Loop
# -----------------------
oof_preds = np.zeros(len(train_df_local))
oof_targets = train_df_local['target'].values
fold_results = {}

for fold, (p_tr_idx, p_val_idx) in enumerate(skf_patients.split(patient_ids, patient_labels), start=1):
    print(f"\n--- Fold {fold}/{n_splits} ---")
    
    train_patients = patient_ids[p_tr_idx]
    val_patients = patient_ids[p_val_idx]

    train_idx = train_df_local[train_df_local['patient_id'].isin(train_patients)].index.values
    val_idx = train_df_local[train_df_local['patient_id'].isin(val_patients)].index.values

    train_fold_df = train_df_local.loc[train_idx].reset_index(drop=True)
    val_fold_df = train_df_local.loc[val_idx].reset_index(drop=True)

    # Sampler pour classes dÃ©sÃ©quilibrÃ©es
    class_counts = train_fold_df['target'].value_counts().to_dict()
    weights = train_fold_df['target'].map(lambda t: 1.0 / class_counts[int(t)]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Datasets & DataLoaders
    train_ds = ISICDataset(train_fold_df, FINAL_TABULAR_FEATURES, transform=train_transform)
    val_ds = ISICDataset(val_fold_df, FINAL_TABULAR_FEATURES, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler,
                              num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=4, worker_init_fn=worker_init_fn, pin_memory=True)

    model = MultiModalModel(len(FINAL_TABULAR_FEATURES), model_name='resnet18', pretrained=False).to(device)
    crit = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_auprc = 0.0
    best_path = f'best_fold_{fold}.pth'
    EPOCHS = 3

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for imgs, tabs, lbls in train_loader:
            imgs, tabs, lbls = imgs.to(device), tabs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs, tabs)
            loss = crit(logits, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, tabs, lbls in val_loader:
                imgs, tabs = imgs.to(device), tabs.to(device)
                logits = model(imgs, tabs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds.extend(probs)
                trues.extend(lbls.numpy().flatten())

        try:
            auc = roc_auc_score(trues, preds)
        except:
            auc = float('nan')
        auprc = average_precision_score(trues, preds)

        print(f"Fold {fold} Epoch {epoch} â€” train_loss: {train_loss:.4f} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f}")

        if auprc > best_auprc:
            best_auprc = auprc
            torch.save(model.state_dict(), best_path)

    # Evaluation OOF
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    all_fold_preds = []
    with torch.no_grad():
        for imgs, tabs, lbls in val_loader:
            imgs, tabs = imgs.to(device), tabs.to(device)
            logits = model(imgs, tabs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_fold_preds.extend(probs)
    
    oof_preds[val_idx] = all_fold_preds

    # Metrics par fold
    fold_auc = roc_auc_score(train_df_local.loc[val_idx, 'target'], all_fold_preds)
    fold_auprc = average_precision_score(train_df_local.loc[val_idx, 'target'], all_fold_preds)
    
    val_pd = train_df_local.loc[val_idx].copy()
    val_pd['pred'] = all_fold_preds
    patient_pred = val_pd.groupby('patient_id')['pred'].max().values
    patient_true = val_pd.groupby('patient_id')['target'].max().values
    patient_auprc = average_precision_score(patient_true, patient_pred)

    fold_results[f'fold_{fold}'] = {
        'image_auc': float(fold_auc),
        'image_auprc': float(fold_auprc),
        'patient_auprc': float(patient_auprc)
    }

    print(f"Fold {fold} â€” AUROC={fold_auc:.4f}, AUPRC={fold_auprc:.4f}, Patient AUPRC={patient_auprc:.4f}")

# Metrics OOF Global
global_auc = roc_auc_score(oof_targets, oof_preds)
global_auprc = average_precision_score(oof_targets, oof_preds)

print("\n==== OOF Summary ====")
print("Global AUROC:", global_auc)
print("Global AUPRC:", global_auprc)
print("Per-fold:", fold_results)


import os
import numpy as np
import pandas as pd
from IPython.display import display

print("### ğŸš€ InfÃ©rence et PrÃ©paration Submission - V. FINALE ROBUSTE")

# --- DÃ©pendances Ã  avoir dÃ©jÃ  dans le notebook ---
# df_adc_info, df_test_star_info, spectral_data, model_mu, model_sigma

# --- Chemin vers le dossier test ---
TEST_ROOT = "/kaggle/input/ariel-data-challenge-2025/test"
FGS1_PREFIX = "FGS1_signal_"
AIRS_PREFIX = "AIRS-CH0_signal_"

# --- Liste des planet_ids (sous-dossiers) ---
planet_ids_test = [f for f in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, f))]

# ParamÃ¨tres spectres
NUM_FGS1_WL = 60
NUM_AIRS_WL = 223  # 283 - 60
TOTAL_WL = NUM_FGS1_WL + NUM_AIRS_WL

results = []

# --- Boucle sur chaque planÃ¨te ---
for pid in planet_ids_test:
    planet_path = os.path.join(TEST_ROOT, pid)

    # --- Lecture FGS1 ---
    fgs1_files = [f for f in os.listdir(planet_path) if f.startswith(FGS1_PREFIX) and f.endswith('.parquet')]
    if fgs1_files:
        fgs1_dfs = [pd.read_parquet(os.path.join(planet_path, f)) for f in fgs1_files]
        fgs1_signal = pd.concat(fgs1_dfs, axis=0).mean(axis=0).values
        fgs1_gain = df_adc_info['FGS1_adc_gain'].values[0]
        fgs1_offset = df_adc_info['FGS1_adc_offset'].values[0]
        fgs1_signal_calib = np.clip(fgs1_signal * fgs1_gain + fgs1_offset, 0, 0.1)
    else:
        fgs1_signal_calib = np.zeros(NUM_FGS1_WL)

    # --- Lecture AIRS ---
    airs_files = [f for f in os.listdir(planet_path) if f.startswith(AIRS_PREFIX) and f.endswith('.parquet')]
    if airs_files:
        airs_dfs = [pd.read_parquet(os.path.join(planet_path, f)) for f in airs_files]
        airs_signal = pd.concat(airs_dfs, axis=0).mean(axis=0).values
        airs_gain = df_adc_info['AIRS-CH0_adc_gain'].values[0]
        airs_offset = df_adc_info['AIRS-CH0_adc_offset'].values[0]
        airs_signal_calib = np.clip(airs_signal * airs_gain + airs_offset, 0, 0.1)
    else:
        airs_signal_calib = np.zeros(NUM_AIRS_WL)

    # --- Resample pour correspondre aux dimensions du trainset ---
    fgs1_signal_calib_resampled = np.interp(
        np.linspace(0, len(fgs1_signal_calib)-1, NUM_FGS1_WL),
        np.arange(len(fgs1_signal_calib)),
        fgs1_signal_calib
    )
    airs_signal_calib_resampled = np.interp(
        np.linspace(0, len(airs_signal_calib)-1, NUM_AIRS_WL),
        np.arange(len(airs_signal_calib)),
        airs_signal_calib
    )

    # --- Vecteur spectral final ---
    spectral_test = np.zeros(TOTAL_WL)
    spectral_test[:NUM_FGS1_WL] = fgs1_signal_calib_resampled
    spectral_test[NUM_FGS1_WL:] = airs_signal_calib_resampled

    # --- Normalisation selon stats du trainset ---
    spectral_test_norm = (spectral_test - spectral_data.mean().values) / spectral_data.std().values

    # --- Features physiques rÃ©elles ---
    try:
        star_info_row = df_test_star_info.loc[df_test_star_info["planet_id"] == int(pid)].iloc[0]
    except:
        print(f"âš ï¸� ID planÃ¨te {pid} non trouvÃ©, valeurs par dÃ©faut utilisÃ©es.")
        star_info_row = {"Rs": 1.0, "Ms": 1.0, "Ts": 5500.0, "Mp": 1.0, "P": 1.0, "sma": 1.0, "i": 90.0}

    Rs_test = star_info_row["Rs"]
    Ms_test = star_info_row["Ms"]
    Ts_test = star_info_row["Ts"]
    Mp_test = star_info_row["Mp"]
    P_test = star_info_row["P"]
    sma_test = star_info_row["sma"]
    i_test = star_info_row["i"]

    # --- Features dÃ©rivÃ©es (optionnel selon ton modÃ¨le) ---
    signal_fgs1_mean = fgs1_signal_calib_resampled.mean()
    signal_airs_mean = airs_signal_calib_resampled.mean()
    ratio_airs_fgs1 = signal_airs_mean / (signal_fgs1_mean + 1e-12)
    std_depth = np.std(np.concatenate([fgs1_signal_calib_resampled, airs_signal_calib_resampled]))
    max_depth = max(fgs1_signal_calib_resampled.max(), airs_signal_calib_resampled.max())
    min_depth = min(fgs1_signal_calib_resampled.min(), airs_signal_calib_resampled.min())
    
    DERIVED_COLS = [signal_fgs1_mean, signal_airs_mean, ratio_airs_fgs1, std_depth, max_depth, min_depth]

    # --- Assemblage features ---
    # Si le modÃ¨le attend 290 features (283 spectrales + 7 physiques), retirez DERIVED_COLS
    X_test_features = np.hstack([
        spectral_test_norm,
        [Rs_test, Ms_test, Ts_test, Mp_test, P_test, sma_test, i_test],
        DERIVED_COLS  # Retirer si le modÃ¨le a 290 features
    ]).reshape(1, -1)

    # --- InfÃ©rence ---
    mu_pred = model_mu.predict(X_test_features.astype(np.float64))[0]
    sigma_pred = np.clip(model_sigma.predict(X_test_features.astype(np.float64))[0], 1e-6, None)

    results.append({
        "planet_id": int(pid),
        "mu": mu_pred,
        "sigma": sigma_pred
    })

# --- CrÃ©ation DataFrame submission ---
df_submission = pd.DataFrame(results)
df_submission = df_submission.sort_values("planet_id").reset_index(drop=True)

# --- Sauvegarde ---
submission_path = "/kaggle/working/submission.csv"
df_submission.to_csv(submission_path, index=False)
print(f"\nâœ… Submission prÃªte : {submission_path}")
display(df_submission.head())


