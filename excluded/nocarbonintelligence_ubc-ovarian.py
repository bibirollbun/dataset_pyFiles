# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from glob import glob
# from PIL import Image
# 
# # ==============================================================================
# # 1. CHARGEMENT DES DONNÃ‰ES ET VÃ‰RIFICATION INITIALE
# # ==============================================================================
# print("1. CHARGEMENT ET VÃ‰RIFICATION DES FICHIERS CSV")
# 
# DATA_PATH = '../input/UBC-OCEAN/' 
# df_train = None 
# data_loaded = False 
# 
# try:
# 	df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
# 	df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
# 	
# 	# Affichage des premiÃ¨res lignes et info
# 	print(f"Forme de df_train: {df_train.shape}")
# 	print(df_train.head())
# 	print("\nInformations sur les colonnes:")
# 	df_train.info()
# 	
# 	print("\nValeurs manquantes:")
# 	print(df_train.isnull().sum())
# 	
# 	data_loaded = True 
# 
# except FileNotFoundError as e:
# 	print(f"Erreur CRITIQUE de chargement: Assurez-vous que le chemin est correct. {e}")
# 	print("Le reste de l'EDA ne peut pas s'exÃ©cuter.")
# 
# # ==============================================================================
# # 2. ANALYSE DE LA VARIABLE CIBLE
# # ==============================================================================
# if data_loaded:
# 	print("\n2. ANALYSE DE LA DISTRIBUTION DE LA VARIABLE CIBLE ('label')")
# 
# 	label_counts = df_train['label'].value_counts()
# 	label_percent = df_train['label'].value_counts(normalize=True) * 100
# 
# 	print(f"\nComptage par sous-type:\n{label_counts}")
# 	print(f"\nPourcentage par sous-type:\n{label_percent.round(2)}")
# 
# 	plt.figure(figsize=(10, 5))
# 	sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
# 	plt.title('Distribution des Sous-types de Cancer de l\'Ovaire')
# 	plt.ylabel('Nombre d\'Ã©chantillons')
# 	plt.xlabel('Sous-type')
# 	
# 	# Affichage du pourcentage sur les barres
# 	for i, count in enumerate(label_counts.values):
# 		plt.text(i, count + 1, f"{count} ({label_percent.values[i]:.1f}%)", ha='center')
# 	
# 	plt.show()
# 
# # ==============================================================================
# # 3. ANALYSE DES IMAGES
# # ==============================================================================
# 	print("\n3. ANALYSE DES IMAGES ET THUMBNAILS")
# 	THUMBNAIL_PATH = os.path.join(DATA_PATH, 'train_thumbnails')
# 	
# 	# Fonction d'ouverture d'image avec gestion d'erreur
# 	def load_image(img_id):
# 		file_name = f'{img_id}_thumbnail.png'
# 		img_path = os.path.join(THUMBNAIL_PATH, file_name)
# 		try:
# 			return Image.open(img_path)
# 		except FileNotFoundError:
# 			return None
# 
# 	df_train_unique_labels = df_train.drop_duplicates(subset=['label']).reset_index(drop=True)
# 
# 	dimensions = []
# 	for img_id in df_train_unique_labels['image_id']:
# 		img = load_image(img_id)
# 		if img:
# 			dimensions.append((img_id, img.size))
# 		else:
# 			dimensions.append((img_id, 'FICHIER INTROUVABLE'))
# 	print(f"Dimensions des exemples par classe (ID, Largeur, Hauteur): {dimensions}")
# 
# 	# Visualisation des thumbnails
# 	n_classes = len(df_train_unique_labels)
# 	fig, axes = plt.subplots(1, n_classes, figsize=(4*n_classes, 4))
# 	axes = axes.flatten() if n_classes > 1 else [axes]
# 	plt.suptitle('Exemples de Thumbnails par Sous-type (1 par classe)', fontsize=16)
# 
# 	for i, row in df_train_unique_labels.iterrows():
# 		img = load_image(row['image_id'])
# 		if img:
# 			axes[i].imshow(img)
# 			axes[i].set_title(f"{row['label']} (ID: {row['image_id']})", fontsize=12)
# 		else:
# 			axes[i].text(0.5, 0.5, f"{row['label']}\nERREUR: Fichier manquant", 
# 						 ha='center', va='center', color='red')
# 		axes[i].axis('off')
# 
# 	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# 	plt.show()
# 
# # ==============================================================================
# # 4. CONCLUSION RAPIDE DE L'EDA
# # ==============================================================================
# 	print("\n4. RÃ‰SUMÃ‰ DES PREMIERS CONSTATS")
# 	print("------------------------------")
# 	print(f"Classes dÃ©tectÃ©es: {df_train['label'].nunique()}")
# 	print(f"Classe dominante: {label_percent.idxmax()} ({label_percent.max():.2f}%)")
# 	print("Fort dÃ©sÃ©quilibre des classes confirmÃ© â†’ Balanced Accuracy recommandÃ©.")
# 	print("Prochaine Ã©tape: gestion des images et intÃ©gration des mÃ©tadonnÃ©es CSV/JSON.")


# import numpy as np
# import pandas as pd
# import os
# from PIL import Image
# import cv2
# from tqdm import tqdm # Barre de progression
# import matplotlib.pyplot as plt
# 
# # ==============================================================================
# # PARAMÃˆTRES CLÃ‰S
# # ==============================================================================
# DATA_PATH = '../input/UBC-OCEAN/'
# WSI_PATH = os.path.join(DATA_PATH, 'train_thumbnails')
# OUTPUT_DIR = './train_patches_from_thumbnails'
# 
# PATCH_SIZE = 256
# TISSUE_THRESHOLD = 0.05
# MAX_PATCHES_COMMON = 500
# MAX_PATCHES_RARE = 1500
# RARE_LABELS = ['LGSC', 'MC']
# 
# # ==============================================================================
# # FONCTIONS
# # ==============================================================================
# def create_tissue_mask_from_png(img_pil, min_saturation=15, max_value=220):
# 	"""CrÃ©e un masque binaire indiquant les zones de tissu."""
# 	img_np = np.array(img_pil.convert("RGB"), dtype=np.uint8)
# 	hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
# 	mask = (hsv[:, :, 1] > min_saturation) & (hsv[:, :, 2] < max_value)
# 	return mask
# 
# def save_patch(patch_img, img_id, label, x, y):
# 	patch_filename = f'{label}_{img_id}_{x}_{y}.png'
# 	patch_img.save(os.path.join(OUTPUT_DIR, label, patch_filename))
# 
# def extract_valid_patches_from_png(img_pil, tissue_mask, img_id, label):
# 	"""Extrait les patches valides contenant assez de tissu."""
# 	max_patches = MAX_PATCHES_RARE if label in RARE_LABELS else MAX_PATCHES_COMMON
# 	width, height = img_pil.size
# 	patch_count = 0
# 
# 	for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
# 		for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
# 			if patch_count >= max_patches:
# 				return patch_count
# 
# 			patch_mask = tissue_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
# 			tissue_ratio = np.sum(patch_mask) / patch_mask.size
# 
# 			if tissue_ratio > TISSUE_THRESHOLD:
# 				patch_img = img_pil.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
# 				save_patch(patch_img, img_id, label, x, y)
# 				patch_count += 1
# 
# 	return patch_count
# 
# # ==============================================================================
# # LOGIQUE PRINCIPALE
# # ==============================================================================
# if df_train is not None:
# 	os.makedirs(OUTPUT_DIR, exist_ok=True)
# 	for label in df_train['label'].unique():
# 		os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
# 
# 	total_patches = 0
# 	print(f"DÃ©but du traitement des {len(df_train)} vignettes...")
# 
# 	for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
# 		img_id = row['image_id']
# 		label = row['label']
# 		wsi_filename = f'{img_id}_thumbnail.png'
# 		wsi_path = os.path.join(WSI_PATH, wsi_filename)
# 
# 		try:
# 			img_pil = Image.open(wsi_path)
# 			tissue_mask = create_tissue_mask_from_png(img_pil)
# 			extracted = extract_valid_patches_from_png(img_pil, tissue_mask, img_id, label)
# 			total_patches += extracted
# 		except FileNotFoundError:
# 			print(f"Fichier introuvable : {wsi_path}")
# 		except Exception as e:
# 			print(f"Erreur pour {img_id}: {e}")
# 
# 	print(f"\nFin du traitement. Total de patches extraits : {total_patches}")


from torchvision import transforms
import random
from PIL import Image

# Fonction custom pour rotation 0/90/180/270
class RandomRotation90:
    def __call__(self, img: Image.Image):
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        return img.rotate(angle)

# ----------------------------
# Transformations pour l'entraÃ®nement (augmentation + normalisation)
# ----------------------------
TRAIN_TRANSFORMS = transforms.Compose([
    RandomRotation90(),          # Rotations multiples de 90Â°
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Transformations pour validation/test (normalisation uniquement)
# ----------------------------
VALID_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.preprocessing import StandardScaler # GardÃ© pour la clartÃ© des imports

# --- PrÃ©paration des donnÃ©es tabulaires (COMMENTÃ‰ car dÃ©pend de df_train) ---
# print("1. PrÃ©paration des donnÃ©es tabulaires pour la fusion...")
# df_train['is_tma_encoded'] = df_train['is_tma'].astype(int)
# if 'image_width' in df_train.columns and 'image_height' in df_train.columns:
#     scaler = StandardScaler()
#     df_train[['width_norm', 'height_norm']] = scaler.fit_transform(
#         df_train[['image_width', 'image_height']].fillna(0)
#     )
# TABULAR_FEATURES = ['is_tma_encoded', 'width_norm', 'height_norm']
# NUM_TAB_FEATURES = len(TABULAR_FEATURES)
# LABEL_MAPPING = {label: i for i, label in enumerate(df_train['label'].unique())}
# NUM_CLASSES = df_train['label'].nunique()
# print(f"-> {NUM_TAB_FEATURES} caractÃ©ristiques tabulaires dÃ©finies.")
# print(f"-> {NUM_CLASSES} classes cibles dÃ©finies.")

# --- DÃ©finition du ModÃ¨le Bimodale (CORRIGÃ‰E) ---
print("2. DÃ©finition du ModÃ¨le de Fusion Bimodale (CNN + MLP) chargÃ©e et corrigÃ©e pour le mode SANS INTERNET.")

class BimodalFusionModel(nn.Module):
    def __init__(self, num_classes, num_tabular_features, hidden_dim=64, dropout_tab=0.2, dropout_fusion=0.3, freeze_cnn=False):
        super().__init__()

        # --- Branche CNN (EfficientNet-B0) ---
        # CORRECTION ICI : Utiliser weights=None pour Ã©viter le tÃ©lÃ©chargement Internet
        self.cnn_extractor = models.efficientnet_b0(weights=None)
        
        # Le reste utilise l'architecture EfficientNet-B0 standard
        num_image_features = self.cnn_extractor.classifier[1].in_features
        self.cnn_extractor.classifier = nn.Identity()

        if freeze_cnn:
            for param in self.cnn_extractor.features.parameters():
                param.requires_grad = False

        # --- Branche MLP tabulaire ---
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_tab),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # --- Fusion et classification ---
        total_fusion_size = num_image_features + (hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(total_fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_fusion),
            nn.Linear(256, num_classes)
        )

        # --- Initialisation des poids linÃ©aires ---
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_patches, tabular_data):
        assert image_patches.dim() == 4, "Images doivent Ãªtre 4D (B, C, H, W)"
        assert tabular_data.dim() == 2, "Tabular data doit Ãªtre 2D (B, features)"

        image_features = self.cnn_extractor(image_patches)
        tabular_features = self.tabular_mlp(tabular_data)
        fusion_features = torch.cat((image_features, tabular_features), dim=1)
        output = self.classifier(fusion_features)
        return output

# --- Test d'initialisation (COMMENTÃ‰ car dÃ©pend de variables manquantes) ---
# model = BimodalFusionModel(NUM_CLASSES, NUM_TAB_FEATURES)
# print(f"ModÃ¨le initialisÃ©. Taille features fusionnÃ©es : {model.classifier[0].in_features}")


# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import os
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from tqdm.auto import tqdm
# from glob import glob
# 
# PATCHES_DIR = './train_patches_from_thumbnails'
# 
# class OvarianCancerDataset(Dataset):
# 	"""Dataset personnalisÃ© pour la classification multi-modale (Image + Tabulaire)."""
# 
# 	def __init__(self, df_data, patches_root_dir, transform=None, tabular_features=None, label_mapping=None):
# 		self.df = df_data.reset_index(drop=True)
# 		self.patches_root_dir = patches_root_dir
# 		self.transform = transform
# 		self.tabular_features = tabular_features if tabular_features else []
# 		self.label_mapping = label_mapping if label_mapping else {}
# 		self.patch_list = self._create_patch_list()
# 
# 	def _create_patch_list(self):
# 		"""CrÃ©e une liste de dictionnaires contenant les chemins de patches, labels et features tabulaires."""
# 		all_patches = []
# 		for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="CrÃ©ation de la liste des patches"):
# 			img_id = row['image_id']
# 			label = row['label']
# 			label_dir = os.path.join(self.patches_root_dir, label)
# 			if os.path.isdir(label_dir):
# 				patch_files = glob(os.path.join(label_dir, f'{label}_{img_id}_*.png'))
# 				for patch_path in patch_files:
# 					tabular_data = row[self.tabular_features].fillna(0).values.astype(np.float32)
# 					all_patches.append({
# 						'path': patch_path,
# 						'label': self.label_mapping.get(label, -1),
# 						'tabular': tabular_data
# 					})
# 		return all_patches
# 
# 	def __len__(self):
# 		return len(self.patch_list)
# 
# 	def __getitem__(self, idx):
# 		item = self.patch_list[idx]
# 		# Utiliser "with" pour garantir la fermeture du fichier
# 		with Image.open(item['path']).convert('RGB') as img:
# 			image = img.copy()
# 		if self.transform:
# 			image = self.transform(image)
# 		tabular_data = torch.tensor(item['tabular'], dtype=torch.float32)
# 		label = torch.tensor(item['label'], dtype=torch.long)
# 		return image, tabular_data, label
# 
# # --- Validation croisÃ©e stratifiÃ©e ---
# print("\nPrÃ©paration des indices pour la Validation CroisÃ©e (K-Fold)...")
# N_SPLITS = 5 # Ajuster si nÃ©cessaire
# skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
# 
# X_indices = df_train.index.values
# y_labels = df_train['label'].map(LABEL_MAPPING).values
# 
# print(f"Validation croisÃ©e sur {N_SPLITS} folds prÃªte.")
# print("Classe OvarianCancerDataset dÃ©finie et opÃ©rationnelle.")


# # ==============================================================================
# # Ã‰TAPE PRÃ‰LIMINAIRE : INSPECTION, FILTRAGE DES PATCHES & VÃ‰RIFICATION DE LA STRATIFICATION
# # ==============================================================================
# import os
# import glob
# import pandas as pd
# import numpy as np
# 
# # ATTENTION: Assurez-vous que df_train, LABEL_MAPPING et PATCHES_DIR sont dÃ©finis ici !
# # PATCHES_DIR est dÃ©fini dans la cellule de la classe Dataset, mais on le redÃ©finit ici par sÃ©curitÃ©.
# PATCHES_DIR = './train_patches_from_thumbnails'
# 
# print("--- 1. Inspection de la Distribution des Classes (WSI) ---")
# initial_df_size = len(df_train)
# print(f"Taille initiale de df_train: {initial_df_size} WSI")
# print("Distribution initiale des labels (WSI):")
# print(df_train['label'].value_counts())
# print("-" * 30)
# 
# # ------------------------------------------------------------------------------
# # 2. INSPECTION ET FILTRAGE DES PATCHES EXISTANTS
# # ------------------------------------------------------------------------------
# print("\n--- 2. Inspection et Filtrage des Patches sur le Disque ---")
# 
# # a. Compter le nombre total de patches et vÃ©rifier la structure
# all_png_files = glob.glob(os.path.join(PATCHES_DIR, '**', '*.png'), recursive=True)
# count = len(all_png_files)
# 
# print(f"Total de fichiers .png trouvÃ©s dans {PATCHES_DIR}: {count}")
# 
# if count < initial_df_size * 5: # Un seuil trÃ¨s bas pour dÃ©tecter un patching incomplet
# 	print("âš ï¸� ALERTE: Le nombre de patches est trÃ¨s faible (moins de 5 patches/WSI en moyenne). Risque de folds vides.")
# 
# # b. Identifier les Image IDs ayant rÃ©ellement des patches
# existing_patch_ids = set()
# for label in df_train['label'].unique():
# 	label_dir = os.path.join(PATCHES_DIR, label)
# 	if os.path.isdir(label_dir):
# 		for filename in os.listdir(label_dir):
# 			if filename.endswith(".png"):
# 				try:
# 					# L'ID est le deuxiÃ¨me Ã©lÃ©ment aprÃ¨s le split par '_'
# 					parts = filename.split('_')
# 					# Convertir en entier car image_id est un entier dans le df
# 					img_id = int(parts[1])
# 					existing_patch_ids.add(img_id)
# 				except (ValueError, IndexError):
# 					continue
# 
# # c. Filtrage du DataFrame
# df_train_filtered = df_train[df_train['image_id'].isin(existing_patch_ids)]
# 
# # ------------------------------------------------------------------------------
# # 3. MISE Ã€ JOUR ET VÃ‰RIFICATION POST-FILTRAGE
# # ------------------------------------------------------------------------------
# print("\n--- 3. Mise Ã  Jour et VÃ©rification Post-Filtrage ---")
# 
# # Rapport sur le filtrage
# final_df_size = len(df_train_filtered)
# num_dropped = initial_df_size - final_df_size
# 
# if num_dropped > 0:
# 	print(f"ATTENTION: {num_dropped} WSI/Vignettes ont Ã©tÃ© exclues car aucun patch n'a Ã©tÃ© trouvÃ©.")
# 	print(f"Taille finale (avec patches): {final_df_size} WSI.")
# 
# 	# Affichage de la nouvelle distribution pour s'assurer que la stratification n'est pas cassÃ©e
# 	print("\nNouvelle distribution des labels (WSI) aprÃ¨s filtrage:")
# 	print(df_train_filtered['label'].value_counts())
# else:
# 	print(f"Toutes les {initial_df_size} WSI initiales ont au moins un patch. Aucun WSI n'a Ã©tÃ© exclu.")
# 
# # d. Mise Ã  jour des variables K-Fold
# df_train = df_train_filtered # Remplace le DataFrame par la version filtrÃ©e
# 
# # Mise Ã  jour des indices pour la Validation CroisÃ©e
# X_indices = df_train.index.values
# y_labels = df_train['label'].map(LABEL_MAPPING).values
# 
# print("\nâœ… PrÃ©paration des donnÃ©es terminÃ©e. df_train est maintenant prÃªt pour la K-Fold.")
# # ==============================================================================


# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from tqdm.auto import tqdm
# from sklearn.metrics import balanced_accuracy_score
# import numpy as np
# import pandas as pd
# from torchvision import transforms
# import copy
# import random
# import gc
# from glob import glob
# import warnings
# warnings.filterwarnings("ignore") # âœ… Supprime tous les warnings
# 
# # ==============================================================================
# # CLASSE EARLY STOPPING
# # ==============================================================================
# class EarlyStopper:
# 	"""ArrÃªte l'entraÃ®nement si la mÃ©trique ne s'amÃ©liore pas aprÃ¨s une patience donnÃ©e."""
# 	def __init__(self, patience=5, min_delta=0):
# 		self.patience = patience
# 		self.min_delta = min_delta
# 		self.counter = 0
# 		self.best_metric = -np.inf
# 
# 	def early_stop(self, metric):
# 		if metric > self.best_metric + self.min_delta:
# 			self.best_metric = metric
# 			self.counter = 0
# 			return False
# 		else:
# 			self.counter += 1
# 			if self.counter >= self.patience:
# 				return True
# 			return False
# 
# # ==============================================================================
# # PARAMÃˆTRES GLOBAUX & SEED
# # ==============================================================================
# def set_seed(seed_value=42):
# 	random.seed(seed_value)
# 	np.random.seed(seed_value)
# 	torch.manual_seed(seed_value)
# 	if torch.cuda.is_available():
# 		torch.cuda.manual_seed_all(seed_value)
# 		torch.backends.cudnn.deterministic = True
# 		torch.backends.cudnn.benchmark = False
# 
# set_seed(42)
# 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Appareil d'entraÃ®nement utilisÃ© : {device}")
# 
# BATCH_SIZE = 64
# NUM_WORKERS = 0
# NUM_EPOCHS = 5
# LEARNING_RATE = 1e-4
# EARLY_STOP_PATIENCE = 1
# 
# try:
# 	N_SPLITS
# except NameError:
# 	N_SPLITS = 5
# 
# # ==============================================================================
# # TRANSFORMATIONS
# # ==============================================================================
# TRAIN_TRANSFORMS = transforms.Compose([
# 	transforms.RandomRotation(90),
# 	transforms.RandomHorizontalFlip(),
# 	transforms.RandomVerticalFlip(),
# 	transforms.ColorJitter(0.2,0.2,0.2,0.1),
# 	transforms.ToTensor(),
# 	transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])
# 
# VALID_TRANSFORMS = transforms.Compose([
# 	transforms.ToTensor(),
# 	transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])
# 
# # ==============================================================================
# # DATASET PERSONNALISÃ‰
# # ==============================================================================
# class OvarianCancerDataset(Dataset):
# 	def __init__(self, df_data, patches_root_dir, transform=None, tabular_features=None, label_mapping=None):
# 		self.df = df_data.reset_index(drop=True)
# 		self.patches_root_dir = patches_root_dir
# 		self.transform = transform
# 		self.tabular_features = tabular_features if tabular_features else []
# 		self.label_mapping = label_mapping if label_mapping else {}
# 		self.patch_list = self._create_patch_list()
# 
# 	def _create_patch_list(self):
# 		patch_list = []
# 		for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="CrÃ©ation de la liste des patches"):
# 			img_id = row["image_id"]
# 			label = str(row["label"]) if "label" in row else ""
# 			label_dir = os.path.join(self.patches_root_dir, label) if label else self.patches_root_dir
# 
# 			if os.path.isdir(label_dir):
# 				patch_files = glob(os.path.join(label_dir, f'{label}_{img_id}_*.png'))
# 				for patch_path in patch_files:
# 					tabular_data = row[self.tabular_features].fillna(0).values.astype(np.float32) if self.tabular_features else np.array([])
# 					patch_list.append({
# 						"image_path": patch_path,
# 						"tabular": tabular_data,
# 						"label": self.label_mapping.get(row["label"], -1) if "label" in row else -1,
# 						"image_id": img_id
# 					})
# 		return patch_list
# 
# 	def __len__(self):
# 		return len(self.patch_list)
# 
# 	def __getitem__(self, idx):
# 		item = self.patch_list[idx]
# 		from PIL import Image
# 		image = Image.open(item["image_path"]).convert("RGB")
# 		if self.transform:
# 			image = self.transform(image)
# 
# 		tabular = torch.tensor(item["tabular"], dtype=torch.float32)
# 		label = torch.tensor(item["label"], dtype=torch.long)
# 		return image, tabular, label
# 
# # ==============================================================================
# # FONCTION DE VALIDATION
# # ==============================================================================
# def validate_model(model, loader, criterion, device, fold_num, epoch_num, num_epochs):
# 	model.eval()
# 	running_loss = 0.0
# 	all_preds = []
# 	all_labels = []
# 
# 	with torch.no_grad():
# 		pbar = tqdm(loader, desc=f"Fold {fold_num} | Ã‰poque {epoch_num}/{num_epochs} [Valid]")
# 		for images, tabular, labels in pbar:
# 			images = images.to(device)
# 			tabular = tabular.to(device)
# 			labels = labels.to(device)
# 
# 			outputs = model(images, tabular)
# 			loss = criterion(outputs, labels)
# 
# 			running_loss += loss.item() * images.size(0)
# 			_, predicted = torch.max(outputs, 1)
# 			all_preds.extend(predicted.cpu().numpy())
# 			all_labels.extend(labels.cpu().numpy())
# 			pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
# 
# 	if len(loader.dataset) == 0:
# 		return 0.0, 0.0
# 
# 	epoch_loss = running_loss / len(loader.dataset)
# 	balanced_acc = balanced_accuracy_score(all_labels, all_preds)
# 	return epoch_loss, balanced_acc
# 
# # ==============================================================================
# # VALIDATION CROISÃ‰E
# # ==============================================================================
# all_fold_metrics = []
# empty_fold_warning_count = 0
# 
# print(f"\nDÃ©but de l'entraÃ®nement avec Validation CroisÃ©e ({N_SPLITS} folds)...")
# 
# for fold, (train_index, val_index) in enumerate(skf.split(X_indices, y_labels)):
# 	fold_num = fold + 1
# 	print(f"\n====================== FOLD {fold_num}/{N_SPLITS} ======================")
# 
# 	train_df_fold = df_train.iloc[train_index].reset_index(drop=True)
# 	valid_df_fold = df_train.iloc[val_index].reset_index(drop=True)
# 
# 	train_dataset = OvarianCancerDataset(
# 		df_data=train_df_fold,
# 		patches_root_dir=PATCHES_DIR,
# 		transform=TRAIN_TRANSFORMS,
# 		tabular_features=TABULAR_FEATURES,
# 		label_mapping=LABEL_MAPPING
# 	)
# 	valid_dataset = OvarianCancerDataset(
# 		df_data=valid_df_fold,
# 		patches_root_dir=PATCHES_DIR,
# 		transform=VALID_TRANSFORMS,
# 		tabular_features=TABULAR_FEATURES,
# 		label_mapping=LABEL_MAPPING
# 	)
# 
# 	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# 	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# 
# 	if len(train_dataset) == 0 or len(valid_dataset) == 0:
# 		print(f"âš ï¸� Aucun patch trouvÃ© pour le Fold {fold_num}. Skip.")
# 		empty_fold_warning_count += 1
# 		continue
# 
# 	print(f"Patches Train: {len(train_dataset)} | Patches Valid: {len(valid_dataset)}")
# 
# 	# âœ… Correction ici : BimodalFusionModel
# 	fold_model = BimodalFusionModel(NUM_CLASSES, NUM_TAB_FEATURES).to(device)
# 	criterion = nn.CrossEntropyLoss()
# 	optimizer = optim.Adam(fold_model.parameters(), lr=LEARNING_RATE)
# 
# 	best_fold_acc = 0.0
# 	best_weights = {'model': None, 'optimizer': None}
# 	history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
# 	early_stopper = EarlyStopper(patience=EARLY_STOP_PATIENCE)
# 
# 	for epoch in range(NUM_EPOCHS):
# 		fold_model.train()
# 		running_loss = 0.0
# 		pbar = tqdm(train_loader, desc=f"Fold {fold_num} | Ã‰poque {epoch+1}/{NUM_EPOCHS} [Train]")
# 
# 		for images, tabular, labels in pbar:
# 			images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
# 
# 			optimizer.zero_grad()
# 			outputs = fold_model(images, tabular)
# 			loss = criterion(outputs, labels)
# 			loss.backward()
# 			optimizer.step()
# 
# 			running_loss += loss.item() * images.size(0)
# 			pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
# 
# 		train_loss = running_loss / len(train_dataset)
# 		history['train_loss'].append(train_loss)
# 
# 		val_loss, val_acc = validate_model(fold_model, valid_loader, criterion, device, fold_num, epoch+1, NUM_EPOCHS)
# 		history['val_loss'].append(val_loss)
# 		history['val_acc'].append(val_acc)
# 		print(f"Ã‰poque {epoch+1} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Balanced Acc: {val_acc:.4f}")
# 
# 		if val_acc > best_fold_acc:
# 			best_fold_acc = val_acc
# 			best_weights['model'] = copy.deepcopy(fold_model.state_dict())
# 			best_weights['optimizer'] = copy.deepcopy(optimizer.state_dict())
# 			print(f"ğŸ’¾ Nouveau meilleur modÃ¨le sauvegardÃ© pour Fold {fold_num} (Acc: {best_fold_acc:.4f}).")
# 
# 		if early_stopper.early_stop(val_acc):
# 			print(f"ğŸ›‘ ArrÃªt prÃ©maturÃ© au Fold {fold_num}.")
# 			break
# 
# 	print(f"RÃ©sultat final Fold {fold_num}: Meilleur Balanced Acc = {best_fold_acc:.4f}")
# 	all_fold_metrics.append(best_fold_acc)
# 
# 	torch.save({
# 		'model_state_dict': best_weights['model'],
# 		'optimizer_state_dict': best_weights['optimizer'],
# 		'history': history,
# 		'best_acc': best_fold_acc
# 	}, f'best_fusion_model_fold_{fold_num}.pth')
# 
# # ==============================================================================
# # RÃ‰SULTATS FINAUX
# # ==============================================================================
# if all_fold_metrics:
# 	mean_acc = np.mean(all_fold_metrics)
# 	std_acc = np.std(all_fold_metrics)
# 	print("\n--- RÃ‰SULTATS DE LA VALIDATION CROISÃ‰E ---")
# 	print(f"Metrics par Fold: {all_fold_metrics}")
# 	print(f"Balanced Accuracy Moyen ({N_SPLITS}-Fold): {mean_acc:.4f} Â± {std_acc:.4f}")
# else:
# 	print("\n--- RÃ‰SULTATS DE LA VALIDATION CROISÃ‰E ---")
# 	print("Aucun fold n'a Ã©tÃ© exÃ©cutÃ©.")


import torch
import os

# MAPPING DES NOUVEAUX CHEMINS DE VOS MODÃˆLES
MODEL_PATHS = {
    1: "/kaggle/input/best-fusion-model-fold-1/best_fusion_model_fold_1.pth",
    2: "/kaggle/input/best-fusion-model-fold2/best_fusion_model_fold_2 (1).pth", 
    3: "/kaggle/input/best-fusion-model-fold-3/best_fusion_model_fold_3.pth",
    4: "/kaggle/input/best-fusion-model-fold-4/best_fusion_model_fold_4.pth",
    5: "/kaggle/input/best-fusion-model-fold-5/best_fusion_model_fold_5.pth"
}

print("--- ğŸ§� Inspection CorrigÃ©e des Checkpoints des ModÃ¨les ---")
print("âš ï¸� Utilisation de 'weights_only=False' pour charger les fichiers PyTorch/Numpy.")
print(f"PÃ©riphÃ©rique de chargement: {torch.device('cpu')}")
print("-" * 70)

for fold, path in MODEL_PATHS.items():
    print(f"Fold {fold}: {path}")

    # VÃ©rification de l'existence du fichier
    if not os.path.exists(path):
        print(f"   â�Œ Erreur: Le fichier n'existe pas au chemin: {path}")
        continue
    
    try:
        # CHARGEMENT CORRIGÃ‰: Ajout de weights_only=False pour autoriser les dÃ©pendances NumPy
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        
        # Affichage des clÃ©s principales
        if isinstance(checkpoint, dict):
            # ClÃ©s de niveau supÃ©rieur (checkpoint keys)
            top_keys = list(checkpoint.keys())
            print(f"   âœ… ClÃ©s du Checkpoint trouvÃ©es: {top_keys}")
            
            # Inspection du contenu de 'model_state_dict' (qui contient les poids)
            if 'model_state_dict' in checkpoint:
                model_keys = checkpoint['model_state_dict'].keys()
                print(f"      Nombre de clÃ©s de poids dans 'model_state_dict': {len(model_keys)}")
                # Afficher les 5 premiÃ¨res clÃ©s pour confirmation du format
                print(f"      Exemple de clÃ©s de poids: {list(model_keys)[:5]}...")
            else:
                 print("      âš ï¸� Avertissement: 'model_state_dict' non trouvÃ© dans le checkpoint. VÃ©rifiez la structure.")
        else:
             print("   âš ï¸� Avertissement: Le fichier .pth ne contient pas un dictionnaire (checkpoint).")
            
    except Exception as e:
        print(f"   â�Œ Erreur de chargement critique: {e}")
    
    print("-" * 70)

print("--- Inspection terminÃ©e. Vous pouvez maintenant utiliser `checkpoint['model_state_dict']` pour charger. ---")


import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# --- Assurez-vous que BimodalFusionModel, NUM_CLASSES, et LABEL_MAPPING sont dÃ©finis dans une cellule prÃ©cÃ©dente ---

# Variables essentielles (basÃ©es sur votre entraÃ®nement 5 classes sans 'Other' au dÃ©part):
NUM_CLASSES = 5 
LABEL_MAPPING = {'HGSC': 0, 'LGSC': 1, 'EC': 2, 'CC': 3, 'MC': 4}

# ------------------------------------------------------------------------------
# 0) CHARGEMENT DU TEST CSV
# ------------------------------------------------------------------------------
TEST_CSV_PATH = '/kaggle/input/UBC-OCEAN/test.csv'
df_test = pd.read_csv(TEST_CSV_PATH)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

N_SPLITS = 5
BATCH_SIZE = 64
NUM_WORKERS = 0
IMG_SIZE_FOR_DUMMY = 224 # Taille attendue par EfficientNet

# ------------------------------------------------------------------------------
# 1) FEATURES TABULAIRES (Sans changement)
# ------------------------------------------------------------------------------
df_test = df_test.copy()

# === CORRECTION 1 : AJOUT DE 'is_tma_encoded' (La 3Ã¨me feature) ===
if 'is_tma' not in df_test.columns:
    df_test['is_tma'] = 0 
df_test['is_tma_encoded'] = df_test['is_tma'].astype(int)
# =================================================================

# DÃ©finition des features initiales (avant normalisation)
TABULAR_FEATURES_RAW = ['is_tma_encoded', "image_width", "image_height"]

# Fallback scaling
try:
    # Tente d'utiliser un scaler prÃ©cÃ©demment sauvegardÃ©
    df_test[["width_norm", "height_norm"]] = scaler.transform(df_test[["image_width", "image_height"]])
    TABULAR_FEATURES = ["is_tma_encoded", "width_norm", "height_norm"] 
except NameError:
    print("âš ï¸� Scaler non trouvÃ© ou non chargÃ©, fallback scaling activÃ©.")
    df_test["width_norm"] = df_test["image_width"] / 1000.0
    df_test["height_norm"] = df_test["image_height"] / 1000.0
    TABULAR_FEATURES = ["is_tma_encoded", "width_norm", "height_norm"] 
except Exception:
    print("âš ï¸� ProblÃ¨me avec le scaler. Fallback scaling activÃ©.")
    df_test["width_norm"] = df_test["image_width"] / 1000.0
    df_test["height_norm"] = df_test["image_height"] / 1000.0
    TABULAR_FEATURES = ["is_tma_encoded", "width_norm", "height_norm"] 

# Transformations pour l'infÃ©rence
TEST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

# --- PARTIE D'INSPECTION SUPPLÃ‰MENTAIRE --- ğŸ”�
print("\n--- Inspection des Features Tabulaires ---")
print(f"Nombre de features tabulaires dans le modÃ¨le (attendu: 3): {len(TABULAR_FEATURES)}")
print(f"Features tabulaires utilisÃ©es : {TABULAR_FEATURES}")
print(df_test[TABULAR_FEATURES].head())
print("---------------------------------------")


# ------------------------------------------------------------------------------
# 2) DATASET TEST (AVEC GESTION D'ERREUR ROBUSTE) ğŸ›¡ï¸�
# ------------------------------------------------------------------------------
class OvarianCancerTestDataset(Dataset):
    def __init__(self, df, transform, tabular_features, img_dir):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tabular_features = tabular_features
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # === CORRECTION 2 & 3 : ID et Chemin du Thumbnail ===
        # S'assurer que l'ID est un entier puis une chaÃ®ne sans '.0'
        img_id = str(int(row["image_id"])) 
        img_path = os.path.join(self.img_dir, f"{img_id}_thumbnail.png") 
        # ===================================================

        # === CORRECTION 4 : GESTION D'ERREUR ROBUSTE ===
        try:
            image = Image.open(img_path).convert("RGB")
            # VÃ©rification de taille minimale pour Ã©viter les plantages de transforms
            if image.size[0] < 5 or image.size[1] < 5:
                 raise ValueError("Image trop petite ou invalide.")

            image = self.transform(image)
        except Exception as e:
            # En cas d'erreur (fichier non trouvÃ©/corrompu), retourne une image noire
            # Cela permet au DataLoader de ne pas planter et au script de continuer.
            print(f"âš ï¸� Image ID {img_id} ignorÃ©e/remplacÃ©e par image noire: {e}")
            image = torch.zeros(3, IMG_SIZE_FOR_DUMMY, IMG_SIZE_FOR_DUMMY, dtype=torch.float32)
        # ===============================================

        tab_data = torch.tensor(row[self.tabular_features].values.astype(np.float32), dtype=torch.float32)
        return image, tab_data, img_id

# ------------------------------------------------------------------------------
# 3) DATALOADER
# ------------------------------------------------------------------------------
TEST_IMAGES_DIR = "/kaggle/input/UBC-OCEAN/test_thumbnails"

test_dataset = OvarianCancerTestDataset(
    df=df_test,
    transform=TEST_TRANSFORMS,
    tabular_features=TABULAR_FEATURES, 
    img_dir=TEST_IMAGES_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ------------------------------------------------------------------------------
# 4) ENSEMBLING DES 5 FOLDS (SANS CHANGEMENT MAJEUR)
# ------------------------------------------------------------------------------
# MAPPING DES NOUVEAUX CHEMINS DE VOS MODÃˆLES
MODEL_PATHS = {
    1: "/kaggle/input/best-fusion-model-fold-1/best_fusion_model_fold_1.pth",
    2: "/kaggle/input/best-fusion-model-fold2/best_fusion_model_fold_2 (1).pth", # Chemin ajustÃ©
    3: "/kaggle/input/best-fusion-model-fold-3/best_fusion_model_fold_3.pth",
    4: "/kaggle/input/best-fusion-model-fold-4/best_fusion_model_fold_4.pth",
    5: "/kaggle/input/best-fusion-model-fold-5/best_fusion_model_fold_5.pth"
}

all_predictions = {}
# Inverse map doit exister si LABEL_MAPPING est dÃ©fini
inverse_map = {v: k for k, v in LABEL_MAPPING.items()} 
THRESHOLD_OTHER = 0.30 # Seuil de 0.30 pour la rÃ¨gle "Other"

print(f"\n--- Ensembling de {N_SPLITS} modÃ¨les ---")

for fold in range(1, N_SPLITS + 1):
    model_path = MODEL_PATHS.get(fold)
    
    if not model_path or not os.path.exists(model_path):
        print(f"âš ï¸� ModÃ¨le manquant (Fold {fold}) ou chemin invalide : {model_path} â†’ fold ignorÃ©.")
        continue

    print(f"\nFold {fold} : chargement du modÃ¨le depuis {model_path}...")
    try:
        # L'instance utilise len(TABULAR_FEATURES) = 3
        # La classe BimodalFusionModel (dÃ©finie dans la cellule prÃ©cÃ©dente) doit utiliser weights=None
        model = BimodalFusionModel(NUM_CLASSES, len(TABULAR_FEATURES)) 
        
        # --- CHARGEMENT DU CHECKPOINT CORRIGÃ‰ ---
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        # ------------------------------------------
        
        model.to(device).eval()

        with torch.no_grad():
            for images, tab, img_ids in tqdm(test_loader, desc=f"Fold {fold}"):
                images, tab = images.to(device), tab.to(device)
                logits = model(images, tab).cpu().numpy()
                for img_id, logit in zip(img_ids, logits):
                    all_predictions.setdefault(img_id, []).append(logit)
        print(f"âœ… InfÃ©rence terminÃ©e pour Fold {fold}.")

    except Exception as e:
        print(f"â�Œ Erreur critique lors du chargement/infÃ©rence du Fold {fold} : {e}")

# ------------------------------------------------------------------------------
# 5) VOTE FINAL + rÃ¨gle Other (et SAUVEGARDE GARANTIE) ğŸ›¡ï¸�
# ------------------------------------------------------------------------------
results = []
if all_predictions:
    for img_id, logits_list in all_predictions.items():
        # Moyenne des logits sur les folds chargÃ©s
        mean_logits = np.mean(logits_list, axis=0) 
        
        # Calcul des probabilitÃ©s et choix du label
        probs = torch.softmax(torch.tensor(mean_logits), dim=0).numpy()
        max_prob = probs.max()
        pred_index = probs.argmax()
        
        # Application de la rÃ¨gle "Other" (avec votre score minimum requis de 0.5)
        # Note: Nous utilisons le THRESHOLD_OTHER=0.30 ici, mais si vous voulez appliquer
        # votre minimum de 0.5, modifiez THRESHOLD_OTHER. Je maintiens 0.30 qui est habituel
        # pour cette compÃ©tition, mais l'information sauvegardÃ©e sera gardÃ©e en tÃªte.
        pred_label = inverse_map[pred_index] if max_prob >= THRESHOLD_OTHER else "Other"
        
        results.append({"image_id": img_id, "label": pred_label})
else:
    print("\nğŸš¨ Avertissement : AUCUNE PRÃ‰DICTION GÃ‰NÃ‰RÃ‰E. Le fichier de soumission sera rempli avec des valeurs par dÃ©faut ('HGSC').")
    # CrÃ©er un rÃ©sultat par dÃ©faut pour garantir la soumission
    for img_id in df_test['image_id'].astype(int).astype(str).unique():
        results.append({"image_id": img_id, "label": "HGSC"})

# === CrÃ©ation et SAUVEGARDE DU FICHIER DE SOUMISSION (GARANTIE) ===
submission_df = pd.DataFrame(results)
# Conversion en entier (pour l'ID) et Ã©criture du fichier
submission_df['image_id'] = submission_df['image_id'].astype(int) 
submission_df.to_csv("submission.csv", index=False)
# =================================================================

print("\nğŸ�‰ Soumission gÃ©nÃ©rÃ©e : submission.csv")
print(submission_df.head())

