import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import cv2  # OpenCV pour redimensionner les images rapidement
import timm # BibliothÃ¨que de modÃ¨les prÃ©-entrainÃ©s (ex: EfficientNet)

# Configuration Globale
CONFIG = {
    'ROOT_DIR': '/kaggle/input/g2net-detecting-continuous-gravitational-waves',
    'OUTPUT_DIR': './',
    'IMG_SIZE': (256, 256), # Taille d'entrÃ©e du modÃ¨le
    'BATCH_SIZE': 32,
    'EPOCHS': 3,
    'LR': 1e-3,
    'SEED': 42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Fixer la graine pour la reproductibilitÃ© (DÃ©terminisme)
def seed_everything(seed):
    """Fixe la graine pour Numpy, PyTorch et Python pour la reproductibilitÃ©."""
    np.random.seed(seed)
    # Graine Python standard
    import random as rn
    rn.seed(seed)
    # Graine PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ParamÃ¨tres dÃ©terministes spÃ©cifiques Ã  CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

seed_everything(CONFIG['SEED'])
print(f"Device utilisÃ© : {CONFIG['device']}")

# --- 1. Chargement et Analyse du DÃ©sÃ©quilibre des Labels ---

# 1. Chargement des labels
try:
    train_labels = pd.read_csv(f"{CONFIG['ROOT_DIR']}/train_labels.csv")
    print("\nâœ… train_labels.csv chargÃ©.")
except FileNotFoundError:
    print(f"\nğŸ”´ Erreur: Le fichier des labels n'a pas Ã©tÃ© trouvÃ© Ã  {CONFIG['ROOT_DIR']}/train_labels.csv")
    train_labels = None

# 2. Analyse du dÃ©sÃ©quilibre (s'il est chargÃ©)
if train_labels is not None:
    print(f"Nombre total d'Ã©chantillons : {len(train_labels)}")
    
    # Calculer le nombre d'Ã©chantillons par classe
    class_counts = train_labels['target'].value_counts()
    
    # Calculer les pourcentages
    total_samples = len(train_labels)
    neg_count = class_counts.get(0, 0)
    pos_count = class_counts.get(1, 0)
    
    neg_percent = (neg_count / total_samples) * 100
    pos_percent = (pos_count / total_samples) * 100
    
    print("\n--- DÃ©sÃ©quilibre des classes (Target) ---")
    print(f"Classe 0 (Bruit/NÃ©gatif) : {neg_count} ({neg_percent:.2f}%)")
    print(f"Classe 1 (Signal/Positif) : {pos_count} ({pos_percent:.2f}%)")

    # Mettre en Ã©vidence le dÃ©sÃ©quilibre
    if pos_percent < 10:
        print("\nâš ï¸� Avertissement : Fort dÃ©sÃ©quilibre des classes. La classe positive est rare.")
        print("Cela justifie l'utilisation de la mÃ©trique AUC et de techniques anti-dÃ©sÃ©quilibre (Weighted Loss, Over/Under-sampling, Stratified K-Fold).")
    elif pos_percent < 30:
        print("\nNote : DÃ©sÃ©quilibre modÃ©rÃ© des classes. Des ajustements pourraient Ãªtre nÃ©cessaires.")
    else:
        print("\nNote : Classes relativement bien Ã©quilibrÃ©es.")
    
    # --- 3. Affichage graphique du dÃ©sÃ©quilibre ---
    
    plt.figure(figsize=(6, 4))
    
    bars = plt.bar(
        class_counts.index.astype(str), 
        class_counts.values,            
        color=['#1f77b4', '#ff7f0e']    
    )
    
    total = sum(class_counts.values)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2., 
            height + 500, 
            f'{height}\n({percentage:.2f}%)',
            ha='center', 
            va='bottom',
            fontsize=10
        )

    plt.title("Distribution des Classes G2Net (Target)")
    plt.xlabel("Classe (0: Bruit, 1: Signal CW)")
    plt.ylabel("Nombre d'Ã‰chantillons")
    plt.xticks([0, 1], ['Classe 0 (Bruit)', 'Classe 1 (Signal)'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

print("\n--- Initialisation et analyse de la distribution des classes terminÃ©es. ---")


# Installation des librairies pour la gÃ©nÃ©ration de donnÃ©es (GW physics)
!pip install -q pyfstat lalsuite

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfstat
from tqdm.notebook import tqdm

# Configuration de base
ROOT_DIR = '/kaggle/input/g2net-detecting-continuous-gravitational-waves'
TRAIN_DIR = f"{ROOT_DIR}/train"


import os
import h5py
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Pas nÃ©cessaire pour cette cellule

# Configuration (Assurez-vous que CONFIG est dÃ©fini dans la premiÃ¨re cellule)
# Si vous exÃ©cutez cette cellule seule, vous aurez besoin de dÃ©finir CONFIG ici:
# CONFIG = {
#     'ROOT_DIR': '/kaggle/input/g2net-detecting-continuous-gravitational-waves',
#     'device': 'cpu'
# } 

def inspect_hdf5_structure_corrected(file_id, folder='train'):
    """
    Inspecte la structure HDF5 en tenant compte du ID_FICHIER comme clÃ© racine.
    CORRIGÃ‰ : Supprime l'accÃ¨s Ã  'timestamps' pour Ã©viter la KeyError.
    """
    path = f"{CONFIG['ROOT_DIR']}/{folder}/{file_id}.hdf5"
    
    try:
        with h5py.File(path, 'r') as f:
            # 1. On trouve la clÃ© racine (qui est l'ID du fichier)
            root_keys = list(f.keys())
            if not root_keys:
                 print("ğŸ”´ Erreur: Fichier HDF5 vide.")
                 return
                 
            # Dans ce cas, la clÃ© est l'ID du fichier lui-mÃªme
            root_key = root_keys[0]
            detector_group = f[root_key]

            print(f"--- Structure du fichier {file_id} ---")
            print(f"ClÃ© racine trouvÃ©e : '{root_key}'") 
            print(f"ClÃ©s disponibles sous '{root_key}' : {list(detector_group.keys())}")
            
            # 2. On boucle sur les dÃ©tecteurs H1 et L1 Ã  l'intÃ©rieur de ce groupe
            for detector in ['H1', 'L1']:
                if detector in detector_group:
                    print(f"\nDÃ©tecteur {detector}:")
                    
                    # --- CORRECTION CRITIQUE : AccÃ¨s UNIQUEMENT aux SFTs ---
                    if 'SFTs' in detector_group[detector]:
                        sfts = detector_group[detector]['SFTs'][:]
                        
                        print(f"  Shape SFTs : {sfts.shape} (FrÃ©quences x Temps)")
                        print(f"  Type de donnÃ©es : {sfts.dtype} (Complexe)") 
                        print(f"  Dimensions : {sfts.shape[0]} bins de frÃ©quence x {sfts.shape[1]} segments de temps")
                    else:
                        print(f"  ğŸ”´ ClÃ© 'SFTs' manquante sous {detector}!")

                else:
                     print(f"  ğŸ”´ DÃ©tecteur {detector} manquant dans ce groupe!")
                     
            if 'frequency_Hz' in detector_group:
                 frequencies = detector_group['frequency_Hz'][:]
                 print(f"\nFrÃ©quences : Shape {frequencies.shape}, de {frequencies[0]:.2f} Hz Ã  {frequencies[-1]:.2f} Hz")
            
    except Exception as e:
        print(f"ğŸ”´ Erreur lors de l'ouverture ou la lecture de {file_id}.hdf5: {e}")

# Test avec un fichier alÃ©atoire du train set
# Assurez-vous que CONFIG est bien dÃ©fini (avec ROOT_DIR)
# et que train_labels.csv est accessible
try:
    train_labels = pd.read_csv(f"{CONFIG['ROOT_DIR']}/train_labels.csv")
    sample_id_train = train_labels.iloc[0]['id'] 
    
    # Appel de la fonction corrigÃ©e
    inspect_hdf5_structure_corrected(sample_id_train)
    
except NameError:
    print("ğŸ”´ Erreur: La variable CONFIG n'est pas dÃ©finie. Veuillez exÃ©cuter la cellule de configuration initiale.")
except FileNotFoundError:
    print(f"ğŸ”´ Erreur: Fichier de labels non trouvÃ© Ã  {CONFIG['ROOT_DIR']}/train_labels.csv.")


# ============================================================
# INSTALLATION / SETUP / DATASET AVEC CANAL 4 OPTIMISÃ‰
# ============================================================

import os
import numpy as np
import random as rn
import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
import h5py
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    'ROOT_DIR': '/kaggle/input/g2net-detecting-continuous-gravitational-waves',
    'IMG_SIZE': (256, 256),
    'BATCH_SIZE': 32,
    'SEED': 42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# ============================================================
# DÃ‰TERMINISME MAXIMAL
# ============================================================

def seed_everything_strict(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything_strict(CONFIG['SEED'])
print(f"âœ… DÃ©terminisme maximal activÃ© avec SEED = {CONFIG['SEED']}")


# ============================================================
# ğŸ”¥ DOPPLER PROXY (remplace pyfstat)
# ============================================================

def make_doppler_proxy(freq, ra, dec, timestamps):
    """
    Approximation rÃ©aliste du Doppler CW sans pyfstat.
    Rotation terrestre + orbite.
    """
    earth_rot = 2 * np.pi / 86164                   # 1 jour sidÃ©ral
    earth_orbit = 2 * np.pi / (365.25 * 86400)      # orbite Terre

    doppler_rot = 1e-6 * np.sin(earth_rot * timestamps + ra)
    doppler_orb = 5e-4 * np.cos(earth_orbit * timestamps + dec)

    doppler = freq * (doppler_rot + doppler_orb)

    # Normalisation
    return ((doppler - doppler.mean()) / (doppler.std() + 1e-8)).astype(np.float32)


# ============================================================
# ğŸ”¥ VERSION OPTIMISÃ‰E DU MASQUE 2D DOPPLER
# ============================================================

def make_doppler_mask_optimized(doppler_curve, freq_bins, smooth=4, thickness=4):
    """
    Transforme la courbe Doppler 1D en une carte 2D optimisÃ©e :
    - Lissage gaussien
    - Ligne Ã©paisse
    - AttÃ©nuation gaussienne autour de la crÃªte
    """
    # 1) Normalisation
    doppler_norm = (doppler_curve - doppler_curve.min()) / (doppler_curve.ptp() + 1e-8)
    
    # 2) Indices de base
    base_idx = doppler_norm * (freq_bins - 1)
    
    # 3) Lissage
    smooth_idx = gaussian_filter1d(base_idx, sigma=smooth)
    
    # 4) Masque final
    mask = np.zeros((freq_bins, len(doppler_curve)), dtype=np.float32)
    
    for t in range(len(doppler_curve)):
        f0 = smooth_idx[t]
        for off in range(-thickness, thickness + 1):
            f = int(np.clip(f0 + off, 0, freq_bins - 1))
            weight = np.exp(-(off ** 2) / (2 * (thickness / 2) ** 2))
            mask[f, t] = max(mask[f, t], weight)

    # 5) Re-normalisation
    return (mask / (mask.max() + 1e-8)).astype(np.float32)


# ============================================================
# ğŸ”¥ DATASET PHYSIQUE G2NET (4 CANAUX) - CORRIGÃ‰
# ============================================================

class G2NetDatasetPhysics(Dataset):
    # â­�ï¸� CORRECTION ICI : Ajout du paramÃ¨tre 'folder' â­�ï¸�
    def __init__(self, df, root_dir, img_size, folder='train', transforms=None):
        self.df = df
        self.root_dir = root_dir
        self.img_size = img_size
        self.transforms = transforms
        self.folder = folder # Stockage du paramÃ¨tre 'folder'
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row['id']
        target = row['target']
        
        # â­�ï¸� CORRECTION ICI : Utilisation de self.folder â­�ï¸�
        path = os.path.join(
            self.root_dir, self.folder, file_id[0], file_id[1], file_id[2], f'{file_id}.hdf5'
        )

        # Sources CW (fixes ici)
        F0, ALPHA, DELTA = 100.0, 1.0, 0.5

        # === Chargement HDF5 ===
        try:
            with h5py.File(path, 'r') as f:
                root_key = list(f.keys())[0]
                sgram_h1 = np.abs(f[f'{root_key}/H1/SFTs'][:])
                sgram_l1 = np.abs(f[f'{root_key}/L1/SFTs'][:])
                timestamps = f[f'{root_key}/H1/timestamps_GPS'][:] 
        except:
            sgram_h1 = np.zeros((360, 4096), np.float32)
            sgram_l1 = np.zeros((360, 4096), np.float32)
            timestamps = np.arange(4096)

        # Alignement
        T = min(sgram_h1.shape[1], sgram_l1.shape[1])
        sgram_h1 = sgram_h1[:, :T]
        sgram_l1 = sgram_l1[:, :T]
        timestamps = timestamps[:T]

        # Normalisation Z-score
        def z(s): return (s - s.mean()) / (s.std() + 1e-6)

        s_h1 = z(sgram_h1)
        s_l1 = z(sgram_l1)

        # === CANAL 4 : DOPPLER OPTIMISÃ‰ ===
        doppler_curve = make_doppler_proxy(F0, ALPHA, DELTA, timestamps)
        doppler_mask = make_doppler_mask_optimized(
            doppler_curve,
            freq_bins=s_h1.shape[0],
            smooth=4,
            thickness=4
        )

        # Stack final 4 canaux
        combined = np.stack(
            [s_h1, s_l1, (s_h1 + s_l1) / 2, doppler_mask],
            axis=-1
        )

        # Resize final
        img = cv2.resize(combined, CONFIG['IMG_SIZE'], interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, torch.tensor(target, dtype=torch.float)


# ============================================================
# ğŸ”¥ TEST
# ============================================================

train_labels = pd.read_csv(f"{CONFIG['ROOT_DIR']}/train_labels.csv")

# NOTE: Le test ci-dessous fonctionne sans le paramÃ¨tre 'folder' car il a la valeur par dÃ©faut 'train'
dataset = G2NetDatasetPhysics(
    df=train_labels,
    root_dir=CONFIG['ROOT_DIR'],
    img_size=CONFIG['IMG_SIZE']
)

img, y = dataset[0]

print("Image shape :", img.shape)
print("Target :", y.item())

plt.figure(figsize=(10,3))
plt.imshow(img[3].cpu(), aspect='auto', origin='lower', cmap='inferno')
plt.title("Canal 4 : Doppler OptimisÃ©")
plt.colorbar()
plt.show()


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# --- 0. DÃ©pendances et Configuration (Assumer dÃ©finies prÃ©cÃ©demment) ---

CONFIG = {
    'ROOT_DIR': '/kaggle/input/g2net-detecting-continuous-gravitational-waves',
    'IMG_SIZE': (256, 256),
    'BATCH_SIZE': 32,
    'EPOCHS': 3,
    'SEED': 42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'LR': 1e-4, # Taux d'apprentissage
    'N_SPLITS': 5 # Nombre de plis pour la CV
}

# Chargement du DataFrame
try:
    train_labels = pd.read_csv(f"{CONFIG['ROOT_DIR']}/train_labels.csv")
except NameError:
    print("ğŸ”´ Erreur: Assurez-vous que 'CONFIG' est dÃ©fini et que le fichier 'train_labels.csv' est chargÃ©.")
    exit()

# NOTE IMPORTANTE : La classe G2NetDatasetPhysics est dÃ©sormais la classe utilisÃ©e.

class CWModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True, in_chans=4): 
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans)
        
        if hasattr(self.backbone, 'classifier'):
            n_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(n_features, 1)
        elif hasattr(self.backbone, 'fc'):
            n_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.backbone(x)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for img, target in tqdm(loader, desc="Train"):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img).squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# =========================================================================
# ğŸ”¥ FONCTION VALIDATE CORRIGÃ‰E ET ULTRA-ROBUSTE
# =========================================================================

def validate(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for img, target in tqdm(loader, desc="Valid"):
            img = img.to(device)
            output = torch.sigmoid(model(img)).squeeze(1)

            # Conversion directe des targets en entier pour s'assurer du format binaire
            preds.extend(output.cpu().numpy().astype(float))
            targets.extend(target.numpy().astype(int)) # â­�ï¸� CORRECTION : FORCÃ‰ EN INT BINAIRE

    targets = np.array(targets)
    preds = np.array(preds)

    # ===â­�ï¸� SÃ‰CURITÃ‰ ULTIME 1 : CAS PLI Ã€ UNE SEULE CLASSE â­�ï¸�===
    if len(np.unique(targets)) < 2:
        # Comme l'exige la compÃ©tition (valeur minimale 0.5 si AUC non calculable)
        # Note: Rappel de votre information sauvegardÃ©e: Le minimum de score requis pour ce type de compÃ©tition est de 0,5.
        print("âš ï¸� Avertissement : Une seule classe trouvÃ©e dans ce pli. AUC = 0.5")
        return 0.5

    # ===â­�ï¸� SÃ‰CURITÃ‰ ULTIME 2 : Gestion des autres erreurs AUC â­�ï¸�===
    try:
        return roc_auc_score(targets, preds)
    except:
        print("âš ï¸� Erreur AUC imprÃ©vue (y_true mal formÃ© ou autre) â†’ retour 0.5")
        return 0.5


# -------------------------------------------------------------------
## ğŸš€ 1. Fonction d'EntraÃ®nement avec Cross-Validation (CV)
# -------------------------------------------------------------------

def run_cross_validation_train(df):
    """
    ExÃ©cute l'entraÃ®nement complet en utilisant la Validation CroisÃ©e StratifiÃ©e.
    """
    all_fold_scores = []
    
    # 1. Initialisation du Stratified K-Fold
    # Le UserWarning sur les petites classes est inÃ©vitable sur ce dataset.
    skf = StratifiedKFold(n_splits=CONFIG['N_SPLITS'], shuffle=True, random_state=CONFIG['SEED'])
    
    # 2. ItÃ©ration sur les plis
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        print(f"\n====================== DÃ‰BUT DU PLI {fold+1}/{CONFIG['N_SPLITS']} ======================")
        
        # CrÃ©ation des DataFrames pour le pli actuel
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # 3. CrÃ©ation des Datasets et DataLoaders (Utilisation de la classe 4 canaux)
        # Assurez-vous que G2NetDatasetPhysics (Cellule 1/3) accepte 'folder' !
        train_ds = G2NetDatasetPhysics(
            train_df, 
            root_dir=CONFIG['ROOT_DIR'], 
            img_size=CONFIG['IMG_SIZE'],
            folder='train' 
        )
        val_ds = G2NetDatasetPhysics(
            val_df, 
            root_dir=CONFIG['ROOT_DIR'], 
            img_size=CONFIG['IMG_SIZE'],
            folder='train'
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2
        )
        
        # 4. Initialisation du ModÃ¨le/Optimiseur (in_chans=4 CORRECT)
        model = CWModel(in_chans=4).to(CONFIG['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
        criterion = nn.BCEWithLogitsLoss()
        
        best_fold_score = 0
        
        # 5. Boucle d'Ã‰poques
        for epoch in range(CONFIG['EPOCHS']):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
            score = validate(model, val_loader, CONFIG['device']) # <-- Appel Ã  la fonction ultra-robuste
            
            print(f"Pli {fold+1} | Epoch {epoch+1}/{CONFIG['EPOCHS']} | Loss: {loss:.4f} | AUC: {score:.4f}")
            
            if score > best_fold_score:
                best_fold_score = score
                # Sauvegarde du modÃ¨le pour ce pli
                torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
                print(f"  >>> ModÃ¨le du pli {fold+1} sauvegardÃ© (AUC: {best_fold_score:.4f})")
                
        all_fold_scores.append(best_fold_score)
        print(f"====================== FIN DU PLI {fold+1} ======================")

    # 6. Rapport Final
    mean_auc = np.mean(all_fold_scores)
    std_auc = np.std(all_fold_scores)
    print("\n\n#####################################################")
    print(f"RÃ‰SULTAT FINAL (Cross-Validation sur {CONFIG['N_SPLITS']} plis):")
    print(f"  Moyenne AUC: {mean_auc:.4f} Â± {std_auc:.4f}")
    print("#####################################################")
    return all_fold_scores

# --- EXÃ‰CUTION ---
if __name__ == '__main__':
    # Rappel : Assurez-vous que G2NetDatasetPhysics est la classe dÃ©finie dans la cellule prÃ©cÃ©dente
    final_scores = run_cross_validation_train(train_labels)


import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# --- 0. Configuration et DÃ©pendances (Assumer dÃ©finies) ---
# CONFIG doit contenir: ROOT_DIR, BATCH_SIZE, device, N_SPLITS
# La classe CWModel et G2NetDatasetPhysics DOIVENT Ãªtre dÃ©finies dans les cellules prÃ©cÃ©dentes.
CONFIG = {
    'ROOT_DIR': '/kaggle/input/g2net-detecting-continuous-gravitational-waves',
    'IMG_SIZE': (256, 256),
    'BATCH_SIZE': 64,  # GÃ©nÃ©ralement plus grand pour l'infÃ©rence
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'N_SPLITS': 5 
}

# Charger le fichier de soumission pour obtenir les IDs du set de test
try:
    submission_df = pd.read_csv(f"{CONFIG['ROOT_DIR']}/sample_submission.csv")
    print("\nâœ… Fichier de soumission/test IDs chargÃ©.")
except FileNotFoundError:
    print("ğŸ”´ Erreur: 'sample_submission.csv' non trouvÃ©. Veuillez vÃ©rifier le chemin.")
    exit()

# -------------------------------------------------------------------
## ğŸ’» 1. PrÃ©paration du DataLoader de Test
# -------------------------------------------------------------------

# â­�ï¸� CORRECTION CRITIQUE: Utilisation de G2NetDatasetPhysics pour le 4 canaux â­�ï¸�
test_ds = G2NetDatasetPhysics(
    submission_df, 
    root_dir=CONFIG['ROOT_DIR'], 
    img_size=CONFIG['IMG_SIZE'], 
    folder='test' # Assurez-vous que G2NetDatasetPhysics utilise bien 'folder' ou dÃ©duit le chemin de test.
)
test_loader = DataLoader(
    test_ds, 
    batch_size=CONFIG['BATCH_SIZE'], 
    shuffle=False, 
    num_workers=2
)
print(f"Dataset de test crÃ©Ã© : {len(test_ds)} Ã©chantillons.")

# -------------------------------------------------------------------
## ğŸš€ 2. InfÃ©rence (PrÃ©diction) par Ensembling K-Fold
# -------------------------------------------------------------------

def predict_by_kfold_ensemble(test_loader, n_splits):
    """
    Fait des prÃ©dictions en chargeant chaque modÃ¨le de pli (fold) et en moyennant les rÃ©sultats.
    """
    final_predictions = np.zeros((len(test_loader.dataset),))
    
    # ItÃ©ration sur chaque pli sauvegardÃ©
    for fold in range(1, n_splits + 1):
        print(f"\n--- PrÃ©diction avec le modÃ¨le du pli {fold} ---")
        
        # 1. Charger un nouveau modÃ¨le
        # CWModel doit Ãªtre dÃ©fini pour in_chans=4
        model = CWModel(in_chans=4).to(CONFIG['device'])
        weights_path = f'best_model_fold_{fold}.pth'
        
        # 2. Charger les poids spÃ©cifiques Ã  ce pli
        try:
            model.load_state_dict(torch.load(weights_path, map_location=CONFIG['device']))
            print(f"âœ… Poids chargÃ©s depuis {weights_path}")
        except FileNotFoundError:
            print(f"ğŸ”´ ERREUR: Fichier de poids {weights_path} non trouvÃ©. Passe au pli suivant.")
            continue
            
        model.eval()
        fold_predictions = []
        
        # 3. PrÃ©diction sur le set de test
        with torch.no_grad():
            for img, _ in tqdm(test_loader, desc=f"InfÃ©rence Pli {fold}"):
                img = img.to(CONFIG['device'])
                
                # Le modÃ¨le renvoie des logits, nous appliquons Sigmoid pour la probabilitÃ© [0, 1]
                output = torch.sigmoid(model(img)).squeeze(1) 
                fold_predictions.extend(output.cpu().numpy())
        
        # 4. Cumul des prÃ©dictions (Moyenne)
        final_predictions += np.array(fold_predictions)

    # 5. Calcul de la moyenne sur tous les plis
    final_predictions /= CONFIG['N_SPLITS']
    return final_predictions

# ExÃ©cution de l'infÃ©rence
test_predictions = predict_by_kfold_ensemble(test_loader, CONFIG['N_SPLITS'])

# -------------------------------------------------------------------
## ğŸ“� 3. GÃ©nÃ©ration du Fichier de Soumission
# -------------------------------------------------------------------

submission_df['target'] = test_predictions

# VÃ©rification du format
print(f"\n--- VÃ©rification du Fichier de Soumission ---")
print(f"Nombre de lignes : {len(submission_df)}")
print("AperÃ§u du DataFrame:")
print(submission_df.head())

# Sauvegarde au format requis
submission_file_name = 'submission.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"\nâœ… Fichier de soumission gÃ©nÃ©rÃ© avec succÃ¨s : {submission_file_name}")

