import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf # ğŸ‘ˆ AJOUT DE TENSORFLOW POUR LA CONFIGURATION

# ======================================================================
# ğŸš¨ OPTIMISATION ANTI-EXPLOSION DE GRADIENT
# Forcer la prÃ©cision Ã  float64 pour Ã©viter les overflows/NaN en float32.
# ======================================================================
try:
    tf.keras.backend.set_floatx('float64')
    print("âœ… PrÃ©cision du Backend Keras dÃ©finie sur 'float64'.")
except Exception as e:
    print(f"â�Œ Avertissement : Ã‰chec de la dÃ©finition de float64. Raison : {e}")


# DÃ©finition du chemin de base (ajustÃ© selon votre structure fournie)
BASE_PATH = "/kaggle/input/h690/h690/h690"
# Assurez-vous que ce chemin est correct dans votre session Kaggle
print(f"Chemin de base dÃ©fini : {BASE_PATH}")


# Chemin d'accÃ¨s au fichier de mÃ©tadonnÃ©es
METADATA_PATH = os.path.join(BASE_PATH, 'jd_sherds_info.csv')

# Charger le DataFrame principal
df_meta = pd.read_csv(METADATA_PATH)

print(f"Chargement de {METADATA_PATH} rÃ©ussi.")
print("AperÃ§u des donnÃ©es :")
print(df_meta.head())
print("\nInformations sur les colonnes et valeurs manquantes :")
print(df_meta.info())


IMAGE_DIR = os.path.join(BASE_PATH, 'sherd_images')

# CrÃ©er une colonne pour l'ID du fragment
df_meta['sherd_id'] = df_meta['image_id'].apply(lambda x: x.split('_')[0])

# CrÃ©er la colonne de chemin d'accÃ¨s complet pour chaque image
def get_image_path(row):
    # Les images sont nommÃ©es JDxxxxx_exterior.jpg ou JDxxxxx_interior.jpg
    return os.path.join(IMAGE_DIR, f"{row['image_id']}.jpg")

df_meta['path'] = df_meta.apply(get_image_path, axis=1)

print(f"\nExemple de chemin d'accÃ¨s crÃ©Ã©: {df_meta['path'].iloc[0]}")


# 1. Nettoyage de la colonne 'unit' (couche stratigraphique)
# On simplifie pour l'entraÃ®nement : extraction du numÃ©ro de la couche
def clean_unit(unit_str):
    if pd.isna(unit_str):
        return 'UNKNOWN'
    if unit_str.startswith('L'):
        return int(unit_str[1:]) # Convertit L01 -> 1, L14 -> 14
    else:
        # Regrouper M, Z et autres cas spÃ©ciaux
        return 'UNKNOWN' 

df_meta['layer_num'] = df_meta['unit'].apply(clean_unit)
df_meta['is_unknown_layer'] = (df_meta['layer_num'] == 'UNKNOWN')

# 2. Filtrage par 'part' (Partie du vase)
# Remplir les valeurs manquantes pour la cohÃ©rence
df_meta['part'] = df_meta['part'].fillna('UNKNOWN_PART')

print("\nRÃ©partition des fragments par partie (part):")
print(df_meta['part'].value_counts())


import cv2
import matplotlib.pyplot as plt
import os

# DÃ©finition des variables de chemin (reprise des Ã©tapes prÃ©cÃ©dentes)
# Assumons que BASE_PATH et IMAGE_DIR sont dÃ©finis:
# BASE_PATH = "/kaggle/input/h690/h690/h690" 
# IMAGE_DIR = os.path.join(BASE_PATH, 'sherd_images') 

# Chemin d'accÃ¨s Ã  l'image du fragment JD00001 en utilisant la variable IMAGE_DIR
EXAMPLE_IMG_PATH = os.path.join(IMAGE_DIR, 'JD00001_exterior.jpg')


import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def extract_final_fragment_contour(image_path, show_plot=False):
    """
    Isole et nettoie le contour du fragment sur un fond clair.
    Utilise le Recadrage, Canny, la Dilatation (pour la fermeture) et le Masquage (pour le nettoyage).
    """
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Erreur de chargement de l'image : {image_path}")
        return None, None

    # --- 1. Recadrage pour Ã©liminer le cadre externe ---
    h, w, _ = img.shape
    crop_percent = 0.05 
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    img_cropped = img[crop_h:h-crop_h, crop_w:w-crop_w]
    
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    
    # --- 2. Flou Gaussien ---
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # --- 3. DÃ©tection de Bords Canny ---
    edges = cv2.Canny(img_blurred, 50, 150) 
    
    # --- 4. Dilatation (pour fermer les bords) ---
    kernel = np.ones((7, 7), np.uint8) 
    edges_dilated = cv2.dilate(edges, kernel, iterations=3) 
    
    # --- 5. Trouver les contours sur les bords dilatÃ©s ---
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Aucun contour principal trouvÃ© aprÃ¨s Canny/Dilatation. Ajustez les hyperparamÃ¨tres.")
        return None, None
        
    # Le plus grand contour est notre fragment (le seul Ã  conserver)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # -------------------------------------------------------------------
    # --- 6. Nettoyage du Contour (CrÃ©ation d'un Masque Propre) ---
    # Pour Ã©liminer les traits internes et les petits contours parasites,
    # nous crÃ©ons un masque rempli uniquement Ã  partir du plus grand contour.
    
    clean_mask = np.zeros(img_cropped.shape[:2], dtype=np.uint8)
    # Remplir le plus grand contour en blanc (255)
    cv2.drawContours(clean_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    
    # Maintenant, nous rÃ©cupÃ©rons le contour du masque PROPRE
    final_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_largest_contour = final_contours[0] if final_contours else largest_contour
    # -------------------------------------------------------------------

    # --- 7. Affichage (pour vÃ©rification) ---
    if show_plot:
        contour_img = img_cropped.copy()
        cv2.drawContours(contour_img, [final_largest_contour], -1, (0, 0, 255), 3) # Contour final en Bleu

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Bords Canny (DilatÃ©s)")
        plt.imshow(edges_dilated, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Masque Binaire NettoyÃ©")
        plt.imshow(clean_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Contour Final (Objet IsolÃ©)")
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Le contour renvoyÃ© est celui du masque propre
    return final_largest_contour, clean_mask


import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_fracture_area(largest_contour, fragment_mask, img_cropped_shape, show_plot=True):
    """
    Calcule et visualise l'Enveloppe Convexe pour identifier la zone de fracture.
    
    Args:
        largest_contour (np.ndarray): Le contour principal nettoyÃ© du fragment.
        fragment_mask (np.ndarray): Le masque binaire propre du fragment (pour la taille).
        img_cropped_shape (tuple): La forme de l'image recadrÃ©e originale (pour la visualisation).
        show_plot (bool): Afficher la visualisation.
    """
    if largest_contour is None or len(largest_contour) < 3:
        print("Contour non valide pour l'analyse de convexitÃ©.")
        return largest_contour

    # 1. Calculer l'Enveloppe Convexe (H)
    convex_hull = cv2.convexHull(largest_contour, returnPoints=True)
    
    # NOTE: L'extraction des DÃ©fauts de ConvexitÃ© (points 2 et 3) est laissÃ©e en commentaire.
    # C'est la bonne approche thÃ©orique, mais pour simplifier le pipeline DML,
    # nous laissons le DML apprendre la diffÃ©rence entre les segments lisses (Convex Hull)
    # et les segments irrÃ©guliers (Fracture).
    
    # 2. PrÃ©paration de l'image de visualisation
    # CrÃ©er une image 3 canaux (BGR) noire de la taille de l'image recadrÃ©e
    img_viz = np.zeros(img_cropped_shape, dtype=np.uint8) 
    
    # --- 3. Dessin des Contours ---
    
    # Dessiner le Masque/Fragment (Gris) comme arriÃ¨re-plan visuel
    cv2.drawContours(img_viz, [largest_contour], -1, (100, 100, 100), thickness=cv2.FILLED)

    # Dessiner le Contour Original (Vert)
    cv2.drawContours(img_viz, [largest_contour], -1, (0, 255, 0), 2)
    
    # Dessiner l'Enveloppe Convexe (Rouge)
    # C'est la ligne rouge qui "coupera les coins" de la fracture
    cv2.drawContours(img_viz, [convex_hull], -1, (255, 0, 0), 2) 
    
    if show_plot:
        plt.figure(figsize=(8, 8))
        plt.title("Contour (Vert) vs. Enveloppe Convexe (Rouge)")
        plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Pour l'entrÃ©e du DML, nous retournons le contour entier.
    return largest_contour


# Assurez-vous d'avoir dÃ©fini EXAMPLE_IMG_PATH et IMAGE_DIR dans une cellule prÃ©cÃ©dente
# import os
# IMAGE_DIR = "/kaggle/input/h690/h690/h690/sherd_images"
# EXAMPLE_IMG_PATH = os.path.join(IMAGE_DIR, 'JD00001_exterior.jpg') 

# ----------------------------------------------------------------------
# Ã‰tape 1 : Extraction et Nettoyage du Contour du Fragment
# ----------------------------------------------------------------------
final_contour, clean_mask = extract_final_fragment_contour(EXAMPLE_IMG_PATH, show_plot=True) 

if final_contour is not None:
    print(f"\nContour extrait avec succÃ¨s : {len(final_contour)} points. PrÃªt pour l'analyse de convexitÃ©.")
    
    # RÃ©cupÃ©rer la forme de l'image recadrÃ©e pour la visualisation (Ã©tape 7 de la fonction prÃ©cÃ©dente)
    # Si vous avez besoin de la forme exacte de l'image recadrÃ©e (pour aligner les couches)
    # on peut la dÃ©duire de la taille du masque
    img_cropped_shape = (*clean_mask.shape, 3) 
    
    # ------------------------------------------------------------------
    # Ã‰tape 2 : Analyse de ConvexitÃ© (Isolation de la Zone de Fracture)
    # ------------------------------------------------------------------
    fracture_contour = visualize_fracture_area(
        final_contour, 
        clean_mask, 
        img_cropped_shape, 
        show_plot=True
    )

    if fracture_contour is not None:
        print("\nAnalyse de la zone de fracture rÃ©ussie. PrÃªt pour la normalisation 3D.")

else:
    print("Ã‰chec de l'extraction. Ajustez les hyperparamÃ¨tres (Canny/Dilatation).")


import pandas as pd
import os
import glob
from typing import Dict, List, Tuple
import itertools
import random
import numpy as np
from sklearn.model_selection import train_test_split
import math
from tensorflow.keras.utils import Sequence
import tensorflow as tf
# ğŸš¨ NOUVEAUX IMPORTS POUR LE TRAITEMENT 2D
import cv2
import matplotlib.pyplot as plt
from typing import Callable 

# ======================================================================
# 0. DÃ‰FINITION DES CHEMINS & HYPERPARAMÃˆTRES MVCNN
# ======================================================================

# REMPLACER CETTE VALEUR PAR LE VRAI CHEMIN RACINE DE VOS DONNÃ‰ES.
BASE_PATH = '/kaggle/input/h690/h690/h690/'
IMAGE_DIR = os.path.join(BASE_PATH, 'sherd_images')
INFO_FILE_PATH = os.path.join(BASE_PATH, 'jd_sherds_info.csv')

# --- MVCNN HYPERPARAMÃˆTRES ---
N_VIEWS = 12       # Nombre de vues simulÃ©es (mÃªme masque dupliquÃ©)
IMG_HEIGHT = 224   # Taille standard des images pour les CNN 2D
IMG_WIDTH = 224
IMG_CHANNELS = 1   # 1 pour le masque (noir et blanc)

# DÃ©finition des noms de colonnes :
FRAGMENT_COL = 'sherd_id'
UNIT_COL = 'unit'
TYPE_COL = 'type'

print(f"BASE_PATH : {BASE_PATH}")

# ======================================================================
# D. DÃ‰FINITION DE LA CLASSE DMLDataGenerator (MVCNN)
# ======================================================================

# RÃ©cupÃ©rer le type de donnÃ©es actuel du backend (doit Ãªtre 'float64')
DTYPE_FLOAT = tf.keras.backend.floatx()
DTYPE_INT = np.uint8 # Pour les images (0-255)

class DMLDataGenerator(Sequence):
    """
    GÃ©nÃ©rateur de donnÃ©es Keras MVCNN (Multi-View CNN)
    Charge les masques 2D (contours) et les met en forme (N_VUES, H, W, C).
    """

    # ğŸš¨ MISE Ã€ JOUR : all_image_masks remplace all_point_clouds
    def __init__(self, dml_pairs: List[Tuple[str, str, int]], all_image_masks: Dict[str, np.ndarray], batch_size: int = 32, shuffle: bool = True):
        self.dml_pairs = dml_pairs
        self.all_image_masks = all_image_masks # Stocke les masques 2D (H, W)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # ğŸš¨ NOUVELLE FORME D'ENTRÃ‰E : (N_VUES, H, W, C)
        # La forme du masque 2D est (H, W).
        if all_image_masks:
            mask_h, mask_w = next(iter(all_image_masks.values())).shape
            self.input_shape = (N_VIEWS, mask_h, mask_w, IMG_CHANNELS)
        else:
            self.input_shape = (N_VIEWS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            
        self.on_epoch_end()

    def __len__(self):
        """ Nombre de lots par Ã©poque """
        return math.ceil(len(self.dml_pairs) / self.batch_size)

    def __getitem__(self, index):
        """ GÃ©nÃ¨re un lot de donnÃ©es """
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.dml_pairs))
        indexes = self.indexes[start_index:end_index]

        batch_pairs = [self.dml_pairs[k] for k in indexes]

        # ğŸš¨ X_A et X_B sont maintenant des TENSEURS 5D (BATCH, N_VUES, H, W, C)
        X_A = np.empty((len(batch_pairs), *self.input_shape), dtype=DTYPE_FLOAT)
        X_B = np.empty((len(batch_pairs), *self.input_shape), dtype=DTYPE_FLOAT)
        Y = np.empty((len(batch_pairs), 1), dtype=DTYPE_FLOAT)

        for i, (id_A, id_B, label) in enumerate(batch_pairs):
            
            # RÃ©cupÃ©ration du masque 2D (H, W)
            mask_A = self.all_image_masks.get(id_A, np.zeros(self.input_shape[1:3], dtype=DTYPE_INT))
            mask_B = self.all_image_masks.get(id_B, np.zeros(self.input_shape[1:3], dtype=DTYPE_INT))

            # 1. Mise Ã  l'Ã©chelle et conversion en float64 (0.0 Ã  1.0)
            img_A = mask_A.astype(DTYPE_FLOAT) / 255.0
            img_B = mask_B.astype(DTYPE_FLOAT) / 255.0
            
            # 2. Ajout du canal (H, W) -> (H, W, 1)
            img_A = np.expand_dims(img_A, axis=-1)
            img_B = np.expand_dims(img_B, axis=-1)
            
            # 3. CrÃ©ation des VUES (H, W, 1) -> (N_VUES, H, W, 1) par duplication
            X_A[i,] = np.tile(img_A, (N_VIEWS, 1, 1, 1))
            X_B[i,] = np.tile(img_B, (N_VIEWS, 1, 1, 1))

            Y[i,] = label

        return ({'input_A': X_A, 'input_B': X_B}, Y)

    def on_epoch_end(self):
        """ MÃ©langer les indices aprÃ¨s chaque Ã©poque si `shuffle` est vrai """
        self.indexes = np.arange(len(self.dml_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # DÃ©finition de la signature de sortie pour tf.data.Dataset.from_generator
    @property
    def output_signature(self):
        # ğŸš¨ NOUVELLE SIGNATURE : Tenseurs 5D
        input_spec = {
            'input_A': tf.TensorSpec(shape=(None, *self.input_shape), dtype=DTYPE_FLOAT),
            'input_B': tf.TensorSpec(shape=(None, *self.input_shape), dtype=DTYPE_FLOAT)
        }
        target_spec = tf.TensorSpec(shape=(None, 1), dtype=DTYPE_FLOAT)
        
        return (input_spec, target_spec)


# ======================================================================
# A. DÃ‰FINITION DE LA FONCTION 1 : Traitement de Tous les Fragments (2D Contour)
# ======================================================================

# Vous DEVEZ exÃ©cuter la cellule contenant cette fonction AVANT celle-ci
# (C'est la fonction extract_final_fragment_contour que vous avez montrÃ©e dans la cellule 7)
def process_all_fragments_by_id(image_dir: str, contour_extractor: Callable) -> Dict[str, np.ndarray]:
    """
    Charge les images, extrait les contours 2D (masques) et les redimensionne.
    
    Args:
        image_dir (str): Chemin vers les images.
        contour_extractor (Callable): Fonction dÃ©finie dans la cellule 7.
        
    Returns:
        Dict[str, np.ndarray]: Dictionnaire {ID_Complet: Masque_2D_RedimensionnÃ©}
    """
    
    all_image_masks = {}
    
    # Trouver toutes les images
    image_paths = glob.glob(os.path.join(image_dir, '*_exterior.jpg'))
    image_paths.extend(glob.glob(os.path.join(image_dir, '*_interior.jpg')))
    
    print(f"Tentative de traitement de {len(image_paths)} images...")

    for i, path in enumerate(image_paths):
        full_id = os.path.basename(path).replace('.jpg', '')
        
        # --- LOGIQUE D'EXTRACTION DE CONTOUR (Cellule 7) ---
        _, mask = contour_extractor(path, show_plot=False)
        
        if mask is not None:
            # Redimensionnement du masque Ã  la taille standard MVCNN (224x224)
            mask_resized = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            all_image_masks[full_id] = mask_resized
        else:
            # Utiliser un masque vide si l'extraction Ã©choue
            all_image_masks[full_id] = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=DTYPE_INT)
        
        if (i + 1) % 5000 == 0:
             print(f"  Fragments traitÃ©s: {i+1}")
             
    print(f"Traitement terminÃ©. {len(all_image_masks)} masques 2D (contours) gÃ©nÃ©rÃ©s/simulÃ©s.")
    return all_image_masks

# ----------------------------------------------------------------------

# ======================================================================
# B. DÃ‰FINITION DE LA FONCTION 2 : CrÃ©ation des Paires DML (inchangÃ©e)
# ======================================================================

def create_dml_pairs(all_image_masks: Dict[str, np.ndarray], assembly_map_full: Dict[str, str]) -> List[Tuple[str, str, int]]:
    """
    CrÃ©e les paires positives et nÃ©gatives. (Utilise all_image_masks pour la clÃ©, mais la logique est la mÃªme)
    """
    # Reste inchangÃ©e (logique de paires uniquement)
    all_fragments = list(all_image_masks.keys())
    assembly_map = {k: v for k, v in assembly_map_full.items() if k in all_fragments}
    
    # 1. Paires Positives (Label=1)
    positive_pairs = []
    groups = {}
    for full_id, group in assembly_map.items():
        if group not in groups: groups[group] = []
        groups[group].append(full_id)
        
    for sherd_list in groups.values():
        if len(sherd_list) >= 2:
            for sherd_A, sherd_B in itertools.combinations(sherd_list, 2):
                positive_pairs.append(tuple(sorted((sherd_A, sherd_B))) + (1,))
                
    df_positive_pairs = pd.DataFrame(positive_pairs, columns=['sherd_A', 'sherd_B', 'label']).drop_duplicates()
    n_positives = len(df_positive_pairs)
    positive_pairs = list(df_positive_pairs.itertuples(index=False, name=None))
    print(f"  - {n_positives} Paires POSITIVES gÃ©nÃ©rÃ©es.")

    # 2. Paires NÃ©gatives (Label=0)
    N_NEG_TARGET = max(100, n_positives * 2)
    all_labeled_fragments = list(assembly_map.keys())
    
    num_attempts = int(N_NEG_TARGET * 3)
    if len(all_labeled_fragments) < 2:
        print("ATTENTION: Pas assez de fragments labellisÃ©s pour gÃ©nÃ©rer des nÃ©gatifs.")
        return positive_pairs
        
    random_ids_A = np.random.choice(all_labeled_fragments, size=num_attempts, replace=True)
    random_ids_B = np.random.choice(all_labeled_fragments, size=num_attempts, replace=True)

    negative_pairs = []
    seen_pairs = set(df_positive_pairs[['sherd_A', 'sherd_B']].apply(tuple, axis=1).tolist())

    for id_A, id_B in zip(random_ids_A, random_ids_B):
        if id_A == id_B: continue
        
        ordered_pair = tuple(sorted((id_A, id_B)))
        
        group_A = assembly_map.get(id_A)
        group_B = assembly_map.get(id_B)

        if group_A and group_B and group_A != group_B and ordered_pair not in seen_pairs:
            seen_pairs.add(ordered_pair)
            negative_pairs.append(ordered_pair + (0,))

            if len(negative_pairs) >= N_NEG_TARGET:
                break
    
    negative_pairs = negative_pairs[:N_NEG_TARGET]
    n_negatives = len(negative_pairs)
    print(f"  - {n_negatives} Paires NÃ‰GATIVES gÃ©nÃ©rÃ©es (OptimisÃ©).")

    # 3. Dataset Final
    all_dml_pairs = positive_pairs + negative_pairs
    random.shuffle(all_dml_pairs)

    return all_dml_pairs

# ----------------------------------------------------------------------

# ======================================================================
# C. LOGIQUE D'EXÃ‰CUTION DU PIPELINE DML (DÃ‰MARRAGE + DIVISION TRAIN/VAL)
# ======================================================================

# Charger le fichier d'information complet
try:
    df_info = pd.read_csv(INFO_FILE_PATH)
except FileNotFoundError:
    print(f"â�Œ ERREUR: Le fichier {INFO_FILE_PATH} est introuvable. VÃ©rifiez BASE_PATH.")
    exit()

# 1. CRÃ‰ATION DU SIGNAL DE VÃ‰RITÃ‰ TERRAIN (unit + type)
df_info['ASSEMBLY_GROUP_ID'] = df_info[UNIT_COL].astype(str) + '_' + df_info[TYPE_COL].astype(str)
df_info_labeled = df_info.dropna(subset=[UNIT_COL, TYPE_COL]).copy()
ASSEMBLY_MAP_BASE = df_info_labeled.set_index(FRAGMENT_COL)['ASSEMBLY_GROUP_ID'].to_dict()
print(f"\nâœ… Fragments de base labellisÃ©s (unit + type) : {len(ASSEMBLY_MAP_BASE)}")

# 2. EXTENSION DE LA CARTE AUX VUES 2D (EXTERIOR/INTERIOR)
ASSEMBLY_MAP_FULL = {}
for base_id, group_name in ASSEMBLY_MAP_BASE.items():
    ASSEMBLY_MAP_FULL[f"{base_id}_exterior"] = group_name
    ASSEMBLY_MAP_FULL[f"{base_id}_interior"] = group_name
print(f"âœ… VÃ©ritÃ© Terrain (ASSEMBLY_MAP_FULL) crÃ©Ã©e pour {len(ASSEMBLY_MAP_FULL)} vues de fragments.")

# ğŸš¨ DÃ‰FINITION DE LA FONCTION D'EXTRACTION DE CONTOUR
# IMPORTANT: Assurez-vous que la fonction 'extract_final_fragment_contour' (de la Cellule 7) est dÃ©finie avant d'exÃ©cuter ceci!
try:
    # Ceci va lever une NameError si la Cellule 7 n'a pas Ã©tÃ© exÃ©cutÃ©e.
    contour_extractor_func = extract_final_fragment_contour
except NameError:
    print("\n-------------------------------------------------------------")
    print("â�Œ ERREUR CRITIQUE: La fonction 'extract_final_fragment_contour' (Cellule 7) n'est pas dÃ©finie.")
    print("Veuillez exÃ©cuter la cellule qui dÃ©finit cette fonction (avec cv2) avant la Cellule 5.")
    print("-------------------------------------------------------------")
    exit()


# 3. EXÃ‰CUTION DU PIPELINE DML
print("\n--- DÃ©marrage de la gÃ©nÃ©ration des masques 2D (Pipeline Image -> Contour) ---")
# ğŸš¨ MISE Ã€ JOUR : all_image_masks remplace all_point_clouds
all_image_masks = process_all_fragments_by_id(IMAGE_DIR, contour_extractor_func)

# Diagnostic de l'intersection
processed_ids = set(all_image_masks.keys())
mapped_ids = set(ASSEMBLY_MAP_FULL.keys())
overlap_ids = processed_ids.intersection(mapped_ids)
print(f"\n--- DIAGNOSTIC D'INTERSECTION ---")
print(f"Fragments 2D avec VÃ©ritÃ© Terrain correspondante (overlap) : {len(overlap_ids)}")

if len(overlap_ids) < 2:
    print("â�Œ Ã‰CHEC : Moins de 2 fragments en commun. Le set d'entraÃ®nement est insuffisant.")
else:
    print("\n--- CrÃ©ation des paires DML complÃ¨tes ---")
    # ğŸš¨ MISE Ã€ JOUR : all_image_masks remplace all_point_clouds
    dml_all_pairs = create_dml_pairs(all_image_masks, ASSEMBLY_MAP_FULL)
    
    print(f"\nTotal des paires DML gÃ©nÃ©rÃ©es : {len(dml_all_pairs)}")

    # 4. DIVISION TRAIN / VALIDATION / TEST (avec stratification)
    
    # SÃ©paration initiale : Train/Val vs. Test (80% / 20%)
    train_val_pairs, dml_test_pairs = train_test_split(
        dml_all_pairs,
        test_size=0.2,
        random_state=42,
        stratify=[p[2] for p in dml_all_pairs]
    )
    
    # SÃ©paration secondaire : Train vs. Validation (75% / 25% de train_val)
    dml_train_pairs, dml_val_pairs = train_test_split(
        train_val_pairs,
        test_size=0.25, # Ce qui correspond Ã  20% du total
        random_state=42,
        stratify=[p[2] for p in train_val_pairs]
    )
    
    # AFFICHAGE DES RÃ‰SULTATS
    print("\n--- RÃ‰SULTATS DE LA DIVISION DES PAIRES (Train/Val/Test) ---")
    print(f"âœ… dml_train_pairs (60% pour l'entraÃ®nement) : {len(dml_train_pairs)} paires")
    print(f"âœ… dml_val_pairs (20% pour la validation) : {len(dml_val_pairs)} paires")
    print(f"âœ… dml_test_pairs (20% pour l'Ã©valuation) : {len(dml_test_pairs)} paires")


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalMaxPooling2D, Input, Dense, Reshape, TimeDistributed, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# ======================================================================
# 0. HYPERPARAMÃˆTRES ET CONFIGURATION (Doivent correspondre Ã  la Cellule 5)
# ======================================================================
# RÃ©cupÃ©rer le type de donnÃ©es actuel du backend (float64 pour la stabilitÃ©)
DTYPE_FLOAT = tf.keras.backend.floatx()

N_VIEWS = 12       
IMG_HEIGHT = 224   
IMG_WIDTH = 224    
IMG_CHANNELS = 1   
EMBEDDING_DIM = 64 # Dimension de l'embedding final (plus petit pour commencer)

# ======================================================================
# A. DÃ‰FINITION DE L'ENCODEUR MVCNN
# ======================================================================

def create_mvcnn_encoder(input_shape=(N_VIEWS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """
    CrÃ©e un encodeur MVCNN qui gÃ©nÃ¨re un embedding stable Ã  partir de N vues 2D.
    """
    # L'entrÃ©e est un tensor de (N_VUES, H, W, C)
    input_views = Input(shape=input_shape, name='mvcnn_input_views', dtype=DTYPE_FLOAT)

    # --- 1. ENCODEUR DE VUE (CNN 2D partagÃ©) ---
    def create_view_cnn():
        """Un petit CNN 2D pour extraire les features d'une seule vue (H, W, C)."""
        input_single_view = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        
        # Blocs Convolutifs (similaire Ã  une base VGG simplifiÃ©e)
        x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(input_single_view)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Max Pooling pour rÃ©duire la dimension spatiale avant le Dense
        x = GlobalMaxPooling2D()(x) # RÃ©duit (H', W', C') -> (C')

        # Couche Dense pour obtenir l'embedding de la vue (taille = EMBEDDING_DIM)
        x = Dense(EMBEDDING_DIM, activation='relu', kernel_regularizer=l2(1e-4))(x)
        
        return Model(inputs=input_single_view, outputs=x)

    # Appliquer le CNN Ã  toutes les N_VIEWS
    view_cnn_model = create_view_cnn()
    # view_features a la forme (BATCH_SIZE, N_VUES, EMBEDDING_DIM)
    view_features = TimeDistributed(view_cnn_model, name='TimeDistributed_CNN')(input_views) 

    # --- 2. AGRÃ‰GATION (Global Max Pooling sur l'axe des vues) ---
    # Pour chaque dimension d'embedding, on prend la valeur maximale sur l'ensemble des 12 vues.
    final_embedding = Lambda(lambda x: K.max(x, axis=1), 
                             output_shape=(EMBEDDING_DIM,), 
                             name='mvcnn_embedding')(view_features)
    
    # 3. Normalisation L2 de l'embedding (Crucial pour le Triplet Loss)
    final_embedding = Lambda(lambda x: K.l2_normalize(x, axis=1), name='L2_Normalize')(final_embedding)

    return Model(inputs=input_views, outputs=final_embedding, name='MVCNN_Encoder')

# ======================================================================
# B. CRÃ‰ATION DU MODÃˆLE SIAMOIS DML
# ======================================================================

def create_siamese_model(encoder: Model):
    """
    CrÃ©e le modÃ¨le Siamois pour le DML.
    """
    # 1. DÃ©finition des entrÃ©es pour la paire A et B
    input_A = Input(shape=encoder.input_shape[1:], name='input_A', dtype=DTYPE_FLOAT)
    input_B = Input(shape=encoder.input_shape[1:], name='input_B', dtype=DTYPE_FLOAT)

    # 2. Partager le mÃªme encodeur pour les deux entrÃ©es
    embedding_A = encoder(input_A)
    embedding_B = encoder(input_B)

    # 3. Couche de SimilaritÃ© (Distance L1 - cruciale pour le Triplet Loss)
    # Calcule la distance absolue entre les deux embeddings (Similitude = -Distance)
    distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), name='L1_distance')([embedding_A, embedding_B])
    
    # Couche dense finale pour prÃ©dire la similaritÃ© (0 ou 1)
    # L'activation sigmoÃ¯de permet de forcer la sortie entre 0 et 1.
    similarity_score = Dense(1, activation='sigmoid', name='similarity_output', dtype=DTYPE_FLOAT)(distance)

    # ModÃ¨le complet
    siamese_model = Model(inputs=[input_A, input_B], outputs=similarity_score, name='Siamese_MVCNN')
    
    return siamese_model

# ======================================================================
# C. INSTANCIATION ET COMPILATION DU MODÃˆLE
# ======================================================================

# 1. CrÃ©ation de l'encodeur MVCNN
mvcnn_encoder = create_mvcnn_encoder()

# 2. CrÃ©ation du modÃ¨le Siamois
siamese_model = create_siamese_model(mvcnn_encoder)

# 3. Compilation
siamese_model.compile(
    # Le Triplet Loss est implicitement gÃ©rÃ© par les paires (y_true) et la Loss Binaire
    # Nous utilisons Binary Cross-Entropy et le modÃ¨le apprend Ã  ajuster la distance
    # pour que l'output_sigmoide soit proche de 1 pour les paires positives.
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')] # AUC est plus pertinent que l'accuracy
)

print(f"âœ… ModÃ¨le MVCNN Encoder crÃ©Ã©. Dimension de l'embedding : {EMBEDDING_DIM} (Type : {DTYPE_FLOAT})")
print(f"âœ… ModÃ¨le Siamois compilÃ© avec Binary Cross-Entropy.")

# Affichage du rÃ©sumÃ© pour vÃ©rifier les formes
print("\n--- RÃ©sumÃ© de l'Encodeur MVCNN (Partie TimeDistributed/Pooling) ---")
mvcnn_encoder.summary()

print("\n--- RÃ©sumÃ© du ModÃ¨le Siamois (Pipeline Complet) ---")
siamese_model.summary()


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
import os
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc

# Note: DMLDataGenerator, siamese_model, dml_train_pairs, dml_val_pairs, et all_image_masks
# DOIVENT Ãªtre dÃ©finis dans les cellules prÃ©cÃ©dentes pour que ceci fonctionne.

# ======================================================================
# 1. DÃ‰FINITION DES HYPERPARAMÃˆTRES D'ENTRAÃ�NEMENT MVCNN
# ======================================================================
EPOCHS = 5
BATCH_SIZE = 32
# ğŸš¨ CORRECTION DU VALUE ERROR (1/2) : Utiliser .weights.h5 quand save_weights_only=True
CHECKPOINT_PATH = "best_siamese_mvcnn_weights.weights.h5" 

# ======================================================================
# 2. CRÃ‰ATION DES GÃ‰NÃ‰RATEURS DE DONNÃ‰ES (MVCNN)
# ======================================================================

print("--- PrÃ©paration des gÃ©nÃ©rateurs de donnÃ©es MVCNN (DÃ©pendance Cellule 5) ---")

try:
    # A. CrÃ©ation des gÃ©nÃ©rateurs Keras Sequence
    train_generator = DMLDataGenerator(
        dml_pairs=dml_train_pairs,
        all_image_masks=all_image_masks, # Utilisation des masques 2D
        batch_size=BATCH_SIZE,
        shuffle=True # Le shuffle est nÃ©cessaire pour un bon apprentissage
    )

    val_generator = DMLDataGenerator(
        dml_pairs=dml_val_pairs,
        all_image_masks=all_image_masks, # Utilisation des masques 2D
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
except NameError as e:
    print("\n-------------------------------------------------------------")
    print(f"â�Œ Erreur critique : {e}. Assurez-vous que la Cellule 5 (Pipeline de DonnÃ©es) est exÃ©cutÃ©e.")
    print("-------------------------------------------------------------")
    exit()

print(f"âœ… GÃ©nÃ©rateur d'entraÃ®nement crÃ©Ã© : {len(train_generator)} lots.")
print(f"âœ… GÃ©nÃ©rateur de validation crÃ©Ã© : {len(val_generator)} lots.")


# ======================================================================
# 3. DÃ‰FINITION DES CALLBACKS (Adaptation pour Keras standard)
# ======================================================================
callbacks_list = [
    ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=True,
        monitor='val_loss', 
        mode='min',
        save_weights_only=True, # Sauvegarder uniquement les poids pour l'infÃ©rence
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=2, # AugmentÃ© pour donner plus de chance au MVCNN
        mode='min',
        restore_best_weights=True,
        verbose=1
    ),
    # ArrÃªte l'entraÃ®nement si la perte devient NaN
    TerminateOnNaN(),
    # RÃ©duit le LR si la perte de validation stagne
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-08, 
        verbose=1
    )
]


# ======================================================================
# 4. LANCEMENT DE L'ENTRAÃ�NEMENT (Utilisation directe du Keras Sequence)
# ======================================================================
print("\n--- DÃ©marrage de l'entraÃ®nement du ModÃ¨le Siamois MVCNN ---") 

try:
    # ğŸš¨ CORRECTION CRITIQUE DU WORKERS (2/2) : Retrait de l'argument 'workers' pour Keras 3
    history = siamese_model.fit(
        train_generator, # Utilisation directe de la Keras Sequence
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks_list,
        # workers=8 <-- Ligne supprimÃ©e
    )
    
except Exception as e:
    print("\n-------------------------------------------------------------")
    # Cette erreur est dÃ©sormais plus susceptible d'Ãªtre une vraie erreur d'exÃ©cution
    print(f"â�Œ ERREUR LORS DE L'ENTRAÃ�NEMENT: {e}")
    print("-------------------------------------------------------------")


# ======================================================================
# 5. Ã‰VALUATION FINALE ET SAUVEGARDE DE L'ENCODEUR
# ======================================================================

# 5.1 Chargement des meilleurs poids
if os.path.exists(CHECKPOINT_PATH):
    siamese_model.load_weights(CHECKPOINT_PATH)
    print(f"\nâœ… Meilleurs poids chargÃ©s depuis {CHECKPOINT_PATH}.")
else:
    print("\nâš ï¸� ATTENTION: Les poids optimaux n'ont pas pu Ãªtre chargÃ©s.")


# 5.2 Ã‰valuation finale
print("\n--- Ã‰valuation finale sur le jeu de test ---")
test_generator = DMLDataGenerator(
    dml_pairs=dml_test_pairs, 
    all_image_masks=all_image_masks, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

try:
    # Evaluation du modÃ¨le sur le gÃ©nÃ©rateur de test
    loss, acc = siamese_model.evaluate(test_generator, verbose=1)[:2] 
    
    print(f"**RÃ‰SULTATS FINAUX (Test Set) :**")
    print(f"- Loss: {loss:.4f}")
    print(f"- Accuracy: {acc:.4f}")
    
except Exception as e:
    print(f"â�Œ Erreur lors de l'Ã©valuation finale : {e}")


# 5.3 Sauvegarde de l'Encodeur seul
ENCODER_PATH = 'mvcnn_encoder_final.h5'
try:
    # L'encodeur MVCNN est le layer qui se nomme 'MVCNN_Encoder'
    mvcnn_encoder = siamese_model.get_layer('MVCNN_Encoder') 
    mvcnn_encoder.save(ENCODER_PATH)
    print(f"\nâœ… ModÃ¨le d'Embedding (MVCNN Encoder) sauvegardÃ© pour la prÃ©diction sous : {ENCODER_PATH}")
except Exception as e:
    print(f"\nâš ï¸� ATTENTION: Ã‰chec de la sauvegarde de l'encodeur MVCNN. Raison : {e}")


import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# ======================================================================
# 0. PRÃ‰PARATION ET CHARGEMENT
# ======================================================================

# ğŸš¨ Assurez-vous que l'encodeur a Ã©tÃ© sauvegardÃ© dans la Cellule 10 !
ENCODER_PATH = 'mvcnn_encoder_final.h5'
EMBEDDING_DIM = 64
N_VIEWS = 12 
IMG_HEIGHT = 224

if not os.path.exists(ENCODER_PATH):
    print(f"â�Œ ERREUR : Encodeur non trouvÃ© Ã  {ENCODER_PATH}. ExÃ©cutez la Cellule 10 d'entraÃ®nement en premier.")
    exit()

# Chargement de l'encodeur MVCNN
# NÃ©cessite custom_objects si des couches personnalisÃ©es avaient Ã©tÃ© utilisÃ©es (ici, non nÃ©cessaire)
# si le modÃ¨le a Ã©tÃ© sauvegardÃ© avec .save(). Nous utilisons load_model() pour le recharger.
# NOTE: Le modÃ¨le a Ã©tÃ© sauvegardÃ© via mvcnn_encoder.save() dans la Cellule 10
mvcnn_encoder = load_model(ENCODER_PATH) 
print(f"âœ… Encodeur MVCNN chargÃ© depuis {ENCODER_PATH}. Dim d'Embedding: {EMBEDDING_DIM}")

# ======================================================================
# 1. GÃ‰NÃ‰RATION DES EMBEDDINGS POUR TOUS LES FRAGMENTS
# ======================================================================

# ğŸš¨ all_image_masks et image_ids doivent Ãªtre disponibles (dÃ©finis dans Cellule 5)
try:
    image_ids = list(all_image_masks.keys())
    # CrÃ©ation de la liste d'entrÃ©es (MVCNN prend (N_VIEWS, H, W, C))
    all_inputs = np.stack([all_image_masks[id] for id in image_ids])
except NameError:
    print("â�Œ ERREUR: all_image_masks ou image_ids non dÃ©finis. ExÃ©cutez la Cellule 5.")
    exit()

print(f"GÃ©nÃ©ration des embeddings pour {len(image_ids)} fragments (MVCNN prÃ©dit sur {all_inputs.shape})...")

# PrÃ©diction
all_embeddings = mvcnn_encoder.predict(all_inputs, batch_size=32, verbose=1)

# CrÃ©ation du dictionnaire d'embeddings (ID image -> Embedding)
embeddings_map = dict(zip(image_ids, all_embeddings))
embeddings_matrix = np.array(all_embeddings)

print(f"âœ… Matrice d'Embeddings gÃ©nÃ©rÃ©e : {embeddings_matrix.shape}")

# ======================================================================
# 2. CLUSTERING ET Ã‰VALUATION INTERNE (MÃ©triques sans vÃ©ritÃ© terrain)
# ======================================================================

# Utilisation d'AgglomerativeClustering pour forcer N_CLUSTERS (les N familles)
# ğŸš¨ N_CLUSTERS doit Ãªtre connu! (HypothÃ¨se: nombre de classes dans dml_train_pairs)
try:
    all_classes = [id.split('_')[0] for id in image_ids]
    true_labels_clustering = np.array(all_classes)
    N_CLUSTERS = len(np.unique(true_labels_clustering))
    print(f"\nHypothÃ¨se: Nombre de clusters (classes rÃ©elles) = {N_CLUSTERS}")
except:
    N_CLUSTERS = 20 # Valeur par dÃ©faut si les labels ne sont pas dÃ©finis

# Algorithme 1: Agglomerative Clustering (HiÃ©rarchique)
agg_cluster = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric='euclidean', linkage='ward')
agg_labels = agg_cluster.fit_predict(embeddings_matrix)

# Algorithme 2: DBSCAN (basÃ© sur la densitÃ©, utile si les clusters ont des formes complexes)
# ğŸ’¡ Ces hyperparamÃ¨tres dÃ©pendent fortement des donnÃ©es
dbscan_cluster = DBSCAN(eps=0.15, min_samples=3, metric='euclidean') 
dbscan_labels = dbscan_cluster.fit_predict(embeddings_matrix)


# Ã‰valuation Interne (Indice de Silhouette)
# Le score de Silhouette mesure la densitÃ© et la sÃ©paration des clusters. Plus c'est proche de 1, mieux c'est.
try:
    agg_score = silhouette_score(embeddings_matrix, agg_labels)
    dbscan_score = silhouette_score(embeddings_matrix, dbscan_labels)
except ValueError:
    agg_score = 'N/A'
    dbscan_score = 'N/A (moins de 2 clusters)'

print("\n--- Ã‰valuation Interne (Clustering) ---")
print(f"Silhouette Score (AgglomÃ©ratif, N={N_CLUSTERS}) : {agg_score:.4f}")
print(f"Silhouette Score (DBSCAN) : {dbscan_score}")


# ======================================================================
# 3. Ã‰VALUATION EXTERNE (Si les vrais labels sont disponibles)
# ======================================================================

if 'true_labels_clustering' in locals() and len(np.unique(true_labels_clustering)) > 1:
    
    # NMI: Normalized Mutual Information (proche de 1 = excellent)
    nmi_agg = normalized_mutual_info_score(true_labels_clustering, agg_labels)
    nmi_dbscan = normalized_mutual_info_score(true_labels_clustering, dbscan_labels)

    print("\n--- Ã‰valuation Externe (VÃ©ritÃ© Terrain) ---")
    print(f"NMI (AgglomÃ©ratif) : {nmi_agg:.4f}")
    print(f"NMI (DBSCAN) : {nmi_dbscan:.4f}")

# ======================================================================
# 4. VISUALISATION (RÃ©duction de dimensionnalitÃ© via t-SNE/UMAP - NÃ©cessite librairies)
# ======================================================================

# NOTE : Pour garder le code simple, cette partie est laissÃ©e comme suggestion.
# NÃ©cessite l'installation des librairies t-SNE (scikit-learn) ou UMAP (umap-learn)
print("\nğŸ’¡ Pour visualiser, utilisez t-SNE ou UMAP sur les 'embeddings_matrix' colorÃ©s par 'agg_labels' ou 'true_labels_clustering'.")


import pandas as pd
import os
import numpy as np

# --- 0. DÃ©finition des Chemins et Variables ClÃ©s ---

# Chemins : Assurez-vous que BASE_PATH est dÃ©fini ou utilisez une valeur par dÃ©faut.
if 'BASE_PATH' not in locals():
    BASE_PATH = './'

INFO_FILE_PATH = os.path.join(BASE_PATH, 'jd_sherds_info.csv') 
FRAGMENT_COL_SUBMISSION = 'image_id'

# ğŸ’¥ VÃ‰RIFICATION CRITIQUE : Utiliser les rÃ©sultats de la Cellule 11 (Agglomerative Clustering)
try:
    # cluster_ids, agg_labels doivent Ãªtre dÃ©finis par la Cellule 11
    _ = cluster_ids
    _ = agg_labels
    _ = embeddings_map 
    
    # ğŸš¨ DÃ©finition des groupes basÃ©s sur le clustering hiÃ©rarchique
    assembly_groups = {
        cluster_ids[i]: f"ModelGroup_{agg_labels[i]}" 
        for i in range(len(cluster_ids))
    }
    
    # Fragments qui n'ont pas d'embeddings (si il y en a, ils sont dans 'unmatched_ids')
    unmatched_ids = [id for id in list(embeddings_map.keys()) if id not in cluster_ids] # Au cas oÃ¹
    
except NameError:
    print("â�Œ ERREUR: Les variables 'cluster_ids', 'agg_labels', ou 'embeddings_map' de la Cellule 11 sont manquantes.")
    exit()

print(f"IDs clusterisÃ©s trouvÃ©s : {len(cluster_ids)}")


# --- 1. CrÃ©ation du Fichier de Soumission Final ---

final_submission_data = []

# Ajout des fragments clusterisÃ©s
for id_ in cluster_ids:
    final_submission_data.append({
        FRAGMENT_COL_SUBMISSION: id_,
        'Assembly Group': assembly_groups[id_]
    })
    
# Ajout des fragments "singletons" (unmatched) (S'il y a des IDs dans le fichier info non traitÃ©s)
# On rÃ©utilise ici l'approche de singleton pour tout ID qui n'aurait pas Ã©tÃ© traitÃ©.
# (Bien que 'unmatched_ids' ne devrait pas Ãªtre vide si la Cellule 11 a traitÃ© tous les fragments)
try:
    info_df = pd.read_csv(INFO_FILE_PATH) 
    all_submission_ids = info_df[FRAGMENT_COL_SUBMISSION].unique().tolist()
    
    # Identifier les IDs qui sont dans le fichier de soumission mais pas dans le clustering
    unprocessed_ids = [id for id in all_submission_ids if id not in assembly_groups]
    
    for id_ in unprocessed_ids:
        final_submission_data.append({
            FRAGMENT_COL_SUBMISSION: id_,
            'Assembly Group': f"Singleton_{id_}" 
        })
        
    print(f"IDs ajoutÃ©s comme singletons : {len(unprocessed_ids)}")

except FileNotFoundError:
    print("âš ï¸� AVERTISSEMENT: Fichier info manquant. Le fichier de soumission ne contiendra que les fragments clusterisÃ©s.")


submission_df = pd.DataFrame(final_submission_data)

# Sauvegarder le fichier de soumission
FINAL_SUBMISSION_PATH = 'submission_groups_mvcnn_final.csv' # ğŸš¨ RenommÃ©
submission_df.to_csv(FINAL_SUBMISSION_PATH, index=False)

print("\nâœ… CrÃ©ation du fichier de soumission FINAL terminÃ©e.")
print(f"Nombre de groupes uniques crÃ©Ã©s : {len(set(submission_df['Assembly Group']))}")
print(f"Fichier sauvegardÃ© sous : {FINAL_SUBMISSION_PATH}")
print("\nExemple de soumission :")
print(submission_df.head(10).to_string(index=False))

