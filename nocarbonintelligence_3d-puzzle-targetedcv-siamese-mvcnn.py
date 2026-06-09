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
from typing import Dict, List, Tuple, Callable
import itertools
import random
import numpy as np
from sklearn.model_selection import train_test_split
import math
from tensorflow.keras.utils import Sequence
import tensorflow as tf
# Imports nÃ©cessaires pour le traitement d'image 2D (OpenCV)
import cv2

# ======================================================================
# 0. DÃ‰FINITION DES CHEMINS & HYPERPARAMÃˆTRES SIAMESE 2D
# ======================================================================

# REMPLACER CETTE VALEUR PAR LE VRAI CHEMIN RACINE DE VOS DONNÃ‰ES.
# NOTE: Le chemin fourni ici est celui de votre environnement Kaggle/Notebook.
BASE_PATH = '/kaggle/input/h690/h690/h690/'
IMAGE_DIR = os.path.join(BASE_PATH, 'sherd_images')
INFO_FILE_PATH = os.path.join(BASE_PATH, 'jd_sherds_info.csv')

# --- CHEMINS DE MISE EN CACHE ---
OUTPUT_DIR = '/kaggle/working/processed_data' # Dossier de sortie (doit exister ou Ãªtre crÃ©Ã©)
CACHE_FILE = os.path.join(OUTPUT_DIR, 'all_image_masks_cache.npz')
# ----------------------------------------

# --- CNN 2D HYPERPARAMÃˆTRES ---
IMG_HEIGHT = 224    # Taille standard des images pour les CNN 2D
IMG_WIDTH = 224
IMG_CHANNELS = 1    # 1 pour le masque (noir et blanc)

# DÃ©finition des noms de colonnes :
FRAGMENT_COL = 'sherd_id'
UNIT_COL = 'unit'
TYPE_COL = 'type'

print(f"BASE_PATH : {BASE_PATH}")
print(f"CACHE_FILE: {CACHE_FILE}")

# ======================================================================
# D. DÃ‰FINITION DE LA CLASSE DMLDataGenerator (AdaptÃ© Ã  2D Siamese)
# ======================================================================

# RÃ©cupÃ©rer le type de donnÃ©es actuel du backend
DTYPE_FLOAT = tf.keras.backend.floatx()
DTYPE_INT = np.uint8 # Pour les images (0-255)

class DMLDataGenerator(Sequence):
    """
    GÃ©nÃ©rateur de donnÃ©es Keras pour le rÃ©seau Siamese 2D.
    Charge les masques 2D (contours) et les met en forme (H, W, C).
    Utilise le format Pairwise ([image_A, image_B], label)
    """

    def __init__(self, dml_pairs: List[Tuple[str, str, int]], all_image_masks: Dict[str, np.ndarray], batch_size: int = 32, shuffle: bool = True):
        self.dml_pairs = dml_pairs
        self.all_image_masks = all_image_masks # Stocke les masques 2D (H, W)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # DÃ©termination de la forme d'entrÃ©e (H, W, C)
        if all_image_masks:
            # RÃ©cupÃ¨re la forme du premier masque dans le dictionnaire
            mask_h, mask_w = next(iter(all_image_masks.values())).shape
            self.input_shape = (mask_h, mask_w, IMG_CHANNELS)
        else:
            self.input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            
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

        # X_A et X_B sont des TENSEURS 4D (BATCH, H, W, C)
        X_A = np.empty((len(batch_pairs), *self.input_shape), dtype=DTYPE_FLOAT)
        X_B = np.empty((len(batch_pairs), *self.input_shape), dtype=DTYPE_FLOAT)
        # Label pour la perte binaire
        Y = np.empty((len(batch_pairs), 1), dtype=DTYPE_FLOAT) 

        for i, (id_A, id_B, label) in enumerate(batch_pairs):
            
            # RÃ©cupÃ©ration du masque 2D (H, W)
            mask_A = self.all_image_masks.get(id_A, np.zeros(self.input_shape[:2], dtype=DTYPE_INT))
            mask_B = self.all_image_masks.get(id_B, np.zeros(self.input_shape[:2], dtype=DTYPE_INT))

            # 1. Mise Ã  l'Ã©chelle et conversion en float (0.0 Ã  1.0)
            img_A = mask_A.astype(DTYPE_FLOAT) / 255.0
            img_B = mask_B.astype(DTYPE_FLOAT) / 255.0
            
            # 2. Ajout du canal (H, W) -> (H, W, 1)
            X_A[i,] = np.expand_dims(img_A, axis=-1)
            X_B[i,] = np.expand_dims(img_B, axis=-1)
            
            # Utilisation du label pour la Perte Binaire (MLP en sortie)
            Y[i,] = label

        # Sortie pour un entraÃ®nement Pairwise (deux entrÃ©es, une cible)
        return ([X_A, X_B], Y)

    def on_epoch_end(self):
        """ MÃ©langer les indices aprÃ¨s chaque Ã©poque si `shuffle` est vrai """
        self.indexes = np.arange(len(self.dml_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @property
    def output_signature(self):
        """ DÃ©finition de la signature de sortie pour tf.data.Dataset.from_generator """
        input_spec = [
            tf.TensorSpec(shape=(None, *self.input_shape), dtype=DTYPE_FLOAT),
            tf.TensorSpec(shape=(None, *self.input_shape), dtype=DTYPE_FLOAT)
        ]
        target_spec = tf.TensorSpec(shape=(None, 1), dtype=DTYPE_FLOAT)
        
        return (input_spec, target_spec)


# ======================================================================
# A. DÃ‰FINITION DE LA FONCTION 1 : Traitement de Tous les Fragments (2D Contour)
# ======================================================================

def process_all_fragments_by_id(image_dir: str, contour_extractor: Callable) -> Dict[str, np.ndarray]:
    """
    Charge les images, extrait les contours 2D (masques) et les redimensionne.
    Retourne un dictionnaire (ID_Vue -> Masque 2D redimensionnÃ©).
    """
    
    all_image_masks = {}
    
    # Trouver toutes les vues (exterior et interior)
    image_paths = glob.glob(os.path.join(image_dir, '*_exterior.jpg'))
    image_paths.extend(glob.glob(os.path.join(image_dir, '*_interior.jpg')))
    
    print(f"Tentative de traitement de {len(image_paths)} images...")

    for i, path in enumerate(image_paths):
        full_id = os.path.basename(path).replace('.jpg', '')
        
        # La fonction contour_extractor est 'extract_final_fragment_contour'
        _, mask = contour_extractor(path, show_plot=False)
        
        if mask is not None:
            # Redimensionnement du masque Ã  la taille standard CNN (224x224)
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
# B. DÃ‰FINITION DE LA FONCTION 2 : CrÃ©ation des Paires DML (FILTRAGE CLÃ‰)
# ======================================================================

def create_dml_pairs(all_image_masks: Dict[str, np.ndarray], assembly_map_full: Dict[str, str]) -> List[Tuple[str, str, int]]:
    """
    CrÃ©e les paires positives (Label=1) et nÃ©gatives (Label=0), 
    en ne conservant QUE les paires basÃ©es sur les vues EXTERIEURES.
    """
    
    # Filtrer les clÃ©s pour ne garder que les vues EXTERNES qui ont un masque ET une carte d'assemblage
    all_external_fragments = [
        k for k in all_image_masks.keys() 
        if k.endswith('_exterior') and k in assembly_map_full
    ]

    # Reconstruire une carte d'assemblage limitÃ©e aux vues externes uniquement
    assembly_map_external = {k: assembly_map_full[k] for k in all_external_fragments}

    # 1. Paires Positives (Label=1)
    positive_pairs = []
    groups = {}
    for full_id, group in assembly_map_external.items():
        if group not in groups: groups[group] = []
        groups[group].append(full_id)
        
    for sherd_list in groups.values():
        if len(sherd_list) >= 2:
            # CrÃ©er des paires POSITIVES (toujours en utilisant les IDs EXTERNES)
            for sherd_A, sherd_B in itertools.combinations(sherd_list, 2):
                positive_pairs.append(tuple(sorted((sherd_A, sherd_B))) + (1,))
                
    df_positive_pairs = pd.DataFrame(positive_pairs, columns=['sherd_A', 'sherd_B', 'label']).drop_duplicates()
    n_positives = len(df_positive_pairs)
    positive_pairs = list(df_positive_pairs.itertuples(index=False, name=None))
    print(f"  - {n_positives} Paires POSITIVES gÃ©nÃ©rÃ©es (uniquement EXTERNES).")

    # 2. Paires NÃ©gatives (Label=0)
    N_NEG_TARGET = max(100, n_positives * 2) # Cible 2x plus de nÃ©gatifs que de positifs
    all_labeled_external_fragments = list(assembly_map_external.keys())
    
    num_attempts = int(N_NEG_TARGET * 3)
    if len(all_labeled_external_fragments) < 2:
        print("ATTENTION: Pas assez de fragments externes labellisÃ©s pour gÃ©nÃ©rer des nÃ©gatifs.")
        return positive_pairs
        
    # Choisir alÃ©atoirement parmi la liste des fragments EXTERNES
    random_ids_A = np.random.choice(all_labeled_external_fragments, size=num_attempts, replace=True)
    random_ids_B = np.random.choice(all_labeled_external_fragments, size=num_attempts, replace=True)

    negative_pairs = []
    # CrÃ©er un ensemble des paires positives vues pour Ã©viter la contamination
    seen_pairs = set(df_positive_pairs[['sherd_A', 'sherd_B']].apply(tuple, axis=1).tolist())

    for id_A, id_B in zip(random_ids_A, random_ids_B):
        if id_A == id_B: continue
        
        ordered_pair = tuple(sorted((id_A, id_B)))
        
        group_A = assembly_map_external.get(id_A)
        group_B = assembly_map_external.get(id_B)

        # La paire est nÃ©gative si elle n'est pas positive, que les fragments sont dans des groupes diffÃ©rents, et non dÃ©jÃ  vue
        if group_A and group_B and group_A != group_B and ordered_pair not in seen_pairs:
            seen_pairs.add(ordered_pair)
            negative_pairs.append(ordered_pair + (0,))

            if len(negative_pairs) >= N_NEG_TARGET:
                break
    
    negative_pairs = negative_pairs[:N_NEG_TARGET]
    n_negatives = len(negative_pairs)
    print(f"  - {n_negatives} Paires NÃ‰GATIVES gÃ©nÃ©rÃ©es (uniquement EXTERNES).")

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
    # Utiliser raise au lieu de exit() dans un environnement notebook
    raise FileNotFoundError(f"Le fichier {INFO_FILE_PATH} est introuvable.")

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

# DÃ‰FINITION DE LA FONCTION D'EXTRACTION DE CONTOUR
# Cette Ã©tape est cruciale : la fonction doit exister.
try:
    # On suppose que 'extract_final_fragment_contour' existe dans le scope.
    contour_extractor_func = extract_final_fragment_contour
except NameError:
    print("\n-------------------------------------------------------------")
    print("â�Œ ERREUR CRITIQUE: La fonction 'extract_final_fragment_contour' n'est pas dÃ©finie.")
    print("Veuillez exÃ©cuter la cellule qui dÃ©finit cette fonction (avec cv2) avant celle-ci.")
    print("-------------------------------------------------------------")
    # Utiliser raise au lieu de exit() dans un environnement notebook
    raise NameError("La fonction 'extract_final_fragment_contour' doit Ãªtre dÃ©finie.")


# 3. EXÃ‰CUTION DU PIPELINE DML AVEC GESTION DU CACHE
print("\n--- DÃ©marrage de la gÃ©nÃ©ration des masques 2D (Pipeline Image -> Contour) ---")

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_image_masks = {}

if os.path.exists(CACHE_FILE):
    print(f"âœ… Masques trouvÃ©s dans le cache ! Chargement depuis {CACHE_FILE}...")
    try:
        # allow_pickle=True est souvent nÃ©cessaire pour les dictionnaires
        with np.load(CACHE_FILE, allow_pickle=True) as data:
            all_image_masks = data['masks'].item()
        print(f"    {len(all_image_masks)} masques chargÃ©s depuis le cache.")
    except Exception as e:
        print(f"âš ï¸� ERREUR lors du chargement du cache : {e}. Reprise du traitement.")

if not all_image_masks or len(all_image_masks) == 0:
    print("ğŸ”„ Le cache est vide ou invalide. Reprise du traitement complet des images...")
    
    all_image_masks = process_all_fragments_by_id(IMAGE_DIR, contour_extractor_func)

    if all_image_masks:
        try:
            # Sauvegarde dans le cache
            np.savez(CACHE_FILE, masks=all_image_masks)
            print(f"âœ… {len(all_image_masks)} masques sauvegardÃ©s dans le cache : {CACHE_FILE}")
        except Exception as e:
            print(f"âš ï¸� AVERTISSEMENT : Ã‰chec de la sauvegarde du cache. {e}")


# Diagnostic de l'intersection
processed_external_ids = {k for k in all_image_masks.keys() if k.endswith('_exterior')}
mapped_external_ids = {k for k in ASSEMBLY_MAP_FULL.keys() if k.endswith('_exterior')}
overlap_ids = processed_external_ids.intersection(mapped_external_ids)
print(f"\n--- DIAGNOSTIC D'INTERSECTION ---")
print(f"Fragments **EXTERNES** avec VÃ©ritÃ© Terrain correspondante (overlap) : {len(overlap_ids)}")

if len(overlap_ids) < 2:
    print("â�Œ Ã‰CHEC : Moins de 2 fragments en commun. Le set d'entraÃ®nement est insuffisant.")
else:
    print("\n--- CrÃ©ation des paires DML **EXTERNES** complÃ¨tes ---")
    
    dml_all_pairs = create_dml_pairs(all_image_masks, ASSEMBLY_MAP_FULL)
    
    print(f"\nTotal des paires DML **EXTERNES** gÃ©nÃ©rÃ©es : {len(dml_all_pairs)}")

    # 4. DIVISION TRAIN / VALIDATION / TEST (avec stratification)
    
    # SÃ©paration initiale : Train/Val vs. Test (80% / 20%)
    train_val_pairs, dml_test_pairs = train_test_split(
        dml_all_pairs,
        test_size=0.2,
        random_state=42,
        # Stratification basÃ©e sur le label (0 ou 1)
        stratify=[p[2] for p in dml_all_pairs]
    )
    
    # SÃ©paration secondaire : Train vs. Validation (75% / 25% de train_val = 20% du total)
    dml_train_pairs, dml_val_pairs = train_test_split(
        train_val_pairs,
        test_size=0.25, # 25% de 80% = 20% du total
        random_state=42,
        # Stratification basÃ©e sur le label (0 ou 1)
        stratify=[p[2] for p in train_val_pairs]
    )
    
    # AFFICHAGE DES RÃ‰SULTATS
    print("\n--- RÃ‰SULTATS DE LA DIVISION DES PAIRES (Train/Val/Test) ---")
    print(f"âœ… dml_train_pairs (60% pour l'entraÃ®nement Siamese EXTERNE) : {len(dml_train_pairs)} paires")
    print(f"âœ… dml_val_pairs (20% pour la validation) : {len(dml_val_pairs)} paires")
    print(f"âœ… dml_test_pairs (20% pour l'Ã©valuation) : {len(dml_test_pairs)} paires")


# ----------------------------------------------------------------------
# --- MVCNN SIAMESE TRAINING CELL (COMPLET) ---
# Ce script inclut dÃ©sormais la dÃ©finition du modÃ¨le Siamois (siamese_model)
# et l'architecture de l'encodeur MVCNN pour corriger l'erreur 'is not defined'.
# ----------------------------------------------------------------------

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, concatenate, Reshape, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K

# Note: DMLDataGenerator, dml_train_pairs, dml_val_pairs, dml_test_pairs, et all_image_masks
# DOIVENT Ãªtre dÃ©finis dans les cellules prÃ©cÃ©dentes pour que ceci fonctionne.

# ======================================================================
# 1. DÃ‰FINITION DES HYPERPARAMÃˆTRES ET CONSTANTES
# ======================================================================
# Assurez-vous que l'input_shape correspond Ã  la taille des masques 2D (ex: 224x224x3)
INPUT_SHAPE = (224, 224, 3) 
EMBEDDING_DIM = 256 # Taille de l'espace d'embedding pour le MVCNN
MARGIN = 1.0 # Marge pour la perte contrastive (Loss)

EPOCHS = 5
BATCH_SIZE = 32
# Le chemin de sauvegarde doit se terminer par '.weights.h5'
CHECKPOINT_PATH = "best_siamese_mvcnn_weights.weights.h5"
ENCODER_LAYER_NAME = 'MVCNN_Encoder_SingleView' # Nom de la couche Ã  extraire pour la prÃ©diction
ENCODER_PATH = 'mvcnn_encoder_final.h5'


# ======================================================================
# 2. DÃ‰FINITION DES ARCHITECTURES CLÃ‰S (MVCNN et SIAMESE)
# ======================================================================

def contrastive_loss(y_true, y_pred):
    """
    Perte Contrastive : Calcule la perte pour un rÃ©seau Siamois.
    y_true (Label): 1 pour paires similaires, 0 pour paires diffÃ©rentes.
    y_pred (Distance): Distance euclidienne L2 entre les embeddings.
    """
    # Perte pour les paires similaires (y_true=1) : minimiser la distance
    loss_s = y_true * K.square(y_pred)
    
    # Perte pour les paires diffÃ©rentes (y_true=0) : maximiser la distance au-delÃ  de la marge
    loss_d = (1 - y_true) * K.square(K.maximum(MARGIN - y_pred, 0))
    
    return K.mean(loss_s + loss_d)

def create_mvcnn_encoder(input_shape, embedding_dim):
    """
    CrÃ©e l'encodeur MVCNN Ã  poids partagÃ©s, basÃ© sur VGG16 (une seule vue).
    """
    # 1. RÃ©seau de base (VGG16)
    base_model = VGG16(
        weights='imagenet', # Poids prÃ©-entraÃ®nÃ©s
        include_top=False, 
        input_shape=input_shape
    )
    
    # 2. Couches d'adaptation post-VGG
    x = base_model.output
    x = Flatten(name='flatten_features')(x)
    x = Dense(1024, activation='relu')(x)
    
    # 3. Couche d'Embedding (sortie finale)
    embedding = Dense(embedding_dim, activation=None, name='embedding_output')(x)
    
    # 4. Normalisation L2
    embedding = Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm_embedding')(embedding)
    
    # Le modÃ¨le encodeur
    encoder = Model(inputs=base_model.input, outputs=embedding, name=ENCODER_LAYER_NAME)
    
    return encoder

def create_siamese_model(encoder, input_shape):
    """
    Assemble le modÃ¨le Siamois complet autour de l'encodeur partagÃ©.
    """
    # 1. DÃ©finir les deux entrÃ©es (Anchor et Positive/Negative)
    input_anchor = Input(shape=input_shape, name='anchor_input')
    input_other = Input(shape=input_shape, name='other_input')
    
    # 2. Obtenir les embeddings via l'encodeur partagÃ©
    embedding_anchor = encoder(input_anchor)
    embedding_other = encoder(input_other)
    
    # 3. Calculer la distance (Euclidienne L2)
    L2_distance = Lambda(
        lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True)),
        name='distance_layer'
    )([embedding_anchor, embedding_other])
    
    # 4. CrÃ©er le modÃ¨le Siamois
    siamese_model = Model(
        inputs=[input_anchor, input_other], 
        outputs=L2_distance, 
        name='Siamese_MVCNN_Model'
    )
    
    # 5. Compiler le modÃ¨le
    siamese_model.compile(
        loss=contrastive_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    
    return siamese_model

# ======================================================================
# 3. INSTANCIATION DU MODÃˆLE SIAMOIS (Correction de l'erreur 'is not defined')
# ======================================================================
print("\n--- DÃ©finition et Compilation du ModÃ¨le Siamois MVCNN ---")

# 3.1 CrÃ©ation de l'encodeur de vue unique
single_view_encoder = create_mvcnn_encoder(INPUT_SHAPE, EMBEDDING_DIM)

# 3.2 CrÃ©ation du modÃ¨le Siamois (cette ligne dÃ©finit 'siamese_model')
siamese_model = create_siamese_model(single_view_encoder, INPUT_SHAPE)

print(f"âœ… ModÃ¨le siamois crÃ©Ã©. Dimension d'Embedding: {EMBEDDING_DIM}")
print("AperÃ§u de l'architecture :")
siamese_model.summary(line_length=150)


# ======================================================================
# 4. CRÃ‰ATION DES GÃ‰NÃ‰RATEURS DE DONNÃ‰ES (MVCNN)
# ======================================================================

print("\n--- PrÃ©paration des gÃ©nÃ©rateurs de donnÃ©es MVCNN ---")

try:
    # A. CrÃ©ation des gÃ©nÃ©rateurs Keras Sequence pour l'entraÃ®nement (shuffle: True)
    train_generator = DMLDataGenerator(
        dml_pairs=dml_train_pairs,
        all_image_masks=all_image_masks, # Utilisation des masques 2D
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # B. CrÃ©ation des gÃ©nÃ©rateurs Keras Sequence pour la validation (shuffle: False)
    val_generator = DMLDataGenerator(
        dml_pairs=dml_val_pairs,
        all_image_masks=all_image_masks, # Utilisation des masques 2D
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
except NameError as e:
    print("\n-------------------------------------------------------------")
    print(f"â�Œ Erreur critique : {e}. Assurez-vous que les dÃ©pendances (DMLDataGenerator, paires, masques) sont dÃ©finies.")
    print("-------------------------------------------------------------")
    exit()

print(f"âœ… GÃ©nÃ©rateur d'entraÃ®nement crÃ©Ã© : {len(train_generator)} lots.")
print(f"âœ… GÃ©nÃ©rateur de validation crÃ©Ã© : {len(val_generator)} lots.")


# ======================================================================
# 5. DÃ‰FINITION DES CALLBACKS (Gestion de l'entraÃ®nement et de l'optimisation)
# ======================================================================
callbacks_list = [
    # Sauvegarde du modÃ¨le avec la meilleure 'val_loss'
    ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=True, # Sauvegarde lÃ©gÃ¨re (uniquement les poids)
        verbose=1
    ),
    # ArrÃªt prÃ©coce pour Ã©viter le surapprentissage
    EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min',
        restore_best_weights=True,
        verbose=1
    ),
    # Stoppe l'entraÃ®nement en cas de valeurs numÃ©riques instables
    TerminateOnNaN(),
    # RÃ©duit le taux d'apprentissage si la perte de validation stagne
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-08,
        verbose=1
    )
]


# ======================================================================
# 6. LANCEMENT DE L'ENTRAÃ�NEMENT (Utilisation directe du Keras Sequence)
# ======================================================================
print("\n--- DÃ©marrage de l'entraÃ®nement du ModÃ¨le Siamois MVCNN ---")

try:
    # L'entraÃ®nement utilise la Keras Sequence personnalisÃ©e DMLDataGenerator
    history = siamese_model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks_list,
        # L'argument 'workers' est correctement omis pour Ã©viter les problÃ¨mes de multiprocessing
    )
    
except Exception as e:
    print("\n-------------------------------------------------------------")
    print(f"â�Œ ERREUR LORS DE L'ENTRAÃ�NEMENT: {e}")
    print("-------------------------------------------------------------")


# ======================================================================
# 7. Ã‰VALUATION FINALE ET SAUVEGARDE DE L'ENCODEUR
# ======================================================================

# 7.1 Chargement des meilleurs poids
if os.path.exists(CHECKPOINT_PATH):
    siamese_model.load_weights(CHECKPOINT_PATH)
    print(f"\nâœ… Meilleurs poids chargÃ©s depuis {CHECKPOINT_PATH}.")
else:
    print("\nâš ï¸� ATTENTION: Les poids optimaux n'ont pas pu Ãªtre chargÃ©s.")


# 7.2 Ã‰valuation finale
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


# 7.3 Sauvegarde de l'Encodeur seul
try:
    # On extrait la sous-couche responsable de l'embedding (nommÃ©e dans la fonction create_mvcnn_encoder)
    mvcnn_encoder = siamese_model.get_layer(ENCODER_LAYER_NAME)
    mvcnn_encoder.save(ENCODER_PATH)
    print(f"\nâœ… ModÃ¨le d'Embedding ({ENCODER_LAYER_NAME}) sauvegardÃ© pour la prÃ©diction sous : {ENCODER_PATH}")
except Exception as e:
    print(f"\nâš ï¸� ATTENTION: Ã‰chec de la sauvegarde de l'encodeur MVCNN. Raison : {e}")


import h5py
import os
import sys

# ======================================================================
# 0. CONFIGURATION DES CHEMINS
# ======================================================================

# Nous utilisons le chemin local du fichier qui serait sauvegardÃ© par la cellule d'entraÃ®nement prÃ©cÃ©dente.
# Remplacez-le par le chemin d'accÃ¨s au jeu de donnÃ©es si vous chargez depuis un emplacement externe.
SIAMESE_WEIGHTS_PATH = "best_siamese_mvcnn_weights.weights.h5"

print("\n--- DÃ©marrage de l'inspection du fichier de poids ---")
print(f"Tentative d'inspection du chemin : {SIAMESE_WEIGHTS_PATH}")

if os.path.exists(SIAMESE_WEIGHTS_PATH):
    
    # Ouvre le fichier en mode lecture
    try:
        with h5py.File(SIAMESE_WEIGHTS_PATH, 'r') as f:
            print(f"\nâœ… Fichier HDF5 ouvert avec succÃ¨s : {SIAMESE_WEIGHTS_PATH}")
            
            # Afficher les clÃ©s de niveau supÃ©rieur (groupes et datasets)
            top_level_keys = list(f.keys())
            print(f"\nGroupes de niveau supÃ©rieur dans le fichier: {top_level_keys}")
            
            print("\n--- Exploration des couches (Top Level) ---")
            
            def print_attrs(name, obj):
                """Fonction utilitaire pour afficher les attributs d'un objet HDF5."""
                sys.stdout.write(f"\nGroupe: {name}")
                if isinstance(obj, h5py.Group):
                    if 'layer_names' in obj.attrs:
                        # Ceci est souvent prÃ©sent dans les groupes de modÃ¨les Keras
                        sys.stdout.write(f" (Contient {len(obj.attrs['layer_names'])} couches Keras)")
                sys.stdout.write(f"\n  Sous-clÃ©s: {list(obj.keys())}")

            f.visititems(print_attrs)
            
            print("\n")
            
    except Exception as e:
        print(f"â�Œ ERREUR lors de l'ouverture du fichier de poids avec h5py : {e}")
        
else:
    print(f"\nâ�Œ ERREUR : Le fichier de poids n'a pas Ã©tÃ© trouvÃ© au chemin spÃ©cifiÃ© : {SIAMESE_WEIGHTS_PATH}")
    print("Veuillez vous assurer que la cellule d'entraÃ®nement a Ã©tÃ© exÃ©cutÃ©e et que le fichier a Ã©tÃ© sauvegardÃ©, ou que le chemin d'accÃ¨s aux donnÃ©es externes est correct.")

print("\n--- Fin de l'inspection ---")


import pandas as pd
import numpy as np
import os
import sys
import glob
import h5py
from typing import List, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from tensorflow.keras.models import Model
# Assurez-vous d'avoir les bonnes importations Keras pour votre modÃ¨le (e.g., VGG16, ResNet, etc.)
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ======================================================================
# 0. CONFIGURATION DES CHEMINS ET HYPERPARAMÃˆTRES
# ======================================================================

# ğŸš¨ CRITIQUE: VÃ©rifiez ce chemin d'accÃ¨s aux poids.
SIAMESE_WEIGHTS_PATH = "best_siamese_mvcnn_weights.weights.h5"
# ğŸš¨ CRITIQUE: VÃ©rifiez le chemin d'accÃ¨s aux images d'entrÃ©e (e.g., /kaggle/input/data/images)
IMAGE_DIR_PATH = '/kaggle/input/h690/h690/h690/sherd_images' 
FINAL_SUBMISSION_PATH = 'submission.csv' 

# Taille d'entrÃ©e pour le modÃ¨le MVCNN Siamese (doit correspondre Ã  l'entraÃ®nement)
INPUT_SHAPE = (224, 224, 3) 
# Le nombre de dimensions de sortie de la couche d'embedding de votre modÃ¨le Siamese
FEATURE_DIMENSION = 512 

# ParamÃ¨tre de Clustering HiÃ©rarchique
# Seuil de coupure pour le clustering (ajustez cette valeur: 0.1 Ã  1.0)
CLUSTERING_THRESHOLD = 0.5 

# Noms de colonnes requis
GROUP_COL_REQ = 'Assembly Group'
IMAGE_ID_COL_REQ = 'image_id'

print(f"--- DÃ©marrage du Pipeline Complet (Seuil: {CLUSTERING_THRESHOLD}) ---")

# ======================================================================
# 1. DÃ‰FINITION ET CHARGEMENT DU MODÃˆLE EXTRACTEUR
# ======================================================================

def create_feature_extractor(input_shape: tuple) -> Model:
    """
    RecrÃ©e le bras extracteur de caractÃ©ristiques (base du Siamese Network)
    pour charger les poids.
    
    ğŸš¨ ATTENTION: REMPLACER PAR VOTRE ARCHITECTURE MVCNN RÃ‰ELLE
    Assurez-vous que cette architecture est un calque exact du bras
    extracteur de votre modÃ¨le Siamese entraÃ®nÃ©.
    """
    print("DÃ©finition de l'architecture de l'extracteur...")
    
    # 1. Base Model (Exemple: VGG16 sans les couches supÃ©rieures)
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    
    # 2. Couche(s) de regroupement et d'embedding (doit correspondre Ã  votre entraÃ®nement)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    # Assurez-vous que le nom 'embedding_layer' correspond Ã  celui utilisÃ© dans votre modÃ¨le Siamese
    feature_vector = Dense(FEATURE_DIMENSION, activation='relu', name='embedding_layer')(x) 
    
    model = Model(inputs=base_model.input, outputs=feature_vector, name='feature_extractor')
    return model

def load_and_configure_extractor(weights_path: str, input_shape: tuple) -> Model:
    """Charge le modÃ¨le et ses poids entraÃ®nÃ©s."""
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Fichier de poids non trouvÃ© : {weights_path}. Veuillez l'entraÃ®ner ou vÃ©rifier le chemin.")
        
    extractor = create_feature_extractor(input_shape)
    
    try:
        # Tente de charger les poids
        extractor.load_weights(weights_path)
        print(f"âœ… Poids chargÃ©s avec succÃ¨s depuis : {weights_path}")
        
        # Test de cohÃ©rence (optionnel, mais utile)
        dummy_input = np.zeros((1, *input_shape))
        _ = extractor.predict(dummy_input)
        
        return extractor

    except Exception as e:
        print(f"â�Œ ERREUR lors du chargement des poids : {e}")
        print("VÃ©rifiez que l'architecture 'create_feature_extractor' est EXACTE par rapport Ã  l'entraÃ®nement.")
        sys.exit(1) # ArrÃªte l'exÃ©cution si le chargement Ã©choue


# ======================================================================
# 2. EXTRACTION DES CARACTÃ‰RISTIQUES (LOGIQUE RÃ‰ELLE)
# ======================================================================

def real_feature_extraction(extractor: Model, image_dir: str) -> Tuple[List[str], np.ndarray]:
    """
    ImplÃ©mentation rÃ©elle de l'extraction de vecteurs.
    
    ğŸš¨ ATTENTION: REMPLACER CECI PAR VOTRE LOGIQUE DE DATAGENERATOR RÃ‰ELLE
    pour gÃ©rer le prÃ©-traitement (resize, normalisation) de Keras.
    """
    print("\n--- 2. Extraction des CaractÃ©ristiques RÃ©elles ---")
    
    all_image_paths = glob.glob(os.path.join(image_dir, '*_exterior.jpg'))
    all_image_paths.extend(glob.glob(os.path.join(image_dir, '*_interior.jpg')))
    
    if not all_image_paths:
        raise FileNotFoundError(f"Aucune image trouvÃ©e dans {image_dir}. VÃ©rifiez le chemin.")
        
    all_image_paths.sort() # Tri pour s'assurer que l'ordre des IDs est le mÃªme que celui des vecteurs
    
    cluster_ids = [os.path.basename(p).replace('.jpg', '') for p in all_image_paths]
    
    print(f"Fragments trouvÃ©s : {len(cluster_ids)}. DÃ©marrage de la prÃ©diction...")
    
    # PrÃ©traitement et PrÃ©diction (Version simplifiÃ©e SANS Keras Data Generator)
    
    images = []
    # Boucler sur les chemins (dans une vraie application, utiliser un Data Generator pour l'efficacitÃ©)
    for i, path in enumerate(all_image_paths):
        # Charge l'image et la redimensionne Ã  la taille d'entrÃ©e du modÃ¨le
        img = load_img(path, target_size=INPUT_SHAPE[:2])
        img_array = img_to_array(img)
        
        # ğŸš¨ IMPORTANT: Appliquez ici la mÃªme normalisation/prÃ©traitement que pendant l'ENTRAÃ�NEMENT
        # (e.g., /255.0, ou la fonction preprocess_input de votre base_model)
        
        # Pour VGG16, il faut gÃ©nÃ©ralement appeler tensorflow.keras.applications.vgg16.preprocess_input
        # Ici, on simule une simple division par 255.0 si votre entraÃ®nement utilisait cela :
        img_array /= 255.0 
        
        images.append(img_array)
        
        # Affichage de progression
        if (i + 1) % 100 == 0 or (i + 1) == len(all_image_paths):
            sys.stdout.write(f"\rTraitement : {i+1}/{len(all_image_paths)} images...")
    
    sys.stdout.write("\n")
    
    # Empilement des images pour la prÃ©diction en lot
    X_test = np.array(images)
    
    # PrÃ©diction
    feature_vectors = extractor.predict(X_test, batch_size=32, verbose=1)
    
    print(f"âœ… Extraction terminÃ©e. Obtenu {feature_vectors.shape[0]} vecteurs de dimension {feature_vectors.shape[1]}.")
    
    return cluster_ids, feature_vectors


# ======================================================================
# 3. CLUSTERING HIÃ‰RARCHIQUE
# ======================================================================

def perform_hierarchical_clustering(feature_vectors: np.ndarray, threshold: float) -> np.ndarray:
    """
    Effectue le clustering hiÃ©rarchique sur les vecteurs de caractÃ©ristiques.
    """
    print(f"\n--- 3. Clustering HiÃ©rarchique (Seuil: {threshold}) ---")
    
    # 3.1. Calcul de la matrice de distance (Distance Cosinus)
    # distance = 1 - similaritÃ© cosinus. MÃ©thode standard pour les embeddings.
    distance_matrix_condensed = pdist(feature_vectors, metric='cosine')
    
    # 3.2. Construction du dendrogramme (algorithme de chaÃ®nage 'complete' - lien max)
    Z = linkage(distance_matrix_condensed, method='complete')
    
    # 3.3. DÃ©coupage du dendrogramme au seuil dÃ©fini
    agg_labels = fcluster(Z, t=threshold, criterion='distance')
    
    num_clusters = len(np.unique(agg_labels))
    print(f"âœ… Clustering terminÃ©. CrÃ©ation de {num_clusters} Assembly Groups.")
    
    # Retourne les labels (e.g., [1, 1, 2, 3, 2, 1, ...])
    return agg_labels

# ======================================================================
# 4. FORMATAGE ET SAUVEGARDE DE LA SOUMISSION
# ======================================================================

def custom_sort_key(image_id: str):
    """
    Trie un ID au format 'JD00001_exterior' : d'abord l'ID de base, 
    puis assure que 'exterior' (0) est triÃ© avant 'interior' (1).
    """
    parts = image_id.rsplit('_', 1)
    base_id = parts[0]
    view = parts[1]
    
    view_priority = 0 if view == 'exterior' else 1
    
    return (base_id, view_priority)


def create_submission_file(cluster_ids: List[str], agg_labels: np.ndarray, all_image_dir: str, final_path: str):
    """
    Formate les IDs et les labels dans le CSV de soumission et applique le tri strict.
    """

    print(f"\n--- 4. Formatage et Sauvegarde de la Soumission ---")
    
    # VÃ©rification des longueurs
    if len(cluster_ids) != len(agg_labels):
        raise ValueError(f"IncohÃ©rence des longueurs : IDs ({len(cluster_ids)}) != Labels ({len(agg_labels)})")

    # Mapping du Cluster NumÃ©rique (1, 2, 3...) vers le Nom Requis (AssemblyGroup1, AssemblyGroup2...)
    unique_numerical_labels = sorted(np.unique(agg_labels))
    label_to_group_name = {
        label: f"AssemblyGroup{i+1}"
        for i, label in enumerate(unique_numerical_labels)
    }

    # CrÃ©er le mappage final ID Fragment -> Nom de Groupe formatÃ©
    cluster_map = {}
    for i, fragment_id in enumerate(cluster_ids):
        numerical_label = agg_labels[i]
        final_group_name = label_to_group_name.get(numerical_label, "Singleton_Error")
        cluster_map[fragment_id] = final_group_name

    print(f"Ã‰tape 4.1: CrÃ©ation de {len(cluster_map)} mappings de cluster.")

    # Collecte de TOUS les fragments d'images
    all_image_paths = glob.glob(os.path.join(all_image_dir, '*_exterior.jpg'))
    all_image_paths.extend(glob.glob(os.path.join(all_image_dir, '*_interior.jpg')))
    all_submission_ids = [os.path.basename(p).replace('.jpg', '') for p in all_image_paths]
    
    # ğŸš¨ APPLICATION DU TRI EXPLICITE POUR L'ORDRE DES LIGNES
    all_submission_ids.sort(key=custom_sort_key)
    
    print(f"TrouvÃ© {len(all_submission_ids)} IDs d'images, triÃ©s.")

    # Construction du DataFrame final
    submission_data = []
    UNPROCESSED_LABEL = "AssemblyGroupSingleton"

    for image_id in all_submission_ids:
        # RÃ©cupÃ©rer le label. Si l'ID n'est pas dans le cluster_ids (ce qui ne devrait pas arriver ici),
        # il sera Ã©tiquetÃ© comme Singleton.
        final_group_label = cluster_map.get(image_id, UNPROCESSED_LABEL)
        submission_data.append({
            GROUP_COL_REQ: final_group_label,
            IMAGE_ID_COL_REQ: image_id
        })

    df_submission = pd.DataFrame(submission_data)

    # Assurer l'ordre final des colonnes : Assembly Group, image_id
    df_submission = df_submission[[GROUP_COL_REQ, IMAGE_ID_COL_REQ]]

    # Sauvegarde
    df_submission.to_csv(final_path, index=False)

    print(f"\nâœ… Fichier de soumission crÃ©Ã© avec succÃ¨s : {final_path}")
    print(f"Total des lignes : {len(df_submission)}")
    print("\nAperÃ§u des premiÃ¨res lignes (VÃ©rifiez le tri JD..._exterior/interior) :")
    print(df_submission.head(10))
    print("\nDistribution des labels (Top 10) :")
    print(df_submission[GROUP_COL_REQ].value_counts().head(10))


# ======================================================================
# EXECUTION DU PIPELINE
# ======================================================================
if __name__ == '__main__':
    try:
        # 1. Charger/Configurer l'extracteur
        extractor = load_and_configure_extractor(SIAMESE_WEIGHTS_PATH, INPUT_SHAPE)
        
        # 2. Extraction des caractÃ©ristiques
        # âš ï¸� REMPLACER 'real_feature_extraction' si votre pipeline de chargement d'image est diffÃ©rent
        cluster_ids, feature_vectors = real_feature_extraction(extractor, IMAGE_DIR_PATH)
        
        # 3. Clustering
        agg_labels = perform_hierarchical_clustering(feature_vectors, CLUSTERING_THRESHOLD)
        
        # 4. Formatage et Sauvegarde de la soumission
        create_submission_file(cluster_ids, agg_labels, IMAGE_DIR_PATH, FINAL_SUBMISSION_PATH)
        
    except Exception as e:
        print(f"\nâ�Œ LE PIPELINE A Ã‰CHOUÃ‰ : {e}")
        # Afficher la trace complÃ¨te de l'erreur pour le dÃ©bogage
        import traceback
        traceback.print_exc(file=sys.stdout)


import pandas as pd
import numpy as np
import os
import sys
import glob 
from typing import List

# ======================================================================
# 0. Configuration des Chemins
# ======================================================================

# Le chemin doit pointer vers le rÃ©pertoire des images d'entrÃ©e (images des fragments .jpg)
IMAGE_DIR_PATH = '/kaggle/input/h690/h690/h690/sherd_images' 
FINAL_SUBMISSION_PATH = 'submission.csv' 

# Noms de colonnes requis par le template (Ordre InversÃ©: Col 1 Assembly Group, Col 2 image_id)
GROUP_COL_REQ = 'Assembly Group'
IMAGE_ID_COL_REQ = 'image_id'

print(f"--- DÃ©marrage de la Soumission (Ordre des Lignes Garanti) ---")

# ======================================================================
# 1. PRÃ‰PARATION DES DONNÃ‰ES DE CLUSTERING 
# ======================================================================

# Fonction pour trier les IDs selon la spÃ©cification : ID de base, puis '_exterior' avant '_interior'
def custom_sort_key(image_id: str):
    """
    Trie un ID au format 'JD00001_exterior' en utilisant d'abord l'ID de base, 
    puis en assurant que 'exterior' est triÃ© avant 'interior'.
    """
    # SÃ©pare l'ID de base du suffixe de vue ('_exterior' ou '_interior')
    parts = image_id.rsplit('_', 1)
    base_id = parts[0]   # e.g., 'JD00001'
    view = parts[1]      # e.g., 'exterior' ou 'interior'
    
    # ğŸš¨ CritÃ¨re de tri explicite : le Base ID puis un indicateur pour la vue
    # 'exterior' aura la prioritÃ© sur 'interior' (0 < 1)
    view_priority = 0 if view == 'exterior' else 1
    
    return (base_id, view_priority)


try:
    if 'cluster_ids' not in locals() or 'agg_labels' not in locals():
        # --- DONNÃ‰ES FACTICES pour la DÃ‰MONSTRATION ---
        print("âš ï¸� Variables de clustering non trouvÃ©es. CrÃ©ation de donnÃ©es factices pour le formatage.")
        np.random.seed(42) 
        
        # Simuler 50 IDs (25 fragments x 2 vues)
        mock_base_ids = [f"JD00{i:03d}" for i in range(1, 26)]  
        mock_external_ids = [f"{i}_exterior" for i in mock_base_ids]
        mock_interior_ids = [f"{i}_interior" for i in mock_base_ids]
        cluster_ids = mock_external_ids + mock_interior_ids # 50 IDs au total
        agg_labels = np.random.randint(0, 5, size=50) # 5 groupes factices (labels de 0 Ã  4)
        # ---------------------------------------------
        
    cluster_ids_list = list(cluster_ids)
    
    if len(cluster_ids_list) != len(agg_labels):
        raise ValueError(f"IncohÃ©rence des longueurs : IDs ({len(cluster_ids_list)}) != Labels ({len(agg_labels)})")

    # Mapping du Cluster NumÃ©rique (0, 1, 2...) vers le Nom Requis (AssemblyGroup1, AssemblyGroup2...)
    unique_numerical_labels = np.unique(agg_labels)
    label_to_group_name = {
        label: f"AssemblyGroup{i+1}" 
        for i, label in enumerate(unique_numerical_labels)
    }

    # CrÃ©er le mappage final ID Fragment -> Nom de Groupe formatÃ©
    cluster_map = {}
    for i, fragment_id in enumerate(cluster_ids_list):
        numerical_label = agg_labels[i]
        final_group_name = label_to_group_name.get(numerical_label, "Singleton_Error")
        cluster_map[fragment_id] = final_group_name
        
    print(f"\nÃ‰tape 1: CrÃ©ation de {len(cluster_map)} mappings de cluster. Labels numÃ©riques mappÃ©s aux groupes : {label_to_group_name}")

    # 2. Collecte et Tri de TOUS les fragments d'images (GARANTIR L'ORDRE)
    print(f"\nÃ‰tape 2: Collecte et tri des IDs d'images pour la soumission...")
    
    all_image_paths = glob.glob(os.path.join(IMAGE_DIR_PATH, '*_exterior.jpg'))
    all_image_paths.extend(glob.glob(os.path.join(IMAGE_DIR_PATH, '*_interior.jpg')))
    
    # Extraire l'ID complet (e.g., 'JD00001_exterior')
    all_submission_ids = [os.path.basename(p).replace('.jpg', '') for p in all_image_paths]
    
    # ğŸš¨ Application du tri personnalisÃ© pour l'ordre exact demandÃ©
    all_submission_ids.sort(key=custom_sort_key)
    
    print(f"TrouvÃ© {len(all_submission_ids)} IDs d'images au total, triÃ©s selon les spÃ©cifications.")
    
    # 3. Construction du DataFrame final (avec l'ordre de colonnes requis)
    submission_data = []
    UNPROCESSED_LABEL = "AssemblyGroupSingleton" 
    
    for image_id in all_submission_ids:
        final_group_label = cluster_map.get(image_id, UNPROCESSED_LABEL)
        submission_data.append({
            GROUP_COL_REQ: final_group_label,  # Colonne 1
            IMAGE_ID_COL_REQ: image_id         # Colonne 2
        })

    df_submission = pd.DataFrame(submission_data)
    
    # Assurer l'ordre des colonnes : Assembly Group, image_id
    df_submission = df_submission[[GROUP_COL_REQ, IMAGE_ID_COL_REQ]]
    
    # 4. Sauvegarde du fichier de soumission
    print(f"Fragments dans le DataFrame de soumission : {len(df_submission)}")
    
    df_submission.to_csv(FINAL_SUBMISSION_PATH, index=False)
    
    print(f"\nâœ… Fichier de soumission crÃ©Ã© avec succÃ¨s : {FINAL_SUBMISSION_PATH}")
    print("AperÃ§u des premiÃ¨res lignes du fichier de soumission (Ordre des colonnes et des lignes vÃ©rifiÃ©) :")
    print(df_submission.head())
    print("\nDistribution des labels (Top 10) :")
    print(df_submission[GROUP_COL_REQ].value_counts().head(10))

except Exception as e:
    print(f"\nâ�Œ ERREUR CRITIQUE DANS LA CRÃ‰ATION DE LA SOUMISSION : {e}")

