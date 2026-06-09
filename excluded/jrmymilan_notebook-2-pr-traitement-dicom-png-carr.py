# =============================================================================
# Section 1 - Imports, configuration et sélection d'un sous-ensemble équilibré
# =============================================================================
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import pydicom

from tqdm.auto import tqdm

# On masque quelques warnings "bruit"
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Répertoires de données (dataset RSNA ajouté via "Add data")
DATA_DIR = Path("/kaggle/input/rsna-breast-cancer-detection")
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"

print("DATA_DIR :", DATA_DIR)
print("train.csv présent ?", TRAIN_CSV.exists())
print("train_images présent ?", TRAIN_IMG_DIR.exists())

# Chargement du CSV principal
train = pd.read_csv(TRAIN_CSV)
print("Taille du DataFrame train :", train.shape)

# --- Sélection d'un sous-ensemble équilibré ---
# Toutes les positives (cancer=1) + même nombre de négatives (cancer=0)

pos_df = train[train["cancer"] == 1].copy()
neg_df = train[train["cancer"] == 0].sample(len(pos_df), random_state=42)

subset_df = pd.concat([pos_df, neg_df], ignore_index=True)
subset_df = subset_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

print("Nombre d'images positives :", len(pos_df))
print("Nombre d'images négatives (échantillon) :", len(neg_df))
print("Taille du sous-ensemble :", len(subset_df))

subset_df["cancer"].value_counts()



# =============================================================================
# Section 2 - Fonctions pour transformer DICOM -> image 512×512 exploitable
# =============================================================================

def load_dicom_raw(patient_id, image_id, verbose=False):
    """
    Lecture "brute" d'un DICOM RSNA.
    - Retourne (img, dcm) ou (None, None) si non décodable.
    - img : array float32 (valeurs originales après rescale).
    """
    dcm_path = TRAIN_IMG_DIR / str(patient_id) / f"{image_id}.dcm"
    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
    except Exception as e:
        if verbose:
            print(f"[WARN] Impossible de lire {dcm_path}: {e}")
        return None, None

    # Application éventuelle de la transformation linéaire (slope / intercept)
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    img = img * slope + intercept

    return img, dcm


def window_image(img, low=5, high=99.5):
    """
    Windowing simple :
    - on garde les intensités entre les percentiles low et high,
    - puis on renormalise entre 0 et 1.
    Cela améliore le contraste en évitant que quelques valeurs extrêmes écrasent tout.
    """
    if img is None:
        return None

    low_val, high_val = np.percentile(img, [low, high])
    if high_val - low_val < 1e-6:
        # Image quasiment constante -> pas exploitable
        return None

    img = np.clip(img, low_val, high_val)
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img


def crop_to_breast(img, threshold=0.05):
    """
    Recadrage approximatif autour du sein :
    - on considère que le fond est presque noir (valeurs proches de 0),
    - on garde la plus petite bounding box qui contient les pixels > threshold.
    """
    if img is None:
        return None

    mask = img > threshold
    if not mask.any():
        # Si la détection échoue, on garde l'image entière
        return img

    # indices des lignes / colonnes contenant au moins un pixel du sein
    y_any = mask.any(axis=1)
    x_any = mask.any(axis=0)
    y_min, y_max = np.where(y_any)[0][[0, -1]]
    x_min, x_max = np.where(x_any)[0][[0, -1]]

    return img[y_min:y_max + 1, x_min:x_max + 1]


def make_square(img):
    """
    Rend l'image carrée en recadrant au centre :
    - on garde un carré de taille min(h, w).
    - cela évite de "déformer" la géométrie avec des étirements.
    """
    if img is None:
        return None

    h, w = img.shape
    if h == w:
        return img

    if h > w:
        start = (h - w) // 2
        return img[start:start + w, :]
    else:
        start = (w - h) // 2
        return img[:, start:start + h]


def to_pil_512(img, size=512):
    """
    Convertit une image float32 [0,1] en PIL.Image 8 bits et la redimensionne en size×size.
    """
    if img is None:
        return None

    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize((size, size), resample=Image.BILINEAR)
    return pil_img


def process_row_to_pil(row, verbose=False):
    """
    Pipeline complet pour une ligne de subset_df :
    DICOM -> windowing -> recadrage -> carré -> resize 512 -> PIL.Image
    Retourne (pil_img, dcm) ou (None, None) si un problème survient.
    """
    img, dcm = load_dicom_raw(row["patient_id"], row["image_id"], verbose=verbose)
    if img is None:
        return None, None

    img = window_image(img)
    if img is None:
        return None, None

    img = crop_to_breast(img)
    img = make_square(img)
    pil_img = to_pil_512(img)

    return pil_img, dcm



# =============================================================================
# Section 3 - Visualisation AVANT / APRÈS pour quelques exemples
# =============================================================================

def show_before_after(row, title_prefix=""):
    """
    Affiche côte à côte :
    - l'image DICOM windowée (non recadrée),
    - l'image finale 512×512.
    Permet de documenter visuellement le pré-traitement.
    """
    # Image brute + windowing
    raw_img, dcm = load_dicom_raw(row["patient_id"], row["image_id"], verbose=False)
    if raw_img is None:
        print("Impossible de lire l'image brute -> on stoppe.")
        return

    windowed = window_image(raw_img)
    processed, _ = process_row_to_pil(row)  # déjà windowé + crop + resize

    if windowed is None or processed is None:
        print("Pré-traitement impossible pour cet exemple.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(windowed, cmap="gray")
    axes[0].set_title(f"{title_prefix}DICOM après windowing")
    axes[0].axis("off")

    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title("Image finale 512×512")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("Taille brute    :", raw_img.shape)
    print("Taille après crop+square :", windowed.shape, "(avant resize)")
    print("Taille finale   :", processed.size)  # (512, 512)


def get_decodable_example(df, label, max_tries=80):
    """
    Cherche une ligne dont l'image est décodable et pré-traitable pour le label donné (0 ou 1).
    On teste jusqu'à max_tries échantillons au hasard, on renvoie le premier qui fonctionne.
    """
    subset = df[df["cancer"] == label]
    if len(subset) == 0:
        raise RuntimeError(f"Aucune image pour label={label} dans le DataFrame.")

    subset = subset.sample(min(max_tries, len(subset)), random_state=0)

    for _, row in subset.iterrows():
        raw_img, _ = load_dicom_raw(row["patient_id"], row["image_id"], verbose=False)
        if raw_img is None:
            continue
        # on vérifie aussi que le pipeline complet fonctionne
        processed, _ = process_row_to_pil(row)
        if processed is not None:
            return row

    raise RuntimeError(f"Aucune image décodable trouvée pour label={label} après {max_tries} essais.")


# Un exemple positif et un exemple négatif (décodables)
print("Exemple POSITIF (cancer=1)")
pos_example = get_decodable_example(subset_df, label=1)
show_before_after(pos_example, title_prefix="[POS] ")

print("Exemple NÉGATIF (cancer=0)")
neg_example = get_decodable_example(subset_df, label=0)
show_before_after(neg_example, title_prefix="[NEG] ")



# =============================================================================
# Section 4 - Génération des PNG 512×512 + CSV de métadonnées
# =============================================================================

# Dossier de sortie dans /kaggle/working (sera visible dans "Output" du notebook)
OUTPUT_DIR = Path("/kaggle/working/rsna-processed-512")
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

metadata = []  # on stocke les infos pour créer processed_metadata.csv
n_ok = 0
n_fail = 0

for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
    pil_img, dcm = process_row_to_pil(row)
    if pil_img is None:
        n_fail += 1
        continue

    label = int(row["cancer"])
    patient_id = int(row["patient_id"])
    image_id = int(row["image_id"])

    # Chemin de sortie : images/train/{label}/patient_image.png
    out_folder = OUTPUT_IMG_DIR / "train" / str(label)
    out_folder.mkdir(parents=True, exist_ok=True)

    out_name = f"{patient_id}_{image_id}.png"
    out_path = out_folder / out_name

    pil_img.save(out_path)

    metadata.append(
        {
            "patient_id": patient_id,
            "image_id": image_id,
            "cancer": label,
            "filepath": str(out_path.relative_to(OUTPUT_DIR)),  # chemin relatif
        }
    )
    n_ok += 1

print("Images traitées avec succès :", n_ok)
print("Images ignorées (erreurs DICOM ou pré-traitement) :", n_fail)

# Création du CSV de métadonnées
meta_df = pd.DataFrame(metadata)
meta_csv_path = OUTPUT_DIR / "processed_metadata.csv"
meta_df.to_csv(meta_csv_path, index=False)

print("CSV de métadonnées sauvegardé sous :", meta_csv_path)
display(meta_df.head())
meta_df["cancer"].value_counts()



# =============================================================================
# Section 5 - Contrôles visuels sur les PNG générés
# =============================================================================

print("Contenu de OUTPUT_DIR :")
print(os.listdir(OUTPUT_DIR))

print("\nNombre de PNG par classe :")
for label in [0, 1]:
    folder = OUTPUT_IMG_DIR / "train" / str(label)
    n_files = len(os.listdir(folder)) if folder.exists() else 0
    print(f"  Classe {label} : {n_files} fichiers")

# Affichage de quelques PNG aléatoires (3x3) pour vérifier la qualité visuelle
sample_meta = meta_df.sample(9, random_state=123)

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
axes = axes.ravel()

for ax, (_, row) in zip(axes, sample_meta.iterrows()):
    img_path = OUTPUT_DIR / row["filepath"]
    img = Image.open(img_path)

    ax.imshow(img, cmap="gray")
    title = f"label={row['cancer']} \n{row['patient_id']}_{row['image_id']}"
    ax.set_title(title, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()


