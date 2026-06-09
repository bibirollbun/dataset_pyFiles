# =============================================================================
# Section 1 - Imports, configuration et chargement des données tabulaires
# =============================================================================
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pydicom  # lecture des fichiers DICOM

# On masque les warnings "bruit" (pandas / seaborn / pydicom)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Style des graphiques
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Répertoires de données (dataset ajouté via "Add data" dans Kaggle)
DATA_DIR = Path("/kaggle/input/rsna-breast-cancer-detection")
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"

print("DATA_DIR :", DATA_DIR)
print("train.csv présent ?", TRAIN_CSV.exists())
print("train_images présent ?", TRAIN_IMG_DIR.exists())

# Chargement du CSV principal
train = pd.read_csv(TRAIN_CSV)

# Aperçu rapide
display(train.head())
train.info()
display(train.describe(include="all").T)



# =============================================================================
# Section 2 - Analyse du label (cancer) et des variables explicatives principales
# =============================================================================

# --- 2.1 Statistiques de base sur le label `cancer` ---
n_images = len(train)
n_patients = train["patient_id"].nunique()
n_cancer = train["cancer"].sum()
n_no_cancer = n_images - n_cancer
ratio_cancer = n_cancer / n_images

print(f"Nombre d'images              : {n_images}")
print(f"Nombre de patientes          : {n_patients}")
print(f"Nombre d'images positives    : {n_cancer}")
print(f"Nombre d'images négatives    : {n_no_cancer}")
print(f"Taux d'images positives (≈)  : {ratio_cancer:.3%}")

label_counts = train["cancer"].value_counts().sort_index()

fig, ax = plt.subplots()
sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax)
ax.set_xticklabels(["No cancer (0)", "Cancer (1)"])
ax.set_ylabel("Nombre d'images")
ax.set_title("Distribution du label cancer")
plt.show()

print("Répartition en pourcentage :")
display((label_counts / label_counts.sum()) * 100)

# --- 2.2 Distribution de l'âge ---
fig, ax = plt.subplots()
train["age"].hist(bins=30, ax=ax)
ax.set_xlabel("Âge")
ax.set_ylabel("Nombre d'images")
ax.set_title("Distribution de l'âge")
plt.show()

# Distribution de l'âge selon le label (cancer / pas cancer)
fig, ax = plt.subplots()
sns.kdeplot(
    data=train,
    x="age",
    hue="cancer",
    common_norm=False,
    fill=True,
    ax=ax
)
ax.set_title("Distribution de l'âge selon le statut cancer")
plt.show()

# --- 2.3 Répartition des vues (CC / MLO) et de la latéralité (LEFT / RIGHT) ---
print("Valeurs uniques de 'view' :", train["view"].unique())
print("Valeurs uniques de 'laterality' :", train["laterality"].unique())

display(pd.crosstab(train["view"], train["cancer"]))
display(pd.crosstab(train["laterality"], train["cancer"]))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=train, x="view", ax=ax[0])
ax[0].set_title("Nombre d'images par vue")

sns.countplot(data=train, x="laterality", ax=ax[1])
ax[1].set_title("Nombre d'images par côté (LEFT/RIGHT)")

plt.tight_layout()
plt.show()



# =============================================================================
# Section 3 - Exploration des fichiers DICOM et visualisation d'exemples
#             -> version robuste aux erreurs de décompression
# =============================================================================

print("Nombre de dossiers patient dans train_images :", len(os.listdir(TRAIN_IMG_DIR)))

example_patient = str(train["patient_id"].iloc[0])
patient_folder = TRAIN_IMG_DIR / example_patient
print("Exemple patient_id       :", example_patient)
print("Chemin du dossier patient:", patient_folder)
print("Quelques fichiers DICOM  :", os.listdir(patient_folder)[:5])

# --- 3.1 Fonctions utilitaires robustes ---


def load_dicom_image(patient_id, image_id, verbose=False):
    """
    Lit une image DICOM RSNA et retourne :
    - img : array numpy 2D normalisé entre 0 et 1, ou None si non décodable
    - dcm : objet DICOM brut ou None
    Certaines images sont compressées (JPEG Lossless) et nécessitent des plugins.
    Ici, si on ne peut pas les décompresser, on les ignore pour l'EDA.
    """
    dcm_path = TRAIN_IMG_DIR / str(patient_id) / f"{image_id}.dcm"
    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
    except Exception as e:
        if verbose:
            print(f"[WARN] Impossible de lire {dcm_path.name} : {e}")
        return None, None

    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    return img, dcm


def show_dicom_from_row(row, title_prefix=""):
    """
    Affiche l'image DICOM correspondant à une ligne du DataFrame.
    Si l'image n'est pas décodable, ne fait rien et retourne None.
    """
    img, dcm = load_dicom_image(row["patient_id"], row["image_id"], verbose=True)
    if img is None:
        print("Image non décodable, on la saute.")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    title = f"{title_prefix}patient {row['patient_id']} - image {row['image_id']}"
    ax.set_title(title)
    plt.show()

    return dcm


def get_decodable_example(df, label, max_tries=40):
    """
    Cherche une image décodable dans df pour un label donné (0 ou 1).
    On prend jusqu'à max_tries lignes au hasard puis on s'arrête au
    premier DICOM lisible.
    """
    subset = df[df["cancer"] == label].sample(
        min(max_tries, (df["cancer"] == label).sum()),
        random_state=0
    )
    for _, row in subset.iterrows():
        img, dcm = load_dicom_image(row["patient_id"], row["image_id"])
        if img is not None:
            return row
    raise RuntimeError(f"Aucune image décodable trouvée pour label={label}.")


# --- 3.2 Exemple d'image positive / négative (en sautant les DICOM non lisibles) ---
pos_row = get_decodable_example(train, label=1)
neg_row = get_decodable_example(train, label=0)

print("Exemple d'image avec cancer (cancer = 1)")
dcm_pos = show_dicom_from_row(pos_row, title_prefix="[POS] ")

print("Exemple d'image sans cancer (cancer = 0)")
dcm_neg = show_dicom_from_row(neg_row, title_prefix="[NEG] ")

if dcm_pos is not None:
    print("Study Description:", getattr(dcm_pos, "StudyDescription", "N/A"))
    print("Manufacturer:", getattr(dcm_pos, "Manufacturer", "N/A"))
    print("Rows x Columns:", dcm_pos.Rows, "x", dcm_pos.Columns)

# --- 3.3 Petite galerie de 4 images décodables ---

# On échantillonne plus large et on remplit les axes au fur et à mesure
sample = train.sample(80, random_state=42)  # 80 lignes pour augmenter les chances

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

shown = 0
for _, row in sample.iterrows():
    if shown >= 4:
        break  # on a déjà rempli les 4 cases

    img, dcm = load_dicom_image(row["patient_id"], row["image_id"])
    if img is None:
        continue  # on saute les DICOM non décodables

    ax = axes[shown]
    ax.imshow(img, cmap="gray")
    title = (
        f"ID {row['image_id']} | cancer={row['cancer']} | "
        f"{row['view']} | {row['laterality']}"
    )
    ax.set_title(title)
    ax.axis("off")
    shown += 1

# Si certaines cases n'ont pas été utilisées, on les masque proprement
for ax in axes[shown:]:
    ax.axis("off")

plt.tight_layout()
plt.show()


