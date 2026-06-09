%pip install celluloid --q
%pip install torchio --q


%matplotlib notebook

from tqdm.notebook import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import nibabel as nib
import pydicom

import torch
import torchio as tio 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from celluloid import Camera
from IPython.display import HTML

import matplotlib.pyplot as plt
%matplotlib inline


labels = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")


root_path = Path("/kaggle/input/rsna-intracranial-aneurysm-detection/series/")
patient_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])


patient_dirs[0]


def get_aneurysm_present_for_patient(patient_dir: Path, labels_df: pd.DataFrame):
    """
    Given a patient directory Path (SeriesInstanceUID) and the labels DataFrame,
    return the Aneurysm Present value (0 or 1).
    """
    # Extract SeriesInstanceUID from folder name
    series_uid = patient_dir.name
    
    # Filter DataFrame
    result = labels_df.loc[labels_df["SeriesInstanceUID"] == series_uid, "Aneurysm Present"]
    
    if not result.empty:
        return int(result.values[0])
    else:
        # UID not found
        return None
        

def get_modality_for_patient(patient_dir: Path, labels_df: pd.DataFrame):
    """
    Given a patient directory Path (SeriesInstanceUID) and the labels DataFrame,
    return the Modality value (e.g., 'CT', 'MR', etc.).
    """
    # Extract SeriesInstanceUID from folder name
    series_uid = patient_dir.name
    
    # Filter DataFrame
    result = labels_df.loc[labels_df["SeriesInstanceUID"] == series_uid, "Modality"]
    
    if not result.empty:
        return result.values[0]
    else:
        # UID not found
        return None


for i in range(5):
    status = get_aneurysm_present_for_patient(patient_dirs[i], labels)
    print(status)


def crop_from_slice(tensor):
    return tensor[..., :30]


process = tio.Compose([
    tio.ToCanonical(),
    tio.Resample((1, 1, 1)),
    tio.RescaleIntensity((-1, 1)),
    tio.Lambda(crop_from_slice),
    tio.CropOrPad((500, 500, 200)),    
])

augmentation = tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10))

train_transform = tio.Compose([process, augmentation])
val_transform = tio.Compose([process])


class AneurysmSubjectDataset(Dataset):
    """
    Loads full 3D volumes (stacked 2D slices) per patient, along with
    the aneurysm label and modality information.

    Args:
        patient_dirs (list[Path]): list of SeriesInstanceUID directories
        labels_df (pd.DataFrame): dataframe with labels
        train (bool): whether to use train split or validation split
        test_size (float): fraction of data to use as validation
        transform: optional transform function (applied to 3D volume)
        random_state (int): random seed for train/validation split
    """
    def __init__(self, patient_dirs, labels_df, train=True, test_size=0.2, transform=None, random_state=42):
        self.labels_df = labels_df
        self.transform = transform

        # Split into train and validation sets
        train_dirs, val_dirs = train_test_split(
            patient_dirs, test_size=test_size, random_state=random_state, shuffle=True
        )
        self.patient_dirs = train_dirs if train else val_dirs

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]

        # --- Load DICOM slices ---
        dicom_files = list(patient_dir.glob("*.dcm"))
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda dcm: int(dcm.InstanceNumber))

        slices = [dcm.pixel_array for dcm in dicoms]

        # Stack into 3D volume: [H, W, D]
        volume = np.stack(slices, axis=-1)  # [H, W, D]
        volume = torch.from_numpy(volume).unsqueeze(0).float()  # [1, H, W, D]
    
        if self.transform:
            import torchio as tio
            subject = tio.Subject(image=tio.ScalarImage(tensor=volume))
            subject = self.transform(subject)
            volume = subject.image.data  # Extract transformed tensor

        # --- Labels and Metadata ---
        label = get_aneurysm_present_for_patient(patient_dir, self.labels_df)
        modality = get_modality_for_patient(patient_dir, self.labels_df)

        label = torch.tensor(label, dtype=torch.long)
        modality = str(modality)  # keep as string, not tensor

        return {
            "image": volume,
            "label": label,
            "modality": modality,
        }


train_dataset = AneurysmSubjectDataset(
    patient_dirs=patient_dirs,
    labels_df=labels,
    train=True,
    test_size=0.5,
    transform=None  
)

val_dataset = AneurysmSubjectDataset(
    patient_dirs=patient_dirs,
    labels_df=labels,
    train=False,
    test_size=0.5,
    transform=None 
)


sample = train_dataset[1]
img = sample["image"]
label = sample["label"]
modal = sample["modality"]

print("Modality:", modal)


for i in range(5):
    sample = train_dataset[i]
    img = sample["image"]
    label = sample["label"]
    modal = sample["modality"]

    print("ImgShape:", img.shape)
    print("AneurysmPresent:", label)
    print("Modality:", modal)


print(len(train_dataset), len(val_dataset))


fig = plt.figure()
camera = Camera(fig)

sample = train_dataset[1]
img = sample["image"]
label = sample["label"]
modal = sample["modality"]

print("ImgShape:", img.shape)
print("AneurysmPresent:", label)
print("Modality:", modal)

for i in range(img.shape[3]): 
    plt.imshow(img[0, :, :, i], cmap="bone")
    camera.snap();

animation = camera.animate(interval=50);
HTML(animation.to_html5_video())




