!nvidia-smi


!pip install monai
!pip install medpy 


import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  

# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt


# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import layers


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
# DEFINE seg-areas  
SEGMENT_CLASSES = {
    0 : 'NOT tumor',   
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE - RED
    2 : 'EDEMA',  # Green
    3 : 'ENHANCING' # original 4 -> converted into 3 later, Yellow
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include

IMG_SIZE=128
import tarfile
file = tarfile.open('../input/brats-2021-task1/BraTS2021_Training_Data.tar')

file.extractall('./BraTS2021_Training_Data')
file.close()
file = tarfile.open('../input/brats-2021-task1/BraTS2021_00621.tar')

file.extractall('./sample_img')
file.close()


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
# DEFINE seg-areas  
SEGMENT_CLASSES = {
    0 : 'NOT tumor',   
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE - RED
    2 : 'EDEMA',  # Green
    3 : 'ENHANCING' # original 4 -> converted into 3 later, Yellow
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 100 
VOLUME_START_AT = 22 # first slice of volume that we will include

IMG_SIZE=128
import tarfile
file = tarfile.open('../input/brats-2021-task1/BraTS2021_Training_Data.tar')

file.extractall('./BraTS2021_Training_Data')
file.close()
file = tarfile.open('../input/brats-2021-task1/BraTS2021_00621.tar')

file.extractall('./sample_img')
file.close()


import os
from glob import glob

# Lokasi hasil ekstraksi
DATA_PATH = "./BraTS2021_Training_Data"

# Hitung jumlah folder pasien
patient_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
num_patients = len(patient_folders)

# Hitung jumlah semua file .nii.gz di dalam folder
nii_files = glob(os.path.join(DATA_PATH, "**", "*.nii.gz"), recursive=True)
num_files = len(nii_files)

print(f"ğŸ§  Jumlah pasien (volume MRI): {num_patients}")
print(f"ğŸ“„ Total file NIfTI (.nii.gz): {num_files}")

# (Opsional) tampilkan beberapa contoh folder dan file
print("\nğŸ“‚ Contoh folder:", patient_folders[:5])
print("ğŸ§© Contoh file:", nii_files[:5])


file = tarfile.open('../input/brats-2021-task1/BraTS2021_00621.tar')

file.extractall('./sample_img')
file.close()

nSample = os.listdir('./sample_img')
nSample


TRAIN_DATASET_PATH = './BraTS2021_Training_Data/'
nSample = os.listdir(TRAIN_DATASET_PATH + 'BraTS2021_01261')
nSample


test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_flair.nii.gz').get_fdata()
test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_t1.nii.gz').get_fdata()
test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_t1ce.nii.gz').get_fdata()
test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_t2.nii.gz').get_fdata()
test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_seg.nii.gz').get_fdata()


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 25
ax1.imshow(test_image_flair[:,:,test_image_flair.shape[0]//2-slice_w], cmap = 'gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:,:,test_image_t1.shape[0]//2-slice_w], cmap = 'gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:,:,test_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:,:,test_image_t2.shape[0]//2-slice_w], cmap = 'gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:,:,test_mask.shape[0]//2-slice_w])
ax5.set_title('Mask')


fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_image_t1[50:-50,:,:]), 90, resize=True), cmap ='gray')


fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_mask[60:-60,:,:]), 90, resize=True), cmap ='gray')


niimg = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_flair.nii.gz')
nimask = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_seg.nii.gz')

fig, axes = plt.subplots(nrows=4, figsize=(30, 40))


nlplt.plot_anat(niimg,
                title='BraTS18_Training_001_flair.nii plot_anat',
                axes=axes[0])

nlplt.plot_epi(niimg,
               title='BraTS18_Training_001_flair.nii plot_epi',
               axes=axes[1])

nlplt.plot_img(niimg,
               title='BraTS18_Training_001_flair.nii plot_img',
               axes=axes[2])

nlplt.plot_roi(nimask, 
               title='BraTS18_Training_001_flair.nii with mask plot_roi',
               bg_img=niimg, 
               axes=axes[3], cmap='Paired')

plt.show()


import tarfile
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


import os
import albumentations as A
import nibabel as nib
import numpy as np


class ImageReader:
    def __init__(self, root: str, img_size: int = 256, normalize: bool = False, single_class: bool = False):
        # Ukuran minimal padding (agar dimensi seragam)
        pad_size = 256 if img_size > 256 else 224

        # Gunakan 'fill' (bukan 'value') untuk Albumentations >= 1.3
        self.resize = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, fill=0),
            A.Resize(img_size, img_size)
        ])

        self.normalize = normalize
        self.single_class = single_class
        self.root = root

    def read_file(self, path: str) -> dict:
        """
        Membaca file NIfTI (.nii.gz) dan mengembalikan hasil preprocess:
        - 'scan' : volume MRI 3D (numpy array)
        - 'segmentation' : mask tumor 3D (numpy array)
        - 'orig_shape' : dimensi asli sebelum resize
        """
        scan_type = path.split('_')[-1]
        raw_image = nib.load(path).get_fdata()
        raw_mask = nib.load(path.replace(scan_type, 'seg.nii.gz')).get_fdata()

        processed_frames, processed_masks = [], []

        for frame_idx in range(raw_image.shape[2]):
            frame = raw_image[:, :, frame_idx]
            mask = raw_mask[:, :, frame_idx]

            # Normalisasi (opsional)
            if self.normalize:
                if frame.max() > 0:
                    frame = frame / frame.max()
                frame = frame.astype(np.float32)
            else:
                frame = frame.astype(np.uint8)

            # Resize + Pad
            resized = self.resize(image=frame, mask=mask)

            # Pastikan mask tetap dalam tipe uint8
            mask_resized = resized['mask'].astype(np.uint8)

            # Jika single_class=True â†’ ubah semua nilai >0 jadi 1
            if self.single_class:
                mask_resized = (mask_resized > 0).astype(np.uint8)

            processed_frames.append(resized['image'])
            processed_masks.append(mask_resized)

        return {
            'scan': np.stack(processed_frames, 0),
            'segmentation': np.stack(processed_masks, 0),
            'orig_shape': raw_image.shape
        }

    def load_patient_scan(self, idx: int, scan_type: str = 'flair') -> dict:
        """
        Memuat scan MRI pasien berdasarkan indeks ID dan tipe scan.
        Contoh path:
        /kaggle/working/BraTS2021_Training_Data/BraTS2021_00095/BraTS2021_00095_flair.nii.gz
        """
        patient_id = str(idx).zfill(5)
        scan_filename = f'{self.root}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_{scan_type}.nii.gz'
        return self.read_file(scan_filename)


import plotly.graph_objects as go
import numpy as np

# generate_3d_scatter(): 
# Fungsi ini membuat sebuah objek Scatter3d dari Plotly untuk menampilkan data tiga dimensi.
# Fungsi menerima parameter seperti x, y, z (array koordinat), colors (array warna),
# size (ukuran titik), opacity (tingkat transparansi), scale (skala warna),
# hover (informasi yang muncul saat kursor diarahkan), dan name (nama objek Scatter3d).
def generate_3d_scatter(
    x:np.array, y:np.array, z:np.array, colors:np.array,
    size:int=3, opacity:float=0.2, scale:str='Teal',
    hover:str='skip', name:str='MRI'
) -> go.Scatter3d:
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers', hoverinfo=hover,
        marker = dict(
            size=size, opacity=opacity,
            color=colors, colorscale=scale
        ),
        name=name
    )


class ImageViewer3d():
    # Konstruktor (__init__) dari kelas ImageViewer3d menerima parameter seperti:
    # reader (objek ImageReader), mri_downsample (tingkat downsampling citra MRI),
    # mri_colorscale (skala warna untuk citra MRI), dan voxel_size (ukuran voxel).
    def __init__(self, reader:ImageReader, mri_downsample:int=10, mri_colorscale:str='Ice', voxel_size:float=0.1) -> None:
        self.reader = reader
        self.mri_downsample = mri_downsample
        self.mri_colorscale = mri_colorscale
        self.voxel_size = voxel_size
        
    # Metode load_clean_mri() digunakan untuk memuat data citra MRI yang sudah dibersihkan.
    # Metode ini menerima parameter image (array numpy dari citra MRI) dan orig_dim (dimensi asli citra).
    # Ia menghitung koordinat x, y, z dari voxel yang bernilai > 0,
    # kemudian mengambil sampel berdasarkan faktor downsampling (mri_downsample),
    # dan mengembalikan dictionary berisi koordinat serta warna tiap titik.
    def load_clean_mri(self, image:np.array, orig_dim:int) -> dict:
        shape_offset = image.shape[1]/orig_dim
        z, x, y = (image > 0).nonzero()
        # hanya mengambil 1 dari setiap (1/mri_downsample) sampel
        x, y, z = x[::self.mri_downsample], y[::self.mri_downsample], z[::self.mri_downsample]
        colors = image[z, x, y]
        return dict(x=x/shape_offset, y=y/shape_offset, z=z, colors=colors)
    
    # Metode load_tumor_segmentation() digunakan untuk memuat data segmentasi tumor.
    # Metode ini menerima parameter image (array numpy segmentasi) dan orig_dim (dimensi asli citra).
    # Ia menghitung koordinat x, y, z dari voxel yang sesuai dengan kelas tumor tertentu,
    # menggunakan faktor sampling berbeda (1/1, 1/3, 1/5) untuk masing-masing kelas,
    # kemudian mengembalikan dictionary berisi koordinat dan warna tiap kelas tumor.
    def load_tumor_segmentation(self, image:np.array, orig_dim:int) -> dict:
        tumors = {}
        shape_offset = image.shape[1]/orig_dim
        # sampling 1/1, 1/3, dan 1/5 untuk kelas jaringan tumor 1 (core), 2 (invaded), dan 4 (enhancing)
        sampling = {
            1: 1, 2: 3, 4: 5
        }
        for class_idx in sampling:
            z, x, y = (image == class_idx).nonzero()
            x, y, z = x[::sampling[class_idx]], y[::sampling[class_idx]], z[::sampling[class_idx]]
            tumors[class_idx] = dict(
                x=x/shape_offset, y=y/shape_offset, z=z,
                colors=class_idx/4
            )
        return tumors

    
    # Fungsi collect_patient_data digunakan untuk mengumpulkan data pasien dan menghitung
    # parameter terkait ukuran serta distribusi area dalam citra MRI.
    # Pertama, fungsi memuat citra MRI bersih (clean_mri) dan segmentasi tumor (tumors)
    # dari argumen scan. 
    # Kemudian, menghitung volume voxel berdasarkan kubik dari voxel_size.
    # Jumlah total titik (markers_created) dihitung dari total titik di citra MRI bersih
    # dan semua area tumor.
    # Selanjutnya, menghitung jumlah titik per area, persentase masing-masing area terhadap total,
    # serta volume (dalam cmÂ³) tiap area.
    # Hasil kemudian dicetak dan fungsi mengembalikan daftar objek Scatter3d
    # untuk setiap area (MRI bersih dan tiga tipe jaringan tumor).
    def collect_patient_data(self, scan:dict) -> tuple:
        clean_mri = self.load_clean_mri(scan['scan'], scan['orig_shape'][0])
        tumors = self.load_tumor_segmentation(scan['segmentation'], scan['orig_shape'][0])
        
        voxel_volume = self.voxel_size ** 3
        markers_created = clean_mri['x'].shape[0] + sum(tumors[class_idx]['x'].shape[0] for class_idx in tumors)
        
        clean_mri_diem = clean_mri['x'].shape[0]
        tumor1_diem = tumors[1]['x'].shape[0]
        tumor2_diem = tumors[2]['x'].shape[0]
        tumor4_diem = tumors[4]['x'].shape[0]
        
        clean_mri_tile = round(clean_mri_diem /markers_created*100, 2)
        tumor1_tile = round(tumor1_diem /markers_created*100, 2)
        tumor2_tile = round(tumor2_diem /markers_created*100, 2)
        tumor4_tile = round(tumor4_diem /markers_created*100, 2)
 
        clean_mri_kichthuoc = str(round(clean_mri_diem * voxel_volume, 2)) + ' cm^3'
        tumor1_kichthuoc = str(round(tumor1_diem * voxel_volume, 2)) + ' cm^3'
        tumor2_kichthuoc = str(round(tumor2_diem * voxel_volume, 2)) + ' cm^3'
        tumor4_kichthuoc = str(round(tumor4_diem * voxel_volume, 2)) + ' cm^3'
        
        print('Citra MRI otak (bersih):', clean_mri_diem ,'titik,', clean_mri_tile ,'%',clean_mri_kichthuoc)
        print('Inti tumor:', tumor1_diem , 'titik,', tumor1_tile ,'%,', tumor1_kichthuoc)
        print('Jaringan di sekitar yang terinfiltrasi tumor:', tumor2_diem ,'titik,', tumor2_tile ,'%,', tumor2_kichthuoc)
        print('Area tumor yang mengalami peningkatan kontras gadolinium:', tumor4_diem ,'titik,', tumor4_tile,'%,', tumor4_kichthuoc)
        
        return [
            generate_3d_scatter(**clean_mri, scale=self.mri_colorscale, opacity=0.3, hover='skip', name='Citra MRI otak (bersih) ('+ clean_mri_kichthuoc +')'),
            generate_3d_scatter(**tumors[1], opacity=0.8, hover='all', name='Inti tumor (' + tumor1_kichthuoc + ')'),
            generate_3d_scatter(**tumors[2], opacity=0.4, hover='all', name='Jaringan sekitar terinfiltrasi (' + tumor2_kichthuoc + ')'),
            generate_3d_scatter(**tumors[4], opacity=0.4, hover='all', name='Area peningkatan gadolinium (' + tumor4_kichthuoc + ')'),
        ], markers_created
 

    def get_3d_scan(self, patient_idx:int, scan_type:str='flair') -> go.Figure:
        scan = self.reader.load_patient_scan(patient_idx, scan_type)
        data, num_markers = self.collect_patient_data(scan)
        fig = go.Figure(data=data)
        fig.update_layout(
            title=f"[Pasien id:{patient_idx}] pemindaian MRI otak ({num_markers} titik)",
            legend_title="Kelas voxel (klik untuk menampilkan/sembunyikan)",
            font=dict(
                family="Courier New, monospace",
                size=14,
            ),
            margin=dict(
                l=0,r=0,b=0,t=30
            ),
            legend=dict(itemsizing='constant')
        )
        return fig



reader = ImageReader('/kaggle/working/BraTS2021_Training_Data', img_size=128, normalize=True, single_class=False)
viewer = ImageViewer3d(reader, mri_downsample=25)


fig = viewer.get_3d_scan(0, 't1')
plotly.offline.iplot(fig)


fig = viewer.get_3d_scan(9, 't1ce')
plotly.offline.iplot(fig)


fig = viewer.get_3d_scan(9, 'flair')
plotly.offline.iplot(fig)


img_id = "00009"
import matplotlib.pyplot as plt
for i, nii in enumerate([f'/kaggle/working/BraTS2021_Training_Data/BraTS2021_{img_id}/BraTS2021_{img_id}_{s_type}.nii.gz' for s_type in ["flair", "t1", "t1ce", "t2", "seg"]]):
    # PLOTTING
    image = nib.load(nii).get_fdata()
    slices = image.shape[-1]
    rows = int(np.ceil((slices/2)/10))
    plt.figure(figsize=(20, rows*2))
    plt.suptitle(f"\n\n\n{nii.rsplit('_', 1)[-1].split('.', 1)[0]} SCAN\n".upper(), fontsize=18, fontweight="bold")
    for j in range(0, slices, 2):
        plt.subplot(rows, 10, 1+j//2)
        plt.axis(False)
        plt.imshow(image[:, :, j], cmap="bone")
    plt.tight_layout()
    plt.show()


import os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib
from skimage.transform import resize
from monai.networks.nets import UNet
from medpy.metric import binary
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


DATA_DIR = "/kaggle/working/BraTS2021_Training_Data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 10
PATIENCE = 5
SCALE_FACTORS = [1.0, 0.75, 0.5, 0.25]


print("ğŸ§  GPU yang tersedia:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  â€¢ {i}: {torch.cuda.get_device_name(i)}")


def downsample_volume(volume, scale_factor):
    new_shape = np.round(np.array(volume.shape) * scale_factor).astype(int)
    return resize(volume, new_shape, order=3, preserve_range=True, anti_aliasing=True)

def downsample_mask(mask, scale_factor):
    new_shape = np.round(np.array(mask.shape) * scale_factor).astype(int)
    return resize(mask, new_shape, order=0, preserve_range=True, anti_aliasing=False)

def pad_to_multiple(x, multiple=16):
    shape = x.shape[-3:]
    pad = [(multiple - s % multiple) % multiple for s in shape]
    pad = [0, pad[2], 0, pad[1], 0, pad[0]]
    return F.pad(x, pad, mode="constant", value=0)


class BraTSDataset(Dataset):
    def __init__(self, flair_list, scale_factor=1.0):
        """
        flair_list: daftar path *_flair.nii.gz
        """
        self.flair_list = flair_list
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.flair_list)

    def __getitem__(self, idx):
        flair_path = self.flair_list[idx]
        base = flair_path.replace("_flair.nii.gz", "")
        mods = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]
        
        vols = []
        for m in mods:
            vol = nib.load(base + m).get_fdata()
            vol = downsample_volume(vol, self.scale_factor)
            vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-5)
            vols.append(vol)
        img = np.stack(vols, axis=0)  # shape (4, D, H, W)

        mask = nib.load(base + "_seg.nii.gz").get_fdata()
        mask = downsample_mask(mask, self.scale_factor)
        mask = np.where(mask > 0, 1, 0)  # binerisasi
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def build_unet():
    model = UNet(
        spatial_dims=3,
        in_channels=4,   # empat modalitas
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    if torch.cuda.device_count() > 1:
        print(f"ğŸš€ Menggunakan {torch.cuda.device_count()} GPU (DataParallel aktif)")
        model = nn.DataParallel(model)
    return model.to(DEVICE)


def dice_score(pred, gt): return binary.dc(pred, gt)
def jaccard_index(pred, gt): return binary.jc(pred, gt)
def hausdorff_distance(pred, gt): return binary.hd95(pred, gt)

def compute_ssim_3d(pred, gt):
    ssim_scores = []
    for z in range(pred.shape[2]):
        try:
            ssim_score = ssim(gt[:, :, z], pred[:, :, z], data_range=1.0)
            ssim_scores.append(ssim_score)
        except ValueError:
            continue
    return np.mean(ssim_scores) if len(ssim_scores) > 0 else 0


def train_model(model, loader, criterion, optimizer):
    scaler = torch.cuda.amp.GradScaler()
    best_loss = np.inf
    patience_ct = 0
    losses = []
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for img, msk in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            img, msk = img.to(DEVICE, non_blocking=True), msk.to(DEVICE, non_blocking=True)
            img, msk = pad_to_multiple(img), pad_to_multiple(msk)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(img)
                min_shape = [min(o, m) for o, m in zip(out.shape, msk.shape)]
                out, msk = out[..., :min_shape[-3], :min_shape[-2], :min_shape[-1]], \
                           msk[..., :min_shape[-3], :min_shape[-2], :min_shape[-1]]
                loss = criterion(out, msk)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"âœ… Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Early stopping (val loader belum digunakan di versi ini)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"â�¹ Early stop di epoch {epoch+1}")
                break

    return best_state, losses, epoch + 1



def evaluate_model(model, loader):
    model.eval()
    dice_list, jacc_list, haus_list, ssim_list = [], [], [], []
    with torch.no_grad():
        for img, msk in loader:
            img = img.to(DEVICE)
            out = torch.sigmoid(model(pad_to_multiple(img))).cpu().numpy()[0,0]
            pred = (out > 0.5).astype(np.uint8)
            gt = msk.numpy()[0,0]
            min_shape = [min(a,b) for a,b in zip(pred.shape, gt.shape)]
            pred, gt = pred[:min_shape[0], :min_shape[1], :min_shape[2]], gt[:min_shape[0], :min_shape[1], :min_shape[2]]
            dice_list.append(dice_score(pred, gt))
            jacc_list.append(jaccard_index(pred, gt))
            haus_list.append(hausdorff_distance(pred, gt))
            ssim_list.append(compute_ssim_3d(pred, gt))
    return np.mean(dice_list), np.mean(jacc_list), np.mean(haus_list), np.mean(ssim_list)


def train_and_evaluate(scale_factor):
    print(f"\nğŸ§© Training skala {scale_factor}x (4 channel, GPU aktif)\n")

    images = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*", "*_flair.nii.gz")))[:100]
    dataset = BraTSDataset(images, scale_factor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = build_unet()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    torch.cuda.synchronize()
    start_train = time.time()
    best_state, losses, last_epoch = train_model(model, loader, criterion, optimizer)
    torch.cuda.synchronize()
    train_time = time.time() - start_train

    model.load_state_dict(best_state)
    torch.cuda.synchronize()
    start_eval = time.time()
    dice, jacc, haus, ssim_score = evaluate_model(model, loader)
    torch.cuda.synchronize()
    eval_time = time.time() - start_eval

    # Visualisasi overlay
    plt.figure(figsize=(12,4))
    img, msk = dataset[0]
    mid = img.shape[-1]//2
    plt.subplot(1,3,1); plt.imshow(msk[0,:,:,mid], cmap='gray'); plt.title("Ground Truth")
    plt.subplot(1,3,2); plt.imshow(img[0,:,:,mid], cmap='gray'); plt.title("FLAIR")
    plt.subplot(1,3,3)
    plt.imshow(img[0,:,:,mid], cmap='gray')
    plt.imshow(msk[0,:,:,mid], cmap='Reds', alpha=0.4)
    plt.title(f"Overlay (scale={scale_factor})")
    plt.tight_layout(); plt.show()

    return {
        "scale": scale_factor,
        "epochs_run": last_epoch,
        "loss_final": losses[-1],
        "dice": dice,
        "jaccard": jacc,
        "hausdorff": haus,
        "ssim": ssim_score,
        "train_time_sec": round(train_time,2),
        "eval_time_sec": round(eval_time,2)
    }


images = sorted(glob.glob(os.path.join(DATA_DIR, "BraTS2021_*", "*_flair.nii.gz")))[:100]

dataset = BraTSDataset(images, scale_factor=1.0)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Total data:", len(dataset))
print("Batch size:", BATCH_SIZE)
print("Total batch:", len(loader))



results = []
for s in SCALE_FACTORS:
    torch.cuda.empty_cache()
    results.append(train_and_evaluate(s))

import pandas as pd
results_df = pd.DataFrame(results)
print("\nğŸ“Š Hasil Evaluasi Multi-Modal:")
print(results_df)


fig, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].plot(results_df["scale"], results_df["dice"], 'o-', label="Dice")
ax[0].plot(results_df["scale"], results_df["jaccard"], 's--', label="Jaccard")
ax[0].plot(results_df["scale"], results_df["ssim"], 'd-.', label="SSIM")
ax[0].set_title("Similarity Metrics vs Downsampling")
ax[0].set_xlabel("Scale Factor (â†“ resolusi)")
ax[0].set_ylabel("Score"); ax[0].legend(); ax[0].grid(True)

ax[1].plot(results_df["scale"], results_df["train_time_sec"], 'o-', label="Train Time")
ax[1].plot(results_df["scale"], results_df["eval_time_sec"], 's--', label="Eval Time")
ax[1].set_title("GPU Runtime vs Downsampling")
ax[1].set_xlabel("Scale Factor"); ax[1].set_ylabel("Seconds")
ax[1].legend(); ax[1].grid(True)

plt.tight_layout(); plt.show()


import os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt


DATA_DIR   = "/kaggle/working/BraTS2021_Training_Data"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
LR         = 1e-4
EPOCHS     = 10
PATIENCE   = 5

print("ğŸ§  GPU tersedia:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  â€¢ {i}: {torch.cuda.get_device_name(i)}")


def downsample_volume(vol, sf):
    new_shape = np.round(np.array(vol.shape) * sf).astype(int)
    return resize(vol, new_shape, order=3, preserve_range=True, anti_aliasing=True)

def downsample_mask(msk, sf):
    new_shape = np.round(np.array(msk.shape) * sf).astype(int)
    return resize(msk, new_shape, order=0, preserve_range=True, anti_aliasing=False)

def pad_to_multiple(x, m=16):
    shp = x.shape[-3:]
    pad = [(m - s % m) % m for s in shp]
    pad = [0,pad[2],0,pad[1],0,pad[0]]
    return F.pad(x, pad, mode="constant", value=0)

class BraTSDataset(Dataset):
    def __init__(self, flair_paths, scale_factor=1.0):
        self.flair_paths  = flair_paths
        self.scale_factor = scale_factor

    def __len__(self): return len(self.flair_paths)

    def __getitem__(self, idx):
        base = self.flair_paths[idx].replace("_flair.nii.gz","")
        mods = ["_flair.nii.gz","_t1.nii.gz","_t1ce.nii.gz","_t2.nii.gz"]
        vols = []
        for m in mods:
            v = nib.load(base+m).get_fdata()
            v = downsample_volume(v, self.scale_factor)
            v = (v - np.mean(v)) / (np.std(v)+1e-5)
            vols.append(v)
        img = np.stack(vols,axis=0)  # (4,D,H,W)
        msk = nib.load(base+"_seg.nii.gz").get_fdata()
        msk = downsample_mask(msk, self.scale_factor)
        msk = np.where(msk>0,1,0)[None]  # binary mask
        return torch.tensor(img,dtype=torch.float32), torch.tensor(msk,dtype=torch.float32)


class UNet3D(nn.Module):
    def __init__(self,in_ch=4,out_ch=1):
        super().__init__()
        def block(ic,oc):
            return nn.Sequential(
                nn.Conv3d(ic,oc,3,padding=1), nn.BatchNorm3d(oc), nn.ReLU(inplace=True),
                nn.Conv3d(oc,oc,3,padding=1), nn.BatchNorm3d(oc), nn.ReLU(inplace=True)
            )
        self.enc1 = block(in_ch,16)
        self.enc2 = block(16,32)
        self.enc3 = block(32,64)
        self.pool = nn.MaxPool3d(2)
        self.up1  = nn.ConvTranspose3d(64,32,2,stride=2)
        self.dec1 = block(64,32)
        self.up2  = nn.ConvTranspose3d(32,16,2,stride=2)
        self.dec2 = block(32,16)
        self.out  = nn.Conv3d(16,out_ch,1)
    def forward(self,x):
        e1=self.enc1(x)
        e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2))
        d1=self.up1(e3); d1=torch.cat([d1,e2],1); d1=self.dec1(d1)
        d2=self.up2(d1); d2=torch.cat([d2,e1],1); d2=self.dec2(d2)
        return self.out(d2)

def build_unet():
    model = UNet3D(in_ch=4,out_ch=1)
    if torch.cuda.device_count()>1:
        print(f"âš¡ Menggunakan {torch.cuda.device_count()} GPU (DataParallel aktif)")
        model = nn.DataParallel(model)
    return model.to(DEVICE)


def train_model_with_val(model,train_loader,val_loader,criterion,optimizer,epochs=100):
    try:    scaler = torch.amp.GradScaler("cuda")
    except: scaler = torch.cuda.amp.GradScaler()
    best_loss=np.inf; patience_ct=0; best_state=None
    tr_losses, vl_losses=[],[]
    for ep in range(epochs):
        # --- Train ---
        model.train(); tot=0
        for img,msk in tqdm(train_loader,desc=f"Epoch {ep+1}/{epochs} (Train)"):
            img,msk=img.to(DEVICE),msk.to(DEVICE)
            img,msk=pad_to_multiple(img),pad_to_multiple(msk)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out=model(img)
                out=out[...,:msk.shape[-3],:msk.shape[-2],:msk.shape[-1]]
                loss=criterion(out,msk)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            tot+=loss.item()
        tr_losses.append(tot/len(train_loader))
        # --- Val ---
        model.eval(); val_tot=0
        with torch.no_grad():
            for img,msk in tqdm(val_loader,desc=f"Epoch {ep+1}/{epochs} (Val)"):
                img,msk=img.to(DEVICE),msk.to(DEVICE)
                img,msk=pad_to_multiple(img),pad_to_multiple(msk)
                out=model(img)
                out=out[...,:msk.shape[-3],:msk.shape[-2],:msk.shape[-1]]
                loss=criterion(out,msk); val_tot+=loss.item()
        vl_losses.append(val_tot/len(val_loader))
        print(f"âœ… Epoch {ep+1:03d} | Train Loss: {tr_losses[-1]:.4f} | Val Loss: {vl_losses[-1]:.4f}")
        if vl_losses[-1]<best_loss:
            best_loss=vl_losses[-1]; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience_ct=0
        else:
            patience_ct+=1
            if patience_ct>=PATIENCE:
                print(f"â�¹ Early stop di epoch {ep+1}"); break
    return best_state,tr_losses,vl_losses


mages=sorted(glob.glob(os.path.join(DATA_DIR,"BraTS2021_*","*_flair.nii.gz")))[:100]
dataset=BraTSDataset(images,scale_factor=1.0)
val_split=int(0.8*len(dataset))
train_ds,val_ds=random_split(dataset,[val_split,len(dataset)-val_split])
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)

model=build_unet()
optimizer=optim.Adam(model.parameters(),lr=LR)
criterion=nn.BCEWithLogitsLoss()

best_state,tr_losses,vl_losses=train_model_with_val(model,train_loader,val_loader,criterion,optimizer,epochs=EPOCHS)
model.load_state_dict(best_state)


plt.figure(figsize=(8,5))
plt.plot(tr_losses,label="Train Loss",marker='o')
plt.plot(vl_losses,label="Validation Loss",marker='x')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve (4-Channel U-Net 3D)")
plt.legend(); plt.grid(True); plt.show()

