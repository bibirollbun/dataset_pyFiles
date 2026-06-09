import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nibabel as nib
from tqdm import tqdm
import glob
from PIL import Image


#### infer https://www.kaggle.com/code/walterok/batch-inference-making-masks-using-nnunet/notebook
def make_if_dont_exist(folder_path,overwrite=False):
    """
    creates a folder if it does not exists
    input: 
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder 
    """
    if os.path.exists(folder_path):
        
        if not overwrite:
            print(f'{folder_path} exists.')
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


make_if_dont_exist('/kaggle/working/nnUNet')
# make_if_dont_exist('/kaggle/working/nnUNet_dep_libs/')


# ### install MedPy-0.4.0
# !cp -rf /kaggle/input/nnunet-packages/packages/MedPy-0.4.0/MedPy-0.4.0  /kaggle/working/nnUNet_dep_libs/
# !cp -rf /kaggle/input/nnunet-packages/packages/batchgenerators-0.23 /kaggle/working/nnUNet_dep_libs/
# !cp -rf /kaggle/input/nnunet-packages/packages/dicom2nifti-2.3.3 /kaggle/working/nnUNet_dep_libs/
# respository_dir = '/kaggle/working/nnUNet_dep_libs/'
# os.chdir(respository_dir)
# !ls


# ### install MedPy
# medpy_dir = f'/kaggle/working/nnUNet_dep_libs/MedPy-0.4.0/'
# os.chdir(medpy_dir)
# !ls
# !pip install -e .


# ### install batchgenerators
# batchgenerators_dir = f'/kaggle/working/nnUNet_dep_libs/batchgenerators-0.23/'
# os.chdir(batchgenerators_dir)
# !ls
# !pip install -e .


# ### install dicom2nifti
# dicom2nifti_dir = f'/kaggle/working/nnUNet_dep_libs/dicom2nifti-2.3.3/dicom2nifti-2.3.3'
# os.chdir(dicom2nifti_dir)
# !ls
# !pip install -e .


### install nnUNet
!cp -rf /kaggle/input/nnunetv2/nnUNet /kaggle/working/
nnunet_dir = '/kaggle/working/'
os.chdir(nnunet_dir)
!ls
!pip install -e .

# Verify installation
!nnUNetv2_train -h
%cd ..


# !git clone https://github.com/MIC-DKFZ/nnUNet.git
# !cd nnUNet
# !pip install -e .


base_dir = "/kaggle/working/"
base_nnunet_dir = "/kaggle/working/input/"
raw_data_base_dir = "/kaggle/working/nnUNet_raw"
preprocessed_dir = "/kaggle/working/nnUNet_preprocessed"
trained_models_dir = "/kaggle/working/nnUNet_results"

task_name = 'Dataset102_UWMGITImageSegmentation' #change here for different task name
task_folder_name = os.path.join(base_dir, 'nnUNet_raw', task_name)
imagestr = os.path.join(task_folder_name,'imagesTr')
imagests = os.path.join(task_folder_name,'imagesTs')
labelstr = os.path.join(task_folder_name,'labelsTr')
output_dir = "/kaggle/working/output/segmented"

# make_if_dont_exist(base_nnunet_dir, overwrite = False)
make_if_dont_exist(raw_data_base_dir, overwrite = False)
make_if_dont_exist(preprocessed_dir, overwrite = False)
make_if_dont_exist(trained_models_dir, overwrite = False)
make_if_dont_exist(imagestr, overwrite = False)
make_if_dont_exist(imagests, overwrite = False)
make_if_dont_exist(labelstr, overwrite = False)
make_if_dont_exist(output_dir, overwrite = False)


train_folder = os.path.join(base_nnunet_dir, "train_images/")
test_folder = os.path.join(base_nnunet_dir, "test_images/")
label_folder = os.path.join(base_nnunet_dir, "masks/")

make_if_dont_exist(train_folder, overwrite = False)
make_if_dont_exist(test_folder, overwrite = False)
make_if_dont_exist(label_folder, overwrite = False)


os.chdir('/kaggle/working/')
!ls


### copy pretrained models
# pretrained_model_dir_new = "/kaggle/working/input/nnunet-pretrained-brain-tumor-segmentation-network"
# !cp -rf ../input/nnunet-pretrained-brain-tumor-segmentation-network/* input/nnunet-pretrained-brain-tumor-segmentation-network/nnUNet


#### inference by https://github.com/Borda/kaggle_image-segm/blob/main/kaggle_imsegm/mask.py
from typing import Dict
def rle_decode(mask_rle: str, img: np.ndarray = None, img_shape: tuple = None, label: int = 1) -> np.ndarray:
    """Create a single label mask for Run-length encoding.
    >>> mask = rle_decode("3 2 11 5 23 3 35 1", img_shape=(8, 10))
    >>> mask = rle_decode("55 3 66 2 77 1", img=mask, label=2)
    >>> mask = rle_decode("26 3 36 2", img=mask, label=3)
    >>> mask
    array([[0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 1, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]], dtype=uint16)
    >>> from pprint import pprint
    >>> pprint(rle_encode(mask))
    {1: '3 2 11 5 23 3 35 1', 2: '55 3 66 2 77 1', 3: '26 3 36 2'}
    """
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths

    if img is None:
        img = np.zeros((np.product(img_shape),), dtype=np.uint16)
    else:
        img_shape = img.shape
        img = img.flatten()
    for begin, end in zip(starts, ends):
        img[begin:end] = label
    return img.reshape(img_shape)


def rle_encode(mask: np.ndarray, label_bg: int = 0) -> Dict[int, str]:
    """Encode mask to Run-length encoding.
    Inspiration took from: https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66
    >>> from pprint import pprint
    >>> mask = np.array([[0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ...                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0],
    ...                  [0, 0, 0, 0, 0, 1, 3, 3, 0, 0],])
    >>> pprint(rle_encode(mask))
    {1: '1 5 13 3 25 1', 2: '16 3', 3: '26 2'}
    """
    vec = mask.flatten()
    nb = len(vec)
    where = np.flatnonzero
    starts = np.r_[0, where(~np.isclose(vec[1:], vec[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, nb])
    values = vec[starts]
    assert len(starts) == len(lengths) == len(values)
    rle = {}
    for start, length, val in zip(starts, lengths, values):
        if val == label_bg:
            continue
        rle[val] = rle.get(val, []) + [str(start), length]
    # post-processing
    rle = {lb: " ".join(map(str, id_lens)) for lb, id_lens in rle.items()}
    return rle

def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, f"*.png")))
    imgs = [np.array(Image.open(p)).tolist() for p in imgs]
    # print([np.max(im) for im in imgs])
    vol = np.array(imgs)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    vol = (vol - v_min) / (v_max - v_min)
    vol = (vol * 255).astype(np.uint8)
    return vol

def create_organs_segm(df_vol, vol_shape):
    df_vol = df_vol.replace(np.nan, '')
    segm = np.zeros(vol_shape, dtype=np.uint8)
    lbs = sorted(df_vol["class"].unique())
#     print(f'lbs is {lbs}')
    for idx_, dfg in df_vol.groupby("Slice"):
        idx = int(idx_) - 1
        mask = segm[idx, :, :]
        for _, (lb, rle) in dfg[["class", "segmentation"]].iterrows():
            lb = lbs.index(lb) + 1
            if not rle:
                continue
            mask = rle_decode(rle, img=mask, label=lb)
        segm[idx, :, :] = mask
        # plt.figure(); plt.imshow(mask)
    return segm


root_dir = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
train_dir = f"{root_dir}/train"
test_dir = f"{root_dir}/test"
train_csv_path = f"{root_dir}/train.csv"
sample_csv_path = f"{root_dir}/sample_submission.csv"


df_train = pd.read_csv(train_csv_path)
df_train


df_test = pd.read_csv(sample_csv_path)
df_test


def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    img_dir = os.path.join(root_dir, "train",
                           f"case{case}", f"case{case}_day{day}", "scans")
    imgs = glob.glob(os.path.join(img_dir, f"slice_{slice_id}_*.png"))
    assert len(imgs) == 1
    img_path = imgs[0].replace(root_dir + "/", "")
    img = os.path.basename(img_path)
    # slice_0001_266_266_1.50_1.50.png
    im_fields = img.split("_")
    return {
        "Case": int(case),
        "Day": int(day),
        "Slice": slice_id,
        "image": img,
        "image_path": img_path, 
        "height": int(im_fields[3]),
        "width": int(im_fields[2]),
    }


df_train[['Case','Day','Slice', 'image', 'image_path', 'height', 'width']] = \
    df_train['id'].apply(lambda x: pd.Series(extract_details(x)))
display(df_train.head())


train_scans_dir = []
for dirname, _, filenames in os.walk(f'{root_dir}/train'):
    if dirname.endswith('scans'):
        train_scans_dir.append(dirname)
        
print(f'Found {len(train_scans_dir)} scans directories')

debug = False ###just select 10 samples for training, please set debug=False for using all train data
for index, s_dir in tqdm(enumerate(train_scans_dir),total=len(train_scans_dir)):
    if debug and index == 10:
        break
    image_id = s_dir.split('/')[-2]
    case_str, day_str= image_id.split('_')
    case = int(case_str.split('case')[-1])
    day = int(day_str.split('day')[-1])

    IMAGE_FOLDER = os.path.join(root_dir, "train", f"case{case}", f"case{case}_day{day}", "scans")
    vol = load_image_volume(img_dir=IMAGE_FOLDER)
            
    ### convert np to nibabel reference https://gist.github.com/tonyreina/64ac5703251b87118cf5d2886169fd5a
    img = nib.Nifti1Image(vol, np.eye(4))  # Save axis for data (just identity)
    img.header.get_xyzt_units()
    img.to_filename(f'{train_folder}/{image_id}.nii.gz')  # Save as NiBabel file
    
    df_ = df_train[(df_train["Case"] == case) & (df_train["Day"] == day)]
    segm = create_organs_segm(df_vol=df_, vol_shape=vol.shape)
    
    mask = nib.Nifti1Image(segm, np.eye(4))  # Save axis for data (just identity)
    mask.header.get_xyzt_units()
    mask.to_filename(f'{label_folder}/{image_id}.nii.gz')  # Save as NiBabel file


#### https://www.kaggle.com/code/outwrest/create-gifs-of-medical-3d-images
scans_dir = []

if df_test.shape[0] == 0: ### select data from train
    for dirname, _, filenames in os.walk(f'{root_dir}/train'):
        if dirname.endswith('scans'):
            scans_dir.append(dirname)
else:
    for dirname, _, filenames in os.walk(f'{root_dir}/test'):
        if dirname.endswith('scans'):
            scans_dir.append(dirname)


if df_test.shape[0] == 0:
    for index, s_dir in tqdm(enumerate(scans_dir),total=len(scans_dir)):
        image_id = s_dir.split('/')[-2]
        case_str, day_str= image_id.split('_')
        case = int(case_str.split('case')[-1])
        day = int(day_str.split('day')[-1])
        if index < 5:
            IMAGE_FOLDER = os.path.join(root_dir, "train", f"case{case}", f"case{case}_day{day}", "scans")
            vol = load_image_volume(img_dir=IMAGE_FOLDER)
            
            ### convert np to nibabel reference https://gist.github.com/tonyreina/64ac5703251b87118cf5d2886169fd5a
            img = nib.Nifti1Image(vol, np.eye(4))  # Save axis for data (just identity)
            img.header.get_xyzt_units()
            img.to_filename(f'{test_folder}/{image_id}.nii.gz')  # Save as NiBabel file
            
else:
    for index, s_dir in tqdm(enumerate(scans_dir),total=len(scans_dir)):
        image_id = s_dir.split('/')[-2]
        case_str, day_str= image_id.split('_')
        case = int(case_str.split('case')[-1])
        day = int(day_str.split('day')[-1])
        
        IMAGE_FOLDER = os.path.join(root_dir, "test", f"case{case}", f"case{case}_day{day}", "scans")
        vol = load_image_volume(img_dir=IMAGE_FOLDER)
        
        ### convert np to nibabel reference https://gist.github.com/tonyreina/64ac5703251b87118cf5d2886169fd5a
        img = nib.Nifti1Image(vol, np.eye(4))  # Save axis for data (just identity)
        img.header.get_xyzt_units()
        img.to_filename(f'{test_folder}/{image_id}.nii.gz')  # Save as NiBabel file


!ls


!pip install --upgrade batchgenerators


import sys
sys.path.append('/kaggle/working/nnUNet_dep_libs/batchgenerators-0.23')
sys.path.append('/kaggle/working/nnUNet/')
# from collections import OrderedDict
from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import shutil


train_patient_names = []
test_patient_names = []
train_patients = subfiles(train_folder, join=False, suffix = 'nii.gz')
test_patients = subfiles(test_folder, join=False, suffix = 'nii.gz')

print(train_patients[0])
print(len(train_patients))
print(test_patients[0])
print(len(test_patients))

for index,patient_name in tqdm(enumerate(train_patients),total=len(train_patients)):
    pex_name = patient_name.split('.')[0]
    image_file = join(train_folder,patient_name)
    label_file = join(label_folder,patient_name)
    
    shutil.copy(image_file, join(imagestr, f'{pex_name}_0000.nii.gz'))
    shutil.copy(label_file, join(labelstr, patient_name))
    
for index,patient_name in tqdm(enumerate(test_patients),total=len(test_patients)):
    pex_name = patient_name.split('.')[0]
    image_file = join(test_folder,patient_name)
    
    shutil.copy(image_file, join(imagests, f'{pex_name}_0000.nii.gz'))


# generate_dataset_json(join(task_folder_name, 'dataset.json'),
#                           imagestr,
#                           imagests,
#                           ('CT',),
#                           {
#                               0: 'background',
#                               1: "large_bowel",
#                               2: "small_bowel",
#                               3: "stomach",
#                           },
#                           task_name,
#                           license='see challenge website',
#                           dataset_description='kaggle-uw-madison-gi-tract-image-segmentation',
#                           dataset_reference='https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation',
#                           dataset_release='0')


len(imagestr)


imagestr


generate_dataset_json(
    output_folder=task_folder_name,          # <-- give the FOLDER, not “…/dataset.json”
    channel_names={0: 'CT'},                 # one channel (index 0) called “CT”
    labels={
        'background': 0,
        'large_bowel': 1,
        'small_bowel': 2,
        'stomach':     3,
    },
    num_training_cases= len(glob.glob(os.path.join(imagestr, "*.nii.gz"))),
    file_ending='.nii.gz',                      # or '.nii.gz' etc. – must match your files
    citation='see challenge website',
    dataset_name=task_name,
    reference='https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation',
    release='0',
    description='kaggle‑uw‑madison‑gi‑tract‑image‑segmentation'
)


os.chdir('/kaggle/working/nnUNet/')
!ls


### set environment veriables
os.environ['nnUNet_raw'] = raw_data_base_dir
os.environ['nnUNet_preprocessed'] = preprocessed_dir
os.environ['nnUNet_results'] = trained_models_dir


!pip install nnunetv2


!nnUNetv2_plan_and_preprocess -d 102 --verify_dataset_integrity


!pip uninstall torch torchvision torchaudio -y
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!nvidia-smi
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should return 12.6
print(torch.cuda.get_device_name(0))  # Should return "Tesla T4"


os.environ['NNUNETv2_DISABLE_TORCH_COMPILE'] = '1'


!nnUNetv2_train 102 2d 0 -tr nnUNetTrainer_20epochs --npz




