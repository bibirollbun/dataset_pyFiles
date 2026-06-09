import pandas as pd
import os

#Train csv to dataframe
df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')

#List all Modalities
modalities = list(df.Modality.unique())

#Root folder for data
DATA_ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/"

#Dictionary
fns = {}

#Get one foldername for each type
for modality in modalities:
    series_id = df[df.Modality==modality].iloc[0].SeriesInstanceUID
    fns[modality] = os.path.join(DATA_ROOT,series_id)
fns


!pip install python-gdcm --no-index --find-links=file:///kaggle/input/read-dicom-set -q
!pip install dicomsdl --no-index --find-links=file:///kaggle/input/read-dicom-set -q
!pip install /kaggle/input/synthstriphelper/freesurfer/offline_packages/surfa-0.6.1-cp311-cp311-linux_x86_64.whl --no-index -q


!python -m pip install dcm2niix -q


import subprocess
from pathlib import Path
import shutil

def run_cmd(cmd):
    """Run a shell command and return (success, output)."""
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True, output.decode()
    except subprocess.CalledProcessError as e:
        return False, e.output.decode()

def convert_dicom_folder(folder_path, nifti_outdir, tmp_dir, verbose=True):
    folder = Path(folder_path)
    series_id = folder.name  # get last part of the folder path

    if verbose:
        print(f"Trying dcm2niix on {folder}...")
    # Run dcm2niix on the original folder
    dcm2niix_cmd = [
        'dcm2niix',
        '-5', '-b', 'y', '-i', 'y', '-z', 'y',
        '-f', '%f',            # filename prefix based on folder name
        '-o', str(nifti_outdir),
        str(folder)
    ]
    success, output = run_cmd(dcm2niix_cmd)

    if success:
        if verbose:
            print(f"dcm2niix succeeded on {folder}")
        return True
    else:
        if verbose:
            print(f"dcm2niix failed on {folder}, error:")
            print(output)
            print("Trying gdcmconv to decompress and rerun dcm2niix...")

        # Use gdcmconv --raw to decompress each DICOM in folder to tmp_folder
        tmp_folder = Path(tmp_dir) / folder.name
        tmp_folder.mkdir(parents=True, exist_ok=True)
        
        for dicom_file in folder.glob('*.dcm'):
            out_file = tmp_folder / dicom_file.name
            gdcmconv_cmd = ['gdcmconv', '--raw', str(dicom_file), str(out_file)]
            success_gdcm, output_gdcm = run_cmd(gdcmconv_cmd)
            if not success_gdcm:
                print(f"gdcmconv failed on {dicom_file}, error:")
                print(output_gdcm)
                return False

        # Run dcm2niix again on decompressed tmp folder
        dcm2niix_cmd_tmp = [
            'dcm2niix',
            '-5', '-b', 'y', '-i', 'y', '-z', 'y',
            '-f', '%f',        
            '-o', str(nifti_outdir),
            str(tmp_folder)
        ]
        success_tmp, output_tmp = run_cmd(dcm2niix_cmd_tmp)
        #cleanup
        # if tmp_folder.exists():
        #     shutil.rmtree(tmp_folder)
            
        if success_tmp:
            if verbose:
                print(f"dcm2niix succeeded on decompressed files for {folder}")
            return True
        else:
            if verbose:
                print(f"dcm2niix failed on decompressed files for {folder}, error:")
                print(output_tmp)
            return False

def batch_convert_folders(folder_list, nifti_outdir, tmp_dir='/kaggle/temp/converted_dicoms'):
    nifti_outdir = Path(nifti_outdir)
    nifti_outdir.mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    for folder in folder_list:
        success = convert_dicom_folder(folder, nifti_outdir, tmp_dir)
        if not success:
            print(f"Conversion failed for folder: {folder}")
        print("="*60)



!ls /kaggle/temp/converted_dicoms


import subprocess
import os

MODEL_PATH = "/kaggle/input/synthstrip/pytorch/default/1/synthstrip.1.pt"
ROOT = "/kaggle/input/synthstriphelper/freesurfer"

def run_synthstrip(input_path, output_path, model_path=MODEL_PATH, root=ROOT, verbose=True):
    """
    Run the mri_synthstrip command with given paths.
    
    Args:
        input_path (str): Path to the input file.
        output_path (str): Path to the output file.
        model_path (str): Path to the model file.
        root (str): Path for FREESURFER_HOME environment variable.
    """
    env = os.environ.copy()
    env["FREESURFER_HOME"] = root

    cmd = [
        "python",
        os.path.join(root, "mri_synthstrip/mri_synthstrip"),
        "-i", input_path,
        "-o", output_path,
        "--model", model_path
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if verbose:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

def apply_window(img, center, width):
    """Apply window level to image for display purposes."""
    lower = center - width / 2
    upper = center + width / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)  # normalize to [0,1]
    return img

def show_stripped(orig_path, mask_path, window_center=96, window_width=150, dilation=3):

    # Load the original and stripped images
    orig = nib.load(orig_path).get_fdata()
    stripped = nib.load(mask_path).get_fdata()
    
    # Pick a slice (adjust based on orientation)
    slice_idx = orig.shape[2] // 2  # Middle axial slice
    
    orig_slice = apply_window(orig[:, :, slice_idx], window_center, window_width)
    mask_slice = stripped[:, :, slice_idx] > 0

    if dilation > 0:
        #dilate mask
        dilator = np.ones((dilation, dilation))
        mask_slice = binary_dilation(mask_slice,dilator)
    
    # Prepare figure
    plt.figure(figsize=(12, 5))
    
    # Show original
    plt.subplot(1, 3, 1)
    plt.imshow(orig_slice, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Show masked result (brain only)
    plt.subplot(1, 3, 2)
    # Mask the original image to show only brain
    brain_only = np.where(mask_slice, orig_slice, 0)

    plt.imshow(brain_only, cmap='gray')
    plt.title('Stripped (brain only)')
    plt.axis('off')
    
    # Show overlay (brain as transparent over original)
    plt.subplot(1, 3, 3)
    plt.imshow(orig_slice, cmap='gray')
    # Overlay: show where stripped image > 0
    plt.imshow(mask_slice, cmap='autumn', alpha=0.3)  # semi-transparent red
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def process_folder(folder):
    nifti_folder = '/kaggle/working/nifti'
    #convert to nifti
    batch_convert_folders([folder], nifti_folder)

    series_id = folder.split('/')[-1]
    nifti_path = os.path.join(nifti_folder,f'{series_id}.nii.gz')
    mask_path = os.path.join(nifti_folder,f'mask_{series_id}.nii.gz')
    
    #run synthstrip
    print("Running...")
    run_synthstrip(nifti_path, mask_path)
    return nifti_path, mask_path


i = 0
folder = fns[modalities[i]]

nifti_path_1, mask_path_1 = process_folder(folder)

print("Show...")
#Original mask
show_stripped(nifti_path_1, mask_path_1, dilation=0)


#mask after dilation to fill in vessel and then pad 
show_stripped(nifti_path_1, mask_path_1, dilation=15)


i = 1
folder = fns[modalities[i]]

nifti_path_2, mask_path_2 = process_folder(folder)

print("Show...")
show_stripped(nifti_path_2, mask_path_2, dilation=3)


i = 2
folder = fns[modalities[i]]

nifti_path_3, mask_path_3 = process_folder(folder)

print("Show...")
show_stripped(nifti_path_3, mask_path_3, window_center=300, window_width=1200, dilation=3)


i = 3
folder = fns[modalities[i]]

nifti_path_4, mask_path_4 = process_folder(folder)

print("Show...")
show_stripped(nifti_path_4, mask_path_4, window_center=300, window_width=1200, dilation=3)


import json

jsonfn = '/kaggle/working/nifti/1.2.826.0.1.3680043.8.498.10030804647049037739144303822498146901.json'
with open(jsonfn, 'r') as file:
    data = json.load(file)

data







