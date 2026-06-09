# Lasts
import shutil
import os
from tqdm import tqdm


source_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations'
folders = os.listdir(source_path)[120:]
dest_dir = '/kaggle/working/segmentations/'
os.makedirs(dest_dir, exist_ok=True)


for folder in tqdm(folders):
    src = os.path.join(source_path, folder)
    dst = os.path.join(dest_dir, folder)
    shutil.copytree(src, dst, dirs_exist_ok=True)

