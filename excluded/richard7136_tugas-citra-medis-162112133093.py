!pip install fpdf


import os
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
from pydicom.errors import InvalidDicomError


import os
import random

# Tentukan path ke folder yang berisi folder pasien
base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'

# Mengambil semua folder pasien
all_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

# Memilih 3 folder secara acak
selected_folders = random.sample(all_folders, 3)

# Menampilkan 3 folder yang dipilih secara acak
print("Selected 3 random patient folders:")
print(selected_folders)



import pydicom
import os
import random

# Fungsi untuk menampilkan metadata DICOM
def display_dicom_metadata(dicom_base_path, selected_folders):
    # Proses setiap folder pasien yang terpilih
    for folder in selected_folders:
        dicom_folder = os.path.join(dicom_base_path, folder)
        
        # Mengambil file DICOM dari folder pasien
        dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
        
        # Pilih satu file DICOM secara acak dari folder pasien
        dicom_file = random.choice(dicom_files)
        dicom_path = os.path.join(dicom_folder, dicom_file)
        
        # Mengecek apakah path tersebut adalah file DICOM
        if os.path.isfile(dicom_path):
            dataset = pydicom.dcmread(dicom_path)
            
            # Menampilkan metadata DICOM
            print(f"Metadata for DICOM file: {dicom_file}")
            print(f"Patient ID: {dataset.PatientID}")
            print(f"Study ID: {dataset.StudyID}")
            print(f"Modality: {dataset.Modality}")
            print(f"Slice Thickness: {dataset.SliceThickness}")
            print(f"Rows: {dataset.Rows}, Columns: {dataset.Columns}")
            print("Image Position (Patient):", dataset.ImagePositionPatient)
            print("Image Orientation (Patient):", dataset.ImageOrientationPatient)
            print("-" * 40)

# Menampilkan metadata DICOM
display_dicom_metadata(dicom_base_path, selected_folders)



import pydicom
import os
import random
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar DICOM
def display_dicom_images(dicom_base_path, selected_folders):
    # Proses setiap folder pasien yang terpilih
    for folder in selected_folders:
        dicom_folder = os.path.join(dicom_base_path, folder)
        
        # Mengambil file DICOM dari folder pasien
        dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
        
        # Pilih satu file DICOM secara acak dari folder pasien
        dicom_file = random.choice(dicom_files)
        dicom_path = os.path.join(dicom_folder, dicom_file)
        
        # Mengecek apakah path tersebut adalah file DICOM
        if os.path.isfile(dicom_path):
            dataset = pydicom.dcmread(dicom_path)
            
            # Mengecek apakah file DICOM memiliki data pixel
            if 'PixelData' in dataset:
                pixel_data = dataset.pixel_array
                
                # Menampilkan gambar DICOM
                plt.figure(figsize=(6, 6))
                plt.imshow(pixel_data, cmap='gray')
                plt.title(f"Patient Folder: {folder} - DICOM File: {dicom_file}")
                plt.axis('off')  # Menonaktifkan axis
                plt.show()

# Menampilkan gambar DICOM
display_dicom_images(dicom_base_path, selected_folders)


