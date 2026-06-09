!pip install pydicom


import pydicom

# Örnek dosya yolu (kendi verinle değiştir)
dosya_yolu = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/ID_000012eaf.dcm"

# DICOM dosyasını oku
dcm = pydicom.dcmread(dosya_yolu)

# Tüm metadataları yazdır
print(dcm)



import os
import time
import pydicom
import pandas as pd
from tqdm import tqdm

# DICOM dosyalarının bulunduğu ana klasör
dicom_folder = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train"
# Oluşturulacak CSV'nin adı
output_csv = "grouped_by_patient.csv"

start_time = time.time()  # Süre ölçümü başlat

# Hasta ID'lerine göre dosyaları tutacak sözlük
patient_files_map = {}

# Klasördeki tüm .dcm dosyalarını tara
for root, dirs, files in os.walk(dicom_folder):
    for file_name in tqdm(files, desc="DICOM Tarama"):
        if file_name.lower().endswith(".dcm"):
            file_path = os.path.join(root, file_name)
            try:
                # DICOM meta verisini oku (piksellere bakmadan hızlı okumak için stop_before_pixels=True)
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                patient_id = ds.get("PatientID", "Unknown_Patient")

                # Bu hasta ID için ilk defa ekleme yapıyorsak liste oluştur
                if patient_id not in patient_files_map:
                    patient_files_map[patient_id] = []
                # Dosya yolunu ekle
                patient_files_map[patient_id].append(file_path)
            except Exception as e:
                print(f"Hata oluştu ({file_path}): {e}")

# CSV'ye yazmak için satırları oluştur
rows = []
for patient_id, file_paths in patient_files_map.items():
    for fpath in file_paths:
        rows.append({
            "PatientID": patient_id,
            "FilePath": fpath
        })

# DataFrame oluştur ve CSV kaydet
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

end_time = time.time()  # Süre ölçümü bitir
print(f"\nİşlem tamamlandı. Dosya: {output_csv}")
print(f"Toplam süre: {end_time - start_time:.2f} saniye")


