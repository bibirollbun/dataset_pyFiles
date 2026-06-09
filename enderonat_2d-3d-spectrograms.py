import os
import pandas as pd
import numpy as np


# Eldeki tüm veriyi ayrıca import edelim

BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
eeg_path = BASE_PATH+"/"+"train_eegs"
import pandas as pd
pd.read_parquet(eeg_path+"/"+os.listdir(eeg_path)[0])

csv = pd.read_csv(BASE_PATH+"/train.csv")
unique_eeg_ids_df = csv.drop_duplicates(subset='eeg_id')
unique_eeg_ids_df = unique_eeg_ids_df[['eeg_id', 'expert_consensus']]
unique_values = unique_eeg_ids_df['expert_consensus'].unique()
print(unique_values)
labels = {0:"Seizure",1:"GPD",2:"LRDA",3:"LPD",4:"GRDA",5:"Other"}
# Invert the labels dictionary to map string labels to their numeric values
label_map = {v: k for k, v in labels.items()}

# Replace the string values in the expert_consensus column with their numeric values
unique_eeg_ids_df['expert_consensus'] = unique_eeg_ids_df['expert_consensus'].map(label_map)
unique_eeg_ids_df


import os
import shutil
from tqdm import tqdm

# Dictionary eşlemesi
label_map = {
    0: "Seizure",
    1: "GPD",
    2: "LRDA",
    3: "LPD",
    4: "GRDA",
    5: "Other"
}

# Kaynak klasör
source_dir = "/kaggle/input/spectrograms-dataset/spectrograms"
# Hedef klasör
target_base = "2D"

# Eğer hedef klasör yoksa oluştur
os.makedirs(target_base, exist_ok=True)

# Her satır için döngü
for _, row in tqdm(unique_eeg_ids_df.iterrows(), total=len(unique_eeg_ids_df)):
    eeg_id = row["eeg_id"]
    label_num = int(row["expert_consensus"])
    label_name = label_map.get(label_num, "Unknown")
    
    src_path = os.path.join(source_dir, f"{eeg_id}.png")
    dest_dir = os.path.join(target_base, label_name)
    dest_path = os.path.join(dest_dir, f"{eeg_id}.png")
    
    # Hedef klasör yoksa oluştur
    os.makedirs(dest_dir, exist_ok=True)
    
    # Dosya varsa taşı
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
    else:
        print(f"Uyarı: {src_path} bulunamadı.")



import os
from PIL import Image
from tqdm import tqdm

# Giriş (2D) ve çıkış (3D) klasörleri
input_base = "2D"
output_base = "3D"

# Sınıflar
classes = ["Seizure", "GPD", "LRDA", "LPD", "GRDA", "Other"]

# Çıkış klasörleri oluştur
for cls in classes:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)

for cls in classes:
    class_dir = os.path.join(input_base, cls)
    if not os.path.exists(class_dir):
        continue

    # Her görseli işle
    for file in tqdm(os.listdir(class_dir), desc=f"Processing {cls}"):
        if not file.endswith(".png"):
            continue

        eeg_id = os.path.splitext(file)[0]
        img_path = os.path.join(class_dir, file)

        try:
            img = Image.open(img_path)
            width, height = img.size  # 1920x1440 bekleniyor

            # Her parça 960x720 olacak
            w2, h2 = width // 2, height // 2

            # Dört parçayı kırp
            crops = {
                "LT": img.crop((0, 0, w2, h2)),             # Sol üst
                "RT": img.crop((w2, 0, width, h2)),         # Sağ üst
                "LB": img.crop((0, h2, w2, height)),        # Sol alt
                "RB": img.crop((w2, h2, width, height))     # Sağ alt
            }

            # Her EEG için klasör oluştur
            eeg_dir = os.path.join(output_base, cls, eeg_id)
            os.makedirs(eeg_dir, exist_ok=True)

            # Parçaları kaydet
            for pos, crop in crops.items():
                out_path = os.path.join(eeg_dir, f"{eeg_id}_{pos}.png")
                crop.save(out_path)

        except Exception as e:
            print(f"Hata: {file} işlenemedi -> {e}")


