# import os
# import pydicom
# from tqdm import tqdm  # progress bar

# dicom_dir = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/"
# sizes = set()

# # loop with progress bar
# for file in tqdm(os.listdir(dicom_dir), desc="Reading DICOM files"):
#     file_path = os.path.join(dicom_dir, file)
#     try:
#         dcm = pydicom.dcmread(file_path)
#         sizes.add((dcm.Columns, dcm.Rows))  # (width, height)
#     except:
#         pass  # skip non-DICOM files

# print("Unique image sizes:", sizes)



import pandas as pd


csv_path = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
df = pd.read_csv(csv_path)


disease_counts = df["class_name"].value_counts()

print("Number of images per disease:")
print(disease_counts)



import pandas as pd


csv_path = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
df = pd.read_csv(csv_path)

keep_classes = ["Aortic enlargement", "Cardiomegaly", "Lung Opacity", "Pleural effusion"]


filtered_df = df[df["class_name"].isin(keep_classes)]


print(filtered_df["class_name"].value_counts())

filtered_df.to_csv("filtered_train.csv", index=False)




df = pd.read_csv("/kaggle/working/filtered_train.csv")
disease_counts = df["class_name"].value_counts()

print("Number of images per disease:")
print(disease_counts)



import os
import pandas as pd
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm


csv_file = "/kaggle/working/filtered_train.csv"
dicom_folder = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train"
output_dir = "/kaggle/working/yolo_dataset"


os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)


df = pd.read_csv(csv_file)


keep_classes = ["Aortic enlargement", "Cardiomegaly", "Lung Opacity", "Pleural effusion"]
df = df[df["class_name"].isin(keep_classes)]


class_map = {cls: i for i, cls in enumerate(keep_classes)}
print("Class mapping:", class_map)

# === Train/Val Split ===
image_ids = df["image_id"].unique()
train_ids, val_ids = train_test_split(image_ids, test_size=0.1, random_state=42)


for img_id in tqdm(image_ids, desc="Converting DICOM to PNG + labels"):
    dcm_path = os.path.join(dicom_folder, f"{img_id}.dicom")  
    if not os.path.exists(dcm_path):
        continue

   
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array

    # Normalize â†’ uint8
    img = cv2.convertScaleAbs(img, alpha=(255.0 / img.max()))

    
    if img_id in train_ids:
        img_out = f"{output_dir}/images/train/{img_id}.png"
        label_out = f"{output_dir}/labels/train/{img_id}.txt"
    else:
        img_out = f"{output_dir}/images/val/{img_id}.png"
        label_out = f"{output_dir}/labels/val/{img_id}.txt"

    
    cv2.imwrite(img_out, img)

   
    rows = df[df["image_id"] == img_id]
    h, w = img.shape
    with open(label_out, "w") as f:
        for _, row in rows.iterrows():
            cls_id = class_map[row["class_name"]]
            # YOLO bbox: x_center, y_center, width, height (normalized)
            x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]
            x_center = (x_min + x_max) / 2 / w
            y_center = (y_min + y_max) / 2 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            f.write(f"{cls_id} {x_center} {y_center} {bw} {bh}\n")

print("âœ… Conversion completed! PNG + YOLO labels are ready.")


def count_images_per_class(subset_ids, subset_name):
    subset_df = df[df["image_id"].isin(subset_ids)]
    counts = subset_df.groupby("class_name")["image_id"].nunique()
    print(f"\nğŸ“Š {subset_name} set class distribution (unique images):")
    print(counts)

count_images_per_class(train_ids, "Train")
count_images_per_class(val_ids, "Val")


assert set(train_ids).isdisjoint(set(val_ids)), "âš ï¸� Data leakage detected!"
print("\nâœ… No data leakage: train/val sets are completely separate.")



!pip install ultralytics


import yaml

classes = ["Aortic enlargement", "Cardiomegaly", "Lung Opacity", "Pleural effusion"]

data_yaml = {
    'train': '/kaggle/working/yolo_dataset/images/train',
    'val': '/kaggle/working/yolo_dataset/images/val',
    'nc': len(classes),
    'names': classes
}

with open("/kaggle/working/chest_xray.yaml", 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("âœ… chest_xray.yaml created!")



from ultralytics import YOLO
model = YOLO("/kaggle/input/yolooo/yolo11m.pt")   


# 4. Train the model
model.train(
    data="chest_xray.yaml",  
    epochs=50,               
    imgsz=640,              
    batch=16,                
    device=0                 
)

