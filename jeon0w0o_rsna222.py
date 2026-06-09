from pathlib import Path
import os
import shutil
import cv2
import pydicom
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ê²½ë¡œ ì„¤ì •
INPUT_PATH = Path("/kaggle/input/rsna-pneumonia-detection-challenge")
DCM_DIR = INPUT_PATH / "stage_2_train_images"
TEST_DCM_DIR = INPUT_PATH / "stage_2_test_images"
LABEL_CSV = INPUT_PATH / "stage_2_train_labels.csv"
CLASS_CSV = INPUT_PATH / "stage_2_detailed_class_info.csv"

# ì¶œë ¥ ê²½ë¡œ
YOLO_IMG_DIR = Path("kaggle/working/yolo/images")
TRAIN_IMG_DIR = YOLO_IMG_DIR / "train"
VAL_IMG_DIR = YOLO_IMG_DIR / "val"
TEST_IMG_DIR = YOLO_IMG_DIR / "test"
SAVE_PATH = Path("kaggle/working")

TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
JPEG_QUALITY = 95

print("complete import")


# 1. ì¤‘ë³µ ì œê±° í›„ ë³‘í•©
labels_df = pd.read_csv(LABEL_CSV).drop_duplicates()
class_df = pd.read_csv(CLASS_CSV).drop_duplicates(subset="patientId")
merged_df = labels_df.merge(class_df, on="patientId", how="left")

print("complete merge")



# 2. ë©”íƒ€ë�°ì�´í„° ì¶”ì¶œ í•¨ìˆ˜
def extract_dicom_meta(dcm_path):
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return {
            "PatientAge": getattr(dcm, "PatientAge", ""),
            "BodyPartExamined": getattr(dcm, "BodyPartExamined", ""),
            "ViewPosition": getattr(dcm, "ViewPosition", ""),
            "PatientSex": getattr(dcm, "PatientSex", "")
        }
    except:
        return {
            "PatientAge": "", "BodyPartExamined": "", "ViewPosition": "", "PatientSex": ""
        }

# 3. Train ë©”íƒ€ë�°ì�´í„° ì¶”ì¶œ
train_meta_list = []
for pid in tqdm(merged_df["patientId"].unique(), desc="Train ë©”íƒ€ë�°ì�´í„° ì¶”ì¶œ"):
    dcm_path = DCM_DIR / f"{pid}.dcm"
    if not dcm_path.exists():
        continue
    meta = extract_dicom_meta(dcm_path)
    meta["patientId"] = pid
    train_meta_list.append(meta)

train_meta_df = pd.DataFrame(train_meta_list)
merged_df = merged_df.merge(train_meta_df, on="patientId", how="left")

merged_df.to_csv(SAVE_PATH / "merged_all.csv", index=False)
print("Train ë³‘í•© + ë©”íƒ€ ì €ì�¥ ì™„ë£Œ")

# ğŸ“Œ 3. Test ë©”íƒ€ë�°ì�´í„° ì¶”ì¶œ
test_meta_list = []
for pid in tqdm([f.stem for f in TEST_DCM_DIR.glob("*.dcm")], desc="Test ë©”íƒ€ë�°ì�´í„° ì¶”ì¶œ"):
    dcm_path = TEST_DCM_DIR / f"{pid}.dcm"
    if not dcm_path.exists():
        continue
    meta = extract_dicom_meta(dcm_path)
    meta["patientId"] = pid
    test_meta_list.append(meta)

test_meta_df = pd.DataFrame(test_meta_list)
test_meta_df.to_csv(SAVE_PATH / "test_metadata.csv", index=False)
print("âœ… Test ë©”íƒ€ ì €ì�¥ ì™„ë£Œ")



# 4. train/val ë¶„í• 
merged_df = pd.read_csv(SAVE_PATH / "merged_all.csv")

unique_ids = merged_df["patientId"].unique()

# patientId ë‹¨ìœ„ë¡œ 8:2 ë¶„í• 
train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

train_df = merged_df[merged_df["patientId"].isin(train_ids)].copy()
val_df = merged_df[merged_df["patientId"].isin(val_ids)].copy()

train_df.to_csv(SAVE_PATH / "merged_train.csv", index=False)
val_df.to_csv(SAVE_PATH / "merged_val.csv", index=False)



# 5. DICOM â†’ JPG ë³€í™˜ ë°� ì €ì�¥
def convert_dicom_to_jpg(patient_ids, dicom_dir, save_dir):
    for pid in tqdm(patient_ids, desc=f"JPG ë³€í™˜ â†’ {save_dir.name}"):
        dcm_path = dicom_dir / f"{pid}.dcm"
        jpg_path = save_dir / f"{pid}.jpg"
        if not dcm_path.exists():
            continue
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array.astype("float32")
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(jpg_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        except Exception as e:
            print(f"âš ï¸� {pid} ë³€í™˜ ì‹¤íŒ¨: {e}")

# ë³€í™˜ ì‹¤í–‰
convert_dicom_to_jpg(train_df["patientId"], DCM_DIR, TRAIN_IMG_DIR)
convert_dicom_to_jpg(val_df["patientId"], DCM_DIR, VAL_IMG_DIR)
convert_dicom_to_jpg(test_df["patientId"], TEST_DCM_DIR, TEST_IMG_DIR)

print(f" ë³€í™˜ë�œ train ì�´ë¯¸ì§€ ìˆ˜: {len(list(TRAIN_IMG_DIR.glob('*.jpg')))}")
print(f" ë³€í™˜ë�œ val ì�´ë¯¸ì§€ ìˆ˜: {len(list(VAL_IMG_DIR.glob('*.jpg')))}")
print(f" ë³€í™˜ë�œ val ì�´ë¯¸ì§€ ìˆ˜: {len(list(TEST_IMG_DIR.glob('*.jpg')))}")


# 6. ë�¼ë²¨ ìƒ�ì„±
def create_yolo_labels(df, label_dir, img_size=640):
    label_dir.mkdir(parents=True, exist_ok=True)

    for pid, group in tqdm(df.groupby("patientId"), desc=f"{label_dir.name} ë�¼ë²¨ ìƒ�ì„±"):
        lines = []
        for _, row in group.iterrows():
            if row["Target"] == 1:
                x = row["x"]
                y = row["y"]
                w = row["width"]
                h = row["height"]

                # ì¤‘ì‹¬ì � ê³„ì‚° + ì •ê·œí™”
                x_center = (x + w / 2) / img_size
                y_center = (y + h / 2) / img_size
                w_norm = w / img_size
                h_norm = h / img_size

                # class_idëŠ” pneumonia 1ê°œ â†’ 0ìœ¼ë¡œ ê³ ì •
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # íŒŒì�¼ ì €ì�¥
        label_path = label_dir / f"{pid}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            
TRAIN_LABEL_DIR = SAVE_PATH / "yolo" / "labels" / "train"
VAL_LABEL_DIR = SAVE_PATH / "yolo" / "labels" / "val"
TEST_LABEL_DIR = SAVE_PATH / "yolo" / "labels" / "test"

create_yolo_labels(train_df, TRAIN_LABEL_DIR)
create_yolo_labels(val_df, VAL_LABEL_DIR)
create_yolo_labels(test_meta_df, TEST_LABEL_DIR)

print(" YOLO ë�¼ë²¨ ìƒ�ì„± ì™„ë£Œ")



# 7. yaml íŒŒì�¼ ìƒ�ì„±
import yaml

DATA_YAML_PATH = SAVE_PATH / "yolo" / "data.yaml"

data_yaml = {
    "path": str((SAVE_PATH / "yolo").resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": ["pneumonia"]
}

# ì €ì�¥
with open(DATA_YAML_PATH, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"âœ… data.yaml ìƒ�ì„± ì™„ë£Œ: {DATA_YAML_PATH}")


