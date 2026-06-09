# kiểm tra GPU
!python - <<'PY'
import torch, sys
print("cuda available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


import os, shutil, subprocess, nibabel as nib
import numpy as np

# -------------------------
# Đường dẫn
# -------------------------
segment_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations"
dicom_base  = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"

NNUNET_RAW_DATA_BASE = "/kaggle/working/nnUNet_raw"
task_id    = "Dataset177_RSNA"
task_folder = os.path.join(NNUNET_RAW_DATA_BASE, task_id)

# -------------------------
# Tạo cấu trúc nnUNet
# -------------------------
os.makedirs(os.path.join(task_folder, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(task_folder, "labelsTr"), exist_ok=True)
os.makedirs(os.path.join(task_folder, "imagesTs"), exist_ok=True)

# -------------------------
# Lấy danh sách series từ segmentation (bỏ _cowseg)
# -------------------------
series_list = []
for f in os.listdir(segment_dir):
    sid = os.path.splitext(f)[0]
    if sid.endswith("_cowseg"):
        sid = sid.replace("_cowseg", "")
    series_list.append(sid)
series_list = list(set(series_list))

print("Tổng số series cần xử lý:", len(series_list))

# -------------------------
# Pipeline convert + copy
# -------------------------
n_case = 0
for i, sid in enumerate(series_list, start=1):
    dicom_path = os.path.join(segment_dir, f"{sid}.nii")
    lbl_path   = os.path.join(segment_dir, f"{sid}_cowseg.nii")

    if not os.path.exists(dicom_path):
        print(f"❌ Không tìm thấy series: {dicom_path}")
        continue

    if not os.path.exists(lbl_path):
        print(f"❌ Thiếu label cho {sid}")
        continue

    n_case += 1
    case_id = f"case{n_case:04d}"
    
    dicom = nib.load(dicom_path)
    dicom_out = nib.Nifti1Image(dicom.get_fdata().astype(np.uint8), dicom.affine)
    dst_dicom = os.path.join(task_folder, "imagesTr", f"{case_id}_0000.nii.gz")
    nib.save(dicom_out, dst_dicom)
    

    seg = nib.load(lbl_path)
    seg_out = nib.Nifti1Image(seg.get_fdata().astype(np.uint8), seg.affine)
    dst_lbl = os.path.join(task_folder, "labelsTr", f"{case_id}.nii.gz")
    nib.save(seg_out, dst_lbl)

print(f"✅ Đã xử lý xong {n_case} case hợp lệ. Lưu tại {task_folder}")



import os
import json

# === CONFIG ===
task_id = "Dataset177_RSNA"
nnunet_raw = "/kaggle/working/nnUNet_raw"
task_folder = os.path.join(nnunet_raw, task_id)

# labels RSNA (13 + background)
labels = {
    "background": 0,
    "Other Posterior Circulation": 1,
    "Basilar Tip": 2,
    "Right Posterior Communicating Artery": 3,
    "Left Posterior Communicating Artery": 4,
    "Right Infraclinoid Internal Carotid Artery": 5,
    "Left Infraclinoid Internal Carotid Artery": 6,
    "Right Supraclinoid Internal Carotid Artery": 7,
    "Left Supraclinoid Internal Carotid Artery": 8,
    "Right Middle Cerebral Artery": 9,
    "Left Middle Cerebral Artery": 10,
    "Anterior Communicating Artery": 11,
    "Right Anterior Cerebral Artery": 12,
    "Left Anterior Cerebral Artery": 13
}

# channel_names: chỉnh sửa nếu có nhiều modality
channel_names = {
    "0": "CT"
}

# === COUNT TRAINING CASES ===
imagesTr = os.path.join(task_folder, "imagesTr")
n_train = len([f for f in os.listdir(imagesTr) if f.endswith(".nii.gz") and "_0000" in f])

print(f"Tìm thấy {n_train} case training trong {imagesTr}")

# === BUILD JSON ===
dataset_dict = {
    "channel_names": channel_names,
    "labels": labels,
    "numTraining": n_train,
    "file_ending": ".nii"
}

# === SAVE JSON ===
json_path = os.path.join(task_folder, "dataset.json")
with open(json_path, "w") as f:
    json.dump(dataset_dict, f, indent=4)

print(f"✅ Saved dataset.json vào {json_path}")





