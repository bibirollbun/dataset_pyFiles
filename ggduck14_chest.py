!nvidia-smi



!pip install ultralytics==8.4.36 --quiet


import ultralytics
print(ultralytics.__version__)  # phải ra 8.4.36


import pandas as pd
import os
import numpy as np
import shutil
import yaml
import matplotlib.pyplot as plt
import random
import cv2

from sklearn import model_selection
from tqdm import tqdm
from glob import glob


size = 1024
TRAIN_LABELS_PATH = './vinbigdata/labels/train'
VAL_LABELS_PATH = './vinbigdata/labels/val'
TRAIN_IMAGES_PATH = './vinbigdata/images/train' #12000
VAL_IMAGES_PATH = './vinbigdata/images/val' #3000
External_DIR = f'../input/vinbigdata-{size}-image-dataset/vinbigdata/train' # 15000
os.makedirs(TRAIN_LABELS_PATH, exist_ok = True)
os.makedirs(VAL_LABELS_PATH, exist_ok = True)
os.makedirs(TRAIN_IMAGES_PATH, exist_ok = True)
os.makedirs(VAL_IMAGES_PATH, exist_ok = True)


original_df = pd.read_csv('../input/vinbigdata-chest-xray-abnormalities-detection/train.csv')
number_of_imageids = len(original_df['image_id'].values)
print(f'Total number of image_ids (train + validation) {number_of_imageids}')

number_of_images = len(os.listdir('../input/vinbigdata-chest-xray-abnormalities-detection/train'))
print(f'Total number of images (train + validation) {number_of_images}')

number_of_labels = len(os.listdir('../input/vinbigdata-yolo-labels-dataset/labels'))
print(f'Total number of labels (train + validation) {number_of_labels}')


df = pd.read_csv('/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv')
number_of_images = len(df['image_id'].values)
print(f'Total number of image ids (train + validation) {number_of_images}')

df = df[df.class_id!=14].reset_index(drop = True)
number_of_images = len(df['image_id'].values)
print(f'Total number of image ids after dropping normal images (train + validation) {number_of_images}')

df.head()


df = df.drop(columns=['class_name', 'rad_id', 'x_min', 'x_max', 'y_min', 'y_max',  'class_id']) # we only need image ids, labels are pre-made
df.head()


df_train, df_valid = model_selection.train_test_split(df, test_size=0.15, random_state=42, shuffle=True)


number_of_images = len(df_train['image_id'].values)
print(f'Total number of training image_ids {number_of_images}')

number_of_images = len(df_valid['image_id'].values)
print(f'Total number of validation image_ids {number_of_images}')



print(f'Total number of training images {len(df_train.image_id.unique())}')
print(f'Total number of validation images {len(df_valid.image_id.unique())}')


def preproccess_data(df, labels_path, images_path):
    for img_id in tqdm(df.image_id.unique()):
        shutil.copy(os.path.join('../input/vinbigdata-yolo-labels-dataset/labels', f"{img_id}"+'.txt'), labels_path)
        shutil.copy(os.path.join(f'/kaggle/input/vinbigdata-{size}-image-dataset/vinbigdata/train', f"{img_id}.png"), images_path)


preproccess_data(df_train, TRAIN_LABELS_PATH, TRAIN_IMAGES_PATH)
preproccess_data(df_valid, VAL_LABELS_PATH, VAL_IMAGES_PATH)


# check that data was preprocessed correctly
print(len(os.listdir(TRAIN_LABELS_PATH)))
print(len(os.listdir(TRAIN_IMAGES_PATH)))

print(len(os.listdir(VAL_LABELS_PATH)))
print(len(os.listdir(VAL_IMAGES_PATH)))


classes = [ 'Aortic enlargement',
            'Atelectasis',
            'Calcification',
            'Cardiomegaly',
            'Consolidation',
            'ILD',
            'Infiltration',
            'Lung Opacity',
            'Nodule/Mass',
            'Other lesion',
            'Pleural effusion',
            'Pleural thickening',
            'Pneumothorax',
            'Pulmonary fibrosis']

data = dict(
    train =  '../vinbigdata/images/train',
    val   =  '../vinbigdata/images/val',
    nc    = 14,
    names = classes
    )

with open('/kaggle/working/vinbigdata.yaml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(os.path.join( os.getcwd() , 'vinbigdata.yaml'), 'r')
print('\nyaml:')
print(f.read())


# import os
# import torch
# from ultralytics import YOLO

# os.environ["WANDB_MODE"] = "dryrun"
# assert torch.cuda.is_available(), "⚠️ GPU NOT AVAILABLE. Check Kaggle Accelerator."


# model = YOLO("yolo11l.pt")

# results = model.train(
#     data='./vinbigdata.yaml',

#     imgsz=1024,
#     batch=8,
#     epochs=120,
#     patience=25,

#     optimizer="AdamW",
#     lr0=0.0015,
#     lrf=0.01,
#     weight_decay=5e-4,
#     cos_lr=True,

#     hsv_h=0.01,
#     hsv_s=0.15,  
#     hsv_v=0.20,
#     degrees=3.0,
#     translate=0.1,
#     scale=0.2,
#     flipud=0.0,
#     fliplr=0.5,
#     mosaic=0.5,
#     mixup=0.1,
#     close_mosaic=20,

#     box=8.0,
#     cls=3.0,
#     dfl=2.0,
#     iou=0.55,

#     device=0,
#     workers=2,
#     amp=True,
#     cache="ram",

#     project="/kaggle/working/runs",
#     name="yolo11l_highprecision_v3",
#     save=True,
#     save_period=1,
#     exist_ok=True,
#     resume=False,
# )


# import os
# import torch
# from ultralytics import YOLO

# os.environ["WANDB_MODE"] = "dryrun"
# assert torch.cuda.is_available(), "⚠️ GPU NOT AVAILABLE. Check Kaggle Accelerator."


# model = YOLO("/kaggle/input/datasets/ggduck14/version21/last.pt")

# results = model.train(
#     data='./vinbigdata.yaml',

#     imgsz=1024,
#     batch=8,
#     epochs=120,
#     patience=25,

#     optimizer="AdamW",
#     lr0=0.0015,
#     lrf=0.01,
#     weight_decay=5e-4,
#     cos_lr=True,

#     hsv_h=0.01,
#     hsv_s=0.15,  
#     hsv_v=0.20,
#     degrees=3.0,
#     translate=0.1,
#     scale=0.2,
#     flipud=0.0,
#     fliplr=0.5,
#     mosaic=0.3,
#     mixup=0.0,
#     close_mosaic=20,

#     box=8.0,
#     cls=3.0,
#     dfl=2.0,
#     iou=0.55,

#     device=0,
#     workers=2,
#     amp=True,
#     cache="ram",

#     project="/kaggle/working/runs",
#     name="yolo11l_highprecision_v3_final",
#     save=True,
#     save_period=1,
#     exist_ok=True,
#     resume=True,
# )


# import os
# import torch
# from ultralytics import YOLO

# os.environ["WANDB_MODE"] = "dryrun"
# assert torch.cuda.is_available(), "⚠️ GPU NOT AVAILABLE. Check Kaggle Accelerator."


# model = YOLO("yolov8l.pt")

# results = model.train(
#     data='./vinbigdata.yaml',

#     imgsz=1024,
#     batch=8,
#     epochs=120,
#     patience=25,

#     optimizer="AdamW",
#     lr0=0.0015,
#     lrf=0.01,
#     weight_decay=5e-4,
#     cos_lr=True,

#     hsv_h=0.01,
#     hsv_s=0.15,  
#     hsv_v=0.20,
#     degrees=3.0,
#     translate=0.1,
#     scale=0.2,
#     flipud=0.0,
#     fliplr=0.5,
#     mosaic=0.3,
#     mixup=0.0,
#     close_mosaic=20,

#     box=8.0,
#     cls=3.0,
#     dfl=2.0,
#     iou=0.55,

#     device=0,
#     workers=2,
#     amp=True,
#     cache="ram",

#     project="/kaggle/working/runs",
#     name="yolo11l_highprecision_v4",
#     save=True,
#     save_period=1,
#     exist_ok=True,
#     resume=True,
# )


# import os
# import torch
# from ultralytics import YOLO

# os.environ["WANDB_MODE"] = "dryrun"
# assert torch.cuda.is_available(), "⚠️ GPU NOT AVAILABLE. Check Kaggle Accelerator."


# model = YOLO("/kaggle/input/datasets/ggduck14/yolov8/lastYolov8.pt")

# results = model.train(
#     data='./vinbigdata.yaml',

#     imgsz=1024,
#     batch=8,
#     epochs=120,
#     patience=25,

#     optimizer="AdamW",
#     lr0=0.0015,
#     lrf=0.01,
#     weight_decay=5e-4,
#     cos_lr=True,

#     hsv_h=0.01,
#     hsv_s=0.15,  
#     hsv_v=0.20,
#     degrees=3.0,
#     translate=0.1,
#     scale=0.2,
#     flipud=0.0,
#     fliplr=0.5,
#     mosaic=0.3,
#     mixup=0.0,
#     close_mosaic=20,

#     box=8.0,
#     cls=3.0,
#     dfl=2.0,
#     iou=0.55,

#     device=0,
#     workers=2,
#     amp=True,
#     cache="ram",

#     project="/kaggle/working/runs",
#     name="yolo11l_highprecision_v4_final",
#     save=True,
#     exist_ok=True,
#     resume=True,
# )


# from ultralytics import YOLO
# model = YOLO("/kaggle/input/version-16/best.pt")

# r = model.val(data="./vinbigdata.yaml", imgsz=640, device="cpu", plots=False, conf=0.325, iou=0.7)
# print(r.box.map50, r.box.map)



# !zip -r /kaggle/working/yolo_all_output.zip /kaggle/working



# from ultralytics import YOLO

# model = YOLO("/kaggle/input/version-16/best.pt")

# for conf in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
#     metrics = model.val(
#         data="/kaggle/working/vinbigdata.yaml",
#         imgsz=1024,
#         split="val",
#         conf=conf,
#         iou=0.5,
#         verbose=False
#     )
#     r = metrics.results_dict
#     print(
#         f"conf={conf:.2f} | "
#         f"P={r['metrics/precision(B)']:.4f} | "
#         f"R={r['metrics/recall(B)']:.4f} | "
#         f"mAP50={r['metrics/mAP50(B)']:.4f} | "
#         f"mAP50-95={r['metrics/mAP50-95(B)']:.4f}"
#     )


# from ultralytics import YOLO

# model = YOLO("/kaggle/input/version-16/best.pt")

# metrics = model.val(
#     data="/kaggle/working/vinbigdata.yaml",
#     imgsz=1024,
#     split="val",
#     conf=0.25,
#     iou=0.5
# )

# print(metrics.results_dict)


# from ultralytics import YOLO

# model = YOLO("/kaggle/input/version-16/best.pt")

# for conf in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
#     metrics = model.val(
#         data="/kaggle/working/vinbigdata.yaml",
#         imgsz=1024,
#         split="val",
#         conf=conf,
#         iou=0.65,
#         verbose=False
#     )
#     r = metrics.results_dict
#     print(
#         r
#     )


# from ultralytics import YOLO

# model = YOLO("/kaggle/input/version-16")

# for conf in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
#     metrics = model.val(
#         data="/kaggle/working/vinbigdata.yaml",
#         imgsz=1024,
#         split="val",
#         conf=conf,
#         iou=0.65,
#         verbose=False
#     )
#     r = metrics.results_dict
#     print(
#         f"conf={conf:.2f} | "
#         f"P={r['metrics/precision(B)']:.4f} | "
#         f"R={r['metrics/recall(B)']:.4f} | "
#         f"mAP50={r['metrics/mAP50(B)']:.4f} | "
#         f"mAP50-95={r['metrics/mAP50-95(B)']:.4f}"
#     )


from ultralytics import YOLO
import pandas as pd

# Load model
model = YOLO("/kaggle/input/version-16/best.pt")

# Run validation
metrics = model.val(
    data="vinbigdata.yaml",
    imgsz=1024,
    split="val",
    conf=0.20,     
    iou=0.55,      
    augment=True,   
    max_det=300,  
    verbose=False
)


# =========================
# 1. OVERALL METRICS
# =========================
print("\n===== OVERALL METRICS YOLOV8 =====")
results = metrics.results_dict

print(f"Precision: {results['metrics/precision(B)']:.3f}")
print(f"Recall:    {results['metrics/recall(B)']:.3f}")
print(f"mAP@50:    {results['metrics/mAP50(B)']:.3f}")
print(f"mAP@50-95: {results['metrics/mAP50-95(B)']:.3f}")

# =========================
# 2. SPECIFICITY TỪ CONFUSION MATRIX
# =========================
# metrics.confusion_matrix.matrix có shape (num_classes + 1, num_classes + 1)
# Hàng = ground truth, Cột = prediction
# Hàng/cột cuối cùng = background (no object)

cm = metrics.confusion_matrix.matrix  # shape: (C+1, C+1)
num_classes = len(model.names)

print("\n===== SPECIFICITY PER-CLASS =====")

specificities = []
rows = []
names = model.names
p = metrics.box.p
r = metrics.box.r
ap50 = metrics.box.ap50
ap = metrics.box.ap

for i in range(num_classes):
    # TP: cm[i, i]
    # FN: tổng hàng i trừ TP (ground truth class i bị predict sai)
    # FP: tổng cột i trừ TP (predict class i nhưng thực ra không phải)
    # TN: tổng tất cả - TP - FP - FN

    TP = cm[i, i]
    FP = cm[:, i].sum() - TP      # cột i, bỏ TP
    FN = cm[i, :].sum() - TP      # hàng i, bỏ TP
    TN = cm.sum() - TP - FP - FN  # còn lại

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    specificities.append(specificity)

    cls_name = names[i]
    precision = float(p[i])
    recall    = float(r[i])
    ap_50     = float(ap50[i])
    ap_5095   = float(ap[i])

    print(f"{cls_name:25s} | P: {precision:.3f} | R: {recall:.3f} "
          f"| Spec: {specificity:.3f} | AP50: {ap_50:.3f} | AP50-95: {ap_5095:.3f}")

    rows.append({
        "class":       cls_name,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "AP50":        ap_50,
        "AP50-95":     ap_5095
    })

# =========================
# 3. MACRO AVERAGE SPECIFICITY
# =========================
macro_specificity = np.mean(specificities)
print(f"\n{'':25s}   Macro-avg Specificity: {macro_specificity:.3f}")

# =========================
# 4. EXPORT DATAFRAME
# =========================
df = pd.DataFrame(rows)
df.loc[len(df)] = {
    "class":       "MEAN",
    "precision":   df["precision"].mean(),
    "recall":      df["recall"].mean(),
    "specificity": macro_specificity,
    "AP50":        df["AP50"].mean(),
    "AP50-95":     df["AP50-95"].mean()
}

print("\n===== SUMMARY TABLE =====")
print(df.to_string(index=False))



from ultralytics import YOLO
import pandas as pd

# Load model
model = YOLO("/kaggle/input/version-16/best.pt")

# Run validation with high-recall settings
metrics = model.val(
    data="vinbigdata.yaml",
    imgsz=1024,
    split="val",
    conf=0.001,      # Hạ xuống mức tối thiểu để lấy trọn PR-curve
    iou=0.65,        # Nới lỏng NMS, cho phép các hộp đè lên nhau nhiều hơn
    augment=True,    # Kích hoạt TTA (Test-Time Augmentation)
    max_det=300,     # (Tùy chọn) Đảm bảo không bị giới hạn số lượng hộp dự đoán
    verbose=False
)



# =========================
# 1. OVERALL METRICS
# =========================
print("\n===== OVERALL METRICS YOLOV8 =====")
results = metrics.results_dict

print(f"Precision: {results['metrics/precision(B)']:.3f}")
print(f"Recall:    {results['metrics/recall(B)']:.3f}")
print(f"mAP@50:    {results['metrics/mAP50(B)']:.3f}")
print(f"mAP@50-95: {results['metrics/mAP50-95(B)']:.3f}")

# =========================
# 2. SPECIFICITY TỪ CONFUSION MATRIX
# =========================
# metrics.confusion_matrix.matrix có shape (num_classes + 1, num_classes + 1)
# Hàng = ground truth, Cột = prediction
# Hàng/cột cuối cùng = background (no object)

cm = metrics.confusion_matrix.matrix  # shape: (C+1, C+1)
num_classes = len(model.names)

print("\n===== SPECIFICITY PER-CLASS =====")

specificities = []
rows = []
names = model.names
p = metrics.box.p
r = metrics.box.r
ap50 = metrics.box.ap50
ap = metrics.box.ap

for i in range(num_classes):
    # TP: cm[i, i]
    # FN: tổng hàng i trừ TP (ground truth class i bị predict sai)
    # FP: tổng cột i trừ TP (predict class i nhưng thực ra không phải)
    # TN: tổng tất cả - TP - FP - FN

    TP = cm[i, i]
    FP = cm[:, i].sum() - TP      # cột i, bỏ TP
    FN = cm[i, :].sum() - TP      # hàng i, bỏ TP
    TN = cm.sum() - TP - FP - FN  # còn lại

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    specificities.append(specificity)

    cls_name = names[i]
    precision = float(p[i])
    recall    = float(r[i])
    ap_50     = float(ap50[i])
    ap_5095   = float(ap[i])

    print(f"{cls_name:25s} | P: {precision:.3f} | R: {recall:.3f} "
          f"| Spec: {specificity:.3f} | AP50: {ap_50:.3f} | AP50-95: {ap_5095:.3f}")

    rows.append({
        "class":       cls_name,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "AP50":        ap_50,
        "AP50-95":     ap_5095
    })

# =========================
# 3. MACRO AVERAGE SPECIFICITY
# =========================
macro_specificity = np.mean(specificities)
print(f"\n{'':25s}   Macro-avg Specificity: {macro_specificity:.3f}")

# =========================
# 4. EXPORT DATAFRAME
# =========================
df = pd.DataFrame(rows)
df.loc[len(df)] = {
    "class":       "MEAN",
    "precision":   df["precision"].mean(),
    "recall":      df["recall"].mean(),
    "specificity": macro_specificity,
    "AP50":        df["AP50"].mean(),
    "AP50-95":     df["AP50-95"].mean()
}

print("\n===== SUMMARY TABLE =====")
print(df.to_string(index=False))



# from ultralytics import YOLO
# import pandas as pd

# # Load model
# model = YOLO("/kaggle/input/datasets/ggduck14/train-yolov8-final/bestYolov8l_final.pt")

# # Run validation
# metrics = model.val(
#     data="vinbigdata.yaml",
#     imgsz=1024,
#     split="val",
#     conf=0.20,
#     iou=0.55,
#     verbose=False
# )

# # =========================
# # 1. OVERALL METRICS
# # =========================
# print("\n===== OVERALL METRICS YOLOV8 =====")
# results = metrics.results_dict

# print(f"Precision: {results['metrics/precision(B)']:.3f}")
# print(f"Recall:    {results['metrics/recall(B)']:.3f}")
# print(f"mAP@50:    {results['metrics/mAP50(B)']:.3f}")
# print(f"mAP@50-95: {results['metrics/mAP50-95(B)']:.3f}")

# # =========================
# # 2. PER-CLASS FULL METRICS
# # =========================
# print("\n===== PER-CLASS METRICS YOLOV8 =====")

# names = model.names
# p = metrics.box.p
# r = metrics.box.r
# ap50 = metrics.box.ap50
# ap = metrics.box.ap

# rows = []

# for i, cls_name in names.items():
#     precision = float(p[i])
#     recall = float(r[i])
#     ap_50 = float(ap50[i])
#     ap_5095 = float(ap[i])

#     print(f"{cls_name:25s} | P: {precision:.3f} | R: {recall:.3f} | AP50: {ap_50:.3f} | AP50-95: {ap_5095:.3f}")

#     rows.append({
#         "class": cls_name,
#         "precision": precision,
#         "recall": recall,
#         "AP50": ap_50,
#         "AP50-95": ap_5095
#     })

