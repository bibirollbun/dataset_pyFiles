# import os

# def print_directory_tree(path, indent_level=0):
#     indent = '    ' * indent_level
#     print(f"{indent}{os.path.basename(path)}/")
#     if os.path.isdir(path):
#         for item in sorted(os.listdir(path)):
#             item_path = os.path.join(path, item)
#             if os.path.isdir(item_path):
#                 print_directory_tree(item_path, indent_level + 1)
#             else:
#                 print(f"{'    ' * (indent_level + 1)}{item}")

# base_path = '/kaggle/input/czii-making-datasets-for-yolo-synthetic-data'
# print_directory_tree(base_path)


!pip install ultralytics


from tqdm import tqdm
import glob, os
from ultralytics import YOLO


# Load a pretrained model
model = YOLO("/kaggle/input/yolo11/pytorch/default/1/yolo11l.pt") # load a pretrained model (recommended for training)


!cp /kaggle/input/czii-synthetic/kaggle/working/czii_conf.yaml .


!sed -i 's|path: .*|path: /kaggle/input/czii-synthetic/kaggle/working/datasets/czii_det2d|g' /kaggle/working/czii_conf.yaml


# Train the model
_ = model.train(
    data="/kaggle/working/czii_conf.yaml",
    epochs=100,
    warmup_epochs=10,
    optimizer='AdamW',
    cos_lr=True,
    lr0=3e-4,
    lrf=0.03,
    imgsz=640,
    device="0",
    weight_decay=0.005,
    batch=8,
    scale=0,
    flipud=0.5,
    fliplr=0.5,
    degrees=45,
    shear=5,
    mixup=0.2,
    copy_paste=0.25,
    seed=8620, # (｡•◡•｡)
)


model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")
metrics = model.val(data="/kaggle/input/czii-yolo-datasets/czii_conf.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0", save_json=True)  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)


results = model("/kaggle/input/czii-making-datasets-for-yolo-synthetic-data/datasets/czii_det2d/images/val/TS_5_4_920.png")
results[0].show()

