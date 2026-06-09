!tar xfvz /kaggle/input/ultralytics-packages/archieve.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


from tqdm import tqdm
import glob, os
from ultralytics import YOLO


#Load a pretrained model
model = YOLO("/kaggle/input/yolo11/pytorch/default/1/yolo11l.pt")


_ = model.train(
    data="/kaggle/input/czii-yolo-ddatasets/czii_conf.yaml",
    epochs=1000,
    warmup_epochs=10,
    optimizer='AdamW',
    cos_lr=True,
    lr0=3e-4,
    lrf=0.03,
    imgsz=640,
    #device="0,1",
    weight_decay=0.005,
    device="cuda",
    batch=16,
    scale=0,
    flipud=0.5,
    fliplr=0.5,
    degrees=45,
    shear=5,
    mixup=0.2,
    copy_paste=0.25,
    seed=5225,
)


model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")
metrics = model.val(data="/kaggle/input/czii-yolo-ddatasets/czii_conf.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0", save_json=True)  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)


results = model("/kaggle/input/czii-yolo-ddatasets/datasets/czii_det2d/images/val/TS_5_4_920.png")
results[0].show()


results = model("/kaggle/input/czii-yolo-ddatasets/datasets/czii_det2d/images/val/TS_5_4_1240.png")
results[0].show()

