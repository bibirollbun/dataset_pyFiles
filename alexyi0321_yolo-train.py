!pip install ultralytics


import yaml

data_yaml = {
    "path": "/kaggle/input/yolo-train-rotate/dataset_rotate",  # change to your actual path
    "train": "images/train",
    "val": "images/train",
    "names": ["panel"]  # change class names
}

with open("train.yaml", "w") as f:
    yaml.dump(data_yaml, f)


from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")

model.train(
    data="train.yaml",
    epochs=20,
    imgsz=2048,
    batch=8,
    project="yolo_obb",
    name="obb-v1-2",
    device='0, 1, 2, 3',         # set to 'auto' if using CPU fallback
    save_period=1
)


!zip -r out.zip /kaggle/working/yolo_obb/obb-v1-2/weights


model.export(format="openvino", dynamic=True, nms=True, imgsz=2048)

