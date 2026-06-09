!pip install -U ultralytics albumentations onnxslim onnxruntime-gpu


from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.metrics import Metric
from ultralytics import RTDETR, settings
from functools import partial
import albumentations as A
import numpy as np
import warnings
import random
import shutil
import torch
import yaml
import json
import os

warnings.filterwarnings("ignore")


class CFG:
    dataset_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
    train_image_path = os.path.join(dataset_path, "train")
    train_label_path = os.path.join(dataset_path, "train_labels.csv")

    debug = False

    seed = 42
    n_fold = 5
    current_fold = 0
    
    epochs = 1 if debug else 10
    image_size = 640
    es_patience = 7
    device = "0"

    yolo_dataset_path = f"/temp/dataset/fold_{current_fold}"
    yolo_yaml_path = f"/temp/dataset/fold_{current_fold}/dataset.yaml"
    yolo_model_name = "rtdetr-l.pt"

    use_wandb = False
    project = "byu-locating-bacterial-flagellar-motors"
    name = yolo_model_name.split(".")[0] + f"_fold_{current_fold}"


torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)


settings.update({
    "runs_dir": "/temp/logs", 
    "tensorboard": False
})

if CFG.use_wandb:
    from kaggle_secrets import UserSecretsClient
    import wandb
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    settings.update({"wandb": True})


if os.path.exists(CFG.yolo_dataset_path):
    shutil.rmtree(CFG.yolo_dataset_path)

shutil.copytree(f"/kaggle/input/byu-flagellar-motor-detection-1-preprocessing/dataset/fold_{CFG.current_fold}", CFG.yolo_dataset_path)
os.makedirs(CFG.project, exist_ok=True)


yaml_data = {
    "path": CFG.yolo_yaml_path,
    "train": os.path.join(CFG.yolo_dataset_path, "images", "train"),
    "val": os.path.join(CFG.yolo_dataset_path, "images", "val"),
    "names": {0: "motor"}
}

with open(CFG.yolo_yaml_path, "w") as f:
    yaml.dump(yaml_data, f)


def __init__(self, p=1.0):
    self.p = p
    self.transform = None
    prefix = colorstr("albumentations: ")

    try:
        spatial_transforms = {
            "Affine",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "CoarseDropout",
            "Crop",
            "CropAndPad",
            "CropNonEmptyMaskIfExists",
            "D4",
            "ElasticTransform",
            "Flip",
            "GridDistortion",
            "GridDropout",
            "HorizontalFlip",
            "Lambda",
            "LongestMaxSize",
            "MaskDropout",
            "MixUp",
            "Morphological",
            "NoOp",
            "OpticalDistortion",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "PixelDropout",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomGridShuffle",
            "RandomResizedCrop",
            "RandomRotate90",
            "RandomScale",
            "RandomSizedBBoxSafeCrop",
            "RandomSizedCrop",
            "Resize",
            "Rotate",
            "SafeRotate",
            "ShiftScaleRotate",
            "SmallestMaxSize",
            "Transpose",
            "VerticalFlip",
            "XYMasking",
        } 
        
        T = [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5)
        ]

        self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
        
        self.transform = (
            A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], check_each_transform=True))
            if self.contains_spatial
            else A.Compose(T)
        )
        
        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(torch.initial_seed())
        
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")

Albumentations.__init__ = __init__


def fitness(self):
    return (5 * self.mp * self.mr) / (4 * self.mp + self.mr)

Metric.fitness = fitness


model = RTDETR(CFG.yolo_model_name)

results = model.train(
    data=CFG.yolo_yaml_path,
    epochs=CFG.epochs,
    batch=6,
    device=CFG.device,
    imgsz=CFG.image_size,
    optimizer='AdamW',
    lr0=1e-4,
    lrf=0.1,
    warmup_epochs=0,
    dropout=0.1,
    project=CFG.project,
    name=CFG.name,
    exist_ok=True,
    patience=CFG.es_patience,
    save=True,
    seed=CFG.seed,
    val=True,
    verbose=True
)


results = model.val(verbose=False, save_json=True)

print(json.dumps({
    "ap": float(results.box.ap[0]),
    "ap50": float(results.box.ap50[0]),
    "f1": float(results.box.f1[0]),
    "map": float(results.box.map),
    "map50": float(results.box.map50),
    "map75": float(results.box.map75),
    "maps": float(results.box.maps[0]),
    "mp": float(results.box.mp),
    "mr": float(results.box.mr),
    "p": float(results.box.p[0]),
    "r": float(results.box.r[0]),
    "f2": (5 * float(results.box.mp) * float(results.box.mr)) / (4 * float(results.box.mp) + float(results.box.mr))
}, indent=4))


model.export(format='torchscript', imgsz=CFG.image_size, optimize=False, batch=8)


shutil.rmtree("wandb", ignore_errors=True)
try:
    os.remove("yolo11n.pt")
    os.remove(CFG.yolo_model_name)
except:
    pass

