!pip install -q 'git+https://github.com/facebookresearch/detectron2.git'


from detectron2.data import detection_utils as utils, build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, hooks
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.notebook import tqdm
import detectron2.data.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import random
import torch
import json
import copy
import cv2
import os

setup_logger()
warnings.filterwarnings("ignore")


class CFG:
    dataset_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
    train_image_path = os.path.join(dataset_path, "train")
    train_label_path = os.path.join(dataset_path, "train_labels.csv")
    sample_sub_path = os.path.join(dataset_path, "sample_submission.csv")

    seed = 42
    n_folds = 5
    current_fold = 0
    
    model_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    output_dir = model_name.split("/")[1].split(".")[0] + "_fold_" + str(current_fold)
    
    box_size = 64
    checkpoint_period = 500
    eval_period = 500
    warmup_iters = 1000
    max_iter = 10000
    learning_rate = 0.001
    batch_size = 8
    threshold = 0.4


os.makedirs(CFG.output_dir, exist_ok=True)


random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def create_dataset(tomogram_ids, labels):
    dataset_dicts = []
    
    for tomo_id in tqdm(tomogram_ids):
        tomo_motors = labels[labels['tomo_id'] == tomo_id]
        
        if len(tomo_motors) == 0:
            continue
            
        array_shape = [
            tomo_motors['Array shape (axis 0)'].iloc[0],
            tomo_motors['Array shape (axis 1)'].iloc[0],
            tomo_motors['Array shape (axis 2)'].iloc[0]
        ]
        
        tomo_path = os.path.join(CFG.train_image_path, str(tomo_id))
        
        for _, motor in tomo_motors.iterrows():
            z_pos = int(motor['Motor axis 0'])
            y_pos = int(motor['Motor axis 1'])
            x_pos = int(motor['Motor axis 2'])
            
            box_size = CFG.box_size
            x1 = max(0, x_pos - box_size//2)
            y1 = max(0, y_pos - box_size//2)
            x2 = min(array_shape[2], x_pos + box_size//2)
            y2 = min(array_shape[1], y_pos + box_size//2)
            
            slice_path = os.path.join(tomo_path, f"slice_{z_pos:04d}.jpg")
            if not os.path.exists(slice_path):
                continue

            img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape  
            record = {
                "file_name": slice_path,
                "image_id": f"{tomo_id}_{z_pos}",
                "height": height,
                "width": width,
                "annotations": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0,
                    }
                ]
            }
            dataset_dicts.append(record)
            
            z_range = 5  # Include 5 slices above and below
            for z_offset in range(-z_range, z_range + 1):
                if z_offset == 0:  # Skip the original slice
                    continue
                    
                adj_z_pos = z_pos + z_offset
                if 0 <= adj_z_pos < array_shape[0]:
                    adj_slice_path = os.path.join(tomo_path, f"{adj_z_pos:04d}.jpg")
                    if os.path.exists(adj_slice_path):
                        adj_record = {
                            "file_name": adj_slice_path,
                            "image_id": f"{tomo_id}_{adj_z_pos}",
                            "height": height,
                            "width": width,
                            "annotations": [
                                {
                                    "bbox": [x1, y1, x2, y2],
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "category_id": 0,
                                }
                            ]
                        }
                        dataset_dicts.append(adj_record)
    
    return dataset_dicts


train_labels = pd.read_csv(CFG.train_label_path)
train_labels = train_labels[train_labels["Number of motors"] == 1].reset_index(drop=True)
train_labels["fold"] = -1

split = StratifiedGroupKFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True).split(train_labels, train_labels['Number of motors'], groups=train_labels["tomo_id"])
for fold_idx, (train_idx, val_idx) in enumerate(split):
    train_labels.loc[val_idx, "fold"] = fold_idx


train_tomo_ids = train_labels[train_labels["fold"] != CFG.current_fold]['tomo_id'].unique()
val_tomo_ids = train_labels[train_labels["fold"] == CFG.current_fold]['tomo_id'].unique()

print(f"Number of training tomograms:   {len(train_tomo_ids)}")
print(f"Number of validation tomograms: {len(val_tomo_ids)}")


DatasetCatalog.clear()
MetadataCatalog.clear()

DatasetCatalog.register("train", lambda: create_dataset(train_tomo_ids, train_labels))
MetadataCatalog.get("train").set(thing_classes=["motor"])

DatasetCatalog.register("val", lambda: create_dataset(val_tomo_ids, train_labels))
MetadataCatalog.get("val").set(thing_classes=["motor"])

train_dataset = DatasetCatalog.get("train")
train_metadata = MetadataCatalog.get("train")

val_dataset = DatasetCatalog.get("val")
val_metadata = MetadataCatalog.get("val")


plt.figure(figsize=(15, 5))

samples = random.sample(train_dataset, 3)
for i, sample in enumerate(samples):
    img = cv2.imread(sample["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    visualizer = Visualizer(img, metadata=train_metadata, scale=1.0)
    vis = visualizer.draw_dataset_dict(sample)

    plt.subplot(1, 3, i + 1)
    plt.imshow(vis.get_image())
    plt.title("/".join(sample["file_name"].split("/")[-2:]), fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()


def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG.model_name))

    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)

    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG.model_name)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.SOLVER.IMS_PER_BATCH = CFG.batch_size
    cfg.SOLVER.BASE_LR = CFG.learning_rate
    cfg.SOLVER.MAX_ITER = CFG.max_iter
    cfg.SOLVER.STEPS = [2000, 4000, 6000, 8000]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = CFG.warmup_iters
    cfg.SOLVER.CHECKPOINT_PERIOD = CFG.checkpoint_period

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CFG.threshold

    cfg.TEST.EVAL_PERIOD = CFG.eval_period

    cfg.OUTPUT_DIR = CFG.output_dir

    return cfg


cfg = setup_cfg()


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
    def build_hooks(self):
        hooks_list = super().build_hooks()
        
        for idx, hook in enumerate(hooks_list):
            if isinstance(hook, hooks.PeriodicCheckpointer):
                hooks_list.pop(idx)
                break
        
        hooks_list.append(
            hooks.BestCheckpointer(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                checkpointer=DetectionCheckpointer(self.model, self.cfg.OUTPUT_DIR),
                val_metric="bbox/AP",
                mode="max",
                file_prefix="best_checkpoint"
            )
        )
        return hooks_list


trainer = Trainer(cfg)


trainer.resume_or_load(resume=False)
trainer.train()


metrics = []
for line in open(f"{CFG.output_dir}/metrics.json"):
    metrics.append(json.loads(line))
    
metrics = pd.DataFrame(metrics)


sns.set_style("whitegrid")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

colors1 = sns.color_palette("Set2", n_colors=3)
for idx, col in enumerate(["total_loss", "loss_box_reg", "loss_cls"]):
    min_val = metrics[col].min()
    min_row = metrics.loc[metrics[col].idxmin()]
    label = f"{col.replace('_', ' ').title()} (min: {min_val:.6f}@i={int(min_row.iteration)})"
    sns.lineplot(x="iteration", y=col, ax=ax1, data=metrics, linewidth=2, label=label, color=colors1[idx])
    ax1.plot(min_row["iteration"], min_row[col], marker="X", markersize=8, color=colors1[idx])

ax1.set_ylabel("Loss", fontsize=14, weight='bold')
ax1.legend(loc="upper right", frameon=True)
ax1.grid(True)

colors2 = sns.color_palette("Dark2", n_colors=3)
for idx, col in enumerate(["bbox/AP", "bbox/AP50", "bbox/AP75"]):
    max_val = metrics[col].max()
    max_row = metrics.loc[metrics[col].idxmax()]
    label = f"{col} (max: {max_val:.2f}@i={int(max_row.iteration)})"
    sns.lineplot(x="iteration", y=col, ax=ax2, data=metrics, linewidth=2, label=label, color=colors2[idx])
    ax2.plot(max_row["iteration"], max_row[col], marker="x", markersize=6, color=colors2[idx])

ax2.set_ylabel("Average Precision", fontsize=14, weight='bold')
ax2.legend(loc="lower right", frameon=True)
ax2.grid(True)

sns.lineplot(x="iteration", y="lr", ax=ax3, linewidth=2, data=metrics, color="tab:blue")
ax3.set_ylabel("Learning Rate", fontsize=14, weight='bold')
ax3.grid(True)


plt.xlabel("Iteration", fontsize=14, weight='bold')
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()


