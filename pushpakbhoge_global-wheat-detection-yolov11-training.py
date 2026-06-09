!pip install ultralytics --no-deps --upgrade
!pip install "decord"
!pip install "ftfy==6.1.1"
!pip install "iopath>=0.1.10"
!git clone -b apple-silicon-support https://github.com/provos/sam3.git # have fixes for CPU runtime
!mv sam3 sam3_repo


from kaggle_secrets import UserSecretsClient
import huggingface_hub as hf
import wandb

#### login to huggingface
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
hf.login(token=hf_token)

#### login to wandb
!yolo settings wandb=True
wandb_key = user_secrets.get_secret("wandbKey")
wandb.login(key=wandb_key)

#### add repository in the system path
import sys
sys.path.insert(0, "./sam3_repo")


from pathlib import Path
import os

RANDOM_SEED = 9235789
ANNOTATION_PATH = "/kaggle/input/global-wheat-detection/train.csv"
DATASET_BASE = Path("/kaggle/working/global_wheat_detection")
TRAIN_IMAGE_DIR = DATASET_BASE / "images" / "train"
TRAIN_LABEL_DIR = DATASET_BASE / "labels" / "train"
VAL_IMAGE_DIR = DATASET_BASE / "images" / "val"
VAL_LABEL_DIR = DATASET_BASE / "labels" / "val"
SRC_IMAGE_DIR = Path("/kaggle/input/global-wheat-detection/train")

DATA_CONFIG = "./dataset_config.yml"
PROJECT_NAME = "SAM3 Comparisons"
EXPERIMENT_NAME = "global_wheat_detection_yolo_v11"

# create the directoris
os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)


from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Union
from tqdm.notebook import tqdm
import pandas as pd
import shutil
import os

def convert_to_yolo_txts(df: pd.DataFrame, out_dir: Path, class_id: int = 0):
    for _, row in df.iterrows():
        image_id = row["image_id"]
        img_w, img_h = row["width"], row["height"]
        xmin, ymin, bw, bh = eval(row["bbox"])

        # convert to YOLO normalized format
        x_center = (xmin + bw / 2) / img_w
        y_center = (ymin + bh / 2) / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h

        # write the line to respective files
        line = f"{class_id} {x_center} {y_center} {w_norm} {h_norm}"
        txt_path = out_dir / f"{image_id}.txt"
        with open(txt_path, "a") as f:
            f.write(line + "\n")

def copy_images(image_ids: List[str], src_path: Path, dest_path: Path):
    with ThreadPoolExecutor() as executor:
        futures = []
        for image_id in image_ids:
            src_image = src_path / f"{image_id}.jpg"
            dst_image = dest_path / f"{image_id}.jpg"
            if not dst_image.exists():
                future = executor.submit(shutil.copy, src_image, dst_image)
                futures.append(future)

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update()
                except Exception as err:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise err

    return True


from sklearn.model_selection import train_test_split
import yaml


# load full annotations
annotation_df = pd.read_csv(ANNOTATION_PATH)

# split the data frame into train or test columns
train_image_ids, val_image_ids = train_test_split(annotation_df['image_id'].unique(), test_size=0.1, random_state=RANDOM_SEED)
train_df = annotation_df[annotation_df['image_id'].isin(train_image_ids)]
val_df = annotation_df[annotation_df['image_id'].isin(val_image_ids)]

# copy the images to the respective directories
copy_images(train_image_ids, src_path=SRC_IMAGE_DIR, dest_path=TRAIN_IMAGE_DIR)
copy_images(val_image_ids, src_path=SRC_IMAGE_DIR, dest_path=VAL_IMAGE_DIR)

# convert annotations to txts
convert_to_yolo_txts(train_df, out_dir=TRAIN_LABEL_DIR)
convert_to_yolo_txts(val_df, out_dir=VAL_LABEL_DIR)

# save data config file
data_config = {
    "path": str(DATASET_BASE),
    "train": "images/train",
    "val": "images/val",
    "names": {0: "wheat_head"}
}

with open(DATA_CONFIG, "w") as f:
    yaml.dump(data_config, f, sort_keys=False)


from ultralytics import YOLO

model = YOLO("yolo11l.pt")
results = model.train(
    data=DATA_CONFIG, 
    project=PROJECT_NAME, 
    name=EXPERIMENT_NAME,
    seed=RANDOM_SEED,
    half=True, 
    exist_ok=True,
    amp=True,
    imgsz=768,
    freeze=None,
    single_cls=True,
    epochs=30,
    patience=10,
    close_mosaic=5,
    batch=12,
    optimizer='Adam',
    lr0=0.001,
    degrees=20,
    translate=0.2,
    flipud=0.5,
    fliplr=0.5
)


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from functools import partial
from PIL import Image
import json
import torch


def evaluate_coco(coco_gt, coco_pred):
    coco_gt = COCO(coco_gt)
    coco_dt = coco_gt.loadRes(coco_pred)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR_1": coco_eval.stats[6],
        "AR_10": coco_eval.stats[7],
        "AR_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }
    return metrics


def generate_coco_formatted_datasets(infer_func: Callable, df: pd.DataFrame, image_dir: Path):
    coco_gt = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "wheat_head"}]
    }
    
    coco_preds = []
    image_id, ann_id = 1, 1
    grouped = df.groupby("image_id")
    for image_file_id, rows in tqdm(grouped):
        coco_gt["images"].append({
            "id": image_id,
            "file_name": f"{image_file_id}.jpg",
            "width": int(rows["width"].iloc[0]),
            "height": int(rows["height"].iloc[0])
        })

        # add annotations
        for _, row in rows.iterrows():
            x1, y1, w, h = eval(row["bbox"])
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        # inference and accumulate predictions
        image_path = image_dir / f"{image_file_id}.jpg"
        coco_preds += infer_func(image_path=image_path, image_id=image_id)
        image_id += 1

    return coco_gt, coco_preds

def nms(boxes, scores, iou_threshold=0.5):
    # boxes: (N, 4) in xyxy
    # scores: (N,)
    idxs = scores.argsort(descending=True)
    keep = []

    while idxs.numel() > 0:
        current = idxs[0].item()
        keep.append(current)

        if idxs.numel() == 1:
            break

        rest = idxs[1:]

        x1 = torch.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = torch.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = torch.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = torch.minimum(boxes[current, 3], boxes[rest, 3])

        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter = inter_w * inter_h

        area_current = (boxes[current, 2] - boxes[current, 0]) * \
                       (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * \
                    (boxes[rest, 3] - boxes[rest, 1])

        union = area_current + area_rest - inter
        iou = inter / union

        idxs = rest[iou < iou_threshold]

    return torch.tensor(keep, dtype=torch.long)

def perform_non_max_suppression(outputs, iou_threshold: float = 0.5):
    keep_idx = nms(outputs['boxes'], outputs['scores'], iou_threshold=iou_threshold)
    outputs['scores'] = outputs['scores'][keep_idx]
    outputs['boxes'] = outputs['boxes'][keep_idx, :]
    outputs['masks'] = outputs['masks'][keep_idx, :, :]
    return outputs

def run_sam3(processor: Sam3Processor, image: Union[str, Image], prompt: str):
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    # run the model
    inference_state  = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return output


########################### Benchmark the best trained model
def infer_yolo_model(model, image_path: Path, image_id: int):
    image_preds = []
    results = model.predict(image_path, verbose=False)[0]
    for pred in results.boxes:
        xmin, ymin, xmax, ymax = pred.xyxy[0].tolist()
        image_preds.append({
            "image_id": image_id,
            "category_id": int(pred.cls),
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "score": float(pred.conf)
        })

    return image_preds

# load the model
best_model_weight = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, 'weights', 'best.pt')
best_model = YOLO(best_model_weight)

# do the inference on validation dataset to get benchmarks
infer_func = partial(infer_yolo_model, model=best_model)
coco_gt, coco_yolo_dt = generate_coco_formatted_datasets(infer_func, val_df, VAL_IMAGE_DIR)

# save the ground truth and detection files
coco_yolo_gt_path = os.path.join(PROJECT_NAME, EXPERIMENT_NAME,  "global_wheat_yolo_gt.json")
coco_yolo_dt_path = os.path.join(PROJECT_NAME, EXPERIMENT_NAME,  "global_wheat_yolo_dt.json")
with open(coco_yolo_gt_path, 'w') as file:
    json.dump(coco_gt, file)
with open(coco_yolo_dt_path, 'w') as file:
    json.dump(coco_yolo_dt, file)

# evaluate and record the metrics
yolo_metrics = evaluate_coco(coco_yolo_gt_path, coco_yolo_dt_path)


########################### Benchmark the SAM3 model

def infer_sam3_model(model, image_path: Path, image_id: int):
    image_preds = []
    sam3_predictions = run_sam3(model, image_path, prompt="wheat, flower, seed head, green seed head, spike, green spike")
    sam3_predictions = perform_non_max_suppression(sam3_predictions, iou_threshold=0.5)
    for score, bbox in zip(sam3_predictions['scores'].tolist(), sam3_predictions['boxes'].tolist()):
        xmin, ymin, xmax, ymax = bbox
        image_preds.append({
            "image_id": image_id,
            "category_id": 0,
            "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
            "score": float(score)
        })

    return image_preds


### build SAM3 model
model = build_sam3_image_model(enable_segmentation=True)
processor = Sam3Processor(model, confidence_threshold=0.3)

# do the inference on validation dataset to get benchmarks
infer_func = partial(infer_sam3_model, model=processor)
coco_gt, coco_sam_dt = generate_coco_formatted_datasets(infer_func, val_df, VAL_IMAGE_DIR)

# save the ground truth and detection files
coco_sam3_gt_path = "./global_wheat_sam3_gt.json"
coco_sam3_dt_path = "./global_wheat_sam3_dt.json"
with open(coco_sam3_gt_path, 'w') as file:
    json.dump(coco_gt, file)
with open(coco_sam3_dt_path, 'w') as file:
    json.dump(coco_sam_dt, file)

# evaluate and record the metrics
sam3_metrics = evaluate_coco(coco_sam3_gt_path, coco_sam3_dt_path)


from IPython.display import display, HTML
import pandas as pd

def compare_coco_metrics(metrics_a, metrics_b, name_a="Model A", name_b="Model B"):
    def safe(v):
        return 0 if v == -1 else v

    df = pd.DataFrame([
        {
            "Metric": key,
            name_a: round(safe(metrics_a[key]), 4),
            name_b: round(safe(metrics_b[key]), 4),
            "Î” (B - A)": round(safe(metrics_b[key]) - safe(metrics_a[key]), 4),
            "% Change": (
                round(((safe(metrics_b[key]) - safe(metrics_a[key])) / safe(metrics_a[key])) * 100, 2)
                if safe(metrics_a[key]) != 0 else None
            )
        }
        for key in metrics_a.keys()
    ])

    # Pretty colors
    def color_diff(val):
        if val is None:
            return ""
        if val > 0:
            return "background-color: #b2fab4"
        elif val < 0:
            return "background-color: #ffb3b3"
        return ""

    styled = (
        df.style
        .applymap(color_diff, subset=["Î” (B - A)", "% Change"])
        .set_properties(**{"text-align": "center"})
        .hide(axis="index")
    )

    display(styled)

    # Scoring weights
    weights = {
        "AP": 3.0, "AP50": 2.0, "AP75": 2.0,
        "AP_small": 0.7, "AP_medium": 0.5, "AP_large": 0.2,
        "AR_100": 3.0, "AR_1": 0.1, "AR_10": 0.1,
        "AR_small": 0.7, "AR_medium": 0.5, "AR_large": 0.2
    }

    score_a, score_b = 0, 0
    for key in metrics_a.keys():
        w = weights.get(key, 1.0)
        score_a += safe(metrics_a[key]) * w
        score_b += safe(metrics_b[key]) * w

    diff_total = score_b - score_a
    pct_change_total = (diff_total / score_a * 100) if score_a != 0 else None

    print("------------------------------------------------------------")
    print("ğŸ�† **Overall Comparison**")
    print(f"{name_a} Score: {score_a:.4f}")
    print(f"{name_b} Score: {score_b:.4f}")
    print(f"Difference (B - A): {diff_total:.4f}")

    if pct_change_total is not None:
        print(f"Percentage Difference: {pct_change_total:.2f}%")
    else:
        print("Percentage Difference: N/A")

    if diff_total > 0:
        print(f"\nğŸ”¥ **{name_b} is overall better by {diff_total:.4f} points ({pct_change_total:.2f}%).**")
    elif diff_total < 0:
        print(f"\nğŸ”¥ **{name_a} is overall better by {-diff_total:.4f} points ({-pct_change_total:.2f}%).**")
    else:
        print("\nğŸ¤� Both models are equally good (tie).")
    print("------------------------------------------------------------")

    return df




comparision = compare_coco_metrics(yolo_metrics, sam3_metrics, name_a='yolov11-nano', name_b='SAM3')
comparision.to_csv('results.csv', index=False)




