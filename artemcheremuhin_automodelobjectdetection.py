# ===========================
# БЛОК 0. Импорты, среда, утилиты логирования
# ===========================
import os, sys, re, ast, json, time, random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
import datasets as hf_datasets  # только чтобы отключить кэш

# ---- Настройки окружения для стабильности ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
hf_datasets.disable_caching()  # чтобы Datasets не лезли в кэш

# детерминированность
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log("Старт. Проверка версий и устройства…")
log(f"Python: {sys.version.split()[0]}")
log(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | Device: {DEVICE}")


# ===========================
# БЛОК 1. Пути, константы, sanity-check
# ===========================
DATA_DIR = Path("/kaggle/input/global-wheat-detection")
CSV_PATH = DATA_DIR / "train.csv"
IMG_DIR_CANDIDATES = [
    DATA_DIR / "train",
    DATA_DIR / "train_images",
    DATA_DIR / "images",
]

MODEL_ID = "microsoft/conditional-detr-resnet-50"
ID2LABEL = {0: "wheat_head"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "640"))
DEBUG_MAX_IMAGES = int(os.environ.get("DEBUG_MAX_IMAGES", "800"))   # ограничение на кол-во изображений
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.85"))

log(f"DATA_DIR = {DATA_DIR}")
assert CSV_PATH.exists(), f"Не найден CSV: {CSV_PATH}"

IMG_DIR = None
for cand in IMG_DIR_CANDIDATES:
    if cand.exists():
        IMG_DIR = cand
        break
assert IMG_DIR is not None, "Не найдена папка с изображениями (train/train_images/images)"
log(f"IMG_DIR = {IMG_DIR}")


# ===========================
# БЛОК 2. Загрузка train.csv, парсинг bbox, сбор записей
# ===========================
df = pd.read_csv(CSV_PATH)
needed_cols = {"image_id", "bbox"}
assert needed_cols.issubset(df.columns), f"В CSV нет нужных колонок: {needed_cols - set(df.columns)}"
log(f"CSV считан: {len(df)} строк")

# Сгруппируем боксы по изображению
grouped = df.groupby("image_id")["bbox"].apply(list).reset_index()
log(f"Сгруппированных изображений: {len(grouped)}")

def parse_bbox_str(bbox_str):
    # bbox хранится как строка вида "[x, y, w, h]"
    # используем `ast.literal_eval` безопасно
    arr = ast.literal_eval(bbox_str)
    if len(arr) != 4:
        raise ValueError("bbox не из 4 элементов")
    x, y, w, h = map(float, arr)
    return x, y, w, h

records = []
skipped_boxes = 0
kept_boxes = 0
missing_images = 0

log("Сбор записей (с прогрессом и проверками)…")
for _, row in tqdm(grouped.iterrows(), total=len(grouped)):
    image_id = row["image_id"]
    img_path = IMG_DIR / f"{image_id}.jpg"
    if not img_path.exists():
        # на всякий случай попробуем .png
        alt = IMG_DIR / f"{image_id}.png"
        if alt.exists(): img_path = alt
    if not img_path.exists():
        missing_images += 1
        continue

    try:
        with Image.open(img_path) as im:
            W, H = im.size
    except Exception as e:
        log(f"⚠️ Не читается {img_path}: {e}")
        continue

    bboxes_xywh = []
    cats = []
    for b in row["bbox"]:
        try:
            x, y, w, h = parse_bbox_str(b)
        except Exception:
            skipped_boxes += 1
            continue

        # клиппинг в границы
        x = max(0.0, min(x, W - 1))
        y = max(0.0, min(y, H - 1))
        w = max(0.0, min(w, W - x))
        h = max(0.0, min(h, H - y))

        if w <= 1e-6 or h <= 1e-6:
            skipped_boxes += 1
            continue

        bboxes_xywh.append([x, y, w, h])
        cats.append(0)  # один класс

    if len(bboxes_xywh) == 0:
        # можно пропустить изображения без валидных боксов
        continue

    kept_boxes += len(bboxes_xywh)
    records.append({
        "image_id": image_id,
        "image_path": str(img_path),
        "width": W,
        "height": H,
        "bboxes": bboxes_xywh,
        "labels": cats
    })

log(f"Итого записей: {len(records)} | kept_boxes={kept_boxes} | skipped_boxes={skipped_boxes} | missing_images={missing_images}")
assert len(records) > 0, "После фильтрации записей не осталось"

# Ограничим для отладки
if DEBUG_MAX_IMAGES and len(records) > DEBUG_MAX_IMAGES:
    records = records[:DEBUG_MAX_IMAGES]
    log(f"DEBUG: Ограничили до {len(records)} изображений")


# ===========================
# БЛОК 3. Трен/вал сплит + sanity-checks
# ===========================
rng = np.random.default_rng(SEED)
idx = np.arange(len(records))
rng.shuffle(idx)
cut = int(TRAIN_FRAC * len(records))
train_idx, val_idx = idx[:cut], idx[cut:]

train_recs = [records[i] for i in train_idx]
val_recs   = [records[i] for i in val_idx]

log(f"Train: {len(train_recs)} | Val: {len(val_recs)}")
assert len(train_recs) > 0 and len(val_recs) > 0, "Проверьте размеры сплита"


# ===========================
# БЛОК 4. Процессор изображений DETR + проверка на одном примере
# ===========================
log("Загружаем процессор изображений (AutoImageProcessor)…")
image_processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    do_resize=True, size={"shortest_edge": IMAGE_SIZE},  # корректно для DETR
    do_pad=True, pad_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
)
log(f"Processor OK. Формат ожидания: {image_processor}")

# Быстрый тест на одном изображении
with Image.open(train_recs[0]["image_path"]) as test_im:
    test_ann = {
        "image_id": 0,
        "annotations": [
            {"bbox": bbox, "category_id": 0} for bbox in train_recs[0]["bboxes"]
        ],
    }
    enc = image_processor(images=test_im, annotations=test_ann, return_tensors="pt")
    log(f"Sanity processor → pixel_values: {tuple(enc['pixel_values'].shape)}; labels-keys: {list(enc['labels'][0].keys())}")


# ===========================
# БЛОК 5. Кастомный Dataset + collate_fn + dry-run
# ===========================
class WheatDataset(Dataset):
    def __init__(self, recs, image_processor):
        self.recs = recs
        self.proc = image_processor

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        with Image.open(r["image_path"]).convert("RGB") as im:
            ann = {
                "image_id": i,  # важно быть int
                "annotations": [
                    {"bbox": bbox, "category_id": 0} for bbox in r["bboxes"]
                ],
            }
            enc = self.proc(images=im, annotations=ann, return_tensors="pt")
            # enc['pixel_values']: (1, 3, H, W); enc['labels']: list из 1 элемента
            item = {
                "pixel_values": enc["pixel_values"].squeeze(0),  # -> (3,H,W)
                "labels": enc["labels"][0],                      # dict: boxes, class_labels
                "orig_size": (r["height"], r["width"]),
                "image_path": r["image_path"],
                "image_id": r["image_id"],
            }
            return item

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])  # (B,3,H,W)
    labels = [{"boxes": b["labels"]["boxes"], "class_labels": b["labels"]["class_labels"]} for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}

train_ds = WheatDataset(train_recs, image_processor)
val_ds   = WheatDataset(val_recs,   image_processor)

# Dry-run DataLoader (важно для отладки зависаний)
log("Dry-run DataLoader (2 батча)…")
dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
for step, batch in enumerate(dl):
    log(f"Batch {step}: pixel_values={tuple(batch['pixel_values'].shape)}; labels={len(batch['labels'])}")
    if step >= 1: break


# ===========================
# БЛОК 6. Модель (та же), загрузка и проверка
# ===========================
log("Грузим модель AutoModelForObjectDetection…")
model = AutoModelForObjectDetection.from_pretrained(
    MODEL_ID,
    ignore_mismatched_sizes=True,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model.to(DEVICE)
log("Модель загружена.")

# Быстрый forward на одном батче (поймать shape/NaN)
batch = next(iter(DataLoader(train_ds, batch_size=2, num_workers=0, collate_fn=collate_fn)))
with torch.no_grad():
    out = model(pixel_values=batch["pixel_values"].to(DEVICE), labels=[
        {"boxes": x["boxes"].to(DEVICE), "class_labels": x["class_labels"].to(DEVICE)} for x in batch["labels"]
    ])
log(f"Sanity forward → loss={float(out.loss):.4f}")


# ===========================
# БЛОК 7. Аргументы тренировки + Trainer (без многопроцессности)
# ===========================
MAX_STEPS = int(os.environ.get("MAX_STEPS", "300"))   # безопасно маленькое значение
PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "2"))
LR = float(os.environ.get("LR", "5e-5"))

args = TrainingArguments(
    output_dir="/kaggle/working/wheat-cond-detr",
    per_device_train_batch_size=PER_DEVICE_BS,
    per_device_eval_batch_size=PER_DEVICE_BS,
    dataloader_num_workers=0,              # критично: без воркеров
    remove_unused_columns=False,           # для object detection
    learning_rate=LR,
    weight_decay=5e-4,
    max_steps=MAX_STEPS,                   # вместо num_train_epochs
    logging_steps=10,
    eval_strategy="no",                    # для стабильности; оценку делаем вручную ниже
    save_strategy="no",
    report_to=[],                          # без W&B
    fp16=torch.cuda.is_available(),        # ускорение, если поддерживается
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=1,
    seed=SEED,
)

def hf_collator(features):
    # Trainer свой collator — используем наш
    return collate_fn(features)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=None,   # оценим вручную
    data_collator=hf_collator,
)

log("Старт trainer.train()… (пара минут макс. при MAX_STEPS=300)")
train_out = trainer.train()
log(f"Тренировка завершена. Итоговый global_step={trainer.state.global_step}, loss={train_out.training_loss:.4f}")


# ===========================
# БЛОК 8. Быстрая ручная оценка mAP на валидации (torchmetrics)
# ===========================
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def xywh_to_xyxy(xywh):
    x, y, w, h = xywh.T
    xyxy = np.stack([x, y, x+w, y+h], axis=1)
    return xyxy

@torch.no_grad()
def evaluate_map(model, dataset, image_processor, max_batches=50, score_thresh=0.05):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    dl = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    n_batches = 0
    for batch in tqdm(dl, total=min(max_batches, len(dl)), desc="Eval"):
        outputs = model(pixel_values=batch["pixel_values"].to(DEVICE))
        # post-process в исходные размеры
        target_sizes = torch.tensor([list(b["orig_size"]) for b in batch], device=DEVICE)
        results = image_processor.post_process_object_detection(outputs, threshold=score_thresh, target_sizes=target_sizes)

        preds, gts = [], []
        for i, res in enumerate(results):
            # предсказания
            boxes = res["boxes"].cpu()  # xyxy
            scores = res["scores"].cpu()
            labels = res["labels"].cpu()

            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

            # таргеты (переведём xywh -> xyxy)
            # Возьмём их из batch['labels'] НЕпосле процессора (они уже в координатах до ресайза)
            # Мы их не держим отдельно; поэтому возьмём из исходных записей dataset.recs
            # Сопоставим индекс по батчу → глобальный индекс через image_path
            rec = next(r for r in dataset.recs if r["image_path"] == batch["image_path"][i] if "image_path" in batch else True)  # fallback
            # Но проще: мы уже сохранили в batch["labels"] xyxy под процессором? Нет, там нормированные под processor размеры.
            # Поэтому сделаем аккуратно: возьмём из dataset.recs исходные xywh.
            # У нас есть batch["image_path"] → добавим его в batch в Dataset.
        # Исправим стратегию: в Dataset мы уже кладём orig_size, а боксы передавали только в processor.
        # Значит, легче прямо сейчас получить GT из dataset по image_id:

        break  # выходим, чтобы аккуратно переписать цикл с корректными GT

    # Перепишем оценку с явной индексацией датасета по DataLoader'у.

@torch.no_grad()
def evaluate_map(model, dataset, image_processor, max_images=100, score_thresh=0.05):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    # Возьмём первые N изображений из вал-датасета
    N = min(max_images, len(dataset))
    idxs = list(range(N))
    for i in tqdm(idxs, desc="Eval img-by-img"):
        item = dataset[i]
        pv = item["pixel_values"].unsqueeze(0).to(DEVICE)
        out = model(pixel_values=pv)

        target_sizes = torch.tensor([list(item["orig_size"])], device=DEVICE)
        res = image_processor.post_process_object_detection(out, threshold=score_thresh, target_sizes=target_sizes)[0]

        # preds
        pred = {
            "boxes": res["boxes"].cpu(),
            "scores": res["scores"].cpu(),
            "labels": res["labels"].cpu(),
        }

        # gts (из исходного набора записей)
        gt_xywh = np.array(dataset.recs[i]["bboxes"], dtype=np.float32)
        gt_xyxy = torch.tensor(xywh_to_xyxy(gt_xywh), dtype=torch.float32)
        gt = {
            "boxes": gt_xyxy,
            "labels": torch.tensor([0]*len(gt_xyxy), dtype=torch.int64),
        }

        metric.update([pred], [gt])

        if (i+1) % 10 == 0:
            inter = metric.compute()
            log(f"  ↳ промежуточно @ {i+1} изображений: mAP50={inter['map_50']:.4f}, mAP={inter['map']:.4f}")

    final = metric.compute()
    log(f"VAL mAP50={final['map_50']:.4f} | mAP={final['map']:.4f} | map_small={final.get('map_small', torch.nan):.4f}")
    return final

log("Оценка на валидации (до 100 изображений)…")
_ = evaluate_map(model, val_ds, image_processor, max_images=100, score_thresh=0.1)


# ===========================
# БЛОК 9. Визуализация одного примера (GT vs Pred)
# ===========================
def draw_boxes(image_path, gt_xywh, pred_xyxy, pred_scores, score_thr=0.5, out_path="/kaggle/working/pred_example.jpg"):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    # GT (зелёный)
    for (x, y, w, h) in gt_xywh:
        draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=2)
    # Pred (красный)
    for (x1,y1,x2,y2), s in zip(pred_xyxy, pred_scores):
        if s < score_thr: continue
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
        draw.text((x1, max(0, y1-12)), f"{s:.2f}", fill=(255,0,0))
    im.save(out_path)
    return out_path

# возьмём произвольное вал-изображение
i = 0 if len(val_ds)==0 else min(0, len(val_ds)-1)
item = val_ds[i]
with torch.no_grad():
    out = model(pixel_values=item["pixel_values"].unsqueeze(0).to(DEVICE))
res = image_processor.post_process_object_detection(out, threshold=0.1, target_sizes=torch.tensor([list(item["orig_size"])], device=DEVICE))[0]

pred_xyxy = res["boxes"].cpu().numpy()
pred_scores = res["scores"].cpu().numpy()
gt_xywh = np.array(val_ds.recs[i]["bboxes"])

out_path = draw_boxes(val_ds.recs[i]["image_path"], gt_xywh, pred_xyxy, pred_scores, score_thr=0.5)
log(f"Сохранил визуализацию: {out_path}")




