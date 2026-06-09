import os, random

images_input_dir = "/kaggle/input/xray-chest-jpg-conversion/train_jpg"
labels_input_dir = "/kaggle/input/chest-xray-yolo-dataset/labels"

jpg_files = sorted([f for f in os.listdir(images_input_dir) if f.lower().endswith(".jpg")])
ids = [os.path.splitext(f)[0] for f in jpg_files if os.path.exists(os.path.join(labels_input_dir, f"{os.path.splitext(f)[0]}.txt"))]

random.seed(42)
random.shuffle(ids)

split = int(len(ids) * 0.85)
train_ids, val_ids = ids[:split], ids[split:]

os.makedirs("/kaggle/working/splits", exist_ok=True)

with open("/kaggle/working/splits/train.txt", "w") as f:
    for i in train_ids:
        f.write(f"{os.path.join(images_input_dir, i+'.jpg')}\n")

with open("/kaggle/working/splits/val.txt", "w") as f:
    for i in val_ids:
        f.write(f"{os.path.join(images_input_dir, i+'.jpg')}\n")

print("Train:", len(train_ids), "Val:", len(val_ids))


yaml_content = """path: /kaggle/working/yolo_symlink
train: images/train
val: images/val
names:
  0: Aortic enlargement
  1: Atelectasis
  2: Calcification
  3: Cardiomegaly
  4: Consolidation
  5: ILD
  6: Infiltration
  7: Lung Opacity
  8: Nodule/Mass
  9: Other lesion
  10: Pleural effusion
  11: Pleural thickening
  12: Pneumothorax
  13: Pulmonary fibrosis
"""

with open("/kaggle/working/vinbig_yolo.yaml", "w") as f:
    f.write(yaml_content)


import os, errno
from tqdm import tqdm

images_input_dir = "/kaggle/input/xray-chest-jpg-conversion/train_jpg"
labels_input_dir = "/kaggle/input/chest-xray-yolo-dataset/labels"

# önceki split’leri kullanıyoruz
train_list = [l.strip() for l in open("/kaggle/working/splits/train.txt").read().splitlines()]
val_list   = [l.strip() for l in open("/kaggle/working/splits/val.txt").read().splitlines()]

root = "/kaggle/working/yolo_symlink"
paths = {
    "images/train": os.path.join(root, "images/train"),
    "images/val":   os.path.join(root, "images/val"),
    "labels/train": os.path.join(root, "labels/train"),
    "labels/val":   os.path.join(root, "labels/val"),
}
for p in paths.values():
    os.makedirs(p, exist_ok=True)

def safe_symlink(src, dst):
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def link_split(img_paths, split):
    for img_path in tqdm(img_paths, desc=f"link {split}"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_input_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue
        safe_symlink(img_path, os.path.join(paths[f"images/{split}"], base + ".jpg"))
        safe_symlink(lbl_path, os.path.join(paths[f"labels/{split}"], base + ".txt"))

link_split(train_list, "train")
link_split(val_list, "val")

print("Symlink dataset root:", root)


!pip install ultralytics --quiet


from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="/kaggle/working/vinbig_yolo.yaml",
    epochs=200,          # üst sınır
    patience=15,         # early stopping
    imgsz=640,
    batch=16,
    workers=2,
    device=0,
    seed=42,
    cache=True,          # hızlı okuma (Kaggle için iyi)
    project="runs", name="vinbig_yolov8s",
    save=True,
    plots=True,
    cos_lr=True,
    close_mosaic=10,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0  # X-ray: renk augment kapalı
)

