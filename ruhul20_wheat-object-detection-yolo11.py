%%capture
!pip install ultralytics --no-deps --upgrade


from ultralytics import YOLO
model = YOLO("yolo11s.pt")



import os, ast, cv2, random
import pandas as pd
from tqdm import tqdm

IMG_DIR = "/kaggle/input/global-wheat-detection/train"
OUT_DIR = "yolo_wheat"

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)

df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")

image_ids = df.image_id.unique().tolist()

# ðŸ”¥ manual split (NO sklearn)
random.seed(42)
random.shuffle(image_ids)

split_idx = int(0.8 * len(image_ids))
train_ids = image_ids[:split_idx]
val_ids = image_ids[split_idx:]



def write_labels(image_id, split):
    rows = df[df.image_id == image_id]
    img_path = f"{IMG_DIR}/{image_id}.jpg"
    img = cv2.imread(img_path)

    if img is None:
        return

    h, w, _ = img.shape

    label_path = f"{OUT_DIR}/labels/{split}/{image_id}.txt"
    with open(label_path, "w") as f:
        for _, r in rows.iterrows():
            x, y, bw, bh = ast.literal_eval(r["bbox"])

            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            bw /= w
            bh /= h

            f.write(f"0 {cx} {cy} {bw} {bh}\n")

    os.system(f"cp {img_path} {OUT_DIR}/images/{split}/")



for img_id in tqdm(train_ids):
    write_labels(img_id, "train")

for img_id in tqdm(val_ids):
    write_labels(img_id, "val")



yaml_path = "/kaggle/working/dataset.yaml"
yaml_content = """
    path: yolo_wheat
    train: images
    val: images
    
    nc: 1
    names: ["wheat"]
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(" dataset.yaml created at /kaggle/working/dataset.yaml")



model.train(
    data="dataset.yaml",
    epochs=40,
    imgsz=1024,
    batch=12,
    device=0,
    optimizer="AdamW",
    lr0=3e-4,
    cos_lr=True,
    amp=True,
    cache=True,
    freeze=None,
    single_cls=True,
    patience=10,
    close_mosaic=5,
    degrees=20,
    translate=0.2,
    flipud=0.5,
    fliplr=0.5
)



from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
# Define the path to the directory containing the images
#image_directory = "/kaggle/working/runs/detect/train2"
image_directory ='/kaggle/working/runs/detect/train'

# Iterate through all image files in the directory
for filename in os.listdir(image_directory):
    if filename.lower().endswith((".jpg",".png")):
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)

        # Display the image
        plt.figure(figsize=(12, 12), dpi=150)
        plt.imshow(image)
        plt.title(f"Image: {filename}", fontsize=20, fontweight='bold', color='blue')  # Customize font properties
        plt.axis("off")  # Hide axes
        plt.show()


import pandas as pd
from tqdm import tqdm

test_df = pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")

preds = []

for image_id in tqdm(test_df.image_id):
    results = model.predict(
        source=f"/kaggle/input/global-wheat-detection/test/{image_id}.jpg",
        conf=0.25,
        iou=0.5,
        verbose=False
    )[0]

    boxes = results.boxes
    pred_str = ""

    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = b.conf[0].item()
            w = x2 - x1
            h = y2 - y1
            pred_str += f"{conf:.4f} {int(x1)} {int(y1)} {int(w)} {int(h)} "

    preds.append(pred_str.strip())

test_df["PredictionString"] = preds
test_df.to_csv("submission.csv", index=False)

print("submission.csv ready âœ…")



test_df.head()




