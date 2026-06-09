pip install --upgrade torch ultralytics


from ultralytics import YOLO
import os, pandas, numpy, cv2
from pathlib import Path


CLASSES=['labels']

yaml_content = f"""
train: /kaggle/input/simulated-object-data/data/train/images
val: /kaggle/input/simulated-object-data/data/val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""

with open("dataset.yaml", "w") as f:
    f.write(yaml_content)

print("dataset.yaml created!")


yolo = YOLO("yolo11m.pt")

yolo.train( data='/kaggle/working/dataset.yaml',
            epochs=150,
            batch=10,
            imgsz=924,
            patience=5,
            lr0=0.001,
            lrf=0.02,
            optimizer="Adam",
            momentum=0.96,
            weight_decay=0.001,
            cos_lr=True,
            dropout=0.3,
            label_smoothing=0.1,
            mosaic=0.5,
            mixup=0.15,
            copy_paste=0.1,
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.4,
            translate=0.2,
            scale=0.5,
            shear=0.2,
            perspective=0.0002,
            val=True,
            workers=8,
            seed=42,
            device=[-1, -1]
        )
valid_results = yolo.val()
print(valid_results)


model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')


output_dir = r"/kaggle/working/predictions/labels"
os.makedirs(output_dir, exist_ok=True)


for i in os.listdir('/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images'):
    img_path = f'/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images/{i}'
    results = model.predict(img_path, conf=0.3, device=0, verbose=False) # 0 - GPU or "cpu"
    output_txt = f"{output_dir}/{i.split('.')[0]}.txt"

    with open(output_txt, "w") as f:
        found = False
        for result in results:
            img_height, img_width = result.orig_shape
            boxes = result.boxes.data

            if boxes is None or len(boxes) == 0:
                continue

            filtered_boxes = boxes[boxes[:, 4] >= 0.05]
            if len(filtered_boxes) == 0:
                continue

            found = True
            for box in filtered_boxes:
                x1, y1, x2, y2, confidence, cls_id = box.tolist()

                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                f.write(f"0 {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        if not found:
            f.write("")


rows = []
output_dir = Path("/kaggle/working/predictions/labels")
TEST = Path('/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images')
test_imgs = {p.stem for p in TEST.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
predicted = set()

for file in output_dir.glob("*.txt"):
    name = file.stem
    predicted.add(name)

    try:
        lines = [l.strip() for l in open(file) if len(l.strip().split()) == 6]
    except:
        lines = []

    rows.append({"image_id": name, "prediction_string": " ".join(lines) if lines else "no boxes"})

for name in test_imgs - predicted:
    rows.append({"image_id": name, "prediction_string": "no boxes"})

work_dir = '/kaggle/working'

for filename in os.listdir(work_dir):
    file_path = os.path.join(work_dir, filename)

    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Error file: {file_path}. Cause: {e}')



rows = pandas.DataFrame(rows)
rows.to_csv("submission.csv", index=False)
rows

