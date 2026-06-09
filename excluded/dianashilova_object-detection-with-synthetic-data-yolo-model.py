! pip install -U ultralytics
! pip install ensemble_boxes


from ultralytics import YOLO
from pathlib import Path
from IPython.display import display, Image
import pandas as pd
import glob
import csv
import cv2
import os
import yaml
import shutil
import argparse
import matplotlib.pyplot as plt
from matplotlib import patches
from ensemble_boxes import weighted_boxes_fusion


model = YOLO("yolov8n.pt")


# # In this case we created dataset with synthetic or real data with new images and labels
def merge_dirs(src_paths, dst_images, dst_labels):
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    for src in src_paths:
        for img in (src / 'images').glob('*.*'):
            shutil.copy(img, dst_images / img.name)
        for lbl in (src / 'labels').glob('*.*'):
            shutil.copy(lbl, dst_labels / lbl.name)

merge_dirs(
    [Path('/kaggle/input/soup-can/train'), Path('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/train')],
    Path('/kaggle/working/dataset/train/images'),
    Path('/kaggle/working/dataset/train/labels')
)

merge_dirs(
    [Path('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/val')],
    Path('/kaggle/working/dataset/val/images'),
    Path('/kaggle/working/dataset/val/labels')
)

merge_dirs(
    [Path('/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages')],
    Path('/kaggle/working/dataset/test/images'),
    Path('/kaggle/working/dataset/test/labels')
)


data_yaml = """
path: /kaggle/working/dataset

train: train/images
val: val/images
test: test/images

nc: 1
names: ['soup can']
"""

with open('data.yaml', 'w') as file:
    file.write(data_yaml)


base_path = '/kaggle/working/dataset'

# Get info about number of image in created dataset
def get_info(base_path):
    for split in ['train', 'val']:
        img_dir = os.path.join(base_path, split, 'images')
        lbl_dir = os.path.join(base_path, split, 'labels')
        print(f'{split}: {len(os.listdir(img_dir))} images, {len(os.listdir(lbl_dir))} labels')

get_info(base_path)


label_files = glob.glob(f"{base_path}/train/labels/*.txt")

# Show number of object in created dataset
def number_of_object(label_files):
    object_counts = []
    for file in label_files:
        with open(file) as f:
            lines = f.readlines()
            object_counts.append(len(lines))
    
    plt.hist(object_counts, bins=20, color='skyblue')
    plt.title("Number of objects in the image (TRAIN)")
    plt.xlabel("Object")
    plt.ylabel("Number of image")
    plt.grid(True)
    plt.show()

number_of_object(label_files)


# Show distribution with height/width 
def show_image_destribution():
    widths, heights = [], []
    
    for file in label_files:
        with open(file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, _, _, w, h = parts
                    widths.append(float(w))
                    heights.append(float(h))
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(widths, bins=30, color='orange')
    plt.title("Width distribution")
    plt.xlabel("Width (normal)")
    
    plt.subplot(1,2,2)
    plt.hist(heights, bins=30, color='green')
    plt.title("Height distribution")
    plt.xlabel("Height (normal)")
    
    plt.tight_layout()
    plt.show()

show_image_destribution()


# Show image with labels
def show_example_image(base_path, img):
    img_path = os.path.join(base_path, "train/images")
    lbl_path = os.path.join(base_path, "train/labels")
    
    example_image = sorted(os.listdir(img_path))[img]
    img_file = os.path.join(img_path, example_image)
    label_file = os.path.join(lbl_path, example_image.replace(".jpg", ".txt").replace(".png", ".txt"))
    
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = image.shape
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    with open(label_file) as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            x_min = (x - w/2) * w_img
            y_min = (y - h/2) * h_img
            rect = patches.Rectangle((x_min, y_min), w*w_img, h*h_img, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, f"Class: {int(cls)}", color='yellow', fontsize=8)
    
    plt.title(f"Example: {example_image}")
    plt.axis('off')
    plt.show()

show_example_image(base_path, img=30)


# We need to check image on errors
def check_errors():
    invalid_labels = 0
    for file in label_files:
        with open(file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"⚠️ Not correct row: {file} → {line}")
                    invalid_labels += 1
                else:
                    _, x, y, w, h = map(float, parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"⚠️ Going beyond boundaries: {file} → {line}")
                        invalid_labels += 1
    
    if invalid_labels == 0:
        print("All normal.")

check_errors()


# # Tune hyperparameters on data for 100 epochs
# results = model.tune(
#     data="data.yaml",
#     epochs=350,
#     imgsz=640,
#     iterations=1,  
#     optimizer="Adam", 
#     plots=True
#             )


results = model.train(
                        data="data.yaml", 
                        epochs=350, 
                        imgsz=640, 
                        lr0=0.01,
                        lrf=0.01,
                        momentum=0.937,
                        weight_decay=0.0005,
                        warmup_epochs=3.0,
                        warmup_momentum=0.8,
                        box=7.5,
                        cls=0.5,
                        dfl=1.5,
                        hsv_h=0.015,
                        hsv_s=0.7,
                        hsv_v=0.4,
                        degrees=0.0,
                        translate=0.1,
                        scale=0.5,
                        shear=0.0,
                        perspective=0.0,
                        flipud=0.0,
                        fliplr=0.5,
                        bgr=0.0,
                        mosaic=1.0,
                        mixup=0.0,
                        cutmix=0.0,
                        copy_paste=0.0,
                        name="final_model_tuned"
                        )


# Evaluate the model's performance on the validation set
# Assuming 'results' is the output from model.val()
results = model.val()

metrics = results.results_dict  


# Create results in DataFrame
df = pd.DataFrame([metrics])
print(f"Score mAP@50: {df.at[0, 'metrics/mAP50(B)']}")


Image(filename='/kaggle/working/runs/detect/final_model_tuned/confusion_matrix.png', width=500)


Image(filename='/kaggle/working/runs/detect/final_model_tuned/results.png', width=500)


img_path = ["/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images/IMG_8324.jpg",
            "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images/IMG_8265.jpg",
           "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images/IMG_8237.jpg"]

def show_results(img_path):
    for img in img_path:
        results = model.predict(img, verbose=False, conf=0.9, augment=True)

        str_split = img.split('/')[7]
        
        res_plot = results[0].plot()  
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Predict YOLO with frames and accuracy on {str_split}")
        plt.show()

show_results(img_path)


test_images_path = Path("/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images")
output_dir = Path("/kaggle/working/predictions/labels")
conf_thresh = 0.9

def save_yolo_predictions_with_tta_wbf(test_images_dir: str, output_dir: str, conf_thresh: float):
    test_dir = Path(test_images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_path in test_dir.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        results = model.predict(img_path, conf=conf_thresh, augment=True, verbose=False)

        boxes_list, scores_list, labels_list = [], [], []

        for r in results:
            boxes = r.boxes.xywhn.cpu().numpy() if r.boxes is not None else []
            scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
            labels = r.boxes.cls.cpu().numpy() if r.boxes is not None else []

            boxes_list.append(boxes.tolist())
            scores_list.append(scores.tolist())
            labels_list.append(labels.tolist())

        if not boxes_list or not any(boxes_list):
            continue

        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=0.5, skip_box_thr=conf_thresh
        )

        img_h, img_w = results[0].orig_shape
        detections = []
        for cls, score, (x, y, w, h) in zip(labels_fused, scores_fused, boxes_fused):
            detections.append(f"{int(cls)} {score:.6f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        if detections:
            (output_path / f"{img_path.stem}.txt").write_text("\n".join(detections))

    print(f"All WBF+TTA detections saved to: {output_path}")

save_yolo_predictions_with_tta_wbf(test_images_path, output_dir, conf_thresh)


def predictions_to_csv(
    preds_folder: str = "/kaggle/working/predictions/labels", 
    output_csv: str = "/kaggle/working/submission.csv", 
    test_images_folder: str = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images",
    allowed_extensions: tuple = (".jpg", ".png", ".jpeg")
):
    preds_path = Path(preds_folder)
    test_images_path = Path(test_images_folder)

    test_image_ids = {p.stem for p in test_images_path.iterdir() if p.suffix.lower() in allowed_extensions}
    
    predictions = []

    for txt_file in preds_path.glob("*.txt"):
        image_id = txt_file.stem
        with open(txt_file, "r") as f:
            lines = [line.strip() for line in f if len(line.strip().split()) == 6]
        prediction = " ".join(lines) if lines else "no boxes"
        predictions.append({"image_id": image_id, "prediction_string": prediction})

    predicted_ids = {p["image_id"] for p in predictions}
    missing_ids = test_image_ids - predicted_ids
    predictions.extend({"image_id": mid, "prediction_string": "no boxes"} for mid in missing_ids)

    pd.DataFrame(predictions).to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    
    print(f"Submission saved to {output_csv}")

predictions_to_csv()


! rm -r /kaggle/working/dataset
! rm -r /kaggle/working/yolo11n.pt
! rm -r /kaggle/working/yolov8n.pt
! rm -r /kaggle/working/predictions
! rm -r /kaggle/working/runs
! rm -r /kaggle/working/data.yaml




