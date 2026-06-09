# !pip install -q gymnasium >= 1.1.0
!pip install -q ultralytics ensemble-boxes opencv-python==4.10.0.84


import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

import os
import glob
import yaml
import random


from pathlib import Path
import csv
from PIL import Image


Train_imgs = '/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/no_clutter_10/train/images/'
Val_imgs = '/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/no_clutter_10/val/images/'


plt.figure(figsize=(20,12))
ls = os.listdir(Train_imgs)
c = 1
for i in ls:
    img = plt.imread(Train_imgs+i)
    plt.subplot(2,4,c)
    plt.title(i)
    plt.imshow(img)
    c+=1


def color_mask(img, lower_bond, upper_bond):
    mask = cv2.inRange(img, lower_bond, upper_bond)
    return mask

def apply_red_mask(img):
    lower_red1 = np.array([0, 50, 15], np.uint8)
    # upper_red1 = np.array([10, 250, 255], np.uint8)
    upper_red1 = np.array([7, 250, 255], np.uint8)
    mask1 = color_mask(img, lower_red1, upper_red1)

    # lower_red2 = np.array([0, 190, 0], np.uint8)
    lower_red2 = np.array([0, 210, 0], np.uint8)
    upper_red2 = np.array([180, 255, 255], np.uint8)
    mask2 = color_mask(img, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask


def apply_white_mask(img):
    lower_white1 = np.array([0, 0, 0], np.uint8)
    upper_white1 = np.array([0, 90, 255], np.uint8)
    mask1 = color_mask(img, lower_white1, upper_white1)

    lower_white2 = np.array([0, 0, 0], np.uint8)
    # upper_white2 = np.array([14, 90, 255], np.uint8)
    upper_white2 = np.array([14, 80, 255], dtype=np.uint8)
    mask2 = color_mask(img, lower_white2, upper_white2)

    white_mask = cv2.bitwise_xor(mask1, mask2)
    return white_mask


def apply_refine_mask(mask, k=(7,7), img=None, min_area = 5000):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
    clean1 =  cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean2 =  cv2.morphologyEx(clean1, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centriods = cv2.connectedComponentsWithStats(clean2, connectivity=4,  ltype = cv2.CV_32S)
    mask = np.zeros_like(clean2)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels==i] = 255
    return mask


def apply_errode_mask(mask, k =(7, 7)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
    mask =  cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    return mask

def apply_dilate_mask(mask, k =(7, 7)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
    mask =  cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask


def apply_contor_box(img,mask, min_area = 5000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cans = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area <= 100000:
            x, y, w, h = cv2.boundingRect(cnt)   
            aspect_ratio = h /w
            if 0.54 <= aspect_ratio < 3.0:
                rec = cv2.minAreaRect(cnt)
                box_points = cv2.boxPoints(rec)
                box_points = np.int_(box_points)
                cv2.drawContours(img, [box_points], -1, (250, 255, 215), 10)
                cans.append((x, y, x+w, y+h))
            else:
                print(area, aspect_ratio)
    
    return cans
    


c=0

fig, axs = plt.subplots(len(ls), 6, figsize=(30,23))
for img in ls:
    img_path = Train_imgs+img
    image = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axs[c,0].set_title(img)
    axs[c,0].imshow(img_rgb)

    smoothed_img = cv2.bilateralFilter(image, 9, 75, 75)
    hsv = cv2.cvtColor(smoothed_img, cv2.COLOR_BGR2HSV)

    red_mask = apply_red_mask(hsv)
    axs[c,1].set_title('Red Mask - '+img)
    axs[c,1].imshow(red_mask, cmap='gray')

    white_mask = apply_white_mask(hsv)
    axs[c,2].set_title('White Mask - '+img)
    axs[c,2].imshow(white_mask, cmap='gray')

    color_masked = cv2.bitwise_or(red_mask, white_mask)
    axs[c,3].set_title('Combine Red-White Mask')
    axs[c,3].imshow(color_masked, cmap='gray')

    mask = apply_refine_mask(color_masked, k=(5, 5), min_area=900)
    axs[c,4].set_title('Refine Mask')
    axs[c,4].imshow(mask, cmap='gray')

    # # errode_mask = apply_errode_mask(mask, (5, 5))
    # dilate_mask = apply_dilate_mask(mask, (5, 5))
    # # axs[c,5].set_title(' Erode & Dilate Mask')
    # # axs[c,5].imshow(dilate_mask, cmap='grey')

    # cans = apply_contor_box(img_rgb, mask)
    # axs[c,5].set_title('Contor Box')
    # axs[c,5].imshow(img_rgb)

    dilate_mask = apply_dilate_mask(mask, (5, 5))
    cans = apply_contor_box(img_rgb,dilate_mask, min_area=1500)
    axs[c,5].set_title('Contor Box')
    axs[c,5].imshow(img_rgb)
    
    print(f"--{img}--")

    c = c+1


path = '/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset'
train_folder = ['large_plant_10', 'film_grain_10_half_clutter', 'table_close_10', 'couch_far_10', 'no_clutter_10', 'far_10_half_clutter','clutter']
train_imgs = [os.path.join(path, x,'train/images') for x in train_folder]
train_labels = [os.path.join(path, x,'train/labels') for x in train_folder]
val_imgs = [os.path.join(path, x,'val/images') for x in train_folder]
val_labels = [os.path.join(path, x,'val/labels') for x in train_folder]
TEST_imgs = os.path.join(path, "TestImages/images")


train_imgs, train_labels, val_imgs, TEST_imgs


import os
import shutil

def copy_folder_and_rename_duplicates(src_folder, dest_folder):
    """
    Copies a source folder to a destination folder, renaming duplicate files
    by appending a counter to their base name.

    Args:
        src_folder (str): The path to the source folder.
        dest_folder (str): The path to the destination folder.
    """
    if not os.path.exists(src_folder):
        print(f"Source folder '{src_folder}' does not exist.")
        return

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created destination folder '{dest_folder}'.")

    for root, _, files in os.walk(src_folder):
        relative_path = os.path.relpath(root, src_folder)
        current_dest_dir = os.path.join(dest_folder, relative_path)

        if not os.path.exists(current_dest_dir):
            os.makedirs(current_dest_dir)

        for filename in files:
            src_file_path = os.path.join(root, filename)
            dest_file_path = os.path.join(current_dest_dir, filename)

            if os.path.exists(dest_file_path):
                base_name, extension = os.path.splitext(filename)
                counter = 1
                new_filename = f"{base_name}_{counter}{extension}"
                new_dest_file_path = os.path.join(current_dest_dir, new_filename)

                while os.path.exists(new_dest_file_path):
                    counter += 1
                    new_filename = f"{base_name}_{counter}{extension}"
                    new_dest_file_path = os.path.join(current_dest_dir, new_filename)
                
                shutil.copy2(src_file_path, new_dest_file_path)
                # print(f"Copied '{filename}' to '{new_filename}' in '{current_dest_dir}'.")
            else:
                shutil.copy2(src_file_path, dest_file_path)
                # print(f"Copied '{filename}' to '{current_dest_dir}'.")



dest_dir = "/kaggle/working/Started_Dataset"
for t in train_imgs:
    copy_folder_and_rename_duplicates(t, dest_dir+"/train/images")

for t in train_labels:
    copy_folder_and_rename_duplicates(t, dest_dir+"/train/labels")

for t in val_imgs:
    copy_folder_and_rename_duplicates(t, dest_dir+"/val/images")

for t in val_labels:
    copy_folder_and_rename_duplicates(t, dest_dir+"/val/labels")

# len(os.listdir("/kaggle/working/Started_Dataset/train/images"))


data = {
    "train": "/kaggle/working/Started_Dataset/train/images",
    "val": "/kaggle/working/Started_Dataset/val/images",
    "test": TEST_imgs,
    "nc": 1,
    "names": ["Soup"]
}
data


with open(os.path.join('/kaggle/working','yolo_params.yaml'), 'w') as file:
    yaml.dump(data, file)


def visual_image_box(img_paths, label_paths, extensions = ('.png', '.jpg', '.jpeg')):
    img_files = [file for file in glob.glob(f"{img_paths}/**/*", recursive=True) if file.lower().endswith(extensions)]
    plt.figure(figsize=(30,23))
    for i, img_path in enumerate(img_files[40:50]):
        label_path = os.path.join(label_paths,img_path.split("/")[-1].split(".")[0]+".txt")
        # print(img_path,"\n",label_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    # convert YOLO -> pixel coords
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1,10,i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(img_path))
    plt.show()


## Display sample image with bounding boxes using YOLO annotation file
visual_image_box("/kaggle/working/Started_Dataset/train/images","/kaggle/working/Started_Dataset/train/labels")


# data_yaml = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/yolo_params.yaml"
model_8m = YOLO("yolov8m.pt")
model_11m = YOLO("yolo11m.pt")
data_yaml = "/kaggle/working/yolo_params.yaml"


results_8m = model_8m.train(
    data=data_yaml,
    epochs=200,
    batch=8,
    imgsz=920,
    device=[0,1],
    patience=20,
    lr0=0.0001,
    lrf=0.001,
    optimizer="Adam",
    weight_decay=0.003,
    cos_lr=True,
    dropout=0.5,
    label_smoothing=0.1,
    mosaic=0.7,
    mixup=0.15,
    copy_paste=0.1,
    fliplr=0.4,
    flipud=0.2,
    hsv_h=0.001,
    hsv_s=0.1,
    hsv_v=0.1,
    translate=0.2,
    scale=0.5,
    shear=0.2,
    perspective=0.0002,
    val=True,
    workers=8,
    seed=61
)


results_11m = model_11m.train(
    data=data_yaml,
    epochs=200,
    batch=4,
    imgsz=1056,
    device=-1,
    patience=200,
    lr0=0.00012,
    lrf=0.0001,
    optimizer="AdamW",
    weight_decay=0.003,
    cos_lr=True,
    dropout=0.01,
    label_smoothing=0.1,
    mosaic=0.5,
    mixup=0.15,
    copy_paste=0.1,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.001,
    hsv_s=0.1,
    hsv_v=0.15,
    degrees = 10,
    translate=0.2,
    scale=0.25,
    shear=0.001,
    erasing=0.05,
    perspective=0.0002,
    val=True,
    workers=8,
    seed=61,
    save=True
)


model_best_8m = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")
model_last_8m = YOLO("/kaggle/working/runs/detect/train/weights/last.pt")


model_best_11m = YOLO("/kaggle/working/runs/detect/train2/weights/best.pt")
model_last_11m = YOLO("/kaggle/working/runs/detect/train2/weights/last.pt")


predict = model_best_8m.predict(TEST_imgs+'/IMG_9682.jpg',conf=0.3)[0].plot() #IMG_9682, IMG_9602, IMG_9617 IMG_9782
plt.imshow(predict)


predict = model_best_11m.predict(TEST_imgs+'/IMG_9682.jpg',conf=0.3)[0].plot() #IMG_9682, IMG_9602, IMG_9617 IMG_9782
plt.imshow(predict)


predict = model_last_11m.predict(TEST_imgs+'/IMG_9682.jpg',conf=0.3)[0].plot() #IMG_9682, IMG_9602, IMG_9617 IMG_9782
plt.imshow(predict)


## Load User defined model - Model trained on custom dataset
model_paths = ["/kaggle/working/runs/detect/train/weights/best.pt",
               "/kaggle/working/runs/detect/train2/weights/best.pt",
               "/kaggle/working/runs/detect/train2/weights/last.pt"]
models = [YOLO(path) for path in model_paths]


def filter_invalid_boxes(boxes, scores, labels):
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    for b, s, l in zip(boxes, scores, labels):
        if abs(b[2] - b[0]) > 1e-6 and abs(b[3] - b[1]) > 1e-6:
            filtered_boxes.append(b)
            filtered_scores.append(s)
            filtered_labels.append(l)
    return filtered_boxes, filtered_scores, filtered_labels
    
def run_inference(models, image_sizes, test_images_path):
    image_paths = [p for p in Path(test_images_path).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    predictions = {}

    for model_idx, model in enumerate(models):
        model.eval()
        predictions[model_idx] = {}
        for size in image_sizes:
            predictions[model_idx][size] = {}
            pred = []
            for img_path in image_paths:
                image_id = img_path.stem
                image = Image.open(img_path)
                img_width, img_height = image.size

                results = model.predict(source=str(img_path), conf=0.3,iou=0.4, max_det=600, augment=True, imgsz=size, verbose=False)
                boxes, scores, labels = [], [], []

                for result in results:
                    if result.boxes is None:
                        continue
                    boxes = result.boxes.xyxy.cpu().numpy().tolist()
                    scores = result.boxes.conf.cpu().numpy().tolist()
                    labels = result.boxes.cls.cpu().numpy().tolist()

                    norm_boxes = [
                        [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height]
                        for x1, y1, x2, y2 in boxes
                    ]
                    norm_boxes, scores, labels = filter_invalid_boxes(norm_boxes, scores, labels)

                predictions[model_idx][size][image_id] = {
                    "boxes": norm_boxes,
                    "scores": scores,
                    "labels": labels
                }
                
                if boxes:
                    prediction_string = " ".join(
                        f"{int(lbl)} {score:.6f} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {(b[2]-b[0]):.6f} {(b[3]-b[1]):.6f}"
                        for b, score, lbl in zip(norm_boxes, scores, labels)
                    )
                else:
                    prediction_string = "no boxes"

                pred.append({
                    "image_id": image_id,
                    "prediction_string": prediction_string
                })

            # Save CSV per model and size
            df = pd.DataFrame(pred)
            csv_path = f"submission_{model_idx}_{size}.csv"
            df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"[saved] {csv_path}")
            print(df.head(10))

    return predictions

def apply_wbf_and_save_final_submission(predictions, image_ids, test_label, output_path="submission_wbf.csv"):
    wbf_results = []
    os.makedirs(test_label, exist_ok=True)
    for image_id in image_ids:
        all_boxes, all_scores, all_labels = [], [], []

        for model_preds in predictions.values():
            for size_preds in model_preds.values():
                if image_id not in size_preds:
                    continue
                pred = size_preds[image_id]
                if not pred["boxes"]:
                    continue
                all_boxes.append(pred["boxes"])
                all_scores.append(pred["scores"])
                all_labels.append(pred["labels"])

        if not all_boxes:
            pred_str = "no boxes"
        else:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr=0.5, skip_box_thr=0.01)
            write_labels(fused_boxes, fused_scores, fused_labels, image_id, test_label)
            pred_str = " ".join(
                f"{int(lbl)} {score:.6f} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {(b[2]-b[0]):.6f} {(b[3]-b[1]):.6f}"
                for b, score, lbl in zip(fused_boxes, fused_scores, fused_labels)
            )

        wbf_results.append({
            "image_id": image_id,
            "prediction_string": pred_str
        })

    wbf_df = pd.DataFrame(wbf_results)
    wbf_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[notice] ✅ WBF submission saved to {output_path}")
    print(wbf_df.head(10))

def write_labels(fused_boxes, fused_scores, fused_labels, filename, output_dir):
    output_txt = os.path.join(output_dir,filename+'.txt')
    with open(output_txt, "w") as f:
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            if score >= 0.65:
                f.write(f"{label} {score:.6f} {(box[0]+box[2])/2:.6f} {(box[1]+box[3])/2:.6f} {(box[2]-box[0]):.6f} {(box[3]-box[1]):.6f}\n")


test_label = os.path.join("/kaggle/working","predictions","labels")
# predictions = run_inference(models, [1056, 1248, 1920, 2560], TEST_imgs)
predictions = run_inference(models, [1056, 1248], TEST_imgs)
image_ids = list(next(iter(next(iter(predictions.values())).values())).keys())
apply_wbf_and_save_final_submission(predictions, image_ids, test_label = test_label)


def display_predicted_images(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    selected_images = random.sample(image_files, min(5, len(image_files)))
    fig, axes = plt.subplots(1, len(selected_images), figsize=(15, 5))
    for ax, img_file in zip(axes, selected_images):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) <= 5:
                        continue
                    class_id, conf, x_center, y_center, box_w, box_h = map(float, parts)
                    # Convert normalized to pixel coordinates
                    x1 = int((x_center - box_w / 2) * w)
                    y1 = int((y_center - box_h / 2) * h)
                    x2 = int((x_center + box_w / 2) * w)
                    y2 = int((y_center + box_h / 2) * h)

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 12)
                    cv2.putText(img, f'{conf:.2f}', (x1, y1+1),cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255, 0), 15)
                    
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(img_file)

    plt.tight_layout()
    plt.show()


display_predicted_images(TEST_imgs,test_label)


## Assuming all the Test images has one or more cans, check if any object detection missed by Yolo

image_files = [f for f in os.listdir(TEST_imgs) if f.lower().endswith(('.jpg', '.png'))]

predict_data = []
for file in image_files:
    file =  os.path.splitext(file)[0]
    label_path = os.path.join(test_label, file + '.txt')
    try:
        lines = [l.strip() for l in open(label_path) if len(l.strip().split()) == 6]
    except:
        lines = []
    predict_data.append({"image_id": file, "prediction_string": " ".join(lines) if lines else "no boxes"})

no_box = [i for i in predict_data if i['prediction_string'] == "no boxes"]
print(no_box)


if len(no_box) == 0:
    df = pd.DataFrame(predict_data)
    print(df.head())
    df.to_csv("submission.csv", index=False)
else:
    print("Fine-tune model, Validate data")




