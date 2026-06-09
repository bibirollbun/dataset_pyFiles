!pip install ultralytics
# !pip install -e git+https://github.com/troobadure/ultralytics@shibarinu\#egg=ultralytics
# !pip install -e git+https://github.com/ultralytics/ultralytics\#egg=ultralytics --exists-action s


import numpy as np
import pandas as pd
import os
import random
import shutil

import cv2
import pydicom
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ultralytics import YOLO
import wandb

np.random.seed(1337)


try:
    shutil.rmtree('/kaggle/working/')
except:
    pass


CSV_FILE = '../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
TRAIN_SRC_DIR = '../input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
TEST_SRC_DIR = '../input/rsna-pneumonia-detection-challenge/stage_2_test_images/'
DATASET_DIR = './dataset/'
TEST_IMG_DIR = 'test_images/'

os.mkdir(DATASET_DIR)
os.mkdir(DATASET_DIR + 'images/')
os.mkdir(DATASET_DIR + 'images/train/')
os.mkdir(DATASET_DIR + 'images/val/')
os.mkdir(DATASET_DIR + 'images/test/')
os.mkdir(DATASET_DIR + 'labels/')
os.mkdir(DATASET_DIR + 'labels/train/')
os.mkdir(DATASET_DIR + 'labels/val/')
os.mkdir(DATASET_DIR + 'labels/test/')
os.mkdir(TEST_IMG_DIR)


annotations = pd.read_csv(CSV_FILE)
print(annotations.info())
annotations.head()


positive_annotations = annotations[annotations.Target == 1]
negative_annotations = annotations[annotations.Target == 0]

print(positive_annotations['patientId'].drop_duplicates().shape[0])
print(negative_annotations['patientId'].drop_duplicates().shape[0])
print(negative_annotations['patientId'].shape[0])

negative_sample = negative_annotations.sample(600)
negative_sample['patientId'].shape[0]

annotations = pd.concat([positive_annotations, negative_sample])
print(annotations.shape)
annotations.head()


patient_id_series = annotations.patientId.drop_duplicates()
print('Number of images:', patient_id_series.size)

train_series, val_series = train_test_split(patient_id_series, test_size=0.1, random_state=42)
print('Train set number:', len(train_series))
print('Validation set number:', len(val_series))


for patient_id in tqdm(train_series):
    src_path = TRAIN_SRC_DIR + patient_id + '.dcm'
    dcm_data = pydicom.dcmread(src_path)
    image_array = dcm_data.pixel_array
    image = Image.fromarray(image_array)
    image.save(DATASET_DIR + 'images/train/' + patient_id + '.jpg')
print('Images moved to train folder:', len(os.listdir(DATASET_DIR + 'images/train/')))
    
for patient_id in tqdm(val_series):
    src_path = TRAIN_SRC_DIR + patient_id + '.dcm'
    dcm_data = pydicom.dcmread(src_path)
    image_array = dcm_data.pixel_array
    image = Image.fromarray(image_array)
    image.save(DATASET_DIR + 'images/val/' + patient_id + '.jpg')
print('Images moved to val folder:', len(os.listdir(DATASET_DIR + 'images/val/')))


def translate_bbox(bbox):
    img_size = 1024 # rsna defualt image size
    
    top_left_x = bbox[0]
    top_left_y = bbox[1]
    absolute_w = bbox[2]
    absolute_h = bbox[3]

    relative_w = absolute_w / img_size
    relative_h = absolute_h / img_size
    
    relative_x = top_left_x / img_size + relative_w / 2
    relative_y = top_left_y / img_size + relative_h / 2
    
    return relative_x, relative_y, relative_w, relative_h
    
def revert_bbox(rx, ry, rw, rh):
    img_size = 1024 # rsna defualt image size
    
    x = (rx-rw/2)*img_size
    y = (ry-rh/2)*img_size
    w = rw*img_size
    h = rh*img_size
    
    return x, y, w, h
    
    
def save_label(label_dir, patient_id, bbox):
    label_fp = os.path.join(label_dir, patient_id + '.txt')
    
    f = open(label_fp, "a")
    if (bbox == 'nan').all():
        f.close()
        return
    
    x, y, w, h = translate_bbox(bbox)
    
    line = f"0 {x} {y} {w} {h}\n"
    
    f.write(line)
    f.close()


LABELS_DIR = "./labels_temp/"
os.mkdir(LABELS_DIR)

for row in annotations.values:
    if pd.notna(row[1:5]).all():
        save_label(LABELS_DIR, row[0], row[1:5])
    
for patient_id in train_series:
    if os.path.isfile(LABELS_DIR + patient_id + '.txt'):
        shutil.copy(LABELS_DIR + patient_id + '.txt', DATASET_DIR + 'labels/train/')
    
for patient_id in val_series:
    if os.path.isfile(LABELS_DIR + patient_id + '.txt'):
        shutil.copy(LABELS_DIR + patient_id + '.txt', DATASET_DIR + 'labels/val/')
    
shutil.rmtree(LABELS_DIR)


demo_patient_id = val_series.values[8]
demo_img_path = DATASET_DIR + 'images/val/' + demo_patient_id + '.jpg'
demo_label_path = DATASET_DIR + 'labels/val/' + demo_patient_id + '.txt'

plt.imshow(cv2.imread(demo_img_path))

with open(demo_label_path, "r") as f:
    for line in f:
        print(line)
        class_id, rx, ry, rw, rh = list(map(float, line.strip().split()))
        
        x, y, w, h = revert_bbox(rx, ry, rw, rh)
        plt.plot([x, x, x+w, x+w, x], [y, y+h, y+h, y, y])


%%writefile config.yaml

path: '/kaggle/working/dataset' # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: pneumonia


model = YOLO('yolov8l.pt') # yaml


wandb.login(key='da27b138f0f392e3f931cf71acdab08543ac649c')
results = model.train(data='config.yaml', epochs=10)


def plot_val_pred(demo_patient_id, verbose=True, split='val'):
    demo_img_path = DATASET_DIR + f'images/{split}/' + demo_patient_id + '.jpg'
    demo_label_path = DATASET_DIR + f'labels/{split}/' + demo_patient_id + '.txt'

    res = model(demo_img_path, verbose=verbose)
    if verbose:
        print(res[0].probs)
        print(res[0].boxes.xywh)

    plt.imshow(cv2.imread(demo_img_path))

    img_size = 1014
    if os.path.isfile(demo_label_path):
        with open(demo_label_path, "r") as f:
            for line in f:
                if verbose:
                    print(line)
                class_id, rx, ry, rw, rh = list(map(float, line.strip().split()))

                x, y, w, h = revert_bbox(rx, ry, rw, rh)
                plt.plot([x, x, x+w, x+w, x], [y, y+h, y+h, y, y], c='blue')

                
    for box in res[0].boxes.xywh.cpu():
        px, py, pw, ph = box
        plt.plot([px-pw/2, px-pw/2, px+pw/2, px+pw/2, px-pw/2], [py-ph/2, py+ph/2, py+ph/2, py-ph/2, py-ph/2], c='orange')


def random_value(series):
    return series.iloc[random.randrange(0, len(series))]

def plot_examples(series, rows = 5, cols = 2, split='val'):
    plt.suptitle(split)
    plt.figure(figsize=(10*cols,10*rows))
    for h in range(rows):
        for w in range(cols):
            plt.subplot(rows, cols, h*2+w+1)
            plot_val_pred(random_value(series), verbose=False, split=split)

plot_examples(train_series, 2, 2, 'train')


plot_examples(val_series, 2, 2, 'val')


for file in tqdm(os.listdir(TEST_SRC_DIR)):
    src_path = TEST_SRC_DIR + file
    dcm_data = pydicom.dcmread(src_path)
    image_array = dcm_data.pixel_array
    image = Image.fromarray(image_array)
    image.save(TEST_IMG_DIR + os.path.splitext(file)[0] + '.jpg')


# model = YOLO('../input/yolov8n-trained-on-rsna-pneumonia-detection/best.pt')


results = model(TEST_IMG_DIR, verbose=False, conf=0.28, stream=True) # conf=0.26 gives better score on private


results = model(TEST_IMG_DIR, verbose=False, conf=0.28, stream=True)

def get_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

for result in results:  # Iterate over the generator
    print(get_id_from_path(result.path))
    print(result.boxes.xywh)
    print(result.boxes.conf)
    break  # Remove this if you want to process all images



with open('submission.csv', 'w') as file:
    file.write("patientId,PredictionString\n")

    for result in tqdm(results):
        line = get_id_from_path(result.path) + ','
        
        for conf, xywh in zip(result.boxes.conf, result.boxes.xywh):
            x, y, w, h = xywh
            line += f"{conf:.2f} {x-w/2:.2f} {y-h/2:.2f} {w:.2f} {h:.2f} "
            
        line = line.strip()
        file.write(line+"\n")


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


path = Path('submission.csv')
df = pd.read_csv(path)
print(path.name)

df['pred_count'] = df['PredictionString'].apply(lambda x: 0 if pd.isna(x) else int(len(x.split()) / 5))
df['pred_count'].value_counts().sort_index()


def str_to_boxes(s: str) -> list:  # return c,x,y,w,h
    if pd.isna(s) or len(s) == 0:
        return []

    boxes = []
    n = len(s.split()) // 5
    for i in range(n):
        box = s.split()[i * 5:i * 5 + 5]
        boxes.append(list(map(float, box)))

    return boxes

def remove_empty_boxes(s: str) -> str:
    if pd.isna(s) or len(s) == 0:
        return s

    n = len(s.split()) // 5
    data = s.split()
    for i in range(n):
        box = data[i * 5:i * 5 + 5]
        if float(box[2]) * float(box[3]) == 0:
            s = s.replace(' '.join(map(str, box)), '').strip()

    return s


df['PredictionString'] = df['PredictionString'].apply(remove_empty_boxes)


def box_areas_utils(box1, box2):  # corner coords
    _, left_x1, left_y1, w1, h1 = box1
    _, left_x2, left_y2, w2, h2 = box2

    assert w1 * h1 * w2 * h2 > 0, 'w or h is 0'

    right_x1, right_x2 = left_x1 + w1, left_x2 + w2
    top_y1, top_y2 = left_y1 + h1, left_y2 + h2

    area1, area2 = w1 * h1, w2 * h2
    right_xi = min(right_x1, right_x2)
    left_xi = max(left_x1, left_x2)
    top_yi = min(top_y1, top_y2)
    bottom_yi = max(left_y1, left_y2)

    if right_xi <= left_xi or top_yi <= bottom_yi:
        intersection = 0
    else:
        intersection = (right_xi - left_xi) * (top_yi - bottom_yi)

    union = area1 + area2 - intersection
    return area1, area2, intersection, union


def iou(box1, box2):
    area1, area2, intersection, union = box_areas_utils(box1, box2)
    return intersection / union


def two_boxes_overlap(box1, box2) -> bool:
    return iou(box1, box2) > 0.3


def one_box_inside_another(box1, box2) -> bool:
    area1, area2, intersection, union = box_areas_utils(box1, box2)
    return intersection / area1 > 0.7 or intersection / area2 > 0.7


def merge_boxes(box1, box2) -> (float, float, float, float, float):  # c,x,y,w,h - bottom left corner (0,0)
    c1, x1, y1, w1, h1 = box1
    c2, x2, y2, w2, h2 = box2
    min_x, min_y = min([x1, x2]), min([y1, y2])
    max_x, max_y = max([x1 + w1, x2 + w2]), max([y1 + h1, y2 + h2])

    w, h = max_x - min_x, max_y - min_y
    # reduce w, h by 10%
    dw, dh = w * 0.05, h * 0.05
    w, h = w * 0.9, h * 0.9

    return (c1 + c2) / 2, (x1 + x2) / 2, (y1 + y2) / 2, (w1+w2) / 2, (h1+h2) / 2


def detect_overlapping(s: str, type='both') -> bool:
    if pd.isna(s):
        return False

    n = len(s.split()) // 5
    for i in range(n):
        box1 = list(map(float, s.split()[i * 5:i * 5 + 5]))
        for j in range(n):
            if i == j:
                continue
            box2 = list(map(float, s.split()[j * 5:j * 5 + 5]))

            if type == 'both':
                if two_boxes_overlap(box1, box2) or one_box_inside_another(box1, box2):
                    return True
            elif type == 'overlap':
                if two_boxes_overlap(box1, box2) and not one_box_inside_another(box1, box2):
                    return True
            elif type == 'inside':
                if one_box_inside_another(box1, box2):
                    return True

    return False


def merge_overlapping(s: str) -> str:
    if pd.isna(s):
        return s

    boxes = str_to_boxes(s)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i] is None or boxes[j] is None:
                continue
            if two_boxes_overlap(boxes[i], boxes[j]) or \
                    one_box_inside_another(boxes[i], boxes[j]):
                boxes[i] = merge_boxes(boxes[i], boxes[j])
                boxes[j] = None

    return ' '.join([' '.join(map(str, c)) for c in boxes if c is not None]).strip()



df['overlaps'] = df['PredictionString'].apply(lambda s: detect_overlapping(s, type='overlap'));
df['inside_box'] = df['PredictionString'].apply(lambda s: detect_overlapping(s, type='inside'));
print(df.query('pred_count>0')['overlaps'].value_counts())
print(df.query('pred_count>0')['inside_box'].value_counts())


df[df['overlaps']]['pred_count'].value_counts().sort_index()


df['PredictionString_'] = df['PredictionString']

s = df.query('inside_box').sample(1).iloc[0]

img = Image.open(f'test_images/{s["patientId"]}.jpg')
img = img.convert('RGB')
ax = plt.gca()
boxes = str_to_boxes(s['PredictionString_'])
for b in boxes:
    rect = patches.Rectangle((b[1], b[2]), b[3], b[4], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

fixed_s = merge_overlapping(s['PredictionString_'])
boxes_fixed = str_to_boxes(fixed_s)
for b in boxes_fixed:
    rect = patches.Rectangle((b[1], b[2]), b[3], b[4], linewidth=1, edgecolor='b', facecolor='none', linestyle=':')
    ax.add_patch(rect)
plt.imshow(img);


df['PredictionString'] = df['PredictionString'].apply(merge_overlapping)


df[['patientId', 'PredictionString']].to_csv(path.name, index=False)


import shutil
from IPython.display import FileLink

# Define paths based on the type of model trained
trained_model_path = "/kaggle/working/runs/detect/train2/weights/best.pt"  # Change 'detect' to 'segment' or 'classify' if needed
save_path = "/kaggle/working/best.pt"

# Copy model to Kaggle working directory
shutil.copy(trained_model_path, save_path)

# Provide a download link for manual download
print("Model saved in Kaggle. Click the link below to download:")
display(FileLink(save_path))

# OPTIONAL: Upload to Google Drive via API
try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive

    # Authenticate
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Upload the file to Google Drive
    file = drive.CreateFile({'title': 'yolo_best.pt'})
    file.SetContentFile(save_path)
    file.Upload()

    print("✅ Model successfully uploaded to Google Drive!")
except Exception as e:
    print("⚠️ Google Drive upload failed. Please download manually.")
    print(e)



model = YOLO("/kaggle/working/best.pt")  # Change if saved elsewhere

# Define test image path (update with your actual image path)
image_path = "/kaggle/working/test_images/00632296-ea6d-438e-982d-370b771a6059.jpg"  # Modify as needed

# Perform inference
results = model(image_path, save=True, conf=0.1)  # Adjust confidence if needed

# Ensure results are not empty
if results and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
    print(f"✅ Detected {len(results[0].boxes.xyxy)} objects!")

    # Display image with bounding boxes using OpenCV
    img = cv2.imread(image_path)  # Load original image
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])  # Convert to integers
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
    
    # Convert BGR to RGB for Matplotlib display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Show the image
    plt.imshow(img)
    plt.axis("off")
    plt.show()
else:
    print("⚠️ No objects detected. Try adjusting the confidence threshold or using a different image.")

# Save and print output image path
output_image_path = results[0].save_dir + "/image0.jpg"  # Adjust if needed
print(f"✅ Output saved at: {output_image_path}")

