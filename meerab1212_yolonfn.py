import os
import pandas as pd
import numpy as np
import pydicom
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2
import glob

!pip install iterative-stratification
!pip install pgzip
!pip install git+https://github.com/ultralytics/ultralytics.git@main
!unzip -q /kaggle/input/lsdc-gen-yolo-data-nfn/data_fold0.zip

#ls

#data_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/"

df = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')
#df = train_data
df = df.fillna('Unknown')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)

fold = 0
for train_index, test_index in mskf.split(df, df.iloc[:,1:]):
    df.loc[test_index, 'fold'] = fold
    fold += 1

df['fold'] = df['fold'].astype(int)
df[['study_id', 'fold']].to_csv('5folds.csv', index=False)


ROOT_DIR=  "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/"
IMG_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
FOLD = 0
OD_INPUT_SIZE = 384
STD_BOX_SIZE = 20
BATCH_SIZE = 32
EPOCHS = 250

SAMPLE = None
CONDITIONS = ['Left Subarticular Stenosis', 'Right Subarticular Stenosis']
SEVERITIES = ['Normal/Mild', 'Moderate', 'Severe']
LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

DATA_DIR = f'data_fold{FOLD}'

train_val_df = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')
train_xy = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')
train_des = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')

if SAMPLE:
    train_val_df = train_val_df.sample(SAMPLE, random_state=2698)

fold_df = pd.read_csv('5folds.csv')
test_df = fold_df[fold_df.fold == FOLD]

train_xy.head(3)

def get_level(text):
    for lev in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
        if lev in text:
            split = lev.split('_')
            split[0] = split[0].capitalize()
            split[1] = split[1].capitalize()
            return '/'.join(split)
    raise ValueError('Level not found '+ lev)
    
def get_condition(text):
    split = text.split('_')
    for i in range(len(split)):
        split[i] = split[i].capitalize()
    split = split[:-2]
    return ' '.join(split)
#     raise ValueError('Condition not found '+ lev)
train_xy['condition'].unique()

#train_df = train_val_df.dropna()
label_df = {'study_id':[], 'condition': [], 'level':[], 'label':[]}

for i, row in train_val_df.iterrows():
    study_id = row['study_id']
    for k, label in row.iloc[1:].to_dict().items():
        level = get_level(k)
        condition = get_condition(k)
        label_df['study_id'].append(study_id)
        label_df['condition'].append(condition)
        label_df['level'].append(level)
        label_df['label'].append(label)
#         break
#     break

label_df = pd.DataFrame(label_df)
label_df = label_df.merge(fold_df, on='study_id')
train_xy = train_xy.merge(train_des, how='inner', on=['study_id', 'series_id'])
label_df = label_df.merge(train_xy, how='inner', on=['study_id', 'condition', 'level'])
def query_train_xy_row(study_id, series_id=None, instance_num=None):
    if series_id is not None and instance_num is not None:
        return label_df[(label_df.study_id==study_id) & (label_df.series_id==series_id) &
            (label_df.instance_number==instance_num)]
    elif series_id is None and instance_num is None:
        return label_df[(label_df.study_id==study_id)]
    else:
        return label_df[(train_xy.study_id==study_id) & (label_df.series_id==series_id)]
def read_dcm(src_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() +1e-6) * 255
    image = np.stack([image]*3, axis=-1).astype('uint8')
    return image

def get_accronym(text):
    split = text.split(' ')
    return ''.join([x[0] for x in split])
# study_id = 4003253 
# series_id = 2448190387
# instance_num = 28

ex = label_df.sample(1).iloc[0]
study_id = ex.study_id
series_id = ex.series_id
instance_num = ex.instance_number

WIDTH = 10

path = os.path.join(IMG_DIR, str(study_id), str(series_id), f'{instance_num}.dcm')
img = read_dcm(path)

tmp_df = query_train_xy_row(study_id, series_id, instance_num)
for i, row in tmp_df.iterrows():
    lbl = f"{get_accronym(row['condition'])}_{row['level']}"
    x, y = row['x'], row['y']
    x1 = int(x - WIDTH)
    x2 = int(x + WIDTH)
    y1 = int(y - WIDTH)
    y2 = int(y + WIDTH)
    color = None
    if row['label'] == 'Normal/Mild':
        color =  (0, 255, 0)
    elif row['label'] == 'Moderate':
        color = (255,255,0) 
    elif row['label'] == 'Severe':
        color = (255,0,0)
        
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, lbl, (x1,y1), fontFace, fontScale, color, thickness, cv2.LINE_AA)

tmp_df

plt.imshow(img)
plt.show()

# label_df[['study_id', 'series_id']].drop_duplicates()
def read_dcm(src_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() +1e-6) * 255
    image = np.stack([image]*3, axis=-1).astype('uint8')
    return image
filtered_df = label_df[label_df.condition.map(lambda x: x in CONDITIONS)]
label2id = {}
id2label = {}
i = 0
for cond in CONDITIONS:
    for level in LEVELS:
        for severity in SEVERITIES:
            cls_ = f"{cond.lower().replace(' ', '_')}_{level}_{severity.lower()}"
            label2id[cls_] = i
            id2label[i] = cls_
            i+=1
id2label

train_df = filtered_df[filtered_df.fold != FOLD]
val_df = filtered_df[filtered_df.fold == FOLD]

_IM_DIR = f'{DATA_DIR}/images/train'
_ANN_DIR = f'{DATA_DIR}/labels/train'
name = np.random.choice(os.listdir(_IM_DIR))[:-4]

im = plt.imread(os.path.join(_IM_DIR, name+'.jpg')).copy()
H,W = im.shape[:2]
anns = np.loadtxt(os.path.join(_ANN_DIR, name+'.txt')).reshape(-1, 5)

for _cls, x,y,w,h in anns.tolist():
    x *= W
    y *= H
    w *= W
    h *= H
    x1 = int(x-w/2)
    x2 = int(x+w/2)
    y1 = int(y-h/2)
    y2 = int(y+h/2)
    label = id2label[_cls]
    
#     if _cls == 0:
#         c = (255,0,0)
#     elif _cls == 1:
#         c = (0,255,0)
#     else:
#         c = (255,255,0)
    c = (0,255,255)

    im = cv2.rectangle(im, (x1,y1), (x2,y2), c, 2)
    cv2.putText(im, label, (x1,y1), fontFace, 0.3, c, 1, cv2.LINE_AA)


plt.imshow(im)

# ls data_fold0/labels/val
# os.path.join(_ANN_DIR, name+'.txt')
# cat 'train_fold0/labels/404602713_1230697721_12.txt'
# Install the ultralytics package from GitHub
!pip install git+https://github.com/ultralytics/ultralytics.git@main

for k, v in id2label.items():
    print(f'{k}: {v}')

#ls


#%%writefile yolo_ss.yaml
#path: "/kaggle/working/data_fold0" # dataset root dir
#train: "images/train"  
#val: "images/val" 
#test: "images/val" 
with open("yolo_nfn.yaml", "w") as f:
    f.write('path: "/kaggle/working/data_fold0"\n')
    f.write('train: "images/train"\n')
    f.write('val: "images/val"\n')
    f.write('test: "images/val"\n')
    f.write("names:\n")
    f.write("    0: left_neural_foraminal_narrowing_l1_l2_normal/mild\n")
    f.write("    1: left_neural_foraminal_narrowing_l1_l2_moderate\n")
    f.write("    2: left_neural_foraminal_narrowing_l1_l2_severe\n")
    f.write("    3: left_neural_foraminal_narrowing_l2_l3_normal/mild\n")
    f.write("    4: left_neural_foraminal_narrowing_l2_l3_moderate\n")
    f.write("    5: left_neural_foraminal_narrowing_l2_l3_severe\n")
    f.write("    6: left_neural_foraminal_narrowing_l3_l4_normal/mild\n")
    f.write("    7: left_neural_foraminal_narrowing_l3_l4_moderate\n")
    f.write("    8: left_neural_foraminal_narrowing_l3_l4_severe\n")
    f.write("    9: left_neural_foraminal_narrowing_l4_l5_normal/mild\n")
    f.write("    10: left_neural_foraminal_narrowing_l4_l5_moderate\n")
    f.write("    11: left_neural_foraminal_narrowing_l4_l5_severe\n")
    f.write("    12: left_neural_foraminal_narrowing_l5_s1_normal/mild\n")
    f.write("    13: left_neural_foraminal_narrowing_l5_s1_moderate\n")
    f.write("    14: left_neural_foraminal_narrowing_l5_s1_severe\n")
    f.write("    15: right_neural_foraminal_narrowing_l1_l2_normal/mild\n")
    f.write("    16: right_neural_foraminal_narrowing_l1_l2_moderate\n")
    f.write("    17: right_neural_foraminal_narrowing_l1_l2_severe\n")
    f.write("    18: right_neural_foraminal_narrowing_l2_l3_normal/mild\n")
    f.write("    19: right_neural_foraminal_narrowing_l2_l3_moderate\n")
    f.write("    20: right_neural_foraminal_narrowing_l2_l3_severe\n")
    f.write("    21: right_neural_foraminal_narrowing_l3_l4_normal/mild\n")
    f.write("    22: right_neural_foraminal_narrowing_l3_l4_moderate\n")
    f.write("    23: right_neural_foraminal_narrowing_l3_l4_severe\n")
    f.write("    24: right_neural_foraminal_narrowing_l4_l5_normal/mild\n")
    f.write("    25: right_neural_foraminal_narrowing_l4_l5_moderate\n")
    f.write("    26: right_neural_foraminal_narrowing_l4_l5_severe\n")
    f.write("    27: right_neural_foraminal_narrowing_l5_s1_normal/mild\n")
    f.write("    28: right_neural_foraminal_narrowing_l5_s1_moderate\n")
    f.write("    29: right_neural_foraminal_narrowing_l5_s1_severe\n")
   
with open('yolo_nfn.yaml', 'r') as f:
    print(f.read())

import wandb
from wandb.integration.ultralytics import add_wandb_callback

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandbkey")
wandb.login(key=secret_value_0)



# Initialize W&B run
wandb.init(
    project="lsdc_yolov8nfn",
#     name=f"Demo_fold0",
#     tags=["baseline", "search-lr", ],
    group=";".join(CONDITIONS),
#     config={
#         "lr": LR,
#         "model-name":"xtremedistill-trim",
#         "dataset": [
#             "raw_compettion",
#             "MPWare",
#             "Nicholas"
#         ]
#     }
)

from ultralytics import YOLO


# Initialize YOLO Model
model = YOLO("yolov8s.pt")

# Add W&B callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train/fine-tune your model
# At the end of each epoch, predictions on validation batches are logged
# to a W&B table with insightful and interactive overlays for
# computer vision tasks
model.train(project="lsdc_yolov8nfn", data="yolo_nfn.yaml", 
            epochs=EPOCHS, imgsz=OD_INPUT_SIZE, batch=BATCH_SIZE)

# Finish the W&B run
wandb.finish()

from collections import defaultdict
# # test generated annotations

_IM_DIR = f'{DATA_DIR}/images/val'
_ANN_DIR = f'{DATA_DIR}/labels/val'
name = np.random.choice(os.listdir(_IM_DIR))[:-4]

path = os.path.join(_IM_DIR, name+'.jpg')

im = plt.imread(path).copy()
H,W = im.shape[:2]
anns = np.loadtxt(os.path.join(_ANN_DIR, name+'.txt')).reshape(-1, 5)

for _cls, x,y,w,h in anns.tolist():
    x *= W
    y *= H
    w *= W
    h *= H
    x1 = int(x-w/2)
    x2 = int(x+w/2)
    y1 = int(y-h/2)
    y2 = int(y+h/2)
    label = id2label[_cls]
    print(label)
    
#     if _cls == 0:
#         c = (255,0,0)
#     elif _cls == 1:
#         c = (0,255,0)
#     else:
#         c = (255,255,0)
    c = (0,255,255)

    im = cv2.rectangle(im, (x1,y1), (x2,y2), c, 2)
    cv2.putText(im, label, (x1,y1), fontFace, 0.3, c, 1, cv2.LINE_AA)


plt.imshow(im)


# Initialize YOLO Model
model = YOLO(glob.glob("lsdc_yolov8nfn/*/weights/best.pt")[0])

# Add W&B callback for Ultralytics
# add_wandb_callback(model, enable_model_checkpointing=True)

# Perform prediction which automatically logs to a W&B Table
# with interactive overlays for bounding boxes, segmentation masks
out = model.predict([path], save=True, conf=0.2)

# Finish the W&B run
wandb.finish()

im = plt.imread(glob.glob(f'{out[0].save_dir}/*.jpg')[0])
plt.imshow(im)


