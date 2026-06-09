!pip install ultralytics


import pandas as pd
import os
import numpy as np
import shutil
import yaml
import matplotlib.pyplot as plt
import random
import cv2

from sklearn import model_selection
from tqdm import tqdm
from glob import glob


size = 512
TRAIN_LABELS_PATH = './vinbigdata/labels/train'
VAL_LABELS_PATH = './vinbigdata/labels/val'
TRAIN_IMAGES_PATH = './vinbigdata/images/train' #12000
VAL_IMAGES_PATH = './vinbigdata/images/val' #3000
External_DIR = f'../input/vinbigdata-{size}-image-dataset/vinbigdata/train' # 15000
os.makedirs(TRAIN_LABELS_PATH, exist_ok = True)
os.makedirs(VAL_LABELS_PATH, exist_ok = True)
os.makedirs(TRAIN_IMAGES_PATH, exist_ok = True)
os.makedirs(VAL_IMAGES_PATH, exist_ok = True)


original_df = pd.read_csv('../input/vinbigdata-chest-xray-abnormalities-detection/train.csv')
number_of_imageids = len(original_df['image_id'].values)
print(f'Total number of image_ids (train + validation) {number_of_imageids}')

number_of_images = len(os.listdir('../input/vinbigdata-chest-xray-abnormalities-detection/train'))
print(f'Total number of images (train + validation) {number_of_images}')

number_of_labels = len(os.listdir('../input/vinbigdata-yolo-labels-dataset/labels'))
print(f'Total number of labels (train + validation) {number_of_labels}')


import os
import cv2
import numpy as np
import pydicom
import multiprocessing
from tqdm import tqdm
from skimage import exposure

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def process_image(dicom_path_output_dir):
    dicom_path, output_dir = dicom_path_output_dir
    file_name = os.path.splitext(os.path.basename(dicom_path))[0]
    image_array = dicom2array(dicom_path)
    equalized_image = exposure.equalize_hist(image_array)
    equalized_image = (equalized_image * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{file_name}.jpeg"), equalized_image)

def saving_image(output_dir, dicom_path_list):
    os.makedirs(output_dir, exist_ok=True)
    dicom_path_output_dir_list = [(path, output_dir) for path in dicom_path_list]

    # Use multiprocessing Pool for parallel processing
    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap(process_image, dicom_path_output_dir_list), total=len(dicom_path_list), desc="Processing Images"))


df = pd.read_csv('/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv')
number_of_images = len(df['image_id'].values)
print(f'Total number of image ids (train + validation) {number_of_images}')

df = df[df.class_id!=14].reset_index(drop = True)
number_of_images = len(df['image_id'].values)
print(f'Total number of image ids after dropping normal images (train + validation) {number_of_images}')

df.head()


df = df.drop(columns=['class_name', 'rad_id', 'x_min', 'x_max', 'y_min', 'y_max',  'class_id']) # we only need image ids, labels are pre-made
df.head()


df_train, df_valid = model_selection.train_test_split(df, test_size=0.15, random_state=42, shuffle=True)


number_of_images = len(df_train['image_id'].values)
print(f'Total number of training image_ids {number_of_images}')

number_of_images = len(df_valid['image_id'].values)
print(f'Total number of validation image_ids {number_of_images}')



# need to delete duplicate image ids, len(labels) should be equal len(df.imageids.values), 


print(f'Total number of training images {len(df_train.image_id.unique())}')
print(f'Total number of validation images {len(df_valid.image_id.unique())}')


def preproccess_data(df, labels_path, images_path):
    for img_id in tqdm(df.image_id.unique()):
        shutil.copy(os.path.join('../input/vinbigdata-yolo-labels-dataset/labels', f"{img_id}"+'.txt'), labels_path)
        shutil.copy(os.path.join(f'/kaggle/input/vinbigdata-{size}-image-dataset/vinbigdata/train', f"{img_id}.png"), images_path)


preproccess_data(df_train, TRAIN_LABELS_PATH, TRAIN_IMAGES_PATH)
preproccess_data(df_valid, VAL_LABELS_PATH, VAL_IMAGES_PATH)


# check that data was preprocessed correctly
print(len(os.listdir(TRAIN_LABELS_PATH)))
print(len(os.listdir(TRAIN_IMAGES_PATH)))

print(len(os.listdir(VAL_LABELS_PATH)))
print(len(os.listdir(VAL_IMAGES_PATH)))


# credit / source https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-train
classes = [ 'Aortic enlargement',
            'Atelectasis',
            'Calcification',
            'Cardiomegaly',
            'Consolidation',
            'ILD',
            'Infiltration',
            'Lung Opacity',
            'Nodule/Mass',
            'Other lesion',
            'Pleural effusion',
            'Pleural thickening',
            'Pneumothorax',
            'Pulmonary fibrosis']

data = dict(
    train =  '../vinbigdata/images/train',
    val   =  '../vinbigdata/images/val',
    nc    = 14,
    names = classes
    )

with open('/kaggle/working/vinbigdata.yaml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(os.path.join( os.getcwd() , 'vinbigdata.yaml'), 'r')
print('\nyaml:')
print(f.read())


import os
from ultralytics import YOLO  
os.environ["WANDB_MODE"] = "dryrun"

model = YOLO('yolo11l.pt')  

model.train(
    data='./vinbigdata.yaml',  
    imgsz=640,                
    epochs=100,               
    device=0                 
)


test_df = pd.read_csv(f'/kaggle/input/vinbigdata-{size}-image-dataset/vinbigdata/test.csv')


test_dir = f'/kaggle/input/vinbigdata-{size}-image-dataset/vinbigdata/test'
os.listdir('/kaggle/working/runs/detect/train/weights/')


weights_dir = "/kaggle/working/runs/detect/train/weights/best.pt"  # Trọng số mô hình đã huấn luyện

model = YOLO(weights_dir)

results = model.predict(
    source=test_dir,
    imgsz=640,
    conf=0.005,
    iou=0.45,
    save_txt=True,
    save_conf=True,
    project="/kaggle/working/runs/detect",  
    name="exp",                            
    exist_ok=True,                         
    save=True,
    verbose=False# Lưu ảnh với bounding box
)

