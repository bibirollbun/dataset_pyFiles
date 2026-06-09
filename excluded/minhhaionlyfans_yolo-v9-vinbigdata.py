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
    
f = open(join( cwd , 'vinbigdata.yaml'), 'r')
print('\nyaml:')
print(f.read())


import os
from ultralytics import YOLO  
os.environ["WANDB_MODE"] = "dryrun"

model = YOLO('yolov9c.pt')  

model.train(
    data='./vinbigdata.yaml',  
    imgsz=640,                
    batch=16,                 
    epochs=30,               
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
    save=True                              # Lưu ảnh với bounding box
)




# credit / source https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-infer
def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height
    
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
    
    return bboxes


len(glob('runs/detect/exp/labels/*txt'))


# credit / source https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-infer
image_ids = []
PredictionStrings = []

def process_submission():
    for file_path in tqdm(glob('runs/detect/exp/labels/*txt')):
        image_id = file_path.split('/')[-1].split('.')[0] # extract image id
        w, h = test_df.loc[test_df.image_id==image_id,['width', 'height']].values[0] #  get the weight & height from  the test df
        f = open(file_path, 'r')  # open the label text file
        data = np.array(f.read().replace('\n', ' ').strip().split(' ')).astype(np.float32).reshape(-1, 6) # move all the labels to the same line..?
        data = data[:, [0, 5, 1, 2, 3, 4]]
        bboxes = list(np.round(np.concatenate((data[:, :2], np.round(yolo2voc(h, w, data[:, 2:]))), axis =1).reshape(-1), 1).astype(str))
        for idx in range(len(bboxes)):
            bboxes[idx] = str(int(float(bboxes[idx]))) if idx%6!=1 else bboxes[idx] # 6 is the length of  the prediction string, so..?
        image_ids.append(image_id)
        PredictionStrings.append(' '.join(bboxes))

    # credit / source: https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class-infer
    pred_df = pd.DataFrame({'image_id':image_ids,
                            'PredictionString':PredictionStrings})
    sub_df = pd.merge(test_df, pred_df, on = 'image_id', how = 'left').fillna("14 1 0 0 1 1")
    sub_df = sub_df[['image_id', 'PredictionString']]
    sub_df.to_csv('/kaggle/working/submission.csv',index = False)
    sub_df.tail()


# !pip uninstall pandas
!pip install -q pandas==1.1.5


process_submission()


import random
from glob import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

files = glob('/kaggle/working/runs/detect/exp/*.[jp][pn]g')  # Tìm cả .jpg và .png
print(f"Số lượng file ảnh: {len(files)}")

def plot_sample_images():
    if not files:
        print("Không tìm thấy file ảnh nào trong '/kaggle/working/runs/detect/exp/'!")
        return
    
    # Số lần hiển thị lưới (tùy chỉnh nếu cần)
    num_grids = min(3, len(files) // 16 + 1)  # Hiển thị tối đa 3 lưới, mỗi lưới 16 ảnh
    
    for _ in range(num_grids):
        row = 4
        col = 4
        num_samples = min(row * col, len(files))  # Giới hạn số mẫu
        grid_files = random.sample(files, num_samples)
        images = []
        
        # Đọc ảnh
        for image_path in tqdm(grid_files):
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            else:
                print(f"Không thể đọc file: {image_path}")
        
        if not images:
            print("Không có ảnh nào để hiển thị trong lượt này!")
            continue
        
        # Tạo lưới ảnh
        fig = plt.figure(figsize=(col * 5, row * 5))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(row, col),
                         axes_pad=0.05)
        
        # Hiển thị ảnh
        for ax, im in zip(grid, images):
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Ẩn các ô trống
        for ax in grid[len(images):]:
            ax.axis('off')
        
        plt.show()




!pip install -q Pillow==4.0.0
!pip install -q PIL
!pip install -q image


plot_sample_images()


import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 2, figsize=(2*5, 3*5), constrained_layout=True)

for row in range(3):
    ax[row][0].imshow(plt.imread(f'/kaggle/working/runs/detect/train/val_batch{row}_labels.jpg'))
    ax[row][0].set_xticks([])
    ax[row][0].set_yticks([])
    ax[row][0].set_title(f'/kaggle/working/runs/detect/train/val_batch{row}_labels.jpg', fontsize=12)
    
    ax[row][1].imshow(plt.imread(f'/kaggle/working/runs/detect/train/val_batch{row}_pred.jpg'))
    ax[row][1].set_xticks([])
    ax[row][1].set_yticks([])
    ax[row][1].set_title(f'/kaggle/working/runs/detect/train/val_batch{row}_pred.jpg', fontsize=12)

plt.show()


plt.figure(figsize=(30,15))
plt.axis('off')
plt.imshow(plt.imread('/kaggle/working/runs/detect/train/confusion_matrix.png'));


plt.figure(figsize=(30,15))
plt.axis('off')
plt.imshow(plt.imread('/kaggle/working/runs/detect/train/results.png'));


plt.figure(figsize=(30,15))
plt.axis('off')
plt.imshow(plt.imread('/kaggle/working/runs/detect/train/F1_curve.png'));


plt.figure(figsize=(30,15))
plt.axis('off')
plt.imshow(plt.imread('/kaggle/working/runs/detect/train/labels.jpg'));


plt.figure(figsize=(30,15))
plt.axis('off')
plt.imshow(plt.imread('//kaggle/working/runs/detect/train/confusion_matrix_normalized.png'));


!pwd


# load yolo submission
yolo = pd.read_csv('/kaggle/working/submission.csv')
effnetb6 = pd.read_csv('/kaggle/input/vinbigdata-2class-prediction/2-cls test pred.csv') # AUC:0.98
pred = pd.merge(yolo, effnetb6, on = 'image_id', how = 'left')
low_thr  = 0.08
high_thr = 0.95


def filter_2cls(row, low_thr=low_thr, high_thr=high_thr):
    prob = row['target']
    if prob<low_thr:
        ## Less chance of having any disease
        row['PredictionString'] = '14 1 0 0 1 1'
    elif low_thr<=prob<high_thr:
        ## More change of having any diesease
        row['PredictionString']+=f' 14 {prob} 0 0 1 1'
    elif high_thr<=prob:
        ## Good chance of having any disease so believe in object detection model
        row['PredictionString'] = row['PredictionString']
    else:
        raise ValueError('Prediction must be from [0-1]')
    return row


sub = pred.apply(filter_2cls, axis=1)
sub[['image_id', 'PredictionString']].to_csv('../submission.csv',index = False)




