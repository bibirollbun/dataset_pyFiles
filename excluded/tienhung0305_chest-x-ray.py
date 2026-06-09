import os
import gc
import sys
import time
import json
import glob
import random
from pathlib import Path
import pandas as pd

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

import itertools
from tqdm import tqdm
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import warnings 
warnings.filterwarnings("ignore")


training_folder = "../input/vinbigdata-chest-xray-abnormalities-detection/train/"
df = pd.read_csv("../input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
df = df.query("class_id<14")
df = df.query("rad_id=='R9'")


df["several_issues"] = df.duplicated(subset=['image_id'])
df["box_size"] = [(row.y_max-row.y_min)*(row.x_max-row.x_min) for idx, row in df.iterrows()]


df.class_name.unique()


df.groupby("class_id")["box_size"].mean()


df.groupby("class_id")["box_size"].std()


df.groupby("class_id").image_id.count()


selected_classes = [0,3,5,7,10]
category_list = ["Aortic enlargement", "Cardiomegaly", "ILD", "Lung Opacity", "Pleural effusion"]
filtered_df = df.query("class_id in @selected_classes")


selected_classes_dict = {"0":0,"3":1,"5":2,"7":3,"10":4}
filtered_df["reclass_id"] = [selected_classes_dict[str(row.class_id)] for idx, row in filtered_df.iterrows()]


filtered_df


def get_mask(img_dimensions, x_min, y_min, x_max, y_max):
    img_height, img_width = img_dimensions
    img_mask = np.full((img_height,img_width),0)
    img_mask[y_min:y_max,x_min:x_max] = 255
    
    return img_mask.astype(np.float32)


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 255)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join([str(x) for x in run_lengths])


def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data


resized_folder = "../working/resized_train/"
os.mkdir(resized_folder)


filtered_df.groupby("class_id").image_id.count()


balanced_filtered_df = pd.DataFrame()
samples_per_class = 500
for class_name in filtered_df.class_name.unique():
    balanced_filtered_df = balanced_filtered_df.append(filtered_df.query("class_name==@class_name")[:samples_per_class], 
                                                       ignore_index=True)


balanced_filtered_df


diagnostic_per_image = []

image_size=512
with tqdm(total=len(balanced_filtered_df)) as pbar:
    for idx,row in balanced_filtered_df.iterrows():
        image_id = row.image_id
        image_df = balanced_filtered_df.query("image_id==@image_id")
        class_list = []
        RLE_list = []
        
        for diagnostic_id, diagnostic in image_df.iterrows():
            class_list.append(diagnostic.reclass_id)

            dicom_image = read_xray(training_folder+image_id+".dicom")
            image_dimensions = dicom_image.shape
            
            resized_img = cv2.resize(dicom_image, (image_size,image_size), interpolation = cv2.INTER_AREA)
            cv2.imwrite(resized_folder+image_id+".jpg", resized_img) 
            
            mask = get_mask(image_dimensions, int(diagnostic.x_min), int(diagnostic.y_min), int(diagnostic.x_max), int(diagnostic.y_max))
            resized_mask = cv2.resize(mask, (image_size,image_size))
            RLE_list.append(rle_encoding(resized_mask))
        diagnostic_per_image.append({"image_id":image_id,
                                     "CategoryId":class_list,
                                     "EncodedPixels":RLE_list})
        pbar.update(1)


samples_df = pd.DataFrame(diagnostic_per_image)
samples_df["Height"] = image_size
samples_df["Width"] = image_size


samples_df


!cp -r ../input/maskrcnn-tf2-keras ../working/maskrcnn-tf2-keras


DATA_DIR = Path('../working/')
ROOT_DIR = "../../working"

NUM_CATS = len(selected_classes)
IMAGE_SIZE = 512
os.chdir('../working/maskrcnn-tf2-keras')
sys.path.append(ROOT_DIR+'/maskrcnn-tf2-keras')
from mrcnn.config import Config

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


COCO_WEIGHTS_PATH = '../../input/coco-weights/mask_rcnn_coco.h5'

class DiagnosticConfig(Config):
    NAME = "Diagnostic"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 #That is the maximum with the memory available on kernels
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'

    POST_NMS_ROIS_TRAINING = 250
    POST_NMS_ROIS_INFERENCE = 150
    MAX_GROUNDTRUTH_INSTANCES = 5
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    BACKBONESHAPE = (8, 16, 24, 32, 48)
    RPN_ANCHOR_SCALES = (8,16,24,32,48)
    ROI_POSITIVE_RATIO = 0.33
    DETECTION_MAX_INSTANCES = 300
    DETECTION_MIN_CONFIDENCE = 0.7    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = int(len(samples_df)*0.9/IMAGES_PER_GPU)
    VALIDATION_STEPS = len(samples_df)-int(len(samples_df)*0.9/IMAGES_PER_GPU)
    
config = DiagnosticConfig()
config.display()


class DiagnosticDataset(utils.Dataset):
    def __init__(self, df):
        super().__init__(self)
        
        # Add classes
        for i, name in enumerate(category_list):
            self.add_class("diagnostic", i+1, name)
        
        # Add images 
        for i, row in df.iterrows():
            self.add_image("diagnostic", 
                           image_id=row.name,
                           path="../"+resized_folder+str(row.image_id)+".jpg", 
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [category_list[int(x)] for x in info['labels']]
    
    def load_image(self, image_id):
        return cv2.imread(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
        return mask, np.array(labels)


training_percentage = 0.9

training_set_size = int(training_percentage*len(samples_df))
validation_set_size = int((1-training_percentage)*len(samples_df))

train_dataset = DiagnosticDataset(samples_df[:training_set_size])
train_dataset.prepare()

valid_dataset = DiagnosticDataset(samples_df[training_set_size:training_set_size+validation_set_size])
valid_dataset.prepare()

for i in range(10):
    image_id = random.choice(train_dataset.image_ids)
    image = train_dataset.load_image(image_id)
    mask, class_ids = train_dataset.load_mask(image_id)
    
    visualize.display_top_masks(image, mask, class_ids, train_dataset.class_names, limit=5)


LR = 1e-4
EPOCHS = [1,18]

model = modellib.MaskRCNN(mode='training', config=config, model_dir="")
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])


%%time
model.train(train_dataset, valid_dataset,
            learning_rate=LR,
            epochs=EPOCHS[0],
            layers='heads')

history = model.keras_model.history.history


%%time
model.train(train_dataset, valid_dataset,
            learning_rate=LR/10,
            epochs=EPOCHS[1],
            layers='all')

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]


epochs = range(EPOCHS[-1])

plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()

plt.show()


best_epoch = np.argmin(history["val_loss"][1:]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][1:][best_epoch-1])


resized_test_folder = "../../working/resized_test/"
os.mkdir(resized_test_folder)


class InferenceConfig(DiagnosticConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    DETECTION_MIN_CONFIDENCE = 0.8
    DETECTION_NMS_THRESHOLD = 0.5

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir="")


glob_list = glob.glob(f'diagnostic*/mask_rcnn_diagnostic_{best_epoch:04d}.h5')
model_path = glob_list[0] if glob_list else ''
model.load_weights(model_path, by_name=True)


from skimage.measure import find_contours
from matplotlib.patches import Polygon


# Fix overlapping masks
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

def decode_rle(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height*width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((height, width)).T

def annotations_to_mask(annotations, height, width):
    if isinstance(annotations, list):
        # The annotation consists in a list of RLE codes
        mask = np.zeros((height, width, len(annotations)))
        for i, rle_code in enumerate(annotations):
            mask[:, :, i] = decode_rle(rle_code, height, width)
    else:
        error_message = "{} is expected to be a list or str but received {}".format(annotation, type(annotation))
        raise TypeError(error_message)
    return mask

def find_anomalies(dicom_image, display=False):

    image_dimensions = dicom_image.shape

    resized_img = cv2.resize(dicom_image, (image_size,image_size), interpolation = cv2.INTER_AREA)
    saved_filename = resized_test_folder+"temp_image.jpg"
    cv2.imwrite(saved_filename, resized_img) 
    img = cv2.imread(saved_filename)

    result = model.detect([img])
    r = result[0]
    
    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y_scale = image_dimensions[0]/IMAGE_SIZE
        x_scale = image_dimensions[1]/IMAGE_SIZE
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        
        masks, rois = refine_masks(masks, rois)
    else:
        masks, rois = r['masks'], r['rois']
        
    if display:
        visualize.display_instances(img, rois, masks, r['class_ids'], 
                                    ['bg']+category_list, r['scores'],
                                    title="prediction", figsize=(12, 12))
    return rois, r['class_ids'], r['scores']


test_folder = "../../input/vinbigdata-chest-xray-abnormalities-detection/test/"
test_file_list = os.listdir(test_folder)[:5]

for test_file in test_file_list:
    dicom_image = read_xray(test_folder+test_file)
    find_anomalies(dicom_image, display=True)


def keep_best_cardiomegaly_box(bbox_list, class_list, confidence_list):
    '''
    go through the boxes and keep only one box for 
    cardiomegaly with the highest confidence score
    '''
    best_cardiomegaly_score = -1
    best_cardiomegaly_bbox = []
    clean_bbox_list, clean_class_list, clean_confidence_list = [],[],[]
    
    for bbox, class_id, confidence in zip(bbox_list, class_list, confidence_list):
        #While the class number if 3 in the dataset, it is 2 in the maskrcnn training process
        # as I have excluded some classes
        if class_id==2:
            if confidence>best_cardiomegaly_score:
                best_cardiomegaly_score = confidence
                best_cardiomegaly_bbox = bbox
        else:
            clean_bbox_list.append(bbox)
            clean_class_list.append(class_id)
            clean_confidence_list.append(confidence)
            
    if best_cardiomegaly_score>0:
        clean_bbox_list.append(best_cardiomegaly_bbox)
        clean_class_list.append(2)
        clean_confidence_list.append(best_cardiomegaly_score)
        
    return clean_bbox_list, clean_class_list, clean_confidence_list


results = []
test_file_list = os.listdir(test_folder)
with tqdm(total=len(test_file_list)) as pbar:
    for image_filename in test_file_list:
        dicom_image = read_xray(test_folder+image_filename)
        image_dimensions = dicom_image.shape
        bbox_list, class_list, confidence_list = find_anomalies(dicom_image, display=False)
        prediction_string = ""
        
        if len(bbox_list)>0:
                    
            bbox_list, class_list, confidence_list = keep_best_cardiomegaly_box(bbox_list, class_list, confidence_list)
            
            for bbox, class_id, confidence in zip(bbox_list, class_list, confidence_list):
                class_id = next(key for key, value in selected_classes_dict.items() if value == int(class_id)-1)
                confidence_score = str(round(confidence,3))

                #HACK: I had to rescale the bounding box here. For some reason,
                #It did not do it in the prediction function.
                y_scale = image_dimensions[0]/image_size
                x_scale = image_dimensions[1]/image_size
                rescaled_bbox = (bbox * [y_scale, x_scale, y_scale, x_scale]).astype(int)

                #organise the bbox into xmin, ymin, xmax, ymax
                ymin = image_dimensions[0]-rescaled_bbox[2]
                ymax = image_dimensions[0]-rescaled_bbox[0]
                xmin = rescaled_bbox[1]
                xmax = rescaled_bbox[3]

                prediction_string += "{} {} {} {} {} {} ".format(class_id, confidence_score, xmin, ymin, xmax, ymax)
            results.append({"image_id":image_filename.replace(".dicom",""), "PredictionString":prediction_string.strip()})
        else:
            results.append({"image_id":image_filename.replace(".dicom",""), "PredictionString":"14 1.0 0 0 1 1"})
        pbar.update(1)


submission_df = pd.DataFrame(results)


submission_df


submission_df.to_csv('../submission.csv', index=False)


#clear all the images from the working directory
!rm -rf ../../working/resized_train/
!rm -rf ../../working/resized_test/

