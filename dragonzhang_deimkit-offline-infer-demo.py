


!cp -r /kaggle/input/deimkit-offline-package/DEIM /kaggle/working











import sys

sys.path.append('/kaggle/working/DEIM')


!cp /kaggle/input/deimkit-0-2-1-whl/deimkit-0.2.1-py3-none-any.whl /kaggle/working/DEIM/





!pip install --no-index --find-links='/kaggle/working/DEIM' deimkit 





from deimkit import list_models

list_models()


'''
from deimkit import Trainer, Config, configure_dataset

conf = Config.from_model_name("deim_hgnetv2_l")

conf = configure_dataset(
    config=conf,
    image_size=[640, 640],
    train_ann_file="/kaggle/input/byu-coco-format-dataset/dataset_json/train/annotations_train.json",
    train_img_folder="/kaggle/input/byu-coco-format-dataset/dataset_json/train",
    val_ann_file="/kaggle/input/byu-coco-format-dataset/dataset_json/val/annotations_valid.json",
    val_img_folder="/kaggle/input/byu-coco-format-dataset/dataset_json/val",
    train_batch_size=8,
    val_batch_size=8,
    num_classes=2,
    output_dir="./outputs",
)
'''


#!cp '/kaggle/input/deimkit-img800-wts/best (8).pth' /kaggle/working/best.pth


#!cp /kaggle/input/deimkit-img640-wts/best.pth  /kaggle/working/best.pth


#!cp '/kaggle/input/original-image-coco-deimkit-wts/best (11).pth'  /kaggle/working/best.pth





#!cp '/kaggle/input/deimkit-640-p94r98-e35-wts/checkpoint0035.pth'  /kaggle/working/best.pth


!cp '/kaggle/input/deimkit-640-p94r98-e35-wts/best (17).pth'  /kaggle/working/best.pth


import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



setup(0, 1)


 #image_size=(640 , 640),


from deimkit import load_model

coco_classes = ["motor"]
model = load_model(
    "deim_hgnetv2_s", 
    checkpoint="/kaggle/working/best.pth",
    class_names=["motor"],
    image_size=(960 , 960),
   
)








#result = model.predict("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test/tomo_01a877/slice_0026.jpg", visualize=False)





#result.visualization


import glob
import cv2
import numpy as np
import pandas as pd
from timeit import default_timer as timer

import matplotlib
import matplotlib.pyplot as plt

#from ultralytics import YOLO, RTDETR

import os,sys
sys.path.append('/kaggle/working/')


#--- helper -------------------- 
class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)

	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)
	else:
		raise NotImplementedError
#--------------------------------



print('IMPORT OK!!!')





KAGGLE_DATA_DIR='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'

MODE='submit'

#MODE = 'local'

if MODE == 'local':
    valid_dir = f'{KAGGLE_DATA_DIR}/train'
    valid_id = ['tomo_00e047', 'tomo_0c3a99', 'tomo_0fe63f', 'tomo_13484c', 'tomo_0363f2', 'tomo_1446aa', 'tomo_19a313', 'tomo_1cc887', 'tomo_221a47', 'tomo_2483bb', 'tomo_2a6ca2', 'tomo_2c9da1', 'tomo_30b580', 'tomo_331130', 'tomo_378f43', 'tomo_3e7783', 'tomo_412d88', 'tomo_455dcd', 'tomo_4b124b', 'tomo_4ee35e', 'tomo_517f70', 'tomo_53e048', 'tomo_57c814', 'tomo_5dd63d', 'tomo_60ddbd', 'tomo_640a74', 'tomo_672101', 'tomo_68e123', 'tomo_6cb0f0', 'tomo_6f2c1f', 'tomo_73173f', 'tomo_79756f', 'tomo_7f0184', 'tomo_82d780', 'tomo_881d84', 'tomo_8e4f7d', 'tomo_8f4d60', 'tomo_93c0b4', 'tomo_98686a', 'tomo_99a3ce', 'tomo_9f1828', 'tomo_a1a9a3', 'tomo_a537dd', 'tomo_a910fe', 'tomo_b03f81', 'tomo_b54396', 'tomo_ba9b3d', 'tomo_be4a3a', 'tomo_c36baf', 'tomo_c7b008', 'tomo_cc65a9', 'tomo_d0aa3b', 'tomo_d26fcb', 'tomo_d6c63f', 'tomo_d9a2af', 'tomo_decb81', 'tomo_e34af8', 'tomo_e63ab4', 'tomo_ec1314', 'tomo_f2fa4a', 'tomo_f871ad', 'tomo_fc5ae4', 'tomo_003acc', 'tomo_04d42b', 'tomo_087d64', 'tomo_0c2749', 'tomo_10a3bd', 'tomo_17143f', 'tomo_1c75ac', 'tomo_221c8e', 'tomo_24a095', 'tomo_288d4f', 'tomo_2c9f35', 'tomo_307f33', 'tomo_37c426', 'tomo_3a3519', 'tomo_3e6ead', 'tomo_466489', 'tomo_4baff0', 'tomo_4e3e37', 'tomo_512f98', 'tomo_569981', 'tomo_5d01e8', 'tomo_646049', 'tomo_6607ec', 'tomo_6a6a3b', 'tomo_6e196d', 'tomo_746d88', 'tomo_7dc063', 'tomo_80bf0f', 'tomo_85708b', 'tomo_8acc4b', 'tomo_8ee8fd', 'tomo_957567', 'tomo_97876d', 'tomo_9dbc12', 'tomo_a4f419', 'tomo_ab78d0', 'tomo_aeaf51', 'tomo_b24f1a', 'tomo_b7becf', 'tomo_b98cf6', 'tomo_bb5ac1', 'tomo_be9b98', 'tomo_c3619a', 'tomo_c596be', 'tomo_c925ee', 'tomo_cae587', 'tomo_d31c96', 'tomo_d8c917', 'tomo_dbc66d', 'tomo_dfdc32', 'tomo_e32b81', 'tomo_e72e60', 'tomo_e8db69', 'tomo_ecbc12', 'tomo_f672c0', 'tomo_f8b835', 'tomo_fc1665', 'tomo_ff7c20']
    valid_id = valid_id[:20]
if MODE == 'submit':
    valid_dir = f'{KAGGLE_DATA_DIR}/test'
    valid_id = glob.glob(f'{valid_dir}/*')
    valid_id = [f.split('/')[-1] for f in valid_id]

print('valid_id:', len(valid_id))
print(valid_id[:5], '...')

cfg = dotdict(
    slice_step=1,
    #batch_size=16,
    #batch_size=1,
    batch_size=32,
    
    #box_min_conf=0.40,
    box_min_conf=0.3,
    box_size=24,
    #iou_threshold=0.5,

    iou_threshold=0.5,

    device='cuda',

    #try cpu, gpu out of quota
    #device='cpu',
    checkpoint=\
    #'/kaggle/input/rtdetr-60e-wts/runs/detect/train/weights/best.pt'
    #'/kaggle/input/rtdetr-100epoch-wts/last.pt'

    '/kaggle/working/best.pth'
    
)

print('MODE:', MODE)
print('SETTING OK!!!')





#double gpu 

models = []
for i in [0,1] :
    #m = YOLO(cfg.checkpoint)
    m = model
   
    models.append(m)

print('MODEL OK!!!')
 


BASE_IMAGE_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
TEST_IMAGE_DIR = os.path.join(BASE_IMAGE_DIR, "test")
test_tomo_dir_list = glob.glob(f'{TEST_IMAGE_DIR}/*')
test_tomo_id_list = [d.split('/')[-1] for d in test_tomo_dir_list]





#deimkit load image methods
IMAGE_SIZE = (640, 640)
from PIL import Image, ImageDraw

def load_images(tomo_id, train_or_test='test', resize_size=IMAGE_SIZE, loader='torchvision'):
    assert loader in ['pil', 'torchvision']
    image_dir = f'{BASE_IMAGE_DIR}/{train_or_test}/{tomo_id}'
    image_files = sorted(glob.glob(f'{image_dir}/*.*'))
    df_image_files = pd.DataFrame({'filepath': image_files})
    df_image_files['no'] = df_image_files['filepath'].map(lambda x: int(x.split('_')[-1].split('.')[0]))
    df_image_files = df_image_files.sort_values(by='no', ascending=True)
  
    num=len(image_files)
    #print(image_files[:5])
    # print(df_image_files['filepath'][:10])
 
    # None : pil/torchvision resize results in slightly different values.
    if loader == 'pil':
        images = [Image.open(f).convert('L') for f in df_image_files['filepath']]
        org_image_size = images[0].size  # (w, h)
        if resize_size is not None:
            images = [image.resize(resize_size) for image in images]
        images = np.stack([np.asarray(image) for image in images])  # (n_frames, h, w)
    elif loader == 'torchvision':
        trainsforms = T.Resize(resize_size) if resize_size is not None else T.Compose([])
        images = [torchvision.io.read_image(f) for f in df_image_files['filepath']]
        org_image_size = (images[0].shape[2], images[0].shape[1])  # (w, h)
        images = [trainsforms(image) for image in images]
        images = torch.concatenate(images, dim=0)  # (n_frames, h, w)
        images = images.numpy()
    return images, df_image_files, org_image_size





# for evaluation
def make_truth_df(valid_id):
    label_df = pd.read_csv(f'{KAGGLE_DATA_DIR}/train_labels.csv',
       dtype={
           'Number of motors': int,
           'Array shape (axis 0)': int,
           'Array shape (axis 1)': int,
           'Array shape (axis 2)': int,
           'Motor axis 0': int,
           'Motor axis 1': int,
           'Motor axis 2': int,
           'Voxel spacing': float,
       })

    truth_df = []
    for tomo_id in valid_id:
        df = label_df[label_df['tomo_id'] == tomo_id]
        zyx = df[['Motor axis 0','Motor axis 1','Motor axis 2']].values[0].tolist()
        spacing = df['Voxel spacing'].values[0]
        num_motor = df['Number of motors'].values[0]
        truth_df.append({
            'tomo_id': tomo_id,
            'Voxel spacing': spacing,
            'Motor axis 0': zyx[0],
            'Motor axis 1': zyx[1],
            'Motor axis 2': zyx[2],
            'Has motor': int(num_motor>0)
        })

    truth_df = pd.DataFrame(truth_df)
    return truth_df

# inference helper

def distance_3d(d1, d2):
    return np.sqrt((d1['z'] - d2['z']) ** 2 +
                   (d1['y'] - d2['y']) ** 2 +
                   (d1['x'] - d2['x']) ** 2)

def do_mns_3d(detection):
    if not detection:
        return []

    distance_threshold = cfg.box_size * cfg.iou_threshold

    # Sort by confidence (highest first)
    detection = sorted(detection, key=lambda x: x['confidence'], reverse=True)

    #detection = sorted(detection, key=lambda x: x['scores'], reverse=True)

    nms = []
    while detection:
        # take the detection with highest confidence
        best_detection = detection.pop(0)
        nms.append(best_detection)
        detection = [d for d in detection if distance_3d(d, best_detection) > distance_threshold]

    return nms



def predict_one(model, tomo_id,imgsize=640):

    #yolo load image
  
    image_file = glob.glob(f'{valid_dir}/{tomo_id}/*.jpg')
    image_file = sorted(image_file)
    m = cv2.imread(image_file[0], cv2.IMREAD_GRAYSCALE)
    D = len(image_file)
    H,W = m.shape
    

    # deimkit load image
    # 1. Load images for target tomo_id
    '''
    image, image_file, org_image_size = load_images(tomo_id=tomo_id, loader='pil', resize_size=IMAGE_SIZE)
    image = image.transpose(1, 2, 0)  # (n_frames, h, w) => (h, w, n_frames)
    z_max = image.shape[-1] - 1
    w_org, h_org = org_image_size
    D = len(image_file)
    '''    

    slice_no = np.arange(D)[::cfg.slice_step]
    image_file = image_file[::cfg.slice_step]
    num_file = len(image_file)
    #print('num_file:',num_file)

    detection=[]
    for i in range(0, num_file, cfg.batch_size):
        batch_z = slice_no[i:i+cfg.batch_size]
        #print(batch_z)

        batch_file = image_file[i:i+cfg.batch_size]
        #print(batch_file)
        #result = model(batch_file, verbose=False)

        #result = model.predict(batch_file, visualize=False)

        result = model.predict_batch(batch_file,batch_size=cfg.batch_size, visualize=False)


        #result=    boxes,labels,scores,class_names,visualization
        

        for j, r in enumerate(result):
            if len(r.boxes) > 0:
                boxes = r.boxes
                #for k, confidence in enumerate(boxes.conf):
                #    if confidence >= cfg.box_min_conf:
                for k, confidence in enumerate(r.scores):
                    if confidence >= cfg.box_min_conf:
                        #x1, y1, x2, y2 = boxes.xyxy[k].cpu().numpy()
                        #yolo detection format
                        #x1, y1, x2, y2 = boxes[k]
                        #x = (x1 + x2) / 2
                        #y = (y1 + y2) / 2
                        #z = batch_z[j]


                        #deimkit format detection prediction, with normalized images 640  xywh??
                        #box format is [y1, x1, y2, x2] (COCO format)??
                        
                        x1, y1, x2, y2 = boxes[k]
                        x = (x1 + x2) / 2
                        y = (y1 + y2) / 2
                        #rescale back
                        #x =W*x/imgsize 
                        #y =H*y/imgsize
                        z = batch_z[j]
                        # Store detection with 3D coordinates
                        detection.append({
                            'z': round(z),
                            'y': round(y),
                            'x': round(x),
                            'confidence': float(confidence)
                        })
    # 3D Non-Maximum Suppression to merge nearby detections across slices
    mns = do_mns_3d(detection)
    mns.sort(key=lambda x: x['confidence'], reverse=True)


    #try no mns_3d
    #mns = detection
    #mns.sort(key=lambda x: x['confidence'], reverse=True)    


    # If there are no detections, return NA values
    if not mns:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1,
            'confidence': cfg.box_min_conf,
        }
    else:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': mns[0]['z'],
            'Motor axis 1': mns[0]['y'],
            'Motor axis 2': mns[0]['x'],
            'confidence': mns[0]['confidence'],
        }




def do_predict(model, valid_id, rank):

    result = []
    total_time_taken = 0
    for i,tomo_id in enumerate(valid_id):
        start_timer = timer()

        r = predict_one(model, tomo_id)
        print(r)
        result.append(r)
        time_taken = timer() - start_timer
        total_time_taken += time_taken
        print('\r',f'rank{rank}', i, r,  time_to_str(total_time_taken, 'min'), end='')
        if 0: #MODE=='local': #show some example
            pass

            if i ==0:
                z = int(r['Motor axis 0'])
                y = int(r['Motor axis 1'])
                x = int(r['Motor axis 2'])
                if z==-1:
                    continue

                image_file = f'{valid_dir}/{tomo_id}/slice_{z:04d}.jpg'
                m = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                
                overlay = np.stack([m, m, m], -1)
                cv2.circle(overlay, (x,y), 6, (0,255,255), 2)

                #overlay = 255 - (255 - overlay) * (1 - p)
                overlay = overlay.astype(np.uint8)
                plt.imshow(overlay)
                plt.show()
    print('')
    return result


#########################################################################################################
if 1:
    from concurrent.futures import ThreadPoolExecutor
    start_timer = timer()
    N = len(valid_id)
  
    with ThreadPoolExecutor(max_workers=2) as executor:
       
        future0 = executor.submit(do_predict, model, valid_id[0::2], 0)
        future1 = executor.submit(do_predict, model, valid_id[1::2], 1)
        result0 = future0.result()
        result1 = future1.result()
    result = result0+result1
    
    '''
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(do_predict, model,valid_id[0::2], 0)

        result = future.result()
   '''

    total_time_taken = timer() - start_timer
    print('** total_time_taken:',time_to_str(total_time_taken, 'min'))
    print('time est for 900 tomograph:', time_to_str(total_time_taken/len(valid_id)*900, 'min'))
    print('')

    result_df = pd.DataFrame(result)
    result_df.to_csv('result.csv',index=False)

    submit_df = result_df[['tomo_id','Motor axis 0','Motor axis 1','Motor axis 2']]
    submit_df.to_csv('submission.csv',index=False)
    print(submit_df)
    print('SUBMIT OK !!!!!!!!!!!!')







