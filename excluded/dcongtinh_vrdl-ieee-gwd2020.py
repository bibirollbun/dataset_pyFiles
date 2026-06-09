# !cp -r /kaggle/input/wheat-src/* .
!cp -r /kaggle/input/gwd2020-src/* .
!ls


!pip install /kaggle/input/wheat-packages/torch-1.4.0-cp37-cp37m-linux_x86_64.whl -f ./ --no-index
!pip install /kaggle/input/wheat-packages/torchvision-0.5.0-cp37-cp37m-linux_x86_64.whl -f ./ --no-index
!pip install /kaggle/input/wheat-packages/ensemble_boxes-1.0.4-py3-none-any.whl -f ./ --no-index
!pip install /kaggle/input/wheat-packages/omegaconf-2.0.0-py3-none-any.whl -f ./ --no-index
!pip install /kaggle/input/wheat-packages/timm-0.1.28-py3-none-any.whl -f ./ --no-index


import os
import numpy as np
import pandas as pd
import gc
import cv2
import random
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
from ensemble_boxes import weighted_boxes_fusion
from utils import get_resolution, save_dict, load_dict, format_prediction_string, make_pseudo_dataframe, refine_checkpoint_in, refine_checkpoint_out


test_df = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')
TEST_DIR = '/kaggle/input/global-wheat-detection/test'
TRAIN_DIR = '/kaggle/input/global-wheat-detection/train'
CHECKPOINT_DIR = '/kaggle/input/wheat2020-checkpoints'

if len(test_df) > 10:
    USE_AMP = True
    VISUALIZE = False
    PSEUDO = True
else:
    USE_AMP = False
    VISUALIZE = True
    PSEUDO = False


if USE_AMP:
    !cp -r /kaggle/input/nvidiaapex/ .
    !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" nvidiaapex/. --user
    !rm -rf nvidiaapex
    from apex import amp


def visualize_result(test_df, TEST_DIR, output_dict):
    for image_id in list(np.unique(test_df.image_id.values)):
        img_path = '{}/{}.jpg'.format(TEST_DIR, image_id)
        image = Image.open(img_path)
        image = image.convert('RGB')
        image = np.array(image)

        boxes, scores = output_dict[image_id]
        if len(boxes) > 0:
            boxes = boxes.astype(np.int32)

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)

        ax.set_axis_off()
        ax.imshow(image)


def run_wbf(test_df, box_pred, score_pred, label_pred, resolution_dict, NMS_THRESH, BOX_THRESH, PP_THRESH):
    output_dict = {}
    for image_id in np.unique(test_df.image_id.values):
        boxes = box_pred[image_id]
        scores = score_pred[image_id]
        labels = label_pred[image_id]

        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=NMS_THRESH, skip_box_thr=BOX_THRESH)
        boxes = np.array(boxes)
        scores = np.array(scores)

        idxs = np.where(scores > PP_THRESH)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]

        if len(boxes) > 0:
            height, width = resolution_dict[image_id]
            boxes[:, [0,2]] = (boxes[:, [0,2]]*width).clip(min=0, max=width-1)
            boxes[:, [1,3]] = (boxes[:, [1,3]]*height).clip(min=0, max=height-1)
        output_dict[image_id] = (boxes, scores)
    return output_dict


def run_wbf_4preds(test_df, 
                   box_pred1, score_pred1, label_pred1, 
                   box_pred2, score_pred2, label_pred2, 
                   box_pred3, score_pred3, label_pred3, 
                   box_pred4, score_pred4, label_pred4,
                   resolution_dict, NMS_THRESH, BOX_THRESH, PP_THRESH):
    output_dict = {}
    for image_id in np.unique(test_df.image_id.values):
        boxes = box_pred1[image_id] + box_pred2[image_id] + box_pred3[image_id] + box_pred4[image_id]
        scores = score_pred1[image_id] + score_pred2[image_id] + score_pred3[image_id] + score_pred4[image_id]
        labels = label_pred1[image_id] + label_pred2[image_id] + label_pred3[image_id] + label_pred4[image_id]

        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=NMS_THRESH, skip_box_thr=BOX_THRESH)
        boxes = np.array(boxes)
        scores = np.array(scores)

        idxs = np.where(scores > PP_THRESH)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]

        if len(boxes) > 0:
            height, width = resolution_dict[image_id]
            boxes[:, [0,2]] = (boxes[:, [0,2]]*width).clip(min=0, max=width-1)
            boxes[:, [1,3]] = (boxes[:, [1,3]]*height).clip(min=0, max=height-1)
        output_dict[image_id] = (boxes, scores)
    return output_dict


# get resolution for each image in test set
with Pool(processes=2) as pool:
    results = [pool.apply_async(get_resolution, args=(image_id, TEST_DIR)) for image_id in list(np.unique(test_df.image_id.values))]
    results = [ret.get() for ret in results]
pool.close()
resolution_dict = {}
for ret in results:
    resolution_dict[ret[0]] = (ret[1], ret[2])


!python predict.py  --network effdet \
                    --backbone ed7 \
                    --img-size 768 \
                    --batch-size 16 \
                    --workers 2 \
                    --test-csv /kaggle/input/global-wheat-detection/sample_submission.csv \
                    --test-dir /kaggle/input/global-wheat-detection/test \
                    --checkpoint-dir /kaggle/input/gwd2020-checkpoints/pytorch/default/4 \
                    --folds 0 1 2 3 4 \
                    --use-amp True

!python predict.py  --network effdet \
                    --backbone ed7 \
                    --img-size 1024 \
                    --batch-size 16 \
                    --workers 2 \
                    --test-csv /kaggle/input/global-wheat-detection/sample_submission.csv \
                    --test-dir /kaggle/input/global-wheat-detection/test \
                    --checkpoint-dir /kaggle/input/gwd2020-checkpoints/pytorch/default/4 \
                    --folds 1 3 \
                    --use-amp True

!python predict.py  --network effdet \
                    --backbone ed5 \
                    --img-size 512 \
                    --batch-size 16 \
                    --workers 2 \
                    --test-csv /kaggle/input/global-wheat-detection/sample_submission.csv \
                    --test-dir /kaggle/input/global-wheat-detection/test \
                    --checkpoint-dir /kaggle/input/gwd2020-checkpoints/pytorch/default/4 \
                    --folds 4 \
                    --use-amp True

!python predict.py  --network fasterrcnn \
                    --backbone resnet152 \
                    --img-size 1024 \
                    --batch-size 16 \
                    --workers 2 \
                    --test-csv /kaggle/input/global-wheat-detection/sample_submission.csv \
                    --test-dir /kaggle/input/global-wheat-detection/test \
                    --checkpoint-dir /kaggle/input/gwd2020-checkpoints/pytorch/default/4 \
                    --folds 1 \
                    --use-amp False

effdet_ed7_768_box_pred = load_dict('effdet_ed7_768_box_pred.pkl')
effdet_ed7_768_score_pred = load_dict('effdet_ed7_768_score_pred.pkl')
effdet_ed7_768_label_pred = load_dict('effdet_ed7_768_label_pred.pkl')

effdet_ed7_1024_box_pred = load_dict('effdet_ed7_1024_box_pred.pkl')
effdet_ed7_1024_score_pred = load_dict('effdet_ed7_1024_score_pred.pkl')
effdet_ed7_1024_label_pred = load_dict('effdet_ed7_1024_label_pred.pkl')

effdet_ed5_box_pred = load_dict('effdet_ed5_512_box_pred.pkl')
effdet_ed5_score_pred = load_dict('effdet_ed5_512_score_pred.pkl')
effdet_ed5_label_pred = load_dict('effdet_ed5_512_label_pred.pkl')

fasterrcnn_box_pred = load_dict('fasterrcnn_resnet152_1024_box_pred.pkl')
fasterrcnn_score_pred = load_dict('fasterrcnn_resnet152_1024_score_pred.pkl')
fasterrcnn_label_pred = load_dict('fasterrcnn_resnet152_1024_label_pred.pkl')

output_dict = run_wbf_4preds(test_df, 
                             effdet_ed7_768_box_pred, effdet_ed7_768_score_pred, effdet_ed7_768_label_pred,
                             effdet_ed7_1024_box_pred, effdet_ed7_1024_score_pred, effdet_ed7_1024_label_pred,
                             effdet_ed5_box_pred, effdet_ed5_score_pred, effdet_ed5_label_pred,
                             fasterrcnn_box_pred, fasterrcnn_score_pred, fasterrcnn_label_pred, 
                             resolution_dict, NMS_THRESH=0.50, BOX_THRESH=0.32, PP_THRESH=0.28)

del effdet_ed7_768_box_pred
del effdet_ed7_768_score_pred
del effdet_ed7_768_label_pred

del effdet_ed7_1024_box_pred
del effdet_ed7_1024_score_pred
del effdet_ed7_1024_label_pred

del effdet_ed5_box_pred
del effdet_ed5_score_pred
del effdet_ed5_label_pred

del fasterrcnn_box_pred
del fasterrcnn_score_pred
del fasterrcnn_label_pred

gc.collect()

## visualize
if VISUALIZE:
    visualize_result(test_df, TEST_DIR, output_dict)


if PSEUDO:
    df = pd.read_csv('csv/gwd2020.csv')   ### convert train.csv to image_id,fold,xmin,ymin,xmax,ymax,isbox,source
    PSEUDO_FOLD = 1
    make_pseudo_dataframe(test_df, output_dict, TEST_DIR, df, TRAIN_DIR, PSEUDO_FOLD)
    del output_dict
    gc.collect()

    !python effdet_train.py --backbone ed6 \
                            --img-size 640 \
                            --batch-size 5 \
                            --pretrain-path /kaggle/input/wheat2020-checkpoints/effdet_ed6_640_fold1.pth \
                            --checkpoint-path ./effdet_ed6_640_fold1_with_optimizer.pth \
                            --epochs 10 \
                            --init-lr 8e-5 \
                            --mixup True \
                            --use-amp True \
                            --load-optimizer False \
                            --save-optimizer True
    
    #remove optimizer in checkpoint
    refine_checkpoint_out('./effdet_ed6_640_fold1_with_optimizer.pth', './effdet_ed6_640_fold1.pth')
    
    !python predict.py  --network effdet \
                        --backbone ed6 \
                        --img-size 640 \
                        --batch-size 16 \
                        --workers 2 \
                        --test-csv /kaggle/input/global-wheat-detection/sample_submission.csv \
                        --test-dir /kaggle/input/global-wheat-detection/test \
                        --checkpoint-dir ./. \
                        --folds 1 \
                        --use-amp True

    box_pred = load_dict('effdet_ed6_640_box_pred.pkl')
    score_pred = load_dict('effdet_ed6_640_score_pred.pkl')
    label_pred = load_dict('effdet_ed6_640_label_pred.pkl')
    output_dict = run_wbf(test_df, box_pred, score_pred, label_pred, 
                          resolution_dict, NMS_THRESH=0.50, BOX_THRESH=0.42, PP_THRESH=0.32)
    del box_pred
    del score_pred
    del label_pred
    gc.collect()

    !rm -rf ./train.csv
    !rm -rf ./valid.csv


if PSEUDO:
    df = pd.read_csv('csv/gwd2020.csv')   ### convert train.csv to image_id,fold,xmin,ymin,xmax,ymax,isbox,source
    PSEUDO_FOLD = 1
    make_pseudo_dataframe(test_df, output_dict, TEST_DIR, df, TRAIN_DIR, PSEUDO_FOLD)
    del output_dict
    gc.collect()

    !python effdet_train.py --backbone ed6 \
                            --img-size 640 \
                            --batch-size 5 \
                            --pretrain-path ./effdet_ed6_640_fold1_with_optimizer.pth \
                            --checkpoint-path ./effdet_ed6_640_fold1.pth \
                            --epochs 6 \
                            --init-lr 1e-5 \
                            --mixup True \
                            --use-amp True \
                            --load-optimizer True \
                            --save-optimizer False
    
    !python predict.py  --network effdet \
                        --backbone ed6 \
                        --img-size 640 \
                        --batch-size 16 \
                        --workers 2 \
                        --test-csv /kaggle/input/global-wheat-detection/sample_submission.csv \
                        --test-dir /kaggle/input/global-wheat-detection/test \
                        --checkpoint-dir ./. \
                        --folds 1 \
                        --use-amp True

    box_pred = load_dict('effdet_ed6_640_box_pred.pkl')
    score_pred = load_dict('effdet_ed6_640_score_pred.pkl')
    label_pred = load_dict('effdet_ed6_640_label_pred.pkl')
    output_dict = run_wbf(test_df, box_pred, score_pred, label_pred, 
                          resolution_dict, NMS_THRESH=0.50, BOX_THRESH=0.44, PP_THRESH=0.34)
    del box_pred
    del score_pred
    del label_pred
    gc.collect()

    !rm -rf ./train.csv
    !rm -rf ./valid.csv
    !rm -rf effdet_ed6_640_fold1_with_optimizer.pth


!rm -rf ./effdet
!rm -rf ./csv
!rm -rf ./*.py
!rm -rf ./*.pkl
!rm -rf ./LICENSE
!rm -rf ./README.md
!rm -rf ./__pycache__
!ls


results = []
for image_id in list(np.unique(test_df.image_id.values)):
    boxes, scores = output_dict[image_id]
    if len(boxes) > 0:
        boxes = boxes.astype(np.int32)
        #xyxy to xywh
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
    result = {
        'image_id': image_id,
        'PredictionString': format_prediction_string(boxes, scores)
    }
    results.append(result)
sub_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
sub_df.to_csv('submission.csv', index=False)
print(sub_df.head(20))

