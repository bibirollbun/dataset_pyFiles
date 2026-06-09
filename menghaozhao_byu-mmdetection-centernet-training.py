from IPython.display import clear_output

!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/pycocotools-2.0.6-cp310-cp310-linux_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/torch-1.12.1+cu116-cp310-cp310-linux_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/torchvision-0.13.1+cu116-cp310-cp310-linux_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/mmcv-2.0.1-cp310-cp310-manylinux1_x86_64.whl 
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/openmim-0.3.9-py2.py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/mmengine-0.7.4-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/addict-2.4.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/mmdet-3.1.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/terminaltables-3.1.10-py2.py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/yapf-0.40.1-py3-none-any.whl
# !pip install yapf==0.40.1
# !pip install terminaltables
# !pip install setuptools==69.5.1
# !pip install --no-index --no-deps /kaggle/input/mmpretrain/einops-0.6.1-py3-none-any.whl
# !pip install --no-index --no-deps /kaggle/input/mmpretrain/mat4py-0.5.0-py2.py3-none-any.whl
# !pip install --no-index --no-deps /kaggle/input/mmpretrain/ordered_set-4.1.0-py3-none-any.whl
# !pip install --no-index --no-deps /kaggle/input/mmpretrain/model_index-0.1.11-py3-none-any.whl
# !pip install --no-index --no-deps /kaggle/input/mmpretrain/modelindex-0.0.2-py3-none-any.whl
# !pip install --no-index --no-deps /kaggle/input/mmpretrain/mmpretrain-1.0.0rc8-py2.py3-none-any.whl
# clear_output()


# !pip install /kaggle/input/mmengine-0-10-7/mmengine-0.10.7-py3-none-any.whl


# !pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/yapf-0.40.1-py3-none-any.whl



from tqdm.notebook import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

from glob import glob
import matplotlib.pyplot as plt


# %rm -r mmdetection


# !git clone https://github.com/open-mmlab/mmdetection.git
%cp -r /kaggle/input/byu-mmdet/mmdetection/mmdetection /kaggle/working
%cd /kaggle/working/mmdetection
# !pip install -e .

# clear_output()


# Check Pytorch installation
import torch, torchvision
print("torch version:",torch.__version__, "cuda:",torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print("mmdetection:",mmdet.__version__)

# Check mmcv installation
import mmcv
print("mmcv:",mmcv.__version__)

# Check mmengine installation
import mmengine
print("mmengine:",mmengine.__version__)


!mkdir -p configs/BYU


%%writefile configs/BYU/custom_center_netconfig.py

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    # '../centernet/centernet_tta.py'
    # '/kaggle/input/byu-mmdet/mmdetection/mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
]

dataset_type = 'CocoDataset'
data_root = '/kaggle/input/byu-coco-dataset/dataset_json/'
# backend_args = None
classes = ('motor',)

# model settings
model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=None),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))


albu_train_transforms = [
    dict(
        type='Sharpen',
        alpha=(0.2, 0.7)
    )
]

img_scale = (768, 768)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        # The cropped images are padded into squares during training,
        # but may be less than crop_size.
        crop_size=img_scale,
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    # Make sure the output is always crop_size.
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args={{_base_.backend_args}},
        to_float32=True),
    # don't need Resize
    dict(
        type='RandomCenterCropPad',
        ratios=None,
        border=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_mode=True,
        test_pad_mode=['logical_or', 31],
        test_pad_add_pix=1),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'border','scale_factor'))
]

# Use RepeatDataset to speed up training
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations_train.json',
            metainfo=dict(classes=classes),
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args={{_base_.backend_args}},
        ))
)

# dataset=dict(
#     _delete_=True,
#     type='RepeatDataset',
#     times=3,
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations_train.json',
#         metainfo=dict(classes=classes),
#         data_prefix=dict(img='train2017/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args={{_base_.backend_args}},
#     ))

# dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations_train.json',
#         metainfo=dict(classes=classes),
#         data_prefix=dict(img='train2017/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args={{_base_.backend_args}},
#     )


val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations_valid.json',
        data_prefix=dict(img='valid2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args={{_base_.backend_args}}))
test_dataloader = val_dataloader
val_evaluator = dict(
ann_file=data_root + 'annotations_valid.json', metric=['bbox'])


# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=0.05),
        # accumulative_counts=4,
)

max_epochs = 4

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[18, 24],  # the real step is [18*5, 24*5]
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)  # the real epoch is 28*5=140
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
auto_scale_lr = dict(base_batch_size=128)

fp16=dict(loss_scale=512.)
work_dir = '../model_output'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
        save_best='coco/bbox_mAP_50'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
load_from = '/kaggle/input/byu-maskrcnn/mask_epoch_2.pth'
resume = False


!python tools/train.py configs/BYU/custom_center_netconfig.py


from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config
config_file = 'configs/BYU/custom_center_netconfig.py'
checkpoint_file = '/kaggle/input/byu-maskrcnn/mask_epoch_2.pth'

cfg = Config.fromfile(config_file )
model = init_detector(cfg, checkpoint_file, device='cuda')



!ls /kaggle/working/mmdetection/configs/_base_/datasets/


img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_00e047_z0166_y0546_x0603.jpg"
img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_423d52_z0150_y0444_x0456.jpg"
img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_ae347a_z0184_y0775_x0596.jpg"
img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_0363f2_z0046_y0574_x0392.jpg"
img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_139d9e_z0154_y0611_x0172.jpg"
# img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_161683_z0190_y0145_x0370.jpg"
# img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_1ab322_z0066_y0580_x0583.jpg"

# img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_1b82d1_z0212_y0697_x0680.jpg"
# img="/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_0de3ee_z0130_y0139_x0326.jpg"
from glob import glob
img = glob("/kaggle/input/byu-coco-dataset/dataset_json/train2017/*.jpg")[1984]

img_array = mmcv.imread(img,channel_order='rgb')
[h, w, c] = img_array.shape 
pred = inference_detector(model,img)
bbox = pred.pred_instances.bboxes[0].detach().cpu().numpy()
bbox,img,pred.pred_instances.scores[:3]


# pred.pred_instances
import json
with open("/kaggle/input/byu-coco-dataset/dataset_json/annotations_train.json", "r") as f:
    annotation = json.load(f)
img_filename = img.split("/")[-1]
img_id = [x for x in annotation["images"] if x['file_name'] == img_filename][0]["id"]
bbox_ann = [x for x in annotation["annotations"] if x["id"] == img_id][0]["bbox"]


import matplotlib.pyplot as plt
import matplotlib.patches as patches
rect = patches.Rectangle((bbox[0],bbox[1]), max(bbox[2]-bbox[0],15), max(bbox[3]-bbox[1],15), linewidth=1, edgecolor='r', facecolor='none')

rect_ann = patches.Rectangle((bbox_ann[0],bbox_ann[1]), bbox_ann[2], bbox_ann[3], linewidth=1, edgecolor='b', facecolor='none')
plt.imshow(img_array)
plt.gca().add_patch(rect)
plt.gca().add_patch(rect_ann)


import pandas as pd
train_labels = pd.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")
sample_submission = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/sample_submission.csv')
train_labels.merge(sample_submission, on='tomo_id')


tomo_folders = glob("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test/*")
# tomo_folders = ["/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_00e047_z0166_y0546_x0603.jpg"]
# tomo_folders = ["/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_423d52_z0150_y0444_x0456.jpg"]
# tomo_folders = ["/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_ae347a_z0184_y0775_x0596.jpg"]
# tomo_folders = ["/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_0363f2_z0046_y0574_x0392.jpg"]
# tomo_folders = ["/kaggle/input/byu-coco-dataset/dataset_json/train2017/tomo_139d9e_z0154_y0611_x0172.jpg"]

from tqdm import tqdm
results = []
for tomo_folder in tomo_folders:
    tomo_id = tomo_folder.split("/")[-1]
    img_files = sorted(glob(tomo_folder+"/*.jpg"))
    all_detections = []
    for img_file in tqdm(img_files):
        pred = inference_detector(model,img_file)
        z = int(img_file.split("_")[-1].split(".")[0])
        if pred.pred_instances.scores[0] >= 0.4:
            bbox = pred.pred_instances.bboxes[0].detach().cpu().numpy()
            x_center = (bbox[0]+bbox[2])/2.0
            y_center = (bbox[1]+bbox[3])/2.0
            all_detections.append({
                                    'z': round(z),
                                    'y': round(y_center),
                                    'x': round(x_center),
                                    'confidence': float(pred.pred_instances.scores[0])
                                })
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)

    if len(all_detections) > 0:
        results.append({
            'tomo_id': tomo_id,
            'Motor axis 0': all_detections[0]['z'],
            'Motor axis 1': all_detections[0]['y'],
            'Motor axis 2': all_detections[0]['x']
        })
    else:
        results.append({
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1
        })
    # print(results[-1])
    # label = train_labels[train_labels.tomo_id==tomo_id]
    # print(label)
    

submission_df = pd.DataFrame(results)

# Ensure proper column order
submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]

# Save the submission file
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)


bbox


submission_df


train_labels.merge(submission_df, on="tomo_id")


!rm -rf /kaggle/working/mmdetection

