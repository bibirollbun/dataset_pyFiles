from IPython.display import clear_output

!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/pycocotools-2.0.6-cp310-cp310-linux_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/torch-1.12.1+cu116-cp310-cp310-linux_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/torchvision-0.13.1+cu116-cp310-cp310-linux_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/mmcv-2.0.1-cp310-cp310-manylinux1_x86_64.whl 
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/openmim-0.3.9-py2.py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/mmengine-0.7.4-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmdetectron-31-wheel/addict-2.4.0-py3-none-any.whl
!pip install yapf==0.40.1
!pip install terminaltables
!pip install setuptools==69.5.1
!pip install --no-index --no-deps /kaggle/input/mmpretrain/einops-0.6.1-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmpretrain/mat4py-0.5.0-py2.py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmpretrain/ordered_set-4.1.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmpretrain/model_index-0.1.11-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmpretrain/modelindex-0.0.2-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/mmpretrain/mmpretrain-1.0.0rc8-py2.py3-none-any.whl
clear_output()


from tqdm.notebook import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

from glob import glob
import matplotlib.pyplot as plt


!git clone https://github.com/open-mmlab/mmdetection.git
%cd /kaggle/working/mmdetection
!pip install -e .

clear_output()


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

_base_ = '../common/lsj-200e_coco-detection.py'

dataset_type = 'CocoDataset'
data_root = '/kaggle/input/byu-coco-dataset/dataset_json/'
backend_args = None
classes = ('motor',)

image_size = (960, 960)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

# model settings
model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments),
    backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=4,  # simply change this from 2 to 16 for 50e - 400e training.
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations_train.json',
            metainfo=dict(classes=classes),
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))



val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations_valid.json',
        data_prefix=dict(img='valid2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader
val_evaluator = dict(
ann_file=data_root + 'annotations_valid.json', metric=['bbox'], format_only=False, backend_args=backend_args)

test_evaluator = val_evaluator

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

max_epochs = 20

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.067, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[18, 24],  # the real step is [18*5, 24*5]
        gamma=0.1)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)  # the real epoch is 28*5=140
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
auto_scale_lr = dict(base_batch_size=16)

fp16=dict(loss_scale=512.)
work_dir = '../model_output'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP_50",
        patience=7,
        min_delta=0.005),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
        save_best='coco/bbox_mAP_50'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
resume_from = None


# !python tools/train.py configs/BYU/custom_center_netconfig.py


!bash ./tools/dist_train.sh\
    configs/BYU/custom_center_netconfig.py\
    2


!rm -rf /kaggle/working/mmdetection

