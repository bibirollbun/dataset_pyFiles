!git clone https://github.com/ShihuaHuang95/DEIM.git


!pip install -r /kaggle/working/DEIM/requirements.txt


#change checkpoint save frequency, due to Kaggle  out of space failure.


!cat /kaggle/working/DEIM/configs/base/dfine_hgnetv2.yml


%%writefile /kaggle/working/DEIM/configs/base/dfine_hgnetv2.yml
task: detection

model: DEIM
criterion: DEIMCriterion
postprocessor: PostProcessor

use_focal_loss: True
eval_spatial_size: [640, 640] # h w
checkpoint_freq: 10   #  4    # save freq

DEIM:
  backbone: HGNetv2
  encoder: HybridEncoder
  decoder: DFINETransformer

# Add, default for step lr scheduler 
lrsheduler: flatcosine
lr_gamma: 1
warmup_iter: 500
flat_epoch: 4000000
no_aug_epoch: 0

HGNetv2:
  pretrained: True
  local_model_dir: ../RT-DETR-main/D-FINE/weight/hgnetv2/

HybridEncoder:
  in_channels: [512, 1024, 2048]
  feat_strides: [8, 16, 32]

  # intra
  hidden_dim: 256
  use_encoder_idx: [2]
  num_encoder_layers: 1
  nhead: 8
  dim_feedforward: 1024
  dropout: 0.
  enc_act: 'gelu'

  # cross
  expansion: 1.0
  depth_mult: 1
  act: 'silu'


DFINETransformer:
  feat_channels: [256, 256, 256]
  feat_strides: [8, 16, 32]
  hidden_dim: 256
  num_levels: 3

  num_layers: 6
  eval_idx: -1
  num_queries: 300

  num_denoising: 100
  label_noise_ratio: 0.5
  box_noise_scale: 1.0

  # NEW
  reg_max: 32
  reg_scale: 4

  # Auxiliary decoder layers dimension scaling
  # "eg. If num_layers: 6 eval_idx: -4,
  # then layer 3, 4, 5 are auxiliary decoder layers."
  layer_scale: 1  # 2


  num_points: [3, 6, 3] # [4, 4, 4] [3, 6, 3]
  cross_attn_method: default # default, discrete
  query_select_method: default # default, agnostic


PostProcessor:
  num_top_queries: 300


DEIMCriterion:
  weight_dict: {loss_vfl: 1, loss_bbox: 5, loss_giou: 2, loss_fgl: 0.15, loss_ddf: 1.5}
  losses: ['vfl', 'boxes', 'local']
  alpha: 0.75
  gamma: 2.0
  reg_max: 32

  matcher:
    type: HungarianMatcher
    weight_dict: {cost_class: 2, cost_bbox: 5, cost_giou: 2}
    alpha: 0.25
    gamma: 2.0











!cat /kaggle/working/DEIM/configs/deim_dfine/dfine_hgnetv2_l_coco.yml


!cat /kaggle/working/DEIM/configs/dataset/custom_detection.yml


!cat /kaggle/input/my-custom-detect-yml/my_custom_detection.yml


!cat /kaggle/working/DEIM/configs/deim_dfine/dfine_hgnetv2_x_coco.yml





%%writefile /kaggle/working/DEIM/configs/deim_dfine/deim_hgnetv2_x_coco_byu.yml
__include__: [
  '../dataset/my_custom_detection.yml',
  '../runtime.yml',
  '../base/my_dataloader.yml',
  '../base/optimizer.yml',
  '../base/dfine_hgnetv2.yml',
]

output_dir: ./output/dfine_hgnetv2_x_coco


DEIM:
  backbone: HGNetv2

HGNetv2:
  name: 'B5'
  return_idx: [1, 2, 3]
  freeze_stem_only: True
  freeze_at: 0
  freeze_norm: True

HybridEncoder:
  # intra
  hidden_dim: 384
  dim_feedforward: 2048

DFINETransformer:
  feat_channels: [384, 384, 384]
  reg_scale: 8

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.0000025
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
      weight_decay: 0.

  lr: 0.00025
  betas: [0.9, 0.999]
  weight_decay: 0.000125


# Increase to search for the optimal ema
epoches: 80 # 72 + 2n
train_dataloader:
  dataset:
    transforms:
      policy:
        epoch: 72
  collate_fn:
    stop_epoch: 72
    ema_restart_decay: 0.9998
    base_size_repeat: 3





%%writefile /kaggle/working/DEIM/configs/deim_dfine/deim_hgnetv2_l_coco_byu.yml
__include__: [
  '../dataset/my_custom_detection.yml',
  '../runtime.yml',
  '../base/my_dataloader.yml',
  '../base/optimizer.yml',
  '../base/dfine_hgnetv2.yml',
]

output_dir: ./outputs/dfine_hgnetv2_l_coco


HGNetv2:
  name: 'B4'
  return_idx: [1, 2, 3]
  freeze_stem_only: True
  freeze_at: 0
  freeze_norm: True

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.0000125
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
      weight_decay: 0.

  lr: 0.00025
  betas: [0.9, 0.999]
  weight_decay: 0.000125


# Increase to search for the optimal ema
epoches: 88 # 72 + 2n
train_dataloader:
  dataset:
    transforms:
      policy:
        #epoch: 72
        epoch: 80
  collate_fn:
    #stop_epoch: 72
    stop_epoch: 80
    ema_restart_decay: 0.9999
    base_size_repeat: 4


#deim_hgnetv2_l_coco_byu.yml  out of memory, use small model 


!cat /kaggle/working/DEIM/configs/deim_dfine/dfine_hgnetv2_s_coco.yml


%%writefile /kaggle/working/DEIM/configs/deim_dfine/deim_hgnetv2_s_coco_byu.yml
__include__: [
  '../dataset/my_custom_detection.yml',
  '../runtime.yml',
  #'../base/dataloader.yml',
  '../base/my_dataloader.yml',
  '../base/optimizer.yml',
  '../base/dfine_hgnetv2.yml',
]

output_dir: ./output/dfine_hgnetv2_s_coco


DEIM:
  backbone: HGNetv2

HGNetv2:
  name: 'B0'
  return_idx: [1, 2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True

DFINETransformer:
  num_layers: 3  # 4 5 6
  eval_idx: -1  # -2 -3 -4

HybridEncoder:
  in_channels: [256, 512, 1024]
  hidden_dim: 256
  depth_mult: 0.34
  expansion: 0.5

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.0001
    -
      params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: 0.0001
      weight_decay: 0.
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.

  lr: 0.0002
  betas: [0.9, 0.999]
  weight_decay: 0.0001


# Increase to search for the optimal ema
epoches: 132 # 120 + 4n
train_dataloader:
  dataset:
    transforms:
      policy:
        epoch: 120
  collate_fn:
    stop_epoch: 120
    ema_restart_decay: 0.9999
    base_size_repeat: 20


!cat /kaggle/working/DEIM/configs/base/dataloader.yml


%%writefile  /kaggle/working/DEIM/configs/base/my_dataloader.yml

train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640], }
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
      policy:
        name: stop_epoch
        epoch: 64 # epoch in [71, ~) stop `ops`
        ops: ['Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']

  collate_fn:
    type: BatchImageCollateFunction
    base_size: 640
    base_size_repeat: 3
    stop_epoch: 56 # epoch in [72, ~) stop `multiscales`

  shuffle: True
  total_batch_size: 8 # total batch size equals to 32 (4 * 8)
  num_workers: 4


val_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [640, 640], }
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
  shuffle: False
  total_batch_size: 8
  num_workers: 4


%%writefile  /kaggle/working/DEIM/configs/dataset/my_custom_detection.yml
task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

num_classes: 1 # your dataset classes
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /kaggle/input/byu-coco-dataset-no-normalized/dataset_json/train
    ann_file: /kaggle/input/byu-coco-dataset-no-normalized/dataset_json/train/annotations_train.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 2
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction


val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /kaggle/input/byu-coco-dataset-no-normalized/dataset_json/val
    ann_file: /kaggle/input/byu-coco-dataset-no-normalized/dataset_json/val/annotations_valid.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 2
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction


!cp /kaggle/input/my-deim-wts/deim_dfine_hgnetv2_l_coco_50e.pth .


!cp '/kaggle/input/deim-dfine-s-wts/deim_dfine_hgnetv2_s_coco_120e (1).pth' ./deim_dfine_hgnetv2_s_coco_120e.pth


!cp '/kaggle/input/my-deim-wts/deim_dfine_hgnetv2_x_coco_50e.pth'  ./deim_dfine_hgnetv2_x_coco_50e.pth


import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}") 


# Explicitly move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dn=torch.cuda.get_device_name('cuda')
print(dn)


available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(available_gpus)





#!cp /kaggle/input/my-deim-train-wts-demo/outputs/dfine_hgnetv2_l_coco/best*.pth /kaggle/working/ 


#!cp /kaggle/input/my-deim-train-wts-demo/outputs/dfine_hgnetv2_l_coco/last.pth /kaggle/working/ 


#! torchrun --master_port=7777 --nproc_per_node=2 /kaggle/working/DEIM/train.py -c /kaggle/working/DEIM/configs/deim_dfine/deim_hgnetv2_l_coco_byu.yml --use-amp --seed=0 -t  deim_dfine_hgnetv2_l_coco_50e.pth 


#! python   /kaggle/working/DEIM/train.py -c /kaggle/working/DEIM/configs/deim_dfine/deim_hgnetv2_s_coco_byu.yml --use-amp --seed=0 -t  deim_dfine_hgnetv2_s_coco_120e.pth


!cp /kaggle/input/my-deim-train-wts-demo/output/dfine_hgnetv2_x_coco/checkpoint0067.pth /kaggle/working/ 

!cp /kaggle/input/my-deim-train-wts-demo/output/dfine_hgnetv2_x_coco/best_stg1.pth /kaggle/working/define_hgnetv2_x_coco.pth 


! python   /kaggle/working/DEIM/train.py -c /kaggle/working/DEIM/configs/deim_dfine/deim_hgnetv2_x_coco_byu.yml --use-amp --seed=0 -t define_hgnetv2_x_coco.pth 




