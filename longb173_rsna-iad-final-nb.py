!mkdir -p /root/.config/Ultralytics
!cp -r /kaggle/input/rsna-yolov5/Arial.ttf /root/.config/Ultralytics/.

!cp -r /kaggle/input/rsna-yolov5/yolo/yolov5-7.0/classify .
!cp -r /kaggle/input/rsna-yolov5/yolo/yolov5-7.0/data .
!cp -r /kaggle/input/rsna-yolov5/yolo/yolov5-7.0/models .
!cp -r /kaggle/input/rsna-yolov5/yolo/yolov5-7.0/segment .
!cp -r /kaggle/input/rsna-yolov5/yolo/yolov5-7.0/utils .
!cp -r /kaggle/input/rsna-yolov5/yolo/yolov5-7.0/export.py .

!cp -r /kaggle/input/rsna-smp/segmentation_models_pytorch .

!ls


import os
import cv2
import glob
import pydicom
import timm
import gc
import shutil
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# Competition API
import kaggle_evaluation.rsna_inference_server

# yolov5
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# smp
from timm.layers import BatchNormAct2d, SelectAdaptivePool2d
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init


class AneurysmModel(nn.Module):
    def __init__(self, model_name, pretrained, image_size, num_classes):
        super(AneurysmModel, self).__init__()
        if 'convnext' in model_name or 'efficientnet' in model_name or str(image_size) in model_name:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3, num_classes=num_classes)
        else:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3, img_size=image_size, num_classes=num_classes)
    
    @autocast(device_type='cuda')
    def forward(self, x):
        return self.backbone(x)

class AneurysmAuxModel(nn.Module):
    def __init__(self, encoder_name=None, decoder_name=None, encoder_weights=None, encoder_feat_dims=None, num_classes=14, test_mode=False):
        super(AneurysmAuxModel, self).__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        self.test_mode = test_mode
        self.encoder_feat_dims = encoder_feat_dims

        if 'mit_b' in encoder_name:
            if encoder_name == "mit_b0":
                self.conv_head = nn.Conv2d(256, self.encoder_feat_dims, 1, 1, bias=False)
            elif encoder_name in ["mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"]: 
                self.conv_head = nn.Conv2d(512, self.encoder_feat_dims, 1, 1, bias=False)
            else:
                raise ValueError()
            self.bn2 = BatchNormAct2d(num_features=self.encoder_feat_dims)
            self.global_pool = SelectAdaptivePool2d()
            if decoder_name == 'fpn':
                self.decoder = FPNDecoder(
                    encoder_channels=self.encoder.out_channels,
                    encoder_depth=5,
                    pyramid_channels=256,
                    segmentation_channels=128,
                    dropout=0.2,
                    merge_policy="add",
                )
                self.segmentation_head = SegmentationHead(
                    in_channels=self.decoder.out_channels,
                    out_channels=2,
                    kernel_size=1,
                    upsampling=4,
                    activation=None,
                )
            elif decoder_name == 'unet':
                decoder_channels = (256, 128, 64, 32, 16)
                self.decoder = UnetDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=5,
                    use_norm="batchnorm",
                    add_center_block=False,
                    attention_type=None,
                )
                self.segmentation_head = SegmentationHead(
                    in_channels=decoder_channels[-1],
                    out_channels=2,
                    kernel_size=3,
                    activation=None,
                )
            else:
                raise ValueError()

        elif "timm-efficientnet" in encoder_name:
            self.conv_head = self.encoder.conv_head
            self.bn2 = self.encoder.bn2
            self.global_pool = self.encoder.global_pool
            if decoder_name == 'unet':
                decoder_channels = (256, 128, 64, 32, 16)
                self.decoder = UnetDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=5,
                    use_norm="batchnorm",
                    add_center_block=False,
                    attention_type=None,
                )
                self.segmentation_head = SegmentationHead(
                    in_channels=decoder_channels[-1],
                    out_channels=2,
                    kernel_size=3,
                    activation=None,
                )
            else:
                raise ValueError()

        
        self.fc = nn.Linear(self.encoder_feat_dims, 1024, bias=True)
        self.cls_head = nn.Linear(1024, num_classes, bias=True)

        init.initialize_head(self.fc)
        init.initialize_head(self.cls_head)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    @autocast(device_type='cuda')
    def forward(self, x):
        x = self.encoder(x)
        y_cls = self.conv_head(x[-1])
        y_cls = self.bn2(y_cls)
        y_cls = self.global_pool(y_cls)
        y_cls = y_cls.view(-1, self.encoder_feat_dims)
        y_cls = self.fc(y_cls)
        y_cls = F.relu(y_cls)
        y_cls = F.dropout(y_cls, p=0.5, training=self.training)
        y_cls = self.cls_head(y_cls)

        if self.test_mode:
            return y_cls
        else:
            y_seg = self.decoder(x)
            y_seg = self.segmentation_head(y_seg)
            return y_cls, y_seg


def dicom2image_process(dcm_path):
    dcm_file = pydicom.dcmread(dcm_path, force=True)
    ImagePositionPatient = dcm_file.get('ImagePositionPatient', [0,0,0])
    z = ImagePositionPatient[2]
    Modality = dcm_file.get('Modality', 'MR')
    image = dcm_file.pixel_array.astype(np.float32)
    if Modality == 'CT':
        window_center = 40
        window_width = 450
        image_min = window_center - window_width // 2
        image_max = window_center + window_width // 2
        image = np.clip(image, image_min, image_max)

    image_min = np.min(image)
    image_max = np.max(image)
    image = (image - image_min) / (image_max - image_min + 1e-7)
    image = (image * 255).astype(np.uint8)

    return image, z, Modality

def img2tensor_yolov5(img, imgsz, stride, device):
    img_tensor = letterbox(img, imgsz, stride=stride, auto=True)[0]
    img_tensor = img_tensor.transpose((2, 0, 1))
    img_tensor = np.ascontiguousarray(img_tensor)

    img_tensor = torch.from_numpy(img_tensor).to(device)
    img_tensor = img_tensor.half()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


global exp3_mit_models, exp3_mit_folds, exp5_mit_models, exp5_mit_folds, vit_models, vit_exps, eva_models, eva_exps, brain_det_model

exp3_mit_models = {}
exp3_mit_folds = [0,1,2,3]
for fold in exp3_mit_folds:
    model = AneurysmAuxModel(
        encoder_name='mit_b4',
        decoder_name='fpn',
        encoder_weights=None, 
        encoder_feat_dims=1280, 
        num_classes=14,
        test_mode=True
    )
    model.cuda()
    model.load_state_dict(torch.load('/kaggle/input/rsna-iad-ckpts/exp3_aux_mit_b4/mit_b4_384_fold{}.pt'.format(fold), weights_only=True))
    model.eval()
    exp3_mit_models[fold] = model

exp5_mit_models = {}
exp5_mit_folds = [0,1,2,3,4]
for fold in exp5_mit_folds:
    model = AneurysmAuxModel(
        encoder_name='mit_b4',
        decoder_name='fpn',
        encoder_weights=None, 
        encoder_feat_dims=1280, 
        num_classes=14,
        test_mode=True
    )
    model.cuda()
    model.load_state_dict(torch.load('/kaggle/input/rsna-iad-ckpts/exp5_aux_pseudo_mit_b4/mit_b4_384_fold{}.pt'.format(fold), weights_only=True))
    model.eval()
    exp5_mit_models[fold] = model

vit_models = {}
vit_exps = ['vit_exp2', 'vit_exp4']
for exp in vit_exps:
    model = AneurysmModel(
        model_name='vit_large_patch14_clip_336.openai_ft_in12k_in1k', 
        pretrained=False, 
        image_size=384,
        num_classes=14,
    )
    model.cuda()
    if exp == 'vit_exp2':
        model.load_state_dict(torch.load('/kaggle/input/rsna-iad-ckpts/exp2_exp4_vit_eva/vit_large_patch14_clip_336.openai_ft_in12k_in1k_384_full_epoch4_exp2.pt', weights_only=True))
    elif exp == 'vit_exp4':
        model.load_state_dict(torch.load('/kaggle/input/rsna-iad-ckpts/exp2_exp4_vit_eva/vit_large_patch14_clip_336.openai_ft_in12k_in1k_384_full_epoch4_exp4.pt', weights_only=True))
    else:
        raise ValueError()
    model.eval()
    vit_models[exp] = model

eva_models = {}
eva_exps = ['eva_exp2', 'eva_exp4']
for exp in eva_exps:  
    model = AneurysmModel(
        model_name='eva_large_patch14_336.in22k_ft_in22k_in1k', 
        pretrained=False, 
        image_size=384,
        num_classes=14,
    )
    model.cuda()
    if exp == 'eva_exp2':
        model.load_state_dict(torch.load('/kaggle/input/rsna-iad-ckpts/exp2_exp4_vit_eva/eva_large_patch14_336.in22k_ft_in22k_in1k_384_full_epoch10_exp2.pt', weights_only=True))
    elif exp == 'eva_exp4':
        model.load_state_dict(torch.load('/kaggle/input/rsna-iad-ckpts/exp2_exp4_vit_eva/eva_large_patch14_336.in22k_ft_in22k_in1k_384_full_epoch10_exp4.pt', weights_only=True))
    else:
        raise ValueError()
    model.eval()
    eva_models[exp] = model

brain_det_model = DetectMultiBackend('/kaggle/input/rsna-iad-ckpts/brain_det_yolov5n_640.pt', device=select_device(0), dnn=False, fp16=True)
brain_det_model.eval()
brain_det_model.warmup(imgsz=(1, 3, 640, 640))


def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    try:
        IMAGE_SIZE = 384
        BATCH_SIZE = 64
        CT_NUM_SLICE_MAX = 300
        CROP_RAT = 0.75
        IOU_THRES = 0.375
        LABEL_COLS = [
            'Left Infraclinoid Internal Carotid Artery',
            'Right Infraclinoid Internal Carotid Artery',
            'Left Supraclinoid Internal Carotid Artery',
            'Right Supraclinoid Internal Carotid Artery',
            'Left Middle Cerebral Artery',
            'Right Middle Cerebral Artery',
            'Anterior Communicating Artery',
            'Left Anterior Cerebral Artery',
            'Right Anterior Cerebral Artery',
            'Left Posterior Communicating Artery',
            'Right Posterior Communicating Artery',
            'Basilar Tip',
            'Other Posterior Circulation',
            'Aneurysm Present'
        ]

        """Make a prediction."""
        series_id = os.path.basename(series_path)
    
        dcm_filepaths = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dcm_filepaths.append(os.path.join(root, file))
        images = []
        z_pos = []
        modalities = []
        for dcm_path in dcm_filepaths:
            image, z, Modality = dicom2image_process(dcm_path)
            if image.ndim == 2:
                images.append(image)
                z_pos.append(z)
                modalities.append(Modality)
            elif image.ndim == 3:
                for i in range(image.shape[0]):
                    image_i = image[i,:,:]
                    images.append(image_i)
                    z_pos.append(i)
                    modalities.append(Modality)
            del image
        images = np.array(images)
        z_pos = np.array(z_pos)
        modalities = np.array(modalities)
    
        brain_image = np.mean(images, 0)
        brain_image_min = np.min(brain_image)
        brain_image_max = np.max(brain_image)
        brain_image = (brain_image - brain_image_min) / (brain_image_max - brain_image_min + 1e-7)
        brain_image = (brain_image * 255).astype(np.uint8)

        brain_height, brain_width = brain_image.shape[0:2]
        brain_image = np.stack([brain_image, brain_image, brain_image], -1)
        brain_image = img2tensor_yolov5(brain_image, imgsz=640, stride=32, device=select_device(0))
        boxes = []
        labels = []
        scores = []
        with autocast("cuda"), torch.no_grad():
            brain_dets = brain_det_model(brain_image, augment=False, visualize=False)
            brain_dets = non_max_suppression(brain_dets, conf_thres=0.1, iou_thres=0.4, classes=None, agnostic=True, max_det=100)
    
            for det in brain_dets:
                if len(det):
                    det[:, :4] = scale_boxes(brain_image.shape[2:], det[:, :4], (brain_height, brain_width)).round()
                    det = det.data.cpu().numpy()
                    for d in det:
                        x1, y1, x2, y2, score, label = int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[4]), int(d[5])
                        x1 = min(max(0, x1), brain_width)
                        x2 = min(max(0, x2), brain_width)
                        y1 = min(max(0, y1), brain_height)
                        y2 = min(max(0, y2), brain_height)
                        if x1 >= x2 or y1 >= y2:
                            continue
                        boxes.append([x1, y1, x2, y2])
                        scores.append(score)
                        labels.append(label)
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            labels = np.array(labels)
            
            idxs = np.argsort(scores)[::-1]
            scores = scores[idxs]
            boxes = boxes[idxs,:]
            labels = labels[idxs]
            brain_x1, brain_y1, brain_x2, brain_y2 = boxes[0,:]
            if labels[0] == 0:
                brain_class_name = 'brain'
            else:
                brain_class_name = 'abnormal'
        else:
            brain_x1, brain_y1, brain_x2, brain_y2, brain_class_name = 0, 0, brain_width, brain_height, 'brain'

        if CROP_RAT != 1:
            brain_xc = 0.5*(brain_x1+brain_x2)
            brain_yc = 0.5*(brain_y1+brain_y2)
            brain_width = CROP_RAT*(brain_x2-brain_x1)
            brain_height = CROP_RAT*(brain_y2-brain_y1)
            crop_x1 = int(brain_xc - 0.5*brain_width)
            crop_x2 = int(brain_xc + 0.5*brain_width)
            crop_y1 = int(brain_yc - 0.5*brain_height)
            crop_y2 = int(brain_yc + 0.5*brain_height)
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = brain_x1, brain_y1, brain_x2, brain_y2

        kernel = np.ones((20,20),np.uint8)  
        
        idxs = np.argsort(z_pos)
        z_pos = z_pos[idxs]
        images = images[idxs,:,:]
        modalities = modalities[idxs]
        
        if images.shape[0] > CT_NUM_SLICE_MAX and 'CT' in modalities and brain_class_name == 'brain':
            ious = []
            for image in images:
                image_erosion = cv2.erode(image, kernel, iterations=1)
                arr = np.nonzero(image_erosion)
                if len(arr[0]) > 0 and len(arr[1]) > 0:
                    arr = np.array(arr)
                    min_pos=np.array(arr).min(axis=1)
                    max_pos=np.array(arr).max(axis=1)+1
                
                    x1, y1, x2, y2 = min_pos[1], min_pos[0], max_pos[1], max_pos[0]
                    iou = bb_intersection_over_union([x1, y1, x2, y2], [brain_x1, brain_y1, brain_x2, brain_y2])
                else:
                    iou = 0
                ious.append(iou)
            ious = np.array(ious, dtype=np.float32)
            idxs1 = np.where(ious > IOU_THRES)[0]
            if len(idxs1) > 0:
                images = images[idxs1,:,:]
                ious = ious[idxs1]
                z_pos = z_pos[idxs1]

                # if len(ious) > CT_NUM_SLICE_MAX:
                #     idxs2 = np.argsort(ious)[::-1][0:CT_NUM_SLICE_MAX]
                #     images = images[idxs2,:,:]
                #     z_pos = z_pos[idxs2]

                idxs3 = np.argsort(z_pos)
                images = images[idxs3,:,:]
                
        images = images[:,crop_y1:crop_y2,crop_x1:crop_x2]

        images = torch.from_numpy(images).unsqueeze(1).type(torch.float16).cuda()
        images = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=True).squeeze().type(torch.float16).cpu()
        images /= 255.0

        del z_pos
        del modalities
        
        exp3_mit_ser_preds = []
        for start in range(0, images.shape[0], BATCH_SIZE):
            end = min(start+BATCH_SIZE, images.shape[0])
            batch_image = []
            for i in range(start, end, 1):
                if i == 0:
                    pre_i = i
                else:
                    pre_i = i-1
                if i == images.size(0) - 1:
                    next_i = i
                else:
                    next_i = i+1
                image = images[[pre_i, i, next_i],:,:]
                batch_image.append(image)
            batch_image = torch.stack(batch_image, 0).cuda()

            batch_pred = []
            with autocast("cuda"), torch.no_grad():
                for fold in exp3_mit_folds:
                    pred = torch.sigmoid(exp3_mit_models[fold](batch_image))
                    pred = pred.data.cpu().numpy()
                    batch_pred.append(pred)
            batch_pred = np.array(batch_pred).mean(0)
            exp3_mit_ser_preds.append(batch_pred)
            del batch_image

        exp3_mit_ser_preds = np.concatenate(exp3_mit_ser_preds, 0)
        best_idxs = []
        for i in range(len(LABEL_COLS)):
            best_idxs.append(int(np.argmax(exp3_mit_ser_preds[:,i])))
        best_idxs = np.unique(best_idxs)
        exp3_mit_ser_preds = exp3_mit_ser_preds[best_idxs,:]

        batch_image = []
        for i in best_idxs:
            if i == 0:
                pre_i = i
            else:
                pre_i = i-1
            if i == images.size(0) - 1:
                next_i = i
            else:
                next_i = i+1
            image = images[[pre_i, i, next_i],:,:]
            batch_image.append(image)
        batch_image = torch.stack(batch_image, 0).cuda()

        exp5_mit_ser_preds = []
        with autocast("cuda"), torch.no_grad():
            for fold in exp5_mit_folds:
                pred = torch.sigmoid(exp5_mit_models[fold](batch_image))
                pred = pred.data.cpu().numpy()
                exp5_mit_ser_preds.append(pred)
        exp5_mit_ser_preds = np.array(exp5_mit_ser_preds).mean(0)

        vit_ser_preds = []
        with autocast("cuda"), torch.no_grad():
            for exp in vit_exps:
                pred = torch.sigmoid(vit_models[exp](batch_image))
                pred = pred.data.cpu().numpy()
                vit_ser_preds.append(pred)
        vit_ser_preds = np.array(vit_ser_preds).mean(0)

        eva_ser_preds = []
        with autocast("cuda"), torch.no_grad():
            for exp in eva_exps:
                pred = torch.sigmoid(eva_models[exp](batch_image))
                pred = pred.data.cpu().numpy()
                eva_ser_preds.append(pred)
        eva_ser_preds = np.array(eva_ser_preds).mean(0)

        ######################### hflip #########################
        batch_image = torch.flip(batch_image, dims=(3,))

        exp3_mit_hflip_ser_preds = []
        with autocast("cuda"), torch.no_grad():
            for fold in exp3_mit_folds:
                pred = torch.sigmoid(exp3_mit_models[fold](batch_image))
                pred = pred.data.cpu().numpy()
                exp3_mit_hflip_ser_preds.append(pred)
        exp3_mit_hflip_ser_preds = np.array(exp3_mit_hflip_ser_preds).mean(0)
        exp3_mit_hflip_ser_preds = exp3_mit_hflip_ser_preds[:,[1,0,3,2,5,4,6,8,7,10,9,11,12,13]]

        exp5_mit_hflip_ser_preds = []
        with autocast("cuda"), torch.no_grad():
            for fold in exp5_mit_folds:
                pred = torch.sigmoid(exp5_mit_models[fold](batch_image))
                pred = pred.data.cpu().numpy()
                exp5_mit_hflip_ser_preds.append(pred)
        exp5_mit_hflip_ser_preds = np.array(exp5_mit_hflip_ser_preds).mean(0)
        exp5_mit_hflip_ser_preds = exp5_mit_hflip_ser_preds[:,[1,0,3,2,5,4,6,8,7,10,9,11,12,13]]

        del batch_image
        del images

        ser_preds = 0.125*exp3_mit_ser_preds + 0.125*exp5_mit_ser_preds + 0.125*exp3_mit_hflip_ser_preds + 0.125*exp5_mit_hflip_ser_preds + 0.25*vit_ser_preds + 0.25*eva_ser_preds
        
        ser_preds = list(np.max(ser_preds, 0))

        predictions = pl.DataFrame(
            data=[ser_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    except Exception as e:
        print(e)
        ser_preds = [0.0] * len(LABEL_COLS)
        predictions = pl.DataFrame(
            data=[ser_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))


!rm -rf /kaggle/working/classify /kaggle/working/data /kaggle/working/models /kaggle/working/segment /kaggle/working/utils /kaggle/working/segmentation_models_pytorch /kaggle/working/export.py
!rm -rf /kaggle/working/__pycache__

