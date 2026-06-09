#!pip install image-quality


#import torch modules

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
#from torch.utils.tensorboard import SummaryWriter
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim


#loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # Compute the cross-entropy loss
        loss = self.loss_fn(inputs, targets)

        return loss


#dlib face detector

import os
import sys
import time
import cv2
import dlib
import yaml
import logging
import datetime
import glob
import concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        #cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark, mask_face
    
    else:
        return None, None, None

def extract_face(frame):
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = '/kaggle/input/dfc-xception/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    frame_org = frame
    frame_mask = None

    # For now only take the biggest face

    cropped_face, landmarks, _ = extract_aligned_face_dlib(face_detector, 
                                                        face_predictor, 
                                                        frame_org, 
                                                        mask=frame_mask)

    if cropped_face is None:
        cropped_face = cv2.resize(frame, (256,256))

    return cropped_face



#Xception Backbone

import os
import argparse
import logging

import math

from typing import Union

logger = logging.getLogger(__name__)



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

def add_gaussian_noise(ins, mean=0, stddev=0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(Xception, self).__init__()
        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        inc = xception_config["inc"]
        dropout = xception_config["dropout"]

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        # used for iid
        final_channel = 2048
        if self.mode == 'adjust_channel_iid':
            final_channel = 512
            self.mode = 'adjust_channel'
        self.last_linear = nn.Linear(final_channel, self.num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(final_channel, self.num_classes)
            )

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
           
    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  

        return x

    def fea_part1_1(self, x):  
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        return x
    
    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)     
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        return x
    
    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
        return x

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x
     
    def features(self, input):
        x = self.fea_part1(input)    

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)

        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        
        return x

    def classifier(self, features,id_feat=None):
        # for iid
        if self.mode == 'adjust_channel':
            x = features
        else:
            x = self.relu(features)

        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        self.last_emb = x
        # for iid
        if id_feat!=None:
            out = self.last_linear(x-id_feat)
        else:
            out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x


#Xception Detector
import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict
import abc


logger = logging.getLogger(__name__)

class XceptionDetector(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0
        
    def build_backbone(self, config):
        # prepare the backbone
        model_config = config['backbone_config']
        backbone = Xception(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self, config):
        return CrossEntropyLoss()
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        overall_loss = loss
        loss_dict = {'overall': overall_loss, 'cls': loss,}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        self.video_names = []
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict


#SPSL Detector

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

class SpslDetector(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        # prepare the backbone
        model_config = config['backbone_config']
        backbone = Xception(model_config)

        # To get a good performance, use the ImageNet-pretrained Xception model
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}

        # remove conv1 from state_dict
        conv1_data = state_dict.pop('conv1.weight')

        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model from {}'.format(config['pretrained']))

        # copy on conv1
        # let new conv1 use old param to balance the network
        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)  # average across the RGB channels
        backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)  # repeat the averaged weights across the 4 new channels
        logger.info('Copy conv1 from pretrained model')
        return backbone

    def build_loss(self, config):
        return CrossEntropyLoss()
    
    def features(self, data_dict: dict, phase_fea) -> torch.tensor:
        features = torch.cat((data_dict['image'], phase_fea), dim=1)
        return self.backbone.features(features)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        self.video_names = []
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the phase features
        phase_fea = self.phase_without_amplitude(data_dict['image'])
        # bp
        features = self.features(data_dict, phase_fea)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict

    def phase_without_amplitude(self, img):
        # Convert to grayscale
        gray_img = torch.mean(img, dim=1, keepdim=True) # shape: (batch_size, 1, 256, 256)
        # Compute the DFT of the input signal
        X = torch.fft.fftn(gray_img,dim=(-1,-2))
        #X = torch.fft.fftn(img)
        # Extract the phase information from the DFT
        phase_spectrum = torch.angle(X)
        # Create a new complex spectrum with the phase information and zero magnitude
        reconstructed_X = torch.exp(1j * phase_spectrum)
        # Use the IDFT to obtain the reconstructed signal
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X,dim=(-1,-2)))
        # reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X))
        return reconstructed_x


#f3net Detector

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

logger = logging.getLogger(__name__)

class F3netDetector(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        # modules only use in FAD
        img_size = config['resolution']
        self.FAD_head = FAD_Head(img_size)

    def build_backbone(self, config):
        # prepare the backbone
        model_config = config['backbone_config']
        backbone = Xception(model_config)
        
        # To get a good performance, use the ImageNet-pretrained Xception model
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        conv1_data = state_dict['conv1.weight'].data
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model from {}'.format(config['pretrained']))

        # copy on conv1
        # let new conv1 use old param to balance the network
        backbone.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
           backbone.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0
        logger.info('Copy conv1 from pretrained model')
        return backbone
    
    def build_loss(self, config):
        return CrossEntropyLoss()
    
    def features(self, data_dict: dict) -> torch.tensor:
        fea_FAD = self.FAD_head(data_dict['image']) # [B, 12, 256, 256]
        return self.backbone.features(fea_FAD)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict


# ===================================== other modules for F3Net # =====================================


# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        return out

# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


#Inference
import numpy as np

from PIL import Image


"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
import argparse


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_image(image):
        # Open the image file
        # Convert the image to a numpy array
        image_data_rgb = np.array(image)

        # Convert from (height, width, 3) to (3, height, width)
        image_data = np.transpose(image_data_rgb, (2, 0, 1))

        # Rescale the values back to the range (-1, 1)
        image_data = np.round(image_data, 4)

        image_data = image_data.astype(np.float32)
        image_data = (image_data / 127.5) - 1.0  # Reverse the normalization
    

        # Convert to a PyTorch tensor
        tensor_image = torch.tensor(image_data).cuda()  # Assuming you want to keep it on the GPU
        return tensor_image

def init_model(detector_path="/kaggle/input/dfc-xception/spsl.yaml", 
          weights_path="/kaggle/input/dfc-xception/spsl_c23_e30.pth",
              detector=SpslDetector):
    # init seed
    
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('/kaggle/input/dfc-xception/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    init_seed(config)
    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    config['pretrained'] = "/kaggle/input/dfc-xception/xception-b5690688.pth"

    #model = XceptionDetector(config).to(device)
    model = detector(config).to(device)
    
    epoch = 0
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}  
        
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    model.eval()

    return model


def infer(model, image):
        '''
        label
        image a tensor of shape torch.Size([batch_num,3, 256, 256])
        tensor([0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                0, 1, 1, 0, 1, 1, 0, 0], device='cuda:0')
        landmark
        None
        mask
        None'
        '''
        tensor_image_cuda = load_image(image)
        # Add a new dimension at the beginning to make it [1, 3, 256, 256]
        tensor_image_cuda = tensor_image_cuda.unsqueeze(0)

        data_dict = {}
        data_dict['image'] = tensor_image_cuda   
        data_dict['landmark'] = None
        data_dict['mask'] = None
        pseudo_label = 1 
        data_dict['label'] = torch.tensor([pseudo_label], device=device)

        predictions = inference(model, data_dict)
        return predictions


#main code

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import cv2
import skimage
#import imquality.brisque as brisque

def extract_frames(video_path, max_frames=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 샘플링할 프레임 인덱스 계산
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    frames = []
    current_idx = 0
    frame_idx_set = set(indices)

    while cap.isOpened():
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(e)
        if not ret:
            break
        if current_idx in frame_idx_set:
            frames.append(frame)
            if len(frames) == max_frames:
                break
        current_idx += 1

    cap.release()
    return np.array(frames)

    
import os

print("model loading start")

model_c23 = init_model(detector_path="/kaggle/input/dfc-xception/spsl.yaml", 
          weights_path="/kaggle/input/dfc-xception/spsl_c23.pth",
                      detector=SpslDetector)

model_c40 = init_model(detector_path="/kaggle/input/dfc-xception/f3net.yaml", 
          weights_path="/kaggle/input/dfc-xception/f3net_c40.pth",
                      detector=F3netDetector)

threshold = 100

def apply_c40(img, quality=40):
    # img: numpy array (H, W, 3), BGR format
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    if not result:
        raise Exception("Could not encode image!")
    # Decode it back
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def infer_file(path):
    frames = extract_frames(path)
            
    p_c23 = 0
    p_c40 = 0
    p_c40_2 = 0
    bs = 0
    
    nsample = len(frames)
            
    for frame in frames:
        face = extract_face(frame)
        try:
            pred = infer(model_c23, face)
            prob = pred['prob'].item()
            p_c23 += prob

            pred = infer(model_c40, face)
            prob = pred['prob'].item()
            p_c40 += prob           
            
            pred = infer(model_c40, apply_c40(face))
            prob = pred['prob'].item()
            p_c40_2 += prob
        except Exception as e:
            print(e, end=" ")
            nsample -= 1

    if nsample >= 1:
        p_c23 /= nsample
        p_c40 /= nsample
        p_c40_2 /= nsample
    
    return p_c23, p_c40, p_c40_2


def find_thres(name, y_true, y_score, threshold):
    from sklearn.metrics import roc_curve,accuracy_score, f1_score, precision_recall_curve
    print(name +" threshold")
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    thres = thresholds[best_idx]

    video_preds = [1 if p >= threshold else 0 for p in y_score]

    # 성능 지표
    acc = accuracy_score(y_true, video_preds)
    f1 = f1_score(y_true, video_preds)

    print(threshold, " thres: acc: ", acc, " f1: ", f1)

    video_preds = [1 if p >= thres else 0 for p in y_score]

    # 성능 지표
    acc = accuracy_score(y_true, video_preds)
    f1 = f1_score(y_true, video_preds)

    print(thres, " thres: acc: ", acc, " f1: ", f1)

yt = []
ys_c23 = []
ys_c40 = []
ys_c40_2 = []
ys_m = []
ys_m_2 = []
threshold = 0.2573552876710892

def infer_folder(path = '/kaggle/input/dfc-xception', resfile = "submission.csv"):
    print("processing start")
    video_names = []
    labels = []
    
    for dirname, _, filenames in os.walk(path):
        print(dirname)
        ans=1
        if "0_" in dirname:
            ans = 0
            
        cnt = 0
        print('[',end='')
        for filename in filenames:

            f = os.path.join(dirname, filename)
            
            if not ".mp4" in f and not ".avi" in f:
                continue
            else:
                print("=",end='')
                
            video_names.append(filename)
            pc23, pc40, pc40_2= infer_file(f)
            
            res = 0
            if (pc23 + pc40) / 2 > threshold:
                res = 1
            
            labels.append(res)

            yt.append(ans)
            ys_c23.append(pc23)
            ys_c40.append(pc40)
            ys_c40_2.append(pc40_2)
            ys_m.append((pc23+pc40)/2)
            ys_m_2.append((pc23+pc40_2)/2)

            
        print(']')
    
    print('done')
    df = pd.DataFrame({
        'ID': video_names,
        'label': labels
    })
    print(df)
    df.to_csv(resfile, index=False)
    print("df to csv done")
    

infer_folder("/kaggle/input/dfc-xception/1", "submission_1.csv")
infer_folder("/kaggle/input/dfc-xception/2", "submission_2.csv")
infer_folder("/kaggle/input/dfc-xception/3", "submission_3.csv")
'''
find_thres("c23", yt, ys_c23, threshold)
find_thres("c40", yt, ys_c40, threshold)
find_thres("c40_2", yt, ys_c40_2,threshold)
find_thres("mean", yt, ys_m, threshold)
find_thres("mean_2", yt, ys_m_2, threshold)
'''

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

