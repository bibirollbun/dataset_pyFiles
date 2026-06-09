import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import recall_score


data_dir = '/kaggle/input/cassava-leaf-disease-classification'
pd_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

X = pd_train
y = pd_train['label']

skf = StratifiedKFold(n_splits=6, random_state=42, shuffle=True)

pd_train['fold'] = -1

for i, (trn_idx, vid_idx) in enumerate(skf.split(X, y)):
    pd_train.loc[vid_idx, 'fold'] = i

pd_train.to_csv(os.path.join('/kaggle/working/', 'pd_folds.csv'), index=False)


class CassavaDataset(Dataset):
    def __init__(self, csv, img_height, img_width, transform):
        self.csv = csv.reset_index()
        self.img_ids = csv['image_id'].values
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = Image.open(data_dir + f'/train_images/{img_id}')
        img = np.array(img)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']

        label = self.csv.iloc[index].label

        return img, label


train_augmentation = A.Compose(
    [
        A.Rotate(20),
        A.Normalize(mean=0.5, std=1),
        ToTensorV2()
    ]
)

valid_augmentation = A.Compose(
    [
        A.Normalize(mean=0.5, std=1),
        ToTensorV2()
    ]
)


trn_idx = pd_train.loc[pd_train['fold'].isin([0, 1, 2, 3, 4])].index
vld_idx = pd_train.loc[pd_train['fold'].isin([5])].index

trn_dataset = CassavaDataset(csv=pd_train.loc[trn_idx][:], img_height=600, img_width=800, transform=train_augmentation)
vld_dataset = CassavaDataset(csv=pd_train.loc[vld_idx][:], img_height=600, img_width=800, transform=valid_augmentation)

trn_loader = DataLoader(trn_dataset, shuffle=True, num_workers=4, batch_size=16)
vld_loader = DataLoader(vld_dataset, shuffle=True, num_workers=4, batch_size=16)


# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import torchvision.models as models
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import types
import re

#################################################################
# You can find the definitions of those models here:
# https://github.com/pytorch/vision/blob/master/torchvision/models
#
# To fit the API, we usually added/redefined some methods and
# renamed some attributs (see below for each models).
#
# However, you usually do not need to see the original model
# definition from torchvision. Just use `print(model)` to see
# the modules and see bellow the `model.features` and
# `model.classifier` definitions.
#################################################################

__all__ = [
    'alexnet',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'inceptionv3',
    'squeezenet1_0', 'squeezenet1_1',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19'
]

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth',
    'densenet169': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth',
    'densenet201': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth',
    'densenet161': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth',
    'inceptionv3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    # 'vgg16_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth',
    # 'vgg19_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth'
}

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in ['inceptionv3']:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]

pretrained_settings = {}

for model_name in __all__:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 1000
        }
    }

# for model_name in ['vgg16', 'vgg19']:
#     pretrained_settings[model_name]['imagenet_caffe'] = {
#         'url': model_urls[model_name + '_caffe'],
#         'input_space': 'BGR',
#         'input_size': input_sizes[model_name],
#         'input_range': [0, 255],
#         'mean': [103.939, 116.779, 123.68],
#         'std': [1., 1., 1.],
#         'num_classes': 1000
#     }

def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict

def load_pretrained(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

    state_dict = torch.load('/kaggle/input/bengali/resnet34-333f7ec4.pth', weights_only=True)
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model

#################################################################
# AlexNet

def modify_alexnet(model):
    # Modify attributs
    model._features = model.features
    del model.features
    model.dropout0 = model.classifier[0]
    model.linear0 = model.classifier[1]
    model.relu0 = model.classifier[2]
    model.dropout1 = model.classifier[3]
    model.linear1 = model.classifier[4]
    model.relu1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def alexnet(num_classes=1000, pretrained='imagenet'):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    model = models.alexnet(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['alexnet'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_alexnet(model)
    return model

###############################################################
# DenseNets

def modify_densenets(model):
    # Modify attributs
    model.last_linear = model.classifier
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

###############################################################
# ResNets

def modify_resnets(model):
    # Modify attributs
    model.last_linear = model.fc
    model.fc = None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def resnet18(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-18 model.
    """
    model = models.resnet18(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet18'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model

def resnet34(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-34 model.
    """
    model = models.resnet34(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet34'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


model_name = 'resnet34'
model = resnet34(pretrained='imagenet')
in_features = model.last_linear.in_features
model.last_linear = nn.Linear(in_features, 5)

model.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=7, factor=0.5)


best_score = -1
for epoch in range(1, 10):
    train_loss = []
    model.train()
    
    for inputs, targets in trn_loader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        logits = model(inputs)
        
        loss = loss_fn(logits, targets)
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
    
    val_loss = []
    val_true = []
    val_pred = []
    
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in vld_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            logits = model(inputs)

            loss = loss_fn(logits, targets)
            val_loss.append(loss.item())
    
            logits = logits.cpu().argmax(dim=1).data.numpy()
            
            val_true.append(targets.cpu().numpy())
            val_pred.append(logits)
    
    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)
    
    val_loss = np.mean(val_loss)
    train_loss = np.mean(train_loss)

    final_score = recall_score(val_true, val_pred, average='macro')

    if final_score > best_score:
        best_score = final_score

        state_dict = model.cpu().state_dict()
        model = model.cuda()
        torch.save(state_dict, os.path.join('/kaggle/working/', "model.pt"))

    print(f'epoch {epoch}')
    print(f'train_loss: {train_loss:.5f}; val_loss: {val_loss:.5f}; score: {final_score:.5f}')





pd_test = pd.DataFrame()
pd_test['image_id'] = [f for f in os.listdir(data_dir + f'/test_images/') if os.path.isfile(os.path.join(data_dir + f'/test_images/', f))]

pd_submit = pd.DataFrame()
pd_submit['image_id'] = pd_test['image_id']
pd_submit['label'] = -1

pd_submit


model.load_state_dict(torch.load(os.path.join('/kaggle/working/', "model.pt"), weights_only=True))
with torch.no_grad():
    for index, image_id in pd_test.itertuples():
        img = Image.open(data_dir + f'/test_images/{image_id}')
        img = np.array(img)
        
        img = valid_augmentation(image=img)['image']

        inputs = img.unsqueeze(dim=0).cuda()
        
        logits = model(inputs)
        label = logits.cpu().argmax(dim=1).data.numpy()

        pd_submit.loc[index, 'label'] = label

pd_submit.to_csv(os.path.join('/kaggle/working/', 'submission.csv'), index=False)

