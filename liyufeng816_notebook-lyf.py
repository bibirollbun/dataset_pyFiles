# # What this is doing? please refer to my above linked kernel
# !pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null
# package_path = '../input/unetmodelscript'
# import sys 
# sys.path.append(package_path)


! ls ../input/ 


import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.pytorch import ToTensorV2
import torch.utils.data as data
# import segmentation_models_pytorch as smp





#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        # df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


# !ls ../input/unetstartermodelfile/


sample_submission_path = '/kaggle/input/new-sample/sample_submission.csv'
test_data_folder = "/kaggle/input/test-image/test_images/"


# initialize test dataloader
best_threshold = 0.5
num_workers = 2
batch_size = 1
print('best_threshold', best_threshold)
min_size = 3500
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2),
#             nn.BatchNorm2d(middle_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x):
#         return self.block(x)

# class CustomUNet(nn.Module):
#     def __init__(self, num_classes=4, encoder_weights='None'):
#         super(CustomUNet, self).__init__()
        
#         resnet = models.resnet18(pretrained=(encoder_weights == 'None'))
        
#         self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [B, 64, H/2, W/2]
#         self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)          # [B, 64, H/4, W/4]
#         self.encoder2 = resnet.layer2                                         # [B, 128, H/8, W/8]
#         self.encoder3 = resnet.layer3                                         # [B, 256, H/16, W/16]
#         self.encoder4 = resnet.layer4                                         # [B, 512, H/32, W/32]
        
#         self.decoder4 = DecoderBlock(512, 256, 256)
#         self.decoder3 = DecoderBlock(256, 128, 128)
#         self.decoder2 = DecoderBlock(128, 64, 64)
#         self.decoder1 = DecoderBlock(64, 64, 64)
        
#         self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
#     def forward(self, x):
#         e0 = self.encoder0(x)    # [B, 64, H/2, W/2]
#         e1 = self.encoder1(e0)   # [B, 64, H/4, W/4]
#         e2 = self.encoder2(e1)   # [B, 128, H/8, W/8]
#         e3 = self.encoder3(e2)   # [B, 256, H/16, W/16]
#         e4 = self.encoder4(e3)   # [B, 512, H/32, W/32]
        
#         d4 = self.decoder4(e4)   # [B, 256, H/16, W/16]
#         d4 = d4 + e3              # [B, 256, H/16, W/16]
        
#         d3 = self.decoder3(d4)   # [B, 128, H/8, W/8]
#         d3 = d3 + e2              # [B, 128, H/8, W/8]
        
#         d2 = self.decoder2(d3)   # [B, 64, H/4, W/4]
#         d2 = d2 + e1              # [B, 64, H/4, W/4]
        
#         d1 = self.decoder1(d2)   # [B, 64, H/2, W/2]
#         d1 = d1 + e0              # [B, 64, H/2, W/2]
        
#         out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, H, W]
#         out = self.final_conv(out)  # [B, num_classes, H, W]
        
#         return out

# # 实例化模型
# model = CustomUNet(num_classes=4, encoder_weights='imagenet')

# # 移动模型到设备
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# # 打印模型结构
# print(model)


from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]
#         print(f'the input mix x is {x.shape}')
        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            
            x = get_overlap_patches(x)
#             print(f'the x after get_overlap_patches is {x.shape}')
            
            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

            x = overlap_embed(x)
#             print(f'the x after overlap_embed is {x.shape}')
            
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x
#             print(f'the mit x is {x.shape}')
            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class Segformer(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** (i+2)  )
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)
#         for k in range(len(layer_outputs)):
#             print(f'the k shape is {layer_outputs[k].shape}')
                       
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        
#         for k in range(len(fused)):
#             print(f'the k shape fused is {fused[k].shape}')
        
        fused = torch.cat(fused, dim = 1)
        
        
        return self.to_segmentation(fused)

# 实例化模型
model = Segformer()

# 移动模型到设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 打印模型结构
print(model)


# import segmentation_models_pytorch as smp
# Initialize mode and load trained weights
#ckpt_path = "../input/unetstartermodelfile/model.pth"
ckpt_path = "/kaggle/input/model_sgformer/pytorch/default/1/model_segformer_new.pth"
device = torch.device("cuda")
#model = Unet("resnet18", encoder_weights=None, classes=4, activation=None)
# model = smp.Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
# model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])


# start prediction
predictions = []
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    batch_preds = torch.sigmoid(model(images.to(device)))
    batch_preds = batch_preds.detach().cpu().numpy()
    for fname, preds in zip(fnames, batch_preds):
        # print(f'the shape of pred is {preds.shape}')
        for cls, pred in enumerate(preds):
            # print(f'cls is {cls}')
            pred, num = post_process(pred, best_threshold, min_size)
            rle = mask2rle(pred)
            name = fname + f"_{cls+1}"
            # predictions.append([fname, rle, cls])
            predictions.append([name, rle])
            

# save predictions to submission.csv
df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])




# print(f'the shape of prediction is {predictions.shape}')
# df = df[~df.applymap(lambda x: x == '').any(axis=1)]
# df = df[~df.applymap(lambda x: x == '').any(axis=1)]

# df = pd.DataFrame(predictions,columns=['ImageId','EncodedPixels','ClassId'])
df = df.fillna('')
# df = df[~df.applymap(lambda x: x == None).any(axis=1)]
# df = df[~df.applymap(lambda x: x == '').any(axis=1)]
df.to_csv("submission.csv", index=False)


df.head()


# print(df.shape[0])
# # num_unique_images = df['ImageId_ClassId'].nunique()
# # print(num_unique_images)

# df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
# num_unique_images = df['ImageId'].nunique()
# print(num_unique_images)

