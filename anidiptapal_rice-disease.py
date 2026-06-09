!nvidia-smi


import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_properties(i)}")


import os, shutil, json, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pandas as pd


src_labelled = "/kaggle/input/rice-diseases-image-dataset/LabelledRice/Labelled"
dst_rice = "/kaggle/working/dataset/RiceDisease"
os.makedirs(dst_rice, exist_ok=True)

rename_map_rice = {
    "BrownSpot": "Rice_Brown_Spot",
    "Healthy": "Rice_Healthy",
    "Hispa": "Rice_Hispa",
    "LeafBlast": "Rice_Blast"
}

for folder, new_name in rename_map_rice.items():
    src_path = os.path.join(src_labelled, folder)
    dst_path = os.path.join(dst_rice, new_name)
    if os.path.exists(src_path):
        print(f"Copying {src_path} → {dst_path}")
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

print("\n✅ RiceDisease final folders:", os.listdir(dst_rice))


src_paddy = "/kaggle/input/paddy-disease-classification/train_images"
dst_rice_doc = "/kaggle/working/dataset/RiceDoc"
os.makedirs(dst_rice_doc, exist_ok=True)

rename_map_rice_paddy = {
    "bacterial_leaf_blight": "Rice_Bacterial_Leaf_Blight",
    "bacterial_leaf_streak": "Rice_Bacterial_Leaf_Streak",
    "bacterial_panicle_blight": "Rice_Bacterial_Panicle_Blight",
    "blast": "Rice_Blast",
    "brown_spot": "Rice_Brown_Spot",
    "dead_heart": "Rice_Dead_Heart",
    "downy_mildew": "Rice_Downy_Mildew",
    "hispa": "Rice_Hispa",
    "normal": "Rice_Healthy",
    "tungro": "Rice_Tungro"
}

for folder, new_name in rename_map_rice_paddy.items():
    src_path = os.path.join(src_paddy, folder)
    dst_path = os.path.join(dst_rice_doc, new_name)
    if os.path.exists(src_path):
        print(f"Copying {src_path} → {dst_path}")
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

print("\n✅ RiceDoc final folders:", os.listdir(dst_rice_doc))


def count_files_in_subfolders(path, dataset_name):
    counts = {}
    for root_dir, dirs, files in os.walk(path):
        if not dirs:
            class_name = os.path.basename(root_dir)
            counts[class_name] = len([f for f in files if not f.startswith('.')])
    return {dataset_name: counts}

datasets = ['RiceDisease', 'RiceDoc']

all_counts = {}
for dataset in datasets:
    dataset_path = os.path.join('/kaggle/working/dataset', dataset)
    all_counts.update(count_files_in_subfolders(dataset_path, dataset))

print(json.dumps(all_counts, indent=4))

flat_data = []
for dataset, classes in all_counts.items():
    for cls, count in classes.items():
        flat_data.append((dataset, cls, count))

df = pd.DataFrame(flat_data, columns=["Dataset", "Class", "Count"])
pivot_df = df.pivot(index="Class", columns="Dataset", values="Count").fillna(0)
ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10,6))

for i, total in enumerate(pivot_df.sum(axis=1)):
    ax.text(i, total + 5, str(int(total)), ha='center', va='bottom', fontsize=10)

plt.ylabel("Image Count")
plt.title("Image Counts per Class across Datasets")
plt.xticks(rotation=60, ha="right")
plt.legend(title="Dataset")
plt.tight_layout()
plt.show()


datasets = {
    "RiceDisease": "/kaggle/working/dataset/RiceDisease",
    "RiceDoc": "/kaggle/working/dataset/RiceDoc"
}

# Mapping disease names 
disease_mapping = {
    "Apple_Scab": "Apple_Scab",
    "Apple_Rust": "Apple_Rust",
    "Apple_Black_Rot": "Apple_Black_Rot",
    "Apple_Healthy": "Apple_Healthy",
    "Rice_Hispa": "Rice_Hispa",
    "Rice_Blast": "Rice_Blast",
    "Rice_Brown_Spot": "Rice_Brown_Spot",
    "Rice_Healthy": "Rice_Healthy",
    "Rice_Bacterial_Leaf_Blight": "Rice_Bacterial_Leaf_Blight",
    "Rice_Bacterial_Leaf_Streak": "Rice_Bacterial_Leaf_Streak",
    "Rice_Bacterial_Panicle_Blight": "Rice_Bacterial_Panicle_Blight",
    "Rice_Dead_Heart": "Rice_Dead_Heart",
    "Rice_Downy_Mildew": "Rice_Downy_Mildew",
    "Rice_Tungro": "Rice_Tungro"
}


final_dst = "/kaggle/working/final_dataset"
os.makedirs(final_dst, exist_ok=True)

# Merge all datasets
for dataset_name, dataset_path in datasets.items():
    for cls_folder in os.listdir(dataset_path):
        src_path = os.path.join(dataset_path, cls_folder)
        if os.path.isdir(src_path):
            unified_name = disease_mapping.get(cls_folder, cls_folder)
            dst_path = os.path.join(final_dst, unified_name)
            os.makedirs(dst_path, exist_ok=True)
            for file in os.listdir(src_path):
                src_file = os.path.join(src_path, file)
                dst_file = os.path.join(dst_path, file)
                # Avoid overwriting files with same name
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dst_file):
                        dst_file = os.path.join(dst_path, f"{base}_{counter}{ext}")
                        counter += 1
                shutil.copy2(src_file, dst_file)

# Summary
final_counts = {cls: len(os.listdir(os.path.join(final_dst, cls))) for cls in os.listdir(final_dst)}
import json
print(json.dumps(final_counts, indent=4))


def show_samples(dataset_path, size=20, rows=2, cols=10):
    classes = os.listdir(dataset_path)
    plt.figure(figsize=(cols*2, rows*2))
    per_class = size // len(classes)

    i = 1
    for cls in classes:
        img_list = os.listdir(os.path.join(dataset_path, cls))
        for img_name in random.sample(img_list, per_class):
            img = mpimg.imread(os.path.join(dataset_path, cls, img_name))
            plt.subplot(rows, cols, i)
            plt.imshow(img)
            plt.axis("off")
            plt.title(cls, fontsize=7)
            i += 1

    plt.tight_layout()
    plt.show()


show_samples("/kaggle/working/final_dataset", size=20, rows=4, cols=5)


!rm -rf /kaggle/working/dataset


def split_dataset(dataset_path, output_path, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, seed=50):
    random.seed(seed)
    for split in ["train","val","test"]:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)
    for cls in os.listdir(dataset_path):
        imgs = os.listdir(os.path.join(dataset_path, cls))
        random.shuffle(imgs)
        n = len(imgs)
        n_train, n_val = int(train_ratio*n), int(val_ratio*n)
        splits = [imgs[:n_train], imgs[n_train:n_train+n_val], imgs[n_train+n_val:]]
        for split, split_imgs in zip(["train","val","test"], splits):
            d = os.path.join(output_path, split, cls); os.makedirs(d, exist_ok=True)
            for img in split_imgs: shutil.copy(os.path.join(dataset_path, cls, img), os.path.join(d, img))

split_dataset("/kaggle/working/final_dataset","/kaggle/working/rice_dataset")


base="/kaggle/working/rice_dataset"
final_counts={s:{c:len(os.listdir(os.path.join(base,s,c))) for c in os.listdir(os.path.join(base,s))} for s in ["train","val","test"]}
print(json.dumps(final_counts,indent=4))


!rm -rf /kaggle/working/final_dataset


# dataset_path = "/kaggle/working/rice_dataset"
# shutil.make_archive("/kaggle/working/rice_dataset", 'zip', dataset_path)
# print("Dataset zipped as rice_dataset.zip")


import torch
import torch.nn as nn
import math


__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y.sigmoid(), 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = nn.Dropout(0.2)

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_channel = output_channel

        output_channel = 1280
        self.classifier = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghost_net(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__=='__main__':
    model = ghost_net(num_classes=10)
    model.eval()
    # print(model)
    input_tensor = torch.randn(32, 3, 224, 224)
    y = model(input_tensor)
    print(f"Output shape: {y.shape}")


import torch as t
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


def channel_shuffle(x, groups=2):
  bat_size, channels, w, h = x.shape
  group_c = channels // groups
  x = x.view(bat_size, groups, group_c, w, h)
  x = t.transpose(x, 1, 2).contiguous()
  x = x.view(bat_size, -1, w, h)
  return x

# used in the block
def conv_1x1_bn(in_c, out_c, stride=1):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU(True)
  )

def conv_bn(in_c, out_c, stride=2):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU(True)
  )


class ShuffleBlock(nn.Module):
  def __init__(self, in_c, out_c, downsample=False):
    super(ShuffleBlock, self).__init__()
    self.downsample = downsample
    half_c = out_c // 2
    if downsample:
      self.branch1 = nn.Sequential(
          # 3*3 dw conv, stride = 2
          nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
          nn.BatchNorm2d(in_c),
          # 1*1 pw conv
          nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )

      self.branch2 = nn.Sequential(
          # 1*1 pw conv
          nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True),
          # 3*3 dw conv, stride = 2
          nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
          nn.BatchNorm2d(half_c),
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )
    else:
      # in_c = out_c
      assert in_c == out_c

      self.branch2 = nn.Sequential(
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True),
          # 3*3 dw conv, stride = 1
          nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
          nn.BatchNorm2d(half_c),
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )


  def forward(self, x):
    out = None
    if self.downsample:
      # if it is downsampling, we don't need to do channel split
      out = t.cat((self.branch1(x), self.branch2(x)), 1)
    else:
      # channel split
      channels = x.shape[1]
      c = channels // 2
      x1 = x[:, :c, :, :]
      x2 = x[:, c:, :, :]
      out = t.cat((x1, self.branch2(x2)), 1)
    return channel_shuffle(out, 2)


class ShuffleNet2(nn.Module):
  def __init__(self, num_classes=2, input_size=224, net_type=1):
    super(ShuffleNet2, self).__init__()
    assert input_size % 32 == 0 # 因为一共会下采样32倍


    self.stage_repeat_num = [4, 8, 4]
    if net_type == 0.5:
      self.out_channels = [3, 24, 48, 96, 192, 1024]
    elif net_type == 1:
      self.out_channels = [3, 24, 116, 232, 464, 1024]
    elif net_type == 1.5:
      self.out_channels = [3, 24, 176, 352, 704, 1024]
    elif net_type == 2:
      self.out_channels = [3, 24, 244, 488, 976, 2948]
    else:
      print("the type is error, you should choose 0.5, 1, 1.5 or 2")

    # let's start building layers
    self.conv1 = nn.Conv2d(3, self.out_channels[1], 3, 2, 1)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    in_c = self.out_channels[1]

    self.stages = []
    for stage_idx in range(len(self.stage_repeat_num)):
      out_c = self.out_channels[2+stage_idx]
      repeat_num = self.stage_repeat_num[stage_idx]
      for i in range(repeat_num):
        if i == 0:
          self.stages.append(ShuffleBlock(in_c, out_c, downsample=True))
        else:
          self.stages.append(ShuffleBlock(in_c, in_c, downsample=False))
        in_c = out_c
    self.stages = nn.Sequential(*self.stages)

    in_c = self.out_channels[-2]
    out_c = self.out_channels[-1]
    self.conv5 = conv_1x1_bn(in_c, out_c, 1)
    self.g_avg_pool = nn.AvgPool2d(kernel_size=(int)(input_size/32)) # 如果输入的是224，则此处为7

    # fc layer
    self.fc = nn.Linear(out_c, num_classes)


  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.stages(x)
    x = self.conv5(x)
    x = self.g_avg_pool(x)
    x = x.view(-1, self.out_channels[-1])
    x = self.fc(x)
    return x

def shufflenet_v2(net_type=1.0, num_classes=10, input_size=224):
    """
    Constructs a ShuffleNetV2 model
    net_type: choose from {0.5, 1.0, 1.5, 2.0}
    """
    return ShuffleNet2(num_classes=num_classes, input_size=input_size, net_type=net_type)


if __name__ == "__main__":
    model = shufflenet_v2(net_type=1.0, num_classes=10, input_size=224)
    model.eval()
    # print(model)
    x = t.randn(32, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")


!pip install thop


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, 
                           confusion_matrix, classification_report, roc_curve, auc, 
                           roc_auc_score, average_precision_score)
from thop import profile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
import os
from collections import defaultdict
import copy
from PIL import Image, ImageFilter
import random


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 10
EARLY_STOPPING_PATIENCE = 5
#32 162s
print(f"Using device: {DEVICE}\nBatch size: {BATCH_SIZE}\nNumber of epochs: {NUM_EPOCHS}\nLearning rate: {LEARNING_RATE}\nNumber of classes: {NUM_CLASSES}\nEarly stopping patience: {EARLY_STOPPING_PATIENCE}\n")


class RiceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_path, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Custom transform augmentations
class AddGaussianNoise:
    def __init__(self, mean=0., std=1., noise_factor=0.05):
        self.std = std
        self.mean = mean
        self.noise_factor = noise_factor
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise * self.noise_factor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomBrightnessDarken:
    def __init__(self, factor=0.05):
        self.factor = factor
    
    def __call__(self, img):
        # Randomly brighten or darken
        brightness_factor = 1.0 + random.uniform(-self.factor, self.factor)
        return transforms.functional.adjust_brightness(img, brightness_factor)

class RandomBlur:
    def __init__(self, radius_range=(0.5, 1.5)):
        self.radius_range = radius_range
    
    def __call__(self, img):
        if random.random() > 0.5:  # Apply blur 50%
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

# Augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    RandomBlur(radius_range=(0.5, 2.0)),
    RandomBrightnessDarken(factor=0.05),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=1., noise_factor=0.05),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load 
base_path = "/kaggle/working/rice_dataset"
train_dataset = RiceDataset(os.path.join(base_path, "train"), transform=train_transform)
val_dataset = RiceDataset(os.path.join(base_path, "val"), transform=val_test_transform)
test_dataset = RiceDataset(os.path.join(base_path, "test"), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# Rice disease class names
class_names = [
    'Rice_Bacterial_Leaf_Blight',
    'Rice_Bacterial_Leaf_Streak', 
    'Rice_Bacterial_Panicle_Blight',
    'Rice_Blast',
    'Rice_Brown_Spot',
    'Rice_Dead_Heart',
    'Rice_Downy_Mildew',
    'Rice_Healthy',
    'Rice_Hispa',
    'Rice_Tungro'
]

print(f"Classes: {class_names}")


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model):
        self.best_weights = copy.deepcopy(model.state_dict())

def get_model_stats(model, input_shape=(1, 3, 224, 224)):
    dummy_input = torch.randn(input_shape).to(DEVICE)
    flops, params = profile(model, inputs=(dummy_input,))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    stats = {
        'FLOPs': flops,
        'GFLOPs': flops / 1e9,
        'MFLOPs': flops / 1e6,
        'Total_Params': total_params,
        'Trainable_Params': trainable_params,
        'Non_Trainable_Params': non_trainable_params,
        'Model_Size_MB': total_params * 4 / (1024 * 1024)
    }
    return stats

def calculate_iou_per_class(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    iou_scores = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        iou_scores.append(iou)
    return iou_scores, np.mean(iou_scores)

def calculate_map_metrics(y_true, y_prob, num_classes):
    y_true_onehot = np.eye(num_classes)[y_true]
    
    ap_scores = []
    for i in range(num_classes):
        ap = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
        ap_scores.append(ap)
    
    # Calculate mAP50-95
    thresholds = np.arange(0.5, 0.95, 0.05)
    map_scores = []
    
    for threshold in thresholds:
        precision_scores = []
        for i in range(num_classes):
            pred_binary = (y_prob[:, i] >= threshold).astype(int)
            if pred_binary.sum() == 0 and y_true_onehot[:, i].sum() == 0:
                precision_scores.append(1.0)
            else:
                prec = precision_score(y_true_onehot[:, i], pred_binary, zero_division=0)
                precision_scores.append(prec)
        map_scores.append(np.mean(precision_scores))
    
    map50_95 = np.mean(map_scores)
    
    # Calculate mAP90
    precision_90 = []
    for i in range(num_classes):
        pred_binary_90 = (y_prob[:, i] >= 0.9).astype(int)
        if pred_binary_90.sum() == 0 and y_true_onehot[:, i].sum() == 0:
            precision_90.append(1.0)
        else:
            prec_90 = precision_score(y_true_onehot[:, i], pred_binary_90, zero_division=0)
            precision_90.append(prec_90)
    
    map90 = np.mean(precision_90)
    
    return {
        'mAP': np.mean(ap_scores),
        'mAP50-95': map50_95,
        'mAP90': map90,
        'AP_per_class': ap_scores
    }


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = running_loss / len(data_loader)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    iou_scores, mean_iou = calculate_iou_per_class(all_labels, all_preds, NUM_CLASSES)
    map_metrics = calculate_map_metrics(all_labels, all_probs, NUM_CLASSES)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'mean_iou': mean_iou,
        'iou_scores': iou_scores,
        'mAP': map_metrics['mAP'],
        'mAP50-95': map_metrics['mAP50-95'],
        'mAP90': map_metrics['mAP90']
    }, all_labels, all_preds, all_probs

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        probs = torch.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())
    
    accuracy = correct / total
    avg_loss = running_loss / len(train_loader)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    iou_scores, mean_iou = calculate_iou_per_class(all_labels, all_preds, NUM_CLASSES)
    map_metrics = calculate_map_metrics(all_labels, all_probs, NUM_CLASSES)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'mean_iou': mean_iou,
        'iou_scores': iou_scores,
        'mAP': map_metrics['mAP'],
        'mAP50-95': map_metrics['mAP50-95'],
        'mAP90': map_metrics['mAP90']
    }

def train_model(model, model_name, train_loader, val_loader, num_epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    
    history = defaultdict(list)
    results_df = []
    
    # Track best metrics
    best_f1 = 0.0
    best_epoch = 0
    best_model_state = None
    best_metrics = {}
    
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['f1'])
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Track best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics.copy()
        
        for key in train_metrics.keys():
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        epoch_data = {
            'model': model_name,
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'train_roc_auc': train_metrics['roc_auc'],
            'train_mean_iou': train_metrics['mean_iou'],
            'train_mAP': train_metrics['mAP'],
            'train_mAP50-95': train_metrics['mAP50-95'],
            'train_mAP90': train_metrics['mAP90'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_mean_iou': val_metrics['mean_iou'],
            'val_mAP': val_metrics['mAP'],
            'val_mAP50-95': val_metrics['mAP50-95'],
            'val_mAP90': val_metrics['mAP90']
        }
        results_df.append(epoch_data)
        
        # Early stopping check
        if early_stopping(val_metrics['f1'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Save models
    os.makedirs('saved_models', exist_ok=True)
    
    # Save final model
    torch.save(model.state_dict(), f'saved_models/{model_name}_final_epoch.pth')
    print(f"Final model saved: saved_models/{model_name}_final_epoch.pth")
    
    # Save best model
    torch.save(best_model_state, f'saved_models/{model_name}_best_f1.pth')
    print(f"Best model saved: saved_models/{model_name}_best_f1.pth")
    
    return history, results_df, training_time, best_epoch, best_metrics, best_model_state

def test_best_model(model, best_model_state, model_name, test_loader, device):
    # Load best model weights
    model.load_state_dict(best_model_state)
    
    criterion = nn.CrossEntropyLoss()
    test_metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\n{model_name} - Best Model Test Results:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"Test mAP: {test_metrics['mAP']:.4f}")
    print(f"Test mAP50-95: {test_metrics['mAP50-95']:.4f}")
    print(f"Test mAP90: {test_metrics['mAP90']:.4f}")
    
    return test_metrics, y_true, y_pred, y_prob


def plot_training_history(history, model_name):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{model_name} Training History', fontsize=16)
    
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mean_iou', 'mAP']
    
    for i, metric in enumerate(metrics):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        
        epochs = range(1, len(history[f'train_{metric}']) + 1)
        ax.plot(epochs, history[f'train_{metric}'], 'b-', label=f'Train {metric}', marker='o')
        ax.plot(epochs, history[f'val_{metric}'], 'r-', label=f'Val {metric}', marker='s')
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[name.replace('Rice_', '') for name in class_names], 
                yticklabels=[name.replace('Rice_', '') for name in class_names])
    plt.title(f'{model_name} - Best Model Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_roc_curves(y_true, y_prob, class_names, model_name):
    y_true_onehot = np.eye(len(class_names))[y_true]
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        try:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            display_name = class_name.replace('Rice_', '')
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{display_name} (AUC = {roc_auc:.2f})')
        except:
            pass
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Best Model ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_sample_predictions(model, test_loader, class_names, model_name, device, num_samples=10):
    model.eval()
    
    images = []
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            for i in range(min(len(data), num_samples - len(images))):
                images.append(data[i].cpu().detach())
                true_labels.append(target[i].cpu().detach().item())
                pred_labels.append(predicted[i].cpu().detach().item())
            
            if len(images) >= num_samples:
                break
    
    # ImageNet normalization values
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'{model_name} - Best Model Sample Predictions', fontsize=16)
    
    for i in range(num_samples):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)
        
        ax.imshow(img)
        true_class = class_names[true_labels[i]].replace('Rice_', '')
        pred_class = class_names[pred_labels[i]].replace('Rice_', '')
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        ax.set_title(f'True: {true_class}\nPred: {pred_class}', color=color, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def print_model_stats(model_stats, model_name):
    print(f"\n{model_name} Model Statistics:")
    print(f"FLOPs: {model_stats['FLOPs']:,.0f}")
    print(f"GFLOPs: {model_stats['GFLOPs']:.3f}")
    print(f"MFLOPs: {model_stats['MFLOPs']:.2f}")
    print(f"Total Parameters: {model_stats['Total_Params']:,}")
    print(f"Trainable Parameters: {model_stats['Trainable_Params']:,}")
    print(f"Non-trainable Parameters: {model_stats['Non_Trainable_Params']:,}")
    print(f"Model Size: {model_stats['Model_Size_MB']:.2f} MB")

def print_best_metrics_summary(model_name, best_epoch, best_metrics):
    print(f"\n{model_name} - Best Model Summary (Epoch {best_epoch}):")
    print(f"Best Validation Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best Validation F1-Score: {best_metrics['f1']:.4f}")
    print(f"Best Validation mAP: {best_metrics['mAP']:.4f}")
    print(f"Best Validation mAP50-95: {best_metrics['mAP50-95']:.4f}")
    print(f"Best Validation mAP90: {best_metrics['mAP90']:.4f}")


all_results = []
model_summaries = []

print("=" * 60)
print("SHUFFLENETV2 EVALUATION")
print("=" * 60)

# Initialize ShuffleNetV2 model
shufflenet_model = shufflenet_v2(net_type=1.0, num_classes=NUM_CLASSES, input_size=224).to(DEVICE)
shufflenet_stats = get_model_stats(shufflenet_model)
print_model_stats(shufflenet_stats, "ShuffleNetV2")

# Train ShuffleNetV2
shufflenet_history, shufflenet_df, shuffle_time, shuffle_best_epoch, shuffle_best_metrics, shuffle_best_state = train_model(
    shufflenet_model, "ShuffleNetV2", train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE
)

all_results.extend(shufflenet_df)
print(f"\nShuffleNetV2 Training Time: {shuffle_time:.2f} seconds")
print_best_metrics_summary("ShuffleNetV2", shuffle_best_epoch, shuffle_best_metrics)

# Test best ShuffleNetV2 model
shuffle_test_metrics, shuffle_y_true, shuffle_y_pred, shuffle_y_prob = test_best_model(
    shufflenet_model, shuffle_best_state, "ShuffleNetV2", test_loader, DEVICE
)

# Plot results for ShuffleNetV2
plot_training_history(shufflenet_history, "ShuffleNetV2")
display_sample_predictions(shufflenet_model, test_loader, class_names, "ShuffleNetV2", DEVICE)
plot_confusion_matrix(shuffle_y_true, shuffle_y_pred, class_names, "ShuffleNetV2")
plot_roc_curves(shuffle_y_true, shuffle_y_prob, class_names, "ShuffleNetV2")

results_df = pd.DataFrame(all_results)
results_df.to_csv('rice_model_comparison_results.csv', index=False)

results_df = pd.DataFrame(shuffle_test_metrics)
results_df.to_csv('shuffle_model_results.csv', index=False)


print("\n" + "=" * 60)
print("GHOSTNET EVALUATION")
print("=" * 60)

# Initialize GhostNet model
ghostnet_model = ghost_net(num_classes=NUM_CLASSES).to(DEVICE)
ghostnet_stats = get_model_stats(ghostnet_model)
print_model_stats(ghostnet_stats, "GhostNet")

# Train GhostNet
ghostnet_history, ghostnet_df, ghost_time, ghost_best_epoch, ghost_best_metrics, ghost_best_state = train_model(
    ghostnet_model, "GhostNet", train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE
)

all_results.extend(ghostnet_df)
print(f"\nGhostNet Training Time: {ghost_time:.2f} seconds")
print_best_metrics_summary("GhostNet", ghost_best_epoch, ghost_best_metrics)

# Test best GhostNet model
ghost_test_metrics, ghost_y_true, ghost_y_pred, ghost_y_prob = test_best_model(
    ghostnet_model, ghost_best_state, "GhostNet", test_loader, DEVICE
)

# Plot results for GhostNet
plot_training_history(ghostnet_history, "GhostNet")
display_sample_predictions(ghostnet_model, test_loader, class_names, "GhostNet", DEVICE)
plot_confusion_matrix(ghost_y_true, ghost_y_pred, class_names, "GhostNet")
plot_roc_curves(ghost_y_true, ghost_y_prob, class_names, "GhostNet")

results_df = pd.DataFrame(all_results)
results_df.to_csv('rice_model_comparison_results_2.csv', index=False)

results_df = pd.DataFrame(ghost_test_metrics)
results_df.to_csv('ghost_model_results.csv', index=False)


# Save all results
results_df = pd.DataFrame(all_results)
results_df.to_csv('rice_model_comparison_results.csv', index=False)
print("\nDetailed results saved to 'rice_model_comparison_results.csv'")

# Create comparison
comparison_data = {
    'Model': ['GhostNet', 'ShuffleNetV2'],
    'Training_Time_Sec': [ghost_time, shuffle_time],
    'FLOPs': [ghostnet_stats['FLOPs'], shufflenet_stats['FLOPs']],
    'GFLOPs': [ghostnet_stats['GFLOPs'], shufflenet_stats['GFLOPs']],
    'Parameters_M': [ghostnet_stats['Total_Params']/1e6, shufflenet_stats['Total_Params']/1e6],
    'Model_Size_MB': [ghostnet_stats['Model_Size_MB'], shufflenet_stats['Model_Size_MB']],
    
    # Best epoch metrics
    'Best_Epoch': [ghost_best_epoch, shuffle_best_epoch],
    'Best_Val_Accuracy': [ghost_best_metrics['accuracy'], shuffle_best_metrics['accuracy']],
    'Best_Val_F1': [ghost_best_metrics['f1'], shuffle_best_metrics['f1']],
    'Best_Val_mAP': [ghost_best_metrics['mAP'], shuffle_best_metrics['mAP']],
    'Best_Val_mAP50-95': [ghost_best_metrics['mAP50-95'], shuffle_best_metrics['mAP50-95']],
    'Best_Val_mAP90': [ghost_best_metrics['mAP90'], shuffle_best_metrics['mAP90']],
    
    # Final epoch metrics
    'Final_Val_Accuracy': [ghostnet_history['val_accuracy'][-1], shufflenet_history['val_accuracy'][-1]],
    'Final_Val_F1': [ghostnet_history['val_f1'][-1], shufflenet_history['val_f1'][-1]],
    'Final_Val_mAP': [ghostnet_history['val_mAP'][-1], shufflenet_history['val_mAP'][-1]],
    'Final_Val_mAP50-95': [ghostnet_history['val_mAP50-95'][-1], shufflenet_history['val_mAP50-95'][-1]],
    'Final_Val_mAP90': [ghostnet_history['val_mAP90'][-1], shufflenet_history['val_mAP90'][-1]],
    
    # Test metrics (best model)
    'Test_Accuracy': [ghost_test_metrics['accuracy'], shuffle_test_metrics['accuracy']],
    'Test_Precision': [ghost_test_metrics['precision'], shuffle_test_metrics['precision']],
    'Test_Recall': [ghost_test_metrics['recall'], shuffle_test_metrics['recall']],
    'Test_F1': [ghost_test_metrics['f1'], shuffle_test_metrics['f1']],
    'Test_mAP': [ghost_test_metrics['mAP'], shuffle_test_metrics['mAP']],
    'Test_mAP50-95': [ghost_test_metrics['mAP50-95'], shuffle_test_metrics['mAP50-95']],
    'Test_mAP90': [ghost_test_metrics['mAP90'], shuffle_test_metrics['mAP90']]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('rice_model_comparison_summary.csv', index=False)

print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON saved to 'rice_model_comparison_summary.csv'")
print("="*80)

# Print detailed best metrics for each model
print("\n" + "="*80)
print("BEST MODEL PERFORMANCE ON RICE DISEASE CLASSIFICATION")
print("="*80)

print(f"\nGHOSTNET - Best Performance (Epoch {ghost_best_epoch}):")
print(f"  Validation Metrics:")
print(f"    - Accuracy: {ghost_best_metrics['accuracy']:.4f}")
print(f"    - F1-Score: {ghost_best_metrics['f1']:.4f}")
print(f"    - mAP: {ghost_best_metrics['mAP']:.4f}")
print(f"    - mAP50-95: {ghost_best_metrics['mAP50-95']:.4f}")
print(f"    - mAP90: {ghost_best_metrics['mAP90']:.4f}")
print(f"  Test Metrics (Best Model):")
print(f"    - Accuracy: {ghost_test_metrics['accuracy']:.4f}")
print(f"    - Precision: {ghost_test_metrics['precision']:.4f}")
print(f"    - Recall: {ghost_test_metrics['recall']:.4f}")
print(f"    - F1-Score: {ghost_test_metrics['f1']:.4f}")
print(f"    - mAP: {ghost_test_metrics['mAP']:.4f}")
print(f"    - mAP50-95: {ghost_test_metrics['mAP50-95']:.4f}")
print(f"    - mAP90: {ghost_test_metrics['mAP90']:.4f}")

print(f"\nSHUFFLENETV2 - Best Performance (Epoch {shuffle_best_epoch}):")
print(f"  Validation Metrics:")
print(f"    - Accuracy: {shuffle_best_metrics['accuracy']:.4f}")
print(f"    - F1-Score: {shuffle_best_metrics['f1']:.4f}")
print(f"    - mAP: {shuffle_best_metrics['mAP']:.4f}")
print(f"    - mAP50-95: {shuffle_best_metrics['mAP50-95']:.4f}")
print(f"    - mAP90: {shuffle_best_metrics['mAP90']:.4f}")
print(f"  Test Metrics (Best Model):")
print(f"    - Accuracy: {shuffle_test_metrics['accuracy']:.4f}")
print(f"    - Precision: {shuffle_test_metrics['precision']:.4f}")
print(f"    - Recall: {shuffle_test_metrics['recall']:.4f}")
print(f"    - F1-Score: {shuffle_test_metrics['f1']:.4f}")
print(f"    - mAP: {shuffle_test_metrics['mAP']:.4f}")
print(f"    - mAP50-95: {shuffle_test_metrics['mAP50-95']:.4f}")
print(f"    - mAP90: {shuffle_test_metrics['mAP90']:.4f}")

# Save per-class performance analysis
print("\n" + "="*80)
print("PER-CLASS RICE DISEASE PERFORMANCE ANALYSIS")
print("="*80)

# GhostNet per-class analysis
ghost_class_report = classification_report(ghost_y_true, ghost_y_pred, 
                                         target_names=class_names, 
                                         output_dict=True)
ghost_class_df = pd.DataFrame(ghost_class_report).transpose()
ghost_class_df['model'] = 'GhostNet'

# ShuffleNetV2 per-class analysis
shuffle_class_report = classification_report(shuffle_y_true, shuffle_y_pred, 
                                           target_names=class_names, 
                                           output_dict=True)
shuffle_class_df = pd.DataFrame(shuffle_class_report).transpose()
shuffle_class_df['model'] = 'ShuffleNetV2'

# Combine and save per-class results
combined_class_df = pd.concat([ghost_class_df, shuffle_class_df])
combined_class_df.to_csv('rice_per_class_performance.csv')

print("Per-class performance saved to 'rice_per_class_performance.csv'")

# Model efficiency comparison
print("\n" + "="*80)
print("RICE DISEASE MODEL EFFICIENCY COMPARISON")
print("="*80)

efficiency_metrics = {
    'Model': ['GhostNet', 'ShuffleNetV2'],
    'Accuracy_per_GFLOP': [ghost_test_metrics['accuracy'] / ghostnet_stats['GFLOPs'],
                          shuffle_test_metrics['accuracy'] / shufflenet_stats['GFLOPs']],
    'F1_per_GFLOP': [ghost_test_metrics['f1'] / ghostnet_stats['GFLOPs'],
                    shuffle_test_metrics['f1'] / shufflenet_stats['GFLOPs']],
    'Accuracy_per_MB': [ghost_test_metrics['accuracy'] / ghostnet_stats['Model_Size_MB'],
                       shuffle_test_metrics['accuracy'] / shufflenet_stats['Model_Size_MB']],
    'F1_per_MB': [ghost_test_metrics['f1'] / ghostnet_stats['Model_Size_MB'],
                 shuffle_test_metrics['f1'] / shufflenet_stats['Model_Size_MB']],
    'Training_Time_per_Epoch': [ghost_time / len(ghostnet_history['val_accuracy']),
                               shuffle_time / len(shufflenet_history['val_accuracy'])]
}

efficiency_df = pd.DataFrame(efficiency_metrics)
efficiency_df.to_csv('rice_model_efficiency_comparison.csv', index=False)

print("\nRice Disease Classification Efficiency Comparison saved to 'rice_model_efficiency_comparison.csv'")

# Summary of dataset-specific insights
print("\n" + "="*80)
print("RICE DISEASE DATASET INSIGHTS")
print("="*80)
print(f"Dataset: Rice Disease Classification with {NUM_CLASSES} classes")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")  
print(f"Test samples: {len(test_dataset)}")
print(f"Image size: 224x224")
print(f"Data augmentation: Horizontal/Vertical flip, Rotation, Brightness/Contrast/Blur/Noise")

# Identify best performing model
best_model = "GhostNet" if ghost_test_metrics['f1'] > shuffle_test_metrics['f1'] else "ShuffleNetV2"
best_f1 = max(ghost_test_metrics['f1'], shuffle_test_metrics['f1'])

print(f"\nBest performing model: {best_model} (Test F1: {best_f1:.4f})")


# !rm -rf /kaggle/working/rice_dataset


import os, zipfile

base_dir = "/kaggle/working/"
zip_path = "/kaggle/working/results.zip"

# Collect all .csv and .pth files
files_to_zip = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".csv") or file.endswith(".pth"):
            files_to_zip.append(os.path.join(root, file))

# Zip 
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file in files_to_zip:
        arcname = os.path.relpath(file, base_dir)
        zipf.write(file, arcname)

print(f"Zipped {len(files_to_zip)} files into {zip_path}")




