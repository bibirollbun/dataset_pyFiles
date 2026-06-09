# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import os
# import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time


def readfile(path, label):
    # 输入参数label为boolean变量，代表是否返回 y 值
    image_dir = sorted(os.listdir(path))#os.listdir返回文件夹包含的文件名字的列表
    x = np.zeros((len(image_dir), 224, 224, 3), dtype=np.uint8)#形状:文件个数*128*128*3
    y = np.zeros((len(image_dir)), dtype=np.uint8)#形状:文件个数
    for i, file in enumerate(image_dir):#遍历文件列表中文件名
        img = cv2.imread(os.path.join(path, file))#cv2读入原图片
        x[i, :, :] = cv2.resize(img,(224,224))#对图片进行缩放, 存储到x的第i个元素中
        if label:
          y[i] = int(file.split("_")[0])#取出文件名中的类别信息
    if label:
      return x, y
    else:
      return x


workspace_dir = "/kaggle/input/ml2020spring-hw3/food-11"
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)#训练集
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)#测试集
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)#预测的地方
print("Size of Testing data = {}".format(len(test_x)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 随机将图片水平翻转
    transforms.RandomRotation(15), # 随机选择图片
    transforms.ToTensor(), # 图片转化为张量Tensor，並把數值 normalize 到 [0,1] (data normalization)
])

# 测试时不需做数据增强
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    #初始化中把所有传入内容赋给属性
    def __init__(self, x, y=None, transform=None):#如果没有传入y, transform, 则默认值为0
        self.x = x
        # label类型应为 LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    # 返回dataset的大小
    def __len__(self):
        return len(self.x)
    # 用[ ]取值時，dataset如何返回. 返回前先对x进行转换
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
#             X = np.array(X, dtype=np.float32)  # PILImage->numpy 输出(h,w,c)
#             X = np.transpose(X, (2, 0, 1))  # np下维度转换使用transpose，类似矩阵转置
#             X = torch.from_numpy(X)  # numpy->tensor, 张量和ndarray共享同一内存, 不能调整大小
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


batch_size = 4
train_set = ImgDataset(train_x, train_y, train_transform)#训练集
val_set = ImgDataset(val_x, val_y, test_transform)#测试集
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)      

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=768, num_layers=12, hidden_dim=3072):
        super(VisionTransformer, self).__init__()

        # 图像尺寸和 patch 设置
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding (将图像分成patch并展平，转换为嵌入向量)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 可学习的类别 token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # (1, 1, embed_dim)

        # Transformer 的位置编码 (Positional Encoding)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  # +1 是为了加上 CLS token
        
        # Transformer 编码器部分 (单层自注意力机制)
        self.encoder = nn.ModuleList([self._build_transformer_block(embed_dim, hidden_dim) for _ in range(num_layers)])
        
        # 分类头 (将 Transformer 输出映射到类数)
        self.fc = nn.Linear(embed_dim, num_classes)

    def _build_transformer_block(self, embed_dim, hidden_dim):
        """构建单层自注意力模块"""
        # 定义 Q、K、V 的线性变换
        self_attention = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim),  # 用于 Q
            nn.Linear(embed_dim, embed_dim),  # 用于 K
            nn.Linear(embed_dim, embed_dim)   # 用于 V
        ])

        # 前馈网络
        feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # 层归一化
        layer_norm = nn.LayerNorm(embed_dim)
        
        return nn.ModuleList([self_attention, feed_forward, layer_norm])

    def forward(self, x):
        # 输入图像形状: (batch_size, 3, img_size, img_size)
        batch_size = x.size(0)

        # Patch Embedding (将输入图像分成patch并嵌入)
        x = self.patch_embed(x)  # 形状: (batch_size, embed_dim, patches, patches)
        x = x.flatten(2)  # 将二维 patch 扁平化为 (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # 形状: (batch_size, num_patches, embed_dim)

        # 在 patch 嵌入后，添加可学习的类别 token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 形状: (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # 形状: (batch_size, num_patches + 1, embed_dim)

        # 加上位置编码
        x = x + self.position_embeddings

        # Transformer 编码器
        for self_attention, feed_forward, layer_norm in self.encoder:
            # 1. 自注意力机制：查询（Q）、键（K）、值（V）是输入 x，本质是计算每个patch对其他patch的关注度
            Q = self_attention[0](x)  # 用于 Q 的线性变换
            K = self_attention[1](x)  # 用于 K 的线性变换
            V = self_attention[2](x)  # 用于 V 的线性变换

            # 计算自注意力 (Attention scores)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # 形状: (batch_size, num_patches+1, num_patches+1)
            attention_scores = attention_scores / (K.size(-1) ** 0.5)  # 缩放

            # 使用 softmax 计算注意力权重
            attention_weights = F.softmax(attention_scores, dim=-1)  # 形状: (batch_size, num_patches+1, num_patches+1)

            # 计算加权的值 (最终的注意力输出)
            attention_output = torch.matmul(attention_weights, V)  # 形状: (batch_size, num_patches+1, embed_dim)

            # 2. 残差连接和层归一化
            x = layer_norm(x + attention_output)

            # 3. 前馈网络
            ff_output = feed_forward(x)

            # 4. 残差连接和层归一化
            x = layer_norm(x + ff_output)

        # 分类头：取出 CLS token 的表示用于分类
        x = x[:, 0]  # 形状: (batch_size, embed_dim)

        # 通过分类层输出类别
        x = self.fc(x)

        return x




def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])

def resnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3])


#model = resnet101(3,1000).cuda()#操作放在GPU
#print(model)
# 示例：初始化模型并打印模型结构
model = VisionTransformer(img_size=224, patch_size=16, num_classes=1000, embed_dim=768).cuda()
# model.to(device)

loss = nn.CrossEntropyLoss() # 分类任务的loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 1
prediction_val = []

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0#训练
    train_loss = 0.0
    val_acc = 0.0#验证
    val_loss = 0.0

    model.train() # 在训练时启用batch normalization和drop out(测试时操作不同)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())#获取预测 Variable 的内部 Tensor, 转换成numpy类型. 获取第二个维度中最大值(即可能性最大)索引,与真实值对比.
        train_loss += batch_loss.item()#张量转值
    
    model.eval()# 在测试时不用batch normalization和drop out
    with torch.no_grad():#内部不会track 梯度
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))


model.eval()
with torch.no_grad():
    for i, data in enumerate(val_loader):
        val_pred = model(data[0].cuda())
        val_label = np.argmax(val_pred.cpu().data.numpy(), axis=1)
        for y in val_label:
            prediction_val.append(y)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

classes = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
y_true = val_y.copy()  # 样本实际标签
y_pred = prediction_val.copy()  

cm = confusion_matrix(y_true, y_pred)



cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(22, 14), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')


train_val_x = np.concatenate((train_x, val_x), axis=0)#两个数组拼接
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)#train_val_set为实例化的1个dataset,然后用Dataloader 包起来


model_best = resnet101(3,1000).cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 30
# num_epoch = 15

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))


test_set = ImgDataset(test_x, transform=test_transform)#实例化一个dataset,然后用Dataloader 包起来
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)


with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))




