!wc -l /kaggle/input/deepfake/phase1/trainset_label.txt
!wc -l /kaggle/input/deepfake/phase1/valset_label.txt
!ls /kaggle/input/deepfake/phase1/trainset/ | wc -l 
!ls /kaggle/input/deepfake/phase1/valset/ | wc -l 


!pip install timm


from PIL import Image
Image.open('/kaggle/input/deepfake/phase1/trainset/920085930764461878d67b71703778e8.jpg')


import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import timm
import time

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm_notebook

train_label = pd.read_csv('/kaggle/input/deepfake/phase1/trainset_label.txt')
val_label = pd.read_csv('/kaggle/input/deepfake/phase1/valset_label.txt')

train_label['path'] = '/kaggle/input/deepfake/phase1/trainset/' + train_label['img_name']
val_label['path'] = '/kaggle/input/deepfake/phase1/valset/' + val_label['img_name']


train_label['target'].value_counts()


val_label['target'].value_counts()


train_label.head(10)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""


    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in tqdm_notebook(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
            losses.update(loss.item(), input.size(0))
            top1.update(acc, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
        return top1

def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in tqdm_notebook(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input)
                output = F.softmax(output, dim=1)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
    
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
        top1.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)


class FFDIDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, torch.from_numpy(np.array(self.img_label[index]))
    
    def __len__(self):
        return len(self.img_path)


import timm
model = timm.create_model('resnet18', pretrained=True, num_classes=2)
model = model.cuda()


train_loader = torch.utils.data.DataLoader(
    FFDIDataset(train_label['path'].head(1000), train_label['target'].head(1000), 
            transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ), batch_size=40, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    FFDIDataset(val_label['path'].head(1000), val_label['target'].head(1000), 
            transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ), batch_size=40, shuffle=False, num_workers=4, pin_memory=True
)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), 0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
best_acc = 0.0
for epoch in range(2):
    scheduler.step()
    print('Epoch: ', epoch)

    train(train_loader, model, criterion, optimizer, epoch)
    val_acc = validate(val_loader, model, criterion)
    
    if val_acc.avg.item() > best_acc:
        best_acc = round(val_acc.avg.item(), 2)
        torch.save(model.state_dict(), f'./model_{best_acc}.pt')


test_loader = torch.utils.data.DataLoader(
    FFDIDataset(val_label['path'], val_label['target'], 
            transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ), batch_size=40, shuffle=False, num_workers=4, pin_memory=True
)

val_label['y_pred'] = predict(test_loader, model, 1)[:, 1]
val_label[['img_name', 'y_pred']].to_csv('submit.csv', index=None)


val_label


from torch.utils.data import Dataset
from PIL import Image

class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # 只有一张图片

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert('RGB')  # 加载图片
        if self.transform:
            image = self.transform(image)  # 应用预处理
        return image


# 图片路径
image_path = '/kaggle/input/test-1/test.jpg'

# 定义预处理操作（与训练时一致）
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建单张图片的数据集
single_image_dataset = SingleImageDataset(image_path, transform=preprocess)

# 创建 DataLoader
test_loader = torch.utils.data.DataLoader(
    single_image_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
)


# 调用 predict 函数
predictions = predict(test_loader, model, 1)  # 假设 predict 返回的是 (batch_size, num_classes) 的张量

# 提取预测结果
if predictions.shape[1] == 1:  # 二分类问题
    y_pred = torch.sigmoid(predictions).item()  # 使用 sigmoid 获取概率
else:  # 多分类问题
    y_pred = torch.softmax(predictions, dim=1).argmax(dim=1).item()  # 使用 softmax 获取类别


import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from torchvision.models import resnet18  # 改用 ResNet18

# 创建模型实例
model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 二分类任务

# 加载模型权重
state_dict = torch.load('/kaggle/working/model_61.6.pt')
model.load_state_dict(state_dict)

# 如果使用 GPU，将模型移到 GPU
if torch.cuda.is_available():
    model = model.cuda()

# 设置为评估模式
model.eval()

# 图片预处理转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 读取单张图片
img_path = '/kaggle/input/deepfake/phase1/trainset/00039e0c7c8e4eb6f3852a76ab862931.jpg'
image = Image.open(img_path)
image = transform(image)  # 转换图片
image = image.unsqueeze(0)  # 添加 batch 维度

# 如果使用 GPU，将图片移到 GPU
if torch.cuda.is_available():
    image = image.cuda()

# 预测
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    pred_prob = probabilities[0, 1].cpu().item()  # 获取第二个类别的概率

# 创建预测结果 DataFrame 并保存
result_df = pd.DataFrame({
    'img_name': ['test.jpg'],
    'y_pred': [pred_prob]
})
result_df


import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path
from torchvision.models import resnet18  # 需要导入 resnet18

# 创建模型实例
model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

# 加载模型权重
state_dict = torch.load('/kaggle/working/model_61.6.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.load_state_dict(state_dict)

if torch.cuda.is_available():
    model = model.cuda()
model.eval()

# 图片预处理转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 获取文件夹中的所有图片
image_folder = '/kaggle/input/data-class'
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png']:  # 支持多种图片格式
    image_files.extend(list(Path(image_folder).glob(ext)))

# 存储预测结果
results = []

# 处理每张图片
for img_path in image_files:
    image = Image.open(img_path).convert("RGB")  # 确保是RGB格式
    image = transform(image)
    image = image.unsqueeze(0)
    
    if torch.cuda.is_available():
        image = image.cuda()
    
    # 预测
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        pred_prob = probabilities[0, 1].cpu().item()  # 类别 1 的概率
        pred_label = torch.argmax(output, dim=1).cpu().item()  # 获取类别索引

    # 保存结果
    results.append({
        'img_name': img_path.name,  # 只保存文件名，不包含路径
        'y_pred': pred_label,  # 预测的类别标签
        'prob_0': probabilities[0, 0].cpu().item(),  # 类别 0 的概率
        'prob_1': pred_prob  # 类别 1 的概率
    })

# 创建预测结果 DataFrame 并保存
result_df = pd.DataFrame(results)
result_df.to_csv('submit.csv', index=None)

# 打印处理的图片数量
print(f"已处理 {len(results)} 张图片")
# 打印预测结果
print(result_df)



file_path = "/kaggle/input/deepfake/phase1/trainset_label.txt"

# 读取文本文件
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 打印前几行内容
for line in lines[:10]:  # 只打印前10行
    print(line.strip())


