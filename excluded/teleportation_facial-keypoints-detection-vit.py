import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchvision.transforms.v2 as transforms  # 至少需要0.23.0版本
from torchvision import tv_tensors


# torchvision版本需要0.23.0以上
!pip show torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"
path_dir = r'/kaggle/input/facial-keypoints-detection'  # 跟目录
path_save = r'/kaggle/working'  # 输出文件目录


# 定义变换
train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.3, contrast=0.2),  # 亮度、对比度调整
    transforms.RandomResizedCrop(size=(96, 96), scale=(0.8, 1.0)),  # 随机裁剪并缩放图像
    transforms.RandomRotation(degrees=20, fill=0),  # 随机旋转图像,空缺用0(黑色)填充
    transforms.CenterCrop(96),  # 裁掉旋转后的边角
    transforms.ToDtype(dtype=torch.float32, scale=True),
])

eval_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(dtype=torch.float32, scale=True),
])


class SDataset(Dataset):  # 构建训练数据集

    def __init__(self, csv_file, transform=None):
        self.d = pd.read_csv(csv_file, delimiter=',')
        self.transform = transform
        # img process
        self.d.dropna(inplace=True)
        self.d.reset_index(drop=True, inplace=True)
        self.imgs = self.d['Image'].apply(lambda x: np.array(x.split(' '), dtype='float') / 255)
        self.labels = self.d.drop(columns=['Image'])
        # self.labels.fillna(method='ffill', inplace=True)
        self.labels_tensor = torch.tensor(self.labels.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        keypoints = tv_tensors.KeyPoints(  # 用于计算随机变换后对应的特征点坐标位置
            self.labels_tensor[ix].reshape(-1, 2),  # 坐标的集合[[x1,y2],[x2,y2],...]
            canvas_size=(96, 96)  # 图片(高, 宽)
        )
        transformed_image, transformed_keypoints = self.transform(self.imgs[ix].reshape(96, 96), keypoints)
        return transformed_image, transformed_keypoints.flatten() / 48.0 - 1.0  # 放缩到[-1, 1]


class ViT(nn.Module):  # VisionTransformer

    def __init__(self, inc=1, patch_size=16, img_size=96,
                 n_head=8, dropout=0.1, n_layers=4, n_classes=30):
        super().__init__()
        self.n_emb = inc * patch_size ** 2  # 1*16*16=256
        self.pathes = (img_size // patch_size) ** 2  # 6*6=36
        self.conv = nn.Conv2d(in_channels=inc, out_channels=self.n_emb,
                              kernel_size=patch_size, stride=patch_size)  # 1,96,96 -> 256,6,6
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_emb, nhead=n_head,
                                                   dim_feedforward=4 * self.n_emb,
                                                   dropout=dropout, activation='gelu',
                                                   batch_first=True)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_tok = nn.Parameter(torch.randn((1, 1, self.n_emb)))  # 1,1,256
        self.pos_tok = nn.Parameter(torch.randn((1, 1 + self.pathes, self.n_emb)))  # 1,37,256
        self.drop_out = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(self.n_emb)
        self.c_proj = nn.Linear(self.n_emb, n_classes)

    def forward(self, x: torch.Tensor) -> None:  # B,1,96,96
        x = self.conv(x)  # B,256,6,6
        x = x.flatten(2).transpose(-1, -2)  # B,256,36->B,36,256
        cls_tok = self.cls_tok.expand(x.shape[0], -1, -1)
        x = torch.concat([cls_tok, x], dim=1)  # B,T=36+1,C=256
        x = x + self.pos_tok  # B,T,C
        x = self.transform(x)  # B,T,C
        y = self.ln(x)[:, 0]  # B,C
        return self.c_proj(y)  # B,30


@torch.no_grad()
def calculate_accuracy(net, data_loader, max_sample=1000):  # 精度计算，每次取大概1000多个样本
    loss_sum = 0
    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        pred = net(x)
        loss_sum += F.mse_loss(pred, y, reduction='sum')
        total += y.shape[0]
        if total > max_sample:
            break

    return loss_sum / total if total > 0 else -1


def model_train(csv_file: str, epochs=10, lr=3e-4, batch_size=128, param_load=None):
    data_set = SDataset(csv_file)
    cols = data_set.labels.columns.tolist()
    n_classes = len(cols)
    print(f"samples: {len(data_set)}")
    print(f"Number of classes: {n_classes}")

    train_size = int(0.9 * len(data_set))
    test_size = len(data_set) - train_size
    train_set, test_set = random_split(data_set, [train_size, test_size])
    train_set.dataset.transform = train_transform
    test_set.dataset.transform = eval_transform

    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

    net = ViT(n_classes=n_classes)
    net.to(device)
    updater = torch.optim.AdamW(net.parameters(), lr=lr)
    loss = nn.MSELoss()

    for epoch in range(epochs):
        net.train()
        for x, y in train_iter:
            updater.zero_grad()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
                y_hat = net(x)
                l = loss(y_hat, y)
            l.backward()
            updater.step()
            if l.item() != l.item():
                print(f"Loss is NAN in {epoch + 1} epoch!")
                return -1

        train_accuracy = calculate_accuracy(net, train_iter)
        test_accuracy = calculate_accuracy(net, test_iter)
        print(f"epoch: {epoch + 1}\t loss: {l.item():.6f}\t train loss: {train_accuracy :.6f}\t test loss:  {test_accuracy:.6f}")

    checkpoint = {
        'updater': updater.state_dict(),
        'model': net.state_dict(),
        'cols': cols
    }
    torch.save(checkpoint, os.path.join(path_save, r'ViT.pt'))


model_train(os.path.join(path_dir, 'training.zip'), epochs=400)


class PDataset(Dataset):  # 构建预测数据集

    def __init__(self, csv_file, transform=eval_transform):
        self.d = pd.read_csv(csv_file, delimiter=',')
        self.transform = transform
        # img process
        self.imgs = self.d['Image'].apply(lambda x: np.array(x.split(' '), dtype='float') / 255)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        transformed_image = self.transform(self.imgs[ix].reshape(96, 96))
        return transformed_image


def pred():  # 预测函数
    checkpoint = torch.load(os.path.join(path_save, r'ViT.pt'))

    dataset = PDataset(os.path.join(path_dir, r'test.zip'), transform=eval_transform)
    data_iter = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)

    # 2 net
    net = ViT(n_classes=len(checkpoint['cols']))
    net.to(device)
    net.load_state_dict(checkpoint['model'])
    net.eval()

    pred_list = []
    with torch.no_grad():
        for img in data_iter:
            img = img.to(device)
            y_hat = net(img)
            pred_list.append(y_hat * 48 + 48)  # 从[-1,1]放缩回[0,96]
        pred = torch.cat(pred_list, dim=0)

    pred = pred.cpu().detach().numpy()
    pred = pd.DataFrame(pred, columns=checkpoint['cols'])

    # kaggle submit
    d = pred.clip(0, 96)  # 最小0，最大96
    d = d.stack().reset_index()
    d.columns = ['ImageId', 'FeatureName', 'pred']
    d_submit = pd.read_csv(os.path.join(path_dir, r'IdLookupTable.csv'))
    d_submit['ImageId'] = d_submit['ImageId'] - 1
    y = d_submit.merge(d, on=['FeatureName', 'ImageId'], how='left')
    y = y[['RowId', 'pred']].rename(columns={'pred': 'Location'})
    y.to_csv(os.path.join(path_save, r'submission.csv'), index=False)


pred()

