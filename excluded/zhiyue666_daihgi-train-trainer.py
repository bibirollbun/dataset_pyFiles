import os
import warnings # 避免一些可以忽略的报错
warnings.filterwarnings('ignore')
import sys
import random
import copy
import math
from tqdm.auto import tqdm
from PIL import Image
import time
import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor # 用于并行计算

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler # 学习率调度器
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL


class CONFIG:
    is_debug = True
    seed = 308
    n_folds = 5
    img_size = [512, 512]

    epochs = 20 if not is_debug else 2
    train_batch_size = 32
    valid_batch_size = 64
    n_workers = os.cpu_count()
    T_max = 79950 // n_folds * (n_folds - 1) // train_batch_size * epochs
    start_lr_backbone = 1e-5
    start_lr_head = 1e-3
    min_lr_backbone = 1e-8
    min_lr_head = 1e-6
    scheduler = 'CosineAnnealingWithWarmupLR'
    n_accumulate = 1.0

    model_name = "tf_efficientnet_b0.ns_jft_in1k"
    is_pretrained = True
    now_cv = -np.inf
    n_classes = 1
    DataParallel = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    my_skf_train_csv = "/kaggle/input/my-skf-train-csv/my_skf_train_csv.csv"
    train_csv = "/kaggle/input/detect-ai-vs-human-generated-images/train.csv"
    train_img_path = "/kaggle/input/ai-vs-human-generated-dataset/train_data"


def set_seed(seed=308):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(CONFIG.seed)


# # 检查两个数据集中的 train.csv 和 test.csv 是否一样
# train1 = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/train.csv")
# train2 = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/train.csv")

# test1 = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/test.csv")
# test2 = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/test.csv")

# print(f"train1 - train2 : \n{(train2 != train1).sum()}")
# print("*" * 50)
# print(f"test1 - test2 : \n{(test2 != test1).sum()}")


# # 查看数据集中的图像的分辨率大小分布
# # 非并行
# start_time = time.time()
# root_train_dir = "/kaggle/input/ai-vs-human-generated-dataset/train_data"
# per_names = os.listdir(root_train_dir)
# size_train = []
# for per_name in tqdm(per_names):
#     img_path = os.path.join(root_train_dir, per_name)
#     img = Image.open(img_path)
#     size_train.append(img.size)

# end_time = time.time()
# print(f"用时 : {end_time - start_time:.2f} s.")


# # 并行计算
# start_time = time.time()
# root_train_dir = "/kaggle/input/ai-vs-human-generated-dataset/train_data"
# per_names = os.listdir(root_train_dir)

# def get_image_size(img_path):
#     try:
#         img = Image.open(img_path)
#         return img.size
#     except Exception as e:
#         print(f"Error reading image {img_path}: {e}")
#         return None

# with ThreadPoolExecutor(max_workers=4) as executor:
#     size_train = []
#     for per_name in tqdm(per_names, desc="Submitting tasks"):
#         img_path = os.path.join(root_train_dir, per_name)
#         size_train.append(executor.submit(get_image_size, img_path).result())

# end_time = time.time()
# print(f"用时 : {end_time - start_time:.2f} s.")


# def get_img_sum(img):
#     return img[0] * img[1]
    
# with ThreadPoolExecutor(max_workers=4) as executor:
#     img_sum = []
#     for per_size_train in tqdm(size_train, desc="Submitting tasks"):
#         img_sum.append(executor.submit(get_img_sum, per_size_train).result())

# math.sqrt(min(img_sum)) # 293.28484447717375
# math.sqrt(max(img_sum)) # 768


if os.path.exists(CONFIG.my_skf_train_csv):
    print(f"\n my_skf_train_csv is exists loading... \n")
    train = pd.read_csv(CONFIG.my_skf_train_csv)
else:
    print(f"\n my_skf_train_csv not exists ctreating... \n")
    train = pd.read_csv(CONFIG.train_csv)
    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=CONFIG.n_folds, shuffle=True, random_state=CONFIG.seed)
    # 创建一个新列来存储K-Fold标签
    train['kfold'] = -1
    # 进行K-Fold交叉验证
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label'])):
        train.loc[val_idx, 'kfold'] = fold
    train.to_csv("my_skf_train_csv.csv", index=False)
train


def transform(img):
    composition = A.Compose([
        A.Resize(CONFIG.img_size[0], CONFIG.img_size[1]),
        ToTensorV2(),
    ])
    return composition(image=img)["image"]


class DAIHGIDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        img_id = row.file_name.split("/")[-1]
        img_path = os.path.join(CONFIG.train_img_path, img_id)
        label = torch.tensor(row.label, dtype=torch.float32)

        img = Image.open(img_path)
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[:,:, np.newaxis]
            img = np.concatenate([img] * 3, axis=2)

        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape)

        return img, label


def prepare_loaders(df, fold=0):
    df_train = df[df["kfold"] != fold]
    df_valid = df[df["kfold"] == fold]
    
    train_datasets = DAIHGIDataset(df=df_train, transform=transform)
    valid_datasets = DAIHGIDataset(df=df_valid, transform=transform)
    
    train_loader = DataLoader(train_datasets, batch_size=CONFIG.train_batch_size, num_workers=CONFIG.n_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_datasets, batch_size=CONFIG.valid_batch_size, num_workers=CONFIG.n_workers, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


# # 以下代码可检查Dataset，DataLoader是否实现基本功能
# train_loader, valid_loader = prepare_loaders(train, 0)
# x_train, y_train = next(iter(train_loader))
# x_valid, y_valid = next(iter(valid_loader))
# print(f"X_train shape : {x_train.shape}") # (batch_size, channels, H, W)
# print(f"y_train shape : {y_train.shape}")
# print(f"x_valid shape : {x_valid.shape}")
# print(f"y_valid shape : {y_valid.shape}")

# # 删除变量，回收垃圾
# del train_loader, valid_loader, x_train, y_train, x_valid, y_valid
# gc.collect()


def evaluate_model(y_true, y_pred):
    """
    评估模型的F1-Score，使用0.5阈值
    :param y_true: 真实标签（True labels）
    :param y_pred: 预测标签（Predicted probabilities）
    :return: F1-Score
    """
    # 将概率值转换为二分类标签（0或1），使用0.5作为阈值
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # 计算F1-Score
    return f1_score(y_true, y_pred_binary)
    
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.7, 0.4, 0.6, 0.6, 0.8])

print(evaluate_model(y_true, y_pred))  # 输出：0.6666666666666666


def updata_req_grad(models, requires_grad=True):
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad


class GeMPool(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMPool, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self, x, p=3, eps=1e-6):
        return torch.mean(x.clamp(min=eps).pow(p), dim=(-2, -1)).pow(1./p)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})'


class DAIHGIModel(nn.Module):
    def __init__(self):
        super(DAIHGIModel, self).__init__()
        self.backbone = timm.create_model(model_name=CONFIG.model_name, 
                                          pretrained=CONFIG.is_pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, CONFIG.n_classes)
        )
        
        
    def forward(self, x):
        _tmp = self.backbone(x)
        output = self.head(_tmp)
        return output


model = DAIHGIModel()
model


criterion = nn.BCELoss()


class Trainer():
    def __init__(self):
        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.num_epochs = CONFIG.epochs
        self.loss_fn = None
        self.evaluate_fn = evaluate_model
        self.formatted_time = None
        self.ckpt_save_path = None
        self.n_folds = CONFIG.n_folds
        self.device = CONFIG.device
        self.historys = None
        self.true = None
        self.historys = None
        self.now_cv = CONFIG.now_cv
        self.model_name = CONFIG.model_name

    def train_one_epoch(self, epoch):
        self.model.train()
        
        y_preds = []
        y_trues = []
        
        dataset_size = 0
        running_loss = 0.0
        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for step, (images, labels) in bar:
            self.optimizer.zero_grad()
            
            batch_size = images.size(0)
            if CONFIG.DataParallel:
                images = images.cuda().float()
                labels = labels.cuda().float()
            else:
                images = images.to(CONFIG.device, dtype=torch.float)
                labels = labels.to(CONFIG.device, dtype=torch.float)
                
            outputs = self.model(images)
            outputs = F.sigmoid(outputs)
            loss = criterion(outputs.flatten(), labels) / CONFIG.n_accumulate
            loss.backward()
            
            if (step + 1) % CONFIG.n_accumulate == 0:
                self.optimizer.step()
    
                # zero the parameter gradients
                self.optimizer.zero_grad()
    
            y_preds.append(outputs.flatten().detach().cpu().numpy())
            y_trues.append(labels.detach().cpu().numpy())
    
            train_cv = self.evaluate_fn(np.concatenate(y_trues).round(), np.concatenate(y_preds))
    
            running_loss += (loss.item() * batch_size)
    
            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size
            
            bar.set_postfix(Epoch=epoch,
                            Train_Loss=epoch_loss,
                            Train_CV_F1=train_cv,
                            LR_backbone=self.optimizer.optimizer1.param_groups[0]['lr'],
                            LR_head=self.optimizer.optimizer2.param_groups[0]['lr'])
        # Ensure that a parameter update is performed after the last accumulation cycle
        if (step + 1) % CONFIG.n_accumulate != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return epoch_loss, train_cv

    @torch.inference_mode()
    def valid_one_epoch(self, epoch):
        self.model.eval()
        
        y_preds = []
        y_trues = []
        dataset_size = 0
        running_loss = 0.0
        bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        with torch.no_grad():
            for step, (images, labels) in bar:
                batch_size = images.size(0)
                if CONFIG.DataParallel:
                    images = images.cuda().float()
                    labels = labels.cuda().float()
                else:
                    images = images.to(CONFIG.device, dtype=torch.float)
                    labels = labels.to(CONFIG.device, dtype=torch.float)
    
                outputs = self.model(images)
                outputs = F.sigmoid(outputs)
                loss = criterion(outputs.flatten(), labels) / CONFIG.n_accumulate
    
                y_preds.append(outputs.flatten().detach().cpu().numpy())
                y_trues.append(labels.detach().cpu().numpy())
                valid_cv = self.evaluate_fn(np.concatenate(y_trues), np.concatenate(y_preds))
            
                running_loss += (loss.item() * batch_size)
    
                dataset_size += batch_size
    
                epoch_loss = running_loss / dataset_size
    
                bar.set_postfix(Epoch=epoch,
                                Valid_Loss=epoch_loss,
                                Valid_CV_F1=valid_cv,
                                LR_backbone=self.optimizer.optimizer1.param_groups[0]['lr'],
                                LR_head=self.optimizer.optimizer2.param_groups[0]['lr'])
            
    
            y_preds = np.concatenate(y_preds)
            y_trues = np.concatenate(y_trues)
            cv = self.evaluate_fn(y_trues, y_preds) 
        
        return epoch_loss, cv
        
    def get_time(self):
        # Get the current time stamp
        current_time = time.time()
        print("Current timestamp:", current_time)
        
        # Convert a timestamp to a local time structure
        local_time = time.localtime(current_time)
        
        # Formatting local time
        self.formatted_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)
        print("now time:", self.formatted_time)
        
        self.ckpt_save_path = f"output/{self.formatted_time}_{self.model_name}_output"
        if os.path.exists(self.ckpt_save_path) is False:
            os.makedirs(self.ckpt_save_path)
    
    def run_training(self, fold):
        if torch.cuda.is_available():
            print("[INFO] Using GPU: {} x {}\n".format(torch.cuda.get_device_name(), torch.cuda.device_count()))
        
        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch_cv = self.now_cv
        best_model_path = None
        history = defaultdict(list)
        
        for epoch in range(1, self.num_epochs + 1):
            gc.collect()
            train_epoch_loss, train_epoch_cv = self.train_one_epoch(epoch)
            valid_epoch_loss, valid_epoch_cv = self.valid_one_epoch(epoch)
            print(f"epoch: {epoch}, LOSS = {valid_epoch_loss}, CV = {valid_epoch_cv}")
            
            history['Train Loss'].append(train_epoch_loss)
            history['Valid Loss'].append(valid_epoch_loss)
            history['Train CV'].append(train_epoch_cv)
            history['Valid CV'].append(valid_epoch_cv)
            history['lr_backbone'].append(self.optimizer.optimizer1.param_groups[0]['lr'])
            history['lr_head'].append(self.optimizer.optimizer2.param_groups[0]['lr'])
            
            # deep copy the model
            if valid_epoch_cv >= best_epoch_cv:
                print(f"{b_}epoch: {epoch}, Validation CV Improved ({best_epoch_cv} ---> {valid_epoch_cv}))")
                best_epoch_cv = valid_epoch_cv
                best_model_wts = copy.deepcopy(self.model.state_dict())
                PATH = "./{}/{}_CV_{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(CONFIG.ckpt_save_path, fold, best_epoch_cv, valid_epoch_loss, epoch)
                best_model_path = PATH
                torch.save(self.model.state_dict(), PATH)
                print(f"Model Saved{sr_}")
                
            print()
        
        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best CV: {:.4f}".format(best_epoch_cv))
    
        # load best model weights
        self.model.load_state_dict(best_model_wts)
    
        return self.model, history, best_model_path

    def run(self):
        self.get_time()
        self.oof = []
        self.true = []
        self.historys = []
        
        
        for fold in range(0, self.n_folds):
            print(f"==================== Train on Fold {fold+1} ====================")
            del self.model
            torch.cuda.empty_cache()
            self.model = DAIHGIModel()
            if CONFIG.DataParallel:
                device_ids = [0, 1]
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
                self.model = model.cuda()
            else:
                self.model = self.model.to(self.device)
                
            self.optimizer = get_optimizer(self.model)
            
            self.train_loader, self.valid_loader = prepare_loaders(train, fold)
            _, history, best_model_path = self.run_training(fold+1)
            self.historys.append(history)
            
            bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
            with torch.no_grad():
                for step, (images, labels) in bar:
                    batch_size = images.size(0)
                    if CONFIG.DataParallel:
                        images = images.cuda().float()
                        labels = labels.cuda().float()
                    else:
                        images = images.to(CONFIG.device, dtype=torch.float)
                        labels = labels.to(CONFIG.device, dtype=torch.float)
        
                    outputs = self.model(images)
                    outputs = F.sigmoid(outputs)
                    
                    oof.append(outputs.flatten().detach().cpu().numpy())
                    true.append(labels.detach().cpu().numpy())
                print()
        
        self.oof = np.concatenate(oof)
        self.true = np.concatenate(true)


class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_epochs=10, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = T_max - warmup_epochs
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [(base_lr * (self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_epoch = self.last_epoch - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cosine_epoch / self.cosine_epochs)) / 2 for base_lr in self.base_lrs]


# lr scheduler
def fetch_scheduler(optimizer, T_max, min_lr):
    if CONFIG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max, 
                                                   eta_min=min_lr)
    elif CONFIG.scheduler == "CosineAnnealingWithWarmupLR":
        scheduler = CosineAnnealingWithWarmupLR(optimizer, T_max=T_max, eta_min=min_lr, warmup_epochs=T_max//CONFIG.epochs)
        
    elif CONFIG.scheduler == None:
        return None
        
    return scheduler


class merge_optim():
    def __init__(self, optimizer1, optimizer2, lr_scheduler1=None, lr_scheduler2=None):
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.lr_scheduler1 = lr_scheduler1
        self.lr_scheduler2 = lr_scheduler2

    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

    def step(self):
        self.optimizer1.step()
        self.optimizer2.step()
        if self.lr_scheduler1 is not None:
            self.lr_scheduler1.step()
        if self.lr_scheduler2 is not None:
            self.lr_scheduler2.step()


def get_optimizer(model):
    optimizer_backbone = optim.AdamW(model.backbone.parameters(), lr=CONFIG.start_lr_backbone)
    optimizer_head = optim.AdamW(model.head.parameters(), lr=CONFIG.start_lr_head)
    
    scheduler_backbone = fetch_scheduler(optimizer_backbone, T_max=CONFIG.T_max, min_lr=CONFIG.min_lr_backbone)
    scheduler_head = fetch_scheduler(optimizer_head, T_max=CONFIG.T_max, min_lr=CONFIG.min_lr_head)
    
    optimizer = merge_optim(optimizer_backbone, optimizer_head, scheduler_backbone, scheduler_head)
    return optimizer


trainer = Trainer()


trainer.run()













