!pip install -q efficientnet_pytorch


!pip install -q torchsampler


import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


import numpy as np 
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import cv2
import random
import time
import re
from datetime import datetime


import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
import timm


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# === Ä�Æ°á»�ng dáº«n dá»¯ liá»‡u ===
PATH = "/kaggle/input/alaska2-image-steganalysis"


SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


class DatasetRetriever(Dataset):
    
    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image = cv2.imread(f'{PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        target = torch.zeros(4, dtype=torch.float32)
        
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)


CLASSES = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
N_SPLITS = 5

dataset = []
for label, kind in enumerate(CLASSES):
    image_paths = glob(os.path.join(PATH, kind, '*.jpg'))
    for path in image_paths:
        dataset.append({
            'kind': kind,
            'image_name': os.path.basename(path),
            'label': label
        })

# Shuffle vÃ  táº¡o DataFrame
random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

# GÃ¡n fold máº·c Ä‘á»‹nh
dataset.loc[:, 'fold'] = 0

# Chia K-Fold theo image_name (Ä‘áº£m báº£o nhÃ³m áº£nh khÃ´ng trÃ¹ng)
gkf = GroupKFold(n_splits=N_SPLITS)

for fold_number, (train_index, val_index) in enumerate(gkf.split
                                                       (X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number


def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):

    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image = cv2.imread(f'{PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            
        target = onehot(4, label)
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)


print("Chuyá»ƒn táº­p dá»¯ liá»‡u thÃ nh cÃ¡c vector one-hot cho tá»«ng lá»›p:")
for i in range(4):
    print(f"Class {i}: {onehot(4, i)}")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)
    
    @property
    def avg(self):
        return self.score


def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        if np.sum(mask) == 0:
            continue

        x_padding = np.linspace(fpr[mask][-1], 1, 100)
            
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min 
            
        score = metrics.auc(x, y)
        submetric = score * weight
        competition_metric += submetric

    return competition_metric / normalization


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


### Lá»›p Fitter â€“ Pipeline Huáº¥n Luyá»‡n Tuá»³ Chá»‰nh Cho BÃ i ToÃ¡n PhÃ¡t Hiá»‡n Giáº¥u Tin

Lá»›p `Fitter` Ä‘Æ°á»£c thiáº¿t káº¿ nhÆ° má»™t vÃ²ng láº·p huáº¥n luyá»‡n tuá»³ chá»‰nh trong PyTorch, bao trÃ¹m toÃ n bá»™ quy trÃ¬nh train/val vÃ  Ä‘Æ°á»£c trang bá»‹ Ä‘á»ƒ:
- Quáº£n lÃ½ optimizer, scheduler
- Theo dÃµi loss/AUC
- LÆ°u/khÃ´i phá»¥c checkpoint
- Tiáº¿p tá»¥c training sau khi ngáº¯t (resume)

---

#### 1. Khá»Ÿi Táº¡o (`__init__`)
HÃ m khá»Ÿi táº¡o bao gá»“m:
- Nháº­n `model`, `device`, vÃ  `config`
- Táº¡o optimizer: `AdamW`
- Thiáº¿t láº­p scheduler (vd: `ReduceLROnPlateau`)
- Loss function: `LabelSmoothing`
- Cáº¥u trÃºc lÆ°u log

```python
self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
self.criterion = LabelSmoothing().to(self.device)
```

---

#### 2. VÃ²ng Huáº¥n Luyá»‡n ChÃ­nh (`fit`)
HÃ m quáº£n lÃ½ toÃ n bá»™ training qua nhiá»�u epoch:

1. Trong má»—i epoch:
   - In learning rate
   - Gá»� `train_model()` huáº¥n luyá»‡n
   - Ghi log loss + AUC training
   - Gá»� `validation()` Ä‘á»ƒ Ä‘Ã¡nh giÃ¡
   - LÆ°u best checkpoint náº¿u loss giáº£m
   - Giá»¯ láº¡i 3 checkpoint tá»‘t nháº¥t (xÃ³a cÃ¡i cÅ©)
   - Cáº­p nháº­t learning rate (náº¿u scheduler Ä‘Æ°á»£c báº­t)

---

#### 3. Giai Ä�oáº¡n Huáº¥n Luyá»‡n (`train_model`)

- MÃ´ hÃ¬nh chuyá»ƒn sang `train()`
- VÃ²ng láº·p qua batches:
  - Load data lÃªn GPU
  - TÃ­nh loss (label smoothing)
  - Cáº­p nháº­t optimizer, metrics (loss + AUC)

Náº¿u `step_scheduler=True` thÃ¬ gá»� `scheduler.step()` sau má»—i batch.

---

## 4. Giai Ä�oáº¡n Ä�Ã¡nh GiÃ¡ (`validation`)

- MÃ´ hÃ¬nh chuyá»ƒn sang `eval()`
- Dá»¯ liá»‡u Ä‘Æ°á»£c dÃ²ng qua `torch.no_grad()`
- TÃ­nh loss + ROC AUC
- Tráº£ vá»� trung bÃ¬nh metrics

---

## 5. LÆ°u & Load Checkpoint (`save` / `load`)

### `save(path)`
- LÆ°u: state_dict cá»§a model, optimizer, scheduler, epoch, best loss

### `load(path)`
- Load tá»« checkpoint trÆ°á»›c vÃ  resume training tá»« epoch +1

```python
self.epoch = checkpoint['epoch'] + 1
```

---

## 6. Ghi Log (`log`)

Ghi cÃ¡c thÃ´ng tin training ra cáº£ terminal vÃ  file `log.txt`, há»®fu Ã­ch khi training trÃªn Kaggle hoáº·c server.

---

## Tá»•ng Káº¿t
Lá»›p `Fitter` giÃºc tá»• chá»©c pipeline training má»™t cÃ¡ch linh hoáº¡t, bá»�n vá»¯ng:

- Há»— trá»£ resume Ä‘Ãºng epoch
- Ghi log chi tiáº¿t
- Cáº­p nháº­t learning rate Ä‘á»™ng
- Theo dÃµi metric (loss, AUC)

Ráº¥t phÃ¹ há»£p cho bÃ i toÃ¡n giá»›i háº¡n thá»�i gian nhÆ° ALASKA2 (giá»›i háº¡n 12h/session).


class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        
        self.base_dir = './'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5 

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = LabelSmoothing().to(self.device)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.epoch, self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            # Training
            t = time.time()
            summary_loss, final_scores = self.train_model(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            # Validation
            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            # Next epoch
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores

    def train_model(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss, final_scores
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        self.log(f'ğŸ”� Ä�Ã£ load checkpoint tá»« {path}, resume tá»« epoch {self.epoch}')
    
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


def parse_log_file(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    epochs = []
    train_loss = []
    train_score = []
    train_time = []

    val_loss = []
    val_score = []
    val_time = []

    lr_list = []

    current_lr = None
    for line in lines:
        if line.startswith('LR:'):
            current_lr = float(line.strip().split(':')[1])
        elif '[RESULT]: Train.' in line:
            epoch = int(re.search(r'Epoch: (\d+)', line).group(1))
            summary_loss = float(re.search(r'summary_loss: ([\d.]+)', line).group(1))
            final_score = float(re.search(r'final_score: ([\d.]+)', line).group(1))
            t_time = float(re.search(r'time: ([\d.]+)', line).group(1))

            epochs.append(epoch)
            train_loss.append(summary_loss)
            train_score.append(final_score)
            train_time.append(t_time)
            lr_list.append(current_lr)  # log lr theo má»—i epoch

        elif '[RESULT]: Val.' in line:
            val_summary_loss = float(re.search(r'summary_loss: ([\d.]+)', line).group(1))
            val_final_score = float(re.search(r'final_score: ([\d.]+)', line).group(1))
            val_t_time = float(re.search(r'time: ([\d.]+)', line).group(1))

            val_loss.append(val_summary_loss)
            val_score.append(val_final_score)
            val_time.append(val_t_time)

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_score': train_score,
        'val_score': val_score,
        'train_time': train_time,
        'val_time': val_time,
        'lr': lr_list
    }


def plot_log_results(metrics, save_path='log_plots.png'):
    epochs = metrics['epochs']

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics from Log File', fontsize=16)

    # 1. Loss
    axs[0, 0].plot(epochs, metrics['train_loss'], label='Train Loss', marker='o')
    axs[0, 0].plot(epochs, metrics['val_loss'], label='Val Loss', marker='x')
    axs[0, 0].set_title('Loss per Epoch')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # 2. Score
    axs[0, 1].plot(epochs, metrics['train_score'], label='Train Score', marker='o')
    axs[0, 1].plot(epochs, metrics['val_score'], label='Val Score', marker='x')
    axs[0, 1].set_title('AUC score per Epoch')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].legend()

    # 3. LR
    axs[1, 0].plot(epochs, metrics['lr'], label='Learning Rate', marker='o')
    axs[1, 0].set_title('Learning Rate')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('LR')
    axs[1, 0].legend()

    # 4. Time
    axs[1, 1].plot(epochs, metrics['train_time'], label='Train Time (s)', marker='o')
    axs[1, 1].plot(epochs, metrics['val_time'], label='Val Time (s)', marker='x')
    axs[1, 1].set_title('Time per Epoch')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Seconds')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"âœ… Ä�Ã£ lÆ°u biá»ƒu Ä‘á»“ táº¡i: {save_path}")


class EffNet(nn.Module):
    
    def __init__(self, out_dim):
        super(EffNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(12, 36, 3, stride=1, padding=1, bias=False)
        self.mybn1 = nn.BatchNorm2d(6)
        self.mybn2 = nn.BatchNorm2d(12)
        self.mybn3 = nn.BatchNorm2d(36)

        self.net = timm.create_model('efficientnet_b0', pretrained=True)
        self.net.conv_stem.weight = nn.Parameter(self.net.conv_stem.weight.repeat(1, 12, 1, 1))

        self.dropout = nn.Dropout(0.5)
        self.net.blocks[5] = nn.Identity()
        self.net.blocks[6] = nn.Sequential(
            nn.Conv2d(self.net.blocks[4][2].conv_pwl.out_channels, self.net.conv_head.in_channels, 1),
            nn.BatchNorm2d(self.net.conv_head.in_channels),
            nn.ReLU6(),
        )
        self.myfc = nn.Linear(self.net.classifier.in_features, out_dim)
        self.net.classifier = nn.Identity()

    def extract(self, x):
        x = F.relu6(self.mybn1(self.conv1(x)))
        x = F.relu6(self.mybn2(self.conv2(x)))
        x = F.relu6(self.mybn3(self.conv3(x)))
        x = self.net(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(self.dropout(x))
        return x


model = EffNet(4).cuda()


# ---------- CONFIG ---------------
class Config:
    batch_size = 16
    n_epochs = 11 # Thá»±c táº¿ vá»›i 12 tiáº¿ng session cá»§a kaggle chá»‰ train Ä‘Æ°á»£c tá»‘i Ä‘a 5 epoch thÃ´i 
    num_workers = 4
    lr = 0.001
    # -----------------------------
    verbose = True
    verbose_step = 1
    # -----------------------------
    step_scheduler = False  # Ko chá»‰nh lr sau má»—i batch
    validation_scheduler = True  # Chá»‰nh sau má»—i epoch
    #------------------------------
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )


fold_number = 0

train_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] != fold_number].kind.values,
    image_names=dataset[dataset['fold'] != fold_number].image_name.values,
    labels=dataset[dataset['fold'] != fold_number].label.values,
    transforms=get_train_transforms(),
)

validation_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] == fold_number].kind.values,
    image_names=dataset[dataset['fold'] == fold_number].image_name.values,
    labels=dataset[dataset['fold'] == fold_number].label.values,
    transforms=get_valid_transforms(),
)

train_loader = DataLoader(
    train_dataset,
    sampler = ImbalancedDatasetSampler(train_dataset, labels=train_dataset.get_labels()),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=False,
    drop_last=True,  
)

val_loader = DataLoader(
    validation_dataset, 
    sampler=SequentialSampler(validation_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    shuffle=False,
    pin_memory=False,
)


class TrainingSession:
    def __init__(self, model, config, train_loader, val_loader,
                 ckpt_folder ='/kaggle/input/alaska-checkpoint', # thÆ° má»¥c chá»©a checkpoint
                 output_log_path ='/kaggle/working/log.txt'):
        
        self.model = model
        self.device = torch.device('cuda:0')
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_folder = ckpt_folder
        self.output_log_path = output_log_path

    def append_previous_log(self):
        prev_log = os.path.join(self.ckpt_folder, 'log.txt')
        if os.path.exists(prev_log):
            with open(prev_log, 'r') as f:
                old_content = f.read()
            with open(self.output_log_path, 'a+') as f:
                f.write('\n\n# ==== Log tá»« phiÃªn trÆ°á»›c ====\n')
                f.write(old_content)
                f.write('\n\n# ==== Báº¯t Ä‘áº§u phiÃªn má»›i ====\n')
            print(f'ğŸ“œ Ghi láº¡i log cÅ© tá»« {prev_log} vÃ o {self.output_log_path}')
        else:
            print(f'âš ï¸� KhÃ´ng tÃ¬m tháº¥y log.txt trong {self.ckpt_folder}')
            
    def get_latest_best_checkpoint(self):
        pattern = re.compile(r'best-checkpoint-(\d+)epoch\.bin')
        max_epoch = -1
        best_path = None

        if not os.path.exists(self.ckpt_folder):
            return None

        for fname in os.listdir(self.ckpt_folder):
            match = pattern.match(fname)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    best_path = os.path.join(self.ckpt_folder, fname)

        return best_path
    
    def run(self):
        fitter = Fitter(self.model, self.device, self.config)
        self.append_previous_log()

        # LuÃ´n luÃ´n tÃ¬m vÃ  load best checkpoint náº¿u cÃ³
        best_ckpt = self.get_latest_best_checkpoint()
        if best_ckpt is not None:
            print(f'ğŸ”� Resume tá»« checkpoint: {best_ckpt}')
            fitter.load(best_ckpt)
        else:
            print(f'ğŸš¨ KhÃ´ng cÃ³ checkpoint nÃ o, báº¯t Ä‘áº§u tá»« Ä‘áº§u (epoch 0)')

        print(f'â–¶ï¸� Báº¯t Ä‘áº§u training tá»« epoch {fitter.epoch}')
        fitter.fit(self.train_loader, self.val_loader)


session = TrainingSession(model, Config, train_loader, val_loader)
#session.run()


#checkpoint = torch.load('../input/alaska-checkpoint/best-checkpoint-004epoch.bin')
#checkpoint = torch.load('../input/alaska-checkpoint/best-checkpoint-008epoch.bin')
checkpoint = torch.load('../input/alaska-checkpoint/best-checkpoint-010epoch.bin')
model.load_state_dict(checkpoint['model_state_dict']);
model.eval();


checkpoint.keys()


metrics = parse_log_file('/kaggle/input/alaska-checkpoint/log.txt')
plot_log_results(metrics, save_path='/kaggle/working/log_plots.png')


def get_test_transforms(mode):
    if mode == 0:
        return A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
    elif mode == 1:
        return A.Compose([
                A.HorizontalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)    
    elif mode == 2:
        return A.Compose([
                A.VerticalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
    else:
        return A.Compose([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)


class DatasetSubmissionRetriever(Dataset):

    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image = cv2.imread(f'{PATH}/Test/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]


results = []
for mode in range(0, 4):
    dataset = DatasetSubmissionRetriever(
        image_names=np.array([path.split('/')[-1] for path in glob('../input/alaska2-image-steganalysis/Test/*.jpg')]),
        transforms=get_test_transforms(mode),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    result = {'Id': [], 'Label': []}
    for step, (image_names, images) in enumerate(data_loader):
        print(step, end='\r')
        
        y_pred = model(images.cuda())
        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        
        result['Id'].extend(image_names)
        result['Label'].extend(y_pred)

    results.append(result)


submissions = []
for mode in range(0,4):
    submission = pd.DataFrame(results[mode])
    submissions.append(submission)


y_pred = model(images.cuda())
y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]

result['Id'].extend(image_names)
result['Label'].extend(y_pred)


submissions = []
for mode in range(0,4):
    submission = pd.DataFrame(results[mode])
    submissions.append(submission)


for mode in range(0,4):
    submissions[mode].to_csv(f'submission_{mode}.csv', index=False)


weight0=5  
weight1=1  
weight2=1  
weight3=3  
weight=weight0+weight1+weight2+weight3


submissions[0]['Label'] = (submissions[0]['Label']*weight0 + submissions[1]['Label']*weight1 
                           + submissions[2]['Label']*weight2 + submissions[3]['Label']*weight3) / weight
submissions[0].to_csv(f'submission.csv', index=False)

