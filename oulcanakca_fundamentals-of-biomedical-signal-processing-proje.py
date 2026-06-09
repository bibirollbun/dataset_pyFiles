import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff',
       'Replace', 'BothReleased']
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import random
from glob import glob
from tqdm.notebook import tqdm
import gc

from sklearn.model_selection import train_test_split
import numpy as np
import pywt
import pandas as pd
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
import scipy
from scipy.signal import butter, lfilter, convolve
from scipy.signal import freqz
from scipy.fftpack import fft, ifft
import argparse
import matplotlib.pyplot as plt

from sklearn import metrics
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(2021)
np.random.seed(2021)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.005, help="adam's learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--in_len", type=int, default=2**10, help="length of the input fed to neural net")
parser.add_argument("--in_channels", type=int, default=32, help="number of signal channels")
parser.add_argument("--out_channels", type=int, default=6, help="number of classes")
parser.add_argument("--chunk", type=int, default=1000, help="length of splited chunks")
parser.add_argument("--lstm_hidden", type=int, default=1024, help="length of splited chunks")
opt, unknown = parser.parse_known_args()
print(device)


# zip the files as ZIPFILE :)
import zipfile
with zipfile.ZipFile("../input/grasp-and-lift-eeg-detection/test.zip","r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/grasp-and-lift-eeg-detection/train.zip","r") as z:
    z.extractall(".")


def wavelet_denoising(x, wavelet='db2', level=3):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


%%time
def read_csv(data, events):
    x = pd.read_csv(data)
    y = pd.read_csv(events)
    id = '_'.join(x.iloc[0, 0].split('_')[:-1])
    x = x.iloc[:,1:].values
    y = y.iloc[:,1:].values
    return x, y
    

trainset = []
gt = []
for filename in tqdm(os.listdir('./train')):
    if 'data' in filename:
        data_file_name = os.path.join('./train', filename)
        id = filename.split('.')[0]
        events_file_name = os.path.join('./train', '_'.join(id.split('_')[:-1]) + '_events.csv')
        x, y = read_csv(data_file_name, events_file_name)
        x = wavelet_denoising(x)
        trainset.append(x.T.astype(np.float32))
        gt.append(y.T.astype(np.float32))


valid_dataset = trainset[-2:]
valid_gt = gt[-2:]
trainset = trainset[:-2]
gt = gt[:-2]


def resample_data(gt, chunk_size=opt.chunk):
    """
    split long signals to smaller chunks, discard no-events chunks  
    """
    total_discard_chunks = 0
    mean_val = []
    threshold = 0.01
    index = []
    
    for i in range(len(gt)):
        for j in range(0, gt[i].shape[1], chunk_size):
            mean_val.append(np.mean(gt[i][:, j:min(gt[i].shape[1],j+chunk_size)]))
            if mean_val[-1] < threshold and j > 0:  # discard chunks with low events time
                total_discard_chunks += 1
                index.extend([(i, k) for k in range(j, min(gt[i].shape[1],j+chunk_size), chunk_size//100)])
            else:
                index.extend([(i, k) for k in range(j, min(gt[i].shape[1],j+chunk_size))])

    plt.plot([0, len(mean_val)], [threshold, threshold], color='r')
    plt.scatter(range(len(mean_val)), mean_val, s=1)
    plt.show()
    print('Total number of chunks discarded: {} chunks'.format(total_discard_chunks))
    print('{}% data'.format(total_discard_chunks/len(mean_val)))
    del mean_val
    gc.collect()
    return index


%%time
class EEGSignalDataset(Dataset):
    def __init__(self, data, gt, soft_label=True, train=True):
        self.data = data
        self.gt = gt
        self.train = train
        self.soft_label = soft_label
        if train:
            self.index = resample_data(gt)
        else:
            self.index = [(i, j) for i in range(len(data)) for j in range(data[i].shape[1])]
    
    def __getitem__(self, i):
        i, j = self.index[i]
        raw_data, label = self.data[i][:,max(0, j-opt.in_len+1):j+1], \
                self.gt[i][:,j]
        pad = opt.in_len - raw_data.shape[1]
        if pad:
            raw_data = np.pad(raw_data, ((0,0),(pad,0)), 'constant',constant_values=0)

        raw_data, label = torch.from_numpy(raw_data.astype(np.float32)),\
                            torch.from_numpy(label.astype(np.float32))
        if self.soft_label:
            label[label < .05] = .05
        return raw_data, label
            
    
    def __len__(self):
        return len(self.index)
    
dataset = EEGSignalDataset(trainset, gt) 
dataloader = DataLoader(dataset, batch_size = opt.batch_size,\
                                       num_workers = opt.n_cpu, shuffle=True)
print(len(dataset))
testset = EEGSignalDataset(valid_dataset, valid_gt, train=False, soft_label=False) 
testloader = DataLoader(testset, batch_size = opt.batch_size,\
                                       num_workers = opt.n_cpu, shuffle=False)
valid_gt = np.concatenate(valid_gt, axis=1)


class NNet(nn.Module):
    def __init__(self, in_channels=opt.in_channels, out_channels=opt.out_channels):
        super(NNet, self).__init__()
        self.hidden = 32
        self.net = nn.Sequential(
            nn.BatchNorm1d(opt.in_channels),
            nn.Conv1d(opt.in_channels, opt.in_channels, 5, padding=2),
            nn.Conv1d(opt.in_channels, self.hidden, 16, stride=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden, self.hidden, 7, padding=3),
        )
        for i in range(2):
            self.net.add_module('conv{}'.format(i), \
                                self.__block(self.hidden, self.hidden)) # 16
        
        self.mid = nn.Sequential(
            self.__block(self.hidden, self.hidden),
            self.__block(self.hidden, self.hidden)
        ) # 4
        self.final = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 6),
            nn.Sigmoid()
        )
        
    def __block(self, inchannels, outchannels):
        return nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Conv1d(inchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(outchannels),
            nn.Conv1d(outchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        x = self.net(x)
        y = self.mid(x)
        y = torch.cat((x[..., -4:], y), dim=-1).view(x.shape[0], -1)
        return self.final(y)


%%time
nnet = NNet()
nnet.to(device)
loss_fnc = nn.BCELoss()
adam = optim.Adam(nnet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
loss_his, train_loss, valid_auc, train_auc = [], [], [], []
nnet.train()
for epoch in range(opt.n_epochs):
    p_bar = tqdm(dataloader)
    for i, (x, y) in enumerate(p_bar):
        x, y = x.to(device), y.to(device)
        pred = nnet(x)
        loss = loss_fnc(pred.squeeze(dim=-1), y)
        adam.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(nnet.parameters(), 2.)
        adam.step()
        train_loss.append(loss.item())
        p_bar.set_description('[Loss: {}]'.format(train_loss[-1]))
        if i % 50 == 0:
            loss_his.append(np.mean(train_loss))
            train_loss.clear()
#         if i % 1500 == 0:
#             y[y<0.1]=0
#             train_auc.append(metrics.roc_auc_score(y.detach().cpu().numpy(), \
#                                 pred.detach().cpu().numpy()))
#             valid_auc.append(calc_valid_auc(nnet))
    print('[Epoch {}/{}] [Loss: {}]'.format(epoch+1, opt.n_epochs, loss_his[-1]))
    
torch.save(nnet.state_dict(), 'model.pt')


plt.plot(range(len(loss_his)), loss_his, label='loss')
plt.legend()
plt.show()


nnet.eval()
y_pred = []
with torch.no_grad():
    for x, _ in tqdm(testloader):
        x = x.to(device)
        pred = nnet(x).detach().cpu().numpy()
        y_pred.append(pred)
y_pred = np.concatenate(y_pred, axis=0)


def plot_roc(y_true, y_pred):
    fig, axs = plt.subplots(3, 2, figsize=(15,15))
    for i, label in enumerate(labels):
        fpr, tpr, _ = metrics.roc_curve(y_true[i], y_pred[i])
        ax = axs[i//2, i%2]
        ax.plot(fpr, tpr,linewidth=3)
        ax.set_title(label+" ROC")
        ax.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    plt.show()


plot_roc(valid_gt, y_pred.T)
print('auc roc: ', metrics.roc_auc_score(valid_gt.T, y_pred))


!pip install tensorboardX -q

from tensorboardX import SummaryWriter

nnet = NNet()
nnet.to(device)
x.to(device)
# 创建一个SummaryWriter实例
writer = SummaryWriter('./runs/model_visualization')

# 添加模型的计算图
writer.add_graph(nnet, x)

# 关闭SummaryWriter
writer.close()


%%time
def read_csv(data, events):
    x = pd.read_csv(data)
    y = pd.read_csv(events)
    id = '_'.join(x.iloc[0, 0].split('_')[:-1])
    x = x.iloc[:,1:].values
    y = y.iloc[:,1:].values
    return x, y
    

trainset = []
gt = []
for filename in tqdm(os.listdir('./train')):
    if 'data' in filename:
        data_file_name = os.path.join('./train', filename)
        id = filename.split('.')[0]
        events_file_name = os.path.join('./train', '_'.join(id.split('_')[:-1]) + '_events.csv')
        x, y = read_csv(data_file_name, events_file_name)
        x = wavelet_denoising(x)
        trainset.append(x.astype(np.float32))
        gt.append(y.astype(np.float32))


valid_dataset = trainset[-2:]
valid_gt = gt[-2:]
trainset = trainset[:-2]
gt = gt[:-2]


gt[0].shape


def resample_data_time_var(gt, chunk_size=opt.chunk):
    """
    split long signals to smaller chunks, discard no-events chunks  
    """
    total_discard_chunks = 0
    mean_val = []
    threshold = 0.01
    index = []
    
    for i in range(len(gt)):
        for j in range(0, gt[i].shape[0], chunk_size):
            mean_val.append(np.mean(gt[i][j:min(gt[i].shape[0],j+chunk_size), :]))
            if mean_val[-1] < threshold and j > 0: 
                total_discard_chunks += 1
                index.extend([(i, k) for k in range(j, min(gt[i].shape[0],j+chunk_size), chunk_size//100)])
            else:
                index.extend([(i, k) for k in range(j, min(gt[i].shape[0],j+chunk_size))])

    del mean_val
    gc.collect()
    return index


%%time
class EEGSignalDataset(Dataset):
    def __init__(self, data, gt, soft_label=True, train=True):
        self.data = data
        self.gt = gt
        self.train = train
        self.soft_label = soft_label
        if train:
            self.index = resample_data_time_var(gt)
        else:
            self.index = [(i, j) for i in range(len(data)) for j in range(data[i].shape[0])]
    
    def __getitem__(self, i):
        i, j = self.index[i]
        raw_data, label = self.data[i][max(0, j-opt.in_len+1):j+1,:], \
                self.gt[i][j,:]
        pad = opt.in_len - raw_data.shape[0]
        if pad:
            raw_data = np.pad(raw_data, ((pad,0),(0,0)), 'constant',constant_values=0)

        raw_data, label = torch.from_numpy(raw_data.astype(np.float32)),\
                            torch.from_numpy(label.astype(np.float32))
        if self.soft_label:
            label[label < .05] = .05
        return raw_data, label
            
    
    def __len__(self):
        return len(self.index)
    
dataset = EEGSignalDataset(trainset, gt) 
dataloader = DataLoader(dataset, batch_size = opt.batch_size,\
                                       num_workers = opt.n_cpu, shuffle=True)
print(len(dataset))
testset = EEGSignalDataset(valid_dataset, valid_gt, train=False, soft_label=False) 
testloader = DataLoader(testset, batch_size = opt.batch_size,\
                                       num_workers = opt.n_cpu, shuffle=False)
valid_gt = np.concatenate(valid_gt, axis=0)


class LSTMNet(nn.Module):
    def __init__(self, in_channels=opt.in_channels, out_channels=opt.out_channels):
        super(LSTMNet, self).__init__()
        self.hidden = 32
        self.seq = opt.in_len
        self.LSTM = nn.Sequential(
        nn.LSTM(input_size = opt.in_channels,
                hidden_size =opt.in_channels,
                batch_first = True,dropout = 0.8,bidirectional = False),
        )  
        self.MLP = nn.Sequential(
        nn.Linear(160,80),
        nn.LeakyReLU(0.1),
        nn.Linear(80,6),
        nn.Sigmoid()
        )
        self.ConvNet = nn.Sequential(
        nn.BatchNorm1d(opt.in_channels),
        nn.Conv1d(opt.in_channels, opt.in_channels, 5, padding=2),
        nn.Conv1d(opt.in_channels, self.hidden, 16, stride=16),
        nn.LeakyReLU(0.1),
        nn.Conv1d(self.hidden, self.hidden, 7, padding=3),
        self.__block(self.hidden,self.hidden),
        self.__block(self.hidden,self.hidden),
        nn.Conv1d(opt.in_channels, self.hidden, 4, stride=4),
        )
        self.Hidden = nn.Sequential(
        self.__block(self.hidden,self.hidden),
        self.__block(self.hidden,self.hidden)
        )
    def __block(self, inchannels, outchannels):
        return nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Conv1d(inchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(outchannels),
            nn.Conv1d(outchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        x,_ = self.LSTM(x)
        x = x.permute(0,2,1)
        x = self.ConvNet(x)
        y = self.Hidden(x)
        y = torch.cat((x[...,-4:],y),dim = 2).view(x.shape[0],-1)
        y = self.MLP(y)
        return y


%%time
lstmnet = LSTMNet()
lstmnet.to(device)
loss_fnc = nn.BCELoss()
adam = optim.Adam(lstmnet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
loss_his, train_loss, valid_auc, train_auc = [], [], [], []
lstmnet.train()
for epoch in range(opt.n_epochs):
    p_bar = tqdm(dataloader)
    for i, (x, y) in enumerate(p_bar):
        x, y = x.to(device), y.to(device)
        pred = lstmnet(x)
        loss = loss_fnc(pred.squeeze(dim=-1), y)
        adam.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(lstmnet.parameters(), 2.)
        adam.step()
        train_loss.append(loss.item())
        p_bar.set_description('[Loss: {}]'.format(train_loss[-1]))
        if i % 50 == 0:
            loss_his.append(np.mean(train_loss))
            train_loss.clear()
#         if i % 1500 == 0:
#             y[y<0.1]=0
#             train_auc.append(metrics.roc_auc_score(y.detach().cpu().numpy(), \
#                                 pred.detach().cpu().numpy()))
#             valid_auc.append(calc_valid_auc(lstmnet))
    print('[Epoch {}/{}] [Loss: {}]'.format(epoch+1, opt.n_epochs, loss_his[-1]))
    
torch.save(lstmnet.state_dict(), 'model.pt')


plt.plot(range(len(loss_his)), loss_his, label='loss')
plt.legend()
plt.show()


lstmnet.eval()
y_pred = []
with torch.no_grad():
    for x, _ in tqdm(testloader):
        x = x.to(device)
        pred = lstmnet(x).detach().cpu().numpy()
        y_pred.append(pred)
y_pred = np.concatenate(y_pred, axis=0)


def plot_roc(y_true, y_pred):
    fig, axs = plt.subplots(3, 2, figsize=(15,15))
    for i, label in enumerate(labels):
        fpr, tpr, _ = metrics.roc_curve(y_true[:,i], y_pred[:,i])
        ax = axs[i//2, i%2]
        ax.plot(fpr, tpr,linewidth=3)
        ax.set_title(label+" ROC")
        ax.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    plt.show()


valid_gt.shape


plot_roc(valid_gt, y_pred)
print('auc roc: ', metrics.roc_auc_score(valid_gt, y_pred))


from tensorboardX import SummaryWriter
lstmnet = LSTMNet()
lstmnet.to(device)
x.to(device)
# 创建一个SummaryWriter实例
writer = SummaryWriter('./runs/model_visualization')

# 添加模型的计算图
writer.add_graph(lstmnet, x)

# 关闭SummaryWriter
writer.close()


%%time

!pip install torch==1.13.0 --upgrade &> null
!pip install ecgmentations &> null # library with 1d signal augmentations
!pip install git+https://github.com/rostepifanov/nnspt@release_v0.0.2 &> null # library with 1d CNNs


import os
import torch
import random
import numpy as np

seed = 1996

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.determenistic = True
torch.use_deterministic_algorithms(True)

os.environ['PYTHONHASHSEED'] = str(seed)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)


%%time

from pathlib import Path
from zipfile import ZipFile

labels = [ 'HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased' ]

zipdir = Path.cwd().parent / 'input' / 'grasp-and-lift-eeg-detection'

for zipfile in zipdir.glob('*.zip'):
    with ZipFile(zipfile, 'r') as zf:
        zf.extractall()


%%time

import pandas as pd
from tqdm import tqdm

xs = []
ys = []
ys_ = []

traindir = Path('train')

for datapath in tqdm([*sorted(traindir.glob('*_data.csv'))]):
    eventpath = datapath.parent / ( datapath.stem[:-5] + '_events.csv' )

    x = pd.read_csv(datapath)
    y = pd.read_csv(eventpath)

    x = x.iloc[:,1:].values
    y_ = y.iloc[:,1:].values

    xs.append(x.astype(np.float32))
    ys.append(y_.astype(np.uint8))

xs_train = xs[:-2]
ys_train = ys[:-2]

xs_valid = xs[-2:]
ys_valid = ys[-2:]


import ecgmentations as E

from torch.utils.data import DataLoader, Dataset

class EEGDataset(Dataset):
    def __init__(self, x, y, augs=dict, train=False):
        self.x = x
        self.y = y
        self.augs = augs
        
        self.train = train

    def __getitem__(self, idx):
        eeg = self.x[idx]
        mask = self.y[idx]

        if self.train:
            length = mask.shape[0]

            size = 5000
            smask = (np.sum(mask[:-size], axis=1) > 0).astype(np.uint8)
            smask = smask * 5 + 1
            p = smask / smask.sum()

            jdx = np.random.choice(length-size, p=p)

            eeg = eeg[jdx:jdx+size]
            mask = mask[jdx:jdx+size]

        auged = self.augs(ecg=eeg, mask=mask)
        eeg, mask = auged['ecg'], auged['mask']

        return eeg.T, mask.T

    def __len__(self):
        return len(self.x)

augs = E.Sequential([
    E.TimeCrop(length=5000, p=1.0),
])

dataset = EEGDataset(xs_train, ys_train, augs, True) 
train_dataloader = DataLoader(dataset, batch_size=25, num_workers=3, shuffle=True)

dataset = EEGDataset(xs_train, ys_train) 
train_dataloader_ = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=False)

dataset = EEGDataset(xs_valid, ys_valid)
valid_dataloader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=False)


import copy
import torch.nn.functional as F

from sklearn import metrics
from nnspt.segmentation.unet import Unet

model = Unet(in_channels=32, out_channels=6, encoder='timm-efficientnet-b1')
model.to(device)

nepochs = 1000

opt = torch.optim.AdamW(model.parameters(), lr=0.00175)
shed = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nepochs*len(train_dataloader))

loss_his, train_loss = [], []

best_score = 0.
best_state_dict = copy.deepcopy(model.state_dict())

for epoch in range(nepochs):
    model.train()

    for i, (eeg_batch, mask_batch) in enumerate(train_dataloader):
        eeg_batch, mask_batch = eeg_batch.to(device), mask_batch.to(device)

        logits = model(eeg_batch)
        loss = F.binary_cross_entropy_with_logits(logits, mask_batch.float())
        loss.backward()

        opt.step()
        shed.step()
        opt.zero_grad()

        train_loss.append(loss.item())

    if (epoch + 1) % 25 == 0:
        loss_his.append(np.mean(train_loss))
        train_loss.clear()

        print('[Epoch {}/{}] [Loss: {}]'.format(epoch+1, nepochs, loss_his[-1]))
        
        model.eval()

        y_pred = []

        size = 10000

        for eeg_batch, _ in tqdm(valid_dataloader):
            for idx in range((eeg_batch.shape[-1] + size - 1) // size):
                with torch.no_grad():
                    eeg_batch_ = eeg_batch[:, :, idx*size: (idx+1)*size].to(device)

                    logits = model(eeg_batch_)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]

                    y_pred.append(probs)

        y_pred = np.concatenate(y_pred, axis=1).T
        y_true = np.concatenate(ys_valid, axis=0)

        score = metrics.roc_auc_score(y_true, y_pred)

        print('[Epoch {}/{}] [Score: {}]'.format(epoch+1, nepochs, score))

        if score > best_score:
            best_score = score
            best_state_dict = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state_dict)


import matplotlib.pyplot as plt

def plot_roc(y_true, y_pred):
    fig, axs = plt.subplots(3, 2, figsize=(15, 13))

    for i, label in enumerate(labels):
        fpr, tpr, _ = metrics.roc_curve(y_true[i], y_pred[i])
        ax = axs[i//2, i%2]
        ax.plot(fpr, tpr)
        ax.set_title(label + ' ROC')
        ax.plot([0, 1], [0, 1], 'k--')

    plt.show()


model.eval()

y_pred = []

size = 10000

for eeg_batch, _ in tqdm(train_dataloader_):
    for idx in range((eeg_batch.shape[-1] + size - 1) // size):
        with torch.no_grad():
            eeg_batch_ = eeg_batch[:, :, idx*size: (idx+1)*size].to(device)

            logits = model(eeg_batch_)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            y_pred.append(probs)

y_pred = np.concatenate(y_pred, axis=1).T
y_true = np.concatenate(ys_train, axis=0)

plot_roc(y_true.T, y_pred.T)

print('roc auc: ', metrics.roc_auc_score(y_true, y_pred))


model.eval()

y_pred = []

size = 10000

for eeg_batch, _ in tqdm(valid_dataloader):
    for idx in range((eeg_batch.shape[-1] + size - 1) // size):
        with torch.no_grad():
            eeg_batch_ = eeg_batch[:, :, idx*size: (idx+1)*size].to(device)

            logits = model(eeg_batch_)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            y_pred.append(probs)

y_pred = np.concatenate(y_pred, axis=1).T
y_true = np.concatenate(ys_valid, axis=0)

plot_roc(y_true.T, y_pred.T)

print('roc auc: ', metrics.roc_auc_score(y_true, y_pred))


import pandas as pd
from tqdm import tqdm

xs_test = []
lengths = {}

testdir = Path('test')

FNAME = 'subj{}_series{}_{}.csv'

for subj in range(1, 13):
    for series in [9, 10]:
        datapath = testdir / FNAME.format(subj, series, 'data')

        x = pd.read_csv(datapath)
        x = x.iloc[:,1:].values

        xs_test.append(x.astype(np.float32))
        lengths['{}_{}'.format(subj, series)] = xs_test[-1].shape[0]


class EEGDatasetTest(Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        eeg = self.x[idx]

        return eeg.T

    def __len__(self):
        return len(self.x)

dataset = EEGDatasetTest(xs_test)
test_dataloader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=False)


model.eval()

y_pred = []

size = 10000

for eeg_batch in tqdm(test_dataloader):
    for idx in range((eeg_batch.shape[-1] + size - 1) // size):
        with torch.no_grad():
            eeg_batch_ = eeg_batch[:, :, idx*size: (idx+1)*size].to(device)

            logits = model(eeg_batch_)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            y_pred.append(probs)

y_pred = np.concatenate(y_pred, axis=1).T


submission = pd.DataFrame(y_pred, index=['subj{}_series{}_{}'.format(sbj, i, j) for sbj in range(1, 13) for i in [9, 10] for j in range(lengths['{}_{}'.format(sbj, i)])], columns=labels)
submission.to_csv('Submission.csv', index_label='id', float_format='%.3f')

submission.tail()


!head Submission.csv


!head sample_submission.csv

