import os
import sys
import time
import datetime
import json

import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose
from bisect import bisect_right

import utils
import network
import transforms as T

import random
import gc

from tqdm.auto import tqdm

# Need to use parallel in apex, torch ddp can cause bugs when computing gradient penalty
# import apex.parallel as parallel


class CFG:
    # model related
    model = 'InversionNet' #  'generator name'
    model_d = 'Discriminator' # 'discriminator name'
    up_mode = None # 'upsampling layer mode such as "nearest", "bicubic", etc.'
    sample_spatial = 1.0 # 'spatial sampling ratio'
    sample_temporal = 1 # 'temporal sampling ratio'

    # Loss related
    lambda_g1v = 100.0
    lambda_g2v = 0.0
    lambda_adv = 1.0
    lambda_gp = 10.0

    # Training ralted
    k = 1 # 'k in log transformation'
    weight_decay = 1e-4
    batch_size = 64
    n_critic = 5 # 'generator & discriminator update ratio'
    lr_g = 0.0001 # 'initial learning rate of generator'
    lr_d = 0.0001 # 'initial learning rate of discriminator'
    lr_milestones = [] # 'decrease lr on milestones'
    momentum = 0.9 # momentum
    lr_gamma = 0.1 # 'decrease lr by a factor of lr-gamma'
    lr_warmup_epochs = 0 # 'number of warmup epochs'
    epoch_block = 40 # 'epochs in a saved block'
    num_block = 5 # 'number of saved block'
    workers = 4
    print_freq = 20 # 'print frequency'
    start_epoch = 0 # 'start epoch'

    pretrained = True
    pretrain_path = '/kaggle/input/waveform-inversion-models/pretrained_models/VelocityGAN/flatvel_b_l2_480.pth'

    resume = None

    output_path = '/kaggle/working/'

    seed = 2025

    run_train = False

args = CFG()

def seed_torch(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_torch(args.seed)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not milestones == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr *
            warmup_factor *
            self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class FWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''
    def __init__(self, inputs, outputs, preload=True, sample_ratio=1, file_size=500,
                    transform_data=None, transform_label=None):
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        if outputs is not None:
            self.batches = [str(inputs[i])+'&'+str(outputs[i]) for i in range(len(inputs))]
        else:
            self.batches = [str(inputs[i]) for i in range(len(inputs))]
        if preload: 
            self.data_list, self.label_list = [], []
            for batch in self.batches: 
                data, label = self.load_every(batch)
                self.data_list.append(data)
                if label is not None:
                    self.label_list.append(label)

    # Load from one line
    def load_every(self, batch):
        batch = batch.split('&')
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')
        if len(batch) > 1:
            label_path = batch[1]
            label = np.load(label_path)
            label = label.astype('float32')
        else:
            label = None
        
        return data, label
        
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx] if len(self.label_list) != 0 else None
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        return data, label if label is not None else np.array([])
        
    def __len__(self):
        return len(self.batches) * self.file_size



all_inputs = [
    f
    for f in
    Path('/kaggle/input/waveform-inversion/train_samples').rglob('*.npy')
    if ('seis' in f.stem) or ('data' in f.stem)
]

def inputs_files_to_output_files(input_files):
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]

all_outputs = inputs_files_to_output_files(all_inputs)

train_inputs = [all_inputs[i] for i in range(0, len(all_inputs), 2)] # Sample every two
valid_inputs = [f for f in all_inputs if not f in train_inputs]

train_outputs = inputs_files_to_output_files(train_inputs)
valid_outputs = inputs_files_to_output_files(valid_inputs)


transform_data = Compose([
    T.LogTransform(k=1),
    T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
])
transform_label = Compose([
    T.MinMaxNormalize(2000, 6000)
])
dataset = FWIDataset(train_inputs[:1], train_outputs[:1], transform_data=transform_data, transform_label=transform_label, file_size=500)
data, label = dataset[0]
print(data.shape)
print(label is None)

del dataset, data, label
gc.collect()


ctx_dict = {
    "flatvel-a": {
        "data_min": -26.95,
        "data_max": 52.77,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvevel-a": {
        "data_min": -27.11,
        "data_max": 55.10,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatvel-b": {
        "data_min": -27.17,
        "data_max": 56.05,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvevel-b": {
        "data_min": -29.04,
        "data_max": 57.03,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
	"flatfault-a": {
        "data_min": -26.10,
        "data_max": 50.86,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvefault-a": {
        "data_min": -26.48,
        "data_max": 52.32,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatfault-b": {
        "data_min": -24.86,
        "data_max": 50.28,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvefault-b": {
        "data_min": -24.93,
        "data_max": 50.98,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "style-a": {
        "data_min": -24.96,
        "data_max": 48.93,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "style-b": {
        "data_min": -23.76,
        "data_max": 46.01,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatvel-tutorial": {
        "data_min": -26.95,
        "data_max": 52.77,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 120,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    }
}


ctx = ctx_dict['flatfault-b']

log_data_min = T.log_transform(ctx['data_min'], k=args.k)
log_data_max = T.log_transform(ctx['data_max'], k=args.k)
transform_data = Compose([
    T.LogTransform(k=args.k),
    T.MinMaxNormalize(log_data_min, log_data_max)
])
transform_label = Compose([
    T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
])

if not args.run_train:
    train_inputs = train_inputs[:10]
    train_outputs = train_outputs[:10]

    valid_inputs = valid_inputs[:10]
    valid_outputs = valid_outputs[:10]
    
dataset_train = FWIDataset(
        train_inputs,
        train_outputs,
        preload=True,
        sample_ratio=args.sample_temporal,
        file_size=ctx['file_size'],
        transform_data=transform_data,
        transform_label=transform_label
    )

dataset_valid = FWIDataset(
    valid_inputs,
    valid_outputs,
    preload=True,
    sample_ratio=args.sample_temporal,
    file_size=ctx['file_size'],
    transform_data=transform_data,
    transform_label=transform_label
)

train_sampler = RandomSampler(dataset_train)
valid_sampler = RandomSampler(dataset_valid)


dataloader_train = DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=train_sampler, num_workers=args.workers,
    pin_memory=True, drop_last=True, collate_fn=default_collate)

dataloader_valid = DataLoader(
    dataset_valid, batch_size=args.batch_size,
    sampler=valid_sampler, num_workers=args.workers,
    pin_memory=True, collate_fn=default_collate)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = network.model_dict[args.model](upsample_mode=args.up_mode, 
        sample_spatial=args.sample_spatial, sample_temporal=args.sample_temporal).to(device)
model_d = network.model_dict[args.model_d]().to(device)


l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

def criterion_g(pred, gt, model_d=None):
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
    if model_d is not None:
        loss_adv = -torch.mean(model_d(pred))
        loss += args.lambda_adv * loss_adv
    return loss, loss_g1v, loss_g2v
criterion_d = utils.Wasserstein_GP(device, args.lambda_gp)


# Scale lr according to effective batch size
lr_g = args.lr_g
lr_d = args.lr_d
optimizer_g = torch.optim.AdamW(model.parameters(), lr=lr_g, betas=(0, 0.9), weight_decay=args.weight_decay)
optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=lr_d, betas=(0, 0.9), weight_decay=args.weight_decay)

# Convert scheduler to be per iteration instead of per epoch
warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
lr_schedulers = [WarmupMultiStepLR(
    optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
    warmup_iters=warmup_iters, warmup_factor=1e-5) for optimizer in [optimizer_g, optimizer_d]]

model_without_ddp = model
model_d_without_ddp = model_d

if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
    model_d_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model_d']))
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d'])
    args.start_epoch = checkpoint['epoch'] + 1
    step = checkpoint['step']
    for i in range(len(lr_schedulers)):
        lr_schedulers[i].load_state_dict(checkpoint['lr_schedulers'][i])
    for lr_scheduler in lr_schedulers:
        lr_scheduler.milestones = lr_milestones


if args.pretrained and args.run_train:
    checkpoint = torch.load(args.pretrain_path)
    model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
    model_d_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model_d']))


step = 0

def train_one_epoch(model, model_d, criterion_g, criterion_d, optimizer_g, optimizer_d, 
                    lr_schedulers, dataloader, device, epoch, print_freq, n_critic=5):
    global step
    model.train()
    model_d.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr_g', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('lr_d', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    itr = 0 # step in this epoch
    max_itr = len(dataloader)


    for data, label in metric_logger.log_every(dataloader, print_freq, header):
        start_time = time.time()
        data, label = data.to(device), label.to(device)

        # Update discribminator first
        optimizer_d.zero_grad()
        with torch.no_grad():
            pred = model(data)
        loss_d, loss_diff, loss_gp = criterion_d(label, pred, model_d)
        loss_d.backward()
        optimizer_d.step()
        metric_logger.update(loss_diff=loss_diff, loss_gp=loss_gp)

        # Update generator occasionally 
        if ((itr + 1) % n_critic == 0) or (itr == max_itr - 1):
            optimizer_g.zero_grad()
            pred = model(data)
            loss_g, loss_g1v, loss_g2v = criterion_g(pred, label, model_d)
            loss_g.backward()
            optimizer_g.step()
            metric_logger.update(loss_g1v=loss_g1v, loss_g2v=loss_g2v)

        batch_size = data.shape[0]
        metric_logger.update(lr_g=optimizer_g.param_groups[0]['lr'],
                            lr_d=optimizer_d.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        step += 1
        itr += 1
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()


def evaluate(model, criterion, dataloader, device, epoch):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data, label in metric_logger.log_every(dataloader, 20, header):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pred = model(data)
            loss, loss_g1v, loss_g2v = criterion(pred, label)
            metric_logger.update(loss=loss.item(), 
                                 loss_g1v=loss_g1v.item(), loss_g2v=loss_g2v.item())

            all_outputs.append(pred.cpu())
            all_labels.append(label.cpu())


    all_output = torch.concat(all_outputs, axis=0)
    all_label = torch.concat(all_labels, axis=0)
    all_output = T.minmax_denormalize(all_output, ctx['label_min'], ctx['label_max'])
    all_label = T.minmax_denormalize(all_label, ctx['label_min'], ctx['label_max'])
    l1loss_eval = l1loss(all_output, all_label)
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Loss {loss.global_avg:.8f}, L1_loss {l1loss_eval:.8f} \n'.format(loss=metric_logger.loss, l1loss_eval=l1loss_eval))
    
    if epoch % 4 == 0:
        y = all_label[0, 0].detach().cpu()
        y_pred = all_output[0, 0].detach().cpu()
        
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
        fig.suptitle(f'Epoch {epoch} | Valid: {l1loss_eval:.5f}')
        ax[0].imshow(y)
        ax[1].imshow(y_pred)
        plt.show()

    return metric_logger.loss.global_avg, l1loss_eval


if args.run_train:

    print('Start training')
    start_time = time.time()
    args.epochs = args.epoch_block * args.num_block
    
    best_loss = 5000
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, model_d, criterion_g, criterion_d, optimizer_g, optimizer_d,
                        lr_schedulers, dataloader_train, device, epoch, 
                        args.print_freq, args.n_critic)
        loss_global_avg, l1loss_eval = evaluate(model, criterion_g, dataloader_valid, device, epoch)
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'model_d': model_d_without_ddp.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'lr_schedulers': [scheduler.state_dict() for scheduler in lr_schedulers],
            'epoch': epoch,
            'step': step,
            'args': args}
    
        if l1loss_eval < best_loss:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'best_model.pth'))
            best_loss = l1loss_eval
        
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
        # Save checkpoint every epoch block
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


import csv  # Use "low-level" CSV to save memory on predictions


%%time
test_files = list(Path('/kaggle/input/waveform-inversion/test').glob('*.npy'))
len(test_files)


x_cols = [f'x_{i}' for i in range(1, 70, 2)]
fieldnames = ['oid_ypos'] + x_cols


class TestDataset(Dataset):
    def __init__(self, test_files, transform_data=None):
        self.test_files = test_files
        self.transform_data = transform_data


    def __len__(self):
        return len(self.test_files)


    def __getitem__(self, i):
        test_file = self.test_files[i]
        data = np.load(test_file)
        if self.transform_data:
            data = self.transform_data(data)

        return data, test_file.stem


ctx_test = ctx_dict['flatfault-b']


log_data_min = T.log_transform(ctx_test['data_min'], k=args.k)
log_data_max = T.log_transform(ctx_test['data_max'], k=args.k)
transform_data = Compose([
    T.LogTransform(k=args.k),
    T.MinMaxNormalize(log_data_min, log_data_max)
])


ds = TestDataset(test_files, transform_data)
dl = DataLoader(ds, batch_size=8, num_workers=4, pin_memory=True)


checkpoint = torch.load('/kaggle/input/gwi-model/cv201_best_model.pth')

model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))

# Train
model.eval()
with open('submission.csv', 'wt', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for inputs, oids_test in tqdm(dl, desc='test'):
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = model(inputs)

        y_preds = outputs[:, 0].cpu().numpy()
        y_preds = T.minmax_denormalize(y_preds, ctx_test['label_min'], ctx_test['label_max'])
        
        for y_pred, oid_test in zip(y_preds, oids_test):
            for y_pos in range(70):
                row = dict(
                    zip(
                        x_cols,
                        [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]
                    )
                )
                row['oid_ypos'] = f"{oid_test}_y_{y_pos}"
            
                writer.writerow(row)

