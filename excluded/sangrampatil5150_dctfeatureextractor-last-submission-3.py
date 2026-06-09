%%writefile dataset.py
# --- START OF FILE dataset.py ---

from torch.utils.data import Dataset
from PIL import Image, ImageFile
import random
import torchvision.transforms as transforms
import torch
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, file_list, labels=None, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform  # Custom transforms (for DCT, etc.)


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f'Error loading image: {img_path}, returning a random sample. Error: {e}')
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        original_pil_image = img.copy()  # Keep a copy of the PIL Image

        if self.transform:
            img = self.transform(img)  # Apply custom transforms

        if self.labels is not None:
            label = self.labels[idx]
            # Return the processed tensor, the original PIL image, and the label.
            return img, torch.tensor(int(label)), img_path, original_pil_image
        else:
            return img, img_path, original_pil_image  # Also return original PIL Image

class TestImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        original_pil_image = img.copy()  # Keep a copy of the PIL Image

        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path), original_pil_image # Return PIL Image


%%writefile dict.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import deque, defaultdict
import time
import datetime


def dct_matrix(size):
    """Generates the DCT transformation matrix."""
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == 0:
                m[i, j] = np.sqrt(1. / size)
            else:
                m[i, j] = np.sqrt(2. / size) * np.cos((2 * j + 1) * i * np.pi / (2 * size))
    return torch.tensor(m, dtype=torch.float32)


class DCTFeatureExtractor(nn.Module):
    def __init__(self, window_size=32, stride=16, num_patches=16, levels=1):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.num_patches = num_patches  # Number of top/bottom patches to consider
        self.levels = levels

        self.dct_basis = nn.Parameter(dct_matrix(window_size), requires_grad=False)
        self.dct_basis_t = nn.Parameter(self.dct_basis.T, requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)

        # No fold operation needed; we'll keep the patches separate.
        self.grade_filters = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False) for _ in range(6)  # Example: 6 learnable grade filters
        ])
        for filter_layer in self.grade_filters:
          nn.init.kaiming_normal_(filter_layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        N, C, H, W = x.shape

        # --- Padding ---
        pad_h = (self.stride - (H - self.window_size) % self.stride) % self.stride
        pad_w = (self.stride - (W - self.window_size) % self.stride) % self.stride
        # Pad symmetrically (reflection padding)
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
        # Now unfold
        patches = self.unfold(x)  # (N, C*window_size*window_size, L)

        L = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(N, L, C, self.window_size, self.window_size)

        # Apply DCT:  (N, L, C, w, w) @ (w, w) -> (N, L, C, w, w)
        dct_coeffs = torch.einsum('nlcwh,xy->nlcwy', patches, self.dct_basis)
        dct_coeffs = torch.einsum('nlcwh,yw->nlcyh', dct_coeffs, self.dct_basis_t)

        # Calculate energy for each patch (sum of squared DCT coefficients).
        energy = torch.sum(dct_coeffs ** 2, dim=(3, 4))  # (N, L, C)

        # --- Handle topk edge cases ---
        num_patches = min(self.num_patches, L)  # Ensure we don't select more patches than exist

        # Get top-k and bottom-k indices *before* combining channels.
        _, topk_indices = torch.topk(energy, num_patches, dim=1)  # (N, num_patches, C)
        _, bottomk_indices = torch.topk(energy, num_patches, dim=1, largest=False)  # (N, num_patches, C)

        # Combine energies across color channels (NOW we sum across channels)
        # energy = energy.sum(dim=2)  # MOVED this line down

        # Gather the top-k and bottom-k patches.
        # Expand the indices for gathering.
        dct_top_patches = torch.gather(patches, 1, topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.window_size, self.window_size))
        dct_bottom_patches = torch.gather(patches, 1, bottomk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.window_size, self.window_size))

        return dct_top_patches, dct_bottom_patches


%%writefile helper.py
import numpy as np
import math

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



%%writefile Model.py
import torch.nn as nn
import torch
import open_clip
import numpy as np
from dict import DCTFeatureExtractor


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        # Simplified HPF - single learnable 3x3 filter
        self.hpf = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        nn.init.kaiming_normal_(self.hpf.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, input):
        return self.hpf(input)


class ResNetBlock(nn.Module):
    #Simplified ResNet Block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetFeatureExtractor(nn.Module):
  def __init__(self, in_channels=3, out_channels=256):
      super().__init__()
      self.layer1 = ResNetBlock(in_channels, 64, stride=2)  # Downsample
      self.layer2 = ResNetBlock(64, 128, stride=2)      # Downsample
      self.layer3 = ResNetBlock(128, out_channels, stride=2) # Downsample
      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


  def forward(self, x):
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.avgpool(x)  # Global average pooling
      return x.view(x.size(0), -1)


class AIDE_Model(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(AIDE_Model, self).__init__()
        self.num_classes = num_classes
        self.hpf = HPF()
        self.dct_extractor = DCTFeatureExtractor()

        # Use open_clip.create_model_and_transforms consistently
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained='laion2b_s34b_b82k_augreg_soup'
        )
        # self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
        #     "RN50", pretrained='openai'
        # )
        self.clip_model = self.clip_model.visual
        self.clip_model.head = nn.Identity()  # Remove classification head

        # Freeze CLIP
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.resnet_top = ResNetFeatureExtractor()
        self.resnet_bottom = ResNetFeatureExtractor()

        # Feature fusion and classification.
        self.fc = nn.Sequential(
            nn.Linear(3584, 1024),  # 1024 from CLIP, 256 from each ResNet
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )




    def forward(self, x, original_pil_image=None):
        N, C, H, W = x.shape

        with torch.cuda.amp.autocast(enabled=True):
            processed_images = torch.stack([self.preprocess(img) for img in original_pil_image]).to(x.device)
            clip_features = self.clip_model(processed_images)

        # print(f"CLIP features shape: {clip_features.shape}")  # Keep this for now

        dct_top_patches, dct_bottom_patches = self.dct_extractor(x)
        # print(f"DCT Top Patches shape: {dct_top_patches.shape}")
        # print(f"DCT Bottom Patches shape: {dct_bottom_patches.shape}")

        dct_top_patches = self.hpf(dct_top_patches.view(-1, C, self.dct_extractor.window_size, self.dct_extractor.window_size))
        dct_bottom_patches = self.hpf(dct_bottom_patches.view(-1, C, self.dct_extractor.window_size, self.dct_extractor.window_size))

        dct_top_features = self.resnet_top(dct_top_patches).view(N, -1, 256)
        dct_bottom_features = self.resnet_bottom(dct_bottom_patches).view(N, -1, 256)
        # print(f"DCT Top Features shape: {dct_top_features.shape}")
        # print(f"DCT Bottom Features shape: {dct_bottom_features.shape}")

        dct_top_features = dct_top_features.mean(dim=1)
        dct_bottom_features = dct_bottom_features.mean(dim=1)
        # print(f"DCT Top Features shape (after mean): {dct_top_features.shape}")
        # print(f"DCT Bottom Features shape (after mean): {dct_bottom_features.shape}")

        combined_features = torch.cat([clip_features, dct_top_features, dct_bottom_features], dim=1)
        # print(f"Combined features shape: {combined_features.shape}") # And this!

        logits = self.fc(combined_features)
        return logits


%%writefile train_one_epoch.py
import os
import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score,
    accuracy_score
)
import numpy as np
import pandas as pd
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = True  # Assuming use_amp is True
    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, original_pil_images) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        original_pil_images = [img for img in original_pil_images]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(samples, original_pil_image=original_pil_images) # Pass PIL images
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        loss /= update_freq
        loss_scaler.scale(loss).backward() # Moved backward() inside the loop, before optimizer.step()

        if (data_iter_step + 1) % update_freq == 0:
             if max_norm > 0:  # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
             loss_scaler.step(optimizer)
             loss_scaler.update()
             optimizer.zero_grad()
             if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        # Calculate accuracy (handle Mixup properly)
        if mixup_fn is None:
            class_acc = (output.argmax(dim=-1) == targets).float().mean()
        else:
            # With Mixup, accuracy needs to be calculated differently.
            # Assuming targets are one-hot encoded after mixup.
            _, predicted = output.topk(1, 1, True, True)
            class_acc = predicted.eq(targets.argmax(dim=-1, keepdim=True)).float().mean()


        min_lr = 1e-4
        max_lr = 10e-5
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]


        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        if weight_decay_value is not None:
            metric_logger.update(weight_decay=weight_decay_value)
       

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            # log_writer.update(grad_norm=grad_norm, head="opt") # Also removed here
            log_writer.set_step()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_submission(data_loader, model, device, use_amp=False):
    """Evaluate the model on the submission data loader and create a submission file."""
    model.eval()
    predictions = []
    image_names = []
    confidences = []

    with torch.no_grad():
        for data, labels, names, original_pil_images in tqdm(data_loader, desc="Predicting on Test Data"): # unpack original_pil_images
            data = data.to(device)
            # We don't need labels, but we have to unpack it. We can ignore it.
            original_pil_images = [img for img in original_pil_images] # Convert to list

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(data, original_pil_image=original_pil_images) # Pass original_pil_images

            # Get predictions (class index)
            preds = output.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())

            # Get confidence scores (probability of class 1)
            confidence = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            confidences.extend(confidence)

            image_names.extend([f"test_data_v2/{name}" for name in names])  # IMPORTANT: Adjust path as needed

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': image_names,
        'label': predictions,
        #'confidence': confidences  # Include confidence
    })
    # Create submission DataFrame
    submission_df_prob = pd.DataFrame({
        'id': image_names,
        'label': predictions,
        'confidence': confidences  # Include confidence
    })

    submission_df.to_csv("submission.csv", index=False)
    submission_df_prob.to_csv("submission_prob.csv", index=False)
    print("Submission file 'submission.csv' created.")

    return pd.read_csv('submission.csv')

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    predictions = []
    labels = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]  # Use the correct target index
        original_pil_images = batch[3]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        original_pil_images = [img for img in original_pil_images]


        # compute output
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(images, original_pil_image=original_pil_images) # Pass PIL images
            loss = criterion(output, target)

        predictions.append(output.detach())
        labels.append(target.detach())

        acc1 = accuracy(output, target, topk=(1,))[0]  # Corrected accuracy call

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))


    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    if utils.is_dist_avail_and_initialized():
        output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
        dist.all_gather(output_ddp, predictions)
        predictions = torch.cat(output_ddp, dim=0)

        labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
        dist.all_gather(labels_ddp, labels)
        labels = torch.cat(labels_ddp, dim=0)


    y_pred = torch.softmax(predictions, dim=1)[:, 1].cpu().numpy()
    y_true = labels.cpu().numpy()

    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap




%%writefile utils.py
import torch
import torch.distributed as dist
import math
from collections import deque , defaultdict
import time 
import datetime
import os 
from pathlib import Path
import numpy as np 

def get_state_dict(model):
    if hasattr(model, 'module'):
        return model.module.state_dict()
    elif hasattr(model, 'state_dict_ema'):
        return model.state_dict_ema()
    else:
        return model.state_dict()
    
def load_state_dict(model, state_dict):
    model.load_state_dict(state_dict)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))




def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    if not is_main_process():
        return

    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch}.pth')
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),  # Save the GradScaler state
        'args': args,
    }

    if model_ema is not None:
         to_save['model_ema'] = get_state_dict(model_ema)

    torch.save(to_save, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")



def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        if model_ema is not None and 'model_ema' in checkpoint:
            load_state_dict(model_ema, checkpoint['model_ema'])

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr




%%writefile main.py
import os
import argparse
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from timm.optim import create_optimizer  # Changed to timm.optim
from torchvision import transforms  # Using torchvision directly
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from timm.utils import ModelEma
import torch
import random
import numpy as np
from Model import AIDE_Model  # Import the model
from dataset import ImageDataset #, TestImageDataset  # Import datasets. We might not use TestImageDataset during training.
from train_one_epoch import train_one_epoch, evaluate
from timm.data import create_transform
import utils
from torch.cuda.amp import GradScaler, autocast  # Import GradScaler
import open_clip
from tqdm import tqdm

class Config:  # Use a class for configuration
    def __init__(self):
        self.seed = 42
        self.batch_size = 64  # Per GPU batch size
        self.num_workers = os.cpu_count()
        self.pin_mem = True
        self.output_dir = 'output'
        self.model_ema_decay = 0.9998
        self.model_ema_force_cpu = False
        self.lr = 2e-5 # Initial learning rate
        self.min_lr = 1e-6 # Minimum learning rate
        self.warmup_epochs = 1
        self.epochs = 1
        self.update_freq = 1 # Gradient accumulation steps
        self.clip_grad = 0.1 # Gradient clipping norm
        self.weight_decay = 0.05
        self.smoothing = 0.1
        self.mixup = 0.8
        self.cutmix = 1.0
        self.cutmix_minmax = None
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = 'batch'
        self.nb_classes = 2  # Binary classification
        self.save_ckpt_freq = 1 # Save checkpoint every 1 epoch
        self.resume = '' # Path to resume from, e.g., 'output/checkpoint_best.pth'
        self.start_epoch = 0 # Start from this epoch when resuming
        self.eval = False  # Set to True for evaluation only mode
        self.model_ema_eval = False
        self.num_patches = 16 # Number of top-k/bottom-k patches for DCT
        self.use_amp = True # Use automatic mixed precision
        self.data_dir = '/kaggle/input/ai-vs-human-generated-dataset'  # Or your data directory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.momentum = 0.9



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Important for reproducibility

# Custom collate function
def custom_collate(batch):
    # Separate the different elements of the batch
    # Check the length of the first item in the batch.  If it's 3, we're
    # dealing with the test set (no labels). If it's 4, we have labels.
    if len(batch[0]) == 3:  # Test set (no labels)
        images, paths, pil_images = zip(*batch)
        labels = None  # No labels, so set to None
    elif len(batch[0]) == 4: # Train/val set
        images, labels, paths, pil_images = zip(*batch)
        labels = torch.stack(labels, 0) # Stack labels
    else:
        raise ValueError("Unexpected batch structure")

    images = torch.stack(images, 0) # Always stack images

    return images, labels, paths, list(pil_images)



def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='adamw', help='optimizer')
    parser.add_argument('--local_rank', type=int, default=0)
    parser_args = parser.parse_args()
    args = Config()
    args.opt = parser_args.opt
    args.gpu = parser_args.local_rank
    args.dist_url = "env://"
    init_distributed_mode(args)


    # Setup distributed training
    if args.distributed:
      args.num_tasks = utils.get_world_size()
      args.global_rank = utils.get_rank()
    else:
      args.num_tasks = 1
      args.global_rank = 0

    seed_everything(args.seed + args.global_rank)

    # --- Data Loading ---
    base_dir = args.data_dir
    train_csv_path = os.path.join(base_dir, 'train.csv')
    test_csv_path = os.path.join(base_dir, 'test.csv')

    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))
    df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))

    all_image_paths = df_train['file_name'].values
    all_labels = df_train['label'].values

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.09, random_state=args.seed, stratify=all_labels
    )

    # --- Transformations ---
    # Custom transforms for the DCT part (applied after getting the PIL Image)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Convert to tensor *before* normalization
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]), # CLIP Normalization
        ])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform=val_transform)


    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        collate_fn=custom_collate  # Use the custom collate function
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
        collate_fn=custom_collate # Use custom collate for val_loader too
    )


    # --- Model, Optimizer, Loss ---
    model = AIDE_Model(num_classes=args.nb_classes)
    model.to(args.device)


    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True) # Add find_unused_parameters
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters}")


    # --- Optimizer and Loss Function ---
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = GradScaler(enabled=args.use_amp) # Use GradScaler for AMP

    if args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # --- Mixup ---
    mixup_fn = None
    if args.mixup > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    model_ema = None
    if args.model_ema_decay > 0:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # Resume from Checkpoint 
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            loss_scaler.load_state_dict(checkpoint['scaler'])
            if model_ema is not None and 'model_ema' in checkpoint:
                model_ema.ema.load_state_dict(checkpoint['model_ema'])

            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")


    # Training Loop
    if not args.eval:
        print("Starting training...")
        best_acc1 = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, train_loader, optimizer, args.device, epoch,
                loss_scaler, args.clip_grad, model_ema, mixup_fn, args=args
            )

            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            #     utils.save_model(
            #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #         loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema
            #         )

            test_stats, acc, ap = evaluate(val_loader, model, args.device, use_amp=args.use_amp)
            print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}% Acc: {acc}")
            if best_acc1 < test_stats["acc1"]:
                best_acc1 = test_stats["acc1"]
                # utils.save_model(
                #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                #     loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {best_acc1:.2f}%')
        print(f"Best accuracy: {best_acc1:.4f}")

    else:
        print("Starting evaluation...")
        test_stats, acc, ap = evaluate(val_loader, model, args.device, use_amp=args.use_amp)
        print(f"Accuracy: {test_stats['acc1']:.4f}, AP: {ap:.4f}")

    print("Starting evaluation...")
    from dataset import TestImageDataset
    test_dataset = TestImageDataset(df_test['id'].values, transform=val_transform)

    # Use SequentialSampler for test data
    test_sampler = SequentialSampler(test_dataset)

    test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
            collate_fn=custom_collate  # Keep custom_collate
    )
    from train_one_epoch import evaluate_submission
    submission_df = evaluate_submission(test_loader, model, args.device, use_amp=args.use_amp)
    print(f"Generated submission file.  Preview:\n{submission_df.head()}")

       # print(f"Accuracy: {test_stats['acc1']:.4f}, AP: {ap:.4f}")

if __name__ == '__main__':
    main()



%%writefile requirements.txt
torch>=1.12.0  # Specify a minimum version, adjust as needed
torchvision>=0.13.0 # Corresponding torchvision version
timm>=0.9.2     # Timm library, adjust as needed
scikit-learn  # For accuracy_score, average_precision_score
numpy
pandas
Pillow  # For image loading (PIL)
open-clip-torch # important, install open-clip-torch
scipy # import scipy


!pip install -r requirements.txt


!torchrun --nproc_per_node=2 main.py

