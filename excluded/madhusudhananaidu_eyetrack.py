!pip install lightning lightning-bolts hydra-core natsort


config = {
    "model": {
        "channels": [2, 8,16, 32,48, 64, 80, 96, 112, 128, 256],  # Input channels and intermediate layers
        "t_kernel_size": 7,  # Temporal kernel size
        "n_depthwise_layers": 5,  # Depthwise layers for efficiency
        "detector_head": True,  # Enable detector head
        "detector_depthwise": True,  # Standard convolutions for detector
        "full_conv3d": True,  # 1+2D convs instead of full 3D
        "norms": "all_bn"  # BatchNorm + GroupNorm combination
    },
    "dataset": {
        "spatial_downsample": [5,5],  # Reduce resolution
        "temporal_scale": False,  # No temporal scaling (disabled as per request)
        "temporal_flip":True,
        "temporal_shift":False,
        "spatial_affine":True,
        "frames_per_segment": 50,  # Number of event frames
        "events_interpolation": "causal_linear",  # Use all frames
        "time_window": 10000
    },
    "trainer": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0002,
        "epochs": 100,
        "device": 0,
        "activity_regularization" : 0
    },
    "submission": {
        "checkpoint_path": "weights/submission.ckpt",
        "output_csv": "submission.csv"
    }
}



#Eye Dataset
import ast
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from natsort import natsorted
from torch.nn import functional as F
from torch.utils.data import Dataset

rand_range = lambda amin, amax: amin + (amax - amin) * np.random.rand()

val_files = ["1_6", "2_4", "4_4", "6_2", "7_4", "9_1", "10_3", "11_2", "12_3"]

def get_index(file_lens, index):
    file_lens_cumsum = np.cumsum(np.array(file_lens))
    file_id = np.searchsorted(file_lens_cumsum, index, side='right')
    sample_id = index - file_lens_cumsum[file_id - 1] if file_id > 0 else index
    return file_id, sample_id


def txt_to_npy(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(ast.literal_eval(line.strip()))
    return np.array(data)


def h5_to_npy(file_path, name):
    with h5py.File(file_path, 'r') as file:
        npy_data = file[name][:]
    return npy_data


def events_to_frames(events, size, num_frames, spatial_downsample, temporal_downsample, mode='bilinear'):
    """Perform bilinear interpolation directly on the events, 
    while converting them to frames and do spatial and temporal downsamplings 
    all at the same time.
    """
    height, width = size    
    p, x, y, t = events
    events_frames = torch.zeros([num_frames, 2, height, width]).type_as(events)
    
    def bilinear_interp(x, scale, x_max):
        if scale == 1:
            return x, x, torch.ones_like(x), torch.zeros_like(x)
        xd1 = x % scale / scale
        xd = 1 - xd1
        x = (x / scale).long().clamp(0, x_max)
        x1 = (x + 1).clamp(0, x_max)
        return x, x1, xd, xd1
    
    if mode == 'nearest':
        p = p.round().long()
        x = (x / spatial_downsample[0]).round().long().clamp(0, width - 1)
        y = (y / spatial_downsample[1]).round().long().clamp(0, height - 1)
        t = (t / temporal_downsample - 0.5).round().long().clamp(0, num_frames - 1)
        events_frames.index_put_((t, p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True)
        return events_frames
    
    x, x1, xd, xd1 = bilinear_interp(x, spatial_downsample[0], width - 1)
    y, y1, yd, yd1 = bilinear_interp(y, spatial_downsample[1], height - 1)
    t, t1, td, td1 = bilinear_interp(t, temporal_downsample, num_frames - 1)
    
    # similar to bilinear, but temporally causal
    if mode == 'causal_linear':
        p = p.long().repeat(4)
        
        x = torch.cat([x.repeat(2), x1.repeat(2)])
        y = torch.cat([y, y1]).repeat(2)
        t = t.repeat(4)
        
        xd = torch.cat([xd.repeat(2), xd1.repeat(2)])
        yd = torch.cat([yd, yd1]).repeat(2)
        td = td1.repeat(4)  # causal
        
        events_frames.index_put_((t, p, y, x), xd * yd * td, accumulate=True)
        return events_frames

    # bilinear
    p = p.long().repeat(8)

    x = torch.cat([x.repeat(4), x1.repeat(4)])
    y = torch.cat([y.repeat(2), y1.repeat(2)]).repeat(2)
    t = torch.cat([t, t1]).repeat(4)

    xd = torch.cat([xd.repeat(4), xd1.repeat(4)])
    yd = torch.cat([yd.repeat(2), yd1.repeat(2)]).repeat(2)
    td = torch.cat([td, td1]).repeat(4)

    events_frames.index_put_((t, p, y, x), xd * yd * td, accumulate=True)
    return events_frames

# def events_to_frames(events, size, num_frames, spatial_downsample,temporal_downsample, mode='adaptive'):
#     """Adaptive binning based on event density."""
#     height, width = size    
#     p, x, y, t = events

#     # Compute event density per frame
#     min_time, max_time = t.min(), t.max()
#     frame_edges = torch.linspace(min_time, max_time, num_frames + 1)
#     event_bins = torch.bucketize(t.to(frame_edges.device), frame_edges) - 1 # Assign events to bins

#     events_frames = torch.zeros([num_frames, 2, height, width]).type_as(events)

#     for i in range(num_frames):
#         mask = (event_bins == i)
#         if mask.sum() == 0:
#             continue  # No events in this bin

#         x_bin, y_bin, p_bin = x[mask], y[mask], p[mask]

#         # Perform bilinear interpolation for spatial assignment
#         x_low, x_high = x_bin.floor().long(), (x_bin + 1).floor().long()
#         y_low, y_high = y_bin.floor().long(), (y_bin + 1).floor().long()

#         x_low = x_low.clamp(0, width - 1)
#         x_high = x_high.clamp(0, width - 1)
#         y_low = y_low.clamp(0, height - 1)
#         y_high = y_high.clamp(0, height - 1)

#         # Assign events to bins with weights
#         events_frames[i, p_bin.long(), y_low, x_low] += 1
#         events_frames[i, p_bin.long(), y_low, x_high] += 1
#         events_frames[i, p_bin.long(), y_high, x_low] += 1
    #     events_frames[i, p_bin.long(), y_high, x_high] += 1

    # return events_frames



class EventRandomAffine():
    """Perform random affine transformations on the events and labels
    """
    def __init__(self, size, 
                 degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), spatial_jitter=None, 
                 augment_flag=True):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.spatial_jitter = spatial_jitter
        self.augment_flag = augment_flag

        self.height, self.width = size

    def normalize(self, coords, backward=False):
        if not backward:
            coords[0] = coords[0] / self.width - 0.5
            coords[1] = coords[1] / self.height - 0.5
        else:
            coords[0] = (coords[0] + 0.5) * self.width
            coords[1] = (coords[1] + 0.5) * self.height

        return coords

    def __call__(self, events, labels):        
        if self.augment_flag:
            degrees = rand_range(-self.degrees, self.degrees) / 180 * math.pi
            translate = [rand_range(-t, t) for t in self.translate]
            scale = [rand_range(*self.scale) for _ in range(2)]

            cos, sin = math.cos(degrees), math.sin(degrees)
            R = torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=torch.float).type_as(events)
            S = torch.tensor([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]], dtype=torch.float).type_as(events)
            T = torch.tensor([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]], dtype=torch.float).type_as(events)

            trans_matrix = T @ R @ S
        
        else:
            trans_matrix = torch.eye(3).type_as(events)
        
        coords = F.pad(events[1:3], (0, 0, 0, 1), value=1)
        coords = self.normalize(trans_matrix @ self.normalize(coords), True)
        if self.spatial_jitter is not None:
            coords += torch.randn_like(coords) * self.spatial_jitter
        
        events[1:3] = coords[:2]
        val_inds = (coords[0] >= 0) & (coords[0] < self.width) & (coords[1] >= 0) & (coords[1] < self.height)
        events = events[:, val_inds]
        
        labels = labels.T
        centers = F.pad(labels[:2], (0, 0, 0, 1), value=1)
        centers = trans_matrix @ self.normalize(centers)
        centers = centers[:2] + 0.5
        
        closes = labels[-1]
        
        return events, centers, closes
    
    
class EyeTrackingDataset(Dataset):
    def __init__(self, 
                 root_path, 
                 mode='train', 
                 device='cpu', 
                 time_window=10000, 
                 frames_per_segment=50, 
                 spatial_downsample=(5, 5), 
                 events_interpolation='bilinear', 
                 spatial_affine=True, 
                 temporal_flip=True, 
                 temporal_scale=True, 
                 temporal_shift=True,
                 test_on_val=False):
        self.mode = mode
        self.time_window = time_window
        self.frames_per_segment = frames_per_segment
        self.time_window_per_segment = time_window * frames_per_segment
        self.spatial_downsample = spatial_downsample
        self.events_interpolation = events_interpolation
        assert time_window == 10000
        
        self.temporal_flip = temporal_flip
        self.temporal_scale = temporal_scale
        self.temporal_shift = temporal_shift
        
        self.test_on_val = test_on_val
        
        root_path = Path(root_path)
        if mode in ['train', 'val']:
        #if mode == 'train':
            base_path = root_path / 'train'
        elif mode == 'test':
            if test_on_val:
                base_path = root_path / 'train'
            else:
                base_path = root_path / 'test'
        else:
            raise ValueError("Invalid mode. Most be train or test.")
        #data_dirs = natsorted(base_path.glob('*'))
        
        self.events, self.labels = [], []
        self.num_frames_list, self.num_segments_list = [], []
        
        dir_paths = natsorted(base_path.glob('*'))
        if mode == 'train':
            dir_paths = [dir_path for dir_path in dir_paths if dir_path.name not in val_files]
        elif mode == 'val' or (mode == 'test' and test_on_val):
            dir_paths = [dir_path for dir_path in dir_paths if dir_path.name in val_files]


        for dir_path in dir_paths:
#        for dir_path in data_dirs:
            assert dir_path.is_dir()
            data_path = dir_path / f'{dir_path.name}.h5'
            label_path = dir_path / 'label.txt' if (mode != 'test' or test_on_val) else dir_path / 'label_zeros.txt'
            
            event, label = h5_to_npy(data_path, 'events'), txt_to_npy(label_path)
            
            num_frames = label.shape[0]
            self.num_frames_list.append(num_frames)
            self.num_segments_list.append(num_frames // frames_per_segment)
            
            # truncating off trailing events with no labels
            final_t = num_frames * time_window
            final_ind = np.searchsorted(event['t'], final_t, 'left')
            event = event[:final_ind]
            
            label = torch.tensor(label, dtype=torch.float, device=device)
            event = np.stack([event['p'].astype('float32'), event['x'].astype('float32'), event['y'].astype('float32'), event['t'].astype('float32')], axis=0)
            event = torch.tensor(event, dtype=torch.float, device=device)  # (4, N)
            
            self.events.append(event)
            self.labels.append(label)
        
        self.total_segments = sum(self.num_segments_list)
        
        # spatial affine transformation
        augment_flag = (mode == 'train') and spatial_affine
        self.augment = EventRandomAffine((480, 640), augment_flag=augment_flag)
            
    def __len__(self):
        if self.mode == 'test':
            return len(self.events)
        return self.total_segments
    
    def _process_data(self, event, label, index=None):
        event, center, close = self.augment(event, label)
        num_frames = self.frames_per_segment if self.mode != 'test' else self.num_frames_list[index]
        
        event = events_to_frames(event, 
                                 (480 // self.spatial_downsample[1], 640 // self.spatial_downsample[0]), 
                                 num_frames, self.spatial_downsample, self.time_window, 
                                 mode=self.events_interpolation)
        
        # time + polarity flip
        if self.mode == 'train' and self.temporal_flip and np.random.rand() > 0.5:
            event = event.flip(0).flip(1)  # (T, C, H, W)
            center = center.flip(-1)
            close = close.flip(-1)
        
        return event.moveaxis(0, 1), center, 1 - close
    
    def __getitem__(self, index):
        if self.mode == 'test':
            event, label = self.events[index], self.labels[index]
            return self._process_data(event, label, index)
        
        file_id, segment_id = get_index(self.num_segments_list, index)
        event, label = self.events[file_id], self.labels[file_id]
        
        start_t = segment_id * self.time_window * self.frames_per_segment
        end_t = start_t + self.time_window * self.frames_per_segment
        
        # random temporal shift
        max_offset = round(self.time_window_per_segment * 0.1)
        if self.mode == 'train' and self.temporal_shift and start_t >= max_offset:
            offset = np.random.rand() * max_offset
            start_t -= offset
            end_t -= offset
        else:
            offset = 0
        
        # random temporal scaling
        num_frames = self.num_frames_list[file_id]
        event = event.clone()
        if self.mode == 'train' and self.temporal_scale and end_t < (num_frames * self.time_window * 0.8):
            # scale_factor = float(rand_range(0.8, 1.2))
            # event[-1] *= scale_factor
            scale_factor = 1
        else:
            scale_factor = 1
        
        start_ind = torch.searchsorted(event[-1], start_t, side='left')
        end_ind = torch.searchsorted(event[-1], end_t, side='left')
        
        event_segment = event[:, start_ind.item():end_ind.item()]
        event_segment[-1] -= start_t
        
        start_label_id = segment_id * self.frames_per_segment
        end_label_id = (segment_id + 1) * self.frames_per_segment
        
        # label interpolation
        label_numpy = label.cpu().numpy()
        num_frame = label_numpy.shape[0]
        arange = np.arange(0, num_frame)
        label_offset = offset / self.time_window
        interp_range = np.linspace(
            (start_label_id - label_offset) / scale_factor, 
            (end_label_id - label_offset - 1) / scale_factor, 
            self.frames_per_segment, 
        )
        x_interp = np.interp(interp_range, arange, label_numpy[:, 0])
        y_interp = np.interp(interp_range, arange, label_numpy[:, 1])
        closeness = label_numpy[start_label_id:end_label_id, -1]
        label_segment = torch.tensor(np.stack([x_interp, y_interp, closeness], axis=1)).type_as(label)
        
        return self._process_data(event_segment, label_segment)
        


#Losses
import copy
import math

import torch
from torch.nn import functional as F


class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)
    
    
class MacsEstimationHook:
    def __init__(self, num_conv_layers):
        self.num_conv_layers = num_conv_layers
        
        self.layer_id = 0
        self._params_per_layer = torch.zeros(num_conv_layers)
        self._macs_per_layer = torch.zeros(num_conv_layers)
        self._macs_per_layer_with_sparsity = torch.zeros(num_conv_layers)
        
        self.nonzeros = torch.zeros(num_conv_layers)
        self.totals = torch.zeros(num_conv_layers)
    
    def __call__(self, module, input, output):
        output_size = math.prod((output.shape[1],) + output.shape[3:])
        macs_weight = output_size * math.prod(module.weight.shape[1:])
        macs_bias = output_size
        
        self._params_per_layer[self.layer_id] = module.weight.numel() + output.shape[1]
        
        macs = macs_weight + macs_bias
        self._macs_per_layer[self.layer_id] = macs
        
        self.nonzeros[self.layer_id] += (input[0] != 0).sum().item()
        self.totals[self.layer_id] += input[0].numel()
        self._macs_per_layer_with_sparsity[self.layer_id] = macs * self.nonzeros[self.layer_id] / self.totals[self.layer_id]
        
        self.layer_id = (self.layer_id + 1) % self.num_conv_layers
        
    @property
    def macs_per_layer(self):
        return self._macs_per_layer.round().long()
    
    @property
    def macs_per_layer_with_sparsity(self):
        return self._macs_per_layer_with_sparsity.round().long()
    
    @property
    def params_per_layer(self):
        return self._params_per_layer.round().long()


class RegularizationLoss():
    def __init__(self, reg_factor, model):

        self.reg_factor = reg_factor # 1e-1 was awesome!!!
        if reg_factor > 0:
            # Hook for regularization of activations of ReLUs
            self.output_hook = OutputHook()
            for mm in model.modules():
                if isinstance(mm, torch.nn.ReLU):
                    mm.register_forward_hook(self.output_hook)

    def __call__(self, ):
        if self.reg_factor > 0:
            l1_penalty = 0.
            for output in self.output_hook:
                l1_penalty += torch.norm(output, 1)/output.numel()
            l1_penalty *= self.reg_factor
            self.output_hook.clear()
            return l1_penalty
        else:
            return 0.


def regression_loss(pred, center, openness):
    x, y = center.moveaxis(1, 0)
    
    pred = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    center_loss = F.smooth_l1_loss(pred, center, beta=0.11, reduction='none').sum(1)  # (batch, frames)
    valid_mask = openness.eq(1) & x.gt(0) & x.lt(1) & y.gt(0) & y.lt(1)
    center_loss = torch.where(valid_mask, center_loss, 0).mean()
    
    return center_loss


def tracking_loss(pred, center, openness, gamma=2):
    device = pred.device
    batch_size, _, frames, height, width = pred.shape
    
    x, y = center.moveaxis(1, 0)
    x_ind = (x * width).long().clamp(0, width - 1)  # (batch, frames)
    y_ind = (y * height).long().clamp(0, height - 1)
    x_mod = (x * width) % 1
    y_mod = (y * height) % 1
    center_mod = torch.stack([x_mod, y_mod], dim=1)  # (batch, 2, frames)
    
    pred = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pred_pupil, pred_center_mod = pred[:, 0], pred[:, 1:]
    
    valid_mask = openness.eq(1) & x.gt(0) & x.lt(1) & y.gt(0) & y.lt(1)
    pupil_mask = torch.zeros_like(pred_pupil).bool()  # (batch, frames, height, width)
    
    batch_range = torch.arange(batch_size, device=device).repeat_interleave(frames)
    frames_range = torch.arange(frames, device=device).repeat(batch_size)
    pupil_mask[batch_range, frames_range, y_ind.flatten(), x_ind.flatten()] = 1
    
    # (batch, frames, height, width)
    center_loss = F.smooth_l1_loss(pred_center_mod, center_mod[..., None, None], beta=0.11, reduction='none').sum(1)
    
    focal_loss = torch.where(
        pupil_mask, 
        -1 * (1 - pred_pupil).pow(gamma) * pred_pupil.log() + center_loss, 
        -1 * pred_pupil.pow(gamma) * (1 - pred_pupil).log(), 
    )  # (batch, frames, height, width)
    
    return focal_loss[valid_mask].sum() / valid_mask.sum()


class Losses():
    """ 
    Gathers the different losses
    """
    def __init__(self, detector_head, reg_factor, model):
        self.prediction_loss = tracking_loss if detector_head else regression_loss
        self.regularization_loss = RegularizationLoss(reg_factor, model)

    def __call__(self, pred, center, openness):
        loss = self.prediction_loss(pred, center, openness)
        loss += self.regularization_loss()
        return loss


def process_detector_prediction(pred):
    device = pred.device

    if len(pred.shape)==3: # basic head case
        batch_size, _, frames = pred.shape
        x = torch.sigmoid(pred[:,0,:])
        y = torch.sigmoid(pred[:,1,:])
        
    else: # centernet head case
        batch_size, _, frames, height, width = pred.shape
        
        pred_pupil, pred_x_mod, pred_y_mod = pred.moveaxis(1, 0)
        pred_x_mod = torch.sigmoid(pred_x_mod)
        pred_y_mod = torch.sigmoid(pred_y_mod)
        
        pupil_ind = pred_pupil.flatten(-2, -1).argmax(-1)  # (batch, frames)
        pupil_ind_x = pupil_ind % width
        pupil_ind_y = pupil_ind // width
        
        batch_range = torch.arange(batch_size, device=device).repeat_interleave(frames)
        frames_range = torch.arange(frames, device=device).repeat(batch_size)
        
        pred_x_mod = pred_x_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]
        pred_y_mod = pred_y_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]

        x = (pupil_ind_x + pred_x_mod.view(batch_size, frames)) / width
        y = (pupil_ind_y + pred_y_mod.view(batch_size, frames)) / height
    
    return torch.stack([x, y], dim=1)
    

def p10_acc(pred, center, openness, detector_head=True, 
            height=60, width=80, tolerance=10):
    pred = pred.detach().clone()
    center = center.detach().clone()
    
    if detector_head:
        pred = process_detector_prediction(pred)
    else:
        pred = torch.sigmoid(pred)
    
    pred[:, 0] *= width
    pred[:, 1] *= height
    center[:, 0] *= width
    center[:, 1] *= height
    
    distances = torch.norm(center - pred, dim=1)
    distances_noblinks = distances[openness == 1]

    return (distances < tolerance).sum() / distances.numel(), (distances_noblinks < tolerance).sum() / distances_noblinks.numel(), distances.mean()


#generate val sub
import os
from pathlib import Path

from omegaconf import OmegaConf as OC

import numpy as np
import torch
import matplotlib.pyplot as plt

# from eye_dataset import EyeTrackingDataset
# from tenn_model import TennSt
# from losses import process_detector_prediction, OutputHook, MacsEstimationHook

torch.set_grad_enabled(False)

def check_val_score(checkpoint_path, checkpoint_config, remove_blinks=False, test_on_val=True):
    data_path = '/kaggle/input/event-based-eye-tracking-cvpr-2025/event_data/event_data'

    if test_on_val:
        # Val Data
        data_files = ["1_6", "2_4", "4_4", "6_2", "7_4", "9_1", "10_3", "11_2", "12_3"]
    else:
        # Test data
        data_files = ["1_1", "2_2", "3_1", "4_2", "5_2", "6_4", "7_5", "8_2", "8_3", "10_2", "11_3", "12_4"]

    print(checkpoint_config)
    print(checkpoint_path)

    config = OC.load(checkpoint_config)
    model = TennSt(**OC.to_container(config.model))
    model.eval()
    
    if checkpoint_path is not None:
        weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        print(list(weights.keys()))
        # mystr = list(weights.keys())[0].split('backbone')[0]
        # mysr = list(weights.keys())[0].split('gru')[0] # get the str before backbone
        # weights = {k.partition(mystr)[2]: v for k, v in weights.items() if k.startswith((mystr,mysr))}
        # model.load_state_dict(weights)
        # Load checkpoint
        # weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        
        # Find the common prefix (before 'backbone' or 'gru')
        prefix_backbone = list(weights.keys())[0].split('backbone')[0]
        prefix_gru = list(weights.keys())[0].split('gru')[0]
        
        # Remove the prefixes and correctly load matching keys
        filtered_weights = {}
        for k, v in weights.items():
            if k.startswith(prefix_backbone):
                new_key = k[len(prefix_backbone):]  # Remove the prefix
            elif k.startswith(prefix_gru):
                new_key = k[len(prefix_gru):]  # Remove the prefix
            else:
                continue  # Skip unrelated keys
            filtered_weights[new_key] = v
        
        # Load weights into model
        model.load_state_dict(filtered_weights,strict=False)  # Allow non-matching keys


    testset = EyeTrackingDataset(data_path, 'test', **OC.to_container(config.dataset), test_on_val=test_on_val)

    collected_distances = np.zeros((0,))

    # Setup per hooks to grab outputs from ReLU layers to measure sparsity
    output_hook = OutputHook()
    relu_counter = 0
    for mm in model.modules():
        if isinstance(mm, torch.nn.ReLU):
            mm.register_forward_hook(output_hook)
            relu_counter+=1

    evdensity_per_layer = []
    for ll in range(relu_counter):
        evdensity_per_layer.append({
            'evs_per_time': np.zeros((0, ))
        })
        
    num_conv_layers = 0
    for mm in model.modules():
        if isinstance(mm, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            num_conv_layers += 1
        
    macs_hook = MacsEstimationHook(num_conv_layers)
    for mm in model.modules():
        if isinstance(mm, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            mm.register_forward_hook(macs_hook)
    
    results = []
    datafile = 0
    for (event, center, openness) in testset:
        pred = model(event.unsqueeze(0))
        pred = process_detector_prediction(pred).squeeze(0)

        # Grab layer outputs for sparsity calculations
        for ll, outs in enumerate(output_hook):
            summation_axis = (0, 1)
            for dim in range(3, outs.ndim):
                summation_axis += (dim,)
            
            evs = torch.sum(outs > 0, axis=summation_axis).detach().numpy() / (np.prod(outs.shape) / outs.shape[2])
            evdensity_per_layer[ll]['evs_per_time'] = np.concatenate((evdensity_per_layer[ll]['evs_per_time'], evs))
        output_hook.clear()
        
        pred[0] *= 80
        pred[1] *= 60
        center[0] *= 80
        center[1] *= 60
        distances = torch.norm(center - pred, dim=0)
        if remove_blinks:
            distances = distances[openness == 1]

        pred = pred.detach().numpy()
        center = center.detach().numpy()
        distances = distances.detach().numpy()

        p10 = (distances < 10).sum() / distances.size
        distances_5th = distances[::5]
        p10_5th = (distances_5th < 10).sum() / distances_5th.size
        distances.mean()
        collected_distances = np.concatenate((collected_distances, distances_5th), axis=0)
        results.append(
            {
                'datafile': data_files[datafile],
                'pred': pred,
                'center': center,
                'distances': distances,
                'p10_all': p10,
                'p10_5th': p10_5th,
                'openness': openness
            }
        )
        datafile += 1

    metrics = {}
    p10_total = (collected_distances < 10).sum() / collected_distances.size
    print('Overall p10 (100Hz): ' + str(p10_total))
    euc_total = collected_distances.mean()
    print('Overall Euc. Dist (100Hz): ' + str(euc_total))
    collected_5th_distances = collected_distances[::5]
    p10_5th_total = (collected_5th_distances < 10).sum() / collected_5th_distances.size
    print('Overall p10  (20Hz): ' + str(p10_5th_total))
    metrics['p10'] = p10_5th_total
    euc_5th_total = collected_5th_distances.mean()
    print('Overall Euc. Dist (20Hz): ' + str(euc_5th_total))
    metrics['distance'] = euc_5th_total
    
    mean_event_densities = []
    for ll in evdensity_per_layer:
        mean_event_densities.append(np.mean(ll['evs_per_time']))
        
    metrics['mean_event_density'] = mean_event_densities
    metrics['macs_per_layer'] = list(macs_hook.macs_per_layer.numpy())
    
    print(f"Parameters per conv layer: {macs_hook.params_per_layer}")
    print(f"Parameters of the network: {sum(macs_hook.params_per_layer)}")
    
    print(f"\nOutput event density per ReLU layer: {[float(f'{density:.3f}') for density in mean_event_densities]}")
    print(f"MACs per frame for each conv layer: {macs_hook.macs_per_layer}")
    print(f"Total MACs per frame of the network: {sum(macs_hook.macs_per_layer)}")
    print(f"MACs per frame for each conv layer (considering sparsity): {macs_hook.macs_per_layer_with_sparsity}")
    print(f"Total MACs per frame of the network (considering sparsity): {sum(macs_hook.macs_per_layer_with_sparsity)}")
    
    return results, metrics


def plot_results(grouped_results, test_on_val=True):
    outdir = './val_results' if test_on_val else './test_results'
    os.makedirs(outdir, exist_ok = True) 
    refres = grouped_results[0]
    for eix in range(len(refres)):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True, sharex=True)
        for rix, results in enumerate(grouped_results):
            expt = results[eix]
            distances = expt['distances']
            axs[0].plot(distances)
            if np.any(distances>10):
                misses = np.where(distances>10)
                axs[0].plot(misses, np.ones_like(misses)*(11+rix), '.r')

            axs[1].plot(expt['pred'][0], alpha=0.3)
            axs[2].plot(expt['pred'][1], alpha=0.3)

        blinks = np.where(expt['openness']==0)
        axs[0].plot(blinks, np.ones_like(blinks)*(-1), '.k')
        axs[0].set_ylabel('Distance')
        axs[0].set_title('Validation File: ' + expt['datafile'])
        axs[0].set_ylim([-1.4, 20])

        axs[1].plot(expt['center'][0], 'xkcd:aqua', label='X', linewidth=3)
        axs[1].plot(blinks, np.ones_like(blinks)*(5), '.k')
        axs[1].set_ylabel('X Position')
        axs[1].set_ylim([0, 80])

        axs[2].plot(expt['center'][1], 'xkcd:salmon', label='Y', linewidth=3)
        axs[2].plot(blinks, np.ones_like(blinks)*(5), '.k')
        axs[2].set_ylabel('Y Position')
        axs[2].set_xlabel('Timestep')
        axs[2].set_ylim([0, 60])
        fig.savefig(os.path.join(outdir, 'results_'+expt['datafile']+'.png'))


# if __name__=='__main__':
    
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''
#     torch.set_grad_enabled(False)
    
#     config = '/kaggle/working/config.yaml'
#     checkpoints = '/kaggle/working/lightning_logs/version_0/checkpoints/last.ckpt'

#     # centernet version
#     # checkpoints = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/11-05-58/lightning_logs/version_0/checkpoints/last.ckpt']
#     # config = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/11-05-58/lightning_logs/version_0/config.yaml']
    
#     # basic version
#     # checkpoints = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/11-42-18/lightning_logs/version_0/checkpoints/last.ckpt']
#     # config = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/11-42-18/lightning_logs/version_0/config.yaml']
    
#     test_on_val = True
#     grouped_results = []
#     for k, checkpoint in enumerate(checkpoints):
#         results, _ = check_val_score(checkpoint_path=checkpoint,
#                                     checkpoint_config = config[k],
#                                     remove_blinks=False,
#                                     test_on_val=test_on_val)
#         grouped_results.append(results)        

#     plot_results(grouped_results,
#                 test_on_val=test_on_val)




# SpatioTemporal Model
import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings

# warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
#     f'{category.__name__}: {message}\n'

class CausalGroupNorm(nn.GroupNorm):
    """A GroupNorm that does not use temporal statistics, to ensure causality
    """
    def __init__(self, num_groups, num_channels, **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)
        
    def forward(self, input):
        x = input.moveaxis(1, 2)  # (B, T, C, H, W)
        x_shape = x.shape
        x = x.flatten(0, 1)  # (B * T, C, H, W)
        x = super().forward(x).reshape(x_shape)
        return x.moveaxis(1, 2)  # (B, C, T, H, W)


act_layer = lambda: nn.ReLU()
bn_block = lambda features: nn.Sequential(nn.BatchNorm3d(features), act_layer())
gn_block = lambda features: nn.Sequential(CausalGroupNorm(4, features), act_layer())
pw_conv = lambda in_channels, out_channels: nn.Conv3d(in_channels, out_channels, 1, bias=False)


class SpatialBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 depthwise=False, 
                 kernel_size=1,
                 full_conv3d=False, 
                 norms='mixed'):
        super().__init__()
        kernel = (kernel_size,3,3)
        self.kernel_size = kernel_size
        self.full_conv3d = full_conv3d
        self.norms = norms
        self.streaming_mode = False
        self.fifo = None  # for streaming inference

        if self.norms=='all_gn':
            norm_block = gn_block
        else :
            norm_block = bn_block

        if depthwise:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel, (1, 2, 2), (0, 1, 1), groups=in_channels, bias=False), 
                norm_block(in_channels), 
                pw_conv(in_channels, out_channels), 
                norm_block(out_channels), 
            )
            
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, (1, 2, 2), (0, 1, 1), bias=False), 
                norm_block(out_channels), 
            )
        
    def streaming(self, enabled=True):
        if enabled:
            assert not self.training, "Can only use streaming mode during evaluation."
        self.streaming_mode = enabled
        
    def reset_memory(self):
        self.fifo = None
    
    def forward(self, input):
        if self.full_conv3d: 
            if self.streaming_mode:
                return self._streaming_forward(input)
            input = F.pad(input, (0, 0, 0, 0, self.kernel_size - 1, 0))
            return self.block(input)
        else:         
            return self.block(input)
            
    def _streaming_forward(self, input):
        if self.fifo is None:
            self.fifo = torch.zeros(*input.shape[:2], self.kernel_size, *input.shape[3:]).type_as(input)
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)
        return self.block(self.fifo)


class TemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 depthwise=False,
                 full_conv3d=False, 
                 norms='mixed'):
        super().__init__()
        assert out_channels % 4 == 0  # needed for group norm to work
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.norms = norms
        kernel = (kernel_size,3,3) if full_conv3d else (kernel_size,1,1)
        
        self.streaming_mode = False
        self.fifo = None  # for streaming inference
        
        if self.norms=='mixed':
            norm1_block = bn_block
            norm2_block = gn_block
        elif self.norms=='all_bn':
            norm1_block = bn_block
            norm2_block = bn_block
        elif self.norms=='all_gn':
            norm1_block = gn_block
            norm2_block = gn_block

        if depthwise:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel, groups=in_channels, bias=False), 
                norm1_block(in_channels), 
                pw_conv(in_channels, out_channels), 
                norm2_block(out_channels), 
            )
            
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, bias=False), 
                norm2_block(out_channels), 
            )

    def streaming(self, enabled=True):
        if enabled:
            assert not self.training, "Can only use streaming mode during evaluation."
        self.streaming_mode = enabled
        
    def reset_memory(self):
        self.fifo = None
    
    def forward(self, input):
        if self.streaming_mode:
            return self._streaming_forward(input)
                  
        input = F.pad(input, (0, 0, 0, 0, self.kernel_size - 1, 0))
        return self.block(input)
    
    def _streaming_forward(self, input):
        if self.fifo is None:
            self.fifo = torch.zeros(*input.shape[:2], self.kernel_size, *input.shape[3:]).type_as(input)
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)
        return self.block(self.fifo)
        
# from mamba_ssm import Mamba

# class MambaLayer(nn.Module):
#     """Replaces GRU with Mamba-based temporal modeling."""
#     def __init__(self, input_dim, hidden_dim, seq_len):
#         super().__init__()
#         self.mamba = Mamba(
#             d_model=hidden_dim,  
#             d_state=64,  # State size for SSM
#             d_conv=4,  
#             expand=2
#         )
#         self.projection = nn.Linear(hidden_dim, input_dim)  # Adjust output

#     def forward(self, x):
#         # x: (batch, seq_len, channels, height, width)
#         batch_size, seq_len, channels, height, width = x.shape
#         x = x.mean((-2, -1))  # Reduce spatial dimensions -> (B, C, T)
#         x = x.permute(0, 2, 1)  # Convert to (B, T, C)
        
#         # Check if the embedding dimension matches Mamba's input
#         # if x.shape[-1] != self.mamba.input_dim:
#         # x = self.projection(x)  # Apply linear projection if needed
        
#         x = self.mamba(x)  # Apply Mamba SSM
#         x = x.permute(0, 2, 1)  # Convert back to (B, C, T)

#         return x.view(batch_size, seq_len, channels)


class TennSt(nn.Module):
    def __init__(
        self, 
        channels, 
        t_kernel_size, 
        n_depthwise_layers, 
        detector_head, 
        detector_depthwise, 
        full_conv3d=False,
        norms='mixed',
    ):
        super().__init__()
        self.detector = detector_head
        
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        temporals = [True, False] * 5

        # self.mamba_layer = MambaLayer(input_dim=channels[-1], hidden_dim=256,seq_len=channels[-1])  
        self.lstm = nn.LSTM(input_size=channels[-1], hidden_size=256, num_layers=2, batch_first=True)
        
        self.backbone = nn.Sequential()
        for i in range(len(depthwises)):
            in_channels, out_channels = channels[i], channels[i+1]
            depthwise = depthwises[i]
            temporal = temporals[i]
            
            if temporal:
                self.backbone.append(TemporalBlock(in_channels, out_channels, 
                                                   kernel_size=t_kernel_size, depthwise=depthwise,
                                                   full_conv3d=full_conv3d, norms=norms))
            else:
                self.backbone.append(SpatialBlock(in_channels, out_channels, depthwise=depthwise,
                                                  full_conv3d=full_conv3d,
                                                  kernel_size=t_kernel_size if full_conv3d else 1,
                                                  norms=norms))
        
        if detector_head:
            self.head = nn.Sequential(
                TemporalBlock(channels[-1], channels[-1], t_kernel_size, depthwise=detector_depthwise), 
                nn.Conv3d(channels[-1], channels[-1], (1, 3, 3), (1, 1, 1), (0, 1, 1)), 
                act_layer(), 
                nn.Conv3d(channels[-1], 3, 1), 
            )
        else:
            self.head = nn.Sequential(
                nn.Conv1d(channels[-1], channels[-1], 1), 
                act_layer(), 
                nn.Conv1d(channels[-1], 2, 1), 
            )
    
    def streaming(self, enabled=True):
        if enabled:
            warnings.warn("You have enabled the streaming mode of the network. It is expected, but not checked, that the input will be of shape (batch, 1, H, W).")
        for name, module in self.named_modules():
            if name and hasattr(module, 'streaming'):
                module.streaming(enabled)
                
    def reset_memory(self):
        for name, module in self.named_modules():
            if name and hasattr(module, 'reset_memory'):
                module.reset_memory()
         
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # if self.detector:
        #     return self.head((self.backbone(input)))
        # else:
        #     return self.head(self.backbone(input).mean((-2, -1)))

        x = self.backbone(input)  # (B, C, T, H, W)
        x = x.mean((-2, -1))  # Reduce spatial dimensions -> (B, C, T)
        
        # Prepare for GRU: Convert to (B, T, C)
        x = x.permute(0, 2, 1)  # (B, T, C)
        # x = self.mamba_layer(x)
        x, _ = self.lstm(x)  # Apply GRU
        x = x.permute(0, 2, 1)  # Convert back to (B, C, T)
        
        # Expand spatial dimensions back before passing to Conv3D
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
        
        return self.head(x) if self.detector else self.head(x.mean(-2,-1))
        


import warnings
warnings.filterwarnings('ignore')


import os
import csv
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import DataLoader

config = OC.create(config)

# import losses
# from eye_dataset import EyeTrackingDataset
# from tenn_model import TennSt
# from generate_val_results import check_val_score

filename = 'results_ablation.csv'

def flatten_2levels_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.
    """
    items = {}
    for pk, pv in d.items():
        parent_key = pk + sep
        for ck, cv in pv.items():
            items[pk+'__'+ck] = cv
    return items


class CustomModule(LightningModule):
    def __init__(self, data_path, config):
        super().__init__()
        self.data_path = data_path
        self.config = config
        
        self.batch_size = config["trainer"]['batch_size']
        epochs = config['trainer']['epochs']
        detector_head = config['model']['detector_head']
        activity_regularization = config['trainer']['activity_regularization']
        
        self.model = TennSt(**OC.to_container(config.model))
        
        self.trainset = EyeTrackingDataset(data_path, 'train', config['trainer']['device'], **OC.to_container(config['dataset']))
        self.valset = EyeTrackingDataset(data_path, 'val', config['trainer']['device'], **OC.to_container(config['dataset'])) # frames_per_segment=127, device=device)
        
        num_steps_per_epoch = len(self.trainset) // self.batch_size
        self.total_train_steps = epochs * num_steps_per_epoch
        
        self.loss_fn = Losses(detector_head, activity_regularization, self.model)
        self.metric_fn = partial(p10_acc, detector_head=detector_head)
            
    def forward(self, input):
        return self.model(input)
    
    def _log(self, name, metric):
        self.log(name, metric, 
                 on_step=False, on_epoch=True, prog_bar=True)
    
    def on_train_start(self):
        log_dir = '/kaggle/working/'
        OC.save(self.config, log_dir + 'config.yaml')

    # Save final weights

    #def on_train_end(self):
        # torch.set_grad_enabled(False)
        
        # log_dir = Path(self.trainer.logger.log_dir)

        # testset = EyeTrackingDataset(self.data_path, 'test', self.config.trainer.device, 
        #                              **OC.to_container(self.config.dataset))

        # predictions = []
        # for (event, _, _) in testset:
        #     pred = self(event[None, ...])
        #     if self.config.model.detector_head:
        #         pred = losses.process_detector_prediction(pred)
        #     else:
        #         pred = torch.sigmoid(pred)
        #     predictions.append(pred.detach().squeeze(0)[..., ::5].cpu().numpy())
            
        # predictions = np.concatenate(predictions, axis=-1).T  # (frames, 2)
        # predictions[:, 0] *= 80
        # predictions[:, 1] *= 60
        # predictions = np.concatenate([np.arange(len(predictions))[:, None], predictions], axis=1)

        # df = pd.DataFrame(predictions, columns=['row_id', 'x', 'y'])
        # df.to_csv(log_dir / 'submission.csv', index=False)
    
    
    def training_step(self, batch, batch_idx):
        event, center, openness = batch
        pred = self(event)        
        loss = self.loss_fn(pred, center, openness)
        metric, metric_noblinks, distance = self.metric_fn(pred, center, openness)
        self._log('train_loss', loss)
        self._log('train_metric', metric)
        self._log('train_metric_noblinks', metric_noblinks)
        self._log('train_distance', distance)
        return loss

    def validation_step(self, batch, batch_idx):
        event, center, openness = batch
        pred = self(event)        
        loss = self.loss_fn(pred, center, openness)
        metric, metric_noblinks, distance = self.metric_fn(pred, center, openness)
        self._log('val_loss', loss)
        self._log('val_metric', metric)
        self._log('val_metric_noblinks', metric_noblinks)
        self._log('val_distance', distance)
            
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.002, weight_decay=0.001)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, round(self.total_train_steps * 0.025), self.total_train_steps, eta_min=1e-5)
        
        scheduler = {'scheduler': scheduler, 
                     'interval': 'step', 
                     'frequency': 1}
        return [optimizer], [scheduler]
            
    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, drop_last=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valset, shuffle=False, drop_last=False, batch_size=self.batch_size)
    


def main():
    data_path = '/kaggle/input/event-based-eye-tracking-cvpr-2025/event_data/event_data'
    module = CustomModule(data_path, config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric', 
        mode='max', 
        save_last=True, 
        every_n_epochs=1, 
        filename='{epoch}-{val_metric:.2f}', 
    )



    trainer = Trainer(
        max_epochs=config['trainer']['epochs'], 
        gradient_clip_val=3.0, 
        accelerator='gpu', 
        devices=[config['trainer']['device']], 
        benchmark=True, 
        log_every_n_steps=5, 
        callbacks=[checkpoint_callback]
    )

    timestamp_start = datetime.now()
    trainer.fit(module)
    timestamp_end = datetime.now()

    log_dir = Path(trainer.log_dir)
    print(log_dir)
    _, resmetrics = check_val_score(log_dir / "checkpoints" / "last.ckpt",
                              "/kaggle/working/config.yaml",
                              remove_blinks=False, test_on_val=True)
    resdict = flatten_2levels_dict(config)
    resdict['time_start'] = timestamp_start.isoformat()
    resdict['time_end'] = timestamp_end.isoformat()
    resdict['training_duration'] = int((timestamp_end-timestamp_start).total_seconds()/60) # training duration in minutes
    resdict['params'] = sum(p.numel() for p in module.parameters())
    resdict['params_trainable'] = sum(p.numel() for p in module.parameters() if p.requires_grad)

    resdict['logdir'] = trainer.log_dir
    resdict['val_p10'] = resmetrics['p10']
    resdict['val_distance'] = resmetrics['distance']
    resdict['event_density'] = resmetrics['mean_event_density']
    resdict['macs'] = resmetrics['macs_per_layer']

    # # Write the header and results to the CSV file
    # with open(os.path.dirname('/kaggle/working/'), 'a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=resdict.keys())
    #     if csvfile.tell() == 0:
    #         writer.writeheader()  # Write header only if file is empty
    #     row = {**resdict}
    #     writer.writerow(row)
    # print("\n\n")

if __name__ == "__main__":
    main()



import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf as OC

# from eye_dataset import EyeTrackingDataset
# from tenn_model import TennSt
# from losses import process_detector_prediction

# NOTE: this submission script runs the network in streaming mode, and does not use the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# torch.set_num_threads(1)
torch.set_grad_enabled(False)


def streaming_inference(model, frames):
    model.eval()
    model.streaming()
    model.reset_memory()
    
    predictions = []
    with torch.inference_mode():
        for frame_id in range(frames.shape[2]):  # stream the frames to the model
            prediction = model(frames[:, :, [frame_id]])
            predictions.append(prediction)
                
    predictions = torch.cat(predictions, dim=2)
    return predictions


config_path = '/kaggle/working/config.yaml'
checkpoint_path = '/kaggle/working/lightning_logs/version_0/checkpoints/last.ckpt'

#config_path = '/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/06-03-29/lightning_logs/version_0/config.yaml'
#checkpoint_path = '/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/06-03-29/lightning_logs/version_0/checkpoints/last.ckpt'

config = OC.load(config_path)
data_path = '/kaggle/input/event-based-eye-tracking-cvpr-2025/event_data/event_data'

weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
mystr = list(weights.keys())[0].split('backbone')[0] # get the str before backbone
weights = {k.partition(mystr)[2]: v for k, v in weights.items() if k.startswith(mystr)}

model = TennSt(**OC.to_container(config.model))
model.eval()
model.load_state_dict(weights)

testset = EyeTrackingDataset(data_path, 'test', **OC.to_container(config.dataset))
event_frames_list = [event_frames for (event_frames, _, _) in testset]

predictions = []
for event_frames in event_frames_list:
    pred = streaming_inference(model, event_frames[None, :])
    pred = process_detector_prediction(pred)
    predictions.append(pred.squeeze(0))
    
predictions = torch.cat(predictions, dim=-1)

predictions[0] *= 80
predictions[1] *= 60

predictions_numpy = predictions.detach().numpy().T
predictions_numpy = np.concatenate([np.arange(len(predictions_numpy))[:, None], predictions_numpy], axis=1)

df = pd.DataFrame(predictions_numpy, columns=['row_id', 'x', 'y'])
df.to_csv('/kaggle/working/submission.csv', index=False)





