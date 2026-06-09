import warnings
import numpy as np

warnings.simplefilter('ignore')

SEED = 333
np.random.seed(SEED)


import torch

vesuvius_path = '/kaggle/input/vesuvius-challenge-ink-detection/'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


import os
from torch import Tensor
import cv2

TIF_STARTS = 27
TIF_RANGE = 10

def load_image_stack(ab_123,
                     train_rectangle=None,
                     test_train='train'):

  surface_volume_filepath = vesuvius_path + \
    f'{test_train}/{ab_123}/surface_volume/'
  tif_filepaths = \
    [surface_volume_filepath + tif_filepath \
     for tif_filepath in sorted(os.listdir(surface_volume_filepath))[:-4]]
  tif_filepaths_stack = tif_filepaths[TIF_STARTS:TIF_STARTS + TIF_RANGE]

  image_stack = []
  for tif_filepath_stack in tif_filepaths_stack:
    loaded_img = Tensor(cv2.imread(tif_filepath_stack,
                                         0) / 65535.0).float() \
      .to(device)
    if train_rectangle is None:
      image_stack.append(loaded_img)
    else:
      image_stack.append(loaded_img[train_rectangle[1]:train_rectangle[1] + \
                                    train_rectangle[3],
                                    train_rectangle[0]:train_rectangle[0] + \
                                    train_rectangle[2]])

  return torch.stack(image_stack, dim=0)


def load_train_mask_label(train_123, train_rectangle=None):

  mask_filepath = vesuvius_path + f'train/{train_123}/mask.png'
  label_filepath = vesuvius_path + f'train/{train_123}/inklabels.png'

  mask = cv2.imread(mask_filepath, 0) / 255.
  label = cv2.imread(label_filepath, 0) / 255.
  mask = mask[train_rectangle[1]:train_rectangle[1] + train_rectangle[3],
              train_rectangle[0]:train_rectangle[0] + train_rectangle[2]]
  label = label[train_rectangle[1]:train_rectangle[1] + train_rectangle[3],
                train_rectangle[0]:train_rectangle[0] + train_rectangle[2]]

  return mask, label


IMAGE_SIZE = 64
RANGE = IMAGE_SIZE // 2
STRIDE = 17

def get_non_zero_indices(mask):
  trim_margins = np.zeros(mask.shape, dtype=bool)
  trim_margins[RANGE:mask.shape[0] - RANGE,
             RANGE:mask.shape[1] - RANGE] = True

  trim_margins_mask = np.array(mask) * trim_margins
  del trim_margins

  sparse_mask = np.zeros(mask.shape, dtype=bool)
  sparse_mask[::STRIDE, ::STRIDE] = True

  return np.argwhere(sparse_mask * trim_margins_mask)


import torch.nn as nn
from torch.utils.data import Dataset

class SubvolumeDataset(Dataset):

  def __init__(self, image_stack, label, non_zero_indices):
    self.image_stack = image_stack
    self.label = Tensor(label).float()
    self.non_zero_indices = non_zero_indices

  def __len__(self):
    return len(self.non_zero_indices)

  def __getitem__(self, idx):
    y, x = self.non_zero_indices[idx]
    subvolume = self.image_stack[:,
                                 y - RANGE:y + RANGE,
                                 x - RANGE:x + RANGE]
    ink_label = self.label[y, x]

    return subvolume, ink_label


from torch.nn import Module, Sequential, Conv3d, ReLU, BatchNorm3d

BATCH_NORM_MOMENTUM = 0.1
FILTERS = [16, 32, 64, 128]
FILTER_SIZES = [1] + FILTERS
FILTER_PAIRS = list(zip(FILTER_SIZES[:-1], FILTER_SIZES[1:]))
STRIDES = [1, 2, 2, 2]
KERNEL_SIZE = 3
PADDING = 1

class Subvolume3DcnnEncoder(Module):

  def __init__(self):

    super().__init__()
    self.conv_layers = Sequential(
        *[Sequential(
            Conv3d(chan_in,
                   chan_out,
                   kernel_size=KERNEL_SIZE,
                   stride=stride, padding=PADDING),
            ReLU(),
            BatchNorm3d(num_features=filter_,
                        momentum=BATCH_NORM_MOMENTUM)) \
          for (chan_in, chan_out), stride, filter_ in zip(FILTER_PAIRS,
                                                          STRIDES,
                                                          FILTERS)])
    self.apply(self.init_weight)

  @staticmethod
  def init_weight(w):
    if isinstance(w, Conv3d):
      nn.init.xavier_uniform_(w.weight)
      nn.init.zeros_(w.bias)

  def forward(self, x):
    return self.conv_layers(x)


from torch.nn import Linear, Flatten, Sigmoid

class LinearInkDecoder(nn.Module):

  def __init__(self, input_shape):

    super().__init__()
    self.linear = Linear(int(np.prod(input_shape)), 1)
    self.flatten = Flatten()
    self.sigmoid = Sigmoid()

  def forward(self, x):
    return self.sigmoid(self.linear(self.flatten(x)))


SUBVOLUME_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, 10]

class InkClassifier3DCNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.encoder = Subvolume3DcnnEncoder()
    self.decoder = LinearInkDecoder(
        self.encoder(torch.zeros((1,
                                  1,
                                  *SUBVOLUME_SHAPE))).shape[1:])
  def forward(self, x):
    return self.decoder(self.encoder(x))


from torch.optim import Adam, lr_scheduler

LR = 3e-3

model = InkClassifier3DCNN().to(device)
torch_rand = torch.rand((5, 1, IMAGE_SIZE, IMAGE_SIZE, 10)).to(device)

TOTAL_TRAINING_STEPS = 14000

SUBVOLUME_TRAINING_STEPS = 4000

loss_fn = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=LR,
                                                total_steps=TOTAL_TRAINING_STEPS)


model.train()


# v4
TRAIN_RECTANGLES = [
     [1, [500, 1500, 3700, 6400]],
     [1, [2000, 400, 2500, 1000]],
     [1, [2200, 400, 2300, 900]],
     [2, [700, 300, 5000, 4300]],
     [2, [500, 4800, 8900, 10000]],
     [2, [1200, 4800, 5400, 9700]],
     [2, [1400, 10600, 7700, 3700]],
     [2, [600, 6100, 8700, 5700]],
     [3, [1950, 800, 1950, 900]],
     [3, [1450, 2350, 3750, 1300]],
     [3, [100, 4150, 4350, 1650]]
    ]


import random
from torch.utils.data import DataLoader
import gc

training_step = 0
BATCH_SIZE = 32

while training_step < TOTAL_TRAINING_STEPS:

  random.shuffle(TRAIN_RECTANGLES)
  for train_123, train_rectangle in TRAIN_RECTANGLES:

    if training_step >= TOTAL_TRAINING_STEPS:
      break

    mask, label = load_train_mask_label(train_123,
                                        train_rectangle=train_rectangle)
    image_stack = load_image_stack(train_123, train_rectangle=train_rectangle)
    non_zero_indices = get_non_zero_indices(mask)
    del mask

    dataset = SubvolumeDataset(image_stack, label, non_zero_indices)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    dataloader_loss = []
    dataloader_step = 0

    for subvolumes, ink_labels in dataloader:

      if dataloader_step >= SUBVOLUME_TRAINING_STEPS \
        or training_step >= TOTAL_TRAINING_STEPS:
        break

      ink_labels = ink_labels.to(device)
      logits = model(subvolumes.permute(0, 2, 3, 1).unsqueeze(dim=1))
      loss = loss_fn(logits, ink_labels.unsqueeze(dim=1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()
      dataloader_loss.append(loss.item())
      training_step += 1
      dataloader_step += 1

    del image_stack, label, dataset, dataloader
    gc.collect()


torch.save(model, 'vc_train_s_333_tts_14000_sts_4000_v4.pt')




