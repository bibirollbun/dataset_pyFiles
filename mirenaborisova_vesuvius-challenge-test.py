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

TIF_STARTS =  27
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


# torch.serialization.add_safe_globals([InkClassifier3DCNN])
# torch.serialization.safe_globals([InkClassifier3DCNN])


from torch.utils.data import DataLoader
BATCH_SIZE = 32
def generate_predictions(ab_123='a', train_rectangle=None):

  if ab_123 in ['a', 'b']:
    mask_filepath = vesuvius_path + f'test/{ab_123}/mask.png'
    mask = Tensor(cv2.imread(mask_filepath, 0) / 255.)
    image_stack = load_image_stack(ab_123, test_train='test')
  elif ab_123 in [1, 2, 3]:
    mask, label = load_train_mask_label(ab_123, train_rectangle=train_rectangle)
    image_stack = load_image_stack(ab_123, train_rectangle=train_rectangle)

  test_non_zero_indices = get_non_zero_indices(mask)

  test_dataset = SubvolumeDataset(image_stack, mask, test_non_zero_indices)
  test_dataloader = DataLoader(test_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=False)

  output = torch.zeros_like(torch.Tensor(mask)).float()
  model = torch.load('/kaggle/input/vesuvius-challenge-train/vc_train_s_333_tts_14000_sts_4000_v4.pt',
                    weights_only=False)
  model.eval()

  with torch.no_grad():
    for i, (subvolumes, _) in enumerate(test_dataloader):
      for j, value in enumerate(
          model(subvolumes.to(device).permute(0, 2, 3, 1).unsqueeze(dim=1))):
          y, x = test_non_zero_indices[i * BATCH_SIZE + j]
          output[y - RANGE:y + RANGE + 1,
                 x - RANGE:x + RANGE + 1] = value

  return output.cpu()


TRAIN_RECTANGLE = [2000, 400, 2500, 1000]


import matplotlib.pyplot as plt

train_123 = 1
train_pred = generate_predictions(ab_123=train_123,
                                  train_rectangle=TRAIN_RECTANGLE)
mask, label = load_train_mask_label(train_123, train_rectangle=TRAIN_RECTANGLE)
plt.imshow(train_pred, cmap='gray')


BETA_SQUARED = 0.5 * 0.5
SMOOTH = 1e-5

def dice_coef_torch(preds, label):
  preds = np.array(preds)
  label = np.array(label)
  y_true_count = label.sum()
  preds_true_count = preds[label == 1].sum()
  preds_false_count = preds[label == 0].sum()

  c_precision = preds_true_count / (preds_true_count + preds_false_count + SMOOTH)
  c_recall = preds_true_count / (y_true_count + SMOOTH)
  dice = (1 + BETA_SQUARED) * (c_precision * c_recall) / \
    (BETA_SQUARED * c_precision + c_recall + SMOOTH)

  return round(dice, 6)


SAMPLES_TO_TEST = 200
best_dice = 0
best_threshold = 0

for threshold in torch.rand(SAMPLES_TO_TEST):
  binary_pred = train_pred.clone().gt(threshold)
  dice_coef = dice_coef_torch(binary_pred, label)

  if dice_coef > best_dice:
    best_dice = dice_coef
    best_threshold = threshold

binary_pred = train_pred.clone().gt(best_threshold)
print(f'BEST_THRESHOLD: {best_threshold}')
plt.imshow(binary_pred, cmap='gray')
# plt.imshow(label, cmap='gray', alpha=0.5)
plt.show()


plt.imshow(label, cmap='gray', alpha=0.5)
plt.show()


import gc
print(f'GC.COLLECT(): {gc.collect()}')


def run_length_encoding(img, threshold=best_threshold):
    img = np.array(img)

    flat_img = img.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ids = np.where(starts)[0] + 2
    ends_ids = np.where(ends)[0] + 2
    lengths = ends_ids - starts_ids

    return starts_ids, lengths


best_threshold


# best_threshold += best_threshold / 10
best_threshold *= 1.2


pred_list = []

for test_ab in ['a', 'b']:
  test_pred = generate_predictions(ab_123=test_ab)
  plt.imshow(test_pred.gt(best_threshold), cmap='gray')
  plt.show()

  starts_ids, lengths = run_length_encoding(test_pred, threshold=best_threshold.item())
  inklabels_rle = ' '.join(map(str, sum(zip(starts_ids, lengths), ())))
  pred_list.append({'Id': str(test_ab).split('/')[-1], 'Predicted': inklabels_rle})


import pandas as pd

df = pd.DataFrame(pred_list)
df


df.to_csv(f'submission_bt_s_333_{best_threshold:0.4f}_tts_14000_sts_4000_v4.csv', index=False)


df.to_csv('submission.csv', index=False)




