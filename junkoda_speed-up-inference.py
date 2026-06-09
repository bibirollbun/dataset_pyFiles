import numpy as np
import pandas as pd
import time
import argparse
from pathlib import Path

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, default_collate
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TTF
import timm
import yaml

import multiprocessing as mp
from queue import Empty


INPUT_PATH = Path('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025')
MODEL_PATH = Path('/kaggle/input/bacterial-public/weights/object/baseline')


def get_tomos(input_path: Path,
              data_type: str,
              *, n=None) -> list[Path]:
    """
    tomo..dachi...?
    Args
      input_path (Path): Kaggle input directory
      data_type (str):   train or test
      n (Optional[int]): Use only first n train tomos (not applicable to test)
    """
    data_path = input_path / data_type
    tomo_paths = sorted(data_path.glob('*'))

    if (n is not None) and (data_type == 'train'):
        tomo_paths = tomo_paths[:n]
    
    return tomo_paths
    

def preprocess(img: torch.Tensor) -> torch.Tensor:
    """
    Resize and normalize

    Arg:
      img (Tensor[uint8]):   (batch_size, C, H, W)

    Returns:
      img (Tensor[float32]): (batch_size, C, size, size); size = 640
    """
    size = (640, 640)

    img = img.to(dtype=torch.float32)
    img = TTF.resize(img, size)  # (batch_size, C, size, size)

    batch_size, nch, h, w = img.shape
    q = torch.Tensor([0.05, 0.95]).to(img.device)
    x_min, x_max = torch.quantile(img.view(batch_size, nch * h * w), q, dim=1)
    x_min = x_min.view(batch_size, 1, 1, 1)
    x_max = x_max.view(batch_size, 1, 1, 1)

    img = (img - x_min) / (x_max - x_min)
    img = torch.clamp(img, 0, 1)

    return img


class Dataset(torch.utils.data.Dataset):
    """
    dataset = Dataset(tomo_path)
    
    Args:
      tomo_path (Path): directory including jpg images    
    """
    def __init__(self, tomo_path: Path):
        self.filenames = sorted(tomo_path.glob('*'))  # list[Path]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> dict:
        filename = self.filenames[i]  # Path
        filebase = filename.stem
        assert filebase[:6] == 'slice_'
        slice_number = int(filebase[6:])  # slice_0000 -> int(0000)

        # Load, resize and normalize image
        img = Image.open(filename)
        W, H = img.size
        img = np.expand_dims(np.array(img), axis=0)  # array[uint8] (1, H, W)

        ret = {'img': img,
               'slice_number': slice_number,
               'shape': np.array((H, W), dtype=int),  # original shape H, W
        }

        return ret

    def loader(self, batch_size: int, num_workers: int):
        loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers)
        return loader



import torch
import torch.nn as nn
import timm


class Model(nn.Module):
    def __init__(self, cfg_model: dict, *, pretrained=True, verbose=True):
        super().__init__()

        # Timm encoder
        name = cfg_model['encoder']
        in_channels = 1
        out_channels = 1

        self.encoder = timm.create_model(name,
                                         in_chans=in_channels,
                                         features_only=True,
                                         pretrained=pretrained)
        encoder_channels = self.encoder.feature_info.channels()

        self.segmentation_head = nn.Conv2d(encoder_channels[-1], out_channels,
                                           kernel_size=3, padding=1)

        self.regression_head = nn.Conv2d(encoder_channels[-1], out_channels=2,
                                         kernel_size=3, padding=1)

        self.criterion_seg = nn.BCEWithLogitsLoss()
        self.criterion_reg = nn.MSELoss()

        if verbose:
            print(name)

    def forward(self, img: torch.Tensor):
        """
        img (Tensor): (batch_size, 1, H, W)

        h, w = H // 32, W // 32
        """
        features = self.encoder(img)
        out = features[-1]                    # (batch_size, embed_dim, h, w)
        y_pred = self.segmentation_head(out)  # (batch_size, 1, h, w)
        t_pred = self.regression_head(out)    # (batch_size, 2, h, w)

        return y_pred, t_pred



def predict(tomo_path: Path,
            models: list[nn.Module],
            cfg: dict) -> dict:
    """
    Predict moter coordinate for one tomo_id
    At most one moter per tomo_id in test
    
    Args:
      model (nn.Module): Pytorch model
      dataset (Dataset): for one tomo_id
    """
    assert len(models) > 0
    tomo_id = tomo_path.name
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    use_amp = cfg['use_amp']
    preprocess_device = cfg['preprocess_device']
    assert preprocess_device in ['cuda', 'cpu']

    dataset = Dataset(tomo_path)
    loader = dataset.loader(batch_size=batch_size, num_workers=num_workers)
    
    device = next(models[0].parameters()).device

    K = len(cfg['tta'])   # 4 for 4 rot90, 1 for no TTA
    tta = TTA(cfg['tta'], device=device)

    # Loop over all slices in one tomo_id
    best = (0, None)
    for d in loader:
        # Input image (batch_size, 1, H, W)
        if preprocess_device == 'cuda':
            img = d['img'].to(device) 
            img = preprocess(img)
        elif preprocess_device == 'cpu':
            img = preprocess(d['img'])
            img = img.to(device)  # input image (batch_size, 1, H, W)
        else:
            raise ValueError(preprocess_device)

        # TTA: batch_size -> K * batch_size
        img = tta.expand(img)   # (K * batch_size, 1, H, W)
        
        y_pred_sum, t_pred_sum = None, None
        for model in models:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda',
                                        enabled=use_amp,
                                        dtype=torch.float16):
                    y_pred, t_pred = model(img) 

            y_pred = y_pred.sigmoid()  # (batch_size, 1, h, w)

            if y_pred_sum is None:
                y_pred_sum = y_pred
                t_pred_sum = t_pred
            else:
                y_pred_sum += y_pred
                t_pred_sum += t_pred

        # TTA: Reduce K * batch_size -> batch_size
        y_pred_sum, t_pred_sum = tta.reduce(y_pred_sum, t_pred_sum)  # (batch_size, 1, h, w)  
        
        y_pred_max = y_pred_sum.max().item() / len(models)
        del y_pred, t_pred

        # Keep most probable coordinate
        if y_pred_max > best[0]:
            bs, _, h, w = y_pred_sum.shape
        
            argmax = torch.unravel_index(y_pred_sum.argmax(), y_pred_sum.shape)  # b, ch, iy, ix
            i, _, iy, ix = [t.item() for t in argmax]
            slice_number = d['slice_number'][i].item()
            offset = t_pred_sum[i, :, iy, ix].cpu().numpy() / len(models)  # (2, )

            # Compute coodinate in original pixels
            H, W = d['shape'][i].numpy()    # Original image size
            x = (ix + offset[0]) * (W / w)
            y = (iy + offset[1]) * (H / h)
            
            best = (y_pred_max, slice_number, y, x)

    assert best[1] is not None

    # Return prediction
    n_slices = len(dataset.filenames)
    pred = {'tomo_id': tomo_id,
            'n_slices': n_slices,
            'y_pred': best[0],
            'zyx': best[1:]}
    return pred



def create_rot_matrices(ks, inverse=False) -> torch.Tensor:
    """
    Create rot90 rotation matrices

    Arg:
      ks (list[int]): list of rot90 indices k=0,1,2,3
      inverse (bool): If True kth rotation is rot90(-k)

    Returns:
      rots (Tensor[float32]): (K, 1, 2, 2) where K = len(ks)
    """
    # Create four rot90 matrices
    a = np.array([(1, 0), (0, 1)], dtype=np.float32)         # identity
    if inverse:
        r90 = np.array([(0, 1), (-1, 0)], dtype=np.float32)  # rotate -90 degrees
    else:
        r90 = np.array([(0, -1), (1, 0)], dtype=np.float32)  # rotate  90 degrees

    rots = [a, ]
    for k in range(4):
        a = a @ r90
        rots.append(a)

    rots = [rots[k] for k in ks]

    rots = np.stack(rots, axis=0)        # (K, 2, 2)
    rots = np.expand_dims(rots, axis=1)  # (K, 1, 2, 2)
    rots = torch.from_numpy(rots)

    return rots


class TTA:
    """
    tta = TTA(ks, dtype, device)

    Args:
      ks (list[int]): list of rot90 indices k=0,1,2,3
      dtype (torch.dtype): torch.float16 if
    """
    def __init__(self,
                 ks: list[int],
                 device: torch.device):
        self.ks, self.no_tta = self._check_ks(ks)
        self.rots = create_rot_matrices(ks, inverse=True).to(device=device)  # (K, 1, 2, 2)
        self.no_tta = (list(ks) == [0, ])

    @staticmethod
    def _check_ks(ks):
        for k in ks:
            assert 0 <= k < 4
        no_tta = (list(ks) == [0, ])

        return ks, no_tta

    def __repr__(self):
        return 'TTA(%s)' % ','.join(['%d' % k for k in self.ks])

    def expand(self, img: torch.Tensor) -> torch.Tensor:
        """
        Expand input image to K rot90 images

        Args:
          img (Tensor): (B, C, H, W)

        Returns:
          img_k (Tensor): (K * B, C, H, W)
        """
        if self.no_tta:
            return img

        stack = []
        for k in self.ks:
            img_rot = torch.rot90(img, k, dims=[2, 3])
            stack.append(img_rot)

        return torch.cat(stack, dim=0)

    def reduce(self,
               img_k: torch.Tensor,
               offset_k: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Average k patterns of rot90 predictions

        Args:
          img_k    (Tensor): (K * B, C, H, W)
          offset_k (Tensor): (K * B, 2, H, W)

        Returns: img, offset
          img     (Tensor): (B, C, H, W)
          offset  (Tensor): (B, C, H, W)
        """
        if self.no_tta:
            return img_k, offset_k

        img = self.rotate_back_image(img_k, average=True)
        offset = self.rotate_back_offset(offset_k, average=True)

        return img, offset
   
    def rotate_back_image(self, img_k: torch.Tensor, *, average=True) -> torch.Tensor:
        """
        Rotate back K TTA images (scaler fields)

        Arg:
          img_k (Tensor): (K * B, C, H, W)
        """
        K = len(self.ks)
        BK, C, H, W = img_k.shape
        assert BK % K == 0
        B = BK // K

        img_k = img_k.view(K, B, C, H, W)

        # Rotate back
        if average:
            # Reduce K TTA patterns by average
            avg = torch.zeros((B, C, H, W), dtype=torch.float32, device=img_k.device)
            for i, k in enumerate(self.ks):
                avg += torch.rot90(img_k[i], 4 - k, dims=(2, 3))
            avg /= K  # (B, C, H, W)
        else:
            out = []
            for i, k in enumerate(self.ks):
                out.append(torch.rot90(img_k[i], 4 - k, dims=(2, 3)))
            return torch.cat(out, dim=0)  # (BK, C, H, W)

        return avg

    def rotate_back_offset(self,
                           offset_k: torch.Tensor, *,
                           average=True) -> torch.Tensor:
        """
        Rotate back K TTA offsets (vector fields)

        Arg:
          offset (Tensor): (4 * B, 2, h, w)

          offset[k] will be rotated by rot90(, k)
          offset[:, :, 0] is dx and offset[:, :, 1] is dy from pixel cell top-left in [0, 1]
        """
        K = len(self.ks)
        BK, ndim, H, W = offset_k.shape
        assert BK % K == 0
        assert ndim == 2
        B = BK // K

        offset = offset_k.view(K, B, 2, H, W)
        dtype = offset.dtype

        # Expand rotation matrix to B batch
        a = self.rots.expand(K, B, 2, 2).reshape(K * B, 2, 2).to(dtype=dtype)

        # Offset vectors from pixel cell center
        offset = (offset - 0.5).permute(0, 1, 3, 4, 2).reshape(K * B, H * W, 2)
        
        # Apply rotation matrix
        offset_rot = torch.bmm(offset, a)
        offset_rot = offset_rot.reshape(K, B, H, W, 2).permute(0, 1, 4, 2, 3)

        # Back to offset from pixel cell top-left corner
        offset_rot += 0.5  # (4, B, 2, H, W)

        # Rotate pixels
        if average:
            # Reduce K TTA patterns by average
            avg = torch.zeros((B, 2, H, W), dtype=torch.float32, device=offset.device)
            for i, k in enumerate(self.ks):
                avg += torch.rot90(offset_rot[i], 4 - k, dims=(2, 3))
            avg /= K  # (B, 2, H, W)
        else:
            out = []
            for i, k in enumerate(self.ks):
                out.append(torch.rot90(offset_rot[i], 4 - k, dims=(2, 3)))
            return torch.cat(out, dim=0)  # (BK, 2, H, W)
        
        return avg



def create_submission(preds: list, th: float, ofilename: str) -> pd.DataFrame:
    """
    Args:
      preds (list[dict]): predictions
      th (float): threshold between no or one moter
      ofilename (str): submission.csv
    """
    rows = []
    count_positive = 0
    for pred in preds:
        if pred['y_pred'] < th:
            zyx = (-1, -1, -1)
        else:
            count_positive += 1
            zyx = pred['zyx']

        row = {'tomo_id': pred['tomo_id'],
               'Motor axis 0': zyx[0],
               'Motor axis 1': zyx[1],
               'Motor axis 2': zyx[2]}
        rows.append(row)

    submit = pd.DataFrame(rows)
    submit.to_csv(ofilename, float_format='%.8e', index=False)

    print('Submit %s: %d positives / %d tomo_ids' % (ofilename, count_positive, len(rows)))

    return submit


def process_fn(process_id: int,
               tomo_queue,
               pred_queue,
               cfg: dict):
    """
    Prediction process for each GPU

    Args:
      process_id (int): 0 or 1 for 2 GPUs
      tomo_queue (): tomo_ids
      pred_queue (): output predictions
      cfg (dict): config
    """
    device = torch.device('cuda:%d' % process_id)

    model_path = cfg['model_path']
    folds = cfg['folds']

    # Load models
    with open(model_path / 'config.yml', 'r') as f:
        cfg_model = yaml.safe_load(f)
            
    models = []
    for ifold in folds:
        model_filename = '%s/model%d.pytorch' % (model_path, ifold)
        model = Model(cfg_model['model'], pretrained=False, verbose=False)
        model.load_state_dict(torch.load(model_filename, weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)

        if process_id == 0:
            print('Load model', model_filename)
        
    # Loop over tomograms
    while not tomo_queue.empty():
        try:
            tomo_path = tomo_queue.get(timeout=1)
            pred = predict(tomo_path, models, cfg)
            pred_queue.put(pred)

        except Empty:
            break
            

#
# Main
#
tb = time.time()

# Config
cfg = {
    'model_path': MODEL_PATH,
    'folds': [0, ],          # [0,1,2,3,4] for all 5-fold mean
    'batch_size': 16,
    'num_workers': 1,
    'use_amp': True,
    'preprocess_device': 'cuda',
    'tta': [0,1,2,3],              # rot90 ks, e.g., [0,] (no TTA) or [0,1,2,3] (all 4 rot90s)
}


#
# List of tomograms
#
tomo_paths = get_tomos(INPUT_PATH, 'test')

if len(tomo_paths) == 3:
    # Some experiment when test is dummy (optional)
    # tomo_paths = get_tomos(INPUT_PATH, 'train', n=20)
    pass

print('Data %d' % len(tomo_paths))
print('TTA', cfg['tta'])

manager = mp.Manager()
tomo_queue = manager.Queue()
pred_queue = manager.Queue()
for tomo_path in tomo_paths:
    tomo_queue.put(tomo_path)

time.sleep(1)
assert not tomo_queue.empty()


#
# Launch process
#
num_processes = 2
tb = time.time()

workers = [mp.Process(target=process_fn,
                      args=(i, tomo_queue, pred_queue, cfg))
           for i in range(num_processes)]

for w in workers:
    w.start()

for w in workers:
    w.join()


dt = time.time() - tb
print('%.2f sec for %d tomos' % (dt, len(tomo_paths)))

# queue to list
preds = []
try:
    while not pred_queue.empty():
        preds.append(pred_queue.get(timeout=1))
except Empty:
    pass

assert len(preds) == len(tomo_paths)


#
# Submit
#
th = 0.5
ofilename = 'submission.csv'
create_submission(preds, th, 'submission.csv')
print(ofilename, 'written')



! head -n 5 submission.csv
! wc -l submission.csv

