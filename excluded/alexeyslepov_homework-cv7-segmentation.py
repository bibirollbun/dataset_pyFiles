import os
import numpy as np
import pandas as pd
import random
import colorsys
from tqdm.auto import tqdm

import cv2
import albumentations as A

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from sklearn.model_selection import train_test_split


!ls /kaggle/input/sartorius-cell-instance-segmentation


ROOT = "/kaggle/input/sartorius-cell-instance-segmentation"


train_df = pd.read_csv(ROOT + "/train.csv")
display(train_df.head(2))
print(f'Train data shape: {train_df.shape}')


cls_map = {value:idx for idx,value in enumerate(train_df["cell_type"].unique())}
cls_map


def get_targets_mask(df, img_id):
    """
    Function to get target masks
    
    rles contains mask's description
    """
    targets = df[df["id"] == img_id]["cell_type"].apply(lambda x: cls_map[x]).values 
    rles = df[df["id"] == img_id]["annotation"].values
    
    return targets, rles


img_id = train_df.iloc[0]["id"]
img = cv2.imread(f'{ROOT}/train/{img_id}.png')
plt.imshow(img)
labels, rles = get_targets_mask(train_df, img_id)
len(rles)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    
    return colors


def decode_rle_mask(rle_mask, shape=(520, 704)):

    """
    Decode run-length encoded segmentation mask string into 2d array

    Parameters
    ----------
    rle_mask (str): Run-length encoded segmentation mask string
    rle_mask - it is a string with start and length coordinate values
    
    shape (tuple): Height and width of the mask

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: Decoded 2d segmentation mask
    """

    rle_mask = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle_mask[0:][::2], rle_mask[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
    
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    mask = mask.reshape(shape[0], shape[1])
    # Transform mask value (necessary step before Image augmentation)
    mask = np.uint8(mask)
    
    return mask


def encode_rle_mask(mask, shape=(520, 704)):
    """
    (Used for create submission file)
    Parameters
    ----------
    mask with a given shape
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 1
    rle[1::2] -= rle[::2]
    
    return rle.tolist()


masks = []

for mask in train_df.loc[train_df['id'] == img_id, 'annotation'].values:
    decoded_mask = decode_rle_mask(rle_mask=mask, shape=img.shape)
    
    masks.append(decoded_mask)

len(masks)


masks_stack = np.stack(masks)
masks_stack.shape


def get_bboxes_from_mask(masks):
    """
    Function to get bboxes in coco format
    """
    coco_boxes = []
    
    for mask in masks:
        pos = np.nonzero(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        coco_boxes.append([xmin, ymin, xmax, ymax])
    
    # Преобразуем в формат np для ДатаЛоадер  
    coco_boxes = np.asarray(coco_boxes)
    
    return coco_boxes

bboxes = get_bboxes_from_mask(masks_stack)
bboxes


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    
    return image



def plot_image_annotations(image, masks, bboxes, labels, aug=None):
    """
    Function to plot masks and bboxes with augmentation
    """
    image = image.copy()
    
    colors = {unique_lbl:random_colors(1, True)[0] for unique_lbl in np.unique(labels)}
    
    if aug is not None:
        augmented = aug(image=image, masks=masks, bboxes=bboxes,
                        labels=labels)
        
        image = augmented['image']
        masks = augmented['masks']
        bboxes = augmented['bboxes']
    
    bboxes = np.stack(bboxes).astype(int)
        
    for idx, box in enumerate(bboxes):
        color = tuple([int(value*255) for value in colors[labels[idx]]])
        image = cv2.rectangle(image, (box[2], box[3]), (box[0], box[1]), color=color, thickness=2)
    
    for idx, mask in enumerate(masks):
        color = colors[labels[idx]]
        image = apply_mask(image, mask, color)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.show()


plot_image_annotations(image=img, masks=masks_stack, bboxes=bboxes, labels=labels)


train_augmentations = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90()
], bbox_params={"format": "pascal_voc",
                "min_area": 0,
                "min_visibility": 0,
                'label_fields': ['labels']
               })


plot_image_annotations(image=img, masks=masks_stack, bboxes=bboxes, labels=labels, aug=train_augmentations)


class CellSegData(torch.utils.data.Dataset):
    
    def __init__(self, root, df, split='train', aug=None, cls_map=None):
        self.augmentations = aug
        self.cls_map = cls_map
        
        train, test = train_test_split(df['id'].unique(), train_size=0.9)
        
        if split=='train':
            self.dataset = train
        else:
            self.dataset = test
        
        
        self.dict_df = {img_id:df[df['id']==img_id] for img_id in tqdm(self.dataset)}
        self.root = root
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_id = self.dataset[index]
        image = cv2.imread(os.path.join(self.root, img_id+'.png'))
        
        info = self.dict_df[img_id]
        n_objects = len(info['annotation'])
                
        
        labels = info["cell_type"].apply(lambda x: self.cls_map[x]).values
        rles = info["annotation"].values
        
        masks = []
        for mask in rles:
            decoded_mask = decode_rle_mask(rle_mask=mask, shape=img.shape)
            masks.append(decoded_mask)
            
        bboxes = get_bboxes_from_mask(masks)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image, masks=masks, bboxes=bboxes,
                            labels=labels)
            image = augmented['image']
            masks = augmented['masks']
            bboxes = augmented['bboxes']
            bboxes = np.stack(bboxes).astype(int)
        
        # Transform to np
        masks = np.asarray(masks)
        
        bboxes = torch.as_tensor(bboxes, dtype=torch.int64)


        is_bad_labels = False
        degenerate_boxes = bboxes[:, 2:] <= bboxes[:, :2]
        
        if degenerate_boxes.any():
            is_bad_labels = True
            # print the first degenerate box
            bb_idxs = torch.where(degenerate_boxes.any(dim=1))[0].numpy()

        labels = torch.as_tensor([1 for i in range(len(bboxes))], dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        if is_bad_labels:
            bboxes = bboxes[[i for i in range(len(bboxes)) if i not in bb_idxs]]
            labels = labels[[i for i in range(len(bboxes)) if i not in bb_idxs]]
            masks = masks[[i for i in range(len(bboxes)) if i not in bb_idxs]]

        image_id = torch.tensor([index])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': bboxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        image = image.transpose((2, 0, 1))
        
        return torch.Tensor(image), target #torch.Tensor(image)/255, target


dataset = CellSegData(f'{ROOT}/train', train_df, 'train', train_augmentations, cls_map)
dataset_test = CellSegData(f'{ROOT}/train', train_df, 'test', train_augmentations, cls_map)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
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
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
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
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


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


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


!pip install pycocotools


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=2, collate_fn=collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn) 


!mkdir -p '/root/.cache/torch/hub/checkpoints'
!cp ../input/maskrcnn_resnet50_fpn/pytorch/default/1/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth


NUM_CLASSES = 2

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, box_detections_per_img=600)

# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
# weights = torch.load('weights/checkpoint.pth')
# move model to the right device
# model.state_dict(weights['model'])
model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

output_dir = 'weights'
os.makedirs(output_dir, exist_ok=True)


def analyze_train_sample(model, ds_train, sample_index):
    img, targets = ds_train[sample_index]
    
    plt.imshow(img.numpy().astype(np.uint8).transpose((1, 2, 0)))
    plt.title("Image")
    plt.show()

    masks = np.zeros((640, 640))
    for mask in targets['masks']:
        masks = np.logical_or(masks, mask)
        
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.imshow(masks, alpha=0.3)
    plt.title("Ground truth")
    plt.show()

    model.eval()
    
    with torch.no_grad():
        preds = model([img.cuda()])[0]

    plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
    all_preds_masks = np.zeros((640, 640))
    for mask in preds['masks'].cpu().detach().numpy():
        
        all_preds_masks = np.logical_or(all_preds_masks, mask[0] > 0.5)
        
    plt.imshow(all_preds_masks, alpha=0.4)
    plt.title("Predictions")
    plt.show()


num_epochs = 10

for epoch in tqdm(range(num_epochs)):
    # train for one epoch, printing every 10 iterations
    model.train()
        
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
        
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    analyze_train_sample(model, dataset, 20)
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)
    if output_dir:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        save_on_master(checkpoint, os.path.join(output_dir, f"model_{epoch}.pth"))
        save_on_master(checkpoint, os.path.join(output_dir, "checkpoint.pth"))


class CellTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.image_ids = [f[:-4]for f in os.listdir(self.image_dir)]
        
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + ".png")
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image, _ = self.transforms(image=image, target=None)
            
        return {'image': image, 'image_id': image_id}

    
    def __len__(self):
        return len(self.image_ids)


RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    

class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, RESNET_MEAN, RESNET_STD)
        return image, target
    
    
class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        
        return image, target
    
    
def get_transform(train):
    transforms = [ToTensor()]
    if True:
        transforms.append(Normalize())

    return Compose(transforms)


ds_test = CellTestDataset(f'{ROOT}/test', transforms=get_transform(train=False))
ds_test[2]


def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
        
    return ' '.join(map(str, run_lengths))


def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask


model.eval();

submission = []

for sample in ds_test:
    img = sample['image']
    image_id = sample['image_id']
    
    with torch.no_grad():
        result = model([img.to(device)])[0]
    
    previous_masks = []
    
    for i, mask in enumerate(result["masks"]):
        # Filter-out low-scoring results. Not tried yet.
        score = result["scores"][i].cpu().item()
        if score < 0.3:
            continue
        
        mask = mask.cpu().numpy()
        # Keep only highly likely pixels
        binary_mask = mask > 0.5
        binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
        previous_masks.append(binary_mask)
        rle = rle_encoding(binary_mask)
        
    submission.append((image_id, rle))
    print(submission)
    
    plt.figure(figsize=(12,12))
    ax1 = plt.subplot(121)
    ax1.imshow(img.numpy().transpose((1,2,0)))
    all_preds_masks = np.zeros((520, 704))
    
    for mask in result['masks'].cpu().detach().numpy():
        all_preds_masks = np.logical_or(all_preds_masks, mask[0] > 0.5)
        
    ax2 = plt.subplot(122)
    ax2.imshow(img.numpy().transpose((1,2,0)))
    ax2.imshow(all_preds_masks, alpha=0.3)
    plt.tight_layout()
    plt.show()


df_sub = pd.DataFrame(submission, columns=['id', 'predicted'])
df_sub.to_csv("submission.csv", index=False)
df_sub.head()


!ls /kaggle/working

