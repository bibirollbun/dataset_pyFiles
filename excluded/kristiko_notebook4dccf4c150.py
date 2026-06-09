!pip install /kaggle/input/effdet-pkgs2/effdet_pkg/effdet-0.4.1-py3-none-any.whl --no-deps
!pip install /kaggle/input/effdet-pkgs2/effdet_pkg/timm-1.0.22-py3-none-any.whl --no-deps
!pip install /kaggle/input/effdet-pkgs2/effdet_pkg/pycocotools-2.0.10-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps
!pip install /kaggle/input/effdet-pkgs2/effdet_pkg/omegaconf-2.3.0-py3-none-any.whl --no-deps
!pip install /kaggle/input/effdet-pkgs2/effdet_pkg/huggingface_hub-1.2.1-py3-none-any.whl --no-deps



!cp -r ../input/efficientdet .
!cp -r ../input/global-wheat-detection .


!ls efficientdet


!cp -r global-wheat-detection/* .



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset,DataLoader,RandomSampler, SequentialSampler
from effdet.bench import DetBenchTrain
from omegaconf import OmegaConf
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
import cv2
from tqdm import tqdm



df = pd.read_csv('train.csv')

bbox = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).values)
for i, col in enumerate(['x', 'y', 'w', 'h']):
    df[col] = bbox[:, i]
df.drop(columns=['bbox'], inplace=True)
df.head()


# transformation shenanigans here
def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), size=(1024, 1024), p=0.3),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(32, 64), hole_width_range=(32, 64), p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0.1,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df_folds = df[['image_id']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:, 'source'] = df[['image_id', 'source']].groupby('image_id').min()['source']
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number


train_path = 'train'
class DatasetRetriever(Dataset):

    def __init__(self, df, image_ids, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.df = df
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        if self.test or random.random() > 0.5:
            image, boxes = self.load_image_and_boxes(index)
        else:
            image, boxes = self.load_cutmix_image_and_boxes(index)

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'].detach().cpu().numpy() if isinstance(target['boxes'], torch.Tensor) else np.asarray(target['boxes']),
                    'labels': labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
                })
                assert len(sample['bboxes']) == len(sample['labels']), 'not equal!'

                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    #yxyx: be warning
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]
                    target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{train_path}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes


train_dataset = DatasetRetriever(
    df = df,
    image_ids = df_folds[df_folds['fold'] != 0].index.values,
    transforms = get_train_transforms(),
    test = False
)

validation_dataset = DatasetRetriever(
    df = df,
    image_ids = df_folds[df_folds['fold'] == 0].index.values,
    transforms = get_valid_transforms(),
    test = True
)


import warnings
from glob import glob
warnings.filterwarnings("ignore")

class TrainGlobalConfig:
    num_workers = 2
    batch_size = 4
    n_epochs = 20
    lr = 0.0003
    mixed_precision = True
    accumulate = 16
    folder = 'checkpoint'

    verbose = True
    verbose_step = 1

    step_scheduler = False
    validation_scheduler = True

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=2,
        # verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )


class AverageMeter(object):
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


import warnings
from glob import glob
import os
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
import time
import random
warnings.filterwarnings("ignore")

class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5
        self.model = model
        self.device = device
        self.mixed_precision = config.mixed_precision
        self.accumulate = config.accumulate
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.model = model.to(device)
        if self.mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()

        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            # Prepare target dict with required fields
            target_dict = {
                'bbox': boxes,
                'cls': labels,
                'img_scale': torch.ones(batch_size, dtype=torch.float32).to(self.device),
                'img_size': torch.tensor([(512, 512)] * batch_size).to(self.device)
            }

            if self.mixed_precision:
                with autocast():
                    loss_dict = self.model(images, target_dict)
                    loss = loss_dict['loss']
                self.scaler.scale(loss).backward()

                if (step + 1) % self.accumulate == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss_dict = self.model(images, target_dict)
                loss = loss_dict['loss']
                loss.backward()

                if (step + 1) % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()

        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                target_dict = {
                    'bbox': boxes,
                    'cls': labels,
                    'img_scale': torch.ones(batch_size, dtype=torch.float32).to(self.device),
                    'img_size': torch.tensor([(512, 512)] * batch_size).to(self.device)
                }

                loss_dict = self.model(images, target_dict)
                loss = loss_dict['loss']
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def save(self, path):

        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


def get_net():
    config = get_efficientdet_config('tf_efficientdet_d5')
    OmegaConf.set_readonly(config, False)

    config.num_classes = 1
    config.image_size = (512, 512)
    config.num_workers = 4

    net = EfficientDet(config, pretrained_backbone=False)

    try:
        checkpoint = torch.load('efficientdet/efficientdet_d5-ef44aea8.pth')
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(checkpoint)} weights")
    except:
        print("fuckass weights file not found")
        return
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    return DetBenchTrain(net, config)

net = get_net()


def collate_fn(batch):
    return tuple(zip(*batch))

def run_training():
    device = torch.device('cuda:0')
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)


run_training()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512

def get_inference_model(checkpoint_path: str):
    # config = get_efficientdet_config('/kaggle/input/timm-efficientnet-b5/tf_efficientnet_b5-c6949ce9.pth')
    config = get_efficientdet_config('tf_efficientdet_d5')
    OmegaConf.set_readonly(config, False)

    config.num_classes = 1
    config.image_size = (IMG_SIZE, IMG_SIZE)
    config.num_workers = 4

    net = EfficientDet(config, pretrained_backbone=False)
    backbone_ckpt = torch.load('efficientdet/efficientdet_d5-ef44aea8.pth', map_location='cpu')

    new_state_dict = {}
    for k, v in backbone_ckpt.items():
        if k.startswith('class_net') or k.startswith('box_net'):
            continue
        new_state_dict[k] = v

    net.load_state_dict(new_state_dict, strict=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    model = DetBenchPredict(net)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.model.load_state_dict(ckpt['model_state_dict'], strict=False)

    model.to(DEVICE)
    model.eval()
    return model

best_ckpts = sorted(glob('checkpoint/last-checkpoint.bin'))
assert len(best_ckpts) > 0, "Không tìm thấy file last-checkpoint.bin trong checkpoint/"
BEST_CKPT_PATH = best_ckpts[-1]
print("Checkpoint:", BEST_CKPT_PATH)

inference_model = get_inference_model(BEST_CKPT_PATH)


!ls /kaggle/input/timm-efficientnet-b5


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, fname='precision_recall_curve.png'):

    i = torch.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    fp = 1 - tp
    tp_sum = torch.cumsum(tp, dim=0)
    fp_sum = torch.cumsum(fp, dim=0)

    n_gt = len(target_cls)

    recall = tp_sum / (n_gt + 1e-16)
    precision = tp_sum / (tp_sum + fp_sum + 1e-16)

    mrec = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    mpre = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))

    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    i = torch.where(mrec[1:] != mrec[:-1])[0]

    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    best_f1_idx = torch.argmax(f1)

    return precision[best_f1_idx], recall[best_f1_idx], ap, f1[best_f1_idx], n_gt

def run_validation_metrics(model, validation_loader, iou_threshold=0.5, conf_threshold=0.5):
    model.eval()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_detections = []
    all_targets = []

    with torch.no_grad():
        for images, targets, image_ids in tqdm(validation_loader, desc="Validating"):
            images = torch.stack(images)
            images = images.to(DEVICE).float()

            outputs = model(images)

            for i in range(len(outputs)):
                output = outputs[i]
                output = output[output[:, 4] >= conf_threshold]

                if len(output) > 0:

                    boxes = output[:, :4]
                    scores = output[:, 4]
                    classes = output[:, 5].int()

                    for box, score, cls in zip(boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()):
                        all_detections.append({'box': box, 'score': score, 'class': cls, 'image_id': image_ids[i]})
                gt_boxes = targets[i]['boxes']

                gt_boxes_xyxy = gt_boxes[:, [1, 0, 3, 2]].cpu().numpy()
                gt_classes = targets[i]['labels'].cpu().numpy()

                for box, cls in zip(gt_boxes_xyxy, gt_classes):
                    all_targets.append({'box': box, 'class': cls, 'image_id': image_ids[i], 'matched': False})

    all_detections.sort(key=lambda x: x['score'], reverse=True)

    tp_list = torch.zeros(len(all_detections))

    gt_of_class = [t for t in all_targets if t['class'] == 1]

    for i, det in enumerate(all_detections):
        if det['class'] != 1:
            continue

        iou_max = 0
        best_match_gt_idx = -1

        for gt_idx, gt in enumerate(gt_of_class):
            if gt['image_id'] == det['image_id'] and not gt['matched']:
                iou = calculate_iou(det['box'], gt['box'])
                if iou > iou_max:
                    iou_max = iou
                    best_match_gt_idx = gt_idx

        if iou_max >= iou_threshold:
            if not gt_of_class[best_match_gt_idx]['matched']:
                tp_list[i] = 1
                gt_of_class[best_match_gt_idx]['matched'] = True
            else:
                tp_list[i] = 0
        else:
            tp_list[i] = 0

    conf_tensor = torch.tensor([d['score'] for d in all_detections])
    pred_cls_tensor = torch.tensor([d['class'] for d in all_detections])
    target_cls_tensor = torch.tensor([t['class'] for t in all_targets])

    p, r, ap, f1, n_gt = ap_per_class(tp_list, conf_tensor, pred_cls_tensor, target_cls_tensor)

    mAP = ap.item()

    return {
        'Precision': p.item(),
        'Recall': r.item(),
        'F1-score': f1.item(),
        'AP@0.5': ap.item(),
        'mAP@0.5': mAP
    }


VALID_BATCH_SIZE = 4
validation_loader = DataLoader(
    validation_dataset,
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    sampler=SequentialSampler(validation_dataset),
    collate_fn=collate_fn
)

print(f"{len(validation_loader)} batches.")


CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

validation_metrics = run_validation_metrics(
    model=inference_model,
    validation_loader=validation_loader,
    iou_threshold=IOU_THRESHOLD,
    conf_threshold=CONF_THRESHOLD
)

print("\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALID ---")
print(f"Ngưỡng Confidence: {CONF_THRESHOLD}, Ngưỡng IoU: {IOU_THRESHOLD}")
for metric, value in validation_metrics.items():
    print(f"{metric}: {value:.4f}")


IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESHOLD = 0.2

submission_df = pd.DataFrame(columns=["image_id", "PredictionString"])
INPUT_DIR = 'test'
OUTPUT_DIR = 'output'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Đã tạo thư mục output: {OUTPUT_DIR}")

print(f"Bắt đầu inference trên toàn bộ ảnh trong thư mục '{INPUT_DIR}'...")
processed_count = 0
found_detections_count = 0
def get_detections_from_image(model, image_path: str, conf_threshold: float):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    orig_h, orig_w, _ = img_bgr.shape
    scale_h, scale_w = orig_h / IMG_SIZE, orig_w / IMG_SIZE
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    input_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
    detections = output[0].cpu().numpy()
    final_boxes = []
    for det in detections:
        confidence = det[4]
        if confidence >= conf_threshold:
            bbox = det[:4]
            x_min = int(bbox[0] * scale_w)
            y_min = int(bbox[1] * scale_h)
            x_max = int(bbox[2] * scale_w)
            y_max = int(bbox[3] * scale_h)
            final_boxes.append([x_min, y_min, x_max, y_max, confidence, int(det[5])])
    return img_bgr, final_boxes

def draw_boxes(image, boxes, class_names=None):
    draw_img = image.copy()
    COLOR = (0, 255, 0)
    THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes:
        x_min, y_min, x_max, y_max, confidence, class_id = box
        cv2.rectangle(draw_img, (x_min, y_min), (x_max, y_max), COLOR, THICKNESS)
        label = f"Class {class_id}: {confidence:.2f}"
        y_text = y_min - 10 if y_min - 10 > 10 else y_min + 10
        cv2.putText(draw_img, label, (x_min, y_text), FONT, 0.7, COLOR, 2)
    return draw_img

try:
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]

    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong thư mục '{INPUT_DIR}'.")

    for filename in tqdm(image_files, desc="Processing Images"):
        input_image_path = os.path.join(INPUT_DIR, filename)
        output_image_path = os.path.join(OUTPUT_DIR, filename.replace('.', '_output.', 1))

        try:
            original_image, detections = get_detections_from_image(
                inference_model,
                input_image_path,
                CONF_THRESHOLD
            )

            processed_count += 1

            if detections:
                output_image = draw_boxes(original_image, detections)
                cv2.imwrite(output_image_path, output_image)
                found_detections_count += 1
            else:
                cv2.imwrite(output_image_path, original_image)
            pred_string_list = []
            for x_min, y_min, x_max, y_max, conf, cls_id in detections:
                width = x_max - x_min
                height = y_max - y_min
                pred_string_list.append(f"{conf:.2f} {x_min} {y_min} {width} {height}")

            pred_string = " ".join(pred_string_list)

            # Append to submission DataFrame
            submission_df = pd.concat([submission_df, pd.DataFrame({
                "image_id": [os.path.splitext(filename)[0]],  # remove extension
                "PredictionString": [pred_string]
            })], ignore_index=True)
        except FileNotFoundError as e:
            print(f"\nLỗi: {e}")
        except Exception as e:
            print(f"\nLỗi xử lý ảnh {filename}: {e}")

    print("\n--- KẾT QUẢ INFERENCE BATCH ---")
    print(f"Tổng số ảnh đã xử lý: {processed_count}")
    print(f"Số ảnh có detections (Conf > {CONF_THRESHOLD}): {found_detections_count}")
    print(f"Đã lưu kết quả vào thư mục '{OUTPUT_DIR}'.")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy thư mục đầu vào '{INPUT_DIR}'. Vui lòng kiểm tra lại đường dẫn.")
except Exception as e:
    print(f"Lỗi tổng quát trong quá trình xử lý batch: {e}")


submission_df.to_csv("submission.csv", index=False)


