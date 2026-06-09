import sys
path = "../input/severstal-packages/"
if path not in sys.path:
    sys.path.append(path)


!mkdir -p /root/.cache/torch/hub/checkpoints/
!cp -r ../input/resnet34-333f7ec4/* /root/.cache/torch/hub/checkpoints/


import os
import polars
from tqdm import tqdm
import numpy as np
import cv2
from numba import njit
from numba.typed.typedlist import List
from torch.utils.data import Dataset

@njit
def _pixels2mask(mask: np.ndarray, pixels: List[int], class_id: int):
    fill_val = class_id
    for i in range(0, len(pixels), 2):
        mask[pixels[i] - 1 : pixels[i] - 1 + pixels[i + 1]] = fill_val


def pixels2mask(pixels: List[int], mask: np.ndarray, class_id: int) -> np.ndarray:
    flattened_mask = mask.reshape(-1, order="F")
    _pixels2mask(flattened_mask, pixels, class_id)
    return flattened_mask.reshape(mask.shape, order="F")


@njit
def _mask2pixels(flattened_mask: np.ndarray, class_id: int) -> List[int]:
    fill_val = class_id
    pixels = List()
    idx = 0
    start_idx = 0
    count = 0
    for i in flattened_mask:
        if i == fill_val:
            if count == 0:
                start_idx = idx
            count += 1
        else:
            if count > 0:
                pixels.append(start_idx + 1)
                pixels.append(count)
                count = 0
        idx += 1
    if count > 0:
        pixels.append(start_idx + 1)
        pixels.append(count)
    return pixels


def mask2pixels(mask: np.ndarray, class_id: int) -> List[int]:
    return _mask2pixels(mask.flatten("F"), class_id)


class SeverstalDataset(Dataset):
    mean = np.array([87.68971919,])
    std = np.array([49.92222673,])

    def __init__(
        self,
        train_images_path: str = "../input/train_images/",
        train_csv_path: str = "../input/train.csv",
        name: str = "train" # train or valid
    ):
        self.train_images_path = train_images_path
        self.train_csv_path = train_csv_path
        self.image_names = os.listdir(train_images_path)
        self.df = polars.read_csv(train_csv_path)
        self.name = name

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image = cv2.imread(
            os.path.join(self.train_images_path, image_name), cv2.IMREAD_GRAYSCALE
        )
        h, w = image.shape
        mask: np.ndarray = np.zeros((h, w), dtype=image.dtype)
        for _, class_id, encoded_pixels in self.df.filter(
            polars.col("ImageId") == image_name
        ).iter_rows():
            mask = pixels2mask(List(map(int, encoded_pixels.split())), mask, class_id)
        # print(image.shape, mask.shape)
        if self.name == "train":
            # 随机翻转
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)  # 水平翻转
                mask = cv2.flip(mask, 1)
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 0)  # 垂直翻转
                mask = cv2.flip(mask, 0)
            # 随机旋转
            angle = np.random.uniform(-10, 10)  # 随机旋转角度
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # 随机平移
            max_shift = 32  # 平移的最大像素
            tx = np.random.randint(-max_shift, max_shift)
            ty = np.random.randint(-max_shift, max_shift)
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, translation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, translation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return np.expand_dims(image, axis=0), mask
        elif self.name == "valid":
            return np.expand_dims(image, axis=0), mask
        else:
            AssertionError()


import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
import torch


class SegModel(pl.LightningModule):
    def __init__(
        self, arch, encoder_name, in_channels, out_classes, t_max, mean, std, **kwargs
    ):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.t_max = t_max
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(std).view(1, in_channels, 1, 1))
        self.register_buffer("mean", torch.tensor(mean).view(1, in_channels, 1, 1))

        # Loss function for multi-class segmentation
        self.mode = smp.losses.MULTICLASS_MODE
        self.loss_fn = smp.losses.DiceLoss(self.mode, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            (pred_mask - 1),
            (mask - 1),
            mode=self.mode,
            num_classes=(self.number_of_classes - 1),
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


BATCH_SIZE = 32
EPOCHS = 100
FOLD = 0

train_dataset = SeverstalDataset(
    train_images_path = "../input/severstal-steel-defect-detection/train_images/",
    train_csv_path = "../input/severstal-steel-defect-detection/train.csv",
    name = "train"
)
valid_dataset = SeverstalDataset(
    train_images_path = "../input/severstal-steel-defect-detection/train_images/",
    train_csv_path = "../input/severstal-steel-defect-detection/train.csv",
    name = "valid"
)
dataset = SeverstalDataset(
    train_images_path = "../input/severstal-steel-defect-detection/train_images/",
    train_csv_path = "../input/severstal-steel-defect-detection/train.csv",
    name = "train"
)

kf = KFold(n_splits=5, random_state=42, shuffle=True)
folds = list(kf.split(list(range(len(train_dataset)))))
train_index, valid_index = folds[FOLD]
train_dataset, valid_dataset = Subset(train_dataset, train_index), Subset(valid_dataset, valid_index)
# print(len(train_dataset), len(valid_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SegModel(
    arch="FPN",
    encoder_name="resnet34",
    in_channels=1,
    out_classes=5,
    t_max=EPOCHS * len(train_dataloader),
    mean=torch.tensor(dataset.mean, dtype=torch.float),
    std=torch.tensor(dataset.std, dtype=torch.float),
)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks=[
        ModelCheckpoint(monitor="valid_dataset_iou", mode="max"),
        EarlyStopping(monitor="valid_dataset_iou", mode="max", patience=20),
    ],
)


trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)




