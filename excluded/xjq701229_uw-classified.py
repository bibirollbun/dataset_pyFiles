%pip install albumentations matplotlib more-itertools numba opencv-python polars scikit-learn
%pip install --no-deps lightning segmentation-models-pytorch


import os
import polars
import numpy as np
import cv2
from numba import njit
from numba.typed.typedlist import List
from torch.utils.data import Dataset
import albumentations as A
from copy import deepcopy
import glob
import re


def extract_glob_stars(pattern: str, text: str) -> tuple[str, ...]:
    parts = pattern.split("*")
    escaped = list(map(re.escape, parts))
    regex = "^" + "(.*)".join(escaped) + "$"
    m = re.match(regex, text)
    return m.groups() if m else None


def replace_glob_stars(pattern: str, replacements: list[str]) -> str:
    parts = pattern.split("*")
    assert len(replacements) == len(parts) - 1
    result = []
    for part, rep in zip(parts, replacements):
        result.append(part)
        result.append(rep)
    result.append(parts[-1])
    return "".join(result)


@njit
def _pixels2mask(mask: np.ndarray, pixels: List[int], class_id: int):
    fill_val = class_id
    for i in range(0, len(pixels), 2):
        mask[pixels[i] - 1 : pixels[i] - 1 + pixels[i + 1]] = fill_val


def pixels2mask(pixels: List[int], mask: np.ndarray, class_id: int = 1) -> np.ndarray:
    flattened_mask = mask.reshape(-1, order="C")
    _pixels2mask(flattened_mask, pixels, class_id)
    return flattened_mask.reshape(mask.shape, order="C")


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


def mask2pixels(mask: np.ndarray, class_id: int = 1) -> List[int]:
    return _mask2pixels(mask.flatten("C"), class_id)


class UwDataset(Dataset):
    LABEL2ID = {
        "large_bowel": 1,
        "small_bowel": 2,
        "stomach": 3,
    }

    def __init__(
        self,
        images_pattern: str = "train/case*/case*_day*/scans/slice_*_*_*_*_*.png",
        csv_path: str = "train.csv",
        label2id: dict[str, int] = None,
        augmentation: A.BaseCompose = None,
        z_channel: int = 0,
        z_step: int = 1,
    ):
        assert z_step <= z_channel + 1 and z_channel % z_step == 0
        self.dim25 = z_channel
        self.z_step = z_step
        self.augmentation = augmentation
        self.label2id = label2id if label2id is not None else self.LABEL2ID
        self.images_pattern = images_pattern.replace("\\", "/")
        self.csv_path = csv_path
        self.image_paths = [
            os.path.abspath(i).replace("\\", "/")
            for i in glob.glob(self.images_pattern)
        ]
        self.df = polars.read_csv(self.csv_path)
        self.classify: bool = False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        (_, case, day, slice, w, h, a, b) = extract_glob_stars(
            self.images_pattern, image_path
        )
        image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape
        assert height == int(h) and width == int(w)

        slice_i = int(slice)
        images = []
        for i in range(slice_i - self.dim25, slice_i + self.dim25 + 1, self.z_step):
            path_i = replace_glob_stars(
                self.images_pattern, [case, case, day, str(i).zfill(4), w, h, a, b]
            )
            images.append(
                cv2.imread(path_i, cv2.IMREAD_GRAYSCALE)
                if os.path.exists(path_i)
                else np.zeros_like(image)
            )
        image = np.stack(images, axis=0)

        image_id = f"case{case}_day{day}_slice_{slice}"
        mask: np.ndarray = np.zeros((height, width), dtype=image.dtype)

        for _, class_id, encoded_pixels in self.df.filter(
            polars.col("id") == image_id
        ).iter_rows():
            if encoded_pixels is None:
                continue
            mask = pixels2mask(
                List(map(int, str.split(encoded_pixels))),
                mask,
                class_id=self.label2id[class_id],
            )

        if self.augmentation:
            image = image.transpose(1, 2, 0)
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            image = image.transpose(2, 0, 1)

        if self.classify:
            return image, int(mask.max() > 0)

        return image, mask

    def subset(self, indices: list[int]):
        subset = deepcopy(self)
        subset.image_paths = [self.image_paths[i] for i in indices]
        return subset


def get_train_augmentation(height: int = 384, width: int = 384):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.OneOf(
            [
                A.Compose(
                    [
                        A.PadIfNeeded(min_height=height, min_width=width),
                        A.RandomCrop(height=height, width=width),
                    ]
                ),
                A.Resize(height=height, width=width),
            ],
            p=1,
        ),
    ]
    return A.Compose(test_transform)


def get_validation_augmentation(height: int = 384, width: int = 384):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(height=height, width=width),
    ]
    return A.Compose(test_transform)


import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import torch


class SegModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, t_max, **kwargs):
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

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.std: torch.Tensor
        self.register_buffer("std", torch.tensor(params["std"]).mean())
        self.mean: torch.Tensor
        self.register_buffer("mean", torch.tensor(params["mean"]).mean())

        # Loss function for multi-class segmentation
        self.mode = smp.losses.MULTICLASS_MODE
        self.loss_fn = smp.losses.DiceLoss(self.mode, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image) -> torch.Tensor:
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image: torch.Tensor
        mask: torch.Tensor
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



from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
# import argparse
from tqdm import tqdm


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--arch", type=str, default="FPN", help="arch of seg model")
#     parser.add_argument("--encoder_name", type=str, default="resnet34")
#     parser.add_argument("--batch_size", type=int, default=32, help="batch size")
#     parser.add_argument("--epoch", type=int, default=100, help="max epochs")
#     args = parser.parse_args()
#     arch: str = getattr(args, "arch")
#     encoder_name: str = getattr(args, "encoder_name")
#     batch_size: int = getattr(args, "batch_size")
#     epoch: int = getattr(args, "epoch")
#     return {
#         "arch": arch,
#         "encoder_name": encoder_name,
#         "batch_size": batch_size,
#         "epoch": epoch,
#     }


FOLD = 0
# args = get_args()
batch_size = 32
epoch = 100
arch = "FPN"
encoder_name = "resnet34"

dataset = UwDataset(
    "/kaggle/input/uw-madison-gi-tract-image-segmentation/train/case*/case*_day*/scans/slice_*_*_*_*_*.png",
    "/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv",
    augmentation=get_validation_augmentation(),
)

kf = KFold(n_splits=5, random_state=42, shuffle=True)
folds = list(kf.split(list(range(len(dataset)))))
train_index, valid_index = folds[FOLD]
train_dataset, valid_dataset = (
    dataset.subset(train_index),
    dataset.subset(valid_index),
)
print(len(train_dataset), len(valid_dataset))

train_dataset.classify, valid_dataset.classify = True, True
train_dataset, valid_dataset = (
    train_dataset.subset(
        [
            i
            for i, (_, label) in enumerate(tqdm(train_dataset, desc="train"))
            if label == 1
        ]
    ),
    valid_dataset.subset(
        [
            i
            for i, (_, label) in enumerate(tqdm(valid_dataset, desc="valid"))
            if label == 1
        ]
    ),
)
train_dataset.classify, valid_dataset.classify = False, False
print(len(train_dataset), len(valid_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = SegModel(
    arch=arch,
    encoder_name=encoder_name,
    in_channels=1,
    out_classes=4,
    t_max=epoch * len(train_dataloader),
)

trainer = pl.Trainer(
    max_epochs=epoch,
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




