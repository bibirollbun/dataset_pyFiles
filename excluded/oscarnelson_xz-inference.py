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
    ):
        self.train_images_path = train_images_path
        self.train_csv_path = train_csv_path
        self.image_names = os.listdir(train_images_path)
        self.df = polars.read_csv(train_csv_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image = cv2.imread(
            os.path.join(self.train_images_path, image_name), cv2.IMREAD_GRAYSCALE
        )
        height, width = image.shape
        mask: np.ndarray = np.zeros((height, width), dtype=image.dtype)
        for _, class_id, encoded_pixels in self.df.filter(
            polars.col("ImageId") == image_name
        ).iter_rows():
            mask = pixels2mask(List(map(int, encoded_pixels.split())), mask, class_id)
        return np.expand_dims(image, axis=0), mask


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

dataset = SeverstalDataset(
    train_images_path = "../input/severstal-steel-defect-detection/train_images/",
    train_csv_path = "../input/severstal-steel-defect-detection/train.csv",
)

kf = KFold(n_splits=5, random_state=42, shuffle=True)
folds = list(kf.split(list(range(len(dataset)))))
train_index, valid_index = folds[FOLD]
train_dataset, valid_dataset = Subset(dataset, train_index), Subset(dataset, valid_index)
print(len(train_dataset), len(valid_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


from scipy.stats import mode

TEST_IMG_PATH = "../input/severstal-steel-defect-detection/test_images/"
DEVICE = "cuda"

ckpts = [
    # "/kaggle/input/train-fpn/lightning_logs/version_0/checkpoints/epoch=87-step=27720.ckpt",
    # "/kaggle/input/train-fpn-1/lightning_logs/version_0/checkpoints/epoch=69-step=22050.ckpt",
    # "/kaggle/input/train-fpn-2/lightning_logs/version_0/checkpoints/epoch=76-step=24255.ckpt",
    # "/kaggle/input/train-fpn-3/lightning_logs/version_0/checkpoints/epoch=66-step=21105.ckpt",
    # "/kaggle/input/xz-train/lightning_logs/version_0/checkpoints/epoch=98-step=31185.ckpt",
    # "/kaggle/input/xz-train-origin/lightning_logs/version_0/checkpoints/epoch=58-step=18585.ckpt",
    # "/kaggle/input/xz-train/lightning_logs/version_0/checkpoints/epoch=85-step=27090.ckpt",
    "/kaggle/input/xz-train-nocolor/lightning_logs/version_0/checkpoints/epoch=98-step=31185.ckpt",
]

models = [
    SegModel.load_from_checkpoint(
        checkpoint_path=ckpt,
        map_location=DEVICE,
        arch="FPN",
        encoder_name="resnet34",
        in_channels=1,
        out_classes=5,
        t_max=EPOCHS * len(train_dataloader),
        mean=torch.tensor(dataset.mean, dtype=torch.float),
        std=torch.tensor(dataset.std, dtype=torch.float),
    ).eval()
    for ckpt in ckpts
]


result = []
for img_name in tqdm(os.listdir(TEST_IMG_PATH)):
    img = cv2.imread(os.path.join(TEST_IMG_PATH, img_name), cv2.IMREAD_GRAYSCALE)
    masks = torch.stack(
        [
            model(torch.tensor(img, dtype=torch.float, device=model.device))
            .contiguous()
            .softmax(dim=1)
            .argmax(dim=1)
            .squeeze()
            for model in models
        ]
    )
    mask = torch.mode(masks, dim=0).values.cpu().numpy()

    for class_id in range(1, 5):
        pixels = " ".join(map(str, mask2pixels(mask, class_id)))
        result.append(
            {
                "ImageId_ClassId": f"{img_name}_{class_id}",
                "EncodedPixels": pixels,
            }
        )


import pandas as pd
pd.DataFrame(result).to_csv("submission.csv", index=False)

