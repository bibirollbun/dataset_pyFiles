from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import models, datasets, io, tv_tensors
from torchvision.transforms import v2 as T

import numpy as np


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
device


@dataclass
class DatasetPaths:
    root: Path

    def __post_init__(self):
        self.train = self.root / "train"
        self.test = self.root / "test"


paths = DatasetPaths(Path("/kaggle/input/spot-the-turtles/root"))



## DO NOT CHANGE
image_height = 544
image_width = 704
## ------------

val_percentage = 0.003

batch_size = 1

train_transforms = T.Compose(
    [
        T.Resize((image_height, image_width)),
        T.ToDtype(torch.float, scale=True),
    ]
)

val_transforms = T.Compose(
    [
        T.Resize((image_height, image_width)),
        T.ToDtype(torch.float, scale=True),
    ]
)


class CocoDetectionV2(datasets.VisionDataset):
    """
    root
    |- coco.json
    |- images/*
    """

    def __init__(
        self,
        root: Path,
        transforms: Callable,
    ):
        super().__init__(Path(root), transforms=transforms)

        from pycocotools.coco import COCO

        self.coco = COCO(self.root / "coco.json")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.K = len(self.coco.getCatIds())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError(f"Index must be of type integer, got {type(idx)} instead.")

        image_id = self.ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]

        image = io.decode_image(self.root / "images" / image_info["file_name"])

        anns = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        mask = np.zeros(
            (
                self.K + 1,
                image_info["height"],
                image_info["width"],
            ),
            dtype=np.uint8,
        )
        for ann in anns:
            if "segmentation" in ann and ann["segmentation"]:
                channel_index = ann["category_id"]
                object_mask = self.coco.annToMask(ann)
                mask[channel_index] = np.logical_or(
                    mask[channel_index], object_mask
                ).astype(mask.dtype)
        mask[0] = ~np.any(mask[1:], axis=0)
        mask = torch.from_numpy(mask).to(torch.float32)
        mask = tv_tensors.Mask(mask)

        image, mask = self.transforms(image, mask)

        return image, mask, image_info["file_name"]


def prepare_train_val_test(
    paths,
    train_transforms,
    val_transforms,
    val_percentage=0.2,
    batch_size=16,
    num_workers=0,
):
    train_dataset = CocoDetectionV2(
        root=paths.train,
        transforms=train_transforms,
    )
    val_dataset = CocoDetectionV2(
        root=paths.train,
        transforms=val_transforms,
    )
    test_dataset = CocoDetectionV2(
        root=paths.test,
        transforms=val_transforms,
    )

    split = [1.0 - val_percentage, val_percentage]
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(train_dataset, split, generator)
    _, val_dataset = random_split(val_dataset, split, generator)

    train = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train, val, test


train, val, test = prepare_train_val_test(
    paths=paths,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    val_percentage=val_percentage,
    batch_size=batch_size,
)


def visualise_with_masks(image, mask, pred_mask=None):
    import matplotlib
    import matplotlib.pyplot as plt

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0.0, 1.0)
    if isinstance(mask, torch.Tensor):
        mask = np.argmax(mask.permute(1, 2, 0).numpy(), axis=2)
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = np.argmax(pred_mask.permute(1, 2, 0).numpy(), axis=2)

    n_plots = 3

    plt.figure(figsize=(16, 16))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    plt.subplot(1, n_plots, 1)
    plt.imshow(image, cmap="plasma")
    plt.axis("off")
    plt.title(f"Original Image {image.shape}")

    plt.subplot(1, n_plots, 2)
    plt.imshow(mask, cmap="plasma", norm=norm)
    plt.axis("off")
    plt.title("Mask")

    if isinstance(pred_mask, np.ndarray):
        plt.subplot(1, n_plots, 3)
        plt.imshow(pred_mask, cmap="plasma", norm=norm)
        plt.axis("off")
        plt.title("Predicted Mask")

    plt.tight_layout()
    plt.show()


images, masks, filenames = next(iter(train))
for image, mask, filename in zip(images, masks, filenames):
    print(filename)
    visualise_with_masks(image, mask)


## DO NOT CHANGE
in_channels = 3
classes = 2
## ------------


## Start here ;)











model = ...












# generate_submission(model, test)


def generate_sample_submission(val):
    import pandas as pd
    from pycocotools import mask as mask_util

    all_predictions = []

    for images, masks, filenames in val:
        predicted_masks = masks.detach().cpu().to(torch.uint8)

        for img, mask, filename in zip(images, predicted_masks, filenames):
            foreground_mask = mask[1].numpy().astype(np.uint8)

            # visualise_with_masks(img, foreground_mask)

            rle = mask_util.encode(np.asfortranarray(foreground_mask))
            assert rle["size"][0] == image_height
            assert rle["size"][1] == image_width

            rle_formatted = {
                "counts": rle["counts"].decode("utf-8")
                if isinstance(rle["counts"], bytes)
                else rle["counts"],
                "size": tuple(rle["size"]),
            }
            all_predictions.append(
                {
                    "id": filename,
                    "rle": rle_formatted,
                }
            )

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(f"sample_submission_{image_height}x{image_width}.csv", index=False)

generate_sample_submission(val)

