# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


! pip install lightning mlflow


import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import h5py
import lightning as L
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import torchvision.transforms as transforms
from PIL import Image
import random
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
import mlflow
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

from transformers import ViTImageProcessor, ViTModel, ViTConfig
from transformers.utils import cached_file

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from datetime import datetime

import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping


class ImageClsDataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        augment: bool = False,
        indices: np.ndarray = None,
        ds_total_len: float = 2,
        image_height: int = 224,
        image_width: int = 224,
        device: torch.device = torch.device("cpu"),
    ):
        """
        hdf5_file: the path of the hdf5 dataset
        augment: choose if doing augmentation on image or not
        indices: used to split the train/val dataset
        ds_total_len: Define how many times the amount of data (data augmentation + original data) the dataset will process is the original dataset
            e.g.: ds_total_len=2 represents that this dataset would process 2 times the amount of images than the original dataset size.
                  In other words, to process all original images and the same number of augmented images as original images.
        image_height | image_width: the size of the image as the input into the model
        device: the CPU/GPU device
        """
        self.hdf5_file = hdf5_file
        self.augment = augment
        self.indices = indices
        self.ds_total_len = ds_total_len
        self.image_height = image_height
        self.image_width = image_width
        self.device = device

        with h5py.File(self.hdf5_file, "r") as hf:
            self.base_length = hf["images"].shape[0]
            # load all labels into memory at one time (high efficiency)
            self.labels = hf["labels"][:]

        # if without given indices, do not do any operations on dataset
        if self.indices is None:
            self.indices = np.arange(len(self.labels))

        # if with indices, update the labels and dataset length
        self.labels = self.labels[self.indices]
        self.base_length = len(self.indices)

        # data augmentation
        self.augmentation_transforms = {
            "Horizontal Flip": transforms.RandomHorizontalFlip(p=1.0),
            "Rotation": transforms.RandomRotation(degrees=10),
            "Color Jitter": transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            "Random Erasing": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), value=0),
                    transforms.ToPILImage(),
                ]
            ),
        }

        # resizing original images
        self.base_transform = transforms.Resize((self.image_height, self.image_width))

        # normalization(fit for ImageNet pretraining)
        self.normalize_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # process all original images and the same number of augmented images as original images
        self.length = self.ds_total_len * self.base_length

    def random_resized_crop(self, img):
        """
        Crop the image randomly
        """
        crop_transform = transforms.RandomResizedCrop(
            self.image_width, scale=(0.3, 1.0)
        )

        return crop_transform(img)

    def center_included_random_crop(self, img):
        """
        Ensure random cropping within the center point
        """
        W, H = img.size
        center_x, center_y = (
            W // 2,
            H // 2,
        )  # calculate the center point of the original image

        scale_min, scale_max = 0.3, 0.6  # allowable crop ratio
        crop_ratio = random.uniform(scale_min, scale_max)  # generate the crop ratio
        crop_W, crop_H = int(W * crop_ratio), int(H * crop_ratio)

        # make sure the original image's center point in the crop frame
        left_min = max(0, center_x - crop_W)
        top_min = max(0, center_y - crop_H)
        left_max = min(W - crop_W, center_x)
        top_max = min(H - crop_H, center_y)

        # random select the crop frame's coordinates
        left = random.randint(left_min, left_max)
        top = random.randint(top_min, top_max)
        right = left + crop_W
        bottom = top + crop_H

        # resizing
        cropped_img = img.crop((left, top, right, bottom))
        resize_transform = transforms.Resize((self.image_height, self.image_width))

        return resize_transform(cropped_img)

    def apply_augmentations(self, img):
        """
        Apply random augmentation to the image
        """

        # random select the cropping method
        crop_methods = [self.random_resized_crop, self.center_included_random_crop]
        img = random.choice(crop_methods)(img)

        # random select other augmentation methods
        num_augmentations = random.randint(1, len(self.augmentation_transforms))
        chosen_augmentations = random.sample(
            list(self.augmentation_transforms.keys()), num_augmentations
        )

        for augmentation in chosen_augmentations:
            img = self.augmentation_transforms[augmentation](img)

        return img

    def __len__(self):
        """
        return the number of data in the dataset
        """
        return self.length

    def __getitem__(self, idx):
        # judge if need do data augmentation (bool)
        is_augmented = idx >= self.base_length

        # if need data augmentation, need to map the idx into original dataset (subtract base_length)
        # and use the original image corresponding to idx for data augmentation to generate the augmented image
        true_idx = idx if not is_augmented else int(idx - int(self.base_length))

        with h5py.File(self.hdf5_file, "r") as hf:
            image = hf["images"][self.indices[true_idx]]
            label = self.labels[true_idx]

        # convert NumPy -> PILï¼ˆfit with torchvision.transformsï¼‰
        image = Image.fromarray(image)

        # resizing first
        image = self.base_transform(image)

        # data augmentation
        if is_augmented and self.augment:
            image = self.apply_augmentations(image)

        # DO normalization, we use pretained weights, so follow this normalization
        image = self.normalize_transform(image)

        # return image.to(self.device), torch.tensor(
        #     label, dtype=torch.long, device=self.device
        # )

        return image, torch.tensor(label, dtype=torch.long)


def compute_class_weights(labels, ds_total_len=2, augment_prob=0.5):
    """
    Compute sampling weights for original and augmented data.
    :param labels: np.array, original dataset labels
    :param ds_total_len: Dataset size multiplier (default = 2x original)
    :param augment_prob: Probability of selecting augmented samples
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Calculate the category sampling weight (the fewer category samples, the greater the weight)
    class_weights = {label: 1.0 / count for label, count in zip(unique_labels, counts)}

    # Original data weight
    original_sample_weights = np.array([class_weights[label] for label in labels])

    # Copy labels so that augmented_labels and original_labels keep the same category ratio
    augmented_labels = np.tile(labels, max(int(ds_total_len - 1), 1))
    augmented_sample_weights = np.array([class_weights[label] for label in augmented_labels])

    # Calculate how much original` vs. augmented should be in batch
    num_original = len(original_sample_weights)
    num_augmented = len(augmented_sample_weights)
    total_samples = num_original + num_augmented

    # Control the proportion of augmented data in batch
    if num_augmented == 0:
        augmented_ratio = 0
    else:
        augmented_ratio = augment_prob * total_samples / num_augmented

    original_ratio = (1 - augment_prob) * total_samples / num_original

    final_sample_weights = np.concatenate([
        original_sample_weights * original_ratio,  
        augmented_sample_weights * augmented_ratio 
    ])

    return final_sample_weights


class ImageClsDataModule(L.LightningDataModule):
    def __init__(
        self,
        hdf5_file: str,
        batch_size:int=32,
        train_ratio: float=0.8,
        ds_total_len:float=2.0,
        initial_augment_prob:float=0.8,
        final_augment_prob:float=0.3,
        num_epochs:int=50,
        image_height:int=224,
        image_width:int=224,
        device: torch.device = torch.device("cpu"),
        num_workers: int = 0,
    ):
        """
        Params:
            hdf5_file: the path of the hdf5 dataset
            batch_size: size of batch
            train_ratio: train/val dataset ratio
            ds_total_len: Define how many times the amount of data (data augmentation + original data) the dataset will process is the original dataset
                e.g.: ds_total_len=2 represents that this dataset would process 2 times the amount of images than the original dataset size.
                    In other words, to process all original images and the same number of augmented images as original images.
            initial_augment_prob: Initial data augmentation ratio
            final_augment_prob: final data augmentation ratio
            num_epochs: the number of epochs
            device: USE GPU/CPU deivce
            num_worker: the number of workers for dataloader
        """
        super().__init__()
        self.hdf5_file = hdf5_file
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.ds_total_len = ds_total_len
        self.initial_augment_prob = initial_augment_prob
        self.final_augment_prob = final_augment_prob
        self.num_epochs = num_epochs
        self.device = device
        self.num_workers = num_workers
        self.image_height=image_height
        self.image_width=image_width

        self.current_epoch = 0

    def setup(self, stage=None):
        """
        Split dataset into train and testsets
        """

        with h5py.File(self.hdf5_file, "r") as hf:
            labels = np.array(hf["labels"])
            num_samples = len(labels)

        # ensure the label balancing
        train_indices, val_indices = train_test_split(
            np.arange(num_samples),
            test_size=1 - self.train_ratio,
            stratify=labels,
            random_state=42,
        )

        # create train and val dataset
        self.train_dataset = ImageClsDataset(
            hdf5_file=self.hdf5_file,
            indices=train_indices,
            augment=True,
            ds_total_len=self.ds_total_len,
            device=self.device,
            image_height=self.image_height,
            image_width=self.image_width
        )

        self.val_dataset = ImageClsDataset(
            hdf5_file=self.hdf5_file,
            indices=val_indices,
            augment=False,
            # hard coded as 1 because onlu use original images in val set
            ds_total_len=1,
            device=self.device,
            image_height=self.image_height,
            image_width=self.image_width
        )

    def update_augment_prob(self):
        """
        update the augmented ratio for each epoch

        use more augmented images in early-stage epoches.
        use fewer augmented images in late-stage epoches.
        """
        # sync the current training epoch
        if self.trainer is not None:
            self.current_epoch = self.trainer.current_epoch

        self.augment_prob = self.initial_augment_prob - (
            self.current_epoch / self.num_epochs
        ) * (self.initial_augment_prob - self.final_augment_prob)

        # print(f"ğŸ“¢ Epoch {self.current_epoch}: Augment Prob = {self.augment_prob:.2f}")
        # logging.info(
        #     f"ğŸ“¢ Epoch {self.current_epoch}: Augment Prob = {self.augment_prob:.2f}"
        # )

        return None

    def train_dataloader(self):
        """
        define how the dataloader created for each epoch
        requirements:
            set `reload_dataloaders_every_n_epochs=1` in L.trainer
        """
        # calculate the current epoch's augment ratio
        self.update_augment_prob()
        # self.train_dataset.update_augment_prob(self.augment_prob)

        # re-build the sampler
        sample_weights = compute_class_weights(
            labels=self.train_dataset.labels,
            ds_total_len=self.ds_total_len,
            augment_prob=self.augment_prob,
        )

        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        # print(
        #     f"TRAIN_DATALODER DATAMODULE FLAG (current epoch num): {self.current_epoch+1}"
        # )

        # logging.info(
        #     f"TRAIN_DATALODER DATAMODULE FLAG (current epoch num): {self.current_epoch+1}"
        # )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        define how the validation dataloader created for each epoch
        requirements:
            set `reload_dataloaders_every_n_epochs=1` in L.trainer
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# enable GPU
if torch.cuda.is_available():
    print("âœ… GPU is available!")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("â�Œ GPU not available. Using CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# add kaggle path
kaggle_output="/kaggle/working/"

os.makedirs("/kaggle/working/vit-base-patch16-224-in21k", exist_ok=True)


# load pre defined parameters
local_config_path = "/kaggle/working/vit-base-patch16-224-in21k"

# label encoder
label_encoder = {
    "dot": 0,
    "scatter": 1,
    "horizontal_bar": 2,
    "line": 3,
    "vertical_bar": 4,
}

# label decoder
label_decoder = {v: k for k, v in label_encoder.items()}

id2label = {idx: label for idx, label in label_decoder.items()}
label2id = {label: idx for idx, label in label_encoder.items()}

# Dataset parameters
image_cls_height = 320
image_cls_width = 320

# dataset size multiplier
ds_multiplier = 2

# Datamodule Parameters
initial_augment_prob = 0.8
final_augment_prob = 0.2

# Dataloader parameters
num_workers = 4
batch_size = 32
num_epochs = 30

# data path
hdf5_file = "/kaggle/input/benetech-ds/ImageTransformation.h5"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


print(f"pretrained model path: {local_config_path}")

print(f"label encoder: {label_encoder}")
print(f"label decoder: {label_decoder}")
print(f"id2label: {id2label}")
print(f"label2id: {label2id}")


mlflow_logger = MLFlowLogger(
    experiment_name="ViT-image-recognition",
    tracking_uri="/kaggle/working/mlruns"
)


# load VIT model's config and processor

config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k",cache_dir=local_config_path)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', cache_dir=local_config_path)


type(config)


# add customized paramaters into config file

config.num_labels = len(id2label)
config.id2label = id2label
config.label2id = label2id
config.hidden_dropout_prob = 0.1 
config.image_size = 320
config.patch_size = 16
# 320/16=20,
# input token num: 320*320 / 16*16 = 20*20=400 
# add CLS: 400+1=401
# each token seq len: 16*16*3=768

model_name = "google/vit-base-patch16-224-in21k"
config_path = cached_file(model_name, "config.json")


config


# load pretrained VIT model

model = ViTModel.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    config=config,
    # I want fine tuning the model with Higher Resolution, change the input size from 224 to 320
    # so set ignore mismatched_size right here
    ignore_mismatched_sizes=True,
    cache_dir=local_config_path,
)


print(model)


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  âœ… {name}: {param.shape}")
    else:
        print(f"  â�Œ {name}: (frozen)")


# freeze some layers

# fine tuning embedding layer
for param in model.embeddings.parameters():
    param.requires_grad = True

# freeze the first 6 layers(total 12)
for name, param in model.encoder.named_parameters():
    if any(f"layer.{i}" in name for i in range(6)):  
        param.requires_grad = False

# ğŸ”¥ fine tuning the last 6 layers
for name, param in model.encoder.named_parameters():
    if any(f"layer.{i}" in name for i in range(6, 12)):  
        param.requires_grad = True

# ğŸ”¥ fine tune pooling layer
for param in model.pooler.parameters():
    param.requires_grad = True


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable Params: {trainable_params:,} / {total_params:,}")


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  âœ… {name}: {param.shape}")
    else:
        print(f"  â�Œ {name}: (frozen)")


# load pretrained VIT model

model = ViTModel.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    config=config,
    # I want fine tuning the model with Higher Resolution, change the input size from 224 to 320
    # so set ignore mismatched_size right here
    ignore_mismatched_sizes=True,
    cache_dir=local_config_path,
)


# initilize DataModule

datamodule = ImageClsDataModule(
    hdf5_file=hdf5_file,
    batch_size=batch_size,
    ds_total_len=ds_multiplier,
    initial_augment_prob=initial_augment_prob,
    final_augment_prob=final_augment_prob,
    num_epochs=num_epochs,
    device=device,
    num_workers=num_workers,
    image_height=image_cls_height,
    image_width=image_cls_width
)


class ViTClassifier(L.LightningModule):
    def __init__(self, pretrained_model, config, num_class, lr=None):
        """
        Params:
            pretrained_model: The loaded ViT model
            config:(transformers.models.vit.configuration_vit.ViTConfig) the model's config file
            num_class: the numer of classes
            lr: learning rate
        """
        super().__init__()
        # saving the information in checkpoints and YAML files
        self.save_hyperparameters()

        self.model = pretrained_model
        self.config = config
        self.num_class = num_class
        self.learning_rate = (
            lr if lr is not None else self.hparams.get("learning_rate", 1e-4)
        )

        # define what layers in the pretrained model should be fine-tuning
        # fine tune the embedding (positional embedding) layer
        for param in self.model.embeddings.parameters():
            param.requires_grad = True

        # freeze the first 9 layers(total 12)
        for name, param in self.model.encoder.named_parameters():
            if any(f"layer.{i}" in name for i in range(9)):
                param.requires_grad = False

        # fine tuning the last 3 layers
        for name, param in self.model.encoder.named_parameters():
            if any(f"layer.{i}" in name for i in range(9, 12)):
                param.requires_grad = True

        # fine tune pooling layer
        for param in self.model.pooler.parameters():
            param.requires_grad = True

        # define the classfier head
        self.classifier = nn.Linear(
            in_features=config.hidden_size, out_features=self.num_class
        )

    def forward(self, x):
        outputs = self.model(x)
        # In VIT model, we rely on CLS token to do classification
        cls_token = outputs.last_hidden_state[:, 0, :]
        # classifier head
        logits = self.classifier(cls_token)

        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self(images)
        val_loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"val_loss": val_loss, "val_acc": acc}

    def predict_step(self, batch, batch_idx):
        images, _ = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        return preds

    def on_train_epoch_end(self):

        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Train Epoch {self.current_epoch + 1} - Avg Loss: {train_loss:.4f}")

        # mlflow
        self.logger.log_metrics(
            {
                "manual_record_train_loss": train_loss.item()
            },
            step=self.current_epoch,
        )

    def on_validation_epoch_end(self):

        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_acc = self.trainer.callback_metrics.get("val_acc")
        if val_loss is not None and val_acc is not None:
            print(
                f"Val Epoch {self.current_epoch + 1} - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

        # mlflow
        self.logger.log_metrics(
            {
                "manual_record_val_loss": val_loss.item(),
                "manual_record_val_acc": val_acc.item(),
            },
            step=self.current_epoch,
        )

    def on_fit_start(self):
        print(f"âœ… Model is on device: {next(self.parameters()).device}")
        if self.logger:
            self.logger.log_hyperparams(self.hparams)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2, # trigger lr scheduler if without imporvement 3 times
            verbose=True,
        )

        # decrease lr based off val loss automatically
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1, # do step each 1 epoch
            },
        }


VitModel=ViTClassifier(pretrained_model=model.to(device),config=config,num_class=len(id2label))


class ViTCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        print(f"ğŸ“‰ Learning Rate after Epoch {trainer.current_epoch + 1}: {lr:.6f}")


# checkpoint
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    dirpath=f"ViTCheckpoints/{timestamp}/", 
    filename="vit-best-{epoch:02d}-{val_acc:.4f}",
    save_weights_only=True,
    verbose=True,
)

# early stopping
early_stop_callback = EarlyStopping(
    monitor="val_acc",
    patience=5, # stop the training if without improvement 5 epochs continusly
    mode="max",
    verbose=True,
)


trainer = L.Trainer(
    max_epochs=num_epochs,
    callbacks=[ViTCallback(), checkpoint_callback, early_stop_callback],
    logger=mlflow_logger,
    enable_checkpointing=True,
    log_every_n_steps=1,
    gradient_clip_val=1.0,
    enable_model_summary=True,
    accumulate_grad_batches=4
)


trainer.fit(VitModel,datamodule)


import shutil

# æ‰“åŒ…æ•´ä¸ª /kaggle/working åˆ°ä¸€ä¸ª zip æ–‡ä»¶ä¸­ï¼ˆå�¯ä»¥åŒ…å�«æ¨¡å�‹ã€�æ—¥å¿—ç­‰ï¼‰
shutil.make_archive("/kaggle/working/kaggle_output", 'zip', "/kaggle/working")




