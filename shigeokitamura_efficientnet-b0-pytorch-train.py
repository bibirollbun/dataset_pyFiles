import time
import os
import random
import gc
import time
import cv2
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import timm


# Check if gpu is available
torch.cuda.is_available()


class CFG:
    seed: int = 42
    debug: bool = False
    apex: bool = False
    print_freq: int = 100
    num_workers: int = 2

    OUTPUT_DIR: str = "kaggle/working/"

    train_data_dir: str = "/kaggle/input/birdclef-2025/train_audio"
    train_csv: str = "/kaggle/input/birdclef-2025/train.csv"
    test_soundscapes: str = "/kaggle/input/birdclef-2025/test_soundscapes"
    submission_csv: str = "/kaggle/input/birdclef-2025/sample_submission.csv"
    taxonomy_csv: str = "/kaggle/input/birdclef-2025/taxonomy.csv"
    spectrogram_npy: str = "/kaggle/input/transforming-audio-to-mel-spec/birdclef2025_melspec_5sec_256_256.npy"

    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    in_channels: int = 1

    TARGET_SHAPE: tuple[int, int] = (256, 256)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    batch_size: int = 32
    criterion: str = "BCEWithLogitsLoss"

    n_fold: int = 5
    selected_folds: list[int] = [0, 1, 2, 3, 4]

    optimizer: str = "AdamW"
    lr: float = 5e-4
    weight_decay: float = 1e-5

    scheduler: str = "CosineAnnealingLR"
    # scheduler: str = "ReduceLROnPlateau"
    # scheduler: str = "StepLR"
    # scheduler: str = "OneCycleLR"
    min_lr: float = 1e-6
    T_max: int = epochs

    aug_prob: float = 0.5
    mixup_alpha: float = 0.5

    def update_debug_settings(self) -> None:
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

cfg = CFG()


# Set seed for reproducibility:

random.seed(cfg.seed)
os.environ["PYTHONHASHSEED"] = str(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BirdCLEFDatasetFromNPY(Dataset):
    """
    Custom PyTorch Dataset class for the BirdCLEF 2025 dataset.
    It loads pre-computed mel spectrograms for training or validation
    and applies data augmentations if necessary.

    Args:
        df (pd.DataFrame): DataFrame containing the data (e.g., train.csv or a subset).
        cfg (CFG): Configuration object containing hyperparameters and settings.
        spectrograms (dict, optional): Dictionary of pre-computed mel spectrograms.
                                       Keys are sample `samplename`, values are the spectrogram numpy arrays.
                                       Defaults to None.
        mode (str, optional): Mode of the dataset, either 'train' or 'valid'.
                              Defaults to 'train'.
    """

    def __init__(self, df: pd.DataFrame, cfg: CFG, spectrograms: [dict[str, np.ndarray] | None]=None, mode: str="train") -> None:
        """
        Initializes the BirdCLEFDatasetFromNPY.
        Sets up the DataFrame, configuration, spectrogram data, and label encoding.
        """

        self.df: pd.DataFrame = df
        self.cfg: CFG = cfg
        self.mode: str = mode

        self.spectrograms: [dict[str, np.ndarray] | None] = spectrograms
        
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids: list[str] = taxonomy_df["primary_label"].tolist()
        self.num_classes: int = len(self.species_ids)
        self.label_to_idx: dict[str, int] = {label: idx for idx, label in enumerate(self.species_ids)}

        if "filepath" not in self.df.columns:
            self.df["filepath"] = self.cfg.train_data_dir + "/" + self.df.filename
        
        if "samplename" not in self.df.columns:
            self.df["samplename"] = self.df.filename.map(lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0])

        sample_names = set(self.df["samplename"])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")
        
        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """

        return len(self.df)
    
    def __getitem__(self, idx) -> dict[str, [torch.Tensor | str]]:
        """
        Loads an item at the given index, applies preprocessing and augmentations.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the following keys:
                  - 'melspec' (torch.Tensor): The mel spectrogram tensor.
                  - 'target' (torch.Tensor): The one-hot encoded target label tensor.
                  - 'filename' (str): The original filename.
        """

        row = self.df.iloc[idx]
        samplename = row["samplename"]
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row["filepath"], self.cfg)

        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)
        
        target = self.encode_label(row["primary_label"])
        
        if "secondary_labels" in row and row["secondary_labels"] not in [[""], None, np.nan]:
            if isinstance(row["secondary_labels"], str):
                secondary_labels = eval(row["secondary_labels"])
            else:
                secondary_labels = row["secondary_labels"]
            
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0
        
        return {
            "melspec": spec, 
            "target": torch.tensor(target, dtype=torch.float32),
            "filename": row["filename"]
        }
    
    def apply_spec_augmentations(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Applies data augmentations to the mel spectrogram.
        Includes Time Masking, Frequency Masking, and random brightness/contrast adjustment.

        Args:
            spec (torch.Tensor): The mel spectrogram tensor to apply augmentations to.

        Returns:
            torch.Tensor: The mel spectrogram tensor with augmentations applied.
        """
    
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0
        
        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0
        
        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1) 
            
        return spec
    
    def encode_label(self, label: str) -> np.ndarray:
        """
        Encodes a single label into a one-hot vector.

        Args:
            label (str): The label to encode.

        Returns:
            np.ndarray: The one-hot encoded label as a numpy array.
        """

        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target


def collate_fn(batch: list[dict[str, [torch.Tensor | str]]]) -> dict[str, [torch.Tensor | list[str]]]:
    """
    Custom collate function to handle batches from BirdCLEFDatasetFromNPY.
    This function is particularly useful for handling potential None items
    returned by __getitem__ and for correctly stacking tensors in the batch.

    Args:
        batch (list[Optional[dict[str, Union[torch.Tensor, str]]]]): A list of samples from the dataset.
                                                                      Each sample is a dictionary or None.

    Returns:
        dict[str, Union[torch.Tensor, list[str]]]: A dictionary where keys are the item names
                                                  (e.g., 'melspec', 'target', 'filename')
                                                  and values are either stacked tensors
                                                  (for 'melspec' and 'target' if shapes are uniform)
                                                  or a list (for 'filename').
                                                  Returns an empty dictionary if the input batch is empty
                                                  or contains only None values.
    """

    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    result = {key: [] for key in batch[0].keys()}
    
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    
    for key in result:
        if key == "target" and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == "melspec" and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
    
    return result


class BirdCLEFModel(nn.Module):
    """Deep learning model for bird song classification using a pre-trained backbone."""
    
    def __init__(self, cfg) -> None:
        """
        Initializes the BirdCLEFModel.
        Sets up the backbone model and the classifier head.
        
        Args:
            cfg (CFG): Configuration object containing model settings and hyperparameters.
        """
        
        super().__init__()
        self.cfg = cfg
        
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df) # Update num_classes in cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )

        # Determine the number of features from the backbone's output
        if "efficientnet" in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif "resnet" in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # Generic approach for models with a get_classifier method
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, "")
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
            
        self.feat_dim = backbone_out
        
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha
            
    def forward(self, x: torch.Tensor, targets: [torch.Tensor | None]=None) -> [torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model. Optionally applies Mixup during training.

        Args:
            x (torch.Tensor): Input tensor (spectrogram batch). Shape (batch_size, channels, height, width).
            targets (Optional[torch.Tensor], optional): Target labels for Mixup.
                                                      Shape (batch_size, num_classes). Defaults to None.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: If training with Mixup, returns a tuple
                                                                  of (logits, loss). Otherwise, returns
                                                                  the logits tensor.
        """

        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)

        # Handle potential dictionary output from some backbones
        if isinstance(features, dict):
            features = features['features']

        # Apply pooling if the output is 4D (convolutional features)
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits
    
    def mixup_data(self, x: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Applies mixup to the data batch and targets.

        Args:
            x (torch.Tensor): Input tensor batch. Shape (batch_size, channels, height, width).
            targets (torch.Tensor): Target labels batch. Shape (batch_size, num_classes).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: A tuple containing:
                                                                  - mixed_x (torch.Tensor): The mixed input tensor.
                                                                  - targets_a (torch.Tensor): First set of targets.
                                                                  - targets_b (torch.Tensor): Second set of targets.
                                                                  - lam (float): The lambda value used for mixing.
        """

        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Applies mixup to the loss function.

        Args:
            criterion (nn.Module): The loss function (e.g., nn.BCEWithLogitsLoss).
            pred (torch.Tensor): The model predictions (logits).
            y_a (torch.Tensor): The first set of targets.
            y_b (torch.Tensor): The second set of targets.
            lam (float): The lambda value used for mixing.

        Returns:
            torch.Tensor: The mixed loss.
        """

        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_optimizer(model: nn.Module, cfg: CFG) -> optim.Optimizer:
    """
    Initializes and returns a PyTorch optimizer based on the configuration.

    Args:
        model (nn.Module): The PyTorch model for which the optimizer is created.
        cfg (CFG): Configuration object containing optimizer type, learning rate, and weight decay.

    Returns:
        torch.optim.Optimizer: An instance of a PyTorch optimizer.

    Raises:
        NotImplementedError: If the optimizer specified in the configuration is not implemented.
    """
    
    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
        
    return optimizer


def get_scheduler(optimizer: optim.Optimizer, cfg: CFG) -> lr_scheduler._LRScheduler | lr_scheduler.ReduceLROnPlateau | None:
    """
    Initializes and returns a PyTorch learning rate scheduler based on the configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is created.
        cfg (CFG): Configuration object containing scheduler type and its specific parameters.

    Returns:
        torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
            An instance of a PyTorch learning rate scheduler, or None if no scheduler is specified
            or the specified scheduler is 'OneCycleLR' (which might be handled differently).
    """
    
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == "OneCycleLR":
        scheduler = None  
    else:
        scheduler = None

    return scheduler


def get_criterion(cfg: CFG) -> nn.Module:
    """
    Initializes and returns a PyTorch loss function based on the configuration.

    Args:
        cfg (CFG): Configuration object containing the criterion (loss function) type.

    Returns:
        torch.nn.Module: An instance of a PyTorch loss function.

    Raises:
        NotImplementedError: If the criterion specified in the configuration is not implemented.
    """
    
    if cfg.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")

    return criterion


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scheduler: lr_scheduler._LRScheduler | lr_scheduler.ReduceLROnPlateau | None = None
) -> tuple[float, float]:
    """
    Performs one training epoch for the model.

    Iterates through the DataLoader, calculates the loss for each batch,
    performs backpropagation, and updates the model's weights using the optimizer.
    Optionally steps the learning rate scheduler if provided.

    Args:
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): DataLoader providing the training data batches.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
        criterion (nn.Module): The loss function.
        device (str): The device to perform training on ('cuda' or 'cpu').
        scheduler (torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None, optional):
            The learning rate scheduler. Expected to be stepped after each batch if it's a OneCycleLR,
            otherwise stepped outside this function after the epoch. Defaults to None.

    Returns:
        tuple[float, float]: A tuple containing the average training loss and the
                             average ROC AUC score for the epoch.
    """
    
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for step, batch in pbar:
        # Handle the case where collate_fn might return lists of tensors
        # (although the current collate_fn primarily stacks fixed-size tensors)
    
        if isinstance(batch["melspec"], list):
            batch_outputs = []
            batch_losses = []
            
            for i in range(len(batch["melspec"])):
                # Ensure inputs and targets are tensors and on the correct device
                inputs = batch["melspec"][i].unsqueeze(0).to(device)
                target = batch["target"][i].unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                output = model(inputs)
                # Assuming output is logits if mixup is not used in forward for single samples
                loss = criterion(output, target)
                loss.backward()
                
                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())
            
            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch["target"].numpy()

        else:
            # Standard batch processing
            inputs = batch["melspec"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Pass targets for potential mixup
            
            if isinstance(outputs, tuple):
                # Model returned (logits, loss) due to mixup
                outputs, loss = outputs  
            else:
                # Model returned logits
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

        # Step scheduler if it's a OneCycleLR (stepped after each batch)
        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            
        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())
        
        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })

    # Concatenate results from all batches
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC for the epoch
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Iterates through the DataLoader in evaluation mode, calculates the loss
    and performance metric (AUC) without backpropagation.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        loader (DataLoader): DataLoader providing the validation data batches.
        criterion (nn.Module): The loss function.
        device (str): The device to perform evaluation on ('cuda' or 'cpu').

    Returns:
        tuple[float, float]: A tuple containing the average validation loss and the
                             average ROC AUC score for the validation set.
    """

    model.eval() # Set the model to evaluation mode
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch in tqdm(loader, desc="Validation"):
            # Handle the case where collate_fn might return lists of tensors
            if isinstance(batch["melspec"], list):
                batch_outputs = []
                batch_losses = []
                
                for i in range(len(batch["melspec"])):
                    # Ensure inputs and targets are tensors and on the correct device
                    inputs = batch["melspec"][i].unsqueeze(0).to(device)
                    target = batch["target"][i].unsqueeze(0).to(device)
                    
                    output = model(inputs)
                    loss = criterion(output, target)
                    
                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())
                
                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch["target"].numpy()
                
            else:
                inputs = batch["melspec"].to(device)
                targets = batch["target"].to(device)
                
                outputs = model(inputs) # No targets needed in forward for validation
                loss = criterion(outputs, targets)
                
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())

    # Concatenate results from all batches
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC for the validation set
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc


def calculate_auc(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Calculates the mean ROC AUC score across all classes.

    Computes the ROC AUC score for each class individually where there are
    positive samples in the target, and then returns the average of these scores.

    Args:
        targets (np.ndarray): Ground truth labels (one-hot encoded). Shape (num_samples, num_classes).
        outputs (np.ndarray): Model predictions (logits or probabilities). Shape (num_samples, num_classes).

    Returns:
        float: The mean ROC AUC score across all classes with at least one positive sample.
               Returns 0.0 if there are no classes with positive samples.
    """

    num_classes = targets.shape[1]
    aucs = []
    
    probs = 1 / (1 + np.exp(-outputs))
    
    for i in range(num_classes):
        
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    
    return np.mean(aucs) if aucs else 0.0


train_df = pd.read_csv(cfg.train_csv)
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)


train_df.head()


taxonomy_df.head()


taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
cfg.num_classes = len(species_ids)


taxonomy_df.head()


print(cfg.debug)
if cfg.debug:
    cfg.update_debug_settings()


spectrograms = None


print("Loading pre-computed mel spectrograms from NPY file...")
spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
print(f"Loaded {len(spectrograms)} pre-computed mel spectrograms")


skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)


best_scores = []


df = train_df.copy()


for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
    if fold not in cfg.selected_folds:
        continue
        
    print(f'\n{"="*30} Fold {fold} {"="*30}')
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f'Training set: {len(train_df)} samples')
    print(f'Validation set: {len(val_df)} samples')
    
    train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, spectrograms=spectrograms, mode='train')
    val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, spectrograms=spectrograms, mode='valid')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    model = BirdCLEFModel(cfg).to(cfg.device)
    optimizer = get_optimizer(model, cfg)
    criterion = get_criterion(cfg)
    
    if cfg.scheduler == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            steps_per_epoch=len(train_loader),
            epochs=cfg.epochs,
            pct_start=0.1
        )
    else:
        scheduler = get_scheduler(optimizer, cfg)
    
    best_auc = 0
    best_epoch = 0
    
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        train_loss, train_auc = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            cfg.device,
            scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
        )
        
        val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

        if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'val_auc': val_auc,
                'train_auc': train_auc,
                'cfg': cfg
            }, f"model_fold{fold}.pth")
    
    best_scores.append(best_auc)
    print(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")
    
    # Clear memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


print("Cross-Validation Results:")
for fold, score in enumerate(best_scores):
    print(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
print(f"Mean AUC: {np.mean(best_scores):.4f}")




