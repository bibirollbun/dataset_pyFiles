!pip install --upgrade pip
!pip install ultralytics



import math
import os
import shutil
from collections import deque

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from ultralytics import YOLO



test_df = pd.read_csv("/kaggle/input/nexar-collision-prediction/test.csv")
print(f"Test shape: {test_df.shape}")



device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model = YOLO("yolov8m-seg.pt")
model.to(device)
vehicle_classes = {"car", "truck", "bus", "motorbike", "bicycle"}

@torch.no_grad()
def extract_mask(frame: np.ndarray, model: YOLO) -> np.ndarray | None:
    """Extracts vehicle masks from a given frame using the YOLO model.

    If no vehicle masks are found, returns None.
    """
    model.eval()

    results = model(frame, verbose=False)[0]
    masks = results.masks
    classes = results.boxes.cls

    if masks is None or masks.data is None or len(masks.data) == 0:
        return None

    H, W = frame.shape[:2]
    mask_out = np.zeros((H, W), dtype=np.uint8)

    # Convert once
    masks_np = masks.data.cpu().numpy().astype(np.uint8)
    classes_np = classes.cpu().numpy()

    for seg, cls_id in zip(masks_np, classes_np):
        class_name = model.model.names[int(cls_id)]
        if class_name in vehicle_classes:
            mask = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_out[mask > 0] = 255

    return mask_out[None, ...]  # Shape: (1, H, W)



def compute_flow_channels(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Compute optical flow channels between two frames.

    Flow is returned as a 2-channel image with shape (2, H, W):
    - Channel 0: Magnitude of flow
    - Channel 1: Angle of flow
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    # Compute Dense Optical Flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and direction
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize magnitude and angle
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    angle = (angle * 180 / np.pi / 2).astype(np.uint8)

    # Stack into a single 2-channel image
    flow_channels = np.dstack((magnitude, angle))

    # Reorder dimensions to (2, H, W)
    return np.moveaxis(flow_channels, -1, 0)  # Shape: (2, H, W)



def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample imput image by a given factor.

    Accepts images in both (C, H, W) and (H, W, C) formats, but returns in (C, H, W) format.
    """
    if image.shape[0] <= 3:
        # If the image is in (C, H, W) format, transpose to (H, W, C) for resizing
        image = np.transpose(image, (1, 2, 0))

    # Downsample the image
    h, w = image.shape[:2]
    downsampled_image = cv2.resize(image, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)

    # If downsampled image now has two dimensions, convert to 1 channels (H, W, 1)
    if len(downsampled_image.shape) == 2:
        downsampled_image = np.expand_dims(downsampled_image, axis=-1)

    # Make sure to resize back to (C, H, W) format
    return np.transpose(downsampled_image, (2, 0, 1))



FRAMES_NEEDED = 6  # 3 for optical flow context + 3 to process

dq = deque(maxlen=3)
for row in tqdm(test_df.to_dict(orient="records")):
    video_path = f"/kaggle/input/nexar-collision-prediction/test/{str(row['id']).zfill(5)}.mp4"
    video = cv2.VideoCapture(video_path)

    # Get total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, total_frames - FRAMES_NEEDED)

    # Seek to frame N - 6
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    save_index = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dq.append(frame)

        # Only save the *last 3 frames* (from N-3 to N-1)
        if frame_index >= total_frames - 3 and len(dq) == 3:
            mask = extract_mask(frame, model)           # Shape: (1, H, W)
            flow = compute_flow_channels(dq[0], frame)  # Shape: (2, H, W)

            flow_downsampled = downsample_image(flow, 3)
            frame_downsampled = downsample_image(frame, 3)

            save_dir = f"/kaggle/working/processed/test/{str(row['id']).zfill(5)}"
            os.makedirs(save_dir, exist_ok=True)

            flow_path = save_dir + f"/flows/{str(save_index).zfill(2)}.pt"
            os.makedirs(os.path.dirname(flow_path), exist_ok=True)
            torch.save(torch.tensor(flow_downsampled).to(dtype=torch.float32), flow_path)

            frame_path = save_dir + f"/frames/{str(save_index).zfill(2)}.pt"
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            torch.save(torch.tensor(frame_downsampled).to(dtype=torch.int16), frame_path)

            if mask is not None:
                mask_downsampled = downsample_image(mask, 3)
                mask_path = save_dir + f"/masks/{str(save_index).zfill(2)}.pt"
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                torch.save(torch.tensor(mask_downsampled).to(dtype=torch.int16), mask_path)

            save_index += 1

        frame_index += 1

    video.release()



# Add path to features
test_df["features_path"] = test_df["id"].apply(lambda x: f"/kaggle/working/processed/test/{str(x).zfill(5)}")

# Add number of frames (for sampling)
test_df["n_frames"] = test_df["features_path"].apply(lambda x: len(os.listdir(x + "/frames")))

test_df.head()



class NexarDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        frame_idx: int | None = None,
        return_label: bool = True,
        transform=None,
    ) -> None:
        self.df = df
        self.frame_idx = frame_idx
        self.return_label = return_label
        self.transform = transform

        self.features_path = df["features_path"].values
        self.n_frames = df["n_frames"].values
        if self.return_label:
            self.labels = df["target"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        data = {"idx": idx}

        # Sample frame
        frame_idx = (
            self.frame_idx
            if self.frame_idx is not None
            else np.random.randint(0, self.n_frames[idx] - 1)
        )
        data["frame_idx"] = frame_idx

        # Load features
        folder = self.features_path[idx]
        frame = torch.load(folder + f"/frames/{str(frame_idx).zfill(2)}.pt")
        flow = torch.load(folder + f"/flows/{str(frame_idx).zfill(2)}.pt")
        try:
            mask = torch.load(folder + f"/masks/{str(frame_idx).zfill(2)}.pt")
        except FileNotFoundError:
            mask = torch.zeros((1, *flow.shape[1:]))

        # Multiply by mask
        frame = frame * (mask > 0).float()
        flow = flow * (mask > 0).float()
        mask_flow = torch.cat([flow, mask], dim=0)

        # Apply transformations
        if self.transform:
            frame = self.apply_transform(frame)
            mask_flow = self.apply_transform(mask_flow)
        data["frame"] = frame.to(torch.float32)
        data["mask_flow"] = mask_flow.to(torch.float32)

        if self.return_label:
            data["label"] = self.labels[idx]

        return data

    def apply_transform(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return self.transform(image)



def pad_to_square(image: torch.Tensor):
    """Pad the image to a square shape by adding zeros to all sides."""
    _, h, w = image.shape
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2

    # Padding format for torch.nn.functional.pad is (left, right, top, bottom)
    padding = (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h)

    # Pad expects input as (N, C, H, W) or (C, H, W)
    return nn.functional.pad(image, padding, mode="constant", value=0)



def build_mlp(
    n_in: int,
    n_out: int,
    hidden_layers: list[int],
    activation_fn: type[nn.Module] = nn.ReLU,
    dropout: float | None = None,
) -> nn.Module:
    """Build a simple MLP."""
    layers = []
    prev_units = n_in

    for units in hidden_layers:
        layers.append(nn.Linear(prev_units, units))
        layers.append(activation_fn())
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev_units = units

    layers.append(nn.Linear(prev_units, n_out))

    return nn.Sequential(*layers)



class NexarClassifier(LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        hidden_layers: list[int] = [],
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Image backbone
        pretrained_weights = models.ResNet34_Weights.DEFAULT
        self.image_backbone = models.resnet34(weights=pretrained_weights)
        image_backbone_features = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Linear(
            in_features=image_backbone_features,
            out_features=image_backbone_features,
        )
        for param in self.image_backbone.parameters():
            param.requires_grad = False
        for param in self.image_backbone.fc.parameters():
            param.requires_grad = True

        # Mask flow backbone
        self.mask_flow_backbone = models.resnet34(weights=pretrained_weights)
        mask_flow_backbone_features = self.mask_flow_backbone.fc.in_features
        self.mask_flow_backbone.fc = nn.Identity()

        # Classifier head
        self.classifier = build_mlp(
            n_in=image_backbone_features + mask_flow_backbone_features,
            n_out=1,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )

        # Loss function and accuracy metric
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")

    def forward(self, x):
        image, mask_flow = x["frame"], x["mask_flow"]
        img_emb = self.image_backbone(image)
        mf_emb = self.mask_flow_backbone(mask_flow)
        emb = torch.cat([img_emb, mf_emb], dim=1)
        return self.classifier(emb)

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        pred = self(batch)

        # Compute training loss
        loss = self.loss_fn(pred.squeeze(), labels.float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        # Compute training accuracy
        self.train_accuracy(pred.squeeze(), labels)
        self.log("train_acc", self.train_accuracy, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        pred = self(batch)

        # Compute validation loss
        loss = self.loss_fn(pred.squeeze(), labels.float())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # Compute validation accuracy
        self.val_accuracy(pred.squeeze(), labels)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.image_backbone.parameters(), "lr": self.hparams.lr * 0.1},      # Lower LR for backbone
                {"params": self.mask_flow_backbone.parameters(), "lr": self.hparams.lr * 0.1},  # Lower LR for backbone
                {"params": self.classifier.parameters(), "lr": self.hparams.lr},                # Default LR for classifier
            ],
            lr=self.hparams.lr,
        )  # fmt: skip
        return {"optimizer": optimizer}



model_paths = [
    "/kaggle/input/nexar_cp21sgvt/pytorch/default/1/epoch13-val_acc0.79.ckpt",  # Seed: 441490
    "/kaggle/input/nexar_ym88wo7m/pytorch/default/1/epoch13-val_acc0.79.ckpt",  # Seed: 879461
]

predictions_per_model = []
for path in model_paths:

    # Load model
    model = NexarClassifier.load_from_checkpoint(path)
    model.to(device)
    model.eval()
    
    # Get predictions for each frame
    predictions = {}
    indices = [0, 1, 2]
    weights = [0.2, 0.3, 0.5]   
    
    for frame_idx in indices:
        test_dataset = NexarDataset(
            test_df, 
            frame_idx=frame_idx, 
            return_label=False, 
            transform=T.Compose([
                T.Lambda(pad_to_square),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)
        
        preds = []
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                pred = model(batch)
            pred = torch.sigmoid(pred).squeeze().detach().tolist()
            preds.extend(pred)
        
        predictions[frame_idx] = preds
    
    # Take weighted average of predictions
    final_predictions = np.zeros(len(test_df))
    for i, frame_idx in enumerate(indices):
        final_predictions += np.array(predictions[frame_idx]) * weights[i]
    predictions_per_model.append(final_predictions / sum(weights))

# Save predictions
submission_df = pd.DataFrame({
    "id": test_df["id"].apply(lambda x: str(x).zfill(5)),
    "target": np.mean(predictions_per_model, axis=0),
})
submission_df.to_csv("submission.csv", index=False)
submission_df.head()



# Remove redundant files
os.remove("/kaggle/working/yolov8m-seg.pt")
shutil.rmtree("/kaggle/working/processed")


