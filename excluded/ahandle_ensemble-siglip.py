import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch import nn, optim
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU instead")
    device = torch.device("cpu")


import logging
import os
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class ISICDataset(Dataset):
    DEFAULT_TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str = None,
        hdf5: Any = None,
        transform: callable = None,
        selected_features: list = None,
        cache_size: int = 1000,
    ):
        if image_dir is None and hdf5 is None:
            raise ValueError("Either image_dir or hdf5 must be provided")

        self.hdf5 = hdf5
        self.image_dir = image_dir
        self.transform = transform or self.DEFAULT_TRANSFORMS

        self.features = None
        self.labels = None
        self.isic_ids = df["isic_id"].values
        self.selected_features = None

        if "target" in df.columns:
            self.labels = torch.tensor(df["target"].values, dtype=torch.float32)

        if selected_features and all(f in df.columns for f in selected_features):
            self.selected_features = selected_features
            self.features = torch.tensor(
                df[selected_features].values, dtype=torch.float32
            )

        self._cache_size = cache_size
        self._load_image_from_disc = lru_cache(maxsize=cache_size)(
            self._load_image_from_disc
        )

    @lru_cache(maxsize=1)
    def _load_image_from_disc(self, isic_id: str) -> Image.Image:
        try:
            return Image.open(os.path.join(self.image_dir, f"{isic_id}.jpg")).convert(
                "RGB"
            )
        except Exception as e:
            logger.error(f"Error loading image {isic_id}: {e}")
            raise

    def _load_image_from_hdf5(self, isic_id) -> Image.Image:
        byte_string = self.hdf5[isic_id][()]
        nparr = np.frombuffer(byte_string, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., ::-1]
        return Image.fromarray(image)

    def _load_image(self, isic_id: int) -> Image.Image:
        if self.image_dir:
            return self._load_image_from_disc(isic_id)
        return self._load_image_from_hdf5(isic_id)

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]

        image = self._load_image(isic_id)
        image = self.transform(image)

        label = torch.tensor(float("nan"))
        if self.labels is not None:
            label = self.labels[idx]

        x = torch.empty(0)
        if self.features is not None:
            x = self.features[idx]

        return {
            "image": image,
            "features": x,
            "label": label,
        }


class Solver:
    def __init__(
        self, model, device, criterion, optimizer, scheduler, model_tag="transformer"
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = criterion
        self.model_tag = model_tag

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    def predict(self, loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(self.device)
                outputs = self.model(inputs)
                pred = outputs.logits if hasattr(outputs, "logits") else outputs
                predictions.append(pred)
        result = torch.cat(predictions)
        return result.cpu().numpy()


class EnsembleInput:
    def __init__(self, prediction_weight: float, solver: Solver):
        self.prediction_weight = prediction_weight
        self.solver = solver


class EnsemblePredictor:
    def __init__(self, solvers: list[EnsembleInput]):
        self.solvers = solvers

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        def weighted_prediction(input):
            return input.solver.predict(data_loader) * input.prediction_weight
        with ThreadPoolExecutor() as executor:
            predictions = list(executor.map(weighted_prediction, self.solvers))
        return np.sum(predictions, axis=0)


def get_siglip_cpt(checkpoint_path):
    model_path = "/kaggle/input/google-siglip-base-patch16-224/pytorch/siglip-base-patch16-224/1/siglip-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 1), nn.Sigmoid()
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.BCELoss()
    
    solver = Solver(model,device,criterion, optimizer, scheduler, model_tag="SigLIP")
    solver.load_checkpoint(checkpoint_path)

    return solver
    


solver_1 = get_siglip_cpt("/kaggle/input/siglip-p01-p03/pytorch/p01-e15-p03-20/1/SigLIP-P01_15_1c6229de.pt")
solver_2 = get_siglip_cpt("/kaggle/input/siglip-p02-e25/pytorch/siglip-p02-e25/1/SigLIP-P02_25_469466c5.pt")
solver_3 = get_siglip_cpt("/kaggle/input/siglip-p01-p03/pytorch/p01-e15-p03-20/1/SigLIP-P03_20_bd7ccb1c.pt")
solver_full = get_siglip_cpt("/kaggle/input/siglip-full-10/pytorch/siglip-full-10/1/SigLIP_FULL_10_372a5e35.pt")


ensemble_inputs = [
    EnsembleInput(0.3, solver_2),
    EnsembleInput(0.4, solver_3),
    EnsembleInput(0.3, solver_full),
]

ensemble_predictor = EnsemblePredictor(ensemble_inputs)


!pip install h5py -q


import h5py
import numpy as np 
import cv2
from PIL import Image

test_hdf5_path = "/kaggle/input/isic-2024-challenge/test-image.hdf5"
test_hdf5 = h5py.File(test_hdf5_path, "r")

def get_image(isic_id: str):
    byte_string = test_hdf5[isic_id][()]
    nparr = np.frombuffer(byte_string, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1]
    return image


import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class HDF5Dataset(Dataset):
    DEFAULT_TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(
        self,
        df,
        hdf5,
        transform=None,
        selected_features=None,
        has_labels=True,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame that may contain:
                - isic_id: HDF5 key
                - target: Optional column if has_labels=True
                - other feature columns if selected_features is not None
            hdf5 (h5py.File): Open HDF5 file handle with images
            transform (callable): Optional transform
            selected_features (list): Optional feature columns
            has_labels (bool): If False, 'target' won't be read
        """
        self.hdf5 = hdf5
        self.transform = transform or self.DEFAULT_TRANSFORMS
        self.selected_features = selected_features

        if "isic_id" in df.columns:
            self.images = df["isic_id"].tolist()
        else:
            self.images = list(hdf5.keys())

        if has_labels and "target" in df.columns:
            self.labels = torch.tensor(df["target"].values, dtype=torch.float32)
        else:
            self.labels = None

        if selected_features and all(f in df.columns for f in selected_features):
            self.features = torch.tensor(df[selected_features].values, dtype=torch.float32)
        else:
            self.features = None

    def get_image(self, isic_id: str) -> Image.Image:
        """Load image from HDF5 memory buffer."""
        byte_string = self.hdf5[isic_id][()]
        nparr = np.frombuffer(byte_string, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            logger.error(f"Failed to decode image {isic_id}")
            raise ValueError(f"Invalid image data for {isic_id}")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        image = self.get_image(image_id)
        image = self.transform(image)

        label = (
            self.labels[idx] if self.labels is not None else torch.tensor(float("nan"))
        )

        if self.features is not None:
            x = self.features[idx]
        else:
            x = torch.empty(0)

        return {
            "image_id": image_id,
            "image": image,
            "features": x,
            "label": label,
        }


test_csv = "/kaggle/input/isic-2024-challenge/test-metadata.csv"
test_df = pd.read_csv(test_csv)


test_dataset = ISICDataset(test_df, hdf5=test_hdf5)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


preds = ensemble_predictor.predict(test_dataloader)


submission_df = test_df.copy()
submission_df["target"] = preds
submission_df = submission_df[['isic_id', 'target']]
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

