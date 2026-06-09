import os, random
from collections import defaultdict
from typing import Any
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler

class ImageDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, image_root: str):
        self.metadata = metadata
        self.image_root = image_root
    
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.image_root, row.path)
        img = Image.open(img_path).convert("RGB")
        return row.image_id, np.array(img)


def get_dino_processor_and_model(model_name: str) -> tuple:
    """Return processor and model."""
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)

    return processor, model


def precompute_embeddings(
    metadata: pd.DataFrame,
    image_root: str,
    processor: Any,
    model: Any,
):
    # we need images in a list, really, don't stack them
    def custom_collate(batch):
        ids = [r[0] for r in batch]
        images = [r[1] for r in batch]
        return ids, images
    dataset = ImageDataset(metadata, image_root)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=os.cpu_count()
    )

    res = []
    with torch.no_grad():
        for image_ids, images in tqdm(dataloader, mininterval=30):
            inputs = processor(images=images, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state
            embeddings = features[:, 0, :].squeeze(0).cpu().numpy()
            
            for image_id, embedding in zip(image_ids, embeddings):
                res.append({
                    "image_id": image_id,
                    "embeddings": embedding
                })
                    
    return pd.DataFrame(res).sort_values("image_id")


metadata = pd.read_csv('/kaggle/input/animal-clef-2025/metadata.csv')

# pre-compute embeddings
processor, model = get_dino_processor_and_model('facebook/dinov2-base')
embeddings_df = precompute_embeddings(
    metadata, '/kaggle/input/animal-clef-2025', processor, model
)
embeddings_df.to_parquet("embeddings.parquet")
display(embeddings_df)

