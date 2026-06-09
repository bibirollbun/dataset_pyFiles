###Install required libraries:
!pip install git+https://github.com/mlfoundations/open_clip.git -q
!pip install faiss-gpu -qq 


import os
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np
import pandas as pd
import torch
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as tfms
import torchvision.transforms as T
import open_clip

from typing import Sequence, Tuple, Any, Dict, List, Optional, Union
from collections import defaultdict
import importlib


data_path = '/kaggle/input/fungi-clef-2025/'


# train_ds_raw = FungiTastic(
#     root=data_path,
#     split='train',
#     transform=transforms.ToTensor()          # yields torch.Tensor C×H×W in [0,1]
# )

# # 2) Accumulate sums and squared sums per channel
# sum_c  = torch.zeros(3)
# sum_sq = torch.zeros(3)
# pix_cnt = 0

# for img, *_ in tqdm(train_ds_raw, desc="Computing mean/std"):  
#     # img has shape [C, H, W], values in [0,1]
#     C, H, W = img.shape
#     sum_c  += img.sum(dim=[1,2])          # sum over H and W → [3]
#     sum_sq += (img**2).sum(dim=[1,2])     # sum of squares → [3]
#     pix_cnt += H * W

# # 3) Compute mean and std
# mean = sum_c  / pix_cnt
# var  = (sum_sq / pix_cnt) - mean**2
# std  = torch.sqrt(var)

# print("Computed means:", mean.tolist())
# print("Computed stds: ", std.tolist())
MEAN = [0.44550442695617676, 0.42286601662635803, 0.3496400713920593] 
STD = [0.2475634068250656, 0.2405535727739334, 0.23902718722820282] 


class FungiTastic(torch.nn.Module):
    
    SPLIT2STR = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

    def __init__(self, root: str, split: str = 'val', transform=None):
        super().__init__()
        self.split = split
        self.transform = transform
        self.df = self._get_df(root, split)

        assert "image_path" in self.df
        if self.split != 'test':
            assert "category_id" in self.df
            self.n_classes = len(self.df['category_id'].unique())
            self.category_id2label = {
                k: v[0] for k, v in self.df.groupby('category_id')['species'].unique().to_dict().items()
            }
            self.label2category_id = {
                v: k for k, v in self.category_id2label.items()
            }

    def add_embeddings(self, embeddings: pd.DataFrame):
        assert isinstance(embeddings, pd.DataFrame), "Embeddings must be a pandas DataFrame."
        assert "embedding" in embeddings.columns, "Embeddings DataFrame must have an 'embedding' column."
        assert len(embeddings) == len(self.df), "Embeddings must match dataset length."

        self.df = pd.merge(self.df, embeddings, on="filename", how="inner")

    def get_embeddings_for_class(self, id):
        # return the embeddings for class class_idx
        class_idxs = self.df[self.df['category_id'] == id].index
        return self.df.iloc[class_idxs]['embedding']
    
    @staticmethod
    def _get_df(data_path: str, split: str) -> pd.DataFrame:
        df_path = os.path.join(
            data_path,
            "metadata",
            "FungiTastic-FewShot",
            f"FungiTastic-FewShot-{FungiTastic.SPLIT2STR[split]}.csv"
        )
        df = pd.read_csv(df_path)
        df["image_path"] = df.filename.apply(
            lambda x: os.path.join(data_path, "FungiTastic-FewShot", split, '300p', x)
        )
        return df

    def __getitem__(self, idx: int):
        file_path = self.df["image_path"].iloc[idx].replace('FungiTastic-FewShot', 'images/FungiTastic-FewShot')
    
        if self.split != 'test':
            category_id = self.df["category_id"].iloc[idx]
        else:
            category_id = None
    
        image = Image.open(file_path)
    
        if self.transform:
            image = self.transform(image)
    
        # Check if embeddings exist
        if "embedding" in self.df.columns:
            emb = torch.tensor(self.df.iloc[idx]['embedding'], dtype=torch.float32).squeeze()
        else:
            emb = None  # No embeddings available

        return image, category_id, file_path, emb


    def __len__(self):
        return len(self.df)

    def get_class_id(self, idx: int) -> int:
        return self.df["category_id"].iloc[idx]

    def show_sample(self, idx: int) -> None:
        image, category_id, _, _ = self.__getitem__(idx)
        class_name = self.category_id2label[category_id]

        plt.imshow(image)
        plt.title(f"Class: {class_name}; id: {idx}")
        plt.axis('off')
        plt.show()

    def get_category_idxs(self, category_id: int) -> List[int]:
        return self.df[self.df.category_id == category_id].index.tolist()


def make_obs_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for obs_id, grp in df.groupby('observationID'):
        # each grp.embedding is a 1‑D numpy array
        embs = np.vstack(grp['embedding'].values)   # (V, D)
        rows.append({
            'observationID': obs_id,
            'category_id'  : grp['category_id'].iloc[0],
            'embedding'    : embs.mean(axis=0)
        })
    return pd.DataFrame(rows)


def make_obs_embeddings_test(df):
    """
    Only aggregates embeddings per observation,
    doesn’t require category_id.
    """
    rows = []
    for obs_id, grp in df.groupby('observationID'):
        emb = np.vstack(grp['embedding'].values).mean(axis=0)
        rows.append({
            'observationID': obs_id,
            'embedding'    : emb
        })
    return pd.DataFrame(rows)

def generate_ensemble_embeddings_raw(dataset):
    rows = []
    for _, row in tqdm(dataset.df.iterrows(),
                      total=len(dataset.df),
                      desc="Ensemble Embeds"):
        fp = row.image_path.replace(
            'FungiTastic-FewShot',
            'images/FungiTastic-FewShot'
        )
        img = Image.open(fp).convert("RGB")

        feats = []
        with torch.no_grad():
            for m, proc in zip(ensembleModels, ensembleProcessors):
                tensor = proc(img).unsqueeze(0).to(device)  # (1,3,H,W)
                f      = m.encode_image(tensor)            # (1, D_i)
                f      = F.normalize(f, dim=-1)            # L2 norm
                feats.append(f)

        fused = torch.cat(feats, dim=1)           # shape [1, 512+768+…]
        fused = F.normalize(fused, dim=-1)        # re‑normalize length
        rows.append({
            'filename' : Path(fp).name,
            'embedding': fused.cpu().numpy().squeeze()
        })

    return pd.DataFrame(rows)


ensembleModels = []
ensembleProcessors = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ensembleSpecs = [
    ('hf-hub:imageomics/bioclip',     None),               # BioCLIP
    ('ViT-L-14',                       'openai'),           # official OpenAI CLIP
    ('ViT-L-14',                       'laion2b_s32b_b82k') # LAION 2B variant
]

for modelName, pretrainedTag in ensembleSpecs:
    if pretrainedTag:
        m, _, proc = open_clip.create_model_and_transforms(
            modelName,
            pretrained=pretrainedTag
        )
    else:
        m, _, proc = open_clip.create_model_and_transforms(modelName)
    m.to(device).eval()
    ensembleModels.append(m)
    ensembleProcessors.append(proc)

print(f"Loaded {len(ensembleModels)} models")


trainDataset = FungiTastic(root=data_path, split='train', )
valDataset = FungiTastic(root = data_path, split = "val")
testDataset = FungiTastic(root = data_path, split = 'test')


train_emb_df = generate_ensemble_embeddings_raw(trainDataset)
trainDataset.add_embeddings(train_emb_df) 

val_emb_df   = generate_ensemble_embeddings_raw(valDataset)
valDataset.add_embeddings(val_emb_df)


trainObs = make_obs_embeddings(trainDataset.df)
valObs   = make_obs_embeddings(valDataset.df) 


proto_df    = (
    trainObs
      .groupby('category_id')['embedding']
      .apply(lambda arrs: np.stack(arrs.values).mean(axis=0))
      .reset_index()
)
proto_labels = proto_df['category_id'].values                        # (C,)
proto_embs   = np.stack(proto_df['embedding'].values).astype('float32')  # (C, D_ensemble)


faiss.normalize_L2(proto_embs)
index = faiss.IndexFlatIP(proto_embs.shape[1])
index.add(proto_embs) 


val_embs = np.stack(valObs['embedding'].values).astype('float32')
faiss.normalize_L2(val_embs)
K = 5
D, I = index.search(val_embs, K)              # top‑K
preds = proto_labels[I]                      # shape (N_val, 5)
true  = valObs['category_id'].values



recall5 = np.mean([ true[i] in preds[i] for i in range(len(true)) ])
print(f"Ensemble Val Recall@5 = {recall5:.4f}")


# test_emb_df = generate_ensemble_embeddings_raw(trainDataset)
# testDataset.add_embeddings(train_emb_df) 



# testObs = make_obs_embeddings_test(testDataset.df)
# test_embs = np.stack(testObs['embedding'].values).astype('float32')
# faiss.normalize_L2(test_embs)
# _, I_test = index.search(test_embs, 10)  # top‑10 for submission


# pred_strs = [" ".join(map(str, proto_labels[idxs])) for idxs in I_test]
# submission = pd.DataFrame({
#     'observationId': testObs['observationID'].values,
#     'predictions':   pred_strs
# })
# submission.to_csv('submission_ensemble.csv', index=False)




