import os, random
from collections import defaultdict
from typing import Any
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import functools
import operator
import torch.nn.functional as F

class EmbeddingDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame,):
        self.metadata = self._filter_metadata(metadata)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        display(self.metadata.groupby("dataset").count())
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.metadata.identity)
    
    def _filter_metadata(self, metadata):
        # only do triplet learning on the database images that have at least 2 images
        metadata = metadata.loc[metadata['split'] == 'database']
        dfs = []
        for identity in metadata.identity.unique():
            id_metadata = metadata.loc[metadata.identity == identity].copy()
            if id_metadata.shape[0] >= 2:
                dfs.append(id_metadata)
        return pd.concat(dfs, axis=0).reset_index()
        
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        row = self.metadata.iloc[idx]
        embedding = torch.from_numpy(row.embeddings).to(self.device)
        idx = torch.tensor(idx).to(device)
        label = torch.tensor(self.label_encoder.transform([row.identity])[0]).to(device)
        return embedding, idx, label

    @functools.cache
    def _filter_positive(self, idx):
        df = self.metadata
        anchor = df.iloc[idx]
        # find all the rows that match the condition and sample 1
        cond = [
            (df.identity == anchor.identity),
            (df.image_id != anchor.image_id)
        ]
        # we only need the index from these so we try to reduce the cache
        return df[functools.reduce(operator.__and__, cond)].index

    @functools.cache
    def _filter_negative(self, idx, same_species=False):
        """find a random row not in the current identity"""
        df = self.metadata
        anchor = df.iloc[idx]
        cond = [(df.identity != anchor.identity)]
        if same_species:
            cond.append((df.dataset == anchor.dataset))
        # we only need the index from these so we try to reduce the cache
        return df[functools.reduce(operator.__and__, cond)].index
    
    def sample_positive(self, idx, n=1):
        """find a row with the same label but different index"""
        return random.choices(self._filter_positive(idx), k=n)

    def sample_negative(self, idx, n=1, same_species=False):
        """find a random row not in the current identity"""
        return random.choices(self._filter_negative(idx), k=n)


class ProjectionHead(nn.Module):
    """
    Simple projection head to transform embeddings.
    
    Tried out nonlinearity (relu) but not sure if this is actually
    appropriate given that dino features give a good representational basis.
    Normalization is useful because it links the cosine distance to 
    euclidean distance via inner product.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        z = self.fc(x)
        z = F.normalize(z, p=2, dim=-1) 
        return z


class TripletLoss(nn.Module):
    """
    Triplet loss with a margin.
    Takes embeddings of an anchor, a positive and a negative sample.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.pairwise_distance(anchor, positive)
        neg_dist = torch.pairwise_distance(anchor, negative)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class MRLTripletLoss(nn.Module):
    """
    Matryoshka Representation Learning with Triplet Loss.
    Applies triplet loss at multiple embedding dimensionalities (prefixes).
    """
    def __init__(self, 
                 margin: float = 1.0, 
                 nested_dims: list = [],
                 loss_weights: list = None,
                ):
        super().__init__()
        self.margin = margin
        self.base_triplet_loss = TripletLoss(margin=margin)
        self.nested_dims = nested_dims
        
        # Weights for each loss component (full + nested).
        # If not provided, defaults to equal weighting (1.0 for each).
        num_losses = 1 + len(self.nested_dims)
        self.loss_weights = loss_weights
        if self.loss_weights is None:
            self.loss_weights = [1.0] * num_losses
        assert len(self.loss_weights) == num_losses
        self.loss_weight_sum = sum(self.loss_weights)

    def forward(self, anchor_full, positive_full, negative_full):
        total_loss = 0.0
        
        loss_full = self.base_triplet_loss(anchor_full, positive_full, negative_full)
        total_loss += self.loss_weights[0] * loss_full
        
        for i, d_prefix in enumerate(self.nested_dims):
            if d_prefix >= anchor_full.shape[-1]:
                continue

            anchor_prefix = anchor_full[:, :d_prefix]
            positive_prefix = positive_full[:, :d_prefix]
            negative_prefix = negative_full[:, :d_prefix]
            
            loss_prefix = self.base_triplet_loss(anchor_prefix, positive_prefix, negative_prefix)
            total_loss += self.loss_weights[i+1] * loss_prefix
            
        return total_loss / self.loss_weight_sum


def train_with_online_triplet_generation(
    metadata,
    projection_head,
    batch_size=200,
    epochs=50,
    margin=1.0,
    learning_rate=5e-4,
    lr_step_size=25,
    lr_gamma=0.1,
    triplets_per_anchor=1,
):
    """
    Train with online triplet generation using a standard DataLoader.

    Args:
        dataset: Instance of EmbeddingDataset
        projection_head: projection head to train on top of embeddings
        batch_size: Batch size for DataLoader
        epochs: Number of training epochs
        margin: Margin for triplet loss
        learning_rate: Learning rate for optimizer
        triplets_per_anchor: Number of triplets to generate per anchor
    """
    # Create standard DataLoader with RandomSampler
    dataset = EmbeddingDataset(metadata)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset),
    )

    optimizer = optim.Adam(projection_head.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True, min_lr=1e-7) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma) 
    device = next(projection_head.parameters()).device

    # Initialize triplet loss
    criterion = TripletLoss(margin=margin)

    # Training loop
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        batch_count = 0

        # Process batches
        for batch_embeddings, batch_indices, _ in dataloader:
            batch_size = batch_embeddings.size(0)

            # Skip small batches
            if batch_size <= 1:
                continue

            batch_embeddings = projection_head(batch_embeddings)
            # NOTE: we want this function to be reset every batch, since
            # we want to be embedding with the updated head online
            @functools.cache
            def _embed(idx):
                emb, _, _ = dataset[idx]
                return projection_head(emb).unsqueeze(0)

            # Process each anchor in the batch
            batch_loss = 0
            batch_triplets = 0

            for anchor_embedding, anchor_index in zip(batch_embeddings, batch_indices):
                anchor_embedding = anchor_embedding.unsqueeze(0)
                anchor_index = int(anchor_index)
                
                # we're going to generate triplets in the following way
                # first we get a random positive of indices
                # then we get a random negative set of indices
                # then we also get random negative set of indices with the same species
                sample_size = triplets_per_anchor
                positive_indices = dataset.sample_positive(anchor_index, sample_size)
                negative_all_indices = dataset.sample_negative(anchor_index, sample_size)

                # the (pos, neg) pairs of indices from our sampled stuff
                pairs = [*zip(positive_indices, negative_all_indices)]

                for pos_idx, neg_idx in pairs:
                    # Compute loss for this triplet
                    pos_embedding = _embed(pos_idx)
                    neg_embedding = _embed(neg_idx)
                    triplet_loss = criterion(
                        anchor_embedding, pos_embedding, neg_embedding
                    )
                    batch_loss += triplet_loss
                    batch_triplets += 1

            # Skip batch if no valid triplets
            if batch_triplets == 0:
                continue

            # Normalize loss by number of triplets
            batch_loss = batch_loss / batch_triplets
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += batch_loss.item()
            batch_count += 1

        # update the scheduler as needed
        scheduler.step()
        
        # Print epoch statistics
        if batch_count > 0:
            epoch_loss = running_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, No valid triplets found")

        # save intermediate heads
        if epoch > 0 and epoch % 10 == 0:
            name =  f"head_epoch={epoch:03d}.pt"
            print(f"saving {name}")
            torch.save(projection_head.state_dict(), name)
            
    return projection_head


def train_batch_semi_hard_negative(
    metadata,
    projection_head,
    batch_size=200,
    epochs=50,
    margin=1.0,
    nested_dims=[64, 32, 16, 2],
    loss_weights=[1, 64/128, 32/128, 16/128, 2/128],
    learning_rate=5e-4,
    lr_step_size=25,
    lr_gamma=0.1,
    triplets_per_anchor=1,
):
    dataset = EmbeddingDataset(metadata)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset),
    )

    optimizer = optim.Adam(projection_head.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma) 
    device = next(projection_head.parameters()).device
    criterion = MRLTripletLoss(margin=margin, nested_dims=nested_dims, loss_weights=loss_weights)

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        batch_count = 0
        epoch_triplets = 0

        for batch_embeddings, batch_indices, batch_labels in dataloader:
            @functools.cache
            def _embed(idx):
                emb, _, _ = dataset[idx]
                return projection_head(emb).unsqueeze(0)

            batch_size = batch_embeddings.size(0)
            if batch_size <= 1:
                continue

            batch_embeddings = projection_head(batch_embeddings)
            batch_pairwise_dist = torch.cdist(
                batch_embeddings, batch_embeddings, p=2
            )

            # Process each anchor in the batch
            batch_loss = 0
            batch_triplets = 0

            for anchor_embedding, anchor_index, anchor_label, anchor_dist in zip(
                batch_embeddings, batch_indices, batch_labels, batch_pairwise_dist
            ):
                anchor_embedding = anchor_embedding.unsqueeze(0)
                anchor_index = int(anchor_index)
                
                # we're going to generate triplets in the following way
                # first we get a random positive of indices
                # then we do batch semi-hard mining by looking for everything within the margin
                # skip if it doesn't exist
                for _ in range(triplets_per_anchor):
                    pos_idx = dataset.sample_positive(anchor_index)[0]
                    pos_embedding = _embed(pos_idx)
                    # distance positive -> dp
                    dp = torch.pairwise_distance(anchor_embedding, pos_embedding)

    
                    # BATCH SEMI-HARD NEGATIVE MINING
                    # Distances from the current anchor (projected) to all other *projected* samples in THIS BATCH
                    # Mask for negatives within the current batch (different true label than anchor)                    
                    
                    # Semi-hard condition mask
                    semi_hard_mask = (
                        # labels that dont match the anchor
                        (batch_labels != anchor_label)
                        # dp < dn < dp+margin
                        & (anchor_dist > dp)
                        & (anchor_dist < (dp + criterion.margin))
                    )

                    # skip because no semi-hard
                    if not torch.any(semi_hard_mask):
                        continue 
    
                    # Get indices (within the current batch) of these semi-hard negatives
                    semi_hard_indices = torch.where(semi_hard_mask)[0]
                    
                    # Select one semi-hard negative (e.g., the hardest of the semi-hard, or random)
                    # To get the "hardest" of the semi-hard (closest to anchor among semi-hard):
                    smallest_neg = torch.min(anchor_dist[semi_hard_indices])
                    # Find which batch index corresponds to this distance among the semi-hard ones
                    neg_indices = torch.where(semi_hard_mask & (anchor_dist == smallest_neg))[0]
                    
                    if neg_indices.numel() == 0: continue
                    neg_idx = neg_indices[0]
    
                    neg_embedding = batch_embeddings[neg_idx].unsqueeze(0)
                    triplet_loss = criterion(
                        anchor_embedding, pos_embedding, neg_embedding
                    )
                    batch_loss += triplet_loss
                    batch_triplets += 1
                    epoch_triplets += 1

            # Skip batch if no valid triplets
            if batch_triplets == 0:
                continue

            # Normalize loss by number of triplets
            batch_loss = batch_loss / batch_triplets
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += batch_loss.item()
            batch_count += 1

        # update the scheduler as needed
        scheduler.step()
        
        # Print epoch statistics
        if batch_count > 0:
            epoch_loss = running_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Triplet Count: {epoch_triplets}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, No valid triplets found")

        # save intermediate heads
        if epoch > 0 and epoch % 10 == 0:
            name =  f"head_epoch={epoch:03d}.pt"
            print(f"saving {name}")
            torch.save(projection_head.state_dict(), name)
            
    return projection_head



metadata = pd.read_csv('/kaggle/input/animal-clef-2025/metadata.csv')
embeddings_df = pd.read_parquet("/kaggle/input/preprocess-triplet-embedding/embeddings.parquet")
merged_df = pd.merge(metadata, embeddings_df, on='image_id', how='inner')
display(merged_df.head(3))

device = "cuda" if torch.cuda.is_available() else "cpu"
proj_head = ProjectionHead(768, 128).to(device)
head = train_batch_semi_hard_negative(
    merged_df, 
    projection_head=proj_head,
    epochs=100,
)

# write the output
torch.save(head.state_dict(), '/kaggle/working/head.pt')

