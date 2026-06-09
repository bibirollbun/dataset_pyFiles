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


all_metadata = pd.read_csv('/kaggle/input/animal-clef-2025/metadata.csv')
metadata = all_metadata.loc[all_metadata['split'] == 'database']
metadata


dfs = []
for identity in metadata.identity.unique():
    id_metadata = metadata.loc[metadata.identity == identity].copy()
    
    if id_metadata.shape[0] >= 2:
        dfs.append(id_metadata)

metadata = pd.concat(dfs, axis=0)


metadata


def get_dino_processor_and_model(model_name: str) -> tuple:
    """Return processor and model."""
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    return processor, model


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        img_dir: str,
        processor: Any,
        model: Any,
        embed_type: str = 'cls',
        target_col: str = 'identity',
        img_id_col: str = 'image_id'
    ):

        self.metadata = metadata
        self.img_dir = img_dir
        self.processor = processor
        self.model = model
        self.embed_type = embed_type
        self.target_col = target_col
        self.img_id_col = img_id_col
        self.device = next(model.parameters()).device

        self.classes = sorted(metadata[target_col].astype(str).unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        row = self.metadata.iloc[idx]

        img_id = row[self.img_id_col]
        img_path = os.path.join(self.img_dir, row['path'])
        img = Image.open(img_path).convert('RGB')

        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state

            if self.embed_type == 'cls':
                embedding = features[:, 0, :].squeeze(0)
            else:
                embedding = features[:, 1:, :].mean(dim=1).squeeze(0)

        target = row[self.target_col]
        target_int = self.class_to_idx[target]
        label = torch.tensor(target_int, dtype=torch.long)

        return embedding, label, target


class TripletLoss(nn.Module):
    """
    Triplet loss with a margin.
    Takes embeddings of an anchor, a positive and a negative sample.
    """
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = torch.pairwise_distance(anchor, positive)
        neg_dist = torch.pairwise_distance(anchor, negative)
        
        # Calculate triplet loss
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class ProjectionHead(nn.Module):
    """
    Simple projection head to transform embeddings.
    """
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)


def train_with_online_triplet_generation(
    dataset,
    model=None,  # Not needed as embeddings are precomputed in this dataset
    batch_size=32,
    epochs=10,
    margin=1.0,
    learning_rate=0.001,
    triplets_per_anchor=1,
    projection_head=None
):
    """
    Train with online triplet generation using a standard DataLoader.
    
    Args:
        dataset: Instance of EmbeddingDataset
        model: The model to train (if needed)
        batch_size: Batch size for DataLoader
        epochs: Number of training epochs
        margin: Margin for triplet loss
        learning_rate: Learning rate for optimizer
        triplets_per_anchor: Number of triplets to generate per anchor
        projection_head: Optional projection head to train on top of embeddings
    """
    # Create standard DataLoader with RandomSampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset),
    )
    
    # Create a label to indices mapping
    label_to_indices = defaultdict(list)
    for idx in tqdm(range(len(dataset))):
        _, label, _ = dataset[idx]
        label_to_indices[label.item()].append(idx)
    
    # Initialize projection head if provided
    if projection_head is not None:
        optimizer = optim.Adam(projection_head.parameters(), lr=learning_rate)
        device = next(projection_head.parameters()).device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize triplet loss
    criterion = TripletLoss(margin=margin)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        triplet_count = 0
        
        # Process batches
        for batch_embeddings, batch_labels, _ in tqdm(dataloader):
            batch_size = batch_embeddings.size(0)
            
            # Skip small batches
            if batch_size <= 1:
                continue
            
            if projection_head is not None:
                # Forward through projection head if provided
                batch_embeddings = projection_head(batch_embeddings)
            
            # Process each anchor in the batch
            batch_loss = 0
            valid_triplets = 0
            
            for i in range(batch_size):
                anchor_embedding = batch_embeddings[i].unsqueeze(0)
                anchor_label = batch_labels[i].item()
                
                # Find all positive indices (same class as anchor)
                # Excluding the anchor itself
                positive_indices = [idx for idx in label_to_indices[anchor_label] if idx != i]
                
                # Find all negative indices (different class from anchor)
                negative_classes = [label for label in label_to_indices.keys() if label != anchor_label]
                
                # Skip if no valid triplets can be formed
                if not positive_indices or not negative_classes:
                    continue
                
                # Generate multiple triplets per anchor
                for _ in range(min(triplets_per_anchor, len(positive_indices))):
                    # Randomly select a positive
                    pos_idx = random.choice(positive_indices)
                    pos_embedding, _, _ = dataset[pos_idx]
                    pos_embedding = pos_embedding.unsqueeze(0)
                    
                    # Randomly select a negative class and then a negative example
                    neg_class = random.choice(negative_classes)
                    neg_idx = random.choice(label_to_indices[neg_class])
                    neg_embedding, _, _ = dataset[neg_idx]
                    neg_embedding = neg_embedding.unsqueeze(0)
                    
                    # Apply projection head if needed
                    if projection_head is not None:
                        pos_embedding = projection_head(pos_embedding)
                        neg_embedding = projection_head(neg_embedding)
                    
                    # Compute loss for this triplet
                    triplet_loss = criterion(anchor_embedding, pos_embedding, neg_embedding)
                    batch_loss += triplet_loss
                    valid_triplets += 1
            
            # Skip batch if no valid triplets
            if valid_triplets == 0:
                continue
            
            # Normalize loss by number of triplets
            batch_loss = batch_loss / valid_triplets
            
            # Backward and optimize if we have a projection head
            if projection_head is not None:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            # Update statistics
            running_loss += batch_loss.item()
            triplet_count += 1
        
        # Print epoch statistics
        if triplet_count > 0:
            epoch_loss = running_loss / triplet_count
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{epochs}, No valid triplets found')
    
    return projection_head if projection_head is not None else None


processor, model = get_dino_processor_and_model('facebook/dinov2-base')


dataset = EmbeddingDataset(metadata, '/kaggle/input/animal-clef-2025', processor, model)


proj_head = ProjectionHead(768, 128).to("cuda" if torch.cuda.is_available() else "cpu")
proj_head


head = train_with_online_triplet_generation(dataset, projection_head=proj_head)


torch.save(head.state_dict(), '/kaggle/working/head.pt')


state_dict = torch.load('/kaggle/working/head.pt')
head = ProjectionHead(768, 128).to("cuda" if torch.cuda.is_available() else "cpu")
head.load_state_dict(state_dict)


dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)


embeddings_df = []


head.eval()
with torch.no_grad():
    for batch_embeddings, batch_labels, batch_targets in tqdm(dataloader):
        out_embeddings = head(batch_embeddings).detach().cpu().numpy()
        out_labels = batch_labels.detach().cpu().numpy()
        out_targets = np.array(batch_targets)

        out = np.concatenate((out_targets.reshape(-1, 1), out_labels.reshape(-1, 1), out_embeddings), axis=1)
        if out.shape != (32, 130):
            print(f'Error! out.shape = {out.shape}')
            
        embeddings_df.append(out)


embeddings_df = np.concatenate(embeddings_df, axis=0)


columns = ['identity', 'label'] + [f'embed_dim{i}' for i in range(len(embeddings_df[0]) - 2)]
embeddings_df = pd.DataFrame(embeddings_df, columns=columns)


embeddings_df


embeddings_df.to_csv('/kaggle/working/triplet_dataset_embeddings.csv', index=False)




