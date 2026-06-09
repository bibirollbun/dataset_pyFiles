import torch
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter": 10, 
    "structural_violation_epoch": 50,
    "balance_weight": False,
}


# Load data

train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
train_labels["pdb_id"] 


all_xyz = []

for pdb_id in tqdm(train_sequences['target_id']):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    xyz = df[['x_1','y_1','z_1']].to_numpy().astype('float32')
    if not np.isnan(xyz).any(): xyz[xyz<-1e17] = float('Nan')

    all_xyz.append(xyz)

df


# filter the data
# Filter and process data
filter_nan = []
max_len = 0
for xyz in all_xyz:
    if len(xyz) > max_len:
        max_len = len(xyz)

    #fill -1e18 masked sequences to nans

    #sugar_xyz = np.stack([nt_xyz['sugar_ring'] for nt_xyz in xyz], axis=0)
    filter_nan.append((np.isnan(xyz).mean() <= 0.5) & \
                      (len(xyz)<config['max_len_filter']) & \
                      (len(xyz)>config['min_len_filter']))

print(f"Longest sequence in train: {max_len}")

filter_nan = np.array(filter_nan)
non_nan_indices = np.arange(len(filter_nan))[filter_nan]

train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
all_xyz=[all_xyz[i] for i in non_nan_indices]


#pack data into a dictionary

data={
    "sequence":train_sequences['sequence'].to_list(),
    "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
    "description": train_sequences['description'].to_list(),
    "all_sequences": train_sequences['all_sequences'].to_list(),
    "xyz": all_xyz
}


# Split data into train and test
all_index = np.arange(len(data['sequence']))
cutoff_date = pd.Timestamp(config['cutoff_date'])
test_cutoff_date = pd.Timestamp(config['test_cutoff_date'])
train_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) <= cutoff_date]
test_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) > cutoff_date and pd.Timestamp(d) <= test_cutoff_date]


print(f"Train size: {len(train_index)}")
print(f"Test size: {len(test_index)}")


from torch.utils.data import Dataset, DataLoader

class RNA3D_Dataset(Dataset):
    def __init__(self,indices,data):
        self.indices=indices
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx=self.indices[idx]
        sequence=[self.tokens[nt] for nt in (self.data['sequence'][idx])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        #get C1' xyz
        xyz=self.data['xyz'][idx]
        xyz=torch.tensor(np.array(xyz))

        if len(sequence)>config['max_len']:
            crop_start = np.random.randint(len(sequence)-config['max_len'])
            crop_end = crop_start+config['max_len']

            sequence = sequence[crop_start:crop_end]
            xyz = xyz[crop_start:crop_end]

        return {
            'sequence': sequence,
            'xyz': xyz,
        }

train_dataset = RNA3D_Dataset(train_index,data)
val_dataset = RNA3D_Dataset(test_index,data)


import sys

sys.path.append("/kaggle/input/ribonanzanet2d-final")

from Network import *
import yaml

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout=0.1
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained: self.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt",map_location='cpu'))

        self.xyz_predictor=nn.Linear(256,3)

    def forward(self, src, mask):
        sequence_features, pairwise_features=self.get_embeddings(src, mask)
        xyz=self.xyz_predictor(sequence_features)

        return xyz


def calculate_distance_matrix(X,Y,epsilon=1e-4):
    return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()

def dRMSD(pred_x,
          gt_x,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_x)
    gt_dm=calculate_distance_matrix(gt_x,gt_x)

    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    if d_clamp is not None:
        rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
    else:
        rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

    return rmsd.sqrt().mean()/Z

def local_dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=30):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)

    mask=(~torch.isnan(gt_dm))*(gt_dm<d_clamp)
    mask[torch.eye(mask.shape[0]).bool()]=False

    rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon
    # rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).sqrt()/Z
    #rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])/Z
    return rmsd.sqrt().mean()/Z

def dRMAE(pred_x, gt_x, mask, epsilon=1e-4, Z=10, d_clamp=None):
    pred_dm = torch.cdist(pred_x, pred_x, p=2) * mask
    gt_dm = torch.cdist(gt_x, gt_x, p=2) * mask

    mask = ~torch.isnan(gt_dm)  
    diff = torch.abs(pred_dm - gt_dm)[mask].mean()

    return diff / Z 

def align_svd_mae(input, target, mask_, Z=10):
    """
    Aligns the input (Nx3) to target (Nx3) using SVD-based Procrustes alignment
    and computes RMSD loss.
    
    Args:
        input (torch.Tensor): Nx3 tensor representing the input points.
        target (torch.Tensor): Nx3 tensor representing the target points.
    
    Returns:
        aligned_input (torch.Tensor): Nx3 aligned input.
        rmsd_loss (torch.Tensor): RMSD loss.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    input=input[mask_ == 1]
    target=target[mask_ == 1]
    
    #mask 
    mask=~torch.isnan(target.sum(-1))

    input=input[mask]
    target=target[mask]
    
    # Compute centroids
    centroid_input = input.mean(dim=0, keepdim=True)
    centroid_target = target.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input - centroid_input.detach()
    target_centered = target - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)

    # Compute rotation matrix
    R = Vt @ U.T

    # Ensure a proper rotation (det(R) = 1, no reflection)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T

    # Rotate input
    aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()

    # # Compute RMSD loss
    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())

    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())
    
    # return aligned_input, rmsd_loss
    return torch.abs(aligned_input-target).mean()/Z


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR

class RibonanzaModel(pl.LightningModule):
    def __init__(self, model, cos_epoch=35, epochs=50):
        super().__init__()
        self.model = model
        self.cos_epoch = cos_epoch
        self.epochs = epochs
        self.automatic_optimization = False
        
    def training_step(self, batch, batch_idx):
        sequence = batch['sequence']
        mask = batch['mask']

        pred_xyz = self.model(sequence, mask)
        gt_xyz = batch['xyz']

        row = mask.unsqueeze(2).expand(-1, -1, mask.size(1))
        col = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        mask_ = (row & col)

        loss1 = dRMAE(pred_xyz, gt_xyz, mask_)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            loss2 = align_svd_mae(pred_xyz[0], gt_xyz[0], mask[0])

        loss = loss1 + loss2
        self.manual_backward(loss)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        opt = self.optimizers()
        opt.step()
        opt.zero_grad()
        
        if self.current_epoch >= self.cos_epoch:
            sch = self.lr_schedulers()
            sch.step()

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequence = batch['sequence']
        mask = batch['mask']

        pred_xyz = self.model(sequence, mask)
        gt_xyz = batch['xyz']

        row = mask.unsqueeze(2).expand(-1, -1, mask.size(1))
        col = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        mask_ = (row & col)

        loss = dRMAE(pred_xyz, gt_xyz, mask_)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(self.epochs - self.cos_epoch) * self.trainer.estimated_stepping_batches
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def configure_callbacks(self):
        return [
            ModelCheckpoint(monitor='val_loss', filename='best', save_top_k=1),
            ModelCheckpoint(filename='last', save_last=True),
        ]

model = finetuned_RibonanzaNet(
    load_config_from_yaml("/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml"),
    pretrained=True
)

batch_size = 2

def collate_fn(batch):
    mlen = max([ len(elem['sequence']) for elem in batch ])

    for elem in batch:
        mask = torch.zeros(mlen).long()
        mask[:len(elem['sequence'])] = 1

        elem['mask'] = mask
        elem['xyz'] = torch.nn.functional.pad(elem['xyz'], (0, 0, 0, mlen-len(elem['sequence'])))
        elem['sequence'] = torch.nn.functional.pad(elem['sequence'], (0, mlen-len(elem['sequence'])))

    batch_ = {
        'mask': torch.stack([ elem['mask'] for elem in batch ]),
        'sequence': torch.stack([ elem['sequence'] for elem in batch ]),
        'xyz': torch.stack([ elem['xyz'] for elem in batch ]),
    }

    return batch_

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices=1,
    precision='bf16-mixed',
)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

plmodel = RibonanzaModel(model)
trainer.fit(plmodel, train_loader, val_loader)

