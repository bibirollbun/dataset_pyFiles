import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
import os
import sys
from tqdm import tqdm


config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 256,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 1,
    "cos_epoch": 0,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter":10,
    "structural_violation_epoch": 50,
    "balance_weight": False,
    "n_times": 1000,
}


# train_sequences=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.v2.csv")
# train_labels=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv")


# train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
# train_labels["pdb_id"] 


# # In[6]:


# float('Nan')


# # In[7]:


# all_xyz=[]

# for pdb_id in tqdm(train_sequences['target_id']):
#     df = train_labels[train_labels["pdb_id"]==pdb_id]
#     #break
#     xyz=df[['x_1','y_1','z_1']].to_numpy().astype('float32')
#     xyz[xyz<-1e17]=float('Nan');
#     all_xyz.append(xyz)


# df


# # In[8]:


# # filter the data
# # Filter and process data
# filter_nan = []
# max_len = 0
# for xyz in all_xyz:
#     if len(xyz) > max_len:
#         max_len = len(xyz)

#     #fill -1e18 masked sequences to nans
    
#     #sugar_xyz = np.stack([nt_xyz['sugar_ring'] for nt_xyz in xyz], axis=0)
#     filter_nan.append((np.isnan(xyz).mean() <= 0.5) & \
#                       (len(xyz)<config['max_len_filter']) & \
#                       (len(xyz)>config['min_len_filter']))

# print(f"Longest sequence in train: {max_len}")

# filter_nan = np.array(filter_nan)
# non_nan_indices = np.arange(len(filter_nan))[filter_nan]

# train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
# all_xyz=[all_xyz[i] for i in non_nan_indices]


# # In[9]:


# #pack data into a dictionary

# data={
#       "sequence":train_sequences['sequence'].to_list(),
#       "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
#       "description": train_sequences['description'].to_list(),
#       "all_sequences": train_sequences['all_sequences'].to_list(),
#       "xyz": all_xyz
# }

#laod preprocessed pickle file
with open("/kaggle/input/stanford-3d-train-v2-pickle/train_data.pkl", "rb") as f:
    data = pickle.load(f)


# Split data into train and test
all_index = np.arange(len(data['sequence']))
cutoff_date = pd.Timestamp(config['cutoff_date'])
test_cutoff_date = pd.Timestamp(config['test_cutoff_date'])
train_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) <= cutoff_date]
test_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) > cutoff_date and pd.Timestamp(d) <= test_cutoff_date]


# In[11]:


print(f"Train size: {len(train_index)}")
print(f"Test size: {len(test_index)}")


from torch.utils.data import Dataset, DataLoader
from ast import literal_eval


from collections import defaultdict

class RNA3D_Dataset(Dataset):
    def __init__(self,indices,data):
        self.indices=indices
        self.data=data
        #set default to 4
        self.tokens=defaultdict(lambda: 4)
        self.tokens['A']=0
        self.tokens['C']=1
        self.tokens['G']=2
        self.tokens['U']=3

        #{nt:i for i,nt in enumerate('ACGU')}

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
            crop_start=np.random.randint(len(sequence)-config['max_len'])
            crop_end=crop_start+config['max_len']

            sequence=sequence[crop_start:crop_end]
            xyz=xyz[crop_start:crop_end]
        
        #center at first atom if first atom does not exit go until it does
        for i in range(len(xyz)):
            if (~torch.isnan(xyz[i])).all():
                break
        xyz=xyz-xyz[i]

        # for i in range(len(xyz)):

        #     if torch.isnan(xyz[i]).any():
        #         if i==0:
        #             xyz[i]=xyz[i+1]
        #         else:
        #             xyz[i]=xyz[i-1]

        return {'sequence':sequence,
                'xyz':xyz}



train_dataset=RNA3D_Dataset(train_index,data)
val_dataset=RNA3D_Dataset(test_index,data)

train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)



sys.path.append("/kaggle/input/ribonanzanet2/pytorch/alpha/1")

import torch.nn as nn
from Network import *

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, rnet_config, config, pretrained=False):
        rnet_config.dropout=0.1
        rnet_config.use_grad_checkpoint=True
        super(finetuned_RibonanzaNet, self).__init__(rnet_config)
        if pretrained:
            self.load_state_dict(torch.load(config.pretrained_weight_path,map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)

        decoder_dim=config.decoder_dim
        self.structure_module=[SimpleStructureModule(d_model=decoder_dim, nhead=config.decoder_nhead, 
                 dim_feedforward=decoder_dim*4, pairwise_dimension=rnet_config.pairwise_dimension, dropout=0.0) for i in range(config.decoder_num_layers)]
        self.structure_module=nn.ModuleList(self.structure_module)

        self.xyz_embedder=nn.Linear(3,decoder_dim)
        self.xyz_norm=nn.LayerNorm(decoder_dim)
        self.xyz_predictor=nn.Linear(decoder_dim,3)
        
        self.adaptor=nn.Sequential(nn.Linear(rnet_config.ninp,decoder_dim),nn.LayerNorm(decoder_dim))

        self.distogram_predictor=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
                                                nn.Linear(rnet_config.pairwise_dimension,40))

        self.time_embedder=SinusoidalPosEmb(decoder_dim)

        self.time_mlp=nn.Sequential(nn.Linear(decoder_dim,decoder_dim),
                                    nn.ReLU(),  
                                    nn.Linear(decoder_dim,decoder_dim))
        self.time_norm=nn.LayerNorm(decoder_dim)

        self.distance2pairwise=nn.Linear(1,rnet_config.pairwise_dimension,bias=False)

        self.pair_mlp=nn.Sequential(nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension),
                                    nn.ReLU(),
                                    nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension))


        #hyperparameters for diffusion
        self.n_times = config.n_times

        #self.model = model
        
        # define linear variance schedule(betas)
        beta_1, beta_T = config.beta_min, config.beta_max
        betas = torch.linspace(start=beta_1, end=beta_T, steps=config.n_times)#.to(device) # follows DDPM paper
        self.sqrt_betas = torch.sqrt(betas)
                                     
        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.data_std=config.data_std


    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward
    
    def embed_pair_distance(self,inputs):
        pairwise_features,xyz=inputs
        distance_matrix=xyz[:,None,:,:]-xyz[:,:,None,:]
        distance_matrix=(distance_matrix**2).sum(-1).clip(2,37**2).sqrt()
        distance_matrix=distance_matrix[:,:,:,None]
        pairwise_features=pairwise_features+self.distance2pairwise(distance_matrix)

        return pairwise_features

    def forward(self,src,xyz,t):
        
        #with torch.no_grad():
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        
        distogram=self.distogram_predictor(pairwise_features)

        sequence_features=self.adaptor(sequence_features)

        decoder_batch_size=xyz.shape[0]
        sequence_features=sequence_features.repeat(decoder_batch_size,1,1)
        

        pairwise_features=pairwise_features.expand(decoder_batch_size,-1,-1,-1)

        pairwise_features= checkpoint.checkpoint(self.custom(self.embed_pair_distance), [pairwise_features,xyz],use_reentrant=False)

        time_embed=self.time_embedder(t).unsqueeze(1)
        tgt=self.xyz_norm(sequence_features+self.xyz_embedder(xyz)+time_embed)

        tgt=self.time_norm(tgt+self.time_mlp(tgt))

        for layer in self.structure_module:
            #tgt=layer([tgt, sequence_features,pairwise_features,xyz,None])
            tgt=checkpoint.checkpoint(self.custom(layer),
            [tgt, sequence_features,pairwise_features,xyz,None],
            use_reentrant=False)
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        
        xyz=self.xyz_predictor(tgt).squeeze(0)
        #.squeeze(0)

        return xyz, distogram
    

    def denoise(self,sequence_features,pairwise_features,xyz,t):
        decoder_batch_size=xyz.shape[0]
        sequence_features=sequence_features.expand(decoder_batch_size,-1,-1)
        pairwise_features=pairwise_features.expand(decoder_batch_size,-1,-1,-1)

        pairwise_features=self.embed_pair_distance([pairwise_features,xyz])

        sequence_features=self.adaptor(sequence_features)
        time_embed=self.time_embedder(t).unsqueeze(1)
        tgt=self.xyz_norm(sequence_features+self.xyz_embedder(xyz)+time_embed)
        tgt=self.time_norm(tgt+self.time_mlp(tgt))
        #xyz_batch_size=xyz.shape[0]
        


        for layer in self.structure_module:
            tgt=layer([tgt, sequence_features,pairwise_features,xyz,None])
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        xyz=self.xyz_predictor(tgt).squeeze(0)
        # print(xyz.shape)
        # exit()
        return xyz


    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1
    
    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5
    
    def make_noisy(self, x_zeros, t): 
        # assume we get raw data, so center and scale by 35
        x_zeros = x_zeros - torch.nanmean(x_zeros,1,keepdim=True)
        x_zeros = x_zeros/self.data_std
        #rotate randomly
        x_zeros = random_rotation_point_cloud_torch_batch(x_zeros)


        # perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)
        epsilon = torch.randn_like(x_zeros).to(x_zeros.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars.to(x_zeros.device), t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars.to(x_zeros.device), t, x_zeros.shape)
        
        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
    
        return noisy_sample.detach(), epsilon
    
    
    # def forward(self, x_zeros):
    #     x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        
    #     B, _, _, _ = x_zeros.shape
        
    #     # (1) randomly choose diffusion time-step
    #     t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(x_zeros.device)
        
    #     # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
    #     perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        
    #     # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
    #     pred_epsilon = self.model(perturbed_images, t)
        
    #     return perturbed_images, epsilon, pred_epsilon
    
    
    def denoise_at_t(self, x_t, sequence_features, pairwise_features, timestep, t):
        B, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(sequence_features.device)
        else:
            z = torch.zeros_like(x_t).to(sequence_features.device)
        
        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred = self.denoise(sequence_features, pairwise_features, x_t, timestep)
        
        alpha = self.extract(self.alphas.to(x_t.device), timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas.to(x_t.device), timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars.to(x_t.device), timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas.to(x_t.device), timestep, x_t.shape)
        
        # denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
        
        return x_t_minus_1#.clamp(-1., 1)
                
    def sample(self, src, N):
        # start from random noise vector, NxLx3
        x_t = torch.randn((N, src.shape[1], 3)).to(src.device)
        
        # autoregressively denoise from x_T to x_0
        #     i.e., generate image from noise, x_T

        #first get conditioning
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        # sequence_features=sequence_features.expand(N,-1,-1)
        # pairwise_features=pairwise_features.expand(N,-1,-1,-1)
        distogram=self.distogram_predictor(pairwise_features).squeeze()
        distogram=distogram.squeeze()[:,:,2:40]*torch.arange(2,40).float().cuda() 
        distogram=distogram.sum(-1)  

        for t in range(self.n_times-1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(src.device)
            x_t = self.denoise_at_t(x_t, sequence_features, pairwise_features, timestep, t)
        
        # denormalize x_0 into 0 ~ 1 ranged values.
        #x_0 = self.reverse_scale_to_zero_to_one(x_t)
        x_0 = x_t * self.data_std
        return x_0, distogram




class SimpleStructureModule(nn.Module):

    def __init__(self, d_model, nhead, 
                 dim_feedforward, pairwise_dimension, dropout=0.1,
                 ):
        super(SimpleStructureModule, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)
        #self.cross_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.pairwise2heads=nn.Linear(pairwise_dimension,nhead,bias=False)
        self.pairwise_norm=nn.LayerNorm(pairwise_dimension)

        #self.distance2heads=nn.Linear(1,nhead,bias=False)
        #self.pairwise_norm=nn.LayerNorm(pairwise_dimension)

        self.activation = nn.GELU()

        
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward

    def forward(self, input):
        tgt , src,  pairwise_features, pred_t, src_mask = input
        
        #src = src*src_mask.float().unsqueeze(-1)

        pairwise_bias=self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)

        


        #print(pairwise_bias.shape,distance_bias.shape)

        #pairwise_bias=pairwise_bias+distance_bias


        res=tgt
        tgt,attention_weights = self.self_attn(tgt, tgt, tgt, mask=pairwise_bias, src_mask=src_mask)
        tgt = res + self.dropout1(tgt)
        tgt = self.norm1(tgt)

        # print(tgt.shape,src.shape)
        # exit()

        res=tgt
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = res + self.dropout2(tgt)
        tgt = self.norm2(tgt)


        return tgt


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





def random_rotation_point_cloud_torch_batch(point_clouds):
    """
    Apply a random 3D rotation to a batch of point clouds (PyTorch version).
    
    Args:
        point_clouds (torch.Tensor): BxNx3 tensor of XYZ points.

    Returns:
        torch.Tensor: Rotated BxNx3 point clouds.
    """
    B, N, _ = point_clouds.shape
    device = point_clouds.device

    # Generate a batch of random orthonormal rotation matrices
    A = torch.randn(B, 3, 3, device=device)
    Q, R = torch.linalg.qr(A)

    # Ensure det(Q) = +1 for proper rotation
    det = torch.det(Q)
    Q[det < 0, :, 0] *= -1

    # Apply batched matrix multiplication
    rotated = torch.matmul(point_clouds, Q.transpose(1, 2))  # (B, N, 3) x (B, 3, 3)^T -> (B, N, 3)

    return rotated




diffusion_config=load_config_from_yaml("/kaggle/input/ribonanzanet2-ddpm-v2/diffusion_config.yaml")
rnet_config=load_config_from_yaml("/kaggle/input/ribonanzanet2/pytorch/alpha/1/pairwise.yaml")

model=finetuned_RibonanzaNet(rnet_config,diffusion_config).cuda()


from tqdm import tqdm

epochs=config['epochs']
cos_epoch=config['cos_epoch']


best_loss=np.inf
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=0.0002) #no weight decay following AF

batch_size=1

#for cycle in range(2):

criterion=torch.nn.CrossEntropyLoss(reduction='none')

#scaler = GradScaler()


schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs-cos_epoch)*len(train_loader)//batch_size)



def calculate_distance_matrix(X,Y,epsilon=1e-4):
    return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()

def dRMAE(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])

    return rmsd.mean()/Z

def align_svd_rmsd(input, target):
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
    return torch.square(aligned_input-target).mean().sqrt()

def compute_lddt(ground_truth_atoms, predicted_atoms, cutoff=30.0, thresholds=[1.0, 2.0, 4.0, 8.0]):
    """
    Computes the lDDT score between ground truth and predicted atoms.
    
    Parameters:
        ground_truth_atoms (np.array): Nx3 array of ground truth atom coordinates.
        predicted_atoms (np.array): Nx3 array of predicted atom coordinates.
        cutoff (float): Distance cutoff in Ångstroms to consider neighbors. Default is 30 Å.
        thresholds (list): List of thresholds in Ångstroms for the lDDT computation. Default is [0.5, 1.0, 2.0, 4.0].
    
    Returns:
        float: The lDDT score.
    """
    # Number of atoms
    num_atoms = ground_truth_atoms.shape[0]
    
    # Initialize array to store lDDT fractions for each threshold
    fractions = np.zeros(len(thresholds))
    
    for i in range(num_atoms):
        # Get the distances from atom i to all other atoms for both ground truth and predicted atoms
        gt_distances = np.linalg.norm(ground_truth_atoms[i] - ground_truth_atoms, axis=1)
        pred_distances = np.linalg.norm(predicted_atoms[i] - predicted_atoms, axis=1)
        
        # print(gt_distances)
        # print(pred_distances)
        # exit()
        # Apply the cutoff to consider only distances within the cutoff range
        mask = (gt_distances > 0) & (gt_distances < cutoff)
        
        # Calculate the absolute difference between ground truth and predicted distances
        distance_diff = np.abs(gt_distances[mask] - pred_distances[mask])

        # Filter out any NaN values from the distance difference calculation
        valid_mask = ~np.isnan(distance_diff)
        distance_diff = distance_diff[valid_mask]

        # Compute the fractions for each threshold
        for j, threshold in enumerate(thresholds):
            if len(distance_diff)>0:
                fractions[j] += np.mean(distance_diff < threshold)
    # print(fractions)
    # print(num_atoms)

    # Average the fractions over the number of atoms
    fractions /= num_atoms
    
    # The final lDDT score is the average of these fractions
    lddt_score = np.mean(fractions)
    
    return lddt_score


for epoch in range(epochs):
    model.train()
    tbar=tqdm(train_loader)
    total_loss=0
    total_distogram_loss=0
    oom=0
    for idx, batch in enumerate(tbar):
        #try:
        sequence=batch['sequence'].cuda()
        gt_xyz=batch['xyz'].squeeze().cuda()
        mask=~torch.isnan(gt_xyz)
        gt_xyz[torch.isnan(gt_xyz)]=0

        distance_matrix=calculate_distance_matrix(gt_xyz,gt_xyz)
        distogram_mask=distance_matrix==distance_matrix
        distance_matrix=distance_matrix.clip(2,39).long()



        gt_xyz=gt_xyz.unsqueeze(0).repeat(48,1,1)
        time_steps=torch.randint(0,config['n_times'],size=(gt_xyz.shape[0],)).to(gt_xyz.device)
        loss_weight=(1-time_steps/config['n_times'])
        noised_xyz, noise=model.make_noisy(gt_xyz, time_steps)

        #exit()
        pred_noise,distogram_pred=model(sequence,noised_xyz,time_steps)#.squeeze()
        #pred_xyz=aug_xyz[:,1:-1]+pred_displacements[:,1:-1]
        #exit()

        
        loss= torch.square(noise-pred_noise)*loss_weight[:,None,None]
        loss=loss[mask.repeat(48,1,1)].mean()
        #exit()
        
        distogram_loss=criterion(distogram_pred.squeeze()[distogram_mask],distance_matrix[distogram_mask]).mean()
        total_distogram_loss+=distogram_loss.item()


        if loss!=loss:
            stop

        
        #(loss/batch_size*len(gt_xyz)).backward()

        #accelerator.backward()

        ((loss+0.2*distogram_loss)/batch_size*len(gt_xyz)).backward()
        
        if (idx+1)%batch_size==0 or idx+1 == len(tbar):

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            # scaler.scale(loss/batch_size).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # scaler.step(optimizer)
            # scaler.update()

            
            if (epoch+1)>cos_epoch:
                schedule.step()
        #schedule.step()
        total_loss+=loss.item()
        
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)} Distogram Loss: {total_distogram_loss/(idx+1)}")
        #break
    # visualize_point_cloud_batch(pred_xyz)
    # visualize_point_cloud_batch(aug_xyz)


    
    total_loss=total_loss/len(tbar)
    
    tbar=tqdm(val_loader)
    model.eval()
    val_preds=[]
    val_loss=0
    val_rmsd=0
    val_lddt=0
    #unwrapped_diffusion=accelerator.unwrap_model(diffusion)
    #unwrapped_model=accelerator.unwrap_model(model)
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        gt_xyz=batch['xyz'].cuda().squeeze()
    
        with torch.no_grad():
            # if accelerator.dis
            #pred_xyz=model.module.decode(sequence,torch.ones_like(sequence).long().cuda()).squeeze()
            pred_xyz=model.sample(sequence,1)[0].squeeze(0)

            #pred_xyz=model(sequence)[-1].squeeze()
            loss=dRMAE(pred_xyz,pred_xyz,gt_xyz,gt_xyz)
    
        val_rmsd+=align_svd_rmsd(pred_xyz,gt_xyz)
        val_lddt+=compute_lddt(pred_xyz.cpu().numpy(),gt_xyz.cpu().numpy())
        val_loss+=loss
    
        val_preds.append([gt_xyz.cpu().numpy(),pred_xyz.cpu().numpy()])
    val_loss=val_loss/len(tbar)
    val_rmsd=val_rmsd/len(tbar)
    val_lddt=val_lddt/len(tbar)
    
    print(f"val loss: {val_loss}")
    print(f"val_rmsd: {val_rmsd}")
    print(f"val_lddt: {val_lddt}")


