# https://www.kaggle.com/code/shujun717/ribonanzanet-3d-finetune
# https://www.deepseek.com/


from IPython.display import Image
Image(filename='/kaggle/input/examples/example_291.png')


Image(filename='/kaggle/input/examples/example_639.png')


Image(filename='/kaggle/input/examples/example_705.png')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from tqdm import tqdm
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.vision.all import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


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


train_sequences=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")


train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
train_labels["pdb_id"]


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


all_xyz = []
for pdb_id in tqdm(train_sequences['target_id']):
    df = train_labels[train_labels["pdb_id"]==pdb_id]
    xyz = df[['x_1','y_1','z_1']].to_numpy().astype('float32')
    xyz[xyz < -1e17] = np.nan
    all_xyz.append(xyz)
print('Done!')


def angle_between_vectors(u, v):
#   Compute the dot product
    dot_product = np.dot(u, v)    
#   Compute the magnitudes of the vectors
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)    
#   Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_u * magnitude_v)    
#   Compute the angle in radians
    theta = np.arccos(cos_theta)
    
    return theta


def givens_rotation(M, k, i, j):
    """
    Compute the ij Givens rotation matrix to zero out M[k, i].
    """
    a = M[k, i]
    b = M[k, j]
    r = np.hypot(a, b)
    
    G = np.eye(3)
    if r == 0: return G
    
#   Compute c and s based on the order of i and j
    if i > j:
        c = b / r
        s = a / r
    else:
        c = a / r
        s = -b / r
    
#   Construct the Givens rotation matrix
    G[i, i] = c
    G[j, j] = c
    G[i, j] = s
    G[j, i] = -s
    
    return G

# Example matrix
M = np.random.rand(3,3)*5

print("Original matrix M:")
print(M)

# Zero out the top triangle
M = M @ givens_rotation(M, 0, 0, 1)
M = M @ givens_rotation(M, 0, 0, 2)
M = M @ givens_rotation(M, 1, 1, 2)
    
print("\nMatrix after zeroing top triangle:")
print(M)


filter_nan = []
max_len = 0
for xyz in all_xyz:
    if len(xyz) > max_len:
        max_len = len(xyz)

    filter_nan.append((np.isnan(xyz).mean() <= 0.5) & \
                      (len(xyz)<config['max_len_filter']) & \
                      (len(xyz)>config['min_len_filter']))

print(f"Longest sequence in train: {max_len}")

filter_nan = np.array(filter_nan)
non_nan_indices = np.arange(len(filter_nan))[filter_nan]

train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
all_xyz = [all_xyz[i] for i in non_nan_indices]


def plt_xyz(xyz,gt,A='prediction',B='gt',size=5,color=None,colorscale='Viridis',opacity=.8):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
            mode='markers',
            name = A,
            marker=dict(
                size=size,
                color=color,
                colorscale=colorscale,
                opacity=opacity
            )
        ),
        go.Scatter3d(
            x=gt[:,0], y=gt[:,1], z=gt[:,2],
            mode='lines',
            name=B,
            marker=dict(
                size=size,
                colorscale=colorscale,
                opacity=opacity
            )
        )])

    fig.show() 


def check_reconstruction(reconstructed,original,display=False):
    mask = (reconstructed.isnan().any(1) + original.isnan().any(1)) == 0
    reconstructed = reconstructed[mask]
    original = original[mask]
    reconstructed -= reconstructed.mean(0)
    original -= original.mean(0)

    cov_matrix = reconstructed.T @ original
    U, S, Vt = torch.svd(cov_matrix)
    R = Vt @ U.T
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T
    reconstructed = reconstructed @ R.T
    
    if display: plt_xyz(reconstructed,original,A='reconstucted',B='original',color=reconstructed[:,2])
    rmsd = ((original - reconstructed)**2).sum(1).mean().sqrt()
    return rmsd


def calculate_internal_coordinates(coords):
    n = coords.shape[0]
    bonds = []
    angles = []
    torsions = []
    
    # Bond lengths
    for i in range(1, n):
        bonds.append(np.linalg.norm(coords[i] - coords[i-1]))
    
    # Bond angles
    for i in range(2, n):
        v1 = coords[i-2] - coords[i-1]
        v2 = coords[i] - coords[i-1]
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angles.append(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    
    # Torsion angles (sign corrected)
    for i in range(3, n):
        p1, p2, p3, p4 = coords[i-3], coords[i-2], coords[i-1], coords[i]
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        n1 = np.cross(v2, v1)
        n2 = np.cross(v3, v2)
        
        n1 /= np.linalg.norm(n1) + 1e-8  # Avoid division by zero
        n2 /= np.linalg.norm(n2) + 1e-8
        v2 /= np.linalg.norm(v2) + 1e-8
        
        m1 = np.cross(n1, v2)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        torsion = np.degrees(np.arctan2(y, x))
        torsions.append(-torsion)  # Invert the sign here
    
    return bonds, angles, torsions


def reconstruct_from_internal(bonds, angles, torsions):
    n = len(bonds) + 1
    coords = np.zeros((n, 3))
    
    # Fix first three atoms in reference frame
    if n >= 1: coords[0] = [0, 0, 0]
    if n >= 2: coords[1] = [bonds[0], 0, 0]
    if n >= 3:
        theta = np.radians(180 - angles[0])
        x = bonds[1] * np.cos(theta) + coords[1][0]
        y = bonds[1] * np.sin(theta)
        coords[2] = [x, y, 0]
    
    # Build subsequent atoms (corrected m_vec calculation)
    for i in range(3, n):
        a, b, c = coords[i-3], coords[i-2], coords[i-1]
        bond = bonds[i-1]
        angle = np.radians(angles[i-2])
        torsion = np.radians(torsions[i-3])
        
        # Local coordinate system
        bc = c - b
        bc_norm = bc / np.linalg.norm(bc)
        n_vec = np.cross(b - a, bc)
        n_norm = n_vec / np.linalg.norm(n_vec)
        m_vec = np.cross(n_norm, bc_norm)  # Corrected cross product order
        
        # Displacement components
        dx = -bond * np.cos(angle)
        dy = bond * np.sin(angle) * np.cos(torsion)
        dz = bond * np.sin(angle) * np.sin(torsion)
        
        # Global displacement
        displacement = dx * bc_norm + dy * m_vec + dz * n_norm
        coords[i] = c + displacement
    
    return coords


def align_to_reference(coords):
    aligned = coords.copy().astype(float)
#   reference search
    for k in range(len(aligned)-3):
        if np.isnan(aligned[k:k+3]).sum() == 0: break
#   Translate first atom to origin
    aligned -= aligned[k]
    
    if len(coords) >= 2:
#       Rotate second atom to x-axis
        aligned = aligned@givens_rotation(aligned, k+1, 1, 0)
        aligned = aligned@givens_rotation(aligned, k+1, 2, 0)
        
    if len(coords) >= 3:
#       Rotate third atom into xy-plane
        aligned = aligned@givens_rotation(aligned, k+2, 2, 0)
#   Pointing check
    if aligned[1,0] < 0:  aligned[:,[0,2]] *= -1
    if aligned[2,1] < 0:  aligned[:,1:] *= -1
    
    return aligned,k


#i = np.random.randint(len(all_xyz))
i = 291#639,675,291,705 Intereasting structures
xyz = all_xyz[i].copy()
aligned,k = align_to_reference(xyz)
bonds, angles, torsions = calculate_internal_coordinates(aligned[k:])
reconstructed_coords = reconstruct_from_internal(bonds, angles, torsions)
rmsd = check_reconstruction(torch.tensor(all_xyz[i][k:]).float(), torch.tensor(reconstructed_coords).float(),display=True)
print(f"Reconstruction RMSD: {rmsd:.6f} Å")


all_k = []
all_bonds = []
all_angles = []
all_torsions = []
for xyz in tqdm(all_xyz):
    xyz,k = align_to_reference(xyz)
    bonds, angles, torsions = calculate_internal_coordinates(xyz[k:])
    all_k.append(k)
    all_bonds.append(bonds)
    all_angles.append(angles)
    all_torsions.append(torsions)


# Bond distances AB
plt.hist(np.concatenate([np.array(b) for b in all_bonds]),100)[2]


# Angles ABC
plt.hist(np.concatenate([np.array(a) for a in all_angles]),100)[2]


# Torsion angles ABCD
plt.hist(np.concatenate([np.array(t) for t in all_torsions]),100)[2]

