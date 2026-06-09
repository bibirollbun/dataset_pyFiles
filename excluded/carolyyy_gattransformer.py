import numpy as np 
import pandas as pd 
import os
whl_file = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_fullname = os.path.join(dirname, filename)
        print(file_fullname)
        if file_fullname.startswith("/kaggle/input/pytorch-geometric-whl"):
            whl_file.append(file_fullname)


# install pytorch-geometric
import subprocess
import sys

up_filename = "/kaggle/input/pytorch-geometric-whl/whl_path"
for f in whl_file:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--find-links",up_filename, f])

import torch_geometric
import torch_scatter
print("Successfully installed!!!")


import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Linear
import matplotlib.pyplot as plt
from   matplotlib import colors
import random
import math
import cv2

class Config:
    def __init__(self):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # data load
        self.base_path = '/kaggle/input/arc-prize-2025/'  
        self.batch_size = 10
        self.aug_prob = 0.5  # data to augment
        # model config
        self.node_dim = 10  # number of 0-9
        self.hidden_dim = 128   
        self.rare_color_threshold = 0.2
        self.top_n = 10   # number of lines to connect in rare colors
        # train config
        self.epochs = 24
        self.lr = 0.00035
        self.weight_decay = 0.005
        self.log_every = 100  
        # loss weights config
        self.foreground_weight = 4   # loss weight of: Foreground prediction as background 
        self.bg_penalty_coeff= 50  # Prediction as full background
        self.cls_loss_weight = 20  # Node classification
        self.fg_ce_weight = 20    # only check in real color area 
        self.diversity_coeff = 10  # compare the color entropy
        self.single_color_penalty_coeff = 50  # one-color fill
        self.count_penalty_coeff = 10  # the types of colors predicted
        self.struct_loss_weight = 15  # structure 
        self.max_aspect_ratio = 6 # Predicted aspect ratio threshold
        self.penalty_coeff = 0.1 # Extreme aspect ratio penalty weight 
        self.hw_weight  = 0.001    # Width and height dimension 
        self.nodes_weight  =  0.001  # Total node count prediction
        self.shape_loss_weight = 0.001  # Shape

params = Config()

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# augment data
class ARCDataAugmenter:
    def __init__(self, aug_prob=params.aug_prob):
        # set the probability of each operation to execute
        self.prob = aug_prob  

    def random_noise(self, grid):
        """randomly replace a small number of colors in non-background cells"""
        if np.random.random() < self.prob:
            h, w = grid.shape
            total_pixels = h * w
            if total_pixels == 0:
                return grid
            # the max num of changed cells = 5
            num_noise = min(5, max(5, int(total_pixels * np.random.uniform(0.01, 0.02))))
            non_bg_mask = (grid != 0)
            if non_bg_mask.sum() < num_noise:
                return grid  
            # choose location randomly
            noise_coords = np.argwhere(non_bg_mask)
            selected_indices = np.random.choice(len(noise_coords), num_noise, replace=False)
            for idx in selected_indices:
                y, x = noise_coords[idx]
                original_color = grid[y, x]
                new_color = np.random.choice([c for c in range(10) if c != original_color])
                grid[y, x] = new_color
        return grid

    def random_flip(self, grid):
        """horizontal / vertical flip"""
        if np.random.random() < self.prob:
            if np.random.random() < 0.5:
                grid = np.fliplr(grid)  # horizontal
            else:
                grid = np.flipud(grid)  # vertical
        return grid

    def random_rotate90(self, grid):
        if np.random.random() < self.prob:
            grid = np.rot90(grid)
        return grid

    def __call__(self, grid):
        grid = self.random_rotate90(grid)
        grid = self.random_flip(grid)  
        grid = self.random_noise(grid)
        return grid


def build_graph(grid,rare_color_threshold=params.rare_color_threshold, top_n=params.top_n):
    """a graph that includes 3×3 local adjacency and same-color global adjacency to improve foreground feature"""
    H, W = grid.shape  
    N = H * W  # total nodes
    flat_grid = grid.flatten()  # shape: (N,)

    # node feature: original colors + color/ground label + location
    x_color = torch.tensor(flat_grid, dtype=torch.long).unsqueeze(1)  # (N, 1)

    # color/ground label:1=colors，0=background
    x_fg = torch.tensor((flat_grid > 0).astype(int), dtype=torch.float32).unsqueeze(1)  # (N, 1)

    # location to core foreground
    fg_coords = np.argwhere(grid > 0)  # shape: (foreground nodes, 2)
    if len(fg_coords) > 0:
        fg_center = fg_coords.mean(axis=0)  # [core row_id, core col_id]
        rel_pos_list = []
        for i in range(H):
            for j in range(W):
                rel_row = (i - fg_center[0]) / H
                rel_col = (j - fg_center[1]) / W
                rel_pos_list.append([rel_row, rel_col])
        x_rel_pos = torch.tensor(rel_pos_list, dtype=torch.float32)  # (N, 2)
    else:
        x_rel_pos = torch.zeros(N, 2, dtype=torch.float32)

    # add a global color histogram feature to each cell
    color_hist = np.bincount(flat_grid, minlength=10) / N  
    color_hist_feat = torch.tensor(color_hist, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
    # add local 3×3 color-distribution statistics to each cell
    local_color_feats = []
    grid_pad = np.pad(grid, 1, mode='constant', constant_values=0)
    for i in range(H):
        for j in range(W):
            local_patch = grid_pad[i:i+3, j:j+3].flatten()
            local_hist = np.bincount(local_patch, minlength=10) / 9.0
            local_color_feats.append(local_hist)
    local_color_feats = torch.tensor(np.array(local_color_feats), dtype=torch.float32)

    # concat features:(N, 4+10+10)：colors + class label + location + global color histogram feature + 3×3 color-distribution
    x = torch.cat([x_color, x_fg, x_rel_pos,color_hist_feat, local_color_feats], dim=1)  

    # edges structure
    edges = []
    edge_weights = []

    # 3*3 
    for i in range(H):
        for j in range(W):
            src = i * W + j  # set index 
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # skip self
                    ni, nj = i + di, j + dj  
                    if 0 <= ni < H and 0 <= nj < W:  
                        dst = ni * W + nj  
                        edges.append([src, dst])
                        distance = abs(di) + abs(dj)
                        edge_weights.append(1.0 if distance == 1 else 0.5) 

    # same-color connect
    unique_colors, color_counts = np.unique(flat_grid, return_counts=True)
    color_count_dict = dict(zip(unique_colors, color_counts))
    for color in unique_colors:
        if color == 0:  
            continue
        color_nodes = np.where(flat_grid == color)[0]  
        color_count = color_count_dict[color]
        color_ratio = color_count / N  # ratio of total nodes

        color_coords = []
        for idx in color_nodes:
            i = idx // W  # row_id
            j = idx % W   # col_id
            color_coords.append((i, j))

        # process rare colors in weight of 0.7
        if color_ratio < rare_color_threshold:
            for i in range(len(color_nodes)):
                for j in range(i + 1, len(color_nodes)):
                    src = color_nodes[i]
                    dst = color_nodes[j]
                    edges.append([src, dst])
                    edges.append([dst, src])
                    edge_weights.append(0.7)
                    edge_weights.append(0.7)
        # process other common colors: connect top_n nodes to generate lines
        else:
            for i in range(len(color_nodes)):
                src_idx = color_nodes[i]
                src_i, src_j = color_coords[i]
                
                distances = []
                for j in range(len(color_nodes)):
                    if i == j:
                        continue 
                    dst_i, dst_j = color_coords[j]
                    dist = np.sqrt((src_i - dst_i)**2 + (src_j - dst_j)** 2)
                    distances.append((dist, j)) 
                distances.sort()  # ascending
                for dist, j in distances[:top_n]:  # take top_n
                    dst_idx = color_nodes[j]
                    edges.append([src_idx, dst_idx])
                    edges.append([dst_idx, src_idx])
                    # process common colors in weight between 0.1 and 1.0
                    weight = max(0.1, 1.0 - dist / max(H, W))
                    edge_weights.extend([weight, weight])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)  # (E,)
    shape = torch.tensor([H, W], dtype=torch.long)  # return data shapes

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, shapes=shape)

class ARCGraphDataset(Dataset):
    def __init__(self, series='train', data_path=params.base_path):
        self.samples = []
        self.data_path = data_path  
        self.series = series
        # only use augmenter in train stage
        self.augmenter = ARCDataAugmenter() if series == 'train' else None
        
        if series == 'train':
            training_challenges = load_json(f"{self.data_path}arc-agi_training_challenges.json")
            for name, split in training_challenges.items():
                if 'train' not in split:
                    continue
                for ex in split['train']:
                    A_grid = np.array(ex['input'], dtype=np.int32)  
                    B_grid = np.array(ex['output'], dtype=np.int32)  
                    self._add_sample(A_grid, B_grid)
            
        elif series == 'val':
            eval_challenges = load_json(f"{self.data_path}arc-agi_evaluation_challenges.json")
            eval_solutions = load_json(f"{self.data_path}arc-agi_evaluation_solutions.json")
            
            for name in eval_challenges.keys():
                if name not in eval_solutions:
                    continue
                
                eval_input = eval_challenges[name]['test'][0]['input']  
                eval_output = eval_solutions[name][0]  
                
                A_grid = np.array(eval_input, dtype=np.int32)
                B_grid = np.array(eval_output, dtype=np.int32)
                self._add_sample(A_grid, B_grid)
        
        else:
            raise ValueError(f"series name error: {series}")

    def _add_sample(self, A_grid, B_grid):
        if self.series == 'train' and self.augmenter is not None:
            A_grid_augmented = self.augmenter(A_grid.copy())
            A_graph = build_graph(A_grid_augmented)
        else:
            A_graph = build_graph(A_grid)
        
        B_graph = build_graph(B_grid)
        
        if not hasattr(A_graph, 'x') or not hasattr(B_graph, 'x'):
            print(f"Warning: Graph object missing attributes. A_graph type: {type(A_graph)}, B_graph type: {type(B_graph)}")
            return
            
        self.samples.append({
            'A_graph': A_graph,
            'B_graph': B_graph,
            'A_grid': A_grid,
            'B_grid': B_grid
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 添加调试信息
        if not hasattr(sample['A_graph'], 'x'):
            print(f"Error: Sample {idx} A_graph is not a valid Data object")
            print(f"A_graph type: {type(sample['A_graph'])}")
            
        if not hasattr(sample['B_graph'], 'x'):
            print(f"Error: Sample {idx} B_graph is not a valid Data object")  
            print(f"B_graph type: {type(sample['B_graph'])}")
            
        return sample['A_graph'], sample['B_graph'], sample['A_grid'], sample['B_grid']

    def get_val_data(self):
        return [(s['A_graph'], s['B_graph'], s['A_grid'], s['B_grid']) for s in self.samples]

def collate_graphs(batch):
    A_graphs, B_graphs, A_grids, B_grids = zip(*batch)
    
    # concat A_graphs
    A_x_list = []
    A_edge_index_list = []
    A_edge_weight_list = []  
    A_shape_list = []
    for g in A_graphs:
        A_x_list.append(g.x.to(params.device))
        A_edge_index_list.append(g.edge_index.to(params.device))
        A_edge_weight_list.append(g.edge_weight.to(params.device))
        A_shape_list.append(g.shapes.to(params.device))
    
    A_x = torch.cat(A_x_list, dim=0)  # (sum(N_A),)
    # cumsum nodes to set offset
    A_node_counts = [x.shape[0] for x in A_x_list]
    A_cum_counts = torch.cumsum(torch.tensor([0] + A_node_counts[:-1], device=params.device), dim=0)
    A_edge_index = []
    for i, e in enumerate(A_edge_index_list):
        offset = A_cum_counts[i]
        A_edge_index.append(e + offset)
        
    A_edge_index = torch.cat(A_edge_index, dim=1)  # (2, sum(E_A))
    A_edge_weight = torch.cat(A_edge_weight_list, dim=0)  # (sum(E_A),)
    A_shapes = torch.stack(A_shape_list, dim=0)  # (batch_size, 2)
    # record belonged sample_id
    A_batch = []
    for i, count in enumerate(A_node_counts):
        A_batch.append(torch.full((count,), i, dtype=torch.long, device=params.device))
    A_batch = torch.cat(A_batch, dim=0)  # (sum(N_A),)
    
    # concat B_graphs
    B_x_list = []
    B_edge_index_list = []
    B_edge_weight_list = []
    B_shape_list = []
    for g in B_graphs:
        if g is not None:
            B_x_list.append(g.x.to(params.device))
            B_edge_index_list.append(g.edge_index.to(params.device))
            B_edge_weight_list.append(g.edge_weight.to(params.device))
            B_shape_list.append(g.shapes.to(params.device)) 
    
    if B_x_list:
        B_x = torch.cat(B_x_list, dim=0)
        B_node_counts = [x.shape[0] for x in B_x_list]
        B_cum_counts = torch.cumsum(torch.tensor([0] + B_node_counts[:-1], device=params.device), dim=0)
        B_edge_index = []
        for i, e in enumerate(B_edge_index_list):
            offset = B_cum_counts[i] 
            B_edge_index.append(e + offset)
        B_edge_index = torch.cat(B_edge_index, dim=1)
        B_edge_weight = torch.cat(B_edge_weight_list, dim=0)
        B_shapes = torch.stack(B_shape_list, dim=0)
        
        B_batch = []
        for i, count in enumerate(B_node_counts):
            B_batch.append(torch.full((count,), i, dtype=torch.long, device=params.device))
        B_batch = torch.cat(B_batch, dim=0)
    else:
        B_x = None
        B_edge_index = None
        B_edge_weight = None
        B_shapes = None
        B_batch = None
        
    A_batch_graph = Data(
        x=A_x,
        edge_index=A_edge_index,
        edge_weight=A_edge_weight,  
        batch=A_batch,
        shapes=A_shapes
    )
    B_batch_graph = Data(
        x=B_x,
        edge_index=B_edge_index,
        edge_weight=B_edge_weight,  
        batch=B_batch,
        shapes=B_shapes
    ) if B_x is not None else None
    
    return A_batch_graph, B_batch_graph, A_grids, B_grids


# get the input color and its' frequency as features
def calculate_color_stats(input_matrix,device = params.device):
    if isinstance(input_matrix, torch.Tensor):
        input_tensor = input_matrix.clone().detach().to(dtype=torch.int32, device=device)
    else:
        input_tensor = torch.tensor(input_matrix, dtype=torch.int32, device=device)
    input_flat = input_tensor.flatten()
    total_pixels = input_flat.numel()
    unique_colors = torch.unique(input_flat)  
    # the number of classes which appeared in input colors
    C_num = len(unique_colors)  
    # normalization
    C_num_norm = C_num / 10.0 

    # record frequency
    color_stats = {}
    for color in unique_colors:
        color_count = (input_flat == color).sum().item()
        C_freq = color_count / total_pixels 
        color_stats[color.item()] = (C_num_norm, C_freq) 

    return color_stats

# process in batch
def batch_color_stats(input_matrices):
    batch_stats = []
    for mat in input_matrices:
        stats = calculate_color_stats(mat)
        batch_stats.append(stats)
    return batch_stats

# add weights in color changed area of input datasets
class ColorChangeAttention(nn.Module):
    def __init__(self, kernel_size=3, attention_scale=3,device=params.device): # 3*3 kernel
        super().__init__()
        self.attention_scale = attention_scale
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.gradient_conv = nn.Conv2d(
            in_channels=1,
            out_channels=8, 
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            bias=True,
            device=device
        )
        self.output_conv = nn.Conv2d(8, 1, kernel_size=1, bias=True, device=device)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform_(self.gradient_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)

    def forward(self, color_grid):
        """
        input: color_grid → (B, H, W) in [0,9]
        output: attention_map → represents the degree of colors changing to give more attention
        """
        B, H, W = color_grid.shape
        if H < self.kernel_size or W < self.kernel_size:
            # process the small size input
            return torch.ones(B, H, W, device=color_grid.device) * 0.2
        
        color_norm = color_grid.float() / 9.0  
        color_norm = color_norm.unsqueeze(1)   
        
        try:
            features = self.relu(self.gradient_conv(color_norm))
            gradient = self.output_conv(features)
            gradient_abs = torch.abs(gradient)
            
            gradient_max = gradient_abs.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
            gradient_min = gradient_abs.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
            
            gradient_norm = (gradient_abs - gradient_min) / (gradient_max - gradient_min + 1e-8)
            gradient_norm = torch.clamp(gradient_norm, 0, 1)
            attention_map = gradient_norm.squeeze(1)  # (B, H, W)
            return attention_map
            
        except Exception as e:
            print(f"Attention error: {e}, falling back to uniform attention")
            return torch.ones(B, H, W, device=color_grid.device) * 0.2

# augment the small colored area 
class DetailFocusAttention(nn.Module):
    def __init__(self, kernel_size=3, attention_scale=2.5):
        super().__init__()
        self.color_attention = ColorChangeAttention(kernel_size, attention_scale)

    # distinguish the background color or number zero
    def _get_background_color(self, color_grid):
        B, H, W = color_grid.shape
        bg_colors = []
        for b in range(B):
            flat = color_grid[b].flatten()
            counts = torch.bincount(flat, minlength=10)  
            # the most frequency color as background
            bg_color = torch.argmax(counts).item() 
            bg_colors.append(bg_color)
        return bg_colors

    def forward(self, color_grid, flat_feat, input_shapes,device=params.device):
        """
        Args:
            color_grid: (B, H, W) 
            flat_feat: (sum(N_B), hidden_dim)
            input_shapes: (B, 2) 
        Returns:
            flat_feat_enhanced: (sum(N_B), hidden_dim) 
        """
        batch_size = color_grid.shape[0]
        hidden_dim = flat_feat.shape[1]
        device = flat_feat.device

        bg_colors = self._get_background_color(color_grid) 
        color_attn = self.color_attention(color_grid)  # (B, H, W)
        # generate foreground mask
        foreground_mask = torch.zeros_like(color_grid, dtype=torch.float32, device=device)
        
        for b in range(batch_size):
            bg_color = bg_colors[b]
            foreground_mask[b] = (color_grid[b] != bg_color).float() 

        area_attn = torch.zeros_like(foreground_mask, device=device)  # (B, H, W)
        for b in range(batch_size):
            H_b, W_b = input_shapes[b].cpu().numpy() 
            mask_b = foreground_mask[b, :H_b, :W_b].cpu().numpy().astype(np.uint8)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_b, connectivity=8)
            total_fg = stats[1:, cv2.CC_STAT_AREA].sum()  
            
            area_attn_b = np.zeros((H_b, W_b), dtype=np.float32)
            for label in range(1, num_labels):
                region_area = stats[label, cv2.CC_STAT_AREA]
                area_ratio = region_area / total_fg if total_fg > 0 else 0.0
                area_attn_b[labels == label] = 1.0 - area_ratio  
            area_attn[b, :H_b, :W_b] = torch.tensor(area_attn_b, device=device)

        area_attn = area_attn / (area_attn.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1) + 1e-8)
        final_attn = color_attn * area_attn  # (B, H, W)

        flat_attn_list = []
        for b in range(batch_size):
            H_b, W_b = input_shapes[b].cpu().numpy()
            attn_b = final_attn[b, :H_b, :W_b].reshape(-1)  # (H_b×W_b,)
            flat_attn_list.append(attn_b)
        flat_attn = torch.cat(flat_attn_list, dim=0)  # (sum(N_B),)
        flat_feat_enhanced = flat_feat * flat_attn.unsqueeze(1)  # (sum(N_B), hidden_dim)

        return flat_feat_enhanced,final_attn  # get attention map


from torch_scatter import scatter_softmax

class GATLayer(nn.Module):
    """Multi-head Edge-weighted Graph Attention Layer"""
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.25):  
        super().__init__()
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=False  
        )

    def forward(self, x, edge_index, edge_weight):
        return self.gat(x, edge_index)  

class GATTransformer(nn.Module):
    def __init__(self, params = params):
        super().__init__()
        self.params = params
        # add in CNN features
        self.cnn_feat = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.node_embedding = nn.Linear(in_features=24+8, out_features=params.hidden_dim)
        
        # encode absolute coordinates (2D) into the hidden dimension
        self.pos_encoding = nn.Linear(in_features=2, out_features=params.hidden_dim)
        # get dynamic bias to add in input colors
        self.dynamic_bias_module = nn.Sequential(
                                        nn.Linear(2, 4), 
                                        nn.ReLU(),
                                        nn.Linear(4, 1)  
                                        )
        
        self.encoder_gat_layers = nn.ModuleList([ GATLayer(params.hidden_dim, params.hidden_dim) for _ in range(6) ]) # 6 layers
        
        self.encoder_proj = Linear(params.hidden_dim, params.hidden_dim)  

        # add detail_attention
        self.detail_attention = DetailFocusAttention(kernel_size=3, attention_scale=1.5)
        
        # 3. predict output nodes number
        self.node_count_predictor = nn.Sequential( 
            Linear(params.hidden_dim, 64),
            nn.GroupNorm(num_groups=2, num_channels=64),
            nn.ReLU(),
            Linear(64, 32),
            nn.GroupNorm(num_groups=2, num_channels=32),
            nn.ReLU(),
            Linear(32, 2),  # output [H, W]
            nn.Sigmoid()  
        )
        
        # 4. Node feature generator (when B graph has different number of nodes than A)
        self.node_generator = nn.Sequential(
            Linear(params.hidden_dim, params.hidden_dim),
            nn.ReLU()
        )
        
        # 5. decode layers
        self.decoder_gat1 = GATLayer(in_channels=params.hidden_dim,out_channels=params.hidden_dim)
        self.decoder_gat2 = GATLayer(in_channels=params.hidden_dim,out_channels=params.hidden_dim)
        self.decoder_gat3 = GATLayer(in_channels=params.hidden_dim,out_channels=params.hidden_dim)
        self.final_proj = Linear(params.hidden_dim, params.node_dim) 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (Linear, nn.Embedding)):
                if isinstance(m, nn.Embedding):
                    torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                else:
                    torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.001)

    def forward(self, A_graph, real_B_counts=None, real_B_shapes=None, use_real_shapes=False, input_cx = None,return_attn=False):
        """
        Args:
            A_graph: (x, edge_index, edge_weight, batch, shapes)
        Returns:
            B_pred_x: logits in (sum(N_B_pred), 10)
            B_pred_counts: predict nodes count in (batch_size,)
            B_pred_shapes: predict output size in (batch_size, 2)
            B_batch:data belonged sample id in (sum(N_B_pred),)
        """
        batch_size = torch.unique(A_graph.batch).numel()
        A_x = A_graph.x  
        A_edge_index = A_graph.edge_index  
        A_edge_weight = A_graph.edge_weight  
        A_batch = A_graph.batch  
        A_shapes = A_graph.shapes 

        abs_pos_list = []
        cnn_feats = []
        for i in range(batch_size):
            current_grid = input_cx[i]
            if isinstance(current_grid, np.ndarray):
                current_grid = torch.tensor(current_grid, device=self.params.device)
            # 1d to 2d
            if current_grid.dim() == 1:
                current_grid = current_grid.unsqueeze(0)
            H, W = current_grid.shape
            
            grid_input = current_grid.float().unsqueeze(0).unsqueeze(0) / 9.0
            cnn_feat = self.cnn_feat(grid_input)  # (1, 8, H, W)
            cnn_feat_flat = cnn_feat.view(8, -1).permute(1, 0)  # (N, 8)
            cnn_feats.append(cnn_feat_flat)
            
            mask = (A_batch == i)
            num_nodes = mask.sum().item()
            if num_nodes == 0:
                continue
            node_indices = torch.arange(num_nodes, device=self.params.device)
            y = (node_indices // W).float() / H  
            x = (node_indices % W).float() / W    
            abs_pos = torch.stack([x, y], dim=1)  # (num_nodes, 2)
            abs_pos_list.append(abs_pos)
            
        A_abs_pos = torch.cat(abs_pos_list, dim=0)  # (sum(N_A), 2)
        pos_feat = self.pos_encoding(A_abs_pos)  # (sum(N_A), hidden_dim)
        cnn_feats = torch.cat(cnn_feats, dim=0)  # (sum(N_A), 8)
        
        # 1. encode A_graph features 
        A_x = torch.cat([A_x.float(), cnn_feats], dim=1)
        node_feat = self.node_embedding(A_x)
        # fuse node features with global positional encoding
        A_feat = node_feat + pos_feat
        for gat_layer in self.encoder_gat_layers:
            A_feat = F.relu(gat_layer(A_feat, A_edge_index, A_edge_weight)) + A_feat
      
        A_encoded = self.encoder_proj(A_feat)  # final output :(sum(N_A), hidden_dim)

        # add DetailFocusAttention
        color_grids = []
        max_H, max_W = 0, 0
        for i in range(batch_size):
            H_i, W_i = A_shapes[i].cpu().numpy()
            if H_i > max_H:
                max_H = H_i
            if W_i > max_W:
                max_W = W_i
        for i in range(batch_size):
            sample_mask = (A_batch == i)
            sample_color = A_x[sample_mask, 0].long()  
            H_i, W_i = A_shapes[i].cpu().numpy()
            sample_grid = sample_color.reshape(H_i, W_i)  
            padded_grid = torch.zeros((max_H, max_W), dtype=torch.long, device=sample_grid.device)
            padded_grid[:H_i, :W_i] = sample_grid
            color_grids.append(padded_grid)
        color_grid_batch = torch.stack(color_grids, dim=0)

        A_encoded,attn_map = self.detail_attention(
            color_grid=color_grid_batch,  
            flat_feat=A_encoded,
            input_shapes=A_shapes  
        )
        
        # predict B's nodes count and size
        A_global_feat = []
        for i in range(batch_size):
            mask = (A_batch == i)
            if mask.sum() == 0:
                A_global_feat.append(torch.zeros(self.params.hidden_dim, device=self.params.device))
            else:
                A_global_feat.append(A_encoded[mask].mean(dim=0))
        A_global_feat = torch.stack(A_global_feat, dim=0)  # (batch_size, hidden_dim)
        
        # based on A's size to predict B's size
        shape_multiplier = self.node_count_predictor(A_global_feat)  # (batch_size, 2)
        shape_multiplier = torch.clamp(shape_multiplier, min=0.9, max=1.8)
        B_pred_shapes_raw = (A_shapes.float() * shape_multiplier).round().long()  # (batch_size, 2) in 1-2x scale
        B_pred_shapes_raw = torch.clamp(B_pred_shapes_raw, min=1)  # min size in (1,1)
        B_pred_counts_raw = (B_pred_shapes_raw[:, 0] * B_pred_shapes_raw[:, 1]).long() 
        
        # use real output size with Training/validation
        if real_B_counts is not None and real_B_shapes is not None and use_real_shapes:
            B_used_counts = real_B_counts 
            B_used_shapes = real_B_shapes
        else:
            B_used_counts = B_pred_counts_raw  
            B_used_shapes = B_pred_shapes_raw
        
        # 3. initial node features for generating B
        B_pred_x_list = []  # store the node features of B for each sample
        B_batch_list = []    # store the node labels of B
        current_node = 0     # Global node count (to generate B_batch)
        
        for i in range(batch_size):
            # extract A's features
            A_mask = (A_batch == i)
            A_feat_i = A_encoded[A_mask]  # (N_A_i, hidden_dim)
            N_A_i = A_feat_i.shape[0]
            N_B_pred = B_used_counts[i].item()  
            
            # match the B's and A's node count
            if N_B_pred <= N_A_i:
                # B_nodes are fewer: aggregate A-graph features by attention
                att_weights = nn.Parameter(torch.randn(N_B_pred, N_A_i, device=self.params.device))
                att_weights = F.softmax(att_weights, dim=1)  # (N_B_pred, N_A_i)
                B_feat_i = torch.matmul(att_weights, A_feat_i)  # (N_B_pred, hidden_dim)
                B_feat_i = self.node_generator(B_feat_i)
            else:
                # A_nodes are fewer: Graph A features + new node features
                pad_size = N_B_pred - N_A_i
                att_weights = nn.Parameter(torch.randn(pad_size, N_A_i, device=self.params.device))
                att_weights = F.softmax(att_weights, dim=1)  # (pad_size, N_A_i)
                new_feats = torch.matmul(att_weights, A_feat_i)  # add features
                new_feats = self.node_generator(new_feats)
                B_feat_i = torch.cat([A_feat_i, new_feats], dim=0)  # (N_B_pred, hidden_dim)
            
            # Record the B's features and node labels of the current sample
            B_pred_x_list.append(B_feat_i)
            B_batch_list.append(torch.full((N_B_pred,), i, dtype=torch.long, device=self.params.device))
            current_node += N_B_pred
        
        # concat the features of B and node tokens across all samples
        B_pred_x = torch.cat(B_pred_x_list, dim=0)  # (sum(N_B_pred), hidden_dim)
        B_batch = torch.cat(B_batch_list, dim=0)    # (sum(N_B_pred),)

        # 4. generate the edge structure and edge weights of B
        B_edge_index_list = []
        B_edge_weight_list = []
        current_node = 0
        for i in range(batch_size):
            H_B, W_B = B_used_shapes[i]  
            N_B = H_B * W_B              
            
            # the closer the distance, the higher the weight
            edges = []
            edge_weights = []
            for hi in range(H_B):
                for wi in range(W_B):
                    src = current_node + hi * W_B + wi 
                    # 3x3 space
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            hj, wj = hi + di, wi + dj
                            if 0 <= hj < H_B and 0 <= wj < W_B:
                                dst = current_node + hj * W_B + wj  
                                edges.append([src, dst])
                                distance = abs(di) + abs(dj)
                                edge_weights.append(1.0 if distance == 1 else 0.6)
            
            B_edge_index_list.append(torch.tensor(edges, dtype=torch.long, device=self.params.device).t())
            B_edge_weight_list.append(torch.tensor(edge_weights, dtype=torch.float32, device=self.params.device))
            current_node += N_B

        B_edge_index = torch.cat(B_edge_index_list, dim=1)  # (2, sum(E_B_pred))
        B_edge_weight = torch.cat(B_edge_weight_list, dim=0)  # (sum(E_B_pred),)
        
        # 5. decode to generate B node classes
        B_feat = self.decoder_gat1(B_pred_x, B_edge_index, B_edge_weight)
        B_feat = F.relu(B_feat)
        B_feat = self.decoder_gat2(B_feat, B_edge_index, B_edge_weight)
        B_feat = F.relu(B_feat)
        B_feat = self.decoder_gat3(B_feat, B_edge_index, B_edge_weight)
        # output color class
        B_pred_x = self.final_proj(B_feat)  # (sum(N_B_pred), node_dim)

        # add dynamic bias in predict B
        if input_cx is not None:
            batch_stats = batch_color_stats(input_cx)
            batch_size = len(batch_stats)
            
            for i in range(batch_size):
                sample_stats = batch_stats[i]
                sample_node_mask = (B_batch == i)
                if not sample_node_mask.any():
                    continue

                for color, (c_num_norm, c_freq) in sample_stats.items():
                    feat = torch.tensor([c_num_norm, c_freq], dtype=torch.float32, device=B_pred_x.device).unsqueeze(0)
                    dynamic_bias = self.dynamic_bias_module(feat).squeeze()
                    B_pred_x[sample_node_mask, color] += dynamic_bias

        if return_attn:
            return B_pred_x, B_pred_counts_raw, B_pred_shapes_raw, B_batch, attn_map
        else:
            return B_pred_x, B_pred_counts_raw, B_pred_shapes_raw, B_batch


# compare the structre
class StructuralCorrelationLoss(nn.Module):
    def __init__(self, window_size=3, sigma=1.0, C3=1e-4):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C3 = C3  
        self.register_buffer("window", self._create_gaussian_window(window_size, sigma))
        
        self.std_small_same_loss = 0.01    # high similar
        self.std_small_diff_loss = 1.99    # less std but different value
        self.default_loss = 1.0            # default
        
        self.std_min_threshold = 1e-5      # std is too small 
        self.mean_diff_threshold = 0.1     # check mean value
        self.grid_min = 0.0                
        self.grid_max = 9.0                

    def _create_gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()  
        window = gauss.ger(gauss).unsqueeze(0).unsqueeze(0)  
        return window

    def forward(self, pred, target):
        # preprocess
        pred_clamped = torch.clamp(pred, min=self.grid_min, max=self.grid_max)
        target_clamped = torch.clamp(target, min=self.grid_min, max=self.grid_max)

        if torch.isnan(pred_clamped).any() or torch.isinf(pred_clamped).any():
            return torch.tensor(self.default_loss, device=pred.device)
        if torch.isnan(target_clamped).any() or torch.isinf(target_clamped).any():
            return torch.tensor(self.default_loss, device=pred.device)

        pred_mean = F.conv2d(pred_clamped, self.window, padding=self.window_size//2, groups=1)
        target_mean = F.conv2d(target_clamped, self.window, padding=self.window_size//2, groups=1)

        pred_var = F.conv2d(pred_clamped**2, self.window, padding=self.window_size//2, groups=1) - pred_mean**2
        pred_var = torch.clamp(pred_var, min=1e-8) 
        target_var = F.conv2d(target_clamped**2, self.window, padding=self.window_size//2, groups=1) - target_mean**2
        target_var = torch.clamp(target_var, min=1e-8)

        pred_std = torch.sqrt(pred_var + 1e-5) 
        target_std = torch.sqrt(target_var + 1e-5)

        covar = F.conv2d(pred_clamped * target_clamped, self.window, padding=self.window_size//2, groups=1) - pred_mean * target_mean
        covar = torch.clamp(covar, min=-81.0, max=81.0)  

        denominator = pred_std * target_std + self.C3 + 1e-5
        struct_sim = (covar + self.C3) / denominator

        struct_sim = torch.clamp(struct_sim, min=-1.0, max=1.0)
        loss = 1 - struct_sim.mean()

        if torch.isnan(loss):
            mean_pred_std = pred_std.mean().item()
            mean_target_std = target_std.mean().item()
            is_std_small = (mean_pred_std < self.std_min_threshold) and (mean_target_std < self.std_min_threshold)
            
            mean_pred = pred_mean.mean().item()
            mean_target = target_mean.mean().item()
            is_mean_close = abs(mean_pred - mean_target) < self.mean_diff_threshold

            if is_std_small and is_mean_close:
                return torch.tensor(self.std_small_same_loss, device=pred.device)
            elif is_std_small and not is_mean_close:
                return torch.tensor(self.std_small_diff_loss, device=pred.device)
            else:
                return torch.tensor(self.default_loss, device=pred.device)

        return loss

struct_loss_module = StructuralCorrelationLoss().to(params.device)

def compute_structural_loss(pred_logits, target_classes, pred_shape, real_shape,used_shape):
    device = pred_logits.device
    H_real, W_real = real_shape
    H_pred, W_pred = pred_shape
    
    pred_grid = graph_to_grid(pred_logits, used_shape)  # (H_pred, W_pred) in validate or (H_real, W_real) in train
    pred_grid = torch.tensor(pred_grid, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,H_pred,W_pred)
    
    target_grid = target_classes.cpu().numpy().reshape(H_real, W_real)  # (H_real, W_real)
    target_grid = torch.tensor(target_grid, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,H_real,W_real)
    
    # preprocess with the condition of different shapes
    if H_pred >= H_real and W_pred >= W_real:
        pred_padded = pred_grid[:, :, :H_real, :W_real] 
    elif H_pred <= H_real and W_pred <= W_real:
        pred_padded = torch.nn.functional.pad(
            pred_grid, 
            (0, W_real - W_pred, 0, H_real - H_pred), 
            mode='constant', 
            value=0.0
        )
    elif H_pred <= H_real and W_pred >= W_real:
        pred_cropped = pred_grid[:, :, :H_pred, :W_real] 
        pred_padded = torch.nn.functional.pad(
            pred_cropped, 
            (0, 0, 0, H_real - H_pred), 
            mode='constant', 
            value=0.0
        )
    else:
        pred_cropped = pred_grid[:, :, :H_real, :W_pred] 
        pred_padded = torch.nn.functional.pad(
            pred_cropped, 
            (0, W_real - W_pred, 0, 0), 
            mode='constant', 
            value=0.0
        )
        
    struct_loss_val = struct_loss_module(pred_padded, target_grid)

    return struct_loss_val


# add color diversity penalty
def color_diversity_penalty(logits_softmax, targets_onehot,diversity_coeff = params.diversity_coeff):
    node_entropy = -torch.sum(logits_softmax * torch.log(logits_softmax + 1e-8), dim=1)
    # get target_entropy as baseline
    target_entropy = -torch.sum(targets_onehot * torch.log(targets_onehot + 1e-8), dim=1)
    entropy_penalty = torch.clamp(abs(target_entropy - node_entropy), min=0.0) * diversity_coeff
    return entropy_penalty.mean()

def color_count_penalty(preds, targets):
    pred_colors = torch.unique(preds)
    target_colors = torch.unique(targets)
    pred_count = len(pred_colors)
    target_count = len(target_colors)
    penalty = (1.0 - pred_count / max(target_count, 1))
    return penalty
    
def get_focal_alpha(targets, B_batch,alpha_min=1.0, alpha_max=6):
    cell_weights = torch.ones_like(targets, dtype=torch.float32, device=targets.device)
    batch_ids = torch.unique(B_batch)  
    
    for batch_id in batch_ids:
        sample_mask = (B_batch == batch_id)
        sample_targets = targets[sample_mask]
        total_nodes = sample_targets.numel()
        if total_nodes == 0:
            continue 
        # count the frequency of each color (include 0) in the current sample
        color_counts = torch.bincount(sample_targets,minlength=10) 
        color_ids = torch.where(color_counts < 10)[0]  
        
        # the lower the frequency, the higher the weight
        # weight = total_nodes / (color frequency × number of color types)
        num_color_types = len(color_ids)    
        color_weight_dict = {}
        
        for color in color_ids:
            freq = color_counts[color]  
            weight = total_nodes / (freq * num_color_types)
            color_weight_dict[color] = weight
        
        # normalized weights,unified into the range [1, 3] 
        if color_weight_dict:
            max_weight = max(color_weight_dict.values())
            min_weight = min(color_weight_dict.values())
            for color in color_weight_dict:
                if max_weight == min_weight:
                    norm_weight = alpha_min
                else:
                    norm_weight = alpha_min + (color_weight_dict[color] - min_weight) * (alpha_max - alpha_min) / (max_weight - min_weight)
                color_weight_dict[color] = norm_weight

        for color, weight in color_weight_dict.items():
            color_mask = (sample_targets == color)
            cell_weights[sample_mask][color_mask] = weight * 2.2
    
    return cell_weights

# add spatial_isolation_weight
def get_spatial_isolation_weight(targets, B_batch, B_shapes, alpha_min=1.0, alpha_max=6.0):
    iso_weight = torch.ones_like(targets, dtype=torch.float32, device=targets.device)
    batch_ids = torch.unique(B_batch)
    
    for batch_id in batch_ids:
        sample_mask = (B_batch == batch_id)
        sample_targets = targets[sample_mask]
        H, W = B_shapes[batch_id].cpu().numpy()
        
        color_counts = torch.bincount(sample_targets, minlength=10).float()
        color_freq = color_counts / (sample_targets.numel() + 1e-8) 
        
        try:
            grid = sample_targets.cpu().numpy().reshape(H, W)
        except ValueError as e:
            print(f"Batch {batch_id} reshape error: target num {sample_targets.numel()}, shape ({H}, {W})")
            raise e
        
        #  count the number of same-color 3×3 neighbors
        for i in range(H):
            for j in range(W):
                cell_flat_idx = i * W + j 
                color = grid[i, j]
                color_idx = int(color)  
                same_neighbor_count = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == color:
                            same_neighbor_count += 1
                
                # get base_weight according to same_neighbor_count
                if same_neighbor_count == 0:
                    base_weight = alpha_max  # completely isolate
                elif same_neighbor_count == 1:
                    base_weight = alpha_max * 0.6  
                elif same_neighbor_count == 2:
                    base_weight = alpha_max * 0.3  
                else:
                    base_weight = alpha_min 
                
                # change the weight according to the color frequency
                freq = color_freq[color_idx]
                # the lower frequency,the higher scale
                freq_scale = 1.0 + (1.0 - freq) * 0.5  #  freq->0:scale->1.5;freq->1:scale->1.0
                final_weight = base_weight * freq_scale
                if isinstance(final_weight, torch.Tensor):
                    final_weight = final_weight.clone().detach()
                else:
                    final_weight = torch.tensor(final_weight,dtype=torch.float32,device=device)
                final_weight = torch.clamp(final_weight, min=alpha_min, max=alpha_max)
                iso_weight[sample_mask][cell_flat_idx] = final_weight
    
    return iso_weight


def calculate_loss(logits,targets,targets_onehot, B_batch, use_real_shapes,
                   B_pred_shapes=None, real_B_shapes=None,
                   foreground_weight=params.foreground_weight, 
                   bg_penalty_coeff=params.bg_penalty_coeff,
                   struct_loss_weight = params.struct_loss_weight,
                   fg_ce_weight = params.fg_ce_weight,
                   count_penalty_coeff = params.count_penalty_coeff,
                   epoch=None, total_epochs=None):
    # shape loss:calculate the losses for length, width, and total node count
    shape_loss = 0.0
    ce_loss_1 = 0.0
    if B_pred_shapes is not None and real_B_shapes is not None:
        h_loss = F.mse_loss(B_pred_shapes[:,0].float(), real_B_shapes[:,0].float())
        w_loss = F.mse_loss(B_pred_shapes[:,1].float(), real_B_shapes[:,1].float())
        count_pred = (B_pred_shapes[:,0] * B_pred_shapes[:,1]).float()
        count_real = (real_B_shapes[:,0] * real_B_shapes[:,1]).float()
        count_loss = F.mse_loss(count_pred, count_real)
        shape_loss = params.hw_weight * (h_loss + w_loss) + params.nodes_weight * count_loss
        
    # 1.1 spatial isolation weight
    iso_weight = get_spatial_isolation_weight(targets, B_batch, real_B_shapes)
    log_softmax = F.log_softmax(logits, dim=1)  # (sum(N_B), 10)
    elementwise_loss = -torch.sum(log_softmax * targets_onehot, dim=1)  # (sum(N_B),)
    weighted_elementwise_loss = elementwise_loss * iso_weight
    ce_loss_1 = weighted_elementwise_loss.mean() 
    # 1.2 imbalanced classes:calculate Focal Loss
    alpha = get_focal_alpha(targets, B_batch) # improve weights of colors
    logits_softmax = F.softmax(logits, dim=1)
    p_t = torch.gather(logits_softmax, dim=1, index=targets.unsqueeze(1)).squeeze(1) # take real class possibilities
    focal_loss = -alpha * ((1 - p_t) ** 2.5) * torch.log(torch.clamp(p_t, 1e-8))  # γ=2.5
    ce_loss_2 = focal_loss.mean()
    # 1.3  add in color_diversity_penalty after half of epochs
    diversity_weight_epoch = 0.0 if (epoch is not None and total_epochs is not None and epoch < total_epochs * 0.5) else 1.0
    ce_loss_3 = color_diversity_penalty(logits_softmax, targets_onehot) * diversity_weight_epoch
    # sum of above loss
    ce_loss = ce_loss_1 + ce_loss_2 + ce_loss_3
    
    # 2. full-background penalty and structure_loss
    batch_size = torch.unique(B_batch).shape[0] if B_batch.numel() > 0 else 0
    full_bg_penalty = 0.0
    struct_loss = torch.tensor(0.0, device=params.device)
    total_struct_loss = torch.tensor(0.0, device=params.device)
    single_color_penalty = 0.0
    
    if batch_size > 0:
        for i in range(batch_size):
            sample_mask = (B_batch == i)
            sample_logits = logits[sample_mask]
            sample_targets = targets[sample_mask]
            
            sample_pred = torch.argmax(sample_logits, dim=1)
            # if model predicts full_background 
            if (sample_pred == 0).all():
                sample_node_num = sample_mask.sum().item()
                sample_foreground_ratio = (sample_targets > 0).float().mean().item()
                # penalty coefficient × number of sample nodes (more nodes, heavier penalty)
                full_bg_penalty += bg_penalty_coeff * sample_node_num * (1 + sample_foreground_ratio)

            # if model predicts one_color full_background
            unique_pred_colors = torch.unique(sample_pred)
            if len(unique_pred_colors) == 1 and unique_pred_colors[0] != 0:
                sample_node_num = sample_mask.sum().item()
                single_color_penalty += params.single_color_penalty_coeff * sample_node_num
                
            if B_pred_shapes is not None and real_B_shapes is not None and sample_logits.numel() > 0 and sample_targets.numel() > 0:
                pred_shape = B_pred_shapes[i].cpu().numpy()
                real_shape = real_B_shapes[i].cpu().numpy()
                used_shape = real_shape if use_real_shapes else pred_shape
                
                sample_struct_loss = compute_structural_loss(
                                pred_logits=sample_logits,
                                target_classes=sample_targets,
                                pred_shape=real_shape,
                                real_shape=real_shape,
                                used_shape=used_shape
                                )
                total_struct_loss += sample_struct_loss 
                
        full_bg_penalty /= batch_size
        single_color_penalty /= batch_size
        if total_struct_loss.numel() > 0:
            struct_loss = total_struct_loss / batch_size 
        
    # calculate accuracy
    preds = torch.argmax(logits, dim=1)
    total_nodes = targets.numel()
    correct_nodes = (preds == targets).sum().item()
    node_accuracy = correct_nodes / total_nodes if total_nodes > 0 else 1.0

    # calculate the proportion of nodes with accurate foreground predictions
    foreground_mask = (targets > 0)
    fg_ce_loss = torch.tensor(0.0, device=params.device)
    if foreground_mask.sum() > 0:  
        fg_logits = logits[foreground_mask]
        fg_targets_onehot = targets_onehot[foreground_mask]
        fg_log_softmax = F.log_softmax(fg_logits, dim=1)
        fg_ce_loss = -torch.sum(fg_log_softmax * fg_targets_onehot, dim=1).mean()

    # add count_penalty after half of epochs
    count_penalty_weight_epoch = 0.0 if (epoch is not None and total_epochs is not None and epoch < total_epochs * 0.5) else 1.0 
    count_penalty = color_count_penalty(preds, targets) * count_penalty_weight_epoch
    ce_loss = ce_loss + fg_ce_weight * fg_ce_loss + count_penalty_coeff * count_penalty
    
    total_foreground = foreground_mask.sum().item()
    correct_foreground = (preds[foreground_mask] == targets[foreground_mask]).sum().item()
    foreground_accuracy = correct_foreground / total_foreground if total_foreground > 0 else 0.0
    
    total_loss = params.cls_loss_weight * ce_loss + params.struct_loss_weight * struct_loss + full_bg_penalty + single_color_penalty + params.shape_loss_weight * shape_loss

    return total_loss,ce_loss,full_bg_penalty, node_accuracy, foreground_accuracy

def aspect_ratio_penalty(B_pred_shapes, max_aspect_ratio=params.max_aspect_ratio, penalty_coeff=params.penalty_coeff):
    if B_pred_shapes.numel() == 0:
        return torch.tensor(0.0, device=B_pred_shapes.device)
    
    H = B_pred_shapes[:, 0].float()
    W = B_pred_shapes[:, 1].float()
    # compute the aspect ratio as the maximum of H/W and W/H, ensuring it is ≥ 1
    aspect_ratio = torch.max(H / W, W / H)
    
    # average over the parts that exceed the threshold
    penalty = torch.max(torch.zeros_like(aspect_ratio), aspect_ratio - max_aspect_ratio)
    aspect_loss = penalty_coeff * torch.mean(penalty)
    return aspect_loss

def validate(model, val_data):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    total_fg_correct = 0
    total_fg_nodes = 0

    with torch.no_grad():
        val_loader = DataLoader(val_data, batch_size=1, collate_fn=collate_graphs, shuffle=False)
        for batch in val_loader:
            A_batch_graph, B_batch_graph, A_grids, B_grids = batch
            
            A_batch_graph = A_batch_graph.to(params.device)
            if B_batch_graph is not None:
                B_batch_graph = B_batch_graph.to(params.device)
            
            real_B_shapes = B_batch_graph.shapes
            real_B_counts = (real_B_shapes[:, 0] * real_B_shapes[:, 1]).long()
            
            B_pred_x, _, B_pred_shapes, B_batch = model(
                A_batch_graph,
                real_B_counts=real_B_counts,
                real_B_shapes=real_B_shapes,
                use_real_shapes=True,
                input_cx=A_grids
            )
            
            targets = B_batch_graph.x[:, 0].long()  
            targets_onehot = F.one_hot(targets, num_classes=params.node_dim).float()
            
            loss, ce_loss, bg_penalty, node_acc, fg_acc = calculate_loss(
                logits=B_pred_x,
                targets=targets,
                targets_onehot=targets_onehot,
                B_batch=B_batch,
                use_real_shapes=True,
                B_pred_shapes=B_pred_shapes,
                real_B_shapes=real_B_shapes
            )
            
            preds = torch.argmax(B_pred_x, dim=1)
            correct = (preds == targets).sum().item()
            fg_mask = (targets > 0)
            fg_correct = (preds[fg_mask] == targets[fg_mask]).sum().item()
            fg_nodes = fg_mask.sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_nodes += targets.numel()
            total_fg_correct += fg_correct
            total_fg_nodes += fg_nodes

    # get mean metrics
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    node_accuracy = (total_correct / total_nodes * 100) if total_nodes > 0 else 0.0
    fg_accuracy = (total_fg_correct / total_fg_nodes * 100) if total_fg_nodes > 0 else 0.0

    model.train()
    return (avg_loss, node_accuracy, fg_accuracy)


def graph_to_grid(node_feat, shape):
    """reconstruct to 2D grid"""
    H, W = shape
    N = H * W
    node_feat = node_feat[:N]  # get the current sample
    # distinguish whether the input is logits (2D) or class indices (1D)
    if isinstance(node_feat, np.ndarray):
        dim = node_feat.ndim 
    elif isinstance(node_feat, torch.Tensor):
        dim = node_feat.dim()  
    else:
        raise TypeError(f"unsupported type:{type(node_feat)}")
    if dim == 2:
        if isinstance(node_feat, np.ndarray):
            digits = np.argmax(node_feat, axis=1)
        else:
            digits = node_feat.argmax(dim=1).cpu().numpy()
    else:
        if isinstance(node_feat, torch.Tensor):
            digits = node_feat.cpu().numpy()
        else:
            digits = node_feat  
    return digits.reshape(H, W)

def visualize_predictions(model, dataset, num_samples, cmap=None, norm=None):
    if cmap is None:
        # 0:black, 1:blue, 2:red, 3:green, 4:yellow, 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
        cmap = colors.ListedColormap(
            ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    if norm is None:
        norm = colors.Normalize(vmin=0, vmax=9)  
    
    model.eval()
    val_data = dataset.get_val_data()
    if len(val_data) == 0:
        print("NO DATA TO DRAW")
        return
    
    # choose sample randomly
    samples = random.sample(val_data, min(num_samples, len(val_data)))
    
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            A_graph, B_graph, A_grid, B_grid = sample
            A_graph = A_graph.to(params.device)
            B_graph = B_graph.to(params.device)
            
            A_grid = np.clip(A_grid, 0, 9)
            B_grid = np.clip(B_grid, 0, 9)
            
            A_batch_graph, _, _, _ = collate_graphs([(A_graph, B_graph, None, None)])
            A_batch_graph = A_batch_graph.to(params.device)

            if isinstance(A_grid, np.ndarray):
                input_cx = torch.tensor(A_grid, device=params.device).unsqueeze(0)  # (1, H, W)
            else:
                input_cx = A_grid.unsqueeze(0).to(params.device) 
            B_pred_x, B_pred_counts, B_pred_shapes, _,attn_map = model(A_batch_graph,input_cx=input_cx,return_attn=True)
            
            B_pred_grid = graph_to_grid(B_pred_x, B_pred_shapes[0].cpu())
            B_pred_grid = np.clip(B_pred_grid, 0, 9) 
            
            # plot input & output results
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(A_grid, cmap=cmap, norm=norm)
            axes[0].set_title(f'Input size A\n: {A_grid.shape}')
            axes[1].imshow(B_grid, cmap=cmap, norm=norm)
            axes[1].set_title(f'Target size B\n: {B_grid.shape}')
            axes[2].imshow(B_pred_grid, cmap=cmap, norm=norm)
            axes[2].set_title(f'Predict size B\'\n: {B_pred_grid.shape}')
            # plot attention map
            axes[3].imshow(attn_map[0].cpu().numpy(), cmap='hot')
            axes[3].set_title('Attention Map')

            # check the model's attention distribution
            print("attn min:", attn_map.min().item(), "max:", attn_map.max().item(), "mean:", attn_map.mean().item())
            
            for ax in axes:
                ax.axis('off')
                ax.grid(color='white', linewidth=0.5) 
            
            plt.tight_layout()  
            plt.show()


from torch.optim.lr_scheduler import ReduceLROnPlateau
def train():
    train_dataset = ARCGraphDataset(series='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        collate_fn=collate_graphs,
        shuffle=True,
        num_workers=0
    )
    print(f"training_data_len: {len(train_dataset)}")

    val_dataset = ARCGraphDataset(series='val')
    val_data = val_dataset.get_val_data()
    print(f"val_data_len: {len(val_data)}")
    
    model = GATTransformer(params).to(params.device) 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3) 
    
    print("start training...")
    best_val_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    # record training loss period result
    counter = 0
    
    for epoch in range(params.epochs):
        model.train()
        total_train_loss = 0.0    # sum loss
        total_ce_loss = 0.0       # color loss
        total_bg_penalty = 0.0    # full background penalty
        total_aspect_loss = 0.0   # H_W ratio penalty
        total_correct_nodes = 0   
        total_nodes = 0           
        total_correct_fg = 0      # correct foreground node count
        total_fg_nodes = 0        # total foreground node count
        epoch_total_samples = 0

        for batch_idx, (A_batch_graph, B_batch_graph, A_grids, B_grids) in enumerate(train_loader):
            torch.cuda.empty_cache()  
            A_batch_graph = A_batch_graph.to(params.device)
            B_batch_graph = B_batch_graph.to(params.device)

            optimizer.zero_grad()
            # get real shapes
            real_B_counts = torch.bincount(B_batch_graph.batch)
            real_B_shapes = B_batch_graph.shapes

            B_pred_x, B_pred_counts, B_pred_shapes, B_batch = model(
                                            A_batch_graph, 
                                            real_B_counts=real_B_counts,
                                            real_B_shapes=real_B_shapes,
                                            use_real_shapes=True,
                                            input_cx = A_grids
                                        )
            targets = B_batch_graph.x[:, 0].long()
            targets_onehot = F.one_hot(targets, num_classes=params.node_dim).float()
            
            # calculate loss
            total_loss, ce_loss, full_bg_penalty, node_accuracy, foreground_accuracy = calculate_loss(
                            logits=B_pred_x,
                            targets = targets,  
                            targets_onehot = targets_onehot,
                            B_batch=B_batch,
                            use_real_shapes = True,   # use real shape in train
                            B_pred_shapes=B_pred_shapes,
                            real_B_shapes=real_B_shapes,
                            epoch = epoch,
                            total_epochs = params.epochs
                            )
            aspect_loss = aspect_ratio_penalty(B_pred_shapes=B_pred_shapes)
            sum_loss = total_loss + aspect_loss
            
            sum_loss.backward()
            # clip grad_norm
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            total_norm_beforeclip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            # update lr
            optimizer.step()
            
            current_batch_size = torch.unique(B_batch).shape[0]  
            current_total_nodes = targets.numel()  
            fg_mask = (targets > 0)
            current_total_fg = fg_mask.sum().item()  
            epoch_total_samples += current_batch_size
            
            total_train_loss += sum_loss.item() * current_batch_size
            total_ce_loss += ce_loss.item() * current_batch_size
            total_bg_penalty += full_bg_penalty * current_batch_size
            total_aspect_loss += aspect_loss.item() * current_batch_size
            
            preds = torch.argmax(B_pred_x, dim=1)
            current_correct = (preds == targets).sum().item()
            total_correct_nodes += current_correct
            total_nodes += current_total_nodes  
            
            current_fg_correct = (preds[fg_mask] == targets[fg_mask]).sum().item()
            total_correct_fg += current_fg_correct  
            total_fg_nodes += current_total_fg  

            if (batch_idx + 1) % params.log_every == 0:
                print(f"Epoch [{epoch+1}/{params.epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")
                print(f"batch_size: {current_batch_size}, total_nodes: {current_total_nodes}")
                print(f"sum_loss: {sum_loss.item():.4f}, ce_loss: {ce_loss.item():.4f}")
        
        # calculate mean loss
        avg_train_loss = total_train_loss / epoch_total_samples if epoch_total_samples > 0 else 0.0
        avg_ce_loss = total_ce_loss / epoch_total_samples if epoch_total_samples > 0 else 0.0
        avg_bg_penalty = total_bg_penalty / epoch_total_samples if epoch_total_samples > 0 else 0.0
        avg_aspect_loss = total_aspect_loss / epoch_total_samples if epoch_total_samples > 0 else 0.0
        
        avg_node_acc = (total_correct_nodes / total_nodes * 100) if total_nodes > 0 else 0.0
        avg_fg_acc = (total_correct_fg / total_fg_nodes * 100) if total_fg_nodes > 0 else 0.0
        train_loss_history.append(avg_train_loss)
        
        # validate
        val_metrics = validate(model, val_data)  
        avg_val_loss, avg_val_node_acc, avg_val_fg_acc = val_metrics
        # use avg_train_loss/avg_val_loss as base
        scheduler.step(avg_val_loss)
        val_loss_history.append(avg_val_loss)
        
        # print epoch logs
        print("="*50)
        print(f"Epoch [{epoch+1}/{params.epochs}] train metrics:")
        print(f"  avg_train_loss: {avg_train_loss:.4f}, avg_ce_loss: {avg_ce_loss:.4f}, avg_bg_penalty: {avg_bg_penalty:.4f}, avg_aspect_loss: {avg_aspect_loss:.4f}")
        print(f"  avg_node_acc: {avg_node_acc:.2f}%, avg_fg_acc: {avg_fg_acc:.2f}%")
        print(f"Epoch [{epoch+1}/{params.epochs}] validate metrics:")
        print(f"  avg_val_loss: {avg_val_loss:.4f}, avg_val_node_acc: {avg_val_node_acc:.2f}%, avg_val_fg_acc: {avg_val_fg_acc:.2f}%") 
        print("="*50)
        
        # print best model metrics
        if avg_val_loss < best_val_loss:
            best_val_loss, best_val_node_acc, best_val_fg_acc = val_metrics
            print(f"===> best_val_loss: {best_val_loss:.2f}, best_val_node_acc: {best_val_node_acc:.2f}%, best_val_fg_acc: {best_val_fg_acc:.2f}%")
            counter = 0
        else:
            counter += 1
            
        # print per 3 epochs
        if (epoch + 1) % 3 == 0:
            print(f"visualize Epoch [{epoch+1}] results:")
            visualize_predictions(model, val_dataset, num_samples=10)  
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-', linewidth=2)
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'r-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss vs Epochs', fontsize=14)
    plt.legend(loc='upper right', fontsize=12, labels=['Training (Blue)', 'Validation (Red)'])
    plt.grid(alpha=0.3)  
    plt.xticks(range(1, len(train_loss_history) + 1, max(1, len(train_loss_history)//10)))
    plt.tight_layout()
    plt.show()

    print(f"=====Recent {counter} epochs the avg_val_loss has not decreased=====")
    print("final several predict outputs:")
    visualize_predictions(model, val_dataset, num_samples=5)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    train()


def predict_test(test_data_path):
    model = GATTransformer().to(params.device)
    model.eval() 

    test_challenges = load_json(test_data_path)
    print(f"test_data_len {len(test_challenges)} ")
    prediction_result = {}

    with torch.no_grad(): 
        for sample_name, sample_data in test_challenges.items():
            test_input = sample_data['test'][0]['input']
            A_grid = np.array(test_input, dtype=np.int32)  

            A_graph = build_graph(A_grid)
            A_batch_graph, _, _, _ = collate_graphs([(A_graph, None, None, None)])
            A_batch_graph = A_batch_graph.to(params.device)
            
            input_cx_tensor = torch.tensor(A_grid, dtype=torch.int32)
            input_cx = input_cx_tensor.unsqueeze(0).to(params.device)
            B_pred_x, B_pred_counts, B_pred_shapes, _ = model(A_batch_graph, input_cx=input_cx)
            # B_pred_x: (sum(N_B), 10) 

            # take the top-2 classes of each node’s logits
            top2_logits, top2_classes = torch.topk(B_pred_x, k=2, dim=1)  # top2_classes: (N_B, 2)
            top1_classes = top2_classes[:, 0].cpu().numpy()  # change to numpy
            top2_classes = top2_classes[:, 1].cpu().numpy()
            
            # get predicted shapes
            pred_shape = tuple(B_pred_shapes[0].cpu().numpy().tolist())  # (H, W)
            attempt_1 = graph_to_grid(top1_classes, pred_shape).tolist()
            attempt_2 = graph_to_grid(top2_classes, pred_shape).tolist()
            
            if len(attempt_1) > 0 and isinstance(attempt_1, list):
                row_len = len(attempt_1[0])
                attempt_1 = [row[:row_len] for row in attempt_1]  
            if len(attempt_2) > 0 and isinstance(attempt_2, list):
                row_len = len(attempt_2[0])
                attempt_2 = [row[:row_len] for row in attempt_2]
            
            prediction_result[sample_name] = [{"attempt_1": attempt_1,"attempt_2": attempt_2}]
    
    output_file_path = "/kaggle/working/submission.json"
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_result, f, ensure_ascii=False)  

    return 'success'

test_data_path = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
predict_test(test_data_path)

