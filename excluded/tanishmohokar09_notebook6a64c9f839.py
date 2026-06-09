!pip install --no-index /kaggle/input/torch-geometric/torch_geometric-2.6.1-py3-none-any.whl


import os
import cv2
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_3d_volume_from_folder(folder_path, downsample_factor=2):
    slice_files = []
    for root, _, files in os.walk(folder_path):
        for file in sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)))):
            if file.endswith(('.jpg', '.png')):
                slice_files.append(os.path.join(root, file))
    
    volume = []
    for file in slice_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            volume.append(img)
    
    if not volume:
        raise ValueError(f"Failed to load any valid images from {folder_path}")
    
    volume = np.stack(volume, axis=0)  # (D, H, W)
    d, h, w = volume.shape
    d_new = (d // downsample_factor) * downsample_factor
    h_new = (h // downsample_factor) * downsample_factor
    w_new = (w // downsample_factor) * downsample_factor

    d_new, h_new, w_new = map(int, (d_new, h_new, w_new))

    volume = volume[:d_new, :h_new, :w_new]
    volume_downsampled = volume[::downsample_factor, ::downsample_factor, ::downsample_factor]

    return volume_downsampled

def get_valid_tomograms(csv_path, root_tomo_folder):
    """Return list of tomogram IDs that have at least one valid motor"""
    df = pd.read_csv(csv_path)
    
    # Filter out entries with all -1 coordinates
    valid_df = df[
        ~((df['Motor axis 2'] == -1) & 
          (df['Motor axis 1'] == -1) & 
          (df['Motor axis 0'] == -1))
    ]
    
    # Get unique tomogram IDs that have valid motors
    valid_tomo_ids = valid_df['tomo_id'].unique()
    
    # Only keep tomograms that exist in our folder and have valid motors
    existing_tomos = set(os.listdir(root_tomo_folder))
    valid_tomos = [t for t in valid_tomo_ids if t in existing_tomos]
    
    return valid_tomos


def get_motor_locations(csv_path, tomo_id, downsample_factor=2):
    """Load and scale motor coordinates, filtering out invalid (-1,-1,-1) entries"""
    df = pd.read_csv(csv_path)
    
    # Filter for current tomogram and remove invalid (-1,-1,-1) entries
    motor_locs = df[
        (df['tomo_id'] == tomo_id) & 
        ~((df['Motor axis 2'] == -1) & 
          (df['Motor axis 1'] == -1) & 
          (df['Motor axis 0'] == -1))
    ]
    
    if motor_locs.empty:
        return np.empty((0, 3))  # Return empty array if no valid motors
    
    motor_locs = motor_locs[['Motor axis 2', 'Motor axis 1', 'Motor axis 0']].values
    return motor_locs // downsample_factor if downsample_factor > 1 else motor_locs


def sample_random_nodes(volume_shape, motor_locations, num_nodes, min_distance=10, max_attempts=100):
    random_nodes = []
    attempts = 0
    
    while len(random_nodes) < num_nodes and attempts < max_attempts:
        node = (np.random.randint(0, volume_shape[2]),
                np.random.randint(0, volume_shape[1]),
                np.random.randint(0, volume_shape[0]))
        
        if len(motor_locations) == 0 or np.all(cdist([node], motor_locations) >= min_distance):
            random_nodes.append(node)
        attempts += 1
    
    if len(random_nodes) < num_nodes:
        print(f"Warning: Only found {len(random_nodes)}/{num_nodes} random nodes after {max_attempts} attempts")
    
    return random_nodes


def compute_node_features(volume, nodes, cube_size=10):
    features = []
    half_size = cube_size // 2

    for node in nodes:
        x, y, z = map(int, node)
        x_min, x_max = max(0, x-half_size), min(volume.shape[2], x+half_size)
        y_min, y_max = max(0, y-half_size), min(volume.shape[1], y+half_size)
        z_min, z_max = max(0, z-half_size), min(volume.shape[0], z+half_size)

        cube = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        avg_intensity = np.mean(cube) if cube.size > 0 else 0
        features.append([avg_intensity, x, y, z])

    return np.array(features, dtype=np.float32)


def build_graph(motor_nodes, random_nodes, motor_features, random_features, connect_threshold=100):
    all_nodes = np.concatenate([motor_nodes, random_nodes], axis=0)
    all_features = np.concatenate([motor_features, random_features], axis=0)

    G = nx.Graph()
    for i, (node, feat) in enumerate(zip(all_nodes, all_features)):
        G.add_node(i, feature=feat, location=tuple(node), is_motor=(i < len(motor_nodes)))

    distance_matrix = cdist(all_nodes, all_nodes)
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    if G.number_of_edges() == 0 and len(all_nodes) >= 2:
        i, j = np.unravel_index(np.argmin(distance_matrix + np.eye(len(all_nodes)) * 1e6), distance_matrix.shape)
        G.add_edge(i, j, weight=distance_matrix[i, j])
        print(f"Added minimum edge between nodes {i} and {j}")

    return G


def nx_to_pyg_data(nx_graph):
    """Convert NetworkX graph to PyG Data with edge weight only."""
    node_features = []
    positions = []
    motor_labels = []

    for _, data in nx_graph.nodes(data=True):
        node_features.append(data['feature'])
        positions.append(data['location'])
        motor_labels.append(1 if data['is_motor'] else 0)

    edge_index = []
    edge_attr = []
    for u, v, data in nx_graph.edges(data=True):
        edge_index.append([u, v])
        edge_index.append([v, u])  # Undirected
        edge_attr.append([data['weight']])
        edge_attr.append([data['weight']])

    # ✅ FIXED: Convert lists to np.arrays before converting to tensors
    node_features = np.array(node_features, dtype=np.float32)
    positions = np.array(positions, dtype=np.float32)
    motor_labels = np.array(motor_labels, dtype=np.float32)

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32).to(device),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32).to(device),
        pos=torch.tensor(positions, dtype=torch.float32).to(device),
        y=torch.tensor(motor_labels, dtype=torch.float32).unsqueeze(1).to(device)
    )



def process_directory(csv_path, root_tomo_folder):
    all_graphs = {}
    valid_tomos = get_valid_tomograms(csv_path, root_tomo_folder)
    
    print(f"Found {len(valid_tomos)} tomograms with valid motors")
    print("\nProcessing tomograms:")
    print("-" * 60)
    print(f"{'Tomogram':<15} | {'Motors':<6} | {'Random':<6} | {'Edges':<6}")
    print("-" * 60)
    
    for tomo_folder in sorted(valid_tomos):
        tomo_path = os.path.join(root_tomo_folder, tomo_folder)
        try:
            # Load data
            volume = load_3d_volume_from_folder(tomo_path)
            motor_locations = get_motor_locations(csv_path, tomo_folder, downsample_factor=2)
            
            # Skip if no motors found (shouldn't happen due to pre-filtering)
            if len(motor_locations) == 0:
                print(f"{tomo_folder:<15} | {'0':<6} | {'0':<6} | {'0':<6} (no motors)")
                continue
                
            # Sample nodes
            random_nodes = sample_random_nodes(volume.shape, motor_locations, len(motor_locations))
            motor_features = compute_node_features(volume, motor_locations)
            random_features = compute_node_features(volume, random_nodes)
            
            # Build graph
            G = build_graph(motor_locations, random_nodes, motor_features, random_features)
            
            # Convert to PyG
            pyg_data = nx_to_pyg_data(G)
            all_graphs[tomo_folder] = pyg_data
            
            # Print stats for this tomogram
            num_motors = len(motor_locations)
            num_random = len(random_nodes)
            num_edges = G.number_of_edges()
            
            print(f"{tomo_folder:<15} | {num_motors:<6} | {num_random:<6} | {num_edges:<6}")
            
        except Exception as e:
            print(f"{tomo_folder:<15} | {'ERROR':<6} | {'ERROR':<6} | {'ERROR':<6} ({str(e)})")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    total_motors = sum(len(get_motor_locations(csv_path, t, 2)) for t in all_graphs.keys())
    total_random = sum(g.num_nodes - len(get_motor_locations(csv_path, t, 2)) 
                      for t, g in all_graphs.items())
    total_edges = sum(g.num_edges for g in all_graphs.values())
    
    print(f"Total tomograms processed: {len(all_graphs)}")
    print(f"Total motor nodes: {total_motors}")
    print(f"Total random nodes: {total_random}")
    print(f"Total edges: {total_edges}")
    print(f"Average motors per tomogram: {total_motors/len(all_graphs):.1f}")
    print(f"Average random nodes per tomogram: {total_random/len(all_graphs):.1f}")
    print(f"Average edges per tomogram: {total_edges/len(all_graphs):.1f}")
    
    return all_graphs
    


# import os
# import numpy as np
# import cv2
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
# from sklearn.neighbors import NearestNeighbors
# import torch
# from torch_geometric.data import Data
# from numba import jit, prange
# import time
# from collections import defaultdict

# # Global cache for KD-trees
# _kd_tree_cache = {}

# @jit(nopython=True, parallel=True)
# def compute_features_batch(volume, nodes, half_size):
#     """Numba-accelerated feature computation"""
#     features = np.empty((len(nodes), 4), dtype=np.float32)
#     for i in prange(len(nodes)):
#         x, y, z = nodes[i]
#         x1 = max(0, x-half_size)
#         y1 = max(0, y-half_size)
#         z1 = max(0, z-half_size)
#         x2 = min(volume.shape[2], x+half_size+1)
#         y2 = min(volume.shape[1], y+half_size+1)
#         z2 = min(volume.shape[0], z+half_size+1)
#         cube = volume[z1:z2, y1:y2, x1:x2]
#         features[i, 0] = np.mean(cube) if cube.size > 0 else 0.0
#         features[i, 1] = x
#         features[i, 2] = y
#         features[i, 3] = z
#     return features

# def load_3d_volume_from_folder(folder_path, downsample_factor=2):
#     """Memory-efficient volume loading with progress tracking"""
#     slice_files = []
#     for root, _, files in os.walk(folder_path):
#         for file in sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)))):
#             if file.endswith(('.jpg', '.png')):
#                 slice_files.append(os.path.join(root, file))
    
#     # Load first image to get dimensions
#     sample_img = cv2.imread(slice_files[0], cv2.IMREAD_GRAYSCALE)
#     h, w = sample_img.shape
    
#     # Parallel loading with ThreadPool
#     def load_image(file):
#         return cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
#     with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
#         volume = list(tqdm(executor.map(load_image, slice_files),
#                          total=len(slice_files),
#                          desc=f"Loading {os.path.basename(folder_path)}"))
    
#     volume = [img for img in volume if img is not None]
#     if not volume:
#         raise ValueError(f"No valid images in {folder_path}")
    
#     volume = np.stack(volume, axis=0)
    
#     # Downsample
#     d, h, w = volume.shape
#     volume = volume[:d-(d%downsample_factor),
#                    :h-(h%downsample_factor),
#                    :w-(w%downsample_factor)]
#     return volume[::downsample_factor, ::downsample_factor, ::downsample_factor], downsample_factor

# def get_motor_locations(csv_path, tomo_id, downsample_factor=1):
#     """Load and scale motor coordinates"""
#     df = pd.read_csv(csv_path)
#     motor_locs = df[df['tomo_id'] == tomo_id][['Motor axis 2', 'Motor axis 1', 'Motor axis 0']].values
#     motor_locs = motor_locs[~np.any(motor_locs == -1, axis=1)]
#     return motor_locs // downsample_factor if downsample_factor > 1 else motor_locs

# def sample_random_nodes(volume_shape, motor_locations, num_nodes, min_distance=10):
#     """Efficient node sampling with spatial caching"""
#     cache_key = hash((*volume_shape, motor_locations.tobytes()))
    
#     if cache_key not in _kd_tree_cache:
#         # Generate all possible coordinates in the volume
#         coords = np.indices(volume_shape).reshape(3, -1).T
        
#         if len(motor_locations) > 0:
#             # Use KD-tree to find coordinates far from motor locations
#             tree = NearestNeighbors(radius=min_distance, algorithm='kd_tree')
#             tree.fit(motor_locations)
#             mask = tree.radius_neighbors(coords, return_distance=False)
#             valid_coords = coords[[not len(n) for n in mask]]
#         else:
#             valid_coords = coords
        
#         _kd_tree_cache[cache_key] = valid_coords
    
#     valid_coords = _kd_tree_cache[cache_key]
    
#     # Sample random nodes from valid coordinates
#     if len(valid_coords) >= num_nodes:
#         return valid_coords[np.random.choice(len(valid_coords), num_nodes, replace=False)]
#     return valid_coords

# def build_graph(motor_nodes, random_nodes, motor_features, random_features, connect_threshold=100):
#     """Optimized graph construction with rich edge attributes"""
#     all_nodes = np.concatenate([motor_nodes, random_nodes])
#     all_features = np.concatenate([motor_features, random_features])
    
#     # KNN search
#     nbrs = NearestNeighbors(n_neighbors=min(10, len(all_nodes)), algorithm='kd_tree')
#     nbrs.fit(all_nodes)
#     distances, indices = nbrs.kneighbors(all_nodes)
    
#     # Create edges with attributes
#     mask = distances <= connect_threshold
#     rows = np.repeat(np.arange(len(indices)), mask.sum(1))
#     cols = indices[mask]
#     edge_dists = distances[mask]
    
#     # Undirected edges (symmetric)
#     edge_index = np.vstack([
#         np.concatenate([rows, cols]),
#         np.concatenate([cols, rows])
#     ])
    
#     # Calculate multiple edge attributes
#     src_pos = all_nodes[edge_index[0]]
#     dst_pos = all_nodes[edge_index[1]]
#     rel_pos = dst_pos - src_pos
#     norm_rel_pos = rel_pos / (np.linalg.norm(rel_pos, axis=1, keepdims=True) + 1e-6)
    
#     # Combine attributes: [distance, rel_x, rel_y, rel_z, norm_rel_x, norm_rel_y, norm_rel_z]
#     edge_attr = np.hstack([
#         edge_dists.repeat(2)[:, None],  # Distance
#         np.tile(rel_pos, (1, 1)),       # Relative position
#         np.tile(norm_rel_pos, (1, 1))    # Normalized direction
#     ])
    
#     # Remove duplicates and self-loops
#     edge_index, unique_idx = np.unique(np.sort(edge_index, axis=0), axis=1, return_index=True)
#     edge_attr = edge_attr[unique_idx]
    
#     return Data(
#         x=torch.tensor(all_features, dtype=torch.float32),
#         edge_index=torch.tensor(edge_index, dtype=torch.long),
#         edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
#         pos=torch.tensor(all_nodes, dtype=torch.float32),
#         y=torch.tensor([1]*len(motor_nodes) + [0]*len(random_nodes), dtype=torch.float32).unsqueeze(1)
#     )

# def process_tomogram(tomo_folder, csv_path, root_tomo_folder, device):
#     """Process single tomogram with detailed statistics"""
#     start_time = time.time()
#     tomo_path = os.path.join(root_tomo_folder, tomo_folder)
#     stats = defaultdict(str)
    
#     try:
#         # Load data
#         volume, downsample_factor = load_3d_volume_from_folder(tomo_path)
#         motor_locations = get_motor_locations(csv_path, tomo_folder, downsample_factor)
        
#         # Filter invalid positions
#         valid_mask = ((motor_locations[:, 0] < volume.shape[2]) &
#                      (motor_locations[:, 1] < volume.shape[1]) &
#                      (motor_locations[:, 2] < volume.shape[0]))
#         motor_locations = motor_locations[valid_mask]
#         num_motors = len(motor_locations)
#         stats['motors'] = f"{num_motors}"
        
#         if num_motors == 0:
#             print(f"{tomo_folder}: No valid motors - SKIPPING")
#             return None
        
#         # Sample nodes
#         random_nodes = sample_random_nodes(volume.shape, motor_locations, num_motors)
#         num_random = len(random_nodes)
#         stats['random_nodes'] = f"{num_random}"
        
#         # Compute features
#         motor_features = compute_features_batch(volume, motor_locations, 5)
#         random_features = compute_features_batch(volume, random_nodes, 5)
        
#         # Build graph
#         graph = build_graph(motor_locations, random_nodes, motor_features, random_features)
#         num_edges = graph.edge_index.shape[1]
#         stats['edges'] = f"{num_edges}"
#         stats['edge_attr'] = f"{graph.edge_attr.shape[1]}D"
        
#         # Calculate statistics
#         proc_time = time.time() - start_time
#         stats['time'] = f"{proc_time:.2f}s"
        
#         # Edge attribute statistics
#         if num_edges > 0:
#             edge_stats = {
#                 'distance': f"{graph.edge_attr[:, 0].mean().item():.2f}±{graph.edge_attr[:, 0].std().item():.2f}",
#                 'rel_pos': [f"{graph.edge_attr[:, i].mean().item():.2f}±{graph.edge_attr[:, i].std().item():.2f}" 
#                            for i in range(1, 4)],
#                 'norm_pos': [f"{graph.edge_attr[:, i].mean().item():.2f}±{graph.edge_attr[:, i].std().item():.2f}" 
#                             for i in range(4, 7)]
#             }
#             stats.update(edge_stats)
        
#         # Print formatted output
#         print(f"{tomo_folder.ljust(15)} | "
#               f"Motors: {stats['motors'].ljust(5)} | "
#               f"Random: {stats['random_nodes'].ljust(5)} | "
#               f"Edges: {stats['edges'].ljust(8)} | "
#               f"EdgeAttr: {stats['edge_attr'].ljust(5)} | "
#               f"Dist: {stats['distance'].ljust(12)} | "
#               f"Time: {stats['time']}")
        
#         if num_edges > 0:
#             print(f"{' ' * 18}RelPos: [{', '.join(stats['rel_pos'])}]")
#             print(f"{' ' * 18}NormPos: [{', '.join(stats['norm_pos'])}]")
        
#         return graph.to(device)
    
#     except Exception as e:
#         print(f"{tomo_folder}: ERROR - {str(e)}")
#         return None

# def process_directory(csv_path, root_tomo_folder, device='cpu'):
#     """Main processing pipeline with comprehensive reporting"""
#     print(f"\n{'=' * 120}")
#     print(f"{'Processing Tomograms':^120}")
#     print(f"{'=' * 120}")
#     print(f"{'Tomogram'.ljust(15)} | {'Motors'.ljust(5)} | {'Random'.ljust(5)} | {'Edges'.ljust(8)} | "
#           f"{'EdgeAttr'.ljust(5)} | {'Distance'.ljust(12)} | {'Time'.ljust(8)}")
#     print(f"{'-' * 120}")
    
#     tomograms = sorted([d for d in os.listdir(root_tomo_folder) 
#                       if os.path.isdir(os.path.join(root_tomo_folder, d))])
    
#     all_graphs = {}
#     for tomo_folder in tqdm(tomograms, desc="Processing"):
#         graph = process_tomogram(tomo_folder, csv_path, root_tomo_folder, device)
#         if graph is not None:
#             all_graphs[tomo_folder] = graph
    
#     # Final summary
#     total_stats = {
#         'motors': sum(g.y.sum().item() for g in all_graphs.values()),
#         'nodes': sum(g.num_nodes for g in all_graphs.values()),
#         'edges': sum(g.num_edges for g in all_graphs.values()),
#         'edge_attrs': all_graphs[next(iter(all_graphs))].edge_attr.shape[1] if all_graphs else 0
#     }
    
#     print(f"\n{'=' * 120}")
#     print(f"{'SUMMARY STATISTICS':^120}")
#     print(f"{'=' * 120}")
#     print(f"Processed {len(all_graphs)} tomograms")
#     print(f"Total motors: {total_stats['motors']}")
#     print(f"Total random nodes: {total_stats['nodes'] - total_stats['motors']}")
#     print(f"Total edges: {total_stats['edges']}")
#     print(f"Edge attributes per edge: {total_stats['edge_attrs']} (distance + relative_pos + normalized_dir)")
    
#     return all_graphs


# from tqdm import tqdm
# csv_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv'
# root_tomo_folder = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'
    
# # Process all tomograms
# all_graphs = process_directory(csv_path, root_tomo_folder)
    
# # Create DataLoaders
# graphs = list(all_graphs.values())
# train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    
# train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True, pin_memory=True)
# val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False, pin_memory=True)
    
# print(f"\nTraining on {len(train_graphs)} graphs, validating on {len(val_graphs)} graphs")


# import pickle

# # Save all_graphs dictionary to a .pkl file
# with open('all_graphs_modi.pkl', 'wb') as f:
#     pickle.dump(all_graphs, f)

# print("Graphs successfully saved to all_graphs.pkl")


# import matplotlib.pyplot as plt
# import networkx as nx
# from torch_geometric.utils import to_networkx

# def visualize_graph(graph_data, title="Graph Visualization", node_size=100, with_labels=False):
#     """Visualize a 3D PyG graph using 2D projection of node locations (x, y only)."""
#     # Convert PyG graph to NetworkX graph
#     nx_graph = to_networkx(graph_data, to_undirected=True)

#     # Extract 2D (x, y) positions from the node features (assume first 3 are x, y, z)
#     pos = {i: (graph_data.x[i][0].item(), graph_data.x[i][1].item()) for i in range(graph_data.num_nodes)}

#     plt.figure(figsize=(8, 6))
#     nx.draw(
#         nx_graph,
#         pos,
#         node_color='skyblue',
#         edge_color='gray',
#         node_size=node_size,
#         with_labels=with_labels,
#         font_size=8
#     )
#     plt.title(title)
#     plt.show()

# # Example: visualize one graph from the dictionary
# graph_to_plot = all_graphs['tomo_00e463']  # Replace with a valid tomogram key
# visualize_graph(graph_to_plot, title="Graph for tomo_226cd8")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt


import pickle
import torch

def load_graph_pickle(filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Safely load the serialized graph dictionary"""
    with open(filepath, 'rb') as f:
        graphs = pickle.load(f)
    
    # Ensure all graphs are on the correct device
    for tomo_id, graph in graphs.items():
        graphs[tomo_id] = graph.to(device)
    
    return graphs

# Usage:
graphs = load_graph_pickle('/kaggle/input/graphss/all_graphs_full.pkl')
print(f"Loaded {len(graphs)} graphs")


graphs = list(graphs.values())
train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False, pin_memory=False)


# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from sklearn.model_selection import train_test_split
# import torch

# # Convert a NetworkX graph to PyTorch Geometric Data
# def nx_to_pyg_data(nx_graph):
#     node_features = []
#     positions = []
#     motor_labels = []
    
#     # Create node features and labels
#     for _, data in nx_graph.nodes(data=True):
#         node_features.append([
#             data['feature'],
#             float(data.get('is_motor', False)),
#             data['location'][0], data['location'][1], data['location'][2]
#         ])
#         positions.append(data['location'])
#         motor_labels.append(1 if data.get('is_motor', False) else 0)
    
#     # Create edges
#     edge_index = []
#     edge_attr = []
#     for u, v, data in nx_graph.edges(data=True):
#         edge_index.append([u, v])
#         edge_index.append([v, u])  # Undirected
        
#         pos_u = positions[u]
#         pos_v = positions[v]
#         distance = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
        
#         edge_attr.append([data['weight'], distance])
#         edge_attr.append([data['weight'], distance])
    
#     return Data(
#         x=torch.tensor(node_features, dtype=torch.float),
#         edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
#         edge_attr=torch.tensor(edge_attr, dtype=torch.float),
#         pos=torch.tensor(positions, dtype=torch.float),
#         y_detect=torch.tensor(motor_labels, dtype=torch.float).unsqueeze(1),
#         y_coords=torch.tensor(positions, dtype=torch.float)
#     )

# # Convert all graphs (assuming all_graphs is a dictionary of NetworkX graphs)
# pyg_graphs = [nx_to_pyg_data(g) for g in all_graphs.values()]

# # Train/validation split
# train_graphs, val_graphs = train_test_split(pyg_graphs, test_size=0.2, random_state=42)

# # Create DataLoader instances
# train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

class MotorGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, downscale_factor=2):
        super().__init__()
        self.downscale_factor = downscale_factor
        
        # Simplified architecture for tiny graphs
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # LayerNorm works better than BatchNorm for small graphs
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        
        # Simplified prediction heads
        self.detect_head = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        self.loc_head = nn.Linear(hidden_channels, 3)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x1 = F.relu(self.ln1(self.conv1(x, edge_index)))
        x2 = F.relu(self.ln2(self.conv2(x1, edge_index)))
        
        motor_probs = self.detect_head(x2)
        pred_coords = self.loc_head(x2) * self.downscale_factor
        
        return motor_probs, pred_coords


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    det_criterion = nn.BCELoss()
    loc_criterion = nn.MSELoss()
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred_probs, pred_coords = model(batch)
        
        # Detection loss
        det_loss = det_criterion(pred_probs, batch.y)
        
        # Localization loss (only on motors)
        motor_mask = (batch.y == 1).squeeze()
        if motor_mask.any():
            loc_loss = loc_criterion(pred_coords[motor_mask], batch.pos[motor_mask])
        else:
            loc_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        loss = det_loss + 0.1 * loc_loss  # Adjust weight as needed
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)

def f_beta_score(tp, fp, fn, beta=2):
    """Calculate F-beta score with given beta value."""
    if (1 + beta**2) * tp + beta**2 * fn + fp == 0:
        return 0.0
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)

def evaluate(model, loader, device, threshold=1000.0):
    """Evaluate model according to competition metrics with 1000Å threshold."""
    model.eval()
    tp, fp, fn = 0, 0, 0
    distances = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_probs, pred_coords = model(batch)
            
            for i in range(batch.num_graphs):
                graph_mask = batch.batch == i
                probs = pred_probs[graph_mask]
                coords = pred_coords[graph_mask]
                true_labels = batch.y[graph_mask]
                true_coords = batch.pos[graph_mask]
                
                # Get predicted motor (node with highest probability)
                pred_idx = torch.argmax(probs)
                pred_pos = coords[pred_idx]
                pred_prob = probs[pred_idx]
                
                # Get ground truth motor
                true_motor_mask = (true_labels == 1).squeeze()
                if true_motor_mask.any():  # Tomogram contains motor
                    true_idx = torch.argmax(true_labels)
                    true_pos = true_coords[true_idx]
                    distance = torch.norm(pred_pos - true_pos).item()
                    distances.append(distance)
                    
                    if distance <= threshold and pred_prob > 0.5:
                        tp += 1
                    else:
                        fn += 1
                else:  # Tomogram has no motor
                    if pred_prob > 0.5:  # False positive (predicted motor where none exists)
                        fp += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f2 = f_beta_score(tp, fp, fn, beta=2)
    
    # Localization metrics (only for TP cases)
    loc_metrics = {
        'mean': np.mean(distances) if distances else float('nan'),
        'median': np.median(distances) if distances else float('nan'),
        'within_threshold': len([d for d in distances if d <= threshold]) / len(distances) if distances else 0
    }
    
    return {
        'precision': precision,
        'recall': recall,
        'f2': f2,
        'localization': loc_metrics,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def generate_submission(model, loader, device, csv_path, threshold=0.5):
    """Generate submission CSV with predictions using tomogram IDs from CSV."""
    model.eval()
    
    # Load all tomogram IDs from CSV
    df_labels = pd.read_csv(csv_path)
    all_tomo_ids = df_labels['tomo_id'].unique()
    predictions = {tomo_id: [-1, -1, -1] for tomo_id in all_tomo_ids}  # Initialize with -1
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_probs, pred_coords = model(batch)
            
            # Get batch indices to match with tomogram IDs
            batch_indices = batch.batch.cpu().numpy()
            unique_indices = np.unique(batch_indices)
            
            for idx in unique_indices:
                # Get predictions for this graph in the batch
                graph_mask = batch_indices == idx
                probs = pred_probs[graph_mask]
                coords = pred_coords[graph_mask]
                
                # Get predicted motor
                pred_idx = torch.argmax(probs)
                pred_prob = probs[pred_idx].item()
                pred_pos = coords[pred_idx].cpu().numpy()
                
                # We'll need to map batch index to tomogram ID
                # Since we can't access tomo_id directly, we'll use the CSV order
                # This assumes loader preserves the same order as CSV
                tomo_id = all_tomo_ids[idx]
                
                if pred_prob > threshold:
                    predictions[tomo_id] = pred_pos.tolist()
    
    # Convert to DataFrame
    submission = []
    for tomo_id, coords in predictions.items():
        submission.append({
            'tomo_id': tomo_id,
            'Motor axis 0': coords[0],
            'Motor axis 1': coords[1],
            'Motor axis 2': coords[2]
        })
    
    df = pd.DataFrame(submission)
    df.to_csv('submission.csv', index=False)
    return df

def train_and_evaluate(model, train_loader, val_loader, epochs, device, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    best_f2 = 0
    no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, device)
        current_f2 = val_metrics['f2']
        scheduler.step(current_f2)

        # Print metrics
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Detection - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F2: {val_metrics['f2']:.4f}")
        print("Localization - Mean: {:.2f}Å, Median: {:.2f}Å, Within threshold: {:.2%}".format(
            val_metrics['localization']['mean'],
            val_metrics['localization']['median'],
            val_metrics['localization']['within_threshold']))
        print(f"TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, FN: {val_metrics['fn']}")
        
        # Early stopping and model saving
        if current_f2 > best_f2:
            best_f2 = current_f2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f2': best_f2,
            }, 'best_model.pth')
            print("Saved new best model!")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping early!")
                break
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nBest model from epoch {checkpoint['epoch']} with F2: {checkpoint['best_f2']:.4f}")
    
    # Final evaluation
    print("\nFinal Validation Results:")
    final_metrics = evaluate(model, val_loader, device)
    print(f"Detection - Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f}, F2: {final_metrics['f2']:.4f}")
    print("Localization - Mean: {:.2f}Å, Median: {:.2f}Å, Within threshold: {:.2%}".format(
        final_metrics['localization']['mean'],
        final_metrics['localization']['median'],
        final_metrics['localization']['within_threshold']))
    print(f"TP: {final_metrics['tp']}, FP: {final_metrics['fp']}, FN: {final_metrics['fn']}")


from torch_geometric.loader import DataLoader

def custom_collate(batch):
    # Just returns the batch as is (already on GPU)
    return batch

train_loader = DataLoader(
    train_graphs,
    batch_size=4,
    shuffle=True,
    collate_fn=custom_collate,  # override
    pin_memory=False  # must be False
)

val_loader = DataLoader(
    val_graphs,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate,
    pin_memory=False
)


csv_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/sample_submission.csv'
root_tomo_folder = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test'
    
# Process all tomograms
all_test_graphs = process_directory(csv_path, root_tomo_folder)
    
# Create DataLoaders
test_graphs = list(all_test_graphs.values())

test_loader = DataLoader(test_graphs, batch_size=32, shuffle=True, pin_memory=False)



if __name__ == "__main__":
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MotorGNN(in_channels=4, hidden_channels=128).to(device)
    
    # Run training and evaluation
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        device=device
    )
    
    # Generate submission
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    generate_submission(model, test_loader, device,'/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/sample_submission.csv')


submission_df = generate_submission(model, test_loader, device, csv_path)
submission_df.to_csv('submission.csv', index=False)

# Verify file exists
import os
assert os.path.exists('submission.csv'), "File not created!"
print("Submission file ready!")




