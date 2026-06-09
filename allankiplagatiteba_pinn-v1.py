# # GDL-PINN for Ariel Data Challenge 2025
# 
# This notebook implements a Geometric Deep Learning approach with Physics-Informed Neural Networks
# for exoplanet spectroscopy analysis.


# Environment Setup and Imports
import torch.nn.functional as F

import os
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


!python3 --version



import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")



# Step 1: Downgrade PyTorch to 2.3.0 + CUDA 11.8
!pip install -q torch==2.3.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install PyTorch Geometric dependencies
!pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

# Step 3: Install torch-geometric
!pip install -q torch-geometric



import torch
import torch_geometric
import torch_scatter
import torch_sparse

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool

print("âœ… PyTorch version:", torch.__version__)
print("âœ… torch_geometric:", torch_geometric.__version__)
print("âœ… torch_scatter:", torch_scatter.__version__)
print("âœ… torch_sparse:", torch_sparse.__version__)



import sys
import torch

print("ğŸ�� Python version:", sys.version)
print("ğŸ”¥ PyTorch version:", torch.__version__)



# List wheels (just to check)
!ls /kaggle/input/pyg-cu118-torch260-zip



!pip install /kaggle/input/pyg-cu118-torch260-zip/torch_scatter-*.whl
!pip install /kaggle/input/pyg-cu118-torch260-zip/torch_sparse-*.whl
!pip install /kaggle/input/pyg-cu118-torch260-zip/torch_geometric-*.whl



import torch_geometric
print("âœ… PyTorch Geometric:", torch_geometric.__version__)



import torch
from torch_geometric.data import Data

# A mini graph with 3 nodes and 2 edges: 0â†’1, 1â†’2
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
edge_index = edge_index.t().contiguous()

# Each node has 2 features
x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)



import torch_geometric
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, DataLoader

print(f"PyTorch Geometric version: {torch_geometric.__version__}")



# ============================================
# PART 1: MODEL ARCHITECTURE COMPONENTS
# ============================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from typing import Dict, Any
from typing import Dict, Tuple, List, Any
from pathlib import Path
import torch.nn.functional as F




class SpectralGraphConstructor:
    """Constructs graph representations from spectral data"""
    
    def __init__(self, k_neighbors: int = 10, wavelength_threshold: float = 0.1):
        self.k_neighbors = k_neighbors
        self.wavelength_threshold = wavelength_threshold
    
    def construct_spectral_graph(self, 
                                wavelengths: np.ndarray,
                                spectral_data: np.ndarray,
                                instrument_info: Dict) -> Data:
        n_channels = len(wavelengths)
        
        # Node features
        node_features = []
        for i, wl in enumerate(wavelengths):
            features = [
                wl,
                np.log10(wl) if wl > 0 else 0,  # Handle zero wavelengths
                spectral_data[i] if len(spectral_data) > i else 0,
                1.0 if wl < 0.8 else 0.0,  # FGS1
                1.0 if 1.95 <= wl <= 3.9 else 0.0,  # AIRS
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Edge construction
        edge_index = []
        edge_attr = []
        
        # Ensure we create edges even for small graphs
        for i in range(n_channels):
            # Connect to next few channels (increase range if needed)
            for j in range(i+1, min(i+10, n_channels)):  # Increased from 5 to 10
                wl_diff = abs(wavelengths[i] - wavelengths[j])
                # Increased threshold to ensure more edges
                if wl_diff < self.wavelength_threshold * 2:  # Doubled threshold
                    edge_index.extend([[i, j], [j, i]])
                    edge_features = [
                        wl_diff,
                        1.0 / (1.0 + wl_diff),
                        float(wavelengths[i] < 0.8 and wavelengths[j] < 0.8),
                    ]
                    edge_attr.extend([edge_features, edge_features])
        
        # If no edges were created, create a minimal connected graph
        if len(edge_index) == 0:
            # Connect each node to its neighbors
            for i in range(n_channels - 1):
                edge_index.extend([[i, i+1], [i+1, i]])
                edge_features = [0.01, 0.99, 1.0]  # Default edge features
                edge_attr.extend([edge_features, edge_features])
        
        # Convert to tensors with proper shape
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            # Fallback: create empty but properly shaped tensors
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Also update the model forward method to handle edge cases better
def fixed_forward(self, batch_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed forward method with better error handling"""
    
    try:
        # Process FGS1
        fgs1_features = self.fgs1_encoder(batch_data['fgs1_features'])
        
        # Process AIRS with graph
        wavelengths = batch_data['wavelengths']
        airs_features = batch_data['airs_features']
        
        # Construct graph
        graph = self.graph_constructor.construct_spectral_graph(
            wavelengths.cpu().numpy(),
            airs_features.cpu().numpy(),
            {'instrument': 'AIRS-CH0'}
        )
        
        # Move to device
        x = graph.x.to(self.device if hasattr(self, 'device') else device)
        edge_index = graph.edge_index.to(self.device if hasattr(self, 'device') else device)
        edge_attr = graph.edge_attr.to(self.device if hasattr(self, 'device') else device)
        
        # Check if we have edges
        if edge_index.shape[1] > 0:
            # Apply graph convolutions
            for conv in self.spectral_convs:
                x = conv(x, edge_index, edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        else:
            # If no edges, just apply linear transformations
            for conv in self.spectral_convs:
                x = conv.lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        # Global pooling
        airs_pooled = x.mean(dim=0, keepdim=True)  # Simple mean if no batch
        
        # Combine features
        combined = torch.cat([airs_pooled, fgs1_features.unsqueeze(0)], dim=-1)
        decoded = self.physics_decoder(combined)
        
        # Generate predictions
        mean_pred = self.mean_head(decoded)
        log_uncertainty = self.uncertainty_head(decoded)
        uncertainty = torch.exp(log_uncertainty).clamp(min=1e-6, max=100)
        
        return mean_pred.squeeze(0), uncertainty.squeeze(0)
    
    except Exception as e:
        print(f"Error in forward pass: {e}")
        # Return default predictions
        return torch.ones(284), torch.ones(284) * 10.0

class PhysicsInformedSpectralConv(MessagePassing):
    """Physics-informed spectral convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, physics_dim: int = 16):
        super().__init__(aggr='add')
        
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(3, physics_dim)
        self.lin_physics = nn.Linear(physics_dim + out_channels, out_channels)
        
        self.opacity_kernel = nn.Parameter(torch.randn(physics_dim))
        self.absorption_kernel = nn.Parameter(torch.randn(physics_dim))
        
    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        edge_embedding = self.lin_edge(edge_attr)
        opacity_weight = torch.sigmoid(torch.matmul(edge_embedding, self.opacity_kernel))
        absorption_weight = torch.tanh(torch.matmul(edge_embedding, self.absorption_kernel))
        message = x_j * opacity_weight.unsqueeze(-1) * (1 - absorption_weight.unsqueeze(-1))
        return message
    
    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.lin_physics(combined)

     
class GeometricPINNExoplanet(nn.Module):
    """Main GDL-PINN model architecture with fixed dimensions"""
    
    def __init__(self, 
                 n_wavelengths: int = 283,
                 hidden_dim: int = 256,
                 n_graph_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_wavelengths = n_wavelengths
        self.hidden_dim = hidden_dim
        
        self.graph_constructor = SpectralGraphConstructor()
        
        # Spectral encoder with graph convolutions
        self.spectral_convs = nn.ModuleList([
            PhysicsInformedSpectralConv(
                5 if i == 0 else hidden_dim,
                hidden_dim,
                physics_dim=32
            ) for i in range(n_graph_layers)
        ])
        
        # FGS1 encoder - output hidden_dim//2 = 128
        self.fgs1_encoder = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)  # Outputs 128
        )
        
        # Physics decoder - expects hidden_dim (256) + hidden_dim//2 (128) = 384
        self.physics_decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),  # 384 -> 256
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # 256 -> 256
        )
        
        # Output heads for mean and uncertainty
        self.mean_head = nn.Linear(hidden_dim, n_wavelengths + 1)  # 256 -> 284
        self.uncertainty_head = nn.Linear(hidden_dim, n_wavelengths + 1)  # 256 -> 284
    
    def forward(self, batch_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Process FGS1
            fgs1_features = self.fgs1_encoder(batch_data['fgs1_features'])  # Shape: [128]
            
            # Process AIRS with graph
            wavelengths = batch_data['wavelengths']
            airs_features = batch_data['airs_features']
            
            # Construct graph
            graph = self.graph_constructor.construct_spectral_graph(
                wavelengths.cpu().numpy(),
                airs_features.cpu().numpy(),
                {'instrument': 'AIRS-CH0'}
            )
            
            # Move to device
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            edge_attr = graph.edge_attr.to(device)
            
            # Apply graph convolutions
            if edge_index.shape[1] > 0:
                for conv in self.spectral_convs:
                    x = conv(x, edge_index, edge_attr)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.1, training=self.training)
            else:
                # Fallback for no edges
                for conv in self.spectral_convs:
                    x = conv.lin(x)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.1, training=self.training)
            
            # Global pooling - ensure we get hidden_dim features
            # x shape: [n_nodes, hidden_dim]
            airs_pooled = x.mean(dim=0)  # Shape: [hidden_dim=256]
            
            # Ensure correct dimensions
            if airs_pooled.dim() == 1:
                airs_pooled = airs_pooled.unsqueeze(0)  # Shape: [1, 256]
            
            if fgs1_features.dim() == 1:
                fgs1_features = fgs1_features.unsqueeze(0)  # Shape: [1, 128]
            
            # Combine features
            # airs_pooled: [1, 256], fgs1_features: [1, 128]
            combined = torch.cat([airs_pooled, fgs1_features], dim=-1)  # Shape: [1, 384]
            
            # Decode
            decoded = self.physics_decoder(combined)  # Shape: [1, 256]
            
            # Generate predictions
            mean_pred = self.mean_head(decoded)  # Shape: [1, 284]
            log_uncertainty = self.uncertainty_head(decoded)  # Shape: [1, 284]
            uncertainty = torch.exp(log_uncertainty).clamp(min=1e-6, max=100)
            
            # Remove batch dimension
            return mean_pred.squeeze(0), uncertainty.squeeze(0)
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Return default predictions
            return torch.ones(284).to(device), torch.ones(284).to(device) * 10.0


# Alternative simpler fix - just add this projection layer after physics_decoder
class SimplifiedGeometricPINNExoplanet(nn.Module):
    """Simplified version with dimension fix"""
    
    def __init__(self, 
                 n_wavelengths: int = 283,
                 hidden_dim: int = 256,
                 n_graph_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_wavelengths = n_wavelengths
        self.hidden_dim = hidden_dim
        
        self.graph_constructor = SpectralGraphConstructor()
        
        # Spectral encoder
        self.spectral_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # FGS1 encoder
        self.fgs1_encoder = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dim, n_wavelengths + 1)
        self.uncertainty_head = nn.Linear(hidden_dim, n_wavelengths + 1)
    
    def forward(self, batch_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simple encoding without graph convolutions
        fgs1_feat = self.fgs1_encoder(batch_data['fgs1_features'])
        
        # Simple AIRS encoding
        airs_feat = self.spectral_encoder(
            torch.cat([
                batch_data['wavelengths'].unsqueeze(-1),
                torch.log10(batch_data['wavelengths'] + 1e-6).unsqueeze(-1),
                batch_data['airs_features'].unsqueeze(-1),
                (batch_data['wavelengths'] < 0.8).float().unsqueeze(-1),
                ((batch_data['wavelengths'] >= 1.95) & (batch_data['wavelengths'] <= 3.9)).float().unsqueeze(-1)
            ], dim=-1).mean(dim=0)  # Average over wavelengths
        )
        
        # Ensure correct dimensions
        if fgs1_feat.dim() == 1:
            fgs1_feat = fgs1_feat.unsqueeze(0)
        if airs_feat.dim() == 1:
            airs_feat = airs_feat.unsqueeze(0)
        
        # Combine
        combined = torch.cat([fgs1_feat, airs_feat], dim=-1)
        fused = self.fusion(combined)
        
        # Predictions
        mean_pred = self.mean_head(fused)
        uncertainty = torch.exp(self.uncertainty_head(fused)).clamp(min=1e-6, max=100)
        
        return mean_pred.squeeze(0), uncertainty.squeeze(0)
# ============================================
# PART 2: DATA PROCESSING FUNCTIONS
# ============================================

def load_competition_data(data_dir: Path, planet_id: str) -> Dict:
    """Load all data for a single planet"""
    
    planet_dir = data_dir / planet_id
    
    # Load ADC info
    adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
    
    # Load FGS1 data
    fgs1_files = sorted(planet_dir.glob('FGS1_signal_*.parquet'))
    fgs1_data = []
    
    fgs1_gain = float(adc_info['FGS1_adc_gain'].iloc[0])
    fgs1_offset = float(adc_info['FGS1_adc_offset'].iloc[0])
    
    for file in fgs1_files:
        df = pd.read_parquet(file)
        signal = df.values * fgs1_gain + fgs1_offset
        fgs1_data.append(signal)

    
    # Load AIRS data
    
    airs_files = sorted(planet_dir.glob('AIRS-CH0_signal_*.parquet'))
    airs_data = []
    
    airs_gain = float(adc_info['AIRS-CH0_adc_gain'].iloc[0])
    airs_offset = float(adc_info['AIRS-CH0_adc_offset'].iloc[0])
    
    for file in airs_files:
        df = pd.read_parquet(file)
        signal = df.values * airs_gain + airs_offset
        airs_data.append(signal)


    
    # Load calibration
    calibration = {}
    for instrument in ['FGS1', 'AIRS-CH0']:
        cal_dir = planet_dir / f'{instrument}_calibration'
        cal_data = {}
        for cal_type in ['dark', 'flat']:
            file_path = cal_dir / f'{cal_type}.parquet'
            if file_path.exists():
                cal_array = pd.read_parquet(file_path).values
                cal_data[cal_type] = cal_array
        calibration[instrument] = cal_data
    
    return {
        'fgs1_raw': fgs1_data,
        'airs_raw': airs_data,
        'calibration': calibration
    }

def apply_basic_calibration(raw_data: np.ndarray, calibration: Dict) -> np.ndarray:
    """Apply basic calibration to raw data"""
    calibrated = raw_data.copy()

    # Dark subtraction
    if 'dark' in calibration and calibration['dark'] is not None:
        calibrated = calibrated - calibration['dark']

    # Flat fielding
    if 'flat' in calibration and calibration['flat'] is not None:
        flat_array = calibration['flat']
        if isinstance(flat_array, np.ndarray):
            flat_norm = flat_array / np.median(flat_array)
            calibrated = calibrated / flat_norm
        else:
            print("âš ï¸� Warning: Flat field calibration is not an array")

    return calibrated


def extract_features(planet_data: Dict) -> Dict:
    """Extract features from planet data"""
    
    # Process FGS1
    fgs1_features = []
    for raw in planet_data['fgs1_raw']:
        cal = apply_basic_calibration(raw, planet_data['calibration']['FGS1'])
        # Reshape from flattened format
        cal_reshaped = cal.reshape(-1, 32, 32)
        # Simple feature: average over spatial dimensions
        light_curve = cal_reshaped.mean(axis=(1, 2))
        # Normalize
        light_curve = light_curve / np.median(light_curve)
        # Take statistics
        features = [
            light_curve.mean(),
            light_curve.std(),
            light_curve.min(),
            light_curve.max(),
            1.0 - light_curve.min(),  # Transit depth
        ]
        fgs1_features.extend(features)
    
    # Pad or truncate to fixed size
    fgs1_features = np.array(fgs1_features)
    if len(fgs1_features) < 1024:
        fgs1_features = np.pad(fgs1_features, (0, 1024 - len(fgs1_features)))
    else:
        fgs1_features = fgs1_features[:1024]
    
    # Process AIRS
    airs_features = []
    for raw in planet_data['airs_raw']:
        cal = apply_basic_calibration(raw, planet_data['calibration']['AIRS-CH0'])
        # Reshape from flattened format
        cal_reshaped = cal.reshape(-1, 32, 356)
        # Average spectrum
        spectrum = cal_reshaped.mean(axis=(0, 1))[:283]
        airs_features.append(spectrum)
    
    # Average over observations
    if len(airs_features) > 0:
        airs_features = np.mean(airs_features, axis=0)
    else:
        airs_features = np.ones(283)  # Default
    
    return {
        'fgs1_features': torch.tensor(fgs1_features, dtype=torch.float32),
        'airs_features': torch.tensor(airs_features, dtype=torch.float32)
    }

# ============================================
# PART 3: INFERENCE AND SUBMISSION
# ============================================

def process_single_planet(planet_id: str, model: nn.Module, data_dir: Path) -> Dict:
    """Process a single planet and return predictions"""
    
    try:
        # Load planet data
        planet_data = load_competition_data(data_dir, planet_id)
        
        # Extract features
        features = extract_features(planet_data)
        
        # Load wavelengths
        wavelengths = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')
        
        # Prepare batch
        batch_data = {
            'fgs1_features': features['fgs1_features'].to(device),
            'airs_features': features['airs_features'].to(device),
            'wavelengths': torch.tensor(wavelengths.values[:283, 0], dtype=torch.float32).to(device),
        }
        
        # Get predictions
        with torch.no_grad():
            pred_mean, pred_uncertainty = model(batch_data)
        
        return {
            'success': True,
            'mean': pred_mean.cpu().numpy(),
            'uncertainty': pred_uncertainty.cpu().numpy()
        }
        
    except Exception as e:
        print(f"Error processing {planet_id}: {e}")
        # Return default predictions
        return {
            'success': False,
            'mean': np.ones(284),
            'uncertainty': np.ones(284) * 10.0
        }
# Fixed version of the submission function

def create_submission(model: nn.Module, test_dir: Path, output_path: str = 'submission.csv'):
    """Create competition submission file"""
    
    # Load test data info
    test_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/test_star_info.csv')
    
    # Initialize results
    results = []
    
    print(f"Processing {len(test_star_info)} test planets...")
    
    # Process each planet
    for idx, row in tqdm(test_star_info.iterrows(), total=len(test_star_info)):
        # FIX: Convert planet_id to string properly
        planet_id_raw = row['planet_id']
        if isinstance(planet_id_raw, float):
            planet_id = str(int(planet_id_raw))
        else:
            planet_id = str(planet_id_raw)
        
        # Get predictions
        pred_result = process_single_planet(planet_id, model, test_dir)
        
        # Format results - planet_id is already a string now
        result = {'planet_id': planet_id}
        
        # Add spectral predictions
        pred_mean = pred_result['mean']
        pred_unc = pred_result['uncertainty']
        
        # AIRS channels (283)
        for i in range(283):
            result[f'wl_{i+1}'] = float(pred_mean[i])
        
        # FGS1 channel
        result['wl_284'] = float(pred_mean[283])
        
        # Uncertainties - using correct column names
        for i in range(283):
            result[f'sigma_{i+1}'] = float(pred_unc[i])
        
        # Note: FGS1 uncertainty might not be needed - check sample_submission.csv
        
        results.append(result)
        
        # Memory cleanup every 100 planets
        if idx % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Create DataFrame
    submission_df = pd.DataFrame(results)
    
    # Verify column order matches sample submission
    sample_sub = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/sample_submission.csv')
    
    # Make sure we have exactly the columns in sample_sub
    submission_df = submission_df[sample_sub.columns]
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    
    return submission_df

# ============================================
# PART 4: OPTIMIZATION UTILITIES
# ============================================

def verify_submission(submission_df: pd.DataFrame) -> bool:
    """Verify submission format"""
    sample = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/sample_submission.csv')
    
    checks = {
        'shape_match': submission_df.shape == sample.shape,
        'columns_match': list(submission_df.columns) == list(sample.columns),
        'no_nulls': submission_df.isnull().sum().sum() == 0,
        'planet_ids_are_strings': submission_df['planet_id'].dtype == 'object',
        'values_are_numeric': all(submission_df.iloc[:, 1:].dtypes == 'float64')
    }
    
    print("Submission verification:")
    for check, passed in checks.items():
        print(f"  {check}: {'âœ“' if passed else 'âœ—'}")
    
    return all(checks.values())


import pandas as pd

adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
print(adc_info.dtypes)
print(adc_info.head())



submission_df = create_submission(model, test_dir, output_path='submission.csv')
verify_submission(submission_df)



import torch

# Automatically use GPU if available, else fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Initialize model
print("Initializing GDL-PINN model...")
model = GeometricPINNExoplanet(
    n_wavelengths=283,
    hidden_dim=256,
    n_graph_layers=4,
    dropout=0.0  # No dropout during inference
)
model.to(device)
model.eval()

# Note: In a real submission, you would load pre-trained weights here
# For now, we'll use random initialization (this won't give good results)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


from pathlib import Path
from tqdm.notebook import tqdm
import gc
import torch



# Path to test dataset
test_dir = Path('/kaggle/input/ariel-data-challenge-2025/test')

# Run model inference on test set and generate submission
submission_df = create_submission(model, test_dir, output_path='submission.csv')

print(f"Submission created with {len(submission_df)} predictions")
print("\nFirst 5 predictions:")
print(submission_df.head())

# Check if submission is valid before uploading to Kaggle
if verify_submission(submission_df):
    print("\nâœ“ Submission format is valid!")
else:
    print("\nâœ— Warning: Submission format issues detected!")


