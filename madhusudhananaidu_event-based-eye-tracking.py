!pip install mlflow tonic


configs = {
    "device": "cuda:0",
    "mlflow_path": "/kaggle/working/mlruns",
    "experiment_name": "trial_experiment",
    "data_dir": "/kaggle/input/event-based-eye-tracking-cvpr-2025/event_data/event_data",
    "run_name": "trial_run",
    "architecture": "BBModel",
    # "lr": 0.001,
    "num_epochs": 200,
    "batch_size": 32,
    "spatial_factor": 0.125,
    "temporal_subsample_factor": 1,
    "val_interval": 2,
    "save_k_best": 2,
    "pixel_tolerances": [5,10,15],
    "sensor_width": 640,
    "sensor_height": 480,
    "train_stride": 15,
    "val_stride": 30,
    "train_length": 30,
    "val_length": 30,
    "n_time_bins": 3,
    "voxel_grid_ch_normaization": False,
    "loss": "euclidean",
    "n_time_bins": 2,            # Number of channels in the event representation.
    "kernel_size_t": 3,          # Temporal convolution kernel size.
    "kernel_size_spatial": 3,    # Spatial convolution kernel size.
    "num_frames": 50,            # Number of event frames per batch.
    "learning_rate": 0.002,
    "weight_decay": 0.005,
}


import argparse, json, os, mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Load config file and create args
# config_file = 'train_baseline.json'
# with open(os.path.join('./configs', config_file), 'r') as f:
#     config = json.load(f)
args = argparse.Namespace(**configs)

# Set up MLflow tracking
mlflow.set_tracking_uri(args.mlflow_path)
mlflow.set_experiment(experiment_name=args.experiment_name)


#CNN_GRU
class CNN_GRU(nn.Module):
    """
        A baseline eye tracking which uses CNN + GRU to predict the pupil center coordinate
    """
    def __init__(self, args):
        super().__init__() 
        self.args = args
        self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.gru = nn.GRU(input_size=36192, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 2)


    def forward(self, x):
        # input is of shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        # permute height and width
        x = x.permute(0, 1, 3, 2)

        x= self.conv1(x)
        x= torch.relu(x)
        x= self.conv2(x)
        x= torch.relu(x)
        x= self.conv3(x)
        x= torch.relu(x)
        x= self.pool(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.gru(x)
        # output shape of x is (batch_size, seq_len, hidden_size)

        x = self.fc(x)
        # output is of shape (batch_size, seq_len, 2)
        return x


#ConvLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.
    It computes all gate operations via one convolution.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # use same padding to preserve spatial dimensions
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        
    def forward(self, input, h_cur, c_cur):
        # Concatenate input and previous hidden state along the channel dimension
        combined = torch.cat([input, h_cur], dim=1)
        conv_output = self.conv(combined)
        # Split the output into 4 parts for input, forget, output, and cell gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMStack(nn.Module):
    """
    Stacks multiple ConvLSTM layers.
    The output of one layer is used as the input to the next.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
            
    def forward(self, input):
        # Input shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len, _, height, width = input.size()
        layer_output = input
        for cell in self.cells:
            h, c = self.init_hidden(batch_size, cell.hidden_channels, height, width, input.device)
            outputs = []
            # Process sequence step by step
            for t in range(seq_len):
                h, c = cell(layer_output[:, t], h, c)
                outputs.append(h)
            # Stack outputs along the time dimension for the next layer
            layer_output = torch.stack(outputs, dim=1)
        return layer_output  # shape: (batch, seq_len, hidden_channels, height, width)
    
    def init_hidden(self, batch_size, hidden_channels, height, width, device):
        h = torch.zeros(batch_size, hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, hidden_channels, height, width, device=device)
        return h, c

class CNN_ConvLSTM(nn.Module):
    """
    A convolutional LSTM based eye tracking model.
    
    The model first applies 2D convolutions (and pooling) to each event-frame
    to extract spatial features. Then, a 4-layer ConvLSTM stack processes the
    sequence of feature maps while preserving spatial structure. Finally, global
    average pooling and a fully connected layer regress the pupil (x, y) coordinates.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Initial convolution layers to extract spatial features
        # (Input channels is set to args.n_time_bins, e.g. the number of channels in the voxel representation)
        self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Stack 4 layers of ConvLSTM; here we use 32 hidden channels in each layer.
        self.num_conv_lstm_layers = 4
        self.hidden_channels = 32
        self.convlstm = ConvLSTMStack(
            input_channels=32, 
            hidden_channels=self.hidden_channels, 
            kernel_size=3, 
            num_layers=self.num_conv_lstm_layers
        )
        
        # After the ConvLSTM, we perform global average pooling over the spatial dimensions
        # and then use a fully connected layer to regress the (x, y) coordinates.
        self.fc = nn.Linear(self.hidden_channels, 2)
        
    def forward(self, x):
        # x: (batch, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        # Process each frame independently via convolution layers.
        # Merge batch and sequence dimensions.
        x = x.view(batch_size * seq_len, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # Get the new spatial dimensions after conv+pool.
        _, feat_channels, feat_h, feat_w = x.shape
        # Reshape back to sequence: (batch, seq_len, feat_channels, feat_h, feat_w)
        x = x.view(batch_size, seq_len, feat_channels, feat_h, feat_w)
        
        # Pass the sequence through the stacked ConvLSTM layers.
        x = self.convlstm(x)  # Output shape: (batch, seq_len, hidden_channels, feat_h, feat_w)
        
        # For each time step, perform spatial global average pooling.
        x = x.mean(dim=[3, 4])  # Now shape: (batch, seq_len, hidden_channels)
        
        # Apply the final fully connected layer to regress the (x, y) coordinates.
        x = self.fc(x)  # Shape: (batch, seq_len, 2)
        return x

# # Example usage:
# if __name__ == "__main__":
#     class Args:
#         # Dummy hyperparameters for demonstration purposes
#         n_time_bins = 3           # for example, 3 channels in the voxel representation
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     args = Args()
#     model = CNN_ConvLSTM(args).to(args.device)
#     # Create a dummy input: batch size = 2, sequence length = 10, 3 channels, height=80, width=60
#     dummy_input = torch.randn(2, 10, args.n_time_bins, 80, 60).to(args.device)
#     output = model(dummy_input)
#     print("Output shape:", output.shape)  # Expected: (2, 10, 2)



#CTEM
class CETM(nn.Module):
    """
    Consistent Eye Tracking Model (CETM)
    
    The model implements:
      1. Representation enhancement: a two-layer convolutional block that refines the binary map representation.
      2. Tracking predictor: three convolution blocks extract spatial features, which are then fed into a GRU to capture temporal context.
         Finally, a fully connected layer regresses the (x, y) coordinates.
         
    Input:
      - x: a tensor of shape (batch_size, seq_len, channels, height, width)
           where the channels correspond to the binary map representation (e.g., 1 or 2 channels)
           
    Output:
      - A tensor of shape (batch_size, seq_len, 2) with the predicted pupil coordinates.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # --- Representation Enhancement Module ---
        # Enhance the input binary map (Bina-rep) with two convolutional layers.
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(args.n_time_bins, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, args.n_time_bins, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(args.n_time_bins),
            nn.ReLU(inplace=True)
        )
        
        # --- Tracking Predictor ---
        # Three convolution blocks; each block: Conv2d -> BatchNorm2d -> ReLU -> Average Pooling.
        # These blocks are designed to extract pupil-space features from each frame.
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Assume the input event resolution is downsampled to 80x60.
        # For example, let’s assume height=60, width=80.
        # After conv_block1: (60/2, 80/2) -> (30, 40)
        # After conv_block2: (30/2, 40/2) -> (15, 20)
        # After conv_block3: (15/2, 20/2) -> (7, 10) [using floor division]
        # The number of output channels after block3 is 32.
        self.flattened_size = 32 * 7 * 10  # 32 channels * 7 * 10 spatial locations
        
        # GRU to capture temporal dependencies.
        self.gru = nn.GRU(input_size=self.flattened_size, hidden_size=128, num_layers=1, batch_first=True)
        
        # Fully connected layer to regress the final (x, y) coordinates.
        self.fc = nn.Linear(128, 2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        
        # Representation enhancement: process each frame independently.
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.enhance_conv(x)
        
        # Tracking predictor: pass through three convolution blocks.
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Flatten spatial dimensions: now shape becomes (batch_size, seq_len, flattened_size)
        x = x.view(batch_size, seq_len, -1)
        
        # Pass through GRU to capture temporal context.
        x, _ = self.gru(x)  # Output shape: (batch_size, seq_len, hidden_size)
        
        # Final fully connected layer: regress pupil coordinates.
        x = self.fc(x)  # Output shape: (batch_size, seq_len, 2)
        return x


#CETM Improved
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Define a Residual Convolution Block
# ------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        # If dimensions differ, use a 1x1 convolution to match
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = None

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.residual is not None:
            identity = self.residual(identity)
        out += identity
        out = self.relu(out)
        return out

# ------------------------------
# Improved Consistent Eye Tracking Model (CETM_Improved)
# ------------------------------
class CETM_Improved(nn.Module):
    """
    Improved Consistent Eye Tracking Model (CETM_Improved)
    
    This model implements:
      1. Representation enhancement using residual blocks.
      2. A tracking predictor that uses three residual convolution blocks (each followed by average pooling)
         to extract spatial features from each frame.
      3. A bidirectional GRU with dropout for temporal modeling.
      4. A fully connected layer to regress the pupil (x, y) coordinates.
         
    Expected input shape: (batch, seq_len, channels, height, width)
    Output shape: (batch, seq_len, 2)
    """
    def __init__(self, args):
        super(CETM_Improved, self).__init__()
        self.args = args
        # --- Representation Enhancement Module ---
        # Enhance the binary map representation with residual blocks.
        # (Assuming the input has args.n_time_bins channels.)
        self.enhance_conv = nn.Sequential(
            ResidualConvBlock(args.n_time_bins, 16),
            ResidualConvBlock(16, args.n_time_bins)
        )
        
        # --- Tracking Predictor ---
        # Use three residual blocks to extract spatial features.
        # For an input resolution of 80x60 (width x height):
        # After pool1: 40x30; after pool2: 20x15; after pool3: 10x7 (using kernel_size=2, stride=2).
        self.block1 = ResidualConvBlock(args.n_time_bins, 32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block2 = ResidualConvBlock(32, 64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block3 = ResidualConvBlock(64, 32)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Compute the flattened size after three pooling layers:
        # Assuming input spatial dimensions are 80 (width) x 60 (height):
        # width: 80 -> 40 -> 20 -> 10; height: 60 -> 30 -> 15 -> 7 (using floor division)
        self.flattened_size = 32 * 10 * 7  # channels * width * height
        
        # --- Temporal Modeling ---
        # Use a bidirectional GRU with dropout to capture temporal context.
        # Bidirectional GRU doubles the output features (hidden_size * 2).
        self.gru = nn.GRU(
            input_size=self.flattened_size, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        # Fully connected layer maps GRU output (128*2=256) to (x, y)
        self.fc = nn.Linear(256, 2)
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        # Process each frame independently by merging batch and sequence dimensions
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Representation enhancement
        x = self.enhance_conv(x)
        
        # Tracking predictor using residual convolution blocks and pooling
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        
        # Flatten the spatial dimensions
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, flattened_size)
        
        # Temporal modeling with bidirectional GRU
        x, _ = self.gru(x)  # x shape: (batch, seq_len, 256)
        
        # Regress pupil coordinates for each time step
        x = self.fc(x)  # (batch, seq_len, 2)
        return x

# # ------------------------------
# # Example Usage
# # ------------------------------
# if __name__ == "__main__":
#     # Dummy configuration (replace with your actual configuration)
#     class Args:
#         # Number of channels in the input event voxel representation (e.g., binary map)
#         n_time_bins = 1  
#         # Device configuration (use 'cuda' if available)
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     args = Args()
    
#     # Instantiate the improved model and move to device.
#     model = CETM_Improved(args).to(args.device)
    
#     # Create a dummy input: batch size = 4, sequence length = 10,
#     # channels = n_time_bins, height = 60, width = 80.
#     dummy_input = torch.randn(4, 10, args.n_time_bins, 60, 80).to(args.device)
#     output = model(dummy_input)
#     print("Output shape:", output.shape)  # Expected: (4, 10, 2)



class CausalTemporalConv(nn.Module):
    """
    Applies a causal convolution along the temporal dimension using 3D convolution.
    Input shape: (B, C, T, H, W). Uses padding only at the beginning of the temporal dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size_t=3):
        super().__init__()
        self.kernel_size_t = kernel_size_t
        # For causal convolution, we pad (kernel_size_t - 1) at the beginning of the time dimension.
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size_t, 1, 1),
            padding=(0, 0, 0)  # Manual padding for causality.
        )
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        pad = (0, 0, 0, 0, self.kernel_size_t - 1, 0)  # Pad only at the beginning of time.
        x = F.pad(x, pad, mode='constant', value=0)
        return self.conv(x)

class SpatialConv(nn.Module):
    """
    Applies a spatial convolution with kernel size (1, k, k) on input of shape (B, C, T, H, W).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding)
        )
        
    def forward(self, x):
        return self.conv(x)

class FactorizedSpatioTemporalBlock(nn.Module):
    """
    A (1+2)D factorized spatio-temporal convolution block.
    First applies a causal temporal convolution, then a spatial convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size_t=3, kernel_size_spatial=3):
        super().__init__()
        self.temporal_conv = CausalTemporalConv(in_channels, out_channels, kernel_size_t)
        self.spatial_conv = SpatialConv(out_channels, out_channels, kernel_size_spatial)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

###############################################
# Detector Head Inspired by CenterNet
###############################################

class DetectorHead(nn.Module):
    """
    The detector head consists of two spatial convolutional layers.
    It takes a feature map per time frame and outputs a grid (3x4) where each grid cell predicts:
      - a probability (pupil presence)
      - relative x and y offsets.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, 3, kernel_size=1)  # 3 outputs per grid cell.
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

###############################################
# Complete BigBrains Model
###############################################

class BBModel(nn.Module):
    """
    Lightweight spatio-temporal network for online eye tracking.
    
    Implements:
      - A causal spatio-temporal backbone using (1+2)D factorized convolutions.
      - A detector head inspired by CenterNet that outputs a 3×4 grid for each frame.
      
    The temporal layers are causal and are designed to work with a FIFO buffer during streaming inference.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_channels = args.n_time_bins  # e.g., number of channels in the event representation
        
        # Backbone: stack three factorized spatio-temporal blocks.
        self.block1 = FactorizedSpatioTemporalBlock(in_channels, 16, kernel_size_t=args.kernel_size_t, kernel_size_spatial=args.kernel_size_spatial)
        self.block2 = FactorizedSpatioTemporalBlock(16, 32, kernel_size_t=args.kernel_size_t, kernel_size_spatial=args.kernel_size_spatial)
        self.block3 = FactorizedSpatioTemporalBlock(32, 64, kernel_size_t=args.kernel_size_t, kernel_size_spatial=args.kernel_size_spatial)
        
        # Adaptive average pooling to produce a fixed spatial grid of 3x4 (time dimension remains unchanged).
        self.adapt_pool = nn.AdaptiveAvgPool3d((None, 3, 4))
        
        # Detector head to process each time frame.
        self.detector_head = DetectorHead(in_channels=64)
        
    def forward(self, x):
        # x: (B, T, channels, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # Convert to (B, channels, T, H, W)
        
        # Backbone: apply factorized spatio-temporal blocks.
        x = self.block1(x)  # (B, 16, T, H, W)
        x = self.block2(x)  # (B, 32, T, H, W)
        x = self.block3(x)  # (B, 64, T, H, W)
        
        # Adaptive pooling to get fixed spatial grid of 3x4.
        x = self.adapt_pool(x)  # (B, 64, T, 3, 4)
        
        # Process each time frame with the detector head.
        B, C, T, H_grid, W_grid = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 64, 3, 4)
        x = x.view(B * T, C, H_grid, W_grid)
        x = self.detector_head(x)  # (B*T, 3, 3, 4)
        x = x.view(B, T, 3, 3, 4)
        return x


#Metrics
import torch
import torch.nn as nn
import numpy as np


def p_acc(target, prediction, width_scale, height_scale, pixel_tolerances=[1,3,5,10]):
    """
    Calculate the accuracy of prediction
    :param target: (N, seq_len, 2) tensor, seq_len could be 1
    :param prediction: (N, seq_len, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)

    bs_times_seqlen = target.shape[0]
    return total_correct, bs_times_seqlen


def p_acc_wo_closed_eye(target, prediction, width_scale, height_scale, pixel_tolerances=[1,3,5,10]):
    """
    Calculate the accuracy of prediction, with p tolerance and only calculated on those with fully opened eyes
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor, the last dimension is whether the eye is closed
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 3)
    prediction = prediction.reshape(-1, 2)

    dis = target[:,:2] - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)
    # check if there is nan in dist
    assert torch.sum(torch.isnan(dist)) == 0

    eye_closed = target[:,2] # 1 is closed eye
    # get the total number frames of those with fully opened eyes
    total_open_eye_frames = torch.sum(eye_closed == 0)

    # get the indices of those with closed eyes
    eye_closed_idx = torch.where(eye_closed == 1)[0]
    dist[eye_closed_idx] = np.inf
    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)
        assert total_correct[f'p{p_tolerance}'] <= total_open_eye_frames

    return total_correct, total_open_eye_frames.item()


def px_euclidean_dist(target, prediction, width_scale, height_scale):
    """
    Calculate the total pixel euclidean distance between target and prediction
    in a batch over the sequence length
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 3)[:, :2]
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_px_euclidean_dist = torch.sum(dist)
    sample_numbers = target.shape[0]
    return total_px_euclidean_dist, sample_numbers


class weighted_MSELoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weights = weights
        self.mseloss = nn.MSELoss(reduction='none')
        
    def forward(self, inputs, targets):
        batch_loss = self.mseloss(inputs, targets) * self.weights
        if self.reduction == 'mean':
            return torch.mean(batch_loss)
        elif self.reduction == 'sum':
            return torch.sum(batch_loss)
        else:
            return batch_loss


# from model.BaselineEyeTrackingModel import CNN_GRU
# from torch.utils.metrics import weighted_MSELoss

# Initialize model and move it to the correct device (GPU/CPU)
model = eval(args.architecture)(args).to(args.device)

# Create the optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# # Select loss function based on args
# if args.loss == "mse":
#     criterion = nn.MSELoss()
# elif args.loss == "weighted_mse":
#     criterion = weighted_MSELoss(
#         weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to(args.device),
#         reduction='mean'
#     )
# else:
#     raise ValueError("Invalid loss name")
def euclidean_loss(pred, target, eps=1e-6):
    # Compute the Euclidean distance for each sample
    # pred and target are assumed to have shape (batch, seq_len, 2)
    loss = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1) + eps)
    return torch.mean(loss)

if args.loss == "mse":
    criterion = nn.MSELoss()
elif args.loss == "weighted_mse":
    criterion = weighted_MSELoss(weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to(args.device), reduction='mean')
elif args.loss == "euclidean":
    criterion = euclidean_loss  # use our custom Euclidean distance loss
else:
    raise ValueError("Invalid loss name")


import numpy as np
# import torch
from tonic.slicers import (
    slice_events_by_time,
)
from typing import Any, List, Tuple

def custom_to_voxel_grid_numpy(events, sensor_size, n_time_bins=10):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Implements the event volume from Zhu et al. 2019, Unsupervised event-based learning of optical
    flow, depth, and egomotion.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H].
        n_time_bins: number of bins in the temporal axis of the voxel grid.

    Returns:
        numpy array of n event volumes (n,w,h,t)
    """
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2

    voxel_grid = np.zeros((n_time_bins, sensor_size[1], sensor_size[0]), float).ravel()

    # normalize the event timestamps so that they lie between 0 and n_time_bins
    time_diff = events["t"][-1] - events["t"][0]
    if time_diff < 1e-6:  # avoid zero-vidision
        ts = np.zeros_like(events["t"], dtype=float)
    else:
        ts = n_time_bins * (events["t"].astype(float) - events["t"][0]) / time_diff

    xs = events["x"].astype(int)
    ys = events["y"].astype(int)
    pols = events["p"]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + tis[valid_indices] * sensor_size[0] * sensor_size[1],
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + (tis[valid_indices] + 1) * sensor_size[0] * sensor_size[1],
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(
        voxel_grid, (n_time_bins, 1, sensor_size[1], sensor_size[0])
    )

    return voxel_grid

class SliceByTimeEventsTargets:
    """
    Modified from tonic.slicers.SliceByTimeEventsTargets in the Tonic Library

    Slices an event array along fixed time window and overlap size. The number of bins depends
    on the length of the recording. Targets are copied.

    >        <overlap>
    >|    window1     |
    >        |   window2     |

    Parameters:
        time_window (int): time for window length (same unit as event timestamps)
        overlap (int): overlap (same unit as event timestamps)
        include_incomplete (bool): include the last incomplete slice that has shorter time
    """

    def __init__(self,time_window, overlap=0.0, seq_length=30, seq_stride=15, include_incomplete=False) -> None:
        self.time_window= time_window
        self.overlap= overlap
        self.seq_length=seq_length
        self.seq_stride=seq_stride
        self.include_incomplete=include_incomplete

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        t = data["t"]
        stride = self.time_window - self.overlap
        assert stride > 0

        if self.include_incomplete:
            n_slices = int(np.ceil(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        else:
            n_slices = int(np.floor(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        n_slices = max(n_slices, 1)  # for strides larger than recording time

        window_start_times = np.arange(n_slices) * stride + t[0]
        window_end_times = window_start_times + self.time_window
        indices_start = np.searchsorted(t, window_start_times)[:n_slices]
        indices_end = np.searchsorted(t, window_end_times)[:n_slices]

        if not self.include_incomplete:
            # get the strided indices for loading labels
            label_indices_start = np.arange(0, targets.shape[0]-self.seq_length, self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
        else:
            label_indices_start = np.arange(0, targets.shape[0], self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
            # the last label indices end should be the last label
            label_indices_end[-1] = targets.shape[0]

        assert targets.shape[0] >= label_indices_end[-1]

        return list(zip(zip(indices_start, indices_end), zip(label_indices_start, label_indices_end)))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ):
        return_data = []
        return_target = []
        for tuple1, tuple2 in metadata:
            return_data.append(data[tuple1[0]:tuple1[1]])
            return_target.append(targets[tuple2[0]:tuple2[1]])

        return return_data, return_target


class SliceLongEventsToShort:
    def __init__(self, time_window, overlap, include_incomplete):
        """
        Initialize the transformation.

        Args:
        - time_window (int): The length of each sub-sequence.
        """
        self.time_window = time_window
        self.overlap = overlap
        self.include_incomplete = include_incomplete

    def __call__(self, events):
        return slice_events_by_time(events, self.time_window, self.overlap, self.include_incomplete)


class EventSlicesToVoxelGrid:
    def __init__(self, sensor_size, n_time_bins, per_channel_normalize):
        """
        Initialize the transformation.

        Args:
        - sensor_size (tuple): The size of the sensor.
        - n_time_bins (int): The number of time bins.
        """
        self.sensor_size = sensor_size
        self.n_time_bins = n_time_bins
        self.per_channel_normalize = per_channel_normalize

    def __call__(self, event_slices):
        """
        Apply the transformation to the given event slices.

        Args:
        - event_slices (Tensor): The input event slices.

        Returns:
        - Tensor: A batched tensor of voxel grids.
        """
        voxel_grids = []
        for event_slice in event_slices:
            voxel_grid = custom_to_voxel_grid_numpy(event_slice, self.sensor_size, self.n_time_bins)
            voxel_grid = voxel_grid.squeeze(-3)
            if self.per_channel_normalize:
                # Calculate mean and standard deviation only at non-zero values
                non_zero_entries = (voxel_grid != 0)
                for c in range(voxel_grid.shape[0]):
                    mean_c = voxel_grid[c][non_zero_entries[c]].mean()
                    std_c = voxel_grid[c][non_zero_entries[c]].std()

                    voxel_grid[c][non_zero_entries[c]] = (voxel_grid[c][non_zero_entries[c]] - mean_c) / (std_c + 1e-10)
            voxel_grids.append(voxel_grid)
        return np.array(voxel_grids).astype(np.float32)


class SplitSequence:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride

    def __call__(self, sequence, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - sequence (Tensor): The input sequence of frames.
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of sub-sequences.
        - Tensor: A batched tensor of corresponding labels.
        """

        sub_sequences = []
        sub_labels = []

        for i in range(0, len(sequence) - self.sub_seq_length + 1, self.stride):
            sub_seq = sequence[i:i + self.sub_seq_length]
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_sequences.append(sub_seq)
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_sequences), np.stack(sub_labels)
    

class SplitLabels:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride
        # print(f"stride is {self.stride}")

    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        sub_labels = []
        
        for i in range(0, len(labels) - self.sub_seq_length + 1, self.stride):
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_labels)

class ScaleLabel:
    def __init__(self, scaling_factor):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.scaling_factor = scaling_factor


    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:,:2] =  labels[:,:2] * self.scaling_factor
        return labels
    
class LabelTemporalSubsample:
    def __init__(self, temporal_subsample_factor):
        self.temp_subsample_factor = temporal_subsample_factor

    def __call__(self, labels):
        """
        temorally subsample the labels
        """
        interval = int(1/self.temp_subsample_factor)
        return labels[::interval]
    

class NormalizeLabel:
    def __init__(self, pseudo_width, pseudo_height):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.pseudo_width = pseudo_width
        self.pseudo_height = pseudo_height
    
    def __call__(self, labels):
        """
        Apply normalization on label, with pseudo width and height

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:, 0] = labels[:, 0] / self.pseudo_width
        labels[:, 1] = labels[:, 1] / self.pseudo_height
        return labels



import tonic
import tonic.transforms as transforms
# from dataset import ScaleLabel, NormalizeLabel, LabelTemporalSubsample

factor = args.spatial_factor  # spatial downsample factor
temp_subsample_factor = args.temporal_subsample_factor  # e.g., 5 to reduce 100Hz to 20Hz

label_transform = transforms.Compose([
    ScaleLabel(factor),
    LabelTemporalSubsample(temp_subsample_factor),
    NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
])


import os
from typing import Any, Callable, Optional, Tuple
import h5py
import numpy as np

from tonic.dataset import Dataset

class ThreeETplus_Eyetracking(Dataset):
    """3ET DVS eye tracking `3ET <https://github.com/qinche106/cb-convlstm-eyetracking>`_
    ::

        @article{chen20233et,
            title={3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network},
            author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
            journal={arXiv preprint arXiv:2308.11771},
            year={2023}
        }

        authors: Qinyu Chen^{1,2}, Zuowen Wang^{1}
        affiliations: 1. Institute of Neuroinformatics, University of Zurich and ETH Zurich, Switzerland
                      2. Univeristy of Leiden, Netherlands

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        split (string, optional): The dataset split to use, ``train`` or ``val``.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.

    Returns:
         A dataset object that can be indexed or iterated over.
         One sample returns a tuple of (events, targets).
    """

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        data_dir = save_to
        data_list_dir = '/kaggle/input/dataset'
        # Load filenames from the provided lists
        if split == "train":
            filenames = self.load_filenames(os.path.join(data_list_dir, "train_files.txt"))
        elif split == "val":
            filenames = self.load_filenames(os.path.join(data_list_dir, "val_files.txt"))
        elif split == "test":
            filenames = self.load_filenames(os.path.join(data_list_dir, "test_files.txt"))
        else:
            raise ValueError("Invalid split name")

        # Get the data file paths and target file paths
        if split == "train" or split == "val":
            self.data = [os.path.join(data_dir, "train", f, f + ".h5") for f in filenames]
            self.targets = [os.path.join(data_dir, "train", f, "label.txt") for f in filenames]
        elif split == "test":
            self.data = [os.path.join(data_dir, "test", f, f + ".h5") for f in filenames]
            # for test set, we load the placeholder labels with all zeros
            self.targets = [os.path.join(data_dir, "test", f, "label_zeros.txt") for f in filenames]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        # get events from .h5 file
        with h5py.File(self.data[index], "r") as f:
            # original events.dtype is dtype([('t', '<u8'), ('x', '<u8'), ('y', '<u8'), ('p', '<u8')])
            # t is in us
            events = f["events"][:].astype(self.dtype)
            events['p'] = events['p']*2 -1  # convert polarity to -1 and 1
            
        # load the sparse labels
        with open(self.targets[index], "r") as f:
            # target is at the frequency of 100 Hz. It will be downsampled to 20 Hz in the target transformation
            target = np.array(
                [list(map(float, line.strip('()\n').split(', '))) for line in f.readlines()], np.float32)

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present()

    def load_filenames(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]


# from dataset import ThreeETplus_Eyetracking
# from tonic import transforms

train_data_orig = ThreeETplus_Eyetracking(
    save_to=args.data_dir, split="train",
    transform=transforms.Downsample(spatial_factor=factor), 
    target_transform=label_transform
)
val_data_orig = ThreeETplus_Eyetracking(
    save_to=args.data_dir, split="val",
    transform=transforms.Downsample(spatial_factor=factor),
    target_transform=label_transform
)


# from dataset import SliceByTimeEventsTargets, SliceLongEventsToShort, EventSlicesToVoxelGrid

slicing_time_window = args.train_length * int(10000 / temp_subsample_factor)  # microseconds
train_stride_time = int(10000 / temp_subsample_factor * args.train_stride)  # microseconds

train_slicer = SliceByTimeEventsTargets(
    slicing_time_window, overlap=slicing_time_window - train_stride_time,
    seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=False
)
val_slicer = SliceByTimeEventsTargets(
    slicing_time_window, overlap=0,
    seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=False
)

post_slicer_transform = transforms.Compose([
    SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0, include_incomplete=True),
    EventSlicesToVoxelGrid(
        sensor_size=(int(640*factor), int(480*factor), 2),
        n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization
    )
])


from tonic import SlicedDataset, DiskCachedDataset

train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform,
                           metadata_path=f"./metadata/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")
val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform,
                         metadata_path=f"./metadata/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}")

train_data = DiskCachedDataset(train_data, cache_path=f'./cached_dataset/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}')
val_data = DiskCachedDataset(val_data, cache_path=f'./cached_dataset/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}')


train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=int(os.cpu_count()-2), pin_memory=True
)
val_loader = DataLoader(
    val_data, batch_size=args.batch_size, shuffle=False,
    num_workers=int(os.cpu_count()-2)
)


import torch
import os
# from utils.metrics import p_acc, p_acc_wo_closed_eye, px_euclidean_dist

def train_epoch(model, pbar, criterion, optimizer, args):
    model.train()
    running_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_euc_error_all  = {f'euc_error_all':0}  # averaged euclidean distance
    total_samples_all, total_sample_p_euc_error_all  = 0, 0

    for batch in pbar:
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs.to(args.device))
        #taking only the last frame's label, and first two dim are coordinate, last is open or close so discarded
        targets = targets.to(args.device)
        loss = criterion(outputs, targets[:,:, :2]) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # calculate pixel tolerated accuracy
        p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, :], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor, \
                                    pixel_tolerances=args.pixel_tolerances)
        total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
        total_samples_all += batch_size

        # calculate averaged euclidean distance
        p_euc_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :], outputs[:, :, :], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor)
        total_p_euc_error_all = {f'euc_error_all': (total_p_euc_error_all[f'euc_error_all'] + p_euc_error_total).item()}
        total_sample_p_euc_error_all += bs_times_seqlen
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': loss.item()})
    
    metrics = {'tr_p_acc_all': {f'tr_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
               'tr_p_euc_error_all': {f'tr_p_euc_error_all': (total_p_euc_error_all[f'euc_error_all']/total_sample_p_euc_error_all)}}
    
    return model, running_loss / len(pbar), metrics


def validate_epoch(model, pbar, criterion, args):
    model.eval()
    running_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_euc_error_all  = {f'euc_error_all':0}
    total_samples_all, total_sample_p_euc_error_all  = 0, 0
    with torch.no_grad():
        for batch in pbar:
            inputs, targets = batch
            outputs = model(inputs.to(args.device))
            targets = targets.to(args.device)
            loss = criterion(outputs, targets[:,:, :2]) 
            running_loss += loss.item()

            # calculate pixel tolerated accuracy
            p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, :], \
                                    width_scale=args.sensor_width*args.spatial_factor, \
                                    height_scale=args.sensor_height*args.spatial_factor, \
                                        pixel_tolerances=args.pixel_tolerances)
            total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
            total_samples_all += batch_size

            # calculate averaged euclidean distance
            p_euc_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :], outputs[:, :, :], \
                                    width_scale=args.sensor_width*args.spatial_factor, \
                                    height_scale=args.sensor_height*args.spatial_factor)
            total_p_euc_error_all = {f'euc_error_all': (total_p_euc_error_all[f'euc_error_all'] + p_euc_error_total).item()}
            total_sample_p_euc_error_all += bs_times_seqlen
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': loss.item()})

    metrics = {'val_p_acc_all': {f'val_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
                'val_p_euc_error_all': {f'val_p_euc_error_all': (total_p_euc_error_all[f'euc_error_all']/total_sample_p_euc_error_all)}}
    
    return running_loss / len(pbar), metrics


def top_k_checkpoints(args, artifact_uri):
    """
    only save the top k model checkpoints with the lowest validation loss.
    """
    # list all files ends with .pth in artifact_uri
    model_checkpoints = [f for f in os.listdir(artifact_uri) if f.endswith(".pth")]

    # but only save at most args.save_k_best models checkpoints
    if len(model_checkpoints) > args.save_k_best:
        # sort all model checkpoints by validation loss in ascending order
        model_checkpoints = sorted([f for f in os.listdir(artifact_uri) if f.startswith("model_best_ep")], \
                                    key=lambda x: float(x.split("_")[-1][:-4]))
        # delete the model checkpoint with the largest validation loss
        os.remove(os.path.join(artifact_uri, model_checkpoints[-1]))




# from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, args):
    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}")
        model, train_loss, metrics = train_epoch(model, train_pbar, criterion, optimizer, args)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metrics(metrics['tr_p_acc_all'], step=epoch)
        mlflow.log_metrics(metrics['tr_p_euc_error_all'], step=epoch)

        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{args.num_epochs}")
            val_loss, val_metrics = validate_epoch(model, val_pbar, criterion, args)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(),
                          f"model_best_ep{epoch}_val_loss_{val_loss:.4f}.pth"))
                top_k_checkpoints(args, mlflow.get_artifact_uri())
            print(f"[Validation] Epoch {epoch+1}: Val Loss: {val_loss:.4f}")
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metrics(val_metrics['val_p_acc_all'], step=epoch)
            mlflow.log_metrics(val_metrics['val_p_euc_error_all'], step=epoch)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
    return model

# Run the training within an MLflow run:
with mlflow.start_run(run_name=args.run_name):
    mlflow.log_params(vars(args))
    with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), 'w') as f:
        json.dump(vars(args), f)
    model = train(model, train_loader, val_loader, criterion, optimizer, args)
    torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), f"model_last_epoch{args.num_epochs}.pth"))


test_config = {
    "device": "cuda:0",
    "data_dir": "/kaggle/input/event-based-eye-tracking-cvpr-2025/event_data/event_data",
    "checkpoint": "/kaggle/working/mlruns/636469345475994249/c53becb1ad65418fb6a52bd597a34c15/artifacts/model_best_ep35_val_loss_0.0824.pth",
    "architecture": "CETM_Improved",
    "batch_size": 1,
    "spatial_factor": 0.125,
    "temporal_subsample_factor": 1,
    "pixel_tolerances": [5,10,15],
    "sensor_width": 640,
    "sensor_height": 480,
    "test_stride": 30,
    "test_length": 30,
    "n_time_bins": 3,
    "voxel_grid_ch_normaization": False
}


"""
Author: Zuowen Wang
Affiliation: Institute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wangzu@ethz.ch
"""

import argparse, json, os, mlflow, csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from model.BaselineEyeTrackingModel import CNN_GRU
# from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, \
#     LabelTemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
#     EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset

def main():
    # Load hyperparameters from JSON configuration file
    args = argparse.Namespace(**test_config)
    # if args.config_file:
    #     with open(os.path.join('./configs', args.config_file), 'r') as f:
    #         config = json.load(f)
    #     # Overwrite hyperparameters with command-line arguments if provided
    #     for key, value in vars(args).items():
    #         if value is not None:
    #             config[key] = value
    #     args = argparse.Namespace(**config)
    # else:
    #     raise ValueError("Please provide a JSON configuration file.")

    # Dump the args to a JSON file in the MLflow artifact directory
    with open('/kaggle/working/mlruns/636469345475994249/c53becb1ad65418fb6a52bd597a34c15/artifacts/args.json', 'w') as f:
        json.dump(vars(args), f)

    # Define your model and move it to the desired device
    model = eval(args.architecture)(args).to(args.device)

    # For testing, we need to use the same spatial and temporal factors as in training.
    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

    # Define label transformation: scale, subsample temporally, and normalize the labels
    label_transform = transforms.Compose([
        ScaleLabel(factor),
        LabelTemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    # Load the raw test event data and apply spatial downsampling and label transformation
    test_data_orig = ThreeETplus_Eyetracking(
        save_to=args.data_dir, split="test",
        transform=transforms.Downsample(spatial_factor=factor),
        target_transform=label_transform
    )

    # Compute slicing parameters for the test data (time window in microseconds)
    slicing_time_window = args.test_length * int(10000 / temp_subsample_factor)  # microseconds

    # Create a slicer for the test sequences (no overlap; include incomplete sequences)
    test_slicer = SliceByTimeEventsTargets(
        slicing_time_window, overlap=0,
        seq_length=args.test_length, seq_stride=args.test_stride, include_incomplete=True
    )

    # Post-slicer transformation: convert slices to an event voxel grid representation
    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(
            sensor_size=(int(640*factor), int(480*factor), 2),
            n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization
        )
    ])

    # Create the sliced dataset for test data
    test_data = SlicedDataset(test_data_orig, test_slicer, transform=post_slicer_transform)

    # Optionally, you can cache the dataset to disk for faster subsequent runs
    # test_data = DiskCachedDataset(test_data, cache_path=f'./cached_dataset/test_l{args.test_length}s{args.test_stride}_ch{args.n_time_bins}')

    # For testing, ensure batch_size is 1 (this works in combination with include_incomplete=True)
    assert args.batch_size == 1, "Batch size must be 1 for testing to avoid collate function errors."
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=int(os.cpu_count()-2)
    )

    # Load model weights from the provided checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load('/kaggle/working/mlruns/636469345475994249/c53becb1ad65418fb6a52bd597a34c15/artifacts/model_best_ep35_val_loss_0.0824.pth'))
    else:
        raise ValueError("Please provide a checkpoint file.")

    # Run inference on the test set and write predictions to a CSV file.
    # with open('/kaggle/working/submission.csv', 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile, delimiter=',')
    # # Write header: row_id, x, y
    #     csv_writer.writerow(['row_id', 'x', 'y'])
    #     row_id = 0
    #     for batch_idx, (data, target_placeholder) in enumerate(test_loader):
    #         data = data.to(args.device)
    #         output = model(data)
        
    #     # Cast the output back to the downsampled sensor space (e.g., 640x480 scaled by factor)
    #         output = output * torch.tensor((640*factor, 480*factor)).to(args.device)
        
    #     # Use the output shape for iteration
    #         n_samples, n_frames, _ = output.shape
    #         for sample in range(n_samples):
    #             for frame_id in range(n_frames):
    #                 row_to_write = output[sample][frame_id].tolist()
    #             # Prepend the row_id to the output row
    #                 row_to_write.insert(0, row_id)
    #                 csv_writer.writerow(row_to_write)
    #                 row_id += 1
    #                 print(f"Processed row {row_id}")

    # Run inference on the test set and write predictions to a CSV file.
    with open('/kaggle/working/submission.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        # Write header: row_id, x, y
        csv_writer.writerow(['row_id', 'x', 'y'])
        row_id = 0
        for batch_idx, (data, target_placeholder) in enumerate(test_loader):
            data = data.to(args.device)
            output = model(data)

            # Cast the output back to the downsampled sensor space (e.g., 640x480 scaled by factor)
            output = output * torch.tensor((640*factor, 480*factor)).to(args.device)

            n_samples = target_placeholder.shape[0]
            n_frames = target_placeholder.shape[1]
            for sample in range(n_samples):
                for frame_id in range(n_frames):
                    row_to_write = output[sample][frame_id].tolist()
                    # Prepend the row_id to the output row
                    row_to_write.insert(0, row_id)
                    csv_writer.writerow(row_to_write)
                    row_id += 1
                    print(f"Processed row {row_id}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    
    # Path to the configuration JSON file
    # parser.add_argument("--config_file", type=str, default='test_config', 
    #                     help="Path to JSON configuration file")
    # # Path to the model checkpoint
    # parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    # # Output CSV file path for submission predictions
    # parser.add_argument("--output_path", type=str, default='./submission.csv')

    # args = parser.parse_args()

    main()



import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings

warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    f'{category.__name__}: {message}\n'

class CausalGroupNorm(nn.GroupNorm):
    """A GroupNorm that does not use temporal statistics, to ensure causality
    """
    def __init__(self, num_groups, num_channels, **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)
        
    def forward(self, input):
        x = input.moveaxis(1, 2)  # (B, T, C, H, W)
        x_shape = x.shape
        x = x.flatten(0, 1)  # (B * T, C, H, W)
        x = super().forward(x).reshape(x_shape)
        return x.moveaxis(1, 2)  # (B, C, T, H, W)


act_layer = lambda: nn.ReLU()
bn_block = lambda features: nn.Sequential(nn.BatchNorm3d(features), act_layer())
gn_block = lambda features: nn.Sequential(CausalGroupNorm(4, features), act_layer())
pw_conv = lambda in_channels, out_channels: nn.Conv3d(in_channels, out_channels, 1, bias=False)


class SpatialBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 depthwise=False, 
                 kernel_size=1,
                 full_conv3d=False, 
                 norms='mixed'):
        super().__init__()
        kernel = (kernel_size,3,3)
        self.kernel_size = kernel_size
        self.full_conv3d = full_conv3d
        self.norms = norms
        self.streaming_mode = False
        self.fifo = None  # for streaming inference

        if self.norms=='all_gn':
            norm_block = gn_block
        else :
            norm_block = bn_block

        if depthwise:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel, (1, 2, 2), (0, 1, 1), groups=in_channels, bias=False), 
                norm_block(in_channels), 
                pw_conv(in_channels, out_channels), 
                norm_block(out_channels), 
            )
            
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, (1, 2, 2), (0, 1, 1), bias=False), 
                norm_block(out_channels), 
            )
        
    def streaming(self, enabled=True):
        if enabled:
            assert not self.training, "Can only use streaming mode during evaluation."
        self.streaming_mode = enabled
        
    def reset_memory(self):
        self.fifo = None
    
    def forward(self, input):
        if self.full_conv3d: 
            if self.streaming_mode:
                return self._streaming_forward(input)
            input = F.pad(input, (0, 0, 0, 0, self.kernel_size - 1, 0))
            return self.block(input)
        else:         
            return self.block(input)
            
    def _streaming_forward(self, input):
        if self.fifo is None:
            self.fifo = torch.zeros(*input.shape[:2], self.kernel_size, *input.shape[3:]).type_as(input)
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)
        return self.block(self.fifo)


class TemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 depthwise=False,
                 full_conv3d=False, 
                 norms='mixed'):
        super().__init__()
        assert out_channels % 4 == 0  # needed for group norm to work
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.norms = norms
        kernel = (kernel_size,3,3) if full_conv3d else (kernel_size,1,1)
        
        self.streaming_mode = False
        self.fifo = None  # for streaming inference
        
        if self.norms=='mixed':
            norm1_block = bn_block
            norm2_block = gn_block
        elif self.norms=='all_bn':
            norm1_block = bn_block
            norm2_block = bn_block
        elif self.norms=='all_gn':
            norm1_block = gn_block
            norm2_block = gn_block

        if depthwise:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel, groups=in_channels, bias=False), 
                norm1_block(in_channels), 
                pw_conv(in_channels, out_channels), 
                norm2_block(out_channels), 
            )
            
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, bias=False), 
                norm2_block(out_channels), 
            )

    def streaming(self, enabled=True):
        if enabled:
            assert not self.training, "Can only use streaming mode during evaluation."
        self.streaming_mode = enabled
        
    def reset_memory(self):
        self.fifo = None
    
    def forward(self, input):
        if self.streaming_mode:
            return self._streaming_forward(input)
                  
        input = F.pad(input, (0, 0, 0, 0, self.kernel_size - 1, 0))
        return self.block(input)
    
    def _streaming_forward(self, input):
        if self.fifo is None:
            self.fifo = torch.zeros(*input.shape[:2], self.kernel_size, *input.shape[3:]).type_as(input)
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)
        return self.block(self.fifo)
        

class TennSt(nn.Module):
    def __init__(
        self, 
        channels, 
        t_kernel_size, 
        n_depthwise_layers, 
        detector_head, 
        detector_depthwise, 
        full_conv3d=False,
        norms='mixed',
    ):
        super().__init__()
        self.detector = detector_head
        
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        temporals = [True, False] * 5
        
        self.backbone = nn.Sequential()
        for i in range(len(depthwises)):
            in_channels, out_channels = channels[i], channels[i+1]
            depthwise = depthwises[i]
            temporal = temporals[i]
            
            if temporal:
                self.backbone.append(TemporalBlock(in_channels, out_channels, 
                                                   kernel_size=t_kernel_size, depthwise=depthwise,
                                                   full_conv3d=full_conv3d, norms=norms))
            else:
                self.backbone.append(SpatialBlock(in_channels, out_channels, depthwise=depthwise,
                                                  full_conv3d=full_conv3d,
                                                  kernel_size=t_kernel_size if full_conv3d else 1,
                                                  norms=norms))
        
        if detector_head:
            self.head = nn.Sequential(
                TemporalBlock(channels[-1], channels[-1], t_kernel_size, depthwise=detector_depthwise), 
                nn.Conv3d(channels[-1], channels[-1], (1, 3, 3), (1, 1, 1), (0, 1, 1)), 
                act_layer(), 
                nn.Conv3d(channels[-1], 3, 1), 
            )
        else:
            self.head = nn.Sequential(
                nn.Conv1d(channels[-1], channels[-1], 1), 
                act_layer(), 
                nn.Conv1d(channels[-1], 2, 1), 
            )
    
    def streaming(self, enabled=True):
        if enabled:
            warnings.warn("You have enabled the streaming mode of the network. It is expected, but not checked, that the input will be of shape (batch, 1, H, W).")
        for name, module in self.named_modules():
            if name and hasattr(module, 'streaming'):
                module.streaming(enabled)
                
    def reset_memory(self):
        for name, module in self.named_modules():
            if name and hasattr(module, 'reset_memory'):
                module.reset_memory()
         
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.detector:
            return self.head((self.backbone(input)))
        else:
            return self.head(self.backbone(input).mean((-2, -1)))
        


import argparse, json, os, mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset

# Import your dataset helper functions/classes.
# from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, LabelTemporalSubsample, SliceLongEventsToShort, EventSlicesToVoxelGrid, SliceByTimeEventsTargets

# Import the BigBrains model code (TennSt) from your module or paste the code above in a cell.
# from bigbrains_model import TennSt
# For the purpose of this notebook, assume TennSt is already defined in a previous cell.

# -------------------------------
# Custom Loss Function: Grid Loss
# -------------------------------

def smooth_l1_loss(pred, target, beta=0.11):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    return loss

def focal_loss(pred, target, gamma=2):
    # pred: predicted probability, target: binary (0 or 1)
    eps = 1e-6
    loss = torch.where(target==1, -((1 - pred) ** gamma) * torch.log(pred + eps),
                              -(pred ** gamma) * torch.log(1 - pred + eps))
    return loss

def grid_loss(pred, target, gamma=2, beta=0.11):
    """
    pred and target shapes: (B, T, grid_y, grid_x, 3)
    The last dimension: index 0 = predicted probability,
                      indices 1,2 = predicted offsets.
    """
    p_pred = pred[..., 0]
    offsets_pred = pred[..., 1:]
    
    p_target = target[..., 0]
    offsets_target = target[..., 1:]
    
    focal = focal_loss(p_pred, p_target, gamma)
    reg = smooth_l1_loss(offsets_pred, offsets_target, beta)
    # Only apply regression loss for grid-cells with pupil presence (p_target == 1)
    reg = reg * p_target
    total_loss = focal + reg
    return total_loss.mean()

# -------------------------------
# Hyperparameter Configuration
# -------------------------------

default_config = {
    "n_time_bins": 2,                   # Number of channels in event representation.
    "kernel_size_t": 3,                 # Temporal conv kernel size.
    "kernel_size_spatial": 3,           # Spatial conv kernel size.
    "batch_size": 32,
    "num_frames": 50,                   # Number of event frames per sample.
    "learning_rate": 0.002,
    "weight_decay": 0.005,
    "num_epochs": 200,
    "spatial_factor": 1,                # Factor for spatial downsampling.
    "temporal_subsample_factor": 1,     # Factor for temporal subsampling.
    "train_length": 50,                 # Number of frames in training sequence.
    "train_stride": 5,                  # Stride for slicing training data.
    "voxel_grid_ch_normaization": True,
    "data_dir": "./data",               # Path to dataset.
    "mlflow_path": "./mlruns",          # MLflow tracking directory.
    "experiment_name": "BigBrains_EET",
    "run_name": "BigBrains_Training",
}

# Optionally override defaults via argparse (if running from command line)
parser = argparse.ArgumentParser()
for key, value in default_config.items():
    parser.add_argument(f"--{key}", type=type(value), default=value)
args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# Training Pipeline
# -------------------------------

def main():
    # Save hyperparameters to MLflow.
    mlflow.set_tracking_uri(args.mlflow_path)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), "w") as f:
            json.dump(vars(args), f)
        
        # Instantiate the model (TennSt) with detector_head enabled.
        model = TennSt(
            channels=[args.n_time_bins, 16, 32, 64, 64],  # Example channel progression.
            t_kernel_size=args.kernel_size_t,
            n_depthwise_layers=2,             # Example value; adjust as needed.
            detector_head=True,
            detector_depthwise=False,
            full_conv3d=False,
            norms='mixed',
        ).to(args.device)
        
        # Define optimizer and (optionally) scheduler.
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Set up dataset and dataloader.
        factor = args.spatial_factor
        temp_subsample_factor = args.temporal_subsample_factor
        
        label_transform = transforms.Compose([
            ScaleLabel(factor),
            LabelTemporalSubsample(temp_subsample_factor),
            NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
        ])
        
        train_data_orig = ThreeETplus_Eyetracking(
            save_to=args.data_dir, split="train",
            transform=transforms.Downsample(spatial_factor=factor),
            target_transform=label_transform
        )
        
        slicing_time_window = args.train_length * int(10000 / temp_subsample_factor)  # in microseconds
        train_slicer = SliceByTimeEventsTargets(
            slicing_time_window,
            overlap=slicing_time_window - int(10000 / temp_subsample_factor * args.train_stride),
            seq_length=args.train_length,
            seq_stride=args.train_stride,
            include_incomplete=False
        )
        
        post_slicer_transform = transforms.Compose([
            SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0, include_incomplete=True),
            EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2),
                                    n_time_bins=args.n_time_bins,
                                    per_channel_normalize=args.voxel_grid_ch_normaization)
        ])
        
        train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform)
        train_data = DiskCachedDataset(train_data, cache_path=f'./cached_dataset/train_l{args.train_length}_s{args.train_stride}_ch{args.n_time_bins}')
        
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        # Training Loop
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(args.device)      # Expect shape: (B, T, C, H, W)
                target = target.to(args.device)  # Expect shape: (B, T, grid_y, grid_x, 3)
                optimizer.zero_grad()
                output = model(data)             # Output shape: (B, T, 3, 3, 4)
                loss = grid_loss(output, target, gamma=2, beta=0.11)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
        # Save final model.
        torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), "model_final.pth"))
        
if __name__ == "__main__":
    main()





