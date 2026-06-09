from pathlib import Path
import polars as pd
import numpy as np
import math
import warnings; warnings.simplefilter("ignore", RuntimeWarning)
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./data")
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data_fpath = DATA_DIR / "training_dataset.parquet"
submission_data_fpath = DATA_DIR / "submission_dataset.parquet"


hidden_dim_gnn = 16    # Hidden dimension for GNN
pre_gnn_dim = 32        # Pre-GNN MLP dimension
hidden_dim_mlp = 128     # Hidden dimension for post GNN MLP
hidden_dim_rnn = 32     # Hidden dimension for RNN
n_transformer_layers=2  # Number of transformer layers
output_dim = 1          # Output dimension (1 for regression)
heads = 1               # Number of attention heads in GAT
learning_rate = 1e-2    # Learning rate for the optimizer
epochs = 300            # Number of training epochs
patience = 5            # Early stopping patience
batch_size = 1024       # Batch size for DataLoader


input_df = pd.read_parquet(train_data_fpath)
inference_df = pd.read_parquet(submission_data_fpath)

numerical_cols = [
    col for col in input_df.columns if col not in ['TimeStamp_StartFormat', 'id', 'is_valid']
]

input_df = input_df.with_columns(
    [pd.col(col).cast(pd.Float32) for col in numerical_cols]
)

# keep only the valid rows - not applicable for temporal models since it breaks continuity
# we take care of bad values in a different way below
# Uncomment the next line if you want to filter out invalid rows
# input_df = input_df.filter(pd.col("is_valid") == True)

numerical_cols = [
    col for col in inference_df.columns if col not in ['TimeStamp_StartFormat', 'id', 'is_valid']
]

inference_df = inference_df.with_columns(
    [pd.col(col).cast(pd.Float32) for col in numerical_cols]
)



input_df.shape, inference_df.shape


selected_columns = [
	'TimeStamp_StartFormat',
	'wtc_AcWindSp_mean;1',	'wtc_AcWindSp_mean;2',	'wtc_AcWindSp_mean;3',	'wtc_AcWindSp_mean;4',	'wtc_AcWindSp_mean;5',	'wtc_AcWindSp_mean;7',
	'wtc_AcWindSp_stddev;1','wtc_AcWindSp_stddev;2','wtc_AcWindSp_stddev;3','wtc_AcWindSp_stddev;4','wtc_AcWindSp_stddev;5','wtc_AcWindSp_stddev;7',
    'wtc_ScYawPos_mean;1',	'wtc_ScYawPos_mean;2',	'wtc_ScYawPos_mean;3',	'wtc_ScYawPos_mean;4',	'wtc_ScYawPos_mean;5',	'wtc_ScYawPos_mean;7',
	'wtc_NacelPos_mean;1',	'wtc_NacelPos_mean;2',	'wtc_NacelPos_mean;3',	'wtc_NacelPos_mean;4',	'wtc_NacelPos_mean;5',	'wtc_NacelPos_mean;7',
	'wtc_PitcPosA_mean;1',	'wtc_PitcPosA_mean;2',	'wtc_PitcPosA_mean;3',	'wtc_PitcPosA_mean;4',	'wtc_PitcPosA_mean;5',	'wtc_PitcPosA_mean;7',
	'wtc_PitcPosB_mean;1',	'wtc_PitcPosB_mean;2',	'wtc_PitcPosB_mean;3',	'wtc_PitcPosB_mean;4',	'wtc_PitcPosB_mean;5',	'wtc_PitcPosB_mean;7',
	'wtc_PitcPosC_mean;1',	'wtc_PitcPosC_mean;2',	'wtc_PitcPosC_mean;3',	'wtc_PitcPosC_mean;4',	'wtc_PitcPosC_mean;5',	'wtc_PitcPosC_mean;7',
	'wtc_ScReToOp_timeon;1',	'wtc_ScReToOp_timeon;2',	'wtc_ScReToOp_timeon;3',	'wtc_ScReToOp_timeon;4',	'wtc_ScReToOp_timeon;5',	'wtc_ScReToOp_timeon;7',
	'wtc_ActPower_mean;1',	'wtc_ActPower_mean;2',	'wtc_ActPower_mean;3',	'wtc_ActPower_mean;4',	'wtc_ActPower_mean;5',	'wtc_ActPower_mean;7',
	'wtc_ActPower_max;1',	'wtc_ActPower_max;2',	'wtc_ActPower_max;3',	'wtc_ActPower_max;4',	'wtc_ActPower_max;5',	'wtc_ActPower_max;7',
	'wtc_ActPower_min;1',	'wtc_ActPower_min;2',	'wtc_ActPower_min;3',	'wtc_ActPower_min;4',	'wtc_ActPower_min;5',	'wtc_ActPower_min;7',
	'wtc_AmbieTmp_mean;1',	'wtc_AmbieTmp_mean;2',	'wtc_AmbieTmp_mean;3',	'wtc_AmbieTmp_mean;4',	'wtc_AmbieTmp_mean;5',	'wtc_AmbieTmp_mean;7',
	'ShutdownDuration;1',	'ShutdownDuration;2',	'ShutdownDuration;3',	'ShutdownDuration;4',	'ShutdownDuration;5',	'ShutdownDuration;7',
    'ERA5_wind_speed_100m',
    'ERA5_wind_direction_100m',
	'id',
	'is_valid',
	'target'
]
data_df = input_df.select(selected_columns)
inference_df = inference_df.select([col for col in data_df.columns if  not col.endswith(';1')][:-1])  # Exclude 'target' column for inference


# Define the turbines to iterate over for input data
turbines = [1, 2, 3, 4, 5, 7]

expressions = []
for turb in turbines:
    expressions.append((pd.col("is_valid").alias(f'is_valid_target;{turb}')))
    expressions.append((pd.col("ERA5_wind_speed_100m").alias(f'ERA5_wind_speed_100m;{turb}')))
    expressions.append((pd.col("ERA5_wind_direction_100m").alias(f'ERA5_wind_direction_100m;{turb}')))

    expressions.append(
    pd.when((pd.col(f'ShutdownDuration;{turb}') == 600) | pd.col(f'wtc_ActPower_mean;{turb}').is_null())
    .then(0)
    .otherwise(1)
    .alias(f'is_valid_source;{turb}')
    )
    expressions.append(
        pd.col(f'wtc_AcWindSp_mean;{turb}').shift(1).fill_null(strategy="zero").alias(f'lagged_wtc_AcWindSp_mean;{turb}')
    )
    expressions.append(
        pd.col(f'wtc_ActPower_mean;{turb}').shift(1).fill_null(strategy="zero").alias(f'lagged_wtc_ActPower_mean;{turb}')
    )
    
# Use `with_columns` to add the new 'is_valid;x' columns
data_df = data_df.with_columns(expressions)

# Impute missing values in the 'target' column
data_df = data_df.with_columns(
    pd.col("target").fill_null(0).alias("target")
)

# Ffill imputation strategy
data_df = data_df.fill_null(strategy="forward").fill_null(strategy="backward")

print("Processing data_df, inference_df...")
expressions = []
for turb in turbines:
    yaw_pos_col = f'wtc_ScYawPos_mean;{turb}'
    nacel_pos_col = f'wtc_NacelPos_mean;{turb}'
    turb_col = f'wtc_Turbulence_mean;{turb}'
    ws_mean_col = f'wtc_AcWindSp_mean;{turb}'
    ws_stddev_col = f'wtc_AcWindSp_stddev;{turb}'

    # Modulo 360 and circular transformation
    expressions.append(pd.col(yaw_pos_col) % 360.0)
    expressions.append((pd.col(yaw_pos_col) % 360.0 * np.pi / 180).sin().alias(f'sin_{yaw_pos_col}'))
    expressions.append((pd.col(yaw_pos_col) % 360.0 * np.pi / 180).cos().alias(f'cos_{yaw_pos_col}'))
    
    expressions.append((pd.col(nacel_pos_col) * np.pi / 180).sin().alias(f'sin_{nacel_pos_col}'))
    expressions.append((pd.col(nacel_pos_col) * np.pi / 180).cos().alias(f'cos_{nacel_pos_col}'))

    #Turbulence feature calculation
    expressions.append(
        pd.when(pd.col(ws_mean_col) == 0)
        .then(0)
        .otherwise(pd.col(ws_stddev_col) / pd.col(ws_mean_col))
        .alias(turb_col)
    )

# Apply all expressions at once
data_df = data_df.with_columns(expressions)

# Find all wtc 1 columns and remove the features so no leakage occurs
# We include the columns for wtc 1 for vectorization purposes, otherwise only the 
# ERA5 data is used for it
columns_to_zero = [
    'wtc_AcWindSp_mean;1',
    'wtc_AcWindSp_stddev;1',
    'wtc_ScYawPos_mean;1',
    'wtc_NacelPos_mean;1',
    'wtc_PitcPosA_mean;1',
    'wtc_PitcPosB_mean;1',
    'wtc_PitcPosC_mean;1',
    'wtc_ScReToOp_timeon;1',
    'wtc_ActPower_mean;1',
    'wtc_ActPower_max;1',
    'wtc_ActPower_min;1',
    'wtc_AmbieTmp_mean;1',
    'ShutdownDuration;1',
    'lagged_wtc_AcWindSp_mean;1',
    'lagged_wtc_ActPower_mean;1',
    'sin_wtc_ScYawPos_mean;1',
    'cos_wtc_ScYawPos_mean;1',
    'sin_wtc_NacelPos_mean;1',
    'cos_wtc_NacelPos_mean;1',
    'wtc_Turbulence_mean;1'

]
zero_expressions = [pd.lit(0).alias(col) for col in columns_to_zero]

inference_df = inference_df.with_columns(zero_expressions)
data_df = data_df.with_columns(zero_expressions)


# Define the turbines to iterate over fro inference data
turbines = [1, 2, 3, 4, 5, 7]

expressions = []
for turb in turbines:
    expressions.append((pd.col("is_valid").alias(f'is_valid_target;{turb}')))
    expressions.append((pd.col("ERA5_wind_speed_100m").alias(f'ERA5_wind_speed_100m;{turb}')))
    expressions.append((pd.col("ERA5_wind_direction_100m").alias(f'ERA5_wind_direction_100m;{turb}')))

    if turb != 1:
        expressions.append(
        pd.when((pd.col(f'ShutdownDuration;{turb}') == 600) | pd.col(f'wtc_ActPower_mean;{turb}').is_null())
        .then(0)
        .otherwise(1)
        .alias(f'is_valid_source;{turb}')
        )
        expressions.append(
            pd.col(f'wtc_AcWindSp_mean;{turb}').shift(1).fill_null(strategy="zero").alias(f'lagged_wtc_AcWindSp_mean;{turb}')
        )
        expressions.append(
            pd.col(f'wtc_ActPower_mean;{turb}').shift(1).fill_null(strategy="zero").alias(f'lagged_wtc_ActPower_mean;{turb}')
        )
    else:
        # For turbine 1, we set the is_valid_source to wtc 1 and lagged values to 0
        expressions.append(pd.col("is_valid").alias(f'is_valid_source;{turb}'))
        expressions.append(pd.lit(0).alias(f'lagged_wtc_AcWindSp_mean;{turb}'))
        expressions.append(pd.lit(0).alias(f'lagged_wtc_ActPower_mean;{turb}'))
    
# Use `with_columns` to add the new 'is_valid;x' columns
inference_df = inference_df.with_columns(expressions)

print("Processing data_df, inference_df...")
expressions = []
for turb in turbines:
    yaw_pos_col = f'wtc_ScYawPos_mean;{turb}'
    nacel_pos_col = f'wtc_NacelPos_mean;{turb}'
    turb_col = f'wtc_Turbulence_mean;{turb}'
    ws_mean_col = f'wtc_AcWindSp_mean;{turb}'
    ws_stddev_col = f'wtc_AcWindSp_stddev;{turb}'

    if turb != 1:
        # Modulo 360 and circular transformation
        expressions.append(pd.col(yaw_pos_col) % 360.0)
        expressions.append((pd.col(yaw_pos_col) % 360.0 * np.pi / 180).sin().alias(f'sin_{yaw_pos_col}'))
        expressions.append((pd.col(yaw_pos_col) % 360.0 * np.pi / 180).cos().alias(f'cos_{yaw_pos_col}'))

        expressions.append((pd.col(nacel_pos_col) * np.pi / 180).sin().alias(f'sin_{nacel_pos_col}'))
        expressions.append((pd.col(nacel_pos_col) * np.pi / 180).cos().alias(f'cos_{nacel_pos_col}'))

        #Turbulence feature calculation
        expressions.append(
            pd.when(pd.col(ws_mean_col) == 0)
            .then(0)
            .otherwise(pd.col(ws_stddev_col) / pd.col(ws_mean_col))
            .alias(turb_col)
        )
    else:
        # For turbine 1, we set the yaw and nacel positions to 0
        expressions.append(pd.lit(1).alias(f'sin_{yaw_pos_col}'))
        expressions.append(pd.lit(1).alias(f'cos_{yaw_pos_col}'))
        expressions.append(pd.lit(1).alias(f'sin_{nacel_pos_col}'))
        expressions.append(pd.lit(1).alias(f'cos_{nacel_pos_col}'))

        # Set turbulence to 0 for turbine 1
        expressions.append(pd.lit(1).alias(turb_col))

# Apply all expressions at once
inference_df = inference_df.with_columns(expressions)


turbines = [1, 2, 3, 4, 5, 7]
num_turbines_input = len(turbines)  # 6 turbines
num_time_steps = data_df.shape[0]

# The columns of interest from the original set
feature_columns = [
    'wtc_ActPower_mean','wtc_ActPower_max', 'wtc_ActPower_min',
    'wtc_AcWindSp_mean', 'wtc_Turbulence_mean',
    'lagged_wtc_ActPower_mean', 'lagged_wtc_AcWindSp_mean',
    #'ShutdownDuration',
    #'sin_wtc_ScYawPos_mean', 'cos_wtc_ScYawPos_mean',
    'sin_wtc_NacelPos_mean', 'cos_wtc_NacelPos_mean',
    'wtc_PitcPosA_mean', 'wtc_PitcPosB_mean', 'wtc_PitcPosC_mean',
    #'wtc_ScReToOp_timeon',
    'wtc_AmbieTmp_mean',
    'ERA5_wind_speed_100m', 'ERA5_wind_direction_100m',
    'is_valid_target', 'is_valid_source'
]
num_features = len(feature_columns)

# --- Define the Fully Connected Graph ---
edge_list = []
for i in range(num_turbines_input):
    for j in range(num_turbines_input):
        if i != j:
            edge_list.append([i, j])
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

print(f"Total time steps: {num_time_steps}")

# --- Model Classes ---
class GNN_MLP_RNN_Model(nn.Module):
    def __init__(self, num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim):
        super(GNN_MLP_RNN_Model, self).__init__()
        
        # GNN layer to capture spatial correlations
        self.gnn = GCNConv(num_features, hidden_dim_gnn)
        
        # Multi-Layer Perceptron (MLP) for pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim_gnn, hidden_dim_mlp),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_mlp, hidden_dim_mlp)
        )
        
        # The RNN takes the MLP's output as input
        self.rnn = nn.GRU(input_size=hidden_dim_mlp, 
                          hidden_size=hidden_dim_rnn, 
                          batch_first=True)
        
        # Final linear layer for prediction
        self.fc = nn.Linear(hidden_dim_rnn, output_dim)
        
    def forward(self, x_time_series, edge_index):
        pooled_outputs = []
        for t in range(x_time_series.size(0)):
            x_t = x_time_series[t]
            gnn_out = self.gnn(x_t, edge_index)
            
            # Aggregate the GNN outputs (e.g., mean pooling)
            aggregated_gnn_out = torch.mean(gnn_out, dim=0)
            
            # Pass the aggregated GNN output through the MLP
            mlp_out = self.mlp(aggregated_gnn_out)
            pooled_outputs.append(mlp_out)
        
        # Stack the MLP outputs into a sequence for the RNN
        rnn_sequence = torch.stack(pooled_outputs).unsqueeze(0)
        
        # Pass the sequence through the RNN
        rnn_out, _ = self.rnn(rnn_sequence)
        
        # Reshape for the final linear layer
        rnn_out = rnn_out.squeeze(0) 
        
        # Pass the entire sequence of RNN outputs through the linear layer
        prediction = self.fc(rnn_out)
        
        return prediction
    
class GNN_MLP_LSTM_Model(nn.Module):
    def __init__(self, num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim):
        super(GNN_MLP_LSTM_Model, self).__init__()
        
        # GNN layer to capture spatial correlations
        self.gnn = GCNConv(num_features, hidden_dim_gnn)
        
        # Multi-Layer Perceptron (MLP) for pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim_gnn, hidden_dim_mlp),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_mlp, hidden_dim_mlp)
        )
        
        # LSTM
        self.rnn = nn.LSTM(input_size=hidden_dim_mlp, 
                          hidden_size=hidden_dim_rnn, 
                          batch_first=True)
        
        # Final linear layer for prediction
        self.fc = nn.Linear(hidden_dim_rnn, output_dim)
        
    def forward(self, x_time_series, edge_index):
        pooled_outputs = []
        for t in range(x_time_series.size(0)):
            x_t = x_time_series[t]
            gnn_out = self.gnn(x_t, edge_index)
            
            aggregated_gnn_out = torch.mean(gnn_out, dim=0)
            
            mlp_out = self.mlp(aggregated_gnn_out)
            pooled_outputs.append(mlp_out)
        
        rnn_sequence = torch.stack(pooled_outputs).unsqueeze(0)
        
        rnn_out, (h_n, c_n) = self.rnn(rnn_sequence)
        
        rnn_out = rnn_out.squeeze(0) 
        
        prediction = self.fc(rnn_out)
        
        return prediction

class GAT_LSTM_Model(nn.Module):
    def __init__(self, num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim):
        super(GAT_LSTM_Model, self).__init__()
        
        # Graph Attention Network (GAT) layer
        # The output dimension is hidden_dim_gat * heads
        self.gat = GATConv(num_features, hidden_dim_gnn, heads=heads, concat=True)
        
        # The LSTM takes the GAT's aggregated output as input
        # The input size is hidden_dim_gat * heads
        self.rnn = nn.LSTM(input_size=hidden_dim_gnn * heads, 
                           hidden_size=hidden_dim_rnn, 
                           batch_first=True)
        
        # Final linear layer for prediction
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim_rnn, hidden_dim_mlp),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_mlp, output_dim)
        )
        
    def forward(self, x_time_series, edge_index):
        # x_time_series shape: (batch_size, num_turbines, num_features)
        
        batch_size = x_time_series.size(0)
        num_turbines = num_turbines_input
        
        # Reshape the input to a 2D tensor for the GAT layer
        x_flat = x_time_series.reshape(batch_size * num_turbines, -1)
        
        # Create a "super-graph" edge_index for the entire batch
        batch_tensor = torch.arange(batch_size, device=x_time_series.device)
        batch_tensor = batch_tensor.repeat_interleave(num_turbines)
        edge_index_batched = edge_index + batch_tensor[edge_index[0]] * num_turbines
        
        # Pass the flattened data through the GAT layer
        # The output shape is (batch_size * num_turbines, hidden_dim_gat * heads)
        gat_out = self.gat(x_flat, edge_index_batched)
        
        # Reshape the GAT output to the original batch structure
        # New shape: (batch_size, num_turbines, hidden_dim_gat * heads)
        gat_out = gat_out.reshape(batch_size, num_turbines, -1)
        
        # Aggregate the outputs per graph (across the turbine dimension)
        # Resulting shape: (batch_size, hidden_dim_gat * heads)
        aggregated_output = torch.mean(gat_out, dim=1)
        
        # Pass the aggregated sequence to the LSTM
        # The batch_first=True LSTM expects (batch_size, sequence_length, features)
        # Here, the batch_size is the number of time steps in the batch
        lstm_out, _ = self.rnn(aggregated_output.unsqueeze(1))

        # Reshape for the final linear layer
        prediction = self.regression_head(lstm_out.squeeze(1))
        
        return prediction

class GAT_MLP_Model(nn.Module):
    def __init__(self, num_features, hidden_dim_gnn, hidden_dim_mlp, output_dim):
        super(GAT_MLP_Model, self).__init__()
        
        self.num_turbines = num_turbines_input
        self.hidden_dim_gat = hidden_dim_gnn
        self.heads = heads
        
        # Graph Attention Network (GAT) layer
        self.gat = GATConv(num_features, hidden_dim_gnn, heads=heads, concat=True)
        
        # MLP to act as the regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim_gnn * heads, hidden_dim_mlp),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp, output_dim)
        )
        
    def forward(self, x_time_series, edge_index):
        # x_time_series shape: (batch_size, num_turbines, num_features)
        
        batch_size = x_time_series.size(0)
        num_turbines = num_turbines_input
        
        # Reshape the input to a 2D tensor for the GAT layer
        x_flat = x_time_series.reshape(batch_size * num_turbines, -1)
        
        # Create a "super-graph" edge_index for the entire batch
        batch_tensor = torch.arange(batch_size, device=x_time_series.device)
        batch_tensor = batch_tensor.repeat_interleave(num_turbines)
        edge_index_batched = edge_index + batch_tensor[edge_index[0]] * num_turbines
        
        # Pass the flattened data through the GAT layer
        gat_out = self.gat(x_flat, edge_index_batched)
        
        # Reshape the GAT output to the original batch structure
        gat_out = gat_out.reshape(batch_size, num_turbines, -1)
        
        # Aggregate the outputs per graph (across the turbine dimension)
        # Resulting shape: (batch_size, hidden_dim_gat * heads)
        aggregated_output = torch.mean(gat_out, dim=1)
        
        # Pass the aggregated output directly through the MLP head
        prediction = self.regression_head(aggregated_output)
        
        return prediction
    
class MLP_GAT_LSTM_MLP_Model(nn.Module):
    def __init__(self, num_features, pre_gnn_dim, hidden_dim_gnn, hidden_dim_mlp, output_dim):
        super(MLP_GAT_LSTM_MLP_Model, self).__init__()
        
        self.num_turbines = num_turbines_input
        self.heads = heads
        
        # MLP to pre-process features for each turbine
        self.pre_gnn_mlp = nn.Sequential(
            nn.Linear(num_features, pre_gnn_dim),
            nn.LeakyReLU(),
            nn.Linear(pre_gnn_dim, pre_gnn_dim)
        )
        
        # The GAT layer's input dimension is the output of the pre-GNN MLP
        self.gat = GATConv(pre_gnn_dim, hidden_dim_gnn, heads=heads, concat=True)

        self.rnn = nn.LSTM(input_size=hidden_dim_gnn * heads, 
                           hidden_size=hidden_dim_rnn, 
                           batch_first=True)
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim_rnn, hidden_dim_mlp),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_mlp, output_dim)
        )
        
    def forward(self, x_time_series, edge_index):
        # x_time_series shape: (batch_size, num_turbines, num_features)
        
        batch_size = x_time_series.size(0)
        num_turbines = num_turbines_input
        
        # Flatten the batch and turbine dimensions to apply the MLP to each turbine's features
        x_flat = x_time_series.reshape(batch_size * num_turbines, -1)
        pre_gnn_out = self.pre_gnn_mlp(x_flat)
        
        # Create a "super-graph" edge_index for the entire batch
        batch_tensor = torch.arange(batch_size, device=x_time_series.device)
        batch_tensor = batch_tensor.repeat_interleave(num_turbines)
        edge_index_batched = edge_index + batch_tensor[edge_index[0]] * num_turbines
        
        gat_out = self.gat(pre_gnn_out, edge_index_batched)
        
        # Reshape the GAT output to the original batch structure
        gat_out = gat_out.reshape(batch_size, num_turbines, -1)
        
        # Aggregate the outputs per graph (across the turbine dimension)
        aggregated_output = torch.mean(gat_out, dim=1)
        
        lstm_out, _ = self.rnn(aggregated_output.unsqueeze(1))

        # Pass the aggregated output directly through the MLP head
        prediction = self.regression_head(lstm_out.squeeze(1))
        
        return prediction
    
class MLP_Model(nn.Module):
    def __init__(self, num_features, hidden_dim_mlp, output_dim):
        super(MLP_Model, self).__init__()
                
        # MLP 
        self.mlp = nn.Sequential(
            nn.Linear(num_features*num_turbines_input, hidden_dim_mlp),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_mlp, 8),
            nn.LeakyReLU(),
            nn.Linear(8, output_dim)
        )
        
        
    def forward(self, x_time_series, edge_index):
        
        batch_size = x_time_series.size(0)
        
        # Pass the aggregated output directly through the MLP head
        prediction = self.mlp(x_time_series.reshape(batch_size, -1))
        
        return prediction
    
class WeightedL1Loss(nn.Module):
    def __init__(self, overprediction_weight=10.0, middle_band_weight=5.0, middle_band_low=500, middle_band_high=1500):
        super(WeightedL1Loss, self).__init__()
        self.overprediction_weight = overprediction_weight
        self.middle_band_weight = middle_band_weight
        self.middle_band_low = middle_band_low
        self.middle_band_high = middle_band_high
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, prediction, target):
        loss = self.l1_loss(prediction, target)
        
        # Create a weight mask initialized with ones
        weight = torch.ones_like(loss)
        
        # Condition 1: High weight for over-predictions when actual power is near zero
        # Assumes normalized target of 0 represents 0 kW
        #overprediction_mask = (target < 0.1) & (prediction > 0.1)
        #weight[overprediction_mask] = self.overprediction_weight
        
        # Condition 2: High weight for errors in the middle of the power band
        # This focuses the model on the most dynamic and often difficult-to-predict region
        middle_band_mask = (target >= self.middle_band_low) & (target <= self.middle_band_high)
        weight[middle_band_mask] = self.middle_band_weight
        
        # Apply the combined weights to the loss
        weighted_loss = loss * weight
        
        # Return the mean of the weighted loss
        return torch.mean(weighted_loss)

class GCGRUCell(nn.Module):
    """A GRU cell with Graph Convolutions."""
    def __init__(self, input_dim, hidden_dim):
        super(GCGRUCell, self).__init__()
        self.hidden_dim = hidden_dim

        # Graph convolution for the update gate
        self.conv_update = GCNConv(input_dim + hidden_dim, hidden_dim)
        
        # Graph convolution for the reset gate
        self.conv_reset = GCNConv(input_dim + hidden_dim, hidden_dim)
        
        # Graph convolution for the new hidden state candidate
        self.conv_candidate = GCNConv(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, edge_index):
        # x_t shape: (num_nodes, input_dim)
        # h_prev shape: (num_nodes, hidden_dim)
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # Update gate
        update_gate = torch.sigmoid(self.conv_update(combined, edge_index))
        
        # Reset gate
        reset_gate = torch.sigmoid(self.conv_reset(combined, edge_index))
        
        # Candidate hidden state
        combined_reset = torch.cat([x_t, reset_gate * h_prev], dim=1)
        candidate_h = torch.tanh(self.conv_candidate(combined_reset, edge_index))
        
        # New hidden state
        h_t = (1.0 - update_gate) * h_prev + update_gate * candidate_h
        
        return h_t

class GCGRU_Model(nn.Module):
    def __init__(self, num_features, hidden_dim_gnn, hidden_dim_mlp, output_dim):
        super(GCGRU_Model, self).__init__()
        self.num_turbines = 6
        self.hidden_dim = hidden_dim_gnn
        
        # GC-GRU cell to process the graph sequence
        self.gc_gru_cell = GCGRUCell(num_features, hidden_dim_gnn)
        
        # Regression head to map final hidden states to output
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim_gnn, hidden_dim_mlp),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_mlp, output_dim)
        )

    def forward(self, x_time_series, edge_index):
        # x_time_series shape: (batch_size, num_turbines, num_features)
        batch_size = x_time_series.size(0)
        
        # Initialize the hidden state for all nodes to zeros
        h = torch.zeros(batch_size, self.num_turbines, self.hidden_dim, device=x_time_series.device)
        
        # Loop through time steps to update node hidden states
        for t in range(batch_size):
            # Update hidden state for all nodes at time t
            h_t = self.gc_gru_cell(x_time_series[t], h[t-1] if t > 0 else h[0], edge_index)
            h[t] = h_t
            
        # Use the final hidden states of all nodes for prediction
        # Aggregate the final hidden states across nodes
        final_h_aggregated = torch.mean(h, dim=1) # Shape: (batch_size, hidden_dim)
        
        # Pass through the regression head
        prediction = self.regression_head(final_h_aggregated)
        
        return prediction
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class GAT_Transformer_Model(nn.Module):
    def __init__(self, num_features, hidden_dim_gnn, hidden_dim_mlp, output_dim):
        super(GAT_Transformer_Model, self).__init__()
        self.num_turbines = 6
        
        # GAT layer for spatial processing
        self.gat = GATConv(num_features, hidden_dim_gnn, heads=heads, concat=True)
        gat_output_dim = hidden_dim_gnn * heads
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=gat_output_dim)
        
        # Transformer Encoder for temporal processing
        encoder_layer = TransformerEncoderLayer(d_model=gat_output_dim, nhead=heads, dim_feedforward=hidden_dim_mlp)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        # Final regression head
        self.regression_head = nn.Linear(gat_output_dim, output_dim)
        
    def forward(self, x_time_series, edge_index):
        # x_time_series shape: (batch_size, num_turbines, num_features)
        batch_size = x_time_series.size(0)

        # Reshape and create batched edge_index for GAT
        x_flat = x_time_series.reshape(batch_size * self.num_turbines, -1)
        edge_index_batched = edge_index.repeat(1, batch_size) + \
            torch.arange(batch_size, device=x_time_series.device).repeat_interleave(edge_index.size(1)) * self.num_turbines
        
        # Apply GAT layer
        gat_out = self.gat(x_flat, edge_index_batched)
        
        # Reshape and aggregate
        gat_out = gat_out.reshape(batch_size, self.num_turbines, -1)
        aggregated_output = torch.mean(gat_out, dim=1) # Shape: (batch_size, gat_output_dim)
        
        # Prepare for Transformer: (seq_len, N, features)
        # Here, the batch is the sequence, so N=1
        transformer_input = aggregated_output.unsqueeze(1) # Shape: (batch_size, 1, gat_output_dim)
        
        # Add positional encoding
        transformer_input = self.pos_encoder(transformer_input)
        
        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(transformer_input)
        
        # Pass through the final regression head
        prediction = self.regression_head(transformer_out.squeeze(1)) # Shape: (batch_size, output_dim)
        
        return prediction
    
class SAGE_LSTM_Model(nn.Module):
    def __init__(self, num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim):
        super(SAGE_LSTM_Model, self).__init__()
        self.num_turbines = 6
        
        # Stacked GraphSAGE layers for richer spatial feature extraction
        self.sage1 = SAGEConv(num_features, hidden_dim_gnn)
        self.sage2 = SAGEConv(hidden_dim_gnn, hidden_dim_gnn)
        
        # LSTM for temporal sequence processing
        self.rnn = nn.LSTM(input_size=hidden_dim_gnn, 
                           hidden_size=hidden_dim_rnn, 
                           batch_first=True)
        
        # Final regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim_rnn, hidden_dim_mlp),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_mlp, output_dim)
        )
        
    def forward(self, x_time_series, edge_index):
        # x_time_series shape: (batch_size, num_turbines, num_features)
        batch_size = x_time_series.size(0)
        
        # Reshape for GNN processing
        x_flat = x_time_series.reshape(batch_size * self.num_turbines, -1)
        
        # Create a batched edge_index for the entire sequence
        edge_index_batched = edge_index.repeat(1, batch_size) + \
            torch.arange(batch_size, device=x_time_series.device).repeat_interleave(edge_index.size(1)) * self.num_turbines

        # Apply GraphSAGE layers with activation
        sage_out = self.sage1(x_flat, edge_index_batched)
        sage_out = F.leaky_relu(sage_out)
        sage_out = self.sage2(sage_out, edge_index_batched)
        
        # Reshape back to (batch_size, num_turbines, hidden_dim_gnn)
        sage_out = sage_out.reshape(batch_size, self.num_turbines, -1)
        
        # Aggregate node features (mean pooling) for each time step
        aggregated_output = torch.mean(sage_out, dim=1) # Shape: (batch_size, hidden_dim_gnn)
        
        # LSTM expects (batch_size, seq_len, features). Here, batch_size is the sequence length.
        # We process the whole sequence as a single batch for the LSTM.
        lstm_input = aggregated_output.unsqueeze(0) # Shape: (1, batch_size, hidden_dim_gnn)
        lstm_out, _ = self.rnn(lstm_input)
        
        # Pass the output of the LSTM through the regression head
        prediction = self.regression_head(lstm_out.squeeze(0))
        
        return prediction


# Extract features and stack into a NumPy array
X_list = [data_df[[f'{col};{turb}' for col in feature_columns]].to_numpy() for turb in turbines]
X_np = np.stack(X_list, axis=1) # Shape: (num_time_steps, num_turbines_input, num_features)
X_np_flat = X_np.reshape(-1, num_features) # Reshape for imputer

# Apply Standard scaling to standardize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_np_flat)

# Reshape the processed data and convert to PyTorch tensors
X_normalized = torch.from_numpy(X_normalized).float()
X_normalized = X_normalized.view(num_time_steps, num_turbines_input, num_features)

# Impute the 'target' column and reshape to a 2D array for the imputer
Y_np = data_df.select('target').fill_null(strategy="forward").fill_null(strategy="backward").to_numpy()
# Convert the imputed numpy array to a PyTorch tensor
Y = torch.from_numpy(Y_np).float()

# Define Split Ratio
train_ratio = 0.70

# Calculate Split Index
train_split_idx = int(num_time_steps * train_ratio)

# Split the Data
X_train, Y_train = X_normalized[:train_split_idx], Y[:train_split_idx]
X_test, Y_test = X_normalized[train_split_idx:], Y[train_split_idx:]

print(f"Training set size: {len(X_train)} time steps")
print(f"Test set size: {len(X_test)} time steps")



# Check if the MPS backend is available
#if torch.backends.mps.is_available():
#    device = torch.device("mps")
#    print("Using Apple Metal (MPS) for acceleration.")
#else:
#    device = torch.device("cpu")
#    print("MPS not available. Falling back to CPU.")
device = torch.device("cpu")

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# --- Instantiate Model, Loss, and Optimizer ---
#model = GNN_MLP_RNN_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim).to(device)
#model = GNN_MLP_LSTM_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim).to(device)
#model = GAT_LSTM_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim).to(device)
#model = GAT_MLP_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, output_dim).to(device)
#model = MLP_GAT_LSTM_MLP_Model(num_features, pre_gnn_dim, hidden_dim_gnn, hidden_dim_mlp, output_dim).to(device)
#model = MLP_Model(num_features, hidden_dim_mlp, output_dim).to(device)
#model = GCGRU_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, output_dim).to(device)
#model = GAT_Transformer_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, output_dim).to(device)
model = SAGE_LSTM_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim).to(device)

X_train, Y_train = X_train.to(device), Y_train.to(device)
edge_index = edge_index.to(device)
criterion = nn.L1Loss().to(device)
#criterion = nn.HuberLoss().to(device)
#criterion = WeightedL1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, threshold=1e-4)

# --- Training Loop ---
print("\nStarting training...")
previous_loss = float('inf')  # Initialize for early stopping check
best_loss = float('inf')      # Initialize for saving the best model

for epoch in range(epochs):
    model.train()
    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        
        optimizer.zero_grad()
        batch_prediction = model(batch_X, edge_index)
        loss = criterion(batch_prediction, batch_Y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}')

    # Check for a new best model and save it
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), 'best_model_weights.pth')
        print(f"New best model saved.")

    # Early stopping and scheduler logic
    if previous_loss != float('inf'):
        loss_improvement = ((previous_loss - loss.item()) / previous_loss) * 100
        print(f"Loss improvement over last epoch: {loss_improvement:.4f}%")
        if abs(loss_improvement) < 0.0001:
            print("Early stopping triggered: Loss improvement is almost zero.")
            break
        if loss_improvement < -10:
            print("Wrong way, stopping training!")
            break

    previous_loss = loss.item()
    scheduler.step(loss)

torch.save(model.state_dict(), 'final_model_weights.pth')



device = torch.device("cpu")

# Load the saved model weights
# Create a new instance of your model class
#loaded_model = GNN_MLP_RNN_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim).to(device)
#loaded_model = GNN_MLP_LSTM_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim).to(device)
#loaded_model = GAT_LSTM_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim).to(device)
#loaded_model = GAT_MLP_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, output_dim).to(device)
#loaded_model = MLP_GAT_LSTM_MLP_Model(num_features, pre_gnn_dim, hidden_dim_gnn, hidden_dim_mlp, output_dim).to(device)
#loaded_model = MLP_Model(num_features, hidden_dim_mlp, output_dim).to(device)
#loaded_model = GCGRU_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, output_dim).to(device)
loaded_model = SAGE_LSTM_Model(num_features, hidden_dim_gnn, hidden_dim_mlp, hidden_dim_rnn, output_dim).to(device)

# Load the saved state dictionary
loaded_model.load_state_dict(torch.load('best_model_weights.pth'))

# Set the model to evaluation mode
loaded_model.eval()


print("\nEvaluating model on the train set...")
model = loaded_model
criterion = nn.L1Loss()
model.eval()
with torch.no_grad():
    # Run inference on the test set
    train_prediction = model(X_train, edge_index)
    
    # Calculate the final train loss (MAE)
    train_loss = criterion(train_prediction, Y_train)
    
    print(f"\nFinal Train Loss (MAE): {train_loss.item():.4f}")


import matplotlib.pyplot as plt

# Create the scatter plot using matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(Y_train, train_prediction, s=1, alpha=0.2)

# Add the red 1-to-1 line
max_val = 2350
plt.plot([0, max_val], [0, max_val], "--r", label="1-to-1 Line")

# Set the axis labels and title
plt.xlabel("Actual Power Output (kW)")
plt.ylabel("Predicted Power Output (kW)")
plt.title("Actual vs. Predicted Power Output")

# Add a grid and legend
plt.grid(True)
plt.legend()

# Display the plot
plt.show()


print("\nEvaluating model on the test set...")
model = loaded_model
criterion = nn.L1Loss()
model.eval()
with torch.no_grad():
    # Run inference on the test set
    test_prediction = model(X_test, edge_index)
    
    # Calculate the final test loss (MAE)
    test_loss = criterion(test_prediction, Y_test)
    
    print(f"\nFinal Test Loss (MAE): {test_loss.item():.4f}")

    # Display a few sample predictions from the test set
    print("\nSample Test Set Predictions vs. Actual Values:")
    for i in range(5):
        sample_idx = np.random.randint(0, len(X_test))
        predicted_power = test_prediction[sample_idx].item()
        actual_power = Y_test[sample_idx].item()
        print(f"  Time Step {sample_idx+1}: Predicted = {predicted_power:.2f} kW, Actual = {actual_power:.2f} kW")


import matplotlib.pyplot as plt

# Create the scatter plot using matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, test_prediction, s=1, alpha=0.2)

# Add the red 1-to-1 line
max_val = 2350
plt.plot([0, max_val], [0, max_val], "--r", label="1-to-1 Line")

# Set the axis labels and title
plt.xlabel("Actual Power Output (kW)")
plt.ylabel("Predicted Power Output (kW)")
plt.title("Actual vs. Predicted Power Output")

# Add a grid and legend
plt.grid(True)
plt.legend()

# Display the plot
plt.show()


inference_num_time_steps = inference_df.shape[0] 
turbines = [1, 2, 3, 4, 5, 7]

# Ffill imputation strategy
inference_df = inference_df.fill_null(strategy="forward").fill_null(strategy="backward")

# Extract features and reshape for imputation
#inference_df = inference_df.select([col for col in data_df.columns if col in inference_df.columns])
inference_X_list = [inference_df[[f'{col};{turb}' for col in feature_columns]].to_numpy() for turb in turbines]
inference_X_np = np.stack(inference_X_list, axis=1)
inference_X_np_flat = inference_X_np.reshape(-1, num_features)

# Use the fitted scaler to transform (standardize)
inference_X_normalized = scaler.transform(inference_X_np_flat)

# Reshape the processed data and convert to PyTorch tensor
inference_X_normalized = torch.from_numpy(inference_X_normalized).float()
inference_X_normalized = inference_X_normalized.view(inference_df.shape[0], len(turbines), num_features)

print(f"Inference set size: {len(inference_X_normalized)} time steps")
print(f"Shape of inference features: {inference_X_normalized.shape}")

# --- Set the model to evaluation mode ---
model.eval()

# --- Perform Inference without gradient calculation ---
with torch.no_grad():
    inferred_power_predictions = model(inference_X_normalized, edge_index)

# --- Post-processing the predictions ---
predicted_powers = inferred_power_predictions.squeeze().numpy()

# Add the predictions as a new column to the DataFrame
predicted_series = pd.Series("predicted_power", predicted_powers)
inference_df = inference_df.with_columns(predicted_series)

# --- Display the results ---
print("\nInference Results:")
print(inference_df[['id', 'is_valid', 'predicted_power']].head())


output_fpath = (OUTPUT_DIR / "model_submission.csv").as_posix()

#(predicted_powers.fillna(0).to_frame(name="prediction").to_csv(output_fpath)) #change this to polars below
predicted_series = pd.Series("prediction", predicted_powers)
id_series = pd.Series("id", range(len(predicted_series)), dtype=pd.Int32)
prediction_df = pd.DataFrame({"id": id_series, "prediction": predicted_series})
prediction_df = prediction_df.fill_null(0)
prediction_df.write_csv(output_fpath)

print(f"Submission file saved to {output_fpath}")


!head -n 5 {output_fpath}


len(predicted_powers), len(predicted_series), len(id_series)


_df = pd.read_csv(output_fpath)

# checking the columns are the expected ones
assert _df.columns.to_list() == ["id", "prediction"], (
    f'Expected columns ["id", "prediction"], found: {_df.columns.to_list()}'
)

# checking no nulls in the data
assert _df.isna().sum().sum() == 0, "There are NA values in the data!"

# checking the row ids are unique and within expected range
duplicated_ids = _df["id"].duplicated()
assert not duplicated_ids.any(), f"There are duplicated ids: {_df['id'][duplicated_ids].values}"
invalid_ids = set(_df["id"].unique()) - set(range(52704))
assert not invalid_ids, f"The following row IDs are not within the expected ones: {invalid_ids}"

