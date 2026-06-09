!pip install torch-geometric


import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. Set seeds for reproducibility
# ============================================================================
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# ============================================================================
# 2. Load and Process Data
#    (Here we use the seeds and historical tournament results)
# ============================================================================
# Load seeds data (Men's and Women's)
w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)

# Filter to include only seasons from 2020 onward.
seed_df = seed_df[seed_df['Season'] >= 2020]

def extract_seed_value(seed_str):
    try:
        return int(seed_str[1:])  # remove letter prefix, e.g. "W05" -> 5
    except (ValueError, TypeError):
        return 16

seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

# Load historical game results
w_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
m_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
historical_df = pd.concat([m_results, w_results], axis=0)

# Filter historical data for seasons from 2020 onward.
historical_df = historical_df[historical_df['Season'] >= 2020]

# For each game, we already have a winner and a loser.
# (WTeamID won over LTeamID.) We'll use these to build training pairs later.
print("Data loaded successfully.")

# ============================================================================
# 3. Build the Graph from Historical Data
#    – Nodes: each unique team (using seed_df to get team IDs)
#    – Node features: here we use the team’s seed value (you can add more features)
#    – Edges: if two teams played a game, we add an edge (bidirectional)
# ============================================================================
# Use unique team IDs from seed_df
team_ids = seed_df['TeamID'].unique()
team_ids = np.sort(team_ids)
num_nodes = len(team_ids)
team_id_to_idx = {team_id: i for i, team_id in enumerate(team_ids)}

# Create node features – here, we use the average seed value per team.
node_features = []
for team_id in team_ids:
    # There may be multiple entries per team; take the mean seed
    seed_values = seed_df.loc[seed_df['TeamID'] == team_id, 'SeedValue'].values
    if len(seed_values) > 0:
        feature = np.mean(seed_values)
    else:
        feature = 16.0
    node_features.append([feature])
node_features = torch.tensor(node_features, dtype=torch.float)  # shape: [num_nodes, 1]

# Build edge_index from historical games.
edge_list = []
for idx, row in historical_df.iterrows():
    team_w = row['WTeamID']
    team_l = row['LTeamID']
    # Only add the edge if both teams are in our mapping.
    if (team_w in team_id_to_idx) and (team_l in team_id_to_idx):
        i = team_id_to_idx[team_w]
        j = team_id_to_idx[team_l]
        # Add bidirectional edge for an undirected graph.
        edge_list.append([i, j])
        edge_list.append([j, i])
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # shape: [2, num_edges]

# ============================================================================
# 4. Prepare Training Pairs from Historical Games
#    – For each game, create two samples:
#         (winner, loser) with label 1, and (loser, winner) with label 0.
# ============================================================================
pair_indices = []
labels = []
for idx, row in historical_df.iterrows():
    team_w = row['WTeamID']
    team_l = row['LTeamID']
    if (team_w in team_id_to_idx) and (team_l in team_id_to_idx):
        i = team_id_to_idx[team_w]
        j = team_id_to_idx[team_l]
        pair_indices.append([i, j])
        labels.append(1.0)
        pair_indices.append([j, i])
        labels.append(0.0)
pair_indices = torch.tensor(pair_indices, dtype=torch.long)  # shape: [num_samples, 2]
labels = torch.tensor(labels, dtype=torch.float)             # shape: [num_samples]

# Split into training and validation pairs (using stratification)
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=seed_value,
                                      stratify=labels.numpy())
train_pairs = pair_indices[train_idx]
train_labels = labels[train_idx]
val_pairs = pair_indices[val_idx]
val_labels = labels[val_idx]

# ============================================================================
# 5. Define the GNN-Based Mixture-of-Experts Model
#    This model uses two GCN layers to obtain team embeddings.
#    For predicting the win probability for a pair of teams, it computes the pair embedding
#    (by concatenating the two team embeddings) and then passes this through multiple expert MLPs.
#    A gating network then assigns weights to each expert’s prediction.
# ============================================================================
class GNNMoEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, mlp_hidden_dim, num_experts):
        super(GNNMoEModel, self).__init__()
        # GCN layers for computing node embeddings.
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        
        self.num_experts = num_experts
        # Define a list of expert MLPs.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim * 2, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, 1)
            ) for _ in range(num_experts)
        ])
        
        # Define a gating network that produces a weight for each expert.
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
    def forward(self, x, edge_index, pair_indices):
        # Compute node embeddings with two GCN layers.
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)  # final node embeddings, shape: [num_nodes, out_dim]
        
        # Extract the embeddings for each pair.
        h1 = x[pair_indices[:, 0]]    # shape: [num_samples, out_dim]
        h2 = x[pair_indices[:, 1]]    # shape: [num_samples, out_dim]
        
        # Concatenate embeddings to form the pair representation.
        pair_emb = torch.cat([h1, h2], dim=1)  # shape: [num_samples, out_dim*2]
        
        # Compute each expert's output.
        expert_logits = []
        for expert in self.experts:
            expert_logits.append(expert(pair_emb))
        # Stack expert outputs: shape becomes [num_experts, num_samples, 1]
        expert_logits = torch.stack(expert_logits, dim=0).squeeze(-1)  # shape: [num_experts, num_samples]
        
        # Compute gating weights from the pair representation.
        gate_logits = self.gate(pair_emb)  # shape: [num_samples, num_experts]
        gate_weights = F.softmax(gate_logits, dim=1)  # shape: [num_samples, num_experts]
        
        # Transpose expert_logits to [num_samples, num_experts] for weighted sum.
        expert_logits = expert_logits.transpose(0, 1)  # shape: [num_samples, num_experts]
        final_logits = torch.sum(gate_weights * expert_logits, dim=1)  # shape: [num_samples]
        
        prob = torch.sigmoid(final_logits)
        return prob, final_logits, x

# ============================================================================
# 6. Train the Model
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
node_features = node_features.to(device)
edge_index = edge_index.to(device)
train_pairs = train_pairs.to(device)
train_labels = train_labels.to(device)
val_pairs = val_pairs.to(device)
val_labels = val_labels.to(device)

# Instantiate the MoE model. (Here we use 3 experts; you can change num_experts as needed.)
model = GNNMoEModel(input_dim=1, hidden_dim=16, out_dim=16, mlp_hidden_dim=16, num_experts=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Variables to track best validation loss and corresponding epoch
best_val_loss = float('inf')
best_epoch = 0

num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    pred, logits, _ = model(node_features, edge_index, train_pairs)
    loss = criterion(pred, train_labels)
    loss.backward()
    optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_pred, _, _ = model(node_features, edge_index, val_pairs)
        val_loss = criterion(val_pred, val_labels)
    
    # Update best validation loss if improved
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_epoch = epoch
        improvement_str = " (new best!)"
    else:
        improvement_str = ""
    
    # Print log every 10 epochs (or for epoch 0)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
              f"Best Val: {best_val_loss:.4f} at epoch {best_epoch+1}{improvement_str}")

# ============================================================================
# 7. Prepare Submission Predictions
#    For each submission game, we extract the team IDs, map them to node indices,
#    then compute the win probability using the trained MoE GNN model.
# ============================================================================
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')

def extract_game_info(id_str):
    """Extract season and team IDs from the submission ID (format: 'Season_TeamID1_TeamID2')."""
    parts = id_str.split('_')
    season = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return season, teamID1, teamID2

# Extract season and team IDs from the submission 'ID' column.
submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(
    lambda x: pd.Series(extract_game_info(x))
)

# Merge seed info for TeamID1
submission_df = pd.merge(
    submission_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID1'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed info for TeamID2
submission_df = pd.merge(
    submission_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID2'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

# If seed values are missing, default them to 16.
submission_df['SeedValue1'] = submission_df['SeedValue1'].fillna(16)
submission_df['SeedValue2'] = submission_df['SeedValue2'].fillna(16)

# For our GNN model, the node features are based solely on team seed values.
# We need to create pairs for submission predictions.
# Map team IDs from submission_df to node indices (if a team is not in our training graph, default to 0).
def map_team(team_id):
    return team_id_to_idx.get(team_id, 0)

submission_pairs = []
for idx, row in submission_df.iterrows():
    i = map_team(row['TeamID1'])
    j = map_team(row['TeamID2'])
    submission_pairs.append([i, j])
submission_pairs = torch.tensor(submission_pairs, dtype=torch.long).to(device)

model.eval()
with torch.no_grad():
    pred_submission, _, _ = model(node_features, edge_index, submission_pairs)
    
submission_df['Pred'] = pred_submission.cpu().numpy()

# ============================================================================
# 8. Save the Submission File
# ============================================================================
submission_df[['ID', 'Pred']].fillna(-1).to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved.")





