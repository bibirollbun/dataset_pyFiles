# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import random
import numpy as np
import torch

# Set the seed for Python's random module
random.seed(42)

# Set the seed for NumPy
np.random.seed(42)

# Set the seed for PyTorch
torch.manual_seed(42)

import warnings
warnings.filterwarnings('ignore')


! pip install torch_geometric


import torch
import numpy as np
from sklearn.metrics import classification_report, top_k_accuracy_score
import pandas as pd
import torch
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy
from IPython.display import display, HTML

def get_xgb_preds(xgb_model, plays_df, batch, feature_encoders, scaler):

    # Get corresponding plays for XGBoost
    ids = list(
        zip(
            batch.game_id[batch.eligible_mask].cpu().tolist(),
            batch.play_id[batch.eligible_mask].cpu().tolist(),
            batch.player_ids[batch.eligible_mask].cpu().tolist(),
        )
    )
    batch_df = filter_by_game_play_ids(plays_df, ids)

    # XGB predictions
    X, y, _, _2 = prepare_route_prediction_data(
        batch_df,
        training=False,
        feature_encoders=feature_encoders,
        scaler=scaler,
    )
    xgb_probs = xgb_model.predict_proba(X)
    return xgb_probs

def evaluate_route_predictions_table(
    model,
    dataloader_,
    dataset_,
    device="cuda",
    entropy_ceiling=3,
    key="route_predictions",
    softmax=False,
    plays_df=None,
    feature_encoders=None, 
    scaler=None
):
    """
    Evaluates a PyTorch route prediction model with enhanced metrics and formatted table output.
    Now includes play-level and field position metrics.

    Parameters:
    - model: PyTorch model
    - dataloader: PyTorch DataLoader for evaluation
    - dataset: MultiRoutePlayDataset instance
    - device: Device to run evaluation on
    """
    try:
        model.eval()
    except Exception as e:
        print('Trying XGB')
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_metadata = []  # New list to store metadata

    with torch.no_grad():
        for batch in dataloader_:
            try:
                batch = batch.to(device)
                targets = batch.route_targets[batch.eligible_mask]
                try:
                    output = model(batch)
                    probabilities = output[key]
                    
                except Exception as e:
                    probabilities = get_xgb_preds(model, plays_df, batch, feature_encoders, scaler)
                    output = {}

                if "target" in output.keys():
                    torch.testing.assert_close(targets, torch.tensor(output["target"]))

                if isinstance(probabilities, torch.Tensor):
                    if softmax:
                        probabilities = torch.softmax(probabilities, dim=1)
                    predictions = probabilities.argmax(dim=1).cpu().numpy()
                    probabilities = probabilities.cpu().numpy()
                else:
                    predictions = torch.tensor(probabilities).argmax(dim=1).cpu().numpy()

                # Extract metadata for eligible players
                metadata = {
                    'play_id': batch.play_id[batch.eligible_mask].cpu().numpy(),
                    'game_id': batch.game_id[batch.eligible_mask].cpu().numpy(),
                    'player_id': batch.player_ids[batch.eligible_mask].cpu().numpy(),
                    'yardline': batch.yardline[batch.eligible_mask].cpu().numpy(),
                    'down': batch.down[batch.eligible_mask].cpu().numpy() if hasattr(batch, 'down') else None,
                    'yards_to_go': batch.yards_to_go[batch.eligible_mask].cpu().numpy() if hasattr(batch, 'yards_to_go') else None
                }
                
                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_targets.append(targets.cpu().numpy())
                all_metadata.append(metadata)
            except Exception as e:
                raise e

    try:
        y_pred = np.concatenate(all_predictions)
        y_pred_proba = np.concatenate(all_probabilities)
        y_test = np.concatenate(all_targets)
        
        # Concatenate metadata
        combined_metadata = {
            key: np.concatenate([batch[key] for batch in all_metadata]) 
            for key in all_metadata[0].keys()
            if all_metadata[0][key] is not None
        }
        
    except Exception as e:
        print(all_predictions)
        raise e

    # Apply entropy filtering
    prediction_entropies = np.apply_along_axis(entropy, 1, y_pred_proba)
    entropy_subset = prediction_entropies < entropy_ceiling

    y_pred = y_pred[entropy_subset]
    y_pred_proba = y_pred_proba[entropy_subset]
    y_test = y_test[entropy_subset]
    
    # Also filter metadata
    filtered_metadata = {
        key: value[entropy_subset] 
        for key, value in combined_metadata.items()
    }

    # Create DataFrame with all predictions and metadata
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'probabilities': y_pred_proba.tolist(),
        'predicted': y_pred,
        'entropy': prediction_entropies[entropy_subset],
        **filtered_metadata
    })
    
    # Convert route indices to names
    predictions_df['actual_route'] = predictions_df['actual'].map(dataset_.idx_to_route)
    predictions_df['predicted_route'] = predictions_df['predicted'].map(dataset_.idx_to_route)

    # Calculate base metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)
    actuals_predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    # Calculate actual vs predicted counts
    for class_label in np.unique(y_test):
        actual_count = len(
            actuals_predictions[actuals_predictions["Actual"] == class_label]
        )
        predicted_count = len(
            actuals_predictions[actuals_predictions["Predicted"] == class_label]
        )
        metrics[str(class_label)]["actual_count"] = actual_count
        metrics[str(class_label)]["predicted_count"] = predicted_count

    # Calculate uncertainty metrics
    def calculate_uncertainty_metrics(probabilities, y_true):
        prediction_entropies = np.apply_along_axis(entropy, 1, probabilities)
        true_label_probs = np.array(
            [prob[true] for prob, true in zip(probabilities, y_true)]
        )
        log_likelihood = np.log(true_label_probs + 1e-10)
        max_entropy = np.log2(probabilities.shape[1])
        normalized_entropies = prediction_entropies / max_entropy
        sorted_probs = np.sort(probabilities, axis=1)
        entropy_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        return {
            "mean_entropy": np.mean(prediction_entropies),
            "median_entropy": np.median(prediction_entropies),
            "mean_normalized_entropy": np.mean(normalized_entropies),
            "mean_entropy_margin": np.mean(entropy_margin),
            "mean_log_likelihood": np.mean(log_likelihood),
            "median_log_likelihood": np.median(log_likelihood),
        }

    metrics["uncertainty_metrics"] = calculate_uncertainty_metrics(y_pred_proba, y_test)

    # # Calculate Top-K accuracy
    # def top_k_accuracy(y_true, y_pred_proba, k):
    #     top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:]

    #     correct = 0
    #     for i, true_label in enumerate(y_true):
    #         if true_label in top_k_predictions[i]:
    #             correct += 1
    #     return correct / len(y_true)

    metrics["top_2_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=2, labels=list(range(0, 13))
    )
    metrics["top_3_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=3, labels=list(range(0, 13))
    )

    # Add predictions DataFrame to metrics
    metrics['predictions_df'] = predictions_df
    
    # Add some basic grouped metrics
    def calculate_grouped_metrics(df):
        play_metrics = {}
        
        # Redzone analysis (inside 20 yard line)
        redzone_mask = df['yardline'] <= 20
        play_metrics['redzone'] = {
            'accuracy': (df[redzone_mask]['actual'] == df[redzone_mask]['predicted']).mean(),
            'sample_size': redzone_mask.sum(),
            'route_distribution': df[redzone_mask]['actual_route'].value_counts().to_dict()
        }
        
        # Play-level accuracy
        play_level_accuracy = df.groupby(['play_id', 'game_id']).apply(
            lambda x: (x['actual'] == x['predicted']).mean()
        ).describe().to_dict()
        play_metrics['play_level'] = play_level_accuracy
        
        # Most common route combinations
        play_metrics['route_combinations'] = (
            df.groupby(['play_id', 'game_id'])['actual_route']
            .agg(lambda x: tuple(sorted(x)))  # Convert to sorted tuple for consistent ordering
            .value_counts()
            .head(10)
            .to_dict()
        )
        
        return play_metrics
    
    def display_formatted_metrics(metrics_dict, dataset_):
        # Per-class metrics table
        class_metrics = []
        for class_label in sorted([k for k in metrics_dict.keys() if k.isdigit()]):
            try:
                class_metrics.append(
                    {
                        "Route": dataset_.idx_to_route[int(class_label)],
                        "Actual Count": metrics_dict[class_label]["actual_count"],
                        "Predicted Count": metrics_dict[class_label]["predicted_count"],
                        "Precision": f"{metrics_dict[class_label]['precision']:.3f}",
                        "Recall": f"{metrics_dict[class_label]['recall']:.3f}",
                        "F1-score": f"{metrics_dict[class_label]['f1-score']:.3f}",
                        "Support": metrics_dict[class_label]["support"],
                    }
                )
            except Exception as e:
                print(f"Could not get metrics for {class_label}")

        class_df = pd.DataFrame(class_metrics)

        # Top-K accuracy table
        topk_metrics = pd.DataFrame(
            [
                {
                    "Metric": "Accuracy Type",
                    "Top-1 (standard)": f"{metrics_dict['accuracy']:.3f}",
                    "Top-2": f"{metrics_dict['top_2_accuracy']:.3f}",
                    "Top-3": f"{metrics_dict['top_3_accuracy']:.3f}",
                }
            ]
        )

        # Uncertainty metrics table
        uncertainty_df = pd.DataFrame(
            [
                {
                    "Mean Entropy": f"{metrics_dict['uncertainty_metrics']['mean_entropy']:.3f}",
                    "Median Entropy": f"{metrics_dict['uncertainty_metrics']['median_entropy']:.3f}",
                    "Mean Normalized Entropy": f"{metrics_dict['uncertainty_metrics']['mean_normalized_entropy']:.3f}",
                    "Mean Entropy Margin": f"{metrics_dict['uncertainty_metrics']['mean_entropy_margin']:.3f}",
                    "Mean Log Likelihood": f"{metrics_dict['uncertainty_metrics']['mean_log_likelihood']:.3f}",
                    "Median Log Likelihood": f"{metrics_dict['uncertainty_metrics']['median_log_likelihood']:.3f}",
                }
            ]
        )

        # Overall metrics table
        overall_metrics = []
        for avg_type in ["macro avg", "weighted avg"]:
            metrics_row = {"Average Type": avg_type}
            metrics_row.update(
                {k: f"{v:.3f}" for k, v in metrics_dict[avg_type].items()}
            )
            overall_metrics.append(metrics_row)
        overall_df = pd.DataFrame(overall_metrics)

        # Display tables with styling
        print("\n=== Per-Class Performance ===")
        display(
            HTML(
                class_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Top-K Accuracy ===")
        display(
            HTML(
                topk_metrics.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Uncertainty Metrics ===")
        display(
            HTML(
                uncertainty_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Overall Metrics ===")
        display(
            HTML(
                overall_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

    metrics["display_tables"] = lambda: display_formatted_metrics(metrics, dataset_)

    metrics['grouped_metrics'] = calculate_grouped_metrics(predictions_df)
    
    return metrics, (y_test, y_pred, y_pred_proba, predictions_df)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch


# Add residual connections and layer normalization
class EnhancedGATBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, heads, dropout, concat=False):
        super().__init__()
        self.gat = GATv2Conv(
            in_dim,
            hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
            concat=concat,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.gat(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class PlayGNN(nn.Module):
    def __init__(
        self,
        num_positions,
        hidden_dim=128,
        num_gnn_layers=3,
        num_route_classes=10,
        dropout=0.1,
        max_downs=4,
        max_quarters=4,
        num_teams=32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Global feature embeddings
        position_emb_dim = 4
        down_emb_dim = 2
        quarter_emb_dim = 2
        team_emb_dim = 8
        player_emb_dim = 16
        self.position_embedding = nn.Embedding(num_positions, position_emb_dim)
        self.down_emb = nn.Embedding(max_downs, down_emb_dim)
        self.quarter_emb = nn.Embedding(max_quarters, quarter_emb_dim)
        self.team_emb = nn.Embedding(num_teams, team_emb_dim)
        self.player_emb = nn.Embedding(1000, player_emb_dim)

        # Remove feature encoders since we're directly concatenating
        # Note: GNN layers will handle the concatenated features

        self.max_per_graph = 16

        emb_dims = position_emb_dim + down_emb_dim + quarter_emb_dim + team_emb_dim + player_emb_dim

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                EnhancedGATBlock(
                    hidden_dim,
                    hidden_dim,
                    edge_dim=3,
                    heads=2,
                    dropout=dropout,
                    concat=False,
                )
            )
        # cannot use residual because shapes do not align

        self.gnn_layers[0] = GATv2Conv(
            5 + 4 + emb_dims,
            hidden_dim,
            edge_dim=3,
            heads=2,
            dropout=dropout,
            concat=False,
        )

        # Frame-level processing
        self.frame_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Historical plays attention
        self.historical_attention = nn.MultiheadAttention(
            hidden_dim * 2,  # bidirectional LSTM output
            num_heads=2,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.max_per_graph),
            nn.GELU(),
            nn.LayerNorm(self.max_per_graph),
            nn.Dropout(dropout),
            nn.Linear(self.max_per_graph, num_route_classes),
        )

    def forward(self, data):
        # Extract batch information
        batch_size = data.batch.max().item() + 1
        device = data.x.device

        # Extract individual features from data.x
        position_idx = data.x[:, 0].long()  # Assuming position index is first column
        raw_features = data.x[:, 1:5]  # Rest of features (weight, height, is_offense, motion)

        # Create position embeddings
        position_embedded = self.position_embedding(
            position_idx
        )  # [num_nodes, hidden_dim]

        # Global features - indexed by batch
        down_embedded = self.down_emb(
            data.down - 1
        )  # [num_nodes, hidden_dim]
        quarter_embedded = self.quarter_emb(
            data.quarter - 1
        )  # [num_nodes, hidden_dim]
        team_embedded = self.team_emb(
            data.offense_team
        )  # [num_nodes, hidden_dim]
        player_embedded = self.player_emb(
            data.player_ids
        )

        game_clock = data.game_clock
        yardline = data.yardline
        yards_to_go = data.yards_to_go

        numeric_features = torch.cat(
            [
                game_clock.unsqueeze(1),
                yardline.unsqueeze(1),
                yards_to_go.unsqueeze(1),
                data.offense_score.unsqueeze(1),
                data.defense_score.unsqueeze(1),
            ],
            dim=1,
        )

        # Concatenate all features
        node_features = torch.cat(
            [
                position_embedded,  # [num_nodes, hidden_dim]
                raw_features,  # [num_nodes, num_raw_features]
                down_embedded,  # [num_nodes, hidden_dim]
                quarter_embedded,  # [num_nodes, hidden_dim]
                team_embedded,  # [num_nodes, hidden_dim]
                numeric_features,
                player_embedded
            ],
            dim=1,
        )

        edge_features = data.edge_attr[:, 0:3]  # Keep edge features as is

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, data.edge_index, edge_features)
            node_features = F.gelu(node_features)

        # Separate historical and current plays using plays_elapsed
        historical_mask = data.plays_elapsed > 0
        current_mask = data.plays_elapsed == 0

        # Create new batch indices for historical and current plays separately
        batch_size = data.batch.max().item() + 1
        historical_batch = data.batch[historical_mask]
        current_batch = data.batch[current_mask]

        # Get dense batched representations
        historical_nodes, historical_lens = to_dense_batch(
            node_features[historical_mask],
            historical_batch,
            max_num_nodes=self.max_per_graph,
        )
        current_nodes, current_lens = to_dense_batch(
            node_features[current_mask], current_batch, max_num_nodes=self.max_per_graph
        )

        # Process through LSTM
        historical_encoded, _ = self.frame_lstm(historical_nodes)
        current_encoded, _ = self.frame_lstm(current_nodes)

        # Pool frames to get play-level representations
        # Use the dense masks from to_dense_batch to properly pool
        historical_sum = historical_lens.sum(dim=1, keepdim=True).clamp(min=1e-6)
        current_sum = current_lens.sum(dim=1, keepdim=True).clamp(min=1e-6)

        historical_plays = (
            torch.sum(historical_encoded * historical_lens.unsqueeze(-1), dim=1)
            / historical_sum
        )
        current_plays = (
            torch.sum(current_encoded * current_lens.unsqueeze(-1), dim=1) / current_sum
        )

        # Apply attention between current play and historical plays
        attended_history, _ = self.historical_attention(
            current_plays, historical_plays, historical_plays
        )

        # Get the play-level context as before
        final_representation = torch.cat(
            [current_plays.squeeze(1), attended_history.squeeze(1)], dim=-1
        )  # [batch_size, hidden_dim*4]

        # print('node feats')
        # print(node_features.shape)

        # Get current eligible receivers
        current_eligible_mask = data.eligible_mask & (data.plays_elapsed == 0)
        eligible_nodes = node_features[
            current_eligible_mask
        ]  # [num_eligible_total, hidden_dim]
        eligible_batch_idx = data.batch[current_eligible_mask]  # [num_eligible_total]

        # Get the corresponding play context for each eligible receiver
        play_context = final_representation[
            eligible_batch_idx
        ]  # [num_eligible_total, hidden_dim*4]

        # Combine receiver features with play context
        combined_features = torch.cat(
            [
                eligible_nodes,  # Receiver-specific features
                play_context,  # Play-level context
            ],
            dim=1,
        )  # [num_eligible_total, hidden_dim + hidden_dim*4]

        # print('combined shape')
        # print(combined_features.shape)
        # print(combined_features.isnan().sum())

        # Generate unique predictions for each eligible receiver
        # print(self.mlp)
        route_predictions = self.mlp(
            combined_features
        )  # [num_eligible_total, num_route_classes]

        # print('mlp shape')
        # print(route_predictions.shape)

        # print(route_predictions.isnan().sum())

        return {
            "route_predictions": route_predictions,
            "eligible_batch_idx": eligible_batch_idx,  # Keep track of which batch each prediction belongs to
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)



from file_imports import (
    HistoricalMultiRoutePlayDataset,
    create_game_play_pairs,
    create_batch_data,
    process_game_scores,
    DataMappings,
    filter_by_game_play_ids,
    create_week_stratified_split,
    load_and_filter_data, 
    filter_passing_plays_only,
    train_route_prediction_pipeline, 
    prepare_route_prediction_data,
    train_route_predictor, 
    MultiRouteLoss,
    
)
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt


import pickle

with open("/kaggle/input/compare-models-and-ensemble/player_id_mapping.pkl", "rb") as f:
    player_id_mapping = pickle.load(f)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

import warnings

tracking, player_id_mapping = load_and_filter_data(weeks=list(range(6,10)), base_path="/kaggle/input", existing_mapping=player_id_mapping)
tracking_passing_only = filter_passing_plays_only(tracking)
tracking_passing_only = process_game_scores(tracking_passing_only)
tracking_passing_only = tracking_passing_only.sort_values(by=["time"])
print(tracking_passing_only.shape)


tracking_passing_only.loc[
    (tracking_passing_only.routeRan.isna()) & (tracking_passing_only.position.isin(('WR', 'TE', 'RB'))), 
     'routeRan'
] = 'BLOCKING'

mappings = DataMappings()
mappings.fit(tracking_passing_only)

# with open("mapping.pkl", "rb") as f:
#     mappings = pickle.load(f)
unique_routes = sorted([route for route in tracking_passing_only.routeRan.unique() if not pd.isna(route)])
unique_routes

dataset = HistoricalMultiRoutePlayDataset(
    df=tracking_passing_only,
    game_play_pairs=create_game_play_pairs(tracking_passing_only),
    target_df=tracking_passing_only[['gameId','playId','routeRan']],
    offense_positions=['QB', 'RB', 'WR', 'TE'],
    defense_positions=['CB', 'SS', 'FS', 'LB', 'OLB', 'ILB'],
    eligible_positions=['WR', 'TE', 'RB'],
    n_frames=1,
    device='cpu',
    unique_routes=unique_routes,
    mappings=mappings,  # Pass the mappings object
    n_workers=1,
    teams_per_chunk=32,
    max_history_plays=6,
    augment=False,
    do_not_augment_weeks=[7, 8]
)


train_weeks = list(range(7,8))
val_weeks = list(range(8,10))
test_weeks = list(range(9,10))

train_loader, val_loader, test_loader, indices, week_ranges = create_week_stratified_split(
        dataset, 
        train_weeks=train_weeks,
        val_weeks=val_weeks,
        test_weeks=test_weeks,
        # train_weeks=list(range(1,3)),
        # val_weeks=list(range(3,4)),
        # test_weeks=list(range(3,4)),
        batch_size=128, 
        random_seed=42,
        preserve_time_order=False,
        val_random=False  # Whether to randomly sample validation set from train period
)

model = PlayGNN(
    num_positions=20,
    hidden_dim=64,
    num_gnn_layers=3,
    num_route_classes=dataset.num_route_classes, # for -1 or NaN
    dropout=0.05,
    max_downs=4,
    max_quarters=5,
    num_teams=32,
)


PATH = "/kaggle/input/compare-models-and-ensemble/model_gnn_1_6.pt"
model.load_state_dict(torch.load(PATH, weights_only=True, map_location=torch.device('cpu')))
model.eval()



import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, top_k_accuracy_score
import numpy as np
import pandas as pd


def accuracy_report_by_entropy(predictions_df, play_grouped, cutoff=2.1, k_values=[1, 2, 3]):
    print(f'Initial DF size: {len(predictions_df)}')
    
    # Filter dataframes as before
    filtered_by_mean_play_entropy = predictions_df[predictions_df.play_id.isin(play_grouped['most_accurate_plays'].reset_index().play_id)]
    filtered_by_play_and_row_entropy = filtered_by_mean_play_entropy[filtered_by_mean_play_entropy.entropy < cutoff]
    filtered_by_row_entropy = predictions_df[predictions_df.entropy < cutoff]
    print(f'Final filtering size: {len(filtered_by_play_and_row_entropy)}')
    
    def calculate_topk_accuracy(actual, predicted_probs):
        # if isinstance(predicted_probs[0], list):
            # If probabilities are stored as lists
        predicted_probs = np.array(predicted_probs)
        # elif isinstance(predicted_probs, pd.DataFrame):
        #     # If probabilities are stored in separate columns
        #     prob_cols = [col for col in predicted_probs.columns if col.startswith('prob_class_')]
        #     predicted_probs = predicted_probs[prob_cols].values
        
        topk_accuracies = {}
        for k in k_values:
            top_k_preds = np.argsort(-predicted_probs, axis=1)[:, :k]
            correct = [actual[i] in top_k_preds[i] for i in range(len(actual))]
            accuracy = np.mean(correct)
            topk_accuracies[f'Top-{k} Accuracy'] = f"{accuracy:.3f}"
        return topk_accuracies

    # Calculate and print Top-K accuracies for each filtered dataset
    print(f"\n=== Filtered by Row Entropy Only - Coverage: N = {len(filtered_by_row_entropy)}/{len(predictions_df)} {round(100 * len(filtered_by_row_entropy)/len(predictions_df), 2)} % ===")
    print(classification_report(filtered_by_row_entropy.actual, filtered_by_row_entropy.predicted))
    topk_metrics = calculate_topk_accuracy(
        filtered_by_row_entropy.actual.values,
        filtered_by_row_entropy.probabilities.tolist()
    )
    print("Top-K Accuracy Metrics:")
    for metric, value in topk_metrics.items():
        print(f"{metric}: {value}")

    print(f"\n=== Filtered by Mean Play Entropy - Coverage: N = {len(filtered_by_mean_play_entropy)}/{len(predictions_df)} {round(100 * len(filtered_by_mean_play_entropy)/len(predictions_df), 2)} % ===")
    print(classification_report(filtered_by_mean_play_entropy.actual, filtered_by_mean_play_entropy.predicted))
    topk_metrics = calculate_topk_accuracy(
        filtered_by_mean_play_entropy.actual.values,
        filtered_by_mean_play_entropy.probabilities.tolist()
    )
    print("Top-K Accuracy Metrics:")
    for metric, value in topk_metrics.items():
        print(f"{metric}: {value}")

    print(f"\n=== Filtered by Both Play and Row Entropy - Coverage: N = {len(filtered_by_play_and_row_entropy)}/{len(predictions_df)} {round(100 * len(filtered_by_play_and_row_entropy)/len(predictions_df), 2)} % ===", )
    print(classification_report(filtered_by_play_and_row_entropy.actual, filtered_by_play_and_row_entropy.predicted))
    topk_metrics = calculate_topk_accuracy(
        filtered_by_play_and_row_entropy.actual.values,
        filtered_by_play_and_row_entropy.probabilities.tolist()
    )
    print("Top-K Accuracy Metrics:")
    for metric, value in topk_metrics.items():
        print(f"{metric}: {value}")

def get_high_confidence_metrics(predictions_df, unique_routes, confidence_threshold=0.7, k=1):
    """
    Calculate classification metrics for high-confidence predictions.
    
    Args:
        predictions_df: DataFrame with 'actual', 'predicted', and 'probabilities' columns
        confidence_threshold: Minimum probability threshold for high confidence predictions
        k: Number of top predictions to consider for top-k accuracy
        
    Returns:
        Dictionary containing classification report and top-k accuracy for high-confidence predictions
    """
    # Get max probability for each prediction
    predictions_df['max_probability'] = predictions_df['probabilities'].apply(max)
    
    # Filter for high confidence predictions
    high_conf_df = predictions_df[predictions_df['max_probability'] >= confidence_threshold].copy()
    
    # Calculate metrics using all possible labels
    classification_metrics = classification_report(
        high_conf_df['actual'],
        high_conf_df['predicted'],
        labels=unique_routes,
        output_dict=True,
        zero_division=0
    )
    
    # Calculate top-k accuracy for high confidence predictions
    def is_actual_in_topk(row, k):
        # Sort probabilities in descending order and get top k indices
        topk_indices = np.argsort(row['probabilities'])[-k:]
        # Convert actual to index (assuming actual is a label that maps to probability index)
        actual_idx = row['actual']
        return actual_idx in topk_indices
    
    topk_accuracy = (
        high_conf_df
        .apply(lambda x: is_actual_in_topk(x, k), axis=1)
        .mean()
    )
    
    # Prepare summary
    n_total = len(predictions_df)
    n_high_conf = len(high_conf_df)
    
    return {
        'classification_report': classification_metrics,
        f'top{k}_accuracy': topk_accuracy,
        'coverage': n_high_conf / n_total,
        'n_predictions': n_high_conf,
        'n_total': n_total,
        'confidence_threshold': confidence_threshold
    }

def print_metrics_summary(metrics, k):
    """Pretty print the metrics summary"""
    accuracy_key = f'top{k}_accuracy'
    print(f"\nHigh Confidence Predictions (threshold >= {metrics['confidence_threshold']:.1%})")
    print(f"Coverage: {metrics['coverage']:.1%} ({metrics['n_predictions']:,} / {metrics['n_total']:,} predictions)")
    print(f"\nTop-{k} Accuracy: {metrics[accuracy_key]:.1%}")
    
    # Print classification report metrics
    report = metrics['classification_report']
    print("\nClassification Report:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    
    # Print metrics for each class
    for class_name, class_metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{class_name:<15} {class_metrics['precision']:>10.3f} "
                  f"{class_metrics['recall']:>10.3f} {class_metrics['f1-score']:>10.3f} "
                  f"{class_metrics['support']:>10}")
    
    # # Print averages
    # print("-" * 55)
    # print(report)
    # print(f"{'Accuracy':<15} {report['accuracy']:>10.3f}")
    # print(f"{'Macro Avg':<15} {report['macro avg']['precision']:>10.3f} "
    #       f"{report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}")

def groupby_entropy_and_accuracy(predictions_df, k=100, extra_groupings=None):
    # Calculate accuracy for each play and sort by worst performing
    extra_groupings = [] if extra_groupings is None else extra_groupings
    predictions_df['is_correct'] = predictions_df['actual'] == predictions_df['predicted']
    play_accuracy = (
        predictions_df
        .groupby(['play_id','game_id'] + extra_groupings)
        .agg({
            'is_correct': ['mean', 'count'],  # get both accuracy and number of routes
            'actual_route': list,              # see what the actual routes were
            'predicted_route': list,           # see what we predicted
            'yardline': 'first',               # include yardline for context
            'entropy': ['mean', 'max'],
        })
    )

    # Flatten the column names
    play_accuracy.columns = ['accuracy', 'num_routes', 'actual_routes', 'predicted_routes', 'yardline', 'entropy', 'max_entropy']

    print(len(play_accuracy))

    # Sort by lowest accuracy (highest error rate)
    worst_plays = (
        play_accuracy
        .sort_values('accuracy', ascending=True)  # ascending=True puts worst plays first
        .assign(error_rate=lambda x: 1 - x['accuracy'])
    )[0:k]

    best_plays = (
        play_accuracy
        .sort_values('accuracy', ascending=False)  # ascending=True puts worst plays first
        .assign(error_rate=lambda x: 1 - x['accuracy'])
    )[0:k]

    uncertain_plays = (
        play_accuracy
        .sort_values('entropy', ascending=False)  # ascending=True puts worst plays first
        .assign(error_rate=lambda x: x['entropy'])
    )[0:k]

    certain_plays = (
        play_accuracy
        .sort_values('entropy', ascending=True)  # ascending=True puts worst plays first
        .assign(error_rate=lambda x: x['accuracy'])
    )[0:k]

    return {
        "least_accurate_plays": worst_plays,
        "most_accurate_plays": best_plays,
        "high_entropy_plays": uncertain_plays,
        "low_entropy_plays": certain_plays,
        "play_accuracy": play_accuracy
    }

def plot_entropy_and_accuracy(
        least_accurate_plays, 
        most_accurate_plays, 
        high_entropy_plays, 
        low_entropy_plays, 
        title_update="Play",
        ax1=None, 
        ax2=None,
        ax3=None,
        plot_type='kde',  # Add plot_type parameter
        *args, 
        **kwargs
    ):

    # Create a figure with 3 subplots arranged vertically
    if ax1 is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(8, 10))

    # Helper function to choose between kde and histogram
    def plot_distribution(data, x, label, ax):
        if plot_type == 'kde':
            sns.kdeplot(data=data, x=x, fill=True, alpha=0.5, label=label, ax=ax)
        else:  # histogram
            sns.histplot(data=data, x=x, alpha=0.5, label=label, ax=ax, stat='density')

    # First subplot - Entropy
    plot_distribution(least_accurate_plays, 'entropy', f'Low Accuracy {title_update}s', ax1)
    plot_distribution(most_accurate_plays, 'entropy', f'High Accuracy {title_update}s', ax1)
    ax1.set_title(f'Distribution of {title_update} Entropy', fontsize=12)
    ax1.set_xlabel('Entropy', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend(fontsize=9)

    # Second subplot - Max Entropy
    plot_distribution(least_accurate_plays, 'max_entropy', f'Low Accuracy {title_update}s', ax2)
    plot_distribution(most_accurate_plays, 'max_entropy', f'High Accuracy {title_update}s', ax2)
    ax2.set_title(f'Distribution of {title_update} Max Entropy', fontsize=12)
    ax2.set_xlabel('Max Entropy', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend(fontsize=9)

    # Third subplot - Accuracy
    plot_distribution(low_entropy_plays, 'accuracy', f'Low Entropy {title_update}s', ax3)
    plot_distribution(high_entropy_plays, 'accuracy', f'High Entropy {title_update}s', ax3)
    ax3.set_title(f'Distribution of {title_update} Accuracy', fontsize=12)
    ax3.set_xlabel('Accuracy', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.legend(fontsize=9)

    # Adjust layout to prevent overlap
    if ax1 is None:
        plt.tight_layout()
        plt.show()

# def plot_entropy_and_accuracy(
#         least_accurate_plays, 
#         most_accurate_plays, high_entropy_plays, 
#         low_entropy_plays, 
#         title_update="Play",
#         ax1=None, 
#         ax2=None,
#         ax3=None,
#         *args, 
#         **kwargs
#     ):

#     # Create a figure with 3 subplots arranged vertically
#     if ax1 is None:
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(8, 10))

#     # First subplot - Entropy
#     sns.kdeplot(data=least_accurate_plays, x='entropy', fill=True, alpha=0.5, label=f'Low Accuracy {title_update}s', ax=ax1)
#     sns.kdeplot(data=most_accurate_plays, x='entropy', fill=True, alpha=0.5, label=f'High Accuracy {title_update}s', ax=ax1)
#     ax1.set_title(f'Distribution of {title_update} Entropy', fontsize=12)
#     ax1.set_xlabel('Entropy', fontsize=10)
#     ax1.set_ylabel('Density', fontsize=10)
#     ax1.legend(fontsize=9)

#     # Second subplot - Max Entropy
#     sns.kdeplot(data=least_accurate_plays, x='max_entropy', fill=True, alpha=0.5, label=f'Low Accuracy {title_update}s', ax=ax2)
#     sns.kdeplot(data=most_accurate_plays, x='max_entropy', fill=True, alpha=0.5, label=f'High Accuracy {title_update}s', ax=ax2)
#     ax2.set_title(f'Distribution of {title_update} Max Entropy', fontsize=12)
#     ax2.set_xlabel('Max Entropy', fontsize=10)
#     ax2.set_ylabel('Density', fontsize=10)
#     ax2.legend(fontsize=9)

#     # Third subplot - Accuracy
#     sns.kdeplot(data=low_entropy_plays, x='accuracy', fill=True, alpha=0.5, label=f'Low Entropy {title_update}s', ax=ax3)
#     sns.kdeplot(data=high_entropy_plays, x='accuracy', fill=True, alpha=0.5, label=f'High Entropy {title_update}s', ax=ax3)
#     ax3.set_title(f'Distribution of {title_update} Accuracy', fontsize=12)
#     ax3.set_xlabel('Accuracy', fontsize=10)
#     ax3.set_ylabel('Density', fontsize=10)
#     ax3.legend(fontsize=9)

#     # Adjust layout to prevent overlap
#     if ax1 is None:
#         plt.tight_layout()
#         plt.show()
        

from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

def get_high_confidence_metrics(predictions_df, unique_routes, confidence_threshold=0.7, k=1):
    """
    Calculate classification metrics for high-confidence predictions.
    
    Args:
        predictions_df: DataFrame with 'actual', 'predicted', and 'probabilities' columns
        unique_routes: List of all possible route labels
        confidence_threshold: Minimum probability threshold for high confidence predictions
        k: Number of top predictions to consider for top-k accuracy
        
    Returns:
        Dictionary containing classification report and top-k accuracy for high-confidence predictions
    """
    # Get max probability for each prediction
    predictions_df['max_probability'] = predictions_df['probabilities'].apply(max)
    
    # Filter for high confidence predictions
    high_conf_df = predictions_df[predictions_df['max_probability'] >= confidence_threshold].copy()
    
    # Calculate top-k accuracy for high confidence predictions
    def is_actual_in_topk(row, k):
        # Get indices of top k probabilities
        topk_indices = np.argsort(row['probabilities'])[-k:]
        # actual is already an index, use it directly
        return row['actual'] in topk_indices
    
    # Map actual and predicted to same label space
    def ensure_label_consistency(df):
        # Map actual and predicted values to indices in unique_routes if they aren't already
        if not isinstance(df['actual'].iloc[0], (int, np.integer)):
            df['actual_idx'] = df['actual'].apply(lambda x: unique_routes.index(x))
        else:
            df['actual_idx'] = df['actual']
            
        if not isinstance(df['predicted'].iloc[0], (int, np.integer)):
            df['predicted_idx'] = df['predicted'].apply(lambda x: unique_routes.index(x))
        else:
            df['predicted_idx'] = df['predicted']
        return df
    
    high_conf_df = ensure_label_consistency(high_conf_df)
    
    # Map integer indices to route names for classification report
    high_conf_df['actual_route'] = high_conf_df['actual'].apply(lambda x: unique_routes[x])
    high_conf_df['predicted_route'] = high_conf_df['predicted'].apply(lambda x: unique_routes[x])
    
    # Calculate metrics using mapped route names
    classification_metrics = classification_report(
        high_conf_df['actual_route'],
        high_conf_df['predicted_route'],
        labels=unique_routes,
        output_dict=True,
        zero_division=0
    )
    
    # Remove the label mapping code since we're using strings directly
    mapped_metrics = classification_metrics
    
    # Calculate top-k accuracies for k=1,2,3
    def calc_topk_accuracy(df, k):
        return df.apply(lambda x: is_actual_in_topk(x, k), axis=1).mean()
    
    topk_accuracies = {
        f'top{k}_accuracy': calc_topk_accuracy(high_conf_df, k)
        for k in [1, 2, 3]
    }

    for key, v in topk_accuracies.items():
        print(f'{key}: {round(v, 2)}')
    
    # Prepare summary
    n_total = len(predictions_df)
    n_high_conf = len(high_conf_df)
    
    # Add debug information
    debug_info = {
        'sample_actual': high_conf_df['actual'].head().tolist(),
        'sample_predicted': high_conf_df['predicted'].head().tolist(),
        'sample_probabilities_shape': [len(p) for p in high_conf_df['probabilities'].head()],
        'unique_actual_values': high_conf_df['actual'].nunique(),
        'unique_predicted_values': high_conf_df['predicted'].nunique(),
        'number_of_routes': len(unique_routes)
    }
    
    return {
        'classification_report': mapped_metrics,
        f'top{k}_accuracy': topk_accuracies[f'top{k}_accuracy'],
        'coverage': n_high_conf / n_total,
        'n_predictions': n_high_conf,
        'n_total': n_total,
        'confidence_threshold': confidence_threshold,
        'debug_info': debug_info
    }


metrics_gnn, (y_test, y_pred, y_pred_proba, predictions_df) = evaluate_route_predictions_table(
    model, val_loader, dataset, "cpu", softmax=True
)
metrics_gnn['display_tables']()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import entropy
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from tqdm import tqdm


def get_initial_center_position(frame_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Get the initial position of the center for normalization reference.
    Falls back to offensive line average if no center is found.

    Returns:
        Tuple of (x, y) coordinates for reference point
    """
    # Get the first frame
    first_frame = frame_data[frame_data["frameId"] == frame_data["frameId"].min()]

    # Find Center's position
    center = first_frame[first_frame["position"] == "C"]
    if len(center) == 1:
        return center["x"].iloc[0], center["y"].iloc[0]

    # Fallback to offensive line average if no Center is found
    offensive_line = first_frame[first_frame["position"].isin(["C", "G", "T", "OL"])]
    if len(offensive_line) > 0:
        return offensive_line["x"].mean(), offensive_line["y"].mean()

    # Final fallback to offensive players average
    offense = first_frame[first_frame["position_group"] == "Offense"]
    return offense["x"].mean(), offense["y"].mean()


def normalize_coordinates(
    frame_data: pd.DataFrame, play_direction: str, ref_x: float, ref_y: float
) -> pd.DataFrame:
    """
    Normalize player coordinates relative to the initial center position and play direction.
    """
    frame_data = frame_data.copy()

    # Normalize relative to reference point
    frame_data["x"] = frame_data["x"] - ref_x
    frame_data["y"] = frame_data["y"] - ref_y

    # Flip coordinates if play is going left
    if play_direction.lower() == "left":
        frame_data["x"] = -frame_data["x"]
        # frame_data["y"] = -frame_data["y"]

    return frame_data


def get_detailed_position_group(position: str) -> str:
    """
    Categorize players into specific position groups for more granular comparison.
    """
    position_groups = {
        "QB": "Quarterback",
        "C": "Offensive Line",
        "G": "Offensive Line",
        "T": "Offensive Line",
        "OL": "Offensive Line",
        "RB": "Skill Players",
        "FB": "Skill Players",
        "WR": "Skill Players",
        "TE": "Skill Players",
        "DE": "Defensive Line",
        "DT": "Defensive Line",
        "NT": "Defensive Line",
        "ILB": "Linebackers",
        "OLB": "Linebackers",
        "MLB": "Linebackers",
        "LB": "Linebackers",
        "CB": "Secondary",
        "S": "Secondary",
        "FS": "Secondary",
        "SS": "Secondary",
        "DB": "Secondary",
    }
    return position_groups.get(position, "Other")


def get_players_by_frame(
    play_df: pd.DataFrame, frame_start: int = None, frame_count: int = None
) -> dict:
    """
    Convert play DataFrame into frame-indexed dictionary of normalized player coordinates by position group.
    """
    # Add position group column
    play_df = play_df.copy()
    play_df["position_group"] = play_df["position"].map(get_detailed_position_group)

    # Get play direction
    play_direction = play_df["playDirection"].iloc[0]

    # Get initial center position for normalization
    ref_x, ref_y = get_initial_center_position(play_df)

    # Get frame range if specified
    if frame_start is not None:
        frame_ids = sorted(play_df["frameId"].unique())
        if frame_start < 0:  # Handle negative indexing
            frame_start = len(frame_ids) + frame_start
        frame_start = max(0, min(frame_start, len(frame_ids) - 1))

        if frame_count is not None:
            frame_ids = frame_ids[frame_start : frame_start + frame_count]
        else:
            frame_ids = frame_ids[frame_start:]

        play_df = play_df[play_df["frameId"].isin(frame_ids)]

    frames = {}
    # Group by frameId for efficiency
    for frame_id, frame_data in play_df.groupby("frameId"):
        # Normalize coordinates for this frame
        normalized_frame = normalize_coordinates(
            frame_data, play_direction, ref_x, ref_y
        )

        frames[frame_id] = {
            "Quarterback": [],
            "Offensive Line": [],
            "Skill Players": [],
            "Defensive Line": [],
            "Linebackers": [],
            "Secondary": [],
            "Other": [],
        }

        # Then group by position_group
        for group, group_data in normalized_frame.groupby("position_group"):
            frames[frame_id][group] = group_data[
                ["x", "y", "nflId", "club", "position"]
            ].to_dict("records")

    return frames


def analyze_receiver_positions(
    frame_data: dict, relative_distances: bool = True
) -> Dict[str, Dict]:
    """
    Analyze the positions and distances between receivers (WR/TE) in a frame.

    Args:
        frame_data: Dictionary containing player positions by position group
        relative_distances: If True, order receivers by distance to each receiver
                          If False, order receivers by absolute Y position

    Returns:
        Dictionary containing:
        - wr_positions: Dict mapping nflId to their position (1-5)
        - distances: Dict mapping nflId to their distances to all receivers
        - nearest_distances: Dict mapping nflId to their closest receiver distance
        - center_distances: Dict mapping nflId to their signed distance from center
    """
    # Extract receivers from skill players group
    receivers = [
        player
        for player in frame_data["Skill Players"]
        if player["position"] in ("WR", "TE")
    ]

    if not receivers:
        return {
            "wr_positions": {},
            "distances": {},
            "nearest_distances": {},
            "center_distances": {},
        }

    # Convert to numpy array for vectorized calculations
    receiver_coords = np.array([[r["x"], r["y"]] for r in receivers])

    # Calculate pairwise distances between all receivers
    distances = np.sqrt(
        np.sum((receiver_coords[:, np.newaxis] - receiver_coords) ** 2, axis=2)
    )

    # Create distance mappings and positions
    distances_dict = {}
    wr_positions = {}
    nearest_distances = {}

    for i, receiver in enumerate(receivers):
        # Get distances to all receivers (will include self as 0)
        all_distances = distances[i]
        receiver_to_others = [(j, d) for j, d in enumerate(all_distances) if j != i]

        if relative_distances:
            # Sort other receivers by distance to current receiver while keeping original indices
            sorted_receivers = sorted(receiver_to_others, key=lambda x: x[1])

            # Map distances to receiver numbers (1-5 based on proximity)
            receiver_distances = {
                position + 1: distance
                for position, (_, distance) in enumerate(sorted_receivers)
            }

            # Store nearest distance (actual minimum distance)
            nearest_distances[receiver["nflId"]] = (
                sorted_receivers[0][1] if sorted_receivers else np.nan
            )

        else:
            # Sort receivers by Y position
            y_positions = sorted(range(len(receivers)), key=lambda x: receivers[x]["y"])
            position_map = {idx: pos + 1 for pos, idx in enumerate(y_positions)}

            # Map distances using Y-based positions
            receiver_distances = {position_map[j]: d for j, d in receiver_to_others}

            # Store Y-based position
            wr_positions[receiver["nflId"]] = position_map[i]

            # Store nearest distance (minimum of all distances)
            nearest_distances[receiver["nflId"]] = (
                min(d for _, d in receiver_to_others) if receiver_to_others else np.nan
            )

        distances_dict[receiver["nflId"]] = receiver_distances

    # Calculate signed distances from center (0, 0)
    center_distances = {
        receiver["nflId"]: receiver[
            "x"
        ]  # X-coordinate is already normalized relative to center
        for receiver in receivers
    }

    return {
        "wr_positions": wr_positions,
        "distances": distances_dict,
        "nearest_distances": nearest_distances,
        "center_distances": center_distances,
    }


def join_receiver_analysis_to_df(
    tracking_df: pd.DataFrame,
    game_id: int,
    play_id: int,
    relative_distances: bool = True,
    frame_data: dict = None,
) -> pd.DataFrame:
    """
    Join receiver analysis results to tracking DataFrame for a specific play.

    Args:
        tracking_df: Original tracking DataFrame
        game_id: Game ID to analyze
        play_id: Play ID to analyze
        relative_distances: If True, order receivers by distance to each receiver
                          If False, order receivers by absolute Y position
        frame_data: Optional pre-computed frame data

    Returns:
        DataFrame with receiver analysis columns added for WR/TE players
    """
    # Filter for specific game and play

    play_df = tracking_df[
        (tracking_df["gameId"] == game_id) & (tracking_df["playId"] == play_id)
    ].copy()

    if frame_data is None:
        frame_data = get_players_by_frame(play_df)

    # Initialize new columns
    if not relative_distances:
        play_df["wr_number"] = (
            np.nan
        )  # Which receiver they are (1-5 based on Y position)
    play_df["wr_center_distance"] = np.nan
    play_df["nearest_wr_distance"] = np.nan

    # Initialize distance columns
    for i in range(1, 6):
        play_df[f"dis_wr_{i}"] = np.nan

    # Process each frame
    for frame_id in play_df["frameId"].unique():
        if frame_id not in frame_data:
            continue

        analysis = analyze_receiver_positions(frame_data[frame_id], relative_distances)

        # Update values for receivers in this frame
        frame_mask = (play_df["frameId"] == frame_id) & (
            play_df["position"].isin(("WR", "TE", "RB"))
        )

        for nfl_id in play_df[frame_mask]["nflId"]:
            if nfl_id in analysis["distances"]:
                idx = play_df[
                    (play_df["frameId"] == frame_id) & (play_df["nflId"] == nfl_id)
                ].index

                # Set position number if using absolute Y positions
                if not relative_distances and nfl_id in analysis["wr_positions"]:
                    play_df.loc[idx, "wr_number"] = analysis["wr_positions"][nfl_id]

                # Set basic info
                play_df.loc[idx, "wr_center_distance"] = analysis["center_distances"][
                    nfl_id
                ]
                play_df.loc[idx, "nearest_wr_distance"] = analysis["nearest_distances"][
                    nfl_id
                ]

                # Set distances to other receivers
                for target_pos, distance in analysis["distances"][nfl_id].items():
                    play_df.loc[idx, f"dis_wr_{target_pos}"] = distance

    return play_df



import tqdm

# Example usage
game_id = 2022091105
play_id = 140

games = list(tracking_passing_only.gameId.unique())

enriched_df = None
tracking_subset = tracking_passing_only.copy()
tracking_subset["distance_from_los"] = np.abs(tracking_subset["x"] - tracking_subset["absoluteYardlineNumber"])
tracking_subset["distance_from_sideline"] = np.minimum(tracking_subset["y"], 53.3 - tracking_subset["y"])
pairs = set()
for item in list(tracking_subset[['gameId', 'playId']].to_numpy()):
    pairs.add((item[0].item(), item[1].item()))
i = 0
for game_id, play_id in list(pairs):
    i += 1
    if i % 500 == 0:
        print(i)
        print(enriched_df.shape)
    # Option 2: Use pre-computed frame data
    frame_data = get_players_by_frame(tracking_subset[(tracking_subset.gameId == game_id) & (tracking_subset.playId == play_id)])
    if enriched_df is None:
        enriched_df = join_receiver_analysis_to_df(tracking_subset, game_id, play_id, True, frame_data)
    else:
        try:
            tmp = join_receiver_analysis_to_df(tracking_subset, game_id, play_id, True, frame_data)
        except Exception as e:
            print(e)
            print(f'game, play: {game_id}, {play_id}')
        enriched_df = pd.concat([enriched_df, tmp])


import pickle

with open("/kaggle/input/compare-models-and-ensemble/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("/kaggle/input/compare-models-and-ensemble/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("/kaggle/input/compare-models-and-ensemble/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


metrics_xgb, (y_test, y_pred, y_pred_proba, predictions_df) = evaluate_route_predictions_table(
    xgb_model, 
    val_loader, 
    dataset, "cpu", 
    softmax=True,
    plays_df=enriched_df,
    feature_encoders=encoders, 
    scaler=scaler
)
metrics_xgb['display_tables']()


gnn_output_by_play = groupby_entropy_and_accuracy(metrics_gnn["predictions_df"], k=200)


# Get metrics for high confidence predictions
metrics = get_high_confidence_metrics(
    metrics_gnn["predictions_df"],
    unique_routes=unique_routes,
    confidence_threshold=0.5,
    k=3
)
print_metrics_summary(metrics, 3)


accuracy_report_by_entropy(metrics_gnn["predictions_df"], gnn_output_by_play, cutoff=2)


xgb_output_by_play = groupby_entropy_and_accuracy(metrics_xgb["predictions_df"], k=200)


# Get metrics for high confidence predictions
metrics = get_high_confidence_metrics(
    metrics_xgb["predictions_df"],
    unique_routes=unique_routes,
    confidence_threshold=0.50,
    k=3
)

# Print formatted summary
print_metrics_summary(metrics, 3)


accuracy_report_by_entropy(metrics_xgb["predictions_df"], xgb_output_by_play, cutoff=2)


xgb_preds = metrics_xgb["predictions_df"].rename(columns={"play_id": "playId", "game_id": "gameId"})
xgb_preds_df_with_team = xgb_preds.merge(tracking_passing_only[['gameId','playId','offenseTeam']], on=['playId', 'gameId'])

xgb_preds_df_agg_team = xgb_preds_df_with_team.groupby("offenseTeam").agg({
    'is_correct': ['mean', 'std'],  # replace with your actual column names
    'entropy': ['mean', 'std']
}).sort_values(('is_correct', 'mean'), ascending=True)


xgb_per_play_df = xgb_output_by_play['play_accuracy'].reset_index().rename(columns={"play_id": "playId", "game_id": "gameId"})
xgb_per_play_df_with_team = xgb_per_play_df.merge(tracking_passing_only[['gameId','playId','offenseTeam']], on=['playId', 'gameId'])

# For entropy sorting (descending order)
xgb_play_preds_df_agg_team = xgb_per_play_df_with_team.groupby("offenseTeam").agg({
    'entropy': ['mean', 'std'],  # replace with your actual column names
    'accuracy': ['mean', 'std']
}).sort_values(('accuracy', 'mean'), ascending=False)


import matplotlib.pyplot as plt
import seaborn as sns


def plot_accuracy(per_play_df_with_team):
    # Create the grouped data
    team_stats = per_play_df_with_team.groupby("offenseTeam").agg({
        'entropy': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names for easier access
    team_stats.columns = ['offenseTeam', 'entropy_mean', 'entropy_std', 'accuracy_mean', 'accuracy_std']
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create the main scatter plot
    plt.scatter(
        team_stats['entropy_mean'], 
        team_stats['accuracy_mean'], 
        alpha=0.6
    )

    # Add team labels to each point
    for idx, row in team_stats.iterrows():
        plt.annotate(
            row['offenseTeam'], 
            (row['entropy_mean'], row['accuracy_mean']),
            xytext=(5, 5), 
            textcoords='offset points'
        )
    
    # # Add error bars
    # plt.errorbar(
    #     team_stats['entropy_mean'],
    #     team_stats['accuracy_mean'],
    #     xerr=team_stats['entropy_std'],
    #     yerr=team_stats['accuracy_std'],
    #     fmt='none',
    #     alpha=0.3
    # )
    
    # Customize the plot
    plt.title('Team Entropy vs Accuracy')
    plt.xlabel('Mean Entropy')
    plt.ylabel('Mean Accuracy')
    
    # Add a trend line
    sns.regplot(
        x=team_stats['entropy_mean'],
        y=team_stats['accuracy_mean'],
        scatter=False,
        color='red'
    )
    
    # Calculate correlation coefficient
    correlation = team_stats['entropy_mean'].corr(team_stats['accuracy_mean'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()


plot_accuracy(xgb_per_play_df_with_team)


gnn_preds = metrics_gnn["predictions_df"].rename(columns={"play_id": "playId", "game_id": "gameId"})
gnn_preds_df_with_team = gnn_preds.merge(tracking_passing_only[['gameId','playId','offenseTeam']], on=['playId', 'gameId'])

gnn_preds_df_agg_team = gnn_preds_df_with_team.groupby("offenseTeam").agg({
    'is_correct': ['mean', 'std'],  # replace with your actual column names
    'entropy': ['mean', 'std']
}).sort_values(('is_correct', 'mean'), ascending=True)


gnn_per_play_df = gnn_output_by_play['play_accuracy'].reset_index().rename(columns={"play_id": "playId", "game_id": "gameId"})
gnn_per_play_df_with_team = gnn_per_play_df.merge(tracking_passing_only[['gameId','playId','offenseTeam']], on=['playId', 'gameId'])

# For entropy sorting (descending order)
gnn_play_preds_df_agg_team = gnn_per_play_df_with_team.groupby("offenseTeam").agg({
    'entropy': ['mean', 'std'],  # replace with your actual column names
    'accuracy': ['mean', 'std']
}).sort_values(('accuracy', 'mean'), ascending=False)


plot_accuracy(gnn_per_play_df_with_team)


# xgb_most_acc = xgb_output_by_play['most_accurate_plays'].reset_index().rename(columns={"play_id": "playId", "game_id": "gameId"})
# xgb_preds = metrics_xgb["predictions_df"].rename(columns={"play_id": "playId", "game_id": "gameId"})
# xgb_preds_df_with_team = xgb_preds.merge(tracking_passing_only[['gameId','playId','offenseTeam']], on=['playId', 'gameId'], suffixes=('', '_play'))
# xgb_preds_df_with_team = xgb_preds_df_with_team.merge(xgb_most_acc, on=['playId', 'gameId'], suffixes=('', '_play'))
# display(xgb_preds_df_with_team)
# xgb_preds_high_certainty_df_agg_team = xgb_preds_df_with_team.groupby("offenseTeam").agg({
#     'is_correct': ['mean', 'std'],  # replace with your actual column names
#     'entropy': ['mean', 'std']
# }).sort_values(('is_correct', 'mean'), ascending=True)


import seaborn as sns

# Use the same data preparation as before
xgb_data = xgb_preds_df_agg_team['is_correct']['mean'].reset_index()
xgb_data = xgb_data.rename(columns={'mean': 'accuracy'})
xgb_data['model'] = 'XGB'

gnn_data = gnn_preds_df_agg_team['is_correct']['mean'].reset_index()
gnn_data = gnn_data.rename(columns={'mean': 'accuracy'})
gnn_data['model'] = 'GNN'

# Combine the dataframes
combined_data = pd.concat([xgb_data, gnn_data])

# Calculate mean accuracy across models and sort
mean_accuracy = combined_data.groupby('offenseTeam')['accuracy'].mean().sort_values(ascending=False)
sorted_teams = mean_accuracy.index

# Set the style and figure size
plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")

# Create the grouped bar plot
ax = sns.barplot(
    data=combined_data,
    x='offenseTeam',
    y='accuracy',
    hue='model',
    order=sorted_teams,
    palette='Set2'
)

# Customize the plot
plt.title('Model Accuracy Comparison by Team', pad=20, size=14)
plt.xlabel('Team', size=12)
plt.ylabel('Accuracy', size=12)
plt.xticks(rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Print mean accuracies for reference
print("\nMean Accuracies:")
print("XGB:", xgb_data['accuracy'].mean())
print("GNN:", gnn_data['accuracy'].mean())

