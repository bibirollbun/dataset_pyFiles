# первичгый анализб , где внизу последовательно подбирались grid_search
https://www.kaggle.com/code/ctrnmbr/mfdp-antifraud-baseline/edit/run/244647036#grid_search-модель---ее-метрики



# ниже скрины предыдущих метрик - там были gridsearch с по бустингам и деревьям с использованием feature_importances от деревьев/бустингов/анализа корреляция/eda / и корреляции Спирмэна


# First install compatible versions # только так сработало
!pip install --upgrade scikit-learn==1.2.2 imbalanced-learn==0.10.1



!pip install --upgrade scikit-learn==1.0.2 imbalanced-learn==0.9.0


# itog





!pip install  numpy pandas torch torch-geometric scikit-learn category-encoders imbalanced-learn cesium 


# First install compatible versions # только так сработало
!pip install --upgrade scikit-learn==1.2.2 imbalanced-learn==0.10.1






import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.nn import Linear as PyGLinear
import gc
import warnings
warnings.filterwarnings('ignore')

class EntityGraphBuilder:
    def __init__(self, entity_types):
        self.entity_types = entity_types
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_counters = {et: 0 for et in entity_types}
        
        self.transaction_data = []
        self.transaction_counter = 0
        
        self.edges = {et: ([], []) for et in entity_types}
        self.entity_frequencies = {et: {} for et in entity_types}
        self.global_max_dt = 0
        self.train_ratio = 0.8
    
    def add_batch(self, batch_df):
        """Add a batch of transactions to the graph"""
        self.transaction_data.append(batch_df)
        
        batch_max_dt = batch_df['TransactionDT'].max()
        if batch_max_dt > self.global_max_dt:
            self.global_max_dt = batch_max_dt
        
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        for et in self.entity_types:
            if et == 'card':
                card1 = batch_df['card1'].fillna('').astype(str).values
                card2 = batch_df['card2'].fillna('').astype(str).values
                card3 = batch_df['card3'].fillna('').astype(str).values
                entity_values = np.array([f"{c1}_{c2}_{c3}" for c1, c2, c3 in zip(card1, card2, card3)])
            elif et == 'addr':
                addr1 = batch_df['addr1'].fillna('').astype(str).values
                addr2 = batch_df['addr2'].fillna('').astype(str).values
                entity_values = np.array([f"{a1}_{a2}" for a1, a2 in zip(addr1, addr2)])
            elif et == 'email':
                entity_values = batch_df['P_emaildomain'].fillna('').astype(str).values
            elif et == 'device':
                entity_values = batch_df['DeviceInfo'].fillna('').astype(str).values
            else:
                continue
            
            valid_mask = (entity_values != '') & (entity_values != '_') & (entity_values != '__')
            entity_values = entity_values[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            for entity_value, tx_idx in zip(entity_values, batch_tx_indices):
                if entity_value not in self.entity_maps[et]:
                    self.entity_maps[et][entity_value] = self.entity_counters[et]
                    self.entity_counters[et] += 1
                    self.entity_frequencies[et][entity_value] = []
                self.entity_frequencies[et][entity_value].append(tx_idx)
            
            global_entity_indices = [self.entity_maps[et][v] for v in entity_values]
            
            self.edges[et][0].extend(batch_tx_indices)
            self.edges[et][1].extend(global_entity_indices)
    
    def build_graph(self):
        """Build the heterogeneous graph with train/validation split"""
        if self.transaction_counter == 0:
            print("No transactions to build graph")
            return HeteroData()
            
        full_df = pd.concat(self.transaction_data, ignore_index=True)
        print(f"Building graph with {len(full_df)} transactions")
        
        num_train = int(len(full_df) * self.train_ratio)
        train_mask = torch.zeros(len(full_df), dtype=torch.bool)
        val_mask = torch.zeros(len(full_df), dtype=torch.bool)
        train_mask[:num_train] = True
        val_mask[num_train:] = True
        
        transaction_features = []
        for _, row in full_df.iterrows():
            features = create_features(row)
            transaction_features.append(features)
        
        scaler = StandardScaler()
        transaction_features = scaler.fit_transform(np.array(transaction_features))
        tx_feature_tensor = torch.tensor(transaction_features, dtype=torch.float32)
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        
        transaction_labels = full_df['isFraud'].values
        tx_label_tensor = torch.tensor(transaction_labels, dtype=torch.float32)
        data['transaction'].y = tx_label_tensor
        print(f"Added transaction labels: {tx_label_tensor.shape}")
        
        for et in self.entity_types:
            num_entities = self.entity_counters[et]
            if num_entities == 0:
                data[et].x = torch.zeros((0, tx_feature_tensor.shape[1]), dtype=torch.float32)
                data['transaction', f'to_{et}', et].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[et, f'from_{et}', 'transaction'].edge_index = torch.empty((2, 0), dtype=torch.long)
                continue
            
            entity_features = np.zeros((num_entities, tx_feature_tensor.shape[1]))
            
            reverse_map = {idx: entity for entity, idx in self.entity_maps[et].items()}
            
            for i in range(num_entities):
                entity_value = reverse_map.get(i)
                if entity_value and entity_value in self.entity_frequencies[et]:
                    tx_indices = self.entity_frequencies[et][entity_value]
                    train_indices = [idx for idx in tx_indices if idx < num_train]
                    if train_indices:
                        entity_features[i] = tx_feature_tensor[train_indices].mean(axis=0).numpy()
            
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            tx_indices = np.array(self.edges[et][0])
            entity_indices = np.array(self.edges[et][1])
            
            if len(tx_indices) > 0:
                edge_index_forward = torch.tensor([tx_indices, entity_indices], dtype=torch.long)
                data['transaction', f'to_{et}', et].edge_index = edge_index_forward
                
                edge_index_reverse = torch.tensor([entity_indices, tx_indices], dtype=torch.long)
                data[et, f'from_{et}', 'transaction'].edge_index = edge_index_reverse
        
        self.transaction_data = []
        self.edges = {et: ([], []) for et in self.entity_types}
        gc.collect()
        
        return data

def create_features(row):
    """Create features for a transaction row with essential features only"""
    features = []
    
    num_features = [
        'TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
        'C11', 'C12', 'C13', 'C14'
    ]
    
    for f in num_features:
        val = row.get(f, 0.0)
        if pd.isna(val):
            val = 0.0
        features.append(val)
    
    transaction_amt = row.get('TransactionAmt', 0.0)
    if pd.isna(transaction_amt):
        transaction_amt = 0.0
    features.append(np.log1p(transaction_amt) if transaction_amt > 0 else 0.0)
    
    transaction_dt = row.get('TransactionDT', 0)
    if pd.isna(transaction_dt):
        transaction_dt = 0
    if transaction_dt != 0:
        hour = (transaction_dt % 86400) // 3600
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
    else:
        features.extend([0.0, 0.0])
    
    feature_array = np.array(features, dtype=np.float32)
    if np.isnan(feature_array).any():
        feature_array = np.nan_to_num(feature_array, nan=0.0)
    return feature_array

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.entity_types = ['card', 'addr', 'email', 'device']
        
        self.lin_transaction = PyGLinear(-1, hidden_channels)
        self.lin_entity = PyGLinear(-1, hidden_channels)
        
        for _ in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                conv_dict[('transaction', f'to_{et}', et)] = SAGEConv(
                    hidden_channels, hidden_channels)
                conv_dict[(et, f'from_{et}', 'transaction')] = SAGEConv(
                    hidden_channels, hidden_channels)
            
            conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(conv)
        
        self.classifier = PyGLinear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.3)
    
    def forward(self, x_dict, edge_index_dict):
        projected_dict = {}
        
        if 'transaction' in x_dict and x_dict['transaction'] is not None:
            projected_dict['transaction'] = F.leaky_relu(self.lin_transaction(x_dict['transaction']))
        
        for et in self.entity_types:
            if et in x_dict and x_dict[et] is not None and x_dict[et].size(0) > 0:
                projected_dict[et] = F.leaky_relu(self.lin_entity(x_dict[et]))
            else:
                projected_dict[et] = torch.zeros(0, self.lin_entity.out_features, 
                                               device=self.lin_entity.weight.device)
        
        for conv in self.convs:
            # Ensure all required node types are in projected_dict
            for key in ['transaction'] + self.entity_types:
                if key not in projected_dict:
                    projected_dict[key] = torch.zeros(0, self.lin_entity.out_features, 
                                                    device=self.lin_entity.weight.device)
            
            x_dict_out = conv(projected_dict, edge_index_dict)
            
            for key in x_dict_out:
                x_dict_out[key] = F.leaky_relu(x_dict_out[key])
                x_dict_out[key] = self.dropout(x_dict_out[key])
            
            projected_dict = x_dict_out
        
        if 'transaction' in projected_dict and projected_dict['transaction'].size(0) > 0:
            return self.classifier(projected_dict['transaction']).squeeze()
        else:
            return torch.zeros(x_dict['transaction'].size(0), device=x_dict['transaction'].device)

def train_gnn_model(transaction_path, identity_path):
    # Configuration
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device']
    
    try:
        identity_df = pd.read_csv(identity_path)
        print(f"Loaded identity data with {len(identity_df)} rows")
        identity_df = identity_df.set_index('TransactionID')
    except Exception as e:
        print(f"Error loading identity data: {e}, proceeding without it")
        identity_df = pd.DataFrame()
    
    graph_builder = EntityGraphBuilder(ENTITY_TYPES)
    
    total_transactions = 0
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    
    for chunk_idx, chunk in enumerate(chunk_iterator):
        total_transactions += len(chunk)
        print(f"Processing chunk {chunk_idx+1} with {len(chunk):,} transactions (Total: {total_transactions:,})")
        
        chunk = chunk.reset_index(drop=True)
        
        # Merge with identity data if available
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        else:
            for col in ['card1', 'card2', 'card3', 'addr1', 'addr2', 'P_emaildomain', 'DeviceInfo']:
                if col not in chunk:
                    chunk[col] = ''
        
        required_columns = ['isFraud', 'TransactionDT', 'TransactionAmt'] + \
                          [f'C{i}' for i in range(1, 15)] + \
                          ['P_emaildomain', 'DeviceInfo', 'card1', 'card2', 'card3', 'addr1', 'addr2']
        
        for col in required_columns:
            if col not in chunk:
                if col in ['isFraud', 'TransactionAmt'] or col.startswith('C'):
                    chunk[col] = 0.0
                else:
                    chunk[col] = ''
        
        batch_df = chunk[required_columns].copy()
        graph_builder.add_batch(batch_df)
    
    print("Building final graph...")
    try:
        graph_data = graph_builder.build_graph()
        print("Graph built successfully!")
    except Exception as e:
        import traceback
        print(f"Error building graph: {e}")
        print(traceback.format_exc())
        return None, 0.0
    
    print("\nGraph structure summary:")
    if 'transaction' in graph_data.node_types:
        num_tx_nodes = graph_data['transaction'].num_nodes
        print(f"Transaction nodes: {num_tx_nodes}")
        print(f"Transaction labels shape: {graph_data['transaction'].y.shape}")
    else:
        print("No transaction nodes in graph")
        return None, 0.0
        
    for et in ENTITY_TYPES:
        if et in graph_data.node_types:
            print(f"{et} nodes: {graph_data[et].num_nodes}")
            edge_type = ('transaction', f'to_{et}', et)
            if edge_type in graph_data.edge_types:
                print(f"  Edges to {et}: {graph_data[edge_type].num_edges}")
    
    model = HeteroGNN(
        hidden_channels=64,
        out_channels=1,
        num_layers=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    graph_data = graph_data.to(device)
    
    edge_index_dict = {}
    for et in ENTITY_TYPES:
        edge_type = ('transaction', f'to_{et}', et)
        if edge_type in graph_data.edge_types:
            edge_index_dict[edge_type] = graph_data[edge_type].edge_index
        
        edge_type = (et, f'from_{et}', 'transaction')
        if edge_type in graph_data.edge_types:
            edge_index_dict[edge_type] = graph_data[edge_type].edge_index
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    train_mask = graph_data['transaction'].train_mask
    if train_mask.sum() > 0:
        train_y = graph_data['transaction'].y[train_mask].cpu().numpy()
        pos_count = max(train_y.sum(), 1)
        neg_count = len(train_y) - pos_count
        pos_weight = torch.tensor([neg_count / pos_count]).to(device)
        print(f"Positive weight: {pos_weight.item():.2f} (pos: {pos_count}, neg: {neg_count})")
    else:
        print("Warning: Using default pos_weight=1.0")
        pos_weight = torch.tensor([1.0]).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training loop
    model.train()
    best_val_pr_auc = 0
    
    for epoch in range(1000):
        optimizer.zero_grad()
        
        pred = model(graph_data.x_dict, edge_index_dict)
        
        loss = criterion(
            pred[graph_data['transaction'].train_mask],
            graph_data['transaction'].y[graph_data['transaction'].train_mask]
        )
        
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}. Skipping update.")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            val_pred = model(graph_data.x_dict, edge_index_dict)
            val_proba = torch.sigmoid(val_pred[graph_data['transaction'].val_mask]).cpu().numpy()
            val_actual = graph_data['transaction'].y[graph_data['transaction'].val_mask].cpu().numpy()
            
            if len(val_actual) > 0:
                if np.isnan(val_proba).any():
                    val_proba = np.nan_to_num(val_proba, nan=0.5)
                if np.isnan(val_actual).any():
                    val_actual = np.nan_to_num(val_actual, nan=0.0)
                
                if len(np.unique(val_actual)) > 1:
                    from sklearn.metrics import average_precision_score
                    val_pr_auc = average_precision_score(val_actual, val_proba)
                else:
                    val_pr_auc = 0.0
                
                if val_pr_auc > best_val_pr_auc:
                    best_val_pr_auc = val_pr_auc
                    torch.save(model.state_dict(), 'best_model.pt')
            else:
                val_pr_auc = 0.0
                print("Warning: No validation samples")
            
            model.train()
        
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Val PR-AUC={val_pr_auc:.4f} (Best: {best_val_pr_auc:.4f})")
    
    model.load_state_dict(torch.load('best_model.pt'))
    return model, best_val_pr_auc


TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"

# Train the model
model, pr_auc = train_gnn_model(TRANSACTION_PATH, IDENTITY_PATH)
print(f"\nFinal Validation PR-AUC: {pr_auc:.4f}")


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear as PyGLinear
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import gc
import warnings
from collections import defaultdict
import time
warnings.filterwarnings('ignore')





class EnhancedGraphBuilder:
    def __init__(self, entity_types):
        self.entity_types = entity_types
        self.transaction_counter = 0
        self.tx_times = []  
        
        
        self.tx_features = []
        self.tx_labels = []
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        self.tx_labels.append(batch_df['isFraud'].values)
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Enhanced feature creation"""
        features = []
        
        
        num_cols = ['TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
        
        
        for col in num_cols:
            if col in batch_df:
                
                median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))  
        features.append(np.where(amt > 0, 1, 0))              
        
        
        amt_mean = np.mean(amt)
        amt_std = np.std(amt) + 1e-7
        features.append((amt - amt_mean) / amt_std)
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        
        if 'ProductCD' in batch_df:
            prod_oh = pd.get_dummies(batch_df['ProductCD'], prefix='prod').values.astype(np.float32)
            features.extend([prod_oh[:, i] for i in range(prod_oh.shape[1])])
        else:
            
            features.extend([np.zeros(len(batch_df), dtype=np.float32)] * 5)
        
        return np.column_stack(features)
    
    def build_graph(self):
        print(f"Building graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        tx_features = np.vstack(self.tx_features)
        tx_labels = np.concatenate(self.tx_labels)
        
        
        scaler = StandardScaler()
        tx_features = scaler.fit_transform(tx_features)
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
        
        
        num_train = int(self.transaction_counter * self.train_ratio)
        train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        train_mask[:num_train] = True
        val_mask[num_train:] = True
        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 1), dtype=np.float32)
            for entity_idx, count in self.entity_counts[et].items():
                if entity_idx < num_entities:
                    entity_features[entity_idx, 0] = np.log1p(count)
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building temporal edges...")
        time_edges = []
        time_window = 300  
        transaction_count = self.transaction_counter
        
        
        sample_size = min(100000, transaction_count)  
        sample_indices = np.random.choice(transaction_count, sample_size, replace=False)
        sample_times = [self.tx_times[i] for i in sample_indices]
        
        
        sorted_indices = sorted(sample_indices, key=lambda i: self.tx_times[i])
        sorted_times = [self.tx_times[i] for i in sorted_indices]
        
        
        for i in range(1, len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_time = sorted_times[i]
            j = i - 1
            while j >= 0 and (current_time - sorted_times[j]) <= time_window:
                time_edges.append((current_idx, sorted_indices[j]))
                time_edges.append((sorted_indices[j], current_idx))
                j -= 1
        
        if time_edges:
            src, dst = zip(*time_edges)
            time_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
            print(f"Added {len(time_edges)} temporal edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class ResidualGNNLayer(nn.Module):
    """Residual layer that handles SAGEConv output correctly"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index):
        
        if edge_index.size(1) == 0:
            if isinstance(x, tuple):
                x_dst = x[1]
                return self.lin(x_dst)
            else:
                return self.lin(x)
        
        
        if isinstance(x, tuple):
            
            x_dst = x[1]
            conv_out = self.conv(x, edge_index)
            return self.norm(F.elu(conv_out) + self.lin(x_dst))
        else:
            return self.norm(F.elu(self.conv(x, edge_index)) + self.lin(x))

class EnhancedFraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.tx_skip = nn.Linear(tx_feature_size, hidden_channels)
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU()
        )
        self.entity_attn = nn.ModuleDict({
            et: nn.Linear(hidden_channels, 1) for et in self.entity_types
        })
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            
            for et in self.entity_types:
                if i == 0:  
                    conv_dict[('transaction', f'to_{et}', et)] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                    conv_dict[(et, f'from_{et}', 'transaction')] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                else:       
                    conv_dict[('transaction', f'to_{et}', et)] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
                    conv_dict[(et, f'from_{et}', 'transaction')] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
            
            
            if i == 0:
                
                conv_dict[('transaction', 'temporal', 'transaction')] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=True)
            else:
                conv_dict[('transaction', 'temporal', 'transaction')] = ResidualGNNLayer(
                    hidden_channels, hidden_channels
                )
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        
        tx_features = data['transaction'].x
        x_dict = {
            'transaction': F.elu(self.tx_proj(tx_features) + self.tx_skip(tx_features))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                entity_x = self.entity_proj(data[et].x)
                attn_weights = torch.sigmoid(self.entity_attn[et](entity_x))
                x_dict[et] = entity_x * attn_weights
            else:
                
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=tx_features.device)
        
        
        layer_outputs = []
        for i, conv in enumerate(self.convs):
            try:
                
                x_dict = conv(x_dict, data.edge_index_dict)
                
                
                x_dict = {k: F.elu(x) for k, x in x_dict.items()}
                
                
                if 'transaction' in x_dict:
                    layer_outputs.append(x_dict['transaction'])
            except Exception as e:
                print(f"Skipping convolution due to error: {e}")
                continue
        
        
        if layer_outputs:
            
            mean_layer = torch.mean(torch.stack(layer_outputs), dim=0)
            
            combined = torch.cat([x_dict['transaction'], mean_layer], dim=1)
        else:
            combined = x_dict['transaction']
        
        
        return self.head(combined).squeeze()





class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss





def evaluate(loader, model):
    """Evaluate model on validation set"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits)
            labels = batch['transaction'].y[:batch['transaction'].batch_size]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    if not all_probs:
        return 0.0
    
    val_probs = torch.cat(all_probs).numpy()
    val_labels = torch.cat(all_labels).numpy()
    
    if len(np.unique(val_labels)) > 1:
        return average_precision_score(val_labels, val_probs)
    return 0.0

def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = EnhancedGraphBuilder(ENTITY_TYPES)
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = ''
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = EnhancedFraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,  
        num_layers=3         
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,  
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    
    fraud_ratio = data['transaction'].y[data['transaction'].train_mask].mean().item()
    criterion = BalancedFocalLoss(alpha=1-fraud_ratio, gamma=2.0)
    print(f"Fraud ratio: {fraud_ratio:.4f}, Using alpha={1-fraud_ratio:.4f}")
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    
    best_pr_auc = 0
    no_improve = 0
    max_epochs = 30
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        val_pr_auc = evaluate(val_loader, model)
        scheduler.step(val_pr_auc)
        
        
        epoch_time = time.time() - start_time
        
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"Epoch {epoch}: Loss={total_loss/batch_count:.4f}, Val PR-AUC={val_pr_auc:.4f}* (Best) [{epoch_time:.1f}s]")
        else:
            no_improve += 1
            print(f"Epoch {epoch}: Loss={total_loss/batch_count:.4f}, Val PR-AUC={val_pr_auc:.4f} [{epoch_time:.1f}s]")
            
            if no_improve >= 5:
                print(f"Early stopping at epoch {epoch}")
                break
    
    
    model.load_state_dict(torch.load('best_model.pt'))
    final_pr_auc = evaluate(val_loader, model)
    return model, final_pr_auc





if __name__ == "__main__":
    TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    
    model, pr_auc = train_fraud_model(TRANSACTION_PATH, IDENTITY_PATH)
    print(f"\nFinal Validation PR-AUC: {pr_auc:.4f}")


# выше 0.35 за 20 эпох


#numeric
'TransactionDT','TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41','V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59','V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77','V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95','V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126','V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141','V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156','V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171','V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186','V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201','V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216','V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231','V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246','V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261','V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276','V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291','V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306','V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321','V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336','V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'



import torch
!pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git
!pip install  numpy pandas torch torch-geometric scikit-learn category-encoders imbalanced-learn cesium 
!pip install --upgrade scikit-learn==1.2.2 imbalanced-learn==0.10.1




import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear as PyGLinear
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import gc
import warnings
from collections import defaultdict
import time
warnings.filterwarnings('ignore')





class EnhancedGraphBuilder:
    def __init__(self, entity_types):
        self.entity_types = entity_types
        self.transaction_counter = 0
        self.tx_times = []  
        
        
        self.tx_features = []
        self.tx_labels = []
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        self.tx_labels.append(batch_df['isFraud'].values)
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        #elif et == 'id':
        #    c1 = batch_df['card1'].fillna('').astype(str).values
        #    c2 = batch_df['card2'].fillna('').astype(str).values
        #    c3 = batch_df['card3'].fillna('').astype(str).values
        #    c4 = batch_df['card4'].fillna('').astype(str).values
        #    c5 = batch_df['card5'].fillna('').astype(str).values
        #    c6 = batch_df['card6'].fillna('').astype(str).values
        #    return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6    
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Enhanced feature creation"""
        features = []
        
        
        num_cols = [#'TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
                   #'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10'
#'TransactionDT',
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'

                   ]
        
        
        for col in num_cols:
            if col in batch_df:
                
                median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))  
        features.append(np.where(amt > 0, 1, 0))              
        
        
        amt_mean = np.mean(amt)
        amt_std = np.std(amt) + 1e-7
        features.append((amt - amt_mean) / amt_std)
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        
        if 'ProductCD' in batch_df:
            prod_oh = pd.get_dummies(batch_df['ProductCD'], prefix='prod').values.astype(np.float32)
            features.extend([prod_oh[:, i] for i in range(prod_oh.shape[1])])
        else:
            
            features.extend([np.zeros(len(batch_df), dtype=np.float32)] * 5)
        
        return np.column_stack(features)
    
    def build_graph(self):
        print(f"Building graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        tx_features = np.vstack(self.tx_features)
        tx_labels = np.concatenate(self.tx_labels)
        
        
        scaler = StandardScaler()
        tx_features = scaler.fit_transform(tx_features)
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
        
        
        num_train = int(self.transaction_counter * self.train_ratio)
        train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        train_mask[:num_train] = True
        val_mask[num_train:] = True
        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 1), dtype=np.float32)
            for entity_idx, count in self.entity_counts[et].items():
                if entity_idx < num_entities:
                    entity_features[entity_idx, 0] = np.log1p(count)
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building temporal edges...")
        time_edges = []
        time_window = 300  
        transaction_count = self.transaction_counter
        
        
        sample_size = min(100000, transaction_count)  
        sample_indices = np.random.choice(transaction_count, sample_size, replace=False)
        sample_times = [self.tx_times[i] for i in sample_indices]
        
        
        sorted_indices = sorted(sample_indices, key=lambda i: self.tx_times[i])
        sorted_times = [self.tx_times[i] for i in sorted_indices]
        
        
        for i in range(1, len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_time = sorted_times[i]
            j = i - 1
            while j >= 0 and (current_time - sorted_times[j]) <= time_window:
                time_edges.append((current_idx, sorted_indices[j]))
                time_edges.append((sorted_indices[j], current_idx))
                j -= 1
        
        if time_edges:
            src, dst = zip(*time_edges)
            time_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
            print(f"Added {len(time_edges)} temporal edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class ResidualGNNLayer(nn.Module):
    """Residual layer that handles SAGEConv output correctly"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index):
        
        if edge_index.size(1) == 0:
            if isinstance(x, tuple):
                x_dst = x[1]
                return self.lin(x_dst)
            else:
                return self.lin(x)
        
        
        if isinstance(x, tuple):
            
            x_dst = x[1]
            conv_out = self.conv(x, edge_index)
            return self.norm(F.elu(conv_out) + self.lin(x_dst))
        else:
            return self.norm(F.elu(self.conv(x, edge_index)) + self.lin(x))

class EnhancedFraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.tx_skip = nn.Linear(tx_feature_size, hidden_channels)
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU()
        )
        self.entity_attn = nn.ModuleDict({
            et: nn.Linear(hidden_channels, 1) for et in self.entity_types
        })
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            
            for et in self.entity_types:
                if i == 0:  
                    conv_dict[('transaction', f'to_{et}', et)] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                    conv_dict[(et, f'from_{et}', 'transaction')] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                else:       
                    conv_dict[('transaction', f'to_{et}', et)] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
                    conv_dict[(et, f'from_{et}', 'transaction')] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
            
            
            if i == 0:
                
                conv_dict[('transaction', 'temporal', 'transaction')] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=True)
            else:
                conv_dict[('transaction', 'temporal', 'transaction')] = ResidualGNNLayer(
                    hidden_channels, hidden_channels
                )
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        
        tx_features = data['transaction'].x
        x_dict = {
            'transaction': F.elu(self.tx_proj(tx_features) + self.tx_skip(tx_features))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                entity_x = self.entity_proj(data[et].x)
                attn_weights = torch.sigmoid(self.entity_attn[et](entity_x))
                x_dict[et] = entity_x * attn_weights
            else:
                
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=tx_features.device)
        
        
        layer_outputs = []
        for i, conv in enumerate(self.convs):
            try:
                
                x_dict = conv(x_dict, data.edge_index_dict)
                
                
                x_dict = {k: F.elu(x) for k, x in x_dict.items()}
                
                
                if 'transaction' in x_dict:
                    layer_outputs.append(x_dict['transaction'])
            except Exception as e:
                print(f"Skipping convolution due to error: {e}")
                continue
        
        
        if layer_outputs:
            
            mean_layer = torch.mean(torch.stack(layer_outputs), dim=0)
            
            combined = torch.cat([x_dict['transaction'], mean_layer], dim=1)
        else:
            combined = x_dict['transaction']
        
        
        return self.head(combined).squeeze()





class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss





def evaluate(loader, model):
    """Evaluate model on validation set"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits)
            labels = batch['transaction'].y[:batch['transaction'].batch_size]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    if not all_probs:
        return 0.0
    
    val_probs = torch.cat(all_probs).numpy()
    val_labels = torch.cat(all_labels).numpy()
    
    if len(np.unique(val_labels)) > 1:
        return average_precision_score(val_labels, val_probs)
    return 0.0

def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = EnhancedGraphBuilder(ENTITY_TYPES)
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = ''
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = EnhancedFraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,  
        num_layers=3         
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,  
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    
    fraud_ratio = data['transaction'].y[data['transaction'].train_mask].mean().item()
    criterion = BalancedFocalLoss(alpha=1-fraud_ratio, gamma=2.0)
    print(f"Fraud ratio: {fraud_ratio:.4f}, Using alpha={1-fraud_ratio:.4f}")
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    
    best_pr_auc = 0
    no_improve = 0
    max_epochs = 30
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        val_pr_auc = evaluate(val_loader, model)
        scheduler.step(val_pr_auc)
        
        
        epoch_time = time.time() - start_time
        
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model_4.pt')
            print(f"Epoch {epoch}: Loss={total_loss/batch_count:.4f}, Val PR-AUC={val_pr_auc:.4f}* (Best) [{epoch_time:.1f}s]")
        else:
            no_improve += 1
            print(f"Epoch {epoch}: Loss={total_loss/batch_count:.4f}, Val PR-AUC={val_pr_auc:.4f} [{epoch_time:.1f}s]")
            
            if no_improve >= 5:
                print(f"Early stopping at epoch {epoch}")
                break
    
    
    model.load_state_dict(torch.load('best_model_4.pt'))
    final_pr_auc = evaluate(val_loader, model)
    return model, final_pr_auc





if __name__ == "__main__":
    TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    
    model, pr_auc = train_fraud_model(TRANSACTION_PATH, IDENTITY_PATH)
    print(f"\nFinal Validation PR-AUC: {pr_auc:.4f}")


# обучается еще быстрее но качество не прям чтобы растет


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import gc
import warnings
from collections import defaultdict
import time
from category_encoders import TargetEncoder
warnings.filterwarnings('ignore')





class TargetEncodingGraphBuilder:
    def __init__(self, entity_types, cat_features):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.tx_times = []
        
        
        self.tx_features = []
        self.tx_labels = []
        self.tx_categorical = {col: [] for col in cat_features}
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8
        self.target_encoders = {}

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        self.tx_labels.append(batch_df['isFraud'].values)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation without categoricals (handled separately)"""
        features = []
        
        
        num_cols = [
                   
            
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
                   ]
        
        for col in num_cols:
            if col in batch_df:
                median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding to categorical features"""
        print("Applying target encoding...")
        start_time = time.time()
        
        
        tx_labels = np.concatenate(self.tx_labels)
        cat_data = {}
        for col in self.cat_features:
            cat_data[col] = np.concatenate(self.tx_categorical[col])
        
        
        num_train = int(self.transaction_counter * self.train_ratio)
        train_mask = np.zeros(self.transaction_counter, dtype=bool)
        train_mask[:num_train] = True
        
        
        encoded_features = []
        for col in self.cat_features:
            
            encoder = TargetEncoder(smoothing=10, min_samples_leaf=20)
            encoder.fit(
                cat_data[col][train_mask], 
                tx_labels[train_mask]
            )
            self.target_encoders[col] = encoder
            
            
            encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
            encoded_features.append(encoded)
        
        print(f"Target encoding completed in {time.time()-start_time:.1f} seconds")
        return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        tx_labels = np.concatenate(self.tx_labels)
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        scaler = StandardScaler()
        tx_features = scaler.fit_transform(tx_features)
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
        
        
        num_train = int(self.transaction_counter * self.train_ratio)
        train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        train_mask[:num_train] = True
        val_mask[num_train:] = True
        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 1), dtype=np.float32)
            for entity_idx, count in self.entity_counts[et].items():
                if entity_idx < num_entities:
                    entity_features[entity_idx, 0] = np.log1p(count)
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building temporal edges...")
        time_edges = []
        time_window = 300
        transaction_count = self.transaction_counter
        
        
        sample_size = min(100000, transaction_count)
        sample_indices = np.random.choice(transaction_count, sample_size, replace=False)
        sample_times = [self.tx_times[i] for i in sample_indices]
        
        
        sorted_indices = sorted(sample_indices, key=lambda i: self.tx_times[i])
        sorted_times = [self.tx_times[i] for i in sorted_indices]
        
        
        for i in range(1, len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_time = sorted_times[i]
            j = i - 1
            while j >= 0 and (current_time - sorted_times[j]) <= time_window:
                time_edges.append((current_idx, sorted_indices[j]))
                time_edges.append((sorted_indices[j], current_idx))
                j -= 1
        
        if time_edges:
            src, dst = zip(*time_edges)
            time_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
            print(f"Added {len(time_edges)} temporal edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times, self.tx_categorical
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class ResidualGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            if isinstance(x, tuple):
                return self.lin(x[1])
            return self.lin(x)
        
        if isinstance(x, tuple):
            x_dst = x[1]
            conv_out = self.conv(x, edge_index)
            return self.norm(F.elu(conv_out) + self.lin(x_dst))
        return self.norm(F.elu(self.conv(x, edge_index)) + self.lin(x))

class EnhancedFraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.tx_skip = nn.Linear(tx_feature_size, hidden_channels)
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU()
        )
        self.entity_attn = nn.ModuleDict({
            et: nn.Linear(hidden_channels, 1) for et in self.entity_types
        })
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                if i == 0:
                    conv_dict[('transaction', f'to_{et}', et)] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                    conv_dict[(et, f'from_{et}', 'transaction')] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                else:
                    conv_dict[('transaction', f'to_{et}', et)] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
                    conv_dict[(et, f'from_{et}', 'transaction')] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
            
            
            if i == 0:
                conv_dict[('transaction', 'temporal', 'transaction')] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True)
            else:
                conv_dict[('transaction', 'temporal', 'transaction')] = ResidualGNNLayer(
                    hidden_channels, hidden_channels
                )
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        
        tx_features = data['transaction'].x
        x_dict = {
            'transaction': F.elu(self.tx_proj(tx_features) + self.tx_skip(tx_features))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                entity_x = self.entity_proj(data[et].x)
                attn_weights = torch.sigmoid(self.entity_attn[et](entity_x))
                x_dict[et] = entity_x * attn_weights
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=tx_features.device)
        
        
        layer_outputs = []
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(x) for k, x in x_dict.items()}
                layer_outputs.append(x_dict['transaction'])
            except Exception as e:
                print(f"Skipping convolution: {e}")
                continue
        
        
        if layer_outputs:
            mean_layer = torch.mean(torch.stack(layer_outputs), dim=0)
            combined = torch.cat([x_dict['transaction'], mean_layer], dim=1)
        else:
            combined = x_dict['transaction']
        
        return self.head(combined).squeeze()





class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss





def evaluate(loader, model):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits)
            labels = batch['transaction'].y[:batch['transaction'].batch_size]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    if not all_probs:
        return 0.0
    
    val_probs = torch.cat(all_probs).numpy()
    val_labels = torch.cat(all_labels).numpy()
    
    if len(np.unique(val_labels)) > 1:
        return average_precision_score(val_labels, val_probs)
    return 0.0

def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = TargetEncodingGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = EnhancedFraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=3
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    
    fraud_ratio = data['transaction'].y[data['transaction'].train_mask].mean().item()
    criterion = BalancedFocalLoss(alpha=1-fraud_ratio, gamma=2.0)
    print(f"Fraud ratio: {fraud_ratio:.4f}, Using alpha={1-fraud_ratio:.4f}")
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    
    best_pr_auc = 0
    no_improve = 0
    max_epochs = 30
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        val_pr_auc = evaluate(val_loader, model)
        scheduler.step(val_pr_auc)
        
        
        epoch_time = time.time() - start_time
        
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model5.pt')
            print(f"Epoch {epoch}: Loss={total_loss/batch_count:.4f}, Val PR-AUC={val_pr_auc:.4f}* (Best) [{epoch_time:.1f}s]")
        else:
            no_improve += 1
            print(f"Epoch {epoch}: Loss={total_loss/batch_count:.4f}, Val PR-AUC={val_pr_auc:.4f} [{epoch_time:.1f}s]")
            
            if no_improve >= 5:
                print(f"Early stopping at epoch {epoch}")
                break
    
    
    model.load_state_dict(torch.load('best_model5.pt'))
    final_pr_auc = evaluate(val_loader, model)
    return model, final_pr_auc





if __name__ == "__main__":
    TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    
    model, pr_auc = train_fraud_model(TRANSACTION_PATH, IDENTITY_PATH)
    print(f"\nFinal Validation PR-AUC: {pr_auc:.4f}")





import torch
import gc
gc.collect()
torch.cuda.empty_cache()





import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear as PyGLinear
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import gc
import warnings
from collections import defaultdict
import time
from category_encoders import TargetEncoder
warnings.filterwarnings('ignore')





class TargetEncodingGraphBuilder:
    def __init__(self, entity_types, cat_features):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.tx_times = []
        
        
        self.tx_features = []
        self.tx_labels = []
        self.tx_categorical = {col: [] for col in cat_features}
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8
        self.target_encoders = {}

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        self.tx_labels.append(batch_df['isFraud'].values)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation without categoricals (handled separately)"""
        features = []
        
        
        num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        for col in num_cols:
            if col in batch_df:
                median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding to categorical features"""
        print("Applying target encoding...")
        start_time = time.time()
        
        
        tx_labels = np.concatenate(self.tx_labels)
        cat_data = {}
        for col in self.cat_features:
            cat_data[col] = np.concatenate(self.tx_categorical[col])
        
        
        num_train = int(self.transaction_counter * self.train_ratio)
        train_mask = np.zeros(self.transaction_counter, dtype=bool)
        train_mask[:num_train] = True
        
        
        encoded_features = []
        for col in self.cat_features:
            
            encoder = TargetEncoder(smoothing=10, min_samples_leaf=20)
            encoder.fit(
                cat_data[col][train_mask], 
                tx_labels[train_mask]
            )
            self.target_encoders[col] = encoder
            
            
            encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
            encoded_features.append(encoded)
        
        print(f"Target encoding completed in {time.time()-start_time:.1f} seconds")
        return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        tx_labels = np.concatenate(self.tx_labels)
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        scaler = StandardScaler()
        tx_features = scaler.fit_transform(tx_features)
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
        
        
        num_train = int(self.transaction_counter * self.train_ratio)
        train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        train_mask[:num_train] = True
        val_mask[num_train:] = True
        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 1), dtype=np.float32)
            for entity_idx, count in self.entity_counts[et].items():
                if entity_idx < num_entities:
                    entity_features[entity_idx, 0] = np.log1p(count)
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building temporal edges...")
        time_edges = []
        time_window = 600  
        transaction_count = self.transaction_counter
        
        
        sample_size = min(150000, transaction_count)  
        sample_indices = np.random.choice(transaction_count, sample_size, replace=False)
        sample_times = [self.tx_times[i] for i in sample_indices]
        
        
        sorted_indices = sorted(sample_indices, key=lambda i: self.tx_times[i])
        sorted_times = [self.tx_times[i] for i in sorted_indices]
        
        
        for i in range(1, len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_time = sorted_times[i]
            j = i - 1
            while j >= 0 and (current_time - sorted_times[j]) <= time_window:
                time_edges.append((current_idx, sorted_indices[j]))
                time_edges.append((sorted_indices[j], current_idx))
                j -= 1
        
        if time_edges:
            src, dst = zip(*time_edges)
            time_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
            print(f"Added {len(time_edges)} temporal edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times, self.tx_categorical
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class ResidualGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            if isinstance(x, tuple):
                return self.lin(x[1])
            return self.lin(x)
        
        if isinstance(x, tuple):
            x_dst = x[1]
            conv_out = self.conv(x, edge_index)
            return self.norm(F.elu(conv_out) + self.lin(x_dst))
        return self.norm(F.elu(self.conv(x, edge_index)) + self.lin(x))

class AdvancedFraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.gat_out_channels = hidden_channels // 4 * 8  
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels * 2),
            nn.ELU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ELU()
        )
        self.tx_skip = nn.Linear(tx_feature_size, hidden_channels)
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ELU()
        )
        self.entity_attn = nn.ModuleDict({
            et: nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ELU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            ) for et in self.entity_types
        })
        
        
        self.gat_projections = nn.ModuleDict()
        for et in self.entity_types:
            self.gat_projections[et] = nn.Linear(self.gat_out_channels, hidden_channels)
        self.gat_projections['temporal'] = nn.Linear(self.gat_out_channels, hidden_channels)
        self.gat_projections['transaction'] = nn.Linear(self.gat_out_channels, hidden_channels)
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            if i == 0:
                
                for et in self.entity_types:
                    conv_dict[('transaction', f'to_{et}', et)] = GATConv(
                        hidden_channels, hidden_channels//4, heads=8, concat=True, add_self_loops=False)
                    conv_dict[(et, f'from_{et}', 'transaction')] = GATConv(
                        hidden_channels, hidden_channels//4, heads=8, concat=True, add_self_loops=False)
                
                conv_dict[('transaction', 'temporal', 'transaction')] = GATConv(
                    hidden_channels, hidden_channels//4, heads=8, concat=True)
            else:
                
                for et in self.entity_types:
                    conv_dict[('transaction', f'to_{et}', et)] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
                    conv_dict[(et, f'from_{et}', 'transaction')] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
                
                conv_dict[('transaction', 'temporal', 'transaction')] = ResidualGNNLayer(
                    hidden_channels, hidden_channels
                )
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels // 2),  
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1)                 
        )
        
        
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
    
    def forward(self, data):
        
        tx_features = data['transaction'].x
        x_dict = {
            'transaction': F.elu(self.tx_proj(tx_features) + self.tx_skip(tx_features))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                entity_x = self.entity_proj(data[et].x)
                attn_weights = self.entity_attn[et](entity_x)
                x_dict[et] = entity_x * attn_weights
            else:
                x_dict[et] = torch.zeros(0, self.hidden_channels, device=tx_features.device)
        
        
        layer_outputs = []
        for i, conv in enumerate(self.convs):
            try:
                
                x_dict = conv(x_dict, data.edge_index_dict)
                
                
                if i == 0:
                    new_x_dict = {}
                    for key, value in x_dict.items():
                        if key in self.gat_projections:
                            new_x_dict[key] = self.gat_projections[key](F.elu(value))
                        else:
                            new_x_dict[key] = F.elu(value)
                    x_dict = new_x_dict
                else:
                    x_dict = {k: F.elu(v) for k, v in x_dict.items()}
                
                
                if 'transaction' in x_dict:
                    layer_outputs.append(x_dict['transaction'])
            except Exception as e:
                print(f"Skipping convolution: {e}")
                continue
        
        
        if layer_outputs:
            weights = F.softmax(self.layer_weights[:len(layer_outputs)], dim=0)
            fused_output = torch.zeros_like(layer_outputs[0])
            for idx, output in enumerate(layer_outputs):
                fused_output += weights[idx] * output
            combined = torch.cat([x_dict['transaction'], fused_output], dim=1)
        else:
            combined = x_dict['transaction']
        
        
        return self.head(combined).squeeze()




class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss





def evaluate(loader, model):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits)
            labels = batch['transaction'].y[:batch['transaction'].batch_size]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    if not all_probs:
        return 0.0
    
    val_probs = torch.cat(all_probs).numpy()
    val_labels = torch.cat(all_labels).numpy()
    
    if len(np.unique(val_labels)) > 1:
        return average_precision_score(val_labels, val_probs)
    return 0.0

def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = TargetEncodingGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = AdvancedFraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=256,  
        num_layers=4          
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [25, 20] for key in data.edge_index_dict},  
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [25, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    
    fraud_ratio = data['transaction'].y[data['transaction'].train_mask].mean().item()
    criterion = BalancedFocalLoss(alpha=1-fraud_ratio, gamma=2.0)
    print(f"Fraud ratio: {fraud_ratio:.4f}, Using alpha={1-fraud_ratio:.4f}")
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    
    best_pr_auc = 0
    no_improve = 0
    max_epochs = 30
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)  
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        val_pr_auc = evaluate(val_loader, model)
        scheduler.step(val_pr_auc)
        
        
        epoch_time = time.time() - start_time
        
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"Epoch {epoch}: Loss={total_loss/batch_count:.4f}, Val PR-AUC={val_pr_auc:.4f}* (Best) [{epoch_time:.1f}s]")
        else:
            no_improve += 1
            print(f"Epoch {epoch}: Loss={total_loss/batch_count:.4f}, Val PR-AUC={val_pr_auc:.4f} [{epoch_time:.1f}s]")
            
            if no_improve >= 4:  
                print(f"Early stopping at epoch {epoch}")
                break
    
    
    model.load_state_dict(torch.load('best_model.pt'))
    final_pr_auc = evaluate(val_loader, model)
    return model, final_pr_auc





if __name__ == "__main__":
    TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    
    model, pr_auc = train_fraud_model(TRANSACTION_PATH, IDENTITY_PATH)
    print(f"\nFinal Validation PR-AUC: {pr_auc:.4f}") 









import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear as PyGLinear
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import gc
import warnings
from collections import defaultdict
import time
from category_encoders import TargetEncoder
warnings.filterwarnings('ignore')





class TargetEncodingGraphBuilder:
    def __init__(self, entity_types, cat_features):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.tx_times = []
        
        
        self.tx_features = []
        self.tx_labels = []
        self.tx_categorical = {col: [] for col in cat_features}
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8
        self.target_encoders = {}

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        self.tx_labels.append(batch_df['isFraud'].values)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation without categoricals (handled separately)"""
        features = []
        
        
        num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        for col in num_cols:
            if col in batch_df:
                median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding to categorical features"""
        print("Applying target encoding...")
        start_time = time.time()
        
        
        tx_labels = np.concatenate(self.tx_labels)
        cat_data = {}
        for col in self.cat_features:
            cat_data[col] = np.concatenate(self.tx_categorical[col])
        
        
        num_train = int(self.transaction_counter * self.train_ratio)
        train_mask = np.zeros(self.transaction_counter, dtype=bool)
        train_mask[:num_train] = True
        
        
        encoded_features = []
        for col in self.cat_features:
            
            encoder = TargetEncoder(smoothing=10, min_samples_leaf=20)
            encoder.fit(
                cat_data[col][train_mask], 
                tx_labels[train_mask]
            )
            self.target_encoders[col] = encoder
            
            
            encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
            encoded_features.append(encoded)
        
        print(f"Target encoding completed in {time.time()-start_time:.1f} seconds")
        return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        tx_labels = np.concatenate(self.tx_labels)
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        scaler = StandardScaler()
        tx_features = scaler.fit_transform(tx_features)
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
        
        
        num_train = int(self.transaction_counter * self.train_ratio)
        train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
        train_mask[:num_train] = True
        val_mask[num_train:] = True
        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 1), dtype=np.float32)
            for entity_idx, count in self.entity_counts[et].items():
                if entity_idx < num_entities:
                    entity_features[entity_idx, 0] = np.log1p(count)
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building temporal edges...")
        time_edges = []
        time_window = 600  
        transaction_count = self.transaction_counter
        
        
        sample_size = min(150000, transaction_count)  
        sample_indices = np.random.choice(transaction_count, sample_size, replace=False)
        sample_times = [self.tx_times[i] for i in sample_indices]
        
        
        sorted_indices = sorted(sample_indices, key=lambda i: self.tx_times[i])
        sorted_times = [self.tx_times[i] for i in sorted_indices]
        
        
        for i in range(1, len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_time = sorted_times[i]
            j = i - 1
            while j >= 0 and (current_time - sorted_times[j]) <= time_window:
                time_edges.append((current_idx, sorted_indices[j]))
                time_edges.append((sorted_indices[j], current_idx))
                j -= 1
        
        if time_edges:
            src, dst = zip(*time_edges)
            time_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
            print(f"Added {len(time_edges)} temporal edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times, self.tx_categorical
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class ResidualGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            if isinstance(x, tuple):
                return self.lin(x[1])
            return self.lin(x)
        
        if isinstance(x, tuple):
            x_dst = x[1]
            conv_out = self.conv(x, edge_index)
            return self.norm(F.elu(conv_out) + self.lin(x_dst))
        return self.norm(F.elu(self.conv(x, edge_index)) + self.lin(x))

class OptimizedFraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels)
        )
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU()
        )
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            
            for et in self.entity_types:
                conv_dict[('transaction', f'to_{et}', et)] = SAGEConv(
                    hidden_channels, hidden_channels)
                conv_dict[(et, f'from_{et}', 'transaction')] = SAGEConv(
                    hidden_channels, hidden_channels)
            
            conv_dict[('transaction', 'temporal', 'transaction')] = SAGEConv(
                hidden_channels, hidden_channels)
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        
        self.skip_connection = nn.Linear(tx_feature_size, hidden_channels)
    
    def forward(self, data):
        
        tx_features = data['transaction'].x
        x_dict = {
            'transaction': self.tx_proj(tx_features)
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                x_dict[et] = self.entity_proj(data[et].x)
            else:
                x_dict[et] = torch.zeros(0, self.hidden_channels, device=tx_features.device)
        
        
        transaction_features = []
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                transaction_features.append(x_dict['transaction'])
            except Exception as e:
                print(f"Skipping convolution: {e}")
                continue
        
        
        if transaction_features:
            final_features = transaction_features[-1]
        else:
            final_features = x_dict['transaction']
            
        
        skip = self.skip_connection(tx_features)
        combined = torch.cat([final_features, skip], dim=1)
        
        
        return self.head(combined).squeeze()




class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss





def evaluate(loader, model):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits)
            labels = batch['transaction'].y[:batch['transaction'].batch_size]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    if not all_probs:
        return 0.0
    
    val_probs = torch.cat(all_probs).numpy()
    val_labels = torch.cat(all_labels).numpy()
    
    if len(np.unique(val_labels)) > 1:
        return average_precision_score(val_labels, val_probs)
    return 0.0

def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = TargetEncodingGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = OptimizedFraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=192,  
        num_layers=3
    )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [25, 20] for key in data.edge_index_dict},  
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [25, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    
    
    fraud_ratio = data['transaction'].y[data['transaction'].train_mask].mean().item()
    criterion = BalancedFocalLoss(alpha=1-fraud_ratio, gamma=2.0)
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    
    best_pr_auc = 0
    no_improve = 0
    max_epochs = 40  
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)  
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        val_pr_auc = evaluate(val_loader, model)
        scheduler.step(val_pr_auc)
        
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model7.pt')
        else:
            no_improve += 1
            if no_improve >= 6:  
                break
    
    
    model.load_state_dict(torch.load('best_model7.pt'))
    final_pr_auc = evaluate(val_loader, model)
    return model, final_pr_auc





if __name__ == "__main__":
    TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    
    model, pr_auc = train_fraud_model(TRANSACTION_PATH, IDENTITY_PATH)
    print(f"\nFinal Validation PR-AUC: {pr_auc:.4f}") 



import torch
import gc
gc.collect()
torch.cuda.empty_cache()


1





import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, average_precision_score, 
                            precision_recall_curve, auc, roc_auc_score)
import gc
import warnings
from collections import defaultdict
import time
import joblib
from category_encoders import TargetEncoder
import os
warnings.filterwarnings('ignore')





class TargetEncodingGraphBuilder:
    def __init__(self, entity_types, cat_features, inference_mode=False, 
                 target_encoders=None, scaler=None, num_medians=None):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.tx_times = []
        self.inference_mode = inference_mode
        self.target_encoders = target_encoders
        self.scaler = scaler
        self.num_medians = num_medians
        
        
        self.tx_features = []
        if not inference_mode:
            self.tx_labels = []
        self.tx_categorical = {col: [] for col in cat_features}
        self.transaction_ids = []  
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        self.transaction_ids.append(batch_df['TransactionID'].values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        
        if not self.inference_mode:
            self.tx_labels.append(batch_df['isFraud'].values)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation without categoricals (handled separately)"""
        features = []
        num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        for i, col in enumerate(num_cols):
            if col in batch_df:
                if self.inference_mode and self.num_medians is not None and col in self.num_medians:
                    
                    median_val = self.num_medians[col]
                else:
                    median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding to categorical features"""
        if self.inference_mode:
            print("Applying target encoding with pre-trained encoders...")
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            encoded_features = []
            for col in self.cat_features:
                encoded = self.target_encoders[col].transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            return np.column_stack(encoded_features)
        else:
            print("Applying target encoding...")
            start_time = time.time()
            
            
            tx_labels = np.concatenate(self.tx_labels)
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = np.zeros(self.transaction_counter, dtype=bool)
            train_mask[:num_train] = True
            
            
            self.target_encoders = {}
            encoded_features = []
            for col in self.cat_features:
                
                encoder = TargetEncoder(smoothing=10, min_samples_leaf=20)
                encoder.fit(
                    cat_data[col][train_mask], 
                    tx_labels[train_mask]
                )
                self.target_encoders[col] = encoder
                
                
                encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            
            print(f"Target encoding completed in {time.time()-start_time:.1f} seconds")
            return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        if not self.inference_mode:
            self.scaler = StandardScaler()
            tx_features = self.scaler.fit_transform(tx_features)
        else:
            tx_features = self.scaler.transform(tx_features)
            
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        
        if not self.inference_mode:
            tx_labels = np.concatenate(self.tx_labels)
            data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            train_mask[:num_train] = True
            val_mask[num_train:] = True
            data['transaction'].train_mask = train_mask
            data['transaction'].val_mask = val_mask
        
        
        if self.inference_mode:
            tx_ids = np.concatenate(self.transaction_ids)
            data['transaction'].transaction_id = torch.tensor(tx_ids, dtype=torch.long)
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 1), dtype=np.float32)
            for entity_idx, count in self.entity_counts[et].items():
                if entity_idx < num_entities:
                    entity_features[entity_idx, 0] = np.log1p(count)
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building temporal edges...")
        time_edges = []
        time_window = 300
        transaction_count = self.transaction_counter
        
        
        sample_size = min(100000, transaction_count)
        sample_indices = np.random.choice(transaction_count, sample_size, replace=False)
        sample_times = [self.tx_times[i] for i in sample_indices]
        
        
        sorted_indices = sorted(sample_indices, key=lambda i: self.tx_times[i])
        sorted_times = [self.tx_times[i] for i in sorted_indices]
        
        
        for i in range(1, len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_time = sorted_times[i]
            j = i - 1
            while j >= 0 and (current_time - sorted_times[j]) <= time_window:
                time_edges.append((current_idx, sorted_indices[j]))
                time_edges.append((sorted_indices[j], current_idx))
                j -= 1
        
        if time_edges:
            src, dst = zip(*time_edges)
            time_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
            print(f"Added {len(time_edges)} temporal edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times, self.tx_categorical
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class ResidualGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            if isinstance(x, tuple):
                return self.lin(x[1])
            return self.lin(x)
        
        if isinstance(x, tuple):
            x_dst = x[1]
            conv_out = self.conv(x, edge_index)
            return self.norm(F.elu(conv_out) + self.lin(x_dst))
        return self.norm(F.elu(self.conv(x, edge_index)) + self.lin(x))

class EnhancedFraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.tx_skip = nn.Linear(tx_feature_size, hidden_channels)
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU()
        )
        self.entity_attn = nn.ModuleDict({
            et: nn.Linear(hidden_channels, 1) for et in self.entity_types
        })
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                if i == 0:
                    conv_dict[('transaction', f'to_{et}', et)] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                    conv_dict[(et, f'from_{et}', 'transaction')] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                else:
                    conv_dict[('transaction', f'to_{et}', et)] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
                    conv_dict[(et, f'from_{et}', 'transaction')] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
            
            
            if i == 0:
                conv_dict[('transaction', 'temporal', 'transaction')] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True)
            else:
                conv_dict[('transaction', 'temporal', 'transaction')] = ResidualGNNLayer(
                    hidden_channels, hidden_channels
                )
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        
        tx_features = data['transaction'].x
        x_dict = {
            'transaction': F.elu(self.tx_proj(tx_features) + self.tx_skip(tx_features))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                entity_x = self.entity_proj(data[et].x)
                attn_weights = torch.sigmoid(self.entity_attn[et](entity_x))
                x_dict[et] = entity_x * attn_weights
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=tx_features.device)
        
        
        layer_outputs = []
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(x) for k, x in x_dict.items()}
                layer_outputs.append(x_dict['transaction'])
            except Exception as e:
                print(f"Skipping convolution: {e}")
                continue
        
        
        if layer_outputs:
            mean_layer = torch.mean(torch.stack(layer_outputs), dim=0)
            combined = torch.cat([x_dict['transaction'], mean_layer], dim=1)
        else:
            combined = x_dict['transaction']
        
        return self.head(combined).squeeze()





class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss





def evaluate_metrics(loader, model, mode='val'):
    """Evaluate model and return comprehensive classification metrics"""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits)
            preds = (probs >= 0.5).long()
            labels = batch['transaction'].y[:batch['transaction'].batch_size]
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    if not all_probs:
        return {}
    
    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    
    if len(np.unique(labels)) < 2:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': np.zeros((2, 2)),
            'pr_auc': 0.0,
            'roc_auc': 0.5
        }
    
    
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc
    }
    
    
    if mode != 'train':
        precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
        metrics['pr_curve'] = (precision_curve, recall_curve)
        metrics['pr_auc_detailed'] = auc(recall_curve, precision_curve)
    
    return metrics

def print_metrics(metrics, set_name='Dataset'):
    """Print formatted classification metrics"""
    print(f"\n===== {set_name} Metrics =====")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
    print(f"PR-AUC:    {metrics.get('pr_auc', 0):.4f}")
    print(f"ROC-AUC:   {metrics.get('roc_auc', 0.5):.4f}")
    
    if 'pr_auc_detailed' in metrics:
        print(f"Detailed PR-AUC: {metrics['pr_auc_detailed']:.4f}")
    
    print("Confusion Matrix:")
    print(metrics.get('confusion_matrix', np.zeros((2, 2))))
    print("="*40)





def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = TargetEncodingGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = EnhancedFraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=3
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    train_indices = data['transaction'].train_mask.nonzero().squeeze()
    subset_size = min(20000, len(train_indices))
    subset_indices = torch.randperm(len(train_indices))[:subset_size]
    train_subset_mask = torch.zeros_like(data['transaction'].train_mask).bool()
    train_subset_mask[train_indices[subset_indices]] = True
    
    train_metrics_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', train_subset_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    
    fraud_ratio = data['transaction'].y[data['transaction'].train_mask].mean().item()
    criterion = BalancedFocalLoss(alpha=1-fraud_ratio, gamma=2.0)
    print(f"Fraud ratio: {fraud_ratio:.4f}, Using alpha={1-fraud_ratio:.4f}")
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    
    history = {'train': [], 'val': []}
    best_pr_auc = 0
    no_improve = 0
    max_epochs = 30
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        train_metrics = evaluate_metrics(train_metrics_loader, model, mode='train')
        val_metrics = evaluate_metrics(val_loader, model, mode='val')
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        
        print(f"\nEpoch {epoch} - Loss: {total_loss/batch_count:.4f}")
        print_metrics(train_metrics, "Training Subset")
        print_metrics(val_metrics, "Validation Set")
        
        
        current_pr_auc = val_metrics.get('pr_auc', 0)
        if current_pr_auc > best_pr_auc:
            best_pr_auc = current_pr_auc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model_rez.pt')
            print(f"New best model saved with PR-AUC: {best_pr_auc:.4f}")
        else:
            no_improve += 1
            
        
        if no_improve >= 5:
            print(f"Early stopping at epoch {epoch}")
            break
            
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.1f} seconds")
    
    
    print("\n===== Final Evaluation =====")
    
    
    model.load_state_dict(torch.load('best_model_rez.pt'))
    
    
    full_train_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=False
    )
    full_train_metrics = evaluate_metrics(full_train_loader, model, mode='train')
    print_metrics(full_train_metrics, "Full Training Set")
    
    
    full_val_metrics = evaluate_metrics(val_loader, model, mode='val')
    print_metrics(full_val_metrics, "Full Validation Set")
    
    
    print("Saving preprocessing artifacts...")
    artifacts = {
        'target_encoders': graph_builder.target_encoders,
        'scaler': graph_builder.scaler,
        'num_medians': {
            
            col: np.median(np.vstack(graph_builder.tx_features)[:, i]) 
            for i, col in enumerate([
                'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
                'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
                'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
                'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
                'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
                'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
                'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
                'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
                'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
                'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
                'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
                'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
                'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
                'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
                'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
                'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
                'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
                'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
                'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
                'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
                'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
                'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
                'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
                'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
            ])
        }
    }
    joblib.dump(artifacts, 'inference_artifacts.pkl')
    
    return model, full_train_metrics, full_val_metrics, artifacts





def predict_fraud(model, transaction_path, identity_path, artifacts):
    """Predict fraud probabilities for test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    
    
    graph_builder = TargetEncodingGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES,
        inference_mode=True,
        target_encoders=artifacts['target_encoders'],
        scaler=artifacts['scaler'],
        num_medians=artifacts['num_medians']
    )
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={
        x : x.replace('-', '_') for x in list(
        set(identity_df.columns)
                                         )  
        }
                          
              ,inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing test chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building test graph...")
    test_data = graph_builder.build_graph()
    test_data = test_data.to(device)
    print(f"Test graph metadata: {test_data}")
    
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [20, 15] for key in test_data.edge_index_dict},
        input_nodes=('transaction', torch.arange(test_data['transaction'].x.size(0))),
        batch_size=4096,  
        shuffle=False
    )
    
    
    all_probs = []
    transaction_ids = []
    with torch.no_grad():
        for batch in test_loader:
            
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits).cpu().numpy()
            all_probs.append(probs)
            
            
            batch_ids = batch['transaction'].transaction_id.cpu().numpy()
            transaction_ids.append(batch_ids)
    
    
    test_probs = np.concatenate(all_probs)
    transaction_ids = np.concatenate(transaction_ids)
    
    
    submission = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': test_probs
    })
    
    return submission





if __name__ == "__main__":
    
    TRAIN_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    TEST_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
    
    
    if not os.path.exists('best_model_rez.pt'):
        print("Training model...")
        model, train_metrics, val_metrics, artifacts = train_fraud_model(
            TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
        print("\n===== Final Training Results =====")
        print_metrics(train_metrics, "Full Training Set")
        print_metrics(val_metrics, "Full Validation Set")
    else:
        print("Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sample = pd.read_csv(TRAIN_TRANSACTION_PATH, nrows=1)
        tx_feature_size = len(sample.columns) + len(CAT_FEATURES)  
        model = EnhancedFraudGNN(tx_feature_size, 128, 3).to(device)
        model.load_state_dict(torch.load('best_model_rez.pt'))
        artifacts = joblib.load('inference_artifacts.pkl')
    
    
    print("\n===== Running Inference on Test Data =====")
    submission = predict_fraud(model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts)
    
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved with shape:", submission.shape)
    print("First 5 predictions:")
    print(submission.head())








import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, average_precision_score, 
                            precision_recall_curve, auc, roc_auc_score)
import gc
import warnings
from collections import defaultdict
import time
import joblib
from category_encoders import TargetEncoder
import os
warnings.filterwarnings('ignore')





class TargetEncodingGraphBuilder:
    def __init__(self, entity_types, cat_features, inference_mode=False, 
                 target_encoders=None, scaler=None, num_medians=None):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.tx_times = []
        self.inference_mode = inference_mode
        self.target_encoders = target_encoders
        self.scaler = scaler
        self.num_medians = num_medians
        
        
        self.tx_features = []
        if not inference_mode:
            self.tx_labels = []
        self.tx_categorical = {col: [] for col in cat_features}
        self.transaction_ids = []  
        self.num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        self.transaction_ids.append(batch_df['TransactionID'].values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        
        if not self.inference_mode:
            self.tx_labels.append(batch_df['isFraud'].values)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation without categoricals (handled separately)"""
        features = []
        
        for col in self.num_cols:
            if col in batch_df:
                if self.inference_mode and self.num_medians is not None and col in self.num_medians:
                    
                    median_val = self.num_medians[col]
                else:
                    median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding to categorical features without leakage"""
        if self.inference_mode:
            print("Applying target encoding with pre-trained encoders...")
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            encoded_features = []
            for col in self.cat_features:
                encoded = self.target_encoders[col].transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            return np.column_stack(encoded_features)
        else:
            print("Applying target encoding...")
            start_time = time.time()
            
            
            tx_labels = np.concatenate(self.tx_labels)
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = np.zeros(self.transaction_counter, dtype=bool)
            train_mask[:num_train] = True
            
            
            self.target_encoders = {}
            encoded_features = []
            for col in self.cat_features:
                
                encoder = TargetEncoder(smoothing=20, min_samples_leaf=50)  
                encoder.fit(
                    cat_data[col][train_mask], 
                    tx_labels[train_mask]  
                )
                self.target_encoders[col] = encoder
                
                
                encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            
            print(f"Target encoding completed in {time.time()-start_time:.1f} seconds")
            return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        
        
        if not self.inference_mode:
            self.num_medians = {}
            for i, col in enumerate(self.num_cols):
                if i < num_features.shape[1]:
                    self.num_medians[col] = np.median(num_features[:, i])
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        if not self.inference_mode:
            self.scaler = StandardScaler()
            tx_features = self.scaler.fit_transform(tx_features)
        else:
            tx_features = self.scaler.transform(tx_features)
            
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        
        if not self.inference_mode:
            tx_labels = np.concatenate(self.tx_labels)
            data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            train_mask[:num_train] = True
            val_mask[num_train:] = True
            data['transaction'].train_mask = train_mask
            data['transaction'].val_mask = val_mask
        
        
        if self.inference_mode:
            tx_ids = np.concatenate(self.transaction_ids)
            data['transaction'].transaction_id = torch.tensor(tx_ids, dtype=torch.long)
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 1), dtype=np.float32)
            for entity_idx, count in self.entity_counts[et].items():
                if entity_idx < num_entities:
                    entity_features[entity_idx, 0] = np.log1p(count)
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building temporal edges...")
        time_edges = []
        time_window = 86400
        transaction_count = self.transaction_counter
        
        
        sample_size = min(100000, transaction_count)
        sample_indices = np.random.choice(transaction_count, sample_size, replace=False)
        sample_times = [self.tx_times[i] for i in sample_indices]
        
        
        sorted_indices = sorted(sample_indices, key=lambda i: self.tx_times[i])
        sorted_times = [self.tx_times[i] for i in sorted_indices]
        
        
        for i in range(1, len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_time = sorted_times[i]
            j = i - 1
            while j >= 0 and (current_time - sorted_times[j]) <= time_window:
                time_edges.append((current_idx, sorted_indices[j]))
                time_edges.append((sorted_indices[j], current_idx))
                j -= 1
        
        if time_edges:
            src, dst = zip(*time_edges)
            time_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
            print(f"Added {len(time_edges)} temporal edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times, self.tx_categorical
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class ResidualGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.2)  
        
    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            if isinstance(x, tuple):
                return self.lin(x[1])
            return self.lin(x)
        
        if isinstance(x, tuple):
            x_dst = x[1]
            conv_out = self.conv(x, edge_index)
            out = self.norm(F.elu(conv_out) + self.lin(x_dst))
        else:
            out = self.norm(F.elu(self.conv(x, edge_index)) + self.lin(x))
        
        return self.dropout(out)

class EnhancedFraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.tx_skip = nn.Linear(tx_feature_size, hidden_channels)
        
        
        for module in self.tx_proj:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
        nn.init.kaiming_normal_(self.tx_skip.weight, nonlinearity='relu')
        self.tx_skip.bias.data.zero_()
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.entity_attn = nn.ModuleDict({
            et: nn.Linear(hidden_channels, 1) for et in self.entity_types
        })
        
        
        for et in self.entity_types:
            nn.init.xavier_uniform_(self.entity_attn[et].weight)
            self.entity_attn[et].bias.data.zero_()
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                if i == 0:
                    conv_dict[('transaction', f'to_{et}', et)] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                    conv_dict[(et, f'from_{et}', 'transaction')] = GATConv(
                        hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                else:
                    conv_dict[('transaction', f'to_{et}', et)] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
                    conv_dict[(et, f'from_{et}', 'transaction')] = ResidualGNNLayer(
                        hidden_channels, hidden_channels
                    )
            
            
            if i == 0:
                conv_dict[('transaction', 'temporal', 'transaction')] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True)
            else:
                conv_dict[('transaction', 'temporal', 'transaction')] = ResidualGNNLayer(
                    hidden_channels, hidden_channels
                )
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(hidden_channels, 1)
        )
        
        
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, data):
        
        tx_features = data['transaction'].x
        x_dict = {
            'transaction': F.elu(self.tx_proj(tx_features) + self.tx_skip(tx_features))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                entity_x = self.entity_proj(data[et].x)
                attn_weights = torch.sigmoid(self.entity_attn[et](entity_x))
                x_dict[et] = entity_x * attn_weights
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=tx_features.device)
        
        
        layer_outputs = []
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}
                layer_outputs.append(x_dict['transaction'])
            except Exception as e:
                print(f"Skipping convolution: {e}")
                continue
        
        
        if layer_outputs:
            mean_layer = torch.mean(torch.stack(layer_outputs), dim=0)
            combined = torch.cat([x_dict['transaction'], mean_layer], dim=1)
        else:
            combined = x_dict['transaction']
        
        return self.head(combined).squeeze()





class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss





def evaluate_metrics(loader, model, mode='val'):
    """Evaluate model and return comprehensive classification metrics"""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits)
            preds = (probs >= 0.5).long()
            labels = batch['transaction'].y[:batch['transaction'].batch_size]
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    if not all_probs:
        return {}
    
    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    
    if len(np.unique(labels)) < 2:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': np.zeros((2, 2)),
            'pr_auc': 0.0,
            'roc_auc': 0.5
        }
    
    
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc
    }
    
    
    if mode != 'train':
        precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
        metrics['pr_curve'] = (precision_curve, recall_curve)
        metrics['pr_auc_detailed'] = auc(recall_curve, precision_curve)
    
    return metrics

def print_metrics(metrics, set_name='Dataset'):
    """Print formatted classification metrics"""
    print(f"\n===== {set_name} Metrics =====")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
    print(f"PR-AUC:    {metrics.get('pr_auc', 0):.4f}")
    print(f"ROC-AUC:   {metrics.get('roc_auc', 0.5):.4f}")
    
    if 'pr_auc_detailed' in metrics:
        print(f"Detailed PR-AUC: {metrics['pr_auc_detailed']:.4f}")
    
    print("Confusion Matrix:")
    print(metrics.get('confusion_matrix', np.zeros((2, 2))))
    print("="*40)





def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = TargetEncodingGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = EnhancedFraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=3
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    train_indices = data['transaction'].train_mask.nonzero().squeeze()
    subset_size = min(20000, len(train_indices))
    subset_indices = torch.randperm(len(train_indices))[:subset_size]
    train_subset_mask = torch.zeros_like(data['transaction'].train_mask).bool()
    train_subset_mask[train_indices[subset_indices]] = True
    
    train_metrics_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', train_subset_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    
    fraud_ratio = data['transaction'].y[data['transaction'].train_mask].mean().item()
    criterion = BalancedFocalLoss(alpha=1-fraud_ratio, gamma=4.0)  
    print(f"Fraud ratio: {fraud_ratio:.4f}, Using alpha={1-fraud_ratio:.4f}, gamma=4.0")
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    
    history = {'train': [], 'val': []}
    best_pr_auc = 0
    no_improve = 0
    max_epochs = 50
    no_improve_threshold = 8  
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        train_metrics = evaluate_metrics(train_metrics_loader, model, mode='train')
        val_metrics = evaluate_metrics(val_loader, model, mode='val')
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        
        print(f"\nEpoch {epoch} - Loss: {total_loss/batch_count:.4f}")
        print_metrics(train_metrics, "Training Subset")
        print_metrics(val_metrics, "Validation Set")
        
        
        current_pr_auc = val_metrics.get('pr_auc', 0)
        if current_pr_auc > best_pr_auc:
            best_pr_auc = current_pr_auc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model_an.pt')
            print(f"New best model saved with PR-AUC: {best_pr_auc:.4f}")
        else:
            no_improve += 1
            
        
        if no_improve >= no_improve_threshold:
            print(f"Early stopping at epoch {epoch}")
            break
            
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.1f} seconds")
    
    
    print("\n===== Final Evaluation =====")
    
    
    model.load_state_dict(torch.load('best_model_an.pt'))
    
    
    full_train_loader = NeighborLoader(
        data,
        num_neighbors={key: [20, 15] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=False
    )
    full_train_metrics = evaluate_metrics(full_train_loader, model, mode='train')
    print_metrics(full_train_metrics, "Full Training Set")
    
    
    full_val_metrics = evaluate_metrics(val_loader, model, mode='val')
    print_metrics(full_val_metrics, "Full Validation Set")
    
    
    print("Saving preprocessing artifacts...")
    artifacts = {
        'target_encoders': graph_builder.target_encoders,
        'scaler': graph_builder.scaler,
        'num_medians': graph_builder.num_medians
    }
    joblib.dump(artifacts, 'inference_artifacts.pkl')
    
    return model, full_train_metrics, full_val_metrics, artifacts





def predict_fraud(model, transaction_path, identity_path, artifacts):
    """Predict fraud probabilities for test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    
    
    graph_builder = TargetEncodingGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES,
        inference_mode=True,
        target_encoders=artifacts['target_encoders'],
        scaler=artifacts['scaler'],
        num_medians=artifacts['num_medians']
    )
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={
        x : x.replace('-', '_') for x in list(
        set(identity_df.columns)
                                         )  
        },inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing test chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building test graph...")
    test_data = graph_builder.build_graph()
    test_data = test_data.to(device)
    print(f"Test graph metadata: {test_data}")
    
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [20, 15] for key in test_data.edge_index_dict},
        input_nodes=('transaction', torch.arange(test_data['transaction'].x.size(0))),
        batch_size=4096,  
        shuffle=False
    )
    
    
    all_probs = []
    transaction_ids = []
    with torch.no_grad():
        for batch in test_loader:
            
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits).cpu().numpy()
            all_probs.append(probs)
            
            
            batch_ids = batch['transaction'].transaction_id.cpu().numpy()
            transaction_ids.append(batch_ids)
    
    
    test_probs = np.concatenate(all_probs)
    transaction_ids = np.concatenate(transaction_ids)
    
    
    submission = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': test_probs
    })
    
    return submission





if __name__ == "__main__":
    
    TRAIN_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    TEST_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
    
    
    if not os.path.exists('best_model_an.pt'):
        print("Training model...")
        model, train_metrics, val_metrics, artifacts = train_fraud_model(
            TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
        print("\n===== Final Training Results =====")
        print_metrics(train_metrics, "Full Training Set")
        print_metrics(val_metrics, "Full Validation Set")
    else:
        print("Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sample = pd.read_csv(TRAIN_TRANSACTION_PATH, nrows=1)
        CAT_FEATURES = [
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo',
            'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
            'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
            'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
        ]
        tx_feature_size = len(sample.columns) + len(CAT_FEATURES)  
        model = EnhancedFraudGNN(tx_feature_size, 128, 3).to(device)
        model.load_state_dict(torch.load('best_model_an.pt'))
        artifacts = joblib.load('inference_artifacts.pkl')
    
    
    print("\n===== Running Inference on Test Data =====")
    submission = predict_fraud(model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts)
    
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved with shape:", submission.shape)
    print("First 5 predictions:")
    print(submission.head())



    Training model...
Using device: cuda
Loaded identity data with 144233 rows
Processing chunk 1
Processing chunk 2
Processing chunk 3
Processing chunk 4
Processing chunk 5
Processing chunk 6
Building graph...
Building graph with 590540 transactions
Applying target encoding...
Target encoding completed in 16.6 seconds
Building temporal edges...
Added 115650812 temporal edges
Graph built in 182.0 seconds
Graph metadata: HeteroData(
  transaction={
    x=[590540, 437],
    y=[590540],
    train_mask=[590540],
    val_mask=[590540],
  },
  card={ x=[14893, 1] },
  addr={ x=[437, 1] },
  email={ x=[742, 1] },
  device={ x=[1786, 1] },
  product={ x=[5, 1] },
  (transaction, to_card, card)={ edge_index=[2, 590540] },
  (card, from_card, transaction)={ edge_index=[2, 590540] },
  (transaction, to_addr, addr)={ edge_index=[2, 524834] },
  (addr, from_addr, transaction)={ edge_index=[2, 524834] },
  (transaction, to_email, email)={ edge_index=[2, 507148] },
  (email, from_email, transaction)={ edge_index=[2, 507148] },
  (transaction, to_device, device)={ edge_index=[2, 118666] },
  (device, from_device, transaction)={ edge_index=[2, 118666] },
  (transaction, to_product, product)={ edge_index=[2, 590540] },
  (product, from_product, transaction)={ edge_index=[2, 590540] },
  (transaction, temporal, transaction)={ edge_index=[2, 115650812] }
)
Model parameters: 1,439,622
Fraud ratio: 0.0351, Using alpha=0.9649, gamma=4.0

Epoch 0 - Loss: 0.0122

===== Training Subset Metrics =====
Precision: 0.8439
Recall:    0.2065
F1 Score:  0.3318
PR-AUC:    0.4321
ROC-AUC:   0.8673
Confusion Matrix:
[[19266    27]
 [  561   146]]
========================================

===== Validation Set Metrics =====
Precision: 0.5836
Recall:    0.1452
F1 Score:  0.2325
PR-AUC:    0.2955
ROC-AUC:   0.8277
Detailed PR-AUC: 0.2954
Confusion Matrix:
[[113623    421]
 [  3474    590]]
========================================
New best model saved with PR-AUC: 0.2955
Epoch completed in 111.6 seconds

Epoch 1 - Loss: 0.0086

===== Training Subset Metrics =====
Precision: 0.8707
Recall:    0.2857
F1 Score:  0.4302
PR-AUC:    0.5172
ROC-AUC:   0.8963
Confusion Matrix:
[[19263    30]
 [  505   202]]
========================================

===== Validation Set Metrics =====
Precision: 0.7430
Recall:    0.1836
F1 Score:  0.2944
PR-AUC:    0.3417
ROC-AUC:   0.8381
Detailed PR-AUC: 0.3417
Confusion Matrix:
[[113786    258]
 [  3318    746]]
========================================
New best model saved with PR-AUC: 0.3417
Epoch completed in 111.4 seconds

Epoch 2 - Loss: 0.0080

===== Training Subset Metrics =====
Precision: 0.8255
Recall:    0.3479
F1 Score:  0.4896
PR-AUC:    0.5563
ROC-AUC:   0.9050
Confusion Matrix:
[[19241    52]
 [  461   246]]
========================================

===== Validation Set Metrics =====
Precision: 0.6648
Recall:    0.2352
F1 Score:  0.3475
PR-AUC:    0.3654
ROC-AUC:   0.8494
Detailed PR-AUC: 0.3654
Confusion Matrix:
[[113562    482]
 [  3108    956]]
========================================
New best model saved with PR-AUC: 0.3654
Epoch completed in 112.0 seconds

Epoch 3 - Loss: 0.0076

===== Training Subset Metrics =====
Precision: 0.8592
Recall:    0.3451
F1 Score:  0.4924
PR-AUC:    0.5773
ROC-AUC:   0.9136
Confusion Matrix:
[[19253    40]
 [  463   244]]
========================================

===== Validation Set Metrics =====
Precision: 0.6690
Recall:    0.2153
F1 Score:  0.3258
PR-AUC:    0.3542
ROC-AUC:   0.8415
Detailed PR-AUC: 0.3541
Confusion Matrix:
[[113611    433]
 [  3189    875]]
========================================
Epoch completed in 112.7 seconds

Epoch 4 - Loss: 0.0072

===== Training Subset Metrics =====
Precision: 0.9072
Recall:    0.3041
F1 Score:  0.4555
PR-AUC:    0.5940
ROC-AUC:   0.9203
Confusion Matrix:
[[19271    22]
 [  492   215]]
========================================

===== Validation Set Metrics =====
Precision: 0.7369
Recall:    0.2040
F1 Score:  0.3195
PR-AUC:    0.3790
ROC-AUC:   0.8448
Detailed PR-AUC: 0.3789
Confusion Matrix:
[[113748    296]
 [  3235    829]]
========================================
New best model saved with PR-AUC: 0.3790
Epoch completed in 112.9 seconds

Epoch 5 - Loss: 0.0070

===== Training Subset Metrics =====
Precision: 0.8861
Recall:    0.3960
F1 Score:  0.5474
PR-AUC:    0.6254
ROC-AUC:   0.9257
Confusion Matrix:
[[19257    36]
 [  427   280]]
========================================

===== Validation Set Metrics =====
Precision: 0.6306
Recall:    0.2411
F1 Score:  0.3489
PR-AUC:    0.3658
ROC-AUC:   0.8376
Detailed PR-AUC: 0.3657
Confusion Matrix:
[[113470    574]
 [  3084    980]]
========================================
Epoch completed in 112.6 seconds

Epoch 6 - Loss: 0.0067

===== Training Subset Metrics =====
Precision: 0.8984
Recall:    0.3876
F1 Score:  0.5415
PR-AUC:    0.6374
ROC-AUC:   0.9314
Confusion Matrix:
[[19262    31]
 [  433   274]]
========================================

===== Validation Set Metrics =====
Precision: 0.6600
Recall:    0.2451
F1 Score:  0.3574
PR-AUC:    0.3787
ROC-AUC:   0.8430
Detailed PR-AUC: 0.3786
Confusion Matrix:
[[113531    513]
 [  3068    996]]
========================================
Epoch completed in 112.3 seconds

Epoch 7 - Loss: 0.0066

===== Training Subset Metrics =====
Precision: 0.8452
Recall:    0.4710
F1 Score:  0.6049
PR-AUC:    0.6555
ROC-AUC:   0.9340
Confusion Matrix:
[[19232    61]
 [  374   333]]
========================================

===== Validation Set Metrics =====
Precision: 0.5561
Recall:    0.2990
F1 Score:  0.3889
PR-AUC:    0.3781
ROC-AUC:   0.8423
Detailed PR-AUC: 0.3780
Confusion Matrix:
[[113074    970]
 [  2849   1215]]
========================================
Epoch completed in 111.8 seconds

Epoch 8 - Loss: 0.0065

===== Training Subset Metrics =====
Precision: 0.8252
Recall:    0.4809
F1 Score:  0.6077
PR-AUC:    0.6630
ROC-AUC:   0.9355
Confusion Matrix:
[[19221    72]
 [  367   340]]
========================================

===== Validation Set Metrics =====
Precision: 0.5574
Recall:    0.3179
F1 Score:  0.4049
PR-AUC:    0.4017
ROC-AUC:   0.8495
Detailed PR-AUC: 0.4017
Confusion Matrix:
[[113018   1026]
 [  2772   1292]]
========================================
New best model saved with PR-AUC: 0.4017
Epoch completed in 111.6 seconds

Epoch 9 - Loss: 0.0063

===== Training Subset Metrics =====
Precision: 0.8809
Recall:    0.4498
F1 Score:  0.5955
PR-AUC:    0.6824
ROC-AUC:   0.9423
Confusion Matrix:
[[19250    43]
 [  389   318]]
========================================

===== Validation Set Metrics =====
Precision: 0.4354
Recall:    0.2729
F1 Score:  0.3355
PR-AUC:    0.3481
ROC-AUC:   0.8417
Detailed PR-AUC: 0.3480
Confusion Matrix:
[[112606   1438]
 [  2955   1109]]
========================================
Epoch completed in 111.7 seconds

Epoch 10 - Loss: 0.0062

===== Training Subset Metrics =====
Precision: 0.8499
Recall:    0.4965
F1 Score:  0.6268
PR-AUC:    0.6876
ROC-AUC:   0.9409
Confusion Matrix:
[[19231    62]
 [  356   351]]
========================================

===== Validation Set Metrics =====
Precision: 0.5453
Recall:    0.3036
F1 Score:  0.3901
PR-AUC:    0.3975
ROC-AUC:   0.8550
Detailed PR-AUC: 0.3975
Confusion Matrix:
[[113015   1029]
 [  2830   1234]]
========================================
Epoch completed in 111.7 seconds

Epoch 11 - Loss: 0.0061

===== Training Subset Metrics =====
Precision: 0.8935
Recall:    0.4866
F1 Score:  0.6300
PR-AUC:    0.6925
ROC-AUC:   0.9427
Confusion Matrix:
[[19252    41]
 [  363   344]]
========================================

===== Validation Set Metrics =====
Precision: 0.4213
Recall:    0.2916
F1 Score:  0.3446
PR-AUC:    0.2998
ROC-AUC:   0.8424
Detailed PR-AUC: 0.2997
Confusion Matrix:
[[112416   1628]
 [  2879   1185]]
========================================
Epoch completed in 112.4 seconds

Epoch 12 - Loss: 0.0060

===== Training Subset Metrics =====
Precision: 0.8638
Recall:    0.5205
F1 Score:  0.6496
PR-AUC:    0.7107
ROC-AUC:   0.9445
Confusion Matrix:
[[19235    58]
 [  339   368]]
========================================

===== Validation Set Metrics =====
Precision: 0.5360
Recall:    0.3263
F1 Score:  0.4056
PR-AUC:    0.4011
ROC-AUC:   0.8567
Detailed PR-AUC: 0.4010
Confusion Matrix:
[[112896   1148]
 [  2738   1326]]
========================================
Epoch completed in 112.8 seconds

Epoch 13 - Loss: 0.0059

===== Training Subset Metrics =====
Precision: 0.8428
Recall:    0.5460
F1 Score:  0.6627
PR-AUC:    0.7061
ROC-AUC:   0.9428
Confusion Matrix:
[[19221    72]
 [  321   386]]
========================================

===== Validation Set Metrics =====
Precision: 0.5495
Recall:    0.3605
F1 Score:  0.4354
PR-AUC:    0.4255
ROC-AUC:   0.8509
Detailed PR-AUC: 0.4255
Confusion Matrix:
[[112843   1201]
 [  2599   1465]]
========================================
New best model saved with PR-AUC: 0.4255
Epoch completed in 112.5 seconds

Epoch 14 - Loss: 0.0058

===== Training Subset Metrics =====
Precision: 0.8943
Recall:    0.5149
F1 Score:  0.6535
PR-AUC:    0.7215
ROC-AUC:   0.9480
Confusion Matrix:
[[19250    43]
 [  343   364]]
========================================

===== Validation Set Metrics =====
Precision: 0.5582
Recall:    0.3091
F1 Score:  0.3978
PR-AUC:    0.3884
ROC-AUC:   0.8426
Detailed PR-AUC: 0.3884
Confusion Matrix:
[[113050    994]
 [  2808   1256]]
========================================
Epoch completed in 112.8 seconds

Epoch 15 - Loss: 0.0057

===== Training Subset Metrics =====
Precision: 0.9186
Recall:    0.5106
F1 Score:  0.6564
PR-AUC:    0.7320
ROC-AUC:   0.9505
Confusion Matrix:
[[19261    32]
 [  346   361]]
========================================

===== Validation Set Metrics =====
Precision: 0.5557
Recall:    0.2997
F1 Score:  0.3894
PR-AUC:    0.3886
ROC-AUC:   0.8454
Detailed PR-AUC: 0.3886
Confusion Matrix:
[[113070    974]
 [  2846   1218]]
========================================
Epoch completed in 111.4 seconds

Epoch 16 - Loss: 0.0056

===== Training Subset Metrics =====
Precision: 0.9210
Recall:    0.5276
F1 Score:  0.6709
PR-AUC:    0.7416
ROC-AUC:   0.9517
Confusion Matrix:
[[19261    32]
 [  334   373]]
========================================

===== Validation Set Metrics =====
Precision: 0.6513
Recall:    0.2918
F1 Score:  0.4031
PR-AUC:    0.4133
ROC-AUC:   0.8431
Detailed PR-AUC: 0.4133
Confusion Matrix:
[[113409    635]
 [  2878   1186]]
========================================
Epoch completed in 109.7 seconds

Epoch 17 - Loss: 0.0055

===== Training Subset Metrics =====
Precision: 0.8842
Recall:    0.5615
F1 Score:  0.6869
PR-AUC:    0.7463
ROC-AUC:   0.9525
Confusion Matrix:
[[19241    52]
 [  310   397]]
========================================

===== Validation Set Metrics =====
Precision: 0.5758
Recall:    0.3327
F1 Score:  0.4217
PR-AUC:    0.4170
ROC-AUC:   0.8465
Detailed PR-AUC: 0.4169
Confusion Matrix:
[[113048    996]
 [  2712   1352]]
========================================
Epoch completed in 109.6 seconds

Epoch 18 - Loss: 0.0055

===== Training Subset Metrics =====
Precision: 0.8894
Recall:    0.5912
F1 Score:  0.7103
PR-AUC:    0.7571
ROC-AUC:   0.9549
Confusion Matrix:
[[19241    52]
 [  289   418]]
========================================

===== Validation Set Metrics =====
Precision: 0.6039
Recall:    0.3482
F1 Score:  0.4417
PR-AUC:    0.4328
ROC-AUC:   0.8387
Detailed PR-AUC: 0.4327
Confusion Matrix:
[[113116    928]
 [  2649   1415]]
========================================
New best model saved with PR-AUC: 0.4328
Epoch completed in 111.0 seconds

Epoch 19 - Loss: 0.0054

===== Training Subset Metrics =====
Precision: 0.8987
Recall:    0.5898
F1 Score:  0.7122
PR-AUC:    0.7630
ROC-AUC:   0.9565
Confusion Matrix:
[[19246    47]
 [  290   417]]
========================================

===== Validation Set Metrics =====
Precision: 0.6290
Recall:    0.3324
F1 Score:  0.4350
PR-AUC:    0.4386
ROC-AUC:   0.8429
Detailed PR-AUC: 0.4386
Confusion Matrix:
[[113247    797]
 [  2713   1351]]
========================================
New best model saved with PR-AUC: 0.4386
Epoch completed in 110.6 seconds

Epoch 20 - Loss: 0.0053

===== Training Subset Metrics =====
Precision: 0.9245
Recall:    0.5545
F1 Score:  0.6932
PR-AUC:    0.7674
ROC-AUC:   0.9600
Confusion Matrix:
[[19261    32]
 [  315   392]]
========================================

===== Validation Set Metrics =====
Precision: 0.6478
Recall:    0.3100
F1 Score:  0.4194
PR-AUC:    0.4251
ROC-AUC:   0.8354
Detailed PR-AUC: 0.4251
Confusion Matrix:
[[113359    685]
 [  2804   1260]]
========================================
Epoch completed in 111.8 seconds

Epoch 21 - Loss: 0.0052

===== Training Subset Metrics =====
Precision: 0.9113
Recall:    0.5813
F1 Score:  0.7098
PR-AUC:    0.7692
ROC-AUC:   0.9606
Confusion Matrix:
[[19253    40]
 [  296   411]]
========================================

===== Validation Set Metrics =====
Precision: 0.5532
Recall:    0.3172
F1 Score:  0.4032
PR-AUC:    0.3976
ROC-AUC:   0.8274
Detailed PR-AUC: 0.3976
Confusion Matrix:
[[113003   1041]
 [  2775   1289]]
========================================
Epoch completed in 112.3 seconds

Epoch 22 - Loss: 0.0052

===== Training Subset Metrics =====
Precision: 0.9335
Recall:    0.5559
F1 Score:  0.6968
PR-AUC:    0.7736
ROC-AUC:   0.9600
Confusion Matrix:
[[19265    28]
 [  314   393]]
========================================

===== Validation Set Metrics =====
Precision: 0.6747
Recall:    0.3027
F1 Score:  0.4179
PR-AUC:    0.4270
ROC-AUC:   0.8363
Detailed PR-AUC: 0.4270
Confusion Matrix:
[[113451    593]
 [  2834   1230]]
========================================
Epoch completed in 112.9 seconds

Epoch 23 - Loss: 0.0051

===== Training Subset Metrics =====
Precision: 0.8685
Recall:    0.6167
F1 Score:  0.7213
PR-AUC:    0.7706
ROC-AUC:   0.9608
Confusion Matrix:
[[19227    66]
 [  271   436]]
========================================

===== Validation Set Metrics =====
Precision: 0.5096
Recall:    0.3669
F1 Score:  0.4266
PR-AUC:    0.4143
ROC-AUC:   0.8518
Detailed PR-AUC: 0.4142
Confusion Matrix:
[[112609   1435]
 [  2573   1491]]
========================================
Epoch completed in 112.7 seconds

Epoch 24 - Loss: 0.0051

===== Training Subset Metrics =====
Precision: 0.9415
Recall:    0.5686
F1 Score:  0.7090
PR-AUC:    0.7850
ROC-AUC:   0.9634
Confusion Matrix:
[[19268    25]
 [  305   402]]
========================================

===== Validation Set Metrics =====
Precision: 0.6954
Recall:    0.3078
F1 Score:  0.4267
PR-AUC:    0.4436
ROC-AUC:   0.8415
Detailed PR-AUC: 0.4436
Confusion Matrix:
[[113496    548]
 [  2813   1251]]
========================================
New best model saved with PR-AUC: 0.4436
Epoch completed in 112.6 seconds

Epoch 25 - Loss: 0.0050

===== Training Subset Metrics =====
Precision: 0.9033
Recall:    0.6209
F1 Score:  0.7360
PR-AUC:    0.7906
ROC-AUC:   0.9644
Confusion Matrix:
[[19246    47]
 [  268   439]]
========================================

===== Validation Set Metrics =====
Precision: 0.5852
Recall:    0.3533
F1 Score:  0.4406
PR-AUC:    0.4254
ROC-AUC:   0.8407
Detailed PR-AUC: 0.4254
Confusion Matrix:
[[113026   1018]
 [  2628   1436]]
========================================
Epoch completed in 110.6 seconds

Epoch 26 - Loss: 0.0049

===== Training Subset Metrics =====
Precision: 0.9095
Recall:    0.6110
F1 Score:  0.7310
PR-AUC:    0.7928
ROC-AUC:   0.9663
Confusion Matrix:
[[19250    43]
 [  275   432]]
========================================

===== Validation Set Metrics =====
Precision: 0.5731
Recall:    0.3425
F1 Score:  0.4288
PR-AUC:    0.4183
ROC-AUC:   0.8336
Detailed PR-AUC: 0.4182
Confusion Matrix:
[[113007   1037]
 [  2672   1392]]
========================================
Epoch completed in 110.4 seconds

Epoch 27 - Loss: 0.0048

===== Training Subset Metrics =====
Precision: 0.8882
Recall:    0.6294
F1 Score:  0.7368
PR-AUC:    0.7933
ROC-AUC:   0.9654
Confusion Matrix:
[[19237    56]
 [  262   445]]
========================================

===== Validation Set Metrics =====
Precision: 0.5752
Recall:    0.3511
F1 Score:  0.4361
PR-AUC:    0.4199
ROC-AUC:   0.8303
Detailed PR-AUC: 0.4199
Confusion Matrix:
[[112990   1054]
 [  2637   1427]]
========================================
Epoch completed in 110.8 seconds

Epoch 28 - Loss: 0.0048

===== Training Subset Metrics =====
Precision: 0.9399
Recall:    0.6195
F1 Score:  0.7468
PR-AUC:    0.7983
ROC-AUC:   0.9672
Confusion Matrix:
[[19265    28]
 [  269   438]]
========================================

===== Validation Set Metrics =====
Precision: 0.5918
Recall:    0.3260
F1 Score:  0.4204
PR-AUC:    0.4084
ROC-AUC:   0.8268
Detailed PR-AUC: 0.4083
Confusion Matrix:
[[113130    914]
 [  2739   1325]]
========================================
Epoch completed in 111.8 seconds

Epoch 29 - Loss: 0.0047

===== Training Subset Metrics =====
Precision: 0.8994
Recall:    0.6322
F1 Score:  0.7425
PR-AUC:    0.8007
ROC-AUC:   0.9671
Confusion Matrix:
[[19243    50]
 [  260   447]]
========================================

===== Validation Set Metrics =====
Precision: 0.5948
Recall:    0.3450
F1 Score:  0.4367
PR-AUC:    0.4216
ROC-AUC:   0.8249
Detailed PR-AUC: 0.4216
Confusion Matrix:
[[113089    955]
 [  2662   1402]]
========================================
Epoch completed in 112.0 seconds

Epoch 30 - Loss: 0.0046

===== Training Subset Metrics =====
Precision: 0.9465
Recall:    0.6252
F1 Score:  0.7530
PR-AUC:    0.8118
ROC-AUC:   0.9714
Confusion Matrix:
[[19268    25]
 [  265   442]]
========================================

===== Validation Set Metrics =====
Precision: 0.6352
Recall:    0.3312
F1 Score:  0.4354
PR-AUC:    0.4307
ROC-AUC:   0.8270
Detailed PR-AUC: 0.4307
Confusion Matrix:
[[113271    773]
 [  2718   1346]]
========================================
Epoch completed in 111.1 seconds

Epoch 31 - Loss: 0.0046

===== Training Subset Metrics =====
Precision: 0.9103
Recall:    0.6605
F1 Score:  0.7656
PR-AUC:    0.8178
ROC-AUC:   0.9717
Confusion Matrix:
[[19247    46]
 [  240   467]]
========================================

===== Validation Set Metrics =====
Precision: 0.5952
Recall:    0.3492
F1 Score:  0.4401
PR-AUC:    0.4125
ROC-AUC:   0.8192
Detailed PR-AUC: 0.4124
Confusion Matrix:
[[113079    965]
 [  2645   1419]]
========================================
Epoch completed in 110.8 seconds

Epoch 32 - Loss: 0.0046

===== Training Subset Metrics =====
Precision: 0.9513
Recall:    0.6082
F1 Score:  0.7420
PR-AUC:    0.8139
ROC-AUC:   0.9710
Confusion Matrix:
[[19271    22]
 [  277   430]]
========================================

===== Validation Set Metrics =====
Precision: 0.6610
Recall:    0.3157
F1 Score:  0.4273
PR-AUC:    0.4186
ROC-AUC:   0.8178
Detailed PR-AUC: 0.4186
Confusion Matrix:
[[113386    658]
 [  2781   1283]]
========================================
Early stopping at epoch 32

===== Final Evaluation =====

===== Full Training Set Metrics =====
Precision: 0.9476
Recall:    0.5551
F1 Score:  0.7001
PR-AUC:    0.7856
ROC-AUC:   0.9644
Confusion Matrix:
[[455323    510]
 [  7385   9214]]
========================================

===== Full Validation Set Metrics =====
Precision: 0.7010
Recall:    0.3063
F1 Score:  0.4264
PR-AUC:    0.4426
ROC-AUC:   0.8403
Detailed PR-AUC: 0.4426
Confusion Matrix:
[[113513    531]
 [  2819   1245]]
========================================
Saving preprocessing artifacts...

===== Final Training Results =====

===== Full Training Set Metrics =====
Precision: 0.9476
Recall:    0.5551
F1 Score:  0.7001
PR-AUC:    0.7856
ROC-AUC:   0.9644
Confusion Matrix:
[[455323    510]
 [  7385   9214]]
========================================

===== Full Validation Set Metrics =====
Precision: 0.7010
Recall:    0.3063
F1 Score:  0.4264
PR-AUC:    0.4426
ROC-AUC:   0.8403
Detailed PR-AUC: 0.4426
Confusion Matrix:
[[113513    531]
 [  2819   1245]]
========================================

===== Running Inference on Test Data =====
Loaded identity data with 141907 rows
Processing test chunk 1
Processing test chunk 2
Processing test chunk 3
Processing test chunk 4
Processing test chunk 5
Processing test chunk 6
Building test graph...
Building graph with 506691 transactions
Applying target encoding with pre-trained encoders...
Building temporal edges...
Added 115175088 temporal edges
Graph built in 157.2 seconds
Test graph metadata: HeteroData(
  transaction={
    x=[506691, 437],
    transaction_id=[506691],
  },
  card={ x=[14326, 1] },
  addr={ x=[394, 1] },
  email={ x=[721, 1] },
  device={ x=[2226, 1] },
  product={ x=[5, 1] },
  (transaction, to_card, card)={ edge_index=[2, 506691] },
  (card, from_card, transaction)={ edge_index=[2, 506691] },
  (transaction, to_addr, addr)={ edge_index=[2, 441082] },
  (addr, from_addr, transaction)={ edge_index=[2, 441082] },
  (transaction, to_email, email)={ edge_index=[2, 448389] },
  (email, from_email, transaction)={ edge_index=[2, 448389] },
  (transaction, to_device, device)={ edge_index=[2, 115057] },
  (device, from_device, transaction)={ edge_index=[2, 115057] },
  (transaction, to_product, product)={ edge_index=[2, 506691] },
  (product, from_product, transaction)={ edge_index=[2, 506691] },
  (transaction, temporal, transaction)={ edge_index=[2, 115175088] }
)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_35/351097643.py in <cell line: 0>()
    959     # Inference
    960     print("\n===== Running Inference on Test Data =====")
--> 961     submission = predict_fraud(model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts)
    962 
    963     # Save submission

/tmp/ipykernel_35/351097643.py in predict_fraud(model, transaction_path, identity_path, artifacts)
    912 
    913     # Create submission
--> 914     submission = pd.DataFrame({
    915         'TransactionID': transaction_ids,
    916         'isFraud': test_probs

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, GCNConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, average_precision_score, 
                            precision_recall_curve, auc, roc_auc_score)
import gc
import warnings
from collections import defaultdict
import time
import joblib
from category_encoders import TargetEncoder
import os
warnings.filterwarnings('ignore')





class FraudFocusedGraphBuilder:
    def __init__(self, entity_types, cat_features, inference_mode=False, 
                 target_encoders=None, scaler=None, num_medians=None):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.tx_times = []
        self.inference_mode = inference_mode
        self.target_encoders = target_encoders
        self.scaler = scaler
        self.num_medians = num_medians
        
        
        self.tx_features = []
        if not self.inference_mode:
            self.tx_labels = []
            self.fraud_indices = []
        self.tx_categorical = {col: [] for col in cat_features}
        self.transaction_ids = []
        self.num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.fraud_entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        self.transaction_ids.append(batch_df['TransactionID'].values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        
        if not self.inference_mode:
            labels = batch_df['isFraud'].values
            self.tx_labels.append(labels)
            fraud_mask = (labels == 1)
            if np.any(fraud_mask):
                self.fraud_indices.extend(tx_indices[fraud_mask])
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1
                
                
                if not self.inference_mode and tx_idx in self.fraud_indices:
                    self.fraud_entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        elif et == 'id':
            
            id_cols = [f'id_{i:02d}' for i in range(12, 39)]
            
            for col in id_cols:
                if col not in batch_df:
                    batch_df[col] = 'MISSING'
            
            return batch_df[id_cols].fillna('MISSING').astype(str).apply(
                lambda row: '_'.join(row.values), axis=1
            ).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation with enhanced fraud-specific features"""
        features = []
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        
        features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        for col in self.num_cols:
            if col in batch_df:
                if self.inference_mode and self.num_medians is not None and col in self.num_medians:
                    median_val = self.num_medians[col]
                else:
                    median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding with fraud focus"""
        if self.inference_mode:
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            encoded_features = []
            for col in self.cat_features:
                encoded = self.target_encoders[col].transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            return np.column_stack(encoded_features)
        else:
            
            tx_labels = np.concatenate(self.tx_labels)
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = np.zeros(self.transaction_counter, dtype=bool)
            train_mask[:num_train] = True
            
            
            self.target_encoders = {}
            encoded_features = []
            for col in self.cat_features:
                
                encoder = TargetEncoder(smoothing=50, min_samples_leaf=100)  
                encoder.fit(
                    cat_data[col][train_mask], 
                    tx_labels[train_mask]
                )
                self.target_encoders[col] = encoder
                
                
                encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            
            return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building fraud-focused graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        
        
        if not self.inference_mode:
            self.num_medians = {}
            for i, col in enumerate(self.num_cols):
                if i < num_features.shape[1]:
                    self.num_medians[col] = np.median(num_features[:, i])
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        if not self.inference_mode:
            self.scaler = StandardScaler()
            tx_features = self.scaler.fit_transform(tx_features)
        else:
            tx_features = self.scaler.transform(tx_features)
            
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        
        if not self.inference_mode:
            tx_labels = np.concatenate(self.tx_labels)
            data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            train_mask[:num_train] = True
            val_mask[num_train:] = True
            data['transaction'].train_mask = train_mask
            data['transaction'].val_mask = val_mask
        
        
        if self.inference_mode:
            tx_ids = np.concatenate(self.transaction_ids)
            data['transaction'].transaction_id = torch.tensor(tx_ids, dtype=torch.long)
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 2), dtype=np.float32)
            for entity_idx in range(num_entities):
                total_count = self.entity_counts[et].get(entity_idx, 0)
                fraud_count = self.fraud_entity_counts[et].get(entity_idx, 0)
                entity_features[entity_idx, 0] = np.log1p(total_count)
                entity_features[entity_idx, 1] = fraud_count / (total_count + 1e-6)  
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building fraud-focused temporal edges...")
        time_edges = []
        time_window = 86400  
        
        
        fraud_time_edges = set()
        for i in self.fraud_indices:
            current_time = self.tx_times[i]
            
            for j in range(max(0, i-1000), min(i+1000, len(self.tx_times))):
                if i == j: 
                    continue
                time_diff = abs(current_time - self.tx_times[j])
                if time_diff <= time_window:
                    fraud_time_edges.add((i, j))
                    fraud_time_edges.add((j, i))
        
        
        sample_size = min(200000, self.transaction_counter)
        sample_indices = np.random.choice(self.transaction_counter, sample_size, replace=False)
        sorted_indices = sorted(sample_indices, key=lambda i: self.tx_times[i])
        sorted_times = [self.tx_times[i] for i in sorted_indices]
        
        for i in range(1, len(sorted_indices)):
            current_idx = sorted_indices[i]
            current_time = sorted_times[i]
            j = i - 1
            while j >= 0 and (current_time - sorted_times[j]) <= time_window:
                if (current_idx, sorted_indices[j]) not in fraud_time_edges:
                    time_edges.append((current_idx, sorted_indices[j]))
                    time_edges.append((sorted_indices[j], current_idx))
                j -= 1
        
        
        all_time_edges = list(fraud_time_edges) + time_edges
        if all_time_edges:
            src, dst = zip(*all_time_edges)
            time_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
            print(f"Added {len(all_time_edges)} temporal edges ({len(fraud_time_edges)} fraud-focused)")
        
        
        if not self.inference_mode and self.fraud_indices:
            print("Building fraud pattern edges...")
            fraud_pattern_edges = []
            
            for i in range(len(self.fraud_indices)):
                for j in range(i+1, min(i+100, len(self.fraud_indices))):
                    idx_i = self.fraud_indices[i]
                    idx_j = self.fraud_indices[j]
                    
                    amt_diff = abs(self.tx_features[idx_i][0] - self.tx_features[idx_j][0])
                    same_product = (self.tx_categorical['ProductCD'][idx_i] == 
                                   self.tx_categorical['ProductCD'][idx_j])
                    if amt_diff < 100 and same_product:
                        fraud_pattern_edges.append((idx_i, idx_j))
                        fraud_pattern_edges.append((idx_j, idx_i))
            
            if fraud_pattern_edges:
                src, dst = zip(*fraud_pattern_edges)
                pattern_edge_index = torch.tensor([src, dst], dtype=torch.long)
                data['transaction', 'fraud_pattern', 'transaction'].edge_index = pattern_edge_index
                print(f"Added {len(fraud_pattern_edges)} fraud pattern edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times, self.tx_categorical
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class FraudAttentionLayer(nn.Module):
    """Specialized attention layer for fraud detection"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.scale = in_channels ** -0.5
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

class FraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product', 'id']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        
        self.fraud_attention = FraudAttentionLayer(hidden_channels)
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                conv_dict[('transaction', f'to_{et}', et)] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                conv_dict[(et, f'from_{et}', 'transaction')] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
            
            
            conv_dict[('transaction', 'temporal', 'transaction')] = GCNConv(
                hidden_channels, hidden_channels)
            
            
            conv_dict[('transaction', 'fraud_pattern', 'transaction')] = GCNConv(
                hidden_channels, hidden_channels)
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels//2)
        )
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, data):
        
        x_dict = {
            'transaction': F.elu(self.tx_proj(data['transaction'].x))
        }
        
        
        x_dict['transaction'] = self.fraud_attention(x_dict['transaction'])
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                x_dict[et] = self.entity_proj(data[et].x)
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=data['transaction'].x.device)
        
        
        fraud_features = []
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}
                fraud_features.append(x_dict['transaction'])
            except Exception as e:
                continue
        
        
        if fraud_features:
            transaction_features = torch.cat(fraud_features, dim=1)
        else:
            transaction_features = x_dict['transaction']
        
        
        if ('transaction', 'fraud_pattern', 'transaction') in data.edge_index_dict:
            pattern_edge_index = data.edge_index_dict[('transaction', 'fraud_pattern', 'transaction')]
            pattern_features = self.pattern_detector(transaction_features)
            pattern_out = F.elu(pattern_features[pattern_edge_index[0]] + pattern_features[pattern_edge_index[1]])
            pattern_out = torch.mean(pattern_out, dim=0, keepdim=True).repeat(transaction_features.size(0), 1)
        else:
            pattern_out = torch.zeros_like(transaction_features[:, :self.hidden_channels//2])
        
        
        combined = torch.cat([transaction_features, pattern_out], dim=1)
        return self.head(combined).squeeze()





class FraudFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss





def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product', 'id']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD'] + [f'id_{i:02d}' for i in range(12, 39)]:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = FraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=3
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [30, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [30, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    train_indices = data['transaction'].train_mask.nonzero().squeeze()
    subset_size = min(20000, len(train_indices))
    subset_indices = torch.randperm(len(train_indices))[:subset_size]
    train_subset_mask = torch.zeros_like(data['transaction'].train_mask).bool()
    train_subset_mask[train_indices[subset_indices]] = True
    
    train_metrics_loader = NeighborLoader(
        data,
        num_neighbors={key: [30, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', train_subset_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    
    criterion = FraudFocalLoss(alpha=0.75, gamma=3.0)
    print("Using FraudFocalLoss with alpha=0.75, gamma=3.0")
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    
    history = {'train': [], 'val': []}
    best_recall = 0
    no_improve = 0
    max_epochs = 50
    no_improve_threshold = 10
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        train_metrics = evaluate_metrics(train_metrics_loader, model, mode='train')
        val_metrics = evaluate_metrics(val_loader, model, mode='val')
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        
        print(f"\nEpoch {epoch} - Loss: {total_loss/batch_count:.4f}")
        print_metrics(train_metrics, "Training Subset")
        print_metrics(val_metrics, "Validation Set")
        
        
        current_recall = val_metrics.get('recall', 0)
        if current_recall > best_recall:
            best_recall = current_recall
            no_improve = 0
            torch.save(model.state_dict(), 'best_fraud_model.pt')
            print(f"New best model saved with Recall: {best_recall:.4f}")
        else:
            no_improve += 1
            
        
        if no_improve >= no_improve_threshold:
            print(f"Early stopping at epoch {epoch}")
            break
            
        
        scheduler.step(val_metrics.get('pr_auc', 0))
            
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.1f} seconds")
    
    
    print("\n===== Final Evaluation =====")
    
    
    model.load_state_dict(torch.load('best_fraud_model.pt'))
    
    
    full_train_loader = NeighborLoader(
        data,
        num_neighbors={key: [30, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=False
    )
    full_train_metrics = evaluate_metrics(full_train_loader, model, mode='train')
    print_metrics(full_train_metrics, "Full Training Set")
    
    
    full_val_metrics = evaluate_metrics(val_loader, model, mode='val')
    print_metrics(full_val_metrics, "Full Validation Set")
    
    
    print("Saving preprocessing artifacts...")
    artifacts = {
        'target_encoders': graph_builder.target_encoders,
        'scaler': graph_builder.scaler,
        'num_medians': graph_builder.num_medians
    }
    joblib.dump(artifacts, 'fraud_inference_artifacts.pkl')
    
    return model, full_train_metrics, full_val_metrics, artifacts





def predict_fraud(model, transaction_path, identity_path, artifacts):
    """Predict fraud probabilities for test data (FIXED)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product', 'id']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES,
        inference_mode=True,
        target_encoders=artifacts['target_encoders'],
        scaler=artifacts['scaler'],
        num_medians=artifacts['num_medians']
    )
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing test chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD'] + [f'id_{i:02d}' for i in range(12, 39)]:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building test graph...")
    test_data = graph_builder.build_graph()
    test_data = test_data.to(device)
    print(f"Test graph metadata: {test_data}")
    
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [30, 20] for key in test_data.edge_index_dict},
        input_nodes=('transaction', torch.arange(test_data['transaction'].x.size(0))),
        batch_size=4096,
        shuffle=False
    )
    
    
    all_probs = []
    transaction_ids = []
    with torch.no_grad():
        for batch in test_loader:
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits).cpu().numpy()
            all_probs.append(probs)
            
            
            seed_ids = batch['transaction'].transaction_id[:batch['transaction'].batch_size].cpu().numpy()
            transaction_ids.append(seed_ids)
    
    
    test_probs = np.concatenate(all_probs)
    transaction_ids = np.concatenate(transaction_ids)
    
    
    submission = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': test_probs
    })
    
    return submission





def evaluate_metrics(loader, model, mode='val'):
    """Evaluate model and return comprehensive classification metrics"""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits)
            preds = (probs >= 0.5).long()
            labels = batch['transaction'].y[:batch['transaction'].batch_size]
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    if not all_probs:
        return {}
    
    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    
    if len(np.unique(labels)) < 2:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': np.zeros((2, 2)),
            'pr_auc': 0.0,
            'roc_auc': 0.5
        }
    
    
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc
    }
    
    
    if mode != 'train':
        precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
        metrics['pr_curve'] = (precision_curve, recall_curve)
        metrics['pr_auc_detailed'] = auc(recall_curve, precision_curve)
    
    return metrics

def print_metrics(metrics, set_name='Dataset'):
    """Print formatted classification metrics"""
    print(f"\n===== {set_name} Metrics =====")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
    print(f"PR-AUC:    {metrics.get('pr_auc', 0):.4f}")
    print(f"ROC-AUC:   {metrics.get('roc_auc', 0.5):.4f}")
    
    if 'pr_auc_detailed' in metrics:
        print(f"Detailed PR-AUC: {metrics['pr_auc_detailed']:.4f}")
    
    print("Confusion Matrix:")
    print(metrics.get('confusion_matrix', np.zeros((2, 2))))
    print("="*40)





if __name__ == "__main__":
    
    TRAIN_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    TEST_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
    
    
    if not os.path.exists('best_fraud_model.pt'):
        print("Training fraud-focused model...")
        model, train_metrics, val_metrics, artifacts = train_fraud_model(
            TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
        print("\n===== Final Training Results =====")
        print_metrics(train_metrics, "Full Training Set")
        print_metrics(val_metrics, "Full Validation Set")
    else:
        print("Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sample = pd.read_csv(TRAIN_TRANSACTION_PATH, nrows=1)
        CAT_FEATURES = [
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo',
            'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
            'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
            'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
        ]
        tx_feature_size = len(sample.columns) + len(CAT_FEATURES)  
        model = FraudGNN(tx_feature_size, 128, 3).to(device)
        model.load_state_dict(torch.load('best_fraud_model.pt'))
        artifacts = joblib.load('fraud_inference_artifacts.pkl')
    
    
    print("\n===== Running Inference on Test Data =====")
    submission = predict_fraud(model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts)
    
    
    submission.to_csv('fraud_submission.csv', index=False)
    print("Submission saved with shape:", submission.shape)
    print("First 5 predictions:")
    print(submission.head())


gc.collect()
torch.cuda.empty_cache()


1


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, average_precision_score, 
                            precision_recall_curve, auc, roc_auc_score)
import gc
import warnings
from collections import defaultdict
import time
import joblib
from category_encoders import TargetEncoder
import os
import bisect
warnings.filterwarnings('ignore')





class FraudFocusedGraphBuilder:
    def __init__(self, entity_types, cat_features, inference_mode=False, 
                 target_encoders=None, scaler=None, num_medians=None):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.tx_times = []
        self.inference_mode = inference_mode
        self.target_encoders = target_encoders
        self.scaler = scaler
        self.num_medians = num_medians
        
        
        self.tx_features = []
        if not self.inference_mode:
            self.tx_labels = []
            self.fraud_indices = []
        self.tx_categorical = {col: [] for col in cat_features}
        self.transaction_ids = []
        self.num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.fraud_entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8
        
        
        self.time_index = None

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.tx_times.extend(batch_df['TransactionDT'].fillna(0).values)
        
        
        self.transaction_ids.append(batch_df['TransactionID'].values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        
        if not self.inference_mode:
            labels = batch_df['isFraud'].values
            self.tx_labels.append(labels)
            fraud_mask = (labels == 1)
            if np.any(fraud_mask):
                self.fraud_indices.extend(tx_indices[fraud_mask])
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1
                
                
                if not self.inference_mode and tx_idx in self.fraud_indices:
                    self.fraud_entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation with enhanced fraud-specific features"""
        features = []
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        
        features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        
        for col in self.num_cols:
            if col in batch_df:
                if self.inference_mode and self.num_medians is not None and col in self.num_medians:
                    median_val = self.num_medians[col]
                else:
                    median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding with fraud focus"""
        if self.inference_mode:
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            encoded_features = []
            for col in self.cat_features:
                encoded = self.target_encoders[col].transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            return np.column_stack(encoded_features)
        else:
            
            tx_labels = np.concatenate(self.tx_labels)
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = np.zeros(self.transaction_counter, dtype=bool)
            train_mask[:num_train] = True
            
            
            self.target_encoders = {}
            encoded_features = []
            for col in self.cat_features:
                
                encoder = TargetEncoder(smoothing=50, min_samples_leaf=100)
                encoder.fit(
                    cat_data[col][train_mask], 
                    tx_labels[train_mask]
                )
                self.target_encoders[col] = encoder
                
                
                encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            
            return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building fraud-focused graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        
        
        if not self.inference_mode:
            self.num_medians = {}
            for i, col in enumerate(self.num_cols):
                if i < num_features.shape[1]:
                    self.num_medians[col] = np.median(num_features[:, i])
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        if not self.inference_mode:
            self.scaler = StandardScaler()
            tx_features = self.scaler.fit_transform(tx_features)
        else:
            tx_features = self.scaler.transform(tx_features)
            
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        
        if not self.inference_mode:
            tx_labels = np.concatenate(self.tx_labels)
            data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            train_mask[:num_train] = True
            val_mask[num_train:] = True
            data['transaction'].train_mask = train_mask
            data['transaction'].val_mask = val_mask
        
        
        if self.inference_mode:
            tx_ids = np.concatenate(self.transaction_ids)
            data['transaction'].transaction_id = torch.tensor(tx_ids, dtype=torch.long)
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 2), dtype=np.float32)
            for entity_idx in range(num_entities):
                total_count = self.entity_counts[et].get(entity_idx, 0)
                fraud_count = self.fraud_entity_counts[et].get(entity_idx, 0)
                entity_features[entity_idx, 0] = np.log1p(total_count)
                entity_features[entity_idx, 1] = fraud_count / (total_count + 1e-6)  
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        if not self.inference_mode and self.fraud_indices:
            print("Building optimized fraud-focused temporal edges...")
            
            
            time_index = list(enumerate(self.tx_times))
            time_index.sort(key=lambda x: x[1])
            sorted_times = [t for idx, t in time_index]
            sorted_indices = [idx for idx, t in time_index]
            
            time_edges = set()
            time_window = 86400  
            
            
            for fraud_idx in self.fraud_indices:
                current_time = self.tx_times[fraud_idx]
                
                
                pos = bisect.bisect_left(sorted_times, current_time)
                
                
                left = pos - 1
                while left >= 0 and (current_time - sorted_times[left]) <= time_window:
                    j = sorted_indices[left]
                    if fraud_idx != j:
                        time_edges.add((fraud_idx, j))
                        time_edges.add((j, fraud_idx))
                    left -= 1
                
                
                right = pos
                while right < len(sorted_times) and (sorted_times[right] - current_time) <= time_window:
                    j = sorted_indices[right]
                    if fraud_idx != j:
                        time_edges.add((fraud_idx, j))
                        time_edges.add((j, fraud_idx))
                    right += 1
            
            if time_edges:
                src, dst = zip(*time_edges)
                time_edge_index = torch.tensor([src, dst], dtype=torch.long)
                data['transaction', 'temporal', 'transaction'].edge_index = time_edge_index
                print(f"Added {len(time_edges)} fraud-focused temporal edges")
        
        
        if not self.inference_mode and self.fraud_indices:
            print("Building fraud pattern edges...")
            fraud_pattern_edges = []
            
            
            for i in range(len(self.fraud_indices)):
                for j in range(i+1, min(i+100, len(self.fraud_indices))):
                    idx_i = self.fraud_indices[i]
                    idx_j = self.fraud_indices[j]
                    
                    
                    same_card = (self.tx_categorical['card1'][idx_i] == self.tx_categorical['card1'][idx_j])
                    amt_diff = abs(self.tx_features[idx_i][0] - self.tx_features[idx_j][0])
                    if same_card and amt_diff < 100:
                        fraud_pattern_edges.append((idx_i, idx_j))
                        fraud_pattern_edges.append((idx_j, idx_i))
            
            if fraud_pattern_edges:
                src, dst = zip(*fraud_pattern_edges)
                pattern_edge_index = torch.tensor([src, dst], dtype=torch.long)
                data['transaction', 'fraud_pattern', 'transaction'].edge_index = pattern_edge_index
                print(f"Added {len(fraud_pattern_edges)} fraud pattern edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_times, self.tx_categorical
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class FraudAttentionLayer(nn.Module):
    """Attention layer focusing on suspicious patterns"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.scale = in_channels ** -0.5
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

class FraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        
        self.fraud_attention = FraudAttentionLayer(hidden_channels)
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                conv_dict[('transaction', f'to_{et}', et)] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
                conv_dict[(et, f'from_{et}', 'transaction')] = GATConv(
                    hidden_channels, hidden_channels//4, heads=4, concat=True, add_self_loops=False)
            
            
            conv_dict[('transaction', 'temporal', 'transaction')] = GCNConv(
                hidden_channels, hidden_channels)
            
            
            conv_dict[('transaction', 'fraud_pattern', 'transaction')] = GCNConv(
                hidden_channels, hidden_channels)
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels//2)
        )
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, data):
        
        x_dict = {
            'transaction': F.elu(self.tx_proj(data['transaction'].x))
        }
        
        
        x_dict['transaction'] = self.fraud_attention(x_dict['transaction'])
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                x_dict[et] = self.entity_proj(data[et].x)
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=data['transaction'].x.device)
        
        
        fraud_features = []
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}
                fraud_features.append(x_dict['transaction'])
            except Exception as e:
                continue
        
        
        if fraud_features:
            transaction_features = torch.cat(fraud_features, dim=1)
        else:
            transaction_features = x_dict['transaction']
        
        
        if ('transaction', 'fraud_pattern', 'transaction') in data.edge_index_dict:
            pattern_edge_index = data.edge_index_dict[('transaction', 'fraud_pattern', 'transaction')]
            pattern_features = self.pattern_detector(transaction_features)
            pattern_out = F.elu(pattern_features[pattern_edge_index[0]] + pattern_features[pattern_edge_index[1]])
            pattern_out = torch.mean(pattern_out, dim=0, keepdim=True).repeat(transaction_features.size(0), 1)
        else:
            pattern_out = torch.zeros_like(transaction_features[:, :self.hidden_channels//2])
        
        
        combined = torch.cat([transaction_features, pattern_out], dim=1)
        return self.head(combined).squeeze()





def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    model = FraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=3
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [30, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [30, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    train_indices = data['transaction'].train_mask.nonzero().squeeze()
    fraud_train_indices = data['transaction'].y[data['transaction'].train_mask].nonzero().squeeze()
    subset_size = min(20000, len(train_indices))
    subset_indices = torch.cat([
        fraud_train_indices,
        torch.randperm(len(train_indices))[:subset_size - len(fraud_train_indices)]
    ])
    train_subset_mask = torch.zeros_like(data['transaction'].train_mask).bool()
    train_subset_mask[train_indices[subset_indices]] = True
    
    train_metrics_loader = NeighborLoader(
        data,
        num_neighbors={key: [30, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', train_subset_mask),
        batch_size=2048,
        shuffle=False
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))
    print("Using weighted BCE loss with pos_weight=20.0")
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    
    history = {'train': [], 'val': []}
    best_recall = 0
    no_improve = 0
    max_epochs = 50
    no_improve_threshold = 10
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        train_metrics = evaluate_metrics(train_metrics_loader, model, mode='train')
        val_metrics = evaluate_metrics(val_loader, model, mode='val')
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        
        print(f"\nEpoch {epoch} - Loss: {total_loss/batch_count:.4f}")
        print_metrics(train_metrics, "Training Subset")
        print_metrics(val_metrics, "Validation Set")
        
        
        current_recall = val_metrics.get('recall', 0)
        if current_recall > best_recall:
            best_recall = current_recall
            no_improve = 0
            torch.save(model.state_dict(), 'best_fraud_model.pt')
            print(f"New best model saved with Recall: {best_recall:.4f}")
        else:
            no_improve += 1
            
        
        if no_improve >= no_improve_threshold:
            print(f"Early stopping at epoch {epoch}")
            break
            
        
        scheduler.step(val_metrics.get('pr_auc', 0))
            
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.1f} seconds")
    
    
    print("\n===== Final Evaluation =====")
    
    
    model.load_state_dict(torch.load('best_fraud_model.pt'))
    
    
    full_train_loader = NeighborLoader(
        data,
        num_neighbors={key: [30, 20] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=2048,
        shuffle=False
    )
    full_train_metrics = evaluate_metrics(full_train_loader, model, mode='train')
    print_metrics(full_train_metrics, "Full Training Set")
    
    
    full_val_metrics = evaluate_metrics(val_loader, model, mode='val')
    print_metrics(full_val_metrics, "Full Validation Set")
    
    
    print("Saving preprocessing artifacts...")
    artifacts = {
        'target_encoders': graph_builder.target_encoders,
        'scaler': graph_builder.scaler,
        'num_medians': graph_builder.num_medians
    }
    joblib.dump(artifacts, 'fraud_inference_artifacts.pkl')
    
    return model, full_train_metrics, full_val_metrics, artifacts

def predict_fraud(model, transaction_path, identity_path, artifacts):
    """Predict fraud probabilities for test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
        'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES,
        inference_mode=True,
        target_encoders=artifacts['target_encoders'],
        scaler=artifacts['scaler'],
        num_medians=artifacts['num_medians']
    )
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing test chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building test graph...")
    test_data = graph_builder.build_graph()
    test_data = test_data.to(device)
    print(f"Test graph metadata: {test_data}")
    
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [30, 20] for key in test_data.edge_index_dict},
        input_nodes=('transaction', torch.arange(test_data['transaction'].x.size(0))),
        batch_size=4096,
        shuffle=False
    )
    
    
    all_probs = []
    transaction_ids = []
    with torch.no_grad():
        for batch in test_loader:
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits).cpu().numpy()
            all_probs.append(probs)
            
            
            seed_ids = batch['transaction'].transaction_id[:batch['transaction'].batch_size].cpu().numpy()
            transaction_ids.append(seed_ids)
    
    
    test_probs = np.concatenate(all_probs)
    transaction_ids = np.concatenate(transaction_ids)
    
    
    submission = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': test_probs
    })
    
    return submission





if __name__ == "__main__":
    
    TRAIN_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    TEST_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
    
    
    if not os.path.exists('best_fraud_model.pt'):
        print("Training fraud-focused model...")
        model, train_metrics, val_metrics, artifacts = train_fraud_model(
            TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
        print("\n===== Final Training Results =====")
        print_metrics(train_metrics, "Full Training Set")
        print_metrics(val_metrics, "Full Validation Set")
    else:
        print("Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sample = pd.read_csv(TRAIN_TRANSACTION_PATH, nrows=1)
        CAT_FEATURES = [
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo',
            'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 
            'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
            'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
        ]
        tx_feature_size = len(sample.columns) + len(CAT_FEATURES)  
        model = FraudGNN(tx_feature_size, 128, 3).to(device)
        model.load_state_dict(torch.load('best_fraud_model.pt'))
        artifacts = joblib.load('fraud_inference_artifacts.pkl')
    
    
    print("\n===== Running Inference on Test Data =====")
    submission = predict_fraud(model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts)
    
    
    submission.to_csv('fraud_submission.csv', index=False)
    print("Submission saved with shape:", submission.shape)
    print("First 5 predictions:")
    print(submission.head())


# еше быстрее упало





Processing 





import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, average_precision_score, 
                            precision_recall_curve, auc, roc_auc_score)
import gc
import warnings
from collections import defaultdict
import time
import joblib
from category_encoders import TargetEncoder
import os
warnings.filterwarnings('ignore')





class FraudFocusedGraphBuilder:
    def __init__(self, entity_types, cat_features, inference_mode=False, 
                 target_encoders=None, scaler=None, num_medians=None):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.inference_mode = inference_mode
        self.target_encoders = target_encoders
        self.scaler = scaler
        self.num_medians = num_medians
        
        
        self.tx_features = []
        if not self.inference_mode:
            self.tx_labels = []
            self.fraud_indices = []
        self.tx_categorical = {col: [] for col in cat_features}
        self.transaction_ids = []
        self.num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.fraud_entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8
        
        
        self.fraud_patterns = defaultdict(list) if not inference_mode else None

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.transaction_ids.append(batch_df['TransactionID'].values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        
        if not self.inference_mode:
            labels = batch_df['isFraud'].values
            self.tx_labels.append(labels)
            fraud_mask = (labels == 1)
            if np.any(fraud_mask):
                fraud_indices = tx_indices[fraud_mask]
                self.fraud_indices.extend(fraud_indices)
                
                
                for idx in fraud_indices:
                    card = batch_df.loc[idx-start_idx, 'card1']
                    product = batch_df.loc[idx-start_idx, 'ProductCD']
                    amount = batch_df.loc[idx-start_idx, 'TransactionAmt']
                    self.fraud_patterns[(card, product, amount)].append(idx)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1
                
                
                if not self.inference_mode and tx_idx in self.fraud_indices:
                    self.fraud_entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation with enhanced fraud-specific features"""
        features = []
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        
        for col in self.num_cols:
            if col in batch_df:
                if self.inference_mode and self.num_medians is not None and col in self.num_medians:
                    median_val = self.num_medians[col]
                else:
                    median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding with fraud focus"""
        if self.inference_mode:
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            encoded_features = []
            for col in self.cat_features:
                encoded = self.target_encoders[col].transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            return np.column_stack(encoded_features)
        else:
            
            tx_labels = np.concatenate(self.tx_labels)
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = np.zeros(self.transaction_counter, dtype=bool)
            train_mask[:num_train] = True
            
            
            self.target_encoders = {}
            encoded_features = []
            for col in self.cat_features:
                
                encoder = TargetEncoder(smoothing=50, min_samples_leaf=100)
                encoder.fit(
                    cat_data[col][train_mask], 
                    tx_labels[train_mask]
                )
                self.target_encoders[col] = encoder
                
                
                encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            
            return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building optimized fraud-focused graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        
        
        if not self.inference_mode:
            self.num_medians = {}
            for i, col in enumerate(self.num_cols):
                if i < num_features.shape[1]:
                    self.num_medians[col] = np.median(num_features[:, i])
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        if not self.inference_mode:
            self.scaler = StandardScaler()
            tx_features = self.scaler.fit_transform(tx_features)
        else:
            tx_features = self.scaler.transform(tx_features)
            
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        
        if not self.inference_mode:
            tx_labels = np.concatenate(self.tx_labels)
            data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            train_mask[:num_train] = True
            val_mask[num_train:] = True
            data['transaction'].train_mask = train_mask
            data['transaction'].val_mask = val_mask
        
        
        if self.inference_mode:
            tx_ids = np.concatenate(self.transaction_ids)
            data['transaction'].transaction_id = torch.tensor(tx_ids, dtype=torch.long)
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 2), dtype=np.float32)
            for entity_idx in range(num_entities):
                total_count = self.entity_counts[et].get(entity_idx, 0)
                fraud_count = self.fraud_entity_counts[et].get(entity_idx, 0)
                entity_features[entity_idx, 0] = np.log1p(total_count)
                entity_features[entity_idx, 1] = fraud_count / (total_count + 1e-6)  
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        if not self.inference_mode and self.fraud_patterns:
            print("Building fraud pattern edges...")
            fraud_pattern_edges = set()
            
            
            for pattern, indices in self.fraud_patterns.items():
                if len(indices) > 1:
                    
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            fraud_pattern_edges.add((indices[i], indices[j]))
                            fraud_pattern_edges.add((indices[j], indices[i]))
            
            if fraud_pattern_edges:
                src, dst = zip(*fraud_pattern_edges)
                pattern_edge_index = torch.tensor([src, dst], dtype=torch.long)
                data['transaction', 'fraud_pattern', 'transaction'].edge_index = pattern_edge_index
                print(f"Added {len(fraud_pattern_edges)} fraud pattern edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_categorical
        if hasattr(self, 'tx_times'):
            del self.tx_times
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class FraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                
                conv_dict[('transaction', f'to_{et}', et)] = SAGEConv(
                    hidden_channels, hidden_channels)
                conv_dict[(et, f'from_{et}', 'transaction')] = SAGEConv(
                    hidden_channels, hidden_channels)
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, data):
        
        x_dict = {
            'transaction': F.elu(self.tx_proj(data['transaction'].x))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                x_dict[et] = self.entity_proj(data[et].x)
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=data['transaction'].x.device)
        
        
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            except Exception as e:
                continue
        
        
        return self.head(x_dict['transaction']).squeeze()





def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=1024,  
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=1024,  
        shuffle=False
    )
    
    
    model = FraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=2  
    )
    model = model.to(device)
    
    
    print("Initializing parameters...")
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            model(batch)
            break
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))
    print("Using weighted BCE loss with pos_weight=20.0")
    
    
    best_recall = 0
    no_improve = 0
    max_epochs = 30
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                fraud_logits = model(batch)
                seed_logits = fraud_logits[:batch['transaction'].batch_size]
                val_preds.extend(torch.sigmoid(seed_logits).cpu().numpy())
                val_labels.extend(batch['transaction'].y[:batch['transaction'].batch_size].cpu().numpy())
        
        
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        recall = recall_score(val_labels, val_preds_binary, zero_division=0)
        
        print(f"Epoch {epoch} - Loss: {total_loss/batch_count:.4f}, Val Recall: {recall:.4f}")
        
        
        if recall > best_recall:
            best_recall = recall
            no_improve = 0
            torch.save(model.state_dict(), 'best_fraud_model.pt')
            print(f"New best model saved with recall: {recall:.4f}")
        else:
            no_improve += 1
            
        
        if no_improve >= 5:
            print(f"Early stopping at epoch {epoch}")
            break
            
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.1f} seconds")
    
    
    model.load_state_dict(torch.load('best_fraud_model.pt'))
    
    
    print("Saving preprocessing artifacts...")
    artifacts = {
        'target_encoders': graph_builder.target_encoders,
        'scaler': graph_builder.scaler,
        'num_medians': graph_builder.num_medians
    }
    joblib.dump(artifacts, 'fraud_inference_artifacts.pkl')
    
    return model, artifacts

def predict_fraud(model, transaction_path, identity_path, artifacts):
    """Predict fraud probabilities for test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo'
    ]
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES,
        inference_mode=True,
        target_encoders=artifacts['target_encoders'],
        scaler=artifacts['scaler'],
        num_medians=artifacts['num_medians']
    )
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing test chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building test graph...")
    test_data = graph_builder.build_graph()
    test_data = test_data.to(device)
    print(f"Test graph metadata: {test_data}")
    
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [15, 10] for key in test_data.edge_index_dict},
        input_nodes=('transaction', torch.arange(test_data['transaction'].x.size(0))),
        batch_size=2048,
        shuffle=False
    )
    
    
    all_probs = []
    transaction_ids = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits).cpu().numpy()
            all_probs.append(probs)
            
            
            seed_ids = batch['transaction'].transaction_id[:batch['transaction'].batch_size].cpu().numpy()
            transaction_ids.append(seed_ids)
    
    
    test_probs = np.concatenate(all_probs)
    transaction_ids = np.concatenate(transaction_ids)
    
    
    submission = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': test_probs
    })
    
    return submission





if __name__ == "__main__":
    
    TRAIN_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    TEST_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
    
    
    if not os.path.exists('best_fraud_model.pt'):
        print("Training fraud-focused model...")
        model, artifacts = train_fraud_model(
            TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
    else:
        print("Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        CAT_FEATURES = [
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo'
        ]
        
        
        tx_feature_size = 300 + len(CAT_FEATURES)
        
        model = FraudGNN(tx_feature_size, 128, 2).to(device)
        model.load_state_dict(torch.load('best_fraud_model.pt'))
        artifacts = joblib.load('fraud_inference_artifacts.pkl')
    
    
    print("\n===== Running Inference on Test Data =====")
    submission = predict_fraud(model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts)
    
    
    submission.to_csv('fraud_submission.csv', index=False)
    print("Submission saved with shape:", submission.shape)
    print("First 5 predictions:")
    print(submission.head())



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, average_precision_score, 
                            precision_recall_curve, auc, roc_auc_score,
                            classification_report,roc_curve)
import gc
import warnings
from collections import defaultdict
import time
import joblib
from category_encoders import TargetEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')





class FraudFocusedGraphBuilder:
    def __init__(self, entity_types, cat_features, inference_mode=False, 
                 target_encoders=None, scaler=None, num_medians=None):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.inference_mode = inference_mode
        self.target_encoders = target_encoders
        self.scaler = scaler
        self.num_medians = num_medians
        
        
        self.tx_features = []
        if not self.inference_mode:
            self.tx_labels = []
            self.fraud_indices = []
        self.tx_categorical = {col: [] for col in cat_features}
        self.transaction_ids = []
        self.num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.fraud_entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8
        
        
        self.fraud_patterns = defaultdict(list) if not inference_mode else None

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.transaction_ids.append(batch_df['TransactionID'].values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        
        if not self.inference_mode:
            labels = batch_df['isFraud'].values
            self.tx_labels.append(labels)
            fraud_mask = (labels == 1)
            if np.any(fraud_mask):
                fraud_indices = tx_indices[fraud_mask]
                self.fraud_indices.extend(fraud_indices)
                
                
                for idx in fraud_indices:
                    card = batch_df.loc[idx-start_idx, 'card1']
                    product = batch_df.loc[idx-start_idx, 'ProductCD']
                    amount = batch_df.loc[idx-start_idx, 'TransactionAmt']
                    self.fraud_patterns[(card, product, amount)].append(idx)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1
                
                
                if not self.inference_mode and tx_idx in self.fraud_indices:
                    self.fraud_entity_counts[et][entity_idx] += 1

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation with enhanced fraud-specific features"""
        features = []
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        
        for col in self.num_cols:
            if col in batch_df:
                if self.inference_mode and self.num_medians is not None and col in self.num_medians:
                    median_val = self.num_medians[col]
                else:
                    median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding with fraud focus"""
        if self.inference_mode:
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            encoded_features = []
            for col in self.cat_features:
                encoded = self.target_encoders[col].transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            return np.column_stack(encoded_features)
        else:
            
            tx_labels = np.concatenate(self.tx_labels)
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = np.zeros(self.transaction_counter, dtype=bool)
            train_mask[:num_train] = True
            
            
            self.target_encoders = {}
            encoded_features = []
            for col in self.cat_features:
                
                encoder = TargetEncoder(smoothing=50, min_samples_leaf=100)
                encoder.fit(
                    cat_data[col][train_mask], 
                    tx_labels[train_mask]
                )
                self.target_encoders[col] = encoder
                
                
                encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            
            return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building optimized fraud-focused graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        
        
        if not self.inference_mode:
            self.num_medians = {}
            for i, col in enumerate(self.num_cols):
                if i < num_features.shape[1]:
                    self.num_medians[col] = np.median(num_features[:, i])
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        if not self.inference_mode:
            self.scaler = StandardScaler()
            tx_features = self.scaler.fit_transform(tx_features)
        else:
            tx_features = self.scaler.transform(tx_features)
            
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        
        if not self.inference_mode:
            tx_labels = np.concatenate(self.tx_labels)
            data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            train_mask[:num_train] = True
            val_mask[num_train:] = True
            data['transaction'].train_mask = train_mask
            data['transaction'].val_mask = val_mask
        
        
        if self.inference_mode:
            tx_ids = np.concatenate(self.transaction_ids)
            data['transaction'].transaction_id = torch.tensor(tx_ids, dtype=torch.long)
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 2), dtype=np.float32)
            for entity_idx in range(num_entities):
                total_count = self.entity_counts[et].get(entity_idx, 0)
                fraud_count = self.fraud_entity_counts[et].get(entity_idx, 0)
                entity_features[entity_idx, 0] = np.log1p(total_count)
                entity_features[entity_idx, 1] = fraud_count / (total_count + 1e-6)  
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        if not self.inference_mode and self.fraud_patterns:
            print("Building fraud pattern edges...")
            fraud_pattern_edges = set()
            
            
            for pattern, indices in self.fraud_patterns.items():
                if len(indices) > 1:
                    
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            fraud_pattern_edges.add((indices[i], indices[j]))
                            fraud_pattern_edges.add((indices[j], indices[i]))
            
            if fraud_pattern_edges:
                src, dst = zip(*fraud_pattern_edges)
                pattern_edge_index = torch.tensor([src, dst], dtype=torch.long)
                data['transaction', 'fraud_pattern', 'transaction'].edge_index = pattern_edge_index
                print(f"Added {len(fraud_pattern_edges)} fraud pattern edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_categorical
        if hasattr(self, 'tx_times'):
            del self.tx_times
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class FraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                
                conv_dict[('transaction', f'to_{et}', et)] = SAGEConv(
                    hidden_channels, hidden_channels)
                conv_dict[(et, f'from_{et}', 'transaction')] = SAGEConv(
                    hidden_channels, hidden_channels)
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, data):
        
        x_dict = {
            'transaction': F.elu(self.tx_proj(data['transaction'].x))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                x_dict[et] = self.entity_proj(data[et].x)
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=data['transaction'].x.device)
        
        
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            except Exception as e:
                continue
        
        
        return self.head(x_dict['transaction']).squeeze()





def compute_metrics(labels, probs, threshold=0.5, dataset_name=""):
    """Compute comprehensive classification metrics"""
    if len(np.unique(labels)) == 1:
        print(f"Warning: Only one class present in {dataset_name} set")
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'roc_auc': 0,
            'pr_auc': 0,
            'confusion_matrix': np.zeros((2,2)),
            'classification_report': ""
        }
    
    preds = (probs > threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], 
                yticklabels=['Legit', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    plt.close()
    
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name} (AUC={pr_auc:.4f})')
    plt.savefig(f'pr_curve_{dataset_name}.png')
    plt.close()
    
    
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} (AUC={roc_auc:.4f})')
    plt.savefig(f'roc_curve_{dataset_name}.png')
    plt.close()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=1024,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=1024,
        shuffle=False
    )
    
    
    train_eval_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=1024,
        shuffle=False
    )
    
    
    model = FraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=2
    )
    model = model.to(device)
    
    
    print("Initializing parameters...")
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            model(batch)
            break
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))
    print("Using weighted BCE loss with pos_weight=20.0")
    
    
    best_f1 = 0
    no_improve = 0
    max_epochs = 30
    history = []
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        
        train_probs, train_labels = [], []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            
            with torch.no_grad():
                probs = torch.sigmoid(seed_logits).cpu().numpy()
                train_probs.append(probs)
                train_labels.append(target.cpu().numpy())
        
        
        train_probs = np.concatenate(train_probs)
        train_labels = np.concatenate(train_labels)
        train_metrics = compute_metrics(train_labels, train_probs, dataset_name=f"Train Epoch {epoch}")
        
        
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                fraud_logits = model(batch)
                seed_logits = fraud_logits[:batch['transaction'].batch_size]
                val_probs.append(torch.sigmoid(seed_logits).cpu().numpy())
                val_labels.append(batch['transaction'].y[:batch['transaction'].batch_size].cpu().numpy())
        
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        val_metrics = compute_metrics(val_labels, val_probs, dataset_name=f"Validation Epoch {epoch}")
        
        
        history.append({
            'epoch': epoch,
            'train_loss': total_loss / batch_count,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })
        
        
        print(f"\nEpoch {epoch} - Loss: {total_loss/batch_count:.4f}")
        print("Train Metrics:")
        print(f"  Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {train_metrics['roc_auc']:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f}")
        print("Validation Metrics:")
        print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
        print("Confusion Matrix (Validation):")
        print(val_metrics['confusion_matrix'])
        
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            no_improve = 0
            torch.save(model.state_dict(), 'best_fraud_model5.pt')
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            no_improve += 1
            
        
        if no_improve >= 5:
            print(f"Early stopping at epoch {epoch}")
            break
            
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.1f} seconds")
    
    
    model.eval()
    full_train_probs, full_train_labels = [], []
    with torch.no_grad():
        for batch in train_eval_loader:
            batch = batch.to(device)
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            full_train_probs.append(torch.sigmoid(seed_logits).cpu().numpy())
            full_train_labels.append(batch['transaction'].y[:batch['transaction'].batch_size].cpu().numpy())
    
    full_train_probs = np.concatenate(full_train_probs)
    full_train_labels = np.concatenate(full_train_labels)
    full_train_metrics = compute_metrics(full_train_labels, full_train_probs, dataset_name="Full Training Set")
    
    print("\n===== Final Training Metrics =====")
    print(f"Precision: {full_train_metrics['precision']:.4f}, Recall: {full_train_metrics['recall']:.4f}, F1: {full_train_metrics['f1']:.4f}")
    print(f"ROC-AUC: {full_train_metrics['roc_auc']:.4f}, PR-AUC: {full_train_metrics['pr_auc']:.4f}")
    
    
    joblib.dump(history, 'training_history5.pkl')
    
    
    model.load_state_dict(torch.load('best_fraud_model5.pt'))
    
    
    print("Saving preprocessing artifacts...")
    artifacts = {
        'target_encoders': graph_builder.target_encoders,
        'scaler': graph_builder.scaler,
        'num_medians': graph_builder.num_medians
    }
    joblib.dump(artifacts, 'fraud_inference_artifacts5.pkl')
    
    return model, artifacts, history

def predict_fraud(model, transaction_path, identity_path, artifacts, test_labels_path=None):
    """Predict fraud probabilities for test data with metrics if labels are available"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo'
    ]
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES,
        inference_mode=True,
        target_encoders=artifacts['target_encoders'],
        scaler=artifacts['scaler'],
        num_medians=artifacts['num_medians']
    )
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    test_labels = None
    if test_labels_path and os.path.exists(test_labels_path):
        test_labels_df = pd.read_csv(test_labels_path)
        test_labels = test_labels_df.set_index('TransactionID')['isFraud']
        print(f"Loaded test labels with {len(test_labels)} rows")
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing test chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building test graph...")
    test_data = graph_builder.build_graph()
    test_data = test_data.to(device)
    print(f"Test graph metadata: {test_data}")
    
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [15, 10] for key in test_data.edge_index_dict},
        input_nodes=('transaction', torch.arange(test_data['transaction'].x.size(0))),
        batch_size=2048,
        shuffle=False
    )
    
    
    all_probs = []
    transaction_ids = []
    test_labels_list = [] if test_labels is not None else None
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits).cpu().numpy()
            all_probs.append(probs)
            
            
            seed_ids = batch['transaction'].transaction_id[:batch['transaction'].batch_size].cpu().numpy()
            transaction_ids.append(seed_ids)
            
            
            if test_labels is not None:
                batch_labels = test_labels.loc[seed_ids].values
                test_labels_list.append(batch_labels)
    
    
    test_probs = np.concatenate(all_probs)
    transaction_ids = np.concatenate(transaction_ids)
    
    
    submission = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': test_probs
    })
    
    
    test_metrics = None
    if test_labels_list is not None and len(test_labels_list) > 0:
        test_labels_full = np.concatenate(test_labels_list)
        test_metrics = compute_metrics(test_labels_full, test_probs, dataset_name="Test Set")
        
        print("\n===== Test Set Metrics =====")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
        print("Confusion Matrix:")
        print(test_metrics['confusion_matrix'])
        
        
        joblib.dump(test_metrics, 'test_metrics5.pkl')
    
    return submission, test_metrics





if __name__ == "__main__":
    
    TRAIN_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    TEST_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
    TEST_LABELS_PATH = None  
    
    
    if not os.path.exists('best_fraud_model5.pt'):
        print("Training fraud-focused model...")
        model, artifacts, history = train_fraud_model(
            TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
    else:
        print("Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        CAT_FEATURES = [
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo'
        ]
        tx_feature_size = 300 + len(CAT_FEATURES)
        
        model = FraudGNN(tx_feature_size, 128, 2).to(device)
        model.load_state_dict(torch.load('best_fraud_model5.pt'))
        artifacts = joblib.load('fraud_inference_artifacts5.pkl')
        history = joblib.load('training_history.pkl') if os.path.exists('training_history5.pkl') else []
    
    
    print("\n===== Running Inference on Test Data =====")
    submission, test_metrics = predict_fraud(
        model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts, TEST_LABELS_PATH)
    
    
    submission.to_csv('fraud_submission5.csv', index=False)
    print("Submission saved with shape:", submission.shape)
    print("First 5 predictions:")
    print(submission.head())


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, average_precision_score, 
                            precision_recall_curve, auc, roc_auc_score,
                            classification_report, roc_curve)
import gc
import warnings
from collections import defaultdict
import time
import joblib
from category_encoders import TargetEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')





class FraudFocusedGraphBuilder:
    def __init__(self, entity_types, cat_features, inference_mode=False, 
                 target_encoders=None, scaler=None, num_medians=None):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.inference_mode = inference_mode
        self.target_encoders = target_encoders
        self.scaler = scaler
        self.num_medians = num_medians
        
        
        self.tx_features = []
        if not self.inference_mode:
            self.tx_labels = []
            self.fraud_indices = []
        self.tx_categorical = {col: [] for col in cat_features}
        self.transaction_ids = []
        self.num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.fraud_entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8
        
        
        self.pattern_registry = defaultdict(list)
        self.pattern_fraud_counts = defaultdict(int)
        self.pattern_total_counts = defaultdict(int)
        self.amt_bins = [0, 10, 50, 100, 200, 500, 1000, 5000, float('inf')]
        self.non_fraud_patterns = defaultdict(list) if not inference_mode else None

    def get_pattern_key(self, row):
        """Create enhanced pattern key using multiple features"""
        card1 = str(row.get('card1', 'MISSING'))
        product = str(row.get('ProductCD', 'MISSING'))
        device = str(row.get('DeviceInfo', 'MISSING')).split('_')[0]  
        p_email = str(row.get('P_emaildomain', 'MISSING')).split('.')[0]  
        addr1 = str(row.get('addr1', 'MISSING'))
        
        
        amt = row.get('TransactionAmt', 0)
        amt_bin = np.digitize(amt, self.amt_bins, right=False)
        
        return (card1, product, amt_bin, device, p_email, addr1)

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.transaction_ids.append(batch_df['TransactionID'].values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        
        if not self.inference_mode:
            labels = batch_df['isFraud'].values
            self.tx_labels.append(labels)
            fraud_mask = (labels == 1)
            if np.any(fraud_mask):
                fraud_indices = tx_indices[fraud_mask]
                self.fraud_indices.extend(fraud_indices)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1
                
                
                if not self.inference_mode and tx_idx in self.fraud_indices:
                    self.fraud_entity_counts[et][entity_idx] += 1
        
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            pattern_key = self.get_pattern_key(row)
            tx_idx = start_idx + i
            self.pattern_registry[pattern_key].append(tx_idx)
            self.pattern_total_counts[pattern_key] += 1
            
            if not self.inference_mode:
                if row.get('isFraud', 0) == 1:
                    self.pattern_fraud_counts[pattern_key] += 1
                else:
                    self.non_fraud_patterns[pattern_key].append(tx_idx)

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation with enhanced fraud-specific features"""
        features = []
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        
        for col in self.num_cols:
            if col in batch_df:
                if self.inference_mode and self.num_medians is not None and col in self.num_medians:
                    median_val = self.num_medians[col]
                else:
                    median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding with fraud focus"""
        if self.inference_mode:
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            encoded_features = []
            for col in self.cat_features:
                encoded = self.target_encoders[col].transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            return np.column_stack(encoded_features)
        else:
            
            tx_labels = np.concatenate(self.tx_labels)
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = np.zeros(self.transaction_counter, dtype=bool)
            train_mask[:num_train] = True
            
            
            self.target_encoders = {}
            encoded_features = []
            for col in self.cat_features:
                
                encoder = TargetEncoder(smoothing=50, min_samples_leaf=100)
                encoder.fit(
                    cat_data[col][train_mask], 
                    tx_labels[train_mask]
                )
                self.target_encoders[col] = encoder
                
                
                encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            
            return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building optimized fraud-focused graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        
        
        if not self.inference_mode:
            self.num_medians = {}
            for i, col in enumerate(self.num_cols):
                if i < num_features.shape[1]:
                    self.num_medians[col] = np.median(num_features[:, i])
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        if not self.inference_mode:
            self.scaler = StandardScaler()
            tx_features = self.scaler.fit_transform(tx_features)
        else:
            tx_features = self.scaler.transform(tx_features)
            
        
        pattern_features = np.zeros((tx_features.shape[0], 1), dtype=np.float32)
        for pattern_key, tx_indices in self.pattern_registry.items():
            fraud_count = self.pattern_fraud_counts.get(pattern_key, 0)
            fraud_ratio = fraud_count / len(tx_indices)
            for tx_idx in tx_indices:
                pattern_features[tx_idx] = fraud_ratio
        
        tx_features = np.hstack([tx_features, pattern_features])
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        
        if not self.inference_mode:
            tx_labels = np.concatenate(self.tx_labels)
            data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            train_mask[:num_train] = True
            val_mask[num_train:] = True
            data['transaction'].train_mask = train_mask
            data['transaction'].val_mask = val_mask
        
        
        if self.inference_mode:
            tx_ids = np.concatenate(self.transaction_ids)
            data['transaction'].transaction_id = torch.tensor(tx_ids, dtype=torch.long)
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 2), dtype=np.float32)
            for entity_idx in range(num_entities):
                total_count = self.entity_counts[et].get(entity_idx, 0)
                fraud_count = self.fraud_entity_counts[et].get(entity_idx, 0)
                entity_features[entity_idx, 0] = np.log1p(total_count)
                entity_features[entity_idx, 1] = fraud_count / (total_count + 1e-6)  
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building pattern-based edges...")
        pattern_edges = set()
        
        
        for pattern_key, tx_indices in self.pattern_registry.items():
            total_count = len(tx_indices)
            
            
            if total_count < 2 or total_count > 500:
                continue
                
            
            for i in range(len(tx_indices)):
                for j in range(i+1, min(i+101, len(tx_indices))):
                    pattern_edges.add((tx_indices[i], tx_indices[j]))
                    pattern_edges.add((tx_indices[j], tx_indices[i]))
        
        if pattern_edges:
            src, dst = zip(*pattern_edges)
            pattern_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'tx_pattern', 'transaction'].edge_index = pattern_edge_index
            print(f"Added {len(pattern_edges)} pattern-based edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_categorical, self.pattern_registry
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class FraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                conv_dict[('transaction', f'to_{et}', et)] = SAGEConv(
                    hidden_channels, hidden_channels)
                conv_dict[(et, f'from_{et}', 'transaction')] = SAGEConv(
                    hidden_channels, hidden_channels)
            
            
            conv_dict[('transaction', 'tx_pattern', 'transaction')] = SAGEConv(
                hidden_channels, hidden_channels)
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, data):
        
        x_dict = {
            'transaction': F.elu(self.tx_proj(data['transaction'].x))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                x_dict[et] = self.entity_proj(data[et].x)
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=data['transaction'].x.device)
        
        
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            except Exception as e:
                continue
        
        
        return self.head(x_dict['transaction']).squeeze()





def compute_metrics(labels, probs, threshold=0.5, dataset_name=""):
    """Compute comprehensive classification metrics"""
    if len(np.unique(labels)) == 1:
        print(f"Warning: Only one class present in {dataset_name} set")
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'roc_auc': 0,
            'pr_auc': 0,
            'confusion_matrix': np.zeros((2,2)),
            'classification_report': ""
        }
    
    preds = (probs > threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], 
                yticklabels=['Legit', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    plt.close()
    
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name} (AUC={pr_auc:.4f})')
    plt.savefig(f'pr_curve_{dataset_name}.png')
    plt.close()
    
    
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} (AUC={roc_auc:.4f})')
    plt.savefig(f'roc_curve_{dataset_name}.png')
    plt.close()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=1024,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=1024,
        shuffle=False
    )
    
    
    train_eval_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=1024,
        shuffle=False
    )
    
    
    model = FraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=2
    )
    model = model.to(device)
    
    
    print("Initializing parameters...")
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            model(batch)
            break
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))
    print("Using weighted BCE loss with pos_weight=20.0")
    
    
    best_f1 = 0
    no_improve = 0
    max_epochs = 30
    history = []
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        
        train_probs, train_labels = [], []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            
            with torch.no_grad():
                probs = torch.sigmoid(seed_logits).cpu().numpy()
                train_probs.append(probs)
                train_labels.append(target.cpu().numpy())
        
        
        train_probs = np.concatenate(train_probs)
        train_labels = np.concatenate(train_labels)
        train_metrics = compute_metrics(train_labels, train_probs, dataset_name=f"Train Epoch {epoch}")
        
        
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                fraud_logits = model(batch)
                seed_logits = fraud_logits[:batch['transaction'].batch_size]
                val_probs.append(torch.sigmoid(seed_logits).cpu().numpy())
                val_labels.append(batch['transaction'].y[:batch['transaction'].batch_size].cpu().numpy())
        
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        val_metrics = compute_metrics(val_labels, val_probs, dataset_name=f"Validation Epoch {epoch}")
        
        
        history.append({
            'epoch': epoch,
            'train_loss': total_loss / batch_count,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })
        
        
        print(f"\nEpoch {epoch} - Loss: {total_loss/batch_count:.4f}")
        print("Train Metrics:")
        print(f"  Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {train_metrics['roc_auc']:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f}")
        print("Validation Metrics:")
        print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
        print("Confusion Matrix (Validation):")
        print(val_metrics['confusion_matrix'])
        
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            no_improve = 0
            torch.save(model.state_dict(), 'best_fraud_model6.pt')
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            no_improve += 1
            
        
        if no_improve >= 5:
            print(f"Early stopping at epoch {epoch}")
            break
            
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.1f} seconds")
    
    
    model.eval()
    full_train_probs, full_train_labels = [], []
    with torch.no_grad():
        for batch in train_eval_loader:
            batch = batch.to(device)
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            full_train_probs.append(torch.sigmoid(seed_logits).cpu().numpy())
            full_train_labels.append(batch['transaction'].y[:batch['transaction'].batch_size].cpu().numpy())
    
    full_train_probs = np.concatenate(full_train_probs)
    full_train_labels = np.concatenate(full_train_labels)
    full_train_metrics = compute_metrics(full_train_labels, full_train_probs, dataset_name="Full Training Set")
    
    print("\n===== Final Training Metrics =====")
    print(f"Precision: {full_train_metrics['precision']:.4f}, Recall: {full_train_metrics['recall']:.4f}, F1: {full_train_metrics['f1']:.4f}")
    print(f"ROC-AUC: {full_train_metrics['roc_auc']:.4f}, PR-AUC: {full_train_metrics['pr_auc']:.4f}")
    
    
    joblib.dump(history, 'training_history6.pkl')
    
    
    model.load_state_dict(torch.load('best_fraud_model6.pt'))
    
    
    print("Saving preprocessing artifacts...")
    artifacts = {
        'target_encoders': graph_builder.target_encoders,
        'scaler': graph_builder.scaler,
        'num_medians': graph_builder.num_medians,
        'tx_feature_size': tx_feature_size  
    }
    joblib.dump(artifacts, 'fraud_inference_artifacts6.pkl')
    
    return model, artifacts, history

def predict_fraud(model, transaction_path, identity_path, artifacts, test_labels_path=None):
    """Predict fraud probabilities for test data with metrics if labels are available"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo'
    ]
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES,
        inference_mode=True,
        target_encoders=artifacts['target_encoders'],
        scaler=artifacts['scaler'],
        num_medians=artifacts['num_medians']
    )
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    test_labels = None
    if test_labels_path and os.path.exists(test_labels_path):
        test_labels_df = pd.read_csv(test_labels_path)
        test_labels = test_labels_df.set_index('TransactionID')['isFraud']
        print(f"Loaded test labels with {len(test_labels)} rows")
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing test chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building test graph...")
    test_data = graph_builder.build_graph()
    test_data = test_data.to(device)
    print(f"Test graph metadata: {test_data}")
    
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [15, 10] for key in test_data.edge_index_dict},
        input_nodes=('transaction', torch.arange(test_data['transaction'].x.size(0))),
        batch_size=2048,
        shuffle=False
    )
    
    
    all_probs = []
    transaction_ids = []
    test_labels_list = [] if test_labels is not None else None
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits).cpu().numpy()
            all_probs.append(probs)
            
            
            seed_ids = batch['transaction'].transaction_id[:batch['transaction'].batch_size].cpu().numpy()
            transaction_ids.append(seed_ids)
            
            
            if test_labels is not None:
                batch_labels = test_labels.loc[seed_ids].values
                test_labels_list.append(batch_labels)
    
    
    test_probs = np.concatenate(all_probs)
    transaction_ids = np.concatenate(transaction_ids)
    
    
    submission = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': test_probs
    })
    
    
    test_metrics = None
    if test_labels_list is not None and len(test_labels_list) > 0:
        test_labels_full = np.concatenate(test_labels_list)
        test_metrics = compute_metrics(test_labels_full, test_probs, dataset_name="Test Set")
        
        print("\n===== Test Set Metrics =====")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
        print("Confusion Matrix:")
        print(test_metrics['confusion_matrix'])
        
        
        joblib.dump(test_metrics, 'test_metrics6.pkl')
    
    return submission, test_metrics





if __name__ == "__main__":
    
    TRAIN_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    TEST_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
    TEST_LABELS_PATH = None
    
    
    if not os.path.exists('best_fraud_model6.pt'):
        print("Training fraud-focused model...")
        model, artifacts, history = train_fraud_model(
            TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
    else:
        print("Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        artifacts = joblib.load('fraud_inference_artifacts6.pkl')
        
        
        tx_feature_size = artifacts.get('tx_feature_size', 322)  
        
        model = FraudGNN(tx_feature_size, 128, 2).to(device)
        
        
        try:
            model.load_state_dict(torch.load('best_fraud_model6.pt'))
        except RuntimeError as e:
            print(f"Model architecture mismatch: {e}")
            print("Re-initializing model and training from scratch...")
            model, artifacts, history = train_fraud_model(
                TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
        
        history = joblib.load('training_history6.pkl') if os.path.exists('training_history6.pkl') else []
    
    
    if 'tx_feature_size' not in artifacts:
        artifacts['tx_feature_size'] = model.tx_proj[0].in_features
        joblib.dump(artifacts, 'fraud_inference_artifacts6.pkl')
    
    
    print("\n===== Running Inference on Test Data =====")
    submission, test_metrics = predict_fraud(
        model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts, TEST_LABELS_PATH)
    
    
    submission.to_csv('fraud_submission6.csv', index=False)
    print("Submission saved with shape:", submission.shape)
    print("First 5 predictions:")
    print(submission.head())





!pip install torch-geometric


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, average_precision_score, 
                            precision_recall_curve, auc, roc_auc_score,
                            classification_report, roc_curve)
import gc
import warnings
from collections import defaultdict
import time
import joblib
from category_encoders import TargetEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import torch_sparse




class FraudFocusedGraphBuilder:
    def __init__(self, entity_types, cat_features, inference_mode=False, 
                 target_encoders=None, scaler=None, num_medians=None):
        self.entity_types = entity_types
        self.cat_features = cat_features
        self.transaction_counter = 0
        self.inference_mode = inference_mode
        self.target_encoders = target_encoders
        self.scaler = scaler
        self.num_medians = num_medians
        
        
        self.tx_features = []
        if not self.inference_mode:
            self.tx_labels = []
            self.fraud_indices = []
        self.tx_categorical = {col: [] for col in cat_features}
        self.transaction_ids = []
        self.num_cols = [
            'TransactionAmt','dist1','dist2','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
            'C14','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','V1','V2','V3','V4',
            'V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
            'V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41',
            'V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
            'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77',
            'V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95',
            'V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111',
            'V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126',
            'V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139','V140','V141',
            'V142','V143','V144','V145','V146','V147','V148','V149','V150','V151','V152','V153','V154','V155','V156',
            'V157','V158','V159','V160','V161','V162','V163','V164','V165','V166','V167','V168','V169','V170','V171',
            'V172','V173','V174','V175','V176','V177','V178','V179','V180','V181','V182','V183','V184','V185','V186',
            'V187','V188','V189','V190','V191','V192','V193','V194','V195','V196','V197','V198','V199','V200','V201',
            'V202','V203','V204','V205','V206','V207','V208','V209','V210','V211','V212','V213','V214','V215','V216',
            'V217','V218','V219','V220','V221','V222','V223','V224','V225','V226','V227','V228','V229','V230','V231',
            'V232','V233','V234','V235','V236','V237','V238','V239','V240','V241','V242','V243','V244','V245','V246',
            'V247','V248','V249','V250','V251','V252','V253','V254','V255','V256','V257','V258','V259','V260','V261',
            'V262','V263','V264','V265','V266','V267','V268','V269','V270','V271','V272','V273','V274','V275','V276',
            'V277','V278','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291',
            'V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306',
            'V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321',
            'V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336',
            'V337','V338','V339','id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10','id_11'
        ]
        
        
        self.entity_maps = {et: {} for et in entity_types}
        self.entity_edges = {et: defaultdict(list) for et in entity_types}
        self.entity_counts = {et: defaultdict(int) for et in entity_types}
        self.fraud_entity_counts = {et: defaultdict(int) for et in entity_types}
        self.train_ratio = 0.8
        
        
        self.pattern_registry = defaultdict(list)
        self.pattern_fraud_counts = defaultdict(int)
        self.pattern_total_counts = defaultdict(int)
        self.amt_bins = [0, 10, 50, 100, 200, 500, 1000, 5000, float('inf')]
        self.non_fraud_patterns = defaultdict(list) if not inference_mode else None

    def get_pattern_key(self, row):
        """Create enhanced pattern key using critical fraud indicators"""
        
        card1 = str(row.get('card1', 'MISSING'))
        card4 = str(row.get('card4', 'MISSING'))  
        card6 = str(row.get('card6', 'MISSING'))  
        
        
        addr1 = str(row.get('addr1', 'MISSING'))
        p_email = str(row.get('P_emaildomain', 'MISSING')).split('.')[0]  
        
        
        device_type = str(row.get('DeviceType', 'MISSING'))
        device_info = str(row.get('DeviceInfo', 'MISSING')).split('_')[0]  
        
        
        product = str(row.get('ProductCD', 'MISSING'))
        m_flags = ''.join(str(row.get(f'M{i}', 'MISSING')) for i in range(1, 10))  
        
        
        amt = row.get('TransactionAmt', 0)
        amt_bin = np.digitize(amt, self.amt_bins, right=False)
        
        return (card1, card4, card6, addr1, p_email, 
                device_type, device_info, product, m_flags, amt_bin)

    def add_batch(self, batch_df):
        start_idx = self.transaction_counter
        end_idx = self.transaction_counter + len(batch_df)
        tx_indices = np.arange(start_idx, end_idx)
        self.transaction_counter = end_idx
        
        
        self.transaction_ids.append(batch_df['TransactionID'].values)
        
        
        features = self.create_features_batch(batch_df)
        self.tx_features.append(features)
        
        if not self.inference_mode:
            labels = batch_df['isFraud'].values
            self.tx_labels.append(labels)
            fraud_mask = (labels == 1)
            if np.any(fraud_mask):
                fraud_indices = tx_indices[fraud_mask]
                self.fraud_indices.extend(fraud_indices)
        
        
        for col in self.cat_features:
            if col in batch_df:
                self.tx_categorical[col].append(batch_df[col].fillna('MISSING').astype(str).values)
            else:
                self.tx_categorical[col].append(np.array(['MISSING'] * len(batch_df)))
        
        
        for et in self.entity_types:
            entities = self.get_entity_values(et, batch_df)
            valid_mask = (entities != '') & (entities != '_') & (entities != '__')
            entities = entities[valid_mask]
            batch_tx_indices = tx_indices[valid_mask]
            
            
            for tx_idx, entity in zip(batch_tx_indices, entities):
                if entity not in self.entity_maps[et]:
                    self.entity_maps[et][entity] = len(self.entity_maps[et])
                entity_idx = self.entity_maps[et][entity]
                self.entity_edges[et][entity_idx].append(tx_idx)
                self.entity_counts[et][entity_idx] += 1
                
                
                if not self.inference_mode and tx_idx in self.fraud_indices:
                    self.fraud_entity_counts[et][entity_idx] += 1
        
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            pattern_key = self.get_pattern_key(row)
            tx_idx = start_idx + i
            self.pattern_registry[pattern_key].append(tx_idx)
            self.pattern_total_counts[pattern_key] += 1
            
            if not self.inference_mode:
                if row.get('isFraud', 0) == 1:
                    self.pattern_fraud_counts[pattern_key] += 1
                else:
                    self.non_fraud_patterns[pattern_key].append(tx_idx)

    def get_entity_values(self, et, batch_df):
        if et == 'card':
            c1 = batch_df['card1'].fillna('').astype(str).values
            c2 = batch_df['card2'].fillna('').astype(str).values
            c3 = batch_df['card3'].fillna('').astype(str).values
            c4 = batch_df['card4'].fillna('').astype(str).values
            c5 = batch_df['card5'].fillna('').astype(str).values
            c6 = batch_df['card6'].fillna('').astype(str).values
            return c1 + "_" + c2 + "_" + c3 + "_" + c4 + "_" + c5 + "_" + c6
        elif et == 'addr':
            a1 = batch_df['addr1'].fillna('').astype(str).values
            a2 = batch_df['addr2'].fillna('').astype(str).values
            return a1 + "_" + a2
        elif et == 'email':
            p_email = batch_df['P_emaildomain'].fillna('').astype(str).values
            r_email = batch_df['R_emaildomain'].fillna('').astype(str).values
            return p_email + "_" + r_email
        elif et == 'device':
            return batch_df['DeviceInfo'].fillna('').astype(str).values
        elif et == 'product':
            return batch_df['ProductCD'].fillna('').astype(str).values
        return np.array([''] * len(batch_df))
    
    def create_features_batch(self, batch_df):
        """Feature creation with enhanced fraud-specific features"""
        features = []
        
        
        amt = batch_df['TransactionAmt'].values
        features.append(np.log1p(np.where(amt > 0, amt, 0)))
        features.append(np.where(amt > 0, 1, 0))
        
        
        dt = batch_df['TransactionDT'].fillna(0).values
        hour = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        
        for col in self.num_cols:
            if col in batch_df:
                if self.inference_mode and self.num_medians is not None and col in self.num_medians:
                    median_val = self.num_medians[col]
                else:
                    median_val = batch_df[col].median()
                feat = batch_df[col].fillna(median_val).values.astype(np.float32)
                features.append(feat)
            else:
                features.append(np.zeros(len(batch_df), dtype=np.float32))
        
        return np.column_stack(features)
    
    def apply_target_encoding(self):
        """Apply target encoding with fraud focus"""
        if self.inference_mode:
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            encoded_features = []
            for col in self.cat_features:
                encoded = self.target_encoders[col].transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            return np.column_stack(encoded_features)
        else:
            
            tx_labels = np.concatenate(self.tx_labels)
            cat_data = {}
            for col in self.cat_features:
                cat_data[col] = np.concatenate(self.tx_categorical[col])
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = np.zeros(self.transaction_counter, dtype=bool)
            train_mask[:num_train] = True
            
            
            self.target_encoders = {}
            encoded_features = []
            for col in self.cat_features:
                
                encoder = TargetEncoder(smoothing=50, min_samples_leaf=100)
                encoder.fit(
                    cat_data[col][train_mask], 
                    tx_labels[train_mask]
                )
                self.target_encoders[col] = encoder
                
                
                encoded = encoder.transform(cat_data[col]).values.astype(np.float32)
                encoded_features.append(encoded)
            
            return np.column_stack(encoded_features)
    
    def build_graph(self):
        print(f"Building optimized fraud-focused graph with {self.transaction_counter} transactions")
        start_time = time.time()
        
        
        num_features = np.vstack(self.tx_features)
        
        
        if not self.inference_mode:
            self.num_medians = {}
            for i, col in enumerate(self.num_cols):
                if i < num_features.shape[1]:
                    self.num_medians[col] = np.median(num_features[:, i])
        
        
        cat_features = self.apply_target_encoding()
        
        
        tx_features = np.hstack([num_features, cat_features])
        
        
        if not self.inference_mode:
            self.scaler = StandardScaler()
            tx_features = self.scaler.fit_transform(tx_features)
        else:
            tx_features = self.scaler.transform(tx_features)
            
        
        pattern_features = np.zeros((tx_features.shape[0], 1), dtype=np.float32)
        for pattern_key, tx_indices in self.pattern_registry.items():
            fraud_count = self.pattern_fraud_counts.get(pattern_key, 0)
            fraud_ratio = fraud_count / len(tx_indices)
            for tx_idx in tx_indices:
                pattern_features[tx_idx] = fraud_ratio
        
        tx_features = np.hstack([tx_features, pattern_features])
        tx_feature_tensor = torch.tensor(tx_features, dtype=torch.float32)
        
        
        data = HeteroData()
        data['transaction'].x = tx_feature_tensor
        
        if not self.inference_mode:
            tx_labels = np.concatenate(self.tx_labels)
            data['transaction'].y = torch.tensor(tx_labels, dtype=torch.float32)
            
            
            num_train = int(self.transaction_counter * self.train_ratio)
            train_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            val_mask = torch.zeros(self.transaction_counter, dtype=torch.bool)
            train_mask[:num_train] = True
            val_mask[num_train:] = True
            data['transaction'].train_mask = train_mask
            data['transaction'].val_mask = val_mask
        
        
        if self.inference_mode:
            tx_ids = np.concatenate(self.transaction_ids)
            data['transaction'].transaction_id = torch.tensor(tx_ids, dtype=torch.long)
        
        
        for et in self.entity_types:
            num_entities = len(self.entity_maps[et])
            
            
            entity_features = np.zeros((num_entities, 2), dtype=np.float32)
            for entity_idx in range(num_entities):
                total_count = self.entity_counts[et].get(entity_idx, 0)
                fraud_count = self.fraud_entity_counts[et].get(entity_idx, 0)
                entity_features[entity_idx, 0] = np.log1p(total_count)
                entity_features[entity_idx, 1] = fraud_count / (total_count + 1e-6)  
                
            entity_feature_tensor = torch.tensor(entity_features, dtype=torch.float32)
            data[et].x = entity_feature_tensor
            
            
            src_list, dst_list = [], []
            for entity_idx, tx_indices in self.entity_edges[et].items():
                if entity_idx < num_entities:
                    for tx_idx in tx_indices:
                        src_list.append(tx_idx)
                        dst_list.append(entity_idx)
            
            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data['transaction', f'to_{et}', et].edge_index = edge_index
            data[et, f'from_{et}', 'transaction'].edge_index = rev_edge_index
        
        
        print("Building pattern-based edges...")
        pattern_edges = set()
        
        
        for pattern_key, tx_indices in self.pattern_registry.items():
            total_count = len(tx_indices)
            
            
            if total_count < 2 or total_count > 500:
                continue
                
            
            for i in range(len(tx_indices)):
                for j in range(i+1, min(i+101, len(tx_indices))):
                    pattern_edges.add((tx_indices[i], tx_indices[j]))
                    pattern_edges.add((tx_indices[j], tx_indices[i]))
        
        if pattern_edges:
            src, dst = zip(*pattern_edges)
            pattern_edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['transaction', 'tx_pattern', 'transaction'].edge_index = pattern_edge_index
            print(f"Added {len(pattern_edges)} pattern-based edges")
        
        
        del self.tx_features, self.entity_edges, self.tx_categorical, self.pattern_registry
        gc.collect()
        
        print(f"Graph built in {time.time()-start_time:.1f} seconds")
        return data





class FraudGNN(nn.Module):
    def __init__(self, tx_feature_size, hidden_channels, num_layers):
        super().__init__()
        self.entity_types = ['card', 'addr', 'email', 'device', 'product']
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        
        self.tx_proj = nn.Sequential(
            nn.Linear(tx_feature_size, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        
        self.entity_proj = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in self.entity_types:
                conv_dict[('transaction', f'to_{et}', et)] = SAGEConv(
                    hidden_channels, hidden_channels)
                conv_dict[(et, f'from_{et}', 'transaction')] = SAGEConv(
                    hidden_channels, hidden_channels)
            
            
            conv_dict[('transaction', 'tx_pattern', 'transaction')] = SAGEConv(
                hidden_channels, hidden_channels)
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, data):
        
        x_dict = {
            'transaction': F.elu(self.tx_proj(data['transaction'].x))
        }
        
        
        for et in self.entity_types:
            if hasattr(data[et], 'x') and data[et].x.size(0) > 0:
                x_dict[et] = self.entity_proj(data[et].x)
            else:
                x_dict[et] = torch.zeros(0, self.entity_proj[0].out_features, 
                                        device=data['transaction'].x.device)
        
        
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, data.edge_index_dict)
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            except Exception as e:
                continue
        
        
        return self.head(x_dict['transaction']).squeeze()






def compute_metrics(labels, probs, threshold=0.5, dataset_name=""):
    """Compute comprehensive classification metrics"""
    if len(np.unique(labels)) == 1:
        print(f"Warning: Only one class present in {dataset_name} set")
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'roc_auc': 0,
            'pr_auc': 0,
            'confusion_matrix': np.zeros((2,2)),
            'classification_report': ""
        }
    
    preds = (probs > threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], 
                yticklabels=['Legit', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    plt.close()
    
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name} (AUC={pr_auc:.4f})')
    plt.savefig(f'pr_curve_{dataset_name}.png')
    plt.close()
    
    
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} (AUC={roc_auc:.4f})')
    plt.savefig(f'roc_curve_{dataset_name}.png')
    plt.close()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

def train_fraud_model(transaction_path, identity_path):
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES
    )
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building graph...")
    data = graph_builder.build_graph()
    print(f"Graph metadata: {data}")
    
    
    data = data.to(device)
    
    
    tx_feature_size = data['transaction'].x.size(1)
    
    
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=1024,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].val_mask),
        batch_size=1024,
        shuffle=False
    )
    
    
    train_eval_loader = NeighborLoader(
        data,
        num_neighbors={key: [15, 10] for key in data.edge_index_dict},
        input_nodes=('transaction', data['transaction'].train_mask),
        batch_size=1024,
        shuffle=False
    )
    
    
    model = FraudGNN(
        tx_feature_size=tx_feature_size,
        hidden_channels=128,
        num_layers=2
    )
    model = model.to(device)
    
    
    print("Initializing parameters...")
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            model(batch)
            break
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))
    print("Using weighted BCE loss with pos_weight=20.0")
    
    
    best_f1 = 0
    no_improve = 0
    max_epochs = 30
    history = []
    
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        
        
        train_probs, train_labels = [], []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            target = batch['transaction'].y[:batch['transaction'].batch_size]
            
            
            loss = criterion(seed_logits, target)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            
            with torch.no_grad():
                probs = torch.sigmoid(seed_logits).cpu().numpy()
                train_probs.append(probs)
                train_labels.append(target.cpu().numpy())
        
        
        train_probs = np.concatenate(train_probs)
        train_labels = np.concatenate(train_labels)
        train_metrics = compute_metrics(train_labels, train_probs, dataset_name=f"Train Epoch {epoch}")
        
        
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                fraud_logits = model(batch)
                seed_logits = fraud_logits[:batch['transaction'].batch_size]
                val_probs.append(torch.sigmoid(seed_logits).cpu().numpy())
                val_labels.append(batch['transaction'].y[:batch['transaction'].batch_size].cpu().numpy())
        
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        val_metrics = compute_metrics(val_labels, val_probs, dataset_name=f"Validation Epoch {epoch}")
        
        
        history.append({
            'epoch': epoch,
            'train_loss': total_loss / batch_count,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })
        
        
        print(f"\nEpoch {epoch} - Loss: {total_loss/batch_count:.4f}")
        print("Train Metrics:")
        print(f"  Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {train_metrics['roc_auc']:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f}")
        print("Validation Metrics:")
        print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
        print("Confusion Matrix (Validation):")
        print(val_metrics['confusion_matrix'])
        
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            no_improve = 0
            torch.save(model.state_dict(), 'best_fraud_model6.pt')
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            no_improve += 1
            
        
        if no_improve >= 5:
            print(f"Early stopping at epoch {epoch}")
            break
            
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.1f} seconds")
    
    
    model.eval()
    full_train_probs, full_train_labels = [], []
    with torch.no_grad():
        for batch in train_eval_loader:
            batch = batch.to(device)
            fraud_logits = model(batch)
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            full_train_probs.append(torch.sigmoid(seed_logits).cpu().numpy())
            full_train_labels.append(batch['transaction'].y[:batch['transaction'].batch_size].cpu().numpy())
    
    full_train_probs = np.concatenate(full_train_probs)
    full_train_labels = np.concatenate(full_train_labels)
    full_train_metrics = compute_metrics(full_train_labels, full_train_probs, dataset_name="Full Training Set")
    
    print("\n===== Final Training Metrics =====")
    print(f"Precision: {full_train_metrics['precision']:.4f}, Recall: {full_train_metrics['recall']:.4f}, F1: {full_train_metrics['f1']:.4f}")
    print(f"ROC-AUC: {full_train_metrics['roc_auc']:.4f}, PR-AUC: {full_train_metrics['pr_auc']:.4f}")
    
    
    joblib.dump(history, 'training_history6.pkl')
    
    
    model.load_state_dict(torch.load('best_fraud_model6.pt'))
    
    
    print("Saving preprocessing artifacts...")
    artifacts = {
        'target_encoders': graph_builder.target_encoders,
        'scaler': graph_builder.scaler,
        'num_medians': graph_builder.num_medians,
        'tx_feature_size': tx_feature_size  
    }
    joblib.dump(artifacts, 'fraud_inference_artifacts6.pkl')
    
    return model, artifacts, history

def predict_fraud(model, transaction_path, identity_path, artifacts, test_labels_path=None):
    """Predict fraud probabilities for test data with metrics if labels are available"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    
    CHUNKSIZE = 100000
    ENTITY_TYPES = ['card', 'addr', 'email', 'device', 'product']
    CAT_FEATURES = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo'
    ]
    
    
    graph_builder = FraudFocusedGraphBuilder(
        entity_types=ENTITY_TYPES,
        cat_features=CAT_FEATURES,
        inference_mode=True,
        target_encoders=artifacts['target_encoders'],
        scaler=artifacts['scaler'],
        num_medians=artifacts['num_medians']
    )
    
    
    try:
        identity_df = pd.read_csv(identity_path)
        identity_df = identity_df.set_index('TransactionID')
        identity_df.rename(columns={x: x.replace('-', '_') for x in identity_df.columns}, inplace=True)
        print(f"Loaded identity data with {len(identity_df)} rows")
    except Exception as e:
        print(f"Error loading identity data: {e}")
        identity_df = pd.DataFrame()
        print("Proceeding without identity data")
    
    
    test_labels = None
    if test_labels_path and os.path.exists(test_labels_path):
        test_labels_df = pd.read_csv(test_labels_path)
        test_labels = test_labels_df.set_index('TransactionID')['isFraud']
        print(f"Loaded test labels with {len(test_labels)} rows")
    
    
    chunk_iterator = pd.read_csv(transaction_path, chunksize=CHUNKSIZE)
    for chunk_idx, chunk in enumerate(chunk_iterator):
        print(f"Processing test chunk {chunk_idx+1}")
        
        
        if not identity_df.empty:
            chunk = chunk.merge(identity_df, on='TransactionID', how='left', suffixes=('', '_id'))
        
        
        for col in CAT_FEATURES + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 
                   'DeviceInfo', 'ProductCD']:
            if col not in chunk:
                chunk[col] = 'MISSING'
        
        
        graph_builder.add_batch(chunk)
        del chunk
        gc.collect()
    
    
    print("Building test graph...")
    test_data = graph_builder.build_graph()
    test_data = test_data.to(device)
    print(f"Test graph metadata: {test_data}")
    
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [15, 10] for key in test_data.edge_index_dict},
        input_nodes=('transaction', torch.arange(test_data['transaction'].x.size(0))),
        batch_size=2048,
        shuffle=False
    )
    
    
    all_probs = []
    transaction_ids = []
    test_labels_list = [] if test_labels is not None else None
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            fraud_logits = model(batch)
            
            
            seed_logits = fraud_logits[:batch['transaction'].batch_size]
            probs = torch.sigmoid(seed_logits).cpu().numpy()
            all_probs.append(probs)
            
            
            seed_ids = batch['transaction'].transaction_id[:batch['transaction'].batch_size].cpu().numpy()
            transaction_ids.append(seed_ids)
            
            
            if test_labels is not None:
                batch_labels = test_labels.loc[seed_ids].values
                test_labels_list.append(batch_labels)
    
    
    test_probs = np.concatenate(all_probs)
    transaction_ids = np.concatenate(transaction_ids)
    
    
    submission = pd.DataFrame({
        'TransactionID': transaction_ids,
        'isFraud': test_probs
    })
    
    
    test_metrics = None
    if test_labels_list is not None and len(test_labels_list) > 0:
        test_labels_full = np.concatenate(test_labels_list)
        test_metrics = compute_metrics(test_labels_full, test_probs, dataset_name="Test Set")
        
        print("\n===== Test Set Metrics =====")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
        print("Confusion Matrix:")
        print(test_metrics['confusion_matrix'])
        
        
        joblib.dump(test_metrics, 'test_metrics6.pkl')
    
    return submission, test_metrics





if __name__ == "__main__":
    
    TRAIN_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
    TRAIN_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/train_identity.csv"
    TEST_TRANSACTION_PATH = "/kaggle/input/ieee-fraud-detection/test_transaction.csv"
    TEST_IDENTITY_PATH = "/kaggle/input/ieee-fraud-detection/test_identity.csv"
    TEST_LABELS_PATH = None
    
    
    if not os.path.exists('best_fraud_model.pt'):
        print("Training fraud-focused model...")
        model, artifacts, history = train_fraud_model(
            TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
    else:
        print("Loading pre-trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        artifacts = joblib.load('fraud_inference_artifacts.pkl')
        tx_feature_size = artifacts.get('tx_feature_size', 322)  
        
        model = FraudGNN(tx_feature_size, 128, 2).to(device)
        
        try:
            model.load_state_dict(torch.load('best_fraud_model.pt'))
        except RuntimeError as e:
            print(f"Model architecture mismatch: {e}")
            print("Re-initializing model and training from scratch...")
            model, artifacts, history = train_fraud_model(
                TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH)
        
        history = joblib.load('training_history.pkl') if os.path.exists('training_history.pkl') else []
    
    
    if 'tx_feature_size' not in artifacts:
        artifacts['tx_feature_size'] = model.tx_proj[0].in_features
        joblib.dump(artifacts, 'fraud_inference_artifacts.pkl')
    
    
    print("\n===== Running Inference on Test Data =====")
    submission, test_metrics = predict_fraud(
        model, TEST_TRANSACTION_PATH, TEST_IDENTITY_PATH, artifacts, TEST_LABELS_PATH)
    
    
    submission.to_csv('fraud_submission.csv', index=False)
    print("Submission saved with shape:", submission.shape)
    print("First 5 predictions:")
    print(submission.head())


import torch
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git



import torch
!pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git
!pip install --upgrade scikit-learn==1.2.2 imbalanced-learn==0.10.1



