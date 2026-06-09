!pip install /kaggle/input/pip-library/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl --no-deps


!pip install /kaggle/input/pip-library/torch_geometric-2.6.1-py3-none-any.whl --no-deps


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

target_cols = ['Tg','FFV','Tc','Density','Rg']

def smiles_to_graph(smiles):
    """SMILES文字列を分子グラフに変換"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 原子特徴量の取得
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetMass()
            ]
            atom_features.append(features)
        
        if len(atom_features) == 0:
            return None
            
        # エッジ（結合）情報の取得
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_features = [
                bond.GetBondTypeAsDouble(),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
                int(bond.GetIsAromatic())
            ]
            
            # 無向グラフとして扱う
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([bond_features, bond_features])
        
        # PyTorch Geometricのデータ形式に変換
        if len(edge_indices) == 0:
            # 結合がない場合（単原子分子）
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # 分子レベルの特徴量も追加
        mol_features = [
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            len(smiles)  # SMILES長
        ]
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                   mol_features=torch.tensor(mol_features, dtype=torch.float))
        
        return data
    except:
        return None

class PolymerGNN(nn.Module):
    def __init__(self, atom_features_dim=8, edge_features_dim=4, mol_features_dim=6, 
                 hidden_dim=128, num_layers=3, num_targets=5, dropout=0.2):
        super(PolymerGNN, self).__init__()
        
        self.num_targets = num_targets
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 最初の層
        self.convs.append(GCNConv(atom_features_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 中間層
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 注意機構を追加
        self.attention = GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=dropout)
        self.attention_bn = nn.BatchNorm1d(hidden_dim)
        
        # 分子レベル特徴量の処理
        self.mol_fc = nn.Sequential(
            nn.Linear(mol_features_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # 最終予測層（各ターゲット別）
        combined_dim = hidden_dim * 2 + hidden_dim//2  # mean + max pooling + mol features
        
        self.predictors = nn.ModuleList()
        for _ in range(num_targets):
            predictor = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, 1)
            )
            self.predictors.append(predictor)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        mol_features = torch.stack([d.mol_features for d in data.to_data_list()]).to(x.device)
        
        # Graph convolutions
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
        # Attention layer
        x = self.attention(x, edge_index)
        x = self.attention_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        # Global pooling
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
    
        # mol features
        mol_emb = self.mol_fc(mol_features)
    
        # 結合
        graph_repr = torch.cat([graph_mean, graph_max, mol_emb], dim=1)
    
        # 予測
        predictions = [predictor(graph_repr) for predictor in self.predictors]
        return torch.cat(predictions, dim=1)

def create_data_loaders(df, target_cols, batch_size=32, is_train=True):
    """データローダーを作成"""
    graphs = []
    targets = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        graph = smiles_to_graph(row['SMILES'])
        if graph is not None:
            graphs.append(graph)
            if is_train:
                # 単一ターゲットの場合は1次元、複数ターゲットの場合は2次元
                if len(target_cols) == 1:
                    target_value = row[target_cols[0]] if pd.notna(row[target_cols[0]]) else 0.0
                    targets.append(target_value)
                else:
                    target_values = [row[col] if pd.notna(row[col]) else 0.0 for col in target_cols]
                    targets.append(target_values)
            valid_indices.append(len(graphs) - 1)
    
    if is_train and len(targets) > 0:
        if len(target_cols) == 1:
            # 単一ターゲットの場合は1次元テンソル
            targets_tensor = torch.tensor(targets, dtype=torch.float)
            for i, graph in enumerate(graphs):
                graph.y = targets_tensor[i:i+1]  # 1次元を保持
        else:
            # 複数ターゲットの場合は2次元テンソル
            targets_tensor = torch.tensor(targets, dtype=torch.float)
            for i, graph in enumerate(graphs):
                graph.y = targets_tensor[i]
    
    if len(graphs) == 0:
        return None, valid_indices
    
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=is_train)
    return loader, valid_indices

# wMAE計算のための重み計算
def calculate_wmae_weights(train_data, target_cols):
    """wMAE用の重みを計算"""
    weights = {}
    
    for target in target_cols:
        nt = train_data[target].notna().sum()
        train_values = train_data[target].dropna()
        rt = train_values.max() - train_values.min()
        wt = 1.0 / (np.sqrt(nt) * rt)
        weights[target] = wt
        print(f'{target}: nt={nt}, rt={rt:.4f}, wt={wt:.6f}')
    
    return weights

def calculate_wmae(y_true_dict, y_pred_dict, weights):
    """重み付き平均絶対誤差を計算"""
    total_wmae = 0.0
    total_weight = 0.0
    
    for target in y_true_dict.keys():
        if target in weights:
            mask = ~np.isnan(y_true_dict[target])
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_true_dict[target][mask] - y_pred_dict[target][mask]))
                weighted_mae = weights[target] * mae
                total_wmae += weighted_mae
                total_weight += weights[target]
    
    return total_wmae / total_weight if total_weight > 0 else 0.0

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# wMAE重みを計算
wmae_weights = calculate_wmae_weights(train, target_cols)

# クロスバリデーション設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 結果保存用
test_preds = pd.DataFrame({'id': test['id']})
oof_preds = pd.DataFrame(index=train.index, columns=target_cols)
target_metrics = {}

# 各ターゲット別に学習（マルチタスク学習も可能だが、個別に学習）
for target_idx, target in enumerate(target_cols):
    print(f'\n==> Training for target: {target}')
    
    # 対象ターゲットが存在する行のみを使用
    mask = train[target].notnull()
    train_subset = train.loc[mask].copy()
    
    fold_rmses = []
    fold_maes = []
    test_fold_preds = np.zeros(len(test))
    
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(train_subset), 1):
        print(f'  Fold {fold}...')
        
        train_fold = train_subset.iloc[tr_idx]
        valid_fold = train_subset.iloc[vl_idx]
        
        # データローダー作成
        train_loader, train_indices = create_data_loaders(train_fold, [target], batch_size=32, is_train=True)
        valid_loader, valid_indices = create_data_loaders(valid_fold, [target], batch_size=32, is_train=True)
        test_loader, test_indices = create_data_loaders(test, target_cols, batch_size=32, is_train=False)
        
        if train_loader is None or valid_loader is None:
            print(f'    Skipping fold {fold} due to data issues')
            continue
        
        # モデル初期化
        model = PolymerGNN(num_targets=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        criterion = nn.MSELoss()
        
        # 訓練データのターゲット値を正規化
        train_targets = []
        for batch in train_loader:
            # バッチ内の各グラフのターゲット値を取得
            batch_targets = batch.y
            if batch_targets.dim() == 1:
                train_targets.extend(batch_targets.tolist())
            else:
                train_targets.extend(batch_targets[:, 0].tolist())
        
        target_scaler = StandardScaler()
        train_targets_array = np.array(train_targets).reshape(-1, 1)
        target_scaler.fit(train_targets_array)
        
        # 学習ループ
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(200):
            # 訓練
            model.train()
            train_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # ターゲット値を正規化
                batch_targets = batch.y
                if batch_targets.dim() == 1:
                    targets_normalized = target_scaler.transform(batch_targets.cpu().numpy().reshape(-1, 1))
                else:
                    targets_normalized = target_scaler.transform(batch_targets[:, 0:1].cpu().numpy())
                targets_normalized = torch.tensor(targets_normalized, dtype=torch.float).to(device)
                
                out = model(batch)
                loss = criterion(out.squeeze(), targets_normalized.squeeze())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            # 検証
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in valid_loader:
                    batch = batch.to(device)
                    batch_targets = batch.y
                    if batch_targets.dim() == 1:
                        targets_normalized = target_scaler.transform(batch_targets.cpu().numpy().reshape(-1, 1))
                    else:
                        targets_normalized = target_scaler.transform(batch_targets[:, 0:1].cpu().numpy())
                    targets_normalized = torch.tensor(targets_normalized, dtype=torch.float).to(device)
                    
                    out = model(batch)
                    loss = criterion(out.squeeze(), targets_normalized.squeeze())
                    val_loss += loss.item()
                    
                    # 予測値を元のスケールに戻す
                    preds_original = target_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1))
                    val_preds.extend(preds_original.flatten())
                    
                    if batch_targets.dim() == 1:
                        val_targets.extend(batch_targets.cpu().numpy())
                    else:
                        val_targets.extend(batch_targets[:, 0].cpu().numpy())
            
            val_loss /= len(valid_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # ベストモデルを保存
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
        
        # ベストモデルをロード
        model.load_state_dict(best_model_state)
        
        # 最終評価
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                out = model(batch)
                preds_original = target_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1))
                val_preds.extend(preds_original.flatten())
                
                batch_targets = batch.y
                if batch_targets.dim() == 1:
                    val_targets.extend(batch_targets.cpu().numpy())
                else:
                    val_targets.extend(batch_targets[:, 0].cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        mae = mean_absolute_error(val_targets, val_preds)
        
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        print(f'    Fold {fold} RMSE: {rmse:.4f}, MAE: {mae:.4f}')
        
        # OOF予測を保存
        valid_original_indices = valid_fold.index[valid_indices]
        oof_preds.loc[valid_original_indices, target] = val_preds
        
        # テストデータの予測
        if test_loader is not None:
            test_preds_fold = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    preds_original = target_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1))
                    test_preds_fold.extend(preds_original.flatten())
            
            test_fold_preds[test_indices] += np.array(test_preds_fold) / kf.n_splits
    
    if len(fold_rmses) > 0:
        test_preds[target] = test_fold_preds
        avg_rmse = np.mean(fold_rmses)
        avg_mae = np.mean(fold_maes)
        target_metrics[target] = {'rmse': avg_rmse, 'mae': avg_mae}
        
        print(f'  >>> Avg RMSE for {target}: {avg_rmse:.4f}')
        print(f'  >>> Avg MAE for {target}: {avg_mae:.4f}')

# 全体評価
print(f'\n=== Overall Evaluation ===')

# 全体RMSE計算
all_oof = []
all_true = []
for target in target_cols:
    mask = train[target].notnull() & oof_preds[target].notnull()
    if mask.sum() > 0:
        all_oof.extend(oof_preds.loc[mask, target].astype(float).values)
        all_true.extend(train.loc[mask, target].values)

if len(all_oof) > 0:
    overall_rmse = np.sqrt(mean_squared_error(all_true, all_oof))
    print(f'Overall OOF RMSE: {overall_rmse:.4f}')

# wMAE計算
y_true_dict = {}
y_pred_dict = {}

for target in target_cols:
    mask = train[target].notnull() & oof_preds[target].notnull()
    if mask.sum() > 0:
        y_true_dict[target] = train.loc[mask, target].values
        y_pred_dict[target] = oof_preds.loc[mask, target].astype(float).values

if y_true_dict:
    overall_wmae = calculate_wmae(y_true_dict, y_pred_dict, wmae_weights)
    print(f'Overall OOF wMAE: {overall_wmae:.6f}')

# 各targetの詳細メトリクス表示
print(f'\n=== Target-wise Metrics ===')
for target in target_cols:
    if target in target_metrics:
        metrics = target_metrics[target]
        weight = wmae_weights[target]
        weighted_mae = weight * metrics['mae']
        print(f'{target:8s}: RMSE={metrics["rmse"]:.4f}, MAE={metrics["mae"]:.4f}, '
              f'Weight={weight:.6f}, Weighted MAE={weighted_mae:.6f}')

# 提出ファイル作成
submission = test_preds[['id'] + target_cols]
submission.to_csv('submission.csv', index=False)
print(f'\n=== Submission Preview ===')
print(submission.head())

