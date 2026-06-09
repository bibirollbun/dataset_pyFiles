# Stanford RNA 3D Folding Competition
# Revised Notebook for RNA 3D Structure Prediction (Improved V3)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from scipy.spatial.distance import cdist
import warnings
import random
import torch.nn.functional as F
warnings.filterwarnings('ignore')

print("Starting Stanford RNA 3D Folding notebook...")

# 1. Data Loading and Exploration
print("Loading datasets...")
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')

# Fill missing coordinate values to avoid NaNs.
train_labels.fillna(0, inplace=True)
validation_labels.fillna(0, inplace=True)

print("\nBasic dataset information:")
print(f"Training sequences: {train_sequences.shape}")
print(f"Training labels: {train_labels.shape}")
print(f"Validation sequences: {validation_sequences.shape}")
print(f"Validation labels: {validation_labels.shape}")
print(f"Test sequences: {test_sequences.shape}")
print(f"Sample submission: {sample_submission.shape}")

# 2. Data Analysis (optional visualizations)
print("\nAnalyzing RNA sequence lengths...")
train_sequences['length'] = train_sequences['sequence'].str.len()
plt.figure(figsize=(12, 6))
sns.histplot(train_sequences['length'], bins=50)
plt.title('Distribution of RNA Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.savefig('sequence_length_distribution.png')
plt.close()

# 3. Data Preprocessing
def preprocess_sequence_data(sequences_df, labels_df=None, is_train=True):
    """
    Preprocess RNA sequence data.
    Convert sequences to numerical form and normalize coordinate targets per sequence.
    """
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    processed_data = []
    
    for idx, row in sequences_df.iterrows():
        seq_id = row['target_id']
        sequence = row['sequence']
        numerical_seq = [nucleotide_map.get(nuc, 4) for nuc in sequence]
        
        structures = None
        if is_train and labels_df is not None:
            sequence_labels = labels_df[labels_df['ID'].str.startswith(seq_id + '_')]
            if not sequence_labels.empty:
                num_structures = (len(sequence_labels.columns) - 3) // 3
                structures = []
                for i in range(1, num_structures + 1):
                    coords = []
                    for _, label_row in sequence_labels.iterrows():
                        x = label_row[f'x_{i}']
                        y = label_row[f'y_{i}']
                        z = label_row[f'z_{i}']
                        coords.append([x, y, z])
                    coords = np.array(coords)
                    # Normalize coordinates per sequence (center and scale)
                    mean = np.mean(coords, axis=0)
                    std = np.std(coords, axis=0) + 1e-8
                    coords_norm = (coords - mean) / std
                    structures.append(coords_norm)
        processed_data.append({
            'id': seq_id,
            'sequence': numerical_seq,
            'structures': structures
        })
    return processed_data

print("Preprocessing training data...")
train_data = preprocess_sequence_data(train_sequences, train_labels)
print("Preprocessing validation data...")
validation_data = preprocess_sequence_data(validation_sequences, validation_labels)
print("Preprocessing test data...")
test_data = preprocess_sequence_data(test_sequences, is_train=False)

# 4. Feature Engineering
def extract_sequence_features(sequence):
    """
    Extract one-hot encoding, positional encoding, and GC-content as features.
    """
    one_hot = np.zeros((len(sequence), 5))
    for i, nucleotide in enumerate(sequence):
        one_hot[i, nucleotide] = 1
    gc_content = []
    window_size = 5
    for i in range(len(sequence)):
        start = max(0, i - window_size // 2)
        end = min(len(sequence), i + window_size // 2 + 1)
        window = sequence[start:end]
        gc_count = sum(1 for n in window if n in [1, 2])
        gc_content.append(gc_count / len(window))
    positions = np.array([[i / len(sequence)] for i in range(len(sequence))])
    features = np.hstack((one_hot, positions, np.array(gc_content).reshape(-1, 1)))
    return features

print("Extracting sequence features...")
for i, data in enumerate(train_data):
    train_data[i]['features'] = extract_sequence_features(data['sequence'])
for i, data in enumerate(validation_data):
    validation_data[i]['features'] = extract_sequence_features(data['sequence'])
for i, data in enumerate(test_data):
    test_data[i]['features'] = extract_sequence_features(data['sequence'])

# 5. RNA Secondary Structure Prediction (simple rule-based)
def predict_rna_secondary_structure(sequence):
    nucleotide_map_inv = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'X'}
    seq_chars = [nucleotide_map_inv[n] for n in sequence]
    structure = ['.' for _ in range(len(seq_chars))]
    complementary = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'X': None}
    for i in range(len(seq_chars)):
        if structure[i] != '.':
            continue
        for j in range(len(seq_chars) - 1, i + 3, -1):
            if structure[j] != '.':
                continue
            if complementary[seq_chars[i]] == seq_chars[j]:
                structure[i] = '('
                structure[j] = ')'
                break
    return ''.join(structure)

def enhance_features_with_ss(data):
    for i, item in enumerate(data):
        seq = item['sequence']
        ss = predict_rna_secondary_structure(seq)
        ss_features = np.zeros((len(ss), 3))
        for j, char in enumerate(ss):
            if char == '.':
                ss_features[j, 0] = 1
            elif char == '(':
                ss_features[j, 1] = 1
            elif char == ')':
                ss_features[j, 2] = 1
        data[i]['features'] = np.hstack((item['features'], ss_features))
    return data

print("Enhancing features with secondary structure information...")
train_data = enhance_features_with_ss(train_data)
validation_data = enhance_features_with_ss(validation_data)
test_data = enhance_features_with_ss(test_data)



class RNADataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        
        # Data augmentation
        if self.augment and random.random() < 0.7:
            if random.random() < 0.1:
                mut_idx = random.randint(0, len(features)-1)
                features[mut_idx,:5] = np.eye(5)[random.choice([0,1,2,3])]
            if len(features) > 20 and random.random() < 0.3:
                start = random.randint(0, len(features)-10)
                features[start:start+5] = features[start:start+5][::-1]
        
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(item['structures'][0], dtype=torch.float32) if item['structures'] else None
        return {
            'features': features,
            'target': target,
            'length': features.shape[0],  # 直接使用整数长度
            'id': item['id']             # 返回序列ID
        }

def collate_fn(batch):
    # 按序列长度排序
    sorted_batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    # 提取各组件
    features = [x['features'] for x in sorted_batch]
    targets = [x['target'] for x in sorted_batch]
    lengths = [x['length'] for x in sorted_batch]
    ids = [x['id'] for x in sorted_batch]
    
    # 填充特征
    max_len = features[0].shape[0]
    feat_dim = features[0].shape[1]
    padded_features = torch.zeros((len(features), max_len, feat_dim))
    for i, feat in enumerate(features):
        padded_features[i, :len(feat)] = feat
    
    # 填充目标（如果有）
    if targets[0] is not None:
        padded_targets = torch.zeros((len(targets), max_len, 3))
        for i, tgt in enumerate(targets):
            padded_targets[i, :len(tgt)] = tgt
    else:
        padded_targets = None
    
    return {
        'features': padded_features,
        'targets': padded_targets,
        'lengths': torch.tensor(lengths),  # 转换为tensor
        'ids': ids
    }

train_dataset = RNADataset(train_data, augment=True)
validation_dataset = RNADataset(validation_data)
test_dataset = RNADataset(test_data)


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True  # 防止多epoch数据重载
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True  # 防止多epoch数据重载
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,  # 减少测试批次大小
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True
)

# 3. Enhanced Model Architecture
class RNAFoldingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=4):
        super().__init__()
        
        # BiLSTM Encoder
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers>1 else 0
        )
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2*hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.3
            ),
            num_layers=2
        )
        
        # Geometric Attention
        self.attention = nn.MultiheadAttention(
            2*hidden_dim, num_heads=8, dropout=0.3
        )
        
        # Dynamic Convolution
        self.conv = nn.Sequential(
            nn.Conv1d(2*hidden_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, x, lengths):
        # BiLSTM
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.cpu().numpy().tolist()
        elif isinstance(lengths, list):
            pass  # 已经是列表形式
        else:
            raise ValueError(f"Unsupported lengths type: {type(lengths)}")
        
        # BiLSTM处理
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Transformer
        transformer_out = self.transformer(lstm_out)
        
        # Attention
        attn_out, _ = self.attention(
            transformer_out.permute(1,0,2),
            transformer_out.permute(1,0,2),
            transformer_out.permute(1,0,2)
        )
        attn_out = attn_out.permute(1,0,2)
        
        # Residual Connection
        combined = transformer_out + attn_out
        
        # Convolution
        conv_out = self.conv(combined.permute(0,2,1)).permute(0,2,1)
        
        # Prediction
        return self.head(conv_out)

# 4. Enhanced Loss Function
class GeometricLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.coord_loss = nn.SmoothL1Loss()
        
    def distance_loss(self, pred, target):
        pred_dist = torch.cdist(pred, pred)
        target_dist = torch.cdist(target, target)
        return F.mse_loss(pred_dist, target_dist)
    
    def forward(self, pred, target, lengths):
        total_loss = 0
        for i, l in enumerate(lengths):
            if l < 2: continue
            pred_i = pred[i,:l]
            target_i = target[i,:l]
            
            coord_loss = self.coord_loss(pred_i, target_i)
            dist_loss = self.distance_loss(pred_i, target_i)
            total_loss += self.alpha*coord_loss + (1-self.alpha)*dist_loss
        return total_loss / len(lengths)

# 5. Enhanced Training Loop
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cpu'):
    device = torch.device(device)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = GeometricLoss(alpha=0.5)
    
    best_tm = 0.0
    history = {'train_loss': [], 'val_loss': [], 'tm_score': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            
            optimizer.zero_grad()
            outputs = model(features, lengths)
            loss = criterion(outputs, targets, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        tm_scores = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)
                lengths = batch['lengths']
                ids = batch['ids']
                
                # 计算验证损失
                outputs = model(features, lengths)
                loss = criterion(outputs, targets, lengths)
                val_loss += loss.item()
                
                # 计算TM-Score
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                for i in range(len(lengths)):
                    l = lengths[i].item()
                    if l < 5:  # 跳过过短序列
                        continue
                    pred_coords = outputs_np[i, :l, :]
                    true_coords = targets_np[i, :l, :]
                    tm = calculate_tm_score(pred_coords, true_coords)
                    tm_scores.append(tm)
        
        # 记录指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_tm = np.mean(tm_scores) if tm_scores else 0.0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['tm_score'].append(avg_tm)
        
        # 打印日志
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"TM-Score: {avg_tm:.4f}")
        
        # 保存最佳模型
        if avg_tm > best_tm:
            best_tm = avg_tm
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0
    batches = 0
    # 修复点1: 正确的解包方式
    for features, targets, seq_lengths in dataloader:
        if targets is None:
            continue
        optimizer.zero_grad()
        features = features.to(device)
        targets = targets.to(device)
        outputs = model(features, seq_lengths)
        loss = GeometricLoss()(outputs, targets, seq_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1
    return epoch_loss / batches if batches > 0 else float('inf')

def validate(model, dataloader, device):
    model.eval()
    val_loss = 0
    batches = 0
    with torch.no_grad():
        # 修复点2: 正确的解包方式
        for features, targets, seq_lengths in dataloader:
            if targets is None:
                continue
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features, seq_lengths)
            loss = GeometricLoss()(outputs, targets, seq_lengths)
            val_loss += loss.item()
            batches += 1
    return val_loss / batches if batches > 0 else float('inf')

def evaluate_model(model, dataloader, device):
    model.eval()
    tm_scores = []
    with torch.no_grad():
        # 修复点3: 正确的解包方式
        for features, targets, seq_lengths in dataloader:
            if targets is None:
                continue
            features = features.to(device)
            outputs = model(features, seq_lengths)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            for i, length in enumerate(seq_lengths):
                pred_coords = outputs[i, :length, :]
                target_coords = targets[i, :length, :]
                tm_score = calculate_tm_score(pred_coords, target_coords)
                tm_scores.append(tm_score)
    return np.mean(tm_scores) if tm_scores else 0


def calculate_tm_score(predicted, reference):
    l_ref = len(reference)
    if l_ref < 5:  # 过短序列不计算
        return 0.0
    
    # 标准d0计算公式
    d0 = 1.24 * (l_ref - 15) ** (1/3) - 1.8
    d0 = max(d0, 0.5)  # 确保最小值
    
    # 结构对齐（假设已预处理对齐）
    aligned_len = min(len(predicted), l_ref)
    pred = predicted[:aligned_len]
    ref = reference[:aligned_len]
    
    # 计算距离矩阵
    pred_dists = cdist(pred, pred)
    ref_dists = cdist(ref, ref)
    
    # 计算TM-Score
    tm = (1/(1 + ((pred_dists - ref_dists)/d0)**2)).sum()
    tm_normalized = tm / (l_ref**2 - l_ref)  # 标准化
    
    return np.clip(tm_normalized, 0.0, 1.0)  # 强制限制在[0,1]



# 9. Model Inference and Multiple Structure Generation
def generate_diverse_structures(model, features, seq_length, num_structures=5, noise_scale=0.05, device='cpu'):
    model.eval()
    structures = []
    with torch.no_grad():
        # 确保长度参数格式正确
        if isinstance(seq_length, torch.Tensor):
            seq_length = seq_length.item()
            
        for i in range(num_structures):
            if i > 0:
                noise = torch.randn_like(features) * noise_scale
                perturbed_features = features + noise
            else:
                perturbed_features = features
                
            # 显式转换长度参数为张量
            length_tensor = torch.tensor([seq_length], dtype=torch.long)
            output = model(
                perturbed_features.unsqueeze(0),
                lengths=length_tensor.to(device)  # 保持设备一致
            )
            coords = output[0, :seq_length, :].cpu().numpy()
            structures.append(coords)
    return structures

def generate_predictions(model, dataloader, device, num_predictions=5):
    model.eval()
    all_predictions = {}
    for batch in dataloader:
        features = batch['features'].to(device)
        lengths = batch['lengths'].tolist()  # 转换为列表
        ids = batch['ids']
        
        for i in range(features.size(0)):
            seq_id = ids[i]
            seq_len = lengths[i]
            
            # 生成预测
            seq_features = features[i, :seq_len, :]
            predictions = generate_diverse_structures(
                model,
                seq_features,
                seq_len,  # 直接使用整数值
                num_structures=num_predictions,
                device=device
            )
            all_predictions[seq_id] = predictions
    return all_predictions

# 10. Submission File Generation

def create_submission_file(predictions, test_sequences_df, output_file='submission.csv'):
    submission_rows = []
    for _, row in test_sequences_df.iterrows():
        seq_id = row['target_id']
        sequence = row['sequence']
        if seq_id in predictions:
            pred_structures = predictions[seq_id]
            num_structures = len(pred_structures)
            for i in range(len(sequence)):
                submission_row = {
                    'ID': f"{seq_id}_{i+1}",
                    'resname': sequence[i],
                    'resid': i+1
                }
                for j in range(5):
                    if j < num_structures:
                        coords = pred_structures[j][i]
                        submission_row[f'x_{j+1}'] = coords[0]
                        submission_row[f'y_{j+1}'] = coords[1]
                        submission_row[f'z_{j+1}'] = coords[2]
                    else:
                        submission_row[f'x_{j+1}'] = submission_row[f'x_{j}']
                        submission_row[f'y_{j+1}'] = submission_row[f'y_{j}']
                        submission_row[f'z_{j+1}'] = submission_row[f'z_{j}']
                submission_rows.append(submission_row)
    submission_df = pd.DataFrame(submission_rows)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        submission_df.to_csv(output_file, index=False)
        print(f"文件已成功保存至: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return None
    
    return submission_df

# 11. Visualization Functions
def visualize_3d_structure(coords, title="RNA 3D Structure"):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', marker='o', s=30, label="C1' atoms")
    for i in range(len(coords) - 1):
        ax.plot([coords[i, 0], coords[i+1, 0]], 
                [coords[i, 1], coords[i+1, 1]], 
                [coords[i, 2], coords[i+1, 2]], 'k-', lw=1)
    ax.set_title(title)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# 12. (Optional) Ensemble Modeling
class ModelEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
    def predict(self, features, seq_lengths=None):
        all_predictions = []
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                output = model(features, seq_lengths)
                all_predictions.append(output * self.weights[i])
        return sum(all_predictions)

# 13. Main Execution
def main():
    print("\n--- Main execution ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = train_data[0]['features'].shape[1]
    model = RNAFoldingModel(input_dim=input_dim).to(device)
    
    print("\nStarting model training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        epochs=20,  # 使用正确的参数名
        lr=0.0005,
        device=device
    )
    
    
    print("\nGenerating predictions on test data...")
    test_predictions = generate_predictions(trained_model, test_loader, device, num_predictions=5)
    print("\nPredictions generated.")
    
    print("\nCreating submission file...")
    submission_file = create_submission_file(test_predictions, test_sequences)
    print(f"\nSubmission file created: submission.csv")
    print(submission_file.head())
    
    print("\nVisualizing a sample prediction (first test sequence)...")
    sample_seq_id = test_sequences['target_id'].iloc[0]
    if sample_seq_id in test_predictions:
        sample_prediction = test_predictions[sample_seq_id][0]
        visualize_3d_structure(sample_prediction, title=f"Predicted 3D Structure - {sample_seq_id}")
        print(f"Visualization saved for {sample_seq_id}.")
    else:
        print("No prediction found for the first test sequence for visualization.")
    
    print("\n--- Main execution completed ---")

if __name__ == '__main__':
    main()

print("\nNotebook execution finished.")

