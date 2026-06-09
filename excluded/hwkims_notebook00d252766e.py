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
# 경로를 환경에 맞게 수정해주세요.
# 예: '/kaggle/input/stanford-rna-3d-folding/' 또는 './stanford-rna-3d-folding/'
# 이 코드는 Kaggle 환경을 가정합니다. 로컬에서 실행 시 경로를 확인하세요.
try:
    base_path = '/kaggle/input/stanford-rna-3d-folding/'
    if not os.path.exists(base_path):
        print(f"Warning: Kaggle input path '{base_path}' not found. Trying current directory.")
        # 로컬 테스트를 위해 현재 디렉토리를 사용하거나, 사용자 정의 경로를 설정할 수 있습니다.
        # 예를 들어, 데이터셋이 'data' 폴더에 있다면 base_path = './data/' 로 설정
        base_path = './' # 현재 디렉토리 또는 사용자가 지정한 경로로 수정
        # 로컬 테스트 시에는 파일들이 실제로 해당 경로에 있는지 확인해야 합니다.
        # train_sequences.csv, train_labels.csv 등이 base_path 아래에 있어야 합니다.
        if not os.path.exists(os.path.join(base_path, 'train_sequences.csv')):
            print("Error: train_sequences.csv not found in the specified base_path. Please check your data paths.")
            exit()


    train_sequences = pd.read_csv(os.path.join(base_path, 'train_sequences.csv'))
    train_labels = pd.read_csv(os.path.join(base_path, 'train_labels.csv'))
    validation_sequences = pd.read_csv(os.path.join(base_path, 'validation_sequences.csv'))
    validation_labels = pd.read_csv(os.path.join(base_path, 'validation_labels.csv'))
    test_sequences = pd.read_csv(os.path.join(base_path, 'test_sequences.csv'))
    sample_submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please ensure the data files are in the correct path.")
    print("If running locally, you might need to download the dataset and adjust the paths.")
    exit()


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
# 저장 경로를 워킹 디렉토리로 변경 (Kaggle에서는 /kaggle/working/)
output_viz_path = 'sequence_length_distribution.png'
if 'KAGGLE_WORKING_DIR' in os.environ:
    output_viz_path = os.path.join(os.environ['KAGGLE_WORKING_DIR'], output_viz_path)
plt.savefig(output_viz_path)
plt.close()
print(f"Sequence length distribution plot saved to {output_viz_path}")

# 3. Data Preprocessing
def preprocess_sequence_data(sequences_df, labels_df=None, is_train=True):
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3} # T를 U로 매핑
    processed_data = []

    for idx, row in sequences_df.iterrows():
        seq_id = row['target_id']
        sequence = row['sequence']
        numerical_seq = [nucleotide_map.get(nuc, 4) for nuc in sequence] # 4 for unknown

        structures = None
        if labels_df is not None: # is_train 조건 제거, labels_df 유무로 판단
            # 레이블 파일의 ID는 'target_id_residue_number' 형식이므로, target_id로 시작하는지 확인
            sequence_labels = labels_df[labels_df['ID'].str.startswith(seq_id + '_')].copy() # SettingWithCopyWarning 방지

            if not sequence_labels.empty:
                # resname, resid 컬럼 제외하고 x_i, y_i, z_i 컬럼만 선택
                coord_cols = [col for col in sequence_labels.columns if col.startswith(('x_', 'y_', 'z_'))]
                num_structures = len(coord_cols) // 3 # 각 구조당 x,y,z 3개 좌표

                structures = []
                for i in range(1, num_structures + 1):
                    coords_list = []
                    # 레이블 파일은 residue 순서대로 정렬되어 있다고 가정
                    # 'resid' 컬럼이 있다면 이를 기준으로 정렬하는 것이 더 안전
                    if 'resid' in sequence_labels.columns:
                        sequence_labels.sort_values('resid', inplace=True)

                    current_structure_coords = sequence_labels[[f'x_{i}', f'y_{i}', f'z_{i}']].values
                    
                    # 시퀀스 길이와 좌표 개수가 일치하는지 확인
                    if len(current_structure_coords) != len(sequence):
                        # print(f"Warning: Mismatch in length for {seq_id}, structure {i}. Seq_len: {len(sequence)}, Coords_len: {len(current_structure_coords)}. Skipping this structure.")
                        # 패딩 또는 다른 처리 방법 고려 가능
                        # 간단히 길이만큼만 사용하거나, 0으로 채우거나, 또는 이 구조를 건너뛸 수 있음
                        # 여기서는 길이만큼만 사용 (만약 좌표가 더 짧다면) 또는 시퀀스 길이로 자름 (좌표가 더 길다면)
                        min_len = min(len(current_structure_coords), len(sequence))
                        current_structure_coords = current_structure_coords[:min_len]
                        if len(current_structure_coords) < len(sequence): # 좌표가 시퀀스보다 짧으면 패딩
                             padding = np.zeros((len(sequence) - len(current_structure_coords), 3))
                             current_structure_coords = np.vstack((current_structure_coords, padding))

                    coords = np.array(current_structure_coords)

                    # Normalize coordinates per sequence (center and scale)
                    mean = np.mean(coords, axis=0)
                    std = np.std(coords, axis=0) + 1e-8 # 분모 0 방지
                    coords_norm = (coords - mean) / std
                    structures.append(coords_norm)
            # else:
                # print(f"No labels found for {seq_id}")
        # else:
            # print(f"No labels_df provided for {seq_id} (e.g., for test data)")

        processed_data.append({
            'id': seq_id,
            'sequence': numerical_seq,
            'structures': structures # structures가 None일 수 있음 (테스트 데이터 또는 레이블 없는 학습 데이터)
        })
    return processed_data


print("Preprocessing training data...")
train_data = preprocess_sequence_data(train_sequences, train_labels)
print("Preprocessing validation data...")
# validation_labels가 train_labels와 형식이 다를 수 있음에 유의
validation_data = preprocess_sequence_data(validation_sequences, validation_labels)
print("Preprocessing test data...")
test_data = preprocess_sequence_data(test_sequences, labels_df=None) # 테스트 데이터에는 레이블 없음

# 4. Feature Engineering
def extract_sequence_features(sequence_numerical): # 입력은 수치화된 시퀀스
    one_hot = np.zeros((len(sequence_numerical), 5)) # 0,1,2,3 (ACGU) + 4 (Unknown)
    for i, nucleotide_code in enumerate(sequence_numerical):
        one_hot[i, nucleotide_code] = 1

    gc_content_feature = []
    window_size = 5
    for i in range(len(sequence_numerical)):
        start = max(0, i - window_size // 2)
        end = min(len(sequence_numerical), i + window_size // 2 + 1)
        window = sequence_numerical[start:end]
        # G is 2, C is 1
        gc_count = sum(1 for n_code in window if n_code == 1 or n_code == 2)
        gc_content_feature.append(gc_count / len(window) if len(window) > 0 else 0)

    positions = np.array([[i / len(sequence_numerical)] for i in range(len(sequence_numerical))])
    features = np.hstack((one_hot, positions, np.array(gc_content_feature).reshape(-1, 1)))
    return features

print("Extracting sequence features...")
for i, data_item in enumerate(train_data):
    train_data[i]['features'] = extract_sequence_features(data_item['sequence'])
for i, data_item in enumerate(validation_data):
    validation_data[i]['features'] = extract_sequence_features(data_item['sequence'])
for i, data_item in enumerate(test_data):
    test_data[i]['features'] = extract_sequence_features(data_item['sequence'])

# 5. RNA Secondary Structure Prediction (simple rule-based)
def predict_rna_secondary_structure(sequence_numerical): # 입력은 수치화된 시퀀스
    nucleotide_map_inv = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'X'} # X for unknown
    seq_chars = [nucleotide_map_inv.get(n_code, 'X') for n_code in sequence_numerical]
    structure = ['.' for _ in range(len(seq_chars))]
    complementary = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'X': None}

    # 더 나은 로직: 스택 기반 또는 동적 프로그래밍의 단순화된 버전
    # 여기서는 최소 루프 크기 (예: 3) 고려
    min_loop_length = 3
    for i in range(len(seq_chars)):
        if structure[i] != '.':
            continue
        for j in range(len(seq_chars) - 1, i + min_loop_length, -1): # j > i + min_loop_length
            if structure[j] != '.':
                continue
            # 슈도넛 방지를 위해 간단히 가장 바깥쪽 쌍부터 찾음
            # (더 정교한 알고리즘은 슈도넛을 허용하거나 다른 방식으로 처리)
            can_pair = True
            # for k in range(i + 1, j): # 현재 쌍 내부에 다른 쌍이 있는지 (간단한 슈도넛 체크)
            #     if structure[k] != '.': # 이미 페어링된 뉴클레오티드가 있다면
            #         # 이것만으로는 슈도넛을 완벽히 막을 수 없음. 더 정교한 로직 필요.
            #         # 여기서는 슈도넛을 고려하지 않는 매우 단순한 페어링
            #         pass

            if complementary.get(seq_chars[i]) == seq_chars[j] and can_pair:
                structure[i] = '('
                structure[j] = ')'
                # 쌍을 찾으면 내부 루프 중단 (가장 멀리 있는 쌍 우선)
                break
    return ''.join(structure)

def enhance_features_with_ss(data_list): # 변수명 변경
    for i, item in enumerate(data_list):
        seq_num = item['sequence']
        ss = predict_rna_secondary_structure(seq_num)
        ss_features = np.zeros((len(ss), 3)) # '.', '(', ')'
        for j, char_ss in enumerate(ss): # 변수명 변경
            if char_ss == '.':
                ss_features[j, 0] = 1
            elif char_ss == '(':
                ss_features[j, 1] = 1
            elif char_ss == ')':
                ss_features[j, 2] = 1
        data_list[i]['features'] = np.hstack((item['features'], ss_features))
    return data_list

print("Enhancing features with secondary structure information...")
train_data = enhance_features_with_ss(train_data)
validation_data = enhance_features_with_ss(validation_data)
test_data = enhance_features_with_ss(test_data)


# 6. PyTorch Dataset and DataLoader
class RNADataset(Dataset):
    def __init__(self, data, augment=False, num_structures_to_use=1):
        self.data = [item for item in data if item['structures'] is not None and len(item['structures']) > 0]
        if not self.data and data: # 원본 데이터는 있었는데 필터링 후 비었다면 경고
            print("Warning: No valid structures found in the provided data for RNADataset. DataLoader might be empty.")
        self.augment = augment
        self.num_structures_to_use = num_structures_to_use # 사용할 구조의 수 (주로 첫 번째)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features'].copy() # 원본 수정을 피하기 위해 복사

        # Data augmentation
        if self.augment and random.random() < 0.5: # 증강 확률
            # Mutation
            if random.random() < 0.2: # 돌연변이 확률
                if len(features) > 0:
                    mut_idx = random.randint(0, len(features)-1)
                    # one-hot 부분 (앞 5개 컬럼)만 변경
                    new_nucleotide_one_hot = np.eye(5)[random.choice([0,1,2,3])] # ACGU 중 하나로 변경
                    features[mut_idx, :5] = new_nucleotide_one_hot
            # Reversing a segment
            if len(features) > 20 and random.random() < 0.2: # 세그먼트 반전 확률
                start = random.randint(0, len(features) - 10)
                segment_len = random.randint(5, 10)
                end = min(start + segment_len, len(features))
                features[start:end] = features[start:end][::-1]

        features_tensor = torch.tensor(features, dtype=torch.float32)

        # 타겟 구조 선택 (여러 구조가 있을 경우 첫 번째 또는 랜덤하게 선택 가능)
        # preprocess_sequence_data 에서 여러 구조를 로드했다면, 그 중 하나를 선택
        # 여기서는 첫 번째 구조를 사용한다고 가정 (item['structures'][0])
        if item['structures'] and len(item['structures']) > 0:
            # 사용할 구조의 인덱스 (예: 항상 첫 번째, 또는 랜덤)
            # 여기서는 항상 첫 번째 구조를 사용
            struct_idx_to_use = 0
            target_coords = item['structures'][struct_idx_to_use]
            target_tensor = torch.tensor(target_coords, dtype=torch.float32)
        else:
            # 테스트 데이터셋의 경우 타겟이 없을 수 있음
            # 또는 학습 데이터셋인데 특정 아이템에 구조가 없는 경우 (이런 경우는 필터링 되어야 함)
            target_tensor = torch.empty(0,3, dtype=torch.float32) # 빈 텐서 또는 None

        return {
            'features': features_tensor,
            'target': target_tensor,
            'length': features_tensor.shape[0],
            'id': item['id']
        }

class RNADatasetTest(Dataset): # 테스트 데이터셋을 위한 별도 클래스
    def __init__(self, data):
        self.data = data # 테스트 데이터에는 'structures'가 없을 수 있음

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        features_tensor = torch.tensor(features, dtype=torch.float32)

        return {
            'features': features_tensor,
            'target': None, # 테스트 시에는 타겟이 없음
            'length': features_tensor.shape[0],
            'id': item['id']
        }


def collate_fn(batch):
    # 길이 기준으로 내림차순 정렬 (pack_padded_sequence에 필요)
    # RNADataset에서 이미 length를 int로 반환하므로 .item() 불필요
    sorted_batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    features_list = [x['features'] for x in sorted_batch]
    lengths_list = [x['length'] for x in sorted_batch] # 이미 int list
    ids_list = [x['id'] for x in sorted_batch]

    # features 패딩
    # features_list[0].shape[1] 은 특성 차원
    padded_features = nn.utils.rnn.pad_sequence(features_list, batch_first=True, padding_value=0.0)

    targets_exist = all(x['target'] is not None and x['target'].nelement() > 0 for x in sorted_batch)

    if targets_exist:
        targets_list = [x['target'] for x in sorted_batch]
        padded_targets = nn.utils.rnn.pad_sequence(targets_list, batch_first=True, padding_value=0.0) # 0으로 패딩
    else:
        padded_targets = None # 타겟이 없는 경우 (예: 테스트 데이터)

    return {
        'features': padded_features,
        'targets': padded_targets,
        'lengths': torch.tensor(lengths_list, dtype=torch.long), # LSTM에 사용될 길이 (정수형 텐서)
        'ids': ids_list
    }

# 데이터셋 인스턴스 생성 전에 train_data와 validation_data에 유효한 구조가 있는지 확인
# preprocess_sequence_data에서 structures가 None이거나 비어있을 수 있음.
# RNADataset은 유효한 structures가 있는 아이템만 사용하도록 수정됨.
train_dataset = RNADataset(train_data, augment=True)
validation_dataset = RNADataset(validation_data) # 검증 시에는 증강 안 함
test_dataset = RNADatasetTest(test_data) # 테스트용 데이터셋

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(validation_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# DataLoader 인스턴스 생성
# num_workers는 환경에 따라 조절. Kaggle에서는 보통 2 또는 4.
# persistent_workers는 DataLoader가 워커 프로세스를 유지하도록 하여 에폭 간 오버헤드 줄임
num_w = 2 if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else 0 # 로컬에서는 0이 더 안정적일 수 있음

train_loader = DataLoader(
    train_dataset,
    batch_size=8, # 배치 크기 조절 가능
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=num_w,
    persistent_workers=True if num_w > 0 else False
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8, # 검증 시에도 동일한 배치 크기 사용 가능
    shuffle=False, # 검증 시에는 셔플 불필요
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=num_w,
    persistent_workers=True if num_w > 0 else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4, # 테스트 시에는 예측 안정성을 위해 더 작은 배치 사용 가능
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=num_w,
    persistent_workers=True if num_w > 0 else False
)


# 7. Model Architecture
class RNAFoldingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_lstm_layers=3, num_transformer_layers=3, nhead_transformer=8): # 파라미터 조정
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # BiLSTM Encoder
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0 # 마지막 레이어 제외한 드롭아웃
        )

        # Transformer Encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * hidden_dim, # BiLSTM 출력 차원
            nhead=nhead_transformer,
            dim_feedforward=hidden_dim * 4, # 일반적인 설정 (d_model * 4)
            dropout=0.2,
            activation='gelu', # GELU 사용
            batch_first=True # batch_first=True로 설정
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_transformer_layers
        )

        # Prediction Head (좌표 3개 예측)
        # Transformer 출력 후 바로 Linear 레이어 사용
        self.fc_out = nn.Linear(2 * hidden_dim, 3)


    def forward(self, x, lengths):
        # x: (batch, seq_len, features)
        # lengths: (batch) - 각 시퀀스의 실제 길이

        # BiLSTM 처리
        # lengths를 CPU로 옮기고 리스트로 변환해야 할 수 있음
        # collate_fn에서 이미 lengths를 LongTensor로 반환함
        # pack_padded_sequence는 lengths가 CPU에 있는 것을 기대할 수 있음
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False # collate_fn에서 정렬하므로 enforce_sorted=True 가능
        )
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # lstm_out: (batch, seq_len, 2 * hidden_dim)

        # Transformer Encoder
        # Transformer는 패딩 마스크가 필요할 수 있음
        # src_key_padding_mask: (batch, seq_len)
        # True인 위치는 무시됨
        max_len = lstm_out.size(1)
        src_key_padding_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]

        transformer_out = self.transformer_encoder(lstm_out, src_key_padding_mask=src_key_padding_mask)
        # transformer_out: (batch, seq_len, 2 * hidden_dim)

        # 최종 좌표 예측
        predictions = self.fc_out(transformer_out)
        # predictions: (batch, seq_len, 3)

        return predictions

# 8. Loss Function and Evaluation Metric
class GeometricLoss(nn.Module):
    def __init__(self, coord_weight=0.6, dist_weight=0.4, bond_len_weight=0.0): # 가중치 조절 가능
        super().__init__()
        self.coord_weight = coord_weight
        self.dist_weight = dist_weight
        self.bond_len_weight = bond_len_weight # 새로운 항: 결합 길이 손실

        self.coord_loss_fn = nn.SmoothL1Loss(reduction='none') # 각 요소별 손실 계산
        self.dist_loss_fn = nn.MSELoss(reduction='none')
        self.bond_len_loss_fn = nn.MSELoss(reduction='none') # 또는 L1Loss

        self.target_bond_length = 3.8 # 대략적인 C1'-C1' 결합 길이 (Angstrom) - 조정 필요

    def forward(self, pred_coords, target_coords, lengths):
        batch_size = pred_coords.size(0)
        total_loss = 0
        num_valid_samples = 0

        for i in range(batch_size):
            l = lengths[i].item()
            if l < 2: continue # 유효한 길이를 가진 샘플만 처리

            pred_i = pred_coords[i, :l]    # (len, 3)
            target_i = target_coords[i, :l]  # (len, 3)

            # 1. Coordinate Loss (SmoothL1)
            loss_coord = self.coord_loss_fn(pred_i, target_i).mean() # 평균내서 스칼라로

            # 2. Distance Matrix Loss (MSE)
            pred_dist_sq = torch.cdist(pred_i.unsqueeze(0), pred_i.unsqueeze(0), p=2).squeeze(0) # (len, len)
            target_dist_sq = torch.cdist(target_i.unsqueeze(0), target_i.unsqueeze(0), p=2).squeeze(0)
            loss_dist = self.dist_loss_fn(pred_dist_sq, target_dist_sq).mean() # 평균내서 스칼라로

            # 3. (Optional) Bond Length Regularization Loss
            loss_bond = 0
            if self.bond_len_weight > 0 and l > 1:
                # 예측된 구조에서 인접한 C1' 원자 간의 거리 계산
                pred_bond_lengths = torch.norm(pred_i[1:] - pred_i[:-1], p=2, dim=1)
                target_bond_lengths_ideal = torch.full_like(pred_bond_lengths, self.target_bond_length)
                loss_bond = self.bond_len_loss_fn(pred_bond_lengths, target_bond_lengths_ideal).mean()

            sample_loss = (self.coord_weight * loss_coord +
                           self.dist_weight * loss_dist +
                           self.bond_len_weight * loss_bond)
            total_loss += sample_loss
            num_valid_samples +=1

        return total_loss / num_valid_samples if num_valid_samples > 0 else torch.tensor(0.0, device=pred_coords.device)


def kabsch_align(P, Q):
    """
    Aligns two sets of points P and Q using the Kabsch algorithm.
    P, Q: Nx3 numpy arrays
    Returns R (rotation matrix), t (translation vector) such that P_aligned = P @ R + t is closest to Q.
    And returns P_aligned.
    """
    P = P.copy()
    Q = Q.copy()

    # Centroid
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P -= centroid_P
    Q -= centroid_Q

    # Covariance matrix
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection check
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_Q - centroid_P @ R
    P_aligned = P @ R + centroid_Q # P를 Q에 맞춤. P@R은 Q의 중심에 대해 회전된 P.
                                   # 여기에 Q의 중심을 더해줌.
                                   # 또는 (P @ R + centroid_P @ R + t - centroid_P @ R)
                                   # = (P - centroid_P) @ R + centroid_Q
    return R, t, P_aligned


def calculate_tm_score_usalign_like(pred_coords_np, true_coords_np):
    """
    Calculates TM-score after Kabsch alignment.
    This is a simplified version and might not perfectly match US-align.
    pred_coords_np, true_coords_np: (L, 3) numpy arrays
    """
    L_ref = true_coords_np.shape[0]
    L_pred = pred_coords_np.shape[0]

    if L_ref == 0 or L_pred == 0:
        return 0.0

    # US-align은 L_pred와 L_ref가 달라도 최적의 부분정렬을 찾지만,
    # 여기서는 간단히 길이가 같은 부분만 고려하거나, 짧은 쪽에 맞춰 자름.
    # 실제 대회에서는 US-align이 이를 처리. 여기서는 길이가 같다고 가정하거나, 맞춰줌.
    common_len = min(L_ref, L_pred)
    if common_len < 2: # TM-score 계산에 의미 없는 길이
        return 0.0

    pred_coords_aligned_subset = pred_coords_np[:common_len]
    true_coords_subset = true_coords_np[:common_len]

    # Kabsch alignment
    # P를 Q에 맞춤. pred_coords_aligned_subset을 true_coords_subset에 맞춤.
    _, _, pred_aligned = kabsch_align(pred_coords_aligned_subset, true_coords_subset)

    # d0 calculation based on L_ref (original length of reference)
    if L_ref >= 30:
        d0 = 1.24 * (L_ref - 15)**(1/3) - 1.8
    elif L_ref >= 24: d0 = 0.7
    elif L_ref >= 20: d0 = 0.6
    elif L_ref >= 16: d0 = 0.5
    elif L_ref >= 12: d0 = 0.4
    else: d0 = 0.3 # For L_ref < 12
    d0 = max(d0, 0.5) # As per AlphaFold

    # Calculate TM-score sum over the aligned common length
    sum_tm = 0
    for i in range(common_len):
        di_sq = np.sum((pred_aligned[i] - true_coords_subset[i])**2) # squared distance
        sum_tm += 1 / (1 + (di_sq / (d0**2)))

    # Normalize by L_ref (as per standard TM-score definition)
    tm_score = sum_tm / L_ref
    return np.clip(tm_score, 0.0, 1.0)


# 9. Training Loop
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cpu', patience=10):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) # AdamW 추천
    criterion = GeometricLoss(coord_weight=0.5, dist_weight=0.5, bond_len_weight=0.1).to(device) # 손실 함수도 디바이스로

    # Learning rate scheduler (옵션)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True)

    best_val_tm_score = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'tm_score': []}

    # 모델 저장 경로
    model_save_path = 'best_model.pth'
    if 'KAGGLE_WORKING_DIR' in os.environ:
        model_save_path = os.path.join(os.environ['KAGGLE_WORKING_DIR'], model_save_path)


    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device) # collate_fn에서 None일 수 있음 (RNADataset에서 필터링)
            lengths = batch['lengths'].to(device) # pack_padded_sequence 위해 CPU로 옮길 필요 없음 (함수 내에서 처리)
            # ids = batch['ids'] # 학습 시에는 보통 사용 안 함

            if targets is None: # RNADataset에서 걸러지지만, 안전장치
                continue

            optimizer.zero_grad()
            outputs = model(features, lengths) # lengths는 LongTensor 여야 함

            loss = criterion(outputs, targets, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 그래디언트 클리핑
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        history['train_loss'].append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        all_tm_scores = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)
                lengths = batch['lengths'] # .to(device) 불필요, 어차피 CPU에서 사용하거나 model forward에서 처리
                # ids = batch['ids']

                if targets is None:
                    continue

                outputs = model(features, lengths.to(device)) # 모델 forward에는 device에 있는 lengths 필요
                loss = criterion(outputs, targets, lengths) # criterion 내에서는 lengths.item() 사용
                total_val_loss += loss.item()
                num_val_batches += 1

                # TM-Score 계산
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                current_lengths_list = lengths.cpu().tolist()

                for i in range(outputs_np.shape[0]):
                    l = current_lengths_list[i]
                    if l < 5: continue # 매우 짧은 시퀀스 TM-score 계산에서 제외

                    pred_coords = outputs_np[i, :l, :]
                    true_coords = targets_np[i, :l, :]

                    # 정규화된 좌표로 TM-score 계산 시 의미가 다를 수 있음.
                    # 이상적으로는 원래 스케일로 복원 후 계산하거나,
                    # US-align과 같은 도구를 사용해야 함.
                    # 여기서는 정규화된 상태로 계산 (상대적 비교용)
                    # tm = calculate_tm_score(pred_coords, true_coords) # 이전 버전
                    tm = calculate_tm_score_usalign_like(pred_coords, true_coords) # Kabsch 정렬 포함
                    all_tm_scores.append(tm)

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        avg_tm_score = np.mean(all_tm_scores) if all_tm_scores else 0.0
        history['val_loss'].append(avg_val_loss)
        history['tm_score'].append(avg_tm_score)

        print(f"Epoch {epoch+1}/{epochs} => "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val TM-Score: {avg_tm_score:.4f}")

        scheduler.step(avg_tm_score) # TM-score 기준으로 LR 스케줄링

        if avg_tm_score > best_val_tm_score:
            best_val_tm_score = avg_tm_score
            torch.save(model.state_dict(), model_save_path)
            print(f"   Best model saved with TM-Score: {best_val_tm_score:.4f} at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"   Early stopping triggered after {patience} epochs without improvement.")
                break
    
    print(f"Training finished. Best validation TM-Score: {best_val_tm_score:.4f}")
    # 가장 좋은 모델 로드 (만약 저장되었다면)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded best model from {model_save_path}")
    return model, history


# 10. Model Inference and Multiple Structure Generation
def generate_diverse_structures(model, features_tensor, seq_length_int, num_structures=5, noise_scale=0.02, device='cpu'):
    model.eval() # 평가 모드
    structures_list = []
    with torch.no_grad():
        # features_tensor는 (seq_len, feature_dim) 형태의 단일 시퀀스 특성
        # seq_length_int는 해당 시퀀스의 실제 길이 (정수)

        for i in range(num_structures):
            current_features = features_tensor.clone() # 원본 변경 방지
            if i > 0: # 첫 번째 예측은 노이즈 없이
                noise = torch.randn_like(current_features) * noise_scale
                current_features += noise

            # 모델 입력 형태로 변경: (batch_size=1, seq_len, feature_dim)
            current_features_batch = current_features.unsqueeze(0).to(device)
            # 길이도 텐서로: (batch_size=1)
            lengths_batch = torch.tensor([seq_length_int], dtype=torch.long).to(device)

            output_coords = model(current_features_batch, lengths_batch)
            # output_coords: (1, seq_len, 3)

            # 실제 길이만큼만 잘라내고 CPU로 이동, numpy로 변환
            coords_np = output_coords[0, :seq_length_int, :].cpu().numpy()
            structures_list.append(coords_np)
    return structures_list


def generate_predictions_for_submission(model, dataloader, device, num_predictions_per_seq=5):
    model.to(device)
    model.eval()
    all_predictions_dict = {} # key: seq_id, value: list of 5 predicted coord arrays

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader): # test_loader 사용
            features_batch = batch['features'].to(device) # (batch, max_len, feat_dim)
            lengths_list = batch['lengths'].cpu().tolist() # 각 시퀀스의 실제 길이 리스트
            ids_list = batch['ids']

            for i in range(features_batch.size(0)): # 배치 내 각 시퀀스에 대해
                seq_id = ids_list[i]
                seq_len_int = lengths_list[i]
                # 해당 시퀀스의 특성만 추출 (패딩 제외)
                # features_batch[i]는 (max_len, feat_dim)
                # 실제 길이만큼만 사용: features_batch[i, :seq_len_int, :]
                single_seq_features = features_batch[i, :seq_len_int, :].clone() # (seq_len, feat_dim)

                # 이 단일 시퀀스에 대해 다양한 구조 생성
                predicted_structures_for_seq = generate_diverse_structures(
                    model,
                    single_seq_features, # (seq_len, feat_dim) 텐서 전달
                    seq_len_int,         # 실제 길이 (int) 전달
                    num_structures=num_predictions_per_seq,
                    noise_scale=0.02, # 노이즈 스케일 조절 가능
                    device=device
                )
                all_predictions_dict[seq_id] = predicted_structures_for_seq
            if (batch_idx + 1) % 10 == 0:
                 print(f"  Processed batch {batch_idx+1}/{len(dataloader)} for predictions.")

    return all_predictions_dict


# 11. Submission File Generation
def create_submission_file(predictions_dict, test_sequences_df, output_filename='submission.csv'):
    submission_rows_list = []

    # test_sequences_df를 순회하며 각 ID에 대한 예측 찾기
    for _, row in test_sequences_df.iterrows():
        seq_id = row['target_id']
        sequence_str = row['sequence'] # 문자열 시퀀스
        seq_len = len(sequence_str)

        if seq_id in predictions_dict:
            pred_structures_for_id = predictions_dict[seq_id] # list of 5 (L,3) numpy arrays
            num_pred_structures = len(pred_structures_for_id)

            for residue_idx in range(seq_len): # 시퀀스의 각 잔기에 대해
                submission_row = {
                    'ID': f"{seq_id}_{residue_idx + 1}", # ID_resid 형식
                    'resname': sequence_str[residue_idx],
                    'resid': residue_idx + 1
                }

                for pred_idx in range(5): # 5개 예측 구조에 대해
                    if pred_idx < num_pred_structures:
                        # 해당 예측 구조에서 현재 잔기의 좌표
                        # pred_structures_for_id[pred_idx]는 (L,3) 배열
                        # residue_idx가 이 배열의 길이를 넘지 않는지 확인
                        if residue_idx < pred_structures_for_id[pred_idx].shape[0]:
                            coords = pred_structures_for_id[pred_idx][residue_idx]
                            submission_row[f'x_{pred_idx+1}'] = coords[0]
                            submission_row[f'y_{pred_idx+1}'] = coords[1]
                            submission_row[f'z_{pred_idx+1}'] = coords[2]
                        else: # 예측된 구조가 실제 시퀀스보다 짧은 경우 (이론상 발생 안해야 함)
                            # 이전 구조의 값으로 채우거나 0으로 채움
                            if pred_idx > 0:
                                submission_row[f'x_{pred_idx+1}'] = submission_row[f'x_{pred_idx}']
                                submission_row[f'y_{pred_idx+1}'] = submission_row[f'y_{pred_idx}']
                                submission_row[f'z_{pred_idx+1}'] = submission_row[f'z_{pred_idx}']
                            else: # 첫번째 예측인데도 짧으면 0
                                submission_row[f'x_{pred_idx+1}'] = 0.0
                                submission_row[f'y_{pred_idx+1}'] = 0.0
                                submission_row[f'z_{pred_idx+1}'] = 0.0
                    else: # 예측된 구조가 5개 미만인 경우, 마지막 유효한 예측으로 채움
                        # (generate_diverse_structures가 항상 5개를 반환하도록 설계되어 이 경우는 적음)
                        # 가장 마지막으로 성공적으로 가져온 좌표 사용
                        last_valid_pred_idx = num_pred_structures # 1-based index for column name
                        submission_row[f'x_{pred_idx+1}'] = submission_row[f'x_{last_valid_pred_idx}']
                        submission_row[f'y_{pred_idx+1}'] = submission_row[f'y_{last_valid_pred_idx}']
                        submission_row[f'z_{pred_idx+1}'] = submission_row[f'z_{last_valid_pred_idx}']
                submission_rows_list.append(submission_row)
        else:
            print(f"Warning: No predictions found for seq_id {seq_id}. Filling with zeros or placeholders.")
            # 예측이 없는 경우 (예: 오류 발생) - 샘플 제출 파일 형식에 맞게 placeholder 채우기
            for residue_idx in range(seq_len):
                submission_row = {
                    'ID': f"{seq_id}_{residue_idx + 1}",
                    'resname': sequence_str[residue_idx],
                    'resid': residue_idx + 1
                }
                for pred_idx in range(5):
                    submission_row[f'x_{pred_idx+1}'] = 0.0
                    submission_row[f'y_{pred_idx+1}'] = 0.0
                    submission_row[f'z_{pred_idx+1}'] = 0.0
                submission_rows_list.append(submission_row)


    submission_df = pd.DataFrame(submission_rows_list)

    # 파일 저장 경로 (Kaggle 환경 고려)
    if 'KAGGLE_WORKING_DIR' in os.environ:
        output_filepath = os.path.join(os.environ['KAGGLE_WORKING_DIR'], output_filename)
    else:
        output_filepath = output_filename

    try:
        submission_df.to_csv(output_filepath, index=False)
        print(f"Submission file successfully saved to: {os.path.abspath(output_filepath)}")
    except Exception as e:
        print(f"Error saving submission file: {e}")
        return None
    return submission_df


# 12. Visualization Functions
def visualize_3d_structure(coords_np, title="RNA_3D_Structure"):
    # matplotlib.pyplot을 함수 내에서 import하여 Kaggle 환경에서 GUI 백엔드 문제 방지
    import matplotlib
    matplotlib.use('Agg') # 비 GUI 백엔드 사용
    import matplotlib.pyplot as plt
    # 3D 플롯을 위해 Axes3D 임포트 필요
    from mpl_toolkits.mplot3d import Axes3D


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # C1' 원자 위치 산점도
    ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], c='blue', marker='o', s=50, label="C1' atoms", depthshade=True)

    # 원자 간 연결선 (backbone)
    for i in range(len(coords_np) - 1):
        ax.plot([coords_np[i, 0], coords_np[i+1, 0]],
                [coords_np[i, 1], coords_np[i+1, 1]],
                [coords_np[i, 2], coords_np[i+1, 2]], 'gray', lw=1.5)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.legend()
    ax.grid(True)

    # 파일명에 사용할 수 없는 문자 제거
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    viz_filename = f"{safe_title}.png"
    if 'KAGGLE_WORKING_DIR' in os.environ:
        viz_filepath = os.path.join(os.environ['KAGGLE_WORKING_DIR'], viz_filename)
    else:
        viz_filepath = viz_filename

    plt.savefig(viz_filepath)
    plt.close(fig) # 메모리 해제
    print(f"3D structure visualization saved to {viz_filepath}")


# 13. Main Execution
def main():
    print("\n--- Main execution ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 입력 차원 결정 (첫 번째 학습 데이터의 특성 개수)
    if not train_dataset:
        print("Error: Training dataset is empty. Cannot determine input_dim or train the model.")
        return
    
    # RNADataset.__getitem__은 딕셔너리를 반환하므로, features를 직접 접근
    # 또는 train_data에서 직접 가져올 수 있음
    # 안전하게 첫 번째 유효한 train_data 아이템에서 특성 차원 가져오기
    first_valid_train_item = next((item for item in train_data if 'features' in item and item['features'] is not None), None)
    if first_valid_train_item is None or first_valid_train_item['features'].shape[1] == 0:
        print("Error: Could not determine input_dim from training data. Features might be missing or empty.")
        # 예시로 고정된 값을 사용하거나, 오류 처리
        # 여기서는 임의의 값으로 설정하고 경고 (실제로는 오류로 중단해야 함)
        input_dim = 10 # 임시 값, 실제로는 오류 발생해야 함
        print(f"Warning: Setting input_dim to a default value of {input_dim}. This is likely an error.")
    else:
        input_dim = first_valid_train_item['features'].shape[1]
    
    print(f"Input feature dimension: {input_dim}")


    # 모델 인스턴스 생성
    model = RNAFoldingModel(
        input_dim=input_dim,
        hidden_dim=384,       # LSTM hidden dim, Transformer d_model의 절반
        num_lstm_layers=3,
        num_transformer_layers=4,
        nhead_transformer=8
    ).to(device)

    print(f"\nModel architecture: {model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")


    print("\nStarting model training...")
    # 에폭 수, 학습률 등은 실험을 통해 조절
    # 실제 대회에서는 더 많은 에폭과 세심한 하이퍼파라미터 튜닝 필요
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        epochs=30, # 예시 에폭 (실제로는 더 많이)
        lr=5e-5,   # 학습률 (조정 필요)
        device=device,
        patience=7 # Early stopping을 위한 patience
    )

    # 학습 과정 시각화 (옵션)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['tm_score'], label='Validation TM-Score')
    plt.title('Validation TM-Score History')
    plt.xlabel('Epoch')
    plt.ylabel('TM-Score')
    plt.legend()
    plt.tight_layout()
    history_plot_path = 'training_history.png'
    if 'KAGGLE_WORKING_DIR' in os.environ:
        history_plot_path = os.path.join(os.environ['KAGGLE_WORKING_DIR'], history_plot_path)
    plt.savefig(history_plot_path)
    plt.close()
    print(f"Training history plot saved to {history_plot_path}")


    print("\nGenerating predictions on test data...")
    # test_loader를 사용하여 예측 생성
    test_predictions_dict = generate_predictions_for_submission(
        trained_model, # 학습된 모델 사용
        test_loader,
        device,
        num_predictions_per_seq=5
    )
    print(f"\nPredictions generated for {len(test_predictions_dict)} test sequences.")

    print("\nCreating submission file...")
    submission_df = create_submission_file(
        test_predictions_dict,
        test_sequences, # test_sequences.csv에서 읽은 DataFrame
        output_filename='submission.csv'
    )

    if submission_df is not None:
        print(f"\nSubmission file preview (first 5 rows):")
        print(submission_df.head())
    else:
        print("Submission file generation failed.")


    print("\nVisualizing a sample prediction (first test sequence if available)...")
    if not test_sequences.empty:
        first_test_seq_id = test_sequences['target_id'].iloc[0]
        if first_test_seq_id in test_predictions_dict and test_predictions_dict[first_test_seq_id]:
            # 첫 번째 예측 구조 시각화
            sample_predicted_coords = test_predictions_dict[first_test_seq_id][0]
            visualize_3d_structure(sample_predicted_coords, title=f"Predicted_3D_Structure_{first_test_seq_id}")
        else:
            print(f"No prediction found or prediction is empty for the first test sequence ({first_test_seq_id}) for visualization.")
    else:
        print("Test sequences data is empty, skipping visualization.")

    print("\n--- Main execution completed ---")

if __name__ == '__main__':
    # 실행 시간 측정 (옵션)
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

print("\nNotebook execution finished.")

