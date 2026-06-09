!pip install biopython


import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

# 재현성을 위한 시드 설정
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# 설정
config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,  # 배치 사이즈 2로 증가
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",
    "epochs": 200,  # 에폭 수 200으로 증가
    "cos_epoch": 150,  # 코사인 스케줄러 시작점 조정
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 1.0,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter": 10, 
    "structural_violation_epoch": 50,
    "balance_weight": False,
    "msa_max_sequences": 32,  # MSA 최대 시퀀스 수
    "msa_feat_dim": 128,      # MSA 특성 차원
    "num_self_attn_layers": 4,  # Self-attention 레이어 수
    "num_cross_attn_layers": 4,  # Cross-attention 레이어 수
    "num_structure_module_layers": 8,  # 구조 모듈 레이어 수
    "dropout": 0.25,  # 드롭아웃 증가 (0.1 → 0.25)
    "n_heads": 8,
    "convert_to_rna": False  # T를 U로 변환 않도록 변경
}


# BioPython을 사용한 FASTA/MSA 파일 로드 함수
def load_msa_with_biopython(fasta_file, convert_to_rna=False):  # 기본값을 False로 변경
    """
    BioPython의 SeqIO를 사용하여 FASTA 형식의 MSA 파일에서 시퀀스를 로드합니다.
    
    Args:
        fasta_file (str): FASTA 파일 경로
        convert_to_rna (bool): DNA를 RNA로 변환할지 여부 (T→U)
        
    Returns:
        list: 시퀀스 리스트 (첫 번째는 쿼리 시퀀스)
    """
    try:
        from Bio import SeqIO
        sequences = []
        
        # FASTA 파일에서 레코드 읽기
        for record in SeqIO.parse(fasta_file, "fasta"):
            # 시퀀스를 문자열로 변환
            seq_str = str(record.seq).upper()
            
            # DNA를 RNA로 변환 (필요한 경우)
            if convert_to_rna:
                seq_str = seq_str.replace('T', 'U')
            
            # 소문자 처리 (일부 MSA 형식에서는 삽입을 나타냄)
            # 여기서는 모든 문자를 대문자로 유지
            
            sequences.append(seq_str)
        
        # 시퀀스가 없는 경우 기본값 반환
        if not sequences:
            return [""]
        
        return sequences
    
    except ImportError:
        print("BioPython이 설치되어 있지 않습니다. 기본 파서를 사용합니다.")
        return load_a3m_fallback(fasta_file, convert_to_rna)
    except Exception as e:
        print(f"FASTA 파일 로드 중 오류 발생: {str(e)}")
        return [""]

# BioPython 없을 경우를 위한 대체 파서
def load_a3m_fallback(fasta_file, convert_to_rna=False):  # 기본값을 False로 변경
    """기본 파서를 사용하여 FASTA 파일을 로드합니다."""
    sequences = []
    current_seq = ""
    
    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                # 대문자로 정규화
                clean_line = line.upper()
                # DNA를 RNA로 변환 (필요한 경우)
                if convert_to_rna:
                    clean_line = clean_line.replace('T', 'U')
                
                current_seq += clean_line
    
    if current_seq:
        sequences.append(current_seq)
    
    # 시퀀스가 없는 경우를 대비
    if not sequences:
        return [""]
    
    return sequences

# MSA 데이터를 처리하는 함수
def process_msa(msa_sequences, max_seq=32):
    """MSA 시퀀스를 처리하여 숫자 인코딩 텐서로 변환합니다."""
    # 시퀀스가 없거나 빈 문자열인 경우 처리
    if not msa_sequences or not msa_sequences[0]:
        # 기본값으로 1x1 배열 반환 (나중에 처리 가능하도록)
        return np.zeros((1, 1), dtype=np.int64)
    
    # 첫 번째 시퀀스는 쿼리 시퀀스
    query_seq = msa_sequences[0]
    
    # 최대 시퀀스 수 제한
    msa_sequences = msa_sequences[:max_seq]
    
    # 시퀀스 길이 확인
    seq_len = len(query_seq)
    
    # 시퀀스가 비어있으면 기본값 반환
    if seq_len == 0:
        return np.zeros((len(msa_sequences), 1), dtype=np.int64)
    
    # 원-핫 인코딩을 위한 사전 (A, C, G, U, T, - 갭, 기타 문자)
    # 'T'를 추가하여 DNA와 RNA 모두 처리 가능하도록 합니다
    vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 4, '-': 5, '.': 5}
    
    # MSA 시퀀스를 숫자로 변환
    msa_encoded = np.zeros((len(msa_sequences), seq_len), dtype=np.int64)
    
    # 각 시퀀스 처리
    for i, seq in enumerate(msa_sequences):
        # 현재 시퀀스가 쿼리 시퀀스보다 짧으면 패딩
        if len(seq) < seq_len:
            seq = seq + '-' * (seq_len - len(seq))
        
        # 시퀀스가 쿼리보다 길면 잘라내기
        seq = seq[:seq_len]
        
        # 문자별 처리
        for j, nt in enumerate(seq):
            # 소문자 'c'와 'a'는 비공식 삽입(들여쓰기) 표시로 사용될 수 있음
            # 여기서는 일반 문자로 처리
            nt_upper = nt.upper()
            
            if nt_upper in vocab:
                msa_encoded[i, j] = vocab[nt_upper]
            else:
                # 모르는 문자는 갭으로 처리
                msa_encoded[i, j] = vocab['-']
    
    return msa_encoded

# 데이터 로드 및 처리
def load_data(config):
    # 기본 데이터 로드
    train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
    train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
    train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0] + '_' + x.split("_")[1])
    
    all_xyz = []
    
    for pdb_id in tqdm(train_sequences['target_id']):
        df = train_labels[train_labels["pdb_id"] == pdb_id]
        xyz = df[['x_1', 'y_1', 'z_1']].to_numpy().astype('float32')
        xyz[xyz < -1e17] = float('NaN')
        all_xyz.append(xyz)
    
    # 필터링
    filter_nan = []
    max_len = 0
    for xyz in all_xyz:
        if len(xyz) > max_len:
            max_len = len(xyz)
        
        filter_nan.append((np.isnan(xyz).mean() <= 0.5) & 
                         (len(xyz) < config['max_len_filter']) & 
                         (len(xyz) > config['min_len_filter']))
    
    print(f"Longest sequence in train: {max_len}")
    
    filter_nan = np.array(filter_nan)
    non_nan_indices = np.arange(len(filter_nan))[filter_nan]
    
    train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
    all_xyz = [all_xyz[i] for i in non_nan_indices]
    
    # MSA 정보 추가
    msa_dir = "/kaggle/input/stanford-rna-3d-folding/MSA"
    msa_data = []
    
    for target_id in tqdm(train_sequences['target_id']):
        msa_file = os.path.join(msa_dir, f"{target_id}.MSA.fasta")
        
        if os.path.exists(msa_file):
            try:
                # BioPython을 사용한 MSA 로드
                msa_sequences = load_msa_with_biopython(msa_file, convert_to_rna=config['convert_to_rna'])
                
                # MSA 정보 출력 (디버깅)
                if len(msa_sequences) > 1:
                    print(f"MSA 로드 성공: {target_id}, {len(msa_sequences)} 시퀀스")
                
                msa_data.append(msa_sequences)
            except Exception as e:
                print(f"MSA 로드 실패: {target_id}, 오류: {str(e)}")
                # 기본값: 원본 시퀀스만 포함
                seq = train_sequences.loc[train_sequences['target_id'] == target_id, 'sequence'].values[0]
                msa_data.append([seq])
        else:
            # MSA 파일이 없는 경우, 원래 시퀀스만 포함
            seq = train_sequences.loc[train_sequences['target_id'] == target_id, 'sequence'].values[0]
            msa_data.append([seq])
            
            if target_id.startswith('8S'):  # 디버깅을 위해 몇 개의 누락된 파일만 기록
                print(f"MSA 파일 없음: {target_id}")
    
    # 데이터 패키징
    data = {
        "sequence": train_sequences['sequence'].to_list(),
        "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
        "description": train_sequences['description'].to_list(),
        "all_sequences": train_sequences['all_sequences'].to_list(),
        "xyz": all_xyz,
        "msa": msa_data
    }
    
    # 데이터 분할
    all_index = np.arange(len(data['sequence']))
    cutoff_date = pd.Timestamp(config['cutoff_date'])
    test_cutoff_date = pd.Timestamp(config['test_cutoff_date'])
    train_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) <= cutoff_date]
    test_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) > cutoff_date and pd.Timestamp(d) <= test_cutoff_date]
    
    print(f"Train size: {len(train_index)}")
    print(f"Test size: {len(test_index)}")
    
    return data, train_index, test_index

# RNA MSA 데이터셋 클래스
class RNA3D_MSA_Dataset(Dataset):
    def __init__(self, indices, data, config):
        self.indices = indices
        self.data = data
        self.config = config
        # 'T'를 추가하여 DNA와 RNA 모두 처리 가능하도록 수정
        self.tokens = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 4}
        self.max_seq = config['msa_max_sequences']
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        
        # 시퀀스 처리
        try:
            sequence = [self.tokens.get(nt, 0) for nt in self.data['sequence'][idx]]
            sequence = np.array(sequence)
            sequence = torch.tensor(sequence)
        except Exception as e:
            print(f"시퀀스 처리 오류: {str(e)}")
            # 기본값 설정 - 길이 1의 시퀀스
            sequence = torch.tensor([0])
        
        # MSA 데이터 가져오기 및 처리
        try:
            msa_sequences = self.data['msa'][idx]
            msa_encoded = process_msa(msa_sequences, self.max_seq)
            msa_encoded = torch.tensor(msa_encoded)
        except Exception as e:
            print(f"MSA 처리 오류: {str(e)}")
            # 기본값 설정 - 1x1 크기의 MSA
            msa_encoded = torch.tensor([[0]])
        
        # XYZ 좌표 가져오기
        try:
            xyz = self.data['xyz'][idx]
            xyz = torch.tensor(np.array(xyz))
        except Exception as e:
            print(f"XYZ 처리 오류: {str(e)}")
            # 기본값 설정 - 1x3 크기의 좌표
            xyz = torch.tensor([[0.0, 0.0, 0.0]])
        
        # 데이터 차원 일관성 확인
        seq_len = len(sequence)
        
        # 필요한 경우 시퀀스 자르기
        if seq_len > self.config['max_len']:
            crop_start = np.random.randint(seq_len - self.config['max_len'])
            crop_end = crop_start + self.config['max_len']
            
            sequence = sequence[crop_start:crop_end]
            
            # xyz 배열이 충분히 길면 자르기
            if len(xyz) >= crop_end:
                xyz = xyz[crop_start:crop_end]
            
            # msa 배열이 2차원이고 두 번째 차원이 충분히 길면 자르기
            if len(msa_encoded.shape) == 2 and msa_encoded.shape[1] >= crop_end:
                msa_encoded = msa_encoded[:, crop_start:crop_end]
        
        # MSA와 시퀀스 길이가 맞지 않는 경우 조정
        if len(msa_encoded.shape) == 2 and msa_encoded.shape[1] != len(sequence):
            # 더 작은 길이로 자르거나 패딩
            min_len = min(msa_encoded.shape[1], len(sequence))
            sequence = sequence[:min_len]
            
            if msa_encoded.shape[1] > min_len:
                msa_encoded = msa_encoded[:, :min_len]
            elif msa_encoded.shape[1] < min_len:
                # 패딩 추가
                padding = torch.zeros((msa_encoded.shape[0], min_len - msa_encoded.shape[1]), dtype=torch.long)
                msa_encoded = torch.cat([msa_encoded, padding], dim=1)
        
        return {
            'sequence': sequence,
            'xyz': xyz,
            'msa': msa_encoded
        }

# MSA Transformer 모델 구현
class MSAAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MSAAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, num_seqs, seq_len, _ = x.shape
        
        q = self.query(x).view(batch_size, num_seqs, seq_len, self.n_heads, self.head_dim)
        k = self.key(x).view(batch_size, num_seqs, seq_len, self.n_heads, self.head_dim)
        v = self.value(x).view(batch_size, num_seqs, seq_len, self.n_heads, self.head_dim)
        
        # 차원 변경: batch_size, n_heads, num_seqs, seq_len, head_dim
        q = q.permute(0, 3, 1, 2, 4)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        
        # 어텐션 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # 차원 되돌리기
        output = output.permute(0, 2, 3, 1, 4).contiguous()
        output = output.view(batch_size, num_seqs, seq_len, self.d_model)
        
        return self.out(output)

class RowAttention(nn.Module):
    """시퀀스 위치 간의 어텐션 (각 MSA 시퀀스 내에서)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(RowAttention, self).__init__()
        self.attention = MSAAttention(d_model, n_heads, dropout)
    
    def forward(self, x, mask=None):
        return self.attention(x, mask)

class ColumnAttention(nn.Module):
    """MSA 시퀀스 간의 어텐션 (각 위치에서)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(ColumnAttention, self).__init__()
        self.attention = MSAAttention(d_model, n_heads, dropout)
    
    def forward(self, x, mask=None):
        batch_size, num_seqs, seq_len, d_model = x.shape
        
        # 차원 변경: (batch_size, seq_len, num_seqs, d_model)
        x = x.permute(0, 2, 1, 3)
        
        if mask is not None:
            mask = mask.permute(0, 2, 1, 3)
        
        x = self.attention(x, mask)
        
        # 차원 되돌리기
        x = x.permute(0, 2, 1, 3)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MSATransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super(MSATransformerLayer, self).__init__()
        self.row_attn = RowAttention(d_model, n_heads, dropout)
        self.col_attn = ColumnAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 행 어텐션 (시퀀스 위치 간)
        row_attn_output = x + self.dropout1(self.row_attn(self.norm1(x), mask))
        
        # 열 어텐션 (MSA 시퀀스 간)
        col_attn_output = row_attn_output + self.dropout2(self.col_attn(self.norm2(row_attn_output), mask))
        
        # 피드포워드
        output = col_attn_output + self.dropout3(self.feed_forward(self.norm3(col_attn_output)))
        
        return output

class MSATransformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, d_ff=2048, dropout=0.1):
        super(MSATransformer, self).__init__()
        self.layers = nn.ModuleList([
            MSATransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value, mask=None):
        batch_size = query.shape[0]
        
        q = self.query(query)
        k = self.key(key_value)
        v = self.value(key_value)
        
        # 어텐션 차원 변경
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 어텐션 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # 차원 되돌리기
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        return self.out(output)

class CrossTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super(CrossTransformerLayer, self).__init__()
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, msa_repr, mask=None):
        # 크로스 어텐션
        cross_attn_output = x + self.dropout1(self.cross_attn(self.norm1(x), self.norm2(msa_repr), mask))
        
        # 피드포워드
        output = cross_attn_output + self.dropout2(self.feed_forward(self.norm3(cross_attn_output)))
        
        return output

class CrossTransformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, d_ff=2048, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.layers = nn.ModuleList([
            CrossTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, msa_repr, mask=None):
        for layer in self.layers:
            x = layer(x, msa_repr, mask)
        return self.norm(x)

class StructureModule(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, d_ff=2048, dropout=0.1):
        super(StructureModule, self).__init__()
        self.layers = nn.ModuleList([
            MSATransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.xyz_predictor = nn.Linear(d_model, 3)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        xyz = self.xyz_predictor(x)
        return xyz

class RNA_MSA_Folding(nn.Module):
    def __init__(self, config):
        super(RNA_MSA_Folding, self).__init__()
        self.config = config
        
        # 임베딩 레이어 - 'T' 토큰 추가로 6개 (A, C, G, U, T, 갭(-))
        self.seq_embedding = nn.Embedding(6, config['msa_feat_dim'])
        
        # 위치 인코딩
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, config['max_len'], config['msa_feat_dim']))
        
        # MSA Transformer
        self.msa_transformer = MSATransformer(
            d_model=config['msa_feat_dim'],
            n_heads=config['n_heads'],
            num_layers=config['num_self_attn_layers'],
            dropout=config['dropout']
        )
        
        # 단일 시퀀스 임베딩을 위한 추가 레이어
        self.single_seq_embedding = nn.Linear(config['msa_feat_dim'], config['msa_feat_dim'])
        
        # Cross Attention
        self.cross_transformer = CrossTransformer(
            d_model=config['msa_feat_dim'],
            n_heads=config['n_heads'],
            num_layers=config['num_cross_attn_layers'],
            dropout=config['dropout']
        )
        
        # 구조 모듈
        self.structure_module = StructureModule(
            d_model=config['msa_feat_dim'],
            n_heads=config['n_heads'],
            num_layers=config['num_structure_module_layers'],
            dropout=config['dropout']
        )
    
    def forward(self, seq, msa):
        # 입력 형태 확인 및 조정
        if len(msa.shape) != 3:
            print(f"Warning: MSA 형태가 비정상적입니다. 형태: {msa.shape}")
            # 최소 3차원 텐서로 조정
            if len(msa.shape) == 2:
                msa = msa.unsqueeze(0)
            elif len(msa.shape) == 1:
                msa = msa.unsqueeze(0).unsqueeze(0)
        
        batch_size, num_seqs, seq_len = msa.shape
        
        # MSA 임베딩
        msa_emb = self.seq_embedding(msa)  # [batch_size, num_seqs, seq_len, d_model]
        
        # 위치 인코딩 추가 (너무 긴 시퀀스 처리)
        if seq_len <= self.pos_embedding.shape[2]:
            pos_emb = self.pos_embedding[:, :, :seq_len, :]
        else:
            # 위치 인코딩 확장
            existing_pos = self.pos_embedding.squeeze(0).squeeze(0)  # [max_len, d_model]
            extended_pos = torch.zeros(seq_len, self.config['msa_feat_dim'], device=msa.device)
            extended_pos[:existing_pos.shape[0], :] = existing_pos
            # 나머지 부분은 마지막 위치 인코딩 복제
            if existing_pos.shape[0] > 0:
                extended_pos[existing_pos.shape[0]:, :] = existing_pos[-1, :]
            pos_emb = extended_pos.unsqueeze(0).unsqueeze(0)
            
        msa_emb = msa_emb + pos_emb
        
        # MSA 처리
        msa_repr = self.msa_transformer(msa_emb)  # [batch_size, num_seqs, seq_len, d_model]
        
        # 쿼리 시퀀스 (첫 번째 시퀀스)에 대한 임베딩
        query_repr = msa_repr[:, 0]  # [batch_size, seq_len, d_model]
        query_repr = self.single_seq_embedding(query_repr)
        
        # MSA 정보를 쿼리 시퀀스에 크로스 어텐션으로 통합
        query_refined = self.cross_transformer(query_repr, msa_repr)
        
        # 구조 모듈로 좌표 예측
        xyz_pred = self.structure_module(query_refined.unsqueeze(1)).squeeze(1)
        
        return xyz_pred


# 단일 시퀀스 버전의 RNA Folding 모델
class RNA_Single_Folding(nn.Module):
    def __init__(self, config):
        super(RNA_Single_Folding, self).__init__()
        self.config = config
        
        # 임베딩 레이어 - 'T' 토큰 추가로 5개 (A, C, G, U, T)
        self.seq_embedding = nn.Embedding(5, config['msa_feat_dim'])
        
        # 위치 인코딩
        self.pos_embedding = nn.Parameter(torch.zeros(1, config['max_len'], config['msa_feat_dim']))
        
        # 자기주의 트랜스포머 레이어 (MSA 없이 단일 시퀀스에 대해서만)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['msa_feat_dim'],
                nhead=config['n_heads'],
                dim_feedforward=config['msa_feat_dim'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_self_attn_layers'] + config['num_cross_attn_layers'])
        ])
        
        self.norm = LayerNorm(config['msa_feat_dim'])
        
        # 구조 모듈
        self.structure_module = StructureModule(
            d_model=config['msa_feat_dim'],
            n_heads=config['n_heads'],
            num_layers=config['num_structure_module_layers'],
            dropout=config['dropout']
        )
    
    def forward(self, seq, msa=None):  # msa는 인터페이스 호환을 위해 있지만 사용하지 않음
        # 시퀀스 임베딩
        seq_emb = self.seq_embedding(seq)  # [batch_size, seq_len, d_model]
        
        # 시퀀스 길이 확인
        seq_len = seq_emb.shape[1]
        
        # 위치 인코딩 추가
        if seq_len <= self.pos_embedding.shape[1]:
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            # 위치 인코딩 확장
            existing_pos = self.pos_embedding.squeeze(0)  # [max_len, d_model]
            extended_pos = torch.zeros(seq_len, self.config['msa_feat_dim'], device=seq.device)
            extended_pos[:existing_pos.shape[0], :] = existing_pos
            # 나머지 부분은 마지막 위치 인코딩 복제
            if existing_pos.shape[0] > 0:
                extended_pos[existing_pos.shape[0]:, :] = existing_pos[-1, :]
            pos_emb = extended_pos.unsqueeze(0)
            
        seq_emb = seq_emb + pos_emb
        
        # 트랜스포머 레이어 통과
        x = seq_emb
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 구조 모듈로 좌표 예측
        xyz_pred = self.structure_module(x.unsqueeze(1)).squeeze(1)
        
        return xyz_pred

# 거리 행렬 계산 함수
def calculate_distance_matrix(X, Y, epsilon=1e-4):
    return (torch.square(X[:, None] - Y[None, :]) + epsilon).sum(-1).sqrt()

# dRMAE 손실 함수
def dRMAE(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)
    
    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()] = False
    
    rmsd = torch.abs(pred_dm[mask] - gt_dm[mask])
    
    return rmsd.mean() / Z

# SVD 기반 정렬 및 MAE 손실 함수
def align_svd_mae(input, target, Z=10):
    """
    SVD 기반 Procrustes 정렬을 사용하여 입력을 타겟에 정렬하고 MAE 손실을 계산합니다.
    """
    assert input.shape == target.shape, "입력과 타겟의 형태가 같아야 합니다"
    
    # 마스크 적용
    mask = ~torch.isnan(target.sum(-1))
    
    input = input[mask]
    target = target[mask]
    
    # 중심점 계산
    centroid_input = input.mean(dim=0, keepdim=True)
    centroid_target = target.mean(dim=0, keepdim=True)
    
    # 중심 정규화
    input_centered = input - centroid_input.detach()
    target_centered = target - centroid_target
    
    # 공분산 행렬 계산
    cov_matrix = input_centered.T @ target_centered
    
    # SVD를 사용한 최적 회전 찾기
    U, S, Vt = torch.svd(cov_matrix)
    
    # 회전 행렬 계산
    R = Vt @ U.T
    
    # 회전이 적절한지 확인 (det(R) = 1, 반사 없음)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T
    
    # 입력 회전
    aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()
    
    return torch.abs(aligned_input - target).mean() / Z


def rigid_transform_3D_safe(A, B):
    """
    자동 미분과 호환되는 안전한 방식으로 최적의 회전 및 변환 행렬을 계산합니다.
    
    Args:
        A: (N, 3) 형태의 텐서 - 예측 좌표
        B: (N, 3) 형태의 텐서 - 실제 좌표
        
    Returns:
        R: 회전 행렬 (3, 3)
        t: 변환 벡터 (3)
    """
    # NaN 값 제거
    mask = ~torch.isnan(B).any(dim=1)
    A_filtered = A[mask]
    B_filtered = B[mask]
    
    if len(A_filtered) < 3:  # 유효한 변환을 위해 최소 3개의 점 필요
        # 항등 회전 및 0 변환 반환
        return torch.eye(3, device=A.device), torch.zeros(3, device=A.device)
    
    # 중심점 계산
    centroid_A = torch.mean(A_filtered, dim=0)
    centroid_B = torch.mean(B_filtered, dim=0)
    
    # 중심 정규화
    A_centered = A_filtered - centroid_A
    B_centered = B_filtered - centroid_B
    
    # 공분산 행렬 계산
    H = A_centered.T @ B_centered
    
    # SVD - 자동 미분과 호환되는 방식으로
    try:
        # 표준 SVD
        U, S, V = torch.linalg.svd(H, full_matrices=False)
    except Exception:
        # 문제가 있으면 더 안정적인 방법 시도
        print("표준 SVD 실패, 안정화된 방법 시도")
        # 작은 값 추가로 안정화
        H_stable = H + torch.eye(3, device=H.device) * 1e-6
        U, S, V = torch.linalg.svd(H_stable, full_matrices=False)
    
    # 회전 행렬 계산
    R = V.T @ U.T
    
    # 결정자 확인 (반사가 아닌 회전인지)
    det = torch.det(R)
    if det < 0:
        # 클론을 생성하여 인플레이스 연산 방지
        V_adjusted = V.clone()
        V_adjusted[-1] = V_adjusted[-1] * -1
        R = V_adjusted.T @ U.T
    
    # 변환 계산
    t = centroid_B - R @ centroid_A
    
    return R, t

def fape_loss_safe(pred, target, clamp_distance=10.0, eps=1e-8):
    """
    자동 미분과 호환되는 안전한 방식으로 Frame Aligned Point Error (FAPE) 손실을 계산합니다.
    
    Args:
        pred: (N, 3) 형태의 예측 좌표 텐서
        target: (N, 3) 형태의 실제 좌표 텐서
        clamp_distance: 최대 거리 (이상치 처리용)
        eps: 0으로 나누기 방지를 위한 작은 값
        
    Returns:
        FAPE 손실 값
    """
    # target의 NaN 값 처리
    mask = ~torch.isnan(target).any(dim=1)
    
    # 유효한 점이 없으면 0 손실 반환
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # 최적의 회전 및 변환 계산
    try:
        R, t = rigid_transform_3D_safe(pred, target)
        
        # 예측값을 target에 정렬
        pred_aligned = (R @ pred.T).T + t
        
        # 유효한 위치에서만 오차 계산
        error = torch.sqrt(torch.sum((pred_aligned[mask] - target[mask])**2, dim=1) + eps)
        
        # 큰 거리는 제한하여 이상치 영향 감소
        error = torch.clamp(error, max=clamp_distance)
        
        return error.mean()
    except Exception as e:
        print(f"FAPE 손실 계산 중 오류: {str(e)}")
        # 오류 발생 시 대체 손실 반환
        return torch.mean(torch.abs(pred[mask] - target[mask]))

def combined_rna_loss_safe(pred_coords, target_coords, sequence, 
                         fape_weight=1.0,
                         rmsd_weight=0.5,
                         violation_weight=0.2,
                         structural_violation_epoch=50,
                         current_epoch=0):
    """
    RNA 구조 예측을 위한 안전한 조합 손실 함수
    
    Args:
        pred_coords: (N, 3) 형태의 예측 좌표 텐서
        target_coords: (N, 3) 형태의 실제 좌표 텐서
        sequence: RNA 시퀀스 인덱스
        fape_weight: FAPE 손실 가중치
        rmsd_weight: RMSD 손실 가중치
        violation_weight: 구조적 위반 가중치
        structural_violation_epoch: 구조적 위반 적용 시작 에폭
        current_epoch: 현재 훈련 에폭
        
    Returns:
        조합 손실 값
    """
    # FAPE 손실 - 안전한 버전 사용
    fape = fape_loss_safe(pred_coords, target_coords)
    
    # RMSD 손실
    try:
        rmsd = dRMAE(pred_coords, pred_coords, target_coords, target_coords)
    except Exception:
        # 오류 발생 시 대체 손실
        mask = ~torch.isnan(target_coords).any(dim=1)
        if mask.sum() > 0:
            rmsd = torch.mean(torch.abs(pred_coords[mask] - target_coords[mask]))
        else:
            rmsd = torch.tensor(0.0, device=pred_coords.device, requires_grad=True)
    
    # 총 손실 초기화
    total_loss = fape_weight * fape + rmsd_weight * rmsd
    
    # 특정 에폭 이후 구조적 위반 추가
    if current_epoch >= structural_violation_epoch:
        try:
            # 기존 structural_violations_loss 함수 호출
            violations = structural_violations_loss(pred_coords, sequence, violation_weight)
            total_loss += violations
        except Exception as e:
            print(f"구조적 위반 계산 중 오류: {str(e)}")
            # 오류 발생 시 구조적 위반 무시
            pass
    
    return total_loss


# 4. Curriculum Learning Implementation

class CurriculumSampler:
    """
    Implements curriculum learning by gradually increasing the complexity
    of training examples based on RNA length and other features.
    """
    def __init__(self, dataset, max_epochs, min_len=10, max_len=500):
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.min_len = min_len
        self.max_len = max_len
        self.epoch = 0
        
        # Calculate sequence lengths for all examples
        self.lengths = []
        for idx in dataset.indices:
            seq_len = len(dataset.data['sequence'][idx])
            self.lengths.append(seq_len)
        
        self.lengths = np.array(self.lengths)
        self.indices = np.arange(len(dataset.indices))
        
        # Initial sort by length
        self.sorted_indices = self.indices[np.argsort(self.lengths)]
        
    def update_epoch(self, epoch):
        """Update current epoch"""
        self.epoch = epoch
        
    def get_indices(self):
        """
        Returns indices for the current epoch based on curriculum.
        Gradually includes longer sequences as training progresses.
        """
        # Calculate progress ratio (0 to 1)
        progress = min(1.0, self.epoch / (self.max_epochs * 0.8))
        
        # Calculate max length for current epoch
        current_max_len = self.min_len + progress * (self.max_len - self.min_len)
        
        # Get indices of sequences up to current max length
        mask = self.lengths <= current_max_len
        curriculum_indices = self.indices[mask]
        
        # If too few sequences, include at least 20% of the data
        if len(curriculum_indices) < len(self.indices) * 0.2:
            min_count = int(len(self.indices) * 0.2)
            curriculum_indices = self.sorted_indices[:min_count]
        
        # Shuffle the selected indices
        np.random.shuffle(curriculum_indices)
        
        return curriculum_indices


# 훈련 함수
def train_model(model, train_loader, val_loader, config, model_path=None):
    # 저장된 모델 불러오기 (있는 경우)
    if model_path and os.path.exists(model_path):
        print(f"모델 불러오기: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path))
            print("모델 로드 성공!")
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
            print("새로운 모델로 훈련을 시작합니다.")
    else:
        print("사전 훈련된 모델이 없습니다. 새로운 모델로 훈련을 시작합니다.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    scaler = GradScaler()
    
    # 코사인 스케줄러
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(config['epochs'] - config['cos_epoch']) * len(train_loader) // config['batch_size']
    )
    
    best_val_loss = float('inf')
    
    # 이슈 추적을 위한 디버그 모드 (필요한 경우 활성화)
    # torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(config['epochs']):
        model.train()
        tbar = tqdm(train_loader)
        total_loss = 0
        
        for idx, batch in enumerate(tbar):
            sequence = batch['sequence'].cuda()
            msa = batch['msa'].cuda()
            gt_xyz = batch['xyz'].cuda()
            
            pred_xyz = model(sequence, msa)
            
            # 배치 차원 제거
            pred_xyz = pred_xyz.squeeze(0)
            gt_xyz = gt_xyz.squeeze(0)
            
            # 새로운 안전한 손실 함수 사용
            if epoch < config.get('structural_violation_epoch', 50):
                # 초기에는 기존 손실 함수로 훈련 (안정성을 위해)
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz) + align_svd_mae(pred_xyz, gt_xyz)
            else:
                # 일정 에폭 이후 개선된 손실 함수 사용
                try:
                    loss = combined_rna_loss_safe(
                        pred_xyz, 
                        gt_xyz, 
                        sequence,
                        fape_weight=1.0,
                        rmsd_weight=0.5,
                        violation_weight=0.2,
                        structural_violation_epoch=config.get('structural_violation_epoch', 50),
                        current_epoch=epoch
                    )
                except Exception as e:
                    print(f"손실 계산 중 오류: {str(e)}")
                    # 오류 발생 시 기존 손실 함수 사용
                    loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz) + align_svd_mae(pred_xyz, gt_xyz)
            
            (loss / config['batch_size']).backward()
            
            if (idx + 1) % config['batch_size'] == 0 or idx + 1 == len(tbar):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
                
                if (epoch + 1) > config['cos_epoch']:
                    schedule.step()
            
            total_loss += loss.item()
            
            tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss / (idx + 1)}")
        
        # 검증
        model.eval()
        val_loss = 0
        val_preds = []
        
        tbar = tqdm(val_loader)
        for idx, batch in enumerate(tbar):
            sequence = batch['sequence'].cuda()
            msa = batch['msa'].cuda()
            gt_xyz = batch['xyz'].cuda()
            
            with torch.no_grad():
                pred_xyz = model(sequence, msa)
                
                # 배치 차원 제거
                pred_xyz = pred_xyz.squeeze(0)
                gt_xyz = gt_xyz.squeeze(0)
                
                # 검증에는 기존 손실 함수 사용 (안정성을 위해)
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz)
                
                # 에폭 50 이후에는 추가 검증 메트릭 출력
                if epoch >= config.get('structural_violation_epoch', 50):
                    try:
                        fape = fape_loss_safe(pred_xyz, gt_xyz)
                        print(f"배치 {idx} FAPE: {fape.item():.4f}")
                    except:
                        pass
            
            val_loss += loss.item()
            val_preds.append([gt_xyz.cpu().numpy(), pred_xyz.cpu().numpy()])
        
        val_loss = val_loss / len(tbar)
        print(f"Validation loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = val_preds
            torch.save(model.state_dict(), 'RNA_MSA_Folding_best.pt')
    
    # 최종 모델 저장
    torch.save(model.state_dict(), 'RNA_MSA_Folding_final.pt')
    
    return best_val_loss, best_preds


config.update({
    "structural_violation_epoch": 25,  # 25번째 에폭부터 구조 위반 손실 적용
    "mixed_precision": "bf16",         # 혼합 정밀도 사용
    "gradient_accumulation_steps": 4,  # 효과적인 배치 크기 증가를 위한 그래디언트 누적
})


try:
    print("데이터 로드 중...")
    data, train_index, test_index = load_data(config)
    
    print(f"MSA 데이터 로드 완료: {len(data['msa'])} 항목")
    print(f"첫 번째 MSA 시퀀스 샘플: {data['msa'][0][:2] if data['msa'] and len(data['msa'][0]) > 1 else 'N/A'}")
    
    # 데이터셋 및 데이터로더 생성
    print("데이터셋 생성 중...")
    train_dataset = RNA3D_MSA_Dataset(train_index, data, config)
    val_dataset = RNA3D_MSA_Dataset(test_index, data, config)
    
    # 데이터셋 샘플 확인
    sample = train_dataset[0]
    print(f"데이터셋 샘플 형태:")
    print(f"  시퀀스: {sample['sequence'].shape}")
    print(f"  MSA: {sample['msa'].shape}")
    print(f"  XYZ: {sample['xyz'].shape}")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

except Exception as e:
    print(f"오류 발생: {str(e)}")
    import traceback
    traceback.print_exc()


def visualize_predictions(pred_data):
    """
    예측된 구조와 실제 구조를 시각화합니다.
    """
    import plotly.graph_objects as go
    
    gt_xyz, pred_xyz = pred_data
    
    # NaN 값 필터링
    mask = ~np.isnan(gt_xyz).any(axis=1)
    gt_xyz_filtered = gt_xyz[mask]
    pred_xyz_filtered = pred_xyz[mask]
    
    # 실제 구조
    fig1 = go.Figure(data=[go.Scatter3d(
        x=gt_xyz_filtered[:, 0],
        y=gt_xyz_filtered[:, 1],
        z=gt_xyz_filtered[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=np.arange(len(gt_xyz_filtered)),
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Ground Truth'
    )])
    
    fig1.update_layout(
        title="Ground Truth RNA 3D Structure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    
    # 예측 구조
    fig2 = go.Figure(data=[go.Scatter3d(
        x=pred_xyz_filtered[:, 0],
        y=pred_xyz_filtered[:, 1],
        z=pred_xyz_filtered[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=np.arange(len(pred_xyz_filtered)),
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Prediction'
    )])
    
    fig2.update_layout(
        title="Predicted RNA 3D Structure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    
    fig1.show()
    fig2.show()


# MSA 모델 생성 및 학습
print("MSA 모델 학습 시작...")
model_msa = RNA_MSA_Folding(config).cuda()

# 사전 훈련된 모델 경로
pretrained_model_path = "/kaggle/input/rna-msa-folding/RNA_MSA_Folding_model.pt"
best_val_loss_msa, best_preds_msa = train_model(model_msa, train_loader, val_loader, config, pretrained_model_path)

# 단일 시퀀스 모델 생성 및 학습 (옵션)
print("단일 시퀀스 모델 학습 시작...")
model_single = RNA_Single_Folding(config).cuda()
best_val_loss_single, best_preds_single = train_model(model_single, train_loader, val_loader, config)

# 결과 비교
print(f"MSA 모델 최종 검증 손실: {best_val_loss_msa}")
print(f"단일 시퀀스 모델 최종 검증 손실: {best_val_loss_single}")


# 예측 시각화
if best_preds:
    print("예측 결과 시각화 중...")
    visualize_predictions(best_preds[0])
else:
    print("시각화를 위한 예측 결과가 없습니다.")

