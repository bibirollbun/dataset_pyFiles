!pip install /kaggle/input/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn
import torch.nn.functional as F

# 재현성을 위한 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 설정
config = {
    "max_len": 384,
    "msa_max_sequences": 32,
    "msa_feat_dim": 128,
    "num_self_attn_layers": 4,
    "num_cross_attn_layers": 4,
    "num_structure_module_layers": 8,
    "n_heads": 8,
    "dropout": 0.25,
    "num_ensemble": 5,  # 5개의 앙상블 예측
}


# BioPython을 사용한 FASTA/MSA 파일 로드 함수
def load_msa_with_biopython(fasta_file, convert_to_rna=False):
    """
    BioPython의 SeqIO를 사용하여 FASTA 형식의 MSA 파일에서 시퀀스를 로드합니다.
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
def load_a3m_fallback(fasta_file, convert_to_rna=False):
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
        # 기본값으로 1x1 배열 반환
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
    
    # 원-핫 인코딩을 위한 사전 (A, C, G, U, - 갭, 기타 문자)
    vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 4, '-': 5, '.': 5}  # Add 'T' as token 4
    
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
            nt_upper = nt.upper()
            
            if nt_upper in vocab:
                msa_encoded[i, j] = vocab[nt_upper]
            else:
                # 모르는 문자는 갭으로 처리
                msa_encoded[i, j] = vocab['-']
    
    return msa_encoded

# 테스트 데이터셋 클래스
class RNA3D_MSA_TestDataset(Dataset):
    def __init__(self, test_data, msa_dir, config):
        self.test_data = test_data
        self.msa_dir = msa_dir
        self.config = config
        # Line ~166: Change the tokens dictionary 
        self.tokens = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 4}  # Add 'T'
    
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        target_id = self.test_data.loc[idx, 'target_id']
        sequence = self.test_data.loc[idx, 'sequence']
        
        # 시퀀스를 숫자로 변환
        seq_encoded = [self.tokens.get(nt, 0) for nt in sequence]
        seq_encoded = torch.tensor(np.array(seq_encoded))
        
        # MSA 파일 경로
        msa_file = os.path.join(self.msa_dir, f"{target_id}.MSA.fasta")
        
        # MSA 데이터 로드
        if os.path.exists(msa_file):
            try:
                msa_sequences = load_msa_with_biopython(msa_file)
                msa_encoded = process_msa(msa_sequences, self.config['msa_max_sequences'])
                msa_encoded = torch.tensor(msa_encoded)
            except Exception as e:
                print(f"MSA 로드 실패: {target_id}, 오류: {str(e)}")
                # 기본값: 원본 시퀀스만 포함
                msa_encoded = torch.tensor([[self.tokens.get(nt, 0) for nt in sequence]])
        else:
            # MSA 파일이 없는 경우, 원래 시퀀스만 포함
            msa_encoded = torch.tensor([[self.tokens.get(nt, 0) for nt in sequence]])
        
        return {
            'target_id': target_id,
            'sequence': seq_encoded,
            'msa': msa_encoded,
            'raw_sequence': sequence
        }

# 모델 정의 (학습 코드와 동일)
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
        
        # 임베딩 레이어
        self.seq_embedding = nn.Embedding(6, config['msa_feat_dim'])  # Change from 5 to 6 to include 'T'

        
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

# 추론 함수 - 앙상블 예측 생성
def predict_ensemble(model, batch, num_ensemble=5):
    """
    드롭아웃을 활성화한 상태에서 여러 번 예측하여 앙상블 결과 생성
    """
    model.train()  # 드롭아웃 활성화
    
    sequence = batch['sequence'].cuda()
    msa = batch['msa'].cuda()
    
    predictions = []
    
    for _ in range(num_ensemble):
        with torch.no_grad():
            pred_xyz = model(sequence, msa).squeeze(0).cpu().numpy()
            predictions.append(pred_xyz)
    
    return predictions



set_seed(42)

# 테스트 데이터 로드
test_data = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
print(f"테스트 데이터 로드 완료: {len(test_data)} 샘플")

# MSA 디렉토리
msa_dir = "/kaggle/input/stanford-rna-3d-folding/MSA"

# 데이터셋 생성
test_dataset = RNA3D_MSA_TestDataset(test_data, msa_dir, config)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 로드
model = RNA_MSA_Folding(config).cuda()
model.load_state_dict(torch.load('/kaggle/input/20250315-rna-msa-folding/RNA_MSA_Folding_best.pt'))
print("모델 로드 완료")

# 예측 실행
all_predictions = []
target_ids = []
raw_sequences = []

for batch in tqdm(test_loader, desc="예측 중"):
    target_id = batch['target_id'][0]
    raw_sequence = batch['raw_sequence'][0]
    
    # 앙상블 예측
    predictions = predict_ensemble(model, batch, num_ensemble=config['num_ensemble'])
    
    all_predictions.append(predictions)
    target_ids.append(target_id)
    raw_sequences.append(raw_sequence)

# 제출 파일 형식으로 변환
data = []

for i in range(len(target_ids)):
    for j in range(len(raw_sequences[i])):
        # ID, resname, resid
        row = [f"{target_ids[i]}_{j+1}", raw_sequences[i][j], j+1]
        
        # 5개 앙상블 모델의 x, y, z 좌표 추가
        for k in range(config['num_ensemble']):
            for coord_idx in range(3):  # x, y, z
                row.append(all_predictions[i][k][j][coord_idx])
        
        data.append(row)

# 열 이름 생성
columns = ['ID', 'resname', 'resid']
for i in range(1, config['num_ensemble'] + 1):
    columns.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

# 데이터프레임 생성 및 저장
submission = pd.DataFrame(data, columns=columns)
submission.to_csv('submission.csv', index=False)

print(f"제출 파일 생성 완료: {len(submission)} 행")
print(submission.head())




