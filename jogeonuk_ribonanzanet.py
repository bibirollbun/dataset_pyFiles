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


import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle


config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "structural_violation_epoch": 50,
    "balance_weight": False,
}


test_data=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


from torch.utils.data import Dataset, DataLoader

class RNADataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)




        return {'sequence':sequence}


test_dataset=RNADataset(test_data)
test_dataset[0]


! pip install einops


import torch
import pandas as pd

# -----------------------------
# 1. 모델 정의
# -----------------------------
class RNA3DModel(torch.nn.Module):
    def __init__(self, input_dim=100):
        super(RNA3DModel, self).__init__()
        self.fc = torch.nn.Linear(input_dim, 3)  # 입력 차원: input_dim, 출력: 3D 좌표

    def forward(self, x):
        return self.fc(x)

# -----------------------------
# 2. 토큰 매핑과 전처리 함수
# -----------------------------
tokens = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
max_len = 100  # 고정된 입력 길이

def preprocess_sequence(sequence, tokens, max_len):
    sequence_tokens = [tokens.get(nt, -1) for nt in sequence[:max_len]]  # 자르기
    if len(sequence_tokens) < max_len:
        sequence_tokens += [-1] * (max_len - len(sequence_tokens))  # 패딩 추가
    return torch.tensor(sequence_tokens, dtype=torch.float)

# -----------------------------
# 3. 모델 로드
# -----------------------------
model_path = '/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D-final.pt'
model = RNA3DModel(input_dim=max_len)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)  # 일부 키 불일치 허용
model.eval()

# -----------------------------
# 4. 테스트 데이터 불러오기 및 전처리
# -----------------------------
test_data = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')

# 시퀀스를 텐서로 변환
test_sequences = test_data['sequence'].apply(lambda seq: preprocess_sequence(seq, tokens, max_len))
test_sequences = torch.stack(list(test_sequences))  # (N, 100)

# -----------------------------
# 5. 예측 수행
# -----------------------------
with torch.no_grad():
    predictions = model(test_sequences)  # (N, 3)

# -----------------------------
# 6. 결과 저장
# -----------------------------
# 예측 결과를 DataFrame으로 변환
df = pd.DataFrame(predictions.numpy(), columns=["x", "y", "z"])

# ID 열이 존재하면 병합
if 'ID' in test_data.columns:
    df.insert(0, 'ID', test_data['ID'])

# CSV 파일로 저장
output_path = '/kaggle/working/submission.csv'
df.to_csv(output_path, index=False)

# 저장 완료 메시지
print(f"✅ 예측 결과가 저장되었습니다: {output_path}")



import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml
import torch.nn as nn

# config 설정
config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml",  # 경로 수정 가능
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "structural_violation_epoch": 50,
    "balance_weight": False,
}

class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}  # ACGU to numeric tokens
        self.invalid_token = -1  # Default token for invalid characters

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure that 'sequence' contains a string
        sequence = self.data.loc[idx, 'sequence']
        
        # Convert each character in the sequence to the corresponding numeric token
        sequence_tokens = []
        for nt in sequence:
            if nt in self.tokens:
                sequence_tokens.append(self.tokens[nt])
            else:
                # Replace invalid characters with the default invalid token
                sequence_tokens.append(self.invalid_token)
                print(f"Warning: Invalid nucleotide character '{nt}' at index {idx} in sequence: {sequence}")
        
        # Convert the sequence to a tensor
        sequence = np.array(sequence_tokens)
        sequence = torch.tensor(sequence)
        return {'sequence': sequence}

# 데이터 로딩
train_data = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.v2.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv")
test_data = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

train_dataset = RNADataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# Config 클래스 정의
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

# finetuned_RibonanzaNet 모델 정의
class finetuned_RibonanzaNet(nn.Module):
    def __init__(self, config, pretrained=False):
        super(finetuned_RibonanzaNet, self).__init__()
        
        # 예측기 하드코딩
        self.xyz_predictor = nn.Linear(256, 3)  # 예시: 256차원 입력, 3차원 출력 (XYZ 좌표 예측)

    def forward(self, src):
        # 입력 src로부터 특징 추출
        sequence_features = self.extract_sequence_features(src)
        pairwise_features = self.extract_pairwise_features(src)

        # 예측 (xyz_predictor는 이미 정의된 예측기를 사용)
        xyz = self.xyz_predictor(sequence_features)
        return xyz

    def extract_sequence_features(self, src):
        # sequence_features 추출 로직
        return torch.randn(src.size(0), 256)  # 예시: 랜덤 피쳐 (실제 로직은 데이터에 맞게 작성 필요)

    def extract_pairwise_features(self, src):
        # pairwise_features 추출 로직
        return torch.randn(src.size(0), 256)  # 예시: 랜덤 피쳐 (실제 로직은 데이터에 맞게 작성 필요)

# 모델 초기화 (GPU가 있는지 확인 후, 적절히 지정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델을 CPU 또는 GPU에 맞게 로드
model = finetuned_RibonanzaNet(load_config_from_yaml("/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml"), pretrained=False).to(device)

# 훈련 함수
def train(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            sequences = batch['sequence'].to(device)  # 데이터도 적절히 디바이스로 이동

            # 예시: 예측 및 손실 계산
            xyz_pred = model(sequences)
            print(f"Epoch {epoch}, Batch prediction shape: {xyz_pred.shape}")

# 테스트 데이터셋 예측
test_dataset = RNADataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# 테스트 데이터셋 예측 함수
def test(model, dataloader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)  # 데이터도 적절히 디바이스로 이동
            xyz_pred = model(sequences)
            results.append(xyz_pred.cpu().numpy())  # 결과는 CPU로 반환하여 처리
    return np.concatenate(results, axis=0)

# 모델 훈련
train(model, train_dataloader, epochs=config['epochs'])

# 예측 실행
test_results = test(model, test_dataloader)

# 예측 결과를 CSV로 저장 (예: 3D 좌표 예측 결과)
test_results_df = pd.DataFrame(test_results, columns=['x', 'y', 'z'])
test_results_df.to_csv("/kaggle/working/test_predictions.csv", index=False)

print("3D 좌표 예측 완료, 결과 저장됨.")



# Check the columns of the train_data DataFrame
print(train_data.columns)

# Ensure the column 'sequence' exists in train_data
# If the column name is different, adjust the code accordingly


