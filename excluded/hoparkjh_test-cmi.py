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


train_filePath = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
train_demographics_filePath = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv'
test_filePath = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
test_demographics_filePath = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'


train_df = pd.read_csv(train_filePath)
train_dem_df = pd.read_csv(train_demographics_filePath)
test_df = pd.read_csv(test_filePath)
test_dem_df = pd.read_csv(test_demographics_filePath)


train_df.head(10)


train_df.columns[train_df.isnull().any()].tolist()


train_df['sequence_type'].value_counts()


feature_columns = list(train_df.columns)
del feature_columns[16:]
del feature_columns[0:9]

feature_columns[0]


IMU_df = train_df[feature_columns]
IMU_df


IMU_df.isnull().sum()


IMU_df[IMU_df.isnull().any(axis=1)]


null_rotations = train_df[train_df[['rot_w', 'rot_x', 'rot_y', 'rot_z']].isnull().any(axis=1)]

# 위에서 찾은 행들의 'sequence_id' 열에서 중복되지 않는 값들만 추출합니다.
unique_sequence_ids = null_rotations['sequence_id'].unique()

# NumPy 배열을 파이썬 리스트로 변환하여 출력합니다.
unique_sequence_ids_list = unique_sequence_ids.tolist()
print(unique_sequence_ids_list)


# rot가 null인 seq_id
null_rot = null_rotations.groupby('sequence_id').first()[['orientation']]
null_seq_len = null_rotations.groupby('sequence_id').last()[['sequence_counter']]
null_rot = pd.merge(null_rot, null_seq_len, on='sequence_id', how='right')
null_rot


train_df[['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']].isnull().sum()


def input_rot(seq_id, last_seq_id, seq_len, seq_c):
    # 원본(last_seq_id) 데이터와 대상(seq_id) 데이터를 미리 필터링
    source_rows = train_df[train_df['sequence_id'] == last_seq_id].sort_values('sequence_counter')
    target_rows_to_fill = train_df[train_df['sequence_id'] == seq_id].sort_values('sequence_counter')

    # 복사할 데이터의 인덱스를 찾습니다. (source_df의 0부터 seq_len만큼)
    source_indices = source_rows.head(seq_len).index

    target_indices = target_rows_to_fill.head(seq_len).index
    # 길이가 같지 않은 경우 오류를 방지합니다.
    while len(source_indices) != len(target_indices):
        
        print(f"오류: 복사할 행의 수({len(source_indices)})와 채워넣을 행의 수({len(target_indices)})가 다릅니다.")
        return
    train_df.loc[target_indices, ['rot_w', 'rot_x', 'rot_y', 'rot_z']] = \
        train_df.loc[source_indices, ['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
    
    print(f"'{seq_id}'에 '{last_seq_id}'의 rot 값을 {len(source_indices)}개 복사했습니다.")


for row in null_rot.itertuples():
    seq_id, orientation, seq_len = row
    filtered_df = train_df[(train_df['orientation'] == orientation)]

    max_counter = 0
    # last seq_counter
    last_counter_df = filtered_df.groupby('sequence_id')['sequence_counter'].max()

    for col in last_counter_df.items():
        last_seq_id, seq_c = col
        if seq_id == last_seq_id:
            continue
        elif seq_c == seq_len:
            input_rot(seq_id, last_seq_id, seq_len + 1, seq_c)
            max_counter = -1
            break
        elif (seq_c > max_counter) & (seq_c < seq_len + 1):
            max_counter = seq_c
            temp_id = last_seq_id
            
    if max_counter != -1:
        print(max_counter)
        input_rot(seq_id, temp_id, seq_len + 1, max_counter)


train_df[['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']].isnull().sum()


IMU_df = train_df[feature_columns]
IMU_df


train_df[train_df[['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']].isnull().any(axis=1)]


train_df[train_df['sequence_id'] == 'SEQ_001160'][['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']]


grouped_train_df = train_df.groupby('sequence_id')


import torch

# 비어있는 리스트를 준비합니다. 이 리스트에 각 시퀀스의 텐서를 담을 것입니다.
sequences_as_tensors = []

# grouped는 (sequence_id, DataFrame) 형태의 튜플을 반환합니다.
for sequence_id, group_df in grouped_train_df:
    # 1. 특성 컬럼만 선택합니다.
    features_df = group_df[feature_columns]

    # 2. DataFrame을 NumPy 배열로 변환합니다.
    features_numpy = features_df.values

    # 3. NumPy 배열을 PyTorch 텐서로 변환합니다.
    # .float()는 데이터 타입을 실수형으로 맞춰줍니다. 딥러닝에서 주로 사용됩니다.
    sequence_tensor = torch.from_numpy(features_numpy).float()
    
    # 4. 변환된 텐서를 리스트에 추가합니다.
    sequences_as_tensors.append(sequence_tensor)

# 이제 sequences_as_tensors 리스트가 예시의 sequences 리스트와 동일한 역할을 합니다.
print(f"총 시퀀스 개수: {len(sequences_as_tensors)}")
print(f"첫 번째 시퀀스 텐서의 크기: {sequences_as_tensors[0].shape}")
print(f"두 번째 시퀀스 텐서의 크기: {sequences_as_tensors[1].shape}")


def pad_to_max_len(sequences, max_len, padding_value=0):
    padded_tensors = []
    for seq in sequences:
        # 현재 시퀀스 길이
        current_len = seq.size(0)
        
        # 패딩할 길이 계산
        padding_len = max_len - current_len
        
        # 패딩 텐서 생성
        # torch.zeros를 사용해 패딩할 텐서를 만들고, padding_value로 채움
        # seq.size(1)은 컬럼 수(특성 수)
        padding_tensor = torch.full((padding_len, seq.size(1)), padding_value, dtype=seq.dtype)
        
        # 원래 시퀀스와 패딩 텐서를 합침
        padded_seq = torch.cat([seq, padding_tensor], dim=0)
        padded_tensors.append(padded_seq)
        
    # 패딩된 텐서들을 하나의 배치 텐서로 묶음
    return torch.stack(padded_tensors, dim=0)



# 이전에 작성했던 pad_to_max_len 함수가 있다고 가정합니다.
# from your_utils import pad_to_max_len

# 모든 시퀀스를 길이 700으로 패딩
padded_X = pad_to_max_len(sequences_as_tensors, max_len=700, padding_value=0)

print(f"패딩 완료된 텐서의 크기: {padded_X.shape}")
# 결과: torch.Size([8151, 700, 7])
# (총 시퀀스 개수, 시퀀스 길이, 특성 수)


target_column = 'sequence_type'

# y 텐서 생성 함수
def create_y_tensor(df, target_col):
    # 'sequence_id'별로 그룹화
    grouped = df.groupby('sequence_id')
    
    y_list = []
    
    # 1. 타겟 레이블을 숫자로 인코딩 (이진 분류)
    # 'Target' -> 1, 'Non-target' -> 0으로 매핑
    target_to_id = {'Non-Target': 0, 'Target': 1}
    print(f"타겟 인코딩 맵: {target_to_id}")

    # 2. 각 시퀀스(그룹)의 타겟 레이블 추출
    for _, group_df in grouped:
        # 시퀀스 내 모든 행의 타겟 레이블이 동일하다고 가정하고 첫 번째 값을 사용
        sequence_type = group_df.iloc[0][target_col]
        y_label = target_to_id[sequence_type]
        y_list.append(y_label)

    # 파이토치 텐서로 변환
    # 'long' 타입은 분류 레이블에 사용됨
    return torch.tensor(y_list, dtype=torch.long)


# y 텐서 생성 및 확인
y_tensor = create_y_tensor(train_df, target_column)
print(f"y 텐서의 크기: {y_tensor.shape}")
print(f"y 텐서: {y_tensor}")


import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어 정의
        # batch_first=True는 입력 텐서의 형태가 (배치 크기, 시퀀스 길이, 특성 수)임을 의미합니다.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # fully connected 레이어 정의
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # h0와 c0는 초기 hidden state와 cell state입니다.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 레이어를 통과시킵니다.
        # out: 모든 time_step의 output, _ : 최종 hidden state와 cell state
        out, _ = self.lstm(x, (h0, c0))  
        
        # 마지막 time_step의 output만 사용합니다.
        # out[:, -1, :]은 (배치 크기, hidden_size) 형태가 됩니다.
        out = self.fc(out[:, -1, :])
        return out


# 모델 초기화
# input_size = 332 (패딩된 텐서의 마지막 차원 크기)
input_size = 332
hidden_size = 128  # 원하는 은닉 상태 크기로 설정
num_layers = 2     # 원하는 레이어 수로 설정
num_classes = 2   # 11가지 동작 분류

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)


# sequence_counter 개수의 
grouped_train_df[['sequence_counter']].max().describe()


seq_len = grouped_train_df[['sequence_counter']].max() + 1
seq_len


# 95%
seq_len.quantile(0.95)

