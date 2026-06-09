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


train_df.head()


train_df['phase'].value_counts()


train_df.head(26)


import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

# 모자이크 플롯으로 시각화
plt.figure(figsize=(15, 10))
mosaic(train_df, ['orientation', 'gesture'], title='Mosaic Plot of Orientation and Gesture')
plt.show()


# 1. 교차표 생성
crosstab_df = pd.crosstab(train_df['orientation'], train_df['gesture'])
print("교차표:\n", crosstab_df)

# 2. 히트맵으로 시각화
plt.figure(figsize=(15, 10))
sns.heatmap(crosstab_df, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Correlation between Orientation and Gesture', fontsize=16)
plt.xlabel('Gesture', fontsize=12)
plt.ylabel('Orientation', fontsize=12)
plt.show()



test_df.head()


test_df['sequence_id'].value_counts()


train_dem_df = pd.read_csv(train_demographics_filePath)

train_dem_df.head()


datasets = {
    "Train Data": train_df,
    "Train Demographics": train_dem_df,
    "Test Data": test_df,
    "Test Demographics": test_dem_df,
}

# Print shapes
for name, df in datasets.items():
    num_rows, num_cols = df.shape
    print(f"{name}:")
    print(f"  Number of Rows: {num_rows}")
    print(f"  Number of Columns: {num_cols}\n")


# Count duplicate rows in train_df
train_duplicates = train_df.duplicated().sum()

# Count duplicate rows in test_df
test_duplicates = test_df.duplicated().sum()

# Count duplicate rows in train_dem_df (optional)
train_dem_duplicates = train_dem_df.duplicated().sum()

# Count duplicate rows in test_dem_df (optional)
test_dem_duplicates = test_dem_df.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_df: {train_duplicates}")
print(f"Number of duplicate rows in test_df: {test_duplicates}")
print(f"Number of duplicate rows in train_dem_df: {train_dem_duplicates}")
print(f"Number of duplicate rows in test_dem_df: {test_dem_duplicates}")


train_seq_df = train_df[['sequence_id', 'sequence_counter']]
test_seq_df = test_df[['sequence_id', 'sequence_counter']]

# Count duplicate rows in train_df
train_duplicates = train_seq_df.duplicated().sum()

# Count duplicate rows in test_df
test_duplicates = test_seq_df.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_df: {train_duplicates}")
print(f"Number of duplicate rows in test_df: {test_duplicates}")


filtered_df = train_df[train_df['sequence_counter'] == 0].copy()
filtered_df


from sklearn.preprocessing import LabelEncoder
import torch

le = LabelEncoder()

filtered_df['orientation'] = le.fit_transform(filtered_df['orientation'])
filtered_df


IMU_df = filtered_df[['orientation', 'acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z',]]
IMU_df


correlation_matrix = IMU_df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    cbar=True
)

plt.title('Correlation Heatmap')
plt.show()


orientation_df = train_df[['orientation', 'acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']]
orientation_df


orientation_df['orientation'] = le.fit_transform(orientation_df['orientation'])


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler



# 피처와 타겟 분리
features = orientation_df[['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']]
target = orientation_df['orientation']

# 데이터 정규화
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 시퀀스 데이터 생성 함수
def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

TIME_STEPS = 25 # 시퀀스 길이
X, y = create_sequences(features_scaled, target.values, TIME_STEPS)

# 타겟 레이블 원-핫 인코딩
num_classes = len(np.unique(y))
y_one_hot = to_categorical(y, num_classes=num_classes)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)


# LSTM 모델 구축
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")




