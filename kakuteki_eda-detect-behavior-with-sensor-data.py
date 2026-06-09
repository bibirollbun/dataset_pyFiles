# 必要なライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ファイルパスの定義
train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
train_demo_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
test_demo_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"

# データの読み込み
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_demo = pd.read_csv(train_demo_path)
test_demo = pd.read_csv(test_demo_path)

# データの基本情報の表示
print("Train Data Info:")
print(train.info())
print("\nTest Data Info:")
print(test.info())
print("\nTrain Demographics Info:")
print(train_demo.info())
print("\nTest Demographics Info:")
print(test_demo.info())

# データの先頭5行の表示
print("\nTrain Data Head:")
print(train.head())
print("\nTest Data Head:")
print(test.head())
print("\nTrain Demographics Head:")
print(train_demo.head())
print("\nTest Demographics Head:")
print(test_demo.head())

# 欠損値の確認
print("\nMissing Values in Train Data:")
print(train.isnull().sum())
print("\nMissing Values in Test Data:")
print(test.isnull().sum())

# ジェスチャーの分布
gesture_counts = train['gesture'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=gesture_counts.index, y=gesture_counts.values)
plt.title('Gesture Distribution in Training Data')
plt.xlabel('Gesture')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# センサーデータの統計量
sensor_columns = [col for col in train.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
print("\nSensor Data Statistics:")
print(train[sensor_columns].describe())


# センサー列の抽出
acc_columns = [col for col in train.columns if col.startswith('acc_')]
thm_columns = [col for col in train.columns if col.startswith('thm_')]
tof_columns = [col for col in train.columns if col.startswith('tof_')]

gesture_name = 'Text on phone'
gesture_data = train[train['gesture'] == gesture_name]

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data[gesture_data['sequence_id'] == sequence_id]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in ['acc_x', 'acc_y', 'acc_z']:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.show()

# 温度センサーデータのプロット
plt.figure(figsize=(12, 6))
for sensor in thm_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[sensor], label=sensor)
plt.title(f'Thermopile Sensor Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# ToFセンサーデータの平均値を計算しプロット
tof_avg = sequence_data[tof_columns].replace(-1, np.nan).mean(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(sequence_data['sequence_counter'], tof_avg)
plt.title(f'Average ToF Sensor Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Average ToF Value')
plt.show()



# センサー列の抽出
acc_columns = [col for col in train.columns if col.startswith('acc_')]
thm_columns = [col for col in train.columns if col.startswith('thm_')]
tof_columns = [col for col in train.columns if col.startswith('tof_')]

gesture_name = 'Cheek - pinch skin'
gesture_data = train[train['gesture'] == gesture_name]

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data[gesture_data['sequence_id'] == sequence_id]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in ['acc_x', 'acc_y', 'acc_z']:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.show()

# 温度センサーデータのプロット
plt.figure(figsize=(12, 6))
for sensor in thm_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[sensor], label=sensor)
plt.title(f'Thermopile Sensor Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# ToFセンサーデータの平均値を計算しプロット
tof_avg = sequence_data[tof_columns].replace(-1, np.nan).mean(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(sequence_data['sequence_counter'], tof_avg)
plt.title(f'Average ToF Sensor Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Average ToF Value')
plt.show()



# センサー列の抽出
acc_columns = [col for col in train.columns if col.startswith('acc_')]
thm_columns = [col for col in train.columns if col.startswith('thm_')]
tof_columns = [col for col in train.columns if col.startswith('tof_')]

gesture_name = 'Forehead - pull hairline'
gesture_data = train[train['gesture'] == gesture_name]

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data[gesture_data['sequence_id'] == sequence_id]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in ['acc_x', 'acc_y', 'acc_z']:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.show()

# 温度センサーデータのプロット
plt.figure(figsize=(12, 6))
for sensor in thm_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[sensor], label=sensor)
plt.title(f'Thermopile Sensor Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# ToFセンサーデータの平均値を計算しプロット
tof_avg = sequence_data[tof_columns].replace(-1, np.nan).mean(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(sequence_data['sequence_counter'], tof_avg)
plt.title(f'Average ToF Sensor Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Average ToF Value')
plt.show()


