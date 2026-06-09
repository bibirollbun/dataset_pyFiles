# 必要なライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import polars as pl


# def of file paths
data_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/"

# load datasets
train = pl.read_csv(os.path.join(data_path, "train.csv"))
test = pl.read_csv(os.path.join(data_path, "test.csv"))


# センサー列の抽出
acc_columns = [col for col in train.columns if col.startswith('acc_')]
thm_columns = [col for col in train.columns if col.startswith('thm_')]
tof_columns = [col for col in train.columns if col.startswith('tof_')]

gesture_name = 'Text on phone'
gesture_data = train.filter(pl.col('gesture') == gesture_name)

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data.filter(pl.col('sequence_id') == sequence_id).to_pandas()

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
gesture_data = train.filter(pl.col('gesture') == gesture_name)

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data.filter(pl.col('sequence_id') == sequence_id).to_pandas()

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



# ▼ Step 1: ターゲット動作の定義
target_gestures = [
    'Above ear - pull hair',
    'Cheek - pinch skin',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Neck - pinch skin',
    'Neck - scratch'
    ]

# ▼ Step 2: ターゲット vs 非ターゲット の二値ラベルを作成
train = train.with_columns(
    pl.when(pl.col("gesture").is_in(target_gestures))
    .then(pl.lit("target"))
    .otherwise(pl.lit("non_target"))
    .alias("is_target")
)
train["is_target"]



# ジェスチャーの分布
gesture_counts = train.group_by('is_target').count().sort('count', descending=True).to_pandas()
plt.figure(figsize=(12,6))
sns.barplot(x=gesture_counts['is_target'], y=gesture_counts['count'])
plt.title('Gesture Distribution in Training Data')
plt.xlabel('Gesture')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()



# センサー列の抽出
acc_columns = [col for col in train.columns if col.startswith('acc_')]
thm_columns = [col for col in train.columns if col.startswith('thm_')]
tof_columns = [col for col in train.columns if col.startswith('tof_')]

gesture_name = 'target'
gesture_data = train.filter(pl.col('is_target') == gesture_name)

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data.filter(pl.col('sequence_id') == sequence_id).to_pandas()

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

gesture_name = 'non_target'
gesture_data = train.filter(pl.col('is_target') == gesture_name)

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data.filter(pl.col('sequence_id') == sequence_id).to_pandas()

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



train_filtered = train.filter(pl.col("sequence_counter") < 30)
train_filtered.head(1)


train_filtered = train_filtered.with_columns([
        # delta_acc = √(Δx² + Δy² + Δz²)
        ((pl.col("acc_x").diff().pow(2) +
          pl.col("acc_y").diff().pow(2) +
          pl.col("acc_z").diff().pow(2)).sqrt()).alias("delta_acc"),
    ])
train_filtered['delta_acc'].head(3)


train_filtered = train_filtered.with_columns([
        # AVM = √(x² + y² + z²)
        ((pl.col("acc_x").pow(2) +
          pl.col("acc_y").pow(2) +
          pl.col("acc_z").pow(2)).sqrt()).alias("avm")
    ])
train_filtered['avm'].head(3)


# ---- 姿勢変化 GM ----
gm_df = (
    train_filtered.group_by("sequence_id")
    .agg([
        pl.mean("acc_x").alias("mean_x"),
        pl.mean("acc_y").alias("mean_y"),
        pl.mean("acc_z").alias("mean_z"),
    ])
    .with_columns([
        ((pl.col("mean_x").pow(2) + pl.col("mean_y").pow(2) + pl.col("mean_z").pow(2)).sqrt()).alias("gm")
    ])
    .select(["sequence_id", "gm"])
)
gm_df.head(3)


train_filtered = train_filtered.with_columns([
        # 閾値フラグ
        (pl.col("delta_acc") > 0.2).cast(pl.Int8).alias("motion_gt_0.2g"),
        (pl.col("avm") > 1.5).cast(pl.Int8).alias("high_intensity_flag")
    ])

# ---- 閾値ベース特徴量集約 ----
agg_exprs = [
    pl.sum("motion_gt_0.2g").alias("motion_count_gt_0.2g"),
    pl.sum("high_intensity_flag").alias("high_intensity_count"),
    pl.mean("high_intensity_flag").alias("high_intensity_ratio"),
]

thresh_features = train_filtered.group_by("sequence_id").agg(agg_exprs)
thresh_features.head(1)


# ---- 結合 ----
train_filtered = train_filtered.join(gm_df, on="sequence_id", how="left")
train_filtered = train_filtered.join(thresh_features, on= "sequence_id", how="left")
train_filtered[['motion_gt_0.2g', 'high_intensity_count', 'high_intensity_ratio']].head(3)


# センサー列の抽出
acc_columns = ['delta_acc', 'avm', 'gm', 'motion_count_gt_0.2g', 'high_intensity_count', 'high_intensity_ratio']

gesture_name = 'Text on phone'
exp_train_filtered = train_filtered.to_pandas()
gesture_data = exp_train_filtered[exp_train_filtered['gesture'] == gesture_name]

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data[gesture_data['sequence_id'] == sequence_id]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in acc_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Values')
plt.legend()
plt.show()


# センサー列の抽出
acc_columns = ['delta_acc', 'avm', 'gm', 'motion_count_gt_0.2g', 'high_intensity_count', 'high_intensity_ratio']

gesture_name = 'Cheek - pinch skin'
exp_train_filtered = train_filtered.to_pandas()
gesture_data = exp_train_filtered[exp_train_filtered['gesture'] == gesture_name]

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data[gesture_data['sequence_id'] == sequence_id]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in acc_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Values')
plt.legend()
plt.show()


# ▼ Step 1: ターゲット動作の定義
target_gestures = [
    'Above ear - pull hair',
    'Cheek - pinch skin',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Neck - pinch skin',
    'Neck - scratch'
    ]

# ▼ Step 2: ターゲット vs 非ターゲット の二値ラベルを作成
train_filtered = train_filtered.with_columns(
    pl.when(pl.col("gesture").is_in(target_gestures))
    .then(pl.lit("target"))
    .otherwise(pl.lit("non_target"))
    .alias("is_target")
)
train_filtered["is_target"]



# センサー列の抽出
acc_columns = ['delta_acc', 'avm', 'gm', 'motion_count_gt_0.2g', 'high_intensity_count', 'high_intensity_ratio']

gesture_name = 'target'
exp_train_filtered = train_filtered.to_pandas()
gesture_data = exp_train_filtered[exp_train_filtered['is_target'] == gesture_name]

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data[gesture_data['sequence_id'] == sequence_id]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in acc_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Values')
plt.legend()
plt.show()


# センサー列の抽出
acc_columns = ['delta_acc', 'avm', 'gm', 'motion_count_gt_0.2g', 'high_intensity_count', 'high_intensity_ratio']

gesture_name = 'target'
exp_train_filtered = train_filtered.to_pandas()
sequence_data = exp_train_filtered[exp_train_filtered['is_target'] == gesture_name]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in acc_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('values')
plt.legend()
plt.show()


# センサー列の抽出
acc_columns = ['delta_acc', 'avm', 'gm', 'motion_count_gt_0.2g', 'high_intensity_count', 'high_intensity_ratio']

gesture_name = 'non_target'
exp_train_filtered = train_filtered.to_pandas()
gesture_data = exp_train_filtered[exp_train_filtered['is_target'] == gesture_name]

# 特定のsequence_idを選択
sequence_id = gesture_data['sequence_id'].unique()[0]
sequence_data = gesture_data[gesture_data['sequence_id'] == sequence_id]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in acc_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.show()


# センサー列の抽出
acc_columns = ['delta_acc', 'avm', 'gm', 'motion_count_gt_0.2g', 'high_intensity_count', 'high_intensity_ratio']

gesture_name = 'non_target'
exp_train_filtered = train_filtered.to_pandas()
sequence_data = exp_train_filtered[exp_train_filtered['is_target'] == gesture_name]

# 加速度データのプロット
plt.figure(figsize=(12, 6))
for axis in acc_columns:
    plt.plot(sequence_data['sequence_counter'], sequence_data[axis], label=axis)
plt.title(f'Accelerometer Data for Gesture: {gesture_name} (Sequence ID: {sequence_id})')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.show()




