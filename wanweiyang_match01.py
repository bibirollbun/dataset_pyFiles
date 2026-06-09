import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 检查GPU配置
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"第二GPU: {torch.cuda.get_device_name(1)}")

# 定义设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ----------------------
# 1. 数据加载与详细探索
# ----------------------
def load_and_explore_data():
    """加载数据并进行详细探索，为后续处理提供依据"""
    print("加载数据...")
    try:
        train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
        test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
        train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
        test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("尝试使用备选路径...")
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        train_demo = pd.read_csv('train_demographics.csv')
        test_demo = pd.read_csv('test_demographics.csv')
    
    # 详细数据探索
    print("\n===== 数据探索 =====")
    print(f"训练集形状: {train.shape}")
    print(f"测试集形状: {test.shape}")
    
    print("\n训练集列名及数据类型（前10列）:")
    print(train.dtypes.head(10))
    print("\n测试集列名及数据类型（前10列）:")
    print(test.dtypes.head(10))
    
    # 检查特征差异
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    train_only_cols = train_cols - test_cols
    test_only_cols = test_cols - train_cols
    print(f"\n训练集特有列: {len(train_only_cols)}个 - {sorted(train_only_cols)[:5]}...")
    print(f"测试集特有列: {len(test_only_cols)}个 - {sorted(test_only_cols)[:5]}...")
    
    # 检查关键列是否存在
    required_columns = ['sequence_id', 'row_id']
    for col in required_columns:
        if col not in train.columns:
            print(f"警告: 训练集缺少必要列 '{col}'")
        if col not in test.columns:
            print(f"警告: 测试集缺少必要列 '{col}'")
    
    # 检查是否有共同的键用于合并人口统计学数据
    merge_key = 'subject' if 'subject' in train.columns and 'subject' in train_demo.columns else None
    if merge_key:
        print(f"\n将使用 '{merge_key}' 合并主数据和人口统计学数据")
    else:
        print("\n警告: 未找到合适的合并键，将不合并人口统计学数据")
    
    return train, test, train_demo, test_demo

# ----------------------
# 2. 数据预处理（核心优化：裁剪TOF特征+类别样本平衡+特征增强）
# ----------------------
def preprocess_data(train, test, train_demo, test_demo):
    """数据预处理：灵活处理不同结构的数据 + 类别平衡 + 特征增强 + TOF特征裁剪"""
    print("\n===== 数据预处理 =====")
    
    # 尝试合并人口统计学数据
    merge_key = 'subject' if 'subject' in train.columns and 'subject' in train_demo.columns else None
    if merge_key:
        print(f"使用 '{merge_key}' 合并数据...")
        train = train.merge(train_demo, on=merge_key, how='left')
        test = test.merge(test_demo, on=merge_key, how='left')
    else:
        print("无法合并人口统计学数据，跳过此步骤")
    
    # ----------------------
    # 核心优化1：裁剪TOF冗余特征（每5列保留1列，减少70%+维度）
    # ----------------------
    tof_cols_train = [col for col in train.columns if col.startswith('tof_')]
    tof_cols_test = [col for col in test.columns if col.startswith('tof_')]
    print(f"原始TOF特征数量（训练集）: {len(tof_cols_train)}")
    
    # 每5列保留1列（按索引步长筛选，兼顾物理含义连续性）
    keep_tof_cols = tof_cols_train[::5] if len(tof_cols_train) > 0 else []
    # 确保测试集与训练集保留的TOF列一致
    keep_tof_cols_test = [col for col in keep_tof_cols if col in tof_cols_test]
    
    # 删除冗余TOF列
    if len(tof_cols_train) > 0:
        drop_tof_train = [col for col in tof_cols_train if col not in keep_tof_cols]
        train = train.drop(columns=drop_tof_train)
    if len(tof_cols_test) > 0:
        drop_tof_test = [col for col in tof_cols_test if col not in keep_tof_cols_test]
        test = test.drop(columns=drop_tof_test)
    
    print(f"裁剪后TOF特征数量（训练集）: {len(keep_tof_cols)}")
    print(f"裁剪后TOF特征数量（测试集）: {len(keep_tof_cols_test)}")
    
    # 确保训练集和测试集有相同的特征列
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common_cols = train_cols.intersection(test_cols)
    train_only_cols = ['gesture', 'gesture_encoded', 'is_target']  # 新增is_target列
    train_only_cols = [col for col in train_only_cols if col in train_cols]
    print(f"仅在训练集中保留的列: {train_only_cols}")
    
    # 特征增强：新增传感器衍生特征（提升区分度）
    def add_sensor_derived_features(df):
        # 识别IMU传感器列（加速度、陀螺仪等）
        imu_cols = [col for col in df.columns if col.startswith(('acc_', 'gyro_', 'rot_', 'mag_'))]
        if len(imu_cols) >=3:
            # 计算加速度向量模长（物理意义：总加速度大小）
            if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
                df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
            # 计算陀螺仪向量模长（旋转强度）
            if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
                df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
            # 计算传感器数据一阶差分（变化率）- 限制数量，避免特征爆炸
            for col in imu_cols[:6]:  # 仅保留前6个IMU列的差分
                df[f'{col}_diff1'] = df.groupby('sequence_id')[col].diff().fillna(0)
        # 人口统计学衍生特征（如BMI）
        if all(col in df.columns for col in ['weight', 'height_cm']):
            df['bmi'] = df['weight'] / ((df['height_cm']/100)**2)
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        return df
    
    # 为训练集和测试集添加衍生特征
    train = add_sensor_derived_features(train)
    test = add_sensor_derived_features(test)
    print("已添加传感器衍生特征（模长、一阶差分）和人口统计学衍生特征（BMI）")
    
    # 重新对齐特征列（包含新增的衍生特征）
    train_features = list(common_cols) + train_only_cols
    test_features = list(common_cols)
    for col in ['sequence_id', 'row_id', 'subject', 'acc_magnitude', 'gyro_magnitude', 'bmi']:
        if col in train.columns and col not in train_features:
            train_features.append(col)
        if col in test.columns and col not in test_features:
            test_features.append(col)
    
    # 应用特征选择
    train = train[train_features]
    test = test[test_features]
    print(f"训练集特征列数量（含衍生特征）: {len(train_features)}")
    print(f"测试集特征列数量（含衍生特征）: {len(test_features)}")
    
    # 检查并处理缺失值（分类型优化填充策略）
    print(f"训练集缺失值比例: {train.isnull().mean().mean():.4f}")
    print(f"测试集缺失值比例: {test.isnull().mean().mean():.4f}")
    
    imu_cols = [col for col in train.columns if col.startswith(('acc_', 'rot_', 'gyro_', 'mag_')) or col.endswith('_diff1')]
    thm_cols = [col for col in train.columns if col.startswith(('thm_', 'temp_', 'thermal_'))]
    tof_cols = [col for col in train.columns if col.startswith(('tof_', 'distance_'))]
    demo_cols = [col for col in train.columns if col in ['age', 'height', 'height_cm', 'weight', 'bmi', 'shoulder_to_wrist', 'shoulder_to_wrist_cm', 'elbow_to_wrist', 'elbow_to_wrist_cm']]
    
    print(f"找到 {len(imu_cols)} 个IMU相关列（含衍生特征）")
    print(f"找到 {len(thm_cols)} 个热传感器列")
    print(f"找到 {len(tof_cols)} 个飞行时间传感器列")
    print(f"找到 {len(demo_cols)} 个人口统计学列（含衍生特征）")
    
    # 分类型填充缺失值（更贴合数据分布）
    if imu_cols:
        train[imu_cols] = train.groupby('sequence_id')[imu_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').fillna(x.median()))
        test[imu_cols] = test.groupby('sequence_id')[imu_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').fillna(x.median()))
    if thm_cols:
        train[thm_cols] = train.groupby('sequence_id')[thm_cols].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean()))
        test[thm_cols] = test.groupby('sequence_id')[thm_cols].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean()))
    if tof_cols:
        train[tof_cols] = train.groupby('sequence_id')[tof_cols].transform(
            lambda x: x.fillna(x.quantile(0.5)))  # 中位数填充，抗异常值
        test[tof_cols] = test.groupby('sequence_id')[tof_cols].transform(
            lambda x: x.fillna(x.quantile(0.5)))
    if demo_cols:
        train[demo_cols] = train[demo_cols].fillna(train[demo_cols].median())
        test[demo_cols] = test[demo_cols].fillna(test[demo_cols].median())
    
    # 编码目标变量
    if 'gesture' not in train.columns:
        raise ValueError("训练集必须包含'gesture'列作为目标变量")
    gesture_le = LabelEncoder()
    train['gesture_encoded'] = gesture_le.fit_transform(train['gesture'])
    
    # 定义目标类型（根据竞赛要求，这里假设需要确定哪些是目标手势）
    unique_gestures = train['gesture'].unique()
    target_gestures = unique_gestures[:len(unique_gestures)//2]
    train['is_target'] = train['gesture'].isin(target_gestures).astype(int)
    
    print(f"已编码 {len(gesture_le.classes_)} 种手势类别")
    print(f"目标手势: {len(target_gestures)}种, 非目标手势: {len(unique_gestures)-len(target_gestures)}种")
    
    # 分析类别分布
    class_distribution = train['gesture_encoded'].value_counts().sort_index()
    print("\n手势类别分布（原始）:")
    for cls, count in class_distribution.items():
        print(f"类别 {cls}: {count} 样本 ({count/len(train):.2%})")
    
    # 二元类别分布
    target_distribution = train['is_target'].value_counts().sort_index()
    print("\n二元目标分布（原始）:")
    for cls, count in target_distribution.items():
        print(f"{'目标' if cls == 1 else '非目标'}: {count} 样本 ({count/len(train):.2%})")
    
    # ----------------------
    # 类别样本平衡（序列级SMOTE过采样，避免单样本重复）
    # ----------------------
    print("\n===== 类别样本平衡（序列级SMOTE过采样） =====")
    max_count = class_distribution.max()
    seq_class_map = train.groupby('sequence_id')['gesture_encoded'].first().to_dict()
    seq_list = train['sequence_id'].unique()
    
    # 按类别分组序列
    seq_by_class = {cls: [] for cls in class_distribution.index}
    for seq_id in seq_list:
        cls = seq_class_map[seq_id]
        seq_by_class[cls].append(seq_id)
    
    # 序列级SMOTE过采样（生成"相似但不重复"的序列）
    def generate_smote_seq(base_seq_df, n_generate, seq_id_prefix):
        generated_seqs = []
        for i in range(n_generate):
            # 对传感器特征添加微小扰动（模拟真实数据变异）
            imu_feat_cols = [col for col in base_seq_df.columns if col.startswith(('acc_', 'gyro_', 'rot_', 'mag_')) or col.endswith('_diff1')]
            perturb_df = base_seq_df.copy()
            if len(imu_feat_cols) > 0:
                perturb_scale = 0.03  # 扰动强度（控制在3%以内，保证物理意义）
                perturb = np.random.normal(0, perturb_scale, size=(len(perturb_df), len(imu_feat_cols)))
                perturb_df[imu_feat_cols] += perturb * perturb_df[imu_feat_cols].std(axis=0).values
            # 生成新序列ID
            new_seq_id = f"{seq_id_prefix}_smote_{i}"
            perturb_df['sequence_id'] = new_seq_id
            generated_seqs.append(perturb_df)
        return pd.concat(generated_seqs, ignore_index=True) if generated_seqs else pd.DataFrame()
    
    # 对少数类生成新序列
    oversampled_train = []
    for cls in class_distribution.index:
        cls_seqs = seq_by_class[cls]
        current_seq_count = len(cls_seqs)
        
        # 若当前序列数小于最大序列数，生成新序列（限制生成量，避免数据膨胀）
        if current_seq_count < max_count:
            need_seq_count = min(max_count - current_seq_count, current_seq_count * 2)  # 最多生成2倍原始序列
            base_seq_ids = np.random.choice(cls_seqs, size=min(need_seq_count, current_seq_count), replace=True)
            generated_samples = []
            for base_seq_id in base_seq_ids:
                base_seq_df = train[train['sequence_id'] == base_seq_id].copy()
                gen_seq = generate_smote_seq(base_seq_df, n_generate=1, seq_id_prefix=base_seq_id)
                if not gen_seq.empty:
                    generated_samples.append(gen_seq)
            # 合并原始序列和生成序列
            cls_original = train[train['sequence_id'].isin(cls_seqs)]
            cls_generated = pd.concat(generated_samples, ignore_index=True) if generated_samples else pd.DataFrame()
            oversampled_train.append(pd.concat([cls_original, cls_generated], ignore_index=True))
        else:
            # 多数类直接保留
            oversampled_train.append(train[train['sequence_id'].isin(cls_seqs)])
    
    # 合并过采样后的数据
    train = pd.concat(oversampled_train, ignore_index=True)
    # 重新计算类别分布
    oversampled_dist = train['gesture_encoded'].value_counts().sort_index()
    print(f"过采样前训练集总样本数: {len(seq_list) * train.groupby('sequence_id').size().mean():.0f}")
    print(f"过采样后训练集总样本数: {len(train)}")
    print(f"过采样后类别分布:")
    for cls, count in oversampled_dist.items():
        print(f"类别 {cls}: {count} 样本 ({count/len(train):.2%})")
    
    return train, test, imu_cols, thm_cols, tof_cols, demo_cols, gesture_le, class_distribution, target_gestures

# ----------------------
# 3. 优化：时序统计特征提取函数（简化计算+分批处理）
# ----------------------
def extract_temporal_stats(group, feature_cols):
    """简化版时序统计特征提取，仅保留核心统计量，减少计算量"""
    stats = {}
    # 1. 仅保留核心统计量（删除自相关、高阶差分等复杂特征）
    stats['mean'] = group[feature_cols].mean()
    stats['std'] = group[feature_cols].std()
    stats['max'] = group[feature_cols].max()
    stats['min'] = group[feature_cols].min()
    stats['median'] = group[feature_cols].median()
    stats['ptp'] = group[feature_cols].max() - group[feature_cols].min()
    
    # 2. 一阶差分的基础统计（捕捉变化率）
    if len(group) > 1:
        diff1 = group[feature_cols].diff().fillna(0)
        stats['diff1_mean'] = diff1.mean()
        stats['diff1_std'] = diff1.std()
    
    # 拼接为DataFrame（每行对应一个序列的统计特征）
    stats_df = pd.DataFrame([stats])
    
    # 确保所有值都是数值类型，并填充任何NaN
    for col in stats_df.columns:
        stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce').fillna(0.0)
    
    # 重命名列（避免与原始特征冲突）
    stats_df.columns = [f'{col}_stat' for col in stats_df.columns]
    return stats_df

# 分批提取时序统计特征（避免内存溢出）
def extract_temporal_stats_batch(data, feature_cols, batch_size=1000):
    """分批处理时序统计特征提取，降低内存占用"""
    sequence_ids = data['sequence_id'].unique()
    stats_list = []
    # 按批次处理序列
    for i in range(0, len(sequence_ids), batch_size):
        batch_seq_ids = sequence_ids[i:i+batch_size]
        batch_data = data[data['sequence_id'].isin(batch_seq_ids)]
        # 按序列计算统计特征
        batch_stats = batch_data.groupby('sequence_id').apply(
            lambda x: extract_temporal_stats(x, feature_cols)
        ).reset_index(drop=True)
        stats_list.append(batch_stats)
        # 释放批次内存
        del batch_data
        gc.collect()
    
    return pd.concat(stats_list, ignore_index=True) if stats_list else pd.DataFrame()

# ----------------------
# 4. 优化：prepare_sequences函数（动态调整序列长度+内存优化）
# ----------------------
def prepare_sequences(train, test, max_seq_length=None):
    print("\n===== 准备序列数据（含时序统计特征） =====")
    
    # 1. 获取共同特征列
    if 'sequence_id' not in train.columns or 'sequence_id' not in test.columns:
        raise ValueError("数据必须包含'sequence_id'列以区分不同序列")
    
    train_exclude_cols = ['sequence_id', 'row_id', 'gesture', 'gesture_encoded', 'subject', 'is_target']
    test_exclude_cols = ['sequence_id', 'row_id', 'subject']
    train_exclude_cols = [col for col in train_exclude_cols if col in train.columns]
    test_exclude_cols = [col for col in test_exclude_cols if col in test.columns]
    
    train_feature_cols = [col for col in train.columns if col not in train_exclude_cols]
    test_feature_cols = [col for col in test.columns if col not in test_exclude_cols]
    common_feature_cols = list(set(train_feature_cols).intersection(set(test_feature_cols)))
    common_feature_cols.sort()
    print(f"共同特征列: {len(common_feature_cols)}")
    
    if len(common_feature_cols) == 0:
        raise ValueError("未找到共同的特征列，无法继续")
    
    # 2. 分批提取序列级统计特征（优化内存）
    print("提取时序统计特征...")
    # 训练集统计特征
    try:
        train_stats = extract_temporal_stats_batch(train, common_feature_cols, batch_size=500)
        # 确保索引正确对应
        unique_train_ids = train['sequence_id'].unique()
        if len(train_stats) != len(unique_train_ids):
            print(f"警告：训练集统计特征数量({len(train_stats)})与序列数量({len(unique_train_ids)})不匹配，重新对齐")
            train_stats = train_stats.iloc[:len(unique_train_ids)]
        train_stats.index = unique_train_ids
    except Exception as e:
        print(f"训练集统计特征提取出错: {e}")
        # 创建默认的统计特征
        train_stats = pd.DataFrame(0.0, index=train['sequence_id'].unique(), 
                                  columns=[f'feat_{i}_stat' for i in range(10)])
    
    # 测试集统计特征
    try:
        test_stats = extract_temporal_stats_batch(test, common_feature_cols, batch_size=500)
        # 确保索引正确对应
        unique_test_ids = test['sequence_id'].unique()
        if len(test_stats) != len(unique_test_ids):
            print(f"警告：测试集统计特征数量({len(test_stats)})与序列数量({len(unique_test_ids)})不匹配，重新对齐")
            test_stats = test_stats.iloc[:len(unique_test_ids)]
        test_stats.index = unique_test_ids
    except Exception as e:
        print(f"测试集统计特征提取出错: {e}")
        # 创建默认的统计特征
        test_stats = pd.DataFrame(0.0, index=test['sequence_id'].unique(), 
                                 columns=[f'feat_{i}_stat' for i in range(10)])
    
    # 3. 处理训练序列（拼接统计特征）
    train_sequences = []
    train_gestures = []
    train_is_target = []  # 新增：二元目标标签
    train_ids = []
    
    print("处理训练序列...")
    for seq_id, group in tqdm(train.groupby('sequence_id'), desc="处理训练序列"):
        try:
            # 原始时序特征
            seq_data = group[common_feature_cols].copy()
            # 填充/转换数值类型
            for col in seq_data.columns:
                try:
                    seq_data[col] = pd.to_numeric(seq_data[col], errors='coerce').fillna(seq_data[col].median())
                except:
                    le = LabelEncoder()
                    seq_data[col] = le.fit_transform(seq_data[col].astype(str))
            
            # 获取当前序列的统计特征
            if seq_id in train_stats.index:
                seq_stat = train_stats.loc[seq_id].values.reshape(1, -1)  # (1, n_stats)
            else:
                # 如果找不到对应的统计特征，使用0填充
                seq_stat = np.zeros((1, train_stats.shape[1]))
                print(f"警告：序列 {seq_id} 未找到统计特征，使用默认值")
            
            # 确保统计特征是有效的数值数组
            seq_stat = np.nan_to_num(seq_stat.astype(np.float32))
            
            # 广播到每个时间步
            seq_stat_broadcast = np.repeat(seq_stat, len(seq_data), axis=0)  # (seq_len, n_stats)
            
            # 确保原始特征是有效的数值数组
            seq_data_values = np.nan_to_num(seq_data.values.astype(np.float32))
            
            # 拼接特征
            seq_values = np.hstack([seq_data_values, seq_stat_broadcast])
            
            # 填充/截断到固定长度
            if len(seq_values) < max_seq_length:
                pad_length = max_seq_length - len(seq_values)
                seq_values = np.pad(seq_values, ((0, pad_length), (0, 0)), mode='edge')
            else:
                seq_values = seq_values[:max_seq_length]
                
            train_sequences.append(seq_values)
            train_gestures.append(group['gesture_encoded'].iloc[0])
            train_is_target.append(group['is_target'].iloc[0])  # 保存二元标签
            train_ids.append(seq_id)
        except Exception as e:
            print(f"处理训练序列 {seq_id} 时出错: {e}")
            # 创建一个默认序列避免中断
            default_seq = np.zeros((max_seq_length, len(common_feature_cols) + train_stats.shape[1]), dtype=np.float32)
            train_sequences.append(default_seq)
            train_gestures.append(0)
            train_is_target.append(0)  # 默认非目标
            train_ids.append(seq_id)
    
    # 4. 处理测试序列
    test_sequences = []
    test_ids = []
    
    print("处理测试序列...")
    for seq_id, group in tqdm(test.groupby('sequence_id'), desc="处理测试序列"):
        try:
            seq_data = group[common_feature_cols].copy()
            for col in seq_data.columns:
                try:
                    seq_data[col] = pd.to_numeric(seq_data[col], errors='coerce').fillna(seq_data[col].median())
                except:
                    le = LabelEncoder()
                    seq_data[col] = le.fit_transform(seq_data[col].astype(str))
            
            # 获取当前序列的统计特征
            if seq_id in test_stats.index:
                seq_stat = test_stats.loc[seq_id].values.reshape(1, -1)  # (1, n_stats)
            else:
                seq_stat = np.zeros((1, test_stats.shape[1]))
                print(f"警告：测试序列 {seq_id} 未找到统计特征，使用默认值")
            
            # 确保统计特征是有效的数值数组
            seq_stat = np.nan_to_num(seq_stat.astype(np.float32))
            
            # 广播到每个时间步
            seq_stat_broadcast = np.repeat(seq_stat, len(seq_data), axis=0)  # (seq_len, n_stats)
            
            # 确保原始特征是有效的数值数组
            seq_data_values = np.nan_to_num(seq_data.values.astype(np.float32))
            
            # 拼接特征
            seq_values = np.hstack([seq_data_values, seq_stat_broadcast])
            
            # 填充/截断
            if len(seq_values) < max_seq_length:
                pad_length = max_seq_length - len(seq_values)
                seq_values = np.pad(seq_values, ((0, pad_length), (0, 0)), mode='edge')
            else:
                seq_values = seq_values[:max_seq_length]
                
            test_sequences.append(seq_values)
            test_ids.append(seq_id)
        except Exception as e:
            print(f"处理测试序列 {seq_id} 时出错: {e}")
            # 创建一个默认序列避免中断
            default_seq = np.zeros((max_seq_length, len(common_feature_cols) + test_stats.shape[1]), dtype=np.float32)
            test_sequences.append(default_seq)
            test_ids.append(seq_id)
    
    # 5. 后续处理
    train_sequences = np.array(train_sequences, dtype=np.float32)
    train_gestures = np.array(train_gestures)
    train_is_target = np.array(train_is_target)  # 转换为numpy数组
    test_sequences = np.array(test_sequences, dtype=np.float32)
    
    print(f"训练序列形状: {train_sequences.shape}")
    print(f"测试序列形状: {test_sequences.shape}")
    
    # 特征标准化
    if len(train_sequences) > 0 and train_sequences.ndim == 3 and train_sequences.shape[2] > 0:
        scaler = StandardScaler()
        n_samples, n_timesteps, n_features = train_sequences.shape
        train_reshaped = train_sequences.reshape(n_samples * n_timesteps, n_features)
        train_reshaped = scaler.fit_transform(train_reshaped)
        train_sequences = train_reshaped.reshape(n_samples, n_timesteps, n_features)
        
        if len(test_sequences) > 0 and test_sequences.ndim == 3 and test_sequences.shape[2] > 0:
            n_samples_test, n_timesteps_test, n_features_test = test_sequences.shape
            test_reshaped = test_sequences.reshape(n_samples_test * n_timesteps_test, n_features_test)
            test_reshaped = scaler.transform(test_reshaped)
            test_sequences = test_reshaped.reshape(n_samples_test, n_timesteps_test, n_features_test)
    
    # 清理内存
    del train_stats, test_stats
    gc.collect()
    
    n_features = train_sequences.shape[2] if (len(train_sequences.shape) > 2 and train_sequences.shape[2] > 0) else 0
    print(f"最终特征数量（原始+统计）: {n_features}")
    
    return (train_sequences, train_gestures, train_is_target, train_ids,
            test_sequences, test_ids, n_features)

# ----------------------
# 5. 修改：SensorSequenceDataset（加入针对少数类的增强）
# ----------------------
class SensorSequenceDataset(Dataset):
    def __init__(self, sequences, gestures=None, is_target=None, is_train=False, aug_prob=0.5, class_counts=None):
        self.sequences = sequences
        self.gestures = gestures
        self.is_target = is_target  # 新增：二元目标标签
        self.is_train = is_train  # 标记是否为训练集（仅训练时增强）
        self.aug_prob = aug_prob  # 基础增强概率
        self.class_counts = class_counts  # 类别分布，用于调整少数类的增强概率
        
        # 计算每个类别的增强权重（少数类增强概率更高）
        if class_counts is not None:
            max_count = class_counts.max()
            self.aug_weights = np.sqrt(max_count / class_counts)  # 少数类权重更高
            self.aug_weights = self.aug_weights / self.aug_weights.max()  # 归一化到[0,1]
        else:
            self.aug_weights = None
    
    def __len__(self):
        return len(self.sequences)
    
    def _augment_sequence(self, seq, aug_strength=1.0):
        """时序数据增强逻辑，支持调整增强强度"""
        seq = seq.copy()
        n_timesteps, n_features = seq.shape
        
        # 1. 高斯噪声注入（概率随增强强度增加）
        if np.random.random() < self.aug_prob * aug_strength:
            noise_scale = 0.05 * aug_strength  # 噪声强度随增强强度增加
            noise = np.random.normal(0, noise_scale * seq.std(axis=0), seq.shape)
            seq += noise
        
        # 2. 时间轴拉伸/压缩（概率随增强强度增加）
        if np.random.random() < self.aug_prob * 0.8 * aug_strength:
            scale_range = 0.2 * aug_strength  # 缩放范围随增强强度增加
            scale = np.random.uniform(1 - scale_range, 1 + scale_range)
            new_timesteps = int(n_timesteps * scale)
            # 线性插值调整长度
            seq_stretched = np.zeros((new_timesteps, n_features))
            for i in range(n_features):
                seq_stretched[:, i] = np.interp(
                    np.linspace(0, 1, new_timesteps),
                    np.linspace(0, 1, n_timesteps),
                    seq[:, i]
                )
            # 恢复原长度（截断/填充）
            if new_timesteps > n_timesteps:
                seq = seq_stretched[:n_timesteps]
            else:
                seq = np.pad(seq_stretched, ((0, n_timesteps - new_timesteps), (0, 0)), mode='edge')
        
        return seq.astype(np.float32)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 训练时应用增强，少数类增强强度更高
        if self.is_train and self.gestures is not None:
            gesture = self.gestures[idx]
            # 根据类别调整增强强度（少数类增强更强）
            if self.aug_weights is not None:
                aug_strength = 1.0 + self.aug_weights[gesture]  # 基础1.0 + 权重
            else:
                aug_strength = 1.0
            
            sequence = self._augment_sequence(sequence, aug_strength)
        
        sequence = torch.FloatTensor(sequence)
        
        if self.gestures is not None and self.is_target is not None:
            gesture = torch.LongTensor([self.gestures[idx]])[0]
            is_target = torch.LongTensor([self.is_target[idx]])[0]
            return sequence, gesture, is_target
        elif self.gestures is not None:
            gesture = torch.LongTensor([self.gestures[idx]])[0]
            return sequence, gesture
        else:
            return sequence

# ----------------------
# 6. 替换：CNN+BiLSTM+Self-Attention模型（轻量化）
# ----------------------
class CNNBiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_gestures, dropout=0.3):
        super(CNNBiLSTMAttentionModel, self).__init__()
        
        # 1. 1D CNN层（提取局部特征）
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),  # BatchNorm加速训练
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 双向LSTM层（输入维度=CNN输出维度）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Self-Attention层（关注关键时间步）
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 输入：BiLSTM输出（2*hidden_dim）
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # 对时间步加权
        )
        
        # 4. 全连接层 - 手势分类和二元分类共享特征
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. 手势分类头
        self.fc_gesture = nn.Linear(hidden_dim, num_gestures)
        
        # 6. 二元分类头（目标vs非目标）
        self.fc_target = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # 1. CNN前向（需转置：(batch, input_dim, seq_len) → 符合Conv1d输入格式）
        x_cnn = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x_cnn = self.cnn(x_cnn)      # (batch, hidden_dim, seq_len)
        x_cnn = x_cnn.permute(0, 2, 1)  # 转回：(batch, seq_len, hidden_dim)
        
        # 2. BiLSTM前向
        lstm_out, _ = self.lstm(x_cnn)  # (batch, seq_len, 2*hidden_dim)
        
        # 3. Self-Attention：对每个时间步加权
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = attn_weights.repeat(1, 1, lstm_out.size(-1))  # (batch, seq_len, 2*hidden_dim)
        attn_out = torch.mul(lstm_out, attn_weights).sum(dim=1)  # (batch, 2*hidden_dim) → 加权求和
        
        # 4. 共享特征层
        shared_features = self.shared_fc(attn_out)  # (batch, hidden_dim)
        
        # 5. 分类输出
        gesture_out = self.fc_gesture(shared_features)  # 手势分类
        target_out = self.fc_target(shared_features)    # 二元分类（目标vs非目标）
        
        return gesture_out, target_out

# ----------------------
# 7. 训练和验证函数（新增二元分类评估）
# ----------------------
def train_epoch(model, train_loader, gesture_criterion, target_criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_gesture_correct = 0
    total_target_correct = 0
    total_samples = 0
    
    for sequences, gestures, is_target in tqdm(train_loader, desc="训练"):
        sequences = sequences.to(device)
        gestures = gestures.to(device)
        is_target = is_target.to(device)
        
        optimizer.zero_grad()
        gesture_out, target_out = model(sequences)
        
        # 计算两个损失并加权求和
        loss_gesture = gesture_criterion(gesture_out, gestures)
        loss_target = target_criterion(target_out, is_target)
        loss = loss_gesture + loss_target  # 同等权重
        
        loss.backward()
        # 梯度裁剪（阈值5.0，防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * sequences.size(0)
        
        # 手势分类准确率
        _, predicted_gesture = torch.max(gesture_out.data, 1)
        total_gesture_correct += (predicted_gesture == gestures).sum().item()
        
        # 二元分类准确率
        _, predicted_target = torch.max(target_out.data, 1)
        total_target_correct += (predicted_target == is_target).sum().item()
        
        total_samples += sequences.size(0)
    
    avg_loss = total_loss / total_samples
    gesture_acc = total_gesture_correct / total_samples
    target_acc = total_target_correct / total_samples
    
    return avg_loss, gesture_acc, target_acc

def validate(model, val_loader, gesture_criterion, target_criterion, device, target_gestures=None, gesture_le=None):
    model.eval()
    total_loss = 0
    total_gesture_correct = 0
    total_target_correct = 0
    total_samples = 0
    
    all_gesture_preds = []
    all_gesture_labels = []
    all_target_preds = []
    all_target_labels = []
    
    with torch.no_grad():
        for sequences, gestures, is_target in tqdm(val_loader, desc="验证"):
            sequences = sequences.to(device)
            gestures = gestures.to(device)
            is_target = is_target.to(device)
            
            gesture_out, target_out = model(sequences)
            
            # 计算两个损失并加权求和
            loss_gesture = gesture_criterion(gesture_out, gestures)
            loss_target = target_criterion(target_out, is_target)
            loss = loss_gesture + loss_target  # 同等权重
            
            total_loss += loss.item() * sequences.size(0)
            
            # 手势分类准确率
            _, predicted_gesture = torch.max(gesture_out.data, 1)
            total_gesture_correct += (predicted_gesture == gestures).sum().item()
            
            # 二元分类准确率
            _, predicted_target = torch.max(target_out.data, 1)
            total_target_correct += (predicted_target == is_target).sum().item()
            
            total_samples += sequences.size(0)
            
            all_gesture_preds.extend(predicted_gesture.cpu().numpy())
            all_gesture_labels.extend(gestures.cpu().numpy())
            all_target_preds.extend(predicted_target.cpu().numpy())
            all_target_labels.extend(is_target.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    gesture_acc = total_gesture_correct / total_samples
    target_acc = total_target_correct / total_samples
    
    # 计算竞赛评估指标
    # 1. 二元F1（目标vs非目标）
    binary_f1 = f1_score(all_target_labels, all_target_preds, average='binary')
    
    # 2. 多类宏F1（仅对非目标手势）
    # 找出非目标样本的索引
    non_target_indices = [i for i, label in enumerate(all_target_labels) if label == 0]
    non_target_gesture_preds = [all_gesture_preds[i] for i in non_target_indices]
    non_target_gesture_labels = [all_gesture_labels[i] for i in non_target_indices]
    
    if len(non_target_indices) > 0 and len(np.unique(non_target_gesture_labels)) > 1:
        non_target_macro_f1 = f1_score(non_target_gesture_labels, non_target_gesture_preds, average='macro')
    else:
        non_target_macro_f1 = 0.0  # 处理特殊情况
    
    # 3. 最终评分：二元F1和非目标宏F1的平均值
    final_score = (binary_f1 + non_target_macro_f1) / 2
    
    # 4. 所有手势的宏F1（用于参考）
    all_gesture_macro_f1 = f1_score(all_gesture_labels, all_gesture_preds, average='macro')
    
    return (avg_loss, gesture_acc, target_acc, binary_f1, non_target_macro_f1, final_score,
            all_gesture_preds, all_gesture_labels, all_target_preds, all_target_labels)

# ----------------------
# 8. 新增：LabelSmoothingCrossEntropyLoss
# ----------------------
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.weight = weight  # 类别权重（兼容之前的类别平衡）
    
    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        # 软化标签：均匀分布 + 真实标签权重
        smooth_label = torch.full_like(logits, self.smoothing / (n_classes - 1))
        smooth_label.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # 计算交叉熵（支持类别权重）
        log_probs = torch.log_softmax(logits, dim=1)
        if self.weight is not None:
            smooth_label = smooth_label * self.weight.unsqueeze(0)  # 应用类别权重
        loss = -torch.sum(log_probs * smooth_label, dim=1).mean()
        return loss

# ----------------------
# 9. 模型训练与验证主函数（支持竞赛评估指标）
# ----------------------
def train_models(train_sequences, train_gestures, train_is_target, train_ids, 
                test_sequences, n_features, num_gestures, class_counts, 
                target_gestures, gesture_le, n_splits=5):
    """训练多折模型并进行预测，支持竞赛评估指标"""
    print("\n===== 训练模型（优化版） =====")
    
    # 处理参与者信息
    subjects = []
    for id in train_ids:
        try:
            if '_' in id:
                parts = id.split('_')
                if len(parts) >= 2:
                    subject = parts[1].split('-')[0]
                    subjects.append(subject)
                    continue
            subjects.append(str(hash(id) % 1000))
        except:
            subjects.append(str(hash(id) % 1000))
    
    subjects = np.array(subjects)
    
    # 计算手势类别权重（解决类别不平衡）
    class_weights = 1.0 / class_counts  # 反比于样本数
    class_weights = class_weights / class_weights.sum() * num_gestures  # 归一化
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"手势类别权重: {class_weights.cpu().numpy()}")
    
    # 计算二元分类权重
    target_counts = np.bincount(train_is_target)
    target_weights = torch.FloatTensor([1.0 / (count / len(train_is_target)) for count in target_counts]).to(device)
    print(f"二元分类权重: {target_weights.cpu().numpy()}")
    
    # 交叉验证
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    test_preds_gesture = np.zeros((len(test_sequences), num_gestures))
    test_preds_target = np.zeros((len(test_sequences), 2))  # 二元分类预测
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_sequences, groups=subjects)):
        print(f"\n===== 折 {fold+1}/{n_splits} =====")
        
        # 划分训练集和验证集
        X_train, X_val = train_sequences[train_idx], train_sequences[val_idx]
        y_gesture_train, y_gesture_val = train_gestures[train_idx], train_gestures[val_idx]
        y_target_train, y_target_val = train_is_target[train_idx], train_is_target[val_idx]
        
        # 创建数据集和数据加载器（加入数据增强标记和类别分布）
        train_dataset = SensorSequenceDataset(
            X_train, y_gesture_train, y_target_train, is_train=True, 
            aug_prob=0.5, class_counts=class_counts
        )
        val_dataset = SensorSequenceDataset(
            X_val, y_gesture_val, y_target_val, is_train=False
        )
        
        # 根据GPU内存调整batch size（双T4优化）
        batch_size = 64
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            if gpu_mem < 16:  # T4通常为16GB，小内存时调整
                print(f"检测到GPU内存 {gpu_mem:.1f}GB，调整batch size为32")
                batch_size = 32
        
        # 优化DataLoader参数（双GPU环境）
        num_workers = min(4, os.cpu_count() // 2)  # CPU核心的一半
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True  # 丢弃不完整批次，避免计算异常
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size * 2,  # 验证集batch_size翻倍（无反向传播）
            shuffle=False, 
            num_workers=num_workers//2,  # 验证集减少worker
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 初始化模型（轻量化）
        hidden_dim = 64 if n_features < 100 else 128  # 轻量化模型
        num_layers = 2  # 减少层数，防止过拟合
        
        model = CNNBiLSTMAttentionModel(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_gestures=num_gestures,
            dropout=0.3
        )
        
        # 多GPU并行（核心优化）
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练!")
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        # 定义损失函数和优化器
        gesture_criterion = LabelSmoothingCrossEntropyLoss(
            smoothing=0.1,  # 软化强度
            weight=class_weights
        )
        
        target_criterion = LabelSmoothingCrossEntropyLoss(
            smoothing=0.05,  # 二元分类轻微软化
            weight=target_weights
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        # 学习率调度：CosineAnnealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        
        # 训练模型
        best_final_score = 0
        patience = 10  # 延长早停耐心值
        counter = 0
        
        for epoch in range(50):
            print(f"\n epoch {epoch+1}/50 | 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 训练阶段
            train_loss, train_gesture_acc, train_target_acc = train_epoch(
                model, train_loader, gesture_criterion, target_criterion, optimizer, device
            )
            
            # 验证
            (val_loss, val_gesture_acc, val_target_acc, binary_f1, 
             non_target_macro_f1, final_score, gesture_preds, 
             gesture_labels, target_preds, target_labels) = validate(
                model, val_loader, gesture_criterion, target_criterion, 
                device, target_gestures, gesture_le
            )
            
            # 打印 epoch 结果
            print(f"训练损失: {train_loss:.4f}, 手势准确率: {train_gesture_acc:.4f}, 目标准确率: {train_target_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 手势准确率: {val_gesture_acc:.4f}, 目标准确率: {val_target_acc:.4f}")
            print(f"二元F1: {binary_f1:.4f}, 非目标宏F1: {non_target_macro_f1:.4f}")
            print(f"最终评分: {final_score:.4f}")
            
            # 学习率调度
            scheduler.step()
            
            # 早停机制：监控最终评分
            if final_score > best_final_score:
                best_final_score = final_score
                torch.save(model.state_dict(), f"best_model_fold_{fold+1}.pth")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f"best_model_fold_{fold+1}.pth"))
        
        # 再次验证
        (_, _, _, binary_f1, non_target_macro_f1, final_score, 
         gesture_preds, gesture_labels, target_preds, target_labels) = validate(
            model, val_loader, gesture_criterion, target_criterion, 
            device, target_gestures, gesture_le
        )
        
        print(f"\n折 {fold+1} 最佳结果:")
        print(f"二元F1: {binary_f1:.4f}, 非目标宏F1: {non_target_macro_f1:.4f}")
        print(f"最终评分: {final_score:.4f}")
        
        # 打印分类报告
        print("\n手势分类报告:")
        print(classification_report(gesture_labels, gesture_preds))
        
        print("\n二元分类报告:")
        print(classification_report(target_labels, target_preds))
        
        fold_results.append({
            'fold': fold+1,
            'binary_f1': binary_f1,
            'non_target_macro_f1': non_target_macro_f1,
            'final_score': final_score
        })
        
        # 测试集预测
        test_dataset = SensorSequenceDataset(test_sequences)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        model.eval()
        fold_test_gesture_preds = []
        fold_test_target_preds = []
        
        with torch.no_grad():
            for sequences in tqdm(test_loader, desc="测试集预测"):
                sequences = sequences.to(device)
                gesture_out, target_out = model(sequences)
                fold_test_gesture_preds.append(torch.softmax(gesture_out, dim=1).cpu().numpy())
                fold_test_target_preds.append(torch.softmax(target_out, dim=1).cpu().numpy())
        
        # 聚合测试集预测
        fold_test_gesture_preds = np.vstack(fold_test_gesture_preds)
        fold_test_target_preds = np.vstack(fold_test_target_preds)
        
        test_preds_gesture += fold_test_gesture_preds / n_splits
        test_preds_target += fold_test_target_preds / n_splits
    
    # 打印交叉验证汇总结果
    print("\n===== 交叉验证汇总 =====")
    results_df = pd.DataFrame(fold_results)
    print(results_df.mean())
    
    return test_preds_gesture, test_preds_target

# ----------------------
# 10. 生成提交文件
# ----------------------
def create_submission(test_preds_gesture, test_preds_target, test_ids, gesture_le, target_gestures):
    """生成符合竞赛要求的提交文件"""
    print("\n===== 生成提交文件 =====")
    
    # 从预测概率获取类别
    test_preds_gesture_class = np.argmax(test_preds_gesture, axis=1)
    test_preds_target_class = np.argmax(test_preds_target, axis=1)
    
    # 解码为原始手势名称
    gestures = gesture_le.inverse_transform(test_preds_gesture_class)
    
    # 生成提交文件，包含手势类别和是否为目标的预测
    submission = pd.DataFrame({
        'sequence_id': test_ids,
        'gesture': gestures,
        'is_target': test_preds_target_class  # 1表示目标，0表示非目标
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print(f"提交文件已生成，形状: {submission.shape}")
    print("前5行预览:")
    print(submission.head())
    
    return submission

# ----------------------
# 11. 主函数
# ----------------------
def main():
    try:
        # 加载数据并探索
        train, test, train_demo, test_demo = load_and_explore_data()
        
        # 预处理数据
        train, test, imu_cols, thm_cols, tof_cols, demo_cols, gesture_le, class_distribution, target_gestures = preprocess_data(
            train, test, train_demo, test_demo
        )
        
        # 自动确定合适的序列长度（核心优化：使用中位数而非1.5倍中位数）
        seq_lengths = train.groupby('sequence_id').size()
        print(f"\n序列长度分布: 平均={seq_lengths.mean():.1f}, 中位数={seq_lengths.median()}, 最大={seq_lengths.max()}")
        max_seq_length = int(seq_lengths.median())  # 直接使用中位数，减少30%序列长度
        print(f"自动设置最大序列长度为: {max_seq_length}")
        
        # 准备序列数据
        (train_sequences, train_gestures, train_is_target, train_ids,
         test_sequences, test_ids, n_features) = prepare_sequences(train, test, max_seq_length)
        
        # 清理内存
        del train, test
        gc.collect()
        
        # 获取手势类别数量
        num_gestures = len(np.unique(train_gestures))
        print(f"手势类别数量: {num_gestures}")
        
        # 如果特征数量为0，说明数据处理有问题
        if n_features == 0:
            raise ValueError("未找到任何特征列，请检查数据格式")
        
        # 训练模型并预测
        test_preds_gesture, test_preds_target = train_models(
            train_sequences, train_gestures, train_is_target, train_ids,
            test_sequences, n_features, num_gestures, 
            class_counts=class_distribution, target_gestures=target_gestures,
            gesture_le=gesture_le, n_splits=5
        )
        
        # 生成提交文件
        submission = create_submission(test_preds_gesture, test_preds_target, test_ids, gesture_le, target_gestures)
        
        print("\n所有任务完成!")
        
    except Exception as e:
        print(f"\n执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 检查GPU配置
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"第二GPU: {torch.cuda.get_device_name(1)}")

# 定义设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ----------------------
# 1. 数据加载与详细探索（新增前5行预览+数据可视化）
# ----------------------
def load_and_explore_data():
    """加载数据并进行详细探索，为后续处理提供依据"""
    print("加载数据...")
    try:
        train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
        test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
        train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
        test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("尝试使用备选路径...")
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        train_demo = pd.read_csv('train_demographics.csv')
        test_demo = pd.read_csv('test_demographics.csv')
    
    # 详细数据探索
    print("\n===== 数据探索 =====")
    print(f"训练集形状: {train.shape}")
    print(f"测试集形状: {test.shape}")
    print(f"训练集人口数据形状: {train_demo.shape}")
    print(f"测试集人口数据形状: {test_demo.shape}")
    
    # 新增：打印各数据前5行，直观查看数据结构
    print("\n----- 训练集前5行 -----")
    print(train.head())
    print("\n----- 测试集前5行 -----")
    print(test.head())
    print("\n----- 训练集人口数据前5行 -----")
    print(train_demo.head())
    print("\n----- 测试集人口数据前5行 -----")
    print(test_demo.head())
    
    print("\n训练集列名及数据类型（前10列）:")
    print(train.dtypes.head(10))
    print("\n测试集列名及数据类型（前10列）:")
    print(test.dtypes.head(10))
    
    # 检查特征差异
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    train_only_cols = train_cols - test_cols
    test_only_cols = test_cols - train_cols
    print(f"\n训练集特有列: {len(train_only_cols)}个 - {sorted(train_only_cols)[:5]}...")
    print(f"测试集特有列: {len(test_only_cols)}个 - {sorted(test_only_cols)[:5]}...")
    
    # 检查关键列是否存在
    required_columns = ['sequence_id', 'row_id']
    for col in required_columns:
        if col not in train.columns:
            print(f"警告: 训练集缺少必要列 '{col}'")
        if col not in test.columns:
            print(f"警告: 测试集缺少必要列 '{col}'")
    
    # 新增：数据分布可视化（核心特征+目标变量）
    print("\n----- 数据分布可视化 -----")
    # 1. 目标变量（gesture）分布（仅训练集）
    if 'gesture' in train.columns:
        plt.figure(figsize=(12, 5))
        # 子图1：手势类别数量分布
        plt.subplot(1, 2, 1)
        gesture_count = train['gesture'].value_counts()
        sns.barplot(x=gesture_count.index, y=gesture_count.values, palette='viridis')
        plt.title('Distribution of Gesture Categories', fontsize=12)
        plt.xlabel('Gesture Category')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        # 子图2：序列长度分布
        plt.subplot(1, 2, 2)
        seq_lengths = train.groupby('sequence_id').size()
        sns.histplot(seq_lengths, bins=20, kde=True, color='skyblue')
        plt.title('Distribution of Sequence Lengths', fontsize=12)
        plt.xlabel('Sequence Length (time steps)')
        plt.ylabel('Number of Sequences')
        plt.tight_layout()
        plt.show()
    
    # 2. 传感器特征分布（以IMU特征为例，取前3个加速度特征）
    imu_cols = [col for col in train.columns if col.startswith(('acc_', 'gyro_'))][:3]
    if len(imu_cols) >= 3:
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(imu_cols):
            plt.subplot(1, 3, i+1)
            sns.histplot(train[col].dropna(), bins=30, kde=True, color=f'C{i}')
            plt.title(f'Distribution of {col}', fontsize=10)
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    
    # 3. 人口统计学数据分布（如年龄、体重）
    demo_cols = [col for col in train_demo.columns if col in ['age', 'weight', 'height_cm']]
    if len(demo_cols) > 0:
        plt.figure(figsize=(10, 4))
        for i, col in enumerate(demo_cols):
            plt.subplot(1, len(demo_cols), i+1)
            sns.histplot(train_demo[col].dropna(), bins=15, kde=True, color='orange')
            plt.title(f'Distribution of {col}', fontsize=10)
            plt.xlabel(col)
        plt.tight_layout()
        plt.show()
    
    # 检查是否有共同的键用于合并人口统计学数据
    merge_key = 'subject' if 'subject' in train.columns and 'subject' in train_demo.columns else None
    if merge_key:
        print(f"\n将使用 '{merge_key}' 合并主数据和人口统计学数据")
    else:
        print("\n警告: 未找到合适的合并键，将不合并人口统计学数据")
    
    return train, test, train_demo, test_demo

# ----------------------
# 2. 数据预处理（核心优化：裁剪TOF特征+类别样本平衡+特征增强）
# ----------------------
def preprocess_data(train, test, train_demo, test_demo):
    """数据预处理：灵活处理不同结构的数据 + 类别平衡 + 特征增强 + TOF特征裁剪"""
    print("\n===== 数据预处理 =====")
    
    # 尝试合并人口统计学数据
    merge_key = 'subject' if 'subject' in train.columns and 'subject' in train_demo.columns else None
    if merge_key:
        print(f"使用 '{merge_key}' 合并数据...")
        train = train.merge(train_demo, on=merge_key, how='left')
        test = test.merge(test_demo, on=merge_key, how='left')
    else:
        print("无法合并人口统计学数据，跳过此步骤")
    
    # ----------------------
    # 核心优化1：裁剪TOF冗余特征（每5列保留1列，减少70%+维度）
    # ----------------------
    tof_cols_train = [col for col in train.columns if col.startswith('tof_')]
    tof_cols_test = [col for col in test.columns if col.startswith('tof_')]
    print(f"原始TOF特征数量（训练集）: {len(tof_cols_train)}")
    
    # 每5列保留1列（按索引步长筛选，兼顾物理含义连续性）
    keep_tof_cols = tof_cols_train[::5] if len(tof_cols_train) > 0 else []
    # 确保测试集与训练集保留的TOF列一致
    keep_tof_cols_test = [col for col in keep_tof_cols if col in tof_cols_test]
    
    # 删除冗余TOF列
    if len(tof_cols_train) > 0:
        drop_tof_train = [col for col in tof_cols_train if col not in keep_tof_cols]
        train = train.drop(columns=drop_tof_train)
    if len(tof_cols_test) > 0:
        drop_tof_test = [col for col in tof_cols_test if col not in keep_tof_cols_test]
        test = test.drop(columns=drop_tof_test)
    
    print(f"裁剪后TOF特征数量（训练集）: {len(keep_tof_cols)}")
    print(f"裁剪后TOF特征数量（测试集）: {len(keep_tof_cols_test)}")
    
    # 确保训练集和测试集有相同的特征列
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common_cols = train_cols.intersection(test_cols)
    train_only_cols = ['gesture', 'gesture_encoded', 'is_target']  # 新增is_target列
    train_only_cols = [col for col in train_only_cols if col in train_cols]
    print(f"仅在训练集中保留的列: {train_only_cols}")
    
    # 特征增强：新增传感器衍生特征（提升区分度）
    def add_sensor_derived_features(df):
        # 识别IMU传感器列（加速度、陀螺仪等）
        imu_cols = [col for col in df.columns if col.startswith(('acc_', 'gyro_', 'rot_', 'mag_'))]
        if len(imu_cols) >=3:
            # 计算加速度向量模长（物理意义：总加速度大小）
            if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
                df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
            # 计算陀螺仪向量模长（旋转强度）
            if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
                df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
            # 计算传感器数据一阶差分（变化率）- 限制数量，避免特征爆炸
            for col in imu_cols[:6]:  # 仅保留前6个IMU列的差分
                df[f'{col}_diff1'] = df.groupby('sequence_id')[col].diff().fillna(0)
        # 人口统计学衍生特征（如BMI）
        if all(col in df.columns for col in ['weight', 'height_cm']):
            df['bmi'] = df['weight'] / ((df['height_cm']/100)**2)
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        return df
    
    # 为训练集和测试集添加衍生特征
    train = add_sensor_derived_features(train)
    test = add_sensor_derived_features(test)
    print("已添加传感器衍生特征（模长、一阶差分）和人口统计学衍生特征（BMI）")
    
    # 重新对齐特征列（包含新增的衍生特征）
    train_features = list(common_cols) + train_only_cols
    test_features = list(common_cols)
    for col in ['sequence_id', 'row_id', 'subject', 'acc_magnitude', 'gyro_magnitude', 'bmi']:
        if col in train.columns and col not in train_features:
            train_features.append(col)
        if col in test.columns and col not in test_features:
            test_features.append(col)
    
    # 应用特征选择
    train = train[train_features]
    test = test[test_features]
    print(f"训练集特征列数量（含衍生特征）: {len(train_features)}")
    print(f"测试集特征列数量（含衍生特征）: {len(test_features)}")
    
    # 检查并处理缺失值（分类型优化填充策略）
    print(f"训练集缺失值比例: {train.isnull().mean().mean():.4f}")
    print(f"测试集缺失值比例: {test.isnull().mean().mean():.4f}")
    
    imu_cols = [col for col in train.columns if col.startswith(('acc_', 'rot_', 'gyro_', 'mag_')) or col.endswith('_diff1')]
    thm_cols = [col for col in train.columns if col.startswith(('thm_', 'temp_', 'thermal_'))]
    tof_cols = [col for col in train.columns if col.startswith(('tof_', 'distance_'))]
    demo_cols = [col for col in train.columns if col in ['age', 'height', 'height_cm', 'weight', 'bmi', 'shoulder_to_wrist', 'shoulder_to_wrist_cm', 'elbow_to_wrist', 'elbow_to_wrist_cm']]
    
    print(f"找到 {len(imu_cols)} 个IMU相关列（含衍生特征）")
    print(f"找到 {len(thm_cols)} 个热传感器列")
    print(f"找到 {len(tof_cols)} 个飞行时间传感器列")
    print(f"找到 {len(demo_cols)} 个人口统计学列（含衍生特征）")
    
    # 分类型填充缺失值（更贴合数据分布）
    if imu_cols:
        train[imu_cols] = train.groupby('sequence_id')[imu_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').fillna(x.median()))
        test[imu_cols] = test.groupby('sequence_id')[imu_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').fillna(x.median()))
    if thm_cols:
        train[thm_cols] = train.groupby('sequence_id')[thm_cols].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean()))
        test[thm_cols] = test.groupby('sequence_id')[thm_cols].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean()))
    if tof_cols:
        train[tof_cols] = train.groupby('sequence_id')[tof_cols].transform(
            lambda x: x.fillna(x.quantile(0.5)))  # 中位数填充，抗异常值
        test[tof_cols] = test.groupby('sequence_id')[tof_cols].transform(
            lambda x: x.fillna(x.quantile(0.5)))
    if demo_cols:
        train[demo_cols] = train[demo_cols].fillna(train[demo_cols].median())
        test[demo_cols] = test[demo_cols].fillna(test[demo_cols].median())
    
    # 编码目标变量
    if 'gesture' not in train.columns:
        raise ValueError("训练集必须包含'gesture'列作为目标变量")
    gesture_le = LabelEncoder()
    train['gesture_encoded'] = gesture_le.fit_transform(train['gesture'])
    
    # 定义目标类型（根据竞赛要求，这里假设需要确定哪些是目标手势）
    unique_gestures = train['gesture'].unique()
    target_gestures = unique_gestures[:len(unique_gestures)//2]
    train['is_target'] = train['gesture'].isin(target_gestures).astype(int)
    
    print(f"已编码 {len(gesture_le.classes_)} 种手势类别")
    print(f"目标手势: {len(target_gestures)}种, 非目标手势: {len(unique_gestures)-len(target_gestures)}种")
    
    # 分析类别分布
    class_distribution = train['gesture_encoded'].value_counts().sort_index()
    print("\n手势类别分布（原始）:")
    for cls, count in class_distribution.items():
        print(f"类别 {cls}: {count} 样本 ({count/len(train):.2%})")
    
    # 二元类别分布
    target_distribution = train['is_target'].value_counts().sort_index()
    print("\n二元目标分布（原始）:")
    for cls, count in target_distribution.items():
        print(f"{'目标' if cls == 1 else '非目标'}: {count} 样本 ({count/len(train):.2%})")
    
    # ----------------------
    # 类别样本平衡（序列级SMOTE过采样，避免单样本重复）
    # ----------------------
    print("\n===== 类别样本平衡（序列级SMOTE过采样） =====")
    max_count = class_distribution.max()
    seq_class_map = train.groupby('sequence_id')['gesture_encoded'].first().to_dict()
    seq_list = train['sequence_id'].unique()
    
    # 按类别分组序列
    seq_by_class = {cls: [] for cls in class_distribution.index}
    for seq_id in seq_list:
        cls = seq_class_map[seq_id]
        seq_by_class[cls].append(seq_id)
    
    # 序列级SMOTE过采样（生成"相似但不重复"的序列）
    def generate_smote_seq(base_seq_df, n_generate, seq_id_prefix):
        generated_seqs = []
        for i in range(n_generate):
            # 对传感器特征添加微小扰动（模拟真实数据变异）
            imu_feat_cols = [col for col in base_seq_df.columns if col.startswith(('acc_', 'gyro_', 'rot_', 'mag_')) or col.endswith('_diff1')]
            perturb_df = base_seq_df.copy()
            if len(imu_feat_cols) > 0:
                perturb_scale = 0.03  # 扰动强度（控制在3%以内，保证物理意义）
                perturb = np.random.normal(0, perturb_scale, size=(len(perturb_df), len(imu_feat_cols)))
                perturb_df[imu_feat_cols] += perturb * perturb_df[imu_feat_cols].std(axis=0).values
            # 生成新序列ID
            new_seq_id = f"{seq_id_prefix}_smote_{i}"
            perturb_df['sequence_id'] = new_seq_id
            generated_seqs.append(perturb_df)
        return pd.concat(generated_seqs, ignore_index=True) if generated_seqs else pd.DataFrame()
    
    # 对少数类生成新序列
    oversampled_train = []
    for cls in class_distribution.index:
        cls_seqs = seq_by_class[cls]
        current_seq_count = len(cls_seqs)
        
        # 若当前序列数小于最大序列数，生成新序列（限制生成量，避免数据膨胀）
        if current_seq_count < max_count:
            need_seq_count = min(max_count - current_seq_count, current_seq_count * 2)  # 最多生成2倍原始序列
            base_seq_ids = np.random.choice(cls_seqs, size=min(need_seq_count, current_seq_count), replace=True)
            generated_samples = []
            for base_seq_id in base_seq_ids:
                base_seq_df = train[train['sequence_id'] == base_seq_id].copy()
                gen_seq = generate_smote_seq(base_seq_df, n_generate=1, seq_id_prefix=base_seq_id)
                if not gen_seq.empty:
                    generated_samples.append(gen_seq)
            # 合并原始序列和生成序列
            cls_original = train[train['sequence_id'].isin(cls_seqs)]
            cls_generated = pd.concat(generated_samples, ignore_index=True) if generated_samples else pd.DataFrame()
            oversampled_train.append(pd.concat([cls_original, cls_generated], ignore_index=True))
        else:
            # 多数类直接保留
            oversampled_train.append(train[train['sequence_id'].isin(cls_seqs)])
    
    # 合并过采样后的数据
    train = pd.concat(oversampled_train, ignore_index=True)
    # 重新计算类别分布
    oversampled_dist = train['gesture_encoded'].value_counts().sort_index()
    print(f"过采样前训练集总样本数: {len(seq_list) * train.groupby('sequence_id').size().mean():.0f}")
    print(f"过采样后训练集总样本数: {len(train)}")
    print(f"过采样后类别分布:")
    for cls, count in oversampled_dist.items():
        print(f"类别 {cls}: {count} 样本 ({count/len(train):.2%})")
    
    # 新增：预处理后数据预览和可视化
    print("\n----- 预处理后数据预览 -----")
    print("训练集前5行:")
    print(train.head())
    print("测试集前5行:")
    print(test.head())
    
    # 新增：预处理后数据分布可视化
    print("\n----- 预处理后数据分布验证 -----")
    # 1. 过采样后的类别分布对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=class_distribution.index, y=class_distribution.values, palette='viridis')
    plt.title('Class Distribution Before Oversampling', fontsize=12)
    plt.xlabel('Gesture Class')
    plt.ylabel('Number of Samples')
    plt.subplot(1, 2, 2)
    sns.barplot(x=oversampled_dist.index, y=oversampled_dist.values, palette='viridis')
    plt.title('Class Distribution After Oversampling', fontsize=12)
    plt.xlabel('Gesture Class')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.show()
    
    # 2. 衍生特征分布验证（以加速度模长为例）
    if 'acc_magnitude' in train.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(train['acc_magnitude'], bins=30, kde=True, color='green')
        plt.title('Distribution of Acceleration Magnitude', fontsize=12)
        plt.xlabel('Acceleration Magnitude')
        plt.ylabel('Frequency')
        plt.show()
    
    # 3. 二元目标分布
    plt.figure(figsize=(8, 4))
    target_dist = train['is_target'].value_counts()
    sns.barplot(x=target_dist.index, y=target_dist.values, palette=['red', 'blue'])
    plt.title('Distribution of Target Classes', fontsize=12)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks([0, 1], ['Non-target', 'Target'])
    plt.show()
    
    return train, test, imu_cols, thm_cols, tof_cols, demo_cols, gesture_le, class_distribution, target_gestures

# ----------------------
# 3. 优化：时序统计特征提取函数（简化计算+分批处理）
# ----------------------
def extract_temporal_stats(group, feature_cols):
    """简化版时序统计特征提取，仅保留核心统计量，减少计算量"""
    stats = {}
    # 1. 仅保留核心统计量（删除自相关、高阶差分等复杂特征）
    stats['mean'] = group[feature_cols].mean()
    stats['std'] = group[feature_cols].std()
    stats['max'] = group[feature_cols].max()
    stats['min'] = group[feature_cols].min()
    stats['median'] = group[feature_cols].median()
    stats['ptp'] = group[feature_cols].max() - group[feature_cols].min()
    
    # 2. 一阶差分的基础统计（捕捉变化率）
    if len(group) > 1:
        diff1 = group[feature_cols].diff().fillna(0)
        stats['diff1_mean'] = diff1.mean()
        stats['diff1_std'] = diff1.std()
    
    # 拼接为DataFrame（每行对应一个序列的统计特征）
    stats_df = pd.DataFrame([stats])
    
    # 确保所有值都是数值类型，并填充任何NaN
    for col in stats_df.columns:
        stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce').fillna(0.0)
    
    # 重命名列（避免与原始特征冲突）
    stats_df.columns = [f'{col}_stat' for col in stats_df.columns]
    return stats_df

# 分批提取时序统计特征（避免内存溢出）
def extract_temporal_stats_batch(data, feature_cols, batch_size=1000):
    """分批处理时序统计特征提取，降低内存占用"""
    sequence_ids = data['sequence_id'].unique()
    stats_list = []
    # 按批次处理序列
    for i in range(0, len(sequence_ids), batch_size):
        batch_seq_ids = sequence_ids[i:i+batch_size]
        batch_data = data[data['sequence_id'].isin(batch_seq_ids)]
        # 按序列计算统计特征
        batch_stats = batch_data.groupby('sequence_id').apply(
            lambda x: extract_temporal_stats(x, feature_cols)
        ).reset_index(drop=True)
        stats_list.append(batch_stats)
        # 释放批次内存
        del batch_data
        gc.collect()
    
    return pd.concat(stats_list, ignore_index=True) if stats_list else pd.DataFrame()

# ----------------------
# 4. 优化：prepare_sequences函数（动态调整序列长度+内存优化）
# ----------------------
def prepare_sequences(train, test, max_seq_length=None):
    print("\n===== 准备序列数据（含时序统计特征） =====")
    
    # 1. 获取共同特征列
    if 'sequence_id' not in train.columns or 'sequence_id' not in test.columns:
        raise ValueError("数据必须包含'sequence_id'列以区分不同序列")
    
    train_exclude_cols = ['sequence_id', 'row_id', 'gesture', 'gesture_encoded', 'subject', 'is_target']
    test_exclude_cols = ['sequence_id', 'row_id', 'subject']
    train_exclude_cols = [col for col in train_exclude_cols if col in train.columns]
    test_exclude_cols = [col for col in test_exclude_cols if col in test.columns]
    
    train_feature_cols = [col for col in train.columns if col not in train_exclude_cols]
    test_feature_cols = [col for col in test.columns if col not in test_exclude_cols]
    common_feature_cols = list(set(train_feature_cols).intersection(set(test_feature_cols)))
    common_feature_cols.sort()
    print(f"共同特征列: {len(common_feature_cols)}")
    
    if len(common_feature_cols) == 0:
        raise ValueError("未找到共同的特征列，无法继续")
    
    # 2. 分批提取序列级统计特征（优化内存）
    print("提取时序统计特征...")
    # 训练集统计特征
    try:
        train_stats = extract_temporal_stats_batch(train, common_feature_cols, batch_size=500)
        # 确保索引正确对应
        unique_train_ids = train['sequence_id'].unique()
        if len(train_stats) != len(unique_train_ids):
            print(f"警告：训练集统计特征数量({len(train_stats)})与序列数量({len(unique_train_ids)})不匹配，重新对齐")
            train_stats = train_stats.iloc[:len(unique_train_ids)]
        train_stats.index = unique_train_ids
    except Exception as e:
        print(f"训练集统计特征提取出错: {e}")
        # 创建默认的统计特征
        train_stats = pd.DataFrame(0.0, index=train['sequence_id'].unique(), 
                                  columns=[f'feat_{i}_stat' for i in range(10)])
    
    # 测试集统计特征
    try:
        test_stats = extract_temporal_stats_batch(test, common_feature_cols, batch_size=500)
        # 确保索引正确对应
        unique_test_ids = test['sequence_id'].unique()
        if len(test_stats) != len(unique_test_ids):
            print(f"警告：测试集统计特征数量({len(test_stats)})与序列数量({len(unique_test_ids)})不匹配，重新对齐")
            test_stats = test_stats.iloc[:len(unique_test_ids)]
        test_stats.index = unique_test_ids
    except Exception as e:
        print(f"测试集统计特征提取出错: {e}")
        # 创建默认的统计特征
        test_stats = pd.DataFrame(0.0, index=test['sequence_id'].unique(), 
                                 columns=[f'feat_{i}_stat' for i in range(10)])
    
    # 3. 处理训练序列（拼接统计特征）
    train_sequences = []
    train_gestures = []
    train_is_target = []  # 新增：二元目标标签
    train_ids = []
    
    print("处理训练序列...")
    for seq_id, group in tqdm(train.groupby('sequence_id'), desc="处理训练序列"):
        try:
            # 原始时序特征
            seq_data = group[common_feature_cols].copy()
            # 填充/转换数值类型
            for col in seq_data.columns:
                try:
                    seq_data[col] = pd.to_numeric(seq_data[col], errors='coerce').fillna(seq_data[col].median())
                except:
                    le = LabelEncoder()
                    seq_data[col] = le.fit_transform(seq_data[col].astype(str))
            
            # 获取当前序列的统计特征
            if seq_id in train_stats.index:
                seq_stat = train_stats.loc[seq_id].values.reshape(1, -1)  # (1, n_stats)
            else:
                # 如果找不到对应的统计特征，使用0填充
                seq_stat = np.zeros((1, train_stats.shape[1]))
                print(f"警告：序列 {seq_id} 未找到统计特征，使用默认值")
            
            # 确保统计特征是有效的数值数组
            seq_stat = np.nan_to_num(seq_stat.astype(np.float32))
            
            # 广播到每个时间步
            seq_stat_broadcast = np.repeat(seq_stat, len(seq_data), axis=0)  # (seq_len, n_stats)
            
            # 确保原始特征是有效的数值数组
            seq_data_values = np.nan_to_num(seq_data.values.astype(np.float32))
            
            # 拼接特征
            seq_values = np.hstack([seq_data_values, seq_stat_broadcast])
            
            # 填充/截断到固定长度
            if len(seq_values) < max_seq_length:
                pad_length = max_seq_length - len(seq_values)
                seq_values = np.pad(seq_values, ((0, pad_length), (0, 0)), mode='edge')
            else:
                seq_values = seq_values[:max_seq_length]
                
            train_sequences.append(seq_values)
            train_gestures.append(group['gesture_encoded'].iloc[0])
            train_is_target.append(group['is_target'].iloc[0])  # 保存二元标签
            train_ids.append(seq_id)
        except Exception as e:
            print(f"处理训练序列 {seq_id} 时出错: {e}")
            # 创建一个默认序列避免中断
            default_seq = np.zeros((max_seq_length, len(common_feature_cols) + train_stats.shape[1]), dtype=np.float32)
            train_sequences.append(default_seq)
            train_gestures.append(0)
            train_is_target.append(0)  # 默认非目标
            train_ids.append(seq_id)
    
    # 4. 处理测试序列
    test_sequences = []
    test_ids = []
    
    print("处理测试序列...")
    for seq_id, group in tqdm(test.groupby('sequence_id'), desc="处理测试序列"):
        try:
            seq_data = group[common_feature_cols].copy()
            for col in seq_data.columns:
                try:
                    seq_data[col] = pd.to_numeric(seq_data[col], errors='coerce').fillna(seq_data[col].median())
                except:
                    le = LabelEncoder()
                    seq_data[col] = le.fit_transform(seq_data[col].astype(str))
            
            # 获取当前序列的统计特征
            if seq_id in test_stats.index:
                seq_stat = test_stats.loc[seq_id].values.reshape(1, -1)  # (1, n_stats)
            else:
                seq_stat = np.zeros((1, test_stats.shape[1]))
                print(f"警告：测试序列 {seq_id} 未找到统计特征，使用默认值")
            
            # 确保统计特征是有效的数值数组
            seq_stat = np.nan_to_num(seq_stat.astype(np.float32))
            
            # 广播到每个时间步
            seq_stat_broadcast = np.repeat(seq_stat, len(seq_data), axis=0)  # (seq_len, n_stats)
            
            # 确保原始特征是有效的数值数组
            seq_data_values = np.nan_to_num(seq_data.values.astype(np.float32))
            
            # 拼接特征
            seq_values = np.hstack([seq_data_values, seq_stat_broadcast])
            
            # 填充/截断
            if len(seq_values) < max_seq_length:
                pad_length = max_seq_length - len(seq_values)
                seq_values = np.pad(seq_values, ((0, pad_length), (0, 0)), mode='edge')
            else:
                seq_values = seq_values[:max_seq_length]
                
            test_sequences.append(seq_values)
            test_ids.append(seq_id)
        except Exception as e:
            print(f"处理测试序列 {seq_id} 时出错: {e}")
            # 创建一个默认序列避免中断
            default_seq = np.zeros((max_seq_length, len(common_feature_cols) + test_stats.shape[1]), dtype=np.float32)
            test_sequences.append(default_seq)
            test_ids.append(seq_id)
    
    # 5. 后续处理
    train_sequences = np.array(train_sequences, dtype=np.float32)
    train_gestures = np.array(train_gestures)
    train_is_target = np.array(train_is_target)  # 转换为numpy数组
    test_sequences = np.array(test_sequences, dtype=np.float32)
    
    print(f"训练序列形状: {train_sequences.shape}")
    print(f"测试序列形状: {test_sequences.shape}")
    
    # 特征标准化
    if len(train_sequences) > 0 and train_sequences.ndim == 3 and train_sequences.shape[2] > 0:
        scaler = StandardScaler()
        n_samples, n_timesteps, n_features = train_sequences.shape
        train_reshaped = train_sequences.reshape(n_samples * n_timesteps, n_features)
        train_reshaped = scaler.fit_transform(train_reshaped)
        train_sequences = train_reshaped.reshape(n_samples, n_timesteps, n_features)
        
        if len(test_sequences) > 0 and test_sequences.ndim == 3 and test_sequences.shape[2] > 0:
            n_samples_test, n_timesteps_test, n_features_test = test_sequences.shape
            test_reshaped = test_sequences.reshape(n_samples_test * n_timesteps_test, n_features_test)
            test_reshaped = scaler.transform(test_reshaped)
            test_sequences = test_reshaped.reshape(n_samples_test, n_timesteps_test, n_features_test)
    
    # 清理内存
    del train_stats, test_stats
    gc.collect()
    
    n_features = train_sequences.shape[2] if (len(train_sequences.shape) > 2 and train_sequences.shape[2] > 0) else 0
    print(f"最终特征数量（原始+统计）: {n_features}")
    
    return (train_sequences, train_gestures, train_is_target, train_ids,
            test_sequences, test_ids, n_features)

# ----------------------
# 5. 修改：SensorSequenceDataset（加入针对少数类的增强）
# ----------------------
class SensorSequenceDataset(Dataset):
    def __init__(self, sequences, gestures=None, is_target=None, is_train=False, aug_prob=0.5, class_counts=None):
        self.sequences = sequences
        self.gestures = gestures
        self.is_target = is_target  # 新增：二元目标标签
        self.is_train = is_train  # 标记是否为训练集（仅训练时增强）
        self.aug_prob = aug_prob  # 基础增强概率
        self.class_counts = class_counts  # 类别分布，用于调整少数类的增强概率
        
        # 计算每个类别的增强权重（少数类增强概率更高）
        if class_counts is not None:
            max_count = class_counts.max()
            self.aug_weights = np.sqrt(max_count / class_counts)  # 少数类权重更高
            self.aug_weights = self.aug_weights / self.aug_weights.max()  # 归一化到[0,1]
        else:
            self.aug_weights = None
    
    def __len__(self):
        return len(self.sequences)
    
    def _augment_sequence(self, seq, aug_strength=1.0):
        """时序数据增强逻辑，支持调整增强强度"""
        seq = seq.copy()
        n_timesteps, n_features = seq.shape
        
        # 1. 高斯噪声注入（概率随增强强度增加）
        if np.random.random() < self.aug_prob * aug_strength:
            noise_scale = 0.05 * aug_strength  # 噪声强度随增强强度增加
            noise = np.random.normal(0, noise_scale * seq.std(axis=0), seq.shape)
            seq += noise
        
        # 2. 时间轴拉伸/压缩（概率随增强强度增加）
        if np.random.random() < self.aug_prob * 0.8 * aug_strength:
            scale_range = 0.2 * aug_strength  # 缩放范围随增强强度增加
            scale = np.random.uniform(1 - scale_range, 1 + scale_range)
            new_timesteps = int(n_timesteps * scale)
            # 线性插值调整长度
            seq_stretched = np.zeros((new_timesteps, n_features))
            for i in range(n_features):
                seq_stretched[:, i] = np.interp(
                    np.linspace(0, 1, new_timesteps),
                    np.linspace(0, 1, n_timesteps),
                    seq[:, i]
                )
            # 恢复原长度（截断/填充）
            if new_timesteps > n_timesteps:
                seq = seq_stretched[:n_timesteps]
            else:
                seq = np.pad(seq_stretched, ((0, n_timesteps - new_timesteps), (0, 0)), mode='edge')
        
        return seq.astype(np.float32)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 训练时应用增强，少数类增强强度更高
        if self.is_train and self.gestures is not None:
            gesture = self.gestures[idx]
            # 根据类别调整增强强度（少数类增强更强）
            if self.aug_weights is not None:
                aug_strength = 1.0 + self.aug_weights[gesture]  # 基础1.0 + 权重
            else:
                aug_strength = 1.0
            
            sequence = self._augment_sequence(sequence, aug_strength)
        
        sequence = torch.FloatTensor(sequence)
        
        if self.gestures is not None and self.is_target is not None:
            gesture = torch.LongTensor([self.gestures[idx]])[0]
            is_target = torch.LongTensor([self.is_target[idx]])[0]
            return sequence, gesture, is_target
        elif self.gestures is not None:
            gesture = torch.LongTensor([self.gestures[idx]])[0]
            return sequence, gesture
        else:
            return sequence

# ----------------------
# 6. 替换：CNN+BiLSTM+Self-Attention模型（轻量化）
# ----------------------
class CNNBiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_gestures, dropout=0.3):
        super(CNNBiLSTMAttentionModel, self).__init__()
        
        # 1. 1D CNN层（提取局部特征）
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),  # BatchNorm加速训练
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 双向LSTM层（输入维度=CNN输出维度）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Self-Attention层（关注关键时间步）
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 输入：BiLSTM输出（2*hidden_dim）
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # 对时间步加权
        )
        
        # 4. 全连接层 - 手势分类和二元分类共享特征
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. 手势分类头
        self.fc_gesture = nn.Linear(hidden_dim, num_gestures)
        
        # 6. 二元分类头（目标vs非目标）
        self.fc_target = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # 1. CNN前向（需转置：(batch, input_dim, seq_len) → 符合Conv1d输入格式）
        x_cnn = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x_cnn = self.cnn(x_cnn)      # (batch, hidden_dim, seq_len)
        x_cnn = x_cnn.permute(0, 2, 1)  # 转回：(batch, seq_len, hidden_dim)
        
        # 2. BiLSTM前向
        lstm_out, _ = self.lstm(x_cnn)  # (batch, seq_len, 2*hidden_dim)
        
        # 3. Self-Attention：对每个时间步加权
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = attn_weights.repeat(1, 1, lstm_out.size(-1))  # (batch, seq_len, 2*hidden_dim)
        attn_out = torch.mul(lstm_out, attn_weights).sum(dim=1)  # (batch, 2*hidden_dim) → 加权求和
        
        # 4. 共享特征层
        shared_features = self.shared_fc(attn_out)  # (batch, hidden_dim)
        
        # 5. 分类输出
        gesture_out = self.fc_gesture(shared_features)  # 手势分类
        target_out = self.fc_target(shared_features)    # 二元分类（目标vs非目标）
        
        return gesture_out, target_out

# ----------------------
# 7. 训练和验证函数（新增二元分类评估）
# ----------------------
def train_epoch(model, train_loader, gesture_criterion, target_criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_gesture_correct = 0
    total_target_correct = 0
    total_samples = 0
    
    for sequences, gestures, is_target in tqdm(train_loader, desc="训练"):
        sequences = sequences.to(device)
        gestures = gestures.to(device)
        is_target = is_target.to(device)
        
        optimizer.zero_grad()
        gesture_out, target_out = model(sequences)
        
        # 计算两个损失并加权求和
        loss_gesture = gesture_criterion(gesture_out, gestures)
        loss_target = target_criterion(target_out, is_target)
        loss = loss_gesture + loss_target  # 同等权重
        
        loss.backward()
        # 梯度裁剪（阈值5.0，防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * sequences.size(0)
        
        # 手势分类准确率
        _, predicted_gesture = torch.max(gesture_out.data, 1)
        total_gesture_correct += (predicted_gesture == gestures).sum().item()
        
        # 二元分类准确率
        _, predicted_target = torch.max(target_out.data, 1)
        total_target_correct += (predicted_target == is_target).sum().item()
        
        total_samples += sequences.size(0)
    
    avg_loss = total_loss / total_samples
    gesture_acc = total_gesture_correct / total_samples
    target_acc = total_target_correct / total_samples
    
    return avg_loss, gesture_acc, target_acc

def validate(model, val_loader, gesture_criterion, target_criterion, device, target_gestures=None, gesture_le=None):
    model.eval()
    total_loss = 0
    total_gesture_correct = 0
    total_target_correct = 0
    total_samples = 0
    
    all_gesture_preds = []
    all_gesture_labels = []
    all_target_preds = []
    all_target_labels = []
    
    with torch.no_grad():
        for sequences, gestures, is_target in tqdm(val_loader, desc="验证"):
            sequences = sequences.to(device)
            gestures = gestures.to(device)
            is_target = is_target.to(device)
            
            gesture_out, target_out = model(sequences)
            
            # 计算两个损失并加权求和
            loss_gesture = gesture_criterion(gesture_out, gestures)
            loss_target = target_criterion(target_out, is_target)
            loss = loss_gesture + loss_target  # 同等权重
            
            total_loss += loss.item() * sequences.size(0)
            
            # 手势分类准确率
            _, predicted_gesture = torch.max(gesture_out.data, 1)
            total_gesture_correct += (predicted_gesture == gestures).sum().item()
            
            # 二元分类准确率
            _, predicted_target = torch.max(target_out.data, 1)
            total_target_correct += (predicted_target == is_target).sum().item()
            
            total_samples += sequences.size(0)
            
            all_gesture_preds.extend(predicted_gesture.cpu().numpy())
            all_gesture_labels.extend(gestures.cpu().numpy())
            all_target_preds.extend(predicted_target.cpu().numpy())
            all_target_labels.extend(is_target.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    gesture_acc = total_gesture_correct / total_samples
    target_acc = total_target_correct / total_samples
    
    # 计算竞赛评估指标
    # 1. 二元F1（目标vs非目标）
    binary_f1 = f1_score(all_target_labels, all_target_preds, average='binary')
    
    # 2. 多类宏F1（仅对非目标手势）
    # 找出非目标样本的索引
    non_target_indices = [i for i, label in enumerate(all_target_labels) if label == 0]
    non_target_gesture_preds = [all_gesture_preds[i] for i in non_target_indices]
    non_target_gesture_labels = [all_gesture_labels[i] for i in non_target_indices]
    
    if len(non_target_indices) > 0 and len(np.unique(non_target_gesture_labels)) > 1:
        non_target_macro_f1 = f1_score(non_target_gesture_labels, non_target_gesture_preds, average='macro')
    else:
        non_target_macro_f1 = 0.0  # 处理特殊情况
    
    # 3. 最终评分：二元F1和非目标宏F1的平均值
    final_score = (binary_f1 + non_target_macro_f1) / 2
    
    # 4. 所有手势的宏F1（用于参考）
    all_gesture_macro_f1 = f1_score(all_gesture_labels, all_gesture_preds, average='macro')
    
    return (avg_loss, gesture_acc, target_acc, binary_f1, non_target_macro_f1, final_score,
            all_gesture_preds, all_gesture_labels, all_target_preds, all_target_labels)

# ----------------------
# 8. 新增：LabelSmoothingCrossEntropyLoss
# ----------------------
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.weight = weight  # 类别权重（兼容之前的类别平衡）
    
    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        # 软化标签：均匀分布 + 真实标签权重
        smooth_label = torch.full_like(logits, self.smoothing / (n_classes - 1))
        smooth_label.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # 计算交叉熵（支持类别权重）
        log_probs = torch.log_softmax(logits, dim=1)
        if self.weight is not None:
            smooth_label = smooth_label * self.weight.unsqueeze(0)  # 应用类别权重
        loss = -torch.sum(log_probs * smooth_label, dim=1).mean()
        return loss

# ----------------------
# 9. 模型训练与验证主函数（支持竞赛评估指标+Loss-Epoch曲线）
# ----------------------
def train_models(train_sequences, train_gestures, train_is_target, train_ids, 
                test_sequences, n_features, num_gestures, class_counts, 
                target_gestures, gesture_le, n_splits=5):
    """训练多折模型并进行预测，支持竞赛评估指标"""
    print("\n===== 训练模型（优化版） =====")
    
    # 处理参与者信息
    subjects = []
    for id in train_ids:
        try:
            if '_' in id:
                parts = id.split('_')
                if len(parts) >= 2:
                    subject = parts[1].split('-')[0]
                    subjects.append(subject)
                    continue
            subjects.append(str(hash(id) % 1000))
        except:
            subjects.append(str(hash(id) % 1000))
    
    subjects = np.array(subjects)
    
    # 计算手势类别权重（解决类别不平衡）
    class_weights = 1.0 / class_counts  # 反比于样本数
    class_weights = class_weights / class_weights.sum() * num_gestures  # 归一化
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"手势类别权重: {class_weights.cpu().numpy()}")
    
    # 计算二元分类权重
    target_counts = np.bincount(train_is_target)
    target_weights = torch.FloatTensor([1.0 / (count / len(train_is_target)) for count in target_counts]).to(device)
    print(f"二元分类权重: {target_weights.cpu().numpy()}")
    
    # 交叉验证
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    test_preds_gesture = np.zeros((len(test_sequences), num_gestures))
    test_preds_target = np.zeros((len(test_sequences), 2))  # 二元分类预测
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_sequences, groups=subjects)):
        print(f"\n===== 折 {fold+1}/{n_splits} =====")
        
        # 划分训练集和验证集
        X_train, X_val = train_sequences[train_idx], train_sequences[val_idx]
        y_gesture_train, y_gesture_val = train_gestures[train_idx], train_gestures[val_idx]
        y_target_train, y_target_val = train_is_target[train_idx], train_is_target[val_idx]
        
        # 创建数据集和数据加载器（加入数据增强标记和类别分布）
        train_dataset = SensorSequenceDataset(
            X_train, y_gesture_train, y_target_train, is_train=True, 
            aug_prob=0.5, class_counts=class_counts
        )
        val_dataset = SensorSequenceDataset(
            X_val, y_gesture_val, y_target_val, is_train=False
        )
        
        # 根据GPU内存调整batch size（双T4优化）
        batch_size = 64
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            if gpu_mem < 16:  # T4通常为16GB，小内存时调整
                print(f"检测到GPU内存 {gpu_mem:.1f}GB，调整batch size为32")
                batch_size = 32
        
        # 优化DataLoader参数（双GPU环境）
        num_workers = min(4, os.cpu_count() // 2)  # CPU核心的一半
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True  # 丢弃不完整批次，避免计算异常
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size * 2,  # 验证集batch_size翻倍（无反向传播）
            shuffle=False, 
            num_workers=num_workers//2,  # 验证集减少worker
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 初始化模型（轻量化）
        hidden_dim = 64 if n_features < 100 else 128  # 轻量化模型
        num_layers = 2  # 减少层数，防止过拟合
        
        model = CNNBiLSTMAttentionModel(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_gestures=num_gestures,
            dropout=0.3
        )
        
        # 多GPU并行（核心优化）
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练!")
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        # 定义损失函数和优化器
        gesture_criterion = LabelSmoothingCrossEntropyLoss(
            smoothing=0.1,  # 软化强度
            weight=class_weights
        )
        
        target_criterion = LabelSmoothingCrossEntropyLoss(
            smoothing=0.05,  # 二元分类轻微软化
            weight=target_weights
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        # 学习率调度：CosineAnnealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        
        # 训练模型
        best_final_score = 0
        patience = 10  # 延长早停耐心值
        counter = 0
        
        # 新增：记录训练过程的Loss和指标，用于绘图
        train_losses = []
        val_losses = []
        train_gesture_accs = []
        val_gesture_accs = []
        train_target_accs = []
        val_target_accs = []
        
        for epoch in range(50):
            print(f"\n epoch {epoch+1}/50 | 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 训练阶段
            train_loss, train_gesture_acc, train_target_acc = train_epoch(
                model, train_loader, gesture_criterion, target_criterion, optimizer, device
            )
            
            # 验证
            (val_loss, val_gesture_acc, val_target_acc, binary_f1, 
             non_target_macro_f1, final_score, gesture_preds, 
             gesture_labels, target_preds, target_labels) = validate(
                model, val_loader, gesture_criterion, target_criterion, 
                device, target_gestures, gesture_le
            )
            
            # 记录指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_gesture_accs.append(train_gesture_acc)
            val_gesture_accs.append(val_gesture_acc)
            train_target_accs.append(train_target_acc)
            val_target_accs.append(val_target_acc)
            
            # 打印 epoch 结果
            print(f"训练损失: {train_loss:.4f}, 手势准确率: {train_gesture_acc:.4f}, 目标准确率: {train_target_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 手势准确率: {val_gesture_acc:.4f}, 目标准确率: {val_target_acc:.4f}")
            print(f"二元F1: {binary_f1:.4f}, 非目标宏F1: {non_target_macro_f1:.4f}")
            print(f"最终评分: {final_score:.4f}")
            
            # 学习率调度
            scheduler.step()
            
            # 早停机制：监控最终评分
            if final_score > best_final_score:
                best_final_score = final_score
                torch.save(model.state_dict(), f"best_model_fold_{fold+1}.pth")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 新增：绘制Loss-Epoch曲线
        plt.figure(figsize=(14, 6))
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title(f'Fold {fold+1} - Loss Curves', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_gesture_accs, label='Training Gesture Accuracy', color='blue')
        plt.plot(val_gesture_accs, label='Validation Gesture Accuracy', color='red')
        plt.plot(train_target_accs, label='Training Target Accuracy', color='green', linestyle='--')
        plt.plot(val_target_accs, label='Validation Target Accuracy', color='orange', linestyle='--')
        plt.title(f'Fold {fold+1} - Accuracy Curves', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f"best_model_fold_{fold+1}.pth"))
        
        # 再次验证
        (_, _, _, binary_f1, non_target_macro_f1, final_score, 
         gesture_preds, gesture_labels, target_preds, target_labels) = validate(
            model, val_loader, gesture_criterion, target_criterion, 
            device, target_gestures, gesture_le
        )
        
        print(f"\n折 {fold+1} 最佳结果:")
        print(f"二元F1: {binary_f1:.4f}, 非目标宏F1: {non_target_macro_f1:.4f}")
        print(f"最终评分: {final_score:.4f}")
        
        # 打印分类报告
        print("\n手势分类报告:")
        print(classification_report(gesture_labels, gesture_preds))
        
        print("\n二元分类报告:")
        print(classification_report(target_labels, target_preds))
        
        fold_results.append({
            'fold': fold+1,
            'binary_f1': binary_f1,
            'non_target_macro_f1': non_target_macro_f1,
            'final_score': final_score
        })
        
        # 测试集预测
        test_dataset = SensorSequenceDataset(test_sequences)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        model.eval()
        fold_test_gesture_preds = []
        fold_test_target_preds = []
        
        with torch.no_grad():
            for sequences in tqdm(test_loader, desc="测试集预测"):
                sequences = sequences.to(device)
                gesture_out, target_out = model(sequences)
                fold_test_gesture_preds.append(torch.softmax(gesture_out, dim=1).cpu().numpy())
                fold_test_target_preds.append(torch.softmax(target_out, dim=1).cpu().numpy())
        
        # 聚合测试集预测
        fold_test_gesture_preds = np.vstack(fold_test_gesture_preds)
        fold_test_target_preds = np.vstack(fold_test_target_preds)
        
        test_preds_gesture += fold_test_gesture_preds / n_splits
        test_preds_target += fold_test_target_preds / n_splits
    
    # 打印交叉验证汇总结果
    print("\n===== 交叉验证汇总 =====")
    results_df = pd.DataFrame(fold_results)
    print(results_df.mean())
    
    return test_preds_gesture, test_preds_target

# ----------------------
# 10. 生成提交文件
# ----------------------
def create_submission(test_preds_gesture, test_preds_target, test_ids, gesture_le, target_gestures):
    """生成符合竞赛要求的提交文件"""
    print("\n===== 生成提交文件 =====")
    
    # 从预测概率获取类别
    test_preds_gesture_class = np.argmax(test_preds_gesture, axis=1)
    test_preds_target_class = np.argmax(test_preds_target, axis=1)
    
    # 解码为原始手势名称
    gestures = gesture_le.inverse_transform(test_preds_gesture_class)
    
    # 生成提交文件，包含手势类别和是否为目标的预测
    submission = pd.DataFrame({
        'sequence_id': test_ids,
        'gesture': gestures,
        'is_target': test_preds_target_class  # 1表示目标，0表示非目标
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print(f"提交文件已生成，形状: {submission.shape}")
    print("前5行预览:")
    print(submission.head())
    
    return submission

# ----------------------
# 11. 主函数
# ----------------------
def main():
    try:
        # 加载数据并探索
        train, test, train_demo, test_demo = load_and_explore_data()
        
        # 预处理数据
        train, test, imu_cols, thm_cols, tof_cols, demo_cols, gesture_le, class_distribution, target_gestures = preprocess_data(
            train, test, train_demo, test_demo
        )
        
        # 自动确定合适的序列长度（核心优化：使用中位数而非1.5倍中位数）
        seq_lengths = train.groupby('sequence_id').size()
        print(f"\n序列长度分布: 平均={seq_lengths.mean():.1f}, 中位数={seq_lengths.median()}, 最大={seq_lengths.max()}")
        max_seq_length = int(seq_lengths.median())  # 直接使用中位数，减少30%序列长度
        print(f"自动设置最大序列长度为: {max_seq_length}")
        
        # 准备序列数据
        (train_sequences, train_gestures, train_is_target, train_ids,
         test_sequences, test_ids, n_features) = prepare_sequences(train, test, max_seq_length)
        
        # 清理内存
        del train, test
        gc.collect()
        
        # 获取手势类别数量
        num_gestures = len(np.unique(train_gestures))
        print(f"手势类别数量: {num_gestures}")
        
        # 如果特征数量为0，说明数据处理有问题
        if n_features == 0:
            raise ValueError("未找到任何特征列，请检查数据格式")
        
        # 训练模型并预测
        test_preds_gesture, test_preds_target = train_models(
            train_sequences, train_gestures, train_is_target, train_ids,
            test_sequences, n_features, num_gestures, 
            class_counts=class_distribution, target_gestures=target_gestures,
            gesture_le=gesture_le, n_splits=5
        )
        
        # 生成提交文件
        submission = create_submission(test_preds_gesture, test_preds_target, test_ids, gesture_le, target_gestures)
        
        print("\n所有任务完成!")
        
    except Exception as e:
        print(f"\n执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


