!apt-get update -y
!apt-get install -y build-essential
!pip install ViennaRNA


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import logging
from tqdm.auto import tqdm
import warnings
from collections import Counter

warnings.filterwarnings('ignore')


# 设置日志记录
def setup_logging(log_file='rna_folding.log'):
    """配置日志系统"""
    logger = logging.getLogger('rna_folding')
    logger.setLevel(logging.INFO)

    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()

    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logging()


# 更新的TM-score计算函数
def calculate_tm_score(pred_coords, true_coords):
    """
    计算TM-score (Template Modeling Score)，按照准确公式实现

    TM-score = max⎛⎝⎜⎜1/Lref ∑i=1 to Lalign 1/(1+(di/d0)²)⎞⎠⎟⎟

    参数:
        pred_coords: 预测结构的坐标, 形状为(n_residues, 3)
        true_coords: 参考结构的坐标, 形状为(n_residues, 3)

    返回:
        tm_score: 0到1之间的分数
    """
    # 检查输入
    if pred_coords.shape != true_coords.shape:
        raise ValueError(f"预测和参考坐标必须有相同的形状。得到 {pred_coords.shape} 和 {true_coords.shape}")

    # 获取参考结构残基数量
    Lref = true_coords.shape[0]

    if Lref < 4:  # 太少的残基无法进行有意义的结构对比
        return 0.0

    # 计算距离归一化因子 d0 (Angstrom)
    if Lref >= 30:
        d0 = 0.6 * (Lref - 0.5) ** 0.5 - 2.5
    elif Lref >= 24:
        d0 = 0.7
    elif Lref >= 20:
        d0 = 0.6
    elif Lref >= 16:
        d0 = 0.5
    elif Lref >= 12:
        d0 = 0.4
    else:  # Lref < 12
        d0 = 0.3

    # 注意: 我们将尝试不同的旋转和平移来最大化TM-score
    # 首先将结构居中
    pred_center = np.mean(pred_coords, axis=0)
    true_center = np.mean(true_coords, axis=0)

    pred_centered = pred_coords - pred_center
    true_centered = true_coords - true_center

    best_tm_score = 0.0

    # 使用多个初始对齐来寻找全局最大值
    # 可以考虑使用不同的初始旋转
    num_rotations = 10  # 可以根据计算资源调整这个值

    # 从不同的随机旋转开始
    for _ in range(num_rotations):
        try:
            # 生成随机旋转
            if _ == 0:
                # 第一次迭代使用SVD进行初始对齐
                covariance = np.dot(pred_centered.T, true_centered)
                U, S, Vt = np.linalg.svd(covariance)
                # 确保旋转矩阵具有正确的手性
                if np.linalg.det(np.dot(U, Vt)) < 0:
                    U[:, -1] = -U[:, -1]
                rotation = np.dot(U, Vt)
            else:
                # 随后使用随机旋转作为初始点
                random_angles = np.random.uniform(0, 2 * np.pi, 3)
                rotation = Rotation.from_euler('xyz', random_angles).as_matrix()

            # 应用旋转
            pred_rotated = np.dot(pred_centered, rotation)

            # 进行精细优化，迭代改进对齐
            for _ in range(10):  # 迭代进行优化
                # 计算当前对齐下的distances
                distances = np.sqrt(np.sum((pred_rotated - true_centered) ** 2, axis=1))

                # 计算距离的权重 w = 1 / (1 + (di/d0)^2)
                weights = 1.0 / (1.0 + (distances / d0) ** 2)

                # 计算加权中心
                weighted_pred_sum = np.sum(weights[:, np.newaxis] * pred_centered, axis=0)
                weighted_true_sum = np.sum(weights[:, np.newaxis] * true_centered, axis=0)
                weight_sum = np.sum(weights)

                if weight_sum > 0:
                    weighted_pred_center = weighted_pred_sum / weight_sum
                    weighted_true_center = weighted_true_sum / weight_sum

                    pred_recentered = pred_centered - weighted_pred_center
                    true_recentered = true_centered - weighted_true_center

                    # 计算加权协方差矩阵
                    weighted_covariance = np.zeros((3, 3))
                    for i in range(Lref):
                        weighted_covariance += weights[i] * np.outer(pred_recentered[i], true_recentered[i])

                    U, S, Vt = np.linalg.svd(weighted_covariance)
                    # 确保旋转矩阵具有正确的手性
                    if np.linalg.det(np.dot(U, Vt)) < 0:
                        U[:, -1] = -U[:, -1]
                    rotation = np.dot(U, Vt)

                    # 应用新的旋转
                    pred_rotated = np.dot(pred_recentered, rotation) + weighted_true_center

                    # 计算新的距离
                    distances = np.sqrt(np.sum((pred_rotated - true_centered) ** 2, axis=1))

                    # 计算当前TM-score
                    tm_score = (1.0 / Lref) * np.sum(1.0 / (1.0 + (distances / d0) ** 2))

                    if tm_score > best_tm_score:
                        best_tm_score = tm_score

        except np.linalg.LinAlgError as e:
            logger.error(f"SVD计算失败: {e}")
            continue

    return best_tm_score


# 评估多个结构预测
def evaluate_tm_scores(predictions, ground_truth):
    """
    评估多个结构预测并返回每个目标的最佳TM分数

    参数:
        predictions: 包含预测的DataFrame
        ground_truth: 包含真值的DataFrame

    返回:
        字典，键为目标ID，值为最佳TM分数
    """
    # 提取唯一的目标ID
    target_ids = np.unique([id.split('_')[0] for id in predictions['ID']])

    tm_scores = {}
    for target_id in target_ids:
        # 获取该目标的预测和真值
        target_preds = predictions[predictions['ID'].str.startswith(target_id)]
        target_truth = ground_truth[ground_truth['ID'].str.startswith(target_id)]

        if len(target_preds) == 0 or len(target_truth) == 0:
            continue

        # 确保行顺序一致
        target_preds = target_preds.sort_values('resid')
        target_truth = target_truth.sort_values('resid')

        # 计算每个预测结构的TM分数
        struct_scores = []
        for i in range(1, 6):  # 5个预测结构
            try:
                pred_coords = target_preds[[f'x_{i}', f'y_{i}', f'z_{i}']].values
                true_coords = target_truth[['x_1', 'y_1', 'z_1']].values

                if pred_coords.shape == true_coords.shape:
                    tm_score = calculate_tm_score(pred_coords, true_coords)
                    struct_scores.append(tm_score)
            except Exception as e:
                logger.warning(f"计算{target_id}的结构{i}时出错: {str(e)}")

        if struct_scores:
            tm_scores[target_id] = max(struct_scores)

    return tm_scores


# 增强的特征工程 - 使用滑动窗口和更多序列特征
def enhanced_feat_eng(df):
    """创建增强的RNA序列特征，包括滑动窗口特征"""
    # 基本长度特征
    result = pd.DataFrame()

    # 确保target_id存在
    if 'target_id' not in df.columns:
        result['target_id'] = df.index
    else:
        result['target_id'] = df['target_id']

    # 计算序列长度
    result['seq_length'] = df['sequence'].str.len()

    # 计算单个核苷酸计数
    for base in ['A', 'C', 'U', 'G']:
        result[f'{base}_cnt'] = df['sequence'].str.count(base)
        # 添加百分比特征
        result[f'{base}_pct'] = result[f'{base}_cnt'] / result['seq_length']

    # 计算GC含量
    result['gc_content'] = (result['G_cnt'] + result['C_cnt']) / result['seq_length']
    result['au_content'] = (result['A_cnt'] + result['U_cnt']) / result['seq_length']
    result['gc_au_ratio'] = result['gc_content'] / result['au_content'].replace(0, 0.001)  # 避免除以零

    # 计算二核苷酸组合
    for base1 in ['A', 'C', 'U', 'G']:
        for base2 in ['A', 'C', 'U', 'G']:
            result[f'{base1}{base2}_cnt'] = df['sequence'].str.count(f'{base1}{base2}')
            # 添加标准化的二核苷酸频率
            result[f'{base1}{base2}_freq'] = result[f'{base1}{base2}_cnt'] / (result['seq_length'] - 1).clip(lower=1)

    # 计算三核苷酸的频率
    important_trimers = ['AAA', 'CCC', 'GGG', 'UUU', 'AUG', 'GCA', 'GUA', 'GUC', 'GUG', 'GUU',
                         'CAG', 'GGC', 'UCA', 'AGU', 'UCG', 'CUG', 'GAC']
    for trimer in important_trimers:
        result[f'{trimer}_freq'] = df['sequence'].apply(
            lambda x: x.count(trimer) / max(1, len(x) - 2) if len(x) > 0 else 0
        )

    # 计算序列复杂度指标 - 香农熵
    def shannon_entropy(seq):
        if not seq:
            return 0
        counts = Counter(seq)
        probs = [count / len(seq) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs)

    result['shannon_entropy'] = df['sequence'].apply(shannon_entropy)

    # 计算序列的长度特征
    result['seq_len_log'] = np.log1p(result['seq_length'])
    result['seq_len_sqrt'] = np.sqrt(result['seq_length'])

    return result


# 创建滑动窗口特征（针对残基级别）
def extract_sliding_window_features(df_seq, df_residues, window_size=5):
    """
    为每个残基创建滑动窗口特征

    参数:
        df_seq: 包含序列的DataFrame
        df_residues: 包含单个残基信息的DataFrame
        window_size: 窗口大小 (奇数)

    返回:
        包含窗口特征的DataFrame
    """
    if window_size % 2 == 0:
        window_size += 1  # 确保窗口大小是奇数

    half_window = window_size // 2
    result = pd.DataFrame()

    # 确保有目标ID列
    if 'target_id' not in df_residues.columns:
        if 'ID' in df_residues.columns:
            df_residues['target_id'] = df_residues['ID'].str.rsplit('_', n=1).str[0]
        else:
            df_residues['target_id'] = df_residues.index

    # 映射序列到目标ID
    target_to_seq = dict(zip(df_seq['target_id'], df_seq['sequence']))

    # 注意残基位置是从1开始的
    def get_window_features(row):
        target_id = row['target_id']
        resid = int(row['resid'])

        if target_id not in target_to_seq:
            return pd.Series({f'pos_{i}': 'X' for i in range(-half_window, half_window + 1)})

        seq = target_to_seq[target_id]
        if not seq:
            return pd.Series({f'pos_{i}': 'X' for i in range(-half_window, half_window + 1)})

        # 提取窗口
        features = {}
        for i in range(-half_window, half_window + 1):
            pos = resid - 1 + i  # 转为0-based索引
            if 0 <= pos < len(seq):
                features[f'pos_{i}'] = seq[pos]
            else:
                features[f'pos_{i}'] = 'X'  # 超出序列范围的填充

        # 添加相对位置特征
        seq_len = len(seq)
        features['rel_pos'] = resid / seq_len  # 相对位置 (0-1)
        features['rel_pos_sin'] = np.sin(2 * np.pi * resid / seq_len)  # 周期性位置特征
        features['rel_pos_cos'] = np.cos(2 * np.pi * resid / seq_len)

        # 添加局部序列特征
        window_start = max(0, resid - 1 - half_window)
        window_end = min(len(seq), resid + half_window)
        window_seq = seq[window_start:window_end]

        # 局部窗口的核苷酸组成
        for base in ['A', 'C', 'G', 'U']:
            features[f'window_{base}_cnt'] = window_seq.count(base)
            features[f'window_{base}_freq'] = features[f'window_{base}_cnt'] / len(window_seq) if window_seq else 0

        # 局部窗口的GC含量
        features['window_gc_content'] = (features['window_G_cnt'] + features['window_C_cnt']) / len(
            window_seq) if window_seq else 0

        # 添加当前位置距离序列两端的距离
        features['dist_to_start'] = resid - 1  # 距离开始
        features['dist_to_end'] = seq_len - resid  # 距离结束

        return pd.Series(features)

    # 应用到每一行
    window_features = df_residues.apply(get_window_features, axis=1)

    # 合并结果
    result = pd.concat([df_residues, window_features], axis=1)

    return result


# 解析目标ID
def extract_target_id(id_col):
    """从ID列提取目标ID"""
    return id_col.str.split('_').str[:2].str.join('_')


# 解析测试序列
def parse_test_sequences(test_df):
    """解析测试序列并创建提交格式"""
    result = []

    for _, row in test_df.iterrows():
        seq_length = len(row['sequence'])
        target_id = row['target_id']

        for i in range(seq_length):
            resname = row['sequence'][i]
            resid = i + 1
            result.append({
                'ID': f"{target_id}_{resid}",
                'resname': resname,
                'resid': resid
            })

    return pd.DataFrame(result)


# 应用物理约束的后处理
def apply_physical_constraints(coords, resnames, max_iterations=50):
    """
    应用物理约束到RNA结构坐标，专注于键长约束和冲突解决

    参数:
        coords: 形状为(n_residues, 3)的坐标数组
        resnames: 每个残基的类型 (A, C, G, U)
        max_iterations: 能量最小化的最大迭代次数

    返回:
        优化后的坐标
    """
    if len(coords) <= 2:
        return coords  # 残基太少，无法应用约束

    # 标准RNA骨架键长 (单位：埃)
    backbone_bond_lengths = {
        'A': 4.0,  # 腺嘌呤
        'C': 4.0,  # 胞嘧啶
        'G': 4.05,  # 鸟嘌呤
        'U': 4.0  # 尿嘧啶
    }

    # 创建结果坐标的副本
    optimized = coords.copy()

    # 迭代优化
    for iter_num in range(max_iterations):
        # 跟踪这次迭代中的总移动距离
        total_movement = 0.0

        # 第1步: 应用键长约束
        for i in range(1, len(coords)):
            res_type = resnames[i]
            ideal_length = backbone_bond_lengths.get(res_type, 4.0)

            # 获取与前一个残基的向量
            vec = optimized[i] - optimized[i - 1]
            current_length = np.linalg.norm(vec)

            if abs(current_length - ideal_length) > 0.1:  # 只有在显著偏离时才调整
                # 正则化并设置为理想长度
                vec = vec / current_length * ideal_length
                new_pos = optimized[i - 1] + vec

                # 计算移动距离
                movement = np.linalg.norm(new_pos - optimized[i])
                total_movement += movement

                # 更新位置
                optimized[i] = new_pos

        # 第2步: 检测并解决空间冲突
        # 计算所有残基对之间的距离
        distances = cdist(optimized, optimized)

        # 设置对角线元素为一个大值，避免检测自身残基
        np.fill_diagonal(distances, 999.0)

        # 设置最小允许距离
        min_allowed_distance = 1.5

        # 找到所有距离小于最小允许距离的残基对
        conflict_indices = np.where(distances < min_allowed_distance)

        for idx in range(len(conflict_indices[0])):
            i, j = conflict_indices[0][idx], conflict_indices[1][idx]

            # 只处理一次每一对 (i < j)
            if i >= j:
                continue

            # 如果是相邻残基，跳过(它们应该已经通过键长约束处理过)
            if abs(i - j) == 1:
                continue

            # 计算当前距离和需要移动的向量
            current_distance = distances[i, j]
            direction = optimized[j] - optimized[i]
            direction = direction / current_distance

            # 计算需要移动的距离
            move_distance = (min_allowed_distance - current_distance) * 0.5
            move_vec = direction * move_distance

            # 移动两个残基，使它们远离对方
            new_i = optimized[i] - move_vec
            new_j = optimized[j] + move_vec

            # 计算移动距离
            movement_i = np.linalg.norm(new_i - optimized[i])
            movement_j = np.linalg.norm(new_j - optimized[j])
            total_movement += movement_i + movement_j

            # 更新位置
            optimized[i] = new_i
            optimized[j] = new_j

        # 如果总移动距离小于阈值，认为优化已收敛
        if total_movement < 0.1:
            logger.info(f"结构优化在第{iter_num + 1}次迭代后收敛")
            break

        # 如果这是最后一次迭代，记录日志
        if iter_num == max_iterations - 1:
            logger.info(f"结构优化达到最大迭代次数({max_iterations})，最终移动距离: {total_movement:.3f}")

    return optimized


# 改进的后处理函数
def post_process_predictions(predictions):
    """对所有预测结构应用后处理，专注于物理约束"""
    result = predictions.copy()

    # 获取唯一目标ID
    target_ids = np.unique([id.rsplit('_', 1)[0] for id in predictions['ID']])

    for target_id in tqdm(target_ids, desc="后处理结构"):
        # 获取该目标的预测
        mask = predictions['ID'].str.startswith(target_id)
        target_preds = predictions[mask].sort_values('resid')

        # 提取残基名称
        resnames = target_preds['resname'].values

        # 对每个预测结构应用物理约束
        for i in range(1, 6):  # 5个预测结构
            coords = target_preds[[f'x_{i}', f'y_{i}', f'z_{i}']].values

            # 应用物理约束
            try:
                constrained_coords = apply_physical_constraints(coords, resnames)

                # 更新预测
                result.loc[mask, f'x_{i}'] = constrained_coords[:, 0]
                result.loc[mask, f'y_{i}'] = constrained_coords[:, 1]
                result.loc[mask, f'z_{i}'] = constrained_coords[:, 2]
            except Exception as e:
                logger.warning(f"处理{target_id}结构{i}时出错: {str(e)}")
                # 保留原始坐标

    return result


# 修复后的RNA结构预测器类，确保特征一致性
class RNAStructurePredictor:
    """管理RNA 3D结构预测工作流的类，包含残基特异性模型"""

    def __init__(self, model_dir="./models"):
        """初始化预测器和模型目录"""
        self.model_dir = model_dir
        self.models = {'A': {}, 'C': {}, 'G': {}, 'U': {}}  # 按残基类型分类的模型
        self.feature_columns = None
        self.categorical_columns = ['resname', 'begin_seq', 'end_seq'] + [f'pos_{i}' for i in range(-2, 3)]

        # 新增：存储每个分类特征的所有可能值
        self.categorical_values = {}

        # 新增：使用OneHotEncoder替代pd.get_dummies
        self.encoders = {}

        # 如果目录不存在则创建
        os.makedirs(model_dir, exist_ok=True)

    def load_data(self, train_path, labels_path):
        """加载并准备训练数据，添加滑动窗口特征"""
        logger.info(f"读取训练数据: {train_path}")
        train_sequences = pd.read_csv(train_path)

        logger.info(f"读取标签数据: {labels_path}")
        train_labels = pd.read_csv(labels_path)

        # 提取目标ID
        train_labels['target_id'] = extract_target_id(train_labels['ID'])

        # 创建基本特征
        logger.info("生成基本特征...")
        train_sequences_features = enhanced_feat_eng(train_sequences)

        # 合并基本数据
        train_data = pd.merge(train_labels, train_sequences_features, on='target_id', how='left')

        # 添加滑动窗口特征
        logger.info("生成滑动窗口特征...")
        train_data = extract_sliding_window_features(train_sequences, train_data)

        # 处理缺失值
        logger.info("处理缺失值...")
        for col in ['x_1', 'y_1', 'z_1']:
            # 按目标ID和resname分组计算均值
            group_means = train_data.groupby(['target_id', 'resname'])[col].transform('mean')
            # 用组均值填充NA值
            train_data[col] = train_data[col].fillna(group_means)

        # 删除剩余的NA
        train_data = train_data.dropna()
        logger.info(f"清理后的训练数据: {train_data.shape[0]}行, {train_data.shape[1]}列")

        return train_data

    def prepare_features(self, data, training=False):
        """
        准备建模特征，确保分类特征正确编码

        参数:
            data: 输入数据
            training: 是否处于训练阶段
        """
        # 确保窗口位置特征被填充
        for pos in range(-2, 3):
            pos_col = f'pos_{pos}'
            if pos_col not in data.columns:
                data[pos_col] = 'X'

        # 创建结构化特征列
        structured_data = data.copy()

        # 处理每个分类特征
        for col in self.categorical_columns:
            if col not in structured_data.columns:
                structured_data[col] = 'X'  # 使用默认值

            # 训练阶段：记录所有可能的值
            if training:
                # 存储该特征的所有唯一值
                self.categorical_values[col] = sorted(structured_data[col].unique().tolist())

                # 为每个分类特征创建一个OneHotEncoder
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                # 使用所有可能的值拟合编码器
                self.encoders[col].fit(np.array(self.categorical_values[col]).reshape(-1, 1))

            # 获取编码后的特征
            if col in self.encoders:
                # 对该列进行one-hot编码
                encoded = self.encoders[col].transform(structured_data[col].values.reshape(-1, 1))

                # 创建包含编码值的DataFrame
                feature_names = [f"{col}_{val}" for val in self.encoders[col].categories_[0]]
                encoded_df = pd.DataFrame(encoded, index=structured_data.index, columns=feature_names)

                # 删除原始列并添加编码列
                structured_data = pd.concat([structured_data.drop(col, axis=1), encoded_df], axis=1)

        # 识别特征列
        if self.feature_columns is None and training:
            self.feature_columns = [col for col in structured_data.columns
                                    if col not in ['ID', 'target_id', 'x_1', 'y_1', 'z_1']]

        # 确保所有特征列都存在
        for col in self.feature_columns or []:
            if col not in structured_data.columns:
                structured_data[col] = 0

        return structured_data[self.feature_columns] if self.feature_columns else structured_data

    def train(self, train_data, valid_fraction=0.2, optimize=False):
        """为每种残基类型训练单独的x, y, z坐标模型"""
        # 分离数据按残基类型
        residue_types = ['A', 'C', 'G', 'U']

        for res_type in residue_types:
            logger.info(f"\n=== 训练残基类型 {res_type} 的模型 ===")

            # 过滤该残基类型的数据
            res_data = train_data[train_data['resname'] == res_type]

            if len(res_data) == 0:
                logger.warning(f"没有找到残基类型 {res_type} 的数据，跳过训练")
                continue

            # 准备特征 - 传入training=True以记录分类特征值
            X = self.prepare_features(res_data, training=True)

            # 准备目标
            y_dict = {
                'x': res_data['x_1'],
                'y': res_data['y_1'],
                'z': res_data['z_1']
            }

            # 数据分割
            X_train, X_valid, y_train, y_valid = {}, {}, {}, {}
            for coord in ['x', 'y', 'z']:
                X_train[coord], X_valid[coord], y_train[coord], y_valid[coord] = train_test_split(
                    X, y_dict[coord], test_size=valid_fraction, random_state=42
                )

            # 如需要优化超参数
            if optimize:
                logger.info(f"为残基类型 {res_type} 执行超参数优化...")
                param_grid = {
                    'max_depth': [6, 10, 15],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'n_estimators': [300, 500, 700],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.7, 0.8, 0.9]
                }

                best_params = {}
                for coord in ['x', 'y', 'z']:
                    logger.info(f"优化残基类型 {res_type} 的 {coord.upper()} 坐标模型...")
                    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1
                    )
                    grid_search.fit(X_train[coord], y_train[coord])
                    best_params[coord] = grid_search.best_params_
                    logger.info(f"残基类型 {res_type} 的 {coord.upper()} 的最佳参数: {grid_search.best_params_}")
            else:
                # 使用默认参数，但为每种残基类型稍微调整
                best_params = {
                    'x': {
                        'max_depth': 12,
                        'learning_rate': 0.1,
                        'n_estimators': 500,
                        'subsample': 0.85,
                        'colsample_bytree': 0.85
                    },
                    'y': {
                        'max_depth': 12,
                        'learning_rate': 0.1,
                        'n_estimators': 500,
                        'subsample': 0.85,
                        'colsample_bytree': 0.85
                    },
                    'z': {
                        'max_depth': 12,
                        'learning_rate': 0.1,
                        'n_estimators': 500,
                        'subsample': 0.85,
                        'colsample_bytree': 0.85
                    }
                }

            # 训练每个坐标的独立模型
            for coord in ['x', 'y', 'z']:
                logger.info(f"\n训练残基类型 {res_type} 的 {coord.upper()} 坐标模型...")

                params = best_params[coord].copy()

                # 确保这些参数被设置
                for key in ['objective', 'random_state', 'n_jobs']:
                    if key not in params:
                        params[key] = {'objective': 'reg:squarederror',
                                       'random_state': 42,
                                       'n_jobs': -1}[key]

                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train[coord], y_train[coord],
                    eval_set=[(X_train[coord], y_train[coord]),
                              (X_valid[coord], y_valid[coord])],
                    eval_metric='rmse',
                    verbose=100
                )

                # 保存模型
                self.models[res_type][coord] = model
                model.save_model(f"{self.model_dir}/xgb_{res_type}_{coord}_model.json")

                # 保存特征名和分类值信息
                self._save_feature_info()

                # 打印特征重要性
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                logger.info(f"\n残基类型 {res_type} 的 {coord.upper()} 模型的前10个重要特征:")
                logger.info(feature_importance.head(10).to_string())

                # 评估模型
                valid_rmse = np.sqrt(mean_squared_error(y_valid[coord], model.predict(X_valid[coord])))
                logger.info(f"残基类型 {res_type} 的 {coord.upper()} 模型验证RMSE: {valid_rmse:.4f}")

    def _save_feature_info(self):
        """保存特征列名和分类特征值"""
        import json

        # 保存特征列名
        if self.feature_columns:
            with open(f"{self.model_dir}/feature_columns.json", 'w') as f:
                json.dump(self.feature_columns, f)

        # 保存分类特征值
        with open(f"{self.model_dir}/categorical_values.json", 'w') as f:
            json.dump(self.categorical_values, f)

    def load_models(self):
        """从磁盘加载训练好的残基特异性模型和特征信息"""
        import json

        # 加载特征列名
        try:
            with open(f"{self.model_dir}/feature_columns.json", 'r') as f:
                self.feature_columns = json.load(f)
        except FileNotFoundError:
            logger.warning("未找到特征列名文件")

        # 加载分类特征值
        try:
            with open(f"{self.model_dir}/categorical_values.json", 'r') as f:
                self.categorical_values = json.load(f)

            # 重新创建编码器
            for col, values in self.categorical_values.items():
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.encoders[col].fit(np.array(values).reshape(-1, 1))
        except FileNotFoundError:
            logger.warning("未找到分类特征值文件")

        # 加载模型
        for res_type in ['A', 'C', 'G', 'U']:
            for coord in ['x', 'y', 'z']:
                model_path = f"{self.model_dir}/xgb_{res_type}_{coord}_model.json"
                if os.path.exists(model_path):
                    model = xgb.XGBRegressor()
                    model.load_model(model_path)
                    self.models[res_type][coord] = model
                    logger.info(f"已加载残基类型 {res_type} 的 {coord.upper()} 坐标模型")
                else:
                    logger.warning(f"在{model_path}未找到残基类型 {res_type} 的 {coord.upper()} 坐标模型")

        # 检查是否所有模型都加载了
        loaded_count = sum([len(models) for models in self.models.values()])
        if loaded_count < 12:  # 4种残基类型 x 3个坐标 = 12个模型
            logger.warning(f"只加载了 {loaded_count}/12 个残基特异性模型")

    def predict(self, test_sequences, num_structures=5):
        """为测试序列生成预测，使用对应残基类型的模型"""
        logger.info("正在准备测试数据...")

        # 解析测试序列
        test_clean = parse_test_sequences(test_sequences)

        # 创建特征
        test_sequences_features = enhanced_feat_eng(test_sequences)

        # 合并数据
        test_data = pd.merge(test_clean, test_sequences_features,
                             left_on=test_clean['ID'].str.rsplit('_', n=1).str[0],
                             right_on='target_id',
                             how='left')

        # 添加滑动窗口特征
        test_data = extract_sliding_window_features(test_sequences, test_data)

        # 初始化预测数据框 - 使用DataFrame而不是字典+列表
        predictions = pd.DataFrame({
            'ID': test_data['ID'],
            'resname': test_data['resname'],
            'resid': test_data['resid']
        })

        # 初始化所有坐标列
        for i in range(1, num_structures + 1):
            for coord in ['x', 'y', 'z']:
                predictions[f'{coord}_{i}'] = 0.0

        logger.info("生成预测中...")

        # 按残基类型分组预测
        for res_type in ['A', 'C', 'G', 'U']:
            # 过滤该残基类型的数据
            res_mask = test_data['resname'] == res_type
            if not res_mask.any():
                continue

            # 准备特征 - 确保使用一致的特征集
            X_test_res = self.prepare_features(test_data[res_mask], training=False)

            # 确保所有所需的模型都存在
            if not all(coord in self.models[res_type] for coord in ['x', 'y', 'z']):
                logger.warning(f"残基类型 {res_type} 缺少一些坐标模型，使用其他残基模型作为后备")
                # 找到有所有坐标模型的残基类型作为后备
                backup_res = next(
                    (r for r in ['A', 'C', 'G', 'U'] if all(coord in self.models[r] for coord in ['x', 'y', 'z'])),
                    None)
                if backup_res:
                    logger.info(f"使用残基类型 {backup_res} 的模型作为 {res_type} 的后备")
                    self.models[res_type] = self.models[backup_res]
                else:
                    raise ValueError("没有可用的完整残基模型集")

            # 为每个结构生成预测
            for i in range(1, num_structures + 1):
                for coord in ['x', 'y', 'z']:
                    # 使用残基特异性模型进行预测
                    if coord in self.models[res_type]:
                        # 使用DataFrame的loc索引方式更新值
                        predictions.loc[res_mask, f'{coord}_{i}'] = self.models[res_type][coord].predict(X_test_res)
                    else:
                        raise ValueError(f"残基类型 {res_type} 的 {coord} 坐标模型未加载")

        # 确保所有预测列为float类型
        for i in range(1, num_structures + 1):
            for coord in ['x', 'y', 'z']:
                col = f'{coord}_{i}'
                predictions[col] = predictions[col].astype(float)

        return predictions

    def evaluate(self, predictions, ground_truth):
        """使用TM-score评估预测"""
        # 按目标ID分组
        tm_scores = evaluate_tm_scores(predictions, ground_truth)

        # 计算TM-score统计信息
        if tm_scores:
            scores_array = np.array(list(tm_scores.values()))
            avg_tm_score = np.mean(scores_array)
            median_tm_score = np.median(scores_array)
            min_tm_score = np.min(scores_array)
            max_tm_score = np.max(scores_array)

            logger.info(f"TM-score统计信息:")
            logger.info(f"平均值: {avg_tm_score:.4f}")
            logger.info(f"中位数: {median_tm_score:.4f}")
            logger.info(f"最小值: {min_tm_score:.4f}")
            logger.info(f"最大值: {max_tm_score:.4f}")

            # 为有价值的分析创建直方图
            plt.figure(figsize=(10, 6))
            plt.hist(scores_array, bins=20, alpha=0.7)
            plt.title('目标TM-score分布')
            plt.xlabel('TM-score')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            plt.savefig('tm_score_distribution.png')
            plt.close()

            return {
                'avg': avg_tm_score,
                'median': median_tm_score,
                'min': min_tm_score,
                'max': max_tm_score,
                'scores': tm_scores
            }
        else:
            logger.warning("无法计算有效的TM-score")
            return None


# 主函数
def main():
    """主执行函数"""
    logger.info("开始RNA 3D结构预测")

    # 数据路径
    data_dir = '/kaggle/input/stanford-rna-3d-folding'
    model_dir = './models'

    # 初始化预测器
    predictor = RNAStructurePredictor(model_dir=model_dir)

    # 尝试从数据目录读取数据文件
    try:
        # 检查是否需要训练模型
        train_model = True

        # 检查是否有任何残基类型的模型已经存在
        for res_type in ['A', 'C', 'G', 'U']:
            if all(os.path.exists(f"{model_dir}/xgb_{res_type}_{coord}_model.json") for coord in ['x', 'y', 'z']):
                train_model = False
                break

        if train_model:
            logger.info("正在加载训练数据...")
            train_data = predictor.load_data(
                f"{data_dir}/train_sequences.csv",
                f"{data_dir}/train_labels.csv"
            )

            logger.info("正在训练残基特异性模型...")
            predictor.train(train_data, optimize=False)
        else:
            logger.info("加载预训练的残基特异性模型...")
            predictor.load_models()

        # 检查验证数据
        val_exists = os.path.exists(f"{data_dir}/validation_sequences.csv") and \
                     os.path.exists(f"{data_dir}/validation_labels.csv")

        if val_exists:
            logger.info("加载验证数据...")
            validation_sequences = pd.read_csv(f"{data_dir}/validation_sequences.csv")
            validation_labels = pd.read_csv(f"{data_dir}/validation_labels.csv")

            logger.info("在验证集上评估模型...")
            val_predictions = predictor.predict(validation_sequences)

            # 使用改进的后处理
            logger.info("对验证预测应用物理约束...")
            val_predictions = post_process_predictions(val_predictions)

            predictor.evaluate(val_predictions, validation_labels)

        # 检查测试数据
        test_exists = os.path.exists(f"{data_dir}/test_sequences.csv")

        if test_exists:
            logger.info("加载测试数据...")
            test_sequences = pd.read_csv(f"{data_dir}/test_sequences.csv")

            logger.info("生成测试集预测...")
            predictions = predictor.predict(test_sequences)

            logger.info("应用物理约束后处理...")
            predictions = post_process_predictions(predictions)

            # 加载样本提交以确保顺序一致
            if os.path.exists(f"{data_dir}/sample_submission.csv"):
                sample_submission = pd.read_csv(f"{data_dir}/sample_submission.csv")
                sample_submission['sort_order'] = range(len(sample_submission))

                # 合并并根据sort_order排序
                predictions = pd.merge(
                    predictions,
                    sample_submission[['ID', 'sort_order']],
                    on='ID',
                    how='left'
                )
                predictions = predictions.sort_values('sort_order').drop('sort_order', axis=1)

            # 保存预测
            predictions.to_csv('submission.csv', index=False)
            logger.info("预测保存至submission.csv")

    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}", exc_info=True)

    logger.info("完成")


if __name__ == "__main__":
    main()

