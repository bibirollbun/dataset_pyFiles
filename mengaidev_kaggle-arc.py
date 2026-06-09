import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from scipy.ndimage import label
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader

# 设置随机种子确保可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# --- 数据加载策略 ---
def find_data_dir():
    """自动检测Kaggle数据目录或本地数据目录"""
    kaggle_base = '/kaggle/input'
    expected_files = {
        'arc-agi_evaluation_challenges.json',
        'arc-agi_evaluation_solutions.json',
        'arc-agi_training_solutions.json'
    }
    if os.path.exists(kaggle_base):
        for folder in os.listdir(kaggle_base):
            full_path = os.path.join(kaggle_base, folder)
            if os.path.isdir(full_path):
                files = set(os.listdir(full_path))
                if expected_files.issubset(files):
                    print(f"Found Kaggle dataset folder: {full_path}")
                    return full_path
    print("Kaggle dataset folder not found, falling back to './data'")
    return './data'

def load_arc_data(challenge_file: str, solution_file: Optional[str] = None) -> Tuple[Dict, Optional[Dict]]:
    """加载ARC挑战数据和解决方案"""
    with open(challenge_file, 'r') as f:
        challenges = json.load(f)
    solutions = None
    if solution_file and os.path.exists(solution_file):
        with open(solution_file, 'r') as f:
            solutions = json.load(f)
    return challenges, solutions

# --- 网格处理工具 ---
def grid_to_numpy(grid: List[List[int]]) -> np.ndarray:
    """将网格转换为NumPy数组"""
    return np.array(grid, dtype=np.int8)

def numpy_to_grid(array: np.ndarray) -> List[List[int]]:
    """将NumPy数组转换回网格格式"""
    return array.tolist()

def resize_grid(grid: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """调整网格大小"""
    old_h, old_w = grid.shape
    new_h, new_w = new_shape
    row_ratio = old_h / new_h
    col_ratio = old_w / new_w
    resized = np.zeros((new_h, new_w), dtype=grid.dtype)
    for r in range(new_h):
        for c in range(new_w):
            src_r = min(int(r * row_ratio), old_h - 1)
            src_c = min(int(c * col_ratio), old_w - 1)
            resized[r, c] = grid[src_r, src_c]
    return resized

def pad_grid(grid: List[List[int]], max_size: int = 30) -> List[List[int]]:
    """填充网格到标准大小"""
    grid = grid_to_numpy(grid)
    h, w = grid.shape
    # 计算填充量
    pad_h = max(0, max_size - h)
    pad_w = max(0, max_size - w)
    # 在右侧和底部添加填充
    padded = np.pad(grid, ((0, pad_h), (0, pad_w)), 
                   mode='constant', constant_values=0)
    return padded.tolist()

# --- 特征提取和相似度计算 ---
def extract_features(grid: np.ndarray) -> np.ndarray:
    """提取网格特征"""
    features = []
    h, w = grid.shape
    features.extend([h, w, h*w])
    # 颜色统计特征
    unique_colors, counts = np.unique(grid, return_counts=True)
    features.append(len(unique_colors))
    # 颜色直方图
    hist = np.zeros(10)
    for color, count in zip(unique_colors, counts):
        if 0 <= color < 10:
            hist[color] = count
    features.extend(hist / (h * w + 1e-5))
    # 对象特征
    num_objects = 0
    total_object_area = 0
    if grid.any():
        labeled_array, num_features = label(grid > 0)
        if num_features > 0:
            num_objects = num_features
            object_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
            total_object_area = sum(object_sizes)
            features.extend([np.mean(object_sizes), np.std(object_sizes), max(object_sizes)])
        else:
            features.extend([0, 0, 0])
    else:
        features.extend([0, 0, 0])
    features.append(num_objects)
    features.append(total_object_area / (h * w + 1e-5))
    return np.array(features, dtype=np.float32)

def combined_similarity(g1: np.ndarray, g2: np.ndarray) -> float:
    """计算网格综合相似度"""
    feat1 = extract_features(g1)
    feat2 = extract_features(g2)
    # 特征嵌入相似度
    sim_embed = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0, 0]
    # 颜色直方图相似度
    max_color = max(g1.max(), g2.max(), 9)
    hist1, _ = np.histogram(g1, bins=np.arange(max_color+2))
    hist2, _ = np.histogram(g2, bins=np.arange(max_color+2))
    hist1 = hist1 / (hist1.sum() + 1e-5)
    hist2 = hist2 / (hist2.sum() + 1e-5)
    sim_hist = np.dot(hist1, hist2) / (np.linalg.norm(hist1)*np.linalg.norm(hist2) + 1e-5)
    return 0.7 * sim_embed + 0.3 * sim_hist

# --- 变换操作 ---
def rotate(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """旋转网格"""
    return np.rot90(grid, k=k)

def flip(grid: np.ndarray, direction: str = 'h') -> np.ndarray:
    """翻转网格"""
    return np.fliplr(grid) if direction == 'h' else np.flipud(grid)

def translate(grid: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """平移网格"""
    shifted = np.zeros_like(grid)
    h, w = grid.shape
    # 计算源区域和目标区域
    x_start_src = max(0, -dx)
    x_end_src = w - max(0, dx)
    y_start_src = max(0, -dy)
    y_end_src = h - max(0, dy)
    x_start_dst = max(0, dx)
    x_end_dst = w - max(0, -dx)
    y_start_dst = max(0, dy)
    y_end_dst = h - max(0, -dy)
    # 执行平移
    shifted[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = grid[y_start_src:y_end_src, x_start_src:x_end_src]
    return shifted

def scale(grid: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """缩放网格"""
    if factor == 1.0:
        return grid.copy()
    h, w = grid.shape
    new_h = max(1, int(h * factor))
    new_w = max(1, int(w * factor))
    return resize_grid(grid, (new_h, new_w))

# --- 神经符号模型 ---
class NeuroSymbolicModel(nn.Module):
    """神经符号混合模型"""
    def __init__(self, max_grid_size: int = 30, max_colors: int = 10, num_primitives: int = 20):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.max_colors = max_colors
        self.num_primitives = num_primitives
        # 卷积特征提取器
        self.encoder = nn.Sequential(
            nn.Conv2d(max_colors, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        # 规则预测头
        self.rule_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_primitives)
        )
        # 参数预测头
        self.param_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # 最多4个参数
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.encoder(x)
        features_flat = features.view(features.size(0), -1)
        rule_logits = self.rule_head(features_flat)
        params = self.param_head(features_flat)
        return rule_logits, params
    
    def grid_to_tensor(self, grid: List[List[int]]) -> torch.Tensor:
        """
        将网格转换为模型输入张量
        返回形状: [max_colors, max_grid_size, max_grid_size]
        """
        grid = grid_to_numpy(grid)
        tensor = np.zeros((self.max_colors, self.max_grid_size, self.max_grid_size))
        for c in range(self.max_colors):
            tensor[c] = (grid == c).astype(np.float32)
        return torch.tensor(tensor, dtype=torch.float32)

# --- 自定义数据集 ---
class ARCTrainDataset(Dataset):
    """ARC训练数据集"""
    def __init__(self, training_data: Dict, max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
        self.samples = []
        
        # 支持的变换类型
        self.primitive_names = [
            'rotate90', 'rotate180', 'rotate270', 
            'flip_h', 'flip_v', 'color_shift',
            'identity'
        ]
        
        # 收集训练样本
        for task_id, task in training_data.items():
            for pair in task['train']:
                input_grid = pad_grid(pair['input'], self.max_grid_size)
                output_grid = pad_grid(pair['output'], self.max_grid_size)
                
                # 识别变换
                input_np = grid_to_numpy(input_grid)
                output_np = grid_to_numpy(output_grid)
                transform_type, params = self.identify_transformation(input_np, output_np)
                
                # 只保留我们支持的变换类型
                if transform_type in self.primitive_names:
                    self.samples.append({
                        'input': input_grid,
                        'transform_type': transform_type,
                        'params': params
                    })
        
        print(f"Dataset initialized with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_tensor = self.grid_to_tensor(sample['input'])
        
        # 规则目标
        rule_idx = self.primitive_names.index(sample['transform_type'])
        
        # 参数目标 (填充到固定长度)
        params = sample['params'] + [0.0] * (4 - len(sample['params']))
        
        return input_tensor, rule_idx, np.array(params, dtype=np.float32)
    
    def grid_to_tensor(self, grid: List[List[int]]) -> torch.Tensor:
        """将网格转换为模型输入张量"""
        grid = grid_to_numpy(grid)
        tensor = np.zeros((10, self.max_grid_size, self.max_grid_size))
        for c in range(10):
            tensor[c] = (grid == c).astype(np.float32)
        return torch.tensor(tensor, dtype=torch.float32)
    
    def identify_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[str, List[float]]:
        """
        识别从输入网格到输出网格的变换
        
        返回: (变换类型, 参数列表)
        """
        # 检查是否相同 (identity)
        if np.array_equal(input_grid, output_grid):
            return 'identity', []
        
        # 检查旋转
        if np.array_equal(output_grid, rotate(input_grid, 1)):
            return 'rotate90', []
        if np.array_equal(output_grid, rotate(input_grid, 2)):
            return 'rotate180', []
        if np.array_equal(output_grid, rotate(input_grid, 3)):
            return 'rotate270', []
        
        # 检查水平翻转
        if np.array_equal(output_grid, flip(input_grid, 'h')):
            return 'flip_h', []
        
        # 检查垂直翻转
        if np.array_equal(output_grid, flip(input_grid, 'v')):
            return 'flip_v', []
        
        # 检查颜色变换
        unique_in = np.unique(input_grid)
        unique_out = np.unique(output_grid)
        
        # 检查是否为简单的颜色偏移
        if len(unique_in) == len(unique_out) and len(unique_in) > 0:
            # 计算可能的颜色映射
            color_map = {}
            consistent = True
            
            for i in range(len(unique_in)):
                diff = unique_out[i] - unique_in[i]
                if i > 0 and diff != color_map.get('diff', diff):
                    consistent = False
                    break
                color_map[unique_in[i]] = unique_out[i]
                color_map['diff'] = diff
            
            if consistent:
                # 验证映射是否一致
                transformed = np.zeros_like(input_grid)
                for r in range(input_grid.shape[0]):
                    for c in range(input_grid.shape[1]):
                        transformed[r, c] = color_map.get(input_grid[r, c], input_grid[r, c])
                
                if np.array_equal(transformed, output_grid):
                    return 'color_shift', [color_map['diff']]
        
        # 默认返回identity (虽然不完全匹配，但比unknown好)
        return 'identity', []

# --- ARC求解器 ---
class HybridARCSolver:
    """混合ARC求解器（神经符号+变换匹配）"""
    def __init__(self, max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
        self.primitive_names = [
            'rotate90', 'rotate180', 'rotate270', 
            'flip_h', 'flip_v', 'color_shift',
            'identity'
        ]
        self.model = NeuroSymbolicModel(
            max_grid_size, 
            num_primitives=len(self.primitive_names)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 初始化时设为评估模式
    
    def train(self, training_data: Dict, num_epochs: int = 20, batch_size: int = 16, lr: float = 1e-3):
        """训练神经符号模型"""
        # 创建数据集和数据加载器
        dataset = ARCTrainDataset(training_data, self.max_grid_size)
        
        if len(dataset) == 0:
            print("Warning: No valid training samples found. Using default model.")
            return
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # 设置模型为训练模式
        self.model.train()
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        rule_criterion = nn.CrossEntropyLoss()
        param_criterion = nn.MSELoss()
        
        # 训练循环
        for epoch in range(num_epochs):
            total_loss = 0
            total_rule_loss = 0
            total_param_loss = 0
            
            for inputs, rule_targets, param_targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # 将数据移动到设备
                inputs = inputs.to(self.device)
                rule_targets = rule_targets.to(self.device)
                param_targets = param_targets.to(self.device)
                
                # 前向传播
                rule_logits, predicted_params = self.model(inputs)
                
                # 计算损失
                rule_loss = rule_criterion(rule_logits, rule_targets)
                param_loss = param_criterion(predicted_params, param_targets)
                
                # 总损失 (加权组合)
                loss = 0.7 * rule_loss + 0.3 * param_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 记录损失
                total_loss += loss.item()
                total_rule_loss += rule_loss.item()
                total_param_loss += param_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_rule_loss = total_rule_loss / len(dataloader)
            avg_param_loss = total_param_loss / len(dataloader)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f} "
                  f"(Rule: {avg_rule_loss:.4f}, Param: {avg_param_loss:.4f})")
        
        # 保存训练好的模型
        torch.save(self.model.state_dict(), 'neuro_symbolic_model.pth')
        print("Model saved to 'neuro_symbolic_model.pth'")
        self.model.eval()  # 训练完成后设回评估模式
    
    def load_model(self, model_path: str = 'neuro_symbolic_model.pth'):
        """加载训练好的模型"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file {model_path} not found")
            return False
    
    def solve_task(self, task: Dict[str, Any]) -> List[List[List[int]]]:
        """解决单个ARC任务"""
        # 预处理训练对
        train_pairs = []
        for pair in task['train']:
            input_grid = pad_grid(pair['input'], self.max_grid_size)
            output_grid = pad_grid(pair['output'], self.max_grid_size)
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # 处理测试输入
        test_inputs = []
        for test in task['test']:
            input_grid = pad_grid(test['input'], self.max_grid_size)
            test_inputs.append({
                'input': input_grid
            })
        
        # 生成预测
        predictions = []
        for test in test_inputs:
            input_grid = test['input']
            # 尝试1：神经符号方法
            try:
                attempt1 = self.neuro_symbolic_solution(input_grid, train_pairs)
            except Exception as e:
                print(f"Neuro-symbolic solution failed: {str(e)}")
                attempt1 = self.transform_based_solution(input_grid, train_pairs)
            
            # 尝试2：变换匹配方法
            attempt2 = self.transform_based_solution(input_grid, train_pairs)
            
            # 选择最佳结果
            best_attempt = self.select_best_attempt(input_grid, attempt1, attempt2, train_pairs)
            predictions.append(best_attempt)
        
        return predictions
    
    def select_best_attempt(self, input_grid: List[List[int]], 
                           attempt1: List[List[int]], attempt2: List[List[int]],
                           train_pairs: List[Dict[str, Any]]) -> List[List[int]]:
        """选择最佳预测结果"""
        input_np = grid_to_numpy(input_grid)
        attempt1_np = grid_to_numpy(attempt1)
        attempt2_np = grid_to_numpy(attempt2)
        
        # 评估每个尝试与训练对的匹配度
        score1 = self.evaluate_attempt(attempt1_np, train_pairs)
        score2 = self.evaluate_attempt(attempt2_np, train_pairs)
        
        # 选择得分更高的尝试
        if score1 >= score2:
            return attempt1
        else:
            return attempt2
    
    def evaluate_attempt(self, output_grid: np.ndarray, 
                        train_pairs: List[Dict[str, Any]]) -> float:
        """评估输出网格与训练对的匹配度"""
        total_score = 0
        count = 0
        
        for train in train_pairs:
            train_out = grid_to_numpy(train['output'])
            # 计算与训练输出的相似度
            sim = combined_similarity(output_grid, train_out)
            total_score += sim
            count += 1
        
        return total_score / count if count > 0 else 0
    
    def neuro_symbolic_solution(self, input_grid: List[List[int]], 
                              train_pairs: List[Dict[str, Any]]) -> List[List[int]]:
        """神经符号解决方案"""
        try:
            # 将网格转换为张量 [channels, height, width]
            tensor = self.model.grid_to_tensor(input_grid)
            # 添加batch维度 [1, channels, height, width]
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # 使用模型预测
            with torch.no_grad():
                rule_logits, params = self.model(tensor)
            
            # 获取最高置信度的规则
            rule_probs = torch.softmax(rule_logits, dim=1)
            top_rule_idx = torch.argmax(rule_probs).item()
            rule_name = self.primitive_names[top_rule_idx]
            
            # 应用规则
            grid_np = grid_to_numpy(input_grid)
            if rule_name == 'rotate90':
                result = rotate(grid_np, 1)
            elif rule_name == 'rotate180':
                result = rotate(grid_np, 2)
            elif rule_name == 'rotate270':
                result = rotate(grid_np, 3)
            elif rule_name == 'flip_h':
                result = flip(grid_np, 'h')
            elif rule_name == 'flip_v':
                result = flip(grid_np, 'v')
            elif rule_name == 'color_shift':
                shift = int(params[0][0].item() % 10)
                if shift == 0:
                    shift = 1  # 避免除零
                result = (grid_np + shift) % 10
            else:  # identity
                result = grid_np
            
            # 移除填充
            return self.remove_padding(result.tolist(), train_pairs)
        except Exception as e:
            print(f"Neuro-symbolic solution failed: {str(e)}")
            # 如果神经符号方法失败，使用简单回退
            return self.transform_based_solution(input_grid, train_pairs)
    
    def transform_based_solution(self, input_grid: List[List[int]], 
                                train_pairs: List[Dict[str, Any]]) -> List[List[int]]:
        """基于变换的解决方案"""
        test_in = grid_to_numpy(input_grid)
        candidates = []
        
        # 生成候选变换
        for flip_dir in ['none', 'h', 'v']:
            base = test_in.copy()
            if flip_dir != 'none':
                base = flip(base, flip_dir)
            for rot in range(4):
                r = rotate(base, rot)
                candidates.append(r)
        
        # 评估候选变换
        best_sim = -1
        best_output = None
        for cand in candidates:
            for train in train_pairs:
                train_in = grid_to_numpy(train['input'])
                train_out = grid_to_numpy(train['output'])
                sim_in = combined_similarity(cand, train_in)
                sim_out = combined_similarity(cand, train_out)
                sim = max(sim_in, sim_out)
                if sim > best_sim:
                    best_sim = sim
                    best_output = train_out
        
        # 返回最佳匹配输出
        if best_output is not None:
            return self.remove_padding(best_output.tolist(), train_pairs)
        
        # 如果没有找到匹配，尝试更复杂的分析
        return self.pattern_based_solution(input_grid, train_pairs)
    
    def pattern_based_solution(self, input_grid: List[List[int]], 
                              train_pairs: List[Dict[str, Any]]) -> List[List[int]]:
        """基于模式的解决方案（更复杂的逻辑）"""
        # 这里可以添加更高级的模式识别和规则归纳
        # 例如：对象识别、关系推理等
        
        # 作为简单示例，返回输入网格
        return input_grid
    
    def remove_padding(self, grid: List[List[int]], train_pairs: List[Dict[str, Any]]) -> List[List[int]]:
        """移除填充，恢复原始网格大小"""
        # 尝试从训练对中确定原始尺寸
        original_height, original_width = None, None
        
        for pair in train_pairs:
            orig_in = pair['input']
            orig_out = pair['output']
            
            # 使用输出尺寸作为参考
            if original_height is None or original_width is None:
                original_height, original_width = len(orig_out), len(orig_out[0])
        
        # 如果无法确定原始尺寸，使用输入尺寸
        if original_height is None or original_width is None:
            original_height, original_width = len(input_grid), len(input_grid[0])
        
        # 裁剪到原始尺寸
        grid_np = grid_to_numpy(grid)
        return grid_np[:original_height, :original_width].tolist()

# --- 主程序 ---
def main():
    # 自动检测数据目录
    DATA_DIR = find_data_dir()
    print(f"Using data directory: {DATA_DIR}")
    print("Files in directory:", os.listdir(DATA_DIR))
    
    # 构建文件路径
    TRAIN_CHALLENGES = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')
    TRAIN_SOLUTIONS = os.path.join(DATA_DIR, 'arc-agi_training_solutions.json')
    EVAL_CHALLENGES = os.path.join(DATA_DIR, 'arc-agi_evaluation_challenges.json')
    TEST_CHALLENGES = os.path.join(DATA_DIR, 'arc-agi_test_challenges.json')
    
    # 初始化求解器
    solver = HybridARCSolver()
    
    # 尝试加载预训练模型
    if solver.load_model():
        print("Using pre-trained model")
    else:
        # 如果模型不存在，检查是否有训练数据
        if os.path.exists(TRAIN_CHALLENGES) and os.path.exists(TRAIN_SOLUTIONS):
            print("Loading training data for model training...")
            with open(TRAIN_CHALLENGES, 'r') as f:
                training_challenges = json.load(f)
            
            # 训练模型
            print("Training neural-symbolic model...")
            solver.train(training_challenges)
        else:
            print("Training data not found. Using default model.")
    
    # 加载测试挑战数据
    print("Loading test challenges...")
    with open(TEST_CHALLENGES, 'r') as f:
        test_challenges = json.load(f)
    
    # 生成提交
    submission = {}
    print("Solving ARC tasks...")
    for task_id, task_data in tqdm(test_challenges.items(), desc="Processing tasks"):
        predictions = solver.solve_task(task_data)
        # ARC要求的提交格式：每个任务一个列表，每个测试输入一个输出网格
        submission[task_id] = predictions
    
    # 保存提交文件
    output_path = 'submission.json'
    print(f"Saving submission to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    print("Done! Submission file saved.")

if __name__ == "__main__":
    main()

