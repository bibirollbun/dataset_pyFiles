import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import psutil
import time
from scipy.sparse import issparse

# 检查系统资源
print(f"CPU核心数: {psutil.cpu_count()}")
print(f"可用内存: {psutil.virtual_memory().available / 1024**3:.2f} GB")

# ================ 1. 数据加载与基础配置 ================
try:
    # 实际 Kaggle 数据集路径
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
    print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")
except FileNotFoundError:
    print("错误：未找到数据集文件！请确保已在 Kaggle Notebook 中附加 'playground-series-s5e6' 数据集。")
    raise

# 特征与标签分离
X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']  # 原始标签（用于分类）
X_test = test_df.drop('id', axis=1)

# 类别编码器
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"肥料类别数量: {num_classes}")
print(f"类别映射: {dict(zip(range(num_classes), label_encoder.classes_))}")

# ================ 2. 优化版数据预处理 ================
# 精简特征工程，只保留最重要的特征
def add_feature_engineering(df):
    # 只计算关键特征，避免冗余计算
    if 'Nitrogen' in df.columns and 'Phosphorous' in df.columns:
        df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-8)
    if 'Nitrogen' in df.columns and 'Potassium' in df.columns:
        df['N_K_Ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-8)
    return df

X = add_feature_engineering(X)
X_test = add_feature_engineering(X_test)

# 特征分类（自动识别类别/数值特征）
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"数值特征({len(num_cols)}): {num_cols}")
print(f"类别特征({len(cat_cols)}): {cat_cols}")

# 优化预处理管道，减少内存占用
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))  # 使用稀疏输出
        ]), cat_cols)
    ],
    sparse_threshold=0.3  # 当密度低于0.3时使用稀疏格式
)

# 预处理执行
print("开始数据预处理...")
start_time = time.time()
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)
print(f"预处理完成，耗时: {time.time() - start_time:.2f}秒")
print(f"预处理后特征维度: {X_processed.shape[1]}")

# 修正稀疏矩阵密度计算逻辑
if issparse(X_processed):
    sparse_density = X_processed.getnnz() / (X_processed.shape[0] * X_processed.shape[1])
    print(f"稀疏矩阵密度: {sparse_density:.6f}")
else:
    # 如果是稠密矩阵，这里简单统计非零元素占比（可根据需求调整）
    total_elements = X_processed.shape[0] * X_processed.shape[1]
    non_zero_elements = np.count_nonzero(X_processed)
    sparse_density = non_zero_elements / total_elements
    print(f"稠密矩阵非零元素占比（类似稀疏密度）: {sparse_density:.6f}")

# ================ 3. 优化版模型训练 ================
# 分层拆分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_encoded, 
    test_size=0.2, 
    stratify=y_encoded, 
    random_state=42
)

print(f"训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}")

# 优化的LightGBM训练函数（平衡速度与精度）
def train_lightgbm(X, y, X_val, y_val):
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',  # 可尝试 'dart' 或 'goss' 以提高速度
        'learning_rate': 0.1,     # 增加学习率以减少迭代次数
        'num_leaves': 31,         # 减少叶子数量，降低模型复杂度
        'max_depth': 6,           # 限制树的深度
        'min_child_samples': 20,  # 减少过拟合
        'feature_fraction': 0.8,  # 每次迭代随机选择特征
        'bagging_fraction': 0.8,  # 启用样本采样
        'bagging_freq': 5,        # 每5次迭代进行一次采样
        'lambda_l1': 0.1,         # L1正则化
        'lambda_l2': 0.1,         # L2正则化
        'random_state': 42,
        'n_jobs': -1,             # 使用所有CPU核心
        'verbose': -1,
        'force_row_wise': True,   # 优化内存使用
    }

    # 构建LightGBM数据集（使用稀疏矩阵）
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)

   # 早停回调（增加预测频率以提前触发）
    early_stopping_callback = lgb.early_stopping(
        stopping_rounds=30,
        first_metric_only=True,
        verbose=True
    )

    # 进度条回调
    progress_callback = lgb.log_evaluation(period=100)
    
    print(f"开始训练LightGBM模型，参数: {params}")
    start_time = time.time()

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,  # 减少最大迭代次数
        valid_sets=[lgb_val],
        callbacks=[early_stopping_callback, progress_callback],
        keep_training_booster=True  # 保留中间模型
    )

    print(f"训练完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"最佳迭代次数: {model.best_iteration}")

    # 评估模型大小
    model_size = model.model_to_string().count('\n')
    print(f"模型大小: {model_size} 行")

    return model

# 训练模型
print("\n=== 开始训练 LightGBM 模型 ===")
model = train_lightgbm(X_train, y_train, X_val, y_val)

# ================ 4. 性能评估 ================
# 验证集预测速度测试
print("\n=== 评估模型预测速度 ===")
start_time = time.time()
y_val_probs = model.predict(X_val[:1000], num_iteration=model.best_iteration)
print(f"1000样本预测耗时: {time.time() - start_time:.4f}秒")

# 完整验证集评估
y_val_probs = model.predict(X_val, num_iteration=model.best_iteration)
val_acc = accuracy_score(y_val, np.argmax(y_val_probs, axis=1))

# 修正后的自定义 MAP@5 计算函数
def map_at5_score(y_true, y_pred_probs, label_encoder):
    y_true = label_encoder.inverse_transform(y_true)  # 还原原始标签
    # 取Top5概率索引
    top5_preds = np.argsort(y_pred_probs, axis=1)[:, ::-1][:, :5]  
    # 将top5_preds展平为一维数组进行逆变换
    top5_preds_flat = top5_preds.flatten()
    top5_labels_flat = label_encoder.inverse_transform(top5_preds_flat)
    # 还原为原形状 (样本数, 5)
    top5_labels = top5_labels_flat.reshape(-1, 5) 

    score = 0.0
    for true, preds in tqdm(zip(y_true, top5_labels), total=len(y_true), desc="计算 MAP@5"):
        true_set = {true}
        for k in range(1, 6):
            pred_k = preds[:k]
            precision = len(set(pred_k) & true_set) / k  # 计算前k个的准确率
            rel = 1 if true in pred_k else 0  # 正确则1，错误则0
            score += precision * rel
    return score / len(y_true)

val_map5 = map_at5_score(y_val, y_val_probs, label_encoder)

print(f"\n验证集准确率: {val_acc:.4f}")
print(f"验证集 MAP@5 分数: {val_map5:.4f}")

# 特征重要性分析
if hasattr(model, 'feature_importance'):
    # 获取特征名称（处理独热编码后的特征名）
    onehot_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    all_features = num_cols + list(onehot_features)
    
    importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': model.feature_importance()
    }).sort_values('Importance', ascending=False)

    print("\n=== 特征重要性 Top 10 ===")
    print(importance.head(10).to_string(index=False))

# ================ 5. 测试集预测与提交 ================
print("\n=== 开始测试集预测 ===")
start_time = time.time()
y_test_probs = model.predict(X_test_processed, num_iteration=model.best_iteration)
print(f"测试集预测耗时: {time.time() - start_time:.2f}秒")

# 生成 Top5 预测结果
top5_indices = np.argsort(y_test_probs, axis=1)[:, ::-1][:, :5]
# 展平后逆变换再还原形状
top5_indices_flat = top5_indices.flatten()
top5_labels_flat = label_encoder.inverse_transform(top5_indices_flat)
top5_labels = top5_labels_flat.reshape(-1, 5)

# 拼接结果文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'predicted_fertilizers': [' '.join(pred) for pred in top5_labels]
})

# 保存结果
submission_path = "submission_top5.csv"
submission.to_csv(submission_path, index=False)
print(f"预测结果已保存到: {submission_path}")
print(f"提交文件大小: {submission.shape[0]} 行")

# 显示预测示例
print("\n=== 预测示例 ===")
for i in range(5):
    print(f"样本 {i}: {submission.iloc[i]['predicted_fertilizers']}")

