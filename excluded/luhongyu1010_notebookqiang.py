import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import time
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 记录开始时间
start_time = time.time()

# 1. 高效数据读取
print("开始读取数据...")
# 仅读取必要的列，减少内存使用
cols_to_use = ['id', 'Temparature', 'Humidity', 'Moisture', 'Soil Type', 
              'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer Name']

train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', usecols=cols_to_use)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', usecols=cols_to_use[:-1])

print(f"数据读取完成，耗时: {time.time()-start_time:.2f}秒")
print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")

# 2. 数据预处理
print("\n开始数据预处理...")
preprocess_start = time.time()

# 修正列名拼写错误
train_df = train_df.rename(columns={'Temparature': 'Temperature'})
test_df = test_df.rename(columns={'Temparature': 'Temperature'})

# 3. 关键特征工程（聚焦肥料预测）
print("\n进行关键特征工程...")
def add_core_features(df):
    # 营养比例特征（对肥料选择最重要）
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    
    # 环境条件综合指数（温度、湿度、水分）
    df['Env_Index'] = (df['Temperature'] + df['Humidity'] + df['Moisture']) / 3
    
    # 土壤-作物组合特征（捕捉交互作用）
    df['Soil_Crop_Combo'] = df['Soil Type'].astype(str) + "_" + df['Crop Type'].astype(str)
    
    return df

train_df = add_core_features(train_df)
test_df = add_core_features(test_df)

# 4. 高效类别特征编码
print("\n进行类别特征编码...")
# 土壤类型编码
soil_le = LabelEncoder()
soil_le.fit(pd.concat([train_df['Soil Type'], test_df['Soil Type']]).astype(str))
train_df['Soil Type'] = soil_le.transform(train_df['Soil Type'].astype(str))
test_df['Soil Type'] = soil_le.transform(test_df['Soil Type'].astype(str))

# 作物类型编码
crop_le = LabelEncoder()
crop_le.fit(pd.concat([train_df['Crop Type'], test_df['Crop Type']]).astype(str))
train_df['Crop Type'] = crop_le.transform(train_df['Crop Type'].astype(str))
test_df['Crop Type'] = crop_le.transform(test_df['Crop Type'].astype(str))

# 土壤-作物组合编码
soil_crop_le = LabelEncoder()
soil_crop_le.fit(pd.concat([train_df['Soil_Crop_Combo'], test_df['Soil_Crop_Combo']]).astype(str))
train_df['Soil_Crop_Combo'] = soil_crop_le.transform(train_df['Soil_Crop_Combo'].astype(str))
test_df['Soil_Crop_Combo'] = soil_crop_le.transform(test_df['Soil_Crop_Combo'].astype(str))

print(f"数据预处理完成，耗时: {time.time()-preprocess_start:.2f}秒")

# 5. 准备训练数据
X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']

# 编码目标变量
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n肥料类别数量: {len(label_encoder.classes_)}")
print(f"特征数量: {X.shape[1]}")

# 6. 划分训练集和验证集
print("\n划分训练集和验证集...")
# 使用较小验证集比例加速评估
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")

# 7. 构建高效预测模型
print("\n开始训练预测模型...")
model_start = time.time()

# 使用高效参数配置
model = LGBMClassifier(
    objective='multiclass',
    num_class=len(label_encoder.classes_),
    n_estimators=100,       # 减少树的数量
    learning_rate=0.15,      # 提高学习率
    num_leaves=31,           # 合理叶子数量
    max_depth=6,             # 适当深度
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,               # 使用所有CPU核心
    verbose=-1
)

# 训练模型
model.fit(X_train, y_train)
print(f"模型训练完成，耗时: {time.time()-model_start:.2f}秒")

# 8. 预测验证集（仅用于快速检查）
val_probs = model.predict_proba(X_val)

# 9. 生成测试集预测排序
print("\n生成测试集肥料预测排序...")
predict_start = time.time()

X_test = test_df.drop('id', axis=1)
test_probs = model.predict_proba(X_test)

# 高效生成top-5预测函数
def generate_top5_predictions(probs, encoder):
    """
    生成前5种最可能的肥料类型，按可能性从高到低排序
    
    参数:
    probs - 概率矩阵 (n_samples, n_classes)
    encoder - 肥料标签编码器
    
    返回:
    预测字符串列表，每个字符串包含5种肥料名称（空格分隔）
    """
    # 获取top-5预测索引（按概率降序）
    top5_indices = np.argsort(-probs, axis=1)[:, :5]
    
    # 批量转换标签
    all_labels = encoder.inverse_transform(top5_indices.reshape(-1))
    
    # 重塑为(n_samples, 5)并连接为字符串
    return [' '.join(all_labels[i:i+5]) for i in range(0, len(all_labels), 5)]

# 生成预测
test_df['Fertilizer Name'] = generate_top5_predictions(test_probs, label_encoder)

# 创建提交文件
submission = test_df[['id', 'Fertilizer Name']]
submission.to_csv('submission.csv', index=False)

print(f"预测生成完成，耗时: {time.time()-predict_start:.2f}秒")
print("\n提交文件已创建: submission.csv")
print("\n前5个样本的预测结果:")
print(submission.head().to_string(index=False))

# 10. 最终报告
total_time = time.time() - start_time
print(f"\n{'='*50}")
print(f"总耗时: {total_time:.2f}秒")
print(f"肥料类别数量: {len(label_encoder.classes_)}")
print(f"特征数量: {X.shape[1]}")
print("任务完成! 已为每个样本预测前5种最可能的肥料类型，并按可能性排序。")




