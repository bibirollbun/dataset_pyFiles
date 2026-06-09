#学号: 2024423320112, 姓名: 林日城

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# 数据加载
print("正在加载数据...")
train_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
print("数据加载完成!")
print(f"训练数据形状: {train_data.shape}")
print(f"测试数据形状: {test_data.shape}\n")

# 数据预处理
print("开始数据预处理...")
# 编码类别特征
cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    print(f"正在编码类别特征: {col}")
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# 编码目标变量
print("正在编码目标变量...")
le_y = LabelEncoder()
train_data['Fertilizer Name'] = le_y.fit_transform(train_data['Fertilizer Name'])
print(f"目标变量类别数量: {len(le_y.classes_)}")
print("数据预处理完成!\n")

# 特征工程
print("开始特征工程...")
def add_features(df):
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    return df

train_data = add_features(train_data)
test_data = add_features(test_data)
print("添加的特征:")
print(" - N_P_ratio: 氮磷比")
print(" - N_K_ratio: 氮钾比")
print(" - Total_NPK: 总NPK含量")
print("特征工程完成!\n")

# 定义特征和目标
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
            'Nitrogen', 'Potassium', 'Phosphorous', 'N_P_ratio', 'N_K_ratio', 'Total_NPK']
target = 'Fertilizer Name'

X = train_data[features]
y = train_data[target]

# 划分训练验证集
print("划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}\n")

# 转换为lightgbm数据集格式
print("准备LightGBM数据集...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
print("数据集准备完成!\n")

# 训练LightGBM模型
print("开始训练LightGBM模型...")
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'random_state': 42
}

print("\n模型参数:")
for key, value in params.items():
    print(f"{key}: {value}")

model = lgb.train(params,
                 train_data,
                 num_boost_round=1000,
                 valid_sets=[val_data],
                 callbacks=[
                     lgb.early_stopping(stopping_rounds=50, verbose=True),
                     lgb.log_evaluation(50)
                 ])

print("\n模型训练完成!")
print(f"最佳迭代次数: {model.best_iteration}")

# 在测试集预测
print("\n开始在测试集上进行预测...")
test_probs = model.predict(test_data[features])
print(f"预测完成! 共生成 {len(test_probs)} 条预测结果")

# 获取top5预测
print("\n生成Top5预测结果...")
top5_preds = []
for i, probs in enumerate(test_probs):
    top5_idx = np.argsort(probs)[::-1][:5]
    top5_labels = le_y.inverse_transform(top5_idx)
    top5_preds.append(" ".join(top5_labels))
    
    # 显示前5条预测结果示例
    if i < 5:
        print(f"样本{i+1}预测结果: {top5_labels}")

# 生成提交文件
print("\n生成提交文件...")
submission = pd.DataFrame({
    'id': test_data['id'],
    'Fertilizer Name': top5_preds
})
submission.to_csv('submission.csv', index=False)
print("提交文件已保存为 submission.csv")

# 显示最终信息
print("\n=== 处理完成 ===")
print(f"总样本数: {len(submission)}")
print("前5条预测结果:")
print(submission.head())

