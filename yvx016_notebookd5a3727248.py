#2024423320101 蔡芝漫
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 1. 数据加载与预处理
# 读取数据（请确保文件路径正确）
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_df = pd.read_csv('sample_submission.csv')

# 查看数据结构（可选，用于确认列名）
print("Train数据结构:", train_df.shape)
print("Test数据结构:", test_df.shape)
print("提交样例结构:", submission_df.shape)

# 分离特征与标签（Fertilizer Name为目标变量）
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
            'Nitrogen', 'Potassium', 'Phosphorous']
X = train_df[features]
y = train_df['Fertilizer Name']

# 处理分类特征：标签编码（适用于非数值特征）
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    # 保存编码器用于测试集转换
    if col == 'Soil Type':
        soil_encoder = le
    elif col == 'Crop Type':
        crop_encoder = le

# 处理测试集分类特征
for col in ['Soil Type', 'Crop Type']:
    if col in test_df.columns:
        if col == 'Soil Type':
            test_df[col] = soil_encoder.transform(test_df[col])
        elif col == 'Crop Type':
            test_df[col] = crop_encoder.transform(test_df[col])

# 划分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 构建并训练XGBoost模型
model = xgb.XGBClassifier(
    objective='multi:softprob',       # 多分类概率输出
    num_class=len(y.unique()),       # 自动获取类别数
    eval_metric='map@5',             # MAP@5评估指标
    n_estimators=500,                # 树数量
    learning_rate=0.05,               # 学习率
    max_depth=5,                     # 树深度，防止过拟合
    random_state=42,
    verbose=100                      # 每100轮输出日志
)

# 训练模型并启用早停
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True
)

# 3. 生成Top5预测并保存结果
# 获取测试集概率预测
test_probs = model.predict_proba(test_df)
# 对每个样本取概率最高的5个类别
top5_indices = np.argsort(-test_probs, axis=1)[:, :5]

# 映射回原始类别名（需确保model.classes_与原始标签一致）
top5_predictions = [model.classes_[indices] for indices in top5_indices]
# 转换为分号分隔的字符串
submission_df['Top5_Fertilizers'] = [';'.join(pred) for pred in top5_predictions]

# 保存提交文件
submission_df.to_csv('submission.csv', index=False)
print("预测完成，结果已保存至submission.csv")

