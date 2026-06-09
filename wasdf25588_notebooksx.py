import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import time

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 1. 生成模拟数据
print("===== 步骤1：生成模拟数据 =====")
np.random.seed(42)
n_samples = 5000

# 生成土壤特征
nitrogen = np.random.randint(0, 100, n_samples)
phosphorus = np.random.randint(0, 50, n_samples)
potassium = np.random.randint(0, 80, n_samples)
ph = np.round(np.random.uniform(4.0, 8.0, n_samples), 1)
temperature = np.round(np.random.uniform(10.0, 40.0, n_samples), 1)

# 肥料类型
fertilizer_types = ['NPK_20-20-20', 'Urea', 'DAP', 'MOP', 'SSP', 'CAN', 'Complex']

# 生成标签（基于简单规则）
labels = []
for i in range(n_samples):
    if nitrogen[i] < 30:
        labels.append(np.random.choice(['Urea', 'CAN'], p=[0.7, 0.3]))
    elif phosphorus[i] < 20:
        labels.append(np.random.choice(['DAP', 'SSP'], p=[0.7, 0.3]))
    elif potassium[i] < 25:
        labels.append(np.random.choice(['MOP', 'NPK_20-20-20'], p=[0.7, 0.3]))
    else:
        labels.append(np.random.choice(['NPK_20-20-20', 'Complex'], p=[0.6, 0.4]))

# 创建数据框
data = pd.DataFrame({
    'Nitrogen': nitrogen,
    'Phosphorus': phosphorus,
    'Potassium': potassium,
    'pH': ph,
    'Temperature': temperature,
    'Fertilizer': labels,
    'ID': range(n_samples)
})

# 分割训练集和测试集
train_df = data.sample(frac=0.8, random_state=42)
test_df = data.drop(train_df.index).drop('Fertilizer', axis=1)

# 重置索引，避免后续访问问题
test_df = test_df.reset_index(drop=True)

print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")
print("肥料类型分布:")
print(train_df['Fertilizer'].value_counts())

# 2. 数据预处理
print("\n===== 步骤2：数据预处理 =====")
# 特征和标签分离
X_train = train_df.drop(['Fertilizer', 'ID'], axis=1)
y_train = train_df['Fertilizer']
X_test = test_df.drop('ID', axis=1)
test_ids = test_df['ID']

# 编码标签
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
num_classes = len(label_encoder.classes_)
print(f"类别数量: {num_classes}, 类别列表: {label_encoder.classes_}")

# 3. 定义MAP@5评分函数
def map_at_5(y_true, y_pred_proba):
    score = 0.0
    for i in range(len(y_true)):
        true_label = y_true[i]
        # 获取前5个预测
        top5 = np.argsort(y_pred_proba[i])[::-1][:5]
        # 计算当前样本的AP
        precision = 0.0
        hits = 0
        for k, pred in enumerate(top5):
            if pred == true_label:
                hits += 1
                precision += hits / (k + 1)
        score += precision / max(hits, 1)  # 避免除以0
    return score / len(y_true)

# 4. 训练模型
print("\n===== 步骤3：模型训练 =====")
# 划分验证集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# 定义XGBoost模型
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# 训练模型
start_time = time.time()
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=100
)
print(f"训练时间: {time.time() - start_time:.2f}秒")

# 验证集评估
val_proba = model.predict_proba(X_val)
val_map5 = map_at_5(y_val, val_proba)
print(f"验证集MAP@5得分: {val_map5:.4f}")

# 5. 生成预测结果
print("\n===== 步骤4：生成预测结果 =====")
# 测试集预测概率
test_proba = model.predict_proba(X_test)

# 获取Top5预测
top5_indices = np.argsort(test_proba, axis=1)[:, ::-1][:, :5]

# 转换为原始类别名称
top5_predictions = []
for indices in top5_indices:
    top5_predictions.append(label_encoder.inverse_transform(indices))

# 生成提交文件
submission = pd.DataFrame({
    'ID': test_ids,
    'prediction': [' '.join(preds) for preds in top5_predictions]
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("提交文件已保存为 'submission.csv'")

# 显示前5个预测结果（使用iloc确保正确访问）
print("\n前5个预测结果示例:")
for i in range(min(5, len(submission))):
    print(f"ID {submission.iloc[i]['ID']}: {submission.iloc[i]['prediction']}")

# 6. 特征重要性
print("\n===== 步骤5：特征重要性 =====")
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', title='特征重要性')
plt.tight_layout()
plt.show()

print("\n所有步骤完成！")

