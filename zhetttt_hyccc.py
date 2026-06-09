import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 数据加载与预处理
# 检查文件是否存在
train_file = '/kaggle/input/playground-series-s5e6/train.csv'
test_file = '/kaggle/input/playground-series-s5e6/test.csv'

if not os.path.exists(train_file) or not os.path.exists(test_file):
    raise FileNotFoundError(f"请确保文件 {train_file} 和 {test_file} 存在于指定路径中。")

# 加载数据
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# 查看数据基本信息
print(train_data.info())
print(train_data.describe())
print(train_data.head())

# 检查列名
print("列名:", train_data.columns)

# 确保目标列名正确
# 假设目标列名为 'Fertilizer'，如果列名不同，请根据实际情况修改
target_column = 'Fertilizer'
if target_column not in train_data.columns:
    # 如果目标列名不存在，尝试找到类似的列名
    similar_columns = [col for col in train_data.columns if 'Fertilizer' in col]
    if similar_columns:
        target_column = similar_columns[0]
        print(f"目标列名已更正为: {target_column}")
    else:
        raise KeyError(f"目标列 '{target_column}' 不存在于数据集中。请检查数据集结构。")

# 处理缺失值
# 检查缺失值情况
print(train_data.isnull().sum())

# 对于数值特征，用中位数填充缺失值
numerical_features = train_data.select_dtypes(include=['float64', 'int64']).columns
train_data[numerical_features] = train_data[numerical_features].fillna(train_data[numerical_features].median())

# 对于类别特征，用众数填充缺失值
categorical_features = train_data.select_dtypes(include=['object']).columns
for feature in categorical_features:
    train_data[feature] = train_data[feature].fillna(train_data[feature].mode()[0])

# 特征工程
# 对类别特征进行编码
label_encoder = LabelEncoder()
for feature in categorical_features:
    train_data[feature] = label_encoder.fit_transform(train_data[feature])

# 特征标准化
scaler = StandardScaler()
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])

# 模型训练与预测
# 分离特征和目标变量
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 多分类模型训练
model = LGBMClassifier(
    objective='multiclass',     # 多分类任务
    metric='multi_logloss',     # 多分类损失
    n_estimators=500,
    learning_rate=0.05
)
model.fit(X_train, y_train)

# 获取预测概率
probabilities = model.predict_proba(X_val)

# 获取每个样本的Top5预测
top5_predictions = np.argsort(probabilities, axis=1)[:, -5:][:, ::-1]

# 性能评估
# 计算MAP@5分数
def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

# 计算MAP@5分数
actual = [[label] for label in y_val]
print(f'MAP@5: {mapk(actual, top5_predictions)}')

# 绘制混淆矩阵
# 获取预测结果
y_pred = model.predict(X_val)

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 特征重要性分析
# 获取特征重要性
feature_importances = model.feature_importances_

# 绘制特征重要性图
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=X.columns)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Analysis')
plt.show()

