# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re

# 设置随机种子确保结果可复现
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =====================
# 自定义评估指标 MAP@5
# =====================
def map5_score(y_true, y_pred):
    """计算Mean Average Precision @ 5 (MAP@5)"""
    top5 = np.argsort(-y_pred, axis=1)[:, :5]
    
    ap_scores = []
    for i in range(len(y_true)):
        actual = y_true[i]
        predicted = top5[i]
        
        ap = 0.0
        correct = 0
        for k in range(5):
            if predicted[k] == actual:
                correct += 1
                ap += correct / (k + 1)
        
        ap_scores.append(ap / min(correct, 1) if correct > 0 else 0.0)
    
    return np.mean(ap_scores)

# =====================
# 数据准备
# =====================
print("正在读取数据...")
# 读取数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# 显示实际列名
print("\n训练集实际列名:", train.columns.tolist())

# 创建列名映射字典
column_mapping = {}
for col in train.columns:
    # 简化列名：移除特殊字符、空格，转换为小写
    simplified = re.sub(r'[^a-zA-Z0-9]', '', col).lower()
    
    # 识别关键特征
    if 'nitrogen' in simplified or 'n' == simplified:
        column_mapping['N'] = col
    elif 'phosphorus' in simplified or 'p' == simplified:
        column_mapping['P'] = col
    elif 'potassium' in simplified or 'k' == simplified:
        column_mapping['K'] = col
    elif 'temp' in simplified:
        column_mapping['temperature'] = col
    elif 'humid' in simplified:
        column_mapping['humidity'] = col
    elif 'ph' in simplified:
        column_mapping['ph'] = col
    elif 'rain' in simplified:
        column_mapping['rainfall'] = col
    elif 'fertilizer' in simplified:
        column_mapping['fertilizer'] = col

# 确保所有必要列都有映射
required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'fertilizer']
for col in required_columns:
    if col not in column_mapping:
        # 尝试基于位置或常见别名
        if col == 'N':
            column_mapping['N'] = next((c for c in train.columns if 'nitrogen' in c.lower() or 'n' == c.lower()), 'N')
        elif col == 'P':
            column_mapping['P'] = next((c for c in train.columns if 'phosphorus' in c.lower() or 'p' == c.lower()), 'P')
        elif col == 'K':
            column_mapping['K'] = next((c for c in train.columns if 'potassium' in c.lower() or 'k' == c.lower()), 'K')
        else:
            column_mapping[col] = col  # 使用标准名称作为后备

print("\n列名映射:")
for standard, actual in column_mapping.items():
    print(f"{standard} -> {actual}")

# 重命名列以标准化
train = train.rename(columns={v: k for k, v in column_mapping.items() if k in required_columns})
test = test.rename(columns={v: k for k, v in column_mapping.items() if k in required_columns and v in test.columns})

# 确保所有必要列都存在
for col in required_columns:
    if col not in train.columns:
        train[col] = 0  # 添加缺失列（仅用于演示，实际应处理）
    if col not in test.columns:
        test[col] = 0

print("\n标准化后列名:", train.columns.tolist())

# 特征工程
print("\n正在进行特征工程...")
# 添加特征：营养元素平衡比
train['NP_ratio'] = train['N'] / (train['P'] + 1e-5)
train['NK_ratio'] = train['N'] / (train['K'] + 1e-5)
train['PK_ratio'] = train['P'] / (train['K'] + 1e-5)
train['N+P+K'] = train['N'] + train['P'] + train['K']
train['temp_humidity'] = train['temperature'] * train['humidity']

test['NP_ratio'] = test['N'] / (test['P'] + 1e-5)
test['NK_ratio'] = test['N'] / (test['K'] + 1e-5)
test['PK_ratio'] = test['P'] / (test['K'] + 1e-5)
test['N+P+K'] = test['N'] + test['P'] + test['K']
test['temp_humidity'] = test['temperature'] * test['humidity']

# 编码目标变量
print("编码目标变量...")
le = LabelEncoder()
train['fertilizer_encoded'] = le.fit_transform(train['fertilizer'])

# 准备特征和目标
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
            'NP_ratio', 'NK_ratio', 'PK_ratio', 'N+P+K', 'temp_humidity']
print("最终使用特征:", features)

# 划分训练集和验证集
print("划分训练集和验证集...")
X = train[features]
y = train['fertilizer_encoded']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]} 样本")
print(f"验证集大小: {X_val.shape[0]} 样本")

# =====================
# 模型训练 - XGBoost
# =====================
print("\n开始训练XGBoost模型...")
start_time = time.time()

# 初始化XGBoost分类器
model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(le.classes_),
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=100,
    random_state=RANDOM_STATE
)

# 训练模型
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

training_time = time.time() - start_time
print(f"\n模型训练完成! 耗时: {training_time:.2f}秒")

# =====================
# 模型评估
# =====================
print("\n评估模型性能...")
# 在验证集上预测
val_probs = model.predict_proba(X_val)
val_map5 = map5_score(y_val.values, val_probs)
print(f"验证集 MAP@5: {val_map5:.5f}")

# =====================
# 特征重要性分析
# =====================
print("\n分析特征重要性...")
# 获取特征重要性
importance = model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title('特征重要性分析', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# =====================
# 测试集预测与提交
# =====================
print("\n生成测试集预测...")
# 在测试集上预测
test_probs = model.predict_proba(test[features])

# 获取每个样本的前5个预测
top5_preds = np.argsort(-test_probs, axis=1)[:, :5]

# 将数字标签转换为原始肥料名称
top5_fertilizers = []
for preds in top5_preds:
    top5_fertilizers.append(' '.join(le.inverse_transform(preds)))

# 创建提交文件
submission['fertilizer'] = top5_fertilizers
submission.to_csv('submission.csv', index=False)

print("\n提交文件已生成!")
print("前5个样本的预测结果:")
print(submission.head())

print("\n所有处理完成! 请下载提交文件: submission.csv")

