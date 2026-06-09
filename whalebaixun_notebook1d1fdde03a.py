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


# 学号与姓名
print("学号: 2024423320207, 姓名: 韩凯然")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 加载数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 字段验证
print("Train 字段:", train.columns.tolist())
print("Test 字段:", test.columns.tolist())


# 统一字段名（修正拼写问题）
train.columns = ['id', 'temperature', 'humidity', 'moisture', 'soil_type', 'crop_type', 'nitrogen', 'potassium', 'phosphorous', 'fertilizer']
test.columns = ['id', 'temperature', 'humidity', 'moisture', 'soil_type', 'crop_type', 'nitrogen', 'potassium', 'phosphorous']

# 合并数据集处理类别特征
combined = pd.concat([train[['soil_type', 'crop_type']], test[['soil_type', 'crop_type']]], axis=0)
combined = pd.get_dummies(combined, columns=['soil_type', 'crop_type'])  # One-Hot 编码

# 分离训练集和测试集
train_encoded = combined.iloc[:len(train)].copy()
test_encoded = combined.iloc[len(train):].copy()

# 合并数值特征
X = pd.concat([train[['temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous']], train_encoded], axis=1)
X_test = pd.concat([test[['temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous']], test_encoded], axis=1)

# 标签编码
le = LabelEncoder()
y = le.fit_transform(train['fertilizer'])

# 标准化数值特征
scaler = StandardScaler()
numeric_cols = ['temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous']
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


# 模型训练
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),  # 自动识别类别数量
    eval_metric='mlogloss',
    n_estimators=500,
    learning_rate=0.05,
    tree_method='hist'
)

# 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 验证集预测
val_preds = model.predict(X_val)
print("验证集准确率:", accuracy_score(y_val, val_preds))


# 获取概率矩阵
probs = model.predict_proba(X_test)

# 关键修改1：只取概率最高的3个索引（原代码是取5个）
top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # 取概率最高的3个，从高到低排序

# 逐行转换索引为标签
top3_labels = np.apply_along_axis(lambda row: le.inverse_transform(row), axis=1, arr=top3_indices)

# Step 8: 生成提交文件（确保只输出3个肥料名）
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]  # 这里直接使用top3_labels
})

# 保存到/kaggle/working/
submission.to_csv('/kaggle/working/submission.csv', index=False)

# 验证输出
print("生成的提交文件示例：")
print(submission.head())
print("\n文件已保存到 /kaggle/working/submission.csv")
!ls -lh /kaggle/working/submission.csv

