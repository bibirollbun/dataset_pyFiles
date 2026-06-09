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


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score

# 加载数据
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 查看数据集的列名
print("训练集列名：", train_data.columns)
print("测试集列名：", test_data.columns)

# 检查是否存在缺失值
print("训练集缺失值：\n", train_data.isnull().sum())
print("测试集缺失值：\n", test_data.isnull().sum())

# 处理缺失值
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)
test_data.fillna(test_data.mean(numeric_only=True), inplace=True)

# 数据预处理
# 提取特征
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']

# 对分类特征进行编码
label_encoders = {}

for column in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column].astype(str))
    test_data[column] = le.transform(test_data[column].astype(str))
    label_encoders[column] = le  # 保存编码器

# 特征归一化/标准化
scaler = StandardScaler()
train_data[features] = scaler.fit_transform(train_data[features])
test_data[features] = scaler.transform(test_data[features])

# 分离特征和目标变量
X = train_data[features]
y = train_data['Fertilizer Name']

# 定义随机森林模型
model = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=10, random_state=42, n_jobs=-1)

# 训练模型
model.fit(X, y)

# 使用交叉验证评估模型
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'交叉验证平均准确率: {np.mean(cv_scores):.4f}')

# 对测试集进行预测
y_test_pred = model.predict_proba(test_data[features])

# 获取每个测试样本的前3个最高概率的肥料名称
top_3_predictions = np.argsort(y_test_pred, axis=1)[:, -3:][:, ::-1]

# 将预测的索引转换回肥料名称
label_encoder = LabelEncoder()
label_encoder.fit(train_data['Fertilizer Name'])
predicted_fertilizer_names = label_encoder.inverse_transform(top_3_predictions.flatten())
predicted_fertilizer_names = predicted_fertilizer_names.reshape(-1, 3)

# 生成提交文件
submission = pd.DataFrame({'id': test_data['id']})
submission['Fertilizer Name'] = [' '.join(names) for names in predicted_fertilizer_names]

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("提交文件已生成，文件名为 'submission.csv'")


