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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 加载数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 数据预处理
def preprocess(df):
    df = df.copy()
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day
        df['DayOfYear'] = df['Timestamp'].dt.dayofyear
        df = df.drop('Timestamp', axis=1)
    return df

train = preprocess(train)
test = preprocess(test)

# 确保训练集和测试集特征顺序一致
features = [col for col in train.columns if col not in ['id', 'Fertilizer Name']]
X = train[features]
y = train['Fertilizer Name']

# 处理分类特征
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

if categorical_cols:
    print(f"编码分类特征: {categorical_cols}")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
    test[categorical_cols] = encoder.transform(test[categorical_cols])

# 编码标签
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 特征缩放 - 确保使用相同的特征顺序
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 关键修复: 确保测试集特征顺序与训练集完全一致
test_features = test[features]
test_scaled = scaler.transform(test_features)

# 训练模型
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_scaled, y_encoded)

# 预测概率
probabilities = model.predict_proba(test_scaled)

# 获取top5预测结果
top5_indices = np.argsort(probabilities, axis=1)[:, -5:]
top5_predictions = []
for indices in top5_indices:
    top_fertilizers = le.inverse_transform(indices)[::-1]
    top5_predictions.append(" ".join(top_fertilizers))

# 创建提交文件
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': top5_predictions
})

# 保存结果
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

