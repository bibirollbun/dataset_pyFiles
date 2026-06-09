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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 学号和姓名
print("学号: 2024423310126, 姓名: 谢浩文")

# 读取数据
df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
print("数据基本信息：")
df.info()

# 查看数据样本
print("\n数据前5行：")
print(df.head())

# 统计肥料类型分布（仅显示前五）
fertilizer_counts = df['Fertilizer Name'].value_counts().head(5)
print("\n肥料类型分布（前五）：")
print(fertilizer_counts)

# 特征与目标变量分离
X = df.drop(['id', 'Fertilizer Name', 'Crop Type'], axis=1)
y = df['Fertilizer Name']

# 处理分类变量
categorical_cols = ['Soil Type']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 标准化数值特征
numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标签编码
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 训练OneVsRest多分类模型
base_model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)

model = OneVsRestClassifier(base_model)
model.fit(X_train, y_train_encoded)

# 预测测试集
y_pred_proba = model.predict_proba(X_test)

# 获取每个样本的前5个预测结果
def get_top_k_predictions(probabilities, k=5):
    top_indices = np.argsort(-probabilities, axis=1)[:, :k]
    top_predictions = []
    
    for indices in top_indices:
        preds = [(le.inverse_transform([idx])[0], probabilities[i, idx]) for i, idx in enumerate(indices)]
        top_predictions.append(preds)
    
    return top_predictions

top5_predictions = get_top_k_predictions(y_pred_proba)

# 手动实现MAP@5计算
def calculate_map_at_5(y_true, y_pred_top5):
    map_score = 0.0
    
    for i, (true_label, predictions) in enumerate(zip(y_true, y_pred_top5)):
        predicted_labels = [pred[0] for pred in predictions]
        precision_sum = 0.0
        correct_count = 0
        
        for k in range(1, 6):
            if k > len(predicted_labels):
                break
                
            if predicted_labels[k-1] == true_label:
                correct_count += 1
                precision_sum += correct_count / k
        
        if correct_count > 0:
            average_precision = precision_sum / min(5, len(predicted_labels))
            map_score += average_precision
    
    return map_score / len(y_true)

# 计算MAP@5分数
map_score = calculate_map_at_5(y_test, top5_predictions)
print(f"\nMAP@5分数: {map_score:.4f}")

# 计算准确率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"准确率: {accuracy:.4f}")

# 分析预测结果中出现频率最高的前5种肥料
predicted_fertilizers = [pred[0][0] for pred in top5_predictions]
predicted_counts = pd.Series(predicted_fertilizers).value_counts().head(5)

# 绘制预测结果中最常出现的前5种肥料
plt.figure(figsize=(10, 6))
sns.barplot(x=predicted_counts.values, y=predicted_counts.index)
plt.title('Top 5 Most Predicted Fertilizers')
plt.xlabel('Prediction Count')
plt.tight_layout()
plt.savefig('top_5_fertilizers.png')
plt.show()

# 保存预测结果
prediction_df = pd.DataFrame({
    'id': df['id'].iloc[len(y_train):],
    'Fertilizer_1': [pred[0][0] for pred in top5_predictions],
    'Fertilizer_2': [pred[1][0] for pred in top5_predictions],
    'Fertilizer_3': [pred[2][0] for pred in top5_predictions],
    'Fertilizer_4': [pred[3][0] for pred in top5_predictions],
    'Fertilizer_5': [pred[4][0] for pred in top5_predictions]
})

# 保存为CSV
prediction_df.to_csv('fertilizer_predictions.csv', index=False)
print("\n预测结果已保存至fertilizer_predictions.csv")

