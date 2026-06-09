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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

# 加载数据
data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')

# 数据预处理
# 编码分类变量
label_encoders = {}
categorical_cols = ['Soil Type', 'Crop Type']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 编码目标变量
le_fertilizer = LabelEncoder()
data['Fertilizer Name'] = le_fertilizer.fit_transform(data['Fertilizer Name'])

# 特征和目标变量
X = data.drop(['id', 'Fertilizer Name'], axis=1)
y = data['Fertilizer Name']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
xgb_model = XGBClassifier(
    objective='multi:softprob',  # 多分类问题
    num_class=len(le_fertilizer.classes_),  # 类别数量
    n_estimators=200,           # 树的数量
    max_depth=6,                # 树的最大深度
    learning_rate=0.1,          # 学习率
    subsample=0.8,              # 样本采样比例
    colsample_bytree=0.8,       # 特征采样比例
    random_state=42,
    n_jobs=-1                  # 使用所有CPU核心
)

xgb_model.fit(X_train, y_train)

# 评估模型
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost模型准确率: {accuracy:.2f}")

# 获取预测概率
y_probs = xgb_model.predict_proba(X_test)

# 获取前5个最可能的肥料类型及其概率
def get_top5_fertilizers(probs, le):
    top5_indices = np.argsort(-probs)[:5]  # 按概率降序排列
    top5_fertilizers = le.inverse_transform(top5_indices)
    top5_probs = probs[top5_indices]
    return list(zip(top5_fertilizers, top5_probs))


# 函数：预测新样本的肥料类型
def predict_fertilizer_xgb(new_data):
    # 预处理新数据
    new_df = pd.DataFrame([new_data])
    
    # 编码分类变量
    for col in categorical_cols:
        if col in new_df.columns:
            # 处理未见过的类别
            new_df[col] = new_df[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
            new_df[col] = label_encoders[col].transform(new_df[col])
    
    # 确保列顺序一致
    new_df = new_df[X.columns]
    
    # 标准化
    new_scaled = scaler.transform(new_df)
    
    # 预测概率
    probs = xgb_model.predict_proba(new_scaled)[0]
    
    # 获取前5个肥料
    top5 = get_top5_fertilizers(probs, le_fertilizer)
    
    return top5

# 示例：预测一个新样本
sample_input = {
    'Temparature': 30,
    'Humidity': 60,
    'Moisture': 40,
    'Soil Type': 'Loamy',
    'Crop Type': 'Wheat',
    'Nitrogen': 25,
    'Potassium': 10,
    'Phosphorous': 15
}

print("\n新样本预测结果(XGBoost):")
top5_pred = predict_fertilizer_xgb(sample_input)
for i, (fert, prob) in enumerate(top5_pred, 1):
    print(f"{i}. {fert}: {prob:.4f}")

# 特征重要性分析
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importance')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()

