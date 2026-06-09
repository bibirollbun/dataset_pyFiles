# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据准备
data_path = '/kaggle/input/playground-series-s5e6/train.csv'
data = pd.read_csv(data_path)

# 查看数据的前几行，了解数据结构
print(data.head())

# 分离特征和标签
X = data.drop(columns=['Fertilizer Name'])
y = data['Fertilizer Name']

# 对标签进行编码（将类别标签转换为数值）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 对类别特征进行编码
categorical_features = ['Soil Type', 'Crop Type']
for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 2. 多分类模型训练
model = LGBMClassifier(
    objective='multiclass',
    metric='multi_logloss',
    n_estimators=500,
    learning_rate=0.05
)
model.fit(X_train, y_train)

# 3. 获取Top5预测
probabilities = model.predict_proba(X_test)
top5_predictions = np.argsort(probabilities, axis=1)[:, -5:][:, ::-1]
top5_predictions_labels = label_encoder.inverse_transform(top5_predictions.flatten()).reshape(-1, 5)

# 创建提交文件
submission = pd.DataFrame(top5_predictions_labels, columns=['Top1', 'Top2', 'Top3', 'Top4', 'Top5'])
submission.insert(0, 'SampleID', X_test.index)
submission.to_csv('/kaggle/working/fertilizer_predictions.csv', index=False)
print(submission.head())

# 4. 特征重要性可视化
feature_importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()

# 5. 预测结果可视化
sample_predictions = submission.head(10).copy()  # 创建一个副本以避免SettingWithCopyWarning
for col in ['Top1', 'Top2', 'Top3', 'Top4', 'Top5']:
    sample_predictions[col] = label_encoder.transform(sample_predictions[col])

# 确保热力图的数据是数值类型
sample_predictions = sample_predictions.set_index('SampleID').astype(int)  # 转换为整数类型

plt.figure(figsize=(10, 6))
sns.heatmap(sample_predictions, annot=True, cmap='viridis', fmt='d')
plt.title('Top 5 Predicted Fertilizers for First 10 Samples')
plt.show()

