import pandas as pd

# 加载CSV文件
data1 = pd.read_csv('/kaggle/input/playground-series-s3e26/test.csv')
data2 = pd.read_csv('/kaggle/input/cirrhosis-patient-survival-prediction/cirrhosis.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s3e26/test.csv')

# 合并两个数据集
data = pd.concat([data1, data2], ignore_index=True)

# 移除数据中的 ID 和 Status 列
data = data.drop(columns=['id', 'ID', 'Status'])

# 删除训练集中含有缺失数据的行
data = data.dropna()

# 将 Stage 列的值减去 1
data['Stage'] = data['Stage'] - 1

# 数据类型转换
data['Drug'] = data['Drug'].astype('category').cat.codes
data['Sex'] = data['Sex'].astype('category').cat.codes
data['Ascites'] = data['Ascites'].astype('category').cat.codes
data['Hepatomegaly'] = data['Hepatomegaly'].astype('category').cat.codes
data['Spiders'] = data['Spiders'].astype('category').cat.codes
data['Edema'] = data['Edema'].astype('category').cat.codes

# 保存处理后的数据
data.to_csv('processed_cirrhosis_data.csv', index=False)

# 查看处理后的数据
print(data.head())


data.to_csv('processed_cirrhosis_data.csv', index=False)


from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 特征和标签
X = data.drop('Stage', axis=1)
y = data['Stage']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 继续使用 LightGBM 进行模型训练
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

# 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'multiclass',  # 多分类任务
    'num_class': 4,  # 类别数量
    'metric': 'multi_logloss',  # 多分类的损失函数
    'boosting_type': 'gbdt',
    'num_leaves': 51,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}

# 训练模型
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=2000)

# 预测
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = [list(x).index(max(x)) for x in y_pred]  # 将概率转换为类别

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))


import lightgbm as lgb
import matplotlib.pyplot as plt

# 训练 LightGBM 模型
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# 获取特征重要性（基于 gain）
feature_importance = model.feature_importance(importance_type='gain')

# 获取特征名称
feature_names = model.feature_name()

# 将特征重要性和特征名称组合成一个 DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# 按重要性排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 打印特征重要性
print(importance_df)

