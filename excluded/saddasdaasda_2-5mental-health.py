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

# 导入数据
data1 = pd.read_csv('/kaggle/input/analyze-the-insights-over-mental-health-data/train.csv')

# 转化为数据框
df1 = pd.DataFrame(data1)






df1





import pandas as pd

# 导入数据
data2 = pd.read_csv('/kaggle/input/analyze-the-insights-over-mental-health-data/test.csv')

# 转化为数据框
df2 = pd.DataFrame(data2)




df2


df1=df1.drop('id',axis=1)
ID=df2['id']
df2=df2.drop('id',axis=1)
ID


df1=df1.drop('Name',axis=1)
df2=df2.drop('Name',axis=1)


# 排除目标列 'target'
features_df = df1.drop(columns=['Depression'])

# 筛选分类型特征名称列表
category = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

# 筛选数值型特征名称列表
numeric = features_df.select_dtypes(exclude=['object', 'category']).columns.tolist()
df1['Depression'] = df1['Depression'].astype('category')


df1.dtypes


df2.dtypes


df1plus=df1.drop('Depression',axis=1)
a=df1plus.dtypes.to_list()
b=df2.dtypes.to_list()


a==b


columns=df2.columns.to_list()
columns


for i in columns:
    iunique1=df1[i].unique().tolist()
    iunique2=df2[i].unique().tolist()
    print(i,iunique1==iunique2)


for i in columns:
    iunique1 = set(df1[i].unique())
    iunique2 = set(df2[i].unique())
    # 找出 df1 中有但 df2 中没有的值
    diff_1_2 = iunique1 - iunique2
    # 找出 df2 中有但 df1 中没有的值
    diff_2_1 = iunique2 - iunique1

    print(f"列名: {i}")
    print(f"df1 中有但 df2 中没有的唯一值: {diff_1_2}")
    print(f"df2 中有但 df1 中没有的唯一值: {diff_2_1}")
    print()


df1['Gender'].unique()


df2['Gender'].unique()


df1['City'].unique()


df2['City'].unique()


columns


import pandas as pd

# 获取 df2 特征列的唯一值
for i in columns:
    unique_values = df2[i].unique()
    df1 = df1[df1[i].isin(unique_values)]
   




for i in columns:
    iunique1 = set(df1[i].unique())
    iunique2 = set(df2[i].unique())
    # 找出 df1 中有但 df2 中没有的值
    diff_1_2 = iunique1 - iunique2
    # 找出 df2 中有但 df1 中没有的值
    diff_2_1 = iunique2 - iunique1

    print(f"列名: {i}")
    print(f"df1 中有但 df2 中没有的唯一值: {diff_1_2}")
    print(f"df2 中有但 df1 中没有的唯一值: {diff_2_1}")
    print()


df1


for i in columns:
    print(i,len(df1[i].unique()))


df1['Depression'].isnull().sum()


df1['Academic Pressure']=df1['Academic Pressure'].fillna(0)
df2['Academic Pressure']=df2['Academic Pressure'].fillna(0)
df1['Work Pressure']=df1['Work Pressure'].fillna(0)
df2['Work Pressure']=df2['Work Pressure'].fillna(0)

df1['Pressure']=df1['Academic Pressure']+df1['Work Pressure']
df1=df1.drop(['Work Pressure','Academic Pressure'],axis=1)
df2['Pressure']=df2['Academic Pressure']+df2['Work Pressure']
df2=df2.drop(['Work Pressure','Academic Pressure'],axis=1)


df1['Study Satisfaction']=df1['Study Satisfaction'].fillna(0)
df2['Study Satisfaction']=df2['Study Satisfaction'].fillna(0)
df1['Job Satisfaction']=df1['Job Satisfaction'].fillna(0)
df2['Job Satisfaction']=df2['Job Satisfaction'].fillna(0)

df1['Satisfaction']=df1['Job Satisfaction']+df1['Study Satisfaction']
df1=df1.drop(['Job Satisfaction','Study Satisfaction'],axis=1)
df2['Satisfaction']=df2['Job Satisfaction']+df2['Study Satisfaction']
df2=df2.drop(['Job Satisfaction','Study Satisfaction'],axis=1)


df1.loc[df1['Working Professional or Student'] == 'Student', 'Profession'] = 'Student'
df2.loc[df2['Working Professional or Student'] == 'Student', 'Profession'] = 'Student'



columns=df2.columns.to_list()


df1[columns]=df1[columns].fillna('na')
df2[columns]=df2[columns].fillna('na')


df1


df2


import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 假设 df1 已经定义
# 这里添加一个示例数据读取，实际使用时请根据你的数据路径修改
# df1 = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')

X = df1.drop('Depression', axis=1)
y = df1['Depression']

# 筛选分类型特征名称列表
# 假设 category 是分类特征的列索引或名称列表
# 例如 category = ['col1', 'col2']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 CatBoostClassifier 模型，并增加更多可调整参数
model = CatBoostClassifier(
    iterations=1000,  # 迭代次数，即树的数量
    learning_rate=0.01,  # 学习率，控制每次迭代步长
    depth=6,  # 树的最大深度
    l2_leaf_reg=3,  # L2 正则化系数，用于防止过拟合
    bagging_temperature=1,  # 随机采样的强度，值越大采样越随机
    random_strength=1,  # 特征分裂时的随机强度，增加模型的随机性
    one_hot_max_size=2,  # 进行独热编码的最大类别数，超过该值使用其它编码方式
    border_count=254,  # 特征分箱的最大数量
    leaf_estimation_iterations=10,  # 叶子节点值的估计迭代次数
    cat_features=category,  # 分类特征的列索引或名称列表
    verbose=100,  # 每 100 轮迭代输出一次信息
    early_stopping_rounds=50,  # 如果验证集上的指标在 50 轮迭代内没有提升，则提前停止训练
    random_seed=42  # 随机数种子，保证结果可复现
)

# 进行交叉验证
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# 打印交叉验证结果
print(f'交叉验证准确率分数: {cv_scores}')
print(f'交叉验证平均准确率分数: {cv_scores.mean()}')

# 训练模型
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 预测
y_pred = model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 打印评估指标
print(f'准确率: {accuracy}')
print(f'精确率: {precision}')
print(f'召回率: {recall}')
print(f'F1 分数: {f1}')


# ... 你现有的代码 ...

# 获取特征重要性
feature_importances = model.get_feature_importance()

# 创建一个包含特征名和其重要性的 DataFrame
feature_names = X.columns
importance_df = pd.DataFrame({'特征名': feature_names, '特征重要性': feature_importances})

# 按特征重要性降序排序
importance_df = importance_df.sort_values(by='特征重要性', ascending=False)

# 输出特征重要性
print(importance_df)


y_pred1 = model.predict(df2)



y_pred1


data={
   'id':ID,
   'Depression':y_pred1
}


df4=pd.DataFrame(data)


df4


df4.to_csv('submission.csv',index=False)

