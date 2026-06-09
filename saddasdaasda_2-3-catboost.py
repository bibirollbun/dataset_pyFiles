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
data1 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')

# 转化为数据框
df1 = pd.DataFrame(data1)

# 排除目标列 'target'
features_df = df1.drop(columns=['PCOS'])

# 筛选分类型特征名称列表
category = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

# 筛选数值型特征名称列表
numeric = features_df.select_dtypes(exclude=['object', 'category']).columns.tolist()




df1


df1.dtypes


df1=df1.drop('ID',axis=1)


for i in category:
    print(i)
    print(df1[i].unique())


df1['Weight_kg'].isnull().sum()
a1=df1['Weight_kg'].mean()
df1['Weight_kg'].fillna(a1,inplace=True)
df1['Weight_kg'].isnull().sum()


for i in category:
    print(i)
    print(df1[i].isnull().sum())


import pandas as pd

# 导入数据
data2 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')

# 转化为数据框
df2= pd.DataFrame(data2)

# 排除目标列 'target'
features_df = df2

# 筛选分类型特征名称列表
category2 = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

# 筛选数值型特征名称列表
numeric = features_df.select_dtypes(exclude=['object', 'category']).columns.tolist()
df2
df2['Weight_kg'].isnull().sum()
df2['Weight_kg'].mean()
df2['Weight_kg'].fillna(df2['Weight_kg'].mean(),inplace=True)
df2['Weight_kg'].isnull().sum()


for i in category2:
    print(i)
    print(df2[i].isnull().sum())


df1copy=df1


df1copy=df1copy.dropna()


df1copy=df1copy.drop('Weight_kg',axis=1)


df1copy.dtypes


pip install pgmpy


import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import networkx as nx
import matplotlib.pyplot as plt

# 假设 df1 是已经准备好的包含上述特征的 DataFrame，且所有特征都是分类变量
# df1 = pd.read_csv('your_data.csv')

# =====================================
# 学习贝叶斯网络结构
# =====================================
# 使用爬山搜索和 BIC 评分来学习网络结构
est = HillClimbSearch(df1copy)
best_model = est.estimate(scoring_method=BicScore(df1copy))

# 创建贝叶斯网络模型
model = BayesianNetwork(best_model.edges())

# 使用最大似然估计来学习参数
model.fit(df1copy, estimator=MaximumLikelihoodEstimator)

# =====================================
# 可视化贝叶斯网络
# =====================================
# 转换为 networkx 图
G = nx.DiGraph(model.edges())

# 绘制图形
# 定义节点布局
pos = nx.spring_layout(G)
# 绘制有向边，设置箭头样式
nx.draw_networkx_edges(G, pos, edge_color='gray', arrowstyle='->', arrowsize=30)
# 绘制节点
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
# 绘制节点标签
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
# 设置图形属性
plt.title('Bayesian Network')
plt.axis('off')
# 显示图形
plt.show()

# =====================================
# 输出模型信息
# =====================================
# 打印模型的边（即变量之间的依赖关系）
print("模型的边：", model.edges())
# 打印每个节点的条件概率分布
print("\n每个节点的条件概率分布：")
for cpd in model.get_cpds():
    print(f"CPD for {cpd.variable}:")
    print(cpd)



import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 假设 df1 已经定义
# 这里添加一个示例数据读取，实际使用时请根据你的数据路径修改
# df1 = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df1['PCOS']=df1['PCOS'].map({'Yes':1,'No':0})
X = df1.drop('PCOS', axis=1)  # 将 'TargetColumn' 替换为实际的目标列名
y = df1['PCOS']


# 将分类特征中的 NaN 值转换为字符串 'nan'
X[category] = X[category].fillna('nan')

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

# 训练模型
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 预测类别
y_pred = model.predict(X_test)

# 预测概率
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_pred_proba)

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# 打印评估指标
print(f'准确率: {accuracy}')
print(f'F1 分数: {f1}')
print(f'AUC: {auc}')

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


df2[category2] = df2[category2].fillna('nan')



df2.isnull().sum()


df2=df2.drop('ID',axis=1)


df2


y_pred_proba_df2 = model.predict_proba(df2)[:, 1]


y_pred_proba_df2


import pandas as pd

# 导入数据
data4 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')

# 转化为数据框
df4= pd.DataFrame(data4)
id1=df4['ID']


id1


data5={
    'ID':id1,
    'PCOS': y_pred_proba_df2
}


df5=pd.DataFrame(data5)


df5


df5.to_csv('submission.csv',index=False)




