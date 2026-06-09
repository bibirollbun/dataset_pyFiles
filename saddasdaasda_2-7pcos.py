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

# 导入训练数据
df1 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')

# 导入测试数据
df2 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')
ID=df2['ID']
# 删除ID列
df1 = df1.drop('ID', axis=1)
df2 = df2.drop('ID', axis=1)

# 排除目标列 'target'
features_df = df1.drop(columns=['PCOS'])

# 筛选分类型特征名称列表
category = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

# 筛选数值型特征名称列表
numeric = features_df.select_dtypes(exclude=['object', 'category']).columns.tolist()




df1=df1.fillna('na')
df2=df2.fillna('na')


df1['PCOS']=df1['PCOS'].map({'Yes':1,'No':0})


import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设您已经有一个名为df1的数据框，其中包含了您的特征和目标变量
# df1 = pd.read_csv('your_dataset.csv')

# 定义特征和目标变量
X = df1.drop('PCOS', axis=1)
y = df1['PCOS']


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建CatBoost分类模型，并指定分类特征
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    depth=6,
    l2_leaf_reg=3,
    random_strength=1,
    one_hot_max_size=2,
    border_count=254,
    leaf_estimation_iterations=10,
    random_seed=42,
    cat_features=category  # 指定分类特征
)

# 训练模型
model.fit(X_train, y_train, cat_features=category, verbose=False)

# 进行预测
y_pred = model.predict(X_test)

# 计算评估指标，例如准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率: {accuracy}')




# 使用模型预测概率
probabilities = model.predict_proba(df2)

# probabilities是一个二维数组，每一行对应一个样本，每一列对应一个类别的概率
# 对于二分类问题，probabilities[:, 1]将给出正类的概率
positive_class_probabilities = probabilities[:, 1]

# 现在您有了每个样本属于正类的概率
# 可以根据需要使用这些概率，例如进行阈值判断或进一步分析



import pandas as pd

# 假设 id 和 target 是已经存在的 Series 对象
# 创建一个新的 DataFrame，将 id 命名为 'id'，target 命名为 'target'
df5 = pd.DataFrame({
    'ID': ID,
    'PCOS': positive_class_probabilities
})

# 将 DataFrame 保存为 CSV 文件
df5.to_csv('submission.csv', index=False)

