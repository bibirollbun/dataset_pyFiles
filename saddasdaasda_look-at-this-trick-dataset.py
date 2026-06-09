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
df1 = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df2 = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
# 导入测试数据
df3 = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
id1=df3['id']
# 删除ID列
df1 = df1.drop('id', axis=1)
df2 = df2.drop('id', axis=1)
df3 = df3.drop('id', axis=1)
# 排除目标列 'target'
features_df = df1.drop(columns=['Price'])

# 筛选分类型特征名称列表
category = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

# 筛选数值型特征名称列表
numeric = features_df.select_dtypes(exclude=['object', 'category']).columns.tolist()




df1 = pd.concat([df1, df2], ignore_index=True)


df1.dtypes


import matplotlib.pyplot   as plt
# 绘制箱线图
df1['Price'].plot(kind='box')

# 设置图形标题和坐标轴标签
plt.title('Box Plot of Price')
plt.ylabel('Price')

# 显示图形
plt.show()


import seaborn as sns
mean_prices = df1.groupby('Brand')['Price'].mean()

# 绘制柱状图
sns.barplot(x=mean_prices.index, y=mean_prices.values)
plt.title('Average Price by Category')
plt.ylabel('Average Price')
plt.show()


for i in category:
    mean_prices = df1.groupby(i)['Price'].mean()
    sns.barplot(x=mean_prices.index, y=mean_prices.values)
    plt.title('Average Price by Category')
    plt.ylabel('Average Price')
    plt.show()  
    


