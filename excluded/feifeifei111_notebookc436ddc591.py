import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 6)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
import numpy as np 
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False  # 允许负号显示


test_data = pd.read_csv(r'/kaggle/input/playground-series-s5e6/test.csv')
train_data = pd.read_csv(r'/kaggle/input/playground-series-s5e6/train.csv')
train_data.head()


test_data = test_data.drop(columns=['id'])
train_data = train_data.drop(columns=['id'])


num_col = [i for i in train_data.columns if train_data[i].dtype == 'int64']
cat_col = [i for i in train_data.columns if  i not in num_col]
print(f'数值变量有{len(num_col)}个, 分类变量有{len(cat_col)}个')


fig, axes = plt.subplots(len(num_col), 2, figsize=(18, 35))
for i, col in enumerate(num_col):
    sns.histplot(train_data[col], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'{col}分布图', fontsize=22)
    axes[i, 0].grid(True, linestyle='--', alpha=0.7)
    
    sns.boxenplot(train_data[col], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'{col}箱线图', fontsize=22)
    axes[i, 1].grid(True, linestyle='--', alpha=0.7)
    
plt.tight_layout()
plt.show()


# cat_col 
fig, ax = plt.subplots(len(cat_col), 2, figsize=(16, 24))
for idx, col in enumerate(cat_col):
    # 柱形图
    sns.countplot(data=train_data, x=col, ax=ax[idx, 0])
    ax[idx, 0].set_title('counts of {}'.format(col))
    ax[idx, 0].grid(True, linestyle='--', alpha=0.7)
    # 饼图
    train_data[col].value_counts().plot(kind='pie', ax=ax[idx, 1])
    ax[idx, 1].set_title('pie of {}'.format(col))
plt.tight_layout()
plt.show()


# 多变量分布
fig, axes = plt.subplots(len(num_col), 3, figsize=(18, 35))
for i, col in enumerate(num_col):
    sns.kdeplot(data=train_data, x=col,fill=True, ax=axes[i, 0], 
                 hue='Fertilizer Name')  
    axes[i, 0].set_title(f'{col}分布图', fontsize=22)
    axes[i, 0].grid(True, linestyle='--', alpha=0.7)

    sns.boxenplot(data=train_data, x=col, ax=axes[i, 1], 
                  hue='Fertilizer Name') 
    axes[i, 1].set_title(f'{col}箱线图', fontsize=22)
    axes[i, 1].grid(True, linestyle='--', alpha=0.7)


    sns.violinplot(data=train_data, x=col, ax=axes[i, 2], 
                   hue='Fertilizer Name') 
    # 添加小提琴图的标题
    axes[i, 2].set_title(f'{col}小提琴图', fontsize=22)
    axes[i, 2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# 特征工程
train_data['Temp_Humid'] = train_data['Temparature'] / train_data['Humidity'] + 1
test_data['Temp_Humid'] = test_data['Temparature'] / test_data['Humidity']

# 微量元素
train_data['sum_Ni_Po_Ph'] = train_data[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis=1)
test_data['sum_Ni_Po_Ph'] = test_data[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis=1)

train_data['N/P_ratio'] = train_data['Nitrogen'] / (train_data['Phosphorous'] + 1) 
test_data['N/P_ratio'] = test_data['Nitrogen'] / (test_data['Phosphorous'] + 1) 

train_data['N/K_ratio'] = train_data['Nitrogen'] / (train_data['Potassium'] + 1)
test_data['N/K_ratio'] = test_data['Nitrogen'] / (test_data['Potassium'] + 1)

train_data['K/P_ratio'] = train_data['Potassium'] / (train_data['Phosphorous'] + 1)
test_data['K/P_ratio'] = test_data['Potassium'] / (test_data['Phosphorous'] + 1)

train_data['NPK_balance'] = train_data['Nitrogen'] + train_data['Phosphorous'] - train_data['Potassium']  # 营养平衡值
test_data['NPK_balance'] = test_data['Nitrogen'] + test_data['Phosphorous'] - test_data['Potassium']

bins = [0, 30, 60, 100]
labels = ['Dry', 'Optimal', 'Wet']
train_data['Moisture_Level'] = pd.cut(train_data['Moisture'], bins=bins, labels=labels)
test_data['Moisture_Level'] = pd.cut(test_data['Moisture'], bins=bins, labels=labels)




