import pandas as pd
train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


train_df.head()


train_df.shape


train_df.isnull().sum()


train_df['y'].value_counts()


print(len(train_df[train_df['y'] == 0]) / len(train_df['y']))
print(len(train_df[train_df['y'] == 1]) / len(train_df['y']))


train_df.describe()


train_df.describe(include=['object'])


import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('whitegrid')

numerical_features_to_analyze = ['age', 'balance', 'duration']

for feature in numerical_features_to_analyze:
    plt.figure(figsize=(16, 6))

    # 1. 绘制整体分布的直方图
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True, bins=50)
    plt.title(f'\'{feature}\' Distribution (Histogram)')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    # 2. 绘制箱线图以观察异常值
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[feature])
    plt.title(f'\'{feature}\' Distribution (Box Plot)')
    plt.xlabel(feature)

    plt.suptitle(f'Univariate Analysis of {feature}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 3. 绘制特征与目标变量 y 的关系
    plt.figure(figsize=(12, 6))
    # 使用直方图，按y的值进行区分
    sns.histplot(data=train_df, x=feature, hue='y', kde=True, common_norm=False, stat="density")
    plt.title(f'\'{feature}\' Distribution by Target \'y\'')
    plt.xlabel(feature)
    plt.ylabel('Density')


feature = 'job'

# 创建一个1行2列的图表，并设置大小
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- 左图: 绘制类别的计数图 ---
    # 按值的计数降序排列
order = train_df[feature].value_counts().index
sns.countplot(data=train_df, y=feature, order=order, ax=axes[0])
axes[0].set_title(f'Frequency Distribution of \'{feature}\'')
axes[0].set_xlabel('Count')
axes[0].set_ylabel(feature)

    # --- 右图: 绘制各类别对应的认购率 ---
    # 计算每个类别的认购率
subscription_rate = train_df.groupby(feature)['y'].mean().sort_values(ascending=False) * 100
sns.barplot(x=subscription_rate.values, y=subscription_rate.index, ax=axes[1])
axes[1].set_title(f'Subscription Rate (%) by \'{feature}\'')
axes[1].set_xlabel('Subscription Rate (%)')
axes[1].set_ylabel('') # 隐藏y轴标签，因为与左图相同

    # 整体标题和布局调整
plt.suptitle(f'Univariate Analysis of {feature}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
plt.show()


feature = 'marital'

# 创建一个1行2列的图表，并设置大小
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- 左图: 绘制类别的计数图 ---
    # 按值的计数降序排列
order = train_df[feature].value_counts().index
sns.countplot(data=train_df, y=feature, order=order, ax=axes[0])
axes[0].set_title(f'Frequency Distribution of \'{feature}\'')
axes[0].set_xlabel('Count')
axes[0].set_ylabel(feature)

    # --- 右图: 绘制各类别对应的认购率 ---
    # 计算每个类别的认购率
subscription_rate = train_df.groupby(feature)['y'].mean().sort_values(ascending=False) * 100
sns.barplot(x=subscription_rate.values, y=subscription_rate.index, ax=axes[1])
axes[1].set_title(f'Subscription Rate (%) by \'{feature}\'')
axes[1].set_xlabel('Subscription Rate (%)')
axes[1].set_ylabel('') # 隐藏y轴标签，因为与左图相同

    # 整体标题和布局调整
plt.suptitle(f'Univariate Analysis of {feature}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
plt.show()


feature = 'education'

# 创建一个1行2列的图表，并设置大小
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- 左图: 绘制类别的计数图 ---
    # 按值的计数降序排列
order = train_df[feature].value_counts().index
sns.countplot(data=train_df, y=feature, order=order, ax=axes[0])
axes[0].set_title(f'Frequency Distribution of \'{feature}\'')
axes[0].set_xlabel('Count')
axes[0].set_ylabel(feature)

    # --- 右图: 绘制各类别对应的认购率 ---
    # 计算每个类别的认购率
subscription_rate = train_df.groupby(feature)['y'].mean().sort_values(ascending=False) * 100
sns.barplot(x=subscription_rate.values, y=subscription_rate.index, ax=axes[1])
axes[1].set_title(f'Subscription Rate (%) by \'{feature}\'')
axes[1].set_xlabel('Subscription Rate (%)')
axes[1].set_ylabel('') # 隐藏y轴标签，因为与左图相同

    # 整体标题和布局调整
plt.suptitle(f'Univariate Analysis of {feature}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
plt.show()


feature = 'poutcome'

# 创建一个1行2列的图表，并设置大小
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- 左图: 绘制类别的计数图 ---
    # 按值的计数降序排列
order = train_df[feature].value_counts().index
sns.countplot(data=train_df, y=feature, order=order, ax=axes[0])
axes[0].set_title(f'Frequency Distribution of \'{feature}\'')
axes[0].set_xlabel('Count')
axes[0].set_ylabel(feature)

    # --- 右图: 绘制各类别对应的认购率 ---
    # 计算每个类别的认购率
subscription_rate = train_df.groupby(feature)['y'].mean().sort_values(ascending=False) * 100
sns.barplot(x=subscription_rate.values, y=subscription_rate.index, ax=axes[1])
axes[1].set_title(f'Subscription Rate (%) by \'{feature}\'')
axes[1].set_xlabel('Subscription Rate (%)')
axes[1].set_ylabel('') # 隐藏y轴标签，因为与左图相同

    # 整体标题和布局调整
plt.suptitle(f'Univariate Analysis of {feature}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
plt.show()


sns.set_style('whitegrid')

# --- 1. 数值特征之间的相关性热力图 ---
plt.figure(figsize=(12, 8))

# 计算相关性矩阵
# 我们只选择部分有意义的数值特征进行分析，去掉id
numeric_cols_for_corr = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y']
corr_matrix = train_df[numeric_cols_for_corr].corr()

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.savefig('bivariate_correlation_heatmap.png')
plt.show()


# --- 2. 类别特征 vs 数值特征: job vs balance ---
import numpy as np
plt.figure(figsize=(14, 8))

# 由于 'balance' 分布极度不均，我们对其进行对数变换以便于观察
# log(x+c) where c is a constant to handle non-positive values
# We will filter out negative balances for this specific visualization for simplicity.
df_positive_balance = train_df[train_df['balance'] > 0].copy()
df_positive_balance['log_balance'] = np.log(df_positive_balance['balance'])

# 按 'job' 类别的中位数余额排序，使图表更具可读性
order = df_positive_balance.groupby('job')['balance'].median().sort_values(ascending=False).index

sns.boxplot(data=df_positive_balance, x='job', y='log_balance', order=order)
plt.xticks(rotation=45)
plt.title('Log of Balance Distribution across Job Types', fontsize=16)
plt.xlabel('Job')
plt.ylabel('Log(Balance)')
plt.tight_layout()
plt.savefig('bivariate_job_vs_balance.png')
plt.show()

