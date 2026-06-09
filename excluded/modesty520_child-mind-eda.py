import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline


#导入数据
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
data_dict = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv')


#定义函数，查看数据的维度及前若干行信息
def display_dataset_info(dataset, num_rows=5, max_columns=10):
    """
    输出数据集的前若干行和数据集的形状信息

    参数:
    dataset (pd.DataFrame): 输入的Pandas数据集
    num_rows (int): 显示的行数，默认为5
    max_columns (int or None): 显示的列数，默认为10（显示最多10列）
    """
    # 显示前 num_rows 行，并控制列显示数
    with pd.option_context('display.max_columns', max_columns):
        display(dataset.head(num_rows))
    # 打印形状
    print(f"Dataset shape: {dataset.shape}")


max_columns = train.shape[1]
display_dataset_info(train, max_columns=max_columns)


max_columns = test.shape[1]
display_dataset_info(test, max_columns=max_columns)


num_rows = data_dict.shape[0]
display_dataset_info(data_dict, num_rows=num_rows)


#查看训练集中存在而测试集中不存在的变量
#一般来说，训练集和数据集的差别在于一个目标变量，但这里却有许多，但都是和目标变量相关的，也有许多大佬用不同的目标变量构建最后的模型
train_cols = set(train.columns)
test_cols = set(test.columns)
columns_not_in_test = sorted(list(train_cols - test_cols)) 
data_dict[data_dict['Field'].isin(columns_not_in_test)].iloc[:, 1]


#定义缺失值统计及可视化函数
def display_missing_values_info(dataset):
    """
    显示数据集中每列的缺失值总数和缺失值比例

    参数:
    dataset (pd.DataFrame): 输入的Pandas数据集

    返回:
    pd.DataFrame: 包含缺失值总数和比例的信息
    """
    # 计算缺失值总数
    missing_count = dataset.isnull().sum()
    # 计算缺失值比例
    missing_percentage = missing_count / len(dataset)
    # 合并到一个 DataFrame 中
    missing_info = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    }).sort_values(by='Missing Count', ascending=False)

    return missing_info

def plot_missing_values_bar(missing_dataframe):
    """
    根据缺失值 DataFrame 绘制缺失值和可用值比例的水平堆叠条形图

    参数:
    missing_dataframe (pd.DataFrame): 包含缺失值信息的 DataFrame，
                                       需要包含 'Missing Count' 和 'Missing Percentage' 列
    """
    # 过滤掉没有缺失值的列
    missing_dataframe = missing_dataframe[missing_dataframe['Missing Count'] > 0]

    if missing_dataframe.empty:
        print("没有缺失值，图形无法绘制。")
        return

    # 提取特征名、缺失值比例和可用值比例
    features = missing_dataframe.index
    missing_ratios = missing_dataframe['Missing Percentage']
    available_ratios = 1 - missing_ratios

    # 设置图形尺寸
    plt.figure(figsize=(10, 8))

    # 绘制缺失值比例（左部分）
    plt.barh(np.arange(len(features)), missing_ratios, color='coral', label='Missing', edgecolor='black')

    # 绘制可用值比例（右部分，堆叠在缺失值之后）
    plt.barh(np.arange(len(features)), available_ratios, left=missing_ratios, color='darkseagreen', label='Available', edgecolor='black')

    # 设置 y 轴为特征名
    plt.yticks(np.arange(len(features)), features)

    # 设置 x 轴为百分比格式
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # 设置轴范围和图例
    plt.xlim(0, 1)
    plt.xlabel('Percentage', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Missing vs Available Values', fontsize=14)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


missing_dataframe = display_missing_values_info(train)
missing_dataframe


plot_missing_values_bar(missing_dataframe)


#定义数值变量直方图和分类变量柱状图的可视化函数
def plot_categorical_distribution(train, categorical_col, include_na=False, ax=None):
    """
    绘制分类变量的分布及其百分比，并可选择是否包含缺失值

    参数:
    train (pd.DataFrame): 输入数据集
    categorical_col (str): 分类变量的列名
    include_na (bool): 是否统计缺失值，默认为 False
    """
    # 计算分类变量的计数
    if include_na:
        counts = train[categorical_col].value_counts(dropna=False).reset_index()
        counts[categorical_col] = counts[categorical_col].fillna('Missing')  # 将缺失值标记为 'Missing'
    else:
        counts = train[categorical_col].value_counts().reset_index()

    counts.columns = [categorical_col, 'count']
    total = counts['count'].sum()
    counts['percentage'] = (counts['count'] / total) * 100

    # 如果 ax 是 None，则创建新的图形和轴
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制分类变量的分布柱状图
    sns.barplot(x=categorical_col, y='count', data=counts, palette='Blues_d', ax=ax)
    ax.set_title(f'Distribution of {categorical_col} (Include Missing: {include_na})', fontsize=14)

    # 在柱状图上显示具体数值和百分比
    for p in ax.patches:
        height = p.get_height()
        # 找到当前高度对应的百分比
        percentage = counts.loc[counts['count'] == height, 'percentage'].values[0]
        ax.text(
            p.get_x() + p.get_width() / 2,
            height + 5, f'{int(height)} ({percentage:.1f}%)',
            ha="center", fontsize=12
        )

    # 设置标签
    ax.set_xlabel(categorical_col, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()



def plot_numeric_distribution(train, numeric_col, bins=20, ax=None):
    """
    绘制数值型变量的分布直方图，自动处理缺失值和无穷值

    参数:
    train (pd.DataFrame): 输入数据集
    numeric_col (str): 数值型变量的列名
    bins (int): 直方图的分箱数量，默认为 20
    """
    # 检查列是否存在
    if numeric_col not in train.columns:
        raise ValueError(f"列 '{numeric_col}' 不存在于数据集中")

    # 处理无穷值，将 inf 和 -inf 替换为 NaN
    numeric_data = train[numeric_col].replace([np.inf, -np.inf], np.nan).dropna()

    # 检查是否有有效数据
    if numeric_data.empty:
        print(f"列 '{numeric_col}' 不包含有效数值数据")
        return

    # 如果 ax 是 None，则创建新的图形和轴
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制直方图
    ax.hist(numeric_data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    # 设置标题和标签
    ax.set_title(f'Distribution of {numeric_col}', fontsize=14)
    ax.set_xlabel(f'{numeric_col}', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()


#对目标变量的含义进行定义后映射
sii_map = {0.0: '0 (None)', 1.0: '1 (Mild)', 2.0: '2 (Moderate)', 3.0: '3 (Severe)'}
train['sii'] = train['sii'].map(sii_map)
# 将非缺失值转换为int类型
train['PCIAT-PCIAT_Total'] = train['PCIAT-PCIAT_Total'].apply(lambda x: int(x) if not pd.isna(x) else x)




fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 调用函数绘制类别变量的分布
plot_categorical_distribution(train, 'sii', include_na=True, ax=axes[0])
# 调用函数绘制数值变量的分布
plot_numeric_distribution(train, 'PCIAT-PCIAT_Total', bins=10, ax=axes[1])

#这里绘制数值型变量直方图时除了点问题，但是单独执行这个代码是没有问题的，不知到为什么设置成子图后就不显示了，本地测试也是没有问题的，
#心累~ ~


#探究目标变量sii和PCIAT-PCIAT_Total之间的关系
pciat_min_max = train.groupby('sii')['PCIAT-PCIAT_Total'].agg(['min', 'max'])
pciat_min_max = pciat_min_max.rename(
    columns={'min': 'Minimum PCIAT total Score', 'max': 'Maximum total PCIAT Score'})
pciat_min_max


#探究变量PCIAT-PCIAT_Total与PCIAT-PCIAT_01 to PCIAT-PCIAT_20之间的关系
#在上面的缺失值情况统计时，我们知道变量PCIAT-PCIAT_Total与PCIAT-PCIAT_01 to PCIAT-PCIAT_20的缺失情况不一致，这说明
#有的人可能只进行了部分问题测试，其中没有回答被记为NA Values,这样就会导致没有完成所有测试问题的孩子分可能被低估
filtered_columns = [col for col in train.columns if 'PCIAT-PCIAT' in col or 'sii' in col]
train_with_sii = train[~train['sii'].isnull()][filtered_columns]

#显示在剔除所有缺失值列后，所有PCIAT-PCIAT系列变量至少含有一个缺失数值的样本
train_with_sii[train_with_sii.isnull().any(axis=1)].style.applymap(
    lambda x: 'background-color: #FFC0CB' if pd.isna(x) else '')


#确定这些数据中的缺失值是否直接被忽略或是当作0值来处理
PCIAT_cols = [f'PCIAT-PCIAT_{i+1:02d}' for i in range(20)]
recalc_total_score = train_with_sii[PCIAT_cols].sum(
    axis=1, skipna=True)
(recalc_total_score == train_with_sii['PCIAT-PCIAT_Total']).all()


#确定这些数据中哪些缺失值可能会影响变量SII的值，思路是对于每个测试中的问题，最高得分为5分，如果
#加上最高分的情况下，SII的值仍然不变（未达到阈值），这说明不会影响SII的结果，反之，则可能影响SII的结果

def recalculate_sii(row):
    max_possible = row['PCIAT-PCIAT_Total'] + row[PCIAT_cols].isna().sum() * 5
    if row['PCIAT-PCIAT_Total'] <= 30 and max_possible <= 30:
        return 0
    elif 31 <= row['PCIAT-PCIAT_Total'] <= 49 and max_possible <= 49:
        return 1
    elif 50 <= row['PCIAT-PCIAT_Total'] <= 79 and max_possible <= 79:
        return 2
    elif row['PCIAT-PCIAT_Total'] >= 80 and max_possible >= 80:
        return 3
    return np.nan



train_with_sii['recalc_sii'] = train_with_sii.apply(recalculate_sii, axis=1)

reversed_sii_map = {v: k for k, v in sii_map.items()}
train_with_sii['sii'] = train_with_sii['sii'].map(reversed_sii_map)

mismatch_rows = train_with_sii.query('recalc_sii != sii')
mismatch_rows




#借鉴非常好用的统计函数
def calculate_stats(data, columns):
    if isinstance(columns, str):
        columns = [columns]

    stats = []
    for col in columns:
        if data[col].dtype in ['object', 'category']:
            counts = data[col].value_counts(dropna=False, sort=False)
            percents = data[col].value_counts(normalize=True, dropna=False, sort=False) * 100
            formatted = counts.astype(str) + ' (' + percents.round(2).astype(str) + '%)'
            stats_col = pd.DataFrame({'count (%)': formatted})
            stats.append(stats_col)
        else:
            stats_col = data[col].describe().to_frame().transpose()
            stats_col['missing'] = data[col].isnull().sum()
            stats_col.index.name = col
            stats.append(stats_col)

    return pd.concat(stats, axis=0)


#Age&Sex
train['sii'].fillna('Missing', inplace=True)
demo_features = [col for col in train.columns if 'Basic_Demos' in col]


#人口统计特征变量基本情况
sex_map = {0: 'Male', 1: 'Female'}
train['Basic_Demos-Sex'] = train['Basic_Demos-Sex'].map(sex_map)



calculate_stats(train, ['Basic_Demos-Sex'])


#注意，cut()默认是左开右闭，而不是左闭右开
train['Age_Group'] = pd.cut(
    train['Basic_Demos-Age'],
    bins=[4, 12, 18, 22],
    labels=['Children (5-12)', 'Adolescents (13-18)', 'Adults (19-22)']
)

calculate_stats(train, ['Basic_Demos-Sex', 'Age_Group', 'Basic_Demos-Enroll_Season'])


#sii by Age & PCIAT_PCIAT_Total by Age
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

#PCIAT-PCIAT_Total by Age_Group
sns.boxplot(
    x='Age_Group', y='PCIAT-PCIAT_Total',
    data=train, palette="Set3", ax=axes[0, 0])
axes[0, 0].set_title('PCIAT_Total by Age Group')
axes[0, 0].set_ylabel('PCIAT_Total values')
axes[0, 0].set_xlabel('Age Group')

#SII by Age_Group
sns.countplot(
    x='Age_Group', hue='sii',
    data=train, palette="Set3", ax=axes[0, 1])
axes[0, 1].set_title('SII by Age Group')
axes[0, 1].set_ylabel('SII Counts')
axes[0, 1].set_xlabel('Age Group')

#PCIAT-PCIAT_Total by Sex
sns.boxplot(
    x='Basic_Demos-Sex', y='PCIAT-PCIAT_Total',
    data=train, palette="Set3", ax=axes[1, 0])
axes[1, 0].set_title('PCIAT_Total by Sex')
axes[1, 0].set_ylabel('PCIAT_Total values')
axes[1, 0].set_xlabel('Sex')

#SII by Sex
sns.countplot(
    x='Basic_Demos-Sex', hue='sii',
    data=train, palette="Set3", ax=axes[1, 1])
axes[1, 1].set_title('SII by Sex')
axes[1, 1].set_ylabel('SII Counts')
axes[1, 1].set_xlabel('Sex')



#SII by Age_Group
stats = train.groupby(['Age_Group', 'sii']).size().unstack(fill_value=0)
stats_prop = stats.div(stats.sum(axis=1), axis=0) * 100
stats = stats.astype(str) +' (' + stats_prop.round(1).astype(str) + '%)'

max_columns = stats.shape[1]
with pd.option_context('display.max_columns', max_columns):
    display(stats)


#SII by Sex
stats = train.groupby(['Basic_Demos-Sex', 'sii']).size().unstack(fill_value=0)
stats_prop = stats.div(stats.sum(axis=1), axis=0) * 100
stats = stats.astype(str) +' (' + stats_prop.round(1).astype(str) + '%)'
stats


#明显存在异常值
calculate_stats(train, 'CGAS-CGAS_Score')
plot_numeric_distribution(train, 'CGAS-CGAS_Score', bins=100)


#这条异常数据绝大部分特征都是缺失的，直接剔除即可
outlier = train.query('`CGAS-CGAS_Score` == 999.0')
train.query('`CGAS-CGAS_Score` != 999.0', inplace=True)
#剔除后的数据分布
plot_numeric_distribution(train, 'CGAS-CGAS_Score', bins=20)


#CGAS-CGAS_Score与目标变量的关系
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#CGAS-CGAS_Score by sii
sns.boxplot(
    x='sii', y='CGAS-CGAS_Score',
    data=train, palette="Set3", ax=axes[0])
axes[0].set_title('CGAS-CGAS_Score by sii')
axes[0].set_ylabel('CGAS-CGAS_Score')
axes[0].set_xlabel('sii')

#回归趋势图
sns.regplot(x='CGAS-CGAS_Score', y='PCIAT-PCIAT_Total', data=train, scatter_kws={'alpha': 0.5}, ax=axes[1])
axes[1].set_title('Regression Plot')

plt.show()


train.rename(columns={'PreInt_EduHx-computerinternet_hoursday': 'hours of internet_use'}, inplace=True)

hours_map = {0.0: '< 1h/day', 1.0: '~ 1h/day', 2.0: '~ 2hs/day', 3.0: '> 3hs/day'}
train['hours of internet_use'] = train['hours of internet_use'].map(hours_map)
train['hours of internet_use'].fillna('Missing', inplace=True)
calculate_stats(train, 'hours of internet_use')


#CGAS-CGAS_Score与目标变量的关系

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#CGAS-CGAS_Score by sii
sns.boxplot(
    x='hours of internet_use', y='PCIAT-PCIAT_Total',
    data=train, palette="Set3", ax=axes[0])
axes[0].set_title('PCIAT-PCIAT_Total by hours of internet_use')
axes[0].set_ylabel('PCIAT-PCIAT_Total')
axes[0].set_xlabel('hours of internet_use')

#SII by hours of internet_use
sns.countplot(
    x='hours of internet_use', hue='sii',
    data=train, palette="Set3", ax=axes[1])
axes[1].set_title('SII by hours of internet_use')
axes[1].set_ylabel('SII Counts')
axes[1].set_xlabel('Age Group')

plt.show()


#hours of internet_use by sex
stats = train.groupby(['Basic_Demos-Sex', 'hours of internet_use']).size().unstack(fill_value=0)
stats_prop = stats.div(stats.sum(axis=1), axis=0) * 100
stats = stats.astype(str) +' (' + stats_prop.round(1).astype(str) + '%)'
stats


#hours of internet_use by Age_Group
stats = train.groupby(['Age_Group', 'hours of internet_use']).size().unstack(fill_value=0)

fig, axes = plt.subplots(1, len(stats), figsize=(18, 5))
for i, age_group in enumerate(stats.index):
    group_counts = stats.loc[age_group] / stats.loc[age_group].sum()
    axes[i].pie(group_counts, labels=group_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette("Set3"), labeldistance=1.1)
    axes[i].set_title(f'Distribution of Hours of Internet Use\n{age_group}')
    axes[i].axis('equal')

plt.tight_layout()
plt.show()



reversed_hours_map = {v: k for k, v in hours_map.items()}
train['hours of internet_use'] = train['hours of internet_use'].map(reversed_hours_map)

# 自定义四分位数函数
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)
# 分组统计
train.groupby('Age_Group')['hours of internet_use'].agg(
    mean=lambda x: round(x.mean(), 2),
    median=lambda x: round(x.median(), 1),
    q25=lambda x: round(q25(x), 1),
    q75=lambda x: round(q75(x), 1))



SDS_features = [col for col in train.columns if 'SDS-SDS' in col]

calculate_stats(train, SDS_features)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_numeric_distribution(train, SDS_features[0], bins=20, ax=axes[0])
plot_numeric_distribution(train, SDS_features[1], bins=20, ax=axes[1])

#为什么第二张子图老是有问题呢 ，本地是没有问题的。。。


#SDS与目标变量的关系
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
#SDS by sii
sns.boxplot(
    x='sii', y=SDS_features[0],
    data=train, palette="Set3", ax=axes[0, 0])
axes[0, 0].set_title(f'{SDS_features[0]} by sii')
axes[0, 0].set_ylabel('CGAS-CGAS_Score')
axes[0, 0].set_xlabel(f'{SDS_features[0]}')

sns.boxplot(
    x='sii', y=SDS_features[1],
    data=train, palette="Set3", ax=axes[0, 1])
axes[0, 1].set_title(f'{SDS_features[1]} by sii')
axes[0, 1].set_ylabel('CGAS-CGAS_Score')
axes[0, 1].set_xlabel(f'{SDS_features[1]}')

#回归趋势图
sns.regplot(x='CGAS-CGAS_Score', y=SDS_features[0], data=train, scatter_kws={'alpha': 0.5}, ax=axes[1, 0])
axes[1, 0].set_title('Regression Plot')

sns.regplot(x='CGAS-CGAS_Score', y=SDS_features[1], data=train, scatter_kws={'alpha': 0.5}, ax=axes[1, 1])
axes[1, 1].set_title('Regression Plot')

plt.show()



# 分组统计
train.groupby('Age_Group')[SDS_features[0]].agg(
    mean=lambda x: round(x.mean(), 2),
    median=lambda x: round(x.median(), 1),
    q25=lambda x: round(q25(x), 1),
    q75=lambda x: round(q75(x), 1))


train.groupby('Basic_Demos-Sex')[SDS_features[0]].agg(
    mean=lambda x: round(x.mean(), 2),
    median=lambda x: round(x.median(), 1),
    q25=lambda x: round(q25(x), 1),
    q75=lambda x: round(q75(x), 1))


train.groupby('Age_Group')[SDS_features[1]].agg(
    mean=lambda x: round(x.mean(), 2),
    median=lambda x: round(x.median(), 1),
    q25=lambda x: round(q25(x), 1),
    q75=lambda x: round(q75(x), 1))


train.groupby('Basic_Demos-Sex')[SDS_features[1]].agg(
    mean=lambda x: round(x.mean(), 2),
    median=lambda x: round(x.median(), 1),
    q25=lambda x: round(q25(x), 1),
    q75=lambda x: round(q75(x), 1))


PAQ_features = ['PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total']
calculate_stats(train, PAQ_features)


#问卷年龄的划分存在问题
A_max_min = train[~train['PAQ_A-PAQ_A_Total'].isnull()]['Basic_Demos-Age'].describe()[['max', 'min']]
A_max_min.name = "Age of Adolescents (13-18)"
C_max_min = train[~train['PAQ_C-PAQ_C_Total'].isnull()]['Basic_Demos-Age'].describe()[['max', 'min']]
C_max_min.name = "Children (5-12)"
pd.concat([A_max_min, C_max_min], axis=1)


#存在争议的样本数据
wrong_age_group = train[~train['PAQ_A-PAQ_A_Total'].isnull()].query('`Basic_Demos-Age` >= 13')
wrong_age_group


#是否存在同时具有PAQ_A-PAQ_A_Total和PAQ_C-PAQ_C_Total值的数据
double_PAQs = train[train['PAQ_A-PAQ_A_Total'].notnull() & train['PAQ_C-PAQ_C_Total'].notnull()]
double_PAQs


#可以考虑剔除上述数据并且将变量PAQ_A-PAQ_A_Total和PAQ_C-PAQ_C_Total进行合并
train = train[~train['id'].isin(double_PAQs['id'])]
train['PAQ_Total_Combined'] = train['PAQ_A-PAQ_A_Total'].combine_first(train['PAQ_C-PAQ_C_Total'])


plot_numeric_distribution(train, 'PAQ_Total_Combined')


#PAQ_Total_Combined与目标变量的关系
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#CGAS-CGAS_Score by sii
sns.boxplot(
    x='sii', y='PAQ_Total_Combined',
    data=train, palette="Set3", ax=axes[0])
axes[0].set_title('PAQ_Total_Combined by sii')
axes[0].set_ylabel('PAQ_Total_Combined_Score')
axes[0].set_xlabel('sii')

#回归趋势图
sns.regplot(x='PAQ_Total_Combined', y='PCIAT-PCIAT_Total', data=train, scatter_kws={'alpha': 0.5}, ax=axes[1])
axes[1].set_title('Regression Plot')
plt.show()


#这里注意，是否剔除具有争议的列直接影响统计结论
train.groupby('Age_Group')['PAQ_Total_Combined'].agg(
    mean=lambda x: round(x.mean(), 2),
    median=lambda x: round(x.median(), 1),
    q25=lambda x: round(q25(x), 1),
    q75=lambda x: round(q75(x), 1))


train[~train['id'].isin(wrong_age_group['id'])].groupby('Age_Group')['PAQ_Total_Combined'].agg(
    mean=lambda x: round(x.mean(), 2),
    median=lambda x: round(x.median(), 1),
    q25=lambda x: round(q25(x), 1),
    q75=lambda x: round(q75(x), 1))


train.groupby('Basic_Demos-Sex')['PAQ_Total_Combined'].agg(
    mean=lambda x: round(x.mean(), 2),
    median=lambda x: round(x.median(), 1),
    q25=lambda x: round(q25(x), 1),
    q75=lambda x: round(q75(x), 1))


#转化度量单位
Physical_features = [col for col in train.columns if 'Physical' in col]
Physical_features.remove('Physical-Season')

lbs_to_kg = 0.453592
inches_to_cm = 2.54

train['Physical-Weight'] = train['Physical-Weight'] * lbs_to_kg
train['Physical-Height'] = train['Physical-Height'] * inches_to_cm
train['Physical-Waist_Circumference'] = train['Physical-Waist_Circumference'] * inches_to_cm


#根据体重和身高计算BMI并于原始数据中的Physical-BMI对比
train['Calculated-BMI'] = np.where(
    train['Physical-Weight'].isna() | train['Physical-Height'].isna(),
    np.nan,
    train['Physical-Weight'] / ((train['Physical-Height'] / 100) ** 2)
)


#计算两者的误差
train['BMI-Absolute-Error'] = abs(train['Calculated-BMI'] - train['Physical-BMI'])
train['BMI-Relative-Error'] = np.where(
    train['Physical-BMI'] != 0,
    train['BMI-Absolute-Error'] / train['Physical-BMI'],
    np.nan  # 避免除以零
)
# 统计误差
mean_abs_error = train['BMI-Absolute-Error'].mean()
std_abs_error = train['BMI-Absolute-Error'].std()
print(f"平均绝对误差: {mean_abs_error:.4f}, 绝对误差标准差: {std_abs_error:.4f}")


# 可视化误差分布
plt.figure(figsize=(10, 6))
plt.hist(train['BMI-Absolute-Error'].dropna(), bins=20, alpha=0.7, edgecolor='black')
plt.title('Distribution of Absolute Errors')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.show()


#异常值分类处理
max_columns = 9
with pd.option_context('display.max_columns', max_columns):
    display(calculate_stats(train, Physical_features))


#Physical-BMI,Physical-Height,Physical-Weight,Physical-Waist_Circumference
#找出所有异常值为0的列,因为在正常情况下，这些指标都不可能为0
BCHW_with_zeros = train.query('`Physical-BMI`==0.00 | `Physical-Weight`==0.00')
BCHW_with_zeros


BCHW_with_zeros[Physical_features].isnull().sum().sort_values(ascending=False)


#Physical-Diastolic_BP,Physical-HeartRate,Physical-Systolic_BP
DHS_with_zeros = train.query('`Physical-Diastolic_BP`==0.00 | `Physical-Systolic_BP`==0.00')
DHS_with_zeros



#脉压（Pulse Pressure）是收缩压（Systolic BP）和舒张压（Diastolic BP）之间的差值,正常情况下，脉压是不可能小于0的
train['Physical-Pulse_Pressure'] = train['Physical-Systolic_BP'] - train['Physical-Diastolic_BP']
PP_below_zeros = train.query('`Physical-Pulse_Pressure` < 0')
PP_below_zeros


#探究Physical变量与目标变量之间的联系

# 创建子图：2行4列，留出额外空间
fig, axes = plt.subplots(2, 4, figsize=(18, 9))  # 2行4列布局，宽度和高度调整

# 遍历特征并绘制
for i, Physical_feature in enumerate(Physical_features):
    ax = axes.flat[i]  # 获取对应的子图位置
    sns.regplot(x=f'{Physical_feature}', y='PCIAT-PCIAT_Total', data=train, scatter_kws={'alpha': 0.5}, ax=ax)
    ax.set_title(f'Regplot: {Physical_feature}')  # 设置标题

# 隐藏多余的子图（第8个空白）
if len(Physical_features) < len(axes.flat):
    for j in range(len(Physical_features), len(axes.flat)):
        axes.flat[j].axis('off')

# 优化布局
plt.tight_layout()
plt.show()


#SII by Physical_features
# 创建子图：2行4列，留出额外空间
fig, axes = plt.subplots(2, 4, figsize=(18, 9))  # 2行4列布局，宽度和高度调整

# 遍历特征并绘制
for i, Physical_feature in enumerate(Physical_features):
    ax = axes.flat[i]  # 获取对应的子图位置
    sns.boxplot(y=f'{Physical_feature}', x='sii', data=train, palette="Set3", ax=ax)
    ax.set_title(f'Boxplot: {Physical_feature}')  # 设置标题

# 隐藏多余的子图（第8个空白）
if len(Physical_features) < len(axes.flat):
    for j in range(len(Physical_features), len(axes.flat)):
        axes.flat[j].axis('off')

# 优化布局
plt.tight_layout()
plt.show()


#Physical变量与目标变量之间的相关关系
data_subset = train[Physical_features + ['PCIAT-PCIAT_Total']]

corr_matrix = data_subset.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

