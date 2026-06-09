import numpy as np
import polars as pl
import pandas as pd
import os
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV, LassoCV, Lasso, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import font_manager
import seaborn as sns
import numpy as np
import random
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings('ignore')


SEED = 643
n_splits = 10
optimize_params = False
n_trials = 25 # n_trials for optuna 
voting = True
base_thresholds = [30, 50, 80]


TRAIN_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
TEST_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/test.csv'
TRAIN_TS_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet'
TEST_TS_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet'
SUBMISSION_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv'
OUTPUT_PATH = '/kaggle/working/'


def time_features(df):
    """从个人的ActiGraph数据中提取特征的函数"""
    # 将一天中的时间转换为小时
    df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
    # 基本特征
    features = [
        df["non-wear_flag"].mean(),
        df["enmo"][df["enmo"] >= 0.05].sum(),
    ]
    
    # 定义夜间、白天和无掩码（完整数据）的条件
    night = ((df["hours"] >= 22) | (df["hours"] <= 5))
    day = ((df["hours"] <= 20) & (df["hours"] >= 7))
    no_mask = np.ones(len(df), dtype=bool)
    
    # 感兴趣的列和掩码列表
    keys = ["enmo", "anglez", "light", "battery_voltage"]
    masks = [no_mask, night, day]
    
    # 特征提取的辅助函数
    def extract_stats(data):
        return [
            data.mean(), 
            data.std(), 
            data.max(), 
            data.min(), 
            data.diff().mean(), 
            data.diff().std()
        ]
    
    # 遍历键和掩码以生成统计数据
    for key in keys:
        for mask in masks:
            filtered_data = df.loc[mask, key]
            features.extend(extract_stats(filtered_data))

    return features

def process_file(filename, dirname):
    # 处理文件并提取时间特征
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return time_features(df), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    # 从目录并行加载时间序列
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    stats, indexes = zip(*results)
    
    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    
    return df


# 加载表格数据
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# 加载时序数据
train_ts = load_time_series(TRAIN_TS_PATH)
test_ts = load_time_series(TEST_TS_PATH)


print(train.shape)
print(train_ts.shape)


train.head()


# 定义输出文件路径
submission_path = os.path.join(OUTPUT_PATH, "train_data.csv")
train.T.reset_index().to_csv(submission_path, index=False)


# 定义输出文件路径
submission_path = os.path.join(OUTPUT_PATH, "train_ts_data.csv")
train_ts.T.reset_index().to_csv(submission_path, index=False)


train_ts.head()


#设置中文字体
font_path = "/kaggle/input/simchn/SimplifiedChinese.ttf/SimplifiedChinese.ttf"
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False


# 生成缺失值条形图
# 过滤 'sii' 非缺失的行
train_filtered = train.dropna(subset=['sii'])

# 计算缺失值比例
missing_ratio = train_filtered.isna().mean().sort_values(ascending=False)

# 绘制缺失值条形图
plt.figure(figsize=(20,16))
sns.barplot(x=missing_ratio, y=missing_ratio.index, palette="autumn")
plt.xlabel("缺失值比例")
plt.ylabel("特征")
plt.title("仅包含 'sii' 非缺失行的缺失值分布")
plt.show()


print('测试集里没有的列:')
print([f for f in train.columns if f not in test.columns])


# 获取时序数据的列名，并去除 'id' 列
time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

# 将时序数据 train_ts 和 test_ts 通过 'id' 合并到主表train和test, 给合并后的数据添加后缀t以区分原数据。
train_t = pd.merge(train, train_ts, how="left", on='id')
test_t = pd.merge(test, test_ts, how="left", on='id')

# 删除 'id' 列，因为它不再需要
train_t = train_t.drop('id', axis=1)
test_t = test_t.drop('id', axis=1)

# 选择用于训练的特征列，包括基本人口统计信息、身体测量数据、体能测试数据、互联网使用情况等
featuresCols = [
    'Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
    'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
    'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
    'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
    'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
    'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
    'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
    'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
    'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
    'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
    'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
    'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
    'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
    'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
    'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
    'PAQ_C-PAQ_C_Total', 'PCIAT-PCIAT_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',
    'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
    'PreInt_EduHx-computerinternet_hoursday', 'sii'  
] # sii 为目标变量

# 将时序特征列添加到特征列表中
featuresCols += time_series_cols

# 仅保留选定的特征列
train_t = train_t[featuresCols]

# 删除目标变量 'sii' 为空的行
train_t = train_t.dropna(subset=['sii'])

# 定义类别变量
cat_c = [
    'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 
    'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season'
]

# 处理类别变量：填充缺失值并转换为类别类型
def update(df):
    for c in cat_c: 
        df[c] = df[c].fillna('Missing')  # 用 'Missing' 填充缺失值
        df[c] = df[c].astype('category')  # 转换为类别类型
    return df
        
train_t = update(train_t)
test_t = update(test_t)

# 创建类别变量的映射，将类别转换为整数索引
def create_mapping(column, dataset):
    unique_values = dataset[column].unique()
    return {value: idx for idx, value in enumerate(unique_values)}


# 应用类别映射，将类别变量转换为整数
for col in cat_c:
    mapping_train = create_mapping(col, train_t)
    mapping_test = create_mapping(col, test_t)
    
    train_t[col] = train_t[col].replace(mapping_train).astype(int)
    test_t[col] = test_t[col].replace(mapping_test).astype(int)

# 打印最终数据集的形状
print(f'Train Shape : {train_t.shape} || Test Shape : {test_t.shape}')



fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. 目标变量 sii 的分布（柱状图）
sns.countplot(x="sii", data=train, palette="autumn", ax=axes[0])
axes[0].set_title("SII各类别分布")
axes[0].set_xlabel("SII等级 (0: 无, 1: 轻度, 2: 中度, 3: 重度)")

# 2. 目标变量 sii 的分布（饼图）
vc = train['sii'].value_counts()

# 映射 sii 级别到中文标签
sii_map = {0: '无', 1: '轻度', 2: '中度', 3: '重度'}
labels = [sii_map[label] for label in vc.index]

# 绘制饼图
axes[1].pie(vc.values, labels=labels, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])
axes[1].set_title("SII各类别分布（饼图）")

# 3. 连续型指标 PCIAT-PCIAT_Total的分布（直方图+核密度估计）
sns.histplot(train["PCIAT-PCIAT_Total"], bins=30, kde=True, color='skyblue', ax=axes[2])
axes[2].set_title("PCIAT-PCIAT_Total分布")
axes[2].set_xlabel("PCIAT-PCIAT_Total")

# 调整布局，使图像不重叠
plt.tight_layout()
plt.show()


# 计算 "Basic_Demos-Enroll_Season" 变量的各类别计数
vc = train['Basic_Demos-Enroll_Season'].value_counts()

# 映射季节名称到中文
season_map = {'Spring': '春季', 'Summer': '夏季', 'Fall': '秋季', 'Winter': '冬季'}

# 使用映射创建中文标签
labels = [season_map[label] for label in vc.index]

plt.pie(vc, labels=labels)

plt.pie(vc.values, labels=labels, autopct="%1.1f%%")  # 显示每个类别的占比
plt.title('季节注册人数分布')  
plt.show()


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
    
# 按年龄给对象进行分组
train_t['Age Group'] = pd.cut(
    train_t['Basic_Demos-Age'],
    bins=[4, 12, 18, 22],
    labels=['儿童(5-12)', '青少年 (13-18)', '成年人 (19-22)']
)
calculate_stats(train_t, 'Age Group')


# 创建 1×2 的画布
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 1. 绘制性别比例饼图
vc = train['Basic_Demos-Sex'].value_counts()  # 计算性别分布
counts = vc.values  # 获取各性别的样本数量
labels = ['男孩', '女孩']  # 'Male' 对应 '男孩'，'Female' 对应 '女孩'

axs[0].pie(counts, labels=labels, autopct="%1.1f%%", colors=['lightblue', 'coral'])  # 绘制饼图
axs[0].set_title('性别比例')  

# 2. 绘制性别年龄分布柱状图
for sex in range(2):
    ax = axs[1]  

    # 通过布尔索引筛选数据
    sex_filter = train['Basic_Demos-Sex'] == sex
    vc = train[sex_filter]['Basic_Demos-Age'].value_counts()  # 计算不同年龄的样本数量

    # 绘制柱状图
    ax.bar(vc.index, vc.values, color=['lightblue', 'coral'][sex], label=['男孩', '女孩'][sex])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    ax.set_ylabel('样本数量')
    ax.legend() 


plt.suptitle('性别与年龄分布', fontsize=16)
axs[1].set_xlabel('年龄')  


plt.tight_layout()
plt.show()


# 创建 1×3 的画布
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. SII 与年龄的关系（箱型图）
sns.boxplot(y=train['Basic_Demos-Age'], x=train_t['sii'], ax=axes[0], palette="Set3")
axes[0].set_title('SII 与年龄的关系')  # 设置标题
axes[0].set_ylabel('年龄')  # 设置 y 轴标签
axes[0].set_xlabel('SII 等级')  # 设置 x 轴标签

# 2. 不同年龄组的完整 PCIAT 评分（箱型图）
sns.boxplot(
    x='Age Group', y='PCIAT-PCIAT_Total',
    data=train_t, palette="Set3", ax=axes[1]
)
axes[1].set_title('不同年龄组的完整 PCIAT 评分')  # 设置标题
axes[1].set_ylabel('完整答题的 PCIAT 总分')  # 设置 y 轴标签
axes[1].set_xlabel('年龄组')  # 设置 x 轴标签

# 3. 不同性别的 PCIAT 总分分布（直方图）
sns.histplot(
    data=train_t, x='PCIAT-PCIAT_Total',
    hue='Basic_Demos-Sex', multiple='stack',
    palette="Set3", bins=20, ax=axes[2]
)
axes[2].set_title('不同性别的 PCIAT 总分分布')  # 设置标题
axes[2].set_xlabel('完整答题的 PCIAT 总分')  # 设置 x 轴标签
axes[2].set_ylabel('频率')  # 设置 y 轴标签

# 调整布局，避免重叠
plt.tight_layout()
plt.show()


# 计算不同年龄组的 SII 分布
stats = train_t.groupby(['Age Group', 'sii']).size().unstack(fill_value=0)

# 创建 1×N 的画布，每个年龄组一个饼图
fig, axes = plt.subplots(1, len(stats), figsize=(18, 5))

# 遍历每个年龄组，绘制 SII 分布饼图
for i, age_group in enumerate(stats.index):
    group_counts = stats.loc[age_group] / stats.loc[age_group].sum()  # 计算各 SII 级别的占比
    axes[i].pie(
        group_counts, labels=group_counts.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette("Set3"),
        labeldistance=1.05, pctdistance=0.80
    )
    axes[i].set_title(f'{age_group} 的 SII 分布') 
    axes[i].axis('equal') 

plt.tight_layout()
plt.show()

# 计算不同性别的 SII 分布
stats = train_t.groupby(['Basic_Demos-Sex', 'sii']).size().unstack(fill_value=0)

# 创建 1×2 的画布，每个性别一个饼图
fig, axes = plt.subplots(1, len(stats), figsize=(18, 5))

# 映射性别 0 和 1 为 "男性" 和 "女性"
sex_map = {0: "男性", 1: "女性"}

# 遍历每个性别，绘制 SII 分布饼图
for i, sex in enumerate(stats.index):
    group_counts = stats.loc[sex] / stats.loc[sex].sum()  # 计算各 SII 级别的占比
    axes[i].pie(
        group_counts, labels=group_counts.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette("Set3"),
        labeldistance=1.05, pctdistance=0.80
    )
    axes[i].set_title(f'{sex_map[sex]} 的 SII 分布')
    axes[i].axis('equal')  


plt.tight_layout()
plt.show()


# 选取连续性变量，计算相关矩阵
continuous_columns = [
    'Basic_Demos-Age', 'Physical-BMI', 'Physical-Height', 'Physical-Weight',
    'FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL',
    'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_SRL', 'FGC-FGC_SRR',
    'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE',
    'BIA-BIA_ECW', 'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI',
    'BIA-BIA_Fat', 'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST',
    'BIA-BIA_SMM', 'BIA-BIA_TBW', 'PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total', 'sii'
]

correlation_matrix = train_t[continuous_columns].corr()

# 绘制相关性热图
plt.figure(figsize=(18, 15))  
sns.heatmap(
    correlation_matrix, 
    annot=True,  
    fmt=".2f",  
    cmap="coolwarm", 
    center=0,  
    annot_kws={"size": 8},  
    cbar_kws={"shrink": 0.8} 
)

plt.xticks(rotation=90, fontsize=10)  
plt.yticks(rotation=0, fontsize=10)   
plt.title("连续变量与 SII 的相关性热图", fontsize=18, color="#004080") 
plt.tight_layout() 

plt.show()


plt.figure(figsize=(18, 5))

# 1. 年龄与体重的关系（散点图）
plt.subplot(1, 3, 1)  
sns.scatterplot(x='Basic_Demos-Age', y='Physical-Weight', data=train)  
plt.title('年龄与体重的关系')  
plt.xlabel('年龄')
plt.ylabel('体重 (kg)') 

# 2. 年龄与身高的关系（散点图）
plt.subplot(1, 3, 2)  
sns.scatterplot(x='Basic_Demos-Age', y='Physical-Height', data=train)  
plt.title('年龄与身高的关系')  
plt.xlabel('年龄')  
plt.ylabel('身高 (cm)')  

plt.tight_layout()  
plt.show()


# 定义 FGC 类相关变量
FGC_columns = ['FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL',
    'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_SRL', 'FGC-FGC_SRR']

# 获取所有年龄组
age_groups = train_t['Age Group'].unique()

# 创建 1×3 的画布，设置共享 y 轴
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# 遍历每个年龄组，绘制相关性热力图
for i, age_group in enumerate(age_groups):
    group_data = train_t[train_t['Age Group'] == age_group]  # 筛选当前年龄组的数据
    corr_matrix = group_data[FGC_columns + ['PCIAT-PCIAT_Total', 'Basic_Demos-Age']].corr()  # 计算相关矩阵
    
    # 绘制相关性热力图
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.1f',
                vmin=-1, vmax=1, ax=axes[i], cbar=i == 0)  
    
    axes[i].set_title(f'{age_group} 组的相关性热力图') 

plt.tight_layout()
plt.show()


# 定义 BIA 类相关变量
BIA_columns = ['BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE',
    'BIA-BIA_ECW', 'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI',
    'BIA-BIA_Fat', 'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST',
    'BIA-BIA_SMM', 'BIA-BIA_TBW']

plt.figure(figsize=(24, 20))

# 遍历每个 BIA 变量，绘制直方图
for idx, col in enumerate(BIA_columns):
    plt.subplot(4, 4, idx + 1)  # 创建 4×4 的子图布局
    sns.histplot(train_t[col].dropna(), bins=20, kde=True)  # 绘制直方图，去除缺失值
    plt.title(f'{col} 的分布')  
    plt.xlabel('数值')  

plt.tight_layout()
plt.show()


# 读取数据字典文件
data_dictionary = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv')

# 筛选出数据类型为整数（int），但不包含类别型（categorical）的变量
integer_columns = list(data_dictionary[
    data_dictionary['Type'].str.contains('int') & ~data_dictionary['Type'].str.contains('categorical')
]["Field"].unique())

# 计算每个整数变量的统计信息
column_stats = {
    col: {
        '缺失值比例': train_t[col].isna().mean() * 100,  # 计算缺失值占比（百分比）
        '唯一值数量': train_t[col].nunique(),  # 计算该列的唯一值数量
        '均值': train_t[col].mean(),  # 计算均值
        '标准差': train_t[col].std(),  # 计算标准差
        '最小值': train_t[col].min(),  # 计算最小值
        '25%分位数': train_t[col].quantile(0.25),  # 计算 25% 分位数（第一四分位数）
        '中位数': train_t[col].median(),  # 计算中位数
        '75%分位数': train_t[col].quantile(0.75),  # 计算 75% 分位数（第三四分位数）
        '最大值': train_t[col].max()  # 计算最大值
    } 
    for col in integer_columns
}

# 将统计信息转换为 DataFrame 以便展示
stats_df = pd.DataFrame.from_dict(column_stats, orient='index').reset_index()
stats_df.columns = [
    '变量名', '缺失值比例', '唯一值数量', 
    '均值', '标准差', '最小值', '25%分位数', '中位数', '75%分位数', '最大值'
]

# 按缺失值比例降序排序
stats_df = stats_df.sort_values(by='缺失值比例', ascending=False).reset_index(drop=True)

# 显示统计信息 DataFrame
stats_df


# 创建 1×2 的画布
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. SII 与睡眠障碍评分的散点图
axes[0].scatter(train['sii'], train['SDS-SDS_Total_Raw'], color='royalblue', alpha=0.7)  # 绘制散点图
axes[0].set_xlabel('SII 等级') 
axes[0].set_ylabel('睡眠障碍评分')  
axes[0].set_title('SII 与睡眠障碍评分的散点图')  
axes[0].grid(True)  

# 2. 睡眠障碍评分的小提琴图
sns.violinplot(x='sii', y='SDS-SDS_Total_Raw', data=train, color='royalblue', ax=axes[1])  # 绘制小提琴图
axes[1].set_xlabel('SII 等级')  
axes[1].set_ylabel('睡眠障碍评分')  
axes[1].set_title('睡眠障碍评分的分布（小提琴图）')  


plt.tight_layout()
plt.show()


def clean_features(df):
    # 移除不合理的数值

    # 限制握力测试数据范围（FGC-FGC_GSND 和 FGC-FGC_GSD）
    df[['FGC-FGC_GSND', 'FGC-FGC_GSD']] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].clip(lower=9, upper=60)

    # 处理不合理的体脂率数据（BIA-BIA_Fat）
    df["BIA-BIA_Fat"] = np.where(df["BIA-BIA_Fat"] < 5, np.nan, df["BIA-BIA_Fat"])  # 低于 5% 设为 NaN
    df["BIA-BIA_Fat"] = np.where(df["BIA-BIA_Fat"] > 60, np.nan, df["BIA-BIA_Fat"])  # 高于 60% 设为 NaN

    # 处理基础代谢率（BIA-BIA_BMR）
    df["BIA-BIA_BMR"] = np.where(df["BIA-BIA_BMR"] > 4000, np.nan, df["BIA-BIA_BMR"])  # 超过 4000 设为 NaN

    # 处理每日能量消耗（BIA-BIA_DEE）
    df["BIA-BIA_DEE"] = np.where(df["BIA-BIA_DEE"] > 8000, np.nan, df["BIA-BIA_DEE"])  # 超过 8000 设为 NaN

    # 处理骨矿物质含量（BIA-BIA_BMC）
    df["BIA-BIA_BMC"] = np.where(df["BIA-BIA_BMC"] <= 0, np.nan, df["BIA-BIA_BMC"])  # 小于等于 0 设为 NaN
    df["BIA-BIA_BMC"] = np.where(df["BIA-BIA_BMC"] > 10, np.nan, df["BIA-BIA_BMC"])  # 超过 10 设为 NaN

    # 处理去脂体重（BIA-BIA_FFM）
    df["BIA-BIA_FFM"] = np.where(df["BIA-BIA_FFM"] <= 0, np.nan, df["BIA-BIA_FFM"])  # 小于等于 0 设为 NaN
    df["BIA-BIA_FFM"] = np.where(df["BIA-BIA_FFM"] > 300, np.nan, df["BIA-BIA_FFM"])  # 超过 300 设为 NaN

    # 处理脂肪质量指数（BIA-BIA_FMI）
    df["BIA-BIA_FMI"] = np.where(df["BIA-BIA_FMI"] < 0, np.nan, df["BIA-BIA_FMI"])  # 小于 0 设为 NaN

    # 处理细胞外水分（BIA-BIA_ECW）
    df["BIA-BIA_ECW"] = np.where(df["BIA-BIA_ECW"] > 100, np.nan, df["BIA-BIA_ECW"])  # 超过 100 设为 NaN

    # 处理瘦干质量（BIA-BIA_LDM）
    df["BIA-BIA_LDM"] = np.where(df["BIA-BIA_LDM"] > 100, np.nan, df["BIA-BIA_LDM"])  # 超过 100 设为 NaN

    # 处理瘦软组织（BIA-BIA_LST）
    df["BIA-BIA_LST"] = np.where(df["BIA-BIA_LST"] > 300, np.nan, df["BIA-BIA_LST"])  # 超过 300 设为 NaN

    # 处理骨骼肌质量（BIA-BIA_SMM）
    df["BIA-BIA_SMM"] = np.where(df["BIA-BIA_SMM"] > 300, np.nan, df["BIA-BIA_SMM"])  # 超过 300 设为 NaN

    # 处理总身体水分（BIA-BIA_TBW）
    df["BIA-BIA_TBW"] = np.where(df["BIA-BIA_TBW"] > 300, np.nan, df["BIA-BIA_TBW"])  # 超过 300 设为 NaN

    return df

# 清理训练集和测试集数据
train = clean_features(train)
test = clean_features(test)


def perform_pca(train, test, n_components=None, random_state=42):
    # 初始化 PCA 模型
    pca = PCA(n_components=n_components, random_state=random_state)
    
    # 在训练数据上拟合 PCA 并转换数据
    train_pca = pca.fit_transform(train)
    
    # 在测试数据上应用相同的 PCA 变换
    test_pca = pca.transform(test)
    
    # 获取各主成分的解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"各主成分的解释方差比:\n {explained_variance_ratio}")
    print(f"总解释方差: {np.sum(explained_variance_ratio)}")
    
    # 将转换后的数据转换为 DataFrame，并为主成分命名
    train_pca_df = pd.DataFrame(train_pca, columns=[f'PC_{i+1}' for i in range(train_pca.shape[1])])
    test_pca_df = pd.DataFrame(test_pca, columns=[f'PC_{i+1}' for i in range(test_pca.shape[1])])
    
    return train_pca_df, test_pca_df, pca


# 移除 'id' 列，仅保留时序数据
df_train = train_ts.drop('id', axis=1)
df_test = test_ts.drop('id', axis=1)

# 在执行PCA之前进行标准化处理
scaler = StandardScaler()
df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)  # 训练数据标准化
df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)  # 测试数据标准化（使用相同的缩放参数）

# 在PCA之前对时序数据进行均值填充（处理缺失值）
for c in df_train.columns:
    m = np.mean(df_train[c])  # 计算该列的均值
    df_train[c].fillna(m, inplace=True)  # 用均值填充训练数据中的缺失值
    df_test[c].fillna(m, inplace=True)  # 用训练数据的均值填充测试数据中的缺失值

# 输出训练数据的形状
print(df_train.shape)

# 执行 PCA 降维，选择15个主成分
df_train_pca, df_test_pca, pca = perform_pca(df_train, df_test, n_components=15, random_state=SEED)

# 重新添加 'id' 列，以便后续合并
df_train_pca['id'] = train_ts['id']
df_test_pca['id'] = test_ts['id']

# 将 PCA 处理后的数据合并回原始训练集和测试集
train = pd.merge(train, df_train_pca, how="left", on='id')
test = pd.merge(test, df_test_pca, how="left", on='id')

# 输出最终训练数据的形状
train.shape


def feature_engineering(df):
    # 1. 删除季节相关变量
    season_cols = [col for col in df.columns if 'Season' in col]
    df = df.drop(season_cols, axis=1)  # 删除所有包含 "Season" 的列
    
    # 2. 创建年龄组（Age Group）
    def assign_group(age):
        thresholds = [5, 6, 7, 8, 10, 12, 14, 17, 22]  # 年龄分组阈值
        for i, j in enumerate(thresholds):
            if age <= j:
                return i  # 根据年龄分配组别
        return np.nan
    
    df["group"] = df['Basic_Demos-Age'].apply(assign_group)  # 应用年龄分组函数
    
    # 3. 归一化 BMI
    BMI_map = {0: 16.3, 1: 15.9, 2: 16.1, 3: 16.8, 4: 17.3, 5: 19.2, 6: 20.2, 7: 22.3, 8: 23.6}
    df['BMI_mean_norm'] = df[['Physical-BMI', 'BIA-BIA_BMI']].mean(axis=1) / df["group"].map(BMI_map)
    
    # 4. FGC 区域特征聚合
    zones = ['FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
             'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
             'FGC-FGC_TL_Zone']
    
    df['FGC_Zones_mean'] = df[zones].mean(axis=1)  # 计算 FGC 区域均值
    df['FGC_Zones_min'] = df[zones].min(axis=1)  # 计算 FGC 区域最小值
    df['FGC_Zones_max'] = df[zones].max(axis=1)  # 计算 FGC 区域最大值
    
    # 5. 握力测试归一化
    GSD_max_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 16.2, 5: 19.9, 6: 26.1, 7: 31.3, 8: 35.4}
    GSD_min_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 14.4, 5: 17.8, 6: 23.4, 7: 27.8, 8: 31.1}
    
    df['GS_max'] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].max(axis=1) / df["group"].map(GSD_max_map)
    df['GS_min'] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].min(axis=1) / df["group"].map(GSD_min_map)
    
    # 6. 仰卧起坐、俯卧撑、躯干抬升归一化
    cu_map = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 10.0, 5: 14.0, 6: 20.0, 7: 20.0, 8: 20.0}
    pu_map = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 7.0, 6: 8.0, 7: 10.0, 8: 14.0}
    tl_map = {0: 8.0, 1: 8.0, 2: 8.0, 3: 9.0, 4: 9.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0}
    
    df["CU_norm"] = df['FGC-FGC_CU'] / df['group'].map(cu_map)
    df["PU_norm"] = df['FGC-FGC_PU'] / df['group'].map(pu_map)
    df["TL_norm"] = df['FGC-FGC_TL'] / df['group'].map(tl_map)
    
    # 7. 伸展测试
    df["SR_min"] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].min(axis=1)  # 计算最小值
    df["SR_max"] = df[['FGC-FGC_SRL', 'FGC-FGC_SRR']].max(axis=1)  # 计算最大值

    # 8. BIA 特征归一化
    # 能量消耗归一化
    bmr_map = {0: 934.0, 1: 941.0, 2: 999.0, 3: 1048.0, 4: 1283.0, 5: 1255.0, 6: 1481.0, 7: 1519.0, 8: 1650.0}
    dee_map = {0: 1471.0, 1: 1508.0, 2: 1640.0, 3: 1735.0, 4: 2132.0, 5: 2121.0, 6: 2528.0, 7: 2566.0, 8: 2793.0}
    
    df["BMR_norm"] = df["BIA-BIA_BMR"] / df["group"].map(bmr_map)
    df["DEE_norm"] = df["BIA-BIA_DEE"] / df["group"].map(dee_map)
    df["DEE_BMR"] = df["BIA-BIA_DEE"] - df["BIA-BIA_BMR"]  # 计算每日能量消耗与基础代谢率的差值

    # 9. 去脂体重归一化
    ffm_map = {0: 42.0, 1: 43.0, 2: 49.0, 3: 54.0, 4: 60.0, 5: 76.0, 6: 94.0, 7: 104.0, 8: 111.0}
    df["FFM_norm"] = df["BIA-BIA_FFM"] / df["group"].map(ffm_map)

    # 10. 细胞外水分与细胞内水分比值
    df["ICW_ECW"] = df["BIA-BIA_ECW"] / df["BIA-BIA_ICW"]
    
    # 11. 删除冗余特征
    drop_feats = ['FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
                  'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone',
                  'Physical-BMI', 'BIA-BIA_BMI', 'FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL', 'FGC-FGC_SRL', 'FGC-FGC_SRR',
                 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_Frame_num', "BIA-BIA_FFM"]
    
    df = df.drop(drop_feats, axis=1)  # 删除不需要的特征
    
    return df
# 实施特征工程
train = feature_engineering(train)
test = feature_engineering(test)


def bin_data(train, test, columns, n_bins=10):
   
    # 1. 合并训练集和测试集，以确保分箱边界一致**
    combined = pd.concat([train, test], axis=0)
    
    # 2. 计算每个特征的分箱边界**
    bin_edges = {}
    for col in columns:
        # 使用 `qcut` 计算分位数分箱边界
        edges = pd.qcut(combined[col], n_bins, retbins=True, labels=range(n_bins), duplicates="drop")[1]
        bin_edges[col] = edges  # 存储分箱边界
    
    # 3. 在训练集和测试集中应用相同的分箱边界
    for col, edges in bin_edges.items():
        train[col] = pd.cut(
            train[col], bins=edges, labels=range(len(edges) - 1), include_lowest=True
        ).astype(float)  # 训练集分箱
        test[col] = pd.cut(
            test[col], bins=edges, labels=range(len(edges) - 1), include_lowest=True
        ).astype(float)  # 测试集分箱
    
    return train, test

# 需要进行分箱的特征列表
columns_to_bin = [
    "PAQ_A-PAQ_A_Total", "BMR_norm", "DEE_norm", "GS_min", "GS_max", "BIA-BIA_FFMI", 
    "BIA-BIA_BMC", "Physical-HeartRate", "BIA-BIA_ICW", "Fitness_Endurance-Time_Sec", 
    "BIA-BIA_LDM", "BIA-BIA_SMM", "BIA-BIA_TBW", "DEE_BMR", "ICW_ECW"
]

# 对训练集和测试集的指定特征进行分箱
train, test = bin_data(train, test, columns_to_bin, n_bins=10)


# 1. 需要排除的特征（因为测试集中没有这些变量）
exclude = ['PCIAT-Season', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03',
           'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07',
           'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11',
           'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15',
           'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19',
           'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total', 'sii', 'id']

# 2. 目标变量
y_model = "PCIAT-PCIAT_Total"  # 预测模型的目标变量（PCIAT 总分）
y_comp = "sii"  # 竞赛目标变量（SII 指数）

# 3. 选取用于训练的特征
features = [f for f in train.columns if f not in exclude]  # 仅保留未被排除的特征

# 4. 处理类别型特征
cat_c = []  # 目前没有需要处理的类别型特征

# 5. 映射类别型特征（如果有的话）
for col in cat_c:
    a_map = {}  # 创建映射字典
    all_unique = set(train[col].unique()) | set(test[col].unique())  # 获取训练集和测试集中的所有唯一值
    for i, value in enumerate(all_unique):
        a_map[value] = i  # 为每个类别分配一个数值

    train[col] = train[col].map(a_map)  # 在训练集中应用映射
    test[col] = test[col].map(a_map)  # 在测试集中应用映射

# 6. 仅保留 `sii` 目标变量非空的行
train = train[train["sii"].notna()]  # 过滤掉 `sii` 为空的样本

# 7. 输出训练集的形状
train.shape


# 使用机器学习模型进行缺失值填充
class Impute_With_Model:
    def __init__(self, na_frac=0.5, min_samples=0):
        self.model_dict = {}  # 存储用于填充的模型
        self.mean_dict = {}  # 存储每个特征的均值（用于均值填充）
        self.features = None  # 需要填充的特征列表
        self.na_frac = na_frac  # 允许的最大缺失值比例
        self.min_samples = min_samples  # 训练模型所需的最小样本数
    
    # 选择用于填充的特征（仅保留缺失比例 <= na_frac 的特征）    
    def find_features(self, data, feature, tmp_features):
        missing_rows = data[feature].isna()  # 找出目标特征缺失的行
        na_fraction = data[missing_rows][tmp_features].isna().mean(axis=0)  # 计算候选特征的缺失比例
        valid_features = np.array(tmp_features)[na_fraction <= self.na_frac]  # 仅保留缺失比例 <= na_frac 的特征
        return valid_features
    
    # 训练填充模型
    def fit_models(self, model, data, features):
        self.features = features
        n_data = data.shape[0]
        
        # 计算每个特征的均值（用于均值填充）
        for feature in features:
            self.mean_dict[feature] = np.mean(data[feature])
        
        # 遍历所有特征，训练填充模型
        for feature in tqdm(features):
            # 仅对存在缺失值的特征进行填充
            if data[feature].isna().sum() > 0:
                model_clone = clone(model)  # 复制模型
                X = data[data[feature].notna()].copy()  # 仅使用非缺失值的数据进行训练
                
                # 选择用于填充的特征
                tmp_features = [f for f in features if f != feature]
                tmp_features = self.find_features(data, feature, tmp_features)
                
                if len(tmp_features) >= 1 and X.shape[0] > self.min_samples:
                    # 仅在特征足够多且样本数足够时训练模型
                    for f in tmp_features:
                        X[f] = X[f].fillna(self.mean_dict[f])  # 用均值填充缺失值
                    model_clone.fit(X[tmp_features], X[feature])  # 训练模型
                    
                    # 存储模型及其使用的特征
                    self.model_dict[feature] = (model_clone, tmp_features.copy())
                else:
                    # 如果特征或样本数不足，则使用均值填充
                    self.model_dict[feature] = ("mean", np.mean(data[feature]))
            
    # 使用训练好的模型填充缺失值
    def impute(self, data):
        
        imputed_data = data.copy()
        
        # 遍历所有填充模型
        for feature, model in self.model_dict.items():
            missing_rows = imputed_data[feature].isna()  # 找出需要填充的行
            
            if missing_rows.any():
                if model[0] == "mean":
                    # 使用均值填充
                    imputed_data[feature].fillna(model[1], inplace=True)
                else:
                    # 使用机器学习模型填充
                    tmp_features = [f for f in self.features if f != feature]
                    X_missing = data.loc[missing_rows, tmp_features].copy()
                    
                    # 用均值填充缺失值
                    for f in tmp_features:
                        X_missing[f] = X_missing[f].fillna(self.mean_dict[f])
                    
                    # 预测缺失值并填充
                    imputed_data.loc[missing_rows, feature] = model[0].predict(X_missing[model[1]])
        
        return imputed_data


# 显示缺失比例 > 30% 的前 60 个特征
missing = pd.DataFrame(train.isna().sum() / len(train))
missing[missing[0] > 0.3][:60]


# 训练填充模型
model = LassoCV(cv=5, random_state=SEED)  # 使用 LassoCV 作为填充模型
imputer = Impute_With_Model(na_frac=0.4)  # 允许的最大缺失比例为 40%
imputer.fit_models(model, train, features)  # 训练填充模型

# 使用填充模型填充训练集和测试集
train = imputer.impute(train)
test = imputer.impute(test)


def calculate_weights(series):
    # 1. 对目标变量进行分箱（将数据划分为 10 个区间）
    bins = pd.cut(series, bins=10, labels=False)  # 使用 `pd.cut` 将数据分成 10 个区间
    
    # 2. 计算每个区间的样本数量
    weights = bins.value_counts().reset_index()  # 统计每个区间的样本数量
    weights.columns = ['target_bins', 'count']  # 重命名列
    
    # 3. 计算权重（样本数量的倒数）
    weights['count'] = 1 / weights['count']  # 计算每个区间的权重（样本数量越少，权重越大）
    
    # 4. 创建权重映射表
    weight_map = weights.set_index('target_bins')['count'].to_dict()  # 将权重映射为字典
    
    # 5. 将权重映射到原始数据
    weights = bins.map(weight_map)  # 根据分箱结果为每个样本分配权重
    
    # 6. 归一化权重（确保均值为 1）
    return weights / weights.mean()  # 归一化权重，使其均值为 1


def round_with_thresholds(raw_preds, thresholds):
    # 根据给定的阈值，将连续预测值转换为离散类别
    return np.where(raw_preds < thresholds[0], int(0),
                    np.where(raw_preds < thresholds[1], int(1),
                             np.where(raw_preds < thresholds[2], int(2), int(3))))

def optimize_thresholds(y_true, raw_preds, start_vals=[0.5, 1.5, 2.5]):
    # 通过优化寻找最佳阈值，使 Cohen's Kappa 分数最大化
    def fun(thresholds, y_true, raw_preds):
        # 计算当前阈值下的 Cohen's Kappa 分数（取负值用于最小化）
        rounded_preds = round_with_thresholds(raw_preds, thresholds)  # 根据当前阈值转换预测值
        return -cohen_kappa_score(y_true, rounded_preds, weights='quadratic')  # 计算 Kappa 分数（取负值用于优化）

    # 使用 Powell 方法优化阈值
    res = minimize(fun, x0=start_vals, args=(y_true, raw_preds), method='Powell')
    
    assert res.success  # 确保优化成功
    return res.x  # 返回优化后的阈值


def cross_validate(model_, data, features, score_col, index_col, cv, sample_weights=False, verbose=False):
    # 使用交叉验证评估模型，并计算每折的 Cohen's Kappa 分数，同时获取全数据集的预测值（Out-of-Fold 预测）。
    kappa_scores = [] 
    oof_score_predictions = np.zeros(len(data))  # 存储 Out-of-Fold 预测值

    score_to_index_thresholds = base_thresholds  
    thresholds = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(data, data[index_col])):
        X_train, X_val = data[features].iloc[train_idx], data[features].iloc[val_idx]
        y_train_score = data[score_col].iloc[train_idx] 
        y_train_index = data[index_col].iloc[train_idx]
        y_val_score = data[score_col].iloc[val_idx]      
        y_val_index = data[index_col].iloc[val_idx]     
        
         # 训练模型
        if sample_weights:
            weights = calculate_weights(y_train_score)
            model_.fit(X_train, y_train_score, sample_weight=weights)
        else:
            model_.fit(X_train, y_train_score)

        y_pred_train_score = model_.predict(X_train)
        y_pred_val_score = model_.predict(X_val)
        
        oof_score_predictions[val_idx] = y_pred_val_score 

         # 优化阈值
        t_1 = optimize_thresholds(y_train_index, y_pred_train_score, start_vals=base_thresholds)
        thresholds.append(t_1) # 存储当前折的最优阈值

        y_pred_val_index = round_with_thresholds(y_pred_val_score, t_1)

        kappa_score = cohen_kappa_score(y_val_index, y_pred_val_index, weights='quadratic')
        kappa_scores.append(kappa_score)
        
        if verbose:
            print(f"Fold {fold_idx}: Optimized Kappa Score = {kappa_score}")
    
    if verbose:
        print(f"## Mean CV Kappa Score: {np.mean(kappa_scores)} ##")
        print(f"## Std CV: {np.std(kappa_scores)}")
    
    return np.mean(kappa_scores), oof_score_predictions, thresholds

def n_cross_validate(model_, data, features, score_col, index_col, cv, seeds, sample_weights=False, verbose=False):
    
    # 进行多次交叉验证，每次使用不同的随机种子，以提高模型评估的稳定性。
    scores = []  # 存储每次交叉验证的 Kappa 分数
    
    for seed in seeds:
        cv.random_state = seed  # 设置交叉验证的随机种子
        score, oof, _ = cross_validate(model_, data, features, score_col, index_col, cv, sample_weights=True, verbose=False)
        scores.append(score)
    
    return score, oof


def cross_validate_tabnet(model, X, y_cont, y_disc, cv, max_epochs=100, patience=10, 
                          batch_size=1024, virtual_batch_size=128,verbose=True):
   
    kappa_list = []
    oof_preds = np.zeros(len(y_cont))
    thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_disc)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr_cont, y_val_cont = y_cont[train_idx], y_cont[val_idx]
        y_tr_disc, y_val_disc = y_disc[train_idx], y_disc[val_idx]
        
        # TabNet 需要目标为二维数组 (n_samples, 1)
        y_tr_cont = y_tr_cont.reshape(-1, 1)
        y_val_cont = y_val_cont.reshape(-1, 1)
        
        # 在本折上训练模型
        model.fit(
            X_tr, y_tr_cont,
            eval_set=[(X_val, y_val_cont)],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
        )
        
        # 在训练集上预测连续输出，并优化阈值
        preds_train = model.predict(X_tr)
        best_thr = optimize_thresholds(y_tr_disc, preds_train, start_vals=base_thresholds)
        thresholds.append(best_thr)
        
        # 在验证集上预测连续输出，经过阈值映射得到离散预测
        preds_val = model.predict(X_val)
        preds_val_disc = round_with_thresholds(preds_val, best_thr)
        
        kappa = cohen_kappa_score(y_val_disc, preds_val_disc, weights="quadratic")
        kappa_list.append(kappa)
        
        oof_preds[val_idx] = preds_val.flatten()
        
        if verbose:
            print(f"Fold {fold_idx}: Optimized Kappa Score = {kappa:.6f}")
            
    mean_kappa = np.mean(kappa_list)
    if verbose:
        print(f"## Mean CV Kappa Score for TabNet: {mean_kappa:.6f} ##")
    return mean_kappa, oof_preds, thresholds


def objective(trial, model_type, X, features, score_col, index_col, cv, sample_weights=False):
    # xgboost
    if model_type == 'xgboost':
        params = {
            'objective': trial.suggest_categorical('objective', ['reg:tweedie', 'reg:pseudohubererror']),
            'random_state': SEED,
            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 2, 30),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.02, 0.05),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1e-1),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e-1),
        }
        if params['objective'] == 'reg:tweedie':
            params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1, 2)
        model = XGBRegressor(**params, use_label_encoder=False)
    
    # lightgbm
    elif model_type == 'lightgbm':
        params = {
            'objective': trial.suggest_categorical('objective', ['poisson', 'tweedie', 'regression']),
            'random_state': SEED,
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100)
        }
        if params['objective'] == 'tweedie':
            params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1, 2)
        model = LGBMRegressor(**params)
    
    # catboost
    elif model_type == 'catboost':
        params = {
            'loss_function': trial.suggest_categorical('objective', ['Tweedie:variance_power=1.5', 
                                                                     'Poisson', 'RMSE']),
            'random_state': SEED,
            'iterations': trial.suggest_int('iterations', 100, 300),
            'depth': trial.suggest_int('depth', 2, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 1e-1),
            'subsample': trial.suggest_float('subsample', 0.5, 0.7),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 60),
        }
        model = CatBoostRegressor(**params, verbose=0)
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
        
    seeds = [random.randint(1, 10000) for _ in range(20)] # Seeds for repeated KFold

    score, _ = n_cross_validate(model, X, features, score_col, index_col, cv, seeds, sample_weights=True, verbose=True)

    return score

def run_optimization(X, features, score_col, index_col, model_type, n_trials=30, cv=None, sample_weights=False):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_type, X, features, score_col, index_col, cv, sample_weights), 
                   n_trials=n_trials)
    
    print(f"Best params for {model_type}: {study.best_params}")
    print(f"Best score: {study.best_value}")
    return study.best_params


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pytorch_tabnet.callbacks import Callback
import os
import torch
from pytorch_tabnet.callbacks import Callback
# 目标函数
def objective_tabnet_kappa(trial):
    # 超参数建议
    n_d = trial.suggest_int("n_d", 8, 32, step=8)
    n_a = n_d  # 通常 n_a 设置为和 n_d 相同
    n_steps = trial.suggest_int("n_steps", 3, 10)
    gamma = trial.suggest_float("gamma", 1.0, 2.0)
    lambda_sparse = trial.suggest_loguniform("lambda_sparse", 1e-6, 1e-3)
    lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
    beta1 = trial.suggest_float("beta1", 0.5, 0.9)
    
    # 构建 TabNetRegressor 模型
    model = TabNetRegressor(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_params=dict(lr=lr, betas=(beta1, 0.999), weight_decay=1e-5),
        mask_type=entmax,
        scheduler_params= dict(mode="min", patience=10, min_lr=1e-5, factor=0.5),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=0,
        seed=SEED,
    )
    
    # 使用 StratifiedKFold 进行 10 折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    mean_kappa, _, _ = cross_validate_tabnet(model, X_tab, y_cont, y_disc, cv, 
                                               max_epochs=100, patience=10,
                                               batch_size=1024, virtual_batch_size=128,
                                               verbose=False)
    
    return mean_kappa


# 选择需要排除的特征列表
exclude = [
    "PC_9", "PC_12", "Fitness_Endurance-Max_Stage", "Basic_Demos-Sex", "BMI_mean_norm", "PC_11", 
    "PC_8", "FGC_Zones_min", "Physical-Systolic_BP", "PC_4", "BIA-BIA_FMI", "BIA-BIA_LST", "Physical-Diastolic_BP", 
    "BIA-BIA_ECW", "Fitness_Endurance-Time_Mins", "PAQ_C-PAQ_C_Total", "PC_10", "BIA-BIA_Fat", "FFM_norm", "PC_14", "PC_7"
]

# 从原始特征列表中移除排除的特征
reduced_features = [f for f in features if f not in exclude]

# 打印最终选定的特征数量
print(len(reduced_features))


kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


# 由下面超参数搜索得到的超参数
lgb_params = {
    'objective': 'poisson', 
    'n_estimators': 295, 
    'max_depth': 4, 
    'learning_rate': 0.04505693066482616, 
    'subsample': 0.6042489155604022, 
    'colsample_bytree': 0.5021876720502726, 
    'min_data_in_leaf': 100
}

xgb_params = {
    'objective': 'reg:tweedie', 
    'num_parallel_tree': 12, 
    'n_estimators': 236, 
    'max_depth': 3, 
    'learning_rate': 0.04223740904479563, 
    'subsample': 0.7157264603586825, 
    'colsample_bytree': 0.7897918901977528, 
    'reg_alpha': 0.005335705058190553, 
    'reg_lambda': 0.0001897435318347022, 
    'tweedie_variance_power': 1.1393958601390142
}

cat_params = {
    'objective': 'RMSE', 
    'iterations': 238, 
    'depth': 4, 
    'learning_rate': 0.044523361750173816, 
    'l2_leaf_reg': 0.09301285673435761, 
    'subsample': 0.6902492783438681, 
    'bagging_temperature': 0.3007304771330199, 
    'random_strength': 3.562201626987314, 
    'min_data_in_leaf': 60
}

xtrees_params = {
    'n_estimators': 500, 
    'max_depth': 15, 
    'min_samples_leaf': 20, 
    'bootstrap': False
}


if optimize_params:
    # LightGBM Optimization
    lgb_params = run_optimization(train, lgb_features, 'PCIAT-PCIAT_Total', 'sii', 'lightgbm', n_trials=n_trials, cv=kf, sample_weights=True)

    # XGBoost Optimization
    xgb_params = run_optimization(train, xgb_features, 'PCIAT-PCIAT_Total', 'sii', 'xgboost', n_trials=n_trials, cv=kf, sample_weights=True)

    # CatBoost Optimization
    cat_params = run_optimization(train, cat_features, 'PCIAT-PCIAT_Total', 'sii', 'catboost', n_trials=n_trials, cv=kf, sample_weights=True)


# 构建模型
lgb_model = LGBMRegressor(**lgb_params, random_state=SEED, verbosity=-1)
xgb_model = XGBRegressor(**xgb_params, random_state=SEED, verbosity=0)
cat_model = CatBoostRegressor(**cat_params, random_state=SEED, verbose=0)
xtrees_model = ExtraTreesRegressor(**xtrees_params, random_state=SEED)

weights = calculate_weights(train['PCIAT-PCIAT_Total'])
# ------------------------------
# 交叉验证评估
# 用 cross_validate 来得到连续预测与最佳阈值

# lightgbm
score_lgb, oof_lgb, lgb_thresholds = cross_validate(
    lgb_model, train, reduced_features, 'PCIAT-PCIAT_Total', 'sii', kf, verbose=True, sample_weights=True
)

# xgboost
score_xgb, oof_xgb, xgb_thresholds = cross_validate(
    xgb_model, train, reduced_features, 'PCIAT-PCIAT_Total', 'sii', kf, verbose=True, sample_weights=True
)

# catboost
score_cat, oof_cat, cat_thresholds = cross_validate(
    cat_model, train, reduced_features, 'PCIAT-PCIAT_Total', 'sii', kf, verbose=True, sample_weights=True
)

# ExtraTree
score_xtrees, oof_xtrees, xtrees_thresholds = cross_validate(
    xtrees_model, train, reduced_features, 'PCIAT-PCIAT_Total', 'sii', kf, verbose=True, sample_weights=True
)

# 所有模型的平均Kappa得分
print(f'Overall Mean Kappa: {np.mean([score_lgb, score_xgb, score_cat, score_xtrees])}') # Ensemble score likely higher


# 由下面的超参数搜索得到
TabNet_Params = {'n_d': 8,
 'n_steps': 7,
 'gamma': 1.199858033402975,
 'lambda_sparse': 1.3135195995947674e-0,
 'optimizer_params':dict(lr=0.004546098269759013, betas=(0.8631657893341091,0.999), weight_decay=1e-5)          
}


# 构造用于 TabNet 的输入数据
X_tab = train[reduced_features].values            # 特征矩阵
y_cont = train[y_model].values            # 连续目标（回归值）
y_disc = train[y_comp].values.astype(int)   # 离散目标，用于阈值优化和分层

best_params = TabNet_Params
if optimize_params:
    # 使用 Optuna 调优，目标方向为 maximize Kappa 分数
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_tabnet_kappa, n_trials=30)
    
    print("Best TabNet Kappa:", study.best_value)
    print("Best hyperparameters:", study.best_params)
    # 利用最佳超参数构建最终模型并进行交叉验证
    best_params = study.best_params

final_tabnet_model = TabNetRegressor(**best_params,verbose=0,seed=SEED,)

# 这里用10折交叉验证保存 OOF 预测及各折阈值以供后续绘图
cv_full = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
tabnet_kappa_full, oof_tabnet, tabnet_thresholds = cross_validate_tabnet(final_tabnet_model, X_tab, y_cont, y_disc, cv_full, verbose=True)
print("Final Mean CV Kappa for TabNet:", tabnet_kappa_full)


lgb_thresholds_ens = np.mean(np.array(lgb_thresholds), axis=0)
xgb_thresholds_ens = np.mean(np.array(xgb_thresholds), axis=0)
cat_thresholds_ens = np.mean(np.array(cat_thresholds), axis=0)
xtrees_thresholds_ens = np.mean(np.array(xtrees_thresholds), axis=0)
tabnet_thresholds_ens = np.mean(np.array(tabnet_thresholds), axis=0)

# Apply the optimized thresholds to OOF predictions
oof_lgb_t = round_with_thresholds(oof_lgb, lgb_thresholds_ens)
print(f"LGBM optimized Kappa: {cohen_kappa_score(train['sii'], oof_lgb_t, weights='quadratic')}")
oof_xgb_t = round_with_thresholds(oof_xgb, xgb_thresholds_ens)
print(f"XGB optimized Kappa: {cohen_kappa_score(train['sii'], oof_xgb_t, weights='quadratic')}")
oof_cat_t = round_with_thresholds(oof_cat, cat_thresholds_ens)
print(f"CAT optimized Kappa: {cohen_kappa_score(train['sii'], oof_cat_t, weights='quadratic')}")
oof_xtrees_t = round_with_thresholds(oof_xtrees, xtrees_thresholds_ens)
print(f"ExtraTrees optimized Kappa: {cohen_kappa_score(train['sii'], oof_xtrees_t, weights='quadratic')}")
oof_tabnet_t = round_with_thresholds(oof_tabnet, tabnet_thresholds_ens)
print(f"TabNet optimized Kappa: {cohen_kappa_score(train['sii'], oof_tabnet_t, weights='quadratic')}")


oof_preds1 = np.array([oof_lgb, oof_xgb, oof_cat])
weighted_oof1 = np.average(oof_preds1, axis=0, weights= [0.3,0.3,0.2])
final_oof1 = np.round(weighted_oof1).astype(int)


# 离散型预测值的散点图
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 4, figsize=(16, 12))
# LightGBM
scatter1 = axes[0].scatter(train['PCIAT-PCIAT_Total'], oof_lgb, c=train["sii"], cmap="autumn", alpha=0.5)
axes[0].set_xlabel("True Score")
axes[0].set_ylabel("OOF Predictions - LGBM")
axes[0].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[0].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[0].set_aspect('equal', adjustable='box')

thresholds = [30, 50, 80]
for threshold in thresholds:
    axes[0].axhline(threshold, color="blue", linestyle="--", lw=1)
    axes[0].axvline(threshold, color="blue", linestyle="--", lw=1)
# XGBoost
scatter2 = axes[1].scatter(train['PCIAT-PCIAT_Total'], oof_xgb, c=train["sii"], cmap="autumn", alpha=0.5)
axes[1].set_xlabel("True Score")
axes[1].set_ylabel("OOF Predictions - XGB")
axes[1].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[1].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[1].set_aspect('equal', adjustable='box')

for threshold in thresholds:
    axes[1].axhline(threshold, color="blue", linestyle="--", lw=1)
    axes[1].axvline(threshold, color="blue", linestyle="--", lw=1)

# CatBoost    
scatter2 = axes[2].scatter(train['PCIAT-PCIAT_Total'], oof_cat, c=train["sii"], cmap="autumn", alpha=0.5)
axes[2].set_xlabel("True Score")
axes[2].set_ylabel("OOF Predictions - Cat")
axes[2].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[2].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[2].set_aspect('equal', adjustable='box')

for threshold in thresholds:
    axes[2].axhline(threshold, color="blue", linestyle="--", lw=1)
    axes[2].axvline(threshold, color="blue", linestyle="--", lw=1)

# 集成模型1
# CatBoost    
scatter3 = axes[3].scatter(train['PCIAT-PCIAT_Total'], final_oof1, c=train["sii"], cmap="autumn", alpha=0.5)
axes[3].set_xlabel("True Score")
axes[3].set_ylabel("OOF Predictions - Ensemble-1")
axes[3].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[3].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[3].set_aspect('equal', adjustable='box')

for threshold in thresholds:
    axes[3].axhline(threshold, color="blue", linestyle="--", lw=1)
    axes[3].axvline(threshold, color="blue", linestyle="--", lw=1)
plt.tight_layout()
plt.show()


# 集成模型1 ： lgb, xgb, cat
oof_preds1 = np.array([oof_lgb_t, oof_xgb_t, oof_cat_t])
weighted_oof1 = np.average(oof_preds1, axis=0, weights= [0.3,0.3,0.2])
final_oof1 = np.round(weighted_oof1).astype(int)
kappa_score1 = cohen_kappa_score(train["sii"], final_oof1, weights='quadratic')
print(f"Ensemble Kappa score: {kappa_score1}")


# 定义 FGC 类相关变量
FGC_columns = ['FGC-FGC_CU', 'FGC-FGC_PU', 'FGC-FGC_TL',
    'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_SRL', 'FGC-FGC_SRR']

# 获取所有年龄组
age_groups = train_t['Age Group'].unique()

# 创建 1×3 的画布，设置共享 y 轴
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# 遍历每个年龄组，绘制相关性热力图
for i, age_group in enumerate(age_groups):
    group_data = train_t[train_t['Age Group'] == age_group]  # 筛选当前年龄组的数据
    corr_matrix = group_data[FGC_columns + ['PCIAT-PCIAT_Total', 'Basic_Demos-Age']].corr()  # 计算相关矩阵
    
    # 绘制相关性热力图
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.1f',
                vmin=-1, vmax=1, ax=axes[i], cbar=i == 0)  
    
    axes[i].set_title(f'{age_group} 组的相关性热力图') 

plt.tight_layout()
plt.show()


oof_list = [oof_lgb_t, oof_xgb_t, oof_cat_t, final_oof1]
modelname = ['lgb', 'xgb', 'cat', 'ensemble-1']
# 分类型预测值热力图
fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)
for i in range(4):
    conf_matrix = confusion_matrix(train["sii"], oof_list[i])
    sns.set_theme(style="whitegrid")
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', ax=axes[i])
    axes[i].set_title('Confusion Matrix-{}'.format(modelname[i]), fontsize=16)
    axes[i].set_xlabel('Predicted', fontsize=12)
    axes[i].set_ylabel('True', fontsize=12)

plt.tight_layout()
plt.show()


oof_preds2 = np.array([oof_lgb, oof_xgb, oof_cat, oof_xtrees])
weighted_oof2 = np.average(oof_preds2, axis=0, weights= [0.3,0.3,0.2,0.2])
final_oof2 = np.round(weighted_oof2).astype(int)


# 离散型预测值的散点图
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(8, 6))
# 极端随机树   
scatter1 = axes[0].scatter(train['PCIAT-PCIAT_Total'], oof_xtrees, c=train["sii"], cmap="autumn", alpha=0.5)
axes[0].set_xlabel("True Score")
axes[0].set_ylabel("OOF Predictions - ExtraTrees")
axes[0].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[0].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[0].set_aspect('equal', adjustable='box')

for threshold in thresholds:
    axes[0].axhline(threshold, color="blue", linestyle="--", lw=1)
    axes[0].axvline(threshold, color="blue", linestyle="--", lw=1)
    
# 集成模型2 
scatter2 = axes[1].scatter(train['PCIAT-PCIAT_Total'], final_oof2, c=train["sii"], cmap="autumn", alpha=0.5)
axes[1].set_xlabel("True Score")
axes[1].set_ylabel("OOF Predictions - Ensemble-2")
axes[1].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[1].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[1].set_aspect('equal', adjustable='box')

for threshold in thresholds:
    axes[1].axhline(threshold, color="blue", linestyle="--", lw=1)
    axes[1].axvline(threshold, color="blue", linestyle="--", lw=1)

plt.tight_layout()
plt.show()


# 集成模型2 ： lgb, xgb, cat, extratrees 
oof_preds2 = np.array([oof_lgb_t, oof_xgb_t, oof_cat_t, oof_xtrees_t])
weighted_oof2 = np.average(oof_preds2, axis=0, weights= [0.3,0.3,0.2,0.2])
final_oof2 = np.round(weighted_oof2).astype(int)
kappa_score2 = cohen_kappa_score(train["sii"], final_oof2, weights='quadratic')
print(f"Ensemble Kappa score: {kappa_score2}")


oof_list = [oof_xtrees_t, final_oof2]
modelname = ['ExtraTrees', 'ensemble-2']
# 分类型预测值热力图
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
for i in range(2):
    conf_matrix = confusion_matrix(train["sii"], oof_list[i])
    sns.set_theme(style="whitegrid")
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', ax=axes[i])
    axes[i].set_title('Confusion Matrix-{}'.format(modelname[i]), fontsize=16)
    axes[i].set_xlabel('Predicted', fontsize=12)
    axes[i].set_ylabel('True', fontsize=12)

plt.tight_layout()
plt.show()


# 集成模型3 ： lgb, xgb, cat, tabnet
oof_preds3 = np.array([oof_lgb, oof_xgb, oof_cat, oof_tabnet])
weighted_oof3 = np.average(oof_preds3, axis=0, weights= [0.3,0.3,0.2,0.1])
final_oof3 = np.round(weighted_oof3).astype(int)


# 离散型预测值的散点图
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(8, 6))

# TabNet
scatter4 =axes[0].scatter(train['PCIAT-PCIAT_Total'], oof_tabnet, c=train["sii"], cmap="autumn", alpha=0.5)
axes[0].set_xlabel("True Score")
axes[0].set_ylabel("OOF Predictions - TabNet")
axes[0].set_ylim(0, np.max(train['PCIAT-PCIAT_Total']))
axes[0].set_xlim(0, np.max(train['PCIAT-PCIAT_Total']))
axes[0].set_aspect('equal', adjustable='box')
for thr in thresholds:
    axes[0].axhline(thr, color="blue", linestyle="--", lw=1)
    axes[0].axvline(thr, color="blue", linestyle="--", lw=1)
    
# 集成模型2 
scatter2 = axes[1].scatter(train['PCIAT-PCIAT_Total'], final_oof3, c=train["sii"], cmap="autumn", alpha=0.5)
axes[1].set_xlabel("True Score")
axes[1].set_ylabel("OOF Predictions - Ensemble-3")
axes[1].set_ylim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[1].set_xlim(0,np.max(train['PCIAT-PCIAT_Total']))
axes[1].set_aspect('equal', adjustable='box')

for threshold in thresholds:
    axes[1].axhline(threshold, color="blue", linestyle="--", lw=1)
    axes[1].axvline(threshold, color="blue", linestyle="--", lw=1)

plt.tight_layout()
plt.show()


# 集成模型3 ： lgb, xgb, cat, tabnet
oof_preds3 = np.array([oof_lgb_t, oof_xgb_t, oof_cat_t, oof_tabnet_t])
weighted_oof3 = np.average(oof_preds3, axis=0, weights= [0.3,0.3,0.2,0.1])
final_oof3 = np.round(weighted_oof3).astype(int)
kappa_score3 = cohen_kappa_score(train["sii"], final_oof3, weights='quadratic')
print(f"Ensemble Kappa score: {kappa_score3}")


oof_list = [oof_tabnet_t, final_oof3]
modelname = ['TabNet', 'ensemble-3']
# 分类型预测值热力图
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
for i in range(2):
    conf_matrix = confusion_matrix(train["sii"], oof_list[i])
    sns.set_theme(style="whitegrid")
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', ax=axes[i])
    axes[i].set_title('Confusion Matrix-{}'.format(modelname[i]), fontsize=16)
    axes[i].set_xlabel('Predicted', fontsize=12)
    axes[i].set_ylabel('True', fontsize=12)

plt.tight_layout()
plt.show()


# 集成模型3各模型的相关度
plt.figure(figsize=(8, 6))  
model_preds = pd.DataFrame({
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    'cat': oof_cat,
    'tabnet': oof_tabnet_t
})
corr_df = model_preds.corr()
sns.heatmap(corr_df, annot=True, cmap="autumn", cbar=False, linewidths=0.5, linecolor='black', )
plt.title("Correlation Between Models")
plt.tight_layout()
plt.show()

