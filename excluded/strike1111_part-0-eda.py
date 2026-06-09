import numpy as np
import pandas as pd
import scipy.stats as stats

import seaborn as sns

import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


target = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx')
categorical = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx')
function = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv')
quantitative = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx')

target.shape, categorical.shape, function.shape,quantitative.shape


len(target['participant_id'].unique()),\
len(categorical['participant_id'].unique()),\
len(function['participant_id'].unique()),\
len(quantitative['participant_id'].unique())
#1213 subjects in total


#combine dataset by participant id
# Function to load all data from https://www.kaggle.com/code/olaflundstrom/wids-datathon-2025-adhd-analysis-notebook
def get_feats(mode='TRAIN'):
    
    # Load quantitative metadata
    feats = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_QUANTITATIVE_METADATA.xlsx")
    
    # Load categorical metadata with the correct filename depending on mode
    if mode == 'TRAIN':
        cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_CATEGORICAL_METADATA.xlsx")
    else:
        cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_CATEGORICAL.xlsx")
    
    # Merge categorical data
    feats = feats.merge(cate, on='participant_id', how='left')
    
    # Load functional connectome matrices
    func = pd.read_csv(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    feats = feats.merge(func, on='participant_id', how='left')
    
    # If training data, merge with solution file
    if mode == 'TRAIN':
        solution = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")
        feats = feats.merge(solution, on='participant_id', how='left')
    
    return feats


train = get_feats(mode='TRAIN')
test = get_feats(mode='TEST')

# Display the first few rows of the training data
train.head()


train.isna().sum()


missing_values = train.isnull().sum()

# 只显示有缺失值的列
missing_values = missing_values[missing_values > 0]

# 绘制柱状图
plt.figure(figsize=(10, 6))
missing_values.sort_values(ascending=False).plot(kind='bar')
plt.title('Missing Values Count')
plt.xlabel('Columns')
plt.ylabel('Count of Missing Values')
plt.xticks(rotation=0)
plt.show()
#Age at time of MRI scan
#Ethnicity of child
##"0= Not Hispanic or Latino
##1= Hispanic or Latino
##2= Decline to specify
##3= Unknown"


y = ['ADHD_Outcome', 'Sex_F']
train[y].value_counts()


# 计算各类别的分布
adhd_counts = train['ADHD_Outcome'].value_counts()
sex_counts = train['Sex_F'].value_counts()
adhd_sex_counts = train[y].value_counts()

# 选定统一的配色
colors = sns.color_palette("Blues", max(len(adhd_counts), len(sex_counts), len(adhd_sex_counts)))

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ADHD 饼图
axes[0].pie(adhd_counts, labels=adhd_counts.index, autopct='%1.1f%%', colors=colors[:len(adhd_counts)])
axes[0].set_title('ADHD Outcome Distribution')

# Sex 饼图
axes[1].pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', colors=colors[:len(sex_counts)])
axes[1].set_title('Sex Distribution (0=Male, 1=Female)')

# ADHD & Sex 联合分布饼图
axes[2].pie(adhd_sex_counts, labels=[f"ADHD:{a}, Sex_F:{s}" for a, s in adhd_sex_counts.index], 
            autopct='%1.1f%%', colors=colors[:len(adhd_sex_counts)])
axes[2].set_title('ADHD & Sex Joint Distribution')

plt.tight_layout()
plt.show()



# Create the contingency table based on observed counts
contingency_table = pd.DataFrame([[581, 250],  # ADHD=1: [Male (Sex_F=0), Female (Sex_F=1)]
                                  [216, 166]], # ADHD=0: [Male (Sex_F=0), Female (Sex_F=1)]
                                 columns=["Sex_F=0 (Male)", "Sex_F=1 (Female)"],
                                 index=["ADHD=1", "ADHD=0"])

# Perform Chi-Square Test of Independence
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print test results
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:\n", pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))


categorical.columns


# 设置图表风格
plt.style.use("ggplot")

# 选择需要绘制的分类变量（排除 participant_id）
categorical_columns = categorical.columns[1:]

# 创建多个子图
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))  # 3x3布局

# 遍历每个分类变量并绘制分布图
for col, ax in zip(categorical_columns, axes.flatten()):
    sns.countplot(data=categorical, x=col, ax=ax, palette="viridis")
    ax.set_title(f"Distribution of {col}")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

# 调整布局
plt.tight_layout()
plt.show()



quantitative.columns


train[quantitative.columns[1:]].describe()


# 设置图像大小
quantitative_vars = quantitative.columns[1:]
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 18))  

for var, ax in zip(quantitative_vars, axes.flatten()):
    if var in ["SDQ_SDQ_Externalizing", "SDQ_SDQ_Internalizing"]:
        min_val = int(train[var].min())
        max_val = int(train[var].max()) + 1
        bins = range(min_val, max_val + 1)  # 只使用整数作为 bins
        sns.histplot(train[var], kde=True, bins=bins, ax=ax, color='steelblue')
        ax.set_xticks(range(min_val, max_val, 2))  # X 轴只显示整数
    else:
        sns.histplot(train[var], kde=True, bins=30, ax=ax, color='steelblue')
    
    ax.set_title(f"Distribution of {var}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()


function.head()


function[function.columns[1:]].max().max(),function[function.columns[1:]].min().min()


fmri_matrix = function.drop(columns=["participant_id"])
fmri_values = fmri_matrix.values.flatten()
participant_strength = fmri_matrix.abs().mean(axis=1)
participant_strength.describe()


mean_connectivity = fmri_matrix.abs().mean(axis=1)  # 均值
median_connectivity = fmri_matrix.abs().median(axis=1)  # 中位数
std_connectivity = fmri_matrix.abs().std(axis=1)  # 标准差

# 合并数据
stats_df = pd.DataFrame({
    "Mean Connectivity Strength": mean_connectivity,
    "Median Connectivity Strength": median_connectivity,
    "STD Connectivity Strength": std_connectivity
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 绘制均值连接强度的 Boxplot
sns.boxplot(y=mean_connectivity, width=0.3, ax=axes[0])
axes[0].set_title("Mean Connectivity Strength")
axes[0].set_ylabel("Connectivity Strength")

# 绘制中位数连接强度的 Boxplot
sns.boxplot(y=median_connectivity, width=0.3, ax=axes[1])
axes[1].set_title("Median Connectivity Strength")
axes[1].set_ylabel("")  # 省略 Y 轴标签，避免重复

# 绘制标准差连接强度的 Boxplot
sns.boxplot(y=std_connectivity, width=0.3, ax=axes[2])
axes[2].set_title("STD Connectivity Strength")
axes[2].set_ylabel("")

# 调整布局
plt.tight_layout()
plt.show()


# 计算平均相关性（绝对值最高的脑区对）
top_connections = fmri_matrix.abs().mean(axis=0).sort_values(ascending=False)
top_connections.head(10)  # 显示前 10 个最强连接的脑区对


adhd_column = y[0]
categorical_columns = [
    "Basic_Demos_Enroll_Year", "Basic_Demos_Study_Site",
    "PreInt_Demos_Fam_Child_Ethnicity", "PreInt_Demos_Fam_Child_Race",
    "MRI_Track_Scan_Location", "Barratt_Barratt_P1_Edu",
    "Barratt_Barratt_P1_Occ", "Barratt_Barratt_P2_Edu",
    "Barratt_Barratt_P2_Occ"
]

# 设置子图布局
fig, axes = plt.subplots(len(categorical_columns), 2, figsize=(14, len(categorical_columns) * 4))

# 遍历分类变量，绘制 Stacked Bar Plot 和 Normalized Bar Plot
for i, col in enumerate(categorical_columns):
    # 计算每个分类变量在 ADHD 和非 ADHD 组中的数量
    adhd_counts = train.groupby([col, adhd_column]).size().unstack().fillna(0)
    
    # Stacked Bar Plot
    adhd_counts.plot(kind='bar', stacked=True, ax=axes[i, 0], colormap="viridis", alpha=0.8)
    axes[i, 0].set_title(f"Stacked Bar Plot: {col} vs ADHD")
    axes[i, 0].set_ylabel("Count")
    axes[i, 0].legend(title="ADHD", labels=["No", "Yes"])

    # Normalized Bar Plot
    normalized_counts = adhd_counts.div(adhd_counts.sum(axis=1), axis=0)
    normalized_counts.plot(kind='bar', stacked=True, ax=axes[i, 1], colormap="viridis", alpha=0.8)
    axes[i, 1].set_title(f"Normalized Bar Plot: {col} vs ADHD")
    axes[i, 1].set_ylabel("Proportion")

# 调整子图布局
plt.tight_layout()
plt.show()



# 确保数据中的分类变量存在
categorical_columns = [
    "Basic_Demos_Enroll_Year", "Basic_Demos_Study_Site",
    "PreInt_Demos_Fam_Child_Ethnicity", "PreInt_Demos_Fam_Child_Race",
    "MRI_Track_Scan_Location", "Barratt_Barratt_P1_Edu",
    "Barratt_Barratt_P1_Occ", "Barratt_Barratt_P2_Edu",
    "Barratt_Barratt_P2_Occ", y[0]
]
available_columns = [col for col in categorical_columns if col in train.columns]

# 仅保留存在的列
correlation_data = train[available_columns].copy()

# 对分类变量进行数值编码（每列单独编码）
correlation_data_encoded = correlation_data.apply(lambda x: pd.factorize(x)[0])

# 计算相关性矩阵
correlation_matrix = correlation_data_encoded.corr()

# 绘制相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Categorical Variables and ADHD")
plt.show()



# 确保 ADHD 变量存在
adhd_column = y[0]  # 确保数据中 ADHD 列名正确

# 数值变量列表
numerical_columns = [
    "EHQ_EHQ_Total", "ColorVision_CV_Score", "APQ_P_APQ_P_CP", "APQ_P_APQ_P_ID",
    "APQ_P_APQ_P_INV", "APQ_P_APQ_P_OPD", "APQ_P_APQ_P_PM", "APQ_P_APQ_P_PP",
    "SDQ_SDQ_Conduct_Problems", "SDQ_SDQ_Difficulties_Total", "SDQ_SDQ_Emotional_Problems",
    "SDQ_SDQ_Externalizing", "SDQ_SDQ_Generating_Impact", "SDQ_SDQ_Hyperactivity",
    "SDQ_SDQ_Internalizing", "SDQ_SDQ_Peer_Problems", "SDQ_SDQ_Prosocial",
    "MRI_Track_Age_at_Scan"
]

# 仅保留存在的列
available_numerical = [col for col in numerical_columns if col in train.columns and adhd_column in train.columns]
filtered_data = train[[adhd_column] + available_numerical].copy()

# 画多个 Boxplot 比较 ADHD (1) vs 非 ADHD (0)
fig, axes = plt.subplots(len(available_numerical) // 3 + 1, 3, figsize=(15, len(available_numerical) * 1.5))
axes = axes.flatten()

for i, col in enumerate(available_numerical):
    sns.boxplot(x=train[adhd_column], y=train[col], ax=axes[i])
    axes[i].set_title(f"Boxplot of {col}")
    axes[i].set_xlabel("ADHD (0 = No, 1 = Yes)")
    axes[i].set_ylabel(col)

# 隐藏多余的子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



# 计算相关性矩阵
correlation_matrix = filtered_data.corr()

# 绘制相关性热力图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix between Numerical Variables and ADHD")
plt.show()



adhd_column = y[0]  # 请确保你的数据中 ADHD 列名正确

# 计算 Mean / Median / Std Connectivity Strength
train["Mean_Connectivity"] = fmri_matrix.abs().mean(axis=1) 
train["Median_Connectivity"] = fmri_matrix.abs().median(axis=1)
train["Std_Connectivity"] = fmri_matrix.abs().std(axis=1) 

connectivity_metrics = ["Mean_Connectivity", "Median_Connectivity", "Std_Connectivity"]

train[connectivity_metrics+[y[0]]].groupby(y[0]).median()


# 创建 Boxplot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(connectivity_metrics):
    sns.boxplot(x=train[adhd_column], y=train[metric], ax=axes[i])
    axes[i].set_title(f"Boxplot of {metric} by ADHD Status")
    axes[i].set_xlabel("ADHD (0 = No, 1 = Yes)")
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.show()





