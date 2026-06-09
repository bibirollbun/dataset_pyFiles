import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col = 0)


data.info()


data.describe()


data.dtypes


TARGET = 'Fertilizer Name'
QUAN_COLUMNS = [col for col in data.columns if data[col].dtype != 'object' and col != TARGET]
CAT_COLUMNS = [col for col in data.columns if data[col].dtype == 'object' and col != TARGET]

print('Target column:', TARGET)
print('Quantitative columns:', QUAN_COLUMNS)
print('Categorical columns:', CAT_COLUMNS)


def gini_coefficient(array):
    """
    Gini 계수를 계산합니다. 클래스별 샘플 수 또는 값의 배열을 입력으로 받습니다.

    Args:
        array (list or np.ndarray): 값의 리스트 (예: 클래스별 샘플 수, 소득 등)

    Returns:
        float: Gini Coefficient (0 = 완전 균형, 1 = 극단적 불균형)
    """
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        raise ValueError("Gini coefficient is not defined for negative values")

    if array.sum() == 0:
        return 0.0  # 모든 값이 0이면 균형으로 간주

    array = np.sort(array)  # 오름차순 정렬
    n = array.shape[0]
    cumulative = np.cumsum(array)
    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * array)) / (n * np.sum(array)) - (n + 1) / n
    return gini



def r2_score(X, y):
    """
    R^2 (결정계수)를 계산합니다. X는 독립 변수, y는 종속 변수입니다.

    Args:
        X (np.ndarray): 독립 변수 배열
        y (np.ndarray): 종속 변수 배열

    Returns:
        float: R^2 score
    """
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)
    return 1 - (ss_residual / ss_total)


# Color Palette for BoxPlot.
col_values = data[TARGET].unique()
col_colors = sns.color_palette("husl", len(col_values))


sns.histplot(data, x = TARGET, kde= True, stat = 'count')


gini = gini_coefficient(data[TARGET].value_counts().to_numpy())
print(f"Gini Coefficient for TARGET: {gini:.4f}")


# For Categorical colums : sns.histplot
fig, ax = plt.subplots(len(CAT_COLUMNS), 2, figsize=(20, 10))

for i, col in enumerate(CAT_COLUMNS):
    sns.histplot(data[col], ax=ax[i,0], kde=True)
    ax[i,0].set_title(f'Distribution of {col}')
    ax[i,0].set_xlabel(col)
    ax[i,0].set_ylabel('Frequency')

    cross_tab = pd.crosstab(data[col], data[TARGET])
    plt.figure(figsize=(16, 8))
    sns.heatmap(cross_tab, cmap='YlGnBu', annot=True, fmt='d', ax = ax[i,1])
    ax[i,1].set_title(f'Cross Tabulation of {col} and {TARGET}')
    ax[i,1].set_xlabel(col)
    ax[i,1].set_ylabel(TARGET)


for col in CAT_COLUMNS:
    gini = gini_coefficient(data[col].value_counts().to_numpy())
    print(f"Gini Coefficient for {col}: {gini:.4f}")


sns.pairplot(data[QUAN_COLUMNS + [TARGET]], hue=TARGET, diag_kind='kde', height=2.5, corner = True)


from itertools import combinations

isLinear = False

for col1, col2 in combinations(QUAN_COLUMNS, 2):
    r2 = r2_score(data[col1].values.reshape(-1, 1), data[col2].values)
    if r2 > 0.2:
        print(f"Linear correlation between {col1} and {col2}: R^2 = {r2:.4f}")
        isLinear = True

if not isLinear:
    print("No significant linear correlation found between quantitative columns.")


fig, ax = plt.subplots(len(QUAN_COLUMNS), 2, figsize=(20, 30))

for i, col in enumerate(QUAN_COLUMNS):

    sns.kdeplot(data,x = col, ax=ax[i,0], fill=True)
    ax[i,0].set_title(f'Distribution of {col}')
    ax[i,0].set_xlabel(col)
    ax[i,0].set_ylabel('Frequency')

    sns.boxplot(x = TARGET, y=col, data=data, ax=ax[i,1], palette = dict(zip(col_values, col_colors)))
    ax[i,1].set_title(f'{col} by {TARGET}')
    ax[i,1].set_xlabel(TARGET)
    ax[i,1].set_ylabel(col)


for col in QUAN_COLUMNS:
    data_bined = pd.cut(data[col], bins=100, labels = False)
    gini = gini_coefficient(data_bined.value_counts().to_numpy())
    print(f"Gini Coefficient for {col}: {gini:.4f}")


comb = list(combinations(CAT_COLUMNS, 2))
n = len(comb)
fig, ax = plt.subplots(n, 2, figsize=(40, 10 * n), squeeze=False)

for i, (col1, col2) in enumerate(comb):
    NEW_COL_NAME = f"{col1}_{col2}"
    data[NEW_COL_NAME] = data[col1].astype(str) + "_" + data[col2].astype(str)

    sns.histplot(data[NEW_COL_NAME], ax=ax[i, 0], kde=True)
    ax[i, 0].set_title(f'Distribution of {NEW_COL_NAME}')
    ax[i, 0].set_xlabel(NEW_COL_NAME)
    ax[i ,0].tick_params(axis = 'x', rotation = 45, labelsize = 9)
    ax[i, 0].set_ylabel('Frequency')

    cross_tab = pd.crosstab(data[NEW_COL_NAME], data[TARGET])
    plt.figure(figsize=(16, 8))
    sns.heatmap(cross_tab, cmap='YlGnBu', annot=True, fmt='d', ax = ax[i,1])
    ax[i,1].set_title(f'Cross Tabulation of {NEW_COL_NAME} and {TARGET}')
    ax[i,1].set_xlabel(NEW_COL_NAME)
    ax[i,1].set_ylabel(TARGET)



comb = list(combinations(QUAN_COLUMNS, 2))
n = len(comb)
fig, ax = plt.subplots(n, 2, figsize=(20, 5 * n), squeeze=False)

for i, (col1, col2) in enumerate(comb):
    NEW_COL_NAME = f"{col1}_{col2}"
    data[NEW_COL_NAME] = data[col1] * data[col2]

    sns.kdeplot(data[NEW_COL_NAME], ax=ax[i, 0], fill=True)
    ax[i, 0].set_title(f'Distribution of {NEW_COL_NAME}')
    ax[i, 0].set_xlabel(NEW_COL_NAME)
    ax[i, 0].set_ylabel('Frequency')

    sns.boxplot(x=TARGET, y=NEW_COL_NAME, data=data, ax=ax[i, 1], palette=dict(zip(col_values, col_colors)))
    ax[i, 1].set_title(f'{NEW_COL_NAME} by {TARGET}')
    ax[i, 1].set_xlabel(TARGET)
    ax[i, 1].set_ylabel(NEW_COL_NAME)



n = len(CAT_COLUMNS)
fig, ax = plt.subplots(n, 2, figsize=(40, 10 * n), squeeze=False)
data["BIN_Temparature"] = data['Temparature'].astype(int)

for i, col1 in enumerate(CAT_COLUMNS):
    NEW_COL_NAME = f"BIN_Temparature_{col1}"
    data[NEW_COL_NAME] = data["BIN_Temparature"].astype(str) + "_" + data[col1].astype(str)

    sns.histplot(data[NEW_COL_NAME], ax=ax[i, 0], kde=True)
    ax[i, 0].set_title(f'Distribution of {NEW_COL_NAME}')
    ax[i, 0].set_xlabel(NEW_COL_NAME)
    ax[i ,0].tick_params(axis = 'x', rotation = 45, labelsize = 9)
    ax[i, 0].set_ylabel('Frequency')

    cross_tab = pd.crosstab(data[NEW_COL_NAME], data[TARGET])
    plt.figure(figsize=(16, 8))
    sns.heatmap(cross_tab, cmap='YlGnBu', fmt='d', ax = ax[i,1])
    ax[i,1].set_title(f'Cross Tabulation of {NEW_COL_NAME} and {TARGET}')
    ax[i,1].set_xlabel(NEW_COL_NAME)
    ax[i,1].set_ylabel(TARGET)



data['Total_Nutrients'] = data['Nitrogen'] + data['Phosphorous'] + data['Potassium']
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.kdeplot(data['Total_Nutrients'], ax=ax[0], fill=True)
ax[0].set_title('Distribution of Total Nutrients')
ax[0].set_xlabel('Total Nutrients')
ax[0].set_ylabel('Frequency')

sns.boxplot(x=TARGET, y='Total_Nutrients', data=data, ax=ax[1], palette=dict(zip(col_values, col_colors)))
ax[1].set_title('Total Nutrients by TARGET')
ax[1].set_xlabel(TARGET)
ax[1].set_ylabel('Total Nutrients')


k = data['Phosphorous'].mean() # smoothing factor to 1. avoid division by zero 2. to avoid too extreme values

data['N_P'] = data['Nitrogen'] / (data['Phosphorous'] + k)  # Avoid division by zero

fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.kdeplot(data['N_P'], ax=ax[0], fill=True)
ax[0].set_title('Distribution of N/P Ratio')
ax[0].set_xlabel('N/P Ratio')
ax[0].set_ylabel('Frequency')

sns.boxplot(x=TARGET, y='N_P', data=data, ax=ax[1], palette=dict(zip(col_values, col_colors)))
ax[1].set_title('N/P Ratio by TARGET')
ax[1].set_xlabel(TARGET)
ax[1].set_ylabel('N/P Ratio')


k = data['Phosphorous'].mean() # smoothing factor to 1. avoid division by zero 2. to avoid too extreme values

data['K_P'] = data['Potassium'] / (data['Phosphorous'] + k)  # Avoid division by zero

fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.kdeplot(data['K_P'], ax=ax[0], fill=True)
ax[0].set_title('Distribution of K/P Ratio')
ax[0].set_xlabel('K/P Ratio')
ax[0].set_ylabel('Frequency')

sns.boxplot(x=TARGET, y='K_P', data=data, ax=ax[1], palette=dict(zip(col_values, col_colors)))
ax[1].set_title('K/P Ratio by TARGET')
ax[1].set_xlabel(TARGET)
ax[1].set_ylabel('K/P Ratio')


k = data['Moisture'].mean() # smoothing factor to 1. avoid division by zero 2. to avoid too extreme values

data['H_M'] = data['Humidity'] / (data['Moisture'] + k)  # Avoid division by zero

fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.kdeplot(data['H_M'], ax=ax[0], fill=True)
ax[0].set_title('Distribution of H/M Ratio')
ax[0].set_xlabel('H/M Ratio')
ax[0].set_ylabel('Frequency')

sns.boxplot(x=TARGET, y='H_M', data=data, ax=ax[1], palette=dict(zip(col_values, col_colors)))
ax[1].set_title('H/M Ratio by TARGET')
ax[1].set_xlabel(TARGET)
ax[1].set_ylabel('H/M Ratio')

