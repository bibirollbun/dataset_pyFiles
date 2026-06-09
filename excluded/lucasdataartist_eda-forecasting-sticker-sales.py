# Packages 
# Data Processing 
import numpy as np 
import pandas as pd 
# Visualization 
import matplotlib.pyplot as plt 
plt.rcParams['figure.dpi'] = 200 
import seaborn as sns 
# Statistics 
import math 
from scipy import stats 
from scipy.stats import norm 
# File Path 
import os 
for dirname, _, filenames in os.walk('/kaggle/input'): 
    for filename in filenames: 
        print(os.path.join(dirname, filename))


# Version check
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")

# Ignore Warning
import warnings
warnings.filterwarnings("ignore")

# setting
path_root = "/kaggle/input/"
seed = 394
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


df_train = pd.read_csv(path_root + "playground-series-s5e1/train.csv")
print("Train shape:",df_train.shape)

df_test = pd.read_csv(path_root + "playground-series-s5e1/test.csv")
print("Test shape:", df_test.shape)

display(df_train.head())
display(df_train.tail())


df_train.columns


print(df_train.info(), df_test.info())


# little feature engineering
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

def feature_engineering(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    return df

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)


# categorical features
list_not_features = ['id', 'num_sold']
list_features = [c for c in df_train.columns if not c in list_not_features]
list_numeric_features = []
list_categorical_features = []

for c in list_features:
    if df_train[c].dtype == "object":
        list_categorical_features.append(c)
    elif c == 'date':
        continue
    else:
        list_numeric_features.append(c)

# save memory
df_train[list_categorical_features] = df_train[list_categorical_features].astype('category')
df_test[list_categorical_features] = df_test[list_categorical_features].astype('category')


print(df_train.info(), df_test.info())


plt.figure(figsize = (5, 20), facecolor = "white")

sns.heatmap(
    df_train.isnull(), vmin = 0, vmax = 1
)

plt.show()


# duplicated rows
print(df_train.loc[df_train.duplicated()])
print(df_test.loc[df_test.duplicated()])


df_train_nonan = df_train.dropna()
print(df_train_nonan.shape)


def summary_numerical_dist(df_data, col, q_min, q_max):
    
    fig = plt.figure(figsize = (8, 4), facecolor = "white")

    layout_plot = (2, 2)
    num_subplot = 4
    axes = [None for _ in range(num_subplot)]
    list_shape_subplot = [[(0, 0), (0, 1), (1, 0), (1, 1)], [1, 1, 1, 1], [1, 1, 1, 1]]
    for i in range(num_subplot):
        axes[i] = plt.subplot2grid(
            layout_plot, list_shape_subplot[0][i],
            rowspan = list_shape_subplot[1][i],
            colspan = list_shape_subplot[2][i]
        )

    sns.histplot(data = df_data, x = col, kde = True, ax = axes[0])
    stats.probplot(x = df_data[col], dist = stats.norm, plot = axes[1])
    sns.boxplot(data = df_data, x = col, ax = axes[2])
    pts = df_data[col].quantile(q = np.arange(q_min, q_max, 0.01))
    sns.lineplot(x = pts.index, y = pts, ax = axes[3])
    axes[3].grid(True)

    list_title = ["Histogram", "QQ plot", "Boxplot", "Outlier"]
    for i in range(num_subplot):
        axes[i].set_title(list_title[i])
    plt.suptitle(f"Distribution of: {col}", fontsize = 15)
    plt.tight_layout()
    plt.show()


def summary_categorical_dist(df_data, col):
    
    fig = plt.figure(figsize = (8, 4), facecolor = "white")

    layout_plot = (1, 2)
    num_subplot = 2
    axes = [None for _ in range(num_subplot)]
    list_shape_subplot = [[(0, 0), (0, 1)], [1, 1], [1, 1]]
    for i in range(num_subplot):
        axes[i] = plt.subplot2grid(
            layout_plot, list_shape_subplot[0][i],
            rowspan = list_shape_subplot[1][i],
            colspan = list_shape_subplot[2][i]
        )
    
    count = df_data[col].value_counts().sort_index()
    
    sns.countplot(data = df_data, y = col, order = count.index, ax = axes[0])
    axes[1].pie(data = df_data, x = count, labels = count.index, autopct = '%1.1f%%', startangle = 90)
    
    list_title = ["Counts", "Proportions"]
    for i in range(num_subplot):
        axes[i].set_title(list_title[i])
    plt.suptitle(f"Distribution of: {col}", fontsize = 15)
    plt.tight_layout()
    plt.show()


# numerical
display(df_train.describe().round(3).T)
display(df_test.describe().round(3).T)


# categorical
display(df_train.describe(include = ['object', 'bool', 'category']).T)
display(df_test.describe(include = ['object', 'bool', 'category']).T)


summary_numerical_dist(df_train, 'num_sold', .95, 1)


for col in list_numeric_features + list_categorical_features:
    summary_categorical_dist(df_train, col)


for col in list_numeric_features + list_categorical_features:
    summary_categorical_dist(df_train_nonan, col)


# moving average
df_train_nonan['num_sold_7days'] = df_train_nonan['num_sold'].rolling(window = 7).mean()
df_train_nonan['num_sold_30days'] = df_train_nonan['num_sold'].rolling(window = 30).mean()
df_train_nonan['num_sold_90days'] = df_train_nonan['num_sold'].rolling(window = 90).mean()


plt.figure(figsize = (60, 6), facecolor = "white")

for i, col in enumerate(['num_sold', 'num_sold_7days', 'num_sold_30days', 'num_sold_90days'], start=1):
    sns.lineplot(
        data = df_train_nonan,
        x = 'date', y = col,
        alpha = 0.25*i, label = col
    )

plt.show()


plt.figure(figsize = (6, 3), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'num_sold', y = 'year',
    orient = 'h'
)

plt.show()


plt.figure(figsize = (6, 3), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'num_sold', y = 'month',
    orient = 'h'
)

plt.show()


plt.figure(figsize = (6, 3), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'num_sold', y = 'day_of_week',
    orient = 'h'
)

plt.show()


plt.figure(figsize = (6, 6), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'num_sold', y = 'year',
    hue = 'country',
    orient = 'h'
)

plt.show()

