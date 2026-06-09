import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def dataset_summary(datasets):
    summary = []

    for name, df, path in datasets:
        size_on_disk = os.path.getsize(path) / (1024 * 1024)  # MB
        size_in_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        rows, cols = df.shape

        summary.append({
            "Dataset": name,
            "Size on Disk (MB)": round(size_on_disk, 2),
            "Size in Memory (MB)": round(size_in_memory, 2),
            "# of Rows": rows,
            "# of Cols": cols
        })

    return pd.DataFrame(summary)



datasets = [
    ("train", train, train_path),
    ("test", test, test_path)
]

dataset_summary(datasets)


train.head()


test.head()


train.isnull().sum()


train.duplicated().sum()


train.info()


pd.concat([train.drop('target', axis=1, errors='ignore').dtypes, 
           test.dtypes], axis=1, keys=['train', 'test'])


train.describe().T[['mean', 'std', 'min', 'max']]


test.describe().T[['mean', 'std', 'min', 'max']]


cols = ['curvature', 'speed_limit', 'num_lanes']  

for col in cols:
    plt.figure(figsize=(6,3))
    sns.kdeplot(train[col], label='Train', fill=True)
    sns.kdeplot(test[col], label='Test', fill=True)
    plt.title(f'Distribution of {col}')
    plt.legend()
    plt.show()


sns.histplot(x='accident_risk',data=train)
plt.title('Distribution of accident risk')
plt.show()


train['accident_risk'].skew()


sns.boxplot(x='accident_risk',data=train)
plt.show()


train.info()


bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    train[col] = train[col].map({True: 1, False: 0}) 
    test[col] = test[col].map({True: 1, False: 0}) 

train.info()


train.nunique()


cat_cols=['road_type','num_lanes','speed_limit','lighting','weather','road_signs_present',
          'public_road','time_of_day','holiday','school_season','num_reported_accidents']

for col in cat_cols:
    print(f"\n{col} value_counts:")
    print(train[col].value_counts())
    sns.countplot(data=train, x=col, order=train[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


sns.histplot(train['curvature'], kde=True, bins=30)
plt.title(f'Distribution of curvature')
plt.show()


sns.boxplot(train['curvature'])
plt.title(f'boxplot of curvature')
plt.show()


sns.scatterplot(x='curvature',y='accident_risk',data=train)
plt.show()


cat_cols=['road_type','num_lanes','speed_limit','lighting','weather','road_signs_present',
          'public_road','holiday','school_season','num_reported_accidents']

for col in cat_cols:
    sns.barplot(data=train, x=col, y='accident_risk')
    plt.title(f'Barplot of {col} vs accident_risk')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    


sns.lineplot(x='time_of_day',y='accident_risk',data=train)
plt.show()


sns.catplot(
    data=train,
    x='road_type',
    y='accident_risk',
    hue='num_lanes',
    col='public_road',
    kind='bar',
    ci=None,
    palette='Set2',
    height=4,
    aspect=1
)

plt.subplots_adjust(top=0.8)
plt.suptitle('Accident Risk by Road Type, Number of Lanes, and Public Road')
plt.show()


sns.catplot(
    data=train,
    x='road_type',
    y='accident_risk',
    hue='num_lanes',
    col='school_season',
    kind='bar',
    ci=None,
    palette='Set2',
    height=4,
    aspect=1
)

plt.subplots_adjust(top=0.8)
plt.suptitle('Accident Risk by Road Type, Number of Lanes, and Public Road')
plt.show()


sns.catplot(
    data=train,
    x='road_type',
    y='accident_risk',
    hue='num_lanes',
    col='road_signs_present',
    kind='bar',
    ci=None,
    palette='Set2',
    height=4,
    aspect=1
)

plt.subplots_adjust(top=0.8)
plt.suptitle('Accident Risk by Road Type, Number of Lanes, and Public Road')
plt.show()


sns.heatmap(train.corr(numeric_only=True).round(2),annot=True,cmap='coolwarm')
plt.show()

