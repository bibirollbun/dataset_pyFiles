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
import numpy as np
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


df_train.head()


df_train['Brand'].unique()


df_train['Material'].unique()


df_train['Size'].unique()


df_train['Style'].unique()


df_train['Color'].unique()


df_train.describe()


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
sns.histplot(df_train["Price"], binwidth=10, kde=True, color='blue')
plt.title("Price Distribution")
plt.xlabel("USD")

plt.subplot(1, 3, 2)
sns.histplot(df_train["Weight Capacity (kg)"], binwidth=5, kde=True, color='green')
plt.title("Weight Capacity Distribution")
plt.xlabel("kg")

plt.subplot(1, 3, 3)
sns.histplot(df_train["Compartments"], binwidth=1, kde=True, color='orange')
plt.title("Compartments Distribution")
plt.xlabel("number")

plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
sns.histplot(df_train[df_train.Brand == 'Puma']["Price"], binwidth=10, kde=True, color='blue')
plt.title("Puma")
plt.xlabel("USD")

plt.subplot(1, 5, 2)
sns.histplot(df_train[df_train.Brand == 'Jansport']["Price"], binwidth=10, kde=True, color='blue')
plt.title("Jansport")
plt.xlabel("USD")

plt.subplot(1, 5, 3)
sns.histplot(df_train[df_train.Brand == 'Under Armour']["Price"], binwidth=10, kde=True, color='blue')
plt.title("Under Armour")
plt.xlabel("USD")

plt.subplot(1, 5, 4)
sns.histplot(df_train[df_train.Brand == 'Nike']["Price"], binwidth=10, kde=True, color='blue')
plt.title("Nike")
plt.xlabel("USD")

plt.subplot(1, 5, 5)
sns.histplot(df_train[df_train.Brand == 'Adidas']["Price"], binwidth=10, kde=True, color='blue')
plt.title("Adidas")
plt.xlabel("USD")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
df_train_nums = df_train.copy()
df_train_nums['Brand'].replace(['Jansport', 'Under Armour', 'Nike', 'Adidas', 'Puma'], 
                               [0, 1, 2, 3, 4],
                               inplace=True)
df_train_nums['Material'].replace(['Leather', 'Canvas', 'Nylon', 'Polyester'],
                                  [0, 1, 2, 3],
                                  inplace=True)
corr = df_train_nums.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
sns.histplot(df_train[df_train.Size == 'Small']["Price"], binwidth=10, kde=True, color='blue', stat='probability')
plt.title("Small")
plt.xlabel("USD")

plt.subplot(1, 3, 2)
sns.histplot(df_train[df_train.Size == 'Medium']["Price"], binwidth=10, kde=True, color='blue', stat='probability')
plt.title("Medium")
plt.xlabel("USD")

plt.subplot(1, 3, 3)
sns.histplot(df_train[df_train.Size == 'Large']["Price"], binwidth=10, kde=True, color='blue', stat='probability')
plt.title("Large")
plt.xlabel("USD")

plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_train[df_train.Waterproof == 'Yes']["Price"], binwidth=10, kde=True, color='blue', stat='proportion')
plt.title("Waterproof")
plt.xlabel("USD")

plt.subplot(1, 2, 2)
sns.histplot(df_train[df_train.Waterproof == 'No']["Price"], binwidth=10, kde=True, color='blue', stat='proportion')
plt.title("Permiate")
plt.xlabel("USD")

plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_train[df_train['Laptop Compartment'] == 'Yes']["Price"], binwidth=10, kde=True, color='orange', stat='probability')
plt.title("Laptop")
plt.xlabel("USD")

plt.subplot(1, 2, 2)
sns.histplot(df_train[df_train['Laptop Compartment'] == 'No']["Price"], binwidth=10, kde=True, color='orange', stat='probability')
plt.title("No Laptop")
plt.xlabel("USD")

plt.tight_layout()
plt.show()


import warnings
warnings.filterwarnings("ignore")
for j, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    plt.figure(figsize=(20, 3))
    for i, color in enumerate(['Black', 'Green', 'Red', 'Blue', 'Gray', 'Pink']):
        plt.subplot(1, 6, i+1)
        sns.histplot(df_train[(df_train['Color'] == color) & (df_train['Brand'] == brand)]["Price"], binwidth=10, kde=True, color='blue', stat='proportion')
        plt.title(f'{color}_{brand}')
        plt.xlabel("USD")
    plt.tight_layout()
    plt.show()
    


for j, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    plt.figure(figsize=(20, 2))
    for i, material in enumerate(['Leather', 'Canvas', 'Nylon', 'Polyester']):
        plt.subplot(1, 6, i+1)
        sns.histplot(df_train[(df_train['Material'] == material) & (df_train['Brand'] == brand)]["Price"], binwidth=10, kde=True, color='orange', stat='proportion')
        plt.title(f'{material}_{brand}')
        plt.xlabel("USD")
    plt.tight_layout()
    plt.show()


for j, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    plt.figure(figsize=(15, 2))
    for i, style in enumerate(['Tote', 'Messenger', 'Backpack']):
        plt.subplot(1, 6, i+1)
        sns.histplot(df_train[(df_train['Style'] == style) & (df_train['Brand'] == brand)]["Price"], binwidth=10, kde=True, color='green', stat='proportion')
        plt.title(f'{style}_{brand}')
        plt.xlabel("USD")
    plt.tight_layout()
    plt.show()


from IPython.display import display


intervals = [(15, 35), (35, 55), (55, 75), (75, 95), (95, 115), (115, 135),
            (135, 155)]
plt.figure(figsize=(15, 20))
for i, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    col = []
    plt.subplot(5, 1, i+1)
    for int_b, int_e in intervals:
        df_train_int = df_train[(df_train['Brand'] == brand) &
            (int_b <= df_train['Price']) &
            (df_train['Price'] < int_e)]
        df_int = df_train_int.groupby('Material').size()
        df_tmp = df_int.to_frame().T
        df_tmp['interval'] = int_e
        col.append(df_tmp)
    df_i = pd.concat(col, axis=0, ignore_index=True).set_index('interval')
    df_ii = df_i.div(df_i.sum(axis=1), axis=0)
    sns.lineplot(df_ii)
    plt.title(f'{brand}')
plt.tight_layout()
plt.show()


intervals = [(15, 35), (35, 55), (55, 75), (75, 95), (95, 115), (115, 135),
            (135, 155)]
plt.figure(figsize=(15, 20))
for i, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    col = []
    plt.subplot(5, 1, i+1)
    for int_b, int_e in intervals:
        df_train_int = df_train[(df_train['Brand'] == brand) &
            (int_b <= df_train['Price']) &
            (df_train['Price'] < int_e)]
        df_int = df_train_int.groupby('Compartments').size()
        df_tmp = df_int.to_frame().T
        df_tmp['interval'] = int_e
        col.append(df_tmp)
    df_i = pd.concat(col, axis=0, ignore_index=True).set_index('interval')
    df_ii = df_i.div(df_i.sum(axis=1), axis=0)
    sns.lineplot(df_ii)
    plt.title(f'{brand}')
plt.tight_layout()
plt.show()


intervals = [(15, 35), (35, 55), (55, 75), (75, 95), (95, 115), (115, 135),
            (135, 155)]
plt.figure(figsize=(15, 20))
for i, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    col = []
    plt.subplot(5, 1, i+1)
    for int_b, int_e in intervals:
        df_train_int = df_train[(df_train['Brand'] == brand) &
            (int_b <= df_train['Price']) &
            (df_train['Price'] < int_e)]
        df_int = df_train_int.groupby('Color').size()
        df_tmp = df_int.to_frame().T
        df_tmp['interval'] = int_e
        col.append(df_tmp)
    df_i = pd.concat(col, axis=0, ignore_index=True).set_index('interval')
    df_ii = df_i.div(df_i.sum(axis=1), axis=0)
    sns.lineplot(df_ii)
    plt.title(f'{brand}')
plt.tight_layout()
plt.show()


intervals = [(15, 35), (35, 55), (55, 75), (75, 95), (95, 115), (115, 135),
            (135, 155)]
plt.figure(figsize=(15, 20))
for i, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    col = []
    plt.subplot(5, 1, i+1)
    for int_b, int_e in intervals:
        df_train_int = df_train[(df_train['Brand'] == brand) &
            (int_b <= df_train['Price']) &
            (df_train['Price'] < int_e)]
        df_int = df_train_int.groupby('Size').size()
        df_tmp = df_int.to_frame().T
        df_tmp['interval'] = int_e
        col.append(df_tmp)
    df_i = pd.concat(col, axis=0, ignore_index=True).set_index('interval')
    df_ii = df_i.div(df_i.sum(axis=1), axis=0)
    sns.lineplot(df_ii)
    plt.title(f'{brand}')
plt.tight_layout()
plt.show()


intervals = [(15, 35), (35, 55), (55, 75), (75, 95), (95, 115), (115, 135),
            (135, 155)]
plt.figure(figsize=(15, 20))
for i, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    col = []
    plt.subplot(5, 1, i+1)
    for int_b, int_e in intervals:
        df_train_int = df_train[(df_train['Brand'] == brand) &
            (int_b <= df_train['Price']) &
            (df_train['Price'] < int_e)]
        df_int = df_train_int.groupby('Style').size()
        df_tmp = df_int.to_frame().T
        df_tmp['interval'] = int_e
        col.append(df_tmp)
    df_i = pd.concat(col, axis=0, ignore_index=True).set_index('interval')
    df_ii = df_i.div(df_i.sum(axis=1), axis=0)
    sns.lineplot(df_ii)
    plt.title(f'{brand}')
plt.tight_layout()
plt.show()


intervals = [(15, 35), (35, 55), (55, 75), (75, 95), (95, 115), (115, 135),
            (135, 155)]
plt.figure(figsize=(15, 20))
for i, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    col = []
    plt.subplot(5, 1, i+1)
    for int_b, int_e in intervals:
        df_train_int = df_train[(df_train['Brand'] == brand) &
            (int_b <= df_train['Price']) &
            (df_train['Price'] < int_e)]
        df_int = df_train_int.groupby('Waterproof').size()
        df_tmp = df_int.to_frame().T
        df_tmp['interval'] = int_e
        col.append(df_tmp)
    df_i = pd.concat(col, axis=0, ignore_index=True).set_index('interval')
    df_ii = df_i.div(df_i.sum(axis=1), axis=0)
    sns.lineplot(df_ii)
    plt.title(f'{brand}')
plt.tight_layout()
plt.show()


intervals = [(15, 35), (35, 55), (55, 75), (75, 95), (95, 115), (115, 135),
            (135, 155)]
plt.figure(figsize=(15, 20))
for i, brand in enumerate(['Under Armour', 'Nike', 'Puma', 'Adidas', 'Jansport']):
    col = []
    plt.subplot(5, 1, i+1)
    for int_b, int_e in intervals:
        df_train_int = df_train[(df_train['Brand'] == brand) &
            (int_b <= df_train['Price']) &
            (df_train['Price'] < int_e)]
        df_int = df_train_int.groupby('Laptop Compartment').size()
        df_tmp = df_int.to_frame().T
        df_tmp['interval'] = int_e
        col.append(df_tmp)
    df_i = pd.concat(col, axis=0, ignore_index=True).set_index('interval')
    df_ii = df_i.div(df_i.sum(axis=1), axis=0)
    sns.lineplot(df_ii)
    plt.title(f'{brand}')
plt.tight_layout()
plt.show()


intervals = [(15, 35), (35, 55), (55, 75), (75, 95), (95, 115), (115, 135),
            (135, 155)]
weights = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 31)]
def to_label(intervals):
    def _to_label(w):
        for i, (b, e) in enumerate(intervals):
            if b <= w < e:
                return i
        return -1
    return _to_label
def q1(x):
    return x.quantile(.25)
def q3(x):
    return x.quantile(.75)
plt.figure(figsize=(15, 5))
df_train_int = df_train.dropna()
df_train_int['WeightBucket'] = df_train_int['Weight Capacity (kg)'].apply(to_label(weights))
sns.violinplot(x="WeightBucket", y="Price", data=df_train_int, hue='Brand', whis=1.5, showmeans=True)
plt.title(f'{brand}')
plt.tight_layout()
plt.show()


weights = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 31)]
def to_label(intervals):
    def _to_label(w):
        for i, (b, e) in enumerate(intervals):
            if b <= w < e:
                return i
        return -1
    return _to_label
def q1(x):
    return x.quantile(.25)
def q3(x):
    return x.quantile(.75)
plt.figure(figsize=(15, 5))
df_train_int = df_train.dropna()
df_train_int['WeightBucket'] = df_train_int['Weight Capacity (kg)'].apply(to_label(weights))
sns.violinplot(x="Brand", y="Price", data=df_train_int, hue='WeightBucket', whis=(0, 100))
plt.title(f'{brand}')
plt.tight_layout()
plt.show()


max_price = df_train['Price'].max()
df_train[df_train['Price'] == max_price].dropna()


plt.figure(figsize=(15, 5))
df_train_int = df_train.dropna()
sns.violinplot(x="Brand", y="Price", data=df_train_int, hue='Waterproof', showmeans=True)
plt.tight_layout()
plt.show()


df_params = df_train.dropna()[['Price', 'Brand', 'Waterproof']]
g = df_params.groupby(['Brand', 'Waterproof'])
df_pp = g.agg(['mean', 'median']).sort_index(level=[0, 1])
df_pp.columns = df_pp.columns.droplevel([0])
df_pp['left_skewed'] = df_pp['median'] - df_pp['mean']
df_pp


plt.figure(figsize=(15, 5))
df_train_int = df_train.dropna()
sns.violinplot(x="Brand", y="Price", data=df_train_int, hue='Laptop Compartment', showmeans=True)
plt.tight_layout()
plt.show()


df_params = df_train.dropna()[['Price', 'Brand', 'Laptop Compartment']]
g = df_params.groupby(['Brand', 'Laptop Compartment'])
df_pp = g.agg(['mean', 'median'])
df_pp.columns = df_pp.columns.droplevel([0])
df_pp['left_skewed'] = df_pp['median'] - df_pp['mean']
df_pp


plt.figure(figsize=(15, 5))
df_train_int = df_train.dropna()
sns.violinplot(x="Brand", y="Price", data=df_train_int, hue='Style', whis=1.5, showmeans=True)
plt.tight_layout()
plt.show()


df_params = df_train.dropna()[['Price', 'Brand', 'Style']]
g = df_params.groupby(['Brand', 'Style'])
df_pp = g.agg(['mean', 'median'])
df_pp.columns = df_pp.columns.droplevel([0])
df_pp['left_skewed'] = df_pp['median'] - df_pp['mean']
df_pp

