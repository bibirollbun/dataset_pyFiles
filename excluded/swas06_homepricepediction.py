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


import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings

warnings.filterwarnings("ignore")


df_train =pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
df_test = pd.read_csv("/kaggle/input/california-homelessness-prediction-challenge/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.info()


df_train.head(3)


df_test.head(3)


df_train.isnull().sum()


df_test.isnull().sum()


df_train.describe()




# Select only the first 31 numeric columns
numeric_cols = df_train.select_dtypes(include='number').columns[:31]
num_features = len(numeric_cols)

cols = 2
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, rows * 2))
fig.suptitle('Boxplots of Selected Numeric Features', fontsize=16)

axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df_train[col], ax=axes[i])
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(axis='y', labelsize=8)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





# Select only the first 31 numeric columns
numeric_cols = df_train.select_dtypes(include='number').columns[:31]
num_features = len(numeric_cols)

cols = 2
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, rows * 2))
fig.suptitle('Boxplots of Selected Numeric Features (Outliers Removed)', fontsize=16)

axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    # Calculate IQR
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_data = df_train[(df_train[col] >= lower_bound) & (df_train[col] <= upper_bound)][col]

    # Plot boxplot without outliers
    sns.boxplot(y=filtered_data, ax=axes[i])
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(axis='y', labelsize=8)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



numeric_cols = df_train.select_dtypes(include='number').columns[:31]
num_features = len(numeric_cols)

cols = 2
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, rows * 2))
fig.suptitle('Histograms of Selected Numeric Features', fontsize=16)

axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(data=df_train, x=col, ax=axes[i], kde=True, bins=30)
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(axis='x', labelsize=8)
    axes[i].tick_params(axis='y', labelsize=8)

# Remove any unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


pairplot_cols = df_train.select_dtypes(include='number').columns

# Plot pairplot
sns.pairplot(df_train[pairplot_cols])


X = df_train.drop('homeless_rate', axis='columns')
y = df_train['homeless_rate']




from sklearn.preprocessing import MinMaxScaler

cols_to_scale = X.select_dtypes(['int64', 'float64']).columns

scaler = MinMaxScaler()

X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
X.describe()


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return vif_df


calculate_vif(X[cols_to_scale])

