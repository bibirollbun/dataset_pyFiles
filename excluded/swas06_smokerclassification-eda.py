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
import warnings
warnings.filterwarnings("ignore")


df_train=pd.read_csv('/kaggle/input/binary-prediction-of-smoker/train.csv')
df_test=pd.read_csv('/kaggle/input/binary-prediction-of-smoker/test.csv')


df_train.info()


df_test.info()


df_train.head(3)


df_train.isnull().sum(),df_test.isnull().sum()


df_train.describe()


cols = df_train.columns
n = len(cols)

# Calculate number of rows automatically (2 columns per row)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4 * nrows))
fig.suptitle('Boxplots of Selected Features', fontsize=16)

axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(y=df_train[col], ax=axes[i])
    axes[i].set_title(col)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



def remove_outliers_iqr(df,):
    df_clean = df.copy()
    for col in df_clean:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Keep only rows within bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Cleaned data
df_train_clean = remove_outliers_iqr(df_train)


cols = df_train_clean.columns
n = len(cols)

# Calculate number of rows automatically (2 columns per row)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4 * nrows))
fig.suptitle('Boxplots of Selected Features', fontsize=16)

axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(y=df_train_clean[col], ax=axes[i])
    axes[i].set_title(col)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



cols = df_train_clean.columns
n = len(cols)

# Calculate number of rows automatically (2 columns per row)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4 * nrows))
fig.suptitle('Boxplots of Selected Features', fontsize=16)

axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(x=df_train_clean[col],kde=True, ax=axes[i])
    axes[i].set_title(col)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


plt.figure(figsize=(15, 12))
sns.heatmap(df_train_clean.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


numeric_cols = df_train_clean.select_dtypes(include=['float64', 'int64']).columns
sns.pairplot(df_train_clean[numeric_cols], diag_kind='kde', plot_kws={'alpha':0.6})
plt.show()


X = df_train_clean.drop('smoking', axis='columns')
y = df_train_clean['smoking']

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


def calculate_woe_iv(df, feature, target):
    grouped = df.groupby(feature)[target].agg(['count','sum'])
    grouped = grouped.rename(columns={'count': 'total', 'sum': 'good'})
    grouped['bad']=grouped['total']-grouped['good']
    
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    grouped['good_pct'] = grouped['good'] / total_good
    grouped['bad_pct'] = grouped['bad'] / total_bad
    grouped['woe'] = np.log(grouped['good_pct']/ grouped['bad_pct'])
    grouped['iv'] = (grouped['good_pct'] -grouped['bad_pct'])*grouped['woe']
    
    grouped['woe'] = grouped['woe'].replace([np.inf, -np.inf], 0)
    grouped['iv'] = grouped['iv'].replace([np.inf, -np.inf], 0)
    
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv


iv_values = {}

for feature in X.columns:
    if X[feature].dtype == 'object':
        _, iv = calculate_woe_iv(pd.concat([X, y],axis=1), feature, 'smoking' )
    else:
        X_binned = pd.cut(X[feature], bins=10, labels=False)
        _, iv = calculate_woe_iv(pd.concat([X_binned, y],axis=1), feature, 'smoking' )
    iv_values[feature] = iv
        
iv_values


def interpret_iv(iv):
    if iv < 0.02:
        return 'Not useful'
    elif iv < 0.1:
        return 'Weak'
    elif iv < 0.3:
        return 'Medium'
    elif iv < 0.5:
        return 'Strong'
    else:
        return 'Suspiciously Predictive'

# Create summary
for feature, iv in iv_values.items():
    print(f"{feature:20} | IV = {iv:.2f} | {interpret_iv(iv)}")

