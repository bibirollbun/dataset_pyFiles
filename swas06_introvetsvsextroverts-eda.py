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


df_train =pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.info()


df_train.head(3)


df_train.isnull().sum(),df_test.isnull().sum()


numeric_cols = df_train.select_dtypes(include='number').columns
categorical_cols = df_train.select_dtypes(include='object').columns


df_train[numeric_cols] = df_train[numeric_cols].fillna(df_train[numeric_cols].mean())


for col in categorical_cols:
    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])


df_train.isnull().sum()


df_train = df_train.drop('id',axis=1)


numeric_cols = df_train.select_dtypes(include='number').columns


df_train[numeric_cols].describe()


import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))  
fig.suptitle('Boxplots of Selected Features', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(df_train[numeric_cols]):
    sns.boxplot(y=df_train[col], ax=axes[i])
    axes[i].set_title(col)
fig.delaxes(axes[-1])

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


import warnings
warnings.filterwarnings("ignore")
def remove_outliers_iqr(df):
    df_cleaned = df_train.copy()  # Copy the original DataFrame
    for col in df_cleaned.select_dtypes(include=np.number):  # Apply only to numeric columns
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

# Apply function to remove outliers
df_train_cleaned = remove_outliers_iqr(df_train)

# Display shape before and after
print(f"Original dataset shape: {df_train.shape}")
print(f"Cleaned dataset shape: {df_train_cleaned.shape}")


import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))  
fig.suptitle('Boxplots of Selected Features', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(df_train_cleaned[numeric_cols]):
    sns.boxplot(y=df_train_cleaned[col], ax=axes[i])
    axes[i].set_title(col)
fig.delaxes(axes[-1])

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))  
fig.suptitle('Distribution of Selected Features', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(df_train_cleaned[numeric_cols]):
    sns.histplot(x=df_train_cleaned[col],kde =True,ax=axes[i])
    axes[i].set_title(col)
fig.delaxes(axes[-1])

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


sns.pairplot(df_train_cleaned)
plt.show()


df_train.columns


corr_matrix = df_train_cleaned[numeric_cols].corr()

plt.figure(figsize=(10, 8))  # Adjust the size as necessary
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


for col in categorical_cols:
    print(f"\nValue counts for '{col}':")
    print(df_train_cleaned[col].value_counts())
    print("\n")


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, len(categorical_cols) * 4))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(len(categorical_cols), 1, i)
    sns.countplot(data=df_train_cleaned, x=col, order=df_train_cleaned[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df_train_cleaned['personality'] = df_train_cleaned['personality'].map({
    'Extrovert': 0,
    'Introvert': 1
})


X = df_train_cleaned.drop('personality', axis='columns')
y = df_train_cleaned['personality']

from sklearn.preprocessing import MinMaxScaler

cols_to_scale = X.select_dtypes(['int64', 'float64']).columns

scaler = MinMaxScaler()

X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
X.describe()


X.columns


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return vif_df


calculate_vif(X[cols_to_scale])


X_train_1 = X.drop('going_outside', axis='columns')
numeric_columns = X_train_1.select_dtypes(['int64', 'float64']).columns
numeric_columns


calculate_vif(X_train_1[numeric_columns])


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

for feature in X_train_1.columns:
    if X_train_1[feature].dtype == 'object':
        _, iv = calculate_woe_iv(pd.concat([X_train_1, y],axis=1), feature, 'personality' )
    else:
        X_binned = pd.cut(X_train_1[feature], bins=10, labels=False)
        _, iv = calculate_woe_iv(pd.concat([X_binned, y],axis=1), feature, 'personality' )
    iv_values[feature] = iv
        
iv_values

