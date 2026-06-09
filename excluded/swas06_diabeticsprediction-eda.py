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


df_train=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_train.info()


df_test.info()


df_train.isnull().sum(),df_test.isnull().sum()


df_train.head(3)


df_train= df_train.drop(['id'],axis =1)
df_test= df_test.drop(['id'],axis =1)


numerical_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df_train.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)
print(f"\nTotal number of numerical columns: {len(numerical_columns)}")

print("\nCategorical Columns:")
print(categorical_columns)
print(f"\nTotal number of categorical columns: {len(categorical_columns)}")


df_train[numerical_columns] = df_train[numerical_columns].apply(lambda x: x.fillna(x.mean()))
df_train[categorical_columns] = df_train[categorical_columns].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))




df_train[numerical_columns].describe()


num_vars = len(numerical_columns)

# Create a grid large enough (6x4 = 24 slots for 19 features)
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(16, 12))
fig.suptitle('Boxplots of Selected Features', fontsize=16)

# Flatten axes
axes = axes.flatten()

# Plot boxplots
for i, col in enumerate(numerical_columns):
    sns.boxplot(y=df_train[col], ax=axes[i])
    axes[i].set_title(col)

# Hide leftover empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


num_vars = len(numerical_columns)

# Create a grid large enough (6x4 = 24 slots for 19 features)
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(16, 12))
fig.suptitle('Histplots of Selected Features', fontsize=16)

# Flatten axes
axes = axes.flatten()

# Plot boxplots
for i, col in enumerate(numerical_columns):
    sns.histplot(x=df_train[col],kde =True, ax=axes[i])
    axes[i].set_title(col)

# Hide leftover empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


features = ['physical_activity_minutes_per_week']


def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Keep only rows within bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Cleaned data
df_train_clean = remove_outliers_iqr(df_train, features)


num_vars = len(numerical_columns)

# Create a grid large enough (6x4 = 24 slots for 19 features)
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(16, 12))
fig.suptitle('Boxplots of Selected Features', fontsize=16)

# Flatten axes
axes = axes.flatten()

# Plot boxplots
for i, col in enumerate(numerical_columns):
    sns.boxplot(y=df_train_clean[col], ax=axes[i])
    axes[i].set_title(col)

# Hide leftover empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df_train_clean[numerical_columns].corr(), annot=True, cmap='coolwarm',fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


sns.pairplot(df_train_clean[numerical_columns ], diag_kind='kde', plot_kws={'alpha':0.6})
plt.show()


for col in categorical_columns:
    print(col, "-->", df_train_clean[col].unique())


import math
n_features = len(categorical_columns)
ncols = 2  # You can adjust the number of columns here
nrows = math.ceil(n_features / ncols)  # Calculate rows needed to fit all features

# Plot the countplot for each categorical feature
plt.figure(figsize=(10, nrows * 5))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(nrows, ncols, i)  # Create subplots with dynamic grid size
    sns.countplot(x=df_train_clean[col], palette="viridis")
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


for col in categorical_columns:
    plt.figure(figsize=(6, 6))
    df_train_clean[col].value_counts().plot(
        kind='pie', autopct='%1.1f%%', startangle=90,
        colors=sns.color_palette("pastel"))
    plt.title(f"{col} Distribution", fontsize=14, weight='bold')
    plt.ylabel('')
    plt.tight_layout()
    #plt.savefig(f"pie_charts/{col}_pie.png", dpi=300)
    plt.show()
    plt.close()


for col in categorical_columns:
    if col != "gender":  # Avoid plotting Gender vs Gender
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_train_clean, x=col, hue="gender", palette="Set2")
        plt.title(f"{col} by Sex", fontsize=14, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        #plt.savefig(f"grouped_bars/{col}_by_gender.png", dpi=300)
        plt.show()
        plt.close()



for i in range(len(categorical_columns)):
    for j in range(i+1, len(categorical_columns)):
        col1, col2 = categorical_columns[i], categorical_columns[j]
        cross_tab = pd.crosstab(df_train_clean[col1], df_train_clean[col2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Heatmap: {col1} vs {col2}", fontsize=14, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        #plt.savefig(f"heatmaps/{col1}_vs_{col2}.png", dpi=300)
        plt.show()
        plt.close()

print("âœ… All plots generated and saved in separate folders.")


X = df_train_clean.drop('diagnosed_diabetes', axis='columns')
y = df_train_clean['diagnosed_diabetes']

from sklearn.preprocessing import MinMaxScaler

cols_to_scale = X.select_dtypes(['int64', 'float64']).columns

scaler = MinMaxScaler()

X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
X.describe()


!pip install statsmodels


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return vif_df




calculate_vif(X[cols_to_scale])


df_train_clean.head(3)


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
        _, iv = calculate_woe_iv(pd.concat([X, y],axis=1), feature, 'diagnosed_diabetes' )
    else:
        X_binned = pd.cut(X[feature], bins=10, labels=False)
        _, iv = calculate_woe_iv(pd.concat([X_binned, y],axis=1), feature, 'diagnosed_diabetes' )
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

