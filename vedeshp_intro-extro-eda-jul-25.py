# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


train.head()


train.info()





def get_missing_info(train):
    missing_counts = train.isnull().sum()
    missing_percent = 100 * train.isnull().sum() / len(train)
    missing_train = pd.DataFrame({
        'column': train.columns,
        'count' : missing_counts,
        'percent': missing_percent
    })
    missing_train = missing_train.sort_values(by='percent', ascending=False)
    missing_train = missing_train[missing_train['count'] > 0].reset_index(drop=True)
    return missing_train


missing_train = get_missing_info(train)
missing_train


import missingno as msno


msno.__version__


msno.bar(train, color='dodgerblue', sort='descending')
plt.title('Missing Values per feature', fontsize=16)
plt.show()


msno.heatmap(train)
plt.title('Correlation of missingness between features', fontsize=16)
plt.show()


print("Distribution of target variable Personality")
print(train['Personality'].value_counts())
print(train['Personality'].value_counts(normalize=True) * 100)

sns.countplot(data=train, x='Personality', palette='viridis')
plt.title('Distribution of Personality')
plt.show()


print('Distribution of feature Drained_after_socializing')
print(train['Drained_after_socializing'].value_counts(dropna=False))
print()
print('Distribution of feature Stage_fear')
print(train['Stage_fear'].value_counts(dropna=False))




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

sns.countplot(
    ax=axes[0],
    data=train,
    x=train['Stage_fear'].fillna('Missing'),
    palette='magma'
)
axes[0].set_title('Distribution of stage fear', fontsize=14)
axes[0].set_xlabel('Response')
axes[0].set_ylabel('Count')

sns.countplot(
    ax=axes[1],
    data=train,
    x=train['Drained_after_socializing'].fillna('Missing'),
    palette='viridis'
)

axes[1].set_title('Distribution of Drained_after_socializing', fontsize=14)
axes[1].set_xlabel('Response')
axes[1].set_ylabel('Count')

fig.suptitle('Distribution of categorical features', fontsize=16)

plt.tight_layout()

plt.show()



categorical_features = train.select_dtypes(include='object').drop('Personality', axis=1).columns

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 20))

axes = axes.flatten()

delnewcols = []

for i, col in enumerate(categorical_features):
    newcolname = str(col) + '_is_missing'
    delnewcols.append(newcolname)
    train[newcolname] = train[col].isnull()
    sns.countplot(ax=axes[i], data=train, x='Personality', hue=newcolname, palette='magma')  
    axes[i].set_title(f'Personality Distribution vs {newcolname} ')
    axes[i].set_xlabel('Personality')
    axes[i].set_ylabel('Count')



fig.suptitle('Personality vs Missingness of Categorical Features', fontsize=18, y=1.01)
plt.tight_layout()
plt.show()

for colname in delnewcols:
    del train[colname]

# train['Stage_fear_is_missing'] = train['Stage_fear'].isnull()

# sns.countplot(data=train, x='Personality', hue='Stage_fear_is_missing', palette='magma')
# plt.title('Personality Distribution vs. Missingness of Stage Fear')
# plt.show()

# # Clean up
# del train['Stage_fear_is_missing']


numerical_features = train.select_dtypes(include='float64').columns

train[numerical_features].hist(bins=30, figsize=(15, 10), layout=(2, 3))
plt.suptitle('Distribution of numerical features')
plt.tight_layout()
plt.show()


print("----------Statistical summary of numerical Features----------")
num_cols=train.select_dtypes(include=['int64', 'float64']).drop('id', axis=1)
print(num_cols.describe().T)


numerical_features = train.select_dtypes(include='float64').columns
print(numerical_features)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 20))

axes = axes.flatten()

for i, col in enumerate(numerical_features):
    sns.boxplot(
        ax=axes[i],
        data=train,
        x='Personality',
        y=col,
        palette='magma'
    )

    axes[i].set_title(f'{col} vs personality')
    axes[i].set_xlabel('')
    axes[i].set_ylabel(col)

if len(numerical_features) < len(axes):
    for j in range(len(numerical_features), len(axes)):
        axes[j].axis('off')

fig.suptitle('Numerical Features vs. Personality Type', fontsize=22, y=1.01)
plt.tight_layout()
plt.show()


categorical_features = train.select_dtypes(include='object').drop('Personality', axis=1).columns

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 20))

axes = axes.flatten()

for i, col in enumerate(categorical_features):
    sns.countplot(ax=axes[i], data=train.fillna({col: 'Missing'}), x=col, hue='Personality', palette='rocket')
    axes[i].set_title(f'{col} vs personality')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].legend(title='Personality')

if len(categorical_features) < len(axes):
    for j in range(len(categorical_features), len(axes)):
        axes[j].axis('off')

fig.suptitle('Categorical vs Personality (Target)', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()


# numerical_features = train.select_dtypes(include='float64').columns

# fig, axes = plt.subplots(nrows=len(numerical_features), ncols=len(numerical_features), figsize=(72, 80))

# axes = axes.flatten()

# for x in numerical_features:
#     for y in numerical_features:
#         if x!= y:
#             sns.scatterplot(ax=axes[i], data=train, x=x, y=y, alpha=0.5)
#             axes[i].set_title(f'{x} vs {y} ')


# plt.tight_layout()
# plt.show()

# can do pairplot here 

# numerical_features = train.select_dtypes(include='float64').columns

# print("Generating Pairplot... This is the most efficient way to see all pairwise relationships.")

# # --- The entire manual loop is replaced by this one command ---
# g = sns.pairplot(
#     data=train,
#     vars=numerical_features,  # Specify which columns to plot
#     hue='Personality',        # <-- The most important parameter! Colors points by target.
#     palette='viridis',        # A nice color palette
#     plot_kws={'alpha': 0.6}   # Add transparency to see overlapping points
# )

# # Add a title for the entire figure
# g.fig.suptitle('Pairwise Relationships Between Numerical Features (Colored by Personality)', fontsize=22, y=1.03)

# plt.show()

# but we have to handle NaN values - therefore we are skipping this currently


plt.figure(figsize=(12, 8))
# We use the numerical_cols DataFrame we created earlier
numerical_features = train.select_dtypes(include='float64').columns
corr_matrix = train[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features', fontsize=16, y=1.03)
plt.show()




