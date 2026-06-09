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


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.info()


train.head(10)


train.describe()


train.isnull().sum()


train.value_counts()


plt.figure(figsize=(12,4))


plt.subplot(1, 3, 1)
plt.hist(train['accident_risk'], bins=50, edgecolor='black')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.title('Distribution of Accident Risk')



plt.subplot(1, 3, 2)
plt.boxplot(train['accident_risk'])
plt.ylabel('Accident Risk')
plt.title('Boxplot of Accident Risk')



plt.subplot(1, 3, 3)
sns.kdeplot(train['accident_risk'], shade=True)
plt.xlabel('Accident Risk')
plt.title('Density Plot of Accident Risk')



# Statistics

print("Target Variable Statistics:")
print(f"Mean: {train['accident_risk'].mean():.3f}")
print(f"Median: {train['accident_risk'].median():.3f}")
print(f"Std: {train['accident_risk'].std():.3f}")
print(f"Min: {train['accident_risk'].min():.3f}")
print(f"Max: {train['accident_risk'].max():.3f}")
print(f"Skewness: {train['accident_risk'].skew():.3f}")


categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']


categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

print("MEAN ACCIDENT RISK BY CATEGORY")
print("=" * 60)

for col in categorical_cols:
    print(f"\n{col.upper()}:")
    risk_by_cat = train.groupby(col)['accident_risk'].agg(['mean', 'count'])
    risk_by_cat = risk_by_cat.sort_values('mean', ascending=False)
    print(risk_by_cat)
    print(f"Range: {risk_by_cat['mean'].max() - risk_by_cat['mean'].min():.3f}")



boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']

for col in boolean_cols:
    print(f"\n{col}:")
    print(f"True: {train[col].sum()} ({train[col].sum()/len(train)*100:.1f}%)")
    print(f"False: {(~train[col]).sum()} ({(~train[col]).sum()/len(train)*100:.1f}%)")
    print(f"Mean risk when True: {train[train[col]]['accident_risk'].mean():.3f}")
    print(f"Mean risk when False: {train[~train[col]]['accident_risk'].mean():.3f}")



fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(boolean_cols):
    risk_data = train.groupby(col)['accident_risk'].mean()
    axes[idx].bar(['False', 'True'], risk_data.values)
    axes[idx].set_ylabel('Mean Accident Risk')
    axes[idx].set_title(f'Accident Risk: {col}')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()



numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']



fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(train[col], bins=30, edgecolor='black')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()



for col in numerical_cols:
    correlation = train[col].corr(train['accident_risk'])
    print(f"{col}: {correlation:.3f}")



fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].scatter(train[col], train['accident_risk'], alpha=0.1, s=1)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('accident_risk')
    axes[idx].set_title(f'{col} vs Accident Risk')

plt.tight_layout()
plt.show()



corr_cols = numerical_cols + ['accident_risk']
correlation_matrix = train[corr_cols].corr()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Print correlations with target sorted
print("\nCorrelations with accident_risk:")
print(correlation_matrix['accident_risk'].sort_values(ascending=False))



interaction = train.groupby(['weather', 'lighting'])['accident_risk'].mean().unstack()
plt.figure(figsize=(10, 6))
sns.heatmap(interaction, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Mean Accident Risk: Weather vs Lighting')
plt.show()



fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, col in enumerate(numerical_cols):
    axes[idx].boxplot(train[col])
    axes[idx].set_title(f'Boxplot: {col}')
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.show()



for col in categorical_cols:
    print(f"\n{col}:")
    print("Train:")
    print(train[col].value_counts(normalize=True))
    print("\nTest:")
    print(test[col].value_counts(normalize=True))
    print("-" * 50)



# Detailed road type analysis

road_risk = train.groupby('road_type').agg({
    'accident_risk': ['mean', 'std', 'min', 'max'],
    'id': 'count'
}).round(3)

road_risk.columns = ['mean_risk', 'std_risk', 'min_risk', 'max_risk', 'count']
road_risk = road_risk.sort_values('mean_risk', ascending=False)

print(road_risk)
print("\n")
print(f"Highest risk: {road_risk.index[0]} with mean risk of {road_risk['mean_risk'].iloc[0]:.3f}")
print(f"Lowest risk:  {road_risk.index[-1]} with mean risk of {road_risk['mean_risk'].iloc[-1]:.3f}")
print(f"Risk difference: {road_risk['mean_risk'].iloc[0] - road_risk['mean_risk'].iloc[-1]:.3f}")

