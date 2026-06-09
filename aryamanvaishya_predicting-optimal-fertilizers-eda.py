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

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train.head()


print(train.shape)
print(train.info())
print(train.describe())


print(test.shape)
print(test.info())
print(test.describe())


test.head()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.countplot(data=train, y='Fertilizer Name', order=train['Fertilizer Name'].value_counts().index)
plt.title('Fertilizer Distribution')
plt.xlabel('Count')
plt.ylabel('Fertilizer Name')
plt.tight_layout()
plt.show()


numeric_cols = train.select_dtypes(include=np.number).columns.drop('id')

train[numeric_cols].hist(figsize=(16, 12), bins=30)
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()


grouped_stats = train.groupby('Fertilizer Name')[numeric_cols].mean().T
grouped_stats.plot(kind='bar', figsize=(16, 8))
plt.title("Mean Feature Values by Fertilizer")
plt.ylabel("Mean Value")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()


numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train, x='Fertilizer Name', y=feature)
    plt.title(f'{feature} by Fertilizer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=train, x='Soil Type', hue='Fertilizer Name')
plt.title("Fertilizer Distribution by Soil Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.countplot(data=train, x='Crop Type', hue='Fertilizer Name')
plt.title("Fertilizer Distribution by Crop Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Encode categorical features first
df = train.copy()
df['Soil Type'] = LabelEncoder().fit_transform(df['Soil Type'])
df['Crop Type'] = LabelEncoder().fit_transform(df['Crop Type'])
X = df[numerical_features + ['Soil Type', 'Crop Type']]
y = df['Fertilizer Name']

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Fertilizer Name'] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Fertilizer Name', palette='tab10')
plt.title("PCA Projection")
plt.tight_layout()
plt.show()


features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=train, x='Fertilizer Name', y=feature, inner='quartile', palette='Set2')
    plt.title(f'Distribution of {feature} by Fertilizer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


pivot = train.groupby(['Crop Type', 'Soil Type'])['Fertilizer Name'] \
             .agg(lambda x: x.value_counts().index[0]) \
             .unstack()

# Create dummy numeric matrix (same shape) just for heatmap structure
dummy = pivot.notnull().astype(int)

plt.figure(figsize=(12, 8))
ax = sns.heatmap(dummy, annot=pivot, fmt='', cmap='Blues', cbar=False, linewidths=0.5, linecolor='gray')
plt.title('Most Used Fertilizer by Crop and Soil Type')
plt.ylabel('Crop Type')
plt.xlabel('Soil Type')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


from seaborn import pairplot

sampled = train.sample(1000, random_state=42)  # downsample to avoid clutter
numeric = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
sns.pairplot(sampled[numeric + ['Fertilizer Name']], hue='Fertilizer Name', palette='husl', corner=True)
plt.suptitle("Pairwise Feature Distribution by Fertilizer", y=1.02)
plt.show()


# from math import pi

# radar_data = train.groupby('Fertilizer Name')[features].mean().reset_index()
# categories = features
# N = len(categories)

# # Radar plot setup
# for i in range(len(radar_data)):
#     values = radar_data.iloc[i, 1:].tolist()
#     values += values[:1]
#     angles = [n / float(N) * 2 * pi for n in range(N)]
#     angles += angles[:1]

#     plt.figure(figsize=(6, 6))
#     ax = plt.subplot(111, polar=True)
#     plt.xticks(angles[:-1], categories)
#     ax.plot(angles, values, linewidth=2, linestyle='solid', label=radar_data['Fertilizer Name'][i])
#     ax.fill(angles, values, alpha=0.25)
#     plt.title(f"Average Profile: {radar_data['Fertilizer Name'][i]}")
#     plt.tight_layout()
#     plt.show()


features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
radar_data = train.groupby('Fertilizer Name')[features].mean().reset_index()
print(radar_data)
print("----------------------------------------------------------------------------------")
radar_data.describe()


from math import pi
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the features you used earlier (replace with your actual column names if different)
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categories = features
N = len(categories)

# Normalize data
scaler = MinMaxScaler()
radar_scaled = radar_data.copy()
radar_scaled[features] = scaler.fit_transform(radar_scaled[features])

# Radar plot per fertilizer
for i in range(len(radar_scaled)):
    values = radar_scaled.iloc[i, 1:].tolist()
    values += values[:1]  # repeat first value to close the loop
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories)
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=radar_scaled['Fertilizer Name'][i])
    ax.fill(angles, values, alpha=0.25)
    plt.title(f"Average Profile (Normalized): {radar_scaled['Fertilizer Name'][i]}")
    plt.tight_layout()
    plt.show()


# Sample model: Predicting most common fertilizer (baseline)

# 1. Get most common fertilizer in train set
most_common_fert = train['Fertilizer Name'].value_counts().idxmax()

# 2. Create a simple baseline prediction (same fertilizer for all rows)
submission = test[['id']].copy()
submission['Fertilizer Name'] = most_common_fert  # or a smarter prediction list

# 3. Save as CSV
submission.to_csv('/kaggle/working/submission.csv', index=False)




