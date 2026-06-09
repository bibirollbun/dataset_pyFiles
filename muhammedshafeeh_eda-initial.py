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


pip install pandas seaborn matplotlib



import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
data = pd.read_csv("/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv")
sample_submission = pd.read_csv("/kaggle/input/trojan-horse-hunt-in-space/sample_submission_solution.csv")

# Display basic info
print("ðŸ”¹ Dataset Shape:", data.shape)
print("\nðŸ”¹ Data Types:")
print(data.dtypes)

# Display first few rows
print("\nðŸ”¹ First 5 rows:")
print(data.head())

# Check for missing values
print("\nðŸ”¹ Missing Values:")
print(data.isnull().sum())

# Describe numerical features
print("\nðŸ”¹ Summary Statistics:")
print(data.describe())

# Check class balance if target column exists
if 'target' in data.columns:
    print("\nðŸ”¹ Target Variable Distribution:")
    print(data['target'].value_counts())
    sns.countplot(x='target', data=data)
    plt.title("Target Variable Distribution")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Histograms for numeric columns
data.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()



sample_submission.head()


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
channels = ['channel_44', 'channel_45', 'channel_46']

for i, channel in enumerate(channels):
    sns.histplot(data[channel], kde=True, ax=axes[i], bins=100)
    axes[i].set_title(f'{channel} Distribution')
    axes[i].set_xlabel('Value')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=data[channels], orient='h')
plt.title('Channel Value Distributions')
plt.show()


sample = data.sample(5000, random_state=42).sort_index()

plt.figure(figsize=(15, 10))
for i, channel in enumerate(channels, 1):
    plt.subplot(3, 1, i)
    plt.plot(sample['id'], sample[channel], lw=0.5)
    plt.title(f'{channel} Time Series')
    plt.xlabel('ID (Time)')
plt.tight_layout()
plt.show()


window = 1000
plt.figure(figsize=(15, 10))

for i, channel in enumerate(channels, 1):
    plt.subplot(3, 1, i)
    data[channel].rolling(window).mean().plot(label='Rolling Mean', alpha=0.7)
    data[channel].rolling(window).std().plot(label='Rolling Std', alpha=0.7)
    plt.legend()
    plt.title(f'{channel} Rolling Mean & Std Dev')
plt.tight_layout()
plt.show()


sns.pairplot(data.sample(10000)[channels], plot_kws={'alpha': 0.1})
plt.suptitle('Pairwise Channel Relationships', y=1.02)
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(data[channels].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Channel Correlation Matrix')
plt.show()


from scipy.stats import zscore

z_scores = data[channels].apply(zscore)
outliers = (z_scores.abs() > 4).any(axis=1)
print(f"Anomaly candidates: {outliers.sum()} ({outliers.sum()/len(data):.4f}%)")


plt.figure(figsize=(12, 6))
for channel in channels:
    plt.scatter(data.loc[outliers, 'id'], data.loc[outliers, channel], 
                s=5, alpha=0.5, label=channel)
plt.title('Anomaly Candidates Across Channels')
plt.xlabel('ID (Time)')
plt.legend()
plt.show()


from sklearn.decomposition import PCA

pca = PCA()
pca_result = pca.fit_transform(data[channels].sample(10000))

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], s=1, alpha=0.1)
plt.xlabel('PC1 (Variance: {:.2f}%)'.format(pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC2 (Variance: {:.2f}%)'.format(pca.explained_variance_ratio_[1]*100))
plt.title('PCA: First Two Principal Components')
plt.show()

