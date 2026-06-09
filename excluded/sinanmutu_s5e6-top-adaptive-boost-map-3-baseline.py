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
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

train.head()


plt.figure(figsize=(12, 6))
sns.countplot(x='Fertilizer Name', data=train, order=train['Fertilizer Name'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Fertilizer Name Distribution')
plt.show()


numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
train[numerical_features].describe()


plt.figure(figsize=(12, 4))
sns.countplot(x='Soil Type', hue='Fertilizer Name', data=train)
plt.title('Fertilizer by Soil Type')
plt.show()

plt.figure(figsize=(12, 4))
sns.countplot(x='Crop Type', hue='Fertilizer Name', data=train)
plt.title('Fertilizer by Crop Type')
plt.show()


# Bin Nitrogen into ranges
train['Nitrogen_bin'] = pd.cut(train['Nitrogen'], bins=[0, 20, 40, 60, 80], labels=['low', 'mid', 'high', 'vhigh'])

# Check fertilizer distribution across Nitrogen bins
cross_tab = pd.crosstab(train['Nitrogen_bin'], train['Fertilizer Name'], normalize='index')
cross_tab


# Analyze fertilizer distribution by Soil Type
soil_cross = pd.crosstab(train['Soil Type'], train['Fertilizer Name'], normalize='index')
print("Soil Type vs Fertilizer Name")
display(soil_cross)

# Analyze fertilizer distribution by Crop Type
crop_cross = pd.crosstab(train['Crop Type'], train['Fertilizer Name'], normalize='index')
print("Crop Type vs Fertilizer Name")
display(crop_cross)


top3 = train['Fertilizer Name'].value_counts().index[:3]
print("Top-3 most frequent fertilizers:", list(top3))

sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
sample_submission['Fertilizer Name'] = ' '.join(top3)
sample_submission.to_csv('naive_top3.csv', index=False)


sample_submission.head()


# Binning Nitrogen
train['Nitrogen_bin'] = pd.cut(train['Nitrogen'], bins=[0, 20, 40, 60, 80], labels=['low', 'mid', 'high', 'vhigh'])
test['Nitrogen_bin'] = pd.cut(test['Nitrogen'], bins=[0, 20, 40, 60, 80], labels=['low', 'mid', 'high', 'vhigh'])

# Get top1 fertilizer per Soil Type
soil_counts = pd.crosstab(train['Soil Type'], train['Fertilizer Name'], normalize='index')
soil_top1 = soil_counts.idxmax(axis=1).to_dict()

# Get top1 fertilizer per Crop Type
crop_counts = pd.crosstab(train['Crop Type'], train['Fertilizer Name'], normalize='index')
crop_top1 = crop_counts.idxmax(axis=1).to_dict()

# Get top1 fertilizer per Nitrogen Bin
nitrogen_counts = pd.crosstab(train['Nitrogen_bin'], train['Fertilizer Name'], normalize='index')
nitrogen_top1 = nitrogen_counts.idxmax(axis=1).to_dict()

# Get general top3 fertilizers (global frequency)
general_top3 = train['Fertilizer Name'].value_counts().index[:3].tolist()

# Build submission list
submission_list = []
for _, row in test.iterrows():
    top1_soil = soil_top1.get(row['Soil Type'], general_top3[0])
    top1_crop = crop_top1.get(row['Crop Type'], general_top3[0])
    top1_nitrogen = nitrogen_top1.get(row['Nitrogen_bin'], general_top3[0])

    # Combine unique top1 picks + general top3, deduplicated and limited to 3
    combined = [top1_soil, top1_crop, top1_nitrogen] + general_top3
    combined_unique = []
    for fert in combined:
        if fert not in combined_unique:
            combined_unique.append(fert)
        if len(combined_unique) == 3:
            break

    submission_list.append(' '.join(combined_unique))

# Save submission file
sample_submission['Fertilizer Name'] = submission_list
sample_submission.to_csv('adaptive_rule_submit.csv', index=False)


sample_submission.head()

