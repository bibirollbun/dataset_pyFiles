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
train_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


# for beautiful dataframes
from IPython.display import display

# for heatmaps
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
%matplotlib inline

# for density plots and histograms
import matplotlib.pyplot as plt

# for maths and other operations
import numpy as np


display(train_csv.head())
train_csv.shape, test_csv.shape


# missing values in the datasets
train_null = train_csv.isnull()
test_null = test_csv.isnull()

# percentage of missing values per feature in the training set
train_not_null = (len(train_csv) - train_null.sum()) / len(train_csv) *100
test_not_null = (len(test_csv) - test_null.sum()) / len(test_csv) *100

# create a single figure with subplots for both datasets
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# heatmap for missing values in the training dataset
sns.heatmap(train_null, cmap='viridis', cbar=False, yticklabels=False, ax=axes[0])
axes[0].set_xlabel('Training features + completeness (%)', fontsize=12)
axes[0].set_ylabel('Entries, yellow=missing', fontsize=12)

# annotate training heatmap columns
for i in range(len(train_null.columns)):
    axes[0].text(i + 0.5, -0.5, f"{train_not_null.iloc[i]:.2f}", ha='center', va='bottom')

# heatmap for missing values in the test dataset
sns.heatmap(test_null, cmap='viridis', cbar=False, yticklabels=False, ax=axes[1])
axes[1].set_xlabel('Test features + Completeness (%)', fontsize=12)
axes[1].set_ylabel('Entries, yellow=missing', fontsize=12)

# annotate test heatmap columns
for i in range(len(test_null.columns)):
    axes[1].text(i + 0.5, -0.5, f"{test_not_null.iloc[i]:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()



train_csv.info()


# grid for the countplots of the categorical values
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 20))

# switch between train/test
df = test_csv

# flatten axes array for easy iteration
axes = axes.flatten()

# define the categorical columns to inspect
cat_feats = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# create countplots
for i, col in enumerate(cat_feats):
    ax = axes[i]
    sns.countplot(x=df[col], ax=ax)

    # add numbers on top of bars
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=10, color='black')

# remove the empty subplot
fig.delaxes(axes[7])

plt.tight_layout()
plt.show()


# grid for the countplots of the categorical values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# switch between train/test
df = train_csv

# flatten axes array for easy iteration
axes = axes.flatten()

# the continuous features to inspect
cont_feats = ['Compartments', 'Weight Capacity (kg)', 'Price']

# create countplots
for i, col in enumerate(cont_feats):
    ax = axes[i]

    # histogram
    df[col].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6, ax=ax)

    # density plot
    df[col].plot(kind='density', color='teal', ax=ax)

    # mean and median vertical lines
    mean = df[col].mean(skipna=True)
    median = df[col].median(skipna=True)
    ax.axvline(mean, color='r', label=f'Mean: {mean:.3f}')
    ax.axvline(median, color='y', label=f'Median: {median:.3f}')

    # labels and formatting
    ax.set(xlabel=col)
    ax.legend()

# remove the empty subplot
fig.delaxes(axes[3])

plt.tight_layout()
plt.show()


# make copies of our data
train_full = train_csv.copy()
test_full = test_csv.copy()

# infer the NaN with the most common categories
for col in cat_feats:
    test_full[col] = test_full[col].fillna(test_full[col].value_counts().idxmax())

# infer the NaN with the mean
train_full['Weight Capacity (kg)'] = train_full['Weight Capacity (kg)'].fillna(train_full['Weight Capacity (kg)'].mean(skipna=True))
test_full['Weight Capacity (kg)'] = test_full['Weight Capacity (kg)'].fillna(test_full['Weight Capacity (kg)'].mean(skipna=True))

# drop the remaining train NaN just for now to see the possible improvements by inferring them
train_full = train_full.dropna(subset=['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'])

# final check for null values
if (train_full.isnull().sum() == 0).all() and (test_full.isnull().sum() == 0).all():
    print('SUCCESS: Datasets clear from NaN values')
    print(train_full.shape, test_full.shape)
else:
    print('ERROR: Datasets NOT clear from NaN values')



    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    sns.histplot(data=original_data.dropna(), x=variable, color=custom_palette[2], kde=True, bins=30, label="Original")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each variable
for variable in variables:
    create_variable_plots(variable)

# Drop the 'Dataset' column after analysis
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)

