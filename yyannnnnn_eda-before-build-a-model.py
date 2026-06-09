# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


training_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv",index_col = "id")
testing_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col = "id")
training_data.head(5)


training_data.info()


# check missing values
print(training_data.isnull().sum())


training_data.describe()


features_numeric = training_data.select_dtypes(include = ['number']).columns.tolist()
features_numeric


for feature in features_numeric:
    plt.figure(figsize = (12,5))
    
    plt.subplot(1,2,1)
    sns.histplot(training_data[feature], kde = True, bins = 10)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    sns.boxplot(x = training_data[feature])
    plt.title(f"Boxplot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for{feature}:")
    print(f"Skewness: {training_data[feature].skew():.2f}")


colors = sns.color_palette('husl', len(features_numeric))

rows = -(-len(features_numeric) // 3)
plt.figure(figsize = (20, 5 * rows))

for i,(col, color) in enumerate(zip(features_numeric, colors), 1):
    plt.subplot(rows,4, i)
    sns.kdeplot(data = training_data, x = col, fill = True, color = color)
    plt.title(f"KDE Plot of {col}", fontsize = 14, color = color)
    plt.xlabel(col)
    plt.ylabel("Density")

plt.tight_layout()
plt.show


variables = ['Soil Type', 'Crop Type', 'Fertilizer Name']

for var in variables:
    counts = training_data[var].value_counts()

    plt.figure(figsize = (6,6))
    # circle = plt.Circle((0,0),0.7,color = "white")

    plt.pie(counts, labels = counts.index, autopct = "%1.2f%%",startangle = 90, textprops = {'fontsize': 14})

    plt.title(f"Distribution of {var}",fontsize = 16)
    plt.axis("equal")
    plt.show()

    print(f"Number of Unique {var}: {training_data[var].nunique()}")

