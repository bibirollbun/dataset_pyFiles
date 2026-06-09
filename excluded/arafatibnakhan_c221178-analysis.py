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


#import libraries & load data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

#load dataset
train=pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')
test=pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv')
submission= pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/Submission.csv')
print("Train shape:",train.shape)
print("Test shape:",test.shape)
train.head()


# ðŸ“Š Step 3: Data Visualization (EDA)
# ============================================================

# Target distribution
sns.countplot(x='satisfaction', data=train)
plt.title("Target Class Distribution")
plt.show()

# Correlation heatmap (numerical)
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
plt.figure(figsize=(10, 6))
sns.heatmap(train[numeric_cols].corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


import pandas as pd

train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

print(train_data.head())

print(train_data.info())

print("\nMissing Values:\n", train_data.isnull().sum())


print("\nSummary Statistics:\n", train_data.describe())
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    print(f"\n{col} unique values: {train_data[col].unique()}")


numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns.drop(['id'])

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='satisfaction', y=col, data=train_data)
    plt.title(f'{col} vs Satisfaction')
    plt.tight_layout()
    plt.show()

