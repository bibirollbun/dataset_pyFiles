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

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')



train.head()


#Inspect Data:


print(train.info())
print(train.describe())


#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')

# Calculate percentage of missing values
missing_data = train.isnull().mean() * 100

# Plot missing data as a bar plot
plt.figure(figsize=(10, 6))
missing_data.plot(kind='bar', color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage Missing (%)')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print missing data summary
print("Missing Data Summary (%):")
print(missing_data)


#Visualize Target Distribution
sns.countplot(x='Personality', data=train)
plt.title('Distribution of Personality (Extrovert vs. Introvert)')
plt.show()




#Feature Distributions:For numerical features:
#For numerical features

numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                 'Friends_circle_size', 'Post_frequency']
train[numerical_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()





#boolean features:
for col in ['Stage_fear', 'Drained_after_socializing']:
    sns.countplot(x=col, hue='Personality', data=train)
    plt.title(f'{col} vs. Personality')
    plt.show()



#Correlation Analysis:
# Convert Personality to binary (Extrovert=1, Introvert=0) for correlation
train['Personality_binary'] = train['Personality'].map({'Extrovert': 1, 'Introvert': 0})
sns.heatmap(train[numerical_cols + ['Personality_binary']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



#Feature vs. Target Analysis:

for col in numerical_cols:
    sns.boxplot(x='Personality', y=col, data=train)
    plt.title(f'{col} by Personality')
    plt.show()

