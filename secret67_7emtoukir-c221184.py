# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# Replace with your actual file path
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Show the first few rows
df.head(10)


# Group by satisfaction and count the number of passengers in each category
satisfaction_counts = df['satisfaction'].value_counts()
print("Satisfaction Counts:\n", satisfaction_counts)

# Group the entire dataframe by satisfaction
satisfaction_group = df.groupby('satisfaction')

# Calculate the mean for each group
satisfaction_mean = satisfaction_group.mean(numeric_only=True)
print("\nAverage Ratings per Satisfaction Group:\n", satisfaction_mean)

# Optional: Show more descriptive statistics
satisfaction_stats = satisfaction_group.describe()
print("\nFull Statistics per Satisfaction Group:\n", satisfaction_stats)


import matplotlib.pyplot as plt
import seaborn as sns


# Countplot for satisfaction
plt.figure(figsize=(6, 4))
sns.countplot(x='satisfaction', data=df, palette='Set2')
plt.title('Passenger Satisfaction Count')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Passengers')
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x='satisfaction', y='Inflight wifi service', data=df, palette='coolwarm')
plt.title('Inflight Wifi Service Rating by Satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Rating')
plt.show()


print(df.columns)


# Group by Gender and Satisfaction
gender_satisfaction_counts = df.groupby(['Gender', 'satisfaction']).size().unstack()

print("Satisfaction count by gender:\n", gender_satisfaction_counts)


# Plot stacked bar chart
gender_satisfaction_counts.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='Paired')
plt.title('Satisfaction by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.legend(title='Satisfaction')
plt.xticks(rotation=0)
plt.show()


# Pie chart for Male
male_data = df[df['Gender'] == 'Male']['satisfaction'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(male_data, labels=male_data.index, autopct='%1.1f%%', startangle=140)
plt.title('Satisfaction Distribution - Male')
plt.show()

# Pie chart for Female
female_data = df[df['Gender'] == 'Female']['satisfaction'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(female_data, labels=female_data.index, autopct='%1.1f%%', startangle=140)
plt.title('Satisfaction Distribution - Female')
plt.show()

