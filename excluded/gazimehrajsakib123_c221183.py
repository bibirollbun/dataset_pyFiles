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
df.head()


# Remove rows where Arrival Delay in Minutes is 0
df = df[df['Departure Delay in Minutes'] != 0]

# Check the result
df.head(25)


# Filter rows where Gender is Female
female_passengers = df[df['Gender'] == 'Female']

# Show the first few
female_passengers.head()


# Count
num_females = female_passengers.shape[0]
print(f"Number of female passengers: {num_females}")


# Filter rows where Gender is Male
male_passengers = df[df['Gender'] == 'Male']

# Show the first few rows
male_passengers.head()


# Count
num_males = male_passengers.shape[0]
print(f"Number of male passengers: {num_males}")


# Filter for personal travel
personal_travel = df[df['Type of Travel'] == 'Personal Travel']

# Count males and females in personal travel
personal_gender_counts = personal_travel['Gender'].value_counts()

print(personal_gender_counts)


import matplotlib.pyplot as plt


import seaborn as sns
import matplotlib.pyplot as plt

# Filter for personal travel
personal_travel = df[df['Type of Travel'] == 'Personal Travel']

# Count males and females
personal_gender_counts = personal_travel['Gender'].value_counts().reset_index()
personal_gender_counts.columns = ['Gender', 'Count']

# Plot
plt.figure(figsize=(8,6))
sns.barplot(x='Gender', y='Count', data=personal_gender_counts, palette='pastel', order=['Male', 'Female'])
plt.title('Number of Male and Female Passengers (Personal Travel)')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


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


# Countplot for satisfaction
plt.figure(figsize=(6, 4))
sns.countplot(x='satisfaction', data=df, palette='Set2')
plt.title('Passenger Satisfaction Count')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Passengers')
plt.show()

