# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

df=pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')
df.head()


import pandas as pd

# Load the dataset
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Separate male and female data
df_male = df[df['Gender'] == 'Male']
df_female = df[df['Gender'] == 'Female']

# Save them separately if needed
df_male.to_csv('/kaggle/working/male_passengers.csv', index=False)
df_female.to_csv('/kaggle/working/female_passengers.csv', index=False)

# Show basic counts
print("Number of Male Passengers:", len(df_male))
print("Number of Female Passengers:", len(df_female))



import pandas as pd

# Load the dataset
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Group by gender and calculate average age
average_age_by_gender = df.groupby('Gender')['Age'].mean()

# Display the result
print("Average Age by Gender:")
print(average_age_by_gender)



import pandas as pd

# Load the dataset
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Group by 'Class' and calculate the average 'Flight Distance'
avg_distance_by_class = df.groupby('Class')['Flight Distance'].mean()

# Display the result
print("Average Flight Distance by Class:")
print(avg_distance_by_class)

# Get the class with the highest average
max_class = avg_distance_by_class.idxmax()
max_value = avg_distance_by_class.max()

print(f"\nðŸš€ Class with Highest Average Flight Distance: {max_class} ({max_value:.2f})")



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Create a cross-tabulation
cross_tab = pd.crosstab(df['Type of Travel'], df['Customer Type'])

print("ðŸ”¢ Distribution Table:\n")
print(cross_tab)

# Optional: visualize using a heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
plt.title('Customer Type Distribution Across Travel Types')
plt.ylabel('Type of Travel')
plt.xlabel('Customer Type')
plt.tight_layout()
plt.show()



import pandas as pd

# Load the dataset
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Check the exact column names (you can uncomment this if needed)
# print(df.columns)

# Calculate correlation
correlation = df['Inflight wifi service'].corr(df['Inflight entertainment'])

print(f"ðŸ“ˆ Correlation between Inflight Wifi Service and Inflight Entertainment: {correlation:.2f}")



import pandas as pd

# Load the dataset
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Drop non-numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlation with 'Arrival Delay in Minutes'
correlation_with_delay = numeric_df.corr()['Arrival Delay in Minutes'].drop('Arrival Delay in Minutes')

# Get top 5 most positively/negatively correlated features
top_5 = correlation_with_delay.abs().sort_values(ascending=False).head(5)

print("ðŸ“Š Top 5 Features Most Correlated with Arrival Delay:")
print(top_5)


