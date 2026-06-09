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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')


competitions.head()


competitions.tail()


competitions.shape


competitions.columns


competitions.info()


competitions.describe()


competitions.describe(include='object')


competitions.isnull().sum().sort_values(ascending=False)



missing_values = competitions.isnull().sum()
missing_percentage = (missing_values / len(competitions)) * 100


missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage
}).sort_values(by='Missing Values', ascending=False)


missing_df.head(15)



columns_to_drop = [
    'HostName', 'ValidationSetName', 'ValidationSetValue',
    'ModelSubmissionDeadlineDate', 'TeamModelDeadlineDate',
    'TeamMergerDeadlineDate', 'ProhibitNewEntrantsDeadlineDate',
    'OrganizationId' 
]


competitions_clean = competitions.drop(columns=columns_to_drop)



print("New shape:", competitions_clean.shape)


competitions_clean.isnull().sum().sort_values(ascending=False).head(10)




competitions_clean.isnull().sum().loc[lambda x: x > 0]




competitions_clean['EnabledDate'] = pd.to_datetime(competitions_clean['EnabledDate'], errors='coerce')


competitions_clean['Year'] = competitions_clean['EnabledDate'].dt.year


competitions_clean[['EnabledDate', 'Year']].head()




reward_data = competitions_clean[['RewardType', 'RewardQuantity', 'Year']].dropna()


competitions_clean[['EnabledDate', 'Year']].tail()



import matplotlib.pyplot as plt
import seaborn as sns


yearly_counts = competitions_clean['Year'].value_counts().sort_index()


sns.set(style='whitegrid')


plt.figure(figsize=(12, 6))
sns.barplot(x=yearly_counts.index, y=yearly_counts.values, palette='coolwarm')


plt.title('Number of Kaggle Competitions Per Year (2010â€“2025)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Competitions', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



sns.set(style='whitegrid')
plt.figure(figsize=(10, 5))
sns.countplot(data=competitions_clean, y='RewardType', order=competitions_clean['RewardType'].value_counts().index)
plt.title('Reward Type Distribution')
plt.xlabel('Count')
plt.ylabel('Reward Type')
plt.tight_layout()
plt.show()



reward_data = competitions_clean[['Year', 'RewardQuantity']].dropna()

plt.figure(figsize=(12, 6))
sns.boxplot(data=reward_data, x='Year', y='RewardQuantity')
plt.title('How Reward Amounts Changed Over the Years')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



teams_per_year = competitions_clean.groupby('Year')['TotalTeams'].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=teams_per_year.index, y=teams_per_year.values, marker='o')
plt.title('Avg. Number of Teams Per Competition (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Avg. Total Teams')
plt.tight_layout()
plt.show()



competitors_per_year = competitions_clean.groupby('Year')['TotalCompetitors'].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=competitors_per_year.index, y=competitors_per_year.values, marker='o', color='orange')
plt.title('Avg. Number of Competitors Per Competition (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Avg. Total Competitors')
plt.tight_layout()
plt.show()




top_rewards = competitions_clean[['Title', 'Year', 'RewardQuantity']].dropna()


top_rewards_sorted = top_rewards.sort_values(by='RewardQuantity', ascending=False).head(10)


plt.figure(figsize=(10, 6))
sns.barplot(data=top_rewards_sorted, y='Title', x='RewardQuantity', palette='magma')
plt.title('Top 10 Highest Paying Kaggle Competitions')
plt.xlabel('Reward ($)')
plt.ylabel('Competition Title')
plt.tight_layout()
plt.show()





