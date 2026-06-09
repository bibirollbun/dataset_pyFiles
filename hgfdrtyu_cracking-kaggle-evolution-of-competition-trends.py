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


import os

for dirname, _, filenames in os.walk('/kaggle/input/meta-kaggle'):
    for filename in filenames:
        print(filename)


import pandas as pd

# Load language-related datasets
kernel_languages = pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')
kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')

# Preview the data
kernel_languages.head()



kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', low_memory=False)
kernel_versions.columns


import pandas as pd
import matplotlib.pyplot as plt

# Load competitions dataset
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')

# View columns
competitions.columns


import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
kernel_languages = pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')
kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', low_memory=False)

# Merge datasets
merged = pd.merge(kernel_versions, kernel_languages, left_on='ScriptLanguageId', right_on='Id')
merged['CreationDate'] = pd.to_datetime(merged['CreationDate'], errors='coerce')
merged['Year'] = merged['CreationDate'].dt.year

# Group by year and language
lang_trends = merged.groupby(['Year', 'DisplayName']).size().unstack().fillna(0)

# Filter only Python and R
lang_trends = lang_trends[['Python', 'R']]

# Stacked area plot
plt.figure(figsize=(12, 6))
lang_trends.plot(kind='area', stacked=True, alpha=0.8, colormap='viridis')
plt.title("Python vs R Usage in Kaggle Kernels", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Kernels", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Language")
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Count competitions per year
comp_per_year = competitions['Year'].value_counts().sort_index()

# Highlight the peak year
peak_year = comp_per_year.idxmax()
colors = ['steelblue' if year != peak_year else 'crimson' for year in comp_per_year.index]

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(comp_per_year.index, comp_per_year.values, color=colors)
plt.title("Number of Kaggle Competitions Per Year", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Competitions", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Annotate the peak year
plt.text(peak_year, comp_per_year[peak_year] + 2,
         f'Peak Year: {peak_year}',
         ha='center', va='bottom', color='crimson', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Filter for monetary rewards
money_types = ['USD', 'EUR', 'GBP']
money_comps = competitions[
    competitions['RewardType'].isin(money_types) & competitions['RewardQuantity'].notnull()
]
avg_rewards = money_comps.groupby('Year')['RewardQuantity'].mean()

# Plot
plt.figure(figsize=(12, 6))
plt.fill_between(avg_rewards.index, avg_rewards.values, color='gold', alpha=0.3)
plt.plot(avg_rewards.index, avg_rewards.values, color='orange', marker='o', linewidth=2)

# Annotate peak
peak_year = avg_rewards.idxmax()
peak_value = avg_rewards.max()
plt.text(peak_year, peak_value + 5000, f'Peak: ${peak_value:,.0f}', ha='center', color='darkorange', fontsize=10)

# Style
plt.title("Average Monetary Prize per Kaggle Competition (by Year)", fontsize=20)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Reward ($ or equivalent)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



import pandas as pd
import plotly.express as px

# Load Users and Kernels
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv', low_memory=False)
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv', low_memory=False)

# Merge on UserId to get country info
merged = kernels.merge(users[['Id', 'Country']], left_on='AuthorUserId', right_on='Id')

# Count kernels by country
country_kernel_counts = merged['Country'].value_counts().reset_index()
country_kernel_counts.columns = ['Country', 'KernelCount']

# Plot choropleth map
fig = px.choropleth(
    country_kernel_counts,
    locations='Country',
    locationmode='country names',
    color='KernelCount',
    color_continuous_scale='Viridis',
    title=' Countries Creating the Most Kaggle Kernels'
)

fig.update_layout(
    geo=dict(showframe=False, projection_type='equirectangular')
)

fig.show()


# Load & preprocess
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'])
competitions['Year'] = competitions['EnabledDate'].dt.year

# Map CompetitionTypeId to readable labels if needed (manual or lookup)
type_counts = competitions.groupby(['Year', 'CompetitionTypeId']).size().unstack().fillna(0)

# Plot
type_counts.plot(kind='area', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title('Growth of Different Competition Types Over Time',fontsize=20)
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.legend(title='Competition Type ID')
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load users
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')

# Clean and count
country_counts = users['Country'].value_counts().dropna().head(10)

# Plot
plt.figure(figsize=(10, 6))
country_counts.sort_values().plot(kind='barh', color='royalblue')
plt.title("Top 10 Countries by Number of Kaggle Users",fontsize=20)
plt.xlabel("Number of Users")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
kernel_votes = pd.read_csv('/kaggle/input/meta-kaggle/KernelVotes.csv')
kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', low_memory=False)
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv', low_memory=False)
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv', low_memory=False)

# Step 1: Map KernelVersionId to ScriptId (kernel ID)
merged_votes = kernel_votes.merge(
    kernel_versions[['Id', 'ScriptId']],
    left_on='KernelVersionId',
    right_on='Id',
    how='left'
)

# Step 2: Count votes per ScriptId (kernel ID)
vote_counts = merged_votes['ScriptId'].value_counts().reset_index()
vote_counts.columns = ['KernelId', 'VoteCount']

# Step 3: Merge with kernel info
top_kernels = vote_counts.merge(
    kernels[['Id', 'CurrentUrlSlug', 'AuthorUserId']],
    left_on='KernelId',
    right_on='Id',
    how='left'
)

# Step 4: Merge with user info
top_kernels = top_kernels.merge(
    users[['Id', 'UserName']],
    left_on='AuthorUserId',
    right_on='Id',
    how='left'
)
top_kernels.rename(columns={'UserName': 'Author'}, inplace=True)

# Step 5: Top 10 kernels
top10_kernels = top_kernels.sort_values(by='VoteCount', ascending=False).head(10)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top10_kernels, y='CurrentUrlSlug', x='VoteCount', palette='crest')
plt.title('Top 10 Most Voted Kaggle Kernels (All Versions Combined)', fontsize=16)
plt.xlabel('Vote Count')
plt.ylabel('Kernel Slug')
plt.tight_layout()
plt.show()




forum_votes = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv')
forum_votes['VoteDate'] = pd.to_datetime(forum_votes['VoteDate'], errors='coerce')
forum_votes['Year'] = forum_votes['VoteDate'].dt.year

# Count votes per year
yearly_votes = forum_votes['Year'].value_counts().sort_index()

# Plot
plt.figure(figsize=(10,5))
sns.lineplot(x=yearly_votes.index, y=yearly_votes.values, marker='o', color='crimson')
plt.title('Forum Engagement Over Time (Votes)')
plt.xlabel('Year')
plt.ylabel('Number of Forum Message Votes')
plt.grid(True)
plt.tight_layout()
plt.show()


