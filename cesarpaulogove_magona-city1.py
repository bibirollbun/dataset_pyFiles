# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np  # numerical operations
import pandas as pd  # data handling and CSV I/O

# Input data files are available in the read-only "../input/" directory
# For example, running this cell will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/)
# Temporary files can be written to /kaggle/temp/, but they won't persist outside the current session


# Useful libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Reading the main datasets
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')
competition_tags = pd.read_csv('/kaggle/input/meta-kaggle/CompetitionTags.csv')



print(users.columns)


# Convert registration date
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')
users['Year'] = users['RegisterDate'].dt.year

# Group by registration year
users_per_year = users.groupby('Year').size()

# Plot
users_per_year.plot(kind='bar', color='#1f77b4')
plt.title("Number of new Kaggle users per year")
plt.xlabel("Year")
plt.ylabel("Number of users")
plt.tight_layout()
plt.savefig("new_kaggle_users_per_year.png")
plt.show()


print(competition_tags.columns)



# Rename 'Id' column in tags to 'TagId'
tags_renamed = tags.rename(columns={'Id': 'TagId'})

# Merge tags with competition_tags
comp_tags = competition_tags.merge(tags_renamed, on='TagId', how='left')

# Count the most frequent tags
tag_counts = comp_tags['Name'].value_counts().head(20)

# Plot
tag_counts.plot(kind='bar', color='#2ca02c')
plt.title("Most frequent tags in Kaggle competitions")
plt.xlabel("Tag (Theme)")
plt.ylabel("Number of competitions")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("frequent_tags_in_Kaggle.png")
plt.show()


print(tags.columns)



# List of themes relevant to the Magona City project
magona_themes = ['education', 'sustainability', 'climate', 'energy', 'health', 'agriculture', 'environment']

# Ensure the 'Name' column is in lowercase for safe comparison
comp_tags['NameLower'] = comp_tags['Name'].str.lower()

# Filter only tags that are in the list of relevant themes
filtered_themes = comp_tags[comp_tags['NameLower'].isin(magona_themes)]

# Count by theme
filtered_themes['NameLower'].value_counts().plot(kind='bar', color='#17becf')
plt.title("Kaggle competitions by themes relevant to Magona City")
plt.xlabel("Theme")
plt.ylabel("Number of competitions")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("themes_relevant.png")
plt.show()


# Merge the filtered themes with the competitions using the Competition ID
detailed_comp = filtered_themes.merge(
    competitions,
    left_on='CompetitionId',
    right_on='Id',
    how='left'
)

# Select important columns for analysis
detailed_comp = detailed_comp[[
    'Name',              # Tag name
    'Title',             # Competition title
    'EnabledDate',       # Start date
    'RewardQuantity',    # Award value
    'RewardType'         # Type of reward
]]

# Rename columns for clarity
detailed_comp.columns = [
    'Tag',
    'Competition Title',
    'Start Date',
    'Award Value',
    'Award Type'
]

# Sort competitions by most recent first
detailed_comp = detailed_comp.sort_values(by='Start Date', ascending=False)

# Display the 15 most recent competitions
detailed_comp.head(15)



# Ensure the date is in datetime format
detailed_comp['Start Date'] = pd.to_datetime(detailed_comp['Start Date'], errors='coerce')

# Extract the year
detailed_comp['Year'] = detailed_comp['Start Date'].dt.year

# Count number of competitions per year and theme
yearly_theme_evolution = detailed_comp.groupby(['Year', 'Tag']).size().unstack(fill_value=0)

# Plot a line chart showing evolution per theme
yearly_theme_evolution.plot(marker='o')
plt.title('Annual evolution of Kaggle competitions by theme relevant to Magona City')
plt.xlabel('Year')
plt.ylabel('Number of competitions')
plt.legend(title='Theme')
plt.grid(True)
plt.tight_layout()
plt.savefig("Annual_evolution_of_Kaggle_competitions.png")
plt.show()

