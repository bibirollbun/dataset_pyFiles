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
import plotly.express as px

# Load core datasets
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
teams = pd.read_csv('/kaggle/input/meta-kaggle/Teams.csv')
#submissions = pd.read_csv('/kaggle/input/meta-kaggle/Submissions.csv', low_memory=False)
kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')
kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
kernel_languages = pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')
tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')



print(competitions.columns.tolist())


# Fix datetime formats using correct column names
competitions['year'] = pd.to_datetime(competitions['DeadlineDate'], errors='coerce').dt.year
kernel_versions['year'] = pd.to_datetime(kernel_versions['CreationDate'], errors='coerce').dt.year


competitions['year'].value_counts().sort_index().plot(kind='bar', figsize=(12, 5), title='Competitions per Year')
plt.xlabel('Year')
plt.ylabel('Count')


users['Country'].value_counts().head(10).sort_values().plot(kind='barh', title='Top 10 Countries by User Count')


kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')

print("KernelVersions columns:\n", kernel_versions.columns.tolist())


kernel_languages = pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')


# Merge language info
kv_lang = pd.merge(kernel_versions, kernel_languages, left_on='ScriptLanguageId', right_on='Id', how='left')


# Count language usage
lang_counts = kv_lang['Name'].value_counts()
labels = lang_counts.index
sizes = lang_counts.values

# Compute % manually
percentages = [f"{label} ({size / sum(sizes) * 100:.1f}%)" for label, size in zip(labels, sizes)]

# Plot without labels in chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, _ = ax.pie(
    sizes,
    startangle=140
)

# Add legend with labels + %
ax.legend(
    wedges,
    percentages,
    title="Languages",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)

plt.title('Kernel Languages Used in All Kernel Versions')
plt.tight_layout()
plt.show()


# Convert CreationDate to datetime (if not already)
kernel_versions['CreationDate'] = pd.to_datetime(kernel_versions['CreationDate'], errors='coerce')

# Extract year into a new column
kernel_versions['year'] = kernel_versions['CreationDate'].dt.year

# Filter LightGBM kernels
lightgbm_kernels = kernel_versions[kernel_versions['Title'].str.contains('lightgbm', case=False, na=False)]

# Plot LightGBM mentions over time
lightgbm_kernels.groupby('year').size().plot(
    marker='o',
    linestyle='-',
    figsize=(10, 5),
    title='LightGBM Mentions in Kernels Over Time'
)

plt.xlabel('Year')
plt.ylabel('Number of Kernels Mentioning LightGBM')
plt.grid(True)
plt.tight_layout()
plt.show()



print(teams.columns.tolist())


print([df_name for df_name in globals() if isinstance(eval(df_name), pd.DataFrame)])


print(users.columns)


team_memberships = pd.read_csv('/kaggle/input/meta-kaggle/TeamMemberships.csv')


team_sizes = team_memberships.groupby('TeamId').size().reset_index(name='TeamSize')
teams = teams.merge(team_sizes, left_on='Id', right_on='TeamId', how='left')


# Extract year from competitions (if not already done)
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['year'] = competitions['EnabledDate'].dt.year

# Merge teams with competitions to get the year
teams_with_year = pd.merge(teams, competitions[['Id', 'year']], left_on='CompetitionId', right_on='Id', how='left')

# Plot average team size per year
teams_with_year.groupby('year')['TeamSize'].mean().plot(
    marker='o',
    title='Average Team Size per Year'
)
plt.xlabel('Year')
plt.ylabel('Avg Team Size')
plt.grid(True)
plt.tight_layout()
plt.show()


tagged_kernels = pd.merge(kernel_tags, tags, left_on='TagId', right_on='Id')
tagged_kernels['Name'].value_counts().head(10).plot(kind='bar', title='Top 10 Kernel Tags')


import pandas as pd
submission = pd.read_csv('/kaggle/input/meta-kaggle/Submissions.csv', low_memory=False)

submission.to_csv('submission.csv', index=False)


