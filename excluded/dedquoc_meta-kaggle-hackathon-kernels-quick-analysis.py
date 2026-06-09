# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


%%time
# Download Meta Kaggle dataset
MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
print("Path to Meta Kaggle dataset files:", MK_PATH)

# Download Meta Kaggle Code dataset
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")
print("Path to Meta Kaggle Code dataset files:", MKC_PATH)


%%time
# Load key CSV files from Meta Kaggle dataset
kernels_df = pd.read_csv(f"{MK_PATH}/Kernels.csv")

# Quick overview of each dataframe
print("Kernels shape:", kernels_df.shape)

# Peek at columns of kernels_df to understand structure
kernels_df.head()


%%time
kernel_versions_df = pd.read_csv(f"{MK_PATH}/KernelVersions.csv")
print("KernelVersions shape:", kernel_versions_df.shape)


%%time
users_df = pd.read_csv(f"{MK_PATH}/Users.csv")
print("Users shape:", users_df.shape)


%%time
kernel_votes_df = pd.read_csv(f"{MK_PATH}/KernelVotes.csv")
print("KernelVotes shape:", kernel_votes_df.shape)


%%time
kernel_languages_df = pd.read_csv(f"{MK_PATH}/KernelLanguages.csv")
print("KernelLanguages shape:", kernel_languages_df.shape)


%%time
# Convert creation date columns to datetime
kernel_versions_df['CreationDate'] = pd.to_datetime(kernel_versions_df['CreationDate'], errors='coerce')

# Extract year-month for time series grouping
kernel_versions_df['YearMonth'] = kernel_versions_df['CreationDate'].dt.to_period('M')

# Check for missing values in important columns
print("Missing Created dates:", kernel_versions_df['CreationDate'].isna().sum())

kernel_versions_df.head()


%%time
# Count kernels created per month
kernels_per_month = kernel_versions_df.groupby('YearMonth').size().reset_index(name='Count')
kernels_per_month['YearMonth'] = kernels_per_month['YearMonth'].dt.to_timestamp()

# Plot
plt.figure()
sns.lineplot(data=kernels_per_month, x='YearMonth', y='Count')
plt.title('Number of Kernel Versions Created Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Kernel Versions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


%%time
# Count votes per kernel version
votes_per_kernel = kernel_votes_df.groupby('KernelVersionId').size().reset_index(name='VoteCount')

# Merge with kernel_versions to get kernel and author info
top_kernels = votes_per_kernel.merge(kernel_versions_df, left_on='KernelVersionId', right_on='Id')

# Merge with users to get author name
top_kernels = top_kernels.merge(users_df[['Id', 'DisplayName']], left_on='Id', right_on='Id', suffixes=('', '_user'))

# Sort by votes descending
top_kernels_sorted = top_kernels.sort_values('VoteCount', ascending=False).head(10)

# Plot
plt.figure()
sns.barplot(data=top_kernels_sorted, y='Title', x='VoteCount', palette='viridis')
plt.title('Top 10 Kernel Versions by Vote Count')
plt.xlabel('Number of Votes')
plt.ylabel('Kernel Title')
plt.tight_layout()
plt.show()


%%time
# Count kernels by language
language_counts = kernel_languages_df['Name'].value_counts().reset_index()
language_counts.columns = ['Name', 'Count']

# Plot
plt.figure()
sns.barplot(data=language_counts, x='Count', y='Name', palette='magma')
plt.title('Distribution of Kernel Languages Used')
plt.xlabel('Number of Kernels')
plt.ylabel('Language')
plt.tight_layout()
plt.show()


%%time

import pandas as pd

# Load datasets
kernel_versions = pd.read_csv(f"{MK_PATH}/KernelVersions.csv", low_memory=False)
kernel_votes = pd.read_csv(f"{MK_PATH}/KernelVotes.csv")
kernels = pd.read_csv(f"{MK_PATH}/Kernels.csv", low_memory=False)


%%time
# recheck related key between datasets to show good plot
kernel_votes_per_kernel = kernel_votes['Id'].value_counts().reset_index()
kernel_votes_per_kernel.columns = ['Id', 'UserId']

# Merge with kernel metadata
merged = kernels.merge(kernel_votes_per_kernel, left_on='Id', right_on='Id')
top_kernels = merged.sort_values('UserId', ascending=False).head(10)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Id', y='UserId', data=top_kernels)
plt.xticks(rotation=75)
plt.title("Top 10 Most Voted Kernels")
plt.tight_layout()
plt.show()


%%time
# Aggregate total votes per KernelVersionId
vote_counts = kernel_votes['Id'].value_counts().reset_index()
vote_counts.columns = ['Id', 'UserId']

# Merge KernelVersions with vote counts
kernels_with_votes = kernel_versions.merge(vote_counts, on='Id', how='left')
kernels_with_votes['UserId'] = kernels_with_votes['UserId'].fillna(0)

# Optional: Merge with base kernel info (e.g., author/user)
kernels_with_votes = kernels_with_votes.merge(
    kernels[['Id', 'AuthorUserId', 'CurrentKernelVersionId']],
    left_on='Id',
    right_on='Id',
    how='left'
)


kernels['Medal'].value_counts()


medal_map = {1.0: "Gold", 2.0: "Silver", 3.0: "Bronze"}
kernels["MedalStr"] = kernels["Medal"].map(medal_map)


%%time
medal_counts = kernels["MedalStr"].value_counts()
medal_order = ["Gold", "Silver", "Bronze"]
medal_counts = medal_counts.reindex(medal_order).dropna()

plt.figure(figsize=(6, 4))
sns.barplot(x=medal_counts.index, y=medal_counts.values, palette="YlOrBr")
plt.title("Count of Kernels by Medal Type")
plt.xlabel("Medal")
plt.ylabel("Number of Kernels")
plt.show()


%%time
votes_by_medal = kernels.groupby("MedalStr")["TotalVotes"].mean().reindex(["Gold", "Silver", "Bronze"])
plt.figure(figsize=(6,4))
sns.barplot(x=votes_by_medal.index, y=votes_by_medal.values, palette="YlOrBr")
plt.title("Average Total Votes by Medal Type")
plt.ylabel("Avg Votes")
plt.xlabel("Medal")
plt.show()


%%time
kernels["CreationDate"] = pd.to_datetime(kernels["CreationDate"])
monthly_kernels = kernels.resample("M", on="CreationDate").size()

plt.figure(figsize=(10, 4))
monthly_kernels.plot()
plt.title("Kernels Created Over Time")
plt.ylabel("Number of Kernels")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()

