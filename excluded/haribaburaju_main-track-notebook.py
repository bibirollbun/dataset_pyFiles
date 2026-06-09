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


# Starter Kaggle Notebook
# Meta Kaggle Hackathon 2025 – Main Track: Evolution of Winning Techniques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Optional: better display settings
pd.set_option('display.max_columns', 100)
sns.set(style="whitegrid")




import pandas as pd

#  Step 1: Load data
data_path = "/kaggle/input/main-track-data"

submissions = pd.read_csv(
    f"{data_path}/Submissions.csv",
    dtype={'ScoreDate': str, 'IsSelected': str},
    engine='python'
)

competitions = pd.read_csv(
    f"{data_path}/Competitions.csv",
    engine='python'
)

users = pd.read_csv(
    f"{data_path}/Users.csv",
    engine='python'
)

print(" Data Loaded")
print(" - Competitions:", competitions.shape)
print(" - Submissions:", submissions.shape)
print(" - Users:", users.shape)

#  Step 2: Merge only Submissions with Users (via SubmittedUserId)
submissions_merged = submissions.merge(
    users[['Id', 'UserName']],
    left_on='SubmittedUserId',
    right_on='Id',
    suffixes=('', '_User')
)

#  Step 3: Convert SubmissionDate to datetime
submissions_merged['SubmissionDate'] = pd.to_datetime(submissions_merged['SubmissionDate'], errors='coerce')

#  Step 4: Basic Insight — First submission per user
first_submissions = submissions_merged.sort_values('SubmissionDate').groupby('UserName').first().reset_index()

#  Preview
first_submissions[['UserName', 'SubmissionDate']].head()


# Competition Year Summary
# Convert EnabledDate to datetime
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Count competitions per year
comp_per_year = competitions.groupby('Year').size().reset_index(name='CompetitionCount')

# Plot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.barplot(data=comp_per_year, x='Year', y='CompetitionCount', palette='viridis')
plt.xticks(rotation=45)
plt.title('Competitions per Year')
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.grid(True)
plt.tight_layout()
plt.show()




# User Behavior Over Time Analysis
#Objective: Analyze how user participation has evolved on Kaggle based on their submission activity.

# Convert submission date:

submissions['SubmissionDate'] = pd.to_datetime(submissions['SubmissionDate'], errors='coerce')


# Merge Submissions with Users:

user_subs = submissions.merge(users, left_on='SubmittedUserId', right_on='Id')


# Extract submission year:

user_subs['Year'] = user_subs['SubmissionDate'].dt.year


#Count submissions per year per user (for behavioral trend analysis):

user_activity = user_subs.groupby(['UserName', 'Year']).size().reset_index(name='Submissions')


#Identify first submission year per user to track user acquisition:

first_submission = user_subs.groupby('UserName')['SubmissionDate'].min().dt.year.value_counts().sort_index()
first_submission = first_submission.reset_index()
first_submission.columns = ['Year', 'NewUsers']


 #Plot New User Join Trend:

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
sns.barplot(data=first_submission, x='Year', y='NewUsers', palette='crest')
plt.title('New Users Joined Kaggle (Based on First Submission Year)')
plt.xlabel('Year')
plt.ylabel('Count of New Users')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# User Behavior Analysis 
# Analyze Competition Trends Over Time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 0: Load Competitions data
competitions = pd.read_csv('/kaggle/input/main-track-data/Competitions.csv')

# Step 1: Convert Competition EnabledDate to datetime
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Step 2: Count number of competitions launched per year
competitions_per_year = competitions['Year'].value_counts().sort_index().reset_index()
competitions_per_year.columns = ['Year', 'CompetitionCount']

# Step 3: Clean the data
competitions_per_year.replace([np.inf, -np.inf], np.nan, inplace=True)
competitions_per_year.dropna(inplace=True)

# Step 4: Plot the competition trend
plt.figure(figsize=(12,6))
sns.lineplot(data=competitions_per_year, x='Year', y='CompetitionCount', marker='o')
plt.title('Number of Competitions Launched Per Year')
plt.xlabel('Year')
plt.ylabel('Competition Count')
plt.grid(True)
plt.tight_layout()
plt.show()


#  Analyze User Participation Behavior Over Time

#  Step 0: Load data from Meta Kaggle files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle")
print("Path to dataset files:", path)

notebooks = pd.read_csv('/kaggle/input/hari-data3/Kernels.csv')  # older name for CodeNotebooks
users = pd.read_csv('/kaggle/input/main-track-data/Users.csv')
comp_tags = pd.read_csv('/kaggle/input/hari-data2/CompetitionTags.csv')

#  Step 1: Merge user info to add 'UserName' to notebook records
notebooks = notebooks.merge(
    users[['Id', 'UserName']],
    how='left',
    left_on='AuthorUserId',
    right_on='Id'
)

#  Step 2: Convert creation date and extract year
notebooks['CreationDate'] = pd.to_datetime(notebooks['CreationDate'], errors='coerce')
notebooks['CreationYear'] = notebooks['CreationDate'].dt.year

#  Step 3: Count number of unique users per year
users_per_year = notebooks.groupby('CreationYear')['UserName'].nunique().reset_index()
users_per_year.columns = ['Year', 'UniqueUsers']

#  Step 4: Plot the user participation trend
plt.figure(figsize=(12,6))
sns.lineplot(data=users_per_year, x='Year', y='UniqueUsers', marker='o', color='green')
plt.title('Number of Unique Notebook Authors Per Year')
plt.xlabel('Year')
plt.ylabel('Unique Users')
plt.grid(True)
plt.tight_layout()
plt.show()


print(kernel_full_df.columns.tolist())


kernel_full_df[['ScriptId', 'AuthorUserId_kernel', 'VersionNumber', 'CreationDate_version', 'TotalVotes_kernel']].head()



# Model Usage Trend Analysis

import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
kernels_df = pd.read_csv("/kaggle/input/meta-kaggle/Kernels.csv", low_memory=False)
kernel_versions_df = pd.read_csv("/kaggle/input/hari-data4/KernelVersions.csv", low_memory=False)

# Merge on KernelId (ScriptId in KernelVersions matches Id in Kernels)
kernel_full_df = pd.merge(
    kernel_versions_df,
    kernels_df[['Id', 'AuthorUserId', 'TotalVotes', 'CreationDate']],
    left_on='ScriptId',
    right_on='Id',
    suffixes=('_version', '_kernel')
)

# Convert date fields
kernel_full_df['CreationDate_version'] = pd.to_datetime(kernel_full_df['CreationDate_version'], errors='coerce')

# Extract year-month for trend analysis
kernel_full_df['Month'] = kernel_full_df['CreationDate_version'].dt.to_period('M')

# Aggregate: Count of Kernel Versions per Month
monthly_kernel_versions = kernel_full_df.groupby('Month').size().reset_index(name='KernelVersionCount')

# Plot the trend
plt.figure(figsize=(14, 6))
plt.plot(monthly_kernel_versions['Month'].astype(str), monthly_kernel_versions['KernelVersionCount'], marker='o')
plt.title('Monthly Trend of Kernel Versions')
plt.xlabel('Month')
plt.ylabel('Number of Kernel Versions')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


#By Programming Language

lang_trend = kernel_full_df.groupby(['Month', 'ScriptLanguageId']).size().unstack(fill_value=0)
lang_trend.plot(figsize=(14,6), title='Monthly Kernel Versions by Language')
plt.xlabel("Month")
plt.ylabel("Kernel Versions")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


#Internet Enabled vs. Not:

net_trend = kernel_full_df.groupby(['Month', 'IsInternetEnabled']).size().unstack(fill_value=0)
net_trend.plot(kind='bar', stacked=True, figsize=(14,6), title='Internet Enabled Kernels Over Time')
plt.xlabel("Month")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



#GPU Usage Trend (AcceleratorTypeId):

gpu_trend = kernel_full_df.groupby(['Month', 'AcceleratorTypeId']).size().unstack(fill_value=0)
gpu_trend.plot(figsize=(14,6), title='Kernel Versions by Accelerator Type')
plt.xlabel("Month")
plt.ylabel("Kernel Versions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#Top Authors by Kernel Submissions:
top_authors = kernel_full_df['AuthorUserId_kernel'].value_counts().head(10)
top_authors.plot(kind='bar', title='Top 10 Kernel Authors by Count')
plt.ylabel("Kernel Count")
plt.tight_layout()
plt.show()



# User Achievement Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load with specific columns and use date parsing to save memory
use_cols = ['UserId', 'AchievementType', 'Tier', 'TierAchievementDate']
achievements_df = pd.read_csv(
    '/kaggle/input/meta-kaggle/UserAchievements.csv',
    usecols=use_cols,
    parse_dates=['TierAchievementDate'],
    low_memory=False
)

# Step 2: Filter for only recent achievements (last 2 years)
achievements_df = achievements_df[achievements_df['TierAchievementDate'] >= '2023-01-01']

# Step 3: If still large, sample
if len(achievements_df) > 100000:
    achievements_df = achievements_df.sample(100000, random_state=42)

# Step 4: Extract month
achievements_df['AchieveMonth'] = achievements_df['TierAchievementDate'].dt.to_period('M').astype(str)

# Step 5: Plot
plt.figure(figsize=(14, 6))
sns.countplot(data=achievements_df, x='AchieveMonth', hue='AchievementType',
              order=sorted(achievements_df['AchieveMonth'].unique()))
plt.xticks(rotation=45)
plt.title("User Achievements by Type Since 2023")
plt.xlabel("Month")
plt.ylabel("Achievement Count")
plt.tight_layout()
plt.show()


# User Followers analysis
import pandas as pd
import matplotlib.pyplot as plt

# Load UserFollowers.csv
followers_df = pd.read_csv("/kaggle/input/meta-kaggle/UserFollowers.csv")

# Check column names and preview
print("Columns:", followers_df.columns.tolist())
print(followers_df.head())

# Rename for clarity (optional)
followers_df.rename(columns={'UserId': 'FollowedUserId', 'FollowerUserId': 'FollowerId'}, inplace=True)

# Count number of followers each user has
follower_counts = followers_df['FollowedUserId'].value_counts().reset_index()
follower_counts.columns = ['UserId', 'NumFollowers']

# Plot top 20 users with the most followers
top_20 = follower_counts.head(20)
plt.figure(figsize=(12, 6))
plt.bar(top_20['UserId'].astype(str), top_20['NumFollowers'], color='skyblue')
plt.xticks(rotation=45)
plt.xlabel("User ID")
plt.ylabel("Number of Followers")
plt.title("Top 20 Most Followed Users on Kaggle")
plt.tight_layout()
plt.show()


# Forum Activity Trends

import pandas as pd
import matplotlib.pyplot as plt

# Load the ForumMessages data
forum_df = pd.read_csv("/kaggle/input/meta-kaggle/ForumMessages.csv", low_memory=False)

# Display initial few rows with correct columns
print(forum_df[['Id', 'ForumTopicId', 'PostUserId', 'PostDate']].head())

# Convert 'PostDate' to datetime
forum_df['PostDate'] = pd.to_datetime(forum_df['PostDate'], errors='coerce')

# Extract Year-Month for trend analysis
forum_df['YearMonth'] = forum_df['PostDate'].dt.to_period('M')

# Group by month and count messages
monthly_trend = forum_df.groupby('YearMonth').size().reset_index(name='MessageCount')

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(monthly_trend['YearMonth'].astype(str), monthly_trend['MessageCount'], marker='o', color='teal')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Number of Forum Messages')
plt.title('Kaggle Forum Activity Over Time')
plt.tight_layout()
plt.show()


# Forum Activity Trends
# 1. Top Forum Contributors
top_users = forum_df['PostUserId'].value_counts().head(10).reset_index()
top_users.columns = ['UserId', 'PostCount']

print(top_users)



import pandas as pd

# Load ForumMessages data again
forum_df = pd.read_csv("/kaggle/input/meta-kaggle/ForumMessages.csv")



# Forum Activity Trends

# Medals Awarded for Forum Posts

medal_counts = forum_df['Medal'].value_counts().reset_index()
medal_counts.columns = ['MedalType', 'Count']

print(medal_counts)



# Forum Activity Trends

# Trend of Medal Awards Over Time

# Drop rows with missing Medal, and make a deep copy
medal_trend = forum_df.dropna(subset=['Medal']).copy()

# Convert MedalAwardDate to datetime
medal_trend['MedalAwardDate'] = pd.to_datetime(medal_trend['MedalAwardDate'], errors='coerce')

# Extract Year-Month period
medal_trend['YearMonth'] = medal_trend['MedalAwardDate'].dt.to_period('M')

# Group and plot
monthly_medals = medal_trend.groupby(['YearMonth', 'Medal']).size().unstack(fill_value=0)

# Plot
monthly_medals.plot(kind='line', figsize=(14, 6), marker='o')
plt.title('Monthly Forum Medal Awards')
plt.xlabel('Month')
plt.ylabel('Medal Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Forum Activity Trends

# Code Snippet: Forum Votes Over Time

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
votes_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv')
print("Columns:", votes_df.columns.tolist())
votes_df.head()



#Forum Activity Trends
#  Check actual columns

import pandas as pd

# Read the dataset
forum_votes_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv')

# Show column names and a sample
print("Columns:", forum_votes_df.columns.tolist())
forum_votes_df.head()


# Forum Activity Trends
#Analyze Vote Trends Over Time

import matplotlib.pyplot as plt

# Convert VoteDate to datetime
forum_votes_df['VoteDate'] = pd.to_datetime(forum_votes_df['VoteDate'], errors='coerce')

# Filter for recent votes
recent_votes = forum_votes_df[forum_votes_df['VoteDate'] >= '2023-01-01'].copy()

# Extract year-month
recent_votes['YearMonth'] = recent_votes['VoteDate'].dt.to_period('M')

# Count monthly votes
monthly_votes = recent_votes.groupby('YearMonth').size()

# Plot the trend
monthly_votes.plot(kind='bar', figsize=(12, 5), color='skyblue')
plt.title('Monthly Forum Votes (Assumed Upvotes Only)')
plt.xlabel('Month')
plt.ylabel('Vote Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Forum Activity Trends
#Analyze Follower Trends

# Read UserFollowers data
followers_df = pd.read_csv('/kaggle/input/meta-kaggle/UserFollowers.csv')

# Preview columns
print("Columns:", followers_df.columns.tolist())
followers_df.head()



print(followers_df.columns.tolist())
followers_df.head()



# Forum Activity Trends
#Plot Monthly Follower Trends

import matplotlib.pyplot as plt
import pandas as pd

# Convert CreationDate to datetime
followers_df['CreationDate'] = pd.to_datetime(followers_df['CreationDate'], errors='coerce')

# Drop invalid rows
followers_df = followers_df.dropna(subset=['CreationDate'])

# Extract Year-Month
followers_df['YearMonth'] = followers_df['CreationDate'].dt.to_period('M')

# Count follows per month
follow_trend = followers_df.groupby('YearMonth').size()

# Plotting
plt.figure(figsize=(12, 6))
follow_trend.plot(kind='line', marker='o', color='purple')
plt.title('Monthly Kaggle User Follow Trends')
plt.xlabel('Year-Month')
plt.ylabel('Number of New Followers')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#Forum Activity Trends
# Plan for KernelVotes.csv
 # Load and Inspect

import pandas as pd

# Load KernelVotes.csv
kernel_votes_df = pd.read_csv('/kaggle/input/meta-kaggle/KernelVotes.csv')

# Show basic info
print(kernel_votes_df.columns.tolist())
print(kernel_votes_df.head())


import pandas as pd

# Load KernelVotes.csv
kernel_votes_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVotes.csv", parse_dates=['VoteDate'])

# Filter for recent votes, e.g., after 2023-01-01
recent_kernel_votes = kernel_votes_df[kernel_votes_df['VoteDate'] >= '2023-01-01'].copy()

# Show confirmation
print(f"Filtered KernelVotes rows: {len(recent_kernel_votes)}")
recent_kernel_votes.head()



#Forum Activity Trends
# Convert Dates and Filter Recent Votes

# Convert VoteDate to datetime
kernel_votes_df['VoteDate'] = pd.to_datetime(kernel_votes_df['VoteDate'], errors='coerce')

# Filter last 2 years
recent_kernel_votes = kernel_votes_df[kernel_votes_df['VoteDate'] >= '2023-01-01'].copy()

# Confirm filtering worked
print(f"Filtered KernelVotes rows (since 2023): {len(recent_kernel_votes)}")
recent_kernel_votes[['UserId', 'KernelVersionId', 'VoteDate']].head()


# Forum Activity Trends
# Analyze and Visualize Monthly Voting Trends

import matplotlib.pyplot as plt

# Create a YearMonth column for grouping
recent_kernel_votes['YearMonth'] = recent_kernel_votes['VoteDate'].dt.to_period('M')

# Count number of votes per month
monthly_votes = recent_kernel_votes.groupby('YearMonth').size()

# Plotting
monthly_votes.plot(kind='bar', figsize=(14, 6), color='skyblue')
plt.title("Monthly Kernel Votes (since 2023)")
plt.xlabel("Month")
plt.ylabel("Number of Votes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Forum Activity Trends
# Analyze and Visualize Monthly Voting Trends

import matplotlib.pyplot as plt

# Create a YearMonth column for grouping
recent_kernel_votes['YearMonth'] = recent_kernel_votes['VoteDate'].dt.to_period('M')

# Count number of votes per month
monthly_votes = recent_kernel_votes.groupby('YearMonth').size()

# Plotting
monthly_votes.plot(kind='bar', figsize=(14, 6), color='skyblue')
plt.title("Monthly Kernel Votes (since 2023)")
plt.xlabel("Month")
plt.ylabel("Number of Votes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import os

base_path = "/kaggle/input/meta-kaggle"
available_files = os.listdir(base_path)
print("Files in meta-kaggle dataset folder:")
for f in available_files:
    print(f)



# Forum Activity Trends

#Identify Top Contributors by Total Kernel Votes

import pandas as pd

# Load required datasets
# Load required datasets with corrected filenames
kernel_votes_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVotes.csv")
kernel_versions_df = pd.read_csv(
    "/kaggle/input/meta-kaggle/KernelVersions.csv",
    low_memory=False)
users_df = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")


# Convert VoteDate to datetime
kernel_votes_df['VoteDate'] = pd.to_datetime(kernel_votes_df['VoteDate'], errors='coerce')

# Filter votes since 2023
recent_votes = kernel_votes_df[kernel_votes_df['VoteDate'] >= '2023-01-01'].copy()

# Merge KernelVotes with KernelVersion to get AuthorUserId
merged = pd.merge(
    recent_votes,
    kernel_versions_df[['Id', 'AuthorUserId']],
    left_on='KernelVersionId',
    right_on='Id',
    how='left'
)

# Merge with Users to get UserName
merged_df = pd.merge(
    merged,
    users_df[['Id', 'UserName']],
    left_on='AuthorUserId',
    right_on='Id',
    how='left'
)

# Check if merged_df is ready
print("Sample from merged_df:\n", merged_df[['UserName', 'VoteDate']].dropna().head())



# Forum Activity Trends
#Top contributors by total kernel votes,

import pandas as pd

# Load data
kernel_votes_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVotes.csv")
kernel_versions_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv", low_memory=False)
users_df = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")

# Optional: convert VoteDate to datetime (useful for time-based trends later)
kernel_votes_df["VoteDate"] = pd.to_datetime(kernel_votes_df["VoteDate"], errors="coerce")

# Merge KernelVotes with KernelVersions to get author info
merged_df = kernel_votes_df.merge(kernel_versions_df[["Id", "AuthorUserId"]], left_on="KernelVersionId", right_on="Id", how="left")

# Merge with Users to get UserName
merged_df = merged_df.merge(users_df[["Id", "UserName"]], left_on="AuthorUserId", right_on="Id", how="left")

# Group by UserName and count total votes
top_authors = (
    merged_df.groupby("UserName")
    .size()
    .reset_index(name="TotalVotes")
    .sort_values(by="TotalVotes", ascending=False)
    .head(10)
)

# Display
print("Top 10 Kernel Contributors by Total Votes:")
print(top_authors)



import pandas as pd

# Load the KernelLanguages.csv file
languages_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelLanguages.csv")

# Now check the columns
print(languages_df.columns.tolist())




print(kernel_versions_df.columns.tolist())


import pandas as pd

# Load both CSVs
kernel_versions_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv", low_memory=False)
kernel_languages_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelLanguages.csv")



# Merge KernelVersions with KernelLanguages to get language names
versions_lang_df = pd.merge(
    kernel_versions_df,
    kernel_languages_df,
    left_on='ScriptLanguageId',
    right_on='Id',
    how='left'
)

# Optional: display some sample results
versions_lang_df[['ScriptId', 'ScriptLanguageId', 'Name', 'DisplayName']].head()



#Confirm Merge Works
versions_lang_df = pd.merge(kernel_versions_df, kernel_languages_df,
                            left_on='ScriptLanguageId', right_on='Id',
                            suffixes=('_version', '_language'),
                            how='left')

print("Merged rows:", len(versions_lang_df))
print("Columns:", versions_lang_df.columns.tolist())
versions_lang_df.head()



#Convert CreationDate to datetime and extract Year
# Convert CreationDate to datetime
versions_lang_df['CreationDate'] = pd.to_datetime(versions_lang_df['CreationDate'], errors='coerce')

# Extract year
versions_lang_df['Year'] = versions_lang_df['CreationDate'].dt.year

# Check how many rows have valid Year and Language info
print("Valid rows with year and language:", versions_lang_df[['Year', 'DisplayName']].dropna().shape[0])


# Drop null years or language names
filtered_lang_df = versions_lang_df.dropna(subset=['Year', 'DisplayName'])

# Optional: convert Year to integer for groupby
filtered_lang_df['Year'] = filtered_lang_df['Year'].astype(int)

# Check top few rows
filtered_lang_df[['Year', 'DisplayName']].drop_duplicates().head()



# Step 4: Group by Year and DisplayName (language) to see trends
lang_trends = filtered_lang_df.groupby(['Year', 'DisplayName']).size().reset_index(name='KernelCount')

# Sort to see popular languages per year
lang_trends.sort_values(['Year', 'KernelCount'], ascending=[True, False]).head(10)



#Plot Language Trends Over the Years
import matplotlib.pyplot as plt
import seaborn as sns

# Focus on top 6 languages overall
top_languages = lang_trends.groupby('DisplayName')['KernelCount'].sum().nlargest(6).index
plot_df = lang_trends[lang_trends['DisplayName'].isin(top_languages)]

# Plot
plt.figure(figsize=(14, 7))
sns.lineplot(data=plot_df, x='Year', y='KernelCount', hue='DisplayName', marker='o')

plt.title('Kernel Language Usage Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Kernels')
plt.grid(True)
plt.legend(title='Language')
plt.tight_layout()
plt.show()


import pandas as pd

# Load the datasets
kernel_versions_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv", low_memory=False)
kernel_languages_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelLanguages.csv")

# Merge language info
kernel_lang_df = pd.merge(
    kernel_versions_df,
    kernel_languages_df,
    left_on='ScriptLanguageId',
    right_on='Id',
    how='left'
)

# Now you can access its columns
print(kernel_lang_df.columns.tolist())



import pandas as pd

# Load the KernelVotes dataset
kernel_votes_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVotes.csv")
kernel_versions_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv")

# Now test your checks
print("Unique KernelVersionIds in votes:", kernel_votes_df['KernelVersionId'].nunique())
print("Unique Ids in versions:", kernel_versions_df['Id'].nunique())

# Optional test merge
merged_test = pd.merge(
    kernel_votes_df,
    kernel_versions_df[['Id', 'CreationDate']],
    left_on='KernelVersionId',
    right_on='Id',
    how='inner'
)
print("Merged records:", merged_test.shape[0])




import pandas as pd

# Load the Users dataset
users_df = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")

# Re-run your merge step
votes_versions_users_df = pd.merge(
    votes_versions_df,
    users_df[['Id', 'UserName']],
    left_on='AuthorUserId',
    right_on='Id',
    how='left'
)

print("Final shape after Users merge:", votes_versions_users_df.shape)



# Force column types to match
kernel_votes_df['KernelVersionId'] = kernel_votes_df['KernelVersionId'].astype('Int64')
kernel_versions_df['Id'] = kernel_versions_df['Id'].astype('Int64')

# Merge with CreationDate
votes_versions_df = pd.merge(kernel_votes_df, kernel_versions_df[['Id', 'CreationDate', 'AuthorUserId']],
                             left_on='KernelVersionId', right_on='Id', how='left')

# Check if merge was successful
print("Rows after merge:", votes_versions_df.shape)

# Rename and convert datetime
votes_versions_df.rename(columns={'CreationDate': 'KernelCreationDate'}, inplace=True)
votes_versions_df['KernelCreationDate'] = pd.to_datetime(votes_versions_df['KernelCreationDate'], errors='coerce')

# Filter nulls if needed
votes_versions_df = votes_versions_df.dropna(subset=['KernelCreationDate'])
print("Rows after dropping nulls:", votes_versions_df.shape)

# Optional: merge with Users
votes_versions_users_df = pd.merge(votes_versions_df, users_df[['Id', 'UserName']],
                                   left_on='AuthorUserId', right_on='Id', how='left')
print("Final shape after Users merge:", votes_versions_users_df.shape)

# Preview
votes_versions_users_df[['KernelVersionId', 'KernelCreationDate', 'UserName']].head()



#Option 1: Trend Analysis – Number of Kernels Created Over Time.

import pandas as pd

# Load required data (adjust path if needed)
kernel_versions_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv", low_memory=False)
kernel_languages_df = pd.read_csv("/kaggle/input/meta-kaggle/KernelLanguages.csv")

# Merge KernelVersions with KernelLanguages to get language info
kernel_lang_df = pd.merge(
    kernel_versions_df,
    kernel_languages_df,
    left_on='ScriptLanguageId',
    right_on='Id',
    how='left'
)

# Rename to avoid confusion
kernel_lang_df.rename(columns={'Id_x': 'VersionId', 'Id_y': 'LanguageId'}, inplace=True)

# Ensure CreationDate is datetime
kernel_lang_df['CreationDate'] = pd.to_datetime(kernel_lang_df['CreationDate'], errors='coerce')
kernel_lang_df = kernel_lang_df.dropna(subset=['CreationDate'])

# Now proceed with trend analysis
kernel_lang_df['YearMonth'] = kernel_lang_df['CreationDate'].dt.to_period('M').astype(str)

monthly_kernel_counts = kernel_lang_df.groupby('YearMonth').size().reset_index(name='KernelCount')

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(monthly_kernel_counts['YearMonth'], monthly_kernel_counts['KernelCount'], marker='o')
plt.title('Monthly Kernel Creation Trend')
plt.xlabel('Year-Month')
plt.ylabel('Number of Kernels')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


#Identify Top Contributors
#Top Contributors by Total Kernel Votes

import pandas as pd

# Load data
kernel_votes = pd.read_csv('/kaggle/input/meta-kaggle/KernelVotes.csv')
kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')

# Merge votes with kernel versions to get AuthorUserId
votes_merged = kernel_votes.merge(kernel_versions[['Id', 'AuthorUserId']], left_on='KernelVersionId', right_on='Id', how='left')

# Merge with Users to get UserName
votes_user_df = votes_merged.merge(users[['Id', 'UserName']], left_on='AuthorUserId', right_on='Id', how='left')

# Group by user
top_kernel_authors = votes_user_df.groupby('UserName').size().reset_index(name='TotalKernelVotes')
top_kernel_authors = top_kernel_authors.sort_values(by='TotalKernelVotes', ascending=False).head(10)

# Display
print(top_kernel_authors)

# Optional: Plot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(data=top_kernel_authors, x='TotalKernelVotes', y='UserName', palette='viridis')
plt.title('Top 10 Contributors by Kernel Votes')
plt.xlabel('Total Votes')
plt.ylabel('User Name')
plt.tight_layout()
plt.show()


#Identify Top Contributors
#Forum-Based Contributors

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
forum_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')
forum_votes = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv')
users_df = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')

# Count messages per user
msg_counts = forum_df.groupby('PostUserId').size().reset_index(name='MessageCount')

# Count medals (optional, good for measuring recognition)
medal_counts = forum_df[forum_df['Medal'].notna()].groupby('PostUserId').size().reset_index(name='MedalCount')

# Count votes (ForumMessageVotes are by message, so we map message to user)
msg_user_map = forum_df[['Id', 'PostUserId']]
votes_with_users = forum_votes.merge(msg_user_map, left_on='ForumMessageId', right_on='Id', how='left')
vote_counts = votes_with_users.groupby('PostUserId').size().reset_index(name='TotalVotes')

# Merge all metrics together
forum_stats = msg_counts.merge(medal_counts, on='PostUserId', how='left') \
                        .merge(vote_counts, on='PostUserId', how='left') \
                        .fillna(0)

# Add usernames
forum_stats = forum_stats.merge(users_df[['Id', 'UserName']], left_on='PostUserId', right_on='Id', how='left')

# Sort by message count and show top contributors
top_forum_users = forum_stats.sort_values(by='MessageCount', ascending=False).head(10)

# Display
print(top_forum_users[['UserName', 'MessageCount', 'MedalCount', 'TotalVotes']])

# Plot
plt.figure(figsize=(10,6))
sns.barplot(data=top_forum_users, x='MessageCount', y='UserName', palette='crest')
plt.title('Top 10 Forum Contributors (by Messages Posted)')
plt.xlabel('Messages Posted')
plt.ylabel('User Name')
plt.tight_layout()
plt.show()



print(followers_df.columns.tolist())
followers_df.head()


# Identify Top Contributors
#Identify Top Users by Follower Count

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
followers_df = pd.read_csv('/kaggle/input/meta-kaggle/UserFollowers.csv')
users_df = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')

# Step 1: Count followers per user
follower_counts = followers_df.groupby('FollowingUserId').size().reset_index(name='FollowerCount')

# Step 2: Merge with user names from Users.csv
top_followed = follower_counts.merge(users_df[['Id', 'UserName']], left_on='FollowingUserId', right_on='Id', how='left')

# Step 3: Sort and show top 10
top_followed = top_followed.sort_values(by='FollowerCount', ascending=False).head(10)

# Step 4: Display
print(top_followed[['UserName', 'FollowerCount']])

# Plot
plt.figure(figsize=(10,6))
sns.barplot(data=top_followed, x='FollowerCount', y='UserName', palette='mako')
plt.title('Top 10 Most Followed Users')
plt.xlabel('Followers')
plt.ylabel('User Name')
plt.tight_layout()
plt.show()



#Identify Top Contributors
# Forum Participation Metrics
#Trend of Forum Messages Over Time

import pandas as pd
import matplotlib.pyplot as plt

# Load ForumMessages dataset
forum_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')

# Convert PostDate to datetime
forum_df['PostDate'] = pd.to_datetime(forum_df['PostDate'], errors='coerce')

# Drop rows with invalid dates
forum_df = forum_df.dropna(subset=['PostDate'])

# Group by Year-Month and count posts
forum_df['YearMonth'] = forum_df['PostDate'].dt.to_period('M').astype(str)
monthly_posts = forum_df.groupby('YearMonth').size()

# Plot
plt.figure(figsize=(14, 6))
monthly_posts.plot(kind='line', marker='o')
plt.title('Forum Message Volume Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Load ForumMessages.csv again
import pandas as pd
import matplotlib.pyplot as plt

forum_df = pd.read_csv("/kaggle/input/meta-kaggle/ForumMessages.csv")

# Confirm structure
print("Loaded ForumMessages.csv")
print(forum_df.columns.tolist())
forum_df.head()



#Identify Top Contributors
#Forum Medal Distribution and Trends

# Filter out rows with medals only
medal_df = forum_df[forum_df['Medal'].notna()].copy()

# Convert MedalAwardDate to datetime
medal_df['MedalAwardDate'] = pd.to_datetime(medal_df['MedalAwardDate'], errors='coerce')
medal_df.dropna(subset=['MedalAwardDate'], inplace=True)

# Create Year-Month for trend analysis
medal_df['YearMonth'] = medal_df['MedalAwardDate'].dt.to_period('M').astype(str)

# Group by medal type and YearMonth
medal_trends = medal_df.groupby(['YearMonth', 'Medal']).size().unstack(fill_value=0)

# Plot trends
medal_trends.plot(kind='line', figsize=(12, 6), marker='o')
plt.title("Forum Medal Award Trends Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Number of Medals")
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.legend(title="Medal Type")
plt.show()



#Top Forum Contributors

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load ForumMessages and Users data
forum_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')
users_df = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')

# Count number of messages per user
top_forum_users = forum_df.groupby('PostUserId').size().reset_index(name='MessageCount')

# Merge with Users to get username
top_forum_users = top_forum_users.merge(users_df[['Id', 'UserName']], left_on='PostUserId', right_on='Id', how='left')

# Sort and get top 10
top_forum_users = top_forum_users.sort_values(by='MessageCount', ascending=False).head(10)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(data=top_forum_users, x='MessageCount', y='UserName', palette='Blues_d')
plt.title('Top 10 Users by Forum Messages Posted')
plt.xlabel('Message Count')
plt.ylabel('User Name')
plt.tight_layout()
plt.show()



# Forum Awards by User

import os

# List files under meta-kaggle directory
print(os.listdir("/kaggle/input/meta-kaggle"))


# Forum Awards by User
import pandas as pd

# Step 1: Load ForumMessages and ForumMessageVotes
forum_msgs_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')
forum_votes_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv')

# Step 2: Check column names
print("ForumMessages Columns:", forum_msgs_df.columns.tolist())
print("ForumMessageVotes Columns:", forum_votes_df.columns.tolist())



import pandas as pd

# Load the required datasets
forum_votes_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv')
forum_msgs_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')




merged_forum_votes = pd.merge(
    forum_votes_df,
    forum_msgs_df,
    left_on='ForumMessageId',
    right_on='Id',
    how='inner'
)




import matplotlib.pyplot as plt

# Step 3: Merge ForumMessages and ForumMessageVotes on ForumMessageId and Id
merged_forum_votes = pd.merge(
    forum_votes_df,
    forum_msgs_df,
    left_on='ForumMessageId',
    right_on='Id',
    how='left'
)

print(f"Merged rows: {len(merged_forum_votes)}")

# Step 4: Convert VoteDate to datetime
merged_forum_votes['VoteDate'] = pd.to_datetime(merged_forum_votes['VoteDate'], errors='coerce')

# Step 5: Count votes per user (to whom the vote was given)
top_voted_users = merged_forum_votes.groupby('ToUserId').size().reset_index(name='TotalVotes')
top_voted_users = top_voted_users.sort_values(by='TotalVotes', ascending=False).head(10)

# Step 6: Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_voted_users['ToUserId'].astype(str), top_voted_users['TotalVotes'], color='teal')
plt.xlabel("Total Votes Received")
plt.ylabel("User ID")
plt.title("Top 10 Forum Contributors by Message Votes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


