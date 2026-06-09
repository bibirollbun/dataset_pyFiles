# Import the Dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to dataset files:", path)


import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to dataset files:", path)


#Import Libraries
#Libraries
# Data Handling
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import re

# Date Handling
import datetime as dt
# Display settings (optional but helpful)
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep output clean

# Plot settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


# Import the CSV files required for competition analysis
# Import the CSV Files
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
competitionsTags = pd.read_csv('/kaggle/input/meta-kaggle/CompetitionTags.csv')
forummessages = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')
tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')
kernel_comp_sources = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv')


# 1. Growth of Competitions Over Time
# Convert EnabledDate to datetime and extract year
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Filter valid years
competitions_by_year = competitions[competitions['Year'].notnull()]

# Group by year and count
yearly_counts = competitions_by_year['Year'].value_counts().sort_index().reset_index()
yearly_counts.columns = ['Year', 'CompetitionCount']

# Create interactive bar chart
fig = px.bar(
    yearly_counts,
    x='Year',
    y='CompetitionCount',
    title=' Growth of Kaggle Competitions Over Time',
    labels={'CompetitionCount': 'Number of Competitions'},
    color='CompetitionCount',
    color_continuous_scale='viridis'
)

fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Competition Count',
    xaxis=dict(type='category'),
    coloraxis_showscale=False
)

fig.show()


# 2. Top Competition Topics (Tags)
# Merge comp_tags with tags to get the tag names
comp_tags_named = pd.merge(competitionsTags, tags, left_on='TagId', right_on='Id', suffixes=('', '_Tag'))

# Optional: merge with competition titles
comp_tags_named = pd.merge(
    comp_tags_named,
    competitions[['Id', 'Title']],
    left_on='CompetitionId',
    right_on='Id',
    suffixes=('', '_Competition')
)

# Count the top 20 most frequent tags
top_tags = comp_tags_named['Name'].value_counts().head(20).reset_index()
top_tags.columns = ['Tag', 'Count']

# Create interactive horizontal bar chart
fig = px.bar(
    top_tags,
    x='Count',
    y='Tag',
    orientation='h',
    title=' Top 20 Competition Tags on Kaggle',
    labels={'Count': 'Number of Competitions', 'Tag': 'Tag'},
    color='Count',
    color_continuous_scale='viridis'
)

fig.update_layout(
    yaxis=dict(autorange='reversed'),  # Highest tag on top
    xaxis_title='Number of Competitions',
    yaxis_title='Tag'
)

fig.show()


# 3. Competitions Aligned with Global Events
# Ensure text fields are strings
for col in ['Title', 'Subtitle', 'Overview']:
    competitions[col] = competitions[col].fillna('').astype(str)

# Define keywords for global topics
keywords = {
    'COVID-19': ['covid', 'coronavirus', 'pandemic'],
    'Climate': ['climate', 'carbon', 'sustainability', 'weather'],
    'Disasters': ['earthquake', 'disaster', 'flood', 'emergency'],
    'Misinformation': ['fake news', 'misinformation', 'propaganda'],
    'Finance': ['stock', 'forex', 'finance', 'bank', 'market'],
    'Medical': ['cancer', 'x-ray', 'lung', 'radiology', 'diabetes', 'disease', 'hospital']
}

# Create a new column and match based on keywords
competitions['GlobalTopic'] = None
for topic, kws in keywords.items():
    pattern = '|'.join(kws)
    mask = (
        competitions['Title'].str.lower().str.contains(pattern) |
        competitions['Subtitle'].str.lower().str.contains(pattern) |
        competitions['Overview'].str.lower().str.contains(pattern)
    )
    competitions.loc[mask, 'GlobalTopic'] = topic

# Count competitions per global topic
topic_counts = competitions['GlobalTopic'].value_counts().sort_values().reset_index()
topic_counts.columns = ['Topic', 'Count']

# Interactive horizontal bar chart
fig = px.bar(
    topic_counts,
    x='Count',
    y='Topic',
    orientation='h',
    title='Kaggle Competitions Aligned with Global Issues',
    labels={'Count': 'Number of Competitions', 'Topic': 'Global Topic'},
    color='Count',
    color_continuous_scale='viridis'
)

fig.update_layout(
    yaxis=dict(autorange='reversed'),
    xaxis_title='Number of Competitions',
    yaxis_title='Global Topic'
)

fig.show()


#4. Trends in Competition Domains Over Time
# Convert EnabledDate to datetime and extract year
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Merge comp_tags with tag names
tagged_comp = pd.merge(competitionsTags, tags, left_on='TagId', right_on='Id', suffixes=('', '_Tag'))

# Merge with competitions to get year info
tagged_comp = pd.merge(tagged_comp, competitions[['Id', 'Year']], left_on='CompetitionId', right_on='Id', suffixes=('', '_Comp'))

# Get top 10 most frequent tags
top_tags = tagged_comp['Name'].value_counts().head(10).index.tolist()

# Filter to only those tags
filtered = tagged_comp[tagged_comp['Name'].isin(top_tags)]
# Count number of competitions per year per tag
tag_year_counts = filtered.groupby(['Year', 'Name']).size().reset_index(name='Count')

# Pivot to wide format
pivot_df = tag_year_counts.pivot(index='Year', columns='Name', values='Count').fillna(0)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
pivot_df.plot(kind='line', marker='o', figsize=(14, 7))
plt.title('Trends in Competition Domains Over Time (Top 10 Tags)')
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.legend(title='Tag')
plt.grid(True)
plt.tight_layout()
plt.show()


#5. Evaluation Metric Trends
# Convert date column and extract year
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Clean and standardize evaluation metric names
competitions['EvaluationAlgorithmName'] = competitions['EvaluationAlgorithmName'].fillna('Unknown').str.strip()

# Filter competitions with valid year and metric
valid_metrics = competitions[(competitions['Year'] >= 2013) & (competitions['EvaluationAlgorithmName'] != 'Unknown')]

# Get top 10 most used evaluation metrics
top_metrics = valid_metrics['EvaluationAlgorithmName'].value_counts().head(10).index.tolist()

# Filter to top 10 only
filtered = valid_metrics[valid_metrics['EvaluationAlgorithmName'].isin(top_metrics)]
# Group by year and metric
metric_trend = filtered.groupby(['Year', 'EvaluationAlgorithmName']).size().reset_index(name='Count')
# Pivot to wide format for plotting
pivot_metric = metric_trend.pivot(index='Year', columns='EvaluationAlgorithmName', values='Count').fillna(0)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
pivot_metric.plot(marker='o')
plt.title('Evaluation Metric Trends in Kaggle Competitions (Top 10)')
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.grid(True)
plt.legend(title='Evaluation Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 6. Research vs Practical Challenges
# Ensure datetime and extract year
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Filter valid years
competitions_filtered = competitions[competitions['Year'] >= 2013]

# Group by year and segment type
segment_trend = competitions_filtered.groupby(['Year', 'HostSegmentTitle']).size().reset_index(name='Count')

# Interactive stacked bar chart
fig = px.bar(
    segment_trend,
    x='Year',
    y='Count',
    color='HostSegmentTitle',
    title='Types of Kaggle Competitions Over Time (Research vs Practical)',
    labels={'HostSegmentTitle': 'Competition Type', 'Count': 'Number of Competitions'},
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Number of Competitions',
    legend_title='Competition Type',
    template='plotly_white'
)

fig.show()



# Ensure datetime and extract year
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

# Filter valid years and exclude 'Community'
competitions_filtered = competitions[
    (competitions['Year'] >= 2013) & 
    (competitions['HostSegmentTitle'] != 'Community')
]

# Group by year and competition type
segment_trend = competitions_filtered.groupby(['Year', 'HostSegmentTitle']).size().reset_index(name='Count')

# Plot side-by-side bars
fig = px.bar(
    segment_trend,
    x='Year',
    y='Count',
    color='HostSegmentTitle',
    title='Kaggle Competitions Over Time (Excluding Community)',
    labels={'HostSegmentTitle': 'Competition Type', 'Count': 'Number of Competitions'},
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Number of Competitions',
    legend_title='Competition Type',
    template='plotly_white'
)

fig.show()


#7. Most Used ML Models in Competition Kernels
# Map KernelVersionId -> KernelId (ScriptId)

version_map = kernel_versions[['Id', 'ScriptId']]
kernel_comp_sources = kernel_comp_sources.merge(
    version_map,
    left_on='KernelVersionId',
    right_on='Id',
    how='left'
)
competition_kernel_ids = kernel_comp_sources['ScriptId'].dropna().unique()

#  Filter tags for competition kernels only

comp_kernel_tags = kernel_tags[kernel_tags['KernelId'].isin(competition_kernel_ids)]
comp_kernel_tags = comp_kernel_tags.merge(tags, left_on='TagId', right_on='Id', how='left')
comp_kernel_tags['TagName'] = comp_kernel_tags['Name'].str.lower()


#  Filter for ML model tags

ml_keywords = [
    "xgboost", "lightgbm", "catboost", "randomforest", "logisticregression",
    "svm", "knn", "naivebayes", "cnn", "rnn", "lstm", "transformer", "bert", "gpt"
]

ml_tags = comp_kernel_tags[comp_kernel_tags['TagName'].isin(ml_keywords)]


# Count frequency of model tags

tag_counts = ml_tags['TagName'].value_counts().reset_index()
tag_counts.columns = ['Model', 'Count']


#  Plot results

plt.figure(figsize=(12,6))
sns.barplot(data=tag_counts, x='Count', y='Model', palette='mako')
plt.title('Top ML Models in Kaggle Competitions (via Kernel Tags)')
plt.xlabel('Number of Competition Kernels Tagged')
plt.ylabel('ML Model')
plt.tight_layout()
plt.show()


# 8. ML Model Trends in Kaggle Kernels Over Time

# Add Year from Kernels to KernelTags
kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'], errors='coerce')
kernels['Year'] = kernels['CreationDate'].dt.year
kernel_versions = kernel_versions.rename(columns={'Id': 'KernelVersionId', 'ScriptId': 'KernelId'})

# Merge to get year per KernelId
kernel_years = kernel_versions.merge(kernels[['Id', 'Year']], left_on='KernelId', right_on='Id', how='left')
kernel_years = kernel_years[['KernelVersionId', 'KernelId', 'Year']]

# Merge with KernelTags
kernel_tags = kernel_tags.merge(kernel_years[['KernelId', 'Year']], on='KernelId', how='left')

# Merge with tag names
kernel_tags = kernel_tags.merge(tags[['Id', 'Name']], left_on='TagId', right_on='Id', how='left')
kernel_tags['TagName'] = kernel_tags['Name'].str.lower()

# Define ML-related tags
ml_keywords = [
    "xgboost", "lightgbm", "catboost", "randomforest", "logisticregression",
    "svm", "knn", "naivebayes", "cnn", "rnn", "lstm", "transformer", "bert", "gpt"
]

ml_tags = kernel_tags[kernel_tags['TagName'].isin(ml_keywords)].dropna(subset=['Year'])

# Count yearly tag usage
yearly_trend = ml_tags.groupby(['Year', 'TagName']).size().unstack(fill_value=0)

# Filter top 10 overall models
top_tags = yearly_trend.sum().sort_values(ascending=False).head(10).index
yearly_trend[top_tags].plot(marker='o', figsize=(14, 6))

# Plot
plt.title('ML Model Trends in Kaggle Kernels Over Time (Tag-Based)')
plt.xlabel('Year')
plt.ylabel('Number of Kernels Tagged')
plt.grid(True)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Import CSV files for Community Analysis

users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
submissions = pd.read_csv('/kaggle/input/meta-kaggle/Submissions.csv')
teams = pd.read_csv('/kaggle/input/meta-kaggle/Teams.csv')
team_memberships = pd.read_csv('/kaggle/input/meta-kaggle/TeamMemberships.csv')
messages = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')
topics = pd.read_csv('/kaggle/input/meta-kaggle/ForumTopics.csv')


#1. User Growth over time
# Convert RegisterDate to datetime
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')

# Extract year
users['JoinYear'] = users['RegisterDate'].dt.year

# Count number of new users per year
new_users = users.groupby('JoinYear')['Id'].nunique().reset_index(name='NewUsers')

# Filter valid years
new_users = new_users[new_users['JoinYear'].notna() & (new_users['JoinYear'] >= 2012)]

# Plot interactive bar chart
fig = px.bar(
    new_users,
    x='JoinYear',
    y='NewUsers',
    title='Kaggle Users per Year (Interactive)',
    labels={'JoinYear': 'Year', 'NewUsers': 'Number of New Users'},
    color='NewUsers',
    color_continuous_scale='viridis'
)

fig.update_layout(xaxis=dict(dtick=1))
fig.show()


#2. Active vs. Inactive Users
# 1. Prepare user registration data
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')
users['JoinYear'] = users['RegisterDate'].dt.year
users = users[['Id', 'JoinYear']].rename(columns={'Id': 'UserId'})

# 2. Submission activity
submissions['SubmissionDate'] = pd.to_datetime(submissions['SubmissionDate'], errors='coerce')
submissions['ActiveYear'] = submissions['SubmissionDate'].dt.year
submission_users = submissions[['SubmittedUserId', 'ActiveYear']].dropna()
submission_users = submission_users.rename(columns={'SubmittedUserId': 'UserId'})

# 3. Team activity
team_memberships['RequestDate'] = pd.to_datetime(team_memberships['RequestDate'], errors='coerce')
team_memberships['ActiveYear'] = team_memberships['RequestDate'].dt.year
team_users = team_memberships[['UserId', 'ActiveYear']].dropna()

# 4. Combine all active users
activity = pd.concat([submission_users, team_users], axis=0)
activity = activity.drop_duplicates()

# 5. Merge with all users to track active/inactive
user_activity = users.merge(activity, on='UserId', how='left')  # may result in NaN if inactive
user_activity['ActiveYear'] = user_activity['ActiveYear'].fillna(-1).astype(int)

# Define engagement status per year: active only if active in same year as join
user_activity['Status'] = user_activity['JoinYear'] == user_activity['ActiveYear']
user_activity['Status'] = user_activity['Status'].map({True: 'Active', False: 'Inactive'})

# Group and count
trend = user_activity.groupby(['JoinYear', 'Status'])['UserId'].nunique().reset_index(name='UserCount')
trend = trend[trend['JoinYear'].notna() & (trend['JoinYear'] >= 2012)]
# Plot
plt.figure(figsize=(12,6))
sns.lineplot(data=trend, x='JoinYear', y='UserCount', hue='Status', marker='o')
plt.title('Active vs. Inactive Users on Kaggle (By Join Year)')
plt.xlabel('Join Year')
plt.ylabel('Number of Users')
plt.grid(True)
plt.tight_layout()
plt.show()


#3. Team Collaboration Patterns
# Count number of users per team
team_sizes = team_memberships.groupby('TeamId')['UserId'].nunique().reset_index(name='TeamSize')

# Merge with competition info (optional if you want time analysis)
team_data = teams[['Id', 'CompetitionId']].rename(columns={'Id': 'TeamId'})
team_sizes = team_sizes.merge(team_data, on='TeamId', how='left')

# Label solo vs. team
team_sizes['TeamType'] = team_sizes['TeamSize'].apply(lambda x: 'Solo' if x == 1 else 'Team')

# Count Solo vs Team
team_type_counts = team_sizes['TeamType'].value_counts().reset_index()
team_type_counts.columns = ['TeamType', 'Count']

# Plot 1: Bar chart Solo vs Team 
plt.figure(figsize=(6,4))
sns.barplot(data=team_type_counts, x='TeamType', y='Count', palette='Set2')
plt.title('Solo vs. Team Participation')
plt.xlabel('Team Type')
plt.ylabel('Number of Teams')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
#  Get Grandmasters
grandmasters = users[users['PerformanceTier'] == 4][['Id']].rename(columns={'Id': 'UserId'})

#  Merge with TeamMemberships to get TeamIds for Grandmasters
gm_team_memberships = team_memberships.merge(grandmasters, on='UserId', how='inner')

#  Count members per team
team_sizes = team_memberships.groupby('TeamId')['UserId'].nunique().reset_index(name='TeamSize')

#  Merge GM teams with their team size
gm_team_info = gm_team_memberships.merge(team_sizes, on='TeamId', how='left')

#  Label team type
gm_team_info['TeamType'] = gm_team_info['TeamSize'].apply(lambda x: 'Solo' if x == 1 else 'Team')

#  Count GM participations by team type
gm_team_type_counts = gm_team_info['TeamType'].value_counts().reset_index()
gm_team_type_counts.columns = ['TeamType', 'Count']
# Plot 
plt.figure(figsize=(6,4))
sns.barplot(data=gm_team_type_counts, x='TeamType', y='Count', palette='pastel')
plt.title('Grandmaster Participation: Solo vs. Team')
plt.xlabel('Team Type')
plt.ylabel('Number of Participations')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#4. Distribution of Kaggle Users Globally
# Count users by country
country_counts = users['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'UserCount']

# Drop missing/unknown countries
country_counts = country_counts[country_counts['Country'].notna()]

# Plot using Plotly
fig = px.choropleth(
    country_counts,
    locations='Country',
    locationmode='country names',
    color='UserCount',
    color_continuous_scale='Viridis',
    title='Kaggle Users by Country (Global Distribution)',
)
fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.show()


#5. Kaggle Forum Messages per Year
# Convert PostDate to datetime
messages['PostDate'] = pd.to_datetime(messages['PostDate'], errors='coerce')

# Extract year
messages['Year'] = messages['PostDate'].dt.year

# Count messages per year
yearly_counts = messages['Year'].value_counts().sort_index().reset_index()
yearly_counts.columns = ['Year', 'MessageCount']

# Filter for reasonable years
yearly_counts = yearly_counts[yearly_counts['Year'] >= 2012]

# Plot
plt.figure(figsize=(12,6))
sns.lineplot(data=yearly_counts, x='Year', y='MessageCount', marker='o')
plt.title('Kaggle Forum Messages per Year')
plt.xlabel('Year')
plt.ylabel('Number of Messages')
plt.grid(True)
plt.tight_layout()
plt.show()


#6. Top 10 Most Discussed Topics
# Count replies per topic
topic_message_counts = messages['ForumTopicId'].value_counts().reset_index()
topic_message_counts.columns = ['ForumTopicId', 'ReplyCount']

# Merge with topic titles
top_topics = topic_message_counts.merge(topics[['Id', 'Title']], left_on='ForumTopicId', right_on='Id', how='left')

# Show top 10 most discussed topics
top_topics[['Title', 'ReplyCount']].head(10)


#7. Top Machine Learning-Related Kaggle Forum Topics
# Confirm the correct topic ID column in messages
import re
topic_id_col = 'ForumTopicId' if 'ForumTopicId' in messages.columns else 'TopicId'

#  Count replies per topic
reply_counts = messages[topic_id_col].value_counts().reset_index()
reply_counts.columns = ['TopicId', 'ReplyCount']

# Merge with forum topic titles
topics_with_replies = reply_counts.merge(
    topics[['Id', 'Title']], left_on='TopicId', right_on='Id', how='left'
)
#  Filter ML-related topics by keywords
ml_keywords = [
    "machine learning", "deep learning", "neural network", "xgboost", "lightgbm",
    "random forest", "svm", "transformer", "bert", "cnn", "rnn", "gpt", "llm"
]
pattern = '|'.join([re.escape(word) for word in ml_keywords])
ml_topics = topics_with_replies[
    topics_with_replies['Title'].str.contains(pattern, case=False, na=False)
]

# Step 5: Get top 10 ML topics by reply count
top_ml_topics = ml_topics.sort_values(by='ReplyCount', ascending=False).head(10)
top_ml_topics = top_ml_topics[['Title', 'ReplyCount']]

# Step 6: Plot
plt.figure(figsize=(12,6))
sns.barplot(data=top_ml_topics, y='Title', x='ReplyCount', palette='mako')
plt.title('Top Machine Learning-Related Kaggle Forum Topics (By Replies)')
plt.xlabel('Number of Replies')
plt.ylabel('Forum Topic Title')
plt.tight_layout()
plt.show()


#8. Top 15 Notebook Tags by Total Votes
# Merge tag names
kernel_tags_named = kernel_tags.merge(tags, left_on='TagId', right_on='Id', how='left')
kernel_tags_named = kernel_tags_named[['KernelId', 'Name']].rename(columns={'Name': 'TagName'})

# Merge with kernel engagement info
kernel_info = kernels[['Id', 'TotalVotes', 'TotalViews', 'TotalComments']]
kernel_tagged = kernel_tags_named.merge(kernel_info, left_on='KernelId', right_on='Id', how='left')

# Remove rows with missing values
kernel_tagged.dropna(subset=['TotalVotes'], inplace=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')

# Merge tag names
kernel_tags_named = kernel_tags.merge(tags, left_on='TagId', right_on='Id', how='left')
kernel_tags_named = kernel_tags_named[['KernelId', 'Name']].rename(columns={'Name': 'TagName'})

# Merge with kernel engagement info
kernel_info = kernels[['Id', 'TotalVotes', 'TotalViews', 'TotalComments']]
kernel_tagged = kernel_tags_named.merge(kernel_info, left_on='KernelId', right_on='Id', how='left')

# Remove rows with missing values
kernel_tagged.dropna(subset=['TotalVotes'], inplace=True)

# Group by tag and aggregate metrics
tag_stats = kernel_tagged.groupby('TagName').agg(
    TotalVotes=('TotalVotes', 'sum'),
    TotalViews=('TotalViews', 'sum'),
    TotalComments=('TotalComments', 'sum'),
    KernelCount=('KernelId', 'nunique')
).reset_index()

# Optional: average metrics per kernel
tag_stats['AvgVotes'] = tag_stats['TotalVotes'] / tag_stats['KernelCount']
tag_stats['AvgViews'] = tag_stats['TotalViews'] / tag_stats['KernelCount']
tag_stats['AvgComments'] = tag_stats['TotalComments'] / tag_stats['KernelCount']

# Top tags by total votes
top_tags = tag_stats.sort_values(by='TotalVotes', ascending=False).head(15)

# Plot
plt.figure(figsize=(12,6))
sns.barplot(data=top_tags, y='TagName', x='TotalVotes', palette='viridis')
plt.title('Top 15 Notebook Tags by Total Votes')
plt.xlabel('Total Votes')
plt.ylabel('Notebook Topic (Tag)')
plt.tight_layout()
plt.show()

