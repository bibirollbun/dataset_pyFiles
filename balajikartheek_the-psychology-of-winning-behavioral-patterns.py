# Data manipulation
import pandas as pd  
import numpy as np   

# Database/SQL
import sqlite3       
from pandasql import sqldf 

import matplotlib.pyplot as plt
import seaborn as sns

# Interactive plots 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Time series/trend analysis
from statsmodels.tsa.seasonal import seasonal_decompose

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# External data sources
import kagglehub

# System Utilities
import os               
import re                
from tqdm import tqdm   
import warnings
warnings.filterwarnings('ignore')


# Download datasets
meta_kaggle_data_path = kagglehub.dataset_download("kaggle/meta-kaggle")
meta_kaggle_code_path = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print(f"Meta Kaggle Data path: {meta_kaggle_data_path}")
print(f"Meta Kaggle Code path: {meta_kaggle_code_path}")


# Load submissions
submissions = pd.read_csv("/kaggle/input/meta-kaggle/Submissions.csv")

# Inspect shape, columns, and sample data
print("Submissions Shape:", submissions.shape)
print("Submissions Columns:\n", submissions.columns)

# Check for nulls
print("\nNull Counts:\n", submissions.isnull().sum())
submissions.head()


# Load Competitions
competitions = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")

print("Competitions Shape:", competitions.shape)
print("Competitions Columns:\n", competitions.columns)
print("\nNull Counts:\n", competitions.isnull().sum())
competitions.head()


# Load Users
users = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")

print("Users Shape:", users.shape)
print("Users Columns:\n", users.columns)
print("\nNull Counts:\n", users.isnull().sum())
users.head()


submissions.head(3)


# Convert dates to datetime
submissions['SubmissionDate'] = pd.to_datetime(submissions['SubmissionDate'])
submissions['ScoreDate'] = pd.to_datetime(submissions['ScoreDate'])

# Drop irrelevant columns and nulls in critical fields
submissions_clean = submissions.drop(columns=[
    'PublicScoreLeaderboardDisplay', 
    'PrivateScoreLeaderboardDisplay'
]).dropna(subset=['SubmittedUserId', 'PublicScoreFullPrecision'])

# Filter valid submissions (non-null scores, non-test submissions)
submissions_clean = submissions_clean[
    (submissions_clean['PublicScoreFullPrecision'].notna()) & 
    (submissions_clean['IsSelected'] == True)
]


submissions_clean.head(3)


submissions_clean.isnull().sum()


competitions.head(2)


# Keep key columns and convert dates
competitions_clean = competitions[[
    'Id', 'Title', 'EnabledDate', 'DeadlineDate', 
    'TotalTeams', 'TotalSubmissions', 'RewardType'
]].copy()
competitions_clean['EnabledDate'] = pd.to_datetime(competitions_clean['EnabledDate'])
competitions_clean['DeadlineDate'] = pd.to_datetime(competitions_clean['DeadlineDate'])

# Drop competitions with missing critical data
competitions_clean = competitions_clean.dropna(subset=['EnabledDate', 'DeadlineDate'])
competitions_clean.head(2)


users.head(3)


# Clean user data
users_clean = users[['Id', 'UserName', 'RegisterDate', 'PerformanceTier']].copy()
users_clean['RegisterDate'] = pd.to_datetime(users_clean['RegisterDate'])

# Drop null usernames
users_clean = users_clean.dropna(subset=['UserName'])
users_clean.head(3)


#                         Competition Phase Timing
# Merge submissions with competitions
merged = pd.merge(
    submissions_clean, 
    competitions_clean, 
    left_on='TeamId', 
    right_on='Id', 
    how='left',
    suffixes=('_sub', '_comp')
)

# Calculate submission timing relative to deadline
merged['DaysToDeadline'] = (merged['DeadlineDate'] - merged['SubmissionDate']).dt.days
# if Days to Deadline is more, then it is a early submission
merged['SubmissionPhase'] = pd.cut(
    merged['DaysToDeadline'],
    bins=[-float('inf'), 0, 7, 30, float('inf')],
    labels=['Late', 'Final Week', 'Early', 'Very Early']
)
merged.head(3)


merged.isnull().sum()


merged.info()


# Calculate user submission stats
user_activity = merged.groupby('SubmittedUserId').agg(
    total_submissions=('Id_sub', 'count'),
    avg_score=('PublicScoreFullPrecision', 'mean'),
    first_submission_date=('SubmissionDate', 'min'),
    last_submission_date=('SubmissionDate', 'max')
).reset_index()

# Merge with user profile data
user_activity = pd.merge(
    user_activity, 
    users_clean, 
    left_on='SubmittedUserId', 
    right_on='Id', 
    how='left'
)

# Calculate tenure and submission frequency
user_activity['tenure_days'] = (user_activity['last_submission_date'] - user_activity['RegisterDate']).dt.days
user_activity['subs_per_day'] = user_activity['total_submissions'] / user_activity['tenure_days']


user_activity.head()


# Compare submission timing by performance tier
timing_analysis = merged.groupby(['PerformanceTier', 'SubmissionPhase']).agg(
    avg_score=('PublicScoreFullPrecision', 'mean'),
    count=('Id_sub', 'count')
).reset_index()

# Visualize
import plotly.express as px
px.bar(
    timing_analysis, 
    x='SubmissionPhase', 
    y='avg_score', 
    color='PerformanceTier',
    title="Score by Submission Timing and User Tier",
    barmode='group'
)




