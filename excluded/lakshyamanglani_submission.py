# Sample Kaggle Notebook for Meta Kaggle Hackathon
# Title: Exploring Kaggle Competition Participation Trends

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# Download Meta Kaggle dataset
path = kagglehub.dataset_download("kaggle/meta-kaggle")
print("Path to dataset files:", path)

# Load relevant CSV files
competitions_path = os.path.join(path, "Competitions.csv")
users_path = os.path.join(path, "Users.csv")
submissions_path = os.path.join(path, "Submissions.csv")

# Read CSVs with low_memory=False to handle mixed types in Submissions.csv
competitions_df = pd.read_csv(competitions_path)
users_df = pd.read_csv(users_path)
submissions_df = pd.read_csv(submissions_path, low_memory=False)

# Basic EDA: Number of competitions over time
# Use 'EnabledDate' instead of 'StartDate'
competitions_df['EnabledDate'] = pd.to_datetime(competitions_df['EnabledDate'], errors='coerce')
competitions_df['Year'] = competitions_df['EnabledDate'].dt.year

# Plot number of competitions per year
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=competitions_df, palette='viridis')
plt.title('Number of Kaggle Competitions per Year')
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.xticks(rotation=45)
plt.show()

# Analyze submission trends
submission_counts = submissions_df.groupby('TeamId').size().reset_index(name='SubmissionCount')
print("Average submissions per team:", submission_counts['SubmissionCount'].mean())

# Merge with competitions to analyze participation by competition type
# Ensure correct column name for competition ID
competition_types = competitions_df[['Id', 'HostSegmentTitle']].rename(columns={'Id': 'CompetitionId'})
# Verify column names in submissions_df if error persists
try:
    submission_trends = submissions_df.merge(competition_types, on='CompetitionId')
    submission_by_type = submission_trends.groupby('HostSegmentTitle').size().reset_index(name='Submissions')

    # Plot submissions by competition type
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Submissions', y='HostSegmentTitle', data=submission_by_type, palette='magma')
    plt.title('Submissions by Competition Type')
    plt.xlabel('Number of Submissions')
    plt.ylabel('Competition Type')
    plt.show()
except KeyError as e:
    print(f"KeyError: {e}. Available columns in submissions_df: {list(submissions_df.columns)}")
    print(f"Available columns in competitions_df: {list(competitions_df.columns)}")

# Save the notebook (handled automatically in Kaggle)
print("Notebook analysis complete. Ready for submission.")

