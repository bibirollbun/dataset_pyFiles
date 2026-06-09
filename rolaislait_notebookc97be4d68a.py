import pandas as pd
import matplotlib.pyplot as plt

competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
submissions = pd.read_csv('/kaggle/input/meta-kaggle/Submissions.csv')
competitions


competitions.columns


competitions.info()


competitions.describe()


competitions.isnull().sum()


kernel_links = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv')



submissions['SubmissionDate'] = pd.to_datetime(submissions['SubmissionDate'], errors='coerce')
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year


merged = submissions.merge(kernel_links[['KernelVersionId', 'SourceCompetitionId']],
                           how='left', left_on='SourceKernelVersionId', right_on='KernelVersionId')



merged.columns


competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

merged = merged.merge(competitions[['Id', 'Year']], how='left', left_on='SourceCompetitionId', right_on='Id')



submissions_per_year = merged.groupby('Year').size().reset_index(name='SubmissionsCount').astype(int)



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

sns.barplot(data=submissions_per_year, x='Year', y='SubmissionsCount', palette='viridis')

plt.title('Number of competition Participants each year', fontsize=16)
plt.xlabel('year', fontsize=14)
plt.ylabel('Participants number', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import pandas as pd

competitions['Year'] = competitions['EnabledDate'].dt.year


import matplotlib.pyplot as plt
import seaborn as sns

competitions_per_year = competitions.groupby('Year').size().reset_index(name='CompetitionsCount')
competitions_per_year['Year'] = competitions_per_year['Year'].astype(int)

plt.figure(figsize=(10, 6))
sns.barplot(data=competitions_per_year, x='Year', y='CompetitionsCount', palette='coolwarm')
plt.title('Number of competitions each year')
plt.xlabel('year')
plt.ylabel('Number of competitions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.barplot(data=competitions_per_year, x='Year', y='CompetitionsCount', palette='mako')
plt.title('Number of competitions each year', fontsize=14, weight='bold')
plt.xlabel('year', fontsize=12)
plt.ylabel('Number of competitions', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


submissions_per_competition = merged.groupby('SourceCompetitionId').size().reset_index(name='SubmissionsCount')
top_competitions = submissions_per_competition.sort_values(by='SubmissionsCount', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_competitions, x='SourceCompetitionId', y='SubmissionsCount', palette='rocket')
plt.title('Top 10 Competitions by Submission Count', fontsize=14, weight='bold')
plt.xlabel('Competition ID')
plt.ylabel('Number of Submissions')
plt.tight_layout()
plt.show()


merged['SubmissionYear'] = merged['SubmissionDate'].dt.year
submissions_by_year = merged.groupby('SubmissionYear').size().reset_index(name='SubmissionsCount')

plt.figure(figsize=(10, 6))
sns.lineplot(data=submissions_by_year, x='SubmissionYear', y='SubmissionsCount', marker='o', color='#2a5d77')
plt.title('Submissions by Year', fontsize=14, weight='bold')
plt.xlabel('Year')
plt.ylabel('Number of Submissions')
plt.xticks(submissions_by_year['SubmissionYear'].dropna().unique(), rotation=45)
plt.tight_layout()
plt.show()



merged['SubmissionYear'] = merged['SubmissionDate'].dt.year

avg_scores_year = merged.groupby('SubmissionYear')[['PublicScoreFullPrecision', 'PrivateScoreFullPrecision']].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_scores_year, x='SubmissionYear', y='PublicScoreFullPrecision', marker='o', label='Public Score', color='#1f77b4')
sns.lineplot(data=avg_scores_year, x='SubmissionYear', y='PrivateScoreFullPrecision', marker='o', label='Private Score', color='#ff7f0e')
plt.title('Average Public vs Private Scores by Year', fontsize=14, weight='bold')
plt.xlabel('Year')
plt.ylabel('Average Score')
plt.legend()
plt.tight_layout()
plt.show()


teams_per_year = merged.groupby('SubmissionYear')['TeamId'].nunique().reset_index(name='UniqueTeams')

plt.figure(figsize=(12,6))
sns.barplot(data=teams_per_year, x='SubmissionYear', y='UniqueTeams', palette='deep')
plt.title('Number of Unique Teams per Year', fontsize=14, weight='bold')
plt.xlabel('Year')
plt.ylabel('Number of Teams')
plt.tight_layout()
plt.show()





