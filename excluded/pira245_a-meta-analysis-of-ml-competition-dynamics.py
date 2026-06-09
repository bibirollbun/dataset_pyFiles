import os
import sys
import datetime
from datetime import datetime
import pathlib
from pathlib import Path
import shutil
import pickle
import warnings
os.environ['OMP_NUM_THREADS'] = '10'
warnings.filterwarnings('ignore')
# Kaggle API (https://github.com/Kaggle/kagglehub)
import kagglehub
from kagglehub import KaggleDatasetAdapter


# General use default library
import numpy as np
import pandas as pd
import time
import re
# visualisation library
import IPython.display
from IPython.display import Image, display
import matplotlib
from matplotlib import pyplot as plt
import itertools
from itertools import cycle
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
sns.set(rc={'figure.figsize':(10,10)})
sns.set_theme()


import meta_kaggle_hackathon_utility_functions as my_utils
print(my_utils.notebook_folder)
data_folders_dictionary = my_utils.data_folder(my_utils.notebook_folder)


# Set the path to the file you'd like to load
competitions_file_path_list = ['/kaggle/input/meta-kaggle/Competitions.csv',
                                  '/kaggle/input/meta-kaggle/Submissions.csv',
                                  '/kaggle/input/meta-kaggle/Teams.csv',
                                  '/kaggle/input/meta-kaggle/Forums.csv',
                                  '/kaggle/input/meta-kaggle/ForumTopics.csv',
                                  '/kaggle/input/meta-kaggle/ForumMessages.csv',
                                  '/kaggle/input/meta-kaggle/ForumMessageVotes.csv',
                                  '/kaggle/input/meta-kaggle/ForumMessageReactions.csv']


# Set the path to the file you'd like to load
notebooks_file_path_list = ['/kaggle/input/meta-kaggle/Kernels.csv',
                            '/kaggle/input/meta-kaggle/KernelVotes.csv',
                            '/kaggle/input/meta-kaggle/KernelVersions.csv',
                            '/kaggle/input/meta-kaggle/KernelVersionKernelSources.csv']


# Load a DataFrame
file_path = competitions_file_path_list[0]
competitions_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
print("Dataframe Shape:", competitions_df.shape)
print("Dataframe Columns:\n", competitions_df.columns)
print("\nNull Counts:\n", competitions_df.isnull().sum())
competitions_df.head(5)


columns_to_be_drop = [
'OrganizationId',
'ProhibitNewEntrantsDeadlineDate',
'TeamMergerDeadlineDate',
'TeamModelDeadlineDate',
'ModelSubmissionDeadlineDate',
'RewardType',
'RewardQuantity',
'ValidationSetName',
'ValidationSetValue',
'HostName',
]


competitions_df = competitions_df.drop(columns_to_be_drop, axis=1)


# Convert 'EnabledDate' to datetime
competitions_df['EnabledDate'] = pd.to_datetime(competitions_df['EnabledDate'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
# Create a helper column for year
competitions_df['Year'] = competitions_df['EnabledDate'].dt.year
competitions_df = competitions_df[competitions_df['Year'].isin(list(range(2010, 2026, 1)))]


my_utils.plot_competitions_by_year(competitions_df)


metric = {
    'TotalCompetitors': 'Total Competitors Statistics 2010-2025'
}
print('Metric: ---------------{}---------------'.format(metric['TotalCompetitors']))
# Plot for a metric:
for feature, title in metric.items():
    # Plot a metric over time with the average
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='mean')
    # Plot a metric over time with the median
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='median')
    # Plot a metric over time with boxplot
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='boxplot')


metric = {
    'TotalTeams': 'Total Teams Statistics 2010-2025'
}
print('Metric: ---------------{}---------------'.format(metric['TotalTeams']))
# Plot for a metric:
for feature, title in metric.items():
    # Plot a metric over time with the average
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='mean')
    # Plot a metric over time with the median
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='median')
    # Plot a metric over time with boxplot
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='boxplot')


metric = {
    'TotalSubmissions': 'Total Submissions Statistics 2010-2025'
}
print('Metric: ---------------{}---------------'.format(metric['TotalSubmissions']))
# Plot for a metric:
for feature, title in metric.items():
    # Plot a metric over time with the average
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='mean')
    # Plot a metric over time with the median
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='median')
    # Plot a metric over time with boxplot
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='boxplot')


metric = {
    'NumScoredSubmissions': 'Number of Scored Submissions Statistics 2010-2025'
}
print('Metric: ---------------{}---------------'.format(metric['NumScoredSubmissions']))
# Plot for a metric:
for feature, title in metric.items():
    # Plot a metric over time with the average
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='mean')
    # Plot a metric over time with the median
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='median')
    # Plot a metric over time with boxplot
    my_utils.plot_feature_stat_trend(competitions_df, feature, title, mode='boxplot')


competitions_df_no_2020 = competitions_df[competitions_df['Year'] != 2020]
metric = {
    'NumScoredSubmissions': 'Number of Scored Submissions Statistics without 2020 data'
}
print('Metric: ---------------{}---------------'.format(metric['NumScoredSubmissions']))
# Plot for a metric:
for feature, title in metric.items():
    # Plot a metric over time with the average
    my_utils.plot_feature_stat_trend(competitions_df_no_2020, feature, title, mode='mean')
    # Plot a metric over time with the median
    my_utils.plot_feature_stat_trend(competitions_df_no_2020, feature, title, mode='median')
    # Plot a metric over time with boxplot
    my_utils.plot_feature_stat_trend(competitions_df_no_2020, feature, title, mode='boxplot')


competitions_df = my_utils.generate_problem_corpus(competitions_df)
# Commonly used ML application words
for keyword in ['health', 'text']:
    my_utils.plot_word_trend(competitions_df, keyword)


# Commonly used ML application words
for keyword in ['finance', 'image']:
    my_utils.plot_word_trend(competitions_df, keyword)


keywords_predictive_maintenance_industrial = {
    "description": "Predictive-Maintenance in Industrial applications",
    "General ML": [
        "predictive maintenance", "anomaly detection", "failure prediction",
        "classification", "regression", "time series"
    ],
    "Equipment": [
        "machine health", "vibration analysis", "sensor data",
        "equipment monitoring", "motor failure", "bearing analysis"
    ],
    "Temporal": [
        "remaining useful life (RUL)", "lifetime estimation", "early warning",
        "condition-based maintenance", "trend detection"
    ],
    "Data Sources": [
        "IoT", "SCADA", "sensor fusion", "data acquisition",
        "telemetry", "log data"
    ]
}
# Save dictionaries:
folder = data_folders_dictionary['process_data']
filepath = data_folders_dictionary['process_data'] / Path('keywords_predictive_maintenance_industrial.pkl')
my_utils.handle_pickle_dict(folder=folder, pickle_filename='keywords_predictive_maintenance_industrial.pkl', data_dict=keywords_predictive_maintenance_industrial)


keywords_predictive_maintenance_energy = {
    "description": "Predictive-Maintenance in Energy Sector",
    "Energy": [
        "turbine", "turbine monitoring", "generator failure", "transformer diagnostics",
        "grid fault detection", "power outages", "oil & gas", "generator", "transformer", "energy demand"
    ],
    "Maintenance": [
        "fault classification", "failure prediction", "degradation modeling",
        "SCADA anomalies", "smart grid diagnostics", "sensor fusion", "pipeline",
        "industrial", "equipment", "maintenance", "vibration", "historical maintenance logs"
    ],
    "Data Types": [
        "power quality", "thermal imaging", "vibration",
        "acoustic signals", "historical maintenance logs"
    ]
}
# Save dictionaries:
folder = data_folders_dictionary['process_data']
filepath = data_folders_dictionary['process_data'] / Path('keywords_predictive_maintenance_energy.pkl')
my_utils.handle_pickle_dict(folder=folder, pickle_filename='keywords_predictive_maintenance_energy.pkl', data_dict=keywords_predictive_maintenance_energy)


#load dictionary
folder = data_folders_dictionary['process_data']
dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='keywords_predictive_maintenance_industrial.pkl')
df = my_utils.assign_topic_categories(competitions_df, dictionary)


my_utils.plot_category_trends(df, dictionary)


my_utils.plot_category_proportions(df, dictionary)


# Group by top Hosts in a subdomain
df_gen_ml_host_segment = df[df["General ML"] == 1].groupby("HostSegmentTitle").size().sort_values(ascending=False)
# plot pie distribution for subdomain based on top 5 frequent host segments:
my_utils.plot_distribution_pie(df=df_gen_ml_host_segment, title="Host distribution for General ML focus competitions", top_n=5)


# Analyse which evaluation algorithn dominate a subdomain
df_gen_ml_eva_algorithm = df[df["General ML"] == 1].groupby("EvaluationAlgorithmName").size().sort_values(ascending=False)
#plot pie distribution for subdomain based on top 10 frequent evaluation algorithm:
my_utils.plot_distribution_pie(df=df_gen_ml_eva_algorithm, title="Evaluation algorithm distribution for General ML focus competitions", top_n=10)


# Group by top Hosts in a subdomain
df_data_sources_host_segment = df[df["Data Sources"] == 1].groupby("HostSegmentTitle").size().sort_values(ascending=False)
# plot pie distribution for subdomain based on top 5 frequent host segments:
my_utils.plot_distribution_pie(df=df_data_sources_host_segment, title="Host distribution for Data Sources focus competitions", top_n=5)


# Analyse which evaluation algorithn dominate a subdomain
df_data_sources_eva_algorithm = df[df["Data Sources"] == 1].groupby("EvaluationAlgorithmName").size().sort_values(ascending=False)
#plot pie distribution for subdomain based on top 10 frequent evaluation algorithm:
my_utils.plot_distribution_pie(df=df_data_sources_eva_algorithm, title="Evaluation algorithm distribution for Data Sources focus competitions", top_n=10)


#load dictionary
folder = data_folders_dictionary['process_data']
dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='keywords_predictive_maintenance_energy.pkl')
df = my_utils.assign_topic_categories(competitions_df, dictionary)


my_utils.plot_category_trends(df, dictionary)


my_utils.plot_category_proportions(df, dictionary)


# Group by top Hosts in a subdomain
df_maintenance_host_segment = df[df["Maintenance"] == 1].groupby("HostSegmentTitle").size().sort_values(ascending=False)
# plot pie distribution for subdomain based on top 5 frequent host segments:
my_utils.plot_distribution_pie(df=df_maintenance_host_segment, title="Host distribution for Mainteance focus competitions", top_n=5)


# Group by top Hosts in a subdomain
df_maintenance_eva_algorithm_segment = df[df["Maintenance"] == 1].groupby("EvaluationAlgorithmName").size().sort_values(ascending=False)
# plot pie distribution for subdomain based on top 10 frequent host segments:
my_utils.plot_distribution_pie(df=df_maintenance_eva_algorithm_segment, title="Evaluation algorithm distribution for Mainteance focus competitions", top_n=10)


# Group by top Hosts in a subdomain
df_energy_host_segment = df[df["Energy"] == 1].groupby("HostSegmentTitle").size().sort_values(ascending=False)
# plot pie distribution for subdomain based on top 5 frequent host segments:
my_utils.plot_distribution_pie(df=df_energy_host_segment , title="Host distribution for Maintenance in Energy sector focus competitions", top_n=5)


# Group by top Hosts in a subdomain
df_energy_eva_algorithm_segment = df[df["Energy"] == 1].groupby("EvaluationAlgorithmName").size().sort_values(ascending=False)
# plot pie distribution for subdomain based on top 10 frequent host segments:
my_utils.plot_distribution_pie(df=df_energy_eva_algorithm_segment, title="Evaluation algorithm distribution for Maintenance in Energy sector focus competitions", top_n=10)


# Check rows where maintenance is true (aka equal 1) and energy is also true:
df_maintenance_energy = df[(df["Maintenance"] == 1) & (df["Energy"] == 1)]
df_maintenance_energy.shape


df_maintenance_energy.head(5)


# Extract top 10 most frequent topics based on a LDA analysis.
topics = my_utils.perform_topic_modeling(df_maintenance_energy, n_topics=10)


 my_utils.plot_lda_topics(topic_dict=topics, top_n_words=10)


# Number of random rows
n = 3
# Subset with n random rows
df_me_subset = df_maintenance_energy.sample(n=n)
for index, row in df_me_subset.iterrows():
    print('\n-o-o-o-new-text'*2)
    #print(row['CombinedText'])


text = 'wind energy is one of the most developed technologies worldwide identify failures'  # You can replace this with your keyword or pattern
for index, row in df_maintenance_energy.iterrows():
    if re.search(text, row['CombinedText'], re.IGNORECASE):  # Case-insensitive search
        print(index)


df_maintenance_energy['Overview'][7631]


row_index = 7631  # The row index to inspect
cols_list = ['Id',
                'Title',
                'Subtitle',
                'HostSegmentTitle',
                'ForumId',
                'Year',
                'EnabledDate',
                'TotalTeams',
                'TotalCompetitors',
                'TotalSubmissions',
                'LicenseName']
# Display selected column values from the specified row
print(df_maintenance_energy.loc[row_index, cols_list])


df_maintenance_energy[df_maintenance_energy['Id'] == 68402]


# Sort competitions directly by each metric:
metrics = ["TotalTeams", "TotalCompetitors", "TotalSubmissions", 'NumScoredSubmissions']

top_teams = df_maintenance_energy.sort_values(by=metrics[0], ascending=False)[
    ["Id", "Title", "Year", "TotalTeams", "TotalCompetitors", "TotalSubmissions", 'NumScoredSubmissions']
].head(5)

top_competitors = df_maintenance_energy.sort_values(by=metrics[1], ascending=False)[
    ["Id", "Title", "Year", "TotalTeams", "TotalCompetitors", "TotalSubmissions", 'NumScoredSubmissions']
].head(5)

top_submissions = df_maintenance_energy.sort_values(by=metrics[2], ascending=False)[
    ["Id", "Title", "Year", "TotalTeams", "TotalCompetitors", "TotalSubmissions", 'NumScoredSubmissions']
].head(5)

# Display the tables
print("ğŸ”¹ Top 5 Competitions by Total Teams:\n", top_teams)
print("\nğŸ”¹ Top 5 Competitions by Total Competitors:\n", top_competitors)
print("\nğŸ”¹ Top 5 Competitions by Total Submissions:\n", top_submissions)


# Load a DataFrame
file_path = competitions_file_path_list[2]
teams_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
print("Dataframe Shape:", teams_df.shape)
print("Dataframe Columns:\n", teams_df.columns)
print("\nNull Counts:\n", teams_df.isnull().sum())
teams_df.head(5)


# Filter teams where competitionId equal 68402
filtered_teams_by_competition = teams_df[teams_df['CompetitionId'].isin([68402])]
# Display the result
filtered_teams_by_competition.shape


df_maintenance_energy[df_maintenance_energy['Id'] == 10684]


# Filter teams where competitionId equal 10684
filtered_teams_by_competition = teams_df[teams_df['CompetitionId'].isin([10684])]
# Display the result
filtered_teams_by_competition.shape


filtered_teams_by_competition.head()


print("Dataframe Shape:", filtered_teams_by_competition.shape)
print("Dataframe Columns:\n", filtered_teams_by_competition.columns)
print("\nNull Counts:\n", filtered_teams_by_competition.isnull().sum())


# Count NaN and non-null values in the column
nan_count = filtered_teams_by_competition['PrivateLeaderboardRank'].isna().sum()
not_nan_count = filtered_teams_by_competition['PrivateLeaderboardRank'].notna().sum()
# Prepare data and labels
counts = [not_nan_count, nan_count]
labels = ['Not NaN (Ranked)', 'NaN (Not Ranked)']
colors = ['#3fb557', '#3f82b5']
# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Distribution of teams participation during and after competition periode')
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.show()


# Filter teams where competitionId equal 10684 and PublicLeaderboardRank is not null:
filtered_teams_by_competition_bd = teams_df[(teams_df['CompetitionId']==10684)&(teams_df['PublicLeaderboardRank'].notna())] # before deadline for bd
# Save DataFrame to a specific CSV file path
file_path = data_folders_dictionary['process_data'] / Path('filtered_teams_by_competition_bd.csv')
filtered_teams_by_competition_bd.to_csv(file_path, index=False)
print(f"âœ… DataFrame saved to: {file_path}")
filtered_teams_by_competition_bd.shape


#file_path = data_folders_dictionary['process_data'] / Path('filtered_teams_by_competition_bd.csv')
#filtered_teams_by_competition_bd = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)


filtered_teams_by_competition_bd.head(5)


competition_10684_team_list = filtered_teams_by_competition_bd['Id'].to_list()


# Load a DataFrame
file_path = competitions_file_path_list[1]
submissions_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
print("Dataframe Shape:", submissions_df.shape)
print("Dataframe Columns:\n", submissions_df.columns)
print("\nNull Counts:\n", submissions_df.isnull().sum())
submissions_df.head(5)


# Filter submissions by selected list of teams for a given competition
filtered_submissions_by_team = submissions_df[submissions_df['TeamId'].isin(competition_10684_team_list)]
# Save DataFrame to a specific CSV file path
#file_path = data_folders_dictionary['process_data'] / Path('filtered_submissions_by_team.csv')
#filtered_submissions_by_team.to_csv(file_path, index=False)
#print(f"âœ… DataFrame saved to: {file_path}")
filtered_submissions_by_team.shape


#filepath = data_folders_dictionary['process_data'] / Path('filtered_submissions_by_team.csv')
#filtered_submissions_by_team = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)


filtered_submissions_by_team.head(10)


# # Load a DataFrame
# file_path = notebooks_file_path_list[1]
# notebooksvotes_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
# print("Dataframe Shape:", notebooksvotes_df.shape)
# print("Dataframe Columns:\n", notebooksvotes_df.columns)
# print("\nNull Counts:\n", notebooksvotes_df.isnull().sum())
# notebooksvotes_df.head(5)


# # Load a DataFrame
# file_path = notebooks_file_path_list[3]
# source_notebooksversion_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
# print("Dataframe Shape:", source_notebooksversion_df.shape)
# print("Dataframe Columns:\n", source_notebooksversion_df.columns)
# print("\nNull Counts:\n", source_notebooksversion_df.isnull().sum())
# source_notebooksversion_df.head(5)


# competition_teams_submissions_df = filtered_submissions_by_team.copy()
# kernel_source_version_df = source_notebooksversion_df.copy()
# kernel_votes_df = notebooksvotes_df.copy()
# filtered_submissions_by_team_votes_enrich_df = my_utils.enrich_with_kernel_votes(competition_teams_submissions_df, kernel_source_version_df, kernel_votes_df)


# # Save DataFrame to a specific CSV file path
# file_path = data_folders_dictionary['process_data'] / Path('filtered_submissions_by_team_votes_enrich_df.csv')
# filtered_submissions_by_team_votes_enrich_df.to_csv(file_path, index=False)
# print(f"âœ… DataFrame saved to: {file_path}")
# filtered_submissions_by_team_votes_enrich_df.head(5)


# file_path = '/kaggle/input/competition-enriched-datasets-id-10684/10684_filtered_submissions_by_team_votes_enrich_df.csv'
# filtered_submissions_by_team_votes_enrich_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
# Load a DataFrame with a specific version of a CSV
filtered_submissions_by_team_votes_enrich_df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"pira245/competition-enriched-datasets-id-10684",
    "10684_filtered_submissions_by_team_votes_enrich_df.csv",
)


print("Dataframe Shape:", filtered_submissions_by_team_votes_enrich_df.shape)
print("Dataframe Columns:\n", filtered_submissions_by_team_votes_enrich_df.columns)
print("\nNull Counts:\n", filtered_submissions_by_team_votes_enrich_df.isnull().sum())


submissions_with_kernel_and_votes = filtered_submissions_by_team_votes_enrich_df[filtered_submissions_by_team_votes_enrich_df['KernelVersionId'].notnull() & (filtered_submissions_by_team_votes_enrich_df['NumberVotes'] > 0)].shape[0]
submissions_with_kernel_and_votes 


top_3_voted_kernels = filtered_submissions_by_team_votes_enrich_df[filtered_submissions_by_team_votes_enrich_df['KernelVersionId'].notnull()].sort_values(by='NumberVotes', ascending=False).head(10)[['Id', 'TeamId', 'SourceKernelVersionId', 'PublicScoreFullPrecision','PrivateScoreFullPrecision', 'KernelVersionId', 'NumberVotes', 'WasVoted']]


top_3_voted_kernels.head()


print(my_utils.format_kernel_path(kernel_id=10096854))


# # Load a DataFrame
# file_path = competitions_file_path_list[4]
# forumtopic_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
# print("Dataframe Shape:", forumtopic_df.shape)
# print("Dataframe Columns:\n", forumtopic_df.columns)
# print("\nNull Counts:\n", forumtopic_df.isnull().sum())
# forumtopic_df.head(5)


# # Load a DataFrame
# file_path = competitions_file_path_list[5]
# forum_msg_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
# print("Dataframe Shape:", forum_msg_df.shape)
# print("Dataframe Columns:\n", forum_msg_df.columns)
# print("\nNull Counts:\n", forum_msg_df.isnull().sum())
# forum_msg_df.head(5)


# # Step 1: Filter topics by ForumId
# forum_id = 104336  # replace as needed
# forumtopic_info_df = forumtopic_df.loc[forumtopic_df['ForumId'] == forum_id,['Id', 'KernelId', 'TotalViews']]

# # Step 2: Extract topic IDs
# forum_topic_Id_list = forumtopic_info_df['Id'].tolist()

# # Step 3: Filter messages related to those topic IDs
# competition_forum_msg_df = forum_msg_df.loc[forum_msg_df['ForumTopicId'].isin(forum_topic_Id_list),['Id', 'ForumTopicId', 'PostDate', 'Message']]

# # Step 4: Rename 'Id' in forum_msg_df to 'MessageId'
# competition_forum_msg_df = competition_forum_msg_df.rename(columns={'Id': 'ForumMessageId'})

# # Step 5: Merge topic info with messages based on topic ID
# merged_forum_df = competition_forum_msg_df.merge(
#     forumtopic_info_df,
#     left_on='ForumTopicId',
#     right_on='Id',
#     how='left',
#     suffixes=('_msg', '_topic')
# )

# # Optional cleanup: drop duplicate 'Id_topic' column if needed
# merged_forum_df = merged_forum_df.drop(columns=['Id'])

# # Preview result
# merged_forum_df.head(5)


# # Load a DataFrame
# file_path = competitions_file_path_list[6]
# forum_msg_votes_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
# print("Dataframe Shape:", forum_msg_votes_df.shape)
# print("Dataframe Columns:\n", forum_msg_votes_df.columns)
# print("\nNull Counts:\n", forum_msg_votes_df.isnull().sum())
# forum_msg_votes_df.head(5)


# # Load a DataFrame
# file_path = competitions_file_path_list[7]
# forum_msg_rea_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
# print("Dataframe Shape:", forum_msg_rea_df.shape)
# print("Dataframe Columns:\n", forum_msg_rea_df.columns)
# print("\nNull Counts:\n", forum_msg_rea_df.isnull().sum())
# forum_msg_rea_df.head(5)


# filtered_forum_messages_df = merged_forum_df.copy()
# forum_messages_votes_reactions_enriched_df = my_utils.enrich_forum_messages_with_votes_and_reactions(filtered_forum_messages_df, forum_msg_votes_df, forum_msg_rea_df)
# # Save DataFrame to a specific CSV file path
# file_path = data_folders_dictionary['process_data'] / Path('forum_messages_votes_reactions_enriched_df.csv')
# forum_messages_votes_reactions_enriched_df.to_csv(file_path, index=False)
# print(f"âœ… DataFrame saved to: {file_path}")


# file_path = '/kaggle/input/competition-enriched-datasets-id-10684/104336_forum_messages_votes_reactions_enriched_df.csv'
# forum_messages_votes_reactions_enriched_df = pd.read_csv(file_path, sep=',', encoding='utf-8', na_filter=True)
# Load a DataFrame with a specific version of a CSV
forum_messages_votes_reactions_enriched_df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"pira245/competition-enriched-datasets-id-10684",
    "104336_forum_messages_votes_reactions_enriched_df.csv",
)


print("Dataframe Shape:", forum_messages_votes_reactions_enriched_df.shape)
print("Dataframe Columns:\n", forum_messages_votes_reactions_enriched_df.columns)
print("\nNull Counts:\n", forum_messages_votes_reactions_enriched_df.isnull().sum())
forum_messages_votes_reactions_enriched_df.head(5)


teams_df = filtered_teams_by_competition_bd.copy()
forum_df = forum_messages_votes_reactions_enriched_df.copy()
submissions_df = filtered_submissions_by_team_votes_enrich_df.copy()


submissions_bd_df = submissions_df[submissions_df['IsAfterDeadline']==False]
submissions_bd_df.shape


my_utils.plot_submission_timeline(submissions_bd_df)


my_utils.plot_submission_timeline(submissions_df)


# Identify top 100 teams based on private leaderboard rank
top_100_teams = teams_df[teams_df ['PrivateLeaderboardRank'].notnull() ]. sort_values('PrivateLeaderboardRank').head(100)
top_100_ids = top_100_teams['Id'].tolist()
# Identify top 10 teams based on private leaderboard rank
top_10_teams = teams_df[teams_df['PrivateLeaderboardRank'].notnull() ]. sort_values('PrivateLeaderboardRank').head(10)
top_10_ids = top_10_teams['Id' ] .tolist()
# Merge submissions with team info to track trajectories
submissions_bd_df['SubmissionDate'] = pd.to_datetime(submissions_bd_df['SubmissionDate'],errors='coerce')
merged_subs = submissions_bd_df.merge(teams_df[['Id','TeamName']], left_on='TeamId', right_on='Id', how='left')
merged_subs = merged_subs. drop(['Id_x' ,'Id_y' ], axis=1)


merged_subs.columns.tolist()


# Plot number of submissions over time for top vs other teams
my_utils.plot_submission_trends(merged_subs, top_team_ids=top_10_ids)


# Plot number of submissions over time for top vs other teams
my_utils.plot_submission_trends(merged_subs, top_team_ids=top_100_ids)


# Generate the static plot
my_utils.plot_public_score_evolution(merged_subs, top_10_ids)


# Generate the static plot
my_utils.plot_private_score_evolution(merged_subs, top_10_ids)


# Convert PostDate to datetime and extract month
forum_df['PostDate'] = pd.to_datetime(forum_df['PostDate'], errors='coerce')
forum_df['Month'] = forum_df['PostDate'].dt.to_period("M")
# Group and sum NumberVotes per month
monthly_votes = forum_df.groupby('Month')['NumberVotes'].sum()

# Apply Bold Minimal style
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 5))
monthly_votes.plot(kind='bar', ax=ax)

ax.set_facecolor('#3A506B')
ax.grid(True, axis='y', color='#41C8A9')
ax.set_title("Forum Votes Over Time", color='#062B3B', fontsize=14)
ax.set_ylabel("Total Votes", color='#062B3B')
ax.set_xlabel("Month", color='#062B3B')
ax.tick_params(colors='#062B3B', rotation=45)

sns.despine()
plt.tight_layout()
plt.show()


# Top 10 most upvoted forum posts
top_forum_posts = forum_df[['PostDate', 'Message', 'Upvotes']].sort_values(by='Upvotes', ascending=False).head(10)
top_forum_posts['Message'] = top_forum_posts['Message'].str.slice(0, 300) + "..."
top_forum_posts.reset_index(drop=True, inplace=True)
top_forum_posts.style.set_properties(**{'text-align': 'left'})    .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])


forum_df['Month'] = forum_df['PostDate'].dt.to_period("M")
# Group and sum NumberVotes per month
monthly_posts = forum_df.groupby('Month').size()

# Apply Bold Minimal style
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 5))
monthly_posts.plot(kind='bar', ax=ax)

ax.set_facecolor('#3A506B')
ax.grid(True, axis='y', color='#41C8A9')
ax.set_title("Forum Posts Over Time", color='#062B3B', fontsize=14)
ax.set_ylabel("Number of Posts", color='#062B3B')
ax.set_xlabel("Month", color='#062B3B')
ax.tick_params(colors='#062B3B', rotation=45)

sns.despine()
plt.tight_layout()
plt.show()


# Teams that linked writeups to forum threads
teams_with_writeups = teams_df[teams_df['WriteUpForumTopicId'].notnull()][['TeamName', 'WriteUpForumTopicId', 'Medal']]
teams_with_writeups = teams_with_writeups.drop_duplicates().sort_values(by='Medal', na_position='last')
teams_with_writeups.reset_index(drop=True, inplace=True)
teams_with_writeups.head(10)

