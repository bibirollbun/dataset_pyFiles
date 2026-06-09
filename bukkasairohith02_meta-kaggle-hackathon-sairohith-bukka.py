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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from datetime import datetime

# Set plotting style
sns.set_style('whitegrid')

# Define the path to the Meta Kaggle dataset
data_path = "/kaggle/input/meta-kaggle/"

# List all files in the input directory to confirm what's available
print("Files available in data_path:")
available_files = os.listdir(data_path)
print(available_files)

# Initialize all DataFrames to None or empty structures to prevent NameErrors
# if a file fails to load
competitions = None
users = None
kernels = None
kernel_votes = None
forum_message_votes = None
dataset_votes = None
model_votes = None
kernel_versions = None
competition_teams = None
submissions = None
datasets = None
dataset_versions = None
forum_messages = None
tags = None
kernel_languages = None # This was the source of the NameError

print("\n--- Data Loading Start ---")
# Use a dictionary to specify files and their corresponding DataFrame names
files_to_load = {
    "Competitions.csv": "competitions",
    "Users.csv": "users",
    "Kernels.csv": "kernels",
    "KernelVotes.csv": "kernel_votes",
    "ForumMessageVotes.csv": "forum_message_votes",
    "DatasetVotes.csv": "dataset_votes",
    "ModelVotes.csv": "model_votes",
    "KernelVersions.csv": "kernel_versions",
    "Teams.csv": "competition_teams", # Using Teams.csv as previously discussed
    "Submissions.csv": "submissions",
    "Datasets.csv": "datasets",
    "DatasetVersions.csv": "dataset_versions",
    "ForumMessages.csv": "forum_messages",
    "Tags.csv": "tags",
    "KernelLanguages.csv": "kernel_languages"
}

for filename, df_name in files_to_load.items():
    file_path = data_path + filename
    if filename in available_files: # Check if the file actually exists
        try:
            # Handle specific low_memory warnings for large files
            if filename in ["KernelVersions.csv", "Submissions.csv", "Datasets.csv"]:
                globals()[df_name] = pd.read_csv(file_path, low_memory=False)
            else:
                globals()[df_name] = pd.read_csv(file_path)
            print(f"Loaded {filename}: shape {globals()[df_name].shape}")
        except Exception as e:
            print(f"Error loading {filename}: {e}. Skipping this file.")
            globals()[df_name] = None # Ensure it's None if loading fails
    else:
        print(f"Skipping {filename}: File not found in data_path. Setting to None.")
        globals()[df_name] = None # Ensure it's None if file is missing

print("\n--- Data Loading Complete ---")


## 2. Initial Data Preprocessing

print("\n--- Initial Data Preprocessing ---")

# Competitions DataFrame processing
print("\nProcessing Competitions data...")
if competitions is not None and not competitions.empty:
    competitions['DeadlineDate'] = pd.to_datetime(competitions['DeadlineDate'], errors='coerce')
    if 'StartDate' in competitions.columns:
        competitions['StartDate'] = pd.to_datetime(competitions['StartDate'], errors='coerce')
    competitions['HostSegmentTitle'] = competitions['HostSegmentTitle'].fillna('Unknown')
    print(f"Dtype of competitions['DeadlineDate'] after conversion: {competitions['DeadlineDate'].dtype}")
    print(f"First 5 DeadlineDate values: \n{competitions['DeadlineDate'].head()}")
else:
    print("Competitions DataFrame not loaded or is empty. Skipping preprocessing.")

# Users DataFrame processing
print("\nProcessing Users data...")
if users is not None and not users.empty:
    users['RegisterDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')
    print(f"Dtype of users['RegisterDate'] after conversion: {users['RegisterDate'].dtype}")
    print(f"First 5 RegisterDate values: \n{users['RegisterDate'].head()}")
else:
    print("Users DataFrame not loaded or is empty. Skipping preprocessing.")

# KernelVotes DataFrame processing
print("\nProcessing KernelVotes data...")
if kernel_votes is not None and not kernel_votes.empty:
    kernel_votes['VoteDate'] = pd.to_datetime(kernel_votes['VoteDate'], errors='coerce')
    print(f"Dtype of kernel_votes['VoteDate'] after conversion: {kernel_votes['VoteDate'].dtype}")
    print(f"First 5 VoteDate values: \n{kernel_votes['VoteDate'].head()}")
else:
    print("KernelVotes DataFrame not loaded or is empty. Skipping preprocessing.")

# Kernels DataFrame processing - CRITICAL for CreationDate
print("\nProcessing Kernels data...")
if kernels is not None and not kernels.empty:
    # Attempt specific format first, then general parse
    try:
        kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
    except ValueError: # Fallback if specific format fails
        print("Warning: Specific date format failed for Kernels['CreationDate']. Trying general parse.")
        kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'], errors='coerce')

    print(f"Dtype of kernels['CreationDate'] AFTER conversion attempt: {kernels['CreationDate'].dtype}")
    print(f"First 10 CreationDate values: \n{kernels['CreationDate'].head(10)}")
    print(f"Number of NaT values in kernels['CreationDate']: {kernels['CreationDate'].isnull().sum()}")

    # Drop rows where CreationDate became NaT
    initial_kernel_rows = kernels.shape[0]
    kernels.dropna(subset=['CreationDate'], inplace=True)
    if kernels.shape[0] < initial_kernel_rows:
        print(f"Dropped {initial_kernel_rows - kernels.shape[0]} rows from 'kernels' due to NaT in 'CreationDate'.")

    print(f"Final Dtype of kernels['CreationDate'] after cleaning: {kernels['CreationDate'].dtype}")
    print(f"Kernels dataframe shape after cleaning CreationDate: {kernels.shape}")
else:
    print("Kernels DataFrame not loaded or is empty. Skipping preprocessing.")

# Datasets DataFrame processing
print("\nProcessing Datasets data...")
if datasets is not None and not datasets.empty:
    datasets['CreationDate'] = pd.to_datetime(datasets['CreationDate'], errors='coerce')
    datasets['TotalDownloads'] = pd.to_numeric(datasets['TotalDownloads'], errors='coerce').fillna(0)
    print(f"Dtype of datasets['CreationDate'] after conversion: {datasets['CreationDate'].dtype}")
    print(f"First 5 CreationDate values: \n{datasets['CreationDate'].head()}")
else:
    print("Datasets DataFrame not loaded or is empty. Skipping preprocessing.")

# Mapping Tags and Kernel Languages
tag_names = {}
if tags is not None and not tags.empty:
    tag_names = tags[['Id', 'Slug']].set_index('Id')['Slug'].to_dict()
    print("Tags successfully loaded and mapped.")
else:
    print("Tags DataFrame not loaded or is empty. Tag mapping will be skipped.")

language_id_to_name = {}
if kernel_languages is not None and not kernel_languages.empty:
    language_id_to_name = kernel_languages.set_index('Id')['Name'].to_dict()
    print("KernelLanguages successfully loaded and mapped.")
else:
    print("KernelLanguages DataFrame not loaded or is empty. Kernel language mapping will be skipped.")



## 3. Exploratory Data Analysis (EDA) and Feature Engineering

### 3.1. Evolution of Kaggle Competitions

print("\n--- Analyzing Competition Evolution ---")

# Number of competitions over time
if competitions is not None and not competitions.empty and \
   not competitions['DeadlineDate'].isnull().all() and \
   pd.api.types.is_datetime64_any_dtype(competitions['DeadlineDate']): # Ensure it's datetime

    competitions['Year'] = competitions['DeadlineDate'].dt.year
    competitions_per_year = competitions.groupby('Year').size().reset_index(name='NumCompetitions')

    if not competitions_per_year.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Year', y='NumCompetitions', data=competitions_per_year, marker='o')
        plt.title('Number of Kaggle Competitions Over Time (by Deadline Year)')
        plt.xlabel('Year')
        plt.ylabel('Number of Competitions')
        plt.grid(True)
        plt.show()
    else:
        print("Skipping 'Number of Kaggle Competitions Over Time' plot: Aggregated data is empty.")
else:
    print("Skipping 'Number of Kaggle Competitions Over Time' plot: Competitions data or DeadlineDate is empty/invalid.")

# Total prize money over time
if competitions is not None and not competitions.empty:
    cash_competitions = competitions[competitions['RewardType'] == 'Cash'].copy()

    if not cash_competitions.empty:
        cash_competitions['RewardQuantity_numeric'] = pd.to_numeric(
            cash_competitions['RewardQuantity'].astype(str).str.replace('$', '').str.replace(',', ''),
            errors='coerce'
        ).fillna(0)

        if not cash_competitions['DeadlineDate'].isnull().all() and \
           pd.api.types.is_datetime64_any_dtype(cash_competitions['DeadlineDate']):
            cash_competitions['Year'] = cash_competitions['DeadlineDate'].dt.year
            prize_money_per_year = cash_competitions.groupby('Year')['RewardQuantity_numeric'].sum().reset_index()

            if not prize_money_per_year.empty:
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Year', y='RewardQuantity_numeric', data=prize_money_per_year, palette='viridis')
                plt.title('Total Cash Prize Money in Kaggle Competitions Over Time (USD)')
                plt.xlabel('Year')
                plt.ylabel('Total Prize Money (USD)')
                plt.ticklabel_format(style='plain', axis='y')
                plt.show()
            else:
                print("Skipping 'Total Cash Prize Money' plot: Aggregated data is empty.")
        else:
            print("Skipping 'Total Cash Prize Money' plot: DeadlineDate column in cash_competitions is empty/invalid.")
    else:
        print("Skipping 'Total Cash Prize Money' plot: No cash competitions found.")
else:
    print("Skipping 'Total Cash Prize Money' plot: Competitions data is empty.")

# Top competition types/tags
if competitions is not None and not competitions.empty:
    plt.figure(figsize=(10, 8))
    top_host_segments = competitions['HostSegmentTitle'].value_counts().head(10)
    if not top_host_segments.empty:
        top_host_segments.plot(kind='barh', color='skyblue')
        plt.title('Top 10 Most Frequent Competition Host Segments/Types')
        plt.xlabel('Number of Competitions')
        plt.ylabel('Host Segment Title')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print("Skipping 'Top 10 Host Segments' plot: No host segments found.")
else:
    print("Skipping 'Top 10 Host Segments' plot: Competitions data is empty.")

# Evolution of Competition Types (HostSegmentTitle)
if competitions is not None and not competitions.empty:
    if 'Year' in competitions.columns and not competitions['Year'].isnull().all():
        competition_types_over_time = competitions.groupby(['Year', 'HostSegmentTitle']).size().unstack(fill_value=0)
        if not competition_types_over_time.empty:
            top_n_types = competitions['HostSegmentTitle'].value_counts().head(5).index
            top_n_types_existing = [t for t in top_n_types if t in competition_types_over_time.columns]
            if top_n_types_existing:
                plt.figure(figsize=(14, 8))
                competition_types_over_time[top_n_types_existing].plot(kind='area', stacked=True, alpha=0.7, figsize=(14, 8))
                plt.title('Evolution of Top 5 Competition Host Segments Over Time')
                plt.xlabel('Year')
                plt.ylabel('Number of Competitions')
                plt.legend(title='Host Segment')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.show()
            else:
                print("Skipping 'Evolution of Top 5 Competition Host Segments' plot: No top host segments to plot.")
        else:
            print("Skipping 'Evolution of Top 5 Competition Host Segments' plot: Aggregated data is empty.")
    else:
        print("Skipping 'Evolution of Top 5 Competition Host Segments' plot: 'Year' column is empty/invalid.")
else:
    print("Skipping 'Evolution of Top 5 Competition Host Segments' plot: Competitions data is empty.")

# Average TotalCompetitors per competition over time
if competitions is not None and not competitions.empty:
    if 'Year' in competitions.columns and not competitions['Year'].isnull().all():
        avg_competitors_per_comp = competitions.groupby('Year')['TotalCompetitors'].mean().reset_index()
        if not avg_competitors_per_comp.empty:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='Year', y='TotalCompetitors', data=avg_competitors_per_comp, marker='o', color='green')
            plt.title('Average Number of Competitors per Competition Over Time')
            plt.xlabel('Year')
            plt.ylabel('Average Total Competitors')
            plt.grid(True)
            plt.show()
        else:
            print("Skipping 'Average Competitors per Competition' plot: Aggregated data is empty.")
    else:
        print("Skipping 'Average Competitors per Competition' plot: 'Year' column is empty/invalid.")
else:
    print("Skipping 'Average Competitors per Competition' plot: Competitions data is empty.")




### 3.2. User Growth and Engagement

print("\n--- Analyzing User Growth and Engagement ---")

# Monthly user registrations
if users is not None and not users.empty and \
   not users['RegisterDate'].isnull().all() and \
   pd.api.types.is_datetime64_any_dtype(users['RegisterDate']):

    users['RegisterMonthYear'] = users['RegisterDate'].dt.to_period('M')
    user_registrations_monthly = users.groupby('RegisterMonthYear').size().reset_index(name='NumRegistrations')
    user_registrations_monthly['RegisterMonthYear'] = user_registrations_monthly['RegisterMonthYear'].astype(str)

    if not user_registrations_monthly.empty:
        plt.figure(figsize=(15, 7))
        sns.lineplot(x='RegisterMonthYear', y='NumRegistrations', data=user_registrations_monthly, color='purple')
        plt.title('Monthly Kaggle User Registrations')
        plt.xlabel('Month-Year')
        plt.ylabel('Number of New Users')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping 'Monthly User Registrations' plot: Aggregated data is empty.")
else:
    print("Skipping 'Monthly User Registrations' plot: Users data or RegisterDate is empty/invalid.")

# Distribution of user PerformanceTiers
if users is not None and not users.empty:
    tier_map = {0: 'Novice', 1: 'Contributor', 2: 'Expert', 3: 'Master', 4: 'Grandmaster', 5: 'Kaggle Team'}
    users['PerformanceTierName'] = users['PerformanceTier'].map(tier_map)

    tier_counts = users['PerformanceTierName'].value_counts()
    if not tier_counts.empty:
        plt.figure(figsize=(10, 6))
        actual_order = [t for t in tier_map.values() if t in tier_counts.index]
        sns.countplot(x='PerformanceTierName', data=users, order=actual_order, palette='plasma')
        plt.title('Distribution of Kaggle User Performance Tiers')
        plt.xlabel('User Tier')
        plt.ylabel('Number of Users')
        plt.show()
    else:
        print("Skipping 'User Performance Tiers' plot: No valid performance tier data.")
else:
    print("Skipping 'User Performance Tiers' plot: Users data is empty.")

# Top countries by Kaggle user base
if users is not None and not users.empty:
    top_countries = users['Country'].value_counts().head(15)
    if not top_countries.empty:
        plt.figure(figsize=(12, 7))
        top_countries.plot(kind='barh', color='teal')
        plt.title('Top 15 Countries by Kaggle User Count')
        plt.xlabel('Number of Users')
        plt.ylabel('Country')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print("Skipping 'Top Countries by User Count' plot: No country data found.")
else:
    print("Skipping 'Top Countries by User Count' plot: Users data is empty.")



### 3.3. Notebook Popularity and Trends

print("\n--- Analyzing Notebook Popularity and Trends ---")

# Number of public notebooks created over time
# CRITICAL check for 'IsPrivate' column in kernels
if kernels is not None and not kernels.empty and \
   'IsPrivate' in kernels.columns and \
   not kernels['CreationDate'].isnull().all() and \
   pd.api.types.is_datetime64_any_dtype(kernels['CreationDate']):

    kernels['Year'] = kernels['CreationDate'].dt.year
    
    if 'Year' in kernels.columns and not kernels['Year'].isnull().all():
        public_kernels_per_year = kernels[kernels['IsPrivate'] == False].groupby('Year').size().reset_index(name='NumPublicKernels')

        if not public_kernels_per_year.empty:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='Year', y='NumPublicKernels', data=public_kernels_per_year, marker='o', color='orange')
            plt.title('Number of Public Kaggle Notebooks Created Over Time')
            plt.xlabel('Year')
            plt.ylabel('Number of Public Notebooks')
            plt.grid(True)
            plt.show()
        else:
            print("Skipping 'Number of Public Notebooks Created Over Time' plot: Data is empty after filtering/grouping.")
    else:
        print("Skipping 'Number of Public Notebooks Created Over Time' plot: 'Year' column could not be created or is invalid.")
else:
    print("Skipping 'Number of Public Notebooks Created Over Time' plot: Kernels data is empty, 'IsPrivate' column missing, or 'CreationDate' is invalid.")


# Top notebooks by votes
if kernel_votes is not None and not kernel_votes.empty and \
   kernels is not None and not kernels.empty and \
   'IsPrivate' in kernels.columns and \
   not kernel_votes['VoteDate'].isnull().all():

    required_kernel_cols = ['Id', 'Title', 'IsPrivate']
    if all(col in kernels.columns for col in required_kernel_cols):
        voted_kernels_merged = kernel_votes.merge(kernels[required_kernel_cols], left_on='KernelId', right_on='Id', how='inner')
        voted_public_kernels = voted_kernels_merged[voted_kernels_merged['IsPrivate'] == False]

        top_voted_kernels = voted_public_kernels.groupby('Title')['KernelId'].count().sort_values(ascending=False).head(10)

        if not top_voted_kernels.empty:
            plt.figure(figsize=(10, 7))
            top_voted_kernels.plot(kind='barh', color='salmon')
            plt.title('Top 10 Most Voted Public Notebooks (by vote count)')
            plt.xlabel('Number of Votes')
            plt.ylabel('Notebook Title')
            plt.gca().invert_yaxis()
            plt.show()
        else:
            print("Skipping 'Top 10 Most Voted Public Notebooks' plot: No public kernel vote data found.")
    else:
        print(f"Skipping 'Top 10 Most Voted Public Notebooks' plot: Missing required columns in 'kernels' dataframe ({required_kernel_cols}).")
else:
    print("Skipping 'Top 10 Most Voted Public Notebooks' plot: Required data (KernelVotes, Kernels, or 'IsPrivate' column) is empty or VoteDate is invalid.")


# Distribution of Script Languages in Public Notebooks
if kernel_versions is not None and not kernel_versions.empty and \
   kernels is not None and not kernels.empty and \
   'IsPrivate' in kernels.columns and \
   kernel_languages is not None and not kernel_languages.empty and \
   language_id_to_name:

    kernel_versions_public = kernel_versions.merge(kernels[['Id', 'IsPrivate']], left_on='ScriptId', right_on='Id', how='inner')
    kernel_versions_public = kernel_versions_public[kernel_versions_public['IsPrivate'] == False]

    if not kernel_versions_public.empty:
        kernel_versions_public['ScriptLanguageName'] = kernel_versions_public['ScriptLanguageId'].map(language_id_to_name).fillna('Unknown/Unmapped')

        language_distribution = kernel_versions_public['ScriptLanguageName'].value_counts()
        if not language_distribution.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=language_distribution.index, y=language_distribution.values, palette='coolwarm')
            plt.title('Distribution of Script Languages in Public Notebooks')
            plt.xlabel('Programming Language')
            plt.ylabel('Number of Notebook Versions')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print("Skipping 'Distribution of Script Languages' plot: No language distribution data found.")
    else:
        print("Skipping 'Distribution of Script Languages' plot: No public kernel versions found after merging.")
else:
    print("Skipping 'Distribution of Script Languages' plot: Required data (kernel_versions, kernels, 'IsPrivate' column, or kernel_languages/mapping) is empty.")


# Average number of votes per public notebook over time
if (kernels is not None and not kernels.empty and 'IsPrivate' in kernels.columns and \
    'Year' in kernels.columns and not kernels['Year'].isnull().all()) and \
   (kernel_votes is not None and not kernel_votes.empty and not kernel_votes['VoteDate'].isnull().all()):

    public_kernels_per_year_recalc = kernels[kernels['IsPrivate'] == False].groupby('Year').size().reset_index(name='NumPublicKernels')

    voted_kernels_merged = kernel_votes.merge(kernels[['Id', 'IsPrivate']], left_on='KernelId', right_on='Id', how='inner')
    voted_public_kernels = voted_kernels_merged[voted_kernels_merged['IsPrivate'] == False]

    if not public_kernels_per_year_recalc.empty and not voted_public_kernels.empty:
        votes_per_kernel_year = voted_public_kernels.groupby(voted_public_kernels['VoteDate'].dt.year)['KernelId'].count().reset_index(name='TotalVotes')
        votes_per_kernel_year.rename(columns={'VoteDate': 'Year'}, inplace=True)
        
        merged_votes_and_kernels = pd.merge(public_kernels_per_year_recalc, votes_per_kernel_year, on='Year', how='left').fillna(0)
        
        merged_votes_and_kernels['AvgVotesPerKernel'] = merged_votes_and_kernels.apply(
            lambda row: row['TotalVotes'] / row['NumPublicKernels'] if row['NumPublicKernels'] > 0 else 0, axis=1
        )

        if not merged_votes_and_kernels.empty and (merged_votes_and_kernels['AvgVotesPerKernel'] > 0).any():
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='Year', y='AvgVotesPerKernel', data=merged_votes_and_kernels, marker='o', color='brown')
            plt.title('Average Votes per Public Notebook Over Time')
            plt.xlabel('Year')
            plt.ylabel('Average Votes')
            plt.grid(True)
            plt.show()
        else:
            print("Skipping 'Average Votes per Public Notebook Over Time' plot: No data for average votes or all averages are zero.")
    else:
        print("Skipping 'Average Votes per Public Notebook Over Time' plot: Public kernels or voted kernels data is empty after recalculation.")
else:
    print("Skipping 'Average Votes per Public Notebook Over Time' plot: Required base data (kernels or kernel_votes), 'IsPrivate' column, or dates/year are invalid.")


### 3.4. Dataset Usage and Growth

print("\n--- Analyzing Dataset Usage and Growth ---")

# Number of public datasets created over time
if datasets is not None and not datasets.empty and \
   'IsPrivate' in datasets.columns and \
   not datasets['CreationDate'].isnull().all() and \
   pd.api.types.is_datetime64_any_dtype(datasets['CreationDate']):

    datasets['Year'] = datasets['CreationDate'].dt.year
    public_datasets_per_year = datasets[datasets['IsPrivate'] == False].groupby('Year').size().reset_index(name='NumPublicDatasets')

    if not public_datasets_per_year.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Year', y='NumPublicDatasets', data=public_datasets_per_year, marker='o', color='darkgreen')
        plt.title('Number of Public Kaggle Datasets Created Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Public Datasets')
        plt.grid(True)
        plt.show()
    else:
        print("Skipping 'Number of Public Datasets Created Over Time' plot: Data is empty.")
else:
    print("Skipping 'Number of Public Datasets Created Over Time' plot: Datasets data is empty, 'IsPrivate' column missing, or CreationDate is invalid.")


# Top downloaded datasets
# CRITICAL: Ensure 'Title' and 'TotalDownloads' are present in datasets
if datasets is not None and not datasets.empty and \
   'Title' in datasets.columns and 'TotalDownloads' in datasets.columns:
    
    top_downloaded_datasets = datasets.sort_values(by='TotalDownloads', ascending=False).head(10)
    
    top_downloaded_datasets = top_downloaded_datasets[top_downloaded_datasets['TotalDownloads'] > 0]

    # FINAL CHECK before plotting: Is the resulting DataFrame empty after all filters?
    if not top_downloaded_datasets.empty:
        plt.figure(figsize=(10, 7))
        sns.barplot(x='TotalDownloads', y='Title', data=top_downloaded_datasets, palette='cividis')
        plt.title('Top 10 Most Downloaded Public Datasets')
        plt.xlabel('Total Downloads')
        plt.ylabel('Dataset Title')
        plt.show()
    else:
        print("Skipping 'Top 10 Most Downloaded Public Datasets' plot: No public datasets with downloads found after filtering.")
else:
    print("Skipping 'Top 10 Most Downloaded Public Datasets' plot: Datasets data is empty or 'Title'/'TotalDownloads' columns are missing.")


# Percentage of public datasets with a license
if datasets is not None and not datasets.empty and 'HasLicense' in datasets.columns:
    license_distribution = datasets['HasLicense'].value_counts()
    if not license_distribution.empty:
        plt.figure(figsize=(8, 6))
        license_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['gold', 'lightcoral'])
        plt.title('Percentage of Public Datasets with a License')
        plt.ylabel('')
        plt.show()
    else:
        print("Skipping 'Percentage of Datasets with License' plot: No license data found.")
else:
    print("Skipping 'Percentage of Datasets with License' plot: Datasets data is empty or 'HasLicense' column is missing.")



import networkx as nx

print("\n--- Starting Advanced Analysis: Knowledge Flow and Influence Networks ---")

# Ensure required files for network analysis are loaded. Adding robust checks.
required_network_files = {
    "KernelVersionKernelSources.csv": "kernel_version_kernel_sources",
    "KernelVersionDatasetSources.csv": "kernel_version_dataset_sources"
}

# Initialize new dataframes to None. These are specific to network analysis sources.
kernel_version_kernel_sources = None
kernel_version_dataset_sources = None

# Loop to load the network-specific source files
for filename, df_name in required_network_files.items():
    file_path = data_path + filename
    if filename in available_files: # Check if the file actually exists in the input directory
        try:
            # All network source files might be large, use low_memory=False
            globals()[df_name] = pd.read_csv(file_path, low_memory=False)
            print(f"Loaded {filename}: shape {globals()[df_name].shape}")
        except Exception as e:
            print(f"Error loading {filename}: {e}. Skipping this file.")
            globals()[df_name] = None # Ensure it's None if loading fails
    else:
        print(f"Skipping {filename}: File not found in data_path. Setting to None.")
        globals()[df_name] = None # Ensure it's None if file is missing

# Initialize public_kernels and public_datasets with *all expected columns*
# This is crucial to prevent KeyError if filtering results in an empty DataFrame
# or if original CSV was missing columns.
public_kernels_cols = ['Id', 'Title', 'CreationDate', 'TotalVotes', 'TotalComments', 'IsPrivate']
public_kernels = pd.DataFrame(columns=public_kernels_cols) # Start with columns, even if empty

public_datasets_cols = ['Id', 'Title', 'TotalDownloads', 'CreationDate', 'IsPrivate', 'HasLicense']
public_datasets = pd.DataFrame(columns=public_datasets_cols) # Start with columns, even if empty

# Proceed only if core dataframes for network analysis are available for further processing
if kernels is None or kernels.empty:
    print("Skipping detailed Network Analysis preparation: 'kernels' DataFrame is not loaded or is empty.")
elif datasets is None or datasets.empty:
    print("Skipping detailed Network Analysis preparation: 'datasets' DataFrame is not loaded or is empty.")
elif kernel_versions is None or kernel_versions.empty:
    print("Skipping detailed Network Analysis preparation: 'kernel_versions' DataFrame is not loaded or is empty.")
else:
    # Create mapping from KernelVersionId to KernelId
    # Ensure kernel_versions is not empty before attempting to use it
    if not kernel_versions.empty:
        kernel_version_id_to_kernel_id = kernel_versions[['Id', 'ScriptId']].set_index('Id')['ScriptId'].to_dict()
        print(f"Created kernel_version_id_to_kernel_id map with {len(kernel_version_id_to_kernel_id)} entries.")
    else:
        print("KernelVersions is empty, cannot create KernelVersionId to KernelId map.")
        kernel_version_id_to_kernel_id = {} # Initialize as empty to prevent errors

    # Create mapping from DatasetVersionId to DatasetId
    dataset_version_id_to_dataset_id = {} # Initialize as empty
    # Ensure dataset_versions is not empty before attempting to use it
    if dataset_versions is not None and not dataset_versions.empty:
        dataset_version_id_to_dataset_id = dataset_versions[['Id', 'DatasetId']].set_index('Id')['DatasetId'].to_dict()
        print(f"Created dataset_version_id_to_dataset_id map with {len(dataset_version_id_to_dataset_id)} entries.")
    else:
        print("DatasetVersions DataFrame not loaded or is empty. Cannot create DatasetVersionId to DatasetId map.")

    # Filter kernels to only public ones for network analysis
    # Ensure 'IsPrivate' column exists in 'kernels' or handle its absence
    if kernels is not None and not kernels.empty: # Check if kernels has data
        if 'IsPrivate' in kernels.columns:
            public_kernels_filtered = kernels[kernels['IsPrivate'] == False].copy()
            # Ensure the pre-initialized public_kernels gets the data
            public_kernels = pd.DataFrame(public_kernels_filtered, columns=public_kernels_cols)
            print(f"Filtered to {public_kernels.shape[0]} public kernels.")
        else:
            print("WARNING: 'IsPrivate' column not found in 'kernels'. Assuming all kernels are public for network analysis.")
            public_kernels = pd.DataFrame(kernels.copy(), columns=public_kernels_cols) # Copy all kernels, retain schema
    else:
        print("WARNING: 'kernels' DataFrame is empty or not loaded. Public kernels will be an empty DataFrame.")
        # public_kernels is already initialized with columns above

    # Filter datasets to only public ones
    # Ensure 'IsPrivate' column exists in 'datasets' or handle its absence
    if datasets is not None and not datasets.empty: # Check if datasets has data
        if 'IsPrivate' in datasets.columns:
            public_datasets_filtered = datasets[datasets['IsPrivate'] == False].copy()
            # Ensure the pre-initialized public_datasets gets the data
            public_datasets = pd.DataFrame(public_datasets_filtered, columns=public_datasets_cols)
            print(f"Filtered to {public_datasets.shape[0]} public datasets.")
        else:
            print("WARNING: 'IsPrivate' column not found in 'datasets'. Assuming all datasets are public for network analysis.")
            public_datasets = pd.DataFrame(datasets.copy(), columns=public_datasets_cols) # Copy all datasets, retain schema
    else:
        print("WARNING: 'datasets' DataFrame is empty or not loaded. Public datasets will be an empty DataFrame.")
        # public_datasets is already initialized with columns above


    # Final check of column presence in the *now populated* public_kernels and public_datasets
    # This loop will add NaN columns if any are still truly missing from the loaded data
    for col in public_kernels_cols:
        if col not in public_kernels.columns:
            public_kernels[col] = np.nan
            print(f"Added missing column '{col}' to public_kernels with NaN values.")
    for col in public_datasets_cols:
        if col not in public_datasets.columns:
            public_datasets[col] = np.nan
            print(f"Added missing column '{col}' to public_datasets with NaN values.")

    # Create sets of valid public kernel and dataset IDs for node filtering
    valid_public_kernel_ids = set(public_kernels['Id'].unique()) if not public_kernels.empty else set()
    valid_public_dataset_ids = set(public_datasets['Id'].unique()) if not public_datasets.empty else set()
    print(f"Identified {len(valid_public_kernel_ids)} valid public kernel IDs.")
    print(f"Identified {len(valid_public_dataset_ids)} valid public dataset IDs.")


# Only proceed if public_kernels is not empty AND all source files are present
if not public_kernels.empty and \
   kernel_version_kernel_sources is not None and not kernel_version_kernel_sources.empty and \
   kernel_version_id_to_kernel_id:

    print("\n--- Building Kernel-to-Kernel Influence Network ---")
    G_kernel_to_kernel = nx.DiGraph()

    edges_k2k = []
    for index, row in kernel_version_kernel_sources.iterrows():
        source_kernel_version_id = row['KernelVersionId']
        target_kernel_version_id = row['SourceKernelVersionId']

        source_kernel_id = kernel_version_id_to_kernel_id.get(source_kernel_version_id)
        target_kernel_id = kernel_version_id_to_kernel_id.get(target_kernel_version_id)

        if source_kernel_id and target_kernel_id and \
           source_kernel_id in valid_public_kernel_ids and \
           target_kernel_id in valid_public_kernel_ids and \
           source_kernel_id != target_kernel_id:
            edges_k2k.append((source_kernel_id, target_kernel_id))

    G_kernel_to_kernel.add_edges_from(edges_k2k)
    print(f"Kernel-to-Kernel Network: {G_kernel_to_kernel.number_of_nodes()} nodes, {G_kernel_to_kernel.number_of_edges()} edges.")

    if G_kernel_to_kernel.number_of_nodes() > 0:
        in_degree_centrality_k2k = nx.in_degree_centrality(G_kernel_to_kernel)
        out_degree_centrality_k2k = nx.out_degree_centrality(G_kernel_to_kernel)

        try:
            # Adjust k based on graph size or a reasonable sample
            betweenness_centrality_k2k = nx.betweenness_centrality(G_kernel_to_kernel, k=min(G_kernel_to_kernel.number_of_nodes(), 500), seed=42) # Reduced k to 500
        except Exception as e:
            print(f"Warning: Betweenness centrality calculation failed: {e}. Skipping.")
            betweenness_centrality_k2k = {}

        in_degree_df = pd.DataFrame(in_degree_centrality_k2k.items(), columns=['KernelId', 'InDegreeCentrality'])
        out_degree_df = pd.DataFrame(out_degree_centrality_k2k.items(), columns=['KernelId', 'OutDegreeCentrality'])
        betweenness_df = pd.DataFrame(betweenness_centrality_k2k.items(), columns=['KernelId', 'BetweennessCentrality'])

        # Merge with public_kernels data using the *explicitly checked* columns
        kernel_cols_for_merge = ['Id', 'Title', 'CreationDate', 'TotalVotes', 'TotalComments']
        
        # Ensure these columns exist before attempting to merge
        if all(col in public_kernels.columns for col in kernel_cols_for_merge):
            in_degree_df = in_degree_df.merge(public_kernels[kernel_cols_for_merge], left_on='KernelId', right_on='Id', how='left')
            out_degree_df = out_degree_df.merge(public_kernels[kernel_cols_for_merge], left_on='KernelId', right_on='Id', how='left')
            betweenness_df = betweenness_df.merge(public_kernels[kernel_cols_for_merge], left_on='KernelId', right_on='Id', how='left')
        else:
            print("WARNING: Skipping merge for centrality DataFrames due to missing columns in public_kernels.")
            in_degree_df, out_degree_df, betweenness_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Set to empty

        if not in_degree_df.empty:
            print("\nTop 10 Kernels by In-Degree Centrality (Most Sourced):")
            print(in_degree_df.sort_values(by='InDegreeCentrality', ascending=False).head(10)[['Title', 'InDegreeCentrality', 'TotalVotes', 'CreationDate']])
            plt.figure(figsize=(12, 7))
            sns.barplot(x='InDegreeCentrality', y='Title', data=in_degree_df.nlargest(10, 'InDegreeCentrality'), palette='viridis')
            plt.title('Top 10 Kernels by In-Degree Centrality (Most Influential/Sourced)')
            plt.xlabel('In-Degree Centrality')
            plt.ylabel('Kernel Title')
            plt.tight_layout()
            plt.show()

        if not out_degree_df.empty:
            print("\nTop 10 Kernels by Out-Degree Centrality (Sourcing Others):")
            print(out_degree_df.sort_values(by='OutDegreeCentrality', ascending=False).head(10)[['Title', 'OutDegreeCentrality', 'TotalVotes', 'CreationDate']])
            plt.figure(figsize=(12, 7))
            sns.barplot(x='OutDegreeCentrality', y='Title', data=out_degree_df.nlargest(10, 'OutDegreeCentrality'), palette='magma')
            plt.title('Top 10 Kernels by Out-Degree Centrality (Sourcing Many Others)')
            plt.xlabel('Out-Degree Centrality')
            plt.ylabel('Kernel Title')
            plt.tight_layout()
            plt.show()

        if not betweenness_df.empty:
            print("\nTop 10 Kernels by Betweenness Centrality (Bridges):")
            print(betweenness_df.sort_values(by='BetweennessCentrality', ascending=False).head(10)[['Title', 'BetweennessCentrality', 'TotalVotes', 'CreationDate']])
            plt.figure(figsize=(12, 7))
            sns.barplot(x='BetweennessCentrality', y='Title', data=betweenness_df.nlargest(10, 'BetweennessCentrality'), palette='cividis')
            plt.title('Top 10 Kernels by Betweenness Centrality (Critical Bridges)')
            plt.xlabel('Betweenness Centrality')
            plt.ylabel('Kernel Title')
            plt.tight_layout()
            plt.show()

    else:
        print("Skipping Kernel-to-Kernel Centrality Analysis: Network has no nodes.")
else:
    print("Skipping Kernel-to-Kernel Network Analysis: Required source data is missing or empty.")


if kernel_version_dataset_sources is not None and not kernel_version_dataset_sources.empty and \
   kernel_version_id_to_kernel_id and dataset_version_id_to_dataset_id:

    print("\n--- Building Kernel-to-Dataset Influence Network ---")
    G_kernel_to_dataset = nx.DiGraph()

    # Prepare edges: (source_kernel_id, target_dataset_id)
    edges_k2d = []
    for index, row in kernel_version_dataset_sources.iterrows():
        source_kernel_version_id = row['KernelVersionId']
        target_dataset_version_id = row['SourceDatasetVersionId']

        source_kernel_id = kernel_version_id_to_kernel_id.get(source_kernel_version_id)
        target_dataset_id = dataset_version_id_to_dataset_id.get(target_dataset_version_id)

        # Only add edge if source kernel is public and target dataset is public and valid
        if source_kernel_id and target_dataset_id and \
           source_kernel_id in valid_public_kernel_ids and \
           target_dataset_id in valid_public_dataset_ids:
            edges_k2d.append((source_kernel_id, target_dataset_id))

    G_kernel_to_dataset.add_edges_from(edges_k2d)
    print(f"Kernel-to-Dataset Network: {G_kernel_to_dataset.number_of_nodes()} nodes, {G_kernel_to_dataset.number_of_edges()} edges.")

    if G_kernel_to_dataset.number_of_nodes() > 0:
        # For datasets, in-degree centrality is very important: how many kernels use them.
        # We need to compute centrality for *all* nodes (kernels and datasets), then filter.
        in_degree_centrality_k2d = nx.in_degree_centrality(G_kernel_to_dataset)

        # Convert to DataFrame
        in_degree_k2d_df = pd.DataFrame(in_degree_centrality_k2d.items(), columns=['NodeId', 'InDegreeCentrality'])

        # Identify which nodes are datasets vs. kernels
        in_degree_k2d_df['Type'] = in_degree_k2d_df['NodeId'].apply(
            lambda x: 'Dataset' if x in valid_public_dataset_ids else ('Kernel' if x in valid_public_kernel_ids else 'Other')
        )

        # Filter for datasets only
        dataset_influence_df = in_degree_k2d_df[in_degree_k2d_df['Type'] == 'Dataset'].copy()

        # Merge with public_datasets to get titles and total downloads
        dataset_influence_df = dataset_influence_df.merge(public_datasets[['Id', 'Title', 'TotalDownloads', 'CreationDate']], left_on='NodeId', right_on='Id', how='left')

        # Top 10 most used datasets (by in-degree centrality)
        print("\nTop 10 Datasets by In-Degree Centrality (Most Used by Kernels):")
        print(dataset_influence_df.sort_values(by='InDegreeCentrality', ascending=False).head(10)[['Title', 'InDegreeCentrality', 'TotalDownloads', 'CreationDate']])

        # Visualization for top datasets
        plt.figure(figsize=(12, 7))
        sns.barplot(x='InDegreeCentrality', y='Title', data=dataset_influence_df.nlargest(10, 'InDegreeCentrality'), palette='crest')
        plt.title('Top 10 Datasets by In-Degree Centrality (Most Used by Kernels)')
        plt.xlabel('In-Degree Centrality')
        plt.ylabel('Dataset Title')
        plt.tight_layout()
        plt.show()

    else:
        print("Skipping Kernel-to-Dataset Centrality Analysis: Network has no nodes.")
else:
    print("Skipping Kernel-to-Dataset Network Analysis: Required source data is missing or empty.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from datetime import datetime
import networkx as nx # Ensure networkx is imported if you're keeping 4.2/4.3 code

# --- (Assuming all previous data loading and preprocessing from sections 1, 2, 3 have been run successfully) ---
# Ensure these DataFrames are available and preprocessed as per previous steps
# kernels, kernel_votes, users, kernel_versions, language_id_to_name

print("\n--- Starting Phase 4: Advanced Insights ---")
print("\n--- 4.1. What Drives Notebook Popularity? ---")

# Ensure required DataFrames are loaded and not empty
if kernels is None or kernels.empty or \
   kernel_votes is None or kernel_votes.empty or \
   users is None or users.empty:
    print("Skipping 'What Drives Notebook Popularity?' analysis: Essential DataFrames (kernels, kernel_votes, users) are not loaded or empty.")
else:
    # 1. Merge Kernels with Users to get author's performance tier
    if 'AuthorUserId' in kernels.columns and 'Id' in users.columns:
        kernel_authors = kernels.merge(users[['Id', 'PerformanceTier', 'RegisterDate', 'PerformanceTierName']],
                                       left_on='AuthorUserId', right_on='Id', how='left', suffixes=('_kernel', '_author'))
        print(f"Merged kernels with user data. Shape: {kernel_authors.shape}")

        if 'Id_author' in kernel_authors.columns:
            kernel_authors.drop(columns=['Id_author'], inplace=True)
            
        if 'CreationDate' in kernel_authors.columns and pd.api.types.is_datetime64_any_dtype(kernel_authors['CreationDate']):
            kernel_authors['CreationYear'] = kernel_authors['CreationDate'].dt.year
        elif 'CreationDate_kernel' in kernel_authors.columns and pd.api.types.is_datetime64_any_dtype(kernel_authors['CreationDate_kernel']):
             kernel_authors['CreationYear'] = kernel_authors['CreationDate_kernel'].dt.year
        else:
            print("Warning: CreationDate or CreationDate_kernel not available or not datetime dtype for 'What Drives Notebook Popularity?'.")
            kernel_authors['CreationYear'] = np.nan

    else:
        print("Skipping author merge for notebook popularity: Missing 'AuthorUserId' in kernels or 'Id' in users.")
        kernel_authors = kernels.copy() # Proceed with kernels data if possible


    # --- Define latest_kernel_versions here, outside the language mapping block ---
    # This ensures it's always defined, even if kernel_versions is empty later.
    latest_kernel_versions = pd.DataFrame() # Initialize as empty DataFrame
    if kernel_versions is not None and not kernel_versions.empty:
        # Ensure required columns exist before sorting/dropping duplicates
        if 'ScriptId' in kernel_versions.columns and 'VersionNumber' in kernel_versions.columns:
            latest_kernel_versions = kernel_versions.sort_values(by='VersionNumber', ascending=False).drop_duplicates(subset='ScriptId').copy()
            print("latest_kernel_versions DataFrame created from kernel_versions.")
        else:
            print("Warning: Missing 'ScriptId' or 'VersionNumber' in kernel_versions. Cannot determine latest version.")
    else:
        print("Warning: kernel_versions DataFrame is empty or not loaded. Cannot determine latest kernel versions.")
    # --- End of latest_kernel_versions definition ---


    # Ensure popularity metrics are available and numeric
    popularity_metrics = ['TotalVotes', 'TotalComments', 'TotalViews']
    for col in popularity_metrics:
        if col not in kernel_authors.columns:
            print(f"Warning: Popularity metric '{col}' not found in kernels. Adding as 0.")
            kernel_authors[col] = 0.0
        else:
            kernel_authors[col] = pd.to_numeric(kernel_authors[col], errors='coerce').fillna(0)
    
    # Ensure language name is mapped for analysis
    # Only proceed if latest_kernel_versions is not empty and language_id_to_name exists
    if not latest_kernel_versions.empty and language_id_to_name:
        # Debugging prints for the merge error
        print(f"DEBUG MERGE: kernel_authors columns: {kernel_authors.columns.tolist()}")
        
        # Add checks before accessing columns of latest_kernel_versions for debug prints
        if not latest_kernel_versions.empty and 'ScriptId' in latest_kernel_versions.columns and 'ScriptLanguageId' in latest_kernel_versions.columns:
            print(f"DEBUG MERGE: latest_kernel_versions columns: {latest_kernel_versions.columns.tolist()}")
            print(f"DEBUG MERGE: kernel_authors 'Id_kernel' head: \n{kernel_authors['Id_kernel'].head()}") 
            print(f"DEBUG MERGE: latest_kernel_versions 'ScriptId' head: \n{latest_kernel_versions['ScriptId'].head()}")
        else:
            print("DEBUG MERGE: latest_kernel_versions is empty or missing 'ScriptId'/'ScriptLanguageId', skipping debug prints for its columns/head.")

        # Ensure latest_kernel_versions has 'ScriptId' and 'ScriptLanguageId' before merge
        required_lv_cols = ['ScriptId', 'ScriptLanguageId']
        if all(col in latest_kernel_versions.columns for col in required_lv_cols):
            latest_kernel_versions_filtered = latest_kernel_versions[required_lv_cols]
            
            # CHANGE HERE (already done in previous fix): Use 'Id_kernel' from kernel_authors to merge with 'ScriptId' from kernel_versions
            kernel_authors = kernel_authors.merge(latest_kernel_versions_filtered,
                                                 left_on='Id_kernel', right_on='ScriptId', how='left')
            
            if 'ScriptId_y' in kernel_authors.columns: # This column name indicates a merge conflict on ScriptId
                 kernel_authors.drop(columns=['ScriptId_y'], inplace=True)

            kernel_authors['ScriptLanguageName'] = kernel_authors['ScriptLanguageId'].map(language_id_to_name).fillna('Unknown')
            print("Successfully merged kernel language info.")
        else:
            print(f"Warning: Missing required columns in latest_kernel_versions for language merge ({required_lv_cols}). Skipping language mapping.")
            kernel_authors['ScriptLanguageName'] = 'Unknown' # Fallback
    else:
        print("Warning: Latest kernel versions data is empty or language mapping not available for notebook popularity analysis. Skipping language mapping.")
        kernel_authors['ScriptLanguageName'] = 'Unknown'


    # Focus on public kernels as popularity is more relevant there
    if 'IsPrivate' in kernel_authors.columns:
        popular_kernels = kernel_authors[kernel_authors['IsPrivate'] == False].copy()
    else:
        print("Warning: 'IsPrivate' column not available for popularity analysis. Using all kernels.")
        popular_kernels = kernel_authors.copy()

    if popular_kernels.empty:
        print("Skipping popularity analysis plots: No public kernels or kernels data available.")
    else:
        # --- Analysis 1: Notebook Popularity by Author Performance Tier ---
        if 'PerformanceTierName' in popular_kernels.columns and 'TotalVotes' in popular_kernels.columns:
            avg_votes_by_tier = popular_kernels.groupby('PerformanceTierName')['TotalVotes'].mean().reindex(
                ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster', 'Kaggle Team']
            ).fillna(0).reset_index()
            
            if not avg_votes_by_tier.empty and (avg_votes_by_tier['TotalVotes'] > 0).any():
                plt.figure(figsize=(10, 6))
                sns.barplot(x='PerformanceTierName', y='TotalVotes', data=avg_votes_by_tier, palette='viridis')
                plt.title('Average Votes per Public Notebook by Author Performance Tier')
                plt.xlabel('Author Tier')
                plt.ylabel('Average Votes')
                plt.show()
                print("Insight: Higher-tiered authors generally receive more votes, indicating established reputation or quality content.")
            else:
                print("Skipping 'Avg Votes by Author Tier' plot: Data empty or all votes are zero.")
        else:
            print("Skipping 'Avg Votes by Author Tier' plot: Missing PerformanceTierName or TotalVotes column.")


        # --- Analysis 2: Notebook Popularity by Programming Language ---
        if 'ScriptLanguageName' in popular_kernels.columns and 'TotalVotes' in popular_kernels.columns:
            avg_votes_by_language = popular_kernels.groupby('ScriptLanguageName')['TotalVotes'].mean().sort_values(ascending=False).reset_index()
            
            if not avg_votes_by_language.empty and (avg_votes_by_language['TotalVotes'] > 0).any():
                plt.figure(figsize=(10, 6))
                sns.barplot(x='TotalVotes', y='ScriptLanguageName', data=avg_votes_by_language.head(5), palette='plasma')
                plt.title('Average Votes per Public Notebook by Programming Language (Top 5)')
                plt.xlabel('Average Votes')
                plt.ylabel('Language')
                plt.show()
                print("Insight: Python notebooks receive highest average votes, confirming its dominance and engagement.")
            else:
                print("Skipping 'Avg Votes by Language' plot: Data empty or all votes are zero.")
        else:
            print("Skipping 'Avg Votes by Language' plot: Missing ScriptLanguageName or TotalVotes column.")


        # --- Analysis 3: Trend of Popularity (Average Votes) Over Time ---
        if 'CreationYear' in popular_kernels.columns and 'TotalVotes' in popular_kernels.columns:
            avg_votes_over_time = popular_kernels.groupby('CreationYear')['TotalVotes'].mean().reset_index()
            
            if not avg_votes_over_time.empty and (avg_votes_over_time['TotalVotes'] > 0).any():
                plt.figure(figsize=(12, 6))
                sns.lineplot(x='CreationYear', y='TotalVotes', data=avg_votes_over_time, marker='o', color='red')
                plt.title('Average Votes per Public Notebook Over Time')
                plt.xlabel('Year of Creation')
                plt.ylabel('Average Votes')
                plt.show()
                print("Insight: Average votes per notebook might show fluctuations, possibly indicating shifts in content volume or user attention span.")
            else:
                print("Skipping 'Avg Votes Over Time' plot: Data empty or all votes are zero.")
        else:
            print("Skipping 'Avg Votes Over Time' plot: Missing CreationYear or TotalVotes column.")


        # --- Analysis 4: Relationship between TotalViews and TotalVotes/Comments ---
        if 'TotalViews' in popular_kernels.columns and 'TotalVotes' in popular_kernels.columns:
            # Scatter plot to show relationship (sample if too large)
            plot_df = popular_kernels[['TotalViews', 'TotalVotes', 'TotalComments']].sample(n=min(10000, len(popular_kernels)), random_state=42)
            
            if not plot_df.empty:
                plt.figure(figsize=(12, 6))
                sns.scatterplot(x='TotalViews', y='TotalVotes', data=plot_df, alpha=0.5, hue='TotalComments', size='TotalComments', sizes=(20, 400), legend='full')
                plt.title('Total Views vs. Total Votes (Sampled)')
                plt.xlabel('Total Views')
                plt.ylabel('Total Votes')
                plt.xscale('log') # Views often have a wide range
                plt.yscale('log') # Votes often have a wide range
                plt.show()
                print("Insight: A positive correlation exists between views and votes, suggesting content visibility drives engagement. High comments often align with higher views/votes.")
            else:
                print("Skipping 'Views vs. Votes' plot: Data empty.")
        else:
            print("Skipping 'Views vs. Votes' plot: Missing TotalViews or TotalVotes column.")


# --- (Assuming relevant DataFrames like tags, competitions, kernels are loaded) ---
# Make sure competition_tags and kernel_tags are loaded or derived

print("\n--- 4.4. Tag-Level Trend Mining ---")

# Load CompetitionTags.csv and KernelTags.csv if not already loaded
required_tag_files = {
    "CompetitionTags.csv": "competition_tags",
    "KernelTags.csv": "kernel_tags"
}

competition_tags = None
kernel_tags = None

for filename, df_name in required_tag_files.items():
    file_path = data_path + filename
    if filename in available_files:
        try:
            globals()[df_name] = pd.read_csv(file_path, low_memory=False)
            print(f"Loaded {filename}: shape {globals()[df_name].shape}")
        except Exception as e:
            print(f"Error loading {filename}: {e}. Skipping this file.")
            globals()[df_name] = None
    else:
        print(f"Skipping {filename}: File not found in data_path. Setting to None.")
        globals()[df_name] = None

if tags is None or tags.empty:
    print("Skipping Tag-Level Trend Mining: 'tags' DataFrame is not loaded or empty.")
    # Set tag_names to empty dict if tags not available
    tag_names = {}

if competitions is None or competitions.empty or 'Year' not in competitions.columns:
    print("Skipping Tag-Level Trend Mining for Competitions: 'competitions' DataFrame is not loaded, empty, or missing 'Year' column.")
if kernels is None or kernels.empty or 'Year' not in kernels.columns:
    print("Skipping Tag-Level Trend Mining for Kernels: 'kernels' DataFrame is not loaded, empty, or missing 'Year' column.")


# --- Function to analyze tags for a given entity (Competitions or Kernels) ---
def analyze_tag_trends(entity_df, entity_tags_df, entity_name, tag_names_map):
    if entity_df is None or entity_df.empty or \
       entity_tags_df is None or entity_tags_df.empty or \
       not tag_names_map:
        print(f"Cannot analyze {entity_name} tag trends: Missing required data.")
        return

    print(f"\nAnalyzing {entity_name} Tag Trends...")

    # Ensure 'Year' column is datetime converted in entity_df
    if 'CreationDate' in entity_df.columns and pd.api.types.is_datetime64_any_dtype(entity_df['CreationDate']):
        entity_df['Year'] = entity_df['CreationDate'].dt.year
    elif 'DeadlineDate' in entity_df.columns and pd.api.types.is_datetime64_any_dtype(entity_df['DeadlineDate']):
        entity_df['Year'] = entity_df['DeadlineDate'].dt.year
    else:
        print(f"Warning: No valid date column found for 'Year' in {entity_name}. Skipping tag trend analysis by year.")
        return # Cannot proceed with time-based analysis

    # Merge entity (competition/kernel) data with their tags
    # Assuming 'entity_df' has an 'Id' column (e.g., CompetitionId or KernelId)
    # And 'entity_tags_df' has 'EntityId' (e.g., CompetitionId or KernelId) and 'TagId'
    merged_tags = entity_tags_df.merge(entity_df[['Id', 'Year']],
                                      left_on='Id', right_on='Id', how='inner') # Assuming 'Id' in tags df actually means entity_id

    # Rename column if necessary (e.g., Id_x might be entity_tags.Id, Id_y might be entity_df.Id)
    if 'Id_x' in merged_tags.columns and 'Id_y' in merged_tags.columns:
        merged_tags.rename(columns={'Id_x': 'TagId', 'Id_y': 'EntityId'}, inplace=True) # Adjust based on actual merge output structure
    elif 'Id' in merged_tags.columns and 'TagId' in merged_tags.columns: # If Id is already entity ID and TagId is separate
        pass # No rename needed if 'Id' is entity_id

    # Map TagId to TagName
    if 'TagId' in merged_tags.columns and tag_names_map:
        merged_tags['TagName'] = merged_tags['TagId'].map(tag_names_map).fillna('Unknown')
    else:
        print(f"Warning: TagId or tag_names_map missing for {entity_name} tag trends.")
        return

    # Count tags per year
    tags_per_year = merged_tags.groupby(['Year', 'TagName']).size().unstack(fill_value=0)

    if tags_per_year.empty:
        print(f"No tag trend data found for {entity_name}.")
        return

    # Calculate overall tag popularity (for top N selection)
    overall_tag_popularity = merged_tags['TagName'].value_counts()

    # Identify top 10 tags
    top_10_tags = overall_tag_popularity.head(10).index.tolist()

    # Plotting: Top 10 Tags Over Time
    if top_10_tags and not tags_per_year.empty:
        # Filter tags_per_year for only top 10 tags that actually exist in columns
        plot_tags = [tag for tag in top_10_tags if tag in tags_per_year.columns]
        if plot_tags:
            plt.figure(figsize=(14, 8))
            tags_per_year[plot_tags].plot(kind='line', marker='o', figsize=(14, 8))
            plt.title(f'Top 10 {entity_name} Tags Trend Over Time')
            plt.xlabel('Year')
            plt.ylabel(f'Number of {entity_name}s')
            plt.legend(title='Tag', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            print(f"Insight: Line plot shows the popularity trajectory of the top {entity_name} tags, revealing which ones have sustained or grown their relevance.")
        else:
            print(f"No plottable top 10 tags for {entity_name}.")
    else:
        print(f"Skipping top 10 tags plot for {entity_name}: No top tags or data to plot.")

    # Identify trending tags (e.g., significant growth in last few years)
    # Simple approach: compare count in last 2 years vs previous 2 years
    current_year = tags_per_year.index.max()
    if current_year and current_year > tags_per_year.index.min() + 2: # Ensure enough years for comparison
        recent_years = [y for y in range(current_year - 2, current_year + 1) if y in tags_per_year.index]
        previous_years = [y for y in range(current_year - 4, current_year - 2) if y in tags_per_year.index]

        if recent_years and previous_years:
            recent_counts = tags_per_year.loc[recent_years].sum()
            previous_counts = tags_per_year.loc[previous_years].sum()

            growth = (recent_counts - previous_counts) / previous_counts.replace(0, np.nan) # Avoid div by zero
            growth = growth.sort_values(ascending=False).dropna()

            print(f"\nTop 10 Trending {entity_name} Tags (Growth in recent years):")
            print(growth.head(10))
            print(f"Insight: These tags represent areas of increasing interest, potentially indicating new research directions or industry adoption.")

            print(f"\nTop 10 Declining {entity_name} Tags:")
            print(growth.tail(10))
            print(f"Insight: These tags may represent mature or less active areas, or areas whose popularity has been superseded.")
        else:
            print(f"Not enough historical data to analyze trending/declining tags for {entity_name}.")
    else:
        print(f"Not enough historical data to analyze trending/declining tags for {entity_name}.")


# --- Execute the tag analysis for Competitions and Kernels ---
if competition_tags is not None and not competition_tags.empty and competitions is not None and not competitions.empty:
    # Ensure competitions has a date column for 'Year'
    if 'DeadlineDate' in competitions.columns:
        analyze_tag_trends(competitions[['Id', 'DeadlineDate']], competition_tags, "Competition", tag_names)
    else:
        print("Competitions DataFrame missing 'DeadlineDate' for tag trend analysis.")
else:
    print("Skipping Competition Tag Trends: Data not available.")

if kernel_tags is not None and not kernel_tags.empty and kernels is not None and not kernels.empty:
    # Ensure kernels has a date column for 'Year'
    if 'CreationDate' in kernels.columns:
        analyze_tag_trends(kernels[['Id', 'CreationDate']], kernel_tags, "Kernel", tag_names)
    else:
        print("Kernels DataFrame missing 'CreationDate' for tag trend analysis.")
else:
    print("Skipping Kernel Tag Trends: Data not available.")


import warnings

# Suppress only FutureWarnings (like dtype casting)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Optionally, suppress all warnings (not recommended unless for presentation)
warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np # Ensure numpy is imported

print("\n--- Starting Phase 5: Modeling ---")
print("\n--- 5.1. Competition Popularity Prediction (Classification) ---")

if competitions is None or competitions.empty:
    print("Skipping Competition Popularity Prediction: 'competitions' DataFrame is not loaded or empty.")
else:
    # --- Debugging competitions date columns ---
    print("\n--- Debugging Competitions Date Columns (Pre-modeling) ---")
    if 'StartDate' in competitions.columns:
        print(f"Dtype of competitions['StartDate']: {competitions['StartDate'].dtype}")
        print(f"NaNs in competitions['StartDate']: {competitions['StartDate'].isnull().sum()}")
    if 'DeadlineDate' in competitions.columns:
        print(f"Dtype of competitions['DeadlineDate']: {competitions['DeadlineDate'].dtype}")
        print(f"NaNs in competitions['DeadlineDate']: {competitions['DeadlineDate'].isnull().sum()}")
    print("--------------------------------------------------")
    # --- End Debugging ---

    popularity_threshold = competitions['TotalCompetitors'].quantile(0.75)
    print(f"Defining 'popular' competitions as having > {popularity_threshold} competitors.")

    model_df = competitions.copy()

    # Ensure RewardQuantity is numeric and handled for NaNs
    if 'RewardQuantity' in model_df.columns:
        model_df['RewardQuantity_numeric'] = pd.to_numeric(model_df['RewardQuantity'], errors='coerce').fillna(0)
    else:
        print("Warning: 'RewardQuantity' column not found. Adding as 0.")
        model_df['RewardQuantity_numeric'] = 0.0

    model_df['IsPopular'] = (model_df['TotalCompetitors'] > popularity_threshold).astype(int)

    # Feature Engineering for the model
    create_start_date_features = False
    if 'StartDate' in model_df.columns and pd.api.types.is_datetime64_any_dtype(model_df['StartDate']) and model_df['StartDate'].notna().any():
        model_df['StartMonth'] = model_df['StartDate'].dt.month
        model_df['StartDayOfWeek'] = model_df['StartDate'].dt.dayofweek
        create_start_date_features = True
    else:
        print("Warning: StartDate not available, not datetime dtype, or all NaNs. Skipping StartMonth/StartDayOfWeek.")
        model_df['StartMonth'] = np.nan # Ensure column exists
        model_df['StartDayOfWeek'] = np.nan # Ensure column exists

    create_duration_feature = False
    if 'DeadlineDate' in model_df.columns and 'StartDate' in model_df.columns and \
       pd.api.types.is_datetime64_any_dtype(model_df['DeadlineDate']) and \
       pd.api.types.is_datetime64_any_dtype(model_df['StartDate']) and \
       model_df['DeadlineDate'].notna().any() and model_df['StartDate'].notna().any():
        model_df['CompetitionDurationDays'] = (model_df['DeadlineDate'] - model_df['StartDate']).dt.days
        create_duration_feature = True
    else:
        print("Warning: Date columns not fully valid for duration calculation. Skipping CompetitionDurationDays.")
        model_df['CompetitionDurationDays'] = np.nan # Ensure column exists

    # Handle 'RewardType' and 'HostSegmentTitle' (categorical features)
    # Ensure these are handled before feature selection
    if 'RewardType' in model_df.columns:
        model_df['RewardType'] = model_df['RewardType'].astype(str).fillna('Missing_Category_Reward')
        le_reward = LabelEncoder()
        model_df['RewardType_encoded'] = le_reward.fit_transform(model_df['RewardType'])
        print("Encoded 'RewardType'.")
    else:
        print("Warning: 'RewardType' column not found for modeling. Skipping encoding.")
        model_df['RewardType_encoded'] = np.nan # Placeholder

    if 'HostSegmentTitle' in model_df.columns:
        model_df['HostSegmentTitle'] = model_df['HostSegmentTitle'].astype(str).fillna('Missing_Category_Host')
        le_host = LabelEncoder()
        model_df['HostSegmentTitle_encoded'] = le_host.fit_transform(model_df['HostSegmentTitle'])
        print("Encoded 'HostSegmentTitle'.")
    else:
        print("Warning: 'HostSegmentTitle' column not found for modeling. Skipping encoding.")
        model_df['HostSegmentTitle_encoded'] = np.nan # Placeholder


    # --- Dynamic Feature Selection ---
    # Start with robust features that are less likely to be all NaN
    features = [
        'MaxDailySubmissions', 'MaxTeamSize', 'BanTeamMergers', 'EnableTeamModels',
        'NumPrizes', 'UserRankMultiplier', 'CanQualifyTiers', 'RewardQuantity_numeric' # Added RewardQuantity
    ]

    # Add date-derived features only if they were successfully created
    if create_start_date_features:
        features.extend(['StartMonth', 'StartDayOfWeek'])
    if create_duration_feature:
        features.append('CompetitionDurationDays')
    
    # Add encoded categorical features only if they were successfully created
    if 'RewardType_encoded' in model_df.columns:
        features.append('RewardType_encoded')
    if 'HostSegmentTitle_encoded' in model_df.columns:
        features.append('HostSegmentTitle_encoded')

    # Final filter: ensure all selected features actually exist in the DataFrame columns
    # and contain at least one non-NaN value (after initial imputation if any)
    final_features = []
    for f in features:
        if f in model_df.columns:
            # Check if column is entirely NaN before adding it as a feature
            if model_df[f].notna().any():
                final_features.append(f)
            else:
                print(f"Warning: Feature '{f}' is entirely NaN after initial processing. Skipping this feature.")
        else:
            print(f"Warning: Feature '{f}' not found in model_df. Skipping this feature.")

    if not final_features: # If no features left to model
        print("Skipping modeling: No valid, non-NaN features available after selection.")
        # Removed the 'return' statement here. The code will now proceed to the next checks.
    else:
        # Impute remaining missing values for numerical features within final_features
        # This is crucial for columns that might have some NaNs, but not entirely NaNs
        numerical_features_to_impute = [f for f in final_features if pd.api.types.is_numeric_dtype(model_df[f])]
        
        if numerical_features_to_impute:
            imputer = SimpleImputer(strategy='mean')
            # Use .loc to avoid SettingWithCopyWarning, applying imputation only to the selected columns
            model_df.loc[:, numerical_features_to_impute] = imputer.fit_transform(model_df[numerical_features_to_impute])
            print(f"Imputed missing values for numerical features: {numerical_features_to_impute}")

        # Final DataFrame for modeling: ensure only the selected features and target are present, and no NaNs remain
        # At this point, all selected features should have been imputed or already non-NaN.
        # The .dropna() here should mostly be a sanity check, resulting in few (if any) dropped rows.
        initial_rows_before_final_drop = model_df.shape[0]
        model_df_cleaned = model_df[final_features + ['IsPopular']].dropna() # Drop any remaining NaNs in selected features or target
        
        if model_df_cleaned.shape[0] < initial_rows_before_final_drop:
            dropped_rows_count = initial_rows_before_final_drop - model_df_cleaned.shape[0]
            print(f"Dropped {dropped_rows_count} rows from model_df due to remaining NaNs before final split.")

        if model_df_cleaned.empty:
            print("Skipping modeling: No complete data rows for selected features after final cleaning.")
        else:
            X = model_df_cleaned[final_features]
            y = model_df_cleaned['IsPopular']

            if y.nunique() < 2:
                print(f"Skipping modeling: Target variable 'IsPopular' has only {y.nunique()} unique class(es) for classification. Need at least 2.")
            else:
                # Check for sufficient samples per class before stratify
                if y.value_counts().min() < 2: # Min samples per class for stratify
                    print("Warning: Insufficient samples for some classes to perform stratified split. Using non-stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                print("\n--- Model Evaluation ---")
                print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))

                if hasattr(model, 'feature_importances_'):
                    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    plt.figure(figsize=(10, 7))
                    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='coolwarm')
                    plt.title('Feature Importances for Competition Popularity Prediction')
                    plt.xlabel('Importance Score')
                    plt.ylabel('Feature')
                    plt.tight_layout()
                    plt.show()
                    print("Insight: Feature importances reveal which competition characteristics are most indicative of popularity. E.g., 'MaxTeamSize' or 'CompetitionDurationDays' might be key drivers.")
                else:
                    print("Skipping Feature Importance plot: Model does not have feature_importances_ attribute.")

