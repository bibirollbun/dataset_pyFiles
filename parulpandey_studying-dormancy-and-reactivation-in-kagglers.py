# Importing necessary libraries
import os
import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import textwrap
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

import warnings
warnings.filterwarnings("ignore")



# Setting Up Paths and Plotting Styles

BASE_PATH = '/kaggle/input/meta-kaggle/'

#The current date for analysis 
current_date = dt.datetime(2025, 7, 15)

# Set plotting style for consistent visuals
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size



# Define dtypes for Users.csv for memory optimization and select columns
users_optimized_dtypes = {
    'Id': 'int64',
    'UserName': 'string',
    'DisplayName': 'string',
    'RegisterDate': 'string',
    'PerformanceTier': 'Int8',
}
columns_to_load_users = list(users_optimized_dtypes.keys())

# 1. Load Users.csv
users = pd.read_csv(os.path.join(BASE_PATH, "Users.csv"),
                       dtype=users_optimized_dtypes,
                       usecols=columns_to_load_users)

# Convert RegisterDate to datetime
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')
users.dropna(subset=['RegisterDate'], inplace=True)

# Excluding the Kaggle team from the analysis i.e Performance Tier = 5
users = users[users['PerformanceTier'] != 5].copy()
users.head()



# Preprocessing
users['RegistrationYear'] = users['RegisterDate'].dt.to_period('Y').dt.to_timestamp()
yearly_registrations = (
    users[users['RegistrationYear'].dt.year <= 2024]
    .groupby('RegistrationYear')
    .size()
    .reset_index(name='count')
)
yearly_registrations['cumulative'] = yearly_registrations['count'].cumsum()

# Formatter
def format_millions(x, _):
    return f'{x / 1e6:.1f}M'

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Yearly Registrations
bars = ax1.bar(
    yearly_registrations['RegistrationYear'],
    yearly_registrations['count'],
    color='skyblue',
    width=250
)
ax1.set(
    title='Yearly User Registrations',
    xlabel='Year',
    ylabel='Users Registered (Millions)'
)
ax1.yaxis.set_major_formatter(FuncFormatter(format_millions))
ax1.tick_params(axis='x', rotation=45, labelsize=9)
ax1.tick_params(axis='y', labelsize=9)
ax1.yaxis.grid(True, linestyle='--', alpha=0.6)

for bar in bars:
    y = bar.get_height()
    ax1.annotate(
        f'{y / 1e6:.1f}M',
        xy=(bar.get_x() + bar.get_width() / 2, y),
        xytext=(0, 4),
        textcoords="offset points",
        ha='center',
        fontsize=8
    )

# Right: Cumulative Registrations
ax2.plot(
    yearly_registrations['RegistrationYear'],
    yearly_registrations['cumulative'],
    marker='o',
    color='skyblue',
    linewidth=2
)
ax2.set(
    title='Cumulative User Registrations',
    xlabel='Year',
    ylabel='Total Users (Millions)'
)
ax2.yaxis.set_major_formatter(FuncFormatter(format_millions))
ax2.tick_params(axis='x', rotation=45, labelsize=9)
ax2.tick_params(axis='y', labelsize=9)
ax2.yaxis.grid(True, linestyle='--', alpha=0.6)

# Final annotation
final_row = yearly_registrations.iloc[-1]
ax2.annotate(
    f'{final_row["cumulative"] / 1e6:.1f}M',
    xy=(final_row['RegistrationYear'], final_row['cumulative']),
    xytext=(0, 5),
    textcoords='offset points',
    ha='center',
    fontsize=8
)

plt.tight_layout()
plt.show()



counts = users['PerformanceTier'].value_counts().sort_index()
total = counts.sum()

tier_table = pd.DataFrame({
    'Users (Raw Count)': counts,

})

# Count and reindex in reverse order
counts = users['PerformanceTier'].value_counts().sort_index()
tier_order = [4, 3, 2, 1]
counts = counts.reindex(tier_order)

# Create DataFrame
tier_table = pd.DataFrame({
    'Users (Raw Count)': counts
})

# Compute max_count based on tiers 1–4 to avoid Tier 0 overpowering the gradient
max_count = counts.loc[[1, 2, 3, 4]].max() * 1.1

data = tier_table['Users (Raw Count)'].to_dict()


# Tier data
labels = {
    4: 'Grandmaster',
    3: 'Master',
    2: 'Expert',
    1: 'Kaggler'
}

# Light pastel tier colors
tier_colors = {
    4: '#FFF9C4',  # light gold
    3: '#FFCDD2',  # soft red
    2: '#E1BEE7',  # lavender
    1: '#BBDEFB'   # sky blue
}

# Build DataFrame
df = pd.DataFrame({
    'Tier': list(data.keys()),
    'Label': [labels[t] for t in data.keys()],
    'Users': list(data.values())
})

# Create HTML rows
html_rows = ""
for _, row in df.iterrows():
    bg = tier_colors[row['Tier']]
    html_rows += f"""
    <tr style="background-color:{bg}; color:#000;">
        <td style="padding:12px;text-align:center;font-weight:600;">{row['Tier']}</td>
        <td style="padding:12px;font-weight:600;">{row['Label']}</td>
        <td style="padding:12px;text-align:right;font-variant-numeric: tabular-nums;">{row['Users']:,}</td>
    </tr>
    """

# Final HTML
html_table = f"""
<table style="border-collapse:collapse;font-family:'Segoe UI', sans-serif;font-size:15px;border:1px solid #ccc;border-radius:6px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
    <thead>
        <tr style="background:#212121;color:white;">
            <th style="padding:12px;">Tier</th>
            <th style="padding:12px;">Label</th>
            <th style="padding:12px;">Users (Raw Count)</th>
        </tr>
    </thead>
    <tbody>
        {html_rows}
    </tbody>
</table>
"""

display(HTML(html_table))


users_df = users.copy()


# Load all other activity-related raw dataframes
kernels_df_raw = pd.read_csv(os.path.join(BASE_PATH, "Kernels.csv"))
kernel_versions_df_raw = pd.read_csv(os.path.join(BASE_PATH, "KernelVersions.csv"))
submissions_df_raw = pd.read_csv(os.path.join(BASE_PATH, "Submissions.csv"))
teams_df_raw = pd.read_csv(os.path.join(BASE_PATH, "Teams.csv"))
team_members_df_raw = pd.read_csv(os.path.join(BASE_PATH, "TeamMemberships.csv"))
forum_messages_df_raw = pd.read_csv(os.path.join(BASE_PATH, "ForumMessages.csv"))
datasets_df_raw = pd.read_csv(os.path.join(BASE_PATH, "Datasets.csv"))
dataset_versions_df_raw = pd.read_csv(os.path.join(BASE_PATH, "DatasetVersions.csv"))



# This list will hold standardized activity DataFrames
all_activities_list = [] 


# Kernels Activity 

kernels_activity = kernel_versions_df_raw.rename(columns={'AuthorUserId': 'UserId', 'CreationDate': 'ActivityDate'})
kernels_activity['ActivityDate'] = pd.to_datetime(kernels_activity['ActivityDate'], errors='coerce')
kernels_activity['Source'] = 'Kernel'
all_activities_list.append(kernels_activity[['UserId', 'ActivityDate', 'Source']].dropna())



# Submissions 

submissions_activity = pd.merge(submissions_df_raw, team_members_df_raw[['TeamId', 'UserId']], on='TeamId', how='inner')
submissions_activity = submissions_activity.rename(columns={'SubmissionDate': 'ActivityDate'})
submissions_activity['ActivityDate'] = pd.to_datetime(submissions_activity['ActivityDate'], errors='coerce')
submissions_activity['Source'] = 'Submission' 
all_activities_list.append(submissions_activity[['UserId', 'ActivityDate', 'Source']].dropna())



# Forum Messages Activity 
forum_activity = forum_messages_df_raw.rename(columns={'PostUserId': 'UserId', 'PostDate': 'ActivityDate'})
forum_activity['ActivityDate'] = pd.to_datetime(forum_activity['ActivityDate'], errors='coerce')
forum_activity['Source'] = 'Forum Post' 
all_activities_list.append(forum_activity[['UserId', 'ActivityDate', 'Source']].dropna())



# Datasets Activity

datasets_activity = pd.merge(dataset_versions_df_raw, datasets_df_raw[['Id', 'CreatorUserId']], left_on='DatasetId', right_on='Id', how='inner', suffixes=('_ver', '_ds'))
datasets_activity = datasets_activity.rename(columns={'CreatorUserId_ds': 'UserId', 'CreationDate': 'ActivityDate'})
datasets_activity['ActivityDate'] = pd.to_datetime(datasets_activity['ActivityDate'], errors='coerce')
datasets_activity['Source'] = 'Dataset' 



all_activities_list.append(datasets_activity[['UserId', 'ActivityDate', 'Source']].dropna())



# Prepare first activity summary
user_first_activity_summary = (
    pd.concat(all_activities_list)
      .sort_values(['UserId', 'ActivityDate'])
      .drop_duplicates('UserId', keep='first')
      .rename(columns={'UserId': 'Id', 'ActivityDate': 'FirstActivityDate', 'Source': 'FirstActivityType'})
      [['Id', 'FirstActivityDate', 'FirstActivityType']]
)

# Prepare last activity summary
user_last_activity_summary = (
    pd.concat(all_activities_list)
      .groupby('UserId', as_index=False)['ActivityDate']
      .max()
      .rename(columns={'UserId': 'Id', 'ActivityDate': 'LastActivityDate'})
)

# Merge with main user data
users_df = (
    users_df
      .merge(user_first_activity_summary, on='Id', how='left')
      .merge(user_last_activity_summary, on='Id', how='left')
)



# Submission counts
user_submission_counts = (
    submissions_df_raw
      .merge(team_members_df_raw[['TeamId', 'UserId']], on='TeamId', how='inner')
      .groupby('UserId')
      .size()
      .reset_index(name='SubmissionCount')
      .rename(columns={'UserId': 'Id'})
)

# Kernel counts
user_kernel_counts = (
    kernel_versions_df_raw[['AuthorUserId', 'ScriptId']]
      .drop_duplicates()
      .groupby('AuthorUserId')
      .size()
      .reset_index(name='KernelCount')
      .rename(columns={'AuthorUserId': 'Id'})
)

# Forum message counts
user_forum_message_counts = (
    forum_activity
      .groupby('UserId')
      .size()
      .reset_index(name='ForumMessageCount')
      .rename(columns={'UserId': 'Id'})
)

# Dataset counts
user_dataset_counts = (
    datasets_activity
      .groupby('UserId')
      .size()
      .reset_index(name='DatasetCount')
      .rename(columns={'UserId': 'Id'})
)

# Merge all activity counts into users_df
users_df = (
    users_df
      .merge(user_submission_counts, on='Id', how='left')
      .merge(user_kernel_counts, on='Id', how='left')
      .merge(user_forum_message_counts, on='Id', how='left')
      .merge(user_dataset_counts, on='Id', how='left')
)

# Fill NaNs with 0 and set appropriate types
users_df['SubmissionCount'] = users_df['SubmissionCount'].fillna(0).astype('Int64')
users_df['KernelCount'] = users_df['KernelCount'].fillna(0).astype('Int64')
users_df['ForumMessageCount'] = users_df['ForumMessageCount'].fillna(0).astype('Int64')
users_df['DatasetCount'] = users_df['DatasetCount'].fillna(0).astype('Int64')



# Save the final, enriched dataframe to a new CSV file for easy reuse.
output_filename = "Enriched_Users_Data.csv"
users_df.to_csv(output_filename, index=False)

print(f"Enriched dataset successfully saved to '{output_filename}'.")
users_df.info()


users_df = pd.read_csv('Enriched_Users_Data.csv',index_col=0)


tier1_df = (
    users_df[users_df['PerformanceTier'] == 1]
      .assign(
          RegisterDate=pd.to_datetime(users_df['RegisterDate']),
          FirstActivityDate=pd.to_datetime(users_df['FirstActivityDate']),
          TimeSinceRegistration_Days=(current_date - pd.to_datetime(users_df['RegisterDate'])).dt.days
      )
      .copy()
)




def get_tenure_category(days_since_reg):
    if days_since_reg < 30:
        return "Short-Term (<1 month)"
    elif days_since_reg < 365:
        return "Middle-Term (1 month–1 year)"
    else:
        return "Long-Term (>1 year)"
tier1_df['KagglerTenureCategory'] = tier1_df['TimeSinceRegistration_Days'].apply(get_tenure_category)



# Define order once and compute counts
ordered_labels = ["Short-Term", "Middle-Term", "Long-Term"]
counts = tier1_df['KagglerTenureCategory'].value_counts()
counts_million = (counts / 1_000_000).round(2)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=counts_million.index, y=counts_million.values, color='skyblue', width=0.4)

plt.title('Categorized Distribution of Kaggler Tenure on Kaggle', fontsize=13)
plt.xlabel('Kaggler Tenure Category', fontsize=11)
plt.ylabel('Users (in Millions)', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Label active/inactive users
tier1_df['ActivityStatus'] = np.where(
    tier1_df['FirstActivityDate'].isna(),
    "Inactive\n(Never Active)",
    "Active\n(At Least One Activity)"
)

# Compute and order counts
counts = tier1_df['ActivityStatus'].value_counts()
counts = counts.reindex(["Active\n(At Least One Activity)", "Inactive\n(Never Active)"]).fillna(0)

# Plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x=counts.index,
    y=counts.values,
    palette=["skyblue", "lightcoral"],
    width=0.2
)

# Format y-axis and annotate
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}M'))
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f'{height / 1e6:.1f}M',
                (bar.get_x() + bar.get_width() / 2, height),
                textcoords="offset points", xytext=(0, 4),
                ha='center', va='bottom', fontsize=9)

# Labels and style
ax.set(title='Active vs. Inactive (Tier 1 Users)',
       xlabel='Status',
       ylabel='Number (in Millions)')
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



active_df = tier1_df[tier1_df['ActivityStatus'] == 'Active\n(At Least One Activity)'].copy()
active_df.head()


# Binary flags for user activity
activity_flags = (active_df[['SubmissionCount', 'KernelCount', 'DatasetCount', 'ForumMessageCount']] > 0)

# Count users per activity
activity_counts = activity_flags.sum().sort_values(ascending=False)

# Label mapping for cleaner x-axis
labels = {
    'SubmissionCount': 'Submission\nCount',
    'KernelCount': 'Kernel\nCount',
    'DatasetCount': 'Dataset\nCount',
    'ForumMessageCount': 'Forum\nMessages'
}
activity_counts.index = [labels[col] for col in activity_counts.index]

# Plot
plt.figure(figsize=(7, 5))
sns.barplot(
    x=activity_counts.index,
    y=activity_counts.values,
    palette=sns.light_palette("skyblue", n_colors=len(activity_counts), reverse=True),
    width = 0.45
)

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}M'))

plt.title('Unique Active Users by Activity Type', fontsize=13)
plt.ylabel('Number of Users', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()




# Calculate the time delta between registration and first activity
active_df['InitialDormancyPeriod'] = active_df['FirstActivityDate'] - active_df['RegisterDate']

def get_initial_dormancy_category(row):
    """Categorizes the time it takes for a user to perform their first activity."""
    if pd.isna(row['FirstActivityDate']): return "Never Active"
    days = row['InitialDormancyPeriod'].days
    if days < 0: return "Data Error" # Should not happen in active_df
    elif days == 0: return "0 Days"
    elif days <= 7: return "1-7 Days"
    elif days <= 30: return "8-30 Days"
    elif days <= 180: return "1-6 Months"
    elif days <= 365: return "6 Months-1 Year"
    else: return ">1 Year"

# Apply the function to create the dormancy category column
active_df['DormancyCategory'] = active_df.apply(get_initial_dormancy_category, axis=1)




# Define category order
category_order = [
    "0 Days",
    "1-7 Days",
    "8-30 Days",
    "1-6 Months",
    "6 Months-1 Year",
    ">1 Year"
]

# Count and reorder
dormancy_counts = active_df['DormancyCategory'].value_counts().reindex(category_order).fillna(0)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(
    x=dormancy_counts.index,
    y=dormancy_counts.values,
    color='skyblue',
    width=0.45
)

# Format y-axis to show millions
def millions(x, _):
    return f'{x*1e-6:.1f}M'

plt.gca().yaxis.set_major_formatter(FuncFormatter(millions))

plt.title('Initial Dormancy Distribution (Active Users)', fontsize=13)
plt.xlabel('Time to First Activity', fontsize=11)
plt.ylabel('User Count', fontsize=11)
plt.xticks(rotation=0, fontsize=9, ha='center', wrap=True)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()



date_cols = ['RegisterDate', 'RegistrationYear', 'FirstActivityDate', 'LastActivityDate']

for col in date_cols:
    active_df[col] = pd.to_datetime(users_df[col], errors='coerce')

# Filter to long-registered users
long_term_users_df = active_df[active_df['RegisterDate'] <= current_date - pd.DateOffset(years=1)]

kernel_versions_df_raw = pd.read_csv("Data/KernelVersions.csv", parse_dates=['CreationDate'])




initiatives_df = pd.DataFrame({
    'InitiativeName': [
        'GenAI Cohort v1 (2024)',
        'GenAI Cohort v2 (2025)',
    ],
    'LaunchDate': pd.to_datetime([
        '2024-11-11', '2025-03-31'
    ]),
    'Type': [
        'Learning/Course',
        'Learning/Course',
    ]
})



# Parameters
MAX_POST_INITIATIVE_DAYS = 120
initiative_dates = initiatives_df['LaunchDate'].dt.normalize().to_list()

# Check if a user's first activity was within [launch, launch + window]
def was_nudged(row):
    first_activity_date = row['FirstActivityDate']
    if pd.isna(first_activity_date): return False
    return any(launch <= first_activity_date <= launch + pd.Timedelta(days=MAX_POST_INITIATIVE_DAYS)
               for launch in initiative_dates)

# Tag and filter the long-term user dataframe
long_term_users_df['InitiativeTriggered'] = long_term_users_df.apply(was_nudged, axis=1)
intervention_activated_users = long_term_users_df[long_term_users_df['InitiativeTriggered']].copy()

print(f"Total long-term Tier-1 users activated during known initiatives: {intervention_activated_users.shape[0]:,}")




# --- Set earliest date to display on x-axis ---
min_plot_date = pd.Timestamp("2024-10-01")

# --- Filter first activities to the relevant time window ---
fa_filtered = intervention_activated_users[
    intervention_activated_users['FirstActivityDate'] >= min_plot_date
]

# --- Count first activities by week ---
fa_counts = (
    fa_filtered['FirstActivityDate']
    .dt.to_period('W')
    .value_counts()
    .sort_index()
)

# --- Fill in missing weeks for a continuous timeline ---
full_weeks = pd.period_range(start=min_plot_date, end=fa_counts.index.max(), freq='W')
fa_counts = fa_counts.reindex(full_weeks, fill_value=0)

# --- Plot first activity counts with cohort launch markers ---
plt.figure(figsize=(12, 6)) # Made the plot slightly wider
fa_counts.plot(kind='line', marker='o', color='skyblue', label='First Activities of Long-Term Users')

for _, row in initiatives_df.iterrows():
    launch = row['LaunchDate']
    name = row['InitiativeName']
    if launch >= min_plot_date:
        plt.axvline(x=launch, color='orange', linestyle='--', linewidth=1.5)
        plt.text(
            x=launch + pd.Timedelta(days=3), # Small offset for readability
            y=plt.ylim()[1] * 0.9,
            s=name,
            rotation=90,
            va='top',
            ha='left', # Adjusted alignment
            fontsize=10,
            color='darkorange'
        )

plt.title("First-Time Activity of Long-Term Users vs. GenAI Launches", fontsize=14)
plt.xlabel("Week", fontsize=12)
plt.ylabel("Number of First Activities", fontsize=12)
plt.xticks(rotation=45)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.legend()
plt.show()



# Get kernel activity for reactivated users

intervention_activated_users = intervention_activated_users.reset_index()
user_kernels_after_activation = pd.merge(
    kernel_versions_df_raw,
    intervention_activated_users[['Id', 'FirstActivityDate','RegisterDate']],
    left_on='AuthorUserId', right_on='Id',
    how='inner',
    suffixes=('_kernel', '_user')
)



# Define keywords based on the GenAI course syllabus
genai_keywords = ['Prompting', 'Embeddings', 'Vector Stores', 'Vector Databases',
                  'Generative Agents', 'Agents','Generative AI','Lesson','Day','Generative','Capstone','Capstone_Project','GenAI']

def contains_genai_keyword(title):
    if pd.isna(title):
        return False
    title_lower = title.lower()
    return any(keyword.lower() in title_lower for keyword in genai_keywords)

user_kernels_after_activation['IsGenAITagged'] = user_kernels_after_activation['Title'].apply(contains_genai_keyword)

# Filter for notebooks that are tagged as GenAI-related
matched = user_kernels_after_activation[user_kernels_after_activation['IsGenAITagged']]





print(f"Found {len(matched):,} GenAI-tagged kernels from reactivated long-term users.")



# Ensure datetime format
matched['RegisterDate'] = pd.to_datetime(matched['RegisterDate'], errors='coerce')

# Count users per registration year
year_counts = matched['RegisterDate'].dt.year.value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=year_counts.index, y=year_counts.values, color='skyblue')
plt.title("Registration Year of Long-Dormant Users Reactivated by GenAI Course", fontsize=14)
plt.xlabel("Original Registration Year", fontsize=12)
plt.ylabel("Number of Reactivated Users", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





matched['Title'].value_counts()[:10]

