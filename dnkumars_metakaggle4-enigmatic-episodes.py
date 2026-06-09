import polars as pl
from IPython.display import IFrame
import kagglehub
import os
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt


MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")
print("âœ… Downloaded Meta-Kaggle data.")
print("ğŸ“‚ MK_PATH =", MK_PATH)
print("ğŸ“‚ MKC_PATH =", MKC_PATH)


Episodes = pl.read_csv("/kaggle/input/meta-kaggle/Episodes.csv")
print(Episodes.columns)
print(Episodes.shape)
Episodes.head()


unique_types = Episodes["Type"].unique().sort()
print(unique_types)


Competitions = pl.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")
print(Competitions.columns)
print(Competitions.shape)
Competitions.head()


unique_competition_ids = Episodes["CompetitionId"].unique().sort()
print("Unique CompetitionId values from Episodes:")
print(f"\nNumber of unique CompetitionId values in Episodes: {len(unique_competition_ids)}")
unique_competition_ids.to_pandas()


matching_competitions = Competitions.filter(pl.col("Id").is_in(unique_competition_ids))
print(f"Rows in Competitions.csv with Id matching CompetitionId in Episodes.csv:")
print(f"Shape: {matching_competitions.shape}")
print(f"\nNumber of matching competitions: {len(matching_competitions)}")
matching_competitions.to_pandas()


unique_HostSegmentTitle = matching_competitions["HostSegmentTitle"].unique().sort()
print(unique_HostSegmentTitle)
unique_host_segment_titles = matching_competitions["HostSegmentTitle"].value_counts().sort("HostSegmentTitle")
print("Unique HostSegmentTitle values and their counts:")
unique_host_segment_titles.to_pandas()


unique_OrganizationId = matching_competitions["OrganizationId"].unique().sort()
print(unique_OrganizationId)
unique_OrganizationId_titles = matching_competitions["OrganizationId"].value_counts().sort("OrganizationId")
print("Unique unique_OrganizationId values and their counts:")
unique_OrganizationId_titles.to_pandas()


Organizations = pl.read_csv("/kaggle/input/meta-kaggle/Organizations.csv")
print(Organizations.columns)
print(Organizations.shape)
Organizations.head()


target_ids = [1623, 3789, 4, 4926]
matching_organizations = Organizations.filter(pl.col("Id").is_in(target_ids))
print(f"Rows in Organizations.csv with Id in {target_ids}:")
print(f"Shape: {matching_organizations.shape}")


matching_organizations.to_pandas()


unique_competition_ids = Episodes["CompetitionId"].unique()
matching_competitions = Competitions.filter(pl.col("Id").is_in(unique_competition_ids))
print("Sample EnabledDate values:")
print(matching_competitions.select("EnabledDate").head())


matching_competitions = matching_competitions.with_columns(
    pl.col("EnabledDate")
    .str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False)
    .dt.year()
    .alias("Year")
)
matching_competitions = matching_competitions.filter(pl.col("Year").is_not_null())
competitions_per_year = matching_competitions.group_by("Year").agg(
    pl.col("Id").count().alias("CompetitionCount")
).sort("Year")
print("Competitions per Year Which Uses Episodes:")
print(competitions_per_year)


fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=competitions_per_year["Year"],
        y=competitions_per_year["CompetitionCount"],
        name="Competitions",
        marker_color="blue"
    )
)
fig.update_layout(
    title="Number of Competitions per Year for Episode",
    xaxis_title="Year",
    yaxis_title="Number of Competitions for Episode",
    template="plotly",
    xaxis=dict(tickmode="linear") 
)
pio.write_html(fig, file="competitions_per_year_plot.html", auto_open=False, include_plotlyjs="cdn")
display(IFrame("competitions_per_year_plot.html", width=1200, height=700))


# Episode Duration Trends
# Objective: Study how the duration of episodes (time between CreateTime and EndTime) changes over time.
# Data: Use CreateTime and EndTime from Episodes.csv.
# Analysis:
# Calculate episode duration as EndTime - CreateTime.
# Group by time intervals (e.g., monthly or yearly) and analyze average or median episode durations.
# Plot trends to see if episodes are processed faster or slower over time.


episodes = pl.read_csv("/kaggle/input/meta-kaggle/Episodes.csv")
episodes = episodes.with_columns([
    pl.col("CreateTime").str.to_datetime("%m/%d/%Y %H:%M:%S"),
    pl.col("EndTime").str.to_datetime("%m/%d/%Y %H:%M:%S")
])
episodes = episodes.with_columns(
    (pl.col("EndTime") - pl.col("CreateTime")).dt.total_seconds().alias("Duration")
)
episodes = episodes.filter(pl.col("Duration").is_not_null())


duration_trend = episodes.group_by(pl.col("CreateTime").dt.truncate("1mo").alias("Month")).agg(
    AvgDuration=pl.col("Duration").mean(),
    MedianDuration=pl.col("Duration").median()
).sort("Month")
plt.figure(figsize=(10, 6))
plt.plot(duration_trend["Month"], duration_trend["AvgDuration"], label="Average Duration (seconds)", color="blue")
plt.plot(duration_trend["Month"], duration_trend["MedianDuration"], label="Median Duration (seconds)", color="orange", linestyle="--")
plt.title("Episode Duration Trends Over Time")
plt.xlabel("Month")
plt.ylabel("Duration (seconds)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout() 
plt.show()


episode_trend = episodes.group_by(pl.col("CreateTime").dt.truncate("1mo").alias("Month")).agg(
    EpisodeCount=pl.col("Id").count()
).sort("Month")
plt.figure(figsize=(10, 6))
plt.plot(episode_trend["Month"], episode_trend["EpisodeCount"], label="Episode Count", color="blue")
plt.title("Episode Frequency Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Episodes")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import gc
gc.collect()


import sys
for name, size in sorted(((name, sys.getsizeof(obj)) for name, obj in globals().items()), key=lambda x: -x[1])[:10]:
    print(f"{name}: {size/1e6:.2f} MB")


for name in dir():
    if not name.startswith('_'):
        del globals()[name]
import gc
gc.collect()


import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import IFrame
import kagglehub
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# EpisodeAgents = pl.read_csv("/kaggle/input/meta-kaggle/EpisodeAgents.csv")
# print(EpisodeAgents.columns)
# print(EpisodeAgents.shape)
# EpisodeAgents.head()
# parquet_path = kagglehub.dataset_download("bwandowando/meta-kaggle-ported-to-parquet-format")
# #https://www.kaggle.com/datasets/bwandowando/meta-kaggle-ported-to-parquet-format/data


# print("Path to dataset files:", parquet_path)
EpisodeAgents = pl.read_parquet("/kaggle/input/meta-kaggle-ported-to-parquet-format/Parquet/EpisodeAgents.parquet")
print(EpisodeAgents.shape)
print(EpisodeAgents.columns)
EpisodeAgents.head()


Episodes = pl.read_csv("/kaggle/input/meta-kaggle/Episodes.csv")
print(Episodes.columns)
print(Episodes.shape)
Episodes.head()


unique_EpisodeId = EpisodeAgents["EpisodeId"].unique().sort()
len(unique_EpisodeId)


unique_SubmissionId = EpisodeAgents["SubmissionId"].unique().sort()
len(unique_SubmissionId)


unique_State = EpisodeAgents["State"].unique().sort()
unique_State


unique_Index = EpisodeAgents["Index"].unique().sort()
unique_Index


unique_Reward = EpisodeAgents["Reward"].unique().sort()
unique_Reward = EpisodeAgents["Reward"].unique()
range_Reward = unique_Reward.max() - unique_Reward.min()
print(f"Reward Range :{range_Reward}")
len(unique_Reward)


CompetitionId=21723


import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Episodes = pl.read_csv("/kaggle/input/meta-kaggle/Episodes.csv")
result = Episodes.filter(pl.col("CompetitionId") == 21723)
print(result.shape)
result.head()


# Episodes = pl.read_csv("/kaggle/input/meta-kaggle/Episodes.csv")
# EpisodeAgents = pl.read_parquet("/kaggle/input/meta-kaggle-ported-to-parquet-format/Parquet/EpisodeAgents.parquet")
result_episodes = Episodes.filter(pl.col("CompetitionId") == 21723)
episode_ids = result_episodes["Id"]
result_agents = EpisodeAgents.filter(pl.col("EpisodeId").is_in(episode_ids))
print(result_agents.shape)
result_agents.head()


del Episodes


del EpisodeAgents


episodes_pd = result_episodes.to_pandas()
agents_pd = result_agents.to_pandas()


episodes_pd['CreateTime'] = pd.to_datetime(episodes_pd['CreateTime'])
episodes_pd['EndTime'] = pd.to_datetime(episodes_pd['EndTime'])
episodes_pd['Duration'] = (episodes_pd['EndTime'] - episodes_pd['CreateTime']).dt.total_seconds() / 60
episodes_pd['Date'] = episodes_pd['CreateTime'].dt.date
episodes_pd['Hour'] = episodes_pd['CreateTime'].dt.hour
episodes_pd['DayOfWeek'] = episodes_pd['CreateTime'].dt.dayofweek
merged_df = agents_pd.merge(episodes_pd[['Id', 'CreateTime', 'EndTime', 'Duration', 'Date', 'Hour', 'DayOfWeek']], 
                           left_on='EpisodeId', right_on='Id', suffixes=('_agent', '_episode'))


plt.figure(figsize=(14, 6))
daily_counts = episodes_pd.groupby('Date').size().reset_index(name='count')
daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])

plt.fill_between(daily_counts['Date'], daily_counts['count'], alpha=0.7, color='steelblue')
plt.plot(daily_counts['Date'], daily_counts['count'], color='darkblue', linewidth=2)
plt.title('Episodes Count Over Time (Spine Plot)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Episodes', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 6))
merged_df_clean = merged_df.dropna(subset=['Reward'])
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
box_data = [merged_df_clean[merged_df_clean['DayOfWeek'] == i]['Reward'].values for i in range(7)]

bp = plt.boxplot(box_data, labels=days, patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, 7))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title('Agent Rewards Distribution by Day of Week', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


merged_with_type = merged_df.merge(episodes_pd[['Id', 'Type']], left_on='EpisodeId', right_on='Id', how='left')
score_data = merged_with_type.dropna(subset=['UpdatedScore'])
types = score_data['Type'].unique()


plt.figure(figsize=(14, 10))
merged_with_type['Week'] = pd.to_datetime(merged_with_type['CreateTime']).dt.isocalendar().week
weeks = sorted(merged_with_type['Week'].unique())[:10] 

for i, week in enumerate(weeks):
    week_data = merged_with_type[merged_with_type['Week'] == week]['UpdatedScore'].dropna()
    if len(week_data) > 0:
        density = np.histogram(week_data, bins=50, density=True)[0]
        bin_edges = np.histogram(week_data, bins=50)[1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        plt.fill_between(bin_centers, i, i + density * 0.5, alpha=0.7, color=plt.cm.viridis(i/len(weeks)))
        plt.plot(bin_centers, i + density * 0.5, color='black', linewidth=1)

plt.yticks(range(len(weeks)), [f'Week {w}' for w in weeks])
plt.title('Score Distribution Evolution Over Weeks', fontsize=16, fontweight='bold')
plt.xlabel('Updated Score', fontsize=12)
plt.ylabel('Week', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
daily_duration = episodes_pd.groupby('Date')['Duration'].mean().reset_index()
daily_duration['Date'] = pd.to_datetime(daily_duration['Date'])
daily_duration = daily_duration.sort_values('Date')

plt.step(daily_duration['Date'], daily_duration['Duration'], where='mid', linewidth=2, color='red')
plt.fill_between(daily_duration['Date'], daily_duration['Duration'], alpha=0.3, color='red', step='mid')
plt.title('Average Episode Duration Over Time (Step Plot)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Duration (minutes)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print("Time trends analysis complete with multiple plot types!")
print(f"Total episodes analyzed: {len(episodes_pd)}")
print(f"Total agent records: {len(agents_pd)}")
print(f"Date range: {episodes_pd['CreateTime'].min()} to {episodes_pd['CreateTime'].max()}")





Submissions = pl.read_csv("/kaggle/input/meta-kaggle/Submissions.csv")
print(Submissions.columns)
print(Submissions.shape)
Submissions.head()


episodes_pd = result_episodes.to_pandas()
agents_pd = result_agents.to_pandas()
submissions_filtered = Submissions.filter(pl.col("Id").is_in(agents_pd['SubmissionId'].dropna()))
submissions_pd = submissions_filtered.to_pandas()
episodes_pd['CreateTime'] = pd.to_datetime(episodes_pd['CreateTime'])
episodes_pd['EndTime'] = pd.to_datetime(episodes_pd['EndTime'])
episodes_pd['Duration'] = (episodes_pd['EndTime'] - episodes_pd['CreateTime']).dt.total_seconds() / 60
submissions_pd['SubmissionDate'] = pd.to_datetime(submissions_pd['SubmissionDate'])
submissions_pd['ScoreDate'] = pd.to_datetime(submissions_pd['ScoreDate'])
episodes_pd['Date'] = episodes_pd['CreateTime'].dt.date
episodes_pd['Week'] = episodes_pd['CreateTime'].dt.isocalendar().week
episodes_pd['Hour'] = episodes_pd['CreateTime'].dt.hour
episodes_pd['DayOfWeek'] = episodes_pd['CreateTime'].dt.dayofweek
submissions_pd['Date'] = submissions_pd['SubmissionDate'].dt.date
submissions_pd['Week'] = submissions_pd['SubmissionDate'].dt.isocalendar().week
submissions_pd['Hour'] = submissions_pd['SubmissionDate'].dt.hour


agents_with_submissions = agents_pd.merge(submissions_pd, left_on='SubmissionId', right_on='Id', suffixes=('_agent', '_submission'))
full_merged = agents_with_submissions.merge(episodes_pd[['Id', 'CreateTime', 'EndTime', 'Duration', 'Date', 'Hour', 'DayOfWeek', 'Type']], 
                                           left_on='EpisodeId', right_on='Id', suffixes=('', '_episode'))


plt.figure(figsize=(14, 6))
daily_episodes = episodes_pd.groupby('Date').size().reset_index(name='episodes')
daily_submissions = submissions_pd.groupby('Date').size().reset_index(name='submissions')
daily_combined = pd.merge(daily_episodes, daily_submissions, on='Date', how='outer').fillna(0)
daily_combined['Date'] = pd.to_datetime(daily_combined['Date'])

plt.fill_between(daily_combined['Date'], daily_combined['episodes'], alpha=0.7, color='steelblue', label='Episodes')
plt.fill_between(daily_combined['Date'], daily_combined['submissions'], alpha=0.5, color='orange', label='Submissions')
plt.plot(daily_combined['Date'], daily_combined['episodes'], color='darkblue', linewidth=2)
plt.plot(daily_combined['Date'], daily_combined['submissions'], color='darkorange', linewidth=2)
plt.title('Episodes vs Submissions Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
score_comparison = pd.DataFrame({
    'Public': full_merged['PublicScoreFullPrecision'].dropna(),
    'Private': full_merged['PrivateScoreFullPrecision'].dropna()
})

bp = plt.boxplot([score_comparison['Public'], score_comparison['Private']], 
                labels=['Public Score', 'Private Score'], patch_artist=True)
colors = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title('Public vs Private Score Distributions', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
score_by_type = full_merged.dropna(subset=['PublicScoreFullPrecision', 'Type'])
types = sorted(score_by_type['Type'].unique())
violin_data = [score_by_type[score_by_type['Type'] == t]['PublicScoreFullPrecision'].values for t in types]

parts = plt.violinplot(violin_data, positions=range(len(types)), showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

plt.xticks(range(len(types)), [f'Type {t}' for t in types])
plt.title('Submission Score Distribution by Episode Type', fontsize=16, fontweight='bold')
plt.xlabel('Episode Type', fontsize=12)
plt.ylabel('Public Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
daily_scores = full_merged.groupby('Date')['PublicScoreFullPrecision'].mean().reset_index()
daily_scores['Date'] = pd.to_datetime(daily_scores['Date'])
daily_scores = daily_scores.sort_values('Date').dropna()

plt.step(daily_scores['Date'], daily_scores['PublicScoreFullPrecision'], where='mid', linewidth=2, color='purple')
plt.fill_between(daily_scores['Date'], daily_scores['PublicScoreFullPrecision'], alpha=0.3, color='purple', step='mid')
plt.title('Average Public Score Trend Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Public Score', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print("Enhanced time trends analysis complete with Submissions data!")
print(f"Total episodes: {len(episodes_pd)}")
print(f"Total agents: {len(agents_pd)}")
print(f"Total submissions: {len(submissions_pd)}")
print(f"Merged records: {len(full_merged)}")


daily_max_scores = full_merged.groupby('Date')['PublicScoreFullPrecision'].max().reset_index()
daily_max_scores['Date'] = pd.to_datetime(daily_max_scores['Date'])
daily_max_scores = daily_max_scores.sort_values('Date').dropna()
weekly_max_scores = full_merged.groupby('Week')['PublicScoreFullPrecision'].max().reset_index()
weekly_max_scores = weekly_max_scores.sort_values('Week').dropna()



plt.figure(figsize=(14, 6))
plt.plot(daily_max_scores['Date'], daily_max_scores['PublicScoreFullPrecision'], color='darkblue', linewidth=2, marker='o')
plt.fill_between(daily_max_scores['Date'], daily_max_scores['PublicScoreFullPrecision'], alpha=0.3, color='steelblue')
plt.title('Highest Public Score Over Time (Daily)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Highest Public Score', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
score_by_day = full_merged.loc[full_merged.groupby('Date')['PublicScoreFullPrecision'].idxmax()]
score_by_day = score_by_day.dropna(subset=['PublicScoreFullPrecision', 'DayOfWeek'])
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
violin_data = [score_by_day[score_by_day['DayOfWeek'] == i]['PublicScoreFullPrecision'].values 
               for i in range(7) if len(score_by_day[score_by_day['DayOfWeek'] == i]) > 0]

if violin_data:
    parts = plt.violinplot(violin_data, positions=[i for i, d in enumerate(violin_data) if len(d) > 0], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    plt.xticks([i for i, d in enumerate(violin_data) if len(d) > 0], 
               [day_names[i] for i, d in enumerate(violin_data) if len(d) > 0])
else:
    plt.text(0.5, 0.5, 'No data available for violin plot', transform=plt.gca().transAxes, 
             ha='center', va='center', fontsize=14)

plt.title('Highest Public Score Distribution by Day of Week', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Highest Public Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print("Highest public score trend analysis complete!")
print(f"Total episodes: {len(episodes_pd)}")
print(f"Total agents: {len(agents_pd)}")
print(f"Total submissions: {len(submissions_pd)}")
print(f"Merged records: {len(full_merged)}")

