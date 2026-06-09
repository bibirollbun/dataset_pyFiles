import os
import glob
import json
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import IFrame
import kagglehub
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")
print("âœ… Downloaded Meta-Kaggle data.")
print("ğŸ“‚ MK_PATH =", MK_PATH)
print("ğŸ“‚ MKC_PATH =", MKC_PATH)


competitions = pl.read_csv(f"{MK_PATH}/Competitions.csv")
print("Competitions.csv Columns:", competitions.columns)
print(competitions.shape)
competitions.head()


hosts = (
    competitions
    .filter(pl.col("HostSegmentTitle").is_not_null() & (pl.col("HostSegmentTitle").str.strip_chars() != ""))
    .select("HostSegmentTitle")
    .unique()
    .sort("HostSegmentTitle")
)
host_list = hosts["HostSegmentTitle"].to_list()
print(f"ğŸŒ� Total Unique Hosts: {len(host_list)}\n")
for host in host_list:
    print(host)


host_counts = (
    competitions
    .filter(
        pl.col("HostSegmentTitle").is_not_null() & 
        (pl.col("HostSegmentTitle").str.strip_chars() != "")
    )
    .group_by("HostSegmentTitle")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
print("ğŸ“Š HostSegmentTitle Value Counts:\n")
print(host_counts)


competitions = pl.read_csv("/kaggle/input/meta-kaggle/Competitions.csv", try_parse_dates=True)


competitions = competitions.with_columns([
    pl.col("EnabledDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("EnabledDateParsed")
])
competitions = competitions.with_columns([
    pl.col("EnabledDateParsed").dt.year().alias("EnabledYear")
])


filtered = competitions.filter(
    pl.col("HostSegmentTitle").is_not_null() &
    (pl.col("HostSegmentTitle").str.strip_chars() != "")
)



grouped = (
    filtered
    .group_by(["EnabledYear", "HostSegmentTitle"])
    .agg(pl.len().alias("count"))
    .sort(["EnabledYear", "HostSegmentTitle"])
)
grouped


df_plot = grouped.to_pandas()


fig_comparison = go.Figure()
for host in df_plot["HostSegmentTitle"].unique():
    data = df_plot[df_plot["HostSegmentTitle"] == host]
    fig_comparison.add_trace(go.Scatter(
        x=data["EnabledYear"],
        y=data["count"],
        mode="lines+markers",
        name=host
    ))

fig_comparison.update_layout(
    title="Competitions per Host Segment per Year",
    xaxis_title="Year",
    yaxis_title="Count",
    template="plotly_white"
)
fig_comparison.write_html("Competitions_year.html")
IFrame("Competitions_year.html", width=1200, height=700)


RewardType = (
    competitions
    .filter(pl.col("RewardType").is_not_null() & (pl.col("RewardType").str.strip_chars() != ""))
    .select("RewardType")
    .unique()
    .sort("RewardType")
)
RewardType_list = RewardType["RewardType"].to_list()
print(f"ğŸŒ� Total Unique RewardType: {len(RewardType_list)}\n")
for res in RewardType_list:
    print(res)


RewardType_counts = (
    competitions
    .filter(pl.col("RewardType").is_not_null() & (pl.col("RewardType").str.strip_chars() != ""))
    .group_by("RewardType")
    .len()
    .sort("len", descending=True)
)
RewardType_counts_list = RewardType_counts.to_numpy().tolist()
print(f"ğŸŒ� Total Unique RewardType: {len(RewardType_counts_list)}\n")
for reward_type, count in RewardType_counts_list:
    print(f"{reward_type}: {count}")


filtered = competitions.filter(
    pl.col("RewardType").is_not_null() & 
    (pl.col("RewardType").str.strip_chars() != "")
)
host_reward_counts = (
    filtered
    .group_by(["HostSegmentTitle", "RewardType"])
    .len()
    .sort(["HostSegmentTitle", "len"], descending=[False, True])
)
print(host_reward_counts)


competitions = competitions.with_columns(
    pl.col("EnabledDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("EnabledDate_parsed")
).with_columns(
    pl.col("EnabledDate_parsed").dt.year().alias("Year")
)


usd_competitions = competitions.filter(
    (pl.col("RewardType") == "USD") &
    pl.col("Year").is_not_null() &
    pl.col("HostSegmentTitle").is_not_null()
)


usd_yearly_counts = (
    usd_competitions
    .group_by(["Year", "HostSegmentTitle"])
    .len()
    .sort(["Year", "HostSegmentTitle"])
)


pivoted = (
    usd_yearly_counts
    .pivot(values="len", index="Year", on="HostSegmentTitle")
    .fill_null(0)
    .sort("Year")
)


pdf = pivoted.to_pandas()
df_melted = pdf.melt(id_vars="Year", var_name="HostSegmentTitle", value_name="Count")
fig_comparison = px.line(
    df_melted,
    x="Year",
    y="Count",
    color="HostSegmentTitle",
    markers=True,
    title="ğŸ“ˆ USD Competitions Over Years by HostSegmentTitle",
    labels={"Count": "Number of Competitions", "Year": "Year"},
    width=1100,
    height=600
)
fig_comparison.write_html("all_compi_comparison.html")
IFrame("all_compi_comparison.html", width=1200, height=700)


competitions = competitions.with_columns(
    pl.col("EnabledDate")
    .str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False)
    .alias("EnabledDate_parsed")
).with_columns(
    pl.col("EnabledDate_parsed").dt.year().alias("Year")
)



usd_rewards = competitions.filter(
    (pl.col("RewardType") == "USD") &
    (pl.col("RewardQuantity").str.strip_chars().is_not_null()) &
    (pl.col("RewardQuantity").str.strip_chars() != "") &
    pl.col("Year").is_not_null()
).with_columns(
    pl.col("RewardQuantity").str.strip_chars().cast(pl.Float64)
)


usd_by_year = (
    usd_rewards
    .group_by("Year")
    .agg(pl.sum("RewardQuantity").alias("TotalUSDReward"))
    .sort("Year")
)


df_usd = usd_by_year.to_pandas()

fig_usd = px.line(
    df_usd,
    x="Year",
    y="TotalUSDReward",
    title="ğŸ’° Total USD Rewards on Kaggle Per Year",
    labels={"Year": "Year", "TotalUSDReward": "Total USD Reward"},
    markers=True,
    width=1000,
    height=500
)
fig_usd.write_html("usd_reward_by_year.html")
IFrame("usd_reward_by_year.html", width=1200, height=600)


Submissions = pl.read_csv("/kaggle/input/meta-kaggle/Submissions.csv")
print(Submissions.columns)
print(Submissions.shape)
Submissions.head()


Submissions = Submissions.with_columns(
    pl.col("SubmissionDate").str.to_datetime("%m/%d/%Y").alias("SubmissionDateDT")
)
submission_trend = Submissions.group_by(pl.col("SubmissionDateDT").dt.truncate("1mo").alias("Month")).agg(
    SubmissionCount=pl.col("Id").count()
).sort("Month")
plt.figure(figsize=(10, 6))
plt.plot(submission_trend["Month"], submission_trend["SubmissionCount"], label="Submission Count", color="blue")
plt.title("Submission Frequency Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Submissions")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


Submissions = pl.read_csv("/kaggle/input/meta-kaggle/Submissions.csv")
Submissions = Submissions.with_columns(pl.col("SubmissionDate").str.to_date("%m/%d/%Y"))
deadline_trends = Submissions.group_by([
    pl.col("SubmissionDate").dt.truncate("1w").alias("Week"),
    "IsAfterDeadline"
]).agg(
    pl.col("Id").count().alias("SubmissionCount")
).sort("Week")
deadline_pivot = deadline_trends.pivot(
    values="SubmissionCount",
    index="Week",
    on="IsAfterDeadline",
    aggregate_function="sum"
).fill_null(0)
deadline_pivot = deadline_pivot.rename({
    "true": "PostDeadline",
    "false": "PreDeadline"
})
fig = make_subplots()
fig.add_trace(
    go.Scatter(
        x=deadline_pivot["Week"],
        y=deadline_pivot["PreDeadline"],
        name="Pre-Deadline Submissions",
        line=dict(color="blue")
    )
)
fig.add_trace(
    go.Scatter(
        x=deadline_pivot["Week"],
        y=deadline_pivot["PostDeadline"],
        name="Post-Deadline Submissions",
        line=dict(color="red")
    )
)
fig.update_layout(
    title="Submission Trends: Pre- vs. Post-Deadline",
    xaxis_title="Week",
    yaxis_title="Number of Submissions",
    template="plotly",
    showlegend=True
)
pio.write_html(fig, file="deadline_behavior_plot.html", auto_open=False, include_plotlyjs="cdn")
display(IFrame("deadline_behavior_plot.html", width=1200, height=700))


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
from IPython.display import IFrame
import kagglehub
import os
import plotly.express as px
import polars as pl
from pathlib import Path
import json
import glob
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


Teams = pl.read_csv("/kaggle/input/meta-kaggle/Teams.csv")
print(Teams.columns)
print(Teams.shape)
Teams.head()


TeamMemberships = pl.read_csv("/kaggle/input/meta-kaggle/TeamMemberships.csv")
print(TeamMemberships.columns)
print(TeamMemberships.shape)
TeamMemberships.head()


Teams = pd.read_csv("/kaggle/input/meta-kaggle/Teams.csv")
TeamMemberships = pd.read_csv("/kaggle/input/meta-kaggle/TeamMemberships.csv")


def clean_dates(df, date_cols):
    for col in date_cols:
        df[col] = df[col].replace('', pd.NaT)
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


TeamMemberships = clean_dates(TeamMemberships, ['RequestDate'])
TeamMemberships['RequestYear'] = TeamMemberships['RequestDate'].dt.year


plt.figure(figsize=(12, 8))
team_formation = TeamMemberships.groupby('RequestYear').size().reset_index()
team_formation.columns = ['Year', 'Members']
team_formation = team_formation.dropna()

plt.plot(team_formation['Year'], team_formation['Members'], marker='s', linewidth=3, markersize=8, color='orange')
plt.title('Team Member Registrations by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('New Team Members', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
team_sizes = TeamMemberships.groupby(['TeamId', 'RequestYear']).size().reset_index()
team_sizes.columns = ['TeamId', 'Year', 'Size']
team_size_yearly = team_sizes.groupby('Year')['Size'].mean().reset_index()

plt.plot(team_size_yearly['Year'], team_size_yearly['Size'], marker='^', linewidth=3, markersize=8, color='purple')
plt.title('Average Team Size Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Team Size', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

