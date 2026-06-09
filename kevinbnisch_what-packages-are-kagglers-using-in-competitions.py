%pip install plotly[express] -q


import kagglehub
import os
import nbformat
import pandas as pd
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import re
import ast
import math
import numpy as np
import plotly.io as pio

from scipy.stats import gaussian_kde
from matplotlib import cm
from IPython.display import display, HTML
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy import array
from kagglehub import KaggleDatasetAdapter

import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 1000)
warnings.filterwarnings('ignore')
pio.renderers.default = 'iframe' #https://www.kaggle.com/discussions/product-announcements/549950


kernel_versions_df = pd.read_csv('/kaggle/input/kaggles-most-used-packages-and-method-calls/meta-kaggle-code-packages-methods_v2/meta-kaggle-code-packages-methods_v2.csv')
kernel_versions_df.head()


kernels_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "Kernels.csv",
)
kernels_df.head()


kernels_kernel_versions_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "KernelVersionKernelSources.csv",
)
kernels_kernel_versions_df.head()


kernel_versions_comp_sources_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "KernelVersionCompetitionSources.csv",
)
kernel_versions_comp_sources_df.head()


competitions_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "Competitions.csv",
)
display(competitions_df)


submissions_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "Submissions.csv",
)
display(submissions_df)


teams_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "Teams.csv",
)
display(teams_df)


if 'Id' in kernel_versions_comp_sources_df.columns:
    kernel_versions_comp_sources_df = kernel_versions_comp_sources_df.rename(columns={'Id': 'KernelVersionToCompetitionId'})
if 'Id' in kernels_kernel_versions_df.columns:
    kernels_kernel_versions_df = kernels_kernel_versions_df.rename(columns={'Id': 'KernelToKernelVersionId'})

# KernelVersionCompetitionSource.csv
df = kernel_versions_df.merge(right=kernel_versions_comp_sources_df, how='inner', left_on='Id', right_on='KernelVersionId')
print(len(df))

# Submissions.csv
df = df.merge(right=submissions_df, how='inner', left_on='Id', right_on='SourceKernelVersionId')

# Competitions.csv
competitions_df = competitions_df.rename(columns={'Id': 'CompetitionId', 'Title': 'CompetitionTitle'})
df = df.merge(right=competitions_df[
    ['CompetitionId', 'CompetitionTitle', 'Slug', 'EvaluationAlgorithmAbbreviation', 'TotalCompetitors', 
     'EnabledDate', 'DeadlineDate', 'HasLeaderboard', 'HostSegmentTitle', 'RewardQuantity']], 
    how='inner', left_on='SourceCompetitionId', right_on='CompetitionId')

df = df.drop(columns = ['Id_y', 'KernelVersionId', 'SourceKernelVersionId'])
df = df.rename(columns={'Id_x': 'Id', 'ScriptId': 'KernelId'})
print(len(df))
print('Unique kernel versions: ', len(df['Id'].unique()))
display(df)


for col in ['Imports', 'MethodCalls']:
    if col in df.columns:
        df[col] = df[col].dropna().apply(ast.literal_eval) 


top_n = 52


df["ScoreDate"] = pd.to_datetime(df["ScoreDate"], errors="coerce")
df["EnabledDate"] = pd.to_datetime(df["EnabledDate"], errors="coerce")
df["DeadlineDate"] = pd.to_datetime(df["DeadlineDate"], errors="coerce")
df["SubmissionDate"] = pd.to_datetime(df["EvaluationDate"], errors="coerce")  # proxy

# --- Filter Valid Competitions ---
df_filtered = df[
    (df["HasLeaderboard"] == True) &
    (df["HostSegmentTitle"].isin(["Featured", "Research", "Analytics"]))
].copy()

df_filtered = df_filtered.dropna(subset=[
    "ScoreDate", "CompetitionId", "EnabledDate", "DeadlineDate", "SubmissionDate"
])

df_filtered = df_filtered[
    (df_filtered["SubmissionDate"] >= df_filtered["EnabledDate"]) &
    (df_filtered["SubmissionDate"] <= df_filtered["DeadlineDate"])
].copy()

def plot_competitions(df_top, competition_meta, suptitle):
    ordered_ids = competition_meta["CompetitionId"].tolist()
    df_top["CompetitionId"] = pd.Categorical(df_top["CompetitionId"], categories=ordered_ids, ordered=True)
    df_top = df_top.sort_values("CompetitionId")
    
    grouped = df_top.groupby("CompetitionId", sort=False)
    meta_lookup = competition_meta.set_index("CompetitionId").to_dict("index")

    n = len(grouped)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharex=False)
    axes = axes.flatten()

    for i, (comp_id, group) in enumerate(grouped):
        ax = axes[i]
        group = group.sort_values("ScoreDate")

        sns.lineplot(data=group, x="ScoreDate", y="PublicScoreLeaderboardDisplay", label="Public", ax=ax)
        sns.lineplot(data=group, x="ScoreDate", y="PrivateScoreLeaderboardDisplay", label="Private", ax=ax)

        meta = meta_lookup[comp_id]
        line2 = f"{meta['EnabledDate'].date()} to {meta['DeadlineDate'].date()} | {meta['HostSegmentTitle']} | {meta['EvaluationAlgorithmAbbreviation']}"
        reward = meta['RewardQuantity']
        reward_str = f"{int(reward)}$" if pd.notnull(reward) else "n/a"
        line3 = f"{meta['TotalCompetitors']} competitors | {reward_str}"

        ax.set_title(f"{line2}\n{line3}", fontsize=9)
        ax.text(0.5, 1.22, meta["CompetitionTitle"], fontsize=12, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)

        ax.set_xlabel("Date")
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', labelrotation=30)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



competition_meta_date = (
    df_filtered[[
        "CompetitionId", "CompetitionTitle", "EnabledDate", "DeadlineDate",
        "HostSegmentTitle", "EvaluationAlgorithmAbbreviation", "Slug", "TotalCompetitors", "RewardQuantity"
    ]]
    .drop_duplicates("CompetitionId")
    .sort_values("EnabledDate", ascending=False)
    .head(top_n)
    .copy()
)
df_top_date = df_filtered[df_filtered["CompetitionId"].isin(competition_meta_date["CompetitionId"])].copy()

plot_competitions(df_top_date, competition_meta_date, f"Top {top_n} Competitions by Start Date")



# Step 1: Compute score differences, excluding flat public scores
grouped = df_filtered.groupby("CompetitionId")

score_diffs = []

for comp_id, group in grouped:
    valid = group.dropna(subset=["PublicScoreLeaderboardDisplay", "PrivateScoreLeaderboardDisplay"])

    # Skip competitions with too few or flat scores
    if (
        valid["PublicScoreLeaderboardDisplay"].nunique() <= 1 or
        valid["PrivateScoreLeaderboardDisplay"].nunique() <= 1 or
        valid.shape[0] < 50 
    ):
        continue

    pub_range = valid["PublicScoreLeaderboardDisplay"].max() - valid["PublicScoreLeaderboardDisplay"].min()
    if pub_range == 0:
        continue 

    valid["rel_diff"] = (valid["PublicScoreLeaderboardDisplay"] - valid["PrivateScoreLeaderboardDisplay"]).abs() / pub_range
    total_rel_diff = valid["rel_diff"].sum()

    score_diffs.append((comp_id, total_rel_diff))

score_diffs_df = (
    pd.DataFrame(score_diffs, columns=["CompetitionId", "abs_diff"])
    .sort_values("abs_diff", ascending=False)
    .head(top_n)
)

# Step 2: Get top n competition IDs
top_diff_ids = score_diffs_df["CompetitionId"].tolist()

# Step 3: Prepare metadata safely
competition_meta_diff = (
    df_filtered[df_filtered["CompetitionId"].isin(top_diff_ids)][[
        "CompetitionId", "CompetitionTitle", "EnabledDate", "DeadlineDate",
        "HostSegmentTitle", "EvaluationAlgorithmAbbreviation", "Slug",
        "TotalCompetitors", "RewardQuantity"
    ]]
    .drop_duplicates("CompetitionId")
    .copy()
)

# Step 4: Filter full data for those 30 competitions
df_top_diff = df_filtered[df_filtered["CompetitionId"].isin(top_diff_ids)].copy()

plot_competitions(df_top_diff, competition_meta_diff, f"Top {top_n} Competitions with Biggest Discrepency")


# Step 1: Calculate avg relative churn percentage per competition
churns = []

for comp_id, group in df_filtered.groupby("CompetitionId"):
    valid = group.dropna(subset=["PublicScoreLeaderboardDisplay", "PrivateScoreLeaderboardDisplay"])
    
    if (
        valid["PublicScoreLeaderboardDisplay"].nunique() <= 1 or
        valid["PrivateScoreLeaderboardDisplay"].nunique() <= 1 or
        valid.shape[0] < 50
    ):
        continue

    pub_range = valid["PublicScoreLeaderboardDisplay"].max() - valid["PublicScoreLeaderboardDisplay"].min()
    if pub_range == 0:
        continue 

    rel_diff = (valid["PublicScoreLeaderboardDisplay"] - valid["PrivateScoreLeaderboardDisplay"]).abs() / pub_range
    avg_rel_diff_pct = rel_diff.mean() * 100

    if avg_rel_diff_pct >= 100:
        continue 
    deadline = group["DeadlineDate"].dropna().iloc[0]
    title = group["CompetitionTitle"].dropna().iloc[0]
    churns.append((comp_id, avg_rel_diff_pct, deadline, title))

# Step 2: Create DataFrame
churn_df = pd.DataFrame(
    churns, 
    columns=["CompetitionId", "avg_rel_churn", "DeadlineDate", "CompetitionTitle"]
)
churn_df["DeadlineDate"] = pd.to_datetime(churn_df["DeadlineDate"])
churn_df["year_month"] = churn_df["DeadlineDate"].dt.to_period("M").dt.to_timestamp()

# Step 3: Aggregate discrepency per month
monthly_churn = churn_df.groupby("year_month")["avg_rel_churn"].mean().reset_index()

# Step 4: Plot dot chart + line
fig = go.Figure()

# Add competition-level dots
fig.add_trace(go.Scatter(
    x=churn_df["DeadlineDate"],
    y=churn_df["avg_rel_churn"],
    mode="markers",
    name="Competitions",
    marker=dict(size=6, opacity=0.6),
    hovertext=churn_df["CompetitionTitle"],
    hovertemplate=(
        "Date: %{x|%Y-%m-%d}<br>" +
        "Discrepency: %{y:.2f}%<br>" +
        "Title: %{hovertext}<extra></extra>"
    )
))

# Add monthly average churn line
fig.add_trace(go.Scatter(
    x=monthly_churn["year_month"],
    y=monthly_churn["avg_rel_churn"],
    mode="lines+markers",
    name="Monthly Average",
    line=dict(width=2)
))

# Final layout
fig.update_layout(
    title="Leaderboard Discrepency per Competition Over Time",
    xaxis_title="Competition Deadline",
    yaxis_title="Average Relative Discrepency (%)",
    yaxis_range=[0, 100],
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()



relevant_imports_df = pd.read_csv('/kaggle/input/ml-and-ai-related-imports/ML_AI_Related_Imports.csv')
display(relevant_imports_df)
relevant_imports = relevant_imports_df['ML/AI Imports'].to_list()
print(relevant_imports[:10])


df_filtered['EvaluationDate'] = pd.to_datetime(df_filtered['EvaluationDate'], errors='coerce')
df_filtered['YearMonth'] = df_filtered['EvaluationDate'].dt.to_period('M').astype(str)
df_exploded = df_filtered.explode('Imports').dropna(subset=['Imports'])
df_exploded['Imports_lower'] = df_exploded['Imports'].str.lower().str.split('.').str[0]


# Step 3: Filter out generic/data/vis packages
category_keywords_set = set(relevant_imports)
def is_generic_or_subpackage(imp):
    return (
        imp in category_keywords_set or
        any(imp.startswith(keyword + ".") for keyword in category_keywords_set)
    )

df_exploded = df_exploded[df_exploded['Imports_lower'].apply(is_generic_or_subpackage)]


top_n = 50

# Step 1: Count total usage of each import (across all months)
overall_counts = (
    df_exploded
    .groupby('Imports_lower')
    .size()
    .reset_index(name='TotalCount')
    .sort_values('TotalCount', ascending=False)
)

# Step 2: Select top N imports overall
top_imports = overall_counts.head(top_n)['Imports_lower'].tolist()

# Step 3: Filter monthly data to include only those top imports
monthly_counts = (
    df_exploded[df_exploded['Imports_lower'].isin(top_imports)]
    .groupby(['YearMonth', 'Imports_lower'])
    .size()
    .reset_index(name='ImportCount')
)

# Step 4: Normalize counts per month
monthly_totals = (
    monthly_counts
    .groupby('YearMonth')['ImportCount']
    .sum()
    .reset_index(name='Total')
)

df_plot = monthly_counts.merge(monthly_totals, on='YearMonth')
df_plot['Percentage'] = df_plot['ImportCount'] / df_plot['Total']

fig = px.bar(
    df_plot,
    y='YearMonth',
    x='Percentage',
    color='Imports_lower',
    barmode='stack',
    orientation='h',
    title=f'Top {top_n} Most Used (Non-Generic) Imports Over Time (Normalized per Month)',
    labels={'Percentage': 'Usage Share', 'YearMonth': 'Time', 'Imports_lower': 'Import'},
    height=1000
)

fig.update_layout(xaxis_tickangle=45)
fig.show()


top_n = 200

total_counts = (
    df_exploded
    .groupby('Imports_lower')
    .size()
    .reset_index(name='TotalCount')
    .sort_values('TotalCount', ascending=False)
)

# Filter to only imports with at least 100 occurrences
filtered_counts = total_counts[total_counts['TotalCount'] >= 5]
# Then take top N from these
top_imports = filtered_counts.head(top_n)['Imports_lower'].tolist()

# Filter df_exploded for only top N imports of all time
df_top = df_exploded[df_exploded['Imports_lower'].isin(top_imports)].copy()

# Count per month for these top imports
monthly_counts = (
    df_top
    .groupby(['YearMonth', 'Imports_lower'])
    .size()
    .reset_index(name='ImportCount')
)

# Normalize per month
monthly_totals = (
    monthly_counts
    .groupby('YearMonth')['ImportCount']
    .sum()
    .reset_index(name='Total')
)

df_plot = monthly_counts.merge(monthly_totals, on='YearMonth')
df_plot['Percentage'] = df_plot['ImportCount'] / df_plot['Total']

# ---------------------------
# Ridge Plot (Joyplot-style)
# ---------------------------

df_plot['YearMonth_dt'] = pd.to_datetime(df_plot['YearMonth'])
df_plot['YearMonth_ord'] = df_plot['YearMonth_dt'].map(pd.Timestamp.toordinal)

imports = sorted(df_plot['Imports_lower'].unique())
positions = np.arange(len(imports))
colors = cm.viridis(np.linspace(0, 1, len(imports)))

plt.figure(figsize=(12, 18))

for pos, imp, color in zip(positions, imports, colors):
    subset = df_plot[df_plot['Imports_lower'] == imp]
    
    if len(subset) < 2:
        continue

    values = subset['YearMonth_ord'].values
    weights = subset['Percentage'].values

    kde = gaussian_kde(values, weights=weights, bw_method=0.2)
    x_range = np.linspace(values.min(), values.max(), 200)
    y = kde(x_range)
    y_norm = y / y.max() if y.max() != 0 else y

    dates = [pd.Timestamp.fromordinal(int(val)) for val in x_range]

    # Draw baseline line for each ridge
    plt.axhline(y=pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    # Plot ridge
    plt.fill_between(dates, pos, pos + y_norm, color=color, alpha=0.8)

# Adjust font size of y-axis tick labels
plt.yticks(positions + 0.5, imports, fontsize=8)
plt.xticks(fontsize=8) 

plt.margins(y=0)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('gray')
plt.tight_layout()
plt.show()





