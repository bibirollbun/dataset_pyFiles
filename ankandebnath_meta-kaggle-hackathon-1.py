# -*- coding: utf-8 -*-
"""
Meta Kaggle Hackathon: Comprehensive Analysis Notebook
Created: 2025-07-20
Author: Ankan Debnath
"""


# Ignore any unnecessary Warnings
import warnings
warnings.filterwarnings('ignore')


%pip install kagglehub nbformat wordcloud -q


import os

file_path = '/kaggle/working/knowledge_transfer_xgboost.png'

try:
    os.remove(file_path)
    print(f"âœ… File removed: {file_path}")
except FileNotFoundError:
    print(f"â�Œ File not found: {file_path}")
except PermissionError:
    print(f"ğŸš« Permission denied when trying to delete: {file_path}")
except Exception as e:
    print(f"âš ï¸� Unexpected error: {e}")


import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from datetime import datetime
import re


# Download datasets
meta_kaggle_path = kagglehub.dataset_download("kaggle/meta-kaggle")
meta_kaggle_code_path = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print(f"Meta Kaggle Path: {meta_kaggle_path}")
print(f"Meta Kaggle Code Path: {meta_kaggle_code_path}")


# Load core tables
competitions = pd.read_csv(f"{meta_kaggle_path}/Competitions.csv")
users = pd.read_csv(f"{meta_kaggle_path}/Users.csv")
teams = pd.read_csv(f"{meta_kaggle_path}/Teams.csv")
submissions = pd.read_csv(f"{meta_kaggle_path}/Submissions.csv")
tags = pd.read_csv(f"{meta_kaggle_path}/Tags.csv")
comp_tags = pd.read_csv(f"{meta_kaggle_path}/CompetitionTags.csv")
kernels = pd.read_csv(f"{meta_kaggle_path}/Kernels.csv")
kernel_versions = pd.read_csv(f"{meta_kaggle_path}/KernelVersions.csv")

# Convert dates
date_cols = ['EnabledDate', 'DeadlineDate', 'MedalAwardDate', 
             'FirstSubmissionDate', 'RegisterDate']
for col in date_cols:
    if col in competitions: competitions[col] = pd.to_datetime(competitions[col])
    if col in users: users[col] = pd.to_datetime(users[col])
    if col in teams: teams[col] = pd.to_datetime(teams[col])
    
# Merge competition tags
comp_tags = pd.merge(comp_tags, tags, left_on='TagId', right_on='Id', how='left')

# Initial data exploration
def explore_data(df, name):
    print(f"\n{name} Data:")
    print(f"Shape: {df.shape}")
    print("Missing values:")
    print(df.isnull().sum())
    print("\nSample data:")
    return df.head(2)

explore_data(competitions, "Competitions")
explore_data(users, "Users")
explore_data(comp_tags, "Competition Tags")


# Competition timeline analysis
competitions['Year'] = competitions['EnabledDate'].dt.year
comp_year = competitions.groupby('Year').size().reset_index(name='Count')

plt.figure(figsize=(12,6))
sns.lineplot(data=comp_year, x='Year', y='Count', marker='o')
plt.title('Kaggle Competitions Over Time (2010-2025)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('competitions_over_time.png', dpi=300)
plt.show()

# Reward analysis
reward_types = competitions['RewardType'].value_counts()
explode = [0.1] + [0]*(len(reward_types)-1)
plt.figure(figsize=(10,6))
reward_types.plot(kind='pie', autopct='%1.1f%%', 
                 colors=sns.color_palette('pastel'), 
                 explode=explode)
plt.title('Competition Reward Types Distribution', fontsize=16)
plt.ylabel('')
plt.savefig('reward_types.png', dpi=300)
plt.show()

# Duration analysis
competitions['Duration'] = (competitions['DeadlineDate'] - competitions['EnabledDate']).dt.days
plt.figure(figsize=(12,6))
sns.histplot(competitions['Duration'].dropna(), bins=30, kde=True)
plt.title('Competition Duration Distribution', fontsize=16)
plt.xlabel('Duration (Days)')
plt.ylabel('Count')
plt.savefig('competition_duration.png', dpi=300)
plt.show()


# compute counts and group small slices
counts = competitions['RewardType'].value_counts()
top_n = 6
top_counts = counts.iloc[:top_n].copy()
others = counts.iloc[top_n:].sum()
top_counts['Others'] = others

# plot donut chart
fig, ax = plt.subplots(figsize=(8,8))
wedges, texts, autotexts = ax.pie(
    top_counts,
    labels=top_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.85,
    wedgeprops=dict(width=0.3)
)
# draw center circle for donut look
centre_circle = plt.Circle((0,0),0.70,fc='white')
ax.add_artist(centre_circle)

ax.set_title('Top Reward Types (Others grouped)', fontsize=14)
plt.savefig('reward_types_donut.png', dpi=300)
plt.show()

plt.figure(figsize=(10,6))
reward_types = competitions['RewardType'].value_counts().iloc[:10]  # top 10
sns.barplot(x=reward_types.values, y=reward_types.index, palette='pastel')
plt.title('Top 10 Reward Types', fontsize=14)
plt.xlabel('Count')
plt.ylabel('')
plt.tight_layout()
plt.savefig('reward_types_bar.png', dpi=300)
plt.show()


# Highlighting the Duration Distribution


#1. Zoomed-In + Tail Subplots
dur = competitions['Duration'].dropna()
zoom_thresh = dur.quantile(0.90)

fig, (ax1, ax2) = plt.subplots(2,1, sharex=False, figsize=(12,8),
                               gridspec_kw={'height_ratios':[3,1]})

# main cluster
sns.histplot(dur[dur <= zoom_thresh], bins=30, kde=False, ax=ax1)
ax1.set_title('Duration â‰¤ 90th Percentile (~{} days)'.format(int(zoom_thresh)))
ax1.set_ylabel('Count')

# tail
sns.histplot(dur[dur > zoom_thresh], bins=30, kde=False, ax=ax2, color='orange')
ax2.set_title('Duration > 90th Percentile')
ax2.set_xlabel('Duration (Days)')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.savefig('duration_zoom_tail.png', dpi=300)
plt.show()


#2. Log-Scale Histogram
plt.figure(figsize=(12,6))
sns.histplot(dur, bins=50, log_scale=(False, True), kde=False)
plt.title('Duration Distribution (Log Y-axis)')
plt.xlabel('Duration (Days)')
plt.ylabel('Log Count')
plt.tight_layout()
plt.savefig('duration_logscale.png', dpi=300)
plt.show()


#3. Boxplot and ECDF
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.boxplot(x=dur, color='skyblue')
plt.title('Duration Boxplot')

plt.subplot(1,2,2)
sns.ecdfplot(dur, color='green')
plt.title('Duration ECDF')
plt.xlabel('Duration (Days)')
plt.tight_layout()
plt.savefig('duration_summary.png', dpi=300)
plt.show()


# Top domains analysis

counts = comp_tags['Name'].value_counts().nlargest(20)
top20 = pd.DataFrame({
    'Domain': counts.index,
    'Count': counts.values
})
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))

# Draw bars
sns.barplot(
    x='Count',
    y='Domain',
    data=top20,
    palette='viridis',
    ax=ax
)
# Annotate each bar with its count
for p in ax.patches:
    width = p.get_width()
    ax.text(
        width + 2,
        p.get_y() + p.get_height() / 2,
        int(width),
        va='center'
    )

ax.set_title('Top 20 Competition Domains', fontsize=16)
ax.set_xlabel('Count')
ax.set_ylabel('Domain Tag')
plt.tight_layout()
plt.savefig('top_domains.png', dpi=300)
plt.show()

# Domain evolution over time
comp_tags_year = pd.merge(comp_tags, competitions[['Id', 'Year']], 
                         left_on='CompetitionId', right_on='Id')
tag_year_counts = comp_tags_year.groupby(['Year', 'Name']).size().reset_index(name='Count')
top_5_tags = tag_year_counts.groupby('Name')['Count'].sum().nlargest(5).index
tag_year_top = tag_year_counts[tag_year_counts['Name'].isin(top_5_tags)]

plt.figure(figsize=(14,8))
sns.lineplot(data=tag_year_top, x='Year', y='Count', hue='Name', 
             marker='o', linewidth=2.5)
plt.title('Evolution of Top Domains (2010-2025)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Competition Count')
plt.legend(title='Domain')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('domain_evolution.png', dpi=300)
plt.show()

# Social impact gap analysis

#unique competitions tagged as â€œsocialâ€¦â€�
social_tags = ['health','education','environment','civic','social',
               'medical','policy','inclusion']
social_ids = (
    comp_tags[comp_tags['Name']
              .str.lower()
              .isin(social_tags)]
    ['CompetitionId']
    .unique()
)

gap_df = pd.DataFrame({
    'Category': ['Social Impact', 'All Competitions'],
    'Count':    [len(social_ids), competitions['Id'].nunique()]
})
plt.figure(figsize=(10,6))
sns.barplot(
    data=gap_df,
    x='Category',
    y='Count',
    palette=['#2a7ae2','#cccccc']
)
plt.title('Social-Impact vs. All Competitions', fontsize=14)
plt.ylabel('Number of Competitions')
for i, v in enumerate(gap_df['Count']):
    plt.text(i, v + max(gap_df['Count'])*0.01, str(v),
             ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('social_gap.png', dpi=300)
plt.show()


# User registration trends
users['RegisterYear'] = users['RegisterDate'].dt.year
users_year = users.groupby('RegisterYear').size().reset_index(name='Count')

plt.figure(figsize=(12,6))
sns.lineplot(data=users_year, x='RegisterYear', y='Count', 
            marker='o', color='purple')
plt.title('Kaggle User Registration Trends', fontsize=16)
plt.xlabel('Year')
plt.ylabel('New Users')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('user_growth.png', dpi=300)
plt.show()


# Performance tier distribution
plt.figure(figsize=(10,6))
sns.countplot(data=users, x='PerformanceTier', 
             palette='Blues', order=users['PerformanceTier'].value_counts().index)
plt.title('User Performance Tier Distribution', fontsize=16)
plt.xlabel('Performance Tier')
plt.ylabel('Count')
plt.savefig('performance_tiers.png', dpi=300)
plt.show()


# Team size analysis

# Compute per-competition team counts
team_counts = teams.groupby('CompetitionId')['Id'].count()
p90 = team_counts.quantile(0.90)
median = team_counts.median()

# Set up 2-plot layout
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12,8),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=False
)

# Main cluster (â‰¤90th percentile)
sns.histplot(
    team_counts[team_counts <= p90],
    bins=30,
    color='skyblue',
    edgecolor='white',
    ax=ax1
)
ax1.axvline(median, color='red', linestyle='--',
            label=f'Median = {int(median)}')
ax1.set_title(f'Team Sizes per Competition (â‰¤ 90th percentile â‰ˆ {int(p90)} teams)')
ax1.set_ylabel('Competition Count')
ax1.legend()

# Tail (>90th percentile)
sns.histplot(
    team_counts[team_counts > p90],
    bins=30,
    color='orange',
    edgecolor='white',
    ax=ax2
)
ax2.set_title('Tail: Team Sizes > 90th Percentile')
ax2.set_xlabel('Number of Teams')
ax2.set_ylabel('Competition Count')
plt.tight_layout()
plt.savefig('team_size_dist.png', dpi=300)
plt.show()

# log-scaled y-axis
plt.figure(figsize=(12,6))
sns.histplot(
    team_counts,
    bins=50,
    log_scale=(False, True),
    color='teal',
    edgecolor='white'
)
plt.axvline(median, color='red', linestyle='--',
            label=f'Median = {int(median)}')
plt.title('Team Size Distribution per Competition (Log Y-axis)')
plt.xlabel('Number of Teams')
plt.ylabel('Log(Competition Count)')
plt.legend()
plt.tight_layout()
plt.savefig('team_size_dist_log.png', dpi=300)
plt.show()


# Compute true TeamSize from submissions
team_sizes = (
    submissions
      .groupby('TeamId')['SubmittedUserId']
      .nunique()
      .reset_index(name='TeamSize')
)

# Merge TeamSize back into teams
teams = teams.merge(
    team_sizes,
    left_on='Id',
    right_on='TeamId',
    how='left'
)
# solos (no entry in submissions) become size 1
teams['TeamSize'] = teams['TeamSize'].fillna(1).astype(int)

# Map Medal codes â†’ names and filter out non-winners
teams['MedalCode'] = pd.to_numeric(teams['Medal'], errors='coerce')
medal_map = {1:'Gold', 2:'Silver', 3:'Bronze'}
teams['MedalName'] = teams['MedalCode'].map(medal_map)

medal_teams = teams[teams['MedalName'].notna()].copy()

# checking conditions
print("Columns now in medal_teams:", medal_teams.columns)
print("Sample TeamSize:", medal_teams['TeamSize'].value_counts().head())

# Downâ€�sample for speed (max 200 points per medal)
plot_df = (
    medal_teams
      .groupby('MedalName', group_keys=False)
      .apply(lambda g: g.sample(n=min(len(g),200), random_state=1))
)

# Plot box + strip    
plt.figure(figsize=(8,5))

# boxplot of the full distribution
sns.boxplot(
    x='TeamSize', 
    y='MedalName',
    data=medal_teams,
    order=['Gold','Silver','Bronze'],
    palette=['#FFD700','#C0C0C0','#CD7F32'],
    fliersize=0
)

# overlay a jittered strip of sampled points
sns.stripplot(
    x='TeamSize',
    y='MedalName',
    data=plot_df,
    order=['Gold','Silver','Bronze'],
    color='k',
    alpha=0.4,
    size=3,
    jitter=0.3
)

plt.title('Team Size Distribution for Medal Winners', fontsize=16)
plt.xlabel('Members per Team')
plt.ylabel('')
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig('team_size_medal_dist.png', dpi=300)
plt.show()


# â”€â”€â”€ merge CompetitionId, unify Score, parse dates â”€â”€â”€â”€â”€
subs = (
    submissions
    .merge(teams[['Id','CompetitionId']], left_on='TeamId', right_on='Id', how='left')
    .assign(
        Score=lambda df: df[['PublicScoreFullPrecision','PrivateScoreFullPrecision']].max(axis=1),
        SubmissionDate=lambda df: pd.to_datetime(df['SubmissionDate'])
    )
)

# â”€â”€â”€ ANALYSIS FUNCTION â”€â”€â”€
def analyze_competition_progress(comp_id):
    df = subs[subs['CompetitionId']==comp_id].sort_values('SubmissionDate').copy()
    if df.empty:
        return {
            'competition_id': comp_id,
            'total_submissions': 0,
            'duration_days': 0,
            'breakthrough_count': 0,
            'final_score': None
        }

    df['CumulativeMax'] = df['Score'].cummax()
    df['TimeDelta'] = (df['SubmissionDate'] - df['SubmissionDate'].min()).dt.days
    df['Improvement'] = df['CumulativeMax'].diff().fillna(0)

    thresh = df['Improvement'].std() * 2
    brk = df[df['Improvement'] > thresh]

    # Plot
    plt.figure(figsize=(14,8))
    sns.lineplot(data=df, x='TimeDelta', y='CumulativeMax', label='Best Score')
    sns.scatterplot(data=brk, x='TimeDelta', y='CumulativeMax',
                    color='red', s=100, label='Breakthrough')

    # Milestone annotations
    first_day, first_score = df['TimeDelta'].iloc[0], df['CumulativeMax'].iloc[0]
    plt.scatter(first_day, first_score, color='blue', s=100, label='First Submission')
    median_val = df['CumulativeMax'].median()
    plt.axhline(median_val, color='green', linestyle=':', label='Median Best Score')
    plt.text(df['TimeDelta'].max()*0.05, median_val,
             f'Median: {median_val:.3f}', color='green', va='bottom')

    plt.title(f'Competition Progression: {comp_id}', fontsize=16)
    plt.xlabel('Days Since Start')
    plt.ylabel('Best Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'competition_progression_{comp_id}.png', dpi=300)
    plt.show()

    return {
        'competition_id':   comp_id,
        'total_submissions': len(df),
        'duration_days':     int(df['TimeDelta'].max()),
        'breakthrough_count': len(brk),
        'final_score':       df['CumulativeMax'].iloc[-1]
    }

# â”€â”€â”€ RUN ON NON-EMPTY COMPETITIONS â”€â”€â”€â”€â”€â”€â”€
valid_ids = subs['CompetitionId'].dropna().unique()
sample_ids = [cid for cid in competitions['Id'].sample(3, random_state=42) if cid in valid_ids]

results = [analyze_competition_progress(cid) for cid in sample_ids]
comp_analysis = pd.DataFrame(results)
print(comp_analysis)



# load mapping of language IDs to names
kernel_langs = pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')  
lang_map = kernel_langs.set_index('Id')['Name']

# attach a LanguageName column
kernel_versions['LanguageName'] = (
    kernel_versions['ScriptLanguageId']
      .map(lang_map)
      .fillna('Unknown')
)

# top 10 languages
top10 = (
    kernel_versions['LanguageName']
      .value_counts()
      .nlargest(10)
      .reset_index(name='Count')
      .rename(columns={'index':'LanguageName'})
)

# plot
plt.figure(figsize=(12,6))
ax = sns.barplot(y='LanguageName', x='Count', data=top10, palette='muted')
for p in ax.patches:
    ax.text(p.get_width()+5, p.get_y() + p.get_height()/2,
            int(p.get_width()), va='center')
ax.set_title('Top 10 Kernel Languages', fontsize=16)
ax.set_xlabel('Number of Kernels')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('kernel_languages.png', dpi=300)
plt.show()


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1) Prepare features & drop rows with missing target or any feature
cols = ['RewardQuantity', 'Duration', 'NumPrizes', 'TotalTeams']
df = competitions[cols].dropna()

X = df[['RewardQuantity', 'Duration', 'NumPrizes']]
y = df['TotalTeams']

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Build pipelines with median imputation + model
models = {
    "RandomForest": Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]),
    "LightGBM": Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('lgb', LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
}

# 4) Fit & evaluate
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"{name:10s} â†’ RMSE: {rmse:.2f}, RÂ²: {r2:.2f}")



# Extract RandomForest pipeline and feature importances
rf_pipe = models["RandomForest"]
imps = rf_pipe.named_steps['rf'].feature_importances_

# Build a sorted Series
feat_imp = pd.Series(imps, index=X.columns).sort_values(ascending=False)

# Plot
plt.figure(figsize=(8,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Random Forest Feature Importance', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=300)
plt.show()


kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', parse_dates=['CreationDate'])
kvc = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv')
comp_tags = pd.read_csv('/kaggle/input/meta-kaggle/CompetitionTags.csv')
tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')

print("KVC columns:", kvc.columns.tolist())  # debug

if 'CompetitionId' not in kvc.columns:
    possible = [c for c in kvc.columns if 'competition' in c.lower()]
    if not possible:
        raise KeyError(f"No competition column in kvc: {kvc.columns.tolist()}")
    kvc = kvc.rename(columns={possible[0]: 'CompetitionId'})
comp_domains = (
    comp_tags
      .merge(tags[['Id','Name']], left_on='TagId', right_on='Id', how='left')
      .rename(columns={'Name':'Domain'})[['CompetitionId','Domain']]
      .drop_duplicates()
)

def analyze_knowledge_transfer(technique):
    tv = kernel_versions[
        kernel_versions['Title'].str.contains(technique, case=False, na=False)
    ].copy()
    if tv.empty:
        return None

    # attach CompetitionId, drop nulls
    tv = (
        tv
        .merge(kvc[['KernelVersionId','CompetitionId']],
               left_on='Id', right_on='KernelVersionId', how='left')
        .dropna(subset=['CompetitionId'])
    )

    # attach Domain, drop nulls
    tv = tv.merge(comp_domains, on='CompetitionId', how='left')
    if tv['Domain'].nunique() < 2:
        return None

    first_use = tv['CreationDate'].min()
    dom_first = (
        tv.groupby('Domain')['CreationDate']
          .min()
          .reset_index(name='FirstUse')
    )
    dom_first['DaysSinceFirstUse'] = (dom_first['FirstUse'] - first_use).dt.days
    avg_time = dom_first['DaysSinceFirstUse'].mean()

    # pick the 15 fastestâ€�adopting domains
    fastest = dom_first.nsmallest(15, 'DaysSinceFirstUse')
    
    plt.figure(figsize=(10,6))
    ax = sns.scatterplot(
        data=fastest,
        x='FirstUse',
        y='Domain',
        s=100,
        color='teal'
    )
    ax.axvline(first_use, color='r', linestyle='--', label='First Use')
    
    ax.set_title(f'Knowledge Transfer of {technique} (Top 15 Fastest Domains)', fontsize=14)
    ax.set_xlabel('First Use Date')
    ax.set_ylabel('Domain')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'knowledge_transfer_{technique}_top15.png', dpi=300)
    plt.show()

    return avg_time

# â”€â”€â”€ Run for sample techniques â”€â”€â”€â”€
for tech in ['transformers','cnn','xgboost','autoencoder']:
    t = analyze_knowledge_transfer(tech)
    if t is not None:
        print(f"{tech}: average transfer = {t:.1f} days")


import plotly.express as px

# pivot into "YearÃ—Domain"
domain_ts = tag_year_counts.pivot_table(
    index='Year', columns='Name', values='Count', aggfunc='sum', fill_value=0
)

# pick top-20 domains by overall volume
top20 = domain_ts.sum().nlargest(20).index
domain_ts_top20 = domain_ts[top20]

fig = px.imshow(
    domain_ts_top20.T, 
    labels=dict(x="Year", y="Domain", color="Competitions"),
    title="Evolution of Top 20 Domains Over Time",
    color_continuous_scale="Viridis",
    aspect="auto"
)
fig.update_layout(
    height=600,
    yaxis=dict(tickfont=dict(size=10)),
    xaxis=dict(tickangle=0)
)
fig.write_html("domain_evolution_top20.html")
fig.show()


import scipy.cluster.hierarchy as sch

domain_ts = (
    tag_year_counts
      .pivot_table(index='Year', columns='Name', values='Count', aggfunc='sum', fill_value=0)
)

# 2) cluster the domains into, say, 4 groups
linkage = sch.linkage(domain_ts.T, method='ward')
clusters = sch.fcluster(linkage, t=4, criterion='maxclust')

# build a lookup DataFrame
domain_clusters = pd.DataFrame({
    'Name': domain_ts.columns,
    'Cluster': clusters.astype(str)   # convert to str for faceting
})

# 3) attach cluster labels back to the original longâ€�form
tag_clustered = (
    tag_year_counts
      .merge(domain_clusters, on='Name', how='left')
)

# 4) plot small multiples by cluster
fig = px.line(
    tag_clustered,
    x='Year',
    y='Count',
    color='Name',
    facet_col='Cluster',
    facet_col_wrap=2,            # two columns of panels
    title='Domain Evolution by Clustered Groups',
    height=800,
    width=1000
)

fig.update_layout(showlegend=False)
fig.for_each_annotation(lambda a: a.update(text=f"Group {a.text.split('=')[1]}"))
fig.show()


# 1) pivot YearÃ—Domain
domain_ts = (
    tag_year_counts
      .pivot_table(index='Year', columns='Name', values='Count', aggfunc='sum', fill_value=0)
)

# 2) hierarchical cluster on the domain timeâ€�series
Z = sch.linkage(domain_ts.T, method='ward')
# cut into 4 clusters (tweak as needed)
cluster_ids = sch.fcluster(Z, t=4, criterion='maxclust')

# map domain â†’ cluster
domain_clusters = pd.Series(cluster_ids, index=domain_ts.columns, name='Cluster')

# 3) collapse each cluster to its mean timeâ€�series
cluster_summary = (
    domain_ts
      .groupby(domain_clusters, axis=1)  # group columns by cluster_id
      .mean()
)

if cluster_summary.empty:
    raise ValueError("No data to plot: check your clustering or your pivot table.")

# 4) transpose to ClusterÃ—Year and plot
plt.figure(figsize=(8, 4 + cluster_summary.shape[1]*0.3))
sns.heatmap(
    cluster_summary.T,
    cmap='viridis',
    cbar_kws={'label': 'Avg # Competitions'},
    linewidths=0.5,
    linecolor='gray'
)
plt.xlabel('Year')
plt.ylabel('Cluster ID')
plt.title('Average Domain Evolution by Cluster')
plt.tight_layout()
plt.savefig('avg_domain_evolution_by_cluster.png', dpi=300)
plt.show()

