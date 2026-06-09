%pip install openai dotenv beautifulsoup4 rapidfuzz networkx pyvis -q


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
import numpy as np
import ast
import math
import networkx as nx
import plotly.io as pio

from pyvis.network import Network
from rapidfuzz import process, fuzz
from matplotlib import cm
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, entropy
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import display, HTML
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy import array
from bs4 import BeautifulSoup
from kagglehub import KaggleDatasetAdapter

import warnings
warnings.filterwarnings("ignore")

load_dotenv()


pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 1000)
warnings.filterwarnings('ignore')
pio.renderers.default = 'iframe' #https://www.kaggle.com/discussions/product-announcements/549950


class Config:
    SAVE_DATASET=False
    DATASET_PATH='/kaggle/input/competition-writeups-with-technologies-and-summary/competition-writeups-technologies.csv'


competitions_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "Competitions.csv",
)
display(competitions_df)


teams_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "Teams.csv",
)
display(teams_df)


forum_topics_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "ForumTopics.csv",
)
display(forum_topics_df)


forum_messages_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "ForumMessages.csv",
)
display(forum_messages_df)


# Rename a few columns
teams_df = teams_df.rename(columns={'Id': 'Id_Teams'})
forum_topics_df = forum_topics_df.rename(columns={'Id': 'Id_ForumTopics',
                                         'Title': 'Title of Writeup',
                                         'CreationDate': 'Date of Writeup'})
forum_messages_df = forum_messages_df.rename(columns={'Id': 'Id_ForumMessages',
                                              'Message': 'Writeup'})
competitions_df = competitions_df.rename(columns={'Id': 'Id_Competitions',
                                           'Title': 'Title of Competition',
                                           'EnabledDate': 'Competition Launch Date'})

# Organize everything
df = teams_df.merge(right=forum_topics_df, how='inner', left_on='WriteUpForumTopicId', right_on='Id_ForumTopics')
df = df.merge(right=forum_messages_df, how='inner', left_on='Id_ForumTopics', right_on='ForumTopicId')
df = df.merge(right=competitions_df, how='inner', left_on='ForumId', right_on='ForumId')
df = df[df['FirstForumMessageId'] == df['Id_ForumMessages']]
df = df[df['HostSegmentTitle'].isin(['Featured','Research'])]

# Add in URLs
df['Id_Competitions'] = df['Id_Competitions'].astype(str)
df['Competition URL'] = 'https://www.kaggle.com/c/'+df['Id_Competitions']
df['Id_ForumTopics'] = df['Id_ForumTopics'].astype(str)
df['Writeup URL'] = 'https://www.kaggle.com/c/'+df['Id_Competitions']+'/discussion/'+df['Id_ForumTopics']

# Final cleanup
df = df[['Competition Launch Date',
         'Title of Competition',
         'Competition URL',
         'Date of Writeup',
         'Title of Writeup',
         'Writeup',
         'Writeup URL']]

print('# of entries: ',df['Writeup'].count())
display(df)


if not os.path.exists(Config.DATASET_PATH):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = """
    You are a helpful assistant that extracts technologies and summarizes technical writeups.

    Given this writeup:
    \"\"\"{WRITEUP}\"\"\"

    Extract a list of the technologies used (e.g., SVM, CNN, Transformer, etc) and give a short summary.

    Return the result as a JSON in this format:

    {{
    "technologies": ["<technology_1>", "<technology_2>", ...],
    "summary": "<summary_text>"
    }}
    """

    def clean_html(raw_html):
        if not raw_html or pd.isna(raw_html):
            return ""
        return BeautifulSoup(raw_html, "html.parser").get_text()

    def strip_code_fence(text):
        return re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text).strip()

    def extract_technologies_and_summary(writeup):
        cur_prompt = prompt.replace('{WRITEUP}', writeup)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "user", "content": cur_prompt}
                ],
            )
            content = response.choices[0].message.content
            cleaned_content = strip_code_fence(content)
            result = ast.literal_eval(cleaned_content)
            return result.get("technologies", []), result.get("summary", ""), ""
        except Exception as e:
            print(f"Error processing writeup: {e}")
            print(cleaned_content)
            print(writeup)
            return [], "", cleaned_content


if os.path.exists(Config.DATASET_PATH):
    df = pd.read_csv(Config.DATASET_PATH)
else:
    tqdm.pandas()
    df[["Technologies", "Summary", "Error"]] = df["Writeup"].progress_apply(
        lambda x: pd.Series(extract_technologies_and_summary(clean_html(x)))
    )


display(df)


if Config.SAVE_DATASET:
    df.to_csv(Config.DATASET_PATH)


for col in ['Technologies']:
    if col in df.columns:
        df[col] = df[col].dropna().apply(ast.literal_eval) 


df = df.explode('Technologies')
df = df[df['Technologies'].notna()]

print(len(df))
display(df)


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def cluster_technologies(tech_list, threshold=85):
    clusters = {}
    for tech in tech_list:
        if tech == '' or tech == None:
            continue
        if not clusters:
            clusters[tech] = [tech]
            continue
        match, score, _ = process.extractOne(
            tech, clusters.keys(), scorer=fuzz.ratio
        )
        if score >= threshold:
            clusters[match].append(tech)
        else:
            clusters[tech] = [tech]
    return clusters

df['Tech_normalized'] = df['Technologies'].astype(str).map(normalize)
unique_techs = df['Tech_normalized'].unique().tolist()
clusters = cluster_technologies(unique_techs)

# Create a mapping from variant -> canonical name
tech_mapping = {}
for canonical, variants in clusters.items():
    for v in variants:
        tech_mapping[v] = canonical


df['Tech_clustered'] = df['Tech_normalized'].map(tech_mapping)
display(df)


# Drop unwanted index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df['Date of Writeup'] = pd.to_datetime(df['Date of Writeup'])
df['Month'] = df['Date of Writeup'].dt.to_period('M').dt.to_timestamp()


# Group by writeup URL to aggregate all associated technologies
grouped = df.groupby("Writeup URL").agg({
    "Date of Writeup": "first",
    "Tech_clustered": lambda x: list(x.unique())
}).reset_index()


# --- 1. Top N technologies ---
all_techs = [tech for tech_list in grouped["Tech_clustered"] for tech in tech_list]
tech_counter = Counter(all_techs)
top_n = 100 
top_techs = tech_counter.most_common(top_n)

# --- 2. Technology Trends Over Time ---
exploded = grouped.explode("Tech_clustered")
exploded["Month"] = exploded["Date of Writeup"].dt.to_period("M").dt.to_timestamp()

# Filter to only top technologies
top_tech_names = [tech for tech, _ in top_techs]
filtered = exploded[exploded["Tech_clustered"].isin(top_tech_names)]

# Group by month and technology
trend_df = filtered.groupby(["Month", "Tech_clustered"]).size().reset_index(name="Count")
pivot_trend = trend_df.pivot(index="Month", columns="Tech_clustered", values="Count").fillna(0)


top_techs_df = pd.DataFrame(top_techs, columns=["Tech_clustered", "Count"])

plt.figure(figsize=(12, 18))
sns.barplot(data=top_techs_df, y="Tech_clustered", x="Count", palette="viridis")
plt.tight_layout()
plt.show()


# Group by writeup and collect normalized technologies
grouped = df.groupby("Writeup URL").agg({
    "Tech_clustered": lambda x: list(set(x))
}).reset_index()

# Determine top N techs by frequency
top_n = 30
all_techs = [tech for sublist in grouped["Tech_clustered"] for tech in sublist]
tech_counter = Counter(all_techs)
top_techs = [tech for tech, _ in tech_counter.most_common(top_n)]

# Build co-occurrence matrix
co_occurrence_counts = Counter()
for techs in grouped["Tech_clustered"]:
    techs = [t for t in techs if t in top_techs]
    for combo in combinations(sorted(techs), 2):
        co_occurrence_counts[combo] += 1

# Initialize empty co-occurrence DataFrame
co_occurrence_df = pd.DataFrame(0, index=top_techs, columns=top_techs)
for (a, b), count in co_occurrence_counts.items():
    co_occurrence_df.loc[a, b] = count
    co_occurrence_df.loc[b, a] = count

# Plot heatmap
plt.figure(figsize=(20, 18))
sns.heatmap(co_occurrence_df, annot=True, fmt="d", cmap="Blues")
plt.title("Co-occurrence Heatmap of Top Technologies (Normalized Clusters)")
plt.tight_layout()
plt.show()


grouped = df.groupby("Writeup URL").agg({
    "Month": "first",
    "Tech_clustered": lambda x: list(x.unique())
}).reset_index()

exploded = grouped.explode("Tech_clustered")

top_n = 100
tech_counts = (
    exploded
    .groupby('Tech_clustered')
    .size()
    .reset_index(name='TotalCount')
    .sort_values('TotalCount', ascending=False)
)

filtered_counts = tech_counts[tech_counts['TotalCount'] >= 5]
top_techs = filtered_counts.head(top_n)['Tech_clustered'].tolist()

df_top = exploded[exploded['Tech_clustered'].isin(top_techs)].copy()

monthly_counts = (
    df_top
    .groupby(['Month', 'Tech_clustered'])
    .size()
    .reset_index(name='TechCount')
)

monthly_totals = (
    monthly_counts
    .groupby('Month')['TechCount']
    .sum()
    .reset_index(name='Total')
)

df_plot = monthly_counts.merge(monthly_totals, on='Month')
df_plot['Percentage'] = df_plot['TechCount'] / df_plot['Total']

df_plot['Month_dt'] = pd.to_datetime(df_plot['Month'])
df_plot['Month_ord'] = df_plot['Month_dt'].map(pd.Timestamp.toordinal)

techs = sorted(df_plot['Tech_clustered'].unique())
positions = np.arange(len(techs))
colors = cm.viridis(np.linspace(0, 1, len(techs)))

plt.figure(figsize=(12, 26))

for pos, tech, color in zip(positions, techs, colors):
    subset = df_plot[df_plot['Tech_clustered'] == tech]

    if len(subset) < 2:
        continue

    values = subset['Month_ord'].values
    weights = subset['Percentage'].values

    kde = gaussian_kde(values, weights=weights, bw_method=0.2)
    x_range = np.linspace(values.min(), values.max(), 200)
    y = kde(x_range)
    y_norm = y / y.max() if y.max() != 0 else y

    dates = [pd.Timestamp.fromordinal(int(val)) for val in x_range]

    plt.axhline(y=pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    plt.fill_between(dates, pos, pos + y_norm, color=color, alpha=0.8)

plt.yticks(positions + 0.5, techs, fontsize=8)
plt.xticks(fontsize=8)

plt.margins(y=0)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('gray')

plt.tight_layout()
plt.show()


top_n = 100

top_techs = (
    df['Tech_clustered']
    .value_counts()
    .nlargest(top_n)
    .index
)

filtered_df = df[df['Tech_clustered'].isin(top_techs)]

adoption_timeline = (
    filtered_df.groupby('Tech_clustered')['Date of Writeup']
    .min()
    .reset_index()
    .sort_values('Date of Writeup')
)

fig, ax = plt.subplots(figsize=(10, len(adoption_timeline) * 0.2))
ax.hlines(
    y=range(len(adoption_timeline)),
    xmin=adoption_timeline['Date of Writeup'],
    xmax=pd.Timestamp.now(),
    color='skyblue'
)
ax.plot(
    adoption_timeline['Date of Writeup'],
    range(len(adoption_timeline)),
    'o',
    color='blue'
)

ax.set_yticks(range(len(adoption_timeline)))
ax.set_yticklabels(adoption_timeline['Tech_clustered'])

ax.set_ylim(-0.5, len(adoption_timeline) - 0.5)

ax.set_title(f"Top {top_n} Technology Adoption Timeline (First Appearance in Writeups)")
ax.set_xlabel("Date of First Appearance")
plt.tight_layout()
plt.show()



top_n = 30

df['Competition Launch Date'] = pd.to_datetime(df['Competition Launch Date'], errors='coerce')
df = df.dropna(subset=['Competition Launch Date'])

recent_competitions = (
    df[['Title of Competition', 'Competition Launch Date']]
    .drop_duplicates()
    .sort_values('Competition Launch Date', ascending=False)
    .head(top_n)['Title of Competition']
)

filtered_df = df[df['Title of Competition'].isin(recent_competitions)]

tech_comp = (
    filtered_df.groupby(['Title of Competition', 'Tech_clustered'])
    .size()
    .reset_index(name='Count')
)

fig = px.treemap(
    tech_comp,
    path=['Title of Competition', 'Tech_clustered'],
    values='Count',
)

fig.update_layout(
    width=1300,
    height=1000,
    margin=dict(t=0, l=5, r=5, b=5)
)

fig.show()



df['Competition Launch Date'] = pd.to_datetime(df['Competition Launch Date'], errors='coerce')

df_recent = df[df['Competition Launch Date'].dt.year >= 2010].copy()
df_recent['Year'] = df_recent['Competition Launch Date'].dt.year

def normalized_entropy(series):
    if len(series) < 5:
        return np.nan
    counts = series.value_counts()
    probs = counts / counts.sum()
    max_entropy = np.log2(len(probs))
    return entropy(probs, base=2) / max_entropy if max_entropy > 0 else 0

def effective_tech_count(series):
    if len(series) < 5:
        return np.nan
    counts = series.value_counts()
    probs = counts / counts.sum()
    return 2 ** entropy(probs, base=2)


# --- Plot 1: Normalized Entropy per Competition ---
tech_diversity_per_competition = (
    df_recent.groupby('Title of Competition')['Tech_clustered']
    .apply(normalized_entropy)
    .dropna()
    .sort_values(ascending=False)
)

tech_diversity_df = tech_diversity_per_competition.reset_index()
tech_diversity_df['Short Title'] = tech_diversity_df['Title of Competition'].apply(
    lambda x: x if len(x) <= 30 else x[:27] + '...'
)
tech_diversity_df['Display Title'] = (
    tech_diversity_df['Short Title'] + ' [' + tech_diversity_df.index.astype(str) + ']'
)

fig1 = px.bar(
    tech_diversity_df,
    x='Display Title',
    y='Tech_clustered',
    labels={'Tech_clustered': 'Normalized Entropy', 'Display Title': 'Competition'},
    hover_data={'Title of Competition': True}
)
fig1.update_layout(
    width=min(tech_diversity_df.shape[0] * 40, 3000),
    height=600,
    xaxis_tickangle=45
)

# --- Plot 2: Normalized Entropy per Year ---
tech_diversity_per_year = (
    df_recent.groupby('Year')['Tech_clustered']
    .apply(normalized_entropy)
    .dropna()
    .sort_index()
)

# --- Plot 3: Effective Number of Techs per Competition ---
effective_techs_per_competition = (
    df_recent.groupby('Title of Competition')['Tech_clustered']
    .apply(effective_tech_count)
    .dropna()
    .sort_values(ascending=False)
)

effective_techs_df = effective_techs_per_competition.reset_index()
effective_techs_df['Short Title'] = effective_techs_df['Title of Competition'].apply(
    lambda x: x if len(x) <= 30 else x[:27] + '...'
)
effective_techs_df['Display Title'] = (
    effective_techs_df['Short Title'] + ' [' + effective_techs_df.index.astype(str) + ']'
)

fig3 = px.bar(
    effective_techs_df,
    x='Display Title',
    y='Tech_clustered',
    labels={'Tech_clustered': 'Effective Number of Techs', 'Display Title': 'Competition'},
    hover_data={'Title of Competition': True}
)
fig3.update_layout(
    width=min(effective_techs_df.shape[0] * 40, 3000),
    height=600,
    xaxis_tickangle=45
)

# --- Plot 4: Effective Number of Techs per Year ---
effective_techs_per_year = (
    df_recent.groupby('Year')['Tech_clustered']
    .apply(effective_tech_count)
    .dropna()
    .sort_index()
)

# --- Combined Plot: Normalized vs. Effective Entropy per Year ---
fig2_and_4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Tech Diversity per Year (Normalized Entropy)",
                    "Effective Tech Count per Year"),
    horizontal_spacing=0.15
)

# Fig2 (Normalized Entropy per Year)
fig2_and_4.add_trace(
    go.Bar(
        x=tech_diversity_per_year.index,
        y=tech_diversity_per_year.values,
        name="Normalized Entropy"
    ),
    row=1, col=1
)

# Fig4 (Effective Tech Count per Year)
fig2_and_4.add_trace(
    go.Bar(
        x=effective_techs_per_year.index,
        y=effective_techs_per_year.values,
        name="Effective Tech Count"
    ),
    row=1, col=2
)

fig2_and_4.update_layout(
    width=1200,
    height=600,
    showlegend=False,
    title_text="Yearly Technology Diversity Overview",
)

fig2_and_4.update_xaxes(title_text="Year", row=1, col=1)
fig2_and_4.update_xaxes(title_text="Year", row=1, col=2)
fig2_and_4.update_yaxes(title_text="Normalized Entropy", row=1, col=1)
fig2_and_4.update_yaxes(title_text="Effective Tech Count", row=1, col=2)


fig1.show()


fig2_and_4.show()


fig3.show()


df['Competition Launch Date'] = pd.to_datetime(df['Competition Launch Date'], errors='coerce')
df['Year'] = df['Competition Launch Date'].dt.year

os.makedirs("img", exist_ok=True)

for year in sorted(df['Year'].dropna().unique()):
    df_year = df[df['Year'] == year]

    # Build the bipartite graph
    B = nx.Graph()
    competition_nodes = df_year['Title of Competition'].unique().tolist()

    for idx, row in df_year.iterrows():
        comp = row['Title of Competition']
        techs = [row['Tech_clustered']] if isinstance(row['Tech_clustered'], str) else row['Tech_clustered']
        for tech in techs:
            B.add_node(comp, group='competition')
            B.add_node(tech, group='technology')
            B.add_edge(comp, tech)

    # Skip empty graphs
    if len(B.nodes) == 0:
        continue

    net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(B)
    net.force_atlas_2based(gravity=-30, central_gravity=0.02, spring_length=100, spring_strength=0.01)
    net.write_html(f"/kaggle/working/bipartite_graph_{year}.html")

print('Graphs saved.')





