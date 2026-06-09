import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv', parse_dates=['EnabledDate', 'DeadlineDate'])
teams = pd.read_csv('/kaggle/input/meta-kaggle/Teams.csv', low_memory=False)
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', low_memory=False)

print("Competitions shape:", competitions.shape)
print("Teams shape:", teams.shape)
print("Users shape:", users.shape)
print("KernelVersions shape:", kernel_versions.shape)

competitions.head(3)



competitions['Year'] = competitions['EnabledDate'].dt.year
comp_trend = competitions.groupby('Year').size().reset_index(name='Competitions')

fig = px.line(
    comp_trend, 
    x='Year', 
    y='Competitions', 
    markers=True, 
    title='Number of Competitions Over Time'
)
fig.update_traces(line=dict(width=3))
fig.show()



ai_keywords = ['deep', 'learning', 'transformer', 'bert', 'gpt', 'vision', 'llm', 'gan']
competitions['TitleLower'] = competitions['Title'].str.lower()

# Count keywords per year
keyword_trends = {}
for k in ai_keywords:
    keyword_trends[k] = (
        competitions[competitions['TitleLower'].str.contains(k, na=False)]
        .groupby('Year')
        .size()
    )

keyword_df = pd.DataFrame(keyword_trends).fillna(0).reset_index().rename(columns={'index': 'Year'})

fig = px.line(
    keyword_df, 
    x='Year', 
    y=ai_keywords, 
    title='AI Keywords in Competition Titles Over Time'
)
fig.show()



competitions['RewardQuantity'] = competitions['RewardQuantity'].fillna(0)
prize_by_year = competitions.groupby('Year')['RewardQuantity'].sum().reset_index()

fig = px.bar(
    prize_by_year, 
    x='Year', 
    y='RewardQuantity', 
    title='Total Prize Money Over Time'
)
fig.show()


comp_category = competitions.groupby(['Year', 'HostSegmentTitle']).size().reset_index(name='Count')

fig = px.bar(
    comp_category, 
    x='Year', 
    y='Count', 
    color='HostSegmentTitle', 
    title='Competitions by Category Over Time', 
    barmode='stack'
)
fig.show()


competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
competitions['Year'] = competitions['EnabledDate'].dt.year

comp_trend = competitions.groupby('Year').size().reset_index(name='Competitions')

comp_trend['RollingAvg'] = comp_trend['Competitions'].rolling(3, center=True).mean()

# Plot
fig = px.line(
    comp_trend, 
    x='Year', 
    y=['Competitions', 'RollingAvg'], 
    markers=True, 
    title='Competitions (3-Year Rolling Average)'
)
fig.show()


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Split competitions into eras
ml_era = competitions[competitions['Year'] <= 2014]
ai_era = competitions[competitions['Year'] > 2014]

# Generate word clouds
def plot_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

plot_wordcloud(ml_era['Title'].dropna(), "ML Era (2010–2014)")
plot_wordcloud(ai_era['Title'].dropna(), "AI Era (2015–2025)")


# Merge kernel and competition data
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
kernels['CreationYear'] = pd.to_datetime(kernels['CreationDate'], errors='coerce').dt.year
kernels_per_year = kernels.groupby('CreationYear').size().reset_index(name='Kernels')

prize_kernels = pd.merge(
    prize_by_year,
    kernels_per_year,
    left_on='Year',
    right_on='CreationYear',
    how='left'
).fillna(0)

fig = px.scatter(
    prize_kernels,
    x='RewardQuantity',
    y='Kernels',
    trendline='ols',
    title='Prize Pool vs Kernel Submissions'
)
fig.show()



fig = px.line(
    prize_kernels,
    x='Year',
    y=['RewardQuantity', 'Kernels'],
    title='Prize Pools & Kernel Submissions Over Time (2010–2025)',
    markers=True
)
fig.update_layout(
    yaxis=dict(title='Value'),
    legend=dict(title='Metrics')
)
fig.show()


# Add competition counts per year
competitions_per_year = competitions.groupby('Year').size().reset_index(name='Competitions')
prize_kernels = prize_kernels.merge(competitions_per_year, on='Year', how='left')

# Compute correlation
corr = prize_kernels[['RewardQuantity', 'Kernels', 'Competitions']].corr()

# Plot as heatmap
fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale='Viridis',
    title='Correlation Heatmap: Prize Pools vs Kernels vs Competitions'
)
fig.show()



kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', usecols=['Id', 'CreationDate'])
kernel_versions['CreationDate'] = pd.to_datetime(kernel_versions['CreationDate'])
kernel_versions['Year'] = kernel_versions['CreationDate'].dt.year

# Count kernels by year
kernel_trend = kernel_versions.groupby('Year').size().reset_index(name='KernelCount')

# Plot
fig = px.line(
    kernel_trend,
    x='Year',
    y='KernelCount',
    markers=True,
    title='Kernel Growth Over Time (AI Tool Influence)'
)
fig.add_vline(x=2021, line_dash="dash", line_color="red", annotation_text="AI Code Assistants Rise", annotation_position="top right")
fig.show()


import plotly.express as px
import plotly.graph_objects as go

# Extract year from EnabledDate
competitions['Year'] = pd.to_datetime(competitions['EnabledDate']).dt.year
yearly_comp = competitions.groupby('Year').size().reset_index(name='Competitions')

# AI milestones
ai_milestones = [
    {"year": 2012, "event": "ImageNet (Deep Learning Breakthrough)"},
    {"year": 2014, "event": "GANs Introduced"},
    {"year": 2017, "event": "Transformers Paper"},
    {"year": 2018, "event": "BERT Released"},
    {"year": 2020, "event": "GPT-3 Launch"},
    {"year": 2022, "event": "ChatGPT Released"}
]

fig = px.line(yearly_comp, x='Year', y='Competitions', title='Kaggle Competitions vs AI Milestones')
for milestone in ai_milestones:
    fig.add_trace(go.Scatter(
        x=[milestone['year']],
        y=[yearly_comp.loc[yearly_comp['Year'] == milestone['year'], 'Competitions'].values[0] if milestone['year'] in yearly_comp['Year'].values else 0],
        mode="markers+text",
        name=milestone['event'],
        text=[milestone['event']],
        textposition="top center"
    ))
fig.show()



comp_type_trend = competitions.groupby(['Year', 'HostSegmentTitle']).size().reset_index(name='Count')

fig = px.area(
    comp_type_trend,
    x='Year',
    y='Count',
    color='HostSegmentTitle',
    title='Shift in Competition Types Over Time',
    line_group='HostSegmentTitle'
)
fig.show()

