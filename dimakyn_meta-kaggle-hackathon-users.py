import os

import pandas as pd

from tqdm import tqdm

import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go

from IPython.display import HTML


import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to Meta-Kaggle dataset files:", MK_PATH)
print("Path to Meta-Kaggle-Code dataset files:", MKC_PATH)


USERS_FILES = ["Users.csv", "UserAchievements.csv", "UserOrganizations.csv", "UserFollowers.csv", "Teams.csv", "TeamMemberships.csv", "Organizations.csv"]





df_users = pd.read_csv(f"{MK_PATH}/Users.csv") 


print(df_users.shape)
df_users.head()


df_users['RegisterDate'] = pd.to_datetime(df_users['RegisterDate'], errors='coerce')
df_users = df_users.dropna(subset=['RegisterDate'])

# Extract year and month
df_users['Year'] = df_users['RegisterDate'].dt.year
df_users['Month'] = df_users['RegisterDate'].dt.month

# Group by year and month
monthly_counts = df_users.groupby(['Year', 'Month']).size().reset_index(name='UserCount')

# Create list of available years
years = sorted(monthly_counts['Year'].unique())


print(df_users.shape)
df_users.head()


# Create traces for each year
fig = go.Figure()

for i, year in enumerate(years):
    df = monthly_counts[monthly_counts['Year'] == year]
    fig.add_trace(go.Bar(
        x=df['Month'],
        y=df['UserCount'],
        name=str(year),
        visible=(i == 0)  # only the first year is visible by default
    ))

# Create dropdown menu
dropdown_buttons = [
    dict(label=str(year),
         method="update",
         args=[{"visible": [year == y for y in years]},
               {"title": f"Users per Month in {year} (Total: {monthly_counts[monthly_counts['Year'] == year].UserCount.sum()} users)",
                "xaxis": {"title": "Month"},
                "yaxis": {"title": "User Count"}}])
    for year in years
]

fig.update_layout(
    updatemenus=[
        dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=1.05,
            xanchor="left",
            y=1.2,
            yanchor="top"
        )
    ],
    title=f"Users per Month in {years[0]}",
    xaxis_title="Month",
    yaxis_title="User Count", 
    height=800
)

# fig.show()

# ✅ Save as standalone HTML file
fig.write_html("users_by_year.html", include_plotlyjs='cdn')

# HTML(filename="/kaggle/working/users_by_year.html")
# py.iplot(fig)
fig.show(renderer='iframe') 



df_users.head()


df_users_1 = df_users[df_users["Country"].notna()]
df_users_1.shape


# Group by year and country
user_counts = df_users_1.groupby(['Year', 'Country']).size().reset_index(name='UserCount')

# Pivot to have countries as columns
pivot = user_counts.pivot(index='Year', columns='Country', values='UserCount').fillna(0)

# Compute cumulative sum
cumsum_df = pivot.cumsum().reset_index()

# Melt back to long format
country_year_cumsum = cumsum_df.melt(id_vars='Year', var_name='Country', value_name='UserCount')

# Optional: filter to top countries by total users
top_countries = (
    country_year_cumsum.groupby('Country')['UserCount'].max()
    .sort_values(ascending=False)
    .head(20)  # top 10 for clarity/speed
    .index.tolist()
)
country_year_cumsum = country_year_cumsum[country_year_cumsum['Country'].isin(top_countries)]


fig = px.bar(
    country_year_cumsum,
    x='UserCount',
    y='Country',
    color='Country',
    orientation='h',
    animation_frame='Year',
    title='Kaggle Users per Country by Year',
    range_x=[0, country_year_cumsum["UserCount"].max() * 1.2],
    height=600
)

fig.update_layout(yaxis={'categoryorder':'total ascending'})
# fig.show()

fig.write_html("users_per_month_by_year.html", include_plotlyjs='cdn')

# HTML(filename="/kaggle/working/users_per_month_by_year.html")
# py.iplot(fig)
fig.show(renderer='iframe')






df_users = pd.read_csv(f"{MK_PATH}/Users.csv") 
df_users = df_users[df_users["PerformanceTier"] > 0]
df_users.shape


users_ids = df_users["Id"].to_list()


file_path = f"{MK_PATH}/UserAchievements.csv"


with open(file_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)


filtered_rows = []

chunk_size = 100_000
n_chunks = total_lines // chunk_size + 1

filtered_rows = []
for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), total=n_chunks, desc="Processing CSV"):
    matched = chunk[chunk['UserId'].isin(users_ids)]
    filtered_rows.append(matched)

df_users_achiev = pd.concat(filtered_rows, ignore_index=True)

print(df_users_achiev)
df_users_achiev.head()


df_users_funny = pd.read_csv("/kaggle/input/meta-kaggle-hackathon-get-funny-names/funny_scores.csv")
df_users_funny


df_users_with_funny = df_users[["Id", "UserName"]].merge(df_users_funny, on="UserName", how="inner")


df_users_achiev_with_funny = df_users_achiev.merge(df_users_with_funny, left_on="UserId", right_on="Id", how="inner")


df_users_achiev_with_funny["AchievementType"].value_counts()


df_users_achiev_with_funny["TopLabel"].value_counts()


df = df_users_achiev_with_funny.copy()

# Define ranking bins
def rank_group(rank):
    if pd.isna(rank):
        return "Unranked"
    elif rank <= 20:
        return "1–20"
    elif rank <= 100:
        return "21–100"
    elif rank <= 500:
        return "101–500"
    elif rank <= 1000:
        return "501–1000"
    else:
        return "1001+"

df['RankingGroup'] = df['HighestRanking'].apply(rank_group)

allowed_groups = ["1–20", "21–100", "101–500"]
df_filtered = df[df['RankingGroup'].isin(allowed_groups)]

grouped = (
    df_filtered
    .groupby(['TopLabel', 'RankingGroup'])
    .size()
    .reset_index(name='UserCount')
)

grouped['RankingGroup'] = pd.Categorical(grouped['RankingGroup'],
                                         categories=allowed_groups)

# Plot
fig = px.bar(
    grouped,
    x='RankingGroup',
    y='UserCount',
    color='TopLabel',
    barmode='group',
    title='User Count in Top Ranking Groups by TopLabel'
)

fig.update_layout(
    xaxis_title='Ranking Group',
    yaxis_title='Number of Users',
    height=600,
    width=1000
)

fig.show(renderer='iframe')


df = df_users_achiev.copy()
df['TierAchievementDate'] = pd.to_datetime(df['TierAchievementDate'], errors='coerce')
df['Year'] = df['TierAchievementDate'].dt.year

metrics = ['Points', 'TotalGold', 'TotalSilver', 'TotalBronze']
achievement_types = df['AchievementType'].dropna().unique()
years = sorted(df['Year'].dropna().unique())

fig = go.Figure()
n_types = len(achievement_types)

for i, metric in enumerate(metrics):
    grouped = (
        df.groupby(['AchievementType', 'Year'])[metric]
        .sum()
        .reset_index()
    )

    for ach_type in achievement_types:
        data = grouped[grouped['AchievementType'] == ach_type]
        fig.add_trace(go.Bar(
            x=data['Year'],
            y=data[metric],
            name=ach_type,
            visible=(i == 0) 
        ))

buttons = []
for i, metric in enumerate(metrics):
    visible = [False] * (n_types * len(metrics))
    for j in range(n_types):
        visible[i * n_types + j] = True

    buttons.append(dict(
        label=metric,
        method="update",
        args=[{"visible": visible},
              {"title": f"Bar Plot of {metric} by Year and AchievementType",
               "yaxis": {"title": metric}}]
    ))

fig.update_layout(
    title="Bar Plot of Points by Year and AchievementType",
    xaxis_title="Year",
    yaxis_title="Points",
    barmode='group',
    updatemenus=[dict(
        buttons=buttons,
        active=0,
        direction="down",
        x=1.1,
        y=1.2,
        showactive=True
    )],
    height=600,
    width=1000
)

fig.show(renderer='iframe') 





