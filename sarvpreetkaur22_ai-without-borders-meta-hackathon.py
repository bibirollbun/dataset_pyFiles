#Installing external packacges
!pip install pycountry
!pip install nbformat
!pip install ipython


# Data manipulation
import pandas as pd
import numpy as np
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import plotly.io as pio

# NLP / Text Preprocessing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Display settings
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")



users = pd.read_csv('/kaggle/input/meta-kaggle-users-cleaned-dataset/users_clean.csv')
teams = pd.read_csv('/kaggle/input/meta-kaggle-dataset-teams-cleaned/teams_clean.csv')
user_achievements = pd.read_csv('/kaggle/input/meta-kaggle-dataset-user-achievements-cleaned/user_achievements_clean (1).csv')
team_members = pd.read_csv('/kaggle/input/meta-kaggle-dataset-team-members-cleaned/team_members_clean.csv')
competitions = pd.read_csv('/kaggle/input/meta-kaggle-competitions-cleaned-dataset/competitions_clean.csv')
kernels = pd.read_csv('/kaggle/input/meta-kaggle-scripts-cleaned-dataset/scripts_clean.csv')


users['Country'] = users['Country'].fillna('Unknown').str.strip()

# Who has been on a competition team
participant_user_ids = team_members['UserId'].unique()
users_participants = users[users['UserId'].isin(participant_user_ids)].copy()

# Join users with achievements (medals)
users_with_medals = users_participants.merge(
    user_achievements, how='left', on='UserId'
)

# Join with team memberships (for analysis by team or competition)
users_with_teams = users_with_medals.merge(
    team_members, how='left', on='UserId'
)

# Join with teams to get submission date and comp info
full_df = users_with_teams.merge(
    teams, how='left', left_on='TeamId', right_on='Id', suffixes=('', '_Team')
)

full_df.head(1000)


# Convert RegisterDate to datetime
users['JoinDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')
users['JoinYear'] = users['JoinDate'].dt.year

# Group data
users_by_year_country = users.groupby(['JoinYear', 'Country'])['UserId'].nunique().reset_index()

# Create Plotly figure
fig = px.line(users_by_year_country, x='JoinYear', y='UserId', color='Country',
              title='Kaggle Participation Over Time by Country')

# Show chart using iframe-connected renderer
fig.show(renderer="iframe_connected")



# Prepare your data
users_by_country = users.groupby('Country')['UserId'].nunique().reset_index()

# Create choropleth
fig = px.choropleth(users_by_country,
                    locations='Country',
                    locationmode='country names',
                    color='UserId',
                    title='Total Kaggle Users per Country',
                    color_continuous_scale='Viridis')

# Use iframe_connected renderer for compatibility
fig.show(renderer="iframe_connected")



user_achievements.columns
user_achievements['AchievementType'].dropna().unique()


user_achievements.head()
user_achievements.info()



user_achievements.head()


# Reset the index to flatten the DataFrame
user_achievements = user_achievements.reset_index()

# Now you should see columns: UserId, TotalGold, TotalSilver, TotalBronze
user_achievements.head()



# Skip the first row (extra header) when reading
user_achievements = pd.read_csv('/kaggle/input/meta-kaggle-dataset-user-achievements-cleaned/user_achievements_clean (1).csv', skiprows=1)



user_achievements.head()


#  Drop any header-like rows (e.g., string headers repeated inside the data)
user_achievements = user_achievements[user_achievements['UserId'].apply(lambda x: str(x).isdigit())]

#  Convert columns to integers safely
user_achievements['UserId'] = user_achievements['UserId'].astype(int)
user_achievements['TotalGold'] = pd.to_numeric(user_achievements['TotalGold'], errors='coerce').fillna(0).astype(int)
user_achievements['TotalSilver'] = pd.to_numeric(user_achievements['TotalSilver'], errors='coerce').fillna(0).astype(int)
user_achievements['TotalBronze'] = pd.to_numeric(user_achievements['TotalBronze'], errors='coerce').fillna(0).astype(int)



# Merge medal counts with users
users_medals = pd.merge(users, user_achievements, on='UserId', how='left')

# Replace NaNs in medal columns with 0s
users_medals[['TotalGold', 'TotalSilver', 'TotalBronze']] = users_medals[['TotalGold', 'TotalSilver', 'TotalBronze']].fillna(0).astype(int)

# Calculate total medals
users_medals['TotalMedals'] = users_medals[['TotalGold', 'TotalSilver', 'TotalBronze']].sum(axis=1)



# Group medal totals by country
medals_by_country = users_medals.groupby('Country')[['TotalGold', 'TotalSilver', 'TotalBronze', 'TotalMedals']].sum().sort_values(by='TotalMedals', ascending=False)

# Show top 15
medals_by_country.head(15)



# Number of users per country
users_per_country = users.groupby('Country')['UserId'].count()

# Merge with medals
medals_by_country['TotalUsers'] = users_per_country
medals_by_country['MedalEfficiency'] = medals_by_country['TotalMedals'] / medals_by_country['TotalUsers']

# Fill NaNs if any (in case some countries had medals but users got filtered)
medals_by_country = medals_by_country.fillna(0)

# Sort by efficiency
medals_by_efficiency = medals_by_country.sort_values('MedalEfficiency', ascending=False)

# Show top efficient countries
medals_by_efficiency.head(15)



## Plotting
top_countries = medals_by_country.head(15)

top_countries[['TotalGold', 'TotalSilver', 'TotalBronze']].plot(
    kind='bar', stacked=True, figsize=(12, 6),
    title='Top 15 Countries by Total Medals on Kaggle'
)
plt.ylabel('Number of Medals')
plt.xlabel('Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



## Heatmap of medal efficiency

plt.figure(figsize=(12, 6))
sns.heatmap(medals_by_efficiency[['MedalEfficiency']].head(20), annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Top 20 Countries by Medal Efficiency')
plt.show()



def plot_engagement_heatmap(kernels_df, users_df, country_list, title_suffix=""):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Merge kernels with country
    kernels_with_country = kernels_df.merge(
        users_df[['UserId', 'Country']],
        left_on='AuthorUserId',
        right_on='UserId',
        how='left'
    )

    # Filter kernels to the specified countries
    filtered_kernels = kernels_with_country[
        kernels_with_country['Country'].isin(country_list)
    ]

    # Compute average engagement metrics
    country_engagement_stats = (
        filtered_kernels
        .groupby('Country')[['TotalVotes', 'TotalComments', 'TotalViews']]
        .mean()
        .loc[country_list]  # keep the order of input list
    )

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        country_engagement_stats,
        annot=True, fmt=".1f", cmap="coolwarm"
    )
    plt.title(f"ğŸ“Š Average Notebook Engagement by Country {title_suffix}")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()

    # Return the engagement stats as a dataframe
    return country_engagement_stats



# Get list of top 15 countries by medals
top_countries_list = top_countries.index

# Run the function
engagement_stats_df = plot_engagement_heatmap(
    kernels,
    users,
    top_countries_list,
    title_suffix="(Top 15 by Medals)"
)

# See the numbers too
engagement_stats_df


# Merge kernels with users to get country information
kernels_with_country = kernels.merge(users[['UserId', 'Country']], left_on='AuthorUserId', right_on='UserId', how='left')
kernels_with_country


tools = [
'xgboost', 'lightgbm', 'pytorch', 'tensorflow', 'keras', 'catboost',
'seaborn', 'matplotlib', 'sklearn', 'numpy', 'pandas', 'nltk', 'spacy',
'plotly', 'altair', 'statsmodels', 'fastai'
]


stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text): return ''
    text = re.sub(r'\W+', ' ', str(text).lower())
    return ' '.join([word for word in text.split() if word not in stop_words])

kernels_with_country = kernels.merge(users[['UserId', 'Country']], left_on='AuthorUserId', right_on='UserId', how='left')

kernels_with_country['CleanTitle'] = kernels_with_country['CurrentUrlSlug'].apply(clean_text)

# Drop rows where Country is missing
df = kernels_with_country.dropna(subset=['CleanTitle', 'Country']).copy()

# Create binary flags for each tool in CleanTitle
for tool in tools:
    df[tool] = df['CleanTitle'].str.contains(fr'\b{tool}\b', case=False, na=False)


# Group by Country and sum tool mentions
tool_usage_by_country = df.groupby('Country')[tools].sum()
tool_usage_by_country.reset_index(inplace=True)
tool_usage_by_country


tool_usage_by_country.to_csv("/kaggle/working/popular_tools.csv", index=False)
print(tool_usage_by_country.columns.tolist())



# Filter for Top 15 countries by medals (top_countries is your previous output)
tool_usage_by_country_copy = tool_usage_by_country.copy()
tool_usage_by_country_copy = tool_usage_by_country_copy.set_index('Country')
top_15_countries = top_countries.index[:15]
tool_usage_top_15 = tool_usage_by_country_copy.loc[top_15_countries]
tool_usage_plot = tool_usage_top_15.T


plt.figure(figsize=(22, 10))
tool_usage_plot.plot(kind='bar', figsize=(22, 10), colormap='tab20')

plt.title('Popular Tools in Top 15 Countries')
plt.ylabel('Number of Notebooks Mentioning the Tool')
plt.xlabel('Tools / Libraries')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Use only top 15 countries from medal count
top_15_keywords = tool_usage_by_country_copy.loc[top_15_countries]

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(top_15_keywords)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(reduced)

# Visualize the clusters
plt.figure(figsize=(12, 7))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='Set1', s=120, edgecolor='black')


kernels['CleanTitle'] = kernels['CurrentUrlSlug'].astype(str).apply(clean_text)
kernels_with_country = kernels.merge(users[['UserId', 'Country']], left_on='AuthorUserId', right_on='UserId', how='left')
vectorizer = CountVectorizer(max_features=100)
X = vectorizer.fit_transform(kernels_with_country['CleanTitle'])


keywords_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
keywords_df['Country'] = kernels_with_country['Country']
keywords_df


print(keywords_df.columns.to_list())
keywords_df.to_csv("/kaggle/working/top_modeling_keywords.csv", index=False)



# Step 1: Group and sum keyword frequencies by country
country_keywords = keywords_df.groupby('Country').sum()

# Step 2: Calculate total keyword count per country
country_keywords['TotalKeywordCount'] = country_keywords.sum(axis=1)

# Step 3: Select top 15 countries by total keyword usage
top_countries_df = country_keywords.sort_values('TotalKeywordCount', ascending=False).head(15).drop(columns='TotalKeywordCount')

# Step 4: From those countries, extract total frequency per keyword
top_keywords = top_countries_df.sum(axis=0).sort_values(ascending=False).head(20).index

# Step 5: Filter to keep only those top 20 keywords
filtered_df = top_countries_df[top_keywords]

# Step 6: Transpose for plotting
selected_keywords = filtered_df.T

# Step 7: Plot
plt.figure(figsize=(18, 7))
selected_keywords.plot(kind='bar', figsize=(18, 7), colormap='viridis')

plt.title('Top 20 Keywords in Top 15 Countries by Modeling Frequency')
plt.ylabel('Frequency')
plt.xlabel('Keyword')
plt.xticks(rotation=75)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Estimate notebook title length (from CurrentUrlSlug)
kernels['NotebookLength'] = kernels['CurrentUrlSlug'].astype(str).apply(lambda x: len(str(x).split('-')) if pd.notnull(x) else 0)

# Merge with users to get Country
kernels_with_country = kernels.merge(users[['UserId', 'Country']], left_on='AuthorUserId', right_on='UserId', how='left')

#  Extract Year from CreationDate
kernels_with_country['Year'] = pd.to_datetime(kernels_with_country['CreationDate'], errors='coerce').dt.year

# Filter to top medal-winning countries
kernels_with_country = kernels_with_country[kernels_with_country['Country'].isin(top_15_countries)]

# Group by Year & Country and compute average title token count
trend_df = kernels_with_country.groupby(['Year', 'Country'])['NotebookLength'].mean().reset_index()

trend_df


trend_df.to_csv("/kaggle/working/notebook_token_trends.csv", index=False)



plt.figure(figsize=(14, 6))
sns.lineplot(data=trend_df, x='Year', y='NotebookLength', hue='Country', marker='o', linewidth=2.2)
plt.title('Average Notebook Title Token Count Over Time (Top 15 Countries)', fontsize=14)
plt.ylabel('Title Length (Tokens)')
plt.xlabel('Year')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

