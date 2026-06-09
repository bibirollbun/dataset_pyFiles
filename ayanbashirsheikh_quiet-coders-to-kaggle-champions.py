import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown,Math ,Latex
import plotly.express as px
import plotly.graph_objects as go


BASE_PATH = '/kaggle/input/meta-kaggle/'



users = pd.read_csv(BASE_PATH + 'Users.csv')


achievements_sample = pd.read_csv(BASE_PATH + 'UserAchievements.csv', nrows=1000000)




# Merge achievement and user data
full_df = achievements_sample.merge(users, on='Id', how='left')


full_df['TierAchievementDate'] = pd.to_datetime(full_df['TierAchievementDate'], errors='coerce')
full_df['RegisterDate'] = pd.to_datetime(full_df['RegisterDate'], errors='coerce')


# Aggregate medal totals per user
medal_profile = achievements_sample.groupby('UserId')[['TotalGold', 'TotalSilver', 'TotalBronze']].mean()

# Normalize across medal types
medal_profile_norm = medal_profile.div(medal_profile.sum(axis=1), axis=0).fillna(0)


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(medal_profile_norm)

kmeans = KMeans(n_clusters=4, random_state=42)
medal_profile_norm['Cluster'] = kmeans.fit_predict(X_scaled)


import matplotlib.pyplot as plt
import numpy as np

# 1. Prepare cluster centroids
centroids = medal_profile_norm.groupby('Cluster').mean()

# 2. Set up radar chart structure
labels = ['TotalGold', 'TotalSilver', 'TotalBronze']
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # repeat first angle to close the polygon

# 3. Plot radar chart
plt.figure(figsize=(10, 6))
for idx, row in centroids.iterrows():
    values = row.tolist()
    values += values[:1]
    plt.polar(angles, values, label=f'Cluster {idx}', linewidth=2)

# 4. Chart aesthetics
plt.xticks(angles[:-1], labels, fontsize=12)
plt.title(" Contributor Medal Archetypes by Cluster", fontsize=16, color='#2e86de')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Step 1: Calculate cluster centroids
centroids = medal_profile_norm.groupby('Cluster')[['TotalGold', 'TotalSilver', 'TotalBronze']].mean()

# Step 2: Set up radar chart parameters
labels = centroids.columns.tolist()
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

# Step 3: Plot radar chart
plt.figure(figsize=(10, 6), facecolor='white')
for idx, row in centroids.iterrows():
    values = row.tolist()
    values += values[:1]
    plt.polar(angles, values, label=f'Cluster {idx}', linewidth=2)

# Step 4: Aesthetic tuning
plt.xticks(angles[:-1], labels, fontsize=12, color="#34495e")
plt.title(" Contributor Personas via Medal Signature Clustering", fontsize=16, color="#2e86de", pad=20)
plt.yticks(np.linspace(0, 1, 5), fontsize=10, color='gray')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()





# Define only medal columns
medal_cols = ['TotalGold', 'TotalSilver', 'TotalBronze']

def label_cluster(row):
    dominant = row[medal_cols].idxmax()
    if dominant == 'TotalGold':
        return ' Gold Specialists'
    elif dominant == 'TotalSilver':
        return 'Balanced Performers'
    elif dominant == 'TotalBronze':
        return ' Volume Contributors'
    else:
        return ' Undefined'

# Apply safely
centroids_labeled = centroids.copy()
centroids_labeled['ClusterLabel'] = centroids_labeled.apply(label_cluster, axis=1)


import seaborn as sns
import matplotlib.pyplot as plt

# Melt labeled centroids for stacked view
centroids_melted = centroids_labeled.reset_index().melt(
    id_vars=['Cluster', 'ClusterLabel'],
    value_vars=['TotalGold', 'TotalSilver', 'TotalBronze'],
    var_name='MedalType',
    value_name='Proportion'
)

# Plot
plt.figure(figsize=(9,6))
sns.barplot(
    data=centroids_melted,
    x='ClusterLabel',
    y='Proportion',
    hue='MedalType',
    palette={'TotalGold':'gold', 'TotalSilver':'silver', 'TotalBronze':'peru'}
)

# Aesthetics
plt.title(" Contributor Personas: Medal Composition by Cluster", fontsize=16, color='#2e86de')
plt.xlabel("")
plt.ylabel("Normalized Medal Proportion")
plt.xticks(rotation=10, fontsize=12)
plt.legend(title='Medal Type', loc='upper right')
plt.tight_layout()
plt.show()





# Filter for gold medals + discussions
gold_df = achievements_sample[achievements_sample['TotalGold'] > 0][['UserId', 'TotalGold']]
discussion_df = achievements_sample[achievements_sample['AchievementType'] == 'Discussion'][['UserId']]

# Count discussions
discussion_count = discussion_df.value_counts().reset_index()
discussion_count.columns = ['UserId', 'TotalDiscussions']

# Merge achievements
efficiency_core = gold_df.groupby('UserId').sum().reset_index()
efficiency_core = efficiency_core.merge(discussion_count, on='UserId', how='outer').fillna(0)


user_dates = users[['Id', 'RegisterDate']].rename(columns={'Id': 'UserId'})
achievements_sample['TierAchievementDate'] = pd.to_datetime(achievements_sample['TierAchievementDate'])

# Latest contribution per user
latest_dates = achievements_sample.groupby('UserId')['TierAchievementDate'].max().reset_index()
user_lifespan = user_dates.merge(latest_dates, on='UserId', how='left')
user_lifespan['RegisterDate'] = pd.to_datetime(user_lifespan['RegisterDate'])
user_lifespan['ActiveMonths'] = ((user_lifespan['TierAchievementDate'] - user_lifespan['RegisterDate']) / np.timedelta64(1, 'm')).round(1)


efficiency_df = efficiency_core.merge(user_lifespan[['UserId', 'ActiveMonths']], on='UserId', how='left')
efficiency_df['AchievementsPerMonth'] = (efficiency_df['TotalGold'] + efficiency_df['TotalDiscussions']) / (efficiency_df['ActiveMonths'] + 1)




data = efficiency_df['AchievementsPerMonth'].dropna()


plt.figure(figsize=(10,6))
counts, bins, patches = plt.hist(data, bins=40, color='steelblue', alpha=0.7, edgecolor='white')


mean_val = np.mean(data)
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Avg Efficiency: {mean_val:.2f}')


plt.title(" Achievement Efficiency: High-Value Contributions per Active Month", fontsize=15)
plt.xlabel("Gold + Discussions per Month")
plt.ylabel("Contributor Count")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


















merged_df = achievements_sample.merge(users, on='Id', how='left')



# Step 2: Aggregate medal totals per user
medal_summary = merged_df.groupby(['Id', 'UserName'])[['TotalGold', 'TotalSilver', 'TotalBronze']].sum().reset_index()

# Step 3: Compute overall medal count
medal_summary['TotalMedals'] = medal_summary[['TotalGold', 'TotalSilver', 'TotalBronze']].sum(axis=1)

# Optional: sort by top performers
medal_summary = medal_summary.sort_values(by='TotalMedals', ascending=False)


import matplotlib.pyplot as plt

# Select top contributors
top_medals = medal_summary.sort_values('TotalMedals', ascending=False).head(20)
usernames = top_medals['UserName'].astype(str)

# Plot total medals
plt.figure(figsize=(10,8))
plt.barh(usernames, top_medals['TotalMedals'], color='#f1c40f', edgecolor='black')
plt.xlabel("Total Medals Earned")
plt.title(" Total Medal Count by Contributor")
plt.gca().invert_yaxis()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# Data
gold = top_medals['TotalGold']
silver = top_medals['TotalSilver']
bronze = top_medals['TotalBronze']

# Plot stacked medals
plt.figure(figsize=(12,8))
plt.barh(usernames, bronze, color='#cd7f32', label='Bronze')
plt.barh(usernames, silver, left=bronze, color='#c0c0c0', label='Silver')
plt.barh(usernames, gold, left=bronze+silver, color='#ffd700', label='Gold')

plt.xlabel("Medal Count")
plt.title(" Medal Composition by Contributor")
plt.legend()
plt.gca().invert_yaxis()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# Assuming medal_summary contains UserId, UserName, TotalGold, TotalSilver, TotalBronze
medal_personas_df = medal_summary.copy()

# Avoid division by zero
medal_personas_df['TotalMedals'] = medal_personas_df[['TotalGold', 'TotalSilver', 'TotalBronze']].sum(axis=1)
medal_personas_df = medal_personas_df[medal_personas_df['TotalMedals'] > 0]

# Normalize medal types
medal_personas_df['GoldRatio'] = medal_personas_df['TotalGold'] / medal_personas_df['TotalMedals']
medal_personas_df['SilverRatio'] = medal_personas_df['TotalSilver'] / medal_personas_df['TotalMedals']
medal_personas_df['BronzeRatio'] = medal_personas_df['TotalBronze'] / medal_personas_df['TotalMedals']


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_medals = medal_personas_df[['GoldRatio', 'SilverRatio', 'BronzeRatio']]
X_scaled = StandardScaler().fit_transform(X_medals)

kmeans = KMeans(n_clusters=4, random_state=42)
medal_personas_df['MedalCluster'] = kmeans.fit_predict(X_scaled)


def tag_medal_persona(row):
    if row['GoldRatio'] > 0.6:
        return ' Elite Specialist'
    elif row['SilverRatio'] > 0.6:
        return ' Consistent Performer'
    elif row['BronzeRatio'] > 0.6:
        return ' Broad Explorer'
    else:
        return ' Balanced Achiever'

medal_personas_df['MedalPersona'] = medal_personas_df.apply(tag_medal_persona, axis=1)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
medal_personas_df['MedalPersona'].value_counts().plot(kind='barh', color='#3498db', edgecolor='black')
plt.title("Medal Persona Distribution")
plt.xlabel("Number of Contributors")
plt.tight_layout()
plt.show()
































import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx

# Load core dataset
orgs = pd.read_csv('/kaggle/input/meta-kaggle/Organizations.csv', parse_dates=['CreationDate'])
orgs.columns = orgs.columns.str.strip()
orgs = orgs.dropna(subset=['CreationDate'])





import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'notebook_connected'  # âœ… Forces rendering in notebooks

# Prepare data
orgs_sorted = orgs.dropna(subset=['CreationDate']).sort_values('CreationDate')
orgs_sorted['Count'] = range(1, len(orgs_sorted)+1)
orgs_sorted['Year'] = orgs_sorted['CreationDate'].dt.year
orgs_sorted['Decade'] = (orgs_sorted['Year'] // 10) * 10

# Extract year and month
orgs['Year'] = orgs['CreationDate'].dt.year
orgs['Month'] = orgs['CreationDate'].dt.month_name()

# Create pivot table
heatmap_data = orgs.groupby(['Year', 'Month']).size().unstack(fill_value=0)

# Order months
months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
heatmap_data = heatmap_data[months_order]

# Plot
plt.figure(figsize=(14,8))
sns.heatmap(heatmap_data, cmap='Blues', linewidths=0.5, linecolor='gray', annot=True, fmt='d')
plt.title(' Kaggle Organization Onboarding Heatmap by Year and Month', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()





nlp = spacy.load('en_core_web_sm')

def classify_domain(text):
    if pd.isna(text): return 'Unknown'
    text = text.lower()
    if 'university' in text or 'education' in text: return 'Academic'
    elif 'government' in text or 'agency' in text: return 'Gov'
    elif 'health' in text or 'hospital' in text: return 'Medical'
    elif 'data' in text or 'ai' in text or 'tech' in text: return 'Tech'
    elif 'library' in text or 'community' in text: return 'Civic'
    else: return 'Other'

orgs['Domain'] = orgs['Description'].apply(classify_domain)

# View distribution
sns.countplot(x='Domain', data=orgs, order=orgs['Domain'].value_counts().index)
plt.title('Domain Distribution of Organizations')
plt.xticks(rotation=45)
plt.show()


datasets = pd.read_csv('/kaggle/input/meta-kaggle/Datasets.csv')
datasets = datasets[datasets['OwnerOrganizationId'].notna()]
datasets['OwnerOrganizationId'] = datasets['OwnerOrganizationId'].astype(int)

# Merge
org_contrib = orgs.merge(datasets, left_on='Id', right_on='OwnerOrganizationId')
top_orgs = org_contrib.groupby('Name')['Id_y'].count().sort_values(ascending=False).head(10)

# Plot
top_orgs.plot(kind='barh', figsize=(10,6), color='mediumseagreen')
plt.title('Top Dataset-Contributing Organizations')
plt.xlabel('Datasets Contributed')
plt.gca().invert_yaxis()
plt.show()


G = nx.Graph()

for _, row in org_contrib.iterrows():
    G.add_edge(row['Name'], row['Slug'])  # Org name â†” dataset slug

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, node_color='lightblue', edge_color='gray', node_size=400, with_labels=True, font_size=8)
plt.title('Organizationâ€“Dataset Contribution Network')
plt.show()


for org in orgs['Name'].head(5):
    desc = orgs[orgs['Name'] == org]['Description'].values[0]
    print(f"\nâ�¡ï¸� {org} Summary Prompt:")
    print(f"\"Summarize the following organizationâ€™s mission in one sentence:\n{desc}\"")








from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all non-null descriptions into one string
text = ' '.join(orgs['Description'].dropna())

# Generate word cloud
wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='Blues').generate(text)

# Display
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(' Dominant Terms in Organization Descriptions', fontsize=16)
plt.show()


import re

def classify_org_type(row):
    # Combine all semantic fields
    text = f"{row['Slug']} {row['Name']} {row['Description']}".lower()
    
    # Token cleanup
    text = re.sub(r'[^\w\s]', '', text)

    # Mapping rules (priority ordered)
    if re.search(r'\b(nlp|text|language|bert|gpt|translation|document)\b', text):
        return ' Natural Language Processing'
    elif re.search(r'\b(image|vision|cv|object detection|segmentation|pose|camera|video)\b', text):
        return ' Computer Vision'
    elif re.search(r'\b(ml|ai|machine learning|artificial intelligence|deep learning|xgboost|sklearn|modeling)\b', text):
        return ' Machine Learning'
    elif re.search(r'\b(healthcare|medical|hospital|diagnosis|drug|bio|genomic|clinical|protein|covid)\b', text):
        return ' Bio & Health'
    elif re.search(r'\b(finance|stock|trading|banking|economics|risk|portfolio|credit)\b', text):
        return ' Finance'
    elif re.search(r'\b(earth|climate|weather|satellite|agriculture|geospatial|sustainability|carbon)\b', text):
        return ' Earth & Sustainability'
    elif re.search(r'\b(game|gaming|sports|chess|moves|strategy|football|basketball)\b', text):
        return ' Gaming & Sports'
    elif re.search(r'\b(education|school|student|learning|tutor|academy)\b', text):
        return ' Education & Learning'
    elif re.search(r'\b(graph|network|social|recommendation|recommender|link|node)\b', text):
        return ' Network & Recommendation Systems'
    elif re.search(r'\b(robot|autonomous|drone|navigation|control|sensor|actuator)\b', text):
        return ' Robotics & Embedded Systems'
    elif re.search(r'\b(ecommerce|retail|shop|store|sales|amazon|product)\b', text):
        return ' Retail & E-commerce'
    else:
        # Fallback: force classification based on any weak signals in Slug or Name
        if 'ml' in row['Slug'] or 'ai' in row['Slug'] or 'data' in row['Slug']:
            return ' Machine Learning'
        elif 'nlp' in row['Slug'] or 'text' in row['Slug']:
            return ' NLP'
        elif 'vision' in row['Slug']:
            return ' Computer Vision'
        else:
            # If nothing matches, default to ML-heavy domain since it's common in Kaggle orgs
            return ' General AI / Data Science'


orgs['SmartDomain'] = orgs.apply(classify_org_type, axis=1)


# Monthly creation counts by domain
monthly_orgs = (
    orgs.groupby(['MonthCreated', 'SmartDomain'])
    .size()
    .reset_index(name='OrganizationsCreated')
)

# Optional: Select top N domains for readability
top_domains = monthly_orgs['SmartDomain'].value_counts().head(6).index
filtered = monthly_orgs[monthly_orgs['SmartDomain'].isin(top_domains)]

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid')
plt.figure(figsize=(14,7))
sns.lineplot(
    data=filtered,
    x='MonthCreated',
    y='OrganizationsCreated',
    hue='SmartDomain',
    marker='o',
    palette='Set2'
)

plt.title("Monthly Organization Creation by Domain (Semantic NLP Classification)", fontsize=16, color='#2c3e50')
plt.xlabel("Month", fontsize=12)
plt.ylabel("Organizations Created", fontsize=12)
plt.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()





import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Step 1: Get monthly organization creation counts
monthly_counts = orgs.groupby('MonthCreated').size()

# Step 2: Convert to time series with proper datetime index
monthly_counts.index = pd.to_datetime(monthly_counts.index)
monthly_counts = monthly_counts.sort_index()

# Step 3: Resample to ensure monthly frequency (fill missing months with 0)
monthly_counts = monthly_counts.asfreq('MS').fillna(0)

# Step 4: Perform additive decomposition
decomposition = seasonal_decompose(monthly_counts, model='additive')

# Step 5: Plot components
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.suptitle("Additive Seasonal Decomposition of Organization Creation", fontsize=16, weight='bold', color='#2c3e50')
plt.tight_layout()
plt.show()


from prophet import Prophet

df = monthly_counts.reset_index().rename(columns={'MonthCreated': 'ds', 0: 'y'})
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)

model.plot(forecast)


!pip install ruptures


import ruptures as rpt

signal = monthly_counts.values
algo = rpt.Pelt(model="rbf").fit(signal)
breaks = algo.predict(pen=10)

rpt.show.display(signal, breaks)


from statsmodels.tsa.stattools import adfuller

result = adfuller(monthly_counts)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')


from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA to differenced data
model = ARIMA(monthly_counts, order=(1,1,1))  # or tune p,d,q via AIC/BIC
results = model.fit()

# Plot forecast
forecast = results.get_forecast(steps=12)
pred_ci = forecast.conf_int()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
monthly_counts.plot(label='Observed', color='#34495e')
forecast.predicted_mean.plot(label='Forecast', color='#2980b9')
plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='skyblue', alpha=0.3)

plt.title(" Forecasting Organization Creation with ARIMA", fontsize=15)
plt.xlabel("Month")
plt.ylabel("Orgs Created")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()




