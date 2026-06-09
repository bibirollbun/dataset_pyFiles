import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to Meta-Kaggle dataset files:", MK_PATH)



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')



competitions.head()


competitions.tail()


competitions.shape


competitions.columns


competitions.info()


competitions.describe()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (adjust path if needed)
df = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")

# Convert dates and create derived columns
df['EnabledDate'] = pd.to_datetime(df['EnabledDate'], errors='coerce')
df['DeadlineDate'] = pd.to_datetime(df['DeadlineDate'], errors='coerce')
df['Year'] = df['EnabledDate'].dt.year
df['DurationDays'] = (df['DeadlineDate'] - df['EnabledDate']).dt.days



import seaborn as sns
import matplotlib.pyplot as plt

# Prepare data
competitions_per_year = df['Year'].value_counts().sort_index()
years = competitions_per_year.index.astype(str)
counts = competitions_per_year.values

# Create color palette (you can try 'coolwarm', 'viridis', 'magma', etc.)
colors = sns.color_palette("viridis", len(competitions_per_year))

# Plot
plt.figure(figsize=(14, 6))
bars = plt.bar(years, counts, color=colors)

# Title & labels
plt.title("ğŸ“… Number of Competitions per Year", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Number of Competitions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Prepare data
top_rewards = df[['Title', 'RewardQuantity', 'RewardType']].dropna()
top_rewards = top_rewards.sort_values(by='RewardQuantity', ascending=False).head(10)

# Set wider figure
plt.figure(figsize=(14, 8))  # wider and taller

# Plot
sns.barplot(x='RewardQuantity', y='Title', data=top_rewards, hue='RewardType', dodge=False)

# Title & labels
plt.title("ğŸ�† Top 10 Highest Rewarded Competitions", fontsize=16)
plt.xlabel("Reward Amount (in Millions USD)")
plt.ylabel("Competition")

# Format x-axis to show millions
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))

# Move legend outside
plt.legend(title="Reward Type", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.show()





metric_counts = df['EvaluationAlgorithmName'].value_counts().head(10)
sns.barplot(x=metric_counts.values, y=metric_counts.index, palette="viridis")
plt.title("ğŸ“� Most Common Evaluation Metrics")
plt.xlabel("Count")
plt.ylabel("Metric")
plt.show()


duration_stats = df['DurationDays'].describe()
print("â�±ï¸� Competition Duration Stats (in Days):")
print(duration_stats)


sns.histplot(df['DurationDays'].dropna(), bins=30, kde=True)
plt.title("â�±ï¸� Distribution of Competition Durations")
plt.xlabel("Duration (Days)")
plt.ylabel("Frequency")
plt.show()


most_competitive = df.sort_values(by='TotalCompetitors', ascending=False).head(10)
sns.barplot(x='TotalCompetitors', y='Title', data=most_competitive, palette="rocket")
plt.title("ğŸ‘¥ Most Competitive Competitions (By Competitors)")
plt.xlabel("Total Competitors")
plt.ylabel("Competition")
plt.show()


top_hosts = df['HostSegmentTitle'].value_counts().drop('Other', errors='ignore').head(10)
sns.barplot(x=top_hosts.values, y=top_hosts.index, palette="coolwarm")
plt.title("ğŸ�¢ Top Competition Hosts")
plt.xlabel("Number of Competitions")
plt.ylabel("Host Organization")
plt.show()


kernel_only_counts = df['OnlyAllowKernelSubmissions'].value_counts()
labels = ['File Upload Allowed', 'Kernel-Only']
plt.pie(kernel_only_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999'])
plt.title("ğŸ’» Share of Kernel-Only Competitions")
plt.axis('equal')
plt.show()


million_dollar_prizes = df[df['RewardQuantity'] >= 1_000_000][['Title', 'RewardQuantity', 'HostSegmentTitle', 'Year']]
print("ğŸ’° Competitions with $1M+ in Prizes:")
print(million_dollar_prizes.sort_values(by='RewardQuantity', ascending=False).to_string(index=False))


team_size_trend = df.groupby('Year')['MaxTeamSize'].mean().dropna()
team_size_trend.plot(marker='o', linestyle='-', color='purple')
plt.title("ğŸ‘¥ Average Max Team Size by Year")
plt.xlabel("Year")
plt.ylabel("Average Max Team Size")
plt.grid(True)
plt.show()


leaderboard_counts = df['HasLeaderboard'].value_counts()
plt.pie(leaderboard_counts, labels=['With Leaderboard', 'Without Leaderboard'], autopct='%1.1f%%', startangle=140, colors=['#8fd9b6','#ffcccb'])
plt.title("ğŸ“ˆ Competitions With vs Without Leaderboards")
plt.axis('equal')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Filter out competitions with missing or zero values in either field
team_submission_data = df[(df['MaxTeamSize'] > 0) & (df['TotalSubmissions'] > 0)]

# Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=team_submission_data, x='MaxTeamSize', y='TotalSubmissions', hue='RewardType', alpha=0.7)

# Labels & title
plt.title("ğŸ‘¥ Team Size vs Total Submissions per Competition", fontsize=16)
plt.xlabel("Max Team Size")
plt.ylabel("Total Submissions")
plt.legend(title="Reward Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Load and preprocess
df = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")
df['EnabledDate'] = pd.to_datetime(df['EnabledDate'], errors='coerce')
df['DeadlineDate'] = pd.to_datetime(df['DeadlineDate'], errors='coerce')
df['DurationDays'] = (df['DeadlineDate'] - df['EnabledDate']).dt.days

# Features to use
features = ['MaxTeamSize', 'RewardQuantity', 'DurationDays', 'TotalSubmissions', 'TotalCompetitors']
plot_df = df[features + ['RewardType', 'Title']].dropna()
plot_df = plot_df[plot_df['TotalSubmissions'] > 0]  # remove competitions with no activity

# Standardize features
X = StandardScaler().fit_transform(plot_df[features])

# PCA transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plot_df['PCA-1'] = X_pca[:, 0]
plot_df['PCA-2'] = X_pca[:, 1]

# Bubble size (scaled & log transformed)
plot_df['Size'] = np.log1p(plot_df['TotalSubmissions']) * 30

# Plot
plt.figure(figsize=(14, 9))
bubble = sns.scatterplot(
    data=plot_df,
    x='PCA-1',
    y='PCA-2',
    hue='RewardType',
    size='Size',
    sizes=(50, 1000),
    alpha=0.75,
    palette='Set2',
    edgecolor='white',
    linewidth=0.6,
    legend='brief'  # Avoid full-size bubble scale
)

# Titles and labels
plt.title("ğŸ§  Clustering of Kaggle Competitions by Metadata (PCA Projection)", fontsize=18)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Fix legend (remove size mapping)
handles, labels = bubble.get_legend_handles_labels()
new_handles = [h for h in handles if not str(h).startswith('<matplotlib.collections.PathCollection')]
new_labels = [l for h, l in zip(handles, labels) if not str(h).startswith('<matplotlib.collections.PathCollection')]
plt.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Reward Type")

plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")

# Convert dates
df['EnabledDate'] = pd.to_datetime(df['EnabledDate'], errors='coerce')
df['DeadlineDate'] = pd.to_datetime(df['DeadlineDate'], errors='coerce')
df['DurationDays'] = (df['DeadlineDate'] - df['EnabledDate']).dt.days

# Select features for correlation
anatomy_features = ['MaxTeamSize', 'RewardQuantity', 'DurationDays', 'TotalSubmissions', 'TotalCompetitors']
anatomy_df = df[anatomy_features].dropna()

# Compute Spearman correlation matrix
correlation_matrix = anatomy_df.corr(method='spearman')

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title("Competition Anatomy: What Drives Engagement?", fontsize=14)
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")

# Prepare date & duration
df['EnabledDate'] = pd.to_datetime(df['EnabledDate'], errors='coerce')
df['DeadlineDate'] = pd.to_datetime(df['DeadlineDate'], errors='coerce')
df['DurationDays'] = (df['DeadlineDate'] - df['EnabledDate']).dt.days

# Prepare data
efficiency_df = df[['Title', 'RewardQuantity', 'TotalSubmissions', 'DurationDays']].dropna()
efficiency_df = efficiency_df[efficiency_df['TotalSubmissions'] > 0]
efficiency_df = efficiency_df[efficiency_df['DurationDays'] > 0]

# Calculate Engagement Score
efficiency_df['EngagementScore'] = efficiency_df['TotalSubmissions'] / (efficiency_df['RewardQuantity'] + 1)
efficiency_df['EngagementScore'] = np.clip(efficiency_df['EngagementScore'], 0, 5000)  # prevent outlier distortion

# Plot
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    data=efficiency_df,
    x='DurationDays',
    y='EngagementScore',
    hue='EngagementScore',
    palette='viridis',
    alpha=0.8,
    edgecolor='white',
    s=80,  # uniform circle size
    linewidth=0.4
)

# Titles & labels
plt.title("ğŸ�¯ Engagement Efficiency vs. Duration", fontsize=18, pad=15)
plt.suptitle("Which Kaggle Competitions Drove the Most Participation per Dollar?", fontsize=12, y=0.91)
plt.xlabel("Competition Duration (Days)", fontsize=12)
plt.ylabel("Engagement Score\n(Total Submissions / Reward Quantity)", fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Colorbar as legend substitute
norm = plt.Normalize(efficiency_df['EngagementScore'].min(), efficiency_df['EngagementScore'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label("Engagement Score", rotation=270, labelpad=15)

plt.tight_layout()
plt.show()


