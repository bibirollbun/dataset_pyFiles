import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style='whitegrid', palette='muted', font_scale=1.1)

# List of input files for reference
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# Competitions and tags for domain analysis
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
competition_tags = pd.read_csv('/kaggle/input/meta-kaggle/CompetitionTags.csv')
tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')

# Kernels and tags for technique analysis
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
kernel_languages = pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')



# Merge competition tags with tag names
comp_tags = competition_tags.merge(tags, left_on='TagId', right_on='Id')

# Example: Set of keywords per domain (expand as needed)
domain_keywords = {
    'Computer Vision': ['image', 'vision', 'cv'],
    'NLP': ['nlp', 'text', 'language', 'translation', 'bert', 'transformer', 'qa'],
    'Tabular': ['tabular', 'classification', 'regression'],
    'Time Series': ['time series', 'forecast', 'signal'],
    'Reinforcement Learning': ['rl', 'reinforcement'],
    'Audio': ['audio', 'speech', 'sound', 'music'],
    'Recommendation': ['recommend', 'recommendation', 'recommender']
}

# Assign domain to each competition
def map_domain(tag_list):
    tags = " ".join(tag_list).lower()
    for domain, keywords in domain_keywords.items():
        if any(kw in tags for kw in keywords):
            return domain
    return 'Other'

# Aggregate competition tags
comp_tag_groups = comp_tags.groupby('CompetitionId')['Name'].apply(list).reset_index()
comp_domains = competitions[['Id', 'Title', 'EnabledDate']].merge(
    comp_tag_groups, left_on='Id', right_on='CompetitionId', how='left'
)
comp_domains['Domain'] = comp_domains['Name'].apply(lambda x: map_domain(x) if isinstance(x, list) else 'Other')
comp_domains['Year'] = pd.to_datetime(comp_domains['EnabledDate'], errors='coerce').dt.year



domain_year = comp_domains.groupby(['Year', 'Domain']).size().unstack(fill_value=0)
domain_year.plot(kind='bar', stacked=True, figsize=(16,7), colormap='tab20')
plt.title('Number of Competitions per AI Domain Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.legend(title='Domain', bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.show()



kv_comp = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv')




import pandas as pd

pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', nrows=0).columns.tolist()




# Load only necessary columns
kernel_versions = pd.read_csv(
    '/kaggle/input/meta-kaggle/KernelVersions.csv',
    usecols=['Id', 'CreationDate'],
    parse_dates=['CreationDate']
)

# Filter kernels created in 2016 or later
kernel_versions = kernel_versions[kernel_versions['CreationDate'] >= '2016-01-01']
kernel_versions['Year'] = kernel_versions['CreationDate'].dt.year

print(f'âœ… Kernel versions loaded after 2015: {len(kernel_versions)}')
kernel_versions.head()





# Load KernelTags and Tags
ktags = kernel_tags.merge(tags, left_on='TagId', right_on='Id')





# Merge kernel tags with kernel versions
ktags_full = ktags.merge(kernel_versions, left_on='KernelId', right_on='Id', how='inner')



# List of machine learning techniques to track
tech_keywords = [
    'xgboost', 'lightgbm', 'catboost',
    'cnn', 'rnn', 'lstm', 'transformer',
    'bert', 'gpt', 'random forest', 
    'ensemble', 'gan', 'tabnet'
]

# Function to extract techniques from tag names
def map_tech(name):
    name = str(name).lower()
    return [kw for kw in tech_keywords if kw in name]

# Apply and extract techniques for each kernel tag
ktags_full['Technique'] = ktags_full['Name'].apply(map_tech)

# Explode so each row has 1 technique
tech_trends = ktags_full.explode('Technique').dropna(subset=['Technique'])



# Pivot technique usage grouped by Year
tech_pivot = tech_trends.groupby(['Year', 'Technique']).size().unstack(fill_value=0)

# Optional: Show only techniques with significant usage over time
top_techniques = tech_pivot.sum().sort_values(ascending=False).head(8).index
tech_plot_data = tech_pivot[top_techniques]



import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
tech_plot_data.plot(marker='o', linewidth=2)

plt.title('ğŸ“ˆ Evolution of Machine Learning Techniques in Kaggle Notebooks (2016â€“2024)')
plt.xlabel('Year')
plt.ylabel('Number of Notebooks Using the Technique')
plt.legend(title='Technique', loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()





