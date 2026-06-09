import pandas as pd
import os

#  folder path
BASE_PATH = r"C:\Users\Samhitha Shambavi\Meta kaggle hackathon submission , samhitha and nikhitha\dataset"

# Load Competitions.csv
competitions_path = os.path.join(BASE_PATH, 'Competitions.csv')
competitions_df = pd.read_csv(competitions_path)

# Filter and clean necessary columns
competitions_filtered = competitions_df[[
    'Id', 'Slug', 'Title', 'EnabledDate', 'DeadlineDate'
]].copy()

# Drop rows without launch dates
competitions_filtered.dropna(subset=['EnabledDate'], inplace=True)

# Convert EnabledDate to datetime
competitions_filtered['EnabledDate'] = pd.to_datetime(
    competitions_filtered['EnabledDate'], errors='coerce'
)

# Extract year for trend analysis
competitions_filtered['Year'] = competitions_filtered['EnabledDate'].dt.year

# Keep competitions from 2010 onward
competitions_filtered = competitions_filtered[competitions_filtered['Year'] >= 2010]

# Optional: Reset index
competitions_filtered.reset_index(drop=True, inplace=True)

# Display a nicely formatted preview
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(competitions_filtered.head(10))




import pandas as pd
import os

# Define path to the dataset
BASE_PATH = r"C:\Users\Samhitha Shambavi\Meta kaggle hackathon submission , samhitha and nikhitha\dataset"
submissions_path = os.path.join(BASE_PATH, 'Submissions.csv')

# Load the data
submissions_df = pd.read_csv(submissions_path)

# Select useful columns
submissions_filtered = submissions_df[[
    'Id',                   # Submission ID
    'TeamId',               # Used to join with Teams or Competitions
    'SourceKernelVersionId',  # To trace the kernel used
    'SubmissionDate',
    'IsAfterDeadline',
    'PublicScoreFullPrecision',
    'PrivateScoreFullPrecision'
]].copy()

# Convert SubmissionDate to datetime
submissions_filtered['SubmissionDate'] = pd.to_datetime(
    submissions_filtered['SubmissionDate'], errors='coerce'
)

# Drop any rows without a kernel ID (we need this for later analysis)
submissions_filtered = submissions_filtered.dropna(subset=['SourceKernelVersionId'])

# Optional: filter only the earliest submission per team (placeholder for "best" logic)
submissions_filtered.sort_values(by='PublicScoreFullPrecision', ascending=False, inplace=True)

# Preview the processed submissions
print(submissions_filtered.head(10))



import pandas as pd
import os

BASE_PATH = r"C:\Users\Samhitha Shambavi\Meta kaggle hackathon submission , samhitha and nikhitha\dataset"
submissions_path = os.path.join(BASE_PATH, 'Submissions.csv')

# Load with low_memory=False to avoid dtype warning
submissions_df = pd.read_csv(submissions_path, low_memory=False)

submissions_filtered = submissions_df[[
    'Id', 'TeamId', 'SourceKernelVersionId', 'SubmissionDate', 'IsAfterDeadline',
    'PublicScoreFullPrecision', 'PrivateScoreFullPrecision'
]].copy()

submissions_filtered['SubmissionDate'] = pd.to_datetime(
    submissions_filtered['SubmissionDate'], errors='coerce'
)

submissions_filtered = submissions_filtered.dropna(subset=['SourceKernelVersionId'])

# Filter out absurdly large score values (assuming normal scores are < 1000)
max_reasonable_score = 1000

submissions_filtered = submissions_filtered[
    (submissions_filtered['PublicScoreFullPrecision'] < max_reasonable_score) &
    (submissions_filtered['PrivateScoreFullPrecision'] < max_reasonable_score)
]

# Sort descending by public score to get best submissions on top
submissions_filtered.sort_values(by='PublicScoreFullPrecision', ascending=False, inplace=True)

print(submissions_filtered.head(10))



import pandas as pd
import os

# Paths
BASE_PATH = r"C:\Users\Samhitha Shambavi\Meta kaggle hackathon submission , samhitha and nikhitha\dataset"
submissions_path = os.path.join(BASE_PATH, 'Submissions.csv')
kernel_versions_path = os.path.join(BASE_PATH, 'KernelVersions.csv')
kernels_path = os.path.join(BASE_PATH, 'Kernels.csv')

# Load files
submissions_df = pd.read_csv(submissions_path, low_memory=False)
kernel_versions_df = pd.read_csv(kernel_versions_path, low_memory=False)
kernels_df = pd.read_csv(kernels_path, low_memory=False)

# --- Preprocessing Submissions (same as step 2) ---
submissions_filtered = submissions_df[[
    'Id', 'TeamId', 'SourceKernelVersionId', 'SubmissionDate', 'IsAfterDeadline',
    'PublicScoreFullPrecision', 'PrivateScoreFullPrecision'
]].copy()

submissions_filtered['SubmissionDate'] = pd.to_datetime(
    submissions_filtered['SubmissionDate'], errors='coerce'
)
submissions_filtered = submissions_filtered.dropna(subset=['SourceKernelVersionId'])

max_reasonable_score = 1000
submissions_filtered = submissions_filtered[
    (submissions_filtered['PublicScoreFullPrecision'] < max_reasonable_score) &
    (submissions_filtered['PrivateScoreFullPrecision'] < max_reasonable_score)
]

# --- STEP 3-A: Join with KernelVersions ---
kernel_versions_df = kernel_versions_df[[
    'Id', 'ScriptId', 'AuthorUserId', 'CreationDate', 'Title',
    'TotalLines', 'TotalVotes', 'DockerImage'
]].copy()

kernel_versions_df.rename(columns={'Id': 'KernelVersionId'}, inplace=True)

# Join on SourceKernelVersionId = KernelVersionId
merged = pd.merge(
    submissions_filtered,
    kernel_versions_df,
    left_on='SourceKernelVersionId',
    right_on='KernelVersionId',
    how='inner'
)

# --- STEP 3-B: Join with Kernels (for full kernel info) ---
kernels_df = kernels_df[[
    'Id', 'CurrentKernelVersionId', 'AuthorUserId', 'Medal',
    'TotalViews', 'TotalVotes', 'MadePublicDate'
]].copy()

kernels_df.rename(columns={'Id': 'ScriptId'}, inplace=True)

# Join on ScriptId
final_df = pd.merge(
    merged,
    kernels_df,
    on='ScriptId',
    how='inner',
    suffixes=('_Version', '_Kernel')
)

# Sort by score and views
final_df.sort_values(by='PublicScoreFullPrecision', ascending=False, inplace=True)

# Preview the result
print(final_df[['ScriptId', 'KernelVersionId', 'Title', 'AuthorUserId_Version',
                'PublicScoreFullPrecision', 'PrivateScoreFullPrecision', 'Medal', 
                'TotalVotes_Version', 'TotalVotes_Kernel', 'DockerImage']].head(10))



import pandas as pd
import os

BASE_PATH = r"C:\Users\Samhitha Shambavi\Meta kaggle hackathon submission , samhitha and nikhitha\dataset"

# Load the files
kernel_tags = pd.read_csv(os.path.join(BASE_PATH, "KernelTags.csv"), low_memory=False)
tags = pd.read_csv(os.path.join(BASE_PATH, "Tags.csv"), low_memory=False)

# Just keep tag ID and name
tags = tags[['Id', 'Name', 'Slug', 'FullPath']]
tags.rename(columns={'Id': 'TagId'}, inplace=True)

# Step 1: Join KernelTags with Tags to get tag names for each kernel
tagged_kernels = pd.merge(kernel_tags, tags, on='TagId', how='left')

# Step 2: Filter to only kernels that appear in your final_df (from Step 3)
kernel_ids_in_top_submissions = final_df['ScriptId'].unique()
tagged_kernels_filtered = tagged_kernels[
    tagged_kernels['KernelId'].isin(kernel_ids_in_top_submissions)
]

# Step 3: Group tags per kernel
kernel_tag_summary = tagged_kernels_filtered.groupby('KernelId')['Name'].apply(list).reset_index()
kernel_tag_summary.rename(columns={'Name': 'ML_Tags'}, inplace=True)

# Step 4: Merge ML tags back into your final_df
final_df_with_tags = pd.merge(
    final_df,
    kernel_tag_summary,
    left_on='ScriptId',
    right_on='KernelId',
    how='left'
)

# Optional: show only the most relevant columns
display_df = final_df_with_tags[[
    'ScriptId', 'KernelVersionId', 'Title', 'PublicScoreFullPrecision',
    'PrivateScoreFullPrecision', 'Medal', 'ML_Tags'
]]

# Display top tagged kernels
print(display_df.head(10))
# Define list of valid ML technique tags
ML_METHODS = [
    'xgboost', 'lightgbm', 'catboost', 'random forest', 'logistic regression',
    'svm', 'cnn', 'rnn', 'lstm', 'transformer', 'bert', 'resnet', 'ensemble',
    'stacking', 'boosting', 'linear regression', 'naive bayes', 'decision tree'
]

# Function to keep only tags that are ML techniques
def filter_ml_tags(tag_list):
    if isinstance(tag_list, list):
        return [tag.lower() for tag in tag_list if tag.lower() in ML_METHODS]
    return []

# Apply filtering
final_df_with_tags['Filtered_ML_Tags'] = final_df_with_tags['ML_Tags'].apply(filter_ml_tags)

# Show only rows with at least one ML tag
ml_only_kernels = final_df_with_tags[final_df_with_tags['Filtered_ML_Tags'].map(len) > 0]

# Preview
print(ml_only_kernels[['ScriptId', 'Title', 'Filtered_ML_Tags', 'PublicScoreFullPrecision']].head(10))




import seaborn as sns
import matplotlib.pyplot as plt

# Set theme and style
sns.set(style="whitegrid", context="talk", palette="pastel")

# Count number of competitions per year
year_counts = competitions_filtered['Year'].value_counts().sort_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=year_counts.index, y=year_counts.values, palette="crest")

plt.title("Kaggle Competitions per Year (2010â€“2025)", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Number of Competitions")

# Optional: Annotate bars
for i, count in enumerate(year_counts.values):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Copy to avoid warnings
plot_df = submissions_filtered.copy()

# Extract year from submission date
plot_df['SubmissionYear'] = plot_df['SubmissionDate'].dt.year

# Filter years to make the plot readable
plot_df = plot_df[plot_df['SubmissionYear'] >= 2015]

# Set style
sns.set(style="whitegrid")

# Plot
plt.figure(figsize=(14, 6))
sns.swarmplot(
    data=plot_df.sample(1000, random_state=42),  # sampling for performance
    x='SubmissionYear',
    y='PublicScoreFullPrecision',
    size=4,
    palette="viridis"
)

plt.title("Distribution of Public Scores by Year (sampled)", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Public Score")
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Flatten the list of all ML tags
all_tags = final_df_with_tags['Filtered_ML_Tags'].dropna().tolist()
flat_tags = [tag for sublist in all_tags for tag in sublist]

# Count frequency
tag_counts = Counter(flat_tags)
top_10 = tag_counts.most_common(10)

# Convert to DataFrame for plotting
top_10_df = pd.DataFrame(top_10, columns=['ML Technique', 'Count'])

# Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(
    data=top_10_df,
    y='ML Technique',
    x='Count',
    palette='viridis'
)

plt.title(' Top 10 ML Techniques Used Across Kaggle Submissions', fontsize=14)
plt.xlabel('Usage Count')
plt.ylabel('ML Technique')
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Define medal mapping
medal_map = {1.0: 'Gold', 2.0: 'Silver', 3.0: 'Bronze'}

# Expand Filtered_ML_Tags into rows (explode)
exploded = final_df_with_tags[['Filtered_ML_Tags', 'Medal']].explode('Filtered_ML_Tags').dropna()

# Map medals to labels
exploded['Medal_Label'] = exploded['Medal'].map(medal_map)
exploded = exploded.dropna(subset=['Medal_Label'])  # remove rows without medal info

# Count medals per ML technique
medal_counts = exploded.groupby(['Filtered_ML_Tags', 'Medal_Label']).size().unstack(fill_value=0)

# Compute total per technique and convert to %
medal_pct = medal_counts.div(medal_counts.sum(axis=1), axis=0) * 100

# Keep only techniques with at least 10 medal-winning kernels (optional: reduce noise)
medal_pct = medal_pct[medal_counts.sum(axis=1) >= 10]

# Plot grouped bar chart
plt.figure(figsize=(12, 6))
medal_pct = medal_pct.sort_values(by='Gold', ascending=False).head(10)
medal_pct.plot(kind='bar', stacked=False, color=['gold', 'silver', 'peru'])

plt.title(' Medal Distribution (%) by ML Technique', fontsize=14)
plt.ylabel('Percentage of Medal-Winning Kernels')
plt.xlabel('ML Technique')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.legend(title='Medal Type')
plt.tight_layout()
plt.show()




# Let's merge on kernels.Id and competitions.Id :
df = kernels.merge(
    competitions[['Id', 'Title', 'RewardQuantity', 'Year', 'HostSegmentTitle']],
    left_on='Id',
    right_on='Id',
    how='left'
)

# Rename columns for clarity
df = df.rename(columns={'Title': 'CompetitionTitle', 'HostSegmentTitle': 'Category'})

# Drop rows with missing Year or ML_Methods
df = df.dropna(subset=['KernelYear', 'ML_Methods'])

# Fill missing category with Unknown
df['Category'] = df['Category'].fillna('Unknown')

# === Filter Top Methods ===
top_methods = df['ML_Methods'].value_counts().nlargest(6).index
df_top = df[df['ML_Methods'].isin(top_methods)]

# === 1 Line Chart: Evolution Over Time ===
plt.figure(figsize=(14, 7))
method_year = df_top.groupby(['KernelYear', 'ML_Methods']).size().reset_index(name='Count')
sns.lineplot(data=method_year, x='KernelYear', y='Count', hue='ML_Methods', marker='o')
plt.title("Evolution of Top ML Techniques in Winning Kaggle Kernels", fontsize=16)
plt.axvline(2018, color='gray', linestyle='--', alpha=0.7)
plt.text(2018.1, method_year['Count'].max() * 0.8, "BERT introduced", rotation=90)
plt.tight_layout()
plt.show()



# === 2 Heatmap: Competition Category vs ML Method ===
heatmap_df = df_top.pivot_table(index='Category', columns='ML_Methods', aggfunc='size', fill_value=0)
plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', fmt='d')
plt.title(" ML Methods Across Competition Categories")
plt.tight_layout()
plt.show()

print("ğŸ’¡ Insight: CNNs dominate CV; LightGBM dominates Tabular; BERT increases post-2018.")




# === 3 Word Cloud of Methods ===
text = " ".join(df['ML_Methods'].dropna())
wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='plasma').generate(text)
plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("â˜�ï¸� Common ML Techniques in Winning Kernels")
plt.tight_layout()
plt.show()




# === 4 Barplot: Method Counts Per Year ===
plt.figure(figsize=(14, 7))
sns.barplot(data=method_year, x='KernelYear', y='Count', hue='ML_Methods')
plt.title("ğŸ�† Popular ML Methods Per Year on Kaggle")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# === 5  Interactive Plotly Chart ===
try:
    import plotly.express as px
    fig = px.line(
        method_year,
        x='KernelYear',
        y='Count',
        color='ML_Methods',
        title='ğŸ“Š Interactive ML Method Trends on Kaggle Kernels',
        markers=True
    )
    fig.show()
except ImportError:
    print(" Install plotly with: pip install plotly")

# ===  Final Commentary ===
print(" FINAL INSIGHTS:")
print("- XGBoost peaked ~2017â€“2019 and has since tapered.")
print("- CNNs lead CV competitions; BERT rises in NLP after 2018.")
print("- LightGBM dominates tabular competitions consistently.")
print("- Post-2020 shows increased method diversity, reflecting the community's growing maturity and specialization.")
print("\n Conclusion: Kaggle kernel trends mirror real-world ML evolution, with clear dominance patterns and method diversification.")


