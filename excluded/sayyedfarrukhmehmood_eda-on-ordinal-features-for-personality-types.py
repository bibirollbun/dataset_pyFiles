
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

class config:
    e=7
    train_path=f"/kaggle/input/playground-series-s5e{e}/train.csv"
    test_path=f"/kaggle/input/playground-series-s5e{e}/test.csv"
    submission_path=f"/kaggle/input/playground-series-s5e{e}/submission.csv"

    target = "Personality"
    ordinal_features = ['Time_spent_Alone', 'Social_event_attendance', 
                    'Going_outside', 'Friends_circle_size', 'Post_frequency']
    categorical_features=['Stage_fear', 'Drained_after_socializing']


train=pd.read_csv(config.train_path).drop(['id'],axis=1)


sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

ordinal_rank_dict = {}

print("ğŸ“Š Rank values in ordinal features:\n")

for feature in train.select_dtypes(exclude=['object']):
    ranks = train[feature].dropna().astype(int).unique()
    ranks.sort()
    ordinal_rank_dict[feature] = ranks
    print(f"{feature}: {ranks.tolist()}\n")

#Display bar plots of rank distributions
n_cols = 3
n_plots = len(ordinal_rank_dict)
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, (feature, ranks) in enumerate(ordinal_rank_dict.items()):
    ax = axes[i]
    sns.countplot(x=train[feature].dropna().astype(int), ax=ax, palette="viridis")
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")

# Hide unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



# Store categorical feature values
categorical_values = {}
print("ğŸ§© Values in categorical features:\n")


for feature in config.categorical_features:
    values = train[feature].dropna().unique()
    categorical_values[feature] = values
    print(f"{feature}: {values.tolist()}\n")

# Visualize top-level distribution for each categorical feature
n_cols = 3
n_plots = len(categorical_values)
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, (feature, values) in enumerate(categorical_values.items()):
    ax = axes[i]
    value_counts = train[feature].value_counts().head(15)  # top 15 for readability
    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette="crest")
    ax.set_title(f"{feature}")
    ax.set_xlabel("Count")
    ax.set_ylabel("Category")

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


sns.countplot(data=train, x='Personality')
plt.show()


import missingno as msno

# Visualize missing value locations
msno.matrix(train)
plt.title("Missing Value Matrix")
plt.show()

# Correlation heatmap of missingness
msno.heatmap(train)
plt.title("Missingness Correlation Heatmap")
plt.show()



# Grid layout for better visual comparison
n_cols = 2
n_rows = (len(config.ordinal_features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(config.ordinal_features):
    ax = axes[i]
    sns.histplot(
        data=train, 
        x=col, 
        hue='Personality', 
        multiple='dodge',
        discrete=True,
        shrink=0.8,
        palette="Set2",
        ax=ax
    )
    ax.set_title(f"{col} Distribution by Personality")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Ordinal Feature Distributions (Grouped by Personality)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


n_cols = 2
n_rows = (len(config.categorical_features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(config.categorical_features):
    ax = axes[i]
    # Crosstab (normalized by row)
    ctab = pd.crosstab(train[col], train['Personality'], normalize='index')
    ctab.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='black')

    ax.set_title(f"{col} vs Personality", fontsize=13)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("")
    ax.legend(title="Personality", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1) 

# Hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Categorical Features vs Personality (Proportional Breakdown)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


train.groupby('Personality')[config.ordinal_features].median()


import plotly.colors as pc

# Drop missing values
df_parallel = train.dropna(subset=config.ordinal_features + ['Personality'])

# Encode Personality as categorical codes (e.g., Introvert: 0, Extrovert: 1)
df_parallel['Personality_code'] = df_parallel['Personality'].astype('category').cat.codes
code_to_label = dict(enumerate(df_parallel['Personality'].astype('category').cat.categories))

# Use a discrete color scale for binary classes
custom_colors = ['#636EFA', '#EF553B']  # Blue, Red (feel free to customize)

# Plot using continuous color but only two values
fig = px.parallel_coordinates(
    df_parallel,
    dimensions=config.ordinal_features,
    color='Personality_code',
    color_continuous_scale=custom_colors,
    labels={col: col.replace('_', ' ') for col in config.ordinal_features},
)

# Update layout and color bar
fig.update_layout(
    title="ğŸ“Š Parallel Coordinates: Ordinal Features by Personality",
    template='plotly_white',
    height=600,
    margin=dict(t=110, l=40, r=40, b=40),
    coloraxis_colorbar=dict(
        title="Personality",
        tickvals=list(code_to_label.keys()),
        ticktext=list(code_to_label.values())
        
    )
)

fig.show()


# Threshold for rare values (e.g., less than 1% of total non-null)
RARE_THRESHOLD = 0.01

for col in config.ordinal_features:
    print(f"\nğŸ”� Checking {col} â€” Value Counts:\n")
    vc = train[col].value_counts(dropna=True).sort_index()
    print(vc)
    
    total = vc.sum()
    rare_values = vc[vc / total < RARE_THRESHOLD]

    if not rare_values.empty:
        print("\nâš ï¸� Rare Values Detected:")
        for val, count in rare_values.items():
            pct = 100 * count / total
            print(f" - Value {val}: {count} instances ({pct:.2f}%)")
    else:
        print("âœ… No rare values below threshold.")


