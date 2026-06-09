import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')
original2 = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')


train.head()


train.info()


test.info()


original.info()


original2.info()


original[original['Time_spent_Alone'].isnull()][:5]


# Get the index values (not just 0 to 4 — actual row indices)
null_indices = original[original['Time_spent_Alone'].isnull()].index[:5]

# Use those indices to select the same rows from original2
original2.loc[null_indices]



original.describe()


train.describe()


# Remove unnecessary feature
train.drop(columns='id', inplace=True)

numerical = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical = train.select_dtypes(include=['object']).columns.tolist()

print(numerical)
print(categorical)


# Loop through categorical columns
for col in categorical:
    counts = train[col].value_counts(dropna=False)
    num_categories = len(counts)

    #print(f"\nValue counts for: {col}")
    #print(counts)


    plt.figure(figsize=(6, 5))
    plt.title(f"Value Counts for '{col}'", fontsize=14)   
    plt.pie(
        counts,
        labels=counts.index.astype(str),
        autopct='%1.1f%%',
        startangle=270,  # rotate 90° clockwise
        wedgeprops={'edgecolor': 'white'},
        colors=['salmon', 'skyblue', 'lightgreen'],
        textprops={'color': 'gray', 'fontsize': 10}
    )
    plt.axis('equal')

    plt.tight_layout()
    plt.show()



sns.set(style="white")

for col in numerical:
    # Convert values to strings, treating NaNs as 'NA'
    col_data = train[col].copy()
    col_data = col_data.astype('Int64')  
    value_counts = col_data.value_counts(dropna=False).sort_index()

    # Convert index to string so we can display 'NA'
    labels = [str(val) if pd.notna(val) else 'NA' for val in value_counts.index]
    counts = value_counts.values
    num_categories = len(labels)

    #print(f"\nValue counts for: {col}")
    #print(f"\nNumber of categories (including NA): {num_categories}")
    #print(pd.Series(counts, index=labels))

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=labels,
        y=counts,
        color="salmon",
        edgecolor='white',
        width=1
    )

    plt.title(f"Distribution of '{col}'", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()



for col in numerical:
    plt.figure(figsize=(6, 4))
    sns.violinplot(
        data=train,
        x="Personality",  
        y=col,
        palette={'Introvert': 'skyblue', 'Extrovert': 'salmon'},
        inner="box", 
        cut=0
    )
    plt.title(f"Violin Plot of '{col}' by Target", fontsize=14)
    plt.tight_layout()
    plt.show()



plt.figure(figsize=(10, 8))
corr = train[numerical].corr()

custom_cmap = LinearSegmentedColormap.from_list(
    "skyblue_salmon", ["skyblue", "white", "salmon"]
)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap=custom_cmap,     
    square=True,
    linewidths=0.5,
    linecolor='white'
)

plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()



# Slow, run only if necessary
sns.pairplot(train[numerical + ['Personality']], 
             hue="Personality", 
             palette={'Introvert': 'skyblue', 'Extrovert': 'salmon'},
            )

plt.show()


for col in categorical[:2]:
    #print(f"\nDistribution of Personality by {col}")
    
    # Compute proportions
    proportions = train.groupby(col)['Personality'].value_counts(normalize=True).unstack()

    # Display table
    #display(proportions)

    # Plot it
    proportions.plot(
        kind='bar', 
        stacked=True, 
        color=['salmon','skyblue'], 
        figsize=(6, 4),
        edgecolor='white'
    )

    plt.title(f"Personality Proportion by {col}")
    plt.ylabel("Proportion")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    
    # Move legend to the right
    plt.legend(
        title="Personality", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        borderaxespad=0.
    )
    
    plt.tight_layout()
    plt.show()



for col in numerical:  
    # Compute proportion of each Personality for each numeric category
    proportions = train.groupby(col)['Personality'].value_counts(normalize=True).unstack()

    # Plot
    proportions.plot(
        kind='bar',
        stacked=True,
        color=['salmon', 'skyblue'],
        figsize=(8, 4),
        edgecolor='white'
    )

    plt.title(f"Personality Proportion by {col}")
    plt.ylabel("Proportion")
    plt.xlabel(col)
    plt.xticks(rotation=0)

    # Move legend to the right
    plt.legend(
        title="Personality",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.show()



# Dataset dictionary for looping
datasets = {
    'Train': train,
    'Original': original
}

# Radar setup
labels = numerical
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # loop back

# Plot setup
fig, axs = plt.subplots(1, 2, figsize=(18, 6), subplot_kw=dict(polar=True))
colors = {'Introvert': 'skyblue', 'Extrovert': 'salmon'}

for ax, (name, df) in zip(axs, datasets.items()):
    grouped = df.groupby("Personality")[numerical].mean()

    for personality in grouped.index:
        values = grouped.loc[personality].tolist()
        values += values[:1]  # close the loop

        ax.plot(angles, values, label=personality, color=colors[personality], linewidth=2)
        ax.fill(angles, values, alpha=0.2, color=colors[personality])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{name} Dataset", fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.suptitle("Radar Plot of Numerical Means by Personality (Across Datasets)", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()



# Deal with missing values
for df in [train, test]:
    for col in numerical:
        metric = original[col].median()
        df[col] = df[col].fillna(metric)
        #print(col, metric)
    #print("\n")

#train.head(5)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

TARGET   = 'Personality' 
COLS    = numerical            
CLS_COLORS = {'Introvert': '#1f77b4', 'Extrovert': '#ff7f0e'}

X_scaled = StandardScaler().fit_transform(train[COLS])

pca = PCA(n_components=2, random_state=42)
PCs = pca.fit_transform(X_scaled)

train['PC1'], train['PC2'] = PCs[:, 0], PCs[:, 1]

print(f'Explained variance: PC1 {pca.explained_variance_ratio_[0]:.2%}, '
      f'PC2 {pca.explained_variance_ratio_[1]:.2%}')

plt.figure(figsize=(8, 6))

for cls, color in CLS_COLORS.items():
    mask = train[TARGET] == cls
    plt.scatter(train.loc[mask, 'PC1'],
                train.loc[mask, 'PC2'],
                s=25, alpha=0.6, label=cls, c=color)

plt.title('Global PCA of Train Set (PC1 vs PC2)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.show()



        
CLASSES  = ['Introvert', 'Extrovert']
OUTLIER_Q = 0.983  # percentile cut‑off
COMPONENTS = 2                    

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
outlier_rows = {}

for ax, cls in zip(axes, CLASSES):
    # isolate one class
    sub = train[train[TARGET] == cls].copy()
    X   = sub[numerical].values
    
    # standardise & PCA
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)
    
    pca = PCA(n_components=COMPONENTS, random_state=42)
    PCs = pca.fit_transform(X_std)
    sub['PC1'], sub['PC2'] = PCs[:, 0], PCs[:, 1]
    
    # simple distance‑from‑centre rule in PC space
    dists = np.linalg.norm(PCs, axis=1)
    thr   = np.quantile(dists, OUTLIER_Q)
    sub['is_outlier'] = dists > thr
    
    # remember indices to inspect raw survey answers later
    outlier_rows[cls] = sub.index[sub['is_outlier']].tolist()
    
    # plot
    ax.scatter(sub.loc[~sub['is_outlier'], 'PC1'],
               sub.loc[~sub['is_outlier'], 'PC2'],
               s=25, alpha=0.6, label='inlier')
    
    ax.scatter(sub.loc[ sub['is_outlier'], 'PC1'],
               sub.loc[ sub['is_outlier'], 'PC2'],
               s=80, facecolors='none', edgecolors='r', linewidths=2,
               label='outlier')
    
    ax.set_title(f'{cls}: PCA (2 components)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()

plt.suptitle('Class‑specific PCA: Outliers Highlighted', fontsize=15)
plt.tight_layout()
plt.show()

# quick printout of suspicious rows
for cls, rows in outlier_rows.items():
    print(f'\n{cls}: {len(rows)} outliers → {rows[:10]}{" ..." if len(rows) > 10 else ""}')



from sklearn.manifold import TSNE
import umap.umap_ as umap  

NUMERIC_COLS = numerical  
CLS_COLOURS  = {'Introvert': '#1f77b4', 'Extrovert': '#ff7f0e'}

X_scaled = StandardScaler().fit_transform(train[NUMERIC_COLS])

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,     
    min_dist=0.1,      
    metric='euclidean',
    random_state=42
)
umap_emb = umap_model.fit_transform(X_scaled)

tsne_model = TSNE(
    n_components=2,
    perplexity=30,    
    learning_rate='auto',
    n_iter=1000,
    init='pca',
    random_state=42
)
tsne_emb = tsne_model.fit_transform(X_scaled)

train['UMAP1'], train['UMAP2'] = umap_emb[:, 0], umap_emb[:, 1]
train['TSNE1'], train['TSNE2'] = tsne_emb[:, 0], tsne_emb[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)

for (x, y, title, ax) in [
    ('UMAP1', 'UMAP2', 'UMAP (2‑D)', axes[0]),
    ('TSNE1', 'TSNE2', 't‑SNE (2‑D)', axes[1])
]:
    for cls, colour in CLS_COLOURS.items():
        mask = train[TARGET] == cls
        ax.scatter(train.loc[mask, x],
                   train.loc[mask, y],
                   s=25, alpha=0.6, c=colour, label=cls)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()

plt.suptitle('UMAP vs t‑SNE on Train Set', fontsize=14)
plt.tight_layout()
plt.show()


