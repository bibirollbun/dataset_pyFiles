!pip install --no-deps /kaggle/input/iterative-stratification-wheel/iterative_stratification-0.1.9-py3-none-any.whl



!pip install --no-index --find-links=/kaggle/input/skorch-offline-install skorch


# Standard library imports
import os
import random
from collections import Counter
from typing import Tuple

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import itertools
import textwrap

# Scikit-learn imports
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.utils import class_weight
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skorch import NeuralNetClassifier

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


data_path = "../input/lish-moa/"
output_path = "/kaggle/working/"


train_features = pd.read_csv(data_path + "train_features.csv")
test_features = pd.read_csv(data_path + "test_features.csv")
sample_submission = pd.read_csv(data_path + "sample_submission.csv")
train_targets_nonscored = pd.read_csv(data_path + "train_targets_nonscored.csv")
train_targets_scored = pd.read_csv(data_path + "train_targets_scored.csv")
train_drug = pd.read_csv(data_path + "train_drug.csv")


print(train_features.shape)
print(test_features.shape)
print(sample_submission.shape)
print(train_targets_nonscored.shape)
print(train_targets_scored.shape)
print(train_drug.shape)


train_features.head()


train_targets_scored.head()


train_targets_nonscored.head()


#this should be disjoint sets
train_targets_nonscored_columns_set = set(train_targets_nonscored.columns.tolist())
train_targets_scored_columns_set = set(train_targets_scored.columns.tolist())
common_targets_scored_nonscored = train_targets_scored_columns_set.intersection(train_targets_nonscored_columns_set)
print(common_targets_scored_nonscored)


train_targets_all = pd.merge(train_targets_scored, train_targets_nonscored, on='sig_id', how='inner')


print(train_targets_all.shape)
train_targets_all.head()


dataset = pd.merge(train_features, train_targets_all, on='sig_id', how='inner').reset_index(drop=True)

print(dataset.shape)
dataset.tail()


dataset.fillna(np.nan)
df_missing_values = dataset.isna().sum()
print([s for s in df_missing_values.index if df_missing_values[s] > 0])


train_len = dataset.shape[0] # length of training samples, as oposed to testing samples
print(train_len)
dataset = pd.concat([dataset, test_features], axis=0).reset_index(drop=True)
print(dataset.shape)
print(dataset.tail()) # here are some NaNs
print(dataset[:train_len]) #verify that train_len is ok


train_features.dtypes


train_features_metadata_cols = [ 'cp_type', 'cp_time', 'cp_dose']
g_cols = [col for col in dataset.columns if col.startswith('g-')]
c_cols = [col for col in dataset.columns if col.startswith('c-')]
print(len(g_cols), len(c_cols))


print(train_features[g_cols].describe())
print(train_features[c_cols].describe())


fig, axes = plt.subplots(1,3, figsize=(15,5))

train_features['cp_type'].value_counts().plot(kind='bar', rot=0, ax = axes[0])
axes[0].set_title('cp_type value counts')
axes[0].set_ylabel('count')
axes[0].set_xlabel('cp_type')

train_features['cp_dose'].value_counts().plot(kind='bar', rot=0, ax=axes[1])
axes[1].set_title('cp_dose value counts')
axes[1].set_ylabel('count')
axes[1].set_xlabel('cp_dose')

train_features['cp_time'].value_counts().loc[[24, 48, 72]].plot(kind='bar', rot=0, ax=axes[2])
axes[2].set_title('cp_time value counts')
axes[2].set_ylabel('count')
axes[2].set_xlabel('cp_time')

plt.tight_layout()
plt.show()


random_sample_cols = pd.Series(g_cols).sample(10, random_state=42).tolist()

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(random_sample_cols):
    train_features[col].plot.kde(ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
plt.tight_layout()
plt.show()


random_sample_cols = pd.Series(c_cols).sample(10, random_state=42).tolist()

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(random_sample_cols):
    train_features[col].plot.kde(ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
plt.tight_layout()
plt.show()


corr_cells = train_features[c_cols].corr()
corr_genes = train_features[g_cols].corr()

treshold=0.8 # treshold of correlation, because of the high number of features
mask_c = np.abs(corr_cells) <= treshold
mask_g = np.abs(corr_genes) <= treshold

masked_corr_cells = corr_cells.where(~mask_c)
masked_corr_genes = corr_genes.where(~mask_g)


sns.heatmap(masked_corr_cells, cmap=sns.diverging_palette(220, 20, as_cmap=True))


sns.heatmap(masked_corr_genes, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# Calculate the cross-correlation matrix between c_cols and g_cols
# Calculate the correlation of the combined dataframe and then select the cross-correlation part
combined_features = pd.concat([train_features[c_cols], train_features[g_cols]], axis=1)
full_corr_matrix = combined_features.corr()

# Extract the cross-correlation part
cross_corr_matrix = full_corr_matrix.loc[c_cols, g_cols]

# Apply the mask with a threshold of 0.8
threshold_cross = 0.8
mask_cross = np.abs(cross_corr_matrix) <= threshold_cross
masked_cross_corr_matrix = cross_corr_matrix.where(~mask_cross)

# Display the heatmap
plt.figure(figsize=(10, 8)) # Adjust figure size as needed
sns.heatmap(masked_cross_corr_matrix, cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0)
plt.title('Cross-Correlation between C- and G- Features (Threshold > 0.8)')
plt.xlabel('G- Features')
plt.ylabel('C- Features')
plt.tight_layout()
plt.show()


ax = train_features.plot.scatter(x='g-224', y='g-226',figsize=(4,4), marker='.', alpha=0.2)


c_corr_matrix = train_features[c_cols].corr()

highly_correlated_c = []

for i in range(len(c_corr_matrix.columns)):
    for j in range(i + 1, len(c_corr_matrix.columns)):
        if abs(c_corr_matrix.iloc[i, j]) > 0.5:
            highly_correlated_c.append((c_corr_matrix.columns[i], c_corr_matrix.columns[j], c_corr_matrix.iloc[i, j]))

print("Highly correlated pairs in c_cols (|correlation| > 0.5):")
# for pair in highly_correlated_c:
    # print(f"{pair[0]} - {pair[1]}: {pair[2]:.4f}")
print(len(highly_correlated_c))


ax = train_features.plot.scatter(x='c-10', y='c-20',figsize=(4,4), marker='.', alpha=0.2)


gene_stats = train_features[g_cols].agg(['min', 'max', 'mean', 'std']).T
gene_stats = gene_stats.reset_index()
gene_stats = gene_stats.rename(columns={'index': 'features'})

# Reshape the DataFrame to long format
gene_stats_long = gene_stats.melt(id_vars='features', var_name='stat', value_name='values')

# Display the first few rows of the resulting DataFrame
display(gene_stats_long[1000:1010])


# Create a FacetGrid for each statistic
g = sns.FacetGrid(gene_stats_long, col="stat", sharex=False, sharey=False, height=4, aspect=1.2)

# Map a density plot onto each facet
g.map(sns.kdeplot, "values", fill=True)

# Add titles and remove legend
g.fig.suptitle("Gene distribution meta statistics", y=1.02)
g.set_titles("{col_name}")
g.set_axis_labels("", "")
g.add_legend(title="State")

plt.tight_layout()
plt.show()


cell_stats = train_features[c_cols].agg(['min', 'max', 'mean', 'std']).T
cell_stats = cell_stats.reset_index()
cell_stats = cell_stats.rename(columns={'index': 'features'})

# Reshape the DataFrame to long format
cell_stats_long = cell_stats.melt(id_vars='features', var_name='stat', value_name='values')

# Display the first few rows of the resulting DataFrame
display(cell_stats_long.head())


# Create a FacetGrid for each statistic
g = sns.FacetGrid(cell_stats_long, col="stat", sharex=False, sharey=False, height=4, aspect=1.2)

# Map a density plot onto each facet
g.map(sns.kdeplot, "values", fill=True)

# Add titles and remove legend
g.fig.suptitle("Cell distribution meta statistics", y=1.02)
g.set_titles("{col_name}")
g.set_axis_labels("", "")
g.add_legend(title="State")

plt.tight_layout()
plt.show()



# Select relevant columns (sig_id and first 7 'c-' columns)
cell_data = train_features[['sig_id'] + c_cols[:7]]

# Melt the DataFrame to long format
cell_data_long = cell_data.melt(id_vars='sig_id', var_name='feature', value_name='value')

# Filter for values less than -4
cell_data_filtered = cell_data_long[cell_data_long['value'] < -4]

# Create density plots for each feature
g = sns.FacetGrid(cell_data_filtered, col="feature", col_wrap=4, sharex=False, sharey=False)
g.map(sns.kdeplot, "value", fill=True)

# Add titles and adjust layout
g.fig.suptitle("Cell viability features - zoom in on negative tail", y=1.02)
g.set_titles("{col_name}")
g.set_axis_labels("", "")

plt.tight_layout()
plt.show()


# Calculate the sum of active MoAs per row
rowstats = train_targets_scored.drop(columns='sig_id').sum(axis=1).reset_index(name='sum')

# Calculate counts and percentages
row_counts = rowstats['sum'].value_counts().reset_index()
row_counts.columns = ['sum', 'n']
row_counts = row_counts.sort_values('sum')
row_counts['total'] = row_counts['n'].sum()
row_counts['perc'] = row_counts['n'] / row_counts['total']

# Create the bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='sum', y='n', data=row_counts, palette='Set2')

# Add percentage labels on top of the bars
for index, row in row_counts.iterrows():
    ax.text(index, row.n + 500, f'{row.perc:.2%}', color='black', ha="center")

# Set plot title and labels
plt.title("Number of Activations per Sample")
plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=0)
plt.ylim(0, row_counts['n'].max() + 1000) # Adjust y-lim to accommodate labels
plt.tight_layout()
plt.show()


# Calculate the sum of active MoAs per row
rowstats_nonscored = train_targets_nonscored.drop(columns='sig_id').sum(axis=1).reset_index(name='sum')

# Calculate counts and percentages
row_counts_nonscored = rowstats_nonscored['sum'].value_counts().reset_index()
row_counts_nonscored.columns = ['sum', 'n']
row_counts_nonscored = row_counts_nonscored.sort_values('sum')
row_counts_nonscored['total'] = row_counts_nonscored['n'].sum()
row_counts_nonscored['perc'] = row_counts_nonscored['n'] / row_counts_nonscored['total']

# Create the bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='sum', y='n', data=row_counts_nonscored, palette='Set2')

# Add percentage labels on top of the bars
for index, row in row_counts_nonscored.iterrows():
    ax.text(index, row.n + 500, f'{row.perc:.2%}', color='black', ha="center")

# Set plot title and labels
plt.title("Number of Activations per Sample")
plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=0)
plt.ylim(0, row_counts_nonscored['n'].max() + 1000) # Adjust y-lim to accommodate labels
plt.tight_layout()
plt.show()


# Calculate the sum of active MoAs for each target class
target_sums = train_targets_scored.drop(columns='sig_id').sum().reset_index(name='sum')
target_sums.columns = ['target', 'sum']

# Create the plots
fig = plt.figure(figsize=(12, 10))

# Plot 1: Density plot of MoA counts per target class
ax1 = fig.add_subplot(2, 1, 1)
sns.kdeplot(data=target_sums, x='sum', fill=True, color='darkorange', ax=ax1)
ax1.axvline(x=40, color='black', linestyle='--')
ax1.set_xscale('log')
ax1.set_title("MoA count per target class")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.text(40, ax1.get_ylim()[1]*0.8, 'Dashed line: 40', horizontalalignment='left', size='small', color='black')
ax1.set_title("MoA count per target class", fontsize=10)
ax1.xaxis.set_tick_params(labelsize=8)
ax1.yaxis.set_tick_params(labelsize=8)


# Plot 2: Classes with most MoAs
ax2 = fig.add_subplot(2, 2, 3)
top_classes = target_sums.sort_values(by='sum', ascending=False).head(5)
top_classes['target'] = top_classes['target'].str.replace('_', ' ')
sns.barplot(x='sum', y='target', data=top_classes, palette='Blues_d', ax=ax2)
ax2.set_title("Classes with most MoAs", fontsize=10)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.xaxis.set_tick_params(labelsize=8)
ax2.yaxis.set_tick_params(labelsize=8)


# Plot 3: Classes with fewest MoAs
ax3 = fig.add_subplot(2, 2, 4)
bottom_classes = target_sums.sort_values(by='sum', ascending=True).head(5)
bottom_classes['target'] = bottom_classes['target'].str.replace('_', ' ')
sns.barplot(x='sum', y='target', data=bottom_classes, palette='Reds_d', ax=ax3)
ax3.set_title("Classes with fewest MoAs", fontsize=10)
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.xaxis.set_tick_params(labelsize=8)
ax3.yaxis.set_tick_params(labelsize=8)


plt.tight_layout()
plt.show()


# Calculate the sum of active MoAs for each target class unscored
target_sums_nonscored = train_targets_nonscored.drop(columns='sig_id').sum().reset_index(name='sum')
target_sums_nonscored.columns = ['target', 'sum']

# Create the plots
fig = plt.figure(figsize=(12, 10))

# Plot 1: Density plot of MoA counts per target class
ax1 = fig.add_subplot(2, 1, 1)
sns.kdeplot(data=target_sums_nonscored, x='sum', fill=True, color='darkorange', ax=ax1)
ax1.axvline(x=7, color='black', linestyle='--')
ax1.set_xscale('log')
ax1.set_title("MoA count per target class (Non-scored)")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.text(7, ax1.get_ylim()[1]*0.8, 'Dashed line: 7', horizontalalignment='left', size='small', color='black')
ax1.set_title("MoA count per target class (Non-scored)", fontsize=10)
ax1.xaxis.set_tick_params(labelsize=8)
ax1.yaxis.set_tick_params(labelsize=8)


# Plot 2: Classes with most MoAs
ax2 = fig.add_subplot(2, 2, 3)
top_classes_nonscored = target_sums_nonscored.sort_values(by='sum', ascending=False).head(5)
top_classes_nonscored['target'] = top_classes_nonscored['target'].str.replace('_', ' ')
sns.barplot(x='sum', y='target', data=top_classes_nonscored, palette='Blues_d', ax=ax2)
ax2.set_title("Classes with most MoAs (Non-scored)", fontsize=10)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.xaxis.set_tick_params(labelsize=8)
ax2.yaxis.set_tick_params(labelsize=8)


# Plot 3: Classes with fewest MoAs
ax3 = fig.add_subplot(2, 2, 4)
bottom_classes_nonscored = target_sums_nonscored.sort_values(by='sum', ascending=True).head(5)
bottom_classes_nonscored['target'] = bottom_classes_nonscored['target'].str.replace('_', ' ')
sns.barplot(x='sum', y='target', data=bottom_classes_nonscored, palette='Reds_d', ax=ax3)
ax3.set_title("Classes with fewest MoAs (Non-scored)", fontsize=10)
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.xaxis.set_tick_params(labelsize=8)
ax3.yaxis.set_tick_params(labelsize=8)


plt.tight_layout()
plt.show()


# Select relevant columns
selected_cols = ['cp_dose', 'cp_time'] + [col for col in train_features.columns if col in ['g-525', 'g-666', 'c-42', 'c-22']]
plot_data = train_features[selected_cols].copy()

# Format cp_time
plot_data['cp_time'] = 'Duration ' + plot_data['cp_time'].astype(str) + 'h'

# Melt the DataFrame to long format
plot_data_long = plot_data.melt(id_vars=['cp_dose', 'cp_time'], var_name='feature', value_name='value')

# Create a FacetGrid
g = sns.FacetGrid(plot_data_long, row='feature', col='cp_time', hue='cp_dose', height=4, aspect=0.8, sharex=False, sharey=False, col_order=['Duration 24h', 'Duration 48h', 'Duration 72h'])

# Map the density plot
g.map(sns.kdeplot, 'value', fill=True, alpha=0.5)


# Add titles and labels
g.fig.suptitle("Treatment features vs example cell & gene distributions", y=1.02)
g.set_titles(row_template='{row_name}', col_template='{col_name}')
g.set_axis_labels("Feature value", "")
g.add_legend(title="Dose")

plt.tight_layout()
plt.show()


# Select relevant columns
selected_cols = ['cp_type', 'cp_time'] + [col for col in train_features.columns if col in ['g-525', 'g-666', 'c-42', 'c-22']]
plot_data = train_features[selected_cols].copy()

# Format cp_time
plot_data['cp_time'] = 'Duration ' + plot_data['cp_time'].astype(str) + 'h'

# Melt the DataFrame to long format
plot_data_long = plot_data.melt(id_vars=['cp_type', 'cp_time'], var_name='feature', value_name='value')

# Create a FacetGrid
g = sns.FacetGrid(plot_data_long, row='feature', col='cp_time', hue='cp_type', height=4, aspect=0.8, sharex=False, sharey=False, col_order=['Duration 24h', 'Duration 48h', 'Duration 72h'])

# Map the density plot
g.map(sns.kdeplot, 'value', fill=True, alpha=0.5)


# Add titles and labels
g.fig.suptitle("Treatment features vs example cell & gene distributions", y=1.02)
g.set_titles(row_template='{row_name}', col_template='{col_name}')
g.set_axis_labels("Feature value", "")
g.add_legend(title="Dose")

plt.tight_layout()
plt.show()




# Combine cp_time and cp_dose into a single grouping variable
train_features['treatment_group'] = train_features['cp_time'].astype(str) + '_' + train_features['cp_dose']

# Get the unique treatment groups
treatment_groups = train_features['treatment_group'].unique()

# Initialize dictionaries to store p-values for g- and c- features
g_p_values = {}
c_p_values = {}

# Perform Kruskal-Wallis test for each g- feature
for col in g_cols:
    feature_data = [train_features[train_features['treatment_group'] == group][col].dropna() for group in treatment_groups]
    if len(feature_data) > 1: # Ensure there's more than one group to compare
        try:
            statistic, p_value = kruskal(*feature_data)
            g_p_values[col] = p_value
        except ValueError:
            # Handle cases where all values in a group are identical
            g_p_values[col] = 1.0 # Assign a high p-value if test cannot be performed


# Perform Kruskal-Wallis test for each c- feature
for col in c_cols:
    feature_data = [train_features[train_features['treatment_group'] == group][col].dropna() for group in treatment_groups]
    if len(feature_data) > 1: # Ensure there's more than one group to compare
         try:
            statistic, p_value = kruskal(*feature_data)
            c_p_values[col] = p_value
         except ValueError:
            # Handle cases where all values in a group are identical
            c_p_values[col] = 1.0 # Assign a high p-value if test cannot be performed


# Convert p-values to Series and sort them
g_p_values_series = pd.Series(g_p_values).sort_values()
c_p_values_series = pd.Series(c_p_values).sort_values()

print("Top 10 g- features with most salient differences across treatment groups:")
print(g_p_values_series.head(10))

print("\nTop 10 c- features with most salient differences across treatment groups:")
print(c_p_values_series.head(10))

# Drop the temporary treatment_group column
train_features = train_features.drop(columns=['treatment_group'])


g_p_values_series[:2].index.to_list()
c_p_values_series[:2].index.to_list()
p_values = pd.concat([g_p_values_series[:2], c_p_values_series[:2]]).index.to_list()
print(p_values)


# Select relevant columns
selected_cols = ['cp_dose', 'cp_time'] + [col for col in train_features.columns if col in p_values]
plot_data = train_features[selected_cols].copy()

# Format cp_time
plot_data['cp_time'] = 'Duration ' + plot_data['cp_time'].astype(str) + 'h'

# Melt the DataFrame to long format
plot_data_long = plot_data.melt(id_vars=['cp_dose', 'cp_time'], var_name='feature', value_name='value')

# Create a FacetGrid
g = sns.FacetGrid(plot_data_long, row='feature', col='cp_time', hue='cp_dose', height=4, aspect=0.8, sharex=False, sharey=False, col_order=['Duration 24h', 'Duration 48h', 'Duration 72h'])

# Map the density plot
g.map(sns.kdeplot, 'value', fill=True, alpha=0.5)


# Add titles and labels
g.fig.suptitle("Treatment features vs example cell & gene distributions", y=1.02)
g.set_titles(row_template='{row_name}', col_template='{col_name}')
g.set_axis_labels("Feature value", "")
g.add_legend(title="Dose")

plt.tight_layout()
plt.show()


##no row with ctl_vehicle has an action, either in scored or nonscored targets
ctl_vehicle_data = dataset[:train_len][dataset[:train_len]['cp_type'] == 'ctl_vehicle']

is_train = dataset.index < train_len
is_ctl_vehicle = dataset['cp_type'] == 'ctl_vehicle'
ctl_vehicle_data = dataset.loc[is_train & is_ctl_vehicle]


ttscored_columns = train_targets_scored.drop(['sig_id'], axis=1).columns
cvd_columns = ctl_vehicle_data[ttscored_columns]
ones = cvd_columns.eq(1).sum()
print(ones.sum())



ttnonscored_columns = train_targets_nonscored.drop(['sig_id'], axis=1).columns
cvd_columns = ctl_vehicle_data[ttnonscored_columns]
ones = cvd_columns.eq(1).sum()
print(ones.sum())


cat_encoder = LabelEncoder()
dataset['cp_type'] = cat_encoder.fit_transform(dataset['cp_type'])


cp_time_mapping = {24: 1, 48: 2, 72: 3}
dataset['cp_time'] = dataset['cp_time'].map(cp_time_mapping)


cp_dose_mapping = {'D1': 1, 'D2': 2}
dataset['cp_dose'] = dataset['cp_dose'].map(cp_dose_mapping)


dataset['cp_time*cp_dose'] = dataset['cp_time'] * dataset['cp_dose']
train_features_metadata_cols = ['cp_type', 'cp_time', 'cp_dose', 'cp_time*cp_dose']


dataset[train_features_metadata_cols].head()


scaler_g = QuantileTransformer(output_distribution='normal', random_state=42)
scaler_c = QuantileTransformer(output_distribution='normal', random_state=42)
fit_g = scaler_g.fit(dataset.loc[:train_len-1,g_cols])#mabe i should use train_features[g_cols]
dataset[g_cols] = scaler_g.transform(dataset[g_cols])#but here is the answer: keep dataset
fit_c = scaler_c.fit(dataset.loc[:train_len-1,c_cols])
dataset[c_cols] = scaler_c.transform(dataset[c_cols])
print(dataset.head())


pca_g = PCA()
pca_g.fit(dataset[:train_len][g_cols])
cumsum = np.cumsum(pca_g.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print('The number of components selected: ', d)


plt.figure(figsize=(10, 6))
plt.plot(cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio for G-Features after PCA')
plt.grid(True)
plt.show()


pca_c = PCA()
pca_c.fit(dataset[:train_len][c_cols])
cumsum = np.cumsum(pca_c.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print('The number of components selected: ', d)


plt.figure(figsize=(10, 6))
plt.plot(cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio for G-Features after PCA')
plt.grid(True)
plt.show()


#hyperparameters
CONFIG = {
    "seed": 42,
    "epochs": 1000,
    "train_batch_size": 1024,
    "learning_rate": 1e-3,
    "T_max": 2000,
    "min_lr": 1e-5,
    "cat_weight": 1./3,
    "cont_weight": 2./3,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}




class AutoencoderC(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(AutoencoderC, self).__init__()
        print(f'input dim : {input_dim}')
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, latent_dim )
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, input_dim)
            # no activation → regression-like output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderG(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super(AutoencoderG, self).__init__()
        print(f'input dim : {input_dim}')
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, latent_dim )
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, input_dim)
            # no activation → regression-like output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder_g = AutoencoderG(input_dim=len(g_cols))
criterion_g = nn.MSELoss()
optimizer_g = torch.optim.Adam(autoencoder_g.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])


autoencoder_c = AutoencoderC(input_dim=len(c_cols))
criterion_c = nn.MSELoss()
optimizer_c = torch.optim.Adam(autoencoder_c.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler_c = lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])


def run_training(model, optimizer, scheduler, loss_fn, train_loader, valid_loader, device, num_epochs, early_stopping_steps, early_stop):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    else:
        print("[INFO] Using CPU\n")

    if len(train_loader) == 0 or len(valid_loader) == 0:
        raise ValueError("DataLoader is empty")

    model = model.to(device)
    early_step = 0
    best_loss = np.inf
    train_losses = []  # Track training loss per epoch
    valid_losses = []  # Track validation loss per epoch

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch['x'].to(device), batch['y'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        scheduler.step()
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_train_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch['x'].to(device), batch['y'].to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, targets).item()
        avg_val_loss = val_loss / len(valid_loader)
        valid_losses.append(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_step = 0
            # torch.save(model.state_dict(), 'best_model.pth')  # Uncomment to save model
        elif early_stop:
            early_step += 1
            print(f"  Early stopping step: {early_step}/{early_stopping_steps}")
            if early_step >= early_stopping_steps:
                print(f"  Early stopping triggered.")
                break

    return best_loss, train_losses, valid_losses


class AutoencoderDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.features[idx, :], dtype=torch.float) # Target is the same as input
        }
        return dct


# Prepare data for autoencoder training
c_features = dataset.loc[:train_len-1, c_cols].to_numpy(dtype=np.float32)


# Split data into training and validation sets for autoencoder
c_train, c_valid = train_test_split(c_features, test_size=0.2, random_state=CONFIG['seed'])

# Create data loaders
train_c_dataset = AutoencoderDataset(c_train)
valid_c_dataset = AutoencoderDataset(c_valid)
train_c_loader = DataLoader(train_c_dataset, batch_size=CONFIG['train_batch_size'], shuffle=True)
valid_c_loader = DataLoader(valid_c_dataset, batch_size=CONFIG['train_batch_size'], shuffle=False)


# Define loss function for autoencoder (MSELoss)
criterion_c = nn.MSELoss()

# Move the model to the device
autoencoder_c.to(CONFIG['device'])
print('before model_run_c')

# Run training for the c- autoencoder
model_run_c, train_losses, valid_losses = run_training(
    model=autoencoder_c,
    optimizer=optimizer_c,
    scheduler=scheduler_c,
    loss_fn=criterion_c,
    train_loader=train_c_loader,
    valid_loader=valid_c_loader,
    device=CONFIG['device'],
    # num_epochs=CONFIG['epochs'],
    num_epochs=50,
    early_stopping_steps=10,
    early_stop=False
)

print(f"Best validation loss for c- autoencoder: {model_run_c:.4f}")

# Save the trained model state dictionary
torch.save(autoencoder_c.state_dict(), output_path + 'autoencoder_c.pth')


def plot_losses(train_losses, valid_losses, num_epochs):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()


plot_losses(train_losses, valid_losses, num_epochs=100)


# Prepare data for autoencoder training
g_features = dataset.loc[:train_len-1, g_cols].to_numpy(dtype=np.float32)

# Split data into training and validation sets for autoencoder
g_train, g_valid = train_test_split(g_features, test_size=0.2, random_state=CONFIG['seed'])

# Create data loaders
train_g_dataset = AutoencoderDataset(g_train)
valid_g_dataset = AutoencoderDataset(g_valid)
train_g_loader = DataLoader(train_g_dataset, batch_size=CONFIG['train_batch_size'], shuffle=True)
valid_g_loader = DataLoader(valid_g_dataset, batch_size=CONFIG['train_batch_size'], shuffle=False)


# Define loss function for autoencoder (MSELoss)
criterion_g = nn.MSELoss()

# Move the model to the device
autoencoder_g.to(CONFIG['device'])
print('before model_run_g')

# Run training for the c- autoencoder
model_run_g, train_losses, valid_losses = run_training(
    model=autoencoder_g,
    optimizer=optimizer_g,
    scheduler=scheduler_g,
    loss_fn=criterion_g,
    train_loader=train_g_loader,
    valid_loader=valid_g_loader,
    device=CONFIG['device'],
    # num_epochs=CONFIG['epochs'],
    num_epochs=100,
    early_stopping_steps=10,
    early_stop=False
)

print(f"Best validation loss for c- autoencoder: {model_run_g:.4f}")

# Save the trained model state dictionary
torch.save(autoencoder_g.state_dict(), output_path +  'autoencoder_g.pth')


plot_losses(train_losses, valid_losses, num_epochs=100)



class EncodingDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct


# Assuming the autoencoders are saved
# torch.save(autoencoder_c.state_dict(), 'autoencoder_c.pth')
# torch.save(autoencoder_g.state_dict(), 'autoencoder_g.pth')

# Load the trained autoencoder models
autoencoder_c = AutoencoderC(input_dim=len(c_cols)) # Make sure encoding_dim is correct
autoencoder_c.load_state_dict(torch.load(output_path + 'autoencoder_c.pth'))
autoencoder_c.to(CONFIG['device'])
autoencoder_c.eval() # Set to evaluation mode

autoencoder_g = AutoencoderG(input_dim=len(g_cols)) # Make sure encoding_dim is correct
autoencoder_g.load_state_dict(torch.load(output_path + 'autoencoder_g.pth'))
autoencoder_g.to(CONFIG['device'])
autoencoder_g.eval() # Set to evaluation mode


# Function to encode data using a trained autoencoder
def encode_data(model, dataloader, device):
    encoded_features = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data['x'].to(device)
            encoded = model.encoder(inputs)
            encoded_features.append(encoded.cpu().numpy())
    return np.concatenate(encoded_features)

# Prepare data for encoding
train_c_features = dataset.loc[:train_len-1, c_cols].to_numpy(dtype=np.float32)
test_c_features = dataset.loc[train_len:, c_cols].to_numpy(dtype=np.float32)
train_g_features = dataset.loc[:train_len-1,g_cols].to_numpy(dtype=np.float32)
test_g_features = dataset.loc[train_len:,g_cols].to_numpy(dtype=np.float32)


train_c_dataset_enc = EncodingDataset(train_c_features)
test_c_dataset_enc = EncodingDataset(test_c_features)
train_g_dataset_enc = EncodingDataset(train_g_features)
test_g_dataset_enc = EncodingDataset(test_g_features)


train_c_loader_enc = DataLoader(train_c_dataset_enc, batch_size=CONFIG['train_batch_size'], shuffle=False)
test_c_loader_enc = DataLoader(test_c_dataset_enc, batch_size=CONFIG['train_batch_size'], shuffle=False)
train_g_loader_enc = DataLoader(train_g_dataset_enc, batch_size=CONFIG['train_batch_size'], shuffle=False)
test_g_loader_enc = DataLoader(test_g_dataset_enc, batch_size=CONFIG['train_batch_size'], shuffle=False)


# Encode the training and test data
train_c_encoded = encode_data(autoencoder_c, train_c_loader_enc, CONFIG['device'])
test_c_encoded = encode_data(autoencoder_c, test_c_loader_enc, CONFIG['device'])
train_g_encoded = encode_data(autoencoder_g, train_g_loader_enc, CONFIG['device'])
test_g_encoded = encode_data(autoencoder_g, test_g_loader_enc, CONFIG['device'])

# Create DataFrames from encoded features
train_c_encoded_df = pd.DataFrame(train_c_encoded, columns=[f'ae_C-{i}' for i in range(train_c_encoded.shape[1])])
test_c_encoded_df = pd.DataFrame(test_c_encoded, columns=[f'ae_C-{i}' for i in range(test_c_encoded.shape[1])])
train_g_encoded_df = pd.DataFrame(train_g_encoded, columns=[f'ae_G-{i}' for i in range(train_g_encoded.shape[1])])
test_g_encoded_df = pd.DataFrame(test_g_encoded, columns=[f'ae_G-{i}' for i in range(test_g_encoded.shape[1])])


# Combine with non-feature columns
train_features_reduced = pd.concat([dataset.loc[:train_len-1,['sig_id', 'cp_type', 'cp_time', 'cp_dose','cp_time*cp_dose']], train_c_encoded_df, train_g_encoded_df], axis=1)
test_features_reduced = pd.concat([dataset.loc[train_len:,['sig_id', 'cp_type', 'cp_time', 'cp_dose','cp_time*cp_dose']].reset_index(), test_c_encoded_df, test_g_encoded_df], axis=1)

print("Reduced training features shape:", train_features_reduced.shape)
print("Reduced test features shape:", test_features_reduced.shape)

display(train_features_reduced.head())
display(test_features_reduced.head())

dataset_reduced = pd.concat([train_features_reduced, test_features_reduced.set_index('index')])
print(dataset_reduced.tail())
print(dataset.tail())


train_processed = (
    train_features_reduced
    .merge(train_targets_all, on='sig_id')
    .query("cp_type != 0")
    .drop(columns='cp_type')
    .reset_index(drop=True)
)
train_len = len(train_processed)

test_processed = (
    test_features_reduced
    .query("cp_type != 0")
    .drop(columns=['index', 'cp_type'], errors='ignore')
    .reset_index(drop=True)
)
test_len = len(test_processed)

dataset_reduced = pd.concat([train_processed, test_processed], ignore_index=True)

g_cols = [col for col in dataset_reduced.columns if col.startswith('pca_G')]
c_cols = [col for col in dataset_reduced.columns if col.startswith('pca_C')]




print(dataset_reduced.head())


#save column names for features and targets apart
feature_columns = [col for col in dataset_reduced.columns if col not in train_targets_all.columns]
target_columns = [col for col in train_targets_all.columns if col != 'sig_id']
print(feature_columns)
dataset_reduced_index = dataset_reduced['sig_id'] # save the index
print(len(dataset_reduced_index))



print(target_columns)
print(feature_columns)
print(dataset_reduced.head())


folds = dataset_reduced[:train_len].copy()

X = folds[feature_columns].values  # Features
y = folds[target_columns].values  # Multi-label targets (2D array)


n_splits = 5  # Number of folds
mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print("Generating fold splits...")
fold_splits = list(mskf.split(X, y))

for fold, (train_idx, val_idx) in enumerate(fold_splits):
    folds.loc[val_idx, 'kfold'] = int(fold)

folds['kfold'] = folds['kfold'].astype(int)
print("DataFrame with 'kfold' column added:")
print(folds)
print(folds.shape)



print(folds.head())


pos_weights_per_fold = []

for fold_num, (train_idx, val_idx) in enumerate(fold_splits):
    print(f"Processing Fold {fold_num + 1}")

    # Get training data for this fold
    train_df = folds.iloc[train_idx]
    val_df = folds.iloc[val_idx]
    print(f"  Train set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # Get the target columns for the training set only
    y_train_matrix = train_df[target_columns].values

    # Calculate pos_weight for each target class in this fold
    num_positives = np.sum(y_train_matrix, axis=0)
    num_negatives = y_train_matrix.shape[0] - num_positives

    # Compute pos_weight = number_of_negatives / number_of_positives
    # Avoid division by zero for classes with no positive samples
    # pos_weight = np.where(num_positives > 0, num_negatives / num_positives, 1.0)
    pos_weight = np.where(num_positives > 0, num_negatives / (num_positives + 1.0), 1.0)
    pos_weight = np.minimum(pos_weight, 20.0)

    # Convert to a PyTorch tensor and move to the correct device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float).to(device)

    # Store pos_weight tensor for this fold
    pos_weights_per_fold.append(pos_weight_tensor)

    print(f"Using device: {device}")
    print(f"Calculated pos_weight tensor for fold {fold_num + 1}: {pos_weight_tensor}")


class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct

class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct



def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        if targets is None:
            raise ValueError("Targets is None in DataLoader")
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
        optimizer.step()
        final_loss += loss.item()

    final_loss /= len(dataloader)

    # Move scheduler step outside the loop
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(final_loss)
    else:
        scheduler.step()

    return final_loss

def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    with torch.no_grad():  # Add no_grad
        for data in dataloader:
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            final_loss += loss.item()
            valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds




class Model(nn.Module):
    def __init__(self, num_features, num_targets):
        super(Model, self).__init__()
        self.dense1 = nn.Linear(num_features, 1024)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.2)  # Reduced slightly to allow more learning

        self.dense2 = nn.Linear(1024, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)

        self.dense3 = nn.Linear(512, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)

        self.dense4 = nn.Linear(256, num_targets)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.dense1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = F.relu(self.dense2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = F.relu(self.dense3(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        x = self.dense4(x)  # No activation/BatchNorm/Dropout on output for logits

        return x


DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = True

num_features=len(feature_columns)
num_targets=len(target_columns)


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.01, pos_weight=None, max_pos_weight=100):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight  # Tensor of shape [num_targets]
        self.max_pos_weight = max_pos_weight  # Clip to prevent explosion (None to disable)

    def forward(self, pred, target):
        # Apply label smoothing: target=1 -> 1 - s, target=0 -> s
        target_smooth = target * (1 - self.smoothing) + (1 - target) * self.smoothing

        # Optional: Clamp for safety (prevents extremes if smoothing is large)
        target_smooth = torch.clamp(target_smooth, self.smoothing, 1 - self.smoothing)

        # Clip pos_weight if enabled (to handle extreme imbalance)
        pos_weight = self.pos_weight
        if self.max_pos_weight is not None and pos_weight is not None:
            pos_weight = torch.clamp(pos_weight, max=self.max_pos_weight)

        # Compute BCE with logits and pos_weight
        return F.binary_cross_entropy_with_logits(
            pred, target_smooth, pos_weight=pos_weight, reduction='mean'
        )


def run_training(fold):

    # --- Data Preparation ---
    # Get validation indices from the 'kfold' column
    val_idx = folds[folds['kfold'] == fold].index.to_numpy()
    # Get training indices (all indices where kfold is not the current fold)
    train_idx = folds[folds['kfold'] != fold].index.to_numpy()

    # Create training and validation DataFrames
    train_df = folds[folds['kfold'] != fold].reset_index(drop=True)
    valid_df = folds[folds['kfold'] == fold].reset_index(drop=True)

    # Extract training features and targets
    x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_df[target_columns].to_numpy(dtype=np.float32)

    # Extract validation features and targets
    x_valid = valid_df[feature_columns].to_numpy(dtype=np.float32)
    y_valid = valid_df[target_columns].to_numpy(dtype=np.float32)


    # Create PyTorch Datasets and DataLoaders
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Setup ---
    model = Model(
        num_features=num_features,
        num_targets=num_targets
    )
    model.to(DEVICE)



    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3)
    # pos_weight_tensor = pos_weights_per_fold[fold]
    # loss_fn = LabelSmoothingBCEWithLogitsLoss(pos_weight=pos_weight_tensor, smoothing=0.01)
    loss_fn = LabelSmoothingBCEWithLogitsLoss( smoothing=0.01) #to do: adjust the weights so they can be used
    eval_loss_fn = nn.BCEWithLogitsLoss()
    # --- Training Loop ---
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    best_loss = np.inf

    # # Initialize Out-Of-Fold predictions array robustly using num_targets
    oof = np.zeros((len(folds), num_targets))
    model_path = f"{output_path}FOLD{fold}_best_model.pth"

    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
        valid_loss, valid_preds = valid_fn(model, eval_loss_fn, validloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), model_path)

        elif EARLY_STOP:
            early_step += 1
            if early_step >= early_stopping_steps:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # --- Prediction ---
    print(f"Loading best model for fold {fold} with loss: {best_loss:.4f}")

    x_test = test_processed[feature_columns].to_numpy(dtype=np.float32)

    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    # # Re-instantiate the model to ensure a clean state before loading the best weights
    inference_model = Model(
        num_features=num_features,
        num_targets=num_targets
    )
    inference_model.load_state_dict(torch.load(model_path))
    inference_model.to(DEVICE)

    # Get predictions on the test set
    predictions = inference_fn(inference_model, testloader, DEVICE)

    return oof, predictions





def run_k_fold(NFOLDS: int, folds: np.ndarray, test_len: int) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(folds), len(target_columns)))
    predictions = np.zeros((test_len, len(target_columns)))

    for fold in range(NFOLDS):
        print(f"========== FOLD {fold} TRAINING ==========")
        oof_, pred_ = run_training(fold)
        predictions += pred_ / NFOLDS
        oof += oof_

    return oof, predictions



oof = np.zeros((len(folds), len(target_columns)))
predictions = np.zeros((test_len, len(target_columns)))

oof, predictions = run_k_fold(NFOLDS, folds=folds, test_len=test_len)





def safe_mean_columnwise_log_loss(y_true, y_pred, labels=None):
    # Extract values and labels if input is pandas DataFrame
    if isinstance(y_true, pd.DataFrame):
        if labels is None:
            labels = y_true.columns
        y_true = y_true.values

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    n_labels = y_true.shape[1]
    scores = []

    for i in range(n_labels):
        label_name = labels[i] if labels is not None else f"column {i}"
        try:
            # Clip predictions to avoid log(0) errors
            pred_clipped = np.clip(y_pred[:, i], 1e-15, 1 - 1e-15)
            score = log_loss(y_true[:, i], pred_clipped)
            scores.append(score)
        except ValueError as e:
            # This error is raised if y_true[:, i] contains only one class.
            # We issue a warning and skip this column.
            warnings.warn(
                f"Skipping log_loss for '{label_name}': {e}",
                UserWarning
            )

    if not scores:
        warnings.warn("No valid columns found to calculate log_loss. Returning NaN.", UserWarning)
        return np.nan

    return np.mean(scores)




dataset_reduced.loc[:train_len-1, target_columns] = oof
dataset_reduced.loc[train_len:, target_columns] = predictions

# Extract predictions and true labels with sig_id
y_pred = dataset_reduced.loc[:train_len-1][['sig_id'] + target_columns].reset_index(drop=True)
y_true = train_targets_all[['sig_id'] + target_columns]
y_true = y_true[y_true['sig_id'].isin(y_pred['sig_id'])].reset_index(drop=True)

# Optional: Sort by sig_id to ensure consistent row order
y_pred = y_pred.sort_values('sig_id').reset_index(drop=True)
y_true = y_true.sort_values('sig_id').reset_index(drop=True)

# Verify shapes
print(y_pred.shape)  # Should be (21948, 609)
print(y_true.shape)  # Should be (21948, 609)

# Check index alignment
assert all(y_pred.index == y_true.index), "Index mismatch between y_pred and y_true"

# Verify shapes
print(y_pred.shape)  # Should be (21948, 609) assuming 608 target columns + sig_id
print(y_true.shape)  # Should match y_pred's shape (21948, 609)



# Verify alignment
assert len(y_pred) == len(y_true), "Mismatch in number of training samples"
assert all(y_pred.index == y_true.index), "Index mismatch between y_pred and y_true"

# Set sig_id as index for alignment
y_true = y_true.set_index('sig_id')
y_pred = y_pred.set_index('sig_id')

# Align y_pred to y_true's index, filling missing rows with 0
y_pred_aligned = y_pred.reindex(y_true.index, fill_value=0)

# Ensure no NaN values
y_pred_aligned = y_pred_aligned.fillna(0)
y_true = y_true.fillna(0)

# Debugging: Check sums of true labels
print("Sum of true labels per column:\n", y_true.sum())

score = safe_mean_columnwise_log_loss(y_true.values, y_pred_aligned.values)
print(f"Mean Columnwise Log Loss: {score:.6f}")

# Optionally, save submission file
# dataset_reduced[['sig_id'] + target_columns].to_csv('submission.csv', index=False)



# Assuming dataset_reduced, predictions, sample_submission, and train_targets_scored are available
# Define target columns (excluding sig_id)
target_columns = [col for col in train_targets_scored.columns if col != 'sig_id']

# Extract test predictions from dataset_reduced
# train_len = 21948, so test data starts at train_len
submission_df = dataset_reduced.loc[train_len:, ['sig_id'] + target_columns].copy()

# Ensure submission_df has the same columns as sample_submission
submission_cols = sample_submission.columns[1:].tolist()  # Exclude sig_id
prediction_cols = target_columns  # From train_targets_scored

# Verify column alignment
if not all(col in submission_df.columns for col in submission_cols):
    raise ValueError("submission_df missing columns from sample_submission")
if not all(col in submission_df.columns for col in prediction_cols):
    raise ValueError("submission_df missing columns from train_targets_scored")

# Set sig_id as index for alignment with sample_submission
submission_df = submission_df.set_index('sig_id')
sample_submission = sample_submission.set_index('sig_id')

# Ensure indices align
submission_df = submission_df.reindex(sample_submission.index, fill_value=0)

# Check shape compatibility
if submission_df[submission_cols].shape != sample_submission[submission_cols].shape:
    raise ValueError("Shape mismatch between submission_df and sample_submission")

# Assign predictions to submission_df
submission_df[submission_cols] = submission_df[prediction_cols]

# Reset index to include sig_id as a column
submission_df = submission_df.reset_index()

# Ensure no NaN values
submission_df = submission_df.fillna(0)

# Verify shapes
print(f"submission_df shape: {submission_df.shape}")  # Expected: (3982, 207)
print(f"sample_submission shape: {sample_submission.shape}")  # Expected: (3982, 207)

# Inspect the first few rows
print(submission_df.head())

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




