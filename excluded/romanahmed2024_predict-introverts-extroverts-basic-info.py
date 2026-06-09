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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# File paths
paths = {
    "Sample Submission": "/kaggle/input/playground-series-s5e7/sample_submission.csv",
    "Train": "/kaggle/input/playground-series-s5e7/train.csv",
    "Test": "/kaggle/input/playground-series-s5e7/test.csv"
}

# Loop through each file and display shape and memory usage
for name, path in paths.items():
    # Load the CSV file
    df = pd.read_csv(path)
    
    # Print shape: number of rows and columns
    print(f"{name} Shape: {df.shape}")
    
    # Print memory usage in MB
    memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"{name} Memory Usage: {memory:.2f} MB\n")



# Load train and test datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# 1. Check for missing (null) values in train and test
print("Missing values in Train:\n", train.isnull().sum().sort_values(ascending=False)[train.isnull().sum() > 0])
print("\nMissing values in Test:\n", test.isnull().sum().sort_values(ascending=False)[test.isnull().sum() > 0])

# 2. Check for duplicate rows in train and test
train_duplicates = train.duplicated().sum()
test_duplicates = test.duplicated().sum()

print(f"\nNumber of duplicate rows in Train: {train_duplicates}")
print(f"Number of duplicate rows in Test: {test_duplicates}")

# 3. Features in train but not in test
train_features = set(train.columns)
test_features = set(test.columns)

extra_train_features = train_features - test_features
print(f"\nFeatures present in Train but not in Test: {extra_train_features}")



train.info()


# Numeric features: dtype includes int, float
train_numeric = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
test_numeric = test.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Categorical features: dtype usually object or category
train_categorical = train.select_dtypes(include=['object', 'category']).columns.tolist()
test_categorical = test.select_dtypes(include=['object', 'category']).columns.tolist()

print("Train Numeric Features:", train_numeric)
print("Train Categorical Features:", train_categorical)

print("\nTest Numeric Features:", test_numeric)
print("Test Categorical Features:", test_categorical)


feature_names = []
dtypes = []
null_counts = []
unique_values = []

for col in train.columns:
    feature_names.append(col)
    dtypes.append(train[col].dtype)
    null_counts.append(train[col].isnull().sum())
    
    unique_vals = train[col].dropna().unique()
    # Take only first 10 unique values as list and convert to string
    unique_sample = unique_vals[:10]
    unique_values.append(", ".join(map(str, unique_sample)))

# Create summary DataFrame
summary_df = pd.DataFrame({
    "Feature Name": feature_names,
    "Data Type": dtypes,
    "Null Count": null_counts,
    "Unique Values (max 10)": unique_values
})

summary_df


train.describe().T


import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")

feature_names = []
dtypes = []
null_counts = []
unique_values = []
unique_counts = []
most_freq_values = []
freq_most_freq = []

for col in train.columns:
    feature_names.append(col)
    dtypes.append(train[col].dtype)
    null_counts.append(train[col].isnull().sum())
    
    unique_vals = train[col].dropna().unique()
    unique_values.append(", ".join(map(str, unique_vals[:10])))
    
    unique_counts.append(train[col].nunique())
    
    mode_val = train[col].mode()
    if not mode_val.empty:
        most_freq_values.append(mode_val.iloc[0])
        freq_most_freq.append(train[col].value_counts().iloc[0])
    else:
        most_freq_values.append(None)
        freq_most_freq.append(0)

summary_df = pd.DataFrame({
    "Feature Name": feature_names,
    "Data Type": dtypes,
    "Null Count": null_counts,
    "Null Percentage": (pd.Series(null_counts) / len(train)) * 100,
    "Unique Count": unique_counts,
    "Unique Values (max 10)": unique_values,
    "Most Frequent Value": most_freq_values,
    "Freq of Most Frequent": freq_most_freq,
})

# Add numeric summary stats
numeric_stats = train.describe().T
summary_df = summary_df.merge(numeric_stats[['mean', '50%', 'std']], left_on='Feature Name', right_index=True, how='left')
summary_df.rename(columns={'50%': 'Median'}, inplace=True)

# Add is constant flag
summary_df['Is Constant'] = summary_df['Unique Count'] == 1

# Feature type category
def feature_type(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return 'Numeric'
    elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
        return 'Categorical'
    else:
        return 'Other'

summary_df['Feature Type'] = summary_df['Data Type'].apply(feature_type)

summary_df



numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col != 'id']

total_rows = train.shape[0]

for col in numeric_features:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
    count = outliers.shape[0]
    percentage = (count / total_rows) * 100

    print(f"{col}: {count} outliers, {percentage:.2f}%")


import matplotlib.pyplot as plt
# Check if 'Personality' column exists and is object type
if 'Personality' in train.columns and train['Personality'].dtype == 'object':
    counts = train['Personality'].value_counts()
    percentages = train['Personality'].value_counts(normalize=True) * 100

    # Combine counts and percentages in labels
    labels = [f"{idx}\n{count} ({percent:.2f}%)" for idx, count, percent in zip(counts.index, counts.values, percentages.values)]

    # Plot bar chart
    plt.figure(figsize=(12,6))
    bars = plt.bar(counts.index, counts.values, color=['skyblue', 'salmon'])

    # Add text labels on top of bars
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(counts.values)*0.01, label, ha='center', va='bottom', fontsize=10)

    plt.title('Personality Distribution in Train Dataset')
    plt.xlabel('Personality')
    plt.ylabel('Count')
    plt.ylim(0, max(counts.values)*1.1)
    plt.show()

else:
    print("Personality column not found or is not of object type.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import math
import warnings
warnings.filterwarnings("ignore")

# Separate features
numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col != 'id']
categorical_features = train.select_dtypes(include=['object', 'category']).columns.tolist()

# Combine features and mark types
all_features = [(col, 'numeric') for col in numeric_features] + [(col, 'categorical') for col in categorical_features]

# Plot settings
features_per_page = 8
total_pages = math.ceil(len(all_features) / features_per_page)

for page in range(total_pages):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    start = page * features_per_page
    end = start + features_per_page
    selected_features = all_features[start:end]

    for ax, (col, ftype) in zip(axes, selected_features):
        if ftype == 'numeric':
            data = train[col].dropna()
            skw = skew(data)
            krt = kurtosis(data)
            sns.histplot(data, bins=30, kde=True, ax=ax, color='skyblue')
            ax.set_title(f"{col}\nSkewness: {skw:.2f}, Kurtosis: {krt:.2f}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        elif ftype == 'categorical':
            counts = train[col].value_counts()
            percentages = counts / counts.sum() * 100
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette='Set2')
            for idx, val in enumerate(counts.values):
                pct = percentages.values[idx]
                ax.text(idx, val + max(counts.values)*0.01, f"{pct:.1f}%", ha='center', fontsize=9)
            ax.set_title(f"{col} (n={counts.sum()})")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

    # Remove any unused subplots
    for i in range(len(selected_features), 8):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(f"Univariate Analysis (Page {page+1}/{total_pages})", fontsize=16, y=1.02)
    plt.show()



# Define target
target = 'Personality'

# Separate feature types
numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col != 'id']
categorical_features = [col for col in train.select_dtypes(include=['object', 'category']).columns if col not in ['id', target]]

# Combine feature list with types
all_features = [(col, 'numeric') for col in numeric_features] + [(col, 'categorical') for col in categorical_features]

# Plot parameters
features_per_page = 8
total_pages = math.ceil(len(all_features) / features_per_page)

for page in range(total_pages):
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.flatten()
    start = page * features_per_page
    end = min(start + features_per_page, len(all_features))
    selected_features = all_features[start:end]

    for ax, (col, ftype) in zip(axes, selected_features):
        if ftype == 'numeric':
            # Boxplot for numeric feature vs Personality
            sns.boxplot(data=train, x=target, y=col, ax=ax, palette='Set2')
            ax.set_title(f"{col} vs {target}", fontsize=11)
            ax.set_xlabel("Personality")
            ax.set_ylabel(col)
        elif ftype == 'categorical':
            # Countplot for categorical feature with hue=Personality
            sns.countplot(data=train, x=col, hue=target, ax=ax, palette='Set2')
            ax.set_title(f"{col} by {target}", fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.legend(title='Personality', loc='upper right', fontsize=9)

    # Remove any unused subplot axes
    for i in range(len(selected_features), 8):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(f"Bivariate Analysis: {target} vs Features (Page {page+1}/{total_pages})", fontsize=16, y=1.02)
    plt.show()



target = 'Personality'

# Identify numeric features (excluding 'id')
numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col != 'id']

# Plot config
features_per_page = 8
total_pages = math.ceil(len(numeric_features) / features_per_page)

for page in range(total_pages):
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.flatten()
    start = page * features_per_page
    end = min(start + features_per_page, len(numeric_features))
    selected_features = numeric_features[start:end]

    for ax, col in zip(axes, selected_features):
        sns.pointplot(data=train, x=target, y=col, ax=ax, errorbar='sd', capsize=.2, color='steelblue')
        ax.set_title(f"{col} Mean vs {target}", fontsize=11)
        ax.set_xlabel("Personality")
        ax.set_ylabel(f"Mean {col}")
        ax.tick_params(axis='x', rotation=0)

    # Remove any unused axes
    for i in range(len(selected_features), 8):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(f"Pointplot - Numeric Features vs Personality (Page {page+1}/{total_pages})", fontsize=16, y=1.02)
    plt.show()


numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                    'Friends_circle_size', 'Post_frequency']

# Define feature pairs (consecutive)
feature_pairs = [(numeric_features[i], numeric_features[i + 1]) for i in range(len(numeric_features) - 1)]

# Setup subplot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

# Loop through pairs and plot
for idx, (x_feat, y_feat) in enumerate(feature_pairs):
    ax = axes[idx]
    sns.scatterplot(data=train, x=x_feat, y=y_feat, hue='Personality', ax=ax, palette='Set2', alpha=0.7)
    ax.set_title(f"{x_feat} vs {y_feat}", fontsize=12)
    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    ax.legend(title='Personality', fontsize=8, loc='best')

# Remove any unused axes (only 4 plots)
for i in range(len(feature_pairs), 4):
    fig.delaxes(axes[i])

plt.suptitle("Numeric Feature Pairs vs Personality (2x2 Subplot)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# Numeric features exclude 'id'
numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                    'Friends_circle_size', 'Post_frequency']

# Create DataFrame to hold all pairs stacked for FacetGrid
pairs = [(numeric_features[i], numeric_features[i + 1]) for i in range(len(numeric_features) - 1)]

# Prepare a combined DataFrame for FacetGrid plotting
df_list = []

for i, (x_feat, y_feat) in enumerate(pairs):
    temp_df = train[['Personality', x_feat, y_feat]].copy()
    temp_df = temp_df.rename(columns={x_feat: 'x_feature', y_feat: 'y_feature'})
    temp_df['feature_pair'] = f"{x_feat} vs {y_feat}"
    df_list.append(temp_df)

plot_df = pd.concat(df_list, axis=0)

# Create FacetGrid with col by feature pairs and hue by Personality
g = sns.FacetGrid(plot_df, col='feature_pair', hue='Personality', col_wrap=2, height=5, palette='Set2', sharex=False, sharey=False)
g.map_dataframe(sns.scatterplot, x='x_feature', y='y_feature', alpha=0.7)
g.add_legend(title='Personality')

# Adjust titles and layout
g.set_axis_labels("", "")
g.set_titles(col_template="{col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Scatterplots of Consecutive Numeric Feature Pairs by Personality', fontsize=16)

plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

selected_numeric = numeric_features[:5]  # তুমি আগে যেটা করেছ

plt.figure(figsize=(14, 12))

for i, feature in enumerate(selected_numeric, 1):
    plt.subplot(4, 2, i)  # 4 rows, 2 cols, i-th plot
    sns.kdeplot(data=train, x=feature, hue='Personality', fill=True, alpha=0.4)
    plt.title(f'Distribution of {feature} by Personality')

plt.tight_layout()
plt.show()




# Define categorical features and target
cat_features = ['Stage_fear', 'Drained_after_socializing']
target = 'Personality'

# 1. Count distribution of Personality classes overall and by categorical features
print("Overall Personality Distribution:\n", train[target].value_counts(normalize=True) * 100)

# Plot overall Personality distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train, x=target, palette='Set2')
plt.title("Overall Personality Distribution")
plt.ylabel("Count")
plt.show()

# 2. Distribution of Personality by 'Stage_fear'
print("\nPersonality distribution by Stage_fear:")
stage_fear_dist = train.groupby(['Stage_fear', target]).size().unstack().fillna(0)
print(stage_fear_dist)

# Plot Personality counts per Stage_fear category
plt.figure(figsize=(8,5))
sns.countplot(data=train, x='Stage_fear', hue=target, palette='Set2')
plt.title("Personality Count by Stage_fear")
plt.ylabel("Count")
plt.show()

# 3. Distribution of Personality by 'Drained_after_socializing'
print("\nPersonality distribution by Drained_after_socializing:")
drained_dist = train.groupby(['Drained_after_socializing', target]).size().unstack().fillna(0)
print(drained_dist)

# Plot Personality counts per Drained_after_socializing category
plt.figure(figsize=(8,5))
sns.countplot(data=train, x='Drained_after_socializing', hue=target, palette='Set2')
plt.title("Personality Count by Drained_after_socializing")
plt.ylabel("Count")
plt.show()

# 4. Percentage distribution of Personality within each Stage_fear category
print("\nPercentage distribution of Personality within Stage_fear categories:")
stage_fear_percent = train.groupby(['Stage_fear', target]).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print(stage_fear_percent)

# Plot percentage stacked bar chart for Personality within Stage_fear
stage_fear_percent.plot(kind='bar', stacked=True, figsize=(8,5), colormap='Set2')
plt.title("Percentage of Personality Types within Stage_fear Categories")
plt.ylabel("Percentage (%)")
plt.xlabel("Stage_fear")
plt.legend(title=target)
plt.show()

# 5. Percentage distribution of Personality within each Drained_after_socializing category
print("\nPercentage distribution of Personality within Drained_after_socializing categories:")
drained_percent = train.groupby(['Drained_after_socializing', target]).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print(drained_percent)

# Plot percentage stacked bar chart for Personality within Drained_after_socializing
drained_percent.plot(kind='bar', stacked=True, figsize=(8,5), colormap='Set2')
plt.title("Percentage of Personality Types within Drained_after_socializing Categories")
plt.ylabel("Percentage (%)")
plt.xlabel("Drained_after_socializing")
plt.legend(title=target)
plt.show()




target = 'Personality'
cat_features = ['Stage_fear', 'Drained_after_socializing']

print("=== Basic Info ===")
print(train[target].value_counts(normalize=True)*100, "\n")  # Personality distribution

# Q1: Percentage distribution of Stage_fear within each Personality group
stage_fear_counts = train.groupby([target, 'Stage_fear']).size().reset_index(name='count')
stage_fear_counts['percentage'] = stage_fear_counts.groupby(target)['count'].transform(lambda x: 100 * x / x.sum())

print("Q1: Percentage distribution of Stage_fear (Yes/No) within each Personality group:")
print(stage_fear_counts.pivot(index=target, columns='Stage_fear', values='percentage'), "\n")

# Plot Q1
plt.figure(figsize=(6,4))
sns.barplot(data=stage_fear_counts, x=target, y='percentage', hue='Stage_fear', palette='Set1')
plt.title("Stage_fear distribution within Personality groups (%)")
plt.ylabel("Percentage")
plt.show()

# Q2: Percentage distribution of Drained_after_socializing within each Personality group
drained_counts = train.groupby([target, 'Drained_after_socializing']).size().reset_index(name='count')
drained_counts['percentage'] = drained_counts.groupby(target)['count'].transform(lambda x: 100 * x / x.sum())

print("Q2: Percentage distribution of Drained_after_socializing (Yes/No) within each Personality group:")
print(drained_counts.pivot(index=target, columns='Drained_after_socializing', values='percentage'), "\n")

# Plot Q2
plt.figure(figsize=(6,4))
sns.barplot(data=drained_counts, x=target, y='percentage', hue='Drained_after_socializing', palette='Set2')
plt.title("Drained_after_socializing distribution within Personality groups (%)")
plt.ylabel("Percentage")
plt.show()

# Q3: Count of Stage_fear = Yes by Personality
print("Q3: Count of Stage_fear=Yes by Personality:")
print(train[train['Stage_fear']=='Yes'].groupby(target).size(), "\n")

# Q4: Count of Drained_after_socializing = Yes by Personality
print("Q4: Count of Drained_after_socializing=Yes by Personality:")
print(train[train['Drained_after_socializing']=='Yes'].groupby(target).size(), "\n")

# Q5: Cross-tabulation of Stage_fear and Drained_after_socializing counts within each Personality group
print("Q5: Cross-tabulation of Stage_fear and Drained_after_socializing counts within each Personality group:\n")
for p in train[target].unique():
    ct = pd.crosstab(train.loc[train[target]==p, 'Stage_fear'], train.loc[train[target]==p, 'Drained_after_socializing'])
    print(f"Personality = {p}:\n{ct}\n")

# Optional: Heatmap of cross-tab for the last personality group
plt.figure(figsize=(6,4))
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
plt.title(f"Stage_fear vs Drained_after_socializing for Personality = {p}")
plt.show()



# Numeric features vs categorical target
import scipy.stats as stats
numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']
target = 'Personality'

for feature in numeric_features:
    groups = [group[feature].values for name, group in train.groupby(target)]
    stat, p = stats.f_oneway(*groups)
    print(f"ANOVA test for {feature} by {target}: F={stat:.2f}, p={p:.4f}")

# Categorical features vs categorical target
cat_features = ['Stage_fear', 'Drained_after_socializing']

def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

for feature in cat_features:
    confusion_matrix = pd.crosstab(train[target], train[feature])
    chi2, p, dof, ex = stats.chi2_contingency(confusion_matrix)
    cramer_v = cramers_v(confusion_matrix)
    print(f"Chi-square test for {feature} vs {target}: p={p:.4f}, Cramér's V={cramer_v:.3f}")


from scipy.stats import kruskal

for feature in numeric_features:
    groups = [group[feature].dropna().values for name, group in train.groupby(target)]
    stat, p = kruskal(*groups)
    print(f"Kruskal-Wallis test for {feature} by {target}: H={stat:.2f}, p={p:.4f}")



numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

# Compute correlation matrix
corr_matrix = train[numeric_features].corr()

# Sort correlations by absolute value (descending)
corr_pairs = corr_matrix.unstack()
# Remove self correlations
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
# Sort by absolute correlation
sorted_corr = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index)

print("=== Sorted Correlation Pairs ===")
print(sorted_corr)

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col != 'id']

# Sample subset if too many features (to keep pairplot readable)
selected_numeric = numeric_features[:5]  # Change count as needed

# Pairplot
sns.pairplot(data=train, vars=selected_numeric, hue='Personality', palette='Set2', diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle("PairPlot: Numeric Features Colored by Personality", y=1.02, fontsize=16)
plt.show()


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(train[selected_numeric].fillna(0))  # fillna to avoid errors

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=train['Personality'], palette='Set2')
plt.title('PCA Projection of Numeric Features')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


