!ls /kaggle/input


# Configuration
RUN_PHASE = False  # Set to True for run phase (saves plots), False for debug phase
NOTEBOOK_NAME = 'eda_competition_data_20250820_120436'
plot_counter = 0

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette('husl')

# Helper function to save plots
def save_plot(fig=None):
    global plot_counter
    if RUN_PHASE:
        plot_counter += 1
        plot_path = f'plots/{NOTEBOOK_NAME}_plot_{plot_counter:03d}.png'
        if fig is None:
            fig = plt.gcf()
        fig.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f'Plot saved to {plot_path}')
    plt.show()

print(f"Notebook running in {'RUN' if RUN_PHASE else 'DEBUG'} phase")
print(f"Plots will {'BE SAVED' if RUN_PHASE else 'NOT BE SAVED'}")


# Define data paths
DATA_DIR = Path('/kaggle/input/jigsaw-agile-community-rules')

# Load datasets
train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test.csv')
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Display column information
print("=" * 50)
print("TRAIN DATA COLUMNS:")
print("=" * 50)
print(train_df.columns.tolist())
print()

print("=" * 50)
print("TEST DATA COLUMNS:")
print("=" * 50)
print(test_df.columns.tolist())


# Display first few rows of train data
print("=" * 50)
print("TRAIN DATA - FIRST 3 ROWS:")
print("=" * 50)
train_df.head(3)


# Data types and missing values
print("=" * 50)
print("DATA TYPES AND MISSING VALUES:")
print("=" * 50)
print("\nTrain Data:")
print(train_df.info())
print("\n" + "="*30 + "\n")
print("Missing values in train:")
print(train_df.isnull().sum())


# Target distribution
target_dist = train_df['rule_violation'].value_counts()
print("Target Distribution:")
print(target_dist)
print(f"\nClass balance: {target_dist[1]/len(train_df)*100:.2f}% violations")

# Visualize target distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
target_dist.plot(kind='bar', ax=ax)
ax.set_title('Distribution of Rule Violations', fontsize=14)
ax.set_xlabel('Rule Violation', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_xticklabels(['No Violation (0)', 'Violation (1)'], rotation=0)

# Add percentage labels
for i, v in enumerate(target_dist.values):
    ax.text(i, v + 10, f'{v}\n({v/len(train_df)*100:.1f}%)', 
            ha='center', va='bottom')

plt.tight_layout()
save_plot(fig)


# Analyze comment lengths
train_df['body_length'] = train_df['body'].str.len()
train_df['body_word_count'] = train_df['body'].str.split().str.len()

test_df['body_length'] = test_df['body'].str.len()
test_df['body_word_count'] = test_df['body'].str.split().str.len()

print("Comment Length Statistics (Train):")
print(train_df[['body_length', 'body_word_count']].describe())


# Visualize text length distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Character length distribution
axes[0, 0].hist(train_df['body_length'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Comment Character Length (Train)', fontsize=12)
axes[0, 0].set_xlabel('Character Count')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(train_df['body_length'].median(), color='red', 
                   linestyle='--', label=f'Median: {train_df["body_length"].median():.0f}')
axes[0, 0].legend()

# Word count distribution
axes[0, 1].hist(train_df['body_word_count'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribution of Comment Word Count (Train)', fontsize=12)
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(train_df['body_word_count'].median(), color='red', 
                   linestyle='--', label=f'Median: {train_df["body_word_count"].median():.0f}')
axes[0, 1].legend()

# Length by violation status
train_df.boxplot(column='body_length', by='rule_violation', ax=axes[1, 0])
axes[1, 0].set_title('Comment Length by Rule Violation Status')
axes[1, 0].set_xlabel('Rule Violation')
axes[1, 0].set_ylabel('Character Count')
axes[1, 0].set_xticklabels(['No Violation', 'Violation'])
plt.sca(axes[1, 0])
plt.xticks([1, 2], ['No Violation', 'Violation'])

# Word count by violation status
train_df.boxplot(column='body_word_count', by='rule_violation', ax=axes[1, 1])
axes[1, 1].set_title('Word Count by Rule Violation Status')
axes[1, 1].set_xlabel('Rule Violation')
axes[1, 1].set_ylabel('Word Count')
axes[1, 1].set_xticklabels(['No Violation', 'Violation'])
plt.sca(axes[1, 1])
plt.xticks([1, 2], ['No Violation', 'Violation'])

plt.suptitle('')  # Remove automatic suptitle from boxplot
plt.tight_layout()
save_plot(fig)


# Analyze unique rules
unique_rules_train = train_df['rule'].unique()
unique_rules_test = test_df['rule'].unique()

print(f"Number of unique rules in train: {len(unique_rules_train)}")
print(f"Number of unique rules in test: {len(unique_rules_test)}")
print("\n" + "="*50)
print("Sample rules:")
print("="*50)
for i, rule in enumerate(unique_rules_train[:5], 1):
    print(f"\n{i}. {rule[:100]}..." if len(rule) > 100 else f"\n{i}. {rule}")


# Analyze subreddits
unique_subreddits_train = train_df['subreddit'].unique()
unique_subreddits_test = test_df['subreddit'].unique()

print(f"Number of unique subreddits in train: {len(unique_subreddits_train)}")
print(f"Number of unique subreddits in test: {len(unique_subreddits_test)}")

# Check overlap
common_subreddits = set(unique_subreddits_train) & set(unique_subreddits_test)
train_only = set(unique_subreddits_train) - set(unique_subreddits_test)
test_only = set(unique_subreddits_test) - set(unique_subreddits_train)

print(f"\nSubreddits in both train and test: {len(common_subreddits)}")
print(f"Subreddits only in train: {len(train_only)}")
print(f"Subreddits only in test: {len(test_only)}")


# Subreddit distribution
subreddit_counts = train_df['subreddit'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top subreddits
top_subreddits = subreddit_counts.head(20)
axes[0].barh(range(len(top_subreddits)), top_subreddits.values)
axes[0].set_yticks(range(len(top_subreddits)))
axes[0].set_yticklabels(top_subreddits.index, fontsize=9)
axes[0].set_xlabel('Count')
axes[0].set_title('Top 20 Subreddits in Training Data')

# Violation rate by subreddit (top subreddits)
violation_by_subreddit = train_df.groupby('subreddit')['rule_violation'].agg(['mean', 'count'])
top_subreddits_violation = violation_by_subreddit.nlargest(20, 'count')

axes[1].barh(range(len(top_subreddits_violation)), 
             top_subreddits_violation['mean'].values)
axes[1].set_yticks(range(len(top_subreddits_violation)))
axes[1].set_yticklabels(top_subreddits_violation.index, fontsize=9)
axes[1].set_xlabel('Violation Rate')
axes[1].set_title('Violation Rate for Top 20 Subreddits')
axes[1].set_xlim([0, 1])

plt.tight_layout()
save_plot(fig)


# Check if examples are unique or repeated
print("Analyzing example uniqueness...")
print("="*50)

# Check positive examples
unique_pos1 = train_df['positive_example_1'].nunique()
unique_pos2 = train_df['positive_example_2'].nunique()
print(f"Unique positive_example_1: {unique_pos1} out of {len(train_df)}")
print(f"Unique positive_example_2: {unique_pos2} out of {len(train_df)}")

# Check negative examples
unique_neg1 = train_df['negative_example_1'].nunique()
unique_neg2 = train_df['negative_example_2'].nunique()
print(f"Unique negative_example_1: {unique_neg1} out of {len(train_df)}")
print(f"Unique negative_example_2: {unique_neg2} out of {len(train_df)}")


# Analyze example lengths
example_cols = ['positive_example_1', 'positive_example_2', 
                'negative_example_1', 'negative_example_2']

for col in example_cols:
    train_df[f'{col}_length'] = train_df[col].str.len()

# Compare example lengths
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for i, col in enumerate(example_cols):
    axes[i].hist(train_df[f'{col}_length'], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Length Distribution: {col}', fontsize=11)
    axes[i].set_xlabel('Character Count')
    axes[i].set_ylabel('Frequency')
    median_val = train_df[f'{col}_length'].median()
    axes[i].axvline(median_val, color='red', linestyle='--', 
                    label=f'Median: {median_val:.0f}')
    axes[i].legend()

plt.tight_layout()
save_plot(fig)


# Create features for correlation analysis
train_df['rule_length'] = train_df['rule'].str.len()
train_df['has_url'] = train_df['body'].str.contains(r'http[s]?://', na=False).astype(int)
train_df['has_caps'] = train_df['body'].str.isupper().astype(int)
train_df['exclamation_count'] = train_df['body'].str.count('!')
train_df['question_count'] = train_df['body'].str.count('\?')

# Select numerical features for correlation
corr_features = ['rule_violation', 'body_length', 'body_word_count', 
                 'rule_length', 'has_url', 'has_caps', 
                 'exclamation_count', 'question_count']

# Calculate correlation matrix
correlation_matrix = train_df[corr_features].corr()

# Visualize correlation matrix
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
save_plot(fig)


# Check for duplicates
print("Data Quality Checks:")
print("="*50)

# Duplicate rows
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()
print(f"Duplicate rows in train: {train_duplicates}")
print(f"Duplicate rows in test: {test_duplicates}")

# Duplicate comments
train_dup_comments = train_df['body'].duplicated().sum()
test_dup_comments = test_df['body'].duplicated().sum()
print(f"\nDuplicate comments in train: {train_dup_comments}")
print(f"Duplicate comments in test: {test_dup_comments}")

# Check for empty or very short comments
very_short_train = (train_df['body_length'] < 10).sum()
very_short_test = (test_df['body_length'] < 10).sum()
print(f"\nVery short comments (<10 chars) in train: {very_short_train}")
print(f"Very short comments (<10 chars) in test: {very_short_test}")


print("="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nğŸ“Š Dataset Sizes:")
print(f"  - Training samples: {len(train_df):,}")
print(f"  - Test samples: {len(test_df):,}")
print(f"  - Total samples: {len(train_df) + len(test_df):,}")

print("\nğŸ�¯ Target Distribution:")
print(f"  - Violations: {target_dist[1]:,} ({target_dist[1]/len(train_df)*100:.2f}%)")
print(f"  - No violations: {target_dist[0]:,} ({target_dist[0]/len(train_df)*100:.2f}%)")

print("\nğŸ“� Text Statistics:")
print(f"  - Avg comment length: {train_df['body_length'].mean():.1f} chars")
print(f"  - Avg word count: {train_df['body_word_count'].mean():.1f} words")
print(f"  - Max comment length: {train_df['body_length'].max():,} chars")
print(f"  - Min comment length: {train_df['body_length'].min()} chars")

print("\nğŸ“‹ Categories:")
print(f"  - Unique rules: {len(unique_rules_train)}")
print(f"  - Unique subreddits: {len(unique_subreddits_train)}")

print("\nğŸ”� Data Quality:")
print(f"  - Missing values in train: {train_df.isnull().sum().sum()}")
print(f"  - Missing values in test: {test_df.isnull().sum().sum()}")
print(f"  - Duplicate comments in train: {train_dup_comments}")

print("\n" + "="*60)

