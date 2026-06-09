# Enhanced Exploratory Data Analysis for Reddit Rule Violation Dataset
# This comprehensive EDA script analyzes patterns in Reddit comment rule violations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



# Load and prepare data
file_path = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
df = pd.read_csv(file_path)
data = df.copy()

print("Dataset Overview:")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\n" + "="*80 + "\n")



# =============================================================================
# FEATURE ENGINEERING - Creating additional metrics for analysis
# =============================================================================

# Text length features
data['body_length'] = data['body'].apply(len)
data['body_word_count'] = data['body'].apply(lambda x: len(str(x).split()))
data['pos_ex_1_len'] = data['positive_example_1'].apply(len)
data['pos_ex_2_len'] = data['positive_example_2'].apply(len)
data['neg_ex_1_len'] = data['negative_example_1'].apply(len)
data['neg_ex_2_len'] = data['negative_example_2'].apply(len)

# Advanced text features
data['avg_word_length'] = data['body'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))
data['sentence_count'] = data['body'].apply(lambda x: len(str(x).split('.')))
data['exclamation_count'] = data['body'].apply(lambda x: str(x).count('!'))
data['question_count'] = data['body'].apply(lambda x: str(x).count('?'))
data['caps_ratio'] = data['body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))

#



# =============================================================================
# 1. BASIC DISTRIBUTION ANALYSIS
# =============================================================================

# Create a single figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Rule Violation Distribution
violation_counts = data['rule_violation'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for no violation, red for violation
ax1.pie(violation_counts.values, labels=['No Violation', 'Violation'], 
        autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Distribution of Rule Violations\n(Class Balance Check)', fontsize=12, fontweight='bold')
# Explanation: Shows the proportion of violations vs non-violations. Important for understanding class imbalance.

# Plot 2: Top 10 Subreddits by Volume
top_subreddits = data['subreddit'].value_counts().head(10)
sns.barplot(x=top_subreddits.values, y=top_subreddits.index, palette='viridis', ax=ax2)
ax2.set_title('Top 10 Subreddits by Comment Volume', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Comments')
# Explanation: Identifies the most active subreddits in the dataset, helping understand data distribution.

# Plot 3: Rule Distribution
rule_counts = data['rule'].value_counts()
sns.barplot(x=rule_counts.values, y=rule_counts.index, palette='coolwarm', ax=ax3)
ax3.set_title('Distribution of Rule Types', fontsize=12, fontweight='bold')
ax3.set_xlabel('Frequency')
# Explanation: Shows which rules are most commonly referenced, indicating rule complexity or frequency.

# Plot 4: Violation Rate by Subreddit (Top 10)
violation_by_sub = data.groupby('subreddit')['rule_violation'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=violation_by_sub.values, y=violation_by_sub.index, palette='Reds_r', ax=ax4)
ax4.set_title('Top 10 Subreddits by Violation Rate', fontsize=12, fontweight='bold')
ax4.set_xlabel('Violation Rate')
# Explanation: Identifies subreddits with highest violation rates, useful for moderation focus.

# Adjust layout and display
plt.tight_layout()
plt.show()


# =============================================================================
# 2. TEXT LENGTH ANALYSIS
# =============================================================================

# Create a single figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Comment Length Distribution
sns.histplot(data=data, x='body_length', hue='rule_violation', bins=50, alpha=0.7, ax=ax1)
ax1.set_title('Comment Length Distribution by Violation Status', fontsize=12, fontweight='bold')
ax1.set_xlabel('Character Count')
# Explanation: Reveals if comment length correlates with rule violations (longer comments might be more complex).

# Plot 2: Word Count Distribution
sns.histplot(data=data, x='body_word_count', hue='rule_violation', bins=50, alpha=0.7, ax=ax2)
ax2.set_title('Word Count Distribution by Violation Status', fontsize=12, fontweight='bold')
ax2.set_xlabel('Word Count')
# Explanation: Shows relationship between verbosity and rule violations.

# Plot 3: Average Word Length Analysis
sns.boxplot(data=data, x='rule_violation', y='avg_word_length', palette='Set2', ax=ax3)
ax3.set_title('Average Word Length by Violation Status', fontsize=12, fontweight='bold')
ax3.set_xlabel('Rule Violation (0=No, 1=Yes)')
ax3.set_ylabel('Average Word Length')
# Explanation: Indicates if violating comments use more complex vocabulary.

# Plot 4: Punctuation Analysis
punct_data = pd.melt(data[['rule_violation', 'exclamation_count', 'question_count']], 
                     id_vars=['rule_violation'], var_name='punct_type', value_name='count')
sns.boxplot(data=punct_data, x='punct_type', y='count', hue='rule_violation', palette='Set1', ax=ax4)
ax4.set_title('Punctuation Usage by Violation Status', fontsize=12, fontweight='bold')
ax4.set_xticklabels(['Exclamations', 'Questions'])
# Explanation: Emotional punctuation might correlate with rule-violating content.

# Adjust layout and display
plt.tight_layout()
plt.show()


# =============================================================================
# 3. RULE-SPECIFIC ANALYSIS
# =============================================================================

# Create a single figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Violation Rate by Rule Type
rule_violation_rate = data.groupby('rule')['rule_violation'].mean().sort_values(ascending=False)
sns.barplot(x=rule_violation_rate.values, y=rule_violation_rate.index, palette='plasma', ax=ax1)
ax1.set_title('Violation Rate by Rule Type', fontsize=12, fontweight='bold')
ax1.set_xlabel('Violation Rate')
# Explanation: Shows which rules are most frequently violated, indicating enforcement challenges.

# Plot 2: Comment Length by Rule Type
sns.boxplot(data=data, x='rule', y='body_length', palette='tab10', ax=ax2)
ax2.tick_params(axis='x', rotation=45)
ax2.set_title('Comment Length Distribution by Rule Type', fontsize=12, fontweight='bold')
# Explanation: Different rules might attract different comment lengths/complexity.

# Plot 3: Rule vs Violation Heatmap
rule_violation_crosstab = pd.crosstab(data['rule'], data['rule_violation'])
sns.heatmap(rule_violation_crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax3)
ax3.set_title('Rule Type vs Violation Count Heatmap', fontsize=12, fontweight='bold')
# Explanation: Visual representation of violation patterns across different rules.

# Plot 4: Caps Usage Analysis
sns.boxplot(data=data, x='rule_violation', y='caps_ratio', palette='coolwarm', ax=ax4)
ax4.set_title('Capital Letters Ratio by Violation Status', fontsize=12, fontweight='bold')
ax4.set_xlabel('Rule Violation (0=No, 1=Yes)')
ax4.set_ylabel('Ratio of Capital Letters')
# Explanation: High caps usage might indicate aggressive/violating content.

# Adjust layout and display
plt.tight_layout()
plt.show()


# =============================================================================
# 4. EXAMPLE TEXT ANALYSIS
# =============================================================================

# Create a single figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Example Text Lengths Comparison
example_lengths = pd.DataFrame({
    'Positive_1': data['pos_ex_1_len'],
    'Positive_2': data['pos_ex_2_len'],
    'Negative_1': data['neg_ex_1_len'],
    'Negative_2': data['neg_ex_2_len']
})
sns.boxplot(data=example_lengths, palette='Set3', ax=ax1)
ax1.set_title('Example Text Lengths Distribution', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
# Explanation: Compares the length distribution of positive vs negative examples provided for rules.

# Plot 2: Example Length Correlation
corr_matrix = example_lengths.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=ax2)
ax2.set_title('Correlation Between Example Lengths', fontsize=12, fontweight='bold')
# Explanation: Shows if example lengths are consistent within positive/negative categories.

# Plot 3: Body Length vs Example Lengths
ax3.scatter(data['body_length'], data['pos_ex_1_len'], alpha=0.5, label='Positive Ex 1', s=10)
ax3.scatter(data['body_length'], data['neg_ex_1_len'], alpha=0.5, label='Negative Ex 1', s=10)
ax3.set_xlabel('Comment Body Length')
ax3.set_ylabel('Example Length')
ax3.set_title('Comment Length vs Example Lengths', fontsize=12, fontweight='bold')
ax3.legend()
# Explanation: Examines if longer comments have correspondingly longer examples.

# Plot 4: Statistical Distribution Test
violation_lengths = data[data['rule_violation'] == 1]['body_length']
no_violation_lengths = data[data['rule_violation'] == 0]['body_length']
sns.kdeplot(violation_lengths, label='Violations', fill=True, ax=ax4)
sns.kdeplot(no_violation_lengths, label='No Violations', fill=True, ax=ax4)
ax4.set_title('Length Distribution Density Comparison', fontsize=12, fontweight='bold')
ax4.set_xlabel('Comment Length')
ax4.legend()
# Explanation: Kernel density estimation to see if length distributions differ significantly.

# Adjust layout and display
plt.tight_layout()
plt.show()


# =============================================================================
# 5. ADVANCED STATISTICAL ANALYSIS
# =============================================================================

# Create a single figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Violation Rate Trend by Comment Length Bins
data['length_bins'] = pd.cut(data['body_length'], bins=10)
violation_by_length = data.groupby('length_bins')['rule_violation'].mean()
bin_centers = [interval.mid for interval in violation_by_length.index]
ax1.plot(bin_centers, violation_by_length.values, marker='o', linewidth=2, markersize=6)
ax1.set_title('Violation Rate by Comment Length Bins', fontsize=12, fontweight='bold')
ax1.set_xlabel('Comment Length (bin centers)')
ax1.set_ylabel('Violation Rate')
ax1.tick_params(axis='x', rotation=45)
# Explanation: Shows how violation probability changes with comment length.

# Plot 2: Subreddit Diversity Analysis
subreddit_stats = data.groupby('subreddit').agg({
    'rule_violation': ['count', 'mean']
}).round(3)
subreddit_stats.columns = ['total_comments', 'violation_rate']
subreddit_stats = subreddit_stats[subreddit_stats['total_comments'] >= 10]  # Filter for reliability

ax2.scatter(subreddit_stats['total_comments'], subreddit_stats['violation_rate'], 
           alpha=0.6, s=50)
ax2.set_xlabel('Total Comments')
ax2.set_ylabel('Violation Rate')
ax2.set_title('Subreddit Activity vs Violation Rate', fontsize=12, fontweight='bold')
# Explanation: Shows relationship between subreddit activity and moderation issues.

# Plot 3: Text Complexity Analysis
sns.scatterplot(data=data, x='body_word_count', y='avg_word_length', 
               hue='rule_violation', alpha=0.6, ax=ax3)
ax3.set_title('Text Complexity: Word Count vs Avg Word Length', fontsize=12, fontweight='bold')
# Explanation: Explores if violating comments have different complexity patterns.

# Plot 4: Feature Importance Proxy
features = ['body_length', 'body_word_count', 'avg_word_length', 'caps_ratio', 
           'exclamation_count', 'question_count']
feature_correlations = [abs(data[feature].corr(data['rule_violation'])) for feature in features]
sns.barplot(x=feature_correlations, y=features, palette='viridis', ax=ax4)
ax4.set_title('Feature Correlation with Rule Violations', fontsize=12, fontweight='bold')
ax4.set_xlabel('Absolute Correlation')
# Explanation: Shows which engineered features correlate most with violations.

# Adjust layout and display
plt.tight_layout()
plt.show()


# =============================================================================
# 6. WORD CLOUD ANALYSIS
# =============================================================================

print("Generating Word Clouds...")

# Create a single figure with 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Word Cloud for Violations
violation_text = " ".join(data[data['rule_violation'] == 1]['body'].astype(str))
if violation_text.strip():
    wordcloud_viol = WordCloud(width=800, height=400, background_color='white', 
                              colormap='Reds', max_words=100).generate(violation_text)
    ax1.imshow(wordcloud_viol, interpolation='bilinear')
    ax1.axis('off')
    ax1.set_title("Word Cloud - Rule Violating Comments", fontsize=16, fontweight='bold')
else:
    ax1.text(0.5, 0.5, 'No violation text available', ha='center', va='center', 
             transform=ax1.transAxes, fontsize=14)
    ax1.set_title("Word Cloud - Rule Violating Comments", fontsize=16, fontweight='bold')

# Word Cloud for Non-Violations
non_violation_text = " ".join(data[data['rule_violation'] == 0]['body'].astype(str))
if non_violation_text.strip():
    wordcloud_non_viol = WordCloud(width=800, height=400, background_color='white', 
                                  colormap='Blues', max_words=100).generate(non_violation_text)
    ax2.imshow(wordcloud_non_viol, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title("Word Cloud - Non-Violating Comments", fontsize=16, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'No non-violation text available', ha='center', va='center', 
             transform=ax2.transAxes, fontsize=14)
    ax2.set_title("Word Cloud - Non-Violating Comments", fontsize=16, fontweight='bold')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Explanation: Visual comparison of common words in violating vs non-violating comments.
# This helps identify language patterns that distinguish rule violations from compliant content.


# =============================================================================
# 7. ADVANCED TEXT ANALYSIS
# =============================================================================

print("Analyzing Text Patterns...")

# Create a single figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Top Bigrams in Violations
try:
    vectorizer_bigrams = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=15)
    violation_bodies = data[data['rule_violation'] == 1]['body'].astype(str)
    X_bigrams = vectorizer_bigrams.fit_transform(violation_bodies)
    
    bigrams = vectorizer_bigrams.get_feature_names_out()
    frequencies = X_bigrams.sum(axis=0).A1
    
    bigram_df = pd.DataFrame({'bigram': bigrams, 'frequency': frequencies}).sort_values('frequency', ascending=True)
    
    ax1.barh(bigram_df['bigram'], bigram_df['frequency'], color='coral')
    ax1.set_title("Top 15 Bigrams in Violating Comments", fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frequency')
    # Explanation: Common two-word phrases in rule violations, useful for pattern detection.

except Exception as e:
    ax1.text(0.5, 0.5, 'Insufficient text data for bigram analysis', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.set_title("Top 15 Bigrams in Violating Comments", fontsize=14, fontweight='bold')

# Plot 2: Top Bigrams in Non-Violations
try:
    vectorizer_bigrams_clean = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=15)
    non_violation_bodies = data[data['rule_violation'] == 0]['body'].astype(str)
    X_bigrams_clean = vectorizer_bigrams_clean.fit_transform(non_violation_bodies)
    
    bigrams_clean = vectorizer_bigrams_clean.get_feature_names_out()
    frequencies_clean = X_bigrams_clean.sum(axis=0).A1
    
    bigram_clean_df = pd.DataFrame({'bigram': bigrams_clean, 'frequency': frequencies_clean}).sort_values('frequency', ascending=True)
    
    ax2.barh(bigram_clean_df['bigram'], bigram_clean_df['frequency'], color='lightblue')
    ax2.set_title("Top 15 Bigrams in Non-Violating Comments", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frequency')
    # Explanation: Common phrases in acceptable comments for contrast.

except Exception as e:
    ax2.text(0.5, 0.5, 'Insufficient text data for bigram analysis', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title("Top 15 Bigrams in Non-Violating Comments", fontsize=14, fontweight='bold')

# Plot 3: Length Distribution by Top Subreddits
top_5_subs = data['subreddit'].value_counts().head(5).index
subset_data = data[data['subreddit'].isin(top_5_subs)]

sns.violinplot(data=subset_data, x='subreddit', y='body_length', hue='rule_violation', 
               split=True, ax=ax3, palette='Set1')
ax3.set_title('Comment Length Distribution in Top 5 Subreddits', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
# Explanation: Shows length patterns across major subreddits, split by violation status.

# Plot 4: Feature Distribution Comparison
features_to_plot = ['body_word_count', 'avg_word_length', 'caps_ratio']
violin_data = data[features_to_plot + ['rule_violation']].melt(id_vars=['rule_violation'], 
                                                               var_name='feature', 
                                                               value_name='value')

sns.boxplot(data=violin_data, x='feature', y='value', hue='rule_violation', ax=ax4, palette='coolwarm')
ax4.set_title('Feature Distributions by Violation Status', fontsize=14, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
# Explanation: Comparative view of key engineered features across violation classes.

# Adjust layout and display
plt.tight_layout()
plt.show()


# =============================================================================
# 8. SUMMARY STATISTICS & KEY INSIGHTS
# =============================================================================

print("\n" + "="*80)
print("ğŸ“Š REDDIT RULE VIOLATION DATASET SUMMARY STATISTICS")
print("="*80)

# Basic Dataset Information
print(f"\nğŸ”� BASIC DATASET INFO:")
print(f"{'Total comments:':<25} {len(data):,}")
print(f"{'Rule violations:':<25} {data['rule_violation'].sum():,} ({data['rule_violation'].mean()*100:.1f}%)")
print(f"{'Compliant comments:':<25} {(len(data) - data['rule_violation'].sum()):,} ({(1-data['rule_violation'].mean())*100:.1f}%)")
print(f"{'Unique subreddits:':<25} {data['subreddit'].nunique()}")
print(f"{'Unique rules:':<25} {data['rule'].nunique()}")

# Text Length Statistics
print(f"\nğŸ“� TEXT LENGTH STATISTICS:")
print(f"{'Avg comment length:':<25} {data['body_length'].mean():.1f} characters")
print(f"{'Median comment length:':<25} {data['body_length'].median():.1f} characters")
print(f"{'Avg word count:':<25} {data['body_word_count'].mean():.1f} words")
print(f"{'Avg word length:':<25} {data['avg_word_length'].mean():.1f} characters")
print(f"{'Avg caps ratio:':<25} {data['caps_ratio'].mean()*100:.1f}%")

# Statistical Comparison: Violations vs Non-Violations
print(f"\nâš–ï¸� VIOLATION vs NON-VIOLATION COMPARISON:")
violation_stats = data.groupby('rule_violation')[['body_length', 'body_word_count', 'avg_word_length', 'caps_ratio']].mean()
violation_stats.index = ['Non-Violation', 'Violation']
print(violation_stats.round(2))

# Calculate statistical significance
from scipy import stats
violation_lengths = data[data['rule_violation'] == 1]['body_length']
non_violation_lengths = data[data['rule_violation'] == 0]['body_length']
t_stat, p_value = stats.ttest_ind(violation_lengths, non_violation_lengths)
print(f"\nğŸ“ˆ STATISTICAL SIGNIFICANCE:")
print(f"{'Length difference p-value:':<25} {p_value:.2e}")
print(f"{'Statistically significant:':<25} {'Yes' if p_value < 0.05 else 'No'}")

# Top Problematic Subreddits
print(f"\nğŸš¨ TOP 5 PROBLEMATIC SUBREDDITS (by violation rate):")
top_problem_subs = data.groupby('subreddit')['rule_violation'].agg(['count', 'mean']).query('count >= 10').sort_values('mean', ascending=False).head()
top_problem_subs.columns = ['Total_Comments', 'Violation_Rate']
print(top_problem_subs.round(3))

# Best Performing Subreddits
print(f"\nâœ… TOP 5 BEST PERFORMING SUBREDDITS (by compliance rate):")
best_subs = data.groupby('subreddit')['rule_violation'].agg(['count', 'mean']).query('count >= 10').sort_values('mean', ascending=True).head()
best_subs.columns = ['Total_Comments', 'Violation_Rate']
print(best_subs.round(3))

# Rule Difficulty Analysis
print(f"\nğŸ“‹ RULE DIFFICULTY RANKING (by violation rate):")
rule_difficulty = data.groupby('rule')['rule_violation'].agg(['count', 'mean']).sort_values('mean', ascending=False)
rule_difficulty.columns = ['Total_Cases', 'Violation_Rate']
print(rule_difficulty.round(3))

# Feature Correlation Summary
print(f"\nğŸ”— FEATURE CORRELATION WITH VIOLATIONS:")
features = ['body_length', 'body_word_count', 'avg_word_length', 'caps_ratio', 'exclamation_count', 'question_count']
correlations = []
for feature in features:
    corr = data[feature].corr(data['rule_violation'])
    correlations.append((feature, corr))
    print(f"{feature:<20}: {corr:>8.3f}")

# Key Insights Summary
print(f"\nğŸ’¡ KEY INSIGHTS:")

# Class balance insight
violation_rate = data['rule_violation'].mean()
if violation_rate < 0.2:
    balance_insight = "Highly imbalanced dataset - consider stratified sampling"
elif violation_rate < 0.4:
    balance_insight = "Moderately imbalanced - standard techniques should work"
else:
    balance_insight = "Relatively balanced dataset"
print(f"â€¢ Class Balance: {balance_insight}")

# Length insight
avg_violation_length = data[data['rule_violation'] == 1]['body_length'].mean()
avg_compliant_length = data[data['rule_violation'] == 0]['body_length'].mean()
if avg_violation_length > avg_compliant_length * 1.1:
    length_insight = "Violating comments tend to be longer"
elif avg_violation_length < avg_compliant_length * 0.9:
    length_insight = "Violating comments tend to be shorter"
else:
    length_insight = "No strong length pattern in violations"
print(f"â€¢ Length Pattern: {length_insight}")

# Subreddit insight
most_problematic = top_problem_subs.index[0]
highest_rate = top_problem_subs.iloc[0]['Violation_Rate']
print(f"â€¢ Most Problematic Subreddit: {most_problematic} ({highest_rate:.1%} violation rate)")

# Rule insight
most_difficult_rule = rule_difficulty.index[0]
rule_violation_rate = rule_difficulty.iloc[0]['Violation_Rate']
print(f"â€¢ Most Difficult Rule: {most_difficult_rule} ({rule_violation_rate:.1%} violation rate)")

# Strongest predictor
strongest_feature = max(correlations, key=lambda x: abs(x[1]))
print(f"â€¢ Strongest Predictor: {strongest_feature[0]} (correlation: {strongest_feature[1]:.3f})")

print(f"\nğŸ�¯ MODELING RECOMMENDATIONS:")
print("â€¢ Use stratified cross-validation due to class imbalance")
print("â€¢ Consider text length and complexity features")
print("â€¢ Implement subreddit-specific models or features")
print("â€¢ Focus on rule-specific patterns for better performance")
print("â€¢ Use ensemble methods to capture different violation patterns")



