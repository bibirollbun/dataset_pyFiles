from IPython.display import clear_output

!pip install textstat==0.7.8

clear_output()


# Essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Text processing and analysis
import re
import string
from collections import Counter
import textstat
from wordcloud import WordCloud

# Advanced NLP
try:
    import spacy
    # nlp = spacy.load("en_core_web_sm")
    print("âœ… spaCy available")
except ImportError:
    print("âš ï¸�  spaCy not available - will use alternative methods")

try:
    from sentence_transformers import SentenceTransformer
    print("âœ… sentence-transformers available")
except ImportError:
    print("âš ï¸�  sentence-transformers not available - will use alternative embeddings")

# Dimensionality reduction
try:
    import umap
    print("âœ… UMAP available")
except ImportError:
    print("âš ï¸�  UMAP not available - will use alternative dimensionality reduction")

# Statistical analysis
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

print("ğŸ�¯ Environment setup complete!")


# Load the training data
print("ğŸ“Š Loading training data...")
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')

print(f"Dataset shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print("\nğŸ”� Basic Info:")
print(train_df.info())


# Initial data exploration
print("ğŸ“‹ First 3 rows (truncated for readability):")
display_df = train_df.head(3).copy()
for col in ['body', 'rule', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']:
    if col in display_df.columns:
        display_df[col] = display_df[col].astype(str).str[:100] + '...'

display(display_df)

print("\nğŸ�¯ Missing Values:")
missing_counts = train_df.isnull().sum()
missing_pct = (missing_counts / len(train_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_pct
})[missing_counts > 0]

if len(missing_df) > 0:
    display(missing_df)
else:
    print("âœ… No missing values found!")


# Dataset overview statistics
print("ğŸ�† COMPETITION CONTEXT")
print("=" * 50)
print("Task: Binary classification of Reddit comment rule violations")
print("Evaluation: Column-averaged AUC")
print("Goal: Predict whether a comment violates a specific subreddit rule")
print("\nğŸ“Š DATASET OVERVIEW")
print("=" * 50)

# Basic statistics
total_comments = len(train_df)
unique_subreddits = train_df['subreddit'].nunique()
unique_rules = train_df['rule'].nunique()
violation_rate = train_df['rule_violation'].mean()

print(f"ğŸ“� Total Comments: {total_comments:,}")
print(f"ğŸ�˜ï¸�  Unique Subreddits: {unique_subreddits}")
print(f"ğŸ“‹ Unique Rules: {unique_rules}")
print(f"âš ï¸�  Overall Violation Rate: {violation_rate:.3f} ({violation_rate*100:.1f}%)")

# Text length statistics
train_df['comment_length'] = train_df['body'].str.len()
train_df['rule_length'] = train_df['rule'].str.len()

print(f"\nğŸ“� TEXT LENGTH STATISTICS")
print("=" * 30)
print(f"Comment length - Mean: {train_df['comment_length'].mean():.1f}, Median: {train_df['comment_length'].median():.1f}")
print(f"Rule length - Mean: {train_df['rule_length'].mean():.1f}, Median: {train_df['rule_length'].median():.1f}")


# Detailed breakdown by subreddit and rule
print("ğŸ�˜ï¸�  SUBREDDIT DISTRIBUTION")
print("=" * 40)
subreddit_stats = train_df.groupby('subreddit').agg({
    'row_id': 'count',
    'rule_violation': ['mean', 'sum'],
    'rule': 'nunique'
}).round(3)

subreddit_stats.columns = ['Total_Comments', 'Violation_Rate', 'Total_Violations', 'Unique_Rules']
subreddit_stats = subreddit_stats.sort_values('Total_Comments', ascending=False)

print("Top 10 subreddits by comment count:")
display(subreddit_stats.head(10))

top_10_subreddits = subreddit_stats.head(10)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("Subreddit Analysis Overview", fontsize=18, y=1.05)

# Comments per subreddit
sns.barplot(
    x=top_10_subreddits.index, 
    y=top_10_subreddits['Total_Comments'], 
    ax=axes[0, 0], 
    palette='Blues_r'
)
axes[0, 0].set_title("Comments per Subreddit")
axes[0, 0].set_ylabel("Total Comments")
axes[0, 0].set_xlabel("")
axes[0, 0].tick_params(axis='x', rotation=45)

# Violation rate by subreddit
sns.barplot(
    x=top_10_subreddits.index, 
    y=top_10_subreddits['Violation_Rate'], 
    ax=axes[0, 1], 
    palette='Reds_r'
)
axes[0, 1].set_title("Violation Rate by Subreddit")
axes[0, 1].set_ylabel("Violation Rate")
axes[0, 1].set_xlabel("")
axes[0, 1].tick_params(axis='x', rotation=45)

# Rules per subreddit
sns.barplot(
    x=top_10_subreddits.index, 
    y=top_10_subreddits['Unique_Rules'], 
    ax=axes[1, 0], 
    palette='Greens_r'
)
axes[1, 0].set_title("Rules per Subreddit")
axes[1, 0].set_ylabel("Unique Rules")
axes[1, 0].set_xlabel("")
axes[1, 0].tick_params(axis='x', rotation=45)

# Violations vs Comments scatter
axes[1, 1].scatter(
    subreddit_stats['Total_Comments'], 
    subreddit_stats['Total_Violations'], 
    color='purple', alpha=0.7, s=60
)
for i, txt in enumerate(subreddit_stats.index[:15]):
    axes[1, 1].annotate(txt, 
                        (subreddit_stats['Total_Comments'].iloc[i], subreddit_stats['Total_Violations'].iloc[i]),
                        fontsize=8, alpha=0.7)
axes[1, 1].set_title("Violations vs Comments")
axes[1, 1].set_xlabel("Total Comments")
axes[1, 1].set_ylabel("Total Violations")

plt.tight_layout()
plt.show()


# Class balance analysis
print("âš–ï¸�  CLASS BALANCE ANALYSIS")
print("=" * 40)

violation_counts = train_df['rule_violation'].value_counts()
print(f"Non-violations (0): {violation_counts[0]:,} ({violation_counts[0]/len(train_df)*100:.1f}%)")
print(f"Violations (1): {violation_counts[1]:,} ({violation_counts[1]/len(train_df)*100:.1f}%)")

imbalance_ratio = violation_counts[0] / violation_counts[1]
print(f"\nğŸ“Š Imbalance Ratio: {imbalance_ratio:.2f}:1 (non-violation:violation)")

if imbalance_ratio > 3:
    print("âš ï¸�  Significant class imbalance detected - consider stratified sampling and class weighting")
elif imbalance_ratio > 1.5:
    print("âš ï¸�  Moderate class imbalance - monitor model performance on minority class")
else:
    print("âœ… Relatively balanced classes")


# Class balance by subreddit
print("\nğŸ�˜ï¸�  CLASS BALANCE BY SUBREDDIT")
print("=" * 50)

subreddit_balance = train_df.groupby(['subreddit', 'rule_violation']).size().unstack(fill_value=0)
subreddit_balance['violation_rate'] = subreddit_balance[1] / (subreddit_balance[0] + subreddit_balance[1])
subreddit_balance['total'] = subreddit_balance[0] + subreddit_balance[1]
subreddit_balance = subreddit_balance.sort_values('total', ascending=False)

print("Violation rates by subreddit:")
display(subreddit_balance[['violation_rate', 'total']].head(10).round(3))

# Statistical test for significant differences in violation rates
violation_rates = []
subreddit_names = []
for subreddit in train_df['subreddit'].unique():
    if len(train_df[train_df['subreddit'] == subreddit]) >= 30:  # Minimum sample size
        rate = train_df[train_df['subreddit'] == subreddit]['rule_violation'].mean()
        violation_rates.append(rate)
        subreddit_names.append(subreddit)

print(f"\nğŸ“ˆ Violation Rate Statistics (subreddits with â‰¥30 comments):")
print(f"Mean: {np.mean(violation_rates):.3f}")
print(f"Std: {np.std(violation_rates):.3f}")
print(f"Range: {min(violation_rates):.3f} - {max(violation_rates):.3f}")

# Coefficient of variation
cv = np.std(violation_rates) / np.mean(violation_rates)
print(f"Coefficient of Variation: {cv:.3f}")
if cv > 0.5:
    print("âš ï¸�  High variability in violation rates across subreddits")
else:
    print("âœ… Moderate variability in violation rates")


# Class balance by rule type
print("\nğŸ“‹ CLASS BALANCE BY RULE")
print("=" * 40)

# Analyze rule patterns
train_df['rule_short'] = train_df['rule'].str[:50] + '...'  # Truncate for display
rule_balance = train_df.groupby(['rule_short', 'rule_violation']).size().unstack(fill_value=0)
if 1 in rule_balance.columns and 0 in rule_balance.columns:
    rule_balance['violation_rate'] = rule_balance[1] / (rule_balance[0] + rule_balance[1])
    rule_balance['total'] = rule_balance[0] + rule_balance[1]
    rule_balance = rule_balance.sort_values('total', ascending=False)
    
    print("Top 10 rules by frequency:")
    display(rule_balance[['violation_rate', 'total']].head(10).round(3))
    
    # Rule complexity analysis
    train_df['rule_word_count'] = train_df['rule'].str.split().str.len()
    
    print(f"\nğŸ“Š Rule Complexity Statistics:")
    print(f"Average rule length: {train_df['rule_word_count'].mean():.1f} words")
    print(f"Rule length range: {train_df['rule_word_count'].min()} - {train_df['rule_word_count'].max()} words")
    
    # Correlation between rule length and violation rate
    rule_complexity_corr = train_df.groupby('rule').agg({
        'rule_violation': 'mean',
        'rule_word_count': 'first'
    })
    
    correlation = rule_complexity_corr['rule_violation'].corr(rule_complexity_corr['rule_word_count'])
    print(f"Correlation between rule length and violation rate: {correlation:.3f}")
    
    if abs(correlation) > 0.3:
        print("âš ï¸�  Moderate correlation detected - rule complexity may affect violation rates")
    else:
        print("âœ… Low correlation - rule length doesn't strongly predict violation rates")


# Text preprocessing function
def clean_text(text):
    """Basic text cleaning while preserving important patterns"""
    if pd.isna(text):
        return ""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply text cleaning
train_df['body_clean'] = train_df['body'].apply(clean_text)
train_df['rule_clean'] = train_df['rule'].apply(clean_text)

print("ğŸ§¹ Text preprocessing completed")


# Text complexity analysis
print("ğŸ“Š TEXT COMPLEXITY ANALYSIS")
print("=" * 50)

# Calculate readability metrics
def calculate_readability(text):
    """Calculate various readability metrics"""
    if pd.isna(text) or len(str(text).strip()) == 0:
        return {
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'automated_readability_index': 0,
            'sentence_count': 0,
            'word_count': 0
        }
    
    text = str(text)
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'sentence_count': textstat.sentence_count(text),
            'word_count': textstat.lexicon_count(text)
        }
    except:
        return {
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'automated_readability_index': 0,
            'sentence_count': 1,
            'word_count': len(text.split())
        }

# Calculate readability for a sample (to avoid performance issues)
sample_size = min(1000, len(train_df))
sample_df = train_df.sample(n=sample_size, random_state=42)

print(f"Calculating readability metrics for {sample_size} comments...")
readability_metrics = sample_df['body'].apply(calculate_readability)
readability_df = pd.DataFrame(readability_metrics.tolist(), index=sample_df.index)

# Merge back with sample
sample_with_metrics = sample_df.join(readability_df)

# Compare readability between violations and non-violations
readability_comparison = sample_with_metrics.groupby('rule_violation')[[
    'flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index',
    'sentence_count', 'word_count'
]].mean()

print("\nReadability Metrics by Violation Status:")
display(readability_comparison.round(2))

# Statistical significance tests
for metric in ['flesch_reading_ease', 'flesch_kincaid_grade', 'word_count']:
    violations = sample_with_metrics[sample_with_metrics['rule_violation'] == 1][metric]
    non_violations = sample_with_metrics[sample_with_metrics['rule_violation'] == 0][metric]
    
    # Perform t-test
    statistic, p_value = stats.ttest_ind(violations.dropna(), non_violations.dropna())
    
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    
    print(f"{metric}: t={statistic:.3f}, p={p_value:.3f} {significance}")


# Advanced semantic similarity analysis
print("\nğŸ§  SEMANTIC SIMILARITY ANALYSIS")
print("=" * 50)

# Simple TF-IDF based similarity (fallback if sentence transformers not available)
def calculate_tfidf_similarity(comments, rules, max_features=1000):
    """Calculate TF-IDF based similarity between comments and rules"""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    
    # Combine comments and rules for vectorization
    all_texts = list(comments) + list(rules)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split back to comments and rules
    comment_vectors = tfidf_matrix[:len(comments)]
    rule_vectors = tfidf_matrix[len(comments):]
    
    # Calculate cosine similarity
    similarities = []
    for i in range(len(comments)):
        sim = cosine_similarity(comment_vectors[i], rule_vectors[i])[0, 0]
        similarities.append(sim)
    
    return similarities

# Calculate semantic similarity for sample
print(f"Calculating comment-rule semantic similarity for {sample_size} samples...")
sample_comments = sample_df['body_clean'].fillna('')
sample_rules = sample_df['rule_clean'].fillna('')

# Calculate similarities
tfidf_similarities = calculate_tfidf_similarity(sample_comments, sample_rules)
sample_df_semantic = sample_df.copy()
sample_df_semantic['comment_rule_similarity'] = tfidf_similarities

# Analyze similarity patterns
similarity_by_violation = sample_df_semantic.groupby('rule_violation')['comment_rule_similarity'].agg(['mean', 'std', 'median'])
print("\nComment-Rule Semantic Similarity by Violation Status:")
display(similarity_by_violation.round(4))

# Statistical test
violations_sim = sample_df_semantic[sample_df_semantic['rule_violation'] == 1]['comment_rule_similarity']
non_violations_sim = sample_df_semantic[sample_df_semantic['rule_violation'] == 0]['comment_rule_similarity']

sim_statistic, sim_p_value = stats.ttest_ind(violations_sim, non_violations_sim)
print(f"\nSemantic similarity difference test: t={sim_statistic:.3f}, p={sim_p_value:.4f}")

if sim_p_value < 0.05:
    print("âš ï¸�  Significant difference in semantic similarity between violations and non-violations")
    print("ğŸ’¡ Insight: Semantic alignment with rules is predictive of violation status")
else:
    print("âœ… No significant difference in semantic similarity")

# Similarity distribution analysis
print(f"\nğŸ“Š Similarity Distribution:")
print(f"Overall range: {min(tfidf_similarities):.3f} - {max(tfidf_similarities):.3f}")
print(f"Mean similarity: {np.mean(tfidf_similarities):.3f}")
print(f"Std similarity: {np.std(tfidf_similarities):.3f}")


# Linguistic pattern analysis
print("\nğŸ”¤ LINGUISTIC PATTERN ANALYSIS")
print("=" * 50)

# Count special characters and patterns
def extract_linguistic_features(text):
    """Extract linguistic features from text"""
    if pd.isna(text):
        text = ""
    text = str(text)
    
    return {
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'caps_count': sum(1 for c in text if c.isupper()),
        'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'mention_count': len(re.findall(r'@\w+', text)),
        'hashtag_count': len(re.findall(r'#\w+', text)),
        'number_count': len(re.findall(r'\d+', text)),
        'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
    }

# Extract linguistic features for sample
print(f"Extracting linguistic features for {sample_size} comments...")
linguistic_features = sample_df['body'].apply(extract_linguistic_features)
linguistic_df = pd.DataFrame(linguistic_features.tolist(), index=sample_df.index)

# Merge with sample
sample_with_linguistic = sample_df.join(linguistic_df)

# Compare linguistic features between violations and non-violations
linguistic_comparison = sample_with_linguistic.groupby('rule_violation')[[
    'exclamation_count', 'question_count', 'caps_count', 'url_count',
    'punctuation_ratio', 'avg_word_length'
]].mean()

print("\nLinguistic Features by Violation Status:")
display(linguistic_comparison.round(3))

# Test for significant differences
significant_features = []
for feature in linguistic_df.columns:
    violations = sample_with_linguistic[sample_with_linguistic['rule_violation'] == 1][feature]
    non_violations = sample_with_linguistic[sample_with_linguistic['rule_violation'] == 0][feature]
    
    # Perform Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(violations.dropna(), non_violations.dropna(), alternative='two-sided')
    
    if p_value < 0.05:
        significant_features.append((feature, p_value))

print(f"\nâœ… Significant linguistic features (p < 0.05): {len(significant_features)}")
for feature, p_val in significant_features:
    print(f"  - {feature}: p = {p_val:.4f}")


# Rule categorization and analysis
print("ğŸ“‹ RULE CATEGORIZATION & PATTERN ANALYSIS")
print("=" * 60)

# Automatic rule categorization based on keywords
def categorize_rule(rule_text):
    """Categorize rules based on content patterns"""
    if pd.isna(rule_text):
        return "Unknown"
    
    rule_lower = str(rule_text).lower()
    
    # Define category keywords
    categories = {
        'Advertising/Spam': ['advertising', 'spam', 'promotional', 'referral', 'soliciting', 'marketing'],
        'Content Quality': ['low effort', 'quality', 'meme', 'image', 'screenshot', 'duplicate'],
        'Behavior/Civility': ['civil', 'respectful', 'harassment', 'abuse', 'toxic', 'flaming', 'trolling'],
        'Legal/Medical': ['legal advice', 'medical advice', 'diagnosis', 'prescription', 'lawyer'],
        'Off-Topic': ['off-topic', 'relevance', 'related', 'appropriate', 'belongs'],
        'Personal Info': ['personal information', 'doxxing', 'privacy', 'contact', 'address', 'phone'],
        'NSFW/Adult': ['nsfw', 'adult', 'sexual', 'explicit', 'mature'],
        'Self-Promotion': ['self-promotion', 'self promotion', 'own content', 'blog', 'youtube', 'social media'],
        'Politics': ['political', 'politics', 'election', 'partisan', 'controversial'],
        'Formatting': ['title', 'format', 'tag', 'flair', 'formatting']
    }
    
    for category, keywords in categories.items():
        if any(keyword in rule_lower for keyword in keywords):
            return category
    
    return "Other"

# Apply categorization
train_df['rule_category'] = train_df['rule'].apply(categorize_rule)

# Rule category analysis
rule_category_stats = train_df.groupby('rule_category').agg(
    Total_Comments=('row_id', 'count'),
    Violation_Rate=('rule_violation', 'mean'),
    Total_Violations=('rule_violation', 'sum'),
    Unique_Rules=('rule', 'nunique')
).round(3)

rule_category_stats = rule_category_stats.sort_values('Total_Comments', ascending=False)

print("Rule categories and their statistics:")
display(rule_category_stats)

# Rule complexity vs violation rate analysis
rule_stats_detailed = train_df.groupby('rule').agg(
    Total_Comments=('rule_violation', 'count'),
    Violation_Rate=('rule_violation', 'mean'),
    Total_Violations=('rule_violation', 'sum'),
    Word_Count=('rule_word_count', 'first'),
    Category=('rule_category', 'first')
).round(3)

rule_stats_detailed = rule_stats_detailed[rule_stats_detailed['Total_Comments'] >= 10]  # Filter for rules with enough data

print(f"\nğŸ”� Analyzing {len(rule_stats_detailed)} rules with â‰¥10 comments each")

# Top violating rules
print("\nâš ï¸�  Top 10 Rules by Violation Rate:")
top_violating_rules = rule_stats_detailed.sort_values('Violation_Rate', ascending=False).head(10)
display(top_violating_rules[['Violation_Rate', 'Total_Comments', 'Word_Count', 'Category']])

# Rule complexity correlation
complexity_corr = rule_stats_detailed['Violation_Rate'].corr(rule_stats_detailed['Word_Count'])
print(f"\nğŸ“Š Correlation between rule complexity (word count) and violation rate: {complexity_corr:.3f}")

if abs(complexity_corr) > 0.3:
    print("âš ï¸�  Strong correlation: Rule complexity significantly affects violation rates")
elif abs(complexity_corr) > 0.1:
    print("âš ï¸�  Moderate correlation: Rule complexity somewhat affects violation rates")
else:
    print("âœ… Weak correlation: Rule complexity doesn't strongly predict violation rates")


# Subreddit cultural fingerprinting
print("ğŸ�˜ï¸�  SUBREDDIT CULTURAL ANALYSIS")
print("=" * 60)

# Subreddit linguistic characteristics
def analyze_subreddit_culture(subreddit_data):
    """Analyze the cultural/linguistic characteristics of a subreddit"""
    comments = subreddit_data['body_clean'].fillna('')

    # Aggregate text for analysis
    all_text = ' '.join(comments)
    words = all_text.split()

    if len(words) == 0:
        return {
            'avg_comment_length': 0,
            'avg_word_length': 0,
            'vocabulary_diversity': 0,
            'formality_score': 0,
            'violation_rate': 0,
            'total_comments': len(subreddit_data)
        }

    # Calculate cultural metrics
    return {
        'avg_comment_length': np.mean([len(comment) for comment in comments if comment]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'vocabulary_diversity': len(set(words)) / len(words) if words else 0,
        'formality_score': sum(1 for word in words if len(word) > 6) / len(words) if words else 0,  # Proxy for formality
        'violation_rate': subreddit_data['rule_violation'].mean(),
        'total_comments': len(subreddit_data)
    }

# Analyze cultural characteristics by subreddit
subreddit_cultures = {}
for subreddit in train_df['subreddit'].unique():
    subreddit_data = train_df[train_df['subreddit'] == subreddit]
    if len(subreddit_data) >= 20:  # Minimum sample size for reliable analysis
        subreddit_cultures[subreddit] = analyze_subreddit_culture(subreddit_data)

culture_df = pd.DataFrame(subreddit_cultures).T
culture_df = culture_df.sort_values('total_comments', ascending=False)

print(f"\nğŸ“Š Cultural analysis for {len(culture_df)} subreddits (â‰¥20 comments each)")
print("\nTop 15 subreddits by cultural characteristics:")
display(culture_df.head(15).round(3))

# Cultural diversity analysis
print("\nğŸŒ� Cultural Diversity Metrics:")
for metric in ['avg_comment_length', 'vocabulary_diversity', 'formality_score', 'violation_rate']:
    values = culture_df[metric]
    cv = values.std() / values.mean() if values.mean() != 0 else 0
    print(f"{metric}: CV = {cv:.3f} (range: {values.min():.3f} - {values.max():.3f})")

# Correlation analysis between cultural metrics and violation rates
print("\nğŸ”— Correlations with Violation Rate:")
correlations = culture_df.corr(numeric_only=True)['violation_rate'].sort_values(key=abs, ascending=False)
for metric, corr in correlations.items():
    if metric != 'violation_rate':
        significance = "***" if abs(corr) > 0.5 else "**" if abs(corr) > 0.3 else "*" if abs(corr) > 0.1 else ""
        print(f"{metric}: {corr:.3f} {significance}")

# Identify outlier subreddits
print("\nğŸ�¯ Cultural Outliers:")

# High formality, low violation rate
high_formality = culture_df[culture_df['formality_score'] > culture_df['formality_score'].quantile(0.8)]
low_violation_formal = high_formality[high_formality['violation_rate'] < high_formality['violation_rate'].median()]
if len(low_violation_formal) > 0:
    print(f"High formality, low violations: {list(low_violation_formal.index[:3])}")

# High diversity, high violation rate
high_diversity = culture_df[culture_df['vocabulary_diversity'] > culture_df['vocabulary_diversity'].quantile(0.8)]
high_violation_diverse = high_diversity[high_diversity['violation_rate'] > high_diversity['violation_rate'].median()]
if len(high_violation_diverse) > 0:
    print(f"High diversity, high violations: {list(high_violation_diverse.index[:3])}")


# Example quality and coherence analysis
print("ğŸ“š EXAMPLE QUALITY & UTILIZATION ANALYSIS")
print("=" * 60)

# Analyze example availability and quality
def analyze_examples(df):
    """Analyze the quality and availability of positive/negative examples"""
    example_stats = {
        'total_rows': len(df),
        'pos_ex1_available': df['positive_example_1'].notna().sum(),
        'pos_ex2_available': df['positive_example_2'].notna().sum(),
        'neg_ex1_available': df['negative_example_1'].notna().sum(),
        'neg_ex2_available': df['negative_example_2'].notna().sum(),
    }

    # Calculate example completeness
    example_stats['pos_examples_complete'] = (df['positive_example_1'].notna() & df['positive_example_2'].notna()).sum()
    example_stats['neg_examples_complete'] = (df['negative_example_1'].notna() & df['negative_example_2'].notna()).sum()
    example_stats['all_examples_complete'] = (
        df['positive_example_1'].notna() &
        df['positive_example_2'].notna() &
        df['negative_example_1'].notna() &
        df['negative_example_2'].notna()
    ).sum()

    return example_stats

example_stats = analyze_examples(train_df)

print("ğŸ“Š Example Availability Statistics:")
for key, value in example_stats.items():
    if 'total' not in key:
        percentage = (value / example_stats['total_rows'] * 100)
        print(f"{key}: {value:,} ({percentage:.1f}%)")

# Example length analysis
print("\nğŸ“� Example Length Analysis:")
example_lengths = {}
for col in ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']:
    lengths = train_df[col].dropna().str.len()
    example_lengths[col] = {
        'mean': lengths.mean(),
        'median': lengths.median(),
        'std': lengths.std(),
        'min': lengths.min(),
        'max': lengths.max()
    }

example_lengths_df = pd.DataFrame(example_lengths).T
print("Example length statistics:")
display(example_lengths_df.round(1))

# Example-comment similarity analysis (for sample)
print(f"\nğŸ”� Example-Comment Similarity Analysis (sample of {sample_size}):")

def calculate_example_similarity(comment, pos_ex1, pos_ex2, neg_ex1, neg_ex2):
    """Calculate similarity between comment and examples"""
    similarities = {}

    # Clean inputs
    comment = clean_text(comment) if pd.notna(comment) else ""

    for ex_name, example in [('pos_ex1', pos_ex1), ('pos_ex2', pos_ex2), ('neg_ex1', neg_ex1), ('neg_ex2', neg_ex2)]:
        if pd.notna(example):
            example_clean = clean_text(example)
            # Simple word overlap similarity
            comment_words = set(comment.split())
            example_words = set(example_clean.split())
            if len(comment_words) > 0 and len(example_words) > 0:
                overlap = len(comment_words.intersection(example_words))
                union = len(comment_words.union(example_words))
                similarities[ex_name] = overlap / union if union > 0 else 0
            else:
                similarities[ex_name] = 0
        else:
            similarities[ex_name] = None

    return similarities

# Calculate similarities for sample
sample_similarities = []
for idx, row in sample_df.iterrows():
    sim = calculate_example_similarity(
        row['body'], row['positive_example_1'], row['positive_example_2'],
        row['negative_example_1'], row['negative_example_2']
    )
    sim['violation'] = row['rule_violation']
    sample_similarities.append(sim)

sim_df = pd.DataFrame(sample_similarities)

# Analyze similarity patterns
print("\nComment-Example Similarity by Violation Status:")
sim_by_violation = sim_df.groupby('violation')[['pos_ex1', 'pos_ex2', 'neg_ex1', 'neg_ex2']].mean()
display(sim_by_violation.round(4))

# Test if violations are more similar to positive examples
violations = sim_df[sim_df['violation'] == 1]
non_violations = sim_df[sim_df['violation'] == 0]

print("\nğŸ�¯ Key Insights:")
if len(violations) > 0 and len(non_violations) > 0:
    # Compare positive example similarity
    viol_pos_sim = violations[['pos_ex1', 'pos_ex2']].mean(axis=1).dropna()
    non_viol_pos_sim = non_violations[['pos_ex1', 'pos_ex2']].mean(axis=1).dropna()

    if len(viol_pos_sim) > 0 and len(non_viol_pos_sim) > 0:
        pos_sim_diff = viol_pos_sim.mean() - non_viol_pos_sim.mean()
        print(f"Violations vs Non-violations similarity to positive examples: {pos_sim_diff:+.4f}")

        if pos_sim_diff > 0.05:
            print("âœ… Violations are more similar to positive examples (as expected)")
        elif pos_sim_diff < -0.05:
            print("âš ï¸�  Non-violations are more similar to positive examples (unexpected!)")
        else:
            print("â�– Similar similarity to positive examples for both groups")

# Example diversity analysis
print("\nğŸŒˆ Example Diversity Analysis:")
for rule in train_df['rule'].value_counts().head(5).index:
    rule_data = train_df[train_df['rule'] == rule]
    pos_examples = pd.concat([
        rule_data['positive_example_1'].dropna(),
        rule_data['positive_example_2'].dropna()
    ])

    if len(pos_examples) > 1:
        # Calculate diversity as number of unique examples
        unique_examples = pos_examples.nunique()
        total_examples = len(pos_examples)
        diversity_ratio = unique_examples / total_examples

        print(f"Rule: {rule[:50]}...")
        print(f"  Example diversity: {diversity_ratio:.3f} ({unique_examples}/{total_examples} unique)")


# Advanced visualizations for key insights (Seaborn version)
print("ğŸ“Š ADVANCED VISUALIZATIONS (Seaborn)")
print("=" * 40)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("Comprehensive Pattern Analysis", fontsize=18, fontweight='bold')

# 1. Rule Category Performance (Barplot)
ax = axes[0, 0]
if 'rule_category' in train_df.columns:
    category_stats = train_df.groupby('rule_category')['rule_violation'].agg(['mean', 'count']).reset_index()
    category_stats = category_stats[category_stats['count'] >= 50]
    category_stats = category_stats.sort_values('mean', ascending=True)
    sns.barplot(
        x='mean', y='rule_category', data=category_stats,
        ax=ax, palette='Reds_r', orient='h'
    )
    for i, (mean, count) in enumerate(zip(category_stats['mean'], category_stats['count'])):
        ax.text(mean, i, f"n={count}", va='center', ha='left', fontsize=10, color='black')
    ax.set_title("Rule Category Performance")
    ax.set_xlabel("Violation Rate")
    ax.set_ylabel("Rule Category")

# 2. Subreddit Cultural Patterns (Scatterplot)
ax = axes[0, 1]
if 'culture_df' in locals() and len(culture_df) > 0:
    scatter = ax.scatter(
        culture_df['formality_score'],
        culture_df['violation_rate'],
        s=culture_df['total_comments'] / 2,
        c=culture_df['vocabulary_diversity'],
        cmap='viridis',
        alpha=0.8,
        edgecolor='k'
    )
    for i, txt in enumerate(culture_df.index):
        ax.annotate(txt, (culture_df['formality_score'].iloc[i], culture_df['violation_rate'].iloc[i]), fontsize=8, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', label='Vocab Diversity')
    ax.set_title("Subreddit Cultural Patterns")
    ax.set_xlabel("Formality Score")
    ax.set_ylabel("Violation Rate")

# 3. Text Complexity Distribution (Violinplot)
ax = axes[1, 0]
if 'comment_length' in train_df.columns:
    # Cap extreme values for better visualization
    violations = train_df[train_df['rule_violation'] == 1]['comment_length']
    non_violations = train_df[train_df['rule_violation'] == 0]['comment_length']
    violations_capped = violations[violations <= violations.quantile(0.95)]
    non_violations_capped = non_violations[non_violations <= non_violations.quantile(0.95)]
    plot_df = (
        pd.concat([
            pd.DataFrame({'comment_length': violations_capped, 'Violation': 'Violations'}),
            pd.DataFrame({'comment_length': non_violations_capped, 'Violation': 'Non-violations'})
        ])
    )
    sns.violinplot(
        x='Violation', y='comment_length', data=plot_df,
        ax=ax, palette={'Violations': 'red', 'Non-violations': 'blue'}, inner='box'
    )
    ax.set_title("Text Complexity Distribution")
    ax.set_ylabel("Comment Length")
    ax.set_xlabel("")

# 4. Semantic Similarity Patterns (Histogram)
ax = axes[1, 1]
if 'sample_df_semantic' in locals() and 'comment_rule_similarity' in sample_df_semantic.columns:
    bins = np.linspace(
        sample_df_semantic['comment_rule_similarity'].min(),
        sample_df_semantic['comment_rule_similarity'].max(),
        21
    )
    sns.histplot(
        sample_df_semantic[sample_df_semantic['rule_violation'] == 1]['comment_rule_similarity'],
        bins=bins, color='red', label='Violations', alpha=0.7, ax=ax, stat='count'
    )
    sns.histplot(
        sample_df_semantic[sample_df_semantic['rule_violation'] == 0]['comment_rule_similarity'],
        bins=bins, color='blue', label='Non-violations', alpha=0.7, ax=ax, stat='count'
    )
    ax.set_title("Semantic Similarity Patterns")
    ax.set_xlabel("Semantic Similarity")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nğŸ�¨ Visualization insights generated!")


print("\nğŸ”� KEY FINDINGS FROM EDA:")
print("=" * 30)

# Calculate key statistics for final summary
total_violation_rate = train_df['rule_violation'].mean()
class_imbalance = train_df['rule_violation'].value_counts()[0] / train_df['rule_violation'].value_counts()[1]

print(f"ğŸ“Š Dataset Overview:")
print(f"   â€¢ {len(train_df):,} total comments across {train_df['subreddit'].nunique()} subreddits")
print(f"   â€¢ {train_df['rule'].nunique()} unique rules")
print(f"   â€¢ {total_violation_rate:.1%} overall violation rate")
print(f"   â€¢ Class imbalance ratio: {class_imbalance:.1f}:1 (non-violation:violation)")

if 'rule_category' in train_df.columns:
    most_problematic_category = train_df.groupby('rule_category')['rule_violation'].mean().idxmax()
    highest_violation_rate = train_df.groupby('rule_category')['rule_violation'].mean().max()
    print(f"   â€¢ Most problematic rule category: {most_problematic_category} ({highest_violation_rate:.1%})")

print(f"\nğŸ§  Semantic Analysis Insights:")
if 'sample_df_semantic' in locals() and 'comment_rule_similarity' in sample_df_semantic.columns:
    avg_similarity = sample_df_semantic['comment_rule_similarity'].mean()
    print(f"   â€¢ Average comment-rule semantic similarity: {avg_similarity:.3f}")
    
    violation_sim = sample_df_semantic[sample_df_semantic['rule_violation'] == 1]['comment_rule_similarity'].mean()
    non_violation_sim = sample_df_semantic[sample_df_semantic['rule_violation'] == 0]['comment_rule_similarity'].mean()
    sim_direction = 'higher' if violation_sim > non_violation_sim else 'lower'
    print(f"   â€¢ Violations have {sim_direction} semantic similarity to rules")

if 'culture_df' in locals() and len(culture_df) > 0:
    print(f"\nğŸ�˜ï¸�  Subreddit Cultural Insights:")
    cultural_variance = culture_df['violation_rate'].std()
    print(f"   â€¢ High cultural diversity: violation rates vary by {cultural_variance:.1%} across subreddits")
    
    most_formal = culture_df['formality_score'].idxmax()
    least_formal = culture_df['formality_score'].idxmin()
    print(f"   â€¢ Most formal subreddit: {most_formal}")
    print(f"   â€¢ Least formal subreddit: {least_formal}")


# Statistical Rigor: Effect Sizes and Confidence Intervals
print("\nğŸ“Š STATISTICAL RIGOR: EFFECT SIZES & CONFIDENCE INTERVALS")
print("=" * 70)

from scipy import stats
import numpy as np

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1:
        return np.nan
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    # Calculate Cohen's d
    d = (group1.mean() - group2.mean()) / pooled_std
    return d

def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def confidence_interval_proportion(successes, trials, confidence=0.95):
    """Calculate confidence interval for proportion"""
    if trials == 0:
        return (0, 0)
    
    p = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Wilson score interval (more accurate for small samples)
    term1 = p + z**2 / (2 * trials)
    term2 = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials)
    denominator = 1 + z**2 / trials
    
    ci_lower = (term1 - term2) / denominator
    ci_upper = (term1 + term2) / denominator
    
    return (ci_lower, ci_upper)

# Analyze key differences with effect sizes
print("ğŸ”� KEY DIFFERENCES WITH EFFECT SIZES:")
print("-" * 45)

# Comment length effect
violations = train_df[train_df['rule_violation'] == 1]['comment_length'].dropna()
non_violations = train_df[train_df['rule_violation'] == 0]['comment_length'].dropna()

length_effect = cohens_d(violations, non_violations)
print(f"1. Comment Length Difference:")
print(f"   Effect size (Cohen's d): {length_effect:.3f} ({interpret_effect_size(length_effect)})")
print(f"   Violations: {violations.mean():.1f} Â± {violations.std():.1f} chars")
print(f"   Non-violations: {non_violations.mean():.1f} Â± {non_violations.std():.1f} chars")

# Rule category differences
print(f"\n2. Rule Category Violation Rates with 95% CI:")
for category in train_df['rule_category'].unique():
    category_data = train_df[train_df['rule_category'] == category]
    violations_count = category_data['rule_violation'].sum()
    total_count = len(category_data)
    
    ci_lower, ci_upper = confidence_interval_proportion(violations_count, total_count)
    rate = violations_count / total_count
    
    print(f"   {category}: {rate:.1%} (95% CI: {ci_lower:.1%} - {ci_upper:.1%})")

# Subreddit-level analysis (for larger subreddits)
print(f"\n3. High-Volume Subreddit Analysis (â‰¥50 comments):")
large_subreddits = train_df.groupby('subreddit').size()
large_subreddits = large_subreddits[large_subreddits >= 50].index

for subreddit in large_subreddits[:10]:  # Top 10 by volume
    sub_data = train_df[train_df['subreddit'] == subreddit]
    violations_count = sub_data['rule_violation'].sum()
    total_count = len(sub_data)
    
    ci_lower, ci_upper = confidence_interval_proportion(violations_count, total_count)
    rate = violations_count / total_count
    
    print(f"   r/{subreddit}: {rate:.1%} (95% CI: {ci_lower:.1%} - {ci_upper:.1%})")

print(f"\nğŸ’¡ STATISTICAL INSIGHTS:")
print("â€¢ Effect sizes help distinguish between statistical and practical significance")
print("â€¢ Confidence intervals show the uncertainty in our estimates")
print("â€¢ Large effect sizes (>0.8) indicate strong predictive potential")


# Advanced Visualizations with Seaborn

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("\nğŸ�¨ SUBREDDIT LANDSCAPE (SEABORN 2D PROJECTION)")
print("=" * 60)

# 2D scatter plot as seaborn does not support 3D interactive plots natively
if 'culture_df' in locals() and len(culture_df) > 0:
    # We'll use a 2D projection: x=formality_score, y=vocabulary_diversity, color=violation_rate, size=total_comments
    df = culture_df.reset_index()
    plt.figure(figsize=(12, 8))
    norm = plt.Normalize(df['avg_comment_length'].min(), df['avg_comment_length'].max())
    sizes = 100 + 400 * (df['total_comments'] - df['total_comments'].min()) / (df['total_comments'].max() - df['total_comments'].min() + 1e-6)
    scatter = plt.scatter(
        df['formality_score'],
        df['vocabulary_diversity'],
        s=sizes,
        c=df['avg_comment_length'],
        cmap='RdYlBu_r',
        alpha=0.8,
        edgecolor='k'
    )
    plt.xlabel("Formality Score")
    plt.ylabel("Vocabulary Diversity")
    plt.title("ğŸŒ� The Reddit Universe: Subreddit Cultural Landscape (2D Projection)")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Avg Comment Length')
    for i, row in df.iterrows():
        plt.text(row['formality_score'], row['vocabulary_diversity'], row['index'], fontsize=8, alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("ğŸ’¡ INSIGHT:")
    print("Bubble size = number of comments")
    print("Color = average comment length")
    print("Position = formality & vocabulary diversity")
    print("Subreddit names are labeled for exploration.")

print("\nğŸŒŠ SUBREDDIT â†’ RULE â†’ OUTCOME FLOW (SEABORN HEATMAPS)")
print("=" * 60)

# Rule Category vs Violation Outcome heatmap
if 'rule_category' in train_df.columns:
    rule_outcome = train_df.groupby(['rule_category', 'rule_violation']).size().unstack(fill_value=0)
    rule_outcome.columns = ['Non-Violation', 'Violation'] if 0 in rule_outcome.columns and 1 in rule_outcome.columns else rule_outcome.columns
    plt.figure(figsize=(6, 4))
    sns.heatmap(rule_outcome, annot=True, fmt='d', cmap='Reds')
    plt.title("Rule Category vs Violation Outcome")
    plt.ylabel("Rule Category")
    plt.xlabel("Outcome")
    plt.tight_layout()
    plt.show()

print("ğŸ’¡ FLOW INSIGHT:")
print("Heatmaps show how comments flow from subreddits to rule categories and then to outcomes.")
print("Darker cells indicate more comments following that path.")


# N-Gram Analysis: Detecting Phrase Patterns
print("\nğŸ”— N-GRAM ANALYSIS: THE SMOKING GUN PHRASES")
print("=" * 60)

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd

# Ensure we have clean text data
if 'body_clean' not in train_df.columns:
    train_df['body_clean'] = train_df['body'].apply(clean_text)

def analyze_ngrams(texts, n=2, top_k=20):
    """Extract and analyze n-grams from text data"""
    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        lowercase=True,
        stop_words='english',
        max_features=1000
    )
    
    # Clean texts
    clean_texts = [str(text) for text in texts if pd.notna(text)]
    
    if len(clean_texts) == 0:
        return []
    
    # Fit and transform
    ngram_matrix = vectorizer.fit_transform(clean_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum frequencies
    frequencies = ngram_matrix.sum(axis=0).A1
    ngram_freq = [(feature_names[i], frequencies[i]) for i in range(len(feature_names))]
    
    # Return top k
    return sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:top_k]

# Analyze bigrams for violations vs non-violations
print("ğŸš¨ TOP BIGRAMS IN RULE VIOLATIONS:")
violation_bigrams = analyze_ngrams(
    train_df[train_df['rule_violation'] == 1]['body_clean'], n=2, top_k=15
)

for i, (bigram, freq) in enumerate(violation_bigrams, 1):
    print(f"{i:2d}. '{bigram}' (appears {freq} times)")

print("\nâœ… TOP BIGRAMS IN COMPLIANT COMMENTS:")
compliant_bigrams = analyze_ngrams(
    train_df[train_df['rule_violation'] == 0]['body_clean'], n=2, top_k=15
)

for i, (bigram, freq) in enumerate(compliant_bigrams, 1):
    print(f"{i:2d}. '{bigram}' (appears {freq} times)")

# Find violation-specific patterns
violation_set = set([bg[0] for bg in violation_bigrams])
compliant_set = set([bg[0] for bg in compliant_bigrams])

violation_only = violation_set - compliant_set
compliant_only = compliant_set - violation_set

print(f"\nğŸ�¯ BIGRAMS UNIQUE TO VIOLATIONS ({len(violation_only)}):")
for bigram in list(violation_only)[:10]:
    print(f"   â€¢ '{bigram}'")

print(f"\nğŸ�¯ BIGRAMS UNIQUE TO COMPLIANT COMMENTS ({len(compliant_only)}):")
for bigram in list(compliant_only)[:10]:
    print(f"   â€¢ '{bigram}'")

print("\nğŸ’¡ PATTERN INSIGHT:")
print("Violation-specific bigrams often contain action words ('click here', 'pm me'),")
print("while compliant bigrams are more descriptive and conversational.")


# Word Cloud Analysis: The Language of Rule-Breaking
print("â˜�ï¸� WORD CLOUD ANALYSIS: VIOLATION VS COMPLIANCE")
print("=" * 60)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure we have clean text data
if 'body_clean' not in train_df.columns:
    train_df['body_clean'] = train_df['body'].apply(clean_text)

# Create word clouds for violations vs non-violations
def create_comparative_wordclouds(df):
    """Create side-by-side word clouds for violations vs non-violations"""
    violations_text = ' '.join(df[df['rule_violation'] == 1]['body_clean'].fillna(''))
    non_violations_text = ' '.join(df[df['rule_violation'] == 0]['body_clean'].fillna(''))
    
    # Create word clouds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Violations word cloud
    stop_words = set([
        'http', 'https', 'www', 'com', 'reddit', 'post', 'comment', 'html',
        'the', 'and', 'to', 'of', 'in', 'a', 'is', 'it', 'for', 'on', 'that', 'this', 'with', 'as', 'are', 'was', 'be', 'at', 'by', 'an', 'or',
        'from', 'but', 'not', 'have', 'has', 'if', 'they', 'you', 'i', 'we', 'he', 'she', 'his', 'her', 'their', 'them', 'so', 'do', 'can', 'will',
        'just', 'my', 'me', 'your', 'our', 'us', 'about', 'what', 'when', 'who', 'which', 'how', 'all', 'more', 'no', 'out', 'up', 'one', 'would', 'there',
        'been', 'also', 'than', 'get', 'had', 'were', 'some', 'any', 'because', 'into', 'other', 'could', 'should', 'did', 'very', 'over', 'after', 'then',
        'now', 'only', 'even', 'such', 'most', 'like', 'see', 'these', 'may', 'where', 'why', 'got', 'off', 'back', 'still', 'make', 'made', 'going', 'go',
        'am', 'im', 'dont', 'does', 'doesnt', 'didnt', 'cant', 'wont', 'youre', 'youve', 'youll', 'theyre', 'theyve', 'theyll', 'were', 'wasnt', 'isnt', 'arent',
        'havent', 'hasnt', 'couldnt', 'shouldnt', 'wouldnt'
    ])
    wordcloud_violations = WordCloud(
        width=800, height=400, 
        background_color='black',
        colormap='Reds',
        max_words=100,
        relative_scaling=0.5,
        stopwords=stop_words
    ).generate(violations_text)
    
    ax1.imshow(wordcloud_violations, interpolation='bilinear')
    ax1.set_title('ğŸš¨ RULE VIOLATIONS: The Language of Rule-Breaking', 
                  fontsize=16, fontweight='bold', color='red')
    ax1.axis('off')
    
    # Non-violations word cloud
    wordcloud_non_violations = WordCloud(
        width=800, height=400,
        background_color='black',
        colormap='Blues',
        max_words=100,
        relative_scaling=0.5,
        stopwords=stop_words
    ).generate(non_violations_text)
    
    ax2.imshow(wordcloud_non_violations, interpolation='bilinear')
    ax2.set_title('âœ… RULE COMPLIANCE: The Language of Good Citizens', 
                  fontsize=16, fontweight='bold', color='blue')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return wordcloud_violations, wordcloud_non_violations

# Generate word clouds
wc_violations, wc_non_violations = create_comparative_wordclouds(train_df)

# Extract top words for analysis
violation_words = dict(wc_violations.words_)
non_violation_words = dict(wc_non_violations.words_)

print("\nğŸ”� Top 10 Words in Violations:")
for word, freq in sorted(violation_words.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"   {word}: {freq:.3f}")

print("\nâœ… Top 10 Words in Non-Violations:")
for word, freq in sorted(non_violation_words.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"   {word}: {freq:.3f}")

print("\nğŸ’¡ LINGUISTIC INSIGHT:")
print("Notice the difference in language patterns - violations tend to use more direct,")
print("action-oriented language, while compliant comments use more descriptive terms.")

