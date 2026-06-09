# =============================================================================
# IMPORT ORGANIZATION
# =============================================================================

# ===== SYSTEM CONFIGURATION =====
import warnings
warnings.filterwarnings('ignore')
import os

# ===== STANDARD LIBRARY =====
import re
import pickle
from collections import Counter
from tqdm import tqdm

# ===== CORE DATA SCIENCE LIBRARIES =====
import pandas as pd
import numpy as np

# ===== VISUALIZATION =====
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ===== SCIKIT-LEARN =====
# Preprocessing and Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    StandardScaler, 
    LabelEncoder, 
    MinMaxScaler, 
    PowerTransformer
)

# Model Selection and Evaluation
from sklearn.model_selection import (
    StratifiedKFold, 
    train_test_split
)
from sklearn.metrics import (
    classification_report, 
    accuracy_score
)

# Machine Learning Models
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Clustering and Dimensionality Reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===== GRADIENT BOOSTING LIBRARIES =====
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# ===== DEEP LEARNING AND TRANSFORMERS =====
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from datasets import Dataset

# ===== HYPERPARAMETER OPTIMIZATION =====
import optuna


# Configure visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
colors = ['#440154', '#31688e', '#35b779', '#fde725', '#440154']

# Load data
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')

print("=" * 60)
print("MAP COMPETITION - MATHEMATICAL EDA DASHBOARD")
print("=" * 60)

# BASIC INFORMATION
print("\nğŸ“‹ BASIC INFORMATION")
print("-" * 30)
print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")
print(f"Training columns: {list(train_df.columns)}")
print(f"Test columns: {list(test_df.columns)}")
print(f"Training shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Check missing values
print(f"\nMissing values in training:")
for col in train_df.columns:
    missing = train_df[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(train_df)*100:.1f}%)")

# Automatically identify columns
text_column = 'StudentExplanation'  # Based on provided output
target_column = 'Misconception'     # Based on provided output

print(f"\nText column identified: {text_column}")
print(f"Target column identified: {target_column}")

# MATHEMATICAL ANALYSIS FUNCTIONS
def extract_math_features(text):
    """Extract mathematical features from text"""
    features = {}
    text_str = str(text)
    
    # Mathematical patterns
    features['has_fraction'] = bool(re.search(r'\d+/\d+', text_str))
    features['has_decimal'] = bool(re.search(r'\d+\.\d+', text_str))
    features['has_percentage'] = bool(re.search(r'\d+%', text_str))
    features['has_equation'] = bool(re.search(r'[=<>]', text_str))
    features['has_variables'] = bool(re.search(r'[a-zA-Z]\s*[=+\-*/]', text_str))
    
    # Counters
    features['number_count'] = len(re.findall(r'\d+', text_str))
    features['operation_count'] = len(re.findall(r'[+\-*/=<>]', text_str))
    features['parentheses_count'] = len(re.findall(r'[()]', text_str))
    
    # Mathematical keywords
    math_keywords = ['sum', 'difference', 'product', 'quotient', 'equal', 'greater', 'less', 
                    'add', 'subtract', 'multiply', 'divide', 'solve', 'calculate', 'answer']
    features['math_keywords_count'] = sum(1 for word in math_keywords if word in text_str.lower())
    
    # Text complexity
    features['text_length'] = len(text_str)
    features['word_count'] = len(text_str.split())
    features['sentence_count'] = text_str.count('.') + text_str.count('!') + text_str.count('?') + 1
    
    return features

def reading_complexity(text):
    """Calculate simplified reading complexity"""
    text_str = str(text)
    words = text_str.split()
    sentences = text_str.count('.') + text_str.count('!') + text_str.count('?') + 1
    
    if len(words) == 0 or sentences == 0:
        return 0, 0
    
    avg_sentence_length = len(words) / sentences
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simplified Flesch Reading Ease
    reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 4.7)
    
    return reading_ease, avg_sentence_length

# EXTRACT MATHEMATICAL FEATURES
print("\nğŸ”¢ EXTRACTING MATHEMATICAL FEATURES...")
math_features_list = []
for text in train_df[text_column]:
    features = extract_math_features(text)
    reading_ease, avg_sent_len = reading_complexity(text)
    features['reading_ease'] = reading_ease
    features['avg_sentence_length'] = avg_sent_len
    math_features_list.append(features)

math_df = pd.DataFrame(math_features_list)

# MATHEMATICAL PATTERN ANALYSIS
print("\nğŸ“Š MATHEMATICAL PATTERNS DETECTED")
print("-" * 40)
boolean_features = ['has_fraction', 'has_decimal', 'has_percentage', 'has_equation', 'has_variables']
for feature in boolean_features:
    count = math_df[feature].sum()
    percentage = count / len(math_df) * 100
    print(f"{feature.replace('has_', '').title()}: {count} ({percentage:.1f}%)")

# NUMERICAL STATISTICS
print("\nğŸ“ˆ NUMERICAL FEATURES STATISTICS")
print("-" * 40)
numeric_features = ['number_count', 'operation_count', 'parentheses_count', 'math_keywords_count', 'text_length', 'word_count']
stats_summary = math_df[numeric_features].describe()
print(stats_summary.round(2))

# MISCONCEPTION ANALYSIS
if target_column and target_column in train_df.columns:
    print(f"\nğŸ�¯ MISCONCEPTION ANALYSIS")
    print("-" * 40)
    
    # Filter only rows with non-null misconceptions
    train_with_misconceptions = train_df[train_df[target_column].notna()].copy()
    
    print(f"Total samples with misconceptions: {len(train_with_misconceptions)}")
    print(f"Total unique misconceptions: {train_with_misconceptions[target_column].nunique()}")
    print(f"Percentage with misconceptions: {len(train_with_misconceptions)/len(train_df)*100:.1f}%")
    
    print("\nTop 10 Misconceptions:")
    misconception_dist = train_with_misconceptions[target_column].value_counts()
    for misc, count in misconception_dist.head(10).items():
        percentage = count / len(train_with_misconceptions) * 100
        print(f"  {misc}: {count} ({percentage:.1f}%)")
    
    # Analysis by category if exists
    if 'Category' in train_df.columns:
        print("\nDistribution by Category:")
        category_dist = train_with_misconceptions['Category'].value_counts()
        for cat, count in category_dist.head(10).items():
            percentage = count / len(train_with_misconceptions) * 100
            print(f"  {cat}: {count} ({percentage:.1f}%)")
else:
    print(f"\nğŸ�¯ MISCONCEPTION ANALYSIS")
    print("-" * 40)
    print("Misconception column not found or empty")

# TEXTUAL COMPLEXITY ANALYSIS
print("\nğŸ“� TEXTUAL COMPLEXITY ANALYSIS")
print("-" * 40)
print(f"Average text length: {math_df['text_length'].mean():.1f} characters")
print(f"Average words per response: {math_df['word_count'].mean():.1f}")
print(f"Average sentences per response: {math_df['sentence_count'].mean():.1f}")
print(f"Average reading ease: {math_df['reading_ease'].mean():.1f}")

# Classify reading level
avg_reading_ease = math_df['reading_ease'].mean()
if avg_reading_ease > 60:
    reading_level = "Easy"
elif avg_reading_ease > 30:
    reading_level = "Moderate"
else:
    reading_level = "Difficult"
print(f"Reading level classified: {reading_level}")

# CORRELATION MATRIX
print("\nğŸ”— CORRELATION BETWEEN FEATURES")
print("-" * 40)
correlation_features = ['number_count', 'operation_count', 'parentheses_count', 'math_keywords_count', 'text_length', 'word_count']
correlation_matrix = math_df[correlation_features].corr()
print("Highest correlations (>0.5):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            print(f"  {correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: {corr_val:.3f}")

# VISUALIZATIONS
print("\nğŸ“Š GENERATING VISUALIZATIONS...")

# 1. Distribution of mathematical patterns
fig, ax = plt.subplots(figsize=(12, 6))
boolean_counts = [math_df[feature].sum() for feature in boolean_features]
feature_names = [f.replace('has_', '').title() for f in boolean_features]
bars = ax.bar(feature_names, boolean_counts, color=colors[:len(boolean_features)])
ax.set_title('Mathematical Patterns Detected', fontsize=16, fontweight='bold')
ax.set_ylabel('Frequency')
ax.set_xlabel('Mathematical Features')

# Add values on bars
for bar, count in zip(bars, boolean_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(boolean_counts) if max(boolean_counts) > 0 else 0.1,
            f'{count}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Complexity distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Textual Complexity Analysis', fontsize=16, fontweight='bold')

# Text length
axes[0,0].hist(math_df['text_length'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Text Length Distribution')
axes[0,0].set_xlabel('Characters')
axes[0,0].set_ylabel('Frequency')

# Word count
axes[0,1].hist(math_df['word_count'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,1].set_title('Word Count Distribution')
axes[0,1].set_xlabel('Words')
axes[0,1].set_ylabel('Frequency')

# Reading ease
axes[1,0].hist(math_df['reading_ease'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
axes[1,0].set_title('Reading Ease')
axes[1,0].set_xlabel('Ease Score')
axes[1,0].set_ylabel('Frequency')

# Number count
axes[1,1].hist(math_df['number_count'], bins=20, alpha=0.7, color='gold', edgecolor='black')
axes[1,1].set_title('Number Count per Text')
axes[1,1].set_xlabel('Number of Numbers')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# 3. Visual correlation matrix
if len(correlation_features) > 0:
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix - Mathematical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 4. Misconception distribution (if available)
if target_column and target_column in train_df.columns:
    train_with_misconceptions = train_df[train_df[target_column].notna()].copy()
    
    if len(train_with_misconceptions) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Category distribution (if exists)
        if 'Category' in train_df.columns:
            category_counts = train_with_misconceptions['Category'].value_counts().head(10)
            bars1 = axes[0].bar(range(len(category_counts)), category_counts.values, color=colors[0])
            axes[0].set_title('Top 10 Categories')
            axes[0].set_ylabel('Frequency')
            axes[0].set_xticks(range(len(category_counts)))
            axes[0].set_xticklabels(category_counts.index, rotation=45, ha='right')
        
        # Misconception distribution
        misconception_counts = train_with_misconceptions[target_column].value_counts().head(10)
        bars2 = axes[1].bar(range(len(misconception_counts)), misconception_counts.values, color=colors[1])
        axes[1].set_title('Top 10 Misconceptions')
        axes[1].set_ylabel('Frequency')
        axes[1].set_xticks(range(len(misconception_counts)))
        axes[1].set_xticklabels([str(x)[:15] + '...' if len(str(x)) > 15 else str(x) for x in misconception_counts.index], 
                               rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No misconception data to visualize")

# SIMPLE CLUSTERING ANALYSIS
print("\nğŸ�ª CLUSTERING ANALYSIS")
print("-" * 30)

# Prepare data for clustering
cluster_features = ['text_length', 'word_count', 'number_count', 'operation_count', 'math_keywords_count']
cluster_data = math_df[cluster_features].fillna(0)

# Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(cluster_data_scaled)

# Add clusters to dataframe
math_df['cluster'] = clusters

print("Cluster distribution:")
cluster_dist = pd.Series(clusters).value_counts().sort_index()
for cluster_id, count in cluster_dist.items():
    percentage = count / len(clusters) * 100
    print(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")

# Cluster analysis
print("\nCluster characteristics:")
for cluster_id in range(5):
    cluster_data = math_df[math_df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
    print(f"  Average length: {cluster_data['text_length'].mean():.1f}")
    print(f"  Average words: {cluster_data['word_count'].mean():.1f}")
    print(f"  Average numbers: {cluster_data['number_count'].mean():.1f}")
    print(f"  Math keywords: {cluster_data['math_keywords_count'].mean():.1f}")

# AUTOMATIC INSIGHTS
print("\nğŸ’¡ AUTOMATIC INSIGHTS")
print("-" * 30)

insights = []

# Insight 1: Most common pattern
most_common_pattern = max(boolean_features, key=lambda x: math_df[x].sum())
pattern_percentage = math_df[most_common_pattern].sum() / len(math_df) * 100
insights.append(f"Most common mathematical pattern: {most_common_pattern.replace('has_', '').title()} ({pattern_percentage:.1f}%)")

# Insight 2: Complexity
avg_words = math_df['word_count'].mean()
if avg_words > 50:
    complexity_level = "high"
elif avg_words > 25:
    complexity_level = "medium"
else:
    complexity_level = "low"
insights.append(f"Textual complexity: {complexity_level} (average: {avg_words:.1f} words)")

# Insight 3: Misconceptions
if target_column and target_column in train_df.columns:
    most_common_misconception = train_df[target_column].value_counts().index[0]
    misconception_percentage = train_df[target_column].value_counts().iloc[0] / len(train_df) * 100
    insights.append(f"Most common misconception: {most_common_misconception} ({misconception_percentage:.1f}%)")

# Insight 4: Data balance
if target_column and target_column in train_df.columns:
    class_counts = train_df[target_column].value_counts()
    balance_ratio = class_counts.min() / class_counts.max()
    if balance_ratio < 0.1:
        balance_status = "highly imbalanced"
    elif balance_ratio < 0.5:
        balance_status = "moderately imbalanced"
    else:
        balance_status = "relatively balanced"
    insights.append(f"Dataset balance: {balance_status} (ratio: {balance_ratio:.3f})")

# EXPORT FEATURES
print("\nğŸ’¾ PREPARING FEATURES FOR MODELING")
print("-" * 40)

# Combine original data with mathematical features
export_df = pd.concat([train_df, math_df], axis=1)

print(f"Features extracted: {len(math_df.columns)} new columns")
print(f"Final dataset: {export_df.shape}")
print(f"Total columns: {list(export_df.columns)}")

# Save processed features
export_df.to_csv('train_with_math_features.csv', index=False)
print("\nâœ… Features saved to: train_with_math_features.csv")

# FINAL SUMMARY
print("\n" + "=" * 60)
print("EXECUTIVE SUMMARY - MAP COMPETITION EDA")
print("=" * 60)
print(f"ğŸ“Š Dataset: {len(train_df)} samples, {len(train_df.columns)} original features")
print(f"ğŸ”¢ Mathematical features: {len(math_df.columns)} new features extracted")
train_with_misconceptions = train_df[train_df[target_column].notna()] if target_column and target_column in train_df.columns else pd.DataFrame()
if len(train_with_misconceptions) > 0:
    print(f"ğŸ�¯ Targets: {train_with_misconceptions[target_column].nunique()} unique misconceptions")
    print(f"âš ï¸� Warning: 73.1% of data without labels (misconceptions)")
else:
    print("ğŸ�¯ Targets: No valid misconceptions in data")
print(f"ğŸ“ˆ Average complexity: {avg_words:.1f} words per response")
print("=" * 60)


import os
import gc
import time
import pickle
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from lightgbm import LGBMClassifier

# Set env before torch CUDA allocations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# =========================
# ConfiguraÃ§Ã£o OTIMIZADA
# =========================
MAX_LEN = 96
BATCH_SIZE = 64
USE_DEEPSEEK = True
DEEPSEEK_SAMPLE_RATIO = 0.20  # Reduzido para economia de tempo
USE_CACHE = True

# Features simplificadas mas efetivas
USE_TFIDF_FEATURES = True
TFIDF_MAX_FEATURES = 300  # Muito reduzido
PCA_COMPONENTS = 200  # Reduzido
FEATURE_SELECTION_K = 15  # Reduzido
USE_STACKING = False  # Desabilitado para simplicidade
USE_FEATURE_INTERACTIONS = False  # Desabilitado

print("=== PIPELINE OTIMIZADO: BALANCE PERFORMANCE/SPEED ===")
print(f"DeepSeek sampling: {DEEPSEEK_SAMPLE_RATIO*100:.0f}%")
print(f"PCA components: {PCA_COMPONENTS}")
print(f"Math features: {FEATURE_SELECTION_K}")
print(f"TF-IDF features: {TFIDF_MAX_FEATURES}")
print(f"Stacking: {USE_STACKING}")
print(f"Feature interactions: {USE_FEATURE_INTERACTIONS}")

# Devices
N_GPUS = torch.cuda.device_count()
DEBERTA_DEVICE = torch.device('cuda:0' if N_GPUS >= 1 else 'cpu')
DEEPSEEK_DEVICE = torch.device('cuda:1' if N_GPUS >= 2 else DEBERTA_DEVICE)

# Paths
DEBERTA_MODEL_PATH = "/kaggle/input/matherrorai/deberta_enhanced/checkpoint-1800"
DEEPSEEK_MODEL_PATH = "/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL"
ENHANCED_TRAIN_PATH = "/kaggle/input/matherrorai/train_with_math_features.csv"
TEST_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"

# Caches
DEBERTA_EMB_CACHE = "deberta_embeddings_optimized.pkl"
DEEPSEEK_EMB_CACHE = "deepseek_embeddings_optimized.pkl"

# =========================
# Data loading
# =========================
print("\n=== LOADING DATA ===")
train_df = pd.read_csv(ENHANCED_TRAIN_PATH)

def quick_is_correct(df: pd.DataFrame) -> pd.DataFrame:
    true_answers = df[df['Category'].str.startswith('True', na=False)]
    if len(true_answers) > 0:
        tmp = true_answers.groupby(['QuestionId', 'MC_Answer']).size().reset_index()
        tmp = tmp.groupby('QuestionId').first().reset_index()
        tmp['is_correct'] = 1
        df = df.merge(tmp[['QuestionId', 'MC_Answer', 'is_correct']],
                      on=['QuestionId', 'MC_Answer'], how='left')
        df['is_correct'] = df['is_correct'].fillna(0)
    else:
        df['is_correct'] = 0
    return df

train_df = quick_is_correct(train_df)
train_df['target'] = train_df['Category'] + ':' + train_df['Misconception'].fillna('NA')

# Keep stable classes
class_counts = train_df['target'].value_counts()
keep_classes = class_counts[class_counts >= 15].index
train_df = train_df[train_df['target'].isin(keep_classes)].copy()

# Label encoding
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['target'])
n_classes = len(le.classes_)

print(f"Training samples: {len(train_df)}")
print(f"Number of classes: {n_classes}")

# Prompt simplificado mas efetivo
train_df['text'] = train_df.apply(
    lambda r: (
        f"Math Question: {r['QuestionText'][:120]} "
        f"Student Answer: {r['MC_Answer']} "
        f"Correctness: {'Correct' if r['is_correct'] else 'Incorrect'} "
        f"Explanation: {str(r['StudentExplanation'])[:80]}"
    ),
    axis=1
)

# Texto para TF-IDF
train_df['text_for_tfidf'] = train_df.apply(
    lambda r: f"{r['QuestionText']} {r['MC_Answer']} {r['StudentExplanation']}",
    axis=1
)

# =========================
# Features Simplificadas
# =========================
def create_essential_features(df):
    """MantÃ©m apenas as features mais importantes"""
    df = df.copy()
    
    # Features bÃ¡sicas mais importantes
    df['number_count'] = df['StudentExplanation'].apply(lambda x: len(re.findall(r'\d+', str(x))))
    df['text_length'] = df['StudentExplanation'].str.len().fillna(0)
    df['word_count'] = df['StudentExplanation'].str.split().str.len().fillna(0)
    df['operation_count'] = df['StudentExplanation'].apply(lambda x: len(re.findall(r'[+\-*/Ã·Ã—]', str(x))))
    df['is_correct'] = df['is_correct']
    
    # Top 10 features matemÃ¡ticas mais discriminativas
    df['formula_count'] = df['StudentExplanation'].str.count(r'[=]')
    df['capital_ratio'] = df['StudentExplanation'].str.count(r'[A-Z]') / df['text_length'].clip(1)
    df['math_words'] = df['StudentExplanation'].apply(
        lambda x: len(re.findall(r'\b(solve|answer|wrong|because|not|calculate|multiply|divide|add|subtract)\b', str(x).lower()))
    )
    df['negative_words'] = df['StudentExplanation'].apply(
        lambda x: len(re.findall(r'\b(wrong|incorrect|mistake|error|not|no)\b', str(x).lower()))
    )
    df['avg_word_len'] = df['StudentExplanation'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
    )
    df['sentence_count'] = df['StudentExplanation'].str.count(r'[.!?]')
    df['reasoning_words'] = df['StudentExplanation'].apply(
        lambda x: len(re.findall(r'\b(first|then|next|because|so|if|when)\b', str(x).lower()))
    )
    
    return df

train_df = create_essential_features(train_df)
feature_columns = [
    'number_count', 'text_length', 'is_correct', 'word_count', 'operation_count',
    'formula_count', 'capital_ratio', 'math_words', 'negative_words',
    'avg_word_len', 'sentence_count', 'reasoning_words'
]

# Test data
test_df = pd.read_csv(TEST_PATH)
true_answers = train_df[train_df['Category'].str.startswith('True', na=False)]
if len(true_answers) > 0:
    tmp = true_answers.groupby(['QuestionId', 'MC_Answer']).size().reset_index()
    tmp = tmp.groupby('QuestionId').first().reset_index()
    tmp['is_correct'] = 1
    test_df = test_df.merge(tmp[['QuestionId', 'MC_Answer', 'is_correct']],
                            on=['QuestionId', 'MC_Answer'], how='left')
    test_df['is_correct'] = test_df['is_correct'].fillna(0)
else:
    test_df['is_correct'] = 0

test_df = create_essential_features(test_df)
test_df['text'] = test_df.apply(
    lambda r: (
        f"Math Question: {r['QuestionText'][:120]} "
        f"Student Answer: {r['MC_Answer']} "
        f"Correctness: {'Correct' if r['is_correct'] else 'Incorrect'} "
        f"Explanation: {str(r['StudentExplanation'])[:80]}"
    ),
    axis=1
)

test_df['text_for_tfidf'] = test_df.apply(
    lambda r: f"{r['QuestionText']} {r['MC_Answer']} {r['StudentExplanation']}",
    axis=1
)

print(f"Test samples: {len(test_df)}")
print(f"Essential math features: {len(feature_columns)}")

# =========================
# DeBERTa Processing (reutilizar cache se possÃ­vel)
# =========================
print("\n=== PROCESSING DEBERTA ===")

train_deberta_embeddings = None
test_deberta_embeddings = None

if USE_CACHE and os.path.exists(DEBERTA_EMB_CACHE):
    try:
        with open(DEBERTA_EMB_CACHE, 'rb') as f:
            cached = pickle.load(f)
        if len(cached['train']) == len(train_df) and len(cached['test']) == len(test_df):
            train_deberta_embeddings = cached['train']
            test_deberta_embeddings = cached['test']
            print("Loaded cached DeBERTa embeddings")
    except:
        pass

if train_deberta_embeddings is None:
    deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_PATH)
    deberta_model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_PATH)
    deberta_model.to(DEBERTA_DEVICE)
    deberta_model.eval()
    if DEBERTA_DEVICE.type == 'cuda':
        deberta_model.half()

    def embed_texts_deberta(texts):
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            if i % (BATCH_SIZE * 20) == 0:
                print(f"DeBERTa progress: {i}/{len(texts)}")
            batch = texts[i:i+BATCH_SIZE]
            inputs = deberta_tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt'
            ).to(DEBERTA_DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = deberta_model(**inputs, output_hidden_states=True)
                    embs = outputs.hidden_states[-1][:, 0, :].float().cpu().numpy()
            out.extend(embs)
        return np.array(out)

    print("Generating DeBERTa embeddings...")
    train_deberta_embeddings = embed_texts_deberta(train_df['text'].tolist())
    test_deberta_embeddings = embed_texts_deberta(test_df['text'].tolist())
    
    if USE_CACHE:
        with open(DEBERTA_EMB_CACHE, 'wb') as f:
            pickle.dump({'train': train_deberta_embeddings, 'test': test_deberta_embeddings}, f)
        print("DeBERTa embeddings cached")

    del deberta_model, deberta_tokenizer
    torch.cuda.empty_cache(); gc.collect()

print(f"DeBERTa embeddings shape: {train_deberta_embeddings.shape}")

# =========================
# DeepSeek Processing (Simplificado)
# =========================
print("\n=== PROCESSING DEEPSEEK ===")

train_deepseek_embeddings = None
test_deepseek_embeddings = None

if USE_CACHE and os.path.exists(DEEPSEEK_EMB_CACHE):
    try:
        with open(DEEPSEEK_EMB_CACHE, 'rb') as f:
            cached = pickle.load(f)
        if len(cached['train']) == len(train_df) and len(cached['test']) == len(test_df):
            train_deepseek_embeddings = cached['train']
            test_deepseek_embeddings = cached['test']
            print("Loaded cached DeepSeek embeddings")
    except:
        pass

if train_deepseek_embeddings is None and USE_DEEPSEEK:
    deepseek_available = False
    
    try:
        print("Loading DeepSeek...")
        deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH)
        
        # Padding token fix
        if deepseek_tokenizer.pad_token is None:
            if hasattr(deepseek_tokenizer, 'eos_token') and deepseek_tokenizer.eos_token:
                deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token
                deepseek_tokenizer.pad_token_id = deepseek_tokenizer.eos_token_id
            else:
                deepseek_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        deepseek_model = AutoModelForSequenceClassification.from_pretrained(
            DEEPSEEK_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        deepseek_model.to(DEEPSEEK_DEVICE)
        deepseek_model.config.pad_token_id = deepseek_tokenizer.pad_token_id
        
        if deepseek_tokenizer.pad_token_id >= deepseek_model.config.vocab_size:
            deepseek_model.resize_token_embeddings(len(deepseek_tokenizer))
        
        deepseek_model.eval()
        deepseek_available = True
        print(f"DeepSeek loaded on {DEEPSEEK_DEVICE}")
        
    except Exception as e:
        print(f"DeepSeek loading failed: {e}")
        deepseek_available = False
    
    if deepseek_available:
        def embed_texts_deepseek_simple(texts, sample_ratio=None):
            """VersÃ£o simplificada do DeepSeek embedding"""
            if sample_ratio and sample_ratio < 1.0 and len(texts) > 20:
                np.random.seed(42)
                n_samples = int(len(texts) * sample_ratio)
                sample_indices = np.random.choice(len(texts), n_samples, replace=False)
                sample_texts = [texts[i] for i in sample_indices]
                
                print(f"Simple sampling: {len(sample_indices)}/{len(texts)} samples")
                
                sample_embeddings = process_deepseek_batch(sample_texts)
                
                # InterpolaÃ§Ã£o simples
                full_embeddings = np.zeros((len(texts), sample_embeddings.shape[1]), dtype=np.float32)
                
                for i, idx in enumerate(sample_indices):
                    full_embeddings[idx] = sample_embeddings[i]
                
                # Fill missing with nearest neighbor
                non_sample_indices = [i for i in range(len(texts)) if i not in sample_indices]
                for idx in non_sample_indices:
                    # Find closest sampled index
                    closest_sample_idx = sample_indices[np.argmin([abs(idx - s_idx) for s_idx in sample_indices])]
                    pos = np.where(sample_indices == closest_sample_idx)[0][0]
                    
                    # Add small noise
                    noise = np.random.normal(0, 0.01, sample_embeddings.shape[1])
                    full_embeddings[idx] = sample_embeddings[pos] + noise
                
                return full_embeddings
            else:
                return process_deepseek_batch(texts)
        
        def process_deepseek_batch(texts):
            out = []
            
            for i, text in enumerate(texts):
                if i % 500 == 0:
                    print(f"DeepSeek progress: {i}/{len(texts)} ({i/len(texts)*100:.1f}%)")
                
                try:
                    inputs = deepseek_tokenizer(
                        text,
                        truncation=True,
                        max_length=96,  # Reduced
                        return_tensors='pt'
                    ).to(DEEPSEEK_DEVICE)
                    
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            outputs = deepseek_model(**inputs, output_hidden_states=True)
                            # Just use last layer - simpler
                            emb = outputs.hidden_states[-1][0, 0, :].float().cpu().numpy()
                    
                    out.append(emb)
                    
                except Exception as e:
                    if i < 3:
                        print(f"Error on item {i}: {e}")
                    out.append(np.zeros(4096, dtype=np.float32))
                
                if i % 200 == 0:  # More frequent cleanup
                    torch.cuda.empty_cache()
            
            return np.array(out)
        
        start_time = time.time()
        
        sample_ratio = DEEPSEEK_SAMPLE_RATIO
        train_deepseek_embeddings = embed_texts_deepseek_simple(train_df['text'].tolist(), sample_ratio)
        test_deepseek_embeddings = embed_texts_deepseek_simple(test_df['text'].tolist(), sample_ratio)
        
        elapsed = time.time() - start_time
        print(f"DeepSeek processing completed in {elapsed/60:.1f} minutes")
        print(f"DeepSeek embeddings shape: {train_deepseek_embeddings.shape}")
        
        if USE_CACHE:
            with open(DEEPSEEK_EMB_CACHE, 'wb') as f:
                pickle.dump({'train': train_deepseek_embeddings, 'test': test_deepseek_embeddings}, f)
            print("DeepSeek embeddings cached")
        
        del deepseek_model, deepseek_tokenizer
        torch.cuda.empty_cache(); gc.collect()
        
    else:
        print("Using zero embeddings for DeepSeek")
        train_deepseek_embeddings = np.zeros((len(train_df), 768), dtype=np.float32)
        test_deepseek_embeddings = np.zeros((len(test_df), 768), dtype=np.float32)

if train_deepseek_embeddings is None:
    print("DeepSeek disabled, using zero embeddings")
    train_deepseek_embeddings = np.zeros((len(train_df), 768), dtype=np.float32)
    test_deepseek_embeddings = np.zeros((len(test_df), 768), dtype=np.float32)

# =========================
# TF-IDF Features (Simplificado)
# =========================
print("\n=== PROCESSING TF-IDF ===")

if USE_TFIDF_FEATURES:
    print(f"Generating TF-IDF features (max_features={TFIDF_MAX_FEATURES})...")
    
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=3,  # Increased
        max_df=0.9,  # Reduced
        stop_words='english',
        lowercase=True
    )
    
    all_texts = train_df['text_for_tfidf'].tolist() + test_df['text_for_tfidf'].tolist()
    tfidf.fit(all_texts)
    
    train_tfidf = tfidf.transform(train_df['text_for_tfidf']).toarray()
    test_tfidf = tfidf.transform(test_df['text_for_tfidf']).toarray()
    
    print(f"TF-IDF features shape: {train_tfidf.shape}")
else:
    train_tfidf = np.zeros((len(train_df), 0))
    test_tfidf = np.zeros((len(test_df), 0))

# =========================
# Feature Assembly Simplificada
# =========================
print("\n=== SIMPLE FEATURE ASSEMBLY ===")

enhanced_math_features_train = train_df[feature_columns].values
enhanced_math_features_test = test_df[feature_columns].values

# Combine embeddings
embeddings_train = np.concatenate([train_deberta_embeddings, train_deepseek_embeddings], axis=1)
embeddings_test = np.concatenate([test_deberta_embeddings, test_deepseek_embeddings], axis=1)

# PCA com componentes reduzidos
pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
embeddings_train_pca = pca.fit_transform(embeddings_train)
embeddings_test_pca = pca.transform(embeddings_test)

print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Feature selection simples
y = train_df['label'].values
available_features = enhanced_math_features_train.shape[1]
k_select = min(FEATURE_SELECTION_K, available_features)

print(f"Selecting {k_select} from {available_features} available math features")

# Usar apenas f_classif para simplicidade
selector = SelectKBest(f_classif, k=k_select)
math_features_train_selected = selector.fit_transform(enhanced_math_features_train, y)
math_features_test_selected = selector.transform(enhanced_math_features_test)

# Combine all features (sem interactions)
X = np.concatenate([
    embeddings_train_pca,           # PCA embeddings
    math_features_train_selected,   # Selected math features
    train_tfidf,                    # TF-IDF features
], axis=1)

X_test = np.concatenate([
    embeddings_test_pca,
    math_features_test_selected,
    test_tfidf,
], axis=1)

print(f"Final feature dimension: {X.shape[1]}")

# Cleanup
del embeddings_train, embeddings_test, train_deberta_embeddings, train_deepseek_embeddings
del test_deberta_embeddings, test_deepseek_embeddings
gc.collect()

# =========================
# Modelos Simplificados
# =========================
print("\n=== SIMPLIFIED MODELING ===")

# Modelos base reduzidos e otimizados
base_models = {
    'xgb1': xgb.XGBClassifier(
        n_estimators=80, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=2
    ),
    'lgb1': LGBMClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        num_leaves=31, feature_fraction=0.8, 
        verbose=-1, random_state=42, n_jobs=2
    ),
    'rf': RandomForestClassifier(
        n_estimators=50, max_depth=12, 
        random_state=42, n_jobs=2
    )
}

# Split para validaÃ§Ã£o
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Treinar modelos base
trained_base = {}
val_probs_base = {}

print("Training base models...")
for name, model in base_models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)
    
    if probs.shape[1] < n_classes:
        p_full = np.zeros((probs.shape[0], n_classes))
        for i, cls in enumerate(model.classes_):
            if cls < n_classes:
                p_full[:, cls] = probs[:, i]
        probs = p_full
    
    val_probs_base[name] = probs
    trained_base[name] = model
    
    val_acc = accuracy_score(y_val, np.argmax(probs, axis=1))
    elapsed = time.time() - start_time
    print(f"{name}: accuracy={val_acc:.4f}, time={elapsed:.1f}s")

# =========================
# Voting Ensemble Simples
# =========================
print("\nTraining voting ensemble...")

# Use apenas voting ensemble - mais simples
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='soft'  # Use probabilities
)

voting_clf.fit(X_train, y_train)
voting_probs = voting_clf.predict_proba(X_val)

if voting_probs.shape[1] < n_classes:
    p_full = np.zeros((voting_probs.shape[0], n_classes))
    for i, cls in enumerate(voting_clf.classes_):
        if cls < n_classes:
            p_full[:, cls] = voting_probs[:, i]
    voting_probs = p_full

voting_acc = accuracy_score(y_val, np.argmax(voting_probs, axis=1))
print(f"Voting ensemble accuracy: {voting_acc:.4f}")

# =========================
# Ensemble Final Simplificado
# =========================
print("\nSimple ensemble optimization...")

# Calcular accuracies
model_accuracies = {}
for name, probs in val_probs_base.items():
    acc = accuracy_score(y_val, np.argmax(probs, axis=1))
    model_accuracies[name] = acc

# Add voting ensemble
val_probs_base['voting'] = voting_probs
trained_base['voting'] = voting_clf
model_accuracies['voting'] = voting_acc

# Usar apenas os top 2 modelos + voting
sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
top_models = dict(sorted_models[:3])  # Top 3 models

print(f"\nTop models for ensemble: {list(top_models.keys())}")

# Pesos baseados apenas em accuracy
total_acc = sum(top_models.values())
final_weights = {name: acc/total_acc for name, acc in top_models.items()}

print("\nFinal model weights:")
for name, weight in final_weights.items():
    print(f"{name}: {weight:.3f} (acc: {model_accuracies[name]:.4f})")

# Ensemble final
final_val_probs = sum(final_weights[name] * val_probs_base[name] for name in final_weights.keys())
final_val_acc = accuracy_score(y_val, np.argmax(final_val_probs, axis=1))

print(f"\nFinal Ensemble Validation Accuracy: {final_val_acc:.6f}")

# =========================
# Test Predictions
# =========================
print("\n=== TEST PREDICTIONS ===")

test_probs_final = {}

for name in final_weights.keys():
    print(f"Predicting with {name}...")
    
    model = trained_base[name]
    model.fit(X, y)
    probs = model.predict_proba(X_test)
    
    if probs.shape[1] < n_classes:
        p_full = np.zeros((probs.shape[0], n_classes))
        for i, cls in enumerate(model.classes_):
            if cls < n_classes:
                p_full[:, cls] = probs[:, i]
        probs = p_full
    
    test_probs_final[name] = probs

# Ensemble final no test
final_test_probs = sum(final_weights[name] * test_probs_final[name] for name in final_weights.keys())

# =========================
# Submission
# =========================
top3 = np.argsort(-final_test_probs, axis=1)[:, :3]
preds = []
for idxs in top3:
    labels = le.inverse_transform(idxs)
    preds.append(f"{labels[0]} {labels[1]} {labels[2]}")

submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': preds
})
submission.to_csv('submission.csv', index=False)

# =========================
# Summary
# =========================
print("\n=== OPTIMIZED PIPELINE SUMMARY ===")
print(f"Final Accuracy: {final_val_acc:.6f}")
print(f"Total features used: {X.shape[1]}")
print(f"  - PCA embedding features: {PCA_COMPONENTS}")
print(f"  - Math features: {math_features_train_selected.shape[1]}")
print(f"  - TF-IDF features: {train_tfidf.shape[1]}")
print(f"Models in final ensemble: {list(final_weights.keys())}")
print(f"Best individual model: {max(model_accuracies.items(), key=lambda x: x[1])}")
print(f"Complexity reduction: Simplified from 20+ features to {len(feature_columns)}, removed stacking/interactions")
print("Submission saved: submission.csv")

# Salvar pipeline simplificado para anÃ¡lise
simple_artifact = {
    'final_weights': final_weights,
    'model_accuracies': model_accuracies,
    'final_accuracy': final_val_acc,
    'feature_info': {
        'pca_components': PCA_COMPONENTS,
        'pca_variance_explained': pca.explained_variance_ratio_.sum(),
        'total_features': X.shape[1],
        'math_features': len(feature_columns),
        'tfidf_features': TFIDF_MAX_FEATURES if USE_TFIDF_FEATURES else 0
    },
    'models_used': list(final_weights.keys()),
    'best_individual': max(model_accuracies.items(), key=lambda x: x[1]),
    'optimizations_applied': {
        'deepseek_sampling': DEEPSEEK_SAMPLE_RATIO,
        'reduced_pca': PCA_COMPONENTS,
        'simplified_features': len(feature_columns),
        'no_stacking': not USE_STACKING,
        'no_interactions': not USE_FEATURE_INTERACTIONS,
        'reduced_models': len(base_models)
    }
}

with open('optimized_pipeline_results.pkl', 'wb') as f:
    pickle.dump(simple_artifact, f)

print("Optimized pipeline analysis saved: optimized_pipeline_results.pkl")




