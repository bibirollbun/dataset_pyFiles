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


"""
ULTIMATE CLICKBAIT SPOILER DETECTION SYSTEM
Fully Robust, Error-Tolerant, Visual-Rich Pipeline
With 20+ NLP Libraries and Comprehensive Fallbacks
"""

# =====================================
# SAFE IMPORTS WITH FALLBACKS
# =====================================
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import subprocess
import json
import re
import gc
import time
import pickle
from collections import Counter, defaultdict
from datetime import datetime

# Core libraries (always available)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# Track installation status
INSTALL_STATUS = {}

def safe_install(package_name, import_name=None):
    """Safely install and import a package"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        INSTALL_STATUS[package_name] = "âœ“ Already installed"
        return True
    except ImportError:
        try:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
            __import__(import_name)
            INSTALL_STATUS[package_name] = "âœ“ Installed successfully"
            return True
        except:
            INSTALL_STATUS[package_name] = "âœ— Failed to install"
            return False

# =====================================
# SMART PACKAGE INSTALLATION
# =====================================
print("=" * 80)
print("ğŸš€ ULTIMATE CLICKBAIT DETECTOR - SETUP")
print("=" * 80)

# Detect environment
IS_KAGGLE = os.path.exists('/kaggle/input')
print(f"ğŸ“� Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"ğŸ“… Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "=" * 80)
print("ğŸ“¦ INSTALLING REQUIRED PACKAGES")
print("=" * 80)

# Core ML packages
packages_to_install = [
    ('xgboost', 'xgboost'),
    ('lightgbm', 'lightgbm'),
    ('catboost', 'catboost'),
    ('sentence-transformers', 'sentence_transformers'),
    ('transformers', 'transformers'),
    ('textstat', 'textstat'),
    ('vaderSentiment', 'vaderSentiment'),
    ('textblob', 'textblob'),
    ('wordcloud', 'wordcloud'),
    ('tqdm', 'tqdm')
]

for package, import_name in packages_to_install:
    safe_install(package, import_name)

# Print installation summary
print("\nğŸ“Š Installation Summary:")
for package, status in INSTALL_STATUS.items():
    print(f"  {package}: {status}")

# =====================================
# DYNAMIC IMPORTS WITH FALLBACKS
# =====================================
print("\n" + "=" * 80)
print("ğŸ”§ LOADING LIBRARIES")
print("=" * 80)

# Dictionary to track what's available
AVAILABLE = {}

# Try importing each library
try:
    from tqdm.auto import tqdm
    AVAILABLE['tqdm'] = True
except:
    AVAILABLE['tqdm'] = False
    # Fallback tqdm
    class tqdm:
        def __init__(self, iterable, desc="", total=None):
            self.iterable = iterable
            self.desc = desc
        def __iter__(self):
            return iter(self.iterable)

try:
    import xgboost as xgb
    AVAILABLE['xgboost'] = True
    print("âœ“ XGBoost loaded")
except:
    AVAILABLE['xgboost'] = False
    print("âœ— XGBoost not available")

try:
    import lightgbm as lgb
    AVAILABLE['lightgbm'] = True
    print("âœ“ LightGBM loaded")
except:
    AVAILABLE['lightgbm'] = False
    print("âœ— LightGBM not available")

try:
    import catboost as cb
    AVAILABLE['catboost'] = True
    print("âœ“ CatBoost loaded")
except:
    AVAILABLE['catboost'] = False
    print("âœ— CatBoost not available")

try:
    from sentence_transformers import SentenceTransformer
    AVAILABLE['sentence_transformers'] = True
    print("âœ“ Sentence Transformers loaded")
except:
    AVAILABLE['sentence_transformers'] = False
    print("âœ— Sentence Transformers not available")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    AVAILABLE['vader'] = True
    print("âœ“ VADER Sentiment loaded")
except:
    AVAILABLE['vader'] = False
    print("âœ— VADER not available")

try:
    from textblob import TextBlob
    AVAILABLE['textblob'] = True
    print("âœ“ TextBlob loaded")
except:
    AVAILABLE['textblob'] = False
    print("âœ— TextBlob not available")

try:
    import textstat
    AVAILABLE['textstat'] = True
    print("âœ“ Textstat loaded")
except:
    AVAILABLE['textstat'] = False
    print("âœ— Textstat not available")

try:
    from wordcloud import WordCloud
    AVAILABLE['wordcloud'] = True
    print("âœ“ WordCloud loaded")
except:
    AVAILABLE['wordcloud'] = False
    print("âœ— WordCloud not available")

# Set random seeds
np.random.seed(42)

# =====================================
# CONFIGURATION
# =====================================
print("\n" + "=" * 80)
print("âš™ï¸� CONFIGURATION")
print("=" * 80)

class Config:
    # Data paths
    if IS_KAGGLE:
        train_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/train.jsonl'
        val_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/val.jsonl'
        test_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/test.jsonl'
    else:
        train_path = 'train.jsonl'
        val_path = 'val.jsonl'
        test_path = 'test.jsonl'
    
    # Feature settings
    max_features = 3000
    use_all_features = True
    
    # Visualization settings
    create_visualizations = True
    figure_size = (12, 8)
    color_palette = 'viridis'

config = Config()
print("âœ“ Configuration loaded")

# =====================================
# DATA LOADING WITH ERROR HANDLING
# =====================================
print("\n" + "=" * 80)
print("ğŸ“‚ LOADING DATA")
print("=" * 80)

def safe_load_jsonl(file_path):
    """Safely load JSONL file with error handling"""
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping line {line_num} due to JSON error: {e}")
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"  Error: File {file_path} not found!")
        return pd.DataFrame()
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return pd.DataFrame()

# Load datasets
train_df = safe_load_jsonl(config.train_path)
val_df = safe_load_jsonl(config.val_path)
test_df = safe_load_jsonl(config.test_path)

# Fix tags column safely
for df in [train_df, val_df]:
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

print(f"âœ“ Training samples: {len(train_df)}")
print(f"âœ“ Validation samples: {len(val_df)}")
print(f"âœ“ Test samples: {len(test_df)}")

if len(train_df) == 0:
    print("\nâš ï¸� No training data loaded. Using synthetic data for demonstration...")
    # Create synthetic data for demonstration
    train_df = pd.DataFrame({
        'postText': ['Click here to see what happens next!'] * 100,
        'targetTitle': ['Amazing story'] * 100,
        'targetParagraphs': [['Paragraph 1', 'Paragraph 2']] * 100,
        'tags': ['phrase'] * 34 + ['passage'] * 33 + ['multi'] * 33
    })
    val_df = train_df.sample(20)
    test_df = train_df.sample(20)

# =====================================
# EXPLORATORY DATA ANALYSIS WITH VISUALIZATIONS
# =====================================
print("\n" + "=" * 80)
print("ğŸ“Š EXPLORATORY DATA ANALYSIS")
print("=" * 80)

def create_eda_visualizations(df):
    """Create comprehensive EDA visualizations"""
    if not config.create_visualizations or len(df) == 0:
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Distribution of spoiler types
    ax1 = plt.subplot(2, 3, 1)
    if 'tags' in df.columns:
        tag_counts = df['tags'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(tag_counts)))
        bars = ax1.bar(tag_counts.index, tag_counts.values, color=colors)
        ax1.set_title('Distribution of Spoiler Types', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Spoiler Type')
        ax1.set_ylabel('Count')
        # Add value labels on bars
        for bar, count in zip(bars, tag_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{count}\n({count/len(df)*100:.1f}%)', 
                    ha='center', va='bottom')
    
    # 2. Text length distribution
    ax2 = plt.subplot(2, 3, 2)
    if 'postText' in df.columns:
        post_lengths = df['postText'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        ax2.hist(post_lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(post_lengths.mean(), color='red', linestyle='--', label=f'Mean: {post_lengths.mean():.1f}')
        ax2.set_title('Post Text Length Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    
    # 3. Paragraph count distribution
    ax3 = plt.subplot(2, 3, 3)
    if 'targetParagraphs' in df.columns:
        para_counts = df['targetParagraphs'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        ax3.hist(para_counts, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax3.axvline(para_counts.mean(), color='red', linestyle='--', label=f'Mean: {para_counts.mean():.1f}')
        ax3.set_title('Number of Target Paragraphs', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Paragraph Count')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    
    # 4. Text length by spoiler type (boxplot)
    ax4 = plt.subplot(2, 3, 4)
    if 'tags' in df.columns and 'postText' in df.columns:
        length_by_tag = []
        tags_list = []
        for tag in df['tags'].unique():
            if pd.notna(tag):
                lengths = df[df['tags'] == tag]['postText'].apply(
                    lambda x: len(str(x).split()) if pd.notna(x) else 0
                )
                length_by_tag.append(lengths)
                tags_list.append(tag)
        
        if length_by_tag:
            bp = ax4.boxplot(length_by_tag, labels=tags_list, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(tags_list)]):
                patch.set_facecolor(color)
            ax4.set_title('Post Length by Spoiler Type', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Spoiler Type')
            ax4.set_ylabel('Word Count')
    
    # 5. Common words in posts (horizontal bar chart)
    ax5 = plt.subplot(2, 3, 5)
    if 'postText' in df.columns:
        all_words = ' '.join(df['postText'].dropna().astype(str)).lower().split()
        # Remove common stop words manually
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be'}
        filtered_words = [w for w in all_words if w not in stop_words and len(w) > 2]
        word_freq = Counter(filtered_words).most_common(10)
        
        if word_freq:
            words, counts = zip(*word_freq)
            y_pos = np.arange(len(words))
            ax5.barh(y_pos, counts, color='coral')
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(words)
            ax5.set_title('Top 10 Most Common Words', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Frequency')
    
    # 6. Correlation heatmap placeholder
    ax6 = plt.subplot(2, 3, 6)
    if 'tags' in df.columns:
        # Create a simple feature correlation matrix
        simple_features = pd.DataFrame()
        simple_features['post_length'] = df['postText'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        simple_features['title_length'] = df.get('targetTitle', pd.Series()).apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        simple_features['para_count'] = df.get('targetParagraphs', pd.Series()).apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        if len(simple_features.columns) > 1:
            corr_matrix = simple_features.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax6, cbar_kws={"shrink": 0.8})
            ax6.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.suptitle('ğŸ“Š Comprehensive Data Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Additional visualizations
    if AVAILABLE['wordcloud'] and 'postText' in df.columns:
        print("\nğŸ“� Generating Word Cloud...")
        try:
            text = ' '.join(df['postText'].dropna().astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                 colormap='viridis', max_words=100).generate(text)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('ğŸŒŸ Word Cloud of Post Texts', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"  Could not generate word cloud: {e}")

# Run EDA
create_eda_visualizations(train_df)

# =====================================
# ROBUST FEATURE ENGINEERING
# =====================================
print("\n" + "=" * 80)
print("ğŸ”¬ FEATURE ENGINEERING")
print("=" * 80)

class RobustFeatureExtractor:
    """Robust feature extractor with comprehensive error handling"""
    
    def __init__(self):
        self.feature_names = []
        if AVAILABLE['vader']:
            self.vader = SentimentIntensityAnalyzer()
    
    def safe_extract(self, func, df, feature_name):
        """Safely extract a feature with error handling"""
        try:
            return func(df)
        except Exception as e:
            print(f"  âš ï¸� Warning: Could not extract {feature_name}: {e}")
            return pd.Series([0] * len(df))
    
    def extract_all_features(self, df):
        """Extract all features with comprehensive error handling"""
        features = pd.DataFrame()
        
        # 1. Basic text statistics (always works)
        print("  ğŸ“Š Extracting basic statistics...")
        features['post_word_count'] = self.safe_extract(
            lambda d: d.get('postText', pd.Series()).apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            ), df, 'post_word_count'
        )
        
        features['title_word_count'] = self.safe_extract(
            lambda d: d.get('targetTitle', pd.Series()).apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            ), df, 'title_word_count'
        )
        
        features['content_word_count'] = self.safe_extract(
            lambda d: d.get('targetParagraphs', pd.Series()).apply(
                lambda x: sum(len(str(p).split()) for p in x) if isinstance(x, list) else 0
            ), df, 'content_word_count'
        )
        
        features['post_char_count'] = self.safe_extract(
            lambda d: d.get('postText', pd.Series()).apply(
                lambda x: len(str(x)) if pd.notna(x) else 0
            ), df, 'post_char_count'
        )
        
        features['num_paragraphs'] = self.safe_extract(
            lambda d: d.get('targetParagraphs', pd.Series()).apply(
                lambda x: len(x) if isinstance(x, list) else 0
            ), df, 'num_paragraphs'
        )
        
        # 2. Punctuation features (with fixed regex)
        print("  â�— Extracting punctuation features...")
        punctuation_patterns = {
            'question_marks': r'\?',
            'exclamation_marks': r'!',
            'periods': r'\.',
            'commas': r',',
            'ellipsis': r'\.{3,}',
            'quotes': r'["\']'
        }
        
        for name, pattern in punctuation_patterns.items():
            features[f'punct_{name}'] = self.safe_extract(
                lambda d: d.get('postText', pd.Series()).apply(
                    lambda x: len(re.findall(pattern, str(x))) if pd.notna(x) else 0
                ), df, f'punct_{name}'
            )
        
        # 3. Clickbait patterns
        print("  ğŸ�¯ Extracting clickbait patterns...")
        features['starts_with_number'] = self.safe_extract(
            lambda d: d.get('postText', pd.Series()).apply(
                lambda x: 1 if pd.notna(x) and re.match(r'^\d+', str(x).strip()) else 0
            ), df, 'starts_with_number'
        )
        
        # Pronoun counts
        pronouns = ['you', 'your', 'this', 'that', 'these', 'those']
        for pronoun in pronouns:
            features[f'has_{pronoun}'] = self.safe_extract(
                lambda d: d.get('postText', pd.Series()).apply(
                    lambda x: 1 if pd.notna(x) and pronoun in str(x).lower() else 0
                ), df, f'has_{pronoun}'
            )
        
        # 4. Sentiment features (if available)
        if AVAILABLE['vader']:
            print("  ğŸ˜Š Extracting sentiment features...")
            try:
                sentiment_scores = df.get('postText', pd.Series()).apply(
                    lambda x: self.vader.polarity_scores(str(x)) if pd.notna(x) else {}
                )
                features['sentiment_compound'] = sentiment_scores.apply(lambda x: x.get('compound', 0))
                features['sentiment_positive'] = sentiment_scores.apply(lambda x: x.get('pos', 0))
                features['sentiment_negative'] = sentiment_scores.apply(lambda x: x.get('neg', 0))
                features['sentiment_neutral'] = sentiment_scores.apply(lambda x: x.get('neu', 0))
            except:
                print("    Could not extract VADER sentiment")
        
        if AVAILABLE['textblob']:
            print("  ğŸ’­ Extracting TextBlob features...")
            try:
                features['textblob_polarity'] = df.get('postText', pd.Series()).apply(
                    lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
                )
                features['textblob_subjectivity'] = df.get('postText', pd.Series()).apply(
                    lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0
                )
            except:
                print("    Could not extract TextBlob sentiment")
        
        # 5. Readability features (if available)
        if AVAILABLE['textstat']:
            print("  ğŸ“– Extracting readability features...")
            try:
                features['flesch_reading_ease'] = df.get('postText', pd.Series()).apply(
                    lambda x: textstat.flesch_reading_ease(str(x)) if pd.notna(x) and len(str(x)) > 10 else 0
                )
                features['syllable_count'] = df.get('postText', pd.Series()).apply(
                    lambda x: textstat.syllable_count(str(x)) if pd.notna(x) else 0
                )
            except:
                print("    Could not extract readability features")
        
        # 6. Ratios and derived features
        print("  ğŸ“� Computing derived features...")
        features['post_to_content_ratio'] = (
            features['post_word_count'] / (features['content_word_count'] + 1)
        ).fillna(0).replace([np.inf, -np.inf], 0)
        
        features['avg_word_length'] = self.safe_extract(
            lambda d: d.get('postText', pd.Series()).apply(
                lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) and len(str(x).split()) > 0 else 0
            ), df, 'avg_word_length'
        )
        
        # Fill any NaN values
        features = features.fillna(0)
        
        print(f"  âœ“ Extracted {len(features.columns)} features")
        self.feature_names = features.columns.tolist()
        
        return features

def create_combined_text(df):
    """Create combined text for model input"""
    texts = []
    for idx, row in df.iterrows():
        post_text = str(row.get('postText', ''))
        target_title = str(row.get('targetTitle', ''))
        
        if 'targetParagraphs' in row and isinstance(row['targetParagraphs'], list):
            target_preview = ' '.join([str(p)[:200] for p in row['targetParagraphs'][:2]])
        else:
            target_preview = str(row.get('targetParagraphs', ''))[:400]
        
        combined = f"{post_text} [SEP] {target_title} [SEP] {target_preview}"
        texts.append(combined)
    
    return texts

# Extract features
print("\nğŸ”¬ Extracting features from datasets...")
feature_extractor = RobustFeatureExtractor()

train_features = feature_extractor.extract_all_features(train_df)
val_features = feature_extractor.extract_all_features(val_df)
test_features = feature_extractor.extract_all_features(test_df)

print(f"\nâœ“ Feature extraction complete!")
print(f"  Train shape: {train_features.shape}")
print(f"  Val shape: {val_features.shape}")
print(f"  Test shape: {test_features.shape}")

# =====================================
# TEXT VECTORIZATION
# =====================================
print("\n" + "=" * 80)
print("ğŸ“� TEXT VECTORIZATION")
print("=" * 80)

# Create combined texts
train_texts = create_combined_text(train_df)
val_texts = create_combined_text(val_df)
test_texts = create_combined_text(test_df)

# TF-IDF Vectorization
print("  Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=config.max_features,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(train_texts)
X_val_tfidf = tfidf.transform(val_texts)
X_test_tfidf = tfidf.transform(test_texts)

print(f"  âœ“ TF-IDF shape: {X_train_tfidf.shape}")

# Try to add embeddings if available
X_train_embeddings = None
X_val_embeddings = None
X_test_embeddings = None

if AVAILABLE['sentence_transformers']:
    print("\n  ğŸ§¬ Generating sentence embeddings...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        X_train_embeddings = model.encode(train_texts, show_progress_bar=True, batch_size=32)
        X_val_embeddings = model.encode(val_texts, show_progress_bar=True, batch_size=32)
        X_test_embeddings = model.encode(test_texts, show_progress_bar=True, batch_size=32)
        print(f"  âœ“ Embeddings shape: {X_train_embeddings.shape}")
    except Exception as e:
        print(f"  âš ï¸� Could not generate embeddings: {e}")

# =====================================
# COMBINE FEATURES
# =====================================
print("\n" + "=" * 80)
print("ğŸ”— COMBINING FEATURES")
print("=" * 80)

# Scale numerical features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)

# Combine all features
feature_list_train = [X_train_tfidf, train_features_scaled]
feature_list_val = [X_val_tfidf, val_features_scaled]
feature_list_test = [X_test_tfidf, test_features_scaled]

if X_train_embeddings is not None:
    feature_list_train.append(X_train_embeddings)
    feature_list_val.append(X_val_embeddings)
    feature_list_test.append(X_test_embeddings)

X_train = hstack(feature_list_train)
X_val = hstack(feature_list_val)
X_test = hstack(feature_list_test)

print(f"âœ“ Final feature dimensions:")
print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")
print(f"  Test: {X_test.shape}")

# Create labels
label_encoder = LabelEncoder()
if 'tags' in train_df.columns:
    y_train = label_encoder.fit_transform(train_df['tags'])
    y_val = label_encoder.transform(val_df['tags'])
else:
    # Fallback for synthetic data
    y_train = np.random.randint(0, 3, len(train_df))
    y_val = np.random.randint(0, 3, len(val_df))
    label_encoder.classes_ = np.array(['phrase', 'passage', 'multi'])

print(f"\nâœ“ Label encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} -> {i}")

# =====================================
# MODEL TRAINING WITH ENSEMBLE
# =====================================
print("\n" + "=" * 80)
print("ğŸ¤– MODEL TRAINING")
print("=" * 80)

def train_model_with_cv(X, y, model, model_name, cv_folds=5):
    """Train model with cross-validation"""
    try:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
        print(f"  {model_name}: CV F1={scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores.mean()
    except Exception as e:
        print(f"  âš ï¸� Could not train {model_name}: {e}")
        return 0

# Train multiple models
models = {}
model_scores = {}

# 1. Logistic Regression (always available)
print("\nğŸ“ˆ Training models...")
lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
lr_score = train_model_with_cv(X_train, y_train, lr, "Logistic Regression", cv_folds=3)
lr.fit(X_train, y_train)
models['lr'] = lr
model_scores['lr'] = lr_score

# 2. Random Forest (always available)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf_score = train_model_with_cv(X_train, y_train, rf, "Random Forest", cv_folds=3)
rf.fit(X_train, y_train)
models['rf'] = rf
model_scores['rf'] = rf_score

# 3. Gradient Boosting (always available)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_score = train_model_with_cv(X_train, y_train, gb, "Gradient Boosting", cv_folds=3)
gb.fit(X_train, y_train)
models['gb'] = gb
model_scores['gb'] = gb_score

# 4. XGBoost (if available)
if AVAILABLE['xgboost']:
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_score = train_model_with_cv(X_train, y_train, xgb_model, "XGBoost", cv_folds=3)
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    model_scores['xgb'] = xgb_score

# 5. LightGBM (if available)
if AVAILABLE['lightgbm']:
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=-1)
    lgb_score = train_model_with_cv(X_train, y_train, lgb_model, "LightGBM", cv_folds=3)
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    model_scores['lgb'] = lgb_score

# 6. CatBoost (if available)
if AVAILABLE['catboost']:
    cb_model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=False)
    cb_score = train_model_with_cv(X_train, y_train, cb_model, "CatBoost", cv_folds=3)
    cb_model.fit(X_train, y_train)
    models['cb'] = cb_model
    model_scores['cb'] = cb_score

# Create ensemble if we have multiple models
if len(models) > 1:
    print("\nğŸ�­ Creating ensemble model...")
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    final_model = ensemble
    model_name = "Ensemble"
else:
    # Use best single model
    best_model_name = max(model_scores, key=model_scores.get)
    final_model = models[best_model_name]
    model_name = best_model_name

# =====================================
# MODEL EVALUATION
# =====================================
print("\n" + "=" * 80)
print("ğŸ“Š MODEL EVALUATION")
print("=" * 80)

# Evaluate on validation set
val_pred = final_model.predict(X_val)
val_f1 = f1_score(y_val, val_pred, average='macro')

print(f"\nâœ… Best Model: {model_name}")
print(f"âœ… Validation F1 Score: {val_f1:.4f}")

# Classification report
print("\nğŸ“‹ Classification Report:")
print(classification_report(y_val, val_pred, target_names=label_encoder.classes_))

# Confusion Matrix Visualization
if config.create_visualizations:
    cm = confusion_matrix(y_val, val_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# Feature importance visualization (if applicable)
if hasattr(final_model, 'feature_importances_') or (hasattr(final_model, 'estimators_') and hasattr(final_model.estimators_[0], 'feature_importances_')):
    print("\nğŸ“Š Generating feature importance plot...")
    
    # Get feature importances
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
    else:
        # For ensemble, average importances
        importances = np.zeros(X_train.shape[1])
        for estimator in final_model.estimators_:
            if hasattr(estimator[1], 'feature_importances_'):
                importances += estimator[1].feature_importances_
        importances /= len(final_model.estimators_)
    
    # Get top features
    num_tfidf = X_train_tfidf.shape[1]
    num_manual = len(feature_extractor.feature_names)
    
    # Separate TF-IDF and manual features
    tfidf_importance = importances[:num_tfidf]
    manual_importance = importances[num_tfidf:num_tfidf+num_manual]
    
    # Top manual features
    if len(manual_importance) > 0:
        top_indices = np.argsort(manual_importance)[-10:]
        top_features = [feature_extractor.feature_names[i] for i in top_indices]
        top_importances = manual_importance[top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importances, color='teal')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

# =====================================
# GENERATE PREDICTIONS
# =====================================
print("\n" + "=" * 80)
print("ğŸ�¯ GENERATING PREDICTIONS")
print("=" * 80)

# Make predictions
test_predictions = final_model.predict(X_test)
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Get prediction probabilities for confidence
if hasattr(final_model, 'predict_proba'):
    test_proba = final_model.predict_proba(X_test)
    confidence_scores = np.max(test_proba, axis=1)
else:
    confidence_scores = np.ones(len(test_predictions))

# Create submission
submission = pd.DataFrame({
    'id': range(len(test_df)),
    'spoilerType': test_predictions_labels
})

# =====================================
# RESULTS SUMMARY
# =====================================
print("\n" + "=" * 80)
print("ğŸ“ˆ RESULTS SUMMARY")
print("=" * 80)

print(f"\nğŸ�† Final Results:")
print(f"  Model: {model_name}")
print(f"  Validation F1: {val_f1:.4f}")
print(f"  Features Used: {X_train.shape[1]}")
print(f"  Libraries Available: {sum(AVAILABLE.values())}/{len(AVAILABLE)}")

print(f"\nğŸ“Š Prediction Distribution:")
pred_counts = submission['spoilerType'].value_counts()
for label, count in pred_counts.items():
    print(f"  {label}: {count} ({count/len(submission)*100:.1f}%)")

# Confidence analysis
print(f"\nğŸ�¯ Prediction Confidence:")
print(f"  Mean: {confidence_scores.mean():.3f}")
print(f"  Min: {confidence_scores.min():.3f}")
print(f"  Max: {confidence_scores.max():.3f}")

# Sample predictions with confidence
print(f"\nğŸ“� Sample Predictions (first 10):")
for i in range(min(10, len(submission))):
    print(f"  ID {i}: {submission.iloc[i]['spoilerType']} (confidence: {confidence_scores[i]:.2f})")

# Save submission
submission.to_csv('submission_final.csv', index=False)
print(f"\nâœ… Predictions saved to 'submission_final.csv'")

# =====================================
# FINAL VISUALIZATIONS
# =====================================
if config.create_visualizations:
    print("\n" + "=" * 80)
    print("ğŸ�¨ FINAL VISUALIZATIONS")
    print("=" * 80)
    
    # Create final dashboard
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Model comparison
    ax1 = plt.subplot(2, 3, 1)
    model_names = list(model_scores.keys())
    scores = list(model_scores.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax1.bar(model_names, scores, color=colors)
    ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CV F1 Score')
    ax1.set_ylim([0, 1])
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 2. Prediction distribution
    ax2 = plt.subplot(2, 3, 2)
    pred_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=colors[:len(pred_counts)])
    ax2.set_title('Test Prediction Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('')
    
    # 3. Confidence distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(confidence_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(confidence_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {confidence_scores.mean():.3f}')
    ax3.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. Training history (placeholder)
    ax4 = plt.subplot(2, 3, 4)
    epochs = range(1, 11)
    train_scores = [0.3 + i*0.05 + np.random.uniform(-0.02, 0.02) for i in range(10)]
    val_scores = [0.25 + i*0.045 + np.random.uniform(-0.03, 0.03) for i in range(10)]
    ax4.plot(epochs, train_scores, 'b-', label='Training', marker='o')
    ax4.plot(epochs, val_scores, 'r-', label='Validation', marker='s')
    ax4.set_title('Training Progress (Simulated)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Validation vs Training comparison
    ax5 = plt.subplot(2, 3, 5)
    if 'tags' in train_df.columns:
        train_dist = train_df['tags'].value_counts(normalize=True)
        val_dist = val_df['tags'].value_counts(normalize=True)
        x = np.arange(len(train_dist))
        width = 0.35
        ax5.bar(x - width/2, train_dist.values, width, label='Train', color='blue', alpha=0.7)
        ax5.bar(x + width/2, val_dist.values, width, label='Val', color='red', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(train_dist.index)
        ax5.set_title('Train vs Val Distribution', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Proportion')
        ax5.legend()
    
    # 6. Performance metrics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    metrics_text = f"""
    ğŸ�† FINAL PERFORMANCE METRICS
    
    Model: {model_name}
    Validation F1: {val_f1:.4f}
    Total Features: {X_train.shape[1]}
    
    Available Libraries: {sum(AVAILABLE.values())}/{len(AVAILABLE)}
    âœ“ TF-IDF Features: {X_train_tfidf.shape[1]}
    âœ“ Manual Features: {len(feature_extractor.feature_names)}
    {'âœ“ Embeddings: Yes' if X_train_embeddings is not None else 'âœ— Embeddings: No'}
    
    Prediction Confidence:
    Mean: {confidence_scores.mean():.3f}
    Std: {confidence_scores.std():.3f}
    """
    ax6.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ğŸš€ Clickbait Detection - Final Results Dashboard', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 80)
print("âœ¨ PIPELINE COMPLETE! âœ¨")
print("=" * 80)
print(f"Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Thank you for using the Ultimate Clickbait Detector!")
print("=" * 80)

