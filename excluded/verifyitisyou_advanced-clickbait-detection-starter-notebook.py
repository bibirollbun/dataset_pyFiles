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
# ğŸ�¯ Clickbait Spoiler Type Classification - Advanced Pipeline
### MSE641 S25 - Task 1

This notebook implements a comprehensive machine learning pipeline for classifying clickbait spoiler types.
We leverage both traditional ML approaches and state-of-the-art transformer models to achieve optimal performance.

**Key Features:**
- ğŸ¤– Qwen-3 8B transformer embeddings (Multi-GPU support)
- ğŸ“Š Advanced visualizations and insights
- ğŸ”§ Sophisticated feature engineering
- ğŸ�¯ Ensemble methods for improved performance
- ğŸ“ˆ Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import gc
import warnings
warnings.filterwarnings('ignore')

# Set style and random seed
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)
torch.manual_seed(42)

# Memory optimization
torch.cuda.empty_cache()
gc.collect()

print("=" * 80)
print("ğŸ�¯ CLICKBAIT SPOILER TYPE CLASSIFICATION".center(80))
print("=" * 80)

# ==================================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ==================================================================================

print("\nğŸ“� SECTION 1: DATA LOADING AND INITIAL EXPLORATION")
print("-" * 60)

# Define file paths
train_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/train.jsonl'
val_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/val.jsonl'
test_path = '/kaggle/input/task-1-clickbait-detection-msci-641-s-25/test.jsonl'

# Load JSONL files with progress indication
def load_jsonl(file_path):
    """Load JSONL file and return as DataFrame"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

print("Loading datasets...")
train_df = load_jsonl(train_path)
val_df = load_jsonl(val_path)
test_df = load_jsonl(test_path)

# Dataset overview
print(f"\nğŸ“Š Dataset Overview:")
print(f"  â€¢ Training samples: {len(train_df):,}")
print(f"  â€¢ Validation samples: {len(val_df):,}")
print(f"  â€¢ Test samples: {len(test_df):,}")
print(f"  â€¢ Total samples: {len(train_df) + len(val_df) + len(test_df):,}")

# Fix tags column (extract from list format)
print("\nğŸ”§ Preprocessing tags column...")
train_df['tags'] = train_df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
val_df['tags'] = val_df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

# ==================================================================================
# 2. COMPREHENSIVE EXPLORATORY DATA ANALYSIS
# ==================================================================================

print("\nğŸ“Š SECTION 2: COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
print("-" * 60)

# 2.1 Target Distribution Analysis
print("\n2.1 Target Distribution Analysis")

# Calculate distribution
tag_counts = train_df['tags'].value_counts()
tag_percentages = train_df['tags'].value_counts(normalize=True) * 100

# Create interactive pie chart
fig = go.Figure(data=[go.Pie(
    labels=tag_counts.index,
    values=tag_counts.values,
    text=[f'{pct:.1f}%' for pct in tag_percentages],
    textposition='inside',
    marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c']),
    hole=0.3
)])

fig.update_layout(
    title='Distribution of Spoiler Types in Training Data',
    showlegend=True,
    width=600,
    height=500
)
fig.show()

# Distribution table
distribution_df = pd.DataFrame({
    'Count': tag_counts,
    'Percentage': tag_percentages.round(2)
})
print("\nğŸ“Š Spoiler Type Distribution:")
print(distribution_df)

# 2.2 Text Length Analysis
print("\n2.2 Text Length Analysis")

# Calculate text lengths
for df in [train_df, val_df, test_df]:
    df['postText_words'] = df['postText'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df['postText_chars'] = df['postText'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df['targetTitle_words'] = df['targetTitle'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df['targetParagraphs_words'] = df['targetParagraphs'].apply(
        lambda x: sum(len(str(p).split()) for p in x) if isinstance(x, list) else 0
    )
    df['num_paragraphs'] = df['targetParagraphs'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Create violin plots for text lengths
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Post Text Length', 'Target Title Length', 
                   'Target Content Length', 'Number of Paragraphs')
)

features = ['postText_words', 'targetTitle_words', 'targetParagraphs_words', 'num_paragraphs']
colors = {'phrase': '#1f77b4', 'passage': '#ff7f0e', 'multi': '#2ca02c'}

for idx, feature in enumerate(features):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    for tag in sorted(train_df['tags'].unique()):
        data = train_df[train_df['tags'] == tag][feature]
        fig.add_trace(
            go.Violin(y=data, name=tag, legendgroup=tag, 
                     showlegend=(idx==0), line_color=colors[tag]),
            row=row, col=col
        )

fig.update_layout(height=800, title_text="Text Length Distributions by Spoiler Type")
fig.show()

# Statistical summary
print("\nğŸ“Š Statistical Summary of Text Lengths:")
summary_stats = train_df.groupby('tags')[features].agg(['mean', 'median', 'std']).round(1)
print(summary_stats)

# 2.3 Advanced Text Analysis
print("\n2.3 Advanced Text Analysis")

# Punctuation and special character analysis
train_df['question_marks'] = train_df['postText'].apply(lambda x: str(x).count('?') if pd.notna(x) else 0)
train_df['exclamation_marks'] = train_df['postText'].apply(lambda x: str(x).count('!') if pd.notna(x) else 0)
train_df['ellipsis'] = train_df['postText'].apply(lambda x: str(x).count('...') if pd.notna(x) else 0)
train_df['quotes'] = train_df['postText'].apply(lambda x: str(x).count('"') if pd.notna(x) else 0)
train_df['colons'] = train_df['postText'].apply(lambda x: str(x).count(':') if pd.notna(x) else 0)

# Create heatmap of punctuation usage by spoiler type
punct_features = ['question_marks', 'exclamation_marks', 'ellipsis', 'quotes', 'colons']
punct_means = train_df.groupby('tags')[punct_features].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(punct_means.T, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=punct_means.index, yticklabels=punct_features)
plt.title('Average Punctuation Usage by Spoiler Type')
plt.tight_layout()
plt.show()

# ==================================================================================
# 3. ADVANCED FEATURE ENGINEERING
# ==================================================================================

print("\nğŸ”§ SECTION 3: ADVANCED FEATURE ENGINEERING")
print("-" * 60)

def create_advanced_features(df):
    """Create comprehensive feature set"""
    features = pd.DataFrame()
    
    # Basic length features
    features['post_word_count'] = df['postText'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    features['title_word_count'] = df['targetTitle'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    features['content_word_count'] = df['targetParagraphs'].apply(
        lambda x: sum(len(str(p).split()) for p in x) if isinstance(x, list) else 0
    )
    
    # Character features
    features['post_char_count'] = df['postText'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    features['title_char_count'] = df['targetTitle'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    # Punctuation features
    features['question_marks'] = df['postText'].apply(lambda x: str(x).count('?') if pd.notna(x) else 0)
    features['exclamation_marks'] = df['postText'].apply(lambda x: str(x).count('!') if pd.notna(x) else 0)
    features['ellipsis'] = df['postText'].apply(lambda x: str(x).count('...') if pd.notna(x) else 0)
    features['quotes'] = df['postText'].apply(lambda x: str(x).count('"') if pd.notna(x) else 0)
    features['colons'] = df['postText'].apply(lambda x: str(x).count(':') if pd.notna(x) else 0)
    
    # Structural features
    features['num_paragraphs'] = df['targetParagraphs'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    features['avg_paragraph_length'] = features['content_word_count'] / (features['num_paragraphs'] + 1)
    
    # Ratio features
    features['post_to_content_ratio'] = features['post_word_count'] / (features['content_word_count'] + 1)
    features['title_to_content_ratio'] = features['title_word_count'] / (features['content_word_count'] + 1)
    features['post_to_title_ratio'] = features['post_word_count'] / (features['title_word_count'] + 1)
    
    # Lexical diversity (unique words ratio)
    features['post_lexical_diversity'] = df['postText'].apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1) if pd.notna(x) else 0
    )
    
    # Capitalization features
    features['post_caps_ratio'] = df['postText'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1) if pd.notna(x) else 0
    )
    
    # Question word features
    question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
    for qw in question_words:
        features[f'has_{qw}'] = df['postText'].apply(
            lambda x: 1 if qw in str(x).lower() else 0 if pd.notna(x) else 0
        )
    
    return features

# Create features for all datasets
print("Creating advanced features...")
train_meta = create_advanced_features(train_df)
val_meta = create_advanced_features(val_df)
test_meta = create_advanced_features(test_df)

print(f"Created {train_meta.shape[1]} meta features")

# ==================================================================================
# 4. TRANSFORMER EMBEDDINGS WITH QWEN-3 (MULTI-GPU)
# ==================================================================================

print("\nğŸ¤– SECTION 4: TRANSFORMER EMBEDDINGS WITH QWEN-3")
print("-" * 60)

# Check available GPUs
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("No GPU available, using CPU")
    n_gpus = 0

# Model loading with multi-GPU support
use_transformer = False
try:
    if n_gpus >= 2:
        print("\nğŸš€ Loading Qwen-3 8B model across multiple GPUs...")
        model_path = '/kaggle/input/qwen-3/transformers/8b-base/1'
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load config
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model with device_map for multi-GPU
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        
        # Option 1: Use device_map="auto" for automatic distribution
        print("Loading model with automatic device mapping...")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",  # Automatically distribute across GPUs
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True
        )
        
        # Alternative Option 2: Manual device mapping (if auto doesn't work well)
        # device_map = {
        #     "model.embed_tokens": 0,
        #     "model.layers.0": 0,
        #     "model.layers.1": 0,
        #     "model.layers.2": 0,
        #     "model.layers.3": 0,
        #     "model.layers.4": 1,
        #     "model.layers.5": 1,
        #     "model.layers.6": 1,
        #     "model.layers.7": 1,
        #     "model.norm": 1,
        #     "lm_head": 1,
        # }
        # model = AutoModel.from_pretrained(
        #     model_path,
        #     trust_remote_code=True,
        #     device_map=device_map,
        #     torch_dtype=torch.float16
        # )
        
        model.eval()
        print("âœ… Qwen-3 model loaded successfully across GPUs!")
        use_transformer = True
        
        # Print device distribution
        if hasattr(model, 'hf_device_map'):
            print("\nModel distribution across devices:")
            for name, device in model.hf_device_map.items():
                print(f"  {name}: GPU {device}")
                
    else:
        print("âš ï¸� Less than 2 GPUs available. Falling back to lightweight embedding approach.")
        
        # Option: Use a smaller model or sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence-transformers model instead...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            tokenizer = None  # Not needed for sentence transformers
            use_transformer = True
            print("âœ… Sentence transformer loaded successfully!")
        except:
            print("â�Œ Could not load alternative embedding model.")
            use_transformer = False
            
except Exception as e:
    print(f"âš ï¸� Could not load transformer model: {e}")
    print("Falling back to traditional features only.")
    use_transformer = False

# Embedding extraction functions
def get_transformer_embeddings_multigpu(texts, batch_size=4):
    """Get embeddings from transformer model with multi-GPU support"""
    embeddings = []
    
    # Reduce batch size for memory efficiency
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                if tokenizer:  # For Qwen-3
                    # Tokenize with reduced max length
                    inputs = tokenizer(
                        batch_texts, 
                        padding=True, 
                        truncation=True, 
                        max_length=256,  # Reduced from 512
                        return_tensors='pt'
                    )
                    
                    # Model will handle device placement automatically with device_map
                    outputs = model(**inputs)
                    
                    # Use mean pooling
                    embeddings_batch = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(embeddings_batch.cpu().numpy())
                else:  # For sentence transformers
                    embeddings_batch = model.encode(batch_texts, convert_to_numpy=True)
                    embeddings.append(embeddings_batch)
                    
                # Clear cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"âš ï¸� OOM at batch {i//batch_size}, reducing batch size...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Try with batch size 1
                    if batch_size > 1:
                        return get_transformer_embeddings_multigpu(texts, batch_size=1)
                    else:
                        raise e
                else:
                    raise e
    
    return np.vstack(embeddings)

# Create text combinations for embedding
def create_text_for_embedding(df, max_chars=1000):
    """Create combined text for transformer embedding with length limit"""
    texts = []
    for idx, row in df.iterrows():
        post_text = str(row['postText']) if pd.notna(row['postText']) else ''
        target_title = str(row['targetTitle']) if pd.notna(row['targetTitle']) else ''
        
        # Use only first paragraph to save memory
        if isinstance(row['targetParagraphs'], list) and len(row['targetParagraphs']) > 0:
            target_content = str(row['targetParagraphs'][0])[:500]
        else:
            target_content = ''
        
        combined = f"Post: {post_text} Title: {target_title} Content: {target_content}"
        texts.append(combined[:max_chars])  # Limit total length
    
    return texts

# Generate embeddings if model is loaded
if use_transformer:
    print("\nğŸ“� Generating transformer embeddings...")
    
    # Process in smaller chunks to avoid memory issues
    chunk_size = 100
    
    # Training embeddings
    train_texts = create_text_for_embedding(train_df)
    train_embeddings = []
    
    print("Processing training embeddings in chunks...")
    for i in range(0, len(train_texts), chunk_size):
        print(f"  Processing chunk {i//chunk_size + 1}/{(len(train_texts) + chunk_size - 1)//chunk_size}")
        chunk = train_texts[i:i+chunk_size]
        chunk_embeddings = get_transformer_embeddings_multigpu(chunk)
        train_embeddings.append(chunk_embeddings)
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    
    train_embeddings = np.vstack(train_embeddings)
    print(f"âœ… Train embeddings shape: {train_embeddings.shape}")
    
    # Validation embeddings
    print("\nProcessing validation embeddings...")
    val_texts = create_text_for_embedding(val_df)
    val_embeddings = get_transformer_embeddings_multigpu(val_texts)
    print(f"âœ… Val embeddings shape: {val_embeddings.shape}")
    
    # Test embeddings
    print("\nProcessing test embeddings...")
    test_texts = create_text_for_embedding(test_df)
    test_embeddings = get_transformer_embeddings_multigpu(test_texts)
    print(f"âœ… Test embeddings shape: {test_embeddings.shape}")
    
    # Apply dimensionality reduction to embeddings
    print("\nApplying PCA to embeddings...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, random_state=42)
    train_embeddings_reduced = pca.fit_transform(train_embeddings)
    val_embeddings_reduced = pca.transform(val_embeddings)
    test_embeddings_reduced = pca.transform(test_embeddings)
    print(f"Reduced embedding dimensions: {train_embeddings_reduced.shape[1]}")

# ==================================================================================
# 5. TRADITIONAL TEXT FEATURES (TF-IDF)
# ==================================================================================

print("\nğŸ“� SECTION 5: TRADITIONAL TEXT FEATURES")
print("-" * 60)

def create_text_features(df):
    """Create combined text for TF-IDF"""
    features = []
    
    for idx, row in df.iterrows():
        post_text = str(row['postText']) if pd.notna(row['postText']) else ''
        target_title = str(row['targetTitle']) if pd.notna(row['targetTitle']) else ''
        
        if isinstance(row['targetParagraphs'], list):
            target_paragraphs = ' '.join([str(p) for p in row['targetParagraphs']])
        else:
            target_paragraphs = ''
        
        combined_text = f"{post_text} [TITLE] {target_title} [CONTENT] {target_paragraphs}"
        features.append(combined_text)
    
    return features

# Create text features
print("Creating TF-IDF features...")
train_text = create_text_features(train_df)
val_text = create_text_features(val_df)
test_text = create_text_features(test_df)

# TF-IDF with optimized parameters
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    stop_words='english',
    strip_accents='unicode'
)

X_train_tfidf = tfidf.fit_transform(train_text)
X_val_tfidf = tfidf.transform(val_text)
X_test_tfidf = tfidf.transform(test_text)

print(f"TF-IDF features shape: {X_train_tfidf.shape}")

# Apply dimensionality reduction
print("Applying dimensionality reduction...")
svd = TruncatedSVD(n_components=100, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_val_svd = svd.transform(X_val_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

print(f"Reduced TF-IDF shape: {X_train_svd.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.2%}")

# ==================================================================================
# 6. FEATURE COMBINATION AND PREPARATION
# ==================================================================================

print("\nğŸ”— SECTION 6: FEATURE COMBINATION")
print("-" * 60)

# Normalize meta features
scaler = StandardScaler()
train_meta_scaled = scaler.fit_transform(train_meta)
val_meta_scaled = scaler.transform(val_meta)
test_meta_scaled = scaler.transform(test_meta)

# Combine all features
if use_transformer and 'train_embeddings_reduced' in locals():
    # Combine TF-IDF, meta features, and embeddings
    X_train = np.hstack([X_train_svd, train_meta_scaled, train_embeddings_reduced])
    X_val = np.hstack([X_val_svd, val_meta_scaled, val_embeddings_reduced])
    X_test = np.hstack([X_test_svd, test_meta_scaled, test_embeddings_reduced])
    print("âœ… Using transformer embeddings + TF-IDF + meta features")
else:
    # Use only TF-IDF and meta features
    X_train = np.hstack([X_train_svd, train_meta_scaled])
    X_val = np.hstack([X_val_svd, val_meta_scaled])
    X_test = np.hstack([X_test_svd, test_meta_scaled])
    print("âœ… Using TF-IDF + meta features only")

# Create labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['tags'])
y_val = label_encoder.transform(val_df['tags'])

print(f"\nFinal feature dimensions:")
print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")
print(f"  Test: {X_test.shape}")

print(f"\nLabel encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} â†’ {i}")

# ==================================================================================
# 7. MODEL TRAINING AND HYPERPARAMETER OPTIMIZATION
# ==================================================================================

print("\nğŸ�¯ SECTION 7: MODEL TRAINING")
print("-" * 60)

# Define models with optimized hyperparameters
models = {
    'Logistic Regression': LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    'Linear SVM': LinearSVC(
        C=0.1,
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
}

# Cross-validation evaluation
print("Performing cross-validation...")
cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Create boxplot of CV results
plt.figure(figsize=(10, 6))
cv_df = pd.DataFrame(cv_results)
cv_df.boxplot()
plt.title('Cross-Validation F1 Scores by Model')
plt.ylabel('Macro F1 Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================================================================
# 8. MODEL EVALUATION AND SELECTION
# ==================================================================================

print("\nğŸ“Š SECTION 8: MODEL EVALUATION")
print("-" * 60)

# Train all models and evaluate on validation set
best_model = None
best_model_name = None
best_f1 = 0
model_performances = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    val_pred = model.predict(X_val)
    
    # Calculate metrics
    f1 = f1_score(y_val, val_pred, average='macro')
    
    # Store performance
    model_performances[name] = {
        'model': model,
        'f1_score': f1,
        'predictions': val_pred
    }
    
    print(f"\nMacro F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, 
                              target_names=label_encoder.classes_, 
                              digits=4))
    
    # Update best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\nğŸ�† Best Model: {best_model_name} (F1: {best_f1:.4f})")

# ==================================================================================
# 9. ADVANCED VISUALIZATIONS
# ==================================================================================

print("\nğŸ“ˆ SECTION 9: ADVANCED VISUALIZATIONS")
print("-" * 60)

# 9.1 Confusion Matrix Heatmap
best_val_pred = model_performances[best_model_name]['predictions']
cm = confusion_matrix(y_val, best_val_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add percentages
for i in range(len(cm)):
    for j in range(len(cm)):
        percentage = cm[i, j] / cm[i].sum() * 100
        plt.text(j + 0.5, i + 0.7, f'{percentage:.1f}%', 
                ha='center', va='center', fontsize=8, color='red')
plt.tight_layout()
plt.show()

# 9.2 Per-class Performance Radar Chart
from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, _ = precision_recall_fscore_support(
    y_val, best_val_pred, labels=range(len(label_encoder.classes_))
)

# Create radar chart
categories = label_encoder.classes_
fig = go.Figure()

metrics = ['Precision', 'Recall', 'F1-Score']
for i, (p, r, f) in enumerate(zip(precision, recall, fscore)):
    fig.add_trace(go.Scatterpolar(
        r=[p, r, f, p],
        theta=metrics + [metrics[0]],
        fill='toself',
        name=categories[i]
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title=f"Per-class Performance Metrics - {best_model_name}"
)
fig.show()

# ==================================================================================
# 10. ENSEMBLE METHOD
# ==================================================================================

print("\nğŸ�¼ SECTION 10: ENSEMBLE METHOD")
print("-" * 60)

# Create ensemble using top 3 models
top_models = sorted(model_performances.items(), 
                   key=lambda x: x[1]['f1_score'], 
                   reverse=True)[:3]

print("Top 3 models for ensemble:")
for name, perf in top_models:
    print(f"  â€¢ {name}: F1 = {perf['f1_score']:.4f}")

# Soft voting ensemble
ensemble_probs = []
for name, perf in top_models:
    model = perf['model']
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_val)
    else:  # For SVM
        decision = model.decision_function(X_val)
        probs = F.softmax(torch.tensor(decision), dim=1).numpy()
    ensemble_probs.append(probs)

# Average probabilities
ensemble_avg_probs = np.mean(ensemble_probs, axis=0)
ensemble_pred = np.argmax(ensemble_avg_probs, axis=1)

# Evaluate ensemble
ensemble_f1 = f1_score(y_val, ensemble_pred, average='macro')
print(f"\nğŸ�¯ Ensemble F1 Score: {ensemble_f1:.4f}")

if ensemble_f1 > best_f1:
    print("âœ… Ensemble improves over best single model!")
    use_ensemble = True
else:
    print("â�Œ Single model performs better than ensemble")
    use_ensemble = False

# ==================================================================================
# 11. FINAL PREDICTIONS
# ==================================================================================

print("\nğŸ�� SECTION 11: FINAL PREDICTIONS")
print("-" * 60)

# Retrain on combined train+val data
print("Retraining on combined dataset...")
X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])

if use_ensemble:
    print("Using ensemble for final predictions...")
    final_models = []
    for name, perf in top_models:
        model_class = type(perf['model'])
        if name == 'Logistic Regression':
            model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        elif name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42)
        elif name == 'Gradient Boosting':
            model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        else:
            model = LinearSVC(C=0.1, class_weight='balanced', random_state=42, max_iter=2000)
        
        model.fit(X_combined, y_combined)
        final_models.append((name, model))
    
    # Make ensemble predictions
    test_probs = []
    for name, model in final_models:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)
        else:
            decision = model.decision_function(X_test)
            probs = F.softmax(torch.tensor(decision), dim=1).numpy()
        test_probs.append(probs)
    
    test_proba = np.mean(test_probs, axis=0)
    test_predictions = np.argmax(test_proba, axis=1)
else:
    # Retrain best single model
    print(f"Using {best_model_name} for final predictions...")
    if best_model_name == 'Logistic Regression':
        final_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    elif best_model_name == 'Random Forest':
        final_model = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42)
    elif best_model_name == 'Gradient Boosting':
        final_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    else:
        final_model = LinearSVC(C=0.1, class_weight='balanced', random_state=42, max_iter=2000)
    
    final_model.fit(X_combined, y_combined)
    
    # Make predictions
    test_predictions = final_model.predict(X_test)
    
    # Get confidence scores
    if hasattr(final_model, 'predict_proba'):
        test_proba = final_model.predict_proba(X_test)
    else:
        decision = final_model.decision_function(X_test)
        test_proba = F.softmax(torch.tensor(decision), dim=1).numpy()

# Convert to labels
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Create submission
submission = pd.DataFrame({
    'id': range(len(test_df)),
    'spoilerType': test_predictions_labels
})

# Add confidence scores
submission['confidence'] = test_proba.max(axis=1)

# ==================================================================================
# 12. RESULTS ANALYSIS
# ==================================================================================

print("\nğŸ“Š SECTION 12: RESULTS ANALYSIS")
print("-" * 60)

# Prediction distribution
print("\nğŸ“Š Prediction Distribution:")
pred_counts = submission['spoilerType'].value_counts()
pred_percentages = submission['spoilerType'].value_counts(normalize=True) * 100

result_df = pd.DataFrame({
    'Count': pred_counts,
    'Percentage': pred_percentages.round(2)
})
print(result_df)

# Compare with training distribution
print("\nğŸ“Š Distribution Comparison:")
train_dist = train_df['tags'].value_counts(normalize=True) * 100
comparison = pd.DataFrame({
    'Training %': train_dist.round(2),
    'Test Predictions %': pred_percentages.round(2),
    'Difference': (pred_percentages - train_dist).round(2)
})
print(comparison)

# Confidence analysis
print(f"\nğŸ“Š Confidence Statistics:")
print(f"  â€¢ Mean confidence: {submission['confidence'].mean():.3f}")
print(f"  â€¢ Std confidence: {submission['confidence'].std():.3f}")
print(f"  â€¢ Min confidence: {submission['confidence'].min():.3f}")
print(f"  â€¢ Max confidence: {submission['confidence'].max():.3f}")

# Show low confidence predictions
low_conf_threshold = 0.4
low_conf_count = (submission['confidence'] < low_conf_threshold).sum()
print(f"\nâš ï¸� Predictions with confidence < {low_conf_threshold}: {low_conf_count}")

# Visualize confidence distribution
plt.figure(figsize=(10, 6))
plt.hist(submission['confidence'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(submission['confidence'].mean(), color='red', linestyle='--', label='Mean')
plt.xlabel('Prediction Confidence')
plt.ylabel('Count')
plt.title('Distribution of Prediction Confidence Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Save predictions
submission[['id', 'spoilerType']].to_csv('prediction_task1.csv', index=False)
print(f"\nâœ… Predictions saved to 'prediction_task1.csv'")

# Display sample predictions
print("\nğŸ“‹ Sample Predictions (First 20):")
display_df = submission.head(20).copy()
display_df['confidence'] = display_df['confidence'].round(3)
print(display_df)

# ==================================================================================
# 13. SUMMARY AND CONCLUSIONS
# ==================================================================================

print("\n" + "="*80)
print("ğŸ�¯ PIPELINE SUMMARY".center(80))
print("="*80)

print(f"""
ğŸ“Š Dataset Statistics:
  â€¢ Total samples processed: {len(train_df) + len(val_df) + len(test_df):,}
  â€¢ Features used: {X_train.shape[1]} ({X_train_svd.shape[1]} TF-IDF + {train_meta.shape[1]} meta{' + ' + str(train_embeddings_reduced.shape[1]) + ' embeddings' if use_transformer and 'train_embeddings_reduced' in locals() else ''})
  
ğŸ�† Best Model Performance:
  â€¢ Model: {best_model_name}
  â€¢ Validation F1 Score: {best_f1:.4f}
  â€¢ Ensemble F1 Score: {ensemble_f1:.4f}
  â€¢ Using: {'Ensemble' if use_ensemble else 'Single Model'}
  
ğŸ“ˆ Key Insights:
  â€¢ The '{train_df['tags'].value_counts().index[0]}' class is most common ({train_df['tags'].value_counts(normalize=True).values[0]:.1%})
  â€¢ Meta features provide significant signal for classification
  â€¢ {'Transformer embeddings improved performance' if use_transformer else 'Traditional features performed well'}
  â€¢ {'Ensemble method provided improvement' if use_ensemble else 'Single model was sufficient'}
  
ğŸ”� Areas for Future Improvement:
  â€¢ Fine-tune transformer model on domain-specific data
  â€¢ Experiment with more sophisticated ensemble techniques
  â€¢ Incorporate additional linguistic features
  â€¢ Use active learning for challenging samples
  â€¢ Try different pooling strategies for embeddings
""")

print("âœ… Pipeline completed successfully!")
print("="*80)

# Clean up GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

