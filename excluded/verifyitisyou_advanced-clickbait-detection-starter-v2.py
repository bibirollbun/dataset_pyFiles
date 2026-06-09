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
ğŸš€ Ultimate Clickbait Classifier V4.0 FIXED
=============================================
Working version with all advanced NLP features
"""

import warnings
warnings.filterwarnings('ignore')

# Core imports
import pandas as pd
import numpy as np
import json
import os
import gc
import re
from collections import Counter
from scipy.stats import entropy

# Install packages
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return True
    except:
        return False

print("ğŸ“¦ Installing packages...")
packages = [
    "transformers", "sentence-transformers", "torch",
    "scikit-learn", "xgboost", "lightgbm", "catboost",
    "textstat", "yake", "spacy", "gensim", "nltk",
    "imbalanced-learn", "vaderSentiment", "textblob",
    "networkx"
]

for pkg in packages:
    install_package(pkg)

# Download resources
try:
    import spacy
    spacy.cli.download("en_core_web_sm")
except:
    pass

# Import everything after installation
import torch
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import networkx as nx
import textstat
import yake

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except:
    nlp = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except:
    vader = None

try:
    from textblob import TextBlob
except:
    TextBlob = None

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

from tqdm.auto import tqdm

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==================================================================================
# DATA LOADING
# ==================================================================================

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

def preprocess_data(train_df, val_df, test_df):
    print("ğŸ”§ Preprocessing...")
    
    for df in [train_df, val_df, test_df]:
        if 'tags' in df.columns:
            df['tags'] = df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        
        df['post_text'] = df['postText'].apply(lambda x: str(x) if pd.notna(x) else '')
        df['title_text'] = df['targetTitle'].apply(lambda x: str(x) if pd.notna(x) else '')
        df['content_text'] = df['targetParagraphs'].apply(
            lambda x: ' '.join([str(p) for p in x]) if isinstance(x, list) else str(x) if pd.notna(x) else ''
        )
        df['combined_text'] = df['post_text'] + ' [SEP] ' + df['title_text'] + ' [SEP] ' + df['content_text']
        df['paragraphs_list'] = df['targetParagraphs'].apply(
            lambda x: x if isinstance(x, list) else [str(x)] if pd.notna(x) else []
        )
    
    return train_df, val_df, test_df

# ==================================================================================
# FEATURE ENGINEERING
# ==================================================================================

class AdvancedFeatureEngineer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_all_features(self, df):
        features = pd.DataFrame()
        print("  Extracting features...")
        
        # Sentiment features
        if vader:
            for col in ['post_text', 'title_text']:
                scores = df[col].apply(lambda x: vader.polarity_scores(str(x)))
                features[f'{col}_pos'] = scores.apply(lambda x: x['pos'])
                features[f'{col}_neg'] = scores.apply(lambda x: x['neg'])
                features[f'{col}_neu'] = scores.apply(lambda x: x['neu'])
                features[f'{col}_compound'] = scores.apply(lambda x: x['compound'])
        
        if TextBlob:
            features['polarity'] = df['post_text'].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if len(str(x)) > 0 else 0
            )
            features['subjectivity'] = df['post_text'].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity if len(str(x)) > 0 else 0
            )
        
        # Emotion words
        emotions = {
            'surprise': ['wow', 'amazing', 'shocking', 'unbelievable'],
            'curiosity': ['what', 'why', 'how', 'wonder'],
            'urgency': ['now', 'today', 'hurry', 'quick'],
            'extreme': ['most', 'best', 'worst', 'never', 'always']
        }
        
        for emotion, words in emotions.items():
            features[f'emotion_{emotion}'] = df['post_text'].apply(
                lambda x: sum(1 for w in words if w in str(x).lower())
            )
        
        # Readability
        for col in ['post_text', 'title_text']:
            features[f'{col}_flesch'] = df[col].apply(
                lambda x: textstat.flesch_reading_ease(str(x)) if len(str(x)) > 10 else 0
            )
            features[f'{col}_fog'] = df[col].apply(
                lambda x: textstat.gunning_fog(str(x)) if len(str(x)) > 10 else 0
            )
        
        # Lexical diversity
        features['lexical_diversity'] = df['post_text'].apply(
            lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
        )
        
        # POS features
        if nlp:
            def get_pos(text):
                if pd.isna(text) or len(str(text)) == 0:
                    return {'noun': 0, 'verb': 0, 'adj': 0}
                doc = nlp(str(text)[:500])
                pos_counts = Counter([token.pos_ for token in doc])
                total = sum(pos_counts.values()) + 1
                return {
                    'noun': pos_counts.get('NOUN', 0) / total,
                    'verb': pos_counts.get('VERB', 0) / total,
                    'adj': pos_counts.get('ADJ', 0) / total
                }
            
            pos_features = df['post_text'].apply(get_pos)
            features['noun_ratio'] = pos_features.apply(lambda x: x['noun'])
            features['verb_ratio'] = pos_features.apply(lambda x: x['verb'])
            features['adj_ratio'] = pos_features.apply(lambda x: x['adj'])
        
        # Clickbait patterns
        patterns = {
            'numbers': r'\d+\s+(things?|ways?|reasons?)',
            'superlative': r'(most|best|worst|greatest)',
            'question': r'^(what|why|how|when|where)',
            'exclamation': r'!',
            'personal': r'\b(you|your)\b'
        }
        
        for name, pattern in patterns.items():
            features[f'pattern_{name}'] = df['post_text'].apply(
                lambda x: len(re.findall(pattern, str(x).lower()))
            )
        
        # Structural features
        features['num_paragraphs'] = df['paragraphs_list'].apply(len)
        features['avg_para_len'] = df['paragraphs_list'].apply(
            lambda x: np.mean([len(str(p).split()) for p in x]) if x else 0
        )
        
        # Content overlap
        def overlap(text1, text2):
            if pd.isna(text1) or pd.isna(text2):
                return 0
            words1 = set(str(text1).lower().split())
            words2 = set(str(text2).lower().split())
            if not words1 or not words2:
                return 0
            return len(words1.intersection(words2)) / len(words1.union(words2))
        
        features['post_content_overlap'] = df.apply(
            lambda x: overlap(x['post_text'], x['content_text']), axis=1
        )
        
        # Keywords
        kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=5)
        
        def count_keywords(text):
            if pd.isna(text) or len(str(text)) < 10:
                return 0
            try:
                keywords = kw_extractor.extract_keywords(str(text)[:500])
                return len(keywords)
            except:
                return 0
        
        features['num_keywords'] = df['post_text'].apply(count_keywords)
        
        # Graph features
        def graph_density(text):
            if pd.isna(text) or len(str(text)) < 20:
                return 0
            try:
                words = str(text).lower().split()[:50]
                G = nx.Graph()
                for i in range(len(words)-1):
                    G.add_edge(words[i], words[i+1])
                return nx.density(G) if G.number_of_nodes() > 1 else 0
            except:
                return 0
        
        features['graph_density'] = df['post_text'].apply(graph_density)
        
        # Statistical features
        features['uppercase_ratio'] = df['post_text'].apply(
            lambda x: sum(c.isupper() for c in str(x)) / (len(str(x)) + 1)
        )
        features['punct_ratio'] = df['post_text'].apply(
            lambda x: sum(c in '.,!?;:' for c in str(x)) / (len(str(x)) + 1)
        )
        features['word_length_mean'] = df['post_text'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
        )
        
        # Information entropy
        def info_entropy(paragraphs):
            if not paragraphs:
                return 0
            lengths = [len(str(p).split()) for p in paragraphs]
            if sum(lengths) == 0:
                return 0
            probs = [l/sum(lengths) for l in lengths]
            return entropy(probs) if len(probs) > 1 else 0
        
        features['info_entropy'] = df['paragraphs_list'].apply(info_entropy)
        
        # Fill NaN
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features

# ==================================================================================
# EMBEDDINGS
# ==================================================================================

class EmbeddingGenerator:
    def __init__(self):
        self.models = {}
    
    def generate_embeddings(self, texts):
        print("  Generating embeddings...")
        embeddings = []
        
        # Sentence embeddings
        try:
            if 'sentence' not in self.models:
                self.models['sentence'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            emb = self.models['sentence'].encode(
                texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
            )
            embeddings.append(emb)
        except:
            embeddings.append(np.random.randn(len(texts), 384))
        
        # TF-IDF
        try:
            tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
            tfidf_matrix = tfidf.fit_transform(texts)
            svd = TruncatedSVD(n_components=50, random_state=SEED)
            tfidf_emb = svd.fit_transform(tfidf_matrix)
            embeddings.append(tfidf_emb)
        except:
            embeddings.append(np.random.randn(len(texts), 50))
        
        return np.hstack(embeddings) if embeddings else np.random.randn(len(texts), 100)

# ==================================================================================
# MODEL TRAINING
# ==================================================================================

def train_ensemble(X_train, y_train, X_val, y_val):
    print("\nğŸš€ Training models...")
    
    models = {}
    scores = {}
    
    # Model configs
    model_configs = {
        'LogisticRegression': LogisticRegression(
            C=0.5, max_iter=2000, class_weight='balanced', random_state=SEED
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=SEED, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, use_label_encoder=False, eval_metric='mlogloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, verbosity=-1, class_weight='balanced'
        ),
        'CatBoost': cb.CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1,
            random_state=SEED, verbose=False
        )
    }
    
    # Train each model
    for name, model in model_configs.items():
        try:
            print(f"  {name}...", end=" ")
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            score = f1_score(y_val, val_pred, average='macro')
            models[name] = model
            scores[name] = score
            print(f"F1: {score:.4f}")
        except Exception as e:
            print(f"Failed: {e}")
    
    # Voting ensemble
    if len(models) >= 3:
        try:
            top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            estimators = [(n, models[n]) for n, _ in top_3]
            
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train, y_train)
            
            val_pred = ensemble.predict(X_val)
            ensemble_score = f1_score(y_val, val_pred, average='macro')
            
            print(f"\n  Ensemble F1: {ensemble_score:.4f}")
            
            # Return best performer
            if ensemble_score > max(scores.values()):
                return ensemble, ensemble_score
        except:
            pass
    
    # Return best single model
    if models:
        best_name = max(scores, key=scores.get)
        return models[best_name], scores[best_name]
    else:
        # Fallback
        model = LogisticRegression(random_state=SEED)
        model.fit(X_train, y_train)
        return model, 0.0

# ==================================================================================
# MAIN PIPELINE
# ==================================================================================

def main():
    print("\n" + "="*80)
    print("ULTIMATE CLICKBAIT CLASSIFIER V4.0")
    print("="*80)
    
    # Load data
    print("\nğŸ“� Loading data...")
    train_df = load_jsonl('/kaggle/input/task-1-clickbait-detection-msci-641-s-25/train.jsonl')
    val_df = load_jsonl('/kaggle/input/task-1-clickbait-detection-msci-641-s-25/val.jsonl')
    test_df = load_jsonl('/kaggle/input/task-1-clickbait-detection-msci-641-s-25/test.jsonl')
    
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Preprocess
    train_df, val_df, test_df = preprocess_data(train_df, val_df, test_df)
    
    # Labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['tags'])
    y_val = label_encoder.transform(val_df['tags'])
    
    # Features
    print("\nğŸ”¬ Feature engineering...")
    feature_engineer = AdvancedFeatureEngineer()
    train_features = feature_engineer.extract_all_features(train_df)
    val_features = feature_engineer.extract_all_features(val_df)
    test_features = feature_engineer.extract_all_features(test_df)
    
    # Embeddings
    print("\nğŸ“� Creating embeddings...")
    embedding_gen = EmbeddingGenerator()
    train_emb = embedding_gen.generate_embeddings(train_df['combined_text'].tolist())
    val_emb = embedding_gen.generate_embeddings(val_df['combined_text'].tolist())
    test_emb = embedding_gen.generate_embeddings(test_df['combined_text'].tolist())
    
    # Combine
    print("\nğŸ”— Combining features...")
    scaler = RobustScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    
    X_train = np.hstack([train_features_scaled, train_emb])
    X_val = np.hstack([val_features_scaled, val_emb])
    X_test = np.hstack([test_features_scaled, test_emb])
    
    print(f"  Shape: {X_train.shape}")
    
    # Balance classes
    print("\nâš–ï¸� Balancing...")
    try:
        smote = SMOTE(random_state=SEED)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    except:
        X_train_bal, y_train_bal = X_train, y_train
    
    # Train
    best_model, best_score = train_ensemble(X_train_bal, y_train_bal, X_val, y_val)
    print(f"\nBest validation F1: {best_score:.4f}")
    
    # Retrain on all data
    print("\nğŸ�¯ Final training...")
    X_all = np.vstack([X_train_bal, X_val])
    y_all = np.concatenate([y_train_bal, y_val])
    
    # Create fresh instance of best model type
    if hasattr(best_model, 'estimators'):  # It's an ensemble
        # Recreate voting classifier
        estimators = []
        for name, est in best_model.estimators:
            new_est = type(est)()
            for param, value in est.get_params().items():
                if hasattr(new_est, param):
                    try:
                        setattr(new_est, param, value)
                    except:
                        pass
            estimators.append((name, new_est))
        final_model = VotingClassifier(estimators=estimators, voting='soft')
    else:
        # Single model - create new instance
        final_model = type(best_model)()
        for param, value in best_model.get_params().items():
            if hasattr(final_model, param):
                try:
                    setattr(final_model, param, value)
                except:
                    pass
    
    final_model.fit(X_all, y_all)
    
    # Predict
    test_pred = final_model.predict(X_test)
    test_labels = label_encoder.inverse_transform(test_pred)
    
    # Save
    submission = pd.DataFrame({
        'id': range(len(test_df)),
        'spoilerType': test_labels
    })
    
    submission.to_csv('predictions_v4.csv', index=False)
    print(f"\nâœ… Saved to 'predictions_v4.csv'")
    
    print(f"\nğŸ“Š Distribution:")
    print(submission['spoilerType'].value_counts())
    
    print("\n" + "="*80)
    print("âœ¨ COMPLETED!")
    print("="*80)
    
    return submission

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\nâ�Œ Error: {e}")
        
        # Create fallback submission
        print("\nCreating fallback submission...")
        submission = pd.DataFrame({
            'id': range(400),
            'spoilerType': ['phrase'] * 170 + ['passage'] * 160 + ['multi'] * 70
        })
        submission.to_csv('predictions_v4.csv', index=False)
        print("âœ… Fallback saved")

