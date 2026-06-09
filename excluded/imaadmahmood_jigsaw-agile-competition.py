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


#!/usr/bin/env python3
"""
Jigsaw Agile Community Rules Classification - Optimized Offline Solution
High-performance solution designed for Kaggle's offline environment
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ===============================
# CONFIGURATION
# ===============================
class Config:
    """Optimized configuration for offline execution"""
    RANDOM_SEED = 777
    N_ENSEMBLE = 3  # Reduced for faster execution
    CV_FOLDS = 3    # Reduced for faster execution
    MAX_TFIDF_FEATURES = 5000  # Reduced for memory efficiency
    NGRAM_RANGE = (1, 2)       # Reduced for speed
    
    # Weights for hybrid ensemble
    SIMILARITY_WEIGHT = 0.4
    ML_WEIGHT = 0.5
    RULE_WEIGHT = 0.1

# ===============================
# OPTIMIZED TEXT PROCESSOR
# ===============================
class FastTextProcessor:
    """Memory and speed optimized text processing"""
    
    def __init__(self):
        # Pre-compiled regex patterns for speed
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.username_pattern = re.compile(r'/u/[a-zA-Z0-9_-]+')
        self.subreddit_pattern = re.compile(r'/r/[a-zA-Z0-9_-]+')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Violation keywords (most common patterns only)
        self.violation_indicators = {
            'spam': ['buy', 'sell', 'click', 'visit', 'subscribe', 'follow', 'deal', 'offer'],
            'harassment': ['stupid', 'idiot', 'moron', 'loser', 'pathetic', 'kill', 'die'],
            'profanity': ['fuck', 'shit', 'damn', 'ass', 'bitch', 'bastard']
        }
    
    def clean_text(self, text):
        """Fast text cleaning"""
        if pd.isna(text) or not str(text).strip():
            return ""
        
        text = str(text).lower()
        text = self.url_pattern.sub(' ', text)
        text = self.username_pattern.sub(' ', text)
        text = self.subreddit_pattern.sub(' ', text)
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()
    
    def extract_basic_features(self, text):
        """Extract essential features only"""
        if not text:
            return np.zeros(8)
        
        features = []
        text_lower = text.lower()
        
        # Basic counts
        features.append(len(text))                    # 0: length
        features.append(len(text.split()))           # 1: word_count
        features.append(text.count('!'))             # 2: exclamations
        features.append(text.count('?'))             # 3: questions
        
        # Ratios
        features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))  # 4: caps_ratio
        features.append(sum(1 for c in text if c in string.punctuation) / max(len(text), 1))  # 5: punct_ratio
        
        # Violation indicators
        spam_score = sum(1 for word in self.violation_indicators['spam'] if word in text_lower)
        harassment_score = sum(1 for word in self.violation_indicators['harassment'] if word in text_lower)
        
        features.append(min(spam_score, 3))          # 6: spam_score (capped)
        features.append(min(harassment_score, 3))    # 7: harassment_score (capped)
        
        return np.array(features)

# ===============================
# FAST SIMILARITY ENGINE
# ===============================
class FastSimilarityEngine:
    """Optimized similarity computation without external dependencies"""
    
    def compute_word_similarity(self, text1, text2):
        """Fast word-overlap similarity"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_char_similarity(self, text1, text2):
        """Fast character n-gram similarity"""
        if not text1 or not text2 or len(text1) < 3 or len(text2) < 3:
            return 0.0
        
        # 3-gram similarity
        ngrams1 = set(text1.lower()[i:i+3] for i in range(len(text1)-2))
        ngrams2 = set(text2.lower()[i:i+3] for i in range(len(text2)-2))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_combined_similarity(self, text1, text2):
        """Combined similarity score"""
        word_sim = self.compute_word_similarity(text1, text2)
        char_sim = self.compute_char_similarity(text1, text2)
        
        # Weighted combination
        return 0.7 * word_sim + 0.3 * char_sim
    
    def get_example_similarities(self, comment, pos_examples, neg_examples):
        """Get similarities to all examples"""
        pos_sims = [self.compute_combined_similarity(comment, ex) for ex in pos_examples if ex]
        neg_sims = [self.compute_combined_similarity(comment, ex) for ex in neg_examples if ex]
        
        avg_pos = np.mean(pos_sims) if pos_sims else 0.0
        avg_neg = np.mean(neg_sims) if neg_sims else 0.0
        max_pos = np.max(pos_sims) if pos_sims else 0.0
        max_neg = np.max(neg_sims) if neg_sims else 0.0
        
        return np.array([avg_pos, avg_neg, max_pos, max_neg, avg_pos - avg_neg])

# ===============================
# STREAMLINED ML PIPELINE
# ===============================
class StreamlinedMLPipeline:
    """Fast, memory-efficient ML pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.text_processor = FastTextProcessor()
        self.similarity_engine = FastSimilarityEngine()
        self.tfidf = None
        self.scaler = None
        self.ensemble = None
    
    def extract_features(self, df, is_training=True):
        """Fast feature extraction"""
        n_samples = len(df)
        
        # Initialize feature matrices
        basic_features = np.zeros((n_samples, 8))
        similarity_features = np.zeros((n_samples, 5))
        
        # Process each row
        for i, (_, row) in enumerate(df.iterrows()):
            # Basic text features
            clean_comment = self.text_processor.clean_text(row['body'])
            basic_features[i] = self.text_processor.extract_basic_features(clean_comment)
            
            # Similarity features
            pos_examples = [row.get('positive_example_1', ''), row.get('positive_example_2', '')]
            neg_examples = [row.get('negative_example_1', ''), row.get('negative_example_2', '')]
            similarity_features[i] = self.similarity_engine.get_example_similarities(
                clean_comment, pos_examples, neg_examples
            )
        
        # Combine text for TF-IDF
        combined_texts = []
        for _, row in df.iterrows():
            parts = [
                self.text_processor.clean_text(row['body']),
                self.text_processor.clean_text(row['rule']),
                self.text_processor.clean_text(row.get('positive_example_1', '')),
                self.text_processor.clean_text(row.get('positive_example_2', ''))
            ]
            combined_texts.append(' '.join(filter(None, parts)))
        
        # TF-IDF features
        if is_training:
            self.tfidf = TfidfVectorizer(
                max_features=self.config.MAX_TFIDF_FEATURES,
                ngram_range=self.config.NGRAM_RANGE,
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            tfidf_features = self.tfidf.fit_transform(combined_texts).toarray()
        else:
            tfidf_features = self.tfidf.transform(combined_texts).toarray()
        
        # Combine all features
        all_features = np.hstack([basic_features, similarity_features, tfidf_features])
        
        # Scale features
        if is_training:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(all_features)
        else:
            scaled_features = self.scaler.transform(all_features)
        
        return scaled_features
    
    def train_ensemble(self, X, y):
        """Train optimized ensemble"""
        models = [
            ('lr_l1', LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=self.config.RANDOM_SEED)),
            ('lr_l2', LogisticRegression(penalty='l2', C=0.5, random_state=self.config.RANDOM_SEED, max_iter=500)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.config.RANDOM_SEED, n_jobs=-1))
        ]
        
        self.ensemble = VotingClassifier(estimators=models, voting='soft')
        self.ensemble.fit(X, y)
        
        # Quick CV score (single fold for speed)
        skf = StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, random_state=self.config.RANDOM_SEED)
        cv_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            temp_ensemble = VotingClassifier(estimators=models, voting='soft')
            temp_ensemble.fit(X_train_fold, y_train_fold)
            
            pred_proba = temp_ensemble.predict_proba(X_val_fold)[:, 1]
            cv_scores.append(roc_auc_score(y_val_fold, pred_proba))
        
        return np.mean(cv_scores)
    
    def predict_proba(self, X):
        """Generate predictions"""
        return self.ensemble.predict_proba(X)[:, 1]

# ===============================
# HYBRID PREDICTOR
# ===============================
class HybridPredictor:
    """Combines all approaches efficiently"""
    
    def __init__(self, config):
        self.config = config
        self.ml_pipeline = StreamlinedMLPipeline(config)
        self.similarity_engine = FastSimilarityEngine()
        self.text_processor = FastTextProcessor()
        self.trained = False
    
    def fit(self, df_train):
        """Train the hybrid model"""
        # Extract features and train ML pipeline
        X = self.ml_pipeline.extract_features(df_train, is_training=True)
        y = df_train['rule_violation'].values
        
        cv_score = self.ml_pipeline.train_ensemble(X, y)
        self.trained = True
        
        return cv_score
    
    def predict_batch(self, df_test):
        """Generate predictions for test set"""
        n_samples = len(df_test)
        all_predictions = []
        
        for ensemble_idx in range(self.config.N_ENSEMBLE):
            predictions = []
            np.random.seed(self.config.RANDOM_SEED + ensemble_idx * 123)
            
            # Get ML predictions
            if self.trained:
                X_test = self.ml_pipeline.extract_features(df_test, is_training=False)
                ml_predictions = self.ml_pipeline.predict_proba(X_test)
            else:
                ml_predictions = np.full(n_samples, 0.5)
            
            # Get similarity predictions
            similarity_predictions = []
            for _, row in df_test.iterrows():
                comment = self.text_processor.clean_text(row['body'])
                pos_examples = [row.get('positive_example_1', ''), row.get('positive_example_2', '')]
                neg_examples = [row.get('negative_example_1', ''), row.get('negative_example_2', '')]
                
                sim_features = self.similarity_engine.get_example_similarities(comment, pos_examples, neg_examples)
                sim_score = sim_features[4]  # pos - neg difference
                sim_prob = 1 / (1 + np.exp(-5 * sim_score))  # sigmoid
                similarity_predictions.append(sim_prob)
            
            similarity_predictions = np.array(similarity_predictions)
            
            # Combine predictions
            combined = (
                self.config.ML_WEIGHT * ml_predictions +
                self.config.SIMILARITY_WEIGHT * similarity_predictions +
                self.config.RULE_WEIGHT * 0.5  # neutral rule weight
            )
            
            # Add small noise for ensemble diversity
            noise = np.random.normal(0, 0.01, len(combined))
            combined = np.clip(combined + noise, 0.01, 0.99)
            
            all_predictions.append(combined)
        
        # Ensemble combination
        final_predictions = np.mean(all_predictions, axis=0)
        return final_predictions

# ===============================
# POST-PROCESSING
# ===============================
class FastPostProcessor:
    """Optimized post-processing"""
    
    @staticmethod
    def apply_exact_matches(df_test, df_train=None):
        """Apply exact match overrides"""
        if df_train is None:
            return df_test
        
        # Build lookup dictionary
        lookup = {}
        
        for df in [df_train, df_test]:
            for _, row in df.iterrows():
                key_base = (row['rule'].strip(), row['subreddit'].strip())
                
                lookup[(*key_base, row['positive_example_1'].strip())] = 0.95
                lookup[(*key_base, row['positive_example_2'].strip())] = 0.95
                lookup[(*key_base, row['negative_example_1'].strip())] = 0.05
                lookup[(*key_base, row['negative_example_2'].strip())] = 0.05
        
        # Apply exact matches
        exact_matches = 0
        for idx, row in df_test.iterrows():
            key = (row['rule'].strip(), row['subreddit'].strip(), row['body'].strip())
            if key in lookup:
                df_test.loc[idx, 'prediction'] = lookup[key]
                exact_matches += 1
        
        return df_test, exact_matches
    
    @staticmethod
    def calibrate_predictions(predictions):
        """Simple calibration"""
        predictions = np.clip(predictions, 0.001, 0.999)
        # Light regularization toward center
        return predictions * 0.9 + 0.05

# ===============================
# MAIN PIPELINE
# ===============================
def main():
    """Streamlined main execution"""
    config = Config()
    
    # Load data
    try:
        df_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
        df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
        has_train = True
    except:
        df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
        df_train = None
        has_train = False
    
    # Initialize predictor
    predictor = HybridPredictor(config)
    
    # Train if data available
    if has_train:
        cv_score = predictor.fit(df_train)
    
    # Generate predictions
    predictions = predictor.predict_batch(df_test)
    df_test['prediction'] = predictions
    
    # Post-processing
    post_processor = FastPostProcessor()
    
    if has_train:
        df_test, exact_matches = post_processor.apply_exact_matches(df_test, df_train)
    
    # Calibrate
    df_test['prediction'] = post_processor.calibrate_predictions(df_test['prediction'])
    
    # Create submission
    submission = pd.DataFrame({
        'row_id': df_test['row_id'],
        'rule_violation': df_test['prediction']
    })
    
    # Save
    submission.to_csv("submission.csv", index=False)
    
    return submission

# ===============================
# EXECUTION
# ===============================
if __name__ == "__main__":
    try:
        np.random.seed(777)
        submission = main()
        # Single success message
        print(f"Success! Generated {len(submission)} predictions")
    except Exception as e:
        # Fallback
        print(f"Error: {e}")
        try:
            test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
            fallback = pd.DataFrame({
                'row_id': test_df['row_id'],
                'rule_violation': [0.5] * len(test_df)
            })
            fallback.to_csv("submission.csv", index=False)
            print("Fallback submission created")
        except:
            print("Critical failure")
            raise

