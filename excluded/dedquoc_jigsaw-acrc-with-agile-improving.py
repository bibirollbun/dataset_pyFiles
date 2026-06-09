# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

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
    np.random.seed(Config.RANDOM_SEED)  # Use centralized config
    submission = None

    try:
        print("ğŸš€ Starting pipeline execution...")
        submission = main()
        
        # Final validation
        assert submission is not None, "Submission DataFrame is None"
        assert 'row_id' in submission.columns, "Missing 'row_id' column"
        assert 'rule_violation' in submission.columns, "Missing 'rule_violation' column"
        assert len(submission) > 0, "Submission is empty"
        assert (submission['rule_violation'] >= 0.0).all() and (submission['rule_violation'] <= 1.0).all(), \
               "Predictions must be in [0, 1]"

        print(f"âœ… Success! Generated {len(submission)} predictions")
        
    except Exception as e:
        import traceback
        print(f"â�Œ Error during execution: {e}")
        traceback.print_exc()
        
        # Fallback: neutral predictions
        try:
            print("ğŸ”„ Generating fallback submission with neutral probabilities (0.5)...")
            test_path = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
            test_df = pd.read_csv(test_path)
            
            fallback_submission = pd.DataFrame({
                'row_id': test_df['row_id'],
                'rule_violation': [0.5] * len(test_df)
            })
            fallback_submission.to_csv("submission.csv", index=False)
            print("âœ… Fallback submission created successfully.")
            
        except Exception as fallback_error:
            print(f"ğŸ’¥ Critical failure in fallback: {fallback_error}")
            raise  # Re-raise only if we can't recover at all   
#!/usr/bin/env python3
"""
Jigsaw Agile Community Rules Classification - Optimized Offline Solution
High-performance hybrid model combining:
- TF-IDF + handcrafted features
- Similarity-based scoring
- Rule-based post-processing
- Ensemble prediction with fallback

Designed for Kaggle's offline environment with robust error handling.
"""

# ===============================
# IMPORTS & CONFIGURATION
# ===============================

import os
import re
import string
import warnings
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# External libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import sparse
from scipy.sparse import hstack as sparse_hstack


# ===============================
# CONFIGURATION
# ===============================

class Config:
    """Optimized configuration for offline execution."""
    
    RANDOM_SEED: int = 777
    N_ENSEMBLE: int = 3                    # Number of ensemble models
    CV_FOLDS: int = 3                      # Number of CV folds
    MAX_TFIDF_FEATURES: int = 50_000       # TF-IDF max features
    NGRAM_RANGE: Tuple[int, int] = (1, 2)  # Unigrams + bigrams
    MIN_DF: int = 2                        # Ignore terms in <2 docs
    MAX_DF: float = 0.95                   # Ignore terms in >95% of docs

    # Ensemble weights
    SIMILARITY_WEIGHT: float = 0.4
    ML_WEIGHT: float = 0.5
    RULE_WEIGHT: float = 0.1


# ===============================
# OPTIMIZED TEXT PROCESSOR
# ===============================

class FastTextProcessor:
    """
    Memory and speed-optimized text processor for Jigsaw toxicity classification.
    
    Features:
        - URL, username (/u/...), subreddit (/r/...) removal
        - Fast cleaning via pre-compiled regex
        - Lightweight feature extraction (length, punctuation, violation keywords)
    """

    def __init__(self):
        # Pre-compiled regex patterns for performance
        self._url_pattern = re.compile(r'http[s]?://\S+|www\.\S+', flags=re.IGNORECASE)
        self._username_pattern = re.compile(r'/u/[a-zA-Z0-9_-]+')
        self._subreddit_pattern = re.compile(r'/r/[a-zA-Z0-9_-]+')
        self._whitespace_pattern = re.compile(r'\s+')

        # Violation keywords (common toxic patterns)
        self._violation_keywords: Dict[str, List[str]] = {
            'spam': [
                'buy', 'sell', 'click', 'visit', 'subscribe', 'follow',
                'deal', 'offer', 'free', 'win', 'prize', 'money'
            ],
            'harassment': [
                'stupid', 'idiot', 'moron', 'loser', 'pathetic',
                'kill', 'die', 'hate', 'worthless', 'trash'
            ],
            'profanity': [
                'fuck', 'shit', 'damn', 'ass', 'bitch', 'bastard',
                'crap', 'dick', 'piss', 'hell'
            ]
        }

    def clean_text(self, text: Optional[str]) -> str:
        """
        Fast text cleaning: removes URLs, usernames, subreddits, extra whitespace.
        
        Args:
            text (str or None): Input text
            
        Returns:
            str: Cleaned, lowercased text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""

        text = text.lower()
        text = self._url_pattern.sub(' ', text)
        text = self._username_pattern.sub(' ', text)
        text = self._subreddit_pattern.sub(' ', text)
        text = self._whitespace_pattern.sub(' ', text)
        return text.strip()

    #def extract_basic_features(self, text: str) ->      
    def extract_basic_features(self, text: str) -> np.ndarray:
        """
        Extract lightweight numerical features from text.
        
        Args:
            text (str): Cleaned input text
            
        Returns:
            np.ndarray: Array of 8 handcrafted features
        """
        if not text:
            return np.zeros(8, dtype=np.float32)

        features = []
        text_lower = text.lower()

        # Basic counts
        features.append(len(text))                    # 0: length
        features.append(len(text.split()))           # 1: word_count
        features.append(text.count('!'))             # 2: exclamations
        features.append(text.count('?'))             # 3: questions

        # Ratios
        n_chars = len(text)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(n_chars, 1)
        punct_ratio = sum(1 for c in text if c in string.punctuation) / max(n_chars, 1)

        features.append(caps_ratio)                  # 4: caps_ratio
        features.append(punct_ratio)                 # 5: punct_ratio

        # Violation indicators (capped to reduce outlier impact)
        spam_score = sum(1 for word in self._violation_keywords['spam'] if f' {word} ' in text_lower)
        harassment_score = sum(1 for word in self._violation_keywords['harassment'] if f' {word} ' in text_lower)

        features.append(min(spam_score, 3))          # 6: spam_score
        features.append(min(harassment_score, 3))    # 7: harassment_score

        return np.array(features, dtype=np.float32)


# ===============================
# FAST SIMILARITY ENGINE
# ===============================

class FastSimilarityEngine:
    """
    High-performance similarity computation using lightweight n-gram and set-based methods.
    
    Designed for offline use without external NLP libraries.
    Combines word-level and character-level similarity for robust matching.
    """

    def compute_word_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between word sets.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Jaccard similarity score in [0, 1]
        """
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def compute_char_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Compute Jaccard similarity on character n-grams (default: 3-grams).
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            n (int): Character n-gram size (default=3)
            
        Returns:
            float: Character n-gram similarity in [0, 1]
        """
        if not text1 or not text2 or len(text1) < n or len(text2) < n:
            return 0.0

        def get_ngrams(s: str, n_size: int) -> set:
            return {s[i:i + n_size] for i in range(len(s) - n_size + 1)}

        ngrams1 = get_ngrams(text1.lower(), n)
        ngrams2 = get_ngrams(text2.lower(), n)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2

        return len(intersection) / len(union)

    
    def compute_combined_similarity(self, text1, text2):
        """Combined similarity score"""
        word_sim = self.compute_word_similarity(text1, text2)
        char_sim = self.compute_char_similarity(text1, text2)
        
        # Weighted combination
        return 0.7 * word_sim + 0.3 * char_sim


    def get_example_similarities(
        self, 
        comment: str, 
        pos_examples: List[str], 
        neg_examples: List[str]
    ) -> np.ndarray:
        """
        Compute similarity features between a comment and positive/negative example sets.
        
        Features returned:
            - Mean similarity to positive examples
            - Mean similarity to negative examples
            - Max similarity to positive examples
            - Max similarity to negative examples
            - Difference: avg_pos - avg_neg
        
        Args:
            comment (str): Input comment to evaluate
            pos_examples (List[str]): List of positive (toxic) example texts
            neg_examples (List[str]): List of negative (clean) example texts
            
        Returns:
            np.ndarray: Array of 5 similarity-based features
        """
        # Filter valid non-empty examples
        pos_examples = [ex for ex in pos_examples if ex and isinstance(ex, str)]
        neg_examples = [ex for ex in neg_examples if ex and isinstance(ex, str)]

        # Compute similarities
        pos_sims = [self.compute_combined_similarity(comment, ex) for ex in pos_examples]
        neg_sims = [self.compute_combined_similarity(comment, ex) for ex in neg_examples]

        # Aggregate stats
        avg_pos = float(np.mean(pos_sims)) if pos_sims else 0.0
        avg_neg = float(np.mean(neg_sims)) if neg_sims else 0.0
        max_pos = float(np.max(pos_sims)) if pos_sims else 0.0
        max_neg = float(np.max(neg_sims)) if neg_sims else 0.0
        diff = avg_pos - avg_neg

        return np.array([avg_pos, avg_neg, max_pos, max_neg, diff], dtype=np.float32)


# ===============================
# STREAMLINED ML PIPELINE
# ===============================

class StreamlinedMLPipeline:
    """
    Fast, memory-efficient ML pipeline for Jigsaw toxicity classification.
    
    Combines:
        - Handcrafted text features
        - Example-based similarity features
        - TF-IDF features (on combined rule + comment + examples)
        - Scaled ensemble learning with Logistic Regression and Random Forest
    """

    def __init__(self, config):
        self.config = config
        self.text_processor = FastTextProcessor()
        self.similarity_engine = FastSimilarityEngine()
        self.tfidf: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.ensemble = None

    def extract_features(self, df: pd.DataFrame, is_training: bool = True) -> sparse.csr_matrix:
        """
        Extract and combine multiple feature types efficiently using sparse matrices.
        
        Args:
            df (pd.DataFrame): Input data with 'body', 'rule', and optional example columns
            is_training (bool): Whether this is training (fit TF-IDF/scaler) or inference
            
        Returns:
            scipy.sparse.csr_matrix: Combined sparse feature matrix
        """
        n_samples = len(df)

        # === 1. Basic Handcrafted Features (Dense) ===
        basic_features = np.zeros((n_samples, 8), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            clean_comment = self.text_processor.clean_text(row['body'])
            basic_features[i] = self.text_processor.extract_basic_features(clean_comment)

        # === 2. Similarity Features (Dense) ===
        similarity_features = np.zeros((n_samples, 5), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            clean_comment = self.text_processor.clean_text(row['body'])
            pos_examples = [
                self.text_processor.clean_text(row.get('positive_example_1', '')),
                self.text_processor.clean_text(row.get('positive_example_2', ''))
            ]
            neg_examples = [
                self.text_processor.clean_text(row.get('negative_example_1', '')),
                self.text_processor.clean_text(row.get('negative_example_2', ''))
            ]
            similarity_features[i] = self.similarity_engine.get_example_similarities(
                clean_comment, pos_examples, neg_examples
            )

        # === 3. TF-IDF Features (Sparse) ===
        # Combine rule, comment, and positive examples for richer context
           
        combined_texts = []
        for _, row in df.iterrows():
            parts = [
                self.text_processor.clean_text(row['body']),
                self.text_processor.clean_text(row['rule']),
                self.text_processor.clean_text(row.get('positive_example_1', '')),
                self.text_processor.clean_text(row.get('positive_example_2', ''))
            ]
            combined_texts.append(' '.join(filter(None, parts)))

        if is_training:
            if self.tfidf is None:
                self.tfidf = TfidfVectorizer(
                    max_features=self.config.MAX_TFIDF_FEATURES,
                    ngram_range=self.config.NGRAM_RANGE,
                    min_df=self.config.MIN_DF,
                    max_df=self.config.MAX_DF,
                    stop_words='english',
                    dtype=np.float32
                )
            tfidf_matrix = self.tfidf.fit_transform(combined_texts)
        else:
            if self.tfidf is None:
                raise ValueError("TF-IDF vectorizer not fitted. Call fit on training data first.")
            tfidf_matrix = self.tfidf.transform(combined_texts)

        # === 4. Combine All Features Efficiently ===
        # Scale dense features only; keep TF-IDF sparse
        dense_features = np.hstack([basic_features, similarity_features])  # (n, 13)

        if is_training:
            if self.scaler is None:
                self.scaler = StandardScaler()
            dense_scaled = self.scaler.fit_transform(dense_features)
        else:
            dense_scaled = self.scaler.transform(dense_features)

        # Convert scaled dense features to sparse matrix for stacking
        dense_sparse = sparse.csr_matrix(dense_scaled)

        # Stack dense + TF-IDF (both sparse)
        all_sparse_features = sparse_hstack([dense_sparse, tfidf_matrix], format='csr')

        return all_sparse_features

    def train_ensemble(self, X: sparse.csr_matrix, y: np.ndarray) -> float:
        # Define base models
        models = [
            ('lr_l1', LogisticRegression(
                penalty='l1', solver='liblinear', C=0.1,
                random_state=self.config.RANDOM_SEED, max_iter=1000
            )),
            ('lr_l2', LogisticRegression(
                penalty='l2', C=0.5,
                random_state=self.config.RANDOM_SEED, max_iter=1000
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=self.config.RANDOM_SEED, n_jobs=-1
            ))
        ]

        # Train full ensemble
        self.ensemble = VotingClassifier(estimators=models, voting='soft')
        self.ensemble.fit(X, y)

        # Cross-validation score
        skf = StratifiedKFold(
            n_splits=self.config.CV_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_SEED
        )
        cv_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train on fold
            fold_ensemble = VotingClassifier(estimators=models, voting='soft')
            fold_ensemble.fit(X_train, y_train)

            # Predict
            y_pred = fold_ensemble.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, y_pred)
            cv_scores.append(fold_auc)

        return np.mean(cv_scores)

    def predict_proba(self, X: sparse.csr_matrix) -> np.ndarray:
        """
        Predict class probabilities using trained ensemble.
        
        Args:
            X (sparse.csr_matrix): Input features
            
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        if self.ensemble is None:
            raise RuntimeError("Model not trained. Call train_ensemble() first.")
        return self.ensemble.predict_proba(X)[:, 1]


# ===============================
# HYBRID PREDICTOR
# ===============================

class HybridPredictor:
    """
    Hybrid prediction system combining:
        - ML model (TF-IDF + handcrafted features + ensemble)
        - Similarity-based scoring (word + char n-gram match to examples)
        - Rule-based prior (optional, currently neutral)
    
    Designed for high performance in offline environments   
Designed for high performance in offline environments with ensemble diversity.
    """

    def __init__(self, config):
        self.config = config
        self.ml_pipeline = StreamlinedMLPipeline(config)
        self.similarity_engine = FastSimilarityEngine()
        self.text_processor = FastTextProcessor()
        self.trained = False

    def fit(self, df_train: pd.DataFrame) -> float:
        """
        Train the hybrid model using training data.
        
        Args:
            df_train (pd.DataFrame): Training DataFrame with 'body', 'rule', examples, and 'rule_violation'
            
        Returns:
            float: Mean CV AUC score from ML pipeline
        """
        if 'rule_violation' not in df_train.columns:
            raise ValueError("Training data must contain 'rule_violation' column.")

        # Extract features and train ML pipeline
        X = self.ml_pipeline.extract_features(df_train, is_training=True)
        y = df_train['rule_violation'].values.astype(np.float32)

        cv_score = self.ml_pipeline.train_ensemble(X, y)
        self.trained = True

        return cv_score

    def predict_batch(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions for a batch of test samples.
        
        Combines:
            - ML model probability (weighted)
            - Similarity-based score (weighted)
            - Rule-based prior (currently neutral)
        
        Adds small noise for ensemble diversity and clips output to valid probability range.
        
        Args:
            df_test (pd.DataFrame): Test data with 'body' and example columns
            
        Returns:
            np.ndarray: Final ensemble predictions in [0, 1]
        """
        if not isinstance(df_test, pd.DataFrame) or df_test.empty:
            raise ValueError("df_test must be a non-empty DataFrame.")

        n_samples = len(df_test)
        all_predictions = []

        # Pre-clean all comments to avoid redundant processing
        cleaned_comments = df_test['body'].apply(self.text_processor.clean_text).tolist()

        # Extract positive/negative examples
        pos_ex1 = df_test.get('positive_example_1', pd.Series([""] * n_samples)).fillna("").apply(
            self.text_processor.clean_text).tolist()
        pos_ex2 = df_test.get('positive_example_2', pd.Series([""] * n_samples)).fillna("").apply(
            self.text_processor.clean_text).tolist()
        neg_ex1 = df_test.get('negative_example_1', pd.Series([""] * n_samples)).fillna("").apply(
            self.text_processor.clean_text).tolist()
        neg_ex2 = df_test.get('negative_example_2', pd.Series([""] * n_samples)).fillna("").apply(
            self.text_processor.clean_text).tolist()

        # Get ML predictions (once per ensemble loop only if trained)
        if self.trained:
            X_test = self.ml_pipeline.extract_features(df_test, is_training=False)
            ml_predictions = self.ml_pipeline.predict_proba(X_test)
        else:
            ml_predictions = np.full(n_samples, 0.5)

        for ensemble_idx in range(self.config.N_ENSEMBLE):
            # Set seed for reproducible noise
            np.random.seed(self.config.RANDOM_SEED + ensemble_idx * 123)

            # === Similarity Predictions ===
            similarity_scores = np.zeros(n_samples)
            for i in range(n_samples):
                comment = cleaned_comments[i]
                pos_examples = [pos_ex1[i], pos_ex2[i]]
                neg_examples = [neg_ex1[i], neg_ex2[i]]

                sim_features = self.similarity_engine.get_example_similarities(comment, pos_examples, neg_examples)
                diff_score = sim_features[4]  # avg_pos - avg_neg
                # Sigmoid scaling for better calibration
                sim_prob = 1 / (1 + np.exp(-5 * diff_score))
                similarity_scores[i] = sim_prob

            # === Combine Predictions ===
            rule_prior = 0.5  # Neutral prior (no rule logic yet)
            combined = (
                self.config.ML_WEIGHT * ml_predictions +
                self.config.SIMILARITY_WEIGHT * similarity_scores +
                self.config.RULE_WEIGHT * rule_prior
            )

            # Add small noise for ensemble diversity
            noise = np.random.normal(0, 0.01, n_samples)
            combined = np.clip(combined + noise, 0.01, 0.99)

            all_predictions.append(combined)

        # Final ensemble average
       
        final_predictions = np.mean(all_predictions, axis=0)
        return final_predictions


# ===============================
# POST-PROCESSING
# ===============================

class FastPostProcessor:
    """
    Optimized post-processing for hybrid predictions.
    
    Features:
        - Exact match overrides using (rule, subreddit, example) â†’ high/low confidence
        - Light calibration to prevent overconfident predictions
        - Immutable design (returns new DataFrame where applicable)
    """

    @staticmethod
    def apply_exact_matches(
        df_test: pd.DataFrame,
        df_train: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Apply high-confidence overrides for exact matches between test comments and known examples.
        
        Matches on: (rule, subreddit, body) == (rule, subreddit, positive/negative example)
        
        Assigns:
            - 0.95 for matches with positive examples
            - 0.05 for matches with negative examples
        
        Args:
            df_test (pd.DataFrame): Test data with 'rule', 'subreddit', 'body', and 'prediction' columns
            df_train (pd.DataFrame, optional): Training data to build lookup from
            
        Returns:
            Tuple[pd.DataFrame, int]: Updated DataFrame and count of exact matches applied
        """
        if df_train is None or df_test.empty:
            return df_test.copy(), 0

        # Initialize prediction column if not present
        df = df_test.copy()
        if 'prediction' not in df.columns:
            df['prediction'] = 0.5

        # Normalize and prepare key columns
        def safe_clean(series: pd.Series) -> pd.Series:
            return series.astype(str).str.strip().replace('nan', '')

        train_rules = safe_clean(df_train['rule'])
        train_subreddits = safe_clean(df_train['subreddit'])
        test_rules = safe_clean(df_test['rule'])
        test_subreddits = safe_clean(df_test['subreddit'])
        test_bodies = safe_clean(df_test['body'])

        # Build exact match lookup using dictionary
        lookup = {}

        for df_lookup in [df_train]:
            rules = safe_clean(df_lookup['rule'])
            subreddits = safe_clean(df_lookup['subreddit'])
            pos1 = safe_clean(df_lookup['positive_example_1'])
            pos2 = safe_clean(df_lookup['positive_example_2'])
            neg1 = safe_clean(df_lookup['negative_example_1'])
            neg2 = safe_clean(df_lookup['negative_example_2'])

            for i in range(len(df_lookup)):
                rule = rules.iloc[i]
                subreddit = subreddits.iloc[i]

                # Add positive examples â†’ high score
                if pos1.iloc[i]:
                    lookup[(rule, subreddit, pos1.iloc[i])] = 0.95
                if pos2.iloc[i]:
                    lookup[(rule, subreddit, pos2.iloc[i])] = 0.95

                # Add negative examples â†’ low score
                if neg1.iloc[i]:
                    lookup[(rule, subreddit, neg1.iloc[i])] = 0.05
                if neg2.iloc[i]:
                    lookup[(rule, subreddit, neg2.iloc[i])] = 0.05

        # Vectorized lookup using zip for key matching
        match_keys = list(zip(test_rules, test_subreddits, test_bodies))
        new_preds = df['prediction'].values.copy()
        exact_matches = 0

        for i, key in enumerate(match_keys):
            if key in lookup:
                new_preds[i] = lookup[key]
                exact_matches += 1

        df['prediction'] = new_preds

        return df, exact_matches

    @staticmethod
    def calibrate_predictions(predictions: np.ndarray) -> np.ndarray:
        """
        Apply light calibration to avoid overconfident predictions.
        
        Clips to [0.001, 0.999] and applies linear shrinkage:
            calibrated = predictions * 0.9 + 0.05
        
        This pulls extreme values slightly toward center while preserving ranking.
        
        Args:
            predictions (np.ndarray): Raw model outputs in [0, 1]
            
        Returns:
            np.ndarray: Calibrated predictions in [0.001, 0.999]
        """
        predictions = np.clip(predictions, 0.001, 0.999)
        return predictions * 0.9 + 0.05


# ===============================
# MAIN PIPELINE
# ===============================

# Configure paths (Kaggle-compatible)
INPUT_DIR = "/kaggle/input/jigsaw-agile-community-rules"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")


def load_data() -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """
    Load training and test datasets with graceful fallback.
    
    Returns:
        Tuple[Optional[pd.DataFrame], pd.DataFrame]: (train_df, test_df)
    """
    try:
        df_test = pd.read_csv(TEST_CSV)
    except Exception as e:
        raise FileNotFoundError(f"Could not load test data from {TEST_CSV}: {e}")

    if not os.path.exists(TRAIN_CSV):
        print("âš ï¸� Training data not found. Running in inference mode.")
        return None, df_test

    try:
        df_train = pd.read_csv(TRAIN_CSV)
        print(f"âœ… Loaded {len(df_train):,} training samples and {len(df_test):,} test samples.")
    except Exception as e:
        print(f"âš ï¸� Failed to load training data: {e}. Proceeding in inference mode.")
        return None, df_test

    return df_train, df_test


def main() -> pd.DataFrame:
    """
    Main execution pipeline for Jigsaw Agile Community Rules Classification.
    
    Workflow:
        1. Load data (train + test)
        2. Train hybrid model (if labels available)
        3. Generate ensemble predictions
        4. Apply post-processing (exact matches + calibration)
        5. Save submission
    
    Returns:
        pd.DataFrame: Submission file
    """
    config = Config()
    print("ğŸš€ Starting hybrid prediction pipeline...")

    # Load data
    df_train, df_test = load_data()

    # Initialize predictor
    predictor = HybridPredictor(config)

    # Train model if training data is available
    if df_train is not None and 'rule_violation' in df_train.columns:
        print("ğŸ“Š Training hybrid model...")
        cv_score = predictor.fit(df_train)
        print(f"ğŸ“ˆ Cross-validated AUC score: {cv_score:.4f}")
    else:
        print("âš ï¸� No training data or labels found. Using neutral predictions.")

    # Generate predictions
    print("ğŸ”® Generating predictions on test set...")
    predictions = predictor.predict_batch(df_test)
    df_test = df_test.copy()  # Ensure no side effects
    df_test['prediction'] = predictions

    # Post-processing
    post_processor = FastPostProcessor()

    if df_train is not None:
        print("ğŸ”§ Applying exact match overrides...")
        df_test, exact_matches = post_processor.apply_exact_matches(df_test, df_train)
        if exact_matches > 0:
            print(f"âœ… Applied exact match corrections to {exact_matches} samples.")

    # Calibration
    print("âš–ï¸� Calibrating predictions...")
    df_test['prediction'] = post_processor.calibrate_predictions(df_test['prediction'])

    # Prepare submission
    required_columns = ['row_id', 'prediction']
    for col in required_columns:
        if col not in df_test.columns:
            raise ValueError(f"Missing required column in test data: {col}")

    submission = pd.DataFrame({
        'row_id': df_test['row_id'],
        'rule_violation': df_test['prediction']
    })

    # Save submission
    output_path = "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"ğŸ’¾ Submission saved to '{output_path}'")

    print("ğŸ�‰ Pipeline completed successfully!")
    return submission


# ===============================
# EXECUTION
# ===============================

if __name__ == "__main__":
    np.random.seed(Config.RANDOM_SEED)
    submission = None

    try:
        print("ğŸš€ Starting pipeline execution...")
        submission = main()
        
        # Final validation
        assert submission is not None, "Submission DataFrame is None"
        assert 'row_id' in submission.columns, "Missing 'row_id' column"
        assert 'rule_violation' in submission.columns, "Missing 'rule_violation' column"
        assert len(submission) > 0, "Submission is empty"
        assert (submission['rule_violation'] >= 0.0).all() and (submission['rule_violation'] <= 1.0).all(), \
               "Predictions must be in [0, 1]"

        print(f"âœ… Success! Generated {len(submission)} predictions")
        
    except Exception as e:
        import traceback
        print(f"â�Œ Error during execution: {e}")
        traceback.print_exc()
        
        # Fallback: neutral predictions
        try:
            print("ğŸ”„ Generating fallback submission with neutral probabilities (0.5)...")
            test_df = pd.read_csv(TEST_CSV)
            
            fallback_submission = pd.DataFrame({
                'row_id': test_df['row_id'],
                'rule_violation': [0.5] * len(test_df)
            })
            fallback_submission.to_csv("submission.csv", index=False)
            print("âœ… Fallback submission created successfully.")
            
        except Exception as fallback_error:
            print(f"ğŸ’¥ Critical failure in fallback: {fallback_error}")
            raise  # Re-raise only if we can't recover at all

    # Final check: ensure submission is valid
    if submission is not None:
        assert 'row_id' in submission.columns
        assert 'rule_violation' in submission.columns
        assert len(submission) > 0
        assert (submission['rule_violation'] >= 0).all() and (submission['rule_violation'] <= 1).all()   

