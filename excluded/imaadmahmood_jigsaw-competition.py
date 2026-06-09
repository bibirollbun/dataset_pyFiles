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


# Add this as your first cell
# !pip install keras-nlp tensorflow transformers accelerate bitsandbytes --quiet


#!/usr/bin/env python3
"""
Jigsaw Agile Community Rules Classification - Lightweight Solution
Optimized for Kaggle's constraints (offline, limited memory)
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try different approaches based on available resources
import sys
from pathlib import Path

# ===============================
# CONFIGURATION
# ===============================
class Config:
    # Data paths
    TRAIN_PATH = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
    TEST_PATH = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
    
    # Generation parameters
    SEED = 777
    N_ENSEMBLE = 3

# ===============================
# LIGHTWEIGHT RULE-BASED APPROACH
# ===============================
class LightweightJigsawModel:
    """Lightweight model using rule-based and similarity approaches"""
    
    def __init__(self, config):
        self.config = config
        self.setup_model()
    
    def setup_model(self):
        """Setup lightweight components"""
        print("Setting up lightweight rule-based model...")
        
        # Try to import sentence transformers for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            # Try to load a small model that might be cached
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
            print("âœ… Using SentenceTransformer embeddings")
        except:
            self.embedder = None
            self.use_embeddings = False
            print("âš ï¸� Using basic text similarity")
    
    def compute_text_similarity(self, text1, text2):
        """Compute similarity between two texts"""
        if self.use_embeddings and self.embedder is not None:
            try:
                emb1 = self.embedder.encode([text1])
                emb2 = self.embedder.encode([text2])
                similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
                return float(similarity)
            except:
                pass
        
        # Fallback: simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def predict_single(self, row, seed=None):
        """Predict rule violation for a single row"""
        if seed is not None:
            np.random.seed(seed)
        
        comment = row.body.strip().lower()
        
        # Compute similarities to examples
        pos_sim1 = self.compute_text_similarity(comment, row.positive_example_1.lower())
        pos_sim2 = self.compute_text_similarity(comment, row.positive_example_2.lower())
        neg_sim1 = self.compute_text_similarity(comment, row.negative_example_1.lower())
        neg_sim2 = self.compute_text_similarity(comment, row.negative_example_2.lower())
        
        # Average similarities
        avg_pos_sim = (pos_sim1 + pos_sim2) / 2
        avg_neg_sim = (neg_sim1 + neg_sim2) / 2
        
        # Rule-based features
        features = self.extract_features(row)
        
        # Combine similarities and features
        similarity_score = avg_pos_sim - avg_neg_sim  # Range: -1 to 1
        feature_score = features
        
        # Weighted combination
        combined_score = 0.7 * similarity_score + 0.3 * feature_score
        
        # Convert to probability (sigmoid-like transformation)
        probability = 1 / (1 + np.exp(-5 * combined_score))  # Scale factor 5
        
        # Add some randomness for ensemble diversity
        if seed is not None:
            noise = np.random.normal(0, 0.02)  # Small noise
            probability = np.clip(probability + noise, 0.01, 0.99)
        
        return probability
    
    def extract_features(self, row):
        """Extract rule-based features"""
        comment = row.body.lower()
        rule = row.rule.lower()
        
        # Feature extraction based on common rule patterns
        feature_score = 0.0
        
        # Length-based features
        if len(comment.split()) < 3:
            feature_score += 0.1  # Very short comments might be violations
        
        # Punctuation patterns
        if comment.count('!') > 2:
            feature_score += 0.15  # Excessive exclamation
        if comment.count('?') > 2:
            feature_score += 0.1   # Multiple questions
        
        # Caps patterns
        caps_ratio = sum(1 for c in comment if c.isupper()) / max(len(comment), 1)
        if caps_ratio > 0.3:
            feature_score += 0.2  # Excessive caps
        
        # Common violation indicators
        violation_keywords = [
            'spam', 'advertisement', 'buy now', 'click here', 'visit my',
            'stupid', 'idiot', 'hate', 'kill', 'die',
            'off-topic', 'unrelated', 'politics', 'religion'
        ]
        
        for keyword in violation_keywords:
            if keyword in comment:
                feature_score += 0.1
        
        # Rule-specific patterns
        if 'spam' in rule:
            if any(word in comment for word in ['buy', 'sell', 'discount', 'offer', 'deal']):
                feature_score += 0.3
        
        if 'harassment' in rule or 'personal attack' in rule:
            if any(word in comment for word in ['you are', 'stupid', 'idiot', 'moron']):
                feature_score += 0.4
        
        if 'off-topic' in rule:
            # This is harder to detect without context, use example similarity more
            feature_score += 0.0
        
        return np.tanh(feature_score)  # Normalize to [-1, 1]
    
    def predict_batch(self, df, seed=None):
        """Process entire dataframe"""
        predictions = []
        
        print(f"Processing {len(df)} samples with lightweight model...")
        
        for idx, row in df.iterrows():
            if idx % 5 == 0:
                print(f"Processing sample {idx+1}/{len(df)}")
            
            try:
                prob = self.predict_single(row, seed)
                predictions.append(prob)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # Fallback: use similarity only
                try:
                    pos_sim = (self.compute_text_similarity(row.body, row.positive_example_1) + 
                              self.compute_text_similarity(row.body, row.positive_example_2)) / 2
                    predictions.append(min(0.9, max(0.1, pos_sim)))
                except:
                    predictions.append(0.5)  # Ultimate fallback
        
        return np.array(predictions)

# ===============================
# ADVANCED PROCESSING
# ===============================
class AdvancedProcessor:
    """Post-processing and ensemble methods"""
    
    @staticmethod
    def ensemble_predictions(predictions_list):
        """Ensemble multiple prediction sets"""
        if len(predictions_list) == 1:
            return predictions_list[0]
        
        # Robust ensemble with outlier handling
        predictions_array = np.array(predictions_list)
        
        # Use median for robustness, mean for smoothness
        median_pred = np.median(predictions_array, axis=0)
        mean_pred = np.mean(predictions_array, axis=0)
        
        # Weighted combination (more weight to median for robustness)
        ensemble = 0.6 * median_pred + 0.4 * mean_pred
        
        return ensemble
    
    @staticmethod
    def apply_exact_matches(df_test, df_train=None):
        """Apply exact match overrides"""
        if df_train is None:
            return df_test
        
        # Create comprehensive example lookup
        example_dict = {}
        # Get common columns (exclude 'pred' and 'rule_violation' columns)
        common_cols = [col for col in df_test.columns if col in df_train.columns and col not in ['pred', 'rule_violation']]
        df_all = pd.concat([df_train[common_cols], df_test[common_cols]]).reset_index(drop=True)
        
        for _, row in df_all.iterrows():
            key_base = (row.rule.strip(), row.subreddit.strip())
            
            # Positive examples (violations) = 1.0
            example_dict[(*key_base, row.positive_example_1.strip())] = 0.95
            example_dict[(*key_base, row.positive_example_2.strip())] = 0.95
            
            # Negative examples (non-violations) = 0.0  
            example_dict[(*key_base, row.negative_example_1.strip())] = 0.05
            example_dict[(*key_base, row.negative_example_2.strip())] = 0.05
        
        # Apply exact matches
        exact_matches = 0
        for idx, row in df_test.iterrows():
            key = (row.rule.strip(), row.subreddit.strip(), row.body.strip())
            if key in example_dict:
                df_test.loc[idx, 'pred'] = example_dict[key]
                exact_matches += 1
        
        print(f"Applied {exact_matches} exact match overrides")
        
        # Also check for very high similarity matches
        similarity_matches = 0
        for idx, row in df_test.iterrows():
            if df_test.loc[idx, 'pred'] in [0.95, 0.05]:  # Skip if already exact match
                continue
                
            # Check for very high similarity to examples
            comment = row.body.strip().lower()
            examples = [
                (row.positive_example_1.strip().lower(), 0.9),
                (row.positive_example_2.strip().lower(), 0.9),
                (row.negative_example_1.strip().lower(), 0.1),
                (row.negative_example_2.strip().lower(), 0.1)
            ]
            
            for example_text, example_label in examples:
                # Simple high similarity check
                if len(comment) > 10 and len(example_text) > 10:
                    # Check for substantial overlap
                    comment_words = set(comment.split())
                    example_words = set(example_text.split())
                    overlap = len(comment_words.intersection(example_words))
                    min_len = min(len(comment_words), len(example_words))
                    
                    if overlap >= min_len * 0.8 and overlap >= 3:  # 80% overlap, at least 3 words
                        df_test.loc[idx, 'pred'] = example_label
                        similarity_matches += 1
                        break
        
        print(f"Applied {similarity_matches} high-similarity overrides")
        return df_test
    
    @staticmethod
    def calibrate_predictions(predictions):
        """Apply probability calibration"""
        predictions = np.array(predictions)
        
        # Clip extreme values
        predictions = np.clip(predictions, 0.001, 0.999)
        
        # Apply slight regularization (pull extreme values towards center)
        regularized = predictions * 0.85 + 0.075  # Pull towards 0.5
        
        # Final clipping
        return np.clip(regularized, 0.01, 0.99)

# ===============================
# MAIN PIPELINE
# ===============================
def main():
    """Main execution pipeline"""
    config = Config()
    
    # Load data
    print("Loading competition data...")
    df_test = pd.read_csv(config.TEST_PATH)
    
    try:
        df_train = pd.read_csv(config.TRAIN_PATH)
        has_train = True
        print(f"Loaded training data: {len(df_train)} samples")
    except:
        df_train = None
        has_train = False
        print("Training data not available")
    
    print(f"Test data: {len(df_test)} samples")
    
    # Initialize lightweight model
    print("\nInitializing lightweight model...")
    model = LightweightJigsawModel(config)
    
    # Generate ensemble predictions
    print(f"\nGenerating {config.N_ENSEMBLE} ensemble predictions...")
    all_predictions = []
    
    for i in range(config.N_ENSEMBLE):
        print(f"\n=== Ensemble Run {i+1}/{config.N_ENSEMBLE} ===")
        seed = config.SEED + i * 123
        
        predictions = model.predict_batch(df_test, seed=seed)
        all_predictions.append(predictions)
        
        print(f"Run {i+1} stats:")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std:  {predictions.std():.4f}")
        print(f"  Min:  {predictions.min():.4f}")
        print(f"  Max:  {predictions.max():.4f}")
    
    # Create ensemble
    print("\n=== Creating Ensemble ===")
    processor = AdvancedProcessor()
    ensemble_predictions = processor.ensemble_predictions(all_predictions)
    
    print(f"Ensemble stats:")
    print(f"  Mean: {ensemble_predictions.mean():.4f}")
    print(f"  Std:  {ensemble_predictions.std():.4f}")
    
    # Apply calibration
    calibrated_predictions = processor.calibrate_predictions(ensemble_predictions)
    df_test['pred'] = calibrated_predictions
    
    # Apply exact matches and high similarity overrides
    if has_train:
        print("\n=== Applying Exact Match Overrides ===")
        df_test = processor.apply_exact_matches(df_test, df_train)
    
    # Create submission
    print("\n=== Creating Final Submission ===")
    df_test['rule_violation'] = df_test['pred']
    submission = df_test[['row_id', 'rule_violation']].copy()
    
    # Final validation and analysis
    print(f"\nSubmission validation:")
    print(f"  Shape: {submission.shape}")
    print(f"  Range: [{submission['rule_violation'].min():.4f}, {submission['rule_violation'].max():.4f}]")
    print(f"  Mean:  {submission['rule_violation'].mean():.4f}")
    print(f"  Std:   {submission['rule_violation'].std():.4f}")
    
    # Analyze by rule if available
    if 'rule' in df_test.columns:
        print(f"\n=== Analysis by Rule ===")
        rule_analysis = df_test.groupby('rule')['rule_violation'].agg(['count', 'mean', 'std']).round(4)
        print(rule_analysis)
    
    # Analyze by subreddit if available
    if 'subreddit' in df_test.columns:
        print(f"\n=== Analysis by Subreddit ===")
        subreddit_analysis = df_test.groupby('subreddit')['rule_violation'].agg(['count', 'mean', 'std']).round(4)
        print(subreddit_analysis)
    
    # Distribution analysis
    print(f"\n=== Prediction Distribution ===")
    print(f"Very low  (< 0.2): {(submission['rule_violation'] < 0.2).sum()}")
    print(f"Low       (0.2-0.4): {((submission['rule_violation'] >= 0.2) & (submission['rule_violation'] < 0.4)).sum()}")
    print(f"Medium    (0.4-0.6): {((submission['rule_violation'] >= 0.4) & (submission['rule_violation'] < 0.6)).sum()}")
    print(f"High      (0.6-0.8): {((submission['rule_violation'] >= 0.6) & (submission['rule_violation'] < 0.8)).sum()}")
    print(f"Very high (> 0.8): {(submission['rule_violation'] > 0.8).sum()}")
    
    # Save submission
    submission.to_csv("submission.csv", index=False)
    print("\nâœ… Submission saved successfully!")
    
    return submission

# ===============================
# EXECUTION
# ===============================
if __name__ == "__main__":
    try:
        submission = main()
        print("\nğŸ�‰ Pipeline completed successfully!")
        print("\nThis lightweight solution uses:")
        print("- Text similarity between comments and examples")
        print("- Rule-based feature extraction")
        print("- Ensemble predictions for robustness")
        print("- Exact match overrides for perfect accuracy on duplicates")
        print("- Advanced calibration for better probability estimates")
        
    except Exception as e:
        print(f"â�Œ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

