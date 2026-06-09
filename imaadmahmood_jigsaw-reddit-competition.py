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
Generalized Kaggle Solution: Jigsaw Agile Community Rules Classification
========================================================================

Focus on generalization to unseen rules through:
- Rule-agnostic features that transfer across different rule types
- Heavy emphasis on positive/negative example similarity
- Simpler, more robust feature engineering
- Avoiding overfitting to specific training rules

Target: Better generalization to unseen test rules
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data():
    """Load training and test data"""
    print("ğŸ”„ Loading data...")
    
    train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    
    # Fill missing values
    text_cols = ['body', 'rule', 'positive_example_1', 'positive_example_2', 
                'negative_example_1', 'negative_example_2']
    
    for col in text_cols:
        train[col] = train[col].fillna('')
        test[col] = test[col].fillna('')
    
    print(f"   âœ… Train: {train.shape}, Test: {test.shape}")
    return train, test

# ============================================================================
# 2. RULE-AGNOSTIC FEATURE ENGINEERING
# ============================================================================

def extract_universal_features(df):
    """Extract features that work across any rule type"""
    
    # Basic text properties
    df['body_len'] = df['body'].str.len()
    df['word_count'] = df['body'].str.split().str.len()
    df['sentence_count'] = df['body'].str.count(r'[.!?]+') + 1
    
    # Formatting signals (universal spam/violation indicators)
    df['caps_ratio'] = df['body'].apply(lambda x: sum(1 for c in x if c.isupper())) / (df['body_len'] + 1)
    df['exclamation_count'] = df['body'].str.count('!')
    df['question_count'] = df['body'].str.count(r'\?')
    
    # URL and link patterns (universal across rules)
    df['url_count'] = df['body'].str.count(r'http[s]?://')
    df['has_url'] = (df['url_count'] > 0).astype(int)
    
    # Universal violation patterns
    df['special_char_ratio'] = df['body'].str.count(r'[^\w\s]') / (df['body_len'] + 1)
    df['number_ratio'] = df['body'].str.count(r'\d') / (df['body_len'] + 1)
    
    return df

# ============================================================================
# 3. ADVANCED SIMILARITY FEATURES (KEY FOR GENERALIZATION)
# ============================================================================

def compute_comprehensive_similarities(df):
    """Compute detailed similarity features - the key to generalization"""
    print("ğŸ”— Computing similarity features...")
    
    # Multiple TF-IDF approaches for robustness
    
    # 1. Word-level TF-IDF
    word_tfidf = TfidfVectorizer(
        max_features=5000, 
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    # 2. Character-level TF-IDF (catches patterns/spam)
    char_tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 4),
        max_features=3000,
        min_df=2
    )
    
    # Fit on all available text
    all_texts = []
    for col in ['body', 'positive_example_1', 'positive_example_2', 
               'negative_example_1', 'negative_example_2']:
        all_texts.extend(df[col].values)
    
    word_tfidf.fit(all_texts)
    char_tfidf.fit(all_texts)
    
    # Transform each column
    body_word = word_tfidf.transform(df['body'])
    pos1_word = word_tfidf.transform(df['positive_example_1'])
    pos2_word = word_tfidf.transform(df['positive_example_2'])
    neg1_word = word_tfidf.transform(df['negative_example_1'])
    neg2_word = word_tfidf.transform(df['negative_example_2'])
    
    body_char = char_tfidf.transform(df['body'])
    pos1_char = char_tfidf.transform(df['positive_example_1'])
    pos2_char = char_tfidf.transform(df['positive_example_2'])
    neg1_char = char_tfidf.transform(df['negative_example_1'])
    neg2_char = char_tfidf.transform(df['negative_example_2'])
    
    # Word-level similarities
    df['word_sim_pos1'] = [cosine_similarity(body_word[i], pos1_word[i])[0,0] for i in range(len(df))]
    df['word_sim_pos2'] = [cosine_similarity(body_word[i], pos2_word[i])[0,0] for i in range(len(df))]
    df['word_sim_neg1'] = [cosine_similarity(body_word[i], neg1_word[i])[0,0] for i in range(len(df))]
    df['word_sim_neg2'] = [cosine_similarity(body_word[i], neg2_word[i])[0,0] for i in range(len(df))]
    
    # Character-level similarities
    df['char_sim_pos1'] = [cosine_similarity(body_char[i], pos1_char[i])[0,0] for i in range(len(df))]
    df['char_sim_pos2'] = [cosine_similarity(body_char[i], pos2_char[i])[0,0] for i in range(len(df))]
    df['char_sim_neg1'] = [cosine_similarity(body_char[i], neg1_char[i])[0,0] for i in range(len(df))]
    df['char_sim_neg2'] = [cosine_similarity(body_char[i], neg2_char[i])[0,0] for i in range(len(df))]
    
    # Aggregate similarity features
    df['max_pos_word_sim'] = np.maximum(df['word_sim_pos1'], df['word_sim_pos2'])
    df['max_neg_word_sim'] = np.maximum(df['word_sim_neg1'], df['word_sim_neg2'])
    df['max_pos_char_sim'] = np.maximum(df['char_sim_pos1'], df['char_sim_pos2'])
    df['max_neg_char_sim'] = np.maximum(df['char_sim_neg1'], df['char_sim_neg2'])
    
    # Discrimination features (key for generalization!)
    df['word_pos_vs_neg'] = df['max_pos_word_sim'] - df['max_neg_word_sim']
    df['char_pos_vs_neg'] = df['max_pos_char_sim'] - df['max_neg_char_sim']
    df['word_sim_ratio'] = df['max_pos_word_sim'] / (df['max_neg_word_sim'] + 0.01)
    df['char_sim_ratio'] = df['max_pos_char_sim'] / (df['max_neg_char_sim'] + 0.01)
    
    # Cross-modal agreement
    df['sim_agreement'] = ((df['word_pos_vs_neg'] > 0) == (df['char_pos_vs_neg'] > 0)).astype(int)
    
    return df, word_tfidf, char_tfidf

# ============================================================================
# 4. SIMPLE BUT EFFECTIVE MODELING
# ============================================================================

def create_feature_matrix(df, word_tfidf, char_tfidf):
    """Create comprehensive feature matrix"""
    
    # Engineered features
    feature_cols = [
        'body_len', 'word_count', 'sentence_count', 'caps_ratio',
        'exclamation_count', 'question_count', 'url_count', 'has_url',
        'special_char_ratio', 'number_ratio',
        'word_sim_pos1', 'word_sim_pos2', 'word_sim_neg1', 'word_sim_neg2',
        'char_sim_pos1', 'char_sim_pos2', 'char_sim_neg1', 'char_sim_neg2',
        'max_pos_word_sim', 'max_neg_word_sim', 'max_pos_char_sim', 'max_neg_char_sim',
        'word_pos_vs_neg', 'char_pos_vs_neg', 'word_sim_ratio', 'char_sim_ratio',
        'sim_agreement'
    ]
    
    # Get engineered features
    X_engineered = df[feature_cols].fillna(0)
    
    # Get text features (reduced dimensions for generalization)
    X_word = word_tfidf.transform(df['body'])
    X_char = char_tfidf.transform(df['body'])
    
    # Scale engineered features
    scaler = StandardScaler()
    X_engineered_scaled = scaler.fit_transform(X_engineered)
    
    # Combine features
    X_combined = hstack([
        csr_matrix(X_engineered_scaled),
        X_word,
        X_char
    ])
    
    return X_combined, scaler

def train_robust_model(X_train, y_train, X_test):
    """Train a model focused on generalization"""
    print("ğŸ�¯ Training robust model...")
    
    # Use simpler models that generalize better
    models = {
        'logistic': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, max_depth=8, 
                                   min_samples_split=20, random_state=42)
    }
    
    # Cross-validation
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    oof_preds = np.zeros(n_train)
    test_preds = np.zeros((n_test, len(models)))
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"   ğŸ“Š Fold {fold + 1}/5")
        
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        fold_preds = []
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_tr, y_tr)
            
            val_pred = model.predict_proba(X_val)[:, 1]
            fold_preds.append(val_pred)
            
            test_pred = model.predict_proba(X_test)[:, 1]
            test_preds[:, i] += test_pred / 5
        
        # Simple average ensemble
        fold_avg = np.mean(fold_preds, axis=0)
        oof_preds[val_idx] = fold_avg
        
        fold_auc = roc_auc_score(y_val, fold_avg)
        print(f"      ğŸ�¯ Fold {fold + 1} AUC: {fold_auc:.4f}")
    
    cv_auc = roc_auc_score(y_train, oof_preds)
    final_test_preds = np.mean(test_preds, axis=1)
    
    print(f"   âœ… CV AUC: {cv_auc:.4f}")
    return final_test_preds, cv_auc

# ============================================================================
# 5. SIMILARITY-BASED POST-PROCESSING
# ============================================================================

def similarity_post_process(predictions, test_df):
    """Apply similarity-based adjustments"""
    print("âš™ï¸� Applying similarity-based post-processing...")
    
    adjusted = predictions.copy()
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Strong positive similarity signal
        if (row['max_pos_word_sim'] > 0.15 and 
            row['word_pos_vs_neg'] > 0.05 and 
            row['sim_agreement'] == 1):
            adjusted[i] = min(0.9, adjusted[i] * 1.3)
        
        # Strong negative similarity signal
        elif (row['max_neg_word_sim'] > row['max_pos_word_sim'] and 
              row['word_pos_vs_neg'] < -0.05):
            adjusted[i] = max(0.1, adjusted[i] * 0.7)
        
        # High character similarity (spam patterns)
        if row['max_pos_char_sim'] > 0.2 and row['char_pos_vs_neg'] > 0.1:
            adjusted[i] = min(0.9, adjusted[i] * 1.2)
    
    return adjusted

# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    print("ğŸš€ Generalized Kaggle Solution for Unseen Rules")
    print("=" * 55)
    
    # Load data
    train, test = load_data()
    
    # Extract universal features
    print("ğŸ”§ Extracting universal features...")
    train = extract_universal_features(train)
    test = extract_universal_features(test)
    
    # Compute similarities on combined data
    combined = pd.concat([train, test]).reset_index(drop=True)
    combined, word_tfidf, char_tfidf = compute_comprehensive_similarities(combined)
    
    # Split back
    train = combined[:len(train)].copy()
    test = combined[len(train):].reset_index(drop=True)
    
    # Create feature matrices
    print("ğŸ“Š Creating feature matrices...")
    X_train, scaler = create_feature_matrix(train, word_tfidf, char_tfidf)
    
    # For test, we need to use the same scaler
    feature_cols = [
        'body_len', 'word_count', 'sentence_count', 'caps_ratio',
        'exclamation_count', 'question_count', 'url_count', 'has_url',
        'special_char_ratio', 'number_ratio',
        'word_sim_pos1', 'word_sim_pos2', 'word_sim_neg1', 'word_sim_neg2',
        'char_sim_pos1', 'char_sim_pos2', 'char_sim_neg1', 'char_sim_neg2',
        'max_pos_word_sim', 'max_neg_word_sim', 'max_pos_char_sim', 'max_neg_char_sim',
        'word_pos_vs_neg', 'char_pos_vs_neg', 'word_sim_ratio', 'char_sim_ratio',
        'sim_agreement'
    ]
    
    X_test_engineered = test[feature_cols].fillna(0)
    X_test_word = word_tfidf.transform(test['body'])
    X_test_char = char_tfidf.transform(test['body'])
    X_test_engineered_scaled = scaler.transform(X_test_engineered)
    
    X_test = hstack([
        csr_matrix(X_test_engineered_scaled),
        X_test_word,
        X_test_char
    ])
    
    print(f"   ğŸ“Š Train features: {X_train.shape}, Test features: {X_test.shape}")
    
    # Train model
    y = train['rule_violation']
    predictions, cv_auc = train_robust_model(X_train, y, X_test)
    
    # Apply post-processing
    final_predictions = similarity_post_process(predictions, test)
    
    # Create submission
    submission = pd.DataFrame({
        'row_id': test['row_id'],
        'rule_violation': final_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    
    # Results
    print("\n" + "=" * 55)
    print("ğŸ�‰ SOLUTION COMPLETED!")
    print("=" * 55)
    print(f"ğŸ“ˆ Cross-Validation AUC: {cv_auc:.4f}")
    print(f"ğŸ“Š Prediction Stats:")
    print(f"   â€¢ Min:  {final_predictions.min():.4f}")
    print(f"   â€¢ Max:  {final_predictions.max():.4f}")
    print(f"   â€¢ Mean: {final_predictions.mean():.4f}")
    print(f"\nğŸ�¯ Focus: Better generalization to unseen rules")
    print("âœ… submission.csv saved!")

if __name__ == "__main__":
    main()

