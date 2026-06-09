import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text




def create_example_features(df):
    # Clean all example texts
    for col in ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']:
        df[f'{col}_clean'] = df[col].apply(clean_text)
    
    # Calculate similarity features
    def calculate_similarity(text1, text2):
        if pd.isna(text1) or pd.isna(text2):
            return 0
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    # Similarity to positive examples
    df['sim_pos1'] = df.apply(lambda x: calculate_similarity(x['body_clean'], x['positive_example_1_clean']), axis=1)
    df['sim_pos2'] = df.apply(lambda x: calculate_similarity(x['body_clean'], x['positive_example_2_clean']), axis=1)
    df['sim_pos_avg'] = (df['sim_pos1'] + df['sim_pos2']) / 2
    
    # Similarity to negative examples
    df['sim_neg1'] = df.apply(lambda x: calculate_similarity(x['body_clean'], x['negative_example_1_clean']), axis=1)
    df['sim_neg2'] = df.apply(lambda x: calculate_similarity(x['body_clean'], x['negative_example_2_clean']), axis=1)
    df['sim_neg_avg'] = (df['sim_neg1'] + df['sim_neg2']) / 2
    
    # Difference between positive and negative similarities
    df['sim_diff'] = df['sim_pos_avg'] - df['sim_neg_avg']
    
    return df




def create_features(df):
    df['body_clean'] = df['body'].apply(clean_text)
    
    df = create_example_features(df)
    
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=3,
        max_df=0.9
    )
    
    tfidf_features = tfidf.fit_transform(df['body_clean'])
    
    df['text_length'] = df['body_clean'].str.len()
    df['word_count'] = df['body_clean'].str.split().str.len()
    df['avg_word_length'] = df['body_clean'].str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
    
    df['has_url'] = df['body'].str.contains(r'http|www|\.com|\.org|\.net', case=False, na=False).astype(int)
    df['has_caps'] = (df['body'].str.count(r'[A-Z]') / df['body'].str.len() > 0.5).astype(int)
    df['has_exclamation'] = df['body'].str.contains(r'!', na=False).astype(int)
    df['has_question'] = df['body'].str.contains(r'\?', na=False).astype(int)
    df['has_links'] = df['body'].str.contains(r'\[.*?\]\(.*?\)', na=False).astype(int)  # Markdown links
    
    df['caps_ratio'] = df['body'].str.count(r'[A-Z]') / df['body'].str.len()
    df['punctuation_ratio'] = df['body'].str.count(r'[!?]') / df['body'].str.len()
    
    # Combine all features
    feature_cols = ['text_length', 'word_count', 'avg_word_length', 
                   'has_url', 'has_caps', 'has_exclamation', 'has_question', 'has_links',
                   'caps_ratio', 'punctuation_ratio',
                   'sim_pos1', 'sim_pos2', 'sim_pos_avg',
                   'sim_neg1', 'sim_neg2', 'sim_neg_avg', 'sim_diff']
    
    additional_features = df[feature_cols].values
    
    return tfidf_features, additional_features, tfidf, feature_cols


def train_ensemble_model(X_train, y_train):
    from sklearn.ensemble import VotingClassifier
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    
    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble



print("Loading data...")

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

print(f"Training data shape: {train_df.shape}")





# Prepare features for training data
print("Creating features...")
tfidf_features_train, additional_features_train, tfidf, feature_cols = create_features(train_df)

X_train = np.hstack([tfidf_features_train.toarray(), additional_features_train])
y_train = train_df['rule_violation'].values

# validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Train 
print("Training ensemble model...")
model = train_ensemble_model(X_train_split, y_train_split)

# validation set
val_pred_proba = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred_proba)
print(f"Validation AUC: {val_auc:.4f}")

# Cross-validation score
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

#  features for test data
print("Preparing test features...")
test_df['body_clean'] = test_df['body'].apply(clean_text)
test_df = create_example_features(test_df)
tfidf_features_test = tfidf.transform(test_df['body_clean'])

# additional features for test data
test_df['text_length'] = test_df['body_clean'].str.len()
test_df['word_count'] = test_df['body_clean'].str.split().str.len()
test_df['avg_word_length'] = test_df['body_clean'].str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
test_df['has_url'] = test_df['body'].str.contains(r'http|www|\.com|\.org|\.net', case=False, na=False).astype(int)
test_df['has_caps'] = (test_df['body'].str.count(r'[A-Z]') / test_df['body'].str.len() > 0.5).astype(int)
test_df['has_exclamation'] = test_df['body'].str.contains(r'!', na=False).astype(int)
test_df['has_question'] = test_df['body'].str.contains(r'\?', na=False).astype(int)
test_df['has_links'] = test_df['body'].str.contains(r'\[.*?\]\(.*?\)', na=False).astype(int)
test_df['caps_ratio'] = test_df['body'].str.count(r'[A-Z]') / test_df['body'].str.len()
test_df['punctuation_ratio'] = test_df['body'].str.count(r'[!?]') / test_df['body'].str.len()

additional_features_test = test_df[feature_cols].values

X_test = np.hstack([tfidf_features_test.toarray(), additional_features_test])

print("Making predictions...")
predictions = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': predictions
})

submission.to_csv('submission.csv', index=False)




submission

