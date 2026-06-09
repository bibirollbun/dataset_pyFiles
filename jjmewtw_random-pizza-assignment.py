# Import libraries
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


print("\nâš ï¸�  DATA LEAKAGE PREVENTION:")
print("   - User flair features removed (assigned AFTER receiving pizza)")
print("   - Upvote/downvote features removed (only available 'at retrieval' time)")
print("   - Only using features available at prediction time")
print("   - This ensures realistic model performance and recommendations")
print("=" * 60)


# Text Analysis Parameters
SENTIMENT_WORDS = {
    'positive': ['please', 'thank', 'grateful', 'appreciate', 'bless', 'help', 'kind', 'generous', 'amazing', 'wonderful'],
    'negative': ['broke', 'hungry', 'desperate', 'struggling', 'crisis', 'emergency', 'starving', 'homeless', 'unemployed'],
    'urgency': ['urgent', 'desperate', 'emergency', 'crisis', 'immediately', 'asap', 'tonight', 'today'],
    'politeness': ['please', 'thank', 'sorry', 'excuse', 'pardon', 'would', 'could', 'might'],
    'family': ['family', 'children', 'kids', 'baby', 'wife', 'husband', 'mother', 'father', 'son', 'daughter'],
    'financial': ['broke', 'money', 'paycheck', 'unemployed', 'bills', 'rent', 'debt', 'financial']
}

# TF-IDF Parameters - Reduced to prevent overfitting
TFIDF_PARAMS = {
    'unigrams': {'max_features': 50, 'ngram_range': (1, 1)},   # Reduced from 200
    'bigrams': {'max_features': 30, 'ngram_range': (2, 2)},    # Reduced from 150
    'trigrams': {'max_features': 20, 'ngram_range': (3, 3)},   # Reduced from 100
    'count': {'max_features': 20, 'ngram_range': (1, 2)}       # Reduced from 100
}

# Model Parameters - Regularized to prevent overfitting
MODEL_PARAMS = {
    'Random Forest': {
        'n_estimators': 50,        # Reduced from 200
        'max_depth': 5,            # Reduced from 15
        'min_samples_split': 20,   # Added regularization
        'min_samples_leaf': 10,    # Added regularization
        'max_features': 'sqrt',    # Added regularization
        'random_state': 42
    },
    'Gradient Boosting': {
        'n_estimators': 50,        # Reduced from 200
        'max_depth': 3,            # Reduced from 6
        'learning_rate': 0.1,      # Added learning rate
        'min_samples_split': 20,   # Added regularization
        'min_samples_leaf': 10,    # Added regularization
        'random_state': 42
    },
    'Logistic Regression': {
        'random_state': 42,
        'max_iter': 1000,
        'C': 0.1,                  # Added regularization (inverse of lambda)
        'penalty': 'l2'            # Added L2 regularization
    }
}

# Feature Engineering Parameters
FEATURE_PARAMS = {
    'account_age_categories': {
        'brand_new': 7,
        'new': 30,
        'established': 365
    },
    'temporal_features': {
        'evening_start': 18,
        'evening_end': 6,
        'late_night_start': 22,
        'late_night_end': 4
    }
}

# Analysis Parameters
ANALYSIS_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'top_features_display': 20
}


def get_config_summary():
    """Display configuration summary for reproducibility"""
    print("=== CONFIGURATION SUMMARY ===")
    print(f"Sentiment Categories: {list(SENTIMENT_WORDS.keys())}")
    print(f"TF-IDF Features: {sum([params['max_features'] for params in TFIDF_PARAMS.values()])}")
    print(f"Models: {list(MODEL_PARAMS.keys())}")
    print(f"Account Age Categories: {FEATURE_PARAMS['account_age_categories']}")
    print(f"Temporal Features: {FEATURE_PARAMS['temporal_features']}")
    print(f"Analysis Parameters: {ANALYSIS_PARAMS}")
    print("=" * 30)

def update_config(**kwargs):
    """Update configuration parameters dynamically"""
    global SENTIMENT_WORDS, TFIDF_PARAMS, MODEL_PARAMS, FEATURE_PARAMS, ANALYSIS_PARAMS
    
    if 'sentiment_words' in kwargs:
        SENTIMENT_WORDS.update(kwargs['sentiment_words'])
    if 'tfidf_params' in kwargs:
        TFIDF_PARAMS.update(kwargs['tfidf_params'])
    if 'model_params' in kwargs:
        MODEL_PARAMS.update(kwargs['model_params'])
    if 'feature_params' in kwargs:
        FEATURE_PARAMS.update(kwargs['feature_params'])
    if 'analysis_params' in kwargs:
        ANALYSIS_PARAMS.update(kwargs['analysis_params'])
    
    print("Configuration updated successfully!")


class PizzaRequestAnalyzer:
    """Main class for analyzing pizza requests with advanced NLP capabilities"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.text_analyzers = {}
        self.scaler = StandardScaler()
        
    def load_data(self, train_path, test_path):
        """Load and prepare the dataset"""
        print("Loading data...")
        with open('../input/random-acts-of-pizza/train.json', 'r') as f:
            self.train_data = json.load(f)
        with open('../input/random-acts-of-pizza/test.json', 'r') as f:
            self.test_data = json.load(f)
            
        self.train_df = pd.DataFrame(self.train_data)
        self.test_df = pd.DataFrame(self.test_data)
        
        print(f"Training data: {self.train_df.shape}")
        print(f"Test data: {self.test_df.shape}")
        print(f"Success rate: {self.train_df['requester_received_pizza'].mean():.2%}")
        
        return self.train_df, self.test_df
    
    def advanced_text_analysis(self, text_series):
        """Advanced text analysis with multiple NLP techniques"""
        print("Performing advanced text analysis...")
        
        # Use global sentiment words configuration
        sentiment_words = SENTIMENT_WORDS
        
        text_features = {}
        
        # Handle potential NaN values
        text_series = text_series.fillna('')
        
        for category, words in sentiment_words.items():
            pattern = '|'.join(words)
            text_features[f'{category}_count'] = text_series.str.lower().str.count(pattern).fillna(0)
            word_counts = text_series.str.split().str.len().fillna(0)
            text_features[f'{category}_ratio'] = text_features[f'{category}_count'] / (word_counts + 1)  # Add 1 to avoid division by zero
        
        # Text complexity features
        text_features['avg_word_length'] = text_series.str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
        text_features['sentence_count'] = text_series.str.count(r'[.!?]+').fillna(0)
        text_features['exclamation_count'] = text_series.str.count('!').fillna(0)
        text_features['question_count'] = text_series.str.count(r'\?').fillna(0)  # Escape the question mark
        text_lengths = text_series.str.len().fillna(0)
        text_features['caps_ratio'] = text_series.str.count(r'[A-Z]').fillna(0) / (text_lengths + 1)  # Add 1 to avoid division by zero
        
        # Readability indicators
        text_features['has_paragraphs'] = text_series.str.contains('\n\n').fillna(False).astype(int)
        text_features['has_emojis'] = text_series.str.contains(r'[ğŸ˜€-ğŸ™�]').fillna(False).astype(int)
        text_features['has_links'] = text_series.str.contains('http').fillna(False).astype(int)
        
        # Story structure analysis
        text_features['has_story_beginning'] = text_series.str.lower().str.contains(r'^(hi|hello|hey|so|well)').fillna(False).astype(int)
        text_features['has_story_ending'] = text_series.str.lower().str.contains(r'(thank|please|help|appreciate)$').fillna(False).astype(int)
        
        return pd.DataFrame(text_features, index=text_series.index)
    
    def create_advanced_features(self, df):
        """Create comprehensive engineered features"""
        print("Creating advanced engineered features...")
        df = df.copy()
        
        # Determine which text column to use (train has 'request_text', test has 'request_text_edit_aware')
        text_column = 'request_text' if 'request_text' in df.columns else 'request_text_edit_aware'
        print(f"Using text column: {text_column}")
        
        # Basic text features
        df['text_length'] = df[text_column].str.len()
        df['title_length'] = df['request_title'].str.len()
        df['text_word_count'] = df[text_column].str.split().str.len()
        df['title_word_count'] = df['request_title'].str.split().str.len()
        
        # Advanced text analysis
        text_features = self.advanced_text_analysis(df[text_column])
        df = pd.concat([df, text_features], axis=1)
        
        # Account features with more granularity
        df['account_age_days'] = df['requester_account_age_in_days_at_request']
        df['account_age_months'] = df['account_age_days'] / 30
        df['account_age_years'] = df['account_age_days'] / 365
        
        # Account age categories using configuration parameters
        age_params = FEATURE_PARAMS['account_age_categories']
        df['is_brand_new'] = (df['account_age_days'] < age_params['brand_new']).astype(int)
        df['is_new'] = ((df['account_age_days'] >= age_params['brand_new']) & (df['account_age_days'] < age_params['new'])).astype(int)
        df['is_established'] = ((df['account_age_days'] >= age_params['new']) & (df['account_age_days'] < age_params['established'])).astype(int)
        df['is_veteran'] = (df['account_age_days'] >= age_params['established']).astype(int)
        
        # Activity features with ratios
        df['total_activity'] = df['requester_number_of_posts_at_request'] + df['requester_number_of_comments_at_request']
        df['raop_activity'] = df['requester_number_of_posts_on_raop_at_request'] + df['requester_number_of_comments_in_raop_at_request']
        df['activity_ratio'] = df['raop_activity'] / (df['total_activity'] + 1)
        df['post_comment_ratio'] = df['requester_number_of_posts_at_request'] / (df['requester_number_of_comments_at_request'] + 1)
        
        # Social features - REMOVED due to data leakage
        # Upvotes/downvotes are only available "at retrieval" time, not at prediction time
        # This was causing perfect AUC scores (data leakage)
        df['upvotes'] = 0
        df['downvotes'] = 0
        df['net_votes'] = 0
        df['total_votes'] = 0
        df['vote_ratio'] = 0
        df['has_downvotes'] = 0
        
        # User flair encoding - REMOVED due to data leakage
        # User flair is assigned AFTER receiving pizza, so it's not available at prediction time
        # This was causing perfect AUC scores (data leakage)
        df['has_shroom_flair'] = 0  # Always 0 since we can't know this at prediction time
        df['has_pif_flair'] = 0     # Always 0 since we can't know this at prediction time
        df['has_no_flair'] = 1      # Always 1 since we assume no flair at prediction time
        
        # Temporal features using configuration parameters
        temporal_params = FEATURE_PARAMS['temporal_features']
        df['request_date'] = pd.to_datetime(df['unix_timestamp_of_request_utc'], unit='s')
        df['hour'] = df['request_date'].dt.hour
        df['day_of_week'] = df['request_date'].dt.dayofweek
        df['month'] = df['request_date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_evening'] = ((df['hour'] >= temporal_params['evening_start']) | (df['hour'] <= temporal_params['evening_end'])).astype(int)
        df['is_late_night'] = ((df['hour'] >= temporal_params['late_night_start']) | (df['hour'] <= temporal_params['late_night_end'])).astype(int)
        
        # Community features
        df['subreddit_diversity'] = df['requester_number_of_subreddits_at_request']
        df['is_community_member'] = (df['requester_days_since_first_post_on_raop_at_request'] > 0).astype(int)
        df['community_tenure'] = df['requester_days_since_first_post_on_raop_at_request']
        
        # Reputation features
        df['karma_per_day'] = df['requester_upvotes_minus_downvotes_at_request'] / (df['account_age_days'] + 1)
        df['has_negative_karma'] = (df['requester_upvotes_minus_downvotes_at_request'] < 0).astype(int)
        
        return df
    
    def create_text_features(self, train_text, test_text):
        """Create advanced text features using multiple vectorization techniques"""
        print("Creating advanced text features...")
        
        # Handle potential NaN values
        train_text = train_text.fillna('')
        test_text = test_text.fillna('')
        
        # Use configuration parameters for TF-IDF
        tfidf_unigrams = TfidfVectorizer(stop_words='english', **TFIDF_PARAMS['unigrams'])
        tfidf_bigrams = TfidfVectorizer(stop_words='english', **TFIDF_PARAMS['bigrams'])
        tfidf_trigrams = TfidfVectorizer(stop_words='english', **TFIDF_PARAMS['trigrams'])
        
        # Count vectorizer for specific patterns
        count_vectorizer = CountVectorizer(stop_words='english', **TFIDF_PARAMS['count'])
        
        # Fit and transform
        tfidf_uni_train = tfidf_unigrams.fit_transform(train_text)
        tfidf_uni_test = tfidf_unigrams.transform(test_text)
        
        tfidf_bi_train = tfidf_bigrams.fit_transform(train_text)
        tfidf_bi_test = tfidf_bigrams.transform(test_text)
        
        tfidf_tri_train = tfidf_trigrams.fit_transform(train_text)
        tfidf_tri_test = tfidf_trigrams.transform(test_text)
        
        count_train = count_vectorizer.fit_transform(train_text)
        count_test = count_vectorizer.transform(test_text)
        
        # Combine all text features
        text_features_train = np.hstack([tfidf_uni_train.toarray(), tfidf_bi_train.toarray(), 
                                       tfidf_tri_train.toarray(), count_train.toarray()])
        text_features_test = np.hstack([tfidf_uni_test.toarray(), tfidf_bi_test.toarray(), 
                                      tfidf_tri_test.toarray(), count_test.toarray()])
        
        # Create feature names
        feature_names = ([f'tfidf_uni_{i}' for i in range(tfidf_uni_train.shape[1])] +
                        [f'tfidf_bi_{i}' for i in range(tfidf_bi_train.shape[1])] +
                        [f'tfidf_tri_{i}' for i in range(tfidf_tri_train.shape[1])] +
                        [f'count_{i}' for i in range(count_train.shape[1])])
        
        return text_features_train, text_features_test, feature_names
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and compare performance with cross-validation"""
        print("Training multiple models with cross-validation...")
        
        # Use configuration parameters for models
        models = {
            'Random Forest': RandomForestClassifier(**MODEL_PARAMS['Random Forest']),
            'Gradient Boosting': GradientBoostingClassifier(**MODEL_PARAMS['Gradient Boosting']),
            'Logistic Regression': LogisticRegression(**MODEL_PARAMS['Logistic Regression'])
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation for more robust performance estimate
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
            
            # Metrics
            train_auc = roc_auc_score(y_train, train_pred)
            val_auc = roc_auc_score(y_val, val_pred)
            
            results[name] = {
                'model': model,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            print(f"{name} - Train AUC: {train_auc:.3f}, Val AUC: {val_auc:.3f}, CV AUC: {cv_mean:.3f} Â± {cv_std:.3f}")
            
            # Feature importance for different model types
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (Random Forest, Gradient Boosting)
                self.feature_importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models (Logistic Regression)
                # Use absolute coefficients as importance
                self.feature_importance[name] = np.abs(model.coef_[0])
            else:
                # Fallback: equal importance for all features
                self.feature_importance[name] = np.ones(X_train.shape[1]) / X_train.shape[1]
        
        self.models = results
        return results
    
    def generate_recommendations(self, feature_names, feature_importance, train_df, model_results=None):
        """Generate data-driven recommendations based on actual model insights"""
        print("\n" + "="*60)
        print("DATA-DRIVEN RECOMMENDATIONS")
        print("="*60)
        
        # Get top features
        top_features = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(ANALYSIS_PARAMS['top_features_display'])
        
        print(f"Top {ANALYSIS_PARAMS['top_features_display']} Most Important Features:")
        print(top_features)
        
        # Analyze actual data patterns
        print(f"\nğŸ“Š DATA-DRIVEN INSIGHTS:")
        
        # 1. Analyze user flair impact - REMOVED due to data leakage
        # User flair is assigned AFTER receiving pizza, so it's not a valid predictor
        print(f"\n1. USER FLAIR IMPACT:")
        print(f"   - Note: User flair is assigned AFTER receiving pizza (data leakage)")
        print(f"   - Cannot be used as a predictor in real-world scenarios")
        
        # 2. Analyze account age impact
        train_df['account_age_category'] = pd.cut(
            train_df['requester_account_age_in_days_at_request'],
            bins=[0, 7, 30, 365, float('inf')],
            labels=['Brand New (<7 days)', 'New (7-30 days)', 'Established (30-365 days)', 'Veteran (>1 year)']
        )
        age_analysis = train_df.groupby('account_age_category')['requester_received_pizza'].agg(['count', 'mean'])
        print(f"\n2. ACCOUNT AGE IMPACT:")
        for age_cat, stats in age_analysis.iterrows():
            success_rate = stats['mean'] * 100
            count = stats['count']
            print(f"   - {age_cat}: {success_rate:.1f}% success rate ({count} requests)")
        
        # 3. Analyze text length impact
        train_df['text_length_category'] = pd.cut(
            train_df['request_text'].str.len(),
            bins=[0, 200, 400, 600, float('inf')],
            labels=['Short (<200 chars)', 'Medium (200-400 chars)', 'Long (400-600 chars)', 'Very Long (>600 chars)']
        )
        text_analysis = train_df.groupby('text_length_category')['requester_received_pizza'].agg(['count', 'mean'])
        print(f"\n3. TEXT LENGTH IMPACT:")
        for text_cat, stats in text_analysis.iterrows():
            success_rate = stats['mean'] * 100
            count = stats['count']
            print(f"   - {text_cat}: {success_rate:.1f}% success rate ({count} requests)")
        
        # 4. Analyze timing impact
        train_df['request_hour'] = pd.to_datetime(train_df['unix_timestamp_of_request_utc'], unit='s').dt.hour
        train_df['time_category'] = pd.cut(
            train_df['request_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Late Night (12am-6am)', 'Morning (6am-12pm)', 'Afternoon (12pm-6pm)', 'Evening (6pm-12am)']
        )
        time_analysis = train_df.groupby('time_category')['requester_received_pizza'].agg(['count', 'mean'])
        print(f"\n4. TIMING IMPACT:")
        for time_cat, stats in time_analysis.iterrows():
            success_rate = stats['mean'] * 100
            count = stats['count']
            print(f"   - {time_cat}: {success_rate:.1f}% success rate ({count} requests)")
        
        # 5. Analyze upvotes impact - REMOVED due to data leakage
        # Upvotes are only available "at retrieval" time, not at prediction time
        print(f"\n5. UPVOTES IMPACT:")
        print(f"   - Note: Upvotes are only available 'at retrieval' time (data leakage)")
        print(f"   - Cannot be used as a predictor in real-world scenarios")
        
        # Generate recommendations based on actual analysis
        print(f"\nğŸ�• DATA-DRIVEN RECOMMENDATIONS:")
        
        # Recommendation 1: Based on top feature importance
        top_feature = top_features.iloc[0]['feature']
        top_importance = top_features.iloc[0]['importance']
        print(f"\n1. PRIORITY #1 - {top_feature.upper().replace('_', ' ')}:")
        print(f"   - This is the most important predictor (importance: {top_importance:.3f})")
        print(f"   - Focus on optimizing this aspect of your request")
        
        # Recommendation 2: Based on account age analysis (moved up since flair is removed)
        best_age = age_analysis['mean'].idxmax()
        best_age_rate = age_analysis.loc[best_age, 'mean'] * 100
        print(f"\n2. ACCOUNT ESTABLISHMENT:")
        print(f"   - {best_age} accounts have {best_age_rate:.1f}% success rate")
        print(f"   - Action: Build account history before requesting")
        
        # Recommendation 3: Based on text length analysis
        best_text = text_analysis['mean'].idxmax()
        best_text_rate = text_analysis.loc[best_text, 'mean'] * 100
        print(f"\n3. REQUEST LENGTH:")
        print(f"   - {best_text} requests have {best_text_rate:.1f}% success rate")
        print(f"   - Action: Write requests of optimal length")
        
        # Recommendation 4: Based on timing analysis
        best_time = time_analysis['mean'].idxmax()
        best_time_rate = time_analysis.loc[best_time, 'mean'] * 100
        print(f"\n4. TIMING STRATEGY:")
        print(f"   - {best_time} posts have {best_time_rate:.1f}% success rate")
        print(f"   - Action: Post during optimal time periods")
        
        # Recommendation 5: Community engagement (based on RAOP activity)
        print(f"\n5. COMMUNITY ENGAGEMENT:")
        print(f"   - Be active in the RAOP community before requesting")
        print(f"   - Comment on other requests to build relationships")
        print(f"   - Action: Build community presence and reputation")
        
        # Additional insights from feature importance
        print(f"\nğŸ“ˆ ADDITIONAL INSIGHTS FROM MODEL:")
        for i, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
            feature = row['feature'].replace('_', ' ').title()
            importance = row['importance']
            print(f"   {i}. {feature}: {importance:.3f} importance")
        
        # Model performance insights
        if model_results:
            print(f"\nğŸ�¯ MODEL PERFORMANCE:")
            for model_name, results in model_results.items():
                train_auc = results['train_auc']
                val_auc = results['val_auc']
                cv_mean = results['cv_mean']
                cv_std = results['cv_std']
                overfitting_gap = train_auc - val_auc
                
                print(f"   - {model_name}:")
                print(f"     * Train AUC: {train_auc:.3f}")
                print(f"     * Val AUC: {val_auc:.3f}")
                print(f"     * CV AUC: {cv_mean:.3f} Â± {cv_std:.3f}")
                print(f"     * Overfitting Gap: {overfitting_gap:.3f}")
                
                # Overfitting warning
                if overfitting_gap > 0.1:
                    print(f"     * âš ï¸�  WARNING: High overfitting detected!")
                elif overfitting_gap > 0.05:
                    print(f"     * âš ï¸�  CAUTION: Some overfitting present")
                else:
                    print(f"     * âœ… Good generalization")
        
        print(f"\nğŸ’¡ KEY TAKEAWAY:")
        print(f"   Focus on the top 3 most important features for maximum impact!")
        
        # Model selection insights
        if model_results:
            print(f"\nğŸ¤” MODEL SELECTION INSIGHTS:")
            print(f"   - Logistic Regression often performs well on small datasets")
            print(f"   - Tree-based models can overfit with limited data")
            print(f"   - Cross-validation provides more robust model comparison")
            print(f"   - Feature importance varies by model type (coefficients vs splits)")
    
    def create_eda_visualizations(self, train_df):
        """Create key visualizations for data storytelling"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Random Acts of Pizza: Data Story', fontsize=16, fontweight='bold')
        
        # 1. Success Rate by Account Age
        train_df['account_age_category'] = pd.cut(
            train_df['requester_account_age_in_days_at_request'],
            bins=[0, 7, 30, 365, float('inf')],
            labels=['Brand New\n(<7 days)', 'New\n(7-30 days)', 'Established\n(30-365 days)', 'Veteran\n(>1 year)']
        )
        age_success = train_df.groupby('account_age_category')['requester_received_pizza'].agg(['count', 'mean'])
        
        axes[0,0].bar(range(len(age_success)), age_success['mean'] * 100, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Success Rate by Account Age', fontweight='bold')
        axes[0,0].set_ylabel('Success Rate (%)')
        axes[0,0].set_xticks(range(len(age_success)))
        axes[0,0].set_xticklabels(age_success.index, rotation=45, ha='right')
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, (idx, row) in enumerate(age_success.iterrows()):
            axes[0,0].text(i, row['mean'] * 100 + 1, f"n={row['count']}", ha='center', fontsize=9)
        
        # 2. Success Rate by Text Length
        train_df['text_length_category'] = pd.cut(
            train_df['request_text'].str.len(),
            bins=[0, 200, 400, 600, float('inf')],
            labels=['Short\n(<200)', 'Medium\n(200-400)', 'Long\n(400-600)', 'Very Long\n(600+)']
        )
        text_success = train_df.groupby('text_length_category')['requester_received_pizza'].agg(['count', 'mean'])
        
        axes[0,1].bar(range(len(text_success)), text_success['mean'] * 100, color='lightcoral', alpha=0.7)
        axes[0,1].set_title('Success Rate by Request Length', fontweight='bold')
        axes[0,1].set_ylabel('Success Rate (%)')
        axes[0,1].set_xticks(range(len(text_success)))
        axes[0,1].set_xticklabels(text_success.index, rotation=45, ha='right')
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # Add count labels
        for i, (idx, row) in enumerate(text_success.iterrows()):
            axes[0,1].text(i, row['mean'] * 100 + 1, f"n={row['count']}", ha='center', fontsize=9)
        
        # 3. Success Rate by Time of Day
        train_df['request_hour'] = pd.to_datetime(train_df['unix_timestamp_of_request_utc'], unit='s').dt.hour
        train_df['time_category'] = pd.cut(
            train_df['request_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Late Night\n(12am-6am)', 'Morning\n(6am-12pm)', 'Afternoon\n(12pm-6pm)', 'Evening\n(6pm-12am)']
        )
        time_success = train_df.groupby('time_category')['requester_received_pizza'].agg(['count', 'mean'])
        
        axes[0,2].bar(range(len(time_success)), time_success['mean'] * 100, color='lightgreen', alpha=0.7)
        axes[0,2].set_title('Success Rate by Time of Day', fontweight='bold')
        axes[0,2].set_ylabel('Success Rate (%)')
        axes[0,2].set_xticks(range(len(time_success)))
        axes[0,2].set_xticklabels(time_success.index, rotation=45, ha='right')
        axes[0,2].grid(axis='y', alpha=0.3)
        
        # Add count labels
        for i, (idx, row) in enumerate(time_success.iterrows()):
            axes[0,2].text(i, row['mean'] * 100 + 1, f"n={row['count']}", ha='center', fontsize=9)
        
        # 4. Overall Success Rate
        success_rate = train_df['requester_received_pizza'].mean()
        axes[1,0].pie([success_rate, 1-success_rate], 
                      labels=[f'Success\n({success_rate:.1%})', f'No Success\n({1-success_rate:.1%})'],
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Overall Success Rate', fontweight='bold')
        
        # 5. Request Volume Over Time
        train_df['request_date'] = pd.to_datetime(train_df['unix_timestamp_of_request_utc'], unit='s')
        daily_requests = train_df.groupby(train_df['request_date'].dt.date).size()
        
        axes[1,1].plot(daily_requests.index, daily_requests.values, color='steelblue', alpha=0.7)
        axes[1,1].set_title('Request Volume Over Time', fontweight='bold')
        axes[1,1].set_ylabel('Daily Requests')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(alpha=0.3)
        
        # 6. Account Age Distribution
        axes[1,2].hist(train_df['requester_account_age_in_days_at_request'], 
                       bins=50, color='gold', alpha=0.7, edgecolor='black')
        axes[1,2].set_title('Account Age Distribution', fontweight='bold')
        axes[1,2].set_xlabel('Account Age (days)')
        axes[1,2].set_ylabel('Number of Requests')
        axes[1,2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print key insights
        print(f"\nğŸ“Š KEY INSIGHTS:")
        print(f"   â€¢ Overall success rate: {success_rate:.1%}")
        print(f"   â€¢ Best performing text length: {text_success['mean'].idxmax()} ({text_success['mean'].max():.1%})")
        print(f"   â€¢ Best posting time: {time_success['mean'].idxmax()} ({time_success['mean'].max():.1%})")
        print(f"   â€¢ Total requests analyzed: {len(train_df):,}")
        print(f"   â€¢ Date range: {train_df['request_date'].min().strftime('%Y-%m-%d')} to {train_df['request_date'].max().strftime('%Y-%m-%d')}")
        
        return train_df


# Display configuration for reproducibility
get_config_summary()


# Initialize analyzer
analyzer = PizzaRequestAnalyzer()

# Load data
train_df, test_df = analyzer.load_data('train.json', 'test.json')


# Create EDA visualizations
train_df = analyzer.create_eda_visualizations(train_df)


# Create advanced features
train_features = analyzer.create_advanced_features(train_df)
test_features = analyzer.create_advanced_features(test_df)


# Select feature columns (exclude target, text, and metadata columns)
exclude_cols = [
    'requester_received_pizza', 'request_text', 'request_text_edit_aware', 
    'request_title', 'request_id', 'requester_username', 'giver_username_if_known', 
    'request_date', 'requester_subreddits_at_request', 'requester_user_flair',
    # Also exclude columns that are only available in training set
    'number_of_upvotes_of_request_at_retrieval', 'number_of_downvotes_of_request_at_retrieval',
    'post_was_edited', 'request_number_of_comments_at_retrieval',
    'requester_account_age_in_days_at_retrieval', 'requester_days_since_first_post_on_raop_at_retrieval',
    'requester_number_of_comments_at_retrieval', 'requester_number_of_comments_in_raop_at_retrieval',
    'requester_number_of_posts_at_retrieval', 'requester_number_of_posts_on_raop_at_retrieval',
    'requester_upvotes_minus_downvotes_at_retrieval', 'requester_upvotes_plus_downvotes_at_retrieval'
]

feature_cols = [col for col in train_features.columns if col not in exclude_cols]


# Ensure both datasets have the same feature columns
common_features = [col for col in feature_cols if col in test_features.columns]
print(f"Created {len(common_features)} common features between train and test")
print(f"Features only in train: {len(feature_cols) - len(common_features)}")


# Use only common features
feature_cols = common_features


# Create advanced text features
# Determine which text column to use
train_text_col = 'request_text' if 'request_text' in train_features.columns else 'request_text_edit_aware'
test_text_col = 'request_text' if 'request_text' in test_features.columns else 'request_text_edit_aware'

text_features_train, text_features_test, text_feature_names = analyzer.create_text_features(
    train_features[train_text_col], test_features[test_text_col]
)


# Combine all features
X_train = np.hstack([train_features[feature_cols].values, text_features_train])
X_test = np.hstack([test_features[feature_cols].values, text_features_test])


# Create final feature names
all_feature_names = feature_cols + text_feature_names

print(f"Final feature matrix: {X_train.shape}")


# Prepare target
y_train = train_features['requester_received_pizza']


# Split for validation using configuration parameters
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, 
    test_size=ANALYSIS_PARAMS['test_size'], 
    random_state=ANALYSIS_PARAMS['random_state'], 
    stratify=y_train
)


# Scale features
X_train_scaled = analyzer.scaler.fit_transform(X_train_split)
X_val_scaled = analyzer.scaler.transform(X_val_split)


# Train models
results = analyzer.train_models(X_train_scaled, y_train_split, X_val_scaled, y_val_split)


# Find best model based on cross-validation (more robust than single validation split)
best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
best_model = results[best_model_name]['model']

print(f"\nBest model: {best_model_name} with CV AUC: {results[best_model_name]['cv_mean']:.3f} Â± {results[best_model_name]['cv_std']:.3f}")


# Generate recommendations
analyzer.generate_recommendations(
    all_feature_names, 
    analyzer.feature_importance.get(best_model_name, np.zeros(len(all_feature_names))),
    train_df,
    results
)


# Scale test data
X_test_scaled = analyzer.scaler.transform(X_test)

# Generate predictions with best model
final_predictions = best_model.predict_proba(X_test_scaled)[:, 1]

# Create submission
submission = pd.DataFrame({
    'request_id': test_features['request_id'],
    'requester_received_pizza': final_predictions
})


print(f"Generated predictions for {len(submission)} test requests")
print(f"Prediction range: {final_predictions.min():.3f} to {final_predictions.max():.3f}")
print(f"Mean prediction: {final_predictions.mean():.3f}")


# Save submission
submission.to_csv('enhanced_submission.csv', index=False)
print(f"Enhanced submission saved to 'enhanced_submission.csv'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("Key improvements: Advanced NLP, multiple models, comprehensive feature engineering")
print("Focus: Community reputation, account establishment, and request quality are critical")

