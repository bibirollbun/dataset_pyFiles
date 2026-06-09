import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import re
import string


# Load the data
print("Loading data...")
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


print("\nTarget distribution:")
target_dist = train['rule_violation'].value_counts()
print(target_dist)
print("="*50)
print("\nRules in training:")
print(train['rule'].value_counts())


plt.style.use('dark_background')
sns.set_palette("viridis")


# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a1a')

# Color palette for consistency
colors = ['#00d4aa', '#ff6b6b']

# Plot 1: Count Plot
target_dist.plot(kind='bar', ax=ax1, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
ax1.set_title('Rule Violation Distribution', fontsize=16, fontweight='bold', color='white', pad=20)
ax1.set_xlabel('Rule Violation', fontsize=12, color='white')
ax1.set_ylabel('Count', fontsize=12, color='white')
ax1.tick_params(colors='white', rotation=0)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_facecolor('#2d2d2d')

# Add value labels on bars
for i, v in enumerate(target_dist.values):
    ax1.text(i, v + max(target_dist.values) * 0.01, str(v), 
            ha='center', va='bottom', color='white', fontweight='bold')

# Plot 2: Pie Chart
wedges, texts, autotexts = ax2.pie(target_dist.values, 
                                    labels=target_dist.index,
                                    colors=colors,
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    explode=(0.05, 0.05),
                                    shadow=True)

# Customize pie chart text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

for text in texts:
    text.set_color('white')
    text.set_fontsize(12)
    text.set_fontweight('bold')

ax2.set_title('Rule Violation Proportion', fontsize=16, fontweight='bold', color='white', pad=20)

# Add summary statistics box
total_samples = target_dist.sum()
majority_class = target_dist.index[0]
majority_pct = (target_dist.iloc[0] / total_samples) * 100

stats_text = f'Total Samples: {total_samples:,}\nMajority Class: {majority_class}\nClass Balance: {majority_pct:.1f}% / {100-majority_pct:.1f}%'
fig.text(0.02, 0.02, stats_text, fontsize=12, color='white', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='#404040', alpha=0.8))

plt.tight_layout()
plt.show()


def clean_text(text):
    """Clean and preprocess text"""
    
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text


def create_features(df):
    """Build clean and combined features for training/testing."""
    # Clean core text columns
    df['body_clean'] = df['body'].apply(clean_text)
    df['rule_clean'] = df['rule'].apply(clean_text)
    df['subreddit_clean'] = df['subreddit'].apply(clean_text)

    # Clean example columns (if present, else empty)
    for col in ['positive_example_1', 'positive_example_2', 
                'negative_example_1', 'negative_example_2']:
        if col in df.columns:
            df[f'{col}_clean'] = df[col].fillna('').apply(clean_text)
        else:
            df[f'{col}_clean'] = ''

    # Combine positive & negative examples
    df['positive_examples'] = df['positive_example_1_clean'] + ' ' + df['positive_example_2_clean']
    df['negative_examples'] = df['negative_example_1_clean'] + ' ' + df['negative_example_2_clean']

    # Main text features (like "entailment" input)
    df['comment_rule'] = df['body_clean'] + ' [SEP] ' + df['rule_clean']
    df['full_context'] = (
        df['body_clean'] + ' [SEP] ' + df['rule_clean'] +
        ' [SEP] ' + df['subreddit_clean'] +
        ' [SEP] positive: ' + df['positive_examples'] +
        ' [SEP] negative: ' + df['negative_examples']
    )

    # Simple numeric features
    df['body_len'] = df['body_clean'].str.len()
    df['rule_len'] = df['rule_clean'].str.len()
    df['body_words'] = df['body_clean'].str.split().str.len()

    return df

print("\nCreating features...")
train_processed = create_features(train.copy())
test_processed = create_features(test.copy())


train_text = train_processed['full_context'].fillna('')
test_text = test_processed['full_context'].fillna('')

print("\nVectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=20_000,  
    ngram_range=(1, 2),  
    min_df=2,            
    max_df=0.95,         
    stop_words='english'
)

X_text = vectorizer.fit_transform(train_text)
X_test_text = vectorizer.transform(test_text)


numerical_features = ['body_len', 'rule_len', 'body_words']
X_num = train_processed[numerical_features].fillna(0)
X_test_num = test_processed[numerical_features].fillna(0)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Convert dense numerical features to sparse format  
X_num_sparse = csr_matrix(X_num_scaled)
X_test_num_sparse = csr_matrix(X_test_num_scaled)

# Stack sparse (TF-IDF) + dense (numerical) safely
X_combined = hstack([X_text, X_num_scaled])
X_test_combined = hstack([X_test_text, X_test_num_scaled])


y = train_processed['rule_violation']

print(f"\nFinal feature matrix shape: {X_combined.shape}")


print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',  
    random_state=42
)

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_model, X_combined, y, cv=cv, scoring='roc_auc')

print(f"CV AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Fit on full data
lr_model.fit(X_combined, y)


print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    use_label_encoder=False, eval_metric='logloss')


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_combined, y, cv=cv, scoring='roc_auc')


print(f"CV AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42)


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_combined, y, cv=cv, scoring='roc_auc')


print(f"CV AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# print("\nMaking predictions on test...")

# xgb_model.fit(X_combined, y)

# test_probs = xgb_model.predict_proba(X_test_combined)[:, 1]

# submission = sample_sub.copy()

# submission['rule_violation'] = test_probs

# submission.to_csv('submission.csv', index=False)

# print("\nSubmission saved as 'submission.csv'")

# print(submission.head())

# print("\nDone! ðŸŽ‰")


print("\nMaking predictions on test...")
rf_model.fit(X_combined, y)
test_probs = rf_model.predict_proba(X_test_combined)[:, 1]

submission = sample_sub.copy()
submission['rule_violation'] = test_probs
submission.to_csv('submission.csv', index=False)

print("\nSubmission saved as 'submission.csv'")
print(submission.head())
print("\nDone! ðŸŽ‰")

