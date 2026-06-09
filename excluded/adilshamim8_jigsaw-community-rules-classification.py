# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import random
import warnings
from tqdm.notebook import tqdm
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve

# ML libraries
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
set_seed(42)
warnings.filterwarnings("ignore")

print("Environment setup complete!")


# Load the training and test data
train_df = pd.read_csv('../input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('../input/jigsaw-agile-community-rules/test.csv')
sample_submission = pd.read_csv('../input/jigsaw-agile-community-rules/sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Display the first few rows of the training data
train_df.head()


# Check for missing values
print("Missing values in training data:")
print(train_df.isnull().sum())

print("\nMissing values in test data:")
print(test_df.isnull().sum())

# Distribution of target variable
print("\nDistribution of rule violations:")
print(train_df['rule_violation'].value_counts(normalize=True) * 100)

# Unique rules in training and test data
train_rules = train_df['rule'].unique()
test_rules = test_df['rule'].unique()

print(f"\nNumber of unique rules in training data: {len(train_rules)}")
print(f"Number of unique rules in test data: {len(test_rules)}")

print("\nRules in training data:")
for rule in train_rules:
    print(f"- {rule}")

print("\nRules in test data:")
for rule in test_rules:
    print(f"- {rule}")

# Identify new rules in test data
new_rules = [rule for rule in test_rules if rule not in train_rules]
print(f"\nNumber of new rules in test data: {len(new_rules)}")
print("New rules in test data:")
for rule in new_rules:
    print(f"- {rule}")


# Analyze distribution of rules in training data
rule_counts = train_df['rule'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=rule_counts.index, y=rule_counts.values)
plt.title('Distribution of Rules in Training Data')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Analyze violation rate by rule
rule_violation_rate = train_df.groupby('rule')['rule_violation'].mean() * 100
plt.figure(figsize=(12, 6))
sns.barplot(x=rule_violation_rate.index, y=rule_violation_rate.values)
plt.title('Rule Violation Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Violation Rate (%)')
plt.tight_layout()
plt.show()

# Analyze distribution of subreddits
subreddit_counts = train_df['subreddit'].value_counts().head(20)
plt.figure(figsize=(14, 8))
sns.barplot(x=subreddit_counts.values, y=subreddit_counts.index)
plt.title('Top 20 Subreddits in Training Data')
plt.tight_layout()
plt.show()

# Analyze comment length
train_df['comment_length'] = train_df['body'].apply(lambda x: len(str(x)))
plt.figure(figsize=(12, 6))
sns.histplot(data=train_df, x='comment_length', hue='rule_violation', bins=50, kde=True)
plt.title('Distribution of Comment Length by Rule Violation')
plt.xlim(0, 1000)  # Limit x-axis for better visualization
plt.tight_layout()
plt.show()


# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '[USER]', text)
        # Remove special characters and numbers, keeping spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Apply cleaning to all text columns
for df in [train_df, test_df]:
    df['body_clean'] = df['body'].apply(clean_text)
    df['positive_example_1_clean'] = df['positive_example_1'].apply(clean_text)
    df['positive_example_2_clean'] = df['positive_example_2'].apply(clean_text)
    df['negative_example_1_clean'] = df['negative_example_1'].apply(clean_text)
    df['negative_example_2_clean'] = df['negative_example_2'].apply(clean_text)
    df['rule_clean'] = df['rule'].apply(clean_text)

# Sample of cleaned text
print("Original vs Cleaned Text Samples:")
for i in range(3):
    print(f"Original: {train_df['body'].iloc[i]}")
    print(f"Cleaned: {train_df['body_clean'].iloc[i]}")
    print('-' * 80)


def engineer_features(df):
    """
    Create features for the given dataframe that work in an offline environment.
    """
    print(f"Engineering features for dataframe with shape {df.shape}")
    
    # Basic text features
    df['body_length'] = df['body'].apply(lambda x: len(str(x)))
    df['body_word_count'] = df['body_clean'].apply(lambda x: len(str(x).split()))
    
    # URL and special character features
    df['url_count'] = df['body'].apply(lambda x: len(re.findall(r'http\S+|www\S+|https\S+', str(x))))
    df['special_char_ratio'] = df['body'].apply(lambda x: len(re.findall(r'[^\w\s]', str(x))) / (len(str(x)) + 1))
    df['caps_ratio'] = df['body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
    
    # Rule-related features
    df['rule_word_count'] = df['rule'].apply(lambda x: len(str(x).split()))
    
    # Word overlap features with positive examples
    df['overlap_pos_ex1'] = df.apply(lambda x: len(set(str(x['body_clean']).split()) & set(str(x['positive_example_1_clean']).split())) / 
                                     (len(set(str(x['body_clean']).split())) + 1), axis=1)
    df['overlap_pos_ex2'] = df.apply(lambda x: len(set(str(x['body_clean']).split()) & set(str(x['positive_example_2_clean']).split())) / 
                                     (len(set(str(x['body_clean']).split())) + 1), axis=1)
    
    # Word overlap features with negative examples
    df['overlap_neg_ex1'] = df.apply(lambda x: len(set(str(x['body_clean']).split()) & set(str(x['negative_example_1_clean']).split())) / 
                                     (len(set(str(x['body_clean']).split())) + 1), axis=1)
    df['overlap_neg_ex2'] = df.apply(lambda x: len(set(str(x['body_clean']).split()) & set(str(x['negative_example_2_clean']).split())) / 
                                     (len(set(str(x['body_clean']).split())) + 1), axis=1)
    
    # Derived overlap features
    df['overlap_pos_avg'] = (df['overlap_pos_ex1'] + df['overlap_pos_ex2']) / 2
    df['overlap_neg_avg'] = (df['overlap_neg_ex1'] + df['overlap_neg_ex2']) / 2
    df['overlap_diff'] = df['overlap_pos_avg'] - df['overlap_neg_avg']
    
    # Rule keyword presence in comment
    df['rule_keyword_presence'] = df.apply(lambda x: sum(1 for word in str(x['rule_clean']).split() 
                                                        if word in str(x['body_clean'])) / 
                                                     (len(str(x['rule_clean']).split()) + 1), axis=1)
    
    # Length ratios (comment length compared to examples)
    df['len_ratio_pos_ex1'] = df['body_length'] / (df['positive_example_1'].apply(lambda x: len(str(x))) + 1)
    df['len_ratio_pos_ex2'] = df['body_length'] / (df['positive_example_2'].apply(lambda x: len(str(x))) + 1)
    df['len_ratio_neg_ex1'] = df['body_length'] / (df['negative_example_1'].apply(lambda x: len(str(x))) + 1)
    df['len_ratio_neg_ex2'] = df['body_length'] / (df['negative_example_2'].apply(lambda x: len(str(x))) + 1)
    
    # Count specific patterns that might indicate rule violations
    df['url_density'] = df['url_count'] / (df['body_length'] + 1)
    df['exclamation_count'] = df['body'].apply(lambda x: str(x).count('!'))
    df['question_count'] = df['body'].apply(lambda x: str(x).count('?'))
    df['money_mentions'] = df['body'].apply(lambda x: len(re.findall(r'[$€£]\d+|\d+[$€£]', str(x))))
    
    # Subreddit as a categorical feature
    if 'subreddit' in df.columns:
        # Get the most common subreddits from training
        top_subreddits = df['subreddit'].value_counts().head(50).index.tolist()
        
        # One-hot encode only the top subreddits
        for subreddit in top_subreddits:
            df[f'subreddit_{subreddit}'] = (df['subreddit'] == subreddit).astype(int)
    
    print(f"Feature engineering complete. Added {len(get_feature_columns(df))} features.")
    return df

def get_feature_columns(df):
    """
    Get all numeric feature columns from the dataframe.
    Excludes row_id and rule_violation.
    """
    exclude_cols = ['row_id', 'rule_violation']
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    return feature_cols

# Apply feature engineering to both train and test
print("Applying feature engineering to training data...")
train_df = engineer_features(train_df)

print("\nApplying feature engineering to test data...")
test_df = engineer_features(test_df)

# Display some features
features = get_feature_columns(train_df)
print(f"\nTotal features: {len(features)}")
print(f"Sample features: {features[:10]}")
print("\nFeature statistics:")
train_df[features[:5]].describe()


def run_cross_validation(train_df, n_splits=5, n_estimators=500):
    """
    Run k-fold cross-validation using LightGBM.
    Returns the trained models, validation results, and feature importances.
    """
    # Get feature columns
    features = get_feature_columns(train_df)
    print(f"Using {len(features)} features for cross-validation")
    
    # Prepare data
    X = train_df[features]
    y = train_df['rule_violation']
    
    # Initialize cross-validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize arrays to store results
    models = []
    oof_predictions = np.zeros(len(train_df))
    fold_scores = []
    all_feature_importances = []
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*50}\nFold {fold+1}/{n_splits}\n{'='*50}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Initialize model
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1
        )
        
        # Train model - FIXED: Removed early_stopping_rounds parameter
        print(f"Training fold {fold+1}...")
        
        # FIX: Correctly use LightGBM's early stopping by first creating callback
        eval_result = {}
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Store model
        models.append(model)
        
        # Make predictions on validation set
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_preds
        
        # Calculate AUC for this fold
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
        
        # Store feature importances
        fold_importances = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        })
        all_feature_importances.append(fold_importances)
    
    # Calculate overall OOF AUC
    oof_auc = roc_auc_score(y, oof_predictions)
    print(f"\nOverall CV AUC: {np.mean(fold_scores):.4f} (std: {np.std(fold_scores):.4f})")
    print(f"Out-of-fold AUC: {oof_auc:.4f}")
    
    # Plot feature importances
    mean_importances = pd.concat(all_feature_importances).groupby('feature').mean().sort_values('importance', ascending=False).reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=mean_importances.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.show()
    
    return models, oof_predictions, fold_scores, mean_importances

# Run cross-validation
cv_models, oof_preds, cv_scores, feature_importances = run_cross_validation(train_df)


def train_final_model(train_df):
    """
    Train the final model on the entire training dataset.
    """
    print(f"\n{'='*50}\nTraining Final Model\n{'='*50}")
    
    # Get feature columns
    features = get_feature_columns(train_df)
    print(f"Using {len(features)} features for final model")
    
    # Prepare data
    X = train_df[features]
    y = train_df['rule_violation']
    
    # Initialize model with the best parameters from cross-validation
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=1000,  # Use more estimators for final model
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # Train model
    print("Training model on full dataset...")
    model.fit(X, y)
    print("Training complete!")
    
    # Plot feature importances
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importances (Final Model)')
    plt.tight_layout()
    plt.show()
    
    return model, features

# Train final model
final_model, model_features = train_final_model(train_df)


def evaluate_model(model, train_df, features):
    """
    Evaluate the model's performance on the training set.
    """
    print(f"\n{'='*50}\nModel Evaluation\n{'='*50}")
    
    # Prepare data
    X = train_df[features]
    y = train_df['rule_violation']
    
    # Make predictions
    print("Generating predictions for evaluation...")
    predictions = model.predict_proba(X)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y, predictions)
    
    # Calculate other metrics using threshold of 0.5
    binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
    report = classification_report(y, binary_preds)
    conf_matrix = confusion_matrix(y, binary_preds)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Violation', 'Violation'],
                yticklabels=['Non-Violation', 'Violation'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Analyze by rule
    print("\nPerformance by Rule:")
    for rule in train_df['rule'].unique():
        rule_mask = train_df['rule'] == rule
        rule_auc = roc_auc_score(y[rule_mask], predictions[rule_mask])
        print(f"Rule '{rule}': AUC = {rule_auc:.4f}")
    
    return auc, report, conf_matrix

# Evaluate the model
model_auc, classification_report, confusion_mat = evaluate_model(final_model, train_df, model_features)


def generate_predictions(model, test_df, model_features):
    """
    Generate predictions for the test set and create a submission file.
    Handles missing features by creating them or using defaults.
    """
    print(f"\n{'='*50}\nGenerating Predictions\n{'='*50}")
    
    # First ensure that all required features exist in test_df
    print(f"Checking for {len(model_features)} required features...")
    
    # Make sure we have a complete set of features in test_df
    missing_features = [feat for feat in model_features if feat not in test_df.columns]
    if missing_features:
        print(f"Creating {len(missing_features)} missing features...")
        
        # Handle each missing feature type
        for feature in missing_features:
            if feature == 'comment_length':
                # Recreate comment_length feature
                test_df['comment_length'] = test_df['body'].apply(lambda x: len(str(x)))
            
            elif feature.startswith('subreddit_'):
                # Extract subreddit name from feature name
                subreddit = feature.replace('subreddit_', '')
                # Create the missing subreddit indicator column with all zeros (not present)
                test_df[feature] = 0
            
            else:
                # For any other missing features, add with zeros as default
                print(f"Adding missing feature '{feature}' with zeros")
                test_df[feature] = 0
    
    # Prepare data - verify all features exist
    print("Verifying features...")
    for feat in model_features:
        if feat not in test_df.columns:
            raise KeyError(f"Feature {feat} still missing after attempted fix")
    
    # Get only the features needed by the model
    X_test = test_df[model_features]
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': predictions
    })
    
    # Save to CSV
    submission.to_csv('submission.csv', index=False)
    print(f"Submission file created with {len(submission)} predictions")
    
    # Display sample
    print("\nSample of predictions:")
    print(submission.head(10))
    
    # Plot distribution of predictions
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50)
    plt.title('Distribution of Predictions')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.show()
    
    # Check predictions by rule type
    print("\nPrediction statistics by rule:")
    rule_predictions = {}
    for rule in test_df['rule'].unique():
        rule_mask = test_df['rule'] == rule
        rule_preds = predictions[rule_mask]
        rule_predictions[rule] = {
            'mean': np.mean(rule_preds),
            'std': np.std(rule_preds),
            'min': np.min(rule_preds),
            'max': np.max(rule_preds),
            'count': len(rule_preds)
        }
    
    # Display rule prediction stats as a DataFrame
    rule_stats = pd.DataFrame(rule_predictions).T
    print(rule_stats)
    
    return submission

# Generate predictions with the fixed function
submission = generate_predictions(final_model, test_df, model_features)

# Verify submission format
print("\nSubmission Verification:")
print(f"Row count matches test set: {len(submission) == len(test_df)}")
print(f"All required columns present: {sorted(submission.columns.tolist()) == ['row_id', 'rule_violation']}")
print(f"No missing values: {submission.isnull().sum().sum() == 0}")

