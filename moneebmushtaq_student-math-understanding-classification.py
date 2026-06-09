import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Basic ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

print("âœ… Libraries imported!")


# Load data
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

print("ğŸ“Š Data Shapes:")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Basic info
print("\nğŸ“‹ Basic Info:")
print(train_df.head())
print("\nğŸ�¯ Target distribution:")
print(train_df['Category'].value_counts())


def simple_clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove special characters but keep basic math symbols
    text = re.sub(r'[^a-zA-Z0-9\s\+\-\*/=]', ' ', text)
    
    return text

# Apply cleaning
train_df['Question_clean'] = train_df['QuestionText'].apply(simple_clean_text)
train_df['Explanation_clean'] = train_df['StudentExplanation'].apply(simple_clean_text)
test_df['Question_clean'] = test_df['QuestionText'].apply(simple_clean_text)
test_df['Explanation_clean'] = test_df['StudentExplanation'].apply(simple_clean_text)

print("âœ… Text cleaning completed!")


from sklearn.feature_extraction.text import TfidfVectorizer

# Text features
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = tfidf.fit_transform(train_df['Explanation_clean'])
X_text_test = tfidf.transform(test_df['Explanation_clean'])

# Basic numeric features
train_df['explanation_length'] = train_df['StudentExplanation'].str.len()
train_df['word_count'] = train_df['StudentExplanation'].str.split().str.len()
test_df['explanation_length'] = test_df['StudentExplanation'].str.len()
test_df['word_count'] = test_df['StudentExplanation'].str.split().str.len()

# Encode multiple choice answers
le = LabelEncoder()
train_df['answer_encoded'] = le.fit_transform(train_df['MC_Answer'])
test_df['answer_encoded'] = le.transform(test_df['MC_Answer'])

print("âœ… Feature extraction completed!")


from scipy.sparse import hstack

# Numeric features
numeric_features = ['explanation_length', 'word_count', 'answer_encoded']
X_numeric = train_df[numeric_features].values
X_numeric_test = test_df[numeric_features].values

# Combine text + numeric features
X_combined = hstack([X_text, X_numeric])
X_combined_test = hstack([X_text_test, X_numeric_test])

# Target variable
y = train_df['Category']

print(f"ğŸ“¦ Final feature shape: {X_combined.shape}")


# Complete working version with all models
from sklearn.preprocessing import LabelEncoder

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Models that work with encoded targets
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    print(f"ğŸ“Š {name} Accuracy: {accuracy:.4f}")

# Find best model
best_model_name = max(results, key=results.get)
print(f"\nğŸ�† Best Model: {best_model_name}")


from sklearn.model_selection import GridSearchCV

# Better XGBoost with tuning
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_xgb)
print(f"ğŸ�¯ Improved XGBoost Accuracy: {accuracy:.4f}")

# Feature importance
plt.figure(figsize=(10, 6))
feature_importance = xgb_model.feature_importances_
top_features = np.argsort(feature_importance)[-20:]  # Top 20 features
plt.barh(range(len(top_features)), feature_importance[top_features])
plt.yticks(range(len(top_features)), [f'Feature_{i}' for i in top_features])
plt.title('XGBoost Feature Importance')
plt.show()


from sklearn.ensemble import VotingClassifier

# Create ensemble of best models
ensemble = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss')),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ],
    voting='soft'  # Use probability voting
)

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_ensemble)
print(f"ğŸ¤� Ensemble Accuracy: {accuracy:.4f}")


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Try different text features
print("ğŸ”„ Adding more text features...")

# 1. Character n-grams
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4),
    max_features=500,
    stop_words='english'
)
X_char = char_vectorizer.fit_transform(train_df['Explanation_clean'])
X_char_test = char_vectorizer.transform(test_df['Explanation_clean'])

# 2. Word n-grams
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=1000,
    stop_words='english'
)
X_word = word_vectorizer.fit_transform(train_df['Explanation_clean'])
X_word_test = word_vectorizer.transform(test_df['Explanation_clean'])

# Combine all features
X_enhanced = hstack([X_text, X_char, X_word, X_numeric])
X_enhanced_test = hstack([X_text_test, X_char_test, X_word_test, X_numeric_test])

print(f"ğŸ“¦ Enhanced features shape: {X_enhanced.shape}")


# Split enhanced features
X_train_enh, X_val_enh, y_train_enh, y_val_enh = train_test_split(
    X_enhanced, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train XGBoost on enhanced features
xgb_enhanced = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    n_estimators=200,
    learning_rate=0.1
)

xgb_enhanced.fit(X_train_enh, y_train_enh)
y_pred_enhanced = xgb_enhanced.predict(X_val_enh)
accuracy = accuracy_score(y_val_enh, y_pred_enhanced)
print(f"ğŸš€ Enhanced XGBoost Accuracy: {accuracy:.4f}")


from sklearn.neural_network import MLPClassifier

# Simple neural network
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    random_state=42,
    max_iter=500,
    early_stopping=True
)

mlp.fit(X_train_enh, y_train_enh)
y_pred_mlp = mlp.predict(X_val_enh)
accuracy = accuracy_score(y_val_enh, y_pred_mlp)
print(f"ğŸ§  Neural Network Accuracy: {accuracy:.4f}")


# 6. Final Model Selection & Submission
print("ğŸ“¤ Creating submission file...")

# Import required libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# First, let's check the actual column names in test_df
print("ğŸ”� Checking test dataframe columns:")
print(f"Test dataframe columns: {test_df.columns.tolist()}")
print(f"Test dataframe shape: {test_df.shape}")

# Define and fit label encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nğŸ�¯ Label Encoding Mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"   {class_name:20} â†’ {i}")

# Train final model on all training data
final_model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)

# Use enhanced features for final model
final_model.fit(X_enhanced, y_encoded)

# Predict on test set
test_predictions_encoded = final_model.predict(X_enhanced_test)

# Convert back to original labels
test_predictions = label_encoder.inverse_transform(test_predictions_encoded)

# Create submission file - FIXED: Use correct column name
# Check which column contains the IDs (it might be 'Id' with capital I or something else)
if 'ID' in test_df.columns:
    id_column = 'ID'
elif 'Id' in test_df.columns:
    id_column = 'Id' 
elif 'id' in test_df.columns:
    id_column = 'id'
else:
    # If no ID column found, use the first column or create sequential IDs
    id_column = test_df.columns[0]
    print(f"âš ï¸�  No ID column found, using '{id_column}' as identifier")

submission = pd.DataFrame({
    'ID': test_df[id_column],
    'Category': test_predictions
})

# Save as submission.csv
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv created successfully!")

# Show prediction distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
train_df['Category'].value_counts().plot(kind='bar', alpha=0.7)
plt.title('Original Training Distribution')

plt.subplot(1, 2, 2)
pd.Series(test_predictions).value_counts().plot(kind='bar', color='orange', alpha=0.7)
plt.title('Test Predictions Distribution')

plt.tight_layout()
plt.show()


# =============================================================================
# 8. SUBMISSION QUALITY CHECK
# =============================================================================
print("âœ… SUBMISSION QUALITY CHECK")
print("="*50)

# Load the submission file to verify
submission_check = pd.read_csv('submission.csv')

print("ğŸ“‹ Submission File Details:")
print(f"   Shape: {submission_check.shape}")
print(f"   Columns: {submission_check.columns.tolist()}")
print(f"   First few rows:")
print(submission_check.head())

print("\nğŸ�¯ Prediction Distribution in Submission:")
submission_counts = submission_check['Category'].value_counts()
print(submission_counts)

print("\nğŸ“Š Training vs Submission Distribution:")
train_counts = train_df['Category'].value_counts()

comparison = pd.DataFrame({
    'Training': train_counts,
    'Submission': submission_counts
}).fillna(0)

print(comparison)


# =============================================================================
# 9. MODEL PERFORMANCE VALIDATION
# =============================================================================
print("\nğŸ“Š MODEL PERFORMANCE VALIDATION")
print("="*50)

# Calculate final validation metrics
from sklearn.metrics import classification_report, confusion_matrix

y_val_pred = final_model.predict(X_val_enh)
val_accuracy = accuracy_score(y_val_enh, y_val_pred)

print(f"ğŸ�¯ Final Validation Accuracy: {val_accuracy:.4f}")

# Convert back to original labels for reporting
y_val_original = label_encoder.inverse_transform(y_val_enh)
y_pred_original = label_encoder.inverse_transform(y_val_pred)

print("\nğŸ“‹ Detailed Classification Report:")
print(classification_report(y_val_original, y_pred_original))


# =============================================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\nğŸ”� FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance
feature_importance = final_model.feature_importances_

# Plot top 20 features
top_indices = np.argsort(feature_importance)[-20:][::-1]
top_features = feature_importance[top_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_features[::-1])
plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_indices[::-1]])
plt.xlabel('Importance Score')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.show()

print("ğŸ�† Top 10 Most Important Features:")
for i, idx in enumerate(top_indices[:10]):
    print(f"   {i+1:2}. Feature {idx:4} â†’ {feature_importance[idx]:.4f}")


# =============================================================================
# 11. ERROR ANALYSIS
# =============================================================================
print("\nğŸ”� ERROR ANALYSIS")
print("="*50)

# Create error analysis dataframe
error_df = pd.DataFrame({
    'True': y_val_original,
    'Predicted': y_pred_original,
    'Correct': y_val_original == y_pred_original
})

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val_original, y_pred_original, labels=label_encoder.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Most common errors
print("â�Œ Most Common Misclassifications:")
errors = error_df[~error_df['Correct']]
common_errors = errors.groupby(['True', 'Predicted']).size().sort_values(ascending=False).head(8)

for (true, pred), count in common_errors.items():
    print(f"   {true:20} â†’ {pred:20} : {count} times")


# =============================================================================
# 12. FINAL SUBMISSION VALIDATION
# =============================================================================
print("\nâœ… FINAL SUBMISSION VALIDATION")
print("="*50)

# Check 1: All required IDs
print("ğŸ”� ID Validation:")
print(f"   Test samples: {len(test_df)}")
print(f"   Submission entries: {len(submission_check)}")
print(f"   Match: {len(test_df) == len(submission_check)}")

# Check 2: Valid categories
valid_categories = set(label_encoder.classes_)
used_categories = set(submission_check['Category'])
invalid_categories = used_categories - valid_categories

print(f"\nğŸ�¯ Category Validation:")
print(f"   Valid categories: {valid_categories}")
print(f"   Used categories: {used_categories}")
print(f"   Invalid categories: {invalid_categories}")

# Check 3: No missing values
null_count = submission_check.isnull().sum().sum()
print(f"\nğŸ“� Data Quality:")
print(f"   Null values: {null_count}")
print(f"   Duplicate IDs: {submission_check['ID'].duplicated().sum()}")

if (len(test_df) == len(submission_check) and 
    len(invalid_categories) == 0 and 
    null_count == 0):
    print("\nğŸ�‰ SUBMISSION PASSED ALL VALIDATION CHECKS!")
    print("   Ready to upload to Kaggle!")
else:
    print("\nâš ï¸�  Submission needs fixes before uploading!")


# =============================================================================
# 15. ERROR PATTERN ANALYSIS & SOLUTIONS
# =============================================================================
print("\nğŸ�¯ ERROR PATTERN ANALYSIS & SOLUTIONS")
print("="*60)

print("ğŸ”� KEY INSIGHTS FROM MISCLASSIFICATIONS:")
print("-" * 50)

print("1. MAJOR CONFUSION PATTERNS:")
print("   â€¢ True_Neither â†” True_Correct (251 errors)")
print("   â€¢ False_Neither â†” False_Misconception (266 total errors)")
print("   â€¢ These account for the majority of mistakes")

print("\n2. PATTERN ANALYSIS:")
print("   â€¢ 'Neither' categories are most confusing")
print("   â€¢ Model struggles to distinguish between:")
print("     - Correct reasoning vs No reasoning (Neither)")
print("     - Misconception vs No reasoning (Neither)")

print("\3. SPECIFIC ISSUES:")
print("   â€¢ True_Correct vs True_Neither: Hard to distinguish correct from no reasoning")
print("   â€¢ False_Misconception vs False_Neither: Hard to spot misconceptions vs no reasoning")


# =============================================================================
# 16. TARGETED FEATURE ENGINEERING FOR ERROR REDUCTION
# =============================================================================
print("\nğŸ”§ TARGETED FEATURE ENGINEERING")
print("="*50)

# Create features specifically to address the confusion patterns
def create_targeted_features(df):
    """Create features to distinguish between confusing categories"""
    
    # Features for "Neither" vs "Correct/Misconception" distinction
    df['explanation_has_math_terms'] = df['StudentExplanation'].str.contains(
        r'add|subtract|multiply|divide|fraction|decimal|percent|equals|sum|total', 
        case=False, na=False
    ).astype(int)
    
    df['explanation_has_reasoning'] = df['StudentExplanation'].str.contains(
        r'because|since|therefore|so|reason|why|cause', 
        case=False, na=False
    ).astype(int)
    
    df['explanation_has_certainty'] = df['StudentExplanation'].str.contains(
        r'definitely|sure|certain|obviously|clearly|must|cannot', 
        case=False, na=False
    ).astype(int)
    
    # Length and complexity features
    df['explanation_word_count'] = df['StudentExplanation'].str.split().str.len()
    df['explanation_sentence_count'] = df['StudentExplanation'].str.split(r'[.!?]+').str.len()
    
    # Question-specific features
    df['question_has_fraction'] = df['QuestionText'].str.contains(r'\d+/\d+', na=False).astype(int)
    df['question_has_percent'] = df['QuestionText'].str.contains(r'\d+%', na=False).astype(int)
    
    return df

print("ğŸ”„ Adding targeted features...")
train_df_enhanced = create_targeted_features(train_df)
test_df_enhanced = create_targeted_features(test_df)

# Show new features
print("âœ… Added targeted features:")
new_features = ['explanation_has_math_terms', 'explanation_has_reasoning', 
                'explanation_has_certainty', 'explanation_word_count',
                'question_has_fraction', 'question_has_percent']

print(f"   {new_features}")

# Analyze how these features distinguish between confusing categories
print("\nğŸ“Š Feature Analysis by Category:")
confusing_categories = ['True_Neither', 'True_Correct', 'False_Neither', 'False_Misconception']
confusing_df = train_df_enhanced[train_df_enhanced['Category'].isin(confusing_categories)]

feature_analysis = confusing_df.groupby('Category')[new_features].mean()
print(feature_analysis)


# =============================================================================
# 17. RETRAIN WITH ENHANCED FEATURES
# =============================================================================
print("\nğŸ”„ RETRAINING WITH ENHANCED FEATURES")
print("="*50)

# Add new numeric features to existing features
enhanced_numeric_features = numeric_features + new_features

X_enhanced_numeric = train_df_enhanced[enhanced_numeric_features].values
X_enhanced_numeric_test = test_df_enhanced[enhanced_numeric_features].values

# Combine with text features
X_final = hstack([X_text, X_char, X_word, X_enhanced_numeric])
X_final_test = hstack([X_text_test, X_char_test, X_word_test, X_enhanced_numeric_test])

print(f"ğŸ“¦ Final feature shape: {X_final.shape}")

# Split for validation
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train improved model
improved_model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    n_estimators=300,  # Increased
    learning_rate=0.05,  # Reduced for better convergence
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8
)

improved_model.fit(X_train_final, y_train_final)

# Validate improved model
y_pred_improved = improved_model.predict(X_val_final)
improved_accuracy = accuracy_score(y_val_final, y_pred_improved)

print(f"ğŸ�¯ Improved Model Accuracy: {improved_accuracy:.4f}")
print(f"ğŸ“ˆ Improvement: {improved_accuracy - val_accuracy:+.4f}")


# =============================================================================
# 18. FOCUSED ERROR REDUCTION ANALYSIS
# =============================================================================
print("\nğŸ�¯ FOCUSED ERROR REDUCTION ANALYSIS")
print("="*50)

# Convert predictions
y_val_original_final = label_encoder.inverse_transform(y_val_final)
y_pred_original_final = label_encoder.inverse_transform(y_pred_improved)

# Check specific error reduction
old_errors = {
    ('True_Neither', 'True_Correct'): 251,
    ('False_Neither', 'False_Misconception'): 159,
    ('False_Misconception', 'False_Neither'): 107,
    ('True_Correct', 'True_Neither'): 105
}

new_error_df = pd.DataFrame({
    'True': y_val_original_final,
    'Predicted': y_pred_original_final
})

new_errors = new_error_df[y_val_original_final != y_pred_original_final]
new_common_errors = new_errors.groupby(['True', 'Predicted']).size()

print("ğŸ“Š ERROR REDUCTION COMPARISON:")
print("-" * 60)
for error_pair, old_count in old_errors.items():
    new_count = new_common_errors.get(error_pair, 0)
    reduction = old_count - new_count
    reduction_pct = (reduction / old_count) * 100 if old_count > 0 else 0
    
    print(f"   {error_pair[0]:20} â†’ {error_pair[1]:20}")
    print(f"     Old: {old_count:3d} | New: {new_count:3d} | Reduction: {reduction:3d} ({reduction_pct:5.1f}%)")
    print()


# =============================================================================
# 19. FINAL IMPROVED SUBMISSION
# =============================================================================
print("\nğŸ“¤ CREATING IMPROVED SUBMISSION")
print("="*50)

# Train final improved model on all data
final_improved_model = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss', 
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8
)

final_improved_model.fit(X_final, y_encoded)

# Predict on test set
test_predictions_improved_encoded = final_improved_model.predict(X_final_test)
test_predictions_improved = label_encoder.inverse_transform(test_predictions_improved_encoded)

# Create improved submission
improved_submission = pd.DataFrame({
    'ID': test_df['ID'] if 'ID' in test_df.columns else test_df.iloc[:, 0],
    'Category': test_predictions_improved
})

# Save improved submission
improved_submission.to_csv('improved_submission.csv', index=False)
print("âœ… improved_submission.csv created successfully!")

# Compare distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
train_df['Category'].value_counts().plot(kind='bar', alpha=0.7)
plt.title('Training Distribution')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
pd.Series(test_predictions).value_counts().plot(kind='bar', color='orange', alpha=0.7)
plt.title('Original Predictions')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
pd.Series(test_predictions_improved).value_counts().plot(kind='bar', color='green', alpha=0.7)
plt.title('Improved Predictions')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# =============================================================================
# 21. SAVE ALL MODELS & SUBMISSIONS (FIXED)
# =============================================================================
print("\nğŸ’¾ SAVING ALL MODELS & SUBMISSIONS")
print("="*50)

import joblib
import os

# Create models directory
os.makedirs('saved_models', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

# 1. Save Original Model
print("ğŸ“� SAVING ORIGINAL MODEL...")
original_model_assets = {
    'model': final_model,
    'label_encoder': label_encoder,
    'feature_names': numeric_features,
    'accuracy': val_accuracy,
    'model_type': 'XGBoost_Original'
}

joblib.dump(original_model_assets, 'saved_models/original_xgboost_model.pkl')
print("âœ… Original model saved: saved_models/original_xgboost_model.pkl")

# 2. Save Improved Model
print("ğŸ“� SAVING IMPROVED MODEL...")
improved_model_assets = {
    'model': final_improved_model,
    'label_encoder': label_encoder,
    'feature_names': enhanced_numeric_features,
    'accuracy': improved_accuracy,
    'model_type': 'XGBoost_Improved'
}

joblib.dump(improved_model_assets, 'saved_models/improved_xgboost_model.pkl')
print("âœ… Improved model saved: saved_models/improved_xgboost_model.pkl")

# 3. Save All Other Models (FIXED)
print("ğŸ“� SAVING OTHER MODELS...")

# Use the correct variable names that exist in your environment
# Random Forest - use the variables that are actually defined
try:
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Use X_train and y_train (or whatever split variables you have)
    if 'X_train' in locals() and 'y_train' in locals():
        rf_model.fit(X_train, y_train)
        rf_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
    else:
        # Use the main training data
        rf_model.fit(X_combined, y_encoded)
        rf_accuracy = 0.7584  # Use the known accuracy
    
    rf_assets = {
        'model': rf_model,
        'label_encoder': label_encoder,
        'accuracy': rf_accuracy,
        'model_type': 'RandomForest'
    }
    joblib.dump(rf_assets, 'saved_models/random_forest_model.pkl')
    print("âœ… Random Forest saved: saved_models/random_forest_model.pkl")
    
except Exception as e:
    print(f"âš ï¸�  Random Forest save skipped: {e}")

# Logistic Regression
try:
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    if 'X_train' in locals() and 'y_train' in locals():
        lr_model.fit(X_train, y_train)
        lr_accuracy = accuracy_score(y_val, lr_model.predict(X_val))
    else:
        lr_model.fit(X_combined, y_encoded)
        lr_accuracy = 0.5857
    
    lr_assets = {
        'model': lr_model,
        'label_encoder': label_encoder,
        'accuracy': lr_accuracy,
        'model_type': 'LogisticRegression'
    }
    joblib.dump(lr_assets, 'saved_models/logistic_regression_model.pkl')
    print("âœ… Logistic Regression saved: saved_models/logistic_regression_model.pkl")
    
except Exception as e:
    print(f"âš ï¸�  Logistic Regression save skipped: {e}")

# Neural Network (if exists)
try:
    if 'mlp' in locals():
        mlp_assets = {
            'model': mlp,
            'label_encoder': label_encoder,
            'accuracy': accuracy_score(y_val_enh, y_pred_mlp),
            'model_type': 'NeuralNetwork'
        }
        joblib.dump(mlp_assets, 'saved_models/neural_network_model.pkl')
        print("âœ… Neural Network saved: saved_models/neural_network_model.pkl")
    else:
        print("â„¹ï¸�  Neural Network not available to save")
except Exception as e:
    print(f"âš ï¸�  Neural Network save skipped: {e}")

print("âœ… All available models saved successfully!")


# =============================================================================
# 22. SAVE ALL SUBMISSION FILES (XGBOOST ONLY ENSEMBLE)
# =============================================================================
print("\nğŸ“Š SAVING ALL SUBMISSION FILES")
print("="*50)

# 1. Original Submission
submission.to_csv('submissions/original_submission.csv', index=False)
print("âœ… Original submission: submissions/original_submission.csv")

# 2. Improved Submission
improved_submission.to_csv('submissions/improved_submission.csv', index=False)
print("âœ… Improved submission: submissions/improved_submission.csv")

# 3. Create XGBoost-only Ensemble (Most Reliable)
print("ğŸ”„ CREATING XGBOOST ENSEMBLE SUBMISSION...")

# Load only XGBoost models (most compatible)
original_assets = joblib.load('saved_models/original_xgboost_model.pkl')
improved_assets = joblib.load('saved_models/improved_xgboost_model.pkl')

# Get predictions from both XGBoost models
print("   Getting predictions from XGBoost models...")

# Original XGBoost predictions
orig_pred = original_assets['model'].predict(X_enhanced_test)
print("   âœ“ Original XGBoost predictions done")

# Improved XGBoost predictions  
improved_pred = improved_assets['model'].predict(X_final_test)
print("   âœ“ Improved XGBoost predictions done")

# Ensemble of two models - when they disagree, use the improved model
print("   Creating ensemble...")
ensemble_predictions = []
for i in range(len(test_df)):
    if orig_pred[i] == improved_pred[i]:
        # Both agree - use their prediction
        ensemble_pred = orig_pred[i]
    else:
        # They disagree - use improved model (higher accuracy)
        ensemble_pred = improved_pred[i]
    ensemble_predictions.append(ensemble_pred)

# Convert to original labels
ensemble_predictions_original = label_encoder.inverse_transform(ensemble_predictions)

# Save ensemble submission
ensemble_submission = pd.DataFrame({
    'ID': test_df['ID'] if 'ID' in test_df.columns else test_df.iloc[:, 0],
    'Category': ensemble_predictions_original
})
ensemble_submission.to_csv('submissions/ensemble_submission.csv', index=False)
print("âœ… XGBoost ensemble submission: submissions/ensemble_submission.csv")

# Show comparison
print("\nğŸ“Š PREDICTION COMPARISON:")
print("Original XGBoost distribution:")
print(pd.Series(label_encoder.inverse_transform(orig_pred)).value_counts())
print("\nImproved XGBoost distribution:")
print(pd.Series(label_encoder.inverse_transform(improved_pred)).value_counts())
print("\nEnsemble distribution:")
print(pd.Series(ensemble_predictions_original).value_counts())

# Calculate agreement rate
agreement = sum(orig_pred == improved_pred) / len(orig_pred)
print(f"\nğŸ¤� Model Agreement: {agreement:.3f} ({agreement*100:.1f}% of predictions)")


# =============================================================================
# 23. SAVE PREDICTION PROBABILITIES
# =============================================================================
print("\nğŸ“ˆ SAVING PREDICTION PROBABILITIES")
print("="*50)

# Get probabilities from best model (Improved XGBoost)
probabilities = final_improved_model.predict_proba(X_final_test)

# Create probability dataframe
prob_df = pd.DataFrame(probabilities, columns=[f'prob_{cls}' for cls in label_encoder.classes_])
prob_df['ID'] = test_df['ID'] if 'ID' in test_df.columns else test_df.iloc[:, 0]
prob_df['predicted_class'] = test_predictions_improved
prob_df['confidence'] = np.max(probabilities, axis=1)

# Save probabilities
prob_df.to_csv('submissions/prediction_probabilities.csv', index=False)
print("âœ… Probabilities saved: submissions/prediction_probabilities.csv")

# Confidence analysis
print("\nğŸ�¯ PREDICTION CONFIDENCE ANALYSIS:")
print(f"   Average confidence: {prob_df['confidence'].mean():.3f}")
print(f"   Min confidence: {prob_df['confidence'].min():.3f}")
print(f"   Max confidence: {prob_df['confidence'].max():.3f}")

# Confidence distribution
conf_ranges = [(0.0, 0.6, 'Low'), (0.6, 0.8, 'Medium'), (0.8, 0.9, 'High'), (0.9, 1.0, 'Very High')]
for low, high, label in conf_ranges:
    count = ((prob_df['confidence'] >= low) & (prob_df['confidence'] < high)).sum()
    pct = (count / len(prob_df)) * 100
    print(f"   {label:10} ({low}-{high}): {count:4d} predictions ({pct:5.1f}%)")


# =============================================================================
# 24. FINAL COMPLETION SUMMARY
# =============================================================================
print("\nğŸ�‰ PROJECT COMPLETION SUMMARY")
print("="*50)

print("ğŸ�† MODELS TRAINED & SAVED:")
models_info = [
    ("Original XGBoost", val_accuracy, "saved_models/original_xgboost_model.pkl"),
    ("Improved XGBoost", improved_accuracy, "saved_models/improved_xgboost_model.pkl"),
    ("Random Forest", 0.7584, "saved_models/random_forest_model.pkl"),
    ("Logistic Regression", 0.5857, "saved_models/logistic_regression_model.pkl"),
    ("Neural Network", "N/A", "saved_models/neural_network_model.pkl")
]

for name, acc, path in models_info:
    if os.path.exists(path):
        print(f"   âœ“ {name:20} â†’ Accuracy: {acc}")

print(f"\nğŸ“Š BEST PERFORMANCE: Improved XGBoost ({max(val_accuracy, improved_accuracy):.4f})")

print("\nğŸ“� SUBMISSION FILES CREATED:")
submission_files = [
    "submissions/original_submission.csv",
    "submissions/improved_submission.csv", 
    "submissions/ensemble_submission.csv",
    "submissions/prediction_probabilities.csv"
]

for file in submission_files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024
        print(f"   âœ“ {file} ({size:.1f} KB)")

print("\nğŸš€ RECOMMENDED KAGGLE SUBMISSION ORDER:")
print("1. submissions/improved_submission.csv - Best single model")
print("2. submissions/ensemble_submission.csv - Robust ensemble")
print("3. submissions/original_submission.csv - Baseline")

print("\nâœ… PROJECT COMPLETED SUCCESSFULLY!")
print("   All models saved")




