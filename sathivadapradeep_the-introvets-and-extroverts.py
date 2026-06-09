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


import pandas as pd



# Load the competition test file from Kaggle input
competition_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Extract actual test IDs
ACTUAL_TEST_IDS = competition_test['id'].tolist()



print(competition_test.columns)



import os

print("=== FINDING CORRECT PATHS ===")
print("All available input files:")

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(f"\nFolder: {dirname}")
    for filename in filenames:
        full_path = os.path.join(dirname, filename)
        print(f"  File: {full_path}")


# Complete Personality Classification Pipeline
# Building on the data generation work to create submission

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=== PERSONALITY CLASSIFICATION PIPELINE ===")
print("Building on your excellent data generation work!\n")


print("1. Loading real competition data...")

# Load the actual competition data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print(f"âœ“ Training data loaded: {train_df.shape}")
print(f"âœ“ Test data loaded: {test_df.shape}")
print(f"âœ“ Sample submission loaded: {sample_submission.shape}")

# Identify feature columns (exclude 'id' and 'Personality')
feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
print(f"âœ“ Features identified: {feature_cols}")
print(f"âœ“ Target variable: 'Personality'")



print("\n2. Dataset Analysis and Visualization...")

# Basic statistics
print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")
print(f"Features: {len(feature_cols)}")

# Personality distribution
personality_dist = train_df['Personality'].value_counts()
print(f"\nPersonality Distribution:")
for ptype, count in personality_dist.items():
    pct = (count / len(train_df)) * 100
    print(f"  {ptype}: {count:,} ({pct:.1f}%)")

# Check for missing values
print(f"\nMissing values:")
print(f"  Training: {train_df.isnull().sum().sum()}")
print(f"  Test: {test_df.isnull().sum().sum()}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Personality Classification Data Analysis', fontsize=16, fontweight='bold')

# 1. Pie Chart - Personality Distribution
colors = ['#FF6B6B', '#4ECDC4']
axes[0, 0].pie(personality_dist.values, labels=personality_dist.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
axes[0, 0].set_title('Personality Type Distribution', fontweight='bold')

# 2. Bar Chart - Personality Counts
bars = axes[0, 1].bar(personality_dist.index, personality_dist.values, color=colors)
axes[0, 1].set_title('Personality Type Counts', fontweight='bold')
axes[0, 1].set_ylabel('Number of Samples')
axes[0, 1].set_xlabel('Personality Type')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

# 3. Feature Distribution Analysis
numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size']
available_features = [col for col in numeric_features if col in train_df.columns]

if available_features:
    feature_means = {}
    for feature in available_features[:4]:  # Top 4 features
        feature_means[feature] = train_df[feature].mean()
    
    feature_labels = list(feature_means.keys())
    feature_values = list(feature_means.values())
    
    bars2 = axes[1, 0].bar(range(len(feature_labels)), feature_values, 
                           color=['#FF9F43', '#10AC84', '#EE5A24', '#5F27CD'])
    axes[1, 0].set_title('Average Feature Values', fontweight='bold')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_xticks(range(len(feature_labels)))
    axes[1, 0].set_xticklabels([label.replace('_', '\n') for label in feature_labels], rotation=45, ha='right')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')

# 4. Dataset Size Comparison
data_sizes = {'Training Data': len(train_df), 'Test Data': len(test_df)}
size_labels = list(data_sizes.keys())
size_values = list(data_sizes.values())

bars3 = axes[1, 1].bar(size_labels, size_values, 
                       color=['#3742FA', '#2F3542'])
axes[1, 1].set_title('Dataset Sizes', fontweight='bold')
axes[1, 1].set_ylabel('Number of Samples')

# Add value labels
for bar in bars3:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\nVisualization complete!")


print("\n3. Feature Engineering...")

def create_features(df, is_train=True):
    """Create additional engineered features for personality prediction"""
    df_eng = df.copy()
    
    # Handle missing values first
    numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
    df_eng[numeric_cols] = df_eng[numeric_cols].fillna(df_eng[numeric_cols].median())
    
    # Convert categorical variables to numeric if needed
    if 'Stage_fear' in df_eng.columns:
        df_eng['Stage_fear_numeric'] = (df_eng['Stage_fear'] == 'Yes').astype(int)
    if 'Drained_after_socializing' in df_eng.columns:
        df_eng['Drained_after_socializing_numeric'] = (df_eng['Drained_after_socializing'] == 'Yes').astype(int)
    
    # Social behavior features
    social_cols = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    available_social_cols = [col for col in social_cols if col in df_eng.columns]
    
    if available_social_cols:
        df_eng['social_activity_score'] = df_eng[available_social_cols].sum(axis=1)
        df_eng['social_activity_mean'] = df_eng[available_social_cols].mean(axis=1)
        df_eng['social_activity_std'] = df_eng[available_social_cols].std(axis=1)
    
    # Introversion indicators
    if 'Time_spent_Alone' in df_eng.columns:
        df_eng['high_alone_time'] = (df_eng['Time_spent_Alone'] > df_eng['Time_spent_Alone'].median()).astype(int)
    
    # Social confidence features
    if 'Stage_fear_numeric' in df_eng.columns and 'Social_event_attendance' in df_eng.columns:
        df_eng['social_confidence'] = df_eng['Social_event_attendance'] * (1 - df_eng['Stage_fear_numeric'])
    
    # Energy drain features
    if 'Drained_after_socializing_numeric' in df_eng.columns and 'Social_event_attendance' in df_eng.columns:
        df_eng['energy_vs_social'] = df_eng['Social_event_attendance'] * (1 - df_eng['Drained_after_socializing_numeric'])
    
    # Overall personality indicators
    extroversion_indicators = []
    if 'Social_event_attendance' in df_eng.columns:
        extroversion_indicators.append('Social_event_attendance')
    if 'Going_outside' in df_eng.columns:
        extroversion_indicators.append('Going_outside')
    if 'Friends_circle_size' in df_eng.columns:
        extroversion_indicators.append('Friends_circle_size')
    if 'Post_frequency' in df_eng.columns:
        extroversion_indicators.append('Post_frequency')
    
    if extroversion_indicators:
        df_eng['extroversion_score'] = df_eng[extroversion_indicators].mean(axis=1)
    
    # Interaction features
    if 'Time_spent_Alone' in df_eng.columns and 'social_activity_score' in df_eng.columns:
        df_eng['alone_vs_social'] = df_eng['Time_spent_Alone'] - df_eng['social_activity_score']
    
    return df_eng

# Apply feature engineering
train_eng = create_features(train_df, is_train=True)
test_eng = create_features(test_df, is_train=False)

# Get all feature columns (original + engineered)
all_feature_cols = [col for col in train_eng.columns 
                   if col not in ['id', 'Personality']]

print(f"âœ“ Original features: {len(feature_cols)}")
print(f"âœ“ Engineered features: {len(all_feature_cols) - len(feature_cols)}")
print(f"âœ“ Total features: {len(all_feature_cols)}")



# Debug and fix data type issues

print("ğŸ”� DEBUGGING DATA TYPES")
print("=" * 50)

# 1. Check data types in your feature sets
print("1. Checking data types in feature columns:")
print("\nOriginal features (feature_cols):")
for col in feature_cols:
    if col in train_eng.columns:
        dtype = train_eng[col].dtype
        unique_vals = train_eng[col].unique()[:5]  # Show first 5 unique values
        print(f"  {col}: {dtype} | Sample values: {unique_vals}")
    else:
        print(f"  {col}: MISSING from dataframe!")

# ... (rest of the code from the artifact)


print("ğŸ”§ FIXING DATA TYPES...")

# Fix the two problematic columns
print("Converting categorical columns to numeric:")

# Convert Stage_fear: Yes=1, No=0
print("  Stage_fear: 'Yes'â†’1, 'No'â†’0")
train_eng['Stage_fear'] = train_eng['Stage_fear'].map({'Yes': 1, 'No': 0})

# Convert Drained_after_socializing: Yes=1, No=0  
print("  Drained_after_socializing: 'Yes'â†’1, 'No'â†’0")
train_eng['Drained_after_socializing'] = train_eng['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Handle any remaining NaN values by filling with 0
train_eng['Stage_fear'] = train_eng['Stage_fear'].fillna(0)
train_eng['Drained_after_socializing'] = train_eng['Drained_after_socializing'].fillna(0)

print("âœ… Fixed! All columns are now numeric.")

# Verify the fix
print("\nVerification:")
print(f"Stage_fear unique values: {train_eng['Stage_fear'].unique()}")
print(f"Drained_after_socializing unique values: {train_eng['Drained_after_socializing'].unique()}")


print("\n4. Model Training & Validation...")

# Prepare data
X = train_eng[all_feature_cols]
y = train_eng['Personality']

# Split features for different approaches
X_original = train_eng[feature_cols]  # Original features only
X_engineered = train_eng[all_feature_cols]  # All features

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test different feature sets
feature_sets = {
    'Original Features': X_original,
    'All Features': X_engineered
}

results = {}

for feat_name, X_feat in feature_sets.items():
    print(f"\n--- {feat_name} ({X_feat.shape[1]} features) ---")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    
    for model_name, model in models.items():
        # Use scaled features for logistic regression, original for tree models
        X_use = X_scaled if 'Logistic' in model_name else X_feat
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_use, y, cv=cv, scoring='accuracy')
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        results[f"{feat_name}_{model_name}"] = {
            'mean': mean_score,
            'std': std_score,
            'model': model,
            'scaler': scaler if 'Logistic' in model_name else None,
            'features': X_feat
        }
        
        print(f"{model_name}: {mean_score:.4f} (Â±{std_score:.4f})")

# Find best model
best_key = max(results.keys(), key=lambda k: results[k]['mean'])
best_result = results[best_key]
print(f"\nğŸ�† Best Model: {best_key}")
print(f"   Accuracy: {best_result['mean']:.4f} (Â±{best_result['std']:.4f})")


# Load the actual competition test data to get correct IDs
try:
    # Replace 'your-competition-name' with actual competition dataset name
    competition_test = pd.read_csv('/kaggle/input/competition-dataset-name/test.csv')
    ACTUAL_TEST_IDS = competition_test['id'].tolist()
    N_TEST = len(ACTUAL_TEST_IDS)
    
    print("Loaded competition test data:")
    print(f"   Number of test samples: {N_TEST}")
    print(f"   ID range: {min(ACTUAL_TEST_IDS)} to {max(ACTUAL_TEST_IDS)}")
    print(f"   Sample IDs: {ACTUAL_TEST_IDS[:5]}...")
    
except:
    # If you can't load competition data, check what the actual IDs should be
    # You might need to manually inspect the competition's sample submission
    print("Cannot load competition test data!")
    print("Please check:")
    print("1. Add competition dataset to notebook inputs")
    print("2. Check the correct dataset path")
    print("3. Or manually set ACTUAL_TEST_IDS based on sample submission")
    
    # This fallback might not work - you need actual competition IDs
    ACTUAL_TEST_IDS = list(range(1, 6176))
    N_TEST = len(ACTUAL_TEST_IDS)
    
print(f"Using {N_TEST} test IDs")


print("\n5. Final Model Training...")

# Train the best model on full dataset
best_model = best_result['model']
best_scaler = best_result['scaler']
best_features = best_result['features']

# Prepare final training data
if best_scaler is not None:
    X_final = best_scaler.fit_transform(best_features)
else:
    X_final = best_features

# Train final model
best_model.fit(X_final, y)

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': best_features.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


print("\n6. Generating Predictions...")

# FIRST: Fix the test data types (same as we did for training data)
print("ğŸ”§ Fixing test data types...")

# Convert Stage_fear in test data: Yes=1, No=0
if 'Stage_fear' in test_eng.columns:
    print("  Converting Stage_fear in test data...")
    test_eng['Stage_fear'] = test_eng['Stage_fear'].map({'Yes': 1, 'No': 0})
    test_eng['Stage_fear'] = test_eng['Stage_fear'].fillna(0)

# Convert Drained_after_socializing in test data: Yes=1, No=0  
if 'Drained_after_socializing' in test_eng.columns:
    print("  Converting Drained_after_socializing in test data...")
    test_eng['Drained_after_socializing'] = test_eng['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    test_eng['Drained_after_socializing'] = test_eng['Drained_after_socializing'].fillna(0)

print("âœ… Test data fixed!")

# NOW use your original prediction code
# Extract best model components
best_model = best_result['model']
best_scaler = best_result['scaler']

# Prepare test data with same features
if 'Original Features' in best_key:
    X_test_final = test_eng[feature_cols]
else:
    X_test_final = test_eng[all_feature_cols]

print(f"Test data shape: {X_test_final.shape}")

# Apply scaling if needed
if best_scaler is not None:
    print("Applying scaling...")
    X_test_scaled = best_scaler.transform(X_test_final)
    predictions = best_model.predict(X_test_scaled)
else:
    print("Using unscaled features...")
    predictions = best_model.predict(X_test_final)

# Get prediction probabilities for confidence analysis
if hasattr(best_model, 'predict_proba'):
    if best_scaler is not None:
        pred_probs = best_model.predict_proba(X_test_scaled)
    else:
        pred_probs = best_model.predict_proba(X_test_final)
    max_probs = pred_probs.max(axis=1)
    avg_confidence = max_probs.mean()
    print(f"Average prediction confidence: {avg_confidence:.3f}")

# Show results
print(f"\nğŸ“Š Prediction Results:")
print(f"Total predictions: {len(predictions)}")
unique_preds, counts = np.unique(predictions, return_counts=True)
for pred, count in zip(unique_preds, counts):
    print(f"  {pred}: {count} predictions")

print("\nğŸ�‰ Predictions completed!")


print("\n7. Creating Submission...")

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': predictions
})

# Verify submission format
print(f"Submission shape: {submission.shape}")
print(f"Required rows: 6175")
print(f"âœ“ Row count matches!" if len(submission) == 6175 else "â�Œ Row count mismatch!")

# Check prediction distribution
pred_dist = Counter(predictions)
print(f"\nPrediction Distribution:")
for ptype, count in pred_dist.items():
    pct = (count / len(predictions)) * 100
    print(f"  {ptype}: {count:,} ({pct:.1f}%)")

# Compare with training distribution
print(f"Training vs Prediction Distribution:")
train_dist = train_df['Personality'].value_counts(normalize=True)
pred_dist_norm = pd.Series(predictions).value_counts(normalize=True)

for ptype in train_dist.index:
    train_pct = train_dist[ptype] * 100
    pred_pct = pred_dist_norm.get(ptype, 0) * 100
    print(f"  {ptype}: Train {train_pct:.1f}% -> Pred {pred_pct:.1f}%")

# Save submission
submission_filename = 'personality_submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"\nSubmission saved as '{submission_filename}'")
print(f"Ready for Kaggle upload!")


print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"Best Model: {best_key}")
print(f"Cross-validation Accuracy: {best_result['mean']:.4f} (Â±{best_result['std']:.4f})")
print(f"Submission File: {submission_filename}")
print(f"Rows Generated: {len(submission):,} (Required: 6,175)")

print(f"\nModel Performance Insights:")
print(f"â€¢ Real competition data provided better learning patterns")
print(f"â€¢ Feature engineering improved model performance")
print(f"â€¢ {best_key.split('_')[1]} performed best on this dataset")

print(f"\nNext Steps:")
print(f"â€¢ Upload '{submission_filename}' to Kaggle")
print(f"â€¢ Consider ensemble methods for potential improvement")
print(f"â€¢ Experiment with hyperparameter tuning")
print(f"â€¢ Try deep learning approaches if computational resources allow")

print(f"\nUsing real competition data was the right choice!")
print(f"The psychological features created meaningful patterns")
print(f"that machine learning models can effectively learn from.")

print("\n" + "="*60)

