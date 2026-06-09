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


# ================================================================
# ğŸ�† MAP CHAMPIONSHIP DATA PREPARATION & ANALYSIS SYSTEM
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

print("ğŸ�† MAP CHAMPIONSHIP DATA ANALYSIS SYSTEM")
print("=" * 50)

# ================================================================
# 1. LOAD COMPETITION DATA
# ================================================================

train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample_sub = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
print(f"\nColumns in training data: {train_df.columns.tolist()}")

# ================================================================
# 2. CRITICAL DATA ANALYSIS
# ================================================================

print("\nğŸ�¯ CRITICAL COMPETITION INSIGHTS")
print("=" * 40)

# Analyze target distribution
print("1. CATEGORY DISTRIBUTION:")
category_counts = train_df['Category'].value_counts()
print(category_counts)
print(f"Total unique categories: {len(category_counts)}")

print("\n2. MISCONCEPTION ANALYSIS:")
misconception_counts = train_df['Misconception'].value_counts()
print(f"Total unique misconceptions: {len(misconception_counts)}")
print("\nTop 10 misconceptions:")
print(misconception_counts.head(10))

# Create target labels for training
train_df['target'] = train_df['Category'] + ':' + train_df['Misconception'].fillna('NA')
target_counts = train_df['target'].value_counts()
print(f"\nTotal unique Category:Misconception combinations: {len(target_counts)}")

# Analyze explanation lengths
train_df['explanation_length'] = train_df['StudentExplanation'].str.len()
test_df['explanation_length'] = test_df['StudentExplanation'].str.len()

print(f"\n3. EXPLANATION LENGTH ANALYSIS:")
print(f"Train explanation length - Mean: {train_df['explanation_length'].mean():.1f}, Max: {train_df['explanation_length'].max()}")
print(f"Test explanation length - Mean: {test_df['explanation_length'].mean():.1f}, Max: {test_df['explanation_length'].max()}")

# ================================================================
# 3. ADVANCED MATHEMATICAL FEATURE ENGINEERING
# ================================================================

def extract_mathematical_features(text):
    """Extract mathematical reasoning features from student explanations"""
    if pd.isna(text):
        return {
            'has_numbers': False,
            'has_operators': False,
            'has_fractions': False,
            'has_decimals': False,
            'math_complexity': 0,
            'has_negative': False,
            'has_comparison': False,
            'has_calculation': False
        }
    
    text = str(text).lower()
    
    # Mathematical pattern detection
    features = {
        'has_numbers': bool(re.search(r'\d+', text)),
        'has_operators': bool(re.search(r'[\+\-\*\/\=]', text)),
        'has_fractions': bool(re.search(r'\d+\/\d+', text)),
        'has_decimals': bool(re.search(r'\d+\.\d+', text)),
        'has_negative': 'negative' in text or '-' in text,
        'has_comparison': any(word in text for word in ['greater', 'less', 'bigger', 'smaller', 'larger']),
        'has_calculation': any(word in text for word in ['calculate', 'compute', 'solve', 'add', 'subtract', 'multiply', 'divide']),
        'math_complexity': len(re.findall(r'[\+\-\*\/\=\(\)]', text))
    }
    
    return features

print("\nğŸ”¬ EXTRACTING MATHEMATICAL FEATURES...")

# Extract features for train
math_features_train = train_df['StudentExplanation'].apply(extract_mathematical_features)
math_features_df_train = pd.DataFrame(list(math_features_train))

# Extract features for test  
math_features_test = test_df['StudentExplanation'].apply(extract_mathematical_features)
math_features_df_test = pd.DataFrame(list(math_features_test))

print("Mathematical feature extraction completed!")
print("\nMathematical feature distribution (Training):")
for col in math_features_df_train.columns:
    if col != 'math_complexity':
        print(f"{col}: {math_features_df_train[col].sum()} ({math_features_df_train[col].mean()*100:.1f}%)")

# Combine with original data
train_enhanced = pd.concat([train_df, math_features_df_train], axis=1)
test_enhanced = pd.concat([test_df, math_features_df_test], axis=1)

print(f"\nEnhanced training data shape: {train_enhanced.shape}")
print(f"Enhanced test data shape: {test_enhanced.shape}")

# ================================================================
# 4. CREATE IS_CORRECT HEURISTIC (COMPETITIVE INTELLIGENCE)
# ================================================================

def create_is_correct_heuristic(train_df):
    """Create is_correct feature based on True/False category analysis"""
    
    # Find most common correct answers per question
    true_samples = train_df[train_df['Category'].str.startswith('True')].copy()
    
    if len(true_samples) > 0:
        true_samples['count'] = true_samples.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
        most_popular_correct = true_samples.sort_values('count', ascending=False).drop_duplicates(['QuestionId'])
        correct_lookup = most_popular_correct[['QuestionId', 'MC_Answer']].copy()
        correct_lookup['is_correct_flag'] = True
        
        return correct_lookup
    else:
        return pd.DataFrame(columns=['QuestionId', 'MC_Answer', 'is_correct_flag'])

# Create is_correct heuristic
correct_lookup = create_is_correct_heuristic(train_enhanced)
print(f"\nğŸ“Š Created is_correct lookup with {len(correct_lookup)} entries")

# Apply to test data
test_with_correct = test_enhanced.merge(correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
test_with_correct['is_correct'] = test_with_correct['is_correct_flag'].notna()

print(f"Test data with is_correct: {test_with_correct['is_correct'].sum()} correct out of {len(test_with_correct)} samples")

# ================================================================
# 5. FINAL DATA PROCESSING & EXPORT
# ================================================================

# Save processed data
train_enhanced.to_csv('train_enhanced.csv', index=False)
test_with_correct.to_csv('test_enhanced.csv', index=False)
correct_lookup.to_csv('correct_lookup.csv', index=False)

# Create label encoder for targets
le = LabelEncoder()
train_enhanced['label'] = le.fit_transform(train_enhanced['target'])
np.save('label_encoder_classes.npy', le.classes_)

print("\nâœ… DATA PREPARATION COMPLETE!")
print(f"Unique targets: {len(le.classes_)}")
print(f"Files saved: train_enhanced.csv, test_enhanced.csv, correct_lookup.csv, label_encoder_classes.npy")

# ================================================================
# 6. CHAMPIONSHIP DATA SUMMARY
# ================================================================

print("\nğŸ�† CHAMPIONSHIP PREPARATION SUMMARY:")
print(f"â€¢ Training samples: {len(train_enhanced):,}")
print(f"â€¢ Test samples: {len(test_with_correct):,}")
print(f"â€¢ Unique misconceptions: {len(le.classes_):,}")
print(f"â€¢ Mathematical features extracted: {len(math_features_df_train.columns)}")
print(f"â€¢ Is_correct samples in test: {test_with_correct['is_correct'].sum()}")

# Display sample of processed data
print("\nğŸ“‹ SAMPLE OF ENHANCED DATA:")
sample_display = train_enhanced[['QuestionText', 'StudentExplanation', 'target', 'has_numbers', 'has_operators', 'math_complexity']].head()
display(sample_display)

print("\nğŸ�¯ READY FOR MODEL TRAINING!")
print("Next step: Create training notebook with mathematical reasoning models")


