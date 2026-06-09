

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 1: DATA LOADING AND INITIAL ANALYSIS")
print("=" * 60)

# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print("✓ Data loaded successfully!")
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Identify target column automatically
target_candidates = ['Personality', 'target', 'label', 'Target']
target_col = None

for col in target_candidates:
    if col in train_df.columns:
        target_col = col
        break

if target_col is None:
    # Use last column as target (common convention)
    target_col = train_df.columns[-1]

print(f"\n✓ Target column identified: '{target_col}'")

# Basic dataset information
print("\n" + "="*40)
print("DATASET OVERVIEW")
print("="*40)

print(f"\nColumn types:")
print(train_df.dtypes.value_counts())

print(f"\nTarget variable analysis:")
print(f"Unique values: {train_df[target_col].nunique()}")
print(f"Value counts:")
print(train_df[target_col].value_counts())
print(f"Class distribution:")
class_dist = train_df[target_col].value_counts(normalize=True)
print(class_dist)

# Check for class balance
if len(class_dist) == 2:
    balance_ratio = min(class_dist) / max(class_dist)
    print(f"Class balance ratio: {balance_ratio:.3f}")
    if balance_ratio < 0.8:
        print("  Dataset is imbalanced - consider using stratified sampling")
    else:
        print("✓ Dataset is reasonably balanced")

# Feature columns
feature_cols = [col for col in train_df.columns if col not in ['id', 'Id', 'ID', target_col]]
print(f"\nFeature columns: {len(feature_cols)}")
print(f"First 10 features: {feature_cols[:10]}")

# Missing values analysis
print("\n" + "="*40)
print("MISSING VALUES ANALYSIS")
print("="*40)

train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

train_missing_cols = train_missing[train_missing > 0].sort_values(ascending=False)
test_missing_cols = test_missing[test_missing > 0].sort_values(ascending=False)

print("Training data missing values:")
if len(train_missing_cols) > 0:
    for col, count in train_missing_cols.items():
        pct = (count / len(train_df)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
else:
    print("  No missing values!")

print("\nTest data missing values:")
if len(test_missing_cols) > 0:
    for col, count in test_missing_cols.items():
        pct = (count / len(test_df)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
else:
    print("  No missing values!")

# Data types analysis
print("\n" + "="*40)
print("DATA TYPES ANALYSIS")
print("="*40)

numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# Remove target and ID columns
numeric_cols = [col for col in numeric_cols if col not in ['id', 'Id', 'ID', target_col]]
categorical_cols = [col for col in categorical_cols if col not in ['id', 'Id', 'ID', target_col]]

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

if len(categorical_cols) > 0:
    print(f"\nCategorical features details:")
    for col in categorical_cols[:5]:  # Show first 5
        unique_count = train_df[col].nunique()
        print(f"  {col}: {unique_count} unique values")

# Basic statistics
print("\n" + "="*40)
print("BASIC STATISTICS")
print("="*40)

if len(numeric_cols) > 0:
    print("Numeric features summary:")
    print(train_df[numeric_cols].describe())

# Save processed information for next phases
print("\n" + "="*40)
print("SAVING PHASE 1 RESULTS")
print("="*40)

# Create a results dictionary for next phases
phase1_results = {
    'target_col': target_col,
    'feature_cols': feature_cols,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'train_missing': train_missing_cols.to_dict() if len(train_missing_cols) > 0 else {},
    'test_missing': test_missing_cols.to_dict() if len(test_missing_cols) > 0 else {},
    'class_distribution': class_dist.to_dict(),
    'total_features': len(feature_cols),
    'has_missing_values': len(train_missing_cols) > 0 or len(test_missing_cols) > 0
}

print("✓ Phase 1 completed successfully!")
print(f"✓ Target column: {target_col}")
print(f"✓ Total features: {len(feature_cols)}")
print(f"✓ Numeric features: {len(numeric_cols)}")
print(f"✓ Categorical features: {len(categorical_cols)}")
print(f"✓ Missing values present: {phase1_results['has_missing_values']}")

print("\n Ready for Data Visualization and EDA")
print("="*60)



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

print("=" * 60)
print("PHASE 2: DATA VISUALIZATION AND EDA")
print("=" * 60)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")



if 'phase1_results' not in globals():
    print("  results not found. Loading data again...")
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    
    # Auto-detect target column
    target_candidates = ['Personality', 'target', 'label', 'Target']
    target_col = None
    for col in target_candidates:
        if col in train_df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = train_df.columns[-1]
    
    feature_cols = [col for col in train_df.columns if col not in ['id', 'Id', 'ID', target_col]]
    numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns 
                   if col not in ['id', 'Id', 'ID', target_col]]
    categorical_cols = [col for col in train_df.select_dtypes(include=['object']).columns 
                       if col not in ['id', 'Id', 'ID', target_col]]
else:
    target_col = phase1_results['target_col']
    feature_cols = phase1_results['feature_cols']
    numeric_cols = phase1_results['numeric_cols']
    categorical_cols = phase1_results['categorical_cols']

print(f"✓ Working with {len(feature_cols)} features")
print(f"✓ Target column: {target_col}")

# 1. TARGET DISTRIBUTION VISUALIZATION
print("\n" + "="*40)
print("1. TARGET DISTRIBUTION")
print("="*40)

plt.figure(figsize=(10, 6))
target_counts = train_df[target_col].value_counts()

plt.subplot(1, 2, 1)
target_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Target Distribution (Counts)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
target_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Target Distribution (Percentage)')
plt.ylabel('')

plt.tight_layout()
plt.show()

print(f"Class distribution:")
for class_val, count in target_counts.items():
    pct = (count / len(train_df)) * 100
    print(f"  {class_val}: {count} ({pct:.1f}%)")

# 2. MISSING VALUES VISUALIZATION
print("\n" + "="*40)
print("2. MISSING VALUES PATTERN")
print("="*40)

# Combined missing values plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Training data missing values
train_missing = train_df.isnull().sum()
train_missing = train_missing[train_missing > 0].sort_values(ascending=True)

if len(train_missing) > 0:
    train_missing.plot(kind='barh', ax=axes[0], color='orange')
    axes[0].set_title('Training Data - Missing Values')
    axes[0].set_xlabel('Missing Count')
else:
    axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_title('Training Data - Missing Values')

# Test data missing values
test_missing = test_df.isnull().sum()
test_missing = test_missing[test_missing > 0].sort_values(ascending=True)

if len(test_missing) > 0:
    test_missing.plot(kind='barh', ax=axes[1], color='red')
    axes[1].set_title('Test Data - Missing Values')
    axes[1].set_xlabel('Missing Count')
else:
    axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Test Data - Missing Values')

plt.tight_layout()
plt.show()

# 3. CORRELATION ANALYSIS
print("\n" + "="*40)
print("3. CORRELATION ANALYSIS")
print("="*40)

if len(numeric_cols) > 1:
    # Prepare data for correlation (handle missing values temporarily)
    corr_data = train_df[numeric_cols + [target_col]].copy()
    
    # Quick missing value handling for correlation
    for col in corr_data.columns:
        if corr_data[col].dtype in ['object']:
            le = LabelEncoder()
            corr_data[col] = le.fit_transform(corr_data[col].astype(str))
        else:
            corr_data[col].fillna(corr_data[col].median(), inplace=True)
    
    # Encode target if categorical
    if corr_data[target_col].dtype == 'object':
        le_target = LabelEncoder()
        corr_data[target_col] = le_target.fit_transform(corr_data[target_col])
    
    correlation_matrix = corr_data.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Show top correlations with target
    target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
    print(f"\nTop 10 features correlated with target:")
    for feature, corr_val in target_corr.head(11).items():  # 11 to exclude target itself
        if feature != target_col:
            print(f"  {feature}: {corr_val:.3f}")

else:
    print("Not enough numeric features for correlation analysis")

# 4. FEATURE IMPORTANCE ANALYSIS
print("\n" + "="*40)
print("4. FEATURE IMPORTANCE ANALYSIS")
print("="*40)

# Prepare data for feature importance
X_temp = train_df[feature_cols].copy()
y_temp = train_df[target_col].copy()

# Handle missing values quickly
from sklearn.impute import SimpleImputer

numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Handle numeric columns
if len(numeric_cols) > 0:
    X_temp[numeric_cols] = numeric_imputer.fit_transform(X_temp[numeric_cols])

# Handle categorical columns
if len(categorical_cols) > 0:
    X_temp[categorical_cols] = categorical_imputer.fit_transform(X_temp[categorical_cols])
    
    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_temp[col] = le.fit_transform(X_temp[col].astype(str))

# Encode target if needed
if y_temp.dtype == 'object':
    le_target = LabelEncoder()
    y_temp = le_target.fit_transform(y_temp)

# Calculate feature importance
rf_importance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_importance.fit(X_temp, y_temp)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_importance.feature_importances_
}).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
top_20_features = feature_importance.head(20)
sns.barplot(data=top_20_features, y='feature', x='importance', palette='viridis')
plt.title('Top 20 Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("Top 10 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"  {i}. {row['feature']}: {row['importance']:.4f}")

# 5. FEATURE DISTRIBUTIONS BY TARGET
print("\n" + "="*40)
print("5. FEATURE DISTRIBUTIONS BY TARGET")
print("="*40)

# Plot distributions for top important features
top_features_for_dist = feature_importance.head(6)['feature'].tolist()

if len(top_features_for_dist) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features_for_dist):
        if i < len(axes):
            if feature in numeric_cols:
                # Numeric feature - histogram by target
                for target_val in train_df[target_col].unique():
                    subset = train_df[train_df[target_col] == target_val]
                    if not subset[feature].isna().all():
                        axes[i].hist(subset[feature].dropna(), alpha=0.6, 
                                   label=f'{target_val}', bins=20, density=True)
                axes[i].set_title(f'{feature} Distribution by Target')
                axes[i].legend()
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Density')
            else:
                # Categorical feature - count plot
                crosstab = pd.crosstab(train_df[feature], train_df[target_col])
                crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0)
                crosstab_pct.plot(kind='bar', ax=axes[i], width=0.8)
                axes[i].set_title(f'{feature} Distribution by Target')
                axes[i].legend(title=target_col)
                axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# 6. SAVE RESULTS FOR NEXT PHASE
print("\n" + "="*40)
print("SAVING PHASE 2 RESULTS")
print("="*40)

phase2_results = {
    'feature_importance': feature_importance,
    'top_10_features': feature_importance.head(10)['feature'].tolist(),
    'processed_X': X_temp,
    'processed_y': y_temp,
    'correlation_available': len(numeric_cols) > 1,
    'imputers_fitted': True
}

print("✓ Phase 2 completed successfully!")
print(f"✓ Feature importance calculated for {len(feature_cols)} features")
print(f"✓ Top feature: {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['importance']:.4f})")
print(f"✓ Correlation analysis: {'✓' if phase2_results['correlation_available'] else '✗'}")

print("\n Ready Data Preprocessing and Feature Engineering")
print("="*60)



import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 3: DATA PREPROCESSING AND FEATURE ENGINEERING")
print("=" * 60)

# Load data if previous phases weren't run
if 'train_df' not in globals():
    print("  Loading data (run previous phases for better performance)...")
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    
    # Auto-detect target column
    target_candidates = ['Personality', 'target', 'label', 'Target']
    target_col = None
    for col in target_candidates:
        if col in train_df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = train_df.columns[-1]
    
    feature_cols = [col for col in train_df.columns if col not in ['id', 'Id', 'ID', target_col]]
    numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns 
                   if col not in ['id', 'Id', 'ID', target_col]]
    categorical_cols = [col for col in train_df.select_dtypes(include=['object']).columns 
                       if col not in ['id', 'Id', 'ID', target_col]]

print(f"✓ Working with {len(feature_cols)} features")

#  MISSING VALUES HANDLING
print("\n" + "="*40)
print("STEP 1: MISSING VALUES HANDLING")
print("="*40)

def handle_missing_values(X_train, X_test, numeric_cols, categorical_cols):
    """Handle missing values consistently across train and test"""
    
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    # Handle numeric missing values
    if len(numeric_cols) > 0:
        print(f"Processing {len(numeric_cols)} numeric columns...")
        
        # Use median for numeric columns
        numeric_imputer = SimpleImputer(strategy='median')
        X_train_clean[numeric_cols] = numeric_imputer.fit_transform(X_train_clean[numeric_cols])
        X_test_clean[numeric_cols] = numeric_imputer.transform(X_test_clean[numeric_cols])
        
        print(f"✓ Numeric missing values handled with median imputation")
    
    # Handle categorical missing values
    if len(categorical_cols) > 0:
        print(f"Processing {len(categorical_cols)} categorical columns...")
        
        # Use most frequent for categorical columns
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X_train_clean[categorical_cols] = categorical_imputer.fit_transform(X_train_clean[categorical_cols])
        X_test_clean[categorical_cols] = categorical_imputer.transform(X_test_clean[categorical_cols])
        
        print(f"✓ Categorical missing values handled with mode imputation")
    
    return X_train_clean, X_test_clean, numeric_imputer if len(numeric_cols) > 0 else None, categorical_imputer if len(categorical_cols) > 0 else None

# Apply missing value handling
X_train = train_df[feature_cols].copy()
X_test = test_df[[col for col in test_df.columns if col not in ['id', 'Id', 'ID']]].copy()
y_train = train_df[target_col].copy()

print(f"Before imputation:")
print(f"  Train missing values: {X_train.isnull().sum().sum()}")
print(f"  Test missing values: {X_test.isnull().sum().sum()}")

X_train_clean, X_test_clean, num_imputer, cat_imputer = handle_missing_values(
    X_train, X_test, numeric_cols, categorical_cols
)

print(f"After imputation:")
print(f"  Train missing values: {X_train_clean.isnull().sum().sum()}")
print(f"  Test missing values: {X_test_clean.isnull().sum().sum()}")

#  CATEGORICAL ENCODING
print("\n" + "="*40)
print("STEP 2: CATEGORICAL ENCODING")
print("="*40)

def encode_categorical_features(X_train, X_test, categorical_cols):
    """Encode categorical features consistently"""
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoders = {}
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            print(f"Encoding {col}...")
            
            # Create label encoder
            le = LabelEncoder()
            
            # Fit on training data
            X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
            
            # Handle unseen categories in test data
            def safe_transform(val):
                try:
                    return le.transform([str(val)])[0]
                except ValueError:
                    # Return most frequent class for unseen values
                    return 0
            
            X_test_encoded[col] = X_test_encoded[col].apply(safe_transform)
            encoders[col] = le
            
            print(f"  ✓ {col}: {len(le.classes_)} unique categories")
    
    return X_train_encoded, X_test_encoded, encoders

X_train_encoded, X_test_encoded, category_encoders = encode_categorical_features(
    X_train_clean, X_test_clean, categorical_cols
)

# Handle target encoding
if y_train.dtype == 'object':
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    print(f"✓ Target encoded: {target_encoder.classes_}")
else:
    y_train_encoded = y_train.copy()
    target_encoder = None

print(f"✓ All categorical features encoded")

#  FEATURE SCALING
print("\n" + "="*40)
print("STEP 3: FEATURE SCALING")
print("="*40)

def scale_features(X_train, X_test, method='robust'):
    """Scale numeric features"""
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return X_train_scaled, X_test_scaled, None
    
    # Only scale numeric columns
    numeric_cols_current = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols_current) > 0:
        X_train_scaled[numeric_cols_current] = scaler.fit_transform(X_train_scaled[numeric_cols_current])
        X_test_scaled[numeric_cols_current] = scaler.transform(X_test_scaled[numeric_cols_current])
        print(f"✓ Scaled {len(numeric_cols_current)} numeric features using {method} scaling")
    
    return X_train_scaled, X_test_scaled, scaler

# Apply robust scaling (less sensitive to outliers)
X_train_scaled, X_test_scaled, feature_scaler = scale_features(
    X_train_encoded, X_test_encoded, method='robust'
)

#  FEATURE ENGINEERING
print("\n" + "="*40)
print("STEP 4: FEATURE ENGINEERING")
print("="*40)

def create_engineered_features(X_train, X_test, y_train, top_n_features=10):
    """Create engineered features"""
    
    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()
    
    # Get feature importance to identify top features
    from sklearn.ensemble import RandomForestClassifier
    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(top_n_features)['feature'].tolist()
    print(f"Top {len(top_features)} features for engineering: {top_features[:5]}...")
    
    # 1. Feature Interactions (top 5 features)
    interaction_count = 0
    top_5_features = top_features[:5]
    
    for i in range(len(top_5_features)):
        for j in range(i+1, len(top_5_features)):
            feat1, feat2 = top_5_features[i], top_5_features[j]
            new_feature = f'{feat1}_x_{feat2}'
            
            X_train_eng[new_feature] = X_train[feat1] * X_train[feat2]
            X_test_eng[new_feature] = X_test[feat1] * X_test[feat2]
            interaction_count += 1
    
    print(f"✓ Created {interaction_count} interaction features")
    
    # 2. Polynomial Features (top 3 features)
    poly_count = 0
    top_3_features = top_features[:3]
    
    for feat in top_3_features:
        if X_train[feat].min() >= 0:  # Only for non-negative features
            X_train_eng[f'{feat}_squared'] = X_train[feat] ** 2
            X_test_eng[f'{feat}_squared'] = X_test[feat] ** 2
            
            X_train_eng[f'{feat}_sqrt'] = np.sqrt(X_train[feat])
            X_test_eng[f'{feat}_sqrt'] = np.sqrt(X_test[feat])
            poly_count += 2
        else:
            X_train_eng[f'{feat}_squared'] = X_train[feat] ** 2
            X_test_eng[f'{feat}_squared'] = X_test[feat] ** 2
            poly_count += 1
    
    print(f"✓ Created {poly_count} polynomial features")
    
    # 3. Statistical Features (grouped by feature importance bins)
    if len(top_features) >= 4:
        # Create feature groups
        group1 = top_features[:len(top_features)//2]
        group2 = top_features[len(top_features)//2:]
        
        # Group statistics
        X_train_eng['top_features_sum'] = X_train[group1].sum(axis=1)
        X_test_eng['top_features_sum'] = X_test[group1].sum(axis=1)
        
        X_train_eng['top_features_mean'] = X_train[group1].mean(axis=1)
        X_test_eng['top_features_mean'] = X_test[group1].mean(axis=1)
        
        X_train_eng['top_features_std'] = X_train[group1].std(axis=1)
        X_test_eng['top_features_std'] = X_test[group1].std(axis=1)
        
        print(f"✓ Created 3 statistical features")
    
    original_features = X_train.shape[1]
    new_features = X_train_eng.shape[1]
    print(f"✓ Feature engineering complete: {original_features} -> {new_features} features")
    
    return X_train_eng, X_test_eng, feature_importance

# Apply feature engineering
X_train_engineered, X_test_engineered, feature_importance_df = create_engineered_features(
    X_train_scaled, X_test_scaled, y_train_encoded, top_n_features=8
)

#  FEATURE SELECTION
print("\n" + "="*40)
print("STEP 5: FEATURE SELECTION")
print("="*40)

def select_best_features(X_train, X_test, y_train, k=100):
    """Select k best features using statistical tests"""
    
    # Use SelectKBest with f_classif
    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Convert back to DataFrame
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    print(f"✓ Selected {len(selected_features)} best features out of {X_train.shape[1]}")
    
    return X_train_selected, X_test_selected, selector, selected_features

# Apply feature selection (keep top 80% of features)
max_features = min(100, int(X_train_engineered.shape[1] * 0.8))
X_train_final, X_test_final, feature_selector, selected_features = select_best_features(
    X_train_engineered, X_test_engineered, y_train_encoded, k=max_features
)

print(f"Selected features include: {selected_features[:5]}...")

# STEP 6: FINAL DATA VALIDATION
print("\n" + "="*40)
print("STEP 6: FINAL DATA VALIDATION")
print("="*40)

def validate_processed_data(X_train, X_test, y_train):
    """Validate the processed data"""
    
    issues = []
    
    # Check for missing values
    if X_train.isnull().sum().sum() > 0:
        issues.append("Training data has missing values")
    if X_test.isnull().sum().sum() > 0:
        issues.append("Test data has missing values")
    
    # Check for infinite values
    if np.isinf(X_train.values).sum() > 0:
        issues.append("Training data has infinite values")
    if np.isinf(X_test.values).sum() > 0:
        issues.append("Test data has infinite values")
    
    # Check data types
    if not all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        issues.append("Training data has non-numeric columns")
    if not all(X_test.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        issues.append("Test data has non-numeric columns")
    
    # Check shape consistency
    if X_train.shape[1] != X_test.shape[1]:
        issues.append("Training and test data have different number of features")
    
    # Check target variable
    if len(np.unique(y_train)) < 2:
        issues.append("Target variable has less than 2 classes")
    
    return issues

validation_issues = validate_processed_data(X_train_final, X_test_final, y_train_encoded)

if len(validation_issues) == 0:
    print(" All validation checks passed!")
else:
    print("  Validation issues found:")
    for issue in validation_issues:
        print(f"  - {issue}")

# STEP 7: SAVE PREPROCESSING RESULTS
print("\n" + "="*40)
print("SAVING PHASE 3 RESULTS")
print("="*40)

# Create comprehensive results dictionary
phase3_results = {
    # Processed data
    'X_train_final': X_train_final,
    'X_test_final': X_test_final,
    'y_train_final': y_train_encoded,
    
    # Feature information
    'selected_features': selected_features,
    'feature_importance': feature_importance_df,
    'original_feature_count': len(feature_cols),
    'final_feature_count': X_train_final.shape[1],
    
    # Preprocessing objects (for applying to new data)
    'numeric_imputer': num_imputer,
    'categorical_imputer': cat_imputer,
    'category_encoders': category_encoders,
    'target_encoder': target_encoder,
    'feature_scaler': feature_scaler,
    'feature_selector': feature_selector,
    
    # Column information
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'target_col': target_col,
    
    # Validation
    'validation_passed': len(validation_issues) == 0,
    'validation_issues': validation_issues
}

print("✓ Phase 3 completed successfully!")
print(f"✓ Final training data shape: {X_train_final.shape}")
print(f"✓ Final test data shape: {X_test_final.shape}")
print(f"✓ Features: {len(feature_cols)} -> {X_train_final.shape[1]}")
print(f"✓ Missing values: {X_train_final.isnull().sum().sum() + X_test_final.isnull().sum().sum()}")
print(f"✓ All numeric: {all(X_train_final.dtypes.apply(lambda x: np.issubdtype(x, np.number)))}")
print(f"✓ Ready for modeling: {phase3_results['validation_passed']}")

# Display final feature importance
print(f"\nTop 10 features in final dataset:")
final_feature_importance = feature_importance_df[
    feature_importance_df['feature'].isin(selected_features)
].head(10)

for i, (_, row) in enumerate(final_feature_importance.iterrows(), 1):
    print(f"  {i}. {row['feature']}: {row['importance']:.4f}")

print("\n Ready for Model Training and Selection")
print("="*60)



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Check for available advanced models
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

# Try XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("✓ XGBoost available")
except ImportError:
    print("  XGBoost not available")

# Skip LightGBM due to cupy conflict
print(" Skipping LightGBM due to dependency conflicts")

# Try CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("✓ CatBoost available")
except ImportError:
    print(" CatBoost not available")

print("=" * 60)
print("PHASE 4: MODEL TRAINING AND SELECTION")
print("=" * 60)

# Load processed data from Phase 3
if 'phase3_results' not in globals():
    print("  Phase 3 results not found. Please run Phase 3 first!")
    print("Using basic fallback...")
    
    # Basic fallback data loading
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    
    # Auto-detect target
    target_candidates = ['Personality', 'target', 'label', 'Target']
    target_col = None
    for col in target_candidates:
        if col in train_df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = train_df.columns[-1]
    
    feature_cols = [col for col in train_df.columns if col not in ['id', 'Id', 'ID', target_col]]
    
    # Basic preprocessing
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    
    X_train_final = train_df[feature_cols].copy()
    y_train_final = train_df[target_col].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    numeric_cols = X_train_final.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X_train_final[numeric_cols] = imputer.fit_transform(X_train_final[numeric_cols])
    
    # Handle categorical
    categorical_cols = X_train_final.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_final[col] = le.fit_transform(X_train_final[col].astype(str))
    
    # Handle target
    if y_train_final.dtype == 'object':
        le_target = LabelEncoder()
        y_train_final = le_target.fit_transform(y_train_final)
        
else:
    X_train_final = phase3_results['X_train_final']
    y_train_final = phase3_results['y_train_final']
    print("✓ Using processed data from Phase 3")

print(f"✓ Training data shape: {X_train_final.shape}")
print(f"✓ Target classes: {len(np.unique(y_train_final))}")

# STEP 1: DEFINE MODELS
print("\n" + "="*40)
print("STEP 1: DEFINE BASELINE MODELS")
print("="*40)

def get_baseline_models():
    """Define baseline models for comparison"""
    
    models = {}
    
    # 1. Logistic Regression
    models['Logistic Regression'] = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear'
    )
    
    # 2. Random Forest
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Extra Trees
    models['Extra Trees'] = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Gradient Boosting
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # 5. SVM (with probability for ensemble)
    models['SVM'] = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    
    # 6. K-Nearest Neighbors
    models['KNN'] = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    )
    
    # 7. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
    
    # 8. CatBoost (if available)
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
    
    return models

baseline_models = get_baseline_models()
print(f"✓ Defined {len(baseline_models)} baseline models")
for model_name in baseline_models.keys():
    print(f"  - {model_name}")

# CROSS-VALIDATION EVALUATION
print("\n" + "="*40)
print("STEP 2: CROSS-VALIDATION EVALUATION")
print("="*40)

def evaluate_models(models, X, y, cv_folds=5, scoring='roc_auc'):
    """Evaluate models using cross-validation"""
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    
    print(f"Evaluating models with {cv_folds}-fold cross-validation...")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            print(f"✓ {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results[name] = {
                'mean_score': 0.0,
                'std_score': 0.0,
                'scores': np.array([0.0]),
                'error': str(e)
            }
    
    return results

# Evaluate baseline models
baseline_results = evaluate_models(baseline_models, X_train_final, y_train_final)

# Sort results by performance
sorted_results = sorted(baseline_results.items(), 
                       key=lambda x: x[1]['mean_score'], reverse=True)

print(f"\n BASELINE MODEL RANKINGS:")
print("-" * 50)
for i, (name, result) in enumerate(sorted_results, 1):
    if 'error' not in result:
        print(f"{i}. {name}: {result['mean_score']:.4f} (+/- {result['std_score']*2:.4f})")
    else:
        print(f"{i}. {name}: FAILED - {result['error']}")

best_baseline_model = sorted_results[0][0]
best_baseline_score = sorted_results[0][1]['mean_score']
print(f"\n BEST BASELINE: {best_baseline_model} ({best_baseline_score:.4f})")

#  HYPERPARAMETER OPTIMIZATION
print("\n" + "="*40)
print("STEP 3: HYPERPARAMETER OPTIMIZATION")
print("="*40)

def optimize_top_models(models, results, X, y, top_n=3):
    """Optimize hyperparameters for top performing models"""
    
    # Get top N models
    top_models = [(name, models[name]) for name, _ in sorted_results[:top_n] 
                  if 'error' not in results[name]]
    
    optimized_models = {}
    optimization_results = {}
    
    for name, model in top_models:
        print(f"\nOptimizing {name}...")
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'Extra Trees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9]
            },
            'CatBoost': {
                'iterations': [100, 200],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        }
        
        if name in param_grids:
            param_grid = param_grids[name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            
            try:
                grid_search.fit(X, y)
                optimized_models[name] = grid_search.best_estimator_
                optimization_results[name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'improvement': grid_search.best_score_ - results[name]['mean_score']
                }
                
                print(f"  ✓ Best score: {grid_search.best_score_:.4f}")
                print(f"  ✓ Improvement: +{optimization_results[name]['improvement']:.4f}")
                print(f"  ✓ Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"  ✗ Optimization failed: {str(e)}")
                optimized_models[name] = model  # Use original model
                optimization_results[name] = {
                    'best_score': results[name]['mean_score'],
                    'best_params': {},
                    'improvement': 0.0
                }
        else:
            # No optimization available for this model
            optimized_models[name] = model
            optimization_results[name] = {
                'best_score': results[name]['mean_score'],
                'best_params': {},
                'improvement': 0.0
            }
            print(f"  No optimization grid available")
    
    return optimized_models, optimization_results

# Optimize top 3 models
optimized_models, optimization_results = optimize_top_models(
    baseline_models, baseline_results, X_train_final, y_train_final, top_n=3
)

print(f"\n OPTIMIZATION RESULTS:")
print("-" * 50)
for name, result in optimization_results.items():
    print(f"{name}:")
    print(f"  Score: {result['best_score']:.4f} (+{result['improvement']:.4f})")
    if result['best_params']:
        print(f"  Best params: {result['best_params']}")

# STEP 4: ENSEMBLE CREATION
print("\n" + "="*40)
print("STEP 4: ENSEMBLE CREATION")
print("="*40)

def create_ensemble(optimized_models, X, y, top_n=3):
    """Create voting ensemble from top models"""
    
    # Select top N models for ensemble
    model_scores = [(name, optimization_results[name]['best_score']) 
                   for name in optimized_models.keys()]
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_models_for_ensemble = model_scores[:top_n]
    print(f"Creating ensemble with top {top_n} models:")
    
    ensemble_estimators = []
    for name, score in top_models_for_ensemble:
        ensemble_estimators.append((name.lower().replace(' ', '_'), optimized_models[name]))
        print(f"  - {name}: {score:.4f}")
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft',
        n_jobs=-1
    )
    
    # Evaluate ensemble
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    ensemble_result = {
        'mean_score': ensemble_scores.mean(),
        'std_score': ensemble_scores.std(),
        'scores': ensemble_scores
    }
    
    print(f"\n ENSEMBLE PERFORMANCE:")
    print(f"  Cross-validation: {ensemble_result['mean_score']:.4f} (+/- {ensemble_result['std_score']*2:.4f})")
    
    # Compare with best individual model
    best_individual_score = max([result['best_score'] for result in optimization_results.values()])
    improvement = ensemble_result['mean_score'] - best_individual_score
    print(f"  vs Best Individual: +{improvement:.4f}")
    
    return ensemble, ensemble_result

# Create ensemble
final_ensemble, ensemble_results = create_ensemble(optimized_models, X_train_final, y_train_final)

# STEP 5: FINAL MODEL TRAINING
print("\n" + "="*40)
print("STEP 5: FINAL MODEL TRAINING")
print("="*40)

# Train final ensemble on full training data
print("Training final ensemble on complete training data...")
final_ensemble.fit(X_train_final, y_train_final)
print("✓ Final ensemble trained successfully!")

# STEP 6: SAVE MODEL RESULTS
print("\n" + "="*40)
print("SAVING PHASE 4 RESULTS")
print("="*40)

phase4_results = {
    # Models
    'final_ensemble': final_ensemble,
    'optimized_models': optimized_models,
    'baseline_models': baseline_models,
    
    # Performance results
    'baseline_results': baseline_results,
    'optimization_results': optimization_results,
    'ensemble_results': ensemble_results,
    
    # Best performances
    'best_baseline_model': best_baseline_model,
    'best_baseline_score': best_baseline_score,
    'final_ensemble_score': ensemble_results['mean_score'],
    
    # Model information
    'models_in_ensemble': len(final_ensemble.estimators_),
    'total_models_tested': len(baseline_models),
    'models_optimized': len(optimized_models)
}

print("✓ Phase 4 completed successfully!")
print(f"✓ Models tested: {len(baseline_models)}")
print(f"✓ Models optimized: {len(optimized_models)}")
print(f"✓ Best baseline: {best_baseline_model} ({best_baseline_score:.4f})")
print(f"✓ Final ensemble: {ensemble_results['mean_score']:.4f} (+/- {ensemble_results['std_score']*2:.4f})")
print(f"✓ Ensemble models: {phase4_results['models_in_ensemble']}")

improvement_over_baseline = ensemble_results['mean_score'] - best_baseline_score
print(f"✓ Total improvement: +{improvement_over_baseline:.4f}")

print("\n Ready for Phase 5: Predictions and Submission")
print("="*60)



import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("GENERATING FINAL PREDICTIONS")
print("=" * 50)

# Check if we have results from previous phases
if 'phase4_results' not in globals() or 'phase3_results' not in globals():
    print(" Previous phase results not found!")
    # Add minimal fallback if needed
    
else:
    # Use results from previous phases
    final_ensemble = phase4_results['final_ensemble']
    X_test_final = phase3_results['X_test_final']
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print(f"✓ Test data shape: {X_test_final.shape}")
print(f"✓ CV Score: {phase4_results['final_ensemble_score']:.4f}")

# Generate predictions
print("\n" + "="*30)
print("GENERATING PREDICTIONS")
print("="*30)

test_probabilities = final_ensemble.predict_proba(X_test_final)
test_predictions = test_probabilities[:, 1]

print(f"✓ Predictions generated: {len(test_predictions)}")
print(f"  Range: {test_predictions.min():.4f} to {test_predictions.max():.4f}")
print(f"  Mean: {test_predictions.mean():.4f}")

# Prediction distribution analysis
introvert_pred = (test_predictions < 0.5).sum()
extrovert_pred = (test_predictions >= 0.5).sum()
confident_pred = ((test_predictions < 0.1) | (test_predictions > 0.9)).sum()

print(f"\nPrediction distribution:")
print(f"  Introverts (< 0.5): {introvert_pred} ({introvert_pred/len(test_predictions)*100:.1f}%)")
print(f"  Extroverts (≥ 0.5): {extrovert_pred} ({extrovert_pred/len(test_predictions)*100:.1f}%)")
print(f"  High confidence: {confident_pred} ({confident_pred/len(test_predictions)*100:.1f}%)")

# Individual model analysis
if hasattr(final_ensemble, 'estimators_'):
    print(f"\nModel contributions:")
    for name, estimator in final_ensemble.named_estimators_.items():
        pred = estimator.predict_proba(X_test_final)[:, 1]
        print(f"  {name.replace('_', ' ').title()}: mean={pred.mean():.4f}, std={pred.std():.4f}")

# Create submission
print("\n" + "="*30)
print("CREATING SUBMISSION")
print("="*30)

submission = sample_submission.copy()
submission.iloc[:, 1] = test_predictions

# Generate clean filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'final_submission_{timestamp}.csv'

# Save submission
submission.to_csv(filename, index=False)

print(f"✓ Submission saved: {filename}")
print(f"✓ Shape: {submission.shape}")
print(f"✓ Validation passed: {submission.isnull().sum().sum() == 0}")

# Show preview
print(f"\nSubmission preview:")
print(submission.head())

print(f"\nREADY !")

