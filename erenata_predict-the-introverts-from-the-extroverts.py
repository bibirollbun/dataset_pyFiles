# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer, SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib and seaborn
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Try different seaborn styles
try:
    sns.set_style("whitegrid")
    sns.set_palette("husl")
except:
    pass

print("âœ… All libraries imported successfully!")
print("ğŸ“Š Ready for data analysis and modeling")



# Load the dataset with proper path handling
import os

def load_data():
    """Load data with dynamic path detection for Kaggle and local environments"""
    
    # Check if we're in Kaggle environment
    if os.path.exists('/kaggle/input/'):
        # Kaggle environment - find the correct dataset path
        kaggle_input = '/kaggle/input/'
        dataset_dirs = [d for d in os.listdir(kaggle_input) if os.path.isdir(os.path.join(kaggle_input, d))]
        
        if dataset_dirs:
            data_path = os.path.join(kaggle_input, dataset_dirs[0])
            print(f"ğŸ“� Found Kaggle dataset at: {data_path}")
        else:
            data_path = kaggle_input
            print(f"ğŸ“� Using Kaggle input directory: {data_path}")
    else:
        # Local environment
        data_path = 'playgroundseries-s5e7'
        print(f"ğŸ“� Using local path: {data_path}")
    
    try:
        # Load training and test data
        train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        
        print(f"âœ… Data loaded successfully!")
        print(f"ğŸ“Š Training data shape: {train_df.shape}")
        print(f"ğŸ“Š Test data shape: {test_df.shape}")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"â�Œ Error loading data: {e}")
        return None, None

# Load the data
train_df, test_df = load_data()



# Basic information about the dataset
print("ğŸ”� DATASET OVERVIEW")
print("=" * 50)

if train_df is not None:
    print(f"ğŸ“Š Training Data Shape: {train_df.shape}")
    print(f"ğŸ“Š Test Data Shape: {test_df.shape}")
    print(f"\nğŸ“‹ Column Names:")
    for i, col in enumerate(train_df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nğŸ“ˆ Data Types:")
    print(train_df.dtypes)
    
    print(f"\nğŸ�¯ Target Variable Distribution:")
    if 'Personality' in train_df.columns:
        target_counts = train_df['Personality'].value_counts()
        print(target_counts)
        print(f"\nPercentages:")
        print(target_counts / len(train_df) * 100)
    
    print(f"\nğŸ“Š First 5 rows:")
    print(train_df.head())
else:
    print("â�Œ No data available for analysis")



# Check for missing values and data quality issues
print("ğŸ”� DATA QUALITY ANALYSIS")
print("=" * 50)

if train_df is not None:
    print("ğŸ“Š Missing Values:")
    missing_data = train_df.isnull().sum()
    missing_percent = (missing_data / len(train_df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print(f"\nğŸ“ˆ Statistical Summary:")
    print(train_df.describe())
    
    # Check for categorical columns that might need special handling
    print(f"\nğŸ”� Checking for categorical data:")
    for col in train_df.columns:
        if col != 'id' and col != 'Personality':
            unique_vals = train_df[col].unique()
            if len(unique_vals) <= 10:  # Likely categorical
                print(f"  {col}: {unique_vals}")
                
    # Check data types and potential issues
    print(f"\nâš ï¸�  Potential Data Type Issues:")
    for col in train_df.columns:
        if col not in ['id', 'Personality']:
            non_numeric = train_df[col].apply(lambda x: not isinstance(x, (int, float, np.integer, np.floating)) and pd.notna(x))
            if non_numeric.any():
                print(f"  {col}: Contains non-numeric values")
                print(f"    Sample values: {train_df[col][non_numeric].head().tolist()}")
else:
    print("â�Œ No data available for analysis")



# Data preprocessing - Handle categorical variables properly
print("ğŸ”§ DATA PREPROCESSING")
print("=" * 50)

def preprocess_data(df, is_train=True):
    """Preprocess the data with proper categorical handling"""
    
    if df is None:
        return None
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Handle categorical columns with proper mapping
    categorical_mappings = {
        'Stage_fear': {'Yes': 1, 'No': 0},
        'Drained_after_socializing': {'Yes': 1, 'No': 0}
    }
    
    print("ğŸ”„ Processing categorical variables...")
    for col, mapping in categorical_mappings.items():
        if col in df_processed.columns:
            # Check current values
            unique_vals = df_processed[col].unique()
            print(f"  {col}: {unique_vals}")
            
            # Apply mapping
            df_processed[col] = df_processed[col].map(mapping)
            
            # Check for any unmapped values
            if df_processed[col].isnull().any():
                unmapped_count = df_processed[col].isnull().sum()
                print(f"    âš ï¸�  Warning: {unmapped_count} values couldn't be mapped")
            else:
                print(f"    âœ… Successfully mapped to: {df_processed[col].unique()}")
    
    # Handle target variable for training data
    if is_train and 'Personality' in df_processed.columns:
        print("ğŸ�¯ Processing target variable...")
        target_mapping = {'Introvert': 0, 'Extrovert': 1}
        df_processed['Personality'] = df_processed['Personality'].map(target_mapping)
        print(f"  Target distribution: {df_processed['Personality'].value_counts().to_dict()}")
    
    return df_processed

# Preprocess the data
train_processed = preprocess_data(train_df, is_train=True)
test_processed = preprocess_data(test_df, is_train=False)

if train_processed is not None:
    print(f"\nâœ… Preprocessing completed!")
    print(f"ğŸ“Š Processed training data shape: {train_processed.shape}")
    print(f"ğŸ“Š Processed test data shape: {test_processed.shape}")
    
    # Check for remaining issues
    print(f"\nğŸ”� Post-processing data quality:")
    for col in train_processed.columns:
        if col not in ['id', 'Personality']:
            null_count = train_processed[col].isnull().sum()
            if null_count > 0:
                print(f"  {col}: {null_count} missing values")
            else:
                print(f"  {col}: âœ… No missing values")
else:
    print("â�Œ Preprocessing failed")



# Target variable distribution visualization
if train_processed is not None:
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Count plot
    plt.subplot(1, 2, 1)
    personality_counts = train_processed['Personality'].value_counts()
    labels = ['Introvert', 'Extrovert']
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = plt.bar(labels, personality_counts.values, color=colors, alpha=0.8)
    plt.title('ğŸ�¯ Personality Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Personality Type')
    plt.ylabel('Count')
    
    # Add value labels on bars
    for bar, count in zip(bars, personality_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(personality_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, explode=(0.05, 0.05))
    plt.title('ğŸ¥§ Personality Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("ğŸ“Š Target Variable Statistics:")
    print(f"Total samples: {len(train_processed)}")
    print(f"Introverts: {personality_counts[0]} ({personality_counts[0]/len(train_processed)*100:.1f}%)")
    print(f"Extroverts: {personality_counts[1]} ({personality_counts[1]/len(train_processed)*100:.1f}%)")
    
    # Check for class imbalance
    ratio = min(personality_counts.values) / max(personality_counts.values)
    print(f"Class balance ratio: {ratio:.3f}")
    if ratio < 0.8:
        print("âš ï¸�  Warning: Classes are somewhat imbalanced")
    else:
        print("âœ… Classes are well balanced")
else:
    print("â�Œ No data available for visualization")



# Feature distributions by personality type
if train_processed is not None:
    # Get feature columns (excluding id and target)
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'Personality']]
    
    # Create subplots for feature distributions
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    
    for i, feature in enumerate(feature_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Check if feature is binary (0/1) or continuous
        unique_vals = train_processed[feature].dropna().unique()
        is_binary = len(unique_vals) <= 2 and all(val in [0, 1] for val in unique_vals)
        
        if is_binary:
            # For binary features, create grouped bar chart
            cross_tab = pd.crosstab(train_processed[feature], train_processed['Personality'])
            cross_tab.columns = ['Introvert', 'Extrovert']
            
            # Calculate percentages
            cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
            
            # Plot grouped bar chart
            x = np.arange(len(cross_tab.index))
            width = 0.35
            
            plt.bar(x - width/2, cross_tab_pct['Introvert'], width, 
                   label='Introvert', color='#FF6B6B', alpha=0.8)
            plt.bar(x + width/2, cross_tab_pct['Extrovert'], width,
                   label='Extrovert', color='#4ECDC4', alpha=0.8)
            
            plt.xlabel(feature)
            plt.ylabel('Percentage (%)')
            plt.title(f'ğŸ“Š {feature} Distribution by Personality')
            plt.xticks(x, ['No (0)', 'Yes (1)'] if len(unique_vals) == 2 else [str(val) for val in sorted(unique_vals)])
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            for j, (intro_pct, extro_pct) in enumerate(zip(cross_tab_pct['Introvert'], cross_tab_pct['Extrovert'])):
                plt.text(j - width/2, intro_pct + 1, f'{intro_pct:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
                plt.text(j + width/2, extro_pct + 1, f'{extro_pct:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
        else:
            # For continuous features, create histograms
            intro_data = train_processed[train_processed['Personality'] == 0][feature].dropna()
            extro_data = train_processed[train_processed['Personality'] == 1][feature].dropna()
            
            plt.hist(intro_data, bins=20, alpha=0.7, label='Introvert', color='#FF6B6B', density=True)
            plt.hist(extro_data, bins=20, alpha=0.7, label='Extrovert', color='#4ECDC4', density=True)
            
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.title(f'ğŸ“Š {feature} Distribution by Personality')
            plt.legend()
            plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics for each feature
    print("ğŸ“Š FEATURE STATISTICS BY PERSONALITY TYPE")
    print("=" * 60)
    
    for feature in feature_cols:
        print(f"\nğŸ”� {feature}:")
        intro_stats = train_processed[train_processed['Personality'] == 0][feature].describe()
        extro_stats = train_processed[train_processed['Personality'] == 1][feature].describe()
        
        print(f"  Introvert - Mean: {intro_stats['mean']:.2f}, Std: {intro_stats['std']:.2f}")
        print(f"  Extrovert - Mean: {extro_stats['mean']:.2f}, Std: {extro_stats['std']:.2f}")
        print(f"  Difference: {abs(intro_stats['mean'] - extro_stats['mean']):.2f}")
        
else:
    print("â�Œ No data available for feature analysis")



# Correlation analysis - Fixed version
if train_processed is not None:
    # Get feature columns (excluding id and target)
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'Personality']]
    
    # Create correlation matrix including target variable
    correlation_data = train_processed[feature_cols + ['Personality']].copy()
    
    # Handle missing values before correlation calculation
    print("ğŸ”� Handling missing values for correlation analysis...")
    missing_before = correlation_data.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"  Missing values before imputation: {missing_before}")
        
        # Use simple imputation for correlation analysis
        imputer = SimpleImputer(strategy='median')
        correlation_data[feature_cols] = imputer.fit_transform(correlation_data[feature_cols])
        
        missing_after = correlation_data.isnull().sum().sum()
        print(f"  Missing values after imputation: {missing_after}")
    
    # Calculate correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('ğŸ”¥ Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Print strongest correlations with target
    print("ğŸ�¯ CORRELATIONS WITH PERSONALITY TYPE")
    print("=" * 50)
    
    target_correlations = correlation_matrix['Personality'].drop('Personality').abs().sort_values(ascending=False)
    
    for feature, corr in target_correlations.items():
        direction = "ğŸ“ˆ Positive" if correlation_matrix['Personality'][feature] > 0 else "ğŸ“‰ Negative"
        strength = "ğŸ”¥ Strong" if abs(corr) > 0.5 else "ğŸŸ¡ Moderate" if abs(corr) > 0.3 else "ğŸ”µ Weak"
        print(f"  {feature}: {corr:.3f} ({direction}, {strength})")
    
    # Find highly correlated feature pairs
    print(f"\nğŸ”� HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.5)")
    print("=" * 50)
    
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_val))
    
    if high_corr_pairs:
        for feat1, feat2, corr in high_corr_pairs:
            direction = "ğŸ“ˆ Positive" if corr > 0 else "ğŸ“‰ Negative"
            print(f"  {feat1} â†” {feat2}: {corr:.3f} ({direction})")
    else:
        print("  âœ… No highly correlated feature pairs found")
    
else:
    print("â�Œ No data available for correlation analysis")



# Feature Engineering - Create meaningful new features
def create_engineered_features(df):
    """Create engineered features for better model performance"""
    
    if df is None:
        return None
    
    df_eng = df.copy()
    
    # Get feature columns (excluding id and target)
    feature_cols = [col for col in df_eng.columns if col not in ['id', 'Personality']]
    
    print("ğŸ”§ Creating engineered features...")
    
    # 1. Social Activity Balance
    # Combination of social event attendance and going outside
    if 'Social_event_attendance' in df_eng.columns and 'Going_outside' in df_eng.columns:
        df_eng['Social_Activity_Balance'] = (df_eng['Social_event_attendance'] + df_eng['Going_outside']) / 2
        print("  âœ… Social_Activity_Balance created")
    
    # 2. Introversion Score
    # Higher score indicates more introverted behavior
    introversion_features = []
    if 'Time_spent_Alone' in df_eng.columns:
        introversion_features.append('Time_spent_Alone')
    if 'Stage_fear' in df_eng.columns:
        introversion_features.append('Stage_fear')
    if 'Drained_after_socializing' in df_eng.columns:
        introversion_features.append('Drained_after_socializing')
    
    if len(introversion_features) >= 2:
        df_eng['Introversion_Score'] = df_eng[introversion_features].mean(axis=1)
        print("  âœ… Introversion_Score created")
    
    # 3. Social Engagement Level
    # Combination of social features
    social_features = []
    if 'Social_event_attendance' in df_eng.columns:
        social_features.append('Social_event_attendance')
    if 'Friends_circle_size' in df_eng.columns:
        social_features.append('Friends_circle_size')
    if 'Post_frequency' in df_eng.columns:
        social_features.append('Post_frequency')
    
    if len(social_features) >= 2:
        # Normalize features to 0-1 scale for combination
        for feat in social_features:
            if df_eng[feat].max() > 1:
                df_eng[f'{feat}_normalized'] = df_eng[feat] / df_eng[feat].max()
            else:
                df_eng[f'{feat}_normalized'] = df_eng[feat]
        
        normalized_social = [f'{feat}_normalized' for feat in social_features]
        df_eng['Social_Engagement_Level'] = df_eng[normalized_social].mean(axis=1)
        
        # Clean up temporary normalized columns
        df_eng.drop(columns=normalized_social, inplace=True)
        print("  âœ… Social_Engagement_Level created")
    
    # 4. Behavioral Consistency Score
    # Measure how consistent someone's behavior is across different social contexts
    if 'Social_event_attendance' in df_eng.columns and 'Going_outside' in df_eng.columns:
        df_eng['Behavioral_Consistency'] = 1 - abs(df_eng['Social_event_attendance'] - df_eng['Going_outside'])
        print("  âœ… Behavioral_Consistency created")
    
    # Print summary of engineered features
    new_features = [col for col in df_eng.columns if col not in df.columns]
    print(f"\nğŸ“Š Created {len(new_features)} new features:")
    for feat in new_features:
        print(f"  - {feat}")
    
    return df_eng

# Apply feature engineering
print("ğŸ”§ FEATURE ENGINEERING")
print("=" * 50)

train_engineered = create_engineered_features(train_processed)
test_engineered = create_engineered_features(test_processed)

if train_engineered is not None:
    print(f"\nâœ… Feature engineering completed!")
    print(f"ğŸ“Š Original features: {train_processed.shape[1] - 2}")  # -2 for id and target
    print(f"ğŸ“Š Total features after engineering: {train_engineered.shape[1] - 2}")
    
    # Check for any missing values in engineered features
    engineered_features = [col for col in train_engineered.columns if col not in train_processed.columns]
    
    if engineered_features:
        print(f"\nğŸ”� Checking engineered features for missing values:")
        for feat in engineered_features:
            missing_count = train_engineered[feat].isnull().sum()
            if missing_count > 0:
                print(f"  {feat}: {missing_count} missing values")
            else:
                print(f"  {feat}: âœ… No missing values")
else:
    print("â�Œ Feature engineering failed")



# Visualize engineered features - Fixed version
if train_engineered is not None:
    # Get all engineered features
    engineered_features = [col for col in train_engineered.columns if col not in train_processed.columns]
    
    if engineered_features:
        print("ğŸ“Š ENGINEERED FEATURES VISUALIZATION")
        print("=" * 50)
        
        # Create subplots for engineered features
        n_features = len(engineered_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(14, 5 * n_rows))
        
        for i, feature in enumerate(engineered_features):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Get data for introverts and extroverts
            intro_data = train_engineered[train_engineered['Personality'] == 0][feature].dropna()
            extro_data = train_engineered[train_engineered['Personality'] == 1][feature].dropna()
            
            # Check if we have data
            if len(intro_data) > 0 and len(extro_data) > 0:
                # Create histograms
                plt.hist(intro_data, bins=20, alpha=0.7, label='Introvert', color='#FF6B6B', density=True)
                plt.hist(extro_data, bins=20, alpha=0.7, label='Extrovert', color='#4ECDC4', density=True)
                
                plt.xlabel(feature)
                plt.ylabel('Density')
                plt.title(f'ğŸ“Š {feature} Distribution')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Add statistics text
                intro_mean = intro_data.mean()
                extro_mean = extro_data.mean()
                diff = abs(intro_mean - extro_mean)
                
                plt.text(0.02, 0.98, f'Intro: {intro_mean:.3f}\nExtro: {extro_mean:.3f}\nDiff: {diff:.3f}', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title(f'ğŸ“Š {feature} Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Print feature statistics
        print(f"\nğŸ“Š ENGINEERED FEATURE STATISTICS")
        print("=" * 50)
        
        for feature in engineered_features:
            if feature in train_engineered.columns:
                # Check if feature has valid data
                valid_data = train_engineered[feature].dropna()
                if len(valid_data) > 0:
                    intro_stats = train_engineered[train_engineered['Personality'] == 0][feature].describe()
                    extro_stats = train_engineered[train_engineered['Personality'] == 1][feature].describe()
                    
                    print(f"\nğŸ”� {feature}:")
                    print(f"  Introvert - Mean: {intro_stats['mean']:.3f}, Std: {intro_stats['std']:.3f}")
                    print(f"  Extrovert - Mean: {extro_stats['mean']:.3f}, Std: {extro_stats['std']:.3f}")
                    print(f"  Difference: {abs(intro_stats['mean'] - extro_stats['mean']):.3f}")
                    
                    # Calculate correlation with target
                    corr_with_target = train_engineered[[feature, 'Personality']].corr().iloc[0, 1]
                    print(f"  Correlation with target: {corr_with_target:.3f}")
                else:
                    print(f"\nğŸ”� {feature}: No valid data available")
    else:
        print("â�Œ No engineered features found")
else:
    print("â�Œ No data available for engineered feature visualization")



# Prepare data for modeling with robust preprocessing
def prepare_modeling_data(train_df, test_df):
    """Prepare data for modeling with comprehensive preprocessing"""
    
    if train_df is None or test_df is None:
        return None, None, None, None
    
    print("ğŸ”§ PREPARING DATA FOR MODELING")
    print("=" * 50)
    
    # Get feature columns (excluding id and target)
    feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
    
    # Separate features and target
    X = train_df[feature_cols].copy()
    y = train_df['Personality'].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"ğŸ“Š Features: {len(feature_cols)}")
    print(f"ğŸ“Š Training samples: {len(X)}")
    print(f"ğŸ“Š Test samples: {len(X_test)}")
    
    # Handle missing values with robust imputation
    print(f"\nğŸ”� Handling missing values...")
    
    # Check for missing values
    train_missing = X.isnull().sum()
    test_missing = X_test.isnull().sum()
    
    if train_missing.sum() > 0 or test_missing.sum() > 0:
        print(f"  Training missing values: {train_missing.sum()}")
        print(f"  Test missing values: {test_missing.sum()}")
        
        # Try KNN imputation first, fall back to simple imputation
        try:
            print("  Trying KNN imputation...")
            knn_imputer = KNNImputer(n_neighbors=5)
            X_imputed = knn_imputer.fit_transform(X)
            X_test_imputed = knn_imputer.transform(X_test)
            
            X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index)
            X_test = pd.DataFrame(X_test_imputed, columns=feature_cols, index=X_test.index)
            print("  âœ… KNN imputation successful")
            
        except Exception as e:
            print(f"  âš ï¸�  KNN imputation failed: {e}")
            print("  Using simple imputation instead...")
            
            simple_imputer = SimpleImputer(strategy='median')
            X_imputed = simple_imputer.fit_transform(X)
            X_test_imputed = simple_imputer.transform(X_test)
            
            X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index)
            X_test = pd.DataFrame(X_test_imputed, columns=feature_cols, index=X_test.index)
            print("  âœ… Simple imputation successful")
    else:
        print("  âœ… No missing values found")
    
    # Feature scaling
    print(f"\nğŸ“� Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    print("  âœ… Feature scaling completed")
    
    # Final data quality check
    print(f"\nâœ… Data preparation completed!")
    print(f"ğŸ“Š Final X shape: {X_scaled.shape}")
    print(f"ğŸ“Š Final X_test shape: {X_test_scaled.shape}")
    print(f"ğŸ“Š Target distribution: {y.value_counts().to_dict()}")
    
    return X_scaled, X_test_scaled, y, feature_cols

# Prepare the data
X_train, X_test, y_train, feature_columns = prepare_modeling_data(train_engineered, test_engineered)



# Train and evaluate multiple models
def train_evaluate_models(X_train, y_train):
    """Train and evaluate multiple models with cross-validation"""
    
    if X_train is None or y_train is None:
        return None
    
    print("ğŸ¤– TRAINING MULTIPLE MODELS")
    print("=" * 50)
    
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    results = {}
    trained_models = {}
    
    print("ğŸ”„ Training models with 5-fold cross-validation...")
    
    for name, model in models.items():
        print(f"\nğŸ�¯ Training {name}...")
        
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            # Train on full dataset
            model.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            trained_models[name] = model
            
            print(f"  âœ… CV Accuracy: {cv_scores.mean():.4f} (Â±{cv_scores.std():.4f})")
            
        except Exception as e:
            print(f"  â�Œ Error training {name}: {e}")
    
    return results, trained_models

# Train models
if X_train is not None:
    model_results, trained_models = train_evaluate_models(X_train, y_train)
    
    if model_results:
        print(f"\nğŸ“Š MODEL PERFORMANCE SUMMARY")
        print("=" * 50)
        
        # Sort models by performance
        sorted_results = sorted(model_results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        
        for name, results in sorted_results:
            print(f"{name:20s}: {results['cv_mean']:.4f} (Â±{results['cv_std']:.4f})")
        
        # Get best model
        best_model_name = sorted_results[0][0]
        best_model = trained_models[best_model_name]
        best_score = sorted_results[0][1]['cv_mean']
        
        print(f"\nğŸ�† Best Model: {best_model_name} (Accuracy: {best_score:.4f})")
    else:
        print("â�Œ No model results available")
else:
    print("â�Œ No training data available")



# Create an ensemble model for better performance
def create_ensemble_model(trained_models, model_results):
    """Create an ensemble of the best performing models"""
    
    if not trained_models or not model_results:
        return None
    
    print("ğŸ�­ CREATING ENSEMBLE MODEL")
    print("=" * 50)
    
    # Select top 3 models for ensemble
    sorted_results = sorted(model_results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
    top_models = sorted_results[:3]
    
    print("ğŸ�† Selected models for ensemble:")
    for name, results in top_models:
        print(f"  {name}: {results['cv_mean']:.4f}")
    
    # Create ensemble
    ensemble_models = [(name, trained_models[name]) for name, _ in top_models]
    
    ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting='soft'  # Use probability voting
    )
    
    # Train ensemble
    print(f"\nğŸ”„ Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"âœ… Ensemble CV Accuracy: {ensemble_scores.mean():.4f} (Â±{ensemble_scores.std():.4f})")
    
    return ensemble, ensemble_scores

# Create ensemble model
if 'model_results' in locals() and 'trained_models' in locals():
    ensemble_model, ensemble_scores = create_ensemble_model(trained_models, model_results)
    
    if ensemble_model:
        print(f"\nğŸ�¯ FINAL MODEL COMPARISON")
        print("=" * 50)
        
        # Compare best individual model vs ensemble
        best_individual_score = max(model_results.values(), key=lambda x: x['cv_mean'])['cv_mean']
        ensemble_score = ensemble_scores.mean()
        
        print(f"Best Individual Model: {best_individual_score:.4f}")
        print(f"Ensemble Model:       {ensemble_score:.4f}")
        
        if ensemble_score > best_individual_score:
            print("ğŸ�† Ensemble model performs better!")
            final_model = ensemble_model
            final_score = ensemble_score
        else:
            print("ğŸ�† Individual model performs better!")
            final_model = trained_models[best_model_name]
            final_score = best_individual_score
        
        print(f"\nğŸ�¯ Final Model Score: {final_score:.4f}")
    else:
        print("â�Œ Failed to create ensemble model")
        final_model = trained_models[best_model_name] if 'best_model_name' in locals() else None
        final_score = best_score if 'best_score' in locals() else None
else:
    print("â�Œ No trained models available for ensemble creation")



# Visualize model performance
if 'model_results' in locals() and model_results:
    print("ğŸ“Š MODEL PERFORMANCE VISUALIZATION")
    print("=" * 50)
    
    # Create performance comparison plot
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Model comparison
    plt.subplot(2, 2, 1)
    model_names = list(model_results.keys())
    cv_means = [results['cv_mean'] for results in model_results.values()]
    cv_stds = [results['cv_std'] for results in model_results.values()]
    
    # Sort by performance
    sorted_indices = np.argsort(cv_means)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    cv_means = [cv_means[i] for i in sorted_indices]
    cv_stds = [cv_stds[i] for i in sorted_indices]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(range(len(model_names)), cv_means, yerr=cv_stds, 
                   color=colors, alpha=0.8, capsize=5)
    
    plt.title('ğŸ�† Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Cross-Validation Accuracy')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Cross-validation scores distribution
    plt.subplot(2, 2, 2)
    best_model_name = model_names[0]
    best_cv_scores = model_results[best_model_name]['cv_scores']
    
    plt.hist(best_cv_scores, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(best_cv_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {best_cv_scores.mean():.3f}')
    plt.title(f'ğŸ“ˆ CV Scores Distribution\n({best_model_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 3: Feature importance (if available)
    plt.subplot(2, 2, 3)
    if hasattr(trained_models[best_model_name], 'feature_importances_'):
        importances = trained_models[best_model_name].feature_importances_
        feature_names = feature_columns
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features
        
        plt.bar(range(len(indices)), importances[indices], color='lightcoral', alpha=0.8)
        plt.title('ğŸ”� Top 10 Feature Importances', fontsize=12, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('ğŸ”� Feature Importance', fontsize=12, fontweight='bold')
    
    # Subplot 4: Model complexity vs performance
    plt.subplot(2, 2, 4)
    complexity_scores = []
    complexity_names = []
    
    for name, results in model_results.items():
        complexity_scores.append(results['cv_mean'])
        complexity_names.append(name)
    
    plt.scatter(range(len(complexity_names)), complexity_scores, 
               s=100, c=colors, alpha=0.7, edgecolors='black')
    
    for i, name in enumerate(complexity_names):
        plt.annotate(name, (i, complexity_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('ğŸ“Š Model Performance Overview', fontsize=12, fontweight='bold')
    plt.xlabel('Model Index')
    plt.ylabel('CV Accuracy')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed performance analysis
    print(f"\nğŸ“Š DETAILED PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    for name, results in model_results.items():
        print(f"\nğŸ”� {name}:")
        print(f"  Mean CV Accuracy: {results['cv_mean']:.4f}")
        print(f"  Std CV Accuracy:  {results['cv_std']:.4f}")
        print(f"  CV Scores: {[f'{score:.4f}' for score in results['cv_scores']]}")
        
        # Performance category
        if results['cv_mean'] >= 0.97:
            category = "ğŸ”¥ Excellent"
        elif results['cv_mean'] >= 0.95:
            category = "ğŸŸ¢ Very Good"
        elif results['cv_mean'] >= 0.90:
            category = "ğŸŸ¡ Good"
        else:
            category = "ğŸ”´ Needs Improvement"
        
        print(f"  Performance: {category}")
        
else:
    print("â�Œ No model results available for visualization")



# Generate predictions and create submission file
def generate_predictions(model, X_test, test_df):
    """Generate predictions and create submission file"""
    
    if model is None or X_test is None:
        return None
    
    print("ğŸ�¯ GENERATING PREDICTIONS")
    print("=" * 50)
    
    # Make predictions
    predictions = model.predict(X_test)
    prediction_probs = model.predict_proba(X_test)
    
    # Convert predictions back to original labels
    prediction_labels = ['Introvert' if pred == 0 else 'Extrovert' for pred in predictions]
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': prediction_labels
    })
    
    print(f"âœ… Generated {len(predictions)} predictions")
    print(f"ğŸ“Š Prediction distribution:")
    pred_counts = pd.Series(prediction_labels).value_counts()
    for label, count in pred_counts.items():
        percentage = (count / len(predictions)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Show prediction confidence
    confidence_scores = np.max(prediction_probs, axis=1)
    print(f"\nğŸ�¯ Prediction Confidence:")
    print(f"  Mean confidence: {confidence_scores.mean():.3f}")
    print(f"  Min confidence:  {confidence_scores.min():.3f}")
    print(f"  Max confidence:  {confidence_scores.max():.3f}")
    
    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print(f"\nğŸ’¾ Submission file saved as 'submission.csv'")
    
    return submission, predictions, prediction_probs

# Generate predictions
if 'final_model' in locals() and final_model is not None and X_test is not None:
    submission, predictions, pred_probs = generate_predictions(final_model, X_test, test_engineered)
    
    if submission is not None:
        print(f"\nğŸ“‹ SUBMISSION PREVIEW:")
        print(submission.head(10))
        
        # Visualize prediction distribution
        plt.figure(figsize=(10, 6))
        
        # Subplot 1: Prediction distribution
        plt.subplot(1, 2, 1)
        pred_counts = submission['Personality'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = plt.bar(pred_counts.index, pred_counts.values, color=colors, alpha=0.8)
        plt.title('ğŸ�¯ Test Set Predictions', fontsize=14, fontweight='bold')
        plt.xlabel('Personality Type')
        plt.ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, pred_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Prediction confidence distribution
        plt.subplot(1, 2, 2)
        confidence_scores = np.max(pred_probs, axis=1)
        
        plt.hist(confidence_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(confidence_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {confidence_scores.mean():.3f}')
        plt.title('ğŸ�¯ Prediction Confidence', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("â�Œ Failed to generate predictions")
else:
    print("â�Œ No final model or test data available for predictions")


