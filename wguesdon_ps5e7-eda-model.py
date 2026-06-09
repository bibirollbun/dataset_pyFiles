# Import Libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

try:
    from IPython.display import display
    JUPYTER_ENV = True
except ImportError:
    def display(df):
        """Fallback display function for non-Jupyter environments"""
        print(df.to_string() if hasattr(df, 'to_string') else df)
    JUPYTER_ENV = False

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set matplotlib style - using a compatible style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
    
sns.set_palette("husl")

# Custom color palette for personality types
COLORS = {
    'Introvert': '#3498db',
    'Extrovert': '#e74c3c',
    'primary': '#2c3e50',
    'secondary': '#95a5a6'
}

print("âœ… Libraries imported successfully")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Environment: {'Jupyter' if JUPYTER_ENV else 'Standard Python'}")


# Load Data
def load_data(base_path='/kaggle/input/playground-series-s5e7/'):
    """Load competition datasets"""
    print("ğŸ“‚ Loading datasets...")
    
    train_df = pd.read_csv(f'{base_path}train.csv')
    test_df = pd.read_csv(f'{base_path}test.csv')
    sample_sub = pd.read_csv(f'{base_path}sample_submission.csv')
    
    print(f"âœ… Train set loaded: {train_df.shape[0]:,} rows, {train_df.shape[1]:,} columns")
    print(f"âœ… Test set loaded: {test_df.shape[0]:,} rows, {test_df.shape[1]:,} columns")
    print(f"âœ… Sample submission loaded: {sample_sub.shape[0]:,} rows")
    
    return train_df, test_df, sample_sub

# Load the data
train_df, test_df, sample_sub = load_data()

# Initial Data Overview
def display_initial_overview(train_df, test_df):
    """Display initial data overview"""
    print("\n" + "="*80)
    print("ğŸ“‹ DATASET OVERVIEW")
    print("="*80)
    
    print("\nğŸ”� Train Dataset Shape:", train_df.shape)
    print("ğŸ”� Test Dataset Shape:", test_df.shape)
    print(f"ğŸ”� Train/Test Ratio: {len(train_df)/len(test_df):.2f}")
    
    print("\nğŸ“Š First 5 rows of training data:")
    display(train_df.head())
    
    print("\nğŸ“Š Column Information:")
    train_df.info()
    
    print("\nğŸ“Š Statistical Summary:")
    display(train_df.describe())
    
    print("\nğŸ�¯ Target Distribution:")
    target_dist = train_df['Personality'].value_counts()
    print(target_dist)
    print(f"\nClass Balance: {target_dist.values[1]/target_dist.values[0]:.2%} ratio")

display_initial_overview(train_df, test_df)



# Data Quality Assessment
def data_quality_check(df, dataset_name="Dataset"):
    """Comprehensive data quality check"""
    print(f"\n{'='*80}")
    print(f"ğŸ”� DATA QUALITY CHECK: {dataset_name}")
    print('='*80)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    print("\nğŸ“Œ Missing Values:")
    if missing.sum() == 0:
        print("âœ… No missing values found!")
    else:
        missing_df = pd.DataFrame({
            'Column': missing[missing > 0].index,
            'Missing_Count': missing[missing > 0].values,
            'Percentage': missing_pct[missing > 0].values
        }).sort_values('Percentage', ascending=False)
        display(missing_df)
    
    # Data types
    print("\nğŸ“Œ Data Types Distribution:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Duplicates
    print("\nğŸ“Œ Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"  Found {duplicates} duplicate rows ({duplicates/len(df)*100:.2%} of data)")
    
    # Unique values
    print("\nğŸ“Œ Unique Values per Column:")
    unique_counts = df.nunique().sort_values()
    display(pd.DataFrame({
        'Column': unique_counts.index,
        'Unique_Values': unique_counts.values,
        'Percentage': (unique_counts.values / len(df)) * 100
    }))
    
    # Memory usage
    print("\nğŸ“Œ Memory Usage:")
    memory_usage = df.memory_usage(deep=True) / 1024**2  # Convert to MB
    print(f"  Total: {memory_usage.sum():.2f} MB")
    print(f"  Average per column: {memory_usage.mean():.2f} MB")
    
    return missing, duplicates

# Check data quality for both datasets
train_missing, train_duplicates = data_quality_check(train_df, "Training Set")
test_missing, test_duplicates = data_quality_check(test_df, "Test Set")


# Identify Feature Types
def identify_feature_types(df):
    """Identify and categorize features"""
    print("\n" + "="*80)
    print("ğŸ�·ï¸� FEATURE CATEGORIZATION")
    print("="*80)
    
    # Separate features by type
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove 'id' and 'Personality' from feature lists
    if 'id' in numeric_features:
        numeric_features.remove('id')
    if 'Personality' in categorical_features:
        categorical_features.remove('Personality')
    
    # Check for binary features
    binary_features = []
    multi_categorical = []
    
    for col in df.columns:
        if col not in ['id', 'Personality']:
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                binary_features.append(col)
            elif unique_vals < 10 and col not in numeric_features:
                multi_categorical.append(col)
    
    print(f"\nğŸ“Š Numeric Features ({len(numeric_features)}): {numeric_features[:5]}..." if len(numeric_features) > 5 else numeric_features)
    print(f"\nğŸ“Š Categorical Features ({len(categorical_features)}): {categorical_features}")
    print(f"\nğŸ“Š Binary Features ({len(binary_features)}): {binary_features[:5]}..." if len(binary_features) > 5 else binary_features)
    print(f"\nğŸ“Š Multi-Categorical Features ({len(multi_categorical)}): {multi_categorical}")
    
    return numeric_features, categorical_features, binary_features, multi_categorical

numeric_features, categorical_features, binary_features, multi_categorical = identify_feature_types(train_df)


# Target Distribution Visualization
def visualize_target_distribution(df):
    """Visualize target variable distribution"""
    print("\n" + "="*80)
    print("ğŸ�¯ TARGET VARIABLE ANALYSIS")
    print("="*80)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    target_counts = df['Personality'].value_counts()
    
    # Bar plot
    ax1 = axes[0]
    bars = ax1.bar(target_counts.index, target_counts.values, 
                   color=[COLORS['Introvert'], COLORS['Extrovert']])
    ax1.set_xlabel('Personality Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Target Distribution (Count)', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, target_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(target_counts.values, 
                                        labels=target_counts.index,
                                        colors=[COLORS['Introvert'], COLORS['Extrovert']],
                                        autopct='%1.1f%%',
                                        startangle=90)
    ax2.set_title('Target Distribution (Percentage)', fontsize=12, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.suptitle('Personality Type Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print("\nğŸ“Š Target Distribution Statistics:")
    print(f"  Introverts: {target_counts.get('Introvert', 0):,} ({target_counts.get('Introvert', 0)/len(df)*100:.2f}%)")
    print(f"  Extroverts: {target_counts.get('Extrovert', 0):,} ({target_counts.get('Extrovert', 0)/len(df)*100:.2f}%)")
    
    # Check class imbalance
    imbalance_ratio = min(target_counts.values) / max(target_counts.values)
    print(f"\nâš–ï¸� Class Imbalance Ratio: {imbalance_ratio:.3f}")
    if imbalance_ratio < 0.5:
        print("  âš ï¸� Warning: Significant class imbalance detected!")
    else:
        print("  âœ… Classes are relatively balanced")

visualize_target_distribution(train_df)


# Numerical Features Distribution
def analyze_numerical_features(df, features, target_col='Personality'):
    """Analyze numerical features distribution"""
    if not features:
        print("No numerical features to analyze")
        return
    
    print("\n" + "="*80)
    print("ğŸ“Š NUMERICAL FEATURES ANALYSIS")
    print("="*80)
    
    # Select subset of features for visualization (max 12)
    viz_features = features[:12] if len(features) > 12 else features
    
    # Distribution plots
    n_cols = 3
    n_rows = (len(viz_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for idx, feature in enumerate(viz_features):
        ax = axes[idx]
        
        # Plot distributions for each personality type
        for personality in df[target_col].unique():
            data = df[df[target_col] == personality][feature].dropna()
            ax.hist(data, alpha=0.6, label=personality, bins=30, 
                   color=COLORS.get(personality, 'gray'), edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(viz_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Numerical Features Distribution by Personality Type', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Statistical comparison
    print("\nğŸ“Š Statistical Tests (t-test) for Numerical Features:")
    print("-" * 60)
    
    results = []
    for feature in features[:20]:  # Analyze top 20 features
        introvert_data = df[df[target_col] == 'Introvert'][feature].dropna()
        extrovert_data = df[df[target_col] == 'Extrovert'][feature].dropna()
        
        # Perform t-test
        t_stat, p_value = ttest_ind(introvert_data, extrovert_data)
        
        # Calculate effect size (Cohen's d)
        cohens_d = (introvert_data.mean() - extrovert_data.mean()) / \
                   np.sqrt(((len(introvert_data)-1)*introvert_data.std()**2 + 
                           (len(extrovert_data)-1)*extrovert_data.std()**2) / 
                          (len(introvert_data) + len(extrovert_data) - 2))
        
        results.append({
            'Feature': feature,
            'Introvert_Mean': introvert_data.mean(),
            'Extrovert_Mean': extrovert_data.mean(),
            'T-Statistic': t_stat,
            'P-Value': p_value,
            'Cohens_D': cohens_d,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    results_df = pd.DataFrame(results).sort_values('P-Value')
    display(results_df.head(10))
    
    return results_df

if numeric_features:
    numerical_analysis_results = analyze_numerical_features(train_df, numeric_features)

# Box Plots for Numerical Features
def create_boxplots(df, features, target_col='Personality'):
    """Create box plots for numerical features"""
    if not features:
        return
    
    # Select top features for visualization
    viz_features = features[:9] if len(features) > 9 else features
    
    n_cols = 3
    n_rows = (len(viz_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for idx, feature in enumerate(viz_features):
        ax = axes[idx]
        
        # Prepare data for box plot
        introvert_data = df[df[target_col] == 'Introvert'][feature].dropna()
        extrovert_data = df[df[target_col] == 'Extrovert'][feature].dropna()
        
        # Create box plot
        bp = ax.boxplot([introvert_data, extrovert_data], 
                        labels=['Introvert', 'Extrovert'],
                        patch_artist=True,
                        notch=True,
                        showmeans=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor(COLORS['Introvert'])
        bp['boxes'][1].set_facecolor(COLORS['Extrovert'])
        
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(viz_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Box Plots: Feature Distribution by Personality Type', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

if numeric_features:
    create_boxplots(train_df, numeric_features)


# Correlation Matrix
def analyze_correlations(df, features, target_col='Personality'):
    """Analyze feature correlations"""
    print("\n" + "="*80)
    print("ğŸ”„ CORRELATION ANALYSIS")
    print("="*80)
    
    # Encode target variable for correlation
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded[target_col + '_encoded'] = le.fit_transform(df_encoded[target_col])
    
    # Select features for correlation
    corr_features = features[:20] if len(features) > 20 else features
    corr_features_with_target = corr_features + [target_col + '_encoded']
    
    # Calculate correlation matrix
    corr_matrix = df_encoded[corr_features_with_target].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Find features most correlated with target
    target_corr = corr_matrix[target_col + '_encoded'].abs().sort_values(ascending=False)[1:11]
    
    print("\nğŸ“Š Top 10 Features Correlated with Personality:")
    print("-" * 50)
    for feature, corr_value in target_corr.items():
        print(f"  {feature}: {corr_value:.4f}")
    
    # Find highly correlated feature pairs
    print("\nğŸ”� Highly Correlated Feature Pairs (|corr| > 0.7):")
    print("-" * 50)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            if target_col + '_encoded' not in [feat1, feat2]:
                print(f"  {feat1} <-> {feat2}: {corr_val:.4f}")
    else:
        print("  No highly correlated feature pairs found")
    
    return corr_matrix, target_corr

if numeric_features:
    correlation_matrix, target_correlations = analyze_correlations(train_df, numeric_features)

# Comprehensive Feature Comparison
def create_feature_comparison_dashboard(df, top_features, target_col='Personality'):
    """Create a comprehensive comparison dashboard for top features"""
    if not top_features:
        return
    
    print("\n" + "="*80)
    print("ğŸ“Š COMPREHENSIVE FEATURE COMPARISON DASHBOARD")
    print("="*80)
    
    # Select top 4 features for detailed analysis
    features_to_analyze = top_features[:4]
    
    fig, axes = plt.subplots(len(features_to_analyze), 4, figsize=(16, 4*len(features_to_analyze)))
    
    if len(features_to_analyze) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, feature in enumerate(features_to_analyze):
        # Distribution plot
        ax1 = axes[idx, 0]
        for personality in df[target_col].unique():
            data = df[df[target_col] == personality][feature].dropna()
            ax1.hist(data, alpha=0.6, label=personality, bins=20, 
                    color=COLORS.get(personality, 'gray'), edgecolor='black', linewidth=0.5)
        ax1.set_title(f'{feature}\nDistribution', fontsize=9, fontweight='bold')
        ax1.set_xlabel('Value', fontsize=8)
        ax1.set_ylabel('Frequency', fontsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[idx, 1]
        introvert_data = df[df[target_col] == 'Introvert'][feature].dropna()
        extrovert_data = df[df[target_col] == 'Extrovert'][feature].dropna()
        bp = ax2.boxplot([introvert_data, extrovert_data], 
                         labels=['Introvert', 'Extrovert'],
                         patch_artist=True, notch=True)
        bp['boxes'][0].set_facecolor(COLORS['Introvert'])
        bp['boxes'][1].set_facecolor(COLORS['Extrovert'])
        ax2.set_title(f'{feature}\nBox Plot', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Value', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Violin plot
        ax3 = axes[idx, 2]
        parts = ax3.violinplot([introvert_data, extrovert_data],
                               positions=[1, 2], widths=0.7,
                               showmeans=True, showmedians=True)
        colors = [COLORS['Introvert'], COLORS['Extrovert']]
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['Introvert', 'Extrovert'], fontsize=8)
        ax3.set_title(f'{feature}\nViolin Plot', fontsize=9, fontweight='bold')
        ax3.set_ylabel('Value', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Statistics table
        ax4 = axes[idx, 3]
        ax4.axis('off')
        
        # Calculate statistics
        stats_data = []
        stats_data.append(['Metric', 'Introvert', 'Extrovert'])
        stats_data.append(['Mean', f'{introvert_data.mean():.3f}', f'{extrovert_data.mean():.3f}'])
        stats_data.append(['Median', f'{introvert_data.median():.3f}', f'{extrovert_data.median():.3f}'])
        stats_data.append(['Std Dev', f'{introvert_data.std():.3f}', f'{extrovert_data.std():.3f}'])
        stats_data.append(['Min', f'{introvert_data.min():.3f}', f'{extrovert_data.min():.3f}'])
        stats_data.append(['Max', f'{introvert_data.max():.3f}', f'{extrovert_data.max():.3f}'])
        
        # T-test
        t_stat, p_value = ttest_ind(introvert_data, extrovert_data)
        stats_data.append(['T-test p-val', f'{p_value:.4f}', ''])
        
        # Create table
        table = ax4.table(cellText=stats_data, 
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.35, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Color header row
        for i in range(3):
            table[(0, i)].set_facecolor('#2c3e50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title(f'{feature}\nStatistics', fontsize=9, fontweight='bold')
    
    plt.suptitle('Feature Comparison Dashboard: Top Features Analysis', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

# Create dashboard for top correlated features
if numeric_features and 'target_correlations' in locals():
    top_correlated_features = target_correlations.index[:4].tolist()
    create_feature_comparison_dashboard(train_df, top_correlated_features)


# Mutual Information
def calculate_feature_importance(df, features, target_col='Personality'):
    """Calculate feature importance using mutual information"""
    print("\n" + "="*80)
    print("ğŸ�¯ FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Prepare data
    X = df[features].fillna(0)
    y = LabelEncoder().fit_transform(df[target_col])
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': features,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    # Visualize top features using matplotlib
    top_features = importance_df.head(15)
    
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(top_features)), top_features['MI_Score'].values, 
                    color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features))))
    
    # Customize the plot
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Mutual Information Score', fontsize=11, fontweight='bold')
    plt.ylabel('Feature Name', fontsize=11, fontweight='bold')
    plt.title('Top 15 Features by Mutual Information Score', fontsize=13, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features['MI_Score'].values)):
        plt.text(value, bar.get_y() + bar.get_height()/2, f'{value:.4f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nğŸ“Š Top 10 Most Important Features:")
    print("-" * 50)
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['MI_Score']:.4f}")
    
    return importance_df

if numeric_features:
    feature_importance = calculate_feature_importance(train_df, numeric_features)


# Outlier Analysis
def detect_outliers(df, features, target_col='Personality'):
    """Detect outliers using IQR method"""
    print("\n" + "="*80)
    print("ğŸ”� OUTLIER DETECTION ANALYSIS")
    print("="*80)
    
    outlier_summary = []
    
    for feature in features[:20]:  # Analyze top 20 features
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        # Outliers by personality type
        outlier_by_type = outliers[target_col].value_counts().to_dict()
        
        outlier_summary.append({
            'Feature': feature,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': outlier_pct,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Introvert_Outliers': outlier_by_type.get('Introvert', 0),
            'Extrovert_Outliers': outlier_by_type.get('Extrovert', 0)
        })
    
    outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Percentage', ascending=False)
    
    print("\nğŸ“Š Features with Most Outliers:")
    display(outlier_df.head(10))
    
    # Visualize outliers for top features
    viz_features = outlier_df.head(6)['Feature'].tolist()
    
    if viz_features:
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(viz_features):
            ax = axes[idx]
            
            # Prepare data
            introvert_data = df[df[target_col] == 'Introvert'][feature].dropna()
            extrovert_data = df[df[target_col] == 'Extrovert'][feature].dropna()
            
            # Create violin plot using matplotlib
            parts = ax.violinplot([introvert_data, extrovert_data],
                                  positions=[1, 2], 
                                  widths=0.7,
                                  showmeans=True, 
                                  showmedians=True,
                                  showextrema=True)
            
            # Color the violin plots
            colors = [COLORS['Introvert'], COLORS['Extrovert']]
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Customize other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(1)
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Introvert', 'Extrovert'])
            ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(len(viz_features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Outlier Distribution by Personality Type (Violin Plots)', 
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
    
    return outlier_df

if numeric_features:
    outlier_analysis = detect_outliers(train_df, numeric_features)


# Train vs Test Distribution
def compare_train_test_distributions(train_df, test_df, features):
    """Compare distributions between train and test sets"""
    print("\n" + "="*80)
    print("ğŸ”„ TRAIN vs TEST DISTRIBUTION COMPARISON")
    print("="*80)
    
    # Statistical tests
    distribution_tests = []
    
    for feature in features[:15]:  # Check top 15 features
        if feature in test_df.columns:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(
                train_df[feature].dropna(),
                test_df[feature].dropna()
            )
            
            distribution_tests.append({
                'Feature': feature,
                'Train_Mean': train_df[feature].mean(),
                'Test_Mean': test_df[feature].mean(),
                'Train_Std': train_df[feature].std(),
                'Test_Std': test_df[feature].std(),
                'KS_Statistic': ks_stat,
                'P_Value': ks_pvalue,
                'Similar_Distribution': 'Yes' if ks_pvalue > 0.05 else 'No'
            })
    
    dist_df = pd.DataFrame(distribution_tests).sort_values('P_Value')
    
    print("\nğŸ“Š Distribution Similarity Test Results:")
    display(dist_df)
    
    # Visualize distributions
    viz_features = dist_df.head(6)['Feature'].tolist()
    
    if viz_features:
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(viz_features):
            ax = axes[idx]
            
            # Get data
            train_data = train_df[feature].dropna()
            test_data = test_df[feature].dropna()
            
            # Create histograms with KDE overlay
            # Plot histograms
            n_bins = 30
            ax.hist(train_data, bins=n_bins, alpha=0.5, 
                   label='Train', color='#3498db', density=True, edgecolor='black', linewidth=0.5)
            ax.hist(test_data, bins=n_bins, alpha=0.5, 
                   label='Test', color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)
            
            # Add KDE curves for smoother comparison
            from scipy.stats import gaussian_kde
            try:
                kde_train = gaussian_kde(train_data)
                kde_test = gaussian_kde(test_data)
                
                x_range = np.linspace(min(train_data.min(), test_data.min()),
                                     max(train_data.max(), test_data.max()), 100)
                
                ax.plot(x_range, kde_train(x_range), color='#2980b9', linewidth=2, 
                       label='Train KDE', linestyle='--')
                ax.plot(x_range, kde_test(x_range), color='#c0392b', linewidth=2, 
                       label='Test KDE', linestyle='--')
            except:
                pass  # Skip KDE if it fails
            
            # Get p-value for annotation
            p_value = dist_df[dist_df['Feature'] == feature]['P_Value'].values[0]
            
            ax.set_title(f'{feature}\n(p-value: {p_value:.4f})', fontsize=10, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(viz_features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Train vs Test Feature Distributions (KS Test)', 
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
    
    # Check for potential data drift
    drift_features = dist_df[dist_df['Similar_Distribution'] == 'No']['Feature'].tolist()
    if drift_features:
        print(f"\nâš ï¸� Warning: Potential data drift detected in {len(drift_features)} features:")
        for feat in drift_features[:5]:
            print(f"  - {feat}")
    else:
        print("\nâœ… No significant data drift detected between train and test sets")
    
    return dist_df

if numeric_features:
    distribution_comparison = compare_train_test_distributions(train_df, test_df, numeric_features)


# Generate Summary Report
def generate_summary_report(train_df, test_df, numeric_features, feature_importance):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("ğŸ“� EDA SUMMARY REPORT")
    print("="*80)
    
    print("\nğŸ�¯ KEY FINDINGS:")
    print("-" * 50)
    
    # Dataset characteristics
    print("\n1ï¸�âƒ£ Dataset Characteristics:")
    print(f"   â€¢ Training samples: {len(train_df):,}")
    print(f"   â€¢ Test samples: {len(test_df):,}")
    print(f"   â€¢ Total features: {len(train_df.columns) - 2}")  # Excluding id and target
    print(f"   â€¢ Numerical features: {len(numeric_features)}")
    
    # Target distribution
    target_dist = train_df['Personality'].value_counts()
    print("\n2ï¸�âƒ£ Target Distribution:")
    print(f"   â€¢ Introverts: {target_dist.get('Introvert', 0):,} ({target_dist.get('Introvert', 0)/len(train_df)*100:.1f}%)")
    print(f"   â€¢ Extroverts: {target_dist.get('Extrovert', 0):,} ({target_dist.get('Extrovert', 0)/len(train_df)*100:.1f}%)")
    
    # Top features
    if feature_importance is not None and len(feature_importance) > 0:
        print("\n3ï¸�âƒ£ Most Important Features (by MI Score):")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   â€¢ {row['Feature']}: {row['MI_Score']:.4f}")


# Generate final summary
generate_summary_report(train_df, test_df, numeric_features, 
                       feature_importance if 'feature_importance' in locals() else None)


"""Personality Type Prediction Model Training and Ensemble Pipeline."""

# ============================================================================
# IMPORTS
# ============================================================================

from typing import Dict, List, Tuple, Optional, Any
import warnings
import shutil
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.special import logit
from koolbox import Trainer

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class CFG:
    """Configuration class for model training parameters."""
    
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    original_path = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv"
    
    target = 'Personality'
    n_folds = 5
    seed = 42
    
    cv = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    metric = accuracy_score
    
    n_optuna_trials = 500

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_submission(
    name: str,
    X_test: pd.DataFrame,
    test_pred_probs: np.ndarray,
    score: float,
    threshold: float = 0.5
) -> pd.DataFrame:
    """Save model predictions to submission file.
    
    Args:
        name: Model name for file naming.
        X_test: Test features DataFrame.
        test_pred_probs: Predicted probabilities for test set.
        score: Model score for file naming.
        threshold: Probability threshold for classification.
    
    Returns:
        DataFrame with first few rows of submission.
    """
    sub = pd.read_csv(CFG.sample_sub_path)
    sub[CFG.target] = (test_pred_probs > threshold).astype(int)
    sub.loc[X_test.match_p == 0, CFG.target] = 1
    sub.loc[X_test.match_p == 1, CFG.target] = 0
    sub[CFG.target] = sub[CFG.target].map({0: "Extrovert", 1: "Introvert"})
    sub.to_csv(f'sub_{name}_{score:.6f}.csv', index=False)
    return sub.head()


def plot_weights(weights: np.ndarray, title: str, model_names: List[str]) -> None:
    """Plot model weights or coefficients.
    
    Args:
        weights: Array of model weights.
        title: Plot title.
        model_names: List of model names.
    """
    sorted_indices = np.argsort(weights[0])[::-1]
    sorted_coeffs = np.array(weights[0])[sorted_indices]
    sorted_model_names = np.array(model_names)[sorted_indices]

    plt.figure(figsize=(10, weights.shape[1] * 0.4))
    ax = sns.barplot(x=sorted_coeffs, y=sorted_model_names, palette="RdYlGn_r")

    for i, (value, name) in enumerate(zip(sorted_coeffs, sorted_model_names)):
        if value >= 0:
            ax.text(value, i, f'{value:.3f}', va='center', ha='left', color='black')
        else:
            ax.text(value, i, f'{value:.3f}', va='center', ha='right', color='black')

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1 * abs(xlim[0]), xlim[1] + 0.1 * abs(xlim[1]))

    plt.title(title)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()


def plot_results(scores: pd.DataFrame) -> None:
    """Plot model performance results.
    
    Args:
        scores: DataFrame containing model scores.
    """
    mean_scores = scores.mean().sort_values(ascending=False)
    order = scores.mean().sort_values(ascending=False).index.tolist()

    min_score = mean_scores.min()
    max_score = mean_scores.max()
    padding = (max_score - min_score) * 0.5
    lower_limit = min_score - padding
    upper_limit = max_score + padding

    fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.4))

    boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient='h', color='grey')
    axs[0].set_title('Fold Accuracy')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')

    barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color='grey')
    axs[1].set_title('Average Accuracy')
    axs[1].set_xlabel('')
    axs[1].set_xlim(left=lower_limit, right=upper_limit)
    axs[1].set_ylabel('')

    for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
        color = 'cyan' if 'logistic' in model.lower() or 'weighted' in model.lower() else 'grey'
        barplot.patches[i].set_facecolor(color)
        boxplot.patches[i].set_facecolor(color)
        barplot.text(score, i, round(score, 6), va='center')

    plt.tight_layout()
    plt.show()

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess training and test data.
    
    Returns:
        Tuple containing X_train, y_train, X_test, and X_test_copy.
    """
    train = pd.read_csv(CFG.train_path, index_col='id')
    test = pd.read_csv(CFG.test_path, index_col='id')
    
    original = pd.read_csv(CFG.original_path)
    original = original.rename(columns={'Personality': 'match_p'})
    original = original.drop_duplicates([
        'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
        'Post_frequency'
    ])
    train = train.merge(original, how='left')
    test = test.merge(original, how='left')
    
    cat_cols = ["Stage_fear", "Drained_after_socializing"]
    train[cat_cols] = train[cat_cols].fillna("missing").astype("category")
    test[cat_cols] = test[cat_cols].fillna("missing").astype("category")
    
    train[CFG.target] = train[CFG.target].map({"Extrovert": 0, "Introvert": 1})
    train["match_p"] = train["match_p"].map({"Extrovert": 0, "Introvert": 1})
    test["match_p"] = test["match_p"].map({"Extrovert": 0, "Introvert": 1})
    
    X = train.drop(CFG.target, axis=1)
    y = train[CFG.target]
    X_test = test
    _X_test = test.copy()
    
    return X, y, X_test, _X_test


def create_exploratory_plots(train: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    """Create exploratory data analysis plots.
    
    Args:
        train: Training DataFrame.
        X: Feature DataFrame.
        y: Target Series.
    """
    sns.set_style("white")
    plt.figure(figsize=(8, 8))
    
    corr_train = train.corr()
    mask_train = np.triu(np.ones_like(corr_train, dtype=bool), k=1)
    
    sns.heatmap(
        data=corr_train,
        annot=True,
        fmt='.4f',
        mask=mask_train,
        square=True,
        cmap='coolwarm',
        annot_kws={'size': 8},
        cbar=False
    )
    
    plt.tight_layout()
    plt.show()
    
    mutual_info = mutual_info_regression(X.fillna(0), y, random_state=CFG.seed)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X.columns
    mutual_info = pd.DataFrame(mutual_info.sort_values(ascending=False), columns=['Mutual Information'])
    mutual_info.style.bar(subset=['Mutual Information'], cmap='RdYlGn')

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

def get_model_parameters() -> Dict[str, Dict[str, Any]]:
    """Get optimized parameters for all models.
    
    Returns:
        Dictionary containing parameters for each model.
    """
    cat_cols = ["Stage_fear", "Drained_after_socializing"]
    
    return {
        "catboost": {
            "border_count": 39,
            "colsample_bylevel": 0.19459088572914465,
            "depth": 2,
            "iterations": 1467,
            "l2_leaf_reg": 31.236169478676036,
            "learning_rate": 0.06852669420904771,
            "min_child_samples": 160,
            "random_state": 42,
            "random_strength": 0.8517786189616939,
            "scale_pos_weight": 1.1691394390533685,
            "subsample": 0.3192330024411618,
            "verbose": False,
            "cat_features": cat_cols
        },
        "xgboost": {
            "colsample_bylevel": 0.8168489864941239,
            "colsample_bynode": 0.8850485490950061,
            "colsample_bytree": 0.8379339940113913,
            "gamma": 2.3977359439809276,
            "learning_rate": 0.0616974880921061,
            "max_depth": 344,
            "max_leaves": 89,
            "min_child_weight": 10,
            "n_estimators": 696,
            "n_jobs": -1,
            "random_state": 42,
            "reg_alpha": 1.849084818346014,
            "reg_lambda": 29.680324563362227,
            "subsample": 0.5902901569391961,
            "verbosity": 0,
            "enable_categorical": True
        },
        "histgradient": {
            "l2_regularization": 28.13576008319012,
            "learning_rate": 0.1543598086529694,
            "max_depth": 325,
            "max_features": 0.323620656779567,
            "max_iter": 2490,
            "max_leaf_nodes": 216,
            "min_samples_leaf": 12,
            "random_state": 42,
            "categorical_features": "from_dtype"
        },
        "lgbm_gbdt": {
            "boosting_type": "gbdt",
            "colsample_bytree": 0.6467443250209886,
            "learning_rate": 0.06547186748153115,
            "min_child_samples": 34,
            "min_child_weight": 0.24399244943904663,
            "n_estimators": 498,
            "n_jobs": -1,
            "num_leaves": 158,
            "random_state": 42,
            "reg_alpha": 6.568921253574134,
            "reg_lambda": 62.66165355751099,
            "subsample": 0.0011019938618584968,
            "verbose": -1
        },
        "lgbm_goss": {
            "boosting_type": "goss",
            "colsample_bytree": 0.8384834064170148,
            "learning_rate": 0.07006829797238343,
            "min_child_samples": 46,
            "min_child_weight": 0.7625394962666617,
            "n_estimators": 1887,
            "n_jobs": -1,
            "num_leaves": 341,
            "random_state": 42,
            "reg_alpha": 10.53082019937197,
            "reg_lambda": 67.44600065144685,
            "subsample": 0.4925008305336127,
            "verbose": -1
        },
        "lgbm_dart": {
            "boosting_type": "dart",
            "colsample_bytree": 0.7592971191793424,
            "learning_rate": 0.046141766106846074,
            "min_child_samples": 18,
            "min_child_weight": 0.4740109054323218,
            "n_estimators": 4035,
            "n_jobs": -1,
            "num_leaves": 393,
            "random_state": 42,
            "reg_alpha": 48.016799341666605,
            "reg_lambda": 89.12860300833658,
            "subsample": 0.016333358901112538,
            "verbose": -1
        }
    }

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_base_models(
    X: pd.DataFrame, 
    y: pd.Series, 
    X_test: pd.DataFrame
) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Train all base models.
    
    Args:
        X: Training features.
        y: Training target.
        X_test: Test features.
    
    Returns:
        Tuple containing scores, OOF predictions, and test predictions dictionaries.
    """
    params = get_model_parameters()
    scores = {}
    oof_pred_probs = {}
    test_pred_probs = {}
    
    # CatBoost
    cb_trainer = Trainer(
        CatBoostClassifier(**params["catboost"]),
        cv=CFG.cv,
        metric=CFG.metric,
        use_early_stopping=False,
        task="binary",
        metric_precision=6,
    )
    cb_trainer.fit(X, y)
    scores["CatBoost"] = cb_trainer.fold_scores
    oof_pred_probs["CatBoost"] = cb_trainer.oof_preds
    test_pred_probs["CatBoost"] = cb_trainer.predict(X_test)
    
    # XGBoost
    xgb_trainer = Trainer(
        XGBClassifier(**params["xgboost"]),
        cv=CFG.cv,
        metric=CFG.metric,
        task="binary",
        metric_precision=6,
    )
    xgb_trainer.fit(X, y)
    scores["XGBoost"] = xgb_trainer.fold_scores
    oof_pred_probs["XGBoost"] = xgb_trainer.oof_preds
    test_pred_probs["XGBoost"] = xgb_trainer.predict(X_test)
    
    # HistGradientBoosting
    hgb_trainer = Trainer(
        HistGradientBoostingClassifier(**params["histgradient"]),
        cv=CFG.cv,
        metric=CFG.metric,
        task="binary",
        metric_precision=6,
    )
    hgb_trainer.fit(X, y)
    scores["HistGradientBoosting"] = hgb_trainer.fold_scores
    oof_pred_probs["HistGradientBoosting"] = hgb_trainer.oof_preds
    test_pred_probs["HistGradientBoosting"] = hgb_trainer.predict(X_test)
    
    # LightGBM (gbdt)
    lgbm_gbdt_trainer = Trainer(
        LGBMClassifier(**params["lgbm_gbdt"]),
        cv=CFG.cv,
        metric=CFG.metric,
        use_early_stopping=False,
        task="binary",
        metric_precision=6,
    )
    lgbm_gbdt_trainer.fit(X, y)
    scores["LightGBM (gbdt)"] = lgbm_gbdt_trainer.fold_scores
    oof_pred_probs["LightGBM (gbdt)"] = lgbm_gbdt_trainer.oof_preds
    test_pred_probs["LightGBM (gbdt)"] = lgbm_gbdt_trainer.predict(X_test)
    
    # LightGBM (goss)
    lgbm_goss_trainer = Trainer(
        LGBMClassifier(**params["lgbm_goss"]),
        cv=CFG.cv,
        metric=CFG.metric,
        use_early_stopping=False,
        task="binary",
        metric_precision=6,
    )
    lgbm_goss_trainer.fit(X, y)
    scores["LightGBM (goss)"] = lgbm_goss_trainer.fold_scores
    oof_pred_probs["LightGBM (goss)"] = lgbm_goss_trainer.oof_preds
    test_pred_probs["LightGBM (goss)"] = lgbm_goss_trainer.predict(X_test)
    
    # LightGBM (dart)
    lgbm_dart_trainer = Trainer(
        LGBMClassifier(**params["lgbm_dart"]),
        cv=CFG.cv,
        metric=CFG.metric,
        use_early_stopping=False,
        task="binary",
        metric_precision=6,
    )
    lgbm_dart_trainer.fit(X, y)
    scores["LightGBM (dart)"] = lgbm_dart_trainer.fold_scores
    oof_pred_probs["LightGBM (dart)"] = lgbm_dart_trainer.oof_preds
    test_pred_probs["LightGBM (dart)"] = lgbm_dart_trainer.predict(X_test)
    
    return scores, oof_pred_probs, test_pred_probs


def load_autogluon_predictions(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Load AutoGluon predictions from saved files.
    
    Args:
        X: Training features.
        y: Training target.
    
    Returns:
        Tuple containing OOF predictions, test predictions, and scores.
    """
    oof_pred_probs_files = glob.glob(
        '/kaggle/input/s05e07-personality-type-prediction-autogluon/*_oof_pred_probs_*.pkl'
    )
    test_pred_probs_files = glob.glob(
        '/kaggle/input/s05e07-personality-type-prediction-autogluon/*_test_pred_probs_*.pkl'
    )
    
    ag_oof_pred_probs = joblib.load(oof_pred_probs_files[0])
    ag_test_pred_probs = joblib.load(test_pred_probs_files[0])
    
    ag_scores = []
    for _, val_idx in CFG.cv.split(X, y):
        y_val = y[val_idx]
        y_preds = ag_oof_pred_probs[val_idx]
        score = accuracy_score(y_val, y_preds >= 0.5)
        ag_scores.append(score)
    
    return ag_oof_pred_probs, ag_test_pred_probs, ag_scores

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

def optimize_logistic_regression(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Optimize LogisticRegression parameters using Optuna.
    
    Args:
        X: Training features (logit transformed).
        y: Training target.
    
    Returns:
        Dictionary containing best parameters and threshold.
    """
    def objective(trial):
        solver_penalty_options = [
            ('liblinear', 'l1'),
            ('liblinear', 'l2'),
            ('lbfgs', 'l2'),
            ('lbfgs', None),
            ('newton-cg', 'l2'),
            ('newton-cg', None),
            ('newton-cholesky', 'l2'),
            ('newton-cholesky', None)
        ]
        solver, penalty = trial.suggest_categorical('solver_penalty', solver_penalty_options)
        
        params = {
            'random_state': CFG.seed,
            'max_iter': 1000,
            'C': trial.suggest_float('C', 0, 1),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'solver': solver,
            'penalty': penalty
        }
        
        threshold = trial.suggest_float('threshold', 0, 1)
        
        trainer = Trainer(
            LogisticRegression(**params),
            cv=CFG.cv,
            metric=CFG.metric,
            metric_precision=6,
            metric_threshold=threshold,
            use_early_stopping=False,
            verbose=False,
            task="binary",
        )
        trainer.fit(X, y)
        
        return np.mean(trainer.fold_scores)
    
    sampler = optuna.samplers.TPESampler(
        seed=CFG.seed, 
        multivariate=True, 
        n_startup_trials=CFG.n_optuna_trials // 10
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=CFG.n_optuna_trials, n_jobs=-1)
    
    return study.best_params


def train_logistic_regression_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    best_params: Dict[str, Any]
) -> Tuple[List[float], np.ndarray, Trainer]:
    """Train LogisticRegression ensemble model.
    
    Args:
        X: Training features (logit transformed).
        y: Training target.
        X_test: Test features (logit transformed).
        best_params: Optimized parameters from Optuna.
    
    Returns:
        Tuple containing scores, test predictions, and trainer object.
    """
    solver, penalty = best_params['solver_penalty']
    lr_params = {
        'random_state': CFG.seed,
        'max_iter': 1000,
        'C': best_params['C'],
        'tol': best_params['tol'],
        'fit_intercept': best_params['fit_intercept'],
        'class_weight': best_params['class_weight'],
        'solver': solver,
        'penalty': penalty
    }
    
    best_threshold = best_params['threshold']
    print(f'Best threshold: {best_threshold:.3f}')
    print(json.dumps(lr_params, indent=2))
    
    lr_trainer = Trainer(
        LogisticRegression(**lr_params),
        cv=CFG.cv,
        metric=CFG.metric,
        metric_threshold=best_threshold,
        metric_precision=6,
        use_early_stopping=False,
        task="binary",
    )
    
    lr_trainer.fit(X, y)
    lr_test_pred_probs = lr_trainer.predict(X_test)
    
    return lr_trainer.fold_scores, lr_test_pred_probs, lr_trainer, best_threshold


def optimize_weighted_average(
    oof_pred_probs: Dict[str, np.ndarray],
    y: pd.Series
) -> Tuple[Dict[str, float], float, float]:
    """Optimize weighted average ensemble using Optuna.
    
    Args:
        oof_pred_probs: Dictionary of OOF predictions from all models.
        y: Training target.
    
    Returns:
        Tuple containing best weights, threshold, and score.
    """
    def objective(trial):
        weights = np.array([trial.suggest_float(m, -1, 1) for m in oof_pred_probs.keys()])
        weights /= np.sum(weights)
        
        preds = np.zeros(len(y))
        for m, weight in zip(oof_pred_probs.keys(), weights):
            preds += oof_pred_probs[m] * weight
        
        threshold = trial.suggest_float('threshold', 0, 1)
        
        return accuracy_score(y, (preds > threshold).astype(int))
    
    sampler = optuna.samplers.TPESampler(
        seed=CFG.seed,
        multivariate=True,
        n_startup_trials=CFG.n_optuna_trials // 10
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=CFG.n_optuna_trials, n_jobs=-1)
    
    best_weights = np.array([study.best_params[m] for m in oof_pred_probs.keys()])
    best_weights /= np.sum(best_weights)
    
    best_weights_dict = {
        model: weight for model, weight in sorted(
            zip(oof_pred_probs.keys(), best_weights),
            key=lambda x: x[1],
            reverse=True
        )
    }
    
    best_threshold = study.best_params['threshold']
    
    return best_weights_dict, best_threshold, study.best_value

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the complete training and ensemble pipeline."""
    
    # Load and preprocess data
    X, y, X_test, _X_test = load_and_preprocess_data()
    
    # Create exploratory plots
    train = pd.read_csv(CFG.train_path, index_col='id')
    train["Stage_fear"] = train["Stage_fear"].map({"No": 0, "Yes": 1})
    train["Drained_after_socializing"] = train["Drained_after_socializing"].map({"No": 0, "Yes": 1})
    train[CFG.target] = train[CFG.target].map({"Extrovert": 0, "Introvert": 1})
    create_exploratory_plots(train, X.select_dtypes(include=[np.number]), y)
    
    # Train base models
    scores, oof_pred_probs, test_pred_probs = train_base_models(X, y, X_test)
    
    # Load AutoGluon predictions
    ag_oof, ag_test, ag_scores = load_autogluon_predictions(X, y)
    oof_pred_probs["AutoGluon"] = ag_oof
    test_pred_probs["AutoGluon"] = ag_test
    scores["AutoGluon"] = ag_scores
    
    # Prepare data for ensemble
    X_ensemble = logit(pd.DataFrame(oof_pred_probs).clip(1e-15, 1-1e-15))
    X_test_ensemble = logit(pd.DataFrame(test_pred_probs).clip(1e-15, 1-1e-15))
    
    joblib.dump(oof_pred_probs, "oof_pred_probs.pkl")
    joblib.dump(test_pred_probs, "test_pred_probs.pkl")
    
    # LogisticRegression ensemble
    best_lr_params = optimize_logistic_regression(X_ensemble, y)
    lr_scores, lr_test_preds, lr_trainer, lr_threshold = train_logistic_regression_ensemble(
        X_ensemble, y, X_test_ensemble, best_lr_params
    )
    scores["LogisticRegression"] = lr_scores
    
    save_submission(
        'logistic-regression',
        _X_test,
        lr_test_preds,
        np.mean(scores['LogisticRegression']),
        lr_threshold
    )
    
    # Plot LogisticRegression coefficients
    lr_coeffs = np.zeros((1, len(X_ensemble.columns)))
    for estimator in lr_trainer.estimators:
        lr_coeffs += estimator.coef_ / CFG.n_folds
    plot_weights(lr_coeffs, 'LR Coefficients', list(oof_pred_probs.keys()))
    
    # Weighted average ensemble
    best_weights, best_threshold, best_score = optimize_weighted_average(oof_pred_probs, y)
    scores['WeightedAverage'] = [best_score] * CFG.n_folds
    
    print(json.dumps(best_weights, indent=2))
    print(f'Best threshold: {best_threshold:.3f}')
    
    weighted_test_preds = np.zeros(len(test_pred_probs["CatBoost"]))
    for m, weight in best_weights.items():
        weighted_test_preds += test_pred_probs[m] * weight
    
    save_submission(
        'weighted-ensemble',
        _X_test,
        weighted_test_preds,
        np.mean(scores['WeightedAverage']),
        best_threshold
    )
    
    # Plot final results
    scores_df = pd.DataFrame(scores)
    plot_results(scores_df)
    
    # Cleanup
    shutil.rmtree('catboost_info', ignore_errors=True)


if __name__ == "__main__":
    main()

