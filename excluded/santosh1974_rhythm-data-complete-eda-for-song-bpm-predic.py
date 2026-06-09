import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_basic_info(file_path):
    """Load data and display basic information"""
    print("=" * 60)
    print("BASIC DATASET INFORMATION")
    print("=" * 60)
    
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn names and types:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def missing_values_analysis(df):
    """Analyze missing values"""
    print("\n" + "=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing_df['Missing Count'].sum() == 0:
        print("âœ… No missing values found!")
    
    return missing_df

def target_analysis(df, target_col='BeatsPerMinute'):
    """Comprehensive analysis of the target variable"""
    print("\n" + "=" * 60)
    print("TARGET VARIABLE ANALYSIS (BeatsPerMinute)")
    print("=" * 60)
    
    target = df[target_col]
    
    # Basic statistics
    print("Basic Statistics:")
    print(f"Mean: {target.mean():.2f}")
    print(f"Median: {target.median():.2f}")
    print(f"Std: {target.std():.2f}")
    print(f"Min: {target.min():.2f}")
    print(f"Max: {target.max():.2f}")
    print(f"Range: {target.max() - target.min():.2f}")
    print(f"Skewness: {target.skew():.3f}")
    print(f"Kurtosis: {target.kurtosis():.3f}")
    
    # Quantiles
    print(f"\nQuantiles:")
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        print(f"{q*100:2.0f}th percentile: {target.quantile(q):.2f}")
    
    # Outlier detection using IQR
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = target[(target < lower_bound) | (target > upper_bound)]
    
    print(f"\nOutlier Analysis (IQR method):")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(target)*100:.2f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram
    axes[0,0].hist(target, bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(target.mean(), color='red', linestyle='--', label=f'Mean: {target.mean():.1f}')
    axes[0,0].axvline(target.median(), color='green', linestyle='--', label=f'Median: {target.median():.1f}')
    axes[0,0].set_title('BPM Distribution')
    axes[0,0].set_xlabel('Beats Per Minute')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    
    # Box plot
    axes[0,1].boxplot(target)
    axes[0,1].set_title('BPM Box Plot')
    axes[0,1].set_ylabel('Beats Per Minute')
    
    # Q-Q plot for normality
    stats.probplot(target, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot (Normal Distribution)')
    
    # Density plot
    target.plot.density(ax=axes[1,1])
    axes[1,1].set_title('BPM Density Plot')
    axes[1,1].set_xlabel('Beats Per Minute')
    
    plt.tight_layout()
    plt.show()
    
    return target.describe()

def feature_analysis(df):
    """Analyze individual features"""
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    # Exclude id and target columns
    feature_cols = [col for col in df.columns if col not in ['id', 'BeatsPerMinute']]
    
    # Basic statistics for all features
    print("Feature Statistics:")
    print(df[feature_cols].describe().round(3))
    
    # Feature distributions
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, col in enumerate(feature_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    """Analyze correlations between features and target"""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Correlations with target
    target_corr = corr_matrix['BeatsPerMinute'].abs().sort_values(ascending=False)
    print("Features correlation with BeatsPerMinute (absolute values):")
    print(target_corr.round(3))
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Feature vs Target scatter plots for top correlated features
    top_features = target_corr.drop('BeatsPerMinute').head(6).index
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        axes[i].scatter(df[feature], df['BeatsPerMinute'], alpha=0.5)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('BeatsPerMinute')
        axes[i].set_title(f'{feature} vs BPM\n(r = {corr_matrix.loc[feature, "BeatsPerMinute"]:.3f})')
        
        # Add trend line
        z = np.polyfit(df[feature], df['BeatsPerMinute'], 1)
        p = np.poly1d(z)
        axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def genre_analysis_by_bpm_ranges(df):
    """Analyze BPM ranges to understand music genres/styles"""
    print("\n" + "=" * 60)
    print("BPM RANGE ANALYSIS (Music Genre Insights)")
    print("=" * 60)
    
    # Define typical BPM ranges for different music styles
    def categorize_bpm(bpm):
        if bpm < 60:
            return 'Very Slow (< 60)'
        elif bpm < 80:
            return 'Slow (60-80)'
        elif bpm < 100:
            return 'Moderate (80-100)'
        elif bpm < 120:
            return 'Medium (100-120)'
        elif bpm < 140:
            return 'Fast (120-140)'
        elif bpm < 160:
            return 'Very Fast (140-160)'
        else:
            return 'Extremely Fast (> 160)'
    
    df['BPM_Category'] = df['BeatsPerMinute'].apply(categorize_bpm)
    
    # Count and percentage by category
    bpm_counts = df['BPM_Category'].value_counts()
    bpm_pct = df['BPM_Category'].value_counts(normalize=True) * 100
    
    print("BPM Range Distribution:")
    for category in bpm_counts.index:
        print(f"{category}: {bpm_counts[category]} songs ({bpm_pct[category]:.1f}%)")
    
    # Visualize BPM categories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    bpm_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Songs by BPM Range')
    ax1.set_xlabel('BPM Category')
    ax1.set_ylabel('Number of Songs')
    ax1.tick_params(axis='x', rotation=45)
    
    # Pie chart
    ax2.pie(bpm_counts.values, labels=bpm_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('BPM Range Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return df

def feature_engineering_insights(df):
    """Suggest feature engineering opportunities"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING INSIGHTS")
    print("=" * 60)
    
    feature_cols = [col for col in df.columns if col not in ['id', 'BeatsPerMinute', 'BPM_Category']]
    
    # Ratio features that might be interesting
    print("Potential Ratio Features:")
    print("- Energy/AudioLoudness ratio (energy per unit loudness)")
    print("- RhythmScore * Energy (rhythmic energy)")
    print("- VocalContent/InstrumentalScore ratio (vocal vs instrumental balance)")
    print("- TrackDurationMs/1000 (duration in seconds)")
    
    # Create some example engineered features
    df['EnergyLoudnessRatio'] = df['Energy'] / (df['AudioLoudness'].abs() + 1e-6)
    df['RhythmicEnergy'] = df['RhythmScore'] * df['Energy']
    df['VocalInstrumentalRatio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-6)
    df['TrackDurationSec'] = df['TrackDurationMs'] / 1000
    
    # Check correlations of engineered features with target
    engineered_features = ['EnergyLoudnessRatio', 'RhythmicEnergy', 'VocalInstrumentalRatio', 'TrackDurationSec']
    
    print("\nCorrelations of engineered features with BPM:")
    for feature in engineered_features:
        corr = df[feature].corr(df['BeatsPerMinute'])
        print(f"{feature}: {corr:.3f}")
    
    # Feature scaling analysis
    print(f"\nFeature Scaling Recommendations:")
    for col in feature_cols:
        col_std = df[col].std()
        col_mean = df[col].mean()
        col_range = df[col].max() - df[col].min()
        print(f"{col}: Mean={col_mean:.2f}, Std={col_std:.2f}, Range={col_range:.2f}")
    
    return df

def pca_analysis(df):
    """Principal Component Analysis for dimensionality insights"""
    print("\n" + "=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS")
    print("=" * 60)
    
    # Select numerical features (excluding id and target)
    feature_cols = [col for col in df.columns if col not in ['id', 'BeatsPerMinute', 'BPM_Category']]
    X = df[feature_cols].fillna(0)  # Handle any NaN values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    print("Explained Variance by Component:")
    for i, (var, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance)):
        print(f"PC{i+1}: {var:.3f} (Cumulative: {cum_var:.3f})")
    
    # Plot explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual component variance
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    ax1.set_title('Explained Variance by Principal Component')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
    ax2.axhline(y=0.9, color='g', linestyle='--', label='90% Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance in first few components
    feature_importance = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=feature_cols
    )
    
    print("\nFeature loadings for first 3 components:")
    print(feature_importance.round(3))
    
    return pca, X_scaled

def main():
    """Main EDA function"""
    # Kaggle notebook paths - update the competition name
    file_path = '/kaggle/input/playground-series-s4e12/train.csv'  # Update with correct competition name
    
    # Alternative: Auto-detect the path
    import os
    if not os.path.exists(file_path):
        # Try to find the train.csv file automatically
        kaggle_input = '/kaggle/input'
        if os.path.exists(kaggle_input):
            for root, dirs, files in os.walk(kaggle_input):
                if 'train.csv' in files:
                    file_path = os.path.join(root, 'train.csv')
                    print(f"Found train.csv at: {file_path}")
                    break
    
    try:
        # 1. Load and basic info
        df = load_and_basic_info(file_path)
        
        # 2. Missing values analysis
        missing_values_analysis(df)
        
        # 3. Target analysis
        target_analysis(df)
        
        # 4. Feature analysis
        feature_analysis(df)
        
        # 5. Correlation analysis
        corr_matrix = correlation_analysis(df)
        
        # 6. BPM range analysis
        df = genre_analysis_by_bpm_ranges(df)
        
        # 7. Feature engineering insights
        df = feature_engineering_insights(df)
        
        # 8. PCA analysis
        pca, X_scaled = pca_analysis(df)
        
        print("\n" + "=" * 60)
        print("EDA COMPLETE!")
        print("=" * 60)
        print("Key insights and recommendations:")
        print("1. Check the correlation analysis for most important features")
        print("2. Consider the BPM range distribution for potential stratification")
        print("3. Use the feature engineering suggestions for model improvement")
        print("4. PCA results can guide dimensionality reduction if needed")
        
        return df, corr_matrix, pca
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        print("Please update the file_path variable with the correct path to your train.csv file")
        return None, None, None

if __name__ == "__main__":
    df, corr_matrix, pca = main()

