# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Additional utilities
import warnings
from scipy import stats

# Configure warnings and display settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# Set professional plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

print("ğŸ“¦ Libraries imported successfully!")



# Define file paths
PATH = "/kaggle/input/playground-series-s5e9/"
PATH_ORIGINAL = "/kaggle/input/bpm-prediction-challenge/"  # Path to the recommended original dataset

# Load the datasets
print("ğŸ“� Loading datasets...")

try:
    df_train = pd.read_csv(PATH + "train.csv")
    df_test = pd.read_csv(PATH + "test.csv")
    
    # Uncomment the following line if you've added the original dataset
    # df_original = pd.read_csv(PATH_ORIGINAL + "train.csv")
    
    print(f"âœ… Training data shape: {df_train.shape}")
    print(f"âœ… Test data shape: {df_test.shape}")
    # print(f"âœ… Original data shape: {df_original.shape}")  # Uncomment if using original data
    
except FileNotFoundError:
    print("âš ï¸�  Files not found. Please check the file paths.")
    # For demonstration purposes, create sample data
    print("ğŸ”§ Creating sample data for demonstration...")
    
    np.random.seed(42)
    n_samples = 10000
    
    df_train = pd.DataFrame({
        'id': range(n_samples),
        'AudioLoudness': np.random.normal(-10, 5, n_samples),
        'MoodScore': np.random.normal(0.5, 0.2, n_samples),
        'Energy': np.random.beta(2, 2, n_samples),
        'RhythmScore': np.random.gamma(2, 0.3, n_samples),
        'InstrumentalScore': np.random.exponential(0.2, n_samples),
        'LivePerformanceLikelihood': np.random.exponential(0.1, n_samples),
        'AcousticQuality': np.random.beta(1.5, 3, n_samples),
        'TrackDurationMs': np.random.normal(200000, 50000, n_samples),
        'BeatsPerMinute': np.random.normal(119, 25, n_samples)
    })
    
    df_test = df_train.drop('BeatsPerMinute', axis=1).copy()
    df_test['id'] = range(n_samples, 2*n_samples)
    
    print(f"ğŸ“Š Sample training data shape: {df_train.shape}")
    print(f"ğŸ“Š Sample test data shape: {df_test.shape}")



# Display the first few rows with enhanced formatting
print("ğŸ”� First 5 rows of the training data:")
print("=" * 60)
display(df_train.head().style.background_gradient(cmap='viridis'))



# Get a concise summary of the dataframe
print("ğŸ“‹ Dataframe Information:")
print("=" * 40)
df_train.info()

print("\nğŸ”¢ Data Types Summary:")
print("-" * 30)
dtype_counts = df_train.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"{dtype}: {count} columns")



# Check for missing values
print("ğŸ”� Missing Values Analysis:")
print("=" * 35)
missing_values = df_train.isnull().sum()
missing_percentage = (missing_values / len(df_train)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})

missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) == 0:
    print("âœ… No missing values found in the dataset!")
else:
    display(missing_df.style.background_gradient(cmap='Reds'))



# Generate descriptive statistics with enhanced formatting
print("ğŸ“Š Descriptive Statistics:")
print("=" * 30)
stats_df = df_train.describe()
display(stats_df.style.background_gradient(cmap='coolwarm').format('{:.4f}'))



# Create a comprehensive target variable analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ğŸ�µ BeatsPerMinute: Comprehensive Distribution Analysis', fontsize=20, y=0.98)

# Main distribution plot
ax1 = axes[0, 0]
sns.histplot(df_train['BeatsPerMinute'], kde=True, bins=50, color='royalblue', 
             edgecolor='white', linewidth=0.5, ax=ax1, alpha=0.7)
ax1.set_title('Distribution of BeatsPerMinute', fontsize=16, pad=15)
ax1.set_xlabel('Beats Per Minute (BPM)')
ax1.set_ylabel('Frequency')

# Add statistical lines
mean_val = df_train['BeatsPerMinute'].mean()
median_val = df_train['BeatsPerMinute'].median()
std_val = df_train['BeatsPerMinute'].std()

ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
ax1.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
ax1.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1 STD: {mean_val + std_val:.2f}')
ax1.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1 STD: {mean_val - std_val:.2f}')
ax1.legend()

# Box plot
ax2 = axes[0, 1]
sns.boxplot(y=df_train['BeatsPerMinute'], color='lightblue', ax=ax2)
ax2.set_title('Box Plot: Outlier Detection', fontsize=16, pad=15)
ax2.set_ylabel('Beats Per Minute (BPM)')

# Q-Q Plot for normality assessment
ax3 = axes[1, 0]
stats.probplot(df_train['BeatsPerMinute'], dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot: Normality Assessment', fontsize=16, pad=15)
ax3.grid(True, alpha=0.3)

# Violin plot for detailed distribution shape
ax4 = axes[1, 1]
sns.violinplot(y=df_train['BeatsPerMinute'], color='lightcoral', ax=ax4)
ax4.set_title('Violin Plot: Distribution Shape', fontsize=16, pad=15)
ax4.set_ylabel('Beats Per Minute (BPM)')

plt.tight_layout()
plt.show()



# Statistical summary of the target variable
print("ğŸ“Š BeatsPerMinute Statistical Summary:")
print("=" * 45)

bpm_stats = {
    'Mean': df_train['BeatsPerMinute'].mean(),
    'Median': df_train['BeatsPerMinute'].median(),
    'Mode': df_train['BeatsPerMinute'].mode().iloc[0],
    'Standard Deviation': df_train['BeatsPerMinute'].std(),
    'Variance': df_train['BeatsPerMinute'].var(),
    'Skewness': df_train['BeatsPerMinute'].skew(),
    'Kurtosis': df_train['BeatsPerMinute'].kurtosis(),
    'Minimum': df_train['BeatsPerMinute'].min(),
    'Maximum': df_train['BeatsPerMinute'].max(),
    'Range': df_train['BeatsPerMinute'].max() - df_train['BeatsPerMinute'].min(),
    'IQR': df_train['BeatsPerMinute'].quantile(0.75) - df_train['BeatsPerMinute'].quantile(0.25)
}

for stat, value in bpm_stats.items():
    print(f"{stat:20}: {value:8.4f}")

# Normality test
print("\nğŸ”¬ Normality Tests:")
print("-" * 25)
shapiro_stat, shapiro_p = stats.shapiro(df_train['BeatsPerMinute'].sample(5000))  # Sample for Shapiro-Wilk
ks_stat, ks_p = stats.kstest(df_train['BeatsPerMinute'], 'norm', 
                            args=(df_train['BeatsPerMinute'].mean(), df_train['BeatsPerMinute'].std()))

print(f"Shapiro-Wilk Test: statistic={shapiro_stat:.6f}, p-value={shapiro_p:.6f}")
print(f"Kolmogorov-Smirnov Test: statistic={ks_stat:.6f}, p-value={ks_p:.6f}")

if shapiro_p > 0.05:
    print("âœ… Data appears to be normally distributed (Shapiro-Wilk)")
else:
    print("â�Œ Data deviates from normal distribution (Shapiro-Wilk)")



# Select all features except ID and target
features = [col for col in df_train.columns if col not in ['id', 'BeatsPerMinute']]
n_features = len(features)

print(f"ğŸ“Š Analyzing {n_features} features: {', '.join(features)}")

# Plot histograms for each feature
df_train[features].hist(bins=30, figsize=(20, 15), layout=(-1, 4), edgecolor='white', lw=0.5)
plt.suptitle('Distribution of Predictor Features', fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()



# Calculate the correlation matrix
corr_matrix = df_train.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix of Features', fontsize=18, pad=20)
plt.show()

# Display correlations with the target variable, sorted
print("\nCorrelation with BeatsPerMinute:")
display(corr_matrix[['BeatsPerMinute']].sort_values(by='BeatsPerMinute', ascending=False))



# We'll use the top 4 correlated features, even though the correlation is weak
top_features = corr_matrix['BeatsPerMinute'].abs().sort_values(ascending=False).index[1:5]

plt.figure(figsize=(18, 12))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(data=df_train, x=feature, y='BeatsPerMinute', alpha=0.2)
    plt.title(f'BeatsPerMinute vs {feature}', fontsize=16)

plt.suptitle('Feature vs. Target Scatter Plots', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


