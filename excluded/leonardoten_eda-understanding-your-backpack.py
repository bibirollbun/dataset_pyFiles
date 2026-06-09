# =====================
# INITIAL SETUP
# =====================
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
from warnings import simplefilter
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
custom_palette = sns.color_palette("husl", 8)
sns.set_palette(custom_palette)


# =====================
# DATA LOADING
# =====================
# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')  
test_df = pd.read_csv('//kaggle/input/playground-series-s5e2/test.csv')    
print("âœ… Data loaded successfully:")
print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
print("\nğŸ•µï¸� First look at training data:")
display(train_df.head().style.set_caption("Sample Training Data"))


# ===========================================
# 2.1 DATASET METADATA COMPARISON
# ===========================================
print("\nğŸ”� Dataset Structure Analysis:")
print(f"Train columns ({len(train_df.columns)}): {list(train_df.columns)}")
print(f"Test columns ({len(test_df.columns)}): {list(test_df.columns)}")

# Target variable check
print("\nğŸ�¯ Target Variable Analysis:")
print(f"Price range: ${train_df['Price'].min():.2f} - ${train_df['Price'].max():.2f}")
print(f"Mean price: ${train_df['Price'].mean():.2f} (Â±${train_df['Price'].std():.2f})")

# ===========================================
# 2.2 DATA TYPES & MISSING VALUES
# ===========================================
def analyze_missing(df, dataset_name):
    analysis = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isna().sum(),
        'missing_%': (df.isna().mean() * 100).round(2),
        'unique': df.nunique()
    })
    print(f"\nğŸ”� {dataset_name} Set - Missing Values & Data Types:")
    return analysis

# Execute analysis
train_analysis = analyze_missing(train_df, "Training")
test_analysis = analyze_missing(test_df, "Test")

# Display formatted tables
display(train_analysis.style.format({'missing_%': '{:.2f}%'})
       .background_gradient(subset=['missing_%'], cmap='Reds')
       .set_caption("Training Set - Data Profile"))

display(test_analysis.style.format({'missing_%': '{:.2f}%'})
       .background_gradient(subset=['missing_%'], cmap='Blues')
       .set_caption("Test Set - Data Profile"))


# =====================
# 3.1 PRICE DISTRIBUTION
# =====================
plt.figure(figsize=(14, 5))

# Histogram with KDE
plt.subplot(1, 2, 1)
sns.histplot(train_df['Price'], kde=True, bins=30, color='teal')
plt.title('Price Distribution', fontweight='bold')
plt.xlabel('Price (USD)')

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(y=train_df['Price'], color='salmon')
plt.title('Price Spread', fontweight='bold')
plt.tight_layout()
plt.show()

# Skewness calculation
price_skew = train_df['Price'].skew()
print(f"Skewness coefficient: {price_skew:.2f} (>{'1' if price_skew>1 else '1'})")



cat_features = ['Brand', 'Material', 'Size', 'Waterproof', 
                'Laptop Compartment', 'Style', 'Color']

plt.figure(figsize=(18, 20))
for i, feature in enumerate(cat_features, 1):
    plt.subplot(4, 2, i)
    
    # Create bar plot with averages
    ax = sns.barplot(x=feature, y='Price', data=train_df, 
                    estimator=np.mean, palette='viridis', errorbar=None)
    
    # Add numeric values on top of the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    plt.title(f'Average Price by {feature}', fontsize=12)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('Average Price (USD)', fontsize=10)

plt.tight_layout()
plt.show()



plt.figure(figsize=(18, 20))
for i, feature in enumerate(cat_features, 1):
    plt.subplot(4, 2, i)
    
    # Create a count plot
    ax = sns.countplot(x=feature, data=train_df, palette='viridis')
    
    # Add numeric values on top of the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    plt.title(f'Count of {feature}', fontsize=12)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('Count', fontsize=10)

plt.tight_layout()
plt.show()


