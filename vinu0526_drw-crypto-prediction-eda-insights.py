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


import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# Read the parquet file
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')


# Display basic info about the dataframe
print(train_df.head())


train_df.isnull().sum()


train_df.describe().transpose()


# Set the style for better visualization
sns.set(style="whitegrid")
plt.figure(figsize=(20, 16))

# Create subplots for each metric
fig, axes = plt.subplots(5, 1, figsize=(20, 20))

# 1. Bid Quantity Trend
axes[0].plot(train_df.index, train_df['bid_qty'], color='blue', linewidth=1)
axes[0].set_title('Bid Quantity Over Time', fontsize=14)
axes[0].set_ylabel('Bid Quantity')
axes[0].grid(True, linestyle='--', alpha=0.7)

# 2. Ask Quantity Trend
axes[1].plot(train_df.index, train_df['ask_qty'], color='red', linewidth=1)
axes[1].set_title('Ask Quantity Over Time', fontsize=14)
axes[1].set_ylabel('Ask Quantity')
axes[1].grid(True, linestyle='--', alpha=0.7)

# 3. Buy Quantity Trend
axes[2].plot(train_df.index, train_df['buy_qty'], color='green', linewidth=1)
axes[2].set_title('Buy Quantity Over Time', fontsize=14)
axes[2].set_ylabel('Buy Quantity')
axes[2].grid(True, linestyle='--', alpha=0.7)

# 4. Sell Quantity Trend
axes[3].plot(train_df.index, train_df['sell_qty'], color='purple', linewidth=1)
axes[3].set_title('Sell Quantity Over Time', fontsize=14)
axes[3].set_ylabel('Sell Quantity')
axes[3].grid(True, linestyle='--', alpha=0.7)

# 5. Volume Trend
axes[4].plot(train_df.index, train_df['volume'], color='orange', linewidth=1)
axes[4].set_title('Trading Volume Over Time', fontsize=14)
axes[4].set_ylabel('Volume')
axes[4].set_xlabel('Timestamp')
axes[4].grid(True, linestyle='--', alpha=0.7)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate full correlation matrix
corr_matrix = train_df[['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            fmt=".2f", linewidths=.5)
plt.title('Lower Triangle Correlation Matrix', fontsize=16)
plt.show()


from scipy.stats import pearsonr
from math import ceil

# Select features for analysis 
focus_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label'] + \
                [f'X{i}' for i in range(1, 891, 5)]  

# Downsample to 10% of data for faster computation
sample_df = train_df[focus_features].iloc[::10].copy()

# Calculate correlation matrix
corr_matrix = sample_df.corr()

# Calculate significant correlations with p-values
significant_corrs = []
alpha = 0.05  # Significance threshold


for i, col1 in enumerate(sample_df.columns):
    for col2 in sample_df.columns[i+1:]:  # Avoid duplicate pairs
        r, p = pearsonr(sample_df[col1].dropna(), sample_df[col2].dropna())  # Added .dropna() for safety
        if p < alpha:
            significant_corrs.append({
                'Feature1': col1,
                'Feature2': col2,
                'Correlation': r,
                'P-value': p,
                'Abs_Correlation': abs(r)
            })

# Convert to DataFrame
corr_results = pd.DataFrame(significant_corrs)


# Separate positive and negative correlations
pos_corrs = corr_results[corr_results['Correlation'] > 0].sort_values('Correlation', ascending=False)
neg_corrs = corr_results[corr_results['Correlation'] < 0].sort_values('Correlation')

# Visualization function
def plot_top_correlations(corr_df, title, n_top=20):
    plt.figure(figsize=(12, 6))
    top_df = corr_df.head(n_top)
    sns.barplot(x='Correlation', y='Feature1', hue='Feature2', data=top_df, dodge=False)
    plt.title(f'Top {n_top} {title} Correlations (p < 0.05)')
    plt.xlabel("Pearson's r")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Plot results
plot_top_correlations(pos_corrs, 'Positive')
plot_top_correlations(neg_corrs, 'Negative')


# Plotting function with feature names
def plot_scatter_grid(corr_pairs, title, n_plots=20):
    n_rows = ceil(n_plots / 4)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.5*n_rows))
    fig.suptitle(f'{title} Correlations', y=1.02, fontsize=18, weight='bold')
    
    for idx, (_, row) in enumerate(corr_pairs.head(n_plots).iterrows()):
        ax = axes[idx//4, idx%4] if n_rows > 1 else axes[idx%4]
        
        # Scatter plot with regression line
        sns.regplot(x=row['Feature1'], y=row['Feature2'], data=sample_df, 
                    ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        
        # Enhanced title with feature names and stats
        title_text = (f"{row['Feature1']} vs {row['Feature2']}\n"
                     f"r = {row['Correlation']:.2f} (p = {row['P-value']:.1e})")
        ax.set_title(title_text, fontsize=12, pad=12)
        
        # Rotate x-labels if needed
        ax.set_xlabel(row['Feature1'], fontsize=10)
        ax.set_ylabel(row['Feature2'], fontsize=10)
        
        # Adjust tick parameters for readability
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Hide empty subplots
    for idx in range(len(corr_pairs.head(n_plots)), n_rows*4):
        if n_rows > 1:
            axes[idx//4, idx%4].axis('off')
        else:
            axes[idx%4].axis('off')
    
    plt.tight_layout()
    plt.show()

# Generate plots with clear feature labels
print(f"Found {len(pos_corrs)} significant positive correlations")
if len(pos_corrs) > 0:
    plot_scatter_grid(pos_corrs, "Top 20 Positive")

print(f"\nFound {len(neg_corrs)} significant negative correlations")
if len(neg_corrs) > 0:
    plot_scatter_grid(neg_corrs, "Top 20 Negative")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import numpy as np

def plot_top_correlations_over_time(df, corr_df, time_col='timestamp', 
                                  n_pairs=20, samples_per_plot=2000,
                                  fig_width=20, row_height=2.5):
    # Prepare the top pairs
    top_pairs = corr_df.head(n_pairs)
    n_pairs = min(n_pairs, len(top_pairs))  # In case fewer than requested pairs exist
    
    # Calculate subplot layout (4 columns)
    n_cols = 4
    n_rows = int(np.ceil(n_pairs / n_cols))
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, row_height*n_rows),
                           sharex=True, squeeze=False)
    fig.suptitle(f'Top {n_pairs} Correlated Features Over Time', y=1.02, 
                fontsize=16, weight='bold')
    
    # Sample data for plotting (for performance)
    plot_df = df.iloc[::max(1, len(df)//samples_per_plot)]
    
    # Plot each correlated pair
    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        ax = axes[idx//n_cols, idx%n_cols]
        
        # Plot both features on same axes
        sns.lineplot(data=plot_df, x=time_col, y=row['Feature1'], 
                    ax=ax, color='royalblue', label=row['Feature1'], alpha=0.7)
        sns.lineplot(data=plot_df, x=time_col, y=row['Feature2'], 
                    ax=ax, color='crimson', label=row['Feature2'], alpha=0.7)
        
        # Set title with correlation info
        corr_type = "Positive" if row['Correlation'] > 0 else "Negative"
        title = (f"{row['Feature1']} & {row['Feature2']}\n"
                f"{corr_type} r = {abs(row['Correlation']):.2f} (p = {row['P-value']:.1e})")
        ax.set_title(title, fontsize=10, pad=8)
        
        # Formatting
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
        
        # Set y-axis scale based on feature ranges
        y_min = min(plot_df[row['Feature1']].min(), plot_df[row['Feature2']].min())
        y_max = max(plot_df[row['Feature1']].max(), plot_df[row['Feature2']].max())
        margin = (y_max - y_min) * 0.1  # 10% margin
        ax.set_ylim(y_min - margin, y_max + margin)
    
    # Hide any empty subplots
    for idx in range(n_pairs, n_rows*n_cols):
        axes[idx//n_cols, idx%n_cols].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# For positive correlations
plot_top_correlations_over_time(train_df, pos_corrs, n_pairs=20)

# For negative correlations
plot_top_correlations_over_time(train_df, neg_corrs, n_pairs=20)


def plot_feature_boxplots(df, features=None, n_top=20, figsize=(20, 10)):
    """
    Plot horizontal boxplots for quick distribution comparison.
    """
    if features is None:
        # Select top numeric features by variance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var().sort_values(ascending=False)
        features = variances.head(n_top).index.tolist()
    
    plt.figure(figsize=figsize)
    df[features].plot(kind='box', vert=False, patch_artist=True)
    plt.title('Feature Distributions (Boxplots)')
    plt.xlabel('Value Range')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_feature_boxplots(train_df)


def plot_order_flow_imbalance(df, window='1H', figsize=(16, 6)):
    """
    Use index for rolling calculations
    """
    df['imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'])
    
    plt.figure(figsize=figsize)
    df['imbalance'].rolling(window).mean().plot()
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f'Order Flow Imbalance ({window} rolling)')
    plt.show()

plot_order_flow_imbalance(train_df)


def plot_time_of_day_patterns(df, feature='volume', figsize=(12, 6)):
    """
    Extract hour from index
    """
    plt.figure(figsize=figsize)
    df.groupby(df.index.hour)[feature].mean().plot(kind='bar')
    plt.title(f'Hourly {feature} Pattern')
    plt.xticks(rotation=0)
    plt.show()

plot_time_of_day_patterns(train_df)

