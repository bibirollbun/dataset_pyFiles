import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


# Set visualization style
sns.set_style('whitegrid')
sns.set_palette("Set2")

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')

# Define target variable
target_col = 'accident_risk'

# Identify numerical and categorical columns
num_cols = train.drop(target_col, axis=1).select_dtypes(
    include=['int', 'float']).columns.tolist()
cat_cols = train.drop(target_col, axis=1).select_dtypes(
    include=['bool', 'object']).columns.tolist()


def quick_eda_summary(train, test, num_cols, cat_cols, target_col):
    """
    Generate quick exploratory data analysis summary
    """
    print("=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)

    # Basic dataset information
    print("\nTRAIN DATASET INFO:\n")
    print(train.info())
    print(f"\nTrain: {train.shape}, Test: {test.shape}")
    print(
        f"Numerical features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

    # Missing values analysis
    train_missing = train.isnull().sum().sum()
    test_missing = test.isnull().sum().sum()
    print("\nMISSING VALUES:")
    print(f"Train: {train_missing}, Test: {test_missing}")

    # Target variable analysis
    print(f"\nTARGET VARIABLE: '{target_col}'")
    print(
        f"Range: [{train[target_col].min():.2f}, {train[target_col].max():.2f}]")
    print(
        f"Mean: {train[target_col].mean():.3f}, Std: {train[target_col].std():.3f}")

    # Top correlations with target
    if num_cols:
        correlations = train[num_cols + [target_col]].corr()[target_col]
        correlations = correlations[correlations.index !=
                                    target_col].sort_values(ascending=False)

        print("\nTOP CORRELATIONS WITH TARGET:")
        for col in correlations.head(3).index:
            print(f"{col}: {correlations[col]:.3f}")

    # Categorical features insights
    if cat_cols:
        print("\nCATEGORICAL FEATURES INSIGHTS:")
        for col in cat_cols[:3]:  # Show first 3 categorical features
            unique_vals = train[col].nunique()
            top_val = train[col].mode(
            ).iloc[0] if not train[col].mode().empty else 'N/A'
            print(f"{col}: {unique_vals} categories, top: '{top_val}'")

    # Identify skewed numerical features
    skewed = [(col, abs(train[col].skew()))
              for col in num_cols if abs(train[col].skew()) > 2]
    if skewed:
        print("\nHIGHLY SKEWED FEATURES (>2):")
        for col, skew in skewed[:2]:
            print(f"{col} (skew: {skew:.1f})")

    # Identify high cardinality categorical features
    high_card = [col for col in cat_cols if train[col].nunique() > 20]
    if high_card:
        print("\nHIGH CARDINALITY FEATURES (>20 categories):")
        for col in high_card[:2]:
            print(f"{col} ({train[col].nunique()} categories)")


# Execute EDA summary
quick_eda_summary(train, test, num_cols, cat_cols, target_col)


def quick_target_plot(df, target_col):
    """
    Create simple target variable distribution plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Histogram with KDE
    sns.histplot(df[target_col], bins=100, ax=ax1, kde=True)
    ax1.set_title(f'Distribution of {target_col}')

    # Boxplot
    sns.boxplot(y=df[target_col], ax=ax2)
    ax2.set_title(f'Boxplot of {target_col}')

    plt.tight_layout()


# Plot target variable distribution
quick_target_plot(train, target_col)


def dist_plot(train, test, num_cols):
    """
    Compare distributions of numerical features between train and test sets
    """
    # Combine train and test data for comparison
    df = pd.concat([train[num_cols].assign(Source='Train'),
                   test[num_cols].assign(Source='Test')],
                   axis=0, ignore_index=True)

    n_cols = len(num_cols)
    fig, axes = plt.subplots(n_cols, 2,
                             figsize=(18, n_cols * 6),
                             gridspec_kw={'hspace': 0.6, 'wspace': 0.3,
                                          'width_ratios': [0.7, 0.3]
                                          }
                             )

    for i, col in enumerate(num_cols):
        # KDE plot for distribution comparison
        ax1 = axes[i, 0]
        sns.kdeplot(data=df, x=col, hue='Source', ax=ax1, linewidth=2)
        ax1.set_title(f'KDE: {col}', fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)

        # Boxplot for distribution comparison
        ax2 = axes[i, 1]
        sns.boxplot(data=df, y=col, x='Source', hue='Source',
                    ax=ax2)
        ax2.set_title(f"Boxplot: {col}", fontsize=14, pad=20)


# Compare distributions between train and test sets
dist_plot(train, test, num_cols)


def scatter_target(df, num_cols, target_col):
    """
    Create scatter plots of numerical features vs target variable
    """
    # Create custom colormap from Set2 palette
    colors = sns.color_palette("Set2", 2)
    custom_cmap = LinearSegmentedColormap.from_list("Set2", colors, N=256)
    n_cols = len(num_cols)

    fig, axes = plt.subplots(n_cols, 1, figsize=(18, n_cols * 6))

    # Handle single column case
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(num_cols):
        # Create scatter plot with color gradient based on target
        scatter = axes[i].scatter(df[col], df[target_col],
                                  c=df[target_col], cmap=custom_cmap,
                                  alpha=0.5)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f'{col} vs {target_col}')
        axes[i].grid(True, alpha=0.3)

        # Add colorbar
        plt.colorbar(scatter, ax=axes[i])

        # Add correlation coefficient annotation
        corr = df[col].corr(df[target_col])
        axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                     transform=axes[i].transAxes,
                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    fig.tight_layout()


# Create scatter plots for numerical features vs target
scatter_target(train, num_cols, target_col)


def cat_plot_vs_target(df, cat_cols, target_col, max_categories=8):
    """
    Analyze relationship between categorical features and target variable
    """
    n_cols = len(cat_cols)
    n_rows = (n_cols + 1) // 2  # 2 columns per row

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 6))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, col in enumerate(cat_cols):
        if i < len(axes):
            # Limit number of categories for better readability
            value_counts = df[col].value_counts().head(max_categories)
            plot_data = df[df[col].isin(value_counts.index)]

            # Create boxplot by categories
            sns.boxplot(data=plot_data, x=col, y=target_col, ax=axes[i])
            axes[i].set_title(
                f'{target_col} Distribution by {col}', fontsize=14)
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)

            # Add mean value annotations
            means = plot_data.groupby(col)[target_col].mean()
            for j, category in enumerate(value_counts.index):
                if category in means:
                    axes[i].text(j, means[category] + 0.02, f'{means[category]:.2f}',
                                 ha='center', va='bottom', color='black', fontweight='bold')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()


# Analyze categorical features vs target
cat_plot_vs_target(train, cat_cols, target_col)


def cat_plot_advanced(train, test, cat_cols):
    """
    Advanced comparison of categorical features between train and test sets
    """
    # Combine train and test data
    df = pd.concat([train[cat_cols].assign(Source='Train'),
                   test[cat_cols].assign(Source='Test')],
                   axis=0, ignore_index=True)

    n_cols = len(cat_cols)
    fig, axes = plt.subplots(n_cols, 2, figsize=(18, n_cols * 4),
                             gridspec_kw={'width_ratios': [0.7, 0.3]})

    for i, col in enumerate(cat_cols):
        ax1, ax2 = axes[i]

        # Get value counts for statistics
        train_counts = train[col].value_counts()
        test_counts = test[col].value_counts()

        # Create comparative bar plot
        x = np.arange(len(train_counts))
        width = 0.35

        ax1.bar(x - width/2, train_counts.values, width,
                label='Train', alpha=0.8)
        ax1.bar(x + width/2, test_counts.values, width,
                label='Test', alpha=0.8)

        ax1.set_xlabel('Category')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution: {col}', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(train_counts.index, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Create statistics panel
        stats_text = f'''
            Statistics for {col}:
            --------------------
            Unique values: {df[col].nunique()}
            Missing in train: {train[col].isnull().sum()}
            Missing in test: {test[col].isnull().sum()}
            Top category train: {train_counts.index[0] if not train_counts.empty else 'N/A'}
            Top category test: {test_counts.index[0] if not test_counts.empty else 'N/A'}
            Train dataset size: {len(train)}
            Test dataset size: {len(test)}
            '''

        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                 fontfamily='monospace', fontsize=10, va='top', linespacing=1.5)
        ax2.axis('off')


# Compare categorical features between train and test
cat_plot_advanced(train, test, cat_cols)


def plot_correlation_matrix(df, num_cols, target_col, figsize=(12, 10)):
    """
    Create correlation matrix heatmap for numerical features
    """
    # Create custom colormap from Set2 palette
    colors = sns.color_palette("Set2", 2)
    custom_cmap = LinearSegmentedColormap.from_list("Set2", colors, N=256)

    # Include target variable in correlation analysis
    corr_cols = num_cols + [target_col]
    correlation_matrix = df[corr_cols].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=custom_cmap,
                center=0, square=True, linewidths=0.5, fmt='.3f')
    plt.title('Numerical Features Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()


# Create correlation matrix
plot_correlation_matrix(train, num_cols, target_col)





