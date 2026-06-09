import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',index_col='id')


train


train.info()


# Define categorical and numerical columns
cat_cols = ['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']
num_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']
target = 'accident_risk'
columns = test.columns.to_list()

# Binning the target to make it easier to visualize
train['accident_risk_level'] = train[target]
# Use pd.cut to divide the continuous accident risk score into 4 bins:
# - [0, 0.25] → "Minimal"
# - (0.25, 0.5] → "Moderate"
# - (0.5, 0.75] → "Significant"
# - (0.75, 1] → "Severe"
# These labels help simplify the interpretation of risk levels
train['accident_risk_level'] = pd.cut(
    train[target],
    bins=[0, 0.25, 0.5, 0.75, 1],
    labels=["Minimal", "Moderate", "Significant", "Severe"],
    include_lowest=True
)


def distribution(train_data, test_data, columns, target):
    colors = sns.color_palette("muted")

    num_plots = len(columns)
    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)

    # Add one more row for the target plot
    fig, axes = plt.subplots(num_rows + 1, num_cols, figsize=(21, 5 * (num_rows + 1)))

    # Plot feature distributions
    for i, feature in enumerate(columns):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        sns.histplot(train_data[feature], kde=True, color=colors[0], label='Train', alpha=0.5, bins=30, ax=ax)
        sns.histplot(test_data[feature], kde=True, color=colors[1], label='Test', alpha=0.5, bins=30, ax=ax)

        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()

    # Turn off any unused subplots in the last feature row
    remaining = num_rows * num_cols - num_plots
    if remaining > 0:
        for j in range(num_cols - remaining, num_cols):
            axes[num_rows - 1, j].axis('off')

    # Plot the target distribution in the first subplot of the last row
    target_ax = axes[-1, 0]
    sns.histplot(train_data[target], kde=True, color=colors[2], label='Target (Train)', alpha=0.6, bins=30, ax=target_ax)
    target_ax.set_title(f'Distribution of target: {target}')
    target_ax.set_xlabel(target)
    target_ax.set_ylabel('Frequency')
    target_ax.legend()

    # Turn off the remaining subplots in the last row (only target uses one)
    for j in range(1, num_cols):
        axes[-1, j].axis('off')

    plt.tight_layout()
    plt.show()



distribution(train, test, num_cols, target)


def plot_categorical_distributions(df, columns, max_unique=20):
    # Apply a modern Seaborn theme
    sns.set_theme(style="whitegrid")

    cat_vars = df.select_dtypes(include=['object', 'category', 'bool']).columns
    filtered_vars = [col for col in columns if df[col].nunique() <= max_unique]

    num_plots = len(filtered_vars)
    if num_plots == 0:
        print("No categorical variables with limited unique values to display.")
        return

    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Flatten axes for easy iteration
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, col in enumerate(filtered_vars):
        ax = axes[i]
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax, palette="coolwarm", hue=col)
        ax.set_title(f"Distribution of '{col}' ({df[col].nunique()} unique)")
        #ax.tick_params(axis='x', rotation=45)
        ax.get_legend().remove()

    # Turn off any unused subplots
    for j in range(len(filtered_vars), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    # Print ignored variables with too many unique values
    ignored = [col for col in columns if df[col].nunique() > max_unique]
    for col in ignored:
        print(f"Variable '{col}' has too many unique values ({df[col].nunique()}), skipped.")



plot_categorical_distributions(train, cat_cols)


temp_cols = num_cols.copy()
temp_cols.append('accident_risk')
plt.figure(figsize=(12, 8))
sns.heatmap(train[temp_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')


def plot_boxplots(df, numeric_columns, target_bin_col, title=None):
    n_cols = 2
    figsize_scale = 6
    n_rows = math.ceil(len(numeric_columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, figsize_scale * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):
        sns.boxplot(
            x=target_bin_col,
            y=col,
            data=df,
            ax=axes[i],
            palette="coolwarm",
            hue=target_bin_col,
            # showfliers=False  # Hide outliers for cleaner plots
        )
        axes[i].set_title(f"{col.replace('_', ' ').title()} by {target_bin_col.replace('_', ' ').title()}", fontsize=12)
        axes[i].get_legend().remove()
    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()


plot_boxplots(
    df=train,
    target_bin_col='accident_risk_level',
    numeric_columns=num_cols,
    title="Numeric Feature Distributions by Accident Risk Level",
)


palette = sns.color_palette("deep")
colors = [palette[i] for i in [2, 0, 1, 3]]
pd.crosstab(train['speed_limit'], train['accident_risk_level']).plot(kind='bar', stacked=True, color=colors)


def plot_stacked_bars_for_cat_cols(train, cat_cols, target_col='accident_risk_level'):
    num_cols = len(cat_cols)
    ncols = 4
    nrows = (num_cols + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows))
    axes = axes.flatten()

    palette = sns.color_palette("deep")
    colors = [palette[i] for i in [2, 0, 1, 3]]

    for i, col in enumerate(cat_cols):
        # Create the crosstab
        ct = pd.crosstab(train[col], train[target_col])
        # Plot it on the corresponding axis
        ct.plot(kind='bar', stacked=True, ax=axes[i], color=colors)
        axes[i].set_title(f'{col} vs {target_col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].legend(title=target_col)

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


plot_stacked_bars_for_cat_cols(train, cat_cols)

