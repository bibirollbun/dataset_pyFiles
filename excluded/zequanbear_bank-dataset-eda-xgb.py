import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train.head(5)


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test.head(5)


train.info()


test.info()


dataset = pd.concat([train, test])


num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']


from matplotlib import font_manager

font_path = "/kaggle/input/font-times-new-roman-ttf/Times New Roman Font.ttf"
my_font = font_manager.FontProperties(fname=font_path)


plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['font.sans-serif'] = my_font.get_name()
plt.rcParams['font.serif'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False


def plot_histograms(df, columns, my_font):
    # Set fixed number of columns for subplots
    cols = 3
    # Calculate required number of rows
    rows = (len(columns) + cols - 1) // cols  # Equivalent to ceiling division
    
    # Create subplots grid
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle('Histogram', fontproperties=my_font, fontsize=18)
    
    # Use dark green color palette
    colors = ['#006400', '#228B22', '#008000', '#004d00', '#32CD32', '#00FF00']
    
    # Flatten axes array for easy iteration (works for single row/column too)
    axes = axes.flatten()
    
    # Plot histograms for each column
    for i, col in enumerate(columns):
        color = colors[i % len(colors)]  # Cycle through colors if needed
        sns.histplot(data=df, 
                     x=col, 
                     ax=axes[i], 
                     color=color,
                     bins=30,
                     edgecolor='black',
                     linewidth=0.8,
                     kde=False)
        # axes[i].set_title(f'{col}', fontproperties=my_font, fontsize=14)
        axes[i].set_ylabel('Frequency', fontproperties=my_font, fontsize=12)
        axes[i].set_xlabel(f'{col}', fontproperties=my_font, fontsize=12)
        
        # Apply font to tick labels
        for label in axes[i].get_xticklabels():
            label.set_fontproperties(my_font)
        for label in axes[i].get_yticklabels():
            label.set_fontproperties(my_font)
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
    


%%time
fig = plot_histograms(dataset, num_cols, my_font)


def plot_pie_charts(df, cat_columns, my_font):
    cols = 3
    rows = (len(cat_columns) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle('Categorical Distribution', fontproperties=my_font, fontsize=18)  # å¢�å¤§æ€»æ ‡é¢˜å­—å�·
    
    colors = plt.cm.Set3.colors
    
    axes = axes.flatten()
    
    for i, col in enumerate(cat_columns):
        counts = df[col].value_counts()
        labels = counts.index.tolist()
        
        wedges, texts = axes[i].pie(
            counts,
            labels=labels,
            colors=[colors[j % len(colors)] for j in range(len(counts))],
            startangle=90,
            wedgeprops=dict(edgecolor='black', linewidth=0.8)
        )
        
        axes[i].set_title(f'{col}', fontproperties=my_font, fontsize=16)
        
        for text in texts:
            text.set_fontproperties(my_font)
            text.set_fontsize(14) 
        axes[i].axis('equal')
    
    for i in range(len(cat_columns), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


%%time
fig = plot_pie_charts(dataset, cat_cols, my_font)


def plot_violin_plots(df, numeric_columns, hue, my_font):
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("First argument must be a pandas DataFrame")
    
    # Validate and prepare hue column
    if hue is not None:
        if hue not in df.columns:
            raise ValueError(f"Hue column '{hue}' not found in DataFrame")
        # Convert hue to category if it's boolean or numeric
        if pd.api.types.is_bool_dtype(df[hue]) or pd.api.types.is_numeric_dtype(df[hue]):
            df[hue] = df[hue].astype('category')
    
    # Validate and prepare numeric columns
    valid_numeric = []
    for col in numeric_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        # Convert boolean columns to numeric
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not numeric")
        valid_numeric.append(col)
    
    # Set up subplot grid
    cols = 3
    rows = (len(valid_numeric) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    fig.suptitle('Distributions by Category', fontproperties=my_font, fontsize=18)
    
    # Handle single subplot case
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Use Set10 color palette
    colors = plt.cm.Set3.colors
    
    # Plot each numeric column
    for i, col in enumerate(valid_numeric):
        # Drop rows with missing values for current plot
        plot_data = df[[col, hue]].dropna() if hue else df[[col]].dropna()

        sns.violinplot(
            data=plot_data,
            x=hue,
            y=col,
            ax=axes[i],
            palette=colors,
            inner='quartile',
            linewidth=0.8,
            edgecolor='black'
        )
        
        # Set titles and labels with proper font
        axes[i].set_title(f'{col}', fontproperties=my_font, fontsize=16)
        axes[i].set_xlabel(hue, fontproperties=my_font, fontsize=14)
        axes[i].set_ylabel(col, fontproperties=my_font, fontsize=14)
        
        # Format tick labels
        for label in axes[i].get_xticklabels():
            label.set_fontproperties(my_font)
            label.set_fontsize(12)
        for label in axes[i].get_yticklabels():
            label.set_fontproperties(my_font)
            label.set_fontsize(12)
    
    # Hide unused subplots
    for i in range(len(valid_numeric), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


%%time
fig = plot_violin_plots(dataset, num_cols, 'y', my_font)


def plot_heatmap(df, numeric_columns=None, title="Correlation Heatmap", my_font=my_font, annot=True, cmap="coolwarm"):

    df = df.copy()
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in numeric_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not numeric")
    
    corr_matrix = df[numeric_columns].corr(method='spearman')
    
    plt.figure(figsize=(10, 8), dpi=240)

    heatmap = sns.heatmap(
        corr_matrix,
        annot=False,
        cmap=cmap,
        linewidths=0.5,
        linecolor='black',
        cbar=True
    )

    if annot:
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.iloc[i, j]
                # Add text annotation with custom font
                heatmap.text(
                    j + 0.5,  # x position
                    i + 0.5,  # y position
                    f"{value:.2f}",  # formatted text
                    ha='center', 
                    va='center',
                    fontproperties=my_font,
                    fontsize=12,
                    color='white',
                )
    
    plt.title(title, fontproperties=my_font, fontsize=18, pad=20)
    
    heatmap.set_xlabel(heatmap.get_xlabel(), fontproperties=my_font, fontsize=14)
    heatmap.set_ylabel(heatmap.get_ylabel(), fontproperties=my_font, fontsize=14)
    
    for label in heatmap.get_xticklabels():
        label.set_fontproperties(my_font)
        label.set_fontsize(12)
    for label in heatmap.get_yticklabels():
        label.set_fontproperties(my_font)
        label.set_fontsize(12)

    plt.tight_layout()
    
    return plt.gcf()


fig = plot_heatmap(dataset, [*num_cols, 'y'])

