# IMPORT BASE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# IMPORT DATASETS
train_dataset = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

# List of both datasets
datasets = [('TRAIN DATASET', train_dataset), ('TEST DATASET', test_dataset)]


# Function to display data info and summary statistics
def understand_data(datasets_list: list):
    for name, df in datasets_list:
        print("="*50 + name + "="*50)
        display(df.head(5))
        print("="*42 +  name + " (SUMMARY STATS)" + "="*42)
        display(df.describe())
        print("="*46 + name + " (INFO)" + "="*48)
        display(df.info())
        print('\n')
    

understand_data(datasets)


def eda_function(datasets_list: list, base_fig_size: tuple,  palette: str, kde: bool=True, target_var: str='y'):
    """
    This function 'just like its name suggests' performs EDA; creates histograms and boxplots side-by-side
    for each individual numerical variable, and countplots with pie charts side-by-side for each categorical variable

    datasets_list: Takes a list of named tuples of available DataFrames (the training and test datasets)
    base_fig_size: Takes a tuple for the dimensions of the plots (also makes sure the diagram sizes are proportional
    to the number of columns in each canvas)
    palette: A string of the desired colour(not colorğŸ˜…) palette
    kde: Boolean value for whether kde lines should appear on histograms
    """
    for name, df in datasets_list:
        df = df.copy() # Use duplicates to avoid modification of the original datasets
        df = df.drop(['id'], axis=1, errors='ignore') # Remove the useless id colğŸ˜’

        # Identify numerical and categorical columns
        num_list = df.select_dtypes(include=np.number).columns.tolist() # returns something like ['age', 'balance', 'day'...]
        cat_list = df.select_dtypes(exclude=np.number).columns.tolist() # returns something like ['job', 'marital'...]

        # NUMERICAL VARIABLES
        n_rows_num = len(num_list)
        fig_height_num = base_fig_size[1] * n_rows_num
        fig, axes = plt.subplots(n_rows_num, 2, figsize=(base_fig_size[0], fig_height_num))

        if n_rows_num == 1:
            axes = np.expand_dims(axes, axis=0)  # Ensure 2D

        for i, var_name in tqdm(enumerate(num_list), total=n_rows_num, desc=f"Plotting {name}'s numerical variables"):
            if target_var in df.columns:
                sns.histplot(data=df, x=var_name, ax=axes[i, 0], kde=kde, hue=target_var, palette=palette)
                sns.boxplot(data=df, y=var_name, x=target_var, ax=axes[i, 1], hue=target_var, palette=palette)
            else:
                sns.histplot(data=df, x=var_name, ax=axes[i, 0], kde=kde, palette=palette)
                sns.boxplot(data=df, x=var_name, ax=axes[i, 1], palette=palette)

            axes[i, 0].set_title(f'Distribution of {var_name}')
            axes[i, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
            axes[i, 1].set_title(f'Boxplot of {var_name}')
            axes[i, 1].grid(True, linestyle='--', linewidth=0.5, alpha=0.9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.subplots_adjust(hspace=0.4)
        plt.show()

        # CATEGORICAL VARIABLES
        n_rows_cat = len(cat_list)
        fig2, axes2 = plt.subplots(n_rows_cat, 2, figsize=(base_fig_size[0], base_fig_size[1] * n_rows_cat))

        if n_rows_cat == 1:
            axes2 = np.expand_dims(axes2, axis=0)

        for i, var_name in tqdm(enumerate(cat_list), total=n_rows_cat, desc=f"Plotting {name}'s categorical variables"):
            # Countplot
            if target_var in df.columns:
                sns.countplot(data=df, x=var_name, ax=axes2[i, 0], hue=target_var, palette=palette)
            else:
                sns.countplot(data=df, x=var_name, ax=axes2[i, 0], palette=palette)

            axes2[i, 0].set_xticks(axes2[i, 0].get_xticks())
            axes2[i, 0].set_xticklabels(axes2[i, 0].get_xticklabels(), rotation=45, fontsize=10)
            axes2[i, 0].set_title(f"Count of {var_name}")
            axes2[i, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.9)

            # Pie chart
            value_counts = df[var_name].value_counts()
            wedges, texts, autotexts = axes2[i, 1].pie(
                value_counts,
                labels=None,
                autopct='%1.1f%%',
                colors=sns.color_palette(palette, value_counts.nunique()),
                startangle=90,
                explode=[0.05] * value_counts.nunique(),
                shadow=True
            )
            
            axes2[i, 1].legend(
                wedges,
                value_counts.index,
                title=var_name,
                loc='center left',
                bbox_to_anchor=(1, 0, 0.5, 1) 
            )
            axes2[i, 1].set_title(f"Percentage Distribution of {var_name}")
            axes2[i, 1].set_aspect('equal') 

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig2.subplots_adjust(hspace=0.4)
        plt.show()

        
eda_function(
    datasets_list=datasets,
    base_fig_size=(20, 8),
    palette='Spectral'
)


# Function to draw correlation matrices for both datasets
def corr_matrix(datasets_list: list):
    fig, axes = plt.subplots(1, len(datasets_list), figsize=(6 * len(datasets_list), 6))

    for i, (name, df) in enumerate(datasets_list):
        df = df.copy().drop(['id'], axis=1)
        corr = df.select_dtypes(include=np.number).corr()
        sns.heatmap(corr, annot=True, fmt=".2f", ax=axes[i])
        axes[i].set_title(f"Correlation Matrix For {name}", fontsize=10)

    plt.tight_layout()
    plt.show()

corr_matrix(datasets)




