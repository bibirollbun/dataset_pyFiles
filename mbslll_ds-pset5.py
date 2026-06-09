# Standard library imports
import os
import time

# Third-party imports
import joblib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure plotting style
plt.style.use('ggplot')

# Set pandas display option
pd.set_option('display.max_columns', 35)


df = pd.read_csv('/kaggle/input/data-science-5-sbu/train.csv')


df.shape


df.describe()


# df['Years Lived'] = df['Years Lived'].astype(int)
# df['Dependent Count'] = df['Dependent Count'].astype(int)
# df['Prior Claims'] = df['Prior Claims'].astype(int)
# df['Automobile Age'] = df['Automobile Age'].astype(int)
# df['Coverage Period'] = df['Coverage Period'].astype(int)


df.head()


df.info()


cat_features = df.select_dtypes(include=['object'])

for cat in cat_features:
    print('------------------------')
    print(f'Unique Values for {cat}:')
    print(df[cat].value_counts())


df.isna().sum()


def plot_numerical_distributions(data):
    """
    Plot distributions of all numerical columns in a dataset using subplots.
    
    Parameters:
    data (pd.DataFrame): Input dataframe containing the data
    """
    # Select only numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    
    # Calculate number of rows needed (3 columns per row)
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten in case of 2D array of axes
    
    # Plot each numerical column
    for i, col in enumerate(numerical_cols):
        if i < len(axes):  # Ensure we don't exceed the number of axes
            sns.histplot(data=data, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel('')
    
    # Hide any unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

plot_numerical_distributions(df)


df['Automobile Age'].value_counts()







