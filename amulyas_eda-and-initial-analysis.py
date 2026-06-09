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


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df


train_df.columns


train_df.info()


train_df.describe()


# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_df.columns,
                              '[TRAIN] No. of Missing Values': train_df.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_df.isnull().sum().values)/len(train_df)*100)})

missing_values_test = pd.DataFrame({'Feature': test_df.columns,
                             '[TEST] No.of Missing Values': test_df.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_df.isnull().sum().values)/len(test_df)*100)})


unique_values = pd.DataFrame({'Feature': train_df.columns,
                              'No. of Unique Values[FROM TRAIN]': train_df.nunique().values})

feature_types = pd.DataFrame({'Feature': train_df.columns,
                              'DataType': train_df.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df



numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = []


# Analysis of all NUMERICAL features

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
train_df['Dataset'] = 'Train'
test_df['Dataset'] = 'Test'

variables = [col for col in train_df.columns if col in numerical_variables]

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_df, test_df]), x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_df, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_df, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN and TEST]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each variable
for variable in variables:
    create_variable_plots(variable)

# Drop the 'Dataset' column after analysis
train_df.drop('Dataset', axis=1, inplace=True)
test_df.drop('Dataset', axis=1, inplace=True)



# Define color palettes
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', 
                     '#7e03a8', '#35b779', '#fde725', '#440154', 
                     '#90d743', '#482173', '#22a884', '#f8961e']

# Here we reuse the same palette for countplot, but you can define a different one if preferred
countplot_palette = pie_chart_palette  

def create_target_plots(variable):
    sns.set_style('whitegrid')
    
    # Combine train_data and non-null rows from original_data
    combined_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Pie Chart ---
    # Using the value counts from train_data for the pie chart.
    counts = train_df[variable].value_counts()
    # Use as many colors as needed from the palette
    colors = pie_chart_palette[:len(counts)]
    axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=140, wedgeprops=dict(width=0.3))
    axes[0].set_title(f"Pie Chart for {variable}")
    
    # --- Countplot (Bar Graph) ---
    sns.countplot(
        data=combined_data, 
        x=variable, 
        palette=countplot_palette,  # Using a palette for different colors
        ax=axes[1],
        alpha=0.8  # Setting 80% opacity
    )
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Bar Graph for {variable} [TRAIN, TEST]")
    
    plt.tight_layout()
    plt.show()

# Example usage:
create_target_plots(target_variable)








