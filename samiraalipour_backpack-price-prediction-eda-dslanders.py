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


# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import math
from IPython.display import display  
 

import warnings
warnings.filterwarnings("ignore")


#Loading the Dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
original_data = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')


print('Shape of Train data is : ' , train_data.shape)
print('Shape of Original data is : ' , original_data.shape)
print('Train data columns: ' , train_data.columns)
print('Original data columns: ' , original_data.columns)


train_data.head()


original_data.info()


train_data.info()


train_data = train_data.drop(['id'] , axis=1)

print('Shape of train data is : ' , train_data.shape)


train_data = pd.concat([train_data, original_data], ignore_index=True)
print('Shape of train data is : ' , train_data.shape)


# Extract test_ids for later use
test_ids = test_data['id']
test_data = test_data.drop(columns=['id'], axis=1)


duplicated_rows = train_data.duplicated()
sum(duplicated_rows)


# Remove duplicates
train_data = train_data.drop_duplicates().reset_index(drop=True)
duplicated_rows = train_data.duplicated()
sum(duplicated_rows)


#Check statistical information of numerical values

numerical_features = train_data.select_dtypes(include=[np.number])
train_data.describe(include=[np.number]).transpose()


#Check statistical information of categorical values

categorial_features = train_data.select_dtypes(include=object)
train_data.describe(include=object)


train_data.isna().sum()


# Missing values heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap', fontsize=16)
plt.show()



# Get the number of unique values for each column
unique_counts = train_data.nunique()
print(unique_counts)


threshold = 10  

# Compute value frequencies for columns with unique values below the threshold  
value_frequencies = {
    col: train_data[col].value_counts().reset_index(name="count")
    for col in train_data.columns if unique_counts[col] <= threshold
}  

# Display results in a structured table format  
for col, frequencies in value_frequencies.items():  
    print(f"\nColumn: '{col}' (Unique values: {unique_counts[col]})")  
    display(frequencies.style.set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
            .set_properties(**{"text-align": "center"}))  


def plot_feature_distributions(data, target='Price', n_cols=3, categorical_override=['Compartments']):
    # Define colors
    color_numeric = "#fc8d62"
    color_categorical = "#66c2a5"

    # Separate features
    features = [col for col in data.columns if col != target]
    
    # Calculate number of rows needed
    n_rows = int(np.ceil((len(features) + 1) / n_cols))  # +1 for target histogram
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()
    
    # Plot histogram for the target variable
    sns.histplot(data=data, x=target, kde=True, ax=axes[0], color=color_numeric)
    axes[0].set_title(f"Distribution of {target}", fontsize=14, fontweight='bold')
    axes[0].set_xlabel(target, fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot features
    for idx, col in enumerate(features, start=1):
        is_categorical = col in categorical_override or data[col].dtype not in ['int64', 'float64']
        
        if is_categorical:
            # Categorical column
            value_counts = data[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[idx], color=color_categorical)
            axes[idx].set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')

            # Rotate x-axis labels for readability
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right', fontsize=10)

            # Adjust bar width and spacing for clarity
            axes[idx].set_xlim(-0.5, len(value_counts) - 0.5)
        else:
            # Numeric column
            sns.histplot(data=data, x=col, kde=True, ax=axes[idx], color=color_numeric)
            axes[idx].set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')
            axes[idx].grid(True, linestyle='--', alpha=0.7)  # Add grid for better visualization

        axes[idx].set_xlabel(col, fontsize=12)
        axes[idx].set_ylabel("Count", fontsize=12)

    # Remove extra empty plots
    for i in range(len(features) + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# Usage
plot_feature_distributions(train_data)



# One-Hot Encoding for categorical features
encoded_data = pd.get_dummies(train_data, drop_first=False)

# Calculate the correlation matrix
correlation_matrix = encoded_data.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.2)
plt.title('Correlation Heatmap of Encoded Features', fontsize=16)
plt.show()



numerical_features = train_data.select_dtypes(include=[np.number]).columns
exclude_columns = ['Compartments']
numerical_features = [col for col in numerical_features if col not in exclude_columns]

# Calculate skewness for each numerical column
skew_newfeatures = train_data[numerical_features].skew().sort_values(ascending=False)

# Set skewness threshold
skew_limit = 0.75

# Identify numerical columns with unique values 0 and 1
binary_cols = [col for col in numerical_features if train_data[col].nunique() == 2]

# Filter out binary columns and apply skewness threshold
skew_cols = (
    skew_newfeatures
    .drop(index=binary_cols)  # Exclude binary columns
    .to_frame(name='Skew')    # Convert to DataFrame and rename the column to 'Skew'
    .query('abs(Skew) > @skew_limit')  # Filter for skewness beyond the limit
)

print(skew_cols)


# Define the number of columns per row in the subplot grid
n_cols = 2

# Calculate the number of rows needed
n_rows = (len(numerical_features) + n_cols - 1) // n_cols

# Create a figure and axes with the calculated number of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
axes = axes.flatten()

# Loop through each filtered column and create a boxplot
for i, col in enumerate(numerical_features):
    sns.boxplot(y=train_data[col].dropna(), ax=axes[i], palette=['#66c2a5'])  # Drop null values for plotting
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Value')

# Remove any extra empty subplots if the number of columns doesn't divide evenly
for j in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Define a function to find outliers based on IQR
def find_outliers(df):
    outliers = {}
    imputed_df = df.copy()
    for col in df.columns:
        v = df[col]
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr  
        upper_bound = q3 + 1.5 * iqr  
        outliers_count = ((v < lower_bound) | (v > upper_bound)).sum()
        perc = outliers_count * 100.0 / len(df)
        outliers[col] = (perc, outliers_count)
        print(f"Column {col} outliers = {perc:.2f}% ({outliers_count} out of {len(df)})")

    return outliers

# Find outliers in the DataFrame
find_outliers(train_data[numerical_features])


# Boxplot for Price by categorical variables
categorical_columns = ['Brand', 'Material', 'Size', 'Style', 'Color','Compartments']  # Select categorical columns of interest
plt.figure(figsize=(12, 8))

for i, col in enumerate(categorical_columns):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=train_data[col], y=train_data['Price'], palette='Set2')
    plt.title(f'Price vs {col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# Pairplot for numerical features
sns.pairplot(train_data[numerical_features], hue='Price', palette='coolwarm', height=2.5)
plt.suptitle('Pairplot of Numerical Features', fontsize=16)
plt.show()




# Define the features to plot
features_to_plot = ['Laptop Compartment_No', 'Laptop Compartment_Yes', 'Waterproof_Yes', 'Waterproof_No']

# Define the number of columns for the subplot grid
n_cols = 2
n_rows = (len(features_to_plot) + n_cols - 1) // n_cols

# Create a figure and axes with the calculated number of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 5))
axes = axes.flatten()

# Define color
color_numeric = "#fc8d62"

# Loop through the features and plot their relationship with Price using sns.histplot
for idx, feature in enumerate(features_to_plot):
    sns.histplot(data=encoded_data, x='Price', hue=feature, kde=True, ax=axes[idx], color=color_numeric, multiple='stack')
    axes[idx].set_title(f"Price Distribution by {feature}", fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Price', fontsize=12)
    axes[idx].set_ylabel('Count', fontsize=12)

# Remove extra empty subplots if the number of columns doesn't divide evenly
for j in range(len(features_to_plot), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


