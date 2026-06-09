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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


print("Shape of Train Data", train_df.shape)


display(train_df.head(10))


train_df.info()


train_df.nunique()


train_df.describe().transpose()


plt.figure(figsize=(12, 6))
soil_counts = train_df['Soil Type'].value_counts()

# Plot barplot
ax = sns.barplot(x=soil_counts.index, y=soil_counts.values, palette="viridis")

plt.title("Distribution of Soil Types ", fontsize=15)
plt.xlabel("Soil Type")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Add percentage labels on top of each bar
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(14, 7))
crop_counts = train_df['Crop Type'].value_counts()

# Plot barplot
ax = sns.barplot(x=crop_counts.index, y=crop_counts.values, palette="magma")
plt.title("Distribution of Crop Types", fontsize=15)
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Add percentage labels
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(14, 7))  # Wider figure for more categories
fert_counts = train_df['Fertilizer Name'].value_counts()

# Plot barplot
ax = sns.barplot(x=fert_counts.index, y=fert_counts.values, palette="plasma")
plt.title("Distribution of Fertilizer Names", fontsize=15)
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=90)  # Rotate 90° if labels overlap

# Add percentage labels
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',
            ha='center', va='center', fontsize=9)  # Smaller font for tight spaces

plt.tight_layout()  # Prevent label cutoff
plt.show()


plt.figure(figsize=(10, 20))
pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name']).plot(kind='bar', stacked=False, colormap='viridis')
plt.title('Fertilizer Preference by Soil Type', fontsize=16)
plt.xlabel('Soil Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 20))
pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name']).plot(kind='bar', stacked=False, colormap='viridis')
plt.title('Fertilizer Preference by Crop Type', fontsize=16)
plt.xlabel('Crop Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# Compare average N-P-K levels per Soil
train_df.groupby('Soil Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().plot(kind='bar', figsize=(14, 6))
plt.title('Average Nutrient Levels by Soil Type')
plt.show()


# Compare average N-P-K levels per crop
train_df.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().plot(kind='bar', figsize=(14, 6))
plt.title('Average Nutrient Levels by Crop Type')
plt.show()


# Heatmap: Crop vs. Fertilizer (Counts)
cross_tab = pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name'])
plt.figure(figsize=(16, 8))
sns.heatmap(cross_tab, cmap='YlGnBu', annot=True, fmt='d')
plt.title('Crop-Fertilizer Frequency', fontsize=16)
plt.xticks(rotation=45)
plt.show()


numerical_df = train_df.select_dtypes(include=['int64', 'float64'])


numerical_df.columns


from scipy import stats
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# Get all pairs of numerical columns
column_pairs = combinations(['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'], 2)

# Set style
sns.set(style="whitegrid")

# Loop through each pair and plot
for col1, col2 in column_pairs:
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with regression line
    sns.regplot(x=col1, y=col2, data=numerical_df, scatter_kws={'alpha':0.6})
    
    # Calculate statistics
    corr_coef, p_value = stats.pearsonr(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    slope, intercept, _, _, _ = stats.linregress(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    
    # Add statistics to plot
    stats_text = (f"Pearson r = {corr_coef:.2f}\n"
                  f"p-value = {p_value:.4f}\n"
                  f"Regression: y = {slope:.2f}x + {intercept:.2f}")
    
    plt.gcf().text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Titles and labels
    plt.title(f'{col1} vs {col2}', fontsize=14)
    plt.xlabel(col1, fontsize=12)
    plt.ylabel(col2, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Automated interpretation
    abs_r = abs(corr_coef)
    
    # Interpret Pearson r
    if abs_r >= 0.8:
        strength = "very strong"
    elif abs_r >= 0.6:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "very weak or no"
    
    direction = "positive" if corr_coef > 0 else "negative" if corr_coef < 0 else "no"
    
    # Interpret p-value
    if p_value < 0.001:
        sig_text = "highly statistically significant (p < 0.001)"
    elif p_value < 0.05:
        sig_text = "statistically significant (p < 0.05)"
    else:
        sig_text = "not statistically significant (p ≥ 0.05)"
    
    # Print interpretation
    print(f"\nInterpretation for {col1} vs {col2}:")
    print(f"- {strength} {direction} linear relationship")
    print(f"- The correlation is {sig_text}\n")
    print("-" * 60)  # Separator line


corr = abs(numerical_df.corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,8))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'Blues', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
plt.xticks(rotation = 50)   # Aesthetic purposes
plt.yticks(rotation = 20)   # Aesthetic purposes
plt.show()


from scipy.stats import skew  # For skewness calculation

# Set up subplots
n_cols = 3  # Number of columns in the grid
n_rows = (len(numerical_df.columns) // n_cols) + 1

# Create a figure with subplots
plt.figure(figsize=(15, 5 * n_rows))  # Adjust size as needed

# Loop through numerical columns and plot KDE + skewness
for i, column in enumerate(numerical_df.columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.kdeplot(data=numerical_df, x=column, fill=True)
    
    # Calculate skewness
    skewness = skew(numerical_df[column].dropna())  # Handle NaN if needed
    skew_text = f'Skewness: {skewness:.2f}'
    
    # Add skewness as text in the plot
    plt.text(0.05, 0.9, skew_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'KDE of {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Plot box plots
plt.figure(figsize=(15, 8))
for i, feature in enumerate(numerical_df.columns, 1):
    plt.subplot(2, 4, i)  # Adjust subplot grid as needed
    sns.boxplot(data=train_df, y=feature, color='skyblue')
    plt.title(f'Box Plot of {feature}')
    plt.tight_layout()
plt.show()

