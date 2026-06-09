# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install autoviz


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from IPython.display import display, HTML
import warnings
warnings.filterwarnings("ignore")


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')

# Set index
train.set_index('id', inplace=True)
test.set_index('id', inplace=True)

train.head(5)


# Target distribution
plt.figure(figsize=(12, 5))
ax = sns.countplot(y=train['Fertilizer Name'], 
                 order=train['Fertilizer Name'].value_counts().index)
plt.title('Fertilizer Distribution', fontsize=20, pad=20)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Fertilizer Name', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add annotations
for p in ax.patches:
    width = p.get_width()
    plt.annotate(f'{width/train.shape[0]:.1%}', 
                 (width + 20, p.get_y() + p.get_height()/2),
                 ha='left', va='center', fontsize=11)

plt.tight_layout()
plt.show()


# Select numerical features
num_features = ['Temparature', 'Humidity', 'Moisture', 
               'Nitrogen', 'Potassium', 'Phosphorous']

# Plot distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.histplot(train[col], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f'{col} Distribution', fontsize=16)
    axes[i].set_xlabel(col, fontsize=14)
    axes[i].set_ylabel('Density', fontsize=14)
    axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics
    stats = train[col].describe()
    text = f"Mean: {stats['mean']:.1f}\nStd: {stats['std']:.1f}\nMin: {stats['min']:.1f}\nMax: {stats['max']:.1f}"
    axes[i].annotate(text, xy=(0.7, 0.7), xycoords='axes fraction', 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", lw=1))

plt.suptitle('Numerical Features Distribution', fontsize=20, y=0.99)
plt.tight_layout()
plt.show()


# Categorical features
cat_features = ['Soil Type', 'Crop Type']

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for i, col in enumerate(cat_features):
    counts = train[col].value_counts()
    ax = sns.barplot(x=counts.values, y=counts.index, ax=axes[i])
    axes[i].set_title(f'{col} Distribution', fontsize=18)
    axes[i].set_xlabel('Count', fontsize=14)
    axes[i].set_ylabel(col, fontsize=14)
    axes[i].tick_params(axis='y', labelsize=12)
    
    # Add percentages
    total = len(train)
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 20, p.get_y() + p.get_height()/2, 
                f'{width/total:.1%}', 
                ha='left', va='center', fontsize=12)

plt.suptitle('Categorical Features Distribution', fontsize=20, y=0.95)
plt.tight_layout()
plt.show()


# Create cross-tab visualizations
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Crop vs Fertilizer
ct1 = pd.crosstab(train['Crop Type'], train['Fertilizer Name'], normalize='index')
sns.heatmap(ct1, annot=True, fmt='.1%', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Percentage'})
axes[0].set_title('Fertilizer Preference by Crop Type', fontsize=18)
axes[0].set_xlabel('Fertilizer Name', fontsize=14)
axes[0].set_ylabel('Crop Type', fontsize=14)

# Soil vs Fertilizer
ct2 = pd.crosstab(train['Soil Type'], train['Fertilizer Name'], normalize='index')
sns.heatmap(ct2, annot=True, fmt='.1%', cmap='YlGnBu', ax=axes[1], cbar_kws={'label': 'Percentage'})
axes[1].set_title('Fertilizer Preference by Soil Type', fontsize=18)
axes[1].set_xlabel('Fertilizer Name', fontsize=14)
axes[1].set_ylabel('Soil Type', fontsize=14)

plt.suptitle('Fertilizer Preferences Analysis', fontsize=22, y=1.03 )
plt.tight_layout()
plt.show()


# Nutrient analysis
fig, axes = plt.subplots(1, 3, figsize=(22, 8))

# Nitrogen
sns.boxplot(x='Fertilizer Name', y='Nitrogen', data=train, ax=axes[0])
axes[0].set_title('Nitrogen Distribution by Fertilizer', fontsize=16)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Phosphorous
sns.boxplot(x='Fertilizer Name', y='Phosphorous', data=train, ax=axes[1])
axes[1].set_title('Phosphorous Distribution by Fertilizer', fontsize=16)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

# Potassium
sns.boxplot(x='Fertilizer Name', y='Potassium', data=train, ax=axes[2])
axes[2].set_title('Potassium Distribution by Fertilizer', fontsize=16)
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.suptitle('Soil Nutrient Profiles by Recommended Fertilizer', fontsize=22, y=1.03)
plt.tight_layout()
plt.show()


# Encode categorical features for correlation
encoder = LabelEncoder()
train_encoded = train.copy()
train_encoded['Soil Type'] = encoder.fit_transform(train['Soil Type'])
train_encoded['Crop Type'] = encoder.fit_transform(train['Crop Type'])
train_encoded['Fertilizer Name'] = encoder.fit_transform(train['Fertilizer Name'])

# Correlation matrix
plt.figure(figsize=(12, 7))
corr = train_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, linewidths=0.5, annot_kws={"size": 12})
plt.title('Feature Correlation Matrix', fontsize=20, pad=20)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# Sample data for performance
sample_df = train.sample(1000, random_state=42)

# Create pairplot
g = sns.pairplot(sample_df, hue='Fertilizer Name', 
                vars=['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous'], 
                palette='viridis', height=3, aspect=1.2,
                plot_kws={'alpha': 0.7, 's': 50})
g.fig.suptitle('Feature Relationships by Fertilizer Type', y=1.02, fontsize=20)
plt.tight_layout()
plt.show()


import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class

%matplotlib inline

AV = AutoViz_Class()

dfte = AV.AutoViz(
    filename="",              
    sep=",",                
    depVar="Fertilizer Name", 
    dfte=train,                 
    header=0,
    verbose=1,
    lowess=False,
    chart_format='svg',
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
    save_plot_dir=None
)


