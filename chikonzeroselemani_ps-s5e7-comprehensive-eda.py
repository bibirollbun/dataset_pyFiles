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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


#  STYLING
COLORS = {
    'introvert': '#3498db',      # Cool blue
    'extrovert': '#e67e22',      # Warm orange
    'neutral': '#95a5a6',        # Gray
    'accent': '#27ae60',         # Green
    'background': '#ecf0f1'      # Light gray
}
plt.style.use('ggplot')
custom_palette = {'Introvert': '#3498db', 'Extrovert': '#e74c3c'}
sns.set_palette(sns.color_palette(list(custom_palette.values())))
plt.rcParams['figure.facecolor'] = COLORS['background']


train_path = "/kaggle/input/playground-series-s5e7/train.csv"
test_path = "/kaggle/input/playground-series-s5e7/test.csv"

train = pd.read_csv(train_path, index_col = "id")
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")



# INITIAL INSPECTION
def initial_inspection(df):
    inspection = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_values': df.isna().sum(),
        'missing_%': (df.isna().mean()*100).round(2),
        'unique_values': df.nunique()
    })
    return inspection

display(initial_inspection(train))


# Personality distribution
personality_counts = train['Personality'].value_counts()
    
# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
# Pie chart
colors = [COLORS['introvert'], COLORS['extrovert']]
axes[0].pie(personality_counts.values, labels=personality_counts.index, 
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('ğŸ§  Personality Distribution', fontsize=16, fontweight='bold')
    
# Bar chart 
bars = axes[1].bar(personality_counts.index, personality_counts.values, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_title('ğŸ“Š Sample Counts by Personality', fontsize=16, fontweight='bold')
axes[1].set_ylabel('Number of Individuals')
    
    # Add value labels on bars
for bar, count in zip(bars, personality_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()
    


# Select numerical features
num_features = ['Time_spent_Alone', 'Social_event_attendance', 
               'Going_outside', 'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(18, 12))

for i, feature in enumerate(num_features, 1):
    plt.subplot(3, 2, i)
    
    # KDE plot split by personality
    sns.kdeplot(data=train, x=feature, hue='Personality', 
               palette=custom_palette, fill=True, alpha=0.6)
    
    plt.title(f'ğŸ“ˆ {feature} Distribution', fontsize=12)
    plt.xlabel(feature.replace('_', ' ').title(), fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(title='Personality')

plt.tight_layout()
plt.show()


# Select numerical features
num_features = ['Time_spent_Alone', 'Social_event_attendance', 
               'Going_outside', 'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(18, 15))

for i, feature in enumerate(num_features, 1):
    plt.subplot(3, 2, i)
    
    # Boxplot
    sns.boxplot(data=train, x='Personality', y=feature, 
                palette=custom_palette, width=0.5)
    plt.title(f'ğŸ“¦ {feature} Distribution', fontsize=12)
    plt.xlabel('Personality', fontsize=10)
    plt.ylabel(feature.replace('_', ' ').title(), fontsize=10)

plt.tight_layout()
plt.show()




# Violin plots for deeper distribution analysis
plt.figure(figsize=(18, 15))
for i, feature in enumerate(num_features, 1):
    plt.subplot(3, 2, i)
    
    # Violin plot
    sns.violinplot(data=train, x='Personality', y=feature,
                  palette=custom_palette, inner='quartile')
    plt.title(f'ğŸ�» {feature} Distribution', fontsize=12)
    plt.xlabel('Personality', fontsize=10)
    plt.ylabel(feature.replace('_', ' ').title(), fontsize=10)

plt.tight_layout()
plt.show()


cat_features = ['Stage_fear', 'Drained_after_socializing']

plt.figure(figsize=(14, 6))

for i, feature in enumerate(cat_features, 1):
    plt.subplot(1, 2, i)
    
    # Stacked bar plot
    ct = pd.crosstab(train[feature], train['Personality'])
    ct.plot(kind='bar', stacked=True, color=[COLORS['introvert'], COLORS['extrovert']],
           edgecolor='black', ax=plt.gca())
    
    plt.title(f'ğŸ“Š {feature.replace("_", " ").title()}', fontsize=12)
    plt.xlabel(feature.replace('_', ' ').title(), fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(rotation=0)
    plt.legend(title='Personality')

plt.tight_layout()
plt.show()


cat_features = ['Stage_fear', 'Drained_after_socializing']

plt.figure(figsize=(16, 6))

for i, feature in enumerate(cat_features, 1):
    plt.subplot(1, 2, i)
    
    # Create cross tabulation
    ct = pd.crosstab(train[feature], train['Personality'])
    total_responses = len(train)
    
    # Calculate percentages
    yes_pct = 100 * ct.loc['Yes'].sum() / total_responses
    no_pct = 100 * ct.loc['No'].sum() / total_responses
    yes_intro_pct = 100 * ct.loc['Yes', 'Introvert'] / ct.loc['Yes'].sum()
    yes_extro_pct = 100 * ct.loc['Yes', 'Extrovert'] / ct.loc['Yes'].sum()
    no_intro_pct = 100 * ct.loc['No', 'Introvert'] / ct.loc['No'].sum()
    no_extro_pct = 100 * ct.loc['No', 'Extrovert'] / ct.loc['No'].sum()
    
    # Outer ring (Yes/No)
    outer_sizes = [ct.loc['Yes'].sum(), ct.loc['No'].sum()]
    outer_labels = [
        f'Yes\n{outer_sizes[0]} ({yes_pct:.1f}%)', 
        f'No\n{outer_sizes[1]} ({no_pct:.1f}%)'
    ]
    outer_colors = [COLORS['accent'], COLORS['neutral']]
    wedges_outer, texts_outer, autotexts_outer = plt.pie(
        outer_sizes, labels=outer_labels, colors=outer_colors,
        wedgeprops=dict(width=0.4, edgecolor='w'), startangle=90,
        autopct='', textprops={'fontsize': 10}
    )
    
    # Inner ring (Personality breakdown)
    inner_sizes = [
        ct.loc['Yes', 'Introvert'], ct.loc['Yes', 'Extrovert'],
        ct.loc['No', 'Introvert'], ct.loc['No', 'Extrovert']
    ]
    inner_labels = [
        f'Introvert\n{yes_intro_pct:.1f}%', f'Extrovert\n{yes_extro_pct:.1f}%',
        f'Introvert\n{no_intro_pct:.1f}%', f'Extrovert\n{no_extro_pct:.1f}%'
    ]
    wedges_inner, texts_inner = plt.pie(
        inner_sizes, radius=0.6,
        colors=[COLORS['introvert'], COLORS['extrovert'],
                COLORS['introvert'], COLORS['extrovert']],
        wedgeprops=dict(width=0.4, edgecolor='w'),
        labels=inner_labels, labeldistance=0.75,
        textprops={'fontsize': 8, 'color': 'white', 'fontweight': 'bold'}
    )
    
    # Center text
    plt.text(0, 0, feature.replace('_', ' ').title(), 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Title
    plt.title(f'ğŸ�© {feature.replace("_", " ").title()} Distribution', 
              fontsize=14, pad=20)

plt.tight_layout()
plt.show()


# Convert categorical features to numerical for correlation
corr_df = train.copy()
corr_df['Personality'] = corr_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})
corr_df['Stage_fear'] = corr_df['Stage_fear'].map({'Yes': 1, 'No': 0})
corr_df['Drained_after_socializing'] = corr_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

plt.figure(figsize=(12, 8))
corr = corr_df.corr()

# Mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Heatmap with custom styling
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
           center=0, linewidths=0.5, cbar_kws={'shrink': 0.8})

plt.title('ğŸ”— Feature Correlation Matrix', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()


# Select key features for pairplot
key_features = ['Time_spent_Alone', 'Social_event_attendance', 
               'Friends_circle_size', 'Personality']

# Create pairplot with proper labeling
g = sns.pairplot(data=train[key_features], hue='Personality', 
                palette=custom_palette, corner=True,
                plot_kws={'alpha': 0.7, 'edgecolor': 'black', 's': 30},
                diag_kws={'alpha': 0.7, 'edgecolor': 'black'})

# Set proper labels for all axes
for ax in g.axes.flatten():
    if ax:
        # X-axis labels
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace('_', ' ').title(), fontsize=10)
        # Y-axis labels
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel().replace('_', ' ').title(), fontsize=10)

plt.suptitle('ğŸ�¨ Feature Relationships by Personality', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

