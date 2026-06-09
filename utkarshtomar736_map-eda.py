import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


DATA_DIR = "/kaggle/input/map-charting-student-math-misunderstandings"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sam_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")


train.head(5)


train.info()


train['Category'].value_counts().plot(kind='bar')


train['explanation_length'] = train['StudentExplanation'].fillna('').str.len()

# Visualize histogram
plt.figure(figsize=(10, 6))
sns.histplot(train['explanation_length'], bins=50, kde=True)
plt.title('Distribution of Student Explanation Lengths')
plt.xlabel('Explanation Length (characters)')
plt.ylabel('Count')
plt.show()


# Create target column
train['target'] = train['Category'] + ':' + train['Misconception']

# Plot top 20 most common labels
plt.figure(figsize=(12, 8))
counts = train['target'].value_counts()[:20]  # Limit to top 20 for readability
sns.barplot(x=counts.index, y=counts.values)
plt.xticks(rotation=90)
plt.title('Top 20 Most Common Target Labels')
plt.xlabel('Target Label')
plt.ylabel('Count')
plt.show()


%pip install -Uqq wordcloud 


from wordcloud import WordCloud

# Get unique categories
categories = train['Category'].unique()

for cat in categories:
    # Combine text for the category (handle NaN)
    text = ' '.join(train.loc[train['Category'] == cat, 'StudentExplanation'].fillna('').astype(str))
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Category: {cat}')
    plt.show()


from scipy.stats import ks_2samp

# Count missing
missing_count = train['StudentExplanation'].isna().sum()
print(f'Missing StudentExplanation in train: {missing_count}')

# Handle missing by filling placeholder
train['StudentExplanation'] = train['StudentExplanation'].fillna('no_explanation')
test['StudentExplanation'] = test['StudentExplanation'].fillna('no_explanation')  # Apply to test too

# Compute lengths for test
test['explanation_length'] = test['StudentExplanation'].str.len()

# KS test for distribution similarity (potential leakage check)
ks_stat, p_value = ks_2samp(train['explanation_length'], test['explanation_length'])
print(f'KS Test: Statistic={ks_stat:.4f}, p-value={p_value:.4f}')
if p_value < 0.05:
    print('Distributions differ significantly—watch for domain shift.')
else:
    print('Distributions are similar—no obvious leakage in lengths.')


plt.figure(figsize=(25, 15)) 
sns.boxplot(x='MC_Answer', y='explanation_length', hue='Category', data=train)
plt.title('Explanation Length by MC_Answer and Category')
plt.xlabel('Multiple Choice Answer')
plt.ylabel('Explanation Length')
plt.xticks(rotation=45, ha='right')  
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()




