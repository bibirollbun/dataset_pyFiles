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


import pandas as pd

BASE_PATH = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/'
train_df = pd.read_parquet(f'{BASE_PATH}/train.parquet')
test_df = pd.read_parquet(f'{BASE_PATH}/test.parquet')


train_df


test_df


train_df.describe()


# Look at the data structure
print("Training data shape:", train_df.shape)
print("\nColumns:", train_df.columns.tolist())
print("\nSample data:")
print(train_df.head())


# Basic data analysis
print("Number of unique languages:", train_df['language'].nunique())
print("\nLanguage distribution:")
print(train_df['language'].value_counts().head())

print("\nNumber of unique models:")
print("Model A:", train_df['model_a'].nunique())
print("Model B:", train_df['model_b'].nunique())

print("\nWinner distribution:")
print(train_df['winner'].value_counts(normalize=True))


def process_data(df):
    # Create binary labels (1 for model_a, 0 for model_b)
    df['label'] = (df['winner'] == 'model_a').astype(int)
    
    # Combine prompt and responses
    df['text_a'] = 'Prompt: ' + df['prompt'] + '\nResponse: ' + df['response_a']
    df['text_b'] = 'Prompt: ' + df['prompt'] + '\nResponse: ' + df['response_b']
    
    return df

train_df = process_data(train_df)


train_df


import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
import scipy.stats as stats
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


def create_visualizations(df):
    # Set up the plotting style
    plt.style.use('seaborn')
    
    # 1. Language Distribution Visualization
    plt.figure(figsize=(12, 6))
    top_10_langs = df['language'].value_counts().head(10)
    sns.barplot(x=top_10_langs.values, y=top_10_langs.index)
    plt.title('Top 10 Languages Distribution')
    plt.xlabel('Number of Samples')
    plt.tight_layout()
    
    # 2. Response Length Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['response_a_length'], bins=50, alpha=0.5, label='Model A')
    plt.hist(df['response_b_length'], bins=50, alpha=0.5, label='Model B')
    plt.title('Response Length Distribution by Model')
    plt.xlabel('Response Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    # 3. Winner vs Loser Length Box Plot
    plt.figure(figsize=(8, 6))
    winner_loser_lengths = pd.DataFrame({
        'Winner': df['winner_length'],
        'Loser': df['loser_length']
    })
    sns.boxplot(data=winner_loser_lengths)
    plt.title('Response Length Distribution: Winners vs Losers')
    plt.ylabel('Length (characters)')
    plt.tight_layout()



def perform_statistical_analysis(df):
    print("\n6. STATISTICAL ANALYSIS")
    print("-" * 50)
    
    # T-test for winner vs loser lengths
    t_stat, p_value = stats.ttest_ind(df['winner_length'], df['loser_length'])
    print("\nT-test for winner vs loser lengths:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Chi-square test for language and winner
    contingency = pd.crosstab(df['language'], df['winner'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    print("\nChi-square test for language and winner:")
    print(f"chi2 statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Correlation between response length and winning
    length_diff = df['response_a_length'] - df['response_b_length']
    a_wins = (df['winner'] == 'model_a').astype(int)
    correlation = stats.pointbiserialr(length_diff, a_wins)
    print("\nCorrelation between length difference and winning:")
    print(f"Correlation coefficient: {correlation.correlation:.4f}")
    print(f"p-value: {correlation.pvalue:.4f}")


def analyze_response_patterns(df):
    print("\n7. RESPONSE PATTERN ANALYSIS")
    print("-" * 50)
    
    # Initialize vectorizer
    vectorizer = CountVectorizer(max_features=100, stop_words='english')
    
    # Analyze winning responses
    winning_responses = np.where(df['winner'] == 'model_a', 
                               df['response_a'], 
                               df['response_b'])
    
    # Get most common words in winning responses
    winning_words = vectorizer.fit_transform(winning_responses)
    word_freq = pd.DataFrame(
        winning_words.sum(axis=0).T,
        index=vectorizer.get_feature_names_out(),
        columns=['frequency']
    ).sort_values('frequency', ascending=False)
    
    print("\nTop 20 words in winning responses:")
    print(word_freq.head(20))
    
    # Calculate average response lengths by model and winner
    print("\nAverage response length by model and outcome:")
    df['response_length_diff'] = abs(df['response_a_length'] - df['response_b_length'])
    length_analysis = df.groupby('winner').agg({
        'response_length_diff': 'mean',
        'response_a_length': 'mean',
        'response_b_length': 'mean'
    }).round(2)
    print(length_analysis)


def enhanced_analysis(df):
    # Run previous analysis
    df = analyze_dataset(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Perform statistical tests
    perform_statistical_analysis(df)
    
    # Analyze response patterns
    analyze_response_patterns(df)
    
    return df

# Run the enhanced analysis
analyzed_df = enhanced_analysis(train_df)

