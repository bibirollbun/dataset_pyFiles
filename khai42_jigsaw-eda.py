import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter


train_df=pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df=pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample=pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')


train_df.head(3)


test_df.head(3)


sample.head(3)


train_df.dtypes


test_df.dtypes


train_df.isnull().sum().sum()


sns.set(style="white")

plt.figure(figsize=(6, 4))
sns.countplot(x='rule_violation', data=train_df)
plt.title('Distribution of Rule Violation (Target Variable)', fontsize=14)
plt.xlabel('Rule Violation (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['No Violation', 'Violation'], rotation=0)
plt.show()

top_subreddits = train_df['subreddit'].value_counts().head(10)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_subreddits.values, y=top_subreddits.index, palette='viridis')
plt.title('Top 10 Subreddits', fontsize=14)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Subreddit', fontsize=12)
plt.show()

top_rules = train_df['rule'].value_counts().head(5)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_rules.values, y=top_rules.index, palette='coolwarm')
plt.title('Top 5 Rules', fontsize=14)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Rule', fontsize=12)
plt.show()

train_df['body_length'] = train_df['body'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(train_df['body_length'], kde=False, color='b', bins=30)
plt.title('Distribution of Text Length in "body" Column', fontsize=14)
plt.xlabel('Text Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


train_df['positive_length_1'] = train_df['positive_example_1'].apply(len)
train_df['positive_length_2'] = train_df['positive_example_2'].apply(len)
train_df['negative_length_1'] = train_df['negative_example_1'].apply(len)
train_df['negative_length_2'] = train_df['negative_example_2'].apply(len)

plt.figure(figsize=(8, 6))
sns.boxplot(data=train_df[['positive_length_1', 'positive_length_2', 'negative_length_1', 'negative_length_2']])
plt.title('Comparison of Text Length for Positive vs Negative Examples', fontsize=14)
plt.ylabel('Text Length', fontsize=12)
plt.xticks(ticks=[0, 1, 2, 3], labels=['Positive 1', 'Positive 2', 'Negative 1', 'Negative 2'])
plt.show()
print()

text = " ".join(body for body in train_df['body'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Popular Words in "body" Column', fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='subreddit', hue='rule_violation', data=train_df, order=train_df['subreddit'].value_counts().index[:10])
plt.title('Rule Violation Distribution Across Top 10 Subreddits', fontsize=14)
plt.xlabel('Subreddit', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=90)
plt.show()


test_rule_counts = test_df['rule'].value_counts()

plt.figure(figsize=(12, 4))
sns.barplot(x=test_rule_counts.values , y=test_rule_counts.index)
plt.title("Distribution of Rules in the Test Dataset")
plt.xlabel("Rule")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 6))
sns.countplot(x='subreddit', hue='rule_violation', data=train_df, palette='Set2')
plt.xticks(rotation=90)
plt.title("Subreddit vs Rule Violation in Train Dataset")
plt.xlabel("Subreddit")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='subreddit', data=test_df, palette='Set2')
plt.xticks(rotation=90)
plt.title("Subreddit Distribution in Test Dataset")
plt.xlabel("Subreddit")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


train_df["positive_example_1_length"] = train_df["positive_example_1"].apply(len)
train_df["negative_example_1_length"] = train_df["negative_example_1"].apply(len)

plt.figure(figsize=(12, 6))

sns.histplot(train_df["positive_example_1_length"], bins=30, color='lightgreen', kde=False, label="Positive Example 1 Length")
sns.histplot(train_df["negative_example_1_length"], bins=30, color='lightcoral', kde=False, label="Negative Example 1 Length")

plt.title("Length Distribution of Positive and Negative Examples")
plt.xlabel("Length of Example Text")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

train_df["positive_example_2_length"] = train_df["positive_example_2"].apply(len)
train_df["negative_example_2_length"] = train_df["negative_example_2"].apply(len)

plt.figure(figsize=(12, 6))

sns.histplot(train_df["positive_example_2_length"], bins=30, color='lightgreen', kde=False, label="Positive Example 2 Length")
sns.histplot(train_df["negative_example_2_length"], bins=30, color='lightcoral', kde=False, label="Negative Example 2 Length")

plt.title("Length Distribution of Positive and Negative Example 2")
plt.xlabel("Length of Example Text")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


train_df["positive_example_1_length"] = train_df["positive_example_1"].apply(len)
train_df["negative_example_1_length"] = train_df["negative_example_1"].apply(len)

plt.figure(figsize=(8, 6))

sns.scatterplot(x=train_df["positive_example_1_length"], y=train_df["negative_example_1_length"],
                hue=train_df["rule_violation"], palette={0: 'lightgreen', 1: 'lightcoral'})

plt.title("Correlation between Positive and Negative Example 1 Lengths (Colored by Rule Violation)")
plt.xlabel("Positive Example 1 Length")
plt.ylabel("Negative Example 1 Length")
plt.tight_layout()
plt.show()

train_df["positive_example_2_length"] = train_df["positive_example_2"].apply(len)
train_df["negative_example_2_length"] = train_df["negative_example_2"].apply(len)

plt.figure(figsize=(8, 6))

sns.scatterplot(x=train_df["positive_example_2_length"], y=train_df["negative_example_2_length"],
                hue=train_df["rule_violation"], palette={0: 'lightgreen', 1: 'lightcoral'})

plt.title("Correlation between Positive and Negative Example 2 Lengths (Colored by Rule Violation)")
plt.xlabel("Positive Example 2 Length")
plt.ylabel("Negative Example 2 Length")
plt.tight_layout()
plt.show()


train_df["combined_examples"] = (train_df["positive_example_1"] + " " + 
                                  train_df["positive_example_2"] + " " + 
                                  train_df["negative_example_1"] + " " + 
                                  train_df["negative_example_2"])


pd.set_option('display.max_colwidth', None)
train_df["combined_examples"].head()


import re

simple_stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
])

def clean_and_remove_stopwords(text):
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove special characters (keep only letters and numbers)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize the text and remove stopwords
    words = text.split()
    words = [word for word in words if word not in simple_stop_words]
    
    # Return the cleaned text
    return " ".join(words)

train_df["cleaned_combined_examples"] = train_df["combined_examples"].apply(clean_and_remove_stopwords)

def lowercase_text(text):
    return text.lower()

train_df["cleaned_combined_examples"] = train_df["cleaned_combined_examples"].apply(lowercase_text)

train_df[["combined_examples", "cleaned_combined_examples"]].head()


train_df["combined_examples_length"] = train_df["combined_examples"].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(train_df["combined_examples_length"], bins=30, color='lightblue', kde=False)
plt.title("Length Distribution of Combined Examples (Character Count)")
plt.xlabel("Length of Combined Example Text (Characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
print()

train_df["combined_examples_word_count"] = train_df["combined_examples"].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 6))
sns.histplot(train_df["combined_examples_word_count"], bins=30, color='lightgreen', kde=False)
plt.title("Word Count Distribution of Combined Examples")
plt.xlabel("Word Count of Combined Example")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


train_df["cleaned_combined_examples_length"] = train_df["cleaned_combined_examples"].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(train_df["cleaned_combined_examples_length"], bins=30, color='lightblue', kde=False)
plt.title("Length Distribution of Cleaned Combined Examples (Character Count)")
plt.xlabel("Length of Cleaned Combined Example Text (Characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
print()
train_df["cleaned_combined_examples_word_count"] = train_df["cleaned_combined_examples"].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 6))
sns.histplot(train_df["cleaned_combined_examples_word_count"], bins=30, color='lightgreen', kde=False)
plt.title("Word Count Distribution of Cleaned Combined Examples")
plt.xlabel("Word Count of Cleaned Combined Example")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(data=train_df, x="cleaned_combined_examples_length", hue="rule_violation", kde=False, bins=30, palette="Set2")
plt.title("Text Length Distribution by Rule Violation")
plt.xlabel("Length of Cleaned Combined Example (Characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=train_df, x="cleaned_combined_examples_word_count", hue="rule_violation", kde=False, bins=30, palette="Set2")
plt.title("Word Count Distribution by Rule Violation")
plt.xlabel("Word Count of Cleaned Combined Example")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


rule_violation_posts = train_df[train_df["rule_violation"] == 1]["cleaned_combined_examples"]
all_words = ' '.join(rule_violation_posts).split()
word_counts = Counter(all_words)

most_common_words = word_counts.most_common(20)
most_common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

plt.figure(figsize=(12, 6))
sns.barplot(x='Frequency', y='Word', data=most_common_words_df, palette='viridis')
plt.title("Most Frequent Words in Posts with Rule Violations")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.tight_layout()
plt.show()


rule_violation_0_posts = train_df[train_df["rule_violation"] == 0]["cleaned_combined_examples"]
rule_violation_1_posts = train_df[train_df["rule_violation"] == 1]["cleaned_combined_examples"]
words_0 = ' '.join(rule_violation_0_posts).split()
words_1 = ' '.join(rule_violation_1_posts).split()

word_counts_0 = Counter(words_0)
word_counts_1 = Counter(words_1)
most_common_words_0 = word_counts_0.most_common(20)
most_common_words_1 = word_counts_1.most_common(20)

df_0 = pd.DataFrame(most_common_words_0, columns=['Word', 'Frequency'])
df_1 = pd.DataFrame(most_common_words_1, columns=['Word', 'Frequency'])
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(x='Frequency', y='Word', data=df_0, palette='Blues', ax=axes[0])
axes[0].set_title("Most Frequent Words in Posts with Rule Violation 0")
axes[0].set_xlabel("Frequency")
axes[0].set_ylabel("Word")

sns.barplot(x='Frequency', y='Word', data=df_1, palette='Reds', ax=axes[1])
axes[1].set_title("Most Frequent Words in Posts with Rule Violation 1")
axes[1].set_xlabel("Frequency")
axes[1].set_ylabel("Word")

plt.tight_layout()
plt.show()


plt.figure(figsize=(14,4))
sns.countplot(y='rule', hue='rule_violation', data=train_df, palette='Set2')
plt.title("Count of Rule Violations by Rule Type")
plt.ylabel("Rule")
plt.xlabel("Count of Rule Violations")
plt.tight_layout()
plt.show()




