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
from sklearn.feature_extraction.text import TfidfVectorizer


train_df =pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df =pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample=pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')


pd.set_option('display.max_colwidth', None)
train_df.sample(4)


test_df.head()


sample.head()


train_df.shape


train_df.dtypes


train_df.isnull().sum()


category_counts = train_df['Category'].value_counts().sort_values(ascending=False)
map_target1 = {category: i for i, category in enumerate(category_counts.index)}
map_target1


misconception_counts = train_df['Misconception'].value_counts().sort_values(ascending=False)
map_target2 = {misconception: i for i, misconception in enumerate(misconception_counts.index)}
map_target2


train_df['target1'] = train_df['Category'].map(map_target1).fillna(-1).astype(int)
train_df['target2'] = train_df['Misconception'].map(map_target2).fillna(-1).astype(int)
train_df.head(3)


train_df.target2.unique()


def create_sentence(row):
    return f"Question: {row['QuestionText']} Answer: {row['MC_Answer']} Explanation: {row['StudentExplanation']}"

train_df['sentence'] = train_df.apply(create_sentence, axis=1)
test_df['sentence'] = test_df.apply(create_sentence, axis=1)

tfidf1 = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), analyzer='word', max_df=0.95, min_df=2, max_features=10000)
all_sentences = pd.concat([train_df[['sentence']],test_df[['sentence']]], ignore_index=True)


all_sentences[:3]


train_df['sentence_word_count'] = train_df['sentence'].apply(lambda x: len(str(x).split()))
train_df[['sentence', 'sentence_word_count']].head()


print('Minimum word: ',train_df['sentence_word_count'].min())
print('Maximum word: ',train_df['sentence_word_count'].max())


plt.figure(figsize=(10, 6))
plt.hist(train_df['sentence_word_count'], bins=30, edgecolor='black')
plt.title('Word Counts in Sentence Column')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Category', y='sentence_word_count')
plt.xticks(rotation=45)
plt.title('Sentence Word Count by Category')
plt.xlabel('Category')
plt.ylabel('Word Count')
plt.tight_layout()
plt.show()


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def remove_stop_words(text):
    words = str(text).split()
    filtered_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
    return ' '.join(filtered_words)

train_df['sentence_no_stopwords'] = train_df['sentence'].apply(remove_stop_words)
train_df[['sentence', 'sentence_no_stopwords']].head()


train_df['sentence_word_count_2'] = train_df['sentence_no_stopwords'].apply(lambda x: len(str(x).split()))
train_df[['sentence_no_stopwords', 'sentence_word_count_2']].head()


print('Minimum word: ',train_df['sentence_word_count_2'].min())
print('Maximum word: ',train_df['sentence_word_count_2'].max())


plt.figure(figsize=(10, 6))
plt.hist(train_df['sentence_word_count_2'], bins=30, edgecolor='black')
plt.title('Word Counts in Sentence Column (no stop words)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Category', y='sentence_word_count_2')
plt.xticks(rotation=45)
plt.title('Sentence Word Count by Category (no stop words)')
plt.xlabel('Category')
plt.ylabel('Word Count')
plt.tight_layout()
plt.show()


train_df['QuestionText_length'] = train_df['QuestionText'].apply(len)

plt.figure(figsize=(12, 6))
plt.hist(train_df['QuestionText_length'], bins=30, edgecolor='black')
plt.title('Distribution of "QuestionText" Lengths')
plt.xlabel('Text Length (characters)')
plt.ylabel('Number of Questions')
plt.tight_layout()
plt.show()


train_df.Category.value_counts()


plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='Category', order=train_df['Category'].value_counts().index)
plt.title('Distribution of Target Column: Category', fontsize=16)
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


category_counts = train_df['Category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 12}
)
plt.title("Category Distribution (Pie Chart)", fontsize=16)
plt.show()


plt.figure(figsize=(15, 8))
sns.countplot(data=train_df, x='Misconception', order=train_df['Misconception'].value_counts().index)
plt.title('Distribution of Target Column: Misconception', fontsize=16)
plt.xlabel('Misconception')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


feature_summary = pd.DataFrame({
    'Column': train_df.columns,
    'Data Type': train_df.dtypes.values,
    'Missing Values': train_df.isnull().sum().values,
    'Unique Values': train_df.nunique().values
})

plt.figure(figsize=(12, 5))
sns.histplot(train_df['QuestionId'], bins=50, kde=True)
plt.title('Distribution of Question IDs', fontsize=14)
plt.xlabel('QuestionId')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

train_df['ExplanationLength'] = train_df['StudentExplanation'].astype(str).apply(len)

plt.figure(figsize=(12, 5))
sns.histplot(train_df['ExplanationLength'], bins=50, kde=True)
plt.title('Distribution of Explanation Lengths', fontsize=14)
plt.xlabel('Length of Explanation')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

top_questions = train_df['QuestionText'].value_counts().nlargest(10)

plt.figure(figsize=(12, 6))
sns.barplot(y=top_questions.index, x=top_questions.values, palette='viridis')
plt.title("Top 10 Most Frequent Question Texts", fontsize=14)
plt.xlabel("Frequency")
plt.ylabel("Question Text")
plt.tight_layout()
plt.show()


question_grouped = train_df.groupby('QuestionId').agg({
    'row_id': 'count',
    'Category': lambda x: x.value_counts().idxmax(),
    'StudentExplanation': lambda x: x.astype(str).apply(len).mean()
}).rename(columns={
    'row_id': 'ResponseCount',
    'Category': 'MostCommonCategory',
    'StudentExplanation': 'AvgExplanationLength'
}).reset_index()

top_questions = question_grouped.sort_values(by='ResponseCount', ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_questions, x='QuestionId', y='ResponseCount', hue='MostCommonCategory')
plt.title("Top 15 Question IDs by Number of Responses and Most Common Category")
plt.xlabel("Question ID")
plt.ylabel("Response Count")
plt.tight_layout()
plt.show()


top_question_ids = train_df['QuestionId'].value_counts().nlargest(10).index
top_questions_df = train_df[train_df['QuestionId'].isin(top_question_ids)]

plt.figure(figsize=(14, 7))
sns.countplot(data=top_questions_df, x='QuestionId', hue='Category', order=top_question_ids)
plt.title("Category Distribution by Top 10 Question IDs", fontsize=16)
plt.xlabel("Question ID")
plt.ylabel("Response Count")
plt.legend(title='Category')
plt.tight_layout()
plt.show()


train_df['ExplanationLength'] = train_df['StudentExplanation'].astype(str).apply(len)
length_by_category = train_df.groupby('Category')['ExplanationLength'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df, x='Category', y='ExplanationLength', order=length_by_category.index)
plt.title("StudentExplanation Distribution by Category", fontsize=16)
plt.xlabel("Category")
plt.ylabel("Explanation Length (Characters)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if 'Category' in train_df.columns:
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=train_df, x='Category', y='QuestionText_length')
    plt.title('"QuestionText" Length by "Category"')
    plt.xlabel('Category')
    plt.ylabel('QuestionText Length characters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    train_df.columns




