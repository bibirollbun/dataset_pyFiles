import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")


# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
category_counts = df['Category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
plt.title('Distribution of Answer Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)


try:
    from wordcloud import WordCloud
    plt.subplot(2, 2, 2)
    text = ' '.join(df['MC_Answer'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Student Answers')
    plt.axis('off')
except ImportError:
    plt.subplot(2, 2, 2)
    plt.text(0.5, 0.5, 'Install wordcloud package\nfor this visualization', 
             ha='center', va='center')
    plt.title('Word Cloud Not Available')
    plt.axis('off')


plt.subplot(2, 2, 3)
df['answer_length'] = df['MC_Answer'].apply(len)
sns.boxplot(x='Category', y='answer_length', data=df, palette="Set2")
plt.title('Answer Length by Category')
plt.xlabel('Category')
plt.ylabel('Answer Length (chars)')
plt.xticks(rotation=45)


plt.subplot(2, 2, 4)
from collections import Counter

def get_top_words(group, n=5):
    all_text = ' '.join(group['MC_Answer'].dropna().str.lower())
    words = all_text.split()
    return Counter(words).most_common(n)

top_words = df.groupby('Category').apply(get_top_words)
for i, (category, words) in enumerate(top_words.items()):
    words, counts = zip(*words)
    y_pos = range(len(words))
    plt.barh(y_pos, counts, label=category, alpha=0.6)
    plt.yticks(y_pos, words)
    if i == 0:  # only show once
        plt.title('Top Words in Answers by Category')
        plt.xlabel('Frequency')
        plt.ylabel('Words')

plt.tight_layout()
plt.show()


print("\nSample Answers by Category:")
for category in df['Category'].unique():
    print(f"\nCategory: {category}")
    sample_answers = df[df['Category'] == category]['MC_Answer'].head(2).tolist()
    for i, answer in enumerate(sample_answers, 1):
        print(f"{i}. {answer}")


print("\nQuestion Analysis:")
print(f"Question ID: {df['QuestionId'].iloc[0]}")
print(f"Question Text: {df['QuestionText'].iloc[0]}")
print(f"Total Answers: {len(df)}")


from nltk.util import ngrams
from collections import Counter

# Extract bigrams
all_text = ' '.join(df['MC_Answer'].str.lower())
tokens = all_text.split()
bigrams = list(ngrams(tokens, 2))

# Get top bigrams
bigram_counts = Counter(bigrams).most_common(15)
bigrams, counts = zip(*bigram_counts)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(bigrams)), counts, color='skyblue')
plt.yticks(range(len(bigrams)), [' '.join(b) for b in bigrams])
plt.title('Top 15 Bigrams in Student Answers')
plt.xlabel('Frequency')
plt.show()


import squarify

# Prepare data
category_counts = df['Category'].value_counts()
sizes = category_counts.values
labels = [f"{label}\n{count} answers" for label, count in zip(category_counts.index, sizes)]

# Plot treemap
plt.figure(figsize=(12, 6))
squarify.plot(sizes=sizes, label=labels, alpha=0.7, 
             color=sns.color_palette("Spectral", len(sizes)))
plt.title('Distribution of Answer Categories (Treemap)')
plt.axis('off')
plt.show()


# If you have misconception data filled
if 'Misconception' in df and not df['Misconception'].isnull().all():
    plt.figure(figsize=(8, 8))
    df['Misconception'].value_counts().plot.pie(autopct='%1.1f%%', 
                                              colors=sns.color_palette("pastel"),
                                              startangle=90)
    plt.title('Distribution of Misconceptions')
    plt.ylabel('')
else:
    print("No misconception data available for visualization")


plt.figure(figsize=(10, 6))
sns.violinplot(x='Category', y='answer_length', data=df, palette="muted", inner="quartile")
plt.title('Distribution of Answer Lengths by Category')
plt.xlabel('Category')
plt.ylabel('Answer Length (characters)')
plt.show()


from sklearn.feature_extraction.text import CountVectorizer

# Get word counts by category
vec = CountVectorizer(stop_words='english', max_features=10)
word_counts = vec.fit_transform(df['MC_Answer'])
words = vec.get_feature_names_out()

# Create DataFrame for plotting
word_df = pd.DataFrame(word_counts.toarray(), columns=words)
word_df['Category'] = df['Category']

# Aggregate and plot
word_agg = word_df.groupby('Category').sum().T
word_agg.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Most Common Words by Answer Category')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend(title='Category')
plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px

# Extract features
tfidf = TfidfVectorizer(stop_words='english', max_features=5)
features = tfidf.fit_transform(df['MC_Answer']).toarray()
feature_names = tfidf.get_feature_names_out()

# Create radar chart
fig = px.line_polar(
    pd.DataFrame(features, columns=feature_names).mean().reset_index(),
    r=0, theta='index', line_close=True,
    title="Top TF-IDF Features Radar"
)
fig.show()

