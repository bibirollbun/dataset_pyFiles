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


data = pd.read_csv("/kaggle/input/train-csv/train.csv")
data.head(10)


data.shape


print(data.columns)


for harm in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print(harm + ":")
    print(data.comment_text[data[harm] == 1].values[42] + "\n")

print("vanilla:")
print(data.comment_text.values[1] + "\n")


import matplotlib.pyplot as plt
import seaborn as sns

freq_harms = {}
for harm in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    freq_harms[harm] = len(data[data[harm] == 1])

freq_isharm = {}
vanilla_mask = (data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] == 0).all(axis=1)
freq_isharm['vanilla'] = len(data[vanilla_mask])
freq_isharm['all_harm'] = len(data) - len(data[vanilla_mask])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=np.array(list(freq_harms.keys())), y=np.array(list(freq_harms.values())), ax=ax1)
sns.barplot(x=np.array(list(freq_isharm.keys())), y=np.array(list(freq_isharm.values())), ax=ax2)

plt.tight_layout()
plt.show()


data['vanilla_mask'] = vanilla_mask.astype(int)
data.columns


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

def clean_words(tokens):
	stops = stopwords.words('english')
	clean_tokens = [token.lower().strip() for token in tokens if token.lower() not in stops and token.isalnum()]

	return clean_tokens

new_len = int(np.round(len(data)*0.1))
data_sample = data.iloc[0:new_len].copy()

data_sample['tokens'] = data_sample['comment_text'].apply(lambda x: word_tokenize(x))
data_sample['tokens'] = data_sample['tokens'].apply(lambda x: clean_words(x))
data_sample[['comment_text', 'tokens']].head


harm_corpus = []
vanilla_corpus = []

for comment in data_sample['tokens'][data['vanilla_mask'] != 1]:
    for token in comment:
        harm_corpus.append(token)

for comment in data_sample['tokens'][data['vanilla_mask'] == 1]:
    for token in comment:
        vanilla_corpus.append(token)

print(harm_corpus[:50])
print()
print(vanilla_corpus[:50])


from collections import Counter

print("Top Words in harm comments:", Counter(harm_corpus).most_common(10))
print("\nTop Words in vanilla comments:", Counter(vanilla_corpus).most_common(10))


from wordcloud import WordCloud

def make_cloud(title, corpus):
    text = ' '.join(corpus)
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False,  # Disable bigram collocations
        normalize_plurals=False).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

make_cloud('Vanilla words', vanilla_corpus)
make_cloud('Harm words', harm_corpus)


'''
for harm in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    corpus = []
    for comment in data_sample['tokens'][data[harm] == 1]:
        for token in comment:
            corpus.append(token)
    make_cloud(harm, corpus)
'''


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

X = data_sample["tokens"].apply(lambda tokens: ' '.join(tokens))
y = data_sample["vanilla_mask"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_valid_vectors = vectorizer.transform(X_valid)


model = LogisticRegression()

model.fit(X_train_vectors, y_train)
predictions = model.predict(X_valid_vectors)

score = accuracy_score(y_valid, predictions)
score_rec = recall_score(y_valid, predictions)
score_f = f1_score(y_valid, predictions)
score_prec = precision_score(y_valid, predictions)

print(score)
print(score_rec)
print(score_f)
print(score_prec)


def test_comment(comment):
    comment_vector = vectorizer.transform([comment])
    prediction = model.predict(comment_vector) 
    if prediction[0] == 0: 
        print("Harmful comment detected.")
    else:
        print("Comment is not harmful.")

test_comment("Nicey flowers everywhere!")
test_comment("Gonna kick your ass, motherfucker!")
test_comment("Gonna kick your ass")

