import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import re
from wordcloud import WordCloud
from joblib import Parallel, delayed

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB,ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,classification_report

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)


import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet

resources = [
    "wordnet",
    "stopwords",
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "maxent_treebank_pos_tagger",
    "averaged_perceptron_tagger_eng"
]

for resource in resources:
    try:
        if resource in ["wordnet", "stopwords"]:
            nltk.data.find(f"corpora/{resource}")
        else:
            nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


stop_words = set(stopwords.words('english'))

lemma = WordNetLemmatizer()


df=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t')


df.head()


df.shape


df.info()


df.isnull().sum()


df.Sentiment.value_counts()


df.Sentiment.value_counts().plot(kind='bar')
plt.show()


def clean_text(text):
    text = text.lower()  # Convert the text to lowercase for uniformity
    text = re.sub(r'http\S+\s*', ' ', text)  # Remove URLs from the text
    text = re.sub(r'RT|cc', ' ', text)  # Remove retweet and "cc" mentions
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'@\S+', ' ', text)  # Remove mentions (usernames)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\d+', '', text)  # Remove digits from the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\r', '', text)  # Remove carriage return characters
    text = re.sub(r'[^\x00-\x7f]', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\&\#[0-9]+\;', ' ', text)  # Remove HTML entities
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', '', text)  # Remove emojis

    return text.strip()


df['Phrase'] = df['Phrase'].apply(clean_text)


df.Phrase.sample()


remove_stop_words = lambda row: " ".join([token for token in row.split(" ") \
                                          if token not in stop_words])


df.loc[:, 'Phrase'] = df['Phrase'].apply(remove_stop_words)


def lemmatize_word(tagged_token):
    """Returns lemmatized word given its tag."""
    root = []
    for token in tagged_token:
        tag = token[1][0]
        word = token[0]
        if tag.startswith('J'):
            root.append(lemma.lemmatize(word, wordnet.ADJ))
        elif tag.startswith('V'):
            root.append(lemma.lemmatize(word, wordnet.VERB))
        elif tag.startswith('N'):
            root.append(lemma.lemmatize(word, wordnet.NOUN))
        elif tag.startswith('R'):
            root.append(lemma.lemmatize(word, wordnet.ADV))
        else:
            root.append(word)
    return root

def lemmatize_doc(document):
    """Tags words then returns sentence with lemmatized words."""
    lemmatized_list = []
    tokenized_sent = sent_tokenize(document)
    for sentence in tokenized_sent:
        no_punctuation = sentence.translate(str.maketrans("", "", "`'\",.!?()"))
        tokenized_word = word_tokenize(no_punctuation)
        tagged_token = pos_tag(tokenized_word)
        lemmatized = lemmatize_word(tagged_token)
        lemmatized_list.extend(lemmatized)
    return " ".join(lemmatized_list)



df['Phrase'] = df['Phrase'].apply(lambda row: lemmatize_doc(row))


positive_text = ' '.join(df[df['Sentiment'] == 4]['Phrase'])
negative_text = ' '.join(df[df['Sentiment'] == 0]['Phrase'])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
negative_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_text)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Words')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Words')

plt.show()


tfidf = TfidfVectorizer(ngram_range=(1, 2))

x = tfidf.fit_transform(df.Phrase)
y = df.Sentiment


def train_and_evaluate_fast(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predict = model.predict(X_test)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, predict), 4),
        "Precision": round(precision_score(y_test, predict, average='weighted', zero_division=0), 4),
        "Recall": round(recall_score(y_test, predict, average='weighted', zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, predict, average='weighted', zero_division=0), 4),
    }

def algo_test(x, y, test_size=0.2, random_state=42, subsample=None):
    # Subsample data if too large (e.g., subsample=0.5 for 50% reduction)
    if subsample and subsample < 1.0:
        x, _, y, _ = train_test_split(x, y, train_size=subsample, random_state=random_state)

    models = [
        ("LogisticRegression", LogisticRegression()),
        ("BernoulliNB", BernoulliNB()),
        ("DecisionTree", DecisionTreeClassifier(max_depth=5)),
        ("RandomForest", RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=5)),
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    results = Parallel(n_jobs=-1)(
        delayed(train_and_evaluate_fast)(model, name, X_train, X_test, y_train, y_test)
        for name, model in models
    )

    results_df = pd.DataFrame(results)
    results_df.sort_values("F1 Score", ascending=False, inplace=True)

    print("\n⚡ Model Performance (Fast Mode):")
    print(results_df.to_markdown(index=False))

    return results_df


algo_test(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(x_train, y_train)

predict = model.predict(x_test)

accuracy = accuracy_score(y_test, predict)
report = classification_report(y_test, predict)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)


test_df=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip', sep='\t')


test_df.head()


df.shape


test_df.isnull().sum()


test_df['Phrase'] = test_df['Phrase'].fillna('')


test_df['Phrase'] = test_df['Phrase'].apply(clean_text)


test_df.loc[:, 'Phrase'] = test_df['Phrase'].apply(remove_stop_words)


test_df['Phrase'] = test_df['Phrase'].apply(lambda row: lemmatize_doc(row))


pred_x = tfidf.transform(test_df.Phrase)

predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['PhraseId'] = test_df['PhraseId']
submision['Sentiment'] = predictions


submision.to_csv('submission.csv', index=False)

