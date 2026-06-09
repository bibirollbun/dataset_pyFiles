import pandas as pd

# load dataset

df_train = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')


# replace non-breaking spaces with spaces

def replace_xa0s(full_text):
    return full_text.replace('\xa0', ' ')

df_train['full_text'] = df_train['full_text'].apply(replace_xa0s)


from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
vectorizer.fit(df_train['full_text'])


!pip install textstat
!pip install pyspellchecker
!pip install language-tool-python
!pip install install-jdk


# 2. Use it to install JDK 17 on Linux
import jdk
from jdk.enums import OperatingSystem

# this will download & unpack OpenJDK 17 (default vendor=Adoptium) into ~/.jdk/
jdk.install('17', operating_system=OperatingSystem.LINUX)



import os

os.environ["JAVA_HOME"] = "/root/.jdk/jdk-17.0.15+6"
os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + ":" + os.environ["PATH"]

# then check
!java -version
!javac -version



import nltk
import numpy as np

import re
import textstat
from spellchecker import SpellChecker
import language_tool_python
import spacy
import nltk


# Initialize resources
_spell = SpellChecker()
_tool = language_tool_python.LanguageTool('en-US')
_nlp = spacy.load('en_core_web_sm')


# helper

def safe_cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


# number of words
def num_words(full_text):
    return len(nltk.word_tokenize(full_text))

# average word length
def avg_word_length(full_text):
    words = nltk.word_tokenize(full_text)
    return sum(len(word) for word in words) / len(words) if words else 0

# number of content words
def num_content_words(full_text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    content_words = [word for word in nltk.word_tokenize(full_text) if word.lower() not in stopwords]
    return len(content_words)

# number of unique content words
def num_unique_content_words(full_text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    content_words = [word for word in nltk.word_tokenize(full_text) if word.lower() not in stopwords]
    return len(set(content_words))

# ratio of content words to total words
def content_word_ratio(full_text):
    n_words = num_words(full_text)
    n_content_words = num_content_words(full_text)
    return n_content_words / n_words if n_words else 0

# ratio of unique content words to content words
def unique_content_word_ratio(full_text):
    n_content_words = num_content_words(full_text)
    n_unique_content_words = num_unique_content_words(full_text)
    return n_unique_content_words / n_content_words if n_content_words else 0





# number of sentences
def num_sentences(full_text):
    return len(nltk.sent_tokenize(full_text))

# average sentence length
def avg_sentence_length(full_text):
    n_sentences = num_sentences(full_text)
    n_words = num_words(full_text)
    return n_words / n_sentences if n_sentences else 0

# variance of sentence lengths
def sentence_length_variance(full_text):
    sentences = nltk.sent_tokenize(full_text)
    lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    if len(lengths) == 0:
        return 0
    return np.var(lengths)

# variance of sentence vector averages
def sentence_vector_variance(full_text):
    sentences = nltk.sent_tokenize(full_text)
    vectors = [vectorizer.transform([sentence]).toarray()[0] for sentence in sentences]
    if len(vectors) == 0:
        return 0
    return np.var([np.mean(vector) for vector in vectors])

# average similarity of neighboring sentences
def avg_neighboring_sentence_similarity(full_text):
    sentences = nltk.sent_tokenize(full_text)
    vectors = [vectorizer.transform([sentence]).toarray()[0] for sentence in sentences]
    if len(vectors) < 2:
        return 0
    similarities = []
    for i in range(len(vectors) - 1):
        similarities.append(safe_cosine_similarity(vectors[i], vectors[i + 1]))
    return np.mean(similarities)

# average similarity of sentences to full text
def avg_sentence_to_full_text_similarity(full_text):
    sentences = nltk.sent_tokenize(full_text)
    vectors = [vectorizer.transform([sentence]).toarray()[0] for sentence in sentences]
    full_text_vector = vectorizer.transform([full_text]).toarray()[0]
    if len(vectors) == 0:
        return 0
    return np.mean([safe_cosine_similarity(vector, full_text_vector) for vector in vectors])


# number of paragraphs
def num_paragraphs(full_text):
    return len(full_text.split('\n\n'))

# average paragraph length
def avg_paragraph_length(full_text):
    n_paragraphs = num_paragraphs(full_text)
    n_words = num_words(full_text)
    return n_words / n_paragraphs if n_paragraphs else 0

# variance of paragraph lengths
def paragraph_length_variance(full_text):
    paragraphs = full_text.split('\n\n')
    lengths = [len(nltk.word_tokenize(paragraph)) for paragraph in paragraphs]
    if len(lengths) == 0:
        return 0
    return np.var(lengths)

# variance of paragraph vector averages
def paragraph_vector_variance(full_text):
    paragraphs = full_text.split('\n\n')
    vectors = [vectorizer.transform([paragraph]).toarray()[0] for paragraph in paragraphs]
    if len(vectors) == 0:
        return 0
    return np.var([np.mean(vector) for vector in vectors])

# average similarity of neighboring paragraphs
def avg_neighboring_paragraph_similarity(full_text):
    paragraphs = full_text.split('\n\n')
    vectors = [vectorizer.transform([paragraph]).toarray()[0] for paragraph in paragraphs]
    if len(vectors) < 2:
        return 0
    similarities = []
    for i in range(len(vectors) - 1):
        similarities.append(safe_cosine_similarity(vectors[i], vectors[i + 1]))
    return np.mean(similarities)

# average similarity of paragraphs to full text
def avg_paragraph_to_full_text_similarity(full_text):
    paragraphs = full_text.split('\n\n')
    vectors = [vectorizer.transform([paragraph]).toarray()[0] for paragraph in paragraphs]
    full_text_vector = vectorizer.transform([full_text]).toarray()[0]
    if len(vectors) == 0:
        return 0
    return np.mean([safe_cosine_similarity(vector, full_text_vector) for vector in vectors])



cohesive_count = 0
grammar_count = 0
avg_clause_count = 0
readability_count = 0
def grammar_checker(text: str):
    global grammar_count
    n_words = num_words(text)
    """Return counts of grammar, punctuation, and tense-related flags from LanguageTool."""
    matches = _tool.check(text)
    
    grammar = sum(1 for m in matches if m.category == 'GRAMMAR')
    punctuation = sum(1 for m in matches if m.category == 'PUNCTUATION')
    typo = sum(1 for m in matches if m.category == 'TYPOS')

    grammar /= n_words
    punctuation /=  n_words
    typo /= n_words 
    grammar_count += 1
    if grammar_count%10 ==0:
        print(f' grammar count: {grammar_count}')
    return  {'grammar': grammar, 'punctuation':punctuation, 'typos': typo}



def cohesive_marker_counts(text: str):
    global cohesive_count
    n_words = num_words(text)
    """Return counts of common cohesive markers in the text."""
    markers = [
        'however', 'moreover', 'therefore', 'in addition',
        'furthermore', 'thus', 'consequently'
    ]
    txt = text.lower()
    counts = {}
    for m in markers:
        pattern = r"\b" + re.escape(m) + r"\b"
        counts[m] = len(re.findall(pattern, txt))
    for m in markers:
        counts[m] /= n_words
    cohesive_count += 1
    if cohesive_count%1000 ==0:
        print(f' cohesive count: {cohesive_count}')
    return counts 

def avg_clauses_per_sentence(text: str) -> float:
    """Return the average number of clauses per sentence using spaCy dependency parsing."""
    global avg_clause_count
    doc = _nlp(text)
    total_clauses = 0
    n_sents = 0
    
    for sent in doc.sents:
        subordinate = sum(1 for tok in sent if tok.dep_ in (
            'csubj','csubjpass','ccomp','xcomp','advcl'
        ))
        # main clause + subordinate clauses
        total_clauses += 1 + subordinate
        n_sents += 1
    score = (total_clauses / n_sents)
    avg_clause_count += 1
    if avg_clause_count%1000 ==0:
        print(f' avg_clause count: {avg_clause_count}')
    return score


def flesch_kincaid_reading_ease(text: str) -> float:
    """Return the Flesch–Kincaid Reading Ease score for the given text."""
    global readability_count
    score = textstat.flesch_reading_ease(text)
    readability_count += 1
    if readability_count%1000 ==0:
        print(f' readability_count : {readability_count}')
    return score





def extract_features(df):
    # advanced features

    #grammar_check = df['full_text'].apply(grammar_checker).apply(pd.Series)
    #df = df.join(grammar_check)
    df['flesch_reading_ease'] = df['full_text'].apply(flesch_kincaid_reading_ease)
    
    df['avg_clauses_per_sentence'] = df['full_text'].apply(avg_clauses_per_sentence)

    print("avg_clauses_per_sentence features processed")



    print("flesch_reading_ease features processed")


    cohesionScores = df['full_text'].apply(cohesive_marker_counts).apply(pd.Series)
    df = df.join(cohesionScores)
    print("cohesionScores features processed")
    print("Advanced features processed")
    # word features
    df['num_words'] = df['full_text'].apply(num_words)
    df['avg_word_length'] = df['full_text'].apply(avg_word_length)
    df['num_content_words'] = df['full_text'].apply(num_content_words)
    df['num_unique_content_words'] = df['full_text'].apply(num_unique_content_words)
    df['content_word_ratio'] = df['full_text'].apply(content_word_ratio)
    df['unique_content_word_ratio'] = df['full_text'].apply(unique_content_word_ratio)

    # sentence features
    df['num_sentences'] = df['full_text'].apply(num_sentences)
    df['avg_sentence_length'] = df['full_text'].apply(avg_sentence_length)
    df['sentence_length_variance'] = df['full_text'].apply(sentence_length_variance)
    df['sentence_vector_variance'] = df['full_text'].apply(sentence_vector_variance)
    df['avg_neighboring_sentence_similarity'] = df['full_text'].apply(avg_neighboring_sentence_similarity)
    df['avg_sentence_to_full_text_similarity'] = df['full_text'].apply(avg_sentence_to_full_text_similarity)

    # paragraph features
    df['num_paragraphs'] = df['full_text'].apply(num_paragraphs)
    df['avg_paragraph_length'] = df['full_text'].apply(avg_paragraph_length)
    df['paragraph_length_variance'] = df['full_text'].apply(paragraph_length_variance)
    df['paragraph_vector_variance'] = df['full_text'].apply(paragraph_vector_variance)
    df['avg_neighboring_paragraph_similarity'] = df['full_text'].apply(avg_neighboring_paragraph_similarity)
    df['avg_paragraph_to_full_text_similarity'] = df['full_text'].apply(avg_paragraph_to_full_text_similarity)


    return df

df_train = extract_features(df_train)


import matplotlib.pyplot as plt
import seaborn as sns

# correlation matrix

corr_features = df_train[[
    'num_words',
    'avg_word_length',
    'num_content_words',
    'num_unique_content_words',
    'content_word_ratio',
    'unique_content_word_ratio',
    'num_sentences',
    'avg_sentence_length',
    'sentence_length_variance',
    'sentence_vector_variance',
    'avg_neighboring_sentence_similarity',
    'avg_sentence_to_full_text_similarity',
    'num_paragraphs',
    'avg_paragraph_length',
    'paragraph_length_variance',
    'paragraph_vector_variance',
    'avg_neighboring_paragraph_similarity',
    'avg_paragraph_to_full_text_similarity',
    'score',
    #'grammar',
    #'punctuation',
    #'typos',
    'avg_clauses_per_sentence',
    'flesch_reading_ease',
    'however',
    'moreover',
    'therefore',
    'in addition',
    'furthermore',
    'thus',
    'consequently'
]]
plt.figure(figsize=(20, 20))
sns.heatmap(corr_features.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# dataset split

X = df_train.drop(columns=['essay_id', 'full_text', 'score'])
y = df_train['score']


# model training and evaluation

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score


class OrdinalGradientBoostingRegressor(GradientBoostingRegressor):
    def predict(self, X):
        return np.clip(np.round(super().predict(X)), 1, 6).astype(int)

class OrdinalRandomForestRegressor(RandomForestRegressor):
    def predict(self, X):
        return np.clip(np.round(super().predict(X)), 1, 6).astype(int)

class OrdinalSVR(SVR):
    def predict(self, X):
        return np.clip(np.round(super().predict(X)), 1, 6).astype(int)

models = {
    'OrdinalGBR': OrdinalGradientBoostingRegressor(n_estimators=100, random_state=42),
    'OrdinalRFR': OrdinalRandomForestRegressor(n_estimators=100, random_state=42),
    'OrdinalSVR': OrdinalSVR(kernel='rbf', C=1.0, epsilon=0.1)
}

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

qwk_scorer = make_scorer(qwk, greater_is_better=True)

for model_name, model in models.items():
    model.fit(X, y)
    cv_score = cross_val_score(model, X, y, cv=5, scoring=qwk_scorer).mean()
    print(f"{model_name} - cv: {cv_score:.4f}")


import pandas as pd

# dataset preparation

df_test = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')
df_test['full_text'] = df_test['full_text'].apply(replace_xa0s)

# feature extraction

df_test = extract_features(df_test)

# prediction

X_test = df_test.drop(columns=['essay_id', 'full_text'])
y_pred = models['OrdinalRFR'].predict(X_test)

# save csv

df_test['score'] = y_pred
df_test[['essay_id', 'score']].to_csv('submission.csv', index=False)

