# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objects as go
import plotly.express as px


import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
test.tail(3)


train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
train.tail(2)


sub= pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')
sub.head(11)


plt.figure(figsize=(13,5))

sns.kdeplot(train['rule_violation'], shade=  True);


train['rule'].value_counts()


plt.style.use('ggplot')

plt.figure(figsize=(10,6))
train['subreddit'].value_counts().head(15).plot(kind='barh')
plt.title('Subreddit')
plt.xlabel('Count')
plt.ylabel('Subreddit')

#https://stackoverflow.com/questions/2051744/how-to-invert-the-x-or-y-axis
plt.gca().invert_yaxis() #invert yaxis Or invert_xaxis()
plt.show()


#3rd row, fourth column 

test.iloc[2,4]


test.iloc[2,7]


test.iloc[8,7]


import re
import string

import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
from tqdm.auto import tqdm
import matplotlib.style as style
style.use('fivethirtyeight')
from numpy import interp
from itertools import cycle


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

def remove_line_breaks(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text

#remove punctuation
def remove_punctuation(text):
    re_replacements = re.compile("__[A-Z]+__")  # such as __NAME__, __LINK__
    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
    '''Escape all the characters in pattern except ASCII letters and numbers'''
    tokens = word_tokenize(text)
    tokens_zero_punctuation = []
    for token in tokens:
        if not re_replacements.match(token):
            token = re_punctuation.sub(" ", token)
        tokens_zero_punctuation.append(token)
    return ' '.join(tokens_zero_punctuation)

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def lowercase(text):
    text_low = [token.lower() for token in word_tokenize(text)]
    return ' '.join(text_low)

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in word_tokens if word not in stop])
    return text

#remobe one character words
def remove_one_character_words(text):
    '''Remove words from dataset that contain only 1 character'''
    text_high_use = [token for token in word_tokenize(text) if len(token)>1]      
    return ' '.join(text_high_use)   
    
#%%
# Stemming with 'Snowball stemmer" package
def stem(text):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    text_stemmed = [stemmer.stem(token) for token in word_tokenize(text)]        
    return ' '.join(text_stemmed)

def lemma(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = nltk.word_tokenize(text)
    text_lemma = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokens])       
    return ' '.join(text_lemma)


#break sentences to individual word list
def sentence_word(text):
    word_tokens = nltk.word_tokenize(text)
    return word_tokens
#break paragraphs to sentence token 
def paragraph_sentence(text):
    sent_token = nltk.sent_tokenize(text)
    return sent_token    


def tokenize(text):
    """Return a list of words in a text."""
    return re.findall(r'\w+', text)

def remove_numbers(text):
    no_nums = re.sub(r'\d+', '', text)
    return ''.join(no_nums)



def clean_text(text):
    _steps = [
    remove_line_breaks,
    remove_one_character_words,
    remove_special_characters,
    lowercase,
    remove_punctuation,
    remove_stopwords,
    stem,
    remove_numbers
]
    for step in _steps:
        text=step(text)
    return text   
#%%


#https://stackoverflow.com/questions/55557004/getting-attributeerror-float-object-has-no-attribute-replace-error-while
#To avoid with tqdm AttributeError: 'float' object has no attribute

train["positive_example_1"] = train["positive_example_1"].astype(str)
train["positive_example_1"] = [x.replace(':',' ') for x in train["positive_example_1"]]


words = train["positive_example_1"].values


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

ls = []

for i in words:
    ls.append(str(i))


ls[:5]


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

# The wordcloud 
plt.figure(figsize=(16,13))
wc = WordCloud(background_color="black", color_func=lambda *args, **kwargs: "white", max_words=1000, max_font_size= 200,  width=1600, height=800)
wc.generate(" ".join(ls))
plt.title("Positive Examples", fontsize=20)
plt.imshow(wc.recolor(random_state=17), alpha=0.98, interpolation="bilinear", )
plt.axis('off');


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

plt.figure(figsize = (4,4))

sns.barplot(data = train, y = 'row_id', x = 'rule_violation', color = 'c')
plt.xticks(fontsize=27, rotation=0)
plt.yticks(fontsize=31, rotation=0)
plt.xlabel('Rule Violations', fontsize = 21)
plt.ylabel('')
plt.title('Rule Violations', fontsize = 30);


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
#original 2018
np.random.seed(2025)
import nltk


stemmer = SnowballStemmer('english')


nltk.download('wordnet')


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


train['positive_example_1'].iloc[10]


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

doc_sample = train['positive_example_1'].iloc[10]
print('original document: ')

words = []

for word in doc_sample.split(' '):
    words.append(word)
    
    
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


train['positive_example_1'] = train['positive_example_1'].astype(str)


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

words = []

for i in train['positive_example_1']:
        words.append(i.split(' '))


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

dictionary = gensim.corpora.Dictionary(words)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# Filter out tokens in the dictionary by their frequency.

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=10000)


bow_corpus = [dictionary.doc2bow(doc) for doc in words]
#Original was 4310 out of index
bow_corpus[1043]


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

bow_doc_1043 = bow_corpus[1043]

for i in range(len(bow_doc_1043)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_1043[i][0], 
                                               dictionary[bow_doc_1043[i][0]], 
bow_doc_1043[i][1]))


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

lda_model = gensim.models.LdaMulticore(bow_corpus,
                                       num_topics=10,
                                       id2word=dictionary,
                                       passes=2,
                                       workers=2)


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf,
                                             num_topics=10,
                                             id2word=dictionary,
                                             passes=2,
                                             workers=4)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


train.iloc[49,4]


train.iloc[18,4]


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

unseen_document = 'Man up, quit being a cry baby and get your forms. They are required by law to give them to you. You didn t file your taxes because of sad feelings? Yeah, not a valid defense.'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


processed_text = train['positive_example_1']


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(processed_text)
print(tfidf.shape)
print('\n')
#print(vectorizer.get_feature_names())


y = train['positive_example_1']


X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(tfidf, y, test_size=0.2, random_state=42)


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train_tf,y_train_tf)
# predict the labels on validation dataset
predictions_NB_tf = Naive.predict(X_test_tf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy -> ",accuracy_score(predictions_NB_tf, y_test_tf)*100)
print(classification_report(predictions_NB_tf,y_test_tf))


#Code by Leon Wolber https://www.kaggle.com/leonwolber/reddit-nlp-topic-modeling-prediction

#Save for the next 

logmodel = LogisticRegression()
logmodel.fit(X_train_tf, y_train_tf)

predictions_LR_tf = logmodel.predict(X_test_tf)

print("LR Accuracy -> ",accuracy_score(predictions_LR_tf, y_test_tf)*100)
print(classification_report(predictions_LR_tf,y_test_tf))

