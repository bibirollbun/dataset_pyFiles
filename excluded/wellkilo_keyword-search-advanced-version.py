import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer


df_train = pd.read_csv('/kaggle/input/home-depot-product-search-relevance/train.csv.zip', encoding = "ISO-8859-1")
df_test = pd.read_csv('/kaggle/input/home-depot-product-search-relevance/test.csv.zip', encoding = "ISO-8859-1")


# Detailed product introduction is useful, because we need more corpus information to support our search
df_desc = pd.read_csv('/kaggle/input/home-depot-product-search-relevance/product_descriptions.csv.zip')


df_train.head()


df_desc.head()


# The training set and the test set are merged first to facilitate unified preprocessing
df_all = pd.concat((df_train, df_test), axis = 0, ignore_index = True)
df_all.head()


df_all.shape


# The product introduction information also needs to be merged
df_all = pd.merge(df_all, df_desc, how = 'left', on = 'product_uid')
df_all.head()


# Next, text preprocessing is carried out
stemmer = SnowballStemmer('english')

# Part-of-speech normalization process
def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

# To calculate the validity of keywords, see how many times the words appear
def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


# Unify the word forms of all text data
df_all['search_term'] = df_all['search_term'].map(lambda x : str_stemmer(x))

df_all['product_title'] = df_all['product_title'].map(lambda x : str_stemmer(x))

df_all['product_description'] = df_all['product_description'].map(lambda x : str_stemmer(x))


!pip install python-Levenshtein
import Levenshtein

Levenshtein.ratio('hello', 'hello world')


df_all['dist_in_title'] = df_all.apply(lambda x : Levenshtein.ratio(x['search_term'], x['product_title']), axis = 1)

df_all['dist_in_desc'] = df_all.apply(lambda x : Levenshtein.ratio(x['search_term'], x['product_description']), axis = 1)


df_all['all_texts'] = df_all['product_title'] + ' . ' + df_all['product_description'] + ' . '


df_all['all_texts'][:5]


# Establish a corpus
from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(list(tokenize(x, errors = 'ignore')) for x in df_all['all_texts'].values)
print(dictionary)


class MyCorpus(object):
    def __iter__(self):
        for x in df_all['all_texts'].values:
            yield dictionary.doc2bow(list(tokenize(x, errors = 'ignore')))

corpus = MyCorpus()


from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus)

tfidf[dictionary.doc2bow(list(tokenize('hello world, good morning', errors = 'ignore')))]


from gensim.similarities import MatrixSimilarity

def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(list(tokenize(text, errors = 'ignore')))]
    return res

def cos_sim(text1, text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1], num_features = len(dictionary))
    sim = index[tfidf2]
    return float(sim[0])


text1 = 'hello world'
text2 = 'hello from the other side'
cos_sim(text1, text2)


df_all['tfidf_cos_sim_in_title'] = df_all.apply(lambda x : cos_sim(x['search_term'], x['product_title']), axis = 1)


df_all['tfidf_cos_sim_in_title'][:5]


df_all['tfidf_cos_sim_in_desc'] = df_all.apply(lambda x : cos_sim(x['search_term'], x['product_description']), axis = 1)


import nltk

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer.tokenize(df_all['all_texts'].values[0])


sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]
sentences = [y for x in sentences for y in x]
len(sentences)


from nltk.tokenize import word_tokenize
w2v_corpus = [word_tokenize(x) for x in sentences]


from gensim.models.word2vec import Word2Vec

model = Word2Vec(w2v_corpus, vector_size = 128, window = 5, min_count = 5, workers = 4) 


model.wv['right']


vocab = model.wv.key_to_index  

def get_vector(text):
    res = np.zeros(128)
    count = 0
    for word in word_tokenize(text):
        if word in model.wv:  
            res += model.wv[word]  
            count += 1
    return res / count if count > 0 else res  


from scipy import spatial

def w2v_cos_sim(text1, text2):
    try:
        w2v1 = get_vector(text1)
        w2v2 = get_vector(text2)
        sim = 1 - spatial.distance.cosine(w2v1, w2v2)
        return float(sim)
    except:
        return float(0)


w2v_cos_sim('hello world', 'hello from the other size')


df_all['w2v_cos_sim_in_title'] = df_all.apply(lambda x : w2v_cos_sim(x['search_term'], x['product_title']), axis = 1)

df_all['w2v_cos_sim_in_desc'] = df_all.apply(lambda x : w2v_cos_sim(x['search_term'], x['product_description']), axis = 1)


df_all.head(5)


# Next, the columns that cannot be processed by the machine learning model will be droped
df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'all_texts'], axis = 1)


# reshape the train/data set
df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]


# Record the test set id
test_ids = df_test['id']


y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis = 1).values
X_test = df_test.drop(['id', 'relevance'], axis = 1).values


# Establish the Ridge model and debug the alpha value

from sklearn.impute import SimpleImputer

if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
if not isinstance(y_train, pd.Series):
    y_train = pd.Series(y_train)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)  

from sklearn.model_selection import cross_val_score

params = [1, 3, 5, 6, 7, 8, 9, 10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators = 30, max_depth = param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(params, test_scores)
plt.title("Param vs CV Error")


# Upload the result
import os

os.makedirs('/kaggle/working/', exist_ok=True)

rf = RandomForestRegressor(n_estimators=30, max_depth=8)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

output_path = '/kaggle/working/submission.csv'
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv(output_path, index=False)




