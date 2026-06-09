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


import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

#Datasets creation
df_solution = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Sample Solution.csv')
df_test = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
df_train = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')



df_solution.head(10)


df_test.head(10)


df_train.head(10)


#Create a df with all the information so we can perform a EDA
df_all = pd.concat([df_test, df_solution['Category']], axis=1)
df_all = pd.concat([df_all, df_train[['ArticleId', 'Text', 'Category']]], axis=0)

df_all.head(10)


#General check, nulls
df_all.info()
df_all['Category'].value_counts()
df_all.isnull().sum()



#Categories chart
sns.countplot(y=df_all['Category'],order=df_all['Category'].value_counts().index, color ="gray")
plt.title('Distribution by category')
plt.show()


df_all["word_count"] = df_all["Text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,5))
sns.barplot(x="word_count", y="Category", data=df_all, estimator=np.mean, ci=None, color="gray")
plt.title("Mean lenght of article by category")
plt.xlabel("Category")
plt.ylabel("Mean lenght")
plt.xticks(rotation=45)
plt.show()


df_all['char_count'] = df_all['Text'].apply(len)
df_all['word_count'] = df_all['Text'].apply(lambda x: len(x.split()))

print(df_all[['char_count','word_count']].describe())


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english', max_features=20)
word_counts = cv.fit_transform(df_all['Text'])
top_words = pd.DataFrame(cv.get_feature_names_out(), columns=["Top Words"])
print("Must frequent words:")
print(top_words.T)



import nltk, matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stop_words.update({"said","mr","also","one","two","new","us","would","could","may","might", "told", "bbc", "bbc news", "bbc world"})

def preprocess_text(text: str):
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stop_words]

def get_trigrams(text: str):
    toks = preprocess_text(text)
    return list(ngrams(toks, 3))

category_trigrams = {cat: [] for cat in df_all['Category'].unique()}
for _, row in df_all.iterrows():
    category_trigrams[row['Category']].extend(get_trigrams(row['Text']))


category_tokens = {cat: [] for cat in df_all['Category'].unique()}
for _, row in df_all.iterrows():
    category_tokens[row['Category']].extend(preprocess_text(row['Text']))

top_n = 20
top_words_per_cat = {}
for cat, toks in category_tokens.items():
    counts = Counter(toks)
    top = [w for w, c in counts.most_common(top_n)]
    top_words_per_cat[cat] = top

df_top_unigrams = pd.DataFrame(top_words_per_cat)
df_top_unigrams.index = [f"Rank {i}" for i in range(1, top_n+1)]
print("Top 20 unigrams per category:")
display(df_top_unigrams)

top_trigrams_per_cat = {}
for cat, trigs in category_trigrams.items():
    trig_counts = Counter(trigs)
    top_trigs = [' '.join(t) for t, c in trig_counts.most_common(top_n)]
    top_trigrams_per_cat[cat] = top_trigs

df_top_trigrams = pd.DataFrame(top_trigrams_per_cat)
df_top_trigrams.index = [f"Rank {i}" for i in range(1, top_n+1)]
print("Top 20 trigrams per category:")
display(df_top_trigrams)


#Required libraries
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from scipy.optimize import linear_sum_assignment

k = df_all["Category"].nunique()

#Since I'm using a new library, I will create a new list of extra stopwords
my_extra_stops = {
    "said","mr","also","one","two","new","us","would","could","may","might",
    "told", "news","bbc", "world", "year", "2004"
}

custom_stop_words = list(ENGLISH_STOP_WORDS.union(my_extra_stops))
    
tfidf = TfidfVectorizer(
    stop_words=custom_stop_words,
    lowercase=True,
    ngram_range=(1,2),
    max_df=0.85,         #ignore too frequent words
    min_df=3,            # ignore weird or less frequent words
    max_features=30000
)

X = tfidf.fit_transform(df_all["Text"])
feature_names = np.array(tfidf.get_feature_names_out())
print(f"TF-IDF shape: {X.shape} | #topics (k): {k}")



#Matrix factorization
nmf = NMF(
    n_components=k,
    init="nndsvda",
    random_state=42,
    max_iter=600,
    beta_loss="frobenius",
    solver="cd"
)
W = nmf.fit_transform(X)
H = nmf.components_ 

# Helper: print top words by category
def print_top_words(H, feature_names, n_top=12, title="NMF Topics"):
    for t, comp in enumerate(H):
        top_idx = np.argsort(comp)[::-1][:n_top]
        words = ", ".join(feature_names[top_idx])
        print(f"[{title}] Topic {t}: {words}")

print_top_words(H, feature_names, n_top=12, title="NMF")

topic_pred = W.argmax(axis=1)
le = LabelEncoder()
y_true = le.fit_transform(df_all["Category"]) 

#Confusion matrix
C = confusion_matrix(y_true, topic_pred, labels=range(k))

#find better labeling
row_ind, col_ind = linear_sum_assignment(C.max() - C)
mapping = {col: row for row, col in zip(row_ind, col_ind)}
y_pred_mapped = np.vectorize(mapping.get)(topic_pred)

acc  = accuracy_score(y_true, y_pred_mapped)
nmi  = NMI(y_true, topic_pred)
ari  = ARI(y_true, topic_pred)

print(f"\n--- Unsupervised NMF evaluation ---")
print(f"Accuracy (with optimal mapping): {acc:.3f}")
print(f"NMI: {nmi:.3f} | ARI: {ari:.3f}")
print("\nClassification report (after mapping):")
print(classification_report(y_true, y_pred_mapped, target_names=le.classes_))


#Libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score



#Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    df_all["Text"], 
    df_all["Category"], 
    test_size=0.2, 
    random_state=42, 
    stratify=df_all["Category"]
)

#TF-IDF + Logistic Regression 
pipe_lr = Pipeline([ 
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000)), 
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")) 
])

pipe_lr.fit(X_train, y_train) 
y_pred_lr = pipe_lr.predict(X_test) 

print("Logistic regression results:") 
print(f"accuracy: {accuracy_score(y_test, y_pred_lr):.3f}") 
print(classification_report(y_test, y_pred_lr)) 

# TF-IDF + Naive Bayes 
pipe_nb = Pipeline([ ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000)), ("clf", MultinomialNB()) ]) 
pipe_nb.fit(X_train, y_train) 
y_pred_nb = pipe_nb.predict(X_test) 

print("Naive Bayes results:") 
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.3f}") 
print(classification_report(y_test, y_pred_nb))


for frac in [.1, .2, .5, .7, .8, .9, .99]:
    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=frac, stratify=y_train, random_state=42)
    pipe_lr.fit(X_sub, y_sub)
    y_pred_sub = pipe_lr.predict(X_test)
    print(f"Train fraction={frac:.0%} | Test accuracy={accuracy_score(y_test, y_pred_sub):.3f}")

