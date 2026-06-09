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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.corpus import treebank

nltk.download('stopwords')
nltk.download('wordnet')

import warnings
warnings.filterwarnings('ignore')
!pip install bertopic

#!pip install -U "tmtoolkit[recommended,lda,sklearn,gensim]"


soln_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Sample Solution.csv')
test_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
train_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')


train_df
#test_df


train_df.info()


train_df.shape


train_df.columns


train_df['Category'].unique()


train_df.isnull().sum()


train_df.isna().sum()


len(train_df['Text'][10])


train_df.describe()


sns.countplot(train_df, x="Category", order=train_df['Category'].value_counts().index)
# Add a title
plt.title('Category Histogram')

# Display the chart
plt.show()


proportions = list(train_df['Category'].value_counts(normalize=True))
print(proportions)


plt.pie(proportions,labels = train_df['Category'].unique(),
        autopct='%1.1f%%',  # Display percentages on slices
        shadow=True,
        startangle=140)

# Ensure the pie chart is circular
plt.axis('equal')

# Add a title
plt.title('Category Distribution')

# Display the chart
plt.show()


word_count_lst = []
for review in train_df['Text']:
    word_count_lst.append(len(str(review).split()))

train_df['word_count'] = word_count_lst
sns.histplot(train_df, x="word_count", binwidth=500)
train_df = train_df.drop(columns=['word_count'])

# Add a title
plt.title('Word Count Distribution')

# Display the chart
plt.show()


for index, row in train_df.iterrows():
    tokens = word_tokenize(row['Text'].lower())
    fdist = FreqDist(tokens)

# Plotting the frequency distribution
fdist.plot(20, cumulative=False)

# Add a title
plt.title('Frequency Distribution of 20 Common Words')

# Display the chart
plt.show()


WORD_TOKENIZER = nltk.tokenize.word_tokenize

def tokenize(text):  
    return text.lower().split()


STEMMER = nltk.PorterStemmer()

def stem(tokens):
    """Stem the tokens. I.e., remove morphological affixes and
    normalize to standardized stem forms.

    Has the side effective of producing "unnatural" forms due to
    stemming standards. E.g. quickly becomes quickli
    """
    return [ STEMMER.stem(token) for token in tokens ]


LEMMATIZER = nltk.WordNetLemmatizer()

def lemmatize(tokens):
    """Lemmatize the tokens.

    Retains more natural word forms than stemming, but assumes all
    tokens are nouns unless tokens are passed as (word, pos) tuples.
    """
    lemmas = []
    for token in tokens:
        if isinstance(token, str):
            lemmas.append(LEMMATIZER.lemmatize(token)) # treats token like a noun
        else: # assume a tuple of (word, pos)
            lemmas.append(LEMMATIZER.lemmatize(*token))
    return lemmas


def remove_stopwords(tokens, stopwords=None):
    """Remove stopwords, i.e. words that we don't want as part of our
    analysis. Defaults to the default set of nltk english stopwords.
    """
    if stopwords is None:
        stopwords = nltk.corpus.stopwords.words("english")
    return [ token for token in tokens if token not in stopwords ]


def remove_punctuation(tokens,
                       strip_mentions=False,
                       strip_hashtags=False,
                       strict=False):
    """Remove punctuation from a list of tokens.

    Has some specialized options for dealing with Tweets:

    strip_mentions=True will strip the @ off of @ mentions
    strip_hashtags=True will strip the # from hashtags

    strict=True will remove all punctuation from all tokens, not merely
    just tokens that are punctuation per se.
    """
    tokens = [t for t in tokens if t not in string.punctuation]
    if strip_mentions:
        tokens = [t.lstrip('@') for t in tokens]
    if strip_hashtags:
        tokens = [t.lstrip('#') for t in tokens]
    if strict:
        cleaned = []
        for t in tokens:
            cleaned.append(
                t.translate(str.maketrans('', '', string.punctuation)).strip())
        tokens = [t for t in cleaned if t]
    return tokens


def remove_length1n2(tokens):
    
    filter_tokens = [word for word in tokens if len(word) > 2]
    
    return filter_tokens


import nltk
from nltk.corpus import stopwords

word_counts = {}
en_stopwords = set(stopwords.words('english'))

train_df['tokens'] = train_df['Text'].apply(lambda text: tokenize(str(text)))
#test_df['tokens'] = test_df['Text'].apply(lambda text: tokenize(str(text)))

for index, row in train_df.iterrows():
#for index, row in test_df.iterrows():
    tk = row['tokens']
    tokens = remove_stopwords(tk, stopwords=en_stopwords)
    tokens = remove_length1n2(tokens)
    tokens = remove_punctuation(tokens, strip_mentions=True, strip_hashtags=True)
    tokens = lemmatize(tokens)
    for word in tokens:
        if word not in word_counts:
            word_counts[word] = 0
        else:
            word_counts[word] += 1


len(word_counts)


sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
sorted_words = [word for word, count in sorted_counts]


sorted_words[:10]


train_df['Category'].unique()


category_mapping = {'business': 0, 'tech': 1, 'politics': 2, 'sport': 3, 'entertainment': 4}

# Apply the mapping to the 'Category' column
train_df['Category_ID'] = train_df['Category'].map(category_mapping)


nmf_df = train_df[['ArticleId', 'Category_ID']]


nmf_df.head()


category_dummies = pd.get_dummies(train_df['Category'], prefix='Category', dtype=int)
train_df_encoded = pd.concat([train_df.drop('Category', axis=1), category_dummies], axis=1)


train_df_encoded.columns


nmf_mat = train_df_encoded[['ArticleId', 'Category_business', 'Category_entertainment',
                           'Category_politics', 'Category_sport','Category_tech']]


nmf_mat.head()


from sklearn.decomposition import NMF


model = NMF(n_components = 5, init='nndsvd', random_state=0) # solver ='mu'
W = model.fit_transform(nmf_mat)
H = model.components_


pred = np.dot(W, H)
print(pred)


from sklearn.metrics import mean_squared_error
from math import sqrt

mask = ~np.isnan(nmf_mat)
actual =  np.array(nmf_mat) 

masked_actual = actual[mask]
masked_predicted = pred[mask]
    
squared_errors = (masked_actual - masked_predicted) ** 2
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)
print(rmse) # 0.16215


correct_predictions = np.sum(masked_actual == masked_predicted)

# Calculate the accuracy as the ratio of correct predictions to the total number of predictions
accuracy = correct_predictions / 1490

print(f"Number of correct predictions: {correct_predictions}")
print(f"Total predictions: 1490")
print(f"Train Data Accuracy: {accuracy:.4f}")


test_df.head()


category_dummies = pd.get_dummies(soln_df['Category'], prefix='Category', dtype=int)
test_df_encoded = pd.concat([category_dummies], axis=1)


#test_df_encoded.head
test_df_encoded.set_index(soln_df['ArticleId']) #735 rows × 5 columns


# test_df_encoded
model_tst = NMF(n_components = 5, init='nndsvd', random_state=0) # solver ='mu'
W = model_tst.fit_transform(test_df_encoded)
H = model_tst.components_


pred = np.dot(W, H)
#print(pred)


from sklearn.metrics import mean_squared_error
from math import sqrt

#mask = ~np.isnan(nmf_mat)
actual =  np.array(test_df_encoded) 

#masked_actual = actual[mask]
#masked_predicted = pred[mask]
    
squared_errors = (actual - pred) ** 2
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)
print(rmse) # 4.854608073107214e-16


true_col_name = test_df_encoded.idxmax(axis=1)
pr_cat = [s[9:] for s in true_col_name]
#pr_cat


correct_predictions = np.sum(soln_df['Category'] == pr_cat)

accuracy = correct_predictions / 735

print(f"Number of correct predictions: {correct_predictions}")
print(f"Total predictions: 735")
print(f"Test Data Accuracy: {accuracy:.4f}")


train_df


tr_df = train_df[['ArticleId', 'tokens']]


tr_df


train_df.head()


train_df['docs'] = [" ".join(tok) for tok in train_df['tokens']]


test_df['docs'] = [" ".join(tok) for tok in test_df['tokens']]


train_df.head()


test_df.head() # (735, 3)


n_df  = pd.merge(soln_df, test_df, on='ArticleId', how='inner')


from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

n_samples = 1490
n_features = 1000
n_components = 5
n_top_words = 20
batch_size = 500
init = "nndsvd"
data_samples = train_df['docs'][:n_samples]

print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)

tfidf = tfidf_vectorizer.fit_transform(data_samples)


tfidf.shape


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()



# Fit the NMF model
print(
    "Fitting the NMF model (kullback-leibler) with tf-idf features, "
    "n_samples=%d and n_features=%d..." % (n_samples, n_features)
)
nmf = NMF(
    n_components=n_components,
    random_state=42,
    init=init,
    beta_loss="kullback-leibler",
    alpha_W=0.1,  
    alpha_H= 0.1,         
    l1_ratio= 0.0,
    max_iter = 500,
    solver = 'mu').fit(tfidf)

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
plot_top_words(
    nmf, tfidf_feature_names, n_top_words, "Topics in NMF model (kullback-leibler)"
)


# Transform the data to get the document-topic matrix
doc_topic_matrix = nmf.fit_transform(tfidf)

df = pd.DataFrame(doc_topic_matrix.argmax(axis=1), columns=['nmf_topic'])
topic_names = {0: 'sport', 1: 'politics', 2: 'business',3: 'tech',4: 'entertainment'}

#train_df['nmf_topic_name'] = df['nmf_topic'].map(topic_names)
train_df['nmf_topic_name'] = df['nmf_topic'].map(topic_names)
#print(train_df[['Text', 'Category', 'nmf_topic_name']])


#valid_indices = train_df['nmf_topic_name'].index[train_df['nmf_topic_name'].notnull()]
#train_df_cleaned = train_df.loc[valid_indices]


category_mapping = { "sport": 0, "politics": 1,
                    "business": 2,  "tech": 3,  "entertainment": 4 }
# 1: 'sport', 2: 'politics', 3: 'tech',4: 'entertainment',5: 'business'
# Apply the mapping to the 'Category' column
#train_df_cleaned['nmf_topic_name_id'] = train_df_cleaned['nmf_topic_name'].map(category_mapping)
train_df['nmf_topic_name_id'] = train_df['nmf_topic_name'].map(category_mapping)


# Apply the mapping to the 'Category' column
#train_df_cleaned['Category_ID'] = train_df_cleaned['Category'].map(category_mapping)
train_df['Category_ID'] = train_df['Category'].map(category_mapping)
#soln_df['Category_ID'] = soln_df['Category'].map(category_mapping)


from sklearn.metrics import classification_report

#y_true = train_df_cleaned['Category_ID']
#y_pred = train_df_cleaned['nmf_topic_name_id']

y_true = train_df['Category_ID']
y_pred = train_df['nmf_topic_name_id']

print(classification_report(y_true, y_pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix as a NumPy array:\n", cm)


test_df.head()


test_df['docs'] = [" ".join(tok) for tok in test_df['tokens']]


from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

n_samples = 735
n_features = 1000
n_components = 5
n_top_words = 20
batch_size = 250
init = "nndsvd"
data_samples = test_df['docs'][:n_samples]

print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.8, min_df= 3, max_features=n_features, stop_words="english"
)

tfidf = tfidf_vectorizer.fit_transform(data_samples)


# Fit the NMF model
print(
    "\n" * 2,
    "Fitting the NMF model (generalized Kullback-Leibler "
    "divergence) with tf-idf features, n_samples=%d and n_features=%d..."
    % (n_samples, n_features),
)

nmf = NMF(
    n_components=n_components,
    random_state=42,
    init=init,
    beta_loss="kullback-leibler",
    solver="mu",
    max_iter=500,
    alpha_W=0.1,
    alpha_H=0.1,
    l1_ratio=0,
).fit(tfidf)


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
plot_top_words(
    nmf,
    tfidf_feature_names,
    n_top_words,
    "Topics in NMF model (generalized Kullback-Leibler divergence)",
)


# Transform the data to get the document-topic matrix
doc_topic_matrix = nmf.fit_transform(tfidf)
df = pd.DataFrame(doc_topic_matrix.argmax(axis=1), columns=['nmf_topic'])

labels = doc_topic_matrix.argmax(axis=1)
score = silhouette_score(doc_topic_matrix, labels)

print(labels,score )

# Map the topic indices to meaningful labels for easier interpretation
topic_names = {
    0: 'sport',
    1: 'politics', 
    2: 'business',
    3: 'tech',
    4: 'entertainment'
}
test_df['nmf_topic_name'] = df['nmf_topic'].map(topic_names)


category_mapping = {'sport': 0, 'politics': 1, 'business': 2, 'tech': 3, 'entertainment': 4}

# Apply the mapping to the 'Category' column
test_df['nmf_topic_name_id'] = test_df['nmf_topic_name'].map(category_mapping)
soln_df['Category_ID'] = soln_df['Category'].map(category_mapping)


data = { 'tr':  np.array(soln_df['Category']),
          'pr' : np.array(test_df['nmf_topic_name']) }
df = pd.DataFrame(data)
#df.head()



df['tr'].value_counts()


df['pr'].value_counts()


from sklearn.metrics import classification_report

y_true = soln_df['Category_ID']
y_pred = test_df['nmf_topic_name_id']

print(classification_report(y_true, y_pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix as a NumPy array:\n", cm)


train_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')
tenp_df = train_df.sample(frac=0.1, random_state=42)
twntp_df = train_df.sample(frac=0.2, random_state=42)
fifp_df = train_df.sample(frac=0.5, random_state=42)


#tenp_df.shape # (149, 3)
#twntp_df.shape #(298, 3)
fifp_df.shape # (745, 3)


category_mapping = {'business': 0, 'tech': 1, 'politics': 2, 'sport': 3, 'entertainment': 4}

# Apply the mapping to the 'Category' column
tenp_df['Category_ID'] = tenp_df['Category'].map(category_mapping)
twntp_df['Category_ID'] = twntp_df['Category'].map(category_mapping)
fifp_df['Category_ID'] = fifp_df['Category'].map(category_mapping)


tenp_df.head()


tenp_nmf_df = tenp_df[['ArticleId', 'Category_ID']]
twntp_nmf_df = twntp_df[['ArticleId', 'Category_ID']]
fifp_nmf_df = fifp_df[['ArticleId', 'Category_ID']]


category_dummies_tn = pd.get_dummies(tenp_df['Category'], prefix='Category', dtype=int)
tenp_df_encoded = pd.concat([tenp_df.drop('Category', axis=1), category_dummies_tn], axis=1)

category_dummies_tw = pd.get_dummies(twntp_df['Category'], prefix='Category', dtype=int)
twntp_df_encoded = pd.concat([twntp_df.drop('Category', axis=1), category_dummies_tw], axis=1)

category_dummies_fp = pd.get_dummies(fifp_df['Category'], prefix='Category', dtype=int)
fifp_df_encoded = pd.concat([fifp_df.drop('Category', axis=1), category_dummies_fp], axis=1)


tenp_nmf_mat = tenp_df_encoded[['ArticleId', 'Category_business', 'Category_entertainment',
                           'Category_politics', 'Category_sport','Category_tech']]

twntp_nmf_mat = twntp_df_encoded[['ArticleId', 'Category_business', 'Category_entertainment',
                           'Category_politics', 'Category_sport','Category_tech']]

fifp_nmf_mat = fifp_df_encoded[['ArticleId', 'Category_business', 'Category_entertainment',
                           'Category_politics', 'Category_sport','Category_tech']]


tenp_nmf_mat.head()


model_tn = NMF(n_components = 5, init='nndsvd', random_state=0, max_iter = 300) 
W_tn = model_tn.fit_transform(tenp_nmf_mat)
H_tn = model_tn.components_

model_tw = NMF(n_components = 5, init='nndsvd', random_state=0, max_iter = 300) 
W_tw = model_tw.fit_transform(twntp_nmf_mat)
H_tw = model_tw.components_

model_fp = NMF(n_components = 5, init='nndsvd', random_state=0, max_iter = 300) 
W_fp = model_fp.fit_transform(fifp_nmf_mat)
H_fp = model_fp.components_


pred_tn = np.dot(W_tn, H_tn)
pred_tw = np.dot(W_tw, H_tw)
pred_fp = np.dot(W_fp, H_fp)
#print(pred)


from sklearn.metrics import mean_squared_error
from math import sqrt

actual_tn =  np.array(tenp_nmf_mat) 
actual_tw =  np.array(twntp_nmf_mat) 
actual_fp =  np.array(fifp_nmf_mat) 

squared_errors_tn = (actual_tn - pred_tn) ** 2
squared_errors_tw = (actual_tw - pred_tw) ** 2
squared_errors_fp = (actual_fp - pred_fp) ** 2

mse_tn = np.mean(squared_errors_tn)
mse_tw = np.mean(squared_errors_tw)
mse_fp = np.mean(squared_errors_fp)

rmse_tn = np.sqrt(mse_tn)
rmse_tw = np.sqrt(mse_tw)
rmse_fp = np.sqrt(mse_fp)

print("RMSE for 10%, 20% and 50% of Train Data are :")
print(rmse_tn, rmse_tw, rmse_fp) # 0.16495546995237578


correct_predictions_tn = np.sum(actual_tn == pred_tn)
correct_predictions_tw = np.sum(actual_tw == pred_tw)
correct_predictions_fp = np.sum(actual_fp == pred_fp)

# Calculate the accuracy as the ratio of correct predictions to the total number of predictions
accuracy_tn = correct_predictions_tn / 149
accuracy_tw = correct_predictions_tw / 298
accuracy_fp = correct_predictions_fp / 745

#print(f"Number of correct predictions: {correct_predictions}")
#print(f"Total predictions: 149")
print("Accuracy for 10%, 20% and 50% of Train Data are:") 
print(accuracy_tn, accuracy_tw, accuracy_fp ) 


test_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
soln_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Sample Solution.csv')


test_df['Category'] = soln_df['Category'] #


test_df = test_df.drop('Text', axis = 1)


tp_df = test_df.sample(frac=0.1, random_state=42)
tw_df = test_df.sample(frac=0.2, random_state=42)
fp_df = test_df.sample(frac=0.5, random_state=42)


#tp_df.shape #(74, 2)
#tw_df.shape # (147, 2)
#fp_df.shape # (368, 2)
tp_df.head()


category_dummies_tp = pd.get_dummies(tp_df['Category'], prefix='Category', dtype=int)
tp_df_encoded = pd.concat([category_dummies_tp], axis=1)

category_dummies_tw = pd.get_dummies(tw_df['Category'], prefix='Category', dtype=int)
tw_df_encoded = pd.concat([category_dummies_tw], axis=1)

category_dummies_fp = pd.get_dummies(fp_df['Category'], prefix='Category', dtype=int)
fp_df_encoded = pd.concat([category_dummies_fp], axis=1)


tp_df_encoded.set_index(tp_df['ArticleId'])
tw_df_encoded.set_index(tw_df['ArticleId'])
fp_df_encoded.set_index(fp_df['ArticleId'])


model_tn = NMF(n_components = 5, init='nndsvd', random_state=0, max_iter = 300) 
W_tn = model_tn.fit_transform(tp_df_encoded)
H_tn = model_tn.components_

model_tw = NMF(n_components = 5, init='nndsvd', random_state=0, max_iter = 300) 
W_tw = model_tw.fit_transform(tw_df_encoded)
H_tw = model_tw.components_

model_fp = NMF(n_components = 5, init='nndsvd', random_state=0, max_iter = 300) 
W_fp = model_fp.fit_transform(fp_df_encoded)
H_fp = model_fp.components_


pred_tn = np.dot(W_tn, H_tn)
pred_tw = np.dot(W_tw, H_tw)
pred_fp = np.dot(W_fp, H_fp)


actual_tn =  np.array(tp_df_encoded) 
actual_tw =  np.array(tw_df_encoded) 
actual_fp =  np.array(fp_df_encoded) 

squared_errors_tn = (actual_tn - pred_tn) ** 2
squared_errors_tw = (actual_tw - pred_tw) ** 2
squared_errors_fp = (actual_fp - pred_fp) ** 2

mse_tn = np.mean(squared_errors_tn)
mse_tw = np.mean(squared_errors_tw)
mse_fp = np.mean(squared_errors_fp)

rmse_tn = np.sqrt(mse_tn)
rmse_tw = np.sqrt(mse_tw)
rmse_fp = np.sqrt(mse_fp)

print("RMSE for 10%, 20% and 50% of Test Data are :")
print(rmse_tn, rmse_tw, rmse_fp)


true_col_name_tp = tp_df_encoded.idxmax(axis=1)
pr_cat_tp = [s[9:] for s in true_col_name_tp]

true_col_name_tw = tw_df_encoded.idxmax(axis=1)
pr_cat_tw = [s[9:] for s in true_col_name_tw]

true_col_name_fp = fp_df_encoded.idxmax(axis=1)
pr_cat_fp = [s[9:] for s in true_col_name_fp]


correct_predictions_tn = np.sum(tp_df['Category'] == pr_cat_tp)
correct_predictions_tw = np.sum(tw_df['Category'] == pr_cat_tw)
correct_predictions_fp = np.sum(fp_df['Category'] == pr_cat_fp)

print(correct_predictions_tn, correct_predictions_tw,correct_predictions_fp )

# Calculate the accuracy as the ratio of correct predictions to the total number of predictions
accuracy_tn = correct_predictions_tn / 74
accuracy_tw = correct_predictions_tw / 147
accuracy_fp = correct_predictions_fp / 368

#print(f"Number of correct predictions: {correct_predictions}")
#print(f"Total predictions: 149")
print("Accuracy for 10%, 20% and 50% of Test Data are:") 
print(accuracy_tn, accuracy_tw, accuracy_fp ) 


tenp_df = train_df.sample(frac=0.1, random_state=42)
twntp_df = train_df.sample(frac=0.2, random_state=42)
fifp_df = train_df.sample(frac=0.5, random_state=42)


test_df.head()


n_df  = pd.merge(soln_df, test_df, on='ArticleId', how='inner')


n_df.shape $(735, 5)


tp_df = n_df.sample(frac=0.1, random_state=42)
tw_df = n_df.sample(frac=0.2, random_state=42)
fp_df = n_df.sample(frac=0.5, random_state=42)


#tenp_df.shape # (149, 3)
#tenp_df.head()
#twntp_df.shape # (298, 5)
#fifp_df.shape  # (745, 5)
#tp_df.shape  #(74, 2)
#tp_df['docs'].isna().sum()
#tw_df.shape # (147, 8)
fp_df.shape # (368, 8)



from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

n_samples =  368  #147   #74    #745 - 50%   # 10% - 149 ; 20% - 298
n_features = 1000
n_components = 5
n_top_words = 30
batch_size =  100   #100 #50
init =  "nndsvd" #"custom"  #"random" #"nndsvd"          #"nndsvda"      #"nndsvdar"
data_samples = fp_df['docs'][:n_samples]

print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)

tfidf = tfidf_vectorizer.fit_transform(data_samples)


# Fit the NMF model
print(
    "Fitting the NMF model (kullback-leibler) with tf-idf features, "
    "n_samples=%d and n_features=%d..." % (n_samples, n_features)
)
nmf = NMF(
    n_components=n_components,
    random_state=45,
    init=init,
    beta_loss="kullback-leibler",
    alpha_W=0.1,  
    alpha_H= 0.1,         
    l1_ratio= 0,
    #max_iter = 200,
    solver = 'mu'
).fit(tfidf)

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
plot_top_words(
    nmf, tfidf_feature_names, n_top_words, "Topics in NMF model (kullback-leibler)"
)


# Transform the data to get the document-topic matrix
doc_topic_matrix = nmf.fit_transform(tfidf)

df = pd.DataFrame(doc_topic_matrix.argmax(axis=1), columns=['nmf_topic'])
#topic_names = {0: 'business', 1: 'entertainment', 2: 'politics', 3: 'sport', 4: 'tech'} # 10%
#topic_names = {0: 'business', 1: 'politics', 2: 'sport', 3: 'entertainment', 4: 'tech'} # 20%
#topic_names = {0: 'sport', 1: 'politics', 2: 'entertainment', 3: 'business', 4: 'tech'} # 50%
#topic_names = {0: 'politics', 1: 'entertainment', 2: 'sports', 3: 'tech', 4: 'business'}

#topic_names = {0: 'tech', 1: 'politics', 2: 'business', 3: 'sport', 4: 'entertainment'} # 10%
#topic_names = {0: 'politics', 1: 'sport', 2: 'business', 3: 'entertainment', 4: 'tech'} # 20%
topic_names = {0: 'tech', 1: 'politics', 2: 'sport', 3: 'business', 4: 'entertainment'}
df['nmf_topic_name_id'] = df['nmf_topic'].map(topic_names)

#print(train_df[['Text', 'Category', 'nmf_topic_name']])


d ={ 'tech': 0, 'politics' : 1, 'sport' : 2, 'business' : 3, 'entertainment' : 4 }
#tp_df['Category_ID'] = tp_df['Category'].map(d)
fp_df['Category_ID'] = fp_df['Category'].map(d)

#d ={ 'business': 0, 'entertainment' : 1, 'politics' : 2, 'sport' : 3,
    #'tech' : 4 } #10%

#tenp_df['Category_ID'] = tenp_df['Category'].map(d)

#d = { 'business': 0,  'politics' : 1,  'sport' : 2, 
      #'entertainment' : 3,  'tech' : 4} # 20%

#twntp_df['Category_ID'] = twntp_df['Category'].map(d)

#d = { 'sport': 0,  'politics' : 1,  'entertainment' : 2, 
      #'business' : 3,  'tech' : 4} # 20%

#fifp_df['Category_ID'] = fifp_df['Category'].map(d)


#tp_df['Category_ID'].isna().sum()
#df['nmf_topic_name']
df.shape


from sklearn.metrics import classification_report

y_true = np.array(fp_df['Category_ID'])
y_pred = np.array(df['nmf_topic'])

print(classification_report(y_true, y_pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix as a NumPy array:\n", cm)


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#pipeline = Pipeline([
   # ('tfidf', TfidfVectorizer()),
    #('nmf', NMF(init='nndsvd', random_state=42))])

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=0.3, max_features=1000, stop_words="english"
)

tfidf = tfidf_vectorizer.fit_transform(test_df['docs'])

nmf = NMF(n_components = 5, init='nndsvd', solver = 'mu', random_state=42)
param_grid = {
    'beta_loss' : ['kullback-leibler', 'frobenius'],
    'alpha_W': [0.1, 0.5, 1.0],
    'alpha_H' : [0.1, 0.5, 1.0],
    'l1_ratio': [0.0, 0.5, 1.0]}

gridsearch = GridSearchCV(nmf, param_grid, cv=3, scoring='accuracy', verbose=2)
gridsearch.fit(tfidf)

print("Best parameters:")
print(gridsearch.best_params_)


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

vectorizer = CountVectorizer(max_df=0.8, min_df=3, stop_words='english')
X = vectorizer.fit_transform(test_df['docs'])

lda = LatentDirichletAllocation(n_components=5,max_iter=10, 
                                learning_method='online', random_state=42)
lda.fit(X)


feature_names = vectorizer.get_feature_names_out()

for tp_idx, tp in enumerate(lda.components_):
    tp_words =  [feature_names[i] for i in tp.argsort()[-11:-1]]
    print("Topic Words", ' '.join(tp_words))
    
    


topic_dist = lda.transform(X)
topics = topic_dist.argmax(axis=1)
print(topics)


d ={ 'tech': 0, 'business' : 1, 'sport' : 2, 'entertainment' : 3, 'politics' : 4 }
#tp_df['Category_ID'] = tp_df['Category'].map(d)
n_df['Category_ID'] = n_df['Category'].map(d)


from sklearn.metrics import classification_report

y_true = np.array(n_df['Category_ID'])
y_pred = np.array(topics)

print(classification_report(y_true, y_pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix as a NumPy array:\n", cm)


n_samples = 735
n_features = 1000
n_components = 5
n_top_words = 30
batch_size = 250
init = "nndsvd"
data_samples = test_df['docs'][:n_samples]

print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.8, min_df= 3, max_features=n_features, stop_words="english"
)

tfidf = tfidf_vectorizer.fit_transform(data_samples)

mbnmf = MiniBatchNMF(
    n_components=n_components,
    random_state=42,
    batch_size=batch_size,
    init=init,
    beta_loss="kullback-leibler",
    alpha_W=0.00005,
    alpha_H=0.00005,
    l1_ratio=0.5,
).fit(tfidf)
#print("done in %0.3fs." % (time() - t0))


tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
plot_top_words(
    mbnmf,
    tfidf_feature_names,
    n_top_words,
    "Topics in MiniBatchNMF model (kullback-leibler)",
)


# Transform the data to get the document-topic matrix
doc_topic_matrix = mbnmf.fit_transform(tfidf)

df = pd.DataFrame(doc_topic_matrix.argmax(axis=1), columns=['nmf_topic'])

topic_names = {0: 'sport', 1: 'politics', 2: 'business', 3: 'tech', 4: 'entertainment'}
df['nmf_topic_name_id'] = df['nmf_topic'].map(topic_names)



#d ={ 'sport': 0, 'politics' : 1, 'business' : 2, 'tech' : 3, 'entertainment' : 4 }
d ={ 'politics': 0, 'business' : 1, 'tech' : 2, 'sport' : 3, 'entertainment' : 4 }
soln_df['Category_ID'] = soln_df['Category'].map(d)



from sklearn.metrics import classification_report

y_true = np.array(soln_df['Category_ID'])
y_pred = np.array(soln_df['cls'])

print(classification_report(y_true, y_pred))


from sklearn.cluster import KMeans

for seed in range(5):
    kmeans = KMeans(
        n_clusters=5,
        max_iter=100,
        n_init=1,
        random_state=45,
    ).fit(tfidf)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    print(f"Number of elements assigned to each cluster: {cluster_sizes}")
print()
print(
    "True number of documents in each category according to the class labels: "
    f"{147}"
)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(tfidf)
kmeans.labels_

#kmeans.predict([[0, 0], [12, 3]])

#kmeans.cluster_centers_



soln_df['cls'] = kmeans.fit_predict(tfidf)


soln_df['cls'].value_counts()


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix as a NumPy array:\n", cm)


from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(train_df['docs'][0])

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title("Tokens Word Cloud")
plt.show()


from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer

vectorizer_model = OnlineCountVectorizer(stop_words="english", ngram_range=(1, 3))
topic_model = BERTopic(vectorizer_model=vectorizer_model)

topics, probs = topic_model.fit_transform(train_df['docs'])


fr = topic_model.get_topic_info()
print("Number of Topics", len(fr))


fr.head(10)


topic_first = fr.iloc[1]["Topic"]
topic_model.get_topic(topic_first)


topic_model.visualize_barchart(top_n_topics = 5)


topic_model.visualize_topics()


topic_model.visualize_hierarchy(top_n_topics = 26)


similar_topics_sports, similarity_sp = topic_model.find_topics("sport", top_n = 150)
similar_topics_politics, similarity_pt = topic_model.find_topics("politics", top_n = 150)
similar_topics_tech, similarity_th = topic_model.find_topics("tech", top_n = 150)
similar_topics_business, similarity_bs = topic_model.find_topics("business", top_n = 150)
similar_topics_entt, similarity_et = topic_model.find_topics("entertainment", top_n = 150)
#similarity


doc_info = topic_model.get_document_info(train_df['docs'])
#doc_info


topic_info =topic_model.get_topic_info()
for topic_id in topic_info['Topic']:
    print(f"Topic {topic_id}: {topic_model.get_topic(topic_id)}")


topic_to_cat ={

    0: "sport", 1: "entertainment", 2: "entertainment", 3: "business", 4:"tech", 5:"business",
    6: "game", 7: "business", 8: "politics", 9: "politics", 10:"tech", 11:"entertainment",
    12: "tech", 13: "business", 14: "tech", 15: "tech", 16:"business", 17:"tech",
    18: "sport", 19: "business", 20: "business", 21: "business", 22:"business", 23:"politics",
    24: "politics"
    
}


doc_info = topic_model.get_document_info(train_df['docs'])
doc_info['Category'] = doc_info['Topic'].map(topic_to_cat)


doc_info.head()


d ={ 'tech': 0, 'business' : 1, 'sport' : 2, 'entertainment' : 3, 'politics' : 4 }
train_df['Category_ID'] = train_df['Category'].map(d)
doc_info['Category_ID'] = doc_info['Category'].map(d)


correct_predictions = np.sum(train_df['Category_ID'] == doc_info['Category_ID'])

# Calculate the accuracy as the ratio of correct predictions to the total number of predictions
accuracy = correct_predictions / 1490

print(f"Number of correct predictions: {correct_predictions}")
print(f"Total predictions: 1490")
print(f"Training Accuracy: {accuracy:.4f}")


doc_info['Category'].value_counts()


#vectorizer_model = OnlineCountVectorizer(stop_words="english",ngram_range=(1, 3))
#topic_model = BERTopic(vectorizer_model=vectorizer_model)

topics, probs = topic_model.fit_transform(test_df['docs'])


fr = topic_model.get_topic_info()
print("Number of Topics", len(fr))


fr.head()


topic_first = fr.iloc[1]["Topic"]
topic_model.get_topic(topic_first)


topic_model.visualize_barchart(top_n_topics = 9)


topic_model.visualize_hierarchy(top_n_topics = 26)


topic_info =topic_model.get_topic_info()
for topic_id in topic_info['Topic']:
    print(f"Topic {topic_id}: {topic_model.get_topic(topic_id)}")


topic_to_cat ={

    0: "politics", 1: "business", 2: "tech", 3: "sport", 4:"sport", 5:"entertainment",
    6: "entertainment", 7: "business"
    
}


doc_info = topic_model.get_document_info(test_df['docs'])
doc_info['Category'] = doc_info['Topic'].map(topic_to_cat)


doc_info['Category'].isna().sum()


doc_info['Category'].value_counts()


test_df.shape


d ={ 'tech': 0, 'business' : 1, 'sport' : 2, 'entertainment' : 3, 'politics' : 4 }
soln_df['Category_ID'] = soln_df['Category'].map(d)
doc_info['Category_ID'] = doc_info['Category'].map(d)


correct_predictions = np.sum(soln_df['Category_ID'] == doc_info['Category_ID'])

# Calculate the accuracy as the ratio of correct predictions to the total number of predictions
accuracy = correct_predictions / 725

print(f"Number of correct predictions: {correct_predictions}")
print(f"Total predictions: 1490")
print(f"Test Accuracy: {accuracy:.4f}")

