# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

import re
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

#stemming
from nltk.stem.porter import PorterStemmer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


temp_df = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')


temp_df.head()


df = temp_df.sample(50000, random_state = 2)
df = df.reset_index(drop=True)


# # Drop rows with missing values (optional but safe)
# temp_df = temp_df.dropna()

# # Separate classes
# df_0 = temp_df[temp_df['is_duplicate'] == 0].sample(25000, random_state=2)
# df_1 = temp_df[temp_df['is_duplicate'] == 1].sample(25000, random_state=2)

# # Combine and shuffle
# df = pd.concat([df_0, df_1]).sample(frac=1, random_state=2).reset_index(drop=True)
# df.head()


df.shape


def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    # Stemming
    ps = PorterStemmer()
    q = ' '.join([ps.stem(word) for word in q.split()])

    
    return q


df['question1'] = df['question1'].apply(preprocess)
df['question2'] = df['question2'].apply(preprocess)


df.info()


df.isnull().sum()


df.duplicated().sum()


print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
df['is_duplicate'].value_counts().plot(kind = 'bar')


# repeated question

qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print("Number of unique questions", qid.unique().shape[0])
x = qid.value_counts() > 1
print('Number of repeated qustion', x[x].shape[0])


plt.hist(qid.value_counts().values, bins =160)
plt.yscale('log')
plt.show()


# Feature Engineering
#length of questions
df['q1_len'] = df['question1'].str.len()
df['q2_len'] = df['question2'].str.len()


df['q1_num_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_num_words'] = df['question2'].apply(lambda row: len(row.split(" ")))
                                           


# For the common words in both question 1 and question 2
def common_word(q1, q2):
    word1 = set(str(q1).lower().split(" "))
    word2 = set(str(q2).lower().split(" "))
    return len(word1 & word2)

df['common_word_count'] = df.apply(lambda row: common_word(row['question1'], row['question2']), axis=1)
df.head()


# Total words:
def total_word(q1, q2):
    word1 = set(str(q1).lower().split(" "))
    word2 = set(str(q2).lower().split(" "))
    return (len(word1) + len(word2))

df['total_words'] = df.apply(lambda row: total_word(row['question1'], row['question2']), axis=1)
df.head()


df['word_share'] = round(df['common_word_count']/df['total_words'],2)
df.head()


# Advanced Features
from nltk.corpus import stopwords

def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']

    SAFE_DIV = 0.0001 

    STOP_WORDS = stopwords.words("english")

    token_features = [0.0]*8

    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # get the non_stopword form the question
    q1_words = set()
    for word in q1_tokens:
        if word not in STOP_WORDS:
            q1_words.add(word)


    q2_words = set()
    for word in q2_tokens:
        if word not in STOP_WORDS:
            q2_words.add(word)

    #Get the stopwords in Questions
    q1_stops = set()
    for word in q1_tokens:
        if word in STOP_WORDS:
            q1_stops.add(word)

    q2_stops = set()
    for word in q2_tokens:
        if word in STOP_WORDS:
            q2_stops.add(word)

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

      
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features
   


token_features = df.apply(fetch_token_features, axis=1)

df["cwc_min"]       = list(map(lambda x: x[0], token_features))
df["cwc_max"]       = list(map(lambda x: x[1], token_features))
df["csc_min"]       = list(map(lambda x: x[2], token_features))
df["csc_max"]       = list(map(lambda x: x[3], token_features))
df["ctc_min"]       = list(map(lambda x: x[4], token_features))
df["ctc_max"]       = list(map(lambda x: x[5], token_features))
df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
df["first_word_eq"] = list(map(lambda x: x[7], token_features))


df.head()


!pip install distance


import distance

def fetch_length_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0]*3
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    
    return length_features


length_features = df.apply(fetch_length_features, axis=1)

df['abs_len_diff'] = list(map(lambda x: x[0], length_features))
df['mean_len'] = list(map(lambda x: x[1], length_features))
df['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))


# Fuzzy Features
from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    
    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features



fuzzy_features = df.apply(fetch_fuzzy_features, axis=1)

# Creating new feature columns for fuzzy features
df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))



df.head()


#sns.pairplot(df[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']],hue='is_duplicate')


#sns.pairplot(df[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']],hue='is_duplicate')


#sns.pairplot(df[['last_word_eq', 'first_word_eq', 'is_duplicate']],hue='is_duplicate')


#sns.pairplot(df[['mean_len', 'abs_len_diff','longest_substr_ratio', 'is_duplicate']],hue='is_duplicate')



df.drop(columns = ['mean_len', 'abs_len_diff'], inplace = True)
df.columns


#sns.pairplot(df[['fuzz_ratio', 'fuzz_partial_ratio','token_sort_ratio','token_set_ratio', 'is_duplicate']],hue='is_duplicate')


# Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 3 dimention
# from sklearn.preprocessing import MinMaxScaler

# X = MinMaxScaler().fit_transform(df[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
# y = df['is_duplicate'].values




# from sklearn.manifold import TSNE

# tsne2d = TSNE(
#     n_components=2,
#     init='random', # pca
#     random_state=101,
#     method='barnes_hut',
#     n_iter=1000,
#     verbose=2,
#     angle=0.5
# ).fit_transform(X)



# x_df = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

# # draw the plot in appropriate place in the grid
# sns.lmplot(data=x_df, x='x', y='y', hue='label', fit_reg=False, height=8,palette="Set1",markers=['s','o'])


# tsne3d = TSNE(
#     n_components=3,
#     init='random', # pca
#     random_state=101,
#     method='barnes_hut',
#     n_iter=1000,
#     verbose=2,
#     angle=0.5
# ).fit_transform(X)



# import plotly.graph_objs as go
# import plotly.tools as tls
# import plotly.offline as py
# py.init_notebook_mode(connected=True)

# trace1 = go.Scatter3d(
#     x=tsne3d[:,0],
#     y=tsne3d[:,1],
#     z=tsne3d[:,2],
#     mode='markers',
#     marker=dict(
#         sizemode='diameter',
#         color = y,
#         colorscale = 'Portland',
#         colorbar = dict(title = 'duplicate'),
#         line=dict(color='rgb(255, 255, 255)'),
#         opacity=0.75
#     )
# )

# data=[trace1]
# layout=dict(height=800, width=800, title='3d embedding with engineered features')
# fig=dict(data=data, layout=layout)
# py.iplot(fig, filename='3DBubble')





# sns.displot(df['q2_len'])
# print('Minimum Charecter', df['q2_len'].min())
# print('Maximum Charecter', df['q2_len'].max())
# print('Average Charecter',int(df['q2_len'].mean()))


# sns.displot(df['q1_num_words'])
# print('Minimum Charecter', df['q1_num_words'].min())
# print('Maximum Charecter', df['q1_num_words'].max())
# print('Average Charecter',int(df['q1_num_words'].mean()))


# sns.displot(df['q2_num_words'])
# print('Minimum Charecter', df['q2_num_words'].min())
# print('Maximum Charecter', df['q2_num_words'].max())
# print('Average Charecter',int(df['q2_num_words'].mean()))


# Distplot for non-duplicates
# sns.distplot(df[df['is_duplicate'] == 0]['common_word_count'],
#              label='Not Duplicate',
#              kde=True,
#              color='blue')

# # Distplot for duplicates
# sns.distplot(df[df['is_duplicate'] == 1]['common_word_count'],
#              label='Duplicate',
#              kde=True,
#              color='red')

# plt.legend()
# plt.title("Distribution of Common Word Count")
# plt.xlabel("Common Word Count")
# plt.ylabel("Density")
# plt.show()


# Distplot for non-duplicates
# sns.distplot(df[df['is_duplicate'] == 0]['total_words'],
#              label='Not Duplicate',
#              kde=True,
#              color='blue')

# # Distplot for duplicates
# sns.distplot(df[df['is_duplicate'] == 1]['total_words'],
#              label='Duplicate',
#              kde=True,
#              color='red')

# plt.legend()
# plt.title("Distribution of total Word Count")
# plt.xlabel("Total Word Count")
# plt.ylabel("Density")
# plt.show()


# Distplot for non-duplicates
# sns.distplot(df[df['is_duplicate'] == 0]['word_share'],
#              label='Not Duplicate',
#              kde=True,
#              color='blue')

# # Distplot for duplicates
# sns.distplot(df[df['is_duplicate'] == 1]['word_share'],
#              label='Duplicate',
#              kde=True,
#              color='red')

# plt.legend()
# plt.title("Distribution of Word share")
# plt.xlabel("Word share")
# plt.ylabel("Density")
# plt.show()


ques_df = df[['question1', 'question2']]
ques_df.head()


final_df = df.drop(columns = ['id','qid1','qid2','question1','question2'])
print(final_df.shape)
final_df.head()


# Now on ques_df we apply BOW

from sklearn.feature_extraction.text import CountVectorizer

# Combine question1 and question2 as a list of strings
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features = 5000,
                    ngram_range = (1,2))

# # Fit and transform the questions, convert to array, and then split vertically
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)


# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf = TfidfVectorizer(max_features=5000)  # Start small, increase gradually
# q1_arr, q2_arr = np.vsplit(tfidf.fit_transform(questions).toarray(), 2)



# Convert each half to a DataFrame, with original indices
temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)


# Concatenate both DataFrames side by side
temp_df3 = pd.concat([temp_df1, temp_df2], axis=1)

# Check shape
temp_df3.shape
temp_df3.head()


final_df = pd.concat([final_df, temp_df3], axis = 1)
print(final_df.shape)
final_df.head()


X = final_df.drop(columns=['is_duplicate'])
y = final_df['is_duplicate']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
rf = RandomForestClassifier(
    max_depth=None,
    max_features='log2',
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=300,
    random_state=42,  # optional for reproducibility
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


from sklearn.metrics import f1_score, roc_auc_score

f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print("F1 Score:", f1)
print("AUC:", auc)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming y_test and y_pred are already defined
cm = confusion_matrix(y_test, y_pred)

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Duplicate", "Duplicate"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()





def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return len(w1 & w2)


def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return (len(w1) + len(w2))


def test_fetch_token_features(q1,q2):
    
    SAFE_DIV = 0.0001 

    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*8
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features


def test_fetch_length_features(q1,q2):
    
    length_features = [0.0]*1
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # Absolute length features
    # length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    # length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[0] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    
    return length_features



def test_fetch_fuzzy_features(q1,q2):
    
    fuzzy_features = [0.0]*4
    
    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


def query_point_creator(q1,q2):
    
    input_query = []
    
    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    input_query.append(test_common_words(q1,q2))
    input_query.append(test_total_words(q1,q2))
    input_query.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))
    
    # fetch token features
    token_features = test_fetch_token_features(q1,q2)
    input_query.extend(token_features)
    
    # fetch length based features
    length_features = test_fetch_length_features(q1,q2)
    input_query.extend(length_features)
    
    # fetch fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1,q2)
    input_query.extend(fuzzy_features)
    
    # bow feature for q1
    q1_bow = cv.transform([q1]).toarray()
    
    # bow feature for q2
    q2_bow = cv.transform([q2]).toarray()
    
    
    
    return np.hstack((np.array(input_query).reshape(1,20),q1_bow,q2_bow))


q1 = 'Where is the capital of India?'
q2 = 'What is the current capital of Pakistan?'
q3 = 'Which city serves as the capital of India?'
q4 = 'What is the business capital of India?'
q5 = "How do I cook the best pasta?"
q6 = "What is the best way to make pasta?"



rf.predict(query_point_creator(q5,q6))


cv








# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.metrics import accuracy_score

# # Assuming X_train, X_test, y_train, y_test are already ready and vectorized (TF-IDF/BOW)

# model = Sequential()
# model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))  # Binary classification (duplicate or not)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)

# # Evaluate
# y_pred_nn = (model.predict(X_test > 0.5)).astype("int32")
# print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))



# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# from sklearn.metrics import accuracy_score

# # Define model
# model = Sequential()
# model.add(Dense(1024, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))  # Output layer

# # Compile
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Early stopping
# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

# # Train
# history = model.fit(
#     X_train, y_train,
#     epochs=30,
#     batch_size=128,
#     validation_split=0.2,
#     callbacks=[early_stop],
#     verbose=1
# )

# # Predict and Evaluate
# y_pred_nn = (model.predict(X_test) > 0.5).astype("int32")
# print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))



# # If X_train and X_test are numpy arrays, convert them to DataFrames
# # X_train = pd.DataFrame(X_train)
# # X_test = pd.DataFrame(X_test)
# # Check for duplicate column names
# duplicates = X_train.columns[X_train.columns.duplicated()]
# print("Duplicate column names:", duplicates)
# # Rename columns to ensure unique feature names
# X_train.columns = [f"feature_{i}" for i in range(X_train.shape[1])]
# X_test.columns = X_train.columns  # ensure same feature names in test

# from xgboost import XGBClassifier
# xgb = XGBClassifier()
# xgb.fit(X_train,y_train)
# y_pred1 = xgb.predict(X_test)
# accuracy_score(y_test,y_pred1)


# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Confusion matrix for Random Forest
# cm_rf = confusion_matrix(y_test, y_pred)
# disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Not_duplicate', 'Duplicate'])

# # Confusion matrix for XGBoost
# cm_xgb = confusion_matrix(y_test, y_pred1)
# disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Not_duplicate', 'Duplicate'])

# # Create a figure with two subplots
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# # Plot the confusion matrices side by side
# disp_rf.plot(ax=ax[0], cmap=plt.cm.Blues)
# ax[0].set_title('Random Forest Confusion Matrix')

# disp_xgb.plot(ax=ax[1], cmap=plt.cm.Blues)
# ax[1].set_title('XGBoost Confusion Matrix')

# plt.tight_layout()
# plt.show()



# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

# # Define the model
# rf = RandomForestClassifier(random_state=42)

# # Define hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],          # Number of trees
#     'max_depth': [None, 10, 20, 30],          # Max depth of each tree
#     'min_samples_split': [2, 5, 10],          # Min samples to split an internal node
#     'min_samples_leaf': [1, 2, 4],            # Min samples at a leaf node
#     'max_features': ['sqrt', 'log2']          # Number of features to consider at each split
# }

# # Grid search with 5-fold cross-validation
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
#                            cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# # Fit to training data
# grid_search.fit(X_train, y_train)

# # Best parameters and score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Accuracy:", grid_search.best_score_)

# # Predict using the best model
# best_rf = grid_search.best_estimator_
# y_pred_best = best_rf.predict(X_test)

# # Accuracy on test set
# from sklearn.metrics import accuracy_score
# print("Test Accuracy with Best Model:", accuracy_score(y_test, y_pred_best))





