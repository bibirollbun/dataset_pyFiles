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
import re 
import numpy as np 


data = pd.read_csv(r'/kaggle/input/jigsaw-agile-community-rules/train.csv')


data.fillna("", inplace=True)

data['body'] = data['body'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
data['positive_example_1'] = data['positive_example_1'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
data['positive_example_2'] = data['positive_example_2'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
data['negative_example_1'] = data['negative_example_1'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
data['negative_example_2'] = data['negative_example_2'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))



data['body'] = data['body'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['positive_example_1'] = data['positive_example_1'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['positive_example_2'] = data['positive_example_2'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['negative_example_1'] = data['negative_example_1'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['negative_example_2'] = data['negative_example_2'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())


data['positive'] = data['positive_example_1'] + " " + data['positive_example_2']
data['negative'] = data['negative_example_1'] + " " + data['negative_example_2']


tokanized_data= data['body'].apply(lambda x:x.split())
tokanized_positive= data['positive'].apply(lambda x:x.split())
tokanized_negative= data['negative'].apply(lambda x:x.split())
tokanized_rule= data['rule'].apply(lambda x:x.split())



from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

data['body'] = tokanized_data.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
data['positive'] = tokanized_positive.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
data['negative'] = tokanized_negative.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
data['rule'] = tokanized_rule.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))



import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


data['positive'] = data['positive'].apply(remove_stopwords)
data['negative'] = data['negative'].apply(remove_stopwords)
data['body'] = data['body'].apply(remove_stopwords)
data['rule'] = data['rule'].apply(remove_stopwords)



data['body'] = data.apply(
    lambda row: row['body'] + " " + row['positive'] if row['rule_violation'] == 0 
    else row['body'] + " " + row['negative'], axis=1 )






from collections import Counter
import pandas as pd

all_positive_words = " ".join(data['positive']).split()
all_negative_words = " ".join(data['negative']).split()

pos_counts = Counter(all_positive_words)
neg_counts = Counter(all_negative_words)

total_pos_words = sum(pos_counts.values())
total_neg_words = sum(neg_counts.values())

all_words = set(pos_counts.keys()).union(set(neg_counts.keys()))



from collections import Counter
import pandas as pd

total_words = total_pos_words + total_neg_words
P_positive = total_pos_words / total_words
P_negative = total_neg_words / total_words

alpha = 1  # Laplace smoothing

# نحسب احتمالات بايز لكل كلمة
word_probs = []
V = len(all_words)  # حجم القاموس

for word in all_words:
    # Likelihoods
    p_word_given_pos = (pos_counts.get(word, 0) + alpha) / (total_pos_words + alpha * V)
    p_word_given_neg = (neg_counts.get(word, 0) + alpha) / (total_neg_words + alpha * V)

    # Evidence (المقام المشترك)
    p_word = (p_word_given_pos * P_positive) + (p_word_given_neg * P_negative)

    # Bayes Rule
    p_pos_given_word = (p_word_given_pos * P_positive) / p_word
    p_neg_given_word = (p_word_given_neg * P_negative) / p_word

    word_probs.append({
        "word": word,
        "P(word|positive)": round(p_word_given_pos, 6),
        "P(word|negative)": round(p_word_given_neg, 6),
        "P(positive|word)": round(p_pos_given_word, 6),
        "P(negative|word)": round(p_neg_given_word, 6)
    })

# نحطهم في DataFrame
df_probs = pd.DataFrame(word_probs)



# نحول جدول الاحتمالات إلى قاموس للكلمة → P(positive|word)
pos_dict = dict(zip(df_probs['word'], df_probs['P(positive|word)']))
neg_dict = dict(zip(df_probs['word'], df_probs['P(negative|word)']))



def get_pos_score(text):
    text = str(text) if pd.notnull(text) else ''
    return sum([pos_dict.get(word, 0.5) for word in text.split()])

def get_neg_score(text):
    text = str(text) if pd.notnull(text) else ''
    return sum([neg_dict.get(word, 0.5) for word in text.split()])

data['positive_score'] = data['body'].apply(get_pos_score)
data['negative_score'] = data['body'].apply(get_neg_score)



#from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()
#data['rule_encoded'] = le.fit_transform(data['rule'])



data['body']  = data['body'] 


data.head(2)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack



df = data[['body' ,'subreddit' , 'rule' , 'positive_score' , 'negative_score' ,'rule_violation']].copy()



df['body'] = df['body'].fillna('').astype(str) 
df['subreddit'] = df['subreddit'].fillna('').astype(str) 

vectorizer_body = TfidfVectorizer(max_features=30000)
comment_vec = vectorizer_body.fit_transform(df['body'])

vectorizer_subreddit = TfidfVectorizer(max_features=30000)
comment_subreddit = vectorizer_subreddit.fit_transform(df['subreddit'])

vectorizer_rule = TfidfVectorizer(max_features=30000)
comment_rule = vectorizer_rule.fit_transform(df['rule'])



from scipy.sparse import csr_matrix

X = hstack([
    comment_vec,
    comment_subreddit ,
    comment_rule ,
    csr_matrix(df[['positive_score', 'negative_score']].values)
])



model = LogisticRegression(max_iter=1000)
model.fit(X, df['rule_violation'])



test = pd.read_csv (r'/kaggle/input/jigsaw-agile-community-rules/test.csv')


data.fillna("", inplace=True)

test['body'] = test['body'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['positive_example_1'] = test['positive_example_1'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['positive_example_2'] = test['positive_example_2'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['negative_example_1'] = test['negative_example_1'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['negative_example_2'] = test['negative_example_2'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))



data.fillna("", inplace=True)

test['body'] = test['body'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['positive_example_1'] = test['positive_example_1'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['positive_example_2'] = test['positive_example_2'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['negative_example_1'] = test['negative_example_1'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['negative_example_2'] = test['negative_example_2'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))



test['body'] = test['body'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['positive_example_1'] = test['positive_example_1'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['positive_example_2'] =test['positive_example_2'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['negative_example_1'] = test['negative_example_1'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['negative_example_2'] = test['negative_example_2'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())


test['positive'] = test['positive_example_1'] + " " + test['positive_example_2']
test['negative'] = test['negative_example_1'] + " " + test['negative_example_2']


tokanized_test= test['body'].apply(lambda x:x.split())
tokanized_positive_test= test['positive'].apply(lambda x:x.split())
tokanized_negative_test= test['negative'].apply(lambda x:x.split())
tokanized_rule_test= test['rule'].apply(lambda x:x.split())



from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

test['body'] = tokanized_test.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
test['positive'] = tokanized_positive_test.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
test['negative'] = tokanized_rule.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))



import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


test['positive'] = test['positive'].apply(remove_stopwords)
test['negative'] = test['negative'].apply(remove_stopwords)
test['body'] = test['body'].apply(remove_stopwords)
test['rule'] = test['rule'].apply(remove_stopwords)



def get_pos_score(text):
    text = str(text) if pd.notnull(text) else ''
    return sum([pos_dict.get(word, 0.5) for word in text.split()])

def get_neg_score(text):
    text = str(text) if pd.notnull(text) else ''
    return sum([neg_dict.get(word, 0.5) for word in text.split()])

test['positive_score'] = test['body'].apply(get_pos_score)
test['negative_score'] = test['body'].apply(get_neg_score)



test['body'] = test['body'] 


test_data = test[['body' , 'subreddit' ,'rule' , 'positive_score' , 'negative_score' ]].copy()



comment_vec_test = vectorizer_body.transform(test['body'].fillna(''))
subreddit_vec_test = vectorizer_subreddit.transform(test['subreddit'].fillna(''))
rule_vec_test = vectorizer_rule.transform(test['rule'].fillna(''))



from scipy.sparse import hstack
import numpy as np

X_test = hstack([
    comment_vec_test,
    subreddit_vec_test,
    rule_vec_test ,
    csr_matrix(test_data[['positive_score' , 'negative_score']].values)
])



y_proba = model.predict_proba(X_test)[:, 1]  # احتمالية يكون rule_violation = 1



y_proba


preds_df = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': y_proba
})



preds_df.to_csv("submission.csv", index=False)





