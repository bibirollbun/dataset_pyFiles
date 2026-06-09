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
data['rule'] = data['rule'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
data['subreddit'] = data['subreddit'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))



data['body'] = data['body'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['positive_example_1'] = data['positive_example_1'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['positive_example_2'] = data['positive_example_2'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['negative_example_1'] = data['negative_example_1'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['negative_example_2'] = data['negative_example_2'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['rule'] = data['rule'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
data['subreddit'] = data['subreddit'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())


data['body_length'] = data['body'].apply(lambda x: len(str(x).split()))
data['has_link'] = data['body'].apply(lambda x: 1 if 'http' in str(x) else 0)



data['positive'] = data['positive_example_1'].fillna('') + " " + data['positive_example_2'].fillna('')
data['negative'] = data['negative_example_1'].fillna('') + " " + data['negative_example_2'].fillna('')



tokanized_data= data['body'].apply(lambda x:x.split())
tokanized_positive= data['positive'].apply(lambda x:x.split())
tokanized_negative= data['negative'].apply(lambda x:x.split())



from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

data['body'] = tokanized_data.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
data['positive'] = tokanized_positive.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
data['negative'] = tokanized_negative.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))



data['body_total'] = data['body'] + ' [SEP]' + data['positive'] + ' [SEP]' +  data['negative']




import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


data['positive'] = data['positive'].apply(remove_stopwords)
data['body_total'] = data['body_total'].apply(remove_stopwords)
data['negative'] = data['negative'].apply(remove_stopwords)
data['body'] = data['body'].apply(remove_stopwords)
data['rule'] = data['rule'].apply(remove_stopwords)
data['subreddit'] = data['subreddit'].apply(remove_stopwords)



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



pos_dict = dict(zip(df_probs['word'], df_probs['P(positive|word)']))
neg_dict = dict(zip(df_probs['word'], df_probs['P(negative|word)']))



def get_pos_score(text):
    text = str(text) if pd.notnull(text) else ''
    return sum([pos_dict.get(word, 0.5) for word in text.split()])

def get_neg_score(text):
    text = str(text) if pd.notnull(text) else ''
    return sum([neg_dict.get(word, 0.5) for word in text.split()])

data['positive_score'] = data['body_total'].apply(get_pos_score)
data['negative_score'] = data['body_total'].apply(get_neg_score)



positive_words = set(df_probs[df_probs["P(positive|word)"] > 0.5]["word"])
def count_positive_words(comment):
    words = comment if isinstance(comment, list) else str(comment).split()
    return sum(1 for word in words if word in positive_words)

data['body_length'] = data['body'].apply(lambda x: len(x) if isinstance(x, list) else len(str(x).split()))

data['positive_word_count'] = data['body'].apply(count_positive_words)

data['positive_word_ratio'] = data.apply(
    lambda row: row['positive_word_count'] / row['body_length'] if row['body_length'] > 0 else 0,
    axis=1
)



negative_words = set(df_probs[df_probs["P(negative|word)"] > 0.5]["word"])
def count_negative_words(comment):
    words = comment if isinstance(comment, list) else str(comment).split()
    return sum(1 for word in words if word in negative_words)
# عدد الكلمات السلبية
data['negative_word_count'] = data['body'].apply(count_negative_words)

# النسبة
data['negative_word_ratio'] = data.apply(
    lambda row: row['negative_word_count'] / row['body_length'] if row['body_length'] > 0 else 0,
    axis=1
)



data['body'] = data['body'].fillna('').astype(str)
#data['body_total'] = data['body_total'].fillna('').astype(str)
data['subreddit'] = data['subreddit'].fillna('').astype(str)
data['rule'] = data['rule'].fillna('').astype(str)

data['body_total'] = data['body_total'] + ' [SEP]' + data['subreddit'] + ' [SEP]' + data['rule']


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack


df = data[['body_total' , 'positive_score' , 'negative_score' , 'has_link' ,'positive_word_ratio' , 'negative_word_ratio' ,'rule_violation']].copy()





df['body_total'] = df['body_total'].fillna('').astype(str) 

vectorizer_body = TfidfVectorizer(max_features=50000,  stop_words='english', ngram_range=(1, 2), min_df=3, max_df=0.8)
comment_vec = vectorizer_body.fit_transform(df['body_total'])




from scipy.sparse import csr_matrix

X = hstack([
    comment_vec,
    csr_matrix(df[['positive_score' , 'negative_score' , 'has_link' ,'positive_word_ratio' , 'negative_word_ratio']].values)
])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df['rule_violation'], test_size=0.2, random_state=42)







from xgboost import XGBClassifier

best_model = XGBClassifier(
    subsample=0.7,
    n_estimators=700,
    max_depth=5,
    learning_rate=0.1,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

best_model.fit(X_train, y_train)



preds = best_model.predict(X_test)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.4f}")



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, preds)
print("Confusion Matrix:\n", cm)



test = pd.read_csv (r'/kaggle/input/jigsaw-agile-community-rules/test.csv')


test.fillna("", inplace=True)

test['body'] = test['body'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['positive_example_1'] = test['positive_example_1'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['positive_example_2'] = test['positive_example_2'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['negative_example_1'] = test['negative_example_1'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['negative_example_2'] = test['negative_example_2'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['rule'] = test['rule'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))
test['subreddit'] = test['subreddit'].apply(lambda x: " ".join([w for w in str(x).split() if len(w) >= 3]))



test['body'] = test['body'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['positive_example_1'] = test['positive_example_1'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['positive_example_2'] =test['positive_example_2'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['negative_example_1'] = test['negative_example_1'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['negative_example_2'] = test['negative_example_2'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['rule'] = test['rule'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
test['subreddit'] = test['subreddit'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())


test['body_length'] = test['body'].apply(lambda x: len(str(x).split()))
test['has_link'] = test['body'].apply(lambda x: 1 if 'http' in str(x) else 0)



test['positive'] = test['positive_example_1'] + " " + test['positive_example_2']
test['negative'] = test['negative_example_1'] + " " + test['negative_example_2']


tokanized_test= test['body'].apply(lambda x:x.split())
tokanized_positive_test= test['positive'].apply(lambda x:x.split())
tokanized_negative_test= test['negative'].apply(lambda x:x.split())



from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

test['body'] = tokanized_test.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
test['positive'] = tokanized_positive_test.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))
test['negative'] = tokanized_negative_test.apply(lambda sentence: " ".join([stemmer.stem(word) for word in sentence]))



import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


test['positive'] = test['positive'].apply(remove_stopwords)
test['negative'] = test['negative'].apply(remove_stopwords)
test['body'] =test['body'].apply(remove_stopwords)
test['rule'] = test['rule'].apply(remove_stopwords)
test['subreddit'] = test['subreddit'].apply(remove_stopwords)



test['positive_score'] = test['body'].apply(get_pos_score)
test['negative_score'] = test['body'].apply(get_neg_score)



positive_words = set(df_probs[df_probs["P(positive|word)"] > 0.5]["word"])
negative_words = set(df_probs[df_probs["P(negative|word)"] > 0.5]["word"])

# دوال لحساب عدد الكلمات
def count_positive_words(comment):
    words = str(comment).split()
    return sum(1 for word in words if word in positive_words)

def count_negative_words(comment):
    words = str(comment).split()
    return sum(1 for word in words if word in negative_words)

# حساب طول الجسم
test['body_length'] = test['body'].apply(lambda x: len(str(x).split()))

# عدد الكلمات الإيجابية والسلبية
test['positive_word_count'] = test['body'].apply(count_positive_words)
test['negative_word_count'] = test['body'].apply(count_negative_words)

# حساب النسب
test['positive_word_ratio'] = test.apply(
    lambda row: row['positive_word_count'] / row['body_length'] if row['body_length'] > 0 else 0,
    axis=1
)

test['negative_word_ratio'] = test.apply(
    lambda row: row['negative_word_count'] / row['body_length'] if row['body_length'] > 0 else 0,
    axis=1
)



test['body'] = test['body'].fillna('').astype(str)
test['subreddit'] = test['subreddit'].fillna('').astype(str)
test['rule'] = test['rule'].fillna('').astype(str)


test['body'] = test['body'] + ' [SEP]' +  test['positive']  + ' [SEP]' + test['negative']

test['body_total'] = test['body'] + ' [SEP]' + test['subreddit'] + ' [SEP]' + test['rule']



test_data = test[['body_total' , 'positive_score' , 'negative_score' , 'has_link' ,'positive_word_ratio' , 'negative_word_ratio']].copy()



comment_vec_test = vectorizer_body.transform(test['body_total'].fillna(''))



from scipy.sparse import hstack
import numpy as np

X_test = hstack([
    comment_vec_test,

    csr_matrix(test_data[['positive_score' , 'negative_score' , 'has_link' ,'positive_word_ratio' , 'negative_word_ratio']].values)
])



y_proba = best_model.predict_proba(X_test)[:, 1]



y_proba 


preds_df = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': y_proba
})



preds_df.to_csv("submission.csv", index=False)





