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


import re
import pandas as pd
from collections import Counter
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords

# ØªØ­Ù…ÙŠÙ„ stopwords

# ØªØ¬Ù‡ÙŠØ² Ø§Ù„Ø£Ø¹Ù…Ø¯Ø©
text_cols = ['body', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2', 'rule', 'subreddit']

# 1ï¸�âƒ£ ØªÙ†Ø¸ÙŠÙ� Ø§Ù„Ø£Ø¹Ù…Ø¯Ø©

# 2ï¸�âƒ£ Ø¥Ù†Ø´Ø§Ø¡ Ø£Ø¹Ù…Ø¯Ø© positive Ùˆ negative
data['positive'] = data['positive_example_1'] + " " + data['positive_example_2']
data['negative'] = data['negative_example_1'] + " " + data['negative_example_2']

# 3ï¸�âƒ£ Ø¥Ø²Ø§Ù„Ø© stopwords

# 4ï¸�âƒ£ Ø­Ø³Ø§Ø¨ Ø§Ù„Ø¥Ø­ØµØ§Ø¦ÙŠØ§Øª Ø¨Ø¹Ø¯ Ø§Ù„ØªÙ†Ø¸ÙŠÙ�
cols_to_count = ['body', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']

for col in cols_to_count:
    data[f'{col}_word_count'] = data[col].apply(lambda x: len(str(x).split()))
    data[f'{col}_char_count'] = data[col].apply(len)
    data[f'{col}_sentence_count'] = data[col].apply(lambda x: len([s for s in re.split(r'[.!?]', str(x)) if s.strip()]))
    data[f'{col}_punctuation_count'] = data[col].apply(lambda x: len(re.findall(r'[?.!ØŸ,;:()<>]', str(x))))

# 5ï¸�âƒ£ Ø§Ù„Ù…ØªÙˆØ³Ø·Ø§Øª ÙˆØ§Ù„Ù�Ø±ÙˆÙ‚Ø§Øª
data['average_postive_count'] = (data['positive_example_1_word_count'] + data['positive_example_2_word_count']) / 2
data['average_neg_count'] = (data['negative_example_1_word_count'] + data['negative_example_2_word_count']) / 2

data['average_postive_char'] = (data['positive_example_1_char_count'] + data['positive_example_2_char_count']) / 2
data['average_neg_char'] = (data['negative_example_1_char_count'] + data['negative_example_2_char_count']) / 2

data['average_postive_sentence_count'] = (data['positive_example_1_sentence_count'] + data['positive_example_2_sentence_count']) / 2
data['average_neg_sentence_count'] = (data['negative_example_1_sentence_count'] + data['negative_example_2_sentence_count']) / 2

data['average_postive_punctuation_count'] = (data['positive_example_1_punctuation_count'] + data['positive_example_2_punctuation_count']) / 2
data['average_neg_punctuation_count'] = (data['negative_example_1_punctuation_count'] + data['negative_example_2_punctuation_count']) / 2

data['length_diff_word_count'] = data['average_postive_count'] - data['average_neg_count']
data['length_diff_char_count'] = data['average_postive_char'] - data['average_neg_char']
data['length_diff_sentence_count'] = data['average_postive_sentence_count'] - data['average_neg_sentence_count']
data['length_diff_punctuation_count'] = data['average_postive_punctuation_count'] - data['average_neg_punctuation_count']

# 6ï¸�âƒ£ Ù‡Ù„ ÙŠØ­ØªÙˆÙŠ Ø§Ù„Ù†Øµ Ø¹Ù„Ù‰ Ø±Ø§Ø¨Ø·
data['has_link'] = data['body'].apply(lambda x: 1 if 'http' in str(x) else 0)
data['has_link_positive_1'] = data['positive_example_1'].apply(lambda x: 1 if 'http' in str(x) else 0)
data['has_link_positive_2'] = data['positive_example_2'].apply(lambda x: 1 if 'http' in str(x) else 0)
data['has_link_negative_example_1'] = data['negative_example_1'].apply(lambda x: 1 if 'http' in str(x) else 0)
data['has_link_negative_example_2'] = data['negative_example_2'].apply(lambda x: 1 if 'http' in str(x) else 0)

data['average_positive_link'] = (data['has_link_positive_1'] + data['has_link_positive_2']) / 2
data['average_negative_link'] = (data['has_link_negative_example_1'] + data['has_link_negative_example_2']) / 2


stop_words = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
}

# âœ… ØªØ¹Ø±ÙŠÙ� Ø¯Ø§Ù„Ø© Ø¥Ø²Ø§Ù„Ø© Ø§Ù„ÙƒÙ„Ù…Ø§Øª Ø§Ù„Ø´Ø§Ø¦Ø¹Ø©
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words])

# âœ… Ø¯Ø§Ù„Ø© ØªÙ†Ø¸ÙŠÙ� Ø§Ù„Ù†ØµÙˆØµ (Ø¥Ø²Ø§Ù„Ø© Ø§Ù„Ø±Ù…ÙˆØ² ÙˆØ§Ù„ÙƒÙ„Ù…Ø§Øª Ø§Ù„Ù‚ØµÙŠØ±Ø© ÙˆØªØ­ÙˆÙŠÙ„ Ù„Ù„Ø­Ø±ÙˆÙ� Ø§Ù„ØµØºÙŠØ±Ø©)
def clean_text(text):
    text = " ".join([w for w in str(text).split() if len(w) >= 3])
    return re.sub(r'[^\w\s]', '', text).lower()


# âœ… Ø¯Ù„ÙˆÙ‚ØªÙŠ ØªÙ‚Ø¯Ø± ØªØ³ØªØ®Ø¯Ù… Ø§Ù„Ø¯Ø§Ù„Ø© Ù…Ù† ØºÙŠØ± Error
for col in ['body', 'positive', 'negative', 'rule', 'subreddit']:
    data[col] = data[col].apply(remove_stopwords)

# 7ï¸�âƒ£ Ø¹Ù…Ù„ stemming
stemmer = PorterStemmer()
for col in ['body', 'positive', 'negative']:
    data[col] = data[col].apply(lambda sentence: " ".join([stemmer.stem(word) for word in str(sentence).split()]))

# 8ï¸�âƒ£ Ø¯Ù…Ø¬ Ø§Ù„Ù†ØµÙˆØµ Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠØ©
data['body_total'] = data['body'] + ' [SEP] ' + data['positive'] + ' [SEP] ' + data['negative']
data['body_total'] = data['body_total'] + ' [SEP] ' + data['subreddit'] + ' [SEP] ' + data['rule']

# 9ï¸�âƒ£ Ø­Ø³Ø§Ø¨ Ø¥Ø­ØµØ§Ø¦ÙŠØ§Øª Bayes Ù„Ù„ÙƒÙ„Ù…Ø§Øª
all_positive_words = " ".join(data['positive']).split()
all_negative_words = " ".join(data['negative']).split()

pos_counts = Counter(all_positive_words)
neg_counts = Counter(all_negative_words)

total_pos_words = sum(pos_counts.values())
total_neg_words = sum(neg_counts.values())
all_words = set(pos_counts.keys()).union(set(neg_counts.keys()))

P_positive = total_pos_words / (total_pos_words + total_neg_words)
P_negative = total_neg_words / (total_pos_words + total_neg_words)
alpha = 1
V = len(all_words)

word_probs = []
for word in all_words:
    p_word_given_pos = (pos_counts.get(word, 0) + alpha) / (total_pos_words + alpha * V)
    p_word_given_neg = (neg_counts.get(word, 0) + alpha) / (total_neg_words + alpha * V)
    p_word = (p_word_given_pos * P_positive) + (p_word_given_neg * P_negative)
    p_pos_given_word = (p_word_given_pos * P_positive) / p_word
    p_neg_given_word = (p_word_given_neg * P_negative) / p_word
    word_probs.append({
        "word": word,
        "P(word|positive)": round(p_word_given_pos, 6),
        "P(word|negative)": round(p_word_given_neg, 6),
        "P(positive|word)": round(p_pos_given_word, 6),
        "P(negative|word)": round(p_neg_given_word, 6)
    })

df_probs = pd.DataFrame(word_probs)

#  ğŸ”Ÿ Ø­Ø³Ø§Ø¨ Ø§Ù„Ù€ positive_score Ùˆ negative_score
pos_dict = dict(zip(df_probs['word'], df_probs['P(positive|word)']))
neg_dict = dict(zip(df_probs['word'], df_probs['P(negative|word)']))

def get_pos_score(text):
    return sum([pos_dict.get(word, 0.5) for word in str(text).split()])

def get_neg_score(text):
    return sum([neg_dict.get(word, 0.5) for word in str(text).split()])

data['positive_score'] = data['body_total'].apply(get_pos_score)
data['negative_score'] = data['body_total'].apply(get_neg_score)

# â“« Ø­Ø³Ø§Ø¨ Ù†Ø³Ø¨ Ø§Ù„ÙƒÙ„Ù…Ø§Øª Ø§Ù„Ø¥ÙŠØ¬Ø§Ø¨ÙŠØ© ÙˆØ§Ù„Ø³Ù„Ø¨ÙŠØ©
positive_words = set(df_probs[df_probs["P(positive|word)"] > 0.5]["word"])
negative_words = set(df_probs[df_probs["P(negative|word)"] > 0.5]["word"])

data['body_length'] = data['body'].apply(lambda x: len(str(x).split()))
data['positive_word_count'] = data['body'].apply(lambda x: sum(1 for word in str(x).split() if word in positive_words))
data['negative_word_count'] = data['body'].apply(lambda x: sum(1 for word in str(x).split() if word in negative_words))

data['positive_word_ratio'] = data['positive_word_count'] / data['body_length']
data['negative_word_ratio'] = data['negative_word_count'] / data['body_length']






pos_link_count = ((data['has_link_positive_1'] == 1) | (data['has_link_positive_2'] == 1)).sum()
neg_link_count = ((data['has_link_negative_example_1'] == 1) | (data['has_link_negative_example_2'] == 1)).sum()

total_positive_examples = len(data)
total_negative_examples = len(data)

P_link_given_positive = pos_link_count / total_positive_examples
P_link_given_negative = neg_link_count / total_negative_examples

P_positive = total_positive_examples / (total_positive_examples + total_negative_examples)
P_negative = total_negative_examples / (total_positive_examples + total_negative_examples)

P_link = (pos_link_count + neg_link_count) / (total_positive_examples + total_negative_examples)

P_positive_given_link = (P_link_given_positive * P_positive) / P_link
P_negative_given_link = (P_link_given_negative * P_negative) / P_link

def link_prob_positive(row):
    if row['has_link_positive_1'] == 1 or row['has_link_positive_2'] == 1:
        return P_positive_given_link
    else:
        return 1 - P_positive_given_link

def link_prob_negative(row):
    if row['has_link_negative_example_1'] == 1 or row['has_link_negative_example_2'] == 1:
        return P_negative_given_link
    else:
        return 1 - P_negative_given_link

data['positive_link_probability'] = data.apply(link_prob_positive, axis=1)
data['negative_link_probability'] = data.apply(link_prob_negative, axis=1)



data.head()





df = data[['body_total' , 'positive_score' , 'negative_score' ,'positive_link_probability','negative_link_probability' ,'positive_word_ratio' , 'negative_word_ratio' ,'length_diff_word_count','length_diff_char_count','length_diff_sentence_count','length_diff_punctuation_count','rule_violation']].copy()




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

df['body_total'] = df['body_total'].fillna('').astype(str) 

vectorizer_body = TfidfVectorizer(max_features=50000,  stop_words='english', ngram_range=(1, 2), min_df=3, max_df=0.8)
comment_vec = vectorizer_body.fit_transform(df['body_total'])




from scipy.sparse import csr_matrix

X = hstack([
    comment_vec,
    csr_matrix(df[[  'positive_score' , 'negative_score'  ,'positive_link_probability','negative_link_probability' ,'positive_word_ratio' , 'negative_word_ratio' ,'length_diff_word_count','length_diff_char_count','length_diff_sentence_count','length_diff_punctuation_count']].values)
])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df['rule_violation'], test_size=0.2, random_state=42)






from xgboost import XGBClassifier

best_model = XGBClassifier(
    subsample=0.8,
    n_estimators=300,
    max_depth=10,
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


import re
import pandas as pd
from collections import Counter
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords


# Ø¯Ø§Ù„Ø© Ø¥Ø²Ø§Ù„Ø© Ø§Ù„ÙƒÙ„Ù…Ø§Øª Ø§Ù„Ø´Ø§Ø¦Ø¹Ø©

# ØªØ¬Ù‡ÙŠØ² Ø§Ù„Ø£Ø¹Ù…Ø¯Ø©
text_cols = ['body', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2', 'rule', 'subreddit']

# 1ï¸�âƒ£ ØªÙ†Ø¸ÙŠÙ� Ø§Ù„Ø£Ø¹Ù…Ø¯Ø©
for col in text_cols:
    test[col] = test[col].fillna("").apply(clean_text)

# 2ï¸�âƒ£ Ø¥Ù†Ø´Ø§Ø¡ Ø£Ø¹Ù…Ø¯Ø© positive Ùˆ negative
test['positive'] = test['positive_example_1'] + " " + test['positive_example_2']
test['negative'] = test['negative_example_1'] + " " + test['negative_example_2']

# 3ï¸�âƒ£ Ø¥Ø²Ø§Ù„Ø© stopwords
for col in ['body', 'positive', 'negative', 'rule', 'subreddit']:
    test[col] = test[col].apply(remove_stopwords)

# 4ï¸�âƒ£ Ø­Ø³Ø§Ø¨ Ø§Ù„Ø¥Ø­ØµØ§Ø¦ÙŠØ§Øª Ø¨Ø¹Ø¯ Ø§Ù„ØªÙ†Ø¸ÙŠÙ�
cols_to_count = ['body', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']

for col in cols_to_count:
    test[f'{col}_word_count'] = test[col].apply(lambda x: len(str(x).split()))
    test[f'{col}_char_count'] = test[col].apply(len)
    test[f'{col}_sentence_count'] = test[col].apply(lambda x: len([s for s in re.split(r'[.!?]', str(x)) if s.strip()]))
    test[f'{col}_punctuation_count'] = test[col].apply(lambda x: len(re.findall(r'[?.!ØŸ,;:()<>]', str(x))))

# 5ï¸�âƒ£ Ø§Ù„Ù…ØªÙˆØ³Ø·Ø§Øª ÙˆØ§Ù„Ù�Ø±ÙˆÙ‚Ø§Øª
test['average_postive_count'] = (test['positive_example_1_word_count'] + test['positive_example_2_word_count']) / 2
test['average_neg_count'] = (test['negative_example_1_word_count'] + test['negative_example_2_word_count']) / 2

test['average_postive_char'] = (test['positive_example_1_char_count'] + test['positive_example_2_char_count']) / 2
test['average_neg_char'] = (test['negative_example_1_char_count'] + test['negative_example_2_char_count']) / 2

test['average_postive_sentence_count'] = (test['positive_example_1_sentence_count'] + test['positive_example_2_sentence_count']) / 2
test['average_neg_sentence_count'] = (test['negative_example_1_sentence_count'] + test['negative_example_2_sentence_count']) / 2

test['average_postive_punctuation_count'] = (test['positive_example_1_punctuation_count'] + test['positive_example_2_punctuation_count']) / 2
test['average_neg_punctuation_count'] = (test['negative_example_1_punctuation_count'] + test['negative_example_2_punctuation_count']) / 2

test['length_diff_word_count'] = test['average_postive_count'] - test['average_neg_count']
test['length_diff_char_count'] = test['average_postive_char'] - test['average_neg_char']
test['length_diff_sentence_count'] = test['average_postive_sentence_count'] - test['average_neg_sentence_count']
test['length_diff_punctuation_count'] = test['average_postive_punctuation_count'] - test['average_neg_punctuation_count']

# 6ï¸�âƒ£ Ù‡Ù„ ÙŠØ­ØªÙˆÙŠ Ø§Ù„Ù†Øµ Ø¹Ù„Ù‰ Ø±Ø§Ø¨Ø·
test['has_link'] = test['body'].apply(lambda x: 1 if 'http' in str(x) else 0)

test['has_link_positive_1'] = test['positive_example_1'].apply(lambda x: 1 if 'http' in str(x) else 0)
test['has_link_positive_2'] = test['positive_example_2'].apply(lambda x: 1 if 'http' in str(x) else 0)
test['has_link_negative_example_1'] = test['negative_example_1'].apply(lambda x: 1 if 'http' in str(x) else 0)
test['has_link_negative_example_2'] = test['negative_example_2'].apply(lambda x: 1 if 'http' in str(x) else 0)

test['average_positive_link'] = (test['has_link_positive_1'] + test['has_link_positive_2']) / 2
test['average_negative_link'] = (test['has_link_negative_example_1'] + test['has_link_negative_example_2']) / 2

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words])

# Ø¯Ø§Ù„Ø© ØªÙ†Ø¸ÙŠÙ� Ø§Ù„Ù†ØµÙˆØµ
def clean_text(text):
    text = " ".join([w for w in str(text).split() if len(w) >= 3])
    return re.sub(r'[^\w\s]', '', text).lower()

# 7ï¸�âƒ£ Ø¹Ù…Ù„ stemming
stemmer = PorterStemmer()
for col in ['body', 'positive', 'negative']:
    test[col] = test[col].apply(lambda sentence: " ".join([stemmer.stem(word) for word in str(sentence).split()]))

# 8ï¸�âƒ£ Ø¯Ù…Ø¬ Ø§Ù„Ù†ØµÙˆØµ Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠØ©
test['body_total'] = test['body'] + ' [SEP] ' + test['positive'] + ' [SEP] ' + test['negative']
test['body_total'] = test['body_total'] + ' [SEP] ' + test['subreddit'] + ' [SEP] ' + test['rule']

# 9ï¸�âƒ£ Ø­Ø³Ø§Ø¨ Ø¥Ø­ØµØ§Ø¦ÙŠØ§Øª Bayes Ù„Ù„ÙƒÙ„Ù…Ø§Øª
all_positive_words = " ".join(test['positive']).split()
all_negative_words = " ".join(test['negative']).split()

pos_counts = Counter(all_positive_words)
neg_counts = Counter(all_negative_words)

total_pos_words = sum(pos_counts.values())
total_neg_words = sum(neg_counts.values())
all_words = set(pos_counts.keys()).union(set(neg_counts.keys()))

P_positive = total_pos_words / (total_pos_words + total_neg_words)
P_negative = total_neg_words / (total_pos_words + total_neg_words)
alpha = 1
V = len(all_words)

word_probs = []
for word in all_words:
    p_word_given_pos = (pos_counts.get(word, 0) + alpha) / (total_pos_words + alpha * V)
    p_word_given_neg = (neg_counts.get(word, 0) + alpha) / (total_neg_words + alpha * V)
    p_word = (p_word_given_pos * P_positive) + (p_word_given_neg * P_negative)
    p_pos_given_word = (p_word_given_pos * P_positive) / p_word
    p_neg_given_word = (p_word_given_neg * P_negative) / p_word
    word_probs.append({
        "word": word,
        "P(word|positive)": round(p_word_given_pos, 6),
        "P(word|negative)": round(p_word_given_neg, 6),
        "P(positive|word)": round(p_pos_given_word, 6),
        "P(negative|word)": round(p_neg_given_word, 6)
    })

df_probs = pd.DataFrame(word_probs)

#  ğŸ”Ÿ Ø­Ø³Ø§Ø¨ Ø§Ù„Ù€ positive_score Ùˆ negative_score
pos_dict = dict(zip(df_probs['word'], df_probs['P(positive|word)']))
neg_dict = dict(zip(df_probs['word'], df_probs['P(negative|word)']))

def get_pos_score(text):
    return sum([pos_dict.get(word, 0.5) for word in str(text).split()])

def get_neg_score(text):
    return sum([neg_dict.get(word, 0.5) for word in str(text).split()])

test['positive_score'] = test['body_total'].apply(get_pos_score)
test['negative_score'] = test['body_total'].apply(get_neg_score)

# â“« Ø­Ø³Ø§Ø¨ Ù†Ø³Ø¨ Ø§Ù„ÙƒÙ„Ù…Ø§Øª Ø§Ù„Ø¥ÙŠØ¬Ø§Ø¨ÙŠØ© ÙˆØ§Ù„Ø³Ù„Ø¨ÙŠØ©
positive_words = set(df_probs[df_probs["P(positive|word)"] > 0.5]["word"])
negative_words = set(df_probs[df_probs["P(negative|word)"] > 0.5]["word"])

test['body_length'] = test['body'].apply(lambda x: len(str(x).split()))
test['positive_word_count'] = test['body'].apply(lambda x: sum(1 for word in str(x).split() if word in positive_words))
test['negative_word_count'] = test['body'].apply(lambda x: sum(1 for word in str(x).split() if word in negative_words))

test['positive_word_ratio'] = test['positive_word_count'] / test['body_length']
test['negative_word_ratio'] = test['negative_word_count'] / test['body_length']



test['positive_link_probability'] = test.apply(link_prob_positive, axis=1)
test['negative_link_probability'] = test.apply(link_prob_negative, axis=1)






test_data  = test[['body_total' , 'positive_score' , 'negative_score' ,'body_word_count', 'body_char_count','body_sentence_count','body_punctuation_count','has_link' ,'positive_link_probability','negative_link_probability' ,'positive_word_ratio' , 'negative_word_ratio' ,'length_diff_word_count','length_diff_char_count','length_diff_sentence_count','length_diff_punctuation_count']].copy()



comment_vec_test = vectorizer_body.transform(test['body_total'].fillna(''))



from scipy.sparse import hstack
import numpy as np

X_test = hstack([
    comment_vec_test,

    csr_matrix(test_data[['positive_score' , 'negative_score' ,'positive_link_probability','negative_link_probability','positive_word_ratio' , 'negative_word_ratio' ,'length_diff_word_count','length_diff_char_count','length_diff_sentence_count','length_diff_punctuation_count']].values)
])



y_proba = best_model.predict_proba(X_test)[:, 1]



y_proba 



preds_df = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': y_proba
})



preds_df.to_csv("submission.csv", index=False)





