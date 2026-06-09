import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


df=pd.read_csv("/kaggle/input/quora-question-pairs/train.csv.zip")
df.shape


df=df.sample(175000, random_state=42)


df.head()


df['question1']=[data if isinstance(data,str) else "" for data in df["question1"]]
df['question2']=[data if isinstance(data,str) else "" for data in df["question2"]]


df.info()


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


import regex as re

def preprocess(q):
  q=str(q).lower().strip()

  q=q.replace('%', ' percent ')
  q=q.replace('$', ' dollar ')
  q=q.replace('₹', ' rupee ')
  q=q.replace('€', ' euro ')
  q=q.replace('@', ' at ')

  q=q.replace('[math]', '')

  q = q.replace(',000,000,000 ', 'b ')
  q = q.replace(',000,000 ', 'm ')
  q = q.replace(',000 ', 'k ')
  q = re.sub(r'([0-9]+)000000000', r'\1b', q)
  q = re.sub(r'([0-9]+)000000', r'\1m', q)
  q = re.sub(r'([0-9]+)000', r'\1k', q)

  q_concat=[]

  for word in q.split():
      if word in contractions:
        word=contractions[word]

      q_concat.append(word)

  q=' '.join(q_concat)
  q=q.replace("'ve", " have")
  q=q.replace("n't", " not")
  q=q.replace("'re", " are")
  q=q.replace("'ll", " will")

  #remove html tags
  q=BeautifulSoup(q)
  q=q.get_text()

  pattern=re.compile('\W')
  q=re.sub(pattern, ' ', q).strip()

  return q


df['preprocess_question1']=df['question1'].apply(preprocess)
df['preprocess_question2']=df['question2'].apply(preprocess)


df.head()


df.isnull().sum()


df.duplicated().sum()


# Distribution of dataset

print(df["is_duplicate"].value_counts())


# Distribution in percentage

print((df["is_duplicate"].value_counts()/df["is_duplicate"].count())*100)


# Repeated questions

qid=pd.Series(df['qid1'].tolist()+df['qid2'].tolist())
print(f"Number of unique questions: {np.unique(qid).shape[0]}")


# Feature Engineering


df['q1_len']=df['preprocess_question1'].str.len()

df['q2_len']=df['preprocess_question2'].str.len()


df['q1_num_words']=df['preprocess_question1'].apply(lambda row: len(row.split(" ")))
df['q2_num_words']=df['preprocess_question2'].apply(lambda row: len(row.split(" ")))


def common_words(row):
  w1=set(map(lambda word: word.lower().strip(), row['preprocess_question1'].split(" ")))
  w2=set(map(lambda word: word.lower().strip(), row['preprocess_question2'].split(" ")))
  return len(w1&w2)


df['word_common']=df.apply(common_words, axis=1)


def total_words(row):
   w1=set(map(lambda word: word.lower().strip(), row['preprocess_question1'].split(" ")))
   w2=set(map(lambda word: word.lower().strip(), row['preprocess_question2'].split(" ")))
   return len(w1)+len(w2)


df['word_total']=df.apply(total_words, axis=1)


df['word_share']=round(df['word_common']/df['word_total'],2)


import nltk
nltk.download('stopwords')


#Advanced Features

from nltk.corpus import stopwords

def fetch_token_features(row):
  q1=str(row['preprocess_question1'])
  q2=str(row['preprocess_question2'])

  SAFE_DIV=0.0001

  STOP_WORDS=stopwords.words("english")

  token_features=[0.0]*8

  # Converting sentence into tokens
  q1_tokens=q1.split()
  q2_tokens=q2.split()

  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return token_features

  # set of non stopwords
  q1_words=set([word for word in q1_tokens if word not in STOP_WORDS])
  q2_words=set([word for word in q2_tokens if word not in STOP_WORDS])

  # stopwords in questions
  q1_stops=set([word for word in q1_tokens if word in STOP_WORDS])
  q2_stops=set([word for word in q2_tokens if word in STOP_WORDS])

  commmon_word_count=len(q1_words.intersection(q2_words))
  common_stop_count=len(q1_stops.intersection(q2_stops))
  common_token_count=len(set(q1_tokens).intersection(set(q2_tokens)))

  token_features[0]=commmon_word_count/(min(len(q1_words),len(q2_words))+SAFE_DIV)
  token_features[1]=commmon_word_count/(max(len(q1_words),len(q2_words))+SAFE_DIV)
  token_features[2]=common_stop_count/(min(len(q1_stops),len(q2_stops))+SAFE_DIV)
  token_features[3]=common_stop_count/(max(len(q1_stops),len(q2_stops))+SAFE_DIV)
  token_features[4]=common_token_count/(min(len(q1_tokens),len(q2_tokens))+SAFE_DIV)
  token_features[5]=common_token_count/(max(len(q1_tokens),len(q2_tokens))+SAFE_DIV)

  token_features[6]=int(q1_tokens[-1]==q2_tokens[-1])
  token_features[7]=int(q1_tokens[0]==q2_tokens[0])

  return token_features


token_features=df.apply(fetch_token_features, axis=1)

df["cwc_min"]=list(map(lambda x:x[0], token_features))
df["cwc_max"]=list(map(lambda x:x[1], token_features))
df["csc_min"]=list(map(lambda x:x[2], token_features))
df["csc_max"]=list(map(lambda x:x[3], token_features))
df["ctc_min"]=list(map(lambda x:x[4], token_features))
df["ctc_max"]=list(map(lambda x:x[5], token_features))
df["last_word_eq"]=list(map(lambda x:x[6], token_features))
df["first_word_eq"]=list(map(lambda x:x[7], token_features))


!pip install distance
import distance


# length based features
def fetch_length_features(row):
  q1=row['preprocess_question1']
  q2=row['preprocess_question2']

  length_features=[0.0]*3

  # Converting the sentence into Tokens:
  q1_tokens=q1.split()
  q2_tokens=q2.split()

  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return length_features

  # Absolute lenght feature
  length_features[0]=abs(len(q1_tokens)-len(q2_tokens))
  # Average token length
  length_features[1]=(len(q1_tokens)+len(q2_tokens))/2

  strs=list(distance.lcsubstrings(q1,q2))
  if len(strs) > 0:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
  else:
        length_features[2] = 0.0
  return length_features


length_features=df.apply(fetch_length_features, axis=1)

df['abs_len_diff']=list(map(lambda x:x[0], length_features))
df['mean_len']=list(map(lambda x:x[1], length_features))
df['longest_substr_ratio']=list(map(lambda x: x[2], length_features))


df.head()


!pip install fuzzywuzzy


# Fuzzy Features
from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
  q1=row['preprocess_question1']
  q2=row['preprocess_question2']

  fuzzy_features=[0.0]*4

  # fuzz ratio
  fuzzy_features[0]=fuzz.QRatio(q1,q2)
  # fuzz partial ratio
  fuzzy_features[1]=fuzz.partial_ratio(q1,q2)
  # token_sort_ratio
  fuzzy_features[2]=fuzz.token_sort_ratio(q1,q2)
  # token_set_ratio
  fuzzy_features[3]=fuzz.token_set_ratio(q1,q2)

  return fuzzy_features


fuzzy_features=df.apply(fetch_fuzzy_features, axis=1)

df['fuzz_ratio']=list(map(lambda x: x[0], fuzzy_features))
df['fuzz_partial_ratio']=list(map(lambda x: x[1], fuzzy_features))
df['token_sort_ratio']=list(map(lambda x: x[0], fuzzy_features))
df['token_set_ratio']=list(map(lambda x:x[3], fuzzy_features))


df.head()


ques_df=df[['preprocess_question1','preprocess_question2']]
ques_df.head()


final_df=df.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2', 'preprocess_question1', 'preprocess_question2'])
print(final_df.shape)
final_df.head()


final_df['is_duplicate']


from sklearn.feature_extraction.text import CountVectorizer

#text merging
questions=list(ques_df['preprocess_question1'])+list(ques_df['preprocess_question2'])

cv=CountVectorizer(max_features=650, ngram_range=(1,3))
q1_arr,q2_arr=np.vsplit(cv.fit_transform(questions).toarray(),2)


temp_df1=pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2=pd.DataFrame(q2_arr, index=ques_df.index)
temp_df=pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape


temp_df.head()


final_df=pd.concat([final_df, temp_df], axis=1)
print(final_df.shape)
final_df.head()


final_df['is_duplicate']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(final_df.iloc[:,1:].values, final_df.iloc[:,0].values, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
print(f'accuracy score: {accuracy_score(y_test,y_pred)}')
print(f'precision: {precision_score(y_test, y_pred)}')
print(f'recall: {recall_score(y_test, y_pred)}')
print(f'f1 score: {f1_score(y_test, y_pred)}')


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

model=XGBClassifier(verbosity=3)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(f'accuracy score: {accuracy_score(y_test,y_pred)}')
print(f'precision: {precision_score(y_test, y_pred)}')
print(f'recall: {recall_score(y_test, y_pred)}')
print(f'f1 score: {f1_score(y_test, y_pred)}')


# import joblib


# joblib.dump(model, 'duplicate_question_xgboost_new.pkl')
# joblib.dump(rf, 'duplicate_question_random_forest_new.pkl')


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm=confusion_matrix(y_test, y_pred)

# Visualise
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Duplicate", "Duplicate"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()


# Now for Test data we have make features

def test_common_words(q1 , q2):
    w1=set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2=set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1&w2)


def test_total_words(q1,q2):
    w1=set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2=set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1)+len(w2)


def test_token_features(q1,q2):
    SAFE_DIV=0.0001
    STOPWORDS=stopwords.words("english")

    token_features=[0.0]*8

    # Tokenizing
    q1_tokens=q1.split()
    q2_tokens=q2.split()

    if len(q1_tokens)==0 or len(q2_tokens)==0:
        return token_features

    # Non Stopwords
    q1_words=set([word for word in q1_tokens if word not in STOPWORDS])
    q2_words=set([word for word in q2_tokens if word not in STOPWORDS])

    # Fetching stop words
    q1_stops=set([word for word in q1_tokens if word in STOPWORDS])
    q2_stops=set([word for word in q2_tokens if word in STOPWORDS])

    # Common Word Count
    common_word=len(q1_words.intersection(q2_words))
    # Common StopWords count
    common_stop=len(q1_stops.intersection(q2_stops))
    # Common token count
    common_token=len(set(q1_tokens).intersection(set(q2_tokens)))

    # Token Features
    token_features[0]=common_word/(min(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[1]=common_word/(max(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[2]=common_stop/(min(len(q1_stops),len(q2_stops))+SAFE_DIV)
    token_features[3]=common_stop/(max(len(q1_stops),len(q2_stops))+SAFE_DIV)
    token_features[4]=common_token/(min(len(q1_tokens),len(q2_tokens))+SAFE_DIV)
    token_features[5]=common_token/(max(len(q1_tokens),len(q2_tokens))+SAFE_DIV)
    
    token_features[6]=int(q1_tokens[-1]==q2_tokens[-1])
    token_features[7]=int(q1_tokens[0]==q2_tokens[0])

    return token_features


def test_length_features(q1,q2):

    length_features=[0.0]*3

    # Tokenizing
    q1_tokens=q1.split()
    q2_tokens=q2.split()

    if len(q1_tokens)==0 or len(q2_tokens)==0:
        return token_features

    # Absolute length features
    length_features[0]=abs(len(q1_tokens)-len(q2_tokens))
    # Average token length of both
    length_features[1]=(len(q1_tokens)+len(q2_tokens))//2
    # Longest Substring Ratio
    strs=list(distance.lcsubstrings(q1,q2))
    if len(strs) > 0:
            length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
            length_features[2] = 0.0

    return length_features


def test_fuzzy_features(q1,q2):
    fuzzy_features=[0.0]*4

    # Fuzz Ratio
    fuzzy_features[0]=fuzz.QRatio(q1,q2)
    # Fuzz Partial Ratio
    fuzzy_features[1]=fuzz.partial_ratio(q1,q2)
    # token_sort_ratio
    fuzzy_features[2]=fuzz.token_sort_ratio(q1,q2)
    # token_set_ratio
    fuzzy_features[3]=fuzz.token_set_ratio(q1,q2)

    return fuzzy_features


def query_creator(q1,q2):

    test_features=[]

    q1=preprocess(q1)
    q2=preprocess(q2)

    # basic features
    test_features.append(len(q1))
    test_features.append(len(q2))
    test_features.append(len(q1.split(" ")))
    test_features.append(len(q2.split(" ")))

    test_features.append(test_common_words(q1,q2))
    test_features.append(test_total_words(q1,q2))
    test_features.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))

    token_features=test_token_features(q1,q2)
    test_features.extend(token_features)

    length_features=test_length_features(q1,q2)
    test_features.extend(length_features)

    fuzzy_features=test_fuzzy_features(q1,q2)
    test_features.extend(fuzzy_features)

    q1_vectorizer=cv.transform([q1]).toarray()
    q2_vectorizer=cv.transform([q2]).toarray()
    
    return np.hstack((np.array(test_features).reshape(1,-1),q1_vectorizer, q2_vectorizer))


q1="Who is the captain of Indian Cricket Team?"
q2="Who is the current Captain of the cricket team of India?"
q3="Which city is the Capital of India?"
q4="Where is the capital of India?"
q5="Which city is the Business Capital of India?"
q6="Which city is the financial Capital of India?"


model.predict(query_creator(q3,q4))


rf.predict(query_creator(q3,q4))


q1 = 'Where is the capital of India?'
q2 = 'What is the current capital of Pakistan?'
q3 = 'Which city serves as the capital of India?'
q4 = 'What is the business capital of India?'


model.predict(query_creator(q1,q4))


rf.predict(query_creator(q1,q4))


test_df = pd.read_csv("/kaggle/input/quora-question-pairs/test.csv.zip")
test_data = test_df.sample(2345796, random_state=42).reset_index(drop=True)


test_df.shape


test_data.head()


test_data.shape


def predict_row(row):
    try:
        features=query_creator(str(row['question1']), str(row['question2']))
        return rf.predict_proba(features)[0][1]
    except:
        return 0


!pip install tqdm


from tqdm import tqdm
tqdm.pandas()

test_data['is_duplicate'] = test_data.progress_apply(predict_row, axis=1)

print("DONE!")

submission = test_data[['test_id', 'is_duplicate']]
submission.to_csv('submission.csv', index=False)

print("DONE!")




