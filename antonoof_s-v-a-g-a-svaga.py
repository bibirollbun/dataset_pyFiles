import re
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from nltk.stem import WordNetLemmatizer
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 68


clean_newlines = re.compile(r'\n+')
clean_spaces = re.compile(r'\s+')
clean_punct = re.compile(r'[^a-zA-Z0-9\s]')

def fast_clean(text):
    text = clean_newlines.sub(' ', text)
    text = clean_spaces.sub(' ', text)
    text = clean_punct.sub('', text)
    return text.strip().lower()

lemmatizer = WordNetLemmatizer()
def fast_lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

train['Misconception'] = train['Misconception'].fillna('NA').astype(str)
train['target_cat'] = train['Category'] + ":" + train['Misconception']

train['sentence'] = ("Question: " + train['QuestionText'].astype(str) + 
                     " Answer: " + train['MC_Answer'].astype(str) + 
                     " Explanation: " + train['StudentExplanation'].astype(str))
test['sentence'] = ("Question: " + test['QuestionText'].astype(str) + 
                    " Answer: " + test['MC_Answer'].astype(str) + 
                    " Explanation: " + test['StudentExplanation'].astype(str))

train['sentence'] = train['sentence'].apply(fast_clean).apply(fast_lemmatize)
test['sentence'] = test['sentence'].apply(fast_clean).apply(fast_lemmatize)

def add_text_features(df):
    df['text_len'] = df['sentence'].apply(len)
    df['word_count'] = df['sentence'].apply(lambda x: len(x.split()))
    df['unique_word_count'] = df['sentence'].apply(lambda x: len(set(x.split())))
    df['digit_count'] = df['sentence'].apply(lambda x: sum(c.isdigit() for c in x))
    df['punct_count'] = df['sentence'].apply(lambda x: sum(c in '.,;:!?-' for c in x))
    return df

train = add_text_features(train)
test = add_text_features(test)

le_cat = LabelEncoder()
le_mis = LabelEncoder()

train['target1'] = le_cat.fit_transform(train['Category'])
train['target2'] = le_mis.fit_transform(train['Misconception'])

tfidf_word = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_df=0.95, min_df=3)
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,5), max_df=0.95, min_df=3)

X_word = tfidf_word.fit_transform(pd.concat([train['sentence'], test['sentence']]))
X_char = tfidf_char.fit_transform(pd.concat([train['sentence'], test['sentence']]))


X_all = hstack([
    X_word[:len(train)],
    X_char[:len(train)],
    train[['text_len', 'word_count', 'unique_word_count', 'digit_count', 'punct_count']].values
]).tocsr()

X_test_all = hstack([
    X_word[len(train):],
    X_char[len(train):],
    test[['text_len', 'word_count', 'unique_word_count', 'digit_count', 'punct_count']].values
]).tocsr()


train_pool_cat = Pool(X_all, train['target1'])
model_cat_full = CatBoostClassifier(iterations=5000, learning_rate=0.2, depth=6,
                                    eval_metric='MultiClass', task_type='GPU',
                                    devices='0:1', verbose=100, random_state=42)
model_cat_full.fit(train_pool_cat)


train_pool_mis = Pool(X_all, train['target2'])
model_mis_full = CatBoostClassifier(iterations=6000, learning_rate=0.2, depth=3,
                                    eval_metric='MultiClass', task_type='GPU',
                                    devices='0:1', verbose=100, random_state=41)
model_mis_full.fit(train_pool_mis)


test_pool = Pool(X_test_all)
pred_1 = model_cat_full.predict_proba(test_pool)
pred_2 = model_mis_full.predict_proba(test_pool)

map_inverse1 = {i: c for i, c in enumerate(le_cat.classes_)}
map_inverse2 = {i: c for i, c in enumerate(le_mis.classes_)}

predicted1 = np.argsort(-pred_1, axis=1)[:, :3]
predicted2 = np.argsort(-pred_2, axis=1)[:, :3]

predict = []
for i in range(len(predicted1)):
    preds = []
    for j in range(3):
        cat_label = map_inverse1[predicted1[i, j]]
        mis_label = map_inverse2[predicted2[i, 0]]
        if 'Misconception' in cat_label:
            preds.append(f"{cat_label}:{mis_label}")
        else:
            preds.append(f"{cat_label}:NA")
    predict.append(" ".join(preds))

sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub['Category:Misconception'] = predict
sub.to_csv("submission.csv", index=False)

