import os


import pandas as pd
import numpy as np


IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


if IS_KAGGLE:
    data_dir = '../input/quora-insincere-questions-classification'
    train_fname = data_dir + '/train.csv'
    test_fname = data_dir + '/test.csv'
    sample_fname = data_dir + '/sample_submission.csv'
else:
    os.environ['KAGGLE_CONFIG_DIR'] = '.'
    !kaggle competitions download -c quora-insincere-questions-classification -f train.csv -p data
    !kaggle competitions download -c quora-insincere-questions-classification -f test.csv -p data
    !kaggle competitions download -c quora-insincere-questions-classification -f sample_submission.csv -p data
    train_fname = 'data/train.csv.zip'
    test_fname = 'data/test.csv.zip'
    sample_fname = 'data/sample_submission.csv.zip' 


raw_df = pd.read_csv(train_fname, low_memory=False)
raw_df


sincere_df = raw_df[raw_df.target == 0]
sincere_df.question_text.values[:10]


insincere_df = raw_df[raw_df.target == 1]
insincere_df.question_text.values[:10]


# normalize = True gives the percentage of counts
raw_df.target.value_counts(normalize=True)


raw_df.target.value_counts(normalize=True).plot(kind='bar')


print(f'Average length of each question is: {np.mean(raw_df.question_text.apply(len))}')


test_df = pd.read_csv(test_fname)
test_df


sub_df = pd.read_csv(sample_fname)
sub_df


if IS_KAGGLE:
    SAMPLE_SIZE = len(raw_df)
else:
    SAMPLE_SIZE = 100_000


sample_df = raw_df.sample(SAMPLE_SIZE, random_state=42)
sample_df


q0 = sincere_df.question_text.values[1]
q0


q1 = insincere_df.question_text.values[0]
q1


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re


# Tokenization
q0_tok = word_tokenize(q0)
q1_tok = word_tokenize(q1)


", ".join(stopwords.words('english'))


def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stopwords.words('english')]


# Removing stopwords
q0_stp = remove_stopwords(q0_tok)
q0_stp


# Removing stopwords
q1_stp = remove_stopwords(q1_tok)
q1_stp


stemmer = SnowballStemmer(language='english')


# Perform stemming
q0_stm = [stemmer.stem(word) for word in q0_stp]
q0_stm


# Perform stemming
q1_stm = [stemmer.stem(word) for word in q1_stp]
q1_stm


small_df = sample_df[:5]
small_df


small_df.question_text.values


from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(binary=True)


cv.fit(small_df.question_text)


cv.vocabulary_


cv.get_feature_names_out()


small_df.question_text.values[0]


vectors = cv.transform(small_df.question_text)
vectors[0].toarray()


def tokenize(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    # Tokenize, remove stopwords, and stemming
    return [stemmer.stem(word) for word in word_tokenize(text) if word not in stopwords.words('english')]


tokenize('What is the really (dealing) here?')


vectorizer = CountVectorizer(tokenizer=tokenize, max_features=1000, binary=True)


%%time
vectorizer.fit(sample_df.question_text)


vectorizer.vocabulary_


vectorizer.get_feature_names_out()[:100]


%%time
inputs = vectorizer.transform(sample_df.question_text)


sample_df.question_text.values[0]


inputs[0].toarray()


test_df


%%time
test_inputs = vectorizer.transform(test_df.question_text)


test_df.question_text.values[0]


test_inputs[0].toarray()


from sklearn.model_selection import train_test_split


train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, sample_df.target, test_size=0.3, random_state=42)


train_inputs.shape, train_targets.shape


val_inputs.shape, val_targets.shape


from sklearn.linear_model import LogisticRegression


MAX_ITER = 1000


model = LogisticRegression(max_iter=MAX_ITER, solver='sag')


%%time
model.fit(train_inputs, train_targets)


train_preds = model.predict(train_inputs)
pd.Series(train_preds).value_counts()


train_targets.value_counts()


model.score(train_inputs, train_targets)


model.score(train_inputs, np.zeros(train_targets.shape))


from sklearn.metrics import f1_score


f1_score(train_targets, train_preds)


f1_score(train_targets, np.zeros(len(train_targets)))


random_preds = np.random.choice((0, 1), len(train_targets))
f1_score(train_targets, random_preds)


val_preds = model.predict(val_inputs)
val_preds


model.score(val_inputs, val_targets)


f1_score(val_targets, val_preds)


sincere_df.question_text.values[:10]


sincere_df.target.values[:10]


model.predict(vectorizer.transform(sincere_df.question_text.values[:10]))


insincere_df.question_text.values[:10]


insincere_df.target.values[:10]


model.predict(vectorizer.transform(insincere_df.question_text.values[:10]))


test_df


test_preds = model.predict(test_inputs)
test_preds


sub_df


sub_df['prediction'] = test_preds
sub_df


sub_df.to_csv('submission.csv', index=None)

