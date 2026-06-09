!ls


!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json




import os

os.listdir('/kaggle/input/')



os.listdir('/kaggle/input/quora-insincere-questions-classification/')



# os.environ['KAGGLE_CONFIG_DIR'] = ' .'


# !pip install -q kaggle



# !kaggle competitions download -c quora-insincere-questions-classification -f train.csv -p data


# !kaggle competitions download -c quora-insincere-questions-classification -f test.csv -p data
# !kaggle competitions download -c quora-insincere-questions-classification -f sample_submission.csv -p data


# Rename .csv to .zip (only if these are actually ZIP archives)
# !mv data/train.csv data/train.zip
# !mv data/test.csv data/test.zip
# !mv data/sample_submission.csv data/sample_submission.zip

# # Unzip all of them to the same 'data/' folder
# !unzip data/train.zip -d data/
# !unzip data/test.zip -d data/
# !unzip data/sample_submission.zip -d data/

# # Optional: Remove the zip files to avoid confusion
# !rm data/*.zip

# # Check extracted files
# !ls data



# train_fname = 'data/train.csv'
# test_fname = 'data/test.csv'
# sample_fname = 'data/sample_submission.csv'


data_dir = '/kaggle/input/quora-insincere-questions-classification'

train_fname = data_dir + '/train.csv'
test_fname = data_dir + '/test.csv'
sample_fname = data_dir + '/sample_submission.csv'



import pandas as pd


raw_df = pd.read_csv(train_fname)


raw_df


sincere_df = raw_df[raw_df.target == 0]


sincere_df.question_text.values[:10]


insincere_df = raw_df[raw_df.target == 1]


insincere_df.question_text.values[:10]


raw_df.target.value_counts(normalize=True).plot(kind='bar')


test_df = pd.read_csv(test_fname)


test_df


sub_df = pd.read_csv(sample_fname)


sub_df


sub_df.prediction.value_counts()


SAMPLE_SIZE = 100_000


sample_df = raw_df.sample(SAMPLE_SIZE, random_state= 42)


sample_df


sincere_df


q0 = sincere_df.question_text.values[1]


q0


q1 = raw_df[raw_df.target == 1].question_text.values[0]


q1


import nltk


from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('punkt_tab')


q0


word_tokenize(q0)


q1


word_tokenize(q1)


q0_tok = word_tokenize(q0)
q1_tok = word_tokenize(q1)


q1_tok


from nltk.corpus import stopwords


nltk.download('stopwords')
english_stopwords = stopwords.words('english')


", ".join(english_stopwords)


def remove_stopwords(tokens):
  return [word for word in tokens if word.lower() not in english_stopwords]


q0_tok


q0_stp = remove_stopwords(q0_tok)


q0_stp


q1_stp = remove_stopwords(q1_tok)


q1_stp


from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer(language ='english')


stemmer.stem('going')


stemmer.stem('supposedly')


q0_stm = [stemmer.stem(word) for word in q0_stp]


q0_stp


q0_stm


q1_stm = [stemmer.stem(word) for word in q1_stp]


q1_stp


q1_stm


import nltk
nltk.download('wordnet')


from nltk.stem import WordNetLemmatizer as wnl
print(wnl().lemmatize('us', 'n'))


sample_df


small_df = sample_df[:5]


small_df.question_text.values


from sklearn.feature_extraction.text import CountVectorizer


small_vect = CountVectorizer()


small_vect.fit(small_df.question_text)


small_vect.get_feature_names_out()


vectors = small_vect.transform(small_df.question_text)


vectors


vectors.toarray()


vectors[0].toarray()


vectors.shape


stemmer = SnowballStemmer(language='english')


def tokenize(text):
    return [stemmer.stem(word) for word in word_tokenize(text)]


tokenize('What is the really (dealing) here?')


vectorizer = CountVectorizer(lowercase=True,
                             tokenizer=tokenize,
                             stop_words=english_stopwords,
                             max_features=1000)


%%time
vectorizer.fit(sample_df.question_text)


len(vectorizer.vocabulary_)


vectorizer.get_feature_names_out()[:100]


%%time
inputs = vectorizer.transform(sample_df.question_text)


inputs.shape


inputs


sample_df.question_text.values[0]


test_df


%%time
test_inputs = vectorizer.transform(test_df.question_text)


sample_df


inputs.shape


from sklearn.model_selection import train_test_split


train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, sample_df.target, test_size=0.3, random_state=42)


train_inputs.shape


train_targets.shape


val_inputs.shape


val_targets.shape


from sklearn.linear_model import LogisticRegression


MAX_ITER = 1000


model = LogisticRegression(max_iter = 1000, solver ='sag')


%%time
model.fit(train_inputs, train_targets)


train_pred = model.predict(train_inputs)


train_targets


train_pred


pd.Series(train_pred).value_counts()


pd.Series(train_targets).value_counts()


from sklearn.metrics import accuracy_score


accuracy_score(train_targets, train_pred)


import numpy as np


accuracy_score(train_targets, np.zeros(len(train_targets)))


from sklearn.metrics import f1_score


f1_score(train_targets, train_pred)


f1_score(train_targets, np.zeros(len(train_targets)))


random_preds = np.random.choice((0, 1), len(train_targets))
f1_score(train_targets, random_preds)


val_preds = model.predict(val_inputs)


accuracy_score(val_targets, val_preds)


f1_score(val_targets, val_preds)


sincere_df.question_text.values[:10]


sincere_df.target.values[:10]


model.predict(vectorizer.transform(sincere_df.question_text.values[:10]))


sincere_df.question_text.values[:10]


insincere_df.target.values[:10]


model.predict(vectorizer.transform(insincere_df.question_text.values[:10]))





test_df


test_inputs.shape


test_preds = model.predict(test_inputs)


sub_df.prediction = test_preds


sub_df.prediction.value_counts()


sub_df


sub_df.to_csv('submission.csv', index=None)


!head submission.csv




