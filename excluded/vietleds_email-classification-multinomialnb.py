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


train_df = pd.read_csv(
    "/kaggle/input/csep546-aut19-kc1/SMSSpamCollection",
    sep='\t',
    header=None,
    names=['label', 'text']
)
train_df['label'] = train_df['label'].astype(str).str.strip().map({'ham':0, 'spam':1})
train_df.head()


train_df.info()


train_df['label'].unique()


test_df = pd.read_csv(
    "/kaggle/input/csep546-aut19-kc1/SMSSpamCollection_test",
    sep='!@#$',  # mỗi dòng 1 bản ghi
    header=None,
    names=['line'],
    encoding='latin-1'
)
# Tách ID và text theo space đầu tiên
test_df[['id', 'text']] = test_df['line'].str.split(' ', n=1, expand=True)
test_df = test_df.drop(columns=['line'])
test_df.head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def OutputSubmission(path, IDs, predictions):
    kNumberExamplesExpected = 100

    if len(IDs) != kNumberExamplesExpected or len(predictions) != kNumberExamplesExpected:
        print("Incorrect number of IDs or predictions. Expected %d." % (kNumberExamplesExpected))
        return

    f = open(path, 'w')
    f.write("ID,<0/1>\n")

    for i in range(len(IDs)):
       f.write("%d,%d\n" % (IDs[i], predictions[i]))

    f.flush()
    f.close()


X = train_df['text']
y = train_df['label']
X_test = test_df['text']
IDs = test_df['id']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_name = 'nb'
models = {
    'nb': MultinomialNB(),
    'lr': LogisticRegression(max_iter=1000),
    'svm': LinearSVC()
}

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('model', models[model_name])
])

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_valid)
print(classification_report(y_valid, y_pred))


OutputSubmission('/kaggle/working/submission_file.txt', IDs.astype(int), pipeline.predict(X_test))


# Chỉ áp dụng với Naive Bayes
if model_name == 'nb':
    # Lấy model và vectorizer từ pipeline
    nb_model = pipeline.named_steps['model']
    vectorizer = pipeline.named_steps['tfidf']

    # Tên feature (từ/ngram)
    feature_names = vectorizer.get_feature_names_out()

    # log probabilities P(x_i | class)
    log_prob_spam = nb_model.feature_log_prob_[1]  # class 1 = spam
    log_prob_ham  = nb_model.feature_log_prob_[0]  # class 0 = ham

    # Lấy top 20 từ spam
    top_spam_idx = np.argsort(log_prob_spam)[-20:][::-1]  # từ lớn nhất → nhỏ dần
    top_spam_keywords = feature_names[top_spam_idx]
    print("Top 20 spam keywords:")
    print(top_spam_keywords)

    # Lấy top 20 từ ham
    top_ham_idx = np.argsort(log_prob_ham)[-20:][::-1]
    top_ham_keywords = feature_names[top_ham_idx]
    print("Top 20 ham keywords:")
    print(top_ham_keywords)


# Minh họa TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love NLP",
    "I love ML",
    "NLP and ML are fun"
]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())

