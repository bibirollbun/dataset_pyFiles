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


train_data = pd.read_csv("/kaggle/input/ml-olympiad-tfugsurabaya-2024/train.tsv", sep='\t')
train_data.head()

train_df = pd.read_csv('/kaggle/input/ml-olympiad-tfugsurabaya-2024/train.tsv', sep='\t')
test_df = pd.read_csv('/kaggle/input/ml-olympiad-tfugsurabaya-2024/test.tsv', sep='\t')
sample_submission_df = pd.read_csv('/kaggle/input/ml-olympiad-tfugsurabaya-2024/sample_submission.csv', sep=';')

print("Data Train")
print(train_df.head())
print(train_df.shape)
print("Label:")
print(train_df['LABEL'].value_counts())

print("Data Test)")
print(test_df.head())
print(test_df.shape)


!pip install nltk -q

# Import library
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Dapatkan daftar stopwords Bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(cleaned_tokens)

train_df['CLEANED_REVIEW'] = train_df['REVIEW'].apply(preprocess_text)

test_df['CLEANED_REVIEW'] = test_df['REVIEW'].apply(preprocess_text)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True, min_df=2, ngram_range=(1, 1))

X_train = vectorizer.fit_transform(train_df['CLEANED_REVIEW'])
y_train = train_df['LABEL']
X_test = vectorizer.transform(test_df['CLEANED_REVIEW'])

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Inisialisasi model
nb_model = MultinomialNB()
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
svm_model = LinearSVC(random_state=42, dual=True)

models = {
    "Naive Bayes": nb_model,
    "Logistic Regression (MaxEnt)": logreg_model,
    "Support Vector Machine (SVM)": svm_model
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")


print("Melatih model final (LinearSVC) pada seluruh data latih...")
final_model = LinearSVC(random_state=42, dual=True)
final_model.fit(X_train, y_train)

# Membuat prediksi pada data uji
predictions = final_model.predict(X_test)

print(predictions[:10])


submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'LABEL': predictions
})

submission_df.to_csv('submission.csv', index=False, sep=',')

print(submission_df.head())

