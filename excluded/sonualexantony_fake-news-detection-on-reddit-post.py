#Necessary Libraries
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

#Preprocessing
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


df = pd.read_csv('/kaggle/input/depi-r-2-competition-1/xy_train.csv')
df.head()


def preprocess_text(text):
    #Convert text to lowercase
    text = text.lower()
    #Remove Special Characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    #Tokenize
    tokens = word_tokenize(text)
    #Remove Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)


tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=3)


models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'Multinomial NB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier()
}


for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)


test_df = pd.read_csv('/kaggle/input/depi-r-2-competition-1/x_test.csv')
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])


for name, model in models.items():
    model.fit(X_tfidf, df['label'])
    y_pred = model.predict(X_test_tfidf)

    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'label': y_pred.ravel()
    })

    submission.to_csv(name[:5]+'_submission.csv', index=False)

