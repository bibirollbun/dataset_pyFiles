import re, string, nltk, joblib, warnings, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB,ComplementNB
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)
from sklearn.ensemble import BaggingClassifier,VotingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

import spacy
from sklearn.feature_selection import SelectKBest, chi2

from nltk.stem import WordNetLemmatizer,SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import (Embedding,LSTM,Dense,
                                     Input,Dropout,GRU,SimpleRNN)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.metrics import AUC

from sklearn.calibration import CalibratedClassifierCV


import scipy.sparse as sp

warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


news_df = pd.read_csv("/kaggle/input/depi-r-3-competition-1/xy_train.csv")
print("shape:",news_df.shape)
news_df.head(3)


def preprocess_text(text):
    """
    Clean and normalize raw text for downstream NLP tasks.

    Steps performed
    ---------------
    1. Lower-case the entire string.
    2. Replace URLs with the literal token ``URL``.
    3. Replace standalone numbers with the literal token ``NUM``.
    4. Remove punctuation and symbols, keeping only letters, spaces, apostrophes
       and hyphens *when they appear between letters*.
    5. Collapse multiple whitespace characters and strip leading/trailing spaces.
    6. Split on whitespace (``str.split``) and drop tokens shorter than 2
       characters.
    7. Remove English stop-words (NLTK list) and tokens whose length is ≤ 2.
    8. Lemmatize remaining tokens with ``WordNetLemmatizer``.
    9. Re-assemble tokens into a single space-separated string.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned, lemmatized, space-separated token string. Returns an empty
        string if no tokens survive filtering.
    """

    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b\d+\b', ' NUM ', text)
    text = re.sub(r"[^a-z\s'-]", " ", text)
    text = re.sub(r"(?<![a-z])[-']|[-'](?![a-z])", " ", text)
    text = re.sub(r"([a-z])[-'](?=[a-z])", r"\1 ", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # there a problem in this word tokenizer as it takes common english words. for example it takes don't and sperate it into do n't
    #tokens = word_tokenize(text)
    # Tokenize words but with a length greater than 1
    #tokens = [word for word in tokens if len(word)>1]


    # Instead of using tokenizer to split text
    tokens = text.split(sep=' ')
    tokens = [word for word in tokens if len(word)>1]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word)>2]

    # lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]

    return ' '.join(tokens)



# Remove records with label value 2 as we have only two options (0 & 1)
news_df = news_df[news_df["label"]!=2]


# Apply preprocessing() function to all texts
news_df["clean_text"] = news_df["text"].apply(preprocess_text)



news_df["clean_text"].sample(10)


# min number of words is 0 mean that after cleaning we still have Missing values
news_df['clean_text'].apply(lambda x: len(x.split())).describe()


# Make a new columns contain number of words in each record
news_df['n_words'] = news_df['clean_text'].apply(lambda x: len(x.split()))


# Maintain the records contain higher than 3 words
news_df = news_df[news_df["n_words"]>=3]


# Check again to make sure
news_df['clean_text'].apply(lambda x: len(x.split())).describe()


Count_Vectorizer = CountVectorizer(analyzer='char',
                              max_features = 120_000,
                              ngram_range=(3,5),
                              min_df=2,max_df=0.9)


textss = Count_Vectorizer.fit_transform(news_df["text"])


x_train, x_val, y_train, y_val = train_test_split(news_df["text"],news_df["label"],test_size=0.2,random_state=7,stratify=news_df["label"])


print("Train:", x_train.shape, "Val:", x_val.shape)



def evaluate_model(model, x_train, x_val, y_train, y_val):
    """
    Quick diagnostic helper for a fitted scikit-learn estimator.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
    x_train : array-like or dataframe
        Training features (raw text allowed if pipeline handles vectorization).
    x_val : array-like or dataframe
        Validation features (same format as ``x_train``).
    y_train : array-like
        Ground-truth labels for the training set.
    y_val : array-like
        Ground-truth labels for the validation set.

    Returns
    -------
    None
        Results are printed:
        1. Training accuracy
        2. Validation accuracy
    """
    print("=====================")
    print(f"Accuracy on training data: {model.score(x_train, y_train)}")
    print("=====================")
    print(f"Accuracy on validation data: {model.score(x_val, y_val)}")


def deep_eval(model, x_val, y_val, name="model"):
    """
    Full diagnostic report for a classifier.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        A **fitted** model that implements ``.predict`` and either
        ``.predict_proba`` or ``.decision_function``.
    x_val : array-like or dataframe
        Validation features (raw text allowed if pipeline handles
        vectorisation).
    y_val : array-like
        True validation labels (0/1).
    name : str, optional
        Nickname printed at the top of the report (default "model").

    Prints
    ------
    - Accuracy
    - Macro F1-score
    - ROC-AUC  (uses ``predict_proba`` if available, else ``decision_function``)
    - Full classification report (precision / recall / f1 per class)
    - Raw confusion matrix

    Returns
    -------
    None
    """
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 classification_report, confusion_matrix)

    y_pred = model.predict(x_val)
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_val, y_pred))
    print("F1       :", f1_score(y_val, y_pred))
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_val)[:, 1]
    else:
        y_prob = model.decision_function(x_val)
    print("ROC-AUC  :", roc_auc_score(y_val, y_prob))
    print("\nClassification report:\n", classification_report(y_val, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))


nb_pipe = nb_pipe = Pipeline([
    ("tfidf", CountVectorizer(analyzer='char',
                              max_features = 120_000,
                              ngram_range=(3,5),
                              max_df=0.9,min_df=2)),
    ('select',SelectKBest(chi2, k=60_000)),
    ("clf", CalibratedClassifierCV(MultinomialNB(fit_prior=True,alpha=0.2),method='isotonic'))
])


nb_pipe.fit(x_train,y_train)


evaluate_model(nb_pipe,x_train,x_val,y_train,y_val)


deep_eval(nb_pipe,x_val,y_val,"Naive Bayes")


# Final Evaluation
print("Balanced accuracy_score: ",balanced_accuracy_score(y_val,nb_pipe.predict(x_val)))


nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=60_000,
                              ngram_range=(1,1),
                              min_df=3,
                              sublinear_tf=False)),
    ('slect',SelectKBest(chi2, k=30_000)),
    ("clf",  CalibratedClassifierCV(estimator= MultinomialNB(alpha=1,
                                                  fit_prior=False)))
])


param = {
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
    'tfidf__max_features': [30_000, 60_000, 100_000],
    'clf__estimator__alpha': [0.1, 0.2, 0.5, 1.0],
    'clf__estimator__fit_prior':[True,False],
    'clf__method':["sigmoid","isotonic"]
}



grid = GridSearchCV(estimator= nb_pipe,cv=3, param_grid= param, scoring='accuracy', verbose=2)


grid.fit(x_train, y_train)



print("Best val accuracy:", grid.best_score_)
print(grid.best_params_)



# Naive bayes model with best params:
nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=grid.best_params_["tfidf__max_features"],
                              ngram_range=grid.best_params_["tfidf__ngram_range"],
                              min_df=2,
                              sublinear_tf=True)),
    ('slect',SelectKBest(chi2, k=30_000)),
    ("clf",  CalibratedClassifierCV(estimator= MultinomialNB(alpha=grid.best_params_["clf__estimator__alpha"],
                                                  fit_prior=grid.best_params_["clf__estimator__fit_prior"])
                                                  ,method=grid.best_params_["clf__method"]))
])


nb_pipe.fit(x_train,y_train)


evaluate_model(nb_pipe,x_train,x_val,y_train,y_val)


char_gram = TfidfVectorizer(analyzer='char',
                           ngram_range=(3, 5),
                           max_features=120_000,
                           min_df=2,
                           max_df=0.95,
                           sublinear_tf=True)

word_gram = TfidfVectorizer(ngram_range=(1, 2),
                           max_features=80_000,
                           min_df=2,
                           max_df=0.9,
                           sublinear_tf=True)


log_pipe = Pipeline([
    ('feature_union', FeatureUnion([('char', char_gram), ('word', word_gram)])),
    ("clf",  LogisticRegression(solver='saga',
                                max_iter=1000))
])


log_pipe.fit(x_train, y_train)


evaluate_model(log_pipe,x_train,x_val,y_train,y_val)


deep_eval(log_pipe,x_val,y_val,"Logistic Regression")


# Final Evaluation
print("Balanced accuracy_score: ",balanced_accuracy_score(y_val,log_pipe.predict(x_val)))


char_gram = TfidfVectorizer(analyzer='char',
                           ngram_range=(3, 5),
                           max_features=120_000,
                           min_df=2,
                           max_df=0.95,
                           sublinear_tf=True)

word_gram = TfidfVectorizer(ngram_range=(1, 2),
                           max_features=80_000,
                           min_df=2,
                           max_df=0.9,
                           sublinear_tf=True)


svm_pipe = Pipeline([
    ('feature_union', FeatureUnion([('char', char_gram), ('word', word_gram)])),
    ('clf',  LinearSVC(C=0.15,
                       class_weight='balanced',
                       max_iter=20_000,
                       loss='squared_hinge'))
])


svm_pipe.fit(x_train,y_train)


evaluate_model(svm_pipe,x_train,x_val,y_train,y_val)


deep_eval(svm_pipe,x_val,y_val,"SVM Model")


# Final Evaluation
print("Balanced accuracy_score: ",balanced_accuracy_score(y_val,svm_pipe.predict(x_val)))


hash_vec = HashingVectorizer(n_features=40_000,
                             ngram_range=(1,3))

sdg_model = SGDClassifier(loss='hinge',
                        penalty='l2',
                        class_weight='balanced',
                        random_state=42)


sdg_pipe = Pipeline([('ext',hash_vec),('clf',sdg_model)])


sdg_pipe.fit(x_train,y_train)


evaluate_model(sdg_pipe,x_train,x_val,y_train,y_val)


# Bossting using 3 best algorithms
Boost_vote_model = VotingClassifier(estimators=[('svc',svm_pipe),("np",nb_pipe),('log_reg',log_pipe)], voting='hard')


Boost_vote_model.fit(x_train, y_train)


evaluate_model(Boost_vote_model,x_train,x_val,y_train,y_val)


# Final Evaluation
print("Balanced accuracy_score: ",balanced_accuracy_score(y_val,Boost_vote_model.predict(x_val)))


rf_pipe = Pipeline([
    ('tfidf', CountVectorizer(max_features=60_000,
                              ngram_range=(1,2),
                              min_df=3)),
    ('clf',RandomForestClassifier(
                      n_estimators=400,
                      max_depth=25,
                      min_samples_split=5,
                      min_samples_leaf=2,
                      class_weight='balanced',
                      random_state=15))
])


rf_pipe.fit(x_train,y_train)


evaluate_model(rf_pipe,x_train,x_val,y_train,y_val)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

vocab_size = len(tokenizer.word_index)+1
print("Vocab size: ",vocab_size)


x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)


x_train[:3]


max_len = 30


x_train = pad_sequences(x_train,padding="post",maxlen=max_len,truncating="post")


x_val = pad_sequences(x_val,padding="post",maxlen=max_len,truncating="post")


x_train = np.array(x_train, dtype=np.uint16)
x_val = np.array(x_val, dtype=np.uint16)



emb_dim = 60
n_epochs = 10
batch_size = 10
lr = 1e-3
drop_out=0.4


rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')

])



rnn_model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=['accuracy'])


early_stop = EarlyStopping(patience=2, restore_best_weights=True)



rnn_model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_data=(x_val,y_val),
              callbacks=[early_stop])


loss, accuracy = rnn_model.evaluate(x_val, y_val, batch_size=batch_size)
print(f'Validation loss : {loss:.4f}')
print(f'Validation Accuracy   : {accuracy:.4f}')






# Save the best model
import joblib, pathlib, datetime

# choose whichever model you trained
model = svm_pipe

fname = f"{model.steps[-1][0]}_{datetime.date.today():%Y%m%d}.pkl"
joblib.dump(model, fname)

from google.colab import files
files.download(fname)


# Get load the model downloaded
model = joblib.load("/content/model.pkl")



x_test = pd.read_csv("/kaggle/input/depi-r-3-competition-1/x_test.csv")



x_test.head(3)


y_test_pred = model.predict(x_test["text"])


x_test["label"] = y_test_pred


x_test.drop(columns=["text"],inplace=True)


x_test.to_csv('predictions.csv', index=False)




