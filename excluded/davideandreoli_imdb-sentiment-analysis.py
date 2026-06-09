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


import zipfile
train_data_labeled = pd.read_csv(zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip").open("labeledTrainData.tsv"), delimiter="\t", header=0, quoting=3)
train_data_unlabeled = pd.read_csv(zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip").open("unlabeledTrainData.tsv"), delimiter="\t", header=0, quoting=3)
test_data = pd.read_csv(zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip").open("testData.tsv"), delimiter="\t", header=0, quoting=3)


from bs4 import BeautifulSoup 
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('punkt_tab')


def preprocess(review: str, remove_puncutation: bool = True, remove_stopwords: bool = True):
    review_text = BeautifulSoup(review).get_text()
    review_text = review_text.lower()
    if remove_puncutation:
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

def clean_review(review: str, remove_puncutation: bool = True, remove_stopwords: bool = True):
    words = preprocess(review, remove_puncutation, remove_stopwords)
    return( " ".join( words ))


train_data_labeled["review_clean"] = train_data_labeled["review"].map(lambda x : clean_review(x))
train_data_unlabeled["review_clean"]  = train_data_unlabeled["review"].map(lambda x : clean_review(x))
positive_reviews = train_data_labeled[train_data_labeled["sentiment"] == 1]
negative_reviews = train_data_labeled[train_data_labeled["sentiment"] == 0]


sns.countplot(train_data_labeled, x="sentiment", hue="sentiment")


fig, axes = plt.subplots(1, 2, figsize=(15, 10))
positive_wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Dark2").generate(" ".join(positive_reviews["review_clean"].to_list()))
negative_wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Dark2").generate(" ".join(negative_reviews["review_clean"].to_list()))
axes[0].imshow(positive_wordcloud, interpolation='bilinear')
axes[0].axis("off")
axes[0].set_title("Positive Reviews")
axes[1].imshow(negative_wordcloud, interpolation='bilinear')
axes[1].axis("off")
axes[1].set_title("Negative Reviews")
plt.show()


from collections import Counter
positive_counter = Counter(re.findall(r'\w+', " ".join(positive_reviews["review_clean"].to_list())))
negative_counter = Counter(re.findall(r'\w+', " ".join(negative_reviews["review_clean"].to_list())))
all_counter = Counter(re.findall(r'\w+', " ".join(train_data_labeled["review_clean"].to_list())))


fig, axes = plt.subplots(3, 1, figsize=(10, 15))
sns.barplot(x=[x[0] for x in positive_counter.most_common()[:10]], y=[x[1] for x in positive_counter.most_common()[:10]], ax=axes[0])
sns.barplot(x=[x[0] for x in negative_counter.most_common()[:10]], y=[x[1] for x in negative_counter.most_common()[:10]], ax=axes[1])
sns.barplot(x=[x[0] for x in all_counter.most_common()[:10]], y=[x[1] for x in all_counter.most_common()[:10]], ax=axes[2])
axes[0].set_title("Positive Reviews")
axes[1].set_title("Negative Reviews")
axes[2].set_title("All Reviews")


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

preprocessor = Pipeline([
    ("vectorizer", CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=5000
    ))
])

X_train, X_test, y_train, y_test = train_test_split(train_data_labeled["review_clean"], train_data_labeled["sentiment"], random_state=42)


param_grids = {
    "LogisticRegression": {
        "model__penalty": ['l1', 'l2'],
        "model__C": [0.1, 1, 10],
        "model__solver": ['liblinear', 'saga'],
        "model__max_iter": [500]
    },
    "LinearSVC": {
        "model__C": [0.1, 1, 10],
        "model__loss": ['hinge', 'squared_hinge'],
        "model__max_iter": [5000]
    },
    "RidgeClassifier": {
        "model__alpha": [0.1, 1.0, 10.0],
        "model__solver": ['auto', 'lsqr', 'sag']
    },
    "SGDClassifier": {
        "model__loss": ['hinge', 'log_loss', 'modified_huber'],
        "model__alpha": [1e-4, 1e-3, 1e-2],
        "model__penalty": ['l2', 'l1', 'elasticnet'],
        "model__max_iter": [1000],
        "model__tol": [1e-3]
    },
    "PassiveAggressiveClassifier": {
        "model__C": [0.1, 1, 10],
        "model__loss": ['hinge', 'squared_hinge'],
        "model__max_iter": [1000],
        "model__tol": [1e-3]
    },
    "ComplementNB": {
        "model__alpha": [0.1, 0.5, 1.0],
        "model__norm": [True, False]
    }
}

models = [
    LogisticRegression(random_state=42),
    LinearSVC(random_state=42),
    RidgeClassifier(),
    SGDClassifier(random_state=42),
    PassiveAggressiveClassifier(random_state=42),
    ComplementNB()
]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_score = 0
best_model = None
best_estimators = []

for model in models:
    model_name = model.__class__.__name__
    print(model_name)

    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    grid_search = GridSearchCV(
        model_pipeline,
        param_grids[model_name],
        scoring={'accuracy': 'accuracy'},
        refit='accuracy',
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    test_score = grid_search.score(X_test, y_test)

    best_estimators.append((model_name, grid_search.best_estimator_.named_steps["model"]))

    if test_score > best_score:
        best_score = test_score
        best_model = grid_search.best_estimator_


accuracy_score(y_test, best_model.predict(X_test))


%%script false --no-raise-error
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from datasets import Dataset
from transformers import Trainer, TrainingArguments

X_train_bert, X_val, y_train_bert, y_val = train_test_split(X_train, y_train, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(
    "prajjwal1/bert-mini",
)
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-mini", num_labels=2)


%%script false --no-raise-error
from datasets import Dataset

MAX_LENGTH = 256
train_dataset = Dataset.from_dict({"text": X_train_bert, "label": y_train_bert})
val_dataset   = Dataset.from_dict({"text": X_val, "label": y_val})
train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], max_length = MAX_LENGTH, truncation = True, padding = True, return_tensors='pt'), batched=True)
val_dataset   = val_dataset.map(lambda x: tokenizer(x["text"], max_length = MAX_LENGTH, truncation = True, padding = True, return_tensors='pt'), batched=True)

#X_train_encoded = tokenizer.batch_encode_plus(X_train.to_list(), max_length = 512, truncation = True, padding = True, return_tensors='pt')
#X_test_encoded = tokenizer.batch_encode_plus(X_test.to_list(), max_length = 512, truncation = True, padding = True, return_tensors='pt')
#X_val_encoded = tokenizer.batch_encode_plus(X_val.to_list(), max_length = 512, truncation = True, padding = True, return_tensors='pt')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    eval_dataset=val_dataset, 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


%%script false --no-raise-error
test_dataset = Dataset.from_dict({"text": X_test, "label": y_test})
test_dataset = test_dataset.map(lambda x: tokenizer(x["text"],max_length=MAX_LENGTH,truncation=True,padding=True),batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

metrics = trainer.evaluate(test_dataset)
print(metrics)


import csv
predictions = best_model.predict(test_data["review"])
output = pd.DataFrame({'id': test_data.id, 'sentiment': predictions})
output.to_csv('submission.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

