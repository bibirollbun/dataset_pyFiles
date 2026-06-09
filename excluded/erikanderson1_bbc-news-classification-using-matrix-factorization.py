from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')
data.head()


data.loc[0, "Text"]


data.isna().sum()


data.duplicated(subset=['Text'], keep=False).sum()


data = data.drop_duplicates(subset=['Text'])
data.duplicated(subset=['Text'], keep=False).sum()


sns.countplot(data=data, x="Category")
plt.title("Articles by category");


data["Text_Length"] = data["Text"].apply(len)

plt.figure(figsize=(10, 6))

for category in data["Category"].unique():
    subset = data[data["Category"] == category]
    sns.kdeplot(subset["Text_Length"], label=category, fill=True, alpha=0.3)

plt.xlabel("Text Length")
plt.ylabel("Density")
plt.title("Kernel Density Estimate of Text Length by Category")
plt.legend(title="Category")
plt.show();


nlp = spacy.load("en_core_web_sm")
docs = list(nlp.pipe(data["Text"], 
                     disable=[
                         "tok2vec", "tagger", "parser", 
                         "attribute_ruler", "ner"])
           )


def stem_text(doc):
    text = ' '.join([token.lemma_ for token in doc 
                     if not token.is_stop 
                     and not token.is_punct
                     and not token.is_space])
    return text


data["text_stem"] = [stem_text(doc) for doc in docs]
data.loc[0, "text_stem"]


vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(data['text_stem'])

print(X_tfidf.shape)


n_topics = len(data['Category'].unique())
nmf = NMF(n_components=n_topics, random_state=42)
X_nmf = nmf.fit_transform(X_tfidf)

print(X_nmf.shape)


X_nmf[0,:]


data["predicted"] = X_nmf.argmax(axis=1)
data.head()


def get_labels(y_true, y_pred):
    topic_mapping = {}
    y_true = np.array(y_true)

    for topic in np.unique(y_pred):
        indices = np.where(y_pred == topic)[0]
        topic_docs_labels = y_true[indices] if len(indices) > 0 else []

        if len(topic_docs_labels) > 0:
            most_common_label = Counter(topic_docs_labels).most_common(1)[0][0]
            topic_mapping[topic] = most_common_label
        else:
            topic_mapping[topic] = -1  # -1 represents "Unknown" category

    return topic_mapping

topic_mapping = get_labels(data['Category'], data['predicted'])
topic_mapping


topic_mapping = get_labels(data['Category'], data['predicted'])
topic_mapping


data["predicted_label"] = data['predicted'].map(topic_mapping)
data.head()


cm = confusion_matrix(data["Category"], data["predicted_label"], labels=list(topic_mapping.values()))

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(topic_mapping.values()))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix for NMF")
plt.show()


print(f'Accuracy: {accuracy_score(data["Category"], data["predicted_label"]):.2%}')


test_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
test_docs = list(nlp.pipe(test_df["Text"], 
                     disable=[
                         "tok2vec", "tagger", "parser", 
                         "attribute_ruler", "ner"])
           )
test_df["text_stem"] = [stem_text(doc) for doc in test_docs]
test_tfidf = vectorizer.transform(test_df['text_stem'])


test_pred = nmf.transform(test_tfidf).argmax(axis=1)
test_pred_labels = np.array([topic_mapping.get(t, "Unknown") for t in test_pred])
submission = pd.DataFrame({
    'ArticleId': test_df['ArticleId'],
    'Category': test_pred_labels
})
submission.to_csv('submission_nmf_defaults.csv', index=False)


y_true = data['Category']


# Custom scoring function for classifier
def nmf_accuracy(model, X, y_true):
    """Computes accuracy based on best topic-to-category mapping."""
    X_topics = model.transform(X)
    y_pred = X_topics.argmax(axis=1)

    topic_mapping = get_labels(y_true, y_pred)
    
    y_pred_labels = np.array([topic_mapping.get(t, "Unknown") for t in y_pred])

    return accuracy_score(y_true, y_pred_labels)


# Define the NMF model
nmf = NMF(random_state=42, max_iter=100)  # Adjust max_iter for stability

# Define the parameter grid
param_grid = {
    'nmf__n_components': [5],
    'nmf__alpha_W': [0.0, 0.1, 0.5],
    'nmf__alpha_H': [0.0, 0.1, 0.5],
    'nmf__l1_ratio': [0.0, 0.5, 1.0],
    'nmf__solver': ['cd', 'mu']
}

# Define the pipeline
pipeline = Pipeline([
    ('nmf', nmf)
])

# Perform Grid Search with accuracy scoring
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring=nmf_accuracy,
    cv=3,
    verbose=2,
    n_jobs=-1  # Use all CPU cores
)

# Fit GridSearchCV on the TF-IDF matrix
grid_search.fit(X_tfidf, y_true)

# Get the best parameters
print("Best Parameters:", grid_search.best_params_)

# Get the best model
best_nmf = grid_search.best_estimator_



print(f"Best Accuracy: {nmf_accuracy(best_nmf, X_tfidf, y_true):.2%}")


# Get labels for best model
X_topics = best_nmf.transform(X_tfidf)
y_pred = X_topics.argmax(axis=1)
topic_mapping = get_labels(y_true, y_pred)
y_pred_labels = np.array([topic_mapping.get(t, "Unknown") for t in y_pred])

cm = confusion_matrix(y_true, y_pred_labels, labels=list(topic_mapping.values()))

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(topic_mapping.values()))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix for NMF")
plt.show()


test_pred = best_nmf.transform(test_tfidf).argmax(axis=1)
test_pred_labels = np.array([topic_mapping.get(t, "Unknown") for t in test_pred])
submission = pd.DataFrame({
    'ArticleId': test_df['ArticleId'],
    'Category': test_pred_labels
})
submission.to_csv('submission_nmf_tuned.csv', index=False)


from sklearn.ensemble import HistGradientBoostingClassifier
import time

X = data["text_stem"]
y = data["Category"]

for pct in [0.1, 0.2, 0.5, 0.7, 0.9]:
    start_time = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - pct, random_state=42)
    
    X_tfidf_train = vectorizer.transform(X_train)
    
    hgbc = HistGradientBoostingClassifier(max_iter=100)
    hgbc = hgbc.fit(X_tfidf_train.toarray(), y_train)

    X_tfidf_test = vectorizer.transform(X_test)
    accuracy = hgbc.score(X_tfidf_test.toarray(), y_test)
    
    end_time = time.time()
    execution_time = (end_time - start_time)

    print(f'Accuracy of {accuracy:.2%} for {pct:.0%} of the data in training set. Execution Time: {execution_time:.2f}s')


test_pred_labels = hgbc.predict(test_tfidf.toarray())
submission = pd.DataFrame({
    'ArticleId': test_df['ArticleId'],
    'Category': test_pred_labels
})
submission.to_csv('submission_hgbc.csv', index=False)

