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


import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from wordcloud import WordCloud

from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


train_file = "/kaggle/input/learn-ai-bbc/BBC News Train.csv"
test_file = "/kaggle/input/learn-ai-bbc/BBC News Test.csv"


train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


print('Train dataset head: \n')
train_df.head()


print('Test dataset head: \n')
test_df.head()


print('Columns info: \n')
train_df.info()


print(f'Number of train dataset rows: {train_df.shape[0]}, columns: {train_df.shape[1]}')
print(f'Number of test dataset rows: {test_df.shape[0]}, columns: {test_df.shape[1]}')


print("Checking for missing values in each column: \n")
train_df.isnull().sum()


def plot_categories_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Category', data=df, color='#005f99') 
    plt.title('Distribution of Categories in News Articles', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()


plot_categories_distribution(train_df)


duplicates = train_df[train_df.duplicated(subset=['Text'], keep=False)].sort_values(by='Text')
duplicates


plot_categories_distribution(duplicates)


train_df = train_df.drop_duplicates(subset=['Text'])
plot_categories_distribution(train_df)


grouped_text = train_df.groupby('Category')['Text'].apply(lambda x: ' '.join(x))

def plot_wordcloud(category, text, ax):
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='ocean', max_words=100).generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f"Word Cloud for {category}", fontsize=16)
    ax.axis('off')

fig, axes = plt.subplots(1, len(grouped_text), figsize=(20, 10), sharex=True, sharey=True)

for idx, (category, text) in enumerate(grouped_text.items()):
    plot_wordcloud(category, text, axes[idx])

plt.tight_layout()
plt.show()


train_df['article_length'] = train_df['Text'].apply(len)
sns.boxplot(data=train_df, x='Category', y='article_length', palette='ocean')
plt.title('Article Length Distribution Across Categories')
plt.xlabel('Category')
plt.ylabel('Article Length')
plt.show()


def pre_process(df):
    reviews = []
    stopwords_set = set(stopwords.words("english"))
    ps = PorterStemmer()
    for p in tqdm(df['Text']):
        # convert to lowercase 
        p = p.lower()
        # remove punctuation and additional empty strings
        p = ''.join([c for c in p if c not in punctuation])
        reviews_split = p.split()
        reviews_wo_stopwords = [word for word in reviews_split if not word in stopwords_set]
        reviews_stemm = [ps.stem(w) for w in reviews_wo_stopwords]
        p = ' '.join(reviews_stemm)
        reviews.append(p)
    return reviews


train_df['Text_pp'] = pre_process(train_df)
test_df['Text_pp'] = pre_process(test_df)

# compare the same phrase before and after pre-processing
print('Phrase before pre-processing: ', train_df['Text'][0])
print('Phrase after pre-processing: ', train_df['Text_pp'][0])


X = train_df['Text_pp']
y = train_df['Category']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# using all texts in the factorizing matrix does not lead to data leak
# because the labels are not included
X_full = pd.concat([train_df['Text_pp'], test_df['Text_pp']])
y_full = pd.concat([y, pd.DataFrame(["unknown"] * test_df['Text_pp'].shape[0])])


def topic_mapping_accuracy(estimator, X, y_true):
    # Transform the input data using the fitted estimator
    pred = estimator.transform(X_val)
    y_pred = np.argmax(pred, axis=1)
    
    # Define the topic mapping
    topic_mapping = {}
    
    # n_components = 5
    for topic in range(5):
        topic_docs_labels = y_val[y_pred == topic]
        most_common_label = Counter(topic_docs_labels).most_common(1)[0][0]
        topic_mapping[topic] = most_common_label
    
    y_pred = [topic_mapping[topic] for topic in y_pred]
    
    # Calculate accuracy
    return accuracy_score(y_val, y_pred)

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('dim_reduction', NMF()),  # Placeholder for dimensionality reduction
])

# Define the parameter grid for both NMF and LSA
param_grid = [
    {
        'tfidf__max_df': [0.8, 1.0],
        'tfidf__min_df': [1, 2, 5],  
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)], 
        'tfidf__lowercase': [True],
        'dim_reduction': [NMF(n_components=5, random_state=42)],
        'dim_reduction__solver': ['mu'],
        'dim_reduction__beta_loss': ['kullback-leibler'],
        'dim_reduction__init': ['nndsvda','nndsvdar']
    },
    {
        'tfidf__max_df': [0.8, 1.0],  
        'tfidf__min_df': [1, 2, 5],  
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],  
        'tfidf__lowercase': [True],
        'dim_reduction': [TruncatedSVD(n_components=5)],
        'dim_reduction__algorithm': ['randomized']
    }
]

# Store results for comparison
results = []

# Define percentages of training data to use
percentages = [0.1, 0.2, 0.5, 1.0]

for pct in percentages:
    print(f"Training with {int(pct * 100)}% of the data...")
    
    # Create a subset of the training data
    if pct == 1.0:
        X_subset, y_subset = X_full, y_full
    else:
        X_subset, _, y_subset, _ = train_test_split(X_full, y_full, train_size=pct, stratify=y_full, random_state=42)
    
    # Grid search on the subset
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=topic_mapping_accuracy,
        cv=3,  # 3-fold cross-validation
        verbose=1
    )
    grid_search.fit(X_subset, y_subset)
    
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['percentage'] = pct  # Add a column for percentage
    
    results.append(cv_results_df)

# Combine all results into a single DataFrame
final_results_df = pd.concat(results, ignore_index=True)

# Display or save the final results DataFrame
final_results_df.head()


final_results_df_10_pct = final_results_df[final_results_df['percentage'] == 0.1]
final_results_df_10_pct.sort_values(by='mean_test_score', ascending=False).head()


final_results_df_20_pct = final_results_df[final_results_df['percentage'] == 0.2]
final_results_df_20_pct.sort_values(by='mean_test_score', ascending=False).head()


final_results_df_50_pct = final_results_df[final_results_df['percentage'] == 0.5]
final_results_df_50_pct.sort_values(by='mean_test_score', ascending=False).head()


final_results_df_100_pct = final_results_df[final_results_df['percentage'] == 1.0]
final_results_df_100_pct.sort_values(by='mean_test_score', ascending=False).head()


best_unsupervised_model = grid_search.best_estimator_


pred = best_unsupervised_model.transform(X_val)
y_pred = np.argmax(pred, axis=1)

topic_mapping = {}

# n_components = 5
for topic in range(5):
    topic_docs_labels = y_val[y_pred == topic]
    most_common_label = Counter(topic_docs_labels).most_common(1)[0][0]
    topic_mapping[topic] = most_common_label

print("Topic Mapping:", topic_mapping)

y_pred = [topic_mapping[topic] for topic in y_pred]
nmf_acc = np.mean(y_val == y_pred)
print("Accuracy Validation Dataset:", nmf_acc)


cm = confusion_matrix(y_val, y_pred, labels=list(topic_mapping.values()))

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(topic_mapping.values()))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix for NMF")
plt.show()


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')), 
    ('model', SVC(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'model': [SVC(random_state=42)],
    'tfidf__max_df': [0.8, 1.0], 
    'tfidf__min_df': [1, 5],  
    'tfidf__ngram_range': [(1, 1), (1, 3)], 
    'model__C': [0.1, 1, 10], 
    'model__kernel': ['linear']
},
{
    'model': [LogisticRegression(random_state=42)],
    'tfidf__max_df': [0.8, 1.0], 
    'tfidf__min_df': [1, 5],  
    'tfidf__ngram_range': [(1, 1), (1, 3)], 
    'model__penalty': ['l2'],
    'model__C': [0.1, 1, 10],
    'model__solver': ['lbfgs', 'liblinear'],
    'model__max_iter': [500]
}

# Store results for comparison
results = []

# Define percentages of training data to use
percentages = [0.1, 0.2, 0.5, 1.0]

for pct in percentages:
    print(f"Training with {int(pct * 100)}% of the data...")
    
    # Create a subset of the training data
    if pct == 1.0:
        X_subset, y_subset = X_train, y_train
    else:
        X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=pct, stratify=y_train, random_state=42)
    
    # Grid search on the subset
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,  # 3-fold cross-validation
        verbose=1
    )
    grid_search.fit(X_subset, y_subset)
    
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['percentage'] = pct  # Add a column for percentage
    
    results.append(cv_results_df)

# Combine all results into a single DataFrame
final_results_df = pd.concat(results, ignore_index=True)

# Display or save the final results DataFrame
final_results_df.head()


final_results_df_10_pct = final_results_df[final_results_df['percentage'] == 0.1]
final_results_df_10_pct.sort_values(by='mean_test_score', ascending=False).head()


final_results_df_20_pct = final_results_df[final_results_df['percentage'] == 0.2]
final_results_df_20_pct.sort_values(by='mean_test_score', ascending=False).head()


final_results_df_50_pct = final_results_df[final_results_df['percentage'] == 0.5]
final_results_df_50_pct.sort_values(by='mean_test_score', ascending=False).head()


final_results_df_100_pct = final_results_df[final_results_df['percentage'] == 1.0]
final_results_df_100_pct.sort_values(by='mean_test_score', ascending=False).head()


best_supervised_model = grid_search.best_estimator_


y_pred = best_supervised_model.predict(X_val)
acc = np.mean(y_val == y_pred)
print("Accuracy Validation Dataset:", acc)


X_test = test_df['Text_pp']


pred_test = best_unsupervised_model.transform(X_test)
y_pred_test = np.argmax(pred_test, axis=1)

y_pred = [topic_mapping[topic] for topic in y_pred_test]

submission = pd.DataFrame({
    'ArticleId': test_df['ArticleId'],
    'Category': y_pred
})
submission.to_csv('submission_nmf.csv', index=False)


y_pred_test = best_supervised_model.predict(X_test)

submission = pd.DataFrame({
    'ArticleId': test_df['ArticleId'],
    'Category': y_pred_test
})
submission.to_csv('submission_svc.csv', index=False)

