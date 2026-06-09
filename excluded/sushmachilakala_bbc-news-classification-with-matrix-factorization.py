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


import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


train_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')
test_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')


print("Training Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)


print(train_df.info())


print(test_df.info())


print(train_df.head())


print(test_df.head())


print("\nMissing Values:\n", train_df.isnull().sum())


nltk.download('punkt')
nltk.download('stopwords')


np.random.seed(42)


# Data Cleaning

# Define text cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Remove stopwords, punctuation, and non-alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

# Apply cleaning to training and test data
train_df['clean_text'] = train_df['Text'].apply(clean_text)
test_df['clean_text'] = test_df['Text'].apply(clean_text)

# Verify cleaning
print("\nSample Original Text:\n", train_df['Text'].iloc[0][:200])
print("\nSample Cleaned Text:\n", train_df['clean_text'].iloc[0][:200])


print("\nCategory Distribution:\n", train_df['Category'].value_counts())


#  Visualize Category Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Category', data=train_df)
plt.title('Distribution of News Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Text Length Analysis
train_df['text_length'] = train_df['clean_text'].apply(lambda x: len(word_tokenize(x)))
print("\nText Length Statistics (Cleaned):\n", train_df['text_length'].describe())


# Histogram of text lengths
plt.figure(figsize=(8, 5))
sns.histplot(train_df['text_length'], bins=30, kde=True)
plt.title('Distribution of Article Lengths (Cleaned Text)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()


# Word Frequency Analysis
all_words = ' '.join(train_df['clean_text']).split()
word_freq = Counter(all_words)
common_words = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Count'])


# Bar plot of top words
plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Word', data=common_words)
plt.title('Top 20 Most Frequent Words (Cleaned Text)')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()


#  Word Cloud by Category
for category in train_df['Category'].unique():
    text = ' '.join(train_df[train_df['Category'] == category]['clean_text'])
    wordcloud = WordCloud(stopwords=stop_words, max_words=50, background_color='white').generate(text)
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {category}')
    plt.show()


train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Category'])
y_train = train_data['Category']
y_val = val_data['Category']
categories = train_df['Category'].unique()


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_data['clean_text'])
X_val = vectorizer.transform(val_data['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

# Inspect TF-IDF matrix
print("\nTF-IDF Matrix Shape (Training):", X_train.shape)
print("Sample Feature Names:", vectorizer.get_feature_names_out()[:10])


# Initialize and train NMF
n_components = 5  # Number of topics = number of categories
nmf = NMF(n_components=n_components, random_state=42)
W_train = nmf.fit_transform(X_train) # Document-topic matrix for training
W_val = nmf.transform(X_val)
W_test = nmf.transform(X_test)  # Document-topic matrix for test

# Category mapping for NMF
train_preds_indices = np.argmax(W_train, axis=1)
category_map = {}
for topic_idx in range(5):
    topic_docs = np.where(train_preds_indices == topic_idx)[0]
    if len(topic_docs) > 0:
        category_map[topic_idx] = y_train.iloc[topic_docs].mode()[0]
    else:
        category_map[topic_idx] = categories[topic_idx % len(categories)]
print("\nCategory Mapping (Topic -> Category):", category_map)

# Convert topic indices to category predictions
train_preds_nmf = [category_map[idx] for idx in train_preds_indices]
val_preds_nmf = [category_map[idx] for idx in np.argmax(W_val, axis=1)]
test_preds_nmf = [category_map[idx] for idx in np.argmax(W_test, axis=1)]

train_accuracy_nmf = accuracy_score(y_train, train_preds_nmf)
val_accuracy_nmf = accuracy_score(y_val, val_preds_nmf)
train_f1_nmf = f1_score(y_train, train_preds_nmf, average='weighted')
val_f1_nmf = f1_score(y_val, val_preds_nmf, average='weighted')




print("NMF Results:")
print(f"Training Accuracy: {train_accuracy_nmf:.4f}")
print(f"Validation Accuracy: {val_accuracy_nmf:.4f}")
print(f"Training F1-Score: {train_f1_nmf:.4f}")
print(f"Validation F1-Score: {val_f1_nmf:.4f}")


# Confusion matrix
cm_nmf = confusion_matrix(y_train, train_preds_nmf, labels=categories)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nmf, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('NMF Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Predict test set
test_preds_indices = np.argmax(W_test, axis=1)
test_preds_nmf = [category_map[idx] for idx in test_preds_indices]



# Save test predictions for Kaggle
submission = pd.DataFrame({'ArticleId': test_df['ArticleId'], 'Category': test_preds_nmf})
submission.to_csv('submission.csv', index=False)
print("NMF submission file created: submission.csv")


train_data, val_data = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['Category'])
y_train = train_data['Category']
y_val = val_data['Category']
categories = train_df['Category'].unique()

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=3500, stop_words='english')
X_train = vectorizer.fit_transform(train_data['clean_text'])
X_val = vectorizer.transform(val_data['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

# Hyperparameter tuning
n_components = 5  # Number of topics = number of categories
nmf = NMF(n_components=n_components, random_state=42)
W_train = nmf.fit_transform(X_train) # Document-topic matrix for training
W_val = nmf.transform(X_val)
W_test = nmf.transform(X_test)  # Document-topic matrix for test

# Category mapping for NMF
train_preds_indices = np.argmax(W_train, axis=1)
category_map = {}
for topic_idx in range(5):
    topic_docs = np.where(train_preds_indices == topic_idx)[0]
    if len(topic_docs) > 0:
        category_map[topic_idx] = y_train.iloc[topic_docs].mode()[0]
    else:
        category_map[topic_idx] = categories[topic_idx % len(categories)]
print("\nCategory Mapping (Topic -> Category):", category_map)

# Convert topic indices to category predictions
train_preds_nmf2 = [category_map[idx] for idx in train_preds_indices]
val_preds_nmf2 = [category_map[idx] for idx in np.argmax(W_val, axis=1)]
test_preds_nmf2 = [category_map[idx] for idx in np.argmax(W_test, axis=1)]

train_accuracy_nmf2 = accuracy_score(y_train, train_preds_nmf2)
val_accuracy_nmf2 = accuracy_score(y_val, val_preds_nmf2)
train_f1_nmf2 = f1_score(y_train, train_preds_nmf2, average='weighted')
val_f1_nmf2 = f1_score(y_val, val_preds_nmf2, average='weighted')      


print("NMF Results:")
print(f"Training Accuracy: {train_accuracy_nmf2:.4f}")
print(f"Validation Accuracy: {val_accuracy_nmf2:.4f}")
print(f"Training F1-Score: {train_f1_nmf2:.4f}")
print(f"Validation F1-Score: {val_f1_nmf2:.4f}")


# Predict test set
test_preds_indices = np.argmax(W_test, axis=1)
test_preds_nmf2 = [category_map[idx] for idx in test_preds_indices]


submission = pd.DataFrame({'ArticleId': test_df['ArticleId'], 'Category': test_preds_nmf2})
submission.to_csv('submission.csv', index=False)
print("NMF submission file created: submission.csv")


lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

# Predict.
train_preds_lr = lr.predict(X_train)
val_preds_lr = lr.predict(X_val)
test_preds_lr = lr.predict(X_test)


train_accuracy = accuracy_score(y_train, train_preds_lr)
val_accuracy = accuracy_score(y_val, val_preds_lr)
train_f1 = f1_score(y_train, train_preds_lr, average='weighted')
val_f1 = f1_score(y_val, val_preds_lr, average='weighted')
print("\nLogistic Regression Results:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Training F1-Score: {train_f1:.4f}")
print(f"Validation F1-Score: {val_f1:.4f}")


# Save Kaggle submission
submission_lr = pd.DataFrame({'ArticleId': test_df['ArticleId'], 'Category': test_preds_lr})
submission_lr.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")


data_sizes = [
    # 0.1,
               # 0.2,
     0.5,
    # 1.0
             ]
results = []

# Evaluation function for each model.
def evaluate_model(model_name, y_train_pred, y_val_pred, y_train_true, y_val_true, fraction):
    train_acc = accuracy_score(y_train_true, y_train_pred)
    val_acc = accuracy_score(y_val_true, y_val_pred)
    train_f1 = f1_score(y_train_true, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val_true, y_val_pred, average='weighted')
    return {
        'model': model_name,
        'fraction': fraction,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1
    }


# Loop over data sizes
for fraction in data_sizes:
    print(f"\nTesting with {fraction*100}% of training data (~{int(len(train_data)*fraction)} articles)")
    if fraction < 1.0:
        other_subset, train_subset = train_test_split(train_df, test_size=fraction, random_state=42, stratify=train_df['Category'])
        train_subset = train_subset.reset_index(drop=True)
    else:
        train_subset = train_data
    
    y_train_subset = train_subset['Category']
    X_train = vectorizer.fit_transform(train_subset['clean_text'])
    X_val = vectorizer.transform(val_data['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train_subset length:", len(y_train_subset))
    assert X_train.shape[0] == len(y_train_subset), "X_train and y_train_subset mismatch"

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train_subset)
    train_preds_lr = lr.predict(X_train)
    val_preds_lr = lr.predict(X_val)
    test_preds_lr = lr.predict(X_test)
    results.append(evaluate_model('Logistic Regression', train_preds_lr, val_preds_lr, y_train_subset, y_val, fraction))
    print(f"LR - Train Acc: {results[-1]['train_accuracy']:.4f}, Val Acc: {results[-1]['val_accuracy']:.4f}, "
          f"Train F1: {results[-1]['train_f1']:.4f}, Val F1: {results[-1]['val_f1']:.4f}")

    # NMF
    n_components = 5
    nmf = NMF(n_components=n_components, random_state=42)
    W_train = nmf.fit_transform(X_train)
    W_val = nmf.transform(X_val)
    W_test = nmf.transform(X_test)
    train_preds_indices = np.argmax(W_train, axis=1)
    category_map = {}
    for topic_idx in range(n_components):
        topic_docs = np.where(train_preds_indices == topic_idx)[0]
        if len(topic_docs) > 0 and topic_docs.max() < len(y_train_subset):
            category_map[topic_idx] = y_train_subset.iloc[topic_docs].mode()[0]
        else:
            category_map[topic_idx] = categories[topic_idx % len(categories)]
    train_preds_nmf = [category_map[idx] for idx in train_preds_indices]
    val_preds_nmf = [category_map[idx] for idx in np.argmax(W_val, axis=1)]
    test_preds_nmf = [category_map[idx] for idx in np.argmax(W_test, axis=1)]
    results.append(evaluate_model('NMF', train_preds_nmf, val_preds_nmf, y_train_subset, y_val, fraction))
    print(f"NMF - Train Acc: {results[-1]['train_accuracy']:.4f}, Val Acc: {results[-1]['val_accuracy']:.4f}, "
          f"Train F1: {results[-1]['train_f1']:.4f}, Val F1: {results[-1]['val_f1']:.4f}")
    
    submission_lr = pd.DataFrame({'ArticleId': test_df['ArticleId'], 'Category': test_preds_lr})
    submission_lr.to_csv('submission.csv', index=False)
    submission_nmf = pd.DataFrame({'ArticleId': test_df['ArticleId'], 'Category': test_preds_nmf})
    submission_nmf.to_csv('nmsubmission.csv', index=False)

print("\nResults Summary:")
for res in results:
    print(f"Model: {res['model']}, Fraction: {res['fraction']*100}%, "
          f"Train Acc: {res['train_accuracy']:.4f}, Val Acc: {res['val_accuracy']:.4f}, "
          f"Train F1: {res['train_f1']:.4f}, Val F1: {res['val_f1']:.4f}")

