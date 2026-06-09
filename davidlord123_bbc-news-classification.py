# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv(dirname + "/BBC News Train.csv")
df_test = pd.read_csv(dirname + "/BBC News Test.csv")


print("Number of articles in the training set:",len(df_train),"\n", df_train.head(), "\n")

print("Number of articles in the test set:",len(df_test),"\n", df_test.head(), "\n")


print("train: \n")
print(df_train.info())
print("test: \n")
print(df_test.info())


#Rename columns to lowercase to standarize
df_train.columns = df_train.columns.str.lower()
df_test.columns = df_test.columns.str.lower()
# Create a new column for word count
df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
df_test['word_count'] = df_test['text'].apply(lambda x: len(str(x).split()))


# Histogram of article categories
ax = sns.countplot(data=df_train, x='category', palette='tab10')

# Annotate the correct bar height on top of each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom'
    )

plt.title('Distribution of Article Categories')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Count the duplicated values based on both 'Text' and 'Category'. If the counts are the same as above,those duplicates have consistent 'Category'
duplicated_textcat_count = df_train.duplicated(subset=['text', 'category']).sum()

# Print the counts
print(f'There are {duplicated_textcat_count} rows with the same texts and categories in the training data.')

# I am curious to see how those duplicates look like. Let's just look at a few.
duplicates = df_train[df_train.duplicated(subset=['text'], keep=False)].sort_values(by='text')
print( duplicates.head(6) )


# Remove duplicates
df_train = df_train.drop_duplicates(subset=['text'])

# Check train data counts. Originally training data has 1490 rows and 50 rows should be dropped. So, I expect 1440 rows left.
print(f'Samples in training data after moving duplicates: {df_train.shape[0]}')


# Histogram of word counts
plt.figure(figsize=(10, 5))
bin_width = 100  # adjust as needed based on your min/max values
bins = range(0, df_train['word_count'].max() + bin_width, bin_width)
sns.histplot(df_train['word_count'], bins=bins, kde=False, color='orange')
plt.title('Training Set Distribution of Article Lengths (Word Count)')
plt.xlabel('Number of Words')
plt.ylabel('Number of Articles')
plt.figure(figsize=(10, 5))
sns.histplot(df_test['word_count'], bins=bins, kde=False, color='blue')
plt.title('Test Set Distribution of Article Lengths (Word Count)')
plt.xlabel('Number of Words')
plt.ylabel('Number of Articles')
plt.tight_layout()
plt.show()
# min_words = df_train['word_count'].min()
# max_words = df_train['word_count'].max()
# median_words = df_train['word_count'].median()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_train, x='category', y='word_count', palette='pastel')
plt.title('Distribution of Article Word Counts by Category')
plt.xlabel('Category')
plt.ylabel('Word Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Group by category and describe the word_count distribution
summary_stats = df_train.groupby('category')['word_count'].describe()

# Print the key percentiles used in the boxplot
print(summary_stats[['min', '25%', '50%', '75%', 'max']])


# function for text preprocessing
def text_preprocessing(df):
    """
    This function does in place replacement of data so it won't return anything
    """
    # convert to lower cases
    df['text']=df['text'].str.lower()

    # remove punctuation
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

    # remove numbers
    df['text'] = df['text'].str.replace(r'\d+', '', regex=True)
    
  



def plot_top_words_per_category(
        df: pd.DataFrame,
        text_col: str = 'text',
        category_col: str = 'category',
        top_n: int = 20,
        n_categories: int = 5,
        stop_words: set | None = None
    ):
    """
    Plot top-N word-frequency histograms for the first `n_categories`
    (ranked by number of rows) in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing text and category columns.
    text_col : str
        Name of the column with raw text.
    category_col : str
        Name of the column with category labels.
    top_n : int
        How many words to show per histogram.
    n_categories : int
        How many distinct categories to plot (default 5).
    stop_words : set | None
        Optional custom stop-word list.  If None, NLTK English stopwords are used.
    """
    # fallback stop-word list
    if stop_words is None:
        stop_words = set(stopwords.words('english'))

    # pick the top-occurring categories
    cat_order = df[category_col].value_counts().index[:n_categories]

    # create one subplot per category
    fig, axes = plt.subplots(
        n_categories, 1,
        figsize=(12, 4 * n_categories),
        constrained_layout=True
    )

    if n_categories == 1:   # make axes iterable if only one subplot
        axes = [axes]

    for ax, cat in zip(axes, cat_order):
        # collect text rows in this category
        texts = df.loc[df[category_col] == cat, text_col].dropna()

        # tokenize, lowercase, filter stopwords / non-alpha
        all_words = []
        for doc in texts:
            tokens = word_tokenize(str(doc).lower())
            all_words.extend(
                w for w in tokens
                if w.isalpha() and w not in stop_words
            )

        # get top-N frequencies
        freq = Counter(all_words).most_common(top_n)
        if not freq:  # skip empty category
            continue
        words, counts = zip(*freq)

        # bar plot
        ax.bar(words, counts)
        ax.set_title(f'{cat}: Top {top_n} Words')
        ax.set_ylabel('Freq')
        ax.tick_params(axis='x', rotation=45)

    plt.show()


#Useful to preprocess the text and remove all numbers, punctuation and standarize the text. 

# take a look at text 0
df_train_copy = df_train.copy()
print('Training Set top 15 words before preprocessing: \n')
plot_top_words_per_category(
    df_train_copy,          # your DataFrame
    text_col='text',   # column with text
    category_col='category',
    top_n=15,          # words per plot
    n_categories=5     # number of categories to show
)
text_preprocessing(df_train_copy)
print('\n Training Set top 15 words before preprocessing: \n')
plot_top_words_per_category(
    df_train_copy,          # your DataFrame
    text_col='text',   # column with text
    category_col='category',
    top_n=15,          # words per plot
    n_categories=5     # number of categories to show
)





#Some words are pretty common but not relevant to the category
custom_stopwords = {'said', 'would', 'one', 'could', 'mr', 'also'}
stop_words = set(stopwords.words('english')).union(custom_stopwords)
print('\n Training Set top 15 words with most common not relevant words removed: \n')
plot_top_words_per_category(
    df_train_copy,          # your DataFrame
    text_col='text',   # column with text
    category_col='category',
    top_n=15,          # words per plot
    n_categories=5  ,   # number of categories to show
    stop_words=stop_words
)



# Label encode categories
df_train = df_train_copy
le = LabelEncoder()
y_train_enc = le.fit_transform(df_train['category'])


# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nmf', NMF(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Set up parameter grid
param_grid = {
    'tfidf__max_features': [3000, 5000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nmf__n_components': [5, 10, 20],
    'clf__C': [0.1, 1, 10]
}

# Grid search
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit the grid search
grid.fit(df_train['text'], y_train_enc)


print("Best Params:", grid.best_params_)


# Predict on cleaned test data
y_test_pred_enc = grid.predict(df_test['text'])

# Convert back to original labels
y_test_pred = le.inverse_transform(y_test_pred_enc)

# Predict on the training set
y_train_pred_enc = grid.predict(df_train['text'])

# Compare to true labels
train_accuracy = accuracy_score(y_train_enc, y_train_pred_enc)
print(f"Training Accuracy: {train_accuracy:.4f}")
test_accuracy = accuracy_score(y_test_pred, y_test_pred_enc)
print(f"Test Accuracy: {train_accuracy:.4f}")


# Confusion matrix
cm = confusion_matrix(y_train_enc, y_train_pred_enc)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix: KMeans Clusters vs. True Categories")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


test_pred = pd.DataFrame(columns=['articleid', 'category'])
test_pred['articleid'] = df_test['articleid']
test_pred['category'] = y_test_pred
test_pred.to_csv("submission.csv", index=False)



import time
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
def benchmark_text_models(df, text_col='text', target_col='category',
                          test_size=0.2, random_state=42):
    """
    Split the data, fit several text-classification models, and
    return their accuracy and run-time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text and label columns.
    text_col : str
        Column name containing raw text.
    target_col : str
        Column name containing target labels.
    test_size : float
        Proportion of the data to hold out for validation.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    pd.DataFrame with columns [model, accuracy, seconds]
    """

    X_train, X_val, y_train, y_val = train_test_split(
        df[text_col], df[target_col],
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )

    #──────────────────────────────────────── pipelines
    pipelines = {
        "LogReg" : Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf',  LogisticRegression(max_iter=1000, multi_class='multinomial'))
        ]),
        "LinearSVC" : Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf',  LinearSVC())
        ]),
        "MultinomialNB" : Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf',  MultinomialNB())
        ]),
        # Random-forest prefers dense input → add SVD to compress TF-IDF
        "RF + SVD" : Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('svd',   TruncatedSVD(n_components=200, random_state=random_state)),
            ('clf',   RandomForestClassifier(n_estimators=300, random_state=random_state))
        ])
    }

    results = []
    for name, pipe in pipelines.items():
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        acc   = accuracy_score(y_val, preds)
        secs  = time.perf_counter() - start
        results.append((name, acc, secs))

    return pd.DataFrame(results, columns=['model', 'accuracy', 'seconds']
                        ).sort_values('accuracy', ascending=False).reset_index(drop=True)


scores = benchmark_text_models(df_train)   # df_train contains text + category
print(scores)


def evaluate_supervised_models(df, model, text_col='text', target_col='category', test_size=0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[target_col],
        test_size=test_size,
        stratify=df[target_col],
        random_state=42
    )

    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', model)
    ])

    # Train and predict
    pipeline.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc  = accuracy_score(y_test, pipeline.predict(X_test))

    return train_acc, test_acc


svc_train, svc_test = evaluate_supervised_models(df_train, LinearSVC())
nb_train, nb_test = evaluate_supervised_models(df_train, MultinomialNB())

print(f"LinearSVC - Train: {svc_train:.3f}, Test: {svc_test:.3f}")
print(f"Naive Bayes - Train: {nb_train:.3f}, Test: {nb_test:.3f}")


# Vectorize
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(df_train['text'])

# Fit NMF
nmf = NMF(n_components=5, random_state=42)
W = nmf.fit_transform(X_tfidf)  # Document-topic matrix
# Predict topic index for each document
topic_preds = W.argmax(axis=1)
# Map each topic to the most frequent true label
df_topics = pd.DataFrame({'topic': topic_preds, 'label': df_train['category']})
topic_to_label = df_topics.groupby('topic')['label'].agg(lambda x: x.value_counts().idxmax()).to_dict()
# Map each predicted topic to its inferred label
y_pred_unsup = [topic_to_label[t] for t in topic_preds]
unsup_accuracy = accuracy_score(df_train['category'], y_pred_unsup)
print(f'Unsupervised NMF Accuracy (train set): {unsup_accuracy:.4f}')


models =[LinearSVC(),MultinomialNB()]
for model in models:
    for frac in [0.05, 0.1, 0.2, 0.5]:
        df_subset = df_train.sample(frac=frac, random_state=42)
        acc_train, acc_test = evaluate_supervised_models(df_subset, model)
        print(f"{model} @ {int(frac*100)}% data → Train: {acc_train:.3f}, Test: {acc_test:.3f}")
    




