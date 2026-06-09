# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
%matplotlib inline
import  matplotlib.pyplot as plt
plt.style.use("ggplot")

from collections import Counter
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords 
nltk.download('stopwords')

#Topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Model
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from scipy.stats import uniform
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#train data
train_df = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Train.csv")
#Test data
test_df = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')

train_df.head()


def summary_info(df):
    print("INFO")
    display(df.info())
    print("Summary information")
    display(df.describe())
    print("Null Values")
    null_values = df.isnull().sum()/df.shape[0]
    display(null_values[null_values >0])
    print('Duplicate Articles')
    display(df.duplicated().sum())

summary_info(train_df)


order_values = train_df['Category'].value_counts().index
sns.countplot(train_df, x='Category', order = order_values);


#compute percentage distribution of category values 
cat_count = (train_df['Category'].value_counts()/train_df.shape[0]*100).round(2)
cat_per = cat_count[order_values]
plt.pie(cat_per, labels=cat_per.index, autopct='%1.2f%%', startangle=90);
plt.title("Category Percentage Distribution");


# check word count in text
train_df['word_count'] = train_df['Text'].apply(lambda x: len(str(x).split()))
train_df['char_count'] = train_df['Text'].apply(lambda x: len(str(x)))

#Histogram of word counts
plt.figure(figsize = (10,5));
sns.histplot(train_df['word_count'],bins=10,kde=True);
plt.title('Histogram of Word Counts');
plt.xlabel('Number of Charecters');
plt.ylabel('Frequency');


#Histogram of Character Counts
plt.figure(figsize=(10,5));
sns.histplot(train_df['char_count'], bins=10,kde=True);
plt.title('Histogram of Character Counts');
plt.xlabel('Number of Characters');
plt.ylabel('Frequency');



#Average and mean lengths per category
length_stats = train_df.groupby('Category').agg(
    avg_words = ('word_count', 'mean'),
    median_words = ("word_count", 'median'),
    avg_chars = ('char_count', 'mean'),
    median_chars = ('char_count', 'median')
).round(2)

length_stats


stop_words = set(stopwords.words('english'))

#function to tokenize and remove stopwords
def tokenize_text(text):
    words = str(text).lower().split()
    words = [w.strip('.,!?;:()[]-"\'%£') for w in words]
    return [w for w in words  if w not in stop_words and w != '']
    
#Most common words 
all_words = []
train_df['Text'].apply(lambda x: all_words.extend(tokenize_text(x)))
overall_counts = Counter(all_words)
top10_words = dict(overall_counts.most_common(10))
for word, count in top10_words.items():
    # print(f'{word}: {count}')
    plt.bar(word,count)
    plt.title("Top 10 Most Common Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45);

#Most common words and Vocabulary per category
vocab_per_category = {}
category_counts = {}
for cat, group in train_df.groupby('Category'):
    words_cat = []
    group['Text'].apply(lambda x: words_cat.extend(tokenize_text(x)))
    category_counts[cat] = Counter(words_cat).most_common(10)
    vocab_per_category[cat] = len(set(words_cat))
    
vocab = pd.DataFrame({
    'Cetegory': list(vocab_per_category.keys()),
    'Vocabary_size': list(vocab_per_category.values())
})




#Most common words by category
fig, axes = plt.subplots(3,2,figsize= (15,15))
axes = axes.flatten()

for ax, (cat,counts) in zip(axes, category_counts.items()):
    words,freq = zip(*counts)
    ax.bar(words, freq)
    ax.set_title(f'Top 10 Words in {cat}')
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation= 45)
# Remove last extra subplot
for i in range(len(category_counts),len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()



vocab


#convert articles into words
vectorizer = CountVectorizer(max_df = 0.95, min_df =2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(train_df['Text'])


#Apply LDA with n_component=5 representing the number of categories in the dataset
lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(doc_term_matrix)


#Show top words per topic
words = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top_words = [words[j] for j in topic.argsort()[-10:]]
    print(f"Topic {i+1}: {', '.join(top_words)}")


#create a heatmap showing how LDA topics align with actual categories.

#1. Get topic distributions for each document
topic_dist =  lda.transform(doc_term_matrix)

#2. Assign a dominant topic per document
train_df['dominat_topic'] = topic_dist.argmax(axis=1)

# 3. Create a cross-tab: topics vs categories
topic_category_matrix = pd.crosstab(train_df['dominat_topic'],train_df['Category'])

# 4. Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(topic_category_matrix, annot=True, fmt='d');
plt.title('Heatmap: LDA Topics Vs Actual Categories');
plt.xlabel('Category');
plt.ylabel('Topic');


Pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    TruncatedSVD(random_state=42),
    LogisticRegression(max_iter=2000, random_state=42)
)

param_distributions = {
    'tfidfvectorizer__max_features': [5000, 10000, 15000],
    'tfidfvectorizer__ngram_range': [(1,1), (1,2)],
    'tfidfvectorizer__min_df': [2, 3, 5],
    'tfidfvectorizer__max_df': [0.85, 0.9, 0.95],
    'truncatedsvd__n_components': [100, 150, 200],
    'logisticregression__C': uniform(0.1, 10),
}

Random_search = RandomizedSearchCV(
    Pipeline,
    param_distributions=param_distributions,
    n_iter=20,          # number of random combinations to try
    scoring='accuracy',
    cv=5,               # 5-fold cross-validation
    verbose=0,
    n_jobs=-1,
    random_state=42
)

Random_search.fit(train_df['Text'], train_df['Category'])
best_params = Random_search.best_params_
print("Best parameters:", best_params)
print("Best CV accuracy:", Random_search.best_score_)



def display_report(model):
    """
    Parameters
    ----------
    model : sklearn estimator (Pipeline or compatible classifier)
        A trained model or pipeline that implements the `.predict()` method.
    
    train_df : pandas.DataFrame
        DataFrame containing at least two columns:
            - 'Text': str, input text features used for prediction.
            - 'Category': str or categorical, ground truth labels.

    Returns
    -------
    None
        Displays the classification report and shows a confusion matrix plot.

    """
    pred = model.predict(train_df['Text'])
    print(classification_report(train_df['Category'],pred))
    cm = confusion_matrix(train_df['Category'].astype(str),  pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=train_df['Category'].astype(str).unique())
    disp.plot(xticks_rotation=45,cmap='Blues')
    plt.show()


Pipeline = make_pipeline(
    TfidfVectorizer(
        stop_words='english',
        max_features=best_params['tfidfvectorizer__max_features'],
        ngram_range=best_params['tfidfvectorizer__ngram_range'],
        min_df=best_params['tfidfvectorizer__min_df'],
        max_df=best_params['tfidfvectorizer__max_df']
    ),
    TruncatedSVD(
        n_components=best_params['truncatedsvd__n_components'],
        random_state=42
    ),
    LogisticRegression(
        max_iter=2000,
        C=best_params['logisticregression__C'],
        random_state=42
    )
)
    # TfidfVectorizer(stop_words='english', max_features=5000),
    # TruncatedSVD(n_components=100,random_state=0),
    # LogisticRegression(max_iter=1000)

Pipeline.fit(train_df['Text'], train_df['Category'])
display_report(Pipeline)



nmf = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),
    NMF(n_components=100, random_state=0, init='nndsvda', max_iter=200),
    LogisticRegression(max_iter=1000)
)
nmf.fit(train_df['Text'], train_df['Category'])

#Model predict
# nmf_preds = nmf.predict(train_df['Text'])
# print(classification_report(train_df['Category'],nmf_preds))
display_report(nmf)
 


xgboost = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),
    XGBClassifier(
        random_state=0, 
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax', # multi-class classification
        eval_metric='mlogloss',
        use_label_encoder=False
    )
)
y_train = train_df['Category']
le = LabelEncoder()
y_train = le.fit_transform(y_train)
# Fit pipeline on training data
xgboost.fit(train_df['Text'], y_train)

preds_numeric = xgboost.predict(train_df['Text'])

# Convert back to original category names
preds_labels = le.inverse_transform(preds_numeric)
print(classification_report(train_df['Category'], preds_labels))
cm = confusion_matrix(train_df['Category'].astype(str),  preds_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_df['Category'].astype(str).unique())
disp.plot(xticks_rotation=45,cmap='Blues')
plt.show()


preds_num = Pipeline.predict(test_df['Text'])
# xgb_preds_labels = le.inverse_transform(preds_num)
submission = pd.DataFrame({
    'ArticleId' : test_df['ArticleId'],
    'Category': preds_num
})
submission.to_csv('submission.csv', index=False)




