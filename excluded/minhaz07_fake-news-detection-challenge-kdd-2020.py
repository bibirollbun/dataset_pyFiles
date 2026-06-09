# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/fakenewskdd2020/train.csv',  sep='\t', encoding='utf-8')


df.head()


df.shape


df.isnull().sum()


df['label'].value_counts()


df['label'].replace('label',1,inplace=True)


df['label'] = df['label'].astype(int)


df['label'].value_counts().plot(kind='bar')


import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from textblob import TextBlob
import re


def transform(text):
    # lowercasing
    text = text.lower()
    # removing numbers
    text = re.sub(r'\d+','',text)
    # punctuation remove
    for p in string.punctuation:
        text = text.replace(p,'')
    # stopword remove and stemmimg
    new_text = []
    for word in text.split():
        if word not in stopwords.words('english'):
            new_text.append(ps.stem(word))
    text = ' '.join(new_text)
    # spelling correction
    blob = TextBlob(text)
    return text
    


df['clean_text'] = df['text'].apply(transform)


df['len'] = df['clean_text'].apply(lambda x: len(x))
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag  
from nltk.tokenize import word_tokenize


from nltk.tokenize import word_tokenize
from nltk import pos_tag

def pos_count(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    unique_pos_tags = set(tag for word, tag in tagged_words)
    return len(unique_pos_tags)


def simple_transform(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p,'')
    text = re.sub(r'\d+','',text)
    return text


df['text_for_pos'] = df['text'].apply(simple_transform)


df['pos_count'] = df['text_for_pos'].apply(pos_count)


plt.figure(figsize=(12,6))
sns.distplot(df[df['label'] == 0]['len'],hist=False)
sns.distplot(df[df['label'] == 1]['len'],hist=False)


plt.figure(figsize=(12,6))
sns.distplot(df[df['label'] == 0]['word_count'],hist=False)
sns.distplot(df[df['label'] == 1]['word_count'],hist=False)


plt.figure(figsize=(12,6))
sns.distplot(df[df['label'] == 0]['pos_count'],hist=False)
sns.distplot(df[df['label'] == 1]['pos_count'],hist=False)


df[['word_count','len','label']].corr()


df.drop(columns=['word_count','len'],inplace=True)


df[['pos_count','label']].corr()


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)


 arr = tfidf.fit_transform(df['clean_text']).toarray()


final = pd.DataFrame(arr,index=df['clean_text'].index)
final = pd.concat([final,df['pos_count']],axis=1)


x = final


y = df['label']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)
x_train = np.array(x_train)
x_test = np.array(x_test)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,precision_score


models = {
    'lr': LogisticRegression(),
    'dtc': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(n_estimators=50, random_state=44),
    'etc': ExtraTreesClassifier(n_estimators=50, random_state=44),
    'xgb': XGBClassifier(n_estimators=50, random_state=44),
    'S': SVC(),
    'mnb': MultinomialNB(),
    'bnb': BernoulliNB(),
    'gnb': GaussianNB()
}


def train_model(clf,x_train,x_test,y_train,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    pre = precision_score(y_test,y_pred)
    return acc,pre


accuracy = []
precision = []
algorithms = []
for name,algorithm in models.items():
    current_accuracy,current_precision = train_model(algorithm,x_train,x_test,y_train,y_test)
    print('for',name)
    print('Accuracy :',current_accuracy)
    print('Precision :',current_precision)
    algorithms.append(name)
    accuracy.append(current_accuracy)
    precision.append(current_precision)


result = pd.DataFrame({'Algorithm': algorithms,'Accuracy': accuracy,'Precision': precision})


result.sort_values(by='Precision',ascending=False)


from sklearn.ensemble import VotingClassifier


rn = RandomForestClassifier(n_estimators=100,random_state=44)
etc = ExtraTreesClassifier(n_estimators=100,random_state=44)
lr = LogisticRegression()


voting = VotingClassifier(estimators=[('rn',rn),('etc',etc),('lr',lr)])


voting.fit(x_train,y_train)


y_pred = voting.predict(x_test)


print(accuracy_score(y_test,y_pred))
print(precision_score(y_test,y_pred))


from sklearn.ensemble import StackingClassifier
rn = RandomForestClassifier(n_estimators=100, random_state=44)
etc = ExtraTreesClassifier(n_estimators=100, random_state=44)
lr = LogisticRegression()

stacking_model = StackingClassifier(
    estimators=[('rf', rn), ('etc', etc), ('lr', lr)],
    final_estimator=LogisticRegression()
)


# Fit the stacking model on the training data
stacking_model.fit(x_train, y_train)

# Make predictions
y_pred = stacking_model.predict(x_test)

# Calculate accuracy and precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Change to 'micro' or 'macro' if needed

# Print the results
print(f"Stacking Model Accuracy: {accuracy}")
print(f"Stacking Model Precision: {precision}")

