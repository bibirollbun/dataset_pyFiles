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


df = pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv')


df.head()


df.shape


# removing the unnecessery column
df.drop(columns=['id'],inplace=True)


df.head()


df['sentiment'].value_counts().plot(kind='bar')


from nltk.tokenize import word_tokenize, sent_tokenize


df['length_of_text'] = df['text'].apply(lambda x:len(x))
df['word_len'] = df['text'].apply(lambda x:len(word_tokenize(x)))
df['sent_len'] = df['text'].apply(lambda x:len(sent_tokenize(x)))


plt.figure(figsize=(8, 5))  # Adjust figure size for clarity

sentiments = ['positive', 'negative', 'neutral']
colors = ['g', 'r', 'b']  # Green, Red, Blue for differentiation

for sentiment, color in zip(sentiments, colors):
    sns.kdeplot(df[df['sentiment'] == sentiment]['length_of_text'], 
                label=sentiment, color=color, shade=True)

plt.xlabel('Length of Text')
plt.ylabel('Density')
plt.title('KDE Plot of Text Length by Sentiment')
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))  # Adjust figure size for clarity

sentiments = ['positive', 'negative', 'neutral']
colors = ['g', 'r', 'b']  # Green, Red, Blue for differentiation

for sentiment, color in zip(sentiments, colors):
    sns.kdeplot(df[df['sentiment'] == sentiment]['word_len'], 
                label=sentiment, color=color, shade=True)

plt.xlabel('Length of Text')
plt.ylabel('Density')
plt.title('KDE Plot of Text Length by Sentiment')
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))  # Adjust figure size for clarity

sentiments = ['positive', 'negative', 'neutral']
colors = ['g', 'r', 'b']  # Green, Red, Blue for differentiation

for sentiment, color in zip(sentiments, colors):
    sns.kdeplot(df[df['sentiment'] == sentiment]['sent_len'], 
                label=sentiment, color=color, shade=True)

plt.xlabel('Length of Text')
plt.ylabel('Density')
plt.title('KDE Plot of Text Length by Sentiment')
plt.legend()
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
arr = cv.fit_transform(df['text']).toarray()
x = pd.DataFrame(arr,index = df['text'].index)
df_x = pd.concat([x,df[['length_of_text','word_len','sent_len']]],axis=1)


from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
df['sentiment'] = encode.fit_transform(df['sentiment'])


x = df_x
y = df['sentiment']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)
x_train = np.array(x_train)
x_test = np.array(x_test)


from sklearn.metrics import precision_score,accuracy_score


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print('precision (macro) : ', precision_score(y_test, y_pred, average='macro'))
print('Accuracy :', accuracy_score(y_test, y_pred))


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_pred = mnb.predict(x_test)
print('precision (macro) : ', precision_score(y_test, y_pred, average='macro'))
print('Accuracy :', accuracy_score(y_test, y_pred))


# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

# Training and Evaluating Models
results = []
for name, model in tqdm(models.items(), desc="Training Models"):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Evaluating performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    results.append([name, accuracy, precision, recall, f1])


results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print(results_df)














