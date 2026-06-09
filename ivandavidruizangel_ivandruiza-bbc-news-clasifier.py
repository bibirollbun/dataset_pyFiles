# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering, KMeans
from itertools import permutations
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Importing Data
train = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')
test = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
y_true = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Sample Solution.csv')


#looking at data
train


train.info()



Idduplicates = train[train['ArticleId'].duplicated(keep=False)]
Text_duplicates = train[train['Text'].duplicated(keep=False)]
categories = train['Category'].unique()

category_counts = train['Category'].value_counts()

#print(f'id:{Idduplicates}')
print(f'text: {Text_duplicates}') 
#print(f'categories: {categories, category_counts}')
train = train.drop_duplicates(subset=['Text'], keep='first')

Text_duplicates = train[train['Text'].duplicated(keep=False)]
print(f'text duplicate after: {Text_duplicates}') 


#create dict mapping id
id2idx = dict(zip(train.ArticleId, (range(len(train)))))

# Count Vector (text to vector https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
corpus = train['Text']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)

# Extracting labels from the 'Category' column
labels = train['Category']


# NMF 
model = NMF(n_components=5, init='nndsvda', solver='mu', beta_loss='kullback-leibler', random_state=42)
W = model.fit_transform(X)
H = model.components_

W2 = model.fit_transform(X2)
H2 = model.components_


#Testing model
X_reconstructed = W.dot(H)
mse = mean_squared_error(X.toarray(), X_reconstructed)
print(f"Mean Squared Error 1: {mse}")

X_reconstructed2 = W2.dot(H2)
mse2 = mean_squared_error(X2.toarray(), X_reconstructed2)
print(f"Mean Squared Error 2: {mse2}")


Kmeans = KMeans(n_clusters=5, init='random').fit(W)
Kmeans2 = KMeans(n_clusters=5, init='random').fit(W2)
Agg = AgglomerativeClustering(n_clusters=5, linkage = 'complete', metric='cosine').fit(W)
Agg2 = AgglomerativeClustering(n_clusters=5, linkage = 'complete', metric='cosine').fit(W2)


def label_permute_compare(ytdf,yp,n=5): #(true, predicted)
    """
    ytdf: labels dataframe object
    yp: clustering label prediction output
    Returns permuted label order and accuracy. 
    Example output: (3, 4, 1, 2, 0), 0.74 
    """
    possible_labels = ['business', 'entertainment', 'politics', 'sport', 'tech']
    true_labels = ytdf
    label_permutations = permutations(possible_labels)
    model_labels = yp
    
    best_acc = 0
    best_labelorder = None

    for perm in label_permutations:
        mapped_labels = [perm[label] for label in model_labels] #number to name in the model_labels (all posible combinations since using this iterative aproach)

        acc = accuracy_score(true_labels, mapped_labels)

        #print(f"Accuracy of {perm}: {accuracy}")
        if acc > best_acc:
            best_acc = acc
            best_labelorder = perm

    #print(f"Best Accuracy: {best_acc}")
    #print(f"Best Label Mapping: {best_labelorder}")
        
    return best_labelorder, best_acc

print(label_permute_compare(labels, Kmeans.labels_))
print(label_permute_compare(labels, Kmeans2.labels_))
print(label_permute_compare(labels, Agg.labels_))
print(label_permute_compare(labels, Agg2.labels_))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train['Category']) 
X = W 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Hacemos predicciones sobre el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Evaluamos el modelo
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


