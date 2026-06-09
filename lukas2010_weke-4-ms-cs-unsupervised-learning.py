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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import re
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Train.csv")
test_data = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Test.csv")
sample_solutions = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Sample Solution.csv")

# Get general information on data
print(train_data.info())
print(train_data.head())


sns.histplot(train_data, x = "Category")


print("Number of available entrie:",  len(train_data["Text"]))
print("Unique text entries: ", train_data["Text"].nunique())

print("Removing duplicate entries")
train_data = train_data.drop_duplicates(subset = ["Text"])
print("Text entries after removal of duplicates: ",  len(train_data["Text"]))

print(train_data)


stemmer = PorterStemmer()

def preprocessing(textfile):
    return_file = list()
    
    for current_text in textfile:
        # convert all letters to lower case (seems to be initially the case but make sure it actually is)
        current_text = current_text.lower()
        
        # remove all numbers
        current_text = re.sub(r'\d+', '', current_text)
        
        # remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        current_text.translate(translator)
        
        # remove whitespace
        current_text = " ".join(current_text.split())
        
        # Apply Porter Stemmer
        word_tokens = word_tokenize(current_text)
        current_text = [stemmer.stem(word) for word in word_tokens]
        
        # current text is list item, change it to string, then return
        return_file.append(" ".join(current_text))
    
    df = pd.DataFrame({"Text": return_file})
    return df
  


preprocessed_data = preprocessing(train_data["Text"])


vectorizer = TfidfVectorizer(max_features = 5000, max_df = 0.9, stop_words="english") #stop_words
result = vectorizer.fit_transform(preprocessed_data["Text"]) # 

print('Print all words with an idf value < 2:')
for ele1, ele2 in zip(vectorizer.get_feature_names_out(), vectorizer.idf_):
    if ele2 < 2:
        print(ele1, ':', ele2)

word_matrix = result.toarray()

print("The shape of the resulting matrix is:", word_matrix.shape)


# create nmf object n_components are known
nmf = NMF(n_components=5)
# fit the tokenized data
y_pred_train_nmf = nmf.fit_transform(result)
# return indices of maximum value
y_pred_index_nmf = y_pred_train_nmf.argmax(axis=1)


# Create svd object, n_components is known previously in this case
svd = TruncatedSVD(n_components=5, algorithm = "arpack")
# fit the tokenized data
y_pred_train_svd = svd.fit_transform(result) # result
# return indices of maximum value
y_pred_index_svd = y_pred_train_svd.argmax(axis=1)


import itertools


def label_permute_compare(ytdf,yp,n=5):
    
    permutation_list = list(itertools.permutations(["business", "entertainment", 
                                                    "politics", "sport", "tech"])) #
    yp = pd.DataFrame(data=yp)
    
    acc = float(0)
    labelorder = list()
    
    for i in range(len(permutation_list)): 
        y_pred = yp
        y_pred = y_pred.replace([0,1,2,3,4], permutation_list[i]) 
        
        if accuracy_score(ytdf, y_pred) > acc:
            acc = accuracy_score(ytdf, y_pred)
            labelorder = permutation_list[i]
            
    return labelorder, acc


perm_nmf, acc_nmf = label_permute_compare(train_data['Category'], y_pred_index_nmf, n=5)
perm_svd, acc_svd = label_permute_compare(train_data['Category'], y_pred_index_svd, n=5)

print("Accuracy for training Data using NMF:", acc_nmf)
print("Accuracy for training Data using SVD:", acc_svd)



y_true = train_data.Category
y_pred_index_nmf = pd.DataFrame(y_pred_index_nmf)
y_pred = y_pred_index_nmf.replace([0,1,2,3,4], perm_nmf)


ConfusionMatrixDisplay.from_predictions(
   y_true, y_pred)


# preprocess test data
test_prep = preprocessing(test_data["Text"])

# Vectorize testdata
#vectorizer = TfidfVectorizer(max_features = 5000, max_df = 0.9, stop_words="english") #stop_words
test_result = vectorizer.transform(test_prep["Text"]) # 

# fit the tokenized data
y_pred_test_nmf = nmf.transform(test_result)
# return indices of maximum value
y_pred_test_index_nmf = y_pred_test_nmf.argmax(axis=1)

y_pred_test_index_nmf = pd.DataFrame(y_pred_test_index_nmf)
y_pred_test = y_pred_test_index_nmf.replace([0,1,2,3,4], perm_nmf)

# Kaggle accuracy score: 
submission = pd.DataFrame({'ArticleId': test_data['ArticleId'],
                           'Category': y_pred_test.iloc[:,0]})

submission.to_csv('submission.csv', index=False)



X_train, X_test, y_train, y_test = train_test_split(train_data.Text, train_data.Category, test_size = 0.2, random_state = 4790)


X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)


parameters_svm = {'loss':("hinge", "squared_hinge")}
svm = LinearSVC(max_iter = 5000, class_weight = "balanced")
grid_obj = GridSearchCV(svm, parameters_svm)
grid_obj.fit(X_train, y_train)


print(grid_obj.best_params_)


svm = LinearSVC(class_weight = "balanced", loss = "squared_hinge")
svm.fit(X_train, y_train)


print("Suport vector machine score:", svm.score(X_test, y_test))


parameters_rf = {'criterion':('gini', 'entropy', 'log_loss'),
                'max_depth': (2,4)} 
rf = RandomForestClassifier()
grid_obj_rf = GridSearchCV(rf, parameters_rf)
grid_obj_rf.fit(X_train, y_train)


print(grid_obj_rf.best_params_)


rf = RandomForestClassifier(criterion= 'gini', max_depth= 4)
rf.fit(X_train, y_train)
print("Random forest score: ", rf.score(X_test, y_test))

