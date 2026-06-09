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


from itertools import permutations
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from wordcloud import WordCloud



data_train = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Train.csv")
data_test = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Test.csv")
sample_solution = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Sample Solution.csv")


print(data_train.info())
data_train.head()


data_train = data_train.drop_duplicates(subset = ["Text"])
data_train.Category = pd.Categorical(data_train.Category)


data_train.Category.value_counts().plot.pie(autopct='%1.1f%%', ylabel='')


print(data_test.info())
data_test.head()


print(sample_solution.info())
sample_solution.head()


vectoriser = TfidfVectorizer(stop_words="english") # lowercase is True by default and analyzer==word.
combined_text = pd.concat([data_train.Text, data_test.Text], ignore_index=True)
vectorised = vectoriser.fit_transform(combined_text)
print(vectorised.shape)


word_frequencies = pd.DataFrame(vectorised.toarray(), columns=vectoriser.get_feature_names_out()).T.sum(axis=1)
wordcloud = WordCloud(background_color='white').generate_from_frequencies(word_frequencies.to_dict())
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


nmf = NMF(n_components=5)
train_pred = nmf.fit_transform(vectorised).argmax(axis=1)[:1440]


def label_permute_compare(ytdf,yp):
    """
    ytdf: labels column from dataframe object.
    yp: label prediction output.
    Returns permuted label order and accuracy.
    Example output: ('business', 'politics', 'sport', 'entertainment', 'tech'), 0.74 .
    """
    y_true = ytdf
    best_acc = 0.
    for perm in permutations(ytdf.cat.categories):
        y_pred = [perm[i] for i in yp]
        accuracy = accuracy_score(y_true, y_pred)
        if accuracy > best_acc:
            best_acc = accuracy
            best_perm = perm
    return best_perm, best_acc

labels, acc = label_permute_compare(data_train.Category, train_pred)
print("Accuracy: %.1f%%" % (acc * 100))
confusion_matrix(data_train.Category, [labels[x] for x in train_pred])


text_without_nums = combined_text.str.replace(r'\d+', 'NUM', regex=True)
vectorised_without_nums = TfidfVectorizer(stop_words="english").fit_transform(text_without_nums)
without_nums_pred = NMF(5).fit_transform(vectorised_without_nums).argmax(axis=1)[:1440]
labels_without_num, acc_without_num = label_permute_compare(data_train.Category, without_nums_pred)
print("Accuracy: %.1f%%" % (acc_without_num * 100))
confusion_matrix(data_train.Category, [labels_without_num[x] for x in without_nums_pred])


def gen_model(vectoriser):
    model = {"vectoriser": vectoriser}
    vectorised = vectoriser.fit_transform(combined_text)
    nmf = NMF(n_components=5)
    model["nmf"] = nmf
    y_pred = nmf.fit_transform(vectorised).argmax(axis=1)[:1440]
    model["pred"] = y_pred
    model["labels"], model["acc"] = label_permute_compare(data_train.Category, y_pred)
    return model

best_accuracy = 0.
for norm in ("l1", "l2", None):
    for max_features in (2000, 4000, 6000, 8000, None):
        model = gen_model(TfidfVectorizer(stop_words="english", norm=norm, max_features=max_features))
        print("norm=%s, max_features=%s, acc=%.5f%%" % (norm, max_features, model["acc"] * 100))
        if model["acc"] > best_accuracy:
            best_model = model
            best_accuracy = model["acc"]
print("Best model accuracy = %.3f%%" % (best_model["acc"] * 100))
confusion_matrix(data_train.Category, [best_model["labels"][x] for x in best_model["pred"]])


vectorised = best_model["vectoriser"].transform(combined_text)
test_pred = best_model["nmf"].transform(vectorised).argmax(axis=1)[1440:]


sample_solution.Category = [best_model["labels"][x] for x in test_pred]
print("The order of solution df and test df is equivalent:", all(sample_solution.ArticleId == data_test.ArticleId))
print(sample_solution)
sample_solution.to_csv("submission.csv", index=False)


train_vectorised = best_model["vectoriser"].transform(data_train.Text)
X_train, X_test, y_train, y_test = train_test_split(train_vectorised, data_train.Category, test_size=.2, random_state=4)
clf = LogisticRegression(random_state=4).fit(X_train, y_train)
print("Training score: %.1f%%" % (clf.score(X_train, y_train) * 100))
print("Test score: %.1f%%" % (clf.score(X_test, y_test) * 100))
confusion_matrix(y_test, clf.predict(X_test))


print("Training size: 60%")
X_train, X_test, y_train, y_test = train_test_split(train_vectorised, data_train.Category, test_size=.4, random_state=4)
clf = LogisticRegression(random_state=4).fit(X_train, y_train)
print("Training score: %.1f%%" % (clf.score(X_train, y_train) * 100))
print("Test score: %.1f%%" % (clf.score(X_test, y_test) * 100))
print("\nTraining size: 40%")
X_train, X_test, y_train, y_test = train_test_split(train_vectorised, data_train.Category, test_size=.6, random_state=4)
clf = LogisticRegression(random_state=4).fit(X_train, y_train)
print("Training score: %.1f%%" % (clf.score(X_train, y_train) * 100))
print("Test score: %.1f%%" % (clf.score(X_test, y_test) * 100))
print("\nTraining size: 20%")
X_train, X_test, y_train, y_test = train_test_split(train_vectorised, data_train.Category, test_size=.8, random_state=4)
clf = LogisticRegression(random_state=4).fit(X_train, y_train)
print("Training score: %.1f%%" % (clf.score(X_train, y_train) * 100))
print("Test score: %.1f%%" % (clf.score(X_test, y_test) * 100))


print("Training size: 80%")
X_train, X_test, y_train, y_test = train_test_split(train_vectorised, data_train.Category, test_size=.2, random_state=4)
clf = LinearSVC(random_state=4).fit(X_train, y_train)
print("Training score: %.1f%%" % (clf.score(X_train, y_train) * 100))
print("Test score: %.1f%%" % (clf.score(X_test, y_test) * 100))


print("Training size: 60%")
X_train, X_test, y_train, y_test = train_test_split(train_vectorised, data_train.Category, test_size=.4, random_state=4)
clf = LinearSVC(random_state=4).fit(X_train, y_train)
print("Training score: %.1f%%" % (clf.score(X_train, y_train) * 100))
print("Test score: %.1f%%" % (clf.score(X_test, y_test) * 100))
print("\nTraining size: 40%")
X_train, X_test, y_train, y_test = train_test_split(train_vectorised, data_train.Category, test_size=.6, random_state=4)
clf = LinearSVC(random_state=4).fit(X_train, y_train)
print("Training score: %.1f%%" % (clf.score(X_train, y_train) * 100))
print("Test score: %.1f%%" % (clf.score(X_test, y_test) * 100))
print("\nTraining size: 20%")
X_train, X_test, y_train, y_test = train_test_split(train_vectorised, data_train.Category, test_size=.8, random_state=4)
clf = LinearSVC(random_state=4).fit(X_train, y_train)
print("Training score: %.1f%%" % (clf.score(X_train, y_train) * 100))
print("Test score: %.1f%%" % (clf.score(X_test, y_test) * 100))




