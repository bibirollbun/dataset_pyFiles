# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data_train = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')

print("Training data")
print("Number of entries: ", data_train.size / data_train.columns.size)
print("Shape: ", data_train.shape)
print("Features: ", data_train.columns)
data_train.head()


print("Unique Categories: ", data_train['Category'].nunique())

data_train['Category'].value_counts().plot(kind='bar', figsize=(8, 5), color='skyblue')

plt.title("Number of Articles per Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


data_train["word_count"] = data_train["Text"].apply(lambda x: len(str(x).split()))

stats_train = data_train.groupby('Category')['word_count'].agg(['mean', 'min', 'max'])

stats_train.plot(kind='bar', figsize=(10, 6))
plt.title("Word Count Stats by Category")
plt.ylabel("Word Count")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



duplicates_across_categories = data_train.groupby("Text")["Category"].nunique()
num_multi_category_texts = (duplicates_across_categories > 1).sum()
total_duplicate_texts = (data_train.duplicated(subset="Text", keep=False)).sum()

print(f"Total texts that appear in more than one category: {num_multi_category_texts}")
print(f"Total duplicate texts (regardless of category): {total_duplicate_texts}")

data_train = data_train.drop_duplicates(subset=['Text'])

total_duplicate_texts = (data_train.duplicated(subset="Text", keep=False)).sum()

print(f"Total duplicate texts (regardless of category): {total_duplicate_texts}")


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_train = tfidf_vectorizer.fit_transform(data_train.Text)

feature_names = tfidf_vectorizer.get_feature_names_out()

X_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
print('Shape of Vectorized words:', X_df.shape)
X_df.head()


model = NMF(n_components=5).fit(X_train)
y_pred_train = model.transform(X_train)
y_pred_train_as_int = y_pred_train.argmax(axis=1)

print(model)


import itertools

# Taken from assignment in week 2
def label_permute_compare(ytdf,yp,n=5):
    """
    ytdf: labels dataframe object
    yp: clustering label prediction output
    Returns permuted label order and accuracy. 
    Example output: (3, 4, 1, 2, 0), 0.74 
    """
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(ytdf.values.ravel())
    
    yp = np.array(yp)
    
    best_acc = 0
    best_perm = None
    
    for perm in itertools.permutations(range(n)):
        mapped_yp = [perm[label] for label in yp]
        acc = accuracy_score(y_true, mapped_yp)
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_perm, best_acc


perm, acc = label_permute_compare(data_train[['Category']], y_pred_train_as_int, n=5)

print("Accuracy for train data: ", acc)


label_encoder = LabelEncoder()
y_true_train = label_encoder.fit_transform(data_train['Category'].values.ravel())

mapped_preds = [perm[label] for label in y_pred_train_as_int]

cm = confusion_matrix(y_true_train, mapped_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(xticks_rotation=45, cmap="Reds", ax=ax)
plt.show()


# transform and run model against test data
data_test = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
X_test = tfidf_vectorizer.transform(data_test.Text)
y_pred_test = model.transform(X_test)
y_pred_test_as_int = y_pred_test.argmax(axis=1)

#mapped_preds_test = [perm[label] for label in y_pred_test_as_int]
#predicted_labels_test = label_encoder.classes_[mapped_preds_test]

# Kaggle accuracy score: 
#submission = pd.DataFrame({'ArticleId': data_test['ArticleId'],
#                           'Category': predicted_labels_test})

#submission.to_csv('submission.csv', index=False)


results = []

param_grid = {
    'init': ['random', 'nndsvd'],
    'solver': ['cd', 'mu'],
    'beta_loss': ['frobenius', 'kullback-leibler'],
    'alpha_H': [0, 0.05, 0.1],
    'alpha_W': [0, 0.05, 0.1]
}

for params in ParameterGrid(param_grid):
    
    model_grid = NMF(n_components=5, init=params['init'], solver=params['solver'], alpha_W=params['alpha_W'], alpha_H=params['alpha_H'], max_iter=500).fit(X_train)
    y_pred_grid = model_grid.transform(X_train)
    y_pred_as_int_grid = y_pred_grid.argmax(axis=1)

    perm_grid, acc_grid = label_permute_compare(data_train[['Category']], y_pred_as_int_grid, n=5)

    results.append({
        'init': params['init'],
        'solver': params['solver'],
        'beta_loss': params['beta_loss'],
        'accuracy': acc_grid,
        'test accuracy': None,
        'perm': perm_grid,
        'model': model_grid
    })

results_df = pd.DataFrame(results)

# Display the table
results_df.sort_values(by='accuracy', ascending=False)


#y_pred_test = results[24]['model'].transform(X_test)
#y_pred_test_as_int = y_pred_test.argmax(axis=1)
#perm = results[24]['perm']

#mapped_preds_test = [perm[label] for label in y_pred_test_as_int]
#predicted_labels_test = label_encoder.classes_[mapped_preds_test]

# Kaggle accuracy score: 
#submission = pd.DataFrame({'ArticleId': data_test['ArticleId'],
#                           'Category': predicted_labels_test})

#submission.to_csv('submission.csv', index=False)

results_df.loc[2, 'test accuracy'] = 0.92380
results_df.loc[6, 'test accuracy'] = 0.92380
results_df.loc[21, 'test accuracy'] = 0.90884
results_df.loc[29, 'test accuracy'] = 0.91292
results_df.loc[24, 'test accuracy'] = 0.67619

results_df.sort_values(by='test accuracy', ascending=False).head(5)


param_grid = {
    'max_df': [0.8, 0.85, 0.9, 0.95, 1.0],
    'min_df': [1, 2, 3, 4],
    'stop_words': [None, 'english'],
}
results = []

for params in ParameterGrid(param_grid):

    tfidf_vectorizer = TfidfVectorizer(max_df=params['max_df'], min_df=params['min_df'], stop_words=params['stop_words'])
    X_train_grid = tfidf_vectorizer.fit_transform(data_train.Text)
    
    model_grid = NMF(n_components=5).fit(X_train_grid)
    y_pred_grid = model_grid.transform(X_train_grid)
    y_pred_as_int_grid = y_pred_grid.argmax(axis=1)

    perm_grid, acc_grid = label_permute_compare(data_train[['Category']], y_pred_as_int_grid, n=5)

    results.append({
        'max_df': params['max_df'],
        'min_df': params['min_df'],
        'stop_words': params['stop_words'],
        'accuracy': acc_grid,
        'test accuracy': None,
        'perm': perm_grid,
        'model': model_grid,
        'vectorizer': tfidf_vectorizer
    })

results_df = pd.DataFrame(results)

# Display the table
results_df.sort_values(by='accuracy', ascending=False)


#X_test = results[7]['vectorizer'].transform(data_test.Text)
#y_pred_test = results[7]['model'].transform(X_test)
#y_pred_as_int_test = y_pred_test.argmax(axis=1)
#perm = results[7]['perm']

#pred_mapped_test = [perm[label] for label in y_pred_as_int_test]
#predicted_labels_test = label_encoder.classes_[pred_mapped_test]

# Kaggle accuracy score: 
#submission = pd.DataFrame({'ArticleId': data_test['ArticleId'],
#                           'Category': predicted_labels_test})

#submission.to_csv('submission.csv', index=False)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

best_vectorizer = results[15]['vectorizer']

X_train = best_vectorizer.transform(data_train.Text)
X_test = best_vectorizer.transform(data_test.Text)

le = LabelEncoder()
y_train = le.fit_transform(data_train['Category'].values.ravel())

model_log = LogisticRegression(max_iter=1000)
model_log.fit(X_train, y_train)

y_pred_log_train = model_log.predict(X_train)
y_pred_log_test = model_log.predict(X_test)

perm_log, acc_log = label_permute_compare(data_train[['Category']], y_pred_log_train, n=5)
print("Log train accuracy: ", acc_log)

#pred_mapped_test = [perm_log[label] for label in y_pred_log_test]
#predicted_labels_test = le.classes_[pred_mapped_test]

# Kaggle accuracy score: 
#submission = pd.DataFrame({'ArticleId': data_test['ArticleId'],
#                           'Category': predicted_labels_test})

#submission.to_csv('submission.csv', index=False)
print("Log test accuracy: 0.98231")

model_svm = LinearSVC()
model_svm.fit(X_train, y_train)

y_pred_svm_train = model_svm.predict(X_train)
y_pred_svm_test = model_svm.predict(X_test)

perm_svm, acc_svm = label_permute_compare(data_train[['Category']], y_pred_svm_train, n=5)
print("SVM train accuracy: ", acc_svm)

#pred_mapped_test = [perm_svm[label] for label in y_pred_svm_test]
#predicted_labels_test = le.classes_[pred_mapped_test]

# Kaggle accuracy score: 
#submission = pd.DataFrame({'ArticleId': data_test['ArticleId'],
#                           'Category': predicted_labels_test})

#submission.to_csv('submission.csv', index=False)
print("SVM test accuracy: 0.98095")


from sklearn.model_selection import train_test_split

for frac in [0.1, 0.2, 0.5, 0.9]:
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, train_size=frac)

    model_log = LogisticRegression(max_iter=1000)
    model_log.fit(X_train_frac, y_train_frac)
    y_pred_log = model_log.predict(X_train)
    perm_log, acc_log = label_permute_compare(data_train[['Category']], y_pred_log, n=5)
    print(f"LogReg ({int(frac*100)}% train): Accuracy = {acc_log:.3f}")

    model_svm = LinearSVC()
    model_svm.fit(X_train_frac, y_train_frac)
    y_pred_svm = model_svm.predict(X_train)
    perm_svm, acc_svm = label_permute_compare(data_train[['Category']], y_pred_svm, n=5)
    print(f"SVM    ({int(frac*100)}% train): Accuracy = {acc_svm:.3f}")
    

