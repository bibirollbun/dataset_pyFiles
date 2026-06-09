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


# libries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/train.csv')
test=pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/test.csv')
submission=pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/sample_submission.csv')



train.head()


# Total row and columns in datasets
print(f'The training datasets are : {train.shape}')
print(f'The testing datasets are : {test.shape}')
print("="*100)

# columns of datsets
print(f'Training datasetscolumns are :{train.columns}')
print(f'Testing datasets columns are :{test.columns}')
print("="*100)
# Datatype
print(train.info())
print("="*100)
# Mssing values
print(f'Missing values in training : {train.isna().sum()}')
print("="*100)
print(f'Missing values in testing : {test.isna().sum()}')

# Check the lebel output (IMBALANCED)
print("="*100)
print('The distibution of the label of training datasets are :',train['sentiment'].value_counts(normalize=True)*100)

# Check the summaray
print("="*100)
print(train.describe())

# Check the duplicate
print("="*100)
print(f'Duplicate columns are :{train.duplicated().sum()}')


plt.figure()
train['sentiment'].value_counts().sort_index().plot(kind='bar')
plt.title('Label counts (0=neg & 1=pos)')
plt.xlabel('sentiment')
plt.ylabel('count')
plt.show()


train['char_len'] = train['review'].astype(str).str.len()
plt.figure()
train['char_len'].plot(kind='hist', bins=30)
plt.title("Review length distribution (chars)")
plt.xlabel("chars"); plt.ylabel("count")
plt.show()


import re
# clean the dataset
def clean(data):
    data=str(data).lower()
    data=re.sub('[^a-z A-Z 0-9]',' ',data)
    data= re.sub(r"\s+", " ", data).strip()  #normalize space
    return data



# Cleaning the training dataset and testting dataset
train['clean_review']=train['review'].apply(clean)
test['clean_review']=test['review'].apply(clean)


train['clean_review']


X_train=train['clean_review'].values
y_train=train['sentiment'].values
X_test=test['clean_review'].values
y_test=submission['sentiment'].values


tfidf_common = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "sublinear_tf": True
}



from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

pipelines = {
    "LogReg": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_common)),
        ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight='balanced'))
    ]),

    "LinearSVC": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_common)),
        ("clf", LinearSVC(C=1.0))
    ]),

    "CompNB": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_common)),
        ("clf", ComplementNB(alpha=0.5))
    ])
}




from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = {}

for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_scores[name] = (scores.mean(), scores.std())
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")



best_model_name = max(cv_scores, key=lambda k: cv_scores[k][0])
print("\nBest model is:", best_model_name)



best_model = pipelines[best_model_name]
best_model.fit(X_train, y_train)



y_pred = best_model.predict(X_test)
y_pred.shape,submission.shape


submission.head()


submission1 = pd.DataFrame({
    "id": test["id"],
    "sentiment": y_pred.astype(int)
})
submission1.head()


submission1.to_csv('submission done.csv', index=False)


