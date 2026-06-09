# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_json("/kaggle/input/cooking123/train.json")
test = pd.read_json("/kaggle/input/cooking123/test.json")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.head()


def join_ingredients(ing_list):
    return " ".join(ing_list)

train["text"] = train["ingredients"].apply(join_ingredients)
test["text"]  = test["ingredients"].apply(join_ingredients)

X_train = train["text"]
y_train = train["cuisine"]
X_test  = test["text"]

print(X_train.iloc[0])
print(y_train.iloc[0])



from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", LinearSVC())
])

scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1, scoring="accuracy")
print("CV accuracy: {:.4f} ± {:.4f}".format(scores.mean(), scores.std()))


model.fit(X_train, y_train)


test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "cuisine": test_preds
})

submission.to_csv("submission.csv", index=False)

submission.head()

