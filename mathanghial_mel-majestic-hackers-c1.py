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


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

train = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
test = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")
y = (train["label"] == "jailbreak").astype(int)
X = train["text"].astype(str)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range =(1,2), min_df = 2, max_features = 80000,
                             strip_accents = "unicode", lowercase = True)),
    ("clf", LogisticRegression(solver = "liblinear", class_weight = "balanced", max_iter = 500, C = 1.0))
])

pipe.fit(X, y)
proba = pipe.predict_proba(test["text"].astype(str))[:, 1]
sub = pd.DataFrame({"Id": test["Id"], "TARGET": proba}).sort_values("Id")
sub.to_csv("submission.csv", index = False)
print(sub.head())

