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
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")


df.head()


df.shape


df.columns


df = df.drop(['row_id',
              'positive_example_1',
              'positive_example_2',
              'negative_example_1',
              'negative_example_2',
              'subreddit'
              ], axis=1)

df.head()


import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+", "URL", text)  # replace links
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special chars
    text = re.sub(r'\s+[a-zA-Z]\s+',' ',text) ## remove all white spaces ex:-\t,\n,\r etc.
    return text.lower()

df["body_clean"] = df["body"].apply(lambda x : clean_text(x))


x = df['body_clean']
y =df['rule_violation']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(x_train)
X_test_tfidf = vectorizer.transform(x_test)


from sklearn.linear_model import RidgeClassifier

# Convert sparse matrix to dense if needed
X_train_dense = X_train_tfidf.toarray()  # only if X_train_tfidf is sparse
X_test_dense = X_test_tfidf.toarray()

model = RidgeClassifier()
model.fit(X_train_dense, y_train)
y_pred = model.predict(X_test_dense)



from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Load test set
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# Clean text same as train
test_df["body_clean"] = test_df["body"].apply(lambda x: clean_text(x))

# Transform text
X_test_final = vectorizer.transform(test_df["body_clean"])

# Predict using trained model
preds = model.predict(X_test_final)

# Load sample submission to match exact column names
sample = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

# Create submission DataFrame with correct format
submission = pd.DataFrame({
    "row_id": test_df["row_id"],          # must match sample_submission
    "rule_violation": preds               # column name must match sample_submission
})

# Save submission file
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv file created successfully!")





