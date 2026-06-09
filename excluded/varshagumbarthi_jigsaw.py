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





# ðŸ“Œ Step 1: Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ðŸ“Œ Step 2: Load the data
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
submission_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

# ðŸ“Œ Step 3: Use 'body' as the text input
train_texts = train_df['body'].fillna('')
train_labels = train_df['rule_violation']
test_texts = test_df['body'].fillna('')

# ðŸ“Œ Step 4: TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# ðŸ“Œ Step 5: Model - Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, train_labels)

# ðŸ“Œ Step 6: Predict
predictions = model.predict_proba(X_test)[:, 1]

# ðŸ“Œ Step 7: Save to submission file
submission_df['rule_violation'] = predictions
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
print(submission_df.head())







