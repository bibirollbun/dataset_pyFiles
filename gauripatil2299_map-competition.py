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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample_sub = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')



train_df['Misconception'] = train_df['Misconception'].fillna('NA')
train_df['Target'] = train_df['Category'] + ':' + train_df['Misconception']

X = train_df['StudentExplanation']
y = train_df['Target']


X = train_df['StudentExplanation']
y = train_df['Target']
le = LabelEncoder()
y_encoded = le.fit_transform(y)



y_counts = pd.Series(y_encoded).value_counts()
valid_classes = y_counts[y_counts >= 3].index
mask = pd.Series(y_encoded).isin(valid_classes)
X_filtered = X[mask]
y_filtered = y_encoded[mask]


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_filtered, y_filtered)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
f1_scores = []

for train_idx, val_idx in skf.split(X_filtered, y_filtered):
    X_tr, X_val = X_filtered.iloc[train_idx], X_filtered.iloc[val_idx]
    y_tr, y_val = y_filtered[train_idx], y_filtered[val_idx]

    pipeline.fit(X_tr, y_tr)
    val_preds = pipeline.predict(X_val)
    
    macro_f1 = f1_score(y_val, val_preds, average='macro')
    f1_scores.append(macro_f1)
    print(f"Fold Macro-F1: {macro_f1:.4f}")

print(f"\n✅ Average CV Macro-F1: {np.mean(f1_scores):.4f}")



pipeline.fit(X, y_encoded)


preds_proba = pipeline.predict_proba(test_df['StudentExplanation'])
top3_indices = preds_proba.argsort(axis=1)[:, -3:][:, ::-1]  # top-3 in descending order

pred_labels = []
for indices in top3_indices:
    labels = le.inverse_transform(indices)
    pred_labels.append(" ".join(labels))


submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': pred_labels
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv saved")

