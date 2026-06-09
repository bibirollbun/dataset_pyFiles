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



import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
submission_df = pd.read_csv('/kaggle/input/mercor-ai-detection/sample_submission.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['clean_text'] = train_df['answer'].apply(clean_text)
test_df['clean_text'] = test_df['answer'].apply(clean_text)


for df in [train_df, test_df]:
    df['char_count'] = df['answer'].apply(len)
    df['word_count'] = df['answer'].apply(lambda x: len(str(x).split()))
    df['avg_word_len'] = df['answer'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)


X_train, X_valid, y_train, y_valid = train_test_split(
    train_df, train_df['is_cheating'], test_size=0.2, random_state=42
)


tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train['clean_text'])
X_valid_tfidf = tfidf.transform(X_valid['clean_text'])
X_test_tfidf = tfidf.transform(test_df['clean_text'])



model1 = LogisticRegression(max_iter=2000, C=1.5, solver='lbfgs')
model1.fit(X_train_tfidf, y_train)


train_pred_proba = model1.predict_proba(X_train_tfidf)[:, 1]
valid_pred_proba = model1.predict_proba(X_valid_tfidf)[:, 1]
test_pred_proba = model1.predict_proba(X_test_tfidf)[:, 1]


X_train_meta = np.column_stack((train_pred_proba, X_train[['char_count','word_count','avg_word_len']].values))
X_valid_meta = np.column_stack((valid_pred_proba, X_valid[['char_count','word_count','avg_word_len']].values))
X_test_meta = np.column_stack((test_pred_proba, test_df[['char_count','word_count','avg_word_len']].values))


model2 = LogisticRegression(max_iter=2000, C=2.0, solver='lbfgs')
model2.fit(X_train_meta, y_train)


valid_pred = model2.predict(X_valid_meta)
valid_pred_proba = model2.predict_proba(X_valid_meta)[:, 1]

accuracy = accuracy_score(y_valid, valid_pred)
roc_auc = roc_auc_score(y_valid, valid_pred_proba)

print("\n==========================")
print("Validation Results")
print("==========================")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, valid_pred))


final_pred_proba = model2.predict_proba(X_test_meta)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'is_cheating': final_pred_proba
})

submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


print(f"Final Validation ROC-AUC Score: {roc_auc:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")









































































