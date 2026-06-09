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
import os

# å¿…è¦�ã�ªãƒ©ã‚¤ãƒ–ãƒ©ãƒªã‚’è¿½åŠ 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier # ãƒ�ãƒ«ãƒ�ãƒ©ãƒ™ãƒ«åˆ†é¡�ç”¨

# --- 1. ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿ (å…¨ãƒ‡ãƒ¼ã‚¿) ---
print("Loading data...")
train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®ãƒ†ã‚­ã‚¹ãƒˆã‚’æº–å‚™
X_train_text = train_df['comment_text']
y_train = train_df[target_columns]
X_test_text = test_df['comment_text']


# --- 2. TF-IDFã�«ã‚ˆã‚‹ãƒ™ã‚¯ãƒˆãƒ«åŒ– (é«˜é€Ÿ) ---
print("Vectorizing text data using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=50000, 
                                   stop_words='english')

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
print(f"Vectorized shape (train): {X_train_tfidf.shape}")


# --- 3. ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ (ãƒ­ã‚¸ã‚¹ãƒ†ã‚£ãƒƒã‚¯å›�å¸°) ---
print("Training model (Logistic Regression)...")
base_model = LogisticRegression(solver='saga', random_state=0, n_jobs=-1)
model = OneVsRestClassifier(base_model)
model.fit(X_train_tfidf, y_train)


# --- 4. ğŸ’¥ã€�å¤‰æ›´ã€‘ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�§ã�®äºˆæ¸¬ (ç¢ºç�‡ã‚’äºˆæ¸¬) ---
print("Predicting probabilities on test data...")


# .predict() ã�§ã�¯ã�ªã�� .predict_proba() ã‚’ä½¿ã�„ã€Œç¢ºç�‡ã€�ã‚’äºˆæ¸¬ã�™ã‚‹
# ã�“ã‚Œã�«ã‚ˆã‚Šã€�[0, 1] ã�§ã�¯ã�ªã�� [0.12, 0.85] ã�®ã‚ˆã�†ã�ªç¢ºç�‡ã�®ãƒªã‚¹ãƒˆã�Œå¾—ã‚‰ã‚Œã�¾ã�™
pred_test_proba = model.predict_proba(X_test_tfidf)


# --- 5. ğŸ’¥ã€�å¤‰æ›´ã€‘æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ� (ç¢ºç�‡ã‚’ä½¿ç”¨) ---
print("Creating submission file...")
# äºˆæ¸¬çµ�æ�œã�®å¤‰æ•°å��(pred_test_proba)ã‚’æ­£ã�—ã��æŒ‡å®šã�™ã‚‹
sub_df = pd.DataFrame(pred_test_proba, columns=target_columns)
sub_df['id'] = test_df['id'].values
sub_df = sub_df[['id'] + target_columns]

# 
sub_df.to_csv("Kazu_submission10.csv", index=False)
print("Submission file created successfully!")
sub_df.head()

