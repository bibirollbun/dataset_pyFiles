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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


train_df= pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
test_df=pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")


test_df.shape


X_train= train_df['Question'].values
y_train= train_df['label'].values
X_test= train_df['Question'].values


NUM_FOLDS=5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Initialize array for out-of-fold predictions
oof_preds = np.zeros(len(X_train), dtype=int)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {fold+1}/{NUM_FOLDS}")
    
    # Split train/validation for this fold
    X_trn, y_trn = X_train[trn_idx], y_train[trn_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X_trn_tfidf = vectorizer.fit_transform(X_trn)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_trn_tfidf, y_trn)
    
    # Predict on validation fold
    y_val_pred = model.predict(X_val_tfidf)
    oof_preds[val_idx] = y_val_pred
    
    fold_f1 = f1_score(y_val, y_val_pred, average="micro")
    print(f"Fold {fold+1} F1 (micro): {fold_f1:.4f}")

# Overall OOF F1
oof_f1 = f1_score(y_train, oof_preds, average="micro")
print("Overall OOF F1 (micro):", oof_f1)



final_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = final_vectorizer.fit_transform(X_train)

final_model = LogisticRegression(max_iter=2000, random_state=42)
final_model.fit(X_train_tfidf, y_train)


X_test = test_df["Question"].values
X_test_tfidf = final_vectorizer.transform(X_test)
test_preds = final_model.predict(X_test_tfidf)


submission = pd.DataFrame({"id": test_df["id"], "label": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")




