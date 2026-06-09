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
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, classification_report, confusion_matrix


# Load Data
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
submission = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')

# Preprocess & prepare labels
le = LabelEncoder()
train['Misconception'] = train['Misconception'].fillna('NA')
train['target'] = train['Category']+":"+train['Misconception']
train['label'] = le.fit_transform(train['target']) # just an id for target
train['text'] = train['QuestionText'].fillna('') + ' ' + train['StudentExplanation'].fillna('')
test['text'] = test['QuestionText'].fillna('') + ' ' + test['StudentExplanation'].fillna('')



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 37
N_SPLITS = 5
MAX_FEATURES = 50000

label2id = {label: idx for idx, label in enumerate(le.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

X = train['text']
y = train['label']

# Vectorization 
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)
X_tfitrain = vectorizer.fit_transform(X)

# Define MAP@k
def mapk(actual, predicted, k=3):
    score = 0.0
    for a, p in zip(actual, predicted):
        if a in p:
            score += 1.0 / (p.index(a) + 1)
    return score / len(actual)


# Cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
log_losses = []
map3_scores = []

print("Initializing cross validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfitrain, y)):
    print(f"\n--- Fold {fold + 1} ---")
    
    X_train, X_val = X_tfitrain[train_idx], X_tfitrain[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        solver='liblinear',
        multi_class='ovr'
    )
    
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    
    # probability matrix with all classes
    n_classes = len(le.classes_)
    y_probs_full = np.zeros((y_probs.shape[0], n_classes))
    
    predicted_classes = model.classes_
    for idx, cls in enumerate(predicted_classes):
        y_probs_full[:, cls] = y_probs[:, idx]
    
    fold_log_loss = log_loss(y_val, y_probs_full, labels=np.arange(n_classes))

    log_losses.append(fold_log_loss)

    # MAP@3
    top3 = np.argsort(y_probs, axis=1)[:, ::-1][:, :3]
    map3 = mapk(y_val.tolist(), top3.tolist(), k=3)
    map3_scores.append(map3)
    
    print(f"Log Loss: {fold_log_loss:.4f} | MAP@3: {map3:.4f}")
    
    if fold == 0:
        print("\nClassification Report:")
        labels_in_fold = np.unique(y_val)
        print(classification_report(
            y_val, y_pred,
            labels=labels_in_fold,
            target_names=le.classes_[labels_in_fold]
        ))

print("\n--- Final Results ---")
print(f"Log Loss: {np.mean(log_losses):.4f}")
print(f"Standard Variation: {np.std(log_losses):.4f}")

print(f"\nMAP@3: {np.mean(map3_scores):.4f}")
print(f"STD MAP@3: {np.std(map3_scores):.4f}")



X_test = vectorizer.transform(test['text'])

y_test_probs = model.predict_proba(X_test)
top3_preds = np.argsort(y_test_probs, axis=1)[:, ::-1][:, :3]  # Top 3 classes
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(-1, 3)  # Decodificar rótulos

joined_preds = [' '.join(row) for row in top3_labels]



# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()

