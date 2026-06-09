import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter


test_data= pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")
test_data


sample= pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv")
sample


train_data= pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
train_data


train_data.info()


train_data.shape


train_data.isnull().sum()


train_data['label'].value_counts()


train_data['Question'][0]


X_train, X_test, y_train, y_test = train_test_split(train_data['Question'], train_data['label'], test_size=0.25, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train_vec, y_train)


y_pred = clf.predict(X_test_vec)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


X_test = vectorizer.transform(test_data['Question'])
test_data['Question'] = clf.predict(X_test)

# Map string labels to integers
label_mapping = {label: idx for idx, label in enumerate(sorted(test_data['Question'].unique()))}
test_data['label'] = test_data['Question'].map(label_mapping)

# Prepare submission
submission = test_data[['id', 'label']]
submission.to_csv("submission.csv", index=False)

