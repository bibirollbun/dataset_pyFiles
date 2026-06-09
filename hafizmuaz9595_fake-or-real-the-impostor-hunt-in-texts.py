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


# Imports
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def read_texts_from_dir(dir_path):
    data = []
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
                    text1 = f1.read().strip()
                with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
                    text2 = f2.read().strip()
                index = int(folder_name[-4:])
                data.append((index, text1, text2))
            except Exception as e:
                print(f"Error reading {folder_name}: {e}")
    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2'])

# Paths (adjust if needed)
train_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

# Read data
df_train = read_texts_from_dir(train_dir)
df_test = read_texts_from_dir(test_dir)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)



df_train


df_test


# Load train labels
df_labels = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")

# Merge labels
df_train = df_train.merge(df_labels, on='id')

# Reshape train into long format
train_long = []
for _, row in df_train.iterrows():
    train_long.append({'id': row['id'], 'text': row['file_1'], 'label': 1 if row['real_text_id'] == 1 else 0})
    train_long.append({'id': row['id'], 'text': row['file_2'], 'label': 1 if row['real_text_id'] == 2 else 0})
df_train_long = pd.DataFrame(train_long)

# Reshape test into long format
test_long = []
for _, row in df_test.iterrows():
    test_long.append({'id': row['id'], 'text': row['file_1'], 'file_num': 1})
    test_long.append({'id': row['id'], 'text': row['file_2'], 'file_num': 2})
df_test_long = pd.DataFrame(test_long)

print("Train long shape:", df_train_long.shape)
print("Test long shape:", df_test_long.shape)  # should be 2 * original test rows



def clean_text(text):
    text = text.lower()
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

df_train_long['clean_text'] = df_train_long['text'].apply(clean_text)
df_test_long['clean_text'] = df_test_long['text'].apply(clean_text)



lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

# Train
df_train_long['tokens'] = df_train_long['clean_text'].apply(word_tokenize)
df_train_long['tokens'] = df_train_long['tokens'].apply(lemmatize_tokens)
df_train_long['lemmas'] = df_train_long['tokens'].apply(lambda x: ' '.join(x))

# Test
df_test_long['tokens'] = df_test_long['clean_text'].apply(word_tokenize)
df_test_long['tokens'] = df_test_long['tokens'].apply(lemmatize_tokens)
df_test_long['lemmas'] = df_test_long['tokens'].apply(lambda x: ' '.join(x))



tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=2000)
tfidf_vectorizer.fit(df_train_long['lemmas'].tolist())

X_train_full = tfidf_vectorizer.transform(df_train_long['lemmas'].values)
y_train_full = df_train_long['label'].values
X_test_full = tfidf_vectorizer.transform(df_test_long['lemmas'].values)



ids_train = df_train_long['id'].values

X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
    X_train_full, y_train_full, ids_train, test_size=0.2, random_state=42, stratify=y_train_full
)

print("X_train:", X_train.shape, "X_val:", X_val.shape)



# Model Training and Testing
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_val)


y_pred = rf_model.predict(X_val)
print(classification_report(y_val, y_pred))


from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = svm_model.predict(X_val)
print(classification_report(y_val, y_pred))


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_val_pred = lr_model.predict(X_val)
print(classification_report(y_val, y_val_pred))



test_preds = lr_model.predict(X_test_full)
df_test_long['pred_label'] = test_preds


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score

# Define the ensemble (hard voting or soft voting)
ensemble_model = VotingClassifier(
    estimators=[
        ('logreg', lr_model),
        ('svm', svm_model),
        ('rf', rf_model)
    ],
    voting='hard'  # use 'soft' if classifiers can output probabilities
)

# Fit the ensemble
ensemble_model.fit(X_train, y_train)

# Predict
y_pred = ensemble_model.predict(X_val)

# Evaluate
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))



test_preds_all = ensemble_model.predict(X_test_full)
df_test_long['pred_label'] = test_preds_all


# Prepare submission: one row per original pair
submission = []
for pair_id in df_test_long['id'].unique():
    pair = df_test_long[df_test_long['id'] == pair_id].sort_values('file_num')
    # first text = file_num 1, second text = file_num 2
    if pair.iloc[0]['pred_label'] == 1:
        real_text_id = 1
    else:
        real_text_id = 2
    submission.append({'id': pair_id, 'real_text_id': real_text_id})

submission_df = pd.DataFrame(submission)
submission_df.to_csv('submission.csv', index=False)
submission_df.head()



submission_df


submission_df.shape




