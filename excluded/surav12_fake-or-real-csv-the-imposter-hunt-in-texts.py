import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Path to data --- #
train_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train/' 
test_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test/'   
train_csv_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv' 

# --- Load training labels CSV --- #
train_labels = pd.read_csv(train_csv_path)

# --- Prepare training data --- #
texts = []   # store all article texts
labels = []  # store corresponding labels (1 = real, 0 = fake)

# Loop through each row in the training labels file
for index, row in train_labels.iterrows():
    article_id = row['id']
    real_text_id = row['real_text_id']
    article_path = os.path.join(train_dir, f'article_{article_id:04d}')

    file1_path = os.path.join(article_path, 'file_1.txt')
    file2_path = os.path.join(article_path, 'file_2.txt')

    # Read the text files
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            file1_text = f.read()
        with open(file2_path, 'r', encoding='utf-8') as f:
            file2_text = f.read()
    except FileNotFoundError:
        continue  # skip if files are missing

    # Assign real and fake text based on label
    if real_text_id == 1:
        real_text = file1_text
        fake_text = file2_text
    else:
        real_text = file2_text
        fake_text = file1_text

    # Add both real and fake text to the dataset
    texts.extend([real_text, fake_text])
    labels.extend([1, 0])

# --- TF-IDF Vectorization --- #
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(texts)
y = labels

# --- Train Logistic Regression Model --- #
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X, y)

# --- Prepare test data and make predictions --- #
test_articles = sorted(os.listdir(test_dir))
submission_rows = []

for filename in test_articles:
    if not filename.startswith('article_'):
        continue
    article_id_str = filename.split('_')[1]
    article_id = int(article_id_str)
    article_path = os.path.join(test_dir, filename)

    file1_path = os.path.join(article_path, 'file_1.txt')
    file2_path = os.path.join(article_path, 'file_2.txt')

    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            file1_text = f.read()
        with open(file2_path, 'r', encoding='utf-8') as f:
            file2_text = f.read()
    except FileNotFoundError:
        continue

    # Vectorize both files
    vec1 = vectorizer.transform([file1_text])
    vec2 = vectorizer.transform([file2_text])

    # Get predicted probability of being real for each file
    prob1 = model.predict_proba(vec1)[0][1]
    prob2 = model.predict_proba(vec2)[0][1]

    # Select the file with higher "real" probability
    if prob1 > prob2:
        predicted_real_text_id = 1
    else:
        predicted_real_text_id = 2

    submission_rows.append({'id': article_id, 'predicted_real_text_id': predicted_real_text_id})

# --- Save submission file --- #
submission_df = pd.DataFrame(submission_rows)
assert len(submission_df) == 1068, f"Submission must have 1068 rows, but got: {len(submission_df)}"
submission_df.to_csv('submission.csv', index=False)

print(f"Created submission.csv with {len(submission_df)} rows.")


