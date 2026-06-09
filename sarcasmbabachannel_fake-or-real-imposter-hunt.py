import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------------
# 1. Paths
# ------------------------
base_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
train_csv = os.path.join(base_path, "train.csv")
train_df = pd.read_csv(train_csv)

# ------------------------
# 2. Load train text files (correct folder names)
# ------------------------
train_texts = []
train_labels = []

for _, row in train_df.iterrows():
    article_id = row["id"]
    real_id = row["real_text_id"]

    # The folder name is like article_0070
    article_folder = os.path.join(base_path, "train", f"article_{int(article_id):04d}")

    # Read both files
    with open(os.path.join(article_folder, "file_1.txt"), encoding="utf-8") as f:
        file1 = f.read()
    with open(os.path.join(article_folder, "file_2.txt"), encoding="utf-8") as f:
        file2 = f.read()

    # Add file1 with correct label
    train_texts.append(file1)
    train_labels.append(1 if real_id == 1 else 0)

    # Add file2 with correct label
    train_texts.append(file2)
    train_labels.append(1 if real_id == 2 else 0)

# Create DataFrame
data = pd.DataFrame({"text": train_texts, "label": train_labels})

# ------------------------
# 3. Split for validation
# ------------------------
X_train, X_val, y_train, y_val = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# ------------------------
# 4. Build pipeline (TF-IDF + Logistic Regression)
# ------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=200))
])

# Train
model.fit(X_train, y_train)

# Validation performance
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# ------------------------
# 5. Predict on test data
# ------------------------
submission = []

test_path = os.path.join(base_path, "test")
for folder_name in sorted(os.listdir(test_path)):
    article_folder = os.path.join(test_path, folder_name)
    if not os.path.isdir(article_folder):
        continue

    # Read both files
    with open(os.path.join(article_folder, "file_1.txt"), encoding="utf-8") as f:
        file1 = f.read()
    with open(os.path.join(article_folder, "file_2.txt"), encoding="utf-8") as f:
        file2 = f.read()

    # Predict probabilities
    prob1 = model.predict_proba([file1])[0][1]
    prob2 = model.predict_proba([file2])[0][1]

    # Decide real file
    real_id = 1 if prob1 > prob2 else 2

    # Save result (remove 'article_' to match submission format)
    submission.append({"id": folder_name.replace("article_", ""), "real_text_id": real_id})

# ------------------------
# 6. Save submission file
# ------------------------
sub_df = pd.DataFrame(submission)
sub_df.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")


