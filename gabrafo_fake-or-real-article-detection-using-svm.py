!pip install scikit-learn


import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

base_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
csv_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"

df_labels = pd.read_csv(csv_path)
df_labels = df_labels.drop(columns=['id'])

texts = []
labels = []

# Iterate over each row in the CSV and load the corresponding text files
for i, row in enumerate(df_labels.itertuples()):
    folder_name = f"article_{i:04d}"
    folder_path = os.path.join(base_path, folder_name)

    real_id = int(row.real_text_id)
    fake_id = 1 if real_id == 2 else 2

    # Read the real (label 1) text
    real_path = os.path.join(folder_path, f"file_{real_id}.txt")
    with open(real_path, "r", encoding="utf-8") as f:
        texts.append(f.read())
        labels.append(1)

    # Read the fake (label 0) text
    fake_path = os.path.join(folder_path, f"file_{fake_id}.txt")
    with open(fake_path, "r", encoding="utf-8") as f:
        texts.append(f.read())
        labels.append(0)


# --- Load test data (unlabeled) ---
test_base_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
test_texts = []
test_ids = []  # To track which text is which

for i in range(1068):  # 0 to 1067
    folder = f"article_{i:04d}"
    path = os.path.join(test_base_path, folder)

    for text_id in [1, 2]:
        full_path = os.path.join(path, f"file_{text_id}.txt")
        with open(full_path, "r", encoding="utf-8") as f:
            test_texts.append(f.read())
            test_ids.append(f"{folder}/{text_id}.txt")


model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("svm", SVC(kernel="sigmoid", probability=True, class_weight='balanced'))
])

# Split to evaluate performance on training set
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))


# --- Predict on test texts ---
test_predictions = model.predict(test_texts)


# Save predictions in CSV
output_df = pd.DataFrame({
    "file": test_ids,
    "predicted_label": test_predictions
})

output_df.to_csv("submissions.csv", index=False)

