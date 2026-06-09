import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


train_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train/'
test_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test/'
train_csv_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'


train_labels = pd.read_csv(train_csv_path)


texts = []
labels = []


for index, row in train_labels.iterrows():
    article_id = row['id']
    real_text_id = row['real_text_id']
    article_path = os.path.join(train_dir, f'article_{article_id:04d}')

    file1_path = os.path.join(article_path, 'file_1.txt')
    file2_path = os.path.join(article_path, 'file_2.txt')

    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            file1_text = f.read()
        with open(file2_path, 'r', encoding='utf-8') as f:
            file2_text = f.read()
    except FileNotFoundError:
        continue

    if real_text_id == 1:
        real_text = file1_text
        fake_text = file2_text
    else:
        real_text = file2_text
        fake_text = file1_text

    texts.extend([real_text, fake_text])
    labels.extend([1, 0])

print(f"Finished processing {len(texts)} training texts!")


vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_full = vectorizer.fit_transform(texts)
y_full = np.array(labels)
print("Text vectorization complete!")


X_train_vec, X_val_vec, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)
print(f"Training data shape: {X_train_vec.shape}")
print(f"Validation data shape: {X_val_vec.shape}")


svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_vec, y_train)


y_val_pred = svm_model.predict(X_val_vec)
y_val_proba = svm_model.predict_proba(X_val_vec)[:, 1]

print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))


fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


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

    vec1 = vectorizer.transform([file1_text])
    vec2 = vectorizer.transform([file2_text])

    prob1 = svm_model.predict_proba(vec1)[0][1]
    prob2 = svm_model.predict_proba(vec2)[0][1]

    if prob1 > prob2:
        predicted_real_text_id = 1
    elif prob2 > prob1:
        predicted_real_text_id = 2
    else:
        predicted_real_text_id = 1

    submission_rows.append({'id': article_id, 'real_text_id': predicted_real_text_id})


submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission.csv', index=False)


submission_df.head()

