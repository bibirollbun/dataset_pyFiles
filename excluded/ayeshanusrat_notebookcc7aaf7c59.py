import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Paths ===
DATA_DIR = "/kaggle/input/make-data-count-finding-data-references"
TRAIN_LABELS = f"{DATA_DIR}/train_labels.csv"
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"

# === Load Labels ===
labels = pd.read_csv(TRAIN_LABELS)

# === Parse XML ===
def parse_xml(xml_path):
    try:
        tree = etree.parse(xml_path)
        return ' '.join(tree.xpath('//text()'))
    except:
        return ""

# === Extract Text ===
def extract_text(row, base_dir):
    article_id = row['article_id']
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == f"{article_id}.xml":
                xml_path = os.path.join(root, file)
                text = parse_xml(xml_path)
                return text if isinstance(text, str) and text.strip() else ""
    return ""

# === Filter for Available XMLs ===
available_xmls = set()
for root, _, files in os.walk(TRAIN_DIR):
    for file in files:
        if file.endswith(".xml"):
            available_xmls.add(file.replace(".xml", ""))

print(f"Available XML files: {len(available_xmls)}")
labels['article_id'] = labels['article_id'].astype(str)
labels = labels[labels['article_id'].isin(available_xmls)]
print(f"Found {len(labels)} articles with matching XMLs.")

# === Extract Text from XMLs ===
labels['text'] = labels.apply(lambda row: extract_text(row, TRAIN_DIR), axis=1)
labels.dropna(subset=['text'], inplace=True)
labels = labels[labels['text'].astype(str).str.strip().astype(bool)]
print(f"Remaining after text extraction: {len(labels)}")

# === Label Encoding ===
le = LabelEncoder()
labels['label'] = le.fit_transform(labels['type'])

# === Train-Val Split ===
X_train, X_val, y_train, y_val = train_test_split(labels['text'], labels['label'], test_size=0.2, random_state=42)

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# === Model ===
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Training ===
model.fit(X_train_tfidf.toarray(), y_train, epochs=5, batch_size=32, validation_data=(X_val_tfidf.toarray(), y_val))

# === Evaluation ===
y_pred = model.predict(X_val_tfidf.toarray())
y_pred_labels = np.argmax(y_pred, axis=1)
print(classification_report(y_val, y_pred_labels, target_names=le.classes_))

# === Prepare Test Data ===
test_articles = []
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.endswith(".xml"):
            test_articles.append(os.path.splitext(file)[0])
test_df = pd.DataFrame({'article_id': test_articles})
test_df['text'] = test_df.apply(lambda row: extract_text(row, TEST_DIR), axis=1)
test_df = test_df[test_df['text'].astype(str).str.strip().astype(bool)]

# === TF-IDF for Test ===
test_tfidf = vectorizer.transform(test_df['text'])
test_preds = model.predict(test_tfidf.toarray())
test_labels = le.inverse_transform(np.argmax(test_preds, axis=1))



# === Test DataFrame ===
test_df = pd.DataFrame({'row_id': range(len(test_files)), 'article_id': [f.replace(".xml", "") for f in test_files]})
test_df['dataset_id'] = ''
test_df = generate_features(test_df)

X_test = test_df[features].astype(float)

# === Predict on test ===
y_test_pred = model.predict(X_test)
pred_labels = label_encoder.inverse_transform(np.argmax(y_test_pred, axis=1))

# === Final submission DataFrame ===
submission = pd.DataFrame({
    'row_id': test_df.index,
    'article_id': test_df['article_id'],
    'dataset_id': test_df['dataset_id'],  # Required, even if empty
    'type': pred_labels
})

# Ensure correct column order and types
submission = submission[['row_id', 'article_id', 'dataset_id', 'type']]
submission['row_id'] = submission['row_id'].astype(int)
submission['article_id'] = submission['article_id'].astype(str)
submission['dataset_id'] = submission['dataset_id'].astype(str)
submission['type'] = submission['type'].astype(str)

# === Save submission ===
submission.to_csv("/kaggle/working/submission.csv", index=False)

