# âœ… Install PyMuPDF if not already installed
!pip install PyMuPDF

# âœ… Import Libraries
import fitz  # PyMuPDF for reading PDFs



# =============================================
# ğŸ“� Make Data Count - Dataset Reference Classification
# Author: Maha Kadhim
# Date: July 2025
# Competition: Make Data Count (Kaggle)
# =============================================
# âœ… 1. Import Libraries

import os
import glob
import fitz  # PyMuPDF for reading PDFs
import nltk
import pandas as pd
import re
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# âœ… 2. Download NLTK Resources (Commented if running without Internet)
# Uncomment these lines if running for the first time to download resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# âœ… 3. Define Preprocessing Functions

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Clean text: lowercase, remove punctuation, remove stopwords, lemmatize
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Label as Primary dataset reference based on key patterns
def label_primary(sentence):
    patterns = [
        r'(generated|produced|collected|uploaded|deposited|available at|created|measured|obtained)',
        r'(dataset.*available)',
        r'(data\\s(is|are)\\savailable)',
        r'(data\\s(set)?\\s(collected|generated|acquired|compiled))',
        r'(data\\s(set)?\\s(from|available at|deposited at))'
    ]
    for p in patterns:
        if re.search(p, sentence, re.I):
            return 'Primary'
    return None

# Label as Secondary dataset reference based on key patterns
def label_secondary(sentence):
    patterns = [
        r'(obtained from|reused from|downloaded from|previously published|sourced from|taken from|adapted from)',
        r'(data.*sourced from)',
        r'(dataset.*obtained from)',
        r'(data\\s(set)?\\s(reused|repurposed|reanalysed))',
        r'(as\\sreported\\s(in|by))'
    ]
    for p in patterns:
        if re.search(p, sentence, re.I):
            return 'Secondary'
    return None

# âœ… 4. Read Train PDFs and Create Training DataFrame

train_pdf_dir = '/kaggle/input/make-data-count-finding-data-references/train/PDF'
train_pdf_files = glob.glob(os.path.join(train_pdf_dir, '*.pdf'))

train_sentences = []

for pdf_file in train_pdf_files:
    doc = fitz.open(pdf_file)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    sentences = sent_tokenize(full_text)
    for s in sentences:
        s_clean = clean_text(s)
        if len(s_clean) < 5:
            continue
        primary_label = label_primary(s_clean)
        secondary_label = label_secondary(s_clean)
        if primary_label:
            train_sentences.append({'sentence': s_clean, 'label': 'Primary'})
        elif secondary_label:
            train_sentences.append({'sentence': s_clean, 'label': 'Secondary'})

train_df = pd.DataFrame(train_sentences)
print("âœ… Train samples:", train_df.shape)
print(train_df['label'].value_counts())

# âœ… 5. Split Data into Train and Validation Sets

X = train_df['sentence']
y = train_df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# âœ… 6. Build Pipeline with Gradient Boosting Classifier

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,4), analyzer='word')

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', GradientBoostingClassifier())
])

param_grid = {
    'tfidf__max_features': [10000, 15000],
    'tfidf__ngram_range': [(1,2),(1,3),(1,4)],
    'clf__n_estimators': [100, 300],
    'clf__learning_rate': [0.05, 0.1],
    'clf__max_depth': [3, 5]
}

# âœ… 7. RandomizedSearchCV for Hyperparameter Tuning

grid = RandomizedSearchCV(pipeline, param_grid, n_iter=10, cv=5, scoring='f1_macro', random_state=42, n_jobs=-1)
grid.fit(X_train, y_train)

print("ğŸ”� Best parameters:", grid.best_params_)
model = grid.best_estimator_

# âœ… 8. Evaluation on Validation Set

y_pred_val = model.predict(X_val)
print("ğŸ”� Validation Classification Report:")
print(classification_report(y_val, y_pred_val))

# âœ… 9. Read Test PDFs and Prepare Test Sentences

test_pdf_dir = '/kaggle/input/make-data-count-finding-data-references/test/PDF'
test_pdf_files = glob.glob(os.path.join(test_pdf_dir, '*.pdf'))

test_sentences = []

for pdf_file in test_pdf_files:
    doc = fitz.open(pdf_file)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    sentences = sent_tokenize(full_text)
    for s in sentences:
        s_clean = clean_text(s)
        if len(s_clean) > 5:
            test_sentences.append(s_clean)

print("âœ… Test sentences count:", len(test_sentences))

# âœ… 10. Predict on Test Data
y_pred_test = model.predict(test_sentences)

# âœ… 11. Create Submission File

submission = pd.DataFrame({
    'Id': range(len(y_pred_test)),
    'Prediction': y_pred_test
})

submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv created successfully")


# âœ… End of notebook with Markdown summary

print("Notebook prepared by Maha Kadhim - Make Data Count Submission July 2025")


