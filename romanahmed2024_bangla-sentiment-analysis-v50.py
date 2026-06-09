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


train=pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")
sample_sub=pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/sample_submission.csv")


import pandas as pd
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define Bangla stopwords
stop_words = set(stopwords.words('bengali'))

# Step 1: Custom replace rules for company_x, See Translation, Payment Tk etc.
def custom_replace(text):
    # Replace specific payment text
    text = re.sub(r"Payment Tk.*?successful", "à¦†à¦®à¦¿ à¦†à¦œ à¦•à¦¿à¦›à§� à¦Ÿà¦¾à¦•à¦¾ à¦ªà¦°à¦¿à¦¶à§‹à¦§ à¦•à¦°à§‡à¦›à¦¿à¥¤", text)
    
    # Replace specific loan sentence
    text = re.sub(
        r"à¦–à§�à¦¬ à¦¦à¦°à¦•à¦¾à¦°à§‡à¦° à¦Ÿà¦¾à¦‡à¦®à§‡ company_x à¦¥à§‡à¦•à§‡ à¦¸à¦¹à¦œà§‡ à¦²à§‹à¦¨ à¦ªà§‡à¦²à¦¾à¦®\?See Translation",
        "à¦†à¦®à¦¿ à¦œà¦°à§�à¦°à¦¿ à¦¸à¦®à§Ÿà§‡ à¦¸à¦¹à¦œà§‡ à¦²à§‹à¦¨ à¦ªà§‡à§Ÿà§‡à¦›à¦¿à¦²à¦¾à¦®à¥¤",
        text
    )
    
    # Remove |See Translation or See Translation
    text = re.sub(r'\|?See Translation', '', text)

    # Replace company_x, company_y appropriately
    if 'company_x' in text or 'company_y' in text:
        if re.search(r'à¦²à§‹à¦¨|à¦¸à§‡à¦¬à¦¾|à¦¨à¦•|à¦¯à§‹à¦—à¦¾à¦¯à§‹à¦—|à¦•à¦²', text):
            text = re.sub(r'company_\w+', 'à¦“à¦‡ à¦ªà§�à¦°à¦¤à¦¿à¦·à§�à¦ à¦¾à¦¨', text)
        else:
            text = re.sub(r'company_\w+', 'à¦�à¦•à¦Ÿà¦¿ à¦•à§‹à¦®à§�à¦ªà¦¾à¦¨à¦¿', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Step 2: Main cleaning function
def full_bangla_text_cleaner(text, remove_english=True, remove_punctuation=True, normalize=True, fix_spacing=True):
    # Apply custom replaces first
    text = custom_replace(text)

    # Lowercase everything
    text = text.lower()

    # Remove English words/numbers
    if remove_english:
        text = re.sub(r'[a-zA-Z0-9_]+', '', text)

    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[à¥¤à¥¥!?.,;:â€œâ€�"\'â€˜â€™â€”â€¦()\[\]{}<>@#$%^&*_+=|\\/~`]', '', text)
        


    # Keep only Bangla characters
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)

    # Fix spacing issues with common errors
    if fix_spacing:
        text = re.sub(r'(\S+)\s à§‡', r'\1à§‡', text)
        text = re.sub(r'(\S+)\s à¦°à§‡', r'\1à¦°à§‡', text)

    # Basic spelling corrections
    common_fixes = {
        "à¦¸ à§�à¦–à§€à¦¨": "à¦¸à§�à¦–à§€à¦¨",
        "à¦¸à¦¿ à¦®à§�à¦ªà¦²": "à¦¸à¦¿à¦®à§�à¦ªà¦²",
        "à¦¨à§‡ à¦­à¦¿à¦—à§‡à¦Ÿ": "à¦¨à§‡à¦­à¦¿à¦—à§‡à¦Ÿ",
        "à¦¶ à§�à¦§à§�": "à¦¶à§�à¦§à§�",
        "à¦¸ à§�à¦›à¦¿": "à¦ªà¦¾à¦šà§�à¦›à¦¿",
        "à¦•à§‹à¦®à§�à¦ª à¦¾à¦¨à¦¿à¦°": "à¦•à§‹à¦®à§�à¦ªà¦¾à¦¨à¦¿à¦°",
        "à¦¨ à§‡à¦­à¦¿à¦—à§‡à¦Ÿ": "à¦¨à§‡à¦­à¦¿à¦—à§‡à¦Ÿ",
        "à§‡ à¦¨à¦•": "à¦� à¦¨à¦•",
        "à¦¬à¦¾à¦¸à§�à¦¤à¦¬ à§‡":"à¦¬à¦¾à¦¸à§�à¦¤à¦¬à§‡",
        "à¦¸à§�à¦¦ à§�à¦›à¦¿" : "à¦¸à§�à¦¦ à¦ªà¦¾à¦šà§�à¦›à¦¿",
        "à¦¸ à¦¿à¦®à§�à¦ªà¦²": "à¦¸à¦¿à¦®à§�à¦ªà¦²",
        "à¦ªà§�à¦°à§‹à¦¡à¦¾à¦•à§�à¦Ÿ à§‡à¦°":"à¦ªà§�à¦°à§‹à¦¡à¦¾à¦•à§�à¦Ÿà§‡à¦°"
    }
    for wrong, right in common_fixes.items():
        text = text.replace(wrong, right)

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Normalize letters
    if normalize:
        tokens = [re.sub(r'(à¦¥à§�)(à¦¯|à¦¯à¦¼)', 'à¦¥à§�à¦¯', word) for word in tokens]
        tokens = [re.sub(r'(à¦…à§�)(à¦¯|à¦¯à¦¼)', 'à¦…à§�à¦¯', word) for word in tokens]

    # Stopword filtering
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)
train["clean_text"] = train["text"].apply(full_bangla_text_cleaner)


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# âœ… Dataset columns: 'clean_text', 'sentiment', 'id'
X_text = train["clean_text"]
y = train["sentiment"]
ids = train["id"]

# âœ… Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# âœ… TF-IDF vectorization
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=3
)
X_tfidf = vectorizer.fit_transform(X_text)

# âœ… Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))

# âœ… Best hyperparameters (replace with real ones if available)
best_params = {
    "nb_alpha": 0.1,
    "rf_n_estimators": 150,
    "svc_c": 1.0
}

# âœ… Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_text), dtype=int)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_tfidf, y_encoded)):
    print(f"ğŸ”� Fold {fold+1}")
    X_train_fold, y_train_fold = X_tfidf[train_idx], y_encoded[train_idx]
    X_val_fold, y_val_fold = X_tfidf[val_idx], y_encoded[val_idx]

    # âœ… Stacking Classifier
    model = StackingClassifier(
        estimators=[
            ("nb", MultinomialNB(alpha=best_params["nb_alpha"])),
            ("rf", RandomForestClassifier(n_estimators=best_params["rf_n_estimators"], class_weight=class_weight_dict, random_state=42)),
            ("svc", SVC(C=best_params["svc_c"], probability=True, class_weight=class_weight_dict, random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000, class_weight=class_weight_dict),
        cv=5,
        n_jobs=-1
    )

    model.fit(X_train_fold, y_train_fold)
    preds = model.predict(X_val_fold)
    oof_preds[val_idx] = preds

# âœ… Decode predictions and true labels
true_labels = le.inverse_transform(y_encoded)
oof_labels = le.inverse_transform(oof_preds)

# âœ… Classification Report
print("\nğŸ“Š Cross-Validation Classification Report:")
print(classification_report(true_labels, oof_labels))



#Save submission file
submission = pd.DataFrame({
  "id": ids,
    "sentiment": oof_labels})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… submission.csv saved.")

