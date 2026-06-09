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


!pip install -q transformers==4.51
!pip install -q huggingface-hub==0.24.6
!pip install sentence-transformers>=2.7.0


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud,STOPWORDS
from bs4 import BeautifulSoup
import re,string,unicodedata

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost.sklearn import XGBClassifier



dataset= pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
num_duplicates = dataset.duplicated().sum() #identify duplicates
print('There are {} duplicate reviews present in the dataset'.format(num_duplicates))
dataset.head(5)


def data_loader(csv_file):
    dataset =pd.read_csv(csv_file)
    train, test= train_test_split(dataset, test_size=0.2, random_state=42)
    X_train,y_train =train["text"],train['label']
    X_test,y_test =test["text"],test["label"]
    tfidf_vect = TfidfVectorizer(stop_words='english') #tfidfVectorizer
    Xtrain_tfidf = tfidf_vect.fit_transform(X_train)
    Xtest_tfidf = tfidf_vect.transform(X_test)
    return Xtrain_tfidf,Xtest_tfidf,y_train,y_test,X_train,X_test,tfidf_vect



# =========================================
# MULTI-MODEL TEXT CLASSIFICATION PIPELINE
# =========================================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ==============================
# 1ï¸�âƒ£ LOAD AND PREPARE DATA
# ==============================
def data_loader(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_encoded'], test_size=0.2, random_state=42)
    tfidf_vect = TfidfVectorizer(stop_words='english', max_features=10000)
    Xtrain_tfidf = tfidf_vect.fit_transform(X_train)
    Xtest_tfidf = tfidf_vect.transform(X_test)
    return Xtrain_tfidf, Xtest_tfidf, y_train, y_test, X_train, X_test, tfidf_vect


# Load dataset
Xtrain_tfidf, Xtest_tfidf, y_train, y_test, X_train, X_test, tfidf_vect = data_loader("/kaggle/input/rmit-hackathon-2025/train.csv")


# ==============================
# 2ï¸�âƒ£ DEFINE MULTIPLE MODELS
# ==============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "NaiveBayes": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
}


# ==============================
# 3ï¸�âƒ£ TRAIN AND EVALUATE MODELS
# ==============================
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nğŸš€ Training {name} ...")
    model.fit(Xtrain_tfidf, y_train)
    y_pred = model.predict(Xtest_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {"accuracy": acc, "f1": f1}
    predictions[name] = y_pred
    print(f"âœ… {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
    print(classification_report(y_test, y_pred))
    # Save checkpoint
    joblib.dump(model, f"{name}_model.pkl")

print("\nğŸ“Š Summary of Models:")
for name, metrics in results.items():
    print(f"{name:20s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")


# ==============================
# 4ï¸�âƒ£ ENSEMBLE (SOFT VOTING)
# ==============================
# Re-load only compatible models (SVC cannot do predict_proba)
voting_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('nb', MultinomialNB()),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    ))
]

ensemble = VotingClassifier(estimators=voting_models, voting='soft')
print("\nğŸ¤� Training Ensemble (Soft Voting)...")
ensemble.fit(Xtrain_tfidf, y_train)
y_pred_ens = ensemble.predict(Xtest_tfidf)
ens_acc = accuracy_score(y_test, y_pred_ens)
ens_f1 = f1_score(y_test, y_pred_ens, average='weighted')

print(f"ğŸ�† Ensemble Results: Accuracy={ens_acc:.4f}, F1={ens_f1:.4f}")
print(classification_report(y_test, y_pred_ens))
joblib.dump(ensemble, "ensemble_soft_voting.pkl")


# ==============================
# 5ï¸�âƒ£ SAVE TF-IDF VECTOR
# ==============================
joblib.dump(tfidf_vect, "tfidf_vectorizer.pkl")

print("\nğŸ’¾ All models and vectorizer saved successfully!")



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 1ï¸�âƒ£ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(list(set(y_test)))  # get unique labels

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2ï¸�âƒ£ Convert classification report to DataFrame for heatmap visualization
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(6, 3))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='YlGnBu')
plt.title('Classification Report Heatmap')
plt.show()

# 3ï¸�âƒ£ Optional: Bar chart of accuracy
acc = accuracy_score(y_test, y_pred)
plt.figure(figsize=(4, 3))
plt.bar(['Accuracy'], [acc], color='green')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.text(0, acc/2, f"{acc:.2f}", ha='center', color='white', fontsize=14)
plt.show()



!pip install -q virtualenv
!virtualenv hf_env
!source hf_env/bin/activate
# Completely clean and install safe versions
!pip install -U pip
!pip uninstall -y transformers huggingface-hub
!pip install -q "transformers==4.39.3" "huggingface-hub==0.25.2" "torch>=2.0" "numpy>=1.26" "google-generativeai>=0.6.1"



!pip install torch transformers google-generativeai sentence-transformers


import os, google.generativeai as genai
os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Example dataset
data = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")  # must have 'text' and 'label'
data.dropna(subset=["text", "label"], inplace=True)

le = LabelEncoder()
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
train_df["label"] = le.fit_transform(train_df["label"])
test_df["label"] = le.transform(test_df["label"])



# ============================================================
# GEMINI + ENSEMBLE TEXT CLASSIFICATION PIPELINE
# ============================================================

import google.generativeai as genai
import numpy as np
import pandas as pd
import time, joblib, xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB   # works with float embeddings

# ============================================================
# 1ï¸�âƒ£  CONFIGURE GEMINI API
# ============================================================
genai.configure(api_key="")  # <- insert your API key

# ---- Embedding Helper ----
def get_gemini_embeddings(texts, model="models/text-embedding-004", sleep_time=0.4):
    """
    Convert texts to embeddings using Gemini embedding model.
    """
    all_embeddings = []
    for i, text in enumerate(texts):
        try:
            print(f"ğŸ”¹ Embedding {i+1}/{len(texts)} ...")
            result = genai.embed_content(model=model, content=text)
            all_embeddings.append(result["embedding"])
        except Exception as e:
            print(f"âš ï¸� Error at {i}: {e}")
            all_embeddings.append([0]*768)
        time.sleep(sleep_time)  # avoid API rate-limit
    return np.array(all_embeddings)


# ============================================================
# 2ï¸�âƒ£  LOAD AND PREPROCESS DATA
# ============================================================
print("ğŸ“‚ Loading dataset ...")
data = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
data.dropna(subset=["text", "label"], inplace=True)

# Label-encode if labels are strings
if data["label"].dtype == object:
    print("ğŸ”  Encoding string labels ...")
    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])
    joblib.dump(le, "label_encoder.pkl")
    print("âœ… Encoded labels:", dict(zip(le.classes_, le.transform(le.classes_))))
else:
    print("âœ… Labels are already numeric.")

# Split
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["label"])
y_train, y_test = train_df["label"], test_df["label"]


# ============================================================
# 3ï¸�âƒ£  GENERATE GEMINI EMBEDDINGS
# ============================================================
print("ğŸ”¹ Generating embeddings for training data ...")
X_train = get_gemini_embeddings(train_df["text"].tolist())

print("ğŸ”¹ Generating embeddings for test data ...")
X_test = get_gemini_embeddings(test_df["text"].tolist())

print(f"âœ… Embeddings generated: train={X_train.shape}, test={X_test.shape}")


# ============================================================
# 4ï¸�âƒ£  DEFINE BASE MODELS
# ============================================================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GaussianNB": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    ),
}

# ============================================================
# 5ï¸�âƒ£  TRAIN & EVALUATE EACH MODEL
# ============================================================
results, predictions = {}, {}

for name, model in models.items():
    print(f"\nğŸš€ Training {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results[name] = {"accuracy": acc, "f1": f1}
    predictions[name] = y_pred

    print(f"âœ… {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"{name}_gemini_model.pkl")

print("\nğŸ“Š Summary of Base Models:")
for name, metrics in results.items():
    print(f"{name:20s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")


# ============================================================
# 6ï¸�âƒ£  BUILD ENSEMBLE (SOFT VOTING)
# ============================================================
voting_models = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("nb", GaussianNB()),
    ("xgb", xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )),
]

ensemble = VotingClassifier(estimators=voting_models, voting="soft")
print("\nğŸ¤� Training Ensemble (Soft Voting) ...")
ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)

ens_acc = accuracy_score(y_test, y_pred_ens)
ens_f1 = f1_score(y_test, y_pred_ens, average="weighted")
print(f"ğŸ�† Ensemble Results: Accuracy={ens_acc:.4f}, F1={ens_f1:.4f}")
print(classification_report(y_test, y_pred_ens))

joblib.dump(ensemble, "ensemble_gemini_soft_voting.pkl")

print("\nğŸ’¾ All models and label encoder saved successfully!")






test_data = "/kaggle/input/rmit-hackathon-2025/test.csv"
test_df = pd.read_csv(test_data)
X_new = test_df['text']

# 3ï¸�âƒ£ Transform the text using loaded vectorizer
X_new_tfidf = tfidf_vect.transform(X_new)

# 4ï¸�âƒ£ Predict using loaded model
y_pred_prob = ensemble.predict_proba(X_new_tfidf)[:, 1]
submission = pd.DataFrame({
    "Id":test_df.index,
    "target": y_pred_prob.round(4)
})
print(submission)
submission.to_csv("/kaggle/working/submission.csv",index=False)

