# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")


import numpy as np
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv").sample(10000)

# Custom text preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()  # Tokenization
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Apply text preprocessing
df["clean_text"] = df["question_text"].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])
y = df["target"].values

# BERT Embeddings
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
X_bert = bert_model.encode(df["clean_text"].tolist(), show_progress_bar=True)

# Standardize BERT features
scaler = StandardScaler()
X_bert_scaled = scaler.fit_transform(X_bert)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=100)
X_bert_pca = pca.fit_transform(X_bert_scaled)

# Split data
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
X_train_bert, X_test_bert, _, _ = train_test_split(X_bert_pca, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
X_train_bert_resampled, y_train_bert_resampled = smote.fit_resample(X_train_bert, y_train)

# Train Logistic Regression on TF-IDF
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf_resampled, y_train_resampled)

y_pred_tfidf = log_reg.predict(X_test_tfidf)
print("Logistic Regression (TF-IDF)\n", classification_report(y_test, y_pred_tfidf))

# Train Gradient Boosting Classifier on BERT
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
gbc.fit(X_train_bert_resampled, y_train_bert_resampled)

y_pred_bert = gbc.predict(X_test_bert)
print("Gradient Boosting (BERT)\n", classification_report(y_test, y_pred_bert))



import numpy as np
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Standardize BERT features
X_bert_scaled = scaler.fit_transform(X_bert)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=100)
X_bert_pca = pca.fit_transform(X_bert_scaled)

# Split data
X_train_bert, X_test_bert, _, _ = train_test_split(X_bert_pca, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_bert_resampled, y_train_bert_resampled = smote.fit_resample(X_train_bert, y_train)

# Train Gradient Boosting Classifier on BERT
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
gbc.fit(X_train_bert_resampled, y_train_bert_resampled)

y_pred_bert = gbc.predict(X_test_bert)
print("Gradient Boosting (BERT)\n", classification_report(y_test, y_pred_bert))





