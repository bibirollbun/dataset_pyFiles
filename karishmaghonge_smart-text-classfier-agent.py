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


# =============================================
# Smart Text Classifier Agent
# =============================================

# Step 1: Import libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import nltk


# Download stopwords if not already
nltk.download('stopwords')


# =============================================
# Step 2: Load Dataset
# =============================================
df = pd.read_csv('/kaggle/input/imdb-dataset/IMDB Dataset.csv')
print(df.head())


# Quick look at data
print("Dataset sample:")
print(df.head())

# Check for missing values
print("Missing values in dataset:", df.isnull().sum())

# Drop missing values if any
df = df.dropna()


# =============================================
# Step 3: Text Preprocessing
# =============================================
def clean_text(text):
    """
    Lowercase, remove punctuation, remove stopwords
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
df['cleaned_text'] = df['review'].apply(clean_text)

print("\nSample cleaned text:")
print(df[['review', 'cleaned_text']].head())


# =============================================
# Step 4: Feature Extraction
# Using TF-IDF Vectorizer
# =============================================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Save vectorizer
os.makedirs('models', exist_ok=True)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nTF-IDF feature extraction complete. Shape:", X.shape)


# =============================================
# Step 5: Split Dataset into Train/Test
# =============================================
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


# =============================================
# Step 6: Train Classifier
# =============================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# =============================================
# Step 7: Evaluate Model
# =============================================
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# =============================================
# Step 8: Save Model
# =============================================
with open('models/text_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved successfully in 'models/text_classifier.pkl'.")


# =============================================
# Step 9: Load Model and Vectorizer for Prediction
# =============================================
with open('models/text_classifier.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)


# =============================================
# Step 10: Prediction Function
# =============================================
def classify_text(text):
    """
    Input: raw text string
    Output: predicted label (sentiment/category)
    """
    cleaned = clean_text(text)
    features = loaded_vectorizer.transform([cleaned])
    prediction = loaded_model.predict(features)
    return prediction[0]


# =============================================
# Step 11: Test Prediction
# =============================================
sample_texts = [
    "The movie was fantastic and thrilling!",
    "I hated the film. It was a waste of time.",
    "An excellent experience with stunning visuals."
]

print("\nPrediction Results:")
for text in sample_texts:
    result = classify_text(text)
    print(f"Text: {text}\nPredicted Sentiment: {result}\n")

