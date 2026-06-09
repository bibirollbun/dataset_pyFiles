============================================================
FILE: requirements.txt
============================================================
numpy
pandas
scikit-learn
joblib
nltk
streamlit


============================================================
FILE: src/data.py
============================================================
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_csv(path="data/raw/imdb_reviews.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError("Place imdb_reviews.csv in data/raw/")
    df = pd.read_csv(path)
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment'")
    return df

def preprocess_df(df):
    df = df.copy()
    df["review_clean"] = df["review"].apply(basic_clean)
    return df

def split_and_save(df):
    os.makedirs("data/processed", exist_ok=True)
    X = df["review_clean"]
    y = df["sentiment"].apply(lambda x: 1 if str(x).lower() == "positive" else 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pd.DataFrame({"review": X_train}).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame({"review": X_test}).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame({"sentiment": y_train}).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame({"sentiment": y_test}).to_csv("data/processed/y_test.csv", index=False)
    print("Saved splits to data/processed/")

if __name__ == "__main__":
    df = load_csv()
    df = preprocess_df(df)
    split_and_save(df)


============================================================
FILE: src/features.py
============================================================
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

VECT_PATH = "models/tfidf_vectorizer.joblib"

def load_texts(path="data/processed/X_train.csv"):
    df = pd.read_csv(path)
    return df["review"].fillna("")

def build_vectorizer():
    texts = load_texts()
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    vect.fit(texts)
    os.makedirs("models", exist_ok=True)
    joblib.dump(vect, VECT_PATH)
    print("Saved TF-IDF vectorizer.")
    return vect

def load_vectorizer():
    if not os.path.exists(VECT_PATH):
        return build_vectorizer()
    return joblib.load(VECT_PATH)

if __name__ == "__main__":
    build_vectorizer()


============================================================
FILE: src/train.py
============================================================
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from features import load_vectorizer

MODEL_DIR = "models"

def load_processed():
    X_train = pd.read_csv("data/processed/X_train.csv")["review"]
    X_test = pd.read_csv("data/processed/X_test.csv")["review"]
    y_train = pd.read_csv("data/processed/y_train.csv")["sentiment"]
    y_test = pd.read_csv("data/processed/y_test.csv")["sentiment"]
    return X_train, X_test, y_train, y_test

def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed()
    vect = load_vectorizer()

    X_train_vec = vect.transform(X_train)
    X_test_vec = vect.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LinearSVM": LinearSVC(max_iter=20000, random_state=42),
    }

    best_name = None
    best_acc = 0
    best_model = None

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    joblib.dump(best_model, f"{MODEL_DIR}/{best_name}.joblib")
    joblib.dump(vect, f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    joblib.dump({"model_name": best_name}, f"{MODEL_DIR}/best_model.joblib")
    print(f"Best model saved as {best_name}")

if __name__ == "__main__":
    train_models()


============================================================
FILE: src/predict.py
============================================================
import os
import joblib
import pandas as pd

MODEL_DIR = "models"

def load_best():
    meta = joblib.load(f"{MODEL_DIR}/best_model.joblib")
    name = meta["model_name"]
    model = joblib.load(f"{MODEL_DIR}/{name}.joblib")
    vect = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    return model, vect

def predict_text(text):
    model, vect = load_best()
    X = vect.transform([text])
    p = int(model.predict(X)[0])
    return "positive" if p == 1 else "negative"

if __name__ == "__main__":
    print(predict_text("This movie was amazing!"))


============================================================
FILE: src/app_streamlit.py
============================================================
import streamlit as st
from predict import predict_text

st.set_page_config(page_title="Movie Review Sentiment")

st.title("ðŸŽ¬ Movie Review Sentiment Classifier")

text = st.text_area("Enter a movie review:")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter text.")
    else:
        label = predict_text(text)
        if label == "positive":
            st.success("Prediction: POSITIVE")
        else:
            st.error("Prediction: NEGATIVE")

