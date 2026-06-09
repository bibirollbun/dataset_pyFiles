# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# Settings
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)
plt.rcParams['figure.figsize'] = (10,6)



# === CHANGE THIS to your dataset path(s) ===
DATA_PATH = "/kaggle/input/your-dataset-folder/"  # or "/kaggle/input/dataset/file.csv"
TRAIN_FILE = Path(DATA_PATH) / "train.csv"  # adjust filename
TEST_FILE  = Path(DATA_PATH) / "test.csv"   # optional

# Load (wrap in try)
try:
    df = pd.read_csv(TRAIN_FILE)
    print("Train shape:", df.shape)
except Exception as e:
    print("Could not load train.csv. Upload your file and set TRAIN_FILE correctly. Error:", e)
    df = None

# Quick peek
if df is not None:
    display(df.head())
    print(df.columns.tolist())



if df is not None:
    # Missing values
    print("Missing values per column:\n", df.isna().sum())

    # If target exists:
    TARGET = "category"  # <<--- set to your target column (e.g. "label" or "priority")
    if TARGET in df.columns:
        print("\nTarget distribution:")
        display(df[TARGET].value_counts(normalize=True).head(20))
    
    # Text insights (if text field exists)
    TEXT_COL = "text"  # <<--- set your main text column
    if TEXT_COL in df.columns:
        df["text_len"] = df[TEXT_COL].astype(str).map(len)
        display(df["text_len"].describe())
        display(df[[TEXT_COL, TARGET]].sample(5, random_state=1))



import re
def clean_text(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    # optionally remove URLs, emails, IDs:
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'\S+@\S+', ' ', s)
    return s

if df is not None:
    if "text" in df.columns:
        df["text_clean"] = df["text"].fillna("").map(clean_text)
    # Fill other missing columns (simple)
    df = df.fillna({"category": "unknown"}) if "category" in df.columns else df.fillna("")



if df is not None and TARGET in df.columns and "text_clean" in df.columns:
    X = df["text_clean"].values
    y = df[TARGET].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)),
        ("clf", LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, preds))
    print("\nClassification report:\n", classification_report(y_val, preds))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, preds)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix (validation)")
    plt.show()



if df is not None and "text_clean" in df.columns:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    param_grid = {
        "tfidf__ngram_range": [(1,1),(1,2)],
        "tfidf__min_df": [2,5],
        "clf__C": [0.1, 1.0, 10]
    }
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    best = gs.best_estimator_
    preds = best.predict(X_val)
    print("Val acc:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))



## 6) Interpretability: most important words for each class (for linear models)



def show_top_coefs(vectorizer, clf, classes, n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    for i, class_label in enumerate(classes):
        coefs = clf.coef_[i]
        top_pos = np.argsort(coefs)[-n:]
        top_neg = np.argsort(coefs)[:n]
        print("Class:", class_label)
        print(" Top +:", ", ".join(feature_names[top_pos][::-1]))
        print(" Top -:", ", ".join(feature_names[top_neg]))
        print()

if 'best' in globals() and isinstance(best.named_steps['clf'], LogisticRegression):
    vec = best.named_steps['tfidf']
    clf = best.named_steps['clf']
    classes = best.named_steps['clf'].classes_
    show_top_coefs(vec, clf, classes, n=10)



# Example: if you have a test.csv with an 'id' and 'text' column
try:
    test_df = pd.read_csv(TEST_FILE)
    test_df["text_clean"] = test_df["text"].fillna("").map(clean_text)
    test_preds = pipeline.predict(test_df["text_clean"])
    submission = pd.DataFrame({
        "id": test_df["id"],      # change if different
        "prediction": test_preds  # change column name per competition
    })
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
except Exception as e:
    print("Could not create submission. Check TEST_FILE, ID column and text column. Error:", e)





